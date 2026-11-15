// Two ops of the DFlash 2 draft pass, each in place of a chain of ggml-cuda kernels (#1181).
// At 16 sequences a pass drafts 128 rows and its head writes 112 rows of 262,144 logits.
//
// - topk_part_kernel + topk_join_kernel: each row's k (at most 32) largest logits. ggml-cuda
//   takes its radix top-k for many rows: four histogram passes and a gather, every one reading
//   the whole row (~2.5 ms for 112 rows on a GB10). Here every row is read once: each CTA keeps
//   the best k of its part of the row (a sorted list per warp, merged in shared memory) and one
//   CTA per row joins the parts. Keys are ggml's (top_k_float_to_ordered: the float's bits made
//   to sort as unsigned integers), so the set is ggml's; among equal keys at the k-th place,
//   where ggml's gather takes whichever its atomics reach first, the lower column wins. The
//   output is sorted, best first (ggml's is in no order; the selector sorts its candidates).
// - conv_kernel: one side of DFlash 2's grouped dynamic causal convolution (v2.rs `conv`), in
//   place of ggml's pad, three broadcast products and sums a tap and the sum over the taps:
//   term = x * dynamic + x * base per tap, out = term_0 + term_1 + ..., every product and sum
//   rounded on its own as ggml's separate kernels round them (__fmul_rn / __fadd_rn: no
//   contraction into an FMA), so the bits are ggml's.
//
// The Rust side (src/spec/dflash/fused.rs) launches the kernels with cudaLaunchKernel. Plain
// SIMT code for every target.

#include <cstdint>

extern "C" {

// Top-k of `rows` rows of `n` floats (row r at x + r * x_nb1 bytes) into dst (row r's k
// indexes at dst + r * dst_nb1 bytes). `part` holds `parts` lists of k packed candidates a
// row; part p covers columns [p * span, min(n, (p + 1) * span)). n and span are multiples of 4.
struct legion_draft_topk {
    const float *        x;
    int64_t              x_nb1;
    int32_t *            dst;
    int64_t              dst_nb1;
    unsigned long long * part;
    int32_t              n, k, rows, parts, span;
};

// One side of the convolution: dst[i, t] = sum over taps of x[i, t - tap] (0 where t - tap is
// before position t's block) times dynamic[t][(side * k + tap) * groups + i / group] and times
// base[(side * k + tap) * e + i], products summed per tap first. Rows of x and dst are positions
// (strides in bytes); dynamic is the kernel projection's output, `2 * k * groups` a position.
struct legion_draft_conv {
    const float * x;
    int64_t       x_nb1;
    const float * dynamic;
    int64_t       dynamic_nb1;
    const float * base;
    float *       dst;
    int64_t       dst_nb1;
    int32_t       e, n, group, groups, k, taps, side, block;
};

}  // extern "C"

namespace {

constexpr int TOPK_THREADS = 256;
constexpr int TOPK_WARPS   = TOPK_THREADS / 32;

typedef unsigned long long u64;

// ggml-cuda's top_k_float_to_ordered (top-k.cu).
__device__ __forceinline__ uint32_t ordered(float v) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// A candidate as one integer: the key above, then the column inverted (lower columns larger).
// Never 0 (the inverted column of a row of at most 2^31 has its top bit set): 0 is "empty".
__device__ __forceinline__ u64 pack(float v, int col) {
    return ((u64) ordered(v) << 32) | (u64) (~(uint32_t) col);
}

// Inserts v (the same in every lane, above the list's last) into the warp's list: lane l < k
// holds the l-th largest so far, lanes past k hold 0.
__device__ __forceinline__ void insert(u64 & mine, u64 v, int k, int lane) {
    const int pos = __popc(__ballot_sync(0xffffffffu, lane < k && mine > v));
    const u64 up  = __shfl_up_sync(0xffffffffu, mine, 1);
    if (lane == pos) {
        mine = v;
    } else if (lane > pos && lane < k) {
        mine = up;
    }
}

// Sorts the m (at most blockDim.x) candidates in s by rank (descending; equal ones, only the
// empty 0s, by position) and returns this thread's rank, or m when it holds none.
__device__ __forceinline__ int rank_of(const u64 * s, int m, int t) {
    if (t >= m) {
        return m;
    }
    const u64 v = s[t];
    int       r = 0;
    for (int j = 0; j < m; j++) {
        const u64 w = s[j];
        r += (w > v) | ((w == v) & (j < t));
    }
    return r;
}

// Grid (parts, rows), TOPK_THREADS threads: the best k of part blockIdx.x of row blockIdx.y.
__global__ void __launch_bounds__(TOPK_THREADS) topk_part_kernel(const legion_draft_topk p) {
    __shared__ u64 lists[TOPK_WARPS * 32];
    const int      lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int      k = p.k;
    const int      c0 = blockIdx.x * p.span;
    const int      c1 = min(p.n, c0 + p.span);
    const float *  row = (const float *) ((const char *) p.x + (int64_t) blockIdx.y * p.x_nb1);
    u64            mine = 0, thr = 0;
    // Each warp takes 128 columns at a time, 4 a lane, every warp of the CTA a stretch of 1,024.
    for (int base = c0 + warp * 128; base < c1; base += TOPK_THREADS * 4) {
        const int c = base + lane * 4;
        u64       cand[4] = { 0, 0, 0, 0 };
        if (c < c1) {
            const float4 v = *(const float4 *) (row + c);
            cand[0]        = pack(v.x, c);
            cand[1]        = pack(v.y, c + 1);
            cand[2]        = pack(v.z, c + 2);
            cand[3]        = pack(v.w, c + 3);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned b = __ballot_sync(0xffffffffu, cand[j] > thr);
            while (b) {
                const int src = __ffs(b) - 1;
                b &= b - 1;
                const u64 v = __shfl_sync(0xffffffffu, cand[j], src);
                if (v > thr) {
                    insert(mine, v, k, lane);
                    thr = __shfl_sync(0xffffffffu, mine, k - 1);
                }
            }
        }
    }
    if (lane < k) {
        lists[warp * k + lane] = mine;
    }
    __syncthreads();
    const int m = TOPK_WARPS * k;
    const int r = rank_of(lists, m, threadIdx.x);
    if (r < k) {
        p.part[((int64_t) blockIdx.y * p.parts + blockIdx.x) * k + r] = lists[threadIdx.x];
    }
}

// Grid rows, a multiple of 32 threads at least parts * k (at most 1,024): row blockIdx.x's k
// best of its parts' lists, best first.
__global__ void __launch_bounds__(1024) topk_join_kernel(const legion_draft_topk p) {
    __shared__ u64 cand[1024];
    const int      m = p.parts * p.k;
    const u64 *    src = p.part + (int64_t) blockIdx.x * m;
    for (int t = threadIdx.x; t < m; t += blockDim.x) {
        cand[t] = src[t];
    }
    __syncthreads();
    const int r = rank_of(cand, m, threadIdx.x);
    if (r < p.k) {
        int32_t * out = (int32_t *) ((char *) p.dst + (int64_t) blockIdx.x * p.dst_nb1);
        out[r]        = (int32_t) ~(uint32_t) cand[threadIdx.x];
    }
}

// Grid (ceil(e / 256), n), 256 threads: one value of dst each.
__global__ void __launch_bounds__(256) conv_kernel(const legion_draft_conv p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= p.e) {
        return;
    }
    const int     g   = i / p.group;
    const int     j   = t % p.block;
    const float * dyn = (const float *) ((const char *) p.dynamic + (int64_t) t * p.dynamic_nb1);
    float         out = 0.0f;
    for (int tap = 0; tap < p.taps; tap++) {
        // ggml's padding is +0.0, and 0 times the coefficients still goes through the sums.
        const float v =
            j >= tap ? ((const float *) ((const char *) p.x + (int64_t) (t - tap) * p.x_nb1))[i] : 0.0f;
        const int   at   = p.side * p.k + tap;
        const float d    = dyn[at * p.groups + g];
        const float b    = p.base[(int64_t) at * p.e + i];
        const float term = __fadd_rn(__fmul_rn(v, d), __fmul_rn(v, b));
        out              = tap == 0 ? term : __fadd_rn(out, term);
    }
    ((float *) ((char *) p.dst + (int64_t) t * p.dst_nb1))[i] = out;
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel:
// top-k part, top-k join, convolution.
extern "C" const void * const legion_dflash_draft_kernels[3] = {
    reinterpret_cast<const void *>(&topk_part_kernel),
    reinterpret_cast<const void *>(&topk_join_kernel),
    reinterpret_cast<const void *>(&conv_kernel),
};
