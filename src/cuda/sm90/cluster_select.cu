// Small-query selections on thread-block clusters (sm_90+: GB10, Hopper, RTX 50), for decode
// steps where a one-CTA-per-row kernel leaves all but a few SMs idle (#554). Two kernels, each a
// cluster of CL CTAs per row exchanging partial results through distributed shared memory:
//
// - topk_cluster_kernel: qsa_attn.cu's topk_kernel (the QSA indexer's top-k) with the row
//   split over the cluster. Each pass every CTA histograms its slice and adds each count into
//   the CTA that owns the bin (CTA r owns bins [r * BINS / CL, (r + 1) * BINS / CL)); the owners
//   share their totals, the owner of the digit finds it among its bins by topk_kernel's search
//   and sends it to every CTA. Only writes cross CTAs (a remote shared-memory read costs about
//   as much as a global one on a GB10). The emit offsets each CTA's ranks by the counts of the
//   CTAs before it. The k-th largest key, the count taken at it and every output position are
//   functions of integer counts only, so the output is topk_kernel's, index for index.
// - argmax_cluster_kernel: ggml-cuda's argmax_f32 (argmax.cu) over a row of at least 1,024
//   floats, with its exact pick. argmax_f32 runs 1,024 threads; thread t takes columns t, t +
//   1024, ... keeping the first strictly larger value (values at or below -FLT_MAX and NaN are
//   never taken), then two xor-butterflies (in each warp, then over the warps' results) keep
//   the strictly larger value and otherwise their own. So among equal maxima the pick is the
//   lowest (bitrev5(warp), bitrev5(lane), column / 1024): a total order, which any reduction
//   order reproduces. Each CTA reduces its slice; CTA r stores its best in CTA 0's shared
//   memory; CTA 0 writes the row's pick (-1 when nothing beats -FLT_MAX, as argmax_f32).
//
// Design after Basalt (github.com/jesdga95/basalt, MIT; qsa_select.cu's
// block_topk_cluster_kernel and sampler.cu's sampler_greedy_cluster_kernel; some of its kernels
// are adapted from Infernix, Apache-2.0). The radix passes, digit search and tie rule are
// topk_kernel's (from Strata-AC922's qsa_select.cu, MIT), so the output is that kernel's.
//
// The Rust side (src/model/cluster_select.rs) launches the kernels itself with
// cudaLaunchKernelExC and a cluster dimension of CL, and only on devices of compute capability
// 9.0+ whose binary holds code for them; older targets compile a trap. The argmax also has a
// cluster-free form for every target (argmax_split_kernel + argmax_join_kernel, #814): the row
// split over up to 64 CTAs, each CTA's best through global memory, joined by one warp a row.

#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#include <cooperative_groups.h>
#define LEGION_CLUSTER 1
#else
#define LEGION_CLUSTER 0
#endif

extern "C" {

// As legion_qsa_topk (qsa_attn.cu).
struct legion_topk_cluster {
    const float * score;  // [n][rows]: row r at score + r * score_nb1 (bytes)
    int64_t score_nb1;
    int32_t * dst;        // [k][rows]
    int64_t dst_nb1;
    int32_t n, k, rows;
};

struct legion_argmax_cluster {
    const float * x;  // [n][rows]: row r at x + r * x_nb1 (bytes)
    int64_t x_nb1;
    int32_t * dst;    // [rows]
    int32_t n, rows;  // n >= 1024
};

// argmax_split_kernel + argmax_join_kernel (no cluster): part holds rows * parts (value bits,
// rank) pairs between the two launches.
struct legion_argmax_split {
    const float * x;  // [n][rows]: row r at x + r * x_nb1 (bytes)
    int64_t x_nb1;
    int32_t * dst;    // [rows]
    uint2 * part;     // [parts][rows]
    int32_t n, rows, parts;
};

}  // extern "C"

namespace {

constexpr int CL = 8;  // CTAs per row: the portable cluster size

#if LEGION_CLUSTER
namespace cg = cooperative_groups;
#endif

// ---- top-k (see the file comment). The passes, scan and digit search are topk_kernel's.

constexpr int TKC_T = 512;
constexpr int TKC_BINS = 2048;
constexpr int TKC_SLICE = TKC_BINS / CL;  // bins each CTA totals

__device__ __forceinline__ uint32_t tkc_key(float v) {
    // ggml's top_k_float_to_ordered.
    const uint32_t bits = __float_as_uint(v);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// Exclusive prefix sum over the CTA's TKC_T threads and the total; sw: TKC_T / 32 ints.
__device__ __forceinline__ int tkc_scan(int x, int * sw, int * total) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int v = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffff, v, o);
        v += lane >= o ? y : 0;
    }
    if (lane == 31) {
        sw[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        int t = lane < TKC_T / 32 ? sw[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, t, o);
            t += lane >= o ? y : 0;
        }
        if (lane < TKC_T / 32) {
            sw[lane] = t;
        }
    }
    __syncthreads();
    const int r = (warp > 0 ? sw[warp - 1] : 0) + v - x;
    *total = sw[TKC_T / 32 - 1];
    __syncthreads();
    return r;
}

// Grid CL x rows (cluster CL), TKC_T threads. CTA r of a row takes entries [r * slice, (r + 1) *
// slice), thread t of it `per` consecutive ones from r * slice + t * per, in registers (PER >=
// per) or read again on every pass (PER == 0).
template <int PER>
__global__ void __launch_bounds__(TKC_T) topk_cluster_kernel(legion_topk_cluster p) {
#if LEGION_CLUSTER
    __shared__ int hist[TKC_BINS];  // this CTA's counts
    __shared__ int tot[TKC_SLICE];  // the cluster's counts of the bins this CTA owns
    __shared__ int ssum[CL];        // the cluster's count of each owner's bins
    __shared__ int osum[CL];        // this CTA's count of each owner's bins
    __shared__ int sw[TKC_T / 32];
    __shared__ int cnt[CL][2];      // each CTA's entries above the threshold and at it
    __shared__ uint32_t s_digit;
    __shared__ int s_need;

    const cg::cluster_group cl = cg::this_cluster();
    const int rank = (int) cl.block_rank();
    const int row = blockIdx.x / CL, t = threadIdx.x, lane = t % 32;
    // Every CTA's tot and ssum zero before any CTA adds into them (the wait is before the first
    // push).
    for (int b = t; b < TKC_SLICE; b += TKC_T) {
        tot[b] = 0;
    }
    if (t < CL) {
        ssum[t] = 0;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    const float * src = (const float *) ((const char *) p.score + row * p.score_nb1);
    const int slice = (p.n + CL - 1) / CL;
    const int lo = min(p.n, rank * slice), hi = min(p.n, lo + slice);
    const int per = (slice + TKC_T - 1) / TKC_T;  // the same in every CTA of the cluster
    const int i0 = lo + t * per, i1 = min(hi, i0 + per);

    uint32_t reg[PER > 0 ? PER : 1];
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < PER; j++) {
            reg[j] = i0 + j < i1 ? tkc_key(src[i0 + j]) : 0;
        }
    }
    auto key = [&](int j) -> uint32_t {
        if constexpr (PER > 0) {
            return reg[j];
        } else {
            return tkc_key(src[i0 + j]);
        }
    };
    constexpr int UNROLL = PER > 0 ? PER : 1;

    uint32_t prefix = 0, pmask = 0;
    int need = p.k;
    const int shifts[3] = {21, 10, 0};
    const int widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; pass++) {
        const int shift = shifts[pass];
        const uint32_t nb = 1u << widths[pass];
        for (int i = t; i < TKC_BINS; i += TKC_T) {
            hist[i] = 0;
        }
        if (t < CL) {
            osum[t] = 0;
        }
        __syncthreads();
        auto count = [&](int j) {
            const uint32_t k = key(j);
            const bool in = i0 + j < i1 && (k & pmask) == prefix;
            const unsigned act = __ballot_sync(0xffffffff, in);
            if (in) {
                const int bin = (int) ((k >> shift) & (nb - 1));
                const unsigned peers = __match_any_sync(act, bin);
                if (lane == __ffs(peers) - 1) {
                    atomicAdd(&hist[bin], __popc(peers));
                }
            }
        };
        if constexpr (PER > 0) {
#pragma unroll
            for (int j = 0; j < UNROLL; j++) {
                if (j < per) {
                    count(j);
                }
            }
        } else {
            for (int j = 0; j < per; j++) {
                count(j);
            }
        }
        __syncthreads();
        if (pass == 0) {
            asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
        }
        // Each bin's count to the CTA that owns it, and the counts of each owner's bins to every
        // CTA (remote shared-memory reads are slow; these adds are not waited for until the
        // barrier). A warp's bins all have one owner.
        for (int b = t; b < TKC_BINS; b += TKC_T) {
            const int c = hist[b];
            if (c != 0) {
                atomicAdd(cl.map_shared_rank(&tot[b % TKC_SLICE], b / TKC_SLICE), c);
            }
            const int s = __reduce_add_sync(0xffffffff, c);
            if (lane == 0 && s != 0) {
                atomicAdd(&osum[b / TKC_SLICE], s);
            }
        }
        __syncthreads();
        if (t < CL * CL && osum[t % CL] != 0) {
            atomicAdd(cl.map_shared_rank(&ssum[t % CL], t / CL), osum[t % CL]);
        }
        cl.sync();  // every owner's tot and every CTA's ssum complete
        // topk_kernel's search on the cluster's counts: the largest digit d whose bins d..
        // hold at least `need` entries. First the owner of d (owners hold ascending runs of
        // bins), then d among its bins.
        int o = CL - 1, above = 0;
        while (o > 0 && above + ssum[o] < need) {
            above += ssum[o];
            o--;
        }
        if (rank == o) {
            const int nd = need - above;
            const int mine = t < TKC_SLICE ? tot[TKC_SLICE - 1 - t] : 0;
            int unused;
            const int before = tkc_scan(mine, sw, &unused);
            if (before < nd && before + mine >= nd) {
                const uint32_t d = (uint32_t) (rank * TKC_SLICE + TKC_SLICE - 1 - t);
                for (int r = 0; r < CL; r++) {
                    *cl.map_shared_rank(&s_digit, r) = d;
                    *cl.map_shared_rank(&s_need, r) = nd - before;
                }
            }
        }
        // This CTA's reads of tot and ssum done: clear them for the next pass, whose adds come
        // after the next cluster barrier.
        __syncthreads();
        for (int b = t; b < TKC_SLICE; b += TKC_T) {
            tot[b] = 0;
        }
        if (t < CL) {
            ssum[t] = 0;
        }
        cl.sync();  // the digit everywhere
        prefix |= s_digit << shift;
        pmask |= (nb - 1) << shift;
        need = s_need;
    }
    // prefix is the k-th largest key of the row; take every entry above it and the first `need`
    // at it, ranked over the whole row.
    int above = 0, eq = 0;
    auto tally = [&](int j) {
        const uint32_t k = key(j);
        above += k > prefix;
        eq += k == prefix;
    };
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            if (i0 + j < i1) {
                tally(j);
            }
        }
    } else {
        for (int j = i0; j < i1; j++) {
            tally(j - i0);
        }
    }
    int n_above, n_eq;
    int a = tkc_scan(above, sw, &n_above);
    int e = tkc_scan(eq, sw, &n_eq);
    if (t < CL) {
        int * c = &cl.map_shared_rank(cnt, t)[rank][0];
        c[0] = n_above;
        c[1] = n_eq;
    }
    // Every CTA's counts everywhere; no CTA touches another's shared memory after this.
    cl.sync();
    int above_all = 0;
    for (int r = 0; r < CL; r++) {
        above_all += cnt[r][0];
        if (r < rank) {
            a += cnt[r][0];
            e += cnt[r][1];
        }
    }
    int32_t * dst = (int32_t *) ((char *) p.dst + row * p.dst_nb1);
    auto emit = [&](int j) {
        const uint32_t k = key(j);
        if (k > prefix) {
            dst[a++] = i0 + j;
        } else if (k == prefix) {
            if (e < need) {
                dst[above_all + e] = i0 + j;
            }
            e++;
        }
    };
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            if (i0 + j < i1) {
                emit(j);
            }
        }
    } else {
        for (int j = i0; j < i1; j++) {
            emit(j - i0);
        }
    }
#else
    (void) p;
    __trap();
#endif
}

// ---- argmax (see the file comment).

constexpr int AM_T = 512;

// A candidate: its value and its rank in argmax_f32's tie order (lower wins); NONE: no
// candidate (argmax_f32's -1).
constexpr uint32_t AM_NONE = 0xffffffffu;

__device__ __forceinline__ uint32_t am_rank(int col) {
    const uint32_t t = (uint32_t) col & 1023u;
    const uint32_t warp = __brev(t >> 5) >> 27, lane = __brev(t & 31u) >> 27;
    return (warp << 26) | (lane << 21) | ((uint32_t) col >> 10);
}

__device__ __forceinline__ int am_col(uint32_t r) {
    const uint32_t warp = __brev(r >> 26) >> 27, lane = __brev((r >> 21) & 31u) >> 27;
    return (int) (((r & 0x1fffffu) << 10) | (warp << 5) | lane);
}

// Whether (v, r) beats (bv, br): larger, or equal and earlier in the tie order. Only values
// above -FLT_MAX are ever candidates, so no NaN compares here.
__device__ __forceinline__ bool am_better(float v, uint32_t r, float bv, uint32_t br) {
    return v > bv || (v == bv && r < br);
}

__device__ __forceinline__ void am_take(float v, int col, float & bv, uint32_t & br) {
    if (v > -FLT_MAX) {
        const uint32_t r = am_rank(col);
        if (am_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
    }
}

// Grid CL x rows (cluster CL), AM_T threads. CTA r of a row takes columns [r * slice, (r + 1) *
// slice), slice a multiple of 4 (float4 loads when the row allows them).
__global__ void __launch_bounds__(AM_T) argmax_cluster_kernel(legion_argmax_cluster p) {
#if LEGION_CLUSTER
    __shared__ float s_v[AM_T / 32];
    __shared__ uint32_t s_r[AM_T / 32];
    __shared__ float c_v[CL];  // CTA 0: each CTA's best
    __shared__ uint32_t c_r[CL];

    const cg::cluster_group cl = cg::this_cluster();
    const int rank = (int) cl.block_rank();
    const int row = blockIdx.x / CL, t = threadIdx.x, lane = t % 32, warp = t / 32;
    // Every CTA running (CTA 0's shared memory exists) before the stores into it below.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    const float * x = (const float *) ((const char *) p.x + row * p.x_nb1);
    const int slice = ((p.n + CL - 1) / CL + 3) & ~3;
    const int lo = min(p.n, rank * slice), hi = min(p.n, lo + slice);

    float bv = -FLT_MAX;
    uint32_t br = AM_NONE;
    if (((uintptr_t) x & 15) == 0) {
        const int hi4 = lo + ((hi - lo) & ~3);
#pragma unroll 4
        for (int c = lo + 4 * t; c < hi4; c += 4 * AM_T) {
            const float4 v = __ldg((const float4 *) (x + c));
            am_take(v.x, c, bv, br);
            am_take(v.y, c + 1, bv, br);
            am_take(v.z, c + 2, bv, br);
            am_take(v.w, c + 3, bv, br);
        }
        for (int c = hi4 + t; c < hi; c += AM_T) {
            am_take(__ldg(x + c), c, bv, br);
        }
    } else {
        for (int c = lo + t; c < hi; c += AM_T) {
            am_take(__ldg(x + c), c, bv, br);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, bv, o);
        const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
        if (am_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
    }
    if (lane == 0) {
        s_v[warp] = bv;
        s_r[warp] = br;
    }
    __syncthreads();
    if (warp == 0) {
        bv = lane < AM_T / 32 ? s_v[lane] : -FLT_MAX;
        br = lane < AM_T / 32 ? s_r[lane] : AM_NONE;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v = __shfl_xor_sync(0xffffffff, bv, o);
            const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
            if (am_better(v, r, bv, br)) {
                bv = v;
                br = r;
            }
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (t == 0) {
        cl.map_shared_rank(c_v, 0)[rank] = bv;
        cl.map_shared_rank(c_r, 0)[rank] = br;
    }
    cl.sync();
    if (rank == 0 && t == 0) {
        bv = c_v[0];
        br = c_r[0];
        for (int r = 1; r < CL; r++) {
            if (am_better(c_v[r], c_r[r], bv, br)) {
                bv = c_v[r];
                br = c_r[r];
            }
        }
        p.dst[row] = br == AM_NONE ? -1 : am_col(br);
    }
#else
    (void) p;
    __trap();
#endif
}

// ---- argmax without clusters (any compute capability; #814: the V100s). The same pick in two
// launches: argmax_split_kernel, grid parts x rows, each CTA its slice's best (value, rank) into
// part[row * parts + slice]; argmax_join_kernel, a warp a row, the parts' best into dst. The
// tie order is a total order (see the file comment), so the split cannot change the pick.

// Grid parts x rows, AM_T threads; slice as in argmax_cluster_kernel with `parts` CTAs a row.
__global__ void __launch_bounds__(AM_T) argmax_split_kernel(legion_argmax_split p) {
    __shared__ float s_v[AM_T / 32];
    __shared__ uint32_t s_r[AM_T / 32];
    const int part = blockIdx.x, row = blockIdx.y, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const float * x = (const float *) ((const char *) p.x + row * p.x_nb1);
    const int slice = ((p.n + p.parts - 1) / p.parts + 3) & ~3;
    const int lo = min(p.n, part * slice), hi = min(p.n, lo + slice);

    float bv = -FLT_MAX;
    uint32_t br = AM_NONE;
    if (((uintptr_t) x & 15) == 0) {
        const int hi4 = lo + ((hi - lo) & ~3);
#pragma unroll 4
        for (int c = lo + 4 * t; c < hi4; c += 4 * AM_T) {
            const float4 v = __ldg((const float4 *) (x + c));
            am_take(v.x, c, bv, br);
            am_take(v.y, c + 1, bv, br);
            am_take(v.z, c + 2, bv, br);
            am_take(v.w, c + 3, bv, br);
        }
        for (int c = hi4 + t; c < hi; c += AM_T) {
            am_take(__ldg(x + c), c, bv, br);
        }
    } else {
        for (int c = lo + t; c < hi; c += AM_T) {
            am_take(__ldg(x + c), c, bv, br);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, bv, o);
        const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
        if (am_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
    }
    if (lane == 0) {
        s_v[warp] = bv;
        s_r[warp] = br;
    }
    __syncthreads();
    if (warp == 0) {
        bv = lane < AM_T / 32 ? s_v[lane] : -FLT_MAX;
        br = lane < AM_T / 32 ? s_r[lane] : AM_NONE;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v = __shfl_xor_sync(0xffffffff, bv, o);
            const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
            if (am_better(v, r, bv, br)) {
                bv = v;
                br = r;
            }
        }
        if (lane == 0) {
            p.part[(int64_t) row * p.parts + part] = make_uint2(__float_as_uint(bv), br);
        }
    }
}

// Grid rows, one warp: the row's parts' best, -1 when no part has a candidate.
__global__ void __launch_bounds__(32) argmax_join_kernel(legion_argmax_split p) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float bv = -FLT_MAX;
    uint32_t br = AM_NONE;
    for (int i = lane; i < p.parts; i += 32) {
        const uint2 c = p.part[(int64_t) row * p.parts + i];
        if (am_better(__uint_as_float(c.x), c.y, bv, br)) {
            bv = __uint_as_float(c.x);
            br = c.y;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, bv, o);
        const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
        if (am_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
    }
    if (lane == 0) {
        p.dst[row] = br == AM_NONE ? -1 : am_col(br);
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's
// cudaLaunchKernelExC (see gate_q8.cu for why they are pointers to anonymous-namespace kernels).
// Top-k: PER = 4, 8, 16, 32, 64, then 0 (entries re-read on every pass).
extern "C" const void * const legion_topk_cluster_kernels[6] = {
    reinterpret_cast<const void *>(&topk_cluster_kernel<4>),
    reinterpret_cast<const void *>(&topk_cluster_kernel<8>),
    reinterpret_cast<const void *>(&topk_cluster_kernel<16>),
    reinterpret_cast<const void *>(&topk_cluster_kernel<32>),
    reinterpret_cast<const void *>(&topk_cluster_kernel<64>),
    reinterpret_cast<const void *>(&topk_cluster_kernel<0>),
};
extern "C" const void * const legion_argmax_cluster_kernel = reinterpret_cast<const void *>(&argmax_cluster_kernel);
extern "C" const void * const legion_argmax_split_kernel = reinterpret_cast<const void *>(&argmax_split_kernel);
extern "C" const void * const legion_argmax_join_kernel = reinterpret_cast<const void *>(&argmax_join_kernel);
