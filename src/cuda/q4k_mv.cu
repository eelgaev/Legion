// A Q4_K weight times 1-8 F32 columns with the bits of ggml-cuda's quantize_q8_1 + mul_mat_vec_q
// pair on a GB10 (#689): the DFlash drafter's Q4_K matmuls of a draft pass (`--spec-draft-weights
// q4`, `--spec-draft-lm-head q4`: Q/K/V and gate/up stored fused, the attention output and the
// head; rows of 5,120 or 4,096 values). The host side, the shapes, the launch and the CPU reference
// are in src/model/core/q4k_mv.rs, which launches the kernels of `legion_q4k_mv_kernels` with
// cudaLaunchKernelExC.
//
// The sums are those of q4k_head.cu (#836), which follows ggml's mul_mat_vec_q<GGML_TYPE_Q4_K, N>
// (mmvq.cu, vecdotq.cuh, quantize.cu, MIT) on compute capability 12.1: virtual thread t (0 .. 32 G
// - 1, G ggml's warps for N columns) adds the dot product of each of its blocks t / 16, t / 16 + 2
// G, ... (ints 2 (t % 16) and 2 (t % 16) + 1 of each block's 32-value group pair) into its own F32
// sum, from 0, in that order; lane l of warp 0 adds warps 1 .. G - 1's sums of lane l to its own in
// that order, and warp_reduce_sum (xor 16, 8, 4, 2, 1) gives the output. Here one warp computes a
// row: lane l does the work of virtual threads l, l + 32, ..., l + 32 (G - 1), each into a sum of
// its own, then adds them in ggml's warp order and runs ggml's warp_reduce_sum. Each dot product is
// ggml's own vec_dot_q4_K_q8_1_impl_vmmq over the values ggml's vec_dot_q4_K_q8_1 hands it, so the
// output has ggml's bits.
//
// What differs from q4k_head.cu is where the work goes, for rows twice as long and 8 columns:
// - A lane decodes its part of a weight block (nibble ints, 6-bit scales and mins, d and dmin)
//   once for all N columns instead of once per column: the column loop is inside the block loop
//   (each sum still adds its blocks in its own order).
// - The input sits in shared memory in the order a lane reads it: each (column, block, group
//   pair, int pair) one 16-byte piece holding the four Q8_1 ints vec_dot_q4_K_q8_1 reads, and the
//   two groups' F16-rounded scales (ggml's __low2float of the block's ds) as one float2.
// - The first rows' weights are loaded before the kernel waits for the one ahead of it
//   (griddepcontrol.wait, ggml's ggml_cuda_pdl_sync; the weights are no kernel's output), and
//   every CTA quantizes the input itself (ggml's quantize_q8_1: a warp per 32-value block, the
//   block's absolute maximum by ggml's warp reduction, d = amax / 127, q = roundf(x / d)), so
//   there is no quantize launch.
//
// The CTAs stay resident (a grid of SMs x CTAs per SM, the host's choice) and their warps walk
// the rows; a warp reads its next R rows with 16-byte loads into registers while it computes the
// rows before from shared memory.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdint>

extern "C" {
struct legion_q4k_mv {
    const void *  w;       // Q4_K rows of k values, w_ld bytes apart
    int64_t       w_ld;    // a multiple of 16
    int64_t       rows;
    const float * x;       // F32 [n][k], x_ld floats between columns
    int64_t       x_ld;
    float *       dst;     // F32 [n][rows], dst_ld floats between columns
    int64_t       dst_ld;
    int32_t       k;       // NB * 256
    int32_t       n;       // 1-8
};
}

namespace {

constexpr int QB = 144;  // bytes of a Q4_K block: dm, 12 scale bytes, 128 bytes of nibbles

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// ggml's warps per CTA for Q4_K at `n` columns on a GB10 for rows of `nb` blocks (q4k_head.cu's
// ggml_warps; src/model/core/q4k_head.rs's ggml_warps is the host's copy).
__host__ __device__ constexpr int ggml_warps(int n, int nb) {
    const int generic = n <= 4 ? 4 : 2;
    if (n != 1) {
        return generic;
    }
    constexpr int per_warp = VDR_Q4_K_Q8_1_MMVQ * WARP_SIZE / QI4_K;  // blocks a warp takes a step: 2
    const bool small_k = nb < generic * per_warp;
    const int  step    = generic * per_warp;
    const int  iters   = (nb + step - 1) / step;
    const int  wide    = (nb + 2 * step - 1) / (2 * step);
    const int  idle    = 2 * wide - iters;
    return !small_k && idle * 8 <= 2 * wide ? 2 * generic : generic;
}

// NB: blocks per row, N: columns, NW: warps per CTA, R: rows a warp takes per step.
template <int NB, int N, int NW, int R>
__global__ void __launch_bounds__(NW * WARP_SIZE) q4k_mv(const legion_q4k_mv p) {
    constexpr int G        = ggml_warps(N, NB);
    constexpr int CH       = NB * QB / 16;                      // 16-byte chunks per row
    constexpr int CPL      = (CH + WARP_SIZE - 1) / WARP_SIZE;  // chunks per lane
    constexpr int per_warp = VDR_Q4_K_Q8_1_MMVQ * WARP_SIZE / QI4_K;  // ggml's blocks a warp starts at: 2
    constexpr int per_iter = per_warp * G;                            // ggml's blocks_per_iter
    constexpr int TRIPS    = (NB + per_iter - 1) / per_iter;          // a virtual thread's blocks, at most
    static_assert(NB * QB % 16 == 0, "whole 16-byte chunks per row");

    // [N][NB][4 group pairs][4 int pairs] pieces of the four Q8_1 ints, then [N][NB][8] scales.
    extern __shared__ uint4 smem[];
    uint4 *       yu   = smem;
    float *       yd   = (float *) (smem + N * NB * 16);
    constexpr int YQ   = N * NB * 16 + N * NB * 8 / 4;  // their 16-byte chunks
    const int     lane = threadIdx.x % WARP_SIZE;
    const int     warp = threadIdx.x / WARP_SIZE;
    uint4 *       buf  = smem + YQ + warp * CH * R;
    const char *  bufc = (const char *) buf;

    const int64_t gwarp  = (int64_t) blockIdx.x * NW + warp;
    const int64_t nwarps = (int64_t) gridDim.x * NW;
    const int64_t step   = nwarps * R;

    uint4 r[R][CPL];
    const auto load = [&](const int64_t row0) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const uint4 * src = (const uint4 *) ((const char *) p.w + (row0 + j) * p.w_ld);
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH && row0 + j < p.rows) {
                    r[j][i] = __ldg(src + c);
                }
            }
        }
    };
    // The weights are no other kernel's output: the first rows go out before the wait.
    if (gwarp * R < p.rows) {
        load(gwarp * R);
    }
    pdl_launch_dependents();
    pdl_wait();

    // The input as ggml's quantize_q8_1 writes it (a warp per 32-value block), into the layout
    // above: value e of group g (0-7) of block kb goes to piece (kb, g / 2, (e % 16) / 4), int
    // 2 (g % 2) + e / 16, byte e % 4.
    // A warp's blocks are loaded QU at a time before any is reduced (one L2 round trip per QU).
    constexpr int QPW = N * NB * 8 / NW;  // blocks a warp quantizes
    constexpr int QU  = QPW < 8 ? QPW : 8;
    static_assert(N * NB * 8 % NW == 0, "whole blocks per warp");
    for (int i0 = 0; i0 < QPW; i0 += QU) {
        float xv[QU];
#pragma unroll
        for (int u = 0; u < QU; ++u) {
            const int b = warp + NW * (i0 + u);
            if (i0 + u < QPW) {
                xv[u] = p.x[(b / (NB * 8)) * p.x_ld + (b % (NB * 8)) * QK8_1 + lane];
            }
        }
#pragma unroll
        for (int u = 0; u < QU; ++u) {
            if (i0 + u >= QPW) {
                break;
            }
            const int   b    = warp + NW * (i0 + u);
            const int   col  = b / (NB * 8);
            const int   ib   = b % (NB * 8);
            const float xi   = xv[u];
            float       amax = fabsf(xi);
            amax = warp_reduce_max<QK8_1>(amax);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            const int    kb = ib / 8;
            const int    g  = ib % 8;
            char *       piece = (char *) (yu + (col * NB + kb) * 16 + (g / 2) * 4 + (lane % 16) / 4);
            piece[(2 * (g % 2) + lane / 16) * 4 + lane % 4] = q;
            if (lane == 0) {
                yd[(col * NB + kb) * 8 + g] = __low2float(make_half2(d, 0.0f));
            }
        }
    }
    __syncthreads();

    const int kqs = VDR_Q4_K_Q8_1_MMVQ * (lane % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));  // ggml's iqs: 0, 2, .., 30
    const int kb0 = lane / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
    // vec_dot_q4_K_q8_1's offsets for iqs = kqs: the group pair (bq8_offset / 2) and the int.
    const int pair = (kqs / 2) / (QI8_1 / 2);
    const int sub  = (kqs / 2) % 4;
    const int j    = pair;      // vec_dot_q4_K_q8_1's j = bq8_offset / 2
    const int jm   = j & 1;
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    for (int64_t row0 = gwarp * R; row0 < p.rows; row0 += step) {
#pragma unroll
        for (int jj = 0; jj < R; ++jj) {
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH) {
                    buf[jj * CH + c] = r[jj][i];
                }
            }
        }
        __syncwarp();
        if (row0 + step < p.rows) {
            load(row0 + step);
        }

#pragma unroll
        for (int jj = 0; jj < R; ++jj) {
            const int64_t row = row0 + jj;
            if (row >= p.rows) {
                break;
            }
            const char * vrow = bufc + jj * CH * 16;
            // ggml's per-thread sums: tmp[c][w] is virtual thread lane + 32 w's for column c.
            float tmp[N][G];
#pragma unroll
            for (int c = 0; c < N; ++c) {
#pragma unroll
                for (int w = 0; w < G; ++w) {
                    tmp[c][w] = 0.0f;
                }
            }
#pragma unroll
            for (int w = 0; w < G; ++w) {
#pragma unroll
                for (int t = 0; t < TRIPS; ++t) {
                    const int kbx = per_warp * w + kb0 + t * per_iter;
                    if (kbx >= NB) {
                        break;
                    }
                    // vec_dot_q4_K_q8_1's weight part for (kbx, kqs).
                    const block_q4_K * bq4_K = (const block_q4_K *) (vrow + kbx * QB);
                    const int *        q4    = (const int *) (bq4_K->qs + 32 * pair + 4 * sub);
                    int                v[2];
                    v[0] = q4[0];
                    v[1] = q4[4];
                    const uint16_t * scales = (const uint16_t *) bq4_K->scales;
                    const uint32_t   s0     = scales[jm + 0];
                    const uint32_t   s2     = scales[jm + 2];
                    const uint32_t   s4     = scales[jm + 4];
                    uint16_t         aux[2];
                    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
                    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
                    const uint8_t * sc = (const uint8_t *) aux;
                    const uint8_t * m  = sc + 2;
                    const half2     dm = bq4_K->dm;
#pragma unroll
                    for (int c = 0; c < N; ++c) {
                        // The input part: u[0..3] and d8[0..1] of blocks bq8_offset, bq8_offset + 1.
                        const uint4  uq = yu[(c * NB + kbx) * 16 + pair * 4 + sub];
                        const float2 dq = *(const float2 *) (yd + (c * NB + kbx) * 8 + 2 * pair);
                        const int    u[4]  = { (int) uq.x, (int) uq.y, (int) uq.z, (int) uq.w };
                        const float  d8[2] = { dq.x, dq.y };
                        tmp[c][w] += vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
                    }
                }
            }
            // ggml: warp 0's lane adds warps 1 .. G - 1 in order, then the warp sum; every lane
            // holds every column's sum, lane c writes column c.
            float out = 0.0f;
#pragma unroll
            for (int c = 0; c < N; ++c) {
                float sum = tmp[c][0];
#pragma unroll
                for (int w = 1; w < G; ++w) {
                    sum += tmp[c][w];
                }
                sum = warp_reduce_sum<WARP_SIZE>(sum);
                if (lane == c) {
                    out = sum;
                }
            }
            if (lane < N) {
                p.dst[lane * p.dst_ld + row] = out;
            }
        }
        __syncwarp();
    }
}

}  // namespace

// The kernels for the host stubs: q4k_mv<NB, N, NW, R> at index ((nb * 8 + N - 1) * 2 + nw) * 2 +
// r, nb 0 for 16 blocks (4,096 values) and 1 for 20 (5,120), nw 0 for 8 warps and 1 for 16, r 0
// for one row a warp step and 1 for two (src/model/core/q4k_mv.rs's kernel_index).
#define LEGION_Q4K_MV_R(NB, N, NW) \
    reinterpret_cast<const void *>(&q4k_mv<NB, N, NW, 1>), reinterpret_cast<const void *>(&q4k_mv<NB, N, NW, 2>)
#define LEGION_Q4K_MV_N(NB, N) LEGION_Q4K_MV_R(NB, N, 8), LEGION_Q4K_MV_R(NB, N, 16)
#define LEGION_Q4K_MV_NB(NB)                                                                           \
    LEGION_Q4K_MV_N(NB, 1), LEGION_Q4K_MV_N(NB, 2), LEGION_Q4K_MV_N(NB, 3), LEGION_Q4K_MV_N(NB, 4), \
        LEGION_Q4K_MV_N(NB, 5), LEGION_Q4K_MV_N(NB, 6), LEGION_Q4K_MV_N(NB, 7), LEGION_Q4K_MV_N(NB, 8)

extern "C" const void * const legion_q4k_mv_kernels[64] = {
    LEGION_Q4K_MV_NB(16),
    LEGION_Q4K_MV_NB(20),
};
