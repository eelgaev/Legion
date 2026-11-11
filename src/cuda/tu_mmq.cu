// Prompt matmuls of Q4_K, Q5_K, Q6_K and Q8_0 weights on Turing GPUs without tensor cores
// (GTX 16xx, MX450/550: TU11x, compute capability 7.5) with ggml-cuda's MMQ bits, on dp4a
// instead of the int8 MMA. The host side, the shapes and the CPU reference are in
// src/model/tu_mmq.rs, which launches the kernels from `legion_tu_mmq_kernels` with
// cudaLaunchKernel.
//
// On compute capability 7.5 ggml-cuda takes MMQ's MMA path (mma.sync m8n8k16 s8), which these
// chips run without tensor cores: about 2.4 int8 TOPS on a GTX 1660 Ti, against 17 for dp4a
// (a microbenchmark of both). A 1024-token Q4_K prompt matmul there runs at ~2.4 TFLOPS.
//
// MMQ's output for one value is a chain in K order over 32-value chunks, each an exact int dot
// C of the weight's and the input's int8 values and a few float operations with the chunk's
// scales (mmq-vec-dot.cuh, the Turing branches; third_party/ggml/src/ggml-cuda, MIT):
// - Q4_K, Q5_K (vec_dot_q8_1_q8_1_mma): sum += dm.x * ds.x * C; sum += dm.y * ds.y
// - Q8_0 (vec_dot_q8_0_q8_1_mma): sum += C * dA * dB
// - Q6_K (vec_dot_q6_K_q8_1_mma): per 128 values tmp = 0, per 32 tmp += (C0 * sc0 + C1 * sc1)
//   * dB (C0, C1 the dots of the two 16-value halves), then sum += tmp * dA.
// When every CTA of ggml's grid holds a whole tile (ggml's stream-k grid of one CTA per tile,
// taken when the tiles fill 90% of the SMs' waves; the host side takes only those plans), no
// partial sums are added up, so each output is that chain alone whatever the tiling. Here:
// - `quantize<ds4>`: ggml's quantize_mmq_q8_1 (quantize.cu) for MMQ_Q8_1_DS_LAYOUT_D4 (Q6_K,
//   Q8_0) or _DS4 (Q4_K, Q5_K), value for value.
// - `mm<type, JT>`: a CTA per 128 x JT output tile; ggml's own tile loader (the MMA layout) and
//   input tile loads, then each of the 256 threads computes 8 x JT/16 outputs with dp4a, the
//   chunks' float operations written as ggml writes them (same compiler flags, so the same
//   contractions). dp4a sums start at 0x4B400000 (the float 1.5 * 2^23), so the int sum read as
//   a float minus 1.5 * 2^23 is C exactly while |C| < 2^22 (32-value chunks of 8-bit values:
//   |C| <= 32 * 127 * 127 < 2^19), the same value as the int converted. Q6_K's combined
//   int is converted as ggml converts it.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"

#include <cstdint>

extern "C" {
struct legion_tu_mmq {
    const char * w;      // weight blocks [rows][w_ld]
    const float * x;     // F32 [n][x_ld]
    int * yq;            // block_q8_1_mmq [k / 128][n_pad]
    float * dst;         // F32 [n][dst_ld]
    int64_t x_ld;
    int64_t dst_ld;
    int32_t w_ld;        // blocks between weight rows
    int32_t k;           // a multiple of 256
    int32_t n;           // columns
    int32_t n_pad;       // columns rounded up to the tile width
    int32_t rows;        // a multiple of 128
};
}

namespace {

constexpr int THREADS = 256;
constexpr int TILE_ROWS = 128;
constexpr int YK = MMQ_TILE_Y_K;  // ints per column of an input tile: 4 scales, 32 quants
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.0f;

// quantize_mmq_q8_1<ds4 ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4, false> for a dense
// input: grid (n, k / 512 rounded up), 128 threads, 4 values each.
template <bool ds4>
__global__ void __launch_bounds__(128) quantize(const legion_tu_mmq p) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    // k is a multiple of 256: a warp's 128 values are all in or all out.
    if (i0 >= p.k) {
        return;
    }
    const float * src = p.x + blockIdx.x * p.x_ld + i0;
    block_q8_1_mmq * y = (block_q8_1_mmq *) p.yq;

    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    // One 16-byte load where aligned (the same values).
    const float4 xi = (reinterpret_cast<uintptr_t>(src) & 15) == 0
        ? *reinterpret_cast<const float4 *>(src)
        : make_float4(src[0], src[1], src[2], src[3]);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    if (ds4) {
        sum = xi.x + xi.y + xi.z + xi.w;

#pragma unroll
        for (int offset = 32/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    const int64_t ib = k_block * p.n_pad + blockIdx.x;
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;
    if (iqs % 32 == 0) {
        if (ds4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32] = d;
        }
    }
}

#if defined(TURING_MMA_AVAILABLE) && !defined(AMPERE_MMA_AVAILABLE)

// 8 ints (32 int8) of the tile row or column at `p` (16-byte aligned).
__device__ __forceinline__ void load8(const int * p, int (&v)[8]) {
    const int4 a = *reinterpret_cast<const int4 *>(p);
    const int4 b = *reinterpret_cast<const int4 *>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// One of ggml's vec_dot calls (128 values of K from `k00` ints into the weight tile) for this
// thread's outputs: rows tr + 16 r, columns tc + 16 c of the tile.
template <ggml_type type, int NC>
__device__ __forceinline__ void dot(
        const int * __restrict__ tx, const int * __restrict__ ty, float (&sum)[8][NC],
        const int k00, const int tr, const int tc) {
    constexpr int S = ggml_cuda_mmq_get_sram_stride(type, 128, false, GGML_PREC_Q8);

    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        const half2 * x_dm = (const half2 *) tx + 2*MMQ_TILE_NE_K;
        const half2 * y_dm = (const half2 *) ty;
#pragma unroll
        for (int kk = 0; kk < MMQ_TILE_NE_K/QI8_1; ++kk) {
            const int k0 = k00 + kk*QI8_1;
            int a[8][8];
            float2 dmA[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = tr + 16*r;
                load8(tx + i*S + k0, a[r]);
                dmA[r] = __half22float2(x_dm[i*S + k0/QI8_1]);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int j = tc + 16*c;
                int b[8];
                load8(ty + j*YK + 4 + kk*QI8_1, b);
                const float2 dsB = __half22float2(y_dm[j*YK + kk]);
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int acc = MAGIC_I;
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        acc = ggml_cuda_dp4a(a[r][v], b[v], acc);
                    }
                    const float C = __int_as_float(acc) - MAGIC_F;
                    sum[r][c] += dmA[r].x*dsB.x*C;
                    sum[r][c] += dmA[r].y*dsB.y;
                }
            }
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const float * x_df = (const float *) tx + 2*MMQ_TILE_NE_K;
        const float * y_df = (const float *) ty;
#pragma unroll
        for (int kk = 0; kk < MMQ_TILE_NE_K/QI8_0; ++kk) {
            const int k0 = k00 + kk*QI8_0;
            int a[8][8];
            float dA[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = tr + 16*r;
                load8(tx + i*S + k0, a[r]);
                dA[r] = x_df[i*S + k0/QI8_0];
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int j = tc + 16*c;
                int b[8];
                load8(ty + j*YK + 4 + kk*QI8_1, b);
                const float dB = y_df[j*YK + kk];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int acc = MAGIC_I;
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        acc = ggml_cuda_dp4a(a[r][v], b[v], acc);
                    }
                    const float C = __int_as_float(acc) - MAGIC_F;
                    sum[r][c] += C*dA[r]*dB;
                }
            }
        }
    } else {
        static_assert(type == GGML_TYPE_Q6_K, "Q4_K, Q5_K, Q6_K or Q8_0");
        const float * x_df = (const float *) tx + 2*MMQ_TILE_NE_K;
        const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
        const float * y_df = (const float *) ty;
        float tmp[8][NC];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                tmp[r][c] = 0.0f;
            }
        }
#pragma unroll
        for (int kk = 0; kk < MMQ_TILE_NE_K/QI8_1; ++kk) {
            const int k0 = k00 + kk*QI8_1;
            int a[8][8];
            int sc0[8], sc1[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = tr + 16*r;
                load8(tx + i*S + k0, a[r]);
                const int8_t * sc = (const int8_t *) (x_sc + i*S);
                sc0[r] = sc[k0/4 + 0];
                sc1[r] = sc[k0/4 + 1];
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int j = tc + 16*c;
                int b[8];
                load8(ty + j*YK + 4 + kk*QI8_1, b);
                const float dB = y_df[j*YK + kk];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    int c0 = 0, c1 = 0;
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        c0 = ggml_cuda_dp4a(a[r][v],     b[v],     c0);
                        c1 = ggml_cuda_dp4a(a[r][v + 4], b[v + 4], c1);
                    }
                    tmp[r][c] += (c0*sc0[r] + c1*sc1[r])*dB;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float dA = x_df[(tr + 16*r)*S];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sum[r][c] += tmp[r][c]*dA;
            }
        }
    }
}

#endif // defined(TURING_MMA_AVAILABLE) && !defined(AMPERE_MMA_AVAILABLE)

// The matmul: grid (rows / 128, n_pad / JT), 32 x 8 threads (ggml's tile loader's layout).
// Shared memory: the input tile (JT columns of 128 values), then the weight tile (128 rows of
// 256 values in ggml's MMA layout).
template <ggml_type type, int JT>
__global__ void __launch_bounds__(THREADS, 1) mm(const legion_tu_mmq p) {
#if defined(TURING_MMA_AVAILABLE) && !defined(AMPERE_MMA_AVAILABLE)
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles =
        ggml_cuda_mmq_get_load_tiles<type, 128, false, GGML_PREC_Q8>();
    constexpr int NC = JT / 16;
    static_assert(JT % 16 == 0 && (JT*YK) % THREADS == 0, "tile width");

    extern __shared__ int smem_tu_mmq[];
    int * tile_y = smem_tu_mmq;
    int * tile_x = smem_tu_mmq + JT*YK;

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int tr = tid % 16;
    const int tc = tid / 16;
    const int it = blockIdx.x;
    const int jt = blockIdx.y;
    const int offset_x = it*TILE_ROWS*p.w_ld;
    const int * y = p.yq + (int64_t) jt*JT*YK;
    const int nblocks = p.k / qk;

    float sum[8][NC];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            sum[r][c] = 0.0f;
        }
    }

    for (int kb0 = 0; kb0 < nblocks; kb0 += blocks_per_iter) {
        load_tiles(p.w, tile_x, offset_x + kb0, p.rows - it*TILE_ROWS - 1, p.w_ld);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int * by0 = y + (int64_t) p.n_pad*((kb0*qk/QK8_1_MMQ) + half)*YK;
#pragma unroll
            for (int l0 = 0; l0 < JT*YK; l0 += THREADS) {
                tile_y[l0 + tid] = by0[l0 + tid];
            }
            __syncthreads();
            dot<type, NC>(tile_x, tile_y, sum, half*MMQ_TILE_NE_K, tr, tc);
            __syncthreads();
        }
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int j = jt*JT + tc + 16*c;
        if (j >= p.n) {
            continue;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            p.dst[j*p.dst_ld + it*TILE_ROWS + tr + 16*r] = sum[r][c];
        }
    }
#else
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE) && !defined(AMPERE_MMA_AVAILABLE)
}

} // namespace

#define LEGION_TU_MMQ_KERNELS(type) \
    reinterpret_cast<const void *>(&mm<type, 128>), \
    reinterpret_cast<const void *>(&mm<type, 64>)

// 0, 1: quantize (D4, DS4); then by weight type (Q4_K, Q5_K, Q6_K, Q8_0) the matmul with
// tiles 128 and 64 columns wide.
extern "C" const void * const legion_tu_mmq_kernels[10] = {
    reinterpret_cast<const void *>(&quantize<false>),
    reinterpret_cast<const void *>(&quantize<true>),
    LEGION_TU_MMQ_KERNELS(GGML_TYPE_Q4_K),
    LEGION_TU_MMQ_KERNELS(GGML_TYPE_Q5_K),
    LEGION_TU_MMQ_KERNELS(GGML_TYPE_Q6_K),
    LEGION_TU_MMQ_KERNELS(GGML_TYPE_Q8_0),
};
