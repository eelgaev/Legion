// Quantized dense matmuls of 2-8 columns on a V100 with the bits of ggml-cuda's quantize_q8_1 +
// mul_mat_vec_q pair (#1210): MTP verify steps and small decode batches of the GGUF checkpoints'
// K-quant, IQ4 and Q8_0 weights. Kernel bodies only: the host side, the shapes, the choice of
// variant and the CPU reference are in src/model/core/q_small_m.rs, which launches the kernels
// from `legion_q_small_m_kernels` with cudaLaunchKernel.
//
// ggml's mul_mat_vec_q<type, N> (mmvq.cu, MIT) for N = 2-8 columns runs CTAs of G warps
// (calc_nwarps: G = 4 for 2-4 columns and 2 for 5-8, on every NVIDIA table) over 2 rows:
// virtual thread t (0 .. 32 G - 1) adds ggml's vec_dot_<type>_q8_1 (vecdotq.cuh, MIT) of blocks
// t / TPB, t / TPB + 32 G / TPB, ... (TPB = qi / vdr lanes a block; vdr (t % TPB) the block's
// part) of each row and column into an F32 sum of its own, from 0, in that order; lane l of
// warp 0 then adds warps 1 .. G - 1's sums of lane l to its own, in that order, and
// warp_reduce_sum (xor 16, 8, 4, 2, 1) gives the output. A row's sum does not depend on the
// rows next to it, nor on the other columns. 1Cat-vLLM's table has that kernel at 1.5x a
// one-column step's time at 4 columns and 2.6x at 8 on a V100 (Q4_K 17408 x 5120); ggml 0.27's
// takes 1.6-1.8x at 4 and 1.8x at 5 (its MMVQ bound for Q4_K). Measured here, the K-quants' MMVQ
// is bound by the dot products' integer work (fixed by the bits), not by loads in flight: the
// variants below beat it on IQ4_XS only, so the Rust side makes the op opt-in.
//
// - `quantize`: x as Q8_1 blocks, ggml's quantize_q8_1 (quantize.cu, MIT) value for value: one
//   thread per value, a warp per 32-value block, amax and sum by ggml's warp reductions,
//   d = amax / 127, q = roundf(x / d), ds = (d, sum) as half2. Column j's blocks at j k / 32.
// - `mv<T, N, R>`: CTAs of NW warps, each warp R rows. Lane l does the work of virtual threads
//   l, l + 32, ..., l + 32 (G - 1) of each of its rows: blocks m BPM + l / TPB (BPM = 32 / TPB)
//   for m = 0, 1, 2, ..., the block of step m into sum m % G (virtual thread 32 (m % G) + l takes
//   exactly those blocks, in increasing m). Each step's G blocks of R rows are unrolled, so a
//   lane has G R blocks' weight loads in flight at once, the weights of a block are decoded once
//   for every column (ggml's vec_dot inlined N times on the same loads: nvcc keeps one copy), and
//   a column's activation ints are loaded once for the R rows. Then each lane adds its G sums in
//   ggml's warp order and runs ggml's warp_reduce_sum: per output the same products added in the
//   same order as ggml's, ggml's bits for any R and NW.
// - `mv_cta<T, N, R>`: ggml's kernel itself with R rows a CTA instead of 2 (an activation load
//   serves R rows): ggml's bits too.
//
// Ideas from 1Cat-vLLM (Apache-2.0) `csrc/sm70_turbomind/ops/qpn_pair_sm70.cuh` and
// `gguf_native_pair_readers_sm70.cuh` (a small-M matmul that keeps many weight loads in flight
// per warp and no shared-memory reduction), and
// v100-skinny (MIT). No code from either: the arithmetic is ggml's own functions.
//
// The same flags as ggml-cuda (--use_fast_math), so the same arithmetic. The kernel bodies are
// compiled for compute capability 7.x only (the Rust side takes the op on a V100 alone, where
// ggml's MMVQ / MMQ choice is the one it follows); newer targets get a trap.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdint>

extern "C" {
struct legion_q_small_m {
    const void *  w;       // rows of k values of the weight's type, w_ld bytes apart
    int64_t       w_ld;
    int32_t       rows;
    int32_t       k;       // a multiple of the type's block (and of 32)
    const float * x;       // F32 [n][k], x_ld floats between columns (the quantization's input)
    int64_t       x_ld;
    void *        xq;      // Q8_1 blocks [n][k / 32]
    float *       dst;     // F32 [n][rows], dst_ld floats between columns
    int64_t       dst_ld;
    int32_t       n;       // 2-8
};
}

namespace {

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// Warps of a matvec CTA (each warp works alone).
constexpr int NW = 4;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
#define LEGION_QSM_BODY 1
#endif

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n).
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_q_small_m p) {
#ifdef LEGION_QSM_BODY
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= p.k) {
        return;
    }
    const int64_t j = blockIdx.y;
    block_q8_1 * y = (block_q8_1 *) p.xq + j * (p.k / QK8_1);
    const int64_t ib  = i0 / QK8_1;
    const int64_t iqs = i0 % QK8_1;

    const float xi = p.x[j * p.x_ld + i0];
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
#else
    __trap();
#endif
}

// mmvq.cu's get_vec_dot_q_cuda and get_vdr_mmvq for the types the op takes.
template <ggml_type T> struct qsm_type;

#define LEGION_QSM_TYPE(T, FN, VDR)                                                              \
    template <> struct qsm_type<T> {                                                             \
        static constexpr int vdr = VDR;                                                          \
        static __device__ __forceinline__ float dot(                                             \
                const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,            \
                const int & kbx, const int & iqs) {                                              \
            return FN(vbq, bq8_1, kbx, iqs);                                                     \
        }                                                                                        \
    };

LEGION_QSM_TYPE(GGML_TYPE_Q4_K,   vec_dot_q4_K_q8_1,   VDR_Q4_K_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_Q5_K,   vec_dot_q5_K_q8_1,   VDR_Q5_K_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_Q6_K,   vec_dot_q6_K_q8_1,   VDR_Q6_K_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_IQ4_XS, vec_dot_iq4_xs_q8_1, VDR_IQ4_XS_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_Q8_0,   vec_dot_q8_0_q8_1,   VDR_Q8_0_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_IQ4_NL, vec_dot_iq4_nl_q8_1, VDR_IQ4_NL_Q8_1_MMVQ)
LEGION_QSM_TYPE(GGML_TYPE_Q3_K,   vec_dot_q3_K_q8_1,   VDR_Q3_K_Q8_1_MMVQ)

#undef LEGION_QSM_TYPE

// ggml's warps per CTA (calc_nwarps) for N = 2-8 columns: the same on every NVIDIA table.
static constexpr __host__ __device__ int ggml_warps(int n) {
    return n <= 4 ? 4 : 2;
}

// Rows [R (blockIdx.x NW + warp), + R) of dst = w . x for N columns (see the file comment).
template <ggml_type T, int N, int R>
__global__ void __launch_bounds__(NW * WARP_SIZE) mv(const legion_q_small_m p) {
#ifdef LEGION_QSM_BODY
    constexpr int qk  = ggml_cuda_type_traits<T>::qk;
    constexpr int qi  = ggml_cuda_type_traits<T>::qi;
    constexpr int vdr = qsm_type<T>::vdr;
    constexpr int G   = ggml_warps(N);
    constexpr int TPB = qi / vdr;           // lanes (virtual threads) a block
    constexpr int BPM = WARP_SIZE / TPB;    // blocks a warp takes a step
    static_assert(WARP_SIZE % TPB == 0, "whole blocks a warp step");

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * NW + threadIdx.x / WARP_SIZE) * R;
    if (row0 >= p.rows) {
        return;
    }
    const int nb  = p.k / qk;
    const int sub = lane / TPB;
    const int kqs = vdr * (lane % TPB);
    const int ys  = p.k / QK8_1;
    const block_q8_1 * y = (const block_q8_1 *) p.xq;

    const char * w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, p.rows - 1);
        w[r] = (const char *) p.w + (int64_t) row * p.w_ld;
    }

    float acc[R][N][G];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int g = 0; g < G; ++g) {
                acc[r][j][g] = 0.0f;
            }
        }
    }

    for (int m0 = 0; m0 * BPM < nb; m0 += G) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int kbx = (m0 + g) * BPM + sub;
            if (kbx < nb) {
                const int kby = kbx * (qk / QK8_1);
#pragma unroll
                for (int j = 0; j < N; ++j) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r][j][g] += qsm_type<T>::dot(w[r], &y[j * ys + kby], kbx, kqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float s = acc[r][j][0];
#pragma unroll
            for (int g = 1; g < G; ++g) {
                s += acc[r][j][g];
            }
            s = warp_reduce_sum<WARP_SIZE>(s);
            if (lane == 0 && row0 + r < p.rows) {
                p.dst[j * p.dst_ld + row0 + r] = s;
            }
        }
    }
#else
    __trap();
#endif
}

// ggml's own layout with R rows a CTA instead of its 2 (`mv_cta<T, N, R>`): CTAs of
// blockDim (32, G), warp w the virtual threads 32 w .. 32 w + 31 of R rows, the sums of warps
// 1 .. G - 1 through shared memory into warp 0 in ggml's order, then ggml's warp_reduce_sum.
// Rows [R blockIdx.x, + R).
template <ggml_type T, int N, int R>
__global__ void __launch_bounds__(ggml_warps(N) * WARP_SIZE) mv_cta(const legion_q_small_m p) {
#ifdef LEGION_QSM_BODY
    constexpr int qk  = ggml_cuda_type_traits<T>::qk;
    constexpr int qi  = ggml_cuda_type_traits<T>::qi;
    constexpr int vdr = qsm_type<T>::vdr;
    constexpr int G   = ggml_warps(N);
    constexpr int blocks_per_iter = vdr * G * WARP_SIZE / qi;

    const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = R * blockIdx.x;
    const int nb   = p.k / qk;
    const int kqs  = vdr * (tid % (qi / vdr));
    const int ys   = p.k / QK8_1;
    const block_q8_1 * y = (const block_q8_1 *) p.xq;

    const char * w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, p.rows - 1);
        w[r] = (const char *) p.w + (int64_t) row * p.w_ld;
    }

    float tmp[N][R] = {{0.0f}};
    for (int kbx = tid / (qi / vdr); kbx < nb; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                tmp[j][r] += qsm_type<T>::dot(w[r], &y[j * ys + kby], kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[G - 1][N][R][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                tmp_shared[threadIdx.y - 1][j][r][threadIdx.x] = tmp[j][r];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int l = 0; l < G - 1; ++l) {
                tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
            }
            tmp[j][r] = warp_reduce_sum<WARP_SIZE>(tmp[j][r]);
        }
    }
    if (threadIdx.x < R && row0 + (int) threadIdx.x < p.rows) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r == (int) threadIdx.x) {
                    p.dst[j * p.dst_ld + row0 + r] = tmp[j][r];
                }
            }
        }
    }
#else
    __trap();
#endif
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel:
// the quantization, then for each type (Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, IQ4_NL, Q3_K: the
// Rust side's TYPES) and each N = 2-8 five matvecs (the Rust side's VARIANTS): `mv` with R = 1,
// 2, 4 rows a warp, `mv_cta` with R = 4, 8 rows a CTA, at 1 + (type * 7 + N - 2) * 5 + variant.
#define LEGION_QSM_N(T, N)                                 \
    reinterpret_cast<const void *>(&mv<T, N, 1>),          \
    reinterpret_cast<const void *>(&mv<T, N, 2>),          \
    reinterpret_cast<const void *>(&mv<T, N, 4>),          \
    reinterpret_cast<const void *>(&mv_cta<T, N, 4>),      \
    reinterpret_cast<const void *>(&mv_cta<T, N, 8>)
#define LEGION_QSM_T(T)                                    \
    LEGION_QSM_N(T, 2), LEGION_QSM_N(T, 3), LEGION_QSM_N(T, 4), LEGION_QSM_N(T, 5), \
    LEGION_QSM_N(T, 6), LEGION_QSM_N(T, 7), LEGION_QSM_N(T, 8)

extern "C" const void * const legion_q_small_m_kernels[1 + 7 * 7 * 5] = {
    reinterpret_cast<const void *>(&quantize),
    LEGION_QSM_T(GGML_TYPE_Q4_K),
    LEGION_QSM_T(GGML_TYPE_Q5_K),
    LEGION_QSM_T(GGML_TYPE_Q6_K),
    LEGION_QSM_T(GGML_TYPE_IQ4_XS),
    LEGION_QSM_T(GGML_TYPE_Q8_0),
    LEGION_QSM_T(GGML_TYPE_IQ4_NL),
    LEGION_QSM_T(GGML_TYPE_Q3_K),
};
