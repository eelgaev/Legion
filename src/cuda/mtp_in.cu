// qwen4exp's MTP head input for prompt take-ins (#1183, with --dense-fp8-prompt): the
// eh_proj matmul's input [enorm(embedding) repeated per stream ; hnorm(h) per stream] in one
// kernel instead of ggml's chain (rms_norm + mul of the embedding, repeat to the hc streams,
// rms_norm + mul of the streams, concat): six nodes, four passes over [2e * hc, n] F32 tensors.
//
// Both norms are ggml-cuda's rms_norm_f32<1024> (rows of 1024 or more values): each of 1024
// threads adds the squares of columns tid, tid + 1024, ... with FFMA, then warp_reduce_sum, the
// 32 warp sums through shared memory and warp_reduce_sum again (block_reduce); mean + eps from
// the sum by 1 / ncols, rsqrtf; then (scale * x) * gamma, each multiply rounded, as the fused
// rms_norm-mul and the unfused rms_norm then mul both compute it. Repeat and concat only move
// values. Built with fast math as ggml-cuda is.
//
// Quantized (Q): the same values written as the eh_proj matmul's MMQ input instead of F32, as
// ggml-cuda's quantize_mmq_q8_1 (quantize.cu, MIT) writes it for a Q8_0 weight
// (MMQ_Q8_1_DS_LAYOUT_D4) with its arithmetic: per 32 values amax, d_inv = 127 / amax,
// q = roundf(x * d_inv), d = 1 / d_inv; value k of column j (token t, stream c: j = t hc + c)
// in block (k / 128) * ncols + j (4 F32 scales, then 128 int8). The matmul's own quantize pass
// goes (src/cuda/mmq_preq.cu runs ggml's MMQ on these blocks).
//
// The Rust side (src/model/qwen4/mtp_in.rs) decides when it runs and launches it with
// cudaLaunchKernel from the table at the end. This file is the kernels only.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {
struct legion_mtp_in {
    const float * tok;  // token t's e embedding values at tok + t * tok_ld
    int64_t tok_ld;
    const float * h;    // token t's hc streams of e values, back to back, at h + t * h_ld
    int64_t h_ld;
    const float * ge;   // enorm's gamma [e]
    const float * gh;   // hnorm's gamma [e * hc]
    // [2e, hc, n] F32 contiguous: per token and stream, enorm then hnorm; quantized, MMQ's
    // blocks of those values (see above; e a multiple of 32)
    float * dst;
    int32_t e, n;
    float eps;
};
}

namespace {

constexpr int MI_THREADS = 1024;  // ggml-cuda's rms_norm block for rows of 1024 or more
constexpr int MI_HC = 4;          // streams (qwen4exp)

// ggml-cuda's warp_reduce_sum (common.cuh): butterfly over the 32 lanes.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// ggml-cuda's block_reduce over the 1024 threads, for N sums at once (each its own tree, the
// same as alone).
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float (*red)[32]) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < N; i++) {
        v[i] = warp_sum(v[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            red[i][warp] = v[i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; i++) {
        v[i] = warp_sum(red[i][lane]);
    }
}

// rms_norm_f32's scale from a row's sum of squares (the same source).
__device__ __forceinline__ float rms_scale(float tmp, int ncols, float eps) {
    const float mean = tmp / ncols;
    return rsqrtf(mean + eps);
}

// quantize_mmq_q8_1's value x of its 32-value group (the calling warp's 32 lanes hold one
// group in lane order): q and the group's scale d. fmaxf is exact, so the butterfly gives
// ggml's amax whatever its order.
__device__ __forceinline__ void q8_of(float x, signed char & q, float & d) {
    float amax = fabsf(x);
#pragma unroll
    for (int m = 16; m >= 1; m /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    const float d_inv = 127.0f / amax;
    q                 = roundf(x * d_inv);
    d                 = 1.0f / d_inv;
}

// Value k (k % 32 == lane) of column j into MMQ's blocks (see above).
__device__ __forceinline__ void q8_put(int8_t * xq, int64_t ncols, int64_t j, int k, signed char q, float d) {
    int8_t * blk = xq + ((int64_t) (k / 128) * ncols + j) * 144;
    blk[16 + k % 128] = q;
    if (k % 32 == 0) {
        ((float *) blk)[(k % 128) / 32] = d;
    }
}

// One block per token. Q: the output quantized (see above).
template <bool Q>
__global__ void __launch_bounds__(MI_THREADS) mtp_in_kernel(legion_mtp_in p) {
    __shared__ float red[MI_HC + 1][32];
    const int tid = threadIdx.x;
    const int64_t t = blockIdx.x;
    const int e = p.e;
    const float * tok = p.tok + t * p.tok_ld;
    const float * h = p.h + t * p.h_ld;

    // The embedding's and each stream's sum of squares.
    float s[MI_HC + 1];
    {
        float acc = 0.0f;
        for (int col = tid; col < e; col += MI_THREADS) {
            const float x = tok[col];
            acc = __fmaf_rn(x, x, acc);
        }
        s[0] = acc;
    }
#pragma unroll
    for (int c = 0; c < MI_HC; c++) {
        float acc = 0.0f;
        for (int col = tid; col < e; col += MI_THREADS) {
            const float x = h[c * e + col];
            acc = __fmaf_rn(x, x, acc);
        }
        s[1 + c] = acc;
    }
    block_sums(s, red);

    float * out = p.dst + t * (int64_t) (2 * e) * MI_HC;
    // Quantized: column t hc + c of hc n, 2e values each (e a multiple of 32, so every warp's
    // 32 columns col below are one group; the loops' bounds are whole warps).
    int8_t * xq = reinterpret_cast<int8_t *>(p.dst);
    const int64_t ncols = (int64_t) p.n * MI_HC;
    const float se = rms_scale(s[0], e, p.eps);
    for (int col = tid; col < e; col += MI_THREADS) {
        const float en = __fmul_rn(__fmul_rn(se, tok[col]), p.ge[col]);
        if constexpr (Q) {
            // The same values in every stream's column: one group's q and scale for all.
            signed char q;
            float d;
            q8_of(en, q, d);
#pragma unroll
            for (int c = 0; c < MI_HC; c++) {
                q8_put(xq, ncols, t * MI_HC + c, col, q, d);
            }
        } else {
#pragma unroll
            for (int c = 0; c < MI_HC; c++) {
                out[c * 2 * e + col] = en;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MI_HC; c++) {
        const float sc = rms_scale(s[1 + c], e, p.eps);
        for (int col = tid; col < e; col += MI_THREADS) {
            const float hn = __fmul_rn(__fmul_rn(sc, h[c * e + col]), p.gh[c * e + col]);
            if constexpr (Q) {
                signed char q;
                float d;
                q8_of(hn, q, d);
                q8_put(xq, ncols, t * MI_HC + c, e + col, q, d);
            } else {
                out[c * 2 * e + e + col] = hn;
            }
        }
    }
}

}  // namespace

// The kernel as the CUDA runtime knows it (its host stub), for Rust's cudaLaunchKernel
// (src/model/qwen4/mtp_in.rs): one block of MI_THREADS per token, no dynamic shared memory.
// F32 output, then quantized.
extern "C" const void * const legion_mtp_in_kernels[2] = {
    reinterpret_cast<const void *>(&mtp_in_kernel<false>),
    reinterpret_cast<const void *>(&mtp_in_kernel<true>),
};
