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
    float * dst;        // [2e, hc, n] contiguous: per token and stream, enorm then hnorm
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

// One block per token.
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
    const float se = rms_scale(s[0], e, p.eps);
    for (int col = tid; col < e; col += MI_THREADS) {
        const float en = __fmul_rn(__fmul_rn(se, tok[col]), p.ge[col]);
#pragma unroll
        for (int c = 0; c < MI_HC; c++) {
            out[c * 2 * e + col] = en;
        }
    }
#pragma unroll
    for (int c = 0; c < MI_HC; c++) {
        const float sc = rms_scale(s[1 + c], e, p.eps);
        for (int col = tid; col < e; col += MI_THREADS) {
            out[c * 2 * e + e + col] = __fmul_rn(__fmul_rn(sc, h[c * e + col]), p.gh[c * e + col]);
        }
    }
}

}  // namespace

// The kernel as the CUDA runtime knows it (its host stub), for Rust's cudaLaunchKernel
// (src/model/qwen4/mtp_in.rs): one block of MI_THREADS per token, no dynamic shared memory.
extern "C" const void * const legion_mtp_in_kernels[1] = {
    reinterpret_cast<const void *>(&mtp_in_kernel),
};
