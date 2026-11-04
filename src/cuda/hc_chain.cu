// qwen4exp (Qwen3.8-Flash-Next) hyper-connection mixer: the elementwise ggml-cuda ops of the
// chain that runs where neither the small-step op nor the fused combine-and-norm takes a step
// (decode and verify steps of BF16 mixers without --hc-mix-bf16, the MTP head), and the low-rank
// gate's scale and SiLU in every mixer's tail (#836: all compute in Legion kernels). The host
// side, when each runs and the CPU references are in src/model/qwen4/hc_chain.rs.
//
// Each kernel is ggml-cuda's own expressions in ggml's order, compiled with the same
// --use_fast_math, so it gives ggml's bits:
//
//   lo_act:   y = silu(scale * x + bias)            ggml_scale (scale.cu), then ggml_silu
//                                                   (unary.cuh: x / (1 + expf(-x)))
//   combine:  w = 2 * sigmoid(inj * (1/hc))         ggml_scale, ggml_sigmoid (unary.cu), ggml_scale
//             r' = out * w + r                      ggml's dsv4_hc_post_f32<false> (dsv4-hc.cu)
//   rms_mul:  y = rsqrt(mean(x^2) + eps) * x * g    ggml's rms_norm_f32 (norm.cu) with its fused
//                                                   multiply: the same per-thread sums over the
//                                                   same block size, the same warp and block
//                                                   reduction, then (scale * x) * g
//
// Adapted from ggml (MIT): ggml/src/ggml-cuda/{scale.cu,unary.cu,unary.cuh,dsv4-hc.cu,norm.cu,
// common.cuh}.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_hc_chain {
    const float * x;    // lo_act: F32 [n] contiguous; combine: out [e, n]; rms_mul: x [e, hc, n]
    const float * res;  // combine: the streams [e, hc, n]
    const float * inj;  // combine: the block's inject weights [hc, n]
    const float * g;    // rms_mul: gamma [e, hc] contiguous
    float *       dst;  // F32, contiguous ([n], [e, hc, n])
    int64_t       e, hc, n;
    // Strides in floats: x (combine: out's row; rms_mul: x's stream and token), res's stream and
    // token, inj's token.
    int64_t x1, x2, r1, r2, i1;
    float   scale, bias;  // lo_act: ggml_scale's; combine: the inject scale (1/hc); rms_mul: eps
    float   scale2;       // combine: the second ggml_scale (2)
};

int legion_hc_lo_act_launch(const legion_hc_chain * p, void * stream);
int legion_hc_combine_launch(const legion_hc_chain * p, void * stream);
int legion_hc_rms_mul_launch(const legion_hc_chain * p, void * stream);

}  // extern "C"

namespace {

constexpr int WARP = 32;

// ggml's scale_f32 then ggml_cuda_op_silu_single, one value a thread (in place: x may be dst).
__global__ void lo_act_kernel(const float * x, float * dst, const float scale, const float bias,
                              const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float s = scale * x[i] + bias;
    dst[i] = s / (1.0f + expf(-s));
}

// ggml's scale, sigmoid and scale of the inject weights (each a kernel writing F32 there, the
// same values here in registers), then dsv4_hc_post_f32<false>: one value a thread in ggml's
// order (i0 fastest, then the stream, then the token).
__global__ void combine_kernel(const legion_hc_chain p, const float bias) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= p.e * p.hc * p.n) {
        return;
    }
    const int64_t i0   = ir % p.e;
    const int64_t idst = (ir / p.e) % p.hc;
    const int64_t it   = ir / (p.e * p.hc);
    const float   a    = p.scale * p.inj[idst + it * p.i1] + bias;
    const float   sg   = 1.0f / (1.0f + expf(-a));
    const float   post = p.scale2 * sg + bias;

    float sum = p.x[i0 + it * p.x1] * post;
    sum += p.res[i0 + idst * p.r1 + it * p.r2];

    p.dst[i0 + idst * p.e + it * p.e * p.hc] = sum;
}

// ggml's warp_reduce_sum (common.cuh).
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

// ggml's rms_norm_f32<block_size, do_multiply> for one row (stream `blockIdx.x` of token
// `blockIdx.y`), dst contiguous.
template <int block_size>
__global__ void __launch_bounds__(block_size) rms_mul_kernel(const legion_hc_chain p) {
    __shared__ float s_sum[WARP];
    const int     tid   = threadIdx.x;
    const int     ncols = (int) p.e;
    const int64_t row   = blockIdx.x;
    const int64_t tok   = blockIdx.y;
    const float * xc    = p.x + tok * p.x2 + row * p.x1;
    const float * mulc  = p.g + row * p.e;
    float *       dstc  = p.dst + (tok * p.hc + row) * p.e;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = xc[col];
        tmp += xi * xi;
    }
    // ggml's block_reduce<SUM, block_size>.
    tmp = warp_sum(tmp);
    if constexpr (block_size > WARP) {
        const int warp_id = tid / WARP;
        const int lane_id = tid % WARP;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = 0.0f;
        if (lane_id < block_size / WARP) {
            tmp = s_sum[lane_id];
        }
        tmp = warp_sum(tmp);
    }

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + p.scale);

    for (int col = tid; col < ncols; col += block_size) {
        dstc[col] = scale * xc[col] * mulc[col];
    }
}

}  // namespace

extern "C" int legion_hc_lo_act_launch(const legion_hc_chain * p, void * stream) {
    if (p->n < 1) {
        return (int) cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int64_t blocks = (p->n + threads - 1) / threads;
    lo_act_kernel<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(p->x, p->dst, p->scale,
                                                                            p->bias, p->n);
    return (int) cudaGetLastError();
}

extern "C" int legion_hc_combine_launch(const legion_hc_chain * p, void * stream) {
    if (p->e < 1 || p->hc < 1 || p->n < 1) {
        return (int) cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int64_t blocks = (p->e * p->hc * p->n + threads - 1) / threads;
    combine_kernel<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(*p, p->bias);
    return (int) cudaGetLastError();
}

extern "C" int legion_hc_rms_mul_launch(const legion_hc_chain * p, void * stream) {
    if (p->e < 1 || p->hc < 1 || p->n < 1 || p->hc > 65535 || p->n > 65535) {
        return (int) cudaErrorInvalidValue;
    }
    const dim3 grid((unsigned) p->hc, (unsigned) p->n, 1);
    const auto st = (cudaStream_t) stream;
    // ggml's choice (rms_norm_f32_cuda / rms_norm_mul_f32_cuda): 256 threads under 1024 columns.
    if (p->e < 1024) {
        rms_mul_kernel<256><<<grid, 256, 0, st>>>(*p);
    } else {
        rms_mul_kernel<1024><<<grid, 1024, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}
