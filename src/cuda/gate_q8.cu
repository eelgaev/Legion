// The sigmoid output gate before a Q8_0 output projection, prompt steps on a GB10: the gated
// values quantized for ggml-cuda's MMQ in the same pass, instead of ggml's fused unary + mul
// (unary_gated_op_kernel<op_sigmoid>, which writes the gated values as F32) and the matmul's own
// quantize_mmq_q8_1 pass (which reads them back). The Rust side (src/model/core/gate_q8.rs)
// launches the kernel itself and hands the blocks to src/cuda/mmq_preq.cu (#526).
//
//   x  = rms_norm(x) * w       optional (a gated DeltaNet layer's output norm): ggml's
//                              rms_norm_f32 + mul on rows of 128 values (norm128)
//   y  = sigmoid(g) * x        ggml's op_sigmoid (unary.cu): 1 / (1 + expf(-g)), times x
//   xq = y quantized as quantize_mmq_q8_1 does for a Q8_0 src0 (MMQ_Q8_1_DS_LAYOUT_D4): per 32
//        values amax, d_inv = 127 / amax, q = roundf(y * d_inv), d = 1 / d_inv; token t's values
//        128 b .. 128 b + 127 in block b * n + t (4 F32 scales, then 128 int8)
//
// The same flags as ggml-cuda (--use_fast_math), so the same arithmetic: ggml's bytes.

#include <cstdint>

extern "C" {

// Value i (of k) of token t of an operand: at t * ld + (i / run) * stride + i % run floats (a
// row, or runs of a head each, as a view of the attention's interleaved query and gate).
struct legion_gate_q8_rows {
    const float * p;
    int64_t       ld, run, stride;
};

struct legion_gate_q8 {
    legion_gate_q8_rows x, g;
    int8_t *            xq;    // MMQ's blocks, see above
    int32_t             k, n;  // k a multiple of 128, run a multiple of 32
    // Null, or x is first RMS-normed per row of 128 values (each block one row) and times nw
    // [128], as ggml's rms_norm + mul (see norm128); ncols = 128, passed as ggml's is.
    const float *       nw;
    float               eps;
    int32_t             ncols;
};

}  // extern "C"

namespace {

// The run's division in 32 bits (k < 2^31): a 64-bit one is a slow call.
__device__ __forceinline__ float gate_q8_at(const legion_gate_q8_rows & r, int64_t t, int i) {
    const unsigned u = i, run = (unsigned) r.run;
    return r.p[t * r.ld + (int64_t) (u / run) * r.stride + u % run];
}

// ggml's rms_norm_f32<256, true> (norm.cu) on a row of 128 values, x[j] holding value 32 j + lane:
// thread c of its block squares value c (tmp = 0 + x * x), each warp sums its 32 by
// warp_reduce_sum's butterfly (warp j: S_j, warps 4-7: 0), then one warp sums the 8 warp sums
// by the same butterfly, which comes to (S0 + S2) + (S1 + S3) (the zeros add exactly); then
// mean = sum / ncols, scale = rsqrtf(mean + eps) and value scale * x * w.
__device__ __forceinline__ void norm128(float (&x)[4], const legion_gate_q8 & p, int lane) {
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float tmp = 0.0f;
        tmp += x[j] * x[j];
#pragma unroll
        for (int m = 16; m >= 1; m /= 2) {
            tmp += __shfl_xor_sync(0xffffffffu, tmp, m);
        }
        s[j] = tmp;
    }
    const float sum   = (s[0] + s[2]) + (s[1] + s[3]);
    const float mean  = sum / p.ncols;
    const float scale = rsqrtf(mean + p.eps);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        x[j] = scale * x[j] * p.nw[32 * j + lane];
    }
}

// A warp per (token, block of 128 values): lane l takes value l of each of the block's 4 groups
// of 32 (all 8 loads issued before the first use).
__global__ void gate_q8_kernel(const legion_gate_q8 p) {
    // 32-bit indices: n * k / 128 < 2^31 (the Rust side checks).
    const unsigned blocks = p.k / 128;
    const unsigned w      = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int      lane   = threadIdx.x % 32;
    if (w >= (unsigned) p.n * blocks) {
        return;
    }
    const int64_t t = w / blocks;
    const int     b = (int) (w % blocks);
    float         z[4], x[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        z[j] = gate_q8_at(p.g, t, 128 * b + 32 * j + lane);
        x[j] = gate_q8_at(p.x, t, 128 * b + 32 * j + lane);
    }
    if (p.nw) {
        norm128(x, p, lane);
    }
    int8_t * blk = p.xq + ((int64_t) b * p.n + t) * 144;
    float    d[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float y = (1.0f / (1.0f + expf(-z[j]))) * x[j];
        // fmaxf is exact, so the butterfly gives ggml's amax.
        float amax = fabsf(y);
#pragma unroll
        for (int m = 16; m >= 1; m /= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        }
        const float d_inv          = 127.0f / amax;
        const signed char q        = roundf(y * d_inv);
        d[j]                       = 1.0f / d_inv;
        blk[16 + 32 * j + lane]    = q;
    }
    if (lane == 0) {
        *(float4 *) blk = make_float4(d[0], d[1], d[2], d[3]);
    }
}

}  // namespace

// The kernel as the CUDA runtime knows it (its host stub), for Rust's cudaLaunchKernel. The
// kernel sits in an anonymous namespace like every other here: with relocatable device code the
// device link left an extern "C" kernel out of the linked image.
extern "C" const void * const legion_gate_q8_kernel = reinterpret_cast<const void *>(&gate_q8_kernel);
