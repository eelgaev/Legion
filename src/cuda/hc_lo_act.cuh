// The low-rank gate's activation of qwen4exp's hyper-connection mixer: ggml's scale_f32
// (scale.cu: scale * x + bias) then ggml_cuda_op_silu_single (unary.cuh: x / (1 + expf(-x))).
// One definition for every kernel that makes it (hc_chain.cu's lo_act_kernel, hc_down.cu's
// epilogue, #1183), each built with the same --use_fast_math, so they give the same bits.
//
// Adapted from ggml (MIT): ggml/src/ggml-cuda/{scale.cu,unary.cuh}.

#pragma once

__device__ __forceinline__ float legion_hc_lo_act(const float x, const float scale, const float bias) {
    const float s = scale * x + bias;
    return s / (1.0f + expf(-s));
}
