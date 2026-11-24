// The per-token FP8 quantization of fp8_act.cu (fp8_quant_dyn's arithmetic) and its gate, as
// device functions for the kernels that write an FP8 GEMM's activation codes themselves
// (--dense-fp8-prompt, #1183): qsa_attn.cu's prompt kernel (the attention output gate) uses
// these very expressions, so its codes and scales are fp8_act's bit for bit. Every Legion .cu
// file is compiled with the same flags (build.rs), so the same expression gives the same bits.
//
// A row's amax is max |y| over its values (exact in any order), its scale
// s = amax / 448 (0 for an all-zero row), and value y's code e4m3(y * (1 / s)), saturated.

#pragma once

#include <cuda_fp8.h>

#include <cstdint>

namespace {

// sigmoid or SiLU as fused.cu's gate_bf16_kernel and gated_norm_kernel compute them.
__device__ __forceinline__ float fp8_gate_of(float v, int silu) {
    return silu ? v / (1.0f + expf(-v)) : 1.0f / (1.0f + expf(-v));
}

// The gate kind's value: op(x) * g (op sigmoid, or SiLU with silu).
__device__ __forceinline__ float fp8_gated(float x, float g, int silu) {
    const float a = fp8_gate_of(x, silu);
    return a * g;
}

// A row's scale from its amax.
__device__ __forceinline__ float fp8_row_scale(float amax) {
    return amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 0.0f;
}

// The multiplier of a row of scale s.
__device__ __forceinline__ float fp8_row_inv(float s) {
    return s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
}

// Value v's E4M3 code in a row of multiplier inv.
__device__ __forceinline__ uint8_t fp8_code(float v, float inv) {
    return __nv_cvt_float_to_fp8(__fmul_rn(v, inv), __NV_SATFINITE, __NV_E4M3);
}

}  // namespace
