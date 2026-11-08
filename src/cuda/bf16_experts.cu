// The expert matmul of BF16 weights (src/model/bf16_experts.rs, whose CPU reference defines it),
// as GLM-5.3-Flash's NVFP4 release ships its MTP layer's routed experts.
// Kernel body only: the Rust side launches it with cudaLaunchKernel.
//
// A grouped GEMV for the few rows an MTP head's MoE runs (its output rows only): one warp per
// output row, four rows per block, grid x over the (slot, token) pairs. Each lane reads 8
// weights (16 bytes) and 8 activations at a time and sums their products in F32; the warp adds
// the lanes' sums. Plain SIMT (sm_70 and up; BF16 widened by a shift); the sums run in another
// order than the CPU's, so values match it to rounding, not bit for bit.

#include <cstdint>

extern "C" {

// `Args` in src/model/bf16_experts.rs.
struct legion_bf16_experts_args {
    const uint8_t * w;    // [k, m, n_expert] BF16, rows of 2k bytes, experts w_nb2 bytes apart
    const float *   x;    // [k, x_n1, n_tok] contiguous; x_n1 is 1 (shared) or n_used
    const char *    ids;  // I32 [n_used, n_tok], byte strides ids_nb0 / ids_nb1
    float *         dst;  // [m, n_used, n_tok] contiguous
    int64_t         k, m, n_expert, w_nb2;
    int64_t         x_n1, n_used;
    int64_t         ids_nb0, ids_nb1;
};

}  // extern "C"

namespace {

// The two BF16 values of a 32-bit word (the lower one first in memory) as floats.
__device__ __forceinline__ float bf16_lo(uint32_t v) {
    return __uint_as_float(v << 16);
}

__device__ __forceinline__ float bf16_hi(uint32_t v) {
    return __uint_as_float(v & 0xffff0000u);
}

__device__ __forceinline__ float dot2(uint32_t w, float a, float b) {
    return bf16_lo(w) * a + bf16_hi(w) * b;
}

__global__ void __launch_bounds__(128) bf16_experts_mmid_kernel(const legion_bf16_experts_args a) {
    const int     warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t row  = (int64_t) blockIdx.y * 4 + warp;
    if (row >= a.m) {
        return;
    }
    const int64_t slot = blockIdx.x;
    const int64_t u = slot % a.n_used, t = slot / a.n_used;
    float * out = a.dst + (t * a.n_used + u) * a.m + row;
    const int32_t e = *(const int32_t *) (a.ids + u * a.ids_nb0 + t * a.ids_nb1);
    if (e < 0 || e >= a.n_expert) {
        if (lane == 0) {
            *out = 0.0f;
        }
        return;
    }
    const uint8_t * w = a.w + e * a.w_nb2 + row * a.k * 2;
    const float *   x = a.x + (t * a.x_n1 + (a.x_n1 == 1 ? 0 : u)) * a.k;
    float acc = 0.0f;
    for (int64_t c = (int64_t) lane * 8; c < a.k; c += 32 * 8) {
        const uint4    raw = __ldg((const uint4 *) (w + c * 2));
        const float4 * xv  = (const float4 *) (x + c);
        const float4   v0  = __ldg(xv);
        const float4   v1  = __ldg(xv + 1);
        acc += dot2(raw.x, v0.x, v0.y) + dot2(raw.y, v0.z, v0.w) + dot2(raw.z, v1.x, v1.y) +
               dot2(raw.w, v1.z, v1.w);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) {
        *out = acc;
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel.
// They sit in an anonymous namespace: with relocatable device code the device link left
// extern "C" kernels out of the linked image.
extern "C" const void * const legion_bf16_experts_kernels[1] = {
    reinterpret_cast<const void *>(&bf16_experts_mmid_kernel),
};
