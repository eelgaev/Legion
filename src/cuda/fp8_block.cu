// The expert matmul of block-scaled FP8 weights (src/model/fp8_block.rs, whose CPU reference
// defines it): E4M3 bytes with one F32 scale per (block_k x block_m) tile, as DeepSeek-style
// `FP8_BLOCK_SCALES` checkpoints ship them (Qwen3.8-Flash-Next's MTP head experts, 128 x 128).
// Kernel body only: the Rust side launches it with cudaLaunchKernel.
//
// A grouped GEMV for the few rows an MTP head's MoE runs (its output rows only): one warp per
// output row, four rows per block, grid x over the (slot, token) pairs. Each lane reads 16 weight
// bytes at a time, which never straddle a scale block (block_k is a multiple of 16), sums the
// 16 products in F32 and adds them times the block's scale. Plain SIMT (sm_70 and up); the sums
// run in another order than the CPU's, so values match it to rounding, not bit for bit.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" {

// `Args` in src/model/fp8_block.rs.
struct legion_fp8_block_args {
    const uint8_t * w;    // [k, m, n_expert] E4M3, rows of k bytes, experts w_nb2 bytes apart
    const float *   s;    // [k / bk, m / bm, n_expert] scales, contiguous
    const float *   x;    // [k, x_n1, n_tok] contiguous; x_n1 is 1 (shared) or n_used
    const char *    ids;  // I32 [n_used, n_tok], byte strides ids_nb0 / ids_nb1
    float *         dst;  // [m, n_used, n_tok] contiguous
    int64_t         k, m, n_expert, w_nb2;
    int64_t         bk, bm;
    int64_t         x_n1, n_used;
    int64_t         ids_nb0, ids_nb1;
};

}  // extern "C"

namespace {

__device__ __forceinline__ float2 e4m3x2(uint16_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) two, __NV_E4M3);
    return __half22float2(*(const __half2 *) &h);
}

__global__ void __launch_bounds__(128) fp8_block_mmid_kernel(const legion_fp8_block_args a) {
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
    const uint8_t * w  = a.w + e * a.w_nb2 + row * a.k;
    const float *   x  = a.x + (t * a.x_n1 + (a.x_n1 == 1 ? 0 : u)) * a.k;
    const int64_t   kb = a.k / a.bk;
    const float *   s  = a.s + (e * (a.m / a.bm) + row / a.bm) * kb;
    float acc = 0.0f;
    for (int64_t c = (int64_t) lane * 16; c < a.k; c += 32 * 16) {
        const uint4      raw   = __ldg((const uint4 *) (w + c));
        const uint16_t * pairs = (const uint16_t *) &raw;
        const float4 *   xv    = (const float4 *) (x + c);
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f0 = e4m3x2(pairs[2 * i]);
            const float2 f1 = e4m3x2(pairs[2 * i + 1]);
            const float4 v  = __ldg(xv + i);
            d += f0.x * v.x + f0.y * v.y + f1.x * v.z + f1.y * v.w;
        }
        acc += __ldg(s + c / a.bk) * d;
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

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of fp8_block.rs's `Kernel`. They sit in an anonymous namespace: with relocatable
// device code the device link left extern "C" kernels out of the linked image.
extern "C" const void * const legion_fp8_block_kernels[1] = {
    reinterpret_cast<const void *>(&fp8_block_mmid_kernel),
};
