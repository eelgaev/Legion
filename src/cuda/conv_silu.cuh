// conv_silu (the gated DeltaNet's causal conv and SiLU, with the l2 norm of its q and k heads)
// shared by fused.cu's conv_silu_kernel and fp8_mm.cu's qkv GEMM that runs it in its epilogue
// (#1183): the parameters, and each output's arithmetic as the same expressions in both, so
// the same bits. Shapes and semantics are in src/model/core/fused.rs.

#pragma once

#include <cstdint>

extern "C" {

// conv_silu: out[c, t, s] = silu(sum_k in[t + k, c, s] * w[k, c]) over in = [state | x] along
// time, and hist = the last `keep` inputs (d_conv - 1: the next step's state; d_conv - 1 + nt:
// all of [state | x], an MTP verify step's window).
struct legion_conv_silu {
    const float * x;      // [C, nt, ns]: element (c, t, s) at x + c + t * x_ld + s * x_sd (floats)
    int64_t x_ld, x_sd;
    const float * state;  // [d - 1, C, ns]: (j, c, s) at state + j + c * st_ld + s * st_sd
    int64_t st_ld, st_sd;
    const float * w;      // [d, C], contiguous
    float * out;          // [C, nt, ns], contiguous
    float * hist;         // [keep, C, ns], contiguous: the last keep inputs of [state | x]
    int32_t C, nt, ns, d, keep;
    int32_t chunk;        // tokens per thread
    // With norm_blocks > 0 (blocks of norm_head threads, one head each): the first norm_blocks
    // blocks' outputs (a gated DeltaNet's q and k heads) are l2-normalized per token as
    // delta_prep does it (eps norm_eps, scale norm_scale), bit for bit.
    int32_t norm_blocks, norm_head;
    float norm_eps, norm_scale;
    uint32_t grid[3];
    uint32_t block;
    // Or null (#689 round 11, LEGION_DEBUG_GDN_PF_CONV=1): pf_bytes of a weight to prefetch into
    // L2, 16-byte aligned, 4 KiB a thread from the kernel's start (compute capability 9.0+).
    const char * pf;
    int64_t pf_bytes;
};

}  // extern "C"

namespace {
namespace conv_math {

// The widest kernel conv_silu takes.
constexpr int MAX_D = 8;

// One output: the window's dot with the taps (win[0] the oldest input, tap order), then SiLU.
__device__ __forceinline__ float silu_dot(const float (&win)[MAX_D], const float (&w)[MAX_D], int d) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_D; k++) {
        if (k < d) {
            acc += win[k] * w[k];
        }
    }
    return acc / (1.0f + __expf(-acc));
}

// An output's square, as delta_prep_kernel's sum over its row starts.
__device__ __forceinline__ float square(float y) {
    float tmp = 0.0f;
    tmp += y * y;
    return tmp;
}

// An output scaled by its head's sum of squares `sum` (delta_prep_kernel's mean and scale).
__device__ __forceinline__ float normed(float y, float sum, int norm_head, float eps, float norm_scale) {
    const float mean = sum / norm_head;
    const float scale = rsqrtf(mean + eps);
    return norm_scale * (scale * y);
}

}  // namespace conv_math
}  // namespace
