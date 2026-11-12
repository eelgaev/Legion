// The down matmul of qwen4exp's hyper-connection mixer in prompt steps with BF16 streams
// (#1183, `legion serve --hc-bf16`): `lo = down @ xn` for the BF16 down weight [320 rows of
// 10240] with xn rebuilt on the fly from the streams the combine-and-norm wrote, so the norm
// skips writing xn as BF16 (20 KB a token).
//
// What runs without it: hc_post_norm_b16_kernel (fused.cu) writes xnb = BF16((v * scale) * g)
// per value (v a BF16 stream value widened, scale the stream's norm scale, g gamma, each
// multiply rounded), and ggml-cuda's cuBLAS BF16 GEMM reads xnb (on a GB10 an nvjet kernel of
// 128 x 128 tiles, the M of 320 padded to 384, split over K). Here the GEMM's B operand is that
// same value, computed from the streams, the scales and gamma with the norm's own two rounded
// multiplies and its round-to-nearest-even BF16 conversion: the bits of xnb. Only the sums'
// order differs from cuBLAS's, so the output moves within F32 rounding (opt-in with --hc-bf16).
//
// The Rust side (src/model/qwen4/hc_down.rs) decides when it runs, picks the column tile, gives
// the kernel its shared memory and launches it with cudaLaunchKernel from the table at the
// end. This file is the kernels only.
//
// hc_down_gemm<NT> (sm80/hc_down.cuh): one CTA of 16 warps per 32 * NT tokens and all 320
// rows; warp w takes rows (w % 4) * 80 (5 m16 tiles) and tokens (w / 4) * 8 * NT. The weights'
// rows, the streams' token rows (64 bytes a row, 16-byte chunks swizzled) and gamma go through 3
// shared-memory stages of 32 k values by cp.async. The B fragments are rebuilt in registers
// after ldmatrix (each value: widened, times the token's scale for the stage's stream, times
// gamma, rounded to BF16). Every output is one chain of m16n8k16 BF16 products with F32 sums
// over k = 0, 16, 32, ... from 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {
struct legion_hc_down {
    const __nv_bfloat16 * w;  // down weight [m rows][k], w_ld values apart (16-byte rows)
    const __nv_bfloat16 * x;  // BF16 streams [n][k] (token t's hc streams of e values), x_ld apart
    const float * sc;         // each token's per-stream norm scales [n][hc], sc_ld floats apart
    const float * g;          // gamma [k]
    float * dst;              // lo [n][m], dst_ld floats apart
    int64_t w_ld, x_ld, sc_ld, dst_ld;
    int32_t m, k, n, e;       // e: values per stream (a multiple of HD_BK)
};
}

namespace {

constexpr int HD_THREADS = 512;  // 16 warps: 4 along the rows, 4 along the tokens
constexpr int HD_M = 320;        // rows: the whole down weight (5 m16 tiles a warp)
constexpr int HD_BK = 32;        // k values a stage (64-byte rows)
constexpr int HD_STAGES = 3;

// Shared memory of hc_down_gemm<NT>: per stage the weights' rows, 32 * NT token rows (64 bytes
// each) and 32 gamma values.
constexpr int hd_smem(int nt) {
    return HD_STAGES * ((HD_M + 32 * nt) * HD_BK * 2 + HD_BK * 4);
}

// A row of 4 16-byte chunks; chunk c of row r sits at c ^ ((r >> 1) & 3), so the 8 rows an
// ldmatrix phase reads (two to a 128-byte line) hit 32 different banks.
__device__ __forceinline__ int hd_swz(int r, int c) {
    return r * 4 + (c ^ ((r >> 1) & 3));
}

#include "sm80/hc_down.cuh"

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel
// (src/model/qwen4/hc_down.rs): the GEMM for NT = 2, 3 and 4 (64, 96 and 128 tokens a CTA).
extern "C" const void * const legion_hc_down_kernels[3] = {
    reinterpret_cast<const void *>(&hc_down_gemm<2>),
    reinterpret_cast<const void *>(&hc_down_gemm<3>),
    reinterpret_cast<const void *>(&hc_down_gemm<4>),
};

// Dynamic shared memory of the kernels, NT = 2, 3, 4.
extern "C" const int legion_hc_down_smem[3] = { hd_smem(2), hd_smem(3), hd_smem(4) };
