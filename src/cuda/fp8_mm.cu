// FP8 W8A16 matmul: y = W x for a weight W stored as raw FP8 E4M3 bytes (the checkpoint's own
// values, no per-block scale; the per-tensor scale is applied by the graph afterwards) and F32
// activations. The graph side (op node, layouts, the CPU reference) is src/model/fp8.rs.
//
// Layouts (ggml order, dim 0 fastest):
// - w:   [k, m] bytes, row j (k bytes) is output j's weights
// - x:   F32 [k, n], column t (k floats, contiguous) is token t; columns `x_ld` floats apart
// - dst: F32 [m, n], contiguous
//
// Two kernels (and an opt-in third, the sliced-K GEMM of sm80/fp8_sliced_k.cuh for 5-64 tokens):
// - GEMV (n <= 4, decode; before sm_80 any n, in chunks of 8): memory-bound. One warp per
//   output row; each lane reads 16 weight bytes per step (one 128-bit load), converts them to
//   F32 (E4M3 -> half is exact, then -> float) and multiplies with the matching activations of
//   every token (read through the cache: all warps of the grid read the same x), accumulating
//   in F32; a warp reduction ends each row. fp8_gemv_rows is the same arithmetic with up to 4
//   rows per warp and the next weight words loaded ahead, for 4-9 tokens on sm_80+ (4 by
//   default, 5-9 when the op asks, gemv_max_n): every token gets its one-token bits.
// - GEMM (n > 8, prefill): tensor cores through the portable WMMA API, BF16 x BF16 -> F32. A
//   CTA computes a 64 x 64 tile of y (64 output rows x 64 tokens), 8 warps of 16 x 32; per
//   K step of 32 it converts its W tile from E4M3 to BF16 (exact) and its x tile from F32 to
//   BF16 (round to nearest: the "A16" of W8A16, as a BF16 cuBLAS GEMM rounds its F32 input),
//   in shared memory.
//
// The tensor-core GEMMs live per architecture in sm80/fp8_mm.cuh (BF16 m16n8k16, W8A16) and
// sm89/fp8_mm.cuh (E4M3 m16n8k32, W8A8), included below; bf16_mma() and fp8_mma() pick them per
// device at run time. The GEMVs and the activation quantizer are portable and stay here.
//
// Launched on the stream of ggml's CUDA backend, between graph segments (src/ggml/external.rs).

#include "ggml-cuda/common.cuh"
#include "device_cache.cuh"
#include "launch_pdl.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>

extern "C" {

struct legion_fp8_mm {
    const uint8_t * w;  // [k, m] E4M3
    const float * x;    // [k, n], columns x_ld elements apart; BF16 when x_bf16
    float * dst;        // [m, n]
    int64_t k, m, n;
    int64_t x_ld;
    int32_t x_bf16;     // activations are BF16 (the large-batch GEMM's input)
    const float * wscale;  // the weight's scale (device), or null: row j's is wscale[j * wscale_stride]
    int64_t wscale_stride; // 0: one per-tensor value; 1: one per output row
    int64_t gemv_max_n;    // sm_80+: up to this many tokens (at most ROWS_MAX_N) take the GEMV
                           // (fp8_gemv_rows); 0 for the default split (GEMV up to GEMV_N)
    int64_t sliced_k;      // sm_80+, BF16 activations of SLICED_MIN_N..SLICED_MAX_N tokens not
                           // taken by the GEMV: 0 the default GEMMs, 1 the sliced-K GEMM
                           // (fp8_gemm_sliced, its tile picked by n), 2 + i its tile i (tuning)
};

// 0 on success; 1 for an unsupported shape (k not a multiple of 16); 2 for a launch error.
int legion_fp8_mm_launch(const legion_fp8_mm * p, void * stream);

// W8A8 (the checkpoint's FP8 scheme: E4M3 weights, E4M3 activations with a static per-tensor
// input scale). legion_fp8_quant writes e4m3(x / scale) (round to nearest even, saturating at
// +-448) for activations F32 [k, n] (columns x_ld floats apart) into out [k, n] bytes;
// legion_fp8_mm8 computes y = scale * (W . xq) for the quantized activations.
//
// W8A8 with dynamic per-token scales (--fp8-act a8-dynamic, for checkpoints without a static
// input scale): legion_fp8_quant_dyn gives every token t its own scale s_t = amax_t / 448
// (amax_t = max |x[., t]|; 0 for an all-zero token) and writes e4m3(x * (1 / s_t)) and s_t;
// legion_fp8_mm8 then takes scale_stride 1 and, for a per-row weight scale, wscale_stride 1.
struct legion_fp8_mm8 {
    const uint8_t * w;    // [k, m] E4M3
    const uint8_t * x;    // [k, n] E4M3, contiguous
    const float * scale;  // the input scale (device): token t's is scale[t * scale_stride]
    float * dst;          // [m, n]
    int64_t k, m, n;
    const float * wscale;  // the weight's scale (device), or null: row j's is wscale[j * wscale_stride]
    int64_t scale_stride;  // 0: one static input scale; 1: one per token
    int64_t wscale_stride; // 0: one per-tensor weight scale; 1: one per output row
    // A residual [m, n] F32 (dst's layout) added to each output with __fadd_rn (never contracted
    // into an FMA): dst then holds the bits of the plain matmul followed by ggml's add. Taken by
    // the large-batch GEMM only (more than 64 tokens on E4M3 tensor cores); null: none.
    const float * res;
};
int legion_fp8_quant_launch(const float * x, int64_t x_ld, uint8_t * out, int64_t k, int64_t n,
                            const float * scale, void * stream);
// x: [k, n] F32, columns x_ld floats apart; out: [k, n] E4M3 codes; scales: [n] F32. 1 when k
// or x_ld is not a multiple of 4.
int legion_fp8_quant_dyn_launch(const float * x, int64_t x_ld, uint8_t * out, float * scales, int64_t k,
                                int64_t n, void * stream);
int legion_fp8_mm8_launch(const legion_fp8_mm8 * p, void * stream);

// Whether the current device runs the large-batch GEMM (BF16 activations): sm_80+.
int legion_fp8_mm_bf16_gemm(void);
// Whether the current device runs the W8A8 GEMMs on E4M3 tensor cores: sm_89+.
int legion_fp8_mm_fp8_gemm(void);

}  // extern "C"

namespace {

constexpr int GEMV_MAX_N = 8;
// Up to this many tokens the F32 GEMV runs; above, the tensor-core GEMM.
constexpr int GEMV_N = 4;
// On GPUs with BF16 tensor cores, ROWS_MIN_N up to ROWS_DEFAULT_N tokens take fp8_gemv_rows
// instead of fp8_gemv (the same bits; at 2-3 tokens fp8_gemv is as fast);
// legion_fp8_mm::gemv_max_n raises that up to ROWS_MAX_N (instead of the tensor-core GEMM:
// other bits, each token's one-token ones). 9 tokens: the GB10's default verify of 8 drafts.
constexpr int ROWS_MIN_N = 4;
constexpr int ROWS_DEFAULT_N = GEMV_N;
constexpr int ROWS_MAX_N = 9;
// fp8_gemv_rows' rows per warp: 4 while that still gives each SM this many warps, else 2
// (4 rows on 5120 outputs leave the GB10's 48 SMs a ragged last wave).
constexpr int ROWS_MIN_WARPS = 32;

__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) two, __NV_E4M3);
    return __half22float2(*(const __half2 *) &h);
}

template <int N>
__global__ void __launch_bounds__(128) fp8_gemv(const legion_fp8_mm p) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t row = (int64_t) blockIdx.x * 4 + warp;
    if (row >= p.m) {
        return;
    }
    const uint8_t * w = p.w + row * p.k;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    for (int64_t k0 = (int64_t) lane * 16; k0 < p.k; k0 += 32 * 16) {
        const uint4 raw = __ldg((const uint4 *) (w + k0));
        const uint16_t * pairs = (const uint16_t *) &raw;
        float wf[16];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float2 f = e4m3x2_to_float2(pairs[i]);
            wf[2 * i]     = f.x;
            wf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            const float4 * xv = (const float4 *) (p.x + t * p.x_ld + k0);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 v = __ldg(xv + i);
                acc[t] += wf[4 * i] * v.x + wf[4 * i + 1] * v.y + wf[4 * i + 2] * v.z + wf[4 * i + 3] * v.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N; t++) {
        float s = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, o);
        }
        if (lane == 0) {
            p.dst[t * p.m + row] = p.wscale ? s * p.wscale[row * p.wscale_stride] : s;
        }
    }
}

// Verify-sized batches (N tokens): fp8_gemv's arithmetic with R rows per warp. Every (row,
// token) sum is fp8_gemv<1>'s: lane l takes the row's 16-byte chunks l, l + 32, ... in order, the
// same products and adds per 4 weights (the same expression, in the same order), the same
// xor-shuffle reduction and epilogue, so each token gets the bits it gets alone in a decode step.
// What changes is the traffic: the warp's activations, loaded once per chunk (all N tokens' at
// once), serve R rows (fp8_gemv reloads them for every row, its limit at 5+ tokens), and the
// next chunk's R weight words are loaded before this chunk's math.
template <int N, int R>
__global__ void __launch_bounds__(128) fp8_gemv_rows(const legion_fp8_mm p) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t row0 = ((int64_t) blockIdx.x * 4 + warp) * R;
    if (row0 >= p.m) {
        return;
    }
    const uint8_t * w[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        // Rows past the last one read the last row again; their sums are not stored.
        w[r] = p.w + min(row0 + r, p.m - 1) * p.k;
    }
    float acc[R][N];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < N; t++) {
            acc[r][t] = 0.0f;
        }
    int64_t k0 = (int64_t) lane * 16;
    uint4 raw[R];
    if (k0 < p.k) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            raw[r] = __ldg((const uint4 *) (w[r] + k0));
        }
    }
    for (; k0 < p.k; k0 += 32 * 16) {
        float wf[R][16];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint16_t * pairs = (const uint16_t *) &raw[r];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float2 f = e4m3x2_to_float2(pairs[i]);
                wf[r][2 * i]     = f.x;
                wf[r][2 * i + 1] = f.y;
            }
        }
        if (k0 + 32 * 16 < p.k) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                raw[r] = __ldg((const uint4 *) (w[r] + k0 + 32 * 16));
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float4 v[N];
#pragma unroll
            for (int t = 0; t < N; t++) {
                v[t] = __ldg((const float4 *) (p.x + t * p.x_ld + k0) + i);
            }
#pragma unroll
            for (int t = 0; t < N; t++)
#pragma unroll
                for (int r = 0; r < R; r++) {
                    acc[r][t] += wf[r][4 * i] * v[t].x + wf[r][4 * i + 1] * v[t].y + wf[r][4 * i + 2] * v[t].z + wf[r][4 * i + 3] * v[t].w;
                }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int64_t row = row0 + r;
#pragma unroll
        for (int t = 0; t < N; t++) {
            float s = acc[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                s += __shfl_xor_sync(0xffffffff, s, o);
            }
            if (lane == 0 && row < p.m) {
                p.dst[t * p.m + row] = p.wscale ? s * p.wscale[row * p.wscale_stride] : s;
            }
        }
    }
}

// sm_80 and newer: the W8A16 GEMMs on mma.sync m16n8k16 BF16 (fp8_gemm, fp8_gemm_big), and the
// opt-in sliced-K GEMM for verify-sized batches (fp8_gemm_sliced).
#include "sm80/fp8_mm.cuh"
#include "sm80/fp8_sliced_k.cuh"

// Tokens the sliced-K GEMM takes when asked (legion_fp8_mm::sliced_k); SLICED_MIN_N and
// SLICED_MAX_N must match SLICED_K_MIN_N / SLICED_K_MAX_N in src/model/fp8.rs, and SLICED_K_ALIGN
// (the smallest K group, 4 warps x 64) its SLICED_K_ALIGN.
constexpr int SLICED_MIN_N = 5;
constexpr int SLICED_MAX_N = 64;
constexpr int SLICED_K_ALIGN = 4 * sk::TILE_K;

template <int KW, int TOK, int STAGES, int MT, int MINB>
void launch_sliced_t(const legion_fp8_mm & p, cudaStream_t st) {
    using C = sk::cfg<KW, TOK, STAGES, MT>;
    static_assert(C::SMEM <= 96 * 1024, "shared memory");
    if constexpr (C::SMEM > 48 * 1024) {
        static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
        legion_device::smem_opt_in(opted_in, fp8_gemm_sliced<KW, TOK, STAGES, MT, MINB>, (int) C::SMEM);
    }
    const dim3 grid((unsigned) ((p.m + C::ROWS - 1) / C::ROWS), (unsigned) ((p.n + TOK - 1) / TOK));
    launch_pdl(fp8_gemm_sliced<KW, TOK, STAGES, MT, MINB>, grid, 32 * KW, (int) C::SMEM, st, p);
}

// The sliced-K tiles: (K warps, tokens per CTA, stages, m16 row tiles, CTAs per SM). Tile i is
// legion_fp8_mm::sliced_k = 2 + i; sliced_tile() picks one by n and m, the others are there for
// tuning (cuda_sliced_k_timings in tests/fp8.rs).
// Tiles 12-19 keep tile 0's 16 K warps (so its sums, bit for bit) with 8 tokens, two stages,
// 32 rows or two CTAs per SM kept resident (#689; `legion fp8-sliced-bench` times them).
constexpr int SLICED_TILES = 20;
constexpr int SLICED_KW[SLICED_TILES] = {16, 8, 4, 8, 8, 4, 8, 4, 8, 4, 2, 4, 16, 16, 16, 16, 16, 16, 16, 16};
constexpr int SLICED_TOK[SLICED_TILES] = {16, 16, 16, 24, 32, 32, 32, 64, 64, 48, 64, 64, 8, 8, 8, 8, 16, 16, 16, 8};

bool launch_sliced_tile(int tile, const legion_fp8_mm & p, cudaStream_t st) {
    if (tile < 0 || tile >= SLICED_TILES || p.k % (SLICED_KW[tile] * sk::TILE_K) != 0 || p.n > SLICED_TOK[tile] * 65535) {
        return false;
    }
    switch (tile) {
        case 0: launch_sliced_t<16, 16, 1, 1, 1>(p, st); break;
        case 1: launch_sliced_t<8, 16, 1, 1, 3>(p, st); break;
        case 2: launch_sliced_t<4, 16, 1, 1, 7>(p, st); break;
        case 3: launch_sliced_t<8, 24, 1, 1, 2>(p, st); break;
        case 4: launch_sliced_t<8, 32, 1, 1, 2>(p, st); break;
        case 5: launch_sliced_t<4, 32, 1, 1, 2>(p, st); break;
        case 6: launch_sliced_t<8, 32, 1, 2, 2>(p, st); break;
        case 7: launch_sliced_t<4, 64, 1, 2, 2>(p, st); break;
        case 8: launch_sliced_t<8, 64, 1, 2, 1>(p, st); break;
        case 9: launch_sliced_t<4, 48, 1, 2, 2>(p, st); break;
        case 10: launch_sliced_t<2, 64, 1, 2, 4>(p, st); break;
        case 11: launch_sliced_t<4, 64, 2, 2, 1>(p, st); break;
        case 12: launch_sliced_t<16, 8, 1, 1, 1>(p, st); break;
        case 13: launch_sliced_t<16, 8, 2, 1, 1>(p, st); break;
        case 14: launch_sliced_t<16, 8, 1, 2, 1>(p, st); break;
        case 15: launch_sliced_t<16, 8, 1, 1, 2>(p, st); break;
        case 16: launch_sliced_t<16, 16, 2, 1, 1>(p, st); break;
        case 17: launch_sliced_t<16, 16, 1, 2, 1>(p, st); break;
        case 18: launch_sliced_t<16, 16, 1, 1, 2>(p, st); break;
        default: launch_sliced_t<16, 8, 2, 2, 1>(p, st); break;
    }
    return true;
}

// The sliced-K tile for n tokens, m rows and k (a multiple of SLICED_K_ALIGN), from GB10 timings
// of the 27B's FP8 shapes (#664): the preferred one, or the 4-warp tile of the same width when k
// is not a multiple of its K group. 33-64 tokens on 8 warps only with enough 32-row CTAs for
// the SMs (at m = 5120 the 8-warp tile runs one 80 KiB CTA per SM and loses 20-35%). Up to 16
// tokens tile 0's two-stage twin (tile 16, the same sums): 0.5-1.7% faster at 8 tokens (#689).
int sliced_tile(int64_t n, int64_t m, int64_t k) {
    int tile, narrow;
    if (n <= 16) {
        tile = 16; narrow = 2;
    } else if (n <= 32) {
        tile = 6; narrow = 5;
    } else {
        tile = m >= 8192 ? 8 : 7; narrow = 7;
    }
    return k % (SLICED_KW[tile] * sk::TILE_K) == 0 ? tile : narrow;
}

void launch_gemm_big(const legion_fp8_mm & p, cudaStream_t st) {
    // A refusal shows as the launch's error.
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    legion_device::smem_opt_in(opted_in, fp8_gemm_big, (int) big::SMEM);
    const dim3 grid((unsigned) ((p.n + big::BN - 1) / big::BN), (unsigned) ((p.m + big::BM - 1) / big::BM));
    fp8_gemm_big<<<grid, 256, big::SMEM, st>>>(p);
}

// Streaming multiprocessors of the current device, cached per device.
int sm_count() {
    // At least one, so a failed query still gives a grid.
    return std::max(1, legion_device::sm_count());
}

template <int BN, int BK, int W>
void launch_gemm_w(const legion_fp8_mm & p, cudaStream_t st) {
    constexpr int BM = 16 * W;
    const dim3 grid((unsigned) ((p.m + BM - 1) / BM), (unsigned) ((p.n + BN - 1) / BN));
    fp8_gemm<BN, BK, W><<<grid, 32 * W, 0, st>>>(p);
}

// Few tokens: memory-bound, so as many CTAs (weight streams) as two per SM, with the tallest
// tiles that still give that many.
template <int BN, int BK>
void launch_gemm(const legion_fp8_mm & p, cudaStream_t st) {
    const int64_t cols = (p.n + BN - 1) / BN, want = 2 * (int64_t) sm_count();
    if ((p.m + 127) / 128 * cols >= want) {
        launch_gemm_w<BN, BK, 8>(p, st);
    } else if ((p.m + 63) / 64 * cols >= want) {
        launch_gemm_w<BN, BK, 4>(p, st);
    } else {
        launch_gemm_w<BN, BK, 2>(p, st);
    }
}

template <int N>
void launch_gemv(const legion_fp8_mm & p, cudaStream_t st) {
    fp8_gemv<N><<<(unsigned) ((p.m + 3) / 4), 128, 0, st>>>(p);
}

template <int N, int R>
void launch_gemv_rows_r(const legion_fp8_mm & p, cudaStream_t st) {
    fp8_gemv_rows<N, R><<<(unsigned) ((p.m + 4 * R - 1) / (4 * R)), 128, 0, st>>>(p);
}

// 4 rows per warp while that still gives every SM ROWS_MIN_WARPS warps (4 per CTA), else 2.
template <int N>
void launch_gemv_rows_n(const legion_fp8_mm & p, cudaStream_t st) {
    if ((p.m + 3) / 4 >= (int64_t) ROWS_MIN_WARPS * sm_count()) {
        launch_gemv_rows_r<N, 4>(p, st);
    } else {
        launch_gemv_rows_r<N, 2>(p, st);
    }
}

// ROWS_MIN_N..ROWS_MAX_N tokens: fp8_gemv_rows.
void launch_gemv_rows(const legion_fp8_mm & p, cudaStream_t st) {
    switch (p.n) {
        case 4: launch_gemv_rows_n<4>(p, st); break;
        case 5: launch_gemv_rows_n<5>(p, st); break;
        case 6: launch_gemv_rows_n<6>(p, st); break;
        case 7: launch_gemv_rows_n<7>(p, st); break;
        case 8: launch_gemv_rows_n<8>(p, st); break;
        default: launch_gemv_rows_n<9>(p, st); break;
    }
}

// Whether the current device has BF16 tensor cores (compute capability 8.0+), cached per device.
bool bf16_mma(cudaStream_t) {
    return legion_device::compute_capability() >= 80;
}


// ---- W8A8 ----

__global__ void fp8_quant(const float * x, int64_t x_ld, uint8_t * out, int64_t k, int64_t n,
                          const float * scale) {
    const float inv = 1.0f / *scale;
    const int64_t pairs = k / 2 * n;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = i / (k / 2), c = i % (k / 2) * 2;
        const float2 v = *(const float2 *) (x + t * x_ld + c);
        const __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
        *(__nv_fp8x2_storage_t *) (out + t * k + c) = q;
    }
}

// The W8A8 epilogue's factor for output row `row` of token `tok`: the token's input scale (one
// static value, or its own) times the row's weight scale (one value, or its own) when there is
// one. With a static input scale and a per-tensor weight scale it is one product for every
// element, `s0 = a8_static(p)`, which the kernels compute once before the epilogue as they
// always did.
__device__ __forceinline__ float a8_static(const legion_fp8_mm8 & p) {
    return p.wscale ? *p.scale * *p.wscale : *p.scale;
}

__device__ __forceinline__ float a8_scale(const legion_fp8_mm8 & p, float s0, int64_t row, int64_t tok) {
    if (p.scale_stride == 0 && p.wscale_stride == 0) {
        return s0;
    }
    const float xs = p.scale[tok * p.scale_stride];
    return p.wscale ? xs * p.wscale[row * p.wscale_stride] : xs;
}

// Dynamic per-token quantization: one CTA of QD_THREADS per token, 4 activations per thread and
// step. Pass 1 finds the token's amax (max is exact, so the order does not matter), pass 2
// reads the row again (from L2 now) and writes the codes. In IEEE F32 with round to nearest:
// s = amax / 448, codes e4m3(x * (1 / s)) saturated at +-448, as `fp8::quantize_dynamic`'s
// CPU reference does. The scheme (amax / 448 per token, multiply by the inverse) follows
// NInfer's fp8_a8_quantize_kernel (src/ops/linear/kernels/fp8/fp8_a8.cu, commit 64d10fb,
// Copyright the NInfer authors, Apache License 2.0); this kernel takes any k (a multiple of 4)
// and F32 activations.
constexpr int QD_THREADS = 256;

__global__ void __launch_bounds__(QD_THREADS) fp8_quant_dyn(const float * x, int64_t x_ld, uint8_t * out,
                                                           float * scales, int64_t k) {
    __shared__ float warp_max[QD_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * row = x + t * x_ld;
    float amax = 0.0f;
    for (int64_t c = (int64_t) threadIdx.x * 4; c < k; c += QD_THREADS * 4) {
        const float4 v = *(const float4 *) (row + c);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = warp_max[0];
#pragma unroll
    for (int w = 1; w < QD_THREADS / 32; w++) {
        amax = fmaxf(amax, warp_max[w]);
    }
    const float s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 0.0f;
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    uint8_t * o = out + t * k;
    for (int64_t c = (int64_t) threadIdx.x * 4; c < k; c += QD_THREADS * 4) {
        const float4 v = *(const float4 *) (row + c);
        const __nv_fp8x2_storage_t lo =
            __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(v.x, inv), __fmul_rn(v.y, inv)), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi =
            __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(v.z, inv), __fmul_rn(v.w, inv)), __NV_SATFINITE, __NV_E4M3);
        *(uint32_t *) (o + c) = (uint32_t) lo | ((uint32_t) hi << 16);
    }
    if (threadIdx.x == 0) {
        scales[t] = s;
    }
}

// Few tokens: one warp per output row, 16 weight and 16 activation bytes per lane and step, F32
// products (exact: E4M3 values multiply exactly in F32) and F32 sums.
template <int N>
__global__ void __launch_bounds__(128) fp8_gemv8(const legion_fp8_mm8 p) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t row = (int64_t) blockIdx.x * 4 + warp;
    if (row >= p.m) {
        return;
    }
    const uint8_t * w = p.w + row * p.k;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    for (int64_t k0 = (int64_t) lane * 16; k0 < p.k; k0 += 32 * 16) {
        const uint4 raw = __ldg((const uint4 *) (w + k0));
        const uint16_t * pairs = (const uint16_t *) &raw;
        float wf[16];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float2 f = e4m3x2_to_float2(pairs[i]);
            wf[2 * i] = f.x;
            wf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            const uint4 xr = __ldg((const uint4 *) (p.x + t * p.k + k0));
            const uint16_t * xp = (const uint16_t *) &xr;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float2 f = e4m3x2_to_float2(xp[i]);
                acc[t] += wf[2 * i] * f.x + wf[2 * i + 1] * f.y;
            }
        }
    }
    const float s = a8_static(p);
#pragma unroll
    for (int t = 0; t < N; t++) {
        float v = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        if (lane == 0) {
            p.dst[t * p.m + row] = v * a8_scale(p, s, row, t);
        }
    }
}

// sm_89 and newer: the W8A8 GEMMs on mma.sync m16n8k32 E4M3 (fp8_gemm8, fp8_gemm8_skinny).
#include "sm89/fp8_mm.cuh"
// sm_90 and newer: fp8_gemm8 fed by TMA (fp8_gemm8_tma).
#include "sm90/fp8_gemm8_tma.cuh"

template <int BN, int W, bool RES, int STAGES>
void launch_gemm8_skinny_w(const legion_fp8_mm8 & p, cudaStream_t st) {
    constexpr size_t SMEM = a8s::smem<BN, W, STAGES>();
    static_assert(SMEM <= 96 * 1024, "shared memory");
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    legion_device::smem_opt_in(opted_in, fp8_gemm8_skinny<BN, W, RES, STAGES>, (int) SMEM);
    const dim3 grid((unsigned) ((p.m + 16 * W - 1) / (16 * W)), (unsigned) ((p.n + BN - 1) / BN));
    fp8_gemm8_skinny<BN, W, RES, STAGES><<<grid, 32 * W, SMEM, st>>>(p);
}

// As many CTAs as two per SM, with the tallest tiles that still give that many. ADA: the stage
// counts that fit two CTAs in an Ada SM (a8s::stages_ada).
template <int BN, bool RES = false, bool ADA = false>
void launch_gemm8_skinny(const legion_fp8_mm8 & p, cudaStream_t st) {
    const int64_t cols = (p.n + BN - 1) / BN, want = 2 * (int64_t) sm_count();
    if ((p.m + 127) / 128 * cols >= want) {
        launch_gemm8_skinny_w<BN, 8, RES, ADA ? a8s::stages_ada<BN, 8>() : a8s::stages<BN, 8>()>(p, st);
    } else if ((p.m + 63) / 64 * cols >= want) {
        launch_gemm8_skinny_w<BN, 4, RES, ADA ? a8s::stages_ada<BN, 4>() : a8s::stages<BN, 4>()>(p, st);
    } else {
        launch_gemm8_skinny_w<BN, 2, RES, ADA ? a8s::stages_ada<BN, 2>() : a8s::stages<BN, 2>()>(p, st);
    }
}

// Ada (compute capability 8.9: an RTX 4090 has 128 SMs where the GB10 has 48), more than 64
// tokens: the big GEMM's 128 x 128 tiles leave most SMs idle while they number under a quarter
// of the SMs, so there the skinny kernel runs over 64-token column tiles instead (#1213, an
// RTX 4090, k 4,096, 2,688 rows, 128 tokens: 21 tiles, 62 -> 29 us; with 40 to 80 tiles it
// measured from 0.9x to 1.2x the GEMM's time, so those stay). The same bits: every output takes
// the same MMAs in the same k order.
bool gemm8_ada_skinny(const legion_fp8_mm8 & p) {
    const int64_t tiles = (p.m + 127) / 128 * ((p.n + a8::BN - 1) / a8::BN);
    return p.k % a8s::BK == 0 && tiles * 4 < sm_count() && legion_device::compute_capability() == 89;
}

template <bool RES, class C>
void launch_gemm8_w(const legion_fp8_mm8 & p, cudaStream_t st) {
    constexpr size_t SMEM = C::SMEM;
    static_assert(SMEM <= 96 * 1024, "shared memory");
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    legion_device::smem_opt_in(opted_in, fp8_gemm8<RES, C>, (int) SMEM);
    const dim3 grid((unsigned) ((p.n + a8::BN - 1) / a8::BN), (unsigned) ((p.m + C::TM - 1) / C::TM));
    fp8_gemm8<RES, C><<<grid, 256, SMEM, st>>>(p);
}

// fp8_gemm8_tma on consumer Blackwell (compute capability 12.x: the GB10, where it was measured,
// #1133); fp8_gemm8 elsewhere.
bool gemm8_tma_device() {
    const int cc = legion_device::compute_capability();
    return cc >= 120 && cc < 130;
}

// The tensor map of `rows` rows of k bytes at `base` (E4M3 weights or activations), in boxes of
// box_k bytes x box_rows rows in the swizzle of a8::Cfg's SWZ layout for BK = box_k; false
// where the driver refuses it.
bool gemm8_tma_map(CUtensorMap * map, const uint8_t * base, int64_t k, int64_t rows, int box_k, int box_rows) {
    if (((uintptr_t) base & 15) != 0 || k % 16 != 0 || rows <= 0) {
        return false;
    }
    const cuuint64_t dims[2]    = { (cuuint64_t) k, (cuuint64_t) rows };
    const cuuint64_t strides[1] = { (cuuint64_t) k };
    const cuuint32_t box[2]     = { (cuuint32_t) box_k, (cuuint32_t) box_rows };
    const cuuint32_t elem[2]    = { 1, 1 };
    const CUtensorMapSwizzle swz = box_k == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void *) base, dims, strides, box, elem,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// 4 warps: the first wave's start spread over GEMM8_STAGGER_NS_PER_K ns per byte of k (about one
// tile's time: 123 us at k 6,144 on the GB10, where 60 us gained about half as much).
constexpr int64_t GEMM8_STAGGER_NS_PER_K = 20;

// fp8_gemm8_tma where the device and the operands allow it (4 warps: and two CTAs fit an SM);
// false: not launched.
template <bool RES, class C>
bool launch_gemm8_tma(const legion_fp8_mm8 & p, cudaStream_t st) {
    constexpr size_t SMEM = gemm8_tma_smem<C>();
    static_assert(SMEM <= 99 * 1024, "shared memory");
    if (!gemm8_tma_device()) {
        return false;
    }
    // The kernel image the device runs must be built for sm_90 or newer: one JIT-compiled from
    // older PTX (a build without a 12.x target) traps.
    static std::atomic<int8_t> image_ok[GGML_CUDA_MAX_DEVICES];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    int8_t ok = image_ok[dev].load(std::memory_order_relaxed);
    if (ok == 0) {
        cudaFuncAttributes attr;
        const bool sm90 = cudaFuncGetAttributes(&attr, fp8_gemm8_tma<RES, C>) == cudaSuccess &&
                          attr.binaryVersion >= 90 && attr.ptxVersion >= 90;
        ok = sm90 ? 1 : -1;
        image_ok[dev].store(ok, std::memory_order_relaxed);
    }
    if (ok < 0) {
        return false;
    }
    a8t::Maps maps;
    if (!gemm8_tma_map(&maps.w, p.w, p.k, p.m, C::BK, C::TM) || !gemm8_tma_map(&maps.x, p.x, p.k, p.n, C::BK, a8::BN)) {
        return false;
    }
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, fp8_gemm8_tma<RES, C>, (int) SMEM)) {
        return false;
    }
    a8t::Stagger stagger;
    if constexpr (C::WARPS == 4) {
        // All of the SM's shared memory as shared memory, then whether two CTAs fit (once per
        // device).
        static std::atomic<int8_t> pair_ok[GGML_CUDA_MAX_DEVICES];
        int8_t two = pair_ok[dev].load(std::memory_order_relaxed);
        if (two == 0) {
            int per_sm = 0;
            const bool fits =
                cudaFuncSetAttribute(fp8_gemm8_tma<RES, C>, cudaFuncAttributePreferredSharedMemoryCarveout, 100) ==
                    cudaSuccess &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fp8_gemm8_tma<RES, C>, 32 * C::WARPS, SMEM) ==
                    cudaSuccess &&
                per_sm >= 2;
            two = fits ? 1 : -1;
            pair_ok[dev].store(two, std::memory_order_relaxed);
        }
        if (two < 0) {
            return false;
        }
        stagger.ctas = 2 * sm_count();
        stagger.ns   = (uint32_t) (GEMM8_STAGGER_NS_PER_K * p.k / stagger.ctas);
    }
    const dim3 grid((unsigned) ((p.n + a8::BN - 1) / a8::BN), (unsigned) ((p.m + C::TM - 1) / C::TM));
    fp8_gemm8_tma<RES, C><<<grid, 32 * C::WARPS, SMEM, st>>>(p, maps, stagger);
    return true;
}

// The large-batch GEMM: 256-row tiles (WM 4) where their rows divide m and they make at least
// GEMM8_WIDE_WAVES waves of CTAs (one per SM), else 128-row tiles: the wide tile copies fewer
// bytes per MMA, but with fewer tiles the last wave's idle SMs cost more (Qwen3.8-27B's k and v
// projections, 1,024 rows, stay at 128). The same bits either way.
// Stage layouts (#749, GB10 at 4,096 tokens, swizzled rows instead of padded ones): the wide
// tile takes two 128-byte K steps (96 KiB), half the barriers of three 64-byte steps, 1.14-1.23x
// faster (qkv 117 -> 136 TFLOPS, the fused-residual output 97 -> 114); where k is not a multiple
// of 128, four 64-byte steps (96 KiB, 1.04-1.11x). The 128-row tile keeps three 64-byte steps,
// now 48 KiB, so two CTAs share an SM without the residual (k/v: 96 -> 136 TFLOPS).
// On consumer Blackwell the wide tile with 128-byte steps runs as fp8_gemm8_tma (#1133, GB10,
// Qwen3.8-27B's projections at 4,096 to 16,384 tokens: 147-152 -> 156-163 TFLOPS, with the
// residual 120-153 -> 128-155; 4 stages of 64 bytes are no faster there, 3 slower). The same bits.
// With the residual, those shapes run as two 4-warp 128 x 128 CTAs per SM (three 64-byte TMA
// stages, 48 KiB each, #1133 r11), whose epilogues overlap the other CTA's MMAs: on the GB10, the
// output projection (6,144 -> 5,120) at 4,096 to 16,384 tokens 1.11-1.13x the wide tile's speed;
// without the residual they are no faster than it (0.99-1.02x), so it keeps the wide tile.
constexpr int64_t GEMM8_WIDE_WAVES = 8;
template <bool RES>
void launch_gemm8(const legion_fp8_mm8 & p, cudaStream_t st) {
    const int64_t wide_tiles = p.m / 256 * ((p.n + a8::BN - 1) / a8::BN);
    if (p.m % 256 == 0 && wide_tiles >= GEMM8_WIDE_WAVES * sm_count()) {
        if (p.k % 128 == 0) {
            if constexpr (RES) {
                if (launch_gemm8_tma<RES, a8::Cfg<2, 3, 64, true, 4>>(p, st)) {
                    return;
                }
            }
            if (!launch_gemm8_tma<RES, a8::Cfg<4, 2, 128, true>>(p, st)) {
                launch_gemm8_w<RES, a8::Cfg<4, 2, 128, true>>(p, st);
            }
        } else {
            launch_gemm8_w<RES, a8::Cfg<4, 4, 64, true>>(p, st);
        }
    } else {
        launch_gemm8_w<RES, a8::Cfg<2, 3, 64, true>>(p, st);
    }
}

// Whether the current device has E4M3 tensor cores (compute capability 8.9+).
bool fp8_mma() {
    return legion_device::compute_capability() >= 89;
}

template <int N>
void launch_gemv8(const legion_fp8_mm8 & p, cudaStream_t st) {
    fp8_gemv8<N><<<(unsigned) ((p.m + 3) / 4), 128, 0, st>>>(p);
}
}  // namespace

int legion_fp8_mm_launch(const legion_fp8_mm * p, void * stream) {
    if (p->k % 32 != 0 || p->x_ld % 4 != 0) {
        return 1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (p->x_bf16) {
        // The graph casts activations to BF16 only for this kernel (sm_80+, k % 64 == 0).
        if (p->k % big::BK != 0 || p->x_ld % 8 != 0 || !bf16_mma(st)) {
            return 1;
        }
        if (p->sliced_k > 0 && p->n >= SLICED_MIN_N && p->n <= SLICED_MAX_N && p->k % SLICED_K_ALIGN == 0) {
            const int tile = p->sliced_k == 1 ? sliced_tile(p->n, p->m, p->k) : (int) (p->sliced_k - 2);
            if (launch_sliced_tile(tile, *p, st)) {
                return cudaGetLastError() == cudaSuccess ? 0 : 2;
            }
        }
        launch_gemm_big(*p, st);
        return cudaGetLastError() == cudaSuccess ? 0 : 2;
    }
    const int64_t rows_max_n = p->gemv_max_n > 0 ? std::min<int64_t>(p->gemv_max_n, ROWS_MAX_N) : ROWS_DEFAULT_N;
    if (p->n >= ROWS_MIN_N && p->n <= rows_max_n && bf16_mma(st)) {
        launch_gemv_rows(*p, st);
    } else if (p->n <= GEMV_N) {
        switch (p->n) {
            case 1: launch_gemv<1>(*p, st); break;
            case 2: launch_gemv<2>(*p, st); break;
            case 3: launch_gemv<3>(*p, st); break;
            case 4: launch_gemv<4>(*p, st); break;
            case 5: launch_gemv<5>(*p, st); break;
            case 6: launch_gemv<6>(*p, st); break;
            case 7: launch_gemv<7>(*p, st); break;
            case 8: launch_gemv<8>(*p, st); break;
            default: return 1;
        }
    } else if (!bf16_mma(st)) {
        // Pre-Ampere: GEMV over chunks of 8 tokens.
        for (int64_t t0 = 0; t0 < p->n; t0 += GEMV_MAX_N) {
            legion_fp8_mm q = *p;
            q.x   = p->x + t0 * p->x_ld;
            q.dst = p->dst + t0 * p->m;
            q.n   = p->n - t0 < GEMV_MAX_N ? p->n - t0 : GEMV_MAX_N;
            if (q.n == GEMV_MAX_N) {
                launch_gemv<GEMV_MAX_N>(q, st);
            } else {
                // The tail: one launch per token.
                for (int64_t t = 0; t < q.n; t++) {
                    legion_fp8_mm r = q;
                    r.x   = q.x + t * q.x_ld;
                    r.dst = q.dst + t * q.m;
                    r.n   = 1;
                    launch_gemv<1>(r, st);
                }
            }
        }
    } else if (p->n <= 16) {
        if (p->k % 64 == 0) {
            launch_gemm<16, 64>(*p, st);
        } else {
            launch_gemm<16, 32>(*p, st);
        }
    } else if (p->n <= 32) {
        if (p->k % 64 == 0) {
            launch_gemm<32, 64>(*p, st);
        } else {
            launch_gemm<32, 32>(*p, st);
        }
    } else {
        launch_gemm_w<128, 32, 8>(*p, st);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_fp8_mm_bf16_gemm(void) {
    return bf16_mma(nullptr) ? 1 : 0;
}

int legion_fp8_quant_launch(const float * x, int64_t x_ld, uint8_t * out, int64_t k, int64_t n,
                            const float * scale, void * stream) {
    if (k % 2 != 0 || x_ld % 2 != 0) {
        return 1;
    }
    const int64_t pairs = k / 2 * n;
    const unsigned blocks = (unsigned) ((pairs + 255) / 256 < 4096 ? (pairs + 255) / 256 : 4096);
    fp8_quant<<<blocks, 256, 0, (cudaStream_t) stream>>>(x, x_ld, out, k, n, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_fp8_quant_dyn_launch(const float * x, int64_t x_ld, uint8_t * out, float * scales, int64_t k,
                                int64_t n, void * stream) {
    if (k % 4 != 0 || x_ld % 4 != 0) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    fp8_quant_dyn<<<(unsigned) n, QD_THREADS, 0, (cudaStream_t) stream>>>(x, x_ld, out, scales, k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_fp8_mm_fp8_gemm(void) {
    return fp8_mma() ? 1 : 0;
}

int legion_fp8_mm8_launch(const legion_fp8_mm8 * p, void * stream) {
    if (p->k % 64 != 0 || (p->res && (p->n <= 64 || !fp8_mma()))) {
        return 1;  // (the residual: the large-batch GEMM only)
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (p->n <= GEMV_MAX_N || !fp8_mma()) {
        // Pre-Ada GPUs: the GEMV over chunks of 8 tokens.
        for (int64_t t0 = 0; t0 < p->n; t0 += GEMV_MAX_N) {
            legion_fp8_mm8 q = *p;
            q.x     = p->x + t0 * p->k;
            q.dst   = p->dst + t0 * p->m;
            q.scale = p->scale + t0 * p->scale_stride;
            q.n   = p->n - t0 < GEMV_MAX_N ? p->n - t0 : GEMV_MAX_N;
            switch (q.n) {
                case 1: launch_gemv8<1>(q, st); break;
                case 2: launch_gemv8<2>(q, st); break;
                case 3: launch_gemv8<3>(q, st); break;
                case 4: launch_gemv8<4>(q, st); break;
                case 5: launch_gemv8<5>(q, st); break;
                case 6: launch_gemv8<6>(q, st); break;
                case 7: launch_gemv8<7>(q, st); break;
                default: launch_gemv8<8>(q, st); break;
            }
        }
    } else if (p->n <= 16 && p->k % a8s::BK == 0) {
        launch_gemm8_skinny<16>(*p, st);
    } else if (p->n <= 32 && p->k % a8s::BK == 0) {
        launch_gemm8_skinny<32>(*p, st);
    } else if (p->n <= 64 && p->k % a8s::BK == 0) {
        launch_gemm8_skinny<64>(*p, st);
    } else if (gemm8_ada_skinny(*p)) {
        if (p->res) {
            launch_gemm8_skinny<64, true, true>(*p, st);
        } else {
            launch_gemm8_skinny<64, false, true>(*p, st);
        }
    } else if (p->res) {
        launch_gemm8<true>(*p, st);
    } else {
        launch_gemm8<false>(*p, st);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
