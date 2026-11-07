// W4A4 NVFP4 matmuls on the block-scaled FP4 tensor cores of sm_120a / sm_121a (GB10, RTX 50):
// `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` takes E2M1 operands with
// one UE4M3 scale per 16 elements of k on both sides and sums in F32, ~4.2x the BF16 rate of
// the W4A16 kernels (497 vs 118 TFLOPS measured on the GB10). Two kernels:
// - `nvfp4_a4_quant`: activations (F32 or BF16 rows) to NVFP4: one F32 decode scale per row
//   (amax / (6 * 448), so the row's largest sub-block scale is 448), one UE4M3 scale per 16
//   elements (round to nearest even of amax16 / 6 in the row's scale units), E2M1 codes (round
//   to nearest even, saturating), packed as ggml packs NVFP4 weights;
// - `nvfp4_a4_gemm`: the weights (ggml NVFP4 rows, as loaded) times those activations, dense,
//   grouped by expert (mul_mat_id) or as the fused SwiGLU gate/up pair (dense or grouped), times
//   the weight's per-tensor or per-expert scale and the token's decode scale in the epilogue.
// Built only for the architecture-specific targets (CMAKE_CUDA_ARCHITECTURES 120a / 121a): the
// FP4 MMA and the E2M1 conversion do not exist elsewhere. Every other architecture compiles the
// kernels empty and `legion_nvfp4_a4_supported` reports 0, so the W4A16 kernels run instead.
//
// Operand layout (PTX ISA, "Matrix fragments for mma.m16n8k64", g = lane / 4, t = lane % 4):
// A a0 = (row g, k 8t..8t+7), a1 = (g + 8, 8t..), a2 = (g, 32 + 8t..), a3 = (g + 8, 32 + 8t..),
// eight E2M1 codes per register; B b0 = (k 8t.., col g), b1 = (k 32 + 8t.., col g); the scale
// registers hold a row's (column's) four UE4M3 sub-block scales, byte i for k 16i..16i+15, and
// with thread-id 0 lanes t = 0 / 1 give rows g / g + 8 of A, lane t = 0 column g of B (as
// ggml's mmq.cuh, MIT). A ggml NVFP4 block is four scale bytes, then four sub-blocks of 8 code
// bytes (element j low, j + 8 high nibble of byte j): word i of the 32 code bytes is k 8i..8i+7
// of the MMA's k order, which permutes k inside each sub-block only, the same way for the
// weights and the quantized activations; the sum does not depend on the order. So a block's
// scale word is the scale register as stored, and ldmatrix reads the code words straight (RAW
// tiles read the weights' a word at a time from the blocks as stored).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define LEGION_FP4_MMA 1
#endif

extern "C" {
struct legion_nvfp4_a4 {
    const uint8_t * w;    // NVFP4 weight rows of w_ld bytes; experts w_e bytes apart
    const uint8_t * xq;   // quantized activations: E2M1 codes, rows of xq_ld bytes
    const uint8_t * xs;   // their UE4M3 sub-block scales, rows of xs_ld bytes
    const float * xd;     // their per-row decode scales
    const float * scale;  // per-tensor (dense) or per-expert (grouped) weight scale, or null
    void * dst;           // F32 (or BF16, see act) rows of m
    int64_t k, m, n, xq_ld, xs_ld, w_ld, w_e;
    int32_t tiles_m, tiles_n;  // row tiles and token tiles (grouped: an upper bound)
    int32_t group_m;           // dense tile order: groups of group_m row tiles
    // Grouped (mul_mat_id): plan as nvfp4_mm/common.cuh's (expert offsets, then the pairs token *
    // n_used + slot grouped by expert); the activations have n_x (1 or n_used) rows per token.
    const int32_t * plan;
    int32_t n_expert, n_used, n_x;
    // Fused SwiGLU: w the gate, w2 the up weight (grouped: experts w_e bytes apart too), scale2
    // its scale (per expert when grouped); act 0 stores F32, any other BF16. Dense GLU, act 3:
    // the tanh GELU of the gate (a GeGLU FFN, Gemma 4) instead of silu, BF16 out.
    const uint8_t * w2;
    const float * scale2;
    int32_t act;  // 0: F32 out; 1: relu(v)^2 as BF16; 2: BF16; 3: GeGLU, BF16
    // Grouped, act 2: each (token, slot) pair's routing weight (F32 [n_tok * n_used]), applied
    // in the epilogue (see legion_nvfp4_a4_combine_launch).
    const float * pw;
    // Dense, act 0: a residual (F32, the dst's layout) added to each output with __fadd_rn (never
    // contracted into an FMA), so dst holds the bits of the plain GEMM followed by ggml's add.
    const float * res;
};

// Quantizes `rows` rows of k (F32 if x_bf16 is 0, else BF16; x_ld elements apart) to the layout
// above: codes rows of q_ld bytes, scales rows of s_ld bytes, decode scales d. Blocks of 64
// between k and the row strides are written as zeros. Needs k % 64 == 0, 16-byte aligned rows.
int legion_nvfp4_a4_quant_launch(const void * x, int x_bf16, int64_t x_ld, int64_t k, int64_t rows, uint8_t * q,
                                 int64_t q_ld, uint8_t * s, int64_t s_ld, float * d, void * stream);

// legion_nvfp4_a4_quant_launch of mul(rms_norm(x, eps), w) (x F32 rows of k, x_ld elements
// apart; w F32 [k]) without storing the normed rows: each row's norm computed as ggml-cuda's
// rms_norm_f32 with its fused mul computes it (the same bits), then quantized as above. 1 for a
// k it does not take (k % 64 != 0, or more than LEGION_NVFP4_A4_RMS_MAX_K), 2 for a launch error.
#define LEGION_NVFP4_A4_RMS_MAX_K 8192
int legion_nvfp4_a4_rms_quant_launch(const float * x, int64_t x_ld, const float * w, float eps, int64_t k,
                                     int64_t rows, uint8_t * q, int64_t q_ld, uint8_t * s, int64_t s_ld, float * d,
                                     void * stream);

// The GEMM: tiles_m * tiles_n CTAs of 256 threads, 8 warps of 32 rows x 8 * nt tokens laid
// out 4 (rows) x 2 (tokens): 128 rows (64 output rows for SwiGLU) x 16 * nt tokens per CTA;
// nt = 2, 4 or 8, or LEGION_NVFP4_A4_TWO_CTAS | 4 (grouped only: the same tokens, sized for two
// CTAs per SM). Grouped when p->plan is set, SwiGLU when p->w2 is (both: the grouped SwiGLU).
// -1 for a missing variant.
// Needs k % 64 == 0; activation rows padded to whole stages (see the quantizer).
#define LEGION_NVFP4_A4_TWO_CTAS 0x100
int legion_nvfp4_a4_gemm_launch(const legion_nvfp4_a4 * p, int nt, void * stream);

// The routed experts' weighted sum: out (F32 rows of m, one per token) = the sum over the
// n_used slots, in order, of y (BF16 rows of m, n_used per token: a grouped GEMM's act 2
// output, already weighted).
int legion_nvfp4_a4_combine_launch(const __nv_bfloat16 * y, int64_t m, int32_t n_used, int64_t n_tok, float * out,
                                   void * stream);

// The grouped decode kernel (p->plan set; few tokens per expert): grid (tiles_m, tiles_n),
// kw * 32 threads; each CTA covers 16 * rt weight rows (SwiGLU, p->w2 set: 8 * rt output rows,
// rt / 2 row tiles of the gate and the same rows of the up weight) and one expert's tile of up
// to 8 * nt tokens. nt = 1, 2 or 4; rt = 1 or 2 (SwiGLU: 2 or 4); kw = 2, 4 or 8. Every act of
// the GEMM. Needs k % 64 == 0 and the weight rows, experts and bases 8-byte aligned; -1 for a
// missing variant or unaligned weights (use the GEMM then).
int legion_nvfp4_a4_mv_launch(const legion_nvfp4_a4 * p, int nt, int rt, int kw, void * stream);

// 1 when this build has the FP4 kernels for the current device, else 0.
int legion_nvfp4_a4_supported(void);
}

namespace {
// A GEMM tile shape: 8 warps, 4 over the weight rows x 2 over the tokens, each 16 * MI rows x
// 8 * NT tokens; KB NVFP4 blocks (64 of k) per pipeline stage, ST stages, MINB CTAs per SM.
// A stage: the weight rows, the activation codes, then the scale words. RAW: the weight rows
// hold their KB blocks as stored (scale word, then codes), else codes only and the scale words
// apart (smaller in shared memory with RAW: 128 x 80 against 128 x 88 bytes at KB 2).
template <int MI_, int NT_, int KB_, int ST_, int MINB_ = 1, bool RAW_ = false> struct Tile {
    static constexpr int MI = MI_, NT = NT_, KB = KB_, ST = ST_, MINB = MINB_, WM = 4, WN = 2;
    static constexpr bool RAW = RAW_;
    static constexpr int BM = 16 * MI * WM, BN = 8 * NT * WN;
    // Smem bytes per row of codes: 32 per block plus 16, so ldmatrix's eight 16-byte rows
    // fall in distinct bank groups.
    static constexpr int LDC = KB * 32 + 16;
    // RAW: smem bytes per weight row, its KB blocks as stored rounded up to 16: the eight rows
    // g of a fragment's 32-bit reads (row stride LDW / 4 words) and its four lanes t land in 32
    // distinct banks.
    static constexpr int LDW = (KB * 36 + 15) / 16 * 16;
    static constexpr int A_BYTES = BM * (RAW ? LDW : LDC);
    static constexpr int A_SCALES = A_BYTES + BN * LDC;  // !RAW: the weight rows' scale words
    static constexpr int B_SCALES = A_SCALES + (RAW ? 0 : BM * KB * 4);
    static constexpr int STAGE = B_SCALES + BN * KB * 4;
    static constexpr int SMEM = ST * STAGE;
    static_assert(SMEM <= 96 * 1024, "shared memory");
};

#ifdef LEGION_FP4_MMA
// The tanh GELU of the GeGLU epilogue (act 3), as ggml-cuda's (ggml_cuda_op_gelu_single, MIT).
__device__ __forceinline__ float a4_gelu(float x) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ void cp_async4(void * s, const void * g, bool valid) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(a), "l"(g), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void cp_async8(void * s, const void * g, bool valid) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(a), "l"(g), "r"(valid ? 8 : 0));
}

__device__ __forceinline__ void cp_async16(void * s, const void * g, bool valid) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(valid ? 16 : 0));
}

// 16 bytes, of which the first n (0..16) are read and the rest zero-filled.
__device__ __forceinline__ void cp_async16n(void * s, const void * g, int n) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(n));
}

__device__ __forceinline__ void mma_fp4(float * c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// Four bytes of E2M1 pairs: byte i = (lo[i] in the low, hi[i] in the high nibble), each rounded
// to nearest even and saturated to +-6.
__device__ __forceinline__ uint32_t e2m1x8(const float * lo, const float * hi) {
    uint32_t r;
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(r)
        : "f"(lo[0]), "f"(hi[0]), "f"(lo[1]), "f"(hi[1]), "f"(lo[2]), "f"(hi[2]), "f"(lo[3]), "f"(hi[3]));
    return r;
}
#endif

// Grouped tiles: token tile `tile` of `bn` pairs is in expert e when it is among the
// ceil(count_e / bn) tiles that follow the earlier experts' tiles (as nvfp4_mm/common.cuh's find_tile,
// the same search): lane l counts the tiles of experts [l * per, (l + 1) * per) with
// independent loads of the plan's offsets, one warp scan of the 32 counts, then the lane
// holding `tile` walks its experts again (cached). Returns false past the last tile. (A running
// scan of 32 experts at a time costs n_expert / 32 dependent rounds per CTA: with 512 experts
// and short k, a large part of a tile's time.)
__device__ __forceinline__ bool find_tile(const int32_t * offs, int n_expert, int bn, int tile, int lane,
                                          int & expert, int & first, int & count) {
    const int per = (n_expert + 31) / 32;
    const int e0 = min(lane * per, n_expert), e1 = min(e0 + per, n_expert);
    int t = 0;
#pragma unroll 8
    for (int e = e0; e < e1; e++) {
        t += (offs[e + 1] - offs[e] + bn - 1) / bn;
    }
    int incl = t;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        incl += lane >= o ? v : 0;
    }
    const int lo = incl - t;
    const unsigned hit = __ballot_sync(0xffffffffu, tile >= lo && tile < incl);
    if (hit == 0) {
        return false;
    }
    const int src = __ffs(hit) - 1;
    int found = 0, f = 0, c = 0;
    if (lane == src) {
        int before = lo;
        for (int e = e0; e < e1; e++) {
            const int a = offs[e], b = offs[e + 1];
            const int te = (b - a + bn - 1) / bn;
            if (tile < before + te) {
                found = e;
                f = a + (tile - before) * bn;
                c = min(bn, b - f);
                break;
            }
            before += te;
        }
    }
    expert = __shfl_sync(0xffffffffu, found, src);
    first = __shfl_sync(0xffffffffu, f, src);
    count = __shfl_sync(0xffffffffu, c, src);
    return true;
}

template <typename T> __device__ __forceinline__ void load8(const T * p, float * v);
template <> __device__ __forceinline__ void load8<float>(const float * p, float * v) {
    const float4 a = *(const float4 *) p, b = *(const float4 *) (p + 4);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16 * p, float * v) {
    const uint4 a = *(const uint4 *) p;
    const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

__device__ __forceinline__ float e4m3_value(uint8_t c) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
}

#ifdef LEGION_FP4_MMA
// |a| and |b| rounded to E2M1 magnitudes (nearest, ties to the even code, saturated at 6), by
// the conversion e2m1x8 codes with, and back: |a| in the low half. The magnitudes are exact in
// F16. (Was a chain of 7 compares per value, the same values: the midpoints 0.25 .. 5 are the
// ties, and 0, 1, 2 and 4 are the even codes.)
__device__ __forceinline__ void e2m1_magnitude2(float a, float b, float & ma, float & mb) {
    uint32_t h;
    asm("{\n.reg .b8 q;\n"
        "cvt.rn.satfinite.e2m1x2.f32 q, %2, %1;\n"
        "cvt.rn.f16x2.e2m1x2 %0, q;\n}"
        : "=r"(h)
        : "f"(a), "f"(b));
    h &= 0x7fff7fffu;
    ma = __half2float(__ushort_as_half((unsigned short) (h & 0xffffu)));
    mb = __half2float(__ushort_as_half((unsigned short) (h >> 16)));
}

// The squared error of a sub-block (elements lo[j], hi[j] = j, j + 8) quantized with scale code
// c, in the row's scale units, summed in the order j, j + 8 for j = 0..7.
__device__ __forceinline__ float scale_error(const float * lo, const float * hi, float enc, uint8_t c) {
    const float sf = e4m3_value(c);
    const float qs = sf > 0.0f ? __fdiv_rn(enc, sf) : 0.0f;
    float err = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        float ml, mh;
        e2m1_magnitude2(__fmul_rn(lo[j], qs), __fmul_rn(hi[j], qs), ml, mh);
        const float dl = __fsub_rn(__fmul_rn(fabsf(lo[j]), enc), __fmul_rn(ml, sf));
        err = __fadd_rn(err, __fmul_rn(dl, dl));
        const float dh = __fsub_rn(__fmul_rn(fabsf(hi[j]), enc), __fmul_rn(mh, sf));
        err = __fadd_rn(err, __fmul_rn(dh, dh));
    }
    return err;
}
#endif

#ifdef LEGION_FP4_MMA
// Sub-blocks [tid, tid + step, ...) of a row (xr, k values; the row's amax known) quantized:
// codes, scales and (thread 0) the decode scale; sub-blocks past k up to s_ld written as zeros.
template <typename T>
__device__ __forceinline__ void quant_row(const T * xr, float amax, int64_t k, int64_t row, uint8_t * q,
                                          int64_t q_ld, uint8_t * s, int64_t s_ld, float * d, int tid, int step) {
    // Codes are x * enc / scale; a sub-block's scale is its amax * 448 / row amax.
    const bool live = amax > 0.0f;
    const float enc = live ? __fdiv_rn(2688.0f, amax) : 0.0f;
    const float enc16 = live ? __fdiv_rn(448.0f, amax) : 0.0f;
    if (tid == 0) {
        d[row] = __fdiv_rn(amax, 2688.0f);
    }
    const int64_t nsb = k / 16;
    for (int64_t sb = tid; sb < s_ld; sb += step) {
        uint32_t c0 = 0, c1 = 0;
        uint8_t sc = 0;
        if (sb < nsb) {
            float lo[8], hi[8];
            load8(xr + sb * 16, lo);
            load8(xr + sb * 16 + 8, hi);
            float bmax = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                bmax = fmaxf(bmax, fmaxf(fabsf(lo[j]), fabsf(hi[j])));
            }
            // The scale rounded to nearest, then the two codes either side if one gives a
            // smaller squared error (as ggml's quantize_mmq_nvfp4, MIT).
            const int first = __nv_cvt_float_to_fp8(__fmul_rn(bmax, enc16), __NV_SATFINITE, __NV_E4M3);
            sc = (uint8_t) first;
            float best = scale_error(lo, hi, enc, sc);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int c = first + (i % 2 ? 1 : -1) * (i / 2 + 1);
                if (c >= 0 && c <= 0x7e) {
                    const float e = scale_error(lo, hi, enc, (uint8_t) c);
                    if (e < best) {
                        best = e;
                        sc = (uint8_t) c;
                    }
                }
            }
            const float sf = e4m3_value(sc);
            const float qs = sf > 0.0f ? __fdiv_rn(enc, sf) : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                lo[j] = __fmul_rn(lo[j], qs);
                hi[j] = __fmul_rn(hi[j], qs);
            }
            c0 = e2m1x8(lo, hi);
            c1 = e2m1x8(lo + 4, hi + 4);
        }
        s[row * s_ld + sb] = sc;
        *(uint2 *) (q + row * q_ld + sb * 8) = make_uint2(c0, c1);
    }
}

// ggml-cuda's warp_reduce_sum (common.cuh, MIT): butterfly over the 32 lanes.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}
#endif

// One CTA of 128 threads per row: the row's amax, then one 16-element sub-block per thread.
// The scale arithmetic is IEEE (__fdiv_rn / __fmul_rn, not affected by --use_fast_math), so
// the CPU reference (src/model/nvfp4_a4.rs) reproduces every bit.
template <typename T>
__global__ void __launch_bounds__(128) nvfp4_a4_quant(const T * x, int64_t x_ld, int64_t k, uint8_t * q, int64_t q_ld,
                                                      uint8_t * s, int64_t s_ld, float * d) {
#ifdef LEGION_FP4_MMA
    const int64_t row = blockIdx.x;
    const T * xr = x + row * x_ld;
    __shared__ float red[4];
    float amax = 0.0f;
    for (int64_t i = threadIdx.x * 8; i < k; i += 128 * 8) {
        float v[8];
        load8(xr + i, v);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            amax = fmaxf(amax, fabsf(v[j]));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    quant_row(xr, amax, k, row, q, q_ld, s, s_ld, d, threadIdx.x, 128);
#endif
}

// nvfp4_a4_quant<float> of y = mul(rms_norm(x, eps), w), one CTA of RQ_THREADS threads per row.
// y has the bits of ggml-cuda's rms_norm_f32<BLOCK, do_multiply> (norm.cu, MIT; BLOCK 1024
// from 1024 columns on, 256 below, as its launcher picks): its thread tid sums the squares of
// columns tid, tid + BLOCK, ... in that order, block_reduce (warp sums, then warp 0's sum of
// them), mean tmp / k, scale rsqrtf(mean + eps), y = scale * x * w, under the same
// --use_fast_math as ggml-cuda (as fp8_act.cu's rms_norm kind). Here each thread plays BLOCK /
// RQ_THREADS of those threads: lane l of warp wp plays thread l + 32 * (wp + RQ_WARPS * j), so
// its warp of the BLOCK-thread block is wp + RQ_WARPS * j and the butterfly of value j over
// the physical warp is that warp's (fewer, fuller threads for the quantization below). The row
// of y stays in shared memory (k floats) between the norm and the quantization, which is
// quant_row's on it: the same codes, scales and decode scale as the two kernels it replaces.
constexpr int RQ_THREADS = 256, RQ_WARPS = RQ_THREADS / 32;

template <int BLOCK>
__global__ void __launch_bounds__(RQ_THREADS) nvfp4_a4_rms_quant(const float * x, int64_t x_ld, const float * w,
                                                                 float eps, int k, uint8_t * q, int64_t q_ld,
                                                                 uint8_t * s, int64_t s_ld, float * d) {
#ifdef LEGION_FP4_MMA
    constexpr int V = BLOCK / RQ_THREADS;
    static_assert(V * RQ_THREADS == BLOCK, "block");
    extern __shared__ __align__(16) float y[];
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const float * xr = x + row * x_ld;
    const int lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    float part[V];
#pragma unroll
    for (int j = 0; j < V; j++) {
        const int tid = lane + 32 * (wp + RQ_WARPS * j);
        float tmp = 0.0f;
        for (int col = tid; col < k; col += BLOCK) {
            const float xi = xr[col];
            tmp += xi * xi;
        }
        part[j] = warp_sum(tmp);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < V; j++) {
            red[wp + RQ_WARPS * j] = part[j];
        }
    }
    __syncthreads();
    float tmp = lane < BLOCK / 32 ? red[lane] : 0.0f;
    tmp = warp_sum(tmp);
    const float mean = tmp / k;
    const float scale = rsqrtf(mean + eps);
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < V; j++) {
        const int tid = lane + 32 * (wp + RQ_WARPS * j);
        for (int col = tid; col < k; col += BLOCK) {
            const float v = scale * xr[col] * w[col];
            y[col] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    __syncthreads();  // every warp has read red (the sum) before it is reused
    if (lane == 0) {
        red[wp] = amax;
    }
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int i = 1; i < RQ_WARPS; i++) {
        amax = fmaxf(amax, red[i]);
    }
    quant_row(y, amax, k, row, q, q_ld, s, s_ld, d, threadIdx.x, RQ_THREADS);
#endif
}

// Tile T: T::BM weight rows x T::BN tokens per CTA (SwiGLU: T::BM / 2 output rows, the first
// half of each warp's 16-row tiles from the gate, the second half the same rows of the up
// weight). k in stages of KB blocks through ST cp.async buffers. Weights copied CW bytes at a
// time (ggml's 36-byte blocks are only 4-byte aligned): T::RAW, a row's KB blocks as stored
// (CW 8 when the rows, experts and the stage's 36 * KB bytes are 8-byte aligned) and the A
// fragments read a word at a time from them; else words, scale words and code words to
// separate arrays, the A fragments read with ldmatrix. Activations 16 bytes at a time, read
// with ldmatrix. A stage's fragments for block b + 1 are read while block b's MMAs run. GROUPED: the
// tile's expert and tokens come from the plan. GROUPED: a warp skips the fragments and MMAs of its n8
// token tiles past the tile's last token (grouped tiles are often partly filled): the other
// tokens' sums are the same instructions on the same values.
// RES: p.res added to the dense F32 outputs (the residual add fused into the epilogue).
// SH (CW 16): weight rows only 8-byte aligned (w_ld % 16 == 8, bases 16-byte aligned): each
// row's copies take the aligned window around its stage, which starts 0 or 8 bytes in by the
// row's parity (a lane's rows all have its g's parity).
#ifdef LEGION_FP4_MMA
template <int MI, int NT, bool GROUPED, bool GLU, bool RES>
__device__ __forceinline__ void a4_epilogue(const legion_nvfp4_a4 & p, const float (&acc)[MI][NT][4], int64_t m0,
                                            int64_t n0, int n_tok, const float * x_dec, const int64_t * d_off,
                                            float sc, float sc2, int wm, int wn, int g, int t);
#endif
template <class T, bool GROUPED, bool GLU, int CW, bool RES = false, bool SH = false>
__global__ void __launch_bounds__(256, T::MINB) nvfp4_a4_gemm(const legion_nvfp4_a4 p) {
#ifdef LEGION_FP4_MMA
    constexpr int MI = T::MI, NT = T::NT, KB = T::KB, ST = T::ST, BM = T::BM, BN = T::BN;
    constexpr int LDC = T::LDC, LDW = T::LDW, A_BYTES = T::A_BYTES, STAGE = T::STAGE, WN = T::WN;
    constexpr int A_SCALES = T::A_SCALES, B_SCALES = T::B_SCALES;
    constexpr bool RAW = T::RAW;
    static_assert(CW == 4 || (CW == 8 && RAW && KB % 2 == 0) || (CW == 16 && RAW && KB % 2 == 0), "copy width");
    constexpr int BM_OUT = GLU ? BM / 2 : BM;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int64_t x_row[BN], d_off[GROUPED ? BN : 1];
    __shared__ float x_dec[BN];
    __shared__ int32_t tile_expert, tile_tokens;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / WN, wn = warp % WN;

    int64_t m0, n0 = 0;
    int n_tok;
    const uint8_t * w = p.w;
    const uint8_t * w2 = p.w2;
    float sc = p.scale ? *p.scale : 1.0f;
    float sc2 = p.scale2 ? *p.scale2 : 1.0f;
    if (!GROUPED) {
        // Groups of group_m row tiles, token tiles inside a group: the CTAs resident at once
        // share weight rows and activations through L2.
        const int pid = blockIdx.x, in_group = p.group_m * p.tiles_n;
        const int first_m = pid / in_group * p.group_m;
        const int group = min(p.tiles_m - first_m, p.group_m);
        m0 = (int64_t) (first_m + pid % in_group % group) * BM_OUT;
        n0 = (int64_t) (pid % in_group / group) * BN;
        n_tok = (int) min((int64_t) BN, p.n - n0);
        for (int i = threadIdx.x; i < BN; i += 256) {
            x_row[i] = i < n_tok ? n0 + i : 0;
            x_dec[i] = i < n_tok ? p.xd[n0 + i] : 0.0f;
        }
    } else {
        // Row tiles vary fastest: the CTAs resident at once share an expert's tokens.
        m0 = (int64_t) (blockIdx.x % p.tiles_m) * BM_OUT;
        const int tile = blockIdx.x / p.tiles_m;
        if (warp == 0) {
            int e, first, count;
            if (!find_tile(p.plan, p.n_expert, BN, tile, lane, e, first, count)) {
                if (lane == 0) {
                    tile_expert = -1;
                }
            } else {
                if (lane == 0) {
                    tile_expert = e;
                    tile_tokens = count;
                }
                for (int i = lane; i < BN; i += 32) {
                    if (i < count) {
                        const int q = p.plan[p.n_expert + 1 + first + i];  // token * n_used + slot
                        const int tk = q / p.n_used, slot = q % p.n_used;
                        const int64_t xr = (int64_t) tk * p.n_x + (p.n_x == 1 ? 0 : slot);
                        x_row[i] = xr;
                        x_dec[i] = p.pw ? p.xd[xr] * p.pw[q] : p.xd[xr];
                        d_off[i] = (int64_t) q * p.m;
                    } else {
                        x_row[i] = 0;
                        x_dec[i] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();
        if (tile_expert < 0) {
            return;
        }
        w += (int64_t) tile_expert * p.w_e;
        sc = p.scale ? p.scale[tile_expert] : 1.0f;
        if (GLU) {
            w2 += (int64_t) tile_expert * p.w_e;
            sc2 = p.scale2 ? p.scale2[tile_expert] : 1.0f;
        }
        n_tok = tile_tokens;
    }
    __syncthreads();
    const int64_t nblk = p.k / 64, steps = (nblk + KB - 1) / KB;

    // Weight row r of the tile (SwiGLU: the gate's or the up weight's).
    auto w_row = [&](int r, int64_t & row) {
        if (GLU) {
            const int in = r % (16 * MI);
            row = m0 + r / (16 * MI) * (8 * MI) + in % (8 * MI);
            return in >= 8 * MI ? w2 : w;
        }
        row = m0 + r;
        return w;
    };
    // Each thread's copies of a stage, the same every stage but for the k offset: computed
    // once (the address arithmetic per copy otherwise costs ~10% of the GEMM's time).
    // CW-byte weight copies; a copy of 8 may cross into the next block, which then is in the
    // stage too (CW 8 needs nblk % KB == 0).
    // CW 16: a row's stage (KB * 36 bytes) starts 0 or 8 bytes into a 16-byte word (KB even,
    // 16-byte aligned rows); the copies take the aligned window around it, W16 words (zero past
    // the row's end), and the fragments read it WSH bytes in.
    constexpr int W16 = (KB * 36) % 16 == 0 ? KB * 36 / 16 : (KB * 36 + 8 + 15) / 16;
    static_assert(CW != 16 || W16 * 16 <= LDW, "window");
    static_assert(!SH || (CW == 16 && (KB * 36) % 16 != 0), "SH: a window with room for the 8-byte start");
    constexpr int PER_ROW = CW == 16 ? W16 : KB * 36 / CW, AW = BM * PER_ROW, NA = (AW + 255) / 256;  // weight copies
    const int64_t row_bytes = nblk * 36;
    constexpr int BW = BN * KB * 2, NB = (BW + 255) / 256;  // activation 16-byte chunks
    const uint8_t * w_src[NA];
    uint32_t w_dst[NA], w_ok = 0, w_sh = 0;  // bit i: copy i's row exists / starts 8 bytes into a word
    int w_blk[NA];
#pragma unroll
    for (int i = 0; i < NA; i++) {
        const int c = threadIdx.x + i * 256;
        w_src[i] = w;
        w_dst[i] = 0;
        w_blk[i] = 0;
        if (AW % 256 == 0 || c < AW) {
            const int r = c / PER_ROW, wd = c % PER_ROW;
            int64_t row;
            const uint8_t * wt = w_row(r, row);
            if (row < p.m) {
                w_ok |= 1u << i;
                w_src[i] = wt + row * p.w_ld + wd * CW;
                if (SH && ((uintptr_t) (wt + row * p.w_ld) & 15)) {
                    w_sh |= 1u << i;
                    w_src[i] -= 8;
                }
            }
            w_blk[i] = CW == 16 ? wd * 16 : wd * CW / 36;  // CW 16: the copy's offset in the window
            if (RAW) {
                w_dst[i] = r * LDW + wd * CW;
            } else {
                // Scale words to their array, code words to the row's codes.
                const int b = wd / 9, wb = wd % 9;
                w_dst[i] = wb == 0 ? A_SCALES + (r * KB + b) * 4 : r * LDC + b * 32 + (wb - 1) * 4;
            }
        }
    }
    const uint8_t * x_src[NB];
    uint32_t x_dst[NB], x_ok = 0;
#pragma unroll
    for (int i = 0; i < NB; i++) {
        const int c = threadIdx.x + i * 256;
        x_src[i] = p.xq;
        x_dst[i] = 0;
        if (BW % 256 == 0 || c < BW) {
            const int tk = c / (KB * 2), ch = c % (KB * 2);
            if (tk < n_tok) {
                x_ok |= 1u << i;
                x_src[i] = p.xq + x_row[tk] * p.xq_ld + ch * 16;
            }
            x_dst[i] = A_BYTES + tk * LDC + ch * 16;
        }
    }
    auto load = [&](int64_t step, int stage) {
        unsigned char * base = smem + stage * STAGE;
        const int64_t b0 = step * KB;
        const int64_t a16 = b0 * 36 & ~(int64_t) 15;
#pragma unroll
        for (int i = 0; i < NA; i++) {
            if (AW % 256 == 0 || threadIdx.x + i * 256 < AW) {
                if (CW == 16) {
                    // SH: offsets from the aligned word before the row's start, 8 bytes ahead.
                    const int sh = SH && (w_sh >> i & 1) ? 8 : 0;
                    const int64_t a = SH ? (b0 * 36 + sh) & ~(int64_t) 15 : a16;
                    const int64_t off = a + w_blk[i];
                    const int nb =
                        (w_ok >> i & 1) ? (int) max((int64_t) 0, min((int64_t) 16, row_bytes + sh - off)) : 0;
                    cp_async16n(base + w_dst[i], nb > 0 ? w_src[i] + a : w, nb);
                    continue;
                }
                const bool ok = (w_ok >> i & 1) && b0 + w_blk[i] < nblk;
                if (CW == 8) {
                    cp_async8(base + w_dst[i], ok ? w_src[i] + b0 * 36 : w, ok);
                } else {
                    cp_async4(base + w_dst[i], ok ? w_src[i] + b0 * 36 : w, ok);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NB; i++) {
            if (BW % 256 == 0 || threadIdx.x + i * 256 < BW) {
                const bool ok = x_ok >> i & 1;
                cp_async16(base + x_dst[i], ok ? x_src[i] + b0 * 32 : p.xq, ok);
            }
        }
        uint32_t * bs = (uint32_t *) (base + B_SCALES);
        for (int tk = threadIdx.x; tk < BN; tk += 256) {
            const bool ok = tk < n_tok;
            const uint8_t * src = p.xs + (ok ? x_row[tk] * p.xs_ld + b0 * 4 : 0);
            if (KB == 1) {
                cp_async4(bs + tk, src, ok);
            } else if (KB == 2) {
                cp_async8(bs + tk * 2, src, ok);
            } else {
#pragma unroll
                for (int q = 0; q < KB; q += 4) {
                    cp_async16(bs + tk * KB + q, src + q * 4, ok);
                }
            }
        }
    };

    // Each weight row's next blocks in one L2 bulk prefetch, PF blocks ahead of the copies
    // (the copies read KB * 36 bytes of a row per stage, too short a run for the LPDDR5X).
    // The two-CTA SwiGLU prefetches 8 blocks at a time, at most 16 ahead: 32 put a CTA's whole
    // k-loop in flight at its start (184 KiB, ~18 MiB over the 96 resident CTAs against the
    // GB10's 24 MiB L2), and the stream ran at ~170-200 GB/s instead of 210-216 (#749, GB10,
    // Qwen3.8-Flash-Next's experts at k 2560: 7-10% faster at 512 to 4096 tokens; 4 blocks per
    // prefetch are too short a run, 6 as fast as 8, 10 or 12 up to 4% slower). The short-k
    // two-CTA GEMMs fetch their whole rows either way.
    // The dense 16-byte-copy tile prefetches 8 blocks at a time too, and only from the first two
    // token tiles of each group of row tiles: the other token tiles of the group run beside or
    // after them and find the rows in L2, and their duplicate prefetches only cost (#749, GB10,
    // Qwen3.8-27B's MLP: SwiGLU k 5120 5% faster at 4096 tokens, 10-23% at 256 to 2048; down
    // k 17408 1% at 4096, 3-8% at 256 to 2048; no prefetch at all is up to 13% slower at 96
    // tokens and 6% on the down projection at 4096; 4 blocks per run too short).
    constexpr bool DENSE16 = !GROUPED && CW == 16;
    constexpr int PF = ((T::MINB == 2 && GLU) || DENSE16 ? 8 : 32) / KB;
    const bool pf_on = !DENSE16 || n0 < 2 * BN;
    auto prefetch = [&](int64_t step) {
        if (!pf_on) {
            return;
        }
        for (int r = threadIdx.x; r < BM; r += 256) {
            if (step >= steps) {
                break;
            }
            int64_t row;
            const uint8_t * wt = w_row(r, row);
            if (row < p.m) {
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + step * KB * 36);
                const uintptr_t e = (uintptr_t) (wt + row * p.w_ld + min(nblk, (step + PF) * KB) * 36);
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
            }
        }
    };
    prefetch(0);
    prefetch(PF);

    float acc[MI][NT][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // This lane's rows: A (RAW: row g of the warp's 16-row tiles, code word t; else its
    // ldmatrix rows; scale row g / g + 8 for lanes t = 0 / 1), B (ldmatrix: tokens of its pairs
    // of n8 tiles; scale token g).
    const int a_row = wm * 16 * MI + (RAW ? g : lane % 8 + (lane / 8) % 2 * 8), a_col = lane / 16 * 16;
    const int sa_row = wm * 16 * MI + g + (t % 2) * 8;
    const int b_tok = wn * 8 * NT + lane / 16 * 8 + lane % 8, b_col = (lane / 8) % 2 * 16;
    const int sb_tok = wn * 8 * NT + g;

    // n8 token tile j of this warp has tokens (always taken by the dense tiles: their partial
    // tiles are the last few only, and the tests cost 2-4% at full tiles).
    auto live = [&](int j) { return !GROUPED || wn * 8 * NT + j * 8 < n_tok; };

    struct Frag {
        uint32_t a[MI][4], b[NT][2], sa[MI], sb[NT];
    };
    int wsh = 0;  // CW 16: where the stage's weight blocks start in the copied window
    int lane_sh = 0;  // SH: 8 when this lane's rows start 8 bytes into a word
    if (SH) {
        int64_t row;
        const uint8_t * wt = w_row(a_row, row);
        lane_sh = (int) ((uintptr_t) (wt + row * p.w_ld) & 15);
    }
    auto fragments = [&](const unsigned char * ac, int blk, Frag & f) {
        const unsigned char * bc = ac + A_BYTES;
        const uint32_t * as = (const uint32_t *) (ac + A_SCALES);
        const uint32_t * bs = (const uint32_t *) (ac + B_SCALES);
#pragma unroll
        for (int i = 0; i < MI; i++) {
            if (RAW) {
                // The block as stored: the scale word, then the code words (a0..a3 as ldmatrix
                // gives them from the codes alone).
                const unsigned char * r = ac + wsh + (a_row + i * 16) * LDW + blk * 36 + 4 + t * 4;
                f.a[i][0] = *(const uint32_t *) r;
                f.a[i][1] = *(const uint32_t *) (r + 8 * LDW);
                f.a[i][2] = *(const uint32_t *) (r + 16);
                f.a[i][3] = *(const uint32_t *) (r + 8 * LDW + 16);
                f.sa[i] = *(const uint32_t *) (ac + wsh + (sa_row + i * 16) * LDW + blk * 36);
            } else {
                const uint32_t addr =
                    (uint32_t) __cvta_generic_to_shared(ac + (a_row + i * 16) * LDC + blk * 32 + a_col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(f.a[i][0]), "=r"(f.a[i][1]), "=r"(f.a[i][2]), "=r"(f.a[i][3])
                             : "r"(addr));
                f.sa[i] = as[(sa_row + i * 16) * KB + blk];
            }
        }
#pragma unroll
        for (int jj = 0; jj < NT / 2; jj++) {
            if (!live(2 * jj)) {
                break;
            }
            const uint32_t addr = (uint32_t) __cvta_generic_to_shared(bc + (b_tok + jj * 16) * LDC + blk * 32 + b_col);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(f.b[2 * jj][0]), "=r"(f.b[2 * jj][1]), "=r"(f.b[2 * jj + 1][0]), "=r"(f.b[2 * jj + 1][1])
                         : "r"(addr));
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            if (live(j)) {
                f.sb[j] = bs[(sb_tok + j * 8) * KB + blk];
            }
        }
    };

#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < steps) {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int64_t step = 0; step < steps; step++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
        __syncthreads();
        const int64_t next = step + ST - 1;
        if (next < steps) {
            load(next, (int) (next % ST));
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (step % PF == 0 && step > 0) {
            prefetch(step + PF);
        }
        if (!live(0)) {
            continue;  // none of this warp's tokens: it only copies
        }
        const unsigned char * ac = smem + (step % ST) * STAGE;
        if (CW == 16) {
            wsh = (int) ((step * KB * 36 + lane_sh) & 15);
        }
        Frag f[2];
        fragments(ac, 0, f[0]);
#pragma unroll
        for (int b = 0; b < KB; b++) {
            if (b + 1 < KB) {
                fragments(ac, b + 1, f[(b + 1) % 2]);
            }
            const Frag & c = f[b % 2];
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    if (live(j)) {
                        mma_fp4(acc[i][j], c.a[i], c.b[j][0], c.b[j][1], c.sa[i], c.sb[j]);
                    }
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    a4_epilogue<MI, NT, GROUPED, GLU, RES>(p, acc, m0, n0, n_tok, x_dec, d_off, sc, sc2, wm, wn, g, t);
#endif
}

#ifdef LEGION_FP4_MMA
// The GEMM's epilogue: this warp's accumulators (MI row tiles x NT n8 token tiles of the tile
// at m0, n0; SwiGLU: row tiles [0, MI / 2) of the gate, then the same rows of the up weight)
// times the scales, through p.act (and RES: plus the residual) to p.dst.
template <int MI, int NT, bool GROUPED, bool GLU, bool RES>
__device__ __forceinline__ void a4_epilogue(const legion_nvfp4_a4 & p, const float (&acc)[MI][NT][4], int64_t m0,
                                            int64_t n0, int n_tok, const float * x_dec, const int64_t * d_off,
                                            float sc, float sc2, int wm, int wn, int g, int t) {
    if constexpr (GLU) {
#pragma unroll
        for (int i = 0; i < MI / 2; i++) {
            const int64_t r0 = m0 + wm * 8 * MI + i * 16 + g;
#pragma unroll
            for (int j = 0; j < NT; j++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int64_t row = r0 + (e / 2) * 8;
                    const int tk = wn * 8 * NT + j * 8 + t * 2 + e % 2;
                    if (row < p.m && tk < n_tok) {
                        // As ggml-cuda's silu, then its mul (F32 out: the same bits as the
                        // grouped GEMMs' F32 outputs through those two ops).
                        const float gate = acc[i][j][e] * sc * x_dec[tk];
                        const float up = acc[i + MI / 2][j][e] * sc2 * x_dec[tk];
                        const float v = (p.act == 3 ? a4_gelu(gate) : gate / (1.0f + expf(-gate))) * up;
                        const int64_t off = (GROUPED ? d_off[tk] : (n0 + tk) * p.m) + row;
                        if (p.act == 0) {
                            ((float *) p.dst)[off] = v;
                        } else {
                            ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(v);
                        }
                    }
                }
            }
        }
    } else {
        // RES (dense, act 0): a row group's residuals loaded before its first store. dst may
        // alias them as far as the compiler knows, so loads between the stores would each wait
        // out the memory latency, with one CTA per SM to hide it.
        const bool add_res = RES && !GROUPED && !GLU && p.act == 0;
#pragma unroll
        for (int i = 0; i < MI; i++) {
            const int64_t r0 = m0 + wm * 16 * MI + i * 16 + g;
            float rv[NT][4];
            if constexpr (RES) {
#pragma unroll
                for (int j = 0; j < NT; j++) {
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int64_t row = r0 + (e / 2) * 8;
                        const int tk = wn * 8 * NT + j * 8 + t * 2 + e % 2;
                        rv[j][e] = add_res && row < p.m && tk < n_tok ? p.res[(n0 + tk) * p.m + row] : 0.0f;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int64_t row = r0 + (e / 2) * 8;
                    const int tk = wn * 8 * NT + j * 8 + t * 2 + e % 2;
                    if (row < p.m && tk < n_tok) {
                        const int64_t off = (GROUPED ? d_off[tk] : (n0 + tk) * p.m) + row;
                        const float v = acc[i][j][e] * sc * x_dec[tk];
                        if (p.act == 1) {
                            const float r = fmaxf(v, 0.0f);
                            ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(r * r);
                        } else if (p.act == 2) {
                            ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(v);
                        } else if (add_res) {
                            ((float *) p.dst)[off] = __fadd_rn(v, rv[j][e]);
                        } else {
                            ((float *) p.dst)[off] = v;
                        }
                    }
                }
            }
        }
    }
}
#endif

// The decode kernel: routed experts' grouped matmuls with a few tokens per expert (decode steps of
// up to 64 tokens), where the matmul is bound by the weight stream. As nvfp4_mm/common.cuh's nvfp4_mvs
// (W4A16): 16 * RT weight rows x 8 * NT tokens of one expert per CTA, KW warps splitting each
// stage's blocks (warp i takes blocks i, i + KW, ...); the CTA copies its rows' blocks as stored
// into shared memory SB blocks (36 * SB bytes of a row) at a time with coalesced CW-byte cp.async,
// two stages in flight, plus an L2 bulk prefetch of each row two stages further ahead (the GB10's
// LPDDR5X streams well only in long runs of each row). Small stages keep several CTAs per SM, which
// hides each CTA's start (the tile lookup) better than deeper or longer stages: measured on the
// GB10, SB 8 (4 for KW 2) streams 5-10% faster than 16 (`cuda_decode_timings`). A warp reads its A fragments and
// scale words from the blocks as stored (the RAW reads of the GEMM) and its B fragments, the
// quantized activations, through L1 one block ahead: one m16n8k64 MMA per block, row tile and n8
// token tile, nothing to dequantize. The KW warps' sums are added in a fixed order, so an
// output's bits depend on k and KW only (not on RT, NT or GLU: the SwiGLU gives the same bits as
// its two grouped matmuls, ggml-cuda's silu and its mul). GLU: row tiles [0, RT / 2) of the
// gate, then the same rows of the up weight. Epilogues as the GEMM's.
constexpr int MV_NS = 2;

// Blocks per stage for KW warps.
template <int KW> __host__ __device__ constexpr int mv_sb() {
    return KW < 4 ? 4 : 8;
}

// Shared memory: the stages, then (reused) the KW warps' sums.
template <int NT, int RT, int KW> __host__ __device__ constexpr int mv_smem() {
    return MV_NS * 16 * RT * (mv_sb<KW>() * 36 + 16) > KW * RT * NT * 4 * 32 * 4
               ? MV_NS * 16 * RT * (mv_sb<KW>() * 36 + 16)
               : KW * RT * NT * 4 * 32 * 4;
}

template <int NT, int RT, int KW, int CW, bool GLU>
__global__ void __launch_bounds__(KW * 32) nvfp4_a4_mv(const legion_nvfp4_a4 p) {
    static_assert(!GLU || RT % 2 == 0, "SwiGLU: gate and up row tiles");
    static_assert(CW == 8 || CW == 16, "copy width");
#ifdef LEGION_FP4_MMA
    constexpr int SB = mv_sb<KW>(), NS = MV_NS, LD = SB * 36 + 16;
    constexpr int BN = 8 * NT, ROWS = 16 * RT, THREADS = KW * 32, STAGE = ROWS * LD;
    constexpr int OUT_ROWS = GLU ? ROWS / 2 : ROWS;
    constexpr int FRAG = RT * NT * 4;
    static_assert(SB % KW == 0 && BN <= 32, "tile");
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int64_t x_row[BN], d_off[BN];
    __shared__ float x_dec[BN];
    __shared__ int32_t tile_expert, tile_tokens;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;

    const int64_t m0 = (int64_t) blockIdx.x * OUT_ROWS;
    if (warp == 0) {
        int e, first, count;
        if (!find_tile(p.plan, p.n_expert, BN, blockIdx.y, lane, e, first, count)) {
            if (lane == 0) {
                tile_expert = -1;
            }
        } else {
            if (lane == 0) {
                tile_expert = e;
                tile_tokens = count;
            }
            if (lane < BN) {
                if (lane < count) {
                    const int q = p.plan[p.n_expert + 1 + first + lane];  // token * n_used + slot
                    const int tk = q / p.n_used, slot = q % p.n_used;
                    const int64_t xr = (int64_t) tk * p.n_x + (p.n_x == 1 ? 0 : slot);
                    x_row[lane] = xr;
                    x_dec[lane] = p.pw ? p.xd[xr] * p.pw[q] : p.xd[xr];
                    d_off[lane] = (int64_t) q * p.m;
                } else {
                    x_row[lane] = 0;
                    x_dec[lane] = 0.0f;
                    d_off[lane] = 0;
                }
            }
        }
    }
    __syncthreads();
    if (tile_expert < 0) {
        return;
    }
    const uint8_t * w = p.w + (int64_t) tile_expert * p.w_e;
    const uint8_t * w2 = GLU ? p.w2 + (int64_t) tile_expert * p.w_e : nullptr;
    const float sc = p.scale ? p.scale[tile_expert] : 1.0f;
    const float sc2 = GLU && p.scale2 ? p.scale2[tile_expert] : 1.0f;
    const int n_tok = tile_tokens;

    const int nblk = (int) (p.k / 64);
    const int nst = (nblk + SB - 1) / SB;
    // Row r of the CTA's weight tiles (past the last row, the last one).
    auto row_ptr = [&](int r) {
        if (GLU) {
            const bool up = r >= ROWS / 2;
            const int64_t row = min(m0 + (up ? r - ROWS / 2 : r), p.m - 1);
            return (up ? w2 : w) + row * p.w_ld;
        }
        return w + min(m0 + r, p.m - 1) * p.w_ld;
    };
    auto load = [&](int st) {
        unsigned char * dst = smem + (st % NS) * STAGE;
        const int blocks = min(SB, nblk - st * SB);
        const int64_t base = (int64_t) st * SB * 36;
        constexpr int CH = SB * 36 / CW;  // copies per row
        const int ch = blocks * 36 / CW;  // whole: the rows are CW-aligned (see the launcher)
#pragma unroll 4
        for (int i = 0; i < (ROWS * CH + THREADS - 1) / THREADS; i++) {
            const int c = threadIdx.x + i * THREADS;
            const int r = c / CH, q = c % CH;
            if (r < ROWS) {
                if (CW == 16) {
                    cp_async16(dst + r * LD + q * 16, row_ptr(r) + base + q * 16, q < ch);
                } else {
                    cp_async8(dst + r * LD + q * 8, row_ptr(r) + base + q * 8, q < ch);
                }
            }
        }
    };
    auto prefetch = [&](int st) {
        if (threadIdx.x < ROWS && st < nst) {
            const uintptr_t a = (uintptr_t) (row_ptr(threadIdx.x) + (int64_t) st * SB * 36);
            const uintptr_t e = a + min(SB, nblk - st * SB) * 36;
            const uintptr_t a16 = a & ~(uintptr_t) 15;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                         "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
        }
    };

    // This lane's B column in each n8 token tile: token g (a padding column reads the tile's
    // first token; it is not stored). The tiles past the last token are skipped.
    const uint8_t * xq[NT];
    const uint8_t * xs[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int c = j * 8 + g < n_tok ? j * 8 + g : 0;
        xq[j] = p.xq + x_row[c] * p.xq_ld + t * 4;
        xs[j] = p.xs + x_row[c] * p.xs_ld;
    }
    auto live = [&](int j) { return j * 8 < n_tok; };

    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[r][j][e] = 0.0f;

    for (int s = NS - 1; s < 2 * NS - 1; s++) {
        prefetch(s);
    }
    // The activations of this warp's next block: code words t and 4 + t, the scale word.
    uint32_t nb0[NT], nb1[NT], nsb[NT];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        if (warp < nblk && live(j)) {
            nb0[j] = __ldg((const uint32_t *) (xq[j] + (int64_t) warp * 32));
            nb1[j] = __ldg((const uint32_t *) (xq[j] + (int64_t) warp * 32 + 16));
            nsb[j] = __ldg((const uint32_t *) (xs[j] + (int64_t) warp * 4));
        }
    }
#pragma unroll
    for (int s = 0; s < NS - 1; s++) {
        if (s < nst) {
            load(s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int st = 0; st < nst; st++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
        __syncthreads();
        if (st + NS - 1 < nst) {
            load(st + NS - 1);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        prefetch(st + 2 * NS - 1);
        const unsigned char * ws = smem + (st % NS) * STAGE;
#pragma unroll
        for (int i = 0; i < SB / KW; i++) {
            const int bl = warp + i * KW, b = st * SB + bl;
            if (b >= nblk) {
                break;
            }
            uint32_t b0[NT], b1[NT], sb[NT];
#pragma unroll
            for (int j = 0; j < NT; j++) {
                b0[j] = nb0[j];
                b1[j] = nb1[j];
                sb[j] = nsb[j];
                if (b + KW < nblk && live(j)) {
                    nb0[j] = __ldg((const uint32_t *) (xq[j] + (int64_t) (b + KW) * 32));
                    nb1[j] = __ldg((const uint32_t *) (xq[j] + (int64_t) (b + KW) * 32 + 16));
                    nsb[j] = __ldg((const uint32_t *) (xs[j] + (int64_t) (b + KW) * 4));
                }
            }
#pragma unroll
            for (int r = 0; r < RT; r++) {
                // Rows g and g + 8 of the row tile: the code words t and 4 + t of the block
                // (after its scale word), and the scale word of row g (lane t 0) or g + 8 (t 1).
                const unsigned char * q = ws + (r * 16 + g) * LD + bl * 36 + 4 + t * 4;
                uint32_t a[4];
                a[0] = *(const uint32_t *) q;
                a[1] = *(const uint32_t *) (q + 8 * LD);
                a[2] = *(const uint32_t *) (q + 16);
                a[3] = *(const uint32_t *) (q + 8 * LD + 16);
                const uint32_t sa = *(const uint32_t *) (ws + (r * 16 + g + (t % 2) * 8) * LD + bl * 36);
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    if (live(j)) {
                        mma_fp4(acc[r][j], a, b0[j], b1[j], sa, sb[j]);
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // f: accumulator index (row tile, token tile, element) of lane ln; GLU: v the gate's, v2 the
    // up weight's of the same row and token.
    auto store = [&](int f, int ln, float v, float v2) {
        const int e = f % 4, j = f / 4 % NT, r = f / 4 / NT;
        const int64_t row = m0 + r * 16 + (e / 2) * 8 + ln / 4;
        const int tok = j * 8 + (ln % 4) * 2 + e % 2;
        if (row < p.m && tok < n_tok) {
            const int64_t off = d_off[tok] + row;
            if (GLU) {
                // As the GEMM's (ggml-cuda's silu, then its mul).
                const float gate = v * sc * x_dec[tok];
                const float up = v2 * sc2 * x_dec[tok];
                const float o = gate / (1.0f + expf(-gate)) * up;
                if (p.act == 0) {
                    ((float *) p.dst)[off] = o;
                } else {
                    ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(o);
                }
            } else {
                const float o = v * sc * x_dec[tok];
                if (p.act == 1) {
                    const float rl = fmaxf(o, 0.0f);
                    ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(rl * rl);
                } else if (p.act == 2) {
                    ((__nv_bfloat16 *) p.dst)[off] = __float2bfloat16_rn(o);
                } else {
                    ((float *) p.dst)[off] = o;
                }
            }
        }
    };
    // The KW warps' sums through shared memory, added in warp order.
    constexpr int OUT = GLU ? FRAG / 2 : FRAG;  // accumulators that are stored (GLU: the gate's)
    __syncthreads();                            // every warp is done with the stages
    float * red = (float *) smem;
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) red[(warp * FRAG + (r * NT + j) * 4 + e) * 32 + lane] = acc[r][j][e];
    __syncthreads();
    for (int i = threadIdx.x; i < OUT * 32; i += THREADS) {
        float v = red[i], v2 = GLU ? red[OUT * 32 + i] : 0.0f;
#pragma unroll
        for (int q = 1; q < KW; q++) {
            v += red[q * FRAG * 32 + i];
            if (GLU) {
                v2 += red[q * FRAG * 32 + OUT * 32 + i];
            }
        }
        store(i / 32, i % 32, v, v2);
    }
#endif
}

// One thread per (row pair, token): two rows at a time, the slots in order.
__global__ void nvfp4_a4_combine(const __nv_bfloat16 * y, int64_t m, int32_t n_used, int64_t n_tok, float * out) {
    const int64_t r = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 2, t = blockIdx.y;
    if (r >= m) {
        return;
    }
    const __nv_bfloat16 * src = y + t * n_used * m + r;
    float a = 0.0f, b = 0.0f;
    if (r + 1 < m && m % 2 == 0) {
        for (int s = 0; s < n_used; s++) {
            const float2 v = __bfloat1622float2(*(const __nv_bfloat162 *) (src + s * m));
            a += v.x;
            b += v.y;
        }
        *(float2 *) (out + t * m + r) = make_float2(a, b);
    } else {
        for (int s = 0; s < n_used; s++) {
            a += __bfloat162float(src[s * m]);
            if (r + 1 < m) {
                b += __bfloat162float(src[s * m + 1]);
            }
        }
        out[t * m + r] = a;
        if (r + 1 < m) {
            out[t * m + r + 1] = b;
        }
    }
}

__global__ void nvfp4_a4_probe(int * out) {
#ifdef LEGION_FP4_MMA
    *out = 1;
#else
    *out = 0;
#endif
}

int status() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}

// The TMA-fed dense GEMM (nvfp4_a4_gemm_tma, dense_tma).
#include "../sm90/tma.cuh"
#include "nvfp4_a4_tma.cuh"

// GROUPED_ONLY: the grouped kernels only (dense calls are -1).
template <class T, bool GROUPED_ONLY = false>
int gemm(const legion_nvfp4_a4 & p, cudaStream_t st) {
    const unsigned tiles = (unsigned) p.tiles_m * (unsigned) p.tiles_n;
    const unsigned grid = tiles;
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<grid, 256, T::SMEM, st>>>(p);
    };
    // RAW: 8-byte weight copies when every row's stages start 8-byte aligned.
    const uint64_t bases = (uint64_t) (uintptr_t) p.w | (uint64_t) (uintptr_t) p.w2 | (uint64_t) p.w_ld |
                           (uint64_t) (p.plan ? p.w_e : 0);
    const bool wide = T::RAW && T::KB % 2 == 0 && bases % 8 == 0 && (p.k / 64) % T::KB == 0;
    constexpr int W = T::RAW && T::KB % 2 == 0 ? 8 : 4;
    auto pick = [&](auto narrow, auto wide_kernel) { wide ? go(wide_kernel) : go(narrow); };
    if (p.w2 && p.plan) {
        pick(nvfp4_a4_gemm<T, true, true, 4>, nvfp4_a4_gemm<T, true, true, W>);
    } else if (GROUPED_ONLY) {
        if (!p.plan) {
            return -1;
        }
        pick(nvfp4_a4_gemm<T, true, false, 4>, nvfp4_a4_gemm<T, true, false, W>);
    } else if (p.w2) {
        pick(nvfp4_a4_gemm<T, false, true, 4>, nvfp4_a4_gemm<T, false, true, W>);
    } else if (p.plan) {
        pick(nvfp4_a4_gemm<T, true, false, 4>, nvfp4_a4_gemm<T, true, false, W>);
    } else if (p.res) {
        if constexpr (!GROUPED_ONLY) {
            pick(nvfp4_a4_gemm<T, false, false, 4, true>, nvfp4_a4_gemm<T, false, false, W, true>);
        }
    } else {
        pick(nvfp4_a4_gemm<T, false, false, 4>, nvfp4_a4_gemm<T, false, false, W>);
    }
    return status();
}

// The dense GEMMs (SwiGLU, residual or plain) of tile T (RAW, KB even) with 16-byte weight
// copies; -1 when the weight rows are not 16-byte aligned.
template <class T> int dense16(const legion_nvfp4_a4 & p, cudaStream_t st) {
    const uint64_t bases = (uint64_t) (uintptr_t) p.w | (uint64_t) (uintptr_t) p.w2 | (uint64_t) p.w_ld;
    if (p.plan || bases % 16 != 0) {
        return -1;
    }
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<(unsigned) p.tiles_m * (unsigned) p.tiles_n, 256, T::SMEM, st>>>(p);
    };
    if (p.w2) {
        go(nvfp4_a4_gemm<T, false, true, 16>);
    } else if (p.res) {
        go(nvfp4_a4_gemm<T, false, false, 16, true>);
    } else {
        go(nvfp4_a4_gemm<T, false, false, 16>);
    }
    return status();
}

// The grouped GEMMs of tile T (RAW, KB even) with 16-byte weight copies; -1 unless grouped with
// 16-byte aligned bases and experts and weight rows 16-byte aligned (k % 256 == 0: SwiGLU or
// plain) or, SH, 8 bytes past it (plain only, the tested case).
template <class T, bool SH> int grouped16(const legion_nvfp4_a4 & p, cudaStream_t st) {
    const uint64_t bases = (uint64_t) (uintptr_t) p.w | (uint64_t) (uintptr_t) p.w2 | (uint64_t) p.w_e;
    if (!p.plan || bases % 16 != 0 || p.w_ld % 16 != (SH ? 8 : 0) || (SH && p.w2) || (p.k / 64) % T::KB != 0) {
        return -1;
    }
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<(unsigned) p.tiles_m * (unsigned) p.tiles_n, 256, T::SMEM, st>>>(p);
    };
    if constexpr (SH) {
        go(nvfp4_a4_gemm<T, true, false, 16, false, true>);
    } else if (p.w2) {
        go(nvfp4_a4_gemm<T, true, true, 16>);
    } else {
        go(nvfp4_a4_gemm<T, true, false, 16>);
    }
    return status();
}

template <int NT, int RT, int KW, bool GLU> int mv(const legion_nvfp4_a4 & p, cudaStream_t st) {
    constexpr int SMEM = mv_smem<NT, RT, KW>();
    const uint64_t bases = (uint64_t) (uintptr_t) p.w | (uint64_t) (uintptr_t) p.w2 | (uint64_t) p.w_ld | (uint64_t) p.w_e;
    if (bases % 8 != 0) {
        return -1;
    }
    auto go = [&](auto kernel) {
        if (SMEM > 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        }
        kernel<<<dim3((unsigned) p.tiles_m, (unsigned) p.tiles_n), KW * 32, SMEM, st>>>(p);
    };
    bases % 16 == 0 ? go(nvfp4_a4_mv<NT, RT, KW, 16, GLU>) : go(nvfp4_a4_mv<NT, RT, KW, 8, GLU>);
    return status();
}

template <int NT, int RT, bool GLU> int mv_kw(const legion_nvfp4_a4 & p, int kw, cudaStream_t st) {
    switch (kw) {
        case 2: return mv<NT, RT, 2, GLU>(p, st);
        case 4: return mv<NT, RT, 4, GLU>(p, st);
        case 8: return mv<NT, RT, 8, GLU>(p, st);
        default: return -1;
    }
}

template <int NT> int mv_rt(const legion_nvfp4_a4 & p, int rt, int kw, cudaStream_t st) {
    if (p.w2) {
        switch (rt) {
            case 2: return mv_kw<NT, 2, true>(p, kw, st);
            case 4: return mv_kw<NT, 4, true>(p, kw, st);
            default: return -1;
        }
    }
    switch (rt) {
        case 1: return mv_kw<NT, 1, false>(p, kw, st);
        case 2: return mv_kw<NT, 2, false>(p, kw, st);
        default: return -1;
    }
}
}  // namespace

int legion_nvfp4_a4_quant_launch(const void * x, int x_bf16, int64_t x_ld, int64_t k, int64_t rows, uint8_t * q,
                                 int64_t q_ld, uint8_t * s, int64_t s_ld, float * d, void * stream) {
    if (rows == 0) {
        return 0;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (x_bf16) {
        nvfp4_a4_quant<<<(unsigned) rows, 128, 0, st>>>((const __nv_bfloat16 *) x, x_ld, k, q, q_ld, s, s_ld, d);
    } else {
        nvfp4_a4_quant<<<(unsigned) rows, 128, 0, st>>>((const float *) x, x_ld, k, q, q_ld, s, s_ld, d);
    }
    return status();
}

int legion_nvfp4_a4_rms_quant_launch(const float * x, int64_t x_ld, const float * w, float eps, int64_t k,
                                     int64_t rows, uint8_t * q, int64_t q_ld, uint8_t * s, int64_t s_ld, float * d,
                                     void * stream) {
    if (k % 64 != 0 || k > LEGION_NVFP4_A4_RMS_MAX_K) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const size_t smem = (size_t) k * sizeof(float);
    if (k < 1024) {
        nvfp4_a4_rms_quant<256><<<(unsigned) rows, RQ_THREADS, smem, st>>>(x, x_ld, w, eps, (int) k, q, q_ld, s, s_ld, d);
    } else {
        nvfp4_a4_rms_quant<1024><<<(unsigned) rows, RQ_THREADS, smem, st>>>(x, x_ld, w, eps, (int) k, q, q_ld, s, s_ld, d);
    }
    return status() == 0 ? 0 : 2;
}

// The tiles: 128 rows x 16 * nt tokens, 2 blocks per stage, 4 stages (90 KiB). Measured on
// the GB10 against 256-row tiles, 1 or 4 blocks per stage, 2 to 7 stages and two CTAs per SM:
// the best for both long k (Qwen3.6-27B's MLP, ~175 TFLOPS) and short k (MoE experts) in one
// CTA per SM. LEGION_NVFP4_A4_TWO_CTAS | 4, for the grouped SwiGLU and short-k grouped GEMMs of
// Qwen3.8-Flash-Next's experts (512 x top-10, 2560 x 640): 64 tokens, 3 stages of RAW weight
// rows (47 KiB, so two CTAs fit an SM's 100 KiB), 8-byte weight copies; 5-10% faster there, as
// one CTA's prologue and epilogue overlap the other's k loop. (RAW in the dense tiles costs
// 2-4% with 4- or 8-byte copies: four 32-bit reads for each ldmatrix.) Dense nt 8 with 16-byte
// aligned weight rows (k % 256 == 0, so whole stages): RAW rows, 4 blocks per stage, 2 stages
// (76 KiB), 16-byte weight copies (9 per row and stage, against 36 four-byte ones): 14-17%
// faster on Qwen3.8-27B's MLP at 96 to 4096 tokens (GB10, #749), the same MMAs in the same
// order (the same bits). The 4-block stage needs the 16-byte copies: with 4- or 8-byte ones
// it gains only 2-10%. Grouped (#749, GB10, Qwen3.8-Flash-Next's experts, 512 x top-10, 512 to
// 4096 tokens): nt 8 with 16-byte aligned rows takes the same RAW 4 x 2 tile with 16-byte
// copies (3-7% faster at k 2560); the two-CTA tile takes 16-byte copies for the plain GEMMs
// whose rows are 8 bytes past a 16-byte word (k 640, the experts' down matmul: 0.4-1.8% faster
// at 4096 tokens, 4-5% at 512 to 1024), but not for 16-byte aligned rows or the SwiGLU, which
// stream the weights from DRAM as fast with 8-byte copies (16-byte ones: 2-8% slower at 4096
// tokens). Every variant has the same MMAs in the same order (the same bits).
// Dense nt 8 with k % 256 == 0 first takes the TMA-fed tile (nvfp4_a4_gemm_tma, #1133): one
// bulk tensor copy per operand and stage instead of 16-byte cp.async, the same MMAs in the same
// order. On the GB10 at Qwen3.8-27B's MLP against the 16-byte-copy tile above: the SwiGLU (k
// 5120, 2 x 17408 rows) 128 x 128 x 2 stages, 219 -> 248 TFLOPS at 8192 tokens, 86 against 83
// at 128; at 16384 tokens 128 x 96 x 3 stages (234 against 225 for 128 x 128). The down matmul
// with its residual (k 17408, 5120 rows) 128 x 96 x 3 stages from 2048 tokens: 180 -> 233
// TFLOPS at 8192 (128 x 128: 203), 128 x 128 below (81 against 72 at 128 tokens).
int legion_nvfp4_a4_gemm_launch(const legion_nvfp4_a4 * p, int nt, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    switch (nt) {
        case LEGION_NVFP4_A4_TWO_CTAS | 4: {
            const int r = grouped16<Tile<2, 4, 2, 3, 2, true>, true>(*p, st);
            return r != -1 ? r : gemm<Tile<2, 4, 2, 3, 2, true>, true>(*p, st);
        }
        case 2: return gemm<Tile<2, 2, 2, 4>>(*p, st);
        case 4: return gemm<Tile<2, 4, 2, 4>>(*p, st);
        case 8: {
            const bool deep = p->w2 ? p->n >= 16384 : p->k >= 8192 && p->n >= 2048;
            int r = deep ? dense_tma<6, 3>(*p, st) : dense_tma<8, 2>(*p, st);
            if (r == -1) {
                r = dense16<Tile<2, 8, 4, 2, 1, true>>(*p, st);
            }
            if (r == -1) {
                r = grouped16<Tile<2, 8, 4, 2, 1, true>, false>(*p, st);
            }
            return r != -1 ? r : gemm<Tile<2, 8, 2, 4>>(*p, st);
        }
        default: return -1;
    }
}

int legion_nvfp4_a4_mv_launch(const legion_nvfp4_a4 * p, int nt, int rt, int kw, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    if (!p->plan || p->k % 64 != 0) {
        return -1;
    }
    switch (nt) {
        case 1: return mv_rt<1>(*p, rt, kw, st);
        case 2: return mv_rt<2>(*p, rt, kw, st);
        case 4: return mv_rt<4>(*p, rt, kw, st);
        default: return -1;
    }
}

int legion_nvfp4_a4_combine_launch(const __nv_bfloat16 * y, int64_t m, int32_t n_used, int64_t n_tok, float * out,
                                   void * stream) {
    if (n_tok == 0) {
        return 0;
    }
    const dim3 grid((unsigned) ((m + 511) / 512), (unsigned) n_tok);
    nvfp4_a4_combine<<<grid, 256, 0, (cudaStream_t) stream>>>(y, m, n_used, n_tok, out);
    return status();
}

int legion_nvfp4_a4_supported(void) {
    int * d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    nvfp4_a4_probe<<<1, 1>>>(d);
    const bool ok = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    cudaGetLastError();
    return ok ? h : 0;
}
