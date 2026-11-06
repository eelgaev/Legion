// W4A16 NVFP4 matmuls: ggml NVFP4 weights (per 64 elements: four UE4M3 sub-block scales, then 32
// bytes of E2M1 codes, element j in the low and j + 8 in the high nibble of byte j of each
// 16-element sub-block) times BF16 activations, F32 accumulation, times the per-tensor (or
// per-expert) scale. Three kernels:
// - `nvfp4_gemm`: tiles of 128 or 256 rows x 8..128 tokens through shared memory (prefill);
// - `nvfp4_mvs`: 16-row tiles per warp, weights staged through shared memory in long runs of
//   each row (decode batches of up to 32 tokens, where the matmul is bound by the weight
//   stream); its grouped SwiGLU variant (IGLU) sums 8 gate and 8 up rows per tile and stores
//   silu(gate) * up;
// - `nvfp4_plan`: groups mul_mat_id's (token, slot) pairs by expert for the grouped variants of
//   both, where a tile's tokens are one expert's.
//
// Kernel bodies only: shapes, tile sizes, the grids and the tile order are decided in Rust
// (src/model/nvfp4_mm.rs); the launch stubs only opt in to the shared memory and launch.
//
// Files (#955: one nvcc process per kernel family and tile width, so they build in parallel):
// - common.cuh: the parameters, the device helpers, the kernel templates, their launch code and
//   the per-part entry points' declarations (namespace legion_nvfp4).
// - gemm_mi1.cu, gemm_mi2.cu: the GEMM of 128 and 256 rows (launch_gemm<MI>); gemm_glu.cu: the
//   fused SwiGLU GEMM (launch_swiglu_gemm).
// - mv_nt1.cu, mv_nt2.cu, mv_nt4.cu, mv_nt8.cu: the mv kernel of 8 * NT tokens (launch_mv<NT>);
//   mvg_nt1.cu, mvg_nt2.cu, mvg_nt4.cu: the grouped mv kernel's own variants (launch_mvg<NT>);
//   swiglu_mv.cu: the fused SwiGLU mv kernel (launch_swiglu_mv).
// - dispatch.cu: the extern "C" entry points, nvfp4_plan and the read probe.
//
// Decoding (exact): a code's E2M1 bits go straight into a BF16's low exponent and top mantissa
// bits, which reads as the code's value times 2^-126 (code 1, 0.5, becomes a BF16 subnormal);
// the sub-block scale is taken as a BF16 times 2^116, so their product is the weight times
// 2^-10, exact (at most 6 significant bits) and a normal BF16. The epilogues multiply by 2^10
// and the per-tensor scale. Products are mma.sync m16n8k16 (BF16 in, F32 sums).
//
// Fragments (PTX ISA, g = lane / 4, t = lane % 4): A {a0,a1} = (row g, k 2t..2t+1), {a2,a3} =
// (g + 8, 2t..), {a4,a5} = (g, 2t+8..), {a6,a7} = (g + 8, 2t+8..); B {b0,b1} = (k 2t..2t+1,
// col g), {b2,b3} = (k 2t+8.., col g); C c0,c1 = (row g, col 2t..2t+1), c2,c3 = (g + 8, ..).
// - The GEMM keeps the physical k order: a lane needs elements c, c + 1, c + 8, c + 9 (c = 2t)
//   of a sub-block, the low and high nibbles of bytes c and c + 1, one 16-bit load per row.
// - The mv kernel permutes k (the sum does not depend on its order, as long as A and B agree):
//   lane t takes all 16 elements of sub-block t, so it loads its 8 code bytes and its scale
//   byte from global memory itself, and k-step s of a block uses bytes 2s and 2s + 1 (elements
//   2s, 2s + 1 as a0a1, 2s + 8, 2s + 9 as a4a5); its B fragments are the matching activations,
//   elements 16t .. 16t + 15 of the block: two 16-byte loads per column.

#pragma once

#include <cuda_bf16.h>
#include <atomic>
#include <cstdint>
#include <type_traits>

extern "C" {
struct legion_nvfp4_mm {
    const uint8_t * w;        // NVFP4 rows of w_ld bytes; experts w_e bytes apart
    const __nv_bfloat16 * x;  // BF16 rows of x_ld elements
    const float * scale;      // per-tensor (dense) or per-expert (grouped) scale, or null
    float * dst;              // F32 rows of m
    int64_t k, m, n, x_ld, w_ld, w_e;
    int32_t tiles_m, tiles_n;  // row tiles and token tiles (grouped: an upper bound)
    int32_t group_m;           // dense GEMM tile order: groups of group_m row tiles
    // Grouped (mul_mat_id): plan = [offsets of each expert's pairs (n_expert + 1), the pairs
    // (token * n_used + slot) grouped by expert (n of them), the token tiles of 8, 16, 32,
    // 64 and 128 pairs over every expert (PLAN_TILES)]; x has n_x (1 or n_used) rows per token, dst
    // n_used.
    const int32_t * plan;
    int32_t n_expert, n_used, n_x;
    // Fused SwiGLU (dense): w is the gate, w2 the up weight (same shape and layout), scale2 the
    // up weight's scale; dst (BF16) = silu(scale * gate . x) * (scale2 * up . x).
    const uint8_t * w2;
    const float * scale2;
    // Epilogue: 0 stores F32; 1 stores relu(v)^2 as BF16 (squared-ReLU FFNs: the next matmul's
    // input, which it would round to BF16 anyway). Fused SwiGLU (dense): the gate's activation,
    // 0 silu, 2 the tanh GELU (a GeGLU FFN; see glu_gate).
    int32_t act;
    // Fused SwiGLU GEMM, above 0: GLM-5.3-Flash's clamp, silu(min(gate, limit)) *
    // clamp(up, -limit, limit) (the mv kernel ignores it).
    float limit;
    // The grouped mv kernel: launched as a programmatic dependent of the kernel ahead (the
    // grouped SwiGLU mv, which lets it start early) on the GB10; see nvfp4_mvs.
    int32_t pdl;
    // Grouped: only the experts with more than sel_lo and at most sel_hi pairs have tiles (the
    // others none), so two launches on one plan split the experts by their pair counts (the
    // busy ones on the GEMM, the rest on the mv kernel). 0 and INT32_MAX: every expert.
    int32_t sel_lo, sel_hi;
    // Grouped, nonzero (#1176): the plan's skipped pairs (ids of -1, no padding rows) get no
    // zeros stored, because nothing reads their rows (the down matmul skips them, and the MoE
    // tail of --tp-moe-ep does not read rows of zero router weights). 0: zeros, as before.
    int32_t unread_skips;
};

// The GEMM: tiles_m * tiles_n CTAs of 256 threads; mi = 1 or 2 (128 * mi rows), nt = 1, 2, 4,
// 8 or 16 (8 * nt tokens). Needs k % 64 == 0 and x_ld % 8 == 0. variant: stages * 10 + CTAs
// per SM, 31 the default; 21, 22, 32 and 41 for grouped tiles of nt 8 or 16 (the same bits).
// -1 for a missing variant.
int legion_nvfp4_gemm_launch(const legion_nvfp4_mm * p, int mi, int nt, int variant, void * stream);

// The mv kernel: grid (tiles_m, tiles_n), kw warps per CTA splitting k; each CTA covers
// 16 * rtw rows and 8 * nt tokens. nt = 1, 2 or 4, rtw = 1 or 2, kw = 4 or 8; nt = 8 only
// with kw = 8 and rpt = 1. Needs
// k % 64 == 0, x_ld % 8 == 0 and 16-byte aligned x. -1 for a missing variant.
int legion_nvfp4_mv_launch(const legion_nvfp4_mm * p, int nt, int rtw, int kw, int rpt, void * stream);

// The grouped mv kernel (plan set), 8 warps: rpt row tiles per CTA, stages of sb blocks. p->w2
// null: 16 * rtw rows per row tile; set (rtw 1): the SwiGLU of the gate (p->w, p->scale) and up
// (p->w2, p->scale2) experts, 8 rows of each per row tile, BF16 out (tiles_m counts tiles of
// 8 * rpt output rows). nt = 1, 2 or 4 (wider tiles and longer stages: nt = 1 only). -1 for a
// missing variant.
// pft: 0 the per-stage L2 prefetch of each row; above 0, whole row tiles (contiguous rows of an
// expert) prefetched pft tiles ahead (nt = 1 and the variants listed in mvg).
int legion_nvfp4_mvg_launch(const legion_nvfp4_mm * p, int nt, int rtw, int rpt, int sb, int pft, void * stream);

// The fused SwiGLU mv kernel (w2, scale2 set): grid (tiles_m of 16 rows, tiles_n), 8 warps.
// nt = 1, 2, 4 or 8.
int legion_nvfp4_swiglu_mv_launch(const legion_nvfp4_mm * p, int nt, void * stream);
// The fused SwiGLU GEMM: tiles_m (128 output rows each: 128 gate and 128 up rows per CTA) *
// tiles_n CTAs of 8 * nt tokens, the dense tile order; nt = 4, 8 or 16. With plan set, grouped
// as legion_nvfp4_gemm_launch's (w2, scale2: the up experts and their per-expert scales; dst:
// BF16 rows of m per (token, slot) pair). variant: as legion_nvfp4_gemm_launch's (grouped, nt
// 8 or 16).
int legion_nvfp4_swiglu_gemm_launch(const legion_nvfp4_mm * p, int nt, int variant, void * stream);

// The grouped plan from ids (I32 [n_used, n_tokens], rows ids_ld apart): n_expert <= 1024.
// rows (I32 [n_tokens], or null): token t's pairs are grouped by token rows[t]'s experts; with
// skip_pad, the pairs of every token t with rows[t] != t (a verify step's padding) are in no
// expert's tiles but listed after them (plan[n_expert] counts the others); so are the pairs
// whose id is -1 (--tp-moe-ep: experts another rank holds). The grouped mv and GEMM kernels
// store zeros for them in the CTAs past the plan's last tile (the host grids keep a spare one).
int legion_nvfp4_plan_launch(const int32_t * ids, int64_t ids_ld, const int32_t * rows, int32_t skip_pad,
                             int32_t n_used, int64_t n_tokens, int32_t n_expert, int32_t * plan, void * stream);

// Streaming multiprocessors of the current device.
int legion_cuda_sm_count(void);

// legion nvfp4-mv-bench's read probe: CTA x reads expert ranges[2x + 1] of weight ranges[2x] (w0,
// w1 or w2, experts w_e bytes apart, w_e % 16 == 0) in chunks of `chunk` bytes (grid y), 16-byte
// streaming loads and nothing else: what a pure stream of the experts the mv kernels read gets.
int legion_nvfp4_read_probe_launch(const uint8_t * w0, const uint8_t * w1, const uint8_t * w2, int64_t w_e,
                                   const int32_t * ranges, int32_t n, int64_t chunk, uint32_t * sink, void * stream);
}

namespace legion_nvfp4 {

// The plan's token tile counts after its pairs (bn = 8, 16, 32, 64, 128: every mv and GEMM token
// tile): src/model/nvfp4_mm.rs's PLAN_TILES.
constexpr int PLAN_TILES = 5;

// The index of the count of bn-pair tiles among the plan's PLAN_TILES, -1 for none.
__host__ __device__ constexpr int plan_tile_index(int bn) {
    return bn == 8 ? 0 : bn == 16 ? 1 : bn == 32 ? 2 : bn == 64 ? 3 : bn == 128 ? 4 : -1;
}

// Whether a grouped launch's token tile `tile` of bn = 8 << TI pairs lies past the routing's
// last one (the host grid is an upper bound, set without a sync on the routing): the plan's own
// tile count, valid with no pair-count selection only (a split's launches count their experts'
// tiles alone, so they scan as before). Every lane of the CTA reads the same answer.
template <int TI> __device__ __forceinline__ bool past_plan_tiles(const legion_nvfp4_mm & p, int tile) {
    if constexpr (TI < 0) {
        return false;
    } else {
        return p.sel_lo == 0 && p.sel_hi == INT32_MAX && tile >= p.plan[p.n_expert + 1 + p.n + TI];
    }
}

// Each part's launch code (the switches below, over its instantiations), defined in its own .cu.
template <int MI> int launch_gemm(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st);
template <> int launch_gemm<1>(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st);
template <> int launch_gemm<2>(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st);
int launch_swiglu_gemm(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st);
template <int NT> int launch_mv(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st);
template <> int launch_mv<1>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st);
template <> int launch_mv<2>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st);
template <> int launch_mv<4>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st);
template <> int launch_mv<8>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st);
template <int NT>
int launch_mvg(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft, cudaStream_t st);
template <> int launch_mvg<1>(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft, cudaStream_t st);
template <> int launch_mvg<2>(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft, cudaStream_t st);
template <> int launch_mvg<4>(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft, cudaStream_t st);
int launch_swiglu_mv(const legion_nvfp4_mm & p, int nt, cudaStream_t st);

// pdl_code's answer for every part (defined once, in dispatch.cu): 0 unknown, 1 yes, 2 no.
extern std::atomic<int> pdl_state;

}  // namespace legion_nvfp4

namespace {
constexpr int BK = 64, STAGES = 3;
// smem bytes per weight row: the three 16-byte words holding the row's 36-byte block (it starts
// 0, 4, 8 or 12 bytes into the first), conflict-free for the fragment reads.
constexpr int LDW = 48;
constexpr int LDX = BK + 8;  // smem BF16 per token row: 144 bytes (conflict-free ldmatrix)
template <int MI, int NT> __host__ __device__ constexpr int stage_bytes() {
    return 128 * MI * LDW + 8 * NT * LDX * 2;
}
template <int MI, int NT, int ST = STAGES> __host__ __device__ constexpr int smem_bytes() {
    return ST * stage_bytes<MI, NT>();  // 90 KiB for MI = 2, NT = 16, three stages
}
static_assert(smem_bytes<2, 16>() <= 96 * 1024, "shared memory");

__device__ __forceinline__ void cp_async4(void * s, const void * g, bool valid) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(a), "l"(g), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void cp_async16(void * s, const void * g, bool valid) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(valid ? 16 : 0));
}

// The first `bytes` (0 to 16) of a 16-byte word; the rest of the shared word is zeroed.
__device__ __forceinline__ void cp_async16_n(void * s, const void * g, int bytes) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(bytes));
}

// A UE4M3 sub-block scale times 2^116 as a BF16 pair (ggml: sign ignored, 0x7f is 0).
__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t d) {
    d &= 0x7f;
    // Normal: exponent (d >> 3) - 7 + 116, mantissa d & 7; subnormal: (d & 7) * 2^-9 * 2^116.
    const uint32_t normal = (d << 4) + 0x7600;
    const uint32_t sub = __float_as_uint((float) d * 0x1p107f) >> 16;
    const uint32_t b = d == 0x7f ? 0 : d < 8 ? sub : normal;
    return b | (b << 16);
}

// The four codes of `z` (two code bytes in bits 0-7 and 16-23) as BF16 pairs times 2^-126:
// lo = (first, second byte's low nibble), hi = (first, second byte's high nibble).
__device__ __forceinline__ void codes_bf16x2(uint32_t z, uint32_t & lo, uint32_t & hi) {
    lo = ((z << 6) & 0x01c001c0u) | ((z << 12) & 0x80008000u);
    hi = ((z << 2) & 0x01c001c0u) | ((z << 8) & 0x80008000u);
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t s) {
    __nv_bfloat162 v = __hmul2(*(const __nv_bfloat162 *) &a, *(const __nv_bfloat162 *) &s);
    return *(const uint32_t *) &v;
}

__device__ __forceinline__ void mma(float * c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One output element with the epilogue's activation (see legion_nvfp4_mm::act).
__device__ __forceinline__ void store_out(const legion_nvfp4_mm & p, int64_t i, float v) {
    if (p.act == 1) {
        const float r = fmaxf(v, 0.0f);
        ((__nv_bfloat16 *) p.dst)[i] = __float2bfloat16_rn(r * r);
    } else {
        p.dst[i] = v;
    }
}

// The dense fused gate / up kernels' activation of the gate (legion_nvfp4_mm::act of a GLU
// launch): 0 silu, as ggml-cuda's swiglu; NVFP4_GLU_GELU the tanh GELU, as ggml-cuda's geglu
// (ggml_cuda_op_gelu_single in ggml-cuda/unary.cuh, MIT): the same expressions, so with the
// same flags (fast math both) the same bits.
constexpr int NVFP4_GLU_GELU = 2;
__device__ __forceinline__ float glu_gate(float x, int act) {
    if (act == NVFP4_GLU_GELU) {
        const float GELU_COEF_A = 0.044715f;
        const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
    return x / (1.0f + expf(-x));
}

// Grouped tiles: token tile `tile` of 'bn' pairs is in expert e when it is among the
// ceil(count_e / bn) tiles that follow the earlier experts' tiles. One warp finds it: lane l
// counts the tiles of experts [l * per, (l + 1) * per) with independent loads of the plan's
// offsets, one warp scan of the 32 counts, then the lane holding `tile` walks its experts
// again (cached). Returns false past the last tile. `first` and `count`: the tile's pairs in
// the plan's list. (A running scan of 32 experts at a time costs n_expert / 32 dependent
// rounds per CTA: with 512 experts, a large part of a decode tile's time.) `total` (if set):
// the number of tiles, written when `tile` is past the last one. Only the experts with more than
// `sel_lo` and at most `sel_hi` pairs have tiles (as legion_nvfp4_mm's).
__device__ __forceinline__ int expert_tiles(int pairs, int bn, int lo, int hi) {
    return pairs > lo && pairs <= hi ? (pairs + bn - 1) / bn : 0;
}

__device__ __forceinline__ bool find_tile(const int32_t * offs, int n_expert, int bn, int tile, int lane, int sel_lo,
                                          int sel_hi, int & expert, int & first, int & count, int * total = nullptr) {
    const int per = (n_expert + 31) / 32;
    const int e0 = min(lane * per, n_expert), e1 = min(e0 + per, n_expert);
    int t = 0;
#pragma unroll 8
    for (int e = e0; e < e1; e++) {
        t += expert_tiles(offs[e + 1] - offs[e], bn, sel_lo, sel_hi);
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
        if (total) {  // the same pointer in every lane
            *total = __shfl_sync(0xffffffffu, incl, 31);
        }
        return false;
    }
    const int src = __ffs(hit) - 1;
    int found = 0, f = 0, c = 0;
    if (lane == src) {
        int before = lo;
        for (int e = e0; e < e1; e++) {
            const int a = offs[e], b = offs[e + 1];
            const int te = expert_tiles(b - a, bn, sel_lo, sel_hi);
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

// 128 * MI rows x 8 * NT tokens per CTA; 8 warps of 16 * MI rows x all tokens, so each weight is
// decoded once per CTA and each decoded A fragment feeds NT n8 MMAs. K steps of one NVFP4 block
// (64) through three cp.async stages. GROUPED: the CTA's expert and tokens come from the plan.
// GLU: MI is 2 and the CTA covers 128 rows of the gate and the same rows of the up weight
// (p.w2): each warp's first 16-row tile from the gate, its second from the up weight (GROUPED:
// both the tile's expert, each times its own per-expert scale).
// A tile's n8 token groups past its last token are skipped (grouped tiles are often partly
// filled): the other tokens' sums are the same instructions on the same values. A full tile
// runs without those checks.
// ST: cp.async stages; MINB: CTAs per SM the registers are limited for (2: two CTAs of up to
// ~49 KiB of stages share an SM, one's copies and epilogue overlapping the other's MMAs). Neither
// changes any sum: every output is the same MMAs over k in the same order (#824).
template <int MI, int NT, bool GROUPED, bool GLU, int ST = STAGES, int MINB = 1>
__global__ void __launch_bounds__(256, MINB) nvfp4_gemm(const legion_nvfp4_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(!GLU || MI == 2, "SwiGLU: gate and up tiles");
    constexpr int BM = 128 * MI, BN = 8 * NT, W_STAGE = BM * LDW, STAGE = stage_bytes<MI, NT>();
    constexpr int BM_OUT = GLU ? BM / 2 : BM;  // output rows per tile
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int64_t x_off[GROUPED ? BN : 1], d_off[GROUPED ? BN : 1];
    __shared__ int32_t tile_expert, tile_tokens;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    int64_t m0, n0 = 0, n_tok;
    const uint8_t * w = p.w;
    const uint8_t * w2 = p.w2;
    float sc = (p.scale ? *p.scale : 1.0f) * 1024.0f;  // 2^10 undoes the decoding's 2^-10
    float sc2 = (p.scale2 ? *p.scale2 : 1.0f) * 1024.0f;
    if (!GROUPED) {
        // Groups of group_m row tiles at a time, token tiles inside a group, so the CTAs resident
        // at once share both weight rows and activation tokens through L2.
        const int pid = blockIdx.x, in_group = p.group_m * p.tiles_n;
        const int first_m = pid / in_group * p.group_m;
        const int group = min(p.tiles_m - first_m, p.group_m);
        m0 = (int64_t) (first_m + pid % in_group % group) * BM_OUT;
        n0 = (int64_t) (pid % in_group / group) * BN;
        n_tok = min((int64_t) BN, p.n - n0);
    } else {
        // Row tiles vary fastest: the CTAs resident at once share an expert's tokens.
        m0 = (int64_t) (blockIdx.x % p.tiles_m) * BM_OUT;
        const int tile = blockIdx.x / p.tiles_m;
        // Past the routing's last tile (the host grid is an upper bound, about a third of it
        // at prefill chunks over 512 experts): nothing to do, so leave before the plan scan
        // (#836); but first store zeros for the plan's skipped pairs (ids of -1, listed after
        // the experts' pairs, nvfp4_plan), which no tile holds (#1176): the CTAs past the last
        // tile split them, BM_OUT rows each, so a skipped pair's result is zero without
        // reading any expert. A plan without such pairs stores nothing here.
        constexpr int TI = legion_nvfp4::plan_tile_index(BN);
        if (legion_nvfp4::past_plan_tiles<TI>(p, tile)) {
            const int routed = p.plan[p.n_expert];
            const int skipped = (int) p.n - routed;
            if (skipped > 0 && !p.unread_skips) {
                const int tiles = p.plan[p.n_expert + 1 + p.n + TI];
                const int spare = tile - tiles, n_spare = (int) (gridDim.x / p.tiles_m) - tiles;
                const int64_t rows = min((int64_t) BM_OUT, p.m - m0);
                for (int j = spare; j < skipped; j += n_spare) {
                    const int64_t off = (int64_t) p.plan[p.n_expert + 1 + routed + j] * p.m + m0;
                    for (int64_t r = threadIdx.x; r < rows; r += 256) {
                        if (GLU) {
                            ((__nv_bfloat16 *) p.dst)[off + r] = __float2bfloat16_rn(0.0f);
                        } else {
                            store_out(p, off + r, 0.0f);
                        }
                    }
                }
            }
            return;
        }
        if (warp == 0) {
            int e, first, count;
            if (!find_tile(p.plan, p.n_expert, BN, tile, lane, p.sel_lo, p.sel_hi, e, first, count)) {
                if (lane == 0) {
                    tile_expert = -1;
                }
            } else {
                if (lane == 0) {
                    tile_expert = e;
                    tile_tokens = count;
                }
                for (int i = lane; i < count; i += 32) {
                    const int q = p.plan[p.n_expert + 1 + first + i];  // token * n_used + slot
                    const int t = q / p.n_used, slot = q % p.n_used;
                    x_off[i] = ((int64_t) t * p.n_x + (p.n_x == 1 ? 0 : slot)) * p.x_ld;
                    d_off[i] = (int64_t) q * p.m;
                }
            }
        }
        __syncthreads();
        if (tile_expert < 0) {
            return;
        }
        w += (int64_t) tile_expert * p.w_e;
        sc = (p.scale ? p.scale[tile_expert] : 1.0f) * 1024.0f;
        if (GLU) {
            w2 += (int64_t) tile_expert * p.w_e;
            sc2 = (p.scale2 ? p.scale2[tile_expert] : 1.0f) * 1024.0f;
        }
        n_tok = tile_tokens;
    }
    const int64_t steps = p.k / BK;

    // Shared row r's weight row (rows past m: row 0, never stored).
    auto row_src = [&](int r) -> const uint8_t * {
        const int64_t row = GLU ? m0 + r / 32 * 16 + r % 16 : m0 + r;
        const uint8_t * wt = GLU && r / 16 % 2 ? w2 : w;
        return wt + (row < p.m ? row : 0) * p.w_ld;
    };
    auto row_ok = [&](int r) { return (GLU ? m0 + r / 32 * 16 + r % 16 : m0 + r) < p.m; };

    auto load = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * STAGE;
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (ws + W_STAGE);
        // Each row's block as the three aligned 16-byte words around it (a third as many copies
        // as 4-byte ones: the copies' instructions, not the bytes, held the GEMM back). The last
        // word stops at the block's end, so nothing past it is read; the bytes before the block
        // in the first word are the same row's (or, at its first block, the previous row's).
#pragma unroll
        for (int i = 0; i < (BM * 3 + 255) / 256; i++) {
            const int c = threadIdx.x + i * 256;
            if (BM * 3 % 256 == 0 || c < BM * 3) {
                const int r = c / 3, part = c % 3;
                const uint8_t * start = row_src(r) + step * 36;
                const uint8_t * src = (const uint8_t *) ((uintptr_t) start & ~(uintptr_t) 15) + part * 16;
                const int bytes = row_ok(r) ? min(16, (int) (start + 36 - src)) : 0;
                cp_async16_n(ws + r * LDW + part * 16, src, bytes);
            }
        }
#pragma unroll
        for (int i = 0; i < (BN * 8 + 255) / 256; i++) {
            const int c = threadIdx.x + i * 256;
            if (BN * 8 % 256 == 0 || c < BN * 8) {
                const int t = c / 8, col = c % 8 * 8;
                const bool ok = t < n_tok;
                const int64_t off = !ok ? 0 : GROUPED ? x_off[t] : (n0 + t) * p.x_ld;
                cp_async16(xs + t * LDX + col, p.x + off + step * BK + col, ok);
            }
        }
    };

    // Each row's next PF blocks in one L2 bulk prefetch, PF blocks ahead of the copies: the
    // copies read one block (36 bytes) of a row per step, too short a run for the LPDDR5X
    // (see nvfp4_mvs).
    constexpr int PF = 32;
    auto prefetch = [&](int64_t step) {
#if __CUDA_ARCH__ >= 900
        if (threadIdx.x < BM && step < steps) {
            const int r = threadIdx.x;
            const int64_t row = GLU ? m0 + r / 32 * 16 + r % 16 : m0 + r;
            if (row < p.m) {
                const uint8_t * wt = GLU && r / 16 % 2 ? w2 : w;
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + step * 36);
                const uintptr_t e = a + min((int64_t) PF, steps - step) * 36;
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
            }
        }
#endif
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

    // This lane's rows: (i, h) = row warp * 16 * MI + i * 16 + h * 8 + lane / 4, and where in its
    // first shared word each row's blocks start (the row start's offset, then 4 bytes on per step).
    uint32_t row_off[MI][2];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            row_off[i][h] = (uint32_t) (uintptr_t) row_src(warp * 16 * MI + i * 16 + h * 8 + lane / 4);
        }

    // One stage's MMAs. FULL: a tile of BN tokens (no per-n8 checks; the same MMAs as with them).
    auto mma_stage = [&](const unsigned char * ws, const __nv_bfloat16 * xs, int64_t step, auto full) {
        constexpr bool FULL = decltype(full)::value;
        // Each lane of a quad converts one of the row's four sub-block scales (lane % 4); the
        // quad shares them.
        const unsigned char * rows[MI][2];
        uint32_t ssc[MI][2];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                rows[i][h] = ws + (warp * 16 * MI + i * 16 + h * 8 + lane / 4) * LDW +
                             ((row_off[i][h] + (uint32_t) step * 36) & 15);
                ssc[i][h] = scale_bf16x2(rows[i][h][lane % 4]);
            }
#pragma unroll
        for (int s = 0; s < 4; s++) {
            uint32_t a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t q = *(const uint16_t *) (rows[i][h] + 4 + 8 * s + 2 * (lane % 4));
                    const uint32_t d = __shfl_sync(0xffffffffu, ssc[i][h], (lane & ~3) | s);
                    uint32_t lo, hi;
                    codes_bf16x2(__byte_perm(q, 0, 0x4140), lo, hi);
                    a[i][h] = mul_bf16x2(lo, d);      // a0a1 (h = 0) / a2a3 (h = 1)
                    a[i][2 + h] = mul_bf16x2(hi, d);  // a4a5 / a6a7
                }
            // B fragments 8 n8 tiles at a time (ldmatrix x4: two n8 tiles; x2 for a single one).
#pragma unroll
            for (int half = 0; half < (NT + 7) / 8; half++) {
                if (!FULL && half * 64 >= n_tok) {
                    break;
                }
                constexpr int JN = NT < 8 ? NT : 8;
                uint32_t b[JN][2];
#pragma unroll
                for (int j = 0; j < JN / 2; j++) {
                    if (!FULL && half * 64 + j * 16 >= n_tok) {
                        break;
                    }
                    const int tok = half * 64 + j * 16 + lane % 8 + (lane / 16) * 8;
                    const __nv_bfloat16 * src = xs + tok * LDX + s * 16 + ((lane / 8) % 2) * 8;
                    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b[2 * j][0]), "=r"(b[2 * j][1]), "=r"(b[2 * j + 1][0]), "=r"(b[2 * j + 1][1])
                        : "r"(addr));
                }
                if (JN == 1) {
                    const __nv_bfloat16 * src = xs + (lane % 8) * LDX + s * 16 + ((lane / 8) % 2) * 8;
                    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b[0][0]), "=r"(b[0][1])
                        : "r"(addr));
                }
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < JN; j++) {
                        if (FULL || half * 64 + j * 8 < n_tok) {
                            mma(acc[i][half * 8 + j], a[i], b[j][0], b[j][1]);
                        }
                    }
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
        const unsigned char * ws = smem + (step % ST) * STAGE;
        const __nv_bfloat16 * xs = (const __nv_bfloat16 *) (ws + W_STAGE);
        if (n_tok == BN) {
            mma_stage(ws, xs, step, std::true_type{});
        } else {
            mma_stage(ws, xs, step, std::false_type{});
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    if constexpr (GLU) {
        const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8;
                const int t = j * 8 + (lane % 4) * 2 + e % 2;
                if (row < p.m && t < n_tok) {
                    float gate = acc[0][j][e] * sc, up = acc[1][j][e] * sc2;
                    if (p.limit > 0.0f) {
                        gate = fminf(gate, p.limit);
                        up = fminf(fmaxf(up, -p.limit), p.limit);
                    }
                    const int64_t off = GROUPED ? d_off[t] : (n0 + t) * p.m;
                    ((__nv_bfloat16 *) p.dst)[off + row] = __float2bfloat16_rn(glu_gate(gate, p.act) * up);
                }
            }
        }
    } else {
#pragma unroll
    for (int i = 0; i < MI; i++) {
        const int64_t r0 = m0 + warp * 16 * MI + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8;
                const int t = j * 8 + (lane % 4) * 2 + e % 2;
                if (row < p.m && t < n_tok) {
                    const int64_t off = GROUPED ? d_off[t] : (n0 + t) * p.m;
                    store_out(p, off + row, acc[i][j][e] * sc);
                }
            }
        }
    }
    }
#endif
}

// The mv kernel: 16 * RTW rows x 8 * NT tokens per CTA, KW warps splitting each stage's blocks
// (warp i takes blocks i, i + KW, ...). The CTA copies its rows' weights into shared memory a
// stage of MVS_BLOCKS blocks at a time with coalesced cp.async (consecutive threads, consecutive
// words of a row: whole cache lines per warp), MVS_STAGES stages in flight, and PF: an L2
// prefetch further ahead. GB10's LPDDR5X streams well only in long runs of each row (~1 KiB):
// loading the fragments straight from global memory (8 rows per warp instruction) reaches
// ~120 GB/s, short stages ~190, these ~210-220 (ggml's MMVQ ~235). The warps read their
// fragments from shared memory (the k permutation in the header); activations through L1, one
// block ahead. The KW warps' sums are added in a fixed order. A16: rows and stages 16-byte
// aligned, 16-byte copies. GROUPED: tiles of one expert's pairs.
template <int SB> __host__ __device__ constexpr int mvs_ld() {
    return SB * 36 + 16;
}
template <int RTW, int SB, int NS> __host__ __device__ constexpr int mvs_smem() {
    return NS * 16 * RTW * mvs_ld<SB>();
}
// The KW warps' partial sums of one row tile (RPT > 1: a buffer of its own, after the stages,
// since the next row tile's stages are in flight while this one is summed).
template <int NT, int RTW, int KW> __host__ __device__ constexpr int mvs_red() {
    return KW * RTW * NT * 4 * 32 * 4;
}

// GLU: RTW is 2, row tile 0 is 16 rows of the gate, tile 1 the same rows of the up weight.
// RPT: row tiles per CTA, one after the other through the same stage ring (the next tile's
// first stages load while this one is summed). Every row tile's arithmetic is the RPT = 1
// kernel's (the same blocks per warp in the same order, the same warp order in the sum), so
// the outputs are the same bits; a CTA finds its expert tile once for RPT tiles of rows, and
// the grid has RPT times fewer CTAs (fewer that only find out they have no tile).
// IGLU (grouped SwiGLU, RTW 1): a row tile is 8 rows of the expert's gate (p.w) in fragment rows
// 0-7 and the same 8 rows of its up weight (p.w2) in rows 8-15, so a lane holds a row's gate and
// up sums side by side; each row's sums are the plain kernel's (the same blocks per warp in the
// same order, the same warp order), and the epilogue stores silu(gate) * up as BF16, the bits of
// the graph's chain (two grouped mvs, ggml's silu and mul, the cast to BF16).
// PFT (grouped, above 0): instead of PF's per-stage prefetch of each row, the L2 prefetch takes
// whole row tiles (the tile's rows of an expert are contiguous: one bulk prefetch per weight),
// PFT tiles ahead of the one being summed, so DRAM reads each expert's rows in one long run.
// Programmatic dependent launch (sm_90+, grouped): the grouped SwiGLU mv lets the kernel after
// it start as soon as all its CTAs run (griddepcontrol.launch_dependents), and every grouped
// kernel waits for the kernel ahead (griddepcontrol.wait) only before it reads the activations:
// the down mv after a SwiGLU mv finds its tile in the plan (written before the SwiGLU mv
// started), copies its first weight stages and prefetches while the SwiGLU mv's last CTAs run.
// Neither changes a sum.
template <int NT, int RTW, int KW, bool GROUPED, bool A16, int MVS_BLOCKS, int MVS_STAGES, bool PF, bool GLU,
          int RPT = 1, bool IGLU = false, int PFT = 0>
__global__ void __launch_bounds__(KW * 32) nvfp4_mvs(const legion_nvfp4_mm p) {
    static_assert(!GLU || (RTW == 2 && !GROUPED), "SwiGLU: dense, gate and up tiles");
    static_assert(RPT == 1 || !GLU, "SwiGLU: one row tile per CTA");
    static_assert(!IGLU || (RTW == 1 && GROUPED && !GLU && KW > 1), "grouped SwiGLU: 8 + 8 row tiles");
    static_assert(PFT == 0 || (GROUPED && !PF), "tile prefetch: grouped, instead of PF");
    constexpr int MVS_LD = mvs_ld<MVS_BLOCKS>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (IGLU) {
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    }
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BN = 8 * NT, ROWS = 16 * RTW, THREADS = KW * 32;
    // Output rows per row tile.
    constexpr int OUT_ROWS = GLU ? 16 : IGLU ? 8 : ROWS;
    constexpr int STAGE = ROWS * MVS_LD;
    constexpr int FRAG = RTW * NT * 4;
    static_assert(MVS_BLOCKS % KW == 0, "stage split");
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int64_t x_off[GROUPED ? BN : 1], d_off[GROUPED ? BN : 1];
    __shared__ int32_t tile_expert, tile_tokens, tile_total;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;

    const int64_t m0 = (int64_t) blockIdx.x * OUT_ROWS * RPT;
    const uint8_t * w = p.w;
    const uint8_t * w2 = p.w2;
    float sc = (p.scale ? *p.scale : 1.0f) * 1024.0f;
    int n_tok;
    if (!GROUPED) {
        n_tok = (int) min((int64_t) BN, p.n - (int64_t) blockIdx.y * BN);
    } else {
        // The host grid is an upper bound (no host sync for the routing): a CTA past the plan's
        // own tile count with no padding pairs to store (none, or unread_skips) has nothing to
        // do, and leaves before the plan scan (one load instead of find_tile's; #836). Selected
        // launches (a split's) scan as before.
        if (legion_nvfp4::past_plan_tiles<legion_nvfp4::plan_tile_index(BN)>(p, blockIdx.y) &&
            (p.plan[p.n_expert] == (int) p.n || p.unread_skips)) {
            return;
        }
        if (warp == 0) {
            int e, first, count, total = 0;
            if (!find_tile(p.plan, p.n_expert, BN, blockIdx.y, lane, p.sel_lo, p.sel_hi, e, first, count,
                           &total)) {
                if (lane == 0) {
                    tile_expert = -1;
                    tile_total  = total;
                }
            } else {
                if (lane == 0) {
                    tile_expert = e;
                    tile_tokens = count;
                }
                if (lane < count) {
                    const int q = p.plan[p.n_expert + 1 + first + lane];
                    const int tk = q / p.n_used, slot = q % p.n_used;
                    x_off[lane] = ((int64_t) tk * p.n_x + (p.n_x == 1 ? 0 : slot)) * p.x_ld;
                    d_off[lane] = (int64_t) q * p.m;
                }
            }
        }
        __syncthreads();
        if (tile_expert < 0) {
            // A plan's skipped padding pairs (listed after the experts' pairs, nvfp4_plan): the
            // CTAs past the last tile store zeros for them, so a padding token's result is
            // finite (zero) without reading any expert. The host grids keep one spare tile.
            const int routed = p.plan[p.n_expert];
            const int pad = (int) p.n - routed, spare = (int) blockIdx.y - tile_total;
            if (pad > 0 && spare >= 0 && !p.unread_skips) {
                // The kernel ahead may still read memory this result reuses (see the PDL
                // note above): stores only after it is done, as the tiles' epilogues.
#if __CUDA_ARCH__ >= 900
                asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
                const int n_spare = (int) gridDim.y - tile_total;
                const int64_t rows = min((int64_t) OUT_ROWS * RPT, p.m - m0);
                for (int j = spare; j < pad; j += n_spare) {
                    const int64_t off = (int64_t) p.plan[p.n_expert + 1 + routed + j] * p.m + m0;
                    for (int64_t r = threadIdx.x; r < rows; r += THREADS) {
                        if (IGLU) {
                            ((__nv_bfloat16 *) p.dst)[off + r] = __float2bfloat16_rn(0.0f);
                        } else {
                            store_out(p, off + r, 0.0f);
                        }
                    }
                }
            }
            return;
        }
        w += (int64_t) tile_expert * p.w_e;
        sc = (p.scale ? p.scale[tile_expert] : 1.0f) * 1024.0f;
        if (IGLU) {
            w2 += (int64_t) tile_expert * p.w_e;
        }
        n_tok = tile_tokens;
    }

    const int nblk = (int) (p.k / BK);
    const int nst = (nblk + MVS_BLOCKS - 1) / MVS_BLOCKS;
    // Units: (row tile, stage) pairs, row tile major; with RPT = 1 a unit is a stage. Row tiles
    // wholly past the last row are skipped.
    const int ntile = RPT == 1 ? 1 : (int) min((int64_t) RPT, (p.m - m0 + OUT_ROWS - 1) / OUT_ROWS);
    const int nu = ntile * nst;
    // Row r of row tile `tile` of the CTA's weights (past the last row, the last one).
    auto row_ptr = [&](int tile, int r) {
        const int64_t mt = m0 + (int64_t) tile * OUT_ROWS;
        if (GLU) {
            return (r < 16 ? w : p.w2) + min(mt + r % 16, p.m - 1) * p.w_ld;
        }
        if (IGLU) {
            return (r < 8 ? w : w2) + min(mt + r % 8, p.m - 1) * p.w_ld;
        }
        return w + min(mt + r, p.m - 1) * p.w_ld;
    };
    auto load = [&](int u) {
        const int tile = RPT == 1 ? 0 : u / nst, st = RPT == 1 ? u : u % nst;
        unsigned char * dst = smem + (u % MVS_STAGES) * STAGE;
        const int blocks = min(MVS_BLOCKS, nblk - st * MVS_BLOCKS);
        const int64_t base = (int64_t) st * MVS_BLOCKS * 36;
        if (A16) {
            constexpr int CH = MVS_BLOCKS * 36 / 16;  // 16-byte chunks per row
            const int ch = (blocks * 36 + 15) / 16;
#pragma unroll
            for (int i = 0; i < (ROWS * CH + THREADS - 1) / THREADS; i++) {
                const int c = threadIdx.x + i * THREADS;
                const int r = c / CH, q = c % CH;
                if (r < ROWS) {
                    cp_async16(dst + r * MVS_LD + q * 16, row_ptr(tile, r) + base + q * 16, q < ch);
                }
            }
        } else {
            constexpr int CH = MVS_BLOCKS * 9;  // words per row
            const int ch = blocks * 9;
#pragma unroll 4
            for (int i = 0; i < (ROWS * CH + THREADS - 1) / THREADS; i++) {
                const int c = threadIdx.x + i * THREADS;
                const int r = c / CH, q = c % CH;
                if (r < ROWS) {
                    cp_async4(dst + r * MVS_LD + q * 4, row_ptr(tile, r) + base + q * 4, q < ch);
                }
            }
        }
    };

    // An L2 prefetch of unit `u` of each row, issued MVS_STAGES units before its copy: the
    // copies then hit L2, and the DRAM reads are whole stages of a row at a time.
    auto prefetch = [&](int u) {
        if (threadIdx.x < ROWS && u < nu) {
            const int tile = RPT == 1 ? 0 : u / nst, st = RPT == 1 ? u : u % nst;
            const uintptr_t a = (uintptr_t) (row_ptr(tile, threadIdx.x) + (int64_t) st * MVS_BLOCKS * 36);
            const uintptr_t e = a + min(MVS_BLOCKS, nblk - st * MVS_BLOCKS) * 36;
            const uintptr_t a16 = a & ~(uintptr_t) 15;
#if __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16), "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
#endif
        }
    };
    // PFT: row tile `tile` of the CTA, whole (IGLU: its gate rows by thread 0, its up rows by
    // thread 1).
    auto prefetch_tile = [&](int tile) {
        if (PFT > 0 && tile < ntile && threadIdx.x < (IGLU ? 2 : 1)) {
            const int64_t mt = m0 + (int64_t) tile * OUT_ROWS;
            const int64_t rows = min((int64_t) OUT_ROWS, p.m - mt);
            const uintptr_t a = (uintptr_t) ((IGLU && threadIdx.x == 1 ? w2 : w) + mt * p.w_ld);
            const uintptr_t a16 = a & ~(uintptr_t) 15;
            const uintptr_t e = a + (uintptr_t) (rows * p.w_ld);
#if __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16), "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
#endif
        }
    };

    const uint4 * xc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int c = nt * 8 + g < n_tok ? nt * 8 + g : 0;
        const int64_t off = GROUPED ? x_off[c] : ((int64_t) blockIdx.y * BN + c) * p.x_ld;
        xc[nt] = (const uint4 *) (p.x + off) + 2 * t;
    }

    float acc[RTW][NT][4];
#pragma unroll
    for (int r = 0; r < RTW; r++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[r][j][e] = 0.0f;

    if (PF) {
        for (int s = MVS_STAGES - 1; s < 2 * MVS_STAGES - 1; s++) {
            prefetch(s);
        }
    }
    if (PFT > 0) {
        for (int tile = 0; tile <= PFT; tile++) {
            prefetch_tile(tile);
        }
    }
    uint4 na[NT], nb[NT];  // the activations of this warp's next block
    auto first_x = [&]() {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            if (warp < nblk) {
                na[nt] = __ldg(xc[nt] + (int64_t) warp * 8);
                nb[nt] = __ldg(xc[nt] + (int64_t) warp * 8 + 1);
            }
        }
    };
    if (!GROUPED) {
        first_x();
    }
#pragma unroll
    for (int s = 0; s < MVS_STAGES - 1; s++) {
        if (s < nu) {
            load(s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    if (GROUPED) {
        // The activations are the kernel ahead's output (see the programmatic launch above).
#if __CUDA_ARCH__ >= 900
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
        first_x();
    }

    // GLU: f in tile 0 (the gate), v2 the up weight's accumulator of the same row and token.
    // IGLU: the expert's up scale.
    const float sc2 = IGLU ? (p.scale2 ? p.scale2[GROUPED ? tile_expert : 0] : 1.0f) * 1024.0f
                    : GLU && p.scale2 ? *p.scale2 * 1024.0f
                                      : 1024.0f;
    // The sums of row tile `tile`: KW > 1, the warps' partials added in warp order (RPT = 1: in
    // the stages' memory, once every warp is done with them; RPT > 1: in a buffer of their own).
    auto epilogue = [&](int tile) {
        const int64_t mt = m0 + (int64_t) tile * OUT_ROWS;
        if constexpr (IGLU) {
            float * red;
            if constexpr (RPT == 1) {
                __syncthreads();  // every warp is done with the stages
                red = (float *) smem;
            } else {
                red = (float *) (smem + mvs_smem<RTW, MVS_BLOCKS, MVS_STAGES>());
            }
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) red[(warp * FRAG + j * 4 + e) * 32 + lane] = acc[0][j][e];
            __syncthreads();
            // Output (j, e < 2): the gate's sums are fragment j * 4 + e (row ln / 4), the up
            // weight's the same row's, fragment j * 4 + e + 2.
            for (int i = threadIdx.x; i < NT * 2 * 32; i += THREADS) {
                const int f = i / 32, ln = i % 32, j = f / 2, e = f % 2;
                const int fg = (j * 4 + e) * 32 + ln, fu = fg + 2 * 32;
                float g = red[fg], u = red[fu];
#pragma unroll
                for (int q = 1; q < KW; q++) {
                    g += red[q * FRAG * 32 + fg];
                    u += red[q * FRAG * 32 + fu];
                }
                const int64_t row = mt + ln / 4;
                const int tok = j * 8 + (ln % 4) * 2 + e;
                if (row < p.m && tok < n_tok) {
                    float gate = g * sc, up = u * sc2;
                    if (p.limit > 0.0f) {
                        gate = fminf(gate, p.limit);
                        up = fminf(fmaxf(up, -p.limit), p.limit);
                    }
                    ((__nv_bfloat16 *) p.dst)[d_off[tok] + row] =
                        __float2bfloat16_rn(gate / (1.0f + expf(-gate)) * up);
                }
            }
            return;
        }
        auto store = [&](int f, int ln, float v, float v2) {
            const int e = f % 4, j = f / 4 % NT, r = f / 4 / NT;
            const int64_t row = mt + r * 16 + (e / 2) * 8 + ln / 4;
            const int tok = j * 8 + (ln % 4) * 2 + e % 2;
            if (row < p.m && tok < n_tok) {
                const int64_t off = GROUPED ? d_off[tok] : ((int64_t) blockIdx.y * BN + tok) * p.m;
                if (GLU) {
                    const float gate = v * sc;
                    ((__nv_bfloat16 *) p.dst)[off + row] = __float2bfloat16_rn(glu_gate(gate, p.act) * (v2 * sc2));
                } else {
                    store_out(p, off + row, v * sc);
                }
            }
        };
        constexpr int OUT = GLU ? FRAG / 2 : FRAG;  // accumulators that are stored (GLU: the gate's)
        if constexpr (KW == 1) {
#pragma unroll
            for (int r = 0; r < RTW; r++)
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        if (!GLU || r == 0) {
                            store((r * NT + j) * 4 + e, lane, acc[r][j][e], acc[RTW - 1][j][e]);
                        }
                    }
        } else {
            float * red;
            if constexpr (RPT == 1) {
                __syncthreads();  // every warp is done with the stages
                red = (float *) smem;
            } else {
                red = (float *) (smem + mvs_smem<RTW, MVS_BLOCKS, MVS_STAGES>());
            }
#pragma unroll
            for (int r = 0; r < RTW; r++)
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
        }
    };

    for (int u = 0; u < nu; u++) {
        const int tile = RPT == 1 ? 0 : u / nst, st = RPT == 1 ? u : u % nst;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(MVS_STAGES - 2));
        __syncthreads();
        if (u + MVS_STAGES - 1 < nu) {
            load(u + MVS_STAGES - 1);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (PF) {
            prefetch(u + 2 * MVS_STAGES - 1);
        }
        if (PFT > 0 && st == 0 && tile > 0) {
            prefetch_tile(tile + PFT);
        }
        const unsigned char * ws = smem + (u % MVS_STAGES) * STAGE;
#pragma unroll
        for (int i = 0; i < MVS_BLOCKS / KW; i++) {
            const int bl = warp + i * KW, b = st * MVS_BLOCKS + bl;
            if (b >= nblk) {
                break;
            }
            // This block's activations (loaded one block ahead), and the next block's.
            uint4 xa[NT], xb[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                xa[nt] = na[nt];
                xb[nt] = nb[nt];
                if (b + KW < nblk) {
                    na[nt] = __ldg(xc[nt] + (int64_t) (b + KW) * 8);
                    nb[nt] = __ldg(xc[nt] + (int64_t) (b + KW) * 8 + 1);
                }
            }
            uint32_t d[RTW][2], c0[RTW][2], c1[RTW][2];
#pragma unroll
            for (int r = 0; r < RTW; r++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t * q = (const uint32_t *) (ws + (r * 16 + h * 8 + g) * MVS_LD + bl * 36);
                    d[r][h] = scale_bf16x2(q[0] >> (8 * t));
                    c0[r][h] = q[1 + 2 * t];
                    c1[r][h] = q[2 + 2 * t];
                }
#pragma unroll
            for (int s = 0; s < 4; s++) {
                uint32_t a[RTW][4];
#pragma unroll
                for (int r = 0; r < RTW; r++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const uint32_t cw = s < 2 ? c0[r][h] : c1[r][h];
                        uint32_t lo, hi;
                        codes_bf16x2(__byte_perm(cw, 0, s % 2 == 0 ? 0x4140 : 0x4342), lo, hi);
                        a[r][h] = mul_bf16x2(lo, d[r][h]);
                        a[r][2 + h] = mul_bf16x2(hi, d[r][h]);
                    }
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const uint32_t b0w = s == 0 ? xa[nt].x : s == 1 ? xa[nt].y : s == 2 ? xa[nt].z : xa[nt].w;
                    const uint32_t b1w = s == 0 ? xb[nt].x : s == 1 ? xb[nt].y : s == 2 ? xb[nt].z : xb[nt].w;
#pragma unroll
                    for (int r = 0; r < RTW; r++) {
                        mma(acc[r][nt], a[r], b0w, b1w);
                    }
                }
            }
        }
        if (RPT > 1 && st == nst - 1 && tile < ntile - 1) {
            // The next row tile: the same activations from block 0 again, fresh sums.
            epilogue(tile);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                if (warp < nblk) {
                    na[nt] = __ldg(xc[nt] + (int64_t) warp * 8);
                    nb[nt] = __ldg(xc[nt] + (int64_t) warp * 8 + 1);
                }
            }
#pragma unroll
            for (int r = 0; r < RTW; r++)
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[r][j][e] = 0.0f;
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    epilogue(ntile - 1);
#endif
}

int status() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}

template <int MI, int NT, bool GLU = false>
int gemm(const legion_nvfp4_mm & p, cudaStream_t st) {
    constexpr int SMEM = smem_bytes<MI, NT>();
    const unsigned grid = (unsigned) p.tiles_m * (unsigned) p.tiles_n;
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        kernel<<<grid, 256, SMEM, st>>>(p);
    };
    if constexpr (GLU) {
        p.plan ? go(nvfp4_gemm<MI, NT, true, true>) : go(nvfp4_gemm<MI, NT, false, true>);
    } else if (p.plan) {
        go(nvfp4_gemm<MI, NT, true, false>);
    } else {
        go(nvfp4_gemm<MI, NT, false, false>);
    }
    return status();
}

// The grouped GEMM's other stage counts and occupancies (#824): ST stages, MINB CTAs per SM.
// Grouped tiles only (the routed experts' prefill). -1 where the stages do not fit.
template <int MI, int NT, bool GLU, int ST, int MINB>
int gemm_tuned(const legion_nvfp4_mm & p, cudaStream_t st) {
    constexpr int SMEM = smem_bytes<MI, NT, ST>();
    // An sm_120 / sm_121 SM has 100 KiB of shared memory, 1 KiB of it reserved per CTA; the
    // kernel's static arrays (the grouped tile's token offsets) take 16 bytes per token.
    constexpr int PER_CTA = SMEM + 16 * 8 * NT + 16 + 1024;
    if constexpr (SMEM > 99 * 1024 || MINB * PER_CTA > 100 * 1024) {
        return -1;
    } else {
        if (!p.plan) {
            return -1;
        }
        const unsigned grid = (unsigned) p.tiles_m * (unsigned) p.tiles_n;
        auto kernel = nvfp4_gemm<MI, NT, true, GLU, ST, MINB>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (MINB > 1) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
        }
        kernel<<<grid, 256, SMEM, st>>>(p);
        return status();
    }
}

// stages * 10 + CTAs per SM: 31 is the default kernel (gemm); the others for 64- and 128-token
// tiles.
template <int MI, int NT, bool GLU>
int gemm_variant(const legion_nvfp4_mm & p, int variant, cudaStream_t st) {
    if (variant == 31) {
        return gemm<MI, NT, GLU>(p, st);
    }
    if constexpr (NT == 8 || NT == 16) {
        switch (variant) {
            case 21: return gemm_tuned<MI, NT, GLU, 2, 1>(p, st);
            case 22: return gemm_tuned<MI, NT, GLU, 2, 2>(p, st);
            case 32: return gemm_tuned<MI, NT, GLU, 3, 2>(p, st);
            case 41: return gemm_tuned<MI, NT, GLU, 4, 1>(p, st);
            default: return -1;
        }
    }
    return -1;
}

template <int MI>
int gemm_nt(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    switch (nt) {
        case 1: return gemm_variant<MI, 1, false>(p, variant, st);
        case 2: return gemm_variant<MI, 2, false>(p, variant, st);
        case 4: return gemm_variant<MI, 4, false>(p, variant, st);
        case 8: return gemm_variant<MI, 8, false>(p, variant, st);
        case 16: return gemm_variant<MI, 16, false>(p, variant, st);
        default: return -1;
    }
}

// RPT: row tiles per CTA (`tiles_m` counts tiles of 16 * RTW * RPT rows).
// Whether `kernel` was built with griddepcontrol (sm_90+ code for this device), so it may be a
// programmatic dependent launch: asked once (these files' kernels are all built for the same
// architectures, and a process uses one kind of GPU).
template <typename K>
bool pdl_code(K kernel) {
    std::atomic<int> & state = legion_nvfp4::pdl_state;  // 0 unknown, 1 yes, 2 no
    int st = state.load(std::memory_order_relaxed);
    if (st == 0) {
        cudaFuncAttributes fa{};
        const cudaError_t before = cudaPeekAtLastError();
        const bool ok =
            cudaFuncGetAttributes(&fa, kernel) == cudaSuccess && fa.binaryVersion >= 90 && fa.ptxVersion >= 90;
        if (!ok && before == cudaSuccess) {
            cudaGetLastError();
        }
        st = ok ? 1 : 2;
        state.store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

template <int NT, int RTW, int KW, int SB, int NS, bool PF, bool GLU = false, int RPT = 1, bool IGLU = false,
          int PFT = 0>
int mvs(const legion_nvfp4_mm & p, cudaStream_t st) {
    const dim3 grid((unsigned) p.tiles_m, (unsigned) p.tiles_n);
    // RPT = 1: the sums are added in the stages' memory, which 64 tokens (NT = 8) outgrow.
    constexpr int STAGES_SMEM = mvs_smem<RTW, SB, NS>(), RED = KW > 1 ? mvs_red<NT, RTW, KW>() : 0;
    constexpr int SMEM = RPT > 1 ? STAGES_SMEM + RED : STAGES_SMEM > RED ? STAGES_SMEM : RED;
    if constexpr (SMEM > 99 * 1024) {
        return -1;  // more than an sm_120 / sm_121 CTA may have
    } else {
    const bool a16 = p.w_ld % 16 == 0 && (uintptr_t) p.w % 16 == 0 && (!p.plan || p.w_e % 16 == 0) &&
                     (!IGLU || (uintptr_t) p.w2 % 16 == 0);
    auto go = [&](auto kernel) {
        if (SMEM > 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        }
        if (p.pdl && p.plan && pdl_code(kernel)) {
            cudaLaunchAttribute attr{};
            attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim          = grid;
            cfg.blockDim         = dim3(KW * 32);
            cfg.dynamicSmemBytes = SMEM;
            cfg.stream           = st;
            cfg.attrs            = &attr;
            cfg.numAttrs         = 1;
            cudaLaunchKernelEx(&cfg, kernel, p);
        } else {
            kernel<<<grid, KW * 32, SMEM, st>>>(p);
        }
    };
    // PFT: grouped and without PF's prefetch.
    constexpr bool PFS = PFT > 0 ? false : PF;
    if constexpr (IGLU) {
        if (!p.plan || !p.w2) {
            return -1;
        }
        a16 ? go(nvfp4_mvs<NT, 1, KW, true, true, SB, NS, PFS, false, RPT, true, PFT>)
            : go(nvfp4_mvs<NT, 1, KW, true, false, SB, NS, PFS, false, RPT, true, PFT>);
    } else if constexpr (GLU) {
        a16 ? go(nvfp4_mvs<NT, RTW, KW, false, true, SB, NS, PF, true>) : go(nvfp4_mvs<NT, RTW, KW, false, false, SB, NS, PF, true>);
    } else if (p.plan) {
        a16 ? go(nvfp4_mvs<NT, RTW, KW, true, true, SB, NS, PFS, GLU, RPT, false, PFT>)
            : go(nvfp4_mvs<NT, RTW, KW, true, false, SB, NS, PFS, GLU, RPT, false, PFT>);
    } else if constexpr (PFT > 0) {
        return -1;
    } else {
        a16 ? go(nvfp4_mvs<NT, RTW, KW, false, true, SB, NS, PF, GLU, RPT>)
            : go(nvfp4_mvs<NT, RTW, KW, false, false, SB, NS, PF, GLU, RPT>);
    }
    return status();
    }
}

// The compiled mv variants: 16-block stages (576 bytes of a row), 2 deep, with the L2 prefetch
// (the best of the shapes tried on the GB10: 8 to 32 blocks, 2 to 4 deep, with and without the
// prefetch; tests/nvfp4_mm.rs, cuda_mv_variants).
constexpr int MV_SB = 16, MV_NS = 2;

template <int NT, int RTW>
int mv_kw(const legion_nvfp4_mm & p, int kw, int rpt, cudaStream_t st) {
    // Several row tiles per CTA: only the 8-warp, one-tile-wide kernel (grouped decode).
    if (rpt != 1 && (RTW != 1 || kw != 8)) {
        return -1;
    }
    switch (kw * 16 + rpt) {
        case 4 * 16 + 1: return mvs<NT, RTW, 4, MV_SB, MV_NS, true>(p, st);
        case 8 * 16 + 1: return mvs<NT, RTW, 8, MV_SB, MV_NS, true>(p, st);
        case 8 * 16 + 2: return mvs<NT, 1, 8, MV_SB, MV_NS, true, false, 2>(p, st);
        case 8 * 16 + 4: return mvs<NT, 1, 8, MV_SB, MV_NS, true, false, 4>(p, st);
        case 8 * 16 + 8: return mvs<NT, 1, 8, MV_SB, MV_NS, true, false, 8>(p, st);
        case 8 * 16 + 16: return mvs<NT, 1, 8, MV_SB, MV_NS, true, false, 16>(p, st);
        default: return -1;
    }
}

template <int NT>
int mv_rtw(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st) {
    switch (rtw) {
        case 1: return mv_kw<NT, 1>(p, kw, rpt, st);
        case 2: return mv_kw<NT, 2>(p, kw, rpt, st);
        default: return -1;
    }
}

// The grouped mv kernel's 8-warp variants (legion_nvfp4_mvg_launch): the plain ones of
// legion_nvfp4_mv_launch, wider row tiles and longer stages (NT = 1, tuning), and the SwiGLU.
template <int NT>
int mvg(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft, cudaStream_t st) {
    if (pft != 0) {
        // Whole-tile prefetch: 16-block stages, 8 tiles per CTA, one tile ahead; more shapes at
        // NT = 1 (tuning).
        if (rtw != 1) {
            return -1;
        }
        // The defaults' tiles (GLU_MV_TILE: 4 tiles one ahead; DOWN_MV_TILE: 8 tiles two
        // ahead) at every token tile.
        switch (sb * 10000 + rpt * 100 + pft) {
            case 16 * 10000 + 8 * 100 + 1:
                return glu ? mvs<NT, 1, 8, 16, MV_NS, true, false, 8, true, 1>(p, st)
                           : mvs<NT, 1, 8, 16, MV_NS, true, false, 8, false, 1>(p, st);
            case 16 * 10000 + 4 * 100 + 1:
                if (glu) {
                    return mvs<NT, 1, 8, 16, MV_NS, true, false, 4, true, 1>(p, st);
                }
                break;
            case 16 * 10000 + 8 * 100 + 2:
                if (!glu) {
                    return mvs<NT, 1, 8, 16, MV_NS, true, false, 8, false, 2>(p, st);
                }
                break;
            default: break;
        }
        if constexpr (NT == 1) {
            switch (sb * 10000 + rpt * 100 + pft) {
                case 16 * 10000 + 8 * 100 + 2:
                    return mvs<NT, 1, 8, 16, MV_NS, true, false, 8, true, 2>(p, st);
                case 16 * 10000 + 16 * 100 + 1:
                    return glu ? mvs<NT, 1, 8, 16, MV_NS, true, false, 16, true, 1>(p, st)
                               : mvs<NT, 1, 8, 16, MV_NS, true, false, 16, false, 1>(p, st);
                case 16 * 10000 + 4 * 100 + 1:
                    return mvs<NT, 1, 8, 16, MV_NS, true, false, 4, false, 1>(p, st);
                case 40 * 10000 + 8 * 100 + 1:
                    return glu ? mvs<NT, 1, 8, 40, MV_NS, true, false, 8, true, 1>(p, st) : -1;
                default: return -1;
            }
        }
        return -1;
    }
    if (glu) {
        if (rtw != 1) {
            return -1;
        }
        switch (sb * 100 + rpt) {
            case 16 * 100 + 8: return mvs<NT, 1, 8, 16, MV_NS, true, false, 8, true>(p, st);
            default: break;
        }
        if constexpr (NT == 1) {
            switch (sb * 100 + rpt) {
                case 16 * 100 + 4: return mvs<NT, 1, 8, 16, MV_NS, true, false, 4, true>(p, st);
                case 16 * 100 + 16: return mvs<NT, 1, 8, 16, MV_NS, true, false, 16, true>(p, st);
                case 24 * 100 + 4: return mvs<NT, 1, 8, 24, MV_NS, true, false, 4, true>(p, st);
                case 24 * 100 + 8: return mvs<NT, 1, 8, 24, MV_NS, true, false, 8, true>(p, st);
                case 24 * 100 + 16: return mvs<NT, 1, 8, 24, MV_NS, true, false, 16, true>(p, st);
                case 40 * 100 + 4: return mvs<NT, 1, 8, 40, MV_NS, true, false, 4, true>(p, st);
                case 40 * 100 + 8: return mvs<NT, 1, 8, 40, MV_NS, true, false, 8, true>(p, st);
                case 40 * 100 + 16: return mvs<NT, 1, 8, 40, MV_NS, true, false, 16, true>(p, st);
                default: return -1;
            }
        }
        return -1;
    }
    if (sb == MV_SB) {
        if (rtw == 1) {
            // mv_kw<NT, 1>(p, 8, rpt, st), compiled in mv_nt<NT>.cu.
            return legion_nvfp4::launch_mv<NT>(p, 1, 8, rpt, st);
        }
        if constexpr (NT == 1) {
            switch (rtw * 100 + rpt) {
                case 2 * 100 + 2: return mvs<NT, 2, 8, MV_SB, MV_NS, true, false, 2>(p, st);
                case 2 * 100 + 4: return mvs<NT, 2, 8, MV_SB, MV_NS, true, false, 4>(p, st);
                case 4 * 100 + 1: return mvs<NT, 4, 8, MV_SB, MV_NS, true, false, 1>(p, st);
                case 4 * 100 + 2: return mvs<NT, 4, 8, MV_SB, MV_NS, true, false, 2>(p, st);
                default: return -1;
            }
        }
        return -1;
    }
    if constexpr (NT == 1) {
        switch (sb * 1000 + rtw * 100 + rpt) {
            case 24 * 1000 + 100 + 8: return mvs<NT, 1, 8, 24, MV_NS, true, false, 8>(p, st);
            case 40 * 1000 + 100 + 8: return mvs<NT, 1, 8, 40, MV_NS, true, false, 8>(p, st);
            default: return -1;
        }
    }
    return -1;
}
}  // namespace
