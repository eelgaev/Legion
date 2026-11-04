// Routed-expert FFN for a batch of tokens (prefill), on FP16 tensor cores with F32 sums.
//
//   out[:, t] = sum_j w[j, t] * down_e(silu(gate_e x_t) * (up_e x_t)),   e = ids[j, t]
//
// The graph side (op node, scratch layout, launch order, CPU reference) is
// src/model/moe_ffn.rs; this file holds the kernels and one thin launcher per kernel.
//
// Steps, all on the stream, no host sync:
// 1. `prep`: x (F32) to FP16 rows; per expert, its routed (token, slot) pairs counted.
// 2. `plan`: one block scans the counts into each expert's first pair and first tile.
// 3. `scatter`: the pairs listed expert by expert (order within an expert is free: a row's
//    products do not depend on which rows share its tile).
// 4. `gemm<GU>`: per (tile of up to TM pairs of one expert, BN weight rows), the gate and up
//    rows interleaved (even: gate, odd: up), dequantized from the GGUF blocks into shared
//    memory as FP16; the X rows gathered by token. SwiGLU in the epilogue -> H (FP16, pair
//    order of the list).
// 5. `gemm<DOWN>`: H times down -> D (F32), each row scaled by its router weight and stored at
//    its pair's place (token-major).
// 6. `reduce`: out[:, t] = D[t, 0] + D[t, 1] + ... in slot order (as ggml adds the views).
//
// Tensor cores: Volta's `mma.sync.m8n8k4` (also runs on later GPUs). Volta truncates when the
// tensor core adds into an accumulator, so each K step of BK products starts from fresh
// accumulators that are added into the F32 totals with ordinary adds (see the bring-up note in
// paged_attn.cu).
// The per-architecture pieces live in sm70/moe_gemm.cuh (the m8n8k4 step), sm80/moe_gemm.cuh
// (the m16n8k16 step), sm80/moe_gemm_raw.cuh (the cp.async raw-staged kernel) and
// sm80/moe_gemm_frag.cuh (Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down with register-dequantized B
// fragments, on the GB10), included below where they are used; use_mma16 and launch_gemm pick the
// path per device at run time.
//
// Expert weights are read through a device-visible address per expert: `w + table[e] * nb2`;
// table == null means slot e. Experts in host memory are staged into VRAM before the op runs
// (src/residency/staging.rs): read in place, the tiles would fetch each expert's weights
// several times over the link.
//
// Adapted from Strata-AC922's src/kernels/cuda/fused_expert.cu (MIT, (c) the Strata authors):
// the tile walk, the m8n8k4 fragment layout, the XOR swizzle and the per-step F32 sums.
// Its "fused experts v2" (K-quant scales from registers, Q5_K gate/up on the fused path) is
// here as well. A wider tile was measured on V100 for #262 and lost: 128 pairs x 128 rows, 16
// warps, one block per SM (half the dequantization per product, the same bits) made the op
// 4-38% slower on Qwen3.8-Flash-Next's Q4_K / Q5_1 layers at 1K-8K tokens.

#include "ggml-cuda/common.cuh"  // ggml-common.h's IQ grids and sign tables, as __device__ arrays
#include "device_cache.cuh"
#include "launch_pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

extern "C" {

// Everything the kernels read: filled by the Rust side (src/model/moe_ffn.rs), which also owns
// the scratch layout, the launch order and the grids.
struct legion_moe_args {
    // Expert weights: [k, m, n_slots] each, rows nb1 bytes apart, experts nb2 bytes apart.
    const uint8_t * up;
    const uint8_t * gate;
    const uint8_t * down;
    int64_t gu_nb1, gu_nb2;      // up and gate share type and strides
    int64_t down_nb1, down_nb2;
    const int32_t * table;       // [n_expert] slot of expert e, or null
    int n_expert, n_used;
    int64_t n_embd, n_ff, n_tok;
    const float * x;             // [n_embd, n_tok], columns x_ld floats apart
    int64_t x_ld;
    const int32_t * ids;         // [n_used, n_tok], columns ids_ld apart
    int64_t ids_ld;
    const float * w;             // [n_used, n_tok], columns w_ld apart
    int64_t w_ld;
    float * out;                 // [n_embd, n_tok], contiguous
    // scratch
    __half * xh;                 // [n_tok, n_embd] x in FP16
    int32_t * count;             // [n_expert] pairs per expert (zeroed before prep)
    int32_t * first;             // [n_expert + 1] first list position of each expert
    int32_t * cursor;            // [n_expert]
    int32_t * tile0;             // [n_expert + 1] first tile of each expert
    int32_t * list;              // [n_pairs] pair index (t * n_used + j), expert by expert
    __half * h;                  // [n_pairs, n_ff] SwiGLU rows in list order
    float * d;                   // [n_pairs, n_embd] weighted expert outputs in pair order
    int tm;                      // pairs per tile
    int n_slots;                 // experts the weight tensors hold (ne[2]): table entries index them
    int d_fmt;                   // D's element type: 0 F32, 1 F16, 2 BF16 (opt-in, see store_d)
    // The shared expert, added by reduce_kernel when sh is not null (#526): out[t] +=
    // sh[t] * sigmoid(sg[t]), as ggml's sigmoid, mul and add of a gated shared expert.
    const float * sh;            // [n_embd, n_tok], columns sh_ld floats apart
    int64_t sh_ld;
    const float * sg;            // the shared expert's gate logit of token t at sg[t * sg_ld]
    int64_t sg_ld;
    // The GEMMs' experts [e_lo, e_hi) (0, n_expert: all): block row y takes the range's tile y,
    // so the experts of weights still being staged run once their part is in.
    int e_lo, e_hi;
    // GLM's SwiGLU limit: silu(min(g, limit)) * clamp(u, -limit, limit); 0: plain SwiGLU.
    float limit;
    // GLM-5.3-Flash's int8 expert GEMMs (#603, opt-in; legion_moe_glm_q8's bits): bit 0, prep
    // writes x as Q8 rows into xh and gate/up runs on them; bit 1, gate/up writes H as Q8 rows
    // and down runs on them. 0: the F16 path.
    int q8;
    // 1: sm_80+ GEMMs keep gemm_kernel where gemm_raw_kernel and the fragment kernels would run
    // (--no-moe-raw-staging, for comparisons; the same bits). 0: the default.
    int raw_off;
    // Not null: the down GEMM makes the per-token sums itself and reduce_kernel does not run
    // (#826, gemm_frag_down_kernel's last-arriver sum, legion_moe_down_sum): per token and
    // 128-column block of out, the count of its pairs whose rows are still to come
    // ([n_tok, n_embd / 128], set by legion_moe_down_sum_init after the gate/up GEMM).
    int32_t * sum_cnt;
    // > 0: the sm_80+ fragment GEMMs launch at most this many block rows and each block walks
    // the plan tiles y, y + gridDim.y, ... below tile0[n_expert] (#858: a 144-row verify step's
    // grid holds ~534 block rows for ~230 tiles, the rest return at once). 0: one row per tile.
    int frag_ycap;
};

// Launchers, one per kernel; each returns cudaGetLastError(). Grids are the caller's.
int legion_moe_prep(const legion_moe_args * a, void * stream);
int legion_moe_plan(const legion_moe_args * a, void * stream);
int legion_moe_scatter(const legion_moe_args * a, void * stream);
// kind 0: gate/up + SwiGLU, 1: down. Tile tm x bn (the compiled shapes: 64 x 128); grid
// (m / bn, max_tiles). -1 when no kernel exists for the types or tile.
int legion_moe_gemm(const legion_moe_args * a, int kind, int wtype, int bn, unsigned grid_x, unsigned grid_y,
                    void * stream);
int legion_moe_reduce(const legion_moe_args * a, void * stream);
// #826: 1 when the down GEMM of weights `down_type` can make the per-token sums itself for these
// arguments (a->sum_cnt set, see there): the GB10's fragment down kernel takes the step, F32 pair
// rows, at most 16 pairs per token, aligned rows. 0: run legion_moe_reduce as usual.
int legion_moe_down_sum(const legion_moe_args * a, int down_type);
// Sets a->sum_cnt's counts (each token's valid pairs) and writes out for tokens without any, as
// reduce_kernel would. After the gate/up GEMM (the counts may live in x's FP16 rows), before
// the down GEMM.
int legion_moe_down_sum_init(const legion_moe_args * a, void * stream);
// legion_moe_gemm for GLM-5.3-Flash (#571): its own kernels where they take the weights
// (sm80/moe_gemm_glm.cuh: IQ2_XXS gate/up), legion_moe_gemm otherwise. The same bits.
int legion_moe_gemm_glm(const legion_moe_args * a, int kind, int wtype, int bn, unsigned grid_x, unsigned grid_y,
                        void * stream);
// 1 when the current device runs the GEMMs on mma.m16n8k16 (sm_80+ with code for it), else 0.
int legion_moe_mma16_supported(void);
// The `q8` bits legion_moe_gemm_glm can take for this step on the current device (#603,
// sm80/moe_gemm_glm_q8.cuh): bit 0 when gate/up is IQ2_XXS at Maya-S's layout and the step is a
// prompt step (more tokens than the decode kernels take), bit 1 when down is IQ2_S or IQ3_XXS as
// well.
int legion_moe_glm_q8(const legion_moe_args * a, int gu_type, int down_type);

}  // extern "C"

namespace {

// GGML_TYPE_*
constexpr int T_Q5_1 = 7, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14;
constexpr int T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ2_S = 22, T_IQ1_M = 29, T_Q2_0 = 42;
constexpr int T_Q2_K = 10, T_Q3_K = 11, T_IQ3_XXS = 18, T_IQ3_S = 21, T_IQ4_XS = 23;

constexpr int BK = 64;   // K step (halves): 8 16-byte chunks per smem row

// ---- steps 1-3: grouping ------------------------------------------------------------------

// Whether pair id `e` names an expert the weights hold. Others (a corrupt or padded routing
// entry) are left out of the grouping, the products and the sum, instead of indexing the
// counters and the weights out of bounds (an MMU fault, or a write into the scratch of another
// expert).
__device__ __forceinline__ bool expert_ok(const legion_moe_args & p, int e) {
    if (e < 0 || e >= p.n_expert) {
        return false;
    }
    const int64_t slot = p.table ? p.table[e] : e;
    return slot >= 0 && slot < p.n_slots;
}

// The shared expert's gate of token t, as reduce_kernel computes it (ggml's op_sigmoid, the
// same expression and flags); 0 without a shared expert. For the sums the down GEMM makes
// itself (#826).
__device__ __forceinline__ float shexp_gate(const legion_moe_args & p, int64_t t) {
    return p.sh ? 1.0f / (1.0f + expf(-p.sg[t * p.sg_ld])) : 0.0f;
}

// Columns of out per down-sum counter (#826): the fragment down kernel's row block.
constexpr int DSUM_BN = 128;
// Pairs per token the down GEMM's own sums take (a bit each, all loads issued before the sum).
constexpr int DSUM_MAX_USED = 16;

__global__ void prep_kernel(const legion_moe_args p) {
    const int64_t t = blockIdx.x;
    const float * xr = p.x + t * p.x_ld;
    __half2 * o = (__half2 *) (p.xh + t * p.n_embd);
    for (int64_t i = threadIdx.x; i < p.n_embd / 2; i += blockDim.x) {
        const float2 v = ((const float2 *) xr)[i];
        o[i] = __floats2half2_rn(v.x, v.y);
    }
    if (threadIdx.x < p.n_used) {
        const int e = p.ids[t * p.ids_ld + threadIdx.x];
        if (expert_ok(p, e)) {
            atomicAdd(&p.count[e], 1);
        }
    }
}

__global__ void plan_kernel(const legion_moe_args p) {
    const int n_expert = p.n_expert, TM = p.tm;
    // one block of 1024 threads; n_expert <= 1024 * 4
    __shared__ int32_t sp[1024], st[1024];
    constexpr int PER = 4;
    const int i0 = threadIdx.x * PER;
    int32_t c[PER], tl[PER], sum = 0, tsum = 0;
    for (int k = 0; k < PER; k++) {
        const int e = i0 + k;
        c[k] = e < n_expert ? p.count[e] : 0;
        tl[k] = (c[k] + TM - 1) / TM;
        sum += c[k];
        tsum += tl[k];
    }
    sp[threadIdx.x] = sum;
    st[threadIdx.x] = tsum;
    __syncthreads();
    for (int off = 1; off < 1024; off *= 2) {
        const int32_t a = threadIdx.x >= off ? sp[threadIdx.x - off] : 0;
        const int32_t b = threadIdx.x >= off ? st[threadIdx.x - off] : 0;
        __syncthreads();
        sp[threadIdx.x] += a;
        st[threadIdx.x] += b;
        __syncthreads();
    }
    int32_t run = sp[threadIdx.x] - sum, trun = st[threadIdx.x] - tsum;
    for (int k = 0; k < PER; k++) {
        const int e = i0 + k;
        if (e < n_expert) {
            p.first[e] = run;
            p.cursor[e] = run;
            p.tile0[e] = trun;
        }
        run += c[k];
        trun += tl[k];
    }
    if (threadIdx.x == 1023) {
        p.first[n_expert] = run;
        p.tile0[n_expert] = trun;
    }
}

__global__ void scatter_kernel(const legion_moe_args p) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n_tok * p.n_used) {
        return;
    }
    const int64_t t = i / p.n_used;
    const int j = (int) (i % p.n_used);
    const int e = p.ids[t * p.ids_ld + j];
    if (!expert_ok(p, e)) {
        return;
    }
    const int pos = atomicAdd(&p.cursor[e], 1);
    p.list[pos] = (int32_t) i;
}

// ---- dequantization of a weight row's 32 values into halves -------------------------------
//
// A thread converts `NV` 32-value pieces of one weight row per K step: piece `v` covers the
// K values [k0 + 32 v, k0 + 32 v + 32) (k0 a multiple of BK = 64). Raw bytes are fetched into
// registers one step ahead (`fetch`) and converted and stored after the step's products
// (`commit`), so the loads overlap the tensor-core work: `prepare` once per piece, then `part(i)`
// gives values 4 i .. 4 i + 3 as two half2 words.

// get_scale_min_k4 of ggml-quants.c on the 12 scale bytes held in registers: byte j & 3 of each
// word, then j < 4: the low 6 bits of s0's (scale) and s1's (min); j >= 4: s2's nibbles with the
// top 2 bits of s0's and s1's above them. Selects instead of a branch (#811: the same values).
__device__ __forceinline__ void scale_min_k4(int j, uint32_t s0, uint32_t s1, uint32_t s2, int & d, int & m) {
    const int b = 8 * (j & 3);
    const uint32_t x0 = s0 >> b, x1 = s1 >> b, x2 = s2 >> b;
    d = (int) (j < 4 ? x0 & 63 : (x2 & 0x0F) | ((x0 >> 2) & 0x30));
    m = (int) (j < 4 ? x1 & 63 : ((x2 >> 4) & 0x0F) | ((x1 >> 2) & 0x30));
}

template <int T> struct Piece;

// Q5_1 and Q8_0 convert in half2 arithmetic (two values per instruction; Volta converts between
// integers, F32 and F16 at a quarter of the F32 rate): a byte b (< 256) becomes the half
// 1024 + b by putting 0x64 above it (the "magic number" trick), and one fused multiply-add on
// exact operands gives each value with ggml's single rounding, since their block scales are
// halves already.
__device__ __forceinline__ __half2 u32_as_h2(uint32_t v) { return *(const __half2 *) &v; }
__device__ __forceinline__ uint32_t h2_as_u32(__half2 v) { return *(const uint32_t *) &v; }
// bytes 0, 1 (lo) or 2, 3 (hi) of x as the halves 1024 + byte
__device__ __forceinline__ __half2 magic_lo(uint32_t x) { return u32_as_h2(__byte_perm(x, 0x64646464u, 0x4140)); }
__device__ __forceinline__ __half2 magic_hi(uint32_t x) { return u32_as_h2(__byte_perm(x, 0x64646464u, 0x4342)); }
// bits 0..3 of b to bit 4 of bytes 0..3
__device__ __forceinline__ uint32_t spread4(uint32_t b) { return ((b & 0xF) * 0x00204081u & 0x01010101u) << 4; }

// K-quants: values d * sc * q - dmin * m for the 4 bytes x (q < 32 each), in F32 as ggml's
// dequantize_q4_K computes them, then rounded to F16 once. (Half2 arithmetic would round
// dmin * m first, and that error is shared by the 32 values of a sub-block: it shows in the
// products, ~30x the rounding noise.) A byte becomes an exact float by putting it in the
// mantissa of 2^23.
//
// sm_70 (#811): two bytes become the F16 subnormals q * 2^-24 with one byte permute, and the
// half-to-float conversion (HADD2.F32, the FMA pipe) gives them as exact floats, where the
// 2^23 trick takes a permute and an add per value. d1 is scaled by 2^24 to match. The product
// d1 * q is exact either way (d has 11 significant bits, sc < 64, q < 32: 22 bits), so the
// fused multiply-add rounds the same real number once: the same F32 values, the same halves.
// Newer GPUs keep the 2^23 form (the GB10's kernels were measured with it).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define LEGION_KSCALE_SUBNORMAL 1
#else
#define LEGION_KSCALE_SUBNORMAL 0
#endif
struct KScale {
    float d1, m1;
    __device__ __forceinline__ void set(uint4 hdr, int j) {
        int s, m;
        scale_min_k4(j, hdr.y, hdr.z, hdr.w, s, m);
        const float2 dm = __half22float2(u32_as_h2(hdr.x));
#if LEGION_KSCALE_SUBNORMAL
        // s and m as exact floats through the 2^23 trick (an add, not an XU conversion)
        d1 = dm.x * (__int_as_float(0x4B000000 | s) - 8388608.0f);
        m1 = dm.y * (__int_as_float(0x4B000000 | m) - 8388608.0f);
        d1 *= 16777216.0f;  // 2^24, exact
#else
        d1 = dm.x * s;
        m1 = dm.y * m;
#endif
    }
#if !LEGION_KSCALE_SUBNORMAL
    __device__ __forceinline__ float val(uint32_t x, uint32_t sel) const {
        const float q = __int_as_float(__byte_perm(x, 0x4B000000u, sel)) - 8388608.0f;
        return d1 * q - m1;
    }
#endif
    __device__ __forceinline__ void put(uint32_t x, uint32_t * o) const {
#if LEGION_KSCALE_SUBNORMAL
        const float2 a = __half22float2(u32_as_h2(__byte_perm(x, 0, 0x4140)));
        const float2 b = __half22float2(u32_as_h2(__byte_perm(x, 0, 0x4342)));
        o[0] = h2_as_u32(__floats2half2_rn(fmaf(d1, a.x, -m1), fmaf(d1, a.y, -m1)));
        o[1] = h2_as_u32(__floats2half2_rn(fmaf(d1, b.x, -m1), fmaf(d1, b.y, -m1)));
#else
        o[0] = h2_as_u32(__floats2half2_rn(val(x, 0x7440), val(x, 0x7441)));
        o[1] = h2_as_u32(__floats2half2_rn(val(x, 0x7442), val(x, 0x7443)));
#endif
    }
};

// Q4_K: 256-value blocks of 144 bytes: d, dmin (halves), scales[12], qs[128]. Values 64 c .. 64 c + 31
// are the low nibbles of qs[32 c .. 32 c + 31] (scale 2 c), the next 32 their high nibbles.
template <> struct Piece<T_Q4_K> {
    uint4 hdr, q0, q1;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k >> 8) * 144;  // k >= 0: shifts, not signed division
        const int c = ((int) k >> 6) & 3;
        hdr = *(const uint4 *) b;
        q0 = ((const uint4 *) (b + 16 + 32 * c))[0];
        q1 = ((const uint4 *) (b + 16 + 32 * c))[1];
    }
    KScale ks;
    int sh;
    __device__ __forceinline__ void prepare(int64_t k) {
        const int c = ((int) k >> 6) & 3, hi = ((int) k >> 5) & 1;
        ks.set(hdr, 2 * c + hi);
        sh = 4 * hi;
    }
    // values 4 i .. 4 i + 3 as 2 half2 words
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        ks.put((w[i] >> sh) & 0x0F0F0F0Fu, o);
    }
};

// Q5_K: 176 bytes: d, dmin, scales[12], qh[32], qs[128]; value 64 c + l (+ 32) adds 16 when bit
// 2 c (2 c + 1) of qh[l] is set.
template <> struct Piece<T_Q5_K> {
    uint4 hdr, q0, q1, h0, h1;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k >> 8) * 176;  // k >= 0: shifts, not signed division
        const int c = ((int) k >> 6) & 3;
        hdr = *(const uint4 *) b;
        h0 = ((const uint4 *) (b + 16))[0];
        h1 = ((const uint4 *) (b + 16))[1];
        q0 = ((const uint4 *) (b + 48 + 32 * c))[0];
        q1 = ((const uint4 *) (b + 48 + 32 * c))[1];
    }
    KScale ks;
    int sh, hb;
    __device__ __forceinline__ void prepare(int64_t k) {
        const int c = ((int) k >> 6) & 3, hi = ((int) k >> 5) & 1;
        ks.set(hdr, 2 * c + hi);
        sh = 4 * hi;
        hb = 2 * c + hi;
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t h[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        ks.put(((w[i] >> sh) & 0x0F0F0F0Fu) | (((h[i] >> hb) & 0x01010101u) << 4), o);
    }
};

// Q6_K: 210 bytes per 256 values: ql[128], qh[64], scales[16] (int8), d (half). In half n = v / 128
// of a block and quarter q = v % 128 / 32, value l < 32 is nibble q / 2 of ql[64 n + 32 (q % 2) + l]
// with bits 2 q .. 2 q + 1 of qh[32 n + l] above it, minus 32, times d * scales[8 n + 2 q + l / 16]
// (in F32 as ggml's dequantize_row_q6_K, then rounded to F16 once). Blocks are only 2-byte
// aligned: 16-bit loads.
template <> struct Piece<T_Q6_K> {
    uint32_t ql[8], qh[8];
    uint16_t sc, dh;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 210;
        const int n = (int) (k % 256) / 128, q = (int) (k % 128) / 32;
        const uint16_t * l16 = (const uint16_t *) (b + 64 * n + 32 * (q & 1));
        const uint16_t * h16 = (const uint16_t *) (b + 128 + 32 * n);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            ql[i] = l16[2 * i] | ((uint32_t) l16[2 * i + 1] << 16);
            qh[i] = h16[2 * i] | ((uint32_t) h16[2 * i + 1] << 16);
        }
        sc = *(const uint16_t *) (b + 192 + 8 * n + 2 * q);
        dh = *(const uint16_t *) (b + 208);
    }
    float d0, d1;
    int sl, sh;
    __device__ __forceinline__ void prepare(int64_t k) {
        const int q = (int) (k % 128) / 32;
        const float d = __half2float(__ushort_as_half(dh));
        d0 = d * (float) (int8_t) (sc & 0xFF);
        d1 = d * (float) (int8_t) (sc >> 8);
        sl = 4 * (q >> 1);
        sh = 2 * q;
    }
    __device__ __forceinline__ float val(uint32_t x, uint32_t sel, float d) const {
        return d * (__int_as_float(__byte_perm(x, 0x4B000000u, sel)) - 8388608.0f - 32.0f);
    }
    // values 4 i .. 4 i + 3 (scale l / 16: the first 16 values d0, the rest d1)
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t x = ((ql[i] >> sl) & 0x0F0F0F0Fu) | (((qh[i] >> sh) & 0x03030303u) << 4);
        const float d = i < 4 ? d0 : d1;
        o[0] = h2_as_u32(__floats2half2_rn(val(x, 0x7440, d), val(x, 0x7441, d)));
        o[1] = h2_as_u32(__floats2half2_rn(val(x, 0x7442, d), val(x, 0x7443, d)));
    }
};

// Q5_1: 32-value blocks of 24 bytes: d, m (halves), qh (u32), qs[16]; value i < 16 is the low
// nibble of qs[i] plus bit i of qh as bit 4, value i + 16 the high nibble plus bit i + 16.
template <> struct Piece<T_Q5_1> {
    uint2 v[3];
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint2 * b = (const uint2 *) (row + (k / 32) * 24);
        v[0] = b[0];
        v[1] = b[1];
        v[2] = b[2];
    }
    __device__ __forceinline__ void prepare(int64_t) {}
    // values 4 i .. 4 i + 3: low nibbles of qs word i (i < 4), or high nibbles of word i - 4
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const __half2 dm = u32_as_h2(v[0].x);
        const __half2 d = __half2half2(__low2half(dm)), m = __half2half2(__high2half(dm));
        const __half2 k1024 = __half2half2(__ushort_as_half(0x6400));
        const uint32_t qh = v[0].y;
        const uint32_t w[4] = {v[1].x, v[1].y, v[2].x, v[2].y};
        const uint32_t x = i < 4 ? (w[i & 3] & 0x0F0F0F0Fu) | spread4(qh >> (4 * i))
                                 : ((w[i & 3] >> 4) & 0x0F0F0F0Fu) | spread4(qh >> (16 + 4 * (i & 3)));
        o[0] = h2_as_u32(__hfma2(__hsub2(magic_lo(x), k1024), d, m));
        o[1] = h2_as_u32(__hfma2(__hsub2(magic_hi(x), k1024), d, m));
    }
};

// Q8_0: 34 bytes: d (half), qs[32] (2-byte aligned). A byte q + 128 (flipping the sign bit)
// becomes 1152 + q exactly.
//
// The two pieces of a K step (k0 + 0 and k0 + 32, k0 a multiple of 64) are two neighbouring
// blocks, 68 bytes from the 4-byte aligned row offset 68 k0 / 64. When the row is 4-byte aligned
// (the usual case: n_embd and n_ff multiples of 64), the piece takes 4-byte words 8 h .. 8 h + 8
// of those 68 bytes (h = 0, 1: which block), as 32- and 64-bit loads, instead of 17 16-bit loads
// (#808: the 16-bit loads made the Q8_0 down GEMM the slowest one). Block h's d is the low
// (h = 0) or high (h = 1) half of word 0, and its quant word i is bytes 2 + 4 i of the block:
// words i, i + 1 funnel-shifted by 16 bits (h = 0), or word i + 1 (h = 1). Same values.
template <> struct Piece<T_Q8_0> {
    uint32_t w[9];
    int h;  // which block of the pair; -1: the 16-bit path (w holds the block's 17 u16 pairs)
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        if (((uintptr_t) row & 3) == 0) {
            h = (int) (k / 32) & 1;
            const uint8_t * b = row + (k / 64) * 68 + 32 * h;
            if (((uintptr_t) b & 7) == 0) {
                const uint2 * b8 = (const uint2 *) b;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint2 v = b8[i];
                    w[2 * i] = v.x;
                    w[2 * i + 1] = v.y;
                }
                w[8] = ((const uint32_t *) b)[8];
            } else {
                w[0] = *(const uint32_t *) b;
                const uint2 * b8 = (const uint2 *) (b + 4);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint2 v = b8[i];
                    w[2 * i + 1] = v.x;
                    w[2 * i + 2] = v.y;
                }
            }
        } else {
            h = -1;
            const uint16_t * b = (const uint16_t *) (row + (k / 32) * 34);
            // as the aligned case with h = 1: d in the high half of word 0, quant word i in i + 1
            w[0] = (uint32_t) b[0] << 16;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                w[i + 1] = (uint32_t) b[1 + 2 * i] | ((uint32_t) b[2 + 2 * i] << 16);
            }
        }
    }
    __device__ __forceinline__ void prepare(int64_t) {}
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t d16 = h == 0 ? w[0] & 0xFFFFu : w[0] >> 16;
        const __half2 d = __half2half2(__ushort_as_half((uint16_t) d16));
        const __half2 k1152 = __half2half2(__ushort_as_half(0x6480));
        const uint32_t q = h == 0 ? __funnelshift_r(w[i], w[i + 1], 16) : w[i + 1];
        const uint32_t x = q ^ 0x80808080u;
        o[0] = h2_as_u32(__hmul2(__hsub2(magic_lo(x), k1152), d));
        o[1] = h2_as_u32(__hmul2(__hsub2(magic_hi(x), k1152), d));
    }
};

// ---- low-bit formats (an IQ2_XS-class Qwen3.8-Flash-Next GGUF mixes these) ------------------
//
// Values as ggml's CUDA dequantizers compute them (ggml-cuda/dequantize.cuh, MIT): F32, then
// rounded to F16 once. A piece is 32 values (one 32-value sub-block of a 256-value block, or half
// of a Q2_0 block); `prepare` decodes its four groups of 8, `part(i)` gives values 4 i .. 4 i + 3.
__device__ __forceinline__ void put4(const float * v, uint32_t * o) {
    o[0] = h2_as_u32(__floats2half2_rn(v[0], v[1]));
    o[1] = h2_as_u32(__floats2half2_rn(v[2], v[3]));
}

// IQ2_XXS: 66 bytes per 256: d (half), qs[32] u16. Sub-block ib: u16 4 ib .. 4 ib + 3: bytes 0..3
// the grid indices of its groups, the top u16 pair's bits 28.. the scale, 7-bit sign indices.
// The IQ2 codebooks (iq2xxs_grid, iq2s_grid) hold only the bytes 8, 25 and 43: an entry's 8
// bytes fit in 16 bits as codes 0, 1, 2 (2 bits each), which a block keeps in shared memory
// (2 KB for IQ2_S) instead of looking up the 8 KB table at random in global memory.
__device__ __forceinline__ uint32_t iq2_value(uint32_t c) { return 8 + 17 * c + (c >> 1); }
__device__ __forceinline__ uint16_t iq2_code(uint64_t e) {
    uint32_t code = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t b = (uint32_t) (e >> (8 * j)) & 0xFF;
        code |= (b == 8 ? 0u : (b == 25 ? 1u : 2u)) << (2 * j);
    }
    return (uint16_t) code;
}

template <> struct Piece<T_IQ2_XXS> {
    uint16_t d;
    uint2 q;
    float val[32];
    const uint16_t * tab = nullptr;  // compact codebook in shared memory, or null
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 66;
        const int ib = (int) (k % 256) / 32;
        d = *(const uint16_t *) b;
        const uint16_t * q2 = (const uint16_t *) (b + 2) + 4 * ib;
        q = make_uint2((uint32_t) q2[0] | ((uint32_t) q2[1] << 16), (uint32_t) q2[2] | ((uint32_t) q2[3] << 16));
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const float dd = __half2float(__ushort_as_half(d)) * (0.5f + (q.y >> 28)) * 0.25f;
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const int idx = (int) ((q.x >> (8 * il)) & 0xFF);
            const uint8_t signs = ksigns_iq2xs[(q.y >> (7 * il)) & 127];
            if (tab) {
                const uint32_t code = tab[idx];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    val[8 * il + j] = dd * (float) iq2_value((code >> (2 * j)) & 3) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            } else {
                const uint8_t * grid = (const uint8_t *) (iq2xxs_grid + idx);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    val[8 * il + j] = dd * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// IQ2_S: 82 bytes per 256: d, qs[64] (grid low bytes, then signs), qh[8], scales[8].
template <> struct Piece<T_IQ2_S> {
    uint16_t d;
    uint32_t ql, sg;
    uint8_t qh, sc;
    float val[32];
    // the codebook in shared memory, 2 bits per value (see iq2_code), or null for iq2s_grid
    const uint16_t * tab = nullptr;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 82;
        const int ib = (int) (k % 256) / 32;
        d = *(const uint16_t *) b;
        const uint8_t * qs = b + 2;
        ql = (uint32_t) qs[4 * ib] | ((uint32_t) qs[4 * ib + 1] << 8) | ((uint32_t) qs[4 * ib + 2] << 16) |
             ((uint32_t) qs[4 * ib + 3] << 24);
        sg = (uint32_t) qs[32 + 4 * ib] | ((uint32_t) qs[32 + 4 * ib + 1] << 8) |
             ((uint32_t) qs[32 + 4 * ib + 2] << 16) | ((uint32_t) qs[32 + 4 * ib + 3] << 24);
        qh = b[66 + ib];
        sc = b[74 + ib];
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const float d0 = __half2float(__ushort_as_half(d));
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const int idx = (int) ((ql >> (8 * il)) & 0xFF) | (((int) qh << (8 - 2 * il)) & 0x300);
            const float dd = d0 * (0.5f + ((sc >> (4 * (il / 2))) & 0xf)) * 0.25f;
            const uint8_t signs = (uint8_t) (sg >> (8 * il));
            if (tab) {
                const uint32_t code = tab[idx];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    val[8 * il + j] = dd * (float) iq2_value((code >> (2 * j)) & 3) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            } else {
                const uint8_t * grid = (const uint8_t *) (iq2s_grid + idx);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    val[8 * il + j] = dd * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// IQ2_XS: 74 bytes per 256: d (half), qs[32] u16, scales[8]. Sub-block ib: u16 4 ib .. 4 ib + 3,
// one per group of 8: bits 0..8 the grid index (iq2xs_grid), bits 9.. the 7-bit sign index;
// scales[ib] has the first two groups' scale in its low nibble, the last two's in its high one
// (ggml's dequantize_iq2_xs). DeepSeek-V4.1-Flash GSQ-RCO GGUFs mix it into their experts (#584).
template <> struct Piece<T_IQ2_XS> {
    uint16_t d;
    uint2 q;
    uint8_t sc;
    float val[32];
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 74;
        const int ib = (int) (k % 256) / 32;
        d = *(const uint16_t *) b;
        const uint16_t * q2 = (const uint16_t *) (b + 2) + 4 * ib;
        q = make_uint2((uint32_t) q2[0] | ((uint32_t) q2[1] << 16), (uint32_t) q2[2] | ((uint32_t) q2[3] << 16));
        sc = b[66 + ib];
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const float d0 = __half2float(__ushort_as_half(d));
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const uint32_t q2 = ((il < 2 ? q.x : q.y) >> (16 * (il & 1))) & 0xFFFF;
            const float dd = d0 * (0.5f + ((sc >> (4 * (il / 2))) & 0xf)) * 0.25f;
            const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (q2 & 511));
            const uint8_t signs = ksigns_iq2xs[q2 >> 9];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                val[8 * il + j] = dd * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// IQ1_M: 56 bytes per 256: qs[32], qh[16], scales[8] (the block's F16 scale in their top nibbles).
template <> struct Piece<T_IQ1_M> {
    uint32_t qs;
    uint16_t qh;
    uint16_t sc[4];
    int ib;
    float val[32];
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 56;
        ib = (int) (k % 256) / 32;
        qs = (uint32_t) b[4 * ib] | ((uint32_t) b[4 * ib + 1] << 8) | ((uint32_t) b[4 * ib + 2] << 16) |
             ((uint32_t) b[4 * ib + 3] << 24);
        qh = (uint16_t) b[32 + 2 * ib] | ((uint16_t) b[32 + 2 * ib + 1] << 8);
        const uint16_t * s16 = (const uint16_t *) (b + 48);
#pragma unroll
        for (int i = 0; i < 4; i++) sc[i] = s16[i];
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const uint16_t u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float scale = __half2float(__ushort_as_half(u));
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const int ib16 = 2 * ib + il / 2;
            const float d = scale * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
            const uint8_t h = (uint8_t) (qh >> (8 * (il / 2)));
            const float delta = h & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
            uint32_t g[2];
            g[0] = iq1s_grid_gpu[((qs >> (8 * il)) & 0xFF) | (((h >> 4 * (il % 2)) & 7) << 8)];
            g[1] = (g[0] >> 4) & 0x0f0f0f0f;
            g[0] &= 0x0f0f0f0f;
            const int8_t * q = (const int8_t *) g;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                val[8 * il + j] = d * (q[j] + delta);
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// Q2_0: 18 bytes per 64: d (half), qs[16]: value j = (code j - 1) d, code j in bits 2 (j % 4) of
// byte j / 4. A piece is half a block.
template <> struct Piece<T_Q2_0> {
    uint16_t d;
    uint32_t c0, c1;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 64) * 18;
        const int h = (int) (k % 64) / 32;
        d = *(const uint16_t *) b;
        const uint16_t * q = (const uint16_t *) (b + 2 + 8 * h);
        c0 = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
        c1 = (uint32_t) q[2] | ((uint32_t) q[3] << 16);
    }
    __device__ __forceinline__ void prepare(int64_t) {}
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const float dd = __half2float(__ushort_as_half(d));
        const uint32_t w = i < 4 ? c0 : c1;
        const uint32_t byte = (w >> (8 * (i & 3))) & 0xFF;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = (float) ((int) ((byte >> (2 * j)) & 3) - 1) * dd;
        }
        put4(v, o);
    }
};

// ---- Project Maya's GLM-5.3-Flash quants (#559) ----------------------------------------------
//
// As above: ggml's CUDA dequantizers' arithmetic (dequantize.cuh, MIT) in F32, rounded to F16
// once. Blocks of 110 and 98 bytes are only 2-byte aligned (16-bit loads), Q2_K's 84 4-byte.

// Q2_K: 84 bytes per 256: scales[16] (low nibble scale, high nibble min), qs[64], d, dmin. In half
// n = v / 128 and quarter j = v % 128 / 32, value l < 32 is bits 2 j of qs[32 n + l], times
// d * (scales[8 n + 2 j + l / 16] & 15), minus dmin * (that >> 4).
template <> struct Piece<T_Q2_K> {
    uint32_t q[8];
    uint16_t sc;
    uint32_t dm;
    int sh;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 84;
        const int n = (int) (k % 256) / 128, j = (int) (k % 128) / 32;
        const uint32_t * q32 = (const uint32_t *) (b + 16 + 32 * n);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            q[i] = q32[i];
        }
        sc = (uint16_t) b[8 * n + 2 * j] | ((uint16_t) b[8 * n + 2 * j + 1] << 8);
        dm = *(const uint32_t *) (b + 80);
    }
    float d0, m0, d1, m1;
    __device__ __forceinline__ void prepare(int64_t k) {
        const float2 f = __half22float2(u32_as_h2(dm));
        const float dall = f.x, dmin = f.y;
        d0 = dall * (sc & 0xF);
        m0 = dmin * ((sc >> 4) & 0xF);
        d1 = dall * ((sc >> 8) & 0xF);
        m1 = dmin * (sc >> 12);
        sh = 2 * ((int) (k % 128) / 32);
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t x = (q[i] >> sh) & 0x03030303u;
        const float d = i < 4 ? d0 : d1, m = i < 4 ? m0 : m1;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = d * (float) ((x >> (8 * j)) & 3) - m;
        }
        put4(v, o);
    }
};

// Q3_K: 110 bytes per 256: hmask[32], qs[64], scales[12] (6-bit, packed), d. In half n and quarter
// j as Q2_K, value l is bits 2 j of qs[32 n + l], less 4 unless bit 4 n + j of hmask[l] is set,
// times d * (scale 8 n + 2 j + l / 16 - 32).
template <> struct Piece<T_Q3_K> {
    uint32_t q[8], h[8];
    uint16_t s[6];
    uint16_t dh;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 110;
        const int n = (int) (k % 256) / 128;
        const uint16_t * h16 = (const uint16_t *) b;
        const uint16_t * q16 = (const uint16_t *) (b + 32 + 32 * n);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            h[i] = h16[2 * i] | ((uint32_t) h16[2 * i + 1] << 16);
            q[i] = q16[2 * i] | ((uint32_t) q16[2 * i + 1] << 16);
        }
        const uint16_t * s16 = (const uint16_t *) (b + 96);
#pragma unroll
        for (int i = 0; i < 6; i++) {
            s[i] = s16[i];
        }
        dh = *(const uint16_t *) (b + 108);
    }
    float d0, d1;
    int sh, hb;
    __device__ __forceinline__ int scale_byte(int i) const { return (s[i / 2] >> (8 * (i % 2))) & 0xFF; }
    // ggml's dequantize_q3_K unpacking of scale `is`
    __device__ __forceinline__ int scale(int is) const {
        return is < 4  ? (scale_byte(is) & 0xF) | (((scale_byte(is + 8) >> 0) & 3) << 4) :
               is < 8  ? (scale_byte(is) & 0xF) | (((scale_byte(is + 4) >> 2) & 3) << 4) :
               is < 12 ? (scale_byte(is - 8) >> 4) | (((scale_byte(is) >> 4) & 3) << 4) :
                         (scale_byte(is - 8) >> 4) | (((scale_byte(is - 4) >> 6) & 3) << 4);
    }
    __device__ __forceinline__ void prepare(int64_t k) {
        const int n = (int) (k % 256) / 128, j = (int) (k % 128) / 32;
        const float d = __half2float(__ushort_as_half(dh));
        d0 = d * (scale(8 * n + 2 * j) - 32);
        d1 = d * (scale(8 * n + 2 * j + 1) - 32);
        sh = 2 * j;
        hb = 4 * n + j;
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t x = (q[i] >> sh) & 0x03030303u, m = (h[i] >> hb) & 0x01010101u;
        const float d = i < 4 ? d0 : d1;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = d * (float) ((int) ((x >> (8 * j)) & 3) - (((m >> (8 * j)) & 1) ? 0 : 4));
        }
        put4(v, o);
    }
};

// IQ3_XXS: 98 bytes per 256: d, qs[64] (grid indices, 8 per sub-block), then a u32 per sub-block
// (7-bit sign indices, the scale in bits 28..). Group il of sub-block ib: entries qs[8 ib + 2 il]
// and qs[8 ib + 2 il + 1] of iq3xxs_grid (4 values each).
template <> struct Piece<T_IQ3_XXS> {
    uint16_t d;
    uint16_t g[4];
    uint32_t aux;
    float val[32];
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 98;
        const int ib = (int) (k % 256) / 32;
        d = *(const uint16_t *) b;
        const uint16_t * q16 = (const uint16_t *) (b + 2 + 8 * ib);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            g[i] = q16[i];
        }
        const uint16_t * a16 = (const uint16_t *) (b + 2 + 64 + 4 * ib);
        aux = (uint32_t) a16[0] | ((uint32_t) a16[1] << 16);
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const float dd = __half2float(__ushort_as_half(d)) * (0.5f + (aux >> 28)) * 0.5f;
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + (g[il] & 0xFF));
            const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + (g[il] >> 8));
            const uint8_t signs = ksigns_iq2xs[(aux >> (7 * il)) & 127];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                val[8 * il + j] = dd * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                val[8 * il + j + 4] = dd * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// IQ3_S: 110 bytes per 256: d, qs[64] (grid low bytes), qh[8] (their 9th bits), signs[32],
// scales[4] (4 bits per sub-block).
template <> struct Piece<T_IQ3_S> {
    uint16_t d;
    uint16_t g[4], sg[2];
    uint8_t qh, sc;
    float val[32];
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint8_t * b = row + (k / 256) * 110;
        const int ib = (int) (k % 256) / 32;
        d = *(const uint16_t *) b;
        const uint16_t * q16 = (const uint16_t *) (b + 2 + 8 * ib);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            g[i] = q16[i];
        }
        const uint16_t * s16 = (const uint16_t *) (b + 74 + 4 * ib);
        sg[0] = s16[0];
        sg[1] = s16[1];
        qh = b[66 + ib];
        sc = (uint8_t) ((b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF);
    }
    __device__ __forceinline__ void prepare(int64_t) {
        const float dd = __half2float(__ushort_as_half(d)) * (1 + 2 * sc);
#pragma unroll
        for (int il = 0; il < 4; il++) {
            const uint8_t * grid1 = (const uint8_t *) (iq3s_grid + ((g[il] & 0xFF) | ((qh << (8 - 2 * il)) & 256)));
            const uint8_t * grid2 = (const uint8_t *) (iq3s_grid + ((g[il] >> 8) | ((qh << (7 - 2 * il)) & 256)));
            const uint8_t signs = (uint8_t) (sg[il / 2] >> (8 * (il % 2)));
#pragma unroll
            for (int j = 0; j < 4; j++) {
                val[8 * il + j] = dd * grid1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                val[8 * il + j + 4] = dd * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
        }
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const { put4(val + 4 * i, o); }
};

// IQ4_XS: 136 bytes per 256 (8-byte aligned): d, scales_h (u16), scales_l[4], qs[128]. Sub-block
// ib: value l < 16 is kvalues_iq4nl[low nibble of qs[16 ib + l]], l + 16 the high nibble's, times
// d * (6-bit scale - 32).
template <> struct Piece<T_IQ4_XS> {
    uint2 h, q0, q1;
    int ib;
    __device__ __forceinline__ void fetch(const uint8_t * row, int64_t k) {
        const uint2 * b = (const uint2 *) (row + (k / 256) * 136);
        ib = (int) (k % 256) / 32;
        h = b[0];
        q0 = b[1 + 2 * ib];
        q1 = b[2 + 2 * ib];
    }
    float dd;
    __device__ __forceinline__ void prepare(int64_t) {
        const float d = __half2float(__ushort_as_half((uint16_t) (h.x & 0xFFFF)));
        const int sh = (int) (h.x >> 16), sl = (int) ((h.y >> (8 * (ib / 2))) & 0xFF);
        dd = d * ((((sl >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4)) - 32);
    }
    __device__ __forceinline__ void part(int i, uint32_t * o) const {
        const uint32_t w[4] = {q0.x, q0.y, q1.x, q1.y};
        const uint32_t x = i < 4 ? w[i] & 0x0F0F0F0Fu : (w[i - 4] >> 4) & 0x0F0F0F0Fu;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = dd * kvalues_iq4nl[(x >> (8 * j)) & 0xF];
        }
        put4(v, o);
    }
};

// ---- the tile GEMM --------------------------------------------------------------------------

// The X tile is [rows][BK] halves, its 16-byte chunks XOR-swizzled by row so that a quarter
// warp's fragment loads (rows r..r+3 and r+8..r+11 of one chunk) hit 8 distinct chunks, and the
// loader's stores (8 chunks of one row) too.
__device__ __forceinline__ int swz(int r) { return (r & 3) | (((r >> 3) & 1) << 2); }
// byte offset of chunk 0 of X row r: chunk c is at row_base(r) ^ (c << 4)
__device__ __forceinline__ int row_base(int r) { return r * (BK * 2) | (swz(r) << 4); }
// The W tile is k-major: 16-byte chunk c (values 8c..8c+7 of the K step) of all rows, then chunk
// c + 1, so a lane's fragment loads are its base address plus constants. Within a chunk plane,
// row r sits at slot `pos(r)`: r with bits 2 and 3 swapped, so the rows a quarter warp loads
// together (r..r+3 and r+8..r+11) fill 8 consecutive slots. In planes 4..7 the slot's bit 2 is
// flipped: the two loader threads of a row store planes q and 4 + q side by side, and so hit
// different banks.
__device__ __forceinline__ int pos(int r) { return (r & ~12) | ((r & 4) << 1) | ((r & 8) >> 1); }
// byte offset of chunk c of row r in a k-major tile of `rows` rows
__device__ __forceinline__ int koff(int r, int c, int rows) {
    return c * rows * 16 + ((pos(r) ^ (c & 4)) * 16);
}

enum Kind { GU = 0, DOWN = 1 };

// sm_70 (Volta) and newer: the mma.sync.m8n8k4 K step (step_mma).
#include "sm70/moe_gemm.cuh"

// sm_80 and newer: the mma.sync.m16n8k16 K step (step_mma16).
#include "sm80/moe_gemm.cuh"

// Tile `tile` of the expert-sorted pair list (tiles of up to `tm` pairs, an expert's tiles
// consecutive): its expert's weight slot, its first list position and its pairs (0: past the
// last tile).
struct Tile {
    int r0, ne;
    int64_t slot;
};
__device__ __forceinline__ Tile tile_of(const legion_moe_args & p, int tile, int tm) {
    tile += p.tile0[p.e_lo];
    if (tile >= p.tile0[p.e_hi]) {
        return {0, 0, 0};
    }
    int lo = p.e_lo, hi = p.e_hi;  // last e with tile0[e] <= tile
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (p.tile0[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int e = lo;
    const int r0 = p.first[e] + (tile - p.tile0[e]) * tm;
    return {r0, min(tm, p.first[e + 1] - r0), p.table ? p.table[e] : e};
}

// D (the weighted per-pair outputs, written by the down GEMM and summed by the reduce) is F32
// by default. --moe-pair-out f16 / bf16 (opt-in) stores it in 16 bits: half the bytes of
// the op's largest stream (each pair's output rounded once; the sums stay F32, in slot order).
__device__ __forceinline__ void store_d(const legion_moe_args & p, int64_t i, float v) {
    if (p.d_fmt == 1) {
        ((__half *) p.d)[i] = __float2half_rn(v);
    } else if (p.d_fmt == 2) {
        ((__nv_bfloat16 *) p.d)[i] = __float2bfloat16_rn(v);
    } else {
        p.d[i] = v;
    }
}
__device__ __forceinline__ float load_d(const legion_moe_args & p, int64_t i) {
    if (p.d_fmt == 1) {
        return __half2float(((const __half *) p.d)[i]);
    }
    if (p.d_fmt == 2) {
        return __bfloat162float(((const __nv_bfloat16 *) p.d)[i]);
    }
    return p.d[i];
}

// A tile's products (E: TM rows of LDE floats in smem, BN columns) to the op's buffers: gate/up
// interleaved -> SwiGLU -> H (FP16, list order), or down -> D scaled by the router weight (pair
// order).
// ROWS (all the GEMM kernels; TM * 8 bytes of smem past E): down looks each row's pair and router
// weight up once into smem, not once per element (an integer division by n_used and two global
// loads each). The same products and stores, so the same bits. Down's K is only n_ff long, so
// those lookups were a large share of its time: without them down is 6-20% faster on V100 (#262)
// and gemm_raw_kernel's 12-20% faster on the GB10 at 4K-8K-token steps (#363).
template <int TM, int BN, int LDE, int NT, Kind K, bool ROWS = false>
__device__ __forceinline__ void store_tile(const legion_moe_args & p, const float * E, int r0, int ne, int rt) {
    if constexpr (K == DOWN && ROWS) {
        int * pr = (int *) (E + TM * LDE);
        float * wr = (float *) (pr + TM);
        for (int r = threadIdx.x; r < ne; r += NT) {
            const int pair = p.list[r0 + r];
            const int64_t t = pair / p.n_used;
            const int j = pair % p.n_used;
            pr[r] = pair;
            wr[r] = p.w[t * p.w_ld + j];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < TM * BN; i += NT) {
            const int r = i / BN, c = i % BN;
            if (r < ne) {
                store_d(p, (int64_t) pr[r] * p.n_embd + (int64_t) rt * BN + c, E[r * LDE + c] * wr[r]);
            }
        }
        return;
    }
    if (K == GU) {
        constexpr int HC = BN / 2;
        for (int i = threadIdx.x; i < TM * HC; i += NT) {
            const int r = i / HC, c = i % HC;
            if (r < ne) {
                float g = E[r * LDE + 2 * c], u = E[r * LDE + 2 * c + 1];
                if (p.limit > 0.0f) {
                    g = fminf(g, p.limit);
                    u = fminf(fmaxf(u, -p.limit), p.limit);
                }
                const float h = g / (1.0f + __expf(-g)) * u;
                p.h[(int64_t) (r0 + r) * p.n_ff + (int64_t) rt * HC + c] = __float2half(h);
            }
        }
    } else {
        for (int i = threadIdx.x; i < TM * BN; i += NT) {
            const int r = i / BN, c = i % BN;
            if (r < ne) {
                const int pair = p.list[r0 + r];
                const int64_t t = pair / p.n_used;
                const int j = pair % p.n_used;
                const float wgt = p.w[t * p.w_ld + j];
                store_d(p, (int64_t) pair * p.n_embd + (int64_t) rt * BN + c, E[r * LDE + c] * wgt);
            }
        }
    }
}

// Block tile: TM pairs x BN weight rows; warps WM x WN each. A warp's 4 quad pairs (QP q, lanes
// 4q..4q+3 and 4q+16..4q+19) each compute MI x NJ blocks of 8 x 8: M blocks (q & 1) + 2 i, N
// blocks (q >> 1) * NJ + j; lane t (0..7) of a QP holds row t of A and column t of B.
template <int TM, int BN, int WM, int WN>
struct Shape {
    static constexpr int WARPS_M = TM / WM, WARPS_N = BN / WN, NT = 32 * WARPS_M * WARPS_N;
    static constexpr int MI = WM / 16, NJ = WN / 16;
    static constexpr int PIECES = BN * BK / 32 / NT;   // 32-value weight pieces per thread and step
    static constexpr int XV = TM * BK / 8 / NT;        // 16-byte X chunks per thread and step
    static constexpr int LDE = BN + 4;
    static constexpr int STAGE = (TM + BN) * BK * 2;   // bytes of one X + W stage
    static constexpr int SMEM_AB = 2 * STAGE;           // double-buffered
    static constexpr int SMEM_E = TM * LDE * 4 + TM * 8;  // E, then store_tile's rows
    static constexpr int SMEM = SMEM_AB > SMEM_E ? SMEM_AB : SMEM_E;
    static_assert(PIECES >= 1 && (BN * BK / 32) % NT == 0, "pieces per thread");
    static_assert(XV >= 1 && (TM * BK / 8) % NT == 0, "x chunks per thread");
};

template <int TM, int BN, int WM, int WN, int WT, Kind K, bool M16 = false>
__global__ void __launch_bounds__(Shape<TM, BN, WM, WN>::NT, 2)
gemm_kernel(const legion_moe_args p) {
    using S = Shape<TM, BN, WM, WN>;
    constexpr int NT = S::NT, MI = S::MI, NJ = S::NJ;
    extern __shared__ __align__(16) unsigned char sm[];

    // which expert and tile
    // blockIdx.x walks the weight rows of one tile, so the blocks running together share the
    // tile's X rows and its expert's weights in L2
    const Tile tl = tile_of(p, blockIdx.y, TM);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int64_t slot = tl.slot;
    const uint8_t * up = p.up;
    const uint8_t * gate = p.gate;
    const uint8_t * down = p.down;
    const int rt = blockIdx.x;

    const int64_t KD = K == GU ? p.n_embd : p.n_ff;

    // this thread's weight rows: piece i covers smem row (tid + NT i) / (BK / 32), half (.. % 2)
    const uint8_t * wrow[S::PIECES];
#pragma unroll
    for (int i = 0; i < S::PIECES; i++) {
        const int pc = threadIdx.x + NT * i, r = pc / 2;
        if (K == GU) {
            const int64_t row = (int64_t) rt * (BN / 2) + r / 2;
            wrow[i] = ((r & 1) ? up : gate) + slot * p.gu_nb2 + row * p.gu_nb1;
        } else {
            const int64_t row = (int64_t) rt * BN + r;
            wrow[i] = down + slot * p.down_nb2 + row * p.down_nb1;
        }
    }
    // this thread's X rows (gathered by token for GU, list order for DOWN)
    const __half * xrow[S::XV];
#pragma unroll
    for (int i = 0; i < S::XV; i++) {
        const int c = threadIdx.x + NT * i, r = c / (BK / 8);
        if (r < ne) {
            if (K == GU) {
                const int pair = p.list[r0 + r];
                xrow[i] = p.xh + (int64_t) (pair / p.n_used) * p.n_embd;
            } else {
                xrow[i] = p.h + (int64_t) (r0 + r) * p.n_ff;
            }
        } else {
            xrow[i] = nullptr;
        }
    }

    Piece<WT> pw[S::PIECES];
    uint4 xr[S::XV];
    auto fetch = [&](int64_t k0) {
#pragma unroll
        for (int i = 0; i < S::XV; i++) {
            const int c = (threadIdx.x + NT * i) % (BK / 8);
            xr[i] = xrow[i] ? *(const uint4 *) (xrow[i] + k0 + 8 * c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < S::PIECES; i++) {
            const int half = (threadIdx.x + NT * i) % 2;
            pw[i].fetch(wrow[i], k0 + 32 * half);
        }
    };
    auto commit = [&](int64_t k0, int stage) {
        char * Xs = (char *) sm + stage * S::STAGE;
        char * Ws = Xs + TM * BK * 2;
#pragma unroll
        for (int i = 0; i < S::XV; i++) {
            const int c = threadIdx.x + NT * i;
            *(uint4 *) (Xs + (row_base(c / (BK / 8)) ^ ((c % (BK / 8)) << 4))) = xr[i];
        }
#pragma unroll
        for (int i = 0; i < S::PIECES; i++) {
            const int pc = threadIdx.x + NT * i, r = pc / 2, half = pc % 2;
            pw[i].prepare(k0 + 32 * half);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t o[4];
                pw[i].part(2 * q, o);
                pw[i].part(2 * q + 1, o + 2);
                *(uint4 *) (Ws + koff(r, 4 * half + q, BN)) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
    };

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / S::WARPS_N, wn = warp % S::WARPS_N;
    const int q = (lane >> 2) & 3, t = (lane & 3) + ((lane >> 4) << 2);
    // M blocks of 8 this lane reads: row 8 ((q & 1) + 2 i) + t of the warp's WM; skip 16-row
    // units past the tile's last pair
    const int mbase = wm * WM;
    int mvalid = (ne - mbase + 15) / 16;
    mvalid = mvalid < 0 ? 0 : (mvalid > MI ? MI : mvalid);
    // Written as one base plus constants (#811), so the compiler addresses the other blocks with
    // immediate offsets instead of rematerializing each address from the thread index every K
    // step: rows 16 apart share the swizzle (row_base(r + 16 i) = row_base(r) + 16 i BK 2), and
    // the 8 j rows added to the B base touch only bits the base leaves clear (NJ a power of two),
    // so pos() maps them on their own.
    static_assert((NJ & (NJ - 1)) == 0, "NJ a power of two");
    int pa[MI], pb[NJ];
    const int pa0 = row_base(mbase + 8 * (q & 1) + t);
    const int pb0 = TM * BK * 2 + koff(wn * WN + 8 * (q >> 1) * NJ + t, 0, BN);
#pragma unroll
    for (int i = 0; i < MI; i++) pa[i] = pa0 + i * 16 * BK * 2;
#pragma unroll
    for (int j = 0; j < NJ; j++) pb[j] = pb0 + koff(8 * j, 0, BN);

    Acc tot[MI][NJ];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int v = 0; v < 8; v++) tot[i][j][v] = 0.0f;
    // sm_80+: 16-row m-tiles and 8-column n-tiles of the warp's WM x WN
    constexpr int MT = WM / 16, NT8 = WN / 8;
    Acc4 tot16[M16 ? MT : 1][M16 ? NT8 : 2];
#pragma unroll
    for (int i = 0; i < (M16 ? MT : 1); i++)
#pragma unroll
        for (int j = 0; j < (M16 ? NT8 : 2); j++) tot16[i][j][0] = tot16[i][j][1] = tot16[i][j][2] = tot16[i][j][3] = 0.0f;

    // two smem stages: the step's products read one while the next step's tiles, fetched into
    // registers before the products, are stored into the other; one barrier per step
    fetch(0);
    commit(0, 0);
    __syncthreads();
    int stage = 0;
    for (int64_t k0 = 0; k0 < KD; k0 += BK, stage ^= 1) {
        const bool more = k0 + BK < KD;
        if (more) {
            fetch(k0 + BK);
        }
        const __half * Xs = (const __half *) (sm + stage * S::STAGE);
        const char * st = (const char *) Xs;
        // 16-row units with pairs; a warp past the tile's last pair has nothing to compute
        if constexpr (M16) {
            if (mvalid > 0) {
                step_mma16<MT, NT8, TM, BN>(st, mbase, wn * WN, mvalid, lane, tot16);
            }
        } else if (mvalid == MI) {
            step_mma<MI, MI, NJ, BN * 16>(st, pa, pb, tot);
        } else if (mvalid > 0) {
            if constexpr (MI >= 4) {
                if (mvalid == 3) step_mma<(MI >= 4 ? 3 : 1), MI, NJ, BN * 16>(st, pa, pb, tot);
            }
            if constexpr (MI >= 3) {
                if (mvalid == 2) step_mma<(MI >= 3 ? 2 : 1), MI, NJ, BN * 16>(st, pa, pb, tot);
            }
            if (mvalid == 1) step_mma<1, MI, NJ, BN * 16>(st, pa, pb, tot);
        }
        if (more) {
            commit(k0 + BK, stage ^ 1);
        }
        __syncthreads();
    }

    // epilogue through smem: element v of lane l in an 8x8 block is row (l & 1) + (v & 2) +
    // 4 (l >= 16), column (v & 4) + (l & 2) + (v & 1)
    float * E = (float *) sm;
    if constexpr (M16) {
        const int g = lane >> 2, tg = lane & 3;
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < NT8; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) {
                    const int r = mbase + 16 * i + g + 8 * (v >> 1);
                    const int c = wn * WN + 8 * j + 2 * tg + (v & 1);
                    E[r * S::LDE + c] = tot16[i][j][v];
                }
    } else {
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const int r = mbase + 8 * ((q & 1) + 2 * i) + (lane & 1) + (v & 2) + ((lane >> 4) << 2);
                const int c = wn * WN + 8 * ((q >> 1) * NJ + j) + (v & 4) + (lane & 2) + (v & 1);
                E[r * S::LDE + c] = tot[i][j][v];
            }
    }
    __syncthreads();
    store_tile<TM, BN, S::LDE, NT, K, true>(p, E, r0, ne, rt);
}

// sm_80+: weights staged raw through a cp.async ring (gemm_raw_kernel).
#include "sm80/moe_gemm_raw.cuh"

// sm_80+: B fragments dequantized from raw blocks in shared memory (gemm_frag_kernel).
#include "sm80/moe_gemm_frag.cuh"

// sm_80+, GLM-5.3-Flash only: IQ2 experts' B fragments from raw super-blocks (legion_moe_gemm_glm).
#include "sm80/moe_gemm_glm.cuh"

// sm_80+, GLM-5.3-Flash only, opt-in: the same GEMMs on int8 tensor cores (legion_moe_glm_q8).
#include "sm80/moe_gemm_glm_q8.cuh"

// ---- step 6 ---------------------------------------------------------------------------------

// Pairs per token the four-column reduce takes (its loads are all issued before the sum).
constexpr int RED_MAX_USED = 16;

// Whether reduce4 takes the token sums: whole four-column groups and aligned rows (F32 D: 16
// bytes, F16 / BF16 D: 8), on compute capability 12.x (the GB10, where it was measured: #526,
// 16-bit D #825).
template <int FMT> __device__ __forceinline__ bool reduce4_ok(const legion_moe_args & p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const bool sh_ok = !p.sh || (p.sh_ld % 4 == 0 && ((uintptr_t) p.sh & 15) == 0);
    return p.d_fmt == FMT && p.n_used <= RED_MAX_USED && p.n_embd % 4 == 0 &&
           ((uintptr_t) p.d & (FMT == 0 ? 15 : 7)) == 0 && ((uintptr_t) p.out & 15) == 0 && sh_ok;
#else
    return false;
#endif
}

// Four of D's values as floats: F32 as stored, F16 / BF16 (four 16-bit values in a uint2)
// converted exactly, as load_d converts them.
template <int FMT> using D4 = std::conditional_t<FMT == 0, float4, uint2>;
template <int FMT> __device__ __forceinline__ float4 d4_float(const D4<FMT> & v) {
    if constexpr (FMT == 0) {
        return v;
    } else if constexpr (FMT == 1) {
        const float2 a = __half22float2(*(const __half2 *) &v.x), b = __half22float2(*(const __half2 *) &v.y);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *) &v.x);
        const float2 b = __bfloat1622float2(*(const __nv_bfloat162 *) &v.y);
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

// reduce_kernel's sums four columns at a time: the token's valid pairs once (a bit each), then
// each pair's four values in one 16-byte streaming load, all of them issued before the sum, so
// a thread has the token's whole column group in flight instead of one load per pair in turn
// (each behind its ids and table reads). The same values summed in the same order and the
// shared expert added the same way: the same bits. FMT: D's d_fmt (16-bit D: one 8-byte load
// per pair and column group, the values converted as load_d does).
template <int FMT>
__device__ __forceinline__ void reduce4(const legion_moe_args & p, int64_t t, float sgate) {
    uint32_t valid = 0;
    for (int j = 0; j < p.n_used; j++) {
        if (expert_ok(p, p.ids[t * p.ids_ld + j])) {
            valid |= 1u << j;
        }
    }
    const int64_t n4 = p.n_embd / 4;
    const int64_t d0 = t * p.n_used * p.n_embd;
    const D4<FMT> * d = FMT == 0 ? (const D4<FMT> *) (p.d + d0) : (const D4<FMT> *) ((const uint16_t *) p.d + d0);
    for (int64_t c = threadIdx.x; c < n4; c += blockDim.x) {
        D4<FMT> v[RED_MAX_USED];
#pragma unroll
        for (int j = 0; j < RED_MAX_USED; j++) {
            if ((valid >> j) & 1u) {
                v[j] = __ldcs(d + (int64_t) j * n4 + c);
            }
        }
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool any = false;
#pragma unroll
        for (int j = 0; j < RED_MAX_USED; j++) {
            if ((valid >> j) & 1u) {
                const float4 f = d4_float<FMT>(v[j]);
                if (any) {
                    acc.x = acc.x + f.x;
                    acc.y = acc.y + f.y;
                    acc.z = acc.z + f.z;
                    acc.w = acc.w + f.w;
                } else {
                    acc = f;
                }
                any = true;
            }
        }
        if (p.sh) {
            const float4 s = ((const float4 *) (p.sh + t * p.sh_ld))[c];
            acc.x = __fadd_rn(acc.x, __fmul_rn(s.x, sgate));
            acc.y = __fadd_rn(acc.y, __fmul_rn(s.y, sgate));
            acc.z = __fadd_rn(acc.z, __fmul_rn(s.z, sgate));
            acc.w = __fadd_rn(acc.w, __fmul_rn(s.w, sgate));
        }
        ((float4 *) (p.out + t * p.n_embd))[c] = acc;
    }
}

// FMT: p.d_fmt (legion_moe_reduce picks the instance), so the F32 kernel keeps its registers.
template <int FMT> __global__ void reduce_kernel(const legion_moe_args p) {
    const int64_t t = blockIdx.x;
    // The shared expert's gate: ggml's op_sigmoid (unary.cu), the same expression and flags.
    const float sgate = p.sh ? 1.0f / (1.0f + expf(-p.sg[t * p.sg_ld])) : 0.0f;
    if (reduce4_ok<FMT>(p)) {
        reduce4<FMT>(p, t, sgate);
        return;
    }
    for (int64_t c = threadIdx.x; c < p.n_embd; c += blockDim.x) {
        const int64_t d0 = t * p.n_used * p.n_embd + c;
        // The pairs in order, as before: the first valid one starts the sum (only pairs left
        // out by expert_ok are skipped; their rows of d are never written).
        float acc = 0.0f;
        bool any = false;
        for (int j = 0; j < p.n_used; j++) {
            if (expert_ok(p, p.ids[t * p.ids_ld + j])) {
                const float v = load_d(p, d0 + (int64_t) j * p.n_embd);
                acc = any ? acc + v : v;
                any = true;
            }
        }
        if (p.sh) {
            // ggml's mul (sh times the gate) and add (the routed sum plus that), each rounded on
            // its own: no FMA.
            acc = __fadd_rn(acc, __fmul_rn(p.sh[t * p.sh_ld + c], sgate));
        }
        p.out[t * p.n_embd + c] = acc;
    }
}

// legion_moe_down_sum_init (#826): per token, its valid pairs as the count of each column
// block's counter; a token without any gets out as reduce_kernel makes it (0, plus the shared
// expert). One thread per token.
__global__ void down_sum_init_kernel(const legion_moe_args p) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n_tok) {
        return;
    }
    int nv = 0;
    for (int j = 0; j < p.n_used; j++) {
        nv += expert_ok(p, p.ids[t * p.ids_ld + j]) ? 1 : 0;
    }
    const int64_t nb = p.n_embd / DSUM_BN;
    for (int64_t b = 0; b < nb; b++) {
        p.sum_cnt[t * nb + b] = nv;
    }
    if (nv == 0) {
        const float sgate = shexp_gate(p, t);
        for (int64_t c = 0; c < p.n_embd; c++) {
            p.out[t * p.n_embd + c] = p.sh ? __fadd_rn(0.0f, __fmul_rn(p.sh[t * p.sh_ld + c], sgate)) : 0.0f;
        }
    }
}

// ---- launchers ------------------------------------------------------------------------------

// The compiled tile: TM pairs x BN weight rows, warps of WM x WN.
constexpr int TM = 64, BN = 128, WM = 32, WN = 32;

// Whether the current device runs the m16n8k16 GEMM: compute capability 8.0 or newer and code
// for it in the binary (else a JIT of older PTX, where its MMA is a trap). Decided once per
// device.
bool use_mma16() {
    static std::atomic<int> state[64];  // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        return false;
    }
    int s = state[dev].load(std::memory_order_acquire);
    if (s == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaFuncAttributes fa{};
        const bool code = cudaFuncGetAttributes(&fa, gemm_kernel<TM, BN, WM, WN, T_Q4_K, GU, true>) == cudaSuccess &&
                          fa.binaryVersion >= 80 && fa.ptxVersion >= 80;
        (void) cudaGetLastError();
        s = major >= 8 && code ? 1 : 2;
        state[dev].store(s, std::memory_order_release);
    }
    return s == 1;
}

template <int WT, Kind K, bool M16>
int launch_gemm_as(const legion_moe_args & a, dim3 grid, cudaStream_t st) {
    using S = Shape<TM, BN, WM, WN>;
    // The shared memory limit is set through the current device's instance of the kernel: set
    // it on each device's first launch, not once per process (a pipeline's second GPU launches
    // the same kernel), and report a failure instead of launching anyway.
    constexpr int MAX_DEVICES = 64;
    static std::atomic<bool> set[MAX_DEVICES];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= MAX_DEVICES) {
        return (int) cudaErrorInvalidDevice;
    }
    if (!set[dev].load(std::memory_order_acquire)) {
        const cudaError_t e = cudaFuncSetAttribute(gemm_kernel<TM, BN, WM, WN, WT, K, M16>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, S::SMEM);
        if (e != cudaSuccess) {
            return (int) e;
        }
        set[dev].store(true, std::memory_order_release);
    }
    gemm_kernel<TM, BN, WM, WN, WT, K, M16><<<grid, S::NT, S::SMEM, st>>>(a);
    return (int) cudaGetLastError();
}

// Whether the raw-staged kernel takes these weights: a format it stages, K in whole stages, and
// 16-byte aligned rows and experts (cp.async). a.raw_off (--no-moe-raw-staging) keeps
// gemm_kernel (comparisons; it also turns the fragment kernels off, which only take what this
// kernel takes).
template <int WT, Kind K>
bool raw_ok(const legion_moe_args & a) {
    if constexpr (WT == T_Q4_K || WT == T_Q5_1 || WT == T_IQ2_S || WT == T_IQ2_XXS) {
        const bool off = a.raw_off != 0;
        const int64_t kd = K == GU ? a.n_embd : a.n_ff;
        const uint8_t * base = K == GU ? a.gate : a.down;
        const uint8_t * base2 = K == GU ? a.up : a.down;
        const int64_t nb1 = K == GU ? a.gu_nb1 : a.down_nb1, nb2 = K == GU ? a.gu_nb2 : a.down_nb2;
        const uintptr_t al = (uintptr_t) base | (uintptr_t) base2 | (uintptr_t) nb1 | (uintptr_t) nb2;
        return !off && kd % RawFmt<WT>::RK == 0 && al % RawFmt<WT>::CP == 0;
    }
    return false;
}

template <int WT, Kind K>
int launch_raw(const legion_moe_args & a, dim3 grid, cudaStream_t st) {
    if constexpr (WT == T_Q4_K || WT == T_Q5_1 || WT == T_IQ2_S || WT == T_IQ2_XXS) {
        using S = RawShape<TM, BN, WM, WN, WT>;
        constexpr int MAX_DEVICES = 64;
        static std::atomic<bool> set[MAX_DEVICES];
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= MAX_DEVICES) {
            return (int) cudaErrorInvalidDevice;
        }
        if (!set[dev].load(std::memory_order_acquire)) {
            const cudaError_t e = cudaFuncSetAttribute(gemm_raw_kernel<TM, BN, WM, WN, WT, K>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize, S::SMEM);
            if (e != cudaSuccess) {
                return (int) e;
            }
            set[dev].store(true, std::memory_order_release);
        }
        gemm_raw_kernel<TM, BN, WM, WN, WT, K><<<grid, S::NT, S::SMEM, st>>>(a);
        return (int) cudaGetLastError();
    }
    return -1;
}

// The fragment kernels' down shape: 128-row blocks (the raw kernel's grid), RK-value weight
// stages, XS X slices in the ring, MB blocks per SM asked of the compiler (__launch_bounds__).
// Q5_1 (118 registers, 49664 bytes) already runs two blocks per GB10 SM (one is 12% slower).
// Q8_0 takes 64-value stages to fit two as well (#826, see FragDown).
template <int WT>
struct DownCfg {
    static constexpr int RK = 128, XS = 3, MB = 1;
};
template <>
struct DownCfg<T_Q8_0> {
    static constexpr int RK = 64, XS = 3, MB = 2;
};
template <int WT>
using DownFrag = FragDown<WT, 128, DownCfg<WT>::RK, DownCfg<WT>::XS>;

// Whether a fragment kernel takes these weights: Q4_K gate/up (gemm_frag_kernel) or Q5_1 down
// (gemm_frag_down_kernel) that the raw kernel would take, Q5_K gate/up or Q8_0 down (#836 part
// 10, which the raw kernel does not stage) with rows and experts aligned for their copies (16
// and 8 bytes), in whole stages along K and whole row blocks, on a compute capability 12.x device
// (the GB10, where they were measured: #363) that grants their shared memory. Same output bits
// as gemm_kernel; --no-moe-raw-staging (a.raw_off) turns them all off.
// Q5_K and Q8_0 take them at the step sizes where they were measured faster on the GB10 (#836
// part 10, Qwen3.8-Flash-Next's shape): from 2 tokens (one token's 10 experts: within noise,
// down +4-9%), and Q8_0 down up to 4096 tokens (the GB10's default step; 6-13% faster there,
// 11-16% slower at 8192, an opt-in --max-batch-tokens).
constexpr int64_t FRAG_MIN_TOK = 2, FRAG_Q8_0_MAX_TOK = 4096;
template <int WT, Kind K>
bool frag_ok(const legion_moe_args & a) {
    if constexpr ((WT == T_Q4_K || WT == T_Q5_K) && K == GU) {
        static_assert(frag::TMF == TM, "the plan's tiles");
        static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES], smem_w[GGML_CUDA_MAX_DEVICES];
        const uintptr_t al = (uintptr_t) a.gate | (uintptr_t) a.up | (uintptr_t) a.gu_nb1 | (uintptr_t) a.gu_nb2;
        const bool staged =
            WT == T_Q4_K ? raw_ok<WT, K>(a) : a.raw_off == 0 && al % 16 == 0 && a.n_tok >= FRAG_MIN_TOK;
        return staged && a.n_embd % 256 == 0 && (2 * a.n_ff) % frag::BN_F == 0 &&
               legion_device::compute_capability() / 10 == 12 &&
               legion_device::smem_opt_in(smem, gemm_frag_kernel<WT, false>, frag::GUShape<WT>::SMEM) &&
               legion_device::smem_opt_in(smem_w, gemm_frag_kernel<WT, true>, frag::GUShape<WT>::SMEM);
    } else if constexpr ((WT == T_Q5_1 || WT == T_Q8_0) && K == DOWN) {
        static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES], smem_w[GGML_CUDA_MAX_DEVICES];
        const uintptr_t al = (uintptr_t) a.down | (uintptr_t) a.down_nb1 | (uintptr_t) a.down_nb2;
        const bool staged = WT == T_Q5_1 ? raw_ok<WT, K>(a)
                                         : a.raw_off == 0 && al % 8 == 0 && a.n_tok >= FRAG_MIN_TOK &&
                                               a.n_tok <= FRAG_Q8_0_MAX_TOK;
        return staged && a.n_ff % 128 == 0 && a.n_embd % 128 == 0 &&
               legion_device::compute_capability() / 10 == 12 &&
               legion_device::smem_opt_in(
                   smem, gemm_frag_down_kernel<WT, 128, DownCfg<WT>::RK, DownCfg<WT>::XS, DownCfg<WT>::MB, false>,
                   DownFrag<WT>::SMEM) &&
               legion_device::smem_opt_in(
                   smem_w, gemm_frag_down_kernel<WT, 128, DownCfg<WT>::RK, DownCfg<WT>::XS, DownCfg<WT>::MB, true>,
                   DownFrag<WT>::SMEM);
    }
    return false;
}

// The fragment kernels' grids: gate/up blocks are 256 rows wide, half the caller's row blocks
// (grid.y is one per plan tile, or frag_ycap rows that walk the tiles: the gate/up blocks of a
// group's other tiles return at once).
template <int WT, Kind K>
int launch_frag(const legion_moe_args & a, dim3 grid, cudaStream_t st) {
    const bool walk = a.frag_ycap > 0 && grid.y > (unsigned) a.frag_ycap;
    if (walk) {
        grid.y = (unsigned) a.frag_ycap;
    }
    if constexpr ((WT == T_Q4_K || WT == T_Q5_K) && K == GU) {
        grid.x = (unsigned) (2 * a.n_ff / frag::BN_F);
        (walk ? gemm_frag_kernel<WT, true> : gemm_frag_kernel<WT, false>)
            <<<grid, frag::NT, frag::GUShape<WT>::SMEM, st>>>(a);
    } else if constexpr ((WT == T_Q5_1 || WT == T_Q8_0) && K == DOWN) {
        grid.x = (unsigned) (a.n_embd / 128);
        (walk ? gemm_frag_down_kernel<WT, 128, DownCfg<WT>::RK, DownCfg<WT>::XS, DownCfg<WT>::MB, true>
              : gemm_frag_down_kernel<WT, 128, DownCfg<WT>::RK, DownCfg<WT>::XS, DownCfg<WT>::MB, false>)
            <<<grid, DownFrag<WT>::NT, DownFrag<WT>::SMEM, st>>>(a);
    } else {
        return -1;
    }
    return (int) cudaGetLastError();
}

template <int WT, Kind K>
int launch_gemm(const legion_moe_args & a, dim3 grid, cudaStream_t st) {
    if (K == DOWN && a.sum_cnt && !(use_mma16() && frag_ok<WT, K>(a))) {
        return -1;  // only the fragment down kernel makes the sums (legion_moe_down_sum)
    }
    if (!use_mma16()) {
        return launch_gemm_as<WT, K, false>(a, grid, st);
    }
    if (frag_ok<WT, K>(a)) {
        return launch_frag<WT, K>(a, grid, st);
    }
    if (!raw_ok<WT, K>(a)) {
        return launch_gemm_as<WT, K, true>(a, grid, st);
    }
    return launch_raw<WT, K>(a, grid, st);
}

template <Kind K>
int launch_typed(const legion_moe_args & a, int wtype, dim3 grid, cudaStream_t st) {
    switch (wtype) {
        case T_Q4_K: return launch_gemm<T_Q4_K, K>(a, grid, st);
        case T_Q5_K: return launch_gemm<T_Q5_K, K>(a, grid, st);
        case T_Q6_K: return launch_gemm<T_Q6_K, K>(a, grid, st);
        case T_Q8_0: return launch_gemm<T_Q8_0, K>(a, grid, st);
        case T_Q5_1: return launch_gemm<T_Q5_1, K>(a, grid, st);
        case T_IQ2_XXS: return launch_gemm<T_IQ2_XXS, K>(a, grid, st);
        case T_IQ2_S: return launch_gemm<T_IQ2_S, K>(a, grid, st);
        case T_IQ2_XS: return launch_gemm<T_IQ2_XS, K>(a, grid, st);
        case T_IQ1_M: return launch_gemm<T_IQ1_M, K>(a, grid, st);
        case T_Q2_0: return launch_gemm<T_Q2_0, K>(a, grid, st);
        case T_Q2_K: return launch_gemm<T_Q2_K, K>(a, grid, st);
        case T_Q3_K: return launch_gemm<T_Q3_K, K>(a, grid, st);
        case T_IQ3_XXS: return launch_gemm<T_IQ3_XXS, K>(a, grid, st);
        case T_IQ3_S: return launch_gemm<T_IQ3_S, K>(a, grid, st);
        case T_IQ4_XS: return launch_gemm<T_IQ4_XS, K>(a, grid, st);
        default: return -1;
    }
}

}  // namespace

int legion_moe_prep(const legion_moe_args * a, void * stream) {
    if (a->q8 & 1) {
        launch_pdl(prep_q8_kernel, dim3((unsigned) a->n_tok), 256, 0, (cudaStream_t) stream, *a);
        return (int) cudaGetLastError();
    }
    prep_kernel<<<(unsigned) a->n_tok, 256, 0, (cudaStream_t) stream>>>(*a);
    return (int) cudaGetLastError();
}

int legion_moe_plan(const legion_moe_args * a, void * stream) {
    if (a->n_expert > 4096) {
        return -1;
    }
    plan_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(*a);
    return (int) cudaGetLastError();
}

int legion_moe_scatter(const legion_moe_args * a, void * stream) {
    const int64_t np = a->n_tok * a->n_used;
    scatter_kernel<<<(unsigned) ((np + 255) / 256), 256, 0, (cudaStream_t) stream>>>(*a);
    return (int) cudaGetLastError();
}

int legion_moe_gemm(const legion_moe_args * a, int kind, int wtype, int bn, unsigned grid_x, unsigned grid_y,
                    void * stream) {
    if (a->tm != TM || bn != BN) {
        return -1;
    }
    const dim3 grid(grid_x, grid_y);
    cudaStream_t st = (cudaStream_t) stream;
    return kind == GU ? launch_typed<GU>(*a, wtype, grid, st) : launch_typed<DOWN>(*a, wtype, grid, st);
}

int legion_moe_mma16_supported(void) { return use_mma16() ? 1 : 0; }

int legion_moe_gemm_glm(const legion_moe_args * a, int kind, int wtype, int bn, unsigned grid_x, unsigned grid_y,
                        void * stream) {
    // the int8 GEMMs where legion_moe_glm_q8 gave the step its bits (gate/up writes H as Q8
    // rows when down takes them)
    if ((a->q8 & 1) && kind == GU) {
        const bool ok = (a->q8 & 2) ? glm_q8_launch<Q8Iq2xxs, GU, Q8Gu, true>(a, grid_y, stream)
                                    : glm_q8_launch<Q8Iq2xxs, GU, Q8Gu, false>(a, grid_y, stream);
        return ok ? (int) cudaGetLastError() : -1;
    }
    if ((a->q8 & 2) && kind == DOWN) {
        const bool ok = wtype == T_IQ3_XXS ? glm_q8_launch<Q8Iq3xxs, DOWN, Q8Down, false>(a, grid_y, stream)
                                           : glm_q8_launch<Q8Iq2s, DOWN, Q8Down, false>(a, grid_y, stream);
        return ok ? (int) cudaGetLastError() : -1;
    }
    if (a->tm == frag::TMF && use_mma16()) {
        // whole super-blocks per row, rows a whole number of 16-byte pieces (see the header)
        // steps of at most 16 tokens: the decode shape, the same bits (#594); configs from a GB10
        // nsys round-robin: register-staged (one shared slot per warp, so 16 warps per SM fit), gate/up
        // 4 warps x 4 super-blocks (212 us vs 274 us for 2 x 3 stages x 4), down 8 x 2 (145 vs 150 us);
        // IQ2_S down with per-row codebook copies, GlmIq2sDec (140 vs 144 us at one token, #871)
        if (kind == GU && wtype == T_IQ2_XXS && a->n_embd % 256 == 0 &&
            ((uintptr_t) a->gate | (uintptr_t) a->up | (uintptr_t) a->gu_nb1 | (uintptr_t) a->gu_nb2) % 16 == 0 &&
            a->gu_nb1 == a->n_embd / 256 * GlmIq2xxs::SB &&
            (glm_dec_launch<GlmIq2xxs, GU, 4, 1, 4, true>(a, grid_y, stream) ||
             glm_launch<GlmIq2xxs, GU, 128, 1, 3>(a, 2 * a->n_ff, grid_y, stream))) {
            return (int) cudaGetLastError();
        }
        if (kind == DOWN && wtype == T_IQ2_S && a->n_ff % 256 == 0 &&
            ((uintptr_t) a->down | (uintptr_t) a->down_nb1 | (uintptr_t) a->down_nb2) % 16 == 0 &&
            a->down_nb1 == a->n_ff / 256 * GlmIq2s::SB &&
            (glm_dec_launch<GlmIq2sDec, DOWN, 8, 1, 2, true>(a, grid_y, stream) ||
             glm_launch<GlmIq2s, DOWN, 256, 1, 4>(a, a->n_embd, grid_y, stream))) {
            return (int) cudaGetLastError();
        }
        // IQ3_XXS down (Maya-S's first and last four MoE layers): the decode shape (156 us vs
        // ~250 us on the generic kernel) and, for prompt steps, gemm_glm_kernel like IQ2_S down
        // (#853; the same bits as the generic kernel)
        if (kind == DOWN && wtype == T_IQ3_XXS && a->n_ff % 256 == 0 &&
            ((uintptr_t) a->down | (uintptr_t) a->down_nb1 | (uintptr_t) a->down_nb2) % 16 == 0 &&
            a->down_nb1 == a->n_ff / 256 * GlmIq3xxs::SB &&
            (glm_dec_launch<GlmIq3xxs, DOWN, 8, 1, 2, true>(a, grid_y, stream) ||
             glm_launch<GlmIq3xxs, DOWN, 256, 1, 4>(a, a->n_embd, grid_y, stream))) {
            return (int) cudaGetLastError();
        }
    }
    return legion_moe_gemm(a, kind, wtype, bn, grid_x, grid_y, stream);
}

int legion_moe_glm_q8(const legion_moe_args * a, int gu_type, int down_type) {
    using Dec = GlmDec<GlmIq2xxs, 2, 3, 4>;
    // prompt steps on sm_80+ only (the decode kernels keep steps of at most MAX_TOK tokens);
    // prep_q8_kernel reads x as float4, whole warps per row
    if (a->n_tok <= Dec::MAX_TOK || a->tm != frag::TMF || !use_mma16() || ((uintptr_t) a->x % 16) != 0 ||
        a->x_ld % 4 != 0) {
        return 0;
    }
    // the layouts legion_moe_gemm_glm checks for its IQ2 kernels
    const uintptr_t gu_al = (uintptr_t) a->gate | (uintptr_t) a->up | (uintptr_t) a->gu_nb1 | (uintptr_t) a->gu_nb2;
    const bool gu = gu_type == T_IQ2_XXS && a->n_embd % 256 == 0 && gu_al % 16 == 0 &&
                    a->gu_nb1 == a->n_embd / 256 * Q8Iq2xxs::SB;
    // down: IQ2_S, or IQ3_XXS (Maya-S's 8 IQ3_XXS-down layers, #594)
    const bool down_al = a->n_ff % 256 == 0 &&
                         ((uintptr_t) a->down | (uintptr_t) a->down_nb1 | (uintptr_t) a->down_nb2) % 16 == 0;
    const bool down_s = down_al && down_type == T_IQ2_S && a->down_nb1 == a->n_ff / 256 * Q8Iq2s::SB &&
                        glm_q8_ready<Q8Iq2s, DOWN, Q8Down, false>(a);
    const bool down_3 = down_al && down_type == T_IQ3_XXS && a->down_nb1 == a->n_ff / 256 * Q8Iq3xxs::SB &&
                        glm_q8_ready<Q8Iq3xxs, DOWN, Q8Down, false>(a);
    if (!gu || !glm_q8_ready<Q8Iq2xxs, GU, Q8Gu, false>(a)) {
        return 0;
    }
    if (!(down_s || down_3) || !glm_q8_ready<Q8Iq2xxs, GU, Q8Gu, true>(a)) {
        return 1;
    }
    return 3;
}

int legion_moe_reduce(const legion_moe_args * a, void * stream) {
    const auto k = a->d_fmt == 1 ? reduce_kernel<1> : a->d_fmt == 2 ? reduce_kernel<2> : reduce_kernel<0>;
    k<<<(unsigned) a->n_tok, 256, 0, (cudaStream_t) stream>>>(*a);
    return (int) cudaGetLastError();
}

int legion_moe_down_sum(const legion_moe_args * a, int down_type) {
    // one launch of the whole step's experts (the counts are set once), F32 rows as reduce4
    // reads them, float4 rows of D, out and the shared expert
    const bool sh_ok = !a->sh || (a->sh_ld % 4 == 0 && ((uintptr_t) a->sh & 15) == 0);
    if (!a->sum_cnt || a->q8 != 0 || a->d_fmt != 0 || a->tm != TM || a->n_used > DSUM_MAX_USED ||
        a->n_embd % DSUM_BN != 0 || a->e_lo != 0 || a->e_hi != a->n_expert || !sh_ok ||
        (((uintptr_t) a->d | (uintptr_t) a->out) & 15) != 0 || !use_mma16()) {
        return 0;
    }
    switch (down_type) {
        case T_Q5_1: return frag_ok<T_Q5_1, DOWN>(*a) ? 1 : 0;
        case T_Q8_0: return frag_ok<T_Q8_0, DOWN>(*a) ? 1 : 0;
        default: return 0;
    }
}

int legion_moe_down_sum_init(const legion_moe_args * a, void * stream) {
    down_sum_init_kernel<<<(unsigned) ((a->n_tok + 127) / 128), 128, 0, (cudaStream_t) stream>>>(*a);
    return (int) cudaGetLastError();
}
