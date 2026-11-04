// qwen4exp (Qwen3.8-Flash-Next) hyper-connection mixer in two kernels, for small steps (decode,
// MTP verify). The host side, the shapes and the CPU reference are in
// src/model/qwen4/hc.rs; the graph it replaces is `hc_mix` in src/model/qwen4.rs:
//
//   xn     = rms_norm(x[c]) * norm            per stream c of the hc streams, x [n_embd, hc, n]
//   lo     = silu((down . xn) / hc)           down: Q8_0 [n_embd * hc -> rank]
//   gate   = up . lo                          up:   Q8_0 [rank -> n_embd * hc]
//   mixed  = (1/hc) sum_c xn[c] * sigmoid(gate[c])
//   inject = inject_w . xn                    F32 [n_embd * hc -> hc], optional
//
// Three kernels:
// - `norm`: per (stream, token), the stream's RMS (block reduction) and xn, in F32 and quantized
//   to Q8_1 as ggml's quantize_q8_1 does.
// - `down`: a CTA per down row for every token of the step, K over four warps as ggml's mmvq
//   does (its Q8_0 x Q8_1 dot products, vecdotq.cuh, MIT), then SiLU; the inject rows (F32) a
//   CTA each. Compiled per token count, as Strata-AC922's hc read kernels are (502fed9, MIT).
// - `quantize_lo`: lo as Q8_1, once.
// - `up_mix`: per (token, column block), the block's hc x UP_COLS up rows (a few lanes each:
//   they are short) and the gated mix.
//
// With `ahead` (the default, #814) the Q8_0 path runs `down_ahead` and `up_mix_ahead` instead:
// `down_ahead` takes two rows a CTA (no idle half) and loads each thread's weights for the whole
// row (AHEAD_DOWN units: all of a Flash-Next row) into registers before the first dot product,
// instead of one unit's at a time; `up_mix_ahead` does the same for its lanes and makes lo's
// Q8_1 itself, in shared memory, while those loads are in flight (no `quantize_lo` launch).
// The same dot products added in the same order: the same bits. At C=1 decode `down` and
// `up_mix` read their weights at 27% and 49% of HBM bandwidth, one load chain per thread.
// On compute capability 7.x two variants are on by default (#814; ncu at Flash-Next's C=1 MTP
// shape found the kernels latency bound, warps waiting on L1TEX loads, not bandwidth bound):
// `norm_regs` loads gamma and the stream once, into registers (`--no-hc-norm-regs`), and at 1-2
// tokens `down_split` splits each down row's units over twice the threads (`--no-hc-down-split`).
// Since #814's second round `down_hoist` (down_ahead with its address math out of the loop,
// `--no-hc-down-hoist`) takes the split's place, and `up_mix_stage` stages each warp's up rows in
// shared memory with 16-byte loads (`--no-hc-up-stage`). All add the same products in the same
// order: the same bits.
// `l2_ahead` (`legion serve --hc-l2-prefetch`) adds CTAs to the kernel before each reader that
// read its weights into L2 meanwhile: the norm's for down's matrix, down_ahead's for up's.
//
// On the GB10 (compute capability 12.1) the same four steps run as `norm_g`, `down_g`,
// `quantize_lo_g` and `up_mix_g` (sm80/qwen4_hc.cuh): the same bits, the weights staged in
// shared memory and each kernel launched early (PDL) to stage its weights while the previous
// one runs. use_gb10() picks them per device. Where its shared memory has room (always for
// Flash-Next), `up_mix_g` makes lo's Q8_1 itself and `quantize_lo_g` is not launched (#827).
//
// BF16 down and up matrices (`bf16`: Qwen3.8-Flash-Next-NVFP4 ships its mixers in BF16; opt-in,
// `legion serve --hc-mix-bf16`, #748): `norm`, then `down` and `up_mix` with V-wide BF16 loads
// against the F32 xn and lo (as ggml's mmvf takes F32 activations; no Q8_1, no quantize_lo).
// On the GB10 they run as `norm_g` (no Q8_1), `down_bf16_g` and `up_mix_bf16_g`
// (sm80/qwen4_hc_bf16.cuh): the weights loaded into registers ahead of the PDL wait.

#include "device_cache.cuh"
#include "launch_pdl.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <atomic>
#include <cstdint>

extern "C" {

struct legion_hc_mix {
    const float * x;          // [n][hc][n_embd], x_nb2 bytes per token (rows contiguous)
    int64_t x_nb1, x_nb2;
    // With `out`, x is first combined with the previous block's output (hc_combine):
    // res[c] = out * (2 sigmoid(inject_prev[c] / hc)) + x[c], written to `res`, then mixed.
    const float * out;        // [n][n_embd], out_nb1 bytes per token, or null
    int64_t out_nb1;
    const float * inject_prev;  // [n][hc], inject_prev_nb1 bytes per token
    int64_t inject_prev_nb1;
    float * res;              // [n][hc][n_embd]
    const float * norm;       // [hc * n_embd]
    const char * down;        // Q8_0 (BF16 with `bf16`) [rank][hc * n_embd], down_nb1 bytes per row
    int64_t down_nb1;
    const char * up;          // Q8_0 (BF16 with `bf16`) [hc * n_embd][rank], up_nb1 bytes per row
    int64_t up_nb1;
    const float * inject_w;   // F32 [hc][hc * n_embd] (inject_nb1 bytes per row), or null
    int64_t inject_nb1;
    float * xn;               // scratch F32 [n][hc * n_embd]
    block_q8_1 * xq;          // scratch: xn as Q8_1, [n][hc * n_embd / 32]
    float * lo;               // scratch F32 [n][rank]
    block_q8_1 * lq;          // scratch: lo as Q8_1, [n][rank / 32]
    float * mixed;            // [n][n_embd]
    float * inject;           // [n][hc], when inject_w
    int32_t n_embd, hc, rank, n;
    float eps;
    int32_t bf16;             // nonzero: down and up are BF16
    int32_t ahead;            // nonzero: Q8_0 down and up_mix with their loads ahead (V100 path)
    block_q8_1 * mixed_q;     // or null: mixed as Q8_1 too, [n][n_embd / 32] (ahead path only)
    int32_t l2_ahead;         // ahead path: L2_DOWN | L2_UP, the next kernel's weights read into L2
    int32_t variant;          // ahead path: V_NORM_REGS | V_DOWN_SPLIT | V_DOWN_HOIST | V_UP_STAGE, with V_DEFAULTS where cc 7.x picks
};

// Launches the kernels on `stream`; returns a cudaError_t.
int legion_hc_mix_launch(const legion_hc_mix * p, void * stream);

}  // extern "C"

namespace {

constexpr int NORM_THREADS = 1024;  // per (stream, token): 80 Q8_1 blocks over 32 warps
constexpr int DOWN_THREADS = 128;   // per down row: K split over 4 warps, as ggml's mmvq does
constexpr int INJ_THREADS  = 256;   // per inject row (F32)
constexpr int UP_THREADS   = 256;   // per UP_COLS columns: hc x UP_COLS rows, 4 per warp
constexpr int UP_COLS      = 8;
constexpr int UP_LANES     = 8;     // lanes per up row (K = rank: a few Q8_0 blocks)
constexpr int MAX_HC       = 8;
constexpr int AHEAD_DOWN   = 10;    // down units per thread loaded at once: K = 10240 in one go
constexpr int AHEAD_UP     = 5;     // up units per lane loaded at once: rank = 320 in one go
constexpr int MAX_LQ       = 64;    // lo's Q8_1 blocks at most (rank <= 2048)
constexpr int L2_DOWN      = 1;     // l2_ahead: norm's extra CTAs read the down matrix into L2
constexpr int L2_UP        = 2;     // l2_ahead: down_ahead's extra CTAs read the up matrix into L2
constexpr int V_NORM_REGS   = 2;    // variant: norm_regs instead of norm (#814)
constexpr int V_DOWN_SPLIT  = 4;    // variant: down_split instead of down_ahead (#814)
constexpr int V_DEFAULTS    = 8;    // variant: the bits only on cc 7.x, down_split at <= 2 tokens
constexpr int V_DOWN_HOIST  = 16;   // variant: down_hoist instead of down_ahead (#814)
constexpr int V_UP_STAGE    = 32;   // variant: up_mix_stage instead of up_mix_ahead (#814)
constexpr int L2_NORM_CTAS = 64;    // norm's extra CTAs (NORM_THREADS each), over all tokens
constexpr int L2_DOWN_CTAS = 64;    // down_ahead's extra CTAs (INJ_THREADS each)

template <int N>
__device__ __forceinline__ float block_sum(float v, float * s) {
    v = warp_reduce_sum(v);
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (lane == 0) {
        s[warp] = v;
    }
    __syncthreads();
    v = lane < N / WARP_SIZE ? s[lane] : 0.0f;
    return warp_reduce_sum(v);
}

// block_sum<DOWN_THREADS> of each half of a 2 x DOWN_THREADS CTA: the same values in the same
// lanes of the same reductions, so the same bits.
__device__ __forceinline__ float halves_sum(float v, float * s) {
    constexpr int W = DOWN_THREADS / WARP_SIZE;
    v = warp_reduce_sum(v);
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (lane == 0) {
        s[warp] = v;
    }
    __syncthreads();
    v = lane < W ? s[warp / W * W + lane] : 0.0f;
    return warp_reduce_sum(v);
}

// Q8_1 of 32 consecutive values, one per lane (ggml's quantize_q8_1).
__device__ __forceinline__ void quantize_lane(float xi, block_q8_1 * b, int lane) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax       = warp_reduce_max(amax);
    sum        = warp_reduce_sum(sum);
    const float d = amax / 127.0f;
    b->qs[lane]   = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (lane == 0) {
        b->ds = make_half2(d, sum);
    }
}

// Reads the 32-byte sectors of `bytes` bytes at w into L2 (ld.global.cg, 4 bytes a sector) as
// worker `part` of `parts` of THREADS threads, U loads a thread in flight at a time (#814). A
// hint for the next kernel, which then finds its weights in L2 instead of waiting on DRAM: the
// values are folded into one word that is only stored on an impossible branch (bytes < 0), so
// nothing any kernel computes changes.
template <int THREADS, int U = 4>
__device__ __forceinline__ void l2_fetch(const char * w, int64_t bytes, int part, int parts) {
    const int64_t sectors = bytes / 32, step = (int64_t) parts * THREADS;
    unsigned acc = 0;
    for (int64_t s = (int64_t) part * THREADS + threadIdx.x; s < sectors; s += U * step) {
        unsigned v[U];
#pragma unroll
        for (int i = 0; i < U; i++) {
            v[i] = s + i * step < sectors ? __ldcg((const unsigned *) (w + (s + i * step) * 32)) : 0u;
        }
#pragma unroll
        for (int i = 0; i < U; i++) {
            acc ^= v[i];
        }
    }
    if (bytes < 0) {
        *(unsigned *) w = acc;
    }
}

// The bytes of a matrix of `rows` rows nb1 apart, `row_bytes` each.
__device__ __forceinline__ int64_t matrix_bytes(int64_t rows, int64_t nb1, int64_t row_bytes) {
    return (rows - 1) * nb1 + row_bytes;
}

// xn = rms_norm(x[c]) * norm for one (stream, token) per CTA, in F32 and as Q8_1. With
// l2_ahead & L2_DOWN, the CTAs past the hc streams read the down matrix into L2 for down_ahead.
__global__ void __launch_bounds__(NORM_THREADS) norm(const legion_hc_mix p) {
    __shared__ float red[NORM_THREADS / WARP_SIZE];
    if ((int) blockIdx.x >= p.hc) {
        const int per = gridDim.x - p.hc;
        l2_fetch<NORM_THREADS>(p.down, matrix_bytes(p.rank, p.down_nb1, p.hc * p.n_embd / QK8_0 * sizeof(block_q8_0)),
                               blockIdx.y * per + (blockIdx.x - p.hc), per * gridDim.y);
        return;
    }
    const int c = blockIdx.x, t = blockIdx.y, K = p.hc * p.n_embd;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const float * xc = (const float *) ((const char *) p.x + t * p.x_nb2 + c * p.x_nb1);
    // The stream after the combine, as ggml computes it: post = 2 * sigmoid(inject / hc)
    // (scale, sigmoid, scale ops), then hc_post's out * post + residual.
    const float * oc = p.out ? (const float *) ((const char *) p.out + t * p.out_nb1) : nullptr;
    float post = 0.0f;
    if (oc) {
        const float v = (1.0f / p.hc) * ((const float *) ((const char *) p.inject_prev + t * p.inject_prev_nb1))[c] + 0.0f;
        post          = 2.0f * (1.0f / (1.0f + expf(-v))) + 0.0f;
    }
    float * rc = oc ? p.res + ((int64_t) t * p.hc + c) * p.n_embd : nullptr;
    auto stream = [&](int i) {
        if (!oc) {
            return xc[i];
        }
        float sum = oc[i] * post;
        sum += xc[i];
        return sum;
    };
    float ss = 0.0f;
    for (int i = threadIdx.x; i < p.n_embd; i += NORM_THREADS) {
        const float xi = stream(i);
        if (rc) {
            rc[i] = xi;
        }
        ss += xi * xi;
    }
    const float scale = rsqrtf(block_sum<NORM_THREADS>(ss, red) / p.n_embd + p.eps);
    for (int b = warp; b < p.n_embd / QK8_1; b += NORM_THREADS / WARP_SIZE) {
        const int i = b * QK8_1 + lane, k = c * p.n_embd + i;
        const float v = scale * stream(i) * p.norm[k];
        p.xn[(int64_t) t * K + k] = v;
        quantize_lane(v, &p.xq[(int64_t) t * (K / QK8_1) + k / QK8_1], lane);
    }
}

// norm with gamma and the stream loaded once, into registers, before the RMS (#814, as norm_g
// in sm80/qwen4_hc.cuh does without its PDL): element m of a thread is i = threadIdx.x +
// m * NORM_THREADS in both passes, so the same values in the same order, the same bits, with
// one load round trip fewer. n_embd <= NORM_THREADS * G_NORM_M (launch_rows checks).
__global__ void __launch_bounds__(NORM_THREADS) norm_regs(const legion_hc_mix p) {
    constexpr int M = 4;
    __shared__ float red[NORM_THREADS / WARP_SIZE];
    const int c = blockIdx.x, t = blockIdx.y, K = p.hc * p.n_embd;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const float * xc = (const float *) ((const char *) p.x + t * p.x_nb2 + c * p.x_nb1);
    const float * oc = p.out ? (const float *) ((const char *) p.out + t * p.out_nb1) : nullptr;
    float g[M], xv[M], ov[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
        const int i = threadIdx.x + m * NORM_THREADS;
        const bool in = i < p.n_embd;
        g[m]  = in ? p.norm[c * p.n_embd + i] : 0.0f;
        xv[m] = in ? xc[i] : 0.0f;
        ov[m] = in && oc ? oc[i] : 0.0f;
    }
    float post = 0.0f;
    if (oc) {
        const float v = (1.0f / p.hc) * ((const float *) ((const char *) p.inject_prev + t * p.inject_prev_nb1))[c] + 0.0f;
        post          = 2.0f * (1.0f / (1.0f + expf(-v))) + 0.0f;
    }
    float * rc = oc ? p.res + ((int64_t) t * p.hc + c) * p.n_embd : nullptr;
    float ss = 0.0f;
#pragma unroll
    for (int m = 0; m < M; m++) {
        const int i = threadIdx.x + m * NORM_THREADS;
        if (i < p.n_embd) {
            float xi = xv[m];
            if (oc) {
                float sum = ov[m] * post;
                sum += xv[m];
                xi = sum;
            }
            xv[m] = xi;
            if (rc) {
                rc[i] = xi;
            }
            ss += xi * xi;
        }
    }
    const float scale = rsqrtf(block_sum<NORM_THREADS>(ss, red) / p.n_embd + p.eps);
#pragma unroll
    for (int m = 0; m < M; m++) {
        const int b = warp + m * (NORM_THREADS / WARP_SIZE);
        if (b < p.n_embd / QK8_1) {
            const int i = b * QK8_1 + lane, k = c * p.n_embd + i;
            const float v = scale * xv[m] * g[m];
            p.xn[(int64_t) t * K + k] = v;
            quantize_lane(v, &p.xq[(int64_t) t * (K / QK8_1) + k / QK8_1], lane);
        }
    }
}

// V BF16 weights at w as floats: one 16-byte load (V = 8) or one value (V = 1).
template <int V>
__device__ __forceinline__ void load_bf16(const __nv_bfloat16 * w, float * f) {
    static_assert(V == 8 || V == 1, "V");
    if constexpr (V == 8) {
        const uint4 u = *(const uint4 *) w;
        const __nv_bfloat162 * h = (const __nv_bfloat162 *) &u;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 v = __bfloat1622float2(h[i]);
            f[2 * i]       = v.x;
            f[2 * i + 1]   = v.y;
        }
    } else {
        f[0] = __bfloat162float(w[0]);
    }
}

// The dot product of V floats w with x (16-byte aligned when V = 8).
template <int V>
__device__ __forceinline__ float dot_f32(const float * w, const float * x) {
    float s = 0.0f;
    if constexpr (V == 8) {
        const float4 a = ((const float4 *) x)[0], b = ((const float4 *) x)[1];
        s = w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    } else {
        s = w[0] * x[0];
    }
    return s;
}

constexpr int Q8_VDR = VDR_Q8_0_Q8_1_MMVQ;

// IT units of a Q8_0 row, from unit u0 on, STRIDE units apart (a unit is VDR ints of a block,
// as ggml's mmvq splits a block over QI8_0 / VDR lanes), loaded into registers before any is
// used; units at or past `units` are left out.
template <int IT, int STRIDE>
__device__ __forceinline__ void load_units(const char * row, int u0, int units, int (&v)[IT][Q8_VDR], float (&d)[IT]) {
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
#pragma unroll
    for (int i = 0; i < IT; i++) {
        const int u = u0 + i * STRIDE;
        if (u < units) {
            const block_q8_0 * b = (const block_q8_0 *) row + u / lanes_per_block;
#pragma unroll
            for (int k = 0; k < Q8_VDR; k++) {
                v[i][k] = get_int_b2(b->qs, Q8_VDR * (u % lanes_per_block) + k);
            }
            d[i] = b->d;
        }
    }
}

// Those units' dot products with the NT tokens' Q8_1 activations (token t's blocks from
// q + t * q_nt), added to s unit by unit, token by token: vec_dot_q8_0_q8_1's arithmetic (ggml,
// vecdotq.cuh, MIT) in the order the one-unit-at-a-time loops add them.
template <int NT, int IT, int STRIDE>
__device__ __forceinline__ void dot_units(const int (&v)[IT][Q8_VDR], const float (&d)[IT], int u0, int units,
                                          const block_q8_1 * q, int64_t q_nt, float (&s)[NT]) {
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
#pragma unroll
    for (int i = 0; i < IT; i++) {
        const int u = u0 + i * STRIDE;
        if (u < units) {
            const int kbx = u / lanes_per_block, iqs = Q8_VDR * (u % lanes_per_block);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const block_q8_1 * b = q + t * q_nt + kbx;
                int x[Q8_VDR];
#pragma unroll
                for (int k = 0; k < Q8_VDR; k++) {
                    x[k] = get_int_b4(b->qs, iqs + k);
                }
                s[t] += vec_dot_q8_0_q8_1_impl<float, Q8_VDR>(v[i], x, d[i], __low2half(b->ds));
            }
        }
    }
}

// inject[h] = inject_w[h] . xn for all NT tokens, a CTA (INJ_THREADS threads).
template <int NT>
__device__ __forceinline__ void inject_row(const legion_hc_mix & p, int h, float * red) {
    const int K = p.hc * p.n_embd;
    const float * w = (const float *) ((const char *) p.inject_w + h * p.inject_nb1);
    float s[NT] = {};
    for (int k = threadIdx.x; k < K; k += INJ_THREADS) {
        const float wk = w[k];
#pragma unroll
        for (int t = 0; t < NT; t++) {
            s[t] += wk * p.xn[(int64_t) t * K + k];
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = block_sum<INJ_THREADS>(s[t], red);
        if (threadIdx.x == 0) {
            p.inject[t * p.hc + h] = v;
        }
    }
}

// lo = silu((down . xn) / hc), a CTA per row for all NT tokens of the step (the row read once);
// then the inject rows (F32), a CTA each. Each token's sums run in the same order whatever NT.
// V = 0: Q8_0 rows against xn as Q8_1 (ggml's mmvq dot products); V = 8 or 1: BF16 rows, V
// weights per load, against the F32 xn.
template <int NT, int V = 0>
__global__ void __launch_bounds__(INJ_THREADS) down(const legion_hc_mix p) {
    __shared__ float red[INJ_THREADS / WARP_SIZE];
    const int K = p.hc * p.n_embd;
    if constexpr (V != 0) {
        if ((int) blockIdx.x < p.rank) {
            const int r = blockIdx.x;
            const __nv_bfloat16 * row = (const __nv_bfloat16 *) (p.down + r * p.down_nb1);
            float s[NT] = {};
#pragma unroll 4
            for (int u = threadIdx.x; u < K / V; u += INJ_THREADS) {
                float w[V];
                load_bf16<V>(row + u * V, w);
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    s[t] += dot_f32<V>(w, p.xn + (int64_t) t * K + u * V);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const float v = block_sum<INJ_THREADS>(s[t], red) * (1.0f / p.hc);
                if (threadIdx.x == 0) {
                    p.lo[(int64_t) t * p.rank + r] = v / (1.0f + expf(-v));
                }
            }
            return;
        }
    }
    if (V == 0 && (int) blockIdx.x < p.rank) {
        if (threadIdx.x >= DOWN_THREADS) {
            return;
        }
        constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, lanes_per_block = QI8_0 / VDR;
        const int r = blockIdx.x;
        const char * row = p.down + r * p.down_nb1;
        float s[NT] = {};
        for (int c = threadIdx.x; c < K / QK8_0 * lanes_per_block; c += DOWN_THREADS) {
            const int kbx = c / lanes_per_block;
#pragma unroll
            for (int t = 0; t < NT; t++) {
                s[t] += vec_dot_q8_0_q8_1(row, &p.xq[(int64_t) t * (K / QK8_1) + kbx], kbx,
                                          VDR * (c % lanes_per_block));
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = block_sum<DOWN_THREADS>(s[t], red) * (1.0f / p.hc);
            if (threadIdx.x == 0) {
                p.lo[(int64_t) t * p.rank + r] = v / (1.0f + expf(-v));
            }
        }
        return;
    }
    inject_row<NT>(p, blockIdx.x - p.rank, red);
}

// down<NT> for Q8_0 rows with the loads ahead: two rows a CTA, DOWN_THREADS threads each (no
// idle half, all CTAs in one wave), each thread's units loaded AHEAD_DOWN at a time before their
// dot products, then the inject rows as in down. The same sums in the same order. With
// l2_ahead & L2_UP, the CTAs past those read the up matrix into L2 for up_mix_ahead.
template <int NT>
__global__ void __launch_bounds__(INJ_THREADS, 3) down_ahead(const legion_hc_mix p) {
    __shared__ float red[INJ_THREADS / WARP_SIZE];
    const int K = p.hc * p.n_embd;
    const int fetch0 = p.rank / 2 + (p.inject_w ? p.hc : 0);
    if ((int) blockIdx.x >= fetch0) {
        l2_fetch<INJ_THREADS>(p.up, matrix_bytes(K, p.up_nb1, p.rank / QK8_0 * sizeof(block_q8_0)),
                              blockIdx.x - fetch0, gridDim.x - fetch0);
        return;
    }
    if ((int) blockIdx.x >= p.rank / 2) {
        inject_row<NT>(p, blockIdx.x - p.rank / 2, red);
        return;
    }
    const int half = threadIdx.x / DOWN_THREADS, tid = threadIdx.x % DOWN_THREADS;
    const int r = 2 * blockIdx.x + half;
    const char * row = p.down + r * p.down_nb1;
    const int units = K / QK8_0 * (QI8_0 / Q8_VDR);
    float s[NT] = {};
    for (int c0 = tid; c0 < units; c0 += AHEAD_DOWN * DOWN_THREADS) {
        int v[AHEAD_DOWN][Q8_VDR];
        float d[AHEAD_DOWN];
        load_units<AHEAD_DOWN, DOWN_THREADS>(row, c0, units, v, d);
        dot_units<NT, AHEAD_DOWN, DOWN_THREADS>(v, d, c0, units, p.xq, K / QK8_1, s);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = halves_sum(s[t], red) * (1.0f / p.hc);
        if (tid == 0) {
            p.lo[(int64_t) t * p.rank + r] = v / (1.0f + expf(-v));
        }
    }
}

// down_ahead<NT> with each row's K split over twice the threads (#814): one row a CTA, its first
// DOWN_THREADS threads take the units down_ahead's thread tid takes first (SPLIT_H of them), the
// other DOWN_THREADS the rest, whose per-unit factors (d8_0 * d8_1 and the dp4a sum, as
// vec_dot_q8_0_q8_1_impl computes them) go through shared memory to the first ones, which add
// them after their own as down_ahead's thread does: the same FMAs in the same order, so the same
// bits, in about half the serial chain per thread and twice the CTAs. ncu at Flash-Next's C=1
// MTP shape: down_ahead's warps are latency bound (20 cycles per instruction, 0.27 eligible warps
// per scheduler, L1TEX scoreboard stalls), not bandwidth bound. Every row's units fit one batch
// (units <= AHEAD_DOWN * DOWN_THREADS, launch_rows checks); then the inject rows as in down.
constexpr int SPLIT_H = AHEAD_DOWN / 2;
template <int NT>
__global__ void __launch_bounds__(INJ_THREADS, 5) down_split(const legion_hc_mix p) {
    constexpr int H2 = AHEAD_DOWN - SPLIT_H;
    __shared__ float red[INJ_THREADS / WARP_SIZE];
    __shared__ float2 part[H2][NT][DOWN_THREADS];
    const int K = p.hc * p.n_embd;
    if ((int) blockIdx.x >= p.rank) {
        inject_row<NT>(p, blockIdx.x - p.rank, red);
        return;
    }
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
    const int second = threadIdx.x / DOWN_THREADS, tid = threadIdx.x % DOWN_THREADS;
    const int r = blockIdx.x;
    const char * row = p.down + r * p.down_nb1;
    const int units = K / QK8_0 * lanes_per_block;
    const int64_t q_nt = K / QK8_1;
    float s[NT] = {};
    if (second) {
        int v[H2][Q8_VDR];
        float d[H2];
        load_units<H2, DOWN_THREADS>(row, tid + SPLIT_H * DOWN_THREADS, units, v, d);
#pragma unroll
        for (int i = 0; i < H2; i++) {
            const int u = tid + (SPLIT_H + i) * DOWN_THREADS;
            if (u < units) {
                const int kbx = u / lanes_per_block, iqs = Q8_VDR * (u % lanes_per_block);
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    const block_q8_1 * b = p.xq + t * q_nt + kbx;
                    int sumi = 0;
#pragma unroll
                    for (int k = 0; k < Q8_VDR; k++) {
                        sumi = ggml_cuda_dp4a(v[i][k], get_int_b4(b->qs, iqs + k), sumi);
                    }
                    const float d8_1 = __low2half(b->ds);
                    part[i][t][tid]  = make_float2(d[i] * d8_1, (float) sumi);
                }
            }
        }
    } else {
        int v[SPLIT_H][Q8_VDR];
        float d[SPLIT_H];
        load_units<SPLIT_H, DOWN_THREADS>(row, tid, units, v, d);
        dot_units<NT, SPLIT_H, DOWN_THREADS>(v, d, tid, units, p.xq, q_nt, s);
    }
    __syncthreads();
    if (!second) {
#pragma unroll
        for (int i = 0; i < H2; i++) {
            if (tid + (SPLIT_H + i) * DOWN_THREADS < units) {
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    const float2 ab = part[i][t][tid];
                    s[t]            = __fmaf_rn(ab.x, ab.y, s[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        // halves_sum's first half: down_ahead's block_sum<DOWN_THREADS> of the row's threads.
        const float v = halves_sum(s[t], red) * (1.0f / p.hc);
        if (threadIdx.x == 0) {
            p.lo[(int64_t) t * p.rank + r] = v / (1.0f + expf(-v));
        }
    }
}

// down_ahead with its per-unit address math out of the loop (#814, V_DOWN_HOIST): when a row's
// units are exactly one batch (units == AHEAD_DOWN * DOWN_THREADS, as for Flash-Next), thread
// tid's unit i is block tid / 4 + i * HOIST_BLOCKS of the row at int 2 * (tid % 4), so the block
// pointers are a constant stride apart and no unit needs a bounds check, a division or an int64
// product. The same values in the same order: the same bits. Measured faster than down_split at
// 1-4 tokens, so it replaces it in the defaults.
constexpr int HOIST_BLOCKS = DOWN_THREADS / (QI8_0 / Q8_VDR);  // blocks between a thread's units

// Thread tid's IT units of a down row (see HOIST_BLOCKS) into registers, as load_units does.
template <int IT>
__device__ __forceinline__ void load_row_units(const char * row, int tid, int (&v)[IT][Q8_VDR], float (&d)[IT]) {
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
    const block_q8_0 * b0 = (const block_q8_0 *) row + tid / lanes_per_block;
    const int iqs = Q8_VDR * (tid % lanes_per_block);
#pragma unroll
    for (int i = 0; i < IT; i++) {
        const block_q8_0 * b = b0 + i * HOIST_BLOCKS;
#pragma unroll
        for (int k = 0; k < Q8_VDR; k++) {
            v[i][k] = get_int_b2(b->qs, iqs + k);
        }
        d[i] = b->d;
    }
}

// dot_units for load_row_units' units: the same products added in the same order.
template <int NT, int IT>
__device__ __forceinline__ void dot_row_units(const int (&v)[IT][Q8_VDR], const float (&d)[IT], const block_q8_1 * q,
                                              int tid, int q_nt, float (&s)[NT]) {
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
    const block_q8_1 * q0 = q + tid / lanes_per_block;
    const int iqs = Q8_VDR * (tid % lanes_per_block);
#pragma unroll
    for (int i = 0; i < IT; i++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const block_q8_1 * b = q0 + i * HOIST_BLOCKS + t * q_nt;
            int x[Q8_VDR];
#pragma unroll
            for (int k = 0; k < Q8_VDR; k++) {
                x[k] = get_int_b4(b->qs, iqs + k);
            }
            s[t] += vec_dot_q8_0_q8_1_impl<float, Q8_VDR>(v[i], x, d[i], __low2half(b->ds));
        }
    }
}

// down_ahead<NT> hoisted (see HOIST_BLOCKS); no l2_ahead CTAs.
template <int NT>
__global__ void __launch_bounds__(INJ_THREADS, 3) down_hoist(const legion_hc_mix p) {
    __shared__ float red[INJ_THREADS / WARP_SIZE];
    if ((int) blockIdx.x >= p.rank / 2) {
        inject_row<NT>(p, blockIdx.x - p.rank / 2, red);
        return;
    }
    const int half = threadIdx.x / DOWN_THREADS, tid = threadIdx.x % DOWN_THREADS;
    const int r = 2 * blockIdx.x + half;
    int v[AHEAD_DOWN][Q8_VDR];
    float d[AHEAD_DOWN];
    load_row_units<AHEAD_DOWN>(p.down + (int64_t) r * p.down_nb1, tid, v, d);
    float s[NT] = {};
    dot_row_units<NT, AHEAD_DOWN>(v, d, p.xq, tid, p.hc * p.n_embd / QK8_1, s);
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = halves_sum(s[t], red) * (1.0f / p.hc);
        if (tid == 0) {
            p.lo[(int64_t) t * p.rank + r] = v / (1.0f + expf(-v));
        }
    }
}

// lo as Q8_1, once (a warp per block): read by every up_mix CTA.
__global__ void quantize_lo(const legion_hc_mix p) {
    const int t = blockIdx.y, b = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (b < p.rank / QK8_1) {
        quantize_lane(p.lo[(int64_t) t * p.rank + b * QK8_1 + threadIdx.x % WARP_SIZE],
                      &p.lq[(int64_t) t * (p.rank / QK8_1) + b], threadIdx.x % WARP_SIZE);
    }
}

// mixed = (1/hc) sum_c xn[c] * sigmoid(up . lo): per column block, its hc x UP_COLS up rows
// (UP_LANES lanes each) for all NT tokens. V as in `down`: Q8_0 rows against lo as Q8_1 (0), or
// BF16 rows against the F32 lo (8 or 1).
template <int NT, int V = 0>
__global__ void __launch_bounds__(UP_THREADS) up_mix(const legion_hc_mix p) {
    __shared__ float gate[NT][UP_COLS * MAX_HC];
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, lanes_per_block = QI8_0 / VDR;
    const int K = p.hc * p.n_embd;
    const int lane = threadIdx.x % WARP_SIZE;
    const int i0 = blockIdx.x * UP_COLS, ncols = min(UP_COLS, p.n_embd - i0);
    const int units = p.rank / QK8_0 * lanes_per_block, sub = threadIdx.x % UP_LANES;

    // Whole warps go round the loop together (the shuffles take every lane).
    for (int j0 = threadIdx.x / WARP_SIZE * (WARP_SIZE / UP_LANES); j0 < ncols * p.hc;
         j0 += UP_THREADS / UP_LANES) {
        const int j = j0 + lane / UP_LANES;
        float s[NT] = {};
        if (V != 0 && j < ncols * p.hc) {
            const int c = j / ncols, i = i0 + j % ncols;
            const __nv_bfloat16 * row =
                (const __nv_bfloat16 *) (p.up + (int64_t) (c * p.n_embd + i) * p.up_nb1);
            constexpr int VV = V == 0 ? 1 : V;
            for (int u = sub; u < p.rank / VV; u += UP_LANES) {
                float w[VV];
                load_bf16<VV>(row + u * VV, w);
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    s[t] += dot_f32<VV>(w, p.lo + (int64_t) t * p.rank + u * VV);
                }
            }
        } else if (j < ncols * p.hc) {
            const int c = j / ncols, i = i0 + j % ncols;
            const char * row = p.up + (int64_t) (c * p.n_embd + i) * p.up_nb1;
            for (int u = sub; u < units; u += UP_LANES) {
                const int kbx = u / lanes_per_block;
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    s[t] += vec_dot_q8_0_q8_1(row, &p.lq[(int64_t) t * (p.rank / QK8_1) + kbx], kbx,
                                              VDR * (u % lanes_per_block));
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            for (int off = UP_LANES / 2; off > 0; off >>= 1) {
                s[t] += __shfl_xor_sync(0xffffffff, s[t], off, WARP_SIZE);
            }
            if (sub == 0 && j < ncols * p.hc) {
                gate[t][j] = s[t];
            }
        }
    }
    __syncthreads();
    for (int jt = threadIdx.x; jt < ncols * NT; jt += UP_THREADS) {
        const int t = jt / ncols, j = jt % ncols, i = i0 + j;
        float sum = 0.0f;
        for (int c = 0; c < p.hc; c++) {
            const float g = gate[t][c * ncols + j];
            sum += p.xn[(int64_t) t * K + c * p.n_embd + i] * (1.0f / (1.0f + expf(-g)));
        }
        p.mixed[(int64_t) t * p.n_embd + i] = sum * (1.0f / p.hc);
    }
}

// up_mix (Q8_0) with each lane's units loaded AHEAD_UP at a time before their dot products, and
// lo's Q8_1 made per CTA in shared memory (quantize_lo's blocks, bit for bit) while the first
// loads are in flight. The same sums in the same order as up_mix<NT> after quantize_lo.
// QUANT (COLS = 32, a Q8_1 block of columns per CTA): mixed also as Q8_1 (#819), ggml's
// quantize_q8_1 of the values written (quantize_lane), for the block input's Q8_0 matvecs
// (src/model/core/q8_group.rs). A row's sums do not depend on the CTA's width, so the same bits.
template <int NT, int COLS = UP_COLS, int THREADS = UP_THREADS, bool QUANT = false>
__global__ void __launch_bounds__(THREADS) up_mix_ahead(const legion_hc_mix p) {
    __shared__ float gate[NT][COLS * MAX_HC];
    __shared__ block_q8_1 lq[NT][MAX_LQ];
    __shared__ float mixed_s[QUANT ? NT : 1][QUANT ? COLS : 1];
    static_assert(!QUANT || COLS == QK8_1, "a Q8_1 block of columns per CTA");
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
    const int K = p.hc * p.n_embd;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    const int i0 = blockIdx.x * COLS, ncols = min(COLS, p.n_embd - i0), rows = ncols * p.hc;
    const int units = p.rank / QK8_0 * lanes_per_block, sub = threadIdx.x % UP_LANES;
    const int nb = p.rank / QK8_1;
    auto row_of = [&](int j) {
        const int c = j / ncols, i = i0 + j % ncols;
        return p.up + (int64_t) (c * p.n_embd + i) * p.up_nb1;
    };

    // The first rows' first units on their way before lo is quantized.
    int j0 = warp * (WARP_SIZE / UP_LANES);
    int v[AHEAD_UP][Q8_VDR];
    float d[AHEAD_UP];
    bool loaded = false;
    if (j0 + lane / UP_LANES < rows) {
        load_units<AHEAD_UP, UP_LANES>(row_of(j0 + lane / UP_LANES), sub, units, v, d);
        loaded = true;
    }
    for (int b = warp; b < NT * nb; b += THREADS / WARP_SIZE) {
        const int t = b / nb, kb = b % nb;
        quantize_lane(p.lo[(int64_t) t * p.rank + kb * QK8_1 + lane], &lq[t][kb], lane);
    }
    __syncthreads();

    // Whole warps go round the loop together (the shuffles take every lane).
    for (; j0 < rows; j0 += THREADS / UP_LANES) {
        const int j = j0 + lane / UP_LANES;
        float s[NT] = {};
        if (j < rows) {
            for (int u0 = sub; u0 < units; u0 += AHEAD_UP * UP_LANES) {
                if (!loaded) {
                    load_units<AHEAD_UP, UP_LANES>(row_of(j), u0, units, v, d);
                }
                loaded = false;
                dot_units<NT, AHEAD_UP, UP_LANES>(v, d, u0, units, &lq[0][0], MAX_LQ, s);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            for (int off = UP_LANES / 2; off > 0; off >>= 1) {
                s[t] += __shfl_xor_sync(0xffffffff, s[t], off, WARP_SIZE);
            }
            if (sub == 0 && j < rows) {
                gate[t][j] = s[t];
            }
        }
    }
    __syncthreads();
    for (int jt = threadIdx.x; jt < ncols * NT; jt += THREADS) {
        const int t = jt / ncols, j = jt % ncols, i = i0 + j;
        float sum = 0.0f;
        for (int c = 0; c < p.hc; c++) {
            const float g = gate[t][c * ncols + j];
            sum += p.xn[(int64_t) t * K + c * p.n_embd + i] * (1.0f / (1.0f + expf(-g)));
        }
        const float m = sum * (1.0f / p.hc);
        p.mixed[(int64_t) t * p.n_embd + i] = m;
        if constexpr (QUANT) {
            mixed_s[t][j] = m;
        }
    }
    if constexpr (QUANT) {
        __syncthreads();
        for (int t = warp; t < NT; t += THREADS / WARP_SIZE) {
            quantize_lane(mixed_s[t][lane], &p.mixed_q[(int64_t) t * (p.n_embd / QK8_1) + blockIdx.x], lane);
        }
    }
}

// up_mix_ahead with each warp's up rows staged in shared memory by aligned 16-byte loads (#814,
// V_UP_STAGE): a warp's UP_ROWS_W rows (WARP_SIZE / UP_LANES) are consecutive columns of one
// stream, so one contiguous run of bytes, 16-byte aligned; ncu found up_mix_ahead waiting on its
// 2-byte weight loads (LDG.U16, a Q8_0 block's ints are only 2-byte aligned). Each lane then
// reads its units from shared memory as load_units does: the same values, the same sums in the
// same order, the same bits. For up_stage_fits' shapes (Flash-Next's): every row in one pass of
// the CTA's warps, one batch a lane, rank <= UP_STAGE_RANK, hc <= 4 and NT <= 4 (the static
// shared memory, 47.5 KB with COLS = 32 at 4 tokens).
constexpr int UP_ROWS_W     = WARP_SIZE / UP_LANES;
constexpr int UP_STAGE_RANK = 320;
constexpr int UP_STAGE_LQ   = UP_STAGE_RANK / QK8_1;
constexpr int UP_STAGE_U4   = UP_ROWS_W * (UP_STAGE_RANK / QK8_0) * (int) sizeof(block_q8_0) / 16;
constexpr int UP_STAGE_L    = (UP_STAGE_U4 + WARP_SIZE - 1) / WARP_SIZE;  // 16-byte loads a lane
static_assert(UP_ROWS_W * (UP_STAGE_RANK / QK8_0) * sizeof(block_q8_0) % 16 == 0, "16-byte runs");

template <int NT, int COLS = UP_COLS, int THREADS = UP_THREADS, bool QUANT = false>
__global__ void __launch_bounds__(THREADS) up_mix_stage(const legion_hc_mix p) {
    static_assert(NT <= 4, "the static shared memory");
    static_assert(!QUANT || COLS == QK8_1, "a Q8_1 block of columns per CTA");
    constexpr int HC = 4;
    __shared__ float gate[NT][COLS * HC];
    __shared__ block_q8_1 lq[NT][UP_STAGE_LQ];
    __shared__ float mixed_s[QUANT ? NT : 1][QUANT ? COLS : 1];
    __shared__ uint4 stage[THREADS / WARP_SIZE][UP_STAGE_U4];
    constexpr int lanes_per_block = QI8_0 / Q8_VDR;
    const int K = p.hc * p.n_embd;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    const int i0 = blockIdx.x * COLS, rows = COLS * p.hc;
    const int units = p.rank / QK8_0 * lanes_per_block, sub = threadIdx.x % UP_LANES;
    const int nb = p.rank / QK8_1;
    // The warp's rows j0 .. j0 + UP_ROWS_W: stream c, columns i0 + j0 % COLS on.
    const int j0 = warp * UP_ROWS_W;
    const bool mine = j0 < rows;
    const int n4 = UP_ROWS_W * (int) p.up_nb1 / 16;
    uint4 w[UP_STAGE_L];
    if (mine) {
        const uint4 * src = (const uint4 *) (p.up + (int64_t) ((j0 / COLS) * p.n_embd + i0 + j0 % COLS) * p.up_nb1);
#pragma unroll
        for (int l = 0; l < UP_STAGE_L; l++) {
            if (lane + l * WARP_SIZE < n4) {
                w[l] = src[lane + l * WARP_SIZE];
            }
        }
    }
    for (int b = warp; b < NT * nb; b += THREADS / WARP_SIZE) {
        const int t = b / nb, kb = b % nb;
        quantize_lane(p.lo[(int64_t) t * p.rank + kb * QK8_1 + lane], &lq[t][kb], lane);
    }
    if (mine) {
#pragma unroll
        for (int l = 0; l < UP_STAGE_L; l++) {
            if (lane + l * WARP_SIZE < n4) {
                stage[warp][lane + l * WARP_SIZE] = w[l];
            }
        }
    }
    __syncthreads();

    if (mine) {
        const int j = j0 + lane / UP_LANES;
        const char * row = (const char *) stage[warp] + (lane / UP_LANES) * p.up_nb1;
        int v[AHEAD_UP][Q8_VDR];
        float d[AHEAD_UP];
        load_units<AHEAD_UP, UP_LANES>(row, sub, units, v, d);
        float s[NT] = {};
        dot_units<NT, AHEAD_UP, UP_LANES>(v, d, sub, units, &lq[0][0], UP_STAGE_LQ, s);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            for (int off = UP_LANES / 2; off > 0; off >>= 1) {
                s[t] += __shfl_xor_sync(0xffffffff, s[t], off, WARP_SIZE);
            }
            if (sub == 0) {
                gate[t][j] = s[t];
            }
        }
    }
    __syncthreads();
    for (int jt = threadIdx.x; jt < COLS * NT; jt += THREADS) {
        const int t = jt / COLS, j = jt % COLS, i = i0 + j;
        float sum = 0.0f;
        for (int c = 0; c < p.hc; c++) {
            const float g = gate[t][c * COLS + j];
            sum += p.xn[(int64_t) t * K + c * p.n_embd + i] * (1.0f / (1.0f + expf(-g)));
        }
        const float m = sum * (1.0f / p.hc);
        p.mixed[(int64_t) t * p.n_embd + i] = m;
        if constexpr (QUANT) {
            mixed_s[t][j] = m;
        }
    }
    if constexpr (QUANT) {
        __syncthreads();
        for (int t = warp; t < NT; t += THREADS / WARP_SIZE) {
            quantize_lane(mixed_s[t][lane], &p.mixed_q[(int64_t) t * (p.n_embd / QK8_1) + blockIdx.x], lane);
        }
    }
}

// Whether up_mix_stage takes this mixer with COLS columns and THREADS threads a CTA.
bool up_stage_fits(const legion_hc_mix & p, int cols, int threads) {
    const int units = p.rank / QK8_0 * (QI8_0 / Q8_VDR);
    return p.n <= 4 && p.hc <= 4 && p.rank <= UP_STAGE_RANK && units <= AHEAD_UP * UP_LANES &&
           p.n_embd % cols == 0 && p.hc * cols <= threads / UP_LANES && (uintptr_t) p.up % 16 == 0 &&
           (UP_ROWS_W * p.up_nb1) % 16 == 0 && p.up_nb1 == p.rank / QK8_0 * (int64_t) sizeof(block_q8_0);
}

#include "sm80/qwen4_hc.cuh"
#include "sm80/qwen4_hc_bf16.cuh"

template <int NT>
void launch_rows(const legion_hc_mix & p, cudaStream_t s) {
    if (p.ahead) {
        const int fetch = p.l2_ahead & L2_UP ? L2_DOWN_CTAS : 0;
        const int units = p.hc * p.n_embd / QK8_0 * (QI8_0 / Q8_VDR);
        bool split = false;
        if constexpr (NT <= 4) {
            split = (p.variant & V_DOWN_SPLIT) && fetch == 0 && units <= AHEAD_DOWN * DOWN_THREADS;
        }
        // down_hoist takes rows of exactly one batch, and no l2_ahead CTAs; down_split goes first
        // when both bits are set (the defaults never set both).
        const bool hoist = (p.variant & V_DOWN_HOIST) && fetch == 0 && units == AHEAD_DOWN * DOWN_THREADS;
        const int n_inj = p.inject_w ? p.hc : 0;
        if (split) {
            if constexpr (NT <= 4) {
                down_split<NT><<<p.rank + n_inj, INJ_THREADS, 0, s>>>(p);
            }
        } else if (hoist) {
            down_hoist<NT><<<p.rank / 2 + n_inj, INJ_THREADS, 0, s>>>(p);
        } else {
            down_ahead<NT><<<p.rank / 2 + n_inj + fetch, INJ_THREADS, 0, s>>>(p);
        }
        // As many warps as UP_COLS-wide CTAs of UP_THREADS, a Q8_1 block's columns each.
        constexpr int qthreads = UP_THREADS * (QK8_1 / UP_COLS);
        const bool stage = p.variant & V_UP_STAGE;
        if constexpr (NT <= 4) {
            if (stage && p.mixed_q && up_stage_fits(p, QK8_1, qthreads)) {
                up_mix_stage<NT, QK8_1, qthreads, true><<<p.n_embd / QK8_1, qthreads, 0, s>>>(p);
                return;
            }
            if (stage && !p.mixed_q && up_stage_fits(p, UP_COLS, UP_THREADS)) {
                up_mix_stage<NT><<<p.n_embd / UP_COLS, UP_THREADS, 0, s>>>(p);
                return;
            }
        }
        if (p.mixed_q) {
            up_mix_ahead<NT, QK8_1, qthreads, true><<<p.n_embd / QK8_1, qthreads, 0, s>>>(p);
        } else {
            up_mix_ahead<NT><<<(p.n_embd + UP_COLS - 1) / UP_COLS, UP_THREADS, 0, s>>>(p);
        }
        return;
    }
    down<NT><<<p.rank + (p.inject_w ? p.hc : 0), INJ_THREADS, 0, s>>>(p);
    quantize_lo<<<dim3((p.rank / QK8_1 + 7) / 8, p.n), 8 * WARP_SIZE, 0, s>>>(p);
    up_mix<NT><<<(p.n_embd + UP_COLS - 1) / UP_COLS, UP_THREADS, 0, s>>>(p);
}

// BF16 down and up: no Q8_1 of lo to make, two kernels after the norm.
template <int NT, int V>
void launch_rows_bf16(const legion_hc_mix & p, cudaStream_t s) {
    down<NT, V><<<p.rank + (p.inject_w ? p.hc : 0), INJ_THREADS, 0, s>>>(p);
    up_mix<NT, V><<<(p.n_embd + UP_COLS - 1) / UP_COLS, UP_THREADS, 0, s>>>(p);
}

template <int V>
void launch_bf16(const legion_hc_mix & p, cudaStream_t s) {
    switch (p.n) {
        case 1: launch_rows_bf16<1, V>(p, s); break;
        case 2: launch_rows_bf16<2, V>(p, s); break;
        case 3: launch_rows_bf16<3, V>(p, s); break;
        case 4: launch_rows_bf16<4, V>(p, s); break;
        case 5: launch_rows_bf16<5, V>(p, s); break;
        case 6: launch_rows_bf16<6, V>(p, s); break;
        case 7: launch_rows_bf16<7, V>(p, s); break;
        case 8: launch_rows_bf16<8, V>(p, s); break;
    }
}

// Whether the BF16 kernels may load 8 weights (16 bytes) at a time: the weights, their rows and
// the F32 xn and lo they meet 16-byte aligned.
bool bf16_wide(const legion_hc_mix & p) {
    const int K = p.hc * p.n_embd;
    return (uintptr_t) p.down % 16 == 0 && p.down_nb1 % 16 == 0 && (uintptr_t) p.up % 16 == 0 &&
           p.up_nb1 % 16 == 0 && (uintptr_t) p.xn % 16 == 0 && (uintptr_t) p.lo % 16 == 0 && K % 8 == 0 &&
           p.rank % 8 == 0;
}

template <int NT>
void launch_rows_g(const legion_hc_mix & p, cudaStream_t s) {
    const int n_inj = p.inject_w ? p.hc * p.n : 0;
    launch_pdl(down_g<NT, G_ROWS>, dim3(n_inj + p.rank / G_ROWS), DOWN_THREADS, G_ROWS * (int) p.down_nb1, s, p,
               n_inj);
    const int rows = p.hc * G_COLS * (int) p.up_nb1;
    const int lq = NT * (p.rank / QK8_1) * (int) sizeof(block_q8_1);
    const int gate = NT * G_COLS * MAX_HC * (int) sizeof(float);  // up_mix_g's static shared memory
    if (rows + lq + gate <= 48 * 1024) {
        // lo's Q8_1 made by up_mix_g itself, after its up rows in shared memory (#827).
        launch_pdl(up_mix_g<NT, G_COLS, true>, dim3(p.n_embd / G_COLS), UP_THREADS, rows + lq, s, p);
        return;
    }
    launch_pdl(quantize_lo_g, dim3((p.rank / QK8_1 + 7) / 8, p.n), 8 * WARP_SIZE, 0, s, p);
    launch_pdl(up_mix_g<NT, G_COLS, false>, dim3(p.n_embd / G_COLS), UP_THREADS, rows, s, p);
}

// Whether the GB10 kernels take this mixer: on a GB10, the weights 16-byte aligned for the
// copies (rows and the CTAs' runs of them), gamma and the stream in norm_g's registers.
bool use_gb10(const legion_hc_mix & p) {
    const int64_t chunk = G_COLS * p.up_nb1;
    return (uintptr_t) p.down % 16 == 0 && p.down_nb1 % 16 == 0 && G_ROWS * p.down_nb1 <= 48 * 1024 &&
           p.rank % G_ROWS == 0 && (uintptr_t) p.up % 16 == 0 && chunk % 16 == 0 &&
           (p.n_embd * p.up_nb1) % 16 == 0 && p.hc * chunk <= 48 * 1024 && p.n_embd % G_COLS == 0 &&
           p.n_embd <= NORM_THREADS * G_NORM_M && gb10_device(up_mix_g<1, G_COLS, false>);
}

template <int NT>
void launch_rows_bf16_g(const legion_hc_mix & p, cudaStream_t s) {
    const int n_inj = p.inject_w ? p.hc * p.n : 0;
    launch_pdl(down_bf16_g<NT>, dim3(n_inj + p.rank / GB_DOWN_ROWS), GB_DOWN_THREADS, 0, s, p, n_inj);
    launch_pdl(up_mix_bf16_g<NT>, dim3(p.n_embd / G_COLS), UP_THREADS, 0, s, p);
}

// Whether the GB10 BF16 kernels take this mixer: on a GB10, the 16-byte loads aligned
// (bf16_wide), a down row in GB_DOWN_U loads per thread, an up row in GB_UP_U per lane, the
// CTA's hc x G_COLS up rows in GB_UP_PASSES passes, gamma and the stream in norm_g's registers.
bool use_gb10_bf16(const legion_hc_mix & p) {
    return bf16_wide(p) && p.hc * p.n_embd <= GB_DOWN_THREADS * 8 * GB_DOWN_U && p.rank % GB_DOWN_ROWS == 0 &&
           p.rank <= UP_LANES * 8 * GB_UP_U && p.hc * G_COLS <= GB_UP_PASSES * (UP_THREADS / UP_LANES) &&
           p.n_embd % G_COLS == 0 && p.n_embd <= NORM_THREADS * G_NORM_M && gb10_device(up_mix_bf16_g<1>);
}

}  // namespace

extern "C" int legion_hc_mix_launch(const legion_hc_mix * p, void * stream) {
    auto s = (cudaStream_t) stream;
    // The defaults' variant bits (#814): measured on the V100 (legion hc-mix-bench), where
    // norm_regs is 6-7% faster per mixer at 1-4 tokens and down_split 3-5% at 1-2 tokens
    // (flat at 3, 13% slower at 4); down_hoist is faster than both down_ahead and down_split at
    // 1-4 tokens, so it replaces the split where it is on, and up_mix_stage is 2-3% faster than
    // up_mix_ahead at 1-4 tokens.
    legion_hc_mix resolved = *p;
    if (resolved.variant & V_DEFAULTS) {
        constexpr int bits = V_NORM_REGS | V_DOWN_SPLIT | V_DOWN_HOIST | V_UP_STAGE;
        int v = legion_device::compute_capability() / 10 == 7 ? resolved.variant & bits : 0;
        if (resolved.n > 2 || (v & V_DOWN_HOIST)) {
            v &= ~V_DOWN_SPLIT;
        }
        resolved.variant = v;
    }
    p = &resolved;
    if (p->hc > MAX_HC || p->n_embd % QK8_1 != 0 || p->rank % 64 != 0 || p->rank > 64 * QK8_1) {
        return cudaErrorInvalidValue;
    }
    if (p->n < 1 || p->n > 8) {
        return cudaErrorInvalidValue;
    }
    // mixed's Q8_1 blocks come from the Q8_0 path with its loads ahead only.
    if (p->mixed_q && (p->bf16 || !p->ahead || use_gb10(*p))) {
        return cudaErrorInvalidValue;
    }
    if (p->bf16 && use_gb10_bf16(*p)) {
        norm_g<false><<<dim3(p->hc, p->n), NORM_THREADS, 0, s>>>(*p);
        switch (p->n) {
            case 1: launch_rows_bf16_g<1>(*p, s); break;
            case 2: launch_rows_bf16_g<2>(*p, s); break;
            case 3: launch_rows_bf16_g<3>(*p, s); break;
            case 4: launch_rows_bf16_g<4>(*p, s); break;
            case 5: launch_rows_bf16_g<5>(*p, s); break;
            case 6: launch_rows_bf16_g<6>(*p, s); break;
            case 7: launch_rows_bf16_g<7>(*p, s); break;
            case 8: launch_rows_bf16_g<8>(*p, s); break;
        }
        return cudaGetLastError();
    }
    if (p->bf16) {
        norm<<<dim3(p->hc, p->n), NORM_THREADS, 0, s>>>(*p);
        if (bf16_wide(*p)) {
            launch_bf16<8>(*p, s);
        } else {
            launch_bf16<1>(*p, s);
        }
        return cudaGetLastError();
    }
    if (use_gb10(*p)) {
        norm_g<true><<<dim3(p->hc, p->n), NORM_THREADS, 0, s>>>(*p);
        switch (p->n) {
            case 1: launch_rows_g<1>(*p, s); break;
            case 2: launch_rows_g<2>(*p, s); break;
            case 3: launch_rows_g<3>(*p, s); break;
            case 4: launch_rows_g<4>(*p, s); break;
            case 5: launch_rows_g<5>(*p, s); break;
            case 6: launch_rows_g<6>(*p, s); break;
            case 7: launch_rows_g<7>(*p, s); break;
            case 8: launch_rows_g<8>(*p, s); break;
        }
        return cudaGetLastError();
    }
    // The down matrix read into L2 by norm's extra CTAs, L2_NORM_CTAS over the tokens' rows.
    const int fetch = p->ahead && (p->l2_ahead & L2_DOWN) ? (L2_NORM_CTAS + p->n - 1) / p->n : 0;
    if (p->ahead && (p->variant & V_NORM_REGS) && fetch == 0 && p->n_embd <= NORM_THREADS * 4) {
        norm_regs<<<dim3(p->hc, p->n), NORM_THREADS, 0, s>>>(*p);
    } else {
        norm<<<dim3(p->hc + fetch, p->n), NORM_THREADS, 0, s>>>(*p);
    }
    switch (p->n) {
        case 1: launch_rows<1>(*p, s); break;
        case 2: launch_rows<2>(*p, s); break;
        case 3: launch_rows<3>(*p, s); break;
        case 4: launch_rows<4>(*p, s); break;
        case 5: launch_rows<5>(*p, s); break;
        case 6: launch_rows<6>(*p, s); break;
        case 7: launch_rows<7>(*p, s); break;
        case 8: launch_rows<8>(*p, s); break;
    }
    return cudaGetLastError();
}
