// Inkling-2's own ops (src/model/inkling2/ops.rs, whose CPU references define them): the depth
// mix of the attention residuals, the loop controllers' input and output, the causal short convs
// with their kept history, the L2 clips, the attention with its relative bias and per-token
// sinks, the sigmoid gate, the SITU activation, the expert matmul of the Q8_0 / MXFP8 / plain
// stacks and the soft-capped head. Kernel bodies only: the Rust side
// (src/model/inkling2/ops/cuda.rs) launches them with cudaLaunchKernel, each node described
// generically (data, shape, byte strides, type) as in legion_ink_args.
//
// Correctness first (plain SIMT); sums run in another order than the CPU's, so values match it
// to rounding, not bit for bit.

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" {

struct legion_ink_args {
    void *       dst;
    int64_t      ne[4];
    int64_t      nb[4];
    const void * src[10];
    int64_t      src_ne[10][4];
    int64_t      src_nb[10][4];
    int32_t      src_type[10];
    int32_t      n_src;
    int32_t      params[8];
};

}  // extern "C"

namespace {

using A = legion_ink_args;

constexpr int TYPE_F16  = 1;
constexpr int TYPE_Q8_0 = 8;
constexpr int TYPE_I8   = 24;
constexpr int TYPE_BF16 = 30;

// ink_mix's most states and queries.
constexpr int MIX_STATES  = 8;
constexpr int MIX_QUERIES = 4;
// ink_conv's most taps.
constexpr int CONV_TAPS = 8;
// ink_attn: query tokens per block, most rows (tokens x heads of one kv head), most relative
// bias ranks, the widest head.
constexpr int ATTN_TQ   = 4;
constexpr int ATTN_ROWS = 16;
constexpr int ATTN_DECODE_ROWS = 4;
constexpr int ATTN_NR   = 16;
constexpr int ATTN_D    = 128;
constexpr int ATTN_WARPS = 4;
// ink_mm_id: output rows per block, pairs per pass, pair list length.
constexpr int MM_ROWS = 64;
constexpr int MM_P    = 4;
constexpr int MM_LIST = 1024;

// A finite stand-in for -inf (--use_fast_math must not see infinities).
constexpr float NEG = -3.0e38f;

__device__ __forceinline__ const char * src(const A & a, int s) {
    return (const char *) a.src[s];
}

__device__ __forceinline__ float pf(int32_t p) {
    return __int_as_float(p);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it.
__device__ float block_sum(float v, float * sh) {
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = blockDim.x / 32;
    __syncthreads();
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    v = l < nw ? sh[l] : 0.f;
    return warp_sum(v);
}

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.f / (1.f + __expf(-x));
}

// FP8 E4M3 (FN) to float.
__device__ __forceinline__ float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) {
        v = ldexpf((float) m, -9);
    } else if (e == 15 && m == 7) {
        v = 0.f;  // NaN in E4M3FN; checkpoints have none
    } else {
        v = ldexpf((float) (8 + m), e - 10);
    }
    return s ? -v : v;
}

// ---- ink_mix: params ns, eps; sources: ns states [e, n], then the queries [e, nq] ----
__global__ void mix_kernel(const A a) {
    const int     ns  = a.params[0];
    const float   eps = pf(a.params[1]);
    const int64_t e = a.ne[0], n = a.ne[1], nq = a.ne[2];
    const int64_t t = blockIdx.x;
    // Each state's sum of squares (k = 0) and query dots (k = 1 + j): every thread's partials
    // of all of them in one pass over its elements, each in block_sum's order over them, then
    // block_sum's tree for each (#1221: not one block_sum after another, a memory latency
    // each, ~40 in a row for one token). The same sums, the same bits.
    constexpr int    NK = 1 + MIX_QUERIES;
    __shared__ float red[MIX_STATES * NK][32];
    __shared__ float sc[MIX_STATES][MIX_QUERIES];
    __shared__ float rs[MIX_STATES];
    const float inv = rsqrtf((float) e);
    float       part[MIX_STATES][NK];
#pragma unroll
    for (int s = 0; s < MIX_STATES; s++) {
#pragma unroll
        for (int k = 0; k < NK; k++) {
            part[s][k] = 0.f;
        }
    }
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
#pragma unroll
        for (int s = 0; s < MIX_STATES; s++) {
            if (s < ns) {
                const float * x = (const float *) (src(a, s) + t * a.src_nb[s][1]);
                part[s][0] += x[i] * x[i];
#pragma unroll
                for (int j = 0; j < MIX_QUERIES; j++) {
                    if (j < nq) {
                        const float * q = (const float *) (src(a, ns) + j * a.src_nb[ns][1]);
                        part[s][1 + j] += x[i] * q[i];
                    }
                }
            }
        }
    }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = blockDim.x / 32;
#pragma unroll
    for (int s = 0; s < MIX_STATES; s++) {
#pragma unroll
        for (int k = 0; k < NK; k++) {
            if (s < ns && k <= nq) {
                const float v = warp_sum(part[s][k]);
                if (l == 0) {
                    red[s * NK + k][w] = v;
                }
            }
        }
    }
    __syncthreads();
    if (w == 0) {
        for (int s = 0; s < ns; s++) {
            const float ss = warp_sum(l < nw ? red[s * NK][l] : 0.f);
            const float r  = rsqrtf(ss / (float) e + eps);
            for (int j = 0; j < nq; j++) {
                const float d = warp_sum(l < nw ? red[s * NK + 1 + j][l] : 0.f);
                if (l == 0) {
                    sc[s][j] = d * r * inv;
                }
            }
            if (l == 0) {
                rs[s] = r;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float     m = sc[0][j];
        for (int s = 1; s < ns; s++) {
            m = fmaxf(m, sc[s][j]);
        }
        float sum = 0.f;
        for (int s = 0; s < ns; s++) {
            sc[s][j] = __expf(sc[s][j] - m);
            sum += sc[s][j];
        }
        for (int s = 0; s < ns; s++) {
            sc[s][j] = sc[s][j] / sum * rs[s];
        }
    }
    __syncthreads();
    float * out = (float *) a.dst;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        float xv[MIX_STATES];
        for (int s = 0; s < ns; s++) {
            xv[s] = ((const float *) (src(a, s) + t * a.src_nb[s][1]))[i];
        }
        for (int j = 0; j < nq; j++) {
            float o = 0.f;
            for (int s = 0; s < ns; s++) {
                o += sc[s][j] * xv[s];
            }
            out[(j * n + t) * e + i] = o;
        }
    }
}

// ---- ink_ctrl_in: params eps; sources cur, msg [e, n]; dst [2e, n] ----
__global__ void ctrl_in_kernel(const A a) {
    const float   eps = pf(a.params[0]);
    const int64_t e   = a.ne[0] / 2;
    const int64_t t   = blockIdx.x;
    __shared__ float red[32];
    float *          out = (float *) a.dst + t * 2 * e;
    for (int h = 0; h < 2; h++) {
        const float * x  = (const float *) src(a, h) + t * e;
        float         ss = 0.f;
        for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
            ss += x[i] * x[i];
        }
        ss            = block_sum(ss, red);
        const float r = rsqrtf(ss / (float) e + eps);
        for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
            out[h * e + i] = x[i] * r;
        }
    }
}

// ---- ink_ctrl_out: sources cur, msg, gain, forget (all [e, n]) ----
__global__ void ctrl_out_kernel(const A a) {
    const int64_t n = a.ne[0] * a.ne[1];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float c = ((const float *) src(a, 0))[i];
    const float m = ((const float *) src(a, 1))[i];
    const float g = tanhf(((const float *) src(a, 2))[i]);
    const float f = tanhf(((const float *) src(a, 3))[i]);
    ((float *) a.dst)[i] = c - f * c + g * m;
}

// One output of ink_conv: win[0..K-1) the K - 1 inputs before this token's, oldest first (its
// own, xv, goes into win[K - 1] here), tw the taps, dr the token's dynamic taps when dyn. The
// one expression of conv_kernel and conv_prompt_kernel, so they give the same bits.
__device__ __forceinline__ float conv_out(
    const float (&tw)[CONV_TAPS], float (&win)[CONV_TAPS], int K, float xv, const float * dr,
    bool dyn, float scale) {
    float o = 0.f;
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        if (j == K - 1) {
            win[j] = xv;
        }
        if (j < K) {
            o += tw[j] * win[j];
        }
    }
    if (dyn) {
#pragma unroll
        for (int lag = 0; lag < CONV_TAPS; lag++) {
            if (lag < K) {
                o += dr[lag] * win[K - 1 - lag];
            }
        }
    }
    return (xv + o) * scale;
}

// ---- ink_conv: params n_tok, off, has_dyn, scale; sources x [c, ng], w [K, c],
// state [K - 1, width, n_seqs] (written back), dyn [K * heads, ng]. One thread per (sequence,
// channel), the window in registers, token after token: a decode step's. ----
__global__ void conv_kernel(const A a) {
    const int64_t n_tok = a.params[0], off = a.params[1];
    const bool    dyn   = a.params[2] != 0;
    const float   scale = pf(a.params[3]);
    const int64_t c = a.ne[0];
    const int     K = (int) a.src_ne[1][0];
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t s  = blockIdx.y;
    if (ch >= c) {
        return;
    }
    float * hist = (float *) ((char *) a.src[2] + s * a.src_nb[2][2] + (off + ch) * a.src_nb[2][1]);
    const float * w = (const float *) (src(a, 1) + ch * a.src_nb[1][1]);
    float         tw[CONV_TAPS], win[CONV_TAPS];
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        tw[j]  = j < K ? w[j] : 0.f;
        win[j] = j < K - 1 ? hist[j] : 0.f;
    }
    int64_t hh = 0;
    if (dyn) {
        const int64_t heads = a.src_ne[3][0] / K;
        hh                  = ch / (c / heads);
    }
    const float * x   = (const float *) src(a, 0);
    float *       out = (float *) a.dst;
    for (int64_t i = 0; i < n_tok; i++) {
        const int64_t t  = s * n_tok + i;
        const float   xv = x[t * c + ch];
        const float * dr = dyn ? (const float *) (src(a, 3) + t * a.src_nb[3][1]) + K * hh : nullptr;
        out[t * c + ch]  = conv_out(tw, win, K, xv, dr, dyn, scale);
#pragma unroll
        for (int j = 0; j + 1 < CONV_TAPS; j++) {
            if (j + 1 < K) {
                win[j] = win[j + 1];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        if (j < K - 1) {
            hist[j] = win[j];
        }
    }
}

// ink_conv's input j of K - 1 before token i (position i - (K - 1) + j of sequence s): the
// step's own x, or the kept history for positions before the step.
__device__ __forceinline__ float conv_in(
    const A & a, const float * hist, int K, int64_t s, int64_t n_tok, int64_t c, int64_t ch,
    int64_t i, int j) {
    const int64_t p = i - (K - 1) + j;
    return p >= 0 ? ((const float *) src(a, 0))[(s * n_tok + p) * c + ch] : hist[i + j];
}

// ---- ink_conv over a prompt step (#1221): conv_kernel's outputs, a thread per (channel,
// token, sequence) with its window read from x and the history, instead of a thread per
// channel walking the tokens one by one (a few CTAs for the whole GPU, a memory latency a
// token). The history is left as it was: conv_hist_kernel writes it next. ----
__global__ void conv_prompt_kernel(const A a) {
    const int64_t n_tok = a.params[0], off = a.params[1];
    const bool    dyn   = a.params[2] != 0;
    const float   scale = pf(a.params[3]);
    const int64_t c = a.ne[0];
    const int     K = (int) a.src_ne[1][0];
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i  = blockIdx.y;
    const int64_t s  = blockIdx.z;
    if (ch >= c) {
        return;
    }
    const float * hist =
        (const float *) ((const char *) a.src[2] + s * a.src_nb[2][2] + (off + ch) * a.src_nb[2][1]);
    const float * w = (const float *) (src(a, 1) + ch * a.src_nb[1][1]);
    float         tw[CONV_TAPS], win[CONV_TAPS];
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        tw[j]  = j < K ? w[j] : 0.f;
        win[j] = j < K - 1 ? conv_in(a, hist, K, s, n_tok, c, ch, i, j) : 0.f;
    }
    int64_t hh = 0;
    if (dyn) {
        const int64_t heads = a.src_ne[3][0] / K;
        hh                  = ch / (c / heads);
    }
    const int64_t t  = s * n_tok + i;
    const float   xv = ((const float *) src(a, 0))[t * c + ch];
    const float * dr = dyn ? (const float *) (src(a, 3) + t * a.src_nb[3][1]) + K * hh : nullptr;
    ((float *) a.dst)[t * c + ch] = conv_out(tw, win, K, xv, dr, dyn, scale);
}

// ---- ink_conv's history after a prompt step: the last K - 1 inputs, as conv_kernel leaves
// it. A thread per (channel, sequence), after conv_prompt_kernel read the old one. ----
__global__ void conv_hist_kernel(const A a) {
    const int64_t n_tok = a.params[0], off = a.params[1];
    const int64_t c = a.ne[0];
    const int     K = (int) a.src_ne[1][0];
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t s  = blockIdx.y;
    if (ch >= c) {
        return;
    }
    float * hist = (float *) ((char *) a.src[2] + s * a.src_nb[2][2] + (off + ch) * a.src_nb[2][1]);
    float   v[CONV_TAPS];
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        v[j] = j < K - 1 ? conv_in(a, hist, K, s, n_tok, c, ch, n_tok, j) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < CONV_TAPS; j++) {
        if (j < K - 1) {
            hist[j] = v[j];
        }
    }
}

// ---- ink_clip: params thr, has_scale, has_tau, floor, alpha; sources x [d, h, n], scale [d, h],
// pos [n]. One warp per row (d at most 128). ----
__global__ void clip_kernel(const A a) {
    const float thr       = pf(a.params[0]);
    const bool  has_scale = a.params[1] != 0, has_tau = a.params[2] != 0;
    const float floor_    = pf(a.params[3]), alpha = pf(a.params[4]);
    const int64_t d = a.ne[0], h = a.ne[1], rows = h * a.ne[2];
    const int     lane = threadIdx.x % 32;
    const int64_t row  = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    const int64_t hd = row % h, t = row / h;
    const float * x  = (const float *) src(a, 0) + row * d;
    const float * sc = has_scale ? (const float *) src(a, 1) + hd * d : nullptr;
    float         v[4];
    float         ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        v[k]            = i < d ? x[i] * (sc ? sc[i] : 1.f) : 0.f;
        ss += v[k] * v[k];
    }
    ss               = warp_sum(ss);
    const float norm = sqrtf(ss) + 1e-6f;
    float       f    = fminf(fmaxf(thr / norm, 0.f), 1.f);
    if (has_tau) {
        const int32_t p = ((const int32_t *) src(a, 1 + (has_scale ? 1 : 0)))[t];
        const float   r = fmaxf(((float) p + 1.f) * (1.f / floor_), 1.f);
        f *= logf(r) * alpha + 1.f;
    }
    float * o = (float *) a.dst + row * d;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        if (i < d) {
            o[i] = v[k] * f;
        }
    }
}

// The cache cell of position p of sequence s through the block table in source ts.
__device__ __forceinline__ int64_t cell(const A & a, int ts, int64_t s, int64_t p, int block) {
    const int32_t * t = (const int32_t *) (src(a, ts) + s * a.src_nb[ts][1]);
    return (int64_t) t[p / block] * block + p % block;
}

// ---- ink_attn: params n_tok, scale, window, block, ring entries; sources q [d, nh, ng],
// kc, vc [d * nkv, cells] F16, wtable, pos [ng], r [nr, nh, ng], rel [nr, extent],
// sinks [nh, ng], after. A block per (kv head, ATTN_TQ tokens of one sequence): its rows are
// those tokens' heads of the kv head; each warp takes key chunks of 32 (a key per lane for the
// scores, DPL dims per lane for the values), with an online softmax per warp, then the warps'
// partial sums are combined with the sink. R bounds the rows the registers hold: ATTN_ROWS, or
// ATTN_DECODE_ROWS for a block of one token (a decode step's), whose rows past its kv head's
// heads are padding (#1221). Each row's sums are the same either way. ----
template <int DPL, int R>
__global__ void attn_kernel(const A a) {
    static_assert(R <= ATTN_ROWS, "attn_kernel: rows");
    constexpr int D     = 32 * DPL;
    const int64_t n_tok = a.params[0];
    const float   scale = pf(a.params[1]);
    const int64_t window = a.params[2];
    const int     block  = a.params[3];
    const int64_t entries = a.params[4];
    const int64_t wrap    = entries > 0 ? entries * block : INT64_MAX;
    const int64_t nh = a.ne[1];
    const int64_t nkv = a.src_ne[1][0] / D;
    const int     pk  = (int) (nh / nkv);
    const int     nr  = (int) a.src_ne[6][0];
    const int64_t extent = a.src_ne[6][1];
    const int64_t g      = blockIdx.x;
    const int64_t tiles  = (n_tok + ATTN_TQ - 1) / ATTN_TQ;
    const int64_t seq    = blockIdx.y / tiles;
    const int64_t tq0    = (blockIdx.y % tiles) * ATTN_TQ;
    const int     ntq    = (int) min((int64_t) ATTN_TQ, n_tok - tq0);
    const int     rows   = ntq * pk;
    const int     warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    __shared__ float   qs[ATTN_ROWS][ATTN_D];
    __shared__ float   rsm[ATTN_ROWS][ATTN_NR];
    __shared__ float   pm[ATTN_WARPS][ATTN_ROWS], pl[ATTN_WARPS][ATTN_ROWS];
    __shared__ float   pacc[ATTN_WARPS][ATTN_ROWS][ATTN_D];
    __shared__ int64_t tpos[ATTN_TQ], tlo[ATTN_TQ];

    const int32_t * pos = (const int32_t *) src(a, 4);
    if (threadIdx.x < ATTN_TQ) {
        const int     tq = threadIdx.x;
        const int64_t t  = seq * n_tok + tq0 + tq;
        if (tq < ntq) {
            const int64_t p = pos[t];
            tpos[tq]        = p;
            tlo[tq]         = window > 0 ? max((int64_t) 0, p - window + 1) : 0;
        } else {
            tpos[tq] = -1;
            tlo[tq]  = INT64_MAX;
        }
    }
    for (int i = threadIdx.x; i < ATTN_ROWS * D; i += blockDim.x) {
        const int r = i / D, k = i % D;
        float     v = 0.f;
        if (r < rows) {
            const int64_t t = seq * n_tok + tq0 + r / pk;
            const int64_t h = g * pk + r % pk;
            v               = ((const float *) src(a, 0))[(t * nh + h) * D + k];
        }
        qs[r][k] = v;
    }
    for (int i = threadIdx.x; i < ATTN_ROWS * ATTN_NR; i += blockDim.x) {
        const int r = i / ATTN_NR, k = i % ATTN_NR;
        float     v = 0.f;
        if (r < rows && k < nr) {
            const int64_t t = seq * n_tok + tq0 + r / pk;
            const int64_t h = g * pk + r % pk;
            v               = ((const float *) src(a, 5))[(t * nh + h) * nr + k];
        }
        rsm[r][k] = v;
    }
    __syncthreads();
    int64_t kstart = INT64_MAX, kend = -1;
    for (int tq = 0; tq < ntq; tq++) {
        kstart = min(kstart, tlo[tq]);
        kend   = max(kend, tpos[tq]);
    }

    float m[R], l[R], acc[R][DPL];
#pragma unroll
    for (int r = 0; r < R; r++) {
        m[r] = NEG;
        l[r] = 0.f;
#pragma unroll
        for (int j = 0; j < DPL; j++) {
            acc[r][j] = 0.f;
        }
    }
    const float * rel = (const float *) src(a, 6);
    const size_t  kvoff = (size_t) g * D * sizeof(__half);
    for (int64_t base = kstart + warp * 32; base <= kend; base += 32 * ATTN_WARPS) {
        const int64_t key   = base + lane;
        const bool    inkey = key <= kend;
        int64_t       c     = 0;
        float         s[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            s[r] = 0.f;
        }
        if (inkey) {
            c                 = cell(a, 3, seq, key % wrap, block);
            const char * krow = src(a, 1) + c * a.src_nb[1][1] + kvoff;
            for (int i = 0; i < D; i += 8) {
                const uint4    u  = *(const uint4 *) (krow + i * sizeof(__half));
                const __half2 * hp = (const __half2 *) &u;
                float           kf[8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 f = __half22float2(hp[j]);
                    kf[2 * j]      = f.x;
                    kf[2 * j + 1]  = f.y;
                }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    if (r < rows) {
                        float d = 0.f;
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            d += qs[r][i + j] * kf[j];
                        }
                        s[r] += d;
                    }
                }
            }
        }
        // Scores, bias and mask; the chunk's max per row.
        bool valid[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            valid[r] = false;
            if (r < rows && inkey) {
                const int tq = r / pk;
                valid[r]     = key >= tlo[tq] && key <= tpos[tq];
                if (valid[r]) {
                    const int64_t dist = tpos[tq] - key;
                    float         b    = 0.f;
                    if (dist < extent) {
                        const float * rr = rel + dist * nr;
                        for (int k = 0; k < nr; k++) {
                            b += rsm[r][k] * rr[k];
                        }
                    }
                    s[r] = s[r] * scale + b;
                }
            }
            const float cm = warp_max(valid[r] ? s[r] : NEG);
            const float mn = fmaxf(m[r], cm);
            const float corr = __expf(m[r] - mn);
            m[r]             = mn;
            l[r] *= corr;
#pragma unroll
            for (int j = 0; j < DPL; j++) {
                acc[r][j] *= corr;
            }
            s[r] = valid[r] ? __expf(s[r] - mn) : 0.f;
            l[r] += s[r];
        }
        // Values: key kk of the chunk, DPL dims per lane.
        const int64_t nk = min((int64_t) 32, kend - base + 1);
        for (int kk = 0; kk < nk; kk++) {
            const int64_t ck   = __shfl_sync(0xffffffffu, c, kk);
            const __half * vr  = (const __half *) (src(a, 2) + ck * a.src_nb[2][1] + kvoff) + lane * DPL;
            float          vf[DPL];
#pragma unroll
            for (int j = 0; j < DPL; j++) {
                vf[j] = __half2float(vr[j]);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float p = __shfl_sync(0xffffffffu, s[r], kk);
#pragma unroll
                for (int j = 0; j < DPL; j++) {
                    acc[r][j] += p * vf[j];
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const float ls = warp_sum(l[r]);
        if (lane == 0) {
            pm[warp][r] = m[r];
            pl[warp][r] = ls;
        }
#pragma unroll
        for (int j = 0; j < DPL; j++) {
            pacc[warp][r][lane * DPL + j] = acc[r][j];
        }
    }
    __syncthreads();
    const float * sinks = (const float *) src(a, 7);
    for (int i = threadIdx.x; i < rows * D; i += blockDim.x) {
        const int     r = i / D, k = i % D;
        const int64_t t = seq * n_tok + tq0 + r / pk;
        const int64_t h = g * pk + r % pk;
        const float   sink = sinks[t * nh + h];
        float         M    = sink;
        for (int w = 0; w < ATTN_WARPS; w++) {
            M = fmaxf(M, pm[w][r]);
        }
        float den = __expf(sink - M), o = 0.f;
        for (int w = 0; w < ATTN_WARPS; w++) {
            const float f = __expf(pm[w][r] - M);
            den += pl[w][r] * f;
            o += pacc[w][r][k] * f;
        }
        ((float *) a.dst)[(t * nh + h) * D + k] = o / den;
    }
}

// ---- ink_sigmul: x * sigmoid(g) ----
__global__ void sigmul_kernel(const A a) {
    const int64_t n = a.ne[0] * a.ne[1] * a.ne[2] * a.ne[3];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ((float *) a.dst)[i] = ((const float *) src(a, 0))[i] * sigmoidf_(((const float *) src(a, 1))[i]);
}

// ---- ink_situ: params beta, lb; source y [2f, k, n] ----
__global__ void situ_kernel(const A a) {
    const float   beta = pf(a.params[0]), lb = pf(a.params[1]);
    const int64_t f = a.ne[0];
    const int64_t n = f * a.ne[1] * a.ne[2];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t row = i / f, c = i % f;
    const float * y   = (const float *) src(a, 0) + row * 2 * f;
    const float   g = y[c], u = y[f + c];
    const float   act = beta * tanhf(g / beta) * sigmoidf_(g);
    const float   uu  = lb > 0.f ? lb * tanhf(u / lb) : u;
    ((float *) a.dst)[i] = act * uu;
}

// Element `lane` of 32-value group gi of row r of expert e of the stack in source 0, times its
// scale (a Q8_0 block's, an MXFP8 group's from source 3).
__device__ __forceinline__ float mm_w(const A & a, int64_t e, int64_t r, int64_t gi, int lane) {
    const char * row = src(a, 0) + e * a.src_nb[0][2] + r * a.src_nb[0][1];
    switch (a.src_type[0]) {
        case TYPE_Q8_0: {
            const char * b = row + gi * 34;
            return __half2float(*(const __half *) b) * (float) ((const int8_t *) (b + 2))[lane];
        }
        case TYPE_I8: {
            const int64_t k = a.src_ne[0][0], m = a.src_ne[0][1];
            const float   s = ((const float *) src(a, 3))[(e * m + r) * (k / 32) + gi];
            return e4m3(((const uint8_t *) row)[gi * 32 + lane]) * s;
        }
        case TYPE_F16:
            return __half2float(((const __half *) row)[gi * 32 + lane]);
        case TYPE_BF16:
            return __bfloat162float(((const __nv_bfloat16 *) row)[gi * 32 + lane]);
        default:
            return ((const float *) row)[gi * 32 + lane];
    }
}

// ---- ink_mm_id: sources w [k, m, E], x [k, nx, n_tok], ids [n_used, n_tok] (I32,
// contiguous), scale. A block per (64 output rows, expert): it lists the pairs routed to its
// expert, then runs them MM_P at a time with their inputs in shared memory, a warp per row. ----
__global__ void mm_id_kernel(const A a) {
    extern __shared__ float xs[];
    __shared__ int list[MM_LIST];
    __shared__ int cnt;
    const int64_t k = a.src_ne[0][0], E = a.src_ne[0][2];
    const int64_t m = a.ne[0], n_used = a.ne[1], n_tok = a.ne[2];
    const int64_t nx = a.src_ne[1][1];
    const int64_t e = blockIdx.y, r0 = (int64_t) blockIdx.x * MM_ROWS;
    const int     warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const int32_t * ids = (const int32_t *) src(a, 2);
    const float *   x   = (const float *) src(a, 1);
    float *         out = (float *) a.dst;
    const int64_t   npairs = n_used * n_tok;
    for (int64_t c0 = 0; c0 < npairs; c0 += MM_LIST) {
        if (threadIdx.x == 0) {
            cnt = 0;
        }
        __syncthreads();
        const int64_t c1 = min(c0 + MM_LIST, npairs);
        for (int64_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
            const int32_t id = ids[i];
            if (id == e) {
                list[atomicAdd(&cnt, 1)] = (int) i;
            } else if (e == 0 && (id < 0 || id >= E)) {
                for (int rr = 0; rr < MM_ROWS && r0 + rr < m; rr++) {
                    out[i * m + r0 + rr] = 0.f;
                }
            }
        }
        __syncthreads();
        const int n = cnt;
        for (int b = 0; b < n; b += MM_P) {
            const int np = min(MM_P, n - b);
            __syncthreads();
            for (int64_t i = threadIdx.x; i < (int64_t) np * k; i += blockDim.x) {
                const int     p  = (int) (i / k);
                const int64_t pr = list[b + p];
                const int64_t t = pr / n_used, j = pr % n_used;
                xs[i]            = x[(t * nx + (nx == 1 ? 0 : j)) * k + i % k];
            }
            __syncthreads();
            for (int rr = warp; rr < MM_ROWS; rr += nw) {
                const int64_t r = r0 + rr;
                if (r >= m) {
                    break;
                }
                float acc[MM_P] = {};
                for (int64_t gi = 0; gi < k / 32; gi++) {
                    const float v = mm_w(a, e, r, gi, lane);
#pragma unroll
                    for (int p = 0; p < MM_P; p++) {
                        if (p < np) {
                            acc[p] += v * xs[p * k + gi * 32 + lane];
                        }
                    }
                }
#pragma unroll
                for (int p = 0; p < MM_P; p++) {
                    const float s = warp_sum(acc[p]);
                    if (lane == 0 && p < np) {
                        out[(int64_t) list[b + p] * m + r] = s;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ---- ink_softcap: params cap; source x [v, n]; dst [n_out, n] ----
__global__ void softcap_kernel(const A a) {
    const float   cap = pf(a.params[0]);
    const int64_t v = a.ne[0], n = a.ne[1];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v * n) {
        return;
    }
    const int64_t t = i / v, c = i % v;
    const float   x = ((const float *) (src(a, 0) + t * a.src_nb[0][1]))[c];
    ((float *) a.dst)[i] = cap > 0.f ? cap * tanhf(x / cap) : x;
}

}  // namespace

extern "C" const void * const legion_ink_kernels[17] = {
    reinterpret_cast<const void *>(&mix_kernel),
    reinterpret_cast<const void *>(&ctrl_in_kernel),
    reinterpret_cast<const void *>(&ctrl_out_kernel),
    reinterpret_cast<const void *>(&conv_kernel),
    reinterpret_cast<const void *>(&clip_kernel),
    reinterpret_cast<const void *>(&attn_kernel<1, ATTN_ROWS>),
    reinterpret_cast<const void *>(&attn_kernel<2, ATTN_ROWS>),
    reinterpret_cast<const void *>(&attn_kernel<4, ATTN_ROWS>),
    reinterpret_cast<const void *>(&sigmul_kernel),
    reinterpret_cast<const void *>(&situ_kernel),
    reinterpret_cast<const void *>(&mm_id_kernel),
    reinterpret_cast<const void *>(&softcap_kernel),
    reinterpret_cast<const void *>(&conv_prompt_kernel),
    reinterpret_cast<const void *>(&conv_hist_kernel),
    reinterpret_cast<const void *>(&attn_kernel<1, ATTN_DECODE_ROWS>),
    reinterpret_cast<const void *>(&attn_kernel<2, ATTN_DECODE_ROWS>),
    reinterpret_cast<const void *>(&attn_kernel<4, ATTN_DECODE_ROWS>),
};
