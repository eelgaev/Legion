// DeepSeek-V4.1-Flash's own ops (src/model/deepseek41/ops.rs, whose CPU references define them):
// RoPE with YaRN, the hyper-connection collapse, Engram's gated update, the latent cache writes,
// the compressor's pooling, the lightning indexer's scores, the candidate-block mask, the
// top-k selection, the sparse attention with sinks and the sqrt(softplus) router. Kernel bodies
// only: the Rust side (src/model/deepseek41/ops/cuda.rs) launches them with cudaLaunchKernel,
// each node described generically (data, shape, byte strides, type) as in legion_ds4_args.
//
// Correctness first (plain SIMT, sm_70 and up); sums run in another order than the CPU's, so
// values match it to rounding, not bit for bit.

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" {

struct legion_ds4_args {
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

using A = legion_ds4_args;

constexpr int TYPE_F16  = 1;
constexpr int TYPE_BF16 = 30;

__device__ __forceinline__ const char * src(const A & a, int s) {
    return (const char *) a.src[s];
}

__device__ __forceinline__ float pf(int32_t p) {
    return __int_as_float(p);
}

__device__ __forceinline__ int32_t i32_at(const A & a, int s, int64_t i) {
    return ((const int32_t *) a.src[s])[i];
}

// Element i of row r of source s (F32, F16 or BF16).
__device__ __forceinline__ float wval(const A & a, int s, int64_t r, int64_t i) {
    const char * p = src(a, s) + r * a.src_nb[s][1];
    switch (a.src_type[s]) {
        case TYPE_F16:
            return __half2float(((const __half *) p)[i]);
        case TYPE_BF16:
            return __bfloat162float(((const __nv_bfloat16 *) p)[i]);
        default:
            return ((const float *) p)[i];
    }
}

// The cache cell of position p of sequence s through the block table in source ts.
__device__ __forceinline__ int64_t cell(const A & a, int ts, int64_t s, int64_t p, int block) {
    const int32_t * t = (const int32_t *) (src(a, ts) + s * a.src_nb[ts][1]);
    return (int64_t) t[p / block] * block + p % block;
}

// ds4_attn's window cell of position p of sequence s: through the window's table (source 7),
// a ring of params[7] entries when that is not 0 (position p in entry (p / block) % entries).
__device__ __forceinline__ int64_t window_cell(const A & a, int64_t s, int64_t p, int block) {
    const int64_t entries = a.params[7];
    return cell(a, 7, s, entries > 0 ? p % (entries * block) : p, block);
}

// n F16 values of the cache in source s: cell c, from element off.
__device__ __forceinline__ const __half * cache_row(const A & a, int s, int64_t c, int64_t off) {
    return (const __half *) (src(a, s) + c * a.src_nb[s][1]) + off;
}

// Bit tests: --use_fast_math must not fold them away.
__device__ __forceinline__ bool finite(float v) {
    return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ bool above_neg_inf(float v) {
    const uint32_t u = __float_as_uint(v);
    return u != 0xff800000u && (finite(v) || (u & 0x007fffffu) == 0);
}

// Larger floats, larger keys.
__device__ __forceinline__ uint32_t order_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u >> 31) ? ~u : (u | 0x80000000u);
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it.
__device__ float block_sum(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = blockDim.x / 32;
    __syncthreads();
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    v = l < nw ? sh[l] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// Exclusive prefix sum over the block in thread order; `total` gets the sum.
__device__ int block_scan(int v, int * sh, int & total) {
    const int l = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (l >= o) {
            x += y;
        }
    }
    __syncthreads();
    if (l == 31) {
        sh[w] = x;
    }
    __syncthreads();
    if (w == 0) {
        int s = l < nw ? sh[l] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (l >= o) {
                s += y;
            }
        }
        if (l < nw) {
            sh[l] = s;
        }
    }
    __syncthreads();
    total = sh[nw - 1];
    return (w > 0 ? sh[w - 1] : 0) + x - v;
}

// The k best candidates of n values (`val(i)`; candidates by `cand`), ties to the lower index,
// visited in index order: `emit(i, slot, picked)` for every i < n (in chunks of blockDim.x, all
// threads of a chunk after its reads), `slot` the picked one's rank in index order. Returns how
// many were picked. Shared: hist[256], sh[34].
template <typename V, typename C, typename E>
__device__ int select_best(int64_t n, int k, V val, C cand, E emit, uint32_t * hist, int * sh) {
    const int B = blockDim.x;
    int count = 0;
    for (int64_t i = threadIdx.x; i < n; i += B) {
        count += cand(val(i)) ? 1 : 0;
    }
    int total;
    block_scan(count, sh, total);
    count = total;
    const bool all = count <= k;
    // The threshold key and how many of the candidates equal to it to take (the first ones).
    uint32_t prefix = 0, mask = 0;
    int rem = k;
    if (!all) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += B) {
                hist[i] = 0;
            }
            __syncthreads();
            for (int64_t i = threadIdx.x; i < n; i += B) {
                const float v = val(i);
                if (!cand(v)) {
                    continue;
                }
                const uint32_t key = order_key(v);
                if ((key & mask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 255u], 1u);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int cum = 0, dg = 255;
                for (; dg > 0; --dg) {
                    if (cum + (int) hist[dg] >= rem) {
                        break;
                    }
                    cum += (int) hist[dg];
                }
                sh[32] = (int) (prefix | ((uint32_t) dg << shift));
                sh[33] = rem - cum;
            }
            __syncthreads();
            prefix = (uint32_t) sh[32];
            rem    = sh[33];
            mask |= 255u << shift;
            __syncthreads();
        }
    }
    int eq_seen = 0, picked = 0;
    for (int64_t c0 = 0; c0 < n; c0 += B) {
        const int64_t i = c0 + threadIdx.x;
        bool is_c = false, eq = false, gt = false;
        if (i < n) {
            const float v = val(i);
            is_c = cand(v);
            if (is_c && !all) {
                const uint32_t key = order_key(v);
                eq = key == prefix;
                gt = key > prefix;
            }
        }
        int n_eq;
        const int eq_rank = eq_seen + block_scan(eq ? 1 : 0, sh, n_eq);
        eq_seen += n_eq;
        const bool pick = is_c && (all || gt || (eq && eq_rank < rem));
        int n_pick;
        const int slot = picked + block_scan(pick ? 1 : 0, sh, n_pick);
        picked += n_pick;
        __syncthreads();
        if (i < n) {
            emit(i, slot, pick);
        }
        __syncthreads();
    }
    return picked;
}

// The CPU's `freq` step by step in IEEE f32 (the power in double, rounded once): the angle is
// position x frequency, so at the long positions a frequency an ulp off (fast-math powf is
// several, its divisions approximate) turns the rotation by hundredths of a radian.
__device__ float rope_freq(int n_rot, float base, bool yarn, float factor, int low, int high, int i) {
    const float f = __frcp_rn((float) pow((double) base, (double) (2 * i) / (double) n_rot));
    if (!yarn) {
        return f;
    }
    const float ramp =
        fminf(fmaxf(__fdiv_rn(__fsub_rn((float) i, (float) low), fmaxf((float) (high - low), 1e-3f)), 0.f), 1.f);
    const float smooth = __fsub_rn(1.f, ramp);
    return __fadd_rn(__fmul_rn(__fdiv_rn(f, factor), __fsub_rn(1.f, smooth)), __fmul_rn(f, smooth));
}

constexpr int MAX_PAIRS = 128;

// ds4_rope: a block per token; the angles once (the f32 product, then sin and cos of it in
// double: exact for the large positions fast-math sinf would miss), then every head.
__global__ void rope_kernel(const A a) {
    __shared__ float sn[MAX_PAIRS], cs[MAX_PAIRS];
    const int64_t d = a.ne[0], h = a.ne[1], t = blockIdx.x;
    const int     n_rot = a.params[0];
    const bool    inverse = a.params[6] != 0;
    const float   ps = (float) (i32_at(a, 1, t) + a.params[7]);
    for (int i = threadIdx.x; i < n_rot / 2; i += blockDim.x) {
        const float th = __fmul_rn(ps, rope_freq(n_rot, pf(a.params[1]), a.params[2] != 0, pf(a.params[3]),
                                                 a.params[4], a.params[5], i));
        double s, c;
        sincos((double) th, &s, &c);
        sn[i] = inverse ? -(float) s : (float) s;
        cs[i] = (float) c;
    }
    __syncthreads();
    const float * x = (const float *) a.src[0] + t * h * d;
    float *       o = (float *) a.dst + t * h * d;
    const int64_t pass = d - n_rot;
    for (int64_t e = threadIdx.x; e < h * d; e += blockDim.x) {
        const int64_t j = e % d;
        if (j < pass) {
            o[e] = x[e];
        } else if ((j - pass) % 2 == 0) {
            const int64_t i = (j - pass) / 2;
            const float   x0 = x[e], x1 = x[e + 1];
            o[e]     = x0 * cs[i] - x1 * sn[i];
            o[e + 1] = x0 * sn[i] + x1 * cs[i];
        }
    }
}

// ds4_collapse: a block per output row.
__global__ void collapse_kernel(const A a) {
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], j = blockIdx.x;
    const bool    has_pre = a.params[0] != 0, has_rows = a.params[1] != 0;
    const int64_t t   = has_rows ? i32_at(a, has_pre ? 2 : 1, j) : j;
    const char *  res = src(a, 0) + t * a.src_nb[0][2];
    float *       o   = (float *) ((char *) a.dst + j * a.nb[1]);
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        if (!has_pre) {
            o[i] = ((const float *) res)[i];
            continue;
        }
        const float * w   = (const float *) (src(a, 1) + t * a.src_nb[1][1]);
        float         acc = 0.f;
        for (int64_t s = 0; s < hc; ++s) {
            acc += w[s] * ((const float *) (res + s * a.src_nb[0][1]))[i];
        }
        o[i] = acc;
    }
}

// ds4_engram: a block per (stream, token).
__global__ void engram_kernel(const A a) {
    __shared__ float sh[32];
    const int64_t e = a.ne[0], hc = a.ne[1], h = blockIdx.x, t = blockIdx.y;
    const float   eps = pf(a.params[0]);
    const float * x   = (const float *) (src(a, 0) + t * a.src_nb[0][2] + h * a.src_nb[0][1]);
    const float * kv  = (const float *) (src(a, 1) + t * a.src_nb[1][1]);
    const float * key = kv + h * e;
    float         sx = 0.f, sk = 0.f, dot = 0.f;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        sx += x[i] * x[i];
        sk += key[i] * key[i];
        dot += x[i] * wval(a, 2, h, i) * wval(a, 3, h, i) * key[i];
    }
    sx  = block_sum(sx, sh);
    sk  = block_sum(sk, sh);
    dot = block_sum(dot, sh);
    const float rstd = rsqrtf(sx / (float) e + eps) * rsqrtf(sk / (float) e + eps);
    const float s    = dot * rstd * rsqrtf((float) e);
    const float g    = copysignf(sqrtf(fmaxf(fabsf(s), 1e-6f)), s);
    const float gate = 1.f / (1.f + expf(-g));
    const float * value = kv + hc * e;
    float *       o     = (float *) a.dst + (t * hc + h) * e;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        o[i] = x[i] + gate * value[i];
    }
}

// ds4_cache_write: a block per token.
__global__ void cache_write_kernel(const A a) {
    const int64_t t = blockIdx.x, w = a.src_ne[1][0];
    const int     off = a.params[0], ratio = a.params[1] > 1 ? a.params[1] : 1;
    if ((i32_at(a, 3, t) + 1) % ratio != 0) {
        return;
    }
    const float * x = (const float *) (src(a, 1) + t * a.src_nb[1][1]);
    __half *      c = (__half *) ((char *) a.src[0] + (int64_t) i32_at(a, 2, t) * a.src_nb[0][1]) + off;
    for (int64_t i = threadIdx.x; i < w; i += blockDim.x) {
        c[i] = __float2half(x[i]);
    }
}

// ds4_pool: a block per token, a thread per channel.
__global__ void pool_kernel(const A a) {
    const int64_t d = a.ne[0], t = blockIdx.x;
    const int     n_tok = a.params[0], ratio = a.params[1], block = a.params[2];
    float *       o  = (float *) ((char *) a.dst + t * a.nb[1]);
    const int64_t pt = i32_at(a, 2, t);
    const bool    end = (pt + 1) % ratio == 0;
    const int64_t s   = t / n_tok;
    for (int64_t c = threadIdx.x; c < d; c += blockDim.x) {
        if (!end) {
            o[c] = 0.f;
            continue;
        }
        float m = -INFINITY;
        for (int64_t q = pt + 1 - ratio; q <= pt; ++q) {
            m = fmaxf(m, __half2float(cache_row(a, 0, cell(a, 1, s, q, block), 0)[d + c]));
        }
        float sum = 0.f, acc = 0.f;
        for (int64_t q = pt + 1 - ratio; q <= pt; ++q) {
            const __half * r  = cache_row(a, 0, cell(a, 1, s, q, block), 0);
            const float    ex = expf(__half2float(r[d + c]) - m);
            sum += ex;
            acc += __half2float(r[c]) * ex;
        }
        o[c] = acc / sum;
    }
}

// ds4_index_scores: a block per (128 compressed positions, query), the query's heads and head
// weights staged in shared memory (dynamic: (di + 1) · ih floats).
__global__ void index_scores_kernel(const A a) {
    extern __shared__ float qs[];
    const int64_t di = a.src_ne[0][0], ih = a.src_ne[0][1];
    const int64_t nc = a.ne[0], t = blockIdx.y;
    const int     n_tok = a.params[0], ratio = a.params[1], koff = a.params[2], block = a.params[3];
    const bool    masked = a.params[4] != 0;
    float *       ws = qs + di * ih;
    for (int64_t e = threadIdx.x; e < di * ih; e += blockDim.x) {
        qs[e] = ((const float *) (src(a, 0) + t * a.src_nb[0][2] + (e / di) * a.src_nb[0][1]))[e % di];
    }
    for (int64_t h = threadIdx.x; h < ih; h += blockDim.x) {
        ws[h] = ((const float *) (src(a, 1) + t * a.src_nb[1][1]))[h];
    }
    __syncthreads();
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nc) {
        return;
    }
    const int64_t pt    = i32_at(a, 4, t);
    const int64_t n_vis = (pt + 1) / ratio < nc ? (pt + 1) / ratio : nc;
    float *       o     = (float *) ((char *) a.dst + t * a.nb[1]);
    if (j >= n_vis) {
        o[j] = -INFINITY;
        return;
    }
    const __half * key = cache_row(a, 2, cell(a, 3, t / n_tok, j * ratio + ratio - 1, block), koff);
    float          sc  = 0.f;
    for (int64_t h = 0; h < ih; ++h) {
        float dot = 0.f;
        for (int64_t i = 0; i < di; ++i) {
            dot += qs[h * di + i] * __half2float(key[i]);
        }
        sc += fmaxf(dot, 0.f) * ws[h];
    }
    if (masked) {
        sc += ((const float *) (src(a, 5) + t * a.src_nb[5][1]))[j];
    }
    o[j] = sc;
}

constexpr int IDX_KEYS = 128;  // compressed positions per block, a thread each
constexpr int IDX_TQ   = 32;   // queries per block (of one sequence)

// ds4_index_scores for DI-wide index heads: a block per (IDX_KEYS compressed positions, up to
// IDX_TQ queries of one sequence), a thread per position holding its key in registers, read once
// for all the block's queries; each query's heads and head weights staged in shared memory
// (dynamic: (DI + 1) · ih floats) in turn. Every score is index_scores_kernel's sum in its order:
// per head the dot over the key's elements in order, relu, times the head weight, the heads in
// order, then the mask.
template <int DI>
__global__ void __launch_bounds__(IDX_KEYS) index_scores_keys_kernel(const A a) {
    extern __shared__ float qs[];
    __shared__ int64_t vis_max;
    const int64_t ih = a.src_ne[0][1], nc = a.ne[0];
    const int     n_tok = a.params[0], ratio = a.params[1], koff = a.params[2], block = a.params[3];
    const bool    masked = a.params[4] != 0;
    const int64_t tiles = (n_tok + IDX_TQ - 1) / IDX_TQ;
    const int64_t s = blockIdx.y / tiles, t0 = s * n_tok + (blockIdx.y % tiles) * IDX_TQ;
    const int64_t t1 = t0 + IDX_TQ < (s + 1) * n_tok ? t0 + IDX_TQ : (s + 1) * n_tok;
    const int64_t j0 = (int64_t) blockIdx.x * IDX_KEYS, j = j0 + threadIdx.x;
    float *       ws = qs + DI * ih;
    if (threadIdx.x == 0) {
        int64_t v = 0;
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t n = (i32_at(a, 4, t) + 1) / ratio;
            v = n > v ? n : v;
        }
        vis_max = v < nc ? v : nc;
    }
    __syncthreads();
    if (j0 >= vis_max) {
        // No query here sees these positions.
        if (j < nc) {
            for (int64_t t = t0; t < t1; ++t) {
                ((float *) ((char *) a.dst + t * a.nb[1]))[j] = -INFINITY;
            }
        }
        return;
    }
    float k[DI];
    if (j < vis_max) {
        const __half * key = cache_row(a, 2, cell(a, 3, s, j * ratio + ratio - 1, block), koff);
        if ((reinterpret_cast<uintptr_t>(key) & 15) == 0) {
#pragma unroll
            for (int i = 0; i < DI; i += 8) {
                const uint4    u = *reinterpret_cast<const uint4 *>(key + i);
                const __half * hv = reinterpret_cast<const __half *>(&u);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    k[i + e] = __half2float(hv[e]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < DI; ++i) {
                k[i] = __half2float(key[i]);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < DI; ++i) {
            k[i] = 0.f;
        }
    }
    for (int64_t t = t0; t < t1; ++t) {
        __syncthreads();
        const float * q = (const float *) (src(a, 0) + t * a.src_nb[0][2]);
        for (int64_t e = threadIdx.x; e < DI * ih; e += blockDim.x) {
            qs[e] = q[e];
        }
        for (int64_t h = threadIdx.x; h < ih; h += blockDim.x) {
            ws[h] = ((const float *) (src(a, 1) + t * a.src_nb[1][1]))[h];
        }
        __syncthreads();
        if (j >= nc) {
            continue;
        }
        const int64_t pt    = i32_at(a, 4, t);
        const int64_t n_vis = (pt + 1) / ratio < nc ? (pt + 1) / ratio : nc;
        float *       o     = (float *) ((char *) a.dst + t * a.nb[1]);
        if (j >= n_vis) {
            o[j] = -INFINITY;
            continue;
        }
        float sc = 0.f;
        for (int64_t h = 0; h < ih; ++h) {
            const float4 * q4  = reinterpret_cast<const float4 *>(qs + h * DI);
            float          dot = 0.f;
#pragma unroll
            for (int i = 0; i < DI / 4; ++i) {
                const float4 qv = q4[i];
                dot += qv.x * k[4 * i];
                dot += qv.y * k[4 * i + 1];
                dot += qv.z * k[4 * i + 2];
                dot += qv.w * k[4 * i + 3];
            }
            sc += fmaxf(dot, 0.f) * ws[h];
        }
        if (masked) {
            sc += ((const float *) (src(a, 5) + t * a.src_nb[5][1]))[j];
        }
        o[j] = sc;
    }
}

// ds4_cand_mask: a block per query. The block maxima go into the output row first (fewer
// than its positions), then the kept flags in place, then the mask spread from the end.
__global__ void cand_mask_kernel(const A a) {
    __shared__ uint32_t hist[256];
    __shared__ int      sh[34];
    const int64_t nc = a.ne[0], t = blockIdx.x;
    const int     ratio = a.params[0], bs = a.params[1], tb = a.params[2];
    const int64_t nb = (nc + bs - 1) / bs;
    const float * sc  = (const float *) (src(a, 0) + t * a.src_nb[0][1]);
    float *       row = (float *) ((char *) a.dst + t * a.nb[1]);
    for (int64_t b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        const int64_t end = (b + 1) * bs < nc ? (b + 1) * bs : nc;
        for (int64_t j = b * bs; j < end; ++j) {
            m = fmaxf(m, sc[j]);
        }
        row[b] = m;
    }
    __syncthreads();
    const int64_t n_vis = (i32_at(a, 1, t) + 1) / ratio;
    if (threadIdx.x == 0 && n_vis > 0 && (n_vis - 1) / bs < nb) {
        row[(n_vis - 1) / bs] = INFINITY;
    }
    __syncthreads();
    select_best(
        nb, tb, [&](int64_t b) { return row[b]; }, [](float v) { return above_neg_inf(v); },
        [&](int64_t b, int, bool pick) { row[b] = pick ? 1.f : 0.f; }, hist, sh);
    __syncthreads();
    const int64_t B  = blockDim.x;
    const int64_t c0 = ((nc - 1) / B) * B;
    for (int64_t base = c0; base >= 0; base -= B) {
        const int64_t j = base + threadIdx.x;
        const float   f = j < nc ? row[j / bs] : 0.f;
        __syncthreads();
        if (j < nc) {
            row[j] = f == 1.f ? 0.f : -INFINITY;
        }
        __syncthreads();
    }
}

// ds4_topk: a block per query; the picks in ascending order, -1 after them.
__global__ void topk_kernel(const A a) {
    __shared__ uint32_t hist[256];
    __shared__ int      sh[34];
    const int64_t nc = a.src_ne[0][0], k = a.ne[0], t = blockIdx.x;
    const float * sc  = (const float *) (src(a, 0) + t * a.src_nb[0][1]);
    int32_t *     out = (int32_t *) ((char *) a.dst + t * a.nb[1]);
    const int     n   = select_best(
        nc, (int) k, [&](int64_t j) { return sc[j]; }, [](float v) { return finite(v); },
        [&](int64_t j, int slot, bool pick) {
            if (pick) {
                out[slot] = (int32_t) j;
            }
        },
        hist, sh);
    for (int64_t i = n + threadIdx.x; i < k; i += blockDim.x) {
        out[i] = -1;
    }
}

constexpr int ATTN_THREADS = 128;
constexpr int ATTN_PER     = 8;  // d up to 1024
constexpr int ATTN_KEYS    = 1024;  // attn_tiled_kernel: window + picks up to this

// ds4_attn: a block per (head, query), online softmax over the window's latents, then the
// selected compressed ones, the sink in the denominator.
__global__ void attn_kernel(const A a) {
    __shared__ float sh[32];
    const int64_t d = a.ne[0], nh = a.ne[1], h = blockIdx.x, t = blockIdx.y;
    const int     n_tok = a.params[0], window = a.params[2], ratio = a.params[3], coff = a.params[4];
    const int     block = a.params[5];
    const bool    has_comp = a.params[6] != 0;
    const float   scale = pf(a.params[1]);
    const int64_t s = t / n_tok, pt = i32_at(a, 3, t);
    const float * q = (const float *) (src(a, 0) + t * a.src_nb[0][2] + h * a.src_nb[0][1]);
    float         qv[ATTN_PER], acc[ATTN_PER];
    for (int r = 0; r < ATTN_PER; ++r) {
        const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
        qv[r]  = i < d ? q[i] : 0.f;
        acc[r] = 0.f;
    }
    float      m = -INFINITY, l = 0.f;
    const auto visit = [&](const __half * k) {
        float part = 0.f;
        for (int r = 0; r < ATTN_PER; ++r) {
            const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
            if (i < d) {
                part += qv[r] * __half2float(k[i]);
            }
        }
        const float sc   = block_sum(part, sh) * scale;
        const float mn   = fmaxf(m, sc);
        const float corr = expf(m - mn), p = expf(sc - mn);
        l = l * corr + p;
        for (int r = 0; r < ATTN_PER; ++r) {
            const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
            acc[r]          = acc[r] * corr + (i < d ? p * __half2float(k[i]) : 0.f);
        }
        m = mn;
    };
    const int64_t lo = pt >= window - 1 ? pt - (window - 1) : 0;
    for (int64_t w = lo; w <= pt; ++w) {
        visit(cache_row(a, 1, window_cell(a, s, w, block), 0));
    }
    if (has_comp) {
        const int64_t   k   = a.src_ne[6][0];
        const int32_t * sel = (const int32_t *) (src(a, 6) + t * a.src_nb[6][1]);
        for (int64_t i = 0; i < k; ++i) {
            const int32_t j = sel[i];
            if (j >= 0) {
                visit(cache_row(a, 5, cell(a, 2, s, (int64_t) j * ratio + ratio - 1, block), coff));
            }
        }
    }
    l += expf(((const float *) a.src[4])[h] - m);
    float * o = (float *) a.dst + (t * nh + h) * d;
    for (int r = 0; r < ATTN_PER; ++r) {
        const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
        if (i < d) {
            o[i] = acc[r] / l;
        }
    }
}

// 16 bytes (n >= 8 halves, both ends 16-byte aligned) from global to shared memory without
// waiting where the GPU has cp.async; else (and for a row's tail) a plain copy of min(n, 8).
__device__ __forceinline__ void copy16(__half * sh, const __half * g, int n) {
    const uintptr_t ends  = reinterpret_cast<uintptr_t>(sh) | reinterpret_cast<uintptr_t>(g);
    const bool      whole = n >= 8 && (ends & 15) == 0;
#if __CUDA_ARCH__ >= 800
    if (whole) {
        const uint32_t a = (uint32_t) __cvta_generic_to_shared(sh);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
        return;
    }
#else
    if (whole) {
        *reinterpret_cast<uint4 *>(sh) = *reinterpret_cast<const uint4 *>(g);
        return;
    }
#endif
    for (int i = 0; i < n && i < 8; ++i) {
        sh[i] = g[i];
    }
}

__device__ __forceinline__ void copy16_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Waits until at most N of the committed groups are in flight.
template <int N> __device__ __forceinline__ void copy16_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// ds4_attn with HB heads of one query per block and KT keys per round: attn_kernel's arithmetic
// for every (head, query), so the same output bits. Each thread still sums its elements
// tid + r · ATTN_THREADS of the dot in order, each warp's sum is the same xor butterfly, the four
// warps' sums add as block_sum adds them ((w0 + w2) + (w1 + w3), each first plus 0 as the
// butterfly does) and the online softmax takes the keys one at a time in the same order. What
// changes: a key's elements are read once for the block's HB heads; the keys stream through
// shared memory SK at a time (cp.async, ST - 1 rounds ahead); the KT · HB dots of a round are
// reduced together (a reduce-scatter, one shuffle per kept dot) between one pair of barriers; and
// one thread per head runs the softmax's max, corrections and weights, which the block then
// applies. PER: values per thread (d up to PER · ATTN_THREADS).
template <int HB, int KT, int PER, int SK>
__global__ void __launch_bounds__(ATTN_THREADS) attn_tiled_kernel(const A a) {
    constexpr int NV    = KT * HB;  // dots per round
    constexpr int STEPS = NV >= 32 ? 5 : NV >= 16 ? 4 : NV >= 8 ? 3 : NV >= 4 ? 2 : NV >= 2 ? 1 : 0;
    static_assert(NV <= 32 && (NV & (NV - 1)) == 0, "KT * HB: a power of two up to 32");
    __shared__ float red[NV * 4];
    __shared__ float cp[KT][HB][2];  // a key's (correction, weight) per head
    __shared__ float lf[HB];
    __shared__ const __half * rows[ATTN_KEYS];
    const int64_t d = a.ne[0], nh = a.ne[1], h0 = (int64_t) blockIdx.x * HB, t = blockIdx.y;
    const int     n_tok = a.params[0], window = a.params[2], ratio = a.params[3], coff = a.params[4];
    const int     block = a.params[5];
    const bool    has_comp = a.params[6] != 0;
    const float   scale = pf(a.params[1]);
    const int64_t s = t / n_tok, pt = i32_at(a, 3, t);
    const int     lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float         qv[HB][PER], acc[HB][PER];
    float         m = -INFINITY, l = 0.f;  // thread hb < HB: head hb's running max and sum
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
        const float * q = (const float *) (src(a, 0) + t * a.src_nb[0][2] + (h0 + hb) * a.src_nb[0][1]);
#pragma unroll
        for (int r = 0; r < PER; ++r) {
            const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
            qv[hb][r]       = i < d ? q[i] : 0.f;
            acc[hb][r]      = 0.f;
        }
    }
    const int64_t   lo    = pt >= window - 1 ? pt - (window - 1) : 0;
    const int64_t   n_win = pt - lo + 1;
    const int64_t   n_sel = has_comp ? a.src_ne[6][0] : 0;
    const int64_t   n_key = n_win + n_sel;  // up to ATTN_KEYS (the launch checks)
    const int32_t * sel   = (const int32_t *) (src(a, 6) + t * a.src_nb[6][1]);
    // Every key's row up front, null for a skipped pick: the cells and the picks are looked up
    // in parallel instead of a chain of dependent loads per round.
    for (int64_t kk = threadIdx.x; kk < n_key; kk += ATTN_THREADS) {
        const __half * row = nullptr;
        if (kk < n_win) {
            row = cache_row(a, 1, window_cell(a, s, lo + kk, block), 0);
        } else {
            const int32_t j = sel[kk - n_win];
            if (j >= 0) {
                row = cache_row(a, 5, cell(a, 2, s, (int64_t) j * ratio + ratio - 1, block), coff);
            }
        }
        rows[kk] = row;
    }
    __syncthreads();
    // The keys stream through shared memory, ST - 1 rounds ahead of the sums.
    constexpr int ST = SK / KT;
    static_assert(ST >= 2 && ST * KT == SK, "SK: at least two rounds of keys");
    __shared__ __align__(16) __half kbuf[ST][KT][PER * ATTN_THREADS];
    const int64_t n_round = (n_key + KT - 1) / KT;
    const int     chunks  = (int) ((d + 7) / 8);  // 16-byte pieces of a row
    const auto    issue   = [&](int64_t rd) {
        if (rd < n_round) {
            for (int c = threadIdx.x; c < KT * chunks; c += ATTN_THREADS) {
                const int      kt  = c / chunks, e = (c - kt * chunks) * 8;
                const int64_t  kk  = rd * KT + kt;
                const __half * row = kk < n_key ? rows[kk] : nullptr;
                if (row != nullptr) {
                    copy16(kbuf[rd % ST][kt] + e, row + e, (int) (d - e));
                }
            }
        }
        copy16_commit();  // empty past the end: every round waits on the same count
    };
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {
        issue(st);
    }
    for (int64_t rd = 0; rd < n_round; ++rd) {
        copy16_wait<ST - 2>();
        __syncthreads();
        issue(rd + ST - 1);  // into the buffer of round rd - 1, read by now
        const int64_t k0 = rd * KT;
        float         kv[KT][PER];
        bool          ok[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            ok[kt] = k0 + kt < n_key && rows[k0 + kt] != nullptr;
#pragma unroll
            for (int r = 0; r < PER; ++r) {
                const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
                kv[kt][r]       = ok[kt] && i < d ? __half2float(kbuf[rd % ST][kt][i]) : 0.f;
            }
        }
        float v[NV];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int hb = 0; hb < HB; ++hb) {
                float part = 0.f;
#pragma unroll
                for (int r = 0; r < PER; ++r) {
                    const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
                    if (i < d) {
                        part += qv[hb][r] * kv[kt][r];
                    }
                }
                v[kt * HB + hb] = part;
            }
        }
        // The butterfly's first STEPS rounds as a reduce-scatter: at the round of offset o the
        // lanes with bit o keep the upper half of their dots and pass the lower half, so each
        // sum is still own + partner's at every offset 16, 8, ... (one shuffle per kept dot
        // instead of one per dot). Then lane `lane >> (5 - STEPS)` holds dot number that.
#pragma unroll
        for (int st = 0; st < STEPS; ++st) {
            const int  n  = NV >> (st + 1);
            const int  o  = 16 >> st;
            const bool up = (lane & o) != 0;
#pragma unroll
            for (int j = 0; j < n; ++j) {
                const float keep = up ? v[j + n] : v[j];
                const float send = up ? v[j] : v[j + n];
                v[j]             = keep + __shfl_xor_sync(0xffffffffu, send, o);
            }
        }
#pragma unroll
        for (int o = 16 >> STEPS; o > 0; o >>= 1) {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
        }
        if ((lane & ((32 >> STEPS) - 1)) == 0) {
            red[(lane >> (5 - STEPS)) * 4 + warp] = v[0];
        }
        __syncthreads();
        // A thread per head: the four warps' sums as block_sum adds them ((w0 + w2) + (w1 + w3),
        // each first plus 0 as its butterfly does), then the online softmax's steps key by key.
        if (threadIdx.x < HB) {
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                if (!ok[kt]) {
                    continue;
                }
                const float * w    = red + (kt * HB + threadIdx.x) * 4;
                const float   tot  = __fadd_rn(__fadd_rn(__fadd_rn(w[0], 0.f), __fadd_rn(w[2], 0.f)),
                                               __fadd_rn(__fadd_rn(w[1], 0.f), __fadd_rn(w[3], 0.f)));
                const float   sc   = tot * scale;
                const float   mn   = fmaxf(m, sc);
                const float   corr = expf(m - mn), p = expf(sc - mn);
                l                  = l * corr + p;
                m                  = mn;
                cp[kt][threadIdx.x][0] = corr;
                cp[kt][threadIdx.x][1] = p;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            if (!ok[kt]) {
                continue;
            }
#pragma unroll
            for (int hb = 0; hb < HB; ++hb) {
                const float corr = cp[kt][hb][0], p = cp[kt][hb][1];
#pragma unroll
                for (int r = 0; r < PER; ++r) {
                    const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
                    acc[hb][r]      = acc[hb][r] * corr + (i < d ? p * kv[kt][r] : 0.f);
                }
            }
        }
    }
    if (threadIdx.x < HB) {
        lf[threadIdx.x] = l + expf(((const float *) a.src[4])[h0 + threadIdx.x] - m);
    }
    __syncthreads();
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
        float * o = (float *) a.dst + (t * nh + h0 + hb) * d;
#pragma unroll
        for (int r = 0; r < PER; ++r) {
            const int64_t i = threadIdx.x + (int64_t) r * ATTN_THREADS;
            if (i < d) {
                o[i] = acc[hb][r] / lf[hb];
            }
        }
    }
}

// The dot of one key with HB heads' queries in one warp, attn_kernel's sum to the bit. Lane l
// holds attn_kernel's threads 4l + c (c < 4): v[h * 4 + c] is that thread's partial for head h
// (its elements 4l + c + 128 r in order of r). The block_sum tree is a butterfly over the bits
// of the thread id, in the order 4, 3, 2 (lane bits 2, 1, 0 here), 1, 0 (c's bits), then the
// warps ((w0 + w2) + (w1 + w3), each first plus 0; lane bits 4, 3). Each step adds the same
// pairs as there (fp addition commutes exactly); the lane steps are a reduce-scatter (each lane
// keeps half of the values, one shuffle per kept value) while a lane holds more than one value.
// Returns the sum in lanes l < 8 with l % (8 / HB) == 0, for head l / (8 / HB).
template <int HB> __device__ __forceinline__ float warp_dot(float (&v)[HB * 4], int lane) {
    constexpr int N = HB * 4;
    // Tree bits 4, 3, 2 at lane bits 2, 1, 0. A reduce-scatter step moves the top index bit
    // of v onto the lane bit (the upper half to the lanes with the bit set).
    int n = N;
#pragma unroll
    for (int st = 0; st < 3; ++st) {
        const int o = 4 >> st;
        if (n > 1) {
            const bool up = (lane & o) != 0;
            n >>= 1;
#pragma unroll
            for (int j = 0; j < N / 2; ++j) {
                if (j < n) {
                    const float keep = up ? v[j + n] : v[j];
                    const float send = up ? v[j] : v[j + n];
                    v[j]             = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, o));
                }
            }
        } else {
            v[0] = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], o));
        }
    }
    // Tree bits 1 and 0 (c's bits 1 and 0): still in v (HB 8), or moved onto lane bits.
    if constexpr (HB == 8) {
        v[0] = __fadd_rn(v[0], v[2]);
        v[1] = __fadd_rn(v[1], v[3]);
        v[0] = __fadd_rn(v[0], v[1]);
    } else if constexpr (HB == 4) {
        // c bit 1 on lane bit 0, v[0..2) over c bit 0: a reduce-scatter onto lane bit 0.
        const bool  up   = (lane & 1) != 0;
        const float keep = up ? v[1] : v[0];
        const float send = up ? v[0] : v[1];
        v[0]             = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, 1));
        v[0]             = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], 1));
    } else if constexpr (HB == 2) {
        // c bits 1, 0 on lane bits 1, 0.
        v[0] = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], 2));
        v[0] = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], 1));
    } else {
        // c bits 1, 0 on lane bits 2, 1.
        static_assert(HB == 1, "HB: 1, 2, 4 or 8");
        v[0] = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], 4));
        v[0] = __fadd_rn(v[0], __shfl_xor_sync(0xffffffffu, v[0], 2));
    }
    // The warps: lane bits 4 (w0 + w2, w1 + w3), then 3.
    float x = __fadd_rn(v[0], 0.f);
    x       = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 16));
    return __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 8));
}

// 4 F16 values from global memory at p (8-byte aligned), as floats.
__device__ __forceinline__ void load4(float (&k)[4], const __half * p) {
    const uint2  u  = __ldg(reinterpret_cast<const uint2 *>(p));
    const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
    k[0]            = lo.x;
    k[1]            = lo.y;
    k[2]            = hi.x;
    k[3]            = hi.y;
}

constexpr int SCORED_D    = 512;   // attn_scored_kernel's head width
constexpr int SCORED_KEYS = 1024;  // keys of all HB heads per chunk of attn_scored_kernel

// ds4_attn for few queries (decode): HB heads of one query per block, NW warps, attn_kernel's
// arithmetic for every (head, query), so the same output bits. Instead of a key at a time (the
// tiled kernels: a block-wide dot, the softmax step, the sums, two barriers per round), per chunk
// of SCORED_KEYS / HB keys:
//  1. scores: a warp per key (all HB heads, warp_dot), the warps over the chunk's keys, reading
//     UA keys at a time straight from global memory, all their loads issued before the first is
//     used (no branch between them: a key past the chunk or a skipped pick loads key 0, a window
//     key, and is not used);
//  2. the online softmax's steps: its running max is a prefix max over the keys (a warp scan per
//     head; max is exact in any order), so every key's correction expf(m - mn) and weight
//     expf(s - mn) come out in parallel from the same operands as attn_kernel's serial steps;
//  3. the sums: each thread its elements of every head, key by key in order, l and acc by the
//     same fma steps as attn_kernel (acc = fma(corr, acc, p · k), l = fma(l, corr, p)), the keys'
//     elements loaded U keys ahead.
// The launch checks d == SCORED_D, nh % HB == 0, at most ATTN_KEYS keys and 8-byte aligned rows.
template <int HB, int NW, int UA, int U>
__global__ void __launch_bounds__(NW * 32) attn_scored_kernel(const A a) {
    constexpr int NT = NW * 32;
    constexpr int CH = SCORED_KEYS / HB;  // keys per chunk
    constexpr int E  = SCORED_D / NT;     // elements per thread in the sums
    static_assert((E == 2 || E == 4) && E * NT == SCORED_D && CH % 32 == 0, "NW: 4 or 8");
    __shared__ const __half * rows[ATTN_KEYS];
    __shared__ float          sc[HB][CH];
    __shared__ float2         cw[HB][CH];  // (correction, weight); correction -1: a skipped pick
    __shared__ float          mx[HB];      // running max per head
    const int64_t nh = a.ne[1], h0 = (int64_t) blockIdx.x * HB, t = blockIdx.y;
    const int     n_tok = a.params[0], window = a.params[2], ratio = a.params[3], coff = a.params[4];
    const int     block = a.params[5];
    const bool    has_comp = a.params[6] != 0;
    const float   scale = pf(a.params[1]);
    const int64_t s = t / n_tok, pt = i32_at(a, 3, t);
    const int     lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t   lo    = pt >= window - 1 ? pt - (window - 1) : 0;
    const int64_t   n_win = pt - lo + 1;  // at least 1: rows[0] is never null
    const int64_t   n_sel = has_comp ? a.src_ne[6][0] : 0;
    const int       n_key = (int) (n_win + n_sel);
    const int32_t * sel   = (const int32_t *) (src(a, 6) + t * a.src_nb[6][1]);
    for (int kk = threadIdx.x; kk < n_key; kk += NT) {
        const __half * row = nullptr;
        if (kk < n_win) {
            row = cache_row(a, 1, window_cell(a, s, lo + kk, block), 0);
        } else {
            const int32_t j = sel[kk - n_win];
            if (j >= 0) {
                row = cache_row(a, 5, cell(a, 2, s, (int64_t) j * ratio + ratio - 1, block), coff);
            }
        }
        rows[kk] = row;
    }
    if (threadIdx.x < HB) {
        mx[threadIdx.x] = -INFINITY;
    }
    float acc[HB][E], l[HB];
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
        l[hb] = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            acc[hb][e] = 0.f;
        }
    }
    const int e0 = threadIdx.x * E;  // this thread's elements in the sums
    __syncthreads();
    for (int c0 = 0; c0 < n_key; c0 += CH) {
        const int c1 = c0 + CH < n_key ? c0 + CH : n_key;
        // 1. Scores. The queries in attn_kernel's layout: q[hb][r][c] = element 4 lane + c + 128 r.
        {
            float q[HB][4][4];
#pragma unroll
            for (int hb = 0; hb < HB; ++hb) {
                const float * qp = (const float *) (src(a, 0) + t * a.src_nb[0][2] + (h0 + hb) * a.src_nb[0][1]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        q[hb][r][c] = qp[4 * lane + c + 128 * r];
                    }
                }
            }
            for (int kb = c0 + warp * UA; kb < c1; kb += NW * UA) {
                float k[UA][4][4];
#pragma unroll
                for (int u = 0; u < UA; ++u) {
                    const __half * row = rows[kb + u < c1 ? kb + u : c1 - 1];
                    row                = row != nullptr ? row : rows[0];
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        load4(k[u][r], row + 4 * lane + 128 * r);
                    }
                }
#pragma unroll
                for (int u = 0; u < UA; ++u) {
                    const int kk = kb + u;
                    if (kk >= c1 || rows[kk] == nullptr) {
                        continue;
                    }
                    float v[HB * 4];
#pragma unroll
                    for (int hb = 0; hb < HB; ++hb) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            float part = 0.f;
#pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                part = __fmaf_rn(q[hb][r][c], k[u][r][c], part);
                            }
                            v[hb * 4 + c] = part;
                        }
                    }
                    const float tot = warp_dot<HB>(v, lane);
                    if (lane < 8 && lane % (8 / HB) == 0) {
                        sc[lane / (8 / HB)][kk - c0] = __fmul_rn(tot, scale);
                    }
                }
            }
        }
        __syncthreads();
        // 2. The softmax steps: a warp per head, CH / 32 consecutive keys per lane.
        for (int hb = warp; hb < HB; hb += NW) {
            constexpr int PL  = CH / 32;
            const int     k0  = c0 + lane * PL;
            float         own = -INFINITY;  // the max of this lane's keys
#pragma unroll 4
            for (int j = 0; j < PL; ++j) {
                if (k0 + j < c1 && rows[k0 + j] != nullptr) {
                    own = fmaxf(own, sc[hb][k0 + j - c0]);
                }
            }
            float inc = own;  // the max over lanes 0 ..= lane
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) {
                    inc = fmaxf(inc, y);
                }
            }
            const float before = __shfl_up_sync(0xffffffffu, inc, 1);
            float       m      = mx[hb];
            if (lane > 0) {
                m = fmaxf(m, before);
            }
#pragma unroll 4
            for (int j = 0; j < PL; ++j) {
                const int kk = k0 + j;
                if (kk >= c1) {
                    break;
                }
                if (rows[kk] == nullptr) {
                    cw[hb][kk - c0] = make_float2(-1.f, 0.f);
                    continue;
                }
                const float x    = sc[hb][kk - c0];
                const float mn   = fmaxf(m, x);
                const float corr = expf(m - mn), p = expf(x - mn);
                m                = mn;
                cw[hb][kk - c0]  = make_float2(corr, p);
            }
            __syncwarp();
            if (lane == 31) {
                mx[hb] = m;
            }
        }
        __syncthreads();
        // 3. The sums, key by key; the keys' elements loaded U keys ahead (as F16 pairs).
        for (int kb = c0; kb < c1; kb += U) {
            __half2 kh[U][E / 2];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const __half * row = rows[kb + u < c1 ? kb + u : c1 - 1];
                row                = row != nullptr ? row : rows[0];
                if constexpr (E == 4) {
                    const uint2 w = __ldg(reinterpret_cast<const uint2 *>(row + e0));
                    kh[u][0]      = *reinterpret_cast<const __half2 *>(&w.x);
                    kh[u][1]      = *reinterpret_cast<const __half2 *>(&w.y);
                } else {
                    kh[u][0] = __ldg(reinterpret_cast<const __half2 *>(row + e0));
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (kb + u >= c1) {
                    break;
                }
                float kv[E];
#pragma unroll
                for (int e = 0; e < E; e += 2) {
                    const float2 f = __half22float2(kh[u][e / 2]);
                    kv[e]          = f.x;
                    kv[e + 1]      = f.y;
                }
#pragma unroll
                for (int hb = 0; hb < HB; ++hb) {
                    const float2 w = cw[hb][kb + u - c0];
                    if (w.x < 0.f) {
                        break;  // a skipped pick (for every head)
                    }
                    l[hb] = __fmaf_rn(l[hb], w.x, w.y);
#pragma unroll
                    for (int e = 0; e < E; ++e) {
                        acc[hb][e] = __fmaf_rn(w.x, acc[hb][e], __fmul_rn(w.y, kv[e]));
                    }
                }
            }
        }
        // No barrier before the next chunk: its scores write sc, which the softmax steps
        // finished reading before the last barrier, and its cw follows the barrier after them.
    }
#pragma unroll
    for (int hb = 0; hb < HB; ++hb) {
        const float lf = l[hb] + expf(((const float *) a.src[4])[h0 + hb] - mx[hb]);
        float *     o  = (float *) a.dst + (t * nh + h0 + hb) * SCORED_D;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            o[e0 + e] = acc[hb][e] / lf;
        }
    }
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    return sqrtf(x > 20.f ? x : log1pf(expf(x)));
}

// ds4_router_ids: a warp per token, k rounds of a warp argmax (ties to the lower id).
__global__ void router_ids_kernel(const A a) {
    const int64_t n_exp = a.src_ne[0][0], k = a.ne[0], t = blockIdx.x;
    const float * lg  = (const float *) (src(a, 0) + t * a.src_nb[0][1]);
    int32_t *     out = (int32_t *) ((char *) a.dst + t * a.nb[1]);
    const int     lane = threadIdx.x;
    for (int64_t r = 0; r < k; ++r) {
        float   best = -INFINITY;
        int32_t id   = INT32_MAX;
        for (int64_t e = lane; e < n_exp; e += 32) {
            bool taken = false;
            for (int64_t q = 0; q < r; ++q) {
                taken = taken || out[q] == (int32_t) e;
            }
            if (taken) {
                continue;
            }
            const float v = sqrt_softplus(lg[e]) + wval(a, 1, 0, e);
            if (v > best || id == INT32_MAX) {
                best = v;
                id   = (int32_t) e;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float   ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (oi != INT32_MAX && (id == INT32_MAX || ob > best || (ob == best && oi < id))) {
                best = ob;
                id   = oi;
            }
        }
        if (lane == 0) {
            out[r] = id;
        }
        __syncwarp();
    }
}

// ds4_router_w: a thread per token.
__global__ void router_w_kernel(const A a) {
    const int64_t k = a.ne[0], n = a.ne[1];
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const bool      norm = a.params[0] != 0;
    const float     scale = pf(a.params[1]);
    const float *   lg  = (const float *) (src(a, 0) + t * a.src_nb[0][1]);
    const int32_t * ids = (const int32_t *) (src(a, 1) + t * a.src_nb[1][1]);
    float *         o   = (float *) ((char *) a.dst + t * a.nb[1]);
    float           sum = 0.f;
    for (int64_t i = 0; i < k; ++i) {
        o[i] = sqrt_softplus(lg[ids[i]]);
        sum += o[i];
    }
    const float div = norm && k > 1 ? sum + 1e-20f : 1.f;
    for (int64_t i = 0; i < k; ++i) {
        o[i] = o[i] / div * scale;
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of ops/cuda.rs's `Kernel`. The kernels sit in an anonymous namespace: with
// relocatable device code the device link left extern "C" kernels out of the linked image.
extern "C" const void * const legion_ds4_kernels[15] = {
    reinterpret_cast<const void *>(&rope_kernel),
    reinterpret_cast<const void *>(&collapse_kernel),
    reinterpret_cast<const void *>(&engram_kernel),
    reinterpret_cast<const void *>(&cache_write_kernel),
    reinterpret_cast<const void *>(&pool_kernel),
    reinterpret_cast<const void *>(&index_scores_kernel),
    reinterpret_cast<const void *>(&cand_mask_kernel),
    reinterpret_cast<const void *>(&topk_kernel),
    reinterpret_cast<const void *>(&attn_kernel),
    reinterpret_cast<const void *>(&router_ids_kernel),
    reinterpret_cast<const void *>(&router_w_kernel),
    reinterpret_cast<const void *>(&index_scores_keys_kernel<128>),
    reinterpret_cast<const void *>(&attn_tiled_kernel<1, 16, 4, 32>),
    reinterpret_cast<const void *>(&attn_tiled_kernel<8, 4, 4, 16>),
    reinterpret_cast<const void *>(&attn_scored_kernel<2, 8, 8, 32>),
};
