// Inkling-2's own ops (src/model/inkling2/ops.rs, whose CPU references define them): the depth
// mix of the attention residuals, the loop controllers' input and output, the causal short convs
// with their kept history, the L2 clips, the attention with its relative bias and per-token
// sinks, the sigmoid gate, the SITU activation, the expert matmul of the Q8_0 / MXFP8 / plain
// stacks, the soft-capped head and an RMS norm with glm_rms_norm's sums. Kernel bodies only:
// the Rust side (src/model/inkling2/ops/cuda.rs) launches them with cudaLaunchKernelEx
// (programmatic dependent launches on compute capability 12.x, see pdl_wait), each node
// described generically (data, shape, byte strides, type) as in legion_ink_args.
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
// ink_mix: a thread's elements loaded at a time.
constexpr int MIX_U = 3;
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
// ink_attn: keys whose value loads go out ahead of their sums.
constexpr int ATTN_VU = 8;

// N halves loaded as one access.
template <int N> struct alignas(2 * N) HalfVec {
    __half h[N];
};
// ink_mm_id: a prompt block's output rows, pairs at a time and K values per pass, its threads
// and pair list length; a decode warp's rows and groups of loads ahead.
constexpr int MM_BM      = 32;
constexpr int MM_BN      = 16;
constexpr int MM_KC      = 128;
constexpr int MM_THREADS = 256;
constexpr int MM_LIST    = 1024;
constexpr int MM_VEC_ROWS = 4;
constexpr int MM_VEC_U    = 4;

// A finite stand-in for -inf (--use_fast_math must not see infinities).
constexpr float NEG = -3.0e38f;

__device__ __forceinline__ const char * src(const A & a, int s) {
    return (const char *) a.src[s];
}

// Programmatic dependent launch (#1221 r5): on compute capability 12.x (the GB10) the Rust side
// launches every kernel here with programmatic stream serialization, as glm5next.cu's kernels
// are (launch_pdl), so the GPU sets a kernel up while the one ahead of it is still running. Each
// kernel's first act is this wait, for the kernel ahead to complete and its writes to be
// visible, as a plain launch does: nothing reads or writes before it (but rms_kernel's leaf
// weights and l2_prefetch's hints), nothing triggers early, so the same bits. Below sm_90 it compiles to nothing (and the launch is plain).
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// A decode kernel's weights (and its conv history) asked into L2 before pdl_wait (#1221 r6), so
// their DRAM latency overlaps the kernel ahead. A prefetch moves no value into a register: L2 is
// where every write lands, so even a line the kernel ahead then writes reads its new value after
// the wait. The same loads, the same bits. Below sm_90 (no programmatic launches) nothing.
__device__ __forceinline__ void l2_prefetch(const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(__cvta_generic_to_global(p)));
#endif
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

// glm5next.cu's warp_sum<double> (rms_norm_kernel's block_sum), the same shuffles and adds.
__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
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

// FP8 E4M3 (FN) to float: (8 + m) 2^(e - 10), m 2^-9 when e is 0. #1221 r8: the float's bits
// put together (exponent e + 120, the mantissa's top 3 bits m) instead of ldexpf, the same
// values (exact either way, all normal floats).
__device__ __forceinline__ float e4m3(uint8_t b) {
    const uint32_t a = b & 0x7f, e = a >> 3, m = a & 7;
    float          v = e ? __uint_as_float((e + 120) << 23 | m << 20) : (float) m * 0.001953125f;
    if (a == 0x7f) {
        v = 0.f;  // NaN in E4M3FN; checkpoints have none
    }
    return b & 0x80 ? -v : v;
}

// ---- ink_mix: params ns, eps; sources: ns states [e, n], then the queries [e, nq] ----
__global__ void mix_kernel(const A a) {
    pdl_wait();
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
    // MIX_U of the thread's elements loaded at a time, then summed in order (#1221: a decode
    // step's one block had a memory latency per element in a row).
    const int64_t step = blockDim.x;
    for (int64_t i0 = threadIdx.x; i0 < e; i0 += MIX_U * step) {
        float xv[MIX_U][MIX_STATES], qv[MIX_U][MIX_QUERIES];
#pragma unroll
        for (int u = 0; u < MIX_U; u++) {
            const int64_t i = i0 + u * step;
#pragma unroll
            for (int s = 0; s < MIX_STATES; s++) {
                xv[u][s] = s < ns && i < e
                               ? ((const float *) (src(a, s) + t * a.src_nb[s][1]))[i]
                               : 0.f;
            }
#pragma unroll
            for (int j = 0; j < MIX_QUERIES; j++) {
                qv[u][j] = j < nq && i < e
                               ? ((const float *) (src(a, ns) + j * a.src_nb[ns][1]))[i]
                               : 0.f;
            }
        }
#pragma unroll
        for (int u = 0; u < MIX_U; u++) {
            if (i0 + u * step < e) {
#pragma unroll
                for (int s = 0; s < MIX_STATES; s++) {
                    if (s < ns) {
                        part[s][0] += xv[u][s] * xv[u][s];
#pragma unroll
                        for (int j = 0; j < MIX_QUERIES; j++) {
                            if (j < nq) {
                                part[s][1 + j] += xv[u][s] * qv[u][j];
                            }
                        }
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
    // A warp per state (blockDim.x / 32 >= MIX_STATES), the same tree for each sum.
    for (int s = w; s < ns; s += nw) {
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
    for (int64_t i0 = threadIdx.x; i0 < e; i0 += MIX_U * step) {
        float xv[MIX_U][MIX_STATES];
#pragma unroll
        for (int u = 0; u < MIX_U; u++) {
            const int64_t i = i0 + u * step;
#pragma unroll
            for (int s = 0; s < MIX_STATES; s++) {
                xv[u][s] = s < ns && i < e
                               ? ((const float *) (src(a, s) + t * a.src_nb[s][1]))[i]
                               : 0.f;
            }
        }
#pragma unroll
        for (int u = 0; u < MIX_U; u++) {
            const int64_t i = i0 + u * step;
            if (i < e) {
                for (int j = 0; j < nq; j++) {
                    float o = 0.f;
#pragma unroll
                    for (int s = 0; s < MIX_STATES; s++) {
                        if (s < ns) {
                            o += sc[s][j] * xv[u][s];
                        }
                    }
                    out[(j * n + t) * e + i] = o;
                }
            }
        }
    }
}

// rms_kernel's block.
constexpr int RMS_THREADS = 256;

// rms_kernel's scale from the block's sum of squares, the one expression of rms_kernel and
// kv_kernel. glm_rms_norm's 1.0f / sqrtf is one MUFU.RSQ under --use_fast_math, but only
// where ptxas folds the two: #1221 r8's rms_kernel<1> came out as a sqrt and a reciprocal (1
// ulp off). rsqrtf is that instruction always.
__device__ __forceinline__ float rms_scale(double sum, int64_t d, float eps) {
    return rsqrtf((float) (sum / d) + eps);
}

// ---- ink_rms: params eps, whether w is a weight; sources x [d, rows] (contiguous), w [d] or
// none (#1221 r6). glm5next.cu's rms_norm_kernel (glm_rms_norm) in its order: the same 256
// threads, thread i summing (double) x² over elements i, i + 256, ... in turn, then block_sum's
// tree (warp sums, then warp 0's sum of them, which every warp makes here: one barrier instead
// of three), s = 1 / sqrt(sum / d + eps), x s w. A thread's K elements are loaded at once and
// kept for the output, and w (a weight, params[1]) before pdl_wait: rms_norm_kernel has a memory
// latency for every few of a thread's elements, twice. The same sums: the same bits. ----
template <int K> __global__ void __launch_bounds__(RMS_THREADS) rms_kernel(const A a) {
    const int64_t d = a.ne[0], r = blockIdx.x;
    const float * w = (const float *) a.src[1];
    float         wv[K];
    if (w && a.params[1]) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            const int64_t i = threadIdx.x + (int64_t) k * RMS_THREADS;
            wv[k] = i < d ? w[i] : 0.f;
        }
    }
    pdl_wait();
    if (w && !a.params[1]) {
#pragma unroll
        for (int k = 0; k < K; k++) {
            const int64_t i = threadIdx.x + (int64_t) k * RMS_THREADS;
            wv[k] = i < d ? w[i] : 0.f;
        }
    }
    const float * x = (const float *) a.src[0] + r * d;
    float         xv[K];
#pragma unroll
    for (int k = 0; k < K; k++) {
        const int64_t i = threadIdx.x + (int64_t) k * RMS_THREADS;
        xv[k] = i < d ? x[i] : 0.f;
    }
    double sum = 0.0;
#pragma unroll
    for (int k = 0; k < K; k++) {
        if (threadIdx.x + (int64_t) k * RMS_THREADS < d) {
            sum += (double) xv[k] * xv[k];
        }
    }
    __shared__ double part[RMS_THREADS / 32];
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        part[warp] = sum;
    }
    __syncthreads();
    sum = warp_sum(lane < RMS_THREADS / 32 ? part[lane] : 0.0);
    const float s = rms_scale(sum, d, pf(a.params[0]));
    float *     o = (float *) a.dst + r * d;
    // params[2] (#1221 r8): the rows rounded to BF16 too, after all the F32 rows (gridDim.x of
    // them), as ggml's cast of them would be.
    __nv_bfloat16 * ob =
        a.params[2] ? (__nv_bfloat16 *) ((float *) a.dst + gridDim.x * d) + r * d : nullptr;
#pragma unroll
    for (int k = 0; k < K; k++) {
        const int64_t i = threadIdx.x + (int64_t) k * RMS_THREADS;
        if (i < d) {
            const float v = w ? xv[k] * s * wv[k] : xv[k] * s;
            o[i]          = v;
            if (ob) {
                ob[i] = __float2bfloat16_rn(v);
            }
        }
    }
}

// ---- ink_ctrl_in: params eps; sources cur, msg [e, n]; dst [2e, n] ----
__global__ void ctrl_in_kernel(const A a) {
    pdl_wait();
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
    pdl_wait();
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

// ---- ink_ctrl_gate (#1221 r5): a controller's gain and forget products and its output ----
//
// A controller's gain and forget products (F32 rows of `rank` values, glm_linear) and
// ink_ctrl_out in one kernel at small steps, the bits of the three. At up to 8 tokens
// glm_linear's F32 kernels (linear_mv_f32, _split, _xs in glm5next.cu; the same bits) sum row r
// as a warp: lane L takes the 32-value blocks L + 32 q in order of q, each as two halves, a
// 16-term FMA chain from 0 per half added to the lane's sum (from 0), then warp_sum (xor 16, 8,
// 4, 2, 1), times the scale. With the rank at most 512 every lane has one block at most and
// lanes 16-31 hold 0, so the xor-16 step gives lane L < 16 its sum plus 0, and the xor 8, 4, 2,
// 1 steps make lane 0's result a balanced tree over lanes 0-15 in bit-reversed order (lane 0,
// 8, 4, 12, 2, ...: pairs, then pairs of pairs). Here one thread makes that tree for one row:
// its blocks' sums in that order, each plus a zero (a parameter, so the add stays), combined as
// a binary counter; a lane without a block adds 0, which leaves a sum as it is.
// The scale's product is __fmul_rn: glm_linear stores it, so it must not fuse into tanhf.

// Rows a block of ink_ctrl_gate (its threads twice that: the gain rows, then the forget rows),
// its tokens and its widest rank.
constexpr int CTRL_ROWS = 32;
constexpr int CTRL_TOK  = 8;
constexpr int CTRL_RANK = 512;

// ink_ctrl_gate: params gain scale, forget scale, a zero; sources cur, msg [e, n], d [k, n]
// (the down projection), gain, forget F32 [k, e]; dst [e, n]: ink_ctrl_out's expression of
// row r's gain and forget products. k a multiple of 32, at most CTRL_RANK; n at most CTRL_TOK.
__global__ void __launch_bounds__(2 * CTRL_ROWS) ctrl_gate_kernel(const A a) {
    const int64_t e = a.ne[0], n = a.ne[1], k = a.src_ne[2][0];
    {
        // this thread's gain or forget row, its 128-byte lines (l2_prefetch)
        const int64_t r = (int64_t) blockIdx.x * CTRL_ROWS + threadIdx.x % CTRL_ROWS;
        const int     g = threadIdx.x / CTRL_ROWS;
        if (r < e) {
            const char * row = src(a, 3 + g) + r * a.src_nb[3 + g][1];
            for (int64_t b = 0; b < k * 4; b += 128) {
                l2_prefetch(row + b);
            }
        }
    }
    pdl_wait();
    const float   zero = pf(a.params[2]);
    const int     nb = (int) (k / 32);
    __shared__ float xs[CTRL_TOK * CTRL_RANK];
    __shared__ float yf[CTRL_TOK][CTRL_ROWS];
    for (int64_t i = threadIdx.x; i < n * k; i += blockDim.x) {
        const int64_t t = i / k, j = i % k;
        xs[i] = ((const float *) (src(a, 2) + t * a.src_nb[2][1]))[j];
    }
    __syncthreads();
    const int     g  = threadIdx.x / CTRL_ROWS;  // 0 the gain row, 1 the forget row
    const int     rl = threadIdx.x % CTRL_ROWS;
    const int64_t r  = (int64_t) blockIdx.x * CTRL_ROWS + rl;
    const bool    live = r < e;
    // the binary counter's partial sums: 1, 2, 4 and 8 lanes
    float s1[CTRL_TOK], s2[CTRL_TOK], s4[CTRL_TOK], s8[CTRL_TOK], y[CTRL_TOK];
    if (live) {
        const float4 * row = (const float4 *) (src(a, 3 + g) + r * a.src_nb[3 + g][1]);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            // lane l of glm_linear's warp: i bit-reversed
            const int l = ((i & 1) << 3) | ((i & 2) << 1) | ((i & 4) >> 1) | ((i & 8) >> 3);
            float     c[CTRL_TOK];
#pragma unroll
            for (int t = 0; t < CTRL_TOK; t++) {
                c[t] = 0.0f;
            }
            if (l < nb) {
                float4 w[8];
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    w[q] = __ldg(row + l * 8 + q);
                }
#pragma unroll
                for (int t = 0; t < CTRL_TOK; t++) {
                    if (t < n) {
                        float acc = 0.0f;
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const float v[16] = {w[4 * h].x,     w[4 * h].y,     w[4 * h].z,     w[4 * h].w,
                                                 w[4 * h + 1].x, w[4 * h + 1].y, w[4 * h + 1].z, w[4 * h + 1].w,
                                                 w[4 * h + 2].x, w[4 * h + 2].y, w[4 * h + 2].z, w[4 * h + 2].w,
                                                 w[4 * h + 3].x, w[4 * h + 3].y, w[4 * h + 3].z, w[4 * h + 3].w};
                            const float * x = xs + t * k + l * 32 + 16 * h;
                            float         s = 0.0f;
#pragma unroll
                            for (int j = 0; j < 16; j++) {
                                s += v[j] * x[j];
                            }
                            acc += s;
                        }
                        c[t] = acc + zero;  // warp_sum's xor 16: lane l + 16 holds 0
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < CTRL_TOK; t++) {
                float v = c[t];
                if ((i & 1) == 0) {
                    s1[t] = v;
                    continue;
                }
                v = s1[t] + v;
                if ((i & 2) == 0) {
                    s2[t] = v;
                    continue;
                }
                v = s2[t] + v;
                if ((i & 4) == 0) {
                    s4[t] = v;
                    continue;
                }
                v = s4[t] + v;
                if ((i & 8) == 0) {
                    s8[t] = v;
                    continue;
                }
                y[t] = s8[t] + v;
            }
        }
        const float sc = g ? pf(a.params[1]) : pf(a.params[0]);
#pragma unroll
        for (int t = 0; t < CTRL_TOK; t++) {
            y[t] = __fmul_rn(y[t], sc);
            if (g == 1 && t < n) {
                yf[t][rl] = y[t];
            }
        }
    }
    __syncthreads();
    if (!live || g != 0) {
        return;
    }
#pragma unroll
    for (int t = 0; t < CTRL_TOK; t++) {
        if (t < n) {
            const int64_t i  = t * e + r;
            const float   c  = ((const float *) src(a, 0))[i];
            const float   m  = ((const float *) src(a, 1))[i];
            const float   gt = tanhf(y[t]);
            const float   ft = tanhf(yf[t][rl]);
            ((float *) a.dst)[i] = c - ft * c + gt * m;
        }
    }
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

// glm_add's `res + y` for ink_conv's residual (params[4], the source after the dynamic taps),
// or y: an add of its own, never fused with conv_out's product.
__device__ __forceinline__ float conv_res(const A & a, bool dyn, int64_t i, float y) {
    if (a.params[4] == 0) {
        return y;
    }
    return __fadd_rn(((const float *) src(a, dyn ? 4 : 3))[i], y);
}

// ---- ink_conv: params n_tok, off, has_dyn, scale, has_res; sources x [c, ng], w [K, c],
// state [K - 1, width, n_seqs] (written back), dyn [K * heads, ng], the residual res [c, ng]
// added to the output. One thread per (sequence,
// channel), the window in registers, token after token: a decode step's. ----
__global__ void conv_kernel(const A a) {
    const int64_t n_tok = a.params[0], off = a.params[1];
    const bool    dyn   = a.params[2] != 0;
    const float   scale = pf(a.params[3]);
    const int64_t c = a.ne[0];
    const int     K = (int) a.src_ne[1][0];
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t s  = blockIdx.y;
    float * hist = (float *) ((char *) a.src[2] + s * a.src_nb[2][2] + (off + ch) * a.src_nb[2][1]);
    const float * w = (const float *) (src(a, 1) + ch * a.src_nb[1][1]);
    if (ch < c) {
        l2_prefetch(w);
        l2_prefetch(hist);
    }
    pdl_wait();
    if (ch >= c) {
        return;
    }
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
        out[t * c + ch]  = conv_res(a, dyn, t * c + ch, conv_out(tw, win, K, xv, dr, dyn, scale));
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
    pdl_wait();
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
    ((float *) a.dst)[t * c + ch] = conv_res(a, dyn, t * c + ch, conv_out(tw, win, K, xv, dr, dyn, scale));
}

// ---- ink_conv's history after a prompt step: the last K - 1 inputs, as conv_kernel leaves
// it. A thread per (channel, sequence), after conv_prompt_kernel read the old one. ----
__global__ void conv_hist_kernel(const A a) {
    pdl_wait();
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

// ink_clip's factor for a row whose lane l holds elements l + 32 k in v[k] (0 past the row):
// the one expression of clip_kernel and kv_kernel.
__device__ __forceinline__ float clip_factor(const float (&v)[4], float thr) {
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        ss += v[k] * v[k];
    }
    ss               = warp_sum(ss);
    const float norm = sqrtf(ss) + 1e-6f;
    return fminf(fmaxf(thr / norm, 0.f), 1.f);
}

// ---- ink_clip: params thr, has_scale, has_tau, floor, alpha; sources x [d, h, n], scale [d, h],
// pos [n]. One warp per row (d at most 128). ----
// ink_clip of one row (lane l with elements l + 32 k): x times sc (if given) clipped to thr,
// times the log-length factor of position p (if pos): the one expression of clip_kernel and
// clip2_kernel.
__device__ __forceinline__ void clip_row(
    const float * x, const float * sc, int64_t d, float thr, const int32_t * pos, int64_t t,
    float floor_, float alpha, float * o, int lane) {
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        v[k]            = i < d ? x[i] * (sc ? sc[i] : 1.f) : 0.f;
    }
    float f = clip_factor(v, thr);
    if (pos) {
        const int32_t p = pos[t];
        const float   r = fmaxf(((float) p + 1.f) * (1.f / floor_), 1.f);
        f *= logf(r) * alpha + 1.f;
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        if (i < d) {
            o[i] = v[k] * f;
        }
    }
}

__global__ void clip_kernel(const A a) {
    pdl_wait();
    const float thr       = pf(a.params[0]);
    const bool  has_scale = a.params[1] != 0, has_tau = a.params[2] != 0;
    const float floor_    = pf(a.params[3]), alpha = pf(a.params[4]);
    const int64_t d = a.ne[0], h = a.ne[1], rows = h * a.ne[2];
    const int     lane = threadIdx.x % 32;
    const int64_t row  = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    const int64_t   hd  = row % h, t = row / h;
    const float *   x   = (const float *) src(a, 0) + row * d;
    const float *   sc  = has_scale ? (const float *) src(a, 1) + hd * d : nullptr;
    const int32_t * pos = has_tau ? (const int32_t *) src(a, 1 + (has_scale ? 1 : 0)) : nullptr;
    clip_row(x, sc, d, thr, pos, t, floor_, alpha, (float *) a.dst + row * d, lane);
}

// ---- ink_clip2 (#1221 r6): ink_clip of two tensors in one launch. Params thr0, thr1, has_tau,
// floor, alpha; sources x0 [d0, h0, n], scale0 [d0, h0], x1 [d1, h1, n], scale1 [d1, h1],
// pos [n]; dst x0's rows, then x1's. A warp per row, as clip_kernel. ----
__global__ void clip2_kernel(const A a) {
    const bool    has_tau = a.params[2] != 0;
    const float   floor_  = pf(a.params[3]), alpha = pf(a.params[4]);
    const int64_t rows0 = a.src_ne[0][1] * a.src_ne[0][2];
    const int64_t rows  = rows0 + a.src_ne[2][1] * a.src_ne[2][2];
    const int     lane  = threadIdx.x % 32;
    int64_t       row   = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const bool    live  = row < rows;
    const int     j   = row >= rows0 ? 1 : 0;
    const int64_t off = j ? rows0 * a.src_ne[0][0] : 0;
    row -= j * rows0;
    const int64_t   d = a.src_ne[2 * j][0], h = a.src_ne[2 * j][1];
    const int64_t   hd = row % h, t = row / h;
    const float *   sc  = (const float *) src(a, 2 * j + 1) + hd * d;
    // the row's scale, a 128-byte line a lane (l2_prefetch)
    if (live && 32 * lane < d) {
        l2_prefetch(sc + 32 * lane);
    }
    pdl_wait();
    if (!live) {
        return;
    }
    const float *   x   = (const float *) src(a, 2 * j) + row * d;
    const int32_t * pos = has_tau ? (const int32_t *) src(a, 4) : nullptr;
    clip_row(x, sc, d, pf(a.params[j]), pos, t, floor_, alpha, (float *) a.dst + off + row * d, lane);
}

// ---- ink_kv (#1221 r6): a decode step's K and V, the nodes ink_conv (K, then V), ink_clip (K,
// no scale or tau), ink_rms (V, no weight) and two ds4_cache_write (ratio 1) in one. Params:
// base (the state channel of K's first; V's follow K's), has_dyn, the conv scale, the clip
// threshold, rms eps, d; sources kp, vp [kvw, n], kw, vw [K, kvw], state [K - 1, width, n]
// (written back), then without has_dyn kc, vc (F16 [row, cells], written) and cells I32 [n],
// with it kdyn, vdyn [K * heads, n] and no cache writes (ggml takes 9 sources; ds4_cache_write
// nodes write them); dst [kvw, n, 2] K then V. A warp per head row (d at most 128), lane l
// with elements l + 32 k:
// its conv outputs are conv_kernel's (conv_out), the clip clip_kernel's lanes (clip_factor), the
// rms rms_kernel<1>'s sums (warp k of its block holds elements 32 k ..: their double warp sums,
// then the sum of the 8 warps', 0 past the row) and rms_scale. ----
__global__ void kv_kernel(const A a) {
    const int64_t base = a.params[0];
    const bool    dyn  = a.params[1] != 0;
    const float   scale = pf(a.params[2]), thr = pf(a.params[3]), eps = pf(a.params[4]);
    const int64_t d = a.params[5], kvw = a.ne[0], n = a.ne[1], nr = kvw / d;
    const int     K    = (int) a.src_ne[2][0];
    const int     lane = threadIdx.x % 32;
    const int64_t row  = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const bool    live = row < 2 * nr * n;
    const int64_t t = row / (2 * nr), hr = row % (2 * nr);
    const int     vv = hr >= nr ? 1 : 0;
    const int64_t head = hr - vv * nr;
    // the lane's conv taps and history (l2_prefetch), read in the loop below
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t ch = head * d + lane + 32 * k;
        if (live && lane + 32 * k < d) {
            l2_prefetch(src(a, 2 + vv) + ch * a.src_nb[2 + vv][1]);
            l2_prefetch((const char *) a.src[4] + t * a.src_nb[4][2] + (base + vv * kvw + ch) * a.src_nb[4][1]);
        }
    }
    pdl_wait();
    if (!live) {
        return;
    }
    const float * x    = (const float *) src(a, vv) + t * kvw;
    const float * dr0  = dyn ? (const float *) (src(a, 5 + vv) + t * a.src_nb[5 + vv][1]) : nullptr;
    const int64_t heads = dyn ? a.src_ne[5 + vv][0] / K : 1;
    float         y[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        y[k]            = 0.f;
        if (i >= d) {
            continue;
        }
        const int64_t ch   = head * d + i;
        float *       hist = (float *) ((char *) a.src[4] + t * a.src_nb[4][2] +
                                  (base + vv * kvw + ch) * a.src_nb[4][1]);
        const float * w    = (const float *) (src(a, 2 + vv) + ch * a.src_nb[2 + vv][1]);
        float         tw[CONV_TAPS], win[CONV_TAPS];
#pragma unroll
        for (int j = 0; j < CONV_TAPS; j++) {
            tw[j]  = j < K ? w[j] : 0.f;
            win[j] = j < K - 1 ? hist[j] : 0.f;
        }
        const int64_t hh = dyn ? ch / (kvw / heads) : 0;
        const float * dr = dyn ? dr0 + K * hh : nullptr;
        y[k]             = conv_out(tw, win, K, x[ch], dr, dyn, scale);
#pragma unroll
        for (int j = 0; j + 1 < CONV_TAPS; j++) {
            if (j + 1 < K) {
                win[j] = win[j + 1];
            }
        }
#pragma unroll
        for (int j = 0; j < CONV_TAPS; j++) {
            if (j < K - 1) {
                hist[j] = win[j];
            }
        }
    }
    float o[4];
    if (!vv) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            v[k] = lane + 32 * k < d ? y[k] * 1.f : 0.f;
        }
        const float f = clip_factor(v, thr);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            o[k] = v[k] * f;
        }
    } else {
        double p[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            double sum = 0.0;
            if (lane + 32 * k < d) {
                sum += (double) y[k] * y[k];
            }
            p[k] = warp_sum(sum);
        }
        const double q = lane == 0 ? p[0] : lane == 1 ? p[1] : lane == 2 ? p[2] : lane == 3 ? p[3] : 0.0;
        const float  s = rms_scale(warp_sum(q), d, eps);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            o[k] = y[k] * s;
        }
    }
    float * out = (float *) a.dst + (vv * n + t) * kvw + head * d;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        if (i < d) {
            out[i] = o[k];
        }
    }
    if (dyn) {
        return;
    }
    __half * c = (__half *) ((char *) a.src[5 + vv] + (int64_t) ((const int32_t *) src(a, 7))[t] * a.src_nb[5 + vv][1]) + head * d;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int64_t i = lane + 32 * k;
        if (i < d) {
            c[i] = __float2half(o[k]);
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
    pdl_wait();
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
    __shared__ float   ob[ATTN_ROWS][ATTN_D];
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
        // Values: key kk of the chunk, DPL dims per lane, VU keys' loads ahead of their sums
        // (#1221; the sums run in the same order). Half as many for prompt rows, which keeps
        // their registers at 3 blocks an SM.
        const int64_t nk = min((int64_t) 32, kend - base + 1);
        constexpr int VU = R > ATTN_DECODE_ROWS ? ATTN_VU / 2 : ATTN_VU;
        for (int kk0 = 0; kk0 < nk; kk0 += VU) {
            HalfVec<DPL> vraw[VU];
#pragma unroll
            for (int u = 0; u < VU; u++) {
                const int64_t ck = __shfl_sync(0xffffffffu, c, kk0 + u);
                if (kk0 + u < nk) {
                    const char * vr = src(a, 2) + ck * a.src_nb[2][1] + kvoff;
                    vraw[u]         = ((const HalfVec<DPL> *) vr)[lane];
                }
            }
#pragma unroll
            for (int u = 0; u < VU; u++) {
                if (kk0 + u < nk) {
                    float vf[DPL];
#pragma unroll
                    for (int j = 0; j < DPL; j++) {
                        vf[j] = __half2float(vraw[u].h[j]);
                    }
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        const float p = __shfl_sync(0xffffffffu, s[r], kk0 + u);
#pragma unroll
                        for (int j = 0; j < DPL; j++) {
                            acc[r][j] += p * vf[j];
                        }
                    }
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
    }
    __syncthreads();
    const float * sinks = (const float *) src(a, 7);
    // Each output's sum of the warps' parts, a warp at a time in warp order: one buffer in
    // shared memory instead of every warp's parts (#1221), the same sums.
#pragma unroll
    for (int w = 0; w < ATTN_WARPS; w++) {
        if (warp == w) {
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r < rows) {
                    const int64_t t = seq * n_tok + tq0 + r / pk;
                    const int64_t h = g * pk + r % pk;
                    float         M = sinks[t * nh + h];
                    for (int w2 = 0; w2 < ATTN_WARPS; w2++) {
                        M = fmaxf(M, pm[w2][r]);
                    }
                    const float f = __expf(pm[w][r] - M);
#pragma unroll
                    for (int j = 0; j < DPL; j++) {
                        float o = w == 0 ? 0.f : ob[r][lane * DPL + j];
                        o += acc[r][j] * f;
                        ob[r][lane * DPL + j] = o;
                    }
                }
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < rows * D; i += blockDim.x) {
        const int     r = i / D, k = i % D;
        const int64_t t = seq * n_tok + tq0 + r / pk;
        const int64_t h = g * pk + r % pk;
        const float   sink = sinks[t * nh + h];
        float         M    = sink;
        for (int w = 0; w < ATTN_WARPS; w++) {
            M = fmaxf(M, pm[w][r]);
        }
        float den = __expf(sink - M);
        for (int w = 0; w < ATTN_WARPS; w++) {
            den += pl[w][r] * __expf(pm[w][r] - M);
        }
        ((float *) a.dst)[(t * nh + h) * D + k] = ob[r][k] / den;
    }
}

// ---- ink_sigmul: x * sigmoid(g) ----
__global__ void sigmul_kernel(const A a) {
    pdl_wait();
    const int64_t n = a.ne[0] * a.ne[1] * a.ne[2] * a.ne[3];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ((float *) a.dst)[i] = ((const float *) src(a, 0))[i] * sigmoidf_(((const float *) src(a, 1))[i]);
}

// ---- ink_situ: params beta, lb, bf16 (BF16 out, rounded to nearest even); source y [2f, k, n] ----
__global__ void situ_kernel(const A a) {
    pdl_wait();
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
    if (a.params[2]) {
        ((__nv_bfloat16 *) a.dst)[i] = __float2bfloat16_rn(act * uu);
    } else {
        ((float *) a.dst)[i] = act * uu;
    }
}

// Element `lane` of 32-value group gi of row r of expert e of the stack in source 0, times its
// scale (a Q8_0 block's, an MXFP8 group's from source 3), for a stack of type TY (F32 for any
// other).
// The element as loaded (load) and as the sums take it (value), so a kernel can have the loads
// in flight while it computes.
template <int TY> struct MmRaw {
    float v;
    __device__ __forceinline__ void load(const A & a, int64_t e, int64_t r, int64_t gi,
                                         int lane) {
        const char * row = src(a, 0) + e * a.src_nb[0][2] + r * a.src_nb[0][1];
        if constexpr (TY == TYPE_Q8_0) {
            const char * b = row + gi * 34;
            d              = *(const __half *) b;
            q              = ((const int8_t *) (b + 2))[lane];
        } else if constexpr (TY == TYPE_I8) {
            const int64_t k = a.src_ne[0][0], m = a.src_ne[0][1];
            s               = ((const float *) src(a, 3))[(e * m + r) * (k / 32) + gi];
            b8              = ((const uint8_t *) row)[gi * 32 + lane];
        } else if constexpr (TY == TYPE_F16) {
            d = ((const __half *) row)[gi * 32 + lane];
        } else if constexpr (TY == TYPE_BF16) {
            bf = ((const __nv_bfloat16 *) row)[gi * 32 + lane];
        } else {
            v = ((const float *) row)[gi * 32 + lane];
        }
    }
    __device__ __forceinline__ float value() const {
        if constexpr (TY == TYPE_Q8_0) {
            return __half2float(d) * (float) q;
        } else if constexpr (TY == TYPE_I8) {
            return e4m3(b8) * s;
        } else if constexpr (TY == TYPE_F16) {
            return __half2float(d);
        } else if constexpr (TY == TYPE_BF16) {
            return __bfloat162float(bf);
        } else {
            return v;
        }
    }
    __half        d;
    int8_t        q;
    uint8_t       b8;
    float         s;
    __nv_bfloat16 bf;
};

template <int TY>
__device__ __forceinline__ float mm_w(const A & a, int64_t e, int64_t r, int64_t gi, int lane) {
    MmRaw<TY> w;
    w.load(a, e, r, gi, lane);
    return w.value();
}

// ---- ink_mm_id: sources w [k, m, E], x [k, nx, n_tok], ids [n_used, n_tok] (I32,
// contiguous), scale. Every output is warp_sum over the 32 lanes of lane l's part, the sum in
// group order of w[32 gi + l] * x[32 gi + l]; both kernels keep exactly that (#1221: they
// replace a warp-per-row kernel with these sums), so any split of the work gives the same bits.
//
// mm_id_kernel, for prompt steps: a block per (MM_BM output rows, expert). It lists the pairs
// routed to its expert, then runs them MM_BN at a time over K in passes of MM_KC, the pass's
// weights (as floats) and inputs in shared memory, a warp per 8 rows x 8 pairs. ----
// A thread's share of a pass: the 4 columns 4 (thread % MM_KQ) + i of rows thread / MM_KQ +
// MM_RS s, loaded raw (packed words) so a pass in flight costs few registers.
constexpr int MM_KQ = MM_KC / 4;
constexpr int MM_RS = MM_THREADS / MM_KQ;
constexpr int MM_WN = MM_BM / MM_RS;
constexpr int MM_XN = MM_BN / MM_RS;
static_assert(MM_KC % 32 == 0 && MM_THREADS % MM_KQ == 0 && MM_BM % MM_RS == 0 &&
                  MM_BN % MM_RS == 0,
              "mm passes");

// The N / 4 words of the N bytes at p (N 4, 8 or 16; p at least 2-aligned), by the widest
// access p's alignment allows.
template <int N> __device__ __forceinline__ void mm_words(uint32_t (&w)[N / 4], const char * p) {
    const uintptr_t al = (uintptr_t) p;
    if constexpr (N == 16) {
        if (al % 16 == 0) {
            const uint4 v = *(const uint4 *) p;
            w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
            return;
        }
    }
    if (al % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N / 4; i++) {
            w[i] = ((const uint32_t *) p)[i];
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < N / 4; i++) {
        const uint16_t * h = (const uint16_t *) p;
        w[i]               = (uint32_t) h[2 * i] | (uint32_t) h[2 * i + 1] << 16;
    }
}

// Weights c .. c + 3 (c a multiple of 4) of a row, raw; value(i) is weight c + i, the same
// float as mm_w gives.
template <int TY> struct MmQuad {
    static constexpr int NW = TY == TYPE_Q8_0 || TY == TYPE_I8              ? 1
                              : TY == TYPE_F16 || TY == TYPE_BF16 ? 2
                                                                  : 4;
    uint32_t             w[NW];
    float                s;
    __half               d;
    __device__ __forceinline__ void load(const A & a, int64_t e, int64_t r, int64_t c) {
        const char *  row = src(a, 0) + e * a.src_nb[0][2] + r * a.src_nb[0][1];
        const int64_t gi = c / 32, l0 = c % 32;
        if constexpr (TY == TYPE_Q8_0) {
            const char * b = row + gi * 34;
            d              = *(const __half *) b;
            mm_words<4>(w, b + 2 + l0);
        } else if constexpr (TY == TYPE_I8) {
            const int64_t k = a.src_ne[0][0], m = a.src_ne[0][1];
            s               = ((const float *) src(a, 3))[(e * m + r) * (k / 32) + gi];
            mm_words<4>(w, row + c);
        } else if constexpr (NW == 2) {
            mm_words<8>(w, row + 2 * c);
        } else {
            mm_words<16>(w, row + 4 * c);
        }
    }
    __device__ __forceinline__ float value(int i) const {
        if constexpr (TY == TYPE_Q8_0) {
            return __half2float(d) * (float) (int8_t) (uint8_t) (w[0] >> 8 * i);
        } else if constexpr (TY == TYPE_I8) {
            return e4m3((uint8_t) (w[0] >> 8 * i)) * s;
        } else if constexpr (TY == TYPE_F16) {
            return __half2float(__ushort_as_half((unsigned short) (w[i / 2] >> 16 * (i % 2))));
        } else if constexpr (TY == TYPE_BF16) {
            const unsigned short h = (unsigned short) (w[i / 2] >> 16 * (i % 2));
            return __bfloat162float(__ushort_as_bfloat16(h));
        } else {
            return __uint_as_float(w[i]);
        }
    }
};

// Loads the thread's share of the pass at k0: weights of rows r0 + ..., inputs of pairs ...
template <int TY>
__device__ __forceinline__ void mm_fetch(const A & a, MmQuad<TY> (&wq)[MM_WN],
                                         uint32_t (&xq)[MM_XN][4], const float * x,
                                         const int64_t * xrow, int64_t e, int64_t r0, int64_t m,
                                         int np, int64_t k, int64_t k0) {
    const int     kq = threadIdx.x % MM_KQ, r1 = threadIdx.x / MM_KQ;
    const int64_t c  = k0 + 4 * kq;
    if (c >= k) {
        return;
    }
#pragma unroll
    for (int s = 0; s < MM_WN; s++) {
        const int64_t r = r0 + r1 + MM_RS * s;
        if (r < m) {
            wq[s].load(a, e, r, c);
        }
    }
#pragma unroll
    for (int s = 0; s < MM_XN; s++) {
        const int p = r1 + MM_RS * s;
        if (p < np) {
            mm_words<16>(xq[s], (const char *) (x + xrow[p] + c));
        }
    }
}

template <int TY>
__device__ __forceinline__ void mm_id_tile(const A & a, float (*ws)[MM_KC], float (*xs)[MM_KC],
                                           int * list, int64_t * xrow, int * cnt) {
    const int64_t k = a.src_ne[0][0], E = a.src_ne[0][2];
    const int64_t m = a.ne[0], n_used = a.ne[1], n_tok = a.ne[2];
    const int64_t nx = a.src_ne[1][1];
    const int64_t e = blockIdx.y, r0 = (int64_t) blockIdx.x * MM_BM;
    const int     warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int     wr = warp % (MM_BM / 8) * 8, wc = warp / (MM_BM / 8) * 8;
    const int32_t * ids = (const int32_t *) src(a, 2);
    const float *   x   = (const float *) src(a, 1);
    float *         out = (float *) a.dst;
    const int64_t   npairs = n_used * n_tok;
    for (int64_t pos = 0; pos < npairs;) {
        if (threadIdx.x == 0) {
            *cnt = 0;
        }
        __syncthreads();
        // List the expert's pairs while the list can take another pass of the block.
        int n;
        do {
            const int64_t i = pos + threadIdx.x;
            if (i < npairs) {
                const int32_t id = ids[i];
                if (id == e) {
                    list[atomicAdd(cnt, 1)] = (int) i;
                } else if (e == 0 && (id < 0 || id >= E)) {
                    for (int rr = 0; rr < MM_BM && r0 + rr < m; rr++) {
                        out[i * m + r0 + rr] = 0.f;
                    }
                }
            }
            pos += blockDim.x;
            __syncthreads();
            n = *cnt;
            __syncthreads();
        } while (pos < npairs && n <= MM_LIST - (int) blockDim.x);
        for (int b = 0; b < n; b += MM_BN) {
            const int np = min(MM_BN, n - b);
            if (threadIdx.x < np) {
                const int64_t pr = list[b + threadIdx.x];
                xrow[threadIdx.x] = (pr / n_used * nx + (nx == 1 ? 0 : pr % n_used)) * k;
            }
            // This warp's pairs (none when not positive).
            const int nj = min(8, np - wc);
            float     acc[8][8] = {};
            // The next pass's weights and inputs, loading while this one computes.
            MmQuad<TY> wq[MM_WN];
            uint32_t   xq[MM_XN][4];
            __syncthreads();
            mm_fetch<TY>(a, wq, xq, x, xrow, e, r0, m, np, k, 0);
            for (int64_t k0 = 0; k0 < k; k0 += MM_KC) {
                const int kc = (int) min((int64_t) MM_KC, k - k0);
                __syncthreads();
                const int kk = 4 * (threadIdx.x % MM_KQ), r1 = threadIdx.x / MM_KQ;
                if (kk < kc) {
#pragma unroll
                    for (int s = 0; s < MM_WN; s++) {
                        const int rr = r1 + MM_RS * s;
                        const bool in = r0 + rr < m;
                        *(float4 *) &ws[rr][kk] =
                            make_float4(in ? wq[s].value(0) : 0.f, in ? wq[s].value(1) : 0.f,
                                        in ? wq[s].value(2) : 0.f, in ? wq[s].value(3) : 0.f);
                    }
#pragma unroll
                    for (int s = 0; s < MM_XN; s++) {
                        const int p = r1 + MM_RS * s;
                        if (p < np) {
                            const uint32_t(&q)[4] = xq[s];
                            *(uint4 *) &xs[p][kk] = make_uint4(q[0], q[1], q[2], q[3]);
                        }
                    }
                }
                __syncthreads();
                if (k0 + MM_KC < k) {
                    mm_fetch<TY>(a, wq, xq, x, xrow, e, r0, m, np, k, k0 + MM_KC);
                }
                if (nj > 0) {
                    for (int g = 0; g < kc / 32; g++) {
                        float wv[8], xv[8];
#pragma unroll
                        for (int ii = 0; ii < 8; ii++) {
                            wv[ii] = ws[wr + ii][g * 32 + lane];
                        }
#pragma unroll
                        for (int jj = 0; jj < 8; jj++) {
                            xv[jj] = jj < nj ? xs[wc + jj][g * 32 + lane] : 0.f;
                        }
#pragma unroll
                        for (int ii = 0; ii < 8; ii++) {
#pragma unroll
                            for (int jj = 0; jj < 8; jj++) {
                                acc[ii][jj] += wv[ii] * xv[jj];
                            }
                        }
                    }
                }
            }
            if (nj > 0) {
#pragma unroll
                for (int ii = 0; ii < 8; ii++) {
#pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        const float   s = warp_sum(acc[ii][jj]);
                        const int64_t r = r0 + wr + ii;
                        if (lane == 0 && jj < nj && r < m) {
                            out[(int64_t) list[b + wc + jj] * m + r] = s;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

static_assert((MM_BM / 8) * (MM_BN / 8) * 32 == MM_THREADS, "ink_mm_id: a warp per 8 x 8");
static_assert(MM_BN <= MM_THREADS && MM_LIST >= 2 * MM_THREADS, "ink_mm_id: list sizes");

__global__ void __launch_bounds__(MM_THREADS) mm_id_kernel(const A a) {
    pdl_wait();
    __shared__ __align__(16) float ws[MM_BM][MM_KC];
    __shared__ __align__(16) float xs[MM_BN][MM_KC];
    __shared__ int     list[MM_LIST];
    __shared__ int64_t xrow[MM_BN];
    __shared__ int     cnt;
    switch (a.src_type[0]) {
        case TYPE_Q8_0:
            mm_id_tile<TYPE_Q8_0>(a, ws, xs, list, xrow, &cnt);
            break;
        case TYPE_I8:
            mm_id_tile<TYPE_I8>(a, ws, xs, list, xrow, &cnt);
            break;
        case TYPE_F16:
            mm_id_tile<TYPE_F16>(a, ws, xs, list, xrow, &cnt);
            break;
        case TYPE_BF16:
            mm_id_tile<TYPE_BF16>(a, ws, xs, list, xrow, &cnt);
            break;
        default:
            mm_id_tile<0>(a, ws, xs, list, xrow, &cnt);
            break;
    }
}

// mm_id_vec_kernel, for steps of no more pairs than experts (decode): a block per (output rows,
// pair), a warp per MM_VEC_ROWS rows reading the weights straight from memory, MM_VEC_U groups
// of loads ahead of their sums.
template <int TY> __device__ __forceinline__ void mm_id_vec(const A & a) {
    const int64_t k = a.src_ne[0][0], E = a.src_ne[0][2];
    const int64_t m = a.ne[0], n_used = a.ne[1];
    const int64_t nx = a.src_ne[1][1];
    const int64_t pr = blockIdx.y;
    const int     lane = threadIdx.x % 32;
    const int64_t r0 = ((int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * MM_VEC_ROWS;
    if (r0 >= m) {
        return;
    }
    const int     nr  = (int) min((int64_t) MM_VEC_ROWS, m - r0);
    float *       out = (float *) a.dst + pr * m + r0;
    const int32_t e   = ((const int32_t *) src(a, 2))[pr];
    if (e < 0 || e >= E) {
        if (lane < nr) {
            out[lane] = 0.f;
        }
        return;
    }
    const float * xr =
        (const float *) src(a, 1) + (pr / n_used * nx + (nx == 1 ? 0 : pr % n_used)) * k;
    const int64_t ng = k / 32;
    float         acc[MM_VEC_ROWS] = {};
    int64_t       gi               = 0;
    for (; gi + MM_VEC_U <= ng; gi += MM_VEC_U) {
        float xv[MM_VEC_U], wv[MM_VEC_U][MM_VEC_ROWS];
#pragma unroll
        for (int u = 0; u < MM_VEC_U; u++) {
            xv[u] = xr[(gi + u) * 32 + lane];
#pragma unroll
            for (int rr = 0; rr < MM_VEC_ROWS; rr++) {
                wv[u][rr] = rr < nr ? mm_w<TY>(a, e, r0 + rr, gi + u, lane) : 0.f;
            }
        }
#pragma unroll
        for (int u = 0; u < MM_VEC_U; u++) {
#pragma unroll
            for (int rr = 0; rr < MM_VEC_ROWS; rr++) {
                acc[rr] += wv[u][rr] * xv[u];
            }
        }
    }
    for (; gi < ng; gi++) {
        const float xv = xr[gi * 32 + lane];
#pragma unroll
        for (int rr = 0; rr < MM_VEC_ROWS; rr++) {
            const float v = rr < nr ? mm_w<TY>(a, e, r0 + rr, gi, lane) : 0.f;
            acc[rr] += v * xv;
        }
    }
#pragma unroll
    for (int rr = 0; rr < MM_VEC_ROWS; rr++) {
        const float s = warp_sum(acc[rr]);
        if (lane == 0 && rr < nr) {
            out[rr] = s;
        }
    }
}

__global__ void mm_id_vec_kernel(const A a) {
    pdl_wait();
    switch (a.src_type[0]) {
        case TYPE_Q8_0:
            mm_id_vec<TYPE_Q8_0>(a);
            break;
        case TYPE_I8:
            mm_id_vec<TYPE_I8>(a);
            break;
        case TYPE_F16:
            mm_id_vec<TYPE_F16>(a);
            break;
        case TYPE_BF16:
            mm_id_vec<TYPE_BF16>(a);
            break;
        default:
            mm_id_vec<0>(a);
            break;
    }
}

// mm_id_stage_kernel (#1221 r8), mm_id_vec_kernel's work for Q8_0 and MXFP8 stacks with the
// weights staged: mm_id_vec's loads are a byte a lane (32 bytes a warp load, and MXFP8's scale
// a load of its own), which kept the decode at ~140 GB/s (Q8_0) and ~90 GB/s (MXFP8) where the
// NVFP4 experts read ~250. Here a warp copies its rows MMS_G groups at a time into its own ring
// of MMS_STAGES shared-memory stages, 16-byte copies (cp.async on sm_80+) running MMS_STAGES - 1
// stages ahead of the sums, which then read the bytes and scales from shared memory. Lane l
// still sums w[32 gi + l] · x[32 gi + l] over the groups in order, the same value for each
// weight (mm_w's), and warp_sum adds the lanes: the same bits as mm_id_vec.
// A stage of a row is the 16-byte chunks covering its MMS_G groups (a Q8_0 row is only 2-byte
// aligned): each chunk holds at least one of the row's bytes, so it is inside the stack's
// memory. MXFP8: the weights' chunks, then the scales' at MMS_SCALES.
constexpr int MMS_ROWS   = 4;
constexpr int MMS_G      = 8;
constexpr int MMS_STAGES = 4;
constexpr int MMS_WARPS  = 8;
constexpr int MMS_SCALES = (MMS_G * 34 + 15 + 15) / 16 * 16;
constexpr int MMS_BYTES  = MMS_SCALES + (MMS_G * 4 + 12 + 15) / 16 * 16;
static_assert((MMS_G * 32 + 15 + 15) / 16 * 16 <= MMS_SCALES, "MXFP8 weights before the scales");
static_assert(MMS_SCALES / 16 <= 28 && (MMS_BYTES - MMS_SCALES) / 16 <= 4,
              "a row's chunks: weights lanes 0-27, scales lanes 28-31");
static_assert(MMS_ROWS == MM_VEC_ROWS && MMS_WARPS * 32 == MM_THREADS, "the vec kernel's grid");

// A 16-byte copy from global memory to shared memory, asynchronous on sm_80+ (and then waited
// for by mms_wait), a plain load and store below.
__device__ __forceinline__ void mms_copy(void * s, const void * g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa),
                 "l"(__cvta_generic_to_global(g))
                 : "memory");
#else
    *(uint4 *) s = *(const uint4 *) g;
#endif
}

__device__ __forceinline__ void mms_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

// Until at most N of this thread's committed stages are still in flight.
template <int N> __device__ __forceinline__ void mms_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// The 16-byte chunks covering [p, p + n): the first chunk's address, and p's offset in it.
__device__ __forceinline__ const char * mms_chunks(const char * p, int64_t n, int & off,
                                                   int & nch) {
    const uintptr_t a = (uintptr_t) p, a0 = a & ~(uintptr_t) 15;
    off = (int) (a - a0);
    nch = (int) ((a + n - a0 + 15) / 16);
    return (const char *) a0;
}

template <int TY>
__device__ __forceinline__ void mm_id_stage(const A & a, char (*ring)[MMS_ROWS][MMS_BYTES]) {
    static_assert(TY == TYPE_Q8_0 || TY == TYPE_I8, "staged stacks");
    constexpr int BPG = TY == TYPE_Q8_0 ? 34 : 32;
    const int64_t k = a.src_ne[0][0], E = a.src_ne[0][2];
    const int64_t m = a.ne[0], n_used = a.ne[1];
    const int64_t nx = a.src_ne[1][1];
    const int64_t pr = blockIdx.y;
    const int     lane = threadIdx.x % 32;
    const int64_t r0 = ((int64_t) blockIdx.x * MMS_WARPS + threadIdx.x / 32) * MMS_ROWS;
    if (r0 >= m) {
        return;
    }
    const int     nr  = (int) min((int64_t) MMS_ROWS, m - r0);
    float *       out = (float *) a.dst + pr * m + r0;
    const int32_t e   = ((const int32_t *) src(a, 2))[pr];
    if (e < 0 || e >= E) {
        if (lane < nr) {
            out[lane] = 0.f;
        }
        return;
    }
    const float * xr =
        (const float *) src(a, 1) + (pr / n_used * nx + (nx == 1 ? 0 : pr % n_used)) * k;
    const int64_t ng  = k / 32;
    const int     nst = (int) ((ng + MMS_G - 1) / MMS_G);
    const char *  w   = src(a, 0) + e * a.src_nb[0][2] + r0 * a.src_nb[0][1];
    const float * sc  = TY == TYPE_I8 ? (const float *) src(a, 3) + (e * m + r0) * ng : nullptr;
    // Stage st of row rr: its groups' bytes (and scales) and where they start in the stage.
    auto issue = [&](int st) {
        char(*buf)[MMS_BYTES] = ring[st % MMS_STAGES];
        const int64_t g0 = (int64_t) st * MMS_G, gn = min((int64_t) MMS_G, ng - g0);
#pragma unroll
        for (int rr = 0; rr < MMS_ROWS; rr++) {
            if (rr < nr) {
                int          off, nch;
                const char * c = mms_chunks(w + rr * a.src_nb[0][1] + g0 * BPG, gn * BPG, off, nch);
                if (lane < nch) {
                    mms_copy(buf[rr] + 16 * lane, c + 16 * lane);
                }
                if constexpr (TY == TYPE_I8) {
                    const char * s =
                        mms_chunks((const char *) (sc + rr * ng + g0), gn * 4, off, nch);
                    const int j = lane - (32 - 4);
                    if (j >= 0 && j < nch) {
                        mms_copy(buf[rr] + MMS_SCALES + 16 * j, s + 16 * j);
                    }
                }
            }
        }
    };
    float acc[MMS_ROWS] = {};
#pragma unroll
    for (int s = 0; s < MMS_STAGES - 1; s++) {
        if (s < nst) {
            issue(s);
        }
        mms_commit();
    }
    for (int st = 0; st < nst; st++) {
        mms_wait<MMS_STAGES - 2>();
        // Every lane's copies of stage st are in, and every lane is done with stage st - 1,
        // whose buffers stage st + MMS_STAGES - 1 takes.
        __syncwarp();
        if (st + MMS_STAGES - 1 < nst) {
            issue(st + MMS_STAGES - 1);
        }
        mms_commit();
        char(*buf)[MMS_BYTES] = ring[st % MMS_STAGES];
        const int64_t g0 = (int64_t) st * MMS_G;
        const int     gn = (int) min((int64_t) MMS_G, ng - g0);
        int           woff[MMS_ROWS], soff[MMS_ROWS];
#pragma unroll
        for (int rr = 0; rr < MMS_ROWS; rr++) {
            int nch;
            mms_chunks(w + rr * a.src_nb[0][1] + g0 * BPG, 0, woff[rr], nch);
            soff[rr] = 0;
            if constexpr (TY == TYPE_I8) {
                mms_chunks((const char *) (sc + rr * ng + g0), 0, soff[rr], nch);
            }
        }
#pragma unroll
        for (int g = 0; g < MMS_G; g++) {
            if (g < gn) {
                const float xv = xr[(g0 + g) * 32 + lane];
#pragma unroll
                for (int rr = 0; rr < MMS_ROWS; rr++) {
                    float v = 0.f;
                    if (rr < nr) {
                        const char * b = buf[rr] + woff[rr] + g * BPG;
                        if constexpr (TY == TYPE_Q8_0) {
                            const __half d = *(const __half *) b;
                            v = __half2float(d) * (float) ((const int8_t *) (b + 2))[lane];
                        } else {
                            const float s =
                                *(const float *) (buf[rr] + MMS_SCALES + soff[rr] + 4 * g);
                            v = e4m3(((const uint8_t *) b)[lane]) * s;
                        }
                    }
                    acc[rr] += v * xv;
                }
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < MMS_ROWS; rr++) {
        const float s = warp_sum(acc[rr]);
        if (lane == 0 && rr < nr) {
            out[rr] = s;
        }
    }
}

__global__ void __launch_bounds__(MMS_WARPS * 32) mm_id_stage_kernel(const A a) {
    pdl_wait();
    __shared__ __align__(16) char ring[MMS_WARPS][MMS_STAGES][MMS_ROWS][MMS_BYTES];
    char(*mine)[MMS_ROWS][MMS_BYTES] = ring[threadIdx.x / 32];
    if (a.src_type[0] == TYPE_Q8_0) {
        mm_id_stage<TYPE_Q8_0>(a, mine);
    } else {
        mm_id_stage<TYPE_I8>(a, mine);
    }
}

// ---- ink_softcap: params cap; source x [v, n]; dst [n_out, n] ----
__global__ void softcap_kernel(const A a) {
    pdl_wait();
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

extern "C" const void * const legion_ink_kernels[26] = {
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
    reinterpret_cast<const void *>(&mm_id_vec_kernel),
    reinterpret_cast<const void *>(&ctrl_gate_kernel),
    reinterpret_cast<const void *>(&rms_kernel<1>),
    reinterpret_cast<const void *>(&rms_kernel<3>),
    reinterpret_cast<const void *>(&rms_kernel<9>),
    reinterpret_cast<const void *>(&rms_kernel<16>),
    reinterpret_cast<const void *>(&kv_kernel),
    reinterpret_cast<const void *>(&clip2_kernel),
    reinterpret_cast<const void *>(&mm_id_stage_kernel),
};
