// qwen4exp's per-layer n-gram embedding (PLE) after its two matmuls, for prompt steps: two
// kernels instead of ggml's chain of about 30 nodes over [e * hc, n_tok] tensors (#439), with
// ggml-cuda's bits (both built with fast math):
//
// - ple_gate: per token, the key's and the query's (the residual streams') rms_norm times their
//   gammas (rms_norm_f32<1024, true> of norm.cu fused with the mul: 1024 threads per row, each
//   adding the squares of columns tid, tid + 1024, ... with FFMA, warp_reduce_sum, the 32 warp
//   sums reduced again; mean + eps as one FFMA of sum by 1 / ncols, rsqrtf, (scale * x) * g),
//   each stream's dot product of the two as mul then sum_rows (reduce_rows_f32 of
//   reduce_rows.cuh: 128 threads per row, or 512 when there are fewer than two rows per SM, eight
//   running sums per thread), then scale, abs, clamp, sqrt, sgn, mul and sigmoid (one value per
//   stream: the gate), the value row times the gate per stream and its rms_norm times the conv
//   gamma (the conv's input, written out with the gates);
// - ple_conv: the depthwise causal conv over [history | tokens] dilated by the n-gram size (one
//   rounded product per tap, added in tap order as the chain's mul and add nodes do), silu, plus
//   the gated value, plus the residual streams: the layer's new streams; then the last `hist`
//   inputs back to the history.
//
// Shapes, the CPU references and the parity tests: src/model/qwen4/ple_fused.rs and its tests.rs.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

extern "C" {

struct legion_ple_gate {
    const float * key;    // token t's hc * e key values at key + t * key_ld
    int64_t key_ld;
    const float * res;    // token t's hc streams of e values, back to back, at res + t * res_ld
    int64_t res_ld;
    const float * value;  // token t's e values at value + t * value_ld
    int64_t value_ld;
    const float * gk;     // [e * hc] each: the key's, the query's and the conv's gammas
    const float * gq;
    const float * gc;
    float * xn;           // [e * hc, n_tok] contiguous: the conv's input
    float * gate;         // [hc, n_tok] contiguous
    int32_t e, hc, n_tok;
    float eps, inv_sqrt_e, clamp_min;
};

int legion_ple_gate_launch(const legion_ple_gate * p, void * stream);

struct legion_ple_conv {
    const float * x;      // channel c of token t of sequence s at x[c + t * x_ld + s * x_sd]
    int64_t x_ld, x_sd;
    float * state;        // history j of channel c of sequence s at state[j + c * st_ld + s * st_sd]
    int64_t st_ld, st_sd;
    const float * w;      // tap k of channel c at w[k + c * kern]
    const float * value;  // column i's e values at value + i * value_ld (i = t + s * nt)
    int64_t value_ld;
    const float * gate;   // column i's hc gates at gate + i * hc
    const float * res;    // column i's streams at res + i * res_ld
    int64_t res_ld;
    float * out;          // [width, nt * ns] contiguous
    int32_t width, e, hc, nt, ns, kern, dil, hist;
};

int legion_ple_conv_launch(const legion_ple_conv * p, void * stream);

}  // extern "C"

namespace {

constexpr int PLE_THREADS = 1024;  // ggml-cuda's rms_norm block for rows of 1024 or more
constexpr int PLE_HC = 4;          // streams (qwen4exp)
constexpr int SUM_UNROLL = 8;      // reduce_row_f32's running sums per thread

// ggml-cuda's warp_reduce_sum (common.cuh): butterfly over the 32 lanes.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// ggml-cuda's block_reduce over the 1024 threads, for N sums at once (each its own tree, the
// same as alone).
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float (*red)[32]) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i < N; i++) {
        v[i] = warp_sum(v[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            red[i][warp] = v[i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; i++) {
        v[i] = warp_sum(red[i][lane]);
    }
    __syncthreads();
}

// rms_norm_f32's scale from a row's sum of squares (the same source: nvcc makes mean + eps one
// FFMA of the sum by MUFU.RCP(ncols), then MUFU.RSQ).
__device__ __forceinline__ float rms_scale(float tmp, int ncols, float eps) {
    const float mean = tmp / ncols;
    return rsqrtf(mean + eps);
}

// One block per token. SUMB: sum_rows' threads per row.
template <int SUMB>
__global__ void __launch_bounds__(PLE_THREADS) ple_gate_kernel(legion_ple_gate p) {
    __shared__ float red[2 * PLE_HC][32];
    __shared__ float dred[PLE_HC][SUMB / 32];
    __shared__ float s_gate[PLE_HC];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int64_t t = blockIdx.x;
    const int e = p.e;
    const float * key = p.key + t * p.key_ld;
    const float * res = p.res + t * p.res_ld;
    const float * val = p.value + t * p.value_ld;

    // The key's and the query's rms_norm sums, per stream.
    float s[2 * PLE_HC];
#pragma unroll
    for (int h = 0; h < PLE_HC; h++) {
        float sk = 0.0f;
        float sq = 0.0f;
        for (int col = tid; col < e; col += PLE_THREADS) {
            const float k = key[h * e + col];
            const float q = res[h * e + col];
            sk = __fmaf_rn(k, k, sk);
            sq = __fmaf_rn(q, q, sq);
        }
        s[h] = sk;
        s[PLE_HC + h] = sq;
    }
    block_sums(s, red);
    float scale_k[PLE_HC], scale_q[PLE_HC];
#pragma unroll
    for (int h = 0; h < PLE_HC; h++) {
        scale_k[h] = rms_scale(s[h], e, p.eps);
        scale_q[h] = rms_scale(s[PLE_HC + h], e, p.eps);
    }

    // Each stream's dot product of the normed key and query as ggml's mul then sum_rows: SUMB
    // threads per stream, thread i taking columns i, i + SUMB, ... into SUM_UNROLL running sums
    // in turn, then their sum, then the block reduce over the SUMB threads. The products are
    // recomputed from the inputs (L1/L2), rounded as the chain stores them.
    constexpr int PER_ROUND = PLE_THREADS / SUMB;
    for (int h0 = 0; h0 < PLE_HC; h0 += PER_ROUND) {
        const int h = h0 + tid / SUMB;
        const int col = tid % SUMB;
        const bool on = h < PLE_HC;
        float sum = 0.0f;
        if (on) {
            float sk_h = 0.0f, sq_h = 0.0f;
#pragma unroll
            for (int i = 0; i < PLE_HC; i++) {
                if (i == h) {
                    sk_h = scale_k[i];
                    sq_h = scale_q[i];
                }
            }
            const float * kr = key + h * e;
            const float * qr = res + h * e;
            const float * gkr = p.gk + h * e;
            const float * gqr = p.gq + h * e;
            float sum_temp[SUM_UNROLL] = {0.0f};
            for (int i = col; i < e;) {
                float temp[SUM_UNROLL];
#pragma unroll
                for (int j = 0; j < SUM_UNROLL; j++) {
                    if (i < e) {
                        const float kn = __fmul_rn(__fmul_rn(sk_h, kr[i]), gkr[i]);
                        const float qn = __fmul_rn(__fmul_rn(sq_h, qr[i]), gqr[i]);
                        temp[j] = __fmul_rn(kn, qn);
                    } else {
                        temp[j] = 0.0f;
                    }
                    i += SUMB;
                }
#pragma unroll
                for (int j = 0; j < SUM_UNROLL; j++) {
                    sum_temp[j] = __fadd_rn(sum_temp[j], temp[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < SUM_UNROLL; j++) {
                sum = __fadd_rn(sum, sum_temp[j]);
            }
        }
        sum = warp_sum(sum);
        if (on && lane == 0) {
            dred[h][col / 32] = sum;
        }
        __syncthreads();
        if (on && col < 32) {
            float v = lane < SUMB / 32 ? dred[h][lane] : 0.0f;
            v = warp_sum(v);
            if (col == 0) {
                // scale (scale * x + bias as one FMA, bias 0), abs, clamp, sqrt, sgn, mul,
                // sigmoid: each as ggml-cuda's kernel computes it.
                const float sc = __fmaf_rn(p.inv_sqrt_e, v, 0.0f);
                const float mag = sqrtf(fminf(fmaxf(fabsf(sc), p.clamp_min), INFINITY));
                const float sg = sc > 0.f ? 1.f : (sc < 0.f ? -1.f : 0.f);
                const float x = __fmul_rn(sg, mag);
                const float g = 1.0f / (1.0f + expf(-x));
                s_gate[h] = g;
                p.gate[t * PLE_HC + h] = g;
            }
        }
        __syncthreads();
    }

    // The value times each stream's gate, rms_norm times the conv's gamma.
    float ss[PLE_HC];
#pragma unroll
    for (int h = 0; h < PLE_HC; h++) {
        const float g = s_gate[h];
        float acc = 0.0f;
        for (int col = tid; col < e; col += PLE_THREADS) {
            const float gv = __fmul_rn(val[col], g);
            acc = __fmaf_rn(gv, gv, acc);
        }
        ss[h] = acc;
    }
    block_sums(ss, red);
    float * xn = p.xn + t * (int64_t) e * PLE_HC;
#pragma unroll
    for (int h = 0; h < PLE_HC; h++) {
        const float g = s_gate[h];
        const float sc = rms_scale(ss[h], e, p.eps);
        for (int col = tid; col < e; col += PLE_THREADS) {
            const float gv = __fmul_rn(val[col], g);
            xn[h * e + col] = __fmul_rn(__fmul_rn(sc, gv), p.gc[h * e + col]);
        }
    }
}

// Input j of channel c of sequence s in [history | tokens] order.
__device__ __forceinline__ float conv_input(const legion_ple_conv & p, int j, int c, int s) {
    return j < p.hist ? p.state[j + c * p.st_ld + s * p.st_sd]
                      : p.x[c + (int64_t) (j - p.hist) * p.x_ld + s * p.x_sd];
}

// One thread per (channel, token, sequence); reads and writes coalesced over the channels.
__global__ void ple_conv_kernel(legion_ple_conv p) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    if (c >= p.width) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < p.kern; k++) {
        const int j = p.hist - (p.kern - 1 - k) * p.dil + t;
        const float term = __fmul_rn(conv_input(p, j, c, s), p.w[k + c * p.kern]);
        acc = k == 0 ? term : __fadd_rn(acc, term);
    }
    // ggml-cuda's silu (unary.cuh), then the two adds, each rounded on its own.
    const float silu = acc / (1.0f + expf(-acc));
    const int64_t i = t + (int64_t) s * p.nt;
    const int h = c / p.e;
    const float gated = __fmul_rn(p.value[i * p.value_ld + (c - h * p.e)], p.gate[i * p.hc + h]);
    p.out[c + i * p.width] = __fadd_rn(p.res[c + i * p.res_ld], __fadd_rn(gated, silu));
}

// The next step's history: the last `hist` inputs of [history | tokens]. One thread per (channel,
// sequence), oldest first: entry j reads entry nt + j (> j) before it is overwritten.
__global__ void ple_keep_kernel(legion_ple_conv p) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= p.width) {
        return;
    }
    for (int j = 0; j < p.hist; j++) {
        const float v = conv_input(p, p.nt + j, c, s);
        p.state[j + c * p.st_ld + s * p.st_sd] = v;
    }
}

}  // namespace

int legion_ple_gate_launch(const legion_ple_gate * p, void * stream) {
    if (p->hc != PLE_HC || p->e < PLE_THREADS || p->n_tok < 1) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    // sum_rows_f32_cuda's block: 512 threads when there are fewer than two rows per SM, else
    // 128 (rows of 1024 or more).
    int dev = 0, nsm = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    const int nrows = p->hc * p->n_tok;
    if (nrows / nsm < 2) {
        ple_gate_kernel<512><<<p->n_tok, PLE_THREADS, 0, st>>>(*p);
    } else {
        ple_gate_kernel<128><<<p->n_tok, PLE_THREADS, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}

int legion_ple_conv_launch(const legion_ple_conv * p, void * stream) {
    if (p->nt < 1 || p->ns < 1 || p->kern < 1 || p->nt > 65535 || p->ns > 65535 ||
        p->width != p->e * p->hc) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    constexpr int threads = 256;
    const unsigned bx = (unsigned) ((p->width + threads - 1) / threads);
    ple_conv_kernel<<<dim3(bx, p->nt, p->ns), threads, 0, st>>>(*p);
    if (p->hist > 0) {
        ple_keep_kernel<<<dim3(bx, p->ns), threads, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}
