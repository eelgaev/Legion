// GLM-5.3-Flash's own ops on CUDA: every op of src/model/glm5next/ops.rs (the graph builders,
// the shapes and the CPU references are there). One launcher takes any of them by id with the
// node's tensors described generically (ggml order, dim 0 fastest; strides in bytes).
//
// - linear: y = scale * ws * (W x) for Q8_0, BF16, F16 or F32 weights, or Project Maya's Q6_K and
//   Q4_K (glm5next_kquant.cuh). Up to 8 tokens a GEMV
//   (a warp per row, 32-value blocks per lane, F32 sums); more a GEMM on tensor cores through
//   the portable WMMA API: 64 x 64 tiles (rows x tokens), K steps of 32, W dequantized and x
//   rounded to F16 in shared memory. Volta's tensor cores truncate long accumulation chains, so
//   each 16-deep product goes into a fresh fragment that is added into the F32 sums.
//   On the GB10 (#567) steps of 256+ tokens take 128 x 128 tiles of the same warp pieces
//   (linear_mm_wide: the same bits). BF16/F16 weights there take linear_tc (#575): raw mma.sync
//   m16n8k16 on 128 x 128 tiles with a 3-stage cp.async pipeline, the same F16 roundings and
//   fresh-from-zero 16-deep products added in K order (the same bits again), sm_80+ only.
//   On sm_80+ (#571), Q6_K weights of 1024+ rows over 256+ tokens take linear_mm_mma: 128 x
//   256 tiles, the raw Q6_K blocks copied to shared memory by
//   cp.async a stage ahead, ldmatrix and mma.sync m16n8k16 with a zero accumulator added into
//   the F32 sums (the same bits as the WMMA fresh fragments). Given linear_h's `xh` (x already
//   rounded to F16 by the f16 op, shared by the products of one input), it copies the token
//   tiles by cp.async instead of rounding them itself (~15% faster, the same halves).
//   Steps of 1024+ tokens (#594) instead copy the Q6_K weights to F16 (wf16: the values
//   linear_mm_mma rounds to in shared memory) and take linear_tc, which then reads `xh` too.
//   On the GB10 (#862) GLM's MLA products hand each other F16 rows there: attn_q_b and
//   attn_v_b store F16 outputs (linear_tc's OH, `__float2half_rn` of the F32 value) that
//   attn_k_b and attn_output read as their x (F16 rows, the rounding they would make): the
//   same bits, half the bytes in between and no f16 copy.
// - kda_scan: a block per (head, sequence), 8 threads per state row (16 keys each, in
//   registers), the tokens in order. On the GB10 (#571) steps of 64+ tokens without a stored
//   state take kda_scan_rows (64 rows a block, the same per-row order) and kda_conv splits into
//   a per-token kernel and a state kernel.
// - mla_attn: a block per (token, 4 heads); the selected latents staged 32 rows at a time in
//   shared memory, online softmax. On the GB10 (#571) steps of 64+ tokens take mla_tiled
//   (16 heads a block, a warp's lanes over the positions; the same chunks and order), and on
//   compute capability 12.1 (#862) mla_gb (sm80/mla_gb10.cuh: its bits, with 4 score warps and 4
//   value warps working a chunk apart, ~3x fewer shared-memory wavefronts).
//   Opt-in only (`--glm-mla-tc`, #575, not the default bits): sm_80+ steps of 64+ tokens take
//   mla_tc, a block per (token, 32 heads) with q and the softmax weights rounded to F16 and
//   the scores and the weighted latents on mma.sync m16n8k16 (~3.4x mla_tiled on the GB10).
// - The rest are row or element kernels; norms sum in double as the references do.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstring>

#include "device_cache.cuh"
#include "launch_pdl.cuh"

extern "C" {

struct legion_glm_args {
    void * dst;
    int64_t ne[4], nb[4];
    const void * src[10];  // null past the node's sources
    int64_t src_ne[10][4], src_nb[10][4];
    int32_t src_type[10];
    int32_t n_src;
    int32_t params[8];  // the node's external params
};

// Enqueues op `op` (GlmOp in src/model/glm5next/ops/cuda.rs) on `stream`. 0 on success, 1 for
// a shape or type the kernels do not take, else a CUDA error code.
int legion_glm_launch(int op, const legion_glm_args * a, void * stream);
}

namespace {

enum Op {
    EMBED, LINEAR, BF16, RMS_NORM, LAYER_NORM, HC_PRE, HC_POST, HC_HEAD, KDA_CONV, KDA_GATE,
    SIGMOID, KDA_SCAN, GATED_RMS, GLU, ROUTER_IDS, ROUTER_W, WEIGHTED_SUM, CACHE_ROWS,
    POOL_KEYS, MLA_ATTN, JOIN_COLS, ARGMAX, PMAX, NEXTN_IN, GET_COLS, ADD, F16, WF16, LINEAR_SET,
    GLU_MV,
};

// GGML_TYPE_*
constexpr int T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_BF16 = 30;
constexpr int T_Q4_K = 12, T_Q6_K = 14;  // Project Maya's dense weights (glm5next_kquant.cuh)
constexpr int T_Q5_K = 13;                // DeepSeek-V4.1-Flash's dense weights (the same header)

using A = legion_glm_args;

// Programmatic dependent launch (#594): on the GB10 every kernel here is launched through
// launch_pdl (launch_pdl.cuh, shared with qwen4_hc.cu), so the GPU sets a kernel up while the
// previous one in the stream is still running (~2.4 us of idle per kernel boundary otherwise,
// ~1700 boundaries per Maya-S decode token), and each kernel's first act is pdl_wait: it waits for
// the previous kernel to complete and its writes to be visible, as a plain launch does. No kernel
// triggers its dependents early, so nothing runs in a different order: the same bits. Other GPUs
// take plain launches, and below sm_90 pdl_wait compiles to nothing.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

float host_f32(int32_t v) {
    float f;
    memcpy(&f, &v, 4);
    return f;
}

__device__ __forceinline__ const char * S(const A & a, int i) { return (const char *) a.src[i]; }
__device__ __forceinline__ float pf(const A & a, int i) { return __int_as_float(a.params[i]); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }

// The byte stride between the sequences of source i, whose tokens are dim k: [.., n_tok, n_seqs]
// views step by nb[k + 1], token-major [.., n] ones by n_tok tokens (seq_stride in ops.rs).
__device__ __forceinline__ int64_t seq_nb(const A & a, int i, int k, int64_t n_tok, int64_t n_seqs) {
    return k < 3 && a.src_ne[i][k] == n_tok && a.src_ne[i][k + 1] == n_seqs ? a.src_nb[i][k + 1]
                                                                            : n_tok * a.src_nb[i][k];
}

template <typename T> __device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// The sum of every thread's v (blockDim.x a multiple of 32, at most 1024).
__device__ double block_sum(double v) {
    __shared__ double part[32];
    v = warp_sum(v);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        part[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int) blockDim.x / 32 ? part[lane] : 0.0;
        v = warp_sum(v);
        if (lane == 0) {
            part[0] = v;
        }
    }
    __syncthreads();
    v = part[0];
    __syncthreads();
    return v;
}

// Q6_K and Q4_K rows (glm5next_kquant.cuh).
__device__ __forceinline__ float kq_val(const char * row, int type, int64_t i);
__device__ __forceinline__ void kq_piece(const char * row, int type, int64_t blk, int h, float * v);
// Q5_K rows (the same header), only in the generic kernels' K5 instantiations: those run for
// Q5_K weights alone (#584), and the K5 = false ones compile to the code they had before.
__device__ __forceinline__ float q5k_val(const char * row, int64_t i);
__device__ __forceinline__ void q5k_piece(const char * row, int64_t blk, int h, float * v);

// Value i of a weight row of type `type` (Q5_K when K5).
template <bool K5 = false> __device__ __forceinline__ float wval(const char * row, int type, int64_t i) {
    if constexpr (K5) {
        return q5k_val(row, i);
    }
    switch (type) {
        case T_F32: return ((const float *) row)[i];
        case T_F16: return __half2float(((const __half *) row)[i]);
        case T_BF16: return bf16f(((const uint16_t *) row)[i]);
        case T_Q4_K:
        case T_Q6_K: return kq_val(row, type, i);
        default: {
            const char * b = row + (i / 32) * 34;
            return __half2float(*(const __half *) b) * (float) (int8_t) b[2 + i % 32];
        }
    }
}

// Values 16 h .. 16 h + 15 of the 32-value block `blk` of a weight row (Q5_K when K5).
template <bool K5 = false>
__device__ __forceinline__ void wpiece(const char * row, int type, int64_t blk, int h, float * v) {
    if constexpr (K5) {
        q5k_piece(row, blk, h, v);
        return;
    }
    if (type == T_Q8_0) {
        const char * b = row + blk * 34;
        const float d = __half2float(*(const __half *) b);
        const uint16_t * q = (const uint16_t *) (b + 2 + 16 * h);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint16_t w = q[i];
            v[2 * i] = (float) (int8_t) (w & 0xFF) * d;
            v[2 * i + 1] = (float) (int8_t) (w >> 8) * d;
        }
    } else if (type == T_Q6_K || type == T_Q4_K) {
        kq_piece(row, type, blk, h, v);
    } else if (type == T_F32) {
        const float4 * p = (const float4 *) (row + (blk * 32 + 16 * h) * 4);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 f = p[i];
            v[4 * i] = f.x;
            v[4 * i + 1] = f.y;
            v[4 * i + 2] = f.z;
            v[4 * i + 3] = f.w;
        }
    } else {
        const uint4 * p = (const uint4 *) (row + (blk * 32 + 16 * h) * 2);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 u = p[i];
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (type == T_F16) {
                    const float2 f = __half22float2(*(const __half2 *) &w[j]);
                    v[8 * i + 2 * j] = f.x;
                    v[8 * i + 2 * j + 1] = f.y;
                } else {
                    v[8 * i + 2 * j] = bf16f((uint16_t) (w[j] & 0xFFFF));
                    v[8 * i + 2 * j + 1] = bf16f((uint16_t) (w[j] >> 16));
                }
            }
        }
    }
}

// ---- linear ----------------------------------------------------------------------------------

struct Lin {
    const char * w;
    int64_t w_nb1, w_nb2;
    int wtype;
    const char * x;
    int64_t x_nbi, x_nbt;  // input row of (matrix i, token t): i * x_nbi + t * x_nbt
    float * dst;
    const float * ws;
    int64_t ws_n;
    float scale;
    int64_t k, m, h, n;
};

__device__ __forceinline__ float lin_scale(const Lin & p, int64_t r) {
    return p.ws ? (p.ws_n > 1 ? p.ws[r] : p.ws[0]) * p.scale : p.scale;
}

#include "glm5next_kquant.cuh"

template <int N, bool K5 = false>
__global__ void __launch_bounds__(128) linear_mv(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const char * row = p.w + r * p.w_nb1 + i * p.w_nb2;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    for (int64_t b = lane; b < p.k / 32; b += 32) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            float v[16];
            wpiece<K5>(row, p.wtype, b, h, v);
#pragma unroll
            for (int t = 0; t < N; t++) {
                if (t < p.n) {
                    const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 32 + 16 * h;
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        s += v[j] * x[j];
                    }
                    acc[t] += s;
                }
            }
        }
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// Blocks of a row a lane of linear_mv_f32 loads before it sums them (32 float4s).
constexpr int LIN_F32_U = 4;

// linear_mv for F32 weights (#606, the routers: 4096 x 288 a MoE layer): each lane sums the same
// blocks in the same order as linear_mv's lane (a 16-element FMA chain a half block, added to its
// sum), but loads LIN_F32_U of its blocks at once instead of one, which left the row's warp waiting
// on one 128-byte load at a time (~18 us a router on a V100). The same bits.
template <int N>
__global__ void __launch_bounds__(128) linear_mv_f32(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const float4 * row = (const float4 *) (p.w + r * p.w_nb1 + i * p.w_nb2);
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 32;
    for (int64_t b0 = lane; b0 < nb; b0 += 32 * LIN_F32_U) {
        float4 w[LIN_F32_U][8];
#pragma unroll
        for (int u = 0; u < LIN_F32_U; u++) {
            const int64_t b = b0 + 32 * u;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                w[u][q] = b < nb ? __ldg(row + b * 8 + q) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int u = 0; u < LIN_F32_U; u++) {
            const int64_t b = b0 + 32 * u;
            if (b >= nb) {
                break;
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const float v[16] = {w[u][4 * h].x,     w[u][4 * h].y,     w[u][4 * h].z,     w[u][4 * h].w,
                                     w[u][4 * h + 1].x, w[u][4 * h + 1].y, w[u][4 * h + 1].z, w[u][4 * h + 1].w,
                                     w[u][4 * h + 2].x, w[u][4 * h + 2].y, w[u][4 * h + 2].z, w[u][4 * h + 2].w,
                                     w[u][4 * h + 3].x, w[u][4 * h + 3].y, w[u][4 * h + 3].z, w[u][4 * h + 3].w};
#pragma unroll
                for (int t = 0; t < N; t++) {
                    if (t < p.n) {
                        const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 32 + 16 * h;
                        float s = 0.0f;
#pragma unroll
                        for (int j = 0; j < 16; j++) {
                            s += v[j] * x[j];
                        }
                        acc[t] += s;
                    }
                }
            }
        }
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// linear_mv_f32 with a block per row (#831): lane L of linear_mv_f32's warp sums its blocks
// L + 32 q in order of q; here warp u of the block takes the blocks of q = u (mod 4), so a round of
// 128 blocks (4096 elements) has its 4 KB per warp loaded by the 4 warps at once. Each computes
// its half-block sums exactly as linear_mv_f32's lane does and leaves them in shared memory; warp
// 0 adds them in linear_mv_f32's order (q, then the half) and meets in the same warp_sum: the same
// bits. A router (288 rows) then takes 288 blocks on all SMs instead of 72 blocks on some.
template <int N>
__global__ void __launch_bounds__(128) linear_mv_f32_split(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = blockIdx.x, i = blockIdx.y;
    const float4 * row = (const float4 *) (p.w + r * p.w_nb1 + i * p.w_nb2);
    __shared__ float part[N][4][2][32];
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 32;
    for (int64_t b0 = lane; b0 - lane < nb; b0 += 128) {
        const int64_t b = b0 + 32 * warp;
        if (b < nb) {
            float4 w[8];
#pragma unroll
            for (int q = 0; q < 8; q++) {
                w[q] = __ldg(row + b * 8 + q);
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const float v[16] = {w[4 * h].x,     w[4 * h].y,     w[4 * h].z,     w[4 * h].w,
                                     w[4 * h + 1].x, w[4 * h + 1].y, w[4 * h + 1].z, w[4 * h + 1].w,
                                     w[4 * h + 2].x, w[4 * h + 2].y, w[4 * h + 2].z, w[4 * h + 2].w,
                                     w[4 * h + 3].x, w[4 * h + 3].y, w[4 * h + 3].z, w[4 * h + 3].w};
#pragma unroll
                for (int t = 0; t < N; t++) {
                    if (t < p.n) {
                        const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 32 + 16 * h;
                        float s = 0.0f;
#pragma unroll
                        for (int j = 0; j < 16; j++) {
                            s += v[j] * x[j];
                        }
                        part[t][warp][h][lane] = s;
                    }
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                if (b0 + 32 * u >= nb) {
                    break;
                }
#pragma unroll
                for (int h = 0; h < 2; h++) {
#pragma unroll
                    for (int t = 0; t < N; t++) {
                        if (t < p.n) {
                            acc[t] += part[t][u][h][lane];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (warp != 0) {
        return;
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// Rows (times matrices) up to which F32 weights take linear_mv_f32_split, a block a row; past
// that linear_mv_f32's warp a row fills the GPU on its own.
constexpr int64_t LIN_F32_SPLIT_ROWS = 4096;

// Q8_0 rows for a few tokens, read the way ggml's mmvq lays a warp over them: lane l takes the
// quarter l % 4 (8 quants) of block l / 4 + 8 j, so each load of a warp covers 8 consecutive
// blocks (272 bytes; 16-bit loads, blocks being 34 bytes), and the activations come in float4s
// that the other rows' warps find in L1. A warp per row; two blocks per lane in flight.
// linear_mv_q8_acc is a lane's part of row r of matrix i, before the warp's sum.
template <int N>
__device__ __forceinline__ void linear_mv_q8_acc(const Lin & p, int64_t r, int64_t i, int lane, float (&acc)[N]) {
    const char * row = p.w + r * p.w_nb1 + i * p.w_nb2;
    const int q = lane % 4;
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 32;
#pragma unroll 2
    for (int64_t b = lane / 4; b < nb; b += 8) {
        const char * blk = row + b * 34;
        const float d = __half2float(__ldg((const __half *) blk));
        const uint16_t * qs = (const uint16_t *) (blk + 2 + 8 * q);
        float v[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint16_t w = __ldg(qs + j);
            v[2 * j] = (float) (int8_t) (w & 0xFF);
            v[2 * j + 1] = (float) (int8_t) (w >> 8);
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            if (t < p.n) {
                const float4 * x =
                    (const float4 *) ((const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 32 + 8 * q);
                const float4 a = __ldg(x), c = __ldg(x + 1);
                const float s = v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * c.x + v[5] * c.y +
                                v[6] * c.z + v[7] * c.w;
                acc[t] += d * s;
            }
        }
    }
}

template <int N>
__global__ void __launch_bounds__(128) linear_mv_q8(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    float acc[N];
    linear_mv_q8_acc<N>(p, r, i, lane, acc);
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// Up to LS_MAX Q8_0 GEMVs of one matrix each and a few tokens in one launch (linear_set in
// ops.rs, #606): segment g's rows take blocks first[g] .. first[g + 1] - 1, each row as
// linear_mv_q8 does it (the same bits), its outputs then through the segment's epilogue:
// LS_SIGMOID sigmoid_kernel's, LS_KDA_GATE kda_gate_kernel's (its sum rounded alone, as the
// output that kernel reads, by __fadd_rn: not contracted with the product).
constexpr int LS_MAX = 3, LS_PLAIN = 0, LS_SIGMOID = 1, LS_KDA_GATE = 2;

struct LinSet {
    Lin p[LS_MAX];
    int64_t first[LS_MAX + 1];
    int epi[LS_MAX];
    int nseg;
    const float * dt;  // LS_KDA_GATE: ssm_dt.bias [m], ssm_a [m / head]
    const float * av;
    int64_t head;
    float lower;
};

template <int N>
__global__ void __launch_bounds__(128) linear_set_q8(const LinSet s) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = 0;
    while (g + 1 < s.nseg && (int64_t) blockIdx.x >= s.first[g + 1]) {
        g++;
    }
    const Lin & p = s.p[g];
    const int64_t r = ((int64_t) blockIdx.x - s.first[g]) * 4 + warp;
    if (r >= p.m) {
        return;
    }
    float acc[N];
    linear_mv_q8_acc<N>(p, r, 0, lane, acc);
    const float sc = lin_scale(p, r);
    const int epi = s.epi[g];
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float v = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            const float z = v * sc;
            float o = z;
            if (epi == LS_SIGMOID) {
                o = sigm(z);
            } else if (epi == LS_KDA_GATE) {
                o = sigm(-(__fadd_rn(z, s.dt[r]) * s.av[r / s.head])) * s.lower;
            }
            p.dst[t * p.m + r] = o;
        }
    }
}

constexpr int GM = 64, GN = 64, GK = 32, LDS = GK + 8, LDC = GN + 4;

template <bool K5 = false> __global__ void __launch_bounds__(128) linear_mm(const Lin p) {
    pdl_wait();
    using namespace nvcuda;
    __shared__ __align__(32) union {
        struct {
            __half w[GM * LDS];
            __half x[GN * LDS];
        } ab;
        float c[GM * LDC];
    } sm;
    const int tid = threadIdx.x, warp = tid / 32;
    const int64_t m0 = (int64_t) blockIdx.x * GM, n0 = (int64_t) blockIdx.y * GN, i = blockIdx.z;
    const int lr = tid / 2, lh = tid % 2;  // the row (and token) and half of a K step a thread loads
    const int wm = warp / 2, wn = warp % 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::fill_fragment(acc[a][b], 0.0f);
        }
    }
    const int64_t r = m0 + lr, t = n0 + lr;
    const char * wrow = p.w + r * p.w_nb1 + i * p.w_nb2;
    const float * xrow = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt);
    for (int64_t k0 = 0; k0 < p.k; k0 += GK) {
        float v[16];
        if (r < p.m) {
            wpiece<K5>(wrow, p.wtype, k0 / 32, lh, v);
        } else {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                v[j] = 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            *(__half2 *) &sm.ab.w[lr * LDS + 16 * lh + j] = __floats2half2_rn(v[j], v[j + 1]);
        }
        if (t < p.n) {
            const float * x = xrow + k0 + 16 * lh;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                v[j] = x[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                v[j] = 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            *(__half2 *) &sm.ab.x[lr * LDS + 16 * lh + j] = __floats2half2_rn(v[j], v[j + 1]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
            for (int a = 0; a < 2; a++) {
                wmma::load_matrix_sync(fa[a], sm.ab.w + (32 * wm + 16 * a) * LDS + kk, LDS);
                wmma::load_matrix_sync(fb[a], sm.ab.x + (32 * wn + 16 * a) * LDS + kk, LDS);
            }
#pragma unroll
            for (int a = 0; a < 2; a++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
                    wmma::fill_fragment(f, 0.0f);
                    wmma::mma_sync(f, fa[a], fb[b], f);
#pragma unroll
                    for (int e = 0; e < f.num_elements; e++) {
                        acc[a][b].x[e] += f.x[e];
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::store_matrix_sync(sm.c + (32 * wm + 16 * a) * LDC + 32 * wn + 16 * b, acc[a][b], LDC,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = tid; e < GM * GN; e += 128) {
        const int rr = e % GM, tt = e / GM;
        const int64_t row = m0 + rr, tok = n0 + tt;
        if (row < p.m && tok < p.n) {
            p.dst[(tok * p.h + i) * p.m + row] = sm.c[rr * LDC + tt] * lin_scale(p, row);
        }
    }
}

// linear_mm on 128 x 128 tiles (#567): 16 warps, each the 32 x 32 piece linear_mm's warps
// compute, with the same fragments, the same fresh 16-deep products added in the same order
// (the same bits); the tiles' next K step loaded while this one multiplies (two buffers, one
// barrier a step). Each weight tile serves twice the tokens and each activation tile twice the
// rows: on a GB10, Maya's Q6_K at 2K tokens ~11 -> ~20-28 TFLOPS. 16-byte aligned activations.
constexpr int WT = 128;

template <bool K5 = false> __global__ void __launch_bounds__(512) linear_mm_wide(const Lin p) {
    pdl_wait();
    using namespace nvcuda;
    __shared__ __align__(32) union {
        struct {
            __half w[2][WT * LDS];
            __half x[2][WT * LDS];
        } ab;
        float c[16][16 * 16];
    } sm;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int64_t m0 = (int64_t) blockIdx.x * WT, n0 = (int64_t) blockIdx.y * WT, i = blockIdx.z;
    const int wm = warp / 4, wn = warp % 4;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::fill_fragment(acc[a][b], 0.0f);
        }
    }
    // Thread `tid` loads half `tid % 2` of a K step of weight row (tid < 256) or token tid / 2.
    const bool wrow = tid < 2 * WT;
    const int lr = (tid % (2 * WT)) / 2, lh = tid % 2;
    const int64_t r = m0 + lr, t = n0 + lr;
    const char * w = p.w + r * p.w_nb1 + i * p.w_nb2;
    const float * xr = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt);
    // A K step's loads issued before the step before it multiplies (`fetch`), converted and
    // stored after (`store`): Q6_K weights as raw quants (the values of kq_piece), the other types
    // through wpiece at store time.
    const bool q6 = p.wtype == T_Q6_K;
    uint16_t ql[8], qh[8];
    float ds = 0.0f;
    float4 xv[4];
    auto fetch = [&](int64_t k0) {
        if (wrow && r < p.m && q6) {
            const int64_t blk = k0 / 32;
            const uint8_t * b = (const uint8_t *) w + (blk / 8) * 210;
            const int v0 = (int) (blk % 8) * 32 + 16 * lh, n = v0 / 128, q = v0 % 128 / 32;
            ds = __half2float(*(const __half *) (b + 208)) * (float) (int8_t) b[192 + 8 * n + 2 * q + lh];
            const uint16_t * l = (const uint16_t *) (b + 64 * n + 32 * (q & 1) + 16 * lh);
            const uint16_t * hh = (const uint16_t *) (b + 128 + 32 * n + 16 * lh);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                ql[j] = l[j];
                qh[j] = hh[j];
            }
        } else if (!wrow && t < p.n) {
            const float4 * x = (const float4 *) (xr + k0 + 16 * lh);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                xv[j] = x[j];
            }
        }
    };
    auto store = [&](int64_t k0, int buf) {
        float v[16];
        if (wrow && r < p.m && q6) {
            const int q = ((int) ((k0 / 32) % 8) * 32 + 16 * lh) % 128 / 32;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int lb = (ql[j / 2] >> (8 * (j % 2))) & 0xFF, hb = (qh[j / 2] >> (8 * (j % 2))) & 0xFF;
                const int x = ((lb >> (4 * (q >> 1))) & 0xF) | (((hb >> (2 * q)) & 3) << 4);
                v[j] = ds * (float) (x - 32);
            }
        } else if (wrow && r < p.m) {
            wpiece<K5>(w, p.wtype, k0 / 32, lh, v);
        } else if (!wrow && t < p.n) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                v[4 * j] = xv[j].x;
                v[4 * j + 1] = xv[j].y;
                v[4 * j + 2] = xv[j].z;
                v[4 * j + 3] = xv[j].w;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                v[j] = 0.0f;
            }
        }
        __half * d = (wrow ? sm.ab.w[buf] : sm.ab.x[buf]) + lr * LDS + 16 * lh;
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
            *(__half2 *) &d[j] = __floats2half2_rn(v[j], v[j + 1]);
        }
    };
    fetch(0);
    store(0, 0);
    __syncthreads();
    int buf = 0;
    for (int64_t k0 = 0; k0 < p.k; k0 += GK) {
        const bool more = k0 + GK < p.k;
        if (more) {
            fetch(k0 + GK);
        }
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
            for (int a = 0; a < 2; a++) {
                wmma::load_matrix_sync(fa[a], sm.ab.w[buf] + (32 * wm + 16 * a) * LDS + kk, LDS);
                wmma::load_matrix_sync(fb[a], sm.ab.x[buf] + (32 * wn + 16 * a) * LDS + kk, LDS);
            }
#pragma unroll
            for (int a = 0; a < 2; a++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
                    wmma::fill_fragment(f, 0.0f);
                    wmma::mma_sync(f, fa[a], fb[b], f);
#pragma unroll
                    for (int e = 0; e < f.num_elements; e++) {
                        acc[a][b].x[e] += f.x[e];
                    }
                }
            }
        }
        if (more) {
            store(k0 + GK, buf ^ 1);
        }
        __syncthreads();
        buf ^= 1;
    }
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
            wmma::store_matrix_sync(sm.c[warp], acc[a][b], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int64_t row = m0 + 32 * wm + 16 * a + e / 16, tok = n0 + 32 * wn + 16 * b + e % 16;
                if (row < p.m && tok < p.n) {
                    p.dst[(tok * p.h + i) * p.m + row] = sm.c[warp][e] * lin_scale(p, row);
                }
            }
            __syncwarp();
        }
    }
}

// linear_mm_wide's products for BF16 and F16 weights on raw mma.sync (#575): every 16-deep
// product is an m16n8k16 F16 x F16 -> F32 mma from zero, added into the F32 sums in K order, of
// the same F16 operands (weights through F32 rounded to F16, activations rounded to F16), so the
// same bits. 128 x 128 tiles, 8 warps of 64 rows x 32 tokens, K steps of 32 copied raw (BF16 and
// F32) by cp.async three steps deep and converted while loading the fragments. XH (#594): the
// activations come as F16 rows (linear_h's xh, `f16_kernel`'s rounding, the one the fragments
// would make), half the bytes, by ldmatrix. OH (#862): the outputs stored as F16
// (`__float2half_rn` of the F32 value, f16_kernel's rounding) for a next product that reads F16
// rows.
constexpr int TCM = 128, TCN = 128, TCK = 32, TCS = 3, TCLW = TCK + 8, TCLX = TCK + 8;
constexpr int TC_SMEM = TCS * (TCM * TCLW * 2 + TCN * TCLX * 4);
constexpr int TC_SMEM_XH = TCS * (TCM * TCLW * 2 + TCN * TCLW * 2);
// Row tiles per group (below): 8 on a GB10 ~45-48 TFLOPS at 4K tokens; 4 and 16 slower, 32 thrashes
// its L2.
constexpr int TCG = 8;

__device__ __forceinline__ void tc_cp16(void * s, const void * g, bool ok) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(ok ? 16 : 0));
}

// Two weight values (BF16 or F16, the lower k in the low half) as an F16 pair.
__device__ __forceinline__ uint32_t tc_h2(uint32_t w, bool f16) {
    if (f16) {
        return w;
    }
    const __half2 h = __floats2half2_rn(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u));
    return *(const uint32_t *) &h;
}

template <bool XH, bool OH = false> __global__ void __launch_bounds__(256) linear_tc(const Lin p) {
    pdl_wait();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) char tc_sm[];
    uint16_t * wsm = (uint16_t *) tc_sm;                        // [TCS][TCM][TCLW]
    float * xsm = (float *) (tc_sm + TCS * TCM * TCLW * 2);     // [TCS][TCN][TCLX]
    uint16_t * xhs = (uint16_t *) xsm;                          // XH: [TCS][TCN][TCLW]
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    // Tiles in groups of TCG row tiles, a group's token tiles in turn: the CTAs resident at once
    // share a few weight tiles and token tiles in L2 (grid.x: row tiles * token tiles).
    const int tm = (int) ((p.m + TCM - 1) / TCM), tn = (int) (gridDim.x / tm), per = TCG * tn;
    const int first = (int) blockIdx.x / per * TCG, gm = min(tm - first, TCG), in = (int) blockIdx.x % per;
    const int64_t m0 = (int64_t) (first + in % gm) * TCM, n0 = (int64_t) (in / gm) * TCN, i = blockIdx.z;
    const int wm = warp / 4, wn = warp % 4;
    const bool f16 = p.wtype == T_F16;
    const char * wb = p.w + i * p.w_nb2;
    const char * xb = p.x + i * p.x_nbi;
    auto load = [&](int64_t k0, int s) {
#pragma unroll
        for (int j = 0; j < 2; j++) {  // 128 rows x 4 16-byte pieces
            const int c = tid + 256 * j, row = c / 4, q = c % 4;
            const bool ok = m0 + row < p.m;
            tc_cp16(wsm + (s * TCM + row) * TCLW + 8 * q, wb + (ok ? m0 + row : 0) * p.w_nb1 + (k0 + 8 * q) * 2, ok);
        }
        if constexpr (XH) {
#pragma unroll
            for (int j = 0; j < 2; j++) {  // 128 tokens x 4 16-byte pieces of F16
                const int c = tid + 256 * j, tok = c / 4, q = c % 4;
                const bool ok = n0 + tok < p.n;
                tc_cp16(xhs + (s * TCN + tok) * TCLW + 8 * q, xb + (ok ? n0 + tok : 0) * p.x_nbt + (k0 + 8 * q) * 2,
                        ok);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {  // 128 tokens x 8 16-byte pieces
                const int c = tid + 256 * j, tok = c / 8, q = c % 8;
                const bool ok = n0 + tok < p.n;
                tc_cp16(xsm + (s * TCN + tok) * TCLX + 4 * q, xb + (ok ? n0 + tok : 0) * p.x_nbt + (k0 + 4 * q) * 4,
                        ok);
            }
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; a++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[a][b][e] = 0.0f;
            }
        }
    }
    const int nk = (int) (p.k / TCK);
#pragma unroll
    for (int s = 0; s < TCS - 1; s++) {
        if (s < nk) {
            load((int64_t) s * TCK, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kt = 0; kt < nk; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(TCS - 2));
        __syncthreads();
        if (kt + TCS - 1 < nk) {
            load((int64_t) (kt + TCS - 1) * TCK, (kt + TCS - 1) % TCS);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        const int s = kt % TCS;
        const uint16_t * ws = wsm + s * TCM * TCLW;
        const float * xs = xsm + s * TCN * TCLX;
#pragma unroll
        for (int kk = 0; kk < TCK; kk += 16) {
            uint32_t fa[4][4], fb[4][2];
#pragma unroll
            for (int a = 0; a < 4; a++) {
                const uint16_t * pa = ws + (64 * wm + 16 * a + lane % 16) * TCLW + kk + 8 * (lane / 16);
                const uint32_t sa = (uint32_t) __cvta_generic_to_shared(pa);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(fa[a][0]), "=r"(fa[a][1]), "=r"(fa[a][2]), "=r"(fa[a][3])
                             : "r"(sa));
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    fa[a][j] = tc_h2(fa[a][j], f16);
                }
            }
            if constexpr (XH) {
                // Two token octets per ldmatrix x4: lanes 0-15 the first (k, k+8), 16-31 the second.
#pragma unroll
                for (int b = 0; b < 4; b += 2) {
                    const uint16_t * px = xhs + s * TCN * TCLW +
                                          (32 * wn + 8 * b + 8 * (lane >> 4) + (lane & 7)) * TCLW + kk +
                                          8 * ((lane >> 3) & 1);
                    const uint32_t sx = (uint32_t) __cvta_generic_to_shared(px);
                    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(fb[b][0]), "=r"(fb[b][1]), "=r"(fb[b + 1][0]), "=r"(fb[b + 1][1])
                        : "r"(sx));
                }
            } else {
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const float * px = xs + (32 * wn + 8 * b + lane / 4) * TCLX + kk + 2 * (lane % 4);
                    const float2 x0 = *(const float2 *) px, x1 = *(const float2 *) (px + 8);
                    const __half2 h0 = __floats2half2_rn(x0.x, x0.y), h1 = __floats2half2_rn(x1.x, x1.y);
                    fb[b][0] = *(const uint32_t *) &h0;
                    fb[b][1] = *(const uint32_t *) &h1;
                }
            }
#pragma unroll
            for (int a = 0; a < 4; a++) {
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    float d[4];
                    asm(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                        "{%8, %9}, {%10, %10, %10, %10};\n"
                        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                        : "r"(fa[a][0]), "r"(fa[a][1]), "r"(fa[a][2]), "r"(fa[a][3]), "r"(fb[b][0]), "r"(fb[b][1]),
                          "f"(0.0f));
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        acc[a][b][e] += d[e];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int a = 0; a < 4; a++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t row = m0 + 64 * wm + 16 * a + lane / 4 + 8 * h;
            if (row >= p.m) {
                continue;
            }
            const float sc = lin_scale(p, row);
#pragma unroll
            for (int b = 0; b < 4; b++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int64_t tok = n0 + 32 * wn + 8 * b + 2 * (lane % 4) + e;
                    if (tok < p.n) {
                        if constexpr (OH) {
                            ((__half *) p.dst)[(tok * p.h + i) * p.m + row] =
                                __float2half_rn(acc[a][b][2 * h + e] * sc);
                        } else {
                            p.dst[(tok * p.h + i) * p.m + row] = acc[a][b][2 * h + e] * sc;
                        }
                    }
                }
            }
        }
    }
#else
    __trap();  // tc_device(): compute capability 8.0+ only
#endif
}

// Whether the current device runs linear_tc (sm_80+: mma.sync m16n8k16).
bool tc_device() {
    int dev = 0, major = 0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8;
}

#include "sm80/linear_tc_gb10.cuh"

// linear_mm_wide's products for Q6_K weights on mma.sync (#571), sm_80+: a block of 8 warps per
// 128 rows x 256 tokens, each warp 64 x 64 (4 x 8 m16n8k16 tiles) from ldmatrix fragments, so a
// weight tile serves twice the tokens and every fragment load feeds 4-8 products (the WMMA
// kernel: 2). Each 16-deep product starts from zero and is added to the F32 sums in K order, as
// the WMMA kernels do (wmma's 16x16x16 step is two m16n8k16 products): the same bits.
// The loads are what limited linear_mm_wide on the GB10 (each of its warp loads touched 16
// rows): here the weights come in raw, half a Q6_K super-block of every row at a time (its
// ql, qh, scale and d bytes, 4-byte copies from the aligned word before each, consecutive lanes
// on consecutive words) through a 2-stage cp.async ring and are dequantized from shared memory
// with linear_mm_wide's expressions; the activations come 8 lanes per token row.
constexpr int XM = 128, XN = 256, XT = 256, XG = 8;
constexpr int XR_W = 30;                 // words of a row's half super-block slot: 17 ql, 9 qh, 3 sc, 1 d
constexpr int XR = XM * XR_W * 4;        // one raw stage (K = 128)
constexpr int XV = XN * (GK / 4) / XT;   // activation float4s per thread and K step

constexpr int xmm_smem() {
    return 2 * (XM + XN) * LDS * 2 + 2 * XR;
}

__device__ __forceinline__ void xcp4(void * smem, const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gmem));
#else
    __trap();
#endif
}
__device__ __forceinline__ void xcp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void xcp_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void xldsm4(uint32_t addr, uint32_t (&r)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#else
    __trap();
#endif
}

// c = a b, from zero (a fresh 16-deep product)
__device__ __forceinline__ void xmma0(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%10,%10,%10};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
#else
    __trap();
#endif
}

// 16 bytes, zeros past `n` bytes of the source (n 0 or 16)
__device__ __forceinline__ void xcp16(void * smem, const void * gmem, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gmem), "r"(n));
#else
    __trap();
#endif
}

// XH: the activations come as F16 rows (`f16_kernel`'s rounding, the one linear_mm_mma's store
// uses), copied to shared memory by cp.async.
template <bool XH> __global__ void __launch_bounds__(XT, 1) linear_mm_mma(const Lin p) {
    pdl_wait();
    extern __shared__ __align__(16) char xsm[];
    __half * sw = (__half *) xsm;      // [2][XM * LDS] weight rows, K-contiguous
    __half * sx = sw + 2 * XM * LDS;  // [2][XN * LDS] tokens, K-contiguous
    char * sr = (char *) (sx + 2 * XN * LDS);  // [2][XR] raw half super-blocks, XR_W words a row
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    // Blocks in groups of XG token tiles of one row tile, so the blocks that run together read
    // a row tile's weights from memory once (row tiles first: once per token tile).
    int64_t m0, n0;
    {
        const int nm = (int) gridDim.x, L = (int) (blockIdx.x + gridDim.x * blockIdx.y);
        const int g0 = L / (XG * nm) * XG, gs = min(XG, (int) gridDim.y - g0), wi = L % (XG * nm);
        n0 = (int64_t) (g0 + wi % gs) * XN;
        m0 = (int64_t) (wi / gs) * XM;
    }
    const int64_t i = blockIdx.z;
    const int wm = warp / 4, wn = warp % 4;
    float acc[4][8][4];
#pragma unroll
    for (int a = 0; a < 4; a++) {
#pragma unroll
        for (int b = 0; b < 8; b++) {
#pragma unroll
            for (int v = 0; v < 4; v++) {
                acc[a][b][v] = 0.0f;
            }
        }
    }
    // Raw stage S (K values 128 S ..): row rr's slot holds the 4-byte words around its ql (64
    // bytes), qh (32), scales (8) and d (2) of that half super-block, all at the same offset `sh`
    // (0 or 2) past their slot word, as 210-byte blocks are 2-byte aligned.
    const char * wbase = p.w + m0 * p.w_nb1 + i * p.w_nb2;
    auto issue_raw = [&](int S) {
        if ((int64_t) S * 128 < p.k) {
            char * dst = sr + (S & 1) * XR;
            const int64_t blk = S / 2, n = S % 2;
            for (int f = tid; f < XM * XR_W; f += XT) {
                const int rr = f / XR_W, c = f % XR_W;
                if (m0 + rr < p.m) {
                    const char * b = wbase + rr * p.w_nb1 + blk * 210;
                    const char * src = c < 17   ? b + 64 * n + 4 * c
                                       : c < 26 ? b + 128 + 32 * n + 4 * (c - 17)
                                       : c < 29 ? b + 192 + 8 * n + 4 * (c - 26)
                                                : b + 208;
                    xcp4(dst + 4 * f, (const void *) ((uintptr_t) src & ~(uintptr_t) 3));
                }
            }
        }
        xcp_commit();
    };
    // Thread `tid` dequantizes half `tid % 2` of a K step of weight row tid / 2, and loads float4
    // tid % 8 of the K step of tokens tid / 8 + 32 u.
    const int lr = tid / 2, lh = tid % 2;
    const int64_t r = m0 + lr;
    const int xc = tid % 8, xt = tid / 8;
    float4 xv[XV];
    auto fetch = [&](int64_t k0, int fb) {
        if constexpr (XH) {
#pragma unroll
            for (int u = 0; u < XN * GK / 8 / XT; u++) {
                const int c = tid + XT * u, tl = c / (GK / 8), part = c % (GK / 8);
                const int64_t t = n0 + tl;
                const char * src = p.x + i * p.x_nbi + (t < p.n ? t : 0) * p.x_nbt + 2 * k0 + 16 * part;
                xcp16(sx + fb * XN * LDS + tl * LDS + 8 * part, src, t < p.n ? 16 : 0);
            }
            xcp_commit();
            return;
        }
#pragma unroll
        for (int u = 0; u < XV; u++) {
            const int64_t t = n0 + xt + 32 * u;
            xv[u] = t < p.n ? ((const float4 *) (p.x + i * p.x_nbi + t * p.x_nbt + 4 * k0))[xc]
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&](int64_t k0, int buf) {
        float v[16];
        if (r < p.m) {
            const int S = (int) (k0 / 128);
            const char * slot = sr + (S & 1) * XR + lr * XR_W * 4;
            const int sh = (int) (((uintptr_t) (wbase + lr * p.w_nb1) + (k0 / 256) * 210) & 3);
            const int q = ((int) ((k0 / 32) % 8) * 32 + 16 * lh) % 128 / 32;
            const float ds = __half2float(*(const __half *) (slot + 116 + sh)) *
                             (float) (int8_t) slot[104 + sh + 2 * q + lh];
            const uint16_t * l = (const uint16_t *) (slot + sh + 32 * (q & 1) + 16 * lh);
            const uint16_t * hh = (const uint16_t *) (slot + 68 + sh + 16 * lh);
            // 4 quants a word; (float) (x - 32) as 2^23 + x (its bits: 0x4B000000 | x) minus
            // 2^23 + 32, exact for x < 2^23: the same value
#pragma unroll
            for (int j4 = 0; j4 < 4; j4++) {
                const uint32_t lw = (uint32_t) l[2 * j4] | (uint32_t) l[2 * j4 + 1] << 16;
                const uint32_t hw = (uint32_t) hh[2 * j4] | (uint32_t) hh[2 * j4 + 1] << 16;
                const uint32_t xw = ((lw >> (4 * (q >> 1))) & 0x0F0F0F0Fu) | (((hw >> (2 * q)) & 0x03030303u) << 4);
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const float xf = __uint_as_float(__byte_perm(xw, 0x4B000000u, 0x7540u + b)) - 8388640.0f;
                    v[4 * j4 + b] = ds * xf;
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; j++) {
                v[j] = 0.0f;
            }
        }
        uint32_t u[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const __half2 hv = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
            u[j] = *(const uint32_t *) &hv;
        }
        __half * d = sw + buf * XM * LDS + lr * LDS + 16 * lh;
        ((uint4 *) d)[0] = make_uint4(u[0], u[1], u[2], u[3]);
        ((uint4 *) d)[1] = make_uint4(u[4], u[5], u[6], u[7]);
        if constexpr (XH) {
            return;
        }
#pragma unroll
        for (int t = 0; t < XV; t++) {
            const __half2 h0 = __floats2half2_rn(xv[t].x, xv[t].y), h1 = __floats2half2_rn(xv[t].z, xv[t].w);
            *(uint2 *) (sx + buf * XN * LDS + (xt + 32 * t) * LDS + 4 * xc) =
                make_uint2(*(const uint32_t *) &h0, *(const uint32_t *) &h1);
        }
    };
    // ldmatrix lanes: A (weights) rows (l & 7) + 8 ((l >> 3) & 1), K half l >> 4; B (tokens) two
    // n-tiles per x4: token 8 (l >> 4) + (l & 7), K half (l >> 3) & 1
    const int ar = (lane & 7) + 8 * ((lane >> 3) & 1), ac = lane >> 4;
    const int br = 8 * (lane >> 4) + (lane & 7), bc = (lane >> 3) & 1;
    const uint32_t sw0 = (uint32_t) __cvta_generic_to_shared(sw), sx0 = (uint32_t) __cvta_generic_to_shared(sx);
    issue_raw(0);
    fetch(0, 0);
    xcp_wait_all();
    __syncthreads();
    store(0, 0);
    __syncthreads();
    int buf = 0;
    for (int64_t k0 = 0; k0 < p.k; k0 += GK) {
        const bool more = k0 + GK < p.k;
        // stage S + 1 into the buffer stage S - 1 used (its last K step was stored before the
        // barrier that ended the previous step)
        if (k0 % 128 == 0) {
            issue_raw((int) (k0 / 128) + 1);
        }
        if (more) {
            fetch(k0 + GK, buf ^ 1);
        }
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            uint32_t bf[8][2];
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t t4[4];
                xldsm4(sx0 + 2 * (buf * XN * LDS + (64 * wn + 8 * j + br) * LDS + kk + 8 * bc), t4);
                bf[j][0] = t4[0];
                bf[j][1] = t4[1];
                bf[j + 1][0] = t4[2];
                bf[j + 1][1] = t4[3];
            }
#pragma unroll
            for (int a = 0; a < 4; a++) {
                uint32_t af[4];
                xldsm4(sw0 + 2 * (buf * XM * LDS + (64 * wm + 16 * a + ar) * LDS + kk + 8 * ac), af);
#pragma unroll
                for (int b = 0; b < 8; b++) {
                    float f[4];
                    xmma0(f, af, bf[b][0], bf[b][1]);
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        acc[a][b][v] += f[v];
                    }
                }
            }
        }
        if (more) {
            if (XH) {
                // the next K step's activations (and stage S + 1) landed
                xcp_wait_all();
            }
            if ((k0 + GK) % 128 == 0) {
                // the next K step opens stage S + 1: every thread's copies of it landed
                xcp_wait_all();
                __syncthreads();
            }
            store(k0 + GK, buf ^ 1);
        }
        __syncthreads();
        buf ^= 1;
    }
    xcp_wait_all();
    const int g = lane >> 2, tg = lane & 3;
#pragma unroll
    for (int a = 0; a < 4; a++) {
#pragma unroll
        for (int v2 = 0; v2 < 2; v2++) {
            const int64_t row = m0 + 64 * wm + 16 * a + g + 8 * v2;
            if (row < p.m) {
                const float sc = lin_scale(p, row);
#pragma unroll
                for (int b = 0; b < 8; b++) {
#pragma unroll
                    for (int v1 = 0; v1 < 2; v1++) {
                        const int64_t tok = n0 + 64 * wn + 8 * b + 2 * tg + v1;
                        if (tok < p.n) {
                            p.dst[(tok * p.h + i) * p.m + row] = acc[a][b][2 * v2 + v1] * sc;
                        }
                    }
                }
            }
        }
    }
}

// `x` rounded to F16 (to nearest, as linear_mm_mma's store): linear's `xh` (glm5next/ops.rs f16)
__global__ void f16_kernel(const float * x, __half * out, int64_t n) {
    pdl_wait();
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn(x[i]);
    }
}

// Q6_K weights [k, m, h] (rows 2-byte aligned) as F16 (#594): the values linear_mm_mma and
// linear_mm_wide compute and round to F16 in shared memory (d * scale in F32, times q - 32,
// rounded to nearest), for linear_tc. A thread per 16 values (a scale's run), written as two
// 16-byte stores.
__global__ void wf16_q6k_kernel(const A a) {
    pdl_wait();
    const int64_t k = a.src_ne[0][0], m = a.src_ne[0][1];
    const int64_t gpr = k / 16, total = m * a.src_ne[0][2] * gpr;
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= total) {
        return;
    }
    const int64_t row = gi / gpr, g = gi % gpr;
    const uint8_t * b = (const uint8_t *) a.src[0] + (row % m) * a.src_nb[0][1] + (row / m) * a.src_nb[0][2] +
                        (g / 16) * 210;
    const int v0 = (int) (g % 16) * 16, n = v0 / 128, q = v0 % 128 / 32, lh = v0 % 32 / 16;
    const float ds = __half2float(*(const __half *) (b + 208)) * (float) (int8_t) b[192 + 8 * n + 2 * q + lh];
    uint32_t u[8];
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        float v[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int lb = b[64 * n + 32 * (q & 1) + 16 * lh + j + e], hb = b[128 + 32 * n + 16 * lh + j + e];
            const int x = ((lb >> (4 * (q >> 1))) & 0xF) | (((hb >> (2 * q)) & 3) << 4);
            v[e] = ds * (float) (x - 32);
        }
        const __half2 hv = __floats2half2_rn(v[0], v[1]);
        u[j / 2] = *(const uint32_t *) &hv;
    }
    uint4 * d = (uint4 *) ((__half *) a.dst + row * k + g * 16);
    d[0] = make_uint4(u[0], u[1], u[2], u[3]);
    d[1] = make_uint4(u[4], u[5], u[6], u[7]);
}

// Whether the current device has sm_80's mma.sync / ldmatrix (cached per device).
bool dev_sm80() {
    static int cc[64];
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) {
        return false;
    }
    if (cc[d] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d);
        cc[d] = major > 0 ? major : -1;
    }
    return cc[d] >= 8;
}

int launch_linear(const A & a, cudaStream_t st) {
    const int wt = a.src_type[0];
    const bool kq = wt == T_Q6_K || wt == T_Q4_K || wt == T_Q5_K;
    if (wt != T_F32 && wt != T_F16 && wt != T_BF16 && wt != T_Q8_0 && !kq) {
        return 1;
    }
    Lin p;
    p.w = (const char *) a.src[0];
    p.w_nb1 = a.src_nb[0][1];
    p.w_nb2 = a.src_nb[0][2];
    p.wtype = wt;
    p.k = a.src_ne[0][0];
    p.m = a.src_ne[0][1];
    p.h = a.src_ne[0][2];
    p.x = (const char *) a.src[1];
    const int64_t * xne = a.src_ne[1];
    const int64_t * xnb = a.src_nb[1];
    if (p.h == 1) {
        p.n = a.ne[1];
        p.x_nbi = 0;
        // a single column per token along dim 1, or along dim 2 when dim 1 is 1
        p.x_nbt = xne[1] == 1 && xne[2] > 1 ? xnb[2] : xnb[1];
    } else {
        p.n = a.ne[2];
        p.x_nbi = xnb[1];
        p.x_nbt = xnb[2];
    }
    p.dst = (float *) a.dst;
    p.scale = host_f32(a.params[0]);
    p.ws = a.params[1] ? (const float *) a.src[2] : nullptr;
    p.ws_n = a.params[1] ? a.src_ne[2][0] * a.src_ne[2][1] * a.src_ne[2][2] * a.src_ne[2][3] : 0;
    // Q8_0 and Q6_K blocks (34 and 210 bytes) are read in 16-bit words, the other types in 16-byte
    // vectors (Q4_K's 144-byte and Q5_K's 176-byte blocks too).
    const int64_t al = wt == T_Q8_0 || wt == T_Q6_K ? 2 : 16;
    if (p.k % (kq ? 256 : 32) != 0 || ((int64_t) ((uintptr_t) p.w | p.w_nb1 | p.w_nb2) % al) != 0 || p.h > 65535) {
        return 1;
    }
    if (p.n <= 0 || p.m <= 0) {
        return 0;
    }
    // The Q8_0 kernel's float4 activations need 16-byte aligned rows.
    const bool x16 = ((int64_t) ((uintptr_t) p.x | p.x_nbi | p.x_nbt) % 16) == 0;
    // linear_tc's alone (#862): x as F16 rows (an earlier product's F16 output, the rounding the
    // kernel would make) and, params[5], F16 outputs
    const bool hin = a.src_type[1] == T_F16, oh = a.params[5] != 0;
    if ((hin || oh) && p.n <= 8) {
        return 1;
    }
    if (p.n <= 8 && wt == T_Q8_0 && x16) {
        const dim3 grid((unsigned) ((p.m + 3) / 4), (unsigned) p.h);
        if (p.n == 1) {
            launch_pdl(linear_mv_q8<1>, grid, 128,0, st, p);
        } else if (p.n == 2) {
            launch_pdl(linear_mv_q8<2>, grid, 128,0, st, p);
        } else if (p.n <= 4) {
            launch_pdl(linear_mv_q8<4>, grid, 128,0, st, p);
        } else {
            launch_pdl(linear_mv_q8<8>, grid, 128,0, st, p);
        }
    } else if (p.n <= 8 && kq && x16) {
        const dim3 grid((unsigned) ((p.m + 3) / 4), (unsigned) p.h);
        // params[2]: the GB10 kernels (ops::set_gb10_linear): the staged Q6_K GEMV, for rows it takes
        const bool st16 = ((int64_t) ((uintptr_t) p.w | p.w_nb1 | p.w_nb2) % 16) == 0;
        if (wt == T_Q6_K && a.params[2] && st16 && p.k % (256 * KQ_STAGE) == 0) {
            if (p.n == 1) {
                launch_pdl(linear_mv_q6k_st<1>, grid, 128,0, st, p);
            } else if (p.n == 2) {
                launch_pdl(linear_mv_q6k_st<2>, grid, 128,0, st, p);
            } else if (p.n <= 4) {
                launch_pdl(linear_mv_q6k_st<4>, grid, 128,0, st, p);
            } else {
                launch_pdl(linear_mv_q6k_st<8>, grid, 128,0, st, p);
            }
        } else if (wt == T_Q6_K) {
            if (p.n == 1) {
                launch_pdl(linear_mv_q6k<1>, grid, 128,0, st, p);
            } else if (p.n == 2) {
                launch_pdl(linear_mv_q6k<2>, grid, 128,0, st, p);
            } else if (p.n <= 4) {
                launch_pdl(linear_mv_q6k<4>, grid, 128,0, st, p);
            } else {
                launch_pdl(linear_mv_q6k<8>, grid, 128,0, st, p);
            }
        } else if (wt == T_Q5_K) {
            if (p.n == 1) {
                launch_pdl(linear_mv_q5k<1>, grid, 128,0, st, p);
            } else if (p.n == 2) {
                launch_pdl(linear_mv_q5k<2>, grid, 128,0, st, p);
            } else if (p.n <= 4) {
                launch_pdl(linear_mv_q5k<4>, grid, 128,0, st, p);
            } else {
                launch_pdl(linear_mv_q5k<8>, grid, 128,0, st, p);
            }
        } else {
            if (p.n == 1) {
                launch_pdl(linear_mv_q4k<1>, grid, 128,0, st, p);
            } else if (p.n == 2) {
                launch_pdl(linear_mv_q4k<2>, grid, 128,0, st, p);
            } else if (p.n <= 4) {
                launch_pdl(linear_mv_q4k<4>, grid, 128,0, st, p);
            } else {
                launch_pdl(linear_mv_q4k<8>, grid, 128,0, st, p);
            }
        }
    } else if (p.n <= 8 && wt == T_Q5_K) {
        const dim3 grid((unsigned) ((p.m + 3) / 4), (unsigned) p.h);
        launch_pdl(linear_mv<8, true>, grid, 128,0, st, p);
    } else if (p.n <= 8 && wt == T_F32 && p.m * p.h <= LIN_F32_SPLIT_ROWS) {
        const dim3 grid((unsigned) p.m, (unsigned) p.h);
        if (p.n == 1) {
            launch_pdl(linear_mv_f32_split<1>, grid, 128,0, st, p);
        } else if (p.n == 2) {
            launch_pdl(linear_mv_f32_split<2>, grid, 128,0, st, p);
        } else if (p.n <= 4) {
            launch_pdl(linear_mv_f32_split<4>, grid, 128,0, st, p);
        } else {
            launch_pdl(linear_mv_f32_split<8>, grid, 128,0, st, p);
        }
    } else if (p.n <= 8 && wt == T_F32) {
        const dim3 grid((unsigned) ((p.m + 3) / 4), (unsigned) p.h);
        if (p.n == 1) {
            launch_pdl(linear_mv_f32<1>, grid, 128,0, st, p);
        } else if (p.n == 2) {
            launch_pdl(linear_mv_f32<2>, grid, 128,0, st, p);
        } else if (p.n <= 4) {
            launch_pdl(linear_mv_f32<4>, grid, 128,0, st, p);
        } else {
            launch_pdl(linear_mv_f32<8>, grid, 128,0, st, p);
        }
    } else if (p.n <= 8) {
        const dim3 grid((unsigned) ((p.m + 3) / 4), (unsigned) p.h);
        if (p.n == 1) {
            launch_pdl(linear_mv<1>, grid, 128,0, st, p);
        } else if (p.n == 2) {
            launch_pdl(linear_mv<2>, grid, 128,0, st, p);
        } else if (p.n <= 4) {
            launch_pdl(linear_mv<4>, grid, 128,0, st, p);
        } else {
            launch_pdl(linear_mv<8>, grid, 128,0, st, p);
        }
    } else {
        const dim3 grid((unsigned) ((p.m + GM - 1) / GM), (unsigned) ((p.n + GN - 1) / GN), (unsigned) p.h);
        if (grid.y > 65535) {
            return 1;
        }
        // params[2]: the GB10 kernels (ops::set_gb10_linear); the same bits
        // linear_mm_mma: Q6_K rows 4-byte aligned and an even count of super-blocks, so the
        // last block of a row starts 2 bytes past a word and its copies end in the row
        const bool xq6 = wt == T_Q6_K && ((int64_t) ((uintptr_t) p.w | p.w_nb1 | p.w_nb2) % 4) == 0 &&
                         (p.k / 256) % 2 == 0;
        const bool tc = (wt == T_BF16 || wt == T_F16) && a.params[2] && x16 && p.n >= 2 * WT && tc_device();
        // params[3]: x in F16 (linear_h's xh, the last source), rows 16-byte aligned
        const int xi = a.params[1] ? 3 : 2;
        const int64_t xh_nbt = a.n_src > xi && a.src_ne[xi][1] == 1 && a.src_ne[xi][2] > 1 ? a.src_nb[xi][2] :
                               a.n_src > xi                                                 ? a.src_nb[xi][1] :
                                                                                              0;
        const bool xh = hin || (a.params[3] && p.h == 1 && a.n_src > xi && a.src_type[xi] == T_F16 &&
                                ((int64_t) ((uintptr_t) a.src[xi] | xh_nbt) % 16) == 0);
        Lin ph = p;
        if (xh && !hin) {
            ph.x = (const char *) a.src[xi];
            ph.x_nbi = 0;
            ph.x_nbt = xh_nbt;
        }
        if ((hin || oh) && !tc) {
            return 1;
        }
        if (tc) {
            // On a GB10 ~45-48 TFLOPS for GLM's attention linears at 4K tokens, linear_mm_wide ~15-30.
            // (per launch: the attribute is per device)
            const cudaError_t e =
                oh ? (xh ? cudaFuncSetAttribute(linear_tc<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                TC_SMEM_XH) :
                           cudaFuncSetAttribute(linear_tc<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                TC_SMEM)) :
                xh ? cudaFuncSetAttribute(linear_tc<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM_XH) :
                     cudaFuncSetAttribute(linear_tc<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM);
            if (e != cudaSuccess) {
                return 1;
            }
            const int64_t tiles = ((p.m + TCM - 1) / TCM) * ((p.n + TCN - 1) / TCN);
            if (tiles > INT32_MAX) {
                return 1;
            }
            const dim3 tg((unsigned) tiles, 1, (unsigned) p.h);
            // params[4] (ops::LinTc, #862): 0 the device's choice, 1 linear_tc, 2 + v shape v of
            // linear_tc_gb (the same bits)
            const int kern = a.params[4];
            if (kern >= 2 && tc_gb_shape(kern - 2, xh, xh ? ph : p, st, oh)) {
                return 0;
            }
            if (kern == 0 && tc_gb_device() && tc_gb_auto(xh, xh ? ph : p, st, oh)) {
                return 0;
            }
            if (oh && xh) {
                launch_pdl(linear_tc<true, true>, tg, 256, TC_SMEM_XH, st, ph);
            } else if (oh) {
                launch_pdl(linear_tc<false, true>, tg, 256, TC_SMEM, st, p);
            } else if (xh) {
                launch_pdl(linear_tc<true>, tg, 256,TC_SMEM_XH, st, ph);
            } else {
                launch_pdl(linear_tc<false>, tg, 256,TC_SMEM, st, p);
            }
        } else if (a.params[2] && x16 && xq6 && p.n >= XN && p.m >= 8 * XM && dev_sm80()) {
            constexpr int smem = xmm_smem();
            const cudaError_t e = xh ? cudaFuncSetAttribute(linear_mm_mma<true>,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
                                     : cudaFuncSetAttribute(linear_mm_mma<false>,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) {
                return (int) e;
            }
            const dim3 xg((unsigned) ((p.m + XM - 1) / XM), (unsigned) ((p.n + XN - 1) / XN), (unsigned) p.h);
            if (xh) {
                launch_pdl(linear_mm_mma<true>, xg, XT,smem, st, ph);
            } else {
                launch_pdl(linear_mm_mma<false>, xg, XT,smem, st, p);
            }
        } else if (a.params[2] && x16 && p.n >= 2 * WT) {
            const dim3 wg((unsigned) ((p.m + WT - 1) / WT), (unsigned) ((p.n + WT - 1) / WT), (unsigned) p.h);
            if (wt == T_Q5_K) {
                launch_pdl(linear_mm_wide<true>, wg, 512,0, st, p);
            } else {
                launch_pdl(linear_mm_wide<false>, wg, 512,0, st, p);
            }
        } else if (wt == T_Q5_K) {
            launch_pdl(linear_mm<true>, grid, 128,0, st, p);
        } else {
            launch_pdl(linear_mm<false>, grid, 128,0, st, p);
        }
    }
    return 0;
}

// LINEAR_SET (linear_set in ops.rs): params [segments, each one's epilogue x3, kda_gate's lower,
// tokens]; sources (w, x) per segment, then ssm_dt.bias and ssm_a with an LS_KDA_GATE epilogue.
// Q8_0 matrices and up to 8 tokens; the outputs one after the other, [m, n] each.
int launch_linear_set(const A & a, cudaStream_t st) {
    LinSet s{};
    s.nseg = a.params[0];
    const int64_t n = a.params[5];
    if (s.nseg < 1 || s.nseg > LS_MAX || n < 1 || n > 8) {
        return 1;
    }
    int64_t first = 0, off = 0;
    bool gate = false;
    for (int g = 0; g < s.nseg; g++) {
        const int wi = 2 * g, xi = 2 * g + 1;
        Lin & p = s.p[g];
        p.w = (const char *) a.src[wi];
        p.w_nb1 = a.src_nb[wi][1];
        p.w_nb2 = 0;
        p.wtype = a.src_type[wi];
        p.k = a.src_ne[wi][0];
        p.m = a.src_ne[wi][1];
        p.h = 1;
        p.x = (const char *) a.src[xi];
        p.x_nbi = 0;
        p.x_nbt = a.src_ne[xi][1] == 1 && a.src_ne[xi][2] > 1 ? a.src_nb[xi][2] : a.src_nb[xi][1];
        p.n = n;
        p.dst = (float *) a.dst + off;
        p.ws = nullptr;
        p.ws_n = 0;
        p.scale = 1.0f;
        if (p.wtype != T_Q8_0 || a.src_ne[wi][2] != 1 || p.k % 32 != 0 || a.src_ne[xi][0] != p.k ||
            ((int64_t) ((uintptr_t) p.w | p.w_nb1) % 2) != 0 ||
            ((int64_t) ((uintptr_t) p.x | p.x_nbt) % 16) != 0) {
            return 1;
        }
        s.epi[g] = a.params[1 + g];
        gate |= s.epi[g] == LS_KDA_GATE;
        s.first[g] = first;
        first += (p.m + 3) / 4;
        off += p.m * n;
    }
    s.first[s.nseg] = first;
    if (gate) {
        const int di = 2 * s.nseg;
        if (a.n_src < di + 2 || a.src_ne[di + 1][0] < 1 || a.src_ne[di][0] % a.src_ne[di + 1][0] != 0) {
            return 1;
        }
        for (int g = 0; g < s.nseg; g++) {
            if (s.epi[g] == LS_KDA_GATE && s.p[g].m != a.src_ne[di][0]) {
                return 1;
            }
        }
        s.dt = (const float *) a.src[di];
        s.av = (const float *) a.src[di + 1];
        s.head = a.src_ne[di][0] / a.src_ne[di + 1][0];
        s.lower = host_f32(a.params[4]);
    }
    const unsigned grid = (unsigned) first;
    if (n == 1) {
        launch_pdl(linear_set_q8<1>, grid, 128, 0, st, s);
    } else if (n == 2) {
        launch_pdl(linear_set_q8<2>, grid, 128, 0, st, s);
    } else if (n <= 4) {
        launch_pdl(linear_set_q8<4>, grid, 128, 0, st, s);
    } else {
        launch_pdl(linear_set_q8<8>, grid, 128, 0, st, s);
    }
    return 0;
}

// ---- element and row kernels -----------------------------------------------------------------

template <bool K5 = false> __global__ void embed_kernel(const A a) {
    pdl_wait();
    const int64_t t = blockIdx.x, e = a.ne[0], hc = a.ne[1];
    const int32_t tok = ((const int32_t *) a.src[1])[t];
    const char * row = S(a, 0) + tok * a.src_nb[0][1];
    float * out = (float *) a.dst + t * hc * e;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        const float v = wval<K5>(row, a.src_type[0], i);
        for (int64_t s = 0; s < hc; s++) {
            out[s * e + i] = v;
        }
    }
}

__global__ void bf16_kernel(const float * x, uint16_t * out, int64_t n) {
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t u = __float_as_uint(x[i]);
        out[i] = (u & 0x7fffffffu) > 0x7f800000u ? (uint16_t) ((u >> 16) | 64)
                                                  : (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
    }
}

__global__ void rms_norm_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.ne[0], r = blockIdx.x;
    const float * x = (const float *) a.src[0] + r * d;
    const float * w = (const float *) a.src[1];
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        sum += (double) x[i] * x[i];
    }
    sum = block_sum(sum);
    const float s = 1.0f / sqrtf((float) (sum / d) + pf(a, 0));
    float * o = (float *) a.dst + r * d;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] = w ? x[i] * s * w[i] : x[i] * s;
    }
}

__global__ void layer_norm_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.src_ne[1][0], r = blockIdx.x;
    const float * x = (const float *) a.src[0] + r * d;
    const float * w = (const float *) a.src[1];
    const float * b = (const float *) a.src[2];
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        sum += x[i];
    }
    const float mean = (float) (block_sum(sum) / d);
    double var = 0.0;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        const double c = x[i] - mean;
        var += c * c;
    }
    var = block_sum(var);
    const float s = 1.0f / sqrtf((float) (var / d) + pf(a, 0));
    float * o = (float *) a.dst + r * d;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] = (x[i] - mean) * s * w[i] + b[i];
    }
}

constexpr int HC_MAX = 8;
// F32 mix-row elements a thread loads ahead in hc_mix_kernel.
constexpr int HC_MIX_U = 8;
// hc_mix_kernel's warps (its 256 threads) and hc_mix_split_kernel's elements loaded ahead: all 64
// of a thread's at GLM's size (4096 x 4 streams, one DRAM round trip instead of four, #831), else 16.
constexpr int HC_MIX_W = 8, HC_SPLIT_U = 16, HC_SPLIT_UW = 64;

// With the sublayer's norm fused (a 5th source, the norm weight; hc_pre_norm in ops.rs, #606) the
// node is [y | pre | post | comb | h | mix], h from a 32-element boundary, the mix rows from the
// next one: a (2 + hc) hc + 1 element row a token, or with params[5] (hc_mix_split_kernel's
// warps) 2 HC_MIX_W floats (the warps' sums of squares, doubles) then each row's HC_MIX_W warp
// sums. hc_finish_kernel writes y while it still reads them.
__device__ __forceinline__ int64_t hc_h_off(int64_t e, int64_t hc, int64_t n) {
    return (e * n + (2 * hc + hc * hc) * n + 31) / 32 * 32;
}

// Token t's hc_mix_kernel scratch: its y row, or its mix row with the norm fused.
__device__ __forceinline__ float * hc_mix_at(const A & a, int64_t t) {
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], n = a.src_ne[0][2];
    float * out = (float *) a.dst;
    if (a.n_src < 5) {
        return out + t * e;
    }
    const int64_t rows = (2 + hc) * hc, stride = a.params[5] ? HC_MIX_W * (rows + 2) : rows + 1;
    return out + (hc_h_off(e, hc, n) + e * n + 31) / 32 * 32 + t * stride;
}

// Mix row r of a token's scratch m, as hc_mix_kernel's thread 0 sums its warps' (the same bits).
__device__ __forceinline__ float hc_mixv(const A & a, const float * m, int64_t r) {
    if (!a.params[5] || a.n_src < 5) {
        return m[r];
    }
    const float * p = m + HC_MIX_W * (2 + r);
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < HC_MIX_W; w++) {
        v += p[w];
    }
    return v;
}

// The streams' mean square of a token's scratch m, as hc_mix_kernel's thread 0 sums it.
__device__ __forceinline__ float hc_mix_sq(const A & a, const float * m) {
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1];
    if (!a.params[5] || a.n_src < 5) {
        return m[(2 + hc) * hc];
    }
    const double * p = (const double *) m;
    double w = 0.0;
#pragma unroll
    for (int i = 0; i < HC_MIX_W; i++) {
        w += p[i];
    }
    return (float) (w / (e * hc));
}

// The hc_pre mix rows, unnormalized: block (r, t) puts row r of the weights times token t's
// streams in slot r of y's token-t row (scratch until hc_weights_kernel reads it), and block 0
// the streams' sum of squares in slot (2 + hc) hc. Q8_0 rows are read in linear_mv_q8's layout, a
// thread per block quarter; the RMS scale comes after the dot (the dot of the scaled streams up
// to rounding), so the rows spread over the SMs instead of a block per token doing all of them.
__global__ void __launch_bounds__(256) hc_mix_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], r = blockIdx.x, t = blockIdx.y;
    const int64_t eh = e * hc;
    const float * x = (const float *) a.src[0] + t * eh;
    const char * row = S(a, 1) + r * a.src_nb[1][1];
    float acc = 0.0f;
    double sq = 0.0;
    if (a.src_type[1] == T_Q8_0) {
        const int q = threadIdx.x % 4;
#pragma unroll 2
        for (int64_t b = threadIdx.x / 4; b < eh / 32; b += blockDim.x / 4) {
            const char * blk = row + b * 34;
            const float d = __half2float(__ldg((const __half *) blk));
            const uint16_t * qs = (const uint16_t *) (blk + 2 + 8 * q);
            const float4 * xv = (const float4 *) (x + b * 32 + 8 * q);
            const float4 u = __ldg(xv), w = __ldg(xv + 1);
            const float xs[8] = {u.x, u.y, u.z, u.w, w.x, w.y, w.z, w.w};
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint16_t v = __ldg(qs + j);
                s += (float) (int8_t) (v & 0xFF) * xs[2 * j] + (float) (int8_t) (v >> 8) * xs[2 * j + 1];
            }
            acc += d * s;
            if (r == 0) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    sq += (double) xs[j] * xs[j];
                }
            }
        }
    } else if (a.src_type[1] == T_F32) {
        // The generic loop's elements in its order (the same bits), HC_MIX_U of them loaded ahead:
        // decode's 24 blocks otherwise wait out one load latency per element (~28 GB/s on a V100).
        const float * w = (const float *) row;
        const int64_t bd = blockDim.x;
        int64_t i = threadIdx.x;
        for (; i + (HC_MIX_U - 1) * bd < eh; i += HC_MIX_U * bd) {
            float wv[HC_MIX_U], xv[HC_MIX_U];
#pragma unroll
            for (int u = 0; u < HC_MIX_U; u++) {
                wv[u] = __ldg(w + i + u * bd);
                xv[u] = __ldg(x + i + u * bd);
            }
#pragma unroll
            for (int u = 0; u < HC_MIX_U; u++) {
                acc += wv[u] * xv[u];
                if (r == 0) {
                    sq += (double) xv[u] * xv[u];
                }
            }
        }
        for (; i < eh; i += bd) {
            acc += w[i] * x[i];
            if (r == 0) {
                sq += (double) x[i] * x[i];
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < eh; i += blockDim.x) {
            acc += wval(row, a.src_type[1], i) * x[i];
            if (r == 0) {
                sq += (double) x[i] * x[i];
            }
        }
    }
    acc = warp_sum(acc);
    if (r == 0) {
        sq = warp_sum(sq);
    }
    __shared__ float part[8];
    __shared__ double sqp[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        part[warp] = acc;
        sqp[warp] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        double w = 0.0;
        for (int i = 0; i < (int) blockDim.x / 32; i++) {
            v += part[i];
            w += sqp[i];
        }
        float * y = hc_mix_at(a, t);
        y[r] = v;
        if (r == 0) {
            y[(2 + hc) * hc] = (float) (w / eh);
        }
    }
}

// hc_mix_kernel for F32 weights and a few tokens with the norm fused (params[5], #606): a block
// per (row, warp of hc_mix_kernel's block, token), a 32-thread block whose threads sum their
// elements in their order and meet in the same warp_sum, its lane 0 storing the warp's sum for
// hc_mixv / hc_mix_sq to add in thread 0's order (the same bits). Decode's 24 blocks of 256 read
// 128 KB each on 24 SMs (~8 us); 192 blocks spread over all of them.
// The body for row r, with SQ (row 0) the sum of squares too: two copies, so that the other rows'
// loads are scheduled without the squares' registers (#831).
template <int U, bool SQ>
__device__ __forceinline__ void hc_mix_split_row(const A & a, int64_t r) {
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], t = blockIdx.y;
    const int64_t warp = blockIdx.x % HC_MIX_W;
    const int64_t eh = e * hc, bd = 32 * HC_MIX_W;
    const float * x = (const float *) a.src[0] + t * eh;
    const float * w = (const float *) (S(a, 1) + r * a.src_nb[1][1]);
    float acc = 0.0f;
    double sq = 0.0;
    int64_t i = warp * 32 + threadIdx.x;
    for (; i + (U - 1) * bd < eh; i += U * bd) {
        float wv[U], xv[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            wv[u] = __ldg(w + i + u * bd);
            xv[u] = __ldg(x + i + u * bd);
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            acc += wv[u] * xv[u];
            if (SQ) {
                sq += (double) xv[u] * xv[u];
            }
        }
    }
    for (; i < eh; i += bd) {
        acc += w[i] * x[i];
        if (SQ) {
            sq += (double) x[i] * x[i];
        }
    }
    acc = warp_sum(acc);
    if (SQ) {
        sq = warp_sum(sq);
    }
    if (threadIdx.x == 0) {
        float * m = hc_mix_at(a, t);
        m[HC_MIX_W * (2 + r) + warp] = acc;
        if (SQ) {
            ((double *) m)[warp] = sq;
        }
    }
}

template <int U>
__global__ void __launch_bounds__(32) hc_mix_split_kernel(const A a) {
    pdl_wait();
    const int64_t r = blockIdx.x / HC_MIX_W;
    if (r == 0) {
        hc_mix_split_row<U, true>(a, r);
    } else {
        hc_mix_split_row<U, false>(a, r);
    }
}

// hc_mix_kernel for F32 weights over HC_MT tokens a block (#567, the GB10's prompt steps): thread
// j sums the same elements of each (row, token) as thread j of hc_mix_kernel's block (r, t) does,
// in the same order, and the sums meet in the same tree (the same bits), but each weight and
// stream value is read once per block instead of once per row or token (the per-row blocks
// re-read every token's streams from L2 (2 + hc) hc times). (2 + hc) hc <= HC_MR.
constexpr int HC_MT = 4, HC_MR = 24;
// Element steps of hc_mix_tiled_kernel's loop whose loads are issued before their sums (#1144).
constexpr int HC_MU = 2;

// Mixer row r's element i as hc_mix_tiled_kernel's type WT reads it (wval's values, #575: the
// NVFP4 checkpoint's mixers are BF16).
template <int WT>
__device__ __forceinline__ float hc_tiled_w(const char * w, int64_t wnb, int r, int64_t i) {
    const char * wr = w + r * wnb;
    if constexpr (WT == T_BF16) {
        return bf16f(((const uint16_t *) wr)[i]);
    } else if constexpr (WT == T_F16) {
        return __half2float(((const __half *) wr)[i]);
    } else {
        return __ldg((const float *) wr + i);
    }
}

// The weight type is a template argument, and with NR = (2 + hc) hc known at compile time (GLM's
// 4 streams: HC_MR) each step's loads (HC_MU steps of HC_MT stream values and NR weights) go out
// unpredicated, all before the step's sums. With the type chosen per load and the rows and tokens
// predicated, each row's loads waited for the sums before them: ~9 us a step at one block an SM,
// ~4 ms a 2048-token prompt step on a V100 (#1144). A short last block's missing tokens read the
// last token's streams; their sums are never stored. The sums are the same operations in the
// same order (the same bits). NR = 0: the rows at run time, predicated.
template <int WT, int NR>
__global__ void __launch_bounds__(256) hc_mix_tiled_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], n = a.src_ne[0][2];
    const int64_t eh = e * hc, t0 = (int64_t) blockIdx.x * HC_MT;
    const int nr = NR ? NR : (int) ((2 + hc) * hc), nt = (int) min((int64_t) HC_MT, n - t0);
    const float * x = (const float *) a.src[0] + t0 * eh;
    const char * w = S(a, 1);
    const int64_t wnb = a.src_nb[1][1];
    int64_t xo[HC_MT];
#pragma unroll
    for (int t = 0; t < HC_MT; t++) {
        xo[t] = (int64_t) min(t, nt - 1) * eh;
    }
    float acc[HC_MR][HC_MT];
    double sq[HC_MT];
#pragma unroll
    for (int t = 0; t < HC_MT; t++) {
        sq[t] = 0.0;
#pragma unroll
        for (int r = 0; r < HC_MR; r++) {
            acc[r][t] = 0.0f;
        }
    }
    const int64_t bd = blockDim.x;
    int64_t i = threadIdx.x;
    for (; i + (HC_MU - 1) * bd < eh; i += HC_MU * bd) {
        float xv[HC_MU][HC_MT], wv[HC_MU][HC_MR];
#pragma unroll
        for (int u = 0; u < HC_MU; u++) {
#pragma unroll
            for (int t = 0; t < HC_MT; t++) {
                xv[u][t] = __ldg(x + xo[t] + i + u * bd);
            }
#pragma unroll
            for (int r = 0; r < HC_MR; r++) {
                wv[u][r] = (NR || r < nr) ? hc_tiled_w<WT>(w, wnb, r, i + u * bd) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < HC_MU; u++) {
#pragma unroll
            for (int t = 0; t < HC_MT; t++) {
                sq[t] += (double) xv[u][t] * xv[u][t];
            }
#pragma unroll
            for (int r = 0; r < HC_MR; r++) {
                if (NR || r < nr) {
#pragma unroll
                    for (int t = 0; t < HC_MT; t++) {
                        acc[r][t] += wv[u][r] * xv[u][t];
                    }
                }
            }
        }
    }
    for (; i < eh; i += bd) {
        float xv[HC_MT];
#pragma unroll
        for (int t = 0; t < HC_MT; t++) {
            xv[t] = __ldg(x + xo[t] + i);
            sq[t] += (double) xv[t] * xv[t];
        }
#pragma unroll
        for (int r = 0; r < HC_MR; r++) {
            if (NR || r < nr) {
                const float wv = hc_tiled_w<WT>(w, wnb, r, i);
#pragma unroll
                for (int t = 0; t < HC_MT; t++) {
                    acc[r][t] += wv * xv[t];
                }
            }
        }
    }
    __shared__ float part[HC_MR * HC_MT][8];
    __shared__ double sqp[HC_MT][8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int t = 0; t < HC_MT; t++) {
        const double s = warp_sum(sq[t]);
        if (lane == 0) {
            sqp[t][warp] = s;
        }
#pragma unroll
        for (int r = 0; r < HC_MR; r++) {
            if (r < nr) {
                const float v = warp_sum(acc[r][t]);
                if (lane == 0) {
                    part[r * HC_MT + t][warp] = v;
                }
            }
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < nr * nt; k += blockDim.x) {
        const int r = k / nt, t = k % nt;
        float v = 0.0f;
        for (int i = 0; i < (int) blockDim.x / 32; i++) {
            v += part[r * HC_MT + t][i];
        }
        float * y = hc_mix_at(a, t0 + t);
        y[r] = v;
        if (r == 0) {
            double s = 0.0;
            for (int i = 0; i < (int) blockDim.x / 32; i++) {
                s += sqp[t][i];
            }
            y[(2 + hc) * hc] = (float) (s / eh);
        }
    }
}

// A block of HC_MAX^2 threads per token after hc_mix_kernel, a thread per comb entry: the RMS
// scale, the weights and llama.cpp's build_hc_sinkhorn (`sinkhorn` in ops.rs) into their outputs.
// Rows and columns are normalized independently, and each thread sums its row or column in the
// serial order, so the bits are those of one thread doing it all (which took ~47 us a call).
// hc_weights_tok is its body for token t, also taken by hc_finish_kernel's blocks of 256 threads
// (those past HC_MAX^2 idle).
__device__ void hc_weights_tok(const A & a, int64_t t) {
    __shared__ float c[HC_MAX * HC_MAX];
    const int64_t e = a.src_ne[0][0], n = a.src_ne[0][2];
    const int hc = (int) a.src_ne[0][1], i = threadIdx.x;
    const bool live = i < hc * hc, in = i < HC_MAX * HC_MAX;
    const int r = live ? i / hc : 0, k = live ? i % hc : 0;
    float * out = (float *) a.dst;
    const float * mix = hc_mix_at(a, t);
    const float s = 1.0f / sqrtf(hc_mix_sq(a, mix) + pf(a, 0));
    const float * scale = (const float *) a.src[2];
    const float * base = (const float *) a.src[3];
    const float eps = pf(a, 1);
    if (i < hc) {
        out[e * n + t * hc + i] = sigm(hc_mixv(a, mix, i) * s * scale[0] + base[i]) + eps;
        out[e * n + hc * n + t * hc + i] = sigm(hc_mixv(a, mix, hc + i) * s * scale[1] + base[hc + i]) * 2.0f;
    }
    float v = live ? hc_mixv(a, mix, 2 * hc + i) * s * scale[2] + base[2 * hc + i] : 0.0f;
    if (in) {
        c[i] = v;
    }
    __syncthreads();
    // Softmax of row r (c + r hc), plus eps.
    if (live) {
        const float * row = c + r * hc;
        float mx = -INFINITY;
        for (int d = 0; d < hc; d++) {
            mx = fmaxf(mx, row[d]);
        }
        double sum = 0.0;
        for (int d = 0; d < hc; d++) {
            sum += expf(row[d] - mx);
        }
        const float inv = (float) (1.0 / sum);
        v = expf(v - mx) * inv + eps;
    }
    __syncthreads();
    if (in) {
        c[i] = v;
    }
    __syncthreads();
    for (int it = 0; it < (int) a.params[2]; it++) {
        if (it > 0) {
            if (live) {
                float sum = 0.0f;
                for (int d = 0; d < hc; d++) {
                    sum += c[r * hc + d];
                }
                sum += eps;
                v /= sum;
            }
            __syncthreads();
            if (in) {
                c[i] = v;
            }
            __syncthreads();
        }
        if (live) {
            float sum = 0.0f;
            for (int d = 0; d < hc; d++) {
                sum += c[k + hc * d];
            }
            sum += eps;
            v /= sum;
        }
        __syncthreads();
        if (in) {
            c[i] = v;
        }
        __syncthreads();
    }
    if (live) {
        out[e * n + 2 * hc * n + t * hc * hc + i] = v;
    }
}

__global__ void hc_weights_kernel(const A a) {
    pdl_wait();
    hc_weights_tok(a, blockIdx.x);
}

// hc_weights_tok's comb weights for HC streams (HC * HC <= 32) by one warp (#831): lane i takes
// entry i, computed with hc_weights_tok's operations for it in the same order (the same bits),
// the row's or column's entries taken from their lanes by shuffles instead of from shared memory
// behind a block barrier. The one thread that did all entries before (#606) ran 16 entries'
// divisions a step in a row, which made it the long block of hc_finish_kernel (~10 us a call).
// All 32 lanes call it; those past HC * HC compute entry 0 and store nothing.
template <int HC>
__device__ void hc_sinkhorn_warp(const A & a, int64_t t, const float * mv, float s) {
    static_assert(HC * HC <= 32, "one lane an entry");
    const int64_t e = a.src_ne[0][0], n = a.src_ne[0][2];
    float * out = (float *) a.dst;
    const float * scale = (const float *) a.src[2];
    const float * base = (const float *) a.src[3];
    const float eps = pf(a, 1);
    const int lane = threadIdx.x % 32;
    const bool live = lane < HC * HC;
    const int i = live ? lane : 0, r = i / HC, k = i % HC;
    float v = mv[2 * HC + i] * s * scale[2] + base[2 * HC + i];
    // Softmax of row r, plus eps.
    {
        float row[HC];
#pragma unroll
        for (int d = 0; d < HC; d++) {
            row[d] = __shfl_sync(0xffffffffu, v, r * HC + d);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int d = 0; d < HC; d++) {
            mx = fmaxf(mx, row[d]);
        }
        double sum = 0.0;
#pragma unroll
        for (int d = 0; d < HC; d++) {
            sum += expf(row[d] - mx);
        }
        const float inv = (float) (1.0 / sum);
        v = expf(v - mx) * inv + eps;
    }
    for (int it = 0; it < (int) a.params[2]; it++) {
        if (it > 0) {
            float c[HC];
#pragma unroll
            for (int d = 0; d < HC; d++) {
                c[d] = __shfl_sync(0xffffffffu, v, r * HC + d);
            }
            float sum = 0.0f;
#pragma unroll
            for (int d = 0; d < HC; d++) {
                sum += c[d];
            }
            sum += eps;
            v /= sum;
        }
        float c[HC];
#pragma unroll
        for (int d = 0; d < HC; d++) {
            c[d] = __shfl_sync(0xffffffffu, v, k + HC * d);
        }
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < HC; d++) {
            sum += c[d];
        }
        sum += eps;
        v /= sum;
    }
    if (live) {
        out[e * n + 2 * HC * n + t * HC * HC + i] = v;
    }
}

// hc_finish_kernel's y and h for token t (hc <= HC streams), each summed in element order. Four
// of a thread's elements are loaded at a time in a loop that is not unrolled further: these blocks
// run alone on an SM, and straight-line code waits out an instruction fetch per instruction (a
// 16-at-once version took ~20 us a call on a V100, its code cold after a layer's other kernels).
template <int HC>
__device__ __forceinline__ void hc_finish_norm(const A & a, int64_t t, const float (&pre)[HC_MAX]) {
    constexpr int U = 4;
    const int64_t n = a.src_ne[0][2];
    const int e = (int) a.src_ne[0][0], bd = (int) blockDim.x;
    const int hc = HC == HC_MAX ? (int) a.src_ne[0][1] : HC;
    const float * x = (const float *) a.src[0] + t * e * hc;
    float * y = (float *) a.dst + t * e;
    double sum = 0.0;
#pragma unroll 1
    for (int i0 = threadIdx.x; i0 < e; i0 += U * bd) {
        float xv[U][HC];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * bd;
#pragma unroll
            for (int h = 0; h < HC; h++) {
                xv[u][h] = i < e && h < hc ? x[h * e + i] : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * bd;
            if (i < e) {
                // The FMAs spelled out: with hc known the compiler may fuse x0 pre0 into the
                // first add instead of x1 pre1, which hc_pre_kernel's loop does (1 ulp off).
                float acc = xv[u][0] * pre[0];
#pragma unroll
                for (int h = 1; h < HC; h++) {
                    if (h < hc) {
                        acc = __fmaf_rn(xv[u][h], pre[h], acc);
                    }
                }
                y[i] = acc;
                sum += (double) acc * acc;
            }
        }
    }
    sum = block_sum(sum);
    const float sn = 1.0f / sqrtf((float) (sum / e) + pf(a, 4));
    const float * w = (const float *) a.src[4];
    float * o = (float *) a.dst + hc_h_off(e, hc, n) + t * e;
#pragma unroll 1
    for (int i0 = threadIdx.x; i0 < e; i0 += U * bd) {
        float yv[U], wv[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * bd;
            yv[u] = i < e ? y[i] : 0.0f;
            wv[u] = i < e ? w[i] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int i = i0 + u * bd;
            if (i < e) {
                o[i] = yv[u] * sn * wv[u];
            }
        }
    }
}

// hc_weights_kernel, hc_pre_kernel and the sublayer's rms_norm_kernel in one launch (#606), after
// hc_mix_kernel with the norm weight given (src 4, the norm's eps in params[4]). Blocks t < n:
// token t's pre weights (computed as hc_weights_tok does, by every thread), then y as
// hc_pre_kernel and h = norm(y) w as rms_norm_kernel does, a thread per element of both in their
// orders (the same 256 threads as rms_norm_kernel's block, the same sums: the same bits). Blocks
// n + t: token t's post and comb weights, for hc = 4 the comb by one warp (hc_sinkhorn_warp),
// else hc_weights_tok's block. Both read the mix rows, so these are past h, not in y; the
// block's threads first sum them (hc_mixv) into shared memory, each row at once.
__global__ void __launch_bounds__(256) hc_finish_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[0][0], n = a.src_ne[0][2], b = blockIdx.x;
    const int hc = (int) a.src_ne[0][1];
    const int64_t t = b < n ? b : b - n;
    if (b >= n && hc != 4) {
        hc_weights_tok(a, t);
        return;
    }
    __shared__ float mv[(2 + HC_MAX) * HC_MAX + 1];
    const float * mix = hc_mix_at(a, t);
    const int rows = (2 + hc) * hc;
    for (int r = threadIdx.x; r <= rows; r += blockDim.x) {
        mv[r] = r < rows ? hc_mixv(a, mix, r) : hc_mix_sq(a, mix);
    }
    __syncthreads();
    const float s = 1.0f / sqrtf(mv[rows] + pf(a, 0));
    const float * scale = (const float *) a.src[2];
    const float * base = (const float *) a.src[3];
    const float eps = pf(a, 1);
    if (b >= n) {
        float * out = (float *) a.dst;
        const int i = threadIdx.x;
        if (i < 4) {
            out[e * n + t * 4 + i] = sigm(mv[i] * s * scale[0] + base[i]) + eps;
            out[e * n + 4 * n + t * 4 + i] = sigm(mv[4 + i] * s * scale[1] + base[4 + i]) * 2.0f;
        } else if (i / 32 == 1) {
            hc_sinkhorn_warp<4>(a, t, mv, s);
        }
        return;
    }
    float pre[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        pre[h] = h < hc ? sigm(mv[h] * s * scale[0] + base[h]) + eps : 0.0f;
    }
    if (hc == 4) {
        hc_finish_norm<4>(a, t, pre);
    } else {
        hc_finish_norm<HC_MAX>(a, t, pre);
    }
}

// hc_finish_kernel for 4 streams of up to HC_WIDE_E elements (#831, decode on a GB10): its norm
// block, one SM's 256 threads reading the 4 streams in four rounds and then y and the norm weight
// in four more, was the kernel's long block. Here 1024 threads load all of a token's streams and
// its norm weight at once (with the mix rows), each element's y computed as hc_finish_norm's
// thread does. The sum of squares keeps that block's order: thread j < 256 adds the squares of
// elements j + 256 m in order of m (from shared memory), the other threads add nothing, and
// block_sum's warp sums of zeros are the zeros its 8 warps' sum is padded with. h is then y sn w
// from the registers: the same bits. The post and comb block is hc_finish_kernel's.
constexpr int HC_WIDE_T = 1024, HC_WIDE_U = 4, HC_WIDE_E = HC_WIDE_T * HC_WIDE_U;

__global__ void __launch_bounds__(HC_WIDE_T) hc_finish_wide_kernel(const A a) {
    pdl_wait();
    const int64_t n = a.src_ne[0][2], b = blockIdx.x;
    const int e = (int) a.src_ne[0][0];
    const int64_t t = b < n ? b : b - n;
    float xv[HC_WIDE_U][4], wv[HC_WIDE_U];
    if (b < n) {
        const float * x = (const float *) a.src[0] + t * e * 4;
        const float * w = (const float *) a.src[4];
#pragma unroll
        for (int u = 0; u < HC_WIDE_U; u++) {
            const int i = threadIdx.x + u * HC_WIDE_T;
#pragma unroll
            for (int h = 0; h < 4; h++) {
                xv[u][h] = i < e ? x[h * e + i] : 0.0f;
            }
            wv[u] = i < e ? w[i] : 0.0f;
        }
    }
    __shared__ float mv[(2 + 4) * 4 + 1];
    __shared__ float ys[HC_WIDE_E];
    const float * mix = hc_mix_at(a, t);
    const int rows = (2 + 4) * 4;
    for (int r = threadIdx.x; r <= rows; r += blockDim.x) {
        mv[r] = r < rows ? hc_mixv(a, mix, r) : hc_mix_sq(a, mix);
    }
    __syncthreads();
    const float s = 1.0f / sqrtf(mv[rows] + pf(a, 0));
    const float * scale = (const float *) a.src[2];
    const float * base = (const float *) a.src[3];
    const float eps = pf(a, 1);
    if (b >= n) {
        float * out = (float *) a.dst;
        const int i = threadIdx.x;
        if (i < 4) {
            out[e * n + t * 4 + i] = sigm(mv[i] * s * scale[0] + base[i]) + eps;
            out[e * n + 4 * n + t * 4 + i] = sigm(mv[4 + i] * s * scale[1] + base[4 + i]) * 2.0f;
        } else if (i / 32 == 1) {
            hc_sinkhorn_warp<4>(a, t, mv, s);
        }
        return;
    }
    float pre[4];
#pragma unroll
    for (int h = 0; h < 4; h++) {
        pre[h] = sigm(mv[h] * s * scale[0] + base[h]) + eps;
    }
    float * y = (float *) a.dst + t * e;
    float yv[HC_WIDE_U];
#pragma unroll
    for (int u = 0; u < HC_WIDE_U; u++) {
        const int i = threadIdx.x + u * HC_WIDE_T;
        // hc_finish_norm<4>'s FMAs, spelled out the same way.
        float acc = xv[u][0] * pre[0];
#pragma unroll
        for (int h = 1; h < 4; h++) {
            acc = __fmaf_rn(xv[u][h], pre[h], acc);
        }
        yv[u] = acc;
        if (i < e) {
            y[i] = acc;
            ys[i] = acc;
        }
    }
    __syncthreads();
    double sum = 0.0;
    if (threadIdx.x < 256) {
        for (int i = threadIdx.x; i < e; i += 256) {
            const float v = ys[i];
            sum += (double) v * v;
        }
    }
    sum = block_sum(sum);
    const float sn = 1.0f / sqrtf((float) (sum / e) + pf(a, 4));
    float * o = (float *) a.dst + hc_h_off(e, 4, n) + t * e;
#pragma unroll
    for (int u = 0; u < HC_WIDE_U; u++) {
        const int i = threadIdx.x + u * HC_WIDE_T;
        if (i < e) {
            o[i] = yv[u] * sn * wv[u];
        }
    }
}

// The mixed input: y = sum_h x[h] pre[h], after hc_weights_kernel.
__global__ void hc_pre_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], n = a.src_ne[0][2], t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e) {
        return;
    }
    const float * x = (const float *) a.src[0] + t * e * hc;
    const float * pre = (const float *) a.dst + e * n + t * hc;
    float acc = x[i] * pre[0];
    for (int h = 1; h < hc; h++) {
        acc += x[h * e + i] * pre[h];
    }
    ((float *) a.dst)[t * e + i] = acc;
}

__global__ void hc_post_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[1][0], hc = a.src_ne[1][1], t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e) {
        return;
    }
    float v = ((const float *) (S(a, 0) + t * a.src_nb[0][1]))[i];
    if (a.params[0]) {
        v += ((const float *) (S(a, 4) + t * a.src_nb[4][1]))[i];
    }
    const float * res = (const float *) a.src[1] + t * hc * e;
    const float * post = (const float *) (S(a, 2) + t * a.src_nb[2][1]);
    const float * comb = (const float *) (S(a, 3) + t * a.src_nb[3][2]);
    float * out = (float *) a.dst + t * hc * e;
    for (int64_t d = 0; d < hc; d++) {
        float acc = v * post[d];
        for (int64_t s = 0; s < hc; s++) {
            acc += res[s * e + i] * comb[d + hc * s];
        }
        out[d * e + i] = acc;
    }
}

__global__ void hc_head_kernel(const A a) {
    pdl_wait();
    extern __shared__ float mean[];
    const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1], r = blockIdx.x;
    const int64_t t = a.params[1] ? ((const int32_t *) a.src[2])[r] : r;
    const float * x = (const float *) a.src[0] + t * hc * e;
    const float * w = (const float *) a.src[1];
    const float inv = 1.0f / (float) hc;
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        float acc = x[i];
        for (int64_t s = 1; s < hc; s++) {
            acc += x[s * e + i];
        }
        acc *= inv;
        mean[i] = acc;
        sum += (double) acc * acc;
    }
    sum = block_sum(sum);
    const float s = 1.0f / sqrtf((float) (sum / e) + pf(a, 0));
    float * o = (float *) a.dst + r * e;
    for (int64_t i = threadIdx.x; i < e; i += blockDim.x) {
        o[i] = mean[i] * s * w[i];
    }
}

// A thread per (channel, sequence), the tokens in order. blockIdx.y picks the projection of a
// fused q/k/v replay (params[4], #676: sources x_q, w_q, state, x_k, w_k, x_v, w_v, dst
// [c, n, 3], projection j's channels from ch_off + j * c); one projection otherwise.
__global__ void kda_conv_kernel(const A a) {
    pdl_wait();
    const int64_t c = a.ne[0], n = a.ne[1];
    const int j = blockIdx.y, xi = j == 0 ? 0 : 2 * j + 1, wi = xi + 1;
    const int64_t ch_off = a.params[0] + j * c, n_tok = a.params[1], n_seqs = n / n_tok;
    const bool verify = a.params[2] != 0;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c * n_seqs) {
        return;
    }
    const int64_t ch = idx % c, s = idx / c;
    const float * wc = (const float *) a.src[wi] + ch * 4;
    char * st = (char *) a.src[2] + (ch_off + ch) * a.src_nb[2][1] + s * a.src_nb[2][2];
    const int64_t snb0 = a.src_nb[2][0];
    float w0 = *(float *) st, w1 = *(float *) (st + snb0), w2 = *(float *) (st + 2 * snb0);
    float * out = (float *) a.dst + j * c * n;
    const char * xs = S(a, xi) + s * seq_nb(a, xi, 1, n_tok, n_seqs);
    for (int64_t t = 0; t < n_tok; t++) {
        const int64_t col = s * n_tok + t;
        const float xv = ((const float *) (xs + t * a.src_nb[xi][1]))[ch];
        if (verify) {
            ((float *) ((char *) a.src[3] + s * a.src_nb[3][2] + t * a.src_nb[3][1]))[ch] = xv;
        }
        float acc = 0.0f;
        acc += w0 * wc[0];
        acc += w1 * wc[1];
        acc += w2 * wc[2];
        acc += xv * wc[3];
        out[col * c + ch] = silu(acc);
        w0 = w1;
        w1 = w2;
        w2 = xv;
    }
    // A verify group leaves the state for the commit.
    if (verify) {
        return;
    }
    *(float *) st = w0;
    *(float *) (st + snb0) = w1;
    *(float *) (st + 2 * snb0) = w2;
}

// kda_conv_kernel for prompt steps on the GB10 (#571): a thread per (channel, token) instead of
// one walking a sequence's tokens (8,192 of them, a few warps an SM), each output from the
// same three earlier inputs (the state's for the first three tokens) with the same sums; the
// state written by kda_conv_state_kernel after it, the next launch. Sequences of 3+ tokens.
constexpr int KC_T = 32;  // tokens a thread

__global__ void kda_conv_tok_kernel(const A a) {
    pdl_wait();
    const int64_t c = a.ne[0], n = a.ne[1];
    const int64_t ch_off = a.params[0], n_tok = a.params[1], n_seqs = n / n_tok;
    const int64_t ch = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, t0 = (int64_t) blockIdx.y * KC_T;
    const int64_t s = blockIdx.z;
    if (ch >= c || t0 >= n_tok) {
        return;
    }
    const float * wc = (const float *) a.src[1] + ch * 4;
    const char * st = (const char *) a.src[2] + (ch_off + ch) * a.src_nb[2][1] + s * a.src_nb[2][2];
    const int64_t snb0 = a.src_nb[2][0];
    const char * xs = S(a, 0) + s * seq_nb(a, 0, 1, n_tok, n_seqs);
    auto in = [&](int64_t t) {
        return t < 0 ? *(const float *) (st + (3 + t) * snb0) : ((const float *) (xs + t * a.src_nb[0][1]))[ch];
    };
    float w0 = in(t0 - 3), w1 = in(t0 - 2), w2 = in(t0 - 1);
    float * out = (float *) a.dst;
    const int64_t t1 = min(n_tok, t0 + KC_T);
    for (int64_t t = t0; t < t1; t++) {
        const float xv = in(t);
        float acc = 0.0f;
        acc += w0 * wc[0];
        acc += w1 * wc[1];
        acc += w2 * wc[2];
        acc += xv * wc[3];
        out[(s * n_tok + t) * c + ch] = silu(acc);
        w0 = w1;
        w1 = w2;
        w2 = xv;
    }
}

__global__ void kda_conv_state_kernel(const A a) {
    pdl_wait();
    const int64_t c = a.ne[0], n = a.ne[1];
    const int64_t ch_off = a.params[0], n_tok = a.params[1], n_seqs = n / n_tok;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c * n_seqs) {
        return;
    }
    const int64_t ch = idx % c, s = idx / c;
    char * st = (char *) a.src[2] + (ch_off + ch) * a.src_nb[2][1] + s * a.src_nb[2][2];
    const char * xs = S(a, 0) + s * seq_nb(a, 0, 1, n_tok, n_seqs);
#pragma unroll
    for (int i = 0; i < 3; i++) {
        *(float *) (st + i * a.src_nb[2][0]) = ((const float *) (xs + (n_tok - 3 + i) * a.src_nb[0][1]))[ch];
    }
}

__global__ void kda_gate_kernel(const A a, int64_t n) {
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float * z = (const float *) a.src[0];
    const float * dt = (const float *) a.src[1];
    const float * av = (const float *) a.src[2];
    const int64_t c = a.src_ne[1][0], head = c / a.src_ne[2][0], ch = i % c;
    ((float *) a.dst)[i] = sigm(-((z[i] + dt[ch]) * av[ch / head])) * pf(a, 0);
}

__global__ void sigmoid_kernel(const float * x, float * out, int64_t n) {
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sigm(x[i]);
    }
}

// KDA's gated delta rule, d = 128: a block per (head, sequence), thread 8 j + l holds the keys
// l, l + 8, ... of state row j.
constexpr int KD = 128, KL = 8, KR = KD / KL;

__global__ void __launch_bounds__(KD * KL) kda_scan_kernel(const A a) {
    pdl_wait();
    __shared__ float qs[KD], ks[KD], dec[KD], part[8];
    const int64_t h = a.ne[1], hd = blockIdx.x, s = blockIdx.y;
    const int64_t n_tok = a.params[0], n_seqs = a.ne[2] / n_tok;
    const bool verify = a.params[1] != 0;
    const int tid = threadIdx.x, j = tid / KL, l = tid % KL;
    const float * q = (const float *) a.src[0];
    const float * k = (const float *) a.src[1];
    const float * v = (const float *) a.src[2];
    const char * g = S(a, 3) + s * seq_nb(a, 3, 2, n_tok, n_seqs) + hd * a.src_nb[3][1];
    const char * beta = S(a, 4) + s * seq_nb(a, 4, 1, n_tok, n_seqs) + hd * 4;
    float * st = (float *) ((char *) a.src[5] + hd * a.src_nb[5][2] + s * a.src_nb[5][3]) + j * KD;
    float S_[KR];
#pragma unroll
    for (int r = 0; r < KR; r++) {
        S_[r] = st[l + KL * r];
    }
    const float qscale = rsqrtf((float) KD);
    float * out = (float *) a.dst;
    for (int64_t t = 0; t < n_tok; t++) {
        const int64_t col = s * n_tok + t, off = (col * h + hd) * KD;
        if (tid < KD) {
            qs[tid] = q[off + tid];
            ks[tid] = k[off + tid];
            const float gv = ((const float *) (g + t * a.src_nb[3][2]))[tid];
            dec[tid] = expf(gv);
            if (verify) {
                ((float *) ((char *) a.src[6] + s * a.src_nb[6][2] + t * a.src_nb[6][1]))[hd * KD + tid] = gv;
            }
        }
        const float bt = *(const float *) (beta + t * a.src_nb[4][1]);
        if (verify && tid == 0) {
            ((float *) ((char *) a.src[7] + s * a.src_nb[7][2] + t * a.src_nb[7][1]))[hd] = bt;
        }
        __syncthreads();
        if (tid < 2 * KD) {
            const float x = tid < KD ? qs[tid] : ks[tid - KD];
            const float sq = warp_sum(x * x);
            if (tid % 32 == 0) {
                part[tid / 32] = sq;
            }
        }
        __syncthreads();
        const float sq = part[0] + part[1] + part[2] + part[3];
        const float sk = part[4] + part[5] + part[6] + part[7];
        const float qn = qscale / sqrtf(sq + 1e-6f), kn = 1.0f / sqrtf(sk + 1e-6f);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < KR; r++) {
            const int i = l + KL * r;
            S_[r] *= dec[i];
            kv += S_[r] * (ks[i] * kn);
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        kv += __shfl_xor_sync(0xffffffffu, kv, 4);
        const float delta = (v[off + j] - kv) * bt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < KR; r++) {
            const int i = l + KL * r;
            S_[r] += delta * (ks[i] * kn);
            o += S_[r] * (qs[i] * qn);
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        o += __shfl_xor_sync(0xffffffffu, o, 4);
        if (l == 0) {
            out[off + j] = o;
        }
        __syncthreads();
    }
    // A verify group leaves the state for the commit.
    if (verify) {
        return;
    }
#pragma unroll
    for (int r = 0; r < KR; r++) {
        st[l + KL * r] = S_[r];
    }
}

// kda_scan_kernel for prompt steps on the GB10 (#571): a state row only meets the other rows
// through the token's q and k norms, so a block takes KS_W warps' rows (4 a warp, the same
// 8 lanes a row and keys l + 8 r a lane) and stages KS_T tokens at a time: their decays, the
// normalized q and k values (each warp computing a token's norms as the kernel above does) and
// the rows' v values, the next tokens' global loads in flight while the warps run these. The
// same values in the same order: the same bits, without kda_scan_kernel's three barriers a
// token and with 2 blocks a head where it had a 1,024-thread block (64 heads, 48 SMs: two
// waves). Not for verify groups.
constexpr int KS_W = 16, KS_T = 8, KS_ROWS = 4 * KS_W, KS_P = 20;  // KS_P: a lane's keys, padded

__device__ __forceinline__ int ks_idx(int i) { return (i % KL) * KS_P + i / KL; }

__global__ void __launch_bounds__(32 * KS_W) kda_scan_rows_kernel(const A a) {
    pdl_wait();
    __shared__ __align__(16) float dec[KS_T][KL * KS_P], kk[KS_T][KL * KS_P], qq[KS_T][KL * KS_P];
    __shared__ float vs[KS_T][KS_ROWS], bs[KS_T];
    const int64_t h = a.ne[1], n_tok = a.params[0];
    const int64_t hd = blockIdx.x / (KD / KS_ROWS), s = blockIdx.y, n_seqs = a.ne[2] / n_tok;
    const int rb = (int) (blockIdx.x % (KD / KS_ROWS)) * KS_ROWS;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int jl = warp * 4 + lane / KL, j = rb + jl, l = lane % KL;
    const float * q = (const float *) a.src[0];
    const float * k = (const float *) a.src[1];
    const float * v = (const float *) a.src[2];
    const char * g = S(a, 3) + s * seq_nb(a, 3, 2, n_tok, n_seqs) + hd * a.src_nb[3][1];
    const char * beta = S(a, 4) + s * seq_nb(a, 4, 1, n_tok, n_seqs) + hd * 4;
    float * st = (float *) ((char *) a.src[5] + hd * a.src_nb[5][2] + s * a.src_nb[5][3]) + j * KD;
    float S_[KR];
#pragma unroll
    for (int r = 0; r < KR; r++) {
        S_[r] = st[l + KL * r];
    }
    const float qscale = rsqrtf((float) KD);
    float * out = (float *) a.dst;
    // Staging: warp w < KS_T takes token w (its q and k norms computed together, as above);
    // every thread KS_T * KS_ROWS / (32 KS_W) v values.
    static_assert(KS_T <= KS_W && KS_T * KS_ROWS == 32 * KS_W, "the staging split");
    const bool sw_on = warp < KS_T;
    const int vt = tid / KS_ROWS, vj = tid % KS_ROWS;
    float qr[4], kr[4], gr[4], vr = 0.0f, br = 0.0f;
    auto fetch = [&](int64_t t0) {
        const int64_t t = t0 + warp;
        if (sw_on && t < n_tok) {
            const int64_t off = ((s * n_tok + t) * h + hd) * KD;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                qr[c] = q[off + 32 * c + lane];
                kr[c] = k[off + 32 * c + lane];
                gr[c] = ((const float *) (g + t * a.src_nb[3][2]))[32 * c + lane];
            }
            br = *(const float *) (beta + t * a.src_nb[4][1]);
        }
        if (t0 + vt < n_tok) {
            vr = v[((s * n_tok + t0 + vt) * h + hd) * KD + rb + vj];
        }
    };
    auto stage = [&](int64_t t0) {
        if (sw_on && t0 + warp < n_tok) {
            float part[8];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                part[c] = warp_sum(qr[c] * qr[c]);
                part[4 + c] = warp_sum(kr[c] * kr[c]);
            }
            const float sq = part[0] + part[1] + part[2] + part[3];
            const float sk = part[4] + part[5] + part[6] + part[7];
            const float qn = qscale / sqrtf(sq + 1e-6f), kn = 1.0f / sqrtf(sk + 1e-6f);
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int i = ks_idx(32 * c + lane);
                qq[warp][i] = qr[c] * qn;
                kk[warp][i] = kr[c] * kn;
                dec[warp][i] = expf(gr[c]);
            }
            if (lane == 0) {
                bs[warp] = br;
            }
        }
        vs[vt][vj] = vr;
    };
    fetch(0);
    for (int64_t t0 = 0; t0 < n_tok; t0 += KS_T) {
        stage(t0);
        __syncthreads();
        if (t0 + KS_T < n_tok) {
            fetch(t0 + KS_T);
        }
        const int nt = (int) min((int64_t) KS_T, n_tok - t0);
        for (int tt = 0; tt < nt; tt++) {
            const float4 * d4 = (const float4 *) &dec[tt][l * KS_P];
            const float4 * k4 = (const float4 *) &kk[tt][l * KS_P];
            const float4 * q4 = (const float4 *) &qq[tt][l * KS_P];
            float dv[KR], kv_[KR];
#pragma unroll
            for (int c = 0; c < KR / 4; c++) {
                const float4 dd = d4[c], kq = k4[c];
                dv[4 * c] = dd.x, dv[4 * c + 1] = dd.y, dv[4 * c + 2] = dd.z, dv[4 * c + 3] = dd.w;
                kv_[4 * c] = kq.x, kv_[4 * c + 1] = kq.y, kv_[4 * c + 2] = kq.z, kv_[4 * c + 3] = kq.w;
            }
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < KR; r++) {
                S_[r] *= dv[r];
                kv += S_[r] * kv_[r];
            }
            kv += __shfl_xor_sync(0xffffffffu, kv, 1);
            kv += __shfl_xor_sync(0xffffffffu, kv, 2);
            kv += __shfl_xor_sync(0xffffffffu, kv, 4);
            const float delta = (vs[tt][jl] - kv) * bs[tt];
            float o = 0.0f;
#pragma unroll
            for (int c = 0; c < KR / 4; c++) {
                const float4 qv = q4[c];
                const float qa[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = 4 * c + e;
                    S_[r] += delta * kv_[r];
                    o += S_[r] * qa[e];
                }
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            o += __shfl_xor_sync(0xffffffffu, o, 4);
            if (l == 0) {
                out[((s * n_tok + t0 + tt) * h + hd) * KD + j] = o;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < KR; r++) {
        st[l + KL * r] = S_[r];
    }
}

__global__ void gated_rms_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.src_ne[1][0], r = blockIdx.x;
    const float * o = (const float *) a.src[0] + r * d;
    const float * w = (const float *) a.src[1];
    const float * g = a.params[1] ? (const float *) a.src[2] + r * d : nullptr;
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        sum += (double) o[i] * o[i];
    }
    sum = block_sum(sum);
    const float s = 1.0f / sqrtf((float) (sum / d) + pf(a, 0));
    float * out = (float *) a.dst + r * d;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        out[i] = o[i] * s * w[i] * (g ? sigm(g[i]) : 1.0f);
    }
}

__global__ void glu_kernel(const float * g, const float * u, float * out, int64_t n, float lim) {
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = lim > 0.0f ? silu(fminf(g[i], lim)) * fminf(fmaxf(u[i], -lim), lim) : silu(g[i]) * u[i];
    }
}

// GLU_MV (ops::glu_mv, #606): glu_kernel over linear_mv_q6k's products of the gate (`p.w`) and up
// (`wu`) rows with the same x, for 1-8 tokens in one launch instead of three: a CTA's warps 0, 2
// take gate rows 2 b, 2 b + 1 and warps 1, 3 the up rows, each summed exactly as linear_mv_q6k's
// warp sums its row (the same code), then glu_kernel's expression on the pair. The same bits.
template <int N>
__global__ void __launch_bounds__(128, Q6K_MIN_BLOCKS(N)) glu_mv_q6k(const Lin p, const char * wu, float lim) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 2 + warp / 2;
    __shared__ float res[2][2][N];
    if (r < p.m) {
        const char * row = (warp % 2 ? wu : p.w) + r * p.w_nb1;
        const int n = lane / 16, l = (lane % 16) * 2;
        float acc[N];
#pragma unroll
        for (int t = 0; t < N; t++) {
            acc[t] = 0.0f;
        }
        const int64_t nb = p.k / 256;
#pragma unroll KqUnroll<N>::v
        for (int64_t b = 0; b < nb; b++) {
            const char * blk = row + b * 210;
            const uint32_t a = __ldg((const uint16_t *) (blk + 64 * n + l));
            const uint32_t c = __ldg((const uint16_t *) (blk + 64 * n + 32 + l));
            const uint32_t h = __ldg((const uint16_t *) (blk + 128 + 32 * n + l));
            const float d = __half2float(__ldg((const __half *) (blk + 208)));
            const int8_t * sc = (const int8_t *) (blk + 192 + 8 * n + l / 16);
            float ds[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                ds[q] = d * i8_f(__ldg(sc + 2 * q));
            }
            float v[4][2];
            {
                const uint32_t ac = __byte_perm(a, c, 0x5410), hh = __byte_perm(h, 0, 0x1010);
                const uint32_t w01 = (ac & 0x0F0F0F0F) | (__byte_perm(hh << 4, hh << 2, 0x7610) & 0x30303030);
                const uint32_t w23 = ((ac >> 4) & 0x0F0F0F0F) | (__byte_perm(hh, hh >> 2, 0x7610) & 0x30303030);
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    v[0][j] = q6_b(w01, j);
                    v[1][j] = q6_b(w01, 2 + j);
                    v[2][j] = q6_b(w23, j);
                    v[3][j] = q6_b(w23, 2 + j);
                }
            }
#pragma unroll
            for (int t = 0; t < N; t++) {
                if (t < p.n) {
                    const float * x = (const float *) (p.x + t * p.x_nbt) + b * 256 + 128 * n + l;
                    float s = 0.0f;
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float2 xv = __ldg((const float2 *) (x + 32 * q));
                        s += ds[q] * (v[q][0] * xv.x + v[q][1] * xv.y);
                    }
                    acc[t] += s;
                }
            }
        }
        const float sc = lin_scale(p, r);
#pragma unroll
        for (int t = 0; t < N; t++) {
            const float s = warp_sum(acc[t]);
            if (lane == 0 && t < p.n) {
                res[warp / 2][warp % 2][t] = s * sc;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 2 * N) {
        const int rr = threadIdx.x / N, t = threadIdx.x % N;
        const int64_t o = (int64_t) blockIdx.x * 2 + rr;
        if (o < p.m && t < p.n) {
            const float g = res[rr][0][t], u = res[rr][1][t];
            p.dst[t * p.m + o] = lim > 0.0f ? silu(fminf(g, lim)) * fminf(fmaxf(u, -lim), lim) : silu(g) * u;
        }
    }
}

// GLU_MV: sources gate and up (Q6_K, one shape and layout) and x (F32, n tokens of k); params
// [limit]. dst F32 [m, n].
int launch_glu_mv(const A & a, cudaStream_t st) {
    if (a.src_type[0] != T_Q6_K || a.src_type[1] != T_Q6_K) {
        return 1;
    }
    Lin p{};
    p.w = (const char *) a.src[0];
    p.w_nb1 = a.src_nb[0][1];
    p.wtype = T_Q6_K;
    p.k = a.src_ne[0][0];
    p.m = a.src_ne[0][1];
    p.h = 1;
    p.x = (const char *) a.src[2];
    const int64_t * xne = a.src_ne[2];
    const int64_t * xnb = a.src_nb[2];
    p.n = a.ne[1];
    p.x_nbt = xne[1] == 1 && xne[2] > 1 ? xnb[2] : xnb[1];
    p.dst = (float *) a.dst;
    p.scale = 1.0f;
    const char * wu = (const char *) a.src[1];
    const bool x16 = ((int64_t) ((uintptr_t) p.x | p.x_nbt) % 16) == 0;
    if (a.src_ne[0][2] != 1 || a.src_ne[1][0] != p.k || a.src_ne[1][1] != p.m || a.src_ne[1][2] != 1 ||
        a.src_nb[1][1] != p.w_nb1 || xne[0] != p.k || p.k % 256 != 0 || p.m < 1 || p.n < 1 || p.n > 8 || !x16 ||
        ((int64_t) ((uintptr_t) p.w | (uintptr_t) wu | p.w_nb1) % 2) != 0) {
        return 1;
    }
    const float lim = host_f32(a.params[0]);
    const unsigned grid = (unsigned) ((p.m + 1) / 2);
    if (p.n == 1) {
        launch_pdl(glu_mv_q6k<1>, grid, 128, 0, st, p, wu, lim);
    } else if (p.n == 2) {
        launch_pdl(glu_mv_q6k<2>, grid, 128, 0, st, p, wu, lim);
    } else if (p.n <= 4) {
        launch_pdl(glu_mv_q6k<4>, grid, 128, 0, st, p, wu, lim);
    } else {
        launch_pdl(glu_mv_q6k<8>, grid, 128, 0, st, p, wu, lim);
    }
    return 0;
}

constexpr int RT_MAX = 16;  // experts per lane: at most 512

// A warp per token: k rounds of the warp's best remaining score (ties to the lower id).
__global__ void router_ids_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.src_ne[1][0], k = a.ne[0], n = a.ne[1];
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (t >= n) {
        return;
    }
    const float * lg = (const float *) a.src[0] + t * e;
    const float * b = (const float *) a.src[1];
    // --glm-route-bias (after Xolotl's --route-bias, Indras-Mirror/Xolotl, MIT): with a third
    // source, the layer's expert cache table, experts whose entry is a VRAM slot (below
    // params[0]) score params[1] more. Selection only: the weights stay unbiased.
    const int32_t * tab = a.n_src > 2 ? (const int32_t *) a.src[2] : nullptr;
    const int32_t n_slots = a.params[0];
    const float rb = pf(a, 1);
    float sc[RT_MAX];
    uint32_t taken = 0;
#pragma unroll
    for (int r = 0; r < RT_MAX; r++) {
        const int64_t i = lane + 32 * r;
        sc[r] = i < e ? sigm(lg[i]) + b[i] : -INFINITY;
        if (tab && i < e && tab[i] < n_slots) {
            sc[r] += rb;
        }
        if (i >= e) {
            taken |= 1u << r;
        }
    }
    int32_t * out = (int32_t *) a.dst + t * k;
    for (int64_t j = 0; j < k; j++) {
        float bv = -INFINITY;
        int bi = INT32_MAX;
#pragma unroll
        for (int r = 0; r < RT_MAX; r++) {
            const int i = lane + 32 * r;
            if (!(taken >> r & 1) && (sc[r] > bv || bi == INT32_MAX)) {
                bv = sc[r];
                bi = i;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oi != INT32_MAX && (bi == INT32_MAX || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            out[j] = bi;
        }
        if (bi % 32 == lane) {
            taken |= 1u << (bi / 32);
        }
    }
}

__global__ void router_w_kernel(const A a) {
    pdl_wait();
    const int64_t k = a.ne[0], n = a.ne[1], e = a.src_ne[0][0];
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const float * lg = (const float *) a.src[0] + t * e;
    const int32_t * ids = (const int32_t *) a.src[1] + t * k;
    float sum = 0.0f;
    for (int64_t j = 0; j < k; j++) {
        sum += sigm(lg[ids[j]]);
    }
    sum = a.params[0] ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
    const float scale = pf(a, 1);
    float * out = (float *) a.dst + t * k;
    for (int64_t j = 0; j < k; j++) {
        out[j] = sigm(lg[ids[j]]) / sum * scale;
    }
}

__global__ void weighted_sum_kernel(const A a) {
    pdl_wait();
    const int64_t e = a.ne[0], t = blockIdx.y, k = a.src_ne[1][0];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e) {
        return;
    }
    const float * y = (const float *) a.src[0] + t * k * e;
    const float * w = (const float *) a.src[1] + t * k;
    float acc = y[i] * w[0];
    for (int64_t j = 1; j < k; j++) {
        acc += y[j * e + i] * w[j];
    }
    ((float *) a.dst)[t * e + i] = acc;
}

__global__ void cache_rows_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.src_ne[1][0], t = blockIdx.y, off = a.params[0];
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const float v = ((const float *) (S(a, 1) + t * a.src_nb[1][1]))[i];
    const int64_t cell = ((const int32_t *) a.src[2])[t];
    __half * row = (__half *) ((char *) a.src[0] + cell * a.src_nb[0][1]);
    row[off + i] = __float2half_rn(v);
}

constexpr int KPOOL_MAX = 16;

__global__ void pool_keys_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.src_ne[3][0], kpool = a.src_ne[3][1], p = blockIdx.x;
    const char * cache = S(a, 0);
    const int64_t cnb1 = a.src_nb[0][1];
    const int32_t * members = (const int32_t *) a.src[1] + p * kpool;
    const int64_t rep = ((const int64_t *) a.src[2])[p];
    const float * ape = (const float *) a.src[3];
    for (int64_t c = threadIdx.x; c < d; c += blockDim.x) {
        float logit[KPOOL_MAX];
        float mx = -INFINITY;
        for (int64_t k = 0; k < kpool; k++) {
            const __half * row = (const __half *) (cache + (int64_t) members[k] * cnb1);
            logit[k] = __half2float(row[d + c]) + ape[c + d * k];
            mx = fmaxf(mx, logit[k]);
        }
        double sum = 0.0;
        for (int64_t k = 0; k < kpool; k++) {
            logit[k] = expf(logit[k] - mx);
            sum += logit[k];
        }
        const float inv = (float) (1.0 / sum);
        float acc = 0.0f;
        for (int64_t k = 0; k < kpool; k++) {
            const __half * row = (const __half *) (cache + (int64_t) members[k] * cnb1);
            acc += logit[k] * inv * __half2float(row[c]);
        }
        ((float *) a.dst)[p * d + c] = acc;
        ((__half *) ((char *) a.src[0] + rep * cnb1))[2 * d + c] = __float2half_rn(acc);
    }
}

// Sparse MLA, latent width 512: a block per (token, MH heads), a warp per head. Each chunk of
// MC selected positions is staged in shared memory (rows padded to 257 words: the score loop's
// lanes read different rows conflict-free); lane p scores position p, then every lane updates
// its 16 output columns (8 half2 words, lane + 32 r).
constexpr int MD = 512, MH = 4, MC = 32, MW = MD / 2 + 1;

// The GB10's prompt steps (#567) take MH_WIDE heads a block: each chunk's rows, which every
// head of a token reads, staged once for 4x the heads (the same per-warp work: the same bits).
constexpr int MH_WIDE = 16;

template <int H> constexpr int mla_smem() {
    return MC * 8 + MC * MW * 4 + H * MD * 4 + H * MC * 4;
}

template <int H>
__global__ void __launch_bounds__(32 * H) mla_attn_kernel(const A a) {
    pdl_wait();
    extern __shared__ __align__(16) char mla_sm[];
    int64_t * srow = (int64_t *) mla_sm;
    uint32_t * sk = (uint32_t *) (mla_sm + MC * 8);
    float * sq = (float *) (mla_sm + MC * 8 + MC * MW * 4);
    float * sp = sq + H * MD;
    const int64_t h = a.ne[1], t = blockIdx.y, h0 = (int64_t) blockIdx.x * H;
    const int64_t n_tok = a.params[0], block = a.params[2];
    const float scale = pf(a, 1);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int64_t s = t / n_tok;
    const char * cache = S(a, 1);
    const int64_t cnb1 = a.src_nb[1][1];
    const int32_t * table = (const int32_t *) (S(a, 2) + s * a.src_nb[2][1]);
    const int32_t * sel = (const int32_t *) (S(a, 3) + t * a.src_nb[3][1]);
    const bool vec = ((uintptr_t) cache | (uintptr_t) cnb1) % 16 == 0;
    for (int i = tid; i < H * MD; i += 32 * H) {
        const int hh = i / MD, c = i % MD;
        sq[i] = ((const float *) (S(a, 0) + t * a.src_nb[0][2] + (h0 + hh) * a.src_nb[0][1]))[c];
    }
    const int count = sel[0];
    float m = -INFINITY, lsum = 0.0f;
    float2 acc[MD / 64];
#pragma unroll
    for (int r = 0; r < MD / 64; r++) {
        acc[r] = make_float2(0.0f, 0.0f);
    }
    for (int c0 = 0; c0 < count; c0 += MC) {
        const int nc = min(MC, count - c0);
        __syncthreads();
        // The chunk's row offsets once, then its rows with MU 16-byte loads in flight per thread
        // (a word at a time, each behind its sel and table reads, took ~20-40 us a chunk).
        if (tid < nc) {
            const int64_t pos = sel[1 + c0 + tid];
            srow[tid] = ((int64_t) table[pos / block] * block + pos % block) * cnb1;
        }
        __syncthreads();
        if (vec) {
            constexpr int MU = 8, Q = MD / 8;
            const int total = nc * Q;
            for (int i0 = tid; i0 < total; i0 += 32 * H * MU) {
                uint4 v[MU];
#pragma unroll
                for (int u = 0; u < MU; u++) {
                    const int i = i0 + u * 32 * H;
                    if (i < total) {
                        v[u] = ((const uint4 *) (cache + srow[i / Q]))[i % Q];
                    }
                }
#pragma unroll
                for (int u = 0; u < MU; u++) {
                    const int i = i0 + u * 32 * H;
                    if (i < total) {
                        uint32_t * d = sk + (i / Q) * MW + 4 * (i % Q);
                        d[0] = v[u].x;
                        d[1] = v[u].y;
                        d[2] = v[u].z;
                        d[3] = v[u].w;
                    }
                }
            }
        } else {
            for (int i = tid; i < nc * (MD / 2); i += 32 * H) {
                const int rr = i / (MD / 2), w = i % (MD / 2);
                sk[rr * MW + w] = ((const uint32_t *) (cache + srow[rr]))[w];
            }
        }
        __syncthreads();
        const float * q = sq + warp * MD;
        float sc = -INFINITY;
        if (lane < nc) {
            float dot = 0.0f;
            const uint32_t * kr = sk + lane * MW;
            const float2 * q2 = (const float2 *) q;
#pragma unroll 8
            for (int w = 0; w < MD / 2; w++) {
                const uint32_t u = kr[w];
                const float2 kv = __half22float2(*(const __half2 *) &u);
                const float2 qq = q2[w];
                dot += qq.x * kv.x + qq.y * kv.y;
            }
            sc = dot * scale;
        }
        float cm = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, o));
        }
        const float mn = fmaxf(m, cm);
        const float alpha = expf(m - mn);
        const float p = lane < nc ? expf(sc - mn) : 0.0f;
        lsum = lsum * alpha + warp_sum(p);
        m = mn;
        sp[warp * MC + lane] = p;
        __syncwarp();
        // Each column's sums in position order as before; positions outer (#575), so each weight
        // is read once for the lane's 16 columns.
#pragma unroll
        for (int r = 0; r < MD / 64; r++) {
            acc[r] = make_float2(acc[r].x * alpha, acc[r].y * alpha);
        }
        for (int pp = 0; pp < nc; pp++) {
            const float w = sp[warp * MC + pp];
            const uint32_t * kr = sk + pp * MW + lane;
#pragma unroll
            for (int r = 0; r < MD / 64; r++) {
                const uint32_t u = kr[32 * r];
                const float2 kv = __half22float2(*(const __half2 *) &u);
                acc[r].x += w * kv.x;
                acc[r].y += w * kv.y;
            }
        }
    }
    const float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    float2 * out = (float2 *) ((float *) a.dst + (t * h + h0 + warp) * MD);
#pragma unroll
    for (int r = 0; r < MD / 64; r++) {
        out[lane + 32 * r] = make_float2(acc[r].x * inv, acc[r].y * inv);
    }
}

// The GB10's prompt steps (#571): mla_attn_kernel's sums, register-tiled, a block of MT_T threads
// per (token, MT_H heads). Scores: a warp per 2 heads, lane p scoring position p of both from
// 8-byte row loads (rows padded to MT_W words: the lanes' loads miss each other's banks) and
// 16-byte query loads. Values: a thread per 4 heads x 4 words, the 4 heads' weights in one load.
// The next chunk's rows load into registers while this one is computed. Every sum has the plain
// kernel's terms in its order (the same chunks, the same online softmax): the same bits.
constexpr int MT_H = 16, MT_T = 256, MT_W = MD / 2 + 2, MT_U = MC * (MD / 8) / MT_T;

constexpr int mla_tiled_smem() {
    return MC * MT_W * 4 + MT_H * MD * 4 + MC * MT_H * 4 + 2 * MT_H * 4;
}

__global__ void __launch_bounds__(MT_T, 1) mla_tiled_kernel(const A a) {
    pdl_wait();
    extern __shared__ __align__(16) char mla_sm[];
    uint32_t * sk = (uint32_t *) mla_sm;
    float * sq = (float *) (mla_sm + MC * MT_W * 4);
    float * sp = sq + MT_H * MD;  // [position][head]
    float * sa = sp + MC * MT_H;  // per head: this chunk's alpha; then 1 / lsum
    const int64_t h = a.ne[1], t = blockIdx.y, h0 = (int64_t) blockIdx.x * MT_H;
    const int64_t n_tok = a.params[0], block = a.params[2];
    const float scale = pf(a, 1);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int64_t s = t / n_tok;
    const char * cache = S(a, 1);
    const int64_t cnb1 = a.src_nb[1][1];
    const int32_t * table = (const int32_t *) (S(a, 2) + s * a.src_nb[2][1]);
    const int32_t * sel = (const int32_t *) (S(a, 3) + t * a.src_nb[3][1]);
    for (int i = tid; i < MT_H * MD; i += MT_T) {
        const int hh = i / MD, c = i % MD;
        sq[i] = ((const float *) (S(a, 0) + t * a.src_nb[0][2] + (h0 + hh) * a.src_nb[0][1]))[c];
    }
    const int count = sel[0];
    // staging: thread tid moves 16-byte piece qd of rows r0, r0 + 4, ..
    const int qd = tid % (MD / 8), r0 = tid / (MD / 8);
    constexpr int RS = MT_T / (MD / 8);
    uint4 v[MT_U];
    auto fetch = [&](int c0) {
        const int nc = min(MC, count - c0);
#pragma unroll
        for (int u = 0; u < MT_U; u++) {
            const int r = r0 + RS * u;
            if (r < nc) {
                const int64_t pos = sel[1 + c0 + r];
                v[u] = ((const uint4 *) (cache + ((int64_t) table[pos / block] * block + pos % block) * cnb1))[qd];
            }
        }
    };
    // values: heads 4 hg .. 4 hg + 3, words 2 wg, 2 wg + 1, MD / 4 + 2 wg, MD / 4 + 2 wg + 1
    const int hg = tid / (MD / 8), wg = tid % (MD / 8);
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    float2 acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            acc[j][k] = make_float2(0.0f, 0.0f);
        }
    }
    if (count > 0) {
        fetch(0);
    }
    for (int c0 = 0; c0 < count; c0 += MC) {
        const int nc = min(MC, count - c0);
        __syncthreads();
#pragma unroll
        for (int u = 0; u < MT_U; u++) {
            const int r = r0 + RS * u;
            if (r < nc) {
                uint2 * d = (uint2 *) (sk + r * MT_W + 4 * qd);
                d[0] = make_uint2(v[u].x, v[u].y);
                d[1] = make_uint2(v[u].z, v[u].w);
            }
        }
        __syncthreads();
        if (c0 + MC < count) {
            fetch(c0 + MC);
        }
        {
            const float4 * qa = (const float4 *) (sq + 2 * warp * MD);
            const float4 * qb = qa + MD / 4;
            const uint2 * kr = (const uint2 *) (sk + lane * MT_W);
            float da = 0.0f, db = 0.0f;
#pragma unroll 4
            for (int j = 0; j < MD / 4; j++) {
                const uint2 u = kr[j];
                const float2 k0 = __half22float2(*(const __half2 *) &u.x);
                const float2 k1 = __half22float2(*(const __half2 *) &u.y);
                const float4 xa = qa[j], xb = qb[j];
                da += xa.x * k0.x + xa.y * k0.y;
                da += xa.z * k1.x + xa.w * k1.y;
                db += xb.x * k0.x + xb.y * k0.y;
                db += xb.z * k1.x + xb.w * k1.y;
            }
            const float sca = lane < nc ? da * scale : -INFINITY;
            const float scb = lane < nc ? db * scale : -INFINITY;
            float cma = sca, cmb = scb;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                cma = fmaxf(cma, __shfl_xor_sync(0xffffffffu, cma, o));
                cmb = fmaxf(cmb, __shfl_xor_sync(0xffffffffu, cmb, o));
            }
            const float mna = fmaxf(ma, cma), mnb = fmaxf(mb, cmb);
            const float alpha_a = expf(ma - mna), alpha_b = expf(mb - mnb);
            const float pa = lane < nc ? expf(sca - mna) : 0.0f;
            const float pb = lane < nc ? expf(scb - mnb) : 0.0f;
            la = la * alpha_a + warp_sum(pa);
            lb = lb * alpha_b + warp_sum(pb);
            ma = mna;
            mb = mnb;
            *(float2 *) (sp + lane * MT_H + 2 * warp) = make_float2(pa, pb);
            if (lane == 0) {
                *(float2 *) (sa + 2 * warp) = make_float2(alpha_a, alpha_b);
            }
        }
        __syncthreads();
        const float4 al = ((const float4 *) sa)[hg];
        const float alj[4] = {al.x, al.y, al.z, al.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                acc[j][k] = make_float2(acc[j][k].x * alj[j], acc[j][k].y * alj[j]);
            }
        }
        for (int pp = 0; pp < nc; pp++) {
            const uint32_t * kr = sk + pp * MT_W;
            const uint2 u0 = ((const uint2 *) kr)[wg], u1 = ((const uint2 *) (kr + MD / 4))[wg];
            float2 kv[4];
            kv[0] = __half22float2(*(const __half2 *) &u0.x);
            kv[1] = __half22float2(*(const __half2 *) &u0.y);
            kv[2] = __half22float2(*(const __half2 *) &u1.x);
            kv[3] = __half22float2(*(const __half2 *) &u1.y);
            const float4 w4 = ((const float4 *) (sp + pp * MT_H))[hg];
            const float wj[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    acc[j][k].x += wj[j] * kv[k].x;
                    acc[j][k].y += wj[j] * kv[k].y;
                }
            }
        }
    }
    if (lane == 0) {
        sa[MT_H + 2 * warp] = la > 0.0f ? 1.0f / la : 0.0f;
        sa[MT_H + 2 * warp + 1] = lb > 0.0f ? 1.0f / lb : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float inv = sa[MT_H + 4 * hg + j];
        float4 * out = (float4 *) ((float *) a.dst + (t * h + h0 + 4 * hg + j) * MD);
        out[wg] = make_float4(acc[j][0].x * inv, acc[j][0].y * inv, acc[j][1].x * inv, acc[j][1].y * inv);
        out[MD / 8 + wg] = make_float4(acc[j][2].x * inv, acc[j][2].y * inv, acc[j][3].x * inv, acc[j][3].y * inv);
    }
}

// The GB10's choice over mla_tiled_kernel (#862): its bits, score and value warps a chunk apart.
#include "sm80/mla_gb10.cuh"

// Opt-in (#575, `legion serve --glm-mla-tc`), sm_80+: mla_attn's prompt steps on tensor cores. A
// block of 8 warps per (token, MQ_H heads), a warp per 16 heads x 128 latent columns: its q
// columns rounded to F16 once into mma fragments (registers), each chunk of MC selected rows (F16
// already) copied to shared memory by cp.async a chunk ahead, the warp's partial scores q k^T over
// its 128 columns by mma.sync m16n8k16 (F32 sums), the 4 partials summed in a fixed order, the
// online softmax in F32 with the weights rounded to F16, and P V for the warp's columns by
// mma.sync again (F32 sums). Not the default kernels' bits (q and the weights in F16, other sums).
constexpr int MQ_H = 32, MQ_T = 256, MQ_L = MD + 8, MQ_SL = MC + 4, MQ_PL = MC + 8, MQ_U = MC * (MD / 8) / MQ_T;

constexpr int mla_tc_smem() {
    return 2 * MC * MQ_L * 2 + 4 * MQ_H * MQ_SL * 4 + MQ_H * MQ_PL * 2 + 2 * MQ_H * 4;
}

__device__ __forceinline__ void mq_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mq_ldsm4_t(uint32_t addr, uint32_t (&r)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t mq_h2(float2 v) {
    const __half2 h = __floats2half2_rn(v.x, v.y);
    return *(const uint32_t *) &h;
}

__global__ void __launch_bounds__(MQ_T, 1) mla_tc_kernel(const A a) {
    pdl_wait();
    extern __shared__ __align__(16) char mla_sm[];
    __half * sk = (__half *) mla_sm;                        // [2][MC][MQ_L] the chunks' rows
    float * ss = (float *) (sk + 2 * MC * MQ_L);            // [4][MQ_H][MQ_SL] partial scores
    __half * sp = (__half *) (ss + 4 * MQ_H * MQ_SL);       // [MQ_H][MQ_PL] weights
    float * sa = (float *) (sp + MQ_H * MQ_PL);             // [MQ_H] this chunk's alpha, then 1 / sum
    const int64_t h = a.ne[1], t = blockIdx.y, h0 = (int64_t) blockIdx.x * MQ_H;
    const int64_t n_tok = a.params[0], block = a.params[2];
    const float scale = pf(a, 1);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int64_t s = t / n_tok;
    const char * cache = S(a, 1);
    const int64_t cnb1 = a.src_nb[1][1];
    const int32_t * table = (const int32_t *) (S(a, 2) + s * a.src_nb[2][1]);
    const int32_t * sel = (const int32_t *) (S(a, 3) + t * a.src_nb[3][1]);
    // heads 16 wm .. 16 wm + 15, latent columns 128 wd .. 128 wd + 127
    const int wm = warp / 4, wd = warp % 4, g = lane / 4, tq = lane % 4;
    uint32_t fq[8][4];
    {
        const float * q0 = (const float *) (S(a, 0) + t * a.src_nb[0][2] + (h0 + 16 * wm + g) * a.src_nb[0][1]);
        const float * q1 = (const float *) ((const char *) q0 + 8 * a.src_nb[0][1]);
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            const int c = 128 * wd + 16 * kk + 2 * tq;
            fq[kk][0] = mq_h2(*(const float2 *) (q0 + c));
            fq[kk][1] = mq_h2(*(const float2 *) (q1 + c));
            fq[kk][2] = mq_h2(*(const float2 *) (q0 + c + 8));
            fq[kk][3] = mq_h2(*(const float2 *) (q1 + c + 8));
        }
    }
    const int count = sel[0];
    // staging: thread tid copies 16-byte piece qd of rows r0, r0 + RS, .. (zeros past the selection)
    const int qd = tid % (MD / 8), r0 = tid / (MD / 8);
    constexpr int RS = MQ_T / (MD / 8);
    int64_t off[MQ_U];
    auto offsets = [&](int c0) {
#pragma unroll
        for (int u = 0; u < MQ_U; u++) {
            const int r = c0 + r0 + RS * u;
            if (r < count) {
                const int64_t pos = sel[1 + r];
                off[u] = ((int64_t) table[pos / block] * block + pos % block) * cnb1 + 16 * qd;
            } else {
                off[u] = -1;
            }
        }
    };
    auto issue = [&](int buf) {
#pragma unroll
        for (int u = 0; u < MQ_U; u++) {
            xcp16(sk + (buf * MC + r0 + RS * u) * MQ_L + 8 * qd, cache + (off[u] < 0 ? 0 : off[u]), off[u] < 0 ? 0 : 16);
        }
        xcp_commit();
    };
    // softmax: heads 4 warp .. 4 warp + 3, lane = position
    float mx[4], lsum[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        mx[j] = -INFINITY;
        lsum[j] = 0.0f;
    }
    float acc[16][4];
#pragma unroll
    for (int b = 0; b < 16; b++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[b][e] = 0.0f;
        }
    }
    const uint32_t sk0 = (uint32_t) __cvta_generic_to_shared(sk), sp0 = (uint32_t) __cvta_generic_to_shared(sp);
    if (count > 0) {
        offsets(0);
        issue(0);
        offsets(MC);
    }
    for (int c0 = 0, buf = 0; c0 < count; c0 += MC, buf ^= 1) {
        const int nc = min(MC, count - c0);
        xcp_wait_all();
        __syncthreads();
        if (c0 + MC < count) {
            issue(buf ^ 1);
            offsets(c0 + 2 * MC);
        }
        const uint32_t kb = sk0 + 2 * buf * MC * MQ_L;
        {
            // the warp's partial scores: positions 8 nt .., its 128 columns
            float d[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    d[nt][e] = 0.0f;
                }
            }
#pragma unroll
            for (int kk = 0; kk < 8; kk += 2) {
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    uint32_t fb[4];
                    xldsm4(kb + 2 * ((8 * nt + lane % 8) * MQ_L + 128 * wd + 16 * kk + 8 * (lane / 8)), fb);
                    mq_mma(d[nt], fq[kk], fb[0], fb[1]);
                    mq_mma(d[nt], fq[kk + 1], fb[2], fb[3]);
                }
            }
            float * sw = ss + (wd * MQ_H + 16 * wm + g) * MQ_SL + 2 * tq;
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                *(float2 *) (sw + 8 * nt) = make_float2(d[nt][0], d[nt][1]);
                *(float2 *) (sw + 8 * MQ_SL + 8 * nt) = make_float2(d[nt][2], d[nt][3]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int hh = 4 * warp + j;
            const float * sr = ss + hh * MQ_SL + lane;
            const float sc = lane < nc ? ((sr[0] + sr[MQ_H * MQ_SL]) + (sr[2 * MQ_H * MQ_SL] + sr[3 * MQ_H * MQ_SL])) * scale
                                       : -INFINITY;
            float cm = sc;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, o));
            }
            const float mn = fmaxf(mx[j], cm);
            const float alpha = expf(mx[j] - mn);
            const __half ph = __float2half_rn(lane < nc ? expf(sc - mn) : 0.0f);
            lsum[j] = lsum[j] * alpha + warp_sum(__half2float(ph));
            mx[j] = mn;
            sp[hh * MQ_PL + lane] = ph;
            if (lane == 0) {
                sa[hh] = alpha;
            }
        }
        __syncthreads();
        {
            const float al0 = sa[16 * wm + g], al1 = sa[16 * wm + g + 8];
#pragma unroll
            for (int b = 0; b < 16; b++) {
                acc[b][0] *= al0;
                acc[b][1] *= al0;
                acc[b][2] *= al1;
                acc[b][3] *= al1;
            }
#pragma unroll
            for (int kp = 0; kp < MC; kp += 16) {
                uint32_t fp[4];
                xldsm4(sp0 + 2 * ((16 * wm + lane % 16) * MQ_PL + kp + 8 * (lane / 16)), fp);
                // V rows kp + (lane % 8) + 8 ((lane / 8) % 2), columns + 8 (lane / 16): two n-tiles a load
                const int vr = kp + lane % 8 + 8 * ((lane / 8) % 2), vc = 128 * wd + 8 * (lane / 16);
#pragma unroll
                for (int b = 0; b < 16; b += 2) {
                    uint32_t fv[4];
                    mq_ldsm4_t(kb + 2 * (vr * MQ_L + vc + 8 * b), fv);
                    mq_mma(acc[b], fp, fv[0], fv[1]);
                    mq_mma(acc[b + 1], fp, fv[2], fv[3]);
                }
            }
        }
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            sa[MQ_H + 4 * warp + j] = lsum[j] > 0.0f ? 1.0f / lsum[j] : 0.0f;
        }
    }
    __syncthreads();
    const float inv0 = sa[MQ_H + 16 * wm + g], inv1 = sa[MQ_H + 16 * wm + g + 8];
    float * o0 = (float *) a.dst + (t * h + h0 + 16 * wm + g) * MD + 128 * wd + 2 * tq;
    float * o1 = o0 + 8 * MD;
#pragma unroll
    for (int b = 0; b < 16; b++) {
        *(float2 *) (o0 + 8 * b) = make_float2(acc[b][0] * inv0, acc[b][1] * inv0);
        *(float2 *) (o1 + 8 * b) = make_float2(acc[b][2] * inv1, acc[b][3] * inv1);
    }
}

// Loads a thread keeps in flight in argmax_kernel and pmax_kernel: a column (the vocabulary, ~155K
// logits) on one CTA is latency bound with one load at a time. Each thread still visits its
// elements i, i + blockDim.x, ... in that order: the same result, the same bits.
constexpr int VOCAB_UNROLL = 16;

__global__ void argmax_kernel(const A a) {
    pdl_wait();
    const int64_t v = a.src_ne[0][0], t = blockIdx.x;
    const float * x = (const float *) a.src[0] + t * v;
    float bv = -INFINITY;
    int64_t bi = -1;
    for (int64_t i0 = threadIdx.x; i0 < v; i0 += VOCAB_UNROLL * blockDim.x) {
        float xs[VOCAB_UNROLL];
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            xs[u]           = i < v ? x[i] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            if (i < v && (bi < 0 || xs[u] > bv)) {
                bv = xs[u];
                bi = i;
            }
        }
    }
    __shared__ float vs[32];
    __shared__ int64_t is[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        vs[warp] = bv;
        is[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) blockDim.x / 32; w++) {
            if (is[w] >= 0 && (bi < 0 || vs[w] > bv || (vs[w] == bv && is[w] < bi))) {
                bv = vs[w];
                bi = is[w];
            }
        }
        ((int32_t *) a.dst)[t] = (int32_t) bi;
    }
}

// argmax_kernel over ARGMAX_SPLIT CTAs a column (ops::argmax_wide, params[0], #606): one CTA read
// a 155K vocabulary at ~3 GB/s, ~190 us a token at the end of every step. dst is I32 [n, 1 + 2 S]:
// row 0 the indices, then each column's S chunk bests, values (as floats) then indices. Each
// chunk takes the first of its largest entries, and so does argmax_fin over the chunks in their
// order: the column's first largest entry, as argmax_kernel finds it (logits without NaNs).
constexpr int ARGMAX_SPLIT = 32;
static_assert(ARGMAX_SPLIT == 32, "argmax_fin_kernel: a chunk a lane");

__device__ __forceinline__ bool argmax_better(float ov, int64_t oi, float bv, int64_t bi) {
    return oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi));
}

__global__ void __launch_bounds__(256) argmax_part_kernel(const A a) {
    pdl_wait();
    const int64_t v = a.src_ne[0][0], n = a.ne[0], t = blockIdx.y, c = blockIdx.x;
    const int64_t len = (v + ARGMAX_SPLIT - 1) / ARGMAX_SPLIT, i1 = min(v, (c + 1) * len);
    const float * x = (const float *) a.src[0] + t * v;
    float bv = -INFINITY;
    int64_t bi = -1;
    for (int64_t i0 = c * len + threadIdx.x; i0 < i1; i0 += VOCAB_UNROLL * blockDim.x) {
        float xs[VOCAB_UNROLL];
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            xs[u]           = i < i1 ? __ldg(x + i) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            if (i < i1 && (bi < 0 || xs[u] > bv)) {
                bv = xs[u];
                bi = i;
            }
        }
    }
    __shared__ float vs[8];
    __shared__ int64_t is[8];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (argmax_better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        vs[warp] = bv;
        is[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) blockDim.x / 32; w++) {
            if (argmax_better(vs[w], is[w], bv, bi)) {
                bv = vs[w];
                bi = is[w];
            }
        }
        int32_t * d = (int32_t *) a.dst;
        ((float *) d)[n + t * ARGMAX_SPLIT + c] = bv;
        d[n + n * ARGMAX_SPLIT + t * ARGMAX_SPLIT + c] = (int32_t) bi;
    }
}

__global__ void __launch_bounds__(32) argmax_fin_kernel(const A a) {
    pdl_wait();
    const int64_t n = a.ne[0], t = blockIdx.x;
    int32_t * d = (int32_t *) a.dst;
    const int c = threadIdx.x;
    float bv = ((const float *) d)[n + t * ARGMAX_SPLIT + c];
    int64_t bi = d[n + n * ARGMAX_SPLIT + t * ARGMAX_SPLIT + c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int64_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (argmax_better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (c == 0) {
        d[t] = (int32_t) bi;
    }
}

// Each column's largest softmax probability, 1 / sum(exp(x - max)).
__global__ void pmax_kernel(const A a) {
    pdl_wait();
    const int64_t v = a.src_ne[0][0], t = blockIdx.x;
    const float * x = (const float *) a.src[0] + t * v;
    __shared__ float ms[32];
    float m = -INFINITY;
    for (int64_t i0 = threadIdx.x; i0 < v; i0 += VOCAB_UNROLL * blockDim.x) {
        float xs[VOCAB_UNROLL];
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            xs[u]           = i < v ? x[i] : -INFINITY;
        }
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            m = fmaxf(m, xs[u]);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    }
    if (threadIdx.x % 32 == 0) {
        ms[threadIdx.x / 32] = m;
    }
    __syncthreads();
    for (int w = 0; w < (int) blockDim.x / 32; w++) {
        m = fmaxf(m, ms[w]);
    }
    double sum = 0.0;
    for (int64_t i0 = threadIdx.x; i0 < v; i0 += VOCAB_UNROLL * blockDim.x) {
        float xs[VOCAB_UNROLL];
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            const int64_t i = i0 + u * blockDim.x;
            xs[u]           = i < v ? x[i] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < VOCAB_UNROLL; u++) {
            if (i0 + u * blockDim.x < v) {
                sum += (double) expf(xs[u] - m);
            }
        }
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        ((float *) a.dst)[t] = (float) (1.0 / sum);
    }
}

// MTP input: [rms(e) * enorm; rms(h) * hnorm] per token, a block per (token, half).
__global__ void nextn_in_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.src_ne[0][0], t = blockIdx.x, half = blockIdx.y;
    const float * x = (const float *) a.src[half] + t * d;
    const float * w = (const float *) a.src[2 + half];
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        sum += (double) x[i] * x[i];
    }
    sum = block_sum(sum);
    const float s = 1.0f / sqrtf((float) (sum / d) + pf(a, 0));
    float * o = (float *) a.dst + t * 2 * d + half * d;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] = x[i] * s * w[i];
    }
}

__global__ void get_cols_kernel(const A a) {
    pdl_wait();
    const int64_t d = a.ne[0], j = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const int64_t r = ((const int32_t *) a.src[1])[j];
    ((float *) a.dst)[j * d + i] = ((const float *) (S(a, 0) + r * a.src_nb[0][1]))[i];
}

__global__ void add_kernel(const float * x, const float * y, float * out, int64_t n) {
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

int64_t elems(const A & a, int i) { return a.src_ne[i][0] * a.src_ne[i][1] * a.src_ne[i][2] * a.src_ne[i][3]; }

}  // namespace

extern "C" int legion_glm_launch(int op, const legion_glm_args * ap, void * stream) {
    const A & a = *ap;
    const cudaStream_t st = (cudaStream_t) stream;
    const int64_t n_out = a.ne[0] * a.ne[1] * a.ne[2] * a.ne[3];
    if (n_out == 0) {
        return 0;
    }
    switch (op) {
        case EMBED:
            if (a.src_type[0] != T_F32 && a.src_type[0] != T_F16 && a.src_type[0] != T_BF16 &&
                a.src_type[0] != T_Q8_0 && a.src_type[0] != T_Q6_K && a.src_type[0] != T_Q4_K &&
                a.src_type[0] != T_Q5_K) {
                return 1;
            }
            if (a.src_type[0] == T_Q5_K) {
                launch_pdl(embed_kernel<true>, (unsigned) a.ne[2], 256, 0, st, a);
            } else {
                launch_pdl(embed_kernel<false>, (unsigned) a.ne[2], 256, 0, st, a);
            }
            break;
        case LINEAR: {
            const int rc = launch_linear(a, st);
            if (rc != 0) {
                return rc;
            }
            break;
        }
        case BF16:
            launch_pdl(bf16_kernel, blocks(n_out, 256), 256, 0, st, (const float *) a.src[0], (uint16_t *) a.dst, n_out);
            break;
        case F16:
            launch_pdl(f16_kernel, blocks(n_out, 256), 256, 0, st, (const float *) a.src[0], (__half *) a.dst, n_out);
            break;
        case WF16:
            if (a.src_type[0] != T_Q6_K || a.src_ne[0][0] % 256 != 0 ||
                ((int64_t) ((uintptr_t) a.src[0] | a.src_nb[0][1] | a.src_nb[0][2]) % 2) != 0 ||
                ((uintptr_t) a.dst % 16) != 0) {
                return 1;
            }
            launch_pdl(wf16_q6k_kernel, blocks(n_out / 16, 256), 256, 0, st, a);
            break;
        case LINEAR_SET:
            return launch_linear_set(a, st);
        case GLU_MV:
            return launch_glu_mv(a, st);
        case RMS_NORM:
            launch_pdl(rms_norm_kernel, (unsigned) (n_out / a.ne[0]), 256, 0, st, a);
            break;
        case LAYER_NORM:
            launch_pdl(layer_norm_kernel, (unsigned) (elems(a, 0) / a.src_ne[1][0]), 256, 0, st, a);
            break;
        case HC_PRE: {
            const int64_t e = a.src_ne[0][0], hc = a.src_ne[0][1];
            if (hc > HC_MAX || (e * hc) % 32 != 0 || e <= (2 + hc) * hc || a.params[2] < 1 ||
                (a.n_src >= 5 && a.src_type[4] != T_F32) ||
                (a.src_type[1] != T_F32 && a.src_type[1] != T_F16 && a.src_type[1] != T_BF16 &&
                 a.src_type[1] != T_Q8_0 && a.src_type[1] != T_Q6_K && a.src_type[1] != T_Q4_K)) {
                return 1;
            }
            // params[3]: the GB10 kernels (ops::set_gb10_linear), for prompt steps
            const int ht = a.src_type[1];
            if (a.params[5]) {  // hc_mix_split_kernel's scratch (hc_pre_norm, a few tokens)
                if (a.n_src < 5 || ht != T_F32 || a.src_nb[1][1] % 4 != 0 || (e * hc) % 32 != 0) {
                    return 1;
                }
                const dim3 grid((unsigned) ((2 + hc) * hc * HC_MIX_W), (unsigned) a.src_ne[0][2]);
                if (e * hc >= HC_SPLIT_UW * 32 * HC_MIX_W) {
                    launch_pdl(hc_mix_split_kernel<HC_SPLIT_UW>, grid, 32, 0, st, a);
                } else {
                    launch_pdl(hc_mix_split_kernel<HC_SPLIT_U>, grid, 32, 0, st, a);
                }
            } else if (a.params[3] && (ht == T_F32 || ht == T_BF16 || ht == T_F16) && (2 + hc) * hc <= HC_MR &&
                a.src_ne[0][2] >= 64 && a.src_nb[1][1] % 4 == 0) {
                const unsigned g = (unsigned) ((a.src_ne[0][2] + HC_MT - 1) / HC_MT);
                const bool full = (2 + hc) * hc == HC_MR;  // GLM's 4 streams
                void (*k)(const A) = full ? &hc_mix_tiled_kernel<T_F32, HC_MR> : &hc_mix_tiled_kernel<T_F32, 0>;
                if (ht == T_BF16) {
                    k = full ? &hc_mix_tiled_kernel<T_BF16, HC_MR> : &hc_mix_tiled_kernel<T_BF16, 0>;
                } else if (ht == T_F16) {
                    k = full ? &hc_mix_tiled_kernel<T_F16, HC_MR> : &hc_mix_tiled_kernel<T_F16, 0>;
                }
                launch_pdl(k, g, 256, 0, st, a);
            } else {
                launch_pdl(hc_mix_kernel, dim3((unsigned) ((2 + hc) * hc), (unsigned) a.src_ne[0][2]), 256, 0, st, a);
            }
            const unsigned n = (unsigned) a.src_ne[0][2];
            if (a.n_src >= 5) {  // the norm fused (#606)
                if (hc == 4 && e <= HC_WIDE_E) {
                    launch_pdl(hc_finish_wide_kernel, 2 * n, HC_WIDE_T, 0, st, a);
                } else {
                    launch_pdl(hc_finish_kernel, 2 * n, 256, 0, st, a);
                }
                break;
            }
            launch_pdl(hc_weights_kernel, n, HC_MAX * HC_MAX, 0, st, a);
            launch_pdl(hc_pre_kernel, dim3(blocks(e, 256), n), 256, 0, st, a);
            break;
        }
        case HC_POST:
            launch_pdl(hc_post_kernel, dim3(blocks(a.ne[0], 256), (unsigned) a.ne[2]), 256, 0, st, a);
            break;
        case HC_HEAD: {
            const size_t smem = (size_t) a.ne[0] * 4;
            if (smem > 48 * 1024) {
                return 1;
            }
            launch_pdl(hc_head_kernel, (unsigned) a.ne[1], 256, smem, st, a);
            break;
        }
        case KDA_CONV:
            // params[3]: the GB10 kernels (ops::set_gb10_linear), for prompt steps
            if (a.params[4]) {  // q, k and v in one launch (#676)
                launch_pdl(kda_conv_kernel, dim3(blocks(a.ne[0] * (a.ne[1] / a.params[1]), 128), 3), 128, 0, st, a);
            } else if (a.params[3] && !a.params[2] && a.params[1] >= 64 && a.ne[1] / a.params[1] <= 65535) {
                const int64_t ns = a.ne[1] / a.params[1];
                launch_pdl(kda_conv_tok_kernel, dim3(blocks(a.ne[0], 128), blocks(a.params[1], KC_T), (unsigned) ns), 128,
                           0, st, a);
                launch_pdl(kda_conv_state_kernel, blocks(a.ne[0] * ns, 128), 128, 0, st, a);
            } else {
                launch_pdl(kda_conv_kernel, blocks(a.ne[0] * (a.ne[1] / a.params[1]), 128), 128, 0, st, a);
            }
            break;
        case KDA_GATE:
            launch_pdl(kda_gate_kernel, blocks(n_out, 256), 256, 0, st, a, n_out);
            break;
        case SIGMOID:
            launch_pdl(sigmoid_kernel, blocks(n_out, 256), 256, 0, st, (const float *) a.src[0], (float *) a.dst, n_out);
            break;
        case KDA_SCAN:
            if (a.ne[0] != KD) {
                return 1;
            }
            // params[2]: the GB10 kernels (ops::set_gb10_linear), for prompt steps
            if (a.params[2] && !a.params[1] && a.params[0] >= 64) {
                launch_pdl(kda_scan_rows_kernel,
                           dim3((unsigned) (a.ne[1] * (KD / KS_ROWS)), (unsigned) (a.ne[2] / a.params[0])), 32 * KS_W,
                           0, st, a);
            } else {
                launch_pdl(kda_scan_kernel, dim3((unsigned) a.ne[1], (unsigned) (a.ne[2] / a.params[0])), KD * KL, 0,
                           st, a);
            }
            break;
        case GATED_RMS:
            launch_pdl(gated_rms_kernel, (unsigned) (n_out / a.src_ne[1][0]), 128, 0, st, a);
            break;
        case GLU:
            launch_pdl(glu_kernel, blocks(n_out, 256), 256, 0, st, (const float *) a.src[0], (const float *) a.src[1],
                       (float *) a.dst, n_out, host_f32(a.params[0]));
            break;
        case ROUTER_IDS:
            if (a.src_ne[1][0] > 32 * RT_MAX || a.ne[0] > a.src_ne[1][0]) {
                return 1;
            }
            launch_pdl(router_ids_kernel, blocks(a.ne[1], 4), 128, 0, st, a);
            break;
        case ROUTER_W:
            launch_pdl(router_w_kernel, blocks(a.ne[1], 128), 128, 0, st, a);
            break;
        case WEIGHTED_SUM:
            launch_pdl(weighted_sum_kernel, dim3(blocks(a.ne[0], 256), (unsigned) a.ne[1]), 256, 0, st, a);
            break;
        case CACHE_ROWS:
            if (a.src_ne[1][1] > 0) {
                launch_pdl(cache_rows_kernel, dim3(blocks(a.src_ne[1][0], 256), (unsigned) a.src_ne[1][1]), 256, 0, st,
                           a);
            }
            break;
        case POOL_KEYS:
            if (a.src_ne[3][1] > KPOOL_MAX) {
                return 1;
            }
            launch_pdl(pool_keys_kernel, (unsigned) a.ne[1], 128, 0, st, a);
            break;
        case MLA_ATTN:
            if (a.ne[0] != MD || a.ne[1] % MH != 0 || a.src_ne[1][0] != MD || a.src_nb[1][1] % 4 != 0) {
                return 1;
            }
            // params[3]: the GB10 kernels (ops::set_gb10_linear), for prompt steps; params[4]: the
            // opt-in tensor-core kernel (ops::set_mla_tc) on sm_80+
            if (a.params[3] && a.params[4] && a.ne[1] % MQ_H == 0 && a.ne[2] >= 64 && dev_sm80() &&
                ((uintptr_t) a.src[0] | (uintptr_t) a.src_nb[0][1] | (uintptr_t) a.src_nb[0][2] |
                 (uintptr_t) a.src[1] | (uintptr_t) a.src_nb[1][1] | (uintptr_t) a.dst) % 16 == 0 &&
                a.nb[1] == MD * 4 && a.nb[2] == a.ne[1] * MD * 4) {
                constexpr int smem = mla_tc_smem();
                const cudaError_t e =
                    cudaFuncSetAttribute(mla_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (e != cudaSuccess) {
                    return (int) e;
                }
                launch_pdl(mla_tc_kernel, dim3((unsigned) (a.ne[1] / MQ_H), (unsigned) a.ne[2]), MQ_T, smem, st, a);
            } else if (a.params[3] && a.params[5] != 1 && a.ne[1] % MT_H == 0 && a.ne[2] >= 64 &&
                       ((uintptr_t) a.src[1] | (uintptr_t) a.src_nb[1][1] | (uintptr_t) a.dst) % 16 == 0 &&
                       a.nb[1] == MD * 4 && a.nb[2] == a.ne[1] * MD * 4 &&
                       (a.params[5] == 2 ? dev_sm80() : gb10_device(mla_gb_kernel))) {
                // params[5]: the prompt kernel (ops::MlaKernel): 0 mla_gb_kernel on the GB10 and
                // mla_tiled_kernel elsewhere, 1 mla_tiled_kernel, 2 mla_gb_kernel on any sm_80+
                // (the same bits); 3 comes with params[4] (mla_tc_kernel above, else as 0)
                constexpr int smem = mla_gb_smem();
                const cudaError_t e =
                    cudaFuncSetAttribute(mla_gb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (e != cudaSuccess) {
                    return (int) e;
                }
                launch_pdl(mla_gb_kernel, dim3((unsigned) (a.ne[1] / MT_H), (unsigned) a.ne[2]), MG_T, smem, st, a);
            } else if (a.params[3] && a.ne[1] % MT_H == 0 && a.ne[2] >= 64 &&
                ((uintptr_t) a.src[1] | (uintptr_t) a.src_nb[1][1] | (uintptr_t) a.dst) % 16 == 0 &&
                a.nb[1] == MD * 4 && a.nb[2] == a.ne[1] * MD * 4) {
                constexpr int smem = mla_tiled_smem();
                const cudaError_t e =
                    cudaFuncSetAttribute(mla_tiled_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (e != cudaSuccess) {
                    return (int) e;
                }
                launch_pdl(mla_tiled_kernel, dim3((unsigned) (a.ne[1] / MT_H), (unsigned) a.ne[2]), MT_T, smem, st, a);
            } else if (a.params[3] && a.ne[1] % MH_WIDE == 0 && a.ne[2] >= 64) {
                constexpr int smem = mla_smem<MH_WIDE>();
                const cudaError_t e = cudaFuncSetAttribute(mla_attn_kernel<MH_WIDE>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (e != cudaSuccess) {
                    return (int) e;
                }
                launch_pdl(mla_attn_kernel<MH_WIDE>, dim3((unsigned) (a.ne[1] / MH_WIDE), (unsigned) a.ne[2]),
                           32 * MH_WIDE, smem, st, a);
            } else {
                launch_pdl(mla_attn_kernel<MH>, dim3((unsigned) (a.ne[1] / MH), (unsigned) a.ne[2]), 32 * MH,
                           mla_smem<MH>(), st, a);
            }
            break;
        case JOIN_COLS: {
            const int64_t d = a.ne[0];
            int64_t col = 0;
            for (int i = 0; i < a.n_src; i++) {
                const int64_t cols = a.src_ne[i][1] * a.src_ne[i][2] * a.src_ne[i][3];
                if (cols == 0) {
                    continue;
                }
                const size_t pitch = cols == 1 ? (size_t) d * 4 : (size_t) a.src_nb[i][1];
                const cudaError_t e = cudaMemcpy2DAsync((float *) a.dst + col * d, d * 4, a.src[i], pitch, d * 4, cols,
                                                        cudaMemcpyDeviceToDevice, st);
                if (e != cudaSuccess) {
                    return (int) e;
                }
                col += cols;
            }
            break;
        }
        case ARGMAX:
            if (a.params[0]) {  // ops::argmax_wide (#606)
                if (a.params[0] != ARGMAX_SPLIT || a.ne[1] != 1 + 2 * ARGMAX_SPLIT) {
                    return 1;
                }
                launch_pdl(argmax_part_kernel, dim3(ARGMAX_SPLIT, (unsigned) a.ne[0]), 256, 0, st, a);
                launch_pdl(argmax_fin_kernel, (unsigned) a.ne[0], 32, 0, st, a);
                break;
            }
            launch_pdl(argmax_kernel, (unsigned) a.ne[0], 256, 0, st, a);
            break;
        case PMAX:
            launch_pdl(pmax_kernel, (unsigned) a.ne[0], 256, 0, st, a);
            break;
        case NEXTN_IN:
            launch_pdl(nextn_in_kernel, dim3((unsigned) a.ne[1], 2), 256, 0, st, a);
            break;
        case GET_COLS:
            launch_pdl(get_cols_kernel, dim3(blocks(a.ne[0], 256), (unsigned) a.ne[1]), 256, 0, st, a);
            break;
        case ADD:
            launch_pdl(add_kernel, blocks(n_out, 256), 256, 0, st, (const float *) a.src[0], (const float *) a.src[1],
                       (float *) a.dst, n_out);
            break;
        default:
            return 1;
    }
    return (int) cudaGetLastError();
}
