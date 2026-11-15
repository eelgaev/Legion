// Fused elementwise ops of the qwen35 / qwen4exp graphs: kernel bodies and thin launchers.
// Shapes, grids and the ops' semantics are in src/model/qwen35/fused.rs (with the CPU
// references).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "nvfp4_plan.cuh"
#include "router_select.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

extern "C" {

// conv_silu: out[c, t, s] = silu(sum_k in[t + k, c, s] * w[k, c]) over in = [state | x] along
// time, and hist = the last `keep` inputs (d_conv - 1: the next step's state; d_conv - 1 + nt:
// all of [state | x], an MTP verify step's window).
struct legion_conv_silu {
    const float * x;      // [C, nt, ns]: element (c, t, s) at x + c + t * x_ld + s * x_sd (floats)
    int64_t x_ld, x_sd;
    const float * state;  // [d - 1, C, ns]: (j, c, s) at state + j + c * st_ld + s * st_sd
    int64_t st_ld, st_sd;
    const float * w;      // [d, C], contiguous
    float * out;          // [C, nt, ns], contiguous
    float * hist;         // [keep, C, ns], contiguous: the last keep inputs of [state | x]
    int32_t C, nt, ns, d, keep;
    int32_t chunk;        // tokens per thread
    // With norm_blocks > 0 (blocks of norm_head threads, one head each): the first norm_blocks
    // blocks' outputs (a gated DeltaNet's q and k heads) are l2-normalized per token as
    // delta_prep does it (eps norm_eps, scale norm_scale), bit for bit.
    int32_t norm_blocks, norm_head;
    float norm_eps, norm_scale;
    uint32_t grid[3];
    uint32_t block;
};

int legion_conv_silu_launch(const legion_conv_silu * p, void * stream);

// router_topk: per token, the k largest of n_expert logits, ids in descending order of value
// (ties: the lower id first). router_weights: their softmax probabilities normalized over the k
// (the softmax denominator cancels: w_j = e^(l_j - l_max) / sum over the k of the same).
struct legion_router {
    const float * logits;  // [n_expert, n_tok], columns ld floats apart
    int64_t ld;
    const int32_t * ids_in;  // router_weights: [k, n_tok] contiguous
    int32_t * ids;           // router_topk out: [k, n_tok] contiguous
    float * w;               // router_weights out: [k, n_tok] contiguous
    int32_t n_expert, k, n_tok;
    // router_topk: nvfp4_once.cu's routing plan (nvfp4_plan.cuh) to write from the ids, or null
    // (#836); n_tok * k <= nvfp4_plan::MAX_PAIRS and n_tok <= 8 then.
    nvfp4_plan::plan_t * plan;
};

int legion_router_topk_launch(const legion_router * p, void * stream);
int legion_router_weights_launch(const legion_router * p, void * stream);
// Both in one launch (#814): ids and w out, ids_in and plan unused (plan null).
int legion_router_select_launch(const legion_router * p, void * stream);

// hc_norm: per token, the hc streams of x RMS-normed one by one and times the gamma (xn), and
// inject = W^T xn (n_inj outputs).
struct legion_hc_norm {
    const float * x;    // [e, hc, n_tok]: token t at x + t * x_ld (floats), streams contiguous
    int64_t x_ld;
    const float * g;    // [e * hc] gamma
    const float * w;    // [e * hc, n_inj] contiguous
    float * xn;         // [e * hc, n_tok] contiguous
    float * inject;     // [n_inj, n_tok] contiguous
    int32_t e, hc, n_inj, n_tok;
    float eps;
};

int legion_hc_norm_launch(const legion_hc_norm * p, void * stream);
// Launches of hc_norm on sm_7x take the variant rounded as hc_post_norm is there (nonzero;
// `legion serve --hc-norm-fma`) or nvcc's own contraction (0, the default).
void legion_hc_norm_set_fma(int on);
// Tests: the same for launches from the calling thread only.
void legion_hc_norm_force_fma(int on);

// hc_post_norm: hc_post then hc_norm in one pass per token. The streams become
// res' = out * w + res with w = 2 sigmoid(inj_prev / hc) per stream (ggml's scale, sigmoid,
// scale and dsv4_hc_post, element by element), written out; then res' is normed as hc_norm does,
// with the same per-thread order of the sums.
struct legion_hc_post_norm {
    const float * res;      // [e, hc, n_tok]: token t at res + t * res_ld, streams contiguous
    int64_t res_ld;
    const float * out;      // [e, n_tok]: token t at out + t * out_ld
    int64_t out_ld;
    const float * inj;      // [hc, n_tok]: token t at inj + t * inj_ld
    int64_t inj_ld;
    const float * g;        // [e * hc] gamma
    const float * w;        // [e * hc, n_inj] contiguous
    float * res_new;        // [e * hc, n_tok] contiguous
    float * xn;             // [e * hc, n_tok] contiguous
    float * inject;         // [n_inj, n_tok] contiguous
    // [e * hc, n_tok] contiguous or null: xn rounded to F16 (__float2half, as ggml-cuda's
    // F32 -> F16 conversion before a cuBLAS F16 GEMM), for the matmul that reads it.
    __half * xn16;
    int32_t e, hc, n_inj, n_tok;
    float eps;
    // With xq, with xn16 on a V100 or with xnb on sm_80+ at qwen4exp's shape (#774, #836;
    // hc_mix16.cu reads them): 0, or
    // where each token's per-stream norm scales go, [hc, n_tok] contiguous
    // at res_new + scale_at (in floats); xn is then not written, and a reader rebuilds it as
    // (res_new * scale) * g, the kernels' own two rounded multiplies (src/cuda/hc_up_mix.cu,
    // #526). In the padding after eps: the struct's size and layout, and so every other
    // kernel's code, stay as they were.
    int32_t scale_at;
    // Null, or xn quantized as ggml-cuda's quantize_mmq_q8_1 quantizes it for a Q8_0 weight's
    // MMQ (block_q8_1_mmq, D4: 4 scales and 128 int8 per block, token t's values 128 b ..
    // 128 b + 127 at block b * n_tok + t), for src/cuda/mmq_preq.cu (#363). sm_80+, e * hc a
    // multiple of 512.
    int8_t * xq;
    // [e * hc, n_tok] contiguous or null: xn rounded to BF16 (__float2bfloat16, as ggml-cuda's
    // F32 -> BF16 conversion before a cuBLAS BF16 GEMM), for the BF16 down weight's matmul
    // (#469). sm_80+, neither xn16 nor xq; with scale_at only at qwen4exp's shape (#836).
    __nv_bfloat16 * xnb;
    // legion_hc_post_norm_rs_launch only (#1176): device memory holding the addresses of rank
    // 0's and rank 1's BF16 partials of a tensor-parallel reduce-scatter laid out as out (token
    // t at rs[r] + t * out_ld), whose sum out is, added here as src/cuda/tp_net.cu's
    // tp_sum_bf16 adds them; out is not read. The exchange writes the two addresses there on
    // the stream before this kernel, so a captured launch reads the current exchange's buffers.
    // At the end: every other field keeps its place.
    const __nv_bfloat16 * const * rs;
    // Lazy streams (#1183, src/model/core/node_fuse/fold.rs): null, or res holds the streams
    // before the previous block's combine, made here first as that block's norm made them,
    // res' = pre_out * w + res with w = 2 sigmoid(pre_inj / hc) per stream (the same
    // expressions, so the same bits): pre_out [e, n_tok] (token t at pre_out + t * pre_out_ld),
    // pre_inj [hc, n_tok] (token t at pre_inj + t * pre_inj_ld). legion_hc_post_norm_fold_launch
    // only.
    const float * pre_out;
    int64_t pre_out_ld;
    const float * pre_inj;
    int64_t pre_inj_ld;
    // Nonzero: the new streams are not written (res_new's streams stay as they are; a reader
    // makes them from this launch's res, out and inj, see pre_out). legion_hc_post_norm_launch's
    // rebuilt BF16-xn form only.
    int32_t no_res;
};

int legion_hc_post_norm_launch(const legion_hc_post_norm * p, void * stream);
// hc_post_norm_v70_kernel's rebuilt BF16-xn form (xnb, scale_at) with out made from p->rs.
int legion_hc_post_norm_rs_launch(const legion_hc_post_norm * p, void * stream);
// Tests: launches of hc_post_norm from the calling thread take the generic kernel (1), the
// register one where it applies (2) or the default choice (0).
void legion_hc_post_norm_force_generic(int on);
// Every launch takes the generic kernel (nonzero; `legion serve --hc-post-norm-generic`).
void legion_hc_post_norm_set_generic(int on);
// hc_post_norm with BF16 residual streams (#826, `legion serve --hc-streams-bf16`, opt-in: it
// moves output). res' = out * w + res is rounded to BF16 (__float2bfloat16) and written as
// BF16 [e, hc, n_tok] at res_new (viewed as __nv_bfloat16 *); the rounded values are the
// ones normed, quantized into xq and injected, so a reader rebuilding xn from the BF16 streams
// and the scales (src/cuda/hc_up_mix.cu) sees what the norm saw. in16: res is BF16 too (res_ld
// in BF16 values), else F32. Only qwen4exp's 4 x 2560 streams with 4 inject outputs, sm_80+,
// with xq and scale_at (the rebuilt form), no xn16 / xnb; -1 otherwise.
int legion_hc_post_norm_s16_launch(const legion_hc_post_norm * p, int in16, void * stream);

// hc_post_norm_moe (#774, `legion serve --hc-fuse`): hc_post_norm whose block output `out` is the
// fused MoE op's per-token sum (src/cuda/moe_gemm.cu), made from the op's per-pair rows here
// instead of written by its reduce_kernel and read back: the token's valid pairs in slot order,
// the first one starting the sum, then the gated shared expert (if any) added as reduce_kernel
// adds it. p->out is not read. qwen4exp's 4 x 2560 streams with 4 inject outputs only
// (hc_post_norm_v70_kernel's shape), F32 pair rows; on sm_70 no xq, on sm_80+ (#826) xq (with
// or without scale_at, the rebuilt form) or neither, and no xn16 / xnb; -1 otherwise.
struct legion_moe_pair_sum {
    const float * d;          // pair (t, j) of the sum at d + (t * n_used + j) * e (F32 rows)
    const int32_t * ids;      // token t's experts at ids + t * ids_ld
    int64_t ids_ld;
    const int32_t * table;    // [n_expert] slot of expert i, or null (slot i)
    int32_t n_expert, n_used, n_slots;
    const float * sh;         // null, or the shared expert's output: token t at sh + t * sh_ld
    int64_t sh_ld;
    const float * sg;         // its gate logit of token t at sg[t * sg_ld]
    int64_t sg_ld;
};

int legion_hc_post_norm_moe_launch(const legion_hc_post_norm * p, const legion_moe_pair_sum * s, void * stream);

// hc_post_norm_fold (#836, folded before the graph is planned again,
// src/model/core/node_fuse/fold.rs): the
// rebuilt BF16-xn hc_post_norm (sm_80+, qwen4exp's 4 x 2560 streams, 4 inject outputs, xnb and
// scale_at, no xq / xn16) whose block output `out` is a MoE layer's tail made here instead of
// written by the tail's kernels and read back: the routed experts' weighted sum, rounded as the
// graph's own nodes round it in that layer (fma: ggml-cuda's fused moe_weighted_reduction, the
// first product and then one fma per expert, as moe_combine_kernel; else ggml's mul and an add
// per expert), then the shared expert gated by its sigmoid added (ggml's op_sigmoid, mul and add, each
// rounded). p->out is not read. -1 if the step does not fit.
struct legion_moe_fold {
    const float * x;   // [e, k, n_tok] contiguous: the experts' outputs, or (w null) weighted
    const float * w;   // [k, n_tok] contiguous router weights, or null
    const float * sh;  // [e, n_tok] contiguous: the shared expert's output
    const float * g;   // token t's gate logit at g + t * g_ld
    int64_t g_ld;
    int32_t k;
    int32_t fma;
    const __nv_bfloat16 * x16;  // null, or (x null) the experts' outputs as BF16, read exactly
};

int legion_hc_post_norm_fold_launch(const legion_hc_post_norm * p, const legion_moe_fold * f, void * stream);

// hc_post_norm with BF16 residual streams for BF16 mixers (#1183, `legion serve --hc-bf16`,
// opt-in: it moves output): the rebuilt BF16-xn form (sm_80+, qwen4exp's 4 x 2560 streams, 4
// inject outputs, xnb and scale_at, no xq / xn16), with res' = out * w + res rounded to BF16
// (__float2bfloat16_rn) and written as BF16 [e, hc, n_tok] at res_new (viewed as
// __nv_bfloat16 *); the rounded values are the ones normed, rounded into xnb and injected, so
// up_mix_bf16 rebuilding xn from the BF16 streams and the scales sees what the norm saw. in16:
// res is BF16 too (res_ld in BF16 values), else F32. f: null, or a folded MoE tail (as
// legion_hc_post_norm_fold_launch; p->out is then not read). Each thread takes pairs of
// adjacent values (4-byte BF16 and 8-byte F32 accesses), so the sums run in another order
// than the F32 kernels'. xnb may be null (src/cuda/hc_down.cu rebuilds it in the down matmul):
// then it is not written. -1 if the step does not fit.
int legion_hc_post_norm_b16_launch(const legion_hc_post_norm * p, const legion_moe_fold * f, int in16, void * stream);

// delta_prep: a gated DeltaNet layer's delta-rule inputs from its conv output, per token:
// q, k = l2_norm of each head_k-wide head (ggml's rms_norm with eps / head_k and then a scale
// of 1 / sqrt(head_k), with the same reduction order as ggml-cuda's 256-thread rms_norm block),
// g = softplus(alpha + dt_bias) * a, beta = sigmoid(beta_raw), each as ggml-cuda's ops compute
// it; and, for a verify step, the replay stores of q, k, g and beta.
struct legion_delta_prep {
    const float * conv;     // [qkv_dim, n_tok] contiguous: q heads, k heads, v heads per token
    const float * alpha;    // [n_v, n_tok]: token t at alpha + t * alpha_ld
    int64_t alpha_ld;
    const float * beta_in;  // [n_v, n_tok]: token t at beta_in + t * beta_ld
    int64_t beta_ld;
    const float * dt_bias;  // [n_v]
    const float * a;        // [n_v]
    float * q;              // [head_k * n_k, n_tok] contiguous
    float * k;              // [head_k * n_k, n_tok] contiguous
    float * g;              // [n_v, n_tok] contiguous
    float * beta;           // [n_v, n_tok] contiguous
    // Replay stores (all null if none): token (t, s) of store i at st[i] + t * st_ld[i] + s * st_sd[i],
    // rows contiguous; stores q, k, g, beta.
    float * st[4];
    int64_t st_ld[4];
    int64_t st_sd[4];
    int32_t head_k, n_k, head_v, n_v, nt, n_tok;
    float eps;        // rms_norm eps (the model's eps / head_k)
    float scale_out;  // 1 / sqrt(head_k)
    int32_t gates_only;  // g and beta only (conv_silu normalized q and k already); q, k unused
};

int legion_delta_prep_launch(const legion_delta_prep * p, void * stream);

// moe_combine: a MoE layer's experts' outputs weighted and summed, always as ggml-cuda's fused
// moe_weighted_reduction computes them (the first product rounded, then one fma per expert),
// whatever the graph's layout (ggml fuses only when the tensors do not overlap in memory and
// runs a mul and adds otherwise, which round differently).
struct legion_moe_combine {
    const float * experts;  // [n_embd, n_used, n_tok] contiguous
    const float * w;        // [n_used, n_tok] contiguous
    float * out;            // [n_embd, n_tok] contiguous
    int64_t n_embd;
    int32_t n_used, n_tok;
};

int legion_moe_combine_launch(const legion_moe_combine * p, void * stream);

// gated_norm: a gated DeltaNet layer's output norm, per token and head of head_v:
// rms_norm(o) * w * gate(z), gate sigmoid or silu, as ggml-cuda's rms_norm (fused with the mul
// by w, 256-thread block order), unary and binary kernels compute it. out_bf16: the result
// rounded to BF16 as ggml's cast rounds it (__float2bfloat16), for a BF16-input matmul.
struct legion_gated_norm {
    const float * o;  // [head_v * n_v, n_tok]: token t at o + t * o_ld
    int64_t o_ld;
    const float * w;  // [head_v]
    const float * z;  // [head_v * n_v, n_tok]: token t at z + t * z_ld
    int64_t z_ld;
    void * out;       // [head_v * n_v, n_tok] contiguous, F32 or (out_bf16) BF16
    int32_t head_v, n_v, n_tok, silu;
    float eps;
    int32_t out_bf16;
};

int legion_gated_norm_launch(const legion_gated_norm * p, void * stream);

// rms_norm_bf16: mul(rms_norm(x, eps), w) per row, as ggml-cuda's rms_norm_f32 fused with the
// mul computes it (1024 threads per row from 1024 columns on, 256 below: each thread sums x^2
// over columns tid, tid + block, ..., warp_reduce_sum, the warp sums reduced by warp 0, mean =
// sum / ncols, rsqrtf(mean + eps), y = scale * x * w), then rounded to BF16 as ggml's cast rounds
// it (__float2bfloat16): the input of BF16-activation matmuls, without the F32 rows.
struct legion_rms_norm_bf16 {
    const float * x;  // [ncols, n_rows]: row r at x + r * x_ld
    int64_t x_ld;
    const float * w;  // [ncols]
    void * out;       // BF16 [ncols, n_rows] contiguous
    int32_t ncols, n_rows;
    float eps;
};

int legion_rms_norm_bf16_launch(const legion_rms_norm_bf16 * p, void * stream);

// gate_bf16: op(x) * g with op sigmoid or (silu) SiLU, as ggml-cuda's fused unary + mul and its
// SwiGLU compute it (1 / (1 + expf(-x)), x / (1 + expf(-x)), then the product), rounded to BF16 as
// ggml's cast rounds it (__float2bfloat16): the input of a BF16-activation matmul, without the F32
// rows. Element i of token t reads x at head i / d, column i % d (x may be a strided view, e.g.
// the gate half of attn_q's interleaved [q | gate] heads).
struct legion_gate_bf16 {
    const float * x;  // token t, head h, column j at x + t * x_ld + h * x_hd + j
    int64_t x_ld, x_hd;
    const float * g;  // [d * n_head, n_tok]: token t at g + t * g_ld
    int64_t g_ld;
    void * out;       // BF16 [d * n_head, n_tok] contiguous
    int32_t d, n_head, n_tok, silu;
};

int legion_gate_bf16_launch(const legion_gate_bf16 * p, void * stream);

// shexp_gate: a MoE layer's output with its gated shared expert, out = moe + sh * sigmoid(g),
// as ggml-cuda's sigmoid, broadcast mul and add kernels compute it (each result rounded; one
// launch instead of three). After Strata-AC922 (MIT), which scales the shared expert's rows by
// their sigmoid gate in one kernel.
struct legion_shexp_gate {
    const float * moe;  // [n_embd, n_tok] contiguous
    const float * sh;   // [n_embd, n_tok] contiguous
    const float * g;    // [n_tok]: token t's gate logit at g + t * g_ld
    int64_t g_ld;
    float * out;        // [n_embd, n_tok] contiguous
    int64_t n_embd;
    int32_t n_tok;
};

int legion_shexp_gate_launch(const legion_shexp_gate * p, void * stream);

}  // extern "C"

namespace {

constexpr int MAX_D = 8;

// ggml-cuda's warp_reduce_sum (common.cuh): butterfly over the 32 lanes.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

constexpr int PREP_BLOCK = 256;  // ggml-cuda's rms_norm block for rows under 1024

// Input j of channel c in [state | x] time order (j < d - 1: the state).
__device__ __forceinline__ float conv_in(const legion_conv_silu & p, int j, int c, int s) {
    return j < p.d - 1 ? p.state[j + c * p.st_ld + s * p.st_sd] : p.x[c + (int64_t) (j - p.d + 1) * p.x_ld + s * p.x_sd];
}

// One thread per (channel, run of `chunk` tokens, sequence): a sliding window of d inputs in
// registers; reads and writes are coalesced over the channels. A norm block (one q or k head,
// blockDim == norm_head) sums each token's squares as delta_prep_kernel's 256-thread block
// does (the warps past the head add zeros there: s_sum's tail is zero here), so its outputs
// are delta_prep's q and k, and the separate pass over them is saved.
__global__ void conv_silu_kernel(legion_conv_silu p) {
    __shared__ float s_sum[PREP_BLOCK / 32];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.z;
    const bool norm = (int) blockIdx.x < p.norm_blocks;  // the whole block: all its c < C
    if (norm && threadIdx.x < PREP_BLOCK / 32) {
        s_sum[threadIdx.x] = 0.0f;
    }
    if (c >= p.C) {
        return;
    }
    const int d = p.d;
    float w[MAX_D], win[MAX_D];
#pragma unroll
    for (int k = 0; k < MAX_D; k++) {
        w[k] = k < d ? p.w[k + c * d] : 0.0f;
    }
    const int t0 = blockIdx.y * p.chunk;
    const int t1 = min(t0 + p.chunk, p.nt);
    // the window for output t holds inputs t .. t + d - 1 (in [state | x] order)
#pragma unroll
    for (int k = 0; k < MAX_D - 1; k++) {
        win[k] = k < d - 1 ? conv_in(p, t0 + k, c, s) : 0.0f;
    }
    for (int t = t0; t < t1; t++) {
        win[d - 1] = conv_in(p, t + d - 1, c, s);
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_D; k++) {
            if (k < d) {
                acc += win[k] * w[k];
            }
        }
        float y = acc / (1.0f + __expf(-acc));
        if (norm) {
            // delta_prep_kernel's sum, mean and scale of the head's row (it reads y back).
            float tmp = 0.0f;
            tmp += y * y;
            tmp = warp_sum(tmp);
            __syncthreads();  // the previous token's s_sum reads are done
            if (threadIdx.x % 32 == 0) {
                s_sum[threadIdx.x / 32] = tmp;
            }
            __syncthreads();
            tmp = threadIdx.x % 32 < PREP_BLOCK / 32 ? s_sum[threadIdx.x % 32] : 0.0f;
            tmp = warp_sum(tmp);
            const float mean = tmp / p.norm_head;
            const float scale = rsqrtf(mean + p.norm_eps);
            y = p.norm_scale * (scale * y);
        }
        p.out[c + (int64_t) t * p.C + (int64_t) s * p.C * p.nt] = y;
#pragma unroll
        for (int k = 0; k < MAX_D - 1; k++) {
            win[k] = win[k + 1];
        }
    }
    if (blockIdx.y == 0) {
        for (int j = 0; j < p.keep; j++) {
            p.hist[j + (int64_t) c * p.keep + (int64_t) s * p.keep * p.C] = conv_in(p, p.nt + d - 1 - p.keep + j, c, s);
        }
    }
}

// One warp per token (router_select.cuh).
constexpr int MAX_PER_LANE = router_select::MAX_PER_LANE;

__global__ void router_topk_kernel(legion_router p) {
    const int lane = threadIdx.x % 32;
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (t >= p.n_tok) {
        return;
    }
    router_select::topk_warp<false>(p.logits + (int64_t) t * p.ld, p.n_expert, p.k, p.ids + (int64_t) t * p.k, lane);
}

// router_topk_kernel with nvfp4_once's routing plan (#836): one CTA, a warp per token, then
// the plan of the CTA's ids, so the step's NVFP4 expert ops need no plan of their own.
__global__ void router_topk_plan_kernel(legion_router p) {
    const int lane = threadIdx.x % 32;
    const int t = threadIdx.x / 32;
    if (t < p.n_tok) {
        router_select::topk_warp<false>(p.logits + (int64_t) t * p.ld, p.n_expert, p.k, p.ids + (int64_t) t * p.k,
                                        lane);
    }
    // The ids lane 0 of each warp wrote are visible to the CTA after the barrier.
    __syncthreads();
    nvfp4_plan::build(*p.plan, p.ids, p.k, p.k, p.n_tok, p.n_expert);
}

// router_topk_kernel and router_weights_kernel in one launch (#814): a warp per token picks the
// ids and then makes the weights from the logits its lanes picked (router_select.cuh), the
// bits of the two kernels.
__global__ void router_select_kernel(legion_router p) {
    const int lane = threadIdx.x % 32;
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (t >= p.n_tok) {
        return;
    }
    router_select::topk_weights_warp<false>(p.logits + (int64_t) t * p.ld, p.n_expert, p.k,
                                            p.ids + (int64_t) t * p.k, p.w + (int64_t) t * p.k, lane);
}

__global__ void router_weights_kernel(legion_router p) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n_tok) {
        return;
    }
    router_select::weights_token<false>(p.logits + (int64_t) t * p.ld, p.ids_in + (int64_t) t * p.k, p.k,
                                        p.w + (int64_t) t * p.k);
}

constexpr int HC_MAX = 8, HC_INJ_MAX = 8, HC_THREADS = 256;

// Sums v over the block (all threads call it); every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float * red) {
#pragma unroll
    for (int m = 16; m >= 1; m /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, m);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < HC_THREADS / 32; i++) {
        t += red[i];
    }
    return t;
}

// One block per token. FMA = false: the arithmetic as nvcc contracts it (the default, kept for
// its bits). For sm_70 it unrolls the squares' loop by four and adds the first square of each
// four with FMUL + FADD (two roundings), the other three with FFMA (one), while
// hc_post_norm_kernel adds every square with FFMA; so a stream's scale can be an ulp apart
// between this kernel and hc_post_norm on the same streams (#409, #376). FMA = true (sm_7x only,
// `legion serve --hc-norm-fma`): the arithmetic pinned to hc_post_norm's for sm_70 (FFMA for
// every square and inject term, (x * scale) * gamma two FMULs): its bits.
template <bool FMA>
__global__ void __launch_bounds__(HC_THREADS) hc_norm_kernel(legion_hc_norm p) {
    __shared__ float red[HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * x = p.x + t * p.x_ld;
    const int n = p.e * p.hc;
    float sq[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        sq[h] = 0.0f;
    }
    for (int k = threadIdx.x; k < n; k += HC_THREADS) {
        const float v = x[k];
        const int h = k / p.e;
#pragma unroll
        for (int i = 0; i < HC_MAX; i++) {
            if (i == h) {
                if constexpr (FMA) {
                    sq[i] = __fmaf_rn(v, v, sq[i]);
                } else {
                    sq[i] += v * v;
                }
            }
        }
    }
    float scale[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        if (h < p.hc) {
            const float s = block_sum(sq[h], red);
            scale[h] = 1.0f / sqrtf(s / p.e + p.eps);
        }
    }
    float acc[HC_INJ_MAX];
#pragma unroll
    for (int j = 0; j < HC_INJ_MAX; j++) {
        acc[j] = 0.0f;
    }
    float * xn = p.xn + t * n;
    for (int k = threadIdx.x; k < n; k += HC_THREADS) {
        const int h = k / p.e;
        float sc = 0.0f;
#pragma unroll
        for (int i = 0; i < HC_MAX; i++) {
            if (i == h) {
                sc = scale[i];
            }
        }
        if constexpr (FMA) {
            const float y = __fmul_rn(__fmul_rn(x[k], sc), p.g[k]);
            xn[k] = y;
#pragma unroll
            for (int j = 0; j < HC_INJ_MAX; j++) {
                if (j < p.n_inj) {
                    acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
                }
            }
        } else {
            const float y = (x[k] * sc) * p.g[k];
            xn[k] = y;
#pragma unroll
            for (int j = 0; j < HC_INJ_MAX; j++) {
                if (j < p.n_inj) {
                    acc[j] += y * p.w[k + (int64_t) n * j];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < HC_INJ_MAX; j++) {
        if (j < p.n_inj) {
            const float s = block_sum(acc[j], red);
            if (threadIdx.x == 0) {
                p.inject[t * p.n_inj + j] = s;
            }
        }
    }
}

// Value k of token t of xn (x) into xq as ggml-cuda's quantize_mmq_q8_1 (quantize.cu, MIT)
// writes it for a Q8_0 src0 (MMQ_Q8_1_DS_LAYOUT_D4), with its arithmetic: amax over the 32
// values of the scale's group, d_inv = 127 / amax, q = roundf(x * d_inv), d = 1 / d_inv. The
// calling warp's 32 lanes hold values k .. k + 31 of one group (k a multiple of 32 in lane 0):
// every launch has e * hc a multiple of HC_THREADS and lane i at k = 32 w + i + j HC_THREADS.
// fmaxf is exact, so the reduction order (a butterfly here, 4 values per thread then 8 lanes
// in ggml) does not change amax.
__device__ __forceinline__ void q8_1_mmq_store(int8_t * xq, int64_t n_tok, int64_t t, int k, float x) {
    float amax = fabsf(x);
#pragma unroll
    for (int m = 16; m >= 1; m /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    const float d_inv = 127.0f / amax;
    const signed char q = roundf(x * d_inv);
    const float d = 1.0f / d_inv;
    int8_t * blk = xq + ((int64_t) (k / 128) * n_tok + t) * 144;
    blk[16 + k % 128] = q;
    if (k % 32 == 0) {
        ((float *) blk)[(k % 128) / 32] = d;
    }
}

// One block per token. Pass 1 writes res' and sums each stream's squares (thread order as
// hc_norm_kernel); pass 2 reads res' back (from L2) and writes xn and the inject sums.
// Q: also xn quantized into p.xq (q8_1_mmq_store).
template <bool Q>
__global__ void __launch_bounds__(HC_THREADS) hc_post_norm_kernel(legion_hc_post_norm p) {
    __shared__ float red[HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * res = p.res + t * p.res_ld;
    const float * out = p.out + t * p.out_ld;
    const int n = p.e * p.hc;
    float * rn = p.res_new + t * n;
    float wpost[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        wpost[h] = 0.0f;
        if (h < p.hc) {
            // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
            const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
            const float s = 1.0f / (1.0f + expf(-v));
            wpost[h] = s * 2.0f;
        }
    }
    float sq[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        sq[h] = 0.0f;
    }
    for (int k = threadIdx.x; k < n; k += HC_THREADS) {
        const int h = k / p.e;
        float wh = 0.0f;
#pragma unroll
        for (int i = 0; i < HC_MAX; i++) {
            if (i == h) {
                wh = wpost[i];
            }
        }
        // dsv4_hc_post_f32: sum = x * post; sum += residual
        float v = out[k - h * p.e] * wh;
        v += res[k];
        rn[k] = v;
#pragma unroll
        for (int i = 0; i < HC_MAX; i++) {
            if (i == h) {
                sq[i] += v * v;
            }
        }
    }
    float scale[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        if (h < p.hc) {
            const float s = block_sum(sq[h], red);
            scale[h] = 1.0f / sqrtf(s / p.e + p.eps);
            if constexpr (Q) {
                if (p.scale_at != 0 && threadIdx.x == 0) {
                    p.res_new[p.scale_at + t * p.hc + h] = scale[h];
                }
            }
        }
    }
    float acc[HC_INJ_MAX];
#pragma unroll
    for (int j = 0; j < HC_INJ_MAX; j++) {
        acc[j] = 0.0f;
    }
    float * xn = p.xn + t * n;
    for (int k = threadIdx.x; k < n; k += HC_THREADS) {
        const int h = k / p.e;
        float sc = 0.0f;
#pragma unroll
        for (int i = 0; i < HC_MAX; i++) {
            if (i == h) {
                sc = scale[i];
            }
        }
        const float y = (rn[k] * sc) * p.g[k];
        if (!Q || p.scale_at == 0) {
            xn[k] = y;
        }
        if constexpr (Q) {
            // xq and xn16 are never both asked for; no xn16 code here (64 registers).
            q8_1_mmq_store(p.xq, p.n_tok, t, k, y);
        } else if (p.xn16 != nullptr) {
            p.xn16[t * n + k] = __float2half(y);
        } else if (p.xnb != nullptr) {
            p.xnb[t * n + k] = __float2bfloat16(y);
        }
#pragma unroll
        for (int j = 0; j < HC_INJ_MAX; j++) {
            if (j < p.n_inj) {
                acc[j] += y * p.w[k + (int64_t) n * j];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < HC_INJ_MAX; j++) {
        if (j < p.n_inj) {
            const float s = block_sum(acc[j], red);
            if (threadIdx.x == 0) {
                p.inject[t * p.n_inj + j] = s;
            }
        }
    }
}

// Blocks (h, t): h < 2 n_k normalizes q (h < n_k) or k head h % n_k of token t; h == 2 n_k
// computes the token's g and beta. 256 threads, as ggml's rms_norm block: thread
// i holds column i of the head (none past head_k), warp sums, then a warp sum of the 8 warps'.
__global__ void delta_prep_kernel(legion_delta_prep p) {
    __shared__ float s_sum[PREP_BLOCK / 32];
    const int h = blockIdx.x + (p.gates_only ? 2 * p.n_k : 0);
    const int t = blockIdx.y;
    const int s = t / p.nt;
    const int ts = t - s * p.nt;
    const int tid = threadIdx.x;
    const int qk = p.head_k * p.n_k;
    const int qkv_dim = 2 * qk + p.head_v * p.n_v;
    const float * row = p.conv + (int64_t) t * qkv_dim;
    if (h < 2 * p.n_k) {
        const int which = h / p.n_k;  // 0: q, 1: k
        const int off = which * qk + (h % p.n_k) * p.head_k;
        float x = 0.0f;
        float tmp = 0.0f;
        if (tid < p.head_k) {
            x = row[off + tid];
            tmp += x * x;
        }
        tmp = warp_sum(tmp);
        if (tid % 32 == 0) {
            s_sum[tid / 32] = tmp;
        }
        __syncthreads();
        tmp = tid % 32 < PREP_BLOCK / 32 ? s_sum[tid % 32] : 0.0f;
        tmp = warp_sum(tmp);
        const float mean = tmp / p.head_k;
        const float scale = rsqrtf(mean + p.eps);
        if (tid < p.head_k) {
            const float y = p.scale_out * (scale * x);
            const int64_t o = (int64_t) t * qk + (off - which * qk) + tid;
            (which == 0 ? p.q : p.k)[o] = y;
            float * st = p.st[which];
            if (st != nullptr) {
                st[ts * p.st_ld[which] + s * p.st_sd[which] + (off - which * qk) + tid] = y;
            }
        }
        return;
    }
    for (int i = tid; i < p.n_v; i += blockDim.x) {
        const float x = p.alpha[(int64_t) t * p.alpha_ld + i] + p.dt_bias[i];
        const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
        const float gv = sp * p.a[i];
        const float b = p.beta_in[(int64_t) t * p.beta_ld + i];
        const float bv = 1.0f / (1.0f + expf(-b));
        p.g[(int64_t) t * p.n_v + i] = gv;
        p.beta[(int64_t) t * p.n_v + i] = bv;
        if (p.st[2] != nullptr) {
            p.st[2][ts * p.st_ld[2] + s * p.st_sd[2] + i] = gv;
            p.st[3][ts * p.st_ld[3] + s * p.st_sd[3] + i] = bv;
        }
    }
}

// Blocks (head, token), 256 threads as ggml's rms_norm block: thread i holds column i.
__global__ void gated_norm_kernel(legion_gated_norm p) {
    __shared__ float s_sum[PREP_BLOCK / 32];
    const int h = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int tid = threadIdx.x;
    const int col = h * p.head_v + tid;
    float x = 0.0f;
    float tmp = 0.0f;
    if (tid < p.head_v) {
        x = p.o[t * p.o_ld + col];
        tmp += x * x;
    }
    tmp = warp_sum(tmp);
    if (tid % 32 == 0) {
        s_sum[tid / 32] = tmp;
    }
    __syncthreads();
    tmp = tid % 32 < PREP_BLOCK / 32 ? s_sum[tid % 32] : 0.0f;
    tmp = warp_sum(tmp);
    const float mean = tmp / p.head_v;
    const float scale = rsqrtf(mean + p.eps);
    if (tid < p.head_v) {
        const float y = scale * x * p.w[tid];
        const float zv = p.z[t * p.z_ld + col];
        const float g = p.silu ? zv / (1.0f + expf(-zv)) : 1.0f / (1.0f + expf(-zv));
        const int64_t i = t * (int64_t) p.head_v * p.n_v + col;
        if (p.out_bf16) {
            ((__nv_bfloat16 *) p.out)[i] = __float2bfloat16(y * g);
        } else {
            ((float *) p.out)[i] = y * g;
        }
    }
}

// One block of BLOCK threads per row, in ggml-cuda's rms_norm_f32 order.
template <int BLOCK>
__global__ void rms_norm_bf16_kernel(legion_rms_norm_bf16 p) {
    __shared__ float s_sum[32];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x = p.x + row * p.x_ld;
    float tmp = 0.0f;
    for (int col = tid; col < p.ncols; col += BLOCK) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    tmp = warp_sum(tmp);
    if (BLOCK > 32) {
        if (tid % 32 == 0) {
            s_sum[tid / 32] = tmp;
        }
        __syncthreads();
        tmp = tid % 32 < BLOCK / 32 ? s_sum[tid % 32] : 0.0f;
        tmp = warp_sum(tmp);
    }
    const float mean = tmp / p.ncols;
    const float scale = rsqrtf(mean + p.eps);
    __nv_bfloat16 * out = (__nv_bfloat16 *) p.out + row * p.ncols;
    for (int col = tid; col < p.ncols; col += BLOCK) {
        out[col] = __float2bfloat16(scale * x[col] * p.w[col]);
    }
}

// One thread per element; blockIdx.y is the token.
__global__ void gate_bf16_kernel(legion_gate_bf16 p) {
    const int64_t t = blockIdx.y;
    const int k = p.d * p.n_head;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float xv = p.x[t * p.x_ld + (int64_t) (i / p.d) * p.x_hd + i % p.d];
    const float a = p.silu ? xv / (1.0f + expf(-xv)) : 1.0f / (1.0f + expf(-xv));
    ((__nv_bfloat16 *) p.out)[t * k + i] = __float2bfloat16(a * p.g[t * p.g_ld + i]);
}

__global__ void moe_combine_kernel(legion_moe_combine p) {
    const int64_t t = blockIdx.x;
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= p.n_embd) {
        return;
    }
    const int64_t first = t * p.n_used;
    float sum = __fmul_rn(p.experts[first * p.n_embd + col], p.w[first]);
    for (int e = 1; e < p.n_used; ++e) {
        sum = __fmaf_rn(p.experts[(first + e) * p.n_embd + col], p.w[first + e], sum);
    }
    p.out[t * p.n_embd + col] = sum;
}

__global__ void shexp_gate_kernel(legion_shexp_gate p) {
    const int64_t t = blockIdx.x;
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= p.n_embd) {
        return;
    }
    // ggml-cuda's op_sigmoid (both built with fast math), then the mul and the add, each
    // rounded on its own as ggml's separate kernels store them (no fma).
    const float gv = p.g[t * p.g_ld];
    const float s = 1.0f / (1.0f + expf(-gv));
    const int64_t i = t * p.n_embd + col;
    p.out[i] = __fadd_rn(p.moe[i], __fmul_rn(p.sh[i], s));
}

// hc_post_norm_kernel with the token's PER * HC_THREADS values (e * hc exactly) held in
// registers between the passes instead of read back, and the streams' and inject outputs' block
// sums done together (HC_MAX values per barrier pair instead of one). The same values per
// thread in the same order and the same reduction tree per sum: the same bits (the
// arithmetic pinned with intrinsics, see below). sm_80+ only: nvcc contracts the generic
// kernel differently for sm_70, which has its own pinned form (hc_post_norm_v70_kernel).
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float (*red)[HC_THREADS / 32]) {
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
        for (int m = 16; m >= 1; m /= 2) {
            v[i] += __shfl_xor_sync(0xffffffffu, v[i], m);
        }
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            red[i][threadIdx.x / 32] = v[i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; i++) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < HC_THREADS / 32; w++) {
            t += red[i][w];
        }
        v[i] = t;
    }
}

template <int PER, bool Q = false>
__global__ void __launch_bounds__(HC_THREADS) hc_post_norm_reg_kernel(legion_hc_post_norm p) {
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * res = p.res + t * p.res_ld;
    const float * out = p.out + t * p.out_ld;
    const int n = p.e * p.hc;
    float * rn = p.res_new + t * n;
    float wpost[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        wpost[h] = 0.0f;
        if (h < p.hc) {
            // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
            const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
            const float s = 1.0f / (1.0f + expf(-v));
            wpost[h] = s * 2.0f;
        }
    }
    float sq[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        sq[h] = 0.0f;
    }
    float val[PER];
    int hs[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int k = threadIdx.x + i * HC_THREADS;
        const int h = k / p.e;
        hs[i] = h;
        float wh = 0.0f;
#pragma unroll
        for (int c = 0; c < HC_MAX; c++) {
            if (c == h) {
                wh = wpost[c];
            }
        }
        // dsv4_hc_post_f32: sum = x * post; sum += residual. The arithmetic is pinned to what
        // the compiler makes of hc_post_norm_kernel's on sm_80+ (one FMA here, the square a
        // separate multiply, the gamma two multiplies, the inject sums FMAs): unrolled, it
        // would contract differently.
        const float v = __fmaf_rn(out[k - h * p.e], wh, res[k]);
        rn[k] = v;
        val[i] = v;
#pragma unroll
        for (int c = 0; c < HC_MAX; c++) {
            if (c == h) {
                sq[c] = __fadd_rn(sq[c], __fmul_rn(v, v));
            }
        }
    }
    block_sums(sq, red);
    float scale[HC_MAX];
#pragma unroll
    for (int h = 0; h < HC_MAX; h++) {
        scale[h] = h < p.hc ? 1.0f / sqrtf(sq[h] / p.e + p.eps) : 0.0f;
    }
    if constexpr (Q) {
        if (p.scale_at != 0 && threadIdx.x == 0) {
#pragma unroll
            for (int h = 0; h < HC_MAX; h++) {
                if (h < p.hc) {
                    p.res_new[p.scale_at + t * p.hc + h] = scale[h];
                }
            }
        }
    }
    float acc[HC_INJ_MAX];
#pragma unroll
    for (int j = 0; j < HC_INJ_MAX; j++) {
        acc[j] = 0.0f;
    }
    float * xn = p.xn + t * n;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int k = threadIdx.x + i * HC_THREADS;
        float sc = 0.0f;
#pragma unroll
        for (int c = 0; c < HC_MAX; c++) {
            if (c == hs[i]) {
                sc = scale[c];
            }
        }
        const float y = __fmul_rn(__fmul_rn(val[i], sc), p.g[k]);
        if (!Q || p.scale_at == 0) {
            xn[k] = y;
        }
        if constexpr (Q) {
            // xq and xn16 are never both asked for; no xn16 code here (64 registers).
            q8_1_mmq_store(p.xq, p.n_tok, t, k, y);
        } else if (p.xn16 != nullptr) {
            p.xn16[t * n + k] = __float2half(y);
        }
#pragma unroll
        for (int j = 0; j < HC_INJ_MAX; j++) {
            if (j < p.n_inj) {
                acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
            }
        }
    }
    block_sums(acc, red);
    if (threadIdx.x < p.n_inj) {
#pragma unroll
        for (int j = 0; j < HC_INJ_MAX; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * p.n_inj + j] = acc[j];
            }
        }
    }
}

// The sm_70 (V100) counterpart of hc_post_norm_reg_kernel, for exactly HC streams of PS *
// HC_THREADS values and NINJ inject outputs (qwen4exp: 4 x 2560, 4). The generic kernel there
// is latency-bound (one long-scoreboard wait per value, the stream picked by a select loop
// over HC_MAX). Here a value's stream is a compile-time constant (i / PS), the token's
// residual and the shared `out` row are loaded once, up front, and the new streams stay in
// registers for pass 2. Same values per thread in the same order and the same reduction trees
// as hc_post_norm_kernel, the arithmetic pinned to what nvcc makes of that kernel for sm_70
// (its SASS: one FFMA for out * post + res, FFMA for every square added, (rn * scale) * gamma
// two FMULs, FFMA for every inject term): the generic kernel's bits. Four blocks per SM (64
// registers, no spills; 4K tokens: 0.76 ms against the generic 1.46 ms, unbounded 255
// registers and one block per SM 1.6 ms).
// SM80: the same kernel for sm_80+ (#483), each square added as hc_post_norm_reg_kernel adds it
// (FMUL then FADD, what nvcc makes of the generic kernel there; all else is the same): its
// bits, at a lower register count (GB10, 4K tokens: see the hc_norm tests' timings).
// Q (sm_80+): also xn quantized into p.xq (q8_1_mmq_store).
// B16 (sm_80+, #469): also xn rounded to BF16 into p.xnb (no per-value test: the kernels without
// it keep their code).
// RB (#774): the streams' norm scales at p.scale_at instead of xn (with xn16, no xq; or, #836,
// with B16 on sm_80+), for a reader that rebuilds xn (hc_mix16.cu).
// RS (#1176, legion_hc_post_norm_rs_launch): out made from the two ranks' BF16 partials at the
// addresses p.rs holds, rank 0's first, as tp_sum_bf16 makes it (each widened to F32, one
// rounded add): out's bits.
// NR (#1183, p.no_res): the new streams are not written; everything else is the same.
template <int HC, int NINJ, int PS, bool SM80 = false, bool Q = false, bool B16 = false, bool RB = false,
          bool RS = false, bool NR = false>
__global__ void __launch_bounds__(HC_THREADS, 4) hc_post_norm_v70_kernel(legion_hc_post_norm p) {
    static_assert(HC <= HC_MAX && NINJ <= HC_INJ_MAX, "hc_post_norm_v70_kernel: too many sums");
    static_assert(!(Q && B16), "hc_post_norm_v70_kernel: xq or xnb, not both");
    constexpr int PER = HC * PS;
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * res = p.res + t * p.res_ld;
    const float * out = p.out + t * p.out_ld;
    const int n = PER * HC_THREADS;
    float * rn = p.res_new + t * n;
    if constexpr (SM80) {
        // #836: every token's pass 2 reads the layer's gamma and inject weights (n * (1 + NINJ)
        // floats, 200 KB for a 10240-wide row), which a decode step finds in DRAM; with one CTA
        // per token (16 at 16 tokens) those reads wait on DRAM one after another. Their 128-byte
        // lines go to L2 now, spread over the grid, while pass 1 runs. A hint: no value changes.
        const int64_t gl = n / 32, lines = gl + (int64_t) n * NINJ / 32;
        for (int64_t u = blockIdx.x * (int64_t) HC_THREADS + threadIdx.x; u < lines;
             u += (int64_t) gridDim.x * HC_THREADS) {
            const float * a = u < gl ? p.g + u * 32 : p.w + (u - gl) * 32;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
    }
    float ov[PS];
    // RS: the partials' addresses, read once from the exchange's slot.
    [[maybe_unused]] const __nv_bfloat16 * rs0 = nullptr;
    [[maybe_unused]] const __nv_bfloat16 * rs1 = nullptr;
    if constexpr (RS) {
        rs0 = p.rs[0];
        rs1 = p.rs[1];
    }
#pragma unroll
    for (int r = 0; r < PS; r++) {
        if constexpr (RS) {
            const int64_t i = t * p.out_ld + threadIdx.x + r * HC_THREADS;
            ov[r] = __fadd_rn(__bfloat162float(rs0[i]), __bfloat162float(rs1[i]));
        } else {
            ov[r] = out[threadIdx.x + r * HC_THREADS];
        }
    }
    float val[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        val[i] = res[threadIdx.x + i * HC_THREADS];
    }
    float wpost[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
        const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
        const float s = 1.0f / (1.0f + expf(-v));
        wpost[h] = s * 2.0f;
    }
    float sq[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        sq[h] = 0.0f;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const int i = h * PS + r;
            // dsv4_hc_post_f32: sum = x * post; sum += residual
            const float v = __fmaf_rn(ov[r], wpost[h], val[i]);
            if constexpr (!NR) {
                rn[threadIdx.x + i * HC_THREADS] = v;
            }
            val[i] = v;
            sq[h] = SM80 ? __fadd_rn(sq[h], __fmul_rn(v, v)) : __fmaf_rn(v, v, sq[h]);
        }
    }
    block_sums(sq, red);
    float scale[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        scale[h] = 1.0f / sqrtf(sq[h] / p.e + p.eps);
    }
    if constexpr (Q || RB) {
        if (p.scale_at != 0 && threadIdx.x == 0) {
#pragma unroll
            for (int h = 0; h < HC; h++) {
                p.res_new[p.scale_at + t * HC + h] = scale[h];
            }
        }
    }
    float acc[NINJ];
#pragma unroll
    for (int j = 0; j < NINJ; j++) {
        acc[j] = 0.0f;
    }
    float * xn = p.xn + t * n;
    // Two copies of the pass, with and without the xn store: a per-value test of scale_at costs
    // the Q kernel registers (52 -> 164 spilled bytes on sm_121a), the two copies none.
    auto pass = [&](auto write_xn) {
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int k = threadIdx.x + i * HC_THREADS;
            const float y = __fmul_rn(__fmul_rn(val[i], scale[i / PS]), p.g[k]);
            if constexpr (decltype(write_xn)::value) {
                xn[k] = y;
            }
            if constexpr (Q) {
                // xq and xn16 are never both asked for; no xn16 code here (64 registers).
                q8_1_mmq_store(p.xq, p.n_tok, t, k, y);
            } else if constexpr (B16) {
                p.xnb[t * n + k] = __float2bfloat16(y);
            } else if (p.xn16 != nullptr) {
                p.xn16[t * n + k] = __float2half(y);
            }
#pragma unroll
            for (int j = 0; j < NINJ; j++) {
                acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
            }
        }
    };
    if constexpr (RB) {
        pass(std::false_type{});
    } else if (!Q || p.scale_at == 0) {
        pass(std::true_type{});
    } else {
        pass(std::false_type{});
    }
    block_sums(acc, red);
    if (threadIdx.x < NINJ) {
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * NINJ + j] = acc[j];
            }
        }
    }
}

// hc_post_norm_v70_kernel<4, 4, 10, true, true> (sm_80+, xq, the rebuilt form) with BF16 streams
// (#826, legion_hc_post_norm_s16_launch): half the bytes of the streams read and written. The
// new stream value is rounded to BF16 before it is squared, quantized and injected.
template <bool IN16>
__global__ void __launch_bounds__(HC_THREADS, 4) hc_post_norm_s16_kernel(legion_hc_post_norm p) {
    constexpr int HC = 4, NINJ = 4, PS = 10, PER = HC * PS;
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * out = p.out + t * p.out_ld;
    const int n = PER * HC_THREADS;
    __nv_bfloat16 * rn = (__nv_bfloat16 *) p.res_new + t * n;
    float ov[PS];
#pragma unroll
    for (int r = 0; r < PS; r++) {
        ov[r] = out[threadIdx.x + r * HC_THREADS];
    }
    static_assert(PS % 2 == 0, "hc_post_norm_s16_kernel: pairs within a stream");
    // The streams as BF16 pairs (values i and i + 1 of this thread): the rounded new ones for
    // pass 2, and with IN16 the old ones, half the registers of F32 values (64, no spills).
    __nv_bfloat162 vb[PER / 2];
    float vf[IN16 ? 1 : PER];
    if constexpr (IN16) {
        const __nv_bfloat16 * res = (const __nv_bfloat16 *) p.res + t * p.res_ld;
#pragma unroll
        for (int m = 0; m < PER / 2; m++) {
            vb[m] = __halves2bfloat162(res[threadIdx.x + 2 * m * HC_THREADS],
                                       res[threadIdx.x + (2 * m + 1) * HC_THREADS]);
        }
    } else {
        const float * res = p.res + t * p.res_ld;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            vf[i] = res[threadIdx.x + i * HC_THREADS];
        }
    }
    auto stream = [&](int i) {
        return (i & 1) ? __high2float(vb[i / 2]) : __low2float(vb[i / 2]);
    };
    float wpost[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
        const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
        const float s = 1.0f / (1.0f + expf(-v));
        wpost[h] = s * 2.0f;
    }
    float sq[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        sq[h] = 0.0f;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const int i = h * PS + r;
            float old;
            if constexpr (IN16) {
                old = stream(i);
            } else {
                old = vf[i];
            }
            const __nv_bfloat16 b = __float2bfloat16(__fmaf_rn(ov[r], wpost[h], old));
            rn[threadIdx.x + i * HC_THREADS] = b;
            if (i & 1) {
                vb[i / 2].y = b;
            } else {
                vb[i / 2].x = b;
            }
            const float v = __bfloat162float(b);
            sq[h] = __fadd_rn(sq[h], __fmul_rn(v, v));
        }
    }
    block_sums(sq, red);
    float scale[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        scale[h] = 1.0f / sqrtf(sq[h] / p.e + p.eps);
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int h = 0; h < HC; h++) {
            p.res_new[p.scale_at + t * HC + h] = scale[h];
        }
    }
    float acc[NINJ];
#pragma unroll
    for (int j = 0; j < NINJ; j++) {
        acc[j] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int k = threadIdx.x + i * HC_THREADS;
        const float y = __fmul_rn(__fmul_rn(stream(i), scale[i / PS]), p.g[k]);
        q8_1_mmq_store(p.xq, p.n_tok, t, k, y);
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
        }
    }
    block_sums(acc, red);
    if (threadIdx.x < NINJ) {
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * NINJ + j] = acc[j];
            }
        }
    }
}

// hc_post_norm_moe (#774): hc_post_norm_v70_kernel<HC, NINJ, PS> (sm_70, no xq) with the token's
// `out` row made from the MoE op's pairs, as moe_gemm.cu's reduce_kernel makes it on sm_70 (its
// per-column path): column c of the sum is d[pair j][c] over the token's valid pairs in slot
// order, the first starting the sum and each later one added (one FADD), then with a shared
// expert sum + sh * sigmoid(gate) (the product and the sum rounded apart, ggml's sigmoid
// expression). Each thread sums its own PS columns, the same values in the same order: the same
// bits as reduce_kernel's row, and from there the same pass as hc_post_norm_v70_kernel.
// Pairs whose loads are issued together; at most MOE_MAX_USED pairs per token (valid bits).
constexpr int MOE_GROUP = 4, MOE_MAX_USED = 16;

// Token t's sum for this thread's PS columns into ov (64 registers, no spills on sm_70/75).
template <int PS>
__device__ __forceinline__ void moe_pair_row(const legion_moe_pair_sum & s, int64_t t, float * ov) {
#pragma unroll
    for (int r = 0; r < PS; r++) {
        ov[r] = 0.0f;
    }
    const float * d = s.d + t * (int64_t) s.n_used * (PS * HC_THREADS);
    // reduce_kernel's expert_ok, a bit per pair: a routing entry the weights do not hold is left
    // out. Lane j of each warp checks pair j (n_used <= 16), so the ids and table reads are one
    // wait each, not one per pair in turn.
    const int lane = threadIdx.x & 31;
    bool ok = false;
    if (lane < s.n_used) {
        const int ex = s.ids[t * s.ids_ld + lane];
        if (ex >= 0 && ex < s.n_expert) {
            const int64_t slot = s.table ? s.table[ex] : ex;
            ok = slot >= 0 && slot < s.n_slots;
        }
    }
    const uint32_t valid = __ballot_sync(0xffffffffu, ok);
    // the shared expert's gate read now, beside the pairs
    const float sgate = s.sh ? 1.0f / (1.0f + expf(-s.sg[t * s.sg_ld])) : 0.0f;
    // MOE_GROUP pairs' loads in flight before their sums (one wait per group, not per pair);
    // the sums still in slot order
    bool any = false;
    for (int j0 = 0; j0 < s.n_used; j0 += MOE_GROUP) {
        float v[MOE_GROUP][PS];
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if ((valid >> (j0 + q)) & 1u) {
                const float * dj = d + (int64_t) (j0 + q) * (PS * HC_THREADS);
#pragma unroll
                for (int r = 0; r < PS; r++) {
                    v[q][r] = __ldcs(dj + threadIdx.x + r * HC_THREADS);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if ((valid >> (j0 + q)) & 1u) {
#pragma unroll
                for (int r = 0; r < PS; r++) {
                    ov[r] = any ? __fadd_rn(ov[r], v[q][r]) : v[q][r];
                }
                any = true;
            }
        }
    }
    if (s.sh) {
        // sgate: ggml-cuda's op_sigmoid (unary.cu), built with fast math as this file is
        const float * sh = s.sh + t * s.sh_ld;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            ov[r] = __fadd_rn(ov[r], __fmul_rn(sh[threadIdx.x + r * HC_THREADS], sgate));
        }
    }
}

// SM80 (#826, sm_80+): hc_post_norm_v70_kernel<HC, NINJ, PS, true, Q>'s pass instead (each
// square added as FMUL then FADD, the L2 prefetch of gamma and the inject weights) after the
// same pair sum, which reduce_kernel's reduce4 path makes with the same bits on sm_80+ (each
// column's pairs in slot order, one FADD each, the shared expert's product and sum rounded
// apart). Q: also xn quantized into p.xq, and with p.scale_at the streams' norm scales instead
// of xn (the rebuilt form, hc_up_mix.cu).
template <int HC, int NINJ, int PS, bool SM80 = false, bool Q = false>
__global__ void __launch_bounds__(HC_THREADS, 4) hc_post_norm_moe_kernel(legion_hc_post_norm p, legion_moe_pair_sum s) {
    static_assert(HC <= HC_MAX && NINJ <= HC_INJ_MAX, "hc_post_norm_moe_kernel: too many sums");
    static_assert(SM80 || !Q, "hc_post_norm_moe_kernel: xq on sm_80+ only");
    constexpr int PER = HC * PS;
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * res = p.res + t * p.res_ld;
    const int n = PER * HC_THREADS;
    float * rn = p.res_new + t * n;
    if constexpr (SM80) {
        // As hc_post_norm_v70_kernel's SM80 hint (#836): a hint, no value changes.
        const int64_t gl = n / 32, lines = gl + (int64_t) n * NINJ / 32;
        for (int64_t u = blockIdx.x * (int64_t) HC_THREADS + threadIdx.x; u < lines;
             u += (int64_t) gridDim.x * HC_THREADS) {
            const float * a = u < gl ? p.g + u * 32 : p.w + (u - gl) * 32;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
    }
    float ov[PS];
    moe_pair_row<PS>(s, t, ov);
    float val[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        val[i] = res[threadIdx.x + i * HC_THREADS];
    }
    float wpost[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
        const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
        const float sg = 1.0f / (1.0f + expf(-v));
        wpost[h] = sg * 2.0f;
    }
    float sq[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        sq[h] = 0.0f;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const int i = h * PS + r;
            // dsv4_hc_post_f32: sum = x * post; sum += residual
            const float v = __fmaf_rn(ov[r], wpost[h], val[i]);
            rn[threadIdx.x + i * HC_THREADS] = v;
            val[i] = v;
            sq[h] = SM80 ? __fadd_rn(sq[h], __fmul_rn(v, v)) : __fmaf_rn(v, v, sq[h]);
        }
    }
    block_sums(sq, red);
    float scale[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        scale[h] = 1.0f / sqrtf(sq[h] / p.e + p.eps);
    }
    if constexpr (Q) {
        if (p.scale_at != 0 && threadIdx.x == 0) {
#pragma unroll
            for (int h = 0; h < HC; h++) {
                p.res_new[p.scale_at + t * HC + h] = scale[h];
            }
        }
    }
    float acc[NINJ];
#pragma unroll
    for (int j = 0; j < NINJ; j++) {
        acc[j] = 0.0f;
    }
    float * xn = p.xn + t * n;
    // Two copies of the pass, with and without the xn store, as in hc_post_norm_v70_kernel.
    auto pass = [&](auto write_xn) {
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int k = threadIdx.x + i * HC_THREADS;
            const float y = __fmul_rn(__fmul_rn(val[i], scale[i / PS]), p.g[k]);
            if constexpr (decltype(write_xn)::value) {
                xn[k] = y;
            }
            if constexpr (Q) {
                q8_1_mmq_store(p.xq, p.n_tok, t, k, y);
            } else if (p.xn16 != nullptr) {
                p.xn16[t * n + k] = __float2half(y);
            }
#pragma unroll
            for (int j = 0; j < NINJ; j++) {
                acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
            }
        }
    };
    if (!Q || p.scale_at == 0) {
        pass(std::true_type{});
    } else {
        pass(std::false_type{});
    }
    block_sums(acc, red);
    if (threadIdx.x < NINJ) {
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * NINJ + j] = acc[j];
            }
        }
    }
}

// Token t's MoE tail for this thread's PS columns into ov (legion_moe_fold): each column's
// experts in order, MOE_GROUP experts' loads in flight before their sums, then the gate. The same
// values in the same order as the tail's kernels: their bits.
template <int PS, bool FMA, bool W, bool X16 = false>
__device__ __forceinline__ void moe_fold_row(const legion_moe_fold & f, int64_t t, float * ov) {
    constexpr int E = PS * HC_THREADS;
    const float * x = f.x + t * (int64_t) f.k * E;
    const __nv_bfloat16 * x16 = f.x16 + t * (int64_t) f.k * E;
    // ggml-cuda's op_sigmoid (unary.cu), built with fast math as this file is
    const float sgate = 1.0f / (1.0f + expf(-f.g[t * f.g_ld]));
    for (int j0 = 0; j0 < f.k; j0 += MOE_GROUP) {
        float v[MOE_GROUP][PS];
        float wj[MOE_GROUP];
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if (j0 + q < f.k) {
                const float * xj = x + (int64_t) (j0 + q) * E;
                const __nv_bfloat16 * xj16 = x16 + (int64_t) (j0 + q) * E;
                wj[q] = W ? f.w[t * f.k + j0 + q] : 1.0f;
#pragma unroll
                for (int r = 0; r < PS; r++) {
                    v[q][r] = X16 ? __bfloat162float(xj16[threadIdx.x + r * HC_THREADS]) : xj[threadIdx.x + r * HC_THREADS];
                }
            }
        }
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if (j0 + q < f.k) {
#pragma unroll
                for (int r = 0; r < PS; r++) {
                    if constexpr (!W) {
                        // ggml's adds of the weighted views (moe_sum_gate_kernel's x[0], then +)
                        ov[r] = j0 + q == 0 ? v[q][r] : __fadd_rn(ov[r], v[q][r]);
                    } else if (j0 + q == 0) {
                        ov[r] = __fmul_rn(v[q][r], wj[q]);
                    } else if constexpr (FMA) {
                        // moe_weighted_reduction_f32: sum += (x * 1) * w, contracted
                        ov[r] = __fmaf_rn(v[q][r], wj[q], ov[r]);
                    } else {
                        ov[r] = __fadd_rn(ov[r], __fmul_rn(v[q][r], wj[q]));
                    }
                }
            }
        }
    }
    const float * sh = f.sh + t * E;
#pragma unroll
    for (int r = 0; r < PS; r++) {
        ov[r] = __fadd_rn(ov[r], __fmul_rn(sh[threadIdx.x + r * HC_THREADS], sgate));
    }
}

// hc_post_norm_fold (#836): hc_post_norm_v70_kernel<4, 4, 10, true, false, true, true>'s pass
// (sm_80+, the BF16 xn and the streams' scales, the L2 prefetch of gamma and the inject weights)
// after moe_fold_row: from the token's out row on, the same values in the same order.
// PRE (#1183, lazy streams, p.pre_out): res holds the streams before the previous block's
// combine, which is made first per value as hc_post_norm_v70_kernel made it in that block's
// norm (the same inject expressions and the same FMA, so the same bits).
template <bool FMA, bool W, bool X16 = false, bool PRE = false>
__global__ void __launch_bounds__(HC_THREADS, 4) hc_post_norm_fold_kernel(legion_hc_post_norm p, legion_moe_fold f) {
    constexpr int HC = 4, NINJ = 4, PS = 10, PER = HC * PS;
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    const float * res = p.res + t * p.res_ld;
    const int n = PER * HC_THREADS;
    float * rn = p.res_new + t * n;
    {
        // As hc_post_norm_v70_kernel's SM80 hint: a hint, no value changes.
        const int64_t gl = n / 32, lines = gl + (int64_t) n * NINJ / 32;
        for (int64_t u = blockIdx.x * (int64_t) HC_THREADS + threadIdx.x; u < lines;
             u += (int64_t) gridDim.x * HC_THREADS) {
            const float * a = u < gl ? p.g + u * 32 : p.w + (u - gl) * 32;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
    }
    float ov[PS];
    moe_fold_row<PS, FMA, W, X16>(f, t, ov);
    float val[PER];
    if constexpr (PRE) {
        float wpre[HC];
#pragma unroll
        for (int h = 0; h < HC; h++) {
            // ggml: scale(inject, 1 / hc), sigmoid, scale(2), as wpost below
            const float v = p.pre_inj[t * p.pre_inj_ld + h] * (1.0f / (float) p.hc);
            const float s = 1.0f / (1.0f + expf(-v));
            wpre[h] = s * 2.0f;
        }
        const float * po = p.pre_out + t * p.pre_out_ld;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const float o = po[threadIdx.x + r * HC_THREADS];
#pragma unroll
            for (int h = 0; h < HC; h++) {
                const int i = h * PS + r;
                val[i] = __fmaf_rn(o, wpre[h], res[threadIdx.x + i * HC_THREADS]);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < PER; i++) {
            val[i] = res[threadIdx.x + i * HC_THREADS];
        }
    }
    float wpost[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
        const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
        const float s = 1.0f / (1.0f + expf(-v));
        wpost[h] = s * 2.0f;
    }
    float sq[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        sq[h] = 0.0f;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const int i = h * PS + r;
            // dsv4_hc_post_f32: sum = x * post; sum += residual
            const float v = __fmaf_rn(ov[r], wpost[h], val[i]);
            rn[threadIdx.x + i * HC_THREADS] = v;
            val[i] = v;
            sq[h] = __fadd_rn(sq[h], __fmul_rn(v, v));
        }
    }
    block_sums(sq, red);
    float scale[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        scale[h] = 1.0f / sqrtf(sq[h] / p.e + p.eps);
    }
    if (p.scale_at != 0 && threadIdx.x == 0) {
#pragma unroll
        for (int h = 0; h < HC; h++) {
            p.res_new[p.scale_at + t * HC + h] = scale[h];
        }
    }
    float acc[NINJ];
#pragma unroll
    for (int j = 0; j < NINJ; j++) {
        acc[j] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int k = threadIdx.x + i * HC_THREADS;
        const float y = __fmul_rn(__fmul_rn(val[i], scale[i / PS]), p.g[k]);
        p.xnb[t * n + k] = __float2bfloat16(y);
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            acc[j] = __fmaf_rn(y, p.w[k + (int64_t) n * j], acc[j]);
        }
    }
    block_sums(acc, red);
    if (threadIdx.x < NINJ) {
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * NINJ + j] = acc[j];
            }
        }
    }
}

// The BF16-stream kernels' layout (#1183): qwen4exp's 4 x 2560 values a token as pairs, thread
// x taking pairs x + m * HC_THREADS (m < B16_PAIRS, B16_PS of them per stream). Pair q holds
// values 2q and 2q + 1 of the token.
constexpr int B16_PS = 5, B16_HC = 4, B16_PAIRS = B16_HC * B16_PS;

// One column of a folded MoE tail: moe_fold_row's arithmetic for the expert at slot j (j0 the
// first) of value v with router weight w.
template <bool FMA, bool W>
__device__ __forceinline__ float moe_fold_step(float o, float v, float w, bool first) {
    if constexpr (!W) {
        return first ? v : __fadd_rn(o, v);
    } else if (first) {
        return __fmul_rn(v, w);
    } else if constexpr (FMA) {
        return __fmaf_rn(v, w, o);
    } else {
        return __fadd_rn(o, __fmul_rn(v, w));
    }
}

// moe_fold_row for the pair layout: token t's MoE tail for this thread's B16_PS column pairs of
// the e = 2560 columns into ov. Each column's value is moe_fold_row's (the same experts in the
// same order, the same roundings), loaded two columns at a time.
template <bool FMA, bool W, bool X16>
__device__ __forceinline__ void moe_fold_pairs(const legion_moe_fold & f, int64_t t, float2 * ov) {
    constexpr int E = 2 * B16_PS * HC_THREADS;
    const float2 * x = (const float2 *) (f.x + t * (int64_t) f.k * E);
    const __nv_bfloat162 * x16 = (const __nv_bfloat162 *) (f.x16 + t * (int64_t) f.k * E);
    const float sgate = 1.0f / (1.0f + expf(-f.g[t * f.g_ld]));
    for (int j0 = 0; j0 < f.k; j0 += MOE_GROUP) {
        float2 v[MOE_GROUP][B16_PS];
        float wj[MOE_GROUP];
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if (j0 + q < f.k) {
                const int64_t row = (int64_t) (j0 + q) * (E / 2);
                wj[q] = W ? f.w[t * f.k + j0 + q] : 1.0f;
#pragma unroll
                for (int r = 0; r < B16_PS; r++) {
                    const int64_t at = row + threadIdx.x + r * HC_THREADS;
                    v[q][r] = X16 ? __bfloat1622float2(x16[at]) : x[at];
                }
            }
        }
#pragma unroll
        for (int q = 0; q < MOE_GROUP; q++) {
            if (j0 + q < f.k) {
#pragma unroll
                for (int r = 0; r < B16_PS; r++) {
                    ov[r].x = moe_fold_step<FMA, W>(ov[r].x, v[q][r].x, wj[q], j0 + q == 0);
                    ov[r].y = moe_fold_step<FMA, W>(ov[r].y, v[q][r].y, wj[q], j0 + q == 0);
                }
            }
        }
    }
    const float2 * sh = (const float2 *) (f.sh + t * E);
#pragma unroll
    for (int r = 0; r < B16_PS; r++) {
        const float2 s = sh[threadIdx.x + r * HC_THREADS];
        ov[r].x = __fadd_rn(ov[r].x, __fmul_rn(s.x, sgate));
        ov[r].y = __fadd_rn(ov[r].y, __fmul_rn(s.y, sgate));
    }
}

// hc_post_norm_v70_kernel<4, 4, 10, true, false, true, true> (the rebuilt BF16-xn form, sm_80+)
// with BF16 streams (#1183, legion_hc_post_norm_b16_launch): res' = out * w + res rounded to
// BF16 and written as BF16, the rounded values normed, rounded to BF16 into xnb and injected.
// Half the bytes of the streams written (and with IN16 read); the pair layout makes every
// access a full 128-byte line per warp (a BF16 one per value would move 64). FOLD: out is a
// folded MoE tail (moe_fold_pairs), as hc_post_norm_fold_kernel makes it.
template <bool IN16, bool FOLD, bool FMA = false, bool W = false, bool X16 = false>
__global__ void __launch_bounds__(HC_THREADS, 4) hc_post_norm_b16_kernel(legion_hc_post_norm p, legion_moe_fold f) {
    constexpr int HC = B16_HC, NINJ = 4, PS = B16_PS, PAIRS = B16_PAIRS;
    constexpr int n = 2 * PAIRS * HC_THREADS;
    __shared__ float red[HC_MAX][HC_THREADS / 32];
    const int64_t t = blockIdx.x;
    {
        // As hc_post_norm_v70_kernel's SM80 hint: a hint, no value changes.
        const int64_t gl = n / 32, lines = gl + (int64_t) n * NINJ / 32;
        for (int64_t u = blockIdx.x * (int64_t) HC_THREADS + threadIdx.x; u < lines;
             u += (int64_t) gridDim.x * HC_THREADS) {
            const float * a = u < gl ? p.g + u * 32 : p.w + (u - gl) * 32;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
        }
    }
    float2 ov[PS] = {};
    if constexpr (FOLD) {
        moe_fold_pairs<FMA, W, X16>(f, t, ov);
    } else {
        const float * out = p.out + t * p.out_ld;
        if ((((uintptr_t) out) & 7) == 0) {
#pragma unroll
            for (int r = 0; r < PS; r++) {
                ov[r] = ((const float2 *) out)[threadIdx.x + r * HC_THREADS];
            }
        } else {
#pragma unroll
            for (int r = 0; r < PS; r++) {
                const int c = 2 * (threadIdx.x + r * HC_THREADS);
                ov[r] = make_float2(out[c], out[c + 1]);
            }
        }
    }
    // The streams as BF16 pairs: the old ones (IN16), then the rounded new ones for pass 2.
    __nv_bfloat162 vb[PAIRS];
    float2 vf[IN16 ? 1 : PAIRS];
    if constexpr (IN16) {
        const __nv_bfloat162 * res = (const __nv_bfloat162 *) ((const __nv_bfloat16 *) p.res + t * p.res_ld);
#pragma unroll
        for (int i = 0; i < PAIRS; i++) {
            vb[i] = res[threadIdx.x + i * HC_THREADS];
        }
    } else {
        const float2 * res = (const float2 *) (p.res + t * p.res_ld);
#pragma unroll
        for (int i = 0; i < PAIRS; i++) {
            vf[i] = res[threadIdx.x + i * HC_THREADS];
        }
    }
    float wpost[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
        const float v = p.inj[t * p.inj_ld + h] * (1.0f / (float) p.hc);
        const float s = 1.0f / (1.0f + expf(-v));
        wpost[h] = s * 2.0f;
    }
    __nv_bfloat162 * rn = (__nv_bfloat162 *) ((__nv_bfloat16 *) p.res_new + t * n);
    float sq[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        sq[h] = 0.0f;
#pragma unroll
        for (int r = 0; r < PS; r++) {
            const int i = h * PS + r;
            float2 old;
            if constexpr (IN16) {
                old = __bfloat1622float2(vb[i]);
            } else {
                old = vf[i];
            }
            // dsv4_hc_post_f32: sum = x * post; sum += residual; then rounded to BF16
            const __nv_bfloat162 b = __floats2bfloat162_rn(__fmaf_rn(ov[r].x, wpost[h], old.x),
                                                          __fmaf_rn(ov[r].y, wpost[h], old.y));
            rn[threadIdx.x + i * HC_THREADS] = b;
            vb[i] = b;
            const float2 v = __bfloat1622float2(b);
            sq[h] = __fadd_rn(sq[h], __fmul_rn(v.x, v.x));
            sq[h] = __fadd_rn(sq[h], __fmul_rn(v.y, v.y));
        }
    }
    block_sums(sq, red);
    float scale[HC];
#pragma unroll
    for (int h = 0; h < HC; h++) {
        scale[h] = 1.0f / sqrtf(sq[h] / p.e + p.eps);
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int h = 0; h < HC; h++) {
            p.res_new[p.scale_at + t * HC + h] = scale[h];
        }
    }
    float acc[NINJ];
#pragma unroll
    for (int j = 0; j < NINJ; j++) {
        acc[j] = 0.0f;
    }
    const float2 * g = (const float2 *) p.g;
    // (Null where the down matmul rebuilds xnb itself, src/cuda/hc_down.cu, #1183.)
    __nv_bfloat162 * xnb =
        p.xnb != nullptr ? (__nv_bfloat162 *) ((__nv_bfloat16 *) p.xnb + t * n) : nullptr;
#pragma unroll
    for (int i = 0; i < PAIRS; i++) {
        const int k = threadIdx.x + i * HC_THREADS;
        const float2 v = __bfloat1622float2(vb[i]);
        const float2 gk = g[k];
        // (rn * scale) * gamma, as up_mix_bf16 rebuilds it from the BF16 streams
        const float y0 = __fmul_rn(__fmul_rn(v.x, scale[i / PS]), gk.x);
        const float y1 = __fmul_rn(__fmul_rn(v.y, scale[i / PS]), gk.y);
        if (xnb != nullptr) {
            xnb[k] = __floats2bfloat162_rn(y0, y1);
        }
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            const float2 wk = ((const float2 *) (p.w + (int64_t) n * j))[k];
            acc[j] = __fmaf_rn(y0, wk.x, acc[j]);
            acc[j] = __fmaf_rn(y1, wk.y, acc[j]);
        }
    }
    block_sums(acc, red);
    if (threadIdx.x < NINJ) {
#pragma unroll
        for (int j = 0; j < NINJ; j++) {
            if (j == (int) threadIdx.x) {
                p.inject[t * NINJ + j] = acc[j];
            }
        }
    }
}

}  // namespace

int legion_moe_combine_launch(const legion_moe_combine * p, void * stream) {
    if (p->n_tok < 1) {
        return -1;
    }
    constexpr int threads = 256;
    const dim3 blocks(p->n_tok, (unsigned) ((p->n_embd + threads - 1) / threads), 1);
    moe_combine_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_shexp_gate_launch(const legion_shexp_gate * p, void * stream) {
    if (p->n_tok < 1) {
        return -1;
    }
    constexpr int threads = 256;
    const dim3 blocks(p->n_tok, (unsigned) ((p->n_embd + threads - 1) / threads), 1);
    shexp_gate_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_gated_norm_launch(const legion_gated_norm * p, void * stream) {
    if (p->head_v > PREP_BLOCK || p->n_tok < 1 || p->n_tok > 65535) {
        return -1;
    }
    gated_norm_kernel<<<dim3(p->n_v, p->n_tok), PREP_BLOCK, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_rms_norm_bf16_launch(const legion_rms_norm_bf16 * p, void * stream) {
    if (p->ncols < 1 || p->n_rows < 1) {
        return -1;
    }
    if (p->ncols < 1024) {
        rms_norm_bf16_kernel<256><<<p->n_rows, 256, 0, (cudaStream_t) stream>>>(*p);
    } else {
        rms_norm_bf16_kernel<1024><<<p->n_rows, 1024, 0, (cudaStream_t) stream>>>(*p);
    }
    return (int) cudaGetLastError();
}

int legion_gate_bf16_launch(const legion_gate_bf16 * p, void * stream) {
    const int64_t k = (int64_t) p->d * p->n_head;
    if (p->d < 1 || p->n_head < 1 || k > INT32_MAX || p->n_tok < 1 || p->n_tok > 65535) {
        return -1;
    }
    constexpr int threads = 256;
    const dim3 blocks((unsigned) ((k + threads - 1) / threads), p->n_tok, 1);
    gate_bf16_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_delta_prep_launch(const legion_delta_prep * p, void * stream) {
    if (p->head_k > PREP_BLOCK || p->n_tok < 1) {
        return -1;
    }
    const int heads = p->gates_only ? 0 : 2 * p->n_k;
    delta_prep_kernel<<<dim3(heads + 1, p->n_tok), PREP_BLOCK, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

static thread_local int t_hc_generic = 0;

void legion_hc_post_norm_force_generic(int on) {
    t_hc_generic = on;
}

static int g_hc_generic = 0;

void legion_hc_post_norm_set_generic(int on) {
    __atomic_store_n(&g_hc_generic, on, __ATOMIC_RELAXED);
}

// Q: the variants that also write p->xq (sm_80+ only, so no sm_70 ones).
template <bool Q>
static void hc_post_norm_dispatch(const legion_hc_post_norm * p, cudaStream_t st, int major, int minor) {
    const int n = p->e * p->hc;
    // Register variants for whole multiples of the block (qwen4exp: 2560 x 4 = 40 per thread).
    // legion_hc_post_norm_set_generic: the generic kernel everywhere (A/B runs).
    const bool all_generic = __atomic_load_n(&g_hc_generic, __ATOMIC_RELAXED) != 0;
    // sm_80+: hc_post_norm_reg_kernel; sm_70: hc_post_norm_v70_kernel (each pinned to what nvcc
    // makes of the generic kernel for that architecture).
    if (t_hc_generic == 1 || all_generic) {
        hc_post_norm_kernel<Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else if (major < 8) {
        if constexpr (!Q) {
            if (major == 7 && minor == 0 && p->hc == 4 && p->n_inj == 4 && p->e == 10 * HC_THREADS) {
                hc_post_norm_v70_kernel<4, 4, 10><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
            } else {
                hc_post_norm_kernel<false><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
            }
        }
    } else if (t_hc_generic == 0 && p->hc == 4 && p->n_inj == 4 && p->e == 10 * HC_THREADS) {
        // qwen4exp's 4 x 2560 streams (tests: legion_hc_post_norm_force_generic(2) takes the
        // register kernel below).
        hc_post_norm_v70_kernel<4, 4, 10, true, Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else if (n % HC_THREADS == 0 && n / HC_THREADS == 40) {
        hc_post_norm_reg_kernel<40, Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else if (n % HC_THREADS == 0 && n / HC_THREADS == 32) {
        hc_post_norm_reg_kernel<32, Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else if (n % HC_THREADS == 0 && n / HC_THREADS == 16) {
        hc_post_norm_reg_kernel<16, Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else {
        hc_post_norm_kernel<Q><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    }
}

int legion_hc_post_norm_launch(const legion_hc_post_norm * p, void * stream) {
    if (p->hc > HC_MAX || p->n_inj > HC_INJ_MAX || p->pre_out != nullptr) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (p->xnb != nullptr) {
        // The BF16 xn (#469): sm_80+, alone. qwen4exp's 4 x 2560 streams take the register
        // kernel's B16 variant, everything else (and the generic switches) the generic kernel.
        if (major < 8 || p->xq != nullptr || p->xn16 != nullptr) {
            return -1;
        }
        if (p->scale_at != 0) {
            // The scales instead of xn (#836, hc_mix16.cu rebuilds xn): the register kernel at
            // qwen4exp's shape only, the generic switches aside (the generic kernel writes xn).
            if (p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS) {
                return -1;
            }
            if (p->no_res != 0) {
                // Lazy streams (#1183): the new streams not written.
                hc_post_norm_v70_kernel<4, 4, 10, true, false, true, true, false, true>
                    <<<p->n_tok, HC_THREADS, 0, st>>>(*p);
            } else {
                hc_post_norm_v70_kernel<4, 4, 10, true, false, true, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
            }
            return (int) cudaGetLastError();
        }
        if (p->no_res != 0) {
            return -1;
        }
        const bool all_generic = __atomic_load_n(&g_hc_generic, __ATOMIC_RELAXED) != 0;
        if (t_hc_generic == 0 && !all_generic && p->hc == 4 && p->n_inj == 4 && p->e == 10 * HC_THREADS) {
            hc_post_norm_v70_kernel<4, 4, 10, true, false, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
        } else {
            hc_post_norm_kernel<false><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
        }
        return (int) cudaGetLastError();
    }
    if (p->no_res != 0) {
        return -1;
    }
    if (p->xq == nullptr && p->scale_at != 0) {
        // The scales instead of xn, with xn16 (#774): the V100 kernel at qwen4exp's shape only.
        if (major != 7 || minor != 0 || p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS ||
            p->xn16 == nullptr) {
            return -1;
        }
        hc_post_norm_v70_kernel<4, 4, 10, false, false, false, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
        return (int) cudaGetLastError();
    }
    if (p->xq != nullptr) {
        // q8_1_mmq_store: whole warps over whole scale groups; MMQ: whole 512-value rows.
        // The Q kernels write no xn16.
        if (major < 8 || (p->e * p->hc) % 512 != 0 || p->xn16 != nullptr) {
            return -1;
        }
        hc_post_norm_dispatch<true>(p, st, major, minor);
    } else {
        hc_post_norm_dispatch<false>(p, st, major, minor);
    }
    return (int) cudaGetLastError();
}

int legion_hc_post_norm_rs_launch(const legion_hc_post_norm * p, void * stream) {
    if (p->pre_out != nullptr || p->no_res != 0) {
        return -1;  // (lazy streams: legion_hc_post_norm_launch / _fold_launch only)
    }
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    // The rebuilt BF16-xn form legion_hc_post_norm_launch takes for xnb with scale_at.
    if (major < 8 || p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS || p->xnb == nullptr ||
        p->scale_at == 0 || p->xq != nullptr || p->xn16 != nullptr || p->rs == nullptr || p->n_tok < 1) {
        return -1;
    }
    hc_post_norm_v70_kernel<4, 4, 10, true, false, true, true, true>
        <<<p->n_tok, HC_THREADS, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_hc_post_norm_s16_launch(const legion_hc_post_norm * p, int in16, void * stream) {
    if (p->pre_out != nullptr || p->no_res != 0) {
        return -1;  // (lazy streams: legion_hc_post_norm_launch / _fold_launch only)
    }
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 8 || p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS || p->xq == nullptr ||
        p->scale_at == 0 || p->xn16 != nullptr || p->xnb != nullptr || p->n_tok < 1) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (in16) {
        hc_post_norm_s16_kernel<true><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else {
        hc_post_norm_s16_kernel<false><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}

int legion_hc_post_norm_moe_launch(const legion_hc_post_norm * p, const legion_moe_pair_sum * s, void * stream) {
    if (p->pre_out != nullptr || p->no_res != 0) {
        return -1;  // (lazy streams: legion_hc_post_norm_launch / _fold_launch only)
    }
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS || p->xnb != nullptr || s->n_used < 1 ||
        s->n_used > MOE_MAX_USED || p->n_tok < 1) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (major >= 8) {
        // #826: the GB10's prompt steps, with xq (and the scales instead of xn) or plain; no xn16.
        if (p->xn16 != nullptr || (p->xq == nullptr && p->scale_at != 0)) {
            return -1;
        }
        if (p->xq != nullptr) {
            hc_post_norm_moe_kernel<4, 4, 10, true, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p, *s);
        } else {
            hc_post_norm_moe_kernel<4, 4, 10, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p, *s);
        }
        return (int) cudaGetLastError();
    }
    if (major != 7 || minor != 0 || p->xq != nullptr || p->scale_at != 0) {
        return -1;
    }
    hc_post_norm_moe_kernel<4, 4, 10><<<p->n_tok, HC_THREADS, 0, st>>>(*p, *s);
    return (int) cudaGetLastError();
}

// hc_post_norm_fold_kernel<FMA, W, X16>, its lazy-streams form for p->pre_out.
template <bool FMA, bool W, bool X16>
static void hc_post_norm_fold_go(const legion_hc_post_norm * p, const legion_moe_fold * f, cudaStream_t st) {
    if (p->pre_out != nullptr) {
        hc_post_norm_fold_kernel<FMA, W, X16, true><<<p->n_tok, HC_THREADS, 0, st>>>(*p, *f);
    } else {
        hc_post_norm_fold_kernel<FMA, W, X16><<<p->n_tok, HC_THREADS, 0, st>>>(*p, *f);
    }
}

int legion_hc_post_norm_fold_launch(const legion_hc_post_norm * p, const legion_moe_fold * f, void * stream) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 8 || p->hc != 4 || p->n_inj != 4 || p->e != 10 * HC_THREADS || p->xnb == nullptr ||
        p->scale_at == 0 || p->xq != nullptr || p->xn16 != nullptr || p->n_tok < 1 || f->k < 1 ||
        f->sh == nullptr || (f->w == nullptr && f->fma != 0) || (f->x == nullptr) == (f->x16 == nullptr) ||
        (f->x16 != nullptr && f->w == nullptr) || p->no_res != 0 || (p->pre_out == nullptr) != (p->pre_inj == nullptr)) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (f->x16 != nullptr) {
        // BF16 rows (--nvfp4-expert-rows-bf16, #1183): with the router weights only.
        if (f->fma != 0) {
            hc_post_norm_fold_go<true, true, true>(p, f, st);
        } else {
            hc_post_norm_fold_go<false, true, true>(p, f, st);
        }
    } else if (f->w == nullptr) {
        hc_post_norm_fold_go<false, false, false>(p, f, st);
    } else if (f->fma != 0) {
        hc_post_norm_fold_go<true, true, false>(p, f, st);
    } else {
        hc_post_norm_fold_go<false, true, false>(p, f, st);
    }
    return (int) cudaGetLastError();
}

// hc_post_norm_b16_kernel of a folded MoE tail, the fold's variant picked as
// legion_hc_post_norm_fold_launch picks it.
template <bool IN16>
static void hc_post_norm_b16_fold(const legion_hc_post_norm * p, const legion_moe_fold * f, cudaStream_t st) {
    const unsigned grid = (unsigned) p->n_tok;
    if (f->x16 != nullptr) {
        if (f->fma != 0) {
            hc_post_norm_b16_kernel<IN16, true, true, true, true><<<grid, HC_THREADS, 0, st>>>(*p, *f);
        } else {
            hc_post_norm_b16_kernel<IN16, true, false, true, true><<<grid, HC_THREADS, 0, st>>>(*p, *f);
        }
    } else if (f->w == nullptr) {
        hc_post_norm_b16_kernel<IN16, true, false, false><<<grid, HC_THREADS, 0, st>>>(*p, *f);
    } else if (f->fma != 0) {
        hc_post_norm_b16_kernel<IN16, true, true, true><<<grid, HC_THREADS, 0, st>>>(*p, *f);
    } else {
        hc_post_norm_b16_kernel<IN16, true, false, true><<<grid, HC_THREADS, 0, st>>>(*p, *f);
    }
}

int legion_hc_post_norm_b16_launch(const legion_hc_post_norm * p, const legion_moe_fold * f, int in16, void * stream) {
    if (p->pre_out != nullptr || p->no_res != 0) {
        return -1;  // (lazy streams: legion_hc_post_norm_launch / _fold_launch only)
    }
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    constexpr int E = 2 * B16_PS * HC_THREADS;
    // The pair layout's 4- and 8-byte accesses: every row starts on a pair.
    const bool aligned = (uintptr_t) p->res % (in16 ? 4 : 8) == 0 && p->res_ld % 2 == 0 &&
                         (uintptr_t) p->res_new % 4 == 0 && (uintptr_t) p->xnb % 4 == 0 &&
                         (uintptr_t) p->g % 8 == 0 && (uintptr_t) p->w % 8 == 0;
    if (major < 8 || p->hc != B16_HC || p->n_inj != 4 || p->e != E || p->scale_at == 0 ||
        p->xq != nullptr || p->xn16 != nullptr || p->rs != nullptr || p->n_tok < 1 || !aligned) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (f == nullptr) {
        if (in16) {
            hc_post_norm_b16_kernel<true, false><<<p->n_tok, HC_THREADS, 0, st>>>(*p, legion_moe_fold{});
        } else {
            hc_post_norm_b16_kernel<false, false><<<p->n_tok, HC_THREADS, 0, st>>>(*p, legion_moe_fold{});
        }
        return (int) cudaGetLastError();
    }
    // As legion_hc_post_norm_fold_launch, the rows read in pairs.
    if (f->k < 1 || f->sh == nullptr || (f->w == nullptr && f->fma != 0) || (f->x == nullptr) == (f->x16 == nullptr) ||
        (f->x16 != nullptr && f->w == nullptr) || (uintptr_t) f->sh % 8 != 0 ||
        (f->x != nullptr && (uintptr_t) f->x % 8 != 0) || (f->x16 != nullptr && (uintptr_t) f->x16 % 4 != 0)) {
        return -1;
    }
    if (in16) {
        hc_post_norm_b16_fold<true>(p, f, st);
    } else {
        hc_post_norm_b16_fold<false>(p, f, st);
    }
    return (int) cudaGetLastError();
}

static thread_local int t_hc_norm_fma = 0;

void legion_hc_norm_force_fma(int on) {
    t_hc_norm_fma = on;
}

static int g_hc_norm_fma = 0;

void legion_hc_norm_set_fma(int on) {
    __atomic_store_n(&g_hc_norm_fma, on, __ATOMIC_RELAXED);
}

int legion_hc_norm_launch(const legion_hc_norm * p, void * stream) {
    if (p->hc > HC_MAX || p->n_inj > HC_INJ_MAX) {
        return -1;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    // The pinned variant only where nvcc rounds the two kernels differently (sm_7x); on sm_80+
    // they already agree.
    bool fma = t_hc_norm_fma || __atomic_load_n(&g_hc_norm_fma, __ATOMIC_RELAXED) != 0;
    if (fma) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        fma = major == 7;
    }
    if (fma) {
        hc_norm_kernel<true><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    } else {
        hc_norm_kernel<false><<<p->n_tok, HC_THREADS, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}

int legion_router_topk_launch(const legion_router * p, void * stream) {
    if (p->n_expert > 32 * MAX_PER_LANE || p->k > p->n_expert) {
        return -1;
    }
    constexpr int WARPS = 4;
    if (p->plan) {
        // One CTA: a warp per token, at least a thread per pair of the plan.
        if (p->n_tok < 1 || p->n_tok > 8 || p->n_tok * p->k > nvfp4_plan::MAX_PAIRS) {
            return -1;
        }
        const int warps = p->n_tok > WARPS ? p->n_tok : WARPS;
        router_topk_plan_kernel<<<1, 32 * warps, 0, (cudaStream_t) stream>>>(*p);
        return (int) cudaGetLastError();
    }
    router_topk_kernel<<<(p->n_tok + WARPS - 1) / WARPS, 32 * WARPS, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_router_weights_launch(const legion_router * p, void * stream) {
    router_weights_kernel<<<(p->n_tok + 127) / 128, 128, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_router_select_launch(const legion_router * p, void * stream) {
    if (p->n_expert > 32 * MAX_PER_LANE || p->k > p->n_expert || p->plan) {
        return -1;
    }
    constexpr int WARPS = 4;
    router_select_kernel<<<(p->n_tok + WARPS - 1) / WARPS, 32 * WARPS, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

int legion_conv_silu_launch(const legion_conv_silu * p, void * stream) {
    if (p->d < 1 || p->d > MAX_D) {
        return -1;
    }
    if (p->norm_blocks > 0 && (p->norm_head != (int) p->block || p->norm_head > PREP_BLOCK || p->norm_head % 32 != 0 ||
                               (int64_t) p->norm_blocks * p->norm_head > p->C)) {
        return -2;
    }
    conv_silu_kernel<<<dim3(p->grid[0], p->grid[1], p->grid[2]), p->block, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}
