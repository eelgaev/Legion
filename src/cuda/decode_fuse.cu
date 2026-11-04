// `legion serve --decode-fuse` (#773): a qwen4exp MoE layer's tail in decode-sized steps as one
// kernel instead of ggml's and Legion's small ones. The Rust side (src/model/core/decode_fuse.rs)
// launches it itself.
//
//   w[j]  = expf(l[id_j] - l[id_0]) / sum_j expf(l[id_j] - l[id_0])
//                              the router weights: router_weights_kernel's arithmetic (fused.cu)
//   moe   = experts[0] * w[0], then one fma per expert
//                              moe_combine_kernel's rounding (ggml-cuda's fused weighted sum)
//   out   = moe + sh * sigmoid(g)
//                              shexp_gate_kernel's (ggml's op_sigmoid, mul and add, each rounded)
//
// The same flags as fused.cu and ggml-cuda (--use_fast_math), so the same arithmetic: the bits of
// router_weights + moe_combine + shexp_gate.

#include <cstdint>

extern "C" {

struct legion_moe_tail {
    const float *   experts;  // [n_embd, n_used, n_tok], contiguous
    const float *   logits;   // token t's router logits at t * l_ld
    int64_t         l_ld;
    const int32_t * ids;      // [n_used, n_tok], contiguous
    const float *   sh;       // [n_embd, n_tok], contiguous
    const float *   g;        // token t's gate logit at t * g_ld
    int64_t         g_ld;
    float *         out;      // [n_embd, n_tok]
    int64_t         n_embd;
    int32_t         n_used;   // at most MOE_TAIL_MAX_USED
    int32_t         n_tok;
};

}  // extern "C"

namespace {

constexpr int MOE_TAIL_MAX_USED = 32;

// Blocks (token, column block); thread 0 finds the token's weights first.
__global__ void moe_tail_kernel(const legion_moe_tail p) {
    __shared__ float w[MOE_TAIL_MAX_USED];
    const int64_t t = blockIdx.x;
    if (threadIdx.x == 0) {
        const float *   l   = p.logits + t * p.l_ld;
        const int32_t * ids = p.ids + t * p.n_used;
        const float     m   = l[ids[0]];
        float           sum = 0.0f;
        for (int j = 0; j < p.n_used; j++) {
            sum += expf(l[ids[j]] - m);
        }
        for (int j = 0; j < p.n_used; j++) {
            w[j] = expf(l[ids[j]] - m) / sum;
        }
    }
    __syncthreads();
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= p.n_embd) {
        return;
    }
    const int64_t first = t * p.n_used;
    float         moe   = __fmul_rn(p.experts[first * p.n_embd + col], w[0]);
    for (int e = 1; e < p.n_used; ++e) {
        moe = __fmaf_rn(p.experts[(first + e) * p.n_embd + col], w[e], moe);
    }
    const float   gv = p.g[t * p.g_ld];
    const float   s  = 1.0f / (1.0f + expf(-gv));
    const int64_t i  = t * p.n_embd + col;
    p.out[i]         = __fadd_rn(moe, __fmul_rn(p.sh[i], s));
}

}  // namespace

// The kernel as the CUDA runtime knows it (its host stub), for Rust's cudaLaunchKernel (see
// gate_q8.cu for why it sits in an anonymous namespace).
extern "C" const void * const legion_moe_tail_kernel = reinterpret_cast<const void *>(&moe_tail_kernel);
