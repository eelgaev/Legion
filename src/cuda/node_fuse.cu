// The kernels of the post-allocation node fusions (src/model/core/node_fuse.rs, #814): chains
// of ggml-cuda's small kernels as one launch each, with ggml's arithmetic. The Rust side
// launches them itself (cudaLaunchKernel).
//
//   moe_sum_gate   a MoE layer's weighted expert sum as ggml's unfused chain rounds it (the mul
//                  by the router weights, then one add per expert, in expert order, each
//                  rounded), and the shared expert gated by its sigmoid added: ggml's
//                  op_sigmoid, mul and add, each rounded (shexp_gate_kernel's, fused.cu).
//   ids_remap      the routed expert ids through the layer's expert table: ggml's cont and
//                  get_rows of the I32 ids (integers, exact).
//
// The same flags as ggml-cuda and fused.cu (--use_fast_math), so expf is ggml's.

#include <cstdint>

extern "C" {

struct legion_moe_sum_gate {
    const float * x;   // [n_embd, k, n_tok] contiguous: the experts' outputs, or already weighted
    const float * w;   // [k, n_tok] contiguous router weights; null: x is weighted
    const float * sh;  // [n_embd, n_tok] contiguous shared expert; null: no gate
    const float * g;   // token t's gate logit at g + t * g_ld
    int64_t       g_ld;
    float *       out;  // [n_embd, n_tok] contiguous
    int64_t       n_embd;
    int32_t       k;
    int32_t       n_tok;
};

struct legion_ids_remap {
    const int32_t * ids;    // [ne0, ne1]: id (i0, i1) at ids + i0 * s0 + i1 * s1
    int64_t         s0;
    int64_t         s1;
    int64_t         ne0;
    int64_t         n;      // ne0 * ne1
    const char *    table;  // row r (one I32) at table + r * table_nb1
    int64_t         table_nb1;
    int32_t *       out;    // [n] contiguous
};

}  // extern "C"

namespace {

// Blocks (token, column block), one column per thread.
__global__ void moe_sum_gate_kernel(const legion_moe_sum_gate p) {
    const int64_t t   = blockIdx.x;
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= p.n_embd) {
        return;
    }
    const int64_t first = t * p.k;
    const float * x     = p.x + first * p.n_embd + col;
    // ggml's mul (each product rounded), then its adds: views[0] + views[1], then the sum so far
    // plus the next view, each rounded (no fma).
    float sum = p.w ? __fmul_rn(x[0], p.w[first]) : x[0];
    for (int j = 1; j < p.k; ++j) {
        const float v = p.w ? __fmul_rn(x[(int64_t) j * p.n_embd], p.w[first + j]) : x[(int64_t) j * p.n_embd];
        sum           = __fadd_rn(sum, v);
    }
    const int64_t i = t * p.n_embd + col;
    if (p.sh) {
        // ggml-cuda's op_sigmoid (both built with fast math), then the mul and the add.
        const float gv = p.g[t * p.g_ld];
        const float s  = 1.0f / (1.0f + expf(-gv));
        sum            = __fadd_rn(sum, __fmul_rn(p.sh[i], s));
    }
    p.out[i] = sum;
}

__global__ void ids_remap_kernel(const legion_ids_remap p) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) {
        return;
    }
    const int64_t i0 = i % p.ne0;
    const int64_t i1 = i / p.ne0;
    const int64_t r  = p.ids[i0 * p.s0 + i1 * p.s1];
    p.out[i]         = *(const int32_t *) (p.table + r * p.table_nb1);
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel
// (see gate_q8.cu for why they sit in an anonymous namespace).
extern "C" const void * const legion_moe_sum_gate_kernel = reinterpret_cast<const void *>(&moe_sum_gate_kernel);
extern "C" const void * const legion_ids_remap_kernel    = reinterpret_cast<const void *>(&ids_remap_kernel);
