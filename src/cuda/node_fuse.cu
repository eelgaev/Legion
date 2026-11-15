// The kernels of the post-allocation node fusions (src/model/core/node_fuse.rs, #814): chains
// of ggml-cuda's small kernels as one launch each, with ggml's arithmetic. The Rust side
// launches them itself (cudaLaunchKernel).
//
//   moe_sum_gate   a MoE layer's weighted expert sum as ggml's unfused chain rounds it (the mul
//                  by the router weights, then one add per expert, in expert order, each
//                  rounded), and the shared expert gated by its sigmoid added: ggml's
//                  op_sigmoid, mul and add, each rounded (shexp_gate_kernel's, fused.cu).
//   moe_tail_fma   moe_sum_gate's tail where ggml-cuda fuses the weighted sum (one fma per
//                  expert), under --tp-moe-ep zero-weight rows not read (#1176).
//                  With out16 (#1176) the values go out rounded to BF16 instead, into the
//                  tensor-parallel send buffer of the reduce-scatter that reads the tail.
//   ids_remap      the routed expert ids through the layer's expert table: ggml's cont and
//                  get_rows of the I32 ids (integers, exact).
//
// The same flags as ggml-cuda and fused.cu (--use_fast_math), so expf is ggml's.

#include <cstdint>
#include <cuda_bf16.h>

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
    int32_t       skip_zero;  // moe_tail_fma: rows of zero router weights not read
    // moe_tail_fma, null or [n_embd, n_tok] contiguous (#1176): out is not written; each value
    // goes here rounded to BF16 (nearest even, as src/cuda/tp_net.cu's tp_pack_bf16 rounds
    // out's values into the send buffer). 8-byte aligned for the 4-column kernel.
    uint16_t *    out16;
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

// moe_tail_fma (#1176): a MoE layer's tail whose weighted sum ggml-cuda fuses, in one launch:
// the first product rounded, then one fma per expert (moe_weighted_reduction_f32, as
// fused.cu's moe_combine_kernel), then the shared expert gated by its sigmoid added
// (moe_sum_gate_kernel's gate). With p.skip_zero (only under --tp-moe-ep) a term whose router
// weight is zero is made from a zero instead of its expert's row, which is not read: there the
// experts another rank holds have zero weights and rows the grouped NVFP4 kernels stored as
// +0.0, so the same values with about half the rows read. Without it every row is read, as
// ggml-cuda reads them (a non-finite row times a zero weight stays NaN). V columns per thread (4: float4 loads, the host checks that
// n_embd is a multiple of 4 and the rows 16-byte aligned).
constexpr int TAIL_GROUP = 8;

template <int V> __device__ __forceinline__ void load_cols(const float * src, float (&v)[V]) {
    if constexpr (V == 4) {
        const float4 q = *(const float4 *) src;
        v[0]           = q.x;
        v[1]           = q.y;
        v[2]           = q.z;
        v[3]           = q.w;
    } else {
#pragma unroll
        for (int r = 0; r < V; ++r) {
            v[r] = src[r];
        }
    }
}

template <int V> __global__ void moe_tail_fma_kernel(const legion_moe_sum_gate p) {
    const int64_t t   = blockIdx.x;
    const int64_t col = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * V;
    if (col >= p.n_embd) {
        return;
    }
    const int64_t first = t * p.k;
    const float * x     = p.x + first * p.n_embd + col;
    float         sum[V] = {};
    for (int j0 = 0; j0 < p.k; j0 += TAIL_GROUP) {
        // A group's rows in flight before their sums (the weights are the token's: uniform).
        float v[TAIL_GROUP][V];
        float wj[TAIL_GROUP];
#pragma unroll
        for (int q = 0; q < TAIL_GROUP; ++q) {
            const int j = j0 + q;
            wj[q]       = j < p.k ? p.w[first + j] : 0.0f;
            if (j < p.k && (!p.skip_zero || wj[q] != 0.0f)) {
                load_cols<V>(x + (int64_t) j * p.n_embd, v[q]);
            } else {
#pragma unroll
                for (int r = 0; r < V; ++r) {
                    v[q][r] = 0.0f;
                }
            }
        }
#pragma unroll
        for (int q = 0; q < TAIL_GROUP; ++q) {
            if (j0 + q < p.k) {
#pragma unroll
                for (int r = 0; r < V; ++r) {
                    sum[r] = j0 + q == 0 ? __fmul_rn(v[q][r], wj[q]) : __fmaf_rn(v[q][r], wj[q], sum[r]);
                }
            }
        }
    }
    // ggml-cuda's op_sigmoid (both built with fast math), then the mul and the add.
    const float gv = p.g[t * p.g_ld];
    const float s  = 1.0f / (1.0f + expf(-gv));
    const int64_t i = t * p.n_embd + col;
    float         sh[V];
    load_cols<V>(p.sh + i, sh);
#pragma unroll
    for (int r = 0; r < V; ++r) {
        sum[r] = __fadd_rn(sum[r], __fmul_rn(sh[r], s));
    }
    if (p.out16) {
        uint16_t b[V];
#pragma unroll
        for (int r = 0; r < V; ++r) {
            b[r] = __bfloat16_as_ushort(__float2bfloat16_rn(sum[r]));
        }
        if constexpr (V == 4) {
            *(uint2 *) (p.out16 + i) = make_uint2((uint32_t) b[0] | (uint32_t) b[1] << 16,
                                                  (uint32_t) b[2] | (uint32_t) b[3] << 16);
        } else {
#pragma unroll
            for (int r = 0; r < V; ++r) {
                p.out16[i + r] = b[r];
            }
        }
    } else if constexpr (V == 4) {
        *(float4 *) (p.out + i) = make_float4(sum[0], sum[1], sum[2], sum[3]);
    } else {
#pragma unroll
        for (int r = 0; r < V; ++r) {
            p.out[i + r] = sum[r];
        }
    }
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
extern "C" const void * const legion_moe_tail_fma_kernel =
    reinterpret_cast<const void *>(&moe_tail_fma_kernel<1>);
extern "C" const void * const legion_moe_tail_fma4_kernel =
    reinterpret_cast<const void *>(&moe_tail_fma_kernel<4>);
