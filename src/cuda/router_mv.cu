// A MoE layer's router in one kernel for steps of 1-3 tokens (#836; semantics and the CPU
// reference in src/model/core/router_mv.rs). The logits `w @ x` are computed as ggml-cuda's
// mul_mat_vec_f<float, float> computes them for up to 3 tokens on tensor-core GPUs (mmvf.cu, MIT:
// one CTA per row, ggml's block size, float2 loads, one FMA per value, warp sums and then the
// CTA's warp sums with ggml's own warp_reduce_sum), so they have its bits. The CTA that finishes
// last then runs router_topk over them and router_weights' arithmetic on the picked logits
// (router_select.cuh, the same body as fused.cu's router_select). One launch per MoE layer
// instead of three. On the GB10 the kernel is a
// programmatic dependent launch: each CTA loads the first chunks of its weight row before it
// waits for the kernel ahead (which writes x).

#include "device_cache.cuh"
#include "launch_pdl.cuh"
#include "router_select.cuh"

#include <cstdint>

extern "C" {

struct legion_router_mv {
    const float * w;  // [k_embd, n_expert]: row e at w + e * w_ld (values; BF16 ones with wb)
    int64_t w_ld;
    const float * x;  // [k_embd, n_tok]: token t at x + t * x_ld
    int64_t x_ld;
    float * logits;   // out [n_expert, n_tok] contiguous
    int32_t * ids;    // out [k, n_tok] contiguous
    float * wts;      // out [k, n_tok] contiguous
    int32_t k_embd, n_expert, k, n_tok;
    int32_t slot;     // the completion counter this launch counts on (one per layer)
    // Nonzero: w holds the F32 weights' values as BF16 (all exact there), each widened back to
    // its F32 bits as it is loaded: the same products and sums from half the bytes (#836).
    int32_t wb;
};

int legion_router_mv_launch(const legion_router_mv * p, void * stream);

// Whether the current device runs the op (compute capability 9.0+).
int legion_router_mv_supported(void);

}  // extern "C"

namespace {

// Completion counters, one per slot and device: each launch's CTAs count up on theirs, and the
// last one sets it back to 0 for the next launch. Launches on the same slot must not overlap
// (the layers of one model on one device run in stream order).
constexpr int SLOTS = 1024;
__device__ unsigned int router_mv_done[SLOTS];

// Weight chunks (float2 per thread) loaded before the wait for the kernel ahead.
constexpr int PRE = 8;

constexpr int MAX_TOK = 3;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// One CTA per expert row; blockDim.x is ggml's block size for k_embd (32-256, see the launcher).
// WB: the row's values are BF16 (p.wb), a pair per 32-bit word, widened to the F32 pair.
template <int NT, bool WB>
__global__ void __launch_bounds__(256) router_mv_kernel(legion_router_mv p) {
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int bs    = blockDim.x;
    const int ncols2 = p.k_embd / 2;
    const float2 * xf = (const float2 *) (p.w + (int64_t) row * p.w_ld);
    const uint32_t * xb = (const uint32_t *) ((const uint16_t *) p.w + (int64_t) row * p.w_ld);
    // The row's values col2 * 2 and col2 * 2 + 1.
    auto ldw = [&](int col2) -> float2 {
        if constexpr (WB) {
            const uint32_t v = __ldg(xb + col2);
            return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
        } else {
            return __ldg(xf + col2);
        }
    };
    const float2 * y2 = (const float2 *) p.x;
    const int64_t y_ld2 = p.x_ld / 2;

    __shared__ float buf_iw[32];
    __shared__ bool last;

    // The weights do not depend on the kernel ahead: their first chunks in flight before it ends.
    float2 pre[PRE];
#pragma unroll
    for (int i = 0; i < PRE; i++) {
        const int col2 = tid + i * bs;
        pre[i] = col2 < ncols2 ? ldw(col2) : make_float2(0.0f, 0.0f);
    }
    pdl_wait();

    if (bs > 32) {
        if (tid < 32) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    // mul_mat_vec_f's loop over col2 = tid, tid + bs, ... and its sums (ggml_cuda_mad: one FMA
    // per value).
    float sumf[NT] = {0.0f};
#pragma unroll
    for (int i = 0; i < PRE; i++) {
        const int col2 = tid + i * bs;
        if (col2 < ncols2) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const float2 tmpy = y2[j * y_ld2 + col2];
                ggml_cuda_mad(sumf[j], pre[i].x, tmpy.x);
                ggml_cuda_mad(sumf[j], pre[i].y, tmpy.y);
            }
        }
    }
    for (int col2 = tid + PRE * bs; col2 < ncols2; col2 += bs) {
        const float2 tmpx = ldw(col2);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float2 tmpy = y2[j * y_ld2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < NT; ++j) {
        sumf[j] = warp_reduce_sum<32>(sumf[j]);
        if (bs > 32) {
            buf_iw[tid / 32] = sumf[j];
            __syncthreads();
            if (tid < 32) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<32>(sumf[j]);
            }
            __syncthreads();
        }
    }
    if (tid < NT) {
        p.logits[(int64_t) tid * p.n_expert + row] = sumf[tid];
    }

    // The last CTA to get here selects: all rows' logits are written and visible by then.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        last = atomicAdd(&router_mv_done[p.slot], 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    if (tid == 0) {
        router_mv_done[p.slot] = 0;
    }
    __threadfence();
    const int lane = tid % 32;
    for (int t = tid / 32; t < p.n_tok; t += bs / 32) {
        const float * l = p.logits + (int64_t) t * p.n_expert;
        int32_t * ids = p.ids + (int64_t) t * p.k;
        router_select::topk_weights_warp<true>(l, p.n_expert, p.k, ids, p.wts + (int64_t) t * p.k, lane);
    }
}

// The kernel for the launch's token count.
template <bool WB>
int launch_tokens(const legion_router_mv * p, dim3 grid, int bs, cudaStream_t s) {
    switch (p->n_tok) {
        case 1:
            launch_pdl(router_mv_kernel<1, WB>, grid, bs, 0, s, *p);
            break;
        case 2:
            launch_pdl(router_mv_kernel<2, WB>, grid, bs, 0, s, *p);
            break;
        default:
            launch_pdl(router_mv_kernel<3, WB>, grid, bs, 0, s, *p);
            break;
    }
    return (int) cudaGetLastError();
}

}  // namespace

int legion_router_mv_supported(void) {
    // Not on a V100 (#814): there one launch of this kernel takes longer (~36 us at 3 tokens)
    // than ggml's matmul and the two selection launches together (~25 us).
    return legion_device::compute_capability() >= 90 ? 1 : 0;
}

int legion_router_mv_launch(const legion_router_mv * p, void * stream) {
    if (p->n_tok < 1 || p->n_tok > MAX_TOK || p->k_embd < 2 || p->k_embd % 2 != 0 || p->w_ld % 2 != 0 ||
        p->x_ld % 2 != 0 || p->n_expert < 1 || p->n_expert > 32 * router_select::MAX_PER_LANE || p->k < 1 ||
        p->k > p->n_expert || p->slot < 0 || p->slot >= SLOTS) {
        return -1;
    }
    // ggml_cuda_mul_mat_vec_f's block size (warp size 32, at most 256): the fewest iterations
    // per thread, the smallest block on ties.
    int64_t bs = 32;
    int64_t niter_best = (p->k_embd + 2 * 32 - 1) / (2 * 32);
    for (int64_t b = 64; b <= 256; b += 32) {
        const int64_t niter = (p->k_embd + 2 * b - 1) / (2 * b);
        if (niter < niter_best) {
            niter_best = niter;
            bs = b;
        }
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const dim3 grid(p->n_expert);
    if (p->wb != 0) {
        return launch_tokens<true>(p, grid, (int) bs, s);
    }
    return launch_tokens<false>(p, grid, (int) bs, s);
}
