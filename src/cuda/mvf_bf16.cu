// Small BF16 matmuls of 1-8 tokens that share one input, in one launch (#689; semantics and the
// CPU reference in src/model/core/mvf_bf16.rs): a gated DeltaNet layer's alpha and beta gate
// projections (48 rows each in Qwen3.8-27B and Flash-Next), which ggml-cuda runs as two
// mul_mat_vec_f launches of one CTA per row. Each row is computed as ggml-cuda's
// mul_mat_vec_f<nv_bfloat16, float, ncols_dst> computes it (mmvf.cu, MIT: one CTA per row,
// ggml's block size for the row length, bf16 pairs widened to float, one FMA per value, the
// warp sums and then the CTA's warp sums with ggml's own warp_reduce_sum), so every output has
// its bits. What differs: one launch for all the weights' rows, and each thread loads all its
// weight words before its first FMA (ggml's loop waits for each load in turn: on a GB10 a
// 48-row launch at 8 tokens takes ~12 us for ~0.5 MB). A plain launch: the kernel ahead of it
// (the z projection's NVFP4 matvec) lets its dependents start early, and resident CTAs waiting
// for it would take its SMs' slots.

#include "device_cache.cuh"

#include <cstdint>

extern "C" {

constexpr int LEGION_MVF_MAX_W = 4;

struct legion_mvf_bf16 {
    const uint16_t * w[LEGION_MVF_MAX_W];  // [k, rows[i]]: row r at w[i] + r * w_ld[i] (BF16 values)
    int64_t w_ld[LEGION_MVF_MAX_W];
    int32_t rows[LEGION_MVF_MAX_W];
    int32_t n_w;
    const float * x;  // [k, n_tok]: token t at x + t * x_ld
    int64_t x_ld;
    float * dst;      // [sum rows, n_tok]: the weights' rows in order, token t at dst + t * dst_ld
    int64_t dst_ld;
    int32_t k, n_tok;
};

int legion_mvf_bf16_launch(const legion_mvf_bf16 * p, void * stream);

// Whether the current device runs the op (compute capability 9.0+).
int legion_mvf_bf16_supported(void);

}  // extern "C"

namespace {

// Weight words (bf16 pairs) per thread loaded before the first FMA: rows of up to
// 2 * PRE * 256 = 8192 values at once, longer ones in a loop after them.
constexpr int MVF_PRE = 16;
constexpr int MVF_MAX_TOK = 8;

// ggml's nv_bfloat162 pair as two floats (element 0 in the low half).
__device__ __forceinline__ float2 mvf_widen(const uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

// One CTA per output row (all weights' rows in order); blockDim.x is ggml's block size for k.
template <int NT>
__global__ void __launch_bounds__(256) mvf_bf16_kernel(const legion_mvf_bf16 p) {
    int row = blockIdx.x;
    int wi = 0;
    while (wi + 1 < p.n_w && row >= p.rows[wi]) {
        row -= p.rows[wi];
        wi++;
    }
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int ncols2 = p.k / 2;
    const uint32_t * xb = (const uint32_t *) (p.w[wi] + (int64_t) row * p.w_ld[wi]);

    uint32_t pre[MVF_PRE];
#pragma unroll
    for (int i = 0; i < MVF_PRE; i++) {
        const int col2 = tid + i * bs;
        pre[i] = col2 < ncols2 ? __ldg(xb + col2) : 0u;
    }

    const float2 * y2 = (const float2 *) p.x;
    const int64_t y_ld2 = p.x_ld / 2;
    __shared__ float buf_iw[32];
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
    for (int i = 0; i < MVF_PRE; i++) {
        const int col2 = tid + i * bs;
        if (col2 < ncols2) {
            const float2 tmpx = mvf_widen(pre[i]);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const float2 tmpy = y2[j * y_ld2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    }
    for (int col2 = tid + MVF_PRE * bs; col2 < ncols2; col2 += bs) {
        const float2 tmpx = mvf_widen(__ldg(xb + col2));
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
        p.dst[(int64_t) tid * p.dst_ld + blockIdx.x] = sumf[tid];
    }
}

template <int NT>
void mvf_go(const legion_mvf_bf16 * p, int total, int bs, cudaStream_t s) {
    mvf_bf16_kernel<NT><<<total, bs, 0, s>>>(*p);
}

}  // namespace

int legion_mvf_bf16_supported(void) {
    return legion_device::compute_capability() >= 90 ? 1 : 0;
}

int legion_mvf_bf16_launch(const legion_mvf_bf16 * p, void * stream) {
    if (p->n_w < 1 || p->n_w > LEGION_MVF_MAX_W || p->n_tok < 1 || p->n_tok > MVF_MAX_TOK || p->k < 2 ||
        p->k % 2 != 0 || p->x_ld % 2 != 0 || p->dst_ld < 1) {
        return -1;
    }
    int64_t total = 0;
    for (int i = 0; i < p->n_w; i++) {
        if (p->rows[i] < 1 || p->w_ld[i] % 2 != 0 || ((uintptr_t) p->w[i]) % 4 != 0) {
            return -1;
        }
        total += p->rows[i];
    }
    if (total > p->dst_ld || total > 0x7fffffff) {
        return -1;
    }
    // ggml_cuda_mul_mat_vec_f's block size (warp size 32, at most 256): the fewest iterations
    // per thread, the smallest block on ties.
    int64_t bs = 32;
    int64_t niter_best = (p->k + 2 * 32 - 1) / (2 * 32);
    for (int64_t b = 64; b <= 256; b += 32) {
        const int64_t niter = (p->k + 2 * b - 1) / (2 * b);
        if (niter < niter_best) {
            niter_best = niter;
            bs = b;
        }
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const int t = (int) total;
    switch (p->n_tok) {
        case 1: mvf_go<1>(p, t, (int) bs, s); break;
        case 2: mvf_go<2>(p, t, (int) bs, s); break;
        case 3: mvf_go<3>(p, t, (int) bs, s); break;
        case 4: mvf_go<4>(p, t, (int) bs, s); break;
        case 5: mvf_go<5>(p, t, (int) bs, s); break;
        case 6: mvf_go<6>(p, t, (int) bs, s); break;
        case 7: mvf_go<7>(p, t, (int) bs, s); break;
        default: mvf_go<8>(p, t, (int) bs, s); break;
    }
    return (int) cudaGetLastError();
}
