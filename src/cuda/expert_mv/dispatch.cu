// expert_mv's extern "C" entry points and stage_copy (see common.cuh); the kernels of each weight
// type are compiled in the type's own .cu.
#include "common.cuh"

namespace {

constexpr int STAGE_THREADS = 256;
constexpr int STAGE_UNROLL  = 8;    // 16-byte loads in flight per thread
constexpr int STAGE_PAIRS   = 512;  // most pairs a step has here (64 tokens x 8, 8 x 64)

// Copies a step's host experts into the staging buffer on few CTAs (`ctas`): each thread keeps
// STAGE_UNROLL loads in flight, enough for a few CTAs to fill the link while the VRAM experts
// keep the rest of the SMs (as Strata-AC922's fetch_blobs does, 756ff34, MIT). Every CTA works
// out the slots alike: a host expert's first pair takes the next slot while slots last.
__global__ void __launch_bounds__(STAGE_THREADS) stage_copy(const legion_expert_stage s) {
    __shared__ int slot[STAGE_PAIRS];
    const int n_pairs = s.n_used * s.n_tok;
    auto id = [&](int q) {
        return *(const int32_t *) ((const char *) s.ids + (q / s.n_used) * s.ids_nb1 + (q % s.n_used) * 4);
    };
    for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
        const int e = id(q);
        bool first  = e >= s.vram_below && e < s.n_w;
        for (int r = 0; r < q && first; r++) {
            first = id(r) != e;
        }
        slot[q] = first;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int rank = 0;
        for (int q = 0; q < n_pairs; q++) {
            int sl = -1;
            if (slot[q]) {
                const int r = rank++;
                if (r % s.t_every < s.t_take) {
                    sl = r / s.t_every * s.t_take + r % s.t_every;
                }
            }
            slot[q] = sl < s.cap ? sl : -1;
        }
    }
    __syncthreads();
    if (s.table && blockIdx.x == 0) {
        for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
            ((int32_t *) s.stage)[q] = slot[q];
        }
    }
    const int64_t step = (int64_t) gridDim.x * STAGE_THREADS * STAGE_UNROLL;
    for (int q = 0; q < n_pairs; q++) {
        if (slot[q] < 0 || (slot[q] % s.c_every < s.c_take) == (s.c_flip != 0)) {
            continue;
        }
        const int64_t e = id(q);
        for (int m = 0; m < s.n_seg; m++) {
            const uint4 * src = (const uint4 *) (s.src[m] + e * s.nb2[m]);
            uint4 * dst       = (uint4 *) (s.stage + s.hdr + slot[q] * s.slot + s.off[m]);
            const int64_t n   = s.nb2[m] / 16;
            for (int64_t base = (int64_t) blockIdx.x * STAGE_THREADS * STAGE_UNROLL + threadIdx.x; base < n;
                 base += step) {
                uint4 v[STAGE_UNROLL];
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < n) {
                        v[i] = __ldg(src + c);
                    }
                }
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < n) {
                        dst[c] = v[i];
                    }
                }
            }
        }
    }
}

// quantize_x (`steps & 1`), then the type's kernels (`steps & 2`, launch_type in the type's .cu).
template <ggml_type type>
int launch_steps(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    if (p.steps & 1) {
        const dim3 grid((p.k + 127) / 128, p.n_x * p.n_tok);
        if (xq_split_for<type>(p)) {
            quantize_x<true><<<grid, 128, 0, stream>>>(p);
        } else {
            quantize_x<false><<<grid, 128, 0, stream>>>(p);
        }
    }
    if (!(p.steps & 2)) {
        return cudaGetLastError();
    }
    return legion_emv::launch_type<type>(p, smem, stream);
}

}  // namespace

extern "C" int legion_expert_mv_launch(int type, const legion_expert_mv * p, int smem, void * stream) {
    auto s = (cudaStream_t) stream;
    switch (type) {
        case GGML_TYPE_Q4_K: return launch_steps<GGML_TYPE_Q4_K>(*p, smem, s);
        case GGML_TYPE_Q5_K: return launch_steps<GGML_TYPE_Q5_K>(*p, smem, s);
        case GGML_TYPE_Q5_1: return launch_steps<GGML_TYPE_Q5_1>(*p, smem, s);
        case GGML_TYPE_Q6_K: return launch_steps<GGML_TYPE_Q6_K>(*p, smem, s);
        case GGML_TYPE_Q8_0: return launch_steps<GGML_TYPE_Q8_0>(*p, smem, s);
        case GGML_TYPE_Q2_K: return launch_steps<GGML_TYPE_Q2_K>(*p, smem, s);
        case GGML_TYPE_Q3_K: return launch_steps<GGML_TYPE_Q3_K>(*p, smem, s);
        case GGML_TYPE_IQ2_XXS: return launch_steps<GGML_TYPE_IQ2_XXS>(*p, smem, s);
        case GGML_TYPE_IQ2_S: return launch_steps<GGML_TYPE_IQ2_S>(*p, smem, s);
        case GGML_TYPE_IQ3_XXS: return launch_steps<GGML_TYPE_IQ3_XXS>(*p, smem, s);
        case GGML_TYPE_IQ3_S: return launch_steps<GGML_TYPE_IQ3_S>(*p, smem, s);
        case GGML_TYPE_IQ4_XS: return launch_steps<GGML_TYPE_IQ4_XS>(*p, smem, s);
        default: return cudaErrorInvalidValue;
    }
}

extern "C" int legion_expert_stage_launch(const legion_expert_stage * s, void * stream) {
    if (s->n_used * s->n_tok > STAGE_PAIRS || s->n_seg < 0 || s->n_seg > 3 || s->ctas < 1 || s->t_every < 1 ||
        s->t_take < 1 || s->t_take > s->t_every || s->c_every < 1) {
        return cudaErrorInvalidValue;
    }
    stage_copy<<<s->ctas, STAGE_THREADS, 0, (cudaStream_t) stream>>>(*s);
    return cudaGetLastError();
}

namespace {

// LEGION_DEBUG_EXPERT_DUMP (docs/debug-env.md): the first 64 bytes of each pair's expert as the
// kernels read it (`w + id * nb2`; zeros for an id outside the stack), a CTA per pair.
__global__ void debug_peek(const uint8_t * w, int64_t nb2, const int32_t * ids, int64_t ids_nb1, int n_used, int n_w,
                           uint8_t * out) {
    const int pair = blockIdx.x;
    const int id   = *(const int32_t *) ((const char *) ids + (pair / n_used) * ids_nb1 + (pair % n_used) * 4);
    const bool in  = id >= 0 && id < n_w;
    out[(int64_t) pair * 64 + threadIdx.x] = in ? w[(int64_t) id * nb2 + threadIdx.x] : 0;
}

// ... and the kernels' SwiGLU of raw gate and up products (the same inline function).
__global__ void debug_swiglu(const float * g, const float * u, float * out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = swiglu(g[i], u[i], limit);
    }
}

}  // namespace

extern "C" int legion_expert_mv_debug_peek(const void * w, int64_t nb2, const void * ids, int64_t ids_nb1,
                                           int n_used, int n_tok, int n_w, void * out, void * stream) {
    debug_peek<<<n_used * n_tok, 64, 0, (cudaStream_t) stream>>>((const uint8_t *) w, nb2, (const int32_t *) ids,
                                                                  ids_nb1, n_used, n_w, (uint8_t *) out);
    return cudaGetLastError();
}

extern "C" int legion_expert_mv_debug_swiglu(const float * g, const float * u, float * out, int n, float limit,
                                             void * stream) {
    debug_swiglu<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(g, u, out, n, limit);
    return cudaGetLastError();
}
