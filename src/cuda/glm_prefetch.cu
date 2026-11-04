// GLM-5.3-Flash's parallel decode, `--glm-prefetch` (#747): layer L + 1's likely uncached
// experts copied into VRAM while layer L finishes. Kernels only; the Rust side
// (src/model/glm5next/par/prefetch.rs, src/model/core/expert_mv/ahead.rs) launches them with
// cudaLaunchKernel and owns the memory (a staging ring per GPU, laid out by
// src/residency/ep_prefetch.rs).
//
// - `route_kernel` (side stream): the next layer's router applied to this layer's FFN input,
//   sigmoid plus bias, top n_used per token, as router_ids_kernel in glm5next.cu selects. A warp
//   per expert computes its logit for every token; the last CTA to finish picks the top experts
//   and writes them rank-major (all tokens' first choices, then their second ones, ...), raw and
//   as this GPU's tiered ids (its own experts only, -1 for the others). A guess only: its bits
//   never reach the step's results.
// - `match_kernel` (compute stream, after the copies): the staging slot of each of the real
//   step's pairs, -1 where the guess did not stage its expert.
// - `cached_kernel` (compute stream, `--glm-pair-balance`): the owners' cached experts plus the
//   ones the guess stages on their owner, from inputs every GPU of the pair has alike.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int ROUTE_WARPS  = 8;   // experts per CTA
constexpr int ROUTE_TOKENS = 16;  // most tokens of a predicted step
constexpr int ROUTE_RT     = 16;  // experts per lane in the selection: up to 512

__device__ __forceinline__ float sigm(float x) {
    return 1.0f / (1.0f + expf(-x));
}

struct Route {
    const float *   x;       // [n][k] F32
    const float *   w;       // [n_expert][k] F32 (the router's ne0 = k)
    const float *   bias;    // [n_expert]
    const int32_t * table;   // the next layer's lookup table on this GPU, [n_expert]
    float *         logits;  // scratch [n][n_expert]
    int32_t *       raw;     // out: [n_used][n] expert ids
    int32_t *       tiered;  // out: [n_used][n] tiered ids of the owned experts, -1 for the others
    uint32_t *      done;    // CTAs finished; the last one resets it
    int32_t         k, n_expert, n_used, n;
    int32_t         lo, hi;  // this GPU's experts
};

__global__ void __launch_bounds__(ROUTE_WARPS * 32) route_kernel(const Route r) {
    const int lane = threadIdx.x % 32;
    const int e    = blockIdx.x * ROUTE_WARPS + threadIdx.x / 32;
    if (e < r.n_expert) {
        float acc[ROUTE_TOKENS];
#pragma unroll
        for (int t = 0; t < ROUTE_TOKENS; t++) {
            acc[t] = 0.0f;
        }
        const float4 * w = (const float4 *) (r.w + (int64_t) e * r.k);
        for (int i = lane; i < r.k / 4; i += 32) {
            const float4 a = __ldg(w + i);
#pragma unroll
            for (int t = 0; t < ROUTE_TOKENS; t++) {
                if (t < r.n) {
                    const float4 b = ((const float4 *) (r.x + (int64_t) t * r.k))[i];
                    acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < ROUTE_TOKENS; t++) {
            float v = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, o);
            }
            if (lane == 0 && t < r.n) {
                r.logits[t * r.n_expert + e] = v;
            }
        }
    }
    // The last CTA selects (threadfence reduction, as CUDA's sample of it).
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(r.done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    for (int t = threadIdx.x / 32; t < r.n; t += ROUTE_WARPS) {
        const float * lg = r.logits + t * r.n_expert;
        float         sc[ROUTE_RT];
        uint32_t      taken = 0;
#pragma unroll
        for (int q = 0; q < ROUTE_RT; q++) {
            const int i = lane + 32 * q;
            sc[q]       = i < r.n_expert ? sigm(__ldcg(lg + i)) + r.bias[i] : -INFINITY;
            if (i >= r.n_expert) {
                taken |= 1u << q;
            }
        }
        for (int j = 0; j < r.n_used; j++) {
            float bv = -INFINITY;
            int   bi = INT32_MAX;
#pragma unroll
            for (int q = 0; q < ROUTE_RT; q++) {
                if (!(taken >> q & 1) && (sc[q] > bv || bi == INT32_MAX)) {
                    bv = sc[q];
                    bi = lane + 32 * q;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int   oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (oi != INT32_MAX && (bi == INT32_MAX || ov > bv || (ov == bv && oi < bi))) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                const bool ok           = bi >= 0 && bi < r.n_expert;
                r.raw[j * r.n + t]      = ok ? bi : -1;
                r.tiered[j * r.n + t]   = ok && bi >= r.lo && bi < r.hi ? r.table[bi] : -1;
            }
            if (bi != INT32_MAX && bi % 32 == lane) {
                taken |= 1u << (bi / 32);
            }
        }
    }
    if (threadIdx.x == 0) {
        *r.done = 0;
    }
}

struct Match {
    const int32_t * ids;     // the step's tiered ids, [n_tok][n_used], ids_nb1 bytes per token
    int64_t         ids_nb1;
    int32_t         n_used, n_tok;
    const int32_t * guess;   // the staged guess: tiered ids, n_guess
    const int32_t * slot;    // stage_copy's slot per guess pair (-1: none)
    int32_t         n_guess;
    int32_t         vram_below;
    int32_t *       out;     // [n_tok * n_used]
};

// One CTA.
__global__ void match_kernel(const Match m) {
    const int n_pairs = m.n_used * m.n_tok;
    for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
        const int e = *(const int32_t *) ((const char *) m.ids + (q / m.n_used) * m.ids_nb1 + (q % m.n_used) * 4);
        int       s = -1;
        if (e >= m.vram_below) {
            for (int g = 0; g < m.n_guess; g++) {
                if (m.guess[g] == e && m.slot[g] >= 0) {
                    s = m.slot[g];
                    break;
                }
            }
        }
        m.out[q] = s;
    }
}

struct Cached {
    const int32_t * cached;  // [n_expert]: the owners' cached experts (1) or not (0)
    const int32_t * raw;     // the guess, n_guess expert ids, rank-major
    int32_t         n_guess, n_expert;
    int32_t         lo, mid, hi;  // the pair's experts, the second GPU's first
    int32_t         cap;          // experts a GPU stages per layer
    int32_t *       out;          // [n_expert]
};

// One CTA. Ranks as stage_copy does: a GPU's uncached guessed experts in the order of their
// first guess pairs take its slots while they last.
__global__ void cached_kernel(const Cached c) {
    for (int e = threadIdx.x; e < c.n_expert; e += blockDim.x) {
        c.out[e] = c.cached[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    int used[2] = { 0, 0 };
    for (int g = 0; g < c.n_guess; g++) {
        const int e = c.raw[g];
        if (e < c.lo || e >= c.hi || c.out[e] != 0) {
            continue;
        }
        const int side = e < c.mid ? 0 : 1;
        if (used[side] < c.cap) {
            used[side]++;
            c.out[e] = 1;
        }
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them, for Rust's cudaLaunchKernel (see gate_q8.cu).
extern "C" const void * const legion_glm_prefetch_route_kernel  = reinterpret_cast<const void *>(&route_kernel);
extern "C" const void * const legion_glm_prefetch_match_kernel  = reinterpret_cast<const void *>(&match_kernel);
extern "C" const void * const legion_glm_prefetch_cached_kernel = reinterpret_cast<const void *>(&cached_kernel);
