// The routing plan of nvfp4_once.cu's warp kernels (`mvw`, `mvh`, #836), shared with fused.cu's
// router top-k: the step's distinct experts, most pairs first and then in the order of their
// first pair (e), each one's pairs in pair order (q[off[d]..] of c[d]), and the work-item
// counters the warps take items from (next[0] for the one-matmul op, next[1] for the SwiGLU op).
// nvfp4_once.cu keeps one plan per device; `quantize` builds it when no router did
// (src/model/core/nvfp4_once.rs), else the router top-k kernel of the same step writes it right
// after the ids, so the op's launches find it ready.

#pragma once

#include <cstdint>

namespace nvfp4_plan {

// src/model/core/nvfp4_once.rs's MAX_PAIRS: (token, slot) pairs of a step.
constexpr int MAX_PAIRS = 128;

struct plan_t {
    int          n;
    unsigned int next[2];
    int          e[MAX_PAIRS];
    int          off[MAX_PAIRS];
    int          c[MAX_PAIRS];
    int          q[MAX_PAIRS];
};

// The plan of the ids (I32 [n_tok][n_used], ids_nb1 ints between tokens) into `plan`, both
// counters at 0, by one CTA of at least MAX_PAIRS threads (all of them call it), a thread per
// pair. Ids out of 0..n_expert take no work. The caller makes sure no kernel still reads the plan.
__device__ __forceinline__ void build(plan_t & plan, const int32_t * ids, const int64_t ids_nb1, const int n_used,
                                      const int n_tok, const int n_expert) {
    __shared__ int s_e[MAX_PAIRS];     // each pair's expert (-1: no pair or out of range)
    __shared__ int s_lead[MAX_PAIRS];  // 1 if the pair is its expert's first
    __shared__ int s_cnt[MAX_PAIRS];   // a leading pair's expert's pairs
    __shared__ int s_off[MAX_PAIRS];   // a leading pair's expert's first slot in q
    const int tid   = threadIdx.x;
    const int pairs = n_used * n_tok;
    int e = -1;
    if (tid < MAX_PAIRS) {
        if (tid < pairs) {
            e = ids[(tid / n_used) * ids_nb1 + tid % n_used];
            if (e < 0 || e >= n_expert) {
                e = -1;
            }
        }
        s_e[tid] = e;
    }
    __syncthreads();
    int first = -1;
    int rank  = 0;
    int cnt   = 0;
    if (e >= 0) {
        for (int j = 0; j < pairs; ++j) {
            if (s_e[j] == e) {
                first = first < 0 ? j : first;
                rank += j < tid;
                ++cnt;
            }
        }
    }
    const bool lead = e >= 0 && first == tid;
    if (tid < MAX_PAIRS) {
        s_lead[tid] = lead;
        s_cnt[tid]  = lead ? cnt : 0;
    }
    const int n_dist = __syncthreads_count(lead);
    if (lead) {
        // Expert index d, heaviest first: the leading pairs of experts with more pairs, or as
        // many and before this one (so the warps take the shared experts' items first and the
        // singletons' short items last, as the tail); its first slot: their pairs.
        int d   = 0;
        int off = 0;
        for (int j = 0; j < pairs; ++j) {
            const bool ahead = s_lead[j] && (s_cnt[j] > cnt || (s_cnt[j] == cnt && j < tid));
            d   += ahead;
            off += ahead ? s_cnt[j] : 0;
        }
        s_off[tid]  = off;
        plan.e[d]   = e;
        plan.c[d]   = cnt;
        plan.off[d] = off;
    }
    if (tid == 0) {
        plan.n       = n_dist;
        plan.next[0] = 0;
        plan.next[1] = 0;
    }
    __syncthreads();
    if (e >= 0) {
        plan.q[s_off[first] + rank] = tid;
    }
}

}  // namespace nvfp4_plan
