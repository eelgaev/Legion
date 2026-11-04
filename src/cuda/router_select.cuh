// A MoE router's selection from its logits, shared by fused.cu (router_topk, router_weights) and
// router_mv.cu (the router matmul and the selection in one kernel, #836): one body, so both give
// the same bits.

#pragma once

#include <climits>
#include <cmath>
#include <cstdint>

namespace router_select {

constexpr int MAX_PER_LANE = 16;  // n_expert <= 512

// A logit: CG through L2 only (__ldcg), for logits other CTAs of the same grid wrote.
template <bool CG>
__device__ __forceinline__ float logit(const float * l, int e) {
    if constexpr (CG) {
        return __ldcg(l + e);
    } else {
        return l[e];
    }
}

// One warp (all 32 lanes call it): the k largest of the n_expert logits l into ids[0..k), in
// descending order of value (ties: the lower id first). Each lane holds n_expert / 32 logits;
// k rounds of a warp argmax. Lane 0 writes ids. Lane r (r < 32) gets round r's logit in `pick`
// (every lane holds the round's winner after the butterfly), so topk_weights_warp needs no
// second read of the logits.
template <bool CG>
__device__ __forceinline__ void topk_warp_picks(const float * l, int n_expert, int k, int32_t * ids, int lane,
                                                float & pick) {
    const int per = (n_expert + 31) / 32;
    float v[MAX_PER_LANE];
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; i++) {
        const int e = lane + 32 * i;
        v[i] = i < per && e < n_expert ? logit<CG>(l, e) : -INFINITY;
    }
    for (int r = 0; r < k; r++) {
        // this lane's best (lowest id on ties: ids grow with i)
        float bv = -INFINITY;
        int be = INT_MAX;
#pragma unroll
        for (int i = 0; i < MAX_PER_LANE; i++) {
            const int e = lane + 32 * i;
            if (i < per && e < n_expert && (v[i] > bv || (v[i] == bv && e < be))) {
                bv = v[i];
                be = e;
            }
        }
#pragma unroll
        for (int m = 16; m >= 1; m /= 2) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            const int oe = __shfl_xor_sync(0xffffffffu, be, m);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
            }
        }
        if (lane == 0) {
            ids[r] = be;
        }
        if (lane == r) {
            pick = bv;
        }
        // take it out
        if (be % 32 == lane) {
#pragma unroll
            for (int i = 0; i < MAX_PER_LANE; i++) {
                if (lane + 32 * i == be) {
                    v[i] = -INFINITY;
                }
            }
        }
    }
}

template <bool CG>
__device__ __forceinline__ void topk_warp(const float * l, int n_expert, int k, int32_t * ids, int lane) {
    float pick = 0.0f;
    topk_warp_picks<CG>(l, n_expert, k, ids, lane, pick);
}

// One thread: the softmax probabilities of the k experts ids over the logits l, normalized over
// the k (the softmax denominator cancels: w_j = e^(l_j - l_max) / sum over the k of the same).
template <bool CG>
__device__ __forceinline__ void weights_token(const float * l, const int32_t * ids, int k, float * w) {
    const float m = logit<CG>(l, ids[0]);
    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
        sum += expf(logit<CG>(l, ids[j]) - m);
    }
    for (int j = 0; j < k; j++) {
        w[j] = expf(logit<CG>(l, ids[j]) - m) / sum;
    }
}

// One warp (all 32 lanes call it): topk_warp, then weights_token's weights into w[0..k) from
// the picked logits the lanes hold (k <= 32) instead of 2k dependent reads of ids and logits
// on one thread: the same values, the same expf terms summed in the same order, so the same
// bits (#814: the V100's router_weights spent ~5 us a layer on those reads).
template <bool CG>
__device__ __forceinline__ void topk_weights_warp(const float * l, int n_expert, int k, int32_t * ids, float * w,
                                                  int lane) {
    float pick = -INFINITY;
    topk_warp_picks<CG>(l, n_expert, k, ids, lane, pick);
    if (k > 32) {
        __syncwarp();
        if (lane == 0) {
            weights_token<CG>(l, ids, k, w);
        }
        return;
    }
    const float m = __shfl_sync(0xffffffffu, pick, 0);
    const float e = expf(pick - m);
    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
        sum += __shfl_sync(0xffffffffu, e, j);
    }
    if (lane < k) {
        w[lane] = e / sum;
    }
}

}  // namespace router_select
