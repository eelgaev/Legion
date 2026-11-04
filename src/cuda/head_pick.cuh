// The MTP draft pick's partials, shared by the draft LM head kernels that write them
// (q8_head.cu on a V100, #814; q4k_head.cu on a GB10, #836) and the pick's argmax and
// probability kernels (q8_head.cu), which read them. The host side is
// src/model/core/q8_head/pick.rs (the same offsets).
//
// The buffer: a header (partials written, the row's largest value, the pmax CTAs' ticket), the
// pmax CTAs' sums, then one uint4 partial per head CTA (best value bits, its tie rank, largest
// value bits). A head of several columns (q4k_head.cu, 2 or more drafting sequences) gives
// column c the slice c of that layout, the host's stride bytes after slice c - 1; a one-column
// head writes slice 0 only.

#pragma once

#include "ggml-cuda/common.cuh"

#include <cfloat>
#include <cstdint>

namespace {

constexpr int PICK_SUMS  = 16;    // byte offset of the 256 sums
constexpr int PICK_PARTS = 2048;  // byte offset of the partials
constexpr int PMAX_T     = 256;   // ggml's soft_max_f32_parallelize_cols block

struct pick_header {
    uint32_t count;
    float    max;
    uint32_t ticket;
    uint32_t pad;
};

// argmax_f32's tie order (src/cuda/sm90/cluster_select.cu's am_rank / am_col / am_better):
// among equal maxima the lowest (bitrev5(warp), bitrev5(lane), column / 1024) of its 1,024
// threads' scan wins; NONE: no candidate.
constexpr uint32_t PK_NONE = 0xffffffffu;

__device__ __forceinline__ uint32_t pk_rank(int64_t col) {
    const uint32_t t = (uint32_t) col & 1023u;
    const uint32_t warp = __brev(t >> 5) >> 27, lane = __brev(t & 31u) >> 27;
    return (warp << 26) | (lane << 21) | ((uint32_t) col >> 10);
}

__device__ __forceinline__ int pk_col(uint32_t r) {
    const uint32_t warp = __brev(r >> 26) >> 27, lane = __brev((r >> 21) & 31u) >> 27;
    return (int) (((r & 0x1fffffu) << 10) | (warp << 5) | lane);
}

__device__ __forceinline__ bool pk_better(float v, uint32_t r, float bv, uint32_t br) {
    return v > bv || (v == bv && r < br);
}

}  // namespace
