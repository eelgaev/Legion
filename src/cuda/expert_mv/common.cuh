// Decode MoE expert matvec kernels: mul_mat_id for a few tokens (decode and MTP verify steps).
// Shapes and the tile size are chosen on the Rust side (src/model/qwen35/expert_mv.rs).
//
// Files (#955: one nvcc process per weight-type family, so they build in parallel):
// - common.cuh: the parameters, constants and device helpers (dot products, Unit<type>, grids),
//   quantize_x, and the per-type entry points' declarations (launch_type).
// - kernels.cuh: the kernel templates and the per-type launch code (launch_any).
// - q4_k.cu, q5_k.cu, q5_1.cu, q6_k.cu, q8_0.cu, k_low.cu (Q2_K, Q3_K), iq2.cu (IQ2_XXS, IQ2_S),
//   iq3.cu (IQ3_XXS, IQ3_S, IQ4_XS):
//   launch_type<type> for their types, the kernels compiled there.
// - dispatch.cu: the extern "C" entry points (quantize_x's launch, the switch over the type)
//   and stage_copy.
//
//   dst[t][j][r] = sum_e W_{ids[t][j]}[r][e] * X[t][j or 0][e]
//
// or, with a second matrix (`w2`, the up projection of the same experts), SwiGLU:
//
//   dst[t][j][r] = silu(W.x) * (W2.x)
//
// or with `limit` > 0 the clamped SwiGLU of GLM-5 (`swiglu_limit`), as moe_gemm.cu clamps it:
//
//   dst[t][j][r] = silu(min(W.x, limit)) * clamp(W2.x, -limit, limit)
//
// Expert `id` starts at `w + id * w_nb2` (mul_mat_id's addressing). Ids below `vram_below` are
// VRAM (a tiered tensor's cache slots, src/residency); the others are host memory the GPU reads
// in place over NVLink2 / ATS. Rows are `w_nb1` bytes apart. The dot products are ggml's own
// (vecdotq.cuh, MIT) against X quantized to Q8_1 the way ggml's quantize_q8_1 does
// (ggml-cuda/quantize.cu, MIT), so results match ggml's mmvq up to summation order.
//
// Three kernels:
// - `quantize_x` writes each activation column once as Q8_1 to `xq` (a scratch tensor of the
//   graph; in steps of 1-2 tokens the IQ types' quants and scales in separate arrays, xq_split).
// - `expert_mv_tile` (host experts): a CTA owns `rows_per_cta` consecutive rows of one expert
//   (grid.x: row tiles, grid.y: (token, slot) pairs). It first issues every load of the tile at
//   once, which is what keeps a high-latency link busy (ggml's mmvq issues one load per lane and
//   waits), parks the tile in shared memory, then computes the dot products for every token that
//   chose its expert. Of the pairs that chose the same expert only the first one's CTAs run, so
//   a verify step reads each host expert over the link once.
// - `expert_mv_vram` (VRAM experts, `vram_kernel`): reads rows in place, two rows per warp, as
//   ggml's mmvq does (HBM latency is low; occupancy and ILP matter there), without the dedup:
//   repeated experts come from L2.
// - With `vram_pf` (`--expert-mv-prefetch`), `expert_mv_tile` computes a pair whose expert is in
//   VRAM (a cache slot, or a staging slot) straight from global memory instead of through the
//   shared-memory tile: the same lanes, units and shuffles as the tile path (bitwise its
//   outputs), but each lane's unit loop has a compile-time trip count (`NI`), so all of a row's
//   weight loads are issued before the first dot product waits on one. Adapted from
//   Strata-AC922's S28 prefetched decode expert kernels (eelgaev/Strata-AC922 5e1aac0, MIT): its
//   kernels load a lane's whole row first, then apply the dot products in the old order. It
//   stays in the tile kernel's launch, so VRAM pairs still overlap the host pairs' link reads.
//   Only experts one pair uses go in place (a shared one's tile serves all its users with one
//   read), not Q8_0 (slower in place on V100), and only in steps of at most 2 tokens
//   (PF_MAX_TOKENS: plain decoding; slower in a 3-token verify step). These variants are
//   compiled for 4 CTAs per SM (PF_MIN_BLOCKS: 128 registers, no spills); the default ones
//   keep 8.
//
// With `link_ctas` (`--expert-mv-link-ctas`, #710) the tile kernel's grid gets that many worker
// CTAs ahead of its (tile, pair) CTAs. A pair whose expert is read in place over the link (a
// host expert not in a staging slot, see is_link) is left by its own CTAs; the workers go over
// those pairs' tiles instead, each tile through tile_one as before (the same bits). The GPU
// dispatches CTAs in grid order, so the link reads start first, from a few CTAs that keep it
// busy (the link needs far fewer loads in flight than the in-place tiles issue), and the VRAM
// pairs' CTAs take the rest of the SMs meanwhile. Without it a host pair's tiles start wherever
// the pair falls in the grid and hold their SM slots while they wait on the link, so the link
// reads and the VRAM pairs' work add up more than they overlap.
//
// `part` and `steps` split one op into launches (src/model/core/expert_mv/overlap.rs): the VRAM
// experts while `stage_copy` brings the host experts into a VRAM staging buffer on a side
// stream, then the host experts from that buffer (`stage`). A pair is computed by the same code
// whichever launch takes it and wherever its weights are read from, so the results are the same
// bits. With the running GPU's idle NVLink peer filling (part of) the buffer
// (src/model/core/expert_mv/peer.rs, #252), parts 3 and 4 split the pairs by the slot table
// instead: those the peer does not stage (VRAM ones and host ones read in place), then the staged.

#pragma once

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdint>
#include <type_traits>

extern "C" {

struct legion_expert_mv {
    const char * w;        // expert 0
    const char * w2;       // with SwiGLU: the up projection (same layout as w), else null
    int64_t w_nb1, w_nb2;  // bytes per row, per expert
    const float * x;       // [n_tok][n_x][k]: x_nb1 bytes per column, x_nb2 per token
    int64_t x_nb1, x_nb2;
    int32_t n_x;           // 1 (shared by all slots) or n_used
    const int32_t * ids;   // [n_tok][n_used], ids_nb1 bytes per token
    int64_t ids_nb1;
    float * dst;           // [n_tok][n_used][rows]
    int64_t dst_nb1, dst_nb2;
    int32_t k, rows, n_used, n_tok;
    int32_t rows_per_cta;  // expert_mv_tile's rows per CTA
    block_q8_1 * xq;       // [n_tok][n_x][k / 32] scratch
    int32_t vram_below;    // ids below this are VRAM (a tiered tensor's cache slots)
    int32_t vram_kernel;   // VRAM experts: 1 through expert_mv_vram, 0 through tiles as host ones,
                           // 2 left to the caller (the PTX kernels, src/cuda/ptx)
    int32_t n_w;           // experts `w` holds: ids outside [0, n_w) are skipped, never read
    // The pairs computed: 0 all, 1 VRAM ones (id < vram_below), 2 host ones; with a staging
    // table (`stage`): 3 every pair but the staged ones (those read in place), 4 the staged ones.
    int32_t part;
    int32_t steps;         // 1: quantize_x, 2: the expert kernels, 3: both
    // Non-null: host pairs read their expert from a staging buffer stage_copy filled: an int32
    // slot per pair at `stage` (-1: none, read in place), slot s's bytes at stage + stage_hdr +
    // s * stage_slot, this matrix at + stage_w, w2 at + stage_w2.
    const char * stage;
    int64_t stage_hdr, stage_slot, stage_w, stage_w2;
    // expert_mv_tile's dot products, one of DOT_*. The same bits whichever.
    int32_t dot;
    // Non-zero: expert_mv_tile computes pairs whose expert is in VRAM from global memory, each
    // lane's loads up front (see the top of the file); the tile path for the others.
    int32_t vram_pf;
    // SwiGLU only: > 0 clamps the gate above and the up product both ways (see the top).
    float limit;
    // `--tp-moe-ep` (#798): `w` holds experts id_base .. id_base + n_w of the router's ids, so
    // every id is taken minus id_base; with zero_skipped, a skipped pair's rows of dst are
    // written 0 (its share of the sum is the other rank's), else left as they are.
    int32_t id_base;
    int32_t zero_skipped;
    // > 0: expert_mv_tile_piped with this many consecutive tiles per CTA (grid.x: groups of
    // them), the next tile's loads in flight while the current one is computed; 0: expert_mv_tile.
    int32_t pipe_tiles;
    // Non-zero (Q4_K SwiGLU at 3-4 tokens): quantize_x also writes each column's per-unit Q4_K
    // activations (q4k_acts_col) at this byte offset of xq, and the gate/up kernel reads them there.
    int64_t q4k_units;
    // `--tp-moe-ep-place` (#803): non-null, the router's id to its slot in `w`, -1 for experts
    // held elsewhere (n_map entries; id_base is 0); null, ids minus id_base.
    const int32_t * id_map;
    int32_t n_map;
    // > 0: the link pairs (host experts read in place, see is_link) go to this many worker CTAs
    // dispatched ahead of the rest of the grid, the other CTAs leave them (link_worker).
    int32_t link_ctas;
    // `--glm-pair-split` (#735): 0 off; 1 or 2: pairs whose id carries HALF_BIT are this GPU's
    // half (the first or the second) of an expert its NVLink peer computes the other half of
    // (expert_mv_split below). Part 0, VRAM experts through the tiles, no staging.
    int32_t halves;
    // Non-zero (and q4k_units 0): the piped Q4_K gate/up kernel at 3-4 tokens holds activations in
    // registers over a tile group (Q4K_HOLD, #816 round 6). The same bits either way.
    int32_t q4k_hold;
    // Non-zero (with pipe_tiles > 0): the Q4_K gate/up at one token goes to expert_mv_tile_bal, its
    // piped tiles balanced over the CTAs the GPU holds at once, instead of a CTA per tile. The same
    // bits.
    int32_t balance;
    // Non-zero (with pipe_tiles > 0): the piped Q5_1 down at 1-3 tokens takes tiles of half the
    // rows_per_cta rows, pipe_tiles of them a CTA (launch_half). The same bits.
    int32_t half_tiles;
    // Non-null (`--moe-pad-skip-tiered`, #814): a verify step's row map, n_tok entries: row t is a
    // padding row when pad_rows[t] != t. A padding row's pair whose expert no real row's pair chose
    // is skipped (expert_of), its rows written 0 (zero_skipped is set with it).
    const int32_t * pad_rows;
};

// stage_copy: copies the host experts of a step's pairs (each expert once, its first pair's,
// at most `cap` of them in pair order) from their matrices into a staging buffer in VRAM.
//
// The step's distinct host experts are ranked in the order of their first pairs (r = 0, 1, ...).
// Rank r takes a slot when r % t_every < t_take, slot (r / t_every) * t_take + r % t_every, while
// below `cap` (1, 1: every rank, slot r). Of those slots this launch copies the ones with
// (slot % c_every < c_take) != c_flip (1, 1, 0: all). Two GPUs fill one buffer that way (#252):
// the idle NVLink peer of the running GPU copies some slots, the running GPU the others, or the
// peer copies every slot and the running GPU reads the other ranks in place.
struct legion_expert_stage {
    const int32_t * ids;   // [n_tok][n_used], ids_nb1 bytes per token
    int64_t ids_nb1;
    int32_t n_used, n_tok;
    int32_t vram_below;    // ids below this are VRAM already: not copied
    int32_t n_w;           // ids outside [0, n_w) are skipped
    char * stage;          // the slot table (an int32 per pair, written when `table`), then slots
    int64_t hdr, slot;     // the table's bytes, a slot's bytes
    int32_t cap;           // slots
    int32_t table;         // write the slot table
    int32_t n_seg;         // matrices to copy (0..3): an expert's nb2[m] bytes from
    const char * src[3];   // src[m] + id * nb2[m] to the slot + off[m]
    int64_t nb2[3];
    int64_t off[3];
    int32_t ctas;          // CTAs of the copy
    int32_t t_every, t_take;          // which ranks take a slot (above)
    int32_t c_every, c_take, c_flip;  // which slots this launch copies (above)
};

// `type`: ggml_type of the weights, Q4_K, Q5_K, Q5_1, Q6_K or Q8_0, or (GLM-5.3-Flash's Maya
// quants, the per-user dot products only) Q2_K, Q3_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS. Runs quantize_x, then the expert kernels
// (`steps`), for the pairs of `part`.
// `smem`: expert_mv_tile's dynamic shared memory (rows_per_cta * w_nb1, twice with w2). Returns a
// cudaError_t.
int legion_expert_mv_launch(int type, const legion_expert_mv * p, int smem, void * stream);

// Returns a cudaError_t.
int legion_expert_stage_launch(const legion_expert_stage * s, void * stream);

}  // extern "C"

namespace {

constexpr int TILE_MAX     = 16 * 1024;  // bytes of a tile: every 16-byte load of it in flight at once
constexpr int MAX_USERS    = 8;          // users a CTA computes at once (more go in chunks)
constexpr int THREADS      = 128;
// expert_mv_tile's dot products (legion_expert_mv::dot; Rust's expert_mv::Dot): auto (by format
// and user count), one vec_dot per user as ggml's mmvq calls it, each weight unit decoded once for
// every user (rows_dot_once), or that with rows of 32 lanes summed by 16, two rows per warp.
enum Dot : int32_t { DOT_AUTO = 0, DOT_PER_USER = 1, DOT_ONCE = 2, DOT_SUB16 = 3 };
// expert_mv_tile with `vram_pf`: CTAs per SM it is compiled for (registers for a lane's loads).
constexpr int PF_MIN_BLOCKS = 4;
// The most tokens of a step `vram_pf` takes; longer steps keep the tiles.
constexpr int PF_MAX_TOKENS = 2;
constexpr int VRAM_ROWS    = 2;  // rows a warp works on at once in expert_mv_vram
// With `halves`: the id bit marking a pair as half of its expert (expert_mv_split).
constexpr int32_t HALF_BIT = 1 << 30;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));  // ggml-cuda's ggml_cuda_op_silu_single
}

// SwiGLU of the gate and up products, clamped when `limit` > 0 (else the plain product).
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return silu(g) * u;
}

template <ggml_type type>
__device__ __forceinline__ float vec_dot(const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return vec_dot_q5_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        return vec_dot_q5_1_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return vec_dot_q6_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return vec_dot_q2_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        return vec_dot_q3_K_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return vec_dot_iq2_s_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        return vec_dot_iq3_s_q8_1(vbq, bq8_1, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        return vec_dot_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs);
    } else {
        return vec_dot_q8_0_q8_1(vbq, bq8_1, kbx, iqs);
    }
}

// The Maya experts' IQ dot products in expert_mv_tile (#574). ggml's vec_dot_iq*_q8_1 look their
// grid up in global memory (4-8 scattered 4-8 byte loads per unit, the L1/TEX pipe the busiest unit
// on V100) and apply the signs with __vcmpne4 and __vsub4, emulated on sm_70. Here load_grid copies
// a type's grid to shared memory once per CTA, and the iq_dot_* below are ggml's
// vec_dot_iq2_xxs_q8_1, vec_dot_iq2_s_q8_1 and vec_dot_iq3_xxs_q8_1 (vecdotq.cuh, MIT) with that
// table and signed_grid: the same values in the same order, the same bits. 12% less time in the
// IQ tile kernels on V100 (Maya-S, MTP k = 1).

// A grid value with ggml's signs applied: byte j negated where bit j of `n` (4 bits) is set.
// ggml forms the mask m with __vcmpne4 (0xFF bytes) and takes __vsub4(g ^ m, m); here the bits
// are spread to the bytes' low bits (b: the 4-bit chunks at shifts 0, 7, 14, 21 never overlap)
// and m = b * 0xFF, and (g ^ m) + b is that subtraction byte for byte, as no grid byte is 0 (all
// are 4..62: no byte carries). Integer operations only, no lookups.
__device__ __forceinline__ int signed_grid(uint32_t g, uint32_t n) {
    const uint32_t b = (n * 0x00204081u) & 0x01010101u;
    return (int) ((g ^ (b * 0xFFu)) + b);
}

// unpack_ksigns's 8 sign bits (vecdotq.cuh): the 7 stored ones and their parity as bit 7.
__device__ __forceinline__ uint32_t ksigns8(uint8_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

struct GridIQ2XXS {
    uint2 grid[256];
};
struct GridIQ2S {
    uint2 grid[1024];
};
struct GridIQ3XXS {
    uint32_t grid[256];
};
template <ggml_type type> struct GridOf {
    using T = char;
};
template <> struct GridOf<GGML_TYPE_IQ2_XXS> {
    using T = GridIQ2XXS;
};
template <> struct GridOf<GGML_TYPE_IQ2_S> {
    using T = GridIQ2S;
};
template <> struct GridOf<GGML_TYPE_IQ3_XXS> {
    using T = GridIQ3XXS;
};

// A type's tables: one static shared variable per type, so a kernel holds only its own.
template <ggml_type type>
__device__ __forceinline__ typename GridOf<type>::T & smem_grid() {
    __shared__ typename GridOf<type>::T t;
    return t;
}

template <ggml_type type>
__host__ __device__ constexpr bool grid_in_smem() {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS;
}

// Every thread of the CTA; a __syncthreads must follow before the first lookup.
template <ggml_type type>
__device__ __forceinline__ void load_grid(int threads) {
    auto & t = smem_grid<type>();
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        for (int i = threadIdx.x; i < 256; i += threads) {
            t.grid[i] = ((const uint2 *) iq2xxs_grid)[i];
        }
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        for (int i = threadIdx.x; i < 1024; i += threads) {
            t.grid[i] = ((const uint2 *) iq2s_grid)[i];
        }
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        for (int i = threadIdx.x; i < 256; i += threads) {
            t.grid[i] = iq3xxs_grid[i];
        }
    }
}

// The IQ types' activations in steps of 1 or 2 tokens (#574): quantize_x writes them split, each
// column's Q8_1 quants contiguous (k bytes per column, 16-byte aligned) and after all the columns
// their half2 (d, sum) scales, in the bytes the Q8_1 blocks took (k + k / 32 * 4 = k / 32 * 36 per
// column). A lane then loads a block's 32 quants with two 16-byte loads instead of eight 4-byte
// ones 36 bytes apart (each of those spans ~9 L1 lines across a warp; the L1 data pipe was the
// busiest unit of the IQ tile kernels on V100). The same values in the same order: the same bits.
// Only with the tile kernels compiled for 1-2 users (MU): the 8-user one spilled more with it
// (IQ2_S down 17% slower on V100), so bigger steps, and `vram_pf` (whose kernels take 8), keep
// block_q8_1.
template <ggml_type type, int MU>
__host__ __device__ constexpr bool xq_split() {
    return grid_in_smem<type>() && MU <= 2;
}

// The layout quantize_x writes and expert_mv_vram reads for a step: the tile kernel's (MU, see
// launch_tile_pf).
template <ggml_type type>
bool xq_split_for(const legion_expert_mv & p) {
    return grid_in_smem<type>() && p.n_tok <= 2 && p.vram_pf == 0;
}

// A split column from block `b` on: its quants at qs (32 a block), its scales at ds.
struct Q8Split {
    const int8_t * qs;
    const half2 * ds;
};

// Column `col` of the activations (`[n_tok][n_x]` columns), split (S) or as block_q8_1.
template <bool S>
__device__ __forceinline__ auto xq_col(const legion_expert_mv & p, int col) {
    if constexpr (S) {
        const int8_t * q = (const int8_t *) p.xq;
        const half2 * ds = (const half2 *) (q + (int64_t) p.n_x * p.n_tok * p.k);
        return Q8Split{q + (int64_t) col * p.k, ds + (int64_t) col * (p.k / QK8_1)};
    } else {
        return (const block_q8_1 *) p.xq + (int64_t) col * (p.k / QK8_1);
    }
}

// Block `b` on of a column.
__device__ __forceinline__ Q8Split xq_at(Q8Split y, int b) {
    return {y.qs + QK8_1 * b, y.ds + b};
}
__device__ __forceinline__ const block_q8_1 * xq_at(const block_q8_1 * y, int b) {
    return y + b;
}

// Block b's 32 quants as 8 ints (get_int_b4 of block_q8_1::qs) and its d: split, both loaded up
// front; block_q8_1, each where it is used (the code these kernels had before the split layout).
struct Q8Block {
    int q[8];
    float s;
    __device__ __forceinline__ int u(int k) const { return q[k]; }
    __device__ __forceinline__ float d() const { return s; }
};
__device__ __forceinline__ Q8Block q8_block(Q8Split y, int b) {
    const int4 lo = __ldg((const int4 *) (y.qs + QK8_1 * b));
    const int4 hi = __ldg((const int4 *) (y.qs + QK8_1 * b + 16));
    return {{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}, __low2float(__ldg(y.ds + b))};
}
struct Q8Ref {
    const block_q8_1 * __restrict__ b;
    __device__ __forceinline__ int u(int k) const { return get_int_b4(b->qs, k); }
    __device__ __forceinline__ float d() const { return __low2float(b->ds); }
};
__device__ __forceinline__ Q8Ref q8_block(const block_q8_1 * __restrict__ y, int b) {
    return {y + b};
}

template <typename Y>
__device__ __forceinline__ float iq_dot_iq2_xxs(const void * __restrict__ vbq, Y y,
                                                int kbx, int iqs, const GridIQ2XXS & t) {
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2         = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    const auto yb        = q8_block(y, iqs / 2);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = t.grid[aux8[k0 / 2]];
        const uint32_t signs = ksigns8((uint8_t) (aux32 >> (7 * k0 / 2)));

        const int grid0 = signed_grid(grid_pos.x, signs & 0x0F);
        const int u0    = yb.u(k0 + 0);
        sumi            = ggml_cuda_dp4a(grid0, u0, sumi);

        const int grid1  = signed_grid(grid_pos.y, signs >> 4);
        const int u1     = yb.u(k0 + 1);
        sumi             = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls  = aux32 >> 27 | 1;
    sumi          = sumi * ls / 8;
    const float d = __half2float(bq2->d) * yb.d();
    return d * sumi;
}

template <typename Y>
__device__ __forceinline__ float iq_dot_iq2_s(const void * __restrict__ vbq, Y y,
                                              int kbx, int iqs, const GridIQ2S & t) {
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t * qs  = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs / 2];

    const int signs_packed_32      = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    const auto yb = q8_block(y, iqs / 2);

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = t.grid[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];

        const uint32_t signs0 = signs_packed_8[l0 / 2] & 0x0F;
        const uint32_t signs1 = signs_packed_8[l0 / 2] >> 4;

        const int grid_l = signed_grid(grid_pos.x, signs0);
        const int grid_h = signed_grid(grid_pos.y, signs1);

        const int u0 = yb.u(l0 + 0);
        const int u1 = yb.u(l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;

    const float d = __half2float(bq2->d) * yb.d();
    return d * sumi;
}

template <typename Y>
__device__ __forceinline__ float iq_dot_iq3_xxs(const void * __restrict__ vbq, Y y,
                                                int kbx, int iqs, const GridIQ3XXS & t) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * q3   = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    const auto yb        = q8_block(y, iqs / 2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t signs = ksigns8((uint8_t) (aux32 >> (7 * l0 / 2)));

        const int grid_l = signed_grid(t.grid[q3[l0 + 0]], signs & 0x0F);

        const int u0 = yb.u(l0 + 0);

        const int grid_h = signed_grid(t.grid[q3[l0 + 1]], signs >> 4);

        const int u1 = yb.u(l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls  = aux32 >> 28;
    sumi          = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * yb.d();
    return d * sumi;
}

// The IQ dot products with grid `t` (shared memory, or ggml's tables in global memory).
template <ggml_type type, typename Y>
__device__ __forceinline__ float vec_dot_iq(const void * vbq, Y y, int kbx, int iqs,
                                            const typename GridOf<type>::T & t) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return iq_dot_iq2_xxs(vbq, y, kbx, iqs, t);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return iq_dot_iq2_s(vbq, y, kbx, iqs, t);
    } else {
        static_assert(type == GGML_TYPE_IQ3_XXS, "an IQ type with its grid in shared memory");
        return iq_dot_iq3_xxs(vbq, y, kbx, iqs, t);
    }
}

// vec_dot, with the IQ grids from shared memory (load_grid ran in this CTA).
template <ggml_type type>
__device__ __forceinline__ float vec_dot_tile(const void * vbq, Q8Split y, int kbx, int iqs) {
    return vec_dot_iq<type>(vbq, y, kbx, iqs, smem_grid<type>());
}
template <ggml_type type>
__device__ __forceinline__ float vec_dot_tile(const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return iq_dot_iq2_xxs(vbq, bq8_1, kbx, iqs, smem_grid<type>());
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return iq_dot_iq2_s(vbq, bq8_1, kbx, iqs, smem_grid<type>());
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return iq_dot_iq3_xxs(vbq, bq8_1, kbx, iqs, smem_grid<type>());
    } else {
        return vec_dot<type>(vbq, bq8_1, kbx, iqs);
    }
}

// vec_dot for expert_mv_vram: the IQ types (split activations) with ggml's grids as they are.
template <ggml_type type>
__device__ __forceinline__ float vec_dot_vram(const void * vbq, Q8Split y, int kbx, int iqs) {
    using T = typename GridOf<type>::T;
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq<type>(vbq, y, kbx, iqs, *(const T *) iq2xxs_grid);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return vec_dot_iq<type>(vbq, y, kbx, iqs, *(const T *) iq2s_grid);
    } else {
        return vec_dot_iq<type>(vbq, y, kbx, iqs, *(const T *) iq3xxs_grid);
    }
}
template <ggml_type type>
__device__ __forceinline__ float vec_dot_vram(const void * vbq, const block_q8_1 * bq8_1, int kbx, int iqs) {
    return vec_dot<type>(vbq, bq8_1, kbx, iqs);
}

template <ggml_type type>
__host__ __device__ constexpr int vdr() {
    return type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q5_K ? VDR_Q5_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q5_1 ? VDR_Q5_1_Q8_1_MMVQ :
           type == GGML_TYPE_Q6_K ? VDR_Q6_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q2_K ? VDR_Q2_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q3_K ? VDR_Q3_K_Q8_1_MMVQ :
           type == GGML_TYPE_IQ2_XXS ? VDR_IQ2_XXS_Q8_1_MMVQ :
           type == GGML_TYPE_IQ2_S ? VDR_IQ2_S_Q8_1_MMVQ :
           type == GGML_TYPE_IQ3_XXS ? VDR_IQ3_XXS_Q8_1_MMVQ :
           type == GGML_TYPE_IQ3_S ? VDR_IQ3_S_Q8_1_MMVQ :
           type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ :
                                    VDR_Q8_0_Q8_1_MMVQ;
}

// Types with a `Unit` (the decoded-once dot products, DOTS 1 and 2). The others (GLM-5.3-Flash's
// Maya quants: Q2_K, Q3_K and the IQ types, #559) always take the per-user vec_dot (DOTS 0).
template <ggml_type type>
__host__ __device__ constexpr bool has_unit() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_Q8_0;
}

// vec_dot calls per row and token.
template <ggml_type type>
__host__ __device__ int units_per_row(int k) {
    return k / ggml_cuda_type_traits<type>::qk * (ggml_cuda_type_traits<type>::qi / vdr<type>());
}

// Lanes sharing a row: each with >= 4 of its vec_dots (ILP), so short rows (Q5_1, K = 640: 40
// calls) run several rows per warp instead of idling lanes.
__host__ __device__ inline int lanes_per_row(int units) {
    int L = WARP_SIZE;
    while (L > 1 && units / L < 4) {
        L /= 2;
    }
    return L;
}

// `s` summed over the G lanes sharing a row (G a power of 2, warp-uniform), in the order of
// `for (off = G / 2; off > 0; off >>= 1) s += shfl_xor(s, off)`: the same bits, but with a
// compile-time trip count (the loop over a runtime G cost a warp ~14 instructions a step in
// convergence checks; #574).
__device__ __forceinline__ float sum_lanes(float s, int G) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        if (off < G) {
            s += __shfl_xor_sync(0xffffffff, s, off, WARP_SIZE);
        }
    }
    return s;
}

// A router id's slot in `w`: through id_map when set (-1 for experts held elsewhere), else
// minus id_base.
__device__ __forceinline__ int slot_of(const legion_expert_mv & p, int id) {
    if (p.id_map) {
        return id >= 0 && id < p.n_map ? p.id_map[id] : -1;
    }
    return id - p.id_base;
}

__device__ __forceinline__ int id_at(const legion_expert_mv & p, int pair) {
    return *(const int32_t *) ((const char *) p.ids + (pair / p.n_used) * p.ids_nb1 + (pair % p.n_used) * 4);
}

// A pair's expert, -1 to skip it. With pad_rows, a padding row's pair whose expert no real row's
// pair chose (a dead padding pair) is skipped too: experts only padding rows chose are never read.
// find_users still lists every pair of the experts it computes (padding ones included), so a
// computed expert has the same users as without pad_rows: the same instructions on the same
// values, the same bits for every real row.
__device__ __forceinline__ int expert_of(const legion_expert_mv & p, int pair) {
    const int e = slot_of(p, id_at(p, pair));
    const int t = pair / p.n_used;
    if (p.pad_rows == nullptr || p.pad_rows[t] == t) {
        return e;
    }
    for (int r = 0; r < p.n_tok; r++) {
        if (p.pad_rows[r] != r) {
            continue;
        }
        for (int j = 0; j < p.n_used; j++) {
            if (slot_of(p, id_at(p, r * p.n_used + j)) == e) {
                return e;
            }
        }
    }
    return -1;
}

// Whether expert `e` is one of the pairs `part` computes: VRAM (1) or host (2, 4) ones, or any
// (0, 3). Parts 3 and 4 also look at the staging table (tile_one).
__device__ __forceinline__ bool in_part(const legion_expert_mv & p, int e) {
    return p.part == 0 || p.part == 3 || (p.part == 1) == (e < p.vram_below);
}

// The Q4_K gate/up kernel's activations per unit (#816 round 4), written by quantize_x beside
// the Q8_1 columns when `q4k_units` is set. Unit c of a row (c < k / 16; vec_dot_q4_K_q8_1's kbx
// = c / 16, iqs = 2 * (c % 16)) reads ints q = c % 4 and q + 4 of the Q8_1 blocks 8 kbx + 2 jj and
// + 1 (jj = (c % 16) / 4), and their scales. Column col's units are
//   u[c]  = {block 0 int q, block 0 int q + 4, block 1 int q, block 1 int q + 4}  (int4)
//   ds[c] = {d8 of block 0, d8 of block 1 (as floats), dot2 of block 0, dot2 of block 1}
// at xq + q4k_units + col * k / 16 * 32 (u), and k / 16 * 16 bytes on (ds): two coalesced 16-byte
// loads per user and unit instead of six narrow ones. d8 is __low2float of the block's ds and dot2
// the sum of the unit's 8 quants of the block (the dp4a's with 0x01010101 the kernel did per user
// and unit): the same values, so the same bits.
__device__ __forceinline__ int4 * q4k_acts_col(const legion_expert_mv & p, int col) {
    return (int4 *) ((char *) p.xq + p.q4k_units) + (int64_t) col * (p.k / 16) * 2;
}

// quantize_x's part of it: the warp holding element el's block (lane el % 32 holds quant q)
// writes the words of the four units that read that block.
__device__ __forceinline__ void q4k_acts_write(const legion_expert_mv & p, int col, int el, int8_t q, half2 ds) {
    const int lane = el % QK8_1;
    // Lane 4 m: int m of the block (little-endian, as get_int_b4 reads it) and its quants' sum.
    const int b = (int) (uint8_t) q;
    const int w = b | (__shfl_down_sync(0xffffffff, b, 1) << 8) | (__shfl_down_sync(0xffffffff, b, 2) << 16) |
                  (__shfl_down_sync(0xffffffff, b, 3) << 24);
    int s4 = q;
    s4 += __shfl_xor_sync(0xffffffff, s4, 1);
    s4 += __shfl_xor_sync(0xffffffff, s4, 2);
    const int w_hi  = __shfl_down_sync(0xffffffff, w, 16);   // int m + 4
    const int s4_hi = __shfl_down_sync(0xffffffff, s4, 16);
    if (lane < 16 && lane % 4 == 0) {
        const int blk  = el / QK8_1;  // the block in the column
        const int unit = (blk / 8) * 16 + ((blk % 8) / 2) * 4 + lane / 4;
        const int s    = blk % 2;
        int * u        = (int *) (q4k_acts_col(p, col) + unit);
        int * d        = (int *) (q4k_acts_col(p, col) + p.k / 16 + unit);
        u[2 * s + 0]   = w;
        u[2 * s + 1]   = w_hi;
        d[s]           = __float_as_int(__low2float(ds));
        d[2 + s]       = s4 + s4_hi;
    }
}

// X -> Q8_1 as ggml's quantize_q8_1: one thread per element, a warp per 32-element block.
// grid.y: column c = token c / n_x, slot c % n_x. With a column per pair (n_x == n_used), only
// the columns of `part`'s pairs.
// SPLIT: the IQ types' layout (xq_split).
template <bool SPLIT>
__global__ void quantize_x(const legion_expert_mv p) {
    const int el = blockIdx.x * blockDim.x + threadIdx.x;
    if (el >= p.k) {
        return;
    }
    const int c = blockIdx.y;
    if (p.part != 0 && p.n_x == p.n_used && !in_part(p, expert_of(p, c))) {
        return;  // the other stream's pair
    }
    const float * col = (const float *) ((const char *) p.x + (c / p.n_x) * p.x_nb2 + (c % p.n_x) * p.x_nb1);
    const float xi    = col[el];
    float amax        = fabsf(xi);
    float sum         = xi;
    amax              = warp_reduce_max<QK8_1>(amax);
    sum               = warp_reduce_sum<QK8_1>(sum);
    const float d     = amax / 127.0f;
    const int8_t q    = amax == 0.0f ? 0 : roundf(xi / d);
    if constexpr (SPLIT) {
        int8_t * qs                  = (int8_t *) p.xq;
        half2 * ds                   = (half2 *) (qs + (int64_t) p.n_x * p.n_tok * p.k);
        qs[(int64_t) c * p.k + el] = q;
        if (el % QK8_1 == 0) {
            ds[(int64_t) c * (p.k / QK8_1) + el / QK8_1] = make_half2(d, sum);
        }
    } else {
        block_q8_1 * b    = p.xq + (int64_t) c * (p.k / QK8_1) + el / QK8_1;
        b->qs[el % QK8_1] = q;
        if (el % QK8_1 == 0) {
            b->ds = make_half2(d, sum);
        }
        if (p.q4k_units != 0) {
            q4k_acts_write(p, c, el, q, make_half2(d, sum));
        }
    }
}

// The pairs a CTA computes: the pairs from `pair` on that chose its expert `e`, skipping the
// first `skip` of them and taking at most CAP (the CTA goes over them in such chunks);
// `total` counts them all. None (n = total = 0) when an earlier pair chose `e` (that pair's
// CTAs do the work). Each warp finds them on its own with coalesced loads and ballots: no
// shared memory, no barrier.
struct Users {
    int n, total;
    int q[MAX_USERS];
};

template <int CAP = MAX_USERS>
__device__ __forceinline__ Users find_users(const legion_expert_mv & p, int pair, int e, int skip) {
    const int lane = threadIdx.x % WARP_SIZE, n_pairs = p.n_used * p.n_tok;
    Users us;
    us.n     = 0;
    us.total = 0;
#pragma unroll
    for (int u = 0; u < MAX_USERS; u++) {
        us.q[u] = pair;
    }
    for (int base = 0; base < n_pairs; base += WARP_SIZE) {
        const int q = base + lane;
        const int id =
            q < n_pairs ? *(const int32_t *) ((const char *) p.ids + (q / p.n_used) * p.ids_nb1 + (q % p.n_used) * 4) : -1;
        unsigned m = __ballot_sync(0xffffffff, q < n_pairs && slot_of(p, id) == e);  // as expert_of
        if (base < pair && (m & (pair - base >= WARP_SIZE ? 0xffffffffu : (1u << (pair - base)) - 1))) {
            us.n     = 0;
            us.total = 0;
            return us;
        }
        if (pair > base) {
            m &= pair - base >= WARP_SIZE ? 0 : ~((1u << (pair - base)) - 1);
        }
        while (m) {
            const int b = __ffs(m) - 1;
            m &= m - 1;
            if (us.total >= skip && us.n < CAP) {
#pragma unroll
                for (int u = 0; u < MAX_USERS; u++) {
                    if (u == us.n) {
                        us.q[u] = base + b;
                    }
                }
                us.n++;
            }
            us.total++;
        }
    }
    return us;
}

// The dot products of `nrows` rows (at `rows_at`, and `rows_at2` with SwiGLU; w_nb1 apart: rows
// r0.. of the expert) with the activations of each user pair (us.n <= NU), written to dst. L
// lanes share a row. NI 0: a lane's units in a loop; NI > 0 (at least the units a lane has): the
// same units in the same order, unrolled, so the compiler issues all of a row's weight loads
// before the first dot product (rows read in place from VRAM, `vram_pf`). expert_mv_tile only: the
// IQ types take vec_dot_tile, the grids its CTA loaded. S: split activations (xq_split). XS (not
// with S): user u's activation column is the copy at xs + u * (k / 32) in shared memory
// (expert_mv_tile_piped's), the same values as in xq.
// expert_mv_split's K halves (the defaults otherwise, which fold away): the rows hold `kk`
// values of K from block kb0 on, `stride` bytes apart.
template <ggml_type type, int NU, bool GLU, int NI = 0, bool S = false, bool XS = false>
__device__ __forceinline__ void rows_dot(const legion_expert_mv & p, const char * rows_at, const char * rows_at2,
                                         int nrows, int r0, const Users & us, const block_q8_1 * xs = nullptr,
                                         int kb0 = 0, int kk = 0, int64_t stride = 0) {
    static_assert(!(S && XS), "shared-memory activations are block_q8_1 columns");
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int VDR             = vdr<type>();
    constexpr int lanes_per_block = qi / VDR;
    constexpr int M               = GLU ? 2 : 1;
    const int units     = units_per_row<type>(kk > 0 ? kk : p.k);
    const int64_t nb1   = stride > 0 ? stride : p.w_nb1;
    const int xcols     = p.k / QK8_1;
    const int L         = lanes_per_row(units);
    const int sub       = threadIdx.x % L;
    decltype(xq_col<S>(p, 0)) ys[NU];
#pragma unroll
    for (int u = 0; u < NU; u++) {
        const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
        if constexpr (S) {
            ys[u] = xq_col<S>(p, t * p.n_x + (p.n_x == 1 ? 0 : j));
        } else if constexpr (XS) {
            ys[u] = xs + u * xcols;
        } else {
            ys[u] = p.xq + (int64_t) (t * p.n_x + (p.n_x == 1 ? 0 : j)) * xcols;
        }
        ys[u] = xq_at(ys[u], kb0 * (qk / QK8_1));
    }
    // Whole warps go round the loop together (the shuffles take every lane).
    const int lane = threadIdx.x % WARP_SIZE;
    for (int rw = threadIdx.x / WARP_SIZE * (WARP_SIZE / L); rw < nrows; rw += THREADS / L) {
        const int r = rw + lane / L;
        const char * row[M];
        row[0] = rows_at + min(r, nrows - 1) * nb1;
        if constexpr (GLU) {
            row[1] = rows_at2 + min(r, nrows - 1) * nb1;
        }
        float acc[M][NU] = {};
        auto unit = [&](int c) {
            const int kbx = c / lanes_per_block, iqs = VDR * (c % lanes_per_block);
#pragma unroll
            for (int u = 0; u < NU; u++) {
                if (NU == 1 || u < us.n) {
#pragma unroll
                    for (int m = 0; m < M; m++) {
                        if constexpr (S) {
                            acc[m][u] += vec_dot_tile<type>(row[m], xq_at(ys[u], kbx * (qk / QK8_1)), kbx, iqs);
                        } else {
                            acc[m][u] += vec_dot_tile<type>(row[m], &ys[u][kbx * (qk / QK8_1)], kbx, iqs);
                        }
                    }
                }
            }
        };
        if constexpr (NI == 0) {
            for (int c = r < nrows ? sub : units; c < units; c += L) {
                unit(c);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NI; i++) {
                const int c = sub + i * L;
                if (r < nrows && c < units) {
                    unit(c);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NU; u++) {
            if (NU == 1 || u < us.n) {
                float s[M];
#pragma unroll
                for (int m = 0; m < M; m++) {
                    s[m] = sum_lanes(acc[m][u], L);
                }
                if (sub == 0 && r < nrows) {
                    const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
                    float * out = (float *) ((char *) p.dst + t * p.dst_nb2 + j * p.dst_nb1) + r0;
                    out[r]      = GLU ? swiglu(s[0], s[M - 1], p.limit) : s[0];
                }
            }
        }
    }
}

// One vec_dot unit (block kbx, quants iqs) split in its two halves, each loaded and decoded once:
// `load` is the weight half of ggml's vec_dot_*_q8_1 (vecdotq.cuh, MIT), `Acts::load` the
// activation half, and `dot` the same *_impl call, so `w.load(x, kbx, iqs); a.load(y + kbx *
// (qk / QK8_1), iqs); w.dot(a)` is `vec_dot<type>(x, y + ..., kbx, iqs)` bit for bit. A row's
// unit then serves every user, and a user's activations both SwiGLU matrices. (Adapted from
// Strata #783 PR-h's decoded-once small-format tables, 2715a75, MIT: the weights' loads and
// unpacking once per row and block, not once per activation column.)
template <ggml_type type> struct Unit;

// The K-quants' 6-bit scale and min of sub-blocks bq8_offset and +1 (ggml's branchless form).
__device__ __forceinline__ void k_scales(const uint8_t * scales_bytes, int bq8_offset, uint16_t (&aux)[2]) {
    const uint16_t * scales = (const uint16_t *) scales_bytes;
    const int j  = bq8_offset / 2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) - (int32_t) (j >= 2);

    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

// The activations of a K-quant unit: 2 x QR ints and QR scales from sub-blocks bq8_offset.. .
template <int QR> struct KActs {
    int u[2 * QR];
    float d8[QR];

    __device__ __forceinline__ void load(const block_q8_1 * bq8_1, int iqs) {
        const int bq8_offset = QR * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
            d8[i]                   = __low2float(bq8i->ds);

            const int * q8 = (const int *) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0]   = q8[0];
            u[2 * i + 1]   = q8[4];
        }
    }
};

template <> struct Unit<GGML_TYPE_Q4_K> {
    using Acts = KActs<QR4_K>;
    int v[2];
    uint16_t aux[2];
    half2 dm;

    __device__ __forceinline__ void load(const void * vbq, int kbx, int iqs) {
        const block_q4_K * b   = (const block_q4_K *) vbq + kbx;
        const int bq8_offset   = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int * q4         = (const int *) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        v[0]                   = q4[0];
        v[1]                   = q4[4];
        k_scales(b->scales, bq8_offset, aux);
        dm = b->dm;
    }

    __device__ __forceinline__ float dot(const Acts & a) const {
        const uint8_t * sc = (const uint8_t *) aux;
        return vec_dot_q4_K_q8_1_impl_vmmq(v, a.u, sc, sc + 2, dm, a.d8);
    }
};

template <> struct Unit<GGML_TYPE_Q5_K> {
    using Acts = KActs<QR5_K>;
    int vl[2], vh[2];
    uint16_t aux[2];
    half2 dm;

    __device__ __forceinline__ void load(const void * vbq, int kbx, int iqs) {
        const block_q5_K * b   = (const block_q5_K *) vbq + kbx;
        const int bq8_offset   = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int * ql         = (const int *) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int * qh         = (const int *) (b->qh + 4 * ((iqs / 2) % 4));
        vl[0]                  = ql[0];
        vl[1]                  = ql[4];
        vh[0]                  = qh[0] >> bq8_offset;
        vh[1]                  = qh[4] >> bq8_offset;
        k_scales(b->scales, bq8_offset, aux);
        dm = b->dm;
    }

    __device__ __forceinline__ float dot(const Acts & a) const {
        const uint8_t * sc = (const uint8_t *) aux;
        return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, a.u, sc, sc + 2, dm, a.d8);
    }
};

template <> struct Unit<GGML_TYPE_Q5_1> {
    struct Acts {
        int u[2 * VDR_Q5_1_Q8_1_MMVQ];
        half2 ds;

        __device__ __forceinline__ void load(const block_q8_1 * bq8_1, int iqs) {
#pragma unroll
            for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
                u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
                u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            }
            ds = bq8_1->ds;
        }
    };
    int vl[VDR_Q5_1_Q8_1_MMVQ], vh[VDR_Q5_1_Q8_1_MMVQ];
    half2 dm;

    __device__ __forceinline__ void load(const void * vbq, int kbx, int iqs) {
        const block_q5_1 * b = (const block_q5_1 *) vbq + kbx;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
            vl[i] = get_int_b4(b->qs, iqs + i);
            vh[i] = get_int_b4(b->qh, 0) >> (4 * (iqs + i));
        }
        dm = b->dm;
    }

    __device__ __forceinline__ float dot(const Acts & a) const {
        return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, a.u, dm, a.ds);
    }
};

template <> struct Unit<GGML_TYPE_Q8_0> {
    struct Acts {
        int u[VDR_Q8_0_Q8_1_MMVQ];
        half d8;

        __device__ __forceinline__ void load(const block_q8_1 * bq8_1, int iqs) {
#pragma unroll
            for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
                u[i] = get_int_b4(bq8_1->qs, iqs + i);
            }
            d8 = __low2half(bq8_1->ds);
        }
    };
    int v[VDR_Q8_0_Q8_1_MMVQ];
    half d;

    __device__ __forceinline__ void load(const void * vbq, int kbx, int iqs) {
        const block_q8_0 * b = (const block_q8_0 *) vbq + kbx;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
            v[i] = get_int_b2(b->qs, iqs + i);
        }
        d = b->d;
    }

    __device__ __forceinline__ float dot(const Acts & a) const {
        return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, a.u, d, a.d8);
    }
};

template <> struct Unit<GGML_TYPE_Q6_K> {
    struct Acts {
        int u[QR6_K];
        float d8[QR6_K];

        __device__ __forceinline__ void load(const block_q8_1 * bq8_1, int iqs) {
            const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                u[i]  = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
                d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
            }
        }
    };
    int vl, vh;
    int8_t sc[4 * (QR6_K - 1) + 1];
    float d;

    __device__ __forceinline__ void load(const void * vbq, int kbx, int iqs) {
        const block_q6_K * b   = (const block_q6_K *) vbq + kbx;
        const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        const int vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
        vl = get_int_b2(b->ql, iqs);
        vh = get_int_b2(b->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            sc[4 * i] = b->scales[scale_offset + 4 * i];
        }
        d = b->d;
    }

    __device__ __forceinline__ float dot(const Acts & a) const {
        return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, a.u, sc, d, a.d8);
    }
};

}  // namespace

namespace legion_emv {

// expert_mv's kernels for one weight type after quantize_x (the `steps & 2` part of
// legion_expert_mv_launch): launch_any<type> in kernels.cuh, defined in the type's own .cu.
template <ggml_type type>
int launch_type(const legion_expert_mv & p, int smem, cudaStream_t stream);

template <> int launch_type<GGML_TYPE_Q4_K>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q5_K>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q5_1>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q6_K>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q8_0>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q2_K>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_Q3_K>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_IQ2_XXS>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_IQ2_S>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_IQ3_XXS>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_IQ3_S>(const legion_expert_mv & p, int smem, cudaStream_t stream);
template <> int launch_type<GGML_TYPE_IQ4_XS>(const legion_expert_mv & p, int smem, cudaStream_t stream);

}  // namespace legion_emv
