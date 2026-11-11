// Paged attention over Legion's KV block pool: attention that reads every sequence's K/V
// blocks in place through its block table, instead of gathering them into a contiguous span.
// The graph side (inputs, layouts, the CPU reference) is src/model/core/paged.rs.
//
// Layouts (ggml order, dim 0 fastest):
// - q:      F32 [d, n_head, n_tok, n_seqs], any byte strides (a view of the RoPE output)
// - K, V:   F16 pools [d * n_head_kv, cells]; cell = block * 64 + offset
// - table:  I32 [table_width, n_seqs]: each sequence's blocks in order; with `ring` > 0 a
//           ring of `ring` blocks (a sliding-window layer's cache): position p sits in entry
//           (p / 64) % ring, which the caller keeps covering the window it reads
// - n_past: I32 [n_seqs]: token t of sequence s is at position n_past[s] + t and sees keys
//           0..=n_past[s] + t (causal); with a sliding window w > 0 only the last w of them,
//           n_past[s] + t - w + 1.. (sliding-window attention layers)
// - dst:    F32 [d, n_head, n_tok, n_seqs], contiguous
//
// Three kernels:
// - decode (n_tok == 1): flash-decoding. One CTA per (kv head, sequence, split); a split is a
//   run of whole blocks, a sequence uses min(max_split, its block count) splits, so short and
//   long sequences of one batch both spread over the GPU. The GQA query heads of a kv head share
//   every K/V load. F32 math throughout (F16 K/V converted on load), online softmax per split,
//   then a combine kernel merges the splits' (max, sum, output) in F32.
// - prefill (n_tok > 1) on sm_80+: flash attention on mma.sync, one pass with an online
//   softmax (paged_prefill_mma, described where it is defined). Given a split scratch
//   (max_split > 1, --paged-verify-split: small groups such as speculative verify steps), the
//   same kernel split over the key blocks as decode is, and a combine kernel (#697).
// - prefill with LEGION_PREFILL_M884 (opt-in; meant for sm_70): one pass with an online softmax
//   on mma.sync m8n8k4 (paged_prefill_m884, described where it is defined).
// - prefill with LEGION_PREFILL_FFMA (opt-in; meant for Turing without tensor cores): one pass
//   with an online softmax on FP32 FMAs (paged_prefill_ffma, described where it is defined).
// - prefill on sm_70 and sm_75 otherwise: tensor cores through the portable WMMA API. A CTA takes 64
//   query rows (64 / G tokens times the G query heads of one kv head) and walks the key blocks
//   twice: first for each row's softmax max and sum, then to accumulate exp(s - max) / sum
//   times V. Recomputing Q K^T costs 1.5x the flops of a one-pass online softmax but needs no
//   per-row rescaling of WMMA accumulators, whose element layout the API does not expose.
//   Given a split scratch (an MTP head's take-in pass, whose last rows alone are read), the
//   last token's rows of the same kernel come split over the key blocks instead, the same bits
//   (sm70/paged_split.cuh); given one for all rows (vsplit_chunk > 0: speculative verify steps
//   and other groups of 2 to 16 tokens per sequence), every row of the same kernel split over the
//   key blocks, the same bits (sm70/paged_vsplit.cuh, #1211).
//
// Both prefill kernels round Q and P to F16 for the MMA (as cuBLAS does for the plain mul_mat
// path); scores, softmax and the output accumulate in F32. On Volta, Turing and consumer
// Blackwell (sm_12x: GB10 measured, relative bias -4.7e-5 over 8K keys) the tensor cores
// truncate (round toward zero) when they add into the accumulator, which biases a long
// accumulator chain toward zero: the WMMA kernel starts every 16-wide K step's MMA from a fresh
// zero fragment and adds it to the running total with ordinary (round-to-nearest) F32 adds; the
// mma.sync kernel keeps its chains to 16 K steps or fewer the same way.
//
// The per-architecture prefill kernels live in sm80/paged_attn.cuh (paged_prefill_mma) and
// sm70/paged_attn.cuh (paged_prefill_m884) and sm75/paged_attn.cuh (paged_prefill_ffma), included below; `launch` picks one per device at run
// time. Decode and the WMMA prefill are portable and stay here, the tiered-pool pieces in
// dispatch.cu.
//
// Files (#955: one nvcc process per head size, so they build in parallel): this header (the
// kernels and their launch code, launch_d), d128.cu, d256.cu, d512.cu (launch_dim<D>, the
// kernels of that head size compiled there), d512_compact.cu (the compact cache's) and dispatch.cu
// (the extern "C" entry points, the prefill kernel choice and the tiered-pool pieces).
//
// The launch goes on the stream of ggml's CUDA backend (legion_cuda_stream in shim.cu),
// between the ggml graph segments before and after it, so no synchronization is needed.

#pragma once

#include "ggml-cuda/common.cuh"
#include "../device_cache.cuh"

#include <mma.h>

#include <cstdint>

extern "C" {

struct legion_paged_attn {
    const float * q;
    int64_t q_nb1, q_nb2, q_nb3;  // bytes
    const half * k;  // F16, or the bytes of a quantized cache (k_type)
    const half * v;
    int64_t kv_row;  // elements per cell: d * n_head_kv
    const int32_t * table;
    int32_t table_width;
    const int32_t * n_past;
    float * dst;
    float * scratch;  // decode with max_split > 1: [d + 2, max_split, n_head, n_seqs]; a
                      // multi-token group on sm_80+ with max_split > 1 (--paged-verify-split):
                      // [d + 2, max_split, n_head, n_tok * n_seqs] (sm80/paged_attn.cuh)
    int32_t max_split;
    int32_t d, n_head, n_head_kv, n_tok, n_seqs;
    float scale;
    int32_t window;  // 0: none
    int32_t ring;    // table entries in a ring, 0: none (see the header)
    int32_t k_type, v_type;  // LEGION_KV_F16 / _Q8_0 / _Q4_0
    int64_t k_cell, v_cell;  // bytes per cell (row) of K and V
    // Prefill over a pool with a host tier (#352): one piece of the key blocks, made by
    // legion_paged_attn_piece. All null / 0: every block through `table`.
    const int32_t * piece_blk;  // [table_width, n_seqs]: per sequence, the piece's blocks (positions / 64), ascending
    const int32_t * piece_ent;  // [table_width, n_seqs]: their pool blocks; e < 0: staging slot -1 - e
    const int32_t * piece_n;    // [n_seqs]: blocks per sequence
    const int32_t * piece_cnt;  // [1]: blocks staged
    const char * k_stage;       // staging slots, laid out as pool blocks
    const char * v_stage;
    float * ml;                 // [2, n_head, n_tok, n_seqs]: each row's running (max, sum); null: none
    int32_t piece;              // 0: write dst (and ml); > 0: merge into dst and ml
    // Non-null (prefill, n_tok > 1): only the last token's rows of each sequence are read
    // afterwards, and the WMMA kernel's result for them is computed split over the key blocks in
    // this scratch (sm70/paged_split.cuh; its size is paged::split_floats on the Rust side);
    // other rows of dst are left as they are.
    float * split;
    int32_t split_seqs;  // sequences the split scratch holds (> 0 with it)
    // Gemma 4's compact global KV cache (#551), read only by the kernels instantiated for it
    // (template parameter C): K and V are views of the same cells, kv heads head_stride
    // elements apart (kv_row = head_stride * n_head_kv), and a piece stages K's cells only
    // (v_stage points into k_stage at V's offset). 0 for every other cache.
    int32_t compact;
    int64_t head_stride;
    // > 0 with `split` (2 to 16 tokens per sequence): every row of the group comes split over
    // the key blocks instead of the last token's (sm70/paged_vsplit.cuh, #1211), the WMMA
    // kernel's bits, with the weights of this many key blocks at a time in the scratch (its size
    // is paged::wmma_split_plan on the Rust side).
    int32_t vsplit_chunk;
};

enum { LEGION_KV_F16 = 0, LEGION_KV_Q8_0 = 1, LEGION_KV_Q4_0 = 2 };

// Whether the current device's paged kernels read quantized caches (the sm_80+ prefill kernel;
// the WMMA one reads F16 only).
int legion_paged_attn_quant_ok(void);

// Prefill kernel choice: LEGION_PREFILL_AUTO (sm_80+: paged_prefill_mma; sm_70/75: the WMMA
// kernel), LEGION_PREFILL_M884 (the m8n8k4 one-pass kernel, F16 caches) or LEGION_PREFILL_FFMA
// (the FP32 FMA one-pass kernel, F16 caches, prefill launches without a split scratch). Set
// for all threads (`this_thread` 0), M884 applies on sm_70 devices only, FFMA on sm_75 ones
// only (the caller sets it for those without tensor cores), AUTO elsewhere (sm_75 lacks M884's
// shared memory, sm_80+ would run m8n8k4 emulated). Set for the calling thread (`this_thread`
// 1, tests), it applies on any GPU from sm_70. `this_thread` 2 (tests of the all-threads
// choice): the calling thread reads `kernel` in place of the all-threads choice, with its
// per-architecture limits, and the other threads keep theirs (-1 ends it).
enum { LEGION_PREFILL_AUTO = 0, LEGION_PREFILL_M884 = 1, LEGION_PREFILL_FFMA = 2 };
void legion_paged_attn_set_prefill(int kernel, int this_thread);

// Prefill launches of the m8n8k4 kernel from the calling thread so far (tests).
int64_t legion_paged_attn_m884_launches(void);

// Prefill launches of the FP32 FMA kernel from the calling thread so far (tests).
int64_t legion_paged_attn_ffma_launches(void);

// Launches of the WMMA kernel's all-row split (vsplit, #1211) from the calling thread so far.
int64_t legion_paged_attn_vsplit_launches(void);

// Whether prefill groups on compute capability 12.x take the queries-as-M kernel
// (sm80/paged_attn_qm.cuh, the same bits as paged_prefill_mma) where it covers them: `on` 1 (the
// default) or 0, for all threads (`this_thread` 0) or for the calling thread only (`this_thread`
// 1, tests and benches; -1 ends it).
void legion_paged_attn_set_qm(int on, int this_thread);

// Launches of the queries-as-M kernel from the calling thread so far.
int64_t legion_paged_attn_qm_launches(void);

// Whether prefill launches on the current device take the WMMA kernel, which a split scratch
// (legion_paged_attn::split) replaces for the last token's rows: sm_70 / sm_75 without the
// m8n8k4 opt-in.
int legion_paged_attn_split_ok(void);

// 0 on success; 1 for an unsupported shape (head size, GQA ratio); 2 for a CUDA launch error.
int legion_paged_attn_launch(const legion_paged_attn * p, void * stream);

// Piece `piece` of a prefill over a pool whose blocks from `hbm_blocks` on are in host memory
// (#352): the device blocks (piece 0) and the host blocks `piece * cap ..` (in the order of the
// sequences and their positions) up to `cap` of them, the host ones copied once into p's
// staging slots. Fills p's piece_* arrays (which must point to device memory for them) and
// `src` ([cap] ints of scratch), on `stream`. 0 on success, 2 for a CUDA launch error.
int legion_paged_attn_piece(const legion_paged_attn * p, int32_t hbm_blocks, int32_t cap, int32_t piece,
                            int32_t * src, void * stream);

}  // extern "C"

// The prefill kernel choice and the m8n8k4 and FFMA launch counts (legion_paged_attn_set_prefill),
// one copy for every translation unit: defined in dispatch.cu.
namespace legion_paged {
extern int g_prefill;
extern thread_local int t_prefill;
extern thread_local int t_prefill_all;
extern thread_local int64_t t_m884_launches;
extern thread_local int64_t t_ffma_launches;
extern thread_local int64_t t_vsplit_launches;
extern int g_qm;
extern thread_local int t_qm;
extern thread_local int64_t t_qm_launches;
}  // namespace legion_paged

namespace {

constexpr int BLK = 64;  // positions per KV block (KV_BLOCK on the Rust side)

using legion_paged::g_prefill;
using legion_paged::t_prefill;
using legion_paged::t_prefill_all;
using legion_paged::t_m884_launches;
using legion_paged::t_ffma_launches;
using legion_paged::t_vsplit_launches;
using legion_paged::t_qm_launches;

// Whether the queries-as-M prefill kernel is on for the calling thread (legion_paged_attn_set_qm).
bool qm_on() {
    return legion_paged::t_qm >= 0 ? legion_paged::t_qm != 0 : __atomic_load_n(&legion_paged::g_qm, __ATOMIC_RELAXED) != 0;
}

// Whether the current device is sm_70 (cached per device).
bool device_is_sm70() {
    return legion_device::compute_capability() == 70;
}

// The all-threads prefill choice as the calling thread reads it (see
// legion_paged_attn_set_prefill).
int prefill_for_all() {
    return t_prefill_all >= 0 ? t_prefill_all : __atomic_load_n(&g_prefill, __ATOMIC_RELAXED);
}

// Whether prefill launches take the m8n8k4 kernel (see legion_paged_attn_set_prefill).
bool use_m884() {
    if (t_prefill >= 0) {
        return t_prefill == LEGION_PREFILL_M884;
    }
    return prefill_for_all() == LEGION_PREFILL_M884 && device_is_sm70();
}

// Whether prefill launches take the FP32 FMA kernel (see legion_paged_attn_set_prefill).
bool use_ffma() {
    if (t_prefill >= 0) {
        return t_prefill == LEGION_PREFILL_FFMA;
    }
    return prefill_for_all() == LEGION_PREFILL_FFMA && legion_device::compute_capability() == 75;
}

// The first cell of the block holding positions b * BLK .. of a sequence whose table is `tab`.
__device__ __forceinline__ int64_t block_cell(const int32_t * tab, int b, int ring) {
    return (int64_t) tab[ring > 0 ? b % ring : b] * BLK;
}

// The key blocks a prefill CTA walks: blocks [b_first, b_last] of its sequence, or (a piece
// of a tiered pool) those of them in the piece. Walk entries i0 .. i1 - 1.
struct Walk {
    const int32_t * blk;  // null: entry i is block i
    const int32_t * ent;  // the sequence's table, or the piece's entries
    int ring, i0, i1;

    __device__ __forceinline__ int block(int i) const { return blk ? blk[i] : i; }

    // The first byte of entry i's block in `base` (a pool), or in `stage` for a staged one.
    __device__ __forceinline__ const char * cells(const void * base, const char * stage, int i, int64_t cell_bytes) const {
        const int e = blk ? ent[i] : ent[ring > 0 ? i % ring : i];
        return e >= 0 ? (const char *) base + (int64_t) e * BLK * cell_bytes
                      : stage + (int64_t) (-1 - e) * BLK * cell_bytes;
    }
};

// The first of the n ascending values at v that is >= x (n if none).
__device__ __forceinline__ int lower_bound(const int32_t * v, int n, int x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (v[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ Walk walk(const legion_paged_attn & p, int seq, int b_first, int b_last) {
    const int64_t row = (int64_t) seq * p.table_width;
    if (!p.piece_blk) {
        return {nullptr, p.table + row, p.ring, b_first, b_last + 1};
    }
    const int32_t * blk = p.piece_blk + row;
    const int n = p.piece_n[seq];
    return {blk, p.piece_ent + row, 0, lower_bound(blk, n, b_first), lower_bound(blk, n, b_last + 1)};
}

// Whether a prefill CTA of a later piece has nothing to add (its rows keep their output).
__device__ __forceinline__ bool piece_idle(const legion_paged_attn & p, const Walk & w) {
    return p.piece > 0 && (*p.piece_cnt == 0 || w.i0 >= w.i1);
}

// The first element of kv head kvh in a cell: kvh * D, or in Gemma 4's compact cache (C, #551)
// kvh * p.head_stride. Only the C instantiations read p.head_stride.
template <int D, bool C>
__device__ __forceinline__ int head_at(const legion_paged_attn & p, int kvh) {
    if constexpr (C) {
        return (int) (kvh * p.head_stride);
    } else {
        return kvh * D;
    }
}

// Merging a row into the earlier pieces' result (#352): with this launch's (m, l) (natural or
// log2 units, LOG2) and the earlier pieces' (m0, l0) in p.ml at `row`: (fa, fb, m', l'), the
// weights of the old and this launch's normalized outputs and the row's new running (max, sum).
// Piece 0: (0, 1, m, l).
template <bool LOG2>
__device__ __forceinline__ float4 merge_row(const legion_paged_attn & p, int64_t row, float m, float l) {
    if (p.piece == 0) {
        return make_float4(0.0f, 1.0f, m, l);
    }
    const int64_t rows = (int64_t) p.n_head * p.n_tok * p.n_seqs;
    const float m0 = p.ml[row], l0 = p.ml[rows + row];
    const float mx = fmaxf(m0, m);
    if (mx == -INFINITY) {
        return make_float4(1.0f, 0.0f, m0, l0);
    }
    const float a = l0 * (LOG2 ? exp2f(m0 - mx) : expf(m0 - mx));
    const float b = l * (LOG2 ? exp2f(m - mx) : expf(m - mx));
    return make_float4(a / (a + b), b / (a + b), mx, a + b);
}

__device__ __forceinline__ void store_ml(const legion_paged_attn & p, int64_t row, float4 f) {
    const int64_t rows = (int64_t) p.n_head * p.n_tok * p.n_seqs;
    p.ml[row] = f.z;
    p.ml[rows + row] = f.w;
}

template <int N>
__device__ __forceinline__ void load_halves(const half * src, float * out);

// Quantized caches (ggml's block formats, 32 values per block): q8_0 = F16 scale + 32 int8;
// q4_0 = F16 scale + 16 bytes of nibbles (values 0..15 low, 16..31 high, minus 8).
constexpr int QK = 32;

// Values e .. e + N - 1 (N a power of two up to 8 or 16, within one block for N <= 32 and
// aligned to N) of the row at `row` (bytes) of a cache of type `ty`, as floats; F16 rows read
// `row` as halves.
template <int N>
__device__ __forceinline__ void load_row(const char * row, int ty, int e, float * out);

template <int N>
__device__ __forceinline__ void load_quant(const char * row, int ty, int e, float * out) {
    if (ty == LEGION_KV_Q8_0) {
        const char * blk = row + (e / QK) * 34;
        const float d = __half2float(*(const half *) blk);
        const int8_t * q = (const int8_t *) (blk + 2) + e % QK;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = d * q[i];
        }
    } else {
        const char * blk = row + (e / QK) * 18;
        const float d = __half2float(*(const half *) blk);
        const int i0 = e % QK;
        const uint8_t * q = (const uint8_t *) (blk + 2) + i0 % 16;
        const int shift = i0 < 16 ? 0 : 4;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = d * (float) ((int) ((q[i] >> shift) & 15) - 8);
        }
    }
}

template <int N>
__device__ __forceinline__ void load_row(const char * row, int ty, int e, float * out) {
    if (ty == LEGION_KV_F16) {
        load_halves<N>((const half *) row + e, out);
    } else if constexpr (N > 16) {
        load_quant<16>(row, ty, e, out);
        load_quant<16>(row, ty, e + 16, out + 16);
    } else {
        load_quant<N>(row, ty, e, out);
    }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, o);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
    }
    return x;
}

// N consecutive halves (N * 2 bytes, aligned to that) as floats.
template <int N>
__device__ __forceinline__ void load_halves(const half * src, float * out) {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16, "unsupported vector width");
    if constexpr (N == 8) {
        const uint4 raw = *(const uint4 *) src;
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 f = __half22float2(h[i]);
            out[2 * i]     = f.x;
            out[2 * i + 1] = f.y;
        }
    } else if constexpr (N == 16) {
        load_halves<8>(src, out);
        load_halves<8>(src + 8, out + 8);
    } else if constexpr (N == 4) {
        const uint2 raw = *(const uint2 *) src;
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const float2 f = __half22float2(h[i]);
            out[2 * i]     = f.x;
            out[2 * i + 1] = f.y;
        }
    } else if constexpr (N == 2) {
        const float2 f = __half22float2(*(const half2 *) src);
        out[0] = f.x;
        out[1] = f.y;
    } else {
        out[0] = __half2float(*src);
    }
}

// Splits of a sequence with `nblk` blocks, and the blocks of split `s`.
__device__ __forceinline__ int n_splits(int max_split, int nblk) {
    return min(max_split, nblk);
}

// The first key position a query at `pos` sees.
__device__ __forceinline__ int first_key(int pos, int window) {
    return window > 0 ? max(0, pos - window + 1) : 0;
}

// Decode: the blocks [*b_first, *b_end) holding the keys of a sequence of `len` positions.
__device__ __forceinline__ void decode_blocks(int len, int window, int * b_first, int * b_end) {
    *b_first = first_key(len - 1, window) / BLK;
    *b_end   = (len + BLK - 1) / BLK;
}

template <int D, int G, bool C = false>
__global__ void __launch_bounds__(128) paged_decode(const legion_paged_attn p) {
    constexpr int NW = 4;               // warps
    constexpr int E  = D / 32;          // K elements per lane
    constexpr int VE = D / (32 * NW);   // V elements (output dims) per thread
    const int kvh = blockIdx.x, seq = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    const int len   = p.n_past[seq] + 1;
    const int first = first_key(len - 1, p.window);
    int bf, be;
    decode_blocks(len, p.window, &bf, &be);
    const int nblk = be - bf;
    const int ns   = n_splits(p.max_split, nblk);
    if (split >= ns) {
        return;
    }
    const int b0 = bf + split * nblk / ns, b1 = bf + (split + 1) * nblk / ns;

    __shared__ float s[G][BLK];
    __shared__ float alpha[G], stat_m[G], stat_l[G];

    float qr[G][E];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const float * qh = (const float *) ((const char *) p.q + seq * p.q_nb3 + (kvh * G + g) * p.q_nb1);
#pragma unroll
        for (int e = 0; e < E; e++) {
            qr[g][e] = qh[lane * E + e] * p.scale;
        }
    }
    if (threadIdx.x < G) {
        stat_m[threadIdx.x] = -INFINITY;
        stat_l[threadIdx.x] = 0.0f;
    }
    float acc[G][VE];
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
        for (int e = 0; e < VE; e++) {
            acc[g][e] = 0.0f;
        }
    }
    const int32_t * tab = p.table + (int64_t) seq * p.table_width;
    __syncthreads();

    for (int b = b0; b < b1; b++) {
        const int64_t cell0 = block_cell(tab, b, p.ring);
        const int npos = min(BLK, len - b * BLK);
        // Positions before the window's first key (in its first block) weigh nothing.
        const int skip = max(0, first - b * BLK);

        // Scores: warp w takes positions w, w + NW, ...
        for (int j = warp; j < BLK; j += NW) {
            float dot[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                dot[g] = 0.0f;
            }
            if (j < npos && j >= skip) {
                float kf[E];
                load_row<E>((const char *) p.k + (cell0 + j) * p.k_cell, p.k_type, head_at<D, C>(p, kvh) + lane * E, kf);
#pragma unroll
                for (int e = 0; e < E; e++) {
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        dot[g] += qr[g][e] * kf[e];
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                dot[g] = warp_sum(dot[g]);
            }
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < G; g++) {
                    s[g][j] = j < npos && j >= skip ? dot[g] : -INFINITY;
                }
            }
        }
        __syncthreads();

        // Online softmax: new max, rescale factor, running sum; scores become weights.
        for (int g = warp; g < G; g += NW) {
            const float a = s[g][lane], c = s[g][lane + 32];
            const float m_old = stat_m[g];
            const float m_new = fmaxf(m_old, warp_max(fmaxf(a, c)));
            const float ea = expf(a - m_new), ec = expf(c - m_new);
            s[g][lane]      = ea;
            s[g][lane + 32] = ec;
            const float sum = warp_sum(ea + ec);
            if (lane == 0) {
                const float al = expf(m_old - m_new);
                alpha[g]  = al;
                stat_l[g] = stat_l[g] * al + sum;
                stat_m[g] = m_new;
            }
        }
        __syncthreads();

        float al[G];
#pragma unroll
        for (int g = 0; g < G; g++) {
            al[g] = alpha[g];
#pragma unroll
            for (int e = 0; e < VE; e++) {
                acc[g][e] *= al[g];
            }
        }
        const char * vb = (const char *) p.v + cell0 * p.v_cell;
        const int ve0 = head_at<D, C>(p, kvh) + threadIdx.x * VE;
        for (int j = skip; j < npos; j++) {
            float vf[VE];
            load_row<VE>(vb + j * p.v_cell, p.v_type, ve0, vf);
#pragma unroll
            for (int g = 0; g < G; g++) {
                const float w = s[g][j];
#pragma unroll
                for (int e = 0; e < VE; e++) {
                    acc[g][e] += w * vf[e];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < G; g++) {
        const int h = kvh * G + g;
        if (ns == 1) {
            const float inv = 1.0f / stat_l[g];
            float * o = p.dst + ((int64_t) seq * p.n_head + h) * D + threadIdx.x * VE;
#pragma unroll
            for (int e = 0; e < VE; e++) {
                o[e] = acc[g][e] * inv;
            }
        } else {
            float * part = p.scratch + (((int64_t) seq * p.n_head + h) * p.max_split + split) * (D + 2);
            if (threadIdx.x == 0) {
                part[0] = stat_m[g];
                part[1] = stat_l[g];
            }
#pragma unroll
            for (int e = 0; e < VE; e++) {
                part[2 + threadIdx.x * VE + e] = acc[g][e];
            }
        }
    }
}

// Merges the decode splits of (head, sequence); one thread per output dim.
template <int D>
__global__ void __launch_bounds__(D) paged_combine(const legion_paged_attn p) {
    const int h = blockIdx.x, seq = blockIdx.y;
    const int len  = p.n_past[seq] + 1;
    int bf, be;
    decode_blocks(len, p.window, &bf, &be);
    const int ns   = n_splits(p.max_split, be - bf);
    if (ns == 1) {
        return;  // written by the decode kernel
    }
    const float * part = p.scratch + ((int64_t) seq * p.n_head + h) * p.max_split * (D + 2);
    float m = -INFINITY;
    for (int s = 0; s < ns; s++) {
        m = fmaxf(m, part[s * (D + 2)]);
    }
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < ns; s++) {
        const float w = expf(part[s * (D + 2)] - m);
        l += w * part[s * (D + 2) + 1];
        o += w * part[s * (D + 2) + 2 + threadIdx.x];
    }
    p.dst[((int64_t) seq * p.n_head + h) * D + threadIdx.x] = o / l;
}

using namespace nvcuda;

using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// Whether the tensor cores of the arch being compiled truncate their accumulation (sm_70,
// sm_75, sm_12x): see the header comment.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800 || __CUDA_ARCH__ >= 1200)
constexpr bool TC_TRUNCATES = true;
#else
constexpr bool TC_TRUNCATES = false;
#endif

// acc += a * b. Where the tensor cores truncate, the product goes into a fresh fragment and is
// added to `acc` in F32 registers (accumulator fragments of one shape share their layout), so
// the truncation is confined to one 16-wide K step instead of compounding over the whole chain.
template <typename FA, typename FB>
__device__ __forceinline__ void mma_acc(AccFrag & acc, const FA & a, const FB & b) {
    if constexpr (TC_TRUNCATES) {
        AccFrag step;
        wmma::fill_fragment(step, 0.0f);
        wmma::mma_sync(step, a, b, step);
#pragma unroll
        for (int i = 0; i < step.num_elements; i++) {
            acc.x[i] += step.x[i];
        }
    } else {
        wmma::mma_sync(acc, a, b, acc);
    }
}

// ROWS query rows a CTA: 64, or 32 where 64 rows' tiles do not fit the device's shared memory
// (head size 512 on Turing's 64 KB a block: 93,184 bytes against 46,592).
template <int D, int ROWS = 64>
struct PrefillSmem {
    static constexpr int QLD = D + 8;    // halves
    static constexpr int SLD = BLK + 4;  // floats
    static constexpr int PLD = BLK + 8;  // halves
    static constexpr size_t Q    = 0;
    static constexpr size_t S    = Q + ROWS * QLD * sizeof(half);
    static constexpr size_t P    = S + ROWS * SLD * sizeof(float);
    static constexpr size_t SIZE = P + ROWS * PLD * sizeof(half);
};

// ROWS (64 or 32) query rows of one kv head a CTA, 8 warps: RT row tiles of 16 rows, each split
// over 8 / RT warps by key columns (S) and output dims (P V). Every 16 x 16 tile of S and of
// the output sums the same products in the same order whatever warp computes it, and the
// softmax keeps four threads a row (the threads past ROWS rows idle there), so 32 rows give
// the 64-row kernel's bits: a CTA's extra masked key blocks add exact zeros.
template <int D, int G, bool C = false, int ROWS = 64>
__global__ void __launch_bounds__(256) paged_prefill(const legion_paged_attn p) {
    static_assert(ROWS == 64 || ROWS == 32, "64 or 32 query rows a CTA");
    using L = PrefillSmem<D, ROWS>;
    constexpr int QT = ROWS / G;     // tokens per CTA
    constexpr int RT = ROWS / 16;    // row tiles
    constexpr int CW = 8 / RT;       // warps a row tile
    constexpr int SW = BLK / CW;     // key columns a warp (S)
    constexpr int DH = D / CW;       // output dims a warp
    extern __shared__ __align__(128) unsigned char smem[];
    half  * qs = (half *) (smem + L::Q);
    float * ss = (float *) (smem + L::S);
    half  * ps = (half *) (smem + L::P);

    const int t0 = blockIdx.x * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int rw = warp % RT;  // row tile: rows 16 * rw ..
    const int dh = warp / RT;  // part: key columns for S, output dims for P V
    const int n_past = p.n_past[seq];
    const int t_last = min(t0 + QT, p.n_tok) - 1;
    const int nblk = (n_past + t_last) / BLK + 1;
    // The tile's first query sees the earliest keys; blocks before them are skipped.
    const int b_first = first_key(n_past + t0, p.window) / BLK;
    const Walk wk = walk(p, seq, b_first, nblk - 1);
    if (piece_idle(p, wk)) {
        return;
    }

    // Q rows: r = token_in_tile * G + head_in_group, as F16; rows past n_tok are zero, and so
    // are the rows past QT * G when G does not divide ROWS (e.g. 12: 5 tokens, 60 rows).
    for (int i = threadIdx.x; i < ROWS * D; i += blockDim.x) {
        const int r = i / D, c = i % D;
        const int t = t0 + r / G;
        float x = 0.0f;
        if (r < QT * G && t < p.n_tok) {
            x = *(const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                  (kvh * G + r % G) * p.q_nb1 + c * sizeof(float));
        }
        qs[r * L::QLD + c] = __float2half(x);
    }
    __syncthreads();

    // S = Q K^T for walk entry i: this warp's 16 rows x SW key columns, into ss.
    auto scores = [&](int i) {
        const half * kblk = (const half *) wk.cells(p.k, p.k_stage, i, p.k_cell) + head_at<D, C>(p, kvh);
        AccFrag acc[SW / 16];
#pragma unroll
        for (int n = 0; n < SW / 16; n++) {
            wmma::fill_fragment(acc[n], 0.0f);
        }
#pragma unroll 4
        for (int k = 0; k < D / 16; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, qs + rw * 16 * L::QLD + k * 16, L::QLD);
#pragma unroll
            for (int n = 0; n < SW / 16; n++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
                const int col = dh * SW + n * 16;
                wmma::load_matrix_sync(kb, kblk + (int64_t) col * p.kv_row + k * 16, p.kv_row);
                mma_acc(acc[n], a, kb);
            }
        }
#pragma unroll
        for (int n = 0; n < SW / 16; n++) {
            wmma::store_matrix_sync(ss + rw * 16 * L::SLD + dh * SW + n * 16, acc[n], L::SLD,
                                    wmma::mem_row_major);
        }
    };

    // Four threads per row, 16 columns each; with 32 rows, warps 4-7 hold no row (whole warps,
    // so the rows' shuffles stay within active warps).
    const int r = threadIdx.x / 4, part = threadIdx.x % 4;
    const bool row_on = ROWS == 64 || r < ROWS;
    const int limit = n_past + t0 + r / G;  // last visible key of row r
    const int lo    = first_key(limit, p.window);  // first visible key of row r

    // Pass 1: each row's max and sum of exp.
    float m = -INFINITY, l = 0.0f;
    for (int i = wk.i0; i < wk.i1; i++) {
        const int b = wk.block(i);
        scores(i);
        __syncthreads();
        if (row_on) {
            float x[16];
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int c = part * 16 + i;
                const int k = b * BLK + c;
                x[i] = k <= limit && k >= lo ? ss[r * L::SLD + c] * p.scale : -INFINITY;
                mx = fmaxf(mx, x[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            const float m_new = fmaxf(m, mx);
            // A row with no visible key yet keeps m = -inf (with a sliding window, a row whose
            // window starts in a later block); exp against 0 then gives 0 instead of NaN.
            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                sum += expf(x[i] - m_use);
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            l = l * expf(m - m_use) + sum;
            m = m_new;
        }
        __syncthreads();
    }
    const float inv_l = 1.0f / l;

    // Pass 2: O += softmax(S) V; this warp's 16 rows x DH output dims.
    AccFrag o[DH / 16];
#pragma unroll
    for (int n = 0; n < DH / 16; n++) {
        wmma::fill_fragment(o[n], 0.0f);
    }
    for (int i = wk.i0; i < wk.i1; i++) {
        const int b = wk.block(i);
        const half * vblk = (const half *) wk.cells(p.v, p.v_stage, i, p.v_cell) + head_at<D, C>(p, kvh);
        scores(i);
        __syncthreads();
        if (row_on) {
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int c = part * 16 + i;
                const int k = b * BLK + c;
                const float w = k <= limit && k >= lo ? expf(ss[r * L::SLD + c] * p.scale - m) * inv_l : 0.0f;
                ps[r * L::PLD + c] = __float2half(w);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BLK / 16; kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, ps + rw * 16 * L::PLD + kk * 16, L::PLD);
#pragma unroll
            for (int n = 0; n < DH / 16; n++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, vblk + (int64_t) (kk * 16) * p.kv_row + dh * DH + n * 16, p.kv_row);
                mma_acc(o[n], a, vb);
            }
        }
    }
    __syncthreads();

    // Pieces (#352): each row's weights of the earlier pieces' output and this one's (in the
    // P buffer, free now), from the row's thread of part 0.
    float2 * mf = (float2 *) ps;
    if (p.ml) {
        if (part == 0 && row_on) {
            const int t = t0 + r / G;
            if (r < QT * G && t < p.n_tok) {
                const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + r % G;
                const float4 f = merge_row<false>(p, row, m, l);
                store_ml(p, row, f);
                mf[r] = make_float2(f.x, f.y);
            }
        }
        __syncthreads();
    }

    // Out through a 16x16 staging tile per warp (in the score buffer, free now).
    float * stage = ss + warp * 256;
#pragma unroll
    for (int n = 0; n < DH / 16; n++) {
        wmma::store_matrix_sync(stage, o[n], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            const int rr = rw * 16 + i / 16;
            const int t = t0 + rr / G;
            if (rr < QT * G && t < p.n_tok) {
                const int h = kvh * G + rr % G;
                float * out = p.dst + (((int64_t) seq * p.n_tok + t) * p.n_head + h) * D + dh * DH + n * 16 + i % 16;
                if (p.ml && p.piece > 0) {
                    *out = mf[rr].x * *out + mf[rr].y * stage[i];
                } else {
                    *out = stage[i];
                }
            }
        }
        __syncwarp();
    }
}

// Prefill on sm_80+: mma.sync m16n8k16 + cp.async, namespace fa.
#include "../sm80/paged_attn.cuh"

// Prefill on compute capability 12.x, queries as the MMA's M: namespace fa_qm (#1133).
#include "../sm80/paged_attn_qm.cuh"

// Prefill on sm_70 (Volta): mma.sync m8n8k4, namespace v70.
#include "../sm70/paged_attn.cuh"

// Prefill on Turing without tensor cores: FP32 FMAs, namespace tu75 (#1214).
#include "../sm75/paged_attn.cuh"

// The WMMA kernel's last-token rows split over the key blocks, namespace split (#436).
#include "../sm70/paged_split.cuh"

// The WMMA kernel's rows of a 2-16-token group split over the key blocks, namespace vsplit (#1211).
#include "../sm70/paged_vsplit.cuh"

template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    if (p.n_tok == 1) {
        paged_decode<D, G, C><<<dim3(p.n_head_kv, p.n_seqs, p.max_split), 128, 0, st>>>(p);
        if (p.max_split > 1) {
            paged_combine<D><<<dim3(p.n_head, p.n_seqs), D, 0, st>>>(p);
        }
    } else if (const int rc = use_m884() && p.k_type == LEGION_KV_F16 && p.v_type == LEGION_KV_F16
                                  ? v70::launch<D, G, C>(p, st)
                                  : -1;
               rc >= 0) {
        return rc;
    } else if (const int rc = use_ffma() && p.k_type == LEGION_KV_F16 && p.v_type == LEGION_KV_F16 && !p.split
                                  ? tu75::launch<D, G, C>(p, st)
                                  : -1;
               rc >= 0) {
        return rc;  // -1: Q's rows unaligned, the default kernels instead
    } else if (fa::available()) {
        // A split scratch (--paged-verify-split): small groups split over the key blocks.
        if (p.max_split > 1 && !p.piece_blk && !p.ml) {
            return fa::launch_split<D, G, C>(p, st);
        }
        // Heads of 256 on compute capability 12.x: queries as M where it covers the op, the same
        // bits (-1: no shared memory for it, paged_prefill_mma instead).
        if constexpr (D == 256 && !C) {
            if (fa_qm::takes<D, G>(p)) {
                if (const int rc = fa_qm::launch<D, G>(p, st); rc >= 0) {
                    return rc;
                }
            }
        }
        return fa::launch<D, G, C>(p, st);
    } else if (p.k_type != LEGION_KV_F16 || p.v_type != LEGION_KV_F16) {
        return 1;  // the WMMA kernel reads F16 only (legion_paged_attn_quant_ok)
    } else if (p.split && p.split_seqs > 0 && !p.piece_blk && p.vsplit_chunk == 0) {
        return split::launch<D, G, C>(p, st);
    } else if (const int rc = p.split && p.split_seqs > 0 && !p.piece_blk && !p.ml && p.vsplit_chunk > 0
                                  ? vsplit::launch<D, G, C>(p, st)
                                  : -1;
               rc >= 0) {
        return rc;  // -1: the device lacks the kernels' shared memory, the WMMA kernel instead
    } else {
        constexpr size_t smem = PrefillSmem<D>::SIZE;
        // Per device, once (the attribute is per function and device).
        static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
        if (!legion_device::smem_opt_in(opted_in, paged_prefill<D, G, C>, (int) smem)) {
            // 64 rows' tiles past the device's shared memory a block (head size 512 on Turing's
            // 64 KB): 32 rows a CTA, the same bits.
            if constexpr (smem > 64 * 1024) {
                constexpr size_t smem32 = PrefillSmem<D, 32>::SIZE;
                static std::atomic<int8_t> opted_in32[GGML_CUDA_MAX_DEVICES];
                if (!legion_device::smem_opt_in(opted_in32, paged_prefill<D, G, C, 32>, (int) smem32)) {
                    return 2;
                }
                constexpr int QT32 = 32 / G;
                paged_prefill<D, G, C, 32>
                    <<<dim3((p.n_tok + QT32 - 1) / QT32, p.n_head_kv, p.n_seqs), 256, smem32, st>>>(p);
                return cudaGetLastError() == cudaSuccess ? 0 : 2;
            } else {
                return 2;
            }
        }
        constexpr int QT = 64 / G;
        paged_prefill<D, G, C><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), 256, smem, st>>>(p);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int D>
int launch_d(const legion_paged_attn & p, cudaStream_t st) {
    switch (p.n_head / p.n_head_kv) {
        case 1: return launch<D, 1>(p, st);
        case 2: return launch<D, 2>(p, st);
        case 4: return launch<D, 4>(p, st);
        case 6: return launch<D, 6>(p, st);  // Qwen3.5 / 3.6 / 3.8-27B: 24 query heads, 4 kv heads
        case 8: return launch<D, 8>(p, st);
        case 12: return launch<D, 12>(p, st);  // qwen4exp's MTP head: 24 query heads, 2 kv heads
        case 16: return launch<D, 16>(p, st);  // Nemotron-H: 32 query heads, 2 kv heads
        default: return 1;
    }
}

}  // namespace

namespace legion_paged {

// The kernels of head size D (launch_d), each in its own translation unit (d128.cu, d256.cu,
// d512.cu), so nvcc builds them in parallel.
template <int D>
int launch_dim(const legion_paged_attn & p, cudaStream_t st);

template <> int launch_dim<128>(const legion_paged_attn & p, cudaStream_t st);
template <> int launch_dim<256>(const legion_paged_attn & p, cudaStream_t st);
template <> int launch_dim<512>(const legion_paged_attn & p, cudaStream_t st);

// Gemma 4's compact global KV cache (#551): its global layers' 512-wide heads only, so the
// other caches' kernels stay as they are. In d512_compact.cu.
int launch_compact(const legion_paged_attn & p, cudaStream_t st);

}  // namespace legion_paged
