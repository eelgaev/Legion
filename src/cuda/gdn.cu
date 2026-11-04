// gdn: the gated delta rule of a gated DeltaNet layer (ggml's GGML_OP_GATED_DELTA_NET, scalar
// gate, S_v = 128) for the few tokens of an MTP verify window or a replay commit, with only the
// output the caller uses:
//
// - rows (verify): the tokens' attention rows; the final state is not written (the verify step
//   keeps the states, the commit replays the kept tokens);
// - state (commit): the final state, written in place over the input state; no token rows.
//
// The arithmetic per state column is ggml-cuda's autoregressive gated_delta_net_cuda<128, false,
// false> (gated_delta_net.cu) operation for operation: 8 lanes per column, 16 rows per lane, the
// same FMA chains and xor-shuffle reductions, expf of the gate and the 1 / sqrt(S_v) scale. So
// the rows and states are bit for bit ggml's (src/model/core/gdn/tests.rs compares them per
// token count). What differs: the token loop is unrolled for 1..8 tokens (the exact-T kernels
// of upstream Strata #783 PR-b by stuchapin909, MIT, via Strata-AC922), the next token's q, k,
// v, g and beta are loaded before this token's reductions, and the unused output is skipped (a
// verify step does not write 64 KiB of state per head and sequence; a commit does not compute
// the rows nor copy the state from the op's tail into the cache).

#include "device_cache.cuh"

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_gdn {
    const float * q;      // [128, neqk1, n_tokens, n_seqs / rq3]: (i, h, t, s) at i + h*sq1 + t*sq2 + s*sq3
    const float * k;      // as q (same strides)
    const float * v;      // [128, H, n_tokens, n_seqs]: (i, h, t, s) at i + h*sv1 + t*sv2 + s*sv3
    const float * g;      // [1, H, n_tokens, n_seqs]: (h, t, s) at h*sb1 + t*sb2 + s*sb3
    const float * beta;   // as g
    const float * state;  // [128, 128, H, n_seqs] contiguous, transposed (column c at c * 128)
    float * rows;         // [128 * H, n_tokens, n_seqs] contiguous, or null
    float * state_out;    // as state (may be state itself), or null
    int64_t H, n_tokens, n_seqs;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    int64_t neqk1, rq3;
    float scale;          // 1 / sqrt(128), computed as ggml does
    int32_t num_warps;    // warps per CTA (0: ggml's choice for this device)
    // Or null. With state_out and no rows: sequence s advances by its keep[s] tokens from token
    // start[s] on (a count word, see gdn_count; 0: left alone); n_tokens is then only the
    // pool's length (#836, one commit launch for every slot of a layer).
    const int32_t * keep;
    // Or null. With rows and keep (#836 part 14, a verify step that also runs the last commit):
    // sequence s first advances by its keep[s] stashed tokens (from row start[s] on), packed per
    // token as [q (128 * neqk1) | k (128 * neqk1) | v (128 * H) | g (H) | beta (H)] at replay +
    // s * rp2 + t * rp1, writes that state to state_out (unless it may leave it, gdn_leaves),
    // then computes its rows from it.
    float * replay;
    int64_t rp1, rp2;
    // With replay: the packed rows per sequence (the room a step that leaves its state has).
    int64_t rp_t;
    // With replay (#836 part 39): one zeroed counter per sequence and q / k head, [neqk1,
    // n_seqs]. After its replay the step stores its own tokens' k, v, g and beta over the packed
    // rows (the next step's replay reads them); the CTAs of a q / k head count themselves in
    // there to find the last of them, which stores the values they all read, and sets it back to
    // 0. q is not stored: no commit reads it.
    int32_t * arrive;
    // With replay: the kernel (#836; the same arithmetic and bits in each). 0: gdn_replay_rows_kernel
    // (each lane holds its tokens' q and k rows in registers; the engine's default); 1, or -1
    // (--gdn-replay-smem): gdn_replay_smem_kernel (the tokens' inputs in shared memory).
    int32_t replay_variant;
    // Or null. With replay (#836): a verify step's MoE row map, I32 [n_tokens * n_seqs]
    // (pad_rows::map): token t of sequence s is padding when t > 0 and pad[s * n_tokens + t] is
    // not s * n_tokens + t. A padding token's row gets zeros instead of its delta rule, and its
    // inputs are not stored over the packed rows: nothing reads either (the commit keeps
    // accepted tokens only, never past the real ones).
    const int32_t * pad;
};

// A commit's conv window copy for every slot at once: for each slot s with keep[s] > 0 and each
// row c, dst[s][c][0..w) = src[s][c][keep[s]..keep[s] + w) (strides in floats).
struct legion_spec_window {
    const float * src;
    float * dst;
    const int32_t * keep;
    int64_t w, rows, n;
    int64_t s1, s2, d1, d2;
};

int legion_gdn_launch(const legion_gdn * p, void * stream);
int legion_gdn_decode_launch(const legion_gdn * p, int variant, void * stream);
int legion_spec_window_launch(const legion_spec_window * p, void * stream);

}  // extern "C"

namespace {

constexpr int GDN_S = 128;
constexpr int GDN_COLS_PER_WARP = 4;
constexpr int GDN_LANES_PER_COL = 32 / GDN_COLS_PER_WARP;  // 8
constexpr int GDN_ROWS_PER_LANE = GDN_S / GDN_LANES_PER_COL;  // 16
constexpr int GDN_MAX_T = 8;
constexpr int GDN_MAX_LIVE = 1 << 30;  // gdn_steps' n_live: every token

// ggml-cuda's warp_reduce_sum<8> (common.cuh).
__device__ __forceinline__ float gdn_reduce(float x) {
#pragma unroll
    for (int offset = GDN_LANES_PER_COL / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, GDN_LANES_PER_COL);
    }
    return x;
}

// One token's inputs, as one lane holds them.
struct gdn_tok {
    float k[GDN_ROWS_PER_LANE];
    float q[GDN_ROWS_PER_LANE];
    float v, g, beta;
};

template <bool ROWS>
__device__ __forceinline__ void gdn_load(gdn_tok & x, const float * __restrict__ q_t,
                                         const float * __restrict__ k_t, const float * __restrict__ v_t,
                                         const float * __restrict__ g_t, const float * __restrict__ b_t,
                                         int lane_in_col, int col) {
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        const int i = r * GDN_LANES_PER_COL + lane_in_col;
        x.k[r] = k_t[i];
        if (ROWS) {
            x.q[r] = q_t[i];
        }
    }
    x.v = v_t[col];
    x.g = *g_t;
    x.beta = *b_t;
}

// Where a sequence's token inputs are read from: the op's own q, k, v, g and beta, or a deferred
// commit's packed replay rows (#836 part 14).
struct gdn_src {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
};

__device__ __forceinline__ gdn_src gdn_src_of(const legion_gdn & p) {
    return {p.q, p.k, p.v, p.g, p.beta, p.sq1, p.sq2, p.sq3, p.sv1, p.sv2, p.sv3, p.sb1, p.sb2, p.sb3};
}

// `in` from token `t` on.
__device__ __forceinline__ gdn_src gdn_src_from(gdn_src in, const int64_t t) {
    in.q += t * in.sq2;
    in.k += t * in.sq2;
    in.v += t * in.sv2;
    in.g += t * in.sb2;
    in.beta += t * in.sb2;
    return in;
}

// A deferred commit's count word (spec_pending, #836): bits 0-7 the kept tokens, 8-15 the row of
// the packed replay rows they start at, bit 16 set where the verify step may leave the state as it
// is (gdn_leaves). A plain count (an all-slot commit's) is a word of start 0.
constexpr int32_t GDN_COUNT_LEAVE = 1 << 16;

__device__ __forceinline__ void gdn_count(const int32_t w, int & keep, int & start) {
    keep = w & 0xff;
    start = (w >> 8) & 0xff;
}

// Whether a replaying verify step of n_tokens leaves sequence's state as it is (#836): its word
// allows it and the packed rows have room for this step's tokens after the kept ones. The step then
// replays the kept tokens in registers only and stores its own after them, so the next step
// replays both runs from the same state, with the same arithmetic per token: the same bits, one
// state write fewer. src/model/core/spec/pending.rs follows the same rule.
__device__ __forceinline__ bool gdn_leaves(const legion_gdn & p, const int32_t w, const int keep, const int start) {
    return (w & GDN_COUNT_LEAVE) != 0 && start + keep + p.n_tokens <= p.rp_t;
}

// The packed replay rows: per token [q | k | v | g | beta] (see legion_gdn::replay).
__device__ __forceinline__ gdn_src gdn_src_replay(const legion_gdn & p) {
    const int64_t qk = (int64_t) GDN_S * p.neqk1;
    const float * q = p.replay;
    const float * v = q + 2 * qk;
    const float * g = v + (int64_t) GDN_S * p.H;
    return {q, q + qk, v, g, g + p.H, GDN_S, p.rp1, p.rp2, GDN_S, p.rp1, p.rp2, 1, p.rp1, p.rp2};
}

// This lane's place: its row within its column, and the column.
__device__ __forceinline__ void gdn_lane(int & lane_in_col, int & col) {
    const int lane = threadIdx.x;
    const int col_in_warp = lane / GDN_LANES_PER_COL;
    lane_in_col = lane - col_in_warp * GDN_LANES_PER_COL;
    col = (blockIdx.z * blockDim.y + threadIdx.y) * GDN_COLS_PER_WARP + col_in_warp;
}

__device__ __forceinline__ int64_t gdn_state_off(const legion_gdn & p, const int64_t h_idx, const int64_t sequence,
                                                 const int col) {
    return (sequence * p.H + h_idx) * GDN_S * GDN_S + (int64_t) col * GDN_S;
}

__device__ __forceinline__ void gdn_state_load(float (&s)[GDN_ROWS_PER_LANE], const float * state, int64_t off,
                                               int lane_in_col) {
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        s[r] = state[off + r * GDN_LANES_PER_COL + lane_in_col];
    }
}

__device__ __forceinline__ void gdn_state_store(const float (&s)[GDN_ROWS_PER_LANE], float * state, int64_t off,
                                                int lane_in_col) {
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        state[off + r * GDN_LANES_PER_COL + lane_in_col] = s[r];
    }
}

// One lane's rows `s` of column `col` of head h_idx of sequence `sequence`, advanced over the
// first n_tokens tokens of `in` (ROWS: their attention rows to p.rows).
// T > 0: exactly T tokens (the loop unrolled); T == 0: n_tokens of them.
// With ROWS, only the first n_live tokens run (a verify step's real rows, #836): the rows of the
// others are zeros and `s` is left after the live ones; each live token's arithmetic is the same.
template <int T, bool ROWS>
__device__ __forceinline__ void gdn_steps(const legion_gdn & p, const gdn_src & in, float (&s)[GDN_ROWS_PER_LANE],
                                          const int64_t h_idx, const int64_t sequence, const int n_tokens,
                                          const int lane_in_col, const int col, const int n_live = GDN_MAX_LIVE) {
    const int live = ROWS && n_live < n_tokens ? n_live : n_tokens;
    const float * __restrict__ q = in.q;
    const float * __restrict__ k = in.k;
    const float * __restrict__ v = in.v;
    const float * __restrict__ g = in.g;
    const float * __restrict__ beta = in.beta;

    const int64_t iq1 = h_idx % p.neqk1;
    const int64_t iq3 = sequence / p.rq3;

    const float * qk_base = q + iq3 * in.sq3 + iq1 * in.sq1;
    const float * kk_base = k + iq3 * in.sq3 + iq1 * in.sq1;
    const float * v_base = v + sequence * in.sv3 + h_idx * in.sv1;
    const int64_t gb_base = sequence * in.sb3 + h_idx * in.sb1;
    float * rows = ROWS ? p.rows + (sequence * p.n_tokens * p.H + h_idx) * GDN_S : nullptr;

    gdn_tok cur;
    gdn_load<ROWS>(cur, qk_base, kk_base, v_base, g + gb_base, beta + gb_base, lane_in_col, col);

#pragma unroll
    for (int t = 0; t < (T > 0 ? T : GDN_MAX_T); t++) {
        if ((T == 0 && t >= n_tokens) || (ROWS && t >= live)) {
            break;
        }
        // The next token's inputs, in flight during this token's reductions.
        gdn_tok nxt;
        const bool more = t + 1 < live;
        if (more) {
            const int64_t tn = t + 1;
            gdn_load<ROWS>(nxt, qk_base + tn * in.sq2, kk_base + tn * in.sq2, v_base + tn * in.sv2,
                           g + gb_base + tn * in.sb2, beta + gb_base + tn * in.sb2, lane_in_col, col);
        }

        const float g_val = expf(cur.g);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
            kv_shard += s[r] * cur.k[r];
        }
        float kv_col = gdn_reduce(kv_shard);

        float delta_col = (cur.v - g_val * kv_col) * cur.beta;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
            s[r] = g_val * s[r] + cur.k[r] * delta_col;
            if (ROWS) {
                attn_partial += s[r] * cur.q[r];
            }
        }
        if (ROWS) {
            float attn_col = gdn_reduce(attn_partial);
            if (lane_in_col == 0) {
                rows[col] = attn_col * p.scale;
            }
            rows += GDN_S * p.H;
        }
        if (more) {
            cur = nxt;
        }
    }
    // Longer runs than the unrolled loop covers (T == 0 only).
    if (T == 0) {
        for (int t = GDN_MAX_T; t < live; t++) {
            const float g_val = expf(cur.g);
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
                kv_shard += s[r] * cur.k[r];
            }
            float kv_col = gdn_reduce(kv_shard);
            float delta_col = (cur.v - g_val * kv_col) * cur.beta;
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
                s[r] = g_val * s[r] + cur.k[r] * delta_col;
                if (ROWS) {
                    attn_partial += s[r] * cur.q[r];
                }
            }
            if (ROWS) {
                float attn_col = gdn_reduce(attn_partial);
                if (lane_in_col == 0) {
                    rows[col] = attn_col * p.scale;
                }
                rows += GDN_S * p.H;
            }
            if (t + 1 < live) {
                const int64_t tn = t + 1;
                gdn_load<ROWS>(cur, qk_base + tn * in.sq2, kk_base + tn * in.sq2, v_base + tn * in.sv2,
                               g + gb_base + tn * in.sb2, beta + gb_base + tn * in.sb2, lane_in_col, col);
            }
        }
    }
    // The padding tokens' rows (`rows` is past the live ones).
    if (ROWS && lane_in_col == 0) {
        for (int t = live; t < n_tokens; t++) {
            rows[col] = 0.0f;
            rows += GDN_S * p.H;
        }
    }
}

// One CTA's columns of head h_idx of sequence `sequence` over n_tokens tokens from token `start`
// on. T > 0: exactly T tokens (the loop unrolled); T == 0: n_tokens of them.
template <int T, bool ROWS, bool STATE>
__device__ __forceinline__ void gdn_run(const legion_gdn & p, const int64_t h_idx, const int64_t sequence,
                                        const int n_tokens, const int start = 0) {
    int lane_in_col, col;
    gdn_lane(lane_in_col, col);
    const int64_t state_off = gdn_state_off(p, h_idx, sequence, col);
    float s[GDN_ROWS_PER_LANE];
    gdn_state_load(s, p.state, state_off, lane_in_col);
    gdn_steps<T, ROWS>(p, gdn_src_from(gdn_src_of(p), start), s, h_idx, sequence, n_tokens, lane_in_col, col);
    if (STATE) {
        gdn_state_store(s, p.state_out, state_off, lane_in_col);
    }
}

// T > 0: exactly T tokens (the loop unrolled); T == 0: n_tokens of them.
template <int T, bool ROWS, bool STATE>
__global__ void __launch_bounds__(128, 2) gdn_kernel(const legion_gdn p) {
    gdn_run<T, ROWS, STATE>(p, blockIdx.x, blockIdx.y, T > 0 ? T : (int) p.n_tokens);
}

// A replaying verify step's real tokens of sequence `sequence` (#836, p.pad): all of them
// without a row map, else up to its first padding token (token 0 is always real).
__device__ __forceinline__ int gdn_live(const legion_gdn & p, const int64_t sequence) {
    const int nt = (int) p.n_tokens;
    if (p.pad == nullptr) {
        return nt;
    }
    const int64_t first = sequence * p.n_tokens;
    for (int t = 1; t < nt; t++) {
        if (p.pad[first + t] != (int32_t) (first + t)) {
            return t;
        }
    }
    return nt;
}

// A replaying step's own v over the packed rows from row `at` on (#836 part 39): this lane's
// column of head h_idx, its 8 lanes taking turns over the first `live` tokens (the real ones).
// Only this column's lanes read that column of the rows, all in this warp, and the caller has
// synchronized it after the replay.
__device__ __forceinline__ void gdn_stash_v(const legion_gdn & p, const int64_t h_idx, const int64_t sequence,
                                            const int lane_in_col, const int col, const int at, const int live) {
    const float * src = p.v + sequence * p.sv3 + h_idx * p.sv1 + col;
    float * dst = p.replay + sequence * p.rp2 + at * p.rp1 + 2 * GDN_S * p.neqk1 + h_idx * GDN_S + col;
    for (int64_t t = lane_in_col; t < live; t += GDN_LANES_PER_COL) {
        dst[t * p.rp1] = src[t * p.sv2];
    }
}

// A replaying step's own k, g and beta over the packed rows (#836 part 39): every CTA of the
// heads sharing q / k head j = h_idx % neqk1 of sequence `sequence` read that head's k and their
// heads' g and beta there in their replay, so the last of them to arrive (p.arrive) stores them
// for all, and sets the counter back to 0 for the next launch. The fences and barriers around
// the count order each CTA's replay reads before it, and the last CTA's stores after it (the
// last-block pattern of the CUDA threadFenceReduction sample). From row `at` on, the first `live`
// tokens, as gdn_stash_v.
__device__ __forceinline__ void gdn_stash_shared(const legion_gdn & p, const int64_t h_idx, const int64_t sequence,
                                                 const int chunks, const int at, const int live) {
    __shared__ int last;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    const int64_t j = h_idx % p.neqk1;
    __syncthreads();
    if (tid == 0) {
        int32_t * a = p.arrive + sequence * p.neqk1 + j;
        const int readers = (int) (p.H / p.neqk1) * chunks;
        __threadfence();
        last = atomicAdd(a, 1) == readers - 1;
        if (last) {
            *a = 0;
            __threadfence();
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    const int64_t qk = GDN_S * p.neqk1;
    for (int64_t t = 0; t < live; t++) {
        float * row = p.replay + sequence * p.rp2 + (at + t) * p.rp1;
        const float * k = p.k + sequence * p.sq3 + t * p.sq2 + j * p.sq1;
        for (int i = tid; i < GDN_S; i += threads) {
            row[qk + j * GDN_S + i] = k[i];
        }
        const int64_t gb = sequence * p.sb3 + t * p.sb2;
        float * g = row + 2 * qk + GDN_S * p.H;
        for (int64_t h = j + (int64_t) tid * p.neqk1; h < p.H; h += (int64_t) threads * p.neqk1) {
            g[h] = p.g[gb + h * p.sb1];
            g[p.H + h] = p.beta[gb + h * p.sb1];
        }
    }
}

// A verify step that runs the last commit first (#836 part 14, p.replay): sequence blockIdx.y
// advances by its keep[s] stashed tokens on the code gdn_keep_kernel runs for it (so its state is
// the same bits), writes that state over the slot, and computes this step's rows from it on the
// code gdn_kernel<T, true, false> runs (so the rows are the same bits as a separate commit and
// verify). The commit's state read and its second pass over the state are what this saves.
// Then it stores this step's inputs over the packed rows for the next one (part 39: copies, so
// the same bits as the separate store of part 38, without its launch and its reads). A sequence
// that may leave its state (gdn_leaves) does not write it and stores its inputs after the kept
// ones instead, for the next step to replay both.
template <int T>
__global__ void __launch_bounds__(128, 2) gdn_replay_rows_kernel(const legion_gdn p) {
    const int64_t h_idx = blockIdx.x;
    const int64_t sequence = blockIdx.y;
    const int32_t w = p.keep[sequence];
    int keep, start;
    gdn_count(w, keep, start);
    const bool leave = gdn_leaves(p, w, keep, start);
    int lane_in_col, col;
    gdn_lane(lane_in_col, col);
    const int64_t state_off = gdn_state_off(p, h_idx, sequence, col);
    float s[GDN_ROWS_PER_LANE];
    gdn_state_load(s, p.state, state_off, lane_in_col);
    if (keep > 0) {
        const gdn_src r = gdn_src_from(gdn_src_replay(p), start);
        switch (keep) {
            case 1: gdn_steps<1, false>(p, r, s, h_idx, sequence, 1, lane_in_col, col); break;
            case 2: gdn_steps<2, false>(p, r, s, h_idx, sequence, 2, lane_in_col, col); break;
            case 3: gdn_steps<3, false>(p, r, s, h_idx, sequence, 3, lane_in_col, col); break;
            case 4: gdn_steps<4, false>(p, r, s, h_idx, sequence, 4, lane_in_col, col); break;
            case 5: gdn_steps<5, false>(p, r, s, h_idx, sequence, 5, lane_in_col, col); break;
            case 6: gdn_steps<6, false>(p, r, s, h_idx, sequence, 6, lane_in_col, col); break;
            case 7: gdn_steps<7, false>(p, r, s, h_idx, sequence, 7, lane_in_col, col); break;
            case 8: gdn_steps<8, false>(p, r, s, h_idx, sequence, 8, lane_in_col, col); break;
            default: gdn_steps<0, false>(p, r, s, h_idx, sequence, keep, lane_in_col, col); break;
        }
        if (!leave) {
            gdn_state_store(s, p.state_out, state_off, lane_in_col);
        }
    }
    const int at = leave ? start + keep : 0;
    // Padding tokens (p.pad) get zero rows and are not stored.
    const int live = gdn_live(p, sequence);
    __syncwarp();
    gdn_stash_v(p, h_idx, sequence, lane_in_col, col, at, live);
    gdn_steps<T, true>(p, gdn_src_of(p), s, h_idx, sequence, T > 0 ? T : (int) p.n_tokens, lane_in_col, col, live);
    gdn_stash_shared(p, h_idx, sequence, (int) gridDim.z, at, live);
}

// A commit of every slot at once (p.keep): sequence blockIdx.y advances by its keep count (from
// its count word's start token on), on the code gdn_kernel<keep, false, true> runs for it alone
// (gdn_kernel<0, ...> past 8 tokens), so its states are the same bits; a slot keeping nothing
// exits.
__global__ void __launch_bounds__(128, 2) gdn_keep_kernel(const legion_gdn p) {
    int keep, st;
    gdn_count(p.keep[blockIdx.y], keep, st);
    switch (keep) {
        case 0: break;
        case 1: gdn_run<1, false, true>(p, blockIdx.x, blockIdx.y, 1, st); break;
        case 2: gdn_run<2, false, true>(p, blockIdx.x, blockIdx.y, 2, st); break;
        case 3: gdn_run<3, false, true>(p, blockIdx.x, blockIdx.y, 3, st); break;
        case 4: gdn_run<4, false, true>(p, blockIdx.x, blockIdx.y, 4, st); break;
        case 5: gdn_run<5, false, true>(p, blockIdx.x, blockIdx.y, 5, st); break;
        case 6: gdn_run<6, false, true>(p, blockIdx.x, blockIdx.y, 6, st); break;
        case 7: gdn_run<7, false, true>(p, blockIdx.x, blockIdx.y, 7, st); break;
        case 8: gdn_run<8, false, true>(p, blockIdx.x, blockIdx.y, 8, st); break;
        default: gdn_run<0, false, true>(p, blockIdx.x, blockIdx.y, keep, st); break;
    }
}

__global__ void spec_window_kernel(const legion_spec_window p) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.w * p.rows * p.n) {
        return;
    }
    const int64_t j = i % p.w;
    const int64_t r = (i / p.w) % p.rows;
    const int64_t s = i / (p.w * p.rows);
    const int64_t keep = p.keep[s];
    if (keep <= 0) {
        return;
    }
    p.dst[s * p.d2 + r * p.d1 + j] = p.src[s * p.s2 + r * p.s1 + keep + j];
}

cudaError_t gdn_replay_dispatch(const legion_gdn & p, dim3 grid, dim3 block, cudaStream_t st) {
    switch (p.n_tokens) {
        case 1: gdn_replay_rows_kernel<1><<<grid, block, 0, st>>>(p); break;
        case 2: gdn_replay_rows_kernel<2><<<grid, block, 0, st>>>(p); break;
        case 3: gdn_replay_rows_kernel<3><<<grid, block, 0, st>>>(p); break;
        case 4: gdn_replay_rows_kernel<4><<<grid, block, 0, st>>>(p); break;
        case 5: gdn_replay_rows_kernel<5><<<grid, block, 0, st>>>(p); break;
        case 6: gdn_replay_rows_kernel<6><<<grid, block, 0, st>>>(p); break;
        case 7: gdn_replay_rows_kernel<7><<<grid, block, 0, st>>>(p); break;
        case 8: gdn_replay_rows_kernel<8><<<grid, block, 0, st>>>(p); break;
        default: gdn_replay_rows_kernel<0><<<grid, block, 0, st>>>(p); break;
    }
    return cudaGetLastError();
}

// ---- The replaying verify step with the tokens' inputs in shared memory (#836).
//
// A replaying verify step at C=16 is close to bandwidth bound: every sequence's state (64 KiB a
// head) is read and, after the replay, written. gdn_replay_rows_kernel holds a token's q and k
// rows (16 + 16 a lane) and the next token's in registers, 128 registers a thread, so 4 CTAs (16
// warps) an SM keep too few state loads in flight. Here a CTA loads up to GDN_TOK_CHUNK tokens' k
// and q (shared by all its columns), its columns' v, g and beta into shared memory at once, and
// each lane reads its 16 rows' values there as it needs them, with the arithmetic of
// gdn_steps in the same order: the same bits, at most 102 registers (5 CTAs an SM). On a GB10
// (gdn-bench --replay) about 3% faster per layer at 1 and 4 tokens with every slot replaying;
// staging the state through shared memory too, 8 warps, a persistent double-buffered grid or a
// 64-register cap were measured and are not faster.

constexpr int GDN_TOK_CHUNK = 8;
constexpr int GDN_SMEM_WARPS = 4;
constexpr int GDN_SMEM_MIN_CTAS = 5;

template <int COLS>
struct gdn_smem_tok {
    float k[GDN_S];
    float q[GDN_S];
    float v[COLS];
    float g, beta;
};

// Tokens t0 .. t0 + n of `in` for head h_idx of sequence `sequence` into tok[0 .. n), the v of
// the CTA's columns col0 .. col0 + COLS (ROWS: q too). Every thread of the CTA takes part.
template <int COLS, int THREADS, bool ROWS>
__device__ __forceinline__ void gdn_fill(gdn_smem_tok<COLS> * tok, const legion_gdn & p, const gdn_src & in,
                                         const int64_t h_idx, const int64_t sequence, const int col0, const int t0,
                                         const int n, const int tid) {
    const int64_t iq1 = h_idx % p.neqk1;
    const int64_t iq3 = sequence / p.rq3;
    const float * q = in.q + iq3 * in.sq3 + iq1 * in.sq1 + t0 * in.sq2;
    const float * k = in.k + iq3 * in.sq3 + iq1 * in.sq1 + t0 * in.sq2;
    const float * v = in.v + sequence * in.sv3 + h_idx * in.sv1 + t0 * in.sv2 + col0;
    const int64_t gb = sequence * in.sb3 + h_idx * in.sb1 + t0 * in.sb2;
    for (int i = tid; i < n * GDN_S; i += THREADS) {
        const int t = i / GDN_S;
        const int c = i - t * GDN_S;
        tok[t].k[c] = k[t * in.sq2 + c];
        if (ROWS) {
            tok[t].q[c] = q[t * in.sq2 + c];
        }
    }
    for (int i = tid; i < n * COLS; i += THREADS) {
        const int t = i / COLS;
        const int c = i - t * COLS;
        tok[t].v[c] = v[t * in.sv2 + c];
    }
    if (tid < n) {
        tok[tid].g = in.g[gb + tid * in.sb2];
        tok[tid].beta = in.beta[gb + tid * in.sb2];
    }
}

// gdn_steps' token loop over tok[0 .. n): the same operations in the same order (ROWS: the
// attention rows to `rows`, which then points past them).
template <int COLS, bool ROWS>
__device__ __forceinline__ void gdn_steps_smem(const legion_gdn & p, const gdn_smem_tok<COLS> * tok, const int n,
                                               float (&s)[GDN_ROWS_PER_LANE], float *& rows, const int lane_in_col,
                                               const int ccol, const int col) {
    for (int t = 0; t < n; t++) {
        const gdn_smem_tok<COLS> & x = tok[t];
        const float g_val = expf(x.g);

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
            kv_shard += s[r] * x.k[r * GDN_LANES_PER_COL + lane_in_col];
        }
        float kv_col = gdn_reduce(kv_shard);

        float delta_col = (x.v[ccol] - g_val * kv_col) * x.beta;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
            s[r] = g_val * s[r] + x.k[r * GDN_LANES_PER_COL + lane_in_col] * delta_col;
            if (ROWS) {
                attn_partial += s[r] * x.q[r * GDN_LANES_PER_COL + lane_in_col];
            }
        }
        if (ROWS) {
            float attn_col = gdn_reduce(attn_partial);
            if (lane_in_col == 0) {
                rows[col] = attn_col * p.scale;
            }
            rows += GDN_S * p.H;
        }
    }
}

// gdn_replay_rows_kernel's work with the tokens' inputs through shared memory: one unit
// (sequence, head, chunk of COLS columns) per CTA in a chunk-first grid.
__global__ void __launch_bounds__(GDN_SMEM_WARPS * 32, GDN_SMEM_MIN_CTAS) gdn_replay_smem_kernel(const legion_gdn p) {
    constexpr int COLS = GDN_SMEM_WARPS * GDN_COLS_PER_WARP;
    constexpr int CHUNKS = GDN_S / COLS;
    constexpr int THREADS = GDN_SMEM_WARPS * 32;
    __shared__ gdn_smem_tok<COLS> tok[GDN_TOK_CHUNK];

    const int64_t unit = blockIdx.x;
    const int chunk = (int) (unit % CHUNKS);
    const int64_t hs = unit / CHUNKS;
    const int64_t h_idx = hs % p.H;
    const int64_t sequence = hs / p.H;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int col_in_warp = threadIdx.x / GDN_LANES_PER_COL;
    const int lane_in_col = threadIdx.x - col_in_warp * GDN_LANES_PER_COL;
    const int ccol = threadIdx.y * GDN_COLS_PER_WARP + col_in_warp;
    const int col0 = chunk * COLS;
    const int col = col0 + ccol;

    const int64_t state_off = gdn_state_off(p, h_idx, sequence, col);
    float s[GDN_ROWS_PER_LANE];
    gdn_state_load(s, p.state, state_off, lane_in_col);
    const int32_t w = p.keep[sequence];
    int keep, start;
    gdn_count(w, keep, start);
    const bool leave = gdn_leaves(p, w, keep, start);
    float * rows = p.rows + (sequence * p.n_tokens * p.H + h_idx) * GDN_S;
    if (keep > 0) {
        const gdn_src rs = gdn_src_from(gdn_src_replay(p), start);
        for (int t0 = 0; t0 < keep; t0 += GDN_TOK_CHUNK) {
            const int n = keep - t0 < GDN_TOK_CHUNK ? keep - t0 : GDN_TOK_CHUNK;
            __syncthreads();
            gdn_fill<COLS, THREADS, false>(tok, p, rs, h_idx, sequence, col0, t0, n, tid);
            __syncthreads();
            gdn_steps_smem<COLS, false>(p, tok, n, s, rows, lane_in_col, ccol, col);
        }
        if (!leave) {
            gdn_state_store(s, p.state_out, state_off, lane_in_col);
        }
    }
    const int at = leave ? start + keep : 0;
    // Every thread's reads of the packed rows' v (filled by any thread of the CTA) before the
    // store over them.
    __syncthreads();
    // Padding tokens (p.pad) get zero rows and are not stored, as in gdn_replay_rows_kernel.
    const int live = gdn_live(p, sequence);
    gdn_stash_v(p, h_idx, sequence, lane_in_col, col, at, live);
    const gdn_src x = gdn_src_of(p);
    for (int t0 = 0; t0 < live; t0 += GDN_TOK_CHUNK) {
        const int n = live - t0 < GDN_TOK_CHUNK ? live - t0 : GDN_TOK_CHUNK;
        __syncthreads();
        gdn_fill<COLS, THREADS, true>(tok, p, x, h_idx, sequence, col0, t0, n, tid);
        __syncthreads();
        gdn_steps_smem<COLS, true>(p, tok, n, s, rows, lane_in_col, ccol, col);
    }
    if (lane_in_col == 0) {
        for (int t = live; t < (int) p.n_tokens; t++) {
            rows[col] = 0.0f;
            rows += GDN_S * p.H;
        }
    }
    gdn_stash_shared(p, h_idx, sequence, CHUNKS, at, live);
}

cudaError_t gdn_replay_smem_launch(const legion_gdn & p, cudaStream_t st) {
    const int64_t units = p.H * p.n_seqs * (GDN_S / (GDN_SMEM_WARPS * GDN_COLS_PER_WARP));
    if (units > 0x7fffffff) {
        return cudaErrorInvalidConfiguration;
    }
    gdn_replay_smem_kernel<<<dim3((unsigned) units), dim3(32, GDN_SMEM_WARPS, 1), 0, st>>>(p);
    return cudaGetLastError();
}

template <bool ROWS, bool STATE>
cudaError_t gdn_dispatch(const legion_gdn & p, dim3 grid, dim3 block, cudaStream_t st) {
    switch (p.n_tokens) {
        case 1: gdn_kernel<1, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 2: gdn_kernel<2, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 3: gdn_kernel<3, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 4: gdn_kernel<4, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 5: gdn_kernel<5, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 6: gdn_kernel<6, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 7: gdn_kernel<7, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        case 8: gdn_kernel<8, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
        default: gdn_kernel<0, ROWS, STATE><<<grid, block, 0, st>>>(p); break;
    }
    return cudaGetLastError();
}

// ---- One-token steps (decode): rows and final states, the state staged through shared memory.
//
// A decode step's delta rule reads and writes every sequence's whole state (64 KiB per head) and
// little else, so it runs at the speed its state traffic streams. ggml's kernel (and gdn_kernel
// above) has each lane load and store its 16 rows of a column one float at a time, 8 lanes per
// 32-byte segment; on a GB10 that pattern streams the read + write at ~195 GB/s (#469b). Here a
// CTA moves its columns' state (COLS x 512 B, contiguous) between global and shared memory in
// 16-byte vectors, and each lane then takes the same 16 rows of the same column out of shared
// memory as before, with the same arithmetic: rows and states stay ggml's bits.

constexpr int GDN_LD = GDN_S + 8;  // a column's stride in shared memory (floats): no bank conflicts

// The variants (the microbenchmark compares them; GDN_DECODE_DEFAULT runs the model).
// bit 0: chunk-first grid (a head's column chunks in consecutive CTAs); bit 1: 8 warps (32
// columns) per CTA instead of 4; bit 2: cp.async loads (no register staging); bit 3: streaming
// (evict-first) cache hints.
constexpr int GDN_DECODE_LINEAR = 1;
constexpr int GDN_DECODE_W8 = 2;
constexpr int GDN_DECODE_ASYNC = 4;
constexpr int GDN_DECODE_STREAM = 8;
constexpr int GDN_DECODE_DEFAULT = GDN_DECODE_LINEAR;

__device__ __forceinline__ void gdn_cp_async16(float * smem, const float * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
#else
    *(float4 *) smem = *(const float4 *) gmem;
#endif
}

__device__ __forceinline__ void gdn_cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

template <int WARPS, bool LINEAR, bool ASYNC, bool STREAM>
__global__ void __launch_bounds__(WARPS * 32) gdn_decode_kernel(const legion_gdn p) {
    constexpr int COLS = WARPS * GDN_COLS_PER_WARP;
    constexpr int CHUNKS = GDN_S / COLS;
    constexpr int THREADS = WARPS * 32;
    constexpr int N4 = COLS * GDN_S / 4;  // float4s of the CTA's state
    static_assert(N4 % THREADS == 0, "whole float4 rounds");
    __shared__ __align__(16) float sm[COLS * GDN_LD];

    int64_t h_idx, sequence;
    int chunk;
    if (LINEAR) {
        const int64_t b = blockIdx.x;
        chunk = (int) (b % CHUNKS);
        const int64_t hs = b / CHUNKS;
        h_idx = hs % p.H;
        sequence = hs / p.H;
    } else {
        h_idx = blockIdx.x;
        sequence = blockIdx.y;
        chunk = blockIdx.z;
    }
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int col_in_warp = threadIdx.x / GDN_LANES_PER_COL;
    const int lane_in_col = threadIdx.x - col_in_warp * GDN_LANES_PER_COL;
    const int ccol = threadIdx.y * GDN_COLS_PER_WARP + col_in_warp;  // column within the CTA
    const int col = chunk * COLS + ccol;

    const int64_t base = (sequence * p.H + h_idx) * GDN_S * GDN_S + (int64_t) chunk * COLS * GDN_S;
    const float * src = p.state + base;

    // The state into shared memory, column c at c * GDN_LD.
#pragma unroll
    for (int j = 0; j < N4 / THREADS; j++) {
        const int f = (tid + j * THREADS) * 4;
        float * d = sm + (f / GDN_S) * GDN_LD + f % GDN_S;
        if (ASYNC) {
            gdn_cp_async16(d, src + f);
        } else if (STREAM) {
            *(float4 *) d = __ldcs((const float4 *) (src + f));
        } else {
            *(float4 *) d = *(const float4 *) (src + f);
        }
    }

    // This token's inputs (ggml's loads) while the state is in flight.
    const int64_t iq1 = h_idx % p.neqk1;
    const int64_t iq3 = sequence / p.rq3;
    const float * q_t = p.q + iq3 * p.sq3 + iq1 * p.sq1;
    const float * k_t = p.k + iq3 * p.sq3 + iq1 * p.sq1;
    const int64_t gb = sequence * p.sb3 + h_idx * p.sb1;
    gdn_tok cur;
    gdn_load<true>(cur, q_t, k_t, p.v + sequence * p.sv3 + h_idx * p.sv1, p.g + gb, p.beta + gb,
                   lane_in_col, col);

    if (ASYNC) {
        gdn_cp_async_wait();
    }
    __syncthreads();

    float * mine = sm + ccol * GDN_LD + lane_in_col;
    float s[GDN_ROWS_PER_LANE];
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        s[r] = mine[r * GDN_LANES_PER_COL];
    }

    // ggml's one-token step, as gdn_kernel computes it.
    const float g_val = expf(cur.g);
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        kv_shard += s[r] * cur.k[r];
    }
    float kv_col = gdn_reduce(kv_shard);
    float delta_col = (cur.v - g_val * kv_col) * cur.beta;
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        s[r] = g_val * s[r] + cur.k[r] * delta_col;
        attn_partial += s[r] * cur.q[r];
    }
    float attn_col = gdn_reduce(attn_partial);
    if (lane_in_col == 0) {
        p.rows[(sequence * p.H + h_idx) * GDN_S + col] = attn_col * p.scale;
    }

    // Each lane overwrites only the values it read, so no barrier before this.
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_LANE; r++) {
        mine[r * GDN_LANES_PER_COL] = s[r];
    }
    __syncthreads();

    float * dst = p.state_out + base;
#pragma unroll
    for (int j = 0; j < N4 / THREADS; j++) {
        const int f = (tid + j * THREADS) * 4;
        const float4 x = *(const float4 *) (sm + (f / GDN_S) * GDN_LD + f % GDN_S);
        if (STREAM) {
            __stcs((float4 *) (dst + f), x);
        } else {
            *(float4 *) (dst + f) = x;
        }
    }
}

template <int WARPS, bool LINEAR, bool ASYNC, bool STREAM>
cudaError_t gdn_decode_go(const legion_gdn & p, cudaStream_t st) {
    constexpr int CHUNKS = GDN_S / (WARPS * GDN_COLS_PER_WARP);
    const dim3 block(32, WARPS, 1);
    if (LINEAR) {
        const int64_t n = p.H * p.n_seqs * CHUNKS;
        if (n > 0x7fffffff) {
            return cudaErrorInvalidConfiguration;
        }
        gdn_decode_kernel<WARPS, LINEAR, ASYNC, STREAM><<<dim3((unsigned) n), block, 0, st>>>(p);
    } else {
        const dim3 grid((unsigned) p.H, (unsigned) p.n_seqs, CHUNKS);
        gdn_decode_kernel<WARPS, LINEAR, ASYNC, STREAM><<<grid, block, 0, st>>>(p);
    }
    return cudaGetLastError();
}

template <int WARPS, bool LINEAR>
cudaError_t gdn_decode_hint(const legion_gdn & p, int variant, cudaStream_t st) {
    const bool async = variant & GDN_DECODE_ASYNC;
    const bool stream = variant & GDN_DECODE_STREAM;
    if (async && stream) {
        return gdn_decode_go<WARPS, LINEAR, true, true>(p, st);
    }
    if (async) {
        return gdn_decode_go<WARPS, LINEAR, true, false>(p, st);
    }
    if (stream) {
        return gdn_decode_go<WARPS, LINEAR, false, true>(p, st);
    }
    return gdn_decode_go<WARPS, LINEAR, false, false>(p, st);
}

}  // namespace

// One token per sequence: the rows into p->rows and the final states into p->state_out (which may
// be p->state), ggml's bits. `variant` < 0 is the default.
extern "C" int legion_gdn_decode_launch(const legion_gdn * p, int variant, void * stream) {
    if (p->rows == nullptr || p->state_out == nullptr || p->n_tokens != 1 || p->n_seqs < 1 ||
        p->H < 1 || p->H > 65535 || p->n_seqs > 65535 || p->neqk1 < 1 || p->rq3 < 1 || variant > 15) {
        return -1;
    }
    // float4 access to both states.
    if ((((uintptr_t) p->state) | ((uintptr_t) p->state_out)) % 16 != 0) {
        return -1;
    }
    if (variant < 0) {
        variant = GDN_DECODE_DEFAULT;
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const bool linear = variant & GDN_DECODE_LINEAR;
    cudaError_t e;
    if (variant & GDN_DECODE_W8) {
        e = linear ? gdn_decode_hint<8, true>(*p, variant, st) : gdn_decode_hint<8, false>(*p, variant, st);
    } else {
        e = linear ? gdn_decode_hint<4, true>(*p, variant, st) : gdn_decode_hint<4, false>(*p, variant, st);
    }
    return (int) e;
}

extern "C" int legion_gdn_launch(const legion_gdn * p, void * stream) {
    const bool rows = p->rows != nullptr;
    const bool state = p->state_out != nullptr;
    if ((!rows && !state) || p->n_tokens < 1 || p->n_seqs < 1 || p->H < 1 || p->H > 65535 ||
        p->n_seqs > 65535 || p->neqk1 < 1 || p->rq3 < 1) {
        return -1;
    }
    // ggml's CTA shape (gated_delta_net.cu): four warps, fewer while that leaves SMs without a
    // CTA. The shape does not change any column's arithmetic.
    int num_warps = p->num_warps;
    if (num_warps <= 0) {
        const int nsm = legion_device::sm_count();
        num_warps = 4;
        while (num_warps > 1 && p->H * p->n_seqs * (GDN_S / (GDN_COLS_PER_WARP * num_warps)) < nsm) {
            num_warps /= 2;
        }
    }
    if (num_warps != 1 && num_warps != 2 && num_warps != 4) {
        return -1;
    }
    const dim3 grid((unsigned) p->H, (unsigned) p->n_seqs, GDN_S / (GDN_COLS_PER_WARP * num_warps));
    const dim3 block(32, num_warps, 1);
    const cudaStream_t st = (cudaStream_t) stream;
    cudaError_t e;
    if (p->replay != nullptr) {
        if (!rows || !state || p->keep == nullptr || p->arrive == nullptr || p->rq3 != 1 ||
            p->H % p->neqk1 != 0 || p->replay_variant > 1 || p->rp_t < p->n_tokens) {
            return -1;
        }
        e = p->replay_variant == 0 ? gdn_replay_dispatch(*p, grid, block, st) : gdn_replay_smem_launch(*p, st);
    } else if (p->keep != nullptr) {
        if (rows || !state) {
            return -1;
        }
        gdn_keep_kernel<<<grid, block, 0, st>>>(*p);
        e = cudaGetLastError();
    } else if (rows && state) {
        e = gdn_dispatch<true, true>(*p, grid, block, st);
    } else if (rows) {
        e = gdn_dispatch<true, false>(*p, grid, block, st);
    } else {
        e = gdn_dispatch<false, true>(*p, grid, block, st);
    }
    return (int) e;
}

extern "C" int legion_spec_window_launch(const legion_spec_window * p, void * stream) {
    if (p->src == nullptr || p->dst == nullptr || p->keep == nullptr || p->w < 0 || p->rows < 0 || p->n < 0) {
        return -1;
    }
    const int64_t total = p->w * p->rows * p->n;
    if (total == 0) {
        return 0;
    }
    const int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 0x7fffffff) {
        return -1;
    }
    spec_window_kernel<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}
