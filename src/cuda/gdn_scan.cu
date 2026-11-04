// gdn_scan: the gated delta rule over a prompt, token after token, with ggml-cuda's arithmetic
// (third_party/ggml/src/ggml-cuda/gated_delta_net.cu, gated_delta_net_cuda, MIT) so the outputs
// and the final state are its bits, faster where that kernel waits on memory (Volta).
//
// Per value head h and state column col, with S[i][col] the state (S_v = 128 rows):
//   kv    = sum_i S[i][col] * k[i]
//   delta = (v[col] - exp(g) * kv) * beta
//   S[i][col] = exp(g) * S[i][col] + k[i] * delta
//   out[col]  = scale * sum_i S[i][col] * q[i]
// ggml splits each column over 8 lanes, lane j holding rows j, j + 8, ..., j + 120, sums its 16
// products in row order and adds the 8 partial sums with xor shuffles (4, 2, 1). This kernel
// keeps that split, those orders and the expressions as ggml writes them (nvcc contracts them
// the same way), so every value rounds the same. What changes is around the arithmetic:
//
// - ggml loads each token's k, q, v, g and beta in the token's iteration, so every step of the
//   recurrence waits for a global load; here a block stages the next T tokens' inputs into
//   shared memory with coalesced loads while it computes the current T (the same idea as
//   upstream Strata 0.1.41's prompt recurrence pipelining, MIT).
// - exp(g) and beta are read once per token for the block instead of once per thread.
// - each 8-lane group holds C columns instead of one, so k and q are read from shared memory
//   once for C columns and the C recurrences interleave (more independent work per warp).
// - k and q sit lane-major in shared memory (16-byte loads), the output store has no branch,
//   and tokens unroll so one token's output sum overlaps the next token's kv sum.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_gdn_scan {
    // Element (i, head, t, seq) of q / k at q + i + head * sq1 + t * sq2 + seq * sq3 (floats);
    // the head of value head h is h % n_qk, the sequence of seq is seq / rq3 (ggml's
    // broadcast). v: (i, h, t, seq) at v + i + h * sv1 + t * sv2 + seq * sv3. g, beta:
    // (h, t, seq) at h * sb1 + t * sb2 + seq * sb3.
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * s0;   // [S_v, S_v, H, n_seqs] contiguous, column-major per head (ggml's)
    float * out;        // [S_v * H, n_tokens * n_seqs]
    float * state;      // [S_v, S_v, H, n_seqs]: the final states
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    int32_t H, n_tokens, n_seqs, n_qk, rq3;
    float scale;
};

// 0, or -1 for shapes it does not take, or a CUDA error. Heads of 128 (the caller checks).
int legion_gdn_scan_launch(const legion_gdn_scan * p, void * stream);

}  // extern "C"

namespace {

constexpr int S_V = 128;
constexpr int LANES = 8;              // lanes per column (ggml: warp_size / gdn_cols_per_warp)
constexpr int ROWS = S_V / LANES;     // rows per lane

// ggml-cuda's warp_reduce_sum<8> (common.cuh).
__device__ __forceinline__ float reduce8(float x) {
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, LANES);
    }
    return x;
}

// One block: value head blockIdx.x, columns blockIdx.y * COLS ..., sequence blockIdx.z; NW warps
// of 4 lane groups, each group C columns. Inputs move through shared memory T tokens at a time,
// double-buffered: the loads of the next T tokens are issued before the current T are computed
// and stored after them. Full chunks unroll U tokens so one token's output sum overlaps the next
// token's kv sum (they only share the updated state).
template <int C, int NW, int T, int U, int MB>
__global__ void __launch_bounds__(32 * NW, MB) gdn_scan_kernel(const legion_gdn_scan p) {
    constexpr int THREADS = 32 * NW;
    constexpr int COLS = 4 * C * NW;
    // k and q rows in shared memory lane-major: lane j's rows j, j + 8, ... at j * KP + r, so a
    // lane reads its 16 values with four 16-byte loads; KP = 20 keeps those loads (and the
    // staging stores) free of bank conflicts.
    constexpr int KP = ROWS + 4;
    constexpr int KROW = LANES * KP;
    __shared__ __align__(16) float ks[2][T][KROW];
    __shared__ __align__(16) float qs[2][T][KROW];
    __shared__ float vs[2][T][COLS];
    __shared__ float2 gb[2][T];  // (exp(g), beta)

    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    const int seq = blockIdx.z;
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int lane = threadIdx.x;
    const int grp = lane / LANES;
    const int j = lane % LANES;
    // column of this thread's c-th recurrence, as an offset in the block's columns
    auto col_of = [&](int c) { return threadIdx.y * 4 * C + c * 4 + grp; };

    const int nt = p.n_tokens;
    const int64_t qk_off = (int64_t) (h % p.n_qk) * p.sq1 + (int64_t) (seq / p.rq3) * p.sq3;
    const float * qg = p.q + qk_off;
    const float * kg = p.k + qk_off;
    const float * vg = p.v + (int64_t) h * p.sv1 + (int64_t) seq * p.sv3 + col0;
    const float * gg = p.g + (int64_t) h * p.sb1 + (int64_t) seq * p.sb3;
    const float * bg = p.beta + (int64_t) h * p.sb1 + (int64_t) seq * p.sb3;

    // Staging: per chunk of T tokens from t0, thread tid loads k and q elements e = tid + m *
    // THREADS of the chunk's T x S_V (token e / S_V, row e % S_V), v elements of the chunk's
    // T x COLS, and g or beta of one token (threads below 2 T); 0 past the prompt's end.
    static_assert((T * S_V) % THREADS == 0 && (T * COLS) % THREADS == 0 && 2 * T <= THREADS, "staging split");
    static_assert(T % U == 0, "unroll");
    constexpr int PKQ = T * S_V / THREADS;
    constexpr int PV = T * COLS / THREADS;
    float pk[PKQ], pq[PKQ], pv[PV], pgb;
    auto fetch = [&](int t0) {
#pragma unroll
        for (int m = 0; m < PKQ; m++) {
            const int e = tid + m * THREADS;
            const int t = t0 + e / S_V;
            const int64_t at = (int64_t) t * p.sq2 + e % S_V;
            pk[m] = t < nt ? kg[at] : 0.0f;
            pq[m] = t < nt ? qg[at] : 0.0f;
        }
#pragma unroll
        for (int m = 0; m < PV; m++) {
            const int e = tid + m * THREADS;
            const int t = t0 + e / COLS;
            pv[m] = t < nt ? vg[(int64_t) t * p.sv2 + e % COLS] : 0.0f;
        }
        const int t = t0 + tid % T;
        pgb = tid < 2 * T && t < nt ? (tid < T ? gg : bg)[(int64_t) t * p.sb2] : 0.0f;
    };
    auto stage = [&](int buf) {
#pragma unroll
        for (int m = 0; m < PKQ; m++) {
            const int e = tid + m * THREADS;
            const int i = e % S_V;
            ks[buf][e / S_V][(i % LANES) * KP + i / LANES] = pk[m];
            qs[buf][e / S_V][(i % LANES) * KP + i / LANES] = pq[m];
        }
#pragma unroll
        for (int m = 0; m < PV; m++) {
            const int e = tid + m * THREADS;
            vs[buf][e / COLS][e % COLS] = pv[m];
        }
        if (tid < T) {
            gb[buf][tid].x = expf(pgb);  // ggml: g_val = expf(*g_t)
        } else if (tid < 2 * T) {
            gb[buf][tid - T].y = pgb;
        }
    };

    float s[C][ROWS];
    const float * s0 = p.s0 + ((int64_t) seq * p.H + h) * S_V * S_V;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            s[c][r] = s0[(int64_t) (col0 + col_of(c)) * S_V + r * LANES + j];
        }
    }

    fetch(0);
    stage(0);
    __syncthreads();

    const int64_t out_ld = (int64_t) S_V * p.H;
    // Every lane of a group stores one of its columns' outputs (lane j: column j % C), the same
    // value as the group's other lanes with that column: no branch around the store.
    const int my_c = j % C;
    float * out = p.out + ((int64_t) seq * nt * p.H + h) * S_V + col0 + col_of(my_c);

    // One token: t in the chunk in buffer buf.
    auto step = [&](int buf, int t, float * o_t) {
        float kr[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r += 4) {
            const float4 x = *reinterpret_cast<const float4 *>(&ks[buf][t][j * KP + r]);
            kr[r] = x.x; kr[r + 1] = x.y; kr[r + 2] = x.z; kr[r + 3] = x.w;
        }
        const float2 gbv = gb[buf][t];
        const float g_val = gbv.x;
        const float beta_val = gbv.y;

        float kv[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                kv_shard += s[c][r] * kr[r];
            }
            kv[c] = kv_shard;
        }
#pragma unroll
        for (int c = 0; c < C; c++) {
            kv[c] = reduce8(kv[c]);
        }

        float qr[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r += 4) {
            const float4 x = *reinterpret_cast<const float4 *>(&qs[buf][t][j * KP + r]);
            qr[r] = x.x; qr[r + 1] = x.y; qr[r + 2] = x.z; qr[r + 3] = x.w;
        }
        float o[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            // delta[col] = (v[col] - g * kv[col]) * beta
            const float delta_col = (vs[buf][t][col_of(c)] - g_val * kv[c]) * beta_val;
            // S[i][col] = g * S[i][col] + k[i] * delta[col]; attn[col] = (S^T @ q)[col]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                s[c][r] = g_val * s[c][r] + kr[r] * delta_col;
                attn_partial += s[c][r] * qr[r];
            }
            o[c] = attn_partial;
        }
        float mine = 0.0f;
#pragma unroll
        for (int c = 0; c < C; c++) {
            o[c] = reduce8(o[c]);
            mine = my_c == c ? o[c] : mine;
        }
        *o_t = mine * p.scale;
    };

    for (int t0 = 0, buf = 0; t0 < nt; t0 += T, buf ^= 1) {
        const bool more = t0 + T < nt;
        if (more) {
            fetch(t0 + T);
        }
        float * o_c = out + (int64_t) t0 * out_ld;
        if (nt - t0 >= T) {
#pragma unroll U
            for (int t = 0; t < T; t++) {
                step(buf, t, o_c + t * out_ld);
            }
        } else {
            for (int t = 0; t < nt - t0; t++) {
                step(buf, t, o_c + t * out_ld);
            }
        }
        if (more) {
            stage(buf ^ 1);
        }
        __syncthreads();
    }

    float * st = p.state + ((int64_t) seq * p.H + h) * S_V * S_V;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            st[(int64_t) (col0 + col_of(c)) * S_V + r * LANES + j] = s[c][r];
        }
    }
}

template <int C, int NW, int T, int U, int MB>
int launch(const legion_gdn_scan & p, cudaStream_t stream) {
    constexpr int COLS = 4 * C * NW;
    const dim3 grid(p.H, S_V / COLS, p.n_seqs);
    gdn_scan_kernel<C, NW, T, U, MB><<<grid, dim3(32, NW), 0, stream>>>(p);
    return (int) cudaGetLastError();
}

}  // namespace

extern "C" int legion_gdn_scan_launch(const legion_gdn_scan * p, void * stream) {
    if (p->n_tokens < 1 || p->H < 1 || p->n_seqs < 1 || p->n_qk < 1 || p->rq3 < 1) {
        return -1;
    }
    // V100, qwen4exp's 48 value heads: 1.50 ms against ggml's 1.86 for 2058 tokens, 2.96 against
    // 4.14 for 4096 (32 heads, 4096 tokens: 2.3 against 3.2). C = 2, 4 warps, 8-token chunks, 4
    // tokens unrolled, at most 168 registers so 3 blocks fit an SM and 48 heads' 192 blocks run
    // in one wave (with 16-token chunks the kernel needs 248 registers, 2 blocks an SM, and two
    // waves there); one column per group or 2-warp blocks measured slower.
    return launch<2, 4, 8, 4, 3>(*p, (cudaStream_t) stream);
}
