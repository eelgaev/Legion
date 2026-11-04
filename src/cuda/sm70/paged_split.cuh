// paged_attn/common.cuh, sm_70+ (WMMA): the WMMA prefill kernel's result for the last token of each
// sequence, split over the key blocks (namespace split, #436).
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after the
// WMMA prefill kernel and the shared helpers it uses (BLK, Walk, walk, first_key, AccFrag,
// mma_acc, PrefillSmem's layout ideas). launch() in paged_attn/common.cuh takes it in place of
// paged_prefill when the op is given a split scratch (legion_paged_attn::split): only the last
// token's rows of each sequence are read afterwards (an MTP head's take-in pass, whose earlier
// rows only write their K/V), and the device runs the WMMA kernel (sm_70 / sm_75 without the
// m8n8k4 opt-in).
//
// Why: paged_prefill gives a CTA 64 query rows of one kv head and walks every key block twice,
// one after another. A take-in pass of qwen4exp's MTP head has 1 to 3 tokens of 24 query heads
// over 2 kv heads: 2 CTAs, so at a 95K context two SMs walk 1,500 blocks twice (about 40 ms a
// pass on a V100, most of a draft window's time).
//
// Same bits: the output rows are what paged_prefill writes for them, bit for bit (checked by
// paged/split_tests.rs against it). Every value paged_prefill derives for a row is computed
// here with the same operations in the same order; only which thread computes it, and when,
// changes:
// - scores: the same WMMA products (16 x 16 x 16 steps, each from a fresh fragment added to
//   the F32 total in key-dimension order) of the same F16 query and key tiles. A WMMA result
//   element depends on its own row and column only, so the last token's G rows sit at rows
//   0..G-1 of one 16-row tile here instead of where paged_prefill's 64-row tile puts them.
// - pass 1 (each row's max and sum): per block the same max and the same sum (four 16-key
//   parts added in key order, then (s0 + s1) + (s2 + s3), as the shuffles add them), against
//   the running max up to that block (a prefix max, exact in any order), then the same
//   sequential chain l = l * exp(m - m_use) + sum over the blocks in order (one thread).
// - pass 2: the same F16 weights exp(s * scale - m) / l, the same WMMA steps of P and V, added
//   to the output in the same (block, 16-key step) order; one warp per 16 output dims.
// The walk covers the blocks paged_prefill's CTA for the last token's tile walks, so the
// masked keys (exact zero weights) are the same too.
//
// Work: the scores and weights of every block in parallel (one CTA per block), the chains on
// one thread per row (pass 1) and one warp per 16 output dims (pass 2), which only add.
//
// Scratch (legion_paged_attn::split), for split_seqs sequences at a time (a launch runs the
// sequences in runs of that many), per (sequence, kv head), with nb = table_width:
//   S     [nb][ROWS][BLK] F32 scores, then P [ROWS][BLK] F16 in place at the start of each block
//   mx    [ROWS][nb]      each row's max per block
//   sum   [ROWS][nb]      each row's sum per block
//   stats [ROWS][2]       each row's final max and 1 / sum
// (paged::split_floats on the Rust side, which checks the size before passing it.)

namespace split {

constexpr int ROWS = 16;  // one WMMA row tile: the last token's G <= 16 query heads, padded

struct Lane {
    float * s;
    float * mx;
    float * sum;
    float * stats;
};

// The scratch of (sequence, kv head), for the `run`-th sequence of a launch's run of them.
__device__ __forceinline__ Lane lane(const legion_paged_attn & p, int run, int kvh) {
    const int64_t nb  = p.table_width;
    const int64_t per = nb * (ROWS * BLK + 2 * ROWS) + 2 * ROWS;
    float * base = p.split + ((int64_t) run * p.n_head_kv + kvh) * per;
    return {base, base + nb * ROWS * BLK, base + nb * (ROWS * BLK + ROWS), base + nb * (ROWS * BLK + 2 * ROWS)};
}

// The last token and the key blocks [b_first, nblk) that paged_prefill's CTA for the tile
// holding it walks.
struct Tile {
    int t_last, b_first, nblk;
};

template <int G>
__device__ __forceinline__ Tile tile(const legion_paged_attn & p, int seq) {
    constexpr int QT = 64 / G;
    const int n_past = p.n_past[seq];
    const int t_last = p.n_tok - 1;
    const int t0     = t_last / QT * QT;
    return {t_last, first_key(n_past + t0, p.window) / BLK, (n_past + t_last) / BLK + 1};
}

// Scores of block blockIdx.x: S = Q K^T for the last token's rows, into the block's scratch.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(128) split_scores(const legion_paged_attn p, int seq0) {
    constexpr int QLD = D + 8;  // halves, as PrefillSmem
    __shared__ __align__(128) half qs[ROWS * QLD];
    const int b = blockIdx.x, kvh = blockIdx.y, seq = seq0 + blockIdx.z;
    const Tile tl = tile<G>(p, seq);
    if (b < tl.b_first || b >= tl.nblk) {
        return;
    }
    const Walk wk = walk(p, seq, tl.b_first, tl.nblk - 1);
    for (int i = threadIdx.x; i < ROWS * D; i += blockDim.x) {
        const int r = i / D, c = i % D;
        float x = 0.0f;
        if (r < G) {
            x = *(const float *) ((const char *) p.q + seq * p.q_nb3 + tl.t_last * p.q_nb2 +
                                  (kvh * G + r) * p.q_nb1 + c * sizeof(float));
        }
        qs[r * QLD + c] = __float2half(x);
    }
    __syncthreads();

    // Warp w: key columns 16 w .. 16 w + 15.
    const int warp = threadIdx.x / 32;
    const half * kblk = (const half *) wk.cells(p.k, p.k_stage, b, p.k_cell) + head_at<D, C>(p, kvh);
    AccFrag acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll 4
    for (int k = 0; k < D / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, qs + k * 16, QLD);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
        wmma::load_matrix_sync(kb, kblk + (int64_t) (warp * 16) * p.kv_row + k * 16, p.kv_row);
        mma_acc(acc, a, kb);
    }
    const Lane ln = lane(p, blockIdx.z, kvh);
    wmma::store_matrix_sync(ln.s + (int64_t) b * ROWS * BLK + warp * 16, acc, BLK, wmma::mem_row_major);
}

// Pass 1 for row blockIdx.x (query head kvh * G + row of the last token): its max and sum per
// block, then the chain over the blocks into (m, 1 / l).
template <int G>
__global__ void __launch_bounds__(256) split_stats(const legion_paged_attn p, int seq0) {
    constexpr int NT = 256;
    const int g = blockIdx.x, kvh = blockIdx.y, seq = seq0 + blockIdx.z;
    const Tile tl = tile<G>(p, seq);
    const Lane ln = lane(p, blockIdx.z, kvh);
    const int64_t nb = p.table_width;
    const int limit = p.n_past[seq] + tl.t_last;  // last visible key of the row
    const int lo    = first_key(limit, p.window);  // first visible key of the row
    float * mx = ln.mx + g * nb;
    float * sm = ln.sum + g * nb;
    // Thread t: a run of consecutive blocks.
    const int per = (tl.nblk - tl.b_first + NT - 1) / NT;
    const int c0 = tl.b_first + threadIdx.x * per, c1 = min(c0 + per, tl.nblk);
    __shared__ float run_max[NT];

    // Each block's max (exact in any order).
    float run = -INFINITY;
    for (int b = c0; b < c1; b++) {
        const float * s = ln.s + (int64_t) b * ROWS * BLK + g * BLK;
        float m = -INFINITY;
        for (int c = 0; c < BLK; c++) {
            const int k = b * BLK + c;
            m = fmaxf(m, k <= limit && k >= lo ? s[c] * p.scale : -INFINITY);
        }
        mx[b] = m;
        run = fmaxf(run, m);
    }
    run_max[threadIdx.x] = run;
    __syncthreads();

    // Each block's sum against the running max through it, as paged_prefill's pass 1 adds it.
    float m = -INFINITY;
    for (int t = 0; t < (int) threadIdx.x; t++) {
        m = fmaxf(m, run_max[t]);
    }
    for (int b = c0; b < c1; b++) {
        m = fmaxf(m, mx[b]);
        const float m_use = m == -INFINITY ? 0.0f : m;
        const float * s = ln.s + (int64_t) b * ROWS * BLK + g * BLK;
        float part[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            float x[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int c = q * 16 + i;
                const int k = b * BLK + c;
                x[i] = k <= limit && k >= lo ? s[c] * p.scale : -INFINITY;
            }
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                sum += expf(x[i] - m_use);
            }
            part[q] = sum;
        }
        sm[b] = (part[0] + part[1]) + (part[2] + part[3]);
    }
    __syncthreads();

    // The chain, in block order.
    if (threadIdx.x == 0) {
        float m = -INFINITY, l = 0.0f;
        for (int b = tl.b_first; b < tl.nblk; b++) {
            const float m_new = fmaxf(m, mx[b]);
            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
            l = l * expf(m - m_use) + sm[b];
            m = m_new;
        }
        const float inv_l = 1.0f / l;
        ln.stats[2 * g]     = m;
        ln.stats[2 * g + 1] = inv_l;
    }
}

// Pass 2's weights of block blockIdx.x: P = exp(S * scale - m) / l as F16, over the block's
// scores (rows past G zero).
template <int G>
__global__ void __launch_bounds__(256) split_probs(const legion_paged_attn p, int seq0) {
    constexpr int PER = ROWS * BLK / 256;
    const int b = blockIdx.x, kvh = blockIdx.y, seq = seq0 + blockIdx.z;
    const Tile tl = tile<G>(p, seq);
    if (b < tl.b_first || b >= tl.nblk) {
        return;
    }
    const Lane ln = lane(p, blockIdx.z, kvh);
    const int limit = p.n_past[seq] + tl.t_last;
    const int lo    = first_key(limit, p.window);
    float * s = ln.s + (int64_t) b * ROWS * BLK;
    float w[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = threadIdx.x + j * 256;
        const int r = i / BLK, c = i % BLK;
        const int k = b * BLK + c;
        w[j] = 0.0f;
        if (r < G) {
            const float m = ln.stats[2 * r], inv_l = ln.stats[2 * r + 1];
            w[j] = k <= limit && k >= lo ? expf(s[i] * p.scale - m) * inv_l : 0.0f;
        }
    }
    __syncthreads();
    half * ps = (half *) s;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        ps[threadIdx.x + j * 256] = __float2half(w[j]);
    }
}

// Pass 2's output for dims 16 n .. 16 n + 15 (n = blockIdx.x): O += P V over the blocks in
// order, then the last token's rows to dst.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(32) split_pv(const legion_paged_attn p, int seq0) {
    const int n = blockIdx.x, kvh = blockIdx.y, seq = seq0 + blockIdx.z;
    const Tile tl = tile<G>(p, seq);
    const Lane ln = lane(p, blockIdx.z, kvh);
    const Walk wk = walk(p, seq, tl.b_first, tl.nblk - 1);
    AccFrag o;
    wmma::fill_fragment(o, 0.0f);
    for (int b = tl.b_first; b < tl.nblk; b++) {
        const half * vblk = (const half *) wk.cells(p.v, p.v_stage, b, p.v_cell) + head_at<D, C>(p, kvh);
        const half * pb = (const half *) (ln.s + (int64_t) b * ROWS * BLK);
#pragma unroll
        for (int kk = 0; kk < BLK / 16; kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, pb + kk * 16, BLK);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
            wmma::load_matrix_sync(vb, vblk + (int64_t) (kk * 16) * p.kv_row + n * 16, p.kv_row);
            mma_acc(o, a, vb);
        }
    }
    __shared__ __align__(128) float stage[256];
    wmma::store_matrix_sync(stage, o, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = threadIdx.x; i < 256; i += 32) {
        const int r = i / 16;
        if (r < G) {
            const int h = kvh * G + r;
            p.dst[(((int64_t) seq * p.n_tok + tl.t_last) * p.n_head + h) * D + n * 16 + i % 16] = stage[i];
        }
    }
}

template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    static_assert(G <= ROWS, "the last token's query heads fill one 16-row tile");
    // The sequences in runs of as many as the scratch holds. One CTA per table entry: those
    // past a sequence's blocks return at once (the host does not know the sequences' lengths,
    // which live on the device).
    for (int seq0 = 0; seq0 < p.n_seqs; seq0 += p.split_seqs) {
        const int n = min(p.split_seqs, p.n_seqs - seq0);
        const dim3 blocks(p.table_width, p.n_head_kv, n);
        split_scores<D, G, C><<<blocks, 128, 0, st>>>(p, seq0);
        split_stats<G><<<dim3(G, p.n_head_kv, n), 256, 0, st>>>(p, seq0);
        split_probs<G><<<blocks, 256, 0, st>>>(p, seq0);
        split_pv<D, G, C><<<dim3(D / 16, p.n_head_kv, n), 32, 0, st>>>(p, seq0);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace split
