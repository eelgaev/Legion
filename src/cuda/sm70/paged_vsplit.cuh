// paged_attn/common.cuh, sm_70+ (WMMA): the WMMA prefill kernel's result for every row of a
// group of 2-16 tokens per sequence (speculative verify steps, short prompt chunks), split over
// the key blocks (namespace vsplit, #1211).
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after
// the WMMA prefill kernel and the helpers it shares with it (BLK, Walk, walk, first_key, head_at,
// AccFrag, mma_acc). launch() in paged_attn/common.cuh takes it in place of paged_prefill when the
// op is given the scratch for it (legion_paged_attn::split with vsplit_chunk > 0; sized by
// paged::wmma_split_plan on the Rust side), on devices that run the WMMA kernel (sm_70 / sm_75
// without the m8n8k4 opt-in).
//
// Why: paged_prefill gives a CTA 64 query rows (64 / G tokens times the G query heads of one kv
// head) and walks every key block of the context twice, one block after another. A verify step of
// Qwen3.8-27B (24 query heads over 4 kv heads, MTP k=3: 4 tokens) is 4 such CTAs at one user, so 4
// of a V100's 80 SMs walk a 32K context's 512 blocks twice per layer. sm70/paged_split.cuh (#436)
// already split the last token's rows this way; this does it for all rows of the step.
//
// Same bits: every output row is what paged_prefill writes for it, bit for bit (checked by
// paged/wmma_split_tests.rs and `legion paged-split-bench` against it). Each prefill CTA of the
// step (a "tile": tokens t0 .. t0 + 64 / G - 1) keeps its rows, in the same row order (token in
// tile times G plus head), padded to whole 16-row WMMA tiles, and its walk (the key blocks
// [b_first, nblk) paged_prefill's CTA for it walks). Every value paged_prefill derives for a row is
// computed with the same operations in the same order; only which thread computes it, and when,
// changes:
// - scores: the same WMMA products (16 x 16 x 16 steps from fresh fragments added to the F32 total
//   in key-dimension order, mma_acc) of the same F16 query and key tiles. A WMMA result element
//   depends on its own row and column only.
// - pass 1 (each row's max and sum): per block the same max and the same sum (four 16-key parts,
//   each added in key order, then (s0 + s1) + (s2 + s3) as the shuffles add them), against the
//   running max through that block (a prefix max, exact in any order), then the same sequential
//   chain l = l * exp(m - m_use) + sum over the blocks in order (one thread per row).
// - pass 2: the same F16 weights exp(s * scale - m) / l, the same WMMA steps of P and V added to
//   the output in the same (block, 16-key step) order, by one warp per 16 rows and 16 output dims,
//   which only adds. The output fragment is carried from one chunk of blocks to the next through
//   F32 scratch (exact).
//
// Kernels, per run of sequences (split_seqs at a time) and per (tile, kv head) unit:
//   vs_max    CTAs sharing out the key blocks: the scores, each row's max per block
//   vs_scan   one thread per row: the running max through each block (in place)
//   vs_sum    CTAs sharing out the key blocks: the scores again, each row's sum per block
//   vs_chain  one thread per row: the chain over the blocks into (m, 1 / l)
//   then per chunk of vsplit_chunk blocks:
//   vs_probs  CTAs sharing out the chunk's blocks: the scores again, the F16 weights P
//   vs_pv     one warp per (16 rows, 16 output dims): O += P V over the chunk's blocks in order,
//             then (last chunk) the rows to dst
// Computing the scores three times instead of twice costs flops, spread over the whole GPU; the
// scratch then holds two floats per row and block and one chunk of weights, not every score.
//
// Ideas from 1Cat-vLLM's grouped verify attention (Apache-2.0; csrc/attention/sm70_grouped_long/
// kernel/grouped-attention.cu, BSD-3, D. Skryabin, and docs/design/sm70_fp16_grouped_verify.md):
// the rows of a short step share each key tile, and the step splits over the key blocks. No code
// is taken from them; their kernels round differently (F32 softmax and P V), this one keeps the
// WMMA kernel's bits.
//
// Scratch (legion_paged_attn::split), per sequence of a run and unit (tile T, kv head kvh), unit
// index T * n_head_kv + kvh, with nr = the tile's rows padded to 16, nb = table_width, ch =
// vsplit_chunk, D the head size, in floats:
//   st    [nb][nr] float2  per block: the row's max (then the running max), its sum
//   fin   [nr] float2      each row's final max and 1 / sum
//   carry [nr / 16][D / 16][256]  the output fragments between chunks
//   P     [ch][nr][BLK] F16  the chunk's weights
// (paged::wmma_split_floats / wmma_split_plan on the Rust side, which check the size.)

namespace vsplit {

constexpr int CTAS = 320;      // CTAs a score kernel aims for (4 per SM of a V100)
constexpr int SLD = BLK + 4;   // floats, as PrefillSmem
constexpr int NT  = 256;       // threads of the score kernels: 4 per row of a 64-row tile

// Rows of a tile of a step of n_tok tokens, padded to whole WMMA tiles (the first tile has the
// most), and the tiles.
template <int G>
__host__ __device__ __forceinline__ int rows16(int n_tok) {
    constexpr int QT = 64 / G;
    return ((n_tok < QT ? n_tok : QT) * G + 15) / 16 * 16;
}

template <int G>
__host__ __device__ __forceinline__ int tiles(int n_tok) {
    constexpr int QT = 64 / G;
    return (n_tok + QT - 1) / QT;
}

// The most rows a tile has at up to 16 tokens.
template <int G>
constexpr int max_rows() {
    return ((16 < 64 / G ? 16 : 64 / G) * G + 15) / 16 * 16;
}

template <int D>
__host__ __device__ constexpr int smem_bytes(int nr) {
    return nr * (D + 8) * (int) sizeof(half) + nr * SLD * (int) sizeof(float);
}

// Tile T of sequence seq: its first token, its valid rows, and the key blocks [b_first, b_first +
// nj) its paged_prefill CTA walks.
struct Tile {
    int t0, rows, b_first, nj;
};

template <int G>
__device__ __forceinline__ Tile tile(const legion_paged_attn & p, int seq, int T) {
    constexpr int QT = 64 / G;
    const int n_past = p.n_past[seq];
    const int t0     = T * QT;
    const int t_last = min(t0 + QT, p.n_tok) - 1;
    const int nblk    = (n_past + t_last) / BLK + 1;
    const int b_first = first_key(n_past + t0, p.window) / BLK;
    // A walk longer than the table would read aliased ring entries: the scratch has room for
    // table_width blocks (the walks the caller makes are never longer).
    return {t0, (t_last - t0 + 1) * G, b_first, min(nblk - b_first, p.table_width)};
}

// The scratch of the `run`-th sequence of a launch's run and unit u.
struct Lane {
    float2 * st;
    float2 * fin;
    float  * carry;
    half   * pw;
};

template <int D>
__device__ __forceinline__ Lane lane(const legion_paged_attn & p, int run, int u, int units, int nr) {
    const int64_t nb  = p.table_width;
    const int64_t per = 2 * nb * nr + 2 * nr + (int64_t) nr * D + (int64_t) p.vsplit_chunk * nr * (BLK / 2);
    float * base  = p.split + ((int64_t) run * units + u) * per;
    float * carry = base + 2 * nb * nr + 2 * nr;
    return {(float2 *) base, (float2 *) (base + 2 * nb * nr), carry, (half *) (carry + (int64_t) nr * D)};
}

// The tile's query rows as F16 in qs ([nr][D + 8]), rows past its valid ones zero, as
// paged_prefill loads them.
template <int D, int G>
__device__ __forceinline__ void load_q(const legion_paged_attn & p, int seq, const Tile & tl, int kvh, int nr, half * qs) {
    constexpr int QLD = D + 8;
    for (int i = threadIdx.x; i < nr * D; i += blockDim.x) {
        const int r = i / D, c = i % D;
        float x = 0.0f;
        if (r < tl.rows) {
            x = *(const float *) ((const char *) p.q + seq * p.q_nb3 + (tl.t0 + r / G) * p.q_nb2 +
                                  (kvh * G + r % G) * p.q_nb1 + c * sizeof(float));
        }
        qs[r * QLD + c] = __float2half(x);
    }
}

// S = Q K^T of key block b (walk entry b) for the tile's nr rows into ss ([nr][SLD]): warp w takes
// key columns 16 (w % 4) .. and row tiles w / 4, w / 4 + 2.
template <int D, bool C>
__device__ __forceinline__ void scores(const legion_paged_attn & p, const Walk & wk, int b, int kvh, int nr,
                                       const half * qs, float * ss) {
    constexpr int QLD = D + 8;
    const int warp = threadIdx.x / 32;
    const int col = warp % 4, rt0 = warp / 4;
    const int nrt = nr / 16;
    if (rt0 >= nrt) {
        return;
    }
    const bool two = rt0 + 2 < nrt;
    const half * kblk = (const half *) wk.cells(p.k, p.k_stage, b, p.k_cell) + head_at<D, C>(p, kvh);
    AccFrag acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);
#pragma unroll 4
    for (int k = 0; k < D / 16; k++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
        wmma::load_matrix_sync(kb, kblk + (int64_t) (col * 16) * p.kv_row + k * 16, p.kv_row);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, qs + rt0 * 16 * QLD + k * 16, QLD);
        mma_acc(acc[0], a, kb);
        if (two) {
            wmma::load_matrix_sync(a, qs + (rt0 + 2) * 16 * QLD + k * 16, QLD);
            mma_acc(acc[1], a, kb);
        }
    }
    wmma::store_matrix_sync(ss + rt0 * 16 * SLD + col * 16, acc[0], SLD, wmma::mem_row_major);
    if (two) {
        wmma::store_matrix_sync(ss + (rt0 + 2) * 16 * SLD + col * 16, acc[1], SLD, wmma::mem_row_major);
    }
}

// What the score kernels share: the tile, its kv head and sequence, the walk, the scratch, and
// the Q rows loaded into shared memory. `j0`, `j1`: the walk's blocks b_first + j for j in [j0,
// j1) this CTA takes, an equal share of the blocks [lo_j, hi_j) of the walk for each of the
// gridDim.x CTAs of the unit (the grid does not depend on the context's length, which is device
// data: graphs are kept across steps); false (nothing to do) when that is empty.
struct Ctx {
    Tile tl;
    int kvh, seq, nr, j0, j1;
    Walk wk;
};

template <int D, int G>
__device__ __forceinline__ bool begin(const legion_paged_attn & p, int seq0, int lo_j, int hi_j, Ctx & c,
                                      half * qs) {
    c.kvh = blockIdx.y % p.n_head_kv;
    c.seq = seq0 + blockIdx.z;
    c.nr  = rows16<G>(p.n_tok);
    c.tl  = tile<G>(p, c.seq, blockIdx.y / p.n_head_kv);
    const int end = min(hi_j, c.tl.nj);
    const int per = (end - lo_j + (int) gridDim.x - 1) / (int) gridDim.x;
    c.j0 = lo_j + blockIdx.x * per;
    c.j1 = min(c.j0 + per, end);
    if (c.j0 >= c.j1) {
        return false;
    }
    c.wk = walk(p, c.seq, c.tl.b_first, c.tl.b_first + c.tl.nj - 1);
    load_q<D, G>(p, c.seq, c.tl, c.kvh, c.nr, qs);
    __syncthreads();
    return true;
}

// Row r's scores of block b (16 columns from 16 part), scaled, the keys it does not see -inf.
__device__ __forceinline__ void row_scores(const legion_paged_attn & p, const float * ss, int r, int part, int b,
                                           int limit, int lo, float * x) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int c = part * 16 + i;
        const int k = b * BLK + c;
        x[i] = k <= limit && k >= lo ? ss[r * SLD + c] * p.scale : -INFINITY;
    }
}

// Pass 1, part 1: each row's max per block.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(NT) vs_max(const legion_paged_attn p, int seq0) {
    extern __shared__ __align__(128) unsigned char smem[];
    half * qs = (half *) smem;
    Ctx c;
    if (!begin<D, G>(p, seq0, 0, 1 << 30, c, qs)) {
        return;
    }
    float * ss = (float *) (smem + c.nr * (D + 8) * sizeof(half));
    const Lane ln = lane<D>(p, blockIdx.z, blockIdx.y, gridDim.y, c.nr);
    const int r = threadIdx.x / 4, part = threadIdx.x % 4;
    const int limit = p.n_past[c.seq] + c.tl.t0 + r / G;
    const int lo    = first_key(limit, p.window);
    for (int j = c.j0; j < c.j1; j++) {
        const int b = c.tl.b_first + j;
        scores<D, C>(p, c.wk, b, c.kvh, c.nr, qs, ss);
        __syncthreads();
        if (r < c.nr) {  // whole warps
            float x[16];
            row_scores(p, ss, r, part, b, limit, lo, x);
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                mx = fmaxf(mx, x[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            if (part == 0) {
                ln.st[(int64_t) j * c.nr + r].x = mx;
            }
        }
        __syncthreads();
    }
}

// Pass 1, part 2: the running max through each block, in place. One thread per row.
template <int D, int G>
__global__ void __launch_bounds__(64) vs_scan(const legion_paged_attn p, int seq0) {
    const int nr = rows16<G>(p.n_tok), r = threadIdx.x;
    if (r >= nr) {
        return;
    }
    const Tile tl = tile<G>(p, seq0 + blockIdx.y, blockIdx.x / p.n_head_kv);
    float2 * st = lane<D>(p, blockIdx.y, blockIdx.x, gridDim.x, nr).st + r;
    float m = -INFINITY;
#pragma unroll 8
    for (int j = 0; j < tl.nj; j++) {
        m = fmaxf(m, st[(int64_t) j * nr].x);
        st[(int64_t) j * nr].x = m;
    }
}

// Pass 1, part 3: each row's sum per block against the running max through it, as
// paged_prefill's pass 1 adds it.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(NT) vs_sum(const legion_paged_attn p, int seq0) {
    extern __shared__ __align__(128) unsigned char smem[];
    half * qs = (half *) smem;
    Ctx c;
    if (!begin<D, G>(p, seq0, 0, 1 << 30, c, qs)) {
        return;
    }
    float * ss = (float *) (smem + c.nr * (D + 8) * sizeof(half));
    const Lane ln = lane<D>(p, blockIdx.z, blockIdx.y, gridDim.y, c.nr);
    const int r = threadIdx.x / 4, part = threadIdx.x % 4;
    const int limit = p.n_past[c.seq] + c.tl.t0 + r / G;
    const int lo    = first_key(limit, p.window);
    for (int j = c.j0; j < c.j1; j++) {
        const int b = c.tl.b_first + j;
        scores<D, C>(p, c.wk, b, c.kvh, c.nr, qs, ss);
        __syncthreads();
        if (r < c.nr) {
            float x[16];
            row_scores(p, ss, r, part, b, limit, lo, x);
            const float m_new = ln.st[(int64_t) j * c.nr + r].x;
            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                sum += expf(x[i] - m_use);
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            if (part == 0) {
                ln.st[(int64_t) j * c.nr + r].y = sum;
            }
        }
        __syncthreads();
    }
}

// Pass 1, part 4: the chain over the blocks in order into each row's (m, 1 / l). One thread per
// row.
template <int D, int G>
__global__ void __launch_bounds__(64) vs_chain(const legion_paged_attn p, int seq0) {
    const int nr = rows16<G>(p.n_tok), r = threadIdx.x;
    if (r >= nr) {
        return;
    }
    const Tile tl = tile<G>(p, seq0 + blockIdx.y, blockIdx.x / p.n_head_kv);
    const Lane ln = lane<D>(p, blockIdx.y, blockIdx.x, gridDim.x, nr);
    const float2 * st = ln.st + r;
    float m = -INFINITY, l = 0.0f;
#pragma unroll 8
    for (int j = 0; j < tl.nj; j++) {
        const float2 e = st[(int64_t) j * nr];
        const float m_new = e.x;
        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
        l = l * expf(m - m_use) + e.y;
        m = m_new;
    }
    const float inv_l = 1.0f / l;
    ln.fin[r] = make_float2(m, inv_l);
}

// Pass 2, part 1: the weights P = exp(S * scale - m) / l as F16 of chunk `chunk`'s blocks.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(NT) vs_probs(const legion_paged_attn p, int seq0, int chunk) {
    extern __shared__ __align__(128) unsigned char smem[];
    half * qs = (half *) smem;
    const int c0 = chunk * p.vsplit_chunk;
    Ctx c;
    if (!begin<D, G>(p, seq0, c0, c0 + p.vsplit_chunk, c, qs)) {
        return;
    }
    float * ss = (float *) (smem + c.nr * (D + 8) * sizeof(half));
    const Lane ln = lane<D>(p, blockIdx.z, blockIdx.y, gridDim.y, c.nr);
    const int r = threadIdx.x / 4, part = threadIdx.x % 4;
    const int limit = p.n_past[c.seq] + c.tl.t0 + r / G;
    const int lo    = first_key(limit, p.window);
    float m = 0.0f, inv_l = 0.0f;
    if (r < c.nr) {
        const float2 f = ln.fin[r];
        m     = f.x;
        inv_l = f.y;
    }
    for (int j = c.j0; j < c.j1; j++) {
        const int b = c.tl.b_first + j;
        scores<D, C>(p, c.wk, b, c.kvh, c.nr, qs, ss);
        __syncthreads();
        if (r < c.nr) {
            half * pw = ln.pw + ((int64_t) (j - c0) * c.nr + r) * BLK;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int cc = part * 16 + i;
                const int k = b * BLK + cc;
                const float w = k <= limit && k >= lo ? expf(ss[r * SLD + cc] * p.scale - m) * inv_l : 0.0f;
                pw[cc] = __float2half(w);
            }
        }
        __syncthreads();
    }
}

// Pass 2, part 2: O += P V over chunk `chunk`'s blocks in order, for rows 16 rt .. and output dims
// 16 n .. (blockIdx.x = rt * D / 16 + n); after the last chunk the rows to dst.
template <int D, int G, bool C = false>
__global__ void __launch_bounds__(32) vs_pv(const legion_paged_attn p, int seq0, int chunk) {
    constexpr int ND = D / 16;
    const int n = blockIdx.x % ND, rt = blockIdx.x / ND;
    const int kvh = blockIdx.y % p.n_head_kv, seq = seq0 + blockIdx.z;
    const int nr = rows16<G>(p.n_tok);
    const Tile tl = tile<G>(p, seq, blockIdx.y / p.n_head_kv);
    const int j0 = chunk * p.vsplit_chunk, j1 = min(j0 + p.vsplit_chunk, tl.nj);
    if (j0 >= j1 || rt * 16 >= nr) {
        return;
    }
    const Lane ln = lane<D>(p, blockIdx.z, blockIdx.y, gridDim.y, nr);
    const Walk wk = walk(p, seq, tl.b_first, tl.b_first + tl.nj - 1);
    float * carry = ln.carry + (int64_t) (rt * ND + n) * 256;
    AccFrag o;
    if (chunk == 0) {
        wmma::fill_fragment(o, 0.0f);
    } else {
        wmma::load_matrix_sync(o, carry, 16, wmma::mem_row_major);
    }
#pragma unroll 2
    for (int j = j0; j < j1; j++) {
        const half * vblk = (const half *) wk.cells(p.v, p.v_stage, tl.b_first + j, p.v_cell) + head_at<D, C>(p, kvh);
        const half * pb = ln.pw + ((int64_t) (j - j0) * nr + rt * 16) * BLK;
#pragma unroll
        for (int kk = 0; kk < BLK / 16; kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, pb + kk * 16, BLK);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
            wmma::load_matrix_sync(vb, vblk + (int64_t) (kk * 16) * p.kv_row + n * 16, p.kv_row);
            mma_acc(o, a, vb);
        }
    }
    if (j1 < tl.nj) {
        wmma::store_matrix_sync(carry, o, 16, wmma::mem_row_major);
        return;
    }
    __shared__ __align__(128) float stage[256];
    wmma::store_matrix_sync(stage, o, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = threadIdx.x; i < 256; i += 32) {
        const int rr = rt * 16 + i / 16;
        if (rr < tl.rows) {
            const int t = tl.t0 + rr / G;
            const int h = kvh * G + rr % G;
            p.dst[(((int64_t) seq * p.n_tok + t) * p.n_head + h) * D + n * 16 + i % 16] = stage[i];
        }
    }
}

// -1 when the device cannot take the kernels (their shared memory), else 0 on success and 2 for a
// launch error.
template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    constexpr int SMEM_MAX = smem_bytes<D>(max_rows<G>());
    if (p.n_tok < 2 || p.n_tok > 16) {
        return -1;
    }
    // Per device, once (the attribute is per function and device).
    static std::atomic<int8_t> opted[3][GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted[0], vs_max<D, G, C>, SMEM_MAX) ||
        !legion_device::smem_opt_in(opted[1], vs_sum<D, G, C>, SMEM_MAX) ||
        !legion_device::smem_opt_in(opted[2], vs_probs<D, G, C>, SMEM_MAX)) {
        return -1;
    }
    t_vsplit_launches++;
    const int nr = rows16<G>(p.n_tok), units = tiles<G>(p.n_tok) * p.n_head_kv;
    const int smem = smem_bytes<D>(nr);
    const int nb = p.table_width, ch = p.vsplit_chunk;
    const int chunks = (nb + ch - 1) / ch;
    for (int seq0 = 0; seq0 < p.n_seqs; seq0 += p.split_seqs) {
        const int n = min(p.split_seqs, p.n_seqs - seq0);
        // CTAs per unit: the blocks shared out at run time.
        const int per_unit = (CTAS + units * n - 1) / (units * n);
        const dim3 all(min(per_unit, nb), units, n);
        vs_max<D, G, C><<<all, NT, smem, st>>>(p, seq0);
        vs_scan<D, G><<<dim3(units, n), 64, 0, st>>>(p, seq0);
        vs_sum<D, G, C><<<all, NT, smem, st>>>(p, seq0);
        vs_chain<D, G><<<dim3(units, n), 64, 0, st>>>(p, seq0);
        for (int c = 0; c < chunks; c++) {
            vs_probs<D, G, C><<<dim3(min(per_unit, ch), units, n), NT, smem, st>>>(p, seq0, c);
            vs_pv<D, G, C><<<dim3(nr / 16 * (D / 16), units, n), 32, 0, st>>>(p, seq0, c);
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace vsplit
