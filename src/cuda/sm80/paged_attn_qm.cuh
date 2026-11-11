// paged_attn/common.cuh, compute capability 12.x (GB10, RTX 50): paged_prefill_mma's prefill with
// the queries as the MMA's M dimension (namespace fa_qm: paged_prefill_qm for heads of 256, #1133;
// paged_prefill_qp for heads of 512, #1181, below). The same bits.
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after
// sm80/paged_attn.cuh (namespace fa: Shape's swizzle, cp_async16, ldsm, ldsm_t, mma, pack).
//
// paged_prefill_mma (sm80/paged_attn.cuh) takes the queries as the MMA's N dimension: 8 query
// columns per warp, K and V as A operands, so every m16n8k16 needs its own ldmatrix.x4 and the
// kernel runs about as fast as shared memory feeds it. Here a warp owns 16 query rows (M), as in
// FlashAttention-2: S = Q K^T with Q's A fragments from a shared-memory copy of the CTA's rows
// (one ldmatrix per 8 MMAs) and K's B fragments (one ldmatrix per 2), then O += P V with P's A
// fragments straight from the score accumulators and V's B fragments (one ldmatrix per 2): about
// 0.56 ldmatrix per MMA. A CTA: 8 warps, 128 query rows (row = token_in_tile * G +
// head_in_group) of one kv head, Q in F16 (64 KB at heads of 256), K and V through a two-stage
// ring of 64 keys x 128 dims (32 KB).
//
// The same bits as paged_prefill_mma, on purpose (so it can be the default): every value goes
// through the same operations in the same order.
// - A score is one chain of D / 16 MMAs from zero over the same dims (operands swapped: an
//   m16n8k16 element is the same dot product in either role). Scale, mask and max as there.
// - paged_prefill_mma's thread sums a column's weights of keys key % 8 == gq, keys in order;
//   here the thread of a row holds keys 2 tq and 2 tq + 1 (mod 8) and keeps one running sum for
//   each, keys in order, then (l, alpha) as there. At the end the eight sums of a row add in the
//   same tree: the pair (2 tq, 2 tq + 1), then lanes tq ^ 1 (paged_prefill_mma's gq ^ 2), then
//   tq ^ 2 (gq ^ 4).
// - Each 8 output dims' P V is one chain of 4 MMAs from zero over the tile's 16-key steps, added
//   as o alpha + acc.
// Tiles a row does not see add nothing (alpha 1 or 0, sums and products of zero weights), so the
// larger tile (more key blocks for its first and last rows) leaves the bits alone.
//
// Not covered (paged_prefill_mma instead): quantized caches, the compact cache, a host tier's
// pieces, key-block splits, other head sizes, short groups (takes()).

namespace fa_qm {

template <int D> struct Shape {
    static constexpr int NW      = 8;                              // warps per CTA
    static constexpr int ROWS    = 16 * NW;                        // query rows per CTA
    static constexpr int CH      = 128;                            // dims per K / V chunk
    static constexpr int NS      = 2;                              // ring stages
    static constexpr int QUNITS  = D / 8;                          // 16-byte units per Q row
    static constexpr int QBYTES  = ROWS * D * (int) sizeof(half);  // Q tile
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per chunk row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);  // bytes per chunk
    static constexpr size_t SMEM = (size_t) QBYTES + (size_t) NS * CHUNK;
};

// The fewest query rows (n_tok * G) a launch takes the kernel at: below, the old kernel's 64-row
// tiles give more CTAs (verify steps), and the bits are the same either way.
constexpr int MIN_ROWS = 256;

template <int D, int G>
__global__ void __launch_bounds__(Shape<D>::NW * 32, 1) paged_prefill_qm(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = Shape<D>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK, QU = SH::QUNITS;
    constexpr int NW = SH::NW, ROWS = SH::ROWS;
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int KC = CH / 16;   // k-steps of S per K chunk
    constexpr int QT = ROWS / G;  // tokens per CTA
    constexpr int ROWB = UNITS * 16, QROWB = QU * 16;
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t qbase = (uint32_t) __cvta_generic_to_shared(smem);
    const uint32_t rbase = qbase + SH::QBYTES;

    // The last query tiles see the most keys: they go first.
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Row r's token. Rows past n_tok, and past QT * G when G does not divide 128, repeat the
    // tile's last token and are not written.
    auto tok_of = [&](int r) { return min(t0 + min(r / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int r) { return n_past + tok_of(r); };
    const int k_end   = pos_of(ROWS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);

    // This warp's rows (for skipping and unmasked tiles) and this thread's two rows of the S / O
    // fragments, gq and gq + 8. A warp of padding rows only (past the last token) computes nothing.
    const int r0 = warp * 16;
    const bool w_rows = r0 < min(QT * G, (p.n_tok - t0) * G);
    const int w_pos_lo = pos_of(r0), w_pos_hi = pos_of(r0 + 15);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        c_pos[r]   = pos_of(r0 + gq + 8 * r);
        c_first[r] = first_key(c_pos[r], p.window);
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / (2 * NC), part = j % (2 * NC);
            const int b = wk.block(i);
            const char * base = part < NC ? wk.cells(p.k, p.k_stage, i, p.k_cell) : wk.cells(p.v, p.v_stage, i, p.v_cell);
            const half * src = (const half *) base + kvh * D + (part % NC) * CH;
            const uint32_t dst = rbase + (uint32_t) (j % NS) * CHUNK;
#pragma unroll
            for (int x = threadIdx.x; x < BLK * UNITS; x += NW * 32) {
                const int r = x / UNITS, u = x % UNITS;
                cp_async16(dst + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + u * 8, b * BLK + r < k_end);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }

    // Q's rows in F16 (as paged_prefill_mma rounds them), rows of QU swizzled 16-byte units.
    for (int x = threadIdx.x; x < ROWS * QU; x += NW * 32) {
        const int r = x / QU, u = x % QU;
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok_of(r) * p.q_nb2 +
                                            (kvh * G + r % G) * p.q_nb1) + u * 8;
        const float2 a = *(const float2 *) qr, b = *(const float2 *) (qr + 2);
        const float2 c = *(const float2 *) (qr + 4), d = *(const float2 *) (qr + 6);
        *(uint4 *) (smem + fa::swz<QU>(r, u)) = make_uint4(pack(a.x, a.y), pack(b.x, b.y), pack(c.x, c.y), pack(d.x, d.y));
    }

    // O fragments: [n] dims 8 n + 2 tq + {0, 1}, rows gq ([0], [1]) and gq + 8 ([2], [3]).
    float o[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; n++) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    }
    // Running max (log2 units) per row, and this thread's sums per row and key class 2 tq + c.
    float m[2] = {-INFINITY, -INFINITY}, l[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const float scale = p.scale * LOG2E;

    // ldmatrix addresses. Every row a lane addresses has r % 8 == lane % 8, so unit 8 q + v
    // (v < 8) of it sits at 16 (8 q + (v ^ lane % 8)). Q (A, rows 16 warp + lane / 8 % 2 * 8 +
    // lane % 8): units 2 ks + lane / 16. K (B, keys lane / 16 * 8 + lane % 8 of a 16-key pair):
    // units 2 ks + lane / 8 % 2. V (B, transposed; keys lane / 8 % 2 * 8 + lane % 8 of a 16-key
    // step): units 2 np + lane / 16.
    const uint32_t q_lane = (uint32_t) (r0 + lane / 8 % 2 * 8 + lane % 8) * QROWB;
    const uint32_t k_lane = (uint32_t) (lane / 16 * 8 + lane % 8) * ROWB;
    const uint32_t v_lane = (uint32_t) (lane / 8 % 2 * 8 + lane % 8) * ROWB;
    uint32_t q_x[4], k_x[4], v_x[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        q_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
        k_x[r] = ((2 * r + lane / 8 % 2) ^ (lane % 8)) << 4;
        v_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
    }

    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        // Whether any row of the warp sees a key of the tile, and whether all see all.
        const bool active = w_rows && k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[8][4];      // S: keys 8 n + 2 tq + {0, 1}, rows gq ([0], [1]) and gq + 8
        uint32_t pa[4][4];  // P as A fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            if (!active) {
                continue;
            }
            const uint32_t st = rbase + (uint32_t) (j % NS) * CHUNK;
            if (part < NC) {
                // S += Q K^T over this chunk's dims: one MMA chain per score over all D / 16
                // k-steps, from zero.
                if (part == 0) {
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
                    }
                }
#pragma unroll
                for (int kl = 0; kl < KC; kl++) {
                    const int ks = part * KC + kl;
                    uint32_t a[4];
                    ldsm(a, qbase + q_lane + q_x[ks % 4] + ks / 4 * 128);
#pragma unroll
                    for (int np = 0; np < 4; np++) {
                        uint32_t b[4];
                        ldsm(b, st + k_lane + k_x[kl % 4] + kl / 4 * 128 + np * 16 * ROWB);
                        mma(s[2 * np], a, b);
                        mma(s[2 * np + 1], a, b + 2);
                    }
                }
                if (part == NC - 1) {
                    // Online softmax per row; element e is key 8 n + 2 tq + e % 2, row
                    // gq + 8 (e / 2).
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int n = 0; n < 8; n++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float x = s[n][e] * scale;
                            if (!whole) {
                                const int key = k0 + n * 8 + 2 * tq + e % 2;
                                x = key <= c_pos[e / 2] && key >= c_first[e / 2] ? x : -INFINITY;
                            }
                            s[n][e] = x;
                            mx[e / 2] = fmaxf(mx[e / 2], x);
                        }
                    }
#pragma unroll
                    for (int r = 0; r < 2; r++) {
#pragma unroll
                        for (int off = 1; off < 4; off <<= 1) {
                            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], off));
                        }
                        const float m_new = fmaxf(m[r], mx[r]);
                        // No visible key yet: exp against 0 gives 0 for the masked scores.
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[r] = exp2f(m[r] - m_use);
                        m[r]  = m_new;
                        mx[r] = m_use;
                    }
                    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
                    for (int n = 0; n < 8; n++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            s[n][e] = exp2f(s[n][e] - mx[e / 2]);
                            sum[e / 2][e % 2] += s[n][e];
                        }
                    }
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        pa[ks][0] = pack(s[2 * ks][0], s[2 * ks][1]);
                        pa[ks][1] = pack(s[2 * ks][2], s[2 * ks][3]);
                        pa[ks][2] = pack(s[2 * ks + 1][0], s[2 * ks + 1][1]);
                        pa[ks][3] = pack(s[2 * ks + 1][2], s[2 * ks + 1][3]);
                    }
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        l[r][0] = l[r][0] * alpha[r] + sum[r][0];
                        l[r][1] = l[r][1] * alpha[r] + sum[r][1];
                    }
                }
            } else {
                // O = O alpha + P V for this chunk's output dims, 16 at a time.
                const int c = part - NC;
#pragma unroll
                for (int np = 0; np < CH / 16; np++) {
                    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t b[4];
                        ldsm_t(b, st + v_lane + v_x[np % 4] + np / 4 * 128 + ks * 16 * ROWB);
                        mma(acc[0], pa[ks], b);
                        mma(acc[1], pa[ks], b + 2);
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float * on = o[c * (CH / 8) + 2 * np + h];
                        on[0] = on[0] * alpha[0] + acc[h][0];
                        on[1] = on[1] * alpha[0] + acc[h][1];
                        on[2] = on[2] * alpha[1] + acc[h][2];
                        on[3] = on[3] * alpha[1] + acc[h][3];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int r = 0; r < 2; r++) {
        // The row's eight sums in paged_prefill_mma's tree.
        float lr = l[r][0] + l[r][1];
        lr += __shfl_xor_sync(0xffffffff, lr, 1);
        lr += __shfl_xor_sync(0xffffffff, lr, 2);
        const int row = r0 + gq + 8 * r;
        const int t = t0 + row / G;
        if (w_rows && row < QT * G && t < p.n_tok) {
            const int64_t orow = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + row % G;
            const float inv = 1.0f / lr;
            float * out = p.dst + orow * D + 2 * tq;
#pragma unroll
            for (int n = 0; n < D / 8; n++) {
                *(float2 *) (out + n * 8) = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
            }
        }
    }
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------------------------
// Heads of 512 (Gemma 4's global layers, #1181): paged_prefill_qp, warp pairs that split the head.
//
// A warp's 16 rows of O at 512 dims would be 256 F32 registers a thread, and Q's rows of a
// 128-row CTA (128 KB in F16) do not fit shared memory. So a CTA takes 64 query rows (the same
// tokens as paged_prefill_mma's 64 columns), and each 16 rows belong to a pair of warps: warp w
// (w < 4) takes dims 0..255 and warp w + 4 dims 256..511, for S's k-steps and for O alike. Each
// keeps its half of O (128 registers), the S tile of its rows, and its half of Q's rows as A
// fragments: the first 12 k-steps in registers (as paged_prefill_mma keeps its Q columns), the
// last 4 in shared memory (registers for all 16 spill O).
//
// paged_prefill_mma's S at heads of 512 is two chains of 16 MMAs from zero, dims 0..255 and
// 256..511 (its 256-dim chunks), added. Here each warp of a pair runs one of the two chains over
// the same dims. The second warp writes its chain to the pair's slot in shared memory; after a
// barrier of the pair the first adds it to its own (F32 adds commute), runs paged_prefill_qm's
// softmax on the sum and writes the weights (P's A fragments, F16), alpha and its sums to the
// slot; after a second barrier the second warp reads them and keeps its sums from them the same
// way, so both hold the same P and l, bit for bit. Each runs P V for its own half of the output
// dims. K and V come through a 2-stage ring of 64 keys x 256 dims: dims 128 c.. of each half
// (chunk c), a sub-block of 64 x 128 per half, laid out as paged_prefill_qm's chunks.
// ---------------------------------------------------------------------------------------------

template <int D> struct PairShape {
    static constexpr int NW      = 8;                              // warps per CTA
    static constexpr int RG      = NW / 2;                         // row groups, a warp pair each
    static constexpr int ROWS    = 16 * RG;                        // query rows per CTA
    static constexpr int H       = D / 2;                          // dims per warp of a pair
    static constexpr int CH      = 128;                            // dims of each half per chunk
    static constexpr int NS      = 2;                              // ring stages
    static constexpr int QR      = 12;                             // Q's k-steps in registers
    static constexpr int QSU     = (H / 16 - QR) * 2;              // 16-byte units of Q's row in shared memory
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per sub-block row
    static constexpr int SUB     = BLK * CH * (int) sizeof(half);  // bytes per half's sub-block
    static constexpr int CHUNK   = 2 * SUB;                        // bytes per chunk
    static constexpr int XWORDS  = 32;                             // 4-byte words per lane of a pair's slot
    static constexpr int XBYTES  = RG * XWORDS * 32 * 4;           // the pairs' slots
    static constexpr int QBYTES  = NW * 16 * QSU * 16;             // Q's last k-steps, 16 rows per warp
    static constexpr size_t SMEM = (size_t) NS * CHUNK + (size_t) XBYTES + (size_t) QBYTES;
};

#ifdef LEGION_PAGED_MMA
// 16 bytes to and from shared memory, and the barrier of row group rg's warp pair (named barrier
// 1 + rg, 64 threads).
__device__ __forceinline__ void st_x4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

__device__ __forceinline__ void ld_x4(uint32_t addr, uint32_t v[4]) {
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "r"(addr)
                 : "memory");
}

__device__ __forceinline__ void pair_sync(int rg) {
    asm volatile("bar.sync %0, 64;\n" ::"r"(1 + rg) : "memory");
}
#endif

template <int D, int G>
__global__ void __launch_bounds__(PairShape<D>::NW * 32, 1) paged_prefill_qp(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = PairShape<D>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, SUB = SH::SUB, CHUNK = SH::CHUNK, H = SH::H;
    constexpr int NW = SH::NW, RG = SH::RG, ROWS = SH::ROWS, QR = SH::QR, QSU = SH::QSU;
    constexpr int NC = H / CH;    // chunks of K, and of V, per key tile
    constexpr int KC = CH / 16;   // k-steps of S per K chunk
    constexpr int QT = ROWS / G;  // tokens per CTA
    static_assert(QSU <= 8 && SH::SMEM <= 99 * 1024, "Q's rows in shared memory: 8 units at most");
    constexpr int ROWB = UNITS * 16;
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t rbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last query tiles see the most keys: they go first.
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad
    const int rg = warp % RG, hf = warp / RG;  // row group, half of the head
    const int n_past = p.n_past[seq];

    // Row r's token, as in paged_prefill_qm.
    auto tok_of = [&](int r) { return min(t0 + min(r / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int r) { return n_past + tok_of(r); };
    const int k_end   = pos_of(ROWS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);

    // This warp's rows (the pair's) and this thread's two rows, gq and gq + 8.
    const int r0 = rg * 16;
    const bool w_rows = r0 < min(QT * G, (p.n_tok - t0) * G);
    const int w_pos_lo = pos_of(r0), w_pos_hi = pos_of(r0 + 15);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        c_pos[r]   = pos_of(r0 + gq + 8 * r);
        c_first[r] = first_key(c_pos[r], p.window);
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks; chunk
    // c of a tile holds dims 128 c.. of either half, half h in sub-block h.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / (2 * NC), part = j % (2 * NC);
            const int b = wk.block(i);
            const char * base = part < NC ? wk.cells(p.k, p.k_stage, i, p.k_cell) : wk.cells(p.v, p.v_stage, i, p.v_cell);
            const half * src = (const half *) base + kvh * D + (part % NC) * CH;
            const uint32_t dst = rbase + (uint32_t) (j % NS) * CHUNK;
#pragma unroll
            for (int x = threadIdx.x; x < BLK * 2 * UNITS; x += NW * 32) {
                const int r = x / (2 * UNITS), h = x / UNITS % 2, u = x % UNITS;
                cp_async16(dst + h * SUB + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + h * H + u * 8,
                           b * BLK + r < k_end);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }

    // Q of this warp's half, rounded to F16 as paged_prefill_mma rounds its Q columns. K-steps
    // below QR as A fragments (k-step ks: dims H hf + 16 ks + 2 tq + {0, 1} and + 8, rows gq and
    // gq + 8); the rest in the warp's 16 rows of QSU swizzled units in shared memory.
    const uint32_t qs_base = rbase + NS * CHUNK + SH::XBYTES + warp * 16 * QSU * 16;
    uint32_t qa[QR][4];
    {
        const float * qr[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = r0 + gq + 8 * r;
            qr[r] = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok_of(row) * p.q_nb2 +
                                     (kvh * G + row % G) * p.q_nb1) + hf * H + 2 * tq;
        }
#pragma unroll
        for (int ks = 0; ks < QR; ks++) {
            const float2 a = *(const float2 *) (qr[0] + ks * 16), b = *(const float2 *) (qr[1] + ks * 16);
            const float2 c = *(const float2 *) (qr[0] + ks * 16 + 8), d = *(const float2 *) (qr[1] + ks * 16 + 8);
            qa[ks][0] = pack(a.x, a.y);
            qa[ks][1] = pack(b.x, b.y);
            qa[ks][2] = pack(c.x, c.y);
            qa[ks][3] = pack(d.x, d.y);
        }
        for (int x = lane; x < 16 * QSU; x += 32) {
            const int r = x / QSU, u = x % QSU, row = r0 + r;
            const float * src = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok_of(row) * p.q_nb2 +
                                                 (kvh * G + row % G) * p.q_nb1) + hf * H + QR * 16 + u * 8;
            const float2 a = *(const float2 *) src, b = *(const float2 *) (src + 2);
            const float2 c = *(const float2 *) (src + 4), d = *(const float2 *) (src + 6);
            const uint4 v = make_uint4(pack(a.x, a.y), pack(b.x, b.y), pack(c.x, c.y), pack(d.x, d.y));
            asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(qs_base + fa::swz<QSU>(r, u)), "r"(v.x),
                         "r"(v.y), "r"(v.z), "r"(v.w)
                         : "memory");
        }
    }

    // O fragments of this warp's half: [n] dims H hf + 8 n + 2 tq + {0, 1}, rows gq ([0], [1])
    // and gq + 8 ([2], [3]).
    float o[H / 8][4];
#pragma unroll
    for (int n = 0; n < H / 8; n++) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const float scale = p.scale * LOG2E;

    // The pair's slot: word w of a lane at (w / 4 * 32 + lane) * 16 + w % 4 * 4, so a lane's 4
    // words at a time are one 16-byte access and a warp's are 512 contiguous bytes. The second
    // warp's chain in words 0..31 (s[n] at 4 n); then P's fragments in words 0..15 (pa[ks] at
    // 4 ks), alpha in 16..17 and the sums in 20..23.
    const uint32_t x_lane = rbase + NS * CHUNK + rg * SH::XWORDS * 32 * 4 + lane * 16;

    // ldmatrix addresses: Q's rows in shared memory, and within a half's sub-block, as
    // paged_prefill_qm's. Unit 2 r + b (b < 2) of a row sits at 16 ((2 r + b) ^ lane % 8) =
    // 16 (b ^ lane % 8) ^ 32 r, and the rows start at multiples of 128 bytes: an address is a
    // lane's base for r = 0, XOR 32 r (one register per operand instead of four offsets).
    const uint32_t q_lane = qs_base + (uint32_t) (lane / 8 % 2 * 8 + lane % 8) * (QSU * 16) + ((lane / 16 ^ lane % 8) << 4);
    const uint32_t k_lane = (uint32_t) (lane / 16 * 8 + lane % 8) * ROWB + hf * SUB + ((lane / 8 % 2 ^ lane % 8) << 4);
    const uint32_t v_lane = (uint32_t) (lane / 8 % 2 * 8 + lane % 8) * ROWB + hf * SUB + ((lane / 16 ^ lane % 8) << 4);

    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        // Whether any row of the pair sees a key of the tile, and whether all see all (the
        // same for both warps of a pair, so both reach its barrier or neither does).
        const bool active = w_rows && k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[8][4];      // S: keys 8 n + 2 tq + {0, 1}, rows gq ([0], [1]) and gq + 8
        uint32_t pa[4][4];  // P as A fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            if (!active) {
                continue;
            }
            const uint32_t st = rbase + (uint32_t) (j % NS) * CHUNK;
            if (part < NC) {
                // This half's chain of S over its dims: KS MMAs from zero.
                if (part == 0) {
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
                    }
                }
#pragma unroll
                for (int kl = 0; kl < KC; kl++) {
                    const int ks = part * KC + kl;
                    uint32_t q_s[4];
                    if (ks >= QR) {
                        ldsm(q_s, q_lane ^ (ks - QR) << 5);
                    }
                    const uint32_t * a = ks < QR ? qa[ks < QR ? ks : 0] : q_s;
#pragma unroll
                    for (int np = 0; np < 4; np++) {
                        uint32_t b[4];
                        ldsm(b, ((st + k_lane) ^ kl % 4 << 5) + kl / 4 * 128 + np * 16 * ROWB);
                        mma(s[2 * np], a, b);
                        mma(s[2 * np + 1], a, b + 2);
                    }
                }
            }
            if (part == NC - 1 && hf == 1) {
                // The second warp: its chain to the slot, then the first warp's weights back.
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    st_x4(x_lane + n * 512, __float_as_uint(s[n][0]), __float_as_uint(s[n][1]), __float_as_uint(s[n][2]),
                          __float_as_uint(s[n][3]));
                }
                pair_sync(rg);
                pair_sync(rg);
                uint32_t w[4];
#pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    ld_x4(x_lane + ks * 512, pa[ks]);
                }
                ld_x4(x_lane + 4 * 512, w);
                alpha[0] = __uint_as_float(w[0]);
                alpha[1] = __uint_as_float(w[1]);
                ld_x4(x_lane + 5 * 512, w);
                l[0][0] = l[0][0] * alpha[0] + __uint_as_float(w[0]);
                l[0][1] = l[0][1] * alpha[0] + __uint_as_float(w[1]);
                l[1][0] = l[1][0] * alpha[1] + __uint_as_float(w[2]);
                l[1][1] = l[1][1] * alpha[1] + __uint_as_float(w[3]);
            } else if (part == NC - 1) {
                {
                    // The first warp: the two halves' chains added, as paged_prefill_mma adds its
                    // two chunks' (dims 0..255 first; F32 adds commute either way).
                    pair_sync(rg);
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        uint32_t y[4];
                        ld_x4(x_lane + n * 512, y);
                        s[n][0] += __uint_as_float(y[0]);
                        s[n][1] += __uint_as_float(y[1]);
                        s[n][2] += __uint_as_float(y[2]);
                        s[n][3] += __uint_as_float(y[3]);
                    }
                    // Online softmax per row as paged_prefill_qm's.
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int n = 0; n < 8; n++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float x = s[n][e] * scale;
                            if (!whole) {
                                const int key = k0 + n * 8 + 2 * tq + e % 2;
                                x = key <= c_pos[e / 2] && key >= c_first[e / 2] ? x : -INFINITY;
                            }
                            s[n][e] = x;
                            mx[e / 2] = fmaxf(mx[e / 2], x);
                        }
                    }
#pragma unroll
                    for (int r = 0; r < 2; r++) {
#pragma unroll
                        for (int off = 1; off < 4; off <<= 1) {
                            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], off));
                        }
                        const float m_new = fmaxf(m[r], mx[r]);
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[r] = exp2f(m[r] - m_use);
                        m[r]  = m_new;
                        mx[r] = m_use;
                    }
                    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
                    for (int n = 0; n < 8; n++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            s[n][e] = exp2f(s[n][e] - mx[e / 2]);
                            sum[e / 2][e % 2] += s[n][e];
                        }
                    }
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        pa[ks][0] = pack(s[2 * ks][0], s[2 * ks][1]);
                        pa[ks][1] = pack(s[2 * ks][2], s[2 * ks][3]);
                        pa[ks][2] = pack(s[2 * ks + 1][0], s[2 * ks + 1][1]);
                        pa[ks][3] = pack(s[2 * ks + 1][2], s[2 * ks + 1][3]);
                    }
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        l[r][0] = l[r][0] * alpha[r] + sum[r][0];
                        l[r][1] = l[r][1] * alpha[r] + sum[r][1];
                    }
                    // The weights, alpha and the sums for the second warp.
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        st_x4(x_lane + ks * 512, pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3]);
                    }
                    st_x4(x_lane + 4 * 512, __float_as_uint(alpha[0]), __float_as_uint(alpha[1]), 0u, 0u);
                    st_x4(x_lane + 5 * 512, __float_as_uint(sum[0][0]), __float_as_uint(sum[0][1]),
                          __float_as_uint(sum[1][0]), __float_as_uint(sum[1][1]));
                    pair_sync(rg);
                }
            } else if (part >= NC) {
                // O = O alpha + P V for this chunk's dims of this warp's half, 16 at a time.
                const int c = part - NC;
#pragma unroll
                for (int np = 0; np < CH / 16; np++) {
                    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t b[4];
                        ldsm_t(b, ((st + v_lane) ^ np % 4 << 5) + np / 4 * 128 + ks * 16 * ROWB);
                        mma(acc[0], pa[ks], b);
                        mma(acc[1], pa[ks], b + 2);
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float * on = o[c * (CH / 8) + 2 * np + h];
                        on[0] = on[0] * alpha[0] + acc[h][0];
                        on[1] = on[1] * alpha[0] + acc[h][1];
                        on[2] = on[2] * alpha[1] + acc[h][2];
                        on[3] = on[3] * alpha[1] + acc[h][3];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int r = 0; r < 2; r++) {
        float lr = l[r][0] + l[r][1];
        lr += __shfl_xor_sync(0xffffffff, lr, 1);
        lr += __shfl_xor_sync(0xffffffff, lr, 2);
        const int row = r0 + gq + 8 * r;
        const int t = t0 + row / G;
        if (w_rows && row < QT * G && t < p.n_tok) {
            const int64_t orow = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + row % G;
            const float inv = 1.0f / lr;
            float * out = p.dst + orow * D + hf * H + 2 * tq;
#pragma unroll
            for (int n = 0; n < H / 8; n++) {
                *(float2 *) (out + n * 8) = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
            }
        }
    }
#else
    __trap();
#endif
}

// The kernel for heads of D (paged_prefill_qm at 256, paged_prefill_qp at 512) and its shape.
template <int D, int G> struct Pick {
    static constexpr bool PAIR = D == 512;
    using SH = std::conditional_t<PAIR, PairShape<D>, Shape<D>>;
    static void (*kernel())(const legion_paged_attn) {
        if constexpr (PAIR) {
            return paged_prefill_qp<D, G>;
        } else {
            return paged_prefill_qm<D, G>;
        }
    }
};

// Whether a launch takes the queries-as-M kernel: on (legion_paged_attn_set_qm), a compute
// capability 12.x device with this binary's sm_80+ code, a prefill group of MIN_ROWS rows or more
// over F16 caches, no compact cache, no host tier's pieces, no key-block split.
template <int D, int G>
bool takes(const legion_paged_attn & p) {
    return qm_on() && p.n_tok * G >= MIN_ROWS && p.k_type == LEGION_KV_F16 && p.v_type == LEGION_KV_F16 &&
           !p.compact && !p.piece_blk && !p.ml && legion_device::sm12x_device(Pick<D, G>::kernel());
}

template <int D, int G>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    using SH = typename Pick<D, G>::SH;
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    void (*k)(const legion_paged_attn) = Pick<D, G>::kernel();
    if (!legion_device::smem_opt_in(opted_in, k, (int) SH::SMEM)) {
        return -1;
    }
    constexpr int QT = SH::ROWS / G;
    k<<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), SH::NW * 32, SH::SMEM, st>>>(p);
    t_qm_launches++;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace fa_qm
