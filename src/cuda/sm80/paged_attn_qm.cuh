// paged_attn/common.cuh, compute capability 12.x (GB10, RTX 50): paged_prefill_mma's prefill with
// the queries as the MMA's M dimension (namespace fa_qm, paged_prefill_qm, #1133). The same bits.
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
// pieces, key-block splits, other head sizes, groups whose grid is faster there (pays()).

namespace fa_qm {

// Variants for benches (legion paged-qm-bench, #1133 round 17), all with the same bits; the
// serving path takes QV_MAIN only. QV_LATE: each chunk's cp.async issue after the part's first
// MMA k-step instead of right after the barrier. QV_C64 / QV_C64N4: chunks of 64 dims in 2 / 4
// ring stages. QV_PAIR (chunks of 64 dims, 2 stages): each warp pair shares its 32 rows, each
// warp computes S and the softmax of its own 16 rows as before and P V of both warps' rows over
// half of the output dims (its P fragments and alphas go to its partner through shared memory),
// so a V fragment feeds 4 MMAs instead of 2. QV_PAIR_LATE: both.
enum { QV_MAIN = 0, QV_LATE = 1, QV_C64 = 2, QV_C64N4 = 3, QV_PAIR = 4, QV_PAIR_LATE = 5, QV_COUNT = 6 };

template <int D, int V = QV_MAIN> struct Shape {
    static constexpr bool PAIR   = V == QV_PAIR || V == QV_PAIR_LATE;
    static constexpr bool LATE   = V == QV_LATE || V == QV_PAIR_LATE;
    static constexpr int NW      = 8;                                         // warps per CTA
    static constexpr int ROWS    = 16 * NW;                                   // query rows per CTA
    static constexpr int CH      = V == QV_MAIN || V == QV_LATE ? 128 : 64;   // dims per K / V chunk
    static constexpr int NS      = V == QV_C64N4 ? 4 : 2;                     // ring stages
    static constexpr int QUNITS  = D / 8;                                     // 16-byte units per Q row
    static constexpr int QBYTES  = ROWS * D * (int) sizeof(half);             // Q tile
    static constexpr int UNITS   = CH / 8;                                    // 16-byte units per chunk row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);             // bytes per chunk
    static constexpr int XWARP   = PAIR ? 4 * 32 * 16 + 32 * 8 : 0;           // a warp's P and alphas
    static constexpr size_t SMEM = (size_t) QBYTES + (size_t) NS * CHUNK + (size_t) NW * XWARP;
};

// Whether a group's grid runs faster here than on paged_prefill_mma (the bits are the same either
// way; measured on a GB10, #1133). Both kernels run one CTA per SM, ROWS query rows here against
// its 64. With few CTAs (verify steps of 2 to 64 tokens a sequence, short chunks over a long
// context) a CTA takes about as long as its keys whatever its rows, here about 1.25x as long, so
// the old kernel's twice as many smaller CTAs win while the SMs have room for them: 0.55-0.95x
// here. With full warps (prompt chunks) a row costs about 8 / 9 of the old kernel's. So both
// grids are costed as waves of CTAs over the SMs times a CTA's tokens, and this kernel takes the
// group up to 7 / 6 of the old grid's cost: there it measured 0.97-1.02x over a long context and
// 1.05-1.29x over a short one, whose causal tiles even out the waves (the context length is on
// the device only, and the choice has to hold for every step of a shape).
template <int D, int G>
bool pays(const legion_paged_attn & p) {
    constexpr int QT = Shape<D>::ROWS / G, QT_MMA = fa::Shape<D>::COLS / G;
    const int64_t sms = legion_device::sm_count();
    if (sms <= 0) {
        return false;
    }
    const int64_t per = (int64_t) p.n_head_kv * p.n_seqs;
    const int64_t waves    = ((p.n_tok + QT - 1) / QT * per + sms - 1) / sms;
    const int64_t waves_mm = ((p.n_tok + QT_MMA - 1) / QT_MMA * per + sms - 1) / sms;
    return waves * QT * 6 <= waves_mm * QT_MMA * 7;
}

template <int D, int G, int V = QV_MAIN>
__global__ void __launch_bounds__(Shape<D, V>::NW * 32, 1) paged_prefill_qm(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = Shape<D, V>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK, QU = SH::QUNITS;
    constexpr int NW = SH::NW, ROWS = SH::ROWS;
    constexpr bool PAIR = SH::PAIR, LATE = SH::LATE;
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int KC = CH / 16;   // k-steps of S per K chunk
    constexpr int QT = ROWS / G;  // tokens per CTA
    constexpr int ROWB = UNITS * 16, QROWB = QU * 16;
    // PAIR: 16-dim steps of P V per V chunk and warp (the warp's half of the chunk's dims).
    constexpr int NPW = PAIR ? CH / 32 : CH / 16;
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t qbase = (uint32_t) __cvta_generic_to_shared(smem);
    const uint32_t rbase = qbase + SH::QBYTES;
    // PAIR: warp w's P fragments (4 16-key steps x 32 lanes x 16 bytes), then its alphas (32
    // lanes x 8 bytes); at the end its rows' sums in the same place.
    unsigned char * const xmem = smem + SH::QBYTES + NS * CHUNK;

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
    // PAIR: the partner warp's rows (P V over both warps' rows), and this warp's half of the dims.
    const int r0_b = (warp ^ 1) * 16, half_d = warp & 1;
    const bool b_rows = PAIR && r0_b < min(QT * G, (p.n_tok - t0) * G);
    const int b_pos_hi = pos_of(r0_b + 15), b_first_lo = first_key(pos_of(r0_b), p.window);
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
    // PAIR: o this warp's rows and ob its partner's, [n] the warp's n-th 8 dims (dim_of).
    constexpr int NO = PAIR ? D / 16 : D / 8;
    float o[NO][4], ob[PAIR ? NO : 1][4];
#pragma unroll
    for (int n = 0; n < NO; n++) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
        if constexpr (PAIR) {
            ob[n][0] = ob[n][1] = ob[n][2] = ob[n][3] = 0.0f;
        }
    }
    // The first dim of fragment n: chunk n / (2 NPW), the warp's 16-dim step in it, 8 dims.
    auto dim_of = [&](int n) {
        if constexpr (PAIR) {
            return n / (2 * NPW) * CH + (half_d * NPW + n % (2 * NPW) / 2) * 16 + n % 2 * 8;
        } else {
            return n * 8;
        }
    };
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
        // PAIR: whether the partner's rows see a key of the tile (it wrote their P and alphas).
        const bool active_b = b_rows && k0 <= b_pos_hi && k0 + BLK - 1 >= b_first_lo;
        float s[8][4];      // S: keys 8 n + 2 tq + {0, 1}, rows gq ([0], [1]) and gq + 8
        uint32_t pa[4][4];  // P as A fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            const bool run = active || (PAIR && part >= NC && active_b);
            // LATE: a warp with work issues after its part's first MMA step.
            if (!LATE || !run) {
                issue(j + NS - 1);
            }
            if (!run) {
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
                    if (LATE && kl == 0) {
                        issue(j + NS - 1);
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
                    if constexpr (PAIR) {
                        // For the partner: the same lane holds the same fragment elements of its
                        // rows, so it reads them as they are (after the next barrier).
                        unsigned char * x = xmem + warp * SH::XWARP;
#pragma unroll
                        for (int ks = 0; ks < 4; ks++) {
                            *(uint4 *) (x + ks * 512 + lane * 16) = make_uint4(pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3]);
                        }
                        *(float2 *) (x + 2048 + lane * 8) = make_float2(alpha[0], alpha[1]);
                    }
                }
            } else {
                // O = O alpha + P V for this chunk's output dims, 16 at a time.
                const int c = part - NC;
                // PAIR: the partner's P and alphas.
                uint32_t pb[4][4];
                float alpha_b[2] = {1.0f, 1.0f};
                if constexpr (PAIR) {
                    if (active_b) {
                        const unsigned char * x = xmem + (warp ^ 1) * SH::XWARP;
#pragma unroll
                        for (int ks = 0; ks < 4; ks++) {
                            const uint4 u = *(const uint4 *) (x + ks * 512 + lane * 16);
                            pb[ks][0] = u.x;
                            pb[ks][1] = u.y;
                            pb[ks][2] = u.z;
                            pb[ks][3] = u.w;
                        }
                        const float2 ab = *(const float2 *) (x + 2048 + lane * 8);
                        alpha_b[0] = ab.x;
                        alpha_b[1] = ab.y;
                    }
                }
#pragma unroll
                for (int npl = 0; npl < NPW; npl++) {
                    const int np = PAIR ? half_d * NPW + npl : npl;
                    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
                    float accb[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t b[4];
                        ldsm_t(b, st + v_lane + v_x[np % 4] + np / 4 * 128 + ks * 16 * ROWB);
                        if (!PAIR || active) {
                            mma(acc[0], pa[ks], b);
                            mma(acc[1], pa[ks], b + 2);
                        }
                        if (PAIR && active_b) {
                            mma(accb[0], pb[ks], b);
                            mma(accb[1], pb[ks], b + 2);
                        }
                    }
                    if (LATE && npl == 0) {
                        issue(j + NS - 1);
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int n = c * (2 * NPW) + 2 * npl + h;
                        if (!PAIR || active) {
                            float * on = o[n];
                            on[0] = on[0] * alpha[0] + acc[h][0];
                            on[1] = on[1] * alpha[0] + acc[h][1];
                            on[2] = on[2] * alpha[1] + acc[h][2];
                            on[3] = on[3] * alpha[1] + acc[h][3];
                        }
                        if constexpr (PAIR) {
                            if (active_b) {
                                float * on = ob[n];
                                on[0] = on[0] * alpha_b[0] + accb[h][0];
                                on[1] = on[1] * alpha_b[0] + accb[h][1];
                                on[2] = on[2] * alpha_b[1] + accb[h][2];
                                on[3] = on[3] * alpha_b[1] + accb[h][3];
                            }
                        }
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // The rows' eight sums in paged_prefill_mma's tree.
    float lr[2], lr_b[2] = {1.0f, 1.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        lr[r] = l[r][0] + l[r][1];
        lr[r] += __shfl_xor_sync(0xffffffff, lr[r], 1);
        lr[r] += __shfl_xor_sync(0xffffffff, lr[r], 2);
    }
    if constexpr (PAIR) {
        // The partner's sums, through its exchange area once every warp is past the loop.
        __syncthreads();
        *(float2 *) (xmem + warp * SH::XWARP + lane * 8) = make_float2(lr[0], lr[1]);
        __syncthreads();
        const float2 lb = *(const float2 *) (xmem + (warp ^ 1) * SH::XWARP + lane * 8);
        lr_b[0] = lb.x;
        lr_b[1] = lb.y;
    }
    // One warp's rows from r0w: its fragments ow and sums lw.
    auto store = [&](const float (*ow)[4], const float * lw, int r0w, bool rows) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = r0w + gq + 8 * r;
            const int t = t0 + row / G;
            if (rows && row < QT * G && t < p.n_tok) {
                const int64_t orow = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + row % G;
                const float inv = 1.0f / lw[r];
                float * out = p.dst + orow * D + 2 * tq;
#pragma unroll
                for (int n = 0; n < NO; n++) {
                    *(float2 *) (out + dim_of(n)) = make_float2(ow[n][2 * r] * inv, ow[n][2 * r + 1] * inv);
                }
            }
        }
    };
    store(o, lr, r0, w_rows);
    if constexpr (PAIR) {
        store(ob, lr_b, r0_b, b_rows);
    }
#else
    __trap();
#endif
}

// Whether a launch takes paged_prefill_qm: on (legion_paged_attn_set_qm), a compute capability
// 12.x device with this binary's sm_80+ code, a multi-token group over F16 caches whose grid pays
// (or any, in mode 2: benches), no compact cache, no host tier's pieces, no key-block split.
template <int D, int G>
bool takes(const legion_paged_attn & p) {
    const int mode = qm_mode();
    return mode > 0 && p.k_type == LEGION_KV_F16 && p.v_type == LEGION_KV_F16 && !p.compact && !p.piece_blk && !p.ml &&
           legion_device::sm12x_device(paged_prefill_qm<D, G>) && (mode >= 2 || pays<D, G>(p));
}

template <int D, int G, int V>
int launch_as(const legion_paged_attn & p, cudaStream_t st) {
    using SH = Shape<D, V>;
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_qm<D, G, V>, (int) SH::SMEM)) {
        return -1;
    }
    constexpr int QT = SH::ROWS / G;
    paged_prefill_qm<D, G, V><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), SH::NW * 32, SH::SMEM, st>>>(p);
    t_qm_launches++;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// The variant (qm_variant(), benches only) for 4 and 6 query heads a kv head (Qwen3.8-27B's 6),
// else QV_MAIN.
template <int D, int G>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    if constexpr (G == 4 || G == 6) {
        switch (qm_variant()) {
            case QV_LATE: return launch_as<D, G, QV_LATE>(p, st);
            case QV_C64: return launch_as<D, G, QV_C64>(p, st);
            case QV_C64N4: return launch_as<D, G, QV_C64N4>(p, st);
            case QV_PAIR: return launch_as<D, G, QV_PAIR>(p, st);
            case QV_PAIR_LATE: return launch_as<D, G, QV_PAIR_LATE>(p, st);
            default: break;
        }
    }
    return launch_as<D, G, QV_MAIN>(p, st);
}

}  // namespace fa_qm
