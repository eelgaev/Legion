// paged_attn/common.cuh, sm_80 and newer: paged_prefill_mma's prefill for small groups of heads of
// 256 at 6 query heads a kv head on a wider grid of warp pairs (namespace fa_grid,
// paged_prefill_grid, #689). The same bits.
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after
// sm80/paged_attn.cuh (namespace fa: swz, cp_async16, ldsm, ldsm_t, mma, transpose, pack).
//
// paged_prefill_mma gives a CTA 64 query columns (column = token * G + head in the group) of one
// kv head. A speculative verify step of one sequence at 6 query heads a kv head (Qwen3.8-27B: 8
// tokens, 48 columns) is then one CTA per kv head, 4 CTAs on a GB10's 48 SMs, each running 8
// warps' MMA chains (2 of them on padding columns) over the whole context.
//
// Here a CTA takes NG groups of 8 consecutive columns of the (sequence, kv head)'s n_tok * G, not
// a tile of whole tokens, so the grid has ceil(n_tok G / 8 NG) CTAs per kv head and no warp
// computes padding past the last column. A group is a warp pair, as in paged_prefill_pair
// (sm80/paged_attn_pair.cuh): warp h takes the 16-key blocks 2 h and 2 h + 1 of a tile's scores
// and the output dims h * 128 .. + 127; the pair swaps its column maxima, the first half's sums and
// its halves of P through 768 bytes of shared memory behind a 64-thread barrier. Every CTA streams
// its kv head's keys through the same ring of 64 keys x 256 dims, three stages. Measured on a GB10
// (1 sequence, 8 tokens, 8K keys): 1-warp groups 0.87x, 2 or 3 groups of a warp each per CTA
// 1.2-1.35x, pairs of 1, 2, 3 groups per CTA 1.6x, 1.8x, 1.45x: the call is bound by each CTA's
// MMA chains and fill, not by the SMs it leaves idle, and halving each warp's chain is the win.
//
// paged_prefill_quad (below) halves it again: a group on 4 warps, one key block each, so 1
// sequence of 8 tokens is 24 CTAs of 4 warps. On a GB10 it is 1.15x the pair kernel at 1 sequence
// of 2 to 8 tokens (8K-119K keys), even at 2 sequences, and slower than the pair once its grid
// passes half the SMs (1 sequence of 16 tokens: 48 CTAs, 0.95x), so it only takes grids up to
// half the SMs (paged::verify_grid_kind).
//
// Every value goes through the same operations in the same order as in paged_prefill_mma:
// - A score is one chain of 16 MMAs from zero; scale and mask as there. A column's maximum is the
//   maximum of the same values (exact in any order).
// - The thread's share of a column's sum adds the weights of key blocks 0 to 3 in order (warp 0
//   blocks 0 and 1 from zero, warp 1 goes on from its partial sum with 2 and 3), then (l, alpha)
//   as there; the epilogue's lane tree is the same.
// - Each 16 output dims' P V is one chain of 4 MMAs from zero over the tile's 16-key steps with the
//   same P fragments, added as o alpha + acc.
// A CTA walks the key blocks its own columns see; a tile none of a group's columns see is skipped,
// and one a column does not see adds nothing to it (alpha 1, zero weights), so the smaller tiles
// leave the bits alone.
//
// Not covered (paged_prefill_mma instead): other head sizes or GQA ratios, the compact cache,
// key-block splits.

namespace fa_grid {

template <int D, int NG> struct Shape {
    static_assert(D == 256, "the wide-grid kernel is for heads of 256");
    static constexpr int NW      = 2 * NG;                         // warps per CTA
    static constexpr int COLS    = 8 * NG;                         // query columns per CTA
    static constexpr int CH      = 256;                            // dims per chunk
    static constexpr int NS      = 3;                              // ring stages
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);  // bytes per chunk
    static constexpr int RING    = NS * CHUNK;
    static constexpr int XB      = 768;  // a pair's exchange: 512 B of P (or maxima), 256 B of sums
    static constexpr size_t SMEM = (size_t) RING + (size_t) NG * XB;
};

// Column groups per CTA: 2 (the fastest of 1, 2 and 3 above).
constexpr int GROUPS = 2;

#ifdef LEGION_PAGED_MMA
// The 64 threads of warp pair `id - 1`.
__device__ __forceinline__ void grid_pair_sync(int id) {
    asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory");
}
#endif

template <int D, int G, int NG>
__global__ void __launch_bounds__(NG * 64, 1) paged_prefill_grid(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = Shape<D, NG>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    using fa::transpose;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK;
    constexpr int NW = SH::NW, COLS = SH::COLS;
    constexpr int NC = D / CH;          // chunks of K, and of V, per key tile
    constexpr int KS = D / 16;          // k-steps of K Q^T
    constexpr int HB = CH / 32;         // a warp's 16-dim output blocks per V chunk
    constexpr int OB = NC * HB;         // a warp's 16-dim output blocks
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last columns see the most keys: they go first.
    const int cols = p.n_tok * G;
    const int c0 = (gridDim.x - 1 - blockIdx.x) * COLS, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int grp = warp / 2, h = warp % 2;   // column group, half of the pair
    const int gq = lane / 4, tq = lane % 4;   // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Column c's token. Columns past the last repeat it and are not written.
    auto tok_of = [&](int c) { return min(c, cols - 1) / G; };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(c0 + COLS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(c0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    // This group's columns (for skipping and unmasked tiles; a group past the last column computes
    // nothing) and this thread's two columns of the S^T / O^T fragments, 2 tq + {0, 1}.
    const int wc = c0 + grp * 8;
    const bool w_cols = wc < cols;
    const int w_pos_lo = pos_of(wc), w_pos_hi = pos_of(wc + 7);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        c_pos[j]   = pos_of(wc + 2 * tq + j);
        c_first[j] = first_key(c_pos[j], p.window);
    }

    // Q as B fragments (k = dims, n = columns): column gq, dims 2 tq + {0, 1} and + 8.
    uint32_t qf[KS][2];
    {
        const int c = min(wc + gq, cols - 1);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + (c / G) * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            qf[ks][0] = pack(a.x, a.y);
            qf[ks][1] = pack(b.x, b.y);
        }
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / (2 * NC), part = j % (2 * NC);
            const int b = wk.block(i);
            const int ty = part < NC ? p.k_type : p.v_type;
            const int64_t cell_bytes = part < NC ? p.k_cell : p.v_cell;
            const char * base = part < NC ? wk.cells(p.k, p.k_stage, i, cell_bytes) : wk.cells(p.v, p.v_stage, i, cell_bytes);
            const int e0 = head_at<D, false>(p, kvh) + (part % NC) * CH;
            const uint32_t dst = sbase + (uint32_t) (j % NS) * CHUNK;
            if (ty == LEGION_KV_F16) {
                const half * src = (const half *) base + e0;
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    cp_async16(dst + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + u * 8, b * BLK + r < k_end);
                }
            } else {
                // Quantized: dequantized to F16 on the way in (synchronous loads).
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    uint4 x = make_uint4(0, 0, 0, 0);
                    if (b * BLK + r < k_end) {
                        float f[8];
                        load_row<8>(base + r * cell_bytes, ty, e0 + u * 8, f);
                        x = make_uint4(pack(f[0], f[1]), pack(f[2], f[3]), pack(f[4], f[5]), pack(f[6], f[7]));
                    }
                    *(uint4 *) (smem + (dst - sbase) + fa::swz<UNITS>(r, u)) = x;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // This warp's O^T blocks: block ob is dims 16 (ob / HB * CH / 16 + h HB + ob % HB) + gq (+ 8
    // for [2], [3]), columns 2 tq + {0, 1}.
    float o[OB][4];
#pragma unroll
    for (int i = 0; i < OB; i++) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    }
    // Running max (log2 units) per column; this thread's share of the sum (kept by warp 1).
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float scale = p.scale * LOG2E;

    // The pair's exchange. xp: a lane's uint4 of P fragments (warp 1's, then warp 0's), or warp
    // 0's column maxima; xs: warp 0's partial sums, a lane's float2, or warp 1's column maxima, or
    // at the end warp 1's sums.
    unsigned char * xb = smem + SH::RING + grp * SH::XB;
    uint4 * xp = (uint4 *) xb;
    float * xs = (float *) (xb + 512);
    float * mx_mine = h == 0 ? (float *) xp : xs;
    const float * mx_other = h == 0 ? xs : (const float *) xp;
    const int bar = 1 + grp;

    // ldmatrix addresses, as in paged_prefill_mma.
    constexpr int ROWB = UNITS * 16;
    const uint32_t k_lane = (lane / 8 % 2 * 8 + lane % 8) * ROWB, v_lane = (lane / 16 * 8 + lane % 8) * ROWB;
    uint32_t k_x[4], v_x[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        k_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
        v_x[r] = ((2 * r + lane / 8 % 2) ^ (lane % 8)) << 4;
    }

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        // Whether any column of the group sees a key of the tile, and whether all see all.
        const bool active = w_cols && k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[2][4];      // S^T: keys 16 (2 h + ii) + gq (+ 8), columns 2 tq + {0, 1}
        uint32_t pf[4][2];  // P as B fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            if (!active) {
                continue;
            }
            const uint32_t st = sbase + (uint32_t) (j % NS) * CHUNK;
            if (part < NC) {
                // S^T += K Q^T over this chunk's dims, this warp's key blocks.
#pragma unroll
                for (int ii = 0; ii < 2; ii++) {
                    const int i = 2 * h + ii;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < CH / 16; ks++) {
                        uint32_t a[4];
                        ldsm(a, st + k_lane + k_x[ks % 4] + i * 16 * ROWB + ks / 4 * 128);
                        mma(acc, a, qf[part * (CH / 16) + ks]);
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[ii][e] = part == 0 ? acc[e] : s[ii][e] + acc[e];
                    }
                }
                if (part == NC - 1) {
                    // Online softmax per column; element e is key 16 (2 h + ii) + gq + 8 (e / 2),
                    // column 2 tq + e % 2.
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float x = s[ii][e] * scale;
                            if (!whole) {
                                const int key = k0 + (2 * h + ii) * 16 + gq + 8 * (e / 2);
                                x = key <= c_pos[e % 2] && key >= c_first[e % 2] ? x : -INFINITY;
                            }
                            s[ii][e] = x;
                            mx[e % 2] = fmaxf(mx[e % 2], x);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < 2; c++) {
#pragma unroll
                        for (int off = 4; off < 32; off <<= 1) {
                            mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], off));
                        }
                    }
                    // The other warp's half of the tile.
                    if (gq == 0) {
                        mx_mine[2 * tq]     = mx[0];
                        mx_mine[2 * tq + 1] = mx[1];
                    }
                    grid_pair_sync(bar);
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        mx[c] = fmaxf(mx[c], mx_other[2 * tq + c]);
                    }
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        const float m_new = fmaxf(m[c], mx[c]);
                        // No visible key yet: exp against 0 gives 0 for the masked scores.
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[c] = exp2f(m[c] - m_use);
                        m[c]  = m_new;
                        mx[c] = m_use;
                    }
                    uint32_t pm[2][2];  // this warp's P fragments, key blocks 2 h + ii
#pragma unroll
                    for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            s[ii][e] = exp2f(s[ii][e] - mx[e % 2]);
                        }
                        pm[ii][0] = transpose(pack(s[ii][0], s[ii][1]));
                        pm[ii][1] = transpose(pack(s[ii][2], s[ii][3]));
                    }
                    // Warp 0: its sum of key blocks 0 and 1 to warp 1, which goes on with 2 and 3;
                    // warp 1: its P to warp 0, then warp 0's P to warp 1.
                    float sum[2] = {0.0f, 0.0f};
                    if (h == 0) {
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                sum[e % 2] += s[ii][e];
                            }
                        }
                        ((float2 *) xs)[lane] = make_float2(sum[0], sum[1]);
                    } else {
                        xp[lane] = make_uint4(pm[0][0], pm[0][1], pm[1][0], pm[1][1]);
                    }
                    grid_pair_sync(bar);
                    if (h == 0) {
                        const uint4 v = xp[lane];
                        xp[lane]      = make_uint4(pm[0][0], pm[0][1], pm[1][0], pm[1][1]);
                        pf[0][0] = pm[0][0];
                        pf[0][1] = pm[0][1];
                        pf[1][0] = pm[1][0];
                        pf[1][1] = pm[1][1];
                        pf[2][0] = v.x;
                        pf[2][1] = v.y;
                        pf[3][0] = v.z;
                        pf[3][1] = v.w;
                    } else {
                        const float2 q = ((const float2 *) xs)[lane];
                        sum[0] = q.x;
                        sum[1] = q.y;
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                sum[e % 2] += s[ii][e];
                            }
                        }
                        l[0] = l[0] * alpha[0] + sum[0];
                        l[1] = l[1] * alpha[1] + sum[1];
                        pf[2][0] = pm[0][0];
                        pf[2][1] = pm[0][1];
                        pf[3][0] = pm[1][0];
                        pf[3][1] = pm[1][1];
                    }
                    grid_pair_sync(bar);
                    if (h == 1) {
                        const uint4 v = xp[lane];
                        pf[0][0] = v.x;
                        pf[0][1] = v.y;
                        pf[1][0] = v.z;
                        pf[1][1] = v.w;
                    }
                }
            } else {
                // O^T = O^T alpha + V^T P for this warp's output dims of this chunk.
                const int c = part - NC;
#pragma unroll
                for (int ib = 0; ib < HB; ib++) {
                    const int i = h * HB + ib;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_lane + v_x[i % 4] + ks * 16 * ROWB + i / 4 * 128);
                        mma(acc, a, pf[ks]);
                    }
                    float * oi = o[c * HB + ib];
                    oi[0] = oi[0] * alpha[0] + acc[0];
                    oi[1] = oi[1] * alpha[1] + acc[1];
                    oi[2] = oi[2] * alpha[0] + acc[2];
                    oi[3] = oi[3] * alpha[1] + acc[3];
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // The sums' lane tree as in paged_prefill_mma (warp 1's, then to warp 0).
    if (h == 1) {
#pragma unroll
        for (int c = 0; c < 2; c++) {
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                l[c] += __shfl_xor_sync(0xffffffff, l[c], off);
            }
        }
        if (gq == 0) {
            xs[2 * tq]     = l[0];
            xs[2 * tq + 1] = l[1];
        }
    }
    grid_pair_sync(bar);
    if (h == 0) {
        l[0] = xs[2 * tq];
        l[1] = xs[2 * tq + 1];
    }

#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int col = wc + 2 * tq + c;
        const bool valid = col < cols;
        const int64_t row = ((int64_t) seq * p.n_tok + col / G) * p.n_head + kvh * G + col % G;
        float * out = p.dst + row * D + gq;
        if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. The column's lanes (both
            // warps of the pair) read its running (max, sum) before one of them writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[c], l[c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            grid_pair_sync(bar);
            if (valid) {
                if (h == 0 && gq == 0) {
                    store_ml(p, row, f);
                }
                const float inv = l[c] > 0.0f ? 1.0f / l[c] : 0.0f;
#pragma unroll
                for (int ob = 0; ob < OB; ob++) {
                    const int i = ob / HB * (CH / 16) + h * HB + ob % HB;
                    const float x0 = o[ob][c] * inv, x1 = o[ob][2 + c] * inv;
                    out[i * 16]     = p.piece > 0 ? f.x * out[i * 16] + f.y * x0 : x0;
                    out[i * 16 + 8] = p.piece > 0 ? f.x * out[i * 16 + 8] + f.y * x1 : x1;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / l[c];
#pragma unroll
            for (int ob = 0; ob < OB; ob++) {
                const int i = ob / HB * (CH / 16) + h * HB + ob % HB;
                out[i * 16]     = o[ob][c] * inv;
                out[i * 16 + 8] = o[ob][2 + c] * inv;
            }
        }
    }
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------------------------
// paged_prefill_quad: a group of 8 columns on 4 warps (#689 round 17). Warp h takes key block h
// of each tile's scores (one chain of D / 16 MMAs, carried over the tile's K chunks) and the
// output dims of blocks h HB .. + HB - 1 of every V chunk, so a warp's share of a tile is a
// quarter of paged_prefill_mma's and half of the pair's. Per tile and group:
// - each warp writes its column maxima (after the lane tree over its block) to shared memory;
//   after a 128-thread barrier every warp takes the maximum of the four (the same value in any
//   order), so all four keep the same running max and alpha;
// - each warp writes its block's P fragments, warps 0-2 also their F32 weights; after a second
//   barrier every warp reads the four blocks' P for its V^T P, and warp 3 adds the weights of
//   blocks 0, 1, 2 (read back) and its own in key order from zero, then l = l alpha + sum, as in
//   paged_prefill_mma; warp 3 alone keeps l and hands it to the others at the end.
// The ring is CH dims x 64 keys a chunk, NS stages; with CH = 128 a key block's score chain runs
// over both K chunks of the tile in the same order. The same bits as paged_prefill_mma.
template <int D, int NG, int CH_, int NS_> struct QShape {
    static_assert(D == 256, "the quad kernel is for heads of 256");
    static constexpr int NW      = 4 * NG;                         // warps per CTA
    static constexpr int COLS    = 8 * NG;                         // query columns per CTA
    static constexpr int CH      = CH_;                            // dims per chunk
    static constexpr int NS      = NS_;                            // ring stages
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);  // bytes per chunk
    static constexpr int RING    = NS * CHUNK;
    // A group's exchange: P of 4 blocks (4 x 32 lanes x 8 B), warps 0-2's weights (3 x 32 x
    // 16 B), the 4 warps' column maxima (4 x 8 floats), the final sums (8 floats).
    static constexpr int XP      = 1024;
    static constexpr int XW      = 1536;
    static constexpr int XM      = 128;
    static constexpr int XB      = XP + XW + XM + 32;
    static constexpr size_t SMEM = (size_t) RING + (size_t) NG * XB;
};

#ifdef LEGION_PAGED_MMA
// The 128 threads of column group `id - 1`.
__device__ __forceinline__ void grid_quad_sync(int id) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(id) : "memory");
}
#endif

// PF: the first line of every K and V block of the CTA's keys prefetched into L2 at the start
// (#689 r19, below).
// SPLIT: the fixed split over the key blocks (legion_paged_attn::split_fixed, #1133): the grid's z
// counts (sequence, split), a CTA walks one run of its keys (fixed_split) and each column goes to
// dst or to its run's scratch slot (fixed_target), as in paged_prefill_mma<.., SPLIT>.
template <int D, int G, int NG, int CH_, int NS_, bool PF = true, bool SPLIT = false>
__global__ void __launch_bounds__(NG * 128, 1) paged_prefill_quad(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = QShape<D, NG, CH_, NS_>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    using fa::transpose;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK;
    constexpr int NW = SH::NW, COLS = SH::COLS;
    constexpr int NC = D / CH;          // chunks of K, and of V, per key tile
    constexpr int KS = D / 16;          // k-steps of K Q^T
    constexpr int HB = CH / 64;         // a warp's 16-dim output blocks per V chunk
    constexpr int OB = NC * HB;         // a warp's 16-dim output blocks
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last columns see the most keys: they go first.
    const int cols = p.n_tok * G;
    const int c0 = (gridDim.x - 1 - blockIdx.x) * COLS, kvh = blockIdx.y;
    const int seq = SPLIT ? blockIdx.z / p.max_split : blockIdx.z;
    const int split = SPLIT ? blockIdx.z % p.max_split : 0;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int grp = warp / 4, h = warp % 4;   // column group, key block / quarter of the group
    const int gq = lane / 4, tq = lane % 4;   // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Column c's token. Columns past the last repeat it and are not written.
    auto tok_of = [&](int c) { return min(c, cols - 1) / G; };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(c0 + COLS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(c0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    // SPLIT: this CTA's run of the keys (fixed_split) and its scratch slot.
    int w_b0 = b_first, w_b1 = b_last, slot = 0;
    if constexpr (SPLIT) {
        slot = b_first / p.split_fixed + split;
        if (split >= fa::fixed_split(p.split_fixed, split, &w_b0, &w_b1)) {
            return;
        }
    }
    const Walk wk = walk(p, seq, w_b0, w_b1);
    if (piece_idle(p, wk)) {
        return;
    }

    const int wc = c0 + grp * 8;
    const bool w_cols = wc < cols;
    const int w_pos_lo = pos_of(wc), w_pos_hi = pos_of(wc + 7);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        c_pos[j]   = pos_of(wc + 2 * tq + j);
        c_first[j] = first_key(c_pos[j], p.window);
    }

    // Q as B fragments (k = dims, n = columns): column gq, dims 2 tq + {0, 1} and + 8.
    uint32_t qf[KS][2];
    {
        const int c = min(wc + gq, cols - 1);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + (c / G) * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            qf[ks][0] = pack(a.x, a.y);
            qf[ks][1] = pack(b.x, b.y);
        }
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / (2 * NC), part = j % (2 * NC);
            const int b = wk.block(i);
            const int ty = part < NC ? p.k_type : p.v_type;
            const int64_t cell_bytes = part < NC ? p.k_cell : p.v_cell;
            const char * base = part < NC ? wk.cells(p.k, p.k_stage, i, cell_bytes) : wk.cells(p.v, p.v_stage, i, cell_bytes);
            const int e0 = head_at<D, false>(p, kvh) + (part % NC) * CH;
            const uint32_t dst = sbase + (uint32_t) (j % NS) * CHUNK;
            if (ty == LEGION_KV_F16) {
                const half * src = (const half *) base + e0;
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    cp_async16(dst + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + u * 8, b * BLK + r < k_end);
                }
            } else {
                // Quantized: dequantized to F16 on the way in (synchronous loads).
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    uint4 x = make_uint4(0, 0, 0, 0);
                    if (b * BLK + r < k_end) {
                        float f[8];
                        load_row<8>(base + r * cell_bytes, ty, e0 + u * 8, f);
                        x = make_uint4(pack(f[0], f[1]), pack(f[2], f[3]), pack(f[4], f[5]), pack(f[6], f[7]));
                    }
                    *(uint4 *) (smem + (dst - sbase) + fa::swz<UNITS>(r, u)) = x;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // This warp's O^T blocks: block ob is dims 16 (ob / HB * CH / 16 + h HB + ob % HB) + gq (+ 8
    // for [2], [3]), columns 2 tq + {0, 1}.
    float o[OB][4];
#pragma unroll
    for (int i = 0; i < OB; i++) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    }
    // Running max (log2 units) per column; this thread's share of the sum (kept by warp 3).
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float scale = p.scale * LOG2E;

    unsigned char * xb = smem + SH::RING + grp * SH::XB;
    uint2 * xp  = (uint2 *) xb;                                // [block][lane] P fragments
    float4 * xw = (float4 *) (xb + SH::XP);                    // [block 0-2][lane] weights
    float * xm  = (float *) (xb + SH::XP + SH::XW);            // [warp][column] maxima
    float * xl  = (float *) (xb + SH::XP + SH::XW + SH::XM);   // [column] final sums
    const int bar = 1 + grp;

    // ldmatrix addresses, as in paged_prefill_mma.
    constexpr int ROWB = UNITS * 16;
    const uint32_t k_lane = (lane / 8 % 2 * 8 + lane % 8) * ROWB, v_lane = (lane / 16 * 8 + lane % 8) * ROWB;
    uint32_t k_x[4], v_x[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        k_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
        v_x[r] = ((2 * r + lane / 8 % 2) ^ (lane % 8)) << 4;
    }

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }
    if constexpr (PF) {
        // While the ring's first loads are in flight: one L2 prefetch of the first line of every
        // K and V block the CTA will read (2 lines a 64-key tile, ~0.1% of its bytes), so every
        // page of its keys is translated long before the loop reaches it. In a model each call
        // comes after other layers' traffic (weights, other layers' K/V), and without this the
        // loop met each new page's translation cold: on a GB10 at 64K:8 in Qwen3.8-27B's verify
        // step 1.70 -> 1.47 ms a call (nsys medians), 1.45 -> 1.37 in a bench repeating one call.
        // Reads nothing into registers and writes nothing: the same bits.
        const int n = (wk.i1 - wk.i0) * 2;
#pragma unroll 4
        for (int x = threadIdx.x; x < n; x += NG * 128) {
            const int    i = wk.i0 + x / 2;
            const char * a = x % 2 == 0 ? wk.cells(p.k, p.k_stage, i, p.k_cell) : wk.cells(p.v, p.v_stage, i, p.v_cell);
            asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a));
        }
    }
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        // Whether any column of the group sees a key of the tile, and whether all see all.
        const bool active = w_cols && k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // S^T: keys 16 h + gq (+ 8), columns 2 tq + {0, 1}
        uint32_t pf[4][2];                      // P as B fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            if (!active) {
                continue;
            }
            const uint32_t st = sbase + (uint32_t) (j % NS) * CHUNK;
            if (part < NC) {
                // S^T += K Q^T over this chunk's dims, key block h: one chain over the tile.
#pragma unroll
                for (int ks = 0; ks < CH / 16; ks++) {
                    uint32_t a[4];
                    ldsm(a, st + k_lane + k_x[ks % 4] + h * 16 * ROWB + ks / 4 * 128);
                    mma(s, a, qf[part * (CH / 16) + ks]);
                }
                if (part == NC - 1) {
                    // Online softmax per column; element e is key 16 h + gq + 8 (e / 2), column
                    // 2 tq + e % 2.
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float x = s[e] * scale;
                        if (!whole) {
                            const int key = k0 + h * 16 + gq + 8 * (e / 2);
                            x = key <= c_pos[e % 2] && key >= c_first[e % 2] ? x : -INFINITY;
                        }
                        s[e] = x;
                        mx[e % 2] = fmaxf(mx[e % 2], x);
                    }
#pragma unroll
                    for (int c = 0; c < 2; c++) {
#pragma unroll
                        for (int off = 4; off < 32; off <<= 1) {
                            mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], off));
                        }
                    }
                    if (gq == 0) {
                        xm[h * 8 + 2 * tq]     = mx[0];
                        xm[h * 8 + 2 * tq + 1] = mx[1];
                    }
                    grid_quad_sync(bar);
#pragma unroll
                    for (int c = 0; c < 2; c++) {
#pragma unroll
                        for (int w = 0; w < 4; w++) {
                            mx[c] = fmaxf(mx[c], xm[w * 8 + 2 * tq + c]);
                        }
                        const float m_new = fmaxf(m[c], mx[c]);
                        // No visible key yet: exp against 0 gives 0 for the masked scores.
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[c] = exp2f(m[c] - m_use);
                        m[c]  = m_new;
                        mx[c] = m_use;
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[e] = exp2f(s[e] - mx[e % 2]);
                    }
                    const uint32_t p0 = transpose(pack(s[0], s[1])), p1 = transpose(pack(s[2], s[3]));
                    xp[h * 32 + lane] = make_uint2(p0, p1);
                    if (h < 3) {
                        xw[h * 32 + lane] = make_float4(s[0], s[1], s[2], s[3]);
                    }
                    grid_quad_sync(bar);
#pragma unroll
                    for (int w = 0; w < 4; w++) {
                        if (w == h) {
                            pf[w][0] = p0;
                            pf[w][1] = p1;
                        } else {
                            const uint2 v = xp[w * 32 + lane];
                            pf[w][0] = v.x;
                            pf[w][1] = v.y;
                        }
                    }
                    if (h == 3) {
                        // The weights of key blocks 0 to 3 in order, from zero.
                        float sum[2] = {0.0f, 0.0f};
#pragma unroll
                        for (int w = 0; w < 3; w++) {
                            const float4 v = xw[w * 32 + lane];
                            sum[0] += v.x;
                            sum[1] += v.y;
                            sum[0] += v.z;
                            sum[1] += v.w;
                        }
                        sum[0] += s[0];
                        sum[1] += s[1];
                        sum[0] += s[2];
                        sum[1] += s[3];
                        l[0] = l[0] * alpha[0] + sum[0];
                        l[1] = l[1] * alpha[1] + sum[1];
                    }
                }
            } else {
                // O^T = O^T alpha + V^T P for this warp's output dims of this chunk.
                const int c = part - NC;
#pragma unroll
                for (int ib = 0; ib < HB; ib++) {
                    const int i = h * HB + ib;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_lane + v_x[i % 4] + ks * 16 * ROWB + i / 4 * 128);
                        mma(acc, a, pf[ks]);
                    }
                    float * oi = o[c * HB + ib];
                    oi[0] = oi[0] * alpha[0] + acc[0];
                    oi[1] = oi[1] * alpha[1] + acc[1];
                    oi[2] = oi[2] * alpha[0] + acc[2];
                    oi[3] = oi[3] * alpha[1] + acc[3];
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // The sums' lane tree as in paged_prefill_mma (warp 3's, then to the others).
    if (h == 3) {
#pragma unroll
        for (int c = 0; c < 2; c++) {
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                l[c] += __shfl_xor_sync(0xffffffff, l[c], off);
            }
        }
        if (gq == 0) {
            xl[2 * tq]     = l[0];
            xl[2 * tq + 1] = l[1];
        }
    }
    grid_quad_sync(bar);
    if (h != 3) {
        l[0] = xl[2 * tq];
        l[1] = xl[2 * tq + 1];
    }

#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int col = wc + 2 * tq + c;
        const bool valid = col < cols;
        const int64_t row = ((int64_t) seq * p.n_tok + col / G) * p.n_head + kvh * G + col % G;
        float * out = p.dst + row * D + gq;
        // SPLIT: 0 not written, 1 dst, 2 this run's partial (max, sum, unnormalized output) in
        // scratch slot `slot` (paged_fixed_combine).
        const int target = SPLIT ? fa::fixed_target(p.split_fixed, slot, c_first[c], c_pos[c]) : 1;
        if (target == 0) {
        } else if (target == 2) {
            if (valid) {
                float * part = p.scratch + (row * p.max_split + slot) * (D + 2);
                if (h == 0 && gq == 0) {
                    part[0] = m[c];
                    part[1] = l[c];
                }
                float * po = part + 2 + gq;
#pragma unroll
                for (int ob = 0; ob < OB; ob++) {
                    const int i = ob / HB * (CH / 16) + h * HB + ob % HB;
                    po[i * 16]     = o[ob][c];
                    po[i * 16 + 8] = o[ob][2 + c];
                }
            }
        } else if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. The column's lanes (all four
            // warps of the group) read its running (max, sum) before one of them writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[c], l[c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            grid_quad_sync(bar);
            if (valid) {
                if (h == 0 && gq == 0) {
                    store_ml(p, row, f);
                }
                const float inv = l[c] > 0.0f ? 1.0f / l[c] : 0.0f;
#pragma unroll
                for (int ob = 0; ob < OB; ob++) {
                    const int i = ob / HB * (CH / 16) + h * HB + ob % HB;
                    const float x0 = o[ob][c] * inv, x1 = o[ob][2 + c] * inv;
                    out[i * 16]     = p.piece > 0 ? f.x * out[i * 16] + f.y * x0 : x0;
                    out[i * 16 + 8] = p.piece > 0 ? f.x * out[i * 16 + 8] + f.y * x1 : x1;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / l[c];
#pragma unroll
            for (int ob = 0; ob < OB; ob++) {
                const int i = ob / HB * (CH / 16) + h * HB + ob % HB;
                out[i * 16]     = o[ob][c] * inv;
                out[i * 16 + 8] = o[ob][2 + c] * inv;
            }
        }
    }
#else
    __trap();
#endif
}

// One kernel choice's launch; -1: no shared memory for it on this device.
template <typename SH, typename K>
int launch_with(K kernel, std::atomic<int8_t> * opted_in, const legion_paged_attn & p, int G, cudaStream_t st) {
    if (!legion_device::smem_opt_in(opted_in, kernel, (int) SH::SMEM)) {
        return -1;
    }
    const int x = (p.n_tok * G + SH::COLS - 1) / SH::COLS;
    kernel<<<dim3(x, p.n_head_kv, p.n_seqs), SH::NW * 32, SH::SMEM, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// Where legion_paged_attn::verify_grid says so (heads of 256, 6 query heads a kv head, a
// non-compact cache, no key-block split; paged::verify_grid on the Rust side): 1 the pair kernel,
// 2 the quad kernel (a group a CTA, 256-dim chunks through 3 stages; 128-dim chunks through 6
// stages and two groups a CTA were slower on a GB10), 3 the quad kernel without its page
// prefetch. -1: no shared memory for it on this device (paged_prefill_mma instead).
template <int D, int G>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    using Q = QShape<D, 1, 256, 3>;
    if (p.verify_grid == 2) {
        static std::atomic<int8_t> o2[GGML_CUDA_MAX_DEVICES];
        return launch_with<Q>(paged_prefill_quad<D, G, 1, 256, 3>, o2, p, G, st);
    }
    if (p.verify_grid == 3) {
        static std::atomic<int8_t> o3[GGML_CUDA_MAX_DEVICES];
        return launch_with<Q>(paged_prefill_quad<D, G, 1, 256, 3, false>, o3, p, G, st);
    }
    static std::atomic<int8_t> o1[GGML_CUDA_MAX_DEVICES];
    return launch_with<Shape<D, GROUPS>>(paged_prefill_grid<D, G, GROUPS>, o1, p, G, st);
}

// The fixed split over the key blocks (p.split_fixed > 0, p.max_split > 1, #1133) on the quad
// kernel, where p.verify_grid picks a wide-grid kernel: a CTA per (column group, kv head,
// sequence, run), then paged_fixed_combine. The same bits as paged_prefill_mma<.., SPLIT> under
// the same rule. -1: no shared memory for it on this device (paged_prefill_mma's split instead).
template <int D, int G>
int launch_split(const legion_paged_attn & p, cudaStream_t st) {
    using Q = QShape<D, 1, 256, 3>;
    static std::atomic<int8_t> os[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(os, paged_prefill_quad<D, G, 1, 256, 3, true, true>, (int) Q::SMEM)) {
        return -1;
    }
    const int x = (p.n_tok * G + Q::COLS - 1) / Q::COLS;
    paged_prefill_quad<D, G, 1, 256, 3, true, true><<<dim3(x, p.n_head_kv, p.n_seqs * p.max_split), Q::NW * 32, Q::SMEM, st>>>(p);
    fa::launch_combine<D, G>(p, st);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace fa_grid
