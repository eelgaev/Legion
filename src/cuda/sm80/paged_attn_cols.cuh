// paged_attn/common.cuh, sm_80 and newer: paged_prefill_mma's prefill for heads of 512 at 8 query
// heads a kv head with 16 query columns a warp (namespace fa_cols, paged_prefill_cols, #1181). The
// same bits.
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after
// sm80/paged_attn.cuh (namespace fa: swz, cp_async16, ldsm, ldsm_t, mma, transpose, pack) and
// sm80/paged_attn_pair.cuh (fa_pair::pair_sync).
//
// paged_prefill_mma at heads of 512 gives a warp 8 query columns: every K or V fragment it loads
// with ldmatrix (512 bytes) feeds one MMA, and the 8 warps of a CTA each read the whole 32 KB chunk.
// Measurement only (#1181 r16): it is exact but 0.78-0.86x of paged_prefill_mma (255 registers, the
// output accumulators spill), and halving paged_prefill_mma's ldmatrix bytes alone is no faster, so
// shared memory bandwidth is not what limits it. Here a warp takes 16 columns (two MMA
// n-tiles), so each fragment feeds two MMAs: half the ldmatrix bytes per MMA. The CTA keeps 64
// columns of one kv head (8 tokens) and the same K/V fill per column; its 8 warps are 4 pairs, pair
// g = warp % 4 owning columns 16 g .. 16 g + 15, and the pair splits the work by head dims, so a
// warp holds Q's B fragments for 256 dims (64 registers) and half the output (128):
// - Scores: warp h takes dims 256 h .. 256 h + 255, i.e. paged_prefill_mma's K chunk h. The K
//   chunks are interleaved so both warps work on every step: K step s (s = 0, 1) holds, for the
//   tile's 64 keys, dims 128 s .. + 127 in units 0-15 and 256 + 128 s .. + 127 in units 16-31;
//   warp h's chain for each 16 keys and 8 columns runs k-steps 8 s .. 8 s + 7 of its chunk on step
//   s, so it is the same 16-MMA chain from zero as there. After step 1 the pair swaps halves
//   (2 KB a warp) so that warp h holds the whole scores s0 + s1 of key blocks 2 h, 2 h + 1.
// - Softmax as in paged_prefill_pair: the column maxima of both halves, warp 0's sum of key blocks
//   0 and 1 continued by warp 1 with 2 and 3 (paged_prefill_mma's order), (l, alpha) as there,
//   and each warp's P fragments to the other.
// - Output: on V chunk c (as paged_prefill_mma's), warp h owns dims c * 256 + h * 128 .. + 127,
//   each 16 dims a chain of 4 MMAs from zero over the tile's 16-key steps, added as o alpha + acc.
// The exchange (S halves, maxima, sums, P: 6.6 KB a pair) uses the ring stage of the K step just
// read, behind one more CTA barrier a tile (XS = 0, three stages), or its own space after a
// two-stage ring (XS = 1).
//
// Not covered (paged_prefill_mma instead): other head sizes or GQA ratios, the compact cache,
// key-block splits.

namespace fa_cols {

template <int D, int XS> struct Shape {
    static_assert(D == 512, "the 16-column kernel is for heads of 512");
    static constexpr int NW      = 8;                              // warps per CTA
    static constexpr int GROUPS  = NW / 2;                         // warp pairs (column groups)
    static constexpr int COLS    = 16 * GROUPS;                    // query columns per CTA
    static constexpr int CH      = 256;                            // dims per chunk
    static constexpr int NS      = XS ? 2 : 3;                     // ring stages
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);  // bytes per chunk
    static constexpr int RING    = NS * CHUNK;
    static constexpr int XB      = 8192;  // a pair's exchange (6784 bytes in use)
    static constexpr int LB      = 64;    // a pair's final sums
    static constexpr size_t SMEM = (size_t) RING + (XS ? (size_t) GROUPS * XB : 0) + (size_t) GROUPS * LB;
    // Exchange offsets: S halves [2 warps][4][32 lanes] float4, maxima [2][16] float, warp 0's
    // partial sums [32] float4, P [2][2][32] uint4.
    static constexpr int X_S = 0, X_MX = 4096, X_SUM = 4224, X_P = 4736;
};

template <int D, int G, int XS>
__global__ void __launch_bounds__(Shape<D, XS>::NW * 32, 1) paged_prefill_cols(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    static_assert(G == 8, "the 16-column kernel is for 8 query heads a kv head");
    using SH = Shape<D, XS>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    using fa::transpose;
    using fa_pair::pair_sync;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK;
    constexpr int NW = SH::NW, COLS = SH::COLS;
    constexpr int KS = CH / 16;   // k-steps of a warp's K Q^T (its 256 dims)
    constexpr int QT = COLS / G;  // tokens per CTA
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last query tiles see the most keys: they go first.
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cg = warp % 4, h = warp / 4;   // pair (column group), half
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Column c's token. Columns past n_tok repeat the tile's last token and are not written.
    auto tok_of = [&](int c) { return min(t0 + min(c / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(COLS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    // This pair's 16 columns (for skipping and unmasked tiles) and this thread's columns of the
    // S^T / O^T fragments: n-tile nt, columns 16 cg + 8 nt + 2 tq + {0, 1}.
    const int c_base = cg * 16;
    const int w_pos_lo = pos_of(c_base), w_pos_hi = pos_of(c_base + 15);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2][2], c_first[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            c_pos[nt][j]   = pos_of(c_base + 8 * nt + 2 * tq + j);
            c_first[nt][j] = first_key(c_pos[nt][j], p.window);
        }
    }

    // Q as B fragments for this warp's dims 256 h ..: column 16 cg + 8 nt + gq, dims 2 tq + {0, 1}
    // and + 8 of k-step ks.
    uint32_t qf[2][KS][2];
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        const int c = c_base + 8 * nt + gq;
        const int t = tok_of(c);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1) +
                           h * CH;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            qf[nt][ks][0] = pack(a.x, a.y);
            qf[nt][ks][1] = pack(b.x, b.y);
        }
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / 4, K steps 0 and 1 then V chunks 0 and 1.
    // Unit u (8 dims) of a row: K step s, dims (u / 16) 256 + 128 s + 8 (u % 16); V chunk c,
    // dims 256 c + 8 u.
    const int n_chunks = (wk.i1 - wk.i0) * 4;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / 4, part = j % 4;
            const int b = wk.block(i);
            const int ty = part < 2 ? p.k_type : p.v_type;
            const int64_t cell_bytes = part < 2 ? p.k_cell : p.v_cell;
            const char * base = part < 2 ? wk.cells(p.k, p.k_stage, i, cell_bytes) : wk.cells(p.v, p.v_stage, i, cell_bytes);
            const int e0 = head_at<D, false>(p, kvh);
            const uint32_t dst = sbase + (uint32_t) (j % NS) * CHUNK;
            auto dim_of = [&](int u) { return part < 2 ? (u / 16) * 256 + part * 128 + (u % 16) * 8 : (part - 2) * 256 + u * 8; };
            if (ty == LEGION_KV_F16) {
                const half * src = (const half *) base + e0;
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    cp_async16(dst + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + dim_of(u), b * BLK + r < k_end);
                }
            } else {
                // Quantized: dequantized to F16 on the way in (synchronous loads).
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    uint4 x = make_uint4(0, 0, 0, 0);
                    if (b * BLK + r < k_end) {
                        float f[8];
                        load_row<8>(base + r * cell_bytes, ty, e0 + dim_of(u), f);
                        x = make_uint4(pack(f[0], f[1]), pack(f[2], f[3]), pack(f[4], f[5]), pack(f[6], f[7]));
                    }
                    *(uint4 *) (smem + (dst - sbase) + fa::swz<UNITS>(r, u)) = x;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // This warp's half of O^T: o[c][ib][nt] is dims 16 (16 c + 8 h + ib) + gq (+ 8 for [2], [3]),
    // columns 16 cg + 8 nt + 2 tq + {0, 1}.
    float o[2][8][2][4];
#pragma unroll
    for (int c = 0; c < 2; c++) {
#pragma unroll
        for (int ib = 0; ib < 8; ib++) {
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                o[c][ib][nt][0] = o[c][ib][nt][1] = o[c][ib][nt][2] = o[c][ib][nt][3] = 0.0f;
            }
        }
    }
    // Running max (log2 units) per column; this thread's share of the sum (kept by warp 1).
    float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
    float l[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const float scale = p.scale * LOG2E;
    const int bar = 1 + cg;
    float * lb = (float *) (smem + SH::RING + (XS ? SH::GROUPS * SH::XB : 0) + cg * SH::LB);

    // ldmatrix addresses, as in paged_prefill_mma; warp h's K units start at 16 h.
    constexpr int ROWB = UNITS * 16;
    const uint32_t k_lane = (lane / 8 % 2 * 8 + lane % 8) * ROWB + h * 256, v_lane = (lane / 16 * 8 + lane % 8) * ROWB + h * 256;
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
        // Whether any column of the pair sees a key of the tile, and whether all see all.
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[4][2][4];      // S^T: keys 16 i + gq (+ 8), columns of n-tile nt
        uint32_t pf[4][2][2];  // P as B fragments per 16-key step and n-tile
        float alpha[2][2];
#pragma unroll
        for (int part = 0; part < 4; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            const uint32_t st = sbase + (uint32_t) (j % NS) * CHUNK;
            if (part < 2) {
                if (active) {
                    // S^T over this warp's K chunk, k-steps 8 part .. 8 part + 7: one chain
                    // from zero over the two steps per 16 keys and n-tile.
                    if (part == 0) {
#pragma unroll
                        for (int i = 0; i < 4; i++) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
                                s[i][nt][0] = s[i][nt][1] = s[i][nt][2] = s[i][nt][3] = 0.0f;
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 4; i++) {
#pragma unroll
                        for (int kk = 0; kk < KS / 2; kk++) {
                            uint32_t a[4];
                            ldsm(a, st + k_lane + k_x[kk % 4] + i * 16 * ROWB + kk / 4 * 128);
                            mma(s[i][0], a, qf[0][part * (KS / 2) + kk]);
                            mma(s[i][1], a, qf[1][part * (KS / 2) + kk]);
                        }
                    }
                }
                if (part == 1) {
                    unsigned char * xb;
                    if constexpr (XS != 0) {
                        xb = smem + SH::RING + cg * SH::XB;
                    } else {
                        // The stage just read: every warp is past it.
                        __syncthreads();
                        xb = smem + (j % NS) * CHUNK + cg * SH::XB;
                    }
                    if (active) {
                        float4 * xs = (float4 *) (xb + SH::X_S);
                        float * xm = (float *) (xb + SH::X_MX);
                        float4 * xsum = (float4 *) (xb + SH::X_SUM);
                        uint4 * xp = (uint4 *) (xb + SH::X_P);
                        // The other warp's key blocks (warp 0 keeps 0 and 1, warp 1 keeps 2 and 3).
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
                                float v[4];
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    v[e] = h ? s[ii][nt][e] : s[2 + ii][nt][e];
                                }
                                xs[(h * 4 + ii * 2 + nt) * 32 + lane] = make_float4(v[0], v[1], v[2], v[3]);
                            }
                        }
                        pair_sync(bar);
                        // Whole scores s0 + s1 of key blocks 2 h + ii (s0 from dims 0-255).
                        float sw[2][2][4];
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
                                const float4 x = xs[((1 - h) * 4 + ii * 2 + nt) * 32 + lane];
                                const float xo[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    const float mine = h ? s[2 + ii][nt][e] : s[ii][nt][e];
                                    sw[ii][nt][e] = h ? xo[e] + mine : mine + xo[e];
                                }
                            }
                        }
                        // Online softmax per column; element e is key 16 (2 h + ii) + gq + 8 (e / 2),
                        // column 2 tq + e % 2 of n-tile nt.
                        float mx[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    float x = sw[ii][nt][e] * scale;
                                    if (!whole) {
                                        const int key = k0 + (2 * h + ii) * 16 + gq + 8 * (e / 2);
                                        x = key <= c_pos[nt][e % 2] && key >= c_first[nt][e % 2] ? x : -INFINITY;
                                    }
                                    sw[ii][nt][e] = x;
                                    mx[nt][e % 2] = fmaxf(mx[nt][e % 2], x);
                                }
                            }
                        }
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                            for (int c = 0; c < 2; c++) {
#pragma unroll
                                for (int off = 4; off < 32; off <<= 1) {
                                    mx[nt][c] = fmaxf(mx[nt][c], __shfl_xor_sync(0xffffffff, mx[nt][c], off));
                                }
                            }
                        }
                        if (gq == 0) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
                                xm[h * 16 + nt * 8 + 2 * tq]     = mx[nt][0];
                                xm[h * 16 + nt * 8 + 2 * tq + 1] = mx[nt][1];
                            }
                        }
                        pair_sync(bar);
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                            for (int c = 0; c < 2; c++) {
                                mx[nt][c] = fmaxf(mx[nt][c], xm[(1 - h) * 16 + nt * 8 + 2 * tq + c]);
                                const float m_new = fmaxf(m[nt][c], mx[nt][c]);
                                // No visible key yet: exp against 0 gives 0 for the masked scores.
                                const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                                alpha[nt][c] = exp2f(m[nt][c] - m_use);
                                m[nt][c]  = m_new;
                                mx[nt][c] = m_use;
                            }
                        }
                        uint32_t pm[2][2][2];  // this warp's P fragments, key blocks 2 h + ii
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    sw[ii][nt][e] = exp2f(sw[ii][nt][e] - mx[nt][e % 2]);
                                }
                                pm[ii][nt][0] = transpose(pack(sw[ii][nt][0], sw[ii][nt][1]));
                                pm[ii][nt][1] = transpose(pack(sw[ii][nt][2], sw[ii][nt][3]));
                            }
                        }
                        // Warp 0: its sums of key blocks 0 and 1 to warp 1, which goes on with 2
                        // and 3; both: their P to the other.
                        float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
                        if (h == 0) {
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                                for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                                    for (int e = 0; e < 4; e++) {
                                        sum[nt][e % 2] += sw[ii][nt][e];
                                    }
                                }
                            }
                            xsum[lane] = make_float4(sum[0][0], sum[0][1], sum[1][0], sum[1][1]);
                        }
                        xp[(h * 2 + 0) * 32 + lane] = make_uint4(pm[0][0][0], pm[0][0][1], pm[0][1][0], pm[0][1][1]);
                        xp[(h * 2 + 1) * 32 + lane] = make_uint4(pm[1][0][0], pm[1][0][1], pm[1][1][0], pm[1][1][1]);
                        pair_sync(bar);
                        if (h == 1) {
                            const float4 q = xsum[lane];
                            sum[0][0] = q.x;
                            sum[0][1] = q.y;
                            sum[1][0] = q.z;
                            sum[1][1] = q.w;
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                                for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                                    for (int e = 0; e < 4; e++) {
                                        sum[nt][e % 2] += sw[ii][nt][e];
                                    }
                                }
                                l[nt][0] = l[nt][0] * alpha[nt][0] + sum[nt][0];
                                l[nt][1] = l[nt][1] * alpha[nt][1] + sum[nt][1];
                            }
                        }
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
                            const uint4 v = xp[((1 - h) * 2 + ii) * 32 + lane];
                            const uint32_t po[2][2] = {{v.x, v.y}, {v.z, v.w}};
#pragma unroll
                            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                                for (int k = 0; k < 2; k++) {
                                    pf[ii][nt][k]     = h ? po[nt][k] : pm[ii][nt][k];
                                    pf[2 + ii][nt][k] = h ? pm[ii][nt][k] : po[nt][k];
                                }
                            }
                        }
                    }
                }
            } else if (active) {
                // O^T = O^T alpha + V^T P for this warp's 128 of this chunk's output dims.
                const int c = part - 2;
#pragma unroll
                for (int ib = 0; ib < 8; ib++) {
                    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_lane + v_x[ib % 4] + ks * 16 * ROWB + ib / 4 * 128);
                        mma(acc[0], a, pf[ks][0]);
                        mma(acc[1], a, pf[ks][1]);
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        float * oi = o[c][ib][nt];
                        oi[0] = oi[0] * alpha[nt][0] + acc[nt][0];
                        oi[1] = oi[1] * alpha[nt][1] + acc[nt][1];
                        oi[2] = oi[2] * alpha[nt][0] + acc[nt][2];
                        oi[3] = oi[3] * alpha[nt][1] + acc[nt][3];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // Warp 1's sums (the lane tree as in paged_prefill_mma) to warp 0.
    if (h == 1) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int c = 0; c < 2; c++) {
#pragma unroll
                for (int off = 4; off < 32; off <<= 1) {
                    l[nt][c] += __shfl_xor_sync(0xffffffff, l[nt][c], off);
                }
            }
            if (gq == 0) {
                lb[nt * 8 + 2 * tq]     = l[nt][0];
                lb[nt * 8 + 2 * tq + 1] = l[nt][1];
            }
        }
    }
    pair_sync(bar);
    if (h == 0) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            l[nt][0] = lb[nt * 8 + 2 * tq];
            l[nt][1] = lb[nt * 8 + 2 * tq + 1];
        }
    }

#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int col = c_base + 8 * nt + 2 * tq + c;
            const int t = t0 + col / G;
            const bool valid = t < p.n_tok;
            const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + col % G;
            float * out = p.dst + row * D + gq;
            if (p.ml) {
                // Pieces (#352): merged into the earlier pieces' output. The column's 16 lanes over
                // both warps read its running (max, sum) before one of them writes it.
                const float4 f = valid ? merge_row<true>(p, row, m[nt][c], l[nt][c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                pair_sync(bar);
                if (valid) {
                    if (h == 0 && gq == 0) {
                        store_ml(p, row, f);
                    }
                    const float inv = l[nt][c] > 0.0f ? 1.0f / l[nt][c] : 0.0f;
#pragma unroll
                    for (int cc = 0; cc < 2; cc++) {
#pragma unroll
                        for (int ib = 0; ib < 8; ib++) {
                            const int i = cc * 16 + h * 8 + ib;
                            const float x0 = o[cc][ib][nt][c] * inv, x1 = o[cc][ib][nt][2 + c] * inv;
                            out[i * 16]     = p.piece > 0 ? f.x * out[i * 16] + f.y * x0 : x0;
                            out[i * 16 + 8] = p.piece > 0 ? f.x * out[i * 16 + 8] + f.y * x1 : x1;
                        }
                    }
                }
            } else if (valid) {
                const float inv = 1.0f / l[nt][c];
#pragma unroll
                for (int cc = 0; cc < 2; cc++) {
#pragma unroll
                    for (int ib = 0; ib < 8; ib++) {
                        const int i = cc * 16 + h * 8 + ib;
                        out[i * 16]     = o[cc][ib][nt][c] * inv;
                        out[i * 16 + 8] = o[cc][ib][nt][2 + c] * inv;
                    }
                }
            }
        }
    }
#else
    __trap();
#endif
}

// Heads of 512 at 8 query heads a kv head, a non-compact cache, no key-block split. -1: no shared
// memory for it on this device (paged_prefill_mma instead).
template <int D, int G, int XS>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_cols<D, G, XS>, (int) Shape<D, XS>::SMEM)) {
        return -1;
    }
    constexpr int QT = Shape<D, XS>::COLS / G;
    paged_prefill_cols<D, G, XS>
        <<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), Shape<D, XS>::NW * 32, Shape<D, XS>::SMEM, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace fa_cols
