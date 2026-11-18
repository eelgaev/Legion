// qsa_attn.cu, attn_gp_kernel (#1176): attn_gw_kernel's arithmetic for two neighbouring tokens
// of a sequence per CTA (8 warps, warps 0-3 the first token, 4-7 the second; one CTA per SM),
// so the K and V rows both select are loaded once.
//
// Not a standalone header: qsa_attn.cu includes it inside its anonymous namespace, after
// sm80/qsa_attn_gb10.cuh (ldmatrix_x4, ld_issue, ...).
//
// A prompt's neighbours pick mostly the same pools, but not at the same chunk index (the shared
// rows drift by a chunk or more), so the CTA runs a position window: each step computes the
// pending chunk of the token whose chunk ends first, and the other token's too when it starts by
// then. The rows of a step's chunks (64 at most) sit in 64 slots; a row the next step needs
// again stays where it is, the new ones go to the slots freed. Each token still runs its own
// chunks in order, every value computed as attn_gw_kernel computes it: only the shared-memory
// row an ldmatrix lane points at changes (rows past a selection point at a row of zeros, as
// attn_gw_kernel's zero-filled rows).
// Warps 0 and 4 plan the next step (which tokens advance, the slot of each row, the rows to
// load) before their scores; every thread copies K after the scores and V after every p.v.

template <int HD>
struct GpSmem {
    static constexpr int R = HD + 8;
    static constexpr int PR = MCH + 8;
    static constexpr int SLOTS = 2 * MCH;  // rows held: two chunks
    __half k[SLOTS + 1][R];                // row SLOTS: zeros
    __half v[SLOTS + 1][R];
    __half ph[2][ROWS][PR];  // [token]
    __half pl[2][ROWS][PR];
    float s[2][ROWS][MCH + 4];
    float qmax[2 * THREADS / 32];
    float alpha[2][ROWS];
    float lsum[2][ROWS];
    float mrow[2][ROWS];
    // By step parity: the position each slot holds (-1 none), the slot of each row of a token's
    // chunk, the rows to load (cell, slot) and their count, which tokens advance.
    alignas(16) int rpos[2][SLOTS];
    int slot[2][2][MCH];
    int lcell[2][SLOTS];
    int lslot[2][SLOTS];
    int nload[2];
    int adv[2][2];
    // The planners' exchange: the pending chunks' positions and rows, the slots kept, the
    // first token's new rows.
    alignas(16) int pp[2][MCH];
    int pn[2];
    unsigned keep[2];
    int cnt0;
};

// A barrier of the `n` threads (whole warps) that name barrier `id`.
__device__ __forceinline__ void bar_named(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

template <int HD>
__global__ void __launch_bounds__(2 * THREADS, 1) attn_gp_kernel(legion_qsa_attn p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    GpSmem<HD> & S = *reinterpret_cast<GpSmem<HD> *>(smem_raw);
    constexpr int NW = THREADS / 32;     // warps per token
    constexpr int PIECES = HD * 2 / 16;  // 16-byte pieces per K or V row
    constexpr int SLOTS = GpSmem<HD>::SLOTS;
    constexpr int KR = MCH / NW;  // K rows (score cells) per warp
    constexpr int DW = HD / NW;   // p.v dims per warp
    constexpr int NT = DW / 8;    // p.v n-tiles per warp
    constexpr int KS = HD / 16;   // score K steps
    static_assert(MCH == 32 && KR == 8 && SLOTS == 64, "one lane per chunk row, 64 slots");
    static_assert(NT % 2 == 0 && KS % 2 == 0, "ldmatrix x4 pairs");
    const int pairs = (p.n_tok + 1) / 2;
    const int s = blockIdx.x / pairs, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int u = warp / NW, wl = warp % NW, tu = t % THREADS;  // token, warp and thread in it
    const int gid = lane >> 2, tig = lane & 3;
    const int ti = blockIdx.x % pairs * 2 + u;
    const bool real = ti < p.n_tok;  // an odd last token: the second has none (n = 0)
    const int tt = real ? ti : ti - 1;
    const int tok = s * p.n_tok + tt;
    const int G = p.n_head / p.n_head_kv;
    const char * qb = (const char *) p.q + (size_t) s * p.q_nb3 + (size_t) tt * p.q_nb2 +
                      (size_t) kvh * G * p.q_nb1;
    float * out = p.dst + ((size_t) tok * p.n_head + (size_t) kvh * G) * HD;
    const int32_t * sel = p.sel + (size_t) tok * p.sel_row;
    const int n = real ? sel[0] : 0;
    const int32_t * ids = sel + 1;
    const int32_t * table = p.table + (size_t) s * p.table_width;

    // The planners (warp 0 of each token), lane l: row l of the token's pending chunk (the next
    // it computes): its position and cell; the next chunk's positions and cells, and the
    // positions of the one after (each loaded a step before it is read).
    const bool planner = wl == 0;
    auto pos_at = [&](int c) { return c * MCH + lane < n ? ld_issue(ids + c * MCH + lane) : -1; };
    auto cell_of = [&](int pos) {
        return pos < 0 ? -1 : ld_issue(table + pos / BLK) * BLK + pos % BLK;
    };
    int pend = 0, pp = -1, pcl = -1, pn1 = -1, pcl1 = -1, pn2 = -1;
    // Plans the step of parity nxt from the step of parity cur (`moved`: this token computed
    // its pending chunk there).
    auto plan = [&](int cur, int nxt, bool moved) {
        if (moved) {
            ++pend;
            pp = pn1;
            pcl = pcl1;
            pn1 = pn2;
            pcl1 = cell_of(pn1);
            pn2 = pos_at(pend + 2);
        }
        const int rows = max(0, min(MCH, n - pend * MCH));
        const int x = lane < rows ? pp : -1;
        S.pp[u][lane] = x;
        if (lane == 0) {
            S.pn[u] = rows;
        }
        if (u == 0 && lane < 2) {
            S.keep[lane] = 0u;
        }
        S.rpos[nxt][u * MCH + lane] = -1;
        bar_named(1, 64);
        // The window: the pending chunk that ends first, the other too if it starts by then.
        const int ra = S.pn[0], rb = S.pn[1];
        const int ea = ra ? S.pp[0][ra - 1] : INT_MAX, eb = rb ? S.pp[1][rb - 1] : INT_MAX;
        const bool aa = ra > 0 && (ea <= eb || S.pp[0][0] <= eb);
        const bool ab = rb > 0 && (eb <= ea || S.pp[1][0] <= ea);
        const bool mine = u == 0 ? aa : ab;
        int found = -1, dup = -1;
        if (mine && x >= 0) {
            const int4 * r4 = reinterpret_cast<const int4 *>(S.rpos[cur]);
#pragma unroll
            for (int j = 0; j < SLOTS / 4; ++j) {
                const int4 r = r4[j];
                found = r.x == x ? 4 * j : found;
                found = r.y == x ? 4 * j + 1 : found;
                found = r.z == x ? 4 * j + 2 : found;
                found = r.w == x ? 4 * j + 3 : found;
            }
            if (u == 1 && aa && found < 0) {
                const int4 * a4 = reinterpret_cast<const int4 *>(S.pp[0]);
#pragma unroll
                for (int j = 0; j < MCH / 4; ++j) {
                    const int4 r = a4[j];
                    dup = r.x == x ? 4 * j : dup;
                    dup = r.y == x ? 4 * j + 1 : dup;
                    dup = r.z == x ? 4 * j + 2 : dup;
                    dup = r.w == x ? 4 * j + 3 : dup;
                }
            }
        }
        if (found >= 0) {
            atomicOr(&S.keep[found >> 5], 1u << (found & 31));
        }
        const bool need = mine && x >= 0 && found < 0 && dup < 0;
        const unsigned nb = __ballot_sync(0xffffffffu, need);
        if (u == 0 && lane == 0) {
            S.cnt0 = __popc(nb);
        }
        bar_named(1, 64);
        // The new rows take the free slots in order: the first token's, then the second's.
        const unsigned f0 = ~S.keep[0], f1 = ~S.keep[1];
        const int rank = (u ? S.cnt0 : 0) + __popc(nb & ((1u << lane) - 1u));
        int sl = found >= 0 ? found : SLOTS;  // rows past the selection: the zero row
        if (need) {
            const int c0 = __popc(f0);
            sl = rank < c0 ? (int) __fns(f0, 0, rank + 1) : 32 + (int) __fns(f1, 0, rank - c0 + 1);
            S.lslot[nxt][rank] = sl;
            S.lcell[nxt][rank] = pcl;
        }
        if (dup < 0) {
            S.slot[nxt][u][lane] = sl;
            if (mine && x >= 0) {
                S.rpos[nxt][sl] = x;
            }
        }
        if (u == 1 && lane == 0) {
            S.nload[nxt] = S.cnt0 + __popc(nb);
            S.adv[nxt][0] = aa;
            S.adv[nxt][1] = ab;
        }
        bar_named(1, 64);
        if (dup >= 0) {
            S.slot[nxt][1][lane] = S.slot[nxt][0][dup];
        }
    };
    // The new rows of the plan of parity par (K or V), every thread a piece at a time.
    auto issue = [&](const half * src, __half (*dst)[GpSmem<HD>::R], int par) {
        const int nl = S.nload[par];
        for (int q = t; q < nl * PIECES; q += 2 * THREADS) {
            const int j = q / PIECES, pc = q % PIECES;
            const long long r = (long long) S.lcell[par][j] * p.kv_row + (long long) kvh * HD;
            cp_async16(&dst[S.lslot[par][j]][pc * 8], reinterpret_cast<const uint4 *>(src + r) + pc,
                       true);
        }
        cp_async_commit();
    };

    if (planner) {
        pp = pos_at(0);
        pn1 = pos_at(1);
        pn2 = pos_at(2);
        pcl = cell_of(pp);
        pcl1 = cell_of(pn1);
        S.rpos[1][u * MCH + lane] = -1;  // nothing held before the first step
        plan(1, 0, false);
    }
    for (int i = t; i < GpSmem<HD>::R; i += 2 * THREADS) {
        S.k[SLOTS][i] = __float2half(0.0f);
        S.v[SLOTS][i] = __float2half(0.0f);
    }

    // q: as attn_gb_kernel, each token's from its own warps.
    float qm = 0.0f;
    for (int i = tu; i < G * HD; i += THREADS) {
        qm = fmaxf(qm, fabsf(*(const float *) (qb + (size_t) (i / HD) * p.q_nb1 + (i % HD) * 4)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    }
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    if (tu < ROWS) {
        S.mrow[u][tu] = -INFINITY;
        S.lsum[u][tu] = 0.0f;
    }
    __syncthreads();
    issue(p.k, S.k, 0);
    issue(p.v, S.v, 0);
    qm = fmaxf(fmaxf(S.qmax[u * NW], S.qmax[u * NW + 1]),
               fmaxf(S.qmax[u * NW + 2], S.qmax[u * NW + 3]));
    int qe = 0;
    if (qm > 0.0f) {
        frexpf(qm, &qe);  // qm < 2^qe
    }
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(p.scale_log2, qe - 14);
    uint32_t qh[KS][4], ql[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = gid + 8 * (i & 1), d = ks * 16 + 2 * tig + 8 * (i >> 1);
            const float * qr = (const float *) (qb + (size_t) h * p.q_nb1) + d;
            const float x0 = h < G ? qr[0] * qup : 0.0f;
            const float x1 = h < G ? qr[1] * qup : 0.0f;
            const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
            const __half2 hi = __halves2half2(h0, h1);
            const __half2 lo = __halves2half2(__float2half_rn(x0 - __half2float(h0)),
                                              __float2half_rn(x1 - __half2float(h1)));
            qh[ks][i] = *reinterpret_cast<const uint32_t *>(&hi);
            ql[ks][i] = *reinterpret_cast<const uint32_t *>(&lo);
        }
    }

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }
    // The first step's K rows, every thread's (its V may stay in flight).
    cp_async_wait<1>();
    __syncthreads();

    for (int k = 0, ci = 0;; ++k) {
        const int cur = k & 1, nxt = cur ^ 1;
        if (!S.adv[cur][0] && !S.adv[cur][1]) {
            break;
        }
        const bool act = S.adv[cur][u];
        if (planner) {
            plan(cur, nxt, act);
        }
        const int nh = min(MCH, n - ci * MCH);
        // scores: warp wl = cells 8 wl..8 wl + 7 of this token's chunk, every dim
        if (act) {
            const int cb = wl * KR;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            if (cb < nh) {
                const __half * kr = &S.k[S.slot[cur][u][cb + (lane & 7)]][8 * (lane >> 3)];
#pragma unroll
                for (int ks = 0; ks < KS; ks += 2) {
                    uint32_t b[4];
                    ldmatrix_x4(b, kr + ks * 16);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float tg[4] = {0.f, 0.f, 0.f, 0.f};
                        mma16816(tg, qh[ks + h], b + 2 * h);
                        mma16816(tg, ql[ks + h], b + 2 * h);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            sc[i] += tg[i];
                        }
                    }
                }
            }
            const int c = cb + 2 * tig;
            S.s[u][gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[u][gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[u][gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[u][gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        // Every score read (and the plan made): the next step's new K rows come in during the
        // softmax and p.v.
        __syncthreads();
        issue(p.k, S.k, nxt);
        // online softmax: as attn_gb_kernel
        if (act) {
            constexpr int PER = MCH / 8;
            constexpr float VUP = 16384.0f;
            const int r = tu >> 3, sub = tu & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[u][r][sub * PER + j];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[u][r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += e;
                const float pu = e * VUP;
                const __half hi = __float2half_rn(pu);
                S.ph[u][r][sub * PER + j] = hi;
                S.pl[u][r][sub * PER + j] = __float2half_rn(pu - __half2float(hi));
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[u][r] = a;
                S.lsum[u][r] = fmaf(S.lsum[u][r], a, sum);
                S.mrow[u][r] = m_new;
            }
        }
        // This step's V rows (the next step's K may stay in flight); p and alpha.
        cp_async_wait<1>();
        __syncthreads();
        // p.v: as attn_gb_kernel
        if (act) {
            constexpr float VDOWN = 1.0f / 16384.0f;
            float tmp[NT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
            }
            const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = 8 * (lane >> 4);
#pragma unroll
            for (int k0 = 0; k0 < MCH; k0 += 16) {
                uint32_t ah[4], al[4];
                ldmatrix_x4(ah, &S.ph[u][lr][k0 + lc]);
                ldmatrix_x4(al, &S.pl[u][lr][k0 + lc]);
                const __half * vr = &S.v[S.slot[cur][u][k0 + lr]][wl * DW + lc];
#pragma unroll
                for (int j = 0; j < NT; j += 2) {
                    uint32_t b[4];
                    ldmatrix_x4_trans(b, vr + j * 8);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                    mma16816(tmp[j + 1], ah, b + 2);
                    mma16816(tmp[j + 1], al, b + 2);
                }
            }
            const float a0 = S.alpha[u][gid], a1 = S.alpha[u][gid + 8];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * VDOWN);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * VDOWN);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * VDOWN);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * VDOWN);
            }
        }
        // The next step's K rows, every thread's; every p.v done (its V slots are free).
        cp_async_wait<0>();
        __syncthreads();
        issue(p.v, S.v, nxt);
        ci += act;
    }
    if (!real) {
        return;
    }
    const float l0 = S.lsum[u][gid], l1 = S.lsum[u][gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int d = wl * DW + j * 8 + 2 * tig;
        if (gid < G) {
            *reinterpret_cast<float2 *>(out + (size_t) gid * HD + d) =
                make_float2(acc[j][0] * i0, acc[j][1] * i0);
        }
        if (gid + 8 < G) {
            *reinterpret_cast<float2 *>(out + (size_t) (gid + 8) * HD + d) =
                make_float2(acc[j][2] * i1, acc[j][3] * i1);
        }
    }
}
