// qsa_attn.cu, the GB10 (compute capability 12.x) prompt kernel: attn_mma_kernel's arithmetic
// with two CTAs per SM (attn_gb_kernel, #828). Ampere (8.0-8.7) takes it too (#1212): sm_86 has
// the GB10's 100 KB per SM, the A100 164 KB (still one attn_mma_kernel CTA per SM at HD 256).
//
// Not a standalone header: qsa_attn.cu includes it inside its anonymous namespace, after
// sm80/qsa_attn.cuh (MCH, mma16816, cp_async16, ...). The ldmatrix wrappers keep the
// __CUDA_ARCH__ guards and __trap() stubs, so the file compiles for every target architecture;
// use_gb<HD>() in qsa_attn.cu picks this kernel per device at run time.
//
// attn_mma_kernel needs ~87 KB of shared memory at HD 256 (q hi + lo, and two stages of K and
// V), so the GB10 (100 KB per SM) runs one CTA of 4 warps per SM: one warp per scheduler, each
// stalling on its own shared-memory loads, MMA chains and barriers (the mma.sync pipe ran at
// about a third of its rate on a 32K prompt). This kernel computes every value the same way,
// in the same order (the bits are attn_mma_kernel's), from ~39 KB, two CTAs per SM:
// - q's hi and lo fragments live in registers for the whole CTA (built once from global
//   memory), instead of shared memory reloaded for every chunk;
// - one K and one V buffer, filled out of phase: K of the next chunk is copied while the
//   softmax and p.v of this one run, V of the next while its scores run;
// - fragments come from ldmatrix (K, and V transposed), and the softmax writes p's hi and lo
//   halves once, where every warp used to convert the whole chunk's p for itself.

template <int HD>
struct GbSmem {
    static constexpr int R = HD + 8;    // halves per K / V row (conflict-free ldmatrix rows)
    static constexpr int PR = MCH + 8;  // halves per p row
    __half k[MCH][R];
    __half v[MCH][R];
    __half ph[ROWS][PR];
    __half pl[ROWS][PR];
    float s[ROWS][MCH + 4];
    float qmax[THREADS / 32];
    float alpha[ROWS];
    float lsum[ROWS];
    float mrow[ROWS];
};

__device__ __forceinline__ void ldmatrix_x4(uint32_t * r, const void * smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t * r, const void * smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    __trap();
#endif
}

template <int HD>
__global__ void __launch_bounds__(THREADS, 2) attn_gb_kernel(legion_qsa_attn p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    GbSmem<HD> & S = *reinterpret_cast<GbSmem<HD> *>(smem_raw);
    constexpr int PIECES = HD * 2 / 16;              // 16-byte pieces per K or V row
    constexpr int ROWS_PER_PASS = THREADS / PIECES;  // rows a pass of the CTA copies
    constexpr int RPT = MCH / ROWS_PER_PASS;         // rows each thread copies
    constexpr int DW = HD / 4;                       // p.v dims per warp
    constexpr int NT = DW / 8;                       // p.v n-tiles per warp
    constexpr int KS = HD / 16;                      // score K steps
    static_assert(THREADS % PIECES == 0 && MCH % ROWS_PER_PASS == 0, "copy layout");
    static_assert(NT % 2 == 0 && KS % 2 == 0, "ldmatrix x4 pairs");
    const int tok = blockIdx.x, kvh = blockIdx.y;
    const int G = p.n_head / p.n_head_kv;
    const int s = tok / p.n_tok, ti = tok % p.n_tok;
    const char * qb = (const char *) p.q + (size_t) s * p.q_nb3 + (size_t) ti * p.q_nb2 +
                      (size_t) kvh * G * p.q_nb1;
    float * out = p.dst + ((size_t) tok * p.n_head + (size_t) kvh * G) * HD;
    const int32_t * sel = p.sel + (size_t) tok * p.sel_row;
    const int n = sel[0];
    const int32_t * ids = sel + 1;
    const int32_t * table = p.table + (size_t) s * p.table_width;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int pc = t % PIECES;

    // The cells of chunk c0's rows this thread copies (-1 past the selection), found once for
    // its K and its V copy.
    int cells[RPT];
    auto locate = [&](int c0) {
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const int c = t / PIECES + i * ROWS_PER_PASS;
            int cell = -1;
            if (c0 + c < n) {
                const int pos = __ldg(ids + c0 + c);
                cell = __ldg(table + pos / BLK) * BLK + pos % BLK;
            }
            cells[i] = cell;
        }
    };
    // One buffer's rows (K or V) of the located chunk; cells past the selection are zeros.
    auto issue = [&](const half * src, __half (*dst)[GbSmem<HD>::R]) {
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const int c = t / PIECES + i * ROWS_PER_PASS;
            const bool valid = cells[i] >= 0;
            const long long r = valid ? (long long) cells[i] * p.kv_row + (long long) kvh * HD : 0;
            cp_async16(&dst[c][pc * 8], reinterpret_cast<const uint4 *>(src + r) + pc, valid);
        }
        cp_async_commit();
    };
    if (n > 0) {
        locate(0);
        issue(p.k, S.k);
        issue(p.v, S.v);
    }

    // q: as attn_mma_kernel (G heads + zero rows, times a power of two that puts its largest
    // value near 2^14, split into hi + lo), straight into this thread's A fragments.
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) {
        qm = fmaxf(qm, fabsf(*(const float *) (qb + (size_t) (i / HD) * p.q_nb1 + (i % HD) * 4)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    }
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    if (t < ROWS) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
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

    // K of the first chunk (V may still be on its way).
    cp_async_wait<1>();
    __syncthreads();
    for (int c0 = 0; c0 < n; c0 += MCH) {
        const int nh = min(MCH, n - c0);
        const bool more = c0 + MCH < n;
        // scores: warp w = cells 8w..8w+7, every dim (K resident)
        {
            const int cb = warp * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < KS; ks += 2) {
                uint32_t b[4];
                ldmatrix_x4(b, &S.k[cb + (lane & 7)][ks * 16 + 8 * (lane >> 3)]);
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
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        // K is free: the next chunk's K comes in during the softmax and p.v.
        if (more) {
            locate(c0 + MCH);
            issue(p.k, S.k);
        }
        // online softmax (base 2): row t / 8, MCH / 8 cells per thread; p (at most 1) times
        // 2^14 so its lo half stays normal, split into hi + lo halves for p.v
        {
            constexpr int PER = MCH / 8;
            constexpr float VUP = 16384.0f;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += e;
                const float pu = e * VUP;
                const __half hi = __float2half_rn(pu);
                S.ph[r][sub * PER + j] = hi;
                S.pl[r][sub * PER + j] = __float2half_rn(pu - __half2float(hi));
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        // V of this chunk (the next chunk's K may stay in flight).
        if (more) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        // p.v: warp w = dims [w HD / 4, (w + 1) HD / 4), the chunk's two K steps (hi then lo)
        // chained from a zero fragment, as attn_mma_kernel
        {
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
                ldmatrix_x4(ah, &S.ph[lr][k0 + lc]);
                ldmatrix_x4(al, &S.pl[lr][k0 + lc]);
#pragma unroll
                for (int j = 0; j < NT; j += 2) {
                    uint32_t b[4];
                    ldmatrix_x4_trans(b, &S.v[k0 + lr][warp * DW + j * 8 + lc]);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                    mma16816(tmp[j + 1], ah, b + 2);
                    mma16816(tmp[j + 1], al, b + 2);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * VDOWN);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * VDOWN);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * VDOWN);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * VDOWN);
            }
        }
        // The next chunk's K resident, every warp done with V, p and alpha: the next V comes
        // in during the next scores.
        cp_async_wait<0>();
        __syncthreads();
        if (more) {
            issue(p.v, S.v);
        }
    }
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int d = warp * DW + j * 8 + 2 * tig;
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

// attn_gw_kernel (#1183): attn_gb_kernel's arithmetic (so attn_mma_kernel's bits) with
// warp-local copies. Every K row a warp's scores read and every V dim its p.v reads, the warp
// copies itself, so it needs no CTA barrier to refill them:
// - warp w copies K rows 8w..8w+7 (a full row per instruction) and the next chunk's right
//   after its own scores; V dims [w HD / 4, (w + 1) HD / 4) of every row, the next chunk's
//   right after its own p.v. Two CTA barriers per chunk instead of three (the scores before
//   the softmax, p and alpha before p.v);
// - the next chunk's cells come from one lane per row, their table loads issued before the
//   scores (the positions one chunk earlier still), where attn_gb_kernel's sat on the path
//   between two barriers; lanes take the rows they copy by shuffles;
// - warps whose 8 cells are all past the selection skip their score products (the cells are
//   -inf either way).

// A 32-bit load from global memory issued where it stands (not moved to its first use).
__device__ __forceinline__ int ld_issue(const int32_t * a) {
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];\n" : "=r"(v) : "l"(a));
    return v;
}

// PROBE (#1176 r40, `legion qsa-bench` only): bit 0 drops every refill after the first chunk
// (the products run on stale K / V), bit 1 drops the mma.sync products (each becomes one add of
// a fragment register, so the ldmatrix loads stay); both give wrong output. Bit 2 looks the
// cells up one chunk earlier (a refill's cells were loaded a whole chunk before it, not at the
// top of the chunk that issues it); bit 3 refills in halves, each half as soon as this warp has
// read it (K: the first half of the dims mid-scores; V: the chunk's first 16 rows between the
// p.v K steps); bits 2 and 3 keep the output bits. 0 is the kernel the model runs.
template <int HD, int PROBE = 0>
__global__ void __launch_bounds__(THREADS, 2) attn_gw_kernel(legion_qsa_attn p) {
    constexpr bool FILL = (PROBE & 1) == 0, MMA = (PROBE & 2) == 0, EARLY = (PROBE & 4) != 0;
    constexpr bool PROG = (PROBE & 8) != 0;
    // Bit 4 (with bit 3; wrong output): after the first chunk, only the halves issued first
    // (half the bytes, the same round trips: is the feed bytes or latency bound?).
    constexpr bool HALFB = (PROBE & 16) != 0;
    static_assert(!HALFB || PROG, "bit 4 needs bit 3");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    GbSmem<HD> & S = *reinterpret_cast<GbSmem<HD> *>(smem_raw);
    constexpr int NW = THREADS / 32;          // warps
    constexpr int PIECES = HD * 2 / 16;       // 16-byte pieces per K or V row
    constexpr int KR = MCH / NW;              // K rows (score cells) per warp
    constexpr int DW = HD / NW;               // p.v dims per warp
    constexpr int VP = DW * 2 / 16;           // 16-byte pieces of a V row per warp
    constexpr int KPL = KR * PIECES / 32;     // K pieces per lane
    constexpr int VPL = MCH * VP / 32;        // V pieces per lane
    constexpr int NT = DW / 8;                // p.v n-tiles per warp
    constexpr int KS = HD / 16;               // score K steps
    static_assert(MCH == 32 && KR == 8, "one lane per chunk row, 8 score cells per warp");
    static_assert(KR * PIECES % 32 == 0 && MCH * VP % 32 == 0, "copy layout");
    static_assert(NT % 2 == 0 && KS % 2 == 0, "ldmatrix x4 pairs");
    const int tok = blockIdx.x, kvh = blockIdx.y;
    const int G = p.n_head / p.n_head_kv;
    const int s = tok / p.n_tok, ti = tok % p.n_tok;
    const char * qb = (const char *) p.q + (size_t) s * p.q_nb3 + (size_t) ti * p.q_nb2 +
                      (size_t) kvh * G * p.q_nb1;
    float * out = p.dst + ((size_t) tok * p.n_head + (size_t) kvh * G) * HD;
    const int32_t * sel = p.sel + (size_t) tok * p.sel_row;
    const int n = sel[0];
    const int32_t * ids = sel + 1;
    const int32_t * table = p.table + (size_t) s * p.table_width;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // Lane l: the position of row l of the chunk at c0 (-1 past the selection), its cell.
    auto pos_at = [&](int c0) { return c0 + lane < n ? ld_issue(ids + c0 + lane) : -1; };
    auto cell_of = [&](int pos) {
        return pos < 0 ? -1 : ld_issue(table + pos / BLK) * BLK + pos % BLK;
    };
    // This warp's K rows / V dims of the chunk whose cells `cell` holds (lane = row); cells past
    // the selection are zeros.
    auto issue_k = [&](int cell) {
        __syncwarp();
#pragma unroll
        for (int i = 0; i < KPL; ++i) {
            const int q = lane + 32 * i, c = warp * KR + q / PIECES, pc = q % PIECES;
            const int cl = __shfl_sync(0xffffffffu, cell, c);
            const bool valid = cl >= 0;
            const long long r = valid ? (long long) cl * p.kv_row + (long long) kvh * HD : 0;
            cp_async16(&S.k[c][pc * 8], reinterpret_cast<const uint4 *>(p.k + r) + pc, valid);
        }
        cp_async_commit();
    };
    auto issue_v = [&](int cell) {
        __syncwarp();
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
            const int q = lane + 32 * i, c = q / VP, pc = warp * VP + q % VP;
            const int cl = __shfl_sync(0xffffffffu, cell, c);
            const bool valid = cl >= 0;
            const long long r = valid ? (long long) cl * p.kv_row + (long long) kvh * HD : 0;
            cp_async16(&S.v[c][pc * 8], reinterpret_cast<const uint4 *>(p.v + r) + pc, valid);
        }
        cp_async_commit();
    };
    // PROG: half h of issue_k (dims [h HD / 2, (h + 1) HD / 2) of this warp's rows) and of
    // issue_v (chunk rows [h MCH / 2, (h + 1) MCH / 2) of this warp's dims), a group each.
    auto issue_k_h = [&](int cell, int h) {
        __syncwarp();
#pragma unroll
        for (int i = 0; i < KPL / 2; ++i) {
            const int q = lane + 32 * i, c = warp * KR + q / (PIECES / 2);
            const int pc = q % (PIECES / 2) + h * (PIECES / 2);
            const int cl = __shfl_sync(0xffffffffu, cell, c);
            const bool valid = cl >= 0;
            const long long r = valid ? (long long) cl * p.kv_row + (long long) kvh * HD : 0;
            cp_async16(&S.k[c][pc * 8], reinterpret_cast<const uint4 *>(p.k + r) + pc, valid);
        }
        cp_async_commit();
    };
    auto issue_v_h = [&](int cell, int h) {
        __syncwarp();
#pragma unroll
        for (int i = h * (VPL / 2); i < (h + 1) * (VPL / 2); ++i) {
            const int q = lane + 32 * i, c = q / VP, pc = warp * VP + q % VP;
            const int cl = __shfl_sync(0xffffffffu, cell, c);
            const bool valid = cl >= 0;
            const long long r = valid ? (long long) cl * p.kv_row + (long long) kvh * HD : 0;
            cp_async16(&S.v[c][pc * 8], reinterpret_cast<const uint4 *>(p.v + r) + pc, valid);
        }
        cp_async_commit();
    };
    static_assert(!PROG || (KPL % 2 == 0 && VPL % 2 == 0 && PIECES % 2 == 0), "half refills");
    int pos_n = -1;   // the next chunk's positions (EARLY: the one after next)
    int cell_nn = -1;  // EARLY: the next chunk's cells
    if (n > 0) {
        const int cell0 = cell_of(pos_at(0));
        if constexpr (PROG) {
            issue_k_h(cell0, 0);
            issue_k_h(cell0, 1);
            issue_v_h(cell0, 0);
            issue_v_h(cell0, 1);
        } else {
            issue_k(cell0);
            issue_v(cell0);
        }
        if constexpr (EARLY) {
            cell_nn = cell_of(pos_at(MCH));
            pos_n = pos_at(2 * MCH);
        } else {
            pos_n = pos_at(MCH);
        }
    }

    // q: as attn_gb_kernel.
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) {
        qm = fmaxf(qm, fabsf(*(const float *) (qb + (size_t) (i / HD) * p.q_nb1 + (i % HD) * 4)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    }
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    if (t < ROWS) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
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

    for (int c0 = 0; c0 < n; c0 += MCH) {
        const int nh = min(MCH, n - c0);
        const bool more = c0 + MCH < n;
        // The next chunk's cells (their loads land during the scores), and the positions of
        // the one after.
        int cell_n;
        if constexpr (EARLY) {
            cell_n = cell_nn;
            cell_nn = cell_of(pos_n);
            pos_n = pos_at(c0 + 3 * MCH);
        } else {
            cell_n = cell_of(pos_n);
            pos_n = pos_at(c0 + 2 * MCH);
        }
        // This warp's K rows (its V dims may still be on their way).
        cp_async_wait<PROG ? 2 : 1>();
        __syncwarp();
        // scores: warp w = cells 8w..8w+7, every dim
        {
            const int cb = warp * KR;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            if (cb < nh) {
#pragma unroll
                for (int ks = 0; ks < KS; ks += 2) {
                    if (PROG && FILL && more && ks == KS / 2) {
                        // (more: every warp has its 8 rows, so every warp gets here)
                        issue_k_h(cell_n, 0);
                    }
                    uint32_t b[4];
                    ldmatrix_x4(b, &S.k[cb + (lane & 7)][ks * 16 + 8 * (lane >> 3)]);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if constexpr (MMA) {
                            float tg[4] = {0.f, 0.f, 0.f, 0.f};
                            mma16816(tg, qh[ks + h], b + 2 * h);
                            mma16816(tg, ql[ks + h], b + 2 * h);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                sc[i] += tg[i];
                            }
                        } else {
                            sc[h] += __uint_as_float(b[2 * h] & 0x3fffffffu);
                            sc[h + 2] += __uint_as_float(b[2 * h + 1] & 0x3fffffffu);
                        }
                    }
                }
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        // This warp's K rows are free: the next chunk's come in during the softmax and p.v.
        if (FILL && more) {
            if constexpr (PROG) {
                if constexpr (HALFB) {
                    cp_async_commit();  // an empty group: the waits count the same
                } else {
                    issue_k_h(cell_n, 1);
                }
            } else {
                issue_k(cell_n);
            }
        }
        __syncthreads();
        // online softmax: as attn_gb_kernel
        {
            constexpr int PER = MCH / 8;
            constexpr float VUP = 16384.0f;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += e;
                const float pu = e * VUP;
                const __half hi = __float2half_rn(pu);
                S.ph[r][sub * PER + j] = hi;
                S.pl[r][sub * PER + j] = __float2half_rn(pu - __half2float(hi));
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        // This warp's V dims of this chunk (the next chunk's K rows may stay in flight); every
        // warp's p and alpha.
        if (more) {
            cp_async_wait<PROG ? 2 : 1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        // p.v: as attn_gb_kernel (each n-tile: the chunk's two K steps, hi then lo, chained from
        // a zero fragment)
        {
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
                ldmatrix_x4(ah, &S.ph[lr][k0 + lc]);
                ldmatrix_x4(al, &S.pl[lr][k0 + lc]);
#pragma unroll
                for (int j = 0; j < NT; j += 2) {
                    uint32_t b[4];
                    ldmatrix_x4_trans(b, &S.v[k0 + lr][warp * DW + j * 8 + lc]);
                    if constexpr (MMA) {
                        mma16816(tmp[j], ah, b);
                        mma16816(tmp[j], al, b);
                        mma16816(tmp[j + 1], ah, b + 2);
                        mma16816(tmp[j + 1], al, b + 2);
                    } else {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            tmp[j + (i >> 1)][i & 1] += __uint_as_float((b[i] ^ ah[i] ^ al[i]) & 0x3fffffffu);
                        }
                    }
                }
                if (PROG && FILL && more && k0 == 0) {
                    issue_v_h(cell_n, 0);  // rows 0..15 of this warp's dims are read
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * VDOWN);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * VDOWN);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * VDOWN);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * VDOWN);
            }
        }
        // This warp's V dims are free: the next chunk's come in during the next scores.
        if (FILL && more) {
            if constexpr (PROG) {
                if constexpr (HALFB) {
                    cp_async_commit();
                } else {
                    issue_v_h(cell_n, 1);
                }
            } else {
                issue_v(cell_n);
            }
        }
    }
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int d = warp * DW + j * 8 + 2 * tig;
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

// ---------------------------------------------------------------------------------------------
// #1176 r40: register-only probes of attn_gw_kernel's mma.sync patterns (`legion qsa-bench`
// only, wrong output). Same grid and block as attn_gw_kernel and the same shared-memory request
// (two CTAs per SM), or about 60 KB (one CTA per SM, the launcher's choice); for every 32-key
// chunk of a token's selection each warp issues 64 mma.m16n8k16, as attn_gw_kernel does, from
// registers only (q's hi / lo fragments as attn_gw_kernel holds them, B fragments made fresh per
// chunk by a few integer ops, so no two products are the same). PAT:
// 0 d1: every product from a zero fragment, added to the F32 sums
// 1 pair: attn_gw_kernel's scores twice (hi then lo into one fragment from zero, then added)
// 2 pv: attn_gw_kernel's p.v twice (8 fragments, each a chain of 4 from zero, then an FMA)
// 3 gw: the scores once and the p.v once, independent of each other
// 4 chain: all 64 into one fragment (the mma latency)
// 5 pair16 / 6 pv16: 1 and 2 with F16 accumulation (m16n8k16.f16.f16.f16.f16)
// 7 gwdep: 3 with p made from the scores and attn_gw_kernel's two CTA barriers per chunk
__device__ __forceinline__ void mma16816_h(uint32_t * c, const uint32_t * a, const uint32_t * b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
        "{%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

__device__ __forceinline__ float h2sum(uint32_t x) {
    const __half2 h = *reinterpret_cast<const __half2 *>(&x);
    return __low2float(h) + __high2float(h);
}

template <int PAT>
__global__ void __launch_bounds__(THREADS, 2) attn_pipe_kernel(legion_qsa_attn p) {
    constexpr int KS = 16, NT = 8;
    constexpr float VDOWN = 1.0f / 16384.0f;
    const int tok = blockIdx.x;
    const int n = p.sel[(size_t) tok * p.sel_row];
    const int t = threadIdx.x;
    // Halves in [1, 2) (hi) and about 2^-11 (lo); a B fragment's mantissas change per chunk.
    auto mant = [](uint32_t v) { return (v & 0x3ffu) * 0x00010001u; };
    uint32_t qh[KS][4], ql[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qh[ks][i] = 0x3c003c00u | mant(t * 7 + ks * 4 + i);
            ql[ks][i] = 0x10001000u | mant(t * 5 + ks * 3 + i);
        }
    }
    uint32_t kb[4], vb[4], pb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kb[i] = 0x3c003c00u | mant(t * 3 + i);
        vb[i] = 0x38003800u | mant(t * 11 + i);
        pb[i] = 0x3c003c00u | mant(t * 13 + i);
    }
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }
    // attn_gw_kernel's scores of one chunk (salt: which pass of which chunk)
    auto scores = [&](uint32_t salt) {
#pragma unroll
        for (int ks = 0; ks < KS; ks += 2) {
            uint32_t b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                b[i] = kb[i] ^ mant(salt + ks);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if constexpr (PAT == 5) {
                    uint32_t tg[2] = {0u, 0u};
                    mma16816_h(tg, qh[ks + h], b + 2 * h);
                    mma16816_h(tg, ql[ks + h], b + 2 * h);
                    sc[h] += h2sum(tg[0]);
                    sc[h + 2] += h2sum(tg[1]);
                } else if constexpr (PAT == 0) {
                    float tg[4] = {0.f, 0.f, 0.f, 0.f}, tl[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(tg, qh[ks + h], b + 2 * h);
                    mma16816(tl, ql[ks + h], b + 2 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sc[i] += tg[i] + tl[i];
                    }
                } else {
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
    };
    // attn_gw_kernel's p.v of one chunk (pd: what p depends on)
    auto pv = [&](uint32_t salt, uint32_t pd) {
        float tmp[NT][4];
        uint32_t th[NT][2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
            th[j][0] = th[j][1] = 0u;
        }
#pragma unroll
        for (int k0 = 0; k0 < 2; ++k0) {
            uint32_t ah[4], al[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ah[i] = pb[i] ^ mant(salt + k0 + pd);
                al[i] = (pb[i] ^ mant(salt + k0 + 7)) & 0x13ff13ffu;
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                uint32_t b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    b[i] = vb[i] ^ mant(salt + k0 * 8 + j);
                }
                if constexpr (PAT == 6) {
                    mma16816_h(th[j], ah, b);
                    mma16816_h(th[j], al, b);
                    mma16816_h(th[j + 1], ah, b + 2);
                    mma16816_h(th[j + 1], al, b + 2);
                } else {
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                    mma16816(tmp[j + 1], ah, b + 2);
                    mma16816(tmp[j + 1], al, b + 2);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if constexpr (PAT == 6) {
                acc[j][0] = fmaf(acc[j][0], 0.75f, h2sum(th[j][0]) * VDOWN);
                acc[j][2] = fmaf(acc[j][2], 0.75f, h2sum(th[j][1]) * VDOWN);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[j][i] = fmaf(acc[j][i], 0.75f, tmp[j][i] * VDOWN);
                }
            }
        }
    };
    float tc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int c0 = 0; c0 < n; c0 += MCH) {
        const uint32_t salt = (uint32_t) c0 >> 3;
        if constexpr (PAT == 0 || PAT == 1 || PAT == 5) {
            scores(salt);
            scores(salt + 1);
        } else if constexpr (PAT == 2 || PAT == 6) {
            pv(salt, 0);
            pv(salt + 1, 0);
        } else if constexpr (PAT == 3) {
            scores(salt);
            pv(salt, 0);
        } else if constexpr (PAT == 7) {
            scores(salt);
            __syncthreads();
            const uint32_t pd = __float_as_uint(sc[0] + sc[1] + sc[2] + sc[3]);
            __syncthreads();
            pv(salt, pd);
        } else {
#pragma unroll
            for (int k = 0; k < 64; ++k) {
                const uint32_t b[2] = {kb[0] ^ mant(salt + k), kb[1]};
                mma16816(tc, (k & 1) ? ql[k >> 2] : qh[k >> 2], b);
            }
        }
    }
    float r = sc[0] + sc[1] + sc[2] + sc[3] + tc[0] + tc[1] + tc[2] + tc[3];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        r += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    }
    p.dst[((size_t) tok * p.n_head + (size_t) blockIdx.y * (p.n_head / p.n_head_kv)) * p.head_dim +
          t] = r;
}
