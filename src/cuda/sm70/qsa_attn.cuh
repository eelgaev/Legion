// qsa_attn.cu, sm_70 (Volta) and newer: the prompt kernel on mma.sync.m8n8k4 (attn_kernel and
// its helpers). m8n8k4 is valid on every sm_70+ device; qsa_attn.cu's launch_attn runs it where
// use_mma<HD>() says no (Volta, or no sm_80 code in the binary).
//
// Not a standalone header: qsa_attn.cu includes it inside its anonymous namespace, after BLK and
// the shared prompt-kernel constants (CH, THREADS, ROWS, GROUPS) it uses.

// ---------------------------------------------------------------------------------------------
// The attention kernel, adapted from Strata-AC922's prompt_attn_v70_kernel
// (src/kernels/cuda/qsa_prompt_attn.cu, MIT, Copyright (c) 2026 Niko1221 and the Strata
// contributors; its m8n8k4 fragment layout is from Klaus Friedel's PR #600 to Strata).
//
// One CTA of AT_WARPS = 8 warps per (token, kv head) walks the token's selected cells in chunks
// of CH with an online softmax. The G query heads of the kv head (padded to 16 rows, two row
// blocks rb of 8) share every K/V load. Warp w works on row block w >> 2 only, in the role
// wq = w & 3 that a warp of the former 4-warp CTA had for both row blocks (#784: 2 CTAs x 8
// warps per SM instead of 2 x 4, the shared memory is the same). Every output element sees the
// same products in the same order as before, so the output is bitwise the same. While a chunk computes, the next chunk's K and V rows are already on their way into
// registers. q.k and p.v are Volta mma.sync.m8n8k4 products (F16 in, F32 out): per warp four
// independent 8x8x4 products, one per quad pair (QP q = lanes 4q..4q+3 and 4q+16..4q+19,
// thread j = (lane & 3) + 4 * (lane >> 4) within it). The thread holds row j of A and column j
// of B (k 0..3 as two registers); C register i is row (lane & 1) + 4 * (lane >> 4) + 2 * ((i >>
// 1) & 1), column (lane & 2) + 4 * (i >> 2) + (i & 1) of its QP's 8x8 tile.
//
// Precision: q and p are split into F16 hi + lo halves (each a product of its own), so they
// keep ~22 bits; K and V are the cache's F16 values (QUANT: a q8_0 / q4_0 cache's values as
// F16, see `qunpack`). Volta's tensor cores truncate (round toward
// zero) when they add into an accumulator. By default every m8n8k4 product starts from a zero
// fragment and is added to the running sums with ordinary F32 adds (one K step = 4 elements).
// With `chain` (Strata-AC922 ea91c64) the hi and the lo products each accumulate in their own
// chain across the K steps and the two chains are added in F32 at the end. The truncation that
// matters is the hi chain's: each add truncates to F32 relative to the running sum, so the error
// grows with the chain length, here HD / 32 K steps for q.k (8 at HD 256) and CH / 4 for p.v
// (4), both restarted per chunk. Re-check the accuracy test before growing CH or the group size.
// Measured against F64 (cuda_matches_oracle_and_dense_path): <= 6.6e-7 with the chain vs 3.0e-7
// without, about 2.2x less accurate but far inside its 3e-5 tolerance.
//   q.k: the head dims in 8 groups of HD / 8: warp role wq, QP q takes group 2 wq + (q >> 1) for
//        the cells 8 (q & 1)..+7 of the chunk. The softmax adds the groups in a fixed order.
//   p.v: warp role wq owns dims [wq HD / 4, (wq + 1) HD / 4); QP q takes HD / 16 of them
//        (HD / 128 n-tiles of 8); k = the CH cells in steps of 4.
// mma.m8n8k4 is valid PTX on every sm_70+ device (newer ones run it slower than their own
// MMA shapes).

__device__ __forceinline__ void mma884(float * c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]),
          "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// One m8n8k4 product from a zero fragment, added to `acc` in F32.
__device__ __forceinline__ void mma884_add(float * acc, uint32_t a0, uint32_t a1, uint32_t b0,
                                           uint32_t b1) {
    float f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    mma884(f, a0, a1, b0, b1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += f[i];
    }
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {  // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t *>(&h);
}

// attn_kernel's CTA: 8 warps, two per SM (its shared memory, ~44.5 KB at HD 256, allows two).
// At 4 warps the kernel ran 8 of an SM's 64 warp slots and stalled on load latency and barriers
// (#706); at 8 the same two CTAs keep 16 warps in flight. The sm_80 kernel keeps THREADS.
constexpr int AT_WARPS = 8;
constexpr int AT_THREADS = AT_WARPS * 32;
static_assert(ROWS == 16, "two row blocks of 8, one per half of the warps");

template <int HD>
struct Smem {
    static constexpr int QS = HD + 8;  // halves per q row (bank-spread fragment loads)
    static constexpr int KR = HD + 8;  // halves per K / V row (16-byte aligned rows)
    __half qh[ROWS][QS];
    __half ql[ROWS][QS];
    __half k[CH][KR];
    __half v[CH][KR];
    alignas(16) float part[GROUPS][ROWS][CH + 4];  // q.k per dim group (quantized: staging too)
    // The chunk's probabilities times 2^14 as F16 hi + lo halves, the p.v A fragments (split
    // once per element by the softmax, not again by each of the 16 threads that read it).
    __half ph[ROWS][CH + 8];
    __half pl[ROWS][CH + 8];
    float qmax[AT_WARPS];
    float alpha[ROWS];
    float lsum[ROWS];
    float mrow[ROWS];
};

// CHAIN: the hi and the lo products each accumulate in one m8n8k4 chain across the K steps
// (Strata-AC922 ea91c64, from fks's PR #600 kernel; chain lengths HD / 32 and CH / 4, see
// Precision above): 23.7 vs 28.6 ms per 4,096 queries at 32K on a V100. Otherwise each K step's
// product goes through a zero fragment and is added in F32.
// KT / VT (#263, after Strata-AC922 fdd9b6f): the K / V cache types, 0 F16 (both) or 1 q8_0 /
// 2 q4_0. Quantized, the threads fetch the next chunk's K and V rows as they are stored with
// coalesced LW-byte loads and stage them in `part` after the chunk's p.v (the softmax is done
// with it); at the chunk's start each thread turns a run of whole 32-value blocks of one row into
// F16 values where the F16 form stores them (`qrun_unpack`); the rest is the same kernel.
template <int HD, bool CHAIN, int KT, int VT>
__global__ void __launch_bounds__(AT_THREADS, 2) attn_kernel(legion_qsa_attn p) {
    constexpr int THREADS = AT_THREADS;  // this kernel's CTA (the shared THREADS is sm_80's)
    constexpr bool QUANT = KT != 0;
    static_assert((KT == 0) == (VT == 0), "F16 K and V or both quantized");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<HD> & S = *reinterpret_cast<Smem<HD> *>(smem_raw);
    constexpr int DW = HD / 4;            // p.v dims per warp role
    constexpr int DG = HD / GROUPS;       // q.k dims per group
    constexpr int KS = DG / 4;            // q.k K steps per group
    constexpr int NT = DW / 32;           // p.v n-tiles of 8 per QP
    constexpr int DQ = DW / 4;            // p.v dims per QP
    constexpr int PIECES = HD * 2 / 16;   // 16-byte pieces per K or V row
    constexpr int PER_T = CH * PIECES / THREADS;  // pieces of K (and of V) per thread
    static_assert(CH * PIECES % THREADS == 0, "pieces cover the threads");
    constexpr int BPR = HD / QB;   // QUANT: blocks per K or V row of a head
    constexpr int R = BPR / 4;     // blocks per thread: a run per warp
    constexpr int LW = HD % 256 == 0 ? 16 : 8;  // load bytes (a head's row: BPR x 34 or 18 bytes)
    constexpr int BK = QUANT ? BPR * qbytes<KT>() : LW;  // bytes per K / V row of a head
    constexpr int BV = QUANT ? BPR * qbytes<VT>() : LW;
    constexpr int PK = BK / LW, PV = BV / LW;            // LW-byte pieces per K / V row
    constexpr int NPC = CH * (PK + PV);                  // pieces per chunk
    constexpr int NP = QUANT ? (NPC + THREADS - 1) / THREADS : 1;  // per thread
    using Piece = typename LoadPiece<LW>::T;
    static_assert(!QUANT || (BPR % 4 == 0 && AT_WARPS == 8 && CH == 16),
                  "warps 0-3 unpack 16 K + 16 V rows, a run of blocks each");
    static_assert(BK % LW == 0 && BV % LW == 0 && CH * (BK + BV) <= (int) sizeof(Smem<HD>::part),
                  "whole pieces per row; the chunk's rows fit the staging");
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
    const int rb = warp >> 2, wq = warp & 3;  // the warp's row block and its role in it
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);
    const int crow = (lane & 1) + 4 * (lane >> 4);
    const int ccol = lane & 2;

    // The index lookups run ahead of the K / V loads (#712): a cell takes two dependent loads
    // (its `ids` entry, then the block `table` entry) before its rows can be fetched, which at 2
    // CTAs per SM nothing hides. Lane l < CH of every warp holds `cell_a`, the cell of entry
    // c0 + l of the chunk the next fetch(c0) loads, and `pos_b`, the position of entry
    // c0 + CH + l; each fetch issues its K / V loads at once from `cell_a`, then the table load
    // of the chunk after (from `pos_b`, loaded a chunk ago) and the ids load of the one after
    // that. The rows loaded and their order are unchanged, so is the output (bitwise).
    auto lookup = [&](int pos) { return __ldg(table + pos / BLK) * BLK + pos % BLK; };
    int cell_a = 0, pos_b = 0;
    if (lane < CH) {
        if (lane < n) {
            cell_a = lookup(__ldg(ids + lane));
        }
        if (CH + lane < n) {
            pos_b = __ldg(ids + CH + lane);
        }
    }
    auto advance = [&](int c0) {
        if (lane < CH) {
            if (c0 + CH + lane < n) {
                cell_a = lookup(pos_b);
            }
            if (c0 + 2 * CH + lane < n) {
                pos_b = __ldg(ids + c0 + 2 * CH + lane);
            }
        }
    };

    // The next chunk's K / V pieces of this thread: piece t + THREADS i is cell
    // (t + THREADS i) / PIECES, 16-byte piece t % PIECES of its row. Quantized: piece
    // t + THREADS i of the chunk's CH K rows then CH V rows. The cells come from lane c's
    // `cell_a`.
    uint4 kx[QUANT ? 1 : PER_T], vx[QUANT ? 1 : PER_T];
    Piece qx[NP];
    auto fetch = [&](int c0) {
        if constexpr (QUANT) {
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int idx = t + THREADS * i;
                const bool v = idx >= CH * PK;
                const int j = v ? idx - CH * PK : idx;
                const int c = v ? j / PV : j / PK, pc = v ? j % PV : j % PK;
                const int cell = __shfl_sync(0xffffffffu, cell_a, c & 31);
                qx[i] = Piece{};
                if (idx < NPC && c0 + c < n) {
                    const char * row = v ? (const char *) p.v + (long long) cell * p.v_nb1 + kvh * BV
                                         : (const char *) p.k + (long long) cell * p.k_nb1 + kvh * BK;
                    qx[i] = __ldg(reinterpret_cast<const Piece *>(row) + pc);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < PER_T; ++i) {
                const int idx = t + THREADS * i;
                const int c = idx / PIECES, pc = idx % PIECES;
                const long long cell = __shfl_sync(0xffffffffu, cell_a, c);
                kx[i] = vx[i] = make_uint4(0, 0, 0, 0);
                if (c0 + c < n) {
                    const long long r = cell * p.kv_row + (long long) kvh * HD;
                    kx[i] = __ldg(reinterpret_cast<const uint4 *>(p.k + r) + pc);
                    vx[i] = __ldg(reinterpret_cast<const uint4 *>(p.v + r) + pc);
                }
            }
        }
        advance(c0);
    };
    fetch(0);
    // Quantized: the fetched rows, as stored, in `part` while it is free (after the chunk's
    // softmax, here once p.v is done: the loads had the whole chunk), for the next chunk's start
    // to unpack.
    char * const stage = reinterpret_cast<char *>(&S.part[0][0][0]);
    auto stage_put = [&]() {
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int idx = t + THREADS * i;
            if (idx < NPC) {
                reinterpret_cast<Piece *>(stage)[idx] = qx[i];
            }
        }
    };

    // q: G heads + zero rows, scaled by a power of two that puts its largest value near 2^14
    // (exact, and the lo halves stay out of F16's subnormal range), split into hi + lo.
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
    __syncthreads();
    qm = S.qmax[0];
#pragma unroll
    for (int w = 1; w < AT_WARPS; ++w) {
        qm = fmaxf(qm, S.qmax[w]);  // a max: the same in any order
    }
    int qe = 0;
    if (qm > 0.0f) {
        frexpf(qm, &qe);  // qm < 2^qe
    }
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(p.scale_log2, qe - 14);
    for (int i = t; i < ROWS * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x =
            h < G ? *(const float *) (qb + (size_t) h * p.q_nb1 + d * 4) * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < ROWS) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }

    if constexpr (QUANT) {
        stage_put();
    }

    constexpr float VUP = 16384.0f, VDOWN = 1.0f / 16384.0f;  // p's scale in p.v
    float acc[NT][8];  // row block rb
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        __syncthreads();  // the previous chunk's p.v is done with k, v, s
        if constexpr (QUANT) {
            // Turn run `wq` of the staged K or V row of cell lane % 16 into F16 (rows 68, 36 or 34
            // words apart: no bank conflicts). One type: warps 0-3, K in lanes 0-15 and V in
            // 16-31. Two types: lanes 0-15, K in warps 0-3 and V in warps 4-7 (one code per warp).
            const int qc = lane & 15;
            if constexpr (KT == VT) {
                if (warp < 4) {
                    const bool v = lane >= 16;
                    qrun_unpack<KT, R, LW>(stage + (v ? CH * BK + qc * BV : qc * BK), wq,
                                           v ? &S.v[qc][0] : &S.k[qc][0]);
                }
            } else if (lane < 16) {
                if (warp < 4) {
                    qrun_unpack<KT, R, LW>(stage + qc * BK, wq, &S.k[qc][0]);
                } else {
                    qrun_unpack<VT, R, LW>(stage + CH * BK + qc * BV, wq, &S.v[qc][0]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < PER_T; ++i) {
                const int idx = t + THREADS * i;
                const int c = idx / PIECES, pc = idx % PIECES;
                *reinterpret_cast<uint4 *>(&S.k[c][pc * 8]) = kx[i];
                *reinterpret_cast<uint4 *>(&S.v[c][pc * 8]) = vx[i];
            }
        }
        if (c0 + CH < n) {
            fetch(c0 + CH);
        }
        __syncthreads();
        // scores: row block rb of the warp
        {
            float tg[8], tgl[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) tg[i] = tgl[i] = 0.0f;
            const int grp = wq * 2 + (qp >> 1);
            const int cell_b = (qp & 1) * 8 + jj;
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                const int d0 = grp * DG + ks * 4;
                const uint2 raw = *reinterpret_cast<const uint2 *>(&S.k[cell_b][d0]);
                const uint2 ah = *reinterpret_cast<const uint2 *>(&S.qh[rb * 8 + jj][d0]);
                const uint2 al = *reinterpret_cast<const uint2 *>(&S.ql[rb * 8 + jj][d0]);
                if constexpr (CHAIN) {
                    mma884(tg, ah.x, ah.y, raw.x, raw.y);
                    mma884(tgl, al.x, al.y, raw.x, raw.y);
                } else {
                    mma884_add(tg, ah.x, ah.y, raw.x, raw.y);
                    mma884_add(tgl, al.x, al.y, raw.x, raw.y);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
                const int cell = (qp & 1) * 8 + ccol + 4 * (i >> 2) + (i & 1);
                S.part[grp][row][cell] = tg[i] + tgl[i];
            }
        }
        __syncthreads();
        // online softmax (base 2): row t / 16, cell t % 16, one per thread. The row's sum is a
        // pairwise tree over its 16 lanes: what the 4-warp kernel's sum of each thread's two
        // cells (0 + e0 + e1, an add with 0 is exact) and its 8-lane tree gave, bit for bit.
        {
            static_assert(CH == 16 && THREADS == ROWS * CH, "a thread per (row, cell)");
            const int r = t >> 4, c = t & 15;
            float x, mx;
            {
                float sum = 0.0f;
#pragma unroll
                for (int g = 0; g < GROUPS; ++g) {
                    sum += S.part[g][r][c];
                }
                x = c < nh ? sum * qdown : -INFINITY;
                mx = x;
            }
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = x == -INFINITY ? 0.0f : exp2f(x - m_new);
            {
                // p (at most 1) times 2^14 (exact) so its lo half stays normal, split into hi + lo
                const float pu = sum * VUP;
                const __half hi = __float2half_rn(pu);
                S.ph[r][c] = hi;
                S.pl[r][c] = __float2half_rn(pu - __half2float(hi));
            }
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (c == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: the hi and the lo halves of p 2^14 (from the softmax) each times V
        {
            float tmp[NT][8], tmpl[NT][8];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) tmp[nt][i] = tmpl[nt][i] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int cc = ks * 4;
                const uint2 ah = *reinterpret_cast<const uint2 *>(&S.ph[rb * 8 + jj][cc]);
                const uint2 al = *reinterpret_cast<const uint2 *>(&S.pl[rb * 8 + jj][cc]);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int d = wq * DW + qp * DQ + nt * 8 + jj;
                    const __half2 h0 = __halves2half2(S.v[cc][d], S.v[cc + 1][d]);
                    const __half2 h1 = __halves2half2(S.v[cc + 2][d], S.v[cc + 3][d]);
                    const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&h1);
                    if constexpr (CHAIN) {
                        mma884(tmp[nt], ah.x, ah.y, b0, b1);
                        mma884(tmpl[nt], al.x, al.y, b0, b1);
                    } else {
                        mma884_add(tmp[nt], ah.x, ah.y, b0, b1);
                        mma884_add(tmpl[nt], al.x, al.y, b0, b1);
                    }
                }
            }
            const float a0 = S.alpha[rb * 8 + crow];
            const float a1 = S.alpha[rb * 8 + crow + 2];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float a = ((i >> 1) & 1) ? a1 : a0;
                    acc[nt][i] = fmaf(acc[nt][i], a, (tmp[nt][i] + tmpl[nt][i]) * VDOWN);
                }
        }
        if constexpr (QUANT) {
            if (c0 + CH < n) {
                stage_put();  // softmax is done with `part`; read after the next chunk's sync
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = rb * 8 + crow + 2 * h;
        if (row >= G) {
            continue;
        }
        const float l = S.lsum[row];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int i = hh * 4 + h * 2;  // registers i, i+1: row h, columns ccol + 4 hh + {0, 1}
                const int d = wq * DW + qp * DQ + nt * 8 + ccol + 4 * hh;
                *reinterpret_cast<float2 *>(out + (size_t) row * HD + d) =
                    make_float2(acc[nt][i] * inv, acc[nt][i + 1] * inv);
            }
    }
}
