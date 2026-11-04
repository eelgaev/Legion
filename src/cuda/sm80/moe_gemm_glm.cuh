// moe_gemm.cu, sm_80 and newer, GLM-5.3-Flash only (#571): gemm_glm_kernel, the routed-expert
// GEMMs on Project Maya's IQ2 experts (Maya-S: IQ2_XXS gate/up, IQ2_S down) in the shape of
// gemm_frag_kernel: each warp converts its own B fragments from the raw super-blocks in shared
// memory into registers, so there is no F16 weight tile, one barrier per slice, and no lane
// converts values it does not use.
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace, after
// sm80/moe_gemm_frag.cuh (it uses Tile, group_of, store_tile, row_base, ldsm_x4, mma16816, the
// cp.async helpers, iq2_code and iq2_value). Only legion_moe_gemm_glm launches it: the kernels
// the other models run are not touched.
//
// Values: Piece<T_IQ2_XXS> and Piece<T_IQ2_S> compute each value as dd * v * sign in F32 (v one
// of the codebooks' 8, 25 and 43, dd the scale of its group of 8) and round it to F16 once. The
// sign multiplies an exact product and F16 rounding is symmetric, so the value's bits are the
// F16 rounding of dd * v with the sign bit set by the sign. Here each k16 step (two groups of 8,
// one scale) rounds its three products once; a byte permute picks two of them per half2 by the
// values' 2-bit codes, and an XOR sets the sign bits. A lane (g, tg) needs values 2 tg, 2 tg + 1
// of each group, so shared-memory tables give, per codebook entry and tg, the byte permute's
// selector, and (IQ2_XXS, 7-bit sign indices with a parity bit) per sign index and tg, the XOR
// mask. The same halves, the same MMAs in the same order: the bits of gemm_raw_kernel.
//
// A block: TPB consecutive plan tiles of one expert (group_of) and BN weight rows (16 per warp;
// gate/up interleaved as gemm_frag_kernel's). Weight stages are whole super-blocks (256 values,
// four slices), copied as the aligned 16-byte pieces that cover them: the rows are a whole
// number of 16-byte pieces at Maya-S's sizes (16 x 66 = 1056 bytes at n_embd 4096, 8 x 82 = 656
// at n_ff 2048; checked by the launcher), so a stage's misalignment is the same for every row
// and the pieces never pass the row's end.

// The word at byte `off` (even) of a 4-byte aligned buffer.
__device__ __forceinline__ uint32_t ld_u32_even(const unsigned char * b, int off) {
    const uint32_t * w = (const uint32_t *) (b + (off & ~3));
    return (off & 2) ? __funnelshift_r(w[0], w[1], 16) : w[0];
}

// The byte permute of the selectors below: their nibbles are byte numbers 0-5, so the PTX
// default mode's sign-replicate bit is clear and this is __byte_perm without the nibble masking
// the compiler adds to it.
__device__ __forceinline__ uint32_t prmt_sel(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// IQ2_XXS: 66 bytes per 256 values: d, then per sub-block of 32 the grid indices (4 bytes) and
// a word of four 7-bit sign indices and the scale (top 4 bits). A lane addresses the tables from
// its own tg's column: entry e of the selectors at byte 8 e, of the sign masks at 16 e.
struct GlmIq2xxs {
    static constexpr int SB = 66, RB = 80;
    static constexpr int SMEM_TAB = 256 * 4 * 2 + 128 * 4 * 4;  // selectors, sign masks
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint32_t * sgm = (uint32_t *) t;
        uint16_t * sel = (uint16_t *) (t + 128 * 4 * 4);
        for (int i = tid; i < 256 * 4; i += nt) {
            const uint32_t cc = (iq2_code(iq2xxs_grid[i / 4]) >> (4 * (i % 4))) & 15;
            sel[i] = (uint16_t) (0x1010u + (cc & 3) * 0x22u + (cc >> 2) * 0x2200u);
        }
        for (int i = tid; i < 128 * 4; i += nt) {
            const uint32_t s7 = i / 4, sb = ((s7 | ((__popc(s7) & 1) << 7)) >> (2 * (i % 4))) & 3;
            sgm[i] = ((sb & 1) << 15) | ((sb >> 1) << 31);
        }
    }
    // this lane's fragments of sub-block ib (k16 steps 2 (ib % 2) and + 1 of its slice)
    __device__ static void frag(const unsigned char * t, const unsigned char * slot, int sh, int ib, int tg,
                                uint32_t (*bq)[2]) {
        const unsigned char * sgm = t + 4 * tg;
        const unsigned char * sel = t + 128 * 4 * 4 + 2 * tg;
        const float d0 = __half2float(__ushort_as_half(*(const uint16_t *) (slot + sh)));
        const uint32_t qx = ld_u32_even(slot, sh + 2 + 8 * ib), qy = ld_u32_even(slot, sh + 6 + 8 * ib);
        const float dd = d0 * (0.5f + (qy >> 28)) * 0.25f;
        const uint32_t h0 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(0)));
        const uint32_t h1 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(1)));
        const uint32_t h2 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(2)));
        const uint32_t lo = h0 | (h1 << 16);
#pragma unroll
        for (int gr = 0; gr < 4; gr++) {
            const uint32_t s = *(const uint16_t *) (sel + 8 * __byte_perm(qx, 0, 0x4440 + gr));
            const uint32_t m = *(const uint32_t *) (sgm + ((gr == 0 ? qy << 4 : qy >> (7 * gr - 4)) & 0x7F0u));
            bq[gr / 2][gr % 2] = prmt_sel(lo, h2, s) ^ m;
        }
    }
};

// IQ2_S: 82 bytes per 256 values: d, the groups' low grid bytes (32), their sign bytes (32), per
// sub-block the grid indices' high bits (qh, 8) and two scale nibbles (8).
struct GlmIq2s {
    static constexpr int SB = 82, RB = 96;
    static constexpr int SMEM_TAB = 1024 * 4 * 2;  // selectors
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint16_t * sel = (uint16_t *) t;
        for (int i = tid; i < 1024 * 4; i += nt) {
            const uint32_t cc = (iq2_code(iq2s_grid[i / 4]) >> (4 * (i % 4))) & 15;
            sel[i] = (uint16_t) (0x1010u + (cc & 3) * 0x22u + (cc >> 2) * 0x2200u);
        }
    }
    __device__ static void frag(const unsigned char * t, const unsigned char * slot, int sh, int ib, int tg,
                                uint32_t (*bq)[2]) {
        const unsigned char * sel = t + 2 * tg;  // entry e at byte 8 e
        const float d0 = __half2float(__ushort_as_half(*(const uint16_t *) (slot + sh)));
        const uint32_t ql = ld_u32_even(slot, sh + 2 + 4 * ib);
        const uint32_t sg = ld_u32_even(slot, sh + 34 + 4 * ib) >> (2 * tg);
        const uint32_t qh = slot[sh + 66 + ib];
        const uint32_t sc = slot[sh + 74 + ib];
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const float dd = d0 * (0.5f + ((sc >> (4 * s)) & 0xf)) * 0.25f;
            const uint32_t h0 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(0)));
            const uint32_t h1 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(1)));
            const uint32_t h2 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(2)));
            const uint32_t lo = h0 | (h1 << 16);
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const int gr = 2 * s + q;
                const int idx = (int) ((ql >> (8 * gr)) & 0xFF) | (((int) qh << (8 - 2 * gr)) & 0x300);
                const uint32_t sb = sg >> (8 * gr);
                bq[s][q] = prmt_sel(lo, h2, *(const uint16_t *) (sel + 8 * idx)) ^
                           (((sb << 15) & 0x8000u) | ((sb << 30) & 0x80000000u));
            }
        }
    }
};

// IQ3_XXS (Maya-S's down in its first and last four MoE layers): 98 bytes per 256 values: d, the
// grid indices (8 bytes per sub-block of 32: two 4-value entries per group of 8), per sub-block a
// word of four 7-bit sign indices and the scale (top 4 bits). Piece<T_IQ3_XXS> rounds
// dd * byte * sign to F16, and as above that is the F16 rounding of dd * byte with the sign bit
// set by the sign. Its grid has more than three byte values, so each value is a product of its
// own: lane tg's values 2 tg, 2 tg + 1 of group gr are bytes (2 tg) % 4 and + 1 of entry
// 2 gr + tg / 2. Tables: the grid (256 words), and per sign index and tg IQ2_XXS's XOR masks
// (ksigns_iq2xs is the 7-bit index with its parity bit on top).
struct GlmIq3xxs {
    static constexpr int SB = 98, RB = 112;
    static constexpr int SMEM_TAB = 128 * 4 * 4 + 256 * 4;  // sign masks, grid
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint32_t * sgm = (uint32_t *) t;
        uint32_t * grd = (uint32_t *) (t + 128 * 4 * 4);
        for (int i = tid; i < 128 * 4; i += nt) {
            const uint32_t s7 = i / 4, sb = ((s7 | ((__popc(s7) & 1) << 7)) >> (2 * (i % 4))) & 3;
            sgm[i] = ((sb & 1) << 15) | ((sb >> 1) << 31);
        }
        for (int i = tid; i < 256; i += nt) {
            grd[i] = iq3xxs_grid[i];
        }
    }
    __device__ static void frag(const unsigned char * t, const unsigned char * slot, int sh, int ib, int tg,
                                uint32_t (*bq)[2]) {
        const unsigned char * sgm = t + 4 * tg;
        const uint32_t * grd = (const uint32_t *) (t + 128 * 4 * 4);
        const float d0 = __half2float(__ushort_as_half(*(const uint16_t *) (slot + sh)));
        const uint32_t aux = ld_u32_even(slot, sh + 66 + 4 * ib);
        const float dd = d0 * (0.5f + (aux >> 28)) * 0.5f;
        // entries 2 gr + tg / 2: bytes tg / 2 and 2 + tg / 2 of each grid-index word
        const uint32_t q0 = ld_u32_even(slot, sh + 2 + 8 * ib), q1 = ld_u32_even(slot, sh + 6 + 8 * ib);
        const int bs = 16 * (tg & 1);
#pragma unroll
        for (int gr = 0; gr < 4; gr++) {
            const uint32_t qw = gr < 2 ? q0 : q1;
            const uint32_t w = grd[(qw >> (16 * (gr & 1) + 8 * (tg >> 1))) & 0xFF] >> bs;
            const uint32_t m = *(const uint32_t *) (sgm + ((gr == 0 ? aux << 4 : aux >> (7 * gr - 4)) & 0x7F0u));
            const __half2 h = __floats2half2_rn(dd * (float) (w & 0xFF), dd * (float) ((w >> 8) & 0xFF));
            bq[gr / 2][gr % 2] = *(const uint32_t *) &h ^ m;
        }
    }
};

// IQ2_S for the decode kernel (#871): GlmIq2s's halves with fewer shared-memory wavefronts. Its
// selector table is read at random: the lanes of the 8 rows g of a fragment hit the same bank
// for different words about 2.5 times a load. Here each row g reads its own copy of the
// codebook, which sits in banks 4 g .. 4 g + 3 only, so a load is one wavefront: entry e at byte
// 16 g + 2 e + 14 (e & ~7) (2 (e % 8) + 128 (e / 8)). A copy's entry is the codebook entry's 2-bit
// codes spread for the lanes (the codes of values 2 tg at bits 2 tg, of values 2 tg + 1 at
// 8 + 2 tg), and the lane computes its selector from it. 16 KB of tables instead of 8 KB; the
// down shape's two blocks per SM still fit. (The same for IQ2_XXS gate/up and IQ3_XXS down, with
// their sign masks computed instead of looked up to stay within gate/up's shared memory, was
// 7% slower and level: those kernels are issue bound, not bank bound.)

// The selector of GlmIq2s's table from a spread code word (see above): 0x1010 + c0 * 0x22 +
// c1 * 0x2200 for the codes c0, c1 of values 2 tg, 2 tg + 1.
__device__ __forceinline__ uint32_t glm_sel(uint32_t w, int tg) { return ((w >> (2 * tg)) & 0x303u) * 0x22u + 0x1010u; }

// The spread code word of a codebook entry's 16-bit code (iq2_code).
__device__ __forceinline__ uint16_t glm_spread(uint32_t code) {
    uint32_t w = 0;
#pragma unroll
    for (int tg = 0; tg < 4; tg++) {
        w |= ((code >> (4 * tg)) & 3u) << (2 * tg);
        w |= ((code >> (4 * tg + 2)) & 3u) << (8 + 2 * tg);
    }
    return (uint16_t) w;
}

// This lane's copy of a per-row table (row g = lane / 4).
__device__ __forceinline__ const unsigned char * glm_copy(const unsigned char * t) {
    return t + 16 * ((threadIdx.x & 31) >> 2);
}

struct GlmIq2sDec {
    static constexpr int SB = 82, RB = 96;
    static constexpr int SMEM_TAB = 8 * 1024 * 2;  // per-row copies of the spread codes
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        for (int e = tid; e < 1024; e += nt) {
            const uint16_t w = glm_spread(iq2_code(iq2s_grid[e]));
#pragma unroll
            for (int g = 0; g < 8; g++) *(uint16_t *) (t + 16 * g + 2 * e + 14 * (e & ~7)) = w;
        }
    }
    __device__ static void frag(const unsigned char * t, const unsigned char * slot, int sh, int ib, int tg,
                                uint32_t (*bq)[2]) {
        const unsigned char * cb = glm_copy(t);
        const float d0 = __half2float(__ushort_as_half(*(const uint16_t *) (slot + sh)));
        const uint32_t ql = ld_u32_even(slot, sh + 2 + 4 * ib);
        const uint32_t sg = ld_u32_even(slot, sh + 34 + 4 * ib) >> (2 * tg);
        const uint32_t qh = slot[sh + 66 + ib];
        const uint32_t sc = slot[sh + 74 + ib];
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const float dd = d0 * (0.5f + ((sc >> (4 * s)) & 0xf)) * 0.25f;
            const uint32_t h0 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(0)));
            const uint32_t h1 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(1)));
            const uint32_t h2 = __half_as_ushort(__float2half_rn(dd * (float) iq2_value(2)));
            const uint32_t lo = h0 | (h1 << 16);
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const int gr = 2 * s + q;
                const uint32_t idx = ((ql >> (8 * gr)) & 0xFF) | ((qh << (8 - 2 * gr)) & 0x300);
                const uint32_t w = *(const uint16_t *) (cb + 2 * idx + 14 * (idx & ~7u));
                const uint32_t sb = sg >> (8 * gr);
                bq[s][q] = prmt_sel(lo, h2, glm_sel(w, tg)) ^ (((sb << 15) & 0x8000u) | ((sb << 30) & 0x80000000u));
            }
        }
    }
};

template <class W, Kind K, int BN, int TPB, int XS>
struct GlmShape {
    static constexpr int TMB = frag::TMF * TPB;  // pairs per block
    static constexpr int RK = 256;               // values per row and weight stage
    static constexpr int D = XS - 1;             // X slices in flight ahead
    static constexpr int NW = BN / 16, NT = 32 * NW, MT = TMB / 16, SPW = RK / BK;
    static constexpr int WST = BN * W::RB, XST = TMB * BK * 2;
    static constexpr int XV = TMB * BK / 8 / NT;
    static constexpr int LDE = BN + 4;
    static constexpr int EPM = frag::TMF;  // pairs per epilogue pass
    static constexpr int SMEM_AB = XS * XST + 2 * WST + W::SMEM_TAB;
    static constexpr int SMEM_E = EPM * LDE * 4 + EPM * 8;
    static constexpr int SMEM = SMEM_AB > SMEM_E ? SMEM_AB : SMEM_E;
    static_assert(XV >= 1 && (TMB * BK / 8) % NT == 0, "x copies per thread");
    static_assert(D <= SPW, "the next weight stage, in slice m's group, is waited for at slice m + D");
    static_assert(MT % TPB == 0, "whole m-tiles per epilogue pass");
};

template <class W, Kind K, int BN, int TPB, int XS>
__global__ void __launch_bounds__(GlmShape<W, K, BN, TPB, XS>::NT, 1) gemm_glm_kernel(const legion_moe_args p) {
    using S = GlmShape<W, K, BN, TPB, XS>;
    constexpr int RB = W::RB, D = S::D, NT = S::NT, MT = S::MT, SPW = S::SPW, XST = S::XST, EPM = S::EPM;
    extern __shared__ __align__(16) unsigned char sm[];

    int kt;
    const Tile tl = group_of(p, blockIdx.y, TPB, kt);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int rt = blockIdx.x;
    const int64_t KD = K == GU ? p.n_embd : p.n_ff;
    const int NS = (int) (KD / BK), NR = (int) (KD / S::RK);

    unsigned char * xbuf = sm;
    unsigned char * wbuf = sm + XS * XST;
    unsigned char * tabs = wbuf + 2 * S::WST;

    // weight row r of the block (gate/up: even rows gate, odd rows up, H column rt * BN / 2 + r / 2)
    auto wrow = [&](int r) -> const uint8_t * {
        if (K == GU) {
            const int64_t row = (int64_t) rt * (BN / 2) + r / 2;
            return ((r & 1) ? p.up : p.gate) + tl.slot * p.gu_nb2 + row * p.gu_nb1;
        }
        return p.down + tl.slot * p.down_nb2 + ((int64_t) rt * BN + r) * p.down_nb1;
    };
    // stage R: the aligned RB bytes holding super-block R of each row
    auto issue_w = [&](int R) {
        if (R < NR) {
            unsigned char * dst = wbuf + (R & 1) * S::WST;
            const int64_t a = ((int64_t) R * W::SB) & ~(int64_t) 15;
            for (int f = threadIdx.x; f < BN * (RB / 16); f += NT) {
                const int r = f / (RB / 16), c = f % (RB / 16);
                cp_async16(dst + r * RB + 16 * c, wrow(r) + a + 16 * c);
            }
        }
    };
    // this thread's X rows: gathered by token (gate/up), in list order (down)
    const __half * xrow[S::XV];
#pragma unroll
    for (int i = 0; i < S::XV; i++) {
        const int r = (threadIdx.x + NT * i) / (BK / 8);
        if (r >= ne) {
            xrow[i] = nullptr;
        } else if (K == GU) {
            xrow[i] = p.xh + (int64_t) (p.list[r0 + r] / p.n_used) * p.n_embd;
        } else {
            xrow[i] = p.h + (int64_t) (r0 + r) * p.n_ff;
        }
    }
    auto issue_x = [&](int m) {
        if (m < NS) {
            unsigned char * dst = xbuf + (m % XS) * XST;
#pragma unroll
            for (int i = 0; i < S::XV; i++) {
                const int c = threadIdx.x + NT * i;
                if (xrow[i]) {
                    cp_async16(dst + (row_base(c / (BK / 8)) ^ ((c % (BK / 8)) << 4)),
                               xrow[i] + (int64_t) m * BK + 8 * (c % (BK / 8)));
                }
            }
        }
    };

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, tg = lane & 3;
    const int mvalid = (ne + 15) / 16;
    const int wr[2] = {16 * warp + g, 16 * warp + 8 + g};
    const int ar = (lane & 7) + 8 * ((lane >> 3) & 1), ac = lane >> 4;

    Acc4 tot[MT][2];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) tot[i][j][0] = tot[i][j][1] = tot[i][j][2] = tot[i][j][3] = 0.0f;

    W::tables(tabs, threadIdx.x, NT);  // visible after the first barrier
    issue_w(0);
    issue_x(0);
    cp_async_commit();
#pragma unroll
    for (int s = 1; s < D; s++) {
        issue_x(s);
        cp_async_commit();
    }
    for (int m = 0; m < NS; m++) {
        // X slice m (and its weight stage) landed; every warp is past slice m - 1, so its X
        // buffer and, at a stage start, the previous stage's are free
        cp_async_wait<D - 1>();
        __syncthreads();
        issue_x(m + D);
        if (m % SPW == 0) {
            issue_w(m / SPW + 1);
        }
        cp_async_commit();

        // B fragments: sub-blocks 2 c and 2 c + 1 of super-block m / SPW
        const int R = m / SPW, c = m % SPW;
        const unsigned char * ws = wbuf + (R & 1) * S::WST;
        const int sh = (R * W::SB) & 15;  // the super-block's offset in its slot
        uint32_t bq[2][4][2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
            for (int b = 0; b < 2; b++) {
                W::frag(tabs, ws + wr[j] * RB, sh, 2 * c + b, tg, bq[j] + 2 * b);
            }
        }

        const uint32_t xs = (uint32_t) __cvta_generic_to_shared(xbuf + (m % XS) * XST);
#pragma unroll
        for (int i = 0; i < MT; i++) {
            if (i < mvalid) {
                uint32_t a[4][4];
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    ldsm_x4(xs + (row_base(16 * i + ar) ^ ((2 * kk + ac) << 4)), a[kk][0], a[kk][1], a[kk][2],
                            a[kk][3]);
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    // the slice's products from zero, k in order, then into the F32 totals
                    Acc4 s = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int kk = 0; kk < 4; kk++) mma16816(s, a[kk], bq[j][kk][0], bq[j][kk][1]);
#pragma unroll
                    for (int v = 0; v < 4; v++) tot[i][j][v] += s[v];
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // the products through E one plan tile at a time
    float * E = (float *) sm;
    constexpr int MP = MT / TPB;
#pragma unroll
    for (int e = 0; e < TPB; e++) {
        if (e * EPM >= ne) {
            break;
        }
#pragma unroll
        for (int i = 0; i < MP; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int v = 0; v < 4; v++) {
                    E[(16 * i + g + 8 * (v >> 1)) * S::LDE + 16 * warp + 8 * j + 2 * tg + (v & 1)] =
                        tot[e * MP + i][j][v];
                }
        __syncthreads();
        store_tile<EPM, BN, S::LDE, NT, K, true>(p, E, r0 + e * EPM, min(EPM, ne - e * EPM), rt);
        __syncthreads();
    }
}

// One launch over n_rows weight rows (gate and up rows counted apart); false when the device
// refuses its shared memory. Shapes measured on the GB10 at 300 to 4096 prompt tokens (2048 and
// about 1000 matter most at --max-batch-tokens 2048): gate/up BN 128, one tile, three X slices;
// down BN 256, one tile, four X slices.
template <class W, Kind K, int BN, int TPB, int XS>
bool glm_launch(const legion_moe_args * a, int64_t n_rows, unsigned grid_y, void * stream) {
    using S = GlmShape<W, K, BN, TPB, XS>;
    static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES];
    if (n_rows % BN != 0 || !legion_device::smem_opt_in(smem, gemm_glm_kernel<W, K, BN, TPB, XS>, S::SMEM)) {
        return false;
    }
    const dim3 grid((unsigned) (n_rows / BN), grid_y);
    gemm_glm_kernel<W, K, BN, TPB, XS><<<grid, S::NT, S::SMEM, (cudaStream_t) stream>>>(*a);
    return true;
}

// The decode shape (#594): steps of at most 16 tokens (decode, MTP, the short pieces of a
// prompt). A token's experts are distinct, so an expert then has at most 16 pairs, one m16 A
// tile, and the GEMMs are weight streams: gemm_glm_kernel ran 256 (gate/up) and 128 (down)
// blocks per launch at one token, under three waves of the GB10's 48 SMs, each block with one
// weight stage of 10-24 KB in flight between barriers, and read the experts at 118 and 133 GB/s.
// gemm_glm_dec_kernel gives each warp 16 weight rows and its own cp.async ring of DP stages and
// no block barriers past the tables. A stage is SPAN consecutive super-blocks of each row: the
// LPDDR5X wants long runs, and the rows' 66- or 82-byte super-blocks one at a time (80- or
// 96-byte pieces, 16 rows apart) stream at about half the rate of 4-super-block pieces. The
// launch's warps walk the (tile, 16-row group) items round-robin, the ring running on across
// their items. The MMAs are transposed: the 16 weight rows are the m16 A operand (gemm_glm_kernel's
// B fragments from W::frag, the same registers) and the pairs the n8 B operand, straight from
// global memory (the step's few X or H rows stay in L1/L2), so a step of at most 8 pairs per
// expert takes one MMA per k16 instead of two. H or D is written from the registers.
// The same products, four chained MMAs per slice from zero and F32 totals as gemm_glm_kernel, and
// the epilogue computes store_tile's expressions on those totals: the same bits.
template <class W, int NW, int DP, int SPAN, bool REG = false>
struct GlmDec {
    static constexpr int NT = 32 * NW;
    // a row's slot: the 16-byte pieces covering SPAN super-blocks at any even misalignment, an
    // odd number of them so that rows g = 0..7 start in different bank quads
    static constexpr int RBC = ((SPAN * W::SB + 14 + 15) / 16) | 1;
    static constexpr int RBW = 16 * RBC;
    static constexpr int STAGE = 16 * RBW;        // one warp's stage: 16 rows' slots
    static constexpr int SPW = 256 / BK;          // slices per super-block
    static constexpr int SMEM = W::SMEM_TAB + NW * DP * STAGE;
    static constexpr int MAX_TOK = 16;            // tokens per step (one m16 tile per expert)
    // REG: the stage in flight in registers (a lane's pieces c = lane / 16 + 2 i), one slot
    static constexpr int NPL = (RBC + 1) / 2;
    static_assert(REG ? DP == 1 : DP >= 2, "a stage in flight while one is read");
};

__device__ __forceinline__ uint32_t ld_h2(const __half * row, int64_t k) {
    return row ? __ldg((const unsigned int *) (row + k)) : 0u;
}

template <class W, Kind K, int NW, int DP, int SPAN, bool REG>
__global__ void __launch_bounds__(GlmDec<W, NW, DP, SPAN, REG>::NT)
    gemm_glm_dec_kernel(const legion_moe_args p, int n_tiles) {
    using S = GlmDec<W, NW, DP, SPAN, REG>;
    constexpr int RBW = S::RBW;
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned char * tabs = sm;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, tg = lane & 3;
    unsigned char * ring = sm + W::SMEM_TAB + warp * DP * S::STAGE;

    const int NR = (int) ((K == GU ? p.n_embd : p.n_ff) / 256);  // super-blocks per row (SPAN | NR)
    const int groups = (int) ((K == GU ? 2 * p.n_ff : p.n_embd) / 16);
    // the plan's tiles: the ones past tile0[n_expert] are empty (group_of)
    const int64_t items = (int64_t) min(n_tiles, p.tile0[p.n_expert]) * groups;
    const int64_t step = (int64_t) gridDim.x * NW;
    const int64_t w0 = (int64_t) blockIdx.x * NW + warp;

    // global weight row G (gate/up: even rows gate, odd rows up, H column G / 2) of slot `slot`
    auto wrow = [&](int64_t slot, int64_t G) -> const uint8_t * {
        if (K == GU) {
            return ((G & 1) ? p.up : p.gate) + slot * p.gu_nb2 + (G / 2) * p.gu_nb1;
        }
        return p.down + slot * p.down_nb2 + G * p.down_nb1;
    };

    // the issue side: its item, first super-block and this lane's row (lane % 16; the two
    // half-warps take alternate pieces)
    int64_t i_item = w0;
    int i_R = 0, i_q = 0;
    const uint8_t * src = nullptr;
    auto sources = [&]() {
        if (i_item < items) {
            int kt;
            const Tile t = group_of(p, (int) (i_item / groups), 1, kt);
            src = wrow(t.slot, (int64_t) (i_item % groups) * 16 + lane % 16);
        }
    };
    // the next stage into ring slot i_q % DP (an empty group past the warp's last item): the
    // aligned pieces covering super-blocks i_R .. i_R + SPAN - 1, never past the row's end (rows
    // are whole pieces)
    uint4 rb[REG ? S::NPL : 1];
    int r_n = 0;
    auto issue = [&]() {
        if (i_item < items) {
            const int a = i_R * W::SB;
            const int a0 = a & ~15, n = ((a & 15) + SPAN * W::SB + 15) / 16;
            if constexpr (REG) {
#pragma unroll
                for (int i = 0; i < S::NPL; i++) {
                    const int c = lane / 16 + 2 * i;
                    if (c < n) {
                        rb[i] = __ldcs((const uint4 *) (src + a0 + 16 * c));
                    }
                }
                r_n = n;
            } else {
                unsigned char * dst = ring + (i_q % DP) * S::STAGE + (lane % 16) * RBW;
                for (int c = lane / 16; c < n; c += 2) {
                    cp_async16(dst + 16 * c, src + a0 + 16 * c);
                }
            }
            if ((i_R += SPAN) == NR) {
                i_R = 0;
                i_item += step;
                sources();
            }
        } else {
            r_n = 0;
        }
        if constexpr (!REG) {
            i_q++;
            cp_async_commit();
        }
    };
    // REG: the stage in the registers into the warp's slot
    auto land = [&]() {
        unsigned char * dst = ring + (lane % 16) * RBW;
#pragma unroll
        for (int i = 0; i < S::NPL; i++) {
            const int c = lane / 16 + 2 * i;
            if (c < r_n) {
                *(uint4 *) (dst + 16 * c) = rb[i];
            }
        }
    };

    sources();
#pragma unroll
    for (int s = 0; s < (REG ? 1 : DP - 1); s++) {
        issue();
    }
    W::tables(tabs, threadIdx.x, S::NT);
    __syncthreads();

    int q = 0;
    for (int64_t it = w0; it < items; it += step) {
        int kt;
        const Tile t = group_of(p, (int) (it / groups), 1, kt);
        const int grp = (int) (it % groups);
        const int ne = t.ne;  // at most 16: the launcher takes steps of at most MAX_TOK tokens
        const bool two = ne > 8;  // a second n8 tile of pairs (warp-uniform)
        // B column g of pair tile nb: X gathered by token (gate/up), H in list order (down)
        auto brow = [&](int r) -> const __half * {
            if (r >= ne) {
                return nullptr;
            }
            return K == GU ? p.xh + (int64_t) (p.list[t.r0 + r] / p.n_used) * p.n_embd
                           : p.h + (int64_t) (t.r0 + r) * p.n_ff;
        };
        const __half * xr[2] = {brow(g), brow(8 + g)};

        Acc4 tot[2];
#pragma unroll
        for (int nb = 0; nb < 2; nb++) tot[nb][0] = tot[nb][1] = tot[nb][2] = tot[nb][3] = 0.0f;
        for (int R0 = 0; R0 < NR; R0 += SPAN, q++) {
            // stage q landed, and every lane is past stage q - 1, whose slot the next copies take
            if constexpr (REG) {
                __syncwarp();
                land();
                __syncwarp();
            } else {
                cp_async_wait<DP - 2>();
                __syncwarp();
            }
            issue();
            const unsigned char * ws = ring + (REG ? 0 : (q % DP) * S::STAGE);
#pragma unroll 1
            for (int R = R0; R < R0 + SPAN; R++) {
                // super-block R at byte sh of the rows' slots
                const int sh = ((R0 * W::SB) & 15) + (R - R0) * W::SB;
#pragma unroll
                for (int c = 0; c < S::SPW; c++) {
                    uint32_t bq[2][4][2];
#pragma unroll
                    for (int j = 0; j < 2; j++) {
#pragma unroll
                        for (int b = 0; b < 2; b++) {
                            W::frag(tabs, ws + (8 * j + g) * RBW, sh, 2 * c + b, tg, bq[j] + 2 * b);
                        }
                    }
                    // the weights as A (m16: rows g, 8 + g; gemm_glm_kernel's B fragments of its
                    // n8 tiles j = 0, 1), the pairs as B (n8: pair g of the tile)
                    const int64_t k0 = (int64_t) (S::SPW * R + c) * BK + 2 * tg;
#pragma unroll
                    for (int nb = 0; nb < 2; nb++) {
                        if (nb == 1 && !two) {
                            break;
                        }
                        // the slice's products from zero, k in order, then into the F32 totals
                        Acc4 s = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                        for (int kk = 0; kk < 4; kk++) {
                            const uint32_t a[4] = {bq[0][kk][0], bq[1][kk][0], bq[0][kk][1], bq[1][kk][1]};
                            mma16816(s, a, ld_h2(xr[nb], k0 + 16 * kk), ld_h2(xr[nb], k0 + 16 * kk + 8));
                        }
#pragma unroll
                        for (int v = 0; v < 4; v++) tot[nb][v] += s[v];
                    }
                }
            }
        }

        // lane (g, tg) holds weight rows 16 grp + g (v < 2) and + 8 (v >= 2) of pairs 8 nb + 2 tg
        // + (v & 1)
#pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            if (nb == 1 && !two) {
                break;
            }
            if (K == GU) {
                // store_tile's SwiGLU: gate rows are even (g even), their up rows the next (g + 1,
                // lane + 4)
                float up[4];
#pragma unroll
                for (int v = 0; v < 4; v++) up[v] = __shfl_xor_sync(0xffffffffu, tot[nb][v], 4);
                if (g % 2 == 0) {
#pragma unroll
                    for (int v = 0; v < 4; v++) {
                        const int r = 8 * nb + 2 * tg + (v & 1);
                        if (r < ne) {
                            float gv = tot[nb][v], u = up[v];
                            if (p.limit > 0.0f) {
                                gv = fminf(gv, p.limit);
                                u = fminf(fmaxf(u, -p.limit), p.limit);
                            }
                            const float h = gv / (1.0f + __expf(-gv)) * u;
                            p.h[(int64_t) (t.r0 + r) * p.n_ff + (int64_t) grp * 8 + g / 2 + 4 * (v >> 1)] =
                                __float2half(h);
                        }
                    }
                }
            } else {
                // store_tile's D: scaled by the pair's router weight
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int r = 8 * nb + 2 * tg + e;
                    if (r < ne) {
                        const int pair = p.list[t.r0 + r];
                        const int64_t tok = pair / p.n_used;
                        const int jj = pair % p.n_used;
                        const float wgt = p.w[tok * p.w_ld + jj];
#pragma unroll
                        for (int hv = 0; hv < 2; hv++) {
                            store_d(p, (int64_t) pair * p.n_embd + (int64_t) grp * 16 + g + 8 * hv,
                                    tot[nb][2 * hv + e] * wgt);
                        }
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
}

// One launch over the plan's first n_tiles tiles, as many blocks as fit the device at once (or
// fewer when there are fewer items); false when the device refuses the shared memory or the
// step has more than MAX_TOK tokens (or the rows are not a whole number of stages).
template <class W, Kind K, int NW, int DP, int SPAN, bool REG = false>
bool glm_dec_launch(const legion_moe_args * a, unsigned n_tiles, void * stream) {
    using S = GlmDec<W, NW, DP, SPAN, REG>;
    static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES];
    static std::atomic<int> occ[GGML_CUDA_MAX_DEVICES];
    if (a->n_tok > S::MAX_TOK || (K == GU ? 2 * a->n_ff : a->n_embd) % 16 != 0 ||
        (K == GU ? a->n_embd : a->n_ff) / 256 % SPAN != 0 ||
        !legion_device::smem_opt_in(smem, gemm_glm_dec_kernel<W, K, NW, DP, SPAN, REG>, S::SMEM)) {
        return false;
    }
    const int dev = legion_device::current();
    if (dev < 0) {
        return false;
    }
    int per_sm = occ[dev].load(std::memory_order_relaxed);
    if (per_sm == 0) {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm_glm_dec_kernel<W, K, NW, DP, SPAN, REG>, S::NT,
                                                          S::SMEM) != cudaSuccess ||
            per_sm < 1) {
            (void) cudaGetLastError();
            per_sm = 1;
        }
        occ[dev].store(per_sm, std::memory_order_relaxed);
    }
    const int64_t groups = (K == GU ? 2 * a->n_ff : a->n_embd) / 16;
    const int64_t want = ((int64_t) n_tiles * groups + NW - 1) / NW;
    const int64_t fit = (int64_t) std::max(legion_device::sm_count(), 1) * per_sm;
    const unsigned blocks = (unsigned) std::max<int64_t>(1, std::min(want, fit));
    gemm_glm_dec_kernel<W, K, NW, DP, SPAN, REG><<<blocks, S::NT, S::SMEM, (cudaStream_t) stream>>>(*a, (int) n_tiles);
    return true;
}
