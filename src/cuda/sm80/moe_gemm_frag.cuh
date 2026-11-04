// moe_gemm.cu, sm_80 and newer: gemm_frag_kernel (Q4_K and Q5_K gate/up) and gemm_frag_down_kernel
// (Q5_1 and Q8_0 down), GEMMs whose warps dequantize their B fragments straight from raw weight
// blocks in shared memory into registers.
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace, after
// sm80/moe_gemm.cuh and sm80/moe_gemm_raw.cuh (it uses KScale, Tile, tile_of, store_tile,
// row_base, ldsm_x4, mma16816 and the cp.async helpers). The PTX keeps its __CUDA_ARCH__ guards
// and __trap() stubs, so the file compiles for every target architecture; launch_gemm in
// moe_gemm.cu picks the kernel per device at run time (frag_ok).
//
// gemm_raw_kernel converts each 64-value slice of its weight rows into an F16 tile in shared
// memory that the warps read back with ldmatrix: a store, a barrier and a load between the raw
// bytes and the tensor cores. Here each warp owns 16 weight rows (two n-tiles of 8) for all TM
// pairs of the tile, so no two warps need the same B fragment: each lane converts its own
// fragment values from the raw blocks, with the same F32 formula and one F16 rounding as
// Piece<T_Q4_K> (KScale::put), and the products, their order and so the output bits are those
// of gemm_raw_kernel and gemm_kernel.
//
// Gate/up (#363): a block takes two consecutive plan tiles of one expert (up to 128 pairs) and
// BN_F = 256 weight rows (16 warps of 16 rows, each warp all 128 pairs: 8 m-tiles), so each B
// fragment a lane converts feeds twice the MMAs of a one-tile block; the conversions were half
// the kernel's issue slots. That fits in shared memory with half super-block weight stages (its
// 16-byte header and the 64 qs bytes of two slices: 80 bytes per row, as gemm_raw_kernel's).
// The GB10 streams such short pieces slowly from DRAM, so the rows come into L2 ahead in long
// runs: each row's next PF super-blocks (288 bytes) in one cp.async.bulk.prefetch.L2, every PF
// super-blocks. The X slices stay three in the ring.
//
// Pipeline: a cp.async group per 64-value K slice holding X slice m + D and, at the first slice
// of each stage, the next weight stage (two stages in the ring); one barrier per slice.

namespace frag {

constexpr int TMF = 64;               // pairs per plan tile: the launchers' TM (checked in frag_ok)
constexpr int TPB = 2;                // plan tiles per block
constexpr int TMB = TMF * TPB;        // pairs per block
constexpr int BN_F = 256;             // weight rows per block (gate and up interleaved)
constexpr int RK = 128;               // values per row and weight stage: half a K-quant super-block
constexpr int PF = 2;                 // super-blocks per row and L2 prefetch
constexpr int XS = 3;                 // X slices in the ring
constexpr int D = XS - 1;             // X slices in flight ahead
constexpr int NW = BN_F / 16;         // warps: 16 weight rows each
constexpr int NT = 32 * NW;           // 512 threads
constexpr int MT = TMB / 16;          // m-tiles of 16 pairs per warp
constexpr int SPW = RK / BK;          // slices per weight stage
constexpr int XST = TMB * BK * 2;     // bytes of an X slice
constexpr int XV = TMB * BK / 8 / NT;  // 16-byte X copies per thread and slice
constexpr int LDE = BN_F + 4;         // epilogue row stride (floats)
constexpr int EPM = TMF;              // pairs per epilogue pass (one plan tile)
constexpr int SMEM_E = EPM * LDE * 4;
static_assert(XV >= 1 && (TMB * BK / 8) % NT == 0, "x copies per thread");
static_assert(D <= SPW, "the next weight stage, in slice m's group, is waited for at slice m + D");
static_assert(SPW == 2 && MT % TPB == 0, "two stages per super-block; whole m-tiles per epilogue pass");

// The gate/up weight formats (#836 part 10: Q5_K beside Q4_K). A row's stage holds its
// super-block's 16-byte header and the 64 qs bytes of the half it covers (QS: where the qs
// bytes start in the super-block of SB bytes). Q5_K's 32 qh bytes (QH) serve both halves, so
// only the first half's stage (the even one, always ring buffer 0) carries them, after the qs
// bytes: RB0 bytes per row in buffer 0, RB1 in buffer 1. A lane reads its qh bits into
// registers at the first slice of a super-block. (One size for both buffers would take 106496
// bytes of shared memory, more than the GB10 grants a block.)
template <int WT> struct GuFmt;
template <> struct GuFmt<T_Q4_K> {
    static constexpr int SB = 144, QS = 16, QH = 0, RB0 = 80, RB1 = 80;
};
template <> struct GuFmt<T_Q5_K> {
    static constexpr int SB = 176, QS = 48, QH = 16, RB0 = 112, RB1 = 80;
};
template <int WT> struct GUShape {
    using F = GuFmt<WT>;
    static constexpr int WST0 = BN_F * F::RB0;  // bytes of ring buffer 0 (even stages)
    static constexpr int WST1 = BN_F * F::RB1;  // ring buffer 1 (odd stages)
    static constexpr int SMEM_AB = XS * XST + WST0 + WST1;
    // Q4_K: 90112 bytes, Q5_K: 98304: an opt-in
    static constexpr int SMEM = SMEM_AB > SMEM_E ? SMEM_AB : SMEM_E;
    static_assert(F::RB0 % 16 == 0 && F::RB1 % 16 == 0, "whole 16-byte copies");
};

}  // namespace frag

// cp.async.bulk.prefetch.L2 (sm_90+; a no-op on older targets, which never run the fragment
// kernels): src and bytes multiples of 16
__device__ __forceinline__ void bulk_prefetch_l2(const void * src, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src), "r"(bytes));
#endif
}

// Plan tiles in groups of tpb consecutive tiles of one expert: the group's first tile gives its
// block all of the group's pairs (ne up to TMF * tpb); the other tiles get none (ne = 0). kt: the
// tile's index in its expert's tiles.
__device__ __forceinline__ Tile group_of(const legion_moe_args & p, int tile, int tpb, int & kt) {
    if (tile >= p.tile0[p.n_expert]) {
        return {0, 0, 0};
    }
    int lo = 0, hi = p.n_expert;  // last e with tile0[e] <= tile
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (p.tile0[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int e = lo;
    kt = tile - p.tile0[e];
    if (kt % tpb != 0) {
        return {0, 0, 0};
    }
    const int r0 = p.first[e] + kt * frag::TMF;
    return {r0, min(frag::TMF * tpb, p.first[e + 1] - r0), p.table ? p.table[e] : e};
}

// One block row's plan tile y of the gate/up GEMM (gemm_frag_kernel walks its rows' tiles).
template <int WT>
__device__ __forceinline__ void gemm_frag_tile(const legion_moe_args & p, const int y) {
    using namespace frag;
    using F = GuFmt<WT>;
    using S = GUShape<WT>;
    extern __shared__ __align__(16) unsigned char sm[];

    int kt;
    const Tile tl = group_of(p, y, TPB, kt);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int rt = blockIdx.x;
    const int NS = (int) (p.n_embd / BK), NR = (int) (p.n_embd / RK), NSB = (int) (p.n_embd / 256);

    unsigned char * xbuf = sm;             // XS X slices
    unsigned char * wbuf = sm + XS * XST;  // 2 raw weight stages: row r at r * RB0 (r * RB1)

    // weight row r of the block: even rows gate, odd rows up (H column rt * BN_F / 2 + r / 2)
    auto wrow = [&](int r) {
        const int64_t row = (int64_t) rt * (BN_F / 2) + r / 2;
        return ((r & 1) ? p.up : p.gate) + tl.slot * p.gu_nb2 + row * p.gu_nb1;
    };
    // super-blocks S .. S + PF - 1 of row threadIdx.x into L2
    auto prefetch = [&](int s0) {
        if (threadIdx.x < BN_F && s0 < NSB) {
            bulk_prefetch_l2(wrow(threadIdx.x) + (int64_t) s0 * F::SB, (uint32_t) (min(PF, NSB - s0) * F::SB));
        }
    };
    // stage R: half R % 2 of super-block R / 2 of each row (its header, then the 64 qs bytes of
    // slices 2 (R % 2) and 2 (R % 2) + 1, then in an even stage the qh bytes), consecutive
    // threads along a row's 16-byte pieces
    auto issue_rows = [&](int R, unsigned char * dst, auto rb) {
        constexpr int RB = decltype(rb)::value, CH = RB / 16;
        const int64_t sb = (int64_t) (R / 2) * F::SB;
        for (int f = threadIdx.x; f < BN_F * CH; f += NT) {
            const int r = f / CH, c = f % CH;
            const int64_t at = c == 0 ? sb
                               : c < 5 ? sb + F::QS + 64 * (R % 2) + 16 * (c - 1)
                                       : sb + F::QH + 16 * (c - 5);
            cp_async16(dst + r * RB + 16 * c, wrow(r) + at);
        }
    };
    auto issue_w = [&](int R) {
        if (R < NR) {
            if constexpr (F::RB0 == F::RB1) {
                issue_rows(R, wbuf + (R & 1) * S::WST0, std::integral_constant<int, F::RB0>{});
            } else if (R & 1) {
                issue_rows(R, wbuf + S::WST0, std::integral_constant<int, F::RB1>{});
            } else {
                issue_rows(R, wbuf, std::integral_constant<int, F::RB0>{});
            }
        }
    };
    // this thread's X rows (gathered by token)
    const __half * xrow[XV];
#pragma unroll
    for (int i = 0; i < XV; i++) {
        const int r = (threadIdx.x + NT * i) / (BK / 8);
        xrow[i] = r < ne ? p.xh + (int64_t) (p.list[r0 + r] / p.n_used) * p.n_embd : nullptr;
    }
    auto issue_x = [&](int m) {
        if (m < NS) {
            unsigned char * dst = xbuf + (m % XS) * XST;
#pragma unroll
            for (int i = 0; i < XV; i++) {
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
    const int mvalid = (ne + 15) / 16;  // m-tiles holding pairs
    // this lane's weight rows: column g of the warp's n-tiles 0 and 1
    const int wr[2] = {16 * warp + g, 16 * warp + 8 + g};
    // ldmatrix lane addressing of the A fragments (as step_mma16)
    const int ar = (lane & 7) + 8 * ((lane >> 3) & 1), ac = lane >> 4;
    // A B fragment (k16 x n8) gives lane (g, tg) values k = 2 tg, 2 tg + 1 (b0) and 2 tg + 8,
    // 2 tg + 9 (b1). In a 64-value slice, value k sits in qs byte k % 32, the low nibble for
    // k < 32: k16 steps 0 and 2 read bytes 0..15, steps 1 and 3 bytes 16..31. bsel picks bytes
    // 2 tg, 2 tg + 1 of one word and of the word 8 bytes further.
    const uint32_t bsel = (tg & 1) ? 0x7632u : 0x5410u;
    // Q5_K: this lane's qh bytes of its two rows' current super-block (bytes 0..15, 16..31)
    uint32_t qh[2][2] = {{0, 0}, {0, 0}};
    (void) qh;

    Acc4 tot[MT][2];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) tot[i][j][0] = tot[i][j][1] = tot[i][j][2] = tot[i][j][3] = 0.0f;

    // prologue: the rows' first 2 PF super-blocks into L2, weight stage 0 with X slice 0, then X
    // slices 1 .. D - 1, a group each
    prefetch(0);
    prefetch(PF);
    issue_w(0);
    issue_x(0);
    cp_async_commit();
#pragma unroll
    for (int s = 1; s < D; s++) {
        issue_x(s);
        cp_async_commit();
    }
    for (int m = 0; m < NS; m++) {
        // X slice m (and its weight stage, issued with an earlier slice) landed; every warp is
        // past slice m - 1, so its X buffer and, at a stage start, the previous stage's are free
        cp_async_wait<D - 1>();
        __syncthreads();
        issue_x(m + D);
        if (m % SPW == 0) {
            // stage R starts super-block R / 2: at every PF-th, the PF after the next PF
            const int R = m / SPW + 1;
            if (R % 2 == 0 && (R / 2) % PF == 0) {
                prefetch(R / 2 + PF);
            }
            issue_w(R);
        }
        cp_async_commit();

        // B fragments: this lane's 16 values of each of its two rows, k16 steps kk = 0..3
        const int odd = (m / SPW) & 1;
        const unsigned char * ws = wbuf + odd * S::WST0;
        const int rb = odd ? F::RB1 : F::RB0;
        const int c = m % SPW;                // the slice's 32 qs bytes in the stage
        const int cs = 2 * odd + c;           // its 64 values in the super-block: scales 2 cs, 2 cs + 1
        if constexpr (F::QH != 0) {
            // a super-block's first slice: this lane's qh bytes (the positions of its qs bytes)
            // from the even stage, for all four of its slices
            if (m % (2 * SPW) == 0) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const uint32_t * q = (const uint32_t *) (ws + wr[j] * F::RB0 + 16 + 64);
                    qh[j][0] = __byte_perm(q[tg >> 1], q[(tg >> 1) + 2], bsel);
                    qh[j][1] = __byte_perm(q[(tg >> 1) + 4], q[(tg >> 1) + 6], bsel);
                }
            }
        }
        uint32_t bq[2][4][2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const unsigned char * blk = ws + wr[j] * rb;
            const uint4 h = *(const uint4 *) blk;
            KScale lo, hi;
            lo.set(h, 2 * cs);
            hi.set(h, 2 * cs + 1);
            const uint32_t * q = (const uint32_t *) (blk + 16 + 32 * c);
            const uint32_t x0 = __byte_perm(q[tg >> 1], q[(tg >> 1) + 2], bsel);        // bytes 0..15
            const uint32_t x1 = __byte_perm(q[(tg >> 1) + 4], q[(tg >> 1) + 6], bsel);  // 16..31
            if constexpr (F::QH != 0) {
                // Q5_K: bit 2 cs (low nibbles) or 2 cs + 1 (high) of qh[l] is bit 4 of value l,
                // as Piece<T_Q5_K>
                const uint32_t h0 = qh[j][0] >> (2 * cs), h1 = qh[j][1] >> (2 * cs);
                lo.put((x0 & 0x0F0F0F0Fu) | ((h0 & 0x01010101u) << 4), bq[j][0]);
                lo.put((x1 & 0x0F0F0F0Fu) | ((h1 & 0x01010101u) << 4), bq[j][1]);
                hi.put(((x0 >> 4) & 0x0F0F0F0Fu) | (((h0 >> 1) & 0x01010101u) << 4), bq[j][2]);
                hi.put(((x1 >> 4) & 0x0F0F0F0Fu) | (((h1 >> 1) & 0x01010101u) << 4), bq[j][3]);
            } else {
                lo.put(x0 & 0x0F0F0F0Fu, bq[j][0]);
                lo.put(x1 & 0x0F0F0F0Fu, bq[j][1]);
                hi.put((x0 >> 4) & 0x0F0F0F0Fu, bq[j][2]);
                hi.put((x1 >> 4) & 0x0F0F0F0Fu, bq[j][3]);
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

    // the products through E one plan tile (EPM pairs, MT / TPB m-tiles) at a time
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
                    E[(16 * i + g + 8 * (v >> 1)) * LDE + 16 * warp + 8 * j + 2 * tg + (v & 1)] = tot[e * MP + i][j][v];
                }
        __syncthreads();
        store_tile<EPM, BN_F, LDE, NT, GU>(p, E, r0 + e * EPM, min(EPM, ne - e * EPM), rt);
        __syncthreads();
    }
}

// WALK (frag_ycap capped grid.y): block row blockIdx.y runs plan tiles blockIdx.y, + gridDim.y,
// ...; the barrier keeps a tile's shared memory from being refilled while the last one still
// reads it. Otherwise one block row per plan tile, as before.
template <int WT, bool WALK>
__global__ void __launch_bounds__(frag::NT, 1) gemm_frag_kernel(const legion_moe_args p) {
    if constexpr (WALK) {
        const int nt = p.tile0[p.n_expert];
        for (int y = blockIdx.y; y < nt; y += gridDim.y) {
            gemm_frag_tile<WT>(p, y);
            __syncthreads();
        }
    } else {
        gemm_frag_tile<WT>(p, blockIdx.y);
    }
}

// ---- down, Q5_1 ---------------------------------------------------------------------------------
//
// gemm_frag_down_kernel: the same structure for the down GEMM on Q5_1 (24-byte blocks of 32
// values: d, m halves, qh, qs[16]). BN weight rows per block (16 per warp), RK values per row and
// weight stage (RK / 32 whole blocks, RB bytes, consecutive lanes along the row), XS X slices.
// Values of a k16 step: block 2 c + kk / 2 of the slice's stage, the low (kk even) or high nibbles
// of qs bytes 2 tg, 2 tg + 1, 2 tg + 8, 2 tg + 9 with qh bits (kk & 1) * 16 + those as bit 4;
// each half2 is Piece<T_Q5_1>'s (q + 1024 - 1024) * d + m, so the bits are gemm_raw_kernel's.
// Unlike gate/up, down does not gain from longer runs per row (K is only 640 values: 320-value
// stages were slower on the GB10); it gains from skipping the F16 tile and one barrier per slice:
// 128-value stages (96 bytes per row), 128-row blocks, three X slices (#363). Its rows (480
// bytes at n_ff = 640) do come into L2 whole, one bulk prefetch each, at the start of the
// expert's first tile.
// Two plan tiles per block did not gain here (K is short: the conversions weigh less; #825: 7-9%
// slower again on the GB10).
//
// Epilogue (#825): at 4K-token steps the kernel moves more bytes than it computes (the used
// experts' rows once, ~0.63 GB at Qwen3.8-Flash-Next's shape, and D, 0.42 GB of F32 written: the
// GB10 writes at ~197 GB/s and reads at ~240), so it runs at ~82% of a copy kernel moving the
// same bytes. The lanes store their products straight from the accumulators (each pair's router
// weight and place looked up into shared memory before the main loop), a quad's 8 consecutive
// values of a row in full 32-byte sectors (16 bytes for 16-bit D), as evict-first stores: D is
// read only by the reduce, long after, and the weight rows and H tiles the next blocks read stay
// in L2. No pass through shared memory and no barrier at the end; the same products and
// roundings (E * w, then store_d's conversion), so the same bits. 2-3% faster on the GB10 for
// F32 D, 5-8% for --moe-pair-out bf16.

//
// Q8_0 down (#836 part 10): 34-byte blocks of 32 values (d, qs[32]), so rows (680 bytes at
// n_ff = 640) are only 8-byte aligned: the stage (RK / 32 blocks, RB = 136 bytes) comes in 8-byte
// copies (cp.async.ca) into rows RS = 144 bytes apart (8 rows' fragment loads on distinct banks).
// Values of a k16 step kk: block 2 c + kk / 2, its bytes 16 (kk & 1) + 2 tg, + 1 and + 8, + 9 as
// Piece<T_Q8_0>'s (q ^ 0x80 + 1024 - 1152) * d, so the bits are gemm_kernel's. The rows' L2
// prefetch takes the 16-byte aligned part of each row (the copies fetch the ends).
// At RK = 128 its ring (61952 bytes) leaves one block per SM on the GB10 (Q5_1's 49664 already
// fit two). Q8_0 down at RK = 64 (#826): a stage is two blocks, 68 bytes, which starts 4 bytes
// into an 8-byte word every other stage (SPLIT), so each row's stage comes as nine 8-byte copies
// from the word holding its first byte (CB = 72 bytes, rows RS = 80 apart, the stage at byte
// (R * RB) % 8 of its row; n_ff % 128 == 0 makes the last stage odd, so no copy reads past a
// row). With one slice per stage the next stage cannot ride in slice m's group (D <= SPW): it
// goes in a group of its own, committed before X slice m + D's, so the wait for slice m + 1
// (all but the newest D - 1 groups) covers it. 45568 bytes: two blocks per
// SM. The same values in the same fragments, so the same bits.

template <int WT, int BN, int RK, int XS>
struct FragDown {
    static constexpr int TMF = frag::TMF;
    static constexpr int BB = WT == T_Q8_0 ? 34 : 24;  // bytes per 32-value block
    static constexpr int RB = RK / 32 * BB;            // bytes per row and stage
    static constexpr int CP = WT == T_Q8_0 ? 8 : 16;   // bytes per copy
    static constexpr bool SPLIT = RB % CP != 0;        // odd stages start mid-word (Q8_0, RK = 64)
    static constexpr int CB = SPLIT ? RB + CP / 2 : RB;  // bytes copied per row and stage
    static constexpr int RS = (CB + 15) / 16 * 16;     // shared memory row stride
    static constexpr int D = XS - 1;
    static constexpr int NW = BN / 16, NT = 32 * NW, MT = TMF / 16, SPW = RK / BK;
    static constexpr int WST = BN * RS, XST = TMF * BK * 2;
    static constexpr int XV = (TMF * BK / 8 + NT - 1) / NT;
    static constexpr int SMEM_AB = XS * XST + 2 * WST;
    static constexpr int SMEM = SMEM_AB + TMF * 8;  // the operand ring, then each pair and weight
    static_assert(CB % CP == 0 && (!SPLIT || 2 * RB % CP == 0) && RS % 16 == 0 && RK % BK == 0,
                  "whole copies, aligned rows, whole slices");
    static_assert(SPLIT ? D - 1 <= SPW : D <= SPW, "the next weight stage is in by slice m + SPW");
    // the down sum (#826) reuses the operand ring: a tile's rows kept back, then each row's flag
    // and the list of rows to sum and its length
    static constexpr int DS_FLAG = TMF * BN * 4;
    static_assert(SMEM_AB >= DS_FLAG + (2 * TMF + 1) * 4, "the down sum's rows and lists fit in the ring");
};

// cp.async of 8 bytes (cache at all levels: the 16-byte form is L2-only and needs 16-byte
// alignment)
__device__ __forceinline__ void cp_async8(void * smem, const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(gmem));
#else
    __trap();
#endif
}

// The down sum (#826, p.sum_cnt): the kernel makes out itself and reduce_kernel does not run.
// A token's pairs are in tiles of different experts, so different blocks at different times;
// the sum must keep reduce_kernel's order (slot order, the first valid pair starts it, then the
// shared expert), so it can only start once all of them are in. Per (token, 128-column block)
// a counter holds the pairs still to come (legion_moe_down_sum_init). After the main loop a
// block reads its rows' counters: a row whose counter is 1 is the token's last pair (all others
// stored and counted), so its block keeps the row in shared memory instead of storing it. The
// other rows are stored, then (fence, barrier) counted down; a row that takes its counter from
// 1 to 0 is the last as well. The block then sums each last row's token over its 128 columns,
// the other pairs read back through L2. The same values, order and roundings as reduce4: the
// same bits. Saves the last pair's store and load (a tenth of D's traffic at 10 experts per
// token) and the reduce kernel's own pass.

// Token t's sum over the four columns at col of out, as reduce4 makes it. Pair `own`'s values
// from ownv when not null (the row this block kept), the others from D.
__device__ __forceinline__ void down_sum4(const legion_moe_args & p, int64_t t, int64_t col, int own,
                                          const float * ownv) {
    uint32_t valid = 0;
    for (int j = 0; j < p.n_used; j++) {
        if (expert_ok(p, p.ids[t * p.ids_ld + j])) {
            valid |= 1u << j;
        }
    }
    const int64_t n4 = p.n_embd / 4;
    const float4 * d = (const float4 *) (p.d + t * p.n_used * p.n_embd + col);
    float4 v[DSUM_MAX_USED];
#pragma unroll
    for (int j = 0; j < DSUM_MAX_USED; j++) {
        if ((valid >> j) & 1u) {
            // L2, not L1: other SMs stored these rows during this kernel
            v[j] = ownv && j == own ? *(const float4 *) ownv : __ldcg(d + (int64_t) j * n4);
        }
    }
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool any = false;
#pragma unroll
    for (int j = 0; j < DSUM_MAX_USED; j++) {
        if ((valid >> j) & 1u) {
            const float4 f = v[j];
            if (any) {
                acc.x = acc.x + f.x;
                acc.y = acc.y + f.y;
                acc.z = acc.z + f.z;
                acc.w = acc.w + f.w;
            } else {
                acc = f;
            }
            any = true;
        }
    }
    if (p.sh) {
        const float sgate = shexp_gate(p, t);
        const float4 s = *(const float4 *) (p.sh + t * p.sh_ld + col);
        acc.x = __fadd_rn(acc.x, __fmul_rn(s.x, sgate));
        acc.y = __fadd_rn(acc.y, __fmul_rn(s.y, sgate));
        acc.z = __fadd_rn(acc.z, __fmul_rn(s.z, sgate));
        acc.w = __fadd_rn(acc.w, __fmul_rn(s.w, sgate));
    }
    *(float4 *) (p.out + t * p.n_embd + col) = acc;
}

// One block row's plan tile y of the down GEMM (gemm_frag_down_kernel walks its rows' tiles).
template <int WT, int BN, int RK, int XS>
__device__ __forceinline__ void gemm_frag_down_tile(const legion_moe_args & p, const int y) {
    using F = FragDown<WT, BN, RK, XS>;
    constexpr int TMF = F::TMF, RB = F::RB, D = F::D, NT = F::NT, MT = F::MT, SPW = F::SPW;
    extern __shared__ __align__(16) unsigned char sm[];

    int kt;
    const Tile tl = group_of(p, y, 1, kt);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int rt = blockIdx.x;
    const int NS = (int) (p.n_ff / BK), NR = (int) (p.n_ff / RK);

    unsigned char * xbuf = sm;
    unsigned char * wbuf = sm + XS * F::XST;
    // each row's pair and router weight (store_tile's ROWS lookup), published by the main
    // loop's first barrier
    int * rpair = (int *) (sm + F::SMEM_AB);
    float * rw = (float *) (rpair + TMF);
    for (int r = threadIdx.x; r < ne; r += NT) {
        const int pair = p.list[r0 + r];
        rpair[r] = pair;
        rw[r] = p.w[(int64_t) (pair / p.n_used) * p.w_ld + pair % p.n_used];
    }
    const uint8_t * wbase = p.down + tl.slot * p.down_nb2 + (int64_t) rt * BN * p.down_nb1;
    // the expert's first tile brings each row whole (n_ff values) into L2 in one request; its
    // other tiles, running alongside, find them there
    if (kt == 0 && threadIdx.x < BN) {
        if constexpr (F::CP == 16) {
            bulk_prefetch_l2(wbase + threadIdx.x * p.down_nb1, (uint32_t) (NR * RB));
        } else {
            const uintptr_t a0 = (uintptr_t) (wbase + threadIdx.x * p.down_nb1);
            const uintptr_t lo = (a0 + 15) & ~(uintptr_t) 15, hi = (a0 + (uintptr_t) NR * RB) & ~(uintptr_t) 15;
            if (hi > lo) {
                bulk_prefetch_l2((const void *) lo, (uint32_t) (hi - lo));
            }
        }
    }

    auto issue_w = [&](int R) {
        if (R < NR) {
            constexpr int CH = F::CB / F::CP;
            unsigned char * dst = wbuf + (R & 1) * F::WST;
            // SPLIT: from the word holding the stage's first byte
            const int64_t at = F::SPLIT ? (int64_t) R * RB / F::CP * F::CP : (int64_t) R * RB;
            for (int f = threadIdx.x; f < BN * CH; f += NT) {
                const int r = f / CH, c = f % CH;
                const uint8_t * src = wbase + r * p.down_nb1 + at + F::CP * c;
                if constexpr (F::CP == 16) {
                    cp_async16(dst + r * F::RS + 16 * c, src);
                } else {
                    cp_async8(dst + r * F::RS + 8 * c, src);
                }
            }
        }
    };
    const __half * xrow[F::XV];
#pragma unroll
    for (int i = 0; i < F::XV; i++) {
        const int c = threadIdx.x + NT * i, r = c / (BK / 8);
        xrow[i] = c < TMF * BK / 8 && r < ne ? p.h + (int64_t) (r0 + r) * p.n_ff : nullptr;
    }
    auto issue_x = [&](int m) {
        if (m < NS) {
            unsigned char * dst = xbuf + (m % XS) * F::XST;
#pragma unroll
            for (int i = 0; i < F::XV; i++) {
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
    const uint32_t bsel = (tg & 1) ? 0x7632u : 0x5410u;
    const __half2 k1024 = __half2half2(__ushort_as_half(0x6400));

    Acc4 tot[MT][2];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) tot[i][j][0] = tot[i][j][1] = tot[i][j][2] = tot[i][j][3] = 0.0f;

    issue_w(0);
    issue_x(0);
    cp_async_commit();
#pragma unroll
    for (int s = 1; s < D; s++) {
        issue_x(s);
        cp_async_commit();
    }
    for (int m = 0; m < NS; m++) {
        cp_async_wait<D - 1>();
        __syncthreads();
        if constexpr (F::SPLIT) {
            if (m % SPW == 0) {
                issue_w(m / SPW + 1);
            }
            cp_async_commit();
            issue_x(m + D);
        } else {
            issue_x(m + D);
            if (m % SPW == 0) {
                issue_w(m / SPW + 1);
            }
        }
        cp_async_commit();

        // SPLIT: the stage at byte (R * RB) % CP of its rows
        const unsigned char * ws =
            wbuf + ((m / SPW) & 1) * F::WST + (F::SPLIT ? (m / SPW) * RB % F::CP : 0);
        const int c = m % SPW;
        uint32_t bq[2][4][2];
        if constexpr (WT == T_Q8_0) {
            const __half2 k1152 = __half2half2(__ushort_as_half(0x6480));
#pragma unroll
            for (int j = 0; j < 2; j++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const unsigned char * blk = ws + wr[j] * F::RS + 34 * (2 * c + b);
                    const __half2 d = __half2half2(*(const __half *) blk);
#pragma unroll
                    for (int hi = 0; hi < 2; hi++) {
                        const uint16_t * q = (const uint16_t *) (blk + 2 + 16 * hi + 2 * tg);
                        const uint32_t x = ((uint32_t) q[0] | ((uint32_t) q[4] << 16)) ^ 0x80808080u;
                        bq[j][2 * b + hi][0] = h2_as_u32(__hmul2(__hsub2(magic_lo(x), k1152), d));
                        bq[j][2 * b + hi][1] = h2_as_u32(__hmul2(__hsub2(magic_hi(x), k1152), d));
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const unsigned char * blk = ws + wr[j] * RB + 24 * (2 * c + b);
                    const __half2 dm = *(const __half2 *) blk;
                    const __half2 d = __half2half2(__low2half(dm)), mm = __half2half2(__high2half(dm));
                    const uint32_t qh = *(const uint32_t *) (blk + 4);
                    const uint32_t * q = (const uint32_t *) (blk + 8);
                    const uint32_t x = __byte_perm(q[tg >> 1], q[(tg >> 1) + 2], bsel);
#pragma unroll
                    for (int hi = 0; hi < 2; hi++) {
                        const uint32_t t = (qh >> (16 * hi + 2 * tg)) & 0x303u;
                        const uint32_t h = ((t & 1u) | ((t & 2u) << 7) | ((t & 0x100u) << 8) | ((t & 0x200u) << 15)) << 4;
                        const uint32_t v = (hi ? (x >> 4) & 0x0F0F0F0Fu : x & 0x0F0F0F0Fu) | h;
                        bq[j][2 * b + hi][0] = h2_as_u32(__hfma2(__hsub2(magic_lo(v), k1024), d, mm));
                        bq[j][2 * b + hi][1] = h2_as_u32(__hfma2(__hsub2(magic_hi(v), k1024), d, mm));
                    }
                }
            }
        }

        const uint32_t xs = (uint32_t) __cvta_generic_to_shared(xbuf + (m % XS) * F::XST);
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

    // the down sum (#826, see down_sum4): the ring is free after the last slice's barrier; each
    // row's flag: 1 its token's last pair before storing, 2 after, 0 not the last
    const bool dsum = p.sum_cnt != nullptr;
    float * dsown = (float *) sm;
    int * dsflag = (int *) (sm + F::DS_FLAG);
    int * dslist = dsflag + TMF;
    int * dsn = dslist + TMF;
    const int64_t NB = p.n_embd / BN;
    if (dsum) {
        static_assert(BN == DSUM_BN, "one counter per block's columns");
        __syncthreads();
        for (int r = threadIdx.x; r < ne; r += NT) {
            dsflag[r] = __ldcg(&p.sum_cnt[(int64_t) (rpair[r] / p.n_used) * NB + rt]) == 1 ? 1 : 0;
        }
        if (threadIdx.x == 0) {
            *dsn = 0;
        }
        __syncthreads();
    }

    // lane (g, tg) holds rows 16 i + g and 16 i + g + 8, columns 16 warp + 8 j + 2 tg and + 1
#pragma unroll
    for (int i = 0; i < MT; i++) {
        if (i < mvalid) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int r = 16 * i + g + 8 * hh;
                if (r < ne) {
                    const float wt = rw[r];
                    const int64_t at = (int64_t) rpair[r] * p.n_embd + (int64_t) rt * BN + 16 * warp + 2 * tg;
                    const bool keep = dsum && dsflag[r] == 1;
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const float v0 = tot[i][j][2 * hh] * wt, v1 = tot[i][j][2 * hh + 1] * wt;
                        if (keep) {
                            *(float2 *) (dsown + r * BN + 16 * warp + 8 * j + 2 * tg) = make_float2(v0, v1);
                        } else if (p.d_fmt == 1) {
                            __stcs((unsigned int *) ((__half *) p.d + at + 8 * j), h2_as_u32(__floats2half2_rn(v0, v1)));
                        } else if (p.d_fmt == 2) {
                            const __nv_bfloat162 b = __floats2bfloat162_rn(v0, v1);
                            __stcs((unsigned int *) ((__nv_bfloat16 *) p.d + at + 8 * j), *(const unsigned int *) &b);
                        } else {
                            __stcs((float2 *) (p.d + at + 8 * j), make_float2(v0, v1));
                        }
                    }
                }
            }
        }
    }
    if (!dsum) {
        return;
    }
    // count the stored rows down; the rows to sum in a list (any order: tokens are independent)
    __threadfence();
    __syncthreads();
    for (int r = threadIdx.x; r < ne; r += NT) {
        int f = dsflag[r];
        if (f == 0 && atomicSub(&p.sum_cnt[(int64_t) (rpair[r] / p.n_used) * NB + rt], 1) == 1) {
            f = 2;
        }
        if (f != 0) {
            dslist[atomicAdd(dsn, 1)] = r;
        }
        dsflag[r] = f;
    }
    __syncthreads();
    __threadfence();
    // 32 threads per row: four columns each
    constexpr int C4 = BN / 4;
    const int nl = *dsn;
    for (int k = threadIdx.x; k < nl * C4; k += NT) {
        const int r = dslist[k / C4], c4 = k % C4;
        const int pair = rpair[r];
        down_sum4(p, pair / p.n_used, (int64_t) rt * BN + 4 * c4, pair % p.n_used,
                  dsflag[r] == 1 ? dsown + r * BN + 4 * c4 : nullptr);
    }
}

// WALK as in gemm_frag_kernel
template <int WT, int BN, int RK, int XS, int MB, bool WALK>
__global__ void __launch_bounds__(FragDown<WT, BN, RK, XS>::NT, MB) gemm_frag_down_kernel(const legion_moe_args p) {
    if constexpr (WALK) {
        const int nt = p.tile0[p.n_expert];
        for (int y = blockIdx.y; y < nt; y += gridDim.y) {
            gemm_frag_down_tile<WT, BN, RK, XS>(p, y);
            __syncthreads();
        }
    } else {
        gemm_frag_down_tile<WT, BN, RK, XS>(p, blockIdx.y);
    }
}
