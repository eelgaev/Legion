// moe_gemm.cu, sm_80 and newer: gemm_raw_kernel, the GEMM that stages the weights raw through a
// cp.async ring and runs the m16n8k16 step of sm80/moe_gemm.cuh.
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace, after
// gemm_kernel (it uses Piece, Tile, tile_of, store_tile, row_base and koff from there). The
// cp.async helpers keep their __CUDA_ARCH__ guards and __trap() stubs, so the file compiles for
// every target architecture; moe_gemm.cu's launch_gemm picks the kernel per device at run time.

// ---- sm_80+: weights staged raw through a cp.async ring ---------------------------------------
//
// The GB10 streams the experts' rows at ~130 GB/s when each K step touches 32-48 bytes of each
// of BN rows 1-2 KB apart (gemm_kernel's loads), and at ~190-200 GB/s when consecutive lanes
// read 80-288 contiguous bytes of a row (measured with the GEMM's grid and rows, no
// arithmetic). So this kernel copies a stage of RK = 128 values per row (Q4_K: the
// super-block's 16-byte header and the 64 bytes of the half it covers; Q5_1: 4 whole blocks)
// into shared memory with cp.async, consecutive lanes on consecutive 16 bytes of a row, two
// stages in flight, and converts each BK slice from there into the F16 tile the m16n8k16 step
// reads: the same conversion and products as gemm_kernel, so the same output bits. One X + W
// tile (two barriers per slice) keeps a block under 48 KB of shared memory: two blocks per SM.

template <int T> struct RawFmt;
template <> struct RawFmt<T_Q4_K> {
    static constexpr int RK = 128, RB = 80, CP = 16, TAB = 0;
    __device__ static uint16_t tab_entry(int) { return 0; }
    // byte offset in the row of 16-byte chunk c of stage R
    __device__ static int64_t src(int R, int c) {
        const int64_t sb = (int64_t) (R / 2) * 144;
        return c == 0 ? sb : sb + 16 + 64 * (R % 2) + 16 * (c - 1);
    }
    // the piece of values kg .. kg + 31 (row k) from its stage's bytes
    __device__ static void fetch(Piece<T_Q4_K> & pw, const uint8_t * raw, int64_t kg) {
        pw.hdr = *(const uint4 *) raw;
        const uint4 * q = (const uint4 *) (raw + 16 + 32 * ((kg % 128) / 64));
        pw.q0 = q[0];
        pw.q1 = q[1];
    }
};
template <> struct RawFmt<T_Q5_1> {
    static constexpr int RK = 128, RB = 96, CP = 16, TAB = 0;
    __device__ static uint16_t tab_entry(int) { return 0; }
    __device__ static int64_t src(int R, int c) { return (int64_t) R * RB + 16 * c; }
    __device__ static void fetch(Piece<T_Q5_1> & pw, const uint8_t * raw, int64_t kg) { pw.fetch(raw, kg % RK); }
};

// IQ2_S: one 82-byte super-block per stage. Its blocks are 2-byte aligned only, so the stage is
// copied in 4-byte pieces from the aligned word before it, into an 84-byte slot; the codebook
// goes to shared memory (TAB bytes).
// (TAB: codebook entries kept in shared memory, 2 bytes each)
template <> struct RawFmt<T_IQ2_S> {
    static constexpr int RK = 256, RB = 84, CP = 4, TAB = 1024;
    __device__ static uint16_t tab_entry(int i) { return iq2_code(iq2s_grid[i]); }
    __device__ static int64_t src(int R, int c) { return ((int64_t) R * 82 & ~(int64_t) 3) + 4 * c; }
    __device__ static void fetch(Piece<T_IQ2_S> & pw, const uint8_t * raw, int64_t kg) {
        pw.fetch(raw + (((kg / 256) * 82) & 3), kg % 256);
    }
};
// IQ2_XXS: 66-byte super-blocks, 2-byte aligned: as IQ2_S, into 68-byte slots
template <> struct RawFmt<T_IQ2_XXS> {
    static constexpr int RK = 256, RB = 68, CP = 4, TAB = 256;
    __device__ static uint16_t tab_entry(int i) { return iq2_code(iq2xxs_grid[i]); }
    __device__ static int64_t src(int R, int c) { return ((int64_t) R * 66 & ~(int64_t) 3) + 4 * c; }
    __device__ static void fetch(Piece<T_IQ2_XXS> & pw, const uint8_t * raw, int64_t kg) {
        pw.fetch(raw + (((kg / 256) * 66) & 3), kg % 256);
    }
};

__device__ __forceinline__ void cp_async4(void * smem, const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gmem));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
// all but the newest `N` groups of this thread's copies have landed
template <int N> __device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <int TM, int BN, int WM, int WN, int WT>
struct RawShape {
    using F = RawFmt<WT>;
    static constexpr int WARPS_M = TM / WM, WARPS_N = BN / WN, NT = 32 * WARPS_M * WARPS_N;
    static constexpr int CH = F::RB / F::CP;             // copies per row and stage
    static constexpr int RAW = BN * F::RB;               // one raw stage
    static constexpr int SPR = F::RK / BK;               // BK slices per raw stage
    static constexpr int XV = TM * BK / 8 / NT;          // 16-byte X chunks per thread and slice
    static constexpr int PIECES = BN * BK / 32 / NT;     // 32-value pieces per thread and slice
    static constexpr int TILE = (TM + BN) * BK * 2;      // X + W (F16) tiles
    static constexpr int SMEM_AB = TILE + 2 * RAW + 2 * F::TAB;
    static constexpr int LDE = BN + 4;
    static constexpr int SMEM_E = TM * LDE * 4 + TM * 8;  // E, then store_tile's rows
    static constexpr int SMEM = SMEM_AB > SMEM_E ? SMEM_AB : SMEM_E;
    static_assert(F::RB % F::CP == 0 && F::RK % BK == 0, "raw stage of whole copies and slices");
    static_assert(PIECES >= 1 && (BN * BK / 32) % NT == 0 && XV >= 1, "pieces per thread");
};

template <int TM, int BN, int WM, int WN, int WT, Kind K>
__global__ void __launch_bounds__(RawShape<TM, BN, WM, WN, WT>::NT, 2)
gemm_raw_kernel(const legion_moe_args p) {
    using S = RawShape<TM, BN, WM, WN, WT>;
    using F = RawFmt<WT>;
    constexpr int NT = S::NT, MT = WM / 16, NT8 = WN / 8;
    extern __shared__ __align__(16) unsigned char sm[];

    const Tile tl = tile_of(p, blockIdx.y, TM);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int rt = blockIdx.x;
    const int64_t KD = K == GU ? p.n_embd : p.n_ff;
    const int NR = (int) (KD / F::RK), NS = (int) (KD / BK);

    auto wrow_of = [&](int r) -> const uint8_t * {
        if (K == GU) {
            const int64_t row = (int64_t) rt * (BN / 2) + r / 2;
            return ((r & 1) ? p.up : p.gate) + tl.slot * p.gu_nb2 + row * p.gu_nb1;
        }
        const int64_t row = (int64_t) rt * BN + r;
        return p.down + tl.slot * p.down_nb2 + row * p.down_nb1;
    };
    unsigned char * raw = sm + S::TILE;
    uint16_t * tab = (uint16_t *) (raw + 2 * S::RAW);  // the codebook behind the raw stages, if any
    // raw stage R into buffer R & 1; an empty group past the last stage
    auto issue_raw = [&](int R) {
        if (R < NR) {
            unsigned char * dst = raw + (R & 1) * S::RAW;
            for (int f = threadIdx.x; f < BN * S::CH; f += NT) {
                const int r = f / S::CH, c = f % S::CH;
                if constexpr (F::CP == 16) {
                    cp_async16(dst + r * F::RB + 16 * c, wrow_of(r) + F::src(R, c));
                } else {
                    cp_async4(dst + r * F::RB + 4 * c, wrow_of(r) + F::src(R, c));
                }
            }
        }
        cp_async_commit();
    };

    // this thread's X rows (gathered by token for GU, list order for DOWN)
    const __half * xrow[S::XV];
#pragma unroll
    for (int i = 0; i < S::XV; i++) {
        const int c = threadIdx.x + NT * i, r = c / (BK / 8);
        if (r < ne) {
            if (K == GU) {
                const int pair = p.list[r0 + r];
                xrow[i] = p.xh + (int64_t) (pair / p.n_used) * p.n_embd;
            } else {
                xrow[i] = p.h + (int64_t) (r0 + r) * p.n_ff;
            }
        } else {
            xrow[i] = nullptr;
        }
    }
    uint4 xr[S::XV];
    auto fetch_x = [&](int64_t k0) {
#pragma unroll
        for (int i = 0; i < S::XV; i++) {
            const int c = (threadIdx.x + NT * i) % (BK / 8);
            xr[i] = xrow[i] ? *(const uint4 *) (xrow[i] + k0 + 8 * c) : make_uint4(0, 0, 0, 0);
        }
    };
    // slice m's X chunks and its weights, converted from its raw stage, into the tile
    auto commit = [&](int m) {
        char * Xs = (char *) sm;
        char * Ws = Xs + TM * BK * 2;
#pragma unroll
        for (int i = 0; i < S::XV; i++) {
            const int c = threadIdx.x + NT * i;
            *(uint4 *) (Xs + (row_base(c / (BK / 8)) ^ ((c % (BK / 8)) << 4))) = xr[i];
        }
        const unsigned char * rb = raw + ((m / S::SPR) & 1) * S::RAW;
#pragma unroll
        for (int i = 0; i < S::PIECES; i++) {
            const int pc = threadIdx.x + NT * i, r = pc / 2, half = pc % 2;
            const int64_t kg = (int64_t) m * BK + 32 * half;
            Piece<WT> pw;
            if constexpr (F::TAB > 0) {
                pw.tab = tab;
            }
            F::fetch(pw, rb + r * F::RB, kg);
            pw.prepare(kg);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t o[4];
                pw.part(2 * q, o);
                pw.part(2 * q + 1, o + 2);
                *(uint4 *) (Ws + koff(r, 4 * half + q, BN)) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
    };

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / S::WARPS_N, wn = warp % S::WARPS_N;
    const int mbase = wm * WM;
    int mvalid = (ne - mbase + 15) / 16;
    mvalid = mvalid < 0 ? 0 : (mvalid > MT ? MT : mvalid);
    Acc4 tot16[MT][NT8];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT8; j++) tot16[i][j][0] = tot16[i][j][1] = tot16[i][j][2] = tot16[i][j][3] = 0.0f;

    // the codebook, compact (made visible by the barrier below)
    for (int i = threadIdx.x; i < F::TAB; i += NT) {
        tab[i] = F::tab_entry(i);
    }
    issue_raw(0);
    issue_raw(1);
    fetch_x(0);
    cp_async_wait<1>();
    __syncthreads();
    commit(0);
    __syncthreads();
    for (int m = 0; m < NS; m++) {
        const bool more = m + 1 < NS;
        // slice m + 1 starts raw stage S1: stage S1 - 1's slices are all converted, so its
        // buffer takes stage S1 + 1; then wait for stage S1 (visible after the barrier below)
        const bool next_stage = more && (m + 1) % S::SPR == 0;
        if (more) {
            fetch_x((int64_t) (m + 1) * BK);
        }
        if (next_stage) {
            issue_raw((m + 1) / S::SPR + 1);
            cp_async_wait<1>();
        }
        if (mvalid > 0) {
            step_mma16<MT, NT8, TM, BN>((const char *) sm, mbase, wn * WN, mvalid, lane, tot16);
        }
        __syncthreads();
        if (more) {
            commit(m + 1);
        }
        __syncthreads();
    }
    cp_async_wait<0>();

    float * E = (float *) sm;
    const int g = lane >> 2, tg = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT8; j++)
#pragma unroll
            for (int v = 0; v < 4; v++) {
                const int r = mbase + 16 * i + g + 8 * (v >> 1);
                const int c = wn * WN + 8 * j + 2 * tg + (v & 1);
                E[r * S::LDE + c] = tot16[i][j][v];
            }
    __syncthreads();
    store_tile<TM, BN, S::LDE, NT, K, true>(p, E, r0, ne, rt);
}
