// nvfp4_a4.cu: the dense W4A4 GEMM with its stages filled by the tensor memory accelerator (TMA)
// instead of per-thread cp.async copies (#1133).
//
// Not a standalone header: nvfp4_a4.cu includes it inside its anonymous namespace, after
// nvfp4_a4_gemm (mma_fp4, a4_epilogue). The kernel body is built only with LEGION_FP4_MMA
// (sm_120a / sm_121a), as the other W4A4 kernels.
//
// nvfp4_a4_gemm's dense 16-byte-copy tile (Tile<2, 8, 4, 2, 1, true>: 128 weight rows x 128
// tokens, four NVFP4 blocks per stage) spends a large part of each stage issuing its copies:
// 9 bounds-checked 16-byte cp.async per weight row, 8 per token's codes and one per token's
// scales, then a CTA barrier at every step. Here one thread issues a stage as three or four TMA
// tiles that complete on the stage's "full" mbarrier: the weight rows as stored (KB 36-byte
// blocks, 144 bytes a row; SwiGLU: the gate's 64 rows, then the same 64 rows of the up weight),
// the tokens' codes (128 bytes a token, in the TMA's 128-byte swizzle) and their scale words (16
// bytes a token). Each warp arrives on the stage's counter when its reads of the stage are done,
// and the last warp to arrive refills it, so no warp waits for the others except on data.
//
// The fragments are nvfp4_a4_gemm's: A read a word at a time from the weight blocks as stored
// (the RAW tile; a 144-byte row pitch, as LDW there), B by ldmatrix from the swizzled codes (16-
// byte chunk c of token r at c ^ (r % 8): the eight token rows of each ldmatrix land in distinct
// bank groups, as the padded rows there), the scale words as stored. Rows past m and tokens past
// n are zero-filled by the TMA as cp.async's zero-size copies do. Every output takes the same
// MMAs in the same k order and the same epilogue: the same bits as nvfp4_a4_gemm.

namespace a4t {
// The tensor maps of a launch: the weight rows (SwiGLU: gate and up; w2 = w otherwise), the
// activation codes and their scale words.
struct Maps {
    CUtensorMap w, w2, xq, xs;
};
}  // namespace a4t

// The tile: 8 warps, 4 over the weight rows x 2 over the tokens, each 32 weight rows (MI 2) x
// 8 * NT tokens; 4 NVFP4 blocks (256 of k) per stage, ST stages, one CTA per SM.
template <int NT_, int ST_> struct TmaTile {
    static constexpr int MI = 2, NT = NT_, KB = 4, ST = ST_, WM = 4, WN = 2;
    static constexpr int BM = 16 * MI * WM, BN = 8 * NT * WN;
    static constexpr int LDW = KB * 36;                    // a weight row's blocks as stored
    static constexpr int X_OFF = BM * LDW;                 // the codes: BN rows of KB * 32 bytes
    static constexpr int XS_OFF = X_OFF + BN * KB * 32;    // the scale words: BN rows of KB * 4
    static constexpr uint32_t TX = BM * LDW + BN * KB * 36;  // bytes a stage receives
    static constexpr int STAGE = (XS_OFF + BN * KB * 4 + 1023) / 1024 * 1024;
    static constexpr int SMEM = ST * STAGE + 1024;  // + 1 KiB to align the stages
    static_assert(KB * 32 == 128 && X_OFF % 1024 == 0 && XS_OFF % 128 == 0, "swizzle and TMA alignment");
    static_assert(SMEM + 1024 <= 99 * 1024, "shared memory");
};

// PROBE (legion nvfp4-a4-bench only; its outputs are not the GEMM's): 1 the loads alone (no
// MMAs, no stores), 2 no refills (the first ST stages' data computed on over and over: the MMAs
// and fragment reads without the memory system), 3 no epilogue stores.
// GEN: the stages' base aligned through an integer (the kernel before #1133 r12), which loses
// the pointer's state space: every A fragment and scale word is a generic load (LD.E) and each
// TMA issue checks its destination's window. Else the base is the shared array plus an offset,
// and those reads are LDS.
template <int NT, int ST, bool GLU, bool RES, int PROBE = 0, bool GEN = false>
__global__ void __launch_bounds__(256, 1) nvfp4_a4_gemm_tma(const legion_nvfp4_a4 p, const __grid_constant__ a4t::Maps maps) {
#ifdef LEGION_FP4_MMA
    using T = TmaTile<NT, ST>;
    using namespace legion_tma;
    constexpr int MI = T::MI, KB = T::KB, BM = T::BM, BN = T::BN, WN = T::WN, LDW = T::LDW;
    constexpr int STAGE = T::STAGE, X_OFF = T::X_OFF, XS_OFF = T::XS_OFF, WARPS = 8;
    constexpr int BM_OUT = GLU ? BM / 2 : BM;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem;
    if (GEN) {
        smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    } else {
        // The static arrays below come first, so the dynamic base is not 1024-aligned whatever
        // its declaration says: the empty asm hides the declared alignment, which would fold the
        // offset to 0.
        uint32_t base = saddr(smem_raw);
        asm volatile("" : "+r"(base));
        smem = smem_raw + ((1024 - (base & 1023)) & 1023);
    }
    __shared__ uint64_t full[ST];
    __shared__ uint32_t released[ST];
    __shared__ float x_dec[BN];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / WN, wn = warp % WN;

    // nvfp4_a4_gemm's dense tile order: groups of group_m row tiles, token tiles inside a group.
    const int pid = blockIdx.x, in_group = p.group_m * p.tiles_n;
    const int first_m = pid / in_group * p.group_m;
    const int group = min(p.tiles_m - first_m, p.group_m);
    const int64_t m0 = (int64_t) (first_m + pid % in_group % group) * BM_OUT;
    const int64_t n0 = (int64_t) (pid % in_group / group) * BN;
    const int n_tok = (int) min((int64_t) BN, p.n - n0);
    for (int i = threadIdx.x; i < BN; i += 256) {
        x_dec[i] = i < n_tok ? p.xd[n0 + i] : 0.0f;
    }
    const float sc = p.scale ? *p.scale : 1.0f;
    const float sc2 = p.scale2 ? *p.scale2 : 1.0f;
    const int64_t nblk = p.k / 64, steps = nblk / KB;

    // Step `step` into stage `stage` (one thread).
    auto issue = [&](int64_t step, int stage) {
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        bar_expect_tx(bar, T::TX);
        tma_2d(s, &maps.w, (int) (step * KB * 36), (int) m0, bar);
        if (GLU) {
            tma_2d(s + BM / 2 * LDW, &maps.w2, (int) (step * KB * 36), (int) m0, bar);
        }
        tma_2d(s + X_OFF, &maps.xq, (int) (step * KB * 32), (int) n0, bar);
        tma_2d(s + XS_OFF, &maps.xs, (int) (step * KB * 4), (int) n0, bar);
    };

    // nvfp4_a4_gemm's L2 bulk prefetch of the weight rows (its DENSE16 tile: 8 blocks per run,
    // only from the first two token tiles of each group of row tiles).
    constexpr int PF = 8 / KB;
    const bool pf_on = n0 < 2 * BN;
    auto prefetch = [&](int64_t step) {
        if (!pf_on) {
            return;
        }
        for (int r = threadIdx.x; r < BM; r += 256) {
            if (step >= steps) {
                break;
            }
            // Row r of nvfp4_a4_gemm's tile (SwiGLU: 16-row tiles of the gate and the up weight).
            int64_t row;
            const uint8_t * wt = p.w;
            if (GLU) {
                const int in = r % (16 * MI);
                row = m0 + r / (16 * MI) * (8 * MI) + in % (8 * MI);
                wt = in >= 8 * MI ? p.w2 : p.w;
            } else {
                row = m0 + r;
            }
            if (row < p.m) {
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + step * KB * 36);
                const uintptr_t e = (uintptr_t) (wt + row * p.w_ld + min(nblk, (step + PF) * KB) * 36);
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
            }
        }
    };
    prefetch(0);
    prefetch(PF);

    if (threadIdx.x == 0) {
        prefetch_map(&maps.w);
        if (GLU) {
            prefetch_map(&maps.w2);
        }
        prefetch_map(&maps.xq);
        prefetch_map(&maps.xs);
#pragma unroll
        for (int s = 0; s < ST; s++) {
            bar_init(full + s, 1);
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            if (s < steps) {
                issue(s, s);
            }
        }
    }

    float acc[MI][NT][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // The stage row of this warp's row tile i (SwiGLU: tile 0 the gate's rows, tile 1 the same
    // rows of the up weight, BM / 2 rows further); B: tokens of the lane's ldmatrix rows, and
    // which 16 bytes of a block's 32 it reads; scale token g.
    auto a_base = [&](int i) { return GLU ? i * (BM / 2) + wm * 16 : wm * 16 * MI + i * 16; };
    const int b_tok = wn * 8 * NT + lane / 16 * 8 + lane % 8, b_hi = (lane / 8) % 2;
    const int sb_tok = wn * 8 * NT + g;

    struct Frag {
        uint32_t a[MI][4], b[NT][2], sa[MI], sb[NT];
    };
    auto fragments = [&](const unsigned char * st, int blk, Frag & f) {
#pragma unroll
        for (int i = 0; i < MI; i++) {
            const unsigned char * r = st + (a_base(i) + g) * LDW + blk * 36 + 4 + t * 4;
            f.a[i][0] = *(const uint32_t *) r;
            f.a[i][1] = *(const uint32_t *) (r + 8 * LDW);
            f.a[i][2] = *(const uint32_t *) (r + 16);
            f.a[i][3] = *(const uint32_t *) (r + 8 * LDW + 16);
            f.sa[i] = *(const uint32_t *) (st + (a_base(i) + g + (t % 2) * 8) * LDW + blk * 36);
        }
#pragma unroll
        for (int jj = 0; jj < NT / 2; jj++) {
            const int row = b_tok + jj * 16;
            const int chunk = (blk * 2 + b_hi) ^ (row % 8);
            const uint32_t addr = saddr(st + X_OFF + row * 128 + chunk * 16);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(f.b[2 * jj][0]), "=r"(f.b[2 * jj][1]), "=r"(f.b[2 * jj + 1][0]), "=r"(f.b[2 * jj + 1][1])
                         : "r"(addr));
        }
        const uint32_t * bs = (const uint32_t *) (st + XS_OFF);
#pragma unroll
        for (int j = 0; j < NT; j++) {
            f.sb[j] = bs[(sb_tok + j * 8) * KB + blk];
        }
    };

    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % ST);
        if (step % PF == 0 && step > 0) {
            prefetch(step + PF);
        }
        if (PROBE != 2 || step < ST) {
            bar_wait(full + stage, (uint32_t) (step / ST) & 1);
        }
        const unsigned char * st = smem + stage * STAGE;
        if (PROBE != 1) {
            Frag f[2];
            fragments(st, 0, f[0]);
#pragma unroll
            for (int b = 0; b < KB; b++) {
                if (b + 1 < KB) {
                    fragments(st, b + 1, f[(b + 1) % 2]);
                }
                const Frag & c = f[b % 2];
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        mma_fp4(acc[i][j], c.a[i], c.b[j][0], c.b[j][1], c.sa[i], c.sb[j]);
                    }
            }
        }
        if (PROBE != 2 && step + ST < steps) {
            // This warp's reads of the stage are done (their values are in the MMAs above). The
            // acq_rel count orders every warp's reads before the last arrival, which refills
            // the stage with step + ST (after a proxy fence: the TMA writes through the async
            // proxy).
            __syncwarp();
            if (lane == 0) {
                uint32_t prev;
                asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                             : "=r"(prev)
                             : "r"(saddr(released + stage))
                             : "memory");
                if (prev % WARPS == WARPS - 1) {
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    issue(step + ST, stage);
                }
            }
            __syncwarp();
        }
    }
    // The probes without stores keep their sums alive by a store no launch asks for.
    if (PROBE == 0 || PROBE == 2 || p.act == 99) {
        a4_epilogue<MI, NT, false, GLU, RES>(p, acc, m0, n0, n_tok, x_dec, nullptr, sc, sc2, wm, wn, g, t);
    }
#else
    __trap();  // launched only where nvfp4_a4_probe answers 1
#endif
}

// nvfp4_a4_gemm_tma with persistent CTAs (#1133 r13): one CTA per SM walks the output tiles
// blockIdx.x, blockIdx.x + gridDim.x, ... (the same tile order as one CTA per tile, so the CTAs
// at work at once share weight rows and activations through L2 as before) and its stages run on
// across them: the stages of the next tile's first steps are refilled while this tile's last
// steps finish, so its loads fly while the epilogue stores this tile, and no CTA starts, inits
// its barriers or waits for its first stage between tiles. Each warp loads its tokens' decode
// scales at the start of a tile (used only by the epilogue, so the load latency is hidden) and
// RES prefetches the tile's residual into L2 there. The stage refill (REL):
// 0: as nvfp4_a4_gemm_tma, each warp counts its release with an acq_rel atom (a MEMBAR.ALL.CTA
//    per warp per stage) and the last one refills;
// 1: each warp arrives on the stage's "empty" mbarrier (release) and warp 0's lane 0 waits for
//    the other warps there (acquire) and refills: one barrier wait per stage in one warp instead
//    of eight fences (warp 0's SMSP partner keeps the tensor cores busy meanwhile).
// Every output: the same MMAs in the same k order and the same epilogue as nvfp4_a4_gemm_tma.
template <int NT, int ST, bool GLU, bool RES, int REL>
__global__ void __launch_bounds__(256, 1) nvfp4_a4_gemm_tmap(const legion_nvfp4_a4 p, const __grid_constant__ a4t::Maps maps) {
#ifdef LEGION_FP4_MMA
    using T = TmaTile<NT, ST>;
    using namespace legion_tma;
    constexpr int MI = T::MI, KB = T::KB, BM = T::BM, BN = T::BN, WN = T::WN, WM = T::WM, LDW = T::LDW;
    constexpr int STAGE = T::STAGE, X_OFF = T::X_OFF, XS_OFF = T::XS_OFF, WARPS = 8;
    constexpr int BM_OUT = GLU ? BM / 2 : BM, WT = 8 * NT, XR = (WT + 31) / 32;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    // As nvfp4_a4_gemm_tma: the empty asm keeps the 1024-byte alignment offset (the static
    // arrays come first, so the dynamic base is not 1024-aligned whatever its declaration says).
    uint32_t sbase = saddr(smem_raw);
    asm volatile("" : "+r"(sbase));
    unsigned char * smem = smem_raw + ((1024 - (sbase & 1023)) & 1023);
    __shared__ uint64_t full[ST], empty[ST];
    __shared__ uint32_t released[ST];
    // Row of warps wm's decode scales of the tile's tokens: warp (wm, wn) writes and reads only
    // its own WT tokens, so no CTA barrier orders them.
    __shared__ float x_dec[WM][BN];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / WN, wn = warp % WN;

    // nvfp4_a4_gemm's dense tile order: groups of group_m row tiles, token tiles inside a group.
    const int in_group = p.group_m * p.tiles_n;
    auto coords = [&](int pid, int64_t & m0, int64_t & n0) {
        const int first_m = pid / in_group * p.group_m;
        const int group = min(p.tiles_m - first_m, p.group_m);
        m0 = (int64_t) (first_m + pid % in_group % group) * BM_OUT;
        n0 = (int64_t) (pid % in_group / group) * BN;
    };
    const int tiles = p.tiles_m * p.tiles_n;
    const int my_tiles = (tiles - (int) blockIdx.x + (int) gridDim.x - 1) / (int) gridDim.x;
    const float sc = p.scale ? *p.scale : 1.0f;
    const float sc2 = p.scale2 ? *p.scale2 : 1.0f;
    const int64_t nblk = p.k / 64, steps = nblk / KB, total = (int64_t) my_tiles * steps;

    // The CTA's step q (step q % steps of its tile q / steps) into stage q % ST (one thread).
    auto issue = [&](int64_t q) {
        const int it = (int) (q / steps);
        const int64_t step = q - it * steps;
        int64_t m0, n0;
        coords((int) blockIdx.x + it * (int) gridDim.x, m0, n0);
        const int stage = (int) (q % ST);
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        bar_expect_tx(bar, T::TX);
        tma_2d(s, &maps.w, (int) (step * KB * 36), (int) m0, bar);
        if (GLU) {
            tma_2d(s + BM / 2 * LDW, &maps.w2, (int) (step * KB * 36), (int) m0, bar);
        }
        tma_2d(s + X_OFF, &maps.xq, (int) (step * KB * 32), (int) n0, bar);
        tma_2d(s + XS_OFF, &maps.xs, (int) (step * KB * 4), (int) n0, bar);
    };

    // nvfp4_a4_gemm_tma's L2 bulk prefetch of the tile's weight rows.
    constexpr int PF = 8 / KB;
    auto prefetch = [&](int64_t m0, bool pf_on, int64_t step) {
        if (!pf_on) {
            return;
        }
        for (int r = threadIdx.x; r < BM; r += 256) {
            if (step >= steps) {
                break;
            }
            int64_t row;
            const uint8_t * wt = p.w;
            if (GLU) {
                const int in = r % (16 * MI);
                row = m0 + r / (16 * MI) * (8 * MI) + in % (8 * MI);
                wt = in >= 8 * MI ? p.w2 : p.w;
            } else {
                row = m0 + r;
            }
            if (row < p.m) {
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + step * KB * 36);
                const uintptr_t e = (uintptr_t) (wt + row * p.w_ld + min(nblk, (step + PF) * KB) * 36);
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
            }
        }
    };

    if (threadIdx.x == 0) {
        prefetch_map(&maps.w);
        if (GLU) {
            prefetch_map(&maps.w2);
        }
        prefetch_map(&maps.xq);
        prefetch_map(&maps.xs);
#pragma unroll
        for (int s = 0; s < ST; s++) {
            bar_init(full + s, 1);
            bar_init(empty + s, WARPS);
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            if (s < total) {
                issue(s);
            }
        }
    }

    auto a_base = [&](int i) { return GLU ? i * (BM / 2) + wm * 16 : wm * 16 * MI + i * 16; };
    const int b_tok = wn * 8 * NT + lane / 16 * 8 + lane % 8, b_hi = (lane / 8) % 2;
    const int sb_tok = wn * 8 * NT + g;

    struct Frag {
        uint32_t a[MI][4], b[NT][2], sa[MI], sb[NT];
    };
    auto fragments = [&](const unsigned char * st, int blk, Frag & f) {
#pragma unroll
        for (int i = 0; i < MI; i++) {
            const unsigned char * r = st + (a_base(i) + g) * LDW + blk * 36 + 4 + t * 4;
            f.a[i][0] = *(const uint32_t *) r;
            f.a[i][1] = *(const uint32_t *) (r + 8 * LDW);
            f.a[i][2] = *(const uint32_t *) (r + 16);
            f.a[i][3] = *(const uint32_t *) (r + 8 * LDW + 16);
            f.sa[i] = *(const uint32_t *) (st + (a_base(i) + g + (t % 2) * 8) * LDW + blk * 36);
        }
#pragma unroll
        for (int jj = 0; jj < NT / 2; jj++) {
            const int row = b_tok + jj * 16;
            const int chunk = (blk * 2 + b_hi) ^ (row % 8);
            const uint32_t addr = saddr(st + X_OFF + row * 128 + chunk * 16);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(f.b[2 * jj][0]), "=r"(f.b[2 * jj][1]), "=r"(f.b[2 * jj + 1][0]), "=r"(f.b[2 * jj + 1][1])
                         : "r"(addr));
        }
        const uint32_t * bs = (const uint32_t *) (st + XS_OFF);
#pragma unroll
        for (int j = 0; j < NT; j++) {
            f.sb[j] = bs[(sb_tok + j * 8) * KB + blk];
        }
    };

    // Probe (legion nvfp4-a4-bench; the outputs do not change): the odd CTAs start their k
    // loop `variant` bits 16-23 microseconds late (their first stages already loading), so
    // half the SMs store their epilogues while the other half compute.
    const uint32_t delay_ns = (uint32_t) (p.variant >> 16 & 0xff) * 1000u;
    if (delay_ns != 0 && (blockIdx.x & 1)) {
        uint64_t t0, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        do {
            __nanosleep(500);
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        } while (now - t0 < delay_ns);
    }

    int64_t q = 0;  // the CTA's step
    for (int it = 0; it < my_tiles; it++) {
        int64_t m0, n0;
        coords((int) blockIdx.x + it * (int) gridDim.x, m0, n0);
        const int n_tok = (int) min((int64_t) BN, p.n - n0);
        const bool pf_on = n0 < 2 * BN;
        // This warp's tokens' decode scales (lane's tokens lane, lane + 32, ... of its WT).
        float xr[XR];
#pragma unroll
        for (int i = 0; i < XR; i++) {
            const int c = lane + 32 * i, tk = wn * WT + c;
            xr[i] = c < WT && tk < n_tok ? p.xd[n0 + tk] : 0.0f;
        }
        if (RES && p.act == 0 && (int) threadIdx.x < n_tok) {
            // The tile's residual rows of token threadIdx.x into L2 for the epilogue.
            const int64_t rows = min((int64_t) BM, p.m - m0);
            const uintptr_t a = (uintptr_t) (p.res + (n0 + (int64_t) threadIdx.x) * p.m + m0);
            const uintptr_t a16 = a & ~(uintptr_t) 15;
            const uintptr_t e = a + rows * 4;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                         "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
        }
        prefetch(m0, pf_on, 0);
        prefetch(m0, pf_on, PF);

        float acc[MI][NT][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

        for (int64_t step = 0; step < steps; step++, q++) {
            const int stage = (int) (q % ST);
            const uint32_t parity = (uint32_t) (q / ST) & 1;
            if (step % PF == 0 && step > 0) {
                prefetch(m0, pf_on, step + PF);
            }
            bar_wait(full + stage, parity);
            const unsigned char * st = smem + stage * STAGE;
            Frag f[2];
            fragments(st, 0, f[0]);
#pragma unroll
            for (int b = 0; b < KB; b++) {
                if (b + 1 < KB) {
                    fragments(st, b + 1, f[(b + 1) % 2]);
                }
                const Frag & c = f[b % 2];
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NT; j++) {
                        mma_fp4(acc[i][j], c.a[i], c.b[j][0], c.b[j][1], c.sa[i], c.sb[j]);
                    }
            }
            if (q + ST < total) {
                // This warp's reads of the stage are done (their values are in the MMAs above);
                // the stage takes the CTA's step q + ST, the next tile's from its last steps.
                __syncwarp();
                if (REL == 0) {
                    if (lane == 0) {
                        uint32_t prev;
                        asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                     : "=r"(prev)
                                     : "r"(saddr(released + stage))
                                     : "memory");
                        if (prev % WARPS == WARPS - 1) {
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            issue(q + ST);
                        }
                    }
                } else if (lane == 0) {
                    bar_arrive(empty + stage);
                    if (warp == 0) {
                        bar_wait(empty + stage, parity);
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        issue(q + ST);
                    }
                }
                __syncwarp();
            }
        }
        // The decode scales for the epilogue (the warp's reads of the last tile's are done).
        __syncwarp();
#pragma unroll
        for (int i = 0; i < XR; i++) {
            const int c = lane + 32 * i;
            if (c < WT) {
                x_dec[wm][wn * WT + c] = xr[i];
            }
        }
        __syncwarp();
        a4_epilogue<MI, NT, false, GLU, RES>(p, acc, m0, n0, n_tok, x_dec[wm], nullptr, sc, sc2, wm, wn, g, t);
    }
#else
    __trap();  // launched only where nvfp4_a4_probe answers 1
#endif
}

// The tensor map of `rows` rows of `row_bytes` bytes `ld` apart at `base`, in boxes of
// box_bytes x box_rows (zero-filled past the ends); false where the driver refuses it.
inline bool a4_tma_map(CUtensorMap * map, const void * base, int64_t row_bytes, int64_t rows, int64_t ld,
                       int box_bytes, int box_rows, CUtensorMapSwizzle swz) {
    if (((uintptr_t) base & 15) != 0 || ld % 16 != 0 || row_bytes <= 0 || rows <= 0 || row_bytes > ld) {
        return false;
    }
    const cuuint64_t dims[2]    = { (cuuint64_t) row_bytes, (cuuint64_t) rows };
    const cuuint64_t strides[1] = { (cuuint64_t) ld };
    const cuuint32_t box[2]     = { (cuuint32_t) box_bytes, (cuuint32_t) box_rows };
    const cuuint32_t elem[2]    = { 1, 1 };
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void *) base, dims, strides, box, elem,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// nvfp4_a4_gemm_tma for a dense GEMM (SwiGLU, residual or plain) of whole stages (k % 256 == 0);
// -1 when it does not apply or a tensor map cannot be made (nothing launched). PROBE: plain
// GEMMs only.
template <int NT, int ST, int PROBE = 0, bool GEN = false> int dense_tma(const legion_nvfp4_a4 & p0, cudaStream_t st) {
    using T = TmaTile<NT, ST>;
    if (p0.plan || p0.k % 256 != 0 || p0.n <= 0 || p0.m <= 0 || (PROBE != 0 && (p0.w2 || p0.res))) {
        return -1;
    }
    legion_nvfp4_a4 p = p0;
    p.tiles_n = (int32_t) ((p.n + T::BN - 1) / T::BN);
    a4t::Maps maps;
    const int64_t row_bytes = p.k / 64 * 36;
    const int w_rows = p.w2 ? T::BM / 2 : T::BM;
    if (!a4_tma_map(&maps.w, p.w, row_bytes, p.m, p.w_ld, T::LDW, w_rows, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        (p.w2 && !a4_tma_map(&maps.w2, p.w2, row_bytes, p.m, p.w_ld, T::LDW, w_rows, CU_TENSOR_MAP_SWIZZLE_NONE)) ||
        !a4_tma_map(&maps.xq, p.xq, p.xq_ld, p.n, p.xq_ld, T::KB * 32, T::BN, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !a4_tma_map(&maps.xs, p.xs, p.xs_ld, p.n, p.xs_ld, T::KB * 4, T::BN, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return -1;
    }
    if (!p.w2) {
        maps.w2 = maps.w;
    }
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<(unsigned) p.tiles_m * (unsigned) p.tiles_n, 256, T::SMEM, st>>>(p, maps);
    };
    if constexpr (PROBE != 0) {
        go(nvfp4_a4_gemm_tma<NT, ST, false, false, PROBE, GEN>);
    } else if (p.w2) {
        go(nvfp4_a4_gemm_tma<NT, ST, true, false, 0, GEN>);
    } else if (p.res) {
        go(nvfp4_a4_gemm_tma<NT, ST, false, true, 0, GEN>);
    } else {
        go(nvfp4_a4_gemm_tma<NT, ST, false, false, 0, GEN>);
    }
    return status();
}

// The current device's SMs (0 when unknown); one query per device.
inline int a4_sm_count() {
    static int cached[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        cudaGetLastError();
        return 0;
    }
    if (cached[dev] <= 0) {
        int n = 0;
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        cached[dev] = n;
    }
    return cached[dev];
}

// dense_tma with persistent CTAs (nvfp4_a4_gemm_tmap): min(tiles, SMs) CTAs, one per SM.
template <int NT, int ST, int REL> int dense_tmap(const legion_nvfp4_a4 & p0, cudaStream_t st) {
    using T = TmaTile<NT, ST>;
    static_assert(ST * T::STAGE + 1024 + 4 * T::BN * 4 + 256 <= 99 * 1024, "shared memory with the static arrays");
    const int sms = a4_sm_count();
    if (p0.plan || p0.k % 256 != 0 || p0.n <= 0 || p0.m <= 0 || sms <= 0) {
        return -1;
    }
    legion_nvfp4_a4 p = p0;
    p.tiles_n = (int32_t) ((p.n + T::BN - 1) / T::BN);
    a4t::Maps maps;
    const int64_t row_bytes = p.k / 64 * 36;
    const int w_rows = p.w2 ? T::BM / 2 : T::BM;
    if (!a4_tma_map(&maps.w, p.w, row_bytes, p.m, p.w_ld, T::LDW, w_rows, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        (p.w2 && !a4_tma_map(&maps.w2, p.w2, row_bytes, p.m, p.w_ld, T::LDW, w_rows, CU_TENSOR_MAP_SWIZZLE_NONE)) ||
        !a4_tma_map(&maps.xq, p.xq, p.xq_ld, p.n, p.xq_ld, T::KB * 32, T::BN, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !a4_tma_map(&maps.xs, p.xs, p.xs_ld, p.n, p.xs_ld, T::KB * 4, T::BN, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return -1;
    }
    if (!p.w2) {
        maps.w2 = maps.w;
    }
    const int64_t tiles = (int64_t) p.tiles_m * p.tiles_n;
    const unsigned grid = (unsigned) (tiles < sms ? tiles : sms);
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<grid, 256, T::SMEM, st>>>(p, maps);
    };
    if (p.w2) {
        go(nvfp4_a4_gemm_tmap<NT, ST, true, false, REL>);
    } else if (p.res) {
        go(nvfp4_a4_gemm_tmap<NT, ST, false, true, REL>);
    } else {
        go(nvfp4_a4_gemm_tmap<NT, ST, false, false, REL>);
    }
    return status();
}

// dense_tma<NT, ST, probe, GEN> (legion nvfp4-a4-bench; probe 0 the GEMM).
template <int NT, int ST, bool GEN> int dense_tma_probe(const legion_nvfp4_a4 & p, cudaStream_t st, int probe) {
    switch (probe) {
        case 0: return dense_tma<NT, ST, 0, GEN>(p, st);
        case 1: return dense_tma<NT, ST, 1, GEN>(p, st);
        case 2: return dense_tma<NT, ST, 2, GEN>(p, st);
        case 3: return dense_tma<NT, ST, 3, GEN>(p, st);
        default: return -1;
    }
}
