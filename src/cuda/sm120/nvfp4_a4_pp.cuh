// nvfp4_a4.cu: the ping-pong dense W4A4 tile for the long-k plain and residual GEMMs (#1133 r23).
//
// Not a standalone header: nvfp4_a4.cu includes it inside its anonymous namespace, after
// nvfp4_a4_wide.cuh (a4t::Maps, a4_tma_map, a4w::tile_of, the TMA helpers, a4_epilogue). The
// kernel bodies are built only with LEGION_FP4_MMA (sm_120a / sm_121a), as the other W4A4 kernels.
//
// nvfp4_a4_gemm_pp: two consumer groups of 4 warps in one CTA per SM, each with its own output
// tile (PPTile: 16 * MI * WM rows x 8 * NT * WN tokens, WM x WN warps of 16 * MI x 8 * NT). The
// CTA walks its tiles (blockIdx.x, + gridDim.x, ... in the grouped order of GM row tiles, token
// tiles inside a group) and hands them to the groups in turn: tile 0 to group 0, tile 1 to group
// 1, tile 2 to group 0, ... One stage ring (ST stages of 4 NVFP4 blocks, filled by TMA as
// nvfp4_a4_gemm_tma's) runs through the CTA's tiles in that order, and the last of a group's 4
// warps to finish a stage refills it with the step ST further on, which may be the other group's
// tile. A group starts its k loop only once the other group has waited on every stage of its own
// tile (a named barrier: the other group arrives after its last stage wait), so the k loops take
// turns and each group's epilogue (a4t::epilogue: a4_epilogue's arithmetic, residual included,
// with fewer residuals in flight) runs while the other group's MMAs do (the CUTLASS sm90 ping-pong schedule, here with
// mma.sync). The turn order is also what keeps the full barriers' parity waits sound: a stage's
// use u is never waited on before use u - 1 has completed.
//
// The 192 x 128 wide tile (#1133 r21) reads fewer L2 bytes per MMA, but its epilogue is serial on
// the CTA; two groups of 96 accumulators a thread can only hold 2 x 12,288 outputs, so each
// group's tile here is the size of nvfp4_a4_gemm_tma's 128 x 96 (the same L2 bytes per MMA), or
// 128 x 128 with 128 accumulators a thread and one fragment set.
//
// Every output takes the same MMAs in the same k order and the same epilogue as
// nvfp4_a4_gemm_tma's: the same bits.

template <int MI_, int NT_, int WM_, int ST_> struct PPTile {
    static constexpr int MI = MI_, NT = NT_, WM = WM_, WN = 4 / WM_, KB = 4, ST = ST_;
    static constexpr int BM = 16 * MI * WM, BN = 8 * NT * WN;
    static constexpr int LDW = KB * 36;  // the weight box (a row's 4 blocks as stored)
    static constexpr int XB = KB * 32;   // a token's codes (the 128-byte swizzle span)
    static constexpr int SB = KB * 4;    // a token's scale words
    static constexpr int X_OFF = (BM * LDW + 1023) / 1024 * 1024;
    static constexpr int XS_OFF = X_OFF + BN * XB;
    static constexpr uint32_t TX = BM * LDW + BN * XB + BN * SB;  // bytes a stage receives
    static constexpr int STAGE = (XS_OFF + BN * SB + 1023) / 1024 * 1024;
    static constexpr int SMEM = ST * STAGE + 1024;  // + 1 KiB to align the stages
    // Fragment sets in flight (two: the next block's read during this block's MMAs).
    static constexpr int NF = MI * NT >= 32 ? 1 : 2;
    static_assert(WM * WN == 4 && NT % 2 == 0, "4 warps a group, ldmatrix pairs of n8 tiles");
    static_assert(X_OFF % 1024 == 0 && XS_OFF % 128 == 0 && STAGE % 1024 == 0, "swizzle and TMA alignment");
    static_assert(BM <= 256 && BN <= 256, "TMA box rows");
    static_assert(SMEM + 2048 <= 99 * 1024, "shared memory");
};

namespace a4p {
// The first byte along a row of step ks's three boxes (weights, codes, scale words). The kernel
// and the host's alignment check (pp_boxes_aligned) use the same functions.
__host__ __device__ constexpr int64_t w_c0(int64_t ks) {
    return ks * 4 * 36;
}
__host__ __device__ constexpr int64_t xq_c0(int64_t ks) {
    return ks * 4 * 32;
}
__host__ __device__ constexpr int64_t xs_c0(int64_t ks) {
    return ks * 4 * 4;
}
}  // namespace a4p

// Every box start and smem destination of a launch 16-byte aligned (on the GB10 an unaligned
// TMA box start is Xid 13, #1133 r21): checked on the host before a launch.
template <class T> bool pp_boxes_aligned(const legion_nvfp4_a4 & p) {
    if (((uintptr_t) p.w & 15) || ((uintptr_t) p.xq & 15) || ((uintptr_t) p.xs & 15) || p.w_ld % 16 || p.xq_ld % 16 ||
        p.xs_ld % 16) {
        return false;
    }
    const int64_t steps = p.k / 256;
    for (int64_t ks = 0; ks < steps; ks++) {
        if (a4p::w_c0(ks) % 16 || a4p::xq_c0(ks) % 16 || a4p::xs_c0(ks) % 16) {
            return false;
        }
    }
    for (int s = 0; s < T::ST; s++) {
        const int64_t o = (int64_t) s * T::STAGE;
        if (o % 1024 || (o + T::X_OFF) % 1024 || (o + T::XS_OFF) % 16) {
            return false;
        }
    }
    // The residual's L2 prefetch: 16-byte runs of whole rows.
    return !p.res || (((uintptr_t) p.res & 15) == 0 && p.m % 4 == 0);
}

#ifdef LEGION_FP4_MMA
namespace a4p {
// Named barriers (0 is __syncthreads): `n` threads sync / arrive.
__device__ __forceinline__ void nb_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void nb_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
}  // namespace a4p
#endif

// PROBE (legion nvfp4-a4-bench only, plain GEMMs; outputs not the GEMM's): 1 the loads alone, 3
// no epilogue. GM: groups of GM row tiles in the tile walk.
template <int MI, int NT, int WM, int ST, int GM, bool RES, int PROBE = 0, bool CS = false>
__global__ void __launch_bounds__(256, 1) nvfp4_a4_gemm_pp(const legion_nvfp4_a4 p, const __grid_constant__ a4t::Maps maps) {
#ifdef LEGION_FP4_MMA
    using T = PPTile<MI, NT, WM, ST>;
    using namespace legion_tma;
    constexpr int BM = T::BM, BN = T::BN, WN = T::WN, LDW = T::LDW, XB = T::XB, KB = T::KB;
    constexpr int STAGE = T::STAGE, X_OFF = T::X_OFF, XS_OFF = T::XS_OFF, GW = 4;  // warps a group
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    // As nvfp4_a4_gemm_tma: the empty asm hides the declared alignment (which would fold the
    // offset to 0), and the base keeps the shared state space (LDS fragment reads).
    uint32_t sbase = saddr(smem_raw);
    asm volatile("" : "+r"(sbase));
    unsigned char * smem = smem_raw + ((1024 - (sbase & 1023)) & 1023);
    __shared__ uint64_t full[ST];
    __shared__ uint32_t released[ST];
    __shared__ float x_dec[2][BN];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int grp = warp / GW, gwarp = warp % GW, wm = gwarp / WN, wn = gwarp % WN;
    const int gtid = threadIdx.x % (GW * 32);
    const int64_t tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN, tiles = tiles_m * tiles_n;
    const int64_t first = blockIdx.x, stride = gridDim.x;
    const int64_t mine = first < tiles ? (tiles - 1 - first) / stride + 1 : 0;
    const int64_t nblk = p.k / 64, steps = nblk / KB, total = mine * steps;
    const float sc = p.scale ? *p.scale : 1.0f;

    // Step gs of this CTA's sequence (tile gs / steps of its share, k step gs % steps) into stage
    // `stage` (one thread).
    auto issue = [&](int64_t gs, int stage) {
        int64_t tm, tn;
        a4w::tile_of(first + gs / steps * stride, tiles_m, tiles_n, GM, tm, tn);
        const int64_t ks = gs % steps;
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        bar_expect_tx(bar, T::TX);
        tma_2d(s, &maps.w, (int) a4p::w_c0(ks), (int) (tm * BM), bar);
        tma_2d(s + X_OFF, &maps.xq, (int) a4p::xq_c0(ks), (int) (tn * BN), bar);
        tma_2d(s + XS_OFF, &maps.xs, (int) a4p::xs_c0(ks), (int) (tn * BN), bar);
    };

    if (threadIdx.x == 0) {
        prefetch_map(&maps.w);
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
            if (s < total) {
                issue(s, s);
            }
        }
    }

    // The stage row of this warp's row tile i; B: tokens of the lane's ldmatrix rows, and which
    // 16 bytes of a block's 32 it reads; scale token g.
    auto a_base = [&](int i) { return wm * 16 * MI + i * 16; };
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
            // The TMA's 128-byte swizzle: 16-byte chunk c of token row r at c ^ (r % 8).
            const int chunk = (blk * 2 + b_hi) ^ (row % 8);
            const uint32_t addr = saddr(st + X_OFF + row * XB + chunk * 16);
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

    for (int64_t it = grp; it < mine; it += 2) {
        if (it > 0) {
            // The other group has waited on every stage of the tile before this one.
            a4p::nb_sync(1 + grp, 256);
        }
        int64_t tm, tn;
        a4w::tile_of(first + it * stride, tiles_m, tiles_n, GM, tm, tn);
        const int64_t m0 = tm * BM, n0 = tn * BN;
        const int n_tok = (int) min((int64_t) BN, p.n - n0);
        float * xd = x_dec[grp];
        for (int i = gtid; i < BN; i += GW * 32) {
            xd[i] = i < n_tok ? p.xd[n0 + i] : 0.0f;
        }
        // xd written before any of the group's epilogue reads (and, by the turn barrier above,
        // after its previous tile's).
        a4p::nb_sync(3 + grp, GW * 32);
        // nvfp4_a4_gemm_tma's L2 bulk prefetch of the weight rows (8 blocks a run, from the first
        // two token tiles only).
        constexpr int PF = 2;
        const bool pf_on = n0 < 2 * BN;
        auto prefetch = [&](int64_t step) {
            if (!pf_on) {
                return;
            }
            for (int r = gtid; r < BM; r += GW * 32) {
                const int64_t row = m0 + r;
                if (step < steps && row < p.m) {
                    const uintptr_t a = (uintptr_t) (p.w + row * p.w_ld + step * KB * 36);
                    const uintptr_t e = (uintptr_t) (p.w + row * p.w_ld + min(nblk, (step + PF) * KB) * 36);
                    const uintptr_t a16 = a & ~(uintptr_t) 15;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                                 "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
                }
            }
        };
        prefetch(0);
        prefetch(PF);

        float acc[MI][NT][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

        for (int64_t ks = 0; ks < steps; ks++) {
            const int64_t gs = it * steps + ks;
            const int stage = (int) (gs % ST);
            if (ks % PF == 0 && ks > 0) {
                prefetch(ks + PF);
            }
            if (RES && ks == steps / 2 && gtid < n_tok) {
                // The tile's residual rows into the L2 for the epilogue.
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p.res + (n0 + gtid) * p.m + m0),
                             "r"((uint32_t) min((int64_t) BM, p.m - m0) * 4));
            }
            bar_wait(full + stage, (uint32_t) (gs / ST) & 1);
            if (ks == steps - 1 && it + 1 < mine) {
                // Every stage of this tile waited on: the other group's k loop may start.
                a4p::nb_arrive(1 + (grp ^ 1), 256);
            }
            const unsigned char * st = smem + stage * STAGE;
            if (PROBE != 1) {
                if constexpr (T::NF == 2) {
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
                } else {
#pragma unroll
                    for (int b = 0; b < KB; b++) {
                        Frag c;
                        fragments(st, b, c);
#pragma unroll
                        for (int i = 0; i < MI; i++)
#pragma unroll
                            for (int j = 0; j < NT; j++) {
                                mma_fp4(acc[i][j], c.a[i], c.b[j][0], c.b[j][1], c.sa[i], c.sb[j]);
                            }
                    }
                }
            }
            if (gs + ST < total) {
                // As nvfp4_a4_gemm_tma: the last of the group's warps out refills the stage
                // (maybe with the other group's tile).
                __syncwarp();
                if (lane == 0) {
                    uint32_t prev;
                    asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev)
                                 : "r"(saddr(released + stage))
                                 : "memory");
                    if (prev % GW == GW - 1) {
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        issue(gs + ST, stage);
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (PROBE == 0) {
            a4t::epilogue<MI, NT, RES, CS>(p, acc, m0, n0, n_tok, xd, sc, wm, wn, g, t);
        } else {
            // The probes keep their sums alive by a store no launch asks for.
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NT; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) s += acc[i][j][e];
            if (p.act == 99) {
                ((float *) p.dst)[threadIdx.x] = s;
            }
        }
    }
#else
    __trap();  // launched only where nvfp4_a4_probe answers 1
#endif
}

// nvfp4_a4_gemm_pp for a dense plain or residual GEMM with F32 outputs (k % 256 == 0, m % 4 ==
// 0); -1 when it does not apply, a box would start unaligned or a tensor map cannot be made
// (nothing launched). PROBE: plain GEMMs only. CS: a4t::epilogue's streaming stores and residual
// loads. A grid of min(tiles, SMs).
template <int MI, int NT, int WM, int ST, int GM, int PROBE = 0, bool CS = false>
int dense_pp(const legion_nvfp4_a4 & p, cudaStream_t st) {
    using T = PPTile<MI, NT, WM, ST>;
    if (p.plan || p.w2 || p.act != 0 || p.k % 256 != 0 || p.n <= 0 || p.m <= 0 || p.m % 4 != 0 ||
        (PROBE != 0 && p.res) || !pp_boxes_aligned<T>(p)) {
        return -1;
    }
    a4t::Maps maps;
    const int64_t row_bytes = p.k / 64 * 36;
    if (!a4_tma_map(&maps.w, p.w, row_bytes, p.m, p.w_ld, T::LDW, T::BM, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !a4_tma_map(&maps.xq, p.xq, p.xq_ld, p.n, p.xq_ld, T::XB, T::BN, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !a4_tma_map(&maps.xs, p.xs, p.xs_ld, p.n, p.xs_ld, T::SB, T::BN, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return -1;
    }
    maps.w2 = maps.w;
    const int64_t tiles = (p.m + T::BM - 1) / T::BM * ((p.n + T::BN - 1) / T::BN);
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    const int64_t grid = tiles < sms ? tiles : sms;
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<(unsigned) grid, 256, T::SMEM, st>>>(p, maps);
    };
    if (p.res) {
        go(nvfp4_a4_gemm_pp<MI, NT, WM, ST, GM, true, 0, CS>);
    } else {
        go(nvfp4_a4_gemm_pp<MI, NT, WM, ST, GM, false, PROBE, CS>);
    }
    return status();
}

// dense_pp with probe `probe` (legion nvfp4-a4-bench; 0 the GEMM).
template <int MI, int NT, int WM, int ST, int GM>
int dense_pp_probe(const legion_nvfp4_a4 & p, cudaStream_t st, int probe) {
    switch (probe) {
        case 0: return dense_pp<MI, NT, WM, ST, GM, 0>(p, st);
        case 1: return dense_pp<MI, NT, WM, ST, GM, 1>(p, st);
        case 3: return dense_pp<MI, NT, WM, ST, GM, 3>(p, st);
        default: return -1;
    }
}
