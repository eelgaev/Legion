// nvfp4_a4.cu: the wide dense W4A4 tile for the long-k plain and residual GEMMs (#1133 r21), and a
// probe of TMA multicast between the two CTAs of a cluster.
//
// Not a standalone header: nvfp4_a4.cu includes it inside its anonymous namespace, after
// nvfp4_a4_tma.cuh (a4t::Maps, a4_tma_map, the TMA helpers). The kernel bodies are built only
// with LEGION_FP4_MMA (sm_120a / sm_121a), as the other W4A4 kernels.
//
// nvfp4_a4_gemm_wide: 192 weight rows x 128 tokens (8 warps, 4 over the rows x 2 over the tokens,
// each 48 rows x 64 tokens), the stages filled by TMA as nvfp4_a4_gemm_tma's. It reads 1.4x
// fewer L2 bytes per MMA than the 128 x 96 tile, but (#1133 r20) with nvfp4_a4_gemm_tma's
// per-element epilogue the residual kernel spills and the epilogue and the next tile's fill
// lie on the critical path. Here:
// - the epilogue goes through shared memory (as fp8_gemm8_pp's): per chunk of 16 tokens (one n8
//   tile of each warp column) every thread writes its scaled sums into a chunk buffer, token-major
//   with a 16-byte pad per token, and 16 threads of warp 0 send each token's rows (up to 768 bytes,
//   contiguous in dst) as one bulk copy. With the residual, the chunk's residual rows are bulk
//   copied into the buffer first (NBUF chunks ahead where the buffers allow), and each thread adds
//   its sums to them in place (__fadd_rn), so no thread holds residuals in registers;
// - PERSIST: one CTA per SM walks its tiles (blockIdx.x, + gridDim.x, ... in the grouped order of
//   GM row tiles, token tiles inside a group), and the stage ring runs on across tiles: the
//   epilogue's chunk buffers are the stage the tile's last k step used, every other stage is
//   refilled with the next tile's first steps while the epilogue runs, and the last stage after it;
// - KB: NVFP4 blocks (64 of k) a stage. KB 2 (3 stages in about the smem of 2 of KB 4) needs boxes
//   of 16-byte multiples: the weight box is 80 bytes (72 used) and the scale box 16 (8 used).
// Every output takes the same MMAs in the same k order and the same epilogue products as
// nvfp4_a4_gemm_tma's (the sum times the weight scale times the token's decode scale, then
// __fadd_rn with the residual): the same bits.

namespace a4w {
// The epilogue's chunk: CT tokens, PITCH bytes apart (192 F32 rows + 16 bytes of pad).
constexpr int CT = 16, PITCH = 192 * 4 + 16, CHUNK = CT * PITCH;
}  // namespace a4w

template <int KB_, int ST_> struct WideTile {
    static constexpr int MI = 3, NT = 8, KB = KB_, ST = ST_, WM = 4, WN = 2;
    static constexpr int BM = 16 * MI * WM, BN = 8 * NT * WN;  // 192 x 128
    static constexpr int LDW = KB == 4 ? 144 : 80;              // the weight box (a row's pitch)
    static constexpr int XB = KB * 32;                          // a token's codes (the swizzle span)
    static constexpr int SB = 16;                               // a token's scale box
    static constexpr int X_OFF = (BM * LDW + 1023) / 1024 * 1024;
    static constexpr int XS_OFF = X_OFF + BN * XB;
    static constexpr uint32_t TX = BM * LDW + BN * XB + BN * SB;  // bytes a stage receives
    static constexpr int STAGE = (XS_OFF + BN * SB + 1023) / 1024 * 1024;
    static constexpr int NBUF = STAGE / a4w::CHUNK < 3 ? STAGE / a4w::CHUNK : 3;
    static constexpr int SMEM = ST * STAGE + 1024;  // + 1 KiB to align the stages
    static_assert(KB == 4 || KB == 2, "4 or 2 blocks a stage");
    static_assert(NBUF >= 2, "two epilogue chunks fit a stage");
    static_assert(SMEM + 2048 <= 99 * 1024, "shared memory");
};

#ifdef LEGION_FP4_MMA
namespace a4w {
using legion_tma::saddr;
// `bytes` from global `src` into shared `dst`, completing on `bar`.
__device__ __forceinline__ void g2s(uint32_t dst, const void * src, uint32_t bytes, uint64_t * bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
                 "l"((uint64_t) src), "r"(bytes), "r"(saddr(bar))
                 : "memory");
}
// `bytes` from shared `src` to global `dst` (this thread's bulk group).
__device__ __forceinline__ void s2g(void * dst, uint32_t src, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"((uint64_t) dst), "r"(src),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void commit() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
// This thread's bulk groups but the newest N have read their shared memory.
template <int N> __device__ __forceinline__ void wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ bool bar_try(uint64_t * bar, uint32_t parity) {
    uint32_t done;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(saddr(bar)), "r"(parity)
        : "memory");
    return done != 0;
}
__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

// Tile t's row and token tiles: groups of gm row tiles, token tiles inside a group.
__device__ __forceinline__ void tile_of(int64_t t, int64_t tiles_m, int64_t tiles_n, int gm, int64_t & tm,
                                        int64_t & tn) {
    const int64_t in_group = gm * tiles_n, first_m = t / in_group * gm;
    const int64_t group = min(tiles_m - first_m, (int64_t) gm);
    tm = first_m + t % in_group % group;
    tn = t % in_group / group;
}

// The bulk epilogue of one tile from the chunk buffers at `buf` (every warp done reading that
// stage); rphase: the residual barriers' phases. Leaves warp 0's bulk groups read on return.
// DR (with RES): each thread reads its residuals from global memory (the L2, prefetched during
// the k loop) instead of the bulk copies into the chunk buffer.
template <class W, bool RES, bool DR>
__device__ __forceinline__ void epilogue(const legion_nvfp4_a4 & p, const float (&acc)[W::MI][W::NT][4], int64_t m0,
                                         int64_t n0, int n_tok, const float * xd, float sc, unsigned char * buf,
                                         uint64_t * rbar, uint32_t & rphase, int wm, int wn, int warp, int lane) {
    constexpr int NBUF = W::NBUF, NC = W::NT, MI = W::MI;
    constexpr bool SMR = RES && !DR;  // the residual through the chunk buffer
    const int g = lane / 4, t = lane % 4;
    const uint32_t rbytes = (uint32_t) min((int64_t) W::BM, p.m - m0) * 4;
    const bool io = warp == 0 && lane < CT;
    // io lane l's token of chunk c: warp column l / 8, its token c * 8 + l % 8.
    const int ltok = (lane / 8) * (W::BN / 2) + lane % 8;
    const uint32_t b0 = saddr(buf);
    auto res_load = [&](int c) {  // io lanes
        const int b = c % NBUF;
        if (lane == 0) {
            const int c0 = min(max(n_tok - c * 8, 0), 8), c1 = min(max(n_tok - W::BN / 2 - c * 8, 0), 8);
            legion_tma::bar_expect_tx(rbar + b, (uint32_t) (c0 + c1) * rbytes);
        }
        __syncwarp(0xffff);
        const int tk = ltok + c * 8;
        if (tk < n_tok) {
            g2s(b0 + b * CHUNK + lane * PITCH, p.res + (n0 + tk) * p.m + m0, rbytes, rbar + b);
        }
    };
    if constexpr (SMR) {
        if (io) {
#pragma unroll
            for (int c = 0; c < NBUF; c++) {
                res_load(c);
            }
        }
    }
    const int64_t rows_left = p.m - m0;
    // DR: chunk c's residuals into rd[c % 2], one chunk ahead of their adds.
    float rd[2][MI][4];
    auto res_direct = [&](int c) {
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = wm * 16 * MI + i * 16 + g + (e / 2) * 8;
                const int tok = wn * (W::BN / 2) + c * 8 + t * 2 + e % 2;
                rd[c % 2][i][e] = tok < n_tok && row < rows_left ? p.res[(n0 + tok) * p.m + m0 + row] : 0.0f;
            }
    };
    if constexpr (DR && RES) {
        res_direct(0);
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const int b = c % NBUF;
        const uint32_t base = b0 + b * CHUNK;
        if constexpr (SMR) {
            legion_tma::bar_wait(rbar + b, rphase >> b & 1);
            rphase ^= 1u << b;
        }
        if constexpr (DR && RES) {
            if (c + 1 < NC) {
                res_direct(c + 1);
            }
        }
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = wm * 16 * MI + i * 16 + g + (e / 2) * 8;
                const int lt = wn * 8 + t * 2 + e % 2;  // in the chunk
                const uint32_t a = base + lt * PITCH + row * 4;
                float v = acc[i][c][e] * sc * xd[wn * (W::BN / 2) + c * 8 + t * 2 + e % 2];
                if constexpr (SMR) {
                    float r;
                    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(r) : "r"(a) : "memory");
                    v = __fadd_rn(v, r);
                } else if constexpr (RES) {
                    v = __fadd_rn(v, rd[c % 2][i][e]);
                }
                asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
            }
        // The writes visible to the bulk copies (async proxy). Without the residual, the copies
        // of chunk c + 1 - NBUF have read the buffer chunk c + 1 rewrites.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (!SMR) {
            if (io) {
                wait_read<NBUF - 2>();
            }
        }
        __syncthreads();
        if (io) {
            const int tk = ltok + c * 8;
            if (tk < n_tok) {
                s2g((float *) p.dst + (n0 + tk) * p.m + m0, base + lane * PITCH, rbytes);
            }
            commit();
            if constexpr (SMR) {
                if (c + NBUF < NC) {
                    // This lane's copy has read its token's row: the residual of chunk c + NBUF
                    // goes there (each lane loads and stores the same rows).
                    wait_read<0>();
                    res_load(c + NBUF);
                }
            }
        }
    }
    if (io) {
        wait_read<0>();
    }
}
}  // namespace a4w
#endif

// PROBE (legion nvfp4-a4-bench only, plain GEMMs; outputs not the GEMM's): 1 the loads alone, 3
// no epilogue. PERSIST, GM, KB, ST: see the top of the file.
template <int KB, int ST, bool PERSIST, int GM, bool RES, int PROBE = 0, bool DR = false>
__global__ void __launch_bounds__(256, 1) nvfp4_a4_gemm_wide(const legion_nvfp4_a4 p, const __grid_constant__ a4t::Maps maps) {
#ifdef LEGION_FP4_MMA
    using T = WideTile<KB, ST>;
    using namespace legion_tma;
    constexpr int MI = T::MI, NT = T::NT, BM = T::BM, BN = T::BN, WN = T::WN, LDW = T::LDW, XB = T::XB;
    constexpr int STAGE = T::STAGE, X_OFF = T::X_OFF, XS_OFF = T::XS_OFF, WARPS = 8;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    // As nvfp4_a4_gemm_tma: the empty asm hides the declared alignment (which would fold the
    // offset to 0), and the base keeps the shared state space (LDS fragment reads).
    uint32_t sbase = saddr(smem_raw);
    asm volatile("" : "+r"(sbase));
    unsigned char * smem = smem_raw + ((1024 - (sbase & 1023)) & 1023);
    __shared__ uint64_t full[ST];
    __shared__ uint32_t released[ST];
    __shared__ uint64_t rbar[T::NBUF];
    __shared__ float x_dec[2][BN];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int wm = warp / WN, wn = warp % WN;
    const int64_t tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN, tiles = tiles_m * tiles_n;
    const int64_t first = blockIdx.x, stride = PERSIST ? gridDim.x : tiles;
    const int64_t mine = first < tiles ? (tiles - 1 - first) / stride + 1 : 0;
    const int64_t nblk = p.k / 64, steps = nblk / KB, total = mine * steps;
    const float sc = p.scale ? *p.scale : 1.0f;
    constexpr bool EPI = PROBE == 0;  // the epilogue takes the last step's stage

    // The bytes before k step ks's weights and scales in its stage (KB 2: 8 on odd steps).
    auto step_off = [](int64_t ks) { return KB == 2 ? (int) (ks & 1) * 8 : 0; };
    // Step gs of this CTA's sequence (tile gs / steps of its share, k step gs % steps) into stage
    // `stage` (one thread).
    auto issue = [&](int64_t gs, int stage) {
        int64_t tm, tn;
        a4w::tile_of(first + gs / steps * stride, tiles_m, tiles_n, GM, tm, tn);
        const int ks = (int) (gs % steps);
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        bar_expect_tx(bar, T::TX);
        // A box's first byte must be 16-byte aligned: KB 2's odd steps start 8 bytes early (the
        // weights at ks * 72, the scales at ks * 8), and the fragments skip them (step_off).
        tma_2d(s, &maps.w, ks * KB * 36 - step_off(ks), (int) (tm * BM), bar);
        tma_2d(s + X_OFF, &maps.xq, ks * KB * 32, (int) (tn * BN), bar);
        tma_2d(s + XS_OFF, &maps.xs, ks * KB * 4 - step_off(ks), (int) (tn * BN), bar);
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
#pragma unroll
        for (int b = 0; b < T::NBUF; b++) {
            bar_init(rbar + b, 1);
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
    auto fragments = [&](const unsigned char * st, int o, int blk, Frag & f) {
#pragma unroll
        for (int i = 0; i < MI; i++) {
            const unsigned char * r = st + o + (a_base(i) + g) * LDW + blk * 36 + 4 + t * 4;
            f.a[i][0] = *(const uint32_t *) r;
            f.a[i][1] = *(const uint32_t *) (r + 8 * LDW);
            f.a[i][2] = *(const uint32_t *) (r + 16);
            f.a[i][3] = *(const uint32_t *) (r + 8 * LDW + 16);
            f.sa[i] = *(const uint32_t *) (st + o + (a_base(i) + g + (t % 2) * 8) * LDW + blk * 36);
        }
#pragma unroll
        for (int jj = 0; jj < NT / 2; jj++) {
            const int row = b_tok + jj * 16;
            // The TMA's swizzle: 16-byte chunk c of token row r at c ^ (r % 8) (128-byte rows)
            // or c ^ (r / 2 % 4) (64-byte rows).
            const int chunk = KB == 4 ? (blk * 2 + b_hi) ^ (row % 8) : (blk * 2 + b_hi) ^ ((row >> 1) & 3);
            const uint32_t addr = saddr(st + X_OFF + row * XB + chunk * 16);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(f.b[2 * jj][0]), "=r"(f.b[2 * jj][1]), "=r"(f.b[2 * jj + 1][0]), "=r"(f.b[2 * jj + 1][1])
                         : "r"(addr));
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            f.sb[j] = *(const uint32_t *) (st + XS_OFF + o + (sb_tok + j * 8) * T::SB + blk * 4);
        }
    };

    uint32_t rphase = 0;
    for (int64_t it = 0; it < mine; it++) {
        int64_t tm, tn;
        a4w::tile_of(first + it * stride, tiles_m, tiles_n, GM, tm, tn);
        const int64_t m0 = tm * BM, n0 = tn * BN;
        const int n_tok = (int) min((int64_t) BN, p.n - n0);
        float * xd = x_dec[it & 1];
        for (int i = threadIdx.x; i < BN; i += 256) {
            xd[i] = i < n_tok ? p.xd[n0 + i] : 0.0f;
        }
        // nvfp4_a4_gemm_tma's L2 bulk prefetch of the weight rows (8 blocks a run, from the first
        // two token tiles only).
        constexpr int PF = 8 / KB;
        const bool pf_on = n0 < 2 * BN;
        auto prefetch = [&](int64_t step) {
            if (!pf_on) {
                return;
            }
            for (int r = threadIdx.x; r < BM; r += 256) {
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
            if (RES && ks == steps / 2 && (int) threadIdx.x < n_tok) {
                // The tile's residual rows into the L2 for the epilogue.
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p.res + (n0 + threadIdx.x) * p.m + m0),
                             "r"((uint32_t) min((int64_t) BM, p.m - m0) * 4));
            }
            bar_wait(full + stage, (uint32_t) (gs / ST) & 1);
            const unsigned char * st = smem + stage * STAGE;
            if (PROBE != 1) {
                Frag f[2];
                const int o = step_off(ks);
                fragments(st, o, 0, f[0]);
#pragma unroll
                for (int b = 0; b < KB; b++) {
                    if (b + 1 < KB) {
                        fragments(st, o, b + 1, f[(b + 1) % 2]);
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
            const bool deferred = EPI && ks == steps - 1;
            if (!deferred && gs + ST < total) {
                // As nvfp4_a4_gemm_tma: the last warp out refills the stage (maybe with the next
                // tile's step).
                __syncwarp();
                if (lane == 0) {
                    uint32_t prev;
                    asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev)
                                 : "r"(saddr(released + stage))
                                 : "memory");
                    if (prev % WARPS == WARPS - 1) {
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        issue(gs + ST, stage);
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (EPI) {
            const int64_t last = it * steps + steps - 1;
            const int ls = (int) (last % ST);
            // Every warp's reads of the last stage (and x_dec's writes) done before the chunks.
            __syncthreads();
            a4w::epilogue<T, RES, DR>(p, acc, m0, n0, n_tok, xd, sc, smem + ls * STAGE, rbar, rphase, wm, wn, warp, lane);
            __syncwarp();
            if (threadIdx.x == 0 && last + ST < total) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                issue(last + ST, ls);
            }
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
    if constexpr (EPI) {
        if (warp == 0 && lane < a4w::CT) {
            asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
        }
    }
#else
    __trap();  // launched only where nvfp4_a4_probe answers 1
#endif
}

// nvfp4_a4_gemm_wide for a dense plain or residual GEMM with F32 outputs (k % 256 == 0, m % 4 ==
// 0, 16-byte aligned dst and residual); -1 when it does not apply or a tensor map cannot be made
// (nothing launched). PROBE: plain GEMMs only. PERSIST: a grid of min(tiles, SMs).
template <int KB, int ST, bool PERSIST, int GM, int PROBE = 0, bool DR = false>
int dense_wide(const legion_nvfp4_a4 & p, cudaStream_t st) {
    using T = WideTile<KB, ST>;
    if (p.plan || p.w2 || p.act != 0 || p.k % 256 != 0 || p.n <= 0 || p.m <= 0 || p.m % 4 != 0 ||
        ((uintptr_t) p.dst & 15) != 0 || (p.res && ((uintptr_t) p.res & 15) != 0) || (PROBE != 0 && p.res)) {
        return -1;
    }
    a4t::Maps maps;
    const int64_t row_bytes = p.k / 64 * 36;
    if (!a4_tma_map(&maps.w, p.w, row_bytes, p.m, p.w_ld, T::LDW, T::BM, CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !a4_tma_map(&maps.xq, p.xq, p.xq_ld, p.n, p.xq_ld, T::XB, T::BN,
                    KB == 4 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B) ||
        !a4_tma_map(&maps.xs, p.xs, p.xs_ld, p.n, p.xs_ld, T::SB, T::BN, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return -1;
    }
    maps.w2 = maps.w;
    const int64_t tiles = (p.m + T::BM - 1) / T::BM * ((p.n + T::BN - 1) / T::BN);
    int64_t grid = tiles;
    if (PERSIST) {
        static int sms = 0;
        if (sms == 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        }
        grid = tiles < sms ? tiles : sms;
    }
    auto go = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        kernel<<<(unsigned) grid, 256, T::SMEM, st>>>(p, maps);
    };
    if (p.res) {
        go(nvfp4_a4_gemm_wide<KB, ST, PERSIST, GM, true, 0, DR>);
    } else {
        go(nvfp4_a4_gemm_wide<KB, ST, PERSIST, GM, false, PROBE>);
    }
    return status();
}

// dense_wide with probe `probe` (legion nvfp4-a4-bench; 0 the GEMM).
template <int KB, int ST, bool PERSIST, int GM>
int dense_wide_probe(const legion_nvfp4_a4 & p, cudaStream_t st, int probe) {
    switch (probe) {
        case 0: return dense_wide<KB, ST, PERSIST, GM, 0>(p, st);
        case 1: return dense_wide<KB, ST, PERSIST, GM, 1>(p, st);
        case 3: return dense_wide<KB, ST, PERSIST, GM, 3>(p, st);
        default: return -1;
    }
}

// The multicast probe (#1133 r21, legion nvfp4-a4-bench --mc-probe): a cluster of two CTAs, each
// with one mbarrier expecting one 128 x 96 box (12 KiB) of the activation codes. MODE 0 each CTA
// loads its own box (plain TMA, the control); 1 CTA 0 multicasts the box into both; 2 each CTA
// multicasts into itself only (mask 1 << rank); 3 CTA 0 loads its own box and multicasts one into
// CTA 1 only. Each CTA waits at most 200 ms on its barrier and writes (1 done / -1 not, the
// microseconds waited, a checksum of the bytes received) to dst[4 * rank ...] (not the GEMM's).
template <int MODE> __global__ void __launch_bounds__(128, 1) nvfp4_a4_mc_probe(const legion_nvfp4_a4 p, const __grid_constant__ a4t::Maps maps) {
#ifdef LEGION_FP4_MMA
    using namespace legion_tma;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    uint32_t sbase = saddr(smem_raw);
    asm volatile("" : "+r"(sbase));
    unsigned char * smem = smem_raw + ((1024 - (sbase & 1023)) & 1023);
    __shared__ uint64_t bar;
    constexpr uint32_t BYTES = 128 * 96;
    const uint32_t rank = cluster_rank();
    if (threadIdx.x == 0) {
        bar_init(&bar, 1);
        bar_init_fence();
    }
    cluster_sync();
    if (threadIdx.x == 0) {
        bar_expect_tx(&bar, BYTES);
    }
    cluster_sync();  // both CTAs expect their bytes before any copy is issued
    if (threadIdx.x == 0) {
        if (MODE == 0) {
            tma_2d(smem, &maps.xq, 0, 0, &bar);
        } else if (MODE == 1) {
            if (rank == 0) {
                tma_2d_mc(smem, &maps.xq, 0, 0, &bar, 3);
            }
        } else if (MODE == 2) {
            tma_2d_mc(smem, &maps.xq, 0, 0, &bar, (uint16_t) (1u << rank));
        } else {
            if (rank == 0) {
                tma_2d(smem, &maps.xq, 0, 0, &bar);
                tma_2d_mc(smem, &maps.xq, 0, 0, &bar, 2);
            }
        }
        const uint64_t t0 = a4w::now_ns();
        uint64_t t1 = t0;
        bool done = false;
        while (!done && t1 - t0 < 200000000ull) {
            done = a4w::bar_try(&bar, 0);
            t1 = a4w::now_ns();
        }
        uint32_t sum = 0;
        if (done) {
            for (uint32_t i = 0; i < BYTES / 4; i++) {
                sum = sum * 31u + ((const uint32_t *) smem)[i];
            }
        }
        float * d = (float *) p.dst + 4 * rank;
        d[0] = done ? 1.0f : -1.0f;
        d[1] = (float) ((t1 - t0) / 1000);
        d[2] = (float) (sum & 0xffffff);
        d[3] = (float) MODE;
    }
    cluster_sync();  // no CTA leaves while the other may still write into its shared memory
#else
    __trap();  // launched only where nvfp4_a4_probe answers 1
#endif
}

// nvfp4_a4_mc_probe<MODE> on a plain GEMM's activation codes (at least 96 tokens, m >= 8).
template <int MODE> int mc_probe(const legion_nvfp4_a4 & p, cudaStream_t st) {
    if (p.plan || p.w2 || p.res || p.act != 0 || p.n < 96 || p.m < 8 || p.k % 256 != 0) {
        return -1;
    }
    a4t::Maps maps;
    if (!a4_tma_map(&maps.xq, p.xq, p.xq_ld, p.n, p.xq_ld, 128, 96, CU_TENSOR_MAP_SWIZZLE_128B)) {
        return -1;
    }
    maps.w = maps.w2 = maps.xs = maps.xq;
    auto kernel = nvfp4_a4_mc_probe<MODE>;
    constexpr int SMEM = 128 * 96 + 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = st;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, p, maps);
    return status();
}
