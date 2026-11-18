// fp8_mm.cu, sm_90 and newer: the wide W8A8 prefill tile (256 x 128, 8 warps of 64 x 64, two
// TMA stages of 128-byte k steps) as a persistent kernel (#1133 round 16).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm90/fp8_gemm8_tma.cuh (a8t::Maps, the TMA helpers, gemm8_stage, gemm8_epilogue). The kernel
// body is built only where __CUDA_ARCH__ >= 900 and traps elsewhere.
//
// fp8_gemm8_tma runs one tile per CTA, one CTA per SM: every SM loads its first two stages, runs
// the k steps, then writes the tile's 128 KiB of F32 outputs with the tensor pipe idle, and as all
// SMs started together they all store at the same time (#1133 r15: without the stores the wide
// tile is 1.19-1.38x faster). Here one CTA per SM walks its share of the tiles (tile
// blockIdx.x, + gridDim.x, ... in fp8_gemm8_tma's launch order), and the stage ring runs on
// across tiles: the refill after a tile's second-to-last step already loads the next tile's
// first step, so the next mainloop starts on a full stage.
//
// The epilogue (V_BULK) stages the outputs in the stage the last k step used instead of storing
// each element: per chunk of 16 tokens (one n8 tile of each warp column) every thread writes its
// 16 scaled sums into shared memory, token-major with a 16-byte pad per token (the 8 rows x 4
// token pairs of a warp's store hit 32 banks), and 16 threads then send each token's 256 rows (1
// KiB, contiguous in dst) as one bulk copy (cp.async.bulk global <- shared). Two chunk buffers
// alternate; a buffer is rewritten only after the copies of the chunk before have read it. The
// stage's refill (the next tile's second step) waits for the last chunk's reads. V_STG keeps
// gemm8_epilogue's per-element stores (persistence alone, for A/B).
//
// The default dispatch (fp8_mm.cu's gemm8_pp_auto) takes V_BULK for the residual GEMMs with a
// long k only: without the residual it ties fp8_gemm8_tma, as the epilogue's shared-memory
// writes and barriers still leave the tensor pipe idle (no-store probe 1.20-1.37x).
//
// The bits: every output takes gemm8_stage's MMAs in the same k order and the same products as
// gemm8_epilogue (the token scale times the row scale, then the sum times that; with a residual
// the same __fadd_rn), so dst holds fp8_gemm8_tma's bytes.

namespace a8p {
// legion_fp8_mm8::variant (#1133 r16). The low byte: V_AUTO (0) the default dispatch, V_MAIN
// the kernels as before round 16, V_STG / V_BULK the persistent wide tile with gemm8_epilogue's
// stores / the staged bulk copies. Bits 8 and 11, probes of the persistent kernel (outputs not
// the GEMM's): P_NO_STORE no epilogue, P_NO_REFILL only the first STAGES steps' copies, later
// steps reread their stage without waiting.
constexpr int V_AUTO = 0, V_MAIN = 1, V_STG = 2, V_BULK = 3;
constexpr int P_NO_STORE = 0x100, P_NO_REFILL = 0x800;
// The bulk epilogue's chunk: CT tokens, each PITCH bytes apart (256 F32 rows + 16 bytes).
constexpr int CT = 16, PITCH = 256 * 4 + 16;
}  // namespace a8p

// Shared memory of fp8_gemm8_pp: fp8_gemm8_tma's (the stages, 1 KiB to align them, a full
// mbarrier and a release count per stage).
template <class C> __host__ __device__ constexpr size_t gemm8_pp_smem() {
    return C::SMEM + 1024 + 2 * C::STAGES * sizeof(uint64_t);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Tile t's row and token tiles, in grouped_tile's order (sm80/fp8_mm.cuh) for pid t.
__device__ __forceinline__ void gemm8_pp_tile(int64_t t, int64_t x_bytes, int64_t tiles_n, int64_t tiles_m,
                                              int64_t & tm, int64_t & tn) {
    constexpr int64_t GROUPED_X_BYTES = 16 << 20;
    if (x_bytes <= GROUPED_X_BYTES) {
        tm = t / tiles_n;
        tn = t % tiles_n;
        return;
    }
    constexpr int64_t GROUP_M = 8;
    const int64_t in_group = GROUP_M * tiles_n;
    const int64_t first_m = t / in_group * GROUP_M;
    const int64_t group = tiles_m - first_m < GROUP_M ? tiles_m - first_m : GROUP_M;
    tm = first_m + t % in_group % group;
    tn = t % in_group / group;
}

// The bulk epilogue of one 256 x 128 tile from the stage at `buf` (all warps done reading it).
// Leaves this thread's bulk groups read (the stage free again) on return.
template <bool RES, class C, int NJ>
__device__ __forceinline__ void gemm8_pp_epilogue(const legion_fp8_mm8 & p, int64_t m0, int64_t n0, int wm, int wn,
                                                  int warp, int lane, const float (&acc)[4][NJ][4],
                                                  unsigned char * buf) {
    using legion_tma::saddr;
    static_assert(NJ == 8 && C::WN == 2, "one n8 tile of each warp column per chunk");
    const float s = a8_static(p);
    const bool each = p.scale_stride != 0 || p.wscale_stride != 0;
    // The thread's token scales (per chunk c: tokens c * 8 + (lane % 4) * 2 + {0, 1} of its warp
    // column) and row scales, loaded before the first shared-memory write.
    float xs[NJ][2], ws[4][2];
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int64_t tok = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2 + e;
            xs[j][e] = each && tok < p.n ? p.scale[tok * p.scale_stride] : 0.0f;
        }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t row = m0 + wm * 64 + i * 16 + lane / 4 + h * 8;
            ws[i][h] = each && p.wscale ? p.wscale[row * p.wscale_stride] : 0.0f;
        }
    // RES: chunk c's residuals, loaded during chunk c - 1.
    float rv[2][4][4];
    auto load_res = [&](int j, float (&r)[4][4]) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = m0 + wm * 64 + i * 16 + lane / 4 + (e / 2) * 8;
                const int64_t tok = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2 + e % 2;
                r[i][e] = tok < p.n ? p.res[tok * p.m + row] : 0.0f;
            }
    };
    if constexpr (RES) {
        load_res(0, rv[0]);
    }
#pragma unroll
    for (int c = 0; c < NJ; c++) {
        if constexpr (RES) {
            if (c + 1 < NJ) {
                load_res(c + 1, rv[(c + 1) % 2]);
            }
        }
        const uint32_t b = saddr(buf + (c % 2) * a8p::CT * a8p::PITCH);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = wm * 64 + i * 16 + lane / 4 + (e / 2) * 8;
                const int tok = wn * 8 + (lane % 4) * 2 + e % 2;  // in the chunk
                const float f = !each ? s : p.wscale ? xs[c][e % 2] * ws[i][e / 2] : xs[c][e % 2];
                float v = acc[i][c][e] * f;
                if constexpr (RES) {
                    v = __fadd_rn(v, rv[c % 2][i][e]);
                }
                asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(b + tok * a8p::PITCH + row * 4), "f"(v) : "memory");
            }
        // The writes visible to the bulk copies (async proxy); warp 0's copies of the chunk
        // before have read the other buffer, which the next chunk rewrites.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if (warp == 0 && lane < a8p::CT) {
            asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        }
        __syncthreads();
        if (warp == 0 && lane < a8p::CT) {
            // Chunk token `lane`: warp column lane / 8, its token c * 8 + lane % 8.
            const int64_t tok = n0 + (lane / 8) * NJ * 8 + c * 8 + lane % 8;
            if (tok < p.n) {
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(
                                 p.dst + tok * p.m + m0),
                             "r"(b + lane * a8p::PITCH), "r"(256 * 4)
                             : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        }
    }
    if (warp == 0 && lane < a8p::CT) {
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
    }
}
#endif

// V: a8p's variant (V_STG or V_BULK, with probe bits). Grid: one CTA per SM (at most the tiles).
template <bool RES, class C, int V>
__global__ void __launch_bounds__(256, 1)
    fp8_gemm8_pp(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps, const a8t::Stagger stagger) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using a8::BN;
    using namespace legion_tma;
    static_assert(C::WARPS == 8 && C::WM == 4 && C::SWZ && C::LD == C::BK && C::BK == 128, "the wide TMA tile");
    constexpr int BK = C::BK, STAGES = C::STAGES, WN = C::WN, NJ = BN / 8 / WN, TM = C::TM, WARPS = C::WARPS;
    constexpr size_t TSTAGE = C::TSTAGE;
    constexpr bool BULK = (V & 0xff) == a8p::V_BULK;
    constexpr bool NO_STORE = (V & a8p::P_NO_STORE) != 0, NO_REFILL = (V & a8p::P_NO_REFILL) != 0;
    static_assert(BULK || (V & 0xff) == a8p::V_STG, "a persistent variant");
    static_assert(2 * a8p::CT * a8p::PITCH <= TSTAGE, "two epilogue chunks fit a stage");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    uint64_t * full = (uint64_t *) (smem + C::SMEM);
    uint32_t * released = (uint32_t *) (full + STAGES);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    const int64_t steps = p.k / BK;
    const int64_t tiles_n = (p.n + BN - 1) / BN, tiles_m = (p.m + TM - 1) / TM, tiles = tiles_n * tiles_m;
    const int64_t x_bytes = p.n * p.k;
    const int64_t first = blockIdx.x, stride = gridDim.x;
    const int64_t mine = first < tiles ? (tiles - 1 - first) / stride + 1 : 0;
    const int64_t total = mine * steps;  // this CTA's k steps over all its tiles

    if ((int64_t) blockIdx.x < stagger.ctas) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        const uint64_t until = t + (uint64_t) blockIdx.x * stagger.ns;
        while (t < until) {
            __nanosleep(256);
            asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        }
    }

    // Step g of this CTA's sequence (tile g / steps of its share, k step g % steps) into
    // stage `stage` (one thread).
    auto issue = [&](int64_t g, int stage) {
        int64_t tm, tn;
        gemm8_pp_tile(first + g / steps * stride, x_bytes, tiles_n, tiles_m, tm, tn);
        const int ks = (int) (g % steps);
        unsigned char * ws = smem + stage * TSTAGE;
        bar_expect_tx(full + stage, (uint32_t) TSTAGE);
        tma_2d(ws, &maps.w, ks * BK, (int) (tm * TM), full + stage);
        tma_2d(ws + TM * BK, &maps.x, ks * BK, (int) (tn * BN), full + stage);
    };
    if (threadIdx.x == 0) {
        prefetch_map(&maps.w);
        prefetch_map(&maps.x);
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            bar_init(full + s, 1);
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            if (s < total) {
                issue(s, s);
            }
        }
    }

    for (int64_t it = 0; it < mine; it++) {
        int64_t tm, tn;
        gemm8_pp_tile(first + it * stride, x_bytes, tiles_n, tiles_m, tm, tn);
        const int64_t m0 = tm * TM, n0 = tn * BN;
        float acc[4][NJ][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

        for (int64_t ks = 0; ks < steps; ks++) {
            const int64_t g = it * steps + ks;
            const int stage = (int) (g % STAGES);
            const uint32_t parity = (uint32_t) (g / STAGES) & 1;
            if (!NO_REFILL || g < STAGES) {
                bar_wait(full + stage, parity);
            }
            gemm8_res_prefetch<RES, C>(p, ks, steps, m0, n0);
            const unsigned char * ws = smem + stage * TSTAGE;
            gemm8_stage<C>(ws, ws + TM * BK, wm, wn, lane, acc);
            // The bulk epilogue writes the last step's stage: its refill waits for that.
            const bool deferred = BULK && ks == steps - 1;
            if (!NO_REFILL && !deferred && g + STAGES < total) {
                // As fp8_gemm8_tma: the last warp out refills the stage, here maybe with the
                // next tile's step.
                __syncwarp();
                if (lane == 0) {
                    uint32_t prev;
                    asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev)
                                 : "r"(saddr(released + stage))
                                 : "memory");
                    if (prev % WARPS == WARPS - 1) {
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        issue(g + STAGES, stage);
                    }
                }
                __syncwarp();
            }
        }
        const int64_t last = it * steps + steps - 1;
        if constexpr (NO_STORE) {
            float t = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) t += acc[i][j][e];
            if (t == 1.25e-37f) {
                p.dst[threadIdx.x] = t;
            }
        } else if constexpr (!BULK) {
            gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
        }
        if constexpr (BULK) {
            // Every warp's reads of the last stage are done before the chunks overwrite it.
            __syncthreads();
            if constexpr (!NO_STORE) {
                gemm8_pp_epilogue<RES, C>(p, m0, n0, wm, wn, warp, lane, acc, smem + (last % STAGES) * TSTAGE);
                __syncwarp();
            }
            if (!NO_REFILL && threadIdx.x == 0 && last + STAGES < total) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                issue(last + STAGES, (int) (last % STAGES));
            }
        }
    }
    if constexpr (BULK && !NO_STORE) {
        if (warp == 0 && lane < a8p::CT) {
            asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
        }
    }
#else
    __trap();  // launch_gemm8_tma checks the image is sm_90+ first
#endif
}
