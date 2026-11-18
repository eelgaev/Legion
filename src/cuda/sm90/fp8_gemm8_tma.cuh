// fp8_mm.cu, sm_90 and newer: fp8_gemm8 with its stages filled by the tensor memory accelerator
// (TMA) instead of per-thread cp.async copies (#1133).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm89/fp8_mm.cuh (a8::Cfg, gemm8_stage, gemm8_epilogue). The kernel body is built only where
// __CUDA_ARCH__ >= 900 and traps elsewhere.
//
// fp8_gemm8 spends a large part of each K step issuing its copies: 12 cp.async of 16 bytes per
// thread for the 256 x 128 tile, each with its own address and bounds check, then a CTA barrier.
// Here two TMA tile copies per stage (the TM weight rows and the BN tokens of BK bytes) complete
// on the stage's "full" mbarrier, and each warp counts itself out of the stage when its ldmatrix
// reads of it are done; the last warp out refills it, so warps no longer wait for each other at
// every step (#1133 r5: the refill was thread 0's, which waited for the slowest warp on an
// "empty" mbarrier). The TMA writes each tile in its 64- or 128-byte swizzle (BK 64 or
// 128), which is exactly a8::Cfg's SWZ layout (16-byte chunk c of row r at c ^ (r / (128 / BK)
// % (BK / 16))) when a stage starts on a 1024-byte boundary, so the MMA code is fp8_gemm8's,
// and rows past m or tokens past n are filled with zeros as cp.async's zero-size copies do:
// every output takes the same MMAs in the same k order, the same bits.

#include "tma.cuh"

namespace a8t {
// The two tensor maps of a launch: the weights [m rows of k bytes] and the activations [n
// tokens of k bytes], boxes of BK bytes x TM rows / BN tokens in the stage's swizzle.
struct Maps {
    CUtensorMap w, x;
};
// The first wave's staggered start: CTA i < ctas (in launch order) waits i * ns nanoseconds
// before its first copy. ctas 0: none.
struct Stagger {
    int ctas = 0;
    uint32_t ns = 0;
};

// legion_fp8_mm8::variant (#1133 r15). The low byte: V_DEFAULT (0 or 1) the kernel as it was;
// V_EPI the epilogue's input and weight scales loaded into registers before its first store
// (a8_scale loads them per element, between stores that may alias them as far as the compiler
// knows, so every store waits out a load); V_EARLY each warp releases a stage once its last
// fragments are loaded, before the last 32-byte k step's MMAs, so the refill starts that much
// earlier; V_BOTH both; V_WIDE V_BOTH on the 8-warp wide tile only, the 4-warp residual tile
// as it was (there V_EPI spills: 255 registers and 200 bytes of stack, 3-6% slower). Every
// output takes the same MMAs in the same k order and the same scale products: the same bits.
// Bits 8-11, probes (the outputs are not the GEMM's): P_NO_STORE
// no epilogue (one store when a sum of the accumulators hits a value it never does), P_NO_MMA
// no MMAs, P_NO_LDSM fragments from a constant instead of the stage, P_NO_REFILL only the
// first STAGES steps' copies, later steps reread their stage without waiting.
constexpr int V_DEFAULT = 1, V_EPI = 2, V_EARLY = 3, V_BOTH = 4, V_WIDE = 5;
constexpr int P_NO_STORE = 0x100, P_NO_MMA = 0x200, P_NO_LDSM = 0x400, P_NO_REFILL = 0x800;
}  // namespace a8t

// gemm8_stage (sm89/fp8_mm.cuh) with a8t's V_EARLY and probes: `at_last()` runs after the last
// fragments' loads, before that k step's MMAs, under V_EARLY.
template <class C, int NJ, int V, class F>
__device__ __forceinline__ void gemm8_stage_v(const unsigned char * ws, const unsigned char * xs, int wm, int wn,
                                              int lane, float (&acc)[4][NJ][4], F && at_last) {
    constexpr int VV = V & 0xff;
    constexpr bool EARLY = VV == a8t::V_EARLY || VV == a8t::V_BOTH;
#pragma unroll
    for (int kk = 0; kk < C::BK; kk += 32) {
        uint32_t a[4][4], b[NJ][2];
        if constexpr ((V & a8t::P_NO_LDSM) != 0) {
            // E4M3 1.0 in every byte.
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int q = 0; q < 4; q++) a[i][q] = 0x38383838u;
#pragma unroll
            for (int j = 0; j < NJ; j++) b[j][0] = b[j][1] = 0x38383838u;
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned char * src = ws + C::off(wm * 64 + i * 16 + lane % 16, kk / 16 + lane / 16);
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                    : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < NJ / 2; j++) {
                const int tok = wn * NJ * 8 + j * 16 + lane % 8 + (lane / 16) * 8;
                const unsigned char * src = xs + C::off(tok, kk / 16 + (lane / 8) % 2);
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b[2 * j][0]), "=r"(b[2 * j][1]), "=r"(b[2 * j + 1][0]), "=r"(b[2 * j + 1][1])
                    : "r"(addr));
            }
        }
        if constexpr (EARLY) {
            if (kk == C::BK - 32) {
                at_last();
            }
        }
        if constexpr ((V & a8t::P_NO_MMA) == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
                    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]));
        }
    }
}

// gemm8_epilogue (sm89/fp8_mm.cuh) under a8t's V_EPI: the same products and adds, with the
// thread's 2 * NJ token scales and 8 row scales loaded before the first store.
template <bool RES, class C, int NJ>
__device__ __forceinline__ void gemm8_epilogue_hoist(const legion_fp8_mm8 & p, int64_t m0, int64_t n0, int wm,
                                                     int wn, int lane, const float (&acc)[4][NJ][4]) {
    const float s = a8_static(p);
    const bool each = p.scale_stride != 0 || p.wscale_stride != 0;
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
            ws[i][h] = each && p.wscale && row < p.m ? p.wscale[row * p.wscale_stride] : 0.0f;
        }
    float rv[4][NJ][4];
    auto load_res = [&](int i) {
        const int64_t r0 = m0 + wm * 64 + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const int64_t t0 = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
                rv[i][j][e] = row < p.m && tok < p.n ? p.res[tok * p.m + row] : 0.0f;
            }
        }
    };
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if constexpr (RES) {
            load_res(i);
        }
        const int64_t r0 = m0 + wm * 64 + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const int64_t t0 = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
                if (row < p.m && tok < p.n) {
                    const float f = !each ? s : p.wscale ? xs[j][e % 2] * ws[i][e / 2] : xs[j][e % 2];
                    const float v = acc[i][j][e] * f;
                    p.dst[tok * p.m + row] = RES ? __fadd_rn(v, rv[i][j][e]) : v;
                }
            }
        }
    }
}

// Shared memory of fp8_gemm8_tma: the stages, 1 KiB to align them, and a full mbarrier and a
// release count per stage. 4 warps (two CTAs per SM in its 100 KB): no alignment slack, as the
// dynamic shared memory of a kernel without static shared memory starts on a 1 KiB boundary
// (checked in the kernel).
template <class C> __host__ __device__ constexpr size_t gemm8_tma_smem() {
    return C::SMEM + (C::WARPS == 4 ? 0 : 1024) + 2 * C::STAGES * sizeof(uint64_t);
}

// 4 warps (#1133 r11): two 128 x 128 CTAs per SM instead of one 256 x 128 one, so one CTA's
// epilogue (the fused residual's loads and the stores) runs while the other's MMAs do. Started
// together, the two drift into step and finish their tiles at the same time, and the residual
// loads of a whole wave then hit memory at once (on the GB10 that ran 0.94x the wide tile): so
// the first wave starts staggered (a8t::Stagger) over about one tile's time, and each tile's
// residual prefetch is spread over its last steps (gemm8_res_prefetch).
// V: a8t's variant (0 the default).
template <bool RES, class C, int V = 0>
__global__ void __launch_bounds__(32 * C::WARPS, 8 / C::WARPS)
    fp8_gemm8_tma(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps, const a8t::Stagger stagger) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using a8::BN;
    using namespace legion_tma;
    static_assert(C::SWZ && C::LD == C::BK && (C::BK == 64 || C::BK == 128), "TMA swizzle layout");
    constexpr int WM = C::WM, BK = C::BK, STAGES = C::STAGES;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM, WARPS = C::WARPS;
    constexpr size_t TSTAGE = C::TSTAGE;
    static_assert(TSTAGE % 1024 == 0 && (TM * BK) % 1024 == 0, "stages on swizzle boundaries");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    if constexpr (WARPS == 4) {
        if (saddr(smem_raw) % 1024 != 0) {
            __trap();  // gemm8_tma_smem left no room to align
        }
    }
    uint64_t * full = (uint64_t *) (smem + C::SMEM);
    uint32_t * released = (uint32_t *) (full + STAGES);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM, n0 = tn * BN;
    const int64_t steps = p.k / BK;

    const int64_t lin = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (lin < stagger.ctas) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        const uint64_t until = t + (uint64_t) lin * stagger.ns;
        while (t < until) {
            __nanosleep(256);
            asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        }
    }

    // Step `step`'s weights and activations into stage `stage` (one thread).
    auto issue = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * TSTAGE;
        bar_expect_tx(full + stage, (uint32_t) TSTAGE);
        tma_2d(ws, &maps.w, (int) (step * BK), (int) m0, full + stage);
        tma_2d(ws + TM * BK, &maps.x, (int) (step * BK), (int) n0, full + stage);
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
            if (s < steps) {
                issue(s, s);
            }
        }
    }

    float acc[4][NJ][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    constexpr int VV = V & 0xff;
    if constexpr (V != 0 && VV != a8t::V_DEFAULT && VV != a8t::V_EPI) {
        // a8t's mainloop variants and probes.
        for (int64_t step = 0; step < steps; step++) {
            const int stage = (int) (step % STAGES);
            const uint32_t parity = (uint32_t) (step / STAGES) & 1;
            if ((V & a8t::P_NO_REFILL) == 0 || step < STAGES) {
                bar_wait(full + stage, parity);
            }
            gemm8_res_prefetch<RES, C>(p, step, steps, m0, n0);
            const unsigned char * ws = smem + stage * TSTAGE;
            auto release = [&]() {
                if ((V & a8t::P_NO_REFILL) == 0 && step + STAGES < steps) {
                    __syncwarp();
                    if (lane == 0) {
                        uint32_t prev;
                        asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                     : "=r"(prev)
                                     : "r"(saddr(released + stage))
                                     : "memory");
                        if (prev % WARPS == WARPS - 1) {
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            issue(step + STAGES, stage);
                        }
                    }
                    __syncwarp();
                }
            };
            gemm8_stage_v<C, NJ, V>(ws, ws + TM * BK, wm, wn, lane, acc, release);
            if constexpr (VV != a8t::V_EARLY && VV != a8t::V_BOTH) {
                release();
            }
        }
        if constexpr ((V & a8t::P_NO_STORE) != 0) {
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
        } else if constexpr (VV == a8t::V_BOTH) {
            gemm8_epilogue_hoist<RES, C, NJ>(p, m0, n0, wm, wn, lane, acc);
        } else {
            gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
        }
        return;
    }
    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % STAGES);
        const uint32_t parity = (uint32_t) (step / STAGES) & 1;
        bar_wait(full + stage, parity);
        gemm8_res_prefetch<RES, C>(p, step, steps, m0, n0);
        const unsigned char * ws = smem + stage * TSTAGE;
        gemm8_stage<C>(ws, ws + TM * BK, wm, wn, lane, acc);
        if (step + STAGES < steps) {
            // This warp's reads of the stage are done. The acq_rel count orders every warp's
            // reads before the last arrival, which refills the stage with step + STAGES (after
            // a proxy fence: the TMA writes through the async proxy).
            __syncwarp();
            if (lane == 0) {
                uint32_t prev;
                asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                             : "=r"(prev)
                             : "r"(saddr(released + stage))
                             : "memory");
                if (prev % WARPS == WARPS - 1) {
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    issue(step + STAGES, stage);
                }
            }
            __syncwarp();
        }
    }
    if constexpr (VV == a8t::V_EPI) {
        gemm8_epilogue_hoist<RES, C, NJ>(p, m0, n0, wm, wn, lane, acc);
    } else {
        gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
    }
#else
    __trap();  // launch_gemm8_tma checks the image is sm_90+ first
#endif
}
