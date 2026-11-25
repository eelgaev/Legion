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
}  // namespace a8t

// Shared memory of fp8_gemm8_tma: the stages, 1 KiB to align them, and a full mbarrier and a
// release count per stage. 4 warps (two CTAs per SM in its 100 KB): no alignment slack, as the
// dynamic shared memory of a kernel without static shared memory starts on a 1 KiB boundary
// (checked in the kernel).
template <class C> __host__ __device__ constexpr size_t gemm8_tma_smem() {
    return C::SMEM + (C::WARPS == 4 ? 0 : 1024) + 2 * C::STAGES * sizeof(uint64_t);
}

// legion_fp8_mm8's variant bit 9: the outputs go to mapped host memory (the TP send buffer,
// #1133), through gemm8_epilogue_staged with streaming stores.
constexpr int32_t GEMM8_STAGED = 0x200;
// legion_fp8_mm8's variant bit 10 (#1183): a GEMM without a residual into device memory (the
// shared expert's gate / up of --dense-fp8-prompt) through gemm8_epilogue_staged with plain
// stores (the next op reads the rows from L2): fp8_gemm8's tiles only. fp8_gemm8_tma ignores
// it: on the GB10 its 256-row tile was 0.99x with it (the gate / up at 12,288 tokens), where
// fp8_gemm8's 128-row tile was 1.18x (8,067 tokens). The same bits.
constexpr int32_t GEMM8_WIDE = 0x400;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// fp8_gemm8_tma's dynamic shared memory, aligned to 1 KiB.
template <class C> __device__ __forceinline__ unsigned char * gemm8_tma_smem_base() {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    if constexpr (C::WARPS == 4) {
        if (legion_tma::saddr(smem_raw) % 1024 != 0) {
            __trap();  // gemm8_tma_smem left no room to align
        }
    }
    return smem;
}

// fp8_gemm8_tma's K loop: the sums of the tile of TM rows from m0 and BN tokens from n0 into
// acc (the MMA layout of gemm8_stage), its stages in `smem` (gemm8_tma_smem_base).
template <bool RES, class C>
__device__ __forceinline__ void gemm8_tma_sums(const legion_fp8_mm8 & p, const a8t::Maps & maps,
                                               const a8t::Stagger & stagger, int64_t m0, int64_t n0,
                                               unsigned char * smem, int wm, int wn, int lane,
                                               float (&acc)[4][a8::BN / 8 / C::WN][4]) {
    using a8::BN;
    using namespace legion_tma;
    static_assert(C::SWZ && C::LD == C::BK && (C::BK == 64 || C::BK == 128), "TMA swizzle layout");
    constexpr int BK = C::BK, STAGES = C::STAGES;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM, WARPS = C::WARPS;
    constexpr size_t TSTAGE = C::TSTAGE;
    static_assert(TSTAGE % 1024 == 0 && (TM * BK) % 1024 == 0, "stages on swizzle boundaries");
    uint64_t * full = (uint64_t *) (smem + C::SMEM);
    uint32_t * released = (uint32_t *) (full + STAGES);
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

#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

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
}
#endif

// 4 warps (#1133 r11): two 128 x 128 CTAs per SM instead of one 256 x 128 one, so one CTA's
// epilogue (the fused residual's loads and the stores) runs while the other's MMAs do. Started
// together, the two drift into step and finish their tiles at the same time, and the residual
// loads of a whole wave then hit memory at once (on the GB10 that ran 0.94x the wide tile): so
// the first wave starts staggered (a8t::Stagger) over about one tile's time, and each tile's
// residual prefetch is spread over its last steps (gemm8_res_prefetch).
// JG (#1133 r25): 0 gemm8_epilogue, -1 gemm8_epilogue_staged with streaming stores (no
// residual; mapped host memory), else gemm8_epilogue_stream with JG token tiles at a time.
template <bool RES, class C, int JG = 0>
__global__ void __launch_bounds__(32 * C::WARPS, 8 / C::WARPS)
    fp8_gemm8_tma(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps, const a8t::Stagger stagger) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using a8::BN;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM;
    unsigned char * smem = gemm8_tma_smem_base<C>();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM, n0 = tn * BN;
    float acc[4][NJ][4];
    gemm8_tma_sums<RES, C>(p, maps, stagger, m0, n0, smem, wm, wn, lane, acc);
    if constexpr (JG < 0) {
        static_assert(!RES && C::WM == 4, "staged: no residual, 64 rows a warp");
        // Every warp's reads of the stages are done (and every stage's copy landed: each was
        // waited for), so their memory takes the staged outputs.
        __syncthreads();
        gemm8_epilogue_staged<C, true>(p, m0, n0, wm, wn, lane, acc, (float *) smem + warp * GEMM8_STAGE_FLOATS);
    } else if constexpr (JG == 0) {
        gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
    } else {
        gemm8_epilogue_stream<RES, C, JG>(p, m0, n0, wm, wn, lane, acc);
    }
#else
    __trap();  // launch_gemm8_tma checks the image is sm_90+ first
#endif
}
