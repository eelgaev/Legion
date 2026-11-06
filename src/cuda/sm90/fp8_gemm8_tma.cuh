// fp8_mm.cu, sm_90 and newer: fp8_gemm8 with its stages filled by the tensor memory accelerator
// (TMA) instead of per-thread cp.async copies (#1133).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm89/fp8_mm.cuh (a8::Cfg, gemm8_stage, gemm8_epilogue). The kernel body is built only where
// __CUDA_ARCH__ >= 900 and is empty elsewhere.
//
// fp8_gemm8 spends a large part of each K step issuing its copies: 12 cp.async of 16 bytes per
// thread for the 256 x 128 tile, each with its own address and bounds check, then a CTA barrier.
// Here one thread issues two TMA tile copies per stage (the TM weight rows and the BN tokens of
// BK bytes) that complete on the stage's "full" mbarrier, and each warp arrives on the stage's
// "empty" mbarrier when its ldmatrix reads of the stage are done, so warps no longer wait for
// each other at every step. The TMA writes each tile in its 64- or 128-byte swizzle (BK 64 or
// 128), which is exactly a8::Cfg's SWZ layout (16-byte chunk c of row r at c ^ (r / (128 / BK)
// % (BK / 16))) when a stage starts on a 1024-byte boundary, so the MMA code is fp8_gemm8's,
// and rows past m or tokens past n are filled with zeros as cp.async's zero-size copies do:
// every output takes the same MMAs in the same k order, the same bits.

namespace a8t {
// The two tensor maps of a launch: the weights [m rows of k bytes] and the activations [n
// tokens of k bytes], boxes of BK bytes x TM rows / BN tokens in the stage's swizzle.
struct Maps {
    CUtensorMap w, x;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ uint32_t saddr(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void bar_init(uint64_t * bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(saddr(bar)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void bar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(saddr(bar)) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t * bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra WAIT;\n"
        "}\n" ::"r"(saddr(bar)),
        "r"(parity)
        : "memory");
}
// A 2D tile (c0: byte along k, c1: row) of `map` into shared memory, completing on `bar`.
__device__ __forceinline__ void tma_2d(void * dst, const CUtensorMap * map, int c0, int c1, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            saddr(dst)),
        "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(saddr(bar))
        : "memory");
}
#endif
}  // namespace a8t

// Shared memory of fp8_gemm8_tma: the stages, 1 KiB to align them, and a full and an empty
// mbarrier per stage.
template <class C> __host__ __device__ constexpr size_t gemm8_tma_smem() {
    return C::SMEM + 1024 + 2 * C::STAGES * sizeof(uint64_t);
}

template <bool RES, class C>
__global__ void __launch_bounds__(256) fp8_gemm8_tma(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using a8::BN;
    using namespace a8t;
    static_assert(C::SWZ && C::LD == C::BK && (C::BK == 64 || C::BK == 128), "TMA swizzle layout");
    constexpr int WM = C::WM, BK = C::BK, STAGES = C::STAGES;
    constexpr int WN = 8 / WM, NJ = BN / 8 / WN, TM = C::TM, WARPS = 8;
    constexpr size_t TSTAGE = C::TSTAGE;
    static_assert(TSTAGE % 1024 == 0 && (TM * BK) % 1024 == 0, "stages on swizzle boundaries");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    uint64_t * full  = (uint64_t *) (smem + C::SMEM);
    uint64_t * empty = full + STAGES;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM, n0 = tn * BN;
    const int64_t steps = p.k / BK;

    // Step `step`'s weights and activations into stage `stage` (one thread).
    auto issue = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * TSTAGE;
        bar_expect_tx(full + stage, (uint32_t) TSTAGE);
        tma_2d(ws, &maps.w, (int) (step * BK), (int) m0, full + stage);
        tma_2d(ws + TM * BK, &maps.x, (int) (step * BK), (int) n0, full + stage);
    };
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];\n" ::"l"((uint64_t) &maps.w) : "memory");
        asm volatile("prefetch.tensormap [%0];\n" ::"l"((uint64_t) &maps.x) : "memory");
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            bar_init(full + s, 1);
            bar_init(empty + s, WARPS);
        }
        // The barriers' initialization visible to the TMA unit (async proxy).
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % STAGES);
        const uint32_t parity = (uint32_t) (step / STAGES) & 1;
        bar_wait(full + stage, parity);
        gemm8_res_prefetch<RES, C>(p, step, steps, m0, n0);
        const unsigned char * ws = smem + stage * TSTAGE;
        gemm8_stage<C>(ws, ws + TM * BK, wm, wn, lane, acc);
        if (step + STAGES < steps) {
            // This warp's reads of the stage are done; the last warp's arrival lets thread 0
            // refill it with step + STAGES.
            __syncwarp();
            if (lane == 0) {
                bar_arrive(empty + stage);
            }
            if (threadIdx.x == 0) {
                bar_wait(empty + stage, parity);
                issue(step + STAGES, stage);
            }
        }
    }
    gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
#endif
}
