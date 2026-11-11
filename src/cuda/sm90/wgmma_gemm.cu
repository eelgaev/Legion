// Hopper (sm_90a) prompt GEMMs on warpgroup MMA (wgmma) fed by the tensor memory accelerator
// (TMA), #1215. UNTESTED ON HARDWARE: written and compiled without an H100/H200/GH200; the Rust
// side (src/model/wgmma_gemm.rs) runs them only under `legion serve --hopper-gemm` and only on a
// device of compute capability 9.0 whose binary holds the sm_90a bodies.
//
//   dst[t * m + r] = scale(r, t) * sum_k W[r][k] * X[t][k]   (+ res[t * m + r])
//
// W: m rows of k values, X: n tokens of k values, both K-major (k contiguous), BF16 or FP8 E4M3;
// dst F32 [m, n] (rows fastest), as fp8_mm.cu's legion_fp8_mm8 (whose scale rules and residual
// add the epilogue copies: a8_static / a8_scale, __fadd_rn).
//
// One CTA = three warpgroups (384 threads): warpgroup 0 is the producer (one thread issues the
// TMA copies, the rest exit after giving their registers back with setmaxnreg), warpgroups 1
// and 2 are the consumers, 64 weight rows each, all BN tokens of the tile. A ring of STAGES
// shared-memory stages, each a BM x 128-byte weight tile and a BN x 128-byte token tile in the
// TMA's 128-byte swizzle, with a "full" mbarrier (the TMA's transaction count) and an "empty"
// mbarrier (one arrival per consumer warp) per stage. wgmma reads both operands straight from
// shared memory through matrix descriptors (K-major, 128-byte swizzle; the descriptor's fixed
// bits come from the Rust side, `wgmma_gemm::desc_hi`, so the CPU tests check them).
//
// - BF16 (KIND_BF16): 128 x 256 tiles, K steps of 64 values (128 bytes), four stages (48 KiB
//   each); per stage 4 K slices x 2 halves of m64n128k16; F32 accumulators straight in the
//   wgmma registers (128 a thread), the previous stage released after the next one is issued
//   (wgmma.wait_group 1).
// - E4M3 (KIND_E4M3): 128 x 128 tiles, K steps of 128 bytes, six stages (32 KiB each); per stage
//   4 x m64n128k32. Hopper's FP8 wgmma adds its products in a narrower accumulator than F32
//   (DeepSeek-V3 report, section 3.3.2; DeepGEMM promotes the same way), so each stage's 128-deep
//   sum lands in its own registers (scale-d 0 on the first slice) and is added to the F32 sums
//   after wgmma.wait_group 0.
//
// Every output's sum runs in another order than fp8_gemm8's (mma.sync) or cuBLAS's: other bits,
// so the kernels stay an opt-in until they are measured on Hopper.
//
// The wgmma / setmaxnreg bodies exist only where __CUDA_ARCH_FEAT_SM90_ALL is defined (an sm_90a
// target); every other target, sm_90 without the "a" included, compiles a __trap(). The host
// cannot see the difference in cudaFuncAttributes' versions, so legion_wgmma_gemm_probe has 16
// bytes of static shared memory only in its sm_90a body: the Rust side reads
// cudaFuncAttributes::sharedSizeBytes of it (no launch, so safe inside a graph capture).
//
// Patterns studied: CUTLASS's sm90 warp-specialized GEMM (BSD-3-Clause, NVIDIA), ThunderKittens
// (MIT) and DeepGEMM (MIT) for the descriptor, swizzle and promotion; no code copied verbatim.
// Kernels only: no host code here, Rust launches them from legion_wgmma_gemm_kernels.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_wgmma_gemm {
    float * dst;           // [m, n] F32, rows fastest
    const float * res;     // the residual (dst's layout) added with __fadd_rn, or null
    const float * scale;   // the input scale: token t's is scale[t * scale_stride]; null: 1
    const float * wscale;  // the weight scale: row r's is wscale[r * wscale_stride]; null: 1
    int64_t k, m, n;
    int64_t scale_stride, wscale_stride;
    uint64_t desc_hi;      // the shared-memory matrix descriptor's fixed bits (wgmma_gemm::desc_hi)
    int32_t tiles_m, tiles_n;
};

// The two tensor maps: weights (m rows of k values) and activations (n tokens of k values), boxes
// of 128 bytes x BM rows / BN tokens in the 128-byte swizzle.
struct legion_wgmma_maps {
    CUtensorMap w, x;
};

}  // extern "C"

namespace {

constexpr int KIND_BF16 = 0;
constexpr int KIND_E4M3 = 1;

constexpr int THREADS = 384;    // producer + two consumer warpgroups
constexpr int BM      = 128;    // weight rows per CTA (64 per consumer warpgroup)
constexpr int ROW     = 128;    // bytes of one K step of one row (the swizzle span)
constexpr int SLICE   = 32;     // bytes of K one wgmma takes (k16 BF16, k32 E4M3)
constexpr int GROUP_M = 8;      // row tiles per group of the tile order

template <int KIND> struct Cfg;
template <> struct Cfg<KIND_BF16> {
    static constexpr int BN = 256, STAGES = 4, BK = 64;  // BK in values
};
template <> struct Cfg<KIND_E4M3> {
    static constexpr int BN = 128, STAGES = 6, BK = 128;
};

template <int KIND> __host__ __device__ constexpr int stage_bytes() {
    return (BM + Cfg<KIND>::BN) * ROW;
}
// Dynamic shared memory: the stages, 1 KiB to align them, two mbarriers per stage.
template <int KIND> __host__ __device__ constexpr int smem_bytes() {
    return Cfg<KIND>::STAGES * stage_bytes<KIND>() + 1024 + 2 * Cfg<KIND>::STAGES * 8;
}
static_assert(smem_bytes<KIND_BF16>() == 197696, "wgmma_gemm::smem_bytes(Bf16)");
static_assert(smem_bytes<KIND_E4M3>() == 197728, "wgmma_gemm::smem_bytes(E4m3)");

// Tile `pid` of the launch as (row tile, token tile): groups of GROUP_M row tiles walked token
// tile by token tile, so a wave's CTAs share weight rows in L2 (wgmma_gemm::tile_of).
__device__ __forceinline__ void tile_of(int pid, int tiles_m, int tiles_n, int & tm, int & tn) {
    const int in_group = GROUP_M * tiles_n;
    const int first_m  = pid / in_group * GROUP_M;
    const int group    = min(tiles_m - first_m, GROUP_M);
    tm = first_m + pid % in_group % group;
    tn = pid % in_group / group;
}

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define LEGION_WGMMA 1
#else
#define LEGION_WGMMA 0
#endif

#if LEGION_WGMMA
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
__device__ __forceinline__ void tma_2d(void * dst, const CUtensorMap * map, int c0, int c1, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            saddr(dst)),
        "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(saddr(bar))
        : "memory");
}

// The matrix descriptor of the 128-byte-swizzled K-major tile at shared address `a` (1024-byte
// aligned rows groups; +32 bytes per K slice inside the swizzle span, as CUTLASS and DeepGEMM
// advance it): the start address in bits 0-13 (16-byte units), the fixed bits from the host.
__device__ __forceinline__ uint64_t desc(uint64_t hi, uint32_t a) {
    return hi | (uint64_t) ((a & 0x3FFFF) >> 4);
}

__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// Keeps the compiler from moving reads or writes of the 64 accumulators across the wgmma
// fences and waits around it (CUTLASS's warpgroup_fence_operand).
__device__ __forceinline__ void fence_regs(float (&d)[64]) {
#pragma unroll
    for (int i = 0; i < 64; i++) {
        asm volatile("" : "+f"(d[i])::"memory");
    }
}

#define LEGION_WG_D64                                                                                  \
    "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),       \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),          \
        "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),        \
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]),        \
        "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),        \
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]),        \
        "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]),        \
        "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),        \
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])

#define LEGION_WG_REGS64                                                                              \
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, " \
    "%21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, "  \
    "%40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, "  \
    "%59, %60, %61, %62, %63}"

// d (64 x 128 F32, the m64nNk F32 fragment) = A * B + (acc ? d : 0) for one K slice: A 64 rows,
// B 128 tokens, both K-major in shared memory behind descriptors da / db.
template <int KIND>
__device__ __forceinline__ void wgmma_n128(float (&d)[64], uint64_t da, uint64_t db, int acc) {
    if constexpr (KIND == KIND_BF16) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " LEGION_WG_REGS64
            ", %64, %65, p, 1, 1, 0, 0;\n"
            "}\n"
            : LEGION_WG_D64
            : "l"(da), "l"(db), "r"(acc));
    } else {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " LEGION_WG_REGS64
            ", %64, %65, p, 1, 1;\n"
            "}\n"
            : LEGION_WG_D64
            : "l"(da), "l"(db), "r"(acc));
    }
}
#endif  // LEGION_WGMMA

// fp8_mm.cu's a8_static / a8_scale with either scale optional.
__device__ __forceinline__ float scale0(const legion_wgmma_gemm & p) {
    const float xs = p.scale ? *p.scale : 1.0f;
    return p.wscale ? xs * *p.wscale : xs;
}
__device__ __forceinline__ float scale_at(const legion_wgmma_gemm & p, float s0, int64_t row, int64_t tok) {
    if (p.scale_stride == 0 && p.wscale_stride == 0) {
        return s0;
    }
    const float xs = p.scale ? p.scale[tok * p.scale_stride] : 1.0f;
    return p.wscale ? xs * p.wscale[row * p.wscale_stride] : xs;
}

// One consumer thread's outputs from its accumulators of a 64 x 128 half: element i of the
// m64n128 fragment is row 16 * warp + lane / 4 + 8 * (i / 2 % 2) and token 8 * (i / 4) +
// 2 * (lane % 4) + i % 2 of the half.
template <bool RES>
__device__ __forceinline__ void store_half(const legion_wgmma_gemm & p, const float (&d)[64], int64_t r0, int64_t t0,
                                           int warp, int lane, float s0) {
#pragma unroll
    for (int i = 0; i < 64; i++) {
        const int64_t row = r0 + 16 * warp + lane / 4 + 8 * (i / 2 % 2);
        const int64_t tok = t0 + 8 * (i / 4) + 2 * (lane % 4) + i % 2;
        if (row < p.m && tok < p.n) {
            const float v = d[i] * scale_at(p, s0, row, tok);
            p.dst[tok * p.m + row] = RES ? __fadd_rn(v, p.res[tok * p.m + row]) : v;
        }
    }
}

template <int KIND, bool RES>
__global__ void __launch_bounds__(THREADS, 1)
    wgmma_gemm(const legion_wgmma_gemm p, const __grid_constant__ legion_wgmma_maps maps) {
#if LEGION_WGMMA
    using C = Cfg<KIND>;
    constexpr int BN = C::BN, STAGES = C::STAGES, NH = BN / 128;
    constexpr int A_BYTES = BM * ROW, STAGE = stage_bytes<KIND>();
    static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0, "stages on swizzle atoms");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    uint64_t * full  = (uint64_t *) (smem + STAGES * STAGE);
    uint64_t * empty = full + STAGES;
    const int wg = threadIdx.x / 128, warp = threadIdx.x / 32 % 4, lane = threadIdx.x % 32;

    int tm, tn;
    tile_of((int) blockIdx.x, p.tiles_m, p.tiles_n, tm, tn);
    const int64_t m0 = (int64_t) tm * BM, n0 = (int64_t) tn * BN;
    const int steps = (int) (p.k / C::BK);

    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];\n" ::"l"((uint64_t) &maps.w) : "memory");
        asm volatile("prefetch.tensormap [%0];\n" ::"l"((uint64_t) &maps.x) : "memory");
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            bar_init(full + s, 1);
            bar_init(empty + s, 8);  // the eight consumer warps
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();  // the last CTA-wide barrier: the producer's idle threads exit below

    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (threadIdx.x == 0) {
            for (int step = 0; step < steps; step++) {
                const int s = step % STAGES, round = step / STAGES;
                if (round > 0) {
                    bar_wait(empty + s, (uint32_t) (round - 1) & 1);
                }
                unsigned char * st = smem + s * STAGE;
                bar_expect_tx(full + s, (uint32_t) STAGE);
                tma_2d(st, &maps.w, step * C::BK, (int) m0, full + s);
                tma_2d(st + A_BYTES, &maps.x, step * C::BK, (int) n0, full + s);
            }
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const int cw = wg - 1;  // consumer 0 or 1: weight rows 64 * cw .. of the tile
    const uint64_t hi = p.desc_hi;
    float acc[NH][64];
#pragma unroll
    for (int h = 0; h < NH; h++)
#pragma unroll
        for (int i = 0; i < 64; i++) acc[h][i] = 0.0f;
    float part[KIND == KIND_E4M3 ? 64 : 1];
    if constexpr (KIND == KIND_E4M3) {
#pragma unroll
        for (int i = 0; i < 64; i++) part[i] = 0.0f;
    }

    for (int step = 0; step < steps; step++) {
        const int s = step % STAGES;
        bar_wait(full + s, (uint32_t) (step / STAGES) & 1);
        const uint32_t a = saddr(smem + s * STAGE) + cw * (64 * ROW);
        const uint32_t b = saddr(smem + s * STAGE + A_BYTES);
        if constexpr (KIND == KIND_BF16) {
#pragma unroll
            for (int h = 0; h < NH; h++) fence_regs(acc[h]);
            wg_fence();
#pragma unroll
            for (int j = 0; j < ROW / SLICE; j++)
#pragma unroll
                for (int h = 0; h < NH; h++)
                    wgmma_n128<KIND>(acc[h], desc(hi, a + j * SLICE), desc(hi, b + h * 128 * ROW + j * SLICE), 1);
            wg_commit();
            // The stage before this one is read once this step's group is the only one left.
            wg_wait<1>();
#pragma unroll
            for (int h = 0; h < NH; h++) fence_regs(acc[h]);
            if (step > 0 && lane == 0) {
                bar_arrive(empty + (step - 1) % STAGES);
            }
        } else {
            fence_regs(part);
            wg_fence();
#pragma unroll
            for (int j = 0; j < ROW / SLICE; j++)
                wgmma_n128<KIND>(part, desc(hi, a + j * SLICE), desc(hi, b + j * SLICE), j > 0 ? 1 : 0);
            wg_commit();
            wg_wait<0>();
            fence_regs(part);
            if (lane == 0) {
                bar_arrive(empty + s);
            }
#pragma unroll
            for (int i = 0; i < 64; i++) acc[0][i] += part[i];
        }
    }
    if constexpr (KIND == KIND_BF16) {
        wg_wait<0>();
#pragma unroll
        for (int h = 0; h < NH; h++) fence_regs(acc[h]);
    }

    const float s0 = scale0(p);
#pragma unroll
    for (int h = 0; h < NH; h++) {
        store_half<RES>(p, acc[h], m0 + 64 * cw, n0 + 128 * h, warp, lane, s0);
    }
#else
    (void) p;
    (void) maps;
    __trap();  // the Rust side checks legion_wgmma_gemm_probe first
#endif
}

// 16 bytes of static shared memory in the sm_90a body only (see the file comment).
__global__ void wgmma_gemm_probe(int * out) {
#if LEGION_WGMMA
    __shared__ int s[4];
    volatile int * v = s;
    v[threadIdx.x % 4] = 1;
    if (out) {
        *out = v[0];
    }
#else
    if (out) {
        *out = 0;
    }
#endif
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel
// (src/model/wgmma_gemm.rs, `Kind` x residual): BF16, BF16 + residual, E4M3, E4M3 + residual.
extern "C" const void * const legion_wgmma_gemm_kernels[4] = {
    reinterpret_cast<const void *>(&wgmma_gemm<KIND_BF16, false>),
    reinterpret_cast<const void *>(&wgmma_gemm<KIND_BF16, true>),
    reinterpret_cast<const void *>(&wgmma_gemm<KIND_E4M3, false>),
    reinterpret_cast<const void *>(&wgmma_gemm<KIND_E4M3, true>),
};
extern "C" const void * const legion_wgmma_gemm_probe = reinterpret_cast<const void *>(&wgmma_gemm_probe);
