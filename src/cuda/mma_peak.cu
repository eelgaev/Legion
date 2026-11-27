// The tensor cores' mma.sync peak (#1183): register-only loops of one MMA form, no memory
// traffic, for `src/model/core/mma_peak.rs` (an ignored --nocapture test prints the table).
//
// Each warp keeps CHAINS independent accumulators and issues CHAINS MMAs a loop trip with the
// same A and B registers (made from the thread index and a seed, so nothing folds to a
// constant); the accumulators are summed into one value a thread and stored once at the end, so
// the compiler keeps every MMA (the asm is volatile too). Thread 0 of each CTA stores %clock64 at
// its start and end (after a barrier) and its %smid, for the clock and the SMs the CTAs ran on.
//
// The forms (FORM, in the order of legion_mma_peak_kernels' rows):
//   0 m16n8k16 BF16 x BF16 -> F32          sm_80+
//   1 m16n8k16 F16 x F16 -> F32            sm_80+
//   2 m16n8k16 F16 x F16 -> F16            sm_80+
//   3 m16n8k32 E4M3 x E4M3 -> F32          sm_89+
//   4 kind::f8f6f4 m16n8k32 E4M3 -> F32    sm_120a / sm_121a
//   5 kind::mxf8f6f4 block_scale scale_vec::1X m16n8k32 E4M3, UE8M0 scales -> F32   sm_12xa
//   6 kind::mxf4nvf4 block_scale scale_vec::4X m16n8k64 E2M1, UE4M3 scales -> F32   sm_12xa
//   7 m16n8k32 S8 x S8 -> S32              sm_80+ (src/cuda/q40_mm.cu's form)
// A form's body exists only where its architecture is the build target's; elsewhere the kernel
// traps. legion_mma_peak_probe reports which forms the running device's code has (a bit per
// form), so the host never launches a trap. The Rust side launches everything with
// cudaLaunchKernel from the tables at the end; this file is the kernels only.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {
struct legion_mma_peak {
    float * out;      // one value a thread: [grid][block]
    long long * clk;  // per CTA: clock64 at the start, at the end, and the SM id
    int32_t iters;    // loop trips; each issues CHAINS MMAs a warp
    uint32_t seed;
};
}

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define MP_SM80 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define MP_SM89 1
#endif
#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define MP_SM12XA 1
#endif

constexpr int FORMS = 8;
constexpr int MAX_THREADS = 1024;  // 32 warps, the sweep's largest CTA

// Whether this architecture's code has form F's body.
template <int F> __host__ __device__ constexpr bool has_form() {
#if defined(MP_SM12XA)
    return true;
#elif defined(MP_SM89)
    return F <= 3 || F == 7;
#elif defined(MP_SM80)
    return F <= 2 || F == 7;
#else
    return false;
#endif
}

// One MMA of form F on accumulator c (four F32 values; form 2 uses c[0] and c[1] as F16 pairs).
template <int F> __device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(MP_SM80)
    if constexpr (F == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == 1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == 2) {
        // Register moves only (no pointer cast: that would put the accumulators in memory).
        uint32_t h0 = __float_as_uint(c[0]), h1 = __float_as_uint(c[1]);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(h0), "+r"(h1)
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        c[0] = __uint_as_float(h0);
        c[1] = __uint_as_float(h1);
    } else if constexpr (F == 7) {
        // Integer sums in the accumulators' bits (register moves only, as form 2).
        uint32_t s0 = __float_as_uint(c[0]), s1 = __float_as_uint(c[1]), s2 = __float_as_uint(c[2]),
                 s3 = __float_as_uint(c[3]);
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3)
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        c[0] = __uint_as_float(s0);
        c[1] = __uint_as_float(s1);
        c[2] = __uint_as_float(s2);
        c[3] = __uint_as_float(s3);
    }
#endif
#if defined(MP_SM89)
    if constexpr (F == 3) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#endif
#if defined(MP_SM12XA)
    if constexpr (F == 4) {
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
                     "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (F == 5) {
        // UE8M0 127: a scale of 1 for every row and column.
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x7f7f7f7fu), "r"(0x7f7f7f7fu));
    } else if constexpr (F == 6) {
        // UE4M3 0x38: a scale of 1 for every 16 values of k.
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x38383838u), "r"(0x38383838u));
    }
#endif
}

// Operand bits of form F for thread value v: small values that vary by thread (the sums are
// stored, never looked at; F16 sums may reach infinity on long runs, which costs nothing).
template <int F> __device__ __forceinline__ uint32_t operand(uint32_t v) {
    if constexpr (F == 0) {
        return 0x3c003c00u | (v & 0x007f007fu);  // BF16 ~2^-7
    } else if constexpr (F == 1 || F == 2) {
        return 0x20002000u | (v & 0x03ff03ffu);  // F16 ~2^-7
    } else if constexpr (F <= 5) {
        return 0x20202020u | (v & 0x07070707u);  // E4M3 ~2^-3
    } else if constexpr (F == 7) {
        return 0x01010101u | (v & 0x0e0e0e0eu);  // S8 1 .. 15
    } else {
        return 0x11111111u | (v & 0x22222222u);  // E2M1 0.5 or 1.5
    }
}

template <int F, int CHAINS> __global__ void __launch_bounds__(MAX_THREADS) mma_peak(const legion_mma_peak p) {
    if constexpr (has_form<F>()) {
        __syncthreads();
        const long long t0 = clock64();
        const uint32_t v = p.seed ^ (threadIdx.x * 0x9e3779b9u);
        uint32_t a[4], b[2];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = operand<F>(v >> i);
        }
        b[0] = operand<F>(v >> 5);
        b[1] = operand<F>(v >> 7);
        float acc[CHAINS][4];
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[c][l] = 0.0f;
            }
        }
#pragma unroll 8
        for (int i = 0; i < p.iters; i++) {
#pragma unroll
            for (int c = 0; c < CHAINS; c++) {
                mma<F>(acc[c], a, b);
            }
        }
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < CHAINS; c++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                s += acc[c][l];
            }
        }
        p.out[(int64_t) blockIdx.x * blockDim.x + threadIdx.x] = s;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned sm;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            p.clk[3 * blockIdx.x]     = t0;
            p.clk[3 * blockIdx.x + 1] = clock64();
            p.clk[3 * blockIdx.x + 2] = sm;
        }
    } else {
        __trap();
    }
}

// Bit F set: the running device's code has form F's body.
__global__ void mma_peak_probe(int * forms) {
    int m = 0;
    if (has_form<0>()) m |= 1 << 0;
    if (has_form<1>()) m |= 1 << 1;
    if (has_form<2>()) m |= 1 << 2;
    if (has_form<3>()) m |= 1 << 3;
    if (has_form<4>()) m |= 1 << 4;
    if (has_form<5>()) m |= 1 << 5;
    if (has_form<6>()) m |= 1 << 6;
    if (has_form<7>()) m |= 1 << 7;
    *forms = m;
}

#define MP_ROW(F) \
    { reinterpret_cast<const void *>(&mma_peak<F, 1>), reinterpret_cast<const void *>(&mma_peak<F, 2>), \
      reinterpret_cast<const void *>(&mma_peak<F, 4>), reinterpret_cast<const void *>(&mma_peak<F, 8>) }

}  // namespace

// The kernels as the CUDA runtime knows them, for Rust's cudaLaunchKernel
// (src/model/core/mma_peak.rs): [form][1, 2, 4, 8 accumulator chains], one legion_mma_peak
// argument; up to MAX_THREADS threads a CTA, no shared memory.
extern "C" const void * const legion_mma_peak_kernels[FORMS][4] = {
    MP_ROW(0), MP_ROW(1), MP_ROW(2), MP_ROW(3), MP_ROW(4), MP_ROW(5), MP_ROW(6), MP_ROW(7),
};

// The probe (one int * argument, one thread).
extern "C" const void * const legion_mma_peak_probe = reinterpret_cast<const void *>(&mma_peak_probe);
