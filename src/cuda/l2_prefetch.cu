// A weight's first bytes read into L2 ahead of the matmul that reads it (#689 round 11;
// src/model/core/l2_prefetch.rs builds the node and launches the kernel from
// `legion_l2_prefetch_kernels` with cudaLaunchKernelExC). Kernels only.
//
// A gated DeltaNet layer of a one-user verify step on a GB10 runs ~60-70 us of small kernels
// (conv, alpha/beta, delta prep, the delta rule, the gated norm) between its qkv and z
// projections, with DRAM mostly idle, and the z projection after them ran ~20 us over its
// bytes' time (off by default: measured, the prefetch costs the stream what it saves z, see
// the .rs file). This kernel waits for the kernel ahead of it (a programmatic dependent launch,
// so it is resident while that one runs and its launch costs nothing), then gives each thread
// one bulk L2 prefetch of `chunk` bytes and exits: the matmul finds its first rows in L2. It
// writes nothing, so no output changes. With `late` it issues the prefetches first and waits
// after (still before it ends, so the kernels after it see the one ahead complete).

#include <cstdint>

namespace {

__device__ __forceinline__ void l2_prefetch_issue(const char * p, int64_t bytes, int32_t chunk) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * chunk;
    if (i < bytes) {
        const int64_t  left = bytes - i;
        const uint32_t n    = (uint32_t) (left < chunk ? left : chunk);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + i), "r"(n) : "memory");
    }
#else
    (void) p;
    (void) bytes;
    (void) chunk;
#endif
}

__global__ void __launch_bounds__(32) l2_prefetch(const char * p, int64_t bytes, int32_t chunk, int32_t late) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (late) {
        l2_prefetch_issue(p, bytes, chunk);
    }
    // After the kernel ahead: its own reads keep DRAM busy until it ends, and the kernels
    // after this one see it complete (the stream's order) only once that one has.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (!late) {
        l2_prefetch_issue(p, bytes, chunk);
    }
#else
    (void) p;
    (void) bytes;
    (void) chunk;
    (void) late;
#endif
}

}  // namespace

extern "C" const void * const legion_l2_prefetch_kernels[1] = {
    reinterpret_cast<const void *>(&l2_prefetch),
};
