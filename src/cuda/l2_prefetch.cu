// A weight's first bytes read into L2 ahead of the matmul that reads it (#689 round 11;
// src/model/core/l2_prefetch.rs builds the node and launches the kernel from
// `legion_l2_prefetch_kernels` with cudaLaunchKernelExC). Kernels only.
//
// A gated DeltaNet layer of a one-user verify step on a GB10 runs ~60-70 us of small kernels
// (conv, alpha/beta, delta prep, the delta rule, the gated norm) between its qkv and z
// projections, with DRAM mostly idle, and the z projection after them ran ~20 us over its
// bytes' time (on the compute stream the prefetch costs the stream what it saves z; on a side
// stream, paced, it does not: see the .rs file). This kernel waits for the kernel ahead of it (a programmatic dependent launch,
// so it is resident while that one runs and its launch costs nothing), then gives each thread
// one bulk L2 prefetch of `chunk` bytes and exits: the matmul finds its first rows in L2. It
// writes nothing, so no output changes. With `late` it issues the prefetches first and waits
// after (still before it ends, so the kernels after it see the one ahead complete). With
// `pace_ns` (the side stream's default) one CTA issues them a block's chunks at a time,
// pace_ns apart (round 12), with an optional start delay and one hold (round 13 probes).

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

__device__ __forceinline__ void l2_sleep_until(uint64_t until) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    while (t < until) {
        __nanosleep(64);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    }
#else
    (void) until;
#endif
}

// Paced (#689 round 12, pace_ns > 0, one CTA): step s issues the block's chunks of bytes
// [s * step, (s + 1) * step) (step = blockDim.x * chunk), then sleeps until pace_ns * (s + 1)
// after its start, so the prefetches reach DRAM at step / pace_ns bytes a ns instead of as one
// burst that the kernels beside it queue behind. Round 13 probes: the first step waits delay_ns
// after the kernel's wait returns, and once hold_at bytes are issued the steps after it wait
// hold_ns more (a gap in the stream of prefetches, so a kernel beside it has DRAM to itself).
__device__ __forceinline__ void l2_prefetch_paced(const char * p, int64_t bytes, int32_t chunk, int32_t pace_ns,
                                                  int32_t delay_ns, int64_t hold_at, int32_t hold_ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int64_t step = (int64_t) blockDim.x * chunk;
    uint64_t      t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    t0 += (uint64_t) delay_ns;
    l2_sleep_until(t0);
    for (int64_t off = 0, s = 1; off < bytes; off += step, ++s) {
        const int64_t i = off + (int64_t) threadIdx.x * chunk;
        if (i < bytes) {
            const int64_t  left = bytes - i;
            const uint32_t n    = (uint32_t) (left < chunk ? left : chunk);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + i), "r"(n) : "memory");
        }
        if (off + step >= bytes) {
            break;
        }
        if (hold_ns > 0 && off < hold_at && off + step >= hold_at) {
            t0 += (uint64_t) hold_ns;
        }
        l2_sleep_until(t0 + (uint64_t) pace_ns * (uint64_t) s);
    }
#else
    (void) p;
    (void) bytes;
    (void) chunk;
    (void) pace_ns;
    (void) delay_ns;
    (void) hold_at;
    (void) hold_ns;
#endif
}

__global__ void __launch_bounds__(32) l2_prefetch(const char * p, int64_t bytes, int32_t chunk, int32_t late,
                                                  int32_t pace_ns, int32_t delay_ns, int64_t hold_at,
                                                  int32_t hold_ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pace_ns > 0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        l2_prefetch_paced(p, bytes, chunk, pace_ns, delay_ns, hold_at, hold_ns);
        return;
    }
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
    (void) pace_ns;
    (void) delay_ns;
    (void) hold_at;
    (void) hold_ns;
#endif
}

}  // namespace

extern "C" const void * const legion_l2_prefetch_kernels[1] = {
    reinterpret_cast<const void *>(&l2_prefetch),
};
