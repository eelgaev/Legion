// Per-device facts the launchers cache (compute capability, SM count, a kernel's shared-memory
// opt-in): filled once per device, read from any thread (each pipeline stage launches from its
// own), so the slots are relaxed atomics; a lost race only repeats an idempotent query. A
// failed CUDA query leaves no pending error of its own behind (the next launch check would
// report it); an error pending from before the query is left for its owner.

#pragma once

#include "ggml-cuda/common.cuh"

#include <atomic>
#include <cstdint>

namespace legion_device {

// Clears the error a failed query left, unless one was pending before it (`before`).
inline void clear_own_error(cudaError_t before) {
    if (before == cudaSuccess) {
        cudaGetLastError();
    }
}

// The current device's ordinal, or -1 when it cannot be queried or is past the caches' range.
inline int current() {
    int dev = 0;
    const cudaError_t before = cudaPeekAtLastError();
    if (cudaGetDevice(&dev) != cudaSuccess) {
        clear_own_error(before);
        return -1;
    }
    return dev >= 0 && dev < GGML_CUDA_MAX_DEVICES ? dev : -1;
}

// Attribute `attr` of the current device, cached in `slots` (0 there: not yet queried); 0 when
// the query fails.
inline int attribute(std::atomic<int> * slots, cudaDeviceAttr attr) {
    const int dev = current();
    if (dev < 0) {
        return 0;
    }
    int v = slots[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        const cudaError_t before = cudaPeekAtLastError();
        if (cudaDeviceGetAttribute(&v, attr, dev) != cudaSuccess) {
            clear_own_error(before);
            return 0;
        }
        slots[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

// Compute capability of the current device as 10 * major + minor; 0 when unknown.
inline int compute_capability() {
    static std::atomic<int> cc[GGML_CUDA_MAX_DEVICES];  // 0: not yet queried
    const int dev = current();
    if (dev < 0) {
        return 0;
    }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        const cudaError_t before = cudaPeekAtLastError();
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            clear_own_error(before);
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

// Streaming multiprocessors of the current device; 0 when unknown.
inline int sm_count() {
    static std::atomic<int> n[GGML_CUDA_MAX_DEVICES];
    return attribute(n, cudaDevAttrMultiProcessorCount);
}

// Gives `kernel` `bytes` of dynamic shared memory on the current device, once per device
// (`state`: 0 not yet, 1 done, -1 refused); whether it may launch with them.
template <typename Kernel>
bool smem_opt_in(std::atomic<int8_t> * state, Kernel kernel, int bytes) {
    const int dev = current();
    if (dev < 0) {
        return false;
    }
    int8_t s = state[dev].load(std::memory_order_relaxed);
    if (s == 0) {
        const cudaError_t before = cudaPeekAtLastError();
        const bool ok = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess;
        if (!ok) {
            clear_own_error(before);
        }
        s = ok ? 1 : -1;
        state[dev].store(s, std::memory_order_relaxed);
    }
    return s > 0;
}

}  // namespace legion_device
