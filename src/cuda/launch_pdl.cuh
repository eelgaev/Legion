// Programmatic dependent launch on the GB10, shared by qwen4_hc.cu and glm5next.cu (#594; moved
// from qwen4_hc.cu). A kernel launched through launch_pdl must wait for the kernel ahead of it
// (griddepcontrol.wait) before it reads that kernel's output.

#pragma once

#include "device_cache.cuh"

#include <atomic>

namespace {

// Whether this device is a GB10 with the sm_80+ kernels built for it (`probe`: one of the
// caller's kernels, compiled as they all are).
template <typename... KArgs>
bool gb10_device(void (*probe)(KArgs...)) {
    static std::atomic<int> state[GGML_CUDA_MAX_DEVICES];  // 0 unknown, 1 yes, 2 no
    const int dev = legion_device::current();
    if (dev < 0) {
        return false;
    }
    int st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        cudaFuncAttributes fa{};
        const cudaError_t before = cudaPeekAtLastError();
        const bool code = cudaFuncGetAttributes(&fa, probe) == cudaSuccess && fa.binaryVersion >= 80 &&
                          fa.ptxVersion >= 80;
        if (!code) {
            legion_device::clear_own_error(before);
        }
        st = legion_device::compute_capability() == 121 && code ? 1 : 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

// Launches `kernel` on `s` with programmatic stream serialization: it may start before the
// kernel ahead of it on the stream ends, and waits for it in griddepcontrol.wait. Other devices
// than the GB10 get a plain launch.
template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, int threads, int smem, cudaStream_t s, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim          = grid;
    cfg.blockDim         = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream           = s;
    cfg.attrs            = &attr;
    cfg.numAttrs         = gb10_device(kernel) ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace
