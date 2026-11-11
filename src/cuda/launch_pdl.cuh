// Programmatic dependent launch on compute capability 12.x (GB10, RTX 50), shared by
// qwen4_hc.cu and glm5next.cu (#594; moved from qwen4_hc.cu; the whole 12.x family since #1200).
// A kernel launched through launch_pdl must wait for the kernel ahead of it
// (griddepcontrol.wait) before it reads that kernel's output.

#pragma once

#include "device_cache.cuh"

namespace {

// Launches `kernel` on `s` with programmatic stream serialization: it may start before the
// kernel ahead of it on the stream ends, and waits for it in griddepcontrol.wait. Devices other
// than compute capability 12.x with the kernel's sm_80+ code (legion_device::sm12x_device) get a
// plain launch, and so does Hopper (9.0) unless --hopper-pdl turns it on there
// (legion_device::pdl_device; untested on hardware, #1215).
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
    cfg.numAttrs         = legion_device::pdl_device(kernel) ? 1 : 0;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace
