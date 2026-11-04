// stage_pull: one GPU copies bytes from host memory into another GPU's VRAM over its own host
// link and the pair's NVLink (peer stores), so a prompt's staging copies into a busy GPU arrive
// through two links (--prefill-peer-stage). On an AC922 the copy engine moves 68-71 GB/s from
// host memory into its own GPU; this kernel on the idle NVLink peer, beside it, moves 62 GB/s
// more into the same GPU (134.7 GB/s together, 2 GB copies). As Project Maya's two-GPU prefill
// staging (mw00/project-maya, MIT), there through a bounce buffer and copy engines.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int PULL_THREADS = 512;

__global__ void __launch_bounds__(PULL_THREADS)
stage_pull(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, size_t n) {
    const size_t tid = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t body = 0;
    if ((((uintptr_t) dst | (uintptr_t) src) & 15) == 0) {
        body = n / 16;
        const uint4 * s = (const uint4 *) src;
        uint4 * d = (uint4 *) dst;
        for (size_t i = tid; i < body; i += stride) {
            d[i] = s[i];
        }
        body *= 16;
    }
    for (size_t i = body + tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

}  // namespace

// Copies `n` bytes from `src` to `dst` on `stream` (of the current device), `ctas` blocks.
extern "C" int legion_stage_pull(void * dst, const void * src, size_t n, int ctas, void * stream) {
    if (n == 0) {
        return 0;
    }
    stage_pull<<<ctas, PULL_THREADS, 0, (cudaStream_t) stream>>>((uint8_t *) dst, (const uint8_t *) src, n);
    return (int) cudaGetLastError();
}
