// Pair tensor parallelism (#668): the exchange between the two GPUs of an NVLink pair. The Rust
// side (src/model/core/pair/link.rs) launches the kernel and owns the memory; the protocol and
// a CPU twin of it are documented there.
//
// Every GPU has an inbox in its own VRAM. A sender writes its payload straight into the
// receiver's inbox over NVLink (peer access), every block fences, and the last block to finish
// raises the receiver's flag to the message's value `v`. A receiver polls the flag in its own
// memory, then copies the payload out (ids) or adds it to its own part (a partial sum). Two
// terms add the same in either order (IEEE addition commutes), so the sum's bits do not depend
// on which GPU holds which half.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

// Must match src/model/core/pair/link.rs (`Kernel`).
struct legion_pair_link {
    char * seg_dst[3];        // send: where each segment goes (the receiver's inbox)
    const char * seg_src[3];  // send: this GPU's tensors
    int64_t seg_bytes[3];     // send: each a multiple of 4
    uint64_t * flag;          // send: the receiver's flag; recv: this GPU's own
    uint32_t * done;          // send: this GPU's block counter (0 between launches)
    uint32_t * err;           // host memory every GPU reaches: set when a wait timed out
    const char * from;        // recv: the payload in this GPU's inbox
    const float * own;        // recv add: this GPU's part
    char * dst;               // recv: the result
    int64_t n;                // recv: values (ids: i32; add: f32)
    uint64_t v;
    int64_t timeout_ns;
    int32_t mode;
    int32_t pad;
};
}

namespace {

constexpr int MODE_SEND = 0;
constexpr int MODE_RECV_COPY = 1;
constexpr int MODE_RECV_ADD = 2;
constexpr int THREADS = 256;

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Copies `bytes` (a multiple of 4) from src to dst with every thread of the grid.
__device__ void grid_copy(char * dst, const char * src, int64_t bytes) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    if (((((uintptr_t) dst) | ((uintptr_t) src) | (uintptr_t) bytes) % 16) == 0) {
        const uint4 * s = (const uint4 *) src;
        uint4 * d       = (uint4 *) dst;
        for (int64_t i = tid; i < bytes / 16; i += stride) {
            d[i] = s[i];
        }
        return;
    }
    const uint32_t * s = (const uint32_t *) src;
    uint32_t * d       = (uint32_t *) dst;
    for (int64_t i = tid; i < bytes / 4; i += stride) {
        d[i] = s[i];
    }
}

// Thread 0 of each block: waits for this GPU's flag to reach `v`. Once one wait timed out (the
// error word), the rest give up at once: the step fails anyway.
__device__ void await_flag(const legion_pair_link & p) {
    const uint64_t t0           = now_ns();
    volatile uint64_t * flag    = (volatile uint64_t *) p.flag;
    volatile uint32_t * err     = (volatile uint32_t *) p.err;
    while (*flag != p.v) {
        if (*err != 0 || (int64_t) (now_ns() - t0) > p.timeout_ns) {
            *err = 1;
            break;
        }
    }
    __threadfence();
}

__global__ void __launch_bounds__(THREADS) pair_link_kernel(const legion_pair_link p) {
    if (p.mode == MODE_SEND) {
        for (int s = 0; s < 3; s++) {
            if (p.seg_bytes[s] > 0) {
                grid_copy(p.seg_dst[s], p.seg_src[s], p.seg_bytes[s]);
            }
        }
        // Every thread's writes reach the receiver before the block counts itself done; the
        // last block raises the flag once all of them have.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint32_t before = atomicAdd(p.done, 1u);
            if (before == gridDim.x - 1) {
                *(volatile uint32_t *) p.done = 0;
                __threadfence_system();
                *(volatile uint64_t *) p.flag = p.v;
            }
        }
        return;
    }
    if (threadIdx.x == 0) {
        await_flag(p);
    }
    __syncthreads();
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    if (p.mode == MODE_RECV_COPY) {
        const int32_t * s = (const int32_t *) p.from;
        int32_t * d       = (int32_t *) p.dst;
        for (int64_t i = tid; i < p.n; i += stride) {
            d[i] = __ldcv(s + i);
        }
        return;
    }
    static_assert(MODE_RECV_ADD == 2, "the last mode");
    const float * part = (const float *) p.from;
    float * d          = (float *) p.dst;
    for (int64_t i = tid; i < p.n; i += stride) {
        d[i] = p.own[i] + __ldcv(part + i);
    }
}

}  // namespace

// The kernel as the CUDA runtime knows it, for Rust's cudaLaunchKernel (see gate_q8.cu).
extern "C" const void * const legion_pair_link_kernel = reinterpret_cast<const void *>(&pair_link_kernel);
