// The sum of the tensor-parallel all-reduce over machines (src/tp_net.rs, #619): every rank's
// partial output added in rank order, one rounding at a time, so every rank gets the same bits
// (and the CPU op's, src/tp_net/comm.rs). The partials of the other ranks are in pinned host
// memory the GPU reads directly; this rank's is its own device tensor.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int MAX_RANKS = 8;  // tp_net::MAX_RANKS

struct tp_sum_args {
    const float * src[MAX_RANKS];
    float * dst;
    int64_t n;
    int32_t n_src;
};

__global__ void tp_sum(const tp_sum_args a) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < a.n; i += stride) {
        float acc = a.src[0][i];
        for (int r = 1; r < a.n_src; ++r) {
            acc = __fadd_rn(acc, a.src[r][i]);
        }
        a.dst[i] = acc;
    }
}

// --tp-wire bf16 (#751): the partials travel as BF16, this rank's included (rounded the same
// way, so every rank adds the same values), added in F32 in rank order.
__global__ void tp_sum_bf16(const tp_sum_args a) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < a.n; i += stride) {
        float acc = __bfloat162float(((const __nv_bfloat16 *) a.src[0])[i]);
        for (int r = 1; r < a.n_src; ++r) {
            acc = __fadd_rn(acc, __bfloat162float(((const __nv_bfloat16 *) a.src[r])[i]));
        }
        a.dst[i] = acc;
    }
}

struct tp_pack_args {
    const float * src;
    __nv_bfloat16 * dst;
    int64_t n;
};

// A partial rounded to BF16 (nearest even) into the pinned send buffer.
__global__ void tp_pack_bf16(const tp_pack_args a) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < a.n; i += stride) {
        a.dst[i] = __float2bfloat16_rn(a.src[i]);
    }
}

struct tp_copy_args {
    const uint16_t * src;
    uint16_t * dst;
    int64_t n;
};

// The BF16 output of a --tp-wire bf16 all-gather (#1176): the values as they travelled, 16
// bytes at a time where both ends are 16-byte aligned.
__global__ void tp_copy_bf16(const tp_copy_args a) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t first = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const bool vec = (((uintptr_t) a.src | (uintptr_t) a.dst) & 15) == 0;
    const int64_t nv = vec ? a.n / 8 : 0;
    for (int64_t i = first; i < nv; i += stride) {
        ((uint4 *) a.dst)[i] = ((const uint4 *) a.src)[i];
    }
    for (int64_t i = nv * 8 + first; i < a.n; i += stride) {
        a.dst[i] = a.src[i];
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel.
extern "C" const void * const legion_tp_net_sum_kernel = reinterpret_cast<const void *>(&tp_sum);
extern "C" const void * const legion_tp_net_sum_bf16_kernel =
    reinterpret_cast<const void *>(&tp_sum_bf16);
extern "C" const void * const legion_tp_net_pack_bf16_kernel =
    reinterpret_cast<const void *>(&tp_pack_bf16);
extern "C" const void * const legion_tp_net_copy_bf16_kernel =
    reinterpret_cast<const void *>(&tp_copy_bf16);
