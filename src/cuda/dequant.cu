// Faster contiguous dequantizers installed in ggml-cuda's conversion dispatchers by the
// linker's --wrap support (build/cuda_wrap.rs). This keeps the pinned ggml submodule pristine.
//
// The four-thread Q8_0 layout and the eight-superblock launches/vector stores are adapted from
// Strata-AC922's MIT-licensed db52f95 and d5d7c0f commits.

#include "ggml-cuda/convert.cuh"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

constexpr int DQ_SUPERBLOCKS = 8;

template <typename T, int N>
__device__ __forceinline__ void store_vec(T * dst, const T (&values)[N]) {
    constexpr int bytes = sizeof(T) * N;
    static_assert(bytes % 8 == 0);
    if constexpr (bytes % 16 == 0) {
#pragma unroll
        for (int i = 0; i < bytes / 16; ++i) {
            reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(values)[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < bytes / 8; ++i) {
            reinterpret_cast<uint2 *>(dst)[i] = reinterpret_cast<const uint2 *>(values)[i];
        }
    }
}

template <typename T>
__device__ __forceinline__ void store_pair(T * dst, const T (&values)[2]) {
    if constexpr (sizeof(T) == 2) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(values);
    } else {
        *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(values);
    }
}

template <typename T>
__global__ void dequant_q8_0(const block_q8_0 * __restrict__ blocks, T * __restrict__ out, int64_t nblocks) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 4 * nblocks) {
        return;
    }
    const int64_t ib = i >> 2;
    const int piece = int(i & 3);
    const block_q8_0 & block = blocks[ib];
    const float d = __half2float(block.d);
    const uint16_t * q = reinterpret_cast<const uint16_t *>(block.qs + 8 * piece);
    __align__(16) T values[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint16_t packed = q[j];
        values[2 * j + 0] = ggml_cuda_cast<T>(float(int8_t(packed & 0xff)) * d);
        values[2 * j + 1] = ggml_cuda_cast<T>(float(int8_t(packed >> 8)) * d);
    }
    store_vec(out + 32 * ib + 8 * piece, values);
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <typename T>
__device__ __forceinline__ void dequant_q4_k(const block_q4_K & block, T * out, int tid) {
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const float dall = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));
    const uint8_t * q = block.qs + 32 * il + 4 * ir;
    uint8_t sc;
    uint8_t m;
    get_scale_min_k4(is, block.scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, block.scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;
    __align__(16) T lo[4];
    __align__(16) T hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = ggml_cuda_cast<T>(d1 * (q[j] & 0x0f) - m1);
        hi[j] = ggml_cuda_cast<T>(d2 * (q[j] >> 4) - m2);
    }
    T * dst = out + 64 * il + 4 * ir;
    store_vec(dst, lo);
    store_vec(dst + 32, hi);
}

template <typename T>
__device__ __forceinline__ void dequant_q5_k(const block_q5_K & block, T * out, int tid) {
#pragma unroll
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16;
        const int ir = tt % 16;
        const int is = 2 * il;
        const float dall = __half2float(__low2half(block.dm));
        const float dmin = __half2float(__high2half(block.dm));
        const uint8_t * ql = block.qs + 32 * il + 2 * ir;
        const uint8_t * qh = block.qh + 2 * ir;
        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, block.scales, sc, m);
        const float d1 = dall * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, block.scales, sc, m);
        const float d2 = dall * sc;
        const float m2 = dmin * m;
        uint8_t hm = uint8_t(1 << (2 * il));
        __align__(8) T lo[2] = {
            ggml_cuda_cast<T>(d1 * ((ql[0] & 0x0f) + ((qh[0] & hm) ? 16 : 0)) - m1),
            ggml_cuda_cast<T>(d1 * ((ql[1] & 0x0f) + ((qh[1] & hm) ? 16 : 0)) - m1),
        };
        hm <<= 1;
        __align__(8) T hi[2] = {
            ggml_cuda_cast<T>(d2 * ((ql[0] >> 4) + ((qh[0] & hm) ? 16 : 0)) - m2),
            ggml_cuda_cast<T>(d2 * ((ql[1] >> 4) + ((qh[1] & hm) ? 16 : 0)) - m2),
        };
        T * dst = out + 64 * il + 2 * ir;
        store_pair(dst, lo);
        store_pair(dst + 32, hi);
    }
}

template <typename T>
__device__ __forceinline__ void dequant_q5_1(const block_q5_1 * blocks, T * out, int tid, int nblocks) {
    const int ib = tid % 8;
    if (ib >= nblocks) {
        return;
    }
    const int il = tid / 8;
    const block_q5_1 & block = blocks[ib];
    const float2 dm = __half22float2(block.dm);
    uint32_t qh;
    memcpy(&qh, block.qh, sizeof(qh));
    __align__(16) T lo[4];
    __align__(16) T hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int h0 = ((qh >> iqs) << 4) & 0x10;
        const int h1 = (qh >> (iqs + 12)) & 0x10;
        lo[j] = ggml_cuda_cast<T>(float((block.qs[iqs] & 0x0f) | h0) * dm.x + dm.y);
        hi[j] = ggml_cuda_cast<T>(float((block.qs[iqs] >> 4) | h1) * dm.x + dm.y);
    }
    T * dst = out + 32 * ib + 4 * il;
    store_vec(dst, lo);
    store_vec(dst + 16, hi);
}

template <typename T, ggml_type TYPE>
__global__ void dequant_superblocks(const void * __restrict__ src, T * __restrict__ out, int64_t n) {
    const int64_t ib = int64_t(blockIdx.x) * DQ_SUPERBLOCKS + threadIdx.y;
    if (ib * QK_K >= n) {
        return;
    }
    T * dst = out + ib * QK_K;
    if constexpr (TYPE == GGML_TYPE_Q4_K) {
        dequant_q4_k(reinterpret_cast<const block_q4_K *>(src)[ib], dst, threadIdx.x);
    } else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        dequant_q5_k(reinterpret_cast<const block_q5_K *>(src)[ib], dst, threadIdx.x);
    } else {
        const int nblocks = int(min(int64_t(QK_K), n - ib * QK_K) / QK5_1);
        dequant_q5_1(
            reinterpret_cast<const block_q5_1 *>(src) + ib * (QK_K / QK5_1), dst, threadIdx.x, nblocks);
    }
}

template <typename T>
void launch_q8_0(const void * src, T * dst, int64_t n, cudaStream_t stream) {
    const int64_t blocks = n / QK8_0;
    const unsigned grid = unsigned((4 * blocks + 255) / 256);
    dequant_q8_0<<<grid, 256, 0, stream>>>(reinterpret_cast<const block_q8_0 *>(src), dst, blocks);
}

template <typename T, ggml_type TYPE>
void launch_superblocks(const void * src, T * dst, int64_t n, cudaStream_t stream) {
    const int64_t count = (n + QK_K - 1) / QK_K;
    const unsigned grid = unsigned((count + DQ_SUPERBLOCKS - 1) / DQ_SUPERBLOCKS);
    dequant_superblocks<T, TYPE><<<grid, dim3(32, DQ_SUPERBLOCKS), 0, stream>>>(src, dst, n);
}

}  // namespace

// GNU ld redirects ggml-cuda's references to these C++ ABI symbols. The `__real` declarations
// resolve to ggml's original dispatchers, which remain the fallback for every other type.
// Q8_0 -> F16 keeps ggml's kernel: on a V100 it already runs at HBM bandwidth (1.00x measured).
#define LEGION_WRAP_GETTER(SUFFIX, TYPE)                                                                      \
    extern "C" to_t_cuda_t<TYPE> real_##SUFFIX(ggml_type)                                                    \
        asm("__real__Z21ggml_get_to_" #SUFFIX "_cuda9ggml_type");                                          \
    extern "C" to_t_cuda_t<TYPE> wrap_##SUFFIX(ggml_type)                                                   \
        asm("__wrap__Z21ggml_get_to_" #SUFFIX "_cuda9ggml_type");                                          \
    extern "C" to_t_cuda_t<TYPE> wrap_##SUFFIX(ggml_type type) {                                           \
        switch (type) {                                                                                       \
            case GGML_TYPE_Q8_0:                                                                              \
                if constexpr (!std::is_same_v<TYPE, half>) {                                                  \
                    return launch_q8_0<TYPE>;                                                                 \
                }                                                                                             \
                break;                                                                                        \
            case GGML_TYPE_Q4_K: return launch_superblocks<TYPE, GGML_TYPE_Q4_K>;                            \
            case GGML_TYPE_Q5_K: return launch_superblocks<TYPE, GGML_TYPE_Q5_K>;                            \
            case GGML_TYPE_Q5_1: return launch_superblocks<TYPE, GGML_TYPE_Q5_1>;                            \
            default: break;                                                                                   \
        }                                                                                                     \
        return real_##SUFFIX(type);                                                                           \
    }

LEGION_WRAP_GETTER(fp16, half)
LEGION_WRAP_GETTER(bf16, nv_bfloat16)
LEGION_WRAP_GETTER(fp32, float)

#undef LEGION_WRAP_GETTER
