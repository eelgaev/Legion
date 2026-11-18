// qwen4exp (Qwen3.8-Flash-Next) hyper-connection mixer, prompt steps: ggml's gated hc_pre
// (`dsv4_hc_pre_f32<true>`, ggml-cuda/dsv4-hc.cu), on a V100 with the mixed block input also
// written as F16, the operand ggml-cuda's cuBLAS path converts it to for every quantized
// projection that reads it (#774); on sm_80+ with BF16 mixer weights (the NVFP4 checkpoints)
// F32 only, xn rebuilt (#836); and ggml's hc_pre itself, xn read and F32 only, wherever
// qwen4exp's mixer would run ggml-cuda's (#836: all compute in Legion kernels). The host side, the shapes, when it runs and the CPU reference
// are in src/model/qwen4/hc_mix16.rs:
//
//   mixed   = scale * sum_c xn[c] * sigmoid(gate[c])      the streams c in order, as ggml
//   mixed16 = (half) mixed                                 ggml's F32 -> F16 rounding, if asked
//
// xn is read, or rebuilt as (res * norm scale) * gamma from the streams and scales hc_post_norm
// writes in its place (its own two rounded multiplies, so the same bits).
//
// One thread per (value, token), as ggml's kernel: the same expressions, compiled with the
// same fast-math flags, so the same bits.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_hc_mix16 {
    const float * gate;     // F32 [n][hc][e], contiguous
    const float * xn;       // F32 [n][hc][e], or null when rebuilt
    const float * res;      // F32 [n][hc][e] (the new streams), or null
    const float * sc;       // F32 [n][hc] each stream's norm scale, or null
    const float * g;        // F32 [hc][e] gamma, or null
    float *       mixed;    // F32 [n][e]
    __half *      mixed16;  // F16 [n][e], or null (F32 only)
    int64_t       e, hc, n;
    float         scale;
};

// Launches the kernel on `stream`; returns a cudaError_t.
int legion_hc_mix16_launch(const legion_hc_mix16 * p, void * stream);

// One rank's rows of the plain hc_pre (#1176, a mixer split over tensor-parallel ranks): the
// values first..first + eh of every token, each stream's gate rows its own [n][eh] block.
struct legion_hc_pre_part {
    const float * gate[4];  // F32 [n][eh] per stream c
    const float * xn;       // F32 [n][hc][e], at row `first` of stream 0
    float *       mixed;    // F32 [n][eh]
    int64_t       e, eh, hc, n;
    float         scale;
};

// Launches the part kernel on `stream`; returns a cudaError_t.
int legion_hc_pre_part_launch(const legion_hc_pre_part * p, void * stream);

}  // extern "C"

namespace {

template <bool RB, bool W16>
__global__ void hc_mix16_kernel(const legion_hc_mix16 p) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= p.e * p.n) {
        return;
    }
    const int64_t i  = ir % p.e;
    const int64_t t  = ir / p.e;
    const int64_t k  = p.e * p.hc;
    float sum = 0.0f;
    for (int64_t c = 0; c < p.hc; ++c) {
        const int64_t at = t * k + c * p.e + i;
        float xv;
        if constexpr (RB) {
            xv = __fmul_rn(__fmul_rn(p.res[at], p.sc[t * p.hc + c]), p.g[c * p.e + i]);
        } else {
            xv = p.xn[at];
        }
        const float wv = 1.0f / (1.0f + expf(-p.gate[at]));
        sum += xv * wv;
    }
    const float m = p.scale * sum;
    p.mixed[t * p.e + i] = m;
    if constexpr (W16) {
        p.mixed16[t * p.e + i] = __float2half(m);
    }
}

// The plain hc_pre (xn read, F32 only) for four consecutive values of a token per thread:
// float4 loads and stores, each value's sum the kernel's above (the same expressions in the same
// order, so the same bits), a quarter of the threads.
__global__ void hc_pre_x4_kernel(const legion_hc_mix16 p) {
    const int64_t e4 = p.e / 4;
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= e4 * p.n) {
        return;
    }
    const int64_t i4 = ir % e4;
    const int64_t t  = ir / e4;
    const int64_t k4 = e4 * p.hc;
    const float4 * x4 = (const float4 *) p.xn;
    const float4 * g4 = (const float4 *) p.gate;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int64_t c = 0; c < p.hc; ++c) {
        const int64_t at = t * k4 + c * e4 + i4;
        const float4 xv = x4[at];
        const float4 gv = g4[at];
        const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
        const float gs[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float wv = 1.0f / (1.0f + expf(-gs[j]));
            s[j] += xs[j] * wv;
        }
    }
    float4 m;
    m.x = p.scale * s[0];
    m.y = p.scale * s[1];
    m.z = p.scale * s[2];
    m.w = p.scale * s[3];
    ((float4 *) p.mixed)[t * e4 + i4] = m;
}

// hc_mix16_kernel<false, false> on one rank's rows: the same expressions in the same order per
// value, so the bits of the whole op's values there.
__global__ void hc_pre_part_kernel(const legion_hc_pre_part p) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= p.eh * p.n) {
        return;
    }
    const int64_t i = ir % p.eh;
    const int64_t t = ir / p.eh;
    const int64_t k = p.e * p.hc;
    float sum = 0.0f;
    for (int64_t c = 0; c < p.hc; ++c) {
        const float xv = p.xn[t * k + c * p.e + i];
        const float wv = 1.0f / (1.0f + expf(-p.gate[c][t * p.eh + i]));
        sum += xv * wv;
    }
    p.mixed[t * p.eh + i] = p.scale * sum;
}

}  // namespace

extern "C" int legion_hc_pre_part_launch(const legion_hc_pre_part * p, void * stream) {
    if (p->e < 1 || p->eh < 1 || p->n < 1 || p->hc < 1 || p->hc > 4 || p->xn == nullptr) {
        return (int) cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int64_t blocks = (p->eh * p->n + threads - 1) / threads;
    hc_pre_part_kernel<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}

extern "C" int legion_hc_mix16_launch(const legion_hc_mix16 * p, void * stream) {
    if (p->e < 1 || p->n < 1 || p->hc < 1) {
        return (int) cudaErrorInvalidValue;
    }
    const bool rb = p->res != nullptr;
    if (rb ? (p->sc == nullptr || p->g == nullptr) : p->xn == nullptr) {
        return (int) cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int64_t blocks = (p->e * p->n + threads - 1) / threads;
    const auto st = (cudaStream_t) stream;
    if (p->mixed16 == nullptr && !rb) {
        // ggml's hc_pre (#836): four values a thread where the rows and pointers allow it.
        const auto a16 = [](const void * q) { return ((uintptr_t) q) % 16 == 0; };
        if (p->e % 4 == 0 && a16(p->xn) && a16(p->gate) && a16(p->mixed)) {
            const int64_t b4 = (p->e / 4 * p->n + threads - 1) / threads;
            hc_pre_x4_kernel<<<(unsigned) b4, threads, 0, st>>>(*p);
        } else {
            hc_mix16_kernel<false, false><<<(unsigned) blocks, threads, 0, st>>>(*p);
        }
    } else if (p->mixed16 == nullptr) {
        // F32 only, rebuilt (#836).
        hc_mix16_kernel<true, false><<<(unsigned) blocks, threads, 0, st>>>(*p);
    } else if (rb) {
        hc_mix16_kernel<true, true><<<(unsigned) blocks, threads, 0, st>>>(*p);
    } else {
        hc_mix16_kernel<false, true><<<(unsigned) blocks, threads, 0, st>>>(*p);
    }
    return (int) cudaGetLastError();
}
