// Per-token FP8 activations (--fp8-act a8-dynamic) written by the op that makes them: one CTA
// per token computes the token's activation row into registers, its amax, the scale
// s = amax / 448 and the E4M3 codes e4m3(y * (1 / s)), in the packed layout of
// fp8_quant_dyn (fp8_mm.cu: codes [k, n], then the n F32 scales). The graph side and the CPU
// references are src/model/fp8_act.rs.
//
// Kinds of row:
// - plain:      y = x                                  (fp8_quant_dyn in one pass: the row
//                                                       stays in registers between the amax and
//                                                       the codes instead of being read twice)
// - rms_norm:   y = mul(rms_norm(x, eps), w)            (the dense models' attn / ffn norms)
// - gate:       y = op(x) * g, op sigmoid or SiLU       (the attention output gate, SwiGLU)
// - gated_norm: y = rms_norm(o) * w * gate(z) per head  (a gated DeltaNet layer's output norm)
//
// y has the bits of the ggml chain the graph builds otherwise (and of fused.cu's BF16 kernels,
// which compute the same values before their cast): thread tid holds columns tid, tid + BLOCK,
// ... as ggml-cuda's rms_norm_f32 does, sums in its order (BLOCK 1024 from 1024 columns on,
// 256 below; per head the 256-thread order of gated_norm_kernel), the same expressions, and the
// same compiler flags (--use_fast_math, as ggml-cuda). The quantization is fp8_quant_dyn's
// arithmetic (max is exact, so the order of the amax does not matter): the same codes and
// scales as the chain followed by fp8_quant_dyn.
//
// Launched on the stream of ggml's CUDA backend, between graph segments (src/ggml/external.rs).

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

struct legion_fp8_act {
    int32_t kind;      // FP8_ACT_* below
    // plain, rms_norm: row t at x + t * x_ld; gate: element i of token t at
    // x + t * x_ld + (i / d) * x_hd + i % d; gated_norm: o, row t at x + t * x_ld
    const float * x;
    int64_t x_ld, x_hd;
    const float * w;   // rms_norm: [k]; gated_norm: [d]
    const float * g;   // gate: [k] of token t at g + t * g_ld; gated_norm: z, the same
    int64_t g_ld;
    uint8_t * out;     // E4M3 codes [k, n] contiguous
    float * scales;    // [n]
    int32_t k, n;
    int32_t d;         // gate: the head width; gated_norm: head_v
    int32_t silu;      // gate, gated_norm: SiLU instead of sigmoid
    float eps;         // rms_norm, gated_norm
};

// 0 on success; 1 for a shape this file does not take (the caller builds the unfused ops);
// 2 for a launch error.
int legion_fp8_act_launch(const legion_fp8_act * p, void * stream);

}  // extern "C"

namespace {

constexpr int FP8_ACT_PLAIN = 0;
constexpr int FP8_ACT_RMS_NORM = 1;
constexpr int FP8_ACT_GATE = 2;
constexpr int FP8_ACT_GATED_NORM = 3;

// Most values per thread (columns per row: 24 * 1024).
constexpr int MAX_NV = 24;

// ggml-cuda's warp_reduce_sum (common.cuh): butterfly over the 32 lanes. Every lane ends with
// the same bits (each step adds the same two values in either order).
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

// sigmoid or SiLU as fused.cu's gate_bf16_kernel and gated_norm_kernel compute them.
__device__ __forceinline__ float gate_of(float v, int silu) {
    return silu ? v / (1.0f + expf(-v)) : 1.0f / (1.0f + expf(-v));
}

template <int BLOCK, int NV, int KIND>
__global__ void __launch_bounds__(BLOCK) fp8_act_kernel(const legion_fp8_act p) {
    __shared__ float s_sum[KIND == FP8_ACT_GATED_NORM ? NV : 1][32];
    __shared__ float s_max[32];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32, warp = tid / 32;
    const int k = p.k;
    float v[NV];

    if constexpr (KIND == FP8_ACT_PLAIN || KIND == FP8_ACT_RMS_NORM) {
        const float * row = p.x + t * p.x_ld;
        float tmp = 0.0f;
#pragma unroll
        for (int i = 0; i < NV; i++) {
            const int col = tid + i * BLOCK;
            v[i] = 0.0f;
            if (col < k) {
                const float xi = row[col];
                v[i] = xi;
                if constexpr (KIND == FP8_ACT_RMS_NORM) {
                    tmp += xi * xi;
                }
            }
        }
        if constexpr (KIND == FP8_ACT_RMS_NORM) {
            // ggml-cuda's rms_norm_f32: block_reduce (warp sums, then warp 0's sum of them).
            tmp = warp_sum(tmp);
            if (BLOCK > 32) {
                if (lane == 0) {
                    s_sum[0][warp] = tmp;
                }
                __syncthreads();
                tmp = lane < BLOCK / 32 ? s_sum[0][lane] : 0.0f;
                tmp = warp_sum(tmp);
            }
            const float mean = tmp / k;
            const float scale = rsqrtf(mean + p.eps);
#pragma unroll
            for (int i = 0; i < NV; i++) {
                const int col = tid + i * BLOCK;
                if (col < k) {
                    v[i] = scale * v[i] * p.w[col];
                }
            }
        }
    } else if constexpr (KIND == FP8_ACT_GATE) {
#pragma unroll
        for (int i = 0; i < NV; i++) {
            const int col = tid + i * BLOCK;
            v[i] = 0.0f;
            if (col < k) {
                const float xv = p.x[t * p.x_ld + (int64_t) (col / p.d) * p.x_hd + col % p.d];
                const float a = gate_of(xv, p.silu);
                v[i] = a * p.g[t * p.g_ld + col];
            }
        }
    } else {
        // gated_norm_kernel per head of d = head_v columns (a multiple of 32 dividing BLOCK):
        // the head of column tid + i * BLOCK is (tid + i * BLOCK) / d, its column tid % d, and
        // its d / 32 warps are the warps of gated_norm_kernel's 256-thread block that hold
        // values (the others hold zeros), so the warp sums and their sum are that kernel's.
        static_assert(KIND == FP8_ACT_GATED_NORM, "kind");
        const int d = p.d, wph = d / 32, c = tid % d;
        const int first = (tid / d) * wph;  // the head's first warp in this block
        const float * o = p.x + t * p.x_ld;
#pragma unroll
        for (int i = 0; i < NV; i++) {
            const int col = tid + i * BLOCK;
            v[i] = col < k ? o[col] : 0.0f;
            float tmp = 0.0f;
            tmp += v[i] * v[i];
            tmp = warp_sum(tmp);
            if (lane == 0) {
                s_sum[i][warp] = tmp;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NV; i++) {
            const int col = tid + i * BLOCK;
            float tmp = lane < wph ? s_sum[i][first + lane] : 0.0f;
            tmp = warp_sum(tmp);
            const float mean = tmp / d;
            const float scale = rsqrtf(mean + p.eps);
            if (col < k) {
                const float y = scale * v[i] * p.w[c];
                const float g = gate_of(p.g[t * p.g_ld + col], p.silu);
                v[i] = y * g;
            }
        }
    }

    // fp8_quant_dyn's quantization of the row.
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = warp_max(amax);
    if (BLOCK > 32) {
        if (lane == 0) {
            s_max[warp] = amax;
        }
        __syncthreads();
        amax = s_max[0];
#pragma unroll
        for (int w = 1; w < BLOCK / 32; w++) {
            amax = fmaxf(amax, s_max[w]);
        }
    }
    const float s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 0.0f;
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    uint8_t * out = p.out + t * k;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int col = tid + i * BLOCK;
        if (col < k) {
            out[col] = __nv_cvt_float_to_fp8(__fmul_rn(v[i], inv), __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (tid == 0) {
        p.scales[t] = s;
    }
}

// fp8_act_kernel<BLOCK, NV, KIND> for the smallest NV of 4, 6, 8, 12, 18, 24 (at most MAXV)
// that holds a row of k values; 1 if none does.
template <int BLOCK, int KIND, int MAXV>
int launch_nv(const legion_fp8_act & p, cudaStream_t st) {
    static_assert(MAXV == 4 || MAXV == MAX_NV, "buckets");
    const int need = (p.k + BLOCK - 1) / BLOCK;
    const dim3 grid((unsigned) p.n);
    if (need <= 4) {
        fp8_act_kernel<BLOCK, 4, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else if constexpr (MAXV == 4) {
        return 1;
    } else if (need <= 6) {
        fp8_act_kernel<BLOCK, 6, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else if (need <= 8) {
        fp8_act_kernel<BLOCK, 8, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else if (need <= 12) {
        fp8_act_kernel<BLOCK, 12, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else if (need <= 18) {
        fp8_act_kernel<BLOCK, 18, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else if (need <= MAX_NV) {
        fp8_act_kernel<BLOCK, MAX_NV, KIND><<<grid, BLOCK, 0, st>>>(p);
    } else {
        return 1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace

int legion_fp8_act_launch(const legion_fp8_act * p, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    if (p->k < 1 || p->n < 0) {
        return 1;
    }
    if (p->n == 0) {
        return 0;
    }
    switch (p->kind) {
        case FP8_ACT_PLAIN:
            // Any order: small blocks (several per SM) up to 256 * MAX_NV columns.
            return p->k <= 256 * MAX_NV ? launch_nv<256, FP8_ACT_PLAIN, MAX_NV>(*p, st)
                                        : launch_nv<1024, FP8_ACT_PLAIN, MAX_NV>(*p, st);
        case FP8_ACT_RMS_NORM:
            // ggml-cuda's rms_norm block: 256 threads below 1024 columns, 1024 from there on.
            return p->k < 1024 ? launch_nv<256, FP8_ACT_RMS_NORM, 4>(*p, st)
                               : launch_nv<1024, FP8_ACT_RMS_NORM, MAX_NV>(*p, st);
        case FP8_ACT_GATE:
            if (p->d < 1) {
                return 1;
            }
            return p->k <= 256 * MAX_NV ? launch_nv<256, FP8_ACT_GATE, MAX_NV>(*p, st)
                                        : launch_nv<1024, FP8_ACT_GATE, MAX_NV>(*p, st);
        case FP8_ACT_GATED_NORM:
            // Heads of whole warps that tile the 1024-thread block, at most gated_norm_kernel's
            // 256 columns.
            if (p->d < 32 || p->d > 256 || p->d % 32 != 0 || 1024 % p->d != 0 || p->k % p->d != 0) {
                return 1;
            }
            return launch_nv<1024, FP8_ACT_GATED_NORM, MAX_NV>(*p, st);
        default:
            return 1;
    }
}
