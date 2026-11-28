// A MoE layer's router logits `w @ x` in Legion's own kernel (#1183; semantics, the CPU reference
// and the switch in src/model/core/legion_router.rs), for the steps that ggml-cuda runs on cuBLAS
// (F32 weights, more than 16 tokens: prompt steps and big verify steps). cuBLAS picks its Sgemm
// algorithm from the token count, so the same token's logits differ between a 2,086-token step
// and a 6,182-token one; here they do not.
//
// The weights are the router's BF16 copy (every value of the F32 router exact in BF16,
// src/model/core/router_bf16.rs). Each activation is split into two BF16 values, hi = bf16(x)
// and lo = bf16(x - hi) (x - hi is exact in F32), so a weight times hi or lo is an exact F32
// product and x is carried to ~16 bits (cuBLAS's TF32 keeps 10). Per 16-deep slice of K, in K
// order: t = mma(w, lo, 0), t = mma(w, hi, t) (mma.sync m16n8k16 BF16 x BF16 -> F32), then
// acc += t in F32. An mma's output element depends only on its own row of A and column of B, so
// every output is the same chain of operations on the same values whatever the step's token
// count, the tile shape or the output's place in the tile: the bits of a token's logits do not
// depend on the step it is in.

#include "device_cache.cuh"

#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>

extern "C" {

struct legion_router_mm {
    const uint16_t * w;  // BF16 [n_expert][k_embd]: row e at w + e * w_ld
    int64_t w_ld;
    const float * x;  // F32 [n_tok][k_embd]: token t at x + t * x_ld
    int64_t x_ld;
    float * out;  // F32 [n_tok][n_expert], contiguous
    int32_t k_embd, n_expert, n_tok;
};

int legion_router_mm_launch(const legion_router_mm * p, void * stream);

// Whether the current device runs the kernel (compute capability 8.0+ and this binary's sm_80+
// code for it).
int legion_router_mm_supported(void);

}  // extern "C"

namespace {

// BM expert rows x BN tokens a block, warps of WM x WN, K steps of BK through an S-stage
// cp.async ring.
template <int BM, int BN, int BK, int S, int WM, int WN> struct Shape {
    static constexpr int NWM = BM / WM, NWN = BN / WN, NT = 32 * NWM * NWN;
    static constexpr int MI = WM / 16, NI = WN / 8;
    static constexpr int LW = BK + 8;  // BF16 values per weight row in shared memory
    static constexpr int LX = BK + 8;  // floats per token row in shared memory
    static constexpr int WST = BM * LW * 2, XST = BN * LX * 4;
    static constexpr int SMEM = S * (WST + XST);
    static constexpr int WP = BM * BK / 8;  // 16-byte pieces of a weight stage
    static constexpr int XP = BN * BK / 4;  // 16-byte pieces of a token stage
    static_assert(WM % 16 == 0 && WN % 8 == 0 && BK % 16 == 0, "whole fragments");
    static_assert(WP % NT == 0 && XP % NT == 0, "whole copies per thread");
    static_assert(S >= 2, "a stage in flight while one is read");
};

__device__ __forceinline__ void cp16(void * s, const void * g, bool ok) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(g), "r"(ok ? 16 : 0));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Two activations (the lower k in the low half) as BF16 pairs hi and lo: x = hi + lo + O(2^-17 x).
__device__ __forceinline__ void split2(float2 v, uint32_t & hi, uint32_t & lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(v.x, v.y);
    const float2 hf = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(v.x - hf.x, v.y - hf.y);
    hi = *(const uint32_t *) &h;
    lo = *(const uint32_t *) &l;
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                         const float (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]),
          "f"(c[3]));
}
#endif

template <int BM, int BN, int BK, int S, int WM, int WN>
__global__ void __launch_bounds__(Shape<BM, BN, BK, S, WM, WN>::NT, 1) router_mm(const legion_router_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using G = Shape<BM, BN, BK, S, WM, WN>;
    constexpr int MI = G::MI, NI = G::NI, LW = G::LW, LX = G::LX, NT = G::NT;
    extern __shared__ __align__(16) char rm_sm[];
    uint16_t * wsm = (uint16_t *) rm_sm;              // [S][BM][LW]
    float * xsm = (float *) (rm_sm + S * G::WST);     // [S][BN][LX]
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    // Row tiles fastest: the blocks that run together read the same tokens.
    const int tm = (p.n_expert + BM - 1) / BM;
    const int64_t m0 = (int64_t) (blockIdx.x % tm) * BM, n0 = (int64_t) (blockIdx.x / tm) * BN;
    const int wm = warp / G::NWN, wn = warp % G::NWN;
    auto load = [&](int64_t k0, int s) {
#pragma unroll
        for (int j = 0; j < G::WP / NT; j++) {
            const int c = tid + NT * j, row = c / (BK / 8), q = c % (BK / 8);
            const bool ok = m0 + row < p.n_expert;
            cp16(wsm + (s * BM + row) * LW + 8 * q, p.w + (ok ? m0 + row : 0) * p.w_ld + k0 + 8 * q, ok);
        }
#pragma unroll
        for (int j = 0; j < G::XP / NT; j++) {
            const int c = tid + NT * j, tok = c / (BK / 4), q = c % (BK / 4);
            const bool ok = n0 + tok < p.n_tok;
            cp16(xsm + (s * BN + tok) * LX + 4 * q, p.x + (ok ? n0 + tok : 0) * p.x_ld + k0 + 4 * q, ok);
        }
    };
    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++) {
#pragma unroll
        for (int b = 0; b < NI; b++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[a][b][e] = 0.0f;
            }
        }
    }
    const int nk = p.k_embd / BK;
#pragma unroll
    for (int s = 0; s < S - 1; s++) {
        if (s < nk) {
            load((int64_t) s * BK, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int kt = 0; kt < nk; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
        __syncthreads();
        if (kt + S - 1 < nk) {
            load((int64_t) (kt + S - 1) * BK, (kt + S - 1) % S);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        const int s = kt % S;
        const uint16_t * ws = wsm + s * BM * LW;
        const float * xs = xsm + s * BN * LX;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t fa[MI][4], hi[NI][2], lo[NI][2];
#pragma unroll
            for (int a = 0; a < MI; a++) {
                const uint16_t * pa = ws + (WM * wm + 16 * a + lane % 16) * LW + kk + 8 * (lane / 16);
                const uint32_t sa = (uint32_t) __cvta_generic_to_shared(pa);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(fa[a][0]), "=r"(fa[a][1]), "=r"(fa[a][2]), "=r"(fa[a][3])
                             : "r"(sa));
            }
#pragma unroll
            for (int b = 0; b < NI; b++) {
                const float * px = xs + (WN * wn + 8 * b + lane / 4) * LX + kk + 2 * (lane % 4);
                split2(*(const float2 *) px, hi[b][0], lo[b][0]);
                split2(*(const float2 *) (px + 8), hi[b][1], lo[b][1]);
            }
#pragma unroll
            for (int a = 0; a < MI; a++) {
#pragma unroll
                for (int b = 0; b < NI; b++) {
                    float t[4], u[4];
                    mma_bf16(t, fa[a], lo[b][0], lo[b][1], zero);
                    mma_bf16(u, fa[a], hi[b][0], hi[b][1], t);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        acc[a][b][e] += u[e];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int a = 0; a < MI; a++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t row = m0 + WM * wm + 16 * a + lane / 4 + 8 * h;
            if (row >= p.n_expert) {
                continue;
            }
#pragma unroll
            for (int b = 0; b < NI; b++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int64_t tok = n0 + WN * wn + 8 * b + 2 * (lane % 4) + e;
                    if (tok < p.n_tok) {
                        p.out[tok * p.n_expert + row] = acc[a][b][2 * h + e];
                    }
                }
            }
        }
    }
#else
    __trap();
#endif
}

// Launches router_mm<...> for `p`; false when the device refuses the shared memory or the grid
// does not fit.
template <int BM, int BN, int BK, int S, int WM, int WN> bool launch(const legion_router_mm & p, cudaStream_t st) {
    using G = Shape<BM, BN, BK, S, WM, WN>;
    static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES];
    if (p.k_embd % BK != 0 || !legion_device::smem_opt_in(smem, router_mm<BM, BN, BK, S, WM, WN>, G::SMEM)) {
        return false;
    }
    const int64_t tiles = ((p.n_expert + BM - 1) / BM) * (((int64_t) p.n_tok + BN - 1) / BN);
    if (tiles > INT32_MAX) {
        return false;
    }
    router_mm<BM, BN, BK, S, WM, WN><<<(unsigned) tiles, G::NT, G::SMEM, st>>>(p);
    return true;
}

}  // namespace

int legion_router_mm_supported(void) {
    static std::atomic<int> state[GGML_CUDA_MAX_DEVICES];  // 0 unknown, 1 yes, 2 no
    const int dev = legion_device::current();
    if (dev < 0) {
        return 0;
    }
    int st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        st = legion_device::compute_capability() >= 80 &&
                     legion_device::sm80_code(router_mm<128, 128, 32, 3, 64, 32>) ?
                 1 :
                 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1 ? 1 : 0;
}

int legion_router_mm_launch(const legion_router_mm * p, void * stream) {
    // 16-byte rows: whole cp.async pieces from aligned starts.
    if (p->n_tok < 1 || p->n_expert < 1 || p->k_embd < 64 || p->k_embd % 64 != 0 || p->w_ld % 8 != 0 ||
        p->x_ld % 4 != 0 || (uintptr_t) p->w % 16 != 0 || (uintptr_t) p->x % 16 != 0) {
        return -1;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    // The shape changes the speed only (see the top). Few tokens (verify steps, short prompts):
    // small row tiles, so more blocks read the weights at once; prompt steps: linear_tc's
    // 128 x 128 tiles of 8 warps (glm5next.cu).
    bool ok;
    if (p->n_tok <= 64) {
        ok = launch<16, 64, 64, 4, 16, 16>(*p, s);
    } else if (p->n_tok <= 1024) {
        ok = launch<64, 64, 32, 3, 32, 32>(*p, s);
    } else {
        ok = launch<128, 128, 32, 3, 64, 32>(*p, s);
    }
    if (!ok) {
        return -2;
    }
    return (int) cudaGetLastError();
}
