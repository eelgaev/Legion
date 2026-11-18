// The Qwen3-VL vision tower (`qwen3vl_merger`) on the GPU (#1195): kernels only, launched from
// Rust by `src/vision/gpu.rs` from the `legion_vision_kernels` table with `cudaLaunchKernel`.
//
// - `gemm_kernel`: C = alpha * A B^T (+ bias, then GELU or + a residual), A [m][k] and B [n][k]
//   F16 rows, F32 accumulators, F32 or F16 out, batched over blockIdx.z. Every linear of the
//   tower and both products of attention. WMMA 16x16x16 (valid on every sm_70+ target), 128x128
//   tiles, k steps of 32 double-buffered through registers.
// - `im2col_kernel`: an image's patches as F16 rows, in merge order (each 2x2 block's four
//   patches next to each other, blocks row by row).
// - `pos_add_kernel`: the patch bias and the learned position table, resampled bilinearly with
//   corners aligned (ggml's `upscale` arithmetic) when the grid is not the table's.
// - `norm_kernel`: LayerNorm (ggml's `norm`, then the weight and bias) to F16 rows.
// - `rope_kernel`: ggml's `GGML_ROPE_TYPE_VISION` rotation of q and k (cos / sin from a table
//   the host computes as ggml's CPU does), q and k to [head][patch][d], v to [head][d][patch].
// - `softmax_kernel`: rows of attention scores to F16 probabilities.
//
// The CPU tower (`src/vision/qwen3vl.rs`, ggml's CPU backend) is the oracle; these kernels compute
// the same graph, not its bits (F16 products everywhere, as ggml's CPU rounds the activations of
// its F16 matrices, but F16 attention products and a GELU without ggml's F16 table).

#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

namespace {

struct Gemm {
    const __half * a;
    const __half * b;
    void *         c;
    const float *  bias;  // [n] or null
    const float *  res;   // [m][ldr] F32 or null (may be `c`: each element read, then written)
    long long      lda, ldb, ldc, ldr;
    long long      sa, sb, sc;  // per-batch strides, in elements
    int            m, n, k;     // k a multiple of 8, lda and ldb too
    float          alpha;
    int            flags;
};

enum { OUT_F16 = 1, GELU = 2 };

struct Op {
    const float *  x;
    const float *  w;
    const float *  b;
    const float2 * cs;
    float *        y;
    __half *       h0;
    __half *       h1;
    __half *       h2;
    int            n, e, heads, d, gx, gy, side, p, w_px, h_px, kpad, chunk, cm;
    float          eps, sfx, sfy;
};

constexpr int BM = 128, BN = 128, BK = 32, LDS = BK + 8;

// ggml's tanh GELU (ggml-cpu/vec.h, GELU_COEF_A and SQRT_2_OVER_PI), without its F16 table.
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// The (row, column) of patch `i` in merge order on a grid `gx` patches wide.
__device__ __forceinline__ void patch_rc(int i, int gx, int & row, int & col) {
    const int blk = i >> 2, q = i & 3, bw = gx >> 1;
    row = (blk / bw) * 2 + (q >> 1);
    col = (blk % bw) * 2 + (q & 1);
}

__global__ void __launch_bounds__(256) gemm_kernel(const Gemm g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ __align__(128) __half sa[2][BM * LDS];
    __shared__ __align__(128) __half sb[2][BN * LDS];
    const int       tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int       wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps, 64 x 32 each
    const long long z  = blockIdx.z;
    const __half *  A  = g.a + z * g.sa;
    const __half *  B  = g.b + z * g.sb;
    const int       m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    uint4      ra[2], rb[2];
    const auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * 256, r = v >> 2, kk = k0 + (v & 3) * 8;
            uint4     x = make_uint4(0, 0, 0, 0), y = make_uint4(0, 0, 0, 0);
            if (kk < g.k) {
                if (m0 + r < g.m) {
                    x = *reinterpret_cast<const uint4 *>(A + (long long) (m0 + r) * g.lda + kk);
                }
                if (n0 + r < g.n) {
                    y = *reinterpret_cast<const uint4 *>(B + (long long) (n0 + r) * g.ldb + kk);
                }
            }
            ra[i] = x;
            rb[i] = y;
        }
    };
    const auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * 256, r = v >> 2, c = (v & 3) * 8;
            *reinterpret_cast<uint4 *>(&sa[buf][r * LDS + c]) = ra[i];
            *reinterpret_cast<uint4 *>(&sb[buf][r * LDS + c]) = rb[i];
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    const int nk = (g.k + BK - 1) / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < nk; ++t) {
        const int cur = t & 1;
        if (t + 1 < nk) {
            load((t + 1) * BK);
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(fa[i], &sa[cur][(wm * 64 + i * 16) * LDS + kk], LDS);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(fb[j], &sb[cur][(wn * 32 + j * 16) * LDS + kk], LDS);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
                }
            }
        }
        if (t + 1 < nk) {
            store(cur ^ 1);
        }
        __syncthreads();
    }

    // Each warp's 16x16 tiles through 1 KB of the (now idle) A buffer, then the epilogue.
    float * st = reinterpret_cast<float *>(&sa[0][0]) + warp * 256;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int m = m0 + wm * 64 + i * 16 + (e >> 4), n = n0 + wn * 32 + j * 16 + (e & 15);
                if (m < g.m && n < g.n) {
                    float v = st[e] * g.alpha;
                    if (g.bias) {
                        v += g.bias[n];
                    }
                    if (g.flags & GELU) {
                        v = gelu(v);
                    }
                    if (g.res) {
                        v += g.res[(long long) m * g.ldr + n];
                    }
                    const long long at = z * g.sc + (long long) m * g.ldc + n;
                    if (g.flags & OUT_F16) {
                        static_cast<__half *>(g.c)[at] = __float2half_rn(v);
                    } else {
                        static_cast<float *>(g.c)[at] = v;
                    }
                }
            }
            __syncwarp();
        }
    }
#else
    __trap();
#endif
}

// One block per patch: its pixels (x = image [3][h_px][w_px] F32) as row `i` of h0 [n][3 p p],
// in the kernel's order (channel, then row, then column).
__global__ void im2col_kernel(const Op o) {
    const int i = blockIdx.x;
    int       row, col;
    patch_rc(i, o.gx, row, col);
    const int pp = o.p * o.p, kn = 3 * pp;
    for (int k = threadIdx.x; k < kn; k += blockDim.x) {
        const int c = k / pp, r = k % pp, ky = r / o.p, kx = r % o.p;
        const float v = o.x[((long long) c * o.h_px + row * o.p + ky) * o.w_px + col * o.p + kx];
        o.h0[(long long) i * kn + k] = __float2half_rn(v);
    }
}

// One block per patch: y [n][e] (the patch products) += bias, then + the position table `w`
// ([side * side][e]) at the patch, resampled when the grid is not side x side: ggml's bilinear
// `upscale` with corners aligned (scale factors sfx, sfy from the host).
__global__ void pos_add_kernel(const Op o) {
    const int i = blockIdx.x;
    int       row, col;
    patch_rc(i, o.gx, row, col);
    float * y = o.y + (long long) i * o.e;
    if (o.gx == o.side && o.gy == o.side) {
        const float * p = o.w + ((long long) row * o.side + col) * o.e;
        for (int c = threadIdx.x; c < o.e; c += blockDim.x) {
            y[c] = (y[c] + o.b[c]) + p[c];
        }
        return;
    }
    const float fx = __fdiv_rn((float) col, o.sfx), fy = __fdiv_rn((float) row, o.sfy);
    int         x0 = (int) floorf(fx), y0 = (int) floorf(fy);
    int         x1 = x0 + 1, y1 = y0 + 1;
    x0             = max(0, min(x0, o.side - 1));
    x1             = max(0, min(x1, o.side - 1));
    y0             = max(0, min(y0, o.side - 1));
    y1             = max(0, min(y1, o.side - 1));
    const float dx = fmaxf(0.0f, fminf(fx - (float) x0, 1.0f));
    const float dy = fmaxf(0.0f, fminf(fy - (float) y0, 1.0f));
    const float * pa = o.w + ((long long) y0 * o.side + x0) * o.e;
    const float * pb = o.w + ((long long) y0 * o.side + x1) * o.e;
    const float * pc = o.w + ((long long) y1 * o.side + x0) * o.e;
    const float * pd = o.w + ((long long) y1 * o.side + x1) * o.e;
    for (int c = threadIdx.x; c < o.e; c += blockDim.x) {
        const float v = pa[c] * (1 - dx) * (1 - dy) + pb[c] * dx * (1 - dy) + pc[c] * (1 - dx) * dy +
                        pd[c] * dx * dy;
        y[c] = (y[c] + o.b[c]) + v;
    }
}

// A warp per row of x [n][e]: ((x - mean) / sqrt(var + eps)) * w + b, to F16 rows of h0.
__global__ void norm_kernel(const Op o) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= o.n) {
        return;
    }
    const float * x = o.x + (long long) r * o.e;
    float         s = 0.0f;
    for (int c = lane; c < o.e; c += 32) {
        s += x[c];
    }
    const float mean = warp_sum(s) / o.e;
    float       v    = 0.0f;
    for (int c = lane; c < o.e; c += 32) {
        const float d = x[c] - mean;
        v += d * d;
    }
    const float scale = 1.0f / sqrtf(warp_sum(v) / o.e + o.eps);
    __half *    h     = o.h0 + (long long) r * o.e;
    for (int c = lane; c < o.e; c += 32) {
        h[c] = __float2half_rn(((x[c] - mean) * scale) * o.w[c] + o.b[c]);
    }
}

// One block per patch of x = qkv [n][3 e] (q, k, v of `heads` heads of d): q and k rotated by
// the patch's row (first d/4 pairs) and column (next d/4), pairs (j, j + d/2), cos / sin from
// cs [max(gx, gy)][d/4]; q to h0 and k to h1 as [head][n][d], v to h2 as [head][d][kpad].
__global__ void rope_kernel(const Op o) {
    const int i = blockIdx.x;
    int       row, col;
    patch_rc(i, o.gx, row, col);
    const int     half = o.d / 2, quarter = o.d / 4;
    const float * src  = o.x + (long long) i * 3 * o.e;
    for (int t = threadIdx.x; t < o.heads * half; t += blockDim.x) {
        const int    h = t / half, j = t % half;
        const float2 c = j < quarter ? o.cs[row * quarter + j] : o.cs[col * quarter + j - quarter];
#pragma unroll
        for (int which = 0; which < 2; ++which) {
            const float * s  = src + which * o.e + h * o.d;
            const float   x0 = s[j], x1 = s[j + half];
            __half *      d  = (which ? o.h1 : o.h0) + ((long long) h * o.n + i) * o.d;
            d[j]             = __float2half_rn(x0 * c.x - x1 * c.y);
            d[j + half]      = __float2half_rn(x0 * c.y + x1 * c.x);
        }
        const float * v = src + 2 * o.e + h * o.d;
        o.h2[((long long) h * o.d + j) * o.kpad + i]        = __float2half_rn(v[j]);
        o.h2[((long long) h * o.d + j + half) * o.kpad + i] = __float2half_rn(v[j + half]);
    }
}

__device__ float block_reduce(float v, bool is_max, float * sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = is_max ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) {
        sh[warp] = v;
    }
    __syncthreads();
    v = lane < nw ? sh[lane] : (is_max ? -INFINITY : 0.0f);
    return is_max ? warp_max(v) : warp_sum(v);
}

// One block per row of scores x [heads * chunk][kpad] (the first cm rows of each head's chunk
// are live, n columns): softmax to F16 h0 (same layout), zeros in columns n..kpad.
__global__ void softmax_kernel(const Op o) {
    __shared__ float sh[32];
    const long long  r = blockIdx.x;
    if ((int) (r % o.chunk) >= o.cm) {
        return;
    }
    const float * x  = o.x + r * o.kpad;
    __half *      y  = o.h0 + r * o.kpad;
    float         mx = -INFINITY;
    for (int c = threadIdx.x; c < o.n; c += blockDim.x) {
        mx = fmaxf(mx, x[c]);
    }
    mx      = block_reduce(mx, true, sh);
    float s = 0.0f;
    for (int c = threadIdx.x; c < o.n; c += blockDim.x) {
        s += expf(x[c] - mx);
    }
    s               = block_reduce(s, false, sh);
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < o.kpad; c += blockDim.x) {
        y[c] = c < o.n ? __float2half_rn(expf(x[c] - mx) * inv) : __float2half_rn(0.0f);
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of src/vision/gpu.rs's `Kernel`.
extern "C" const void * const legion_vision_kernels[6] = {
    reinterpret_cast<const void *>(&gemm_kernel),    reinterpret_cast<const void *>(&im2col_kernel),
    reinterpret_cast<const void *>(&pos_add_kernel), reinterpret_cast<const void *>(&norm_kernel),
    reinterpret_cast<const void *>(&rope_kernel),    reinterpret_cast<const void *>(&softmax_kernel),
};
