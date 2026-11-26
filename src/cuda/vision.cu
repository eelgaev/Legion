// The Qwen3-VL vision tower (`qwen3vl_merger`) on the GPU (#1195): kernels only, launched from
// Rust by `src/vision/gpu.rs` from the `legion_vision_kernels` table with `cudaLaunchKernel`.
//
// - `gemm_kernel`: C = alpha * A B^T (+ bias, then GELU or + a residual), A [m][k] and B [n][k]
//   F16 rows, F32 accumulators, F32 or F16 out, batched. Every linear of the tower and both
//   products of attention. WMMA 16x16x16 (valid on every sm_70+ target), 128x128 tiles, k steps
//   of 32 double-buffered through registers.
//
// Every kernel takes a 1-D grid of any size and loops over its work items (tiles, patches, rows)
// at a stride of the grid, in the order a grid of one block per item would place them. The host
// caps the grid (src/vision/gpu.rs) so an encode holds only part of the GPU and the language
// model's kernels beside it find room at once; an item's math does not depend on the grid.
// - `im2col_kernel`: an image's patches as F16 rows, in merge order (each 2x2 block's four
//   patches next to each other, blocks row by row).
// - `pos_add_kernel`: the patch bias and the learned position table, resampled bilinearly with
//   corners aligned (ggml's `upscale` arithmetic) when the grid is not the table's.
// - `norm_kernel`: LayerNorm (ggml's `norm`, then the weight and bias) to F16 rows.
// - `rope_kernel`: ggml's `GGML_ROPE_TYPE_VISION` rotation of q and k (cos / sin from a table
//   the host computes as ggml's CPU does), q and k to [head][patch][d], v to [head][d][patch].
// - `softmax_kernel`: rows of attention scores to F16 probabilities.
//
// Gemma 4's tower (`gemma4v`, #1217) shares the GEMM and the softmax and has its own: patches
// in raster order scaled to [-1, 1], the column and row position tables, RMS norms (to F16 rows,
// or a post-norm added to the residual), per-head q / k / v norms with the half-and-half NEOX
// rotation, the gated GELU and the 3x3 average pool with its standardization and norm; the
// smaller towers' clipped linears (#1220) clamp their inputs and products in place.
//
// GLM-5.3-Flash's tower (`glm5v`, #1219) takes Qwen3-VL's patches and rotation with Gemma 4's
// RMS norm, and has its own per-head q / k norms before the rotation, the clamped SwiGLU and the
// merger's LayerNorm with an exact GELU.
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
    int            batch;  // products, at strides sa, sb, sc
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
    int            n, e, heads, d, gx, gy, side, p, w_px, h_px, kpad, chunk, cm, pool;
    float          eps, sfx, sfy;
    int            ld;   // im2col's row stride (0: 3 p p), padded with zeros
    float          lim;  // the SwiGLU clamp
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
// One 128x128 tile of product z = `tile / (tx * ty)`: rows m0 = `tile / tx % ty * BM`, columns
// n0 = `tile % tx * BN`.
__device__ __forceinline__ void gemm_tile(const Gemm & g, long long tile, int tx, int ty,
                                          __half (&sa)[2][BM * LDS], __half (&sb)[2][BN * LDS]) {
    using namespace nvcuda;
    const int       tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int       wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps, 64 x 32 each
    const long long z  = tile / ((long long) tx * ty);
    const __half *  A  = g.a + z * g.sa;
    const __half *  B  = g.b + z * g.sb;
    const int       m0 = (int) (tile / tx % ty) * BM, n0 = (int) (tile % tx) * BN;

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
}
#endif

__global__ void __launch_bounds__(256) gemm_kernel(const Gemm g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ __align__(128) __half sa[2][BM * LDS];
    __shared__ __align__(128) __half sb[2][BN * LDS];
    const int       tx = (g.n + BN - 1) / BN, ty = (g.m + BM - 1) / BM;
    const long long tiles = (long long) tx * ty * g.batch;
    // Tiles in the order of a (tx, ty, batch) grid: n fastest, then m, then the product.
    for (long long tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        if (tile != blockIdx.x) {
            __syncthreads();  // the last tile's epilogue is done with sa
        }
        gemm_tile(g, tile, tx, ty, sa, sb);
    }
#else
    __trap();
#endif
}

// A block per patch i: its pixels (x = image [3][h_px][w_px] F32) as row `i` of h0 [n][ld]
// (ld = 3 p p, or `o.ld` with zeros after the pixels), in the kernel's order (channel, then row,
// then column).
__global__ void im2col_kernel(const Op o) {
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        int row, col;
        patch_rc(i, o.gx, row, col);
        const int pp = o.p * o.p, kn = 3 * pp, ld = o.ld ? o.ld : kn;
        for (int k = threadIdx.x; k < ld; k += blockDim.x) {
            float v = 0.0f;
            if (k < kn) {
                const int c = k / pp, r = k % pp, ky = r / o.p, kx = r % o.p;
                v = o.x[((long long) c * o.h_px + row * o.p + ky) * o.w_px + col * o.p + kx];
            }
            o.h0[(long long) i * ld + k] = __float2half_rn(v);
        }
    }
}

// Patch i: y [n][e] (the patch products) += bias, then + the position table `w` ([side * side][e])
// at the patch, resampled when the grid is not side x side: ggml's bilinear `upscale` with corners
// aligned (scale factors sfx, sfy from the host).
__device__ __forceinline__ void pos_add_patch(const Op & o, int i) {
    int row, col;
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

// A block per patch: pos_add_patch.
__global__ void pos_add_kernel(const Op o) {
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        pos_add_patch(o, i);
    }
}

// A warp per row r of x [n][e]: ((x - mean) / sqrt(var + eps)) * w + b, to F16 row r of h0.
__device__ __forceinline__ void norm_row(const Op & o, int r) {
    const int lane = threadIdx.x & 31;
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

// A block per 8 rows (a warp each): norm_row.
__global__ void norm_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            norm_row(o, r);
        }
    }
}

// Patch i of x = qkv [n][3 e] (q, k, v of `heads` heads of d): q and k rotated by the patch's row
// (first d/4 pairs) and column (next d/4), pairs (j, j + d/2), cos / sin from cs [max(gx, gy)][d/4];
// q to h0 and k to h1 as [head][n][d], v to h2 as [head][d][kpad].
__device__ __forceinline__ void rope_patch(const Op & o, int i) {
    int row, col;
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

// A block per patch: rope_patch.
__global__ void rope_kernel(const Op o) {
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        rope_patch(o, i);
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

// Row r of scores x [heads * chunk][kpad] (n columns): softmax to F16 h0 (same layout), zeros in
// columns n..kpad. The whole block, `sh` its reduction scratch.
__device__ __forceinline__ void softmax_row(const Op & o, long long r, float * sh) {
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

// A block per row of the scores (the first cm rows of each head's chunk are live): softmax_row.
__global__ void softmax_kernel(const Op o) {
    __shared__ float sh[32];
    for (long long r = blockIdx.x; r < (long long) o.heads * o.chunk; r += gridDim.x) {
        if ((int) (r % o.chunk) < o.cm) {
            softmax_row(o, r, sh);
        }
    }
}

// Gemma 4's tower (`gemma4v`, #1217). Patches are in raster order (row by row). Like the
// kernels above, each loops over its items at the grid's stride.

// A block per patch i: its pixels scaled to [-1, 1] (2 x - 1) as row `i` of h0 [n][3 p p].
__global__ void g4_im2col_kernel(const Op o) {
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        const int row = i / o.gx, col = i % o.gx;
        const int pp = o.p * o.p, kn = 3 * pp;
        for (int k = threadIdx.x; k < kn; k += blockDim.x) {
            const int   c = k / pp, r = k % pp, ky = r / o.p, kx = r % o.p;
            const float v = o.x[((long long) c * o.h_px + row * o.p + ky) * o.w_px + col * o.p + kx];
            o.h0[(long long) i * kn + k] = __float2half_rn(v * 2.0f - 1.0f);
        }
    }
}

// A block per patch i: y [n][e] += the column table's row `col`, then the row table's row `row`
// (w = [2][side][e]: columns, then rows).
__global__ void g4_pos_kernel(const Op o) {
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        const int     row = i / o.gx, col = i % o.gx;
        float *       y  = o.y + (long long) i * o.e;
        const float * px = o.w + (long long) col * o.e;
        const float * py = o.w + ((long long) o.side + row) * o.e;
        for (int c = threadIdx.x; c < o.e; c += blockDim.x) {
            y[c] = (y[c] + px[c]) + py[c];
        }
    }
}

// A warp per row r of x [n][e]: x / sqrt(mean(x^2) + eps) (* w when there is one), to F16 row r
// of h0.
__device__ __forceinline__ void rms_row(const Op & o, int r) {
    const int     lane = threadIdx.x & 31;
    const float * x    = o.x + (long long) r * o.e;
    float         s    = 0.0f;
    for (int c = lane; c < o.e; c += 32) {
        s += x[c] * x[c];
    }
    const float scale = 1.0f / sqrtf(warp_sum(s) / o.e + o.eps);
    __half *    h     = o.h0 + (long long) r * o.e;
    for (int c = lane; c < o.e; c += 32) {
        h[c] = __float2half_rn(o.w ? (x[c] * scale) * o.w[c] : x[c] * scale);
    }
}

// A block per 8 rows (a warp each): rms_row.
__global__ void rms_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            rms_row(o, r);
        }
    }
}

// A warp per row r: y [n][e] += rms(x) * w, x [n][e] a block's output (the post-norms).
__device__ __forceinline__ void rms_add_row(const Op & o, int r) {
    const int     lane = threadIdx.x & 31;
    const float * x    = o.x + (long long) r * o.e;
    float *       y    = o.y + (long long) r * o.e;
    float         s    = 0.0f;
    for (int c = lane; c < o.e; c += 32) {
        s += x[c] * x[c];
    }
    const float scale = 1.0f / sqrtf(warp_sum(s) / o.e + o.eps);
    for (int c = lane; c < o.e; c += 32) {
        y[c] = (x[c] * scale) * o.w[c] + y[c];
    }
}

// A block per 8 rows (a warp each): rms_add_row.
__global__ void rms_add_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            rms_add_row(o, r);
        }
    }
}

// Patch i of x = qkv [n][3 e] (q, k, v of `heads` heads of d <= 128): q and k RMS-normed per head
// with their weights (w for q, b for k), v without; q and k then rotated, the first half of each
// head by the patch's column and the second by its row, NEOX pairs (j, j + d/4) within each half,
// cos / sin from cs [max(gx, gy)][d/4]; q to h0 and k to h1 as [head][n][d], v to h2 as
// [head][d][kpad]. A warp per head, a row of `sh` its scratch.
__device__ __forceinline__ void g4_qkv_patch(const Op & o, int i, float (&sh)[8][128]) {
    const int row = i / o.gx, col = i % o.gx;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int half = o.d / 2, q4 = o.d / 4;
    float *   s = sh[warp];
    for (int h = warp; h < o.heads; h += 8) {
#pragma unroll
        for (int which = 0; which < 3; ++which) {
            const float * src = o.x + (long long) i * 3 * o.e + which * o.e + h * o.d;
            float         ss  = 0.0f;
            for (int c = lane; c < o.d; c += 32) {
                const float v = src[c];
                s[c]          = v;
                ss += v * v;
            }
            const float   scale = 1.0f / sqrtf(warp_sum(ss) / o.d + o.eps);
            const float * nw    = which == 0 ? o.w : o.b;
            __syncwarp();
            for (int c = lane; c < o.d; c += 32) {
                s[c] = which == 2 ? s[c] * scale : (s[c] * scale) * nw[c];
            }
            __syncwarp();
            if (which == 2) {
                for (int c = lane; c < o.d; c += 32) {
                    o.h2[((long long) h * o.d + c) * o.kpad + i] = __float2half_rn(s[c]);
                }
            } else {
                __half * d = (which ? o.h1 : o.h0) + ((long long) h * o.n + i) * o.d;
                for (int t = lane; t < half; t += 32) {
                    const int    sec = t / q4, j = t % q4, base = sec * half;
                    const float2 c   = o.cs[(sec ? row : col) * q4 + j];
                    const float  x0 = s[base + j], x1 = s[base + j + q4];
                    d[base + j]      = __float2half_rn(x0 * c.x - x1 * c.y);
                    d[base + j + q4] = __float2half_rn(x0 * c.y + x1 * c.x);
                }
            }
            __syncwarp();
        }
    }
}

// A block per patch: g4_qkv_patch. The warps' last __syncwarp leaves `sh` free for the next.
__global__ void g4_qkv_kernel(const Op o) {
    __shared__ float sh[8][128];
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        g4_qkv_patch(o, i, sh);
    }
}

// A warp per row r of x [n][2 ff] F16 (the gate's ff columns, then the up's): gelu(gate) * up to
// F16 row r of h0 [n][ff] (ff = e).
__device__ __forceinline__ void geglu_row(const Op & o, int r) {
    const int      lane = threadIdx.x & 31;
    const __half * x    = o.h1 + (long long) r * 2 * o.e;
    __half *       y    = o.h0 + (long long) r * o.e;
    for (int c = lane; c < o.e; c += 32) {
        y[c] = __float2half_rn(gelu(__half2float(x[c])) * __half2float(x[o.e + c]));
    }
}

// A block per 8 rows (a warp each): geglu_row.
__global__ void geglu_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            geglu_row(o, r);
        }
    }
}

// Pooled row t (raster over the gx / pool x gy / pool grid): the mean of its pool x pool patches
// of x [n][e], * sqrt(e), (- b) * w (the standardization), into y [rows][e] F32, then RMS-normed
// without a weight to F16 rows of h0. The whole block, `sh` its reduction scratch.
__device__ __forceinline__ void g4_pool_row(const Op & o, int t, float * sh) {
    const int   ox = o.gx / o.pool, tx = t % ox, ty = t / ox;
    const float inv = 1.0f / (float) (o.pool * o.pool), se = sqrtf((float) o.e);
    float *     y  = o.y + (long long) t * o.e;
    float       ss = 0.0f;
    for (int c = threadIdx.x; c < o.e; c += blockDim.x) {
        float m = 0.0f;
        for (int dy = 0; dy < o.pool; ++dy) {
            for (int dx = 0; dx < o.pool; ++dx) {
                m += o.x[((long long) (ty * o.pool + dy) * o.gx + tx * o.pool + dx) * o.e + c];
            }
        }
        const float v = ((m * inv) * se - o.b[c]) * o.w[c];
        y[c]          = v;
        ss += v * v;
    }
    const float scale = 1.0f / sqrtf(block_reduce(ss, false, sh) / o.e + o.eps);
    for (int c = threadIdx.x; c < o.e; c += blockDim.x) {
        o.h0[(long long) t * o.e + c] = __float2half_rn(y[c] * scale);
    }
}

// A block per pooled row: g4_pool_row (block_reduce syncs before it reuses `sh`).
__global__ void g4_pool_kernel(const Op o) {
    __shared__ float sh[32];
    const int        rows = (o.gx / o.pool) * (o.gy / o.pool);
    for (int t = blockIdx.x; t < rows; t += gridDim.x) {
        g4_pool_row(o, t, sh);
    }
}

// GLM-5.3-Flash's tower (`glm5v`, #1219): Qwen3-VL's patches in merge order (im2col_kernel, rows
// padded to whole 8s: a 14-pixel patch is 588 values) and the GEMM, softmax and RMS norm above.

// Patch i (merge order) of x = qkv [n][3 e] (q, k, v of `heads` heads of d <= 128, the bias in):
// q and k RMS-normed per head with their weights (w for q, b for k), then rotated as rope_patch
// rotates them; v as it is. q to h0 and k to h1 as [head][n][d], v to h2 as [head][d][kpad]. A
// warp per head, a row of `sh` its scratch.
__device__ __forceinline__ void glm_qkv_patch(const Op & o, int i, float (&sh)[8][128]) {
    int row, col;
    patch_rc(i, o.gx, row, col);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int half = o.d / 2, quarter = o.d / 4;
    float *   s = sh[warp];
    for (int h = warp; h < o.heads; h += 8) {
        const float * v = o.x + (long long) i * 3 * o.e + 2 * o.e + h * o.d;
        for (int c = lane; c < o.d; c += 32) {
            o.h2[((long long) h * o.d + c) * o.kpad + i] = __float2half_rn(v[c]);
        }
#pragma unroll
        for (int which = 0; which < 2; ++which) {
            const float * src = o.x + (long long) i * 3 * o.e + which * o.e + h * o.d;
            float         ss  = 0.0f;
            for (int c = lane; c < o.d; c += 32) {
                const float x = src[c];
                s[c]          = x;
                ss += x * x;
            }
            const float   scale = 1.0f / sqrtf(warp_sum(ss) / o.d + o.eps);
            const float * nw    = which == 0 ? o.w : o.b;
            __syncwarp();
            for (int c = lane; c < o.d; c += 32) {
                s[c] = (s[c] * scale) * nw[c];
            }
            __syncwarp();
            __half * d = (which ? o.h1 : o.h0) + ((long long) h * o.n + i) * o.d;
            for (int j = lane; j < half; j += 32) {
                const float2 c  = j < quarter ? o.cs[row * quarter + j] : o.cs[col * quarter + j - quarter];
                const float  x0 = s[j], x1 = s[j + half];
                d[j]            = __float2half_rn(x0 * c.x - x1 * c.y);
                d[j + half]     = __float2half_rn(x0 * c.y + x1 * c.x);
            }
            __syncwarp();
        }
    }
}

// A block per patch: glm_qkv_patch. The warps' last __syncwarp leaves `sh` free for the next.
__global__ void glm_qkv_kernel(const Op o) {
    __shared__ float sh[8][128];
    for (int i = blockIdx.x; i < o.n; i += gridDim.x) {
        glm_qkv_patch(o, i, sh);
    }
}

// A warp per row r of x [n][2 ff] F16 (the gate's ff columns, then the up's): ggml's
// swiglu_clamp, silu(min(gate, lim)) * clamp(up, -lim, lim), to F16 row r of h0 [n][ff] (ff = e).
__device__ __forceinline__ void swiglu_clamp_row(const Op & o, int r) {
    const int      lane = threadIdx.x & 31;
    const __half * x    = o.h1 + (long long) r * 2 * o.e;
    __half *       y    = o.h0 + (long long) r * o.e;
    for (int c = lane; c < o.e; c += 32) {
        const float g = fminf(__half2float(x[c]), o.lim);
        const float u = fminf(fmaxf(__half2float(x[o.e + c]), -o.lim), o.lim);
        y[c]          = __float2half_rn(g / (1.0f + expf(-g)) * u);
    }
}

// A block per 8 rows (a warp each): swiglu_clamp_row.
__global__ void swiglu_clamp_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            swiglu_clamp_row(o, r);
        }
    }
}

// A warp per row r of x [n][e]: LayerNorm (as norm_row), then ggml's exact GELU
// (0.5 v (1 + erf(v / sqrt 2))), to F16 row r of h0.
__device__ __forceinline__ void ln_gelu_row(const Op & o, int r) {
    const int     lane = threadIdx.x & 31;
    const float * x    = o.x + (long long) r * o.e;
    float         s    = 0.0f;
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
        const float t = ((x[c] - mean) * scale) * o.w[c] + o.b[c];
        h[c]          = __float2half_rn(0.5f * t * (1.0f + erff(t * 0.70710678118654752440f)));
    }
}

// A block per 8 rows (a warp each): ln_gelu_row.
__global__ void ln_gelu_kernel(const Op o) {
    for (int g = blockIdx.x; g * 8 < o.n; g += gridDim.x) {
        const int r = g * 8 + (threadIdx.x >> 5);
        if (r < o.n) {
            ln_gelu_row(o, r);
        }
    }
}

// A clipped linear's clamp (#1220), in place: rows [rows][width] of F32 or F16 (`f16`), column c
// clamped to segment s = min(c / seg, 2)'s bounds, as ggml's `clamp` (min with hi, then max with
// lo). A stacked matrix's products (q, k, v; gate, up) each take their own segment.
struct Clamp {
    void * x;
    int    rows, width, seg, f16;
    float  lo[3], hi[3];
};

// A block per row.
__global__ void g4_clamp_kernel(const Clamp o) {
    for (int r = blockIdx.x; r < o.rows; r += gridDim.x) {
        for (int c = threadIdx.x; c < o.width; c += blockDim.x) {
            const int       s = min(c / o.seg, 2);
            const long long i = (long long) r * o.width + c;
            if (o.f16) {
                __half * p = static_cast<__half *>(o.x);
                p[i]       = __float2half_rn(fmaxf(fminf(__half2float(p[i]), o.hi[s]), o.lo[s]));
            } else {
                float * p = static_cast<float *>(o.x);
                p[i]      = fmaxf(fminf(p[i], o.hi[s]), o.lo[s]);
            }
        }
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of src/vision/gpu.rs's `Kernel`.
extern "C" const void * const legion_vision_kernels[17] = {
    reinterpret_cast<const void *>(&gemm_kernel),      reinterpret_cast<const void *>(&im2col_kernel),
    reinterpret_cast<const void *>(&pos_add_kernel),   reinterpret_cast<const void *>(&norm_kernel),
    reinterpret_cast<const void *>(&rope_kernel),      reinterpret_cast<const void *>(&softmax_kernel),
    reinterpret_cast<const void *>(&g4_im2col_kernel), reinterpret_cast<const void *>(&g4_pos_kernel),
    reinterpret_cast<const void *>(&rms_kernel),       reinterpret_cast<const void *>(&rms_add_kernel),
    reinterpret_cast<const void *>(&g4_qkv_kernel),    reinterpret_cast<const void *>(&geglu_kernel),
    reinterpret_cast<const void *>(&g4_pool_kernel),   reinterpret_cast<const void *>(&glm_qkv_kernel),
    reinterpret_cast<const void *>(&swiglu_clamp_kernel), reinterpret_cast<const void *>(&ln_gelu_kernel),
    reinterpret_cast<const void *>(&g4_clamp_kernel),
};
