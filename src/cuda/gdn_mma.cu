// gdn_mma (#781, #790): the chunked tensor-core gated delta rule for prompt groups whose head
// counts ggml-cuda keeps on its per-token kernel (a tensor parallel rank's 8 key / 24 value
// heads of Qwen3.8-Flash-Next). The Rust side (src/model/core/gdn_mma.rs) decides when it runs,
// picks the schedule (whole heads or value slices, by SM count), gives the kernels their shared
// memory and launches them with cudaLaunchKernel from the table at the end. This file is the
// kernels only.
//
// Ported from ggml (third_party/ggml/src/ggml-cuda/gated-delta-net-mma.cu, the kernel
// `gdn_single`, its `matrix`/`shared` layouts and `load`/`product`/`store`; and from mma.cuh the
// pieces it uses on NVIDIA sm_80+: the m16n8k16 BF16 `tile`s, `swizzle`, the swizzled
// `ldmatrix` loads and the BF16 `mma`). The arithmetic and its order are ggml's, unchanged, so
// the bits are those of PR #789, which compiled ggml's file in this place with the same flags.
// The HIP/MUSA/Volta branches are left out; nothing here includes ggml-cuda.
//
// ggml is MIT licensed:
//
//   MIT License
//
//   Copyright (c) 2023-2026 The ggml authors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define LEGION_GDN_MMA_AVAILABLE
#endif

extern "C" {

// The kernel's argument (ggml_cuda_gdn_mma_args without `eligible`), filled by gdn_mma.rs's
// `Args` (the same layout, checked by the static_assert below and a Rust test). q/k element
// (i, head, t, seq) at i + head * sq1 + t * sq2 + (seq / rq3) * sq3 floats, key head h % H_k for
// value head h; v (i, h, t, seq) at i + h * sv1 + t * sv2 + seq * sv3; g and beta (h, t, seq) at
// h * sb1 + t * sb2 + seq * sb3; state and state_out [128, 128, H, n_seqs] contiguous; dst
// [128 * H, n_tokens * n_seqs].
struct legion_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
};

}  // extern "C"

static_assert(sizeof(legion_gdn_mma_args) == 184 && offsetof(legion_gdn_mma_args, H) == 64 &&
                  offsetof(legion_gdn_mma_args, sq1) == 104 && offsetof(legion_gdn_mma_args, scale) == 176,
              "gdn_mma.rs's Args has this layout");

namespace {

// mma.cuh (ggml_cuda_mma), the NVIDIA DATA_LAYOUT_I_MAJOR tiles and operations the kernel uses.
namespace mma {

// A row-major I x J tile of 32-bit elements spread over a warp (J counts physical 32-bit
// elements: two BF16 values for nv_bfloat162).
template <int I_, int J_, typename T> struct tile {
    static constexpr int I  = I_;
    static constexpr int J  = J_;
    static constexpr int ne = I * J / 32;
    T                    x[ne] = { 0 };

    static __device__ __forceinline__ int get_i(const int l) {
        static_assert(I == 16 && J == 8, "only the accumulator's indices are used");
        return ((l / 2) * 8) + (threadIdx.x / 4);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        static_assert(I == 16 && J == 8, "only the accumulator's indices are used");
        return ((threadIdx.x % 4) * 2) + (l % 2);
    }
};

template <int I_, int J_> struct tile<I_, J_, nv_bfloat162> {
    static constexpr int I  = I_;
    static constexpr int J  = J_;
    static constexpr int ne = I * J / 32;
    nv_bfloat162         x[ne] = { { 0.0f, 0.0f } };
};

template <int stride, typename T>
static __device__ __forceinline__ uint32_t swizzle(const uint32_t offset, const uint32_t i) {
    static_assert(sizeof(T) <= 4, "unsupported type size");
    constexpr int stride_bytes = stride * sizeof(T);
    static_assert(stride_bytes % 16 == 0, "bad stride");
    constexpr uint32_t shift = sizeof(T) == 1 ? 4 : (sizeof(T) == 2 ? 3 : 2);
    if (stride_bytes % 32 != 0) {
        return offset;  // Equivalent to padding with 16 bytes.
    }
    if (stride_bytes % 64 != 0) {
        return offset ^ (((i / 4) % 2) << shift);
    }
    if (stride_bytes % 128 != 0) {
        return offset ^ (((i / 2) % 4) << shift);
    }
    return offset ^ ((i % 8) << shift);
}

#ifdef LEGION_GDN_MMA_AVAILABLE
template <int stride, int I, int J, typename T>
static __device__ __forceinline__ void load_ldmatrix_swizzled(tile<I, J, T> & t, const T * __restrict__ xs0,
                                                              const int offset) {
    static_assert(I == 8 || I == 16, "bad tile width");
    static_assert(J == 8, "bad tile height");
    const int i         = threadIdx.x % t.I;
    const int j         = ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
    int       offset_ij = offset + i * stride + j;
    offset_ij           = swizzle<stride, T>(offset_ij, i);
    int * xi            = (int *) t.x;
    if constexpr (I == 16) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                     : "l"(xs0 + offset_ij));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                     : "=r"(xi[0]), "=r"(xi[1])
                     : "l"(xs0 + offset_ij));
    }
}

template <int stride, int I, typename T>
static __device__ __forceinline__ void load_ldmatrix_trans_swizzled(tile<I, 8, T> & t, const T * __restrict__ xs0,
                                                                    const int offset) {
    static_assert(I == 8 || I == 16, "bad tile width");
    const int i         = threadIdx.x % 16;
    const int j         = I == 16 ? (threadIdx.x / 16) * (t.J / 2) : 0;
    int       offset_ij = offset + i * stride + j;
    offset_ij           = swizzle<stride, T>(offset_ij, i);
    int * xi            = (int *) t.x;
    if constexpr (I == 16) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
                     : "l"(xs0 + offset_ij));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                     : "=r"(xi[0]), "=r"(xi[1])
                     : "l"(xs0 + offset_ij));
    }
}

static __device__ __forceinline__ void mma(tile<16, 8, float> & D, const tile<16, 8, nv_bfloat162> & A,
                                           const tile<8, 8, nv_bfloat162> & B) {
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int *       Dxi = (int *) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, "
        "%2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
}
#endif  // LEGION_GDN_MMA_AVAILABLE

}  // namespace mma

// gated-delta-net-mma.cu from here on.

constexpr int D = 128;
using bf16      = nv_bfloat16;

template <int R, int C> struct matrix {
    bf16 hi[R * C], lo[R * C];

    static __device__ __forceinline__ int index(int r, int c) {
        return 2 * mma::swizzle<C / 2, nv_bfloat162>(r * (C / 2) + c / 2, r) + c % 2;
    }

    __device__ __forceinline__ void store2(int r, int c, float2 x) {
        const int    i                            = index(r, c);
        const auto   h                            = __float22bfloat162_rn(x);
        const float2 rounded                      = __bfloat1622float2(h);
        *reinterpret_cast<nv_bfloat162 *>(hi + i) = h;
        *reinterpret_cast<nv_bfloat162 *>(lo + i) =
            __float22bfloat162_rn(make_float2(x.x - rounded.x, x.y - rounded.y));
    }

    __device__ __forceinline__ void store(int r, int c, float x) {
        const int  i = index(r, c);
        const bf16 h = __float2bfloat16(x);
        hi[i]        = h;
        lo[i]        = __float2bfloat16(x - __bfloat162float(h));
    }
};

template <int C, int V> struct alignas(128) shared {
    matrix<C, D> q, k;
    matrix<C, C> p, inverse;
    matrix<C, V> delta;

    union {
        matrix<D, V> state;
        matrix<C, V> solved;
    } scratch;

    float lower[C * C], prefix[C], beta[C];

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }
};

// gdn_mma.rs's `smem`: 19584 + 576 * V bytes for C = 16.
static_assert(sizeof(shared<16, 32>) == 38016, "gdn_mma.rs's smem(32)");
static_assert(sizeof(shared<16, 128>) == 93312, "gdn_mma.rs's smem(128)");

#ifdef LEGION_GDN_MMA_AVAILABLE
using acc    = mma::tile<16, 8, float>;
using a_tile = mma::tile<16, 8, nv_bfloat162>;
using b_tile = mma::tile<8, 8, nv_bfloat162>;

template <class tile_t, bool TRANS, int R, int C>
__device__ __forceinline__ tile_t load(const matrix<R, C> & m, int row, int col, bool low) {
    tile_t       t;
    const bf16 * data   = low ? m.lo : m.hi;
    const auto * packed = reinterpret_cast<const nv_bfloat162 *>(data);
    if constexpr (TRANS) {
        mma::load_ldmatrix_trans_swizzled<C / 2>(t, packed, col * (C / 2) + row / 2);
    } else {
        mma::load_ldmatrix_swizzled<C / 2>(t, packed, row * (C / 2) + col / 2);
    }
    return t;
}

// B is addressed as B-transpose. Preserve both BF16 first-order residual terms.
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR, int BC>
__device__ __forceinline__ void product(acc & c, const matrix<AR, AC> & a, const matrix<BR, BC> & b, int row, int col) {
#pragma unroll
    for (int k = 0; k < K; k += 16) {
        const auto ah = load<a_tile, TA>(a, row, k, false);
        const auto al = load<a_tile, TA>(a, row, k, true);
        const auto bh = load<b_tile, TB>(b, col, k, false);
        const auto bl = load<b_tile, TB>(b, col, k, true);
        mma::mma(c, al, bh);
        mma::mma(c, ah, bl);
        mma::mma(c, ah, bh);
    }
}

template <int R, int C> __device__ __forceinline__ void store(matrix<R, C> & m, const acc & x, int r, int c) {
#pragma unroll
    for (int i = 0; i < acc::ne; i += 2) {
        m.store2(r + acc::get_i(i), c + acc::get_j(i), make_float2(x.x[i], x.x[i + 1]));
    }
}
#endif  // LEGION_GDN_MMA_AVAILABLE

// Each block owns V independent value columns and advances by C tokens.
// Keep recurrent state in FP32; pack high/residual BF16 operands inside the block.
template <int C, int V, int BLOCKS = 1>
static __global__ __launch_bounds__(256, BLOCKS) void gdn_single(legion_gdn_mma_args a) {
#ifdef LEGION_GDN_MMA_AVAILABLE
    constexpr int WARPS = 8;
    constexpr int N     = 8;
    static_assert(C >= 16 && (C & (C - 1)) == 0 && V % N == 0 && D % V == 0, "invalid GDN tile");
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &                                         s    = *reinterpret_cast<shared<C, V> *>(bytes);
    const int                                      lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int                                      slice = blockIdx.x % (D / V);
    const int                                      h     = (blockIdx.x / (D / V)) % a.H;
    const int                                      seq   = blockIdx.x / ((D / V) * a.H);
    const int                                      v0    = slice * V;
    const int64_t                                  soff  = ((int64_t) seq * a.H + h) * D * D;
    const int64_t                                  qoff  = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t                                  voff  = seq * a.sv3 + h * a.sv1 + v0;
    const int64_t                                  goff  = seq * a.sb3 + h * a.sb1;
    float *                                        out   = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D + v0;
    constexpr int                                  STATE_TILES = (D / 16) * (V / N);
    constexpr int                                  PER_WARP    = STATE_TILES / WARPS;
    static_assert(STATE_TILES % WARPS == 0, "state must divide across warps");
    acc state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int tile_index = warp + j * WARPS, r = (tile_index / (V / N)) * 16, c = (tile_index % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            state[j].x[i] = a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)];
        }
    }
    auto read_qk = [&](int64_t t, float4 & q, float4 & k) {
        t += warp;
        q = k = make_float4(0, 0, 0, 0);
        if (t < a.n_tokens) {
            const int64_t off = qoff + t * a.sq2 + 4 * lane;
            if ((((uintptr_t) (a.q + off) | (uintptr_t) (a.k + off)) & 15) == 0) {
                q = *reinterpret_cast<const float4 *>(a.q + off);
                k = *reinterpret_cast<const float4 *>(a.k + off);
            } else {
                q = make_float4(a.q[off], a.q[off + 1], a.q[off + 2], a.q[off + 3]);
                k = make_float4(a.k[off], a.k[off + 1], a.k[off + 2], a.k[off + 3]);
            }
        }
    };
    // Carry the first eight input rows across chunks to hide global-load latency.
    constexpr bool PREFETCH = C == 16 && V == 32;
    float4         next_q, next_k;
    if constexpr (PREFETCH) {
        read_qk(0, next_q, next_k);
    }
    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C) {
        const int valid = min((int64_t) C, a.n_tokens - t0);
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            float4    q, k;
            if (PREFETCH && j == 0) {
                q = next_q;
                k = next_k;
            } else {
                read_qk(t0 + 8 * j, q, k);
            }
            s.q.store2(t, d, make_float2(q.x * a.scale, q.y * a.scale));
            s.q.store2(t, d + 2, make_float2(q.z * a.scale, q.w * a.scale));
            s.k.store2(t, d, make_float2(k.x, k.y));
            s.k.store2(t, d + 2, make_float2(k.z, k.w));
        }
        // Load gates in parallel before the FP32 prefix scan.
        if (warp == 0) {
            float carry = 0.f;
            for (int base = 0; base < C; base += 32) {
                const int   t    = base + lane;
                float       g    = t < valid ? a.g[goff + (t0 + t) * a.sb2] : 0.f;
                const float beta = t < valid ? a.beta[goff + (t0 + t) * a.sb2] : 0.f;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                g += carry;
                if (t < C) {
                    s.prefix[t] = g;
                    s.beta[t]   = beta;
                }
                carry = __shfl_sync(0xffffffff, g, 31, 32);
            }
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS;
            store(s.scratch.state, state[j], (ti / (V / N)) * 16, (ti % (V / N)) * N);
        }
        __syncthreads();
        if constexpr (PREFETCH) {
            if (t0 + C < a.n_tokens) {
                read_qk(t0 + C, next_q, next_k);
            }
        }
        // V32 overlaps Gram work with one output tile per projection warp.
        constexpr bool SPLIT_PROJECTION = C == 16 && V == 32;
        acc            base_output[((C / 16) * (V / N) + WARPS - 1) / WARPS];
        // delta = beta * (v - decay * K S), base output = decay * Q S
        auto           project = [&]() {
            if constexpr (V == 32) {
                if (warp < 4) {
                    return;
                }
            }
            for (int ti = (V == 32 ? warp - 4 : warp); ti < (C / 16) * (V / N); ti += (V == 32 ? 4 : WARPS)) {
                const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
                acc       qs, ks;
                product<D, false, true>(qs, s.q, s.scratch.state, r, c);
                product<D, false, true>(ks, s.k, s.scratch.state, r, c);
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   t = r + acc::get_i(i), v = c + acc::get_j(i);
                    const float decay = expf(s.prefix[t]);
                    const float vv    = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - decay * ks.x[i]));
                    base_output[ti / WARPS].x[i] = decay * qs.x[i];
                }
            }
        };
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        for (int ti = warp; ti < 2 * (C / 16) * (C / N) && (V != 32 || warp < 4); ti += (V == 32 ? 4 : WARPS)) {
            const bool gram = ti < (C / 16) * (C / N);
            const int  t = ti % ((C / 16) * (C / N)), r = (t / (C / N)) * 16, c = (t % (C / N)) * N;
            acc        x;
            if (gram) {
                product<D>(x, s.k, s.k, r, c);
            } else {
                product<D>(x, s.q, s.k, r, c);
            }
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int   rr = r + acc::get_i(i), cc = c + acc::get_j(i);
                const float v = rr < valid && cc <= rr ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                if (gram) {
                    if (cc < rr) {
                        s.lower[s.lower_index(rr, cc)] = v * s.beta[rr];
                    }
                } else {
                    s.p.store(rr, cc, v);
                }
            }
        }
        if constexpr (V == 32) {
            project();
        }
        __syncthreads();
        // inverse = (I + A)^-1 by forward substitution
        for (int row = warp; row < C; row += WARPS) {
            float x[(C + 31) / 32];
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                const int col = lane + 32 * j;
                x[j]          = col == row ? 1.f : (col < row ? -s.lower[s.lower_index(row, col)] : 0.f);
            }
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#pragma unroll
                for (int j = 0; j < (C + 31) / 32; ++j) {
                    const int col = lane + 32 * j;
                    if (col < pivot) {
                        x[j] -= xp * s.lower[s.lower_index(pivot, col)];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                if (lane + 32 * j < C) {
                    s.inverse.store(row, lane + 32 * j, x[j]);
                }
            }
        }
        if constexpr (V != 32) {
            project();
        }
        __syncthreads();
        // solved = inverse * delta
        for (int ti = warp; ti < (C / 16) * (V / N); ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.inverse, s.delta, r, c);
            store(s.scratch.solved, x, r, c);
        }
        __syncthreads();
        // out = base output + P * solved
        for (int ti = (SPLIT_PROJECTION ? warp - 4 : warp); ti < (C / 16) * (V / N) && (!SPLIT_PROJECTION || warp >= 4);
             ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.p, s.scratch.solved, r, c);
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int t = r + acc::get_i(i), v = c + acc::get_j(i);
                if (t < valid) {
                    out[(t0 + t) * a.H * D + v] = base_output[ti / WARPS].x[i] + x.x[i];
                }
            }
        }
        // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
        for (int i = tid; i < C * V; i += 256) {
            const int   t = i / V, v = i % V, j = s.scratch.solved.index(t, v);
            const float x = __bfloat162float(s.scratch.solved.hi[j]) + __bfloat162float(s.scratch.solved.lo[j]);
            s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
        }
        __syncthreads();
        const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                state[j].x[i] *= decay;
            }
            product<C, true, true>(state[j], s.k, s.delta, r, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            a.state_out[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)] = state[j].x[i];
        }
    }
#else
    (void) a;
    __trap();  // gdn_mma.rs launches it on compute capability 8.0+ only
#endif  // LEGION_GDN_MMA_AVAILABLE
}

// gdn_staged (#1183): `gdn_single<16, 128>` (whole heads) with the same arithmetic in the same
// order, so the same bits, laid out for a GPU that waits on memory (the GB10):
// - The next chunk's q, k, v, g and beta are copied into shared memory (cp.async, two buffers)
//   while the current chunk computes; gdn_single loads them from global memory inside the chunk.
// - The state's BF16 high and low halves for Q S and K S come from the warp's own state
//   accumulators (a warp holds every row of its two 8-column blocks), moved into the B operand
//   layout with movmatrix, instead of going through 64 KB of shared memory every chunk. The
//   halves are matrix::store2's roundings and the products take ggml's operand order.
// - A operands shared by a warp's two column blocks are loaded once.
template <int C> struct alignas(128) staged_shared {
    matrix<C, D> q, k;
    matrix<C, C> p, inverse;
    matrix<C, D> delta, solved;
    float        lower[C * C], prefix[C], beta[C];

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }

    struct alignas(16) stage_t {
        float q[C * D], k[C * D], v[C * D], g[C], beta[C];
    } stage[2];
};

// gdn_mma.rs's `Schedule::Staged.smem()`.
static_assert(sizeof(staged_shared<16>) == 85376, "gdn_mma.rs's staged smem");

#ifdef LEGION_GDN_MMA_AVAILABLE
static __device__ __forceinline__ void cp_async16(float * dst, const float * src, bool ok) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(src), "r"(ok ? 16 : 0) : "memory");
}

static __device__ __forceinline__ void cp_async4(float * dst, const float * src, bool ok) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(s), "l"(src), "r"(ok ? 4 : 0) : "memory");
}

// The B operand halves of a 16 x 8 state accumulator (rows: key dims, columns: value dims), as
// `load<b_tile, true>` reads them back after `store(m, x, r, c)`.
static __device__ __forceinline__ void state_b(const acc & x, b_tile & hi, b_tile & lo) {
    int * h = (int *) hi.x;
    int * l = (int *) lo.x;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const float2       v       = make_float2(x.x[2 * p], x.x[2 * p + 1]);
        const nv_bfloat162 vh      = __float22bfloat162_rn(v);
        const float2       rounded = __bfloat1622float2(vh);
        const nv_bfloat162 vl      = __float22bfloat162_rn(make_float2(v.x - rounded.x, v.y - rounded.y));
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(h[p]) : "r"(*(const int *) &vh));
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(l[p]) : "r"(*(const int *) &vl));
    }
}
#endif  // LEGION_GDN_MMA_AVAILABLE

template <int C> static __global__ __launch_bounds__(256, 1) void gdn_staged(legion_gdn_mma_args a) {
#ifdef LEGION_GDN_MMA_AVAILABLE
    constexpr int WARPS = 8;
    constexpr int N     = 8;
    constexpr int V     = D;
    static_assert(C == 16, "one 16-row tile of tokens");
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &                                         s    = *reinterpret_cast<staged_shared<C> *>(bytes);
    const int                                      lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int                                      h    = blockIdx.x % a.H;
    const int                                      seq  = blockIdx.x / a.H;
    const int64_t                                  soff = ((int64_t) seq * a.H + h) * D * D;
    const int64_t                                  qoff = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t                                  voff = seq * a.sv3 + h * a.sv1;
    const int64_t                                  goff = seq * a.sb3 + h * a.sb1;
    float *                                        out  = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D;
    constexpr int                                  STATE_TILES = (D / 16) * (V / N);
    constexpr int                                  PER_WARP    = STATE_TILES / WARPS;
    static_assert(PER_WARP == 16, "a warp owns column blocks warp and warp + 8 of every row block");
    // State tile j of a warp: rows 16 * (j / 2), columns 8 * (warp + 8 * (j % 2)).
    acc state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int tile_index = warp + j * WARPS, r = (tile_index / (V / N)) * 16, c = (tile_index % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            state[j].x[i] = a.state[soff + (c + acc::get_j(i)) * D + r + acc::get_i(i)];
        }
    }
    // Tokens past the end are zeros (cp.async's zero fill), as gdn_single reads them.
    auto issue = [&](int buf, int64_t t0) {
        auto & st = s.stage[buf];
        for (int i = tid; i < C * (D / 4); i += 256) {
            const int     t = i / (D / 4), d = 4 * (i % (D / 4));
            const bool    ok = t0 + t < a.n_tokens;
            const int64_t tt = ok ? t0 + t : 0;
            cp_async16(st.q + t * D + d, a.q + qoff + tt * a.sq2 + d, ok);
            cp_async16(st.k + t * D + d, a.k + qoff + tt * a.sq2 + d, ok);
            cp_async16(st.v + t * D + d, a.v + voff + tt * a.sv2 + d, ok);
        }
        if (tid < 2 * C) {
            const int     t  = tid % C;
            const bool    ok = t0 + t < a.n_tokens;
            const int64_t tt = ok ? t0 + t : 0;
            if (tid < C) {
                cp_async4(st.g + t, a.g + goff + tt * a.sb2, ok);
            } else {
                cp_async4(st.beta + t, a.beta + goff + tt * a.sb2, ok);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    issue(0, 0);
    int buf = 0;
    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C, buf ^= 1) {
        const int valid = min((int64_t) C, a.n_tokens - t0);
        // The other buffer was last read before the previous chunk's solve.
        if (t0 + C < a.n_tokens) {
            issue(buf ^ 1, t0 + C);
        } else {
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
        const auto & st = s.stage[buf];
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int    t = warp + 8 * j, d = 4 * lane;
            const float4 q = *reinterpret_cast<const float4 *>(st.q + t * D + d);
            const float4 k = *reinterpret_cast<const float4 *>(st.k + t * D + d);
            s.q.store2(t, d, make_float2(q.x * a.scale, q.y * a.scale));
            s.q.store2(t, d + 2, make_float2(q.z * a.scale, q.w * a.scale));
            s.k.store2(t, d, make_float2(k.x, k.y));
            s.k.store2(t, d + 2, make_float2(k.z, k.w));
        }
        if (warp == 0) {
            float carry = 0.f;
            for (int base = 0; base < C; base += 32) {
                const int   t    = base + lane;
                float       g    = t < valid ? st.g[t] : 0.f;
                const float beta = t < valid ? st.beta[t] : 0.f;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                g += carry;
                if (t < C) {
                    s.prefix[t] = g;
                    s.beta[t]   = beta;
                }
                carry = __shfl_sync(0xffffffff, g, 31, 32);
            }
        }
        __syncthreads();
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        for (int ti = warp; ti < 2 * (C / 16) * (C / N); ti += WARPS) {
            const bool gram = ti < (C / 16) * (C / N);
            const int  t = ti % ((C / 16) * (C / N)), r = (t / (C / N)) * 16, c = (t % (C / N)) * N;
            acc        x;
            if (gram) {
                product<D>(x, s.k, s.k, r, c);
            } else {
                product<D>(x, s.q, s.k, r, c);
            }
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int   rr = r + acc::get_i(i), cc = c + acc::get_j(i);
                const float v  = rr < valid && cc <= rr ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                if (gram) {
                    if (cc < rr) {
                        s.lower[s.lower_index(rr, cc)] = v * s.beta[rr];
                    }
                } else {
                    s.p.store(rr, cc, v);
                }
            }
        }
        __syncthreads();
        // inverse = (I + A)^-1 by forward substitution
        for (int row = warp; row < C; row += WARPS) {
            float x[(C + 31) / 32];
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                const int col = lane + 32 * j;
                x[j]          = col == row ? 1.f : (col < row ? -s.lower[s.lower_index(row, col)] : 0.f);
            }
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#pragma unroll
                for (int j = 0; j < (C + 31) / 32; ++j) {
                    const int col = lane + 32 * j;
                    if (col < pivot) {
                        x[j] -= xp * s.lower[s.lower_index(pivot, col)];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                if (lane + 32 * j < C) {
                    s.inverse.store(row, lane + 32 * j, x[j]);
                }
            }
        }
        // delta = beta * (v - decay * K S), base output = decay * Q S, for the warp's column
        // blocks warp and warp + 8 (gdn_single's projection tiles ti = warp, warp + 8).
        acc base_output[2];
        {
            acc qs[2], ks[2];
#pragma unroll
            for (int kb = 0; kb < D / 16; ++kb) {
                const auto qh = load<a_tile, false>(s.q, 0, 16 * kb, false);
                const auto ql = load<a_tile, false>(s.q, 0, 16 * kb, true);
                const auto kh = load<a_tile, false>(s.k, 0, 16 * kb, false);
                const auto kl = load<a_tile, false>(s.k, 0, 16 * kb, true);
#pragma unroll
                for (int cb = 0; cb < 2; ++cb) {
                    b_tile bh, bl;
                    state_b(state[2 * kb + cb], bh, bl);
                    mma::mma(qs[cb], ql, bh);
                    mma::mma(qs[cb], qh, bl);
                    mma::mma(qs[cb], qh, bh);
                    mma::mma(ks[cb], kl, bh);
                    mma::mma(ks[cb], kh, bl);
                    mma::mma(ks[cb], kh, bh);
                }
            }
#pragma unroll
            for (int cb = 0; cb < 2; ++cb) {
                const int c = (warp + WARPS * cb) * N;
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   t = acc::get_i(i), v = c + acc::get_j(i);
                    const float decay = expf(s.prefix[t]);
                    const float vv    = t < valid ? st.v[t * D + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - decay * ks[cb].x[i]));
                    base_output[cb].x[i] = decay * qs[cb].x[i];
                }
            }
        }
        __syncthreads();
        // solved = inverse * delta
        {
            const auto ih = load<a_tile, false>(s.inverse, 0, 0, false);
            const auto il = load<a_tile, false>(s.inverse, 0, 0, true);
#pragma unroll
            for (int cb = 0; cb < 2; ++cb) {
                const int  c  = (warp + WARPS * cb) * N;
                const auto bh = load<b_tile, true>(s.delta, c, 0, false);
                const auto bl = load<b_tile, true>(s.delta, c, 0, true);
                acc        x;
                mma::mma(x, il, bh);
                mma::mma(x, ih, bl);
                mma::mma(x, ih, bh);
                store(s.solved, x, 0, c);
            }
        }
        __syncthreads();
        // out = base output + P * solved
        {
            const auto ph = load<a_tile, false>(s.p, 0, 0, false);
            const auto pl = load<a_tile, false>(s.p, 0, 0, true);
#pragma unroll
            for (int cb = 0; cb < 2; ++cb) {
                const int  c  = (warp + WARPS * cb) * N;
                const auto bh = load<b_tile, true>(s.solved, c, 0, false);
                const auto bl = load<b_tile, true>(s.solved, c, 0, true);
                acc        x;
                mma::mma(x, pl, bh);
                mma::mma(x, ph, bl);
                mma::mma(x, ph, bh);
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int t = acc::get_i(i), v = c + acc::get_j(i);
                    if (t < valid) {
                        out[(t0 + t) * a.H * D + v] = base_output[cb].x[i] + x.x[i];
                    }
                }
            }
        }
        // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
        for (int i = tid; i < C * V; i += 256) {
            const int   t = i / V, v = i % V, j = s.solved.index(t, v);
            const float x = __bfloat162float(s.solved.hi[j]) + __bfloat162float(s.solved.lo[j]);
            s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
        }
        __syncthreads();
        const float decay = expf(s.prefix[valid - 1]);
        b_tile      dh[2], dl[2];
#pragma unroll
        for (int cb = 0; cb < 2; ++cb) {
            const int c = (warp + WARPS * cb) * N;
            dh[cb]      = load<b_tile, true>(s.delta, c, 0, false);
            dl[cb]      = load<b_tile, true>(s.delta, c, 0, true);
        }
#pragma unroll
        for (int kb = 0; kb < D / 16; ++kb) {
            const auto kh = load<a_tile, true>(s.k, 16 * kb, 0, false);
            const auto kl = load<a_tile, true>(s.k, 16 * kb, 0, true);
#pragma unroll
            for (int cb = 0; cb < 2; ++cb) {
                acc & x = state[2 * kb + cb];
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    x.x[i] *= decay;
                }
                mma::mma(x, kl, dh[cb]);
                mma::mma(x, kh, dl[cb]);
                mma::mma(x, kh, dh[cb]);
            }
        }
        // The next chunk's first barrier orders these reads of k, delta and prefix before its
        // writes.
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            a.state_out[soff + (c + acc::get_j(i)) * D + r + acc::get_i(i)] = state[j].x[i];
        }
    }
#else
    (void) a;
    __trap();  // gdn_mma.rs launches it on compute capability 8.0+ only
#endif  // LEGION_GDN_MMA_AVAILABLE
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of gdn_mma.rs's `Schedule`: value slices of 32 at one and two blocks an SM, whole
// heads, whole heads staged. They sit in an anonymous namespace: with relocatable device code the
// device link left extern "C" kernels out of the linked image.
extern "C" const void * const legion_gdn_mma_kernels[4] = {
    reinterpret_cast<const void *>(&gdn_single<16, 32, 1>),
    reinterpret_cast<const void *>(&gdn_single<16, 32, 2>),
    reinterpret_cast<const void *>(&gdn_single<16, 128>),
    reinterpret_cast<const void *>(&gdn_staged<16>),
};
