// Q4_0 prompt matmuls with ggml-cuda's MMQ bits from a pipelined main loop (src/model/q40_mm.rs):
// the quantization of the F32 input (ggml's quantize_mmq_q8_1 values for a Q4_0 weight), then
// the matmul, mma.sync m16n8k32 s8 x s8 -> s32 on output tiles, the weights and the quantized
// input streamed by cp.async through a shared-memory ring.
//
// The bits. For a Q4_0 weight ggml's MMQ (MMA path, sm_75 and newer) loads each block's quants as
// int8 q - 8 (mmq-load-tiles.cuh, ggml_cuda_mmq_load_tiles_q4_0) and its F16 scale as F32, and
// sums, for every output, `sum += C * dA * dB` over the K blocks in order (mmq-vec-dot.cuh,
// ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma with the DS4 layout): C the block's exact integer dot
// product, dA the weight's scale, dB the input block's scale as the F16 it is stored in (DS4:
// half2(d, sum)). Compiled with fast math, that is t = fl(C * dA) (FMUL), then
// sum = fma(t, dB, sum) (FFMA), from sum = 0. Where ggml runs a CTA per tile (its plan's grid
// has no stream-k split: src/model/q40_mm.rs takes only those), the whole chain of every output
// runs in one thread in K order, so any tiling and raster order that keeps the chain gives the
// same bits. Here:
// - the integer dot product starts at 0x4B400000 (1.5 * 2^23 as a float's bits) in the MMA's
//   accumulator: |C| <= 32 * 8 * 127 < 2^22, so the result's bits are the float
//   M + C (M = 12582912) exactly;
// - t = fma(M + C, dA, -M * dA): M * dA is exact (dA has 11 significant bits, M two), so the
//   FFMA's exact value is C * dA and its one rounding gives fl(C * dA), ggml's FMUL;
// - sum += t * dB is ggml's FFMA, the blocks in K order.
// The first version takes the nibbles' q - 8 as int8: ((q & 0xF) + 0x78) ^ 0x80 per byte (no
// carries: 0x78 + 15 < 0x100), the values ggml's __vsub4 gives. The op's takes 16 (q - 8):
// (q << 4) ^ 0x80 per byte, one LOP3 a word; its MMA's result reads as the float M + 16 C
// (|16 C| <= 32 * 128 * 127 < 2^22), and t = fma(M + 16 C, dA / 16, -M * dA / 16) has the same
// exact value C * dA (dA / 16 and M * dA / 16 are exact), so the same one rounding. Since round
// 6 the op's input scales are stored as dB / 16 (exact) and its matmul takes dA itself:
// t' = fma(M + 16 C, dA, -M * dA) = 16 fl(C * dA) (one rounding of 16 C dA, a power of two away),
// and fma(t', dB / 16, sum) has the exact product fl(C * dA) * dB of ggml's FFMA.
//
// The op's matmul (q40_cfg, q40_gemm): 64 x 128 tiles, 4 warps of 32 rows x 64 columns, a
// two-stage ring of 4 K blocks a stage in 47616 bytes of shared memory, two CTAs an SM (where a
// CTA waits at its stage's barrier the other's MMAs run); the weights in 16-byte copies; the
// input in layout 1 (one 16-byte read gives a lane the B fragments of two n8 tiles, another their
// scales); tiles in groups of 4 column tiles, row tiles slowest in a group (a wave of 96 CTAs: 24
// row tiles x 4 column tiles, the input's 4 column tiles staying in L2 over the group). The first
// version's (q40_gemm_v1): 128 x 128 tiles, 8 warps of 32 x 64, a three-stage ring in 86016 bytes,
// one CTA an SM, 8-byte weight copies, layout 0, groups of 8 row tiles; kept for the timing test
// (src/model/q40_mm/tests.rs) with a 128 x 128 tile of the op's code. All run the same chains, so
// they give the same bits.

#include <cuda_fp16.h>

#include <cstdint>

extern "C" {

// The op's launch arguments (src/model/q40_mm.rs, `Params`).
struct legion_q40_mm {
    const void * w;    // Q4_0 weight, rows w_nb1 bytes apart
    const float * x;   // F32 input, columns x_ld floats apart
    int * yq;          // quantized input [k / 32][n_pad][8 words] (layout of q40_quant)
    float * yd;        // its scales [k / 32][n_pad] (F16 values as F32; layout 1: / 16)
    float * dst;       // F32 [n][rows], columns dst_ld floats apart
    long long x_ld;
    long long w_nb1;
    long long dst_ld;
    int k;
    int n;
    int n_pad;         // columns of the quantized input: ntx * 128
    int rows;
    int ntx;           // column tiles
    int nty;           // row tiles
    // The GeGLU pair (q40_gemm<q40_op, 1>): w the gate, w2 the up weight (rows w_nb1 bytes
    // apart too), and the quantization of their GeGLU (layout 1, [rows / 32][n_pad][8 words],
    // the scales after) in place of dst.
    const void * w2;
    int * oq;
    float * od;
};

}

namespace {

// The first version's matmul.
constexpr int Q40_BM          = 128;  // tile rows
constexpr int Q40_BN          = 128;  // tile columns
constexpr int Q40_SB          = 4;    // K blocks a stage
constexpr int Q40_STAGES      = 3;
constexpr int Q40_WRS         = 20;   // words a weight row takes in a stage (18 used)
constexpr int Q40_W_WORDS     = Q40_BM * Q40_WRS;
constexpr int Q40_Q_WORDS     = Q40_SB * Q40_BN * 8;
constexpr int Q40_D_WORDS     = Q40_SB * Q40_BN;
constexpr int Q40_STAGE_WORDS = Q40_W_WORDS + Q40_Q_WORDS + Q40_D_WORDS;
constexpr int Q40_THREADS     = 256;

static_assert(Q40_STAGES * Q40_STAGE_WORDS * 4 == 86016, "src/model/q40_mm.rs SMEM_BYTES_V1");

// The matmul's helpers: device code for sm_80 and newer only (cp.async).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

constexpr int   Q40_GROUP      = 8;  // row tiles a raster group of the first version
constexpr int   Q40_MAGIC_BITS = 0x4B400000;
constexpr float Q40_MAGIC      = 12582912.0f;

__device__ __forceinline__ void q40_cp8(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void q40_cp16(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

// A 16-byte copy of which the first `src` bytes are read and the rest zero-filled.
__device__ __forceinline__ void q40_cp16z(void * smem, const void * gmem, const int src) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src));
}

// Hints the 32-byte sector at p into L2.
__device__ __forceinline__ void q40_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

__device__ __forceinline__ void q40_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void q40_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// d = a x b + (M, M, M, M) (the magic start, see the top).
__device__ __forceinline__ void q40_mma(int (&d)[4], const uint32_t (&a)[4], const int b0, const int b1) {
    const int m = Q40_MAGIC_BITS;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m), "r"(m), "r"(m), "r"(m));
}

// A word of 4 nibbles (0 .. 15 in the low 4 bits of each byte) as the int8 values q - 8.
__device__ __forceinline__ uint32_t q40_s8(const uint32_t nib) {
    return ((nib & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u;
}

// A word of 4 nibbles (in the high 4 bits of each byte) as the int8 values 16 (q - 8):
// (nib & 0xF0F0F0F0) ^ 0x80808080 as one LOP3 (LUT 0xF0 & 0xCC ^ 0xAA = 0x6A; the compiler's
// own takes two).
__device__ __forceinline__ uint32_t q40_s8x16(const uint32_t nib) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;\n" : "=r"(r) : "r"(nib), "r"(0xF0F0F0F0u), "r"(0x80808080u));
    return r;
}

// A compile-time int (the matmul's block parity).
template <int V> struct q40_k {
    static constexpr int value = V;
};

#endif

// ggml's quantize_mmq_q8_1 values for a Q4_0 weight (MMQ_Q8_1_DS_LAYOUT_DS4): each 32-value
// block's absmax over 8 threads of 4 values, d_inv = 127 / amax, roundf(x * d_inv), d = 1 / d_inv
// stored as F16. Column col's block kb, slot s (values 4s .. 4s + 3) is word w = 2s for s < 4,
// 2(s - 4) + 1 else (so slots s and s + 4 are adjacent words).
// - Layout 0 (the first version's): the 8 words at yq[(kb * n_pad + col) * 8], the scale at
//   yd[kb * n_pad + col].
// - Layout 1 (the op's): permuted within each 128-column tile so that a lane's B fragments and
//   scales of two n8 tiles are 16 contiguous bytes each. Of tile column c = 8 nt + g (= 8 nt + 2 r
//   + e), word w (= 2 v + f) is at word ((nt / 2 * 8 + g) * 4 + v) * 4 + nt % 2 * 2 + f of the
//   tile's 1024 words of block kb (yq[(kb * n_pad + 128 jt) * 8 ..]); the scale at r * 32 + 2 nt
//   + e of its 128 (yd[kb * n_pad + 128 jt ..]), stored / 16 (the op's matmul's factor).
//
// q40_quant4 writes values i0 .. i0 + 3 (xi) of column col: the 8 lanes of an aligned group of 8
// in a warp hold one 32-value block (i0 / 4 = the lane's index mod 8 within it), every lane of
// the warp calls it; with `store` false (a group's column past the input's) it writes nothing.
template <int LAYOUT>
__device__ __forceinline__ void q40_quant4(int * yq, float * yd, const int n_pad, const float4 xi, const int col,
                                           const int i0, const bool store = true) {
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    const int kb = i0 / 32;
    const int s  = (i0 / 4) % 8;
    const int w  = s < 4 ? 2 * s : 2 * (s - 4) + 1;
    long long qat, dat;
    if constexpr (LAYOUT == 0) {
        const long long at = (long long) kb * n_pad + col;
        qat = at * 8 + w;
        dat = at;
    } else {
        const long long tile = (long long) kb * n_pad + (col & ~127);
        const int c = col & 127, nt = c / 8, g = c % 8;
        qat = tile * 8 + ((nt / 2 * 8 + g) * 4 + w / 2) * 4 + nt % 2 * 2 + w % 2;
        dat = tile + (g / 2) * 32 + 2 * nt + g % 2;
    }
    if (store) {
        yq[qat] = *reinterpret_cast<const int *>(&q);
        if (s == 0) {
            const float dh = __half2float(__float2half(d));
            yd[dat] = LAYOUT == 1 ? dh * 0.0625f : dh;
        }
    }
}

// 4 floats from src: one 16-byte load where aligned.
__device__ __forceinline__ float4 q40_load4(const float * src) {
    return (reinterpret_cast<uintptr_t>(src) & 15) == 0 ? *reinterpret_cast<const float4 *>(src)
                                                        : make_float4(src[0], src[1], src[2], src[3]);
}

// Grid (n, k / 256), 64 threads.
template <int LAYOUT> __global__ void __launch_bounds__(64) q40_quant(const legion_q40_mm p) {
    const int i0  = blockIdx.y * 256 + threadIdx.x * 4;
    const int col = blockIdx.x;
    const float4 xi = q40_load4(p.x + (long long) col * p.x_ld + i0);
    q40_quant4<LAYOUT>(p.yq, p.yd, p.n_pad, xi, col, i0);
}

}  // namespace

extern "C" {

// The launch arguments of the fused quantizations (src/model/q40_mm.rs, `QuantParams`): the
// layout-1 quantization of the values of a producer that ggml would write as F32, from its
// inputs, without that F32 tensor.
struct legion_q40_quant {
    const float * x;   // GEGLU: the gate; norm: the residual (columns x_ld floats apart)
    const float * x2;  // GEGLU: the up values (columns x2_ld floats apart); norm: its weight [k]
    float * y;         // the F32 values quantized, columns of k floats (null: not written)
    int * yq;          // as legion_q40_mm's
    float * yd;
    long long x_ld;
    long long x2_ld;
    float eps;         // norm: rms_norm's
    int k;
    int n;
    int n_pad;
};

}

namespace {

// ggml-cuda's tanh GELU (ggml_cuda_op_gelu_single in ggml-cuda/unary.cuh, MIT): the same
// expressions under the same --use_fast_math, as nvfp4_mm's GeGLU epilogue.
__device__ __forceinline__ float q40_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// The quantization of ggml's geglu_split(gate, up) (unary_gated_op_kernel with op_gelu, MIT:
// `op(x[j0]) * g[j1]`, the gate x and the up values g), each value as ggml's kernel computes it,
// then q40_quant4's. Grid (n, k / 256), 64 threads, as q40_quant.
__global__ void __launch_bounds__(64) q40_quant_geglu(const legion_q40_quant p) {
    const int i0  = blockIdx.y * 256 + threadIdx.x * 4;
    const int col = blockIdx.x;
    const float4 g = q40_load4(p.x + (long long) col * p.x_ld + i0);
    const float4 u = q40_load4(p.x2 + (long long) col * p.x2_ld + i0);
    float4 v;
    v.x = q40_gelu(g.x) * u.x;
    v.y = q40_gelu(g.y) * u.y;
    v.z = q40_gelu(g.z) * u.z;
    v.w = q40_gelu(g.w) * u.w;
    if (p.y) {
        *reinterpret_cast<float4 *>(p.y + (long long) col * p.k + i0) = v;
    }
    q40_quant4<1>(p.yq, p.yd, p.n_pad, v, col, i0);
}

// Threads of a column of q40_quant_norm, and its warps.
constexpr int Q40_NQ_THREADS = 256, Q40_NQ_WARPS = Q40_NQ_THREADS / 32;

__device__ __forceinline__ float q40_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// The quantization of mul(rms_norm(x, eps), w), one CTA of 256 threads a column. The values
// are ggml-cuda's rms_norm_f32<BLOCK, do_multiply> (norm.cu, MIT; BLOCK 1024 from 1024 values
// on, 256 below, as its launcher picks), as src/cuda/sm120/nvfp4_a4.cu's nvfp4_a4_rms_quant
// computes them: thread tid of BLOCK sums the squares of values tid, tid + BLOCK, ... in that
// order, block_reduce (warp sums, then warp 0's sum of them), mean tmp / k, scale
// rsqrtf(mean + eps), y = scale * x * w; here lane l of warp wp plays thread
// l + 32 * (wp + 8 j) of BLOCK. Then each value again from x (the same expression) into
// q40_quant4, 4 a thread (a warp's 128 values a pass: k % 256 == 0 keeps the passes whole).
template <int BLOCK> __global__ void __launch_bounds__(Q40_NQ_THREADS) q40_quant_norm(const legion_q40_quant p) {
    constexpr int V = BLOCK / Q40_NQ_THREADS;
    static_assert(V * Q40_NQ_THREADS == BLOCK, "block");
    __shared__ float red[32];
    const int col = blockIdx.x;
    const float * xr = p.x + (long long) col * p.x_ld;
    const float * w  = p.x2;
    const int k = p.k, lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    float part[V];
#pragma unroll
    for (int j = 0; j < V; j++) {
        const int tid = lane + 32 * (wp + Q40_NQ_WARPS * j);
        float tmp = 0.0f;
        for (int c = tid; c < k; c += BLOCK) {
            const float xi = xr[c];
            tmp += xi * xi;
        }
        part[j] = q40_warp_sum(tmp);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < V; j++) {
            red[wp + Q40_NQ_WARPS * j] = part[j];
        }
    }
    __syncthreads();
    float tmp = lane < BLOCK / 32 ? red[lane] : 0.0f;
    tmp = q40_warp_sum(tmp);
    const float mean  = tmp / k;
    const float scale = rsqrtf(mean + p.eps);
    for (int i0 = threadIdx.x * 4; i0 < k; i0 += Q40_NQ_THREADS * 4) {
        const float4 xi = q40_load4(xr + i0);
        const float4 wi = q40_load4(w + i0);
        float4 v;
        v.x = scale * xi.x * wi.x;
        v.y = scale * xi.y * wi.y;
        v.z = scale * xi.z * wi.z;
        v.w = scale * xi.w * wi.w;
        if (p.y) {
            *reinterpret_cast<float4 *>(p.y + (long long) col * k + i0) = v;
        }
        q40_quant4<1>(p.yq, p.yd, p.n_pad, v, col, i0);
    }
}

// The first version's matmul: a CTA per 128 x 128 tile (grid ntx * nty), 256 threads, 86016
// bytes of dynamic shared memory, the input in layout 0.
__global__ void __launch_bounds__(Q40_THREADS, 1) q40_gemm_v1(const legion_q40_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint32_t q40_smem[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int wm = warp % 4, wn = warp / 4;

    // The tile: groups of Q40_GROUP row tiles, row tiles fastest within a group.
    const int b         = blockIdx.x;
    const int per_group = Q40_GROUP * p.ntx;
    const int grp       = b / per_group;
    const int r0        = grp * Q40_GROUP;
    const int gsz       = min(Q40_GROUP, p.nty - r0);
    const int in        = b - grp * per_group;
    const int it        = r0 + in % gsz;
    const int jt        = in / gsz;
    const int row0      = it * Q40_BM;
    const int col0      = jt * Q40_BN;
    const int nst       = p.k / (32 * Q40_SB);

    const char * wbase = (const char *) p.w + (long long) row0 * p.w_nb1;

    // Stage st's copies into ring slot st % 3 (an empty group past the last stage).
    auto load = [&](const int st) {
        if (st < nst) {
            uint32_t * sw = q40_smem + (st % Q40_STAGES) * Q40_STAGE_WORDS;
            uint32_t * sq = sw + Q40_W_WORDS;
            uint32_t * sd = sq + Q40_Q_WORDS;
            const char * ws = wbase + (long long) st * (Q40_SB * 18);
            for (int c = tid; c < Q40_BM * 9; c += Q40_THREADS) {
                const int r = c / 9, ch = c - r * 9;
                q40_cp8(sw + r * Q40_WRS + ch * 2, ws + (long long) r * p.w_nb1 + ch * 8);
            }
            const int kb0 = st * Q40_SB;
#pragma unroll
            for (int c = tid; c < Q40_SB * Q40_BN * 2; c += Q40_THREADS) {
                const int bl = c / (Q40_BN * 2), ch = c % (Q40_BN * 2);
                q40_cp16(sq + bl * Q40_BN * 8 + ch * 4,
                         p.yq + ((long long) (kb0 + bl) * p.n_pad + col0) * 8 + ch * 4);
            }
            if (tid < Q40_SB * Q40_BN / 4) {
                const int bl = tid / (Q40_BN / 4), ch = tid % (Q40_BN / 4);
                q40_cp16(sd + bl * Q40_BN + ch * 4, p.yd + (long long) (kb0 + bl) * p.n_pad + col0 + ch * 4);
            }
        }
        q40_commit();
    };

    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[mt][nt][l] = 0.0f;
            }
        }
    }

    load(0);
    load(1);
    for (int st = 0; st < nst; st++) {
        q40_wait<1>();
        __syncthreads();  // stage st landed for all; slot (st + 2) % 3 (stage st - 1) read by all
        load(st + 2);

        const uint32_t * sw = q40_smem + (st % Q40_STAGES) * Q40_STAGE_WORDS;
        const uint32_t * sq = sw + Q40_W_WORDS;
        const float    * sd = (const float *) (sq + Q40_Q_WORDS);
#pragma unroll
        for (int bl = 0; bl < Q40_SB; bl++) {
            // The warp's weight fragments: rows wm * 32 + mt * 16 + g (+ 8). Block bl starts at
            // byte 18 bl of the row's stage: an even block's scale is the low half of word 9m
            // (m = bl / 2) and its quants straddle words 9m .. 9m + 4; an odd block's scale is
            // the high half of word 9m + 4 and its quants words 9m + 5 .. 9m + 8.
            uint32_t a[2][4];
            float da[2][2], mda[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t * rw = sw + (wm * 32 + mt * 16 + h * 8 + g) * Q40_WRS + 9 * (bl / 2);
                    uint32_t qs;
                    unsigned short dh;
                    if (bl % 2 == 0) {
                        const uint32_t w0 = rw[0];
                        qs = __byte_perm(rw[q], rw[q + 1], 0x5432);
                        dh = (unsigned short) (w0 & 0xFFFF);
                    } else {
                        qs = rw[5 + q];
                        dh = (unsigned short) (rw[4] >> 16);
                    }
                    a[mt][h]     = q40_s8(qs);       // K slot q: values 4q .. 4q + 3
                    a[mt][h + 2] = q40_s8(qs >> 4);  // K slot q + 4: values 16 + 4q ..
                    da[mt][h]  = __half2float(__ushort_as_half(dh));
                    mda[mt][h] = __fmul_rn(-Q40_MAGIC, da[mt][h]);
                }
            }
            const uint32_t * bq = sq + bl * Q40_BN * 8;
            const float    * bd = sd + bl * Q40_BN;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int c = wn * 64 + nt * 8;
                const uint2  bv = *reinterpret_cast<const uint2 *>(bq + (c + g) * 8 + 2 * q);
                const float2 dv = *reinterpret_cast<const float2 *>(bd + c + 2 * q);
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    int d[4];
                    q40_mma(d, a[mt], (int) bv.x, (int) bv.y);
                    // ggml's `sum += C*dA*dB`: t = fl(C * dA) exactly as its FMUL, then its FFMA.
                    const float t0 = __fmaf_rn(__int_as_float(d[0]), da[mt][0], mda[mt][0]);
                    const float t1 = __fmaf_rn(__int_as_float(d[1]), da[mt][0], mda[mt][0]);
                    const float t2 = __fmaf_rn(__int_as_float(d[2]), da[mt][1], mda[mt][1]);
                    const float t3 = __fmaf_rn(__int_as_float(d[3]), da[mt][1], mda[mt][1]);
                    acc[mt][nt][0] = __fmaf_rn(t0, dv.x, acc[mt][nt][0]);
                    acc[mt][nt][1] = __fmaf_rn(t1, dv.y, acc[mt][nt][1]);
                    acc[mt][nt][2] = __fmaf_rn(t2, dv.x, acc[mt][nt][2]);
                    acc[mt][nt][3] = __fmaf_rn(t3, dv.y, acc[mt][nt][3]);
                }
            }
        }
    }
    q40_wait<0>();

    // The tile's sums to dst (ggml's write back: columns past n are left out).
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = row0 + wm * 32 + mt * 16 + g + (l >> 1) * 8;
                const int j = col0 + wn * 64 + nt * 8 + 2 * q + (l & 1);
                if (j < p.n) {
                    p.dst[(long long) j * p.dst_ld + i] = acc[mt][nt][l];
                }
            }
        }
    }
#else
    __trap();
#endif
}

// A configuration of the op's matmul: BM x BN tiles, WARPS_M x WARPS_N warps of (BM / WARPS_M) x
// (BN / WARPS_N), a STAGES-deep ring of 4 K blocks a stage, MINB CTAs an SM, raster groups of
// GROUP row tiles (COLS 0, column tiles slowest in a group) or GROUP column tiles (COLS 1, row
// tiles slowest). RP 1 reads the weights repacked at load (q40_repack): a stage's 32-row units
// are one contiguous 2304-byte chunk each, copied as is.
template <int BM_, int BN_, int WARPS_M_, int WARPS_N_, int STAGES_, int MINB_, int GROUP_, int COLS_, int RP_ = 0>
struct q40_cfg {
    static constexpr int BM          = BM_;
    static constexpr int BN          = BN_;
    static constexpr int WARPS_M     = WARPS_M_;
    static constexpr int WARPS_N     = WARPS_N_;
    static constexpr int STAGES      = STAGES_;
    static constexpr int MINB        = MINB_;
    static constexpr int GROUP       = GROUP_;
    static constexpr int COLS        = COLS_;
    static constexpr int RP          = RP_;
    static constexpr int THREADS     = 32 * WARPS_M * WARPS_N;
    static constexpr int MT          = BM / WARPS_M / 16;  // m16 tiles a warp
    static constexpr int NT          = BN / WARPS_N / 8;   // n8 tiles a warp
    static constexpr int SB          = 4;
    static constexpr int WRS         = 20;               // words a weight row (18 + 2 of the shift)
    static constexpr int D_BL        = BN / 128 * 144;   // scale words a block: 4 rows of 32, 36 apart
    static constexpr int W_WORDS     = RP ? BM * 18 : BM * WRS;
    static constexpr int Q_WORDS     = SB * BN * 8;
    static constexpr int D_WORDS     = SB * D_BL;
    static constexpr int STAGE_WORDS = W_WORDS + Q_WORDS + D_WORDS;
    static constexpr int SMEM        = STAGES * STAGE_WORDS * 4;
    static constexpr int WCH         = RP ? BM / 32 * 144 : BM * 5;  // 16-byte weight copies a stage
    static constexpr int WCPT        = (WCH + THREADS - 1) / THREADS;
    static constexpr int QCH         = SB * BN * 2;      // 16-byte quant copies a stage
    static constexpr int QCPT        = QCH / THREADS;
    static constexpr int DCH         = SB * BN / 4;      // 16-byte scale copies a stage
    static_assert(MT >= 1 && NT % 2 == 0 && BN % 128 == 0, "warp tiles of n8 tile pairs");
    static_assert(QCH % THREADS == 0 && DCH <= THREADS, "copies");
    static_assert(SMEM <= 99 * 1024, "shared memory");
    static_assert(!RP || MT == 2, "repacked: a warp's rows are one 32-row unit");
};

// The repacked weights (q40_repack, RP 1): 32-row units u (rows 32 u .. 32 u + 31), each the
// k / 128 stages of 4 K blocks in order, a stage's 4 blocks of the unit 2304 bytes:
// - 512 words of quants: block bl's word q (its bytes 4q .. 4q + 3) of unit row
//   16 (j / 2) + 8 (j % 2) + g at word ((bl * 8 + g) * 4 + q) * 4 + j, so lane (g, q) of the
//   warp reading the unit takes its 4 rows' A words (j = 2 mt + h) of a block in one 16-byte
//   read;
// - then 128 F16 scales: block bl's of that row at half (bl * 8 + g) * 4 + j (8 bytes a lane).
// The same values as the Q4_0 rows, only moved: the MMAs and the sums are the same.
constexpr int Q40_RP_UNIT  = 32;    // rows a unit
constexpr int Q40_RP_CHUNK = 2304;  // bytes a unit's stage
constexpr int Q40_RP_WORDS = Q40_RP_CHUNK / 4;

// The op's matmul of configuration C: a CTA per BM x BN tile (grid ntx * nty, C::THREADS threads,
// C::SMEM bytes of dynamic shared memory), the input in layout 1 (n_pad a multiple of BN).
//
// The weights come in 16-byte copies (rows 16-byte aligned: src/model/q40_mm.rs checks it): a
// stage's 72 bytes of a row start at byte 72 st of the row, 16-byte aligned on even stages, 8
// bytes past it on odd ones, so the copy takes the 80 bytes from the aligned byte below into the
// row's 20 words (on even stages the last 8 are zero-filled, not read: they are past the stage,
// maybe past the weight); the row's blocks start 2 words in on odd stages.
//
// EPI 1, the GeGLU pair (FFN gate and up over one input, Gemma 4): row tile it holds gate rows
// 32 it .. 32 it + 31 (w) in its first 32 rows and the same up rows (w2) in its last 32, so warp
// row 0 sums the gate's outputs and warp row 1 the up's, each output's chain as in EPI 0 (the
// same bits as the two matmuls). The epilogue stages both in shared memory, takes
// q40_gelu(gate) * up as q40_quant_geglu does, and writes their quantization (q40_quant4, block
// it of each column) to oq / od as the down matmul's input: the bytes of the two matmuls'
// F32 outputs through q40_quant_geglu, without those outputs.
//
// The scales (the same bits as round 5's, which applied d / 16 to the MMA's M + 16 C): the
// input's scales are stored as dB / 16 (q40_quant4<1>), so t' = fma(M + 16 C, dA, -M dA), whose
// exact value is 16 C dA, rounds once to 16 fl(C dA) (a power of two: no over- or underflow at
// these magnitudes), and fma(t', dB / 16, sum) has the exact product fl(C dA) dB of ggml's FFMA,
// so the same rounding. The weight's scale goes in as converted, with one FMUL for -M dA.
//
// PF > 0 hints the weights PF stage pairs ahead into L2 (on even stages: 5 32-byte sectors a
// row's 144 bytes of a pair), so their copies hit L2 (the timing test's "no prefetch" is PF 0).
template <class C, int EPI = 0, int PF = 6>
__global__ void __launch_bounds__(C::THREADS, C::MINB) q40_gemm(const legion_q40_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int SB = C::SB, MT = C::MT, NT = C::NT, BM = C::BM, BN = C::BN, S = C::STAGES;
    constexpr int HR = BM / 2;  // EPI 1: hidden rows a tile
    static_assert(!EPI || (C::WARPS_M == 2 && HR == 16 * MT && HR == 32), "GeGLU pair: a warp row per weight");
    extern __shared__ __align__(16) uint32_t q40_smem[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int wm = warp % C::WARPS_M, wn = warp / C::WARPS_M;

    // The tile (src/model/q40_mm.rs, `tile_of`).
    const int b         = blockIdx.x;
    const int nmaj      = C::COLS ? p.ntx : p.nty;
    const int nmin      = C::COLS ? p.nty : p.ntx;
    const int per_group = C::GROUP * nmin;
    const int grp       = b / per_group;
    const int r0        = grp * C::GROUP;
    const int gsz       = min(C::GROUP, nmaj - r0);
    const int in        = b - grp * per_group;
    const int it        = C::COLS ? in / gsz : r0 + in % gsz;
    const int jt        = C::COLS ? r0 + in % gsz : in / gsz;
    const int row0      = it * BM;
    const int col0      = jt * BN;
    const int nst       = p.k / (32 * SB);

    // This thread's weight copies: copy c = tid + i * THREADS is row c / 5's 16 bytes c % 5.
    const char * wg = (const char *) p.w + (long long) row0 * p.w_nb1;
    int  woff[C::WCPT];
    bool wend[C::WCPT];
    // EPI 1: the copies' rows, of the gate (tile rows below HR) or the up weight.
    const char * wp[EPI || C::RP ? C::WCPT : 1];
    // RP: unit x of the tile (BM / 32 of them; EPI: the gate's unit it, then the up's) from its
    // first stage; copy c is unit c / 144's 16 bytes c % 144 of a stage.
    const long long ustride = (long long) nst * Q40_RP_CHUNK;
    auto unit = [&](const int x) -> const char * {
        if constexpr (EPI) {
            return (const char *) (x == 0 ? p.w : p.w2) + it * ustride;
        } else {
            return (const char *) p.w + (row0 / Q40_RP_UNIT + x) * ustride;
        }
    };
#pragma unroll
    for (int i = 0; i < C::WCPT; i++) {
        const int c = tid + i * C::THREADS;
        if constexpr (C::RP) {
            wp[i] = unit(min(c, C::WCH - 1) / 144) + (c % 144) * 16;
            woff[i] = 0;
            wend[i] = false;
        } else {
            woff[i] = (c / 5) * (int) p.w_nb1 + (c % 5) * 16;
            wend[i] = c % 5 == 4;
            if constexpr (EPI) {
                const int r = c / 5;
                wp[i] = (const char *) (r < HR ? p.w : p.w2) + (long long) (it * HR + r % HR) * p.w_nb1 + (c % 5) * 16;
            }
        }
    }
    const long long qstep = (long long) p.n_pad * 8;  // words a block
    // Quant copy i of a stage: block (i * THREADS) / (2 BN), its 16 bytes (i * THREADS) % (2 BN)
    // + tid (THREADS divides 2 BN or 2 BN divides THREADS, so no copy crosses a block).
    static_assert((2 * BN) % C::THREADS == 0 || C::THREADS % (2 * BN) == 0, "quant copies");
    // PF: prefetch c = tid + i * THREADS is row c / 5's sector c % 5 of a pair.
    // RP: c is unit c / 144's sector c % 144 of a pair (4608 bytes).
    constexpr int PSEC = C::RP ? BM / 32 * 144 : 5 * BM;  // sectors a pair
    constexpr int PCPT = PF ? (PSEC + C::THREADS - 1) / C::THREADS : 0;
    const char * pfp[PCPT ? PCPT : 1];
#pragma unroll
    for (int i = 0; i < PCPT; i++) {
        const int c = tid + i * C::THREADS, r = c / 5;
        if constexpr (C::RP) {
            pfp[i] = unit(min(c, PSEC - 1) / 144) + (c % 144) * 32;
        } else if constexpr (EPI) {
            pfp[i] = (const char *) (r < HR ? p.w : p.w2) + (long long) (it * HR + r % HR) * p.w_nb1 + (c % 5) * 32;
        } else {
            pfp[i] = wg + (long long) r * p.w_nb1 + (c % 5) * 32;
        }
    }
    const int * qg = p.yq + (long long) col0 * 8 + tid * 4;
    // Scale copy tid: block tid / (BN / 4), 4 scales tid % (BN / 4) of the block's BN.
    const int dbl = tid / (BN / 4), dch = tid % (BN / 4);
    const float * dg = p.yd + (long long) dbl * p.n_pad + col0 + dch * 4;
    const int ds = dbl * C::D_BL + dch / 32 * 144 + dch % 32 / 8 * 36 + dch % 8 * 4;

    // Stage st's copies into ring slot st % S (an empty group past the last stage).
    auto load = [&](const int st) {
        if (st < nst) {
            const int odd = st & 1;
            uint32_t * sw = q40_smem + (st % S) * C::STAGE_WORDS;
            uint32_t * sq = sw + C::W_WORDS;
            uint32_t * sd = sq + C::Q_WORDS;
            // The rows' 16-byte aligned window (72 bytes 0 or 8 in) and the bytes of its last copy.
            const int wo   = (st >> 1) * 144 + odd * 64;
            const int last = odd ? 16 : 8;
#pragma unroll
            for (int i = 0; i < C::WCPT; i++) {
                const int c = tid + i * C::THREADS;
                if constexpr (C::RP) {
                    if (C::WCH % C::THREADS == 0 || c < C::WCH) {
                        q40_cp16(sw + c * 4, wp[i] + (long long) st * Q40_RP_CHUNK);
                    }
                } else if (C::WCH % C::THREADS == 0 || c < C::WCH) {
                    const char * src;
                    if constexpr (EPI) {
                        src = wp[i] + wo;
                    } else {
                        src = wg + (woff[i] + wo);
                    }
                    q40_cp16z(sw + c * 4, src, wend[i] ? last : 16);
                }
            }
            const int * qs = qg + (long long) st * (SB * qstep);
#pragma unroll
            for (int i = 0; i < C::QCPT; i++) {
                constexpr int T = C::THREADS;
                const int bl = i * T / (2 * BN), wi = i * T % (2 * BN);
                q40_cp16(sq + bl * BN * 8 + (wi + tid) * 4, qs + bl * qstep + wi * 4);
            }
            if (C::DCH == C::THREADS || tid < C::DCH) {
                q40_cp16(sd + ds, dg + (long long) st * (SB * p.n_pad));
            }
            if constexpr (PF > 0) {
                const int pair = st / 2 + PF;
                if (odd == 0 && pair < nst / 2) {
#pragma unroll
                    for (int i = 0; i < PCPT; i++) {
                        if (PSEC % C::THREADS == 0 || tid + i * C::THREADS < PSEC) {
                            if constexpr (C::RP) {
                                q40_prefetch_l2(pfp[i] + (long long) pair * (2 * Q40_RP_CHUNK));
                            } else {
                                q40_prefetch_l2(pfp[i] + pair * 144);
                            }
                        }
                    }
                }
            }
        }
        q40_commit();
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[mt][nt][l] = 0.0f;
            }
        }
    }

    // Stage st's MMAs.
    auto compute = [&](const int st) {
        const uint32_t * sw = q40_smem + (st % S) * C::STAGE_WORDS;
        const uint32_t * sq = sw + C::W_WORDS;
        const float    * sd = (const float *) (sq + C::Q_WORDS);
        const uint32_t * sa = sw + (st & 1) * 2 + (wm * 16 * MT + g) * C::WRS;
        // Block 2 bp + E of the stage.
        auto block = [&](const int bp, auto ev) {
            constexpr int E = decltype(ev)::value;
            const int bl = 2 * bp + E;
            // The warp's weight fragments: rows wm * 16 MT + mt * 16 + g (+ 8), as in the first
            // version (an even block's quants straddle words 9m .. 9m + 4, the high nibbles of
            // slot q at bits 16 .. of its words q, q + 1; an odd block's are words 9m + 5 ..).
            uint32_t a[MT][4];
            float da[MT][2], mda[MT][2];
            if constexpr (C::RP) {
                // The warp's unit wm: its 4 rows' words q of block bl (j = 2 mt + h) and scales.
                const uint32_t * su = sw + wm * Q40_RP_WORDS;
                const uint4 qv = *reinterpret_cast<const uint4 *>(su + bl * 128 + (g * 4 + q) * 4);
                const uint2 dv = *reinterpret_cast<const uint2 *>(su + 512 + bl * 16 + g * 2);
                const uint32_t qw[4] = { qv.x, qv.y, qv.z, qv.w };
                const uint32_t dw[2] = { dv.x, dv.y };
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const uint32_t hi = qw[2 * mt + h];
                        const unsigned short dh = (unsigned short) (h ? dw[mt] >> 16 : dw[mt] & 0xFFFF);
                        a[mt][h]     = q40_s8x16(hi << 4);  // K slot q: values 4q .. 4q + 3
                        a[mt][h + 2] = q40_s8x16(hi);       // K slot q + 4: values 16 + 4q ..
                        da[mt][h]  = __half2float(__ushort_as_half(dh));
                        mda[mt][h] = __fmul_rn(da[mt][h], -Q40_MAGIC);
                    }
                }
            } else {
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t * rw = sa + (mt * 16 + h * 8) * C::WRS + 9 * bp;
                    uint32_t lo, hi;
                    unsigned short dh;
                    if constexpr (E == 0) {
                        const uint32_t x0 = rw[q], x1 = rw[q + 1];
                        lo = __funnelshift_r(x0, x1, 12);
                        hi = __funnelshift_r(x0, x1, 16);
                        dh = (unsigned short) (rw[0] & 0xFFFF);
                    } else {
                        hi = rw[5 + q];
                        lo = hi << 4;
                        dh = (unsigned short) (rw[4] >> 16);
                    }
                    a[mt][h]     = q40_s8x16(lo);  // K slot q: values 4q .. 4q + 3
                    a[mt][h + 2] = q40_s8x16(hi);  // K slot q + 4: values 16 + 4q ..
                    da[mt][h]  = __half2float(__ushort_as_half(dh));
                    mda[mt][h] = __fmul_rn(da[mt][h], -Q40_MAGIC);
                }
            }
            }
            const uint32_t * bq = sq + bl * BN * 8;
            const float    * bd = sd + bl * C::D_BL;
#pragma unroll
            for (int pi = 0; pi < NT / 2; pi++) {
                const int pr = wn * (NT / 2) + pi;  // n8 tile pair among the BN / 16
                const uint4  bv = *reinterpret_cast<const uint4 *>(bq + ((pr * 8 + g) * 4 + q) * 4);
                const float4 dv = *reinterpret_cast<const float4 *>(bd + pr / 8 * 144 + q * 36 + pr % 8 * 4);
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int nt = 2 * pi + e;
                    const int b0 = (int) (e ? bv.z : bv.x), b1 = (int) (e ? bv.w : bv.y);
                    const float d0 = e ? dv.z : dv.x, d1 = e ? dv.w : dv.y;
#pragma unroll
                    for (int mt = 0; mt < MT; mt++) {
                        float * o = acc[mt][nt];
                        int d[4];
                        q40_mma(d, a[mt], b0, b1);
                        // t' = 16 fl(C * dA) from M + 16 C, then ggml's FFMA with dB / 16.
                        const float t0 = __fmaf_rn(__int_as_float(d[0]), da[mt][0], mda[mt][0]);
                        const float t1 = __fmaf_rn(__int_as_float(d[1]), da[mt][0], mda[mt][0]);
                        const float t2 = __fmaf_rn(__int_as_float(d[2]), da[mt][1], mda[mt][1]);
                        const float t3 = __fmaf_rn(__int_as_float(d[3]), da[mt][1], mda[mt][1]);
                        o[0] = __fmaf_rn(t0, d0, o[0]);
                        o[1] = __fmaf_rn(t1, d1, o[1]);
                        o[2] = __fmaf_rn(t2, d0, o[2]);
                        o[3] = __fmaf_rn(t3, d1, o[3]);
                    }
                }
            }
        };
#pragma unroll
        for (int bp = 0; bp < SB / 2; bp++) {
            block(bp, q40_k<0>());
            block(bp, q40_k<1>());
        }
    };

#pragma unroll
    for (int s = 0; s < S - 1; s++) {
        load(s);
    }
    for (int st = 0; st < nst; st++) {
        q40_wait<S - 2>();
        __syncthreads();  // stage st landed for all; slot (st - 1) % S read by all
        load(st + S - 1);
        compute(st);
    }
    q40_wait<0>();

    if constexpr (EPI) {
        // Column c's 32 gate sums at se[c * ES ..], its up sums at se[(BN + c) * ES ..] (ES 36:
        // the stores below and the 16-byte loads after hit distinct banks).
        constexpr int ES = 36;
        static_assert(2 * BN * ES * 4 <= C::SMEM, "GeGLU staging");
        float * se = reinterpret_cast<float *>(q40_smem);
        __syncthreads();  // every warp is done with the ring
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    const int r = mt * 16 + g + (l >> 1) * 8;
                    const int c = wn * 8 * NT + nt * 8 + 2 * q + (l & 1);
                    se[(wm * BN + c) * ES + r] = acc[mt][nt][l];
                }
            }
        }
        __syncthreads();
        // 8 lanes a column (an aligned group of a warp), lane s of them values 4s .. 4s + 3 of
        // its block: q40_quant_geglu's values and q40_quant4's quantization.
        for (int t = tid; t < BN * 8; t += C::THREADS) {
            const int c = t / 8, s = t % 8;
            const float4 gv = *reinterpret_cast<const float4 *>(se + c * ES + 4 * s);
            const float4 uv = *reinterpret_cast<const float4 *>(se + (BN + c) * ES + 4 * s);
            float4 v;
            v.x = q40_gelu(gv.x) * uv.x;
            v.y = q40_gelu(gv.y) * uv.y;
            v.z = q40_gelu(gv.z) * uv.z;
            v.w = q40_gelu(gv.w) * uv.w;
            q40_quant4<1>(p.oq, p.od, p.n_pad, v, col0 + c, it * HR + 4 * s, col0 + c < p.n);
        }
        return;
    }

    // The tile's sums to dst (ggml's write back: columns past n are left out).
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = row0 + wm * 16 * MT + mt * 16 + g + (l >> 1) * 8;
                const int j = col0 + wn * 8 * NT + nt * 8 + 2 * q + (l & 1);
                if (j < p.n) {
                    p.dst[(long long) j * p.dst_ld + i] = acc[mt][nt][l];
                }
            }
        }
    }
#else
    __trap();
#endif
}

using q40_op  = q40_cfg<64, 128, 2, 2, 2, 2, 4, 1>;   // the op's: 4 warps of 32 x 64, 2 CTAs an SM
using q40_128 = q40_cfg<128, 128, 4, 2, 3, 1, 8, 0>;  // 8 warps of 32 x 64, 1 CTA an SM
using q40_rp  = q40_cfg<64, 128, 2, 2, 2, 2, 4, 1, 1>;  // the op's on repacked weights

static_assert(q40_op::SMEM == 47616, "src/model/q40_mm.rs SMEM_BYTES");
static_assert(q40_rp::SMEM <= q40_op::SMEM, "the GeGLU pair's launch");

}  // namespace

extern "C" {

// The repack's launch arguments (src/model/q40_mm.rs, `RepackParams`): a Q4_0 weight of `rows`
// rows (a multiple of 32) of `k` values (of 128), rows w_nb1 bytes apart, into `out`
// (rows * k / 32 * 18 bytes, 16-byte aligned) in the layout above q40_repack.
struct legion_q40_repack {
    const void * w;
    void * out;
    long long w_nb1;
    int k;
    int rows;
};

}

namespace {

// One output word a thread (a grid-stride loop): chunk x = u * nst + st of unit u, stage st;
// its word i < 512 quants, else two scales.
__global__ void __launch_bounds__(256) q40_repack(const legion_q40_repack p) {
    const int nst = p.k / 128;
    const long long words = (long long) p.rows / Q40_RP_UNIT * nst * Q40_RP_WORDS;
    const unsigned char * w = (const unsigned char *) p.w;
    uint32_t * out = (uint32_t *) p.out;
    for (long long x = (long long) blockIdx.x * blockDim.x + threadIdx.x; x < words;
         x += (long long) gridDim.x * blockDim.x) {
        const long long ch = x / Q40_RP_WORDS;
        const int i = (int) (x - ch * Q40_RP_WORDS);
        const long long u = ch / nst;
        const int st = (int) (ch - u * nst);
        // The unit row of A word j of lane g: 16 (j / 2) + 8 (j % 2) + g.
        auto row = [&](const int j, const int g) {
            return u * Q40_RP_UNIT + (j / 2) * 16 + (j % 2) * 8 + g;
        };
        uint32_t v;
        if (i < 512) {
            const int j = i % 4, q = i / 4 % 4, g = i / 16 % 8, bl = i / 128;
            const unsigned char * b = w + row(j, g) * p.w_nb1 + (long long) (st * 4 + bl) * 18 + 2 + 4 * q;
            const unsigned short * h = (const unsigned short *) b;  // 2-byte aligned
            v = (uint32_t) h[0] | ((uint32_t) h[1] << 16);
        } else {
            v = 0;
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int hx = 2 * (i - 512) + e;
                const int j = hx % 4, g = hx / 4 % 8, bl = hx / 32;
                const unsigned char * b = w + row(j, g) * p.w_nb1 + (long long) (st * 4 + bl) * 18;
                v |= (uint32_t) *(const unsigned short *) b << (16 * e);
            }
        }
        out[x] = v;
    }
}

}  // namespace

extern "C" {

// A matmul of the table: its quantization and matmul kernels, shared memory, threads, tile.
struct legion_q40_kernel {
    const void * quant;
    const void * gemm;
    int smem;
    int threads;
    int bm;
    int bn;
    const char * name;
    int repacked;  // reads the weights repacked (q40_repack), RP 1
};

// The op's matmul first, then the others the timing test runs (src/model/q40_mm.rs, `Kernel`).
extern const legion_q40_kernel legion_q40_mm_table[] = {
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_op>),
      q40_op::SMEM, q40_op::THREADS, q40_op::BM, q40_op::BN, "q40_mm", 0 },
    { reinterpret_cast<const void *>(&q40_quant<0>), reinterpret_cast<const void *>(&q40_gemm_v1),
      Q40_STAGES * Q40_STAGE_WORDS * 4, Q40_THREADS, Q40_BM, Q40_BN, "first version", 0 },
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_128>),
      q40_128::SMEM, q40_128::THREADS, q40_128::BM, q40_128::BN, "128 x 128 tiles", 0 },
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_op, 0, 0>),
      q40_op::SMEM, q40_op::THREADS, q40_op::BM, q40_op::BN, "no prefetch", 0 },
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_rp>),
      q40_rp::SMEM, q40_rp::THREADS, q40_rp::BM, q40_rp::BN, "repacked", 1 },
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_rp, 0, 0>),
      q40_rp::SMEM, q40_rp::THREADS, q40_rp::BM, q40_rp::BN, "repacked, no prefetch", 1 },
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm<q40_rp, 0, 2>),
      q40_rp::SMEM, q40_rp::THREADS, q40_rp::BM, q40_rp::BN, "repacked, prefetch 2", 1 },
};
extern const int legion_q40_mm_table_len = sizeof(legion_q40_mm_table) / sizeof(legion_q40_mm_table[0]);

// The fused quantizations (src/model/q40_mm.rs, `cuda::q40_quant`), all in layout 1 (the op's
// matmul's): the GEGLU's (grid (n, k / 256), 64 threads), the norm's of fewer than 1024 values
// and of 1024 or more (grid n, 256 threads). Their arguments are a legion_q40_quant.
extern const void * const legion_q40_quant_kernels[3] = {
    reinterpret_cast<const void *>(&q40_quant_geglu),
    reinterpret_cast<const void *>(&q40_quant_norm<256>),
    reinterpret_cast<const void *>(&q40_quant_norm<1024>),
};

// The GeGLU pair (src/model/q40_mm.rs, `cuda::q40_geglu`): the op's matmul over a gate and an
// up weight, grid ntx * (rows / 32), q40_op's threads and shared memory, writing the quantized
// GeGLU. Its arguments are a legion_q40_mm.
extern const void * const legion_q40_geglu_gemm = reinterpret_cast<const void *>(&q40_gemm<q40_op, 1>);

// The same on repacked gate and up weights (q40_rp, the same shared memory or less).
extern const void * const legion_q40_geglu_gemm_rp = reinterpret_cast<const void *>(&q40_gemm<q40_rp, 1>);

// The repack (src/model/q40_mm.rs, `cuda::repack`): 256 threads, any grid. Its arguments are a
// legion_q40_repack.
extern const void * const legion_q40_repack_kernel = reinterpret_cast<const void *>(&q40_repack);

}
