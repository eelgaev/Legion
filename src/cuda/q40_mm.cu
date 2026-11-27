// Q4_0 prompt matmuls with ggml-cuda's MMQ bits from a pipelined main loop (src/model/q40_mm.rs):
// the quantization of the F32 input (ggml's quantize_mmq_q8_1 values for a Q4_0 weight), then
// the matmul, mma.sync m16n8k32 s8 x s8 -> s32 on 128 x 128 output tiles, the weights and the
// quantized input streamed by cp.async through a three-stage shared-memory ring.
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
// The nibbles' q - 8 as int8 is ((q & 0xF) + 0x78) ^ 0x80 per byte (no carries: 0x78 + 15 <
// 0x100), the values ggml's __vsub4 gives.
//
// Layout. A CTA owns rows it * 128 .. and columns jt * 128 .. (tiles in groups of 8 row tiles,
// column tiles slowest in a group, so the CTAs of a wave share weight rows and input columns in
// L2); 8 warps of 32 rows x 64 columns. A stage is 4 K blocks (128 values): the CTA's weight rows'
// 72 bytes each (8-byte copies; rows 20 words apart, so a warp's fragment reads hit 32 distinct
// banks), the input's quants of the 128 columns (8 words a column a block, slots q and q + 4
// adjacent: a lane's B fragment is one 8-byte read) and their scales.

#include <cuda_fp16.h>

#include <cstdint>

extern "C" {

// The op's launch arguments (src/model/q40_mm.rs, `Params`).
struct legion_q40_mm {
    const void * w;    // Q4_0 weight, rows w_nb1 bytes apart
    const float * x;   // F32 input, columns x_ld floats apart
    int * yq;          // quantized input [k / 32][n_pad][8 words]
    float * yd;        // its scales [k / 32][n_pad] (F16 values as F32)
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
};

}

namespace {

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

static_assert(Q40_STAGES * Q40_STAGE_WORDS * 4 == 86016, "src/model/q40_mm.rs SMEM_BYTES");

// The matmul's helpers: device code for sm_80 and newer only (cp.async).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

constexpr int   Q40_GROUP      = 8;  // row tiles a raster group
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

#endif

// ggml's quantize_mmq_q8_1 values for a Q4_0 weight (MMQ_Q8_1_DS_LAYOUT_DS4): each 32-value
// block's absmax over 8 threads of 4 values, d_inv = 127 / amax, roundf(x * d_inv), d = 1 / d_inv
// stored as F16. Column col's block kb: its 8 quant words at yq[(kb * n_pad + col) * 8], slot s
// (values 4s .. 4s + 3) at word 2s for s < 4, 2(s - 4) + 1 else; its scale at yd[kb * n_pad +
// col]. Grid (n, k / 256), 64 threads.
__global__ void __launch_bounds__(64) q40_quant(const legion_q40_mm p) {
    const int i0  = blockIdx.y * 256 + threadIdx.x * 4;
    const int col = blockIdx.x;
    const float * src = p.x + (long long) col * p.x_ld + i0;
    const float4 xi = (reinterpret_cast<uintptr_t>(src) & 15) == 0
        ? *reinterpret_cast<const float4 *>(src)
        : make_float4(src[0], src[1], src[2], src[3]);
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
    const int s  = threadIdx.x % 8;
    const long long at = (long long) kb * p.n_pad + col;
    p.yq[at * 8 + (s < 4 ? 2 * s : 2 * (s - 4) + 1)] = *reinterpret_cast<const int *>(&q);
    if (s == 0) {
        p.yd[at] = __half2float(__float2half(d));
    }
}

// The matmul: a CTA per 128 x 128 tile (grid ntx * nty), 256 threads, 86016 bytes of dynamic
// shared memory.
__global__ void __launch_bounds__(Q40_THREADS, 1) q40_gemm(const legion_q40_mm p) {
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

}  // namespace

// The op's kernels (src/model/q40_mm.rs): the quantization, the matmul.
extern "C" const void * const legion_q40_mm_kernels[2] = {
    reinterpret_cast<const void *>(&q40_quant),
    reinterpret_cast<const void *>(&q40_gemm),
};
