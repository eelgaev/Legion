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
// Layout. A CTA owns rows it * 128 .. and columns jt * 128 .. (tiles in groups of GROUP row
// tiles, column tiles slowest in a group, so the CTAs of a wave share weight rows and input
// columns in L2). A stage is SB K blocks: the CTA's weight rows' 18 * SB bytes each (9 copies of
// 2 * SB bytes a row; rows WRS words apart, so a warp's fragment reads hit distinct banks), the
// input's quants of the 128 columns (8 words a column a block) and their scales; a three-stage
// ring.
//
// Two configurations (q40_cfg): the op's, 2 K blocks a stage in 45 KB of shared memory and at
// most 128 registers a thread, so two CTAs share an SM (16 warps to hide the MMAs' and the
// shared-memory reads' latency, where one CTA of 8 warps left the tensor cores idle most of the
// time), warps of 16 rows x 128 columns (one A fragment a block), the input in layout 1 (one
// 16-byte read gives a lane the B fragments of two n8 tiles, another their scales), groups of 16
// row tiles (a wave of 96 CTAs: 16 row tiles x 6 column tiles, about as many weight bytes as
// input bytes); and the first version's, 4 K blocks a stage, 84 KB, one CTA an SM, warps of 32
// rows x 64 columns, layout 0, groups of 8, kept for the timing test (src/model/q40_mm/tests.rs).
// Both run the same chains, so they give the same bits.

#include <cuda_fp16.h>

#include <cstdint>

extern "C" {

// The op's launch arguments (src/model/q40_mm.rs, `Params`).
struct legion_q40_mm {
    const void * w;    // Q4_0 weight, rows w_nb1 bytes apart
    const float * x;   // F32 input, columns x_ld floats apart
    int * yq;          // quantized input [k / 32][n_pad][8 words] (layout of q40_quant)
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

constexpr int Q40_BM      = 128;  // tile rows
constexpr int Q40_BN      = 128;  // tile columns
constexpr int Q40_STAGES  = 3;
constexpr int Q40_THREADS = 256;

// A matmul configuration: SB K blocks a stage, weight rows WRS words apart in a stage (9 * SB / 2
// used), MINB CTAs an SM, raster groups of GROUP row tiles; the 8 warps are WARPS_M along the rows
// (16 MT rows each) by 8 / WARPS_M along the columns (8 NT each); the quantized input in layout
// LAYOUT (q40_quant). In layout 1 a block's 128 scales sit in shared memory as 4 rows of 32, 36
// words apart (the four 16-byte reads of a warp's lanes hit distinct banks).
template <int SB_, int WRS_, int MINB_, int GROUP_, int MT_, int LAYOUT_> struct q40_cfg {
    static constexpr int SB          = SB_;
    static constexpr int WRS         = WRS_;
    static constexpr int MINB        = MINB_;
    static constexpr int GROUP       = GROUP_;
    static constexpr int MT          = MT_;
    static constexpr int LAYOUT      = LAYOUT_;
    static constexpr int WARPS_M     = Q40_BM / (16 * MT);
    static constexpr int NT          = Q40_BN / 8 / (Q40_THREADS / 32 / WARPS_M);  // n8 tiles a warp
    static constexpr int D_BL        = LAYOUT == 0 ? Q40_BN : 4 * 36;  // scale words a block
    static constexpr int W_WORDS     = Q40_BM * WRS;
    static constexpr int Q_WORDS     = SB * Q40_BN * 8;
    static constexpr int D_WORDS     = SB * D_BL;
    static constexpr int STAGE_WORDS = W_WORDS + Q_WORDS + D_WORDS;
    static constexpr int CP          = 2 * SB;  // bytes a weight copy (9 a row a stage)
    static constexpr int SMEM        = Q40_STAGES * STAGE_WORDS * 4;
    static_assert(SB == 2 || SB == 4, "copies of 4 or 8 bytes");
    static_assert(9 * SB / 2 <= WRS, "a stage's weight row fits");
    static_assert(LAYOUT == 0 || NT % 2 == 0, "layout 1 reads two n8 tiles at once");
};

using q40_op = q40_cfg<2, 12, 2, 16, 1, 1>;  // the op's
using q40_v1 = q40_cfg<4, 20, 1, 8, 2, 0>;   // the first version's

static_assert(q40_op::SMEM == 46464, "src/model/q40_mm.rs SMEM_BYTES");
static_assert(q40_v1::SMEM == 86016, "src/model/q40_mm.rs SMEM_BYTES_V1");

// The matmul's helpers: device code for sm_80 and newer only (cp.async).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

constexpr int   Q40_MAGIC_BITS = 0x4B400000;
constexpr float Q40_MAGIC      = 12582912.0f;

// A weight copy of N (4 or 8) bytes.
template <int N> __device__ __forceinline__ void q40_cpw(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(gmem), "n"(N));
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
// stored as F16. Column col's block kb, slot s (values 4s .. 4s + 3) is word w = 2s for s < 4,
// 2(s - 4) + 1 else (so slots s and s + 4 are adjacent words).
// - Layout 0 (the first version's): the 8 words at yq[(kb * n_pad + col) * 8], the scale at
//   yd[kb * n_pad + col].
// - Layout 1 (the op's): permuted within each 128-column tile so that a lane's B fragments and
//   scales of two n8 tiles are 16 contiguous bytes each. Of tile column c = 8 nt + g (= 8 nt + 2 r
//   + e), word w (= 2 v + f) is at word ((nt / 2 * 8 + g) * 4 + v) * 4 + nt % 2 * 2 + f of the
//   tile's 1024 words of block kb (yq[(kb * n_pad + 128 jt) * 8 ..]); the scale at r * 32 + 2 nt
//   + e of its 128 (yd[kb * n_pad + 128 jt ..]).
// Grid (n, k / 256), 64 threads.
template <int LAYOUT> __global__ void __launch_bounds__(64) q40_quant(const legion_q40_mm p) {
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
    const int w  = s < 4 ? 2 * s : 2 * (s - 4) + 1;
    long long qat, dat;
    if constexpr (LAYOUT == 0) {
        const long long at = (long long) kb * p.n_pad + col;
        qat = at * 8 + w;
        dat = at;
    } else {
        const long long tile = (long long) kb * p.n_pad + (col & ~127);
        const int c = col & 127, nt = c / 8, g = c % 8;
        qat = tile * 8 + ((nt / 2 * 8 + g) * 4 + w / 2) * 4 + nt % 2 * 2 + w % 2;
        dat = tile + (g / 2) * 32 + 2 * nt + g % 2;
    }
    p.yq[qat] = *reinterpret_cast<const int *>(&q);
    if (s == 0) {
        p.yd[dat] = __half2float(__float2half(d));
    }
}

// The matmul of configuration C: a CTA per 128 x 128 tile (grid ntx * nty), 256 threads,
// C::SMEM bytes of dynamic shared memory.
template <class C> __global__ void __launch_bounds__(Q40_THREADS, C::MINB) q40_gemm(const legion_q40_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int SB = C::SB, WRS = C::WRS, GROUP = C::GROUP, CP = C::CP, MT = C::MT, NT = C::NT;
    extern __shared__ __align__(16) uint32_t q40_smem[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int wm = warp % C::WARPS_M, wn = warp / C::WARPS_M;

    // The tile: groups of GROUP row tiles, row tiles fastest within a group.
    const int b         = blockIdx.x;
    const int per_group = GROUP * p.ntx;
    const int grp       = b / per_group;
    const int r0        = grp * GROUP;
    const int gsz       = min(GROUP, p.nty - r0);
    const int in        = b - grp * per_group;
    const int it        = r0 + in % gsz;
    const int jt        = in / gsz;
    const int row0      = it * Q40_BM;
    const int col0      = jt * Q40_BN;
    const int nst       = p.k / (32 * SB);

    // This thread's copies, the same every stage: weight row tid / 2's copies 0 .. 4 (even
    // threads) or 5 .. 8 (odd), the quants' 16 bytes tid of every block, and (the first SB * 32
    // threads) 16 bytes of the scales.
    const int    wr = tid / 2, wc0 = (tid % 2) * 5, wcn = tid % 2 ? 4 : 5;
    const char * wg = (const char *) p.w + (long long) (row0 + wr) * p.w_nb1 + wc0 * CP;
    const int    ws = wr * WRS + wc0 * (CP / 4);
    const int  * qg = p.yq + (long long) col0 * 8 + tid * 4;
    const long long qstep = (long long) p.n_pad * 8;  // words a block
    const int    dbl = tid / 32, dch = tid % 32;
    const float * dg = p.yd + (long long) dbl * p.n_pad + col0 + dch * 4;
    const int    ds  = dbl * C::D_BL + (C::LAYOUT == 0 ? dch * 4 : dch / 8 * 36 + dch % 8 * 4);

    // Stage st's copies into ring slot st % 3 (an empty group past the last stage).
    auto load = [&](const int st) {
        if (st < nst) {
            uint32_t * sw = q40_smem + (st % Q40_STAGES) * C::STAGE_WORDS;
            uint32_t * sq = sw + C::W_WORDS;
            uint32_t * sd = sq + C::Q_WORDS;
            const char * w = wg + (long long) st * (SB * 18);
#pragma unroll
            for (int c = 0; c < 5; c++) {
                if (c < wcn) {
                    q40_cpw<CP>(sw + ws + c * (CP / 4), w + c * CP);
                }
            }
            const long long kb0 = (long long) st * SB;
#pragma unroll
            for (int bl = 0; bl < SB; bl++) {
                q40_cp16(sq + bl * Q40_BN * 8 + tid * 4, qg + (kb0 + bl) * qstep);
            }
            if (tid < SB * Q40_BN / 4) {
                q40_cp16(sd + ds, dg + kb0 * p.n_pad);
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

    load(0);
    load(1);
    for (int st = 0; st < nst; st++) {
        q40_wait<1>();
        __syncthreads();  // stage st landed for all; slot (st + 2) % 3 (stage st - 1) read by all
        load(st + 2);

        const uint32_t * sw = q40_smem + (st % Q40_STAGES) * C::STAGE_WORDS;
        const uint32_t * sq = sw + C::W_WORDS;
        const float    * sd = (const float *) (sq + C::Q_WORDS);
#pragma unroll
        for (int bl = 0; bl < SB; bl++) {
            // The warp's weight fragments: rows wm * 16 MT + mt * 16 + g (+ 8). Block bl starts
            // at byte 18 bl of the row's stage: an even block's scale is the low half of word 9m
            // (m = bl / 2) and its quants straddle words 9m .. 9m + 4; an odd block's scale is
            // the high half of word 9m + 4 and its quants words 9m + 5 .. 9m + 8.
            uint32_t a[MT][4];
            float da[MT][2], mda[MT][2];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t * rw = sw + (wm * 16 * MT + mt * 16 + h * 8 + g) * WRS + 9 * (bl / 2);
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
            const float    * bd = sd + bl * C::D_BL;
            uint4  bv2;  // layout 1: this lane's B fragments of n8 tiles nt & ~1 and nt | 1
            float4 dv2;  // and their scales
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                uint2  bv;
                float2 dv;
                if constexpr (C::LAYOUT == 0) {
                    const int c = wn * 8 * NT + nt * 8;
                    bv = *reinterpret_cast<const uint2 *>(bq + (c + g) * 8 + 2 * q);
                    dv = *reinterpret_cast<const float2 *>(bd + c + 2 * q);
                } else {
                    const int pr = (wn * NT + nt) / 2;  // the pair of n8 tiles in the tile
                    if (nt % 2 == 0) {
                        bv2 = *reinterpret_cast<const uint4 *>(bq + ((pr * 8 + g) * 4 + q) * 4);
                        dv2 = *reinterpret_cast<const float4 *>(bd + q * 36 + pr * 4);
                    }
                    bv = nt % 2 == 0 ? make_uint2(bv2.x, bv2.y) : make_uint2(bv2.z, bv2.w);
                    dv = nt % 2 == 0 ? make_float2(dv2.x, dv2.y) : make_float2(dv2.z, dv2.w);
                }
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
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

// Round 3's matmul configuration: BM x BN tiles, WARPS_M x WARPS_N warps of (BM / WARPS_M) x
// (BN / WARPS_N), a STAGES-deep ring of 4 K blocks a stage, MINB CTAs an SM, raster groups of
// GROUP row tiles; the input in layout 1.
template <int BM_, int BN_, int WARPS_M_, int WARPS_N_, int STAGES_, int MINB_, int GROUP_, int LATE_ = 0, int COLS_ = 0> struct q40_cfg3 {
    static constexpr int BM          = BM_;
    static constexpr int BN          = BN_;
    static constexpr int WARPS_M     = WARPS_M_;
    static constexpr int WARPS_N     = WARPS_N_;
    static constexpr int STAGES      = STAGES_;
    static constexpr int MINB        = MINB_;
    static constexpr int GROUP       = GROUP_;
    static constexpr int LATE        = LATE_;            // the next stage's copies after block 0
    static constexpr int COLS        = COLS_;            // groups of GROUP column tiles
    static constexpr int THREADS     = 32 * WARPS_M * WARPS_N;
    static constexpr int MT          = BM / WARPS_M / 16;
    static constexpr int NT          = BN / WARPS_N / 8;
    static constexpr int SB          = 4;
    static constexpr int WRS         = 20;               // words a weight row (18 + 2 of the shift)
    static constexpr int D_BL        = BN / 128 * 144;   // scale words a block: 4 rows of 32, 36 apart
    static constexpr int W_WORDS     = BM * WRS;
    static constexpr int Q_WORDS     = SB * BN * 8;
    static constexpr int D_WORDS     = SB * D_BL;
    static constexpr int STAGE_WORDS = W_WORDS + Q_WORDS + D_WORDS;
    static constexpr int SMEM        = STAGES * STAGE_WORDS * 4;
    static constexpr int WCH         = BM * 5;           // 16-byte weight copies a stage
    static constexpr int WCPT        = (WCH + THREADS - 1) / THREADS;
    static constexpr int QCH         = SB * BN * 2;      // 16-byte quant copies a stage
    static constexpr int QCPT        = QCH / THREADS;
    static constexpr int DCH         = SB * BN / 4;      // 16-byte scale copies a stage
    static_assert(MT >= 1 && NT % 2 == 0 && BN % 128 == 0, "warp tiles of n8 tile pairs");
    static_assert(QCH % THREADS == 0 && DCH <= THREADS, "copies");
    static_assert(SMEM <= 99 * 1024, "shared memory");
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// A 16-byte copy of which the first `src` bytes are read and the rest zero-filled.
__device__ __forceinline__ void q40_cp16z(void * smem, const void * gmem, const int src) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src));
}
#endif

// Round 3's matmul: a CTA per BM x BN tile (grid ntx * nty, C::THREADS threads, C::SMEM bytes of
// dynamic shared memory), the input in layout 1 (q40_quant<1>, n_pad a multiple of BN).
//
// The weights come in 16-byte copies: a stage's 72 bytes of a row start at byte 72 st of the row,
// 16-byte aligned on even stages, 8 bytes past it on odd ones, so the copy takes the 80 bytes
// from the aligned byte below (the last 8 of an even stage's zero-filled: past the stage, maybe
// past the weight) into the row's 20 words; the row's blocks start 2 words in on odd stages.
//
// A's int8 values are 16 (q - 8): (nibble << 4) ^ 0x80 per byte, one LOP3 a word for the high
// nibbles (two for the low ones), and the MMA's result reads as the float M + 16 C, so
// t = fma(M + 16 C, dA / 16, -M * dA / 16): the exact value C * dA again (dA / 16 and M * dA / 16
// are exact), rounded once, ggml's fl(C * dA).
template <class C> __global__ void __launch_bounds__(C::THREADS, C::MINB) q40_gemm3(const legion_q40_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int SB = C::SB, MT = C::MT, NT = C::NT, BM = C::BM, BN = C::BN, S = C::STAGES;
    extern __shared__ __align__(16) uint32_t q40_smem[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int wm = warp % C::WARPS_M, wn = warp / C::WARPS_M;

    // Groups of GROUP row tiles, column after column (or of GROUP column tiles, row after row).
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
#pragma unroll
    for (int i = 0; i < C::WCPT; i++) {
        const int c = tid + i * C::THREADS;
        woff[i]     = (c / 5) * (int) p.w_nb1 + (c % 5) * 16;
        wend[i]     = c % 5 == 4;
    }
    const long long qstep = (long long) p.n_pad * 8;  // words a block
    const int * qg = p.yq + (long long) col0 * 8;
    // Scale copy tid: block tid / (BN / 4), 4 scales tid % (BN / 4) of the block's BN.
    const int dbl = tid / (BN / 4), dch = tid % (BN / 4);
    const float * dg = p.yd + (long long) dbl * p.n_pad + col0 + dch * 4;
    const int ds = dbl * C::D_BL + dch / 32 * 144 + dch % 32 / 8 * 36 + dch % 8 * 4;

    auto load = [&](const int st) {
        if (st < nst) {
            uint32_t * sw = q40_smem + (st % S) * C::STAGE_WORDS;
            uint32_t * sq = sw + C::W_WORDS;
            uint32_t * sd = sq + C::Q_WORDS;
            const int odd = st & 1;
            const char * w = wg + (long long) (st >> 1) * 144 + odd * 64;
#pragma unroll
            for (int i = 0; i < C::WCPT; i++) {
                const int c = tid + i * C::THREADS;
                if (C::WCH % C::THREADS == 0 || c < C::WCH) {
                    q40_cp16z(sw + c * 4, w + woff[i], wend[i] && !odd ? 8 : 16);
                }
            }
            const long long kb0 = (long long) st * SB;
#pragma unroll
            for (int i = 0; i < C::QCPT; i++) {
                const int c = tid + i * C::THREADS, bl = c / (2 * BN), wi = c % (2 * BN);
                q40_cp16(sq + bl * BN * 8 + wi * 4, qg + (kb0 + bl) * qstep + wi * 4);
            }
            if (C::DCH == C::THREADS || tid < C::DCH) {
                q40_cp16(sd + ds, dg + kb0 * p.n_pad);
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

#pragma unroll
    for (int s = 0; s < S - 1; s++) {
        load(s);
    }
    for (int st = 0; st < nst; st++) {
        q40_wait<S - 2>();
        __syncthreads();  // stage st landed for all; slot (st - 1) % S read by all
        if (!C::LATE) {
            load(st + S - 1);
        }

        const uint32_t * sw = q40_smem + (st % S) * C::STAGE_WORDS;
        const uint32_t * sq = sw + C::W_WORDS;
        const float    * sd = (const float *) (sq + C::Q_WORDS);
        const uint32_t * sa = sw + (st & 1) * 2 + (wm * 16 * MT + g) * C::WRS;
#pragma unroll
        for (int bl = 0; bl < SB; bl++) {
            if (C::LATE && bl == 1) {
                load(st + S - 1);
            }
            uint32_t a[MT][4];
            float da[MT][2], mda[MT][2];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t * rw = sa + (mt * 16 + h * 8) * C::WRS + 9 * (bl / 2);
                    uint32_t lo, hi;
                    unsigned short dh;
                    if (bl % 2 == 0) {
                        const uint32_t x0 = rw[q], x1 = rw[q + 1];
                        lo = __funnelshift_r(x0, x1, 12);
                        hi = __funnelshift_r(x0, x1, 16);
                        dh = (unsigned short) (rw[0] & 0xFFFF);
                    } else {
                        hi = rw[5 + q];
                        lo = hi << 4;
                        dh = (unsigned short) (rw[4] >> 16);
                    }
                    a[mt][h]     = (lo & 0xF0F0F0F0u) ^ 0x80808080u;  // K slot q: 16 (q - 8)
                    a[mt][h + 2] = (hi & 0xF0F0F0F0u) ^ 0x80808080u;  // K slot q + 4
                    const float d = __half2float(__ushort_as_half(dh));
                    da[mt][h]  = __fmul_rn(d, 0.0625f);
                    mda[mt][h] = __fmul_rn(d, -0.0625f * Q40_MAGIC);
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
                        int d[4];
                        q40_mma(d, a[mt], b0, b1);
                        const float t0 = __fmaf_rn(__int_as_float(d[0]), da[mt][0], mda[mt][0]);
                        const float t1 = __fmaf_rn(__int_as_float(d[1]), da[mt][0], mda[mt][0]);
                        const float t2 = __fmaf_rn(__int_as_float(d[2]), da[mt][1], mda[mt][1]);
                        const float t3 = __fmaf_rn(__int_as_float(d[3]), da[mt][1], mda[mt][1]);
                        acc[mt][nt][0] = __fmaf_rn(t0, d0, acc[mt][nt][0]);
                        acc[mt][nt][1] = __fmaf_rn(t1, d1, acc[mt][nt][1]);
                        acc[mt][nt][2] = __fmaf_rn(t2, d0, acc[mt][nt][2]);
                        acc[mt][nt][3] = __fmaf_rn(t3, d1, acc[mt][nt][3]);
                    }
                }
            }
        }
    }
    q40_wait<0>();

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

// The variants the timing test compares (src/model/q40_mm.rs, `Kernel`).
using q40_r3a   = q40_cfg3<128, 128, 4, 2, 3, 1, 8>;        // 8 warps of 32 x 64
using q40_r3f   = q40_cfg3<64, 128, 2, 2, 2, 2, 16>;        // 4 warps of 32 x 64, 2 CTAs an SM
using q40_r3f8  = q40_cfg3<64, 128, 2, 2, 2, 2, 8>;
using q40_r3f32 = q40_cfg3<64, 128, 2, 2, 2, 2, 32>;
using q40_r3c4  = q40_cfg3<64, 128, 2, 2, 2, 2, 4, 0, 1>;   // groups of column tiles
using q40_r3c8  = q40_cfg3<64, 128, 2, 2, 2, 2, 8, 0, 1>;
using q40_r3c16 = q40_cfg3<64, 128, 2, 2, 2, 2, 16, 0, 1>;

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
};

#define Q40_R3(C, NAME) \
    { reinterpret_cast<const void *>(&q40_quant<1>), reinterpret_cast<const void *>(&q40_gemm3<C>), C::SMEM, \
      C::THREADS, C::BM, C::BN, NAME }

// The op's matmul first, then the others the timing test runs (src/model/q40_mm.rs).
extern const legion_q40_kernel legion_q40_mm_table[] = {
    Q40_R3(q40_r3f, "r3 64x128 x2 row group 16"),
    { reinterpret_cast<const void *>(&q40_quant<0>), reinterpret_cast<const void *>(&q40_gemm<q40_v1>), q40_v1::SMEM,
      256, 128, 128, "first version" },
    Q40_R3(q40_r3a, "r3 128x128 8w 32x64"),
    Q40_R3(q40_r3f8, "r3 64x128 x2 row group 8"),
    Q40_R3(q40_r3f32, "r3 64x128 x2 row group 32"),
    Q40_R3(q40_r3c4, "r3 64x128 x2 col group 4"),
    Q40_R3(q40_r3c8, "r3 64x128 x2 col group 8"),
    Q40_R3(q40_r3c16, "r3 64x128 x2 col group 16"),
};
extern const int legion_q40_mm_table_len = sizeof(legion_q40_mm_table) / sizeof(legion_q40_mm_table[0]);

}
