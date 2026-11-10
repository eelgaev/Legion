// NVFP4 dense matvecs of 1-8 columns with ggml's bits on tensor cores (#1133): ggml-cuda's
// quantize_q8_1 + mul_mat_vec_q<GGML_TYPE_NVFP4> pair (mmvq.cu, vecdotq.cuh, quantize.cu, MIT)
// for `w @ x` with a 2D weight, optionally times a one-value scale (ggml's `mul` after it), bit
// for bit. The host side, the shapes and the CPU reference are in src/model/core/nvfp4_mvq.rs,
// which launches the kernels from `legion_nvfp4_mvq_kernels` with cudaLaunchKernelExC.
//
// ggml's MMVQ spends about five instructions per weight byte at 8 columns on the CUDA cores: the
// E2M1 table lookups (byte permutes), four dp4a per 16-value sub-block and column, then the
// float scale math. Here the integer part runs on the tensor cores and the float part replays
// ggml's order exactly:
//
// - The integer sum of each 16-value sub-block and column (ggml's `sumi`, four dp4a) comes from
//   one F16 mma.sync m16n8k16 (16 rows, one sub-block, 8 columns): the weights as F16 E2M1
//   values (cvt.rn.f16x2.e2m1x2, one instruction per byte), the activations as F16 2 * q of
//   their Q8_1 values (exact). Every product is an integer of at most 1524 in magnitude and
//   every sum one of at most 24384, so the F32 accumulation is exact: C is ggml's `sumi`, as
//   a float. Inside an MMA the K order is free when A and B agree, so a thread takes bytes
//   2t and 2t + 1 of a sub-block (values 2t, 2t + 8, 2t + 1, 2t + 9) and `quantize` stores the
//   activations in that order.
// - The float part is ggml's: per (row, column) and per ggml thread (kbx, half h), sum = 0,
//   then sum += d * sumi for the half's two sub-blocks, d = ue4m3 / 2 * yd (exact; yd stored
//   halved, so d = ue4m3 * yd), added to the thread's partial sum in its K order (kbx = r,
//   r + blocks_per_iter, ...); then the partial sums of ggml's warps 1.. added to warp 0's in
//   order, then warp_reduce_sum's butterfly over the 32 lanes (lane 2 * (r % 16) + h). The
//   butterfly is a fixed tree (each add commutes), so a CTA's 4 warps take the residues
//   r = w + 4j: the tree's first two levels (lanes 16 and 8 apart: r % 16 bits 3 and 2) are in
//   each thread, the last three (r % 16 bits 1 and 0, then h) across the warps in shared
//   memory. ggml runs 4 warps (blocks_per_iter 64) up to 4 columns, 2 (32) at 5-8: the
//   template's NWG.
// - The epilogue: times the scale when there is one (ggml's `mul`, one rounding).
// - The weights reach shared memory by cp.async in stages of 16 rows x 16 blocks (576
//   contiguous bytes a row, in row order, two at a time, double-buffered): each thread's 2-byte
//   fragments straight from global memory touched 8 rows a load instruction and ran at
//   about half ggml's bandwidth.
// - `quantize` is ggml's quantize_q8_1 value for value (one thread per value, a warp per
//   32-value block, amax by ggml's warp reduction, d = amax / 127, q = roundf(x / d)) into
//   device-global scratch in the layout above: the op's two kernels run back to back on the
//   compute stream, as ggml's pair shares its pool buffer.
// - On sm_90+ every kernel waits for the kernel ahead of it (griddepcontrol.wait, ggml's
//   ggml_cuda_pdl_sync) before it reads activations, and lets the next one launch early
//   (ggml_cuda_pdl_lc), as ggml-cuda launches its kernels on a GB10.
//
// The matvec needs the sm_120a/sm_121a conversion (the FP4 code of nvfp4_a4.cu); elsewhere it
// is a trap, and the host takes the op only where nvfp4_a4's probe says the device runs that
// code.

#include "ggml-cuda/common.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define LEGION_MVQ_MMA 1
#endif

extern "C" {
struct legion_nvfp4_mvq {
    const float * x;        // F32 [n][k]
    int64_t       x_nb1;    // floats between x's columns
    const void *  w;        // NVFP4 [rows][k / 64]
    int64_t       w_nb1;    // bytes between rows
    float *       dst;      // F32 [n][rows]
    int64_t       dst_nb1;  // floats between dst's columns
    const float * scale;    // F32 [1], or null
    int32_t       k;        // a multiple of 64, at most MAX_K
    int32_t       rows;     // a multiple of 16 * the variant's RT
    int32_t       n;        // 1-8
};
}

namespace {

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// src/model/core/nvfp4_mvq.rs's MAX_K and MAX_N.
constexpr int MAX_K = 65536;
constexpr int MAX_N = 8;

// The activations as F16 2 * q: [column][kbx][MMA thread t of 4][sub-block][4 values], 32
// bytes a (column, kbx, t): two uint4 loads.
__device__ uint4 scratch_xh[MAX_N * MAX_K * 2 / 16];
// The Q8_1 scales yd / 2: [kbx][t][Q8 block of the kbx][column 2t, 2t + 1].
__device__ float4 scratch_yd[MAX_K / 64 * 4];

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n).
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_nvfp4_mvq p) {
    pdl_launch_dependents();
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= p.k) {
        return;
    }
    const int t = blockIdx.y;

    pdl_wait();
    const float xi = p.x[t * p.x_nb1 + i0];
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    // Value e of sub-block s of block kbx goes to thread (e % 8) / 2 of the MMA, at its slot
    // (e % 2) * 2 + e / 8 (see the header).
    const int kbx = i0 >> 6;
    const int s   = (i0 >> 4) & 3;
    const int e   = i0 & 15;
    const int th  = (e & 7) >> 1;
    const int pos = ((e & 1) << 1) | (e >> 3);
    __half * xh = reinterpret_cast<__half *>(scratch_xh);
    xh[(((int64_t) t * (p.k >> 6) + kbx) * 4 + th) * 16 + s * 4 + pos] = __int2half_rn(2 * q);

    if ((i0 & (QK8_1 - 1)) > 0) {
        return;
    }
    // ggml's ds.x (make_half2(d, sum)): d rounded to F16; halved here, exactly.
    const float yd = __half2float(__float2half_rn(d)) * 0.5f;
    float * ys = reinterpret_cast<float *>(scratch_yd);
    ys[((kbx * 4 + (t >> 1)) * 2 + ((i0 >> 5) & 1)) * 2 + (t & 1)] = yd;
}

#ifdef LEGION_MVQ_MMA
// The four UE4M3 scales of an NVFP4 block as floats, as ggml_cuda_ue4m3_to_fp32 without its
// halving (codes 0x7F and 0xFF give 0).
__device__ __forceinline__ void ue4m3x4(uint32_t d, float (&out)[4]) {
    const uint32_t nan = ((d & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
    d &= ~((nan >> 7) * 0xffu);
    const __half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (d & 0xffffu), __NV_E4M3));
    const __half2 hi = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (d >> 16), __NV_E4M3));
    const float2 a = __half22float2(lo);
    const float2 b = __half22float2(hi);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
}

// Two bytes of E2M1 pairs as F16 pairs: the low byte into `lo`, the high one into `hi`, each
// byte's low nibble in the low half.
__device__ __forceinline__ void e2m1x4(uint16_t v, uint32_t & lo, uint32_t & hi) {
    asm("{\n.reg .b8 b0, b1;\n"
        "mov.b16 {b0, b1}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n}"
        : "=r"(lo), "=r"(hi)
        : "h"(v));
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}
#endif

constexpr int MV_WARPS = 4;
// A stage: 16 rows of 16 consecutive NVFP4 blocks (576 bytes a row) in shared memory, rows 592
// bytes apart (37 uint4: the 8 rows a load instruction reads start in 8 different banks).
constexpr int STAGE_BLOCKS = 16;
constexpr int STAGE_ROW    = STAGE_BLOCKS * 36 + 16;
constexpr int STAGE_BYTES  = 16 * STAGE_ROW;
// Stages load GROUP at a time (1152 contiguous bytes a row: 576-byte pieces of 16 rows ran at
// about 80% of ggml's bandwidth), double-buffered.
constexpr int GROUP  = 2;
constexpr int NSTAGE = 2 * GROUP;

#ifdef LEGION_MVQ_MMA
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t) __cvta_generic_to_shared(smem)),
                 "l"(gmem)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// Blocks s0.. (at most GROUP stages, a multiple of 4 short of the row end) of the CTA's 16 rows
// into the GROUP consecutive stages from `buf`, by uint4s (a stage row is 36 of them).
__device__ __forceinline__ void group_load(uint8_t * buf, const uint8_t * w0, int64_t w_nb1, int s0, int bpr) {
    const int nb  = min(GROUP * STAGE_BLOCKS, bpr - s0);
    const int cnt = nb * 36 / 16;
    for (int i = threadIdx.x; i < 16 * cnt; i += MV_WARPS * WARP_SIZE) {
        const int row = i / cnt;
        const int u   = i - row * cnt;
        cp_async16(buf + (u / 36) * STAGE_BYTES + row * STAGE_ROW + (u % 36) * 16,
                   w0 + row * w_nb1 + (int64_t) s0 * 36 + u * 16);
    }
}
#endif

// `w @ x` for 16 rows a CTA, with ggml's MMVQ order for NWG of its warps (4: 1-4 columns, 2:
// 5-8). Thread (g, t) of a warp (lane 4g + t) holds rows g and g + 8 and columns 2t, 2t + 1
// (the MMA's C layout). ggml's warp l (of NWG) walks the residues 16 l + 0..15 (its lane
// 2 (r % 16) + h), each over the blocks r + BPI i (BPI = 16 NWG, ggml's blocks_per_iter).
// The CTA walks the row in order, a stage of 16 blocks at a time: stage s0 is ggml warp
// l = s0 / 16 % NWG's, and warp w here holds its residues 16 l + w + 4 P (P = 0..3), each
// thread with a partial sum for every ggml warp, added in ggml's warp order at the end.
template <int NWG>
__global__ void __launch_bounds__(MV_WARPS * WARP_SIZE, 2) mv(const legion_nvfp4_mvq p) {
    pdl_launch_dependents();
#ifdef LEGION_MVQ_MMA
    constexpr int BPI = 16 * NWG;
    static_assert(NWG % GROUP == 0, "a group is whole slots of one ggml iteration");
    __shared__ __align__(16) uint8_t stage[NSTAGE][STAGE_BYTES];
    __shared__ float part[MV_WARPS][2][16 * 8];

    const int w    = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int bpr  = p.k >> 6;
    const int row0 = blockIdx.x * 16;
    const uint8_t * w0 = (const uint8_t *) p.w + (int64_t) row0 * p.w_nb1;

    const bool col_ok = g < p.n;
    const uint4 * xh = scratch_xh + ((int64_t) g * bpr * 4 + t) * 2;

    // tmp[l][P][h][c]: ggml warp l's lane 2 * (w + 4P) + h; c is the C slot (row g or g + 8,
    // column 2t or 2t + 1).
    float tmp[NWG][4][2][4];
#pragma unroll
    for (int l = 0; l < NWG; ++l) {
#pragma unroll
        for (int P = 0; P < 4; ++P) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    tmp[l][P][h][c] = 0.0f;
                }
            }
        }
    }

    // The weights are no other kernel's output: their first two groups go out before the wait.
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (q * GROUP * STAGE_BLOCKS < bpr) {
            group_load(stage[q * GROUP], w0, p.w_nb1, q * GROUP * STAGE_BLOCKS, bpr);
        }
        cp_async_commit();
    }

    pdl_wait();

    for (int it0 = 0; it0 < bpr; it0 += BPI) {
#pragma unroll
        for (int l = 0; l < NWG; ++l) {
            const int s0 = it0 + STAGE_BLOCKS * l;
            if (s0 >= bpr) {
                break;
            }
            // NWG is a multiple of GROUP: a group starts at an l of GROUP's multiples.
            const int  si    = s0 / STAGE_BLOCKS;
            const bool first = l % GROUP == 0;

            // This stage's activations (L2-resident) while its weights land.
            uint4  b01[4], b23[4];
            float4 y[4];
#pragma unroll
            for (int P = 0; P < 4; ++P) {
                const int kbx = s0 + w + 4 * P;
                b01[P] = make_uint4(0, 0, 0, 0);
                b23[P] = make_uint4(0, 0, 0, 0);
                y[P]   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (kbx < bpr) {
                    if (col_ok) {
                        b01[P] = xh[(int64_t) kbx * 8];
                        b23[P] = xh[(int64_t) kbx * 8 + 1];
                    }
                    y[P] = scratch_yd[kbx * 4 + t];
                }
            }
            if (first) {
                // This group's loads (the next group's may still be in flight).
                cp_async_wait<1>();
                __syncthreads();
            }

            const uint8_t * sb = stage[si % NSTAGE];
#pragma unroll
            for (int P = 0; P < 4; ++P) {
                const int bi = w + 4 * P;
                if (s0 + bi >= bpr) {
                    continue;
                }
                const uint8_t * r0 = sb + g * STAGE_ROW + bi * 36;
                const uint8_t * r1 = r0 + 8 * STAGE_ROW;
                float dw[2][4];
                ue4m3x4(*(const uint32_t *) r0, dw[0]);
                ue4m3x4(*(const uint32_t *) r1, dw[1]);
                const uint32_t b[4][2] = {{b01[P].x, b01[P].y}, {b01[P].z, b01[P].w},
                                          {b23[P].x, b23[P].y}, {b23[P].z, b23[P].w}};
                // yd / 2 of Q8 block 2 kbx + h, column 2t + (c & 1).
                const float yv[2][2] = {{y[P].x, y[P].y}, {y[P].z, y[P].w}};
                float cs[4][4];
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    uint32_t a0, a1, a2, a3;
                    e2m1x4(*(const uint16_t *) (r0 + 4 + 8 * s + 2 * t), a0, a2);
                    e2m1x4(*(const uint16_t *) (r1 + 4 + 8 * s + 2 * t), a1, a3);
                    mma16816(cs[s], a0, a1, a2, a3, b[s][0], b[s][1]);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        // vec_dot_nvfp4_q8_1's half h: sub-blocks 2h and 2h + 1.
                        float sum = 0.0f;
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const float d = dw[c >> 1][2 * h + i] * yv[h][c & 1];
                            sum += d * cs[2 * h + i][c];
                        }
                        tmp[l][P][h][c] += sum;
                    }
                }
            }
            if (l % GROUP == GROUP - 1) {
                // Everyone is done with this group's buffers before the group after next's
                // loads overwrite them.
                __syncthreads();
                const int ns = (si + 1 + GROUP) * STAGE_BLOCKS;
                if (ns < bpr) {
                    group_load(stage[(si / GROUP % 2) * GROUP], w0, p.w_nb1, ns, bpr);
                }
                cp_async_commit();
            }
        }
    }
    cp_async_wait<0>();

    // ggml's cross-warp adds: warp 0's sums, then warps 1.. in order (a warp without blocks
    // adds its zeros, as ggml's does).
    float acc[4][2][4];
#pragma unroll
    for (int P = 0; P < 4; ++P) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[P][h][c] = tmp[0][P][h][c];
#pragma unroll
                for (int l = 1; l < NWG; ++l) {
                    acc[P][h][c] += tmp[l][P][h][c];
                }
            }
        }
    }

    // warp_reduce_sum's first two levels in the thread (lanes 16 apart: P and P ^ 2; then 8:
    // P and P ^ 1), the rest across the warps.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float v = (acc[0][h][c] + acc[2][h][c]) + (acc[1][h][c] + acc[3][h][c]);
            part[w][h][((c >> 1) * 8 + g) * 8 + 2 * t + (c & 1)] = v;
        }
    }
    __syncthreads();
    const bool  scaled = p.scale != nullptr;
    const float sc     = scaled ? *p.scale : 1.0f;
    for (int i = threadIdx.x; i < 16 * p.n; i += MV_WARPS * WARP_SIZE) {
        const int col = i / 16;
        const int row = i % 16;
        const int o   = row * 8 + col;
        // Lanes 4 apart (warps w and w ^ 2), 2 apart (w ^ 1), then 1 apart (h).
        float v = ((part[0][0][o] + part[2][0][o]) + (part[1][0][o] + part[3][0][o])) +
                  ((part[0][1][o] + part[2][1][o]) + (part[1][1][o] + part[3][1][o]));
        if (scaled) {
            v *= sc;
        }
        p.dst[col * p.dst_nb1 + row0 + row] = v;
    }
#else
    __trap();
#endif
}

}  // namespace

// The kernels for the host stubs: the quantization, then mv<NWG> for NWG 4 and 2
// (src/model/core/nvfp4_mvq.rs's kernel index).
extern "C" const void * const legion_nvfp4_mvq_kernels[3] = {
    reinterpret_cast<const void *>(&quantize),
    reinterpret_cast<const void *>(&mv<4>),
    reinterpret_cast<const void *>(&mv<2>),
};
