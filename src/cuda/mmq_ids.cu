// Expert-grid matmuls of Q4_K and Q5_1 experts for steps of 9-63 tokens: mul_mat_id with ggml's
// MMQ bits (#836, #814 for the V100 chain; opt-in, `--expert-mmq`). Shapes and the choice to use it are on the Rust side
// (src/model/core/mmq_ids.rs).
//
//   dst[t][j][r] = sum_e W_{ids[t][j]}[r][e] * X[t][j or 0][e]
//
// ggml-cuda (ggml-cuda/mmq.cuh, mmq-vec-dot.cuh, mmq-load-tiles.cuh, quantize.cu; MIT) computes
// these steps with MMQ: X quantized to Q8_1 per 32 values (ggml's DS4 layout: a half2 of scale
// and sum), then on tensor cores a tile of 128 rows x J >= n_tok columns per (expert, row block),
// one CTA per tile (stream-k at grid = tiles: every tile in one CTA, no fixup). Each output is
// summed in K order, one 32-value chunk c after the other:
//
//   sum += dmA.x*dsB.x*C;  sum += dmA.y*dsB.y;      (C: the exact int dot of the chunk)
//
// with dmA the weight block's (scale, -min) as ggml's tile loader builds it in half2. Only that
// chain decides the bits, so this file computes the same chain another way: a thread per row
// (its warp's rows staged through shared memory with coalesced cp.async copies, one stage ahead;
// no tensor cores, no columns beyond the expert's own tokens) keeps the sums of up to NT tokens in
// registers. The int dots are dp4a sums,
// exact as the mma's, and the two statements above are written the same way and built with the
// same flags (-O3 --use_fast_math) so they contract to the same FMAs.
//
// On a V100 (compute capability 7.0, no int8 tensor cores) ggml's MMQ takes its dp4a path
// instead (ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a / _q8_1_q8_1_dp4a with the ampere config's
// 128-row tiles, stream-k at grid = tiles again), whose chain per output is, in K order:
//
//   Q4_K: sum += vec_dot_q4_K_q8_1_impl_mmq(...)  per 64 values (two sub-blocks)
//   Q5_1: sum += vec_dot_q8_1_q8_1_impl<8>(...)   per 32-value block
//
// (vecdotq.cuh: the int dots, the half2 scale products and the float sums inside those calls).
// `dp4a` != 0 makes the same thread-per-row kernel call those two functions on the same values,
// so the result is MMQ's on that GPU bit for bit.
//
// Two kernels per op:
// - `prep`: rows 0..R-1 quantize X as ggml's quantize_mmq_q8_1 does (copied below, DS4 layout,
//   one row per (token, slot), or per token when the slots share X); the extra row of CTAs
//   groups the (token, slot) pairs by expert: the active experts in id order, each with its
//   pairs (in any order: each pair's outputs are computed alone).
// - `mm`: grid (row blocks, active experts); CTAs past the active count return at once.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdint>

extern "C" {

struct legion_mmq_ids {
    const char * w;        // expert 0; expert e at w + e * w_nb2, rows w_nb1 bytes apart
    int64_t w_nb1, w_nb2;
    const float * x;       // [n_tok][n_x][k]: x_nb1 bytes per slot, x_nb2 per token
    int64_t x_nb1, x_nb2;
    int32_t n_x;           // 1 (shared by all slots: gate, up) or n_used (down)
    const int32_t * ids;   // [n_tok][n_used], ids_nb1 bytes per token
    int64_t ids_nb1;
    float * dst;           // [n_tok][n_used][rows]
    int64_t dst_nb1, dst_nb2;
    int32_t k, rows, n_used, n_tok, n_expert;
    // Scratch: int8 qs [R][k] at 0, half2 ds [R][k/32] at ds_off, the plan (int32) at plan_off:
    // n_active, act_e[P], act_off[P + 1], pairs[P] with P = n_used * n_tok and R = n_x * n_tok.
    // A scratch buffer of the GPU kept per stream (outside the graphs: a graph tensor moved the
    // graph's layout and with it ggml-cuda's fusions).
    char * scratch;
    int64_t ds_off, plan_off;
    int32_t dp4a;          // 0: the tensor-core MMQ chain (sm_80+); else MMQ's dp4a chain (V100)
};

// Returns a cudaError_t.
int legion_mmq_ids_launch(int type, const legion_mmq_ids * p, void * stream);

}  // extern "C"

namespace {

constexpr int THREADS = 128;  // rows per CTA of `mm`; threads of `prep`
constexpr int NT      = 16;   // tokens an `mm` thread sums at once (more: another pass)

// ggml-cuda/mmq-load-tiles.cuh (MIT): a Q4_K block's 6-bit scales, four per int.
__device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

// ggml-cuda/quantize.cu's quantize_mmq_q8_1 (MIT), DS4 layout: 4 values per thread, 8 threads per
// 32 values, the same max, sum, shuffles and rounding. Writes row `r`'s values at qs + r*k and its
// half2 (scale, sum) per 32 values at ds + r*k/32.
__device__ __forceinline__ void quantize_row(const legion_mmq_ids & p, const int r) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (i0 >= p.k) {
        return;
    }
    const int t = r / p.n_x;
    const int j = r % p.n_x;
    const float4 * x4 = (const float4 *) ((const char *) p.x + t * p.x_nb2 + j * p.x_nb1);

    const float4 xi = x4[i0/4];
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    char4 * qs = (char4 *) (p.scratch + (int64_t) r * p.k);
    qs[i0/4] = q;
    if (i0 % 32 == 0) {
        half2 * ds = (half2 *) (p.scratch + p.ds_off);
        ds[(int64_t) r * (p.k/32) + i0/32] = make_half2(d, sum);
    }
}

// The pairs grouped by expert (one CTA): counts, a scan over the experts in id order, then each
// pair placed in its expert's range.
__device__ __forceinline__ void group_pairs(const legion_mmq_ids & p) {
    extern __shared__ int sh[];
    int * cnt = sh;                 // [n_expert]
    int * cur = sh + p.n_expert;    // [n_expert]
    __shared__ int scan_a[THREADS];
    __shared__ int scan_n[THREADS];

    const int P = p.n_used * p.n_tok;
    int * plan    = (int *) (p.scratch + p.plan_off);
    int * act_e   = plan + 1;
    int * act_off = act_e + P;
    int * pairs   = act_off + P + 1;

    for (int e = threadIdx.x; e < p.n_expert; e += blockDim.x) {
        cnt[e] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int e = *(const int32_t *) ((const char *) p.ids + (i / p.n_used) * p.ids_nb1 + (i % p.n_used) * 4);
        if (e >= 0 && e < p.n_expert) {
            atomicAdd(&cnt[e], 1);
        }
    }
    __syncthreads();

    const int per = (p.n_expert + THREADS - 1) / THREADS;
    const int e0  = min(p.n_expert, (int) threadIdx.x * per);
    const int e1  = min(p.n_expert, e0 + per);
    int na = 0, np = 0;
    for (int e = e0; e < e1; ++e) {
        na += cnt[e] > 0;
        np += cnt[e];
    }
    scan_a[threadIdx.x] = na;
    scan_n[threadIdx.x] = np;
    __syncthreads();
    for (int s = 1; s < THREADS; s <<= 1) {
        const int a = threadIdx.x >= s ? scan_a[threadIdx.x - s] : 0;
        const int n = threadIdx.x >= s ? scan_n[threadIdx.x - s] : 0;
        __syncthreads();
        scan_a[threadIdx.x] += a;
        scan_n[threadIdx.x] += n;
        __syncthreads();
    }
    int a   = scan_a[threadIdx.x] - na;
    int off = scan_n[threadIdx.x] - np;
    for (int e = e0; e < e1; ++e) {
        if (cnt[e] > 0) {
            act_e[a]   = e;
            act_off[a] = off;
            cur[e]     = off;
            ++a;
            off += cnt[e];
        }
    }
    if (threadIdx.x == THREADS - 1) {
        plan[0]    = a;
        act_off[a] = off;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int e = *(const int32_t *) ((const char *) p.ids + (i / p.n_used) * p.ids_nb1 + (i % p.n_used) * 4);
        if (e >= 0 && e < p.n_expert) {
            pairs[atomicAdd(&cur[e], 1)] = i;
        }
    }
}

// grid (ceil(k / 4 / THREADS), R + 1): quantize row blockIdx.y, or group the pairs (last row).
__global__ void __launch_bounds__(THREADS) mmq_ids_prep(const legion_mmq_ids p) {
    const int R = p.n_x * p.n_tok;
    if ((int) blockIdx.y < R) {
        quantize_row(p, blockIdx.y);
    } else if (blockIdx.x == 0) {
        group_pairs(p);
    }
}

__device__ __forceinline__ int dot32(const int * w, const int4 * y) {
    const int4 y0 = y[0];
    const int4 y1 = y[1];
    int c = 0;
    c = ggml_cuda_dp4a(w[0], y0.x, c);
    c = ggml_cuda_dp4a(w[1], y0.y, c);
    c = ggml_cuda_dp4a(w[2], y0.z, c);
    c = ggml_cuda_dp4a(w[3], y0.w, c);
    c = ggml_cuda_dp4a(w[4], y1.x, c);
    c = ggml_cuda_dp4a(w[5], y1.y, c);
    c = ggml_cuda_dp4a(w[6], y1.z, c);
    c = ggml_cuda_dp4a(w[7], y1.w, c);
    return c;
}

// One 32-value chunk `c` of the weights `v` (32 int8 values as 8 ints, scale and -min `dmA`) for
// every token of the pass: MMQ's two statements per chunk (mmq-vec-dot.cuh,
// ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma).
__device__ __forceinline__ void chunk(
        float * sum, const int * v, const float2 dmA, const int c, const int nt, const int * col,
        const char * qs, const half2 * ds, const int k) {
#pragma unroll
    for (int jj = 0; jj < NT; ++jj) {
        if (jj < nt) {
            const int C = dot32(v, (const int4 *) (qs + (int64_t) col[jj] * k + 32*c));
            const float2 dsB = __half22float2(ds[(int64_t) col[jj] * (k/32) + c]);
            sum[jj] += dmA.x*dsB.x*C;
            sum[jj] += dmA.y*dsB.y;
        }
    }
}

// The weights go through shared memory a warp at a time: a "piece" (one Q4_K superblock, four
// Q5_1 blocks) of each of the warp's 32 rows per stage, loaded with coalesced 16-byte cp.async
// copies (a row's piece is contiguous) one stage ahead of the compute; each lane then reads its own
// row's piece. Rows are padded in shared memory so the lanes' 16-byte reads do not conflict.
template <ggml_type type> struct fmt;
template <> struct fmt<GGML_TYPE_Q4_K> {
    static constexpr int PIECE  = sizeof(block_q4_K);  // 144 bytes, 256 values
    static constexpr int VALUES = QK_K;
    static constexpr int STRIDE = 144;
};
template <> struct fmt<GGML_TYPE_Q5_1> {
    static constexpr int PIECE  = 4 * sizeof(block_q5_1);  // 96 bytes, 128 values
    static constexpr int VALUES = 4 * QK5_1;
    static constexpr int STRIDE = 112;
};

__device__ __forceinline__ void copy16(void * dst, const void * src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned int d = (unsigned int) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(d), "l"(src));
#else
    *(int4 *) dst = __ldg((const int4 *) src);
#endif
}

__device__ __forceinline__ void copy_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void copy_wait_prev() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}

// MMQ's dp4a chain (V100) for one piece, every token of the pass (at most NTT).
template <ggml_type type, int NTT>
__device__ __forceinline__ void compute_piece_dp4a(
        const int * b, const int pc, float * sum, const int nt, const int * col,
        const char * qs, const half2 * ds, const int k) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        // The dp4a tile loader's x_dm (the raw d, dmin) and x_sc (scales of sub-blocks 0-7,
        // then their mins, one byte each); vec_dot_q4_K_q8_1_dp4a's 64-value steps.
        const half2 dm4 = *(const half2 *) &b[0];
        const int * scales = b + 1;
        const int * q      = b + 4;
        int sc[4];
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) {
            sc[ksc] = unpack_scales_q45_K(scales, ksc);
        }
        const uint8_t * sc8 = (const uint8_t *) sc;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
#pragma unroll
            for (int jj = 0; jj < NTT; ++jj) {
                if (jj < nt) {
                    const int * u = (const int *) (qs + (int64_t) col[jj] * k + 256*pc + 64*h);
                    const half2 * d8 = ds + (int64_t) col[jj] * (k/32) + 8*pc + 2*h;
                    sum[jj] += vec_dot_q4_K_q8_1_impl_mmq(q + 8*h, u, sc8 + 2*h, sc8 + 8 + 2*h, dm4, d8);
                }
            }
        }
    } else {
        // Four Q5_1 blocks; vec_dot_q8_1_q8_1_dp4a's 32-value steps.
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int * blk = b + 6*h;
            const half2 dm8 = *(const half2 *) &blk[0];
            const int qh_all = blk[1];
            int v[8];
#pragma unroll
            for (int kqsx = 0; kqsx < 4; ++kqsx) {
                // ggml_cuda_mmq_load_tiles_q5_1 (mmq-load-tiles.cuh, MIT)
                const int ql = blk[2 + kqsx];
                const int qh = qh_all >> (4 * kqsx);

                int qs0 = (ql >>  0) & 0x0F0F0F0F;
                qs0    |= (qh <<  4) & 0x00000010;
                qs0    |= (qh << 11) & 0x00001000;
                qs0    |= (qh << 18) & 0x00100000;
                qs0    |= (qh << 25) & 0x10000000;

                int qs1 = (ql >>  4) & 0x0F0F0F0F;
                qs1    |= (qh >> 12) & 0x00000010;
                qs1    |= (qh >>  5) & 0x00001000;
                qs1    |= (qh <<  2) & 0x00100000;
                qs1    |= (qh <<  9) & 0x10000000;

                v[kqsx + 0] = qs0;
                v[kqsx + 4] = qs1;
            }
#pragma unroll
            for (int jj = 0; jj < NTT; ++jj) {
                if (jj < nt) {
                    const int c = 4*pc + h;
                    const int * u = (const int *) (qs + (int64_t) col[jj] * k + 32*c);
                    const half2 ds8 = ds[(int64_t) col[jj] * (k/32) + c];
                    sum[jj] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>(v, u, dm8, ds8);
                }
            }
        }
    }
}

template <ggml_type type, bool DP4A, int NTT>
__device__ __forceinline__ void compute_piece(
        const int * b, const int pc, float * sum, const int nt, const int * col,
        const char * qs, const half2 * ds, const int k) {
    if constexpr (DP4A) {
        compute_piece_dp4a<type, NTT>(b, pc, sum, nt, col, qs, ds, k);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        // A superblock: dm, 12 scale bytes, 128 bytes of nibbles; its 8 chunks are the sub-blocks
        // in order, as MMQ's tile loader lays them out.
        const half2 dm = (*(const half2 *) &b[0]) * make_half2(1.0f, -1.0f);
        const int * scales = b + 1;
        const int * q      = b + 4;
#pragma unroll
        for (int ksc = 0; ksc < 2; ++ksc) {
            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);
            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *) &m32;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int s = 4*ksc + l;  // the sub-block
                const float2 dmA = __half22float2(dm*make_half2(sc8[l], m8[l]));
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[i] = (q[8*(s/2) + i] >> (4*(s%2))) & 0x0F0F0F0F;
                }
                chunk(sum, v, dmA, pc*8 + s, nt, col, qs, ds, k);
            }
        }
    } else {
        // Four Q5_1 blocks: dm, qh, qs[16] each.
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int * blk = b + 6*h;
            const float2 dmA = __half22float2(*(const half2 *) &blk[0]);
            const int qh_all = blk[1];
            int v[8];
#pragma unroll
            for (int kqsx = 0; kqsx < 4; ++kqsx) {
                // ggml_cuda_mmq_load_tiles_q5_1 (mmq-load-tiles.cuh, MIT)
                const int ql = blk[2 + kqsx];
                const int qh = qh_all >> (4 * kqsx);

                int qs0 = (ql >>  0) & 0x0F0F0F0F;
                qs0    |= (qh <<  4) & 0x00000010;
                qs0    |= (qh << 11) & 0x00001000;
                qs0    |= (qh << 18) & 0x00100000;
                qs0    |= (qh << 25) & 0x10000000;

                int qs1 = (ql >>  4) & 0x0F0F0F0F;
                qs1    |= (qh >> 12) & 0x00000010;
                qs1    |= (qh >>  5) & 0x00001000;
                qs1    |= (qh <<  2) & 0x00100000;
                qs1    |= (qh <<  9) & 0x10000000;

                v[kqsx + 0] = qs0;
                v[kqsx + 4] = qs1;
            }
            chunk(sum, v, dmA, 4*pc + h, nt, col, qs, ds, k);
        }
    }
}

// grid (rows / THREADS, active experts at most); rows a multiple of THREADS. NTT: the token sums a
// thread keeps (NT on the tensor-core chain, 4 on the V100's; more tokens pass over the rows
// again, each output's sum unchanged).
template <ggml_type type, bool DP4A, int NTT>
__global__ void __launch_bounds__(THREADS) mmq_ids_mm(const legion_mmq_ids p) {
    static_assert(DP4A || NTT == NT, "the tensor-core chain's chunk() sums NT tokens");
    using F = fmt<type>;
    constexpr int NL = F::PIECE / 16;  // 16-byte loads of a row's piece
    __shared__ int4 stage[THREADS/WARP_SIZE][2][WARP_SIZE*F::STRIDE/16];

    const int * plan = (const int *) (p.scratch + p.plan_off);
    const int P = p.n_used * p.n_tok;
    const int a = blockIdx.y;
    if (a >= plan[0]) {
        return;
    }
    const int * act_e   = plan + 1;
    const int * act_off = act_e + P;
    const int * pairs   = act_off + P + 1;
    const int e   = act_e[a];
    const int beg = act_off[a];
    const int end = act_off[a + 1];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int r0   = blockIdx.x * THREADS + warp * WARP_SIZE;
    const int r    = r0 + lane;
    const char  * wexp = p.w + e * p.w_nb2 + r0 * p.w_nb1;
    const char  * qs   = p.scratch;
    const half2 * ds   = (const half2 *) (p.scratch + p.ds_off);
    const int npc = p.k / F::VALUES;

    auto issue = [&](const int pc, const int buf) {
        char * dst = (char *) stage[warp][buf];
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int idx  = i * WARP_SIZE + lane;
            const int rr   = idx / NL;
            const int part = idx % NL;
            copy16(dst + rr * F::STRIDE + part * 16, wexp + rr * p.w_nb1 + pc * F::PIECE + part * 16);
        }
        copy_commit();
    };

    for (int g = beg; g < end; g += NTT) {
        const int nt = min(NTT, end - g);
        int   col[NTT];
        float sum[NTT];
#pragma unroll
        for (int jj = 0; jj < NTT; ++jj) {
            const int i = jj < nt ? pairs[g + jj] : 0;
            col[jj] = p.n_x == 1 ? i / p.n_used : i;
            sum[jj] = 0.0f;
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        issue(0, 0);
        for (int pc = 0; pc < npc; ++pc) {
            if (pc + 1 < npc) {
                issue(pc + 1, (pc + 1) & 1);
            } else {
                copy_commit();
            }
            copy_wait_prev();
            __syncwarp();
            int b[F::PIECE / 4];
            const int4 * src = (const int4 *) ((const char *) stage[warp][pc & 1] + lane * F::STRIDE);
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const int4 v = src[l];
                b[4*l + 0] = v.x;
                b[4*l + 1] = v.y;
                b[4*l + 2] = v.z;
                b[4*l + 3] = v.w;
            }
            __syncwarp();
            compute_piece<type, DP4A, NTT>(b, pc, sum, nt, col, qs, ds, p.k);
        }
#else
        // No cp.async (V100): the next piece's coalesced loads go to registers before this
        // piece's products, so they are in flight meanwhile, and land in the stage buffer after.
        int4 nxt[NL];
        auto load = [&](const int pc) {
#pragma unroll
            for (int i = 0; i < NL; ++i) {
                const int idx  = i * WARP_SIZE + lane;
                const int rr   = idx / NL;
                const int part = idx % NL;
                nxt[i] = __ldg((const int4 *) (wexp + rr * p.w_nb1 + pc * F::PIECE + part * 16));
            }
        };
        char * buf = (char *) stage[warp][0];
        load(0);
        for (int pc = 0; pc < npc; ++pc) {
            __syncwarp();
#pragma unroll
            for (int i = 0; i < NL; ++i) {
                const int idx  = i * WARP_SIZE + lane;
                *(int4 *) (buf + (idx / NL) * F::STRIDE + (idx % NL) * 16) = nxt[i];
            }
            __syncwarp();
            if (pc + 1 < npc) {
                load(pc + 1);
            }
            int b[F::PIECE / 4];
            const int4 * src = (const int4 *) (buf + lane * F::STRIDE);
#pragma unroll
            for (int l = 0; l < NL; ++l) {
                const int4 v = src[l];
                b[4*l + 0] = v.x;
                b[4*l + 1] = v.y;
                b[4*l + 2] = v.z;
                b[4*l + 3] = v.w;
            }
            compute_piece<type, DP4A, NTT>(b, pc, sum, nt, col, qs, ds, p.k);
        }
#endif

#pragma unroll
        for (int jj = 0; jj < NTT; ++jj) {
            if (jj < nt) {
                const int i = pairs[g + jj];
                float * out = (float *) ((char *) p.dst + (i / p.n_used) * p.dst_nb2 + (i % p.n_used) * p.dst_nb1);
                out[r] = sum[jj];
            }
        }
    }
}

}  // namespace

extern "C" int legion_mmq_ids_launch(int type, const legion_mmq_ids * pp, void * stream) {
    const legion_mmq_ids p = *pp;
    auto s = (cudaStream_t) stream;
    const int R = p.n_x * p.n_tok;
    const int P = p.n_used * p.n_tok;
    const dim3 grid_prep((p.k/4 + THREADS - 1) / THREADS, R + 1);
    const size_t smem = 2 * (size_t) p.n_expert * sizeof(int);
    mmq_ids_prep<<<grid_prep, THREADS, smem, s>>>(p);
    const dim3 grid_mm((p.rows + THREADS - 1) / THREADS, min(P, p.n_expert));
    // The dp4a chain (V100) keeps 4 token sums per thread: an expert rarely gets more than one
    // sequence's tokens (the token and its drafts), and the registers 16 would take cost
    // occupancy. One V100, 9-48 token steps: the reads 260-300 GB/s at 16 sums, 370-550 at 4.
    const bool dp4a = p.dp4a != 0;
    switch (type) {
        case GGML_TYPE_Q4_K:
            if (dp4a) {
                mmq_ids_mm<GGML_TYPE_Q4_K, true, 4><<<grid_mm, THREADS, 0, s>>>(p);
            } else {
                mmq_ids_mm<GGML_TYPE_Q4_K, false, NT><<<grid_mm, THREADS, 0, s>>>(p);
            }
            break;
        case GGML_TYPE_Q5_1:
            if (dp4a) {
                mmq_ids_mm<GGML_TYPE_Q5_1, true, 4><<<grid_mm, THREADS, 0, s>>>(p);
            } else {
                mmq_ids_mm<GGML_TYPE_Q5_1, false, NT><<<grid_mm, THREADS, 0, s>>>(p);
            }
            break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
