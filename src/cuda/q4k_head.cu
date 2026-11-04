// A Q4_K weight times 1-8 F32 columns with the bits of ggml-cuda's quantize_q8_1 + mul_mat_vec_q
// pair on a GB10 (#836): the MTP draft pass's LM head there (`--spec-draft-lm-head q4`, the auto
// choice for qwen4exp's NVFP4 checkpoint on an integrated GPU: 65,536 x 2560 Q4_K, 94 MB). The
// host side, the shapes and the CPU reference are in src/model/core/q4k_head.rs.
//
// ggml's mul_mat_vec_q<GGML_TYPE_Q4_K, N> (mmvq.cu, MIT) on compute capability 12.1
// (MMVQ_PARAMETERS_GB10) runs CTAs of G warps: virtual thread t (0 .. 32 G - 1) adds
// vec_dot_q4_K_q8_1 (vecdotq.cuh) of the 256-value blocks t / 16, t / 16 + 2 G, ... (ints
// 2 (t % 16) and 2 (t % 16) + 1 of each block's 32-value group pair) of each row into its own F32
// sum, from 0, in that order; lane l of warp 0 then adds warps 1 .. G - 1's sums of lane l to its
// own in that order, and warp_reduce_sum (xor 16, 8, 4, 2, 1) gives the output. G is 8 for one
// column at 2,560 values (its K loop halved: `should_halve_iters`), 4 for 2-4 columns and 2 for
// 5-8. One column of the 65,536-row draft head is then 65,536 CTAs of 256 threads, 160 of them
// with a block to read, each lane reading 4-byte pieces: ~474 us a pass, ~199 GB/s (nsys, #836).
//
// Here one warp computes a whole row: lane l does the work of virtual threads l, l + 32, ...,
// l + 32 (G - 1), each into a sum of its own, then adds them in ggml's warp order and runs ggml's
// warp_reduce_sum: every sum is the same expression (ggml's own vec_dot_q4_K_q8_1) over the same
// values in the same order, so the output has ggml's bits. The CTAs stay resident (a grid of SMs
// x CTAs per SM) and their warps walk the rows; a warp reads its next R rows with 16-byte loads
// into registers while it computes the rows before from shared memory. Each CTA first quantizes
// the input into shared memory as ggml's quantize_q8_1 does (one value per lane, a warp per
// 32-value block: the block's absolute maximum and sum by ggml's warp reductions, d = amax / 127,
// q = roundf(x / d)), so the op needs no quantize launch of its own.
//
// The draft pick (#836, as q8_head.cu's for a V100, #814; src/model/core/q8_head/pick.rs): with
// `pick` set (1-7 columns), every warp also keeps each column's best (value, tie rank) and the
// largest value of the rows it computed, and each CTA writes its warps' into `pick` (one 16-byte
// partial a CTA and column, in the column's slice of head_pick.cuh's layout), so the 65,536
// logits are not read again for the argmax. q8_head.cu's q8_head_argmax_kernel and
// q8_head_pmax_kernel (a CTA row per column) then give argmax_f32's picks and their
// probabilities with the bits of ggml's soft_max + get_rows (ggml-cuda runs the cooperative
// soft_max_f32_parallelize_cols for rows of 65,536 values on a GB10 too: a row does not fit in a
// CTA's shared memory; it loops over the rows, each summed as one row, while a row has more than
// 8,192 values per row of the tensor, so up to 7 columns of a 65,536-row head).

#include "device_cache.cuh"
#include "head_pick.cuh"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <atomic>
#include <cstdint>

extern "C" {
struct legion_q4k_head {
    const void *  w;       // Q4_K rows of k values, w_ld bytes apart
    int64_t       w_ld;    // a multiple of 16
    int64_t       rows;
    const float * x;       // F32 [n][k], x_ld floats between columns
    int64_t       x_ld;
    float *       dst;     // F32 [n][rows], dst_ld floats between columns
    int64_t       dst_ld;
    int32_t       k;       // 2560
    int32_t       n;       // 1-8
    void *        pick;         // the draft pick's buffer (head_pick.cuh), or null
    int32_t       pick_cap;     // partials a column's slice of the buffer holds
    int64_t       pick_stride;  // bytes between the buffer's column slices
};

// Enqueues dst = w . x on `stream`. 0 on success; 1 for a shape, an alignment or a device the
// kernel does not take; a CUDA error code for a failed launch.
int legion_q4k_head_launch(const legion_q4k_head * p, void * stream);

// Whether the current device runs the kernel: compute capability 12.1 in a build whose highest
// architecture for it is 12.1 (ggml's MMVQ_PARAMETERS_GB10 table, whose sums the kernel follows).
int legion_q4k_head_supported(void);

// Launches every kernel from now on with `warps` per CTA (4, 8 or 16), at most `ctas` CTAs per
// SM (0: as many as fit) and `rows` rows a warp step (1, 2 or 4); `warps` 0 restores the default.
// Same bits either way, for tuning. 0 on success, 1 for a variant the kernel does not have.
int legion_q4k_head_force(int warps, int ctas, int rows);
}

namespace {

constexpr int QB  = 144;            // bytes of a Q4_K block: dm, 12 scale bytes, 128 bytes of nibbles
constexpr int Q81 = sizeof(block_q8_1);

// Whether the next launch takes a forced variant: warps | ctas << 8 | rows << 16, 0 for the default.
std::atomic<int> forced{0};

// ggml's warps per CTA for Q4_K at `n` columns on a GB10 (calc_nwarps with
// MMVQ_PARAMETERS_GB10) for rows of `nb` blocks: the generic 4 (1-4 columns) or 2 (5-8), doubled
// for one column when that halves the K loop with at most an eighth of it idle
// (`should_halve_iters`) and the K loop is not the small-K one (`should_use_small_k`, which
// keeps the warps). The kernel only needs G: the rows a CTA takes do not change a row's sum.
__host__ __device__ constexpr int ggml_warps(int n, int nb) {
    const int generic = n <= 4 ? 4 : 2;
    if (n != 1) {
        return generic;
    }
    constexpr int per_warp = VDR_Q4_K_Q8_1_MMVQ * WARP_SIZE / QI4_K;  // blocks a warp takes a step: 2
    const bool small_k = nb < generic * per_warp;
    const int  step    = generic * per_warp;
    const int  iters   = (nb + step - 1) / step;
    const int  wide    = (nb + 2 * step - 1) / (2 * step);
    const int  idle    = 2 * wide - iters;
    return !small_k && idle * 8 <= 2 * wide ? 2 * generic : generic;
}

// NB: blocks per row, N: columns, NW: warps per CTA, R: rows a warp takes per step, PICK: also
// write the draft pick's partials (each column's into its slice of the buffer).
template <int NB, int N, int NW, int R, bool PICK>
__global__ void __launch_bounds__(NW * WARP_SIZE) head_mv(const legion_q4k_head p) {
    constexpr int G   = ggml_warps(N, NB);
    constexpr int CH  = NB * QB / 16;                        // 16-byte chunks per row
    constexpr int CPL = (CH + WARP_SIZE - 1) / WARP_SIZE;    // chunks per lane
    constexpr int YB  = NB * QK_K / QK8_1;                   // Q8_1 blocks per column
    constexpr int PN  = PICK ? N : 1;                        // columns the pick tracks
    static_assert(NB * QB % 16 == 0, "whole 16-byte chunks per row");

    extern __shared__ uint4 smem[];
    block_q8_1 * yq = (block_q8_1 *) smem;                    // [N][YB]
    constexpr int YQ = (N * YB * Q81 + 15) / 16;              // its 16-byte chunks
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    uint4 * buf = smem + YQ + warp * CH * R;
    const char * bufc = (const char *) buf;

    // The input as ggml's quantize_q8_1 writes it: a warp per 32-value block.
    for (int b = warp; b < N * YB; b += NW) {
        const int   col  = b / YB;
        const int   ib   = b % YB;
        const float xi   = p.x[col * p.x_ld + ib * QK8_1 + lane];
        float       amax = fabsf(xi);
        float       sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        yq[b].qs[lane] = q;
        if (lane == 0) {
            yq[b].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

    const int64_t gwarp  = (int64_t) blockIdx.x * NW + warp;
    const int64_t nwarps = (int64_t) gridDim.x * NW;

    uint4 r[R][CPL];
    const auto load = [&](const int64_t row0) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const uint4 * src = (const uint4 *) ((const char *) p.w + (row0 + j) * p.w_ld);
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH && row0 + j < p.rows) {
                    r[j][i] = __ldg(src + c);
                }
            }
        }
    };
    const int64_t step = nwarps * R;
    // The pick: this warp's best row of each column (-1: none above -FLT_MAX; every lane holds
    // the same sums) and largest value. The tie rank is only needed between equal values.
    float   bv[PN];
    int64_t brow[PN];
    float   mx[PN];
#pragma unroll
    for (int c = 0; c < PN; ++c) {
        bv[c]   = -FLT_MAX;
        brow[c] = -1;
        mx[c]   = -INFINITY;
    }
    if (gwarp * R < p.rows) {
        load(gwarp * R);
    }
    const int kqs = VDR_Q4_K_Q8_1_MMVQ * (lane % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
    const int kb0 = lane / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
    constexpr int per_warp  = VDR_Q4_K_Q8_1_MMVQ * WARP_SIZE / QI4_K;  // ggml's blocks a warp starts at
    constexpr int per_iter  = per_warp * G;                            // ggml's blocks_per_iter
    for (int64_t row0 = gwarp * R; row0 < p.rows; row0 += step) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH) {
                    buf[j * CH + c] = r[j][i];
                }
            }
        }
        __syncwarp();
        if (row0 + step < p.rows) {
            load(row0 + step);
        }

#pragma unroll
        for (int j = 0; j < R; ++j) {
            const int64_t row = row0 + j;
            if (row >= p.rows) {
                break;
            }
            const void * vx = bufc + j * CH * 16;
#pragma unroll
            for (int c = 0; c < N; ++c) {
                // ggml's per-thread sums: tmp[w] is virtual thread lane + 32 w's.
                float tmp[G];
#pragma unroll
                for (int w = 0; w < G; ++w) {
                    tmp[w] = 0.0f;
#pragma unroll
                    for (int kbx = per_warp * w + kb0; kbx < NB; kbx += per_iter) {
                        tmp[w] += vec_dot_q4_K_q8_1(vx, yq + c * YB + kbx * (QK_K / QK8_1), kbx, kqs);
                    }
                }
                // ggml: warp 0's lane adds warps 1 .. G - 1 in order, then the warp sum.
                float sum = tmp[0];
#pragma unroll
                for (int w = 1; w < G; ++w) {
                    sum += tmp[w];
                }
                sum = warp_reduce_sum<WARP_SIZE>(sum);
                if (lane == 0) {
                    p.dst[c * p.dst_ld + row] = sum;
                }
                if constexpr (PICK) {
                    // bv starts at -FLT_MAX: a value at or below it (or NaN) is never taken.
                    if (sum > bv[c]) {
                        bv[c]   = sum;
                        brow[c] = row;
                    } else if (sum == bv[c] && brow[c] >= 0 && pk_rank(row) < pk_rank(brow[c])) {
                        brow[c] = row;
                    }
                    mx[c] = fmaxf(mx[c], sum);
                }
            }
        }
        __syncwarp();
    }
    if constexpr (PICK) {
        // The CTA's partial of each column: its warps' best in warp order and their largest
        // value, as q8_head.cu's head_mv writes it, into the column's slice.
        __shared__ float    s_v[PN][NW];
        __shared__ uint32_t s_r[PN][NW];
        __shared__ float    s_m[PN][NW];
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < PN; ++c) {
                s_v[c][warp] = bv[c];
                s_r[c][warp] = brow[c] < 0 ? PK_NONE : pk_rank(brow[c]);
                s_m[c][warp] = mx[c];
            }
        }
        __syncthreads();
        if (threadIdx.x < PN) {
            const int c  = threadIdx.x;
            float     v  = s_v[c][0];
            uint32_t  br = s_r[c][0];
            float     m  = s_m[c][0];
            for (int w = 1; w < NW; ++w) {
                if (pk_better(s_v[c][w], s_r[c][w], v, br)) {
                    v  = s_v[c][w];
                    br = s_r[c][w];
                }
                m = fmaxf(m, s_m[c][w]);
            }
            char *  slot = (char *) p.pick + c * p.pick_stride;
            uint4 * part = (uint4 *) (slot + PICK_PARTS);
            part[blockIdx.x] = make_uint4(__float_as_uint(v), br, __float_as_uint(m), 0);
            if (blockIdx.x == 0) {
                ((pick_header *) slot)->count = gridDim.x;
            }
        }
    }
}

template <int NB, int N, int NW, int R, bool PICK>
int launch(const legion_q4k_head & p, cudaStream_t s, int ctas_cap) {
    const auto kernel = head_mv<NB, N, NW, R, PICK>;
    constexpr int YB   = NB * QK_K / QK8_1;
    const int     smem = (N * YB * Q81 + 15) / 16 * 16 + NW * R * NB * QB;
    static std::atomic<int8_t> opt_in[GGML_CUDA_MAX_DEVICES];
    static std::atomic<int>    per_sm[GGML_CUDA_MAX_DEVICES];
    if (smem > 48 * 1024 && !legion_device::smem_opt_in(opt_in, kernel, smem)) {
        return 1;
    }
    const int dev = legion_device::current();
    const int sms = legion_device::sm_count();
    if (dev < 0 || sms <= 0) {
        return 1;
    }
    int fit = per_sm[dev].load(std::memory_order_relaxed);
    if (fit == 0) {
        const cudaError_t before = cudaPeekAtLastError();
        // The pick kernel takes at most the plain kernel's CTAs an SM (as q8_head.cu's): its own
        // register count can differ and fit more, which is slower.
        int plain = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, NW * WARP_SIZE, smem) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain, head_mv<NB, N, NW, R, false>, NW * WARP_SIZE,
                                                          smem) != cudaSuccess ||
            fit <= 0) {
            legion_device::clear_own_error(before);
            return 1;
        }
        if (plain > 0 && plain < fit) {
            fit = plain;
        }
        per_sm[dev].store(fit, std::memory_order_relaxed);
    }
    if (ctas_cap > 0 && ctas_cap < fit) {
        fit = ctas_cap;
    }
    const int64_t need = (p.rows + NW * R - 1) / (NW * R);
    const int     grid = (int) (need < (int64_t) sms * fit ? need : (int64_t) sms * fit);
    if (PICK && grid > p.pick_cap) {
        return 1;
    }
    head_mv<NB, N, NW, R, PICK><<<grid, NW * WARP_SIZE, smem, s>>>(p);
    return 0;
}

template <int NB, int N, int R, bool PICK>
int launch_warps(const legion_q4k_head & p, cudaStream_t s, int warps, int ctas) {
    switch (warps) {
        case 4:
            return launch<NB, N, 4, R, PICK>(p, s, ctas);
        case 8:
            return launch<NB, N, 8, R, PICK>(p, s, ctas);
        case 16:
            return launch<NB, N, 16, R, PICK>(p, s, ctas);
        default:
            return 1;
    }
}

template <int NB, int N, bool PICK = false>
int launch_variant(const legion_q4k_head & p, cudaStream_t s, int warps, int ctas, int rows) {
    switch (rows) {
        case 1:
            return launch_warps<NB, N, 1, PICK>(p, s, warps, ctas);
        case 2:
            return launch_warps<NB, N, 2, PICK>(p, s, warps, ctas);
        case 4:
            return launch_warps<NB, N, 4, PICK>(p, s, warps, ctas);
        default:
            return 1;
    }
}

template <int NB>
int launch_cols(const legion_q4k_head & p, cudaStream_t s, int warps, int ctas, int rows) {
    if (p.pick != nullptr) {
        // The pick: 1-7 columns (the caller checks; ggml's soft_max runs a CTA a row from 8).
        switch (p.n) {
            case 1: return launch_variant<NB, 1, true>(p, s, warps, ctas, rows);
            case 2: return launch_variant<NB, 2, true>(p, s, warps, ctas, rows);
            case 3: return launch_variant<NB, 3, true>(p, s, warps, ctas, rows);
            case 4: return launch_variant<NB, 4, true>(p, s, warps, ctas, rows);
            case 5: return launch_variant<NB, 5, true>(p, s, warps, ctas, rows);
            case 6: return launch_variant<NB, 6, true>(p, s, warps, ctas, rows);
            case 7: return launch_variant<NB, 7, true>(p, s, warps, ctas, rows);
            default: return 1;
        }
    }
    switch (p.n) {
        case 1: return launch_variant<NB, 1>(p, s, warps, ctas, rows);
        case 2: return launch_variant<NB, 2>(p, s, warps, ctas, rows);
        case 3: return launch_variant<NB, 3>(p, s, warps, ctas, rows);
        case 4: return launch_variant<NB, 4>(p, s, warps, ctas, rows);
        case 5: return launch_variant<NB, 5>(p, s, warps, ctas, rows);
        case 6: return launch_variant<NB, 6>(p, s, warps, ctas, rows);
        case 7: return launch_variant<NB, 7>(p, s, warps, ctas, rows);
        case 8: return launch_variant<NB, 8>(p, s, warps, ctas, rows);
        default: return 1;
    }
}

// The default variant per column count (GB10, 65,536 and 248,320 rows of 2,560, #836): 16 warps of
// 2 rows a step for one column (-8% against ggml), 8 warps of 2 rows for 2-4 (-9 to -12%), and one
// row a step from 5 columns, where 2 or 4 rows a step run 4-5x slower than ggml (8 warps of 1 row: -4
// to -9%). Warps per CTA, CTAs per SM cap (0: all that fit), rows a warp step.
constexpr int DEFAULT_CTAS = 0;
constexpr int default_warps(int n) { return n == 1 ? 16 : 8; }
constexpr int default_rows(int n) { return n <= 4 ? 2 : 1; }

}  // namespace

int legion_q4k_head_force(int warps, int ctas, int rows) {
    if (warps == 0) {
        forced.store(0, std::memory_order_relaxed);
        return 0;
    }
    if ((warps != 4 && warps != 8 && warps != 16) || ctas < 0 || ctas > 255 || (rows != 1 && rows != 2 && rows != 4)) {
        return 1;
    }
    forced.store(warps | ctas << 8 | rows << 16, std::memory_order_relaxed);
    return 0;
}

int legion_q4k_head_supported(void) {
    return legion_device::compute_capability() == 121 &&
                   ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_DGX_SPARK) == GGML_CUDA_CC_DGX_SPARK
               ? 1
               : 0;
}

int legion_q4k_head_launch(const legion_q4k_head * p, void * stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (p->rows <= 0 || p->n < 1 || p->n > 8 || p->w_ld % 16 != 0 || p->w_ld < (int64_t) p->k / QK_K * QB ||
        ((uintptr_t) p->w) % 16 != 0 || p->x_ld < p->k || p->dst_ld < p->rows || (p->pick != nullptr && p->n > 7) ||
        legion_q4k_head_supported() == 0) {
        return 1;
    }
    const int f     = forced.load(std::memory_order_relaxed);
    const int warps = f != 0 ? (f & 0xff) : default_warps(p->n);
    const int ctas  = f != 0 ? (f >> 8) & 0xff : DEFAULT_CTAS;
    const int rows  = f != 0 ? f >> 16 : default_rows(p->n);
    int r = 1;
    switch (p->k) {
        case 2560:
            r = launch_cols<10>(*p, s, warps, ctas, rows);
            break;
        default:
            return 1;
    }
    if (r != 0) {
        return r;
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}
