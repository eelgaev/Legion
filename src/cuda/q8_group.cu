// Q8_0 matvecs that share one input, quantized once (#819): `w_i @ x` for 1-4 Q8_0 weights and one
// F32 input `x` of 1-8 columns, as ggml-cuda's quantize_q8_1 + mul_mat_vec_q pair per weight would
// compute them. The host side, the shapes and the CPU reference are in src/model/core/q8_group.rs,
// which launches the kernels from `legion_q8_group_kernels` with cudaLaunchKernel.
//
// - `quantize`: x as Q8_1 blocks, ggml's quantize_q8_1 (quantize.cu, MIT) value for value: one
//   thread per value, a warp per 32-value block, amax and sum by ggml's warp reductions,
//   d = amax / 127, q = roundf(x / d). Its own graph node, shared by every group that reads x
//   (or replaced by the blocks qwen4exp's hyper-connection mixer writes beside x).
// - `mv<N, NW, RPB, GLU>`: ggml's mul_mat_vec_q<GGML_TYPE_Q8_0, N> (mmvq.cu, MIT) for N columns,
//   with its warps (NW) and rows per CTA (RPB) for NVIDIA compute capability 7.x-8.x
//   (calc_nwarps / calc_rows_per_block, MMVQ_PARAMETERS_GENERIC and _TURING give the same for
//   Q8_0) and its K loop, vec_dot_q8_0_q8_1 (vecdotq.cuh), cross-warp sums in shared memory and
//   warp_reduce_sum: the same products added in the same order, so ggml's bits. One launch covers
//   every weight: CTA b works on the weight whose range of CTAs (`start`) holds it. With GLU (two
//   weights, gate and up) a CTA takes the same rows of both and writes up * silu(gate), as ggml's
//   fused gate/up matvec does (`has_fusion` with `use_gate`) and as its separate swiglu op does.
// - Q8OUT (with GLU, round 6): the SwiGLU's Q8_1 blocks for the next matvec (a shared expert's
//   down projection) too, so that matvec has no quantize launch. A 32-row block's values come
//   from 32 / RPB CTAs: each CTA counts itself in on the block's counter after its stores, and
//   the last one (a threadfence reduction) reads the block's 32 values of each column back and
//   quantizes them as `quantize` does (one value per lane, the same warp max and sum), into the
//   slot's blocks. SLOTIN: the matvec reads its Q8_1 blocks from the slot instead of `xq`. The
//   slots (one per layer) and their counters are device globals: zero at load, and the last CTA
//   sets its counter back to zero, so every launch finds them so.
// - GB10 (compute capability 12.1, #311): ggml's mmvq there is the same kernel at 2-8 columns
//   (MMVQ_PARAMETERS_GB10 widens only one column's CTA) plus an L2 prefetch of the weight blocks
//   two K iterations ahead (`mmvq_prefetch_l2`), which the kernels issue the same way under
//   `__CUDA_ARCH__ == 1210`; it moves no bits. On sm_90+ every kernel waits for the kernel ahead
//   of it (griddepcontrol.wait, ggml's ggml_cuda_pdl_sync) before it reads or writes activations,
//   and the quantization lets the next one launch early (ggml's ggml_cuda_pdl_lc), so the Rust
//   launcher may start them as programmatic dependent launches as ggml-cuda does there.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include "ggml-cuda/unary.cuh"

#include <cstdint>

extern "C" {
struct legion_q8_group {
    const float * x;      // F32 [n][k], x_ld floats between columns
    int64_t       x_ld;
    void *        xq;     // block_q8_1 [n][k / 32], written by `quantize`, read by `mv`
    const void *  w[4];   // Q8_0 rows of k values, w_ld blocks apart
    int64_t       w_ld[4];
    float *       dst[4]; // F32 [n][rows[i]]
    int32_t       rows[4];
    int32_t       start[5];  // first CTA of each weight; start[nw] = the grid
    int32_t       k;      // a multiple of 32
    int32_t       nw;     // weights, 1-4
    int32_t       slot;   // Q8OUT / SLOTIN: the slot of the SwiGLU's Q8_1 blocks
};
}

namespace {

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// The SwiGLU outputs' Q8_1 blocks (Q8OUT writes them, SLOTIN reads them): src/model/core/q8_group.rs's
// SLOTS, SLOT_BLOCKS (rows / 32 * columns) and SLOT_ROW_BLOCKS (rows / 32).
constexpr int SLOTS           = 64;
constexpr int SLOT_BLOCKS     = 256;
constexpr int SLOT_ROW_BLOCKS = 64;

__device__ block_q8_1   slot_q8[SLOTS][SLOT_BLOCKS];
__device__ unsigned int slot_done[SLOTS][SLOT_ROW_BLOCKS];

// ggml's programmatic dependent launch primitives (common.cuh: ggml_cuda_pdl_sync and
// ggml_cuda_pdl_lc); no-ops below sm_90 and in a plain launch.
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

// ggml's GB10 L2 prefetch of a weight block (mmvq.cu: mmvq_prefetch_l2, for Q8_0 on DGX Spark).
__device__ __forceinline__ void prefetch_l2(const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
#else
    (void) p;
#endif
}

// What a matvec reads and writes besides ggml's pair: plain, the SwiGLU's Q8_1 blocks written
// to the slot (Q8OUT, GLU only), or its input's Q8_1 blocks read from the slot (SLOTIN).
enum { IO_PLAIN = 0, IO_Q8OUT = 1, IO_SLOTIN = 2 };

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n).
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_q8_group p) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= p.k) {
        return;
    }
    const int64_t col = blockIdx.y;
    block_q8_1 * y = (block_q8_1 *) p.xq + col * (p.k / QK8_1);
    const int64_t ib  = i0 / QK8_1;
    const int64_t iqs = i0 % QK8_1;

    const float xi = p.x[col * p.x_ld + i0];
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// ggml's mul_mat_vec_q<GGML_TYPE_Q8_0, N> on the weight CTA blockIdx.x falls in; with GLU, rows
// of weight 0 (gate) and weight 1 (up) together, up * silu(gate) written to dst[0].
template <int N, int NW, int RPB, bool GLU, int IO = IO_PLAIN>
__global__ void __launch_bounds__(NW * WARP_SIZE, 1) mv(const legion_q8_group p) {
    static_assert(IO != IO_Q8OUT || GLU, "Q8OUT writes a SwiGLU's blocks");
    constexpr int qk  = QK8_0;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = WARP_SIZE;

    int wi = 0;
    if (!GLU) {
        while (wi + 1 < p.nw && (int) blockIdx.x >= p.start[wi + 1]) {
            ++wi;
        }
    }

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = RPB*((int) blockIdx.x - p.start[wi]);
    const     int blocks_per_row_x = p.k / qk;
    constexpr int blocks_per_iter = vdr * NW*warp_size / qi;

    const void       * vx = p.w[wi];
    const void       * vg = p.w[1];
    const int          stride_row_x = (int) p.w_ld[wi];
    const int          stride_row_g = (int) p.w_ld[1];
    const int          stride_col_y = p.k / QK8_1;
    const block_q8_1 * y  = IO == IO_SLOTIN ? slot_q8[p.slot] : (const block_q8_1 *) p.xq;

    // partial sum for each thread (with GLU: tmp the gate, tmp_up the up rows)
    float tmp[N][RPB] = {{0.0f}};
    float tmp_up[GLU ? N : 1][GLU ? RPB : 1] = {{0.0f}};

    const int kbx_offset = row0*stride_row_x;
    const int kbg_offset = row0*stride_row_g;

    // The weights are constant: their first prefetches (GB10) go out before the wait for the
    // kernel ahead, everything that reads or writes activations after it.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    constexpr int pf_dist = 2; // loop iterations, not blocks (ggml's)
    const auto prefetch = [&](const int kbx_pf) {
        if (kbx_pf < blocks_per_row_x) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                prefetch_l2((const char *) vx + (size_t) (kbx_offset + i*stride_row_x + kbx_pf) * sizeof(block_q8_0));
                if constexpr (GLU) {
                    prefetch_l2((const char *) vg + (size_t) (kbg_offset + i*stride_row_g + kbx_pf) * sizeof(block_q8_0));
                }
            }
        }
    };
#pragma unroll
    for (int it = 0; it < pf_dist; ++it) {
        prefetch(tid / (qi/vdr) + it*blocks_per_iter);
    }
#endif
    pdl_wait();

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // ggml's GB10 mmvq: start the next iterations' weight loads early
        prefetch(kbx + pf_dist*blocks_per_iter);
#endif

#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                tmp[j][i] += vec_dot_q8_0_q8_1(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (GLU) {
                    tmp_up[j][i] += vec_dot_q8_0_q8_1(vg, &y[j*stride_col_y + kby], kbg_offset + i*stride_row_g + kbx, kqs);
                }
            }
        }
    }

    __shared__ float tmp_shared[NW-1 > 0 ? NW-1 : 1][N][RPB][warp_size];
    __shared__ float tmp_shared_up[GLU && NW-1 > 0 ? NW-1 : 1][GLU ? N : 1][GLU ? RPB : 1][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (GLU) {
                    tmp_shared_up[threadIdx.y-1][j][i][threadIdx.x] = tmp_up[j][i];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    float * dst = p.dst[wi] + row0;
    const int stride_col_dst = p.rows[wi];

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
#pragma unroll
            for (int l = 0; l < NW-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (GLU) {
                    tmp_up[j][i] += tmp_shared_up[l][j][i][threadIdx.x];
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (GLU) {
                tmp_up[j][i] = warp_reduce_sum<warp_size>(tmp_up[j][i]);
            }

            if (threadIdx.x == i) {
                if constexpr (GLU) {
                    float result = tmp_up[j][i];
                    result *= ggml_cuda_op_silu_single(tmp[j][i]);
                    dst[j*stride_col_dst + i] = result;
                } else {
                    dst[j*stride_col_dst + i] = tmp[j][i];
                }
            }
        }
    }

    if constexpr (IO == IO_Q8OUT) {
        // This CTA's rows are stored; the last of the 32 / RPB CTAs of their 32-row block
        // quantizes the block's values of every column.
        __threadfence();
        __syncwarp();
        const int blk = row0 / QK8_1;
        unsigned int before = 0;
        if (threadIdx.x == 0) {
            before = atomicAdd(&slot_done[p.slot][blk], 1u);
        }
        before = __shfl_sync(0xffffffff, before, 0);
        if (before != QK8_1 / RPB - 1) {
            return;
        }
        __threadfence();
        const int    m  = p.rows[0];
        block_q8_1 * yq = slot_q8[p.slot];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            // ggml's quantize_q8_1 of the value, as `quantize`.
            const float xi = __ldcg(p.dst[0] + j*m + blk*QK8_1 + threadIdx.x);
            float amax = fabsf(xi);
            float sum = xi;

            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);

            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

            block_q8_1 & b = yq[j*(m / QK8_1) + blk];
            b.qs[threadIdx.x] = q;
            if (threadIdx.x == 0) {
                b.ds = make_half2(d, sum);
            }
        }
        if (threadIdx.x == 0) {
            slot_done[p.slot][blk] = 0;
        }
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel, in
// the order of src/model/core/q8_group.rs's `KERNELS`: the quantization, then the matvec of 1-8
// columns (ggml's warps and rows per CTA for each), then the same with GLU, with GLU and Q8OUT,
// and with SLOTIN.
extern "C" const void * const legion_q8_group_kernels[33] = {
    reinterpret_cast<const void *>(&quantize),
    reinterpret_cast<const void *>(&mv<1, 4, 1, false>),
    reinterpret_cast<const void *>(&mv<2, 4, 2, false>),
    reinterpret_cast<const void *>(&mv<3, 4, 2, false>),
    reinterpret_cast<const void *>(&mv<4, 4, 2, false>),
    reinterpret_cast<const void *>(&mv<5, 2, 2, false>),
    reinterpret_cast<const void *>(&mv<6, 2, 2, false>),
    reinterpret_cast<const void *>(&mv<7, 2, 2, false>),
    reinterpret_cast<const void *>(&mv<8, 2, 2, false>),
    reinterpret_cast<const void *>(&mv<1, 4, 1, true>),
    reinterpret_cast<const void *>(&mv<2, 4, 2, true>),
    reinterpret_cast<const void *>(&mv<3, 4, 2, true>),
    reinterpret_cast<const void *>(&mv<4, 4, 2, true>),
    reinterpret_cast<const void *>(&mv<5, 2, 2, true>),
    reinterpret_cast<const void *>(&mv<6, 2, 2, true>),
    reinterpret_cast<const void *>(&mv<7, 2, 2, true>),
    reinterpret_cast<const void *>(&mv<8, 2, 2, true>),
    reinterpret_cast<const void *>(&mv<1, 4, 1, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<2, 4, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<3, 4, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<4, 4, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<5, 2, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<6, 2, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<7, 2, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<8, 2, 2, true, IO_Q8OUT>),
    reinterpret_cast<const void *>(&mv<1, 4, 1, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<2, 4, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<3, 4, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<4, 4, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<5, 2, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<6, 2, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<7, 2, 2, false, IO_SLOTIN>),
    reinterpret_cast<const void *>(&mv<8, 2, 2, false, IO_SLOTIN>),
};
