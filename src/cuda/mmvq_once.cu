// Quantized routed-expert matvecs that stream each picked expert's rows once per step (#1176):
// ggml's `mul_mat_id(w, x, ids)` for Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0 expert stacks and steps of
// 1-8 tokens (decode and MTP verify steps), as ggml-cuda's quantize_q8_1 + MMVQ pair computes it,
// bit for bit, and the gate and up matmuls of a SwiGLU with the activation in one launch. The host
// side, the shapes and the CPU reference are in src/model/core/mmvq_once.rs, which launches the
// kernels from `legion_mmvq_once_kernels` with cudaLaunchKernelExC.
//
// ggml's 1-token MUL_MAT_ID runs a CTA per (row, slot) (4 rows for short rows): on a GB10 a
// Flash-Next gate or up matmul is 6,400 CTAs of 1.4 KB each, and its 2-8 token kernel a warp per
// (token, slot) pair, each pair streaming its expert's rows again. Here a grid of one wave of
// resident CTAs splits the step's distinct experts' rows (as one run of rows, expert after
// expert) into equal contiguous ranges, and each CTA streams its range through shared memory with
// cp.async, a ring of 3 stages, while it computes the stage before: the loads of two stages are in
// flight while the CTA's warps compute, and each expert's rows are read once for all its pairs.
//
// - `quantize`: x as Q8_1 blocks, ggml's quantize_q8_1 (quantize.cu, MIT) value for value, as
//   nvfp4_once.cu's; its extra CTA (blockIdx.z == n_tok) writes the routing plan
//   (nvfp4_plan.cuh: distinct experts, their pairs) for the matvec.
// - `mv<type, MATS, NV>`: MATS 1 (one matmul) or 2 (the up and the gate matmul of the same
//   expert and rows, then ggml's silu(gate) * up). The warps of a CTA take a stage's rows in
//   turn, and per row every pair of its expert: ggml's own vec_dot_*_q8_1 (vecdotq.cuh, MIT) on
//   the staged row in ggml's thread layout and order. NV is the warps of the ggml kernel the
//   bits follow: 4 for one token (mul_mat_vec_q, 4 warps per row: lane l plays threads l,
//   32 + l, 64 + l and 96 + l, each with its own partial sum over ggml's K loop, then
//   ((t0 + t1) + t2) + t3 as ggml's warp 0 adds the other warps' sums from shared memory), 1 for
//   2-8 tokens (mul_mat_vec_q_moe, a warp per pair), then ggml's warp_reduce_sum. So each output
//   is ggml's sum of the same products in the same order.
// - The epilogue for MATS 2: unary.cuh's ggml_cuda_op_silu_single(gate) * up, the bits of
//   ggml's `silu` and `mul` nodes (or their fused unary + mul kernel).
// - On sm_90+ every kernel waits for the kernel ahead of it (griddepcontrol.wait) before it reads
//   activations, ids or the plan, and lets the next one launch early, as nvfp4_once.cu's.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include "ggml-cuda/unary.cuh"
#include "nvfp4_plan.cuh"

#include <cstdint>

extern "C" {
struct legion_mmvq_once {
    const float *   x;         // F32 [n_tok][n_x][k]
    int64_t         x_nb1;     // floats between x's rows (slots)
    int64_t         x_nb2;     // floats between x's tokens
    int32_t         n_x;       // 1 (one input per token) or n_used (one per pair)
    const char *    w;         // [n_expert][rows][row_bytes] (MATS 2: the up matmul's)
    const char *    wg;        // MATS 2: the gate matmul's, w's strides
    int64_t         w_nb2;     // bytes between experts
    int32_t         row_bytes; // bytes of a row (rows are contiguous)
    const int32_t * ids;       // I32 [n_tok][n_used]
    int64_t         ids_nb1;   // ints between tokens
    float *         dst;       // F32 [n_tok][n_used][rows]
    int64_t         dst_nb1;   // floats between slots
    int64_t         dst_nb2;   // floats between tokens
    int32_t         k;         // a multiple of the type's block
    int32_t         rows;
    int32_t         n_used;
    int32_t         n_tok;
    int32_t         n_expert;
    int32_t         rt;        // rows of a stage
    int32_t         unit;      // bytes per cp.async: 16, 8 or 4 (divides row_bytes and the bases)
};
}

namespace {

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// src/model/core/mmvq_once.rs's MAX_PAIRS, SCRATCH_BLOCKS and STAGES.
constexpr int MAX_PAIRS      = 128;
constexpr int SCRATCH_BLOCKS = 8192;
constexpr int STAGES         = 3;
static_assert(nvfp4_plan::MAX_PAIRS == MAX_PAIRS, "the plan holds every pair");

__device__ block_q8_1 scratch_q8[SCRATCH_BLOCKS];

// The step's routing plan (nvfp4_plan.cuh), written by `quantize`'s extra CTA.
__device__ nvfp4_plan::plan_t plan;

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

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n_x, n_tok + 1); row
// (t, j) at blocks (t * n_x + j) * k / 32. The last z slice's first CTA writes the plan.
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_mmvq_once p) {
    pdl_launch_dependents();
    if (blockIdx.z == p.n_tok) {
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            pdl_wait();
            nvfp4_plan::build(plan, p.ids, p.ids_nb1, p.n_used, p.n_tok, p.n_expert);
        }
        return;
    }
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= p.k) {
        return;
    }
    const int64_t j = blockIdx.y;
    const int64_t t = blockIdx.z;
    block_q8_1 * y = scratch_q8 + (t * p.n_x + j) * (p.k / QK8_1);
    const int64_t ib  = i0 / QK8_1;
    const int64_t iqs = i0 % QK8_1;

    pdl_wait();
    const float xi = p.x[t * p.x_nb2 + j * p.x_nb1 + i0];
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

// The types: ggml's vec_dot_*_q8_1 and MMVQ VDR (mmvq.cu's get_vec_dot_q_cuda, get_vdr_mmvq).
template <ggml_type type> struct qt;
template <> struct qt<GGML_TYPE_Q4_K> {
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vx, const block_q8_1 * y, const int & kbx, const int & iqs) {
        return vec_dot_q4_K_q8_1(vx, y, kbx, iqs);
    }
};
template <> struct qt<GGML_TYPE_Q5_K> {
    static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vx, const block_q8_1 * y, const int & kbx, const int & iqs) {
        return vec_dot_q5_K_q8_1(vx, y, kbx, iqs);
    }
};
template <> struct qt<GGML_TYPE_Q6_K> {
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vx, const block_q8_1 * y, const int & kbx, const int & iqs) {
        return vec_dot_q6_K_q8_1(vx, y, kbx, iqs);
    }
};
template <> struct qt<GGML_TYPE_Q5_1> {
    static constexpr int vdr = VDR_Q5_1_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vx, const block_q8_1 * y, const int & kbx, const int & iqs) {
        return vec_dot_q5_1_q8_1(vx, y, kbx, iqs);
    }
};
template <> struct qt<GGML_TYPE_Q8_0> {
    static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vx, const block_q8_1 * y, const int & kbx, const int & iqs) {
        return vec_dot_q8_0_q8_1(vx, y, kbx, iqs);
    }
};

// One row's sum against one pair's Q8_1 row y, warp-reduced (every lane holds it): ggml's
// threads v = 32 * vw + lane of a kernel of NV warps, each over kbx = v / (qi / vdr), + NV * 32
// * vdr / qi, ... at kqs = vdr * (v % (qi / vdr)) (mmvq.cu's K loop), their sums added warp
// after warp, then warp_reduce_sum. `vx` is the row's first block.
template <ggml_type type, int NV>
__device__ __forceinline__ float row_dot(const void * vx, const block_q8_1 * y, const int blocks_per_row) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = qt<type>::vdr;
    constexpr int blocks_per_iter = vdr * NV * WARP_SIZE / qi;
    const int lane = threadIdx.x;
    float t[NV];
#pragma unroll
    for (int vw = 0; vw < NV; ++vw) {
        const int v = vw * WARP_SIZE + lane;
        const int kqs = vdr * (v % (qi/vdr));
        t[vw] = 0.0f;
        for (int kbx = v / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            t[vw] += qt<type>::dot(vx, &y[kby], kbx, kqs);
        }
    }
    float r = t[0];
#pragma unroll
    for (int vw = 1; vw < NV; ++vw) {
        r += t[vw];
    }
    return warp_reduce_sum<WARP_SIZE>(r);
}

// Copies `bytes` from global `src` to shared `dst` in `unit`-byte cp.async pieces, thread `tid`
// of `nthreads` taking every nthreads-th piece.
__device__ __forceinline__ void stage_copy(char * dst, const char * src, const int bytes, const int unit,
                                           const int tid, const int nthreads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned base = (unsigned) __cvta_generic_to_shared(dst);
    if (unit == 16) {
        for (int c = 16 * tid; c < bytes; c += 16 * nthreads) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(base + c), "l"(src + c) : "memory");
        }
    } else if (unit == 8) {
        for (int c = 8 * tid; c < bytes; c += 8 * nthreads) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(base + c), "l"(src + c) : "memory");
        }
    } else {
        for (int c = 4 * tid; c < bytes; c += 4 * nthreads) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(base + c), "l"(src + c) : "memory");
        }
    }
#else
    for (int c = 4 * tid; c < bytes; c += 4 * nthreads) {
        *(int *) (dst + c) = *(const int *) (src + c);
    }
    GGML_UNUSED(unit);
#endif
}

__device__ __forceinline__ void stage_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Waits until at most one committed group of this thread is in flight.
__device__ __forceinline__ void stage_wait_but_one() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

// The stage at flattened row f (expert plan.e[f / rows], row f % rows) of a range ending at
// f_end: its expert index d, first row r and rows n (up to rt, within the expert and the range).
struct stage_t {
    int d;
    int r;
    int n;
};

__device__ __forceinline__ stage_t stage_at(const int64_t f, const int64_t f_end, const int rows, const int rt) {
    stage_t s;
    s.d = (int) (f / rows);
    s.r = (int) (f - (int64_t) s.d * rows);
    s.n = (int) min((int64_t) min(rt, rows - s.r), f_end - f);
    return s;
}

// ggml's MMVQ of every pair of the step (MATS 2: the up and gate matmuls and the SwiGLU), each
// distinct expert's rows streamed once: CTA b of the grid takes the rows [b * total / grid,
// (b + 1) * total / grid) of the distinct experts' rows one after another (total = experts *
// rows), in stages of up to rt rows of one expert, STAGES of them in a dynamic shared-memory
// ring (each MATS * rt * row_bytes).
template <ggml_type type, int MATS, int NV>
__global__ void __launch_bounds__(256) mv(const legion_mmvq_once p) {
    extern __shared__ __align__(16) char smem[];
    constexpr int qk = ggml_cuda_type_traits<type>::qk;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = threadIdx.y;
    const int n_warps = blockDim.y;

    pdl_wait();
    const int64_t total = (int64_t) plan.n * p.rows;
    const int64_t f_begin = total * blockIdx.x / gridDim.x;
    const int64_t f_end   = total * (blockIdx.x + 1) / gridDim.x;
    const int stage_bytes = p.rt * p.row_bytes;
    const int blocks_per_row = p.k / qk;

    // The next stage to load and its slot.
    int64_t f_load = f_begin;
    auto load = [&](const int slot) {
        if (f_load < f_end) {
            const stage_t s = stage_at(f_load, f_end, p.rows, p.rt);
            const int64_t off = (int64_t) plan.e[s.d] * p.w_nb2 + (int64_t) s.r * p.row_bytes;
            char * dst = smem + (int64_t) slot * MATS * stage_bytes;
            stage_copy(dst, p.w + off, s.n * p.row_bytes, p.unit, tid, nthreads);
            if constexpr (MATS == 2) {
                stage_copy(dst + stage_bytes, p.wg + off, s.n * p.row_bytes, p.unit, tid, nthreads);
            }
            f_load += s.n;
        }
        stage_commit();
    };
    load(0);
    load(1);

    int slot = 0;
    for (int64_t f = f_begin; f < f_end;) {
        stage_wait_but_one();
        __syncthreads();
        // The slot two stages on, which every warp finished computing before the sync.
        load(slot == 0 ? 2 : slot - 1);

        const stage_t s = stage_at(f, f_end, p.rows, p.rt);
        const int c   = plan.c[s.d];
        const int off = plan.off[s.d];
        const char * base = smem + (int64_t) slot * MATS * stage_bytes;
        for (int i = warp; i < s.n; i += n_warps) {
            const char * row = base + (int64_t) i * p.row_bytes;
            for (int u = 0; u < c; ++u) {
                const int q = plan.q[off + u];
                const int t = q / p.n_used;
                const int j = q - t * p.n_used;
                const block_q8_1 * y = scratch_q8 + ((int64_t) t * p.n_x + (p.n_x == 1 ? 0 : j)) * (p.k / QK8_1);
                float result = row_dot<type, NV>(row, y, blocks_per_row);
                if constexpr (MATS == 2) {
                    const float gate_value = row_dot<type, NV>(row + stage_bytes, y, blocks_per_row);
                    result = ggml_cuda_op_silu_single(gate_value) * result;
                }
                if (threadIdx.x == 0) {
                    p.dst[(int64_t) t * p.dst_nb2 + (int64_t) j * p.dst_nb1 + s.r + i] = result;
                }
            }
        }
        f += s.n;
        slot = slot == STAGES - 1 ? 0 : slot + 1;
    }
    pdl_launch_dependents();
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), in the order of
// src/model/core/mmvq_once.rs's `kernel_index`: the quantization, then per type (Q4_K, Q5_K,
// Q6_K, Q5_1, Q8_0) the matmul and the SwiGLU pair, each in ggml's 1-token order (NV 4) and its
// 2-8 token order (NV 1).
#define LEGION_MMVQ_ONCE_TYPE(T)                                       \
    reinterpret_cast<const void *>(&mv<T, 1, 4>),                      \
    reinterpret_cast<const void *>(&mv<T, 1, 1>),                      \
    reinterpret_cast<const void *>(&mv<T, 2, 4>),                      \
    reinterpret_cast<const void *>(&mv<T, 2, 1>)

extern "C" const void * const legion_mmvq_once_kernels[1 + 5 * 4] = {
    reinterpret_cast<const void *>(&quantize),
    LEGION_MMVQ_ONCE_TYPE(GGML_TYPE_Q4_K),
    LEGION_MMVQ_ONCE_TYPE(GGML_TYPE_Q5_K),
    LEGION_MMVQ_ONCE_TYPE(GGML_TYPE_Q6_K),
    LEGION_MMVQ_ONCE_TYPE(GGML_TYPE_Q5_1),
    LEGION_MMVQ_ONCE_TYPE(GGML_TYPE_Q8_0),
};
