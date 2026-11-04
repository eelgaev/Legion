// NVFP4 expert matvecs that read each routed expert's weights once per step (#836): ggml's
// `mul_mat_id(w, x, ids)` for NVFP4 expert stacks and steps of 2-8 tokens (an MTP verify step), as
// ggml-cuda's quantize_q8_1 + mul_mat_vec_q_moe<GGML_TYPE_NVFP4> pair computes it, bit for bit,
// optionally times each expert's scale, and the gate and up matmuls of a SwiGLU with the
// activation in one launch. The host side, the shapes and the CPU reference are in
// src/model/core/nvfp4_once.rs, which launches the kernels from `legion_nvfp4_once_kernels` with
// cudaLaunchKernelExC.
//
// ggml's MoE matvec runs a warp per (token, slot) pair: when the step's tokens pick the same
// expert (most of a verify step's rows do), each warp streams that expert's rows again. Here a
// work item per (expert, rows) tile takes every pair routed to its expert, and each warp walks its
// rows once, adding each pair's products from the same weight registers.
//
// - `quantize`: x as Q8_1 blocks, ggml's quantize_q8_1 (quantize.cu, MIT) value for value: one
//   thread per value, a warp per 32-value block, amax and sum by ggml's warp reductions,
//   d = amax / 127, q = roundf(x / d). Into a device-global scratch: the op's two kernels run
//   back to back on the compute stream (the matvec waits for the quantization), as ggml's pair
//   shares its pool buffer.
// - `mv<MATS, RPW, ITERS, USERS>`: 4 warps of RPW rows each; MATS 1 (one matmul) or 2 (the up
//   and the gate matmul of the same expert and rows, then ggml's SwiGLU of the two); up to
//   USERS pairs of an expert per pass over its rows (the launcher picks the smallest of 2, 4, 8
//   that holds the step's tokens, so a short step keeps few partial sums in registers). A CTA first finds
//   the step's distinct experts (the first pair routed to each, in shared memory), then takes
//   the (expert, tile) work items blockIdx.x, + gridDim.x, ... (a grid of one wave of resident
//   CTAs: no CTA starts only to find nothing to do). Per item every warp loads ITERS K steps of
//   its rows' weights (each lane: ITERS * RPW * MATS * 5 words) before the first dot, so the
//   loads of a whole row are in flight at once, then for each pair routed to the expert adds,
//   per row, ggml's vec_dot_nvfp4_q8_1 (vecdotq.cuh, MIT) of the same 32-lane layout (lane l
//   takes blocks l / 2, l / 2 + 16, ... and half 4 * (l % 2) of each) in the same K order, then
//   ggml's warp_reduce_sum: per output the same products added in the same order as ggml's
//   warp, so ggml's bits for any RPW and ITERS. The weights of a row are decoded (E2M1 table,
//   UE4M3 scales) once and used for every pair.
// - `mvw<MATS, USERS, PRE, OCC>`: the same sums with warps that work alone: each warp takes
//   (distinct expert, 2 rows) items from a counter of the routing plan `quantize` writes, so a
//   step of spread routing (most experts with one pair) runs about as many warps per SM as
//   ggml's MMVQ, with no CTA prologue and no tail of whole items, while a shared expert's rows
//   are still read once for all its pairs.
// - `mvh<MATS, UMAX, OCC>`: `mvw`'s items, each picking its pass on the device from its
//   expert's pairs in the plan (the plan lists the experts with most pairs first): a singleton
//   on the one-pair body, a shared expert on 2, 4 or 8 pairs a pass (up to UMAX), in one launch.
// - The epilogue: times the expert's scale when there is one (ggml's `mul` after mul_mat_id,
//   one rounding, as ggml-cuda's MMVQ fusion does it), and for MATS 2 ggml's
//   silu(gate) * up (unary.cuh's ggml_cuda_op_silu_single, then the product), the bits of
//   ggml's `silu` and `mul` nodes.
// - On sm_90+ every kernel waits for the kernel ahead of it (griddepcontrol.wait, ggml's
//   ggml_cuda_pdl_sync) before it reads activations or ids, and lets the next one launch early
//   (ggml_cuda_pdl_lc), so the Rust launcher starts them as programmatic dependent launches on a
//   GB10 as ggml-cuda does there.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include "ggml-cuda/unary.cuh"
#include "nvfp4_plan.cuh"

#include <cstdint>

extern "C" {
struct legion_nvfp4_once {
    const float *   x;        // F32 [n_tok][n_x][k]
    int64_t         x_nb1;    // floats between x's rows (slots)
    int64_t         x_nb2;    // floats between x's tokens
    int32_t         n_x;      // 1 (one input per token) or n_used (one per pair)
    const void *    w;        // NVFP4 [n_expert][rows][k / 64] (MATS 2: the up matmul's)
    int64_t         w_nb1;    // blocks between rows
    int64_t         w_nb2;    // blocks between experts
    const int32_t * ids;      // I32 [n_tok][n_used]
    int64_t         ids_nb1;  // ints between tokens
    float *         dst;      // F32 [n_tok][n_used][rows]
    int64_t         dst_nb1;  // floats between slots
    int64_t         dst_nb2;  // floats between tokens
    int32_t         k;        // a multiple of 64
    int32_t         rows;     // a multiple of 4 * the variant's RPW
    int32_t         n_used;
    int32_t         n_tok;
    int32_t         n_expert;
    const void *    wg;       // MATS 2: the gate matmul's weights, w's strides
    const float *   s_w;      // F32 [n_expert]: w's per-expert scale, or null
    const float *   s_g;      // F32 [n_expert]: wg's per-expert scale, or null
};

// The device address of this device's routing plan (nvfp4_plan.cuh), for the router top-k
// kernel to write it: 0 or a CUDA error.
int legion_nvfp4_once_plan(void ** ptr);
}

namespace {

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// src/model/core/nvfp4_once.rs's MAX_PAIRS and SCRATCH_BLOCKS.
constexpr int MAX_PAIRS      = 128;
constexpr int SCRATCH_BLOCKS = 8192;

// Warps of a matvec CTA: one thread per pair of the step in the prologue.
constexpr int NW = 4;
static_assert(NW * WARP_SIZE == MAX_PAIRS, "the prologue takes a pair per thread");
static_assert(nvfp4_plan::MAX_PAIRS == MAX_PAIRS, "the plan holds every pair");

__device__ block_q8_1 scratch_q8[SCRATCH_BLOCKS];

// The step's routing plan for `mvw` and `mvh` (nvfp4_plan.cuh): written by `quantize`'s extra
// CTA, or by the router top-k kernel of the step (fused.cu, `legion_nvfp4_once_plan` gives it the
// address), in which case the launcher leaves the extra CTA out.
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

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n_x, n_tok); row (t, j)
// at blocks (t * n_x + j) * k / 32. One more CTA (blockIdx.z == n_tok, when the launcher's
// grid has n_tok + 1 in z: no router wrote the plan) writes the routing plan alongside.
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_nvfp4_once p) {
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

// ggml's mul_mat_vec_q_moe<GGML_TYPE_NVFP4> (with its scale and SwiGLU fusions for MATS 2) for
// every pair of the step, each routed expert's rows read once. The grid: work items
// (distinct expert, tile of NW * RPW rows) blockIdx.x, + gridDim.x, ...
// CTAs per SM a variant asks the compiler for (registers per thread at most 65536 / 128 / it):
// the matvecs are bound by the weight loads in flight, so small variants take more CTAs.
constexpr int min_ctas(int words) {
    return words <= 4 ? 8 : words <= 8 ? 6 : words <= 16 ? 4 : 2;
}

template <int MATS, int RPW, int ITERS, int USERS>
__global__ void __launch_bounds__(NW * WARP_SIZE, min_ctas(MATS * RPW * ITERS * USERS)) mv(const legion_nvfp4_once p) {
    constexpr int qk  = QK_NVFP4;
    constexpr int qi  = QI_NVFP4;
    constexpr int vdr = VDR_NVFP4_Q8_1_MMVQ;
    constexpr int warp_size = WARP_SIZE;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int chunks = MAX_PAIRS / warp_size;
    static_assert(vdr == 4 && qi == 8, "the loads below are ggml's NVFP4 MMVQ layout");

    __shared__ int s_id[MAX_PAIRS];    // each pair's expert (-1: no pair or out of range)
    __shared__ int s_lead[MAX_PAIRS];  // the distinct experts' first pairs, in pair order
    __shared__ int s_cnt[NW];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * warp_size + lane;
    const int pairs = p.n_used * p.n_tok;

    // The step's distinct experts: pair tid leads its expert if no pair before it has it.
    pdl_wait();
    int e_t = -1;
    if (tid < pairs) {
        e_t = p.ids[(tid / p.n_used) * p.ids_nb1 + tid % p.n_used];
        if (e_t < 0 || e_t >= p.n_expert) {
            e_t = -1;
        }
    }
    s_id[tid] = e_t;
    __syncthreads();
    bool lead = e_t >= 0;
    for (int j = 0; lead && j < tid; ++j) {
        lead = s_id[j] != e_t;
    }
    const unsigned lead_mask = __ballot_sync(0xffffffff, lead);
    if (lane == 0) {
        s_cnt[warp] = __popc(lead_mask);
    }
    __syncthreads();
    int base = 0;
    int n_dist = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        base   += w < warp ? s_cnt[w] : 0;
        n_dist += s_cnt[w];
    }
    if (lead) {
        s_lead[base + __popc(lead_mask & ((1u << lane) - 1))] = tid;
    }
    __syncthreads();

    const int blocks_per_row_x = p.k / qk;
    const int stride_row_x     = (int) p.w_nb1;
    const int n_tiles = p.rows / (NW * RPW);
    const int n_work  = n_dist * n_tiles;
    // This lane's block within a K step and half of it (ggml's kbx and kqs).
    const int kb_lane = lane / (qi/vdr);
    const int kqs     = vdr * (lane % (qi/vdr));

    for (int item = blockIdx.x; item < n_work; item += gridDim.x) {
        const int d    = item / n_tiles;
        const int tile = item - d * n_tiles;
        const int e    = s_id[s_lead[d]];
        const int row0 = (tile * NW + warp) * RPW;

        const block_nvfp4 * vx[MATS];
        vx[0] = (const block_nvfp4 *) p.w + (int64_t) e * p.w_nb2 + (int64_t) row0 * stride_row_x;
        if constexpr (MATS == 2) {
            vx[1] = (const block_nvfp4 *) p.wg + (int64_t) e * p.w_nb2 + (int64_t) row0 * stride_row_x;
        }

        // The pairs routed to e: `um[c]` bit l is pair 32 c + l.
        unsigned um[chunks];
#pragma unroll
        for (int c = 0; c < chunks; ++c) {
            um[c] = __ballot_sync(0xffffffff, s_id[c * warp_size + lane] == e);
        }

        while (true) {
            // The next (up to) USERS pairs of e, lowest first.
            int  qs[USERS];
            bool has[USERS];
#pragma unroll
            for (int u = 0; u < USERS; ++u) {
                has[u] = false;
                qs[u]  = 0;
#pragma unroll
                for (int c = 0; c < chunks; ++c) {
                    if (!has[u] && um[c] != 0) {
                        qs[u]  = c * warp_size + __ffs(um[c]) - 1;
                        um[c] &= um[c] - 1;
                        has[u] = true;
                    }
                }
            }
            if (!has[0]) {
                break;
            }
            const block_q8_1 * y[USERS];
#pragma unroll
            for (int u = 0; u < USERS; ++u) {
                const int t = qs[u] / p.n_used;
                const int j = p.n_x == 1 ? 0 : qs[u] % p.n_used;
                y[u] = scratch_q8 + ((int64_t) t * p.n_x + j) * (p.k / QK8_1);
            }

            // partial sums of each pair's rows (ggml's tmp and tmp_gate of each pair's warp)
            float tmp[MATS][RPW][USERS];
#pragma unroll
            for (int m = 0; m < MATS; ++m) {
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
#pragma unroll
                    for (int u = 0; u < USERS; ++u) {
                        tmp[m][i][u] = 0.0f;
                    }
                }
            }

            for (int kb0 = 0; kb0 < blocks_per_row_x; kb0 += ITERS * blocks_per_iter) {
                // Every weight word of ITERS K steps first: the qs ints iqs kqs .. kqs + 3 and
                // the four UE4M3 scales of each block.
                int      wq[ITERS][MATS][RPW][4];
                uint32_t wd[ITERS][MATS][RPW];
#pragma unroll
                for (int it = 0; it < ITERS; ++it) {
                    const int kbx = kb0 + it * blocks_per_iter + kb_lane;
#pragma unroll
                    for (int m = 0; m < MATS; ++m) {
#pragma unroll
                        for (int i = 0; i < RPW; ++i) {
                            if (kbx < blocks_per_row_x) {
                                const block_nvfp4 * b = vx[m] + i * stride_row_x + kbx;
                                wd[it][m][i] = *(const uint32_t *) b->d;
#pragma unroll
                                for (int l = 0; l < 4; ++l) {
                                    wq[it][m][i][l] = get_int_b4(b->qs, kqs + l);
                                }
                            } else {
                                wd[it][m][i] = 0;
#pragma unroll
                                for (int l = 0; l < 4; ++l) {
                                    wq[it][m][i][l] = 0;
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int it = 0; it < ITERS; ++it) {
                    const int kbx = kb0 + it * blocks_per_iter + kb_lane;
                    if (kbx >= blocks_per_row_x) {
                        continue;
                    }
                    const int kby = kbx * (qk/QK8_1);
                    // This K step's weights decoded once (E2M1 values, UE4M3 scales):
                    // vec_dot_nvfp4_q8_1's sub-step s takes iqs kqs + 2 s and kqs + 2 s + 1,
                    // scale is = kqs / 2 + s.
                    int2  v[MATS][RPW][2][2];
                    float dw[MATS][RPW][2];
#pragma unroll
                    for (int m = 0; m < MATS; ++m) {
#pragma unroll
                        for (int i = 0; i < RPW; ++i) {
#pragma unroll
                            for (int s = 0; s < 2; ++s) {
                                v[m][i][s][0] = get_int_from_table_16(wq[it][m][i][2*s + 0], kvalues_mxfp4);
                                v[m][i][s][1] = get_int_from_table_16(wq[it][m][i][2*s + 1], kvalues_mxfp4);
                                const int is = (kqs >> 1) + s;
                                dw[m][i][s] = ggml_cuda_ue4m3_to_fp32((uint8_t) (wd[it][m][i] >> (8 * is)));
                            }
                        }
                    }
                    // Then pair by pair: its Q8_1 block of this lane (vec_dot_nvfp4_q8_1's
                    // bq8_1 + (is >> 1), the same block for both sub-steps, Q8_1 ints
                    // 4 s .. 4 s + 3) against every row.
#pragma unroll
                    for (int u = 0; u < USERS; ++u) {
                        if (!has[u]) {
                            continue;
                        }
                        const block_q8_1 * bq8 = y[u] + kby + (kqs >> 2);
                        const float yd = __low2float(bq8->ds);
                        int yq[8];
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            yq[l] = get_int_b4(bq8->qs, l);
                        }
#pragma unroll
                        for (int m = 0; m < MATS; ++m) {
#pragma unroll
                            for (int i = 0; i < RPW; ++i) {
                                float sum = 0.0f;
#pragma unroll
                                for (int s = 0; s < 2; ++s) {
                                    const int i8 = 4 * s;
                                    int sumi = ggml_cuda_dp4a(v[m][i][s][0].x, yq[i8 + 0], 0);
                                    sumi = ggml_cuda_dp4a(v[m][i][s][0].y, yq[i8 + 2], sumi);
                                    sumi = ggml_cuda_dp4a(v[m][i][s][1].x, yq[i8 + 1], sumi);
                                    sumi = ggml_cuda_dp4a(v[m][i][s][1].y, yq[i8 + 3], sumi);

                                    const float dd = dw[m][i][s] * yd;
                                    sum += dd * float(sumi);
                                }
                                tmp[m][i][u] += sum;
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int u = 0; u < USERS; ++u) {
                if (!has[u]) {
                    continue;
                }
#pragma unroll
                for (int m = 0; m < MATS; ++m) {
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        tmp[m][i][u] = warp_reduce_sum<warp_size>(tmp[m][i][u]);
                    }
                }
                const int q = qs[u];
                float * dst = p.dst + (int64_t) (q / p.n_used) * p.dst_nb2 + (int64_t) (q % p.n_used) * p.dst_nb1 + row0;
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    if (lane != i) {
                        continue;
                    }
                    float result = tmp[0][i][u];
                    if (p.s_w) {
                        result *= p.s_w[e];
                    }
                    if constexpr (MATS == 2) {
                        float gate_value = tmp[1][i][u];
                        if (p.s_g) {
                            gate_value *= p.s_g[e];
                        }
                        result = ggml_cuda_op_silu_single(gate_value) * result;
                    }
                    dst[i] = result;
                }
            }
        }
    }
    pdl_launch_dependents();
}

// The warp-item matvec: as `mv`, every pair of an expert from the same weight loads with
// ggml's per-pair sums, but each warp takes its own work items (distinct expert d, RPW_W rows)
// from the plan's counter one at a time, so no warp waits for the others of its CTA or for a
// CTA-wide prologue, a warp with work left never idles at the end of a launch (no tail of
// whole items), and an expert of one pair costs what it costs ggml's warp of that pair, at
// about ggml's registers (OCC CTAs of 4 warps an SM: 8 at <= 64 registers, 7 at <= 72).
// The pairs of an expert go USERS per pass over its rows (more passes read the rows again,
// mostly from L1 or L2). PRE: a K step's weights decoded once for all pairs of the pass (more
// registers), else per pair (more ALU, fewer registers).
constexpr int RPW_W = 2;  // ggml's rows per block of mul_mat_vec_q_moe

// One work item of the warp kernels: expert e's c pairs (plan.q[off ..]) on rows row0 ..
// row0 + RPW_W - 1, USERS pairs per pass.
template <int MATS, int USERS, bool PRE>
__device__ __forceinline__ void warp_item(
        const legion_nvfp4_once & p, const int e, const int c, const int off, const int row0, const int lane) {
    constexpr int qk  = QK_NVFP4;
    constexpr int qi  = QI_NVFP4;
    constexpr int vdr = VDR_NVFP4_Q8_1_MMVQ;
    constexpr int warp_size = WARP_SIZE;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int RPW = RPW_W;
    static_assert(vdr == 4 && qi == 8, "the loads below are ggml's NVFP4 MMVQ layout");

    const int blocks_per_row_x = p.k / qk;
    const int stride_row_x     = (int) p.w_nb1;
    const int kb_lane = lane / (qi/vdr);
    const int kqs     = vdr * (lane % (qi/vdr));
    {
        const block_nvfp4 * vx[MATS];
        vx[0] = (const block_nvfp4 *) p.w + (int64_t) e * p.w_nb2 + (int64_t) row0 * stride_row_x;
        if constexpr (MATS == 2) {
            vx[1] = (const block_nvfp4 *) p.wg + (int64_t) e * p.w_nb2 + (int64_t) row0 * stride_row_x;
        }

        for (int u0 = 0; u0 < c; u0 += USERS) {
            int  qs[USERS];
            bool has[USERS];
            const block_q8_1 * y[USERS];
#pragma unroll
            for (int u = 0; u < USERS; ++u) {
                has[u] = u0 + u < c;
                qs[u]  = has[u] ? plan.q[off + u0 + u] : 0;
                const int t = qs[u] / p.n_used;
                const int j = p.n_x == 1 ? 0 : qs[u] % p.n_used;
                y[u] = scratch_q8 + ((int64_t) t * p.n_x + j) * (p.k / QK8_1);
            }

            float tmp[MATS][RPW][USERS];
#pragma unroll
            for (int m = 0; m < MATS; ++m) {
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
#pragma unroll
                    for (int u = 0; u < USERS; ++u) {
                        tmp[m][i][u] = 0.0f;
                    }
                }
            }

            for (int kbx = kb_lane; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                const int kby = kbx * (qk/QK8_1);
                // This K step's weight words of every row first (all loads in flight).
                int      wq[MATS][RPW][4];
                uint32_t wd[MATS][RPW];
#pragma unroll
                for (int m = 0; m < MATS; ++m) {
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        const block_nvfp4 * b = vx[m] + i * stride_row_x + kbx;
                        wd[m][i] = *(const uint32_t *) b->d;
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            wq[m][i][l] = get_int_b4(b->qs, kqs + l);
                        }
                    }
                }
                // vec_dot_nvfp4_q8_1 of one row and one pair, from the row's decoded weights
                // (sub-step s: iqs kqs + 2 s and kqs + 2 s + 1, scale kqs / 2 + s) and the
                // pair's Q8_1 block of this lane (ints 4 s .. 4 s + 3), as in `mv`.
                const auto dot = [&](const int2 (&v)[2][2], const float (&dw)[2], const int u) {
                    const block_q8_1 * bq8 = y[u] + kby + (kqs >> 2);
                    const float yd = __low2float(bq8->ds);
                    float sum = 0.0f;
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const int i8 = 4 * s;
                        int sumi = ggml_cuda_dp4a(v[s][0].x, get_int_b4(bq8->qs, i8 + 0), 0);
                        sumi = ggml_cuda_dp4a(v[s][0].y, get_int_b4(bq8->qs, i8 + 2), sumi);
                        sumi = ggml_cuda_dp4a(v[s][1].x, get_int_b4(bq8->qs, i8 + 1), sumi);
                        sumi = ggml_cuda_dp4a(v[s][1].y, get_int_b4(bq8->qs, i8 + 3), sumi);

                        const float dd = dw[s] * yd;
                        sum += dd * float(sumi);
                    }
                    return sum;
                };
                const auto decode = [&](const int m, const int i, int2 (&v)[2][2], float (&dw)[2]) {
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        v[s][0] = get_int_from_table_16(wq[m][i][2*s + 0], kvalues_mxfp4);
                        v[s][1] = get_int_from_table_16(wq[m][i][2*s + 1], kvalues_mxfp4);
                        const int is = (kqs >> 1) + s;
                        dw[s] = ggml_cuda_ue4m3_to_fp32((uint8_t) (wd[m][i] >> (8 * is)));
                    }
                };
                if constexpr (PRE) {
                    int2  v[MATS][RPW][2][2];
                    float dw[MATS][RPW][2];
#pragma unroll
                    for (int m = 0; m < MATS; ++m) {
#pragma unroll
                        for (int i = 0; i < RPW; ++i) {
                            decode(m, i, v[m][i], dw[m][i]);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < USERS; ++u) {
                        if (!has[u]) {
                            continue;
                        }
#pragma unroll
                        for (int m = 0; m < MATS; ++m) {
#pragma unroll
                            for (int i = 0; i < RPW; ++i) {
                                tmp[m][i][u] += dot(v[m][i], dw[m][i], u);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int m = 0; m < MATS; ++m) {
#pragma unroll
                        for (int i = 0; i < RPW; ++i) {
                            int2  v[2][2];
                            float dw[2];
                            decode(m, i, v, dw);
#pragma unroll
                            for (int u = 0; u < USERS; ++u) {
                                if (has[u]) {
                                    tmp[m][i][u] += dot(v, dw, u);
                                }
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int u = 0; u < USERS; ++u) {
                if (!has[u]) {
                    continue;
                }
#pragma unroll
                for (int m = 0; m < MATS; ++m) {
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        tmp[m][i][u] = warp_reduce_sum<warp_size>(tmp[m][i][u]);
                    }
                }
                const int q = qs[u];
                float * dst = p.dst + (int64_t) (q / p.n_used) * p.dst_nb2 + (int64_t) (q % p.n_used) * p.dst_nb1 + row0;
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    if (lane != i) {
                        continue;
                    }
                    float result = tmp[0][i][u];
                    if (p.s_w) {
                        result *= p.s_w[e];
                    }
                    if constexpr (MATS == 2) {
                        float gate_value = tmp[1][i][u];
                        if (p.s_g) {
                            gate_value *= p.s_g[e];
                        }
                        result = ggml_cuda_op_silu_single(gate_value) * result;
                    }
                    dst[i] = result;
                }
            }
        }
    }
}

// The warp kernels' loop: each warp takes (distinct expert d, RPW_W rows) items, the first by
// its number in the grid, the rest from the plan's counter of the op (next[MATS - 1], so the
// SwiGLU and the down op of one router plan count on their own) after those, and runs `run(e,
// c, off, row0)` on each. (Every n_warps-th item from the first, without the counter, was 2-4%
// slower on a GB10: the counter balances the warps.)
template <int MATS, typename Run>
__device__ __forceinline__ void warp_items(const legion_nvfp4_once & p, const Run & run) {
    const int lane = threadIdx.x;
    pdl_wait();

    const int n_tiles = p.rows / RPW_W;
    const int n_items = plan.n * n_tiles;
    const int n_warps = gridDim.x * NW;
    int item = blockIdx.x * NW + threadIdx.y;
    while (item < n_items) {
        // The next item's number, in flight while this one runs.
        int next = 0;
        if (lane == 0) {
            next = n_warps + (int) atomicAdd(&plan.next[MATS - 1], 1u);
        }
        const int d    = item / n_tiles;
        const int tile = item - d * n_tiles;
        run(plan.e[d], plan.c[d], plan.off[d], tile * RPW_W, lane);
        item = __shfl_sync(0xffffffff, next, 0);
    }
    pdl_launch_dependents();
}

template <int MATS, int USERS, bool PRE, int OCC>
__global__ void __launch_bounds__(NW * WARP_SIZE, OCC) mvw(const legion_nvfp4_once p) {
    warp_items<MATS>(p, [&](const int e, const int c, const int off, const int row0, const int lane) {
        warp_item<MATS, USERS, PRE>(p, e, c, off, row0, lane);
    });
}

// The hybrid warp kernel: as `mvw`, but each item picks its pass width on the device from its
// expert's pairs in the plan: a singleton expert (most of them when the routing is spread) on
// the one-pair body (ggml's warp of that pair, its weights decoded once), a shared expert on
// 2, 4 or (UMAX 8) 8 pairs a pass, so it reads its rows once for up to UMAX pairs. One launch
// takes both kinds of experts at the register budget of OCC CTAs of 4 warps an SM.
template <int MATS, int UMAX, int OCC>
__global__ void __launch_bounds__(NW * WARP_SIZE, OCC) mvh(const legion_nvfp4_once p) {
    static_assert(UMAX == 2 || UMAX == 4 || UMAX == 8, "2, 4 or 8 pairs a pass at most");
    warp_items<MATS>(p,[&](const int e, const int c, const int off, const int row0, const int lane) {
        if (c == 1) {
            warp_item<MATS, 1, true>(p, e, c, off, row0, lane);
        } else if (UMAX == 2 || c == 2) {
            warp_item<MATS, 2, false>(p, e, c, off, row0, lane);
        } else if constexpr (UMAX >= 4) {
            if (UMAX == 4 || c <= 4) {
                warp_item<MATS, 4, false>(p, e, c, off, row0, lane);
            } else if constexpr (UMAX == 8) {
                warp_item<MATS, 8, false>(p, e, c, off, row0, lane);
            }
        }
    });
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), in the order of
// src/model/core/nvfp4_once.rs's `kernel_index`: the quantization, then for each of its `TILES`
// (rows per warp) the matvec of one matmul and of the SwiGLU pair, each with 1 and 3 K steps
// loaded at once, each for 2, 4 and 8 pairs per pass.
#define LEGION_NVFP4_ONCE_USERS(MATS, RPW, ITERS)                      \
    reinterpret_cast<const void *>(&mv<MATS, RPW, ITERS, 2>),          \
    reinterpret_cast<const void *>(&mv<MATS, RPW, ITERS, 4>),          \
    reinterpret_cast<const void *>(&mv<MATS, RPW, ITERS, 8>)
#define LEGION_NVFP4_ONCE_TILE(RPW)                                    \
    LEGION_NVFP4_ONCE_USERS(1, RPW, 1),                                \
    LEGION_NVFP4_ONCE_USERS(1, RPW, 3),                                \
    LEGION_NVFP4_ONCE_USERS(2, RPW, 1),                                \
    LEGION_NVFP4_ONCE_USERS(2, RPW, 3)

// Then the warp-item matvecs (`mvw`): for the matmul and the SwiGLU pair, for 1, 2, 4 and 8
// pairs per pass, weights decoded per pair and once, at 8 and 7 CTAs an SM.
#define LEGION_NVFP4_ONCE_WARP_OCC(MATS, USERS, PRE)                   \
    reinterpret_cast<const void *>(&mvw<MATS, USERS, PRE, 8>),         \
    reinterpret_cast<const void *>(&mvw<MATS, USERS, PRE, 7>)
#define LEGION_NVFP4_ONCE_WARP_USERS(MATS, USERS)                      \
    LEGION_NVFP4_ONCE_WARP_OCC(MATS, USERS, false),                    \
    LEGION_NVFP4_ONCE_WARP_OCC(MATS, USERS, true)
#define LEGION_NVFP4_ONCE_WARP(MATS)                                   \
    LEGION_NVFP4_ONCE_WARP_USERS(MATS, 1),                             \
    LEGION_NVFP4_ONCE_WARP_USERS(MATS, 2),                             \
    LEGION_NVFP4_ONCE_WARP_USERS(MATS, 4),                             \
    LEGION_NVFP4_ONCE_WARP_USERS(MATS, 8)

// Then the hybrid warp kernels (`mvh`): for the matmul and the SwiGLU pair, at most 2, 4 and 8
// pairs a pass, at 8, 6 and 4 CTAs an SM.
#define LEGION_NVFP4_ONCE_HYBRID_OCC(MATS, UMAX)                       \
    reinterpret_cast<const void *>(&mvh<MATS, UMAX, 8>),               \
    reinterpret_cast<const void *>(&mvh<MATS, UMAX, 6>),               \
    reinterpret_cast<const void *>(&mvh<MATS, UMAX, 4>)
#define LEGION_NVFP4_ONCE_HYBRID(MATS)                                 \
    LEGION_NVFP4_ONCE_HYBRID_OCC(MATS, 2),                             \
    LEGION_NVFP4_ONCE_HYBRID_OCC(MATS, 4),                             \
    LEGION_NVFP4_ONCE_HYBRID_OCC(MATS, 8)

extern "C" const void * const legion_nvfp4_once_kernels[1 + 4 * 4 * 3 + 2 * 4 * 2 * 2 + 2 * 3 * 3] = {
    reinterpret_cast<const void *>(&quantize),
    LEGION_NVFP4_ONCE_TILE(1),
    LEGION_NVFP4_ONCE_TILE(2),
    LEGION_NVFP4_ONCE_TILE(4),
    LEGION_NVFP4_ONCE_TILE(8),
    LEGION_NVFP4_ONCE_WARP(1),
    LEGION_NVFP4_ONCE_WARP(2),
    LEGION_NVFP4_ONCE_HYBRID(1),
    LEGION_NVFP4_ONCE_HYBRID(2),
};

int legion_nvfp4_once_plan(void ** ptr) {
    return (int) cudaGetSymbolAddress(ptr, plan);
}
