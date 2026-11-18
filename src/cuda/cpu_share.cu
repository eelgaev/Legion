// The device side of the hybrid expert share (src/residency/cpu_share.rs): the host computes
// some of a MoE layer's host-resident experts while the GPU computes the rest. Adapted from
// Strata-AC922's POWER9 CPU expert share (eelgaev/Strata-AC922, MIT).
//
// - `publish` picks the host's experts (the rule of `select` in cpu_share.rs, computed here so
//   nothing waits for a plan), writes the step's activations and the host's expert ids into
//   host memory the CPU reads (the device's mapped exchange buffer), and outputs the ids with
//   the host's pairs replaced by -2, which the expert matvecs skip.
// - `collect` writes the host's rows (from the exchange buffer) over the down projection's
//   output for the pairs marked -2 (in place; with a separate output it copies the GPU's rows
//   too), once the host raised its "done" flag: the host pairs'
//   blocks poll it themselves (the other pairs copy meanwhile), or the stream waited for it
//   before the launch (the way before, LEGION_DEBUG_SHARE_STREAM_WAIT=1).
//
// The exact share (x86, `exact`; src/cpu/gpu_exact.rs computes the expert matvecs' bits on the
// host): `publish` writes x as the Q8_1 blocks the expert kernels' quantize_x makes of it (not
// f32) and gives the host the first of the step's distinct host experts in routing order (the
// link keeps the last ones); `activate`, between the SwiGLU and down ops, makes the host's
// pairs' SwiGLU from the raw gate and up sums the host wrote (the kernels' own `swiglu`, whose
// MUFU approximations the host cannot reproduce) and their Q8_1 blocks for the host's down rows.

#include "expert_mv/common.cuh"

#include <cstdint>
#include <cuda_runtime.h>

// Pending reads `publish` ranks last (ops.rs MAX_PENDING; more: routing order alone).
#define LEGION_CPU_SHARE_MAX_PENDING 16

extern "C" {

struct legion_cpu_share_publish {
    const int32_t * ids;  // [n_tok][n_used] tiered ids, ids_nb1 bytes per token
    int64_t ids_nb1;
    const float * x;      // [n_tok][k], x_nb1 bytes per token
    int64_t x_nb1;
    int32_t n_used, n_tok, k;
    int32_t n_slots;      // ids at or past this are host experts (id - n_slots)
    int32_t n_expert;     // host experts
    uint32_t threshold;   // the share, of 1024
    uint32_t dither;      // the layer's dither, in [0, 1024)
    int32_t * ids_host;   // [n_tok][n_used]: the host's expert, or -1 (the exchange buffer)
    float * x_host;       // [n_tok][k], dense (the exchange buffer)
    int32_t * ids_dev;    // [n_tok][n_used]: the ids, -2 for the host's pairs (the op's output)
    int32_t exact;        // non-zero: the exact share (x_host unused, the fields below)
    const uint32_t * thr; // non-null: the share of 1024 to take (threshold unused), host-written
    void * xq_host;       // [n_tok][k / 32] block_q8_1 of x (the exchange buffer)
    uint32_t * n_miss;    // the step's distinct host experts (the exchange buffer)
    uint32_t all_upto;    // the host takes every host expert of a step with at most this many
    // The host experts (id - n_slots) whose reads from the checkpoint are still under way
    // (#37 round 35): the exact share ranks them after the others, so the host takes the ones
    // already in RAM first and the link the ones being read.
    int32_t n_pending;
    int32_t pending[LEGION_CPU_SHARE_MAX_PENDING];
};

struct legion_cpu_share_activate {
    const int32_t * ids;  // [n_tok][n_used], dense: publish's output
    int32_t n_used, n_tok, rows;
    const float * src;    // the SwiGLU op's output, [n_tok][n_used][rows]
    int64_t src_nb1, src_nb2;
    const float * gu;     // the host's raw sums, [n_tok][n_used][2][rows]: gate, then up
    void * aq;            // [n_tok][n_used][rows / 32] block_q8_1 of the host pairs' SwiGLU
    float * dst;          // the op's output, dense
    float limit;          // the SwiGLU op's clamp (0: none)
    const uint32_t * wait; // non-null: the host's "gate/up done" flag, which the host pairs'
                           // blocks poll for `seq` (null: the stream waited for it)
    uint32_t seq;
};

struct legion_cpu_share_collect {
    const int32_t * ids;  // [n_tok][n_used], dense: publish's output
    int32_t n_used, n_tok, rows;
    const float * host;   // the host's rows, [n_tok][n_used][rows], dense
    const float * src;    // the down projection's output, [n_tok][n_used][rows]
    int64_t src_nb1, src_nb2;
    float * dst;          // the op's output, dense
    const uint32_t * wait; // non-null: the host's "done" flag, which the host pairs' blocks poll
                           // for `seq` (null: the stream waited for it)
    uint32_t seq;
};

int legion_cpu_share_publish_launch(const legion_cpu_share_publish * p, void * stream);
int legion_cpu_share_collect_launch(const legion_cpu_share_collect * p, void * stream);
int legion_cpu_share_activate_launch(const legion_cpu_share_activate * p, void * stream);

}  // extern "C"

namespace {

constexpr int MAX_PAIRS = 128;  // ops::MAX_TOKENS x ops::MAX_USED
constexpr int HOST_PAIR = -2;
// Loads a thread keeps in flight from host memory (#37): `collect`'s 256 threads take a
// 2560-value row in one pass, `activate`'s 128 a 640-row pair's gate and up.
constexpr int COPY_ILP = 10;
constexpr int ACT_ILP  = 5;

// Value `el` of a column `col` (k values, k a multiple of 32) as quantize_x writes it
// (src/cuda/expert_mv/common.cuh, the same operations): the warp holds one 32-value block.
__device__ __forceinline__ void quantize_q8_1_el(const float xi, block_q8_1 * col, const int el) {
    float amax    = fabsf(xi);
    float sum     = xi;
    amax          = warp_reduce_max<QK8_1>(amax);
    sum           = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1 * b    = col + el / QK8_1;
    b->qs[el % QK8_1] = q;
    if (el % QK8_1 == 0) {
        b->ds = make_half2(d, sum);
    }
}

// Block 0 picks the host's experts; blocks 1.. copy one token's activations each (exact: its
// Q8_1 blocks).
__global__ void publish(const legion_cpu_share_publish p) {
    if (blockIdx.x > 0) {
        const int t = blockIdx.x - 1;
        const float * x = (const float *) ((const char *) p.x + t * p.x_nb1);
        if (p.exact) {
            block_q8_1 * col = (block_q8_1 *) p.xq_host + (int64_t) t * (p.k / QK8_1);
            // Whole warps per pass (k is a multiple of 32): the reductions see full blocks.
            for (int i = threadIdx.x; i < p.k; i += blockDim.x) {
                quantize_q8_1_el(x[i], col, i);
            }
            return;
        }
        for (int i = threadIdx.x; i < p.k; i += blockDim.x) {
            p.x_host[(int64_t) t * p.k + i] = x[i];
        }
        return;
    }
    __shared__ int32_t ids[MAX_PAIRS];
    __shared__ uint32_t key[MAX_PAIRS];  // users << 20 | expert, for a host expert's first pair
    __shared__ int first_of[MAX_PAIRS];  // the pair's expert's first pair (host experts), or -1
    __shared__ int n_host;
    const int n = p.n_tok * p.n_used;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int t = i / p.n_used, j = i % p.n_used;
        ids[i] = *(const int32_t *) ((const char *) p.ids + t * p.ids_nb1 + j * 4);
    }
    if (threadIdx.x == 0) {
        n_host = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t id = ids[i];
        const bool host = id >= p.n_slots && id - p.n_slots < p.n_expert;
        int first = -1;
        uint32_t users = 0;
        if (host) {
            for (int o = 0; o < n; ++o) {
                if (ids[o] == id) {
                    first = first < 0 ? o : first;
                    ++users;
                }
            }
            if (first == i) {
                atomicAdd(&n_host, 1);
            }
        }
        first_of[i] = first;
        // Exact: routing order (the host takes the first experts, the link reads the last),
        // the experts still being read after the others.
        bool pending = false;
        for (int q = 0; q < p.n_pending; ++q) {
            pending = pending || p.pending[q] == id - p.n_slots;
        }
        key[i] = p.exact ? (uint32_t) i + (pending ? (uint32_t) MAX_PAIRS : 0u)
                         : users << 20 | (uint32_t) (id - p.n_slots);
    }
    __syncthreads();
    const uint32_t threshold = p.thr != nullptr ? *(const volatile uint32_t *) p.thr : p.threshold;
    const uint32_t c = (uint32_t) n_host <= p.all_upto
        ? (uint32_t) n_host
        : (min(threshold, 1024u) * (uint32_t) n_host + p.dither) >> 10;
    if (threadIdx.x == 0 && p.n_miss != nullptr) {
        *p.n_miss = (uint32_t) n_host;
    }
    // A host expert's first pair: owned when fewer than `c` host experts have a smaller key.
    __shared__ bool owned[MAX_PAIRS];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        bool own = false;
        if (first_of[i] == i) {
            uint32_t rank = 0;
            for (int o = 0; o < n; ++o) {
                rank += first_of[o] == o && key[o] < key[i];
            }
            own = rank < c;
        }
        owned[i] = own;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int f = first_of[i];
        const bool own = f >= 0 && owned[f];
        p.ids_dev[i] = own ? HOST_PAIR : ids[i];
        p.ids_host[i] = own ? ids[i] - p.n_slots : -1;
    }
}

// The host's flag at `flag` (mapped host memory) as a system-scope acquire: the host's writes
// before it raised the flag are seen by this thread's loads after it.
__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t * flag) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
    return v;
}

// A host pair's block waits until the host raised `flag` to `seq` (wrapping compare, as the
// stream wait's GEQ): thread 0 polls, then every thread acquires the flag itself, so each
// thread's reads of the host's results come after the host's writes.
__device__ __forceinline__ void wait_host(const uint32_t * flag, const uint32_t seq) {
    if (flag == nullptr) {
        return;
    }
    if (threadIdx.x == 0) {
        while ((int32_t) (load_acquire_sys(flag) - seq) < 0) {
            __nanosleep(100);
        }
    }
    __syncthreads();
    (void) load_acquire_sys(flag);
}

// grid.x: (token, slot) pairs; each copies its row from the host's results or the GPU's. With
// `wait` set, a host pair's block first waits for the host's "done", and so does block 0
// whatever its pair: the kernel, like the stream wait, ends only after the host is done with
// the exchange buffer and the layer's experts, which the next layer's `publish` and the next
// step's fetch rely on (cpu_share.rs). The other blocks copy at once.
__global__ void collect(const legion_cpu_share_collect p) {
    const int pair = blockIdx.x;
    const int t = pair / p.n_used, j = pair % p.n_used;
    const bool host = p.ids[pair] == HOST_PAIR;
    if (host || pair == 0) {
        wait_host(p.wait, p.seq);
    }
    const float * src = host ? p.host + (int64_t) pair * p.rows
                             : (const float *) ((const char *) p.src + t * p.src_nb2 + j * p.src_nb1);
    float * dst = p.dst + (int64_t) pair * p.rows;
    if (src == dst) {
        return;  // in place: the GPU pair's row is already there
    }
    // A host row comes over the link, ~1 us a load: each thread issues its COPY_ILP loads before
    // any store (one store before the next load would wait out every load in turn, src and dst
    // being plain pointers the compiler cannot tell apart). The same values are copied.
    for (int r0 = threadIdx.x; r0 < p.rows; r0 += COPY_ILP * blockDim.x) {
        float v[COPY_ILP];
#pragma unroll
        for (int i = 0; i < COPY_ILP; ++i) {
            const int r = r0 + i * blockDim.x;
            v[i] = r < p.rows ? src[r] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < COPY_ILP; ++i) {
            const int r = r0 + i * blockDim.x;
            if (r < p.rows) {
                dst[r] = v[i];
            }
        }
    }
}

// grid.x: (token, slot) pairs. A GPU pair copies the SwiGLU op's row; a host pair makes its row
// from the host's gate and up sums as the fused kernel does (swiglu of the two products) and
// writes its Q8_1 blocks for the host's down rows. rows is a multiple of 32 and the block's
// threads go over it in whole warps, so each warp reduces whole blocks. With `wait` set, a host
// pair's block first waits for the host's "gate/up done", and so does block 0 whatever its pair
// (the kernel ends after the host's gate and up, as with the stream wait).
__global__ void activate(const legion_cpu_share_activate p) {
    const int pair = blockIdx.x;
    const int t = pair / p.n_used, j = pair % p.n_used;
    float * dst = p.dst + (int64_t) pair * p.rows;
    const bool host = p.ids[pair] == HOST_PAIR;
    if (host || pair == 0) {
        wait_host(p.wait, p.seq);
    }
    if (!host) {
        const float * src = (const float *) ((const char *) p.src + t * p.src_nb2 + j * p.src_nb1);
        if (src == dst) {
            return;  // in place: the SwiGLU op's row is already there
        }
        for (int r = threadIdx.x; r < p.rows; r += blockDim.x) {
            dst[r] = src[r];
        }
        return;
    }
    const float * g = p.gu + (int64_t) pair * 2 * p.rows;
    const float * u = g + p.rows;
    block_q8_1 * col = (block_q8_1 *) p.aq + (int64_t) pair * (p.rows / QK8_1);
    // The host's sums come over the link: all of a pass's loads first (as `collect`), then each
    // value as before. `r < rows` holds for a whole warp or none of it (rows and the block are
    // whole warps), so every warp still reduces whole blocks: the same rows and blocks.
    for (int r0 = threadIdx.x; r0 < p.rows; r0 += ACT_ILP * blockDim.x) {
        float gv[ACT_ILP], uv[ACT_ILP];
#pragma unroll
        for (int i = 0; i < ACT_ILP; ++i) {
            const int r = r0 + i * blockDim.x;
            gv[i] = r < p.rows ? g[r] : 0.0f;
            uv[i] = r < p.rows ? u[r] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < ACT_ILP; ++i) {
            const int r = r0 + i * blockDim.x;
            if (r < p.rows) {
                const float v = swiglu(gv[i], uv[i], p.limit);
                dst[r] = v;
                quantize_q8_1_el(v, col, r);
            }
        }
    }
}

}  // namespace

extern "C" int legion_cpu_share_publish_launch(const legion_cpu_share_publish * p, void * stream) {
    if (p->n_tok <= 0 || p->n_used <= 0 || p->n_tok * p->n_used > MAX_PAIRS) {
        return p->n_tok * p->n_used > MAX_PAIRS ? (int) cudaErrorInvalidValue : 0;
    }
    if (p->exact && p->k % QK8_1 != 0) {
        return (int) cudaErrorInvalidValue;
    }
    publish<<<1 + p->n_tok, 128, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError();
}

extern "C" int legion_cpu_share_collect_launch(const legion_cpu_share_collect * p, void * stream) {
    if (p->n_tok <= 0 || p->n_used <= 0) {
        return 0;
    }
    collect<<<p->n_tok * p->n_used, 256, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError();
}

extern "C" int legion_cpu_share_activate_launch(const legion_cpu_share_activate * p, void * stream) {
    if (p->n_tok <= 0 || p->n_used <= 0) {
        return 0;
    }
    if (p->rows % QK8_1 != 0 || p->n_tok * p->n_used > MAX_PAIRS) {
        return (int) cudaErrorInvalidValue;
    }
    activate<<<p->n_tok * p->n_used, 128, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError();
}
