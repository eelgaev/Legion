// The device side of the hybrid expert share (src/residency/cpu_share.rs): the host computes
// some of a MoE layer's host-resident experts while the GPU computes the rest. Adapted from
// Strata-AC922's POWER9 CPU expert share (eelgaev/Strata-AC922, MIT).
//
// - `publish` picks the host's experts (the rule of `select` in cpu_share.rs, computed here so
//   nothing waits for a plan), writes the step's activations and the host's expert ids into
//   host memory the CPU reads (the device's mapped exchange buffer), and outputs the ids with
//   the host's pairs replaced by -2, which the expert matvecs skip.
// - `collect` copies the down projection's output, with the host's rows (from the exchange
//   buffer) for the pairs marked -2, after the stream waited for the host's "done" flag.

#include <cstdint>
#include <cuda_runtime.h>

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
};

struct legion_cpu_share_collect {
    const int32_t * ids;  // [n_tok][n_used], dense: publish's output
    int32_t n_used, n_tok, rows;
    const float * host;   // the host's rows, [n_tok][n_used][rows], dense
    const float * src;    // the down projection's output, [n_tok][n_used][rows]
    int64_t src_nb1, src_nb2;
    float * dst;          // the op's output, dense
};

int legion_cpu_share_publish_launch(const legion_cpu_share_publish * p, void * stream);
int legion_cpu_share_collect_launch(const legion_cpu_share_collect * p, void * stream);

}  // extern "C"

namespace {

constexpr int MAX_PAIRS = 128;  // ops::MAX_TOKENS x ops::MAX_USED
constexpr int HOST_PAIR = -2;

// Block 0 picks the host's experts; blocks 1.. copy one token's activations each.
__global__ void publish(const legion_cpu_share_publish p) {
    if (blockIdx.x > 0) {
        const int t = blockIdx.x - 1;
        const float * x = (const float *) ((const char *) p.x + t * p.x_nb1);
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
        key[i] = users << 20 | (uint32_t) (id - p.n_slots);
    }
    __syncthreads();
    const uint32_t c = (min(p.threshold, 1024u) * (uint32_t) n_host + p.dither) >> 10;
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

// grid.x: (token, slot) pairs; each copies its row from the host's results or the GPU's.
__global__ void collect(const legion_cpu_share_collect p) {
    const int pair = blockIdx.x;
    const int t = pair / p.n_used, j = pair % p.n_used;
    const float * src = p.ids[pair] == HOST_PAIR
                            ? p.host + (int64_t) pair * p.rows
                            : (const float *) ((const char *) p.src + t * p.src_nb2 + j * p.src_nb1);
    float * dst = p.dst + (int64_t) pair * p.rows;
    for (int r = threadIdx.x; r < p.rows; r += blockDim.x) {
        dst[r] = src[r];
    }
}

}  // namespace

extern "C" int legion_cpu_share_publish_launch(const legion_cpu_share_publish * p, void * stream) {
    if (p->n_tok <= 0 || p->n_used <= 0 || p->n_tok * p->n_used > MAX_PAIRS) {
        return p->n_tok * p->n_used > MAX_PAIRS ? (int) cudaErrorInvalidValue : 0;
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
