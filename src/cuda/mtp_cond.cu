// The device-side stop check of an MTP draft round (#827). After a pass, one thread reads the
// probabilities of its picks and keeps the round's flag up only if every row reached the
// confidence floor (mtp_decide_kernel); the pass queued after it is the body of a conditional
// graph node whose condition another one-thread kernel sets from that flag (mtp_cond_kernel),
// so it runs only if the round went on. The Rust side (src/ggml/stage_graph/cond.rs) launches
// the first and builds the graph around the second; it owns the memory.

#include <cuda_runtime.h>

extern "C" {

// Must match src/ggml/stage_graph/cond.rs (`DecideArgs`).
struct legion_mtp_decide {
    const float * pmax;    // the pass's pick probabilities, `rows` of them
    unsigned int * alive;  // nonzero while the round goes on
    float p_min;           // the confidence floor
    unsigned int rows;
    unsigned int reset;    // the pass ran unconditionally: the round went on up to it
    unsigned int pad;
};

// Must match src/ggml/stage_graph/cond.rs (`CondArgs`).
struct legion_mtp_cond {
    unsigned long long handle;    // cudaGraphConditionalHandle of the IF node
    const unsigned int * alive;
};
}

namespace {

// The round goes on only if it went on up to this pass and no row's probability is below the
// floor: `pmax < p_min` stops a row, as on the host (a NaN goes on there too).
__global__ void mtp_decide_kernel(const legion_mtp_decide p) {
    unsigned int on = (p.reset != 0u || *p.alive != 0u) ? 1u : 0u;
    for (unsigned int i = 0; i < p.rows; ++i) {
        if (p.pmax[i] < p.p_min) {
            on = 0u;
        }
    }
    *p.alive = on;
}

__global__ void mtp_cond_kernel(const legion_mtp_cond p) {
    cudaGraphSetConditional(p.handle, *p.alive != 0u ? 1u : 0u);
}

}  // namespace

// The kernels as the CUDA runtime knows them, for Rust's launch and graph node (see gate_q8.cu).
extern "C" const void * const legion_mtp_decide_kernel = reinterpret_cast<const void *>(&mtp_decide_kernel);
extern "C" const void * const legion_mtp_cond_kernel = reinterpret_cast<const void *>(&mtp_cond_kernel);
