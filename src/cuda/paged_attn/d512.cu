// paged_attn's kernels for head size 512 (see common.cuh): their own translation unit, compiled
// in parallel with the other head sizes'.
#include "common.cuh"

template <> int legion_paged::launch_dim<512>(const legion_paged_attn & p, cudaStream_t st) {
    return launch_d<512>(p, st);
}
