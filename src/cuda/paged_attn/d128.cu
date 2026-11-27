// paged_attn's kernels for head size 128 (see common.cuh): their own translation unit, compiled
// in parallel with the other head sizes'.
#include "common.cuh"

template <> int legion_paged::launch_dim<128>(const legion_paged_attn & p, cudaStream_t st) {
    return launch_d<128>(p, st);
}

template <> int legion_paged::launch_part_dim<128>(const legion_paged_attn & p, int n0, int part, cudaStream_t st) {
    return launch_part_d<128>(p, n0, part, st);
}
