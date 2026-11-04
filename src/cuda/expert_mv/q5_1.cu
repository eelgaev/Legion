// expert_mv's kernels for Q5_1 (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_Q5_1>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_Q5_1>(p, smem, stream);
}
