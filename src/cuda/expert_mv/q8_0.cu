// expert_mv's kernels for Q8_0 (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_Q8_0>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_Q8_0>(p, smem, stream);
}
