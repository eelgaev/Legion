// expert_mv's kernels for Q2_K and Q3_K (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_Q2_K>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_Q2_K>(p, smem, stream);
}

template <> int legion_emv::launch_type<GGML_TYPE_Q3_K>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_Q3_K>(p, smem, stream);
}
