// expert_mv's kernels for IQ2_XXS and IQ2_S (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_IQ2_XXS>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_IQ2_XXS>(p, smem, stream);
}

template <> int legion_emv::launch_type<GGML_TYPE_IQ2_S>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_IQ2_S>(p, smem, stream);
}
