// expert_mv's kernels for IQ3_XXS, IQ3_S and IQ4_XS (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_IQ3_XXS>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_IQ3_XXS>(p, smem, stream);
}

template <> int legion_emv::launch_type<GGML_TYPE_IQ3_S>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_IQ3_S>(p, smem, stream);
}

template <> int legion_emv::launch_type<GGML_TYPE_IQ4_XS>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_IQ4_XS>(p, smem, stream);
}
