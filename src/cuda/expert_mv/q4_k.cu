// expert_mv's kernels for Q4_K (see common.cuh): their own translation unit, compiled in
// parallel with the other types'.
#include "kernels.cuh"

template <> int legion_emv::launch_type<GGML_TYPE_Q4_K>(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return launch_any<GGML_TYPE_Q4_K>(p, smem, stream);
}
