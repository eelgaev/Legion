// nvfp4_mm's GEMM of 128 rows per CTA (see common.cuh): its own translation unit, compiled in
// parallel with the other parts.
#include "common.cuh"

template <> int legion_nvfp4::launch_gemm<1>(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    return gemm_nt<1>(p, nt, variant, st);
}
