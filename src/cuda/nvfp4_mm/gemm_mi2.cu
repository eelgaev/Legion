// nvfp4_mm's GEMM of 256 rows per CTA (see common.cuh): its own translation unit, compiled in
// parallel with the other parts.
#include "common.cuh"

template <> int legion_nvfp4::launch_gemm<2>(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    return gemm_nt<2>(p, nt, variant, st);
}
