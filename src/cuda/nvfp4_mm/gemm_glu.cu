// nvfp4_mm's fused SwiGLU GEMM (see common.cuh): its own translation unit, compiled in parallel
// with the other parts.
#include "common.cuh"

int legion_nvfp4::launch_swiglu_gemm(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    switch (nt) {
        case 4: return variant == 31 ? gemm<2, 4, true>(p, st) : -1;
        case 8: return gemm_variant<2, 8, true>(p, variant, st);
        case 16: return gemm_variant<2, 16, true>(p, variant, st);
        default: return -1;
    }
}
