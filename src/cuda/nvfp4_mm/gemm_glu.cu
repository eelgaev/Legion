// nvfp4_mm's fused SwiGLU GEMM (see common.cuh): its own translation unit, compiled in parallel
// with the other parts. Also its TMA-fed grouped variant (gemm_glu_tma.cuh, #1183).
#include <cuda.h>

#include "common.cuh"

namespace {
#include "../sm90/tma.cuh"
#include "gemm_glu_tma.cuh"
}  // namespace

int legion_nvfp4::launch_swiglu_gemm(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    switch (nt) {
        case 4: return variant == 31 ? gemm<2, 4, true>(p, st) : -1;
        case 8: return gemm_variant<2, 8, true>(p, variant, st);
        case 16: return gemm_variant<2, 16, true>(p, variant, st);
        default: return -1;
    }
}

int legion_nvfp4_swiglu_gemm_tma_launch(const legion_nvfp4_mm * p, int nt, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    switch (nt) {
        case 8: return launch_glu_tma<8>(*p, st);
        case 16: return launch_glu_tma<16>(*p, st);
        default: return -1;
    }
}
