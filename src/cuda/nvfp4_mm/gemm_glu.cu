// nvfp4_mm's fused SwiGLU GEMM (see common.cuh): its own translation unit, compiled in parallel
// with the other parts.
#include "common.cuh"

#include <cuda.h>

namespace {
#include "../sm90/tma.cuh"
#include "../sm90/nvfp4_glu_ws.cuh"
}  // namespace

int legion_nvfp4::launch_swiglu_gemm(const legion_nvfp4_mm & p, int nt, int variant, cudaStream_t st) {
    switch (nt) {
        case 4:
            if (const int r = gemm_l1<2, 4, true>(p, variant, st); r != -2) {
                return r;
            }
            return variant == 31 ? gemm<2, 4, true>(p, st) : -1;
        case 8: return gemm_variant<2, 8, true>(p, variant, st);
        case 16:
            if (const int r = glu_ws(p, variant, st); r != -2) {
                return r;
            }
            return variant >= 100 ? gemm_diag<2, 16, true>(p, variant, st) : gemm_variant<2, 16, true>(p, variant, st);
        default: return -1;
    }
}
