// nvfp4_mm's fused SwiGLU mv kernel (see common.cuh): its own translation unit, compiled in
// parallel with the other parts.
#include "common.cuh"

int legion_nvfp4::launch_swiglu_mv(const legion_nvfp4_mm & p, int nt, cudaStream_t st) {
    switch (nt) {
        case 1: return mvs<1, 2, 8, MV_SB, MV_NS, true, true>(p, st);
        case 2: return mvs<2, 2, 8, MV_SB, MV_NS, true, true>(p, st);
        case 4: return mvs<4, 2, 8, MV_SB, MV_NS, true, true>(p, st);
        case 8: return mvs<8, 2, 8, MV_SB, MV_NS, true, true>(p, st);
        default: return -1;
    }
}
