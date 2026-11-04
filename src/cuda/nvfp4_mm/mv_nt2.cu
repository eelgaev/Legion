// nvfp4_mm's mv kernel of 16 tokens per CTA (see common.cuh): its own translation unit, compiled
// in parallel with the other parts.
#include "common.cuh"

template <> int legion_nvfp4::launch_mv<2>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st) {
    return mv_rtw<2>(p, rtw, kw, rpt, st);
}
