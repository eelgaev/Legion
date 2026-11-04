// nvfp4_mm's grouped mv variants of 16 tokens per CTA (see common.cuh): its own translation
// unit, compiled in parallel with the other parts. The plain grouped variants (16-block stages,
// 16-row tiles) are mv_nt2.cu's.
#include "common.cuh"

template <>
int legion_nvfp4::launch_mvg<2>(const legion_nvfp4_mm & p, bool glu, int rtw, int rpt, int sb, int pft,
                                cudaStream_t st) {
    return mvg<2>(p, glu, rtw, rpt, sb, pft, st);
}
