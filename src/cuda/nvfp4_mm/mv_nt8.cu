// nvfp4_mm's mv kernel of 64 tokens per CTA (see common.cuh): its own translation unit, compiled
// in parallel with the other parts.
#include "common.cuh"

template <> int legion_nvfp4::launch_mv<8>(const legion_nvfp4_mm & p, int rtw, int kw, int rpt, cudaStream_t st) {
    // 33-64 tokens (--nvfp4-wide-mv): the decode kernel's 8 warps, 16 or 32 rows per CTA.
    if (kw != 8 || rpt != 1) {
        return -1;
    }
    return rtw == 1 ? mvs<8, 1, 8, MV_SB, MV_NS, true>(p, st)
         : rtw == 2 ? mvs<8, 2, 8, MV_SB, MV_NS, true>(p, st)
                    : -1;
}
