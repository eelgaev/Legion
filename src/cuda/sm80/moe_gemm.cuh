// moe_gemm.cu, sm_80 and newer: the mma.sync.m16n8k16 K step (step_mma16) gemm_kernel<.., true>
// and gemm_raw_kernel run.
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace, after the tile
// layout helpers (BK, row_base, koff) it uses. ldmatrix and the MMA keep their __CUDA_ARCH__
// guards and __trap() stubs, so the file compiles for every target architecture; use_mma16 in
// moe_gemm.cu decides per device at run time whether this path runs.

// ---- sm_80 and newer: mma.sync.m16n8k16 -----------------------------------------------------
//
// The same tiles in shared memory. A fragments (16 pairs x 16 K) come from the X tile with
// ldmatrix.x4: lane l gives row (l & 7) + 8 ((l >> 3) & 1) of the 16, chunk 2 kk + (l >> 4) (the
// swizzled row addresses as the loader stores them). B fragments (16 K x 8 weight rows, two
// n-tiles per ldmatrix.x4) come from the k-major W tile: lane l gives row 8 (l >> 4) + (l & 7) of
// the 16, chunk 2 kk + ((l >> 3) & 1). Accumulator c: rows g and g + 8 (g = lane / 4), columns
// 2 (lane % 4) and + 1. Each K step (4 products of 16) starts from zero and is added to the F32
// totals: the tensor cores of sm_12x truncate when they accumulate (see paged_attn.cu).

using Acc4 = float[4];

__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mma16816(Acc4 & c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// One K step (BK) of a warp's MT x NT products (16 x 8 each) on the stage at `st`: m-tile i at
// X rows mrow + 16 i (only the first `mvalid` hold pairs), n-tile j at W rows nrow + 8 j.
template <int MT, int NT, int TM, int BN>
__device__ __forceinline__ void step_mma16(const char * st, int mrow, int nrow, int mvalid, int lane,
                                           Acc4 (&tot)[MT][NT]) {
    static_assert(NT % 2 == 0, "two n-tiles per ldmatrix");
    const uint32_t xs = (uint32_t) __cvta_generic_to_shared(st);
    const uint32_t ws = xs + TM * BK * 2;
    Acc4 acc[MT][NT];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    const int ar = (lane & 7) + 8 * ((lane >> 3) & 1), ac = lane >> 4;
    const int br = 8 * (lane >> 4) + (lane & 7), bc = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < BK / 16; kk++) {
        uint32_t b[NT][2];
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
            ldsm_x4(ws + koff(nrow + 8 * j + br, 2 * kk + bc, BN), b[j][0], b[j][1], b[j + 1][0], b[j + 1][1]);
        }
#pragma unroll
        for (int i = 0; i < MT; i++) {
            if (i < mvalid) {
                uint32_t a[4];
                ldsm_x4(xs + (row_base(mrow + 16 * i + ar) ^ ((2 * kk + ac) << 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int j = 0; j < NT; j++) mma16816(acc[i][j], a, b[j][0], b[j][1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int v = 0; v < 4; v++) tot[i][j][v] += acc[i][j][v];
}
