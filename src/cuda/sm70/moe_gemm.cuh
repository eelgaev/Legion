// moe_gemm.cu, sm_70 (Volta) and newer: the mma.sync.m8n8k4 K step (step_mma) gemm_kernel<..,
// false> runs. m8n8k4 is valid on every sm_70+ device; use_mma16 in moe_gemm.cu keeps it on
// Volta (and wherever no sm_80 code is in the binary).
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace, after the tile
// layout helpers (BK, koff) it uses.

using Acc = float[8];  // an m8n8k4 F32 accumulator

// 16 bytes from a shared-memory address
__device__ __forceinline__ uint4 lds128(uint32_t a) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
    return v;
}

__device__ __forceinline__ void mma884(Acc & d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// One K step of a warp's products on the stage at `st`: rows of A (pairs) at byte offsets
// pa[0..MV) (swizzled rows), rows of B (weights) at pb[0..NJ) (chunk planes PB bytes apart). The
// chunk after the current one is loaded into the other register buffer while the current one's
// products run. The step's sums start from zero and are added to `tot` in F32 (Volta truncates
// when it accumulates).
template <int MV, int MI, int NJ, int PB>
__device__ __forceinline__ void step_mma(const char * st, const int (&pa)[MI], const int (&pb)[NJ],
                                         Acc (&tot)[MI][NJ]) {
    const uint32_t s0 = (uint32_t) __cvta_generic_to_shared(st);
    uint32_t sa[MV], sb[NJ];
#pragma unroll
    for (int i = 0; i < MV; i++) sa[i] = s0 + pa[i];
#pragma unroll
    for (int j = 0; j < NJ; j++) sb[j] = s0 + pb[j];
    // planes 4..7: slot bit 2 flipped (see koff)
    uint32_t sb4[NJ];
#pragma unroll
    for (int j = 0; j < NJ; j++) sb4[j] = sb[j] ^ 64;
    Acc acc[MV][NJ];
#pragma unroll
    for (int i = 0; i < MV; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int v = 0; v < 8; v++) acc[i][j][v] = 0.0f;
    uint4 a[2][MV], b[2][NJ];
    auto load = [&](int c, int u) {
#pragma unroll
        for (int j = 0; j < NJ; j++) b[u][j] = lds128((c < 4 ? sb[j] : sb4[j]) + c * PB);
#pragma unroll
        for (int i = 0; i < MV; i++) a[u][i] = lds128(sa[i] ^ (c << 4));
    };
    load(0, 0);
#pragma unroll
    for (int c = 0; c < BK / 8; c++) {
        const int u = c & 1;
        if (c + 1 < BK / 8) load(c + 1, u ^ 1);
#pragma unroll
        for (int i = 0; i < MV; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) mma884(acc[i][j], a[u][i].x, a[u][i].y, b[u][j].x, b[u][j].y);
#pragma unroll
        for (int i = 0; i < MV; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) mma884(acc[i][j], a[u][i].z, a[u][i].w, b[u][j].z, b[u][j].w);
    }
#pragma unroll
    for (int i = 0; i < MV; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int v = 0; v < 8; v++) tot[i][j][v] += acc[i][j][v];
}
