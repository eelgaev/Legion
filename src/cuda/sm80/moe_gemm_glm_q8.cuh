// moe_gemm.cu, sm_80 and newer, GLM-5.3-Flash only, opt-in (#603: `legion serve
// --glm-expert-q8`): the prompt steps' routed-expert GEMMs on Project Maya's IQ2 experts
// (Maya-S: IQ2_XXS gate/up; IQ2_S down, IQ3_XXS down in 8 layers, #594) on int8 tensor cores,
// MMQ-style. The activations are
// quantized to Q8_1-style blocks (int8 values, one F32 scale per 32: d = amax / 127, q =
// round(x / d)), once per step for gate and up (prep_q8_kernel) and in the gate/up epilogue for
// down (store_hq); the IQ2 / IQ3 grid values are decoded straight to int8 (IQ2 codebook bytes 8,
// 25 and 43, IQ3_XXS's 4..62, signs applied); each 32-value block's integer dot product is scaled in F32 by the weight
// block's and the activation block's scales. Output-moving: the activations are rounded to 8
// bits instead of 16, so this is a switch, off by default.
//
// The approach (int8 activations with per-32 scales, IQ2 grid values as int8 operands of
// mma.sync, per-block F32 scaling of the integer sums) follows llama.cpp's
// ggml/src/ggml-cuda/mmq.cuh and quantize.cu (MIT, (c) the ggml authors); the code is Legion's.
//
// Not a standalone header: moe_gemm.cu includes it inside its anonymous namespace after
// sm80/moe_gemm_glm.cuh (it uses Tile, group_of, store_tile, store_d, ld_u32_even, ldsm_x4,
// the cp.async helpers, frag::TMF and the GGUF grid tables) and launch_pdl.cuh.
//
// The MMAs are transposed as in gemm_glm_dec_kernel: 16 weight rows are the A operand (m16),
// 8 pairs the B operand (n8). A block: TPB plan tiles of one expert (64 pairs, eight n8 tiles,
// each) x RPB row tiles of BN weight rows (WMT m16 tiles per warp), one row tile after the
// other (see Q8Cfg). Weight stages are whole super-blocks as in gemm_glm_kernel; X
// slices are 128 int8 values per pair (8 swizzled 16-byte chunks, read with ldmatrix) plus the
// pair's four scales.
//
// Integer sums to floats: each MMA starts from C = 0x4B400000 (the bits of 1.5 * 2^23), so the
// int32 result's bits are the float 12582912 + dot exactly (|dot| <= 32 * 127 * 62 < 2^22),
// and one fused multiply-add with the activation scale removes the offset: no I2F.

__device__ __forceinline__ void q8_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

constexpr uint32_t Q8_MAGIC = 0x4B400000u;  // 1.5 * 2^23 as a float
constexpr float Q8_OFF = -12582912.0f;

// d = c * a * b (m16n8k32, int8 x int8 -> int32), every accumulator starting from c0
__device__ __forceinline__ void mma_s8_16832(uint32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                             uint32_t c0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c0));
#else
    __trap();
#endif
}

// the same for m16n8k16 (A rows g, g + 8: a0, a1; B: b0)
__device__ __forceinline__ void mma_s8_16816(uint32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t c0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(c0));
#else
    __trap();
#endif
}

// Bits 0..3 of n as byte masks 0x00 / 0xFF.
__host__ __device__ constexpr uint32_t q8_nibble_mask(uint32_t n) { return ((n * 0x00204081u) & 0x01010101u) * 0xFFu; }

// The decode tables. A lane (g, tg) needs values 4 (tg & 1) .. + 3 of groups of 8 (tg >> 1) and
// (tg >> 1) + 2 of each 32-value block: per codebook entry and half, the four bytes and their
// negations ({pos, neg}, neg = -pos per byte: the grid bytes are 8..43, so no borrows), and a
// byte mask per sign nibble that picks between them.
//
// IQ2_XXS: 66 bytes per 256 values (see GlmIq2xxs). Sign masks per 7-bit sign index (parity bit
// added as ggml's ksigns_iq2xs) and half. One scale per 32 values.
struct Q8Iq2xxs {
    static constexpr int SB = 66, RB = 80, NSC = 1;
    static constexpr int TAB_G = 256 * 2 * 8;
    static constexpr int SMEM_TAB = TAB_G + 128 * 2 * 4;
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint2 * gt = (uint2 *) t;
        for (int i = tid; i < 256 * 2; i += nt) {
            const uint32_t v = (uint32_t) (iq2xxs_grid[i / 2] >> (32 * (i % 2)));
            gt[i] = make_uint2(v, ~v + 0x01010101u);
        }
        uint32_t * sg = (uint32_t *) (t + TAB_G);
        for (int i = tid; i < 128 * 2; i += nt) {
            const uint32_t s7 = i / 2, s8 = s7 | ((__popc(s7) & 1) << 7);
            sg[i] = q8_nibble_mask((s8 >> (4 * (i % 2))) & 15);
        }
    }
    // this lane's A words of sub-block ib of one weight row (k 4 tg.., k 16 + 4 tg..) and its scale
    __device__ static void frag(const unsigned char * t, const unsigned char * row, int sh, int ib, int tg, float d0,
                                uint32_t & lo, uint32_t & hi, float (&dw)[NSC]) {
        const uint2 * gt = (const uint2 *) t + (tg & 1);
        const uint32_t * sg = (const uint32_t *) (t + TAB_G) + (tg & 1);
        const uint32_t qx = ld_u32_even(row, sh + 2 + 8 * ib), qy = ld_u32_even(row, sh + 6 + 8 * ib);
        dw[0] = d0 * (0.5f + (qy >> 28)) * 0.25f;
        uint32_t w[2];
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const int grp = (tg >> 1) + 2 * q;
            const uint2 v = gt[2 * ((qx >> (8 * grp)) & 0xFF)];
            const uint32_t m = sg[2 * ((qy >> (7 * grp)) & 127)];
            w[q] = (v.x & ~m) | (v.y & m);
        }
        lo = w[0];
        hi = w[1];
    }
};

// IQ2_S: 82 bytes per 256 values (see GlmIq2s): 10-bit grid indices, a sign byte per group of 8
// (masks per nibble), two scales per 32 values (one per 16: low nibble, high nibble).
struct Q8Iq2s {
    static constexpr int SB = 82, RB = 96, NSC = 2;
    static constexpr int TAB_G = 1024 * 2 * 8;
    static constexpr int SMEM_TAB = TAB_G + 16 * 4;
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint2 * gt = (uint2 *) t;
        for (int i = tid; i < 1024 * 2; i += nt) {
            const uint32_t v = (uint32_t) (iq2s_grid[i / 2] >> (32 * (i % 2)));
            gt[i] = make_uint2(v, ~v + 0x01010101u);
        }
        uint32_t * nm = (uint32_t *) (t + TAB_G);
        for (int i = tid; i < 16; i += nt) {
            nm[i] = q8_nibble_mask((uint32_t) i);
        }
    }
    __device__ static void frag(const unsigned char * t, const unsigned char * row, int sh, int ib, int tg, float d0,
                                uint32_t & lo, uint32_t & hi, float (&dw)[NSC]) {
        const uint2 * gt = (const uint2 *) t + (tg & 1);
        const uint32_t * nm = (const uint32_t *) (t + TAB_G);
        const uint32_t ql = ld_u32_even(row, sh + 2 + 4 * ib), sg = ld_u32_even(row, sh + 34 + 4 * ib);
        const uint32_t qh = row[sh + 66 + ib], sc = row[sh + 74 + ib];
        dw[0] = d0 * (0.5f + (sc & 0xf)) * 0.25f;
        dw[1] = d0 * (0.5f + (sc >> 4)) * 0.25f;
        uint32_t w[2];
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const int grp = (tg >> 1) + 2 * q;
            const uint32_t idx = ((ql >> (8 * grp)) & 0xFF) | ((qh << (8 - 2 * grp)) & 0x300);
            const uint2 v = gt[2 * idx];
            const uint32_t m = nm[(sg >> (8 * grp + 4 * (tg & 1))) & 15];
            w[q] = (v.x & ~m) | (v.y & m);
        }
        lo = w[0];
        hi = w[1];
    }
};

// IQ3_XXS: 98 bytes per 256 values (see Piece<T_IQ3_XXS>): d, the grid indices (8 per sub-block,
// one per 4 values), then per sub-block a word of four 7-bit sign indices and the scale (top 4
// bits). A lane's 4 values of a group are one whole iq3xxs_grid entry (bytes 4..62), so the table
// is the grid with its negations; the sign masks are Q8Iq2xxs's. One scale per 32 values.
struct Q8Iq3xxs {
    static constexpr int SB = 98, RB = 112, NSC = 1;
    static constexpr int TAB_G = 256 * 8;
    static constexpr int SMEM_TAB = TAB_G + 128 * 2 * 4;
    __device__ static void tables(unsigned char * t, int tid, int nt) {
        uint2 * gt = (uint2 *) t;
        for (int i = tid; i < 256; i += nt) {
            const uint32_t v = iq3xxs_grid[i];
            gt[i] = make_uint2(v, ~v + 0x01010101u);
        }
        uint32_t * sg = (uint32_t *) (t + TAB_G);
        for (int i = tid; i < 128 * 2; i += nt) {
            const uint32_t s7 = i / 2, s8 = s7 | ((__popc(s7) & 1) << 7);
            sg[i] = q8_nibble_mask((s8 >> (4 * (i % 2))) & 15);
        }
    }
    __device__ static void frag(const unsigned char * t, const unsigned char * row, int sh, int ib, int tg, float d0,
                                uint32_t & lo, uint32_t & hi, float (&dw)[NSC]) {
        const uint2 * gt = (const uint2 *) t;
        const uint32_t * sg = (const uint32_t *) (t + TAB_G) + (tg & 1);
        const uint32_t aux = ld_u32_even(row, sh + 66 + 4 * ib);
        dw[0] = d0 * (0.5f + (aux >> 28)) * 0.5f;
        uint32_t w[2];
#pragma unroll
        for (int q = 0; q < 2; q++) {
            // group grp = (tg >> 1) + 2 q, its entry 2 grp + (tg & 1)
            const int grp = (tg >> 1) + 2 * q;
            const uint2 v = gt[row[sh + 2 + 8 * ib + tg + 4 * q]];
            const uint32_t m = sg[2 * ((aux >> (7 * grp)) & 127)];
            w[q] = (v.x & ~m) | (v.y & m);
        }
        lo = w[0];
        hi = w[1];
    }
};

// x (F32) to Q8 rows for the gate/up GEMM: xq [n_tok][n_embd] int8 at p.xh, then the scales
// xd [n_tok][n_embd / 32] (F32); the pairs counted as prep_kernel counts them. 256 threads, four
// values each, eight threads per 32-value block (n_embd % 256 == 0: whole warps).
__global__ void __launch_bounds__(256) prep_q8_kernel(const legion_moe_args p) {
    q8_pdl_wait();
    const int64_t t = blockIdx.x;
    const float4 * xr = (const float4 *) (p.x + t * p.x_ld);
    int8_t * xq = (int8_t *) p.xh;
    uint32_t * q = (uint32_t *) (xq + t * p.n_embd);
    float * xd = (float *) (xq + p.n_tok * p.n_embd) + t * (p.n_embd / 32);
    for (int64_t i = threadIdx.x; i < p.n_embd / 4; i += blockDim.x) {
        const float4 v = xr[i];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
        for (int o = 1; o < 8; o *= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        // ggml's quantize_q8_1: d = amax / 127, q = round(x / d)
        const float d = amax / 127.0f;
        auto qv = [&](float x) -> uint32_t {
            return (uint32_t) (uint8_t) (int8_t) (amax == 0.0f ? 0.0f : roundf(x / d));
        };
        q[i] = qv(v.x) | (qv(v.y) << 8) | (qv(v.z) << 16) | (qv(v.w) << 24);
        if (i % 8 == 0) {
            xd[i / 8] = d;
        }
    }
    if (threadIdx.x < p.n_used) {
        const int e = p.ids[t * p.ids_ld + threadIdx.x];
        if (expert_ok(p, e)) {
            atomicAdd(&p.count[e], 1);
        }
    }
}

// A launch shape: NW warps of WMT m16 tiles (BN = 16 WMT NW weight rows), XS X slices in the
// ring, TPB consecutive plan tiles of one expert per block (group_of; TM = 64 TPB pairs), each
// row's next PF super-blocks brought into L2 in one bulk prefetch every PF stages (0: none), and
// RPB row tiles per block, one after the other (the prefetch runs on across them).
template <int NW_, int WMT_, int XS_, int TPB_, int PF_, int RPB_ = 1>
struct Q8Cfg {
    static constexpr int NW = NW_, WMT = WMT_, XS = XS_, TPB = TPB_, PF = PF_, RPB = RPB_;
};

template <class W, Kind K, class C>
struct Q8Shape {
    static constexpr int NW = C::NW, WMT = C::WMT, XS = C::XS, TPB = C::TPB, PF = C::PF;
    static constexpr int NT = 32 * NW, BN = 16 * WMT * NW, TE = frag::TMF, TM = TE * TPB;
    static constexpr int NTN = TM / 8;      // n8 tiles of pairs
    static constexpr int SK = 128;          // values per X slice
    static constexpr int KB = SK / 32;      // 32-value blocks per slice
    static constexpr int SPW = 256 / SK;    // slices per weight stage (super-block)
    static constexpr int D = XS - 1;        // X slices in flight ahead
    static constexpr int XQ = TM * SK;      // a slice: the rows' bytes, then their KB scales each
    static constexpr int XST = XQ + TM * 4 * KB;
    static constexpr int XC = TM * (SK / 16 + 1);  // its 16-byte copies
    static constexpr int XV = (XC + NT - 1) / NT;
    static constexpr int WST = BN * W::RB;
    static constexpr int LDE = BN + 4;
    static constexpr int SMEM_AB = XS * XST + 2 * WST + W::SMEM_TAB;
    static constexpr int SMEM_E = TE * LDE * 4 + TE * 8;  // the epilogue, one plan tile at a time
    static constexpr int SMEM = SMEM_AB > SMEM_E ? SMEM_AB : SMEM_E;
    static_assert(TE == 64, "eight n8 tiles of pairs per plan tile");
    static_assert(BN <= NT, "one row per thread in the L2 prefetch");
    static_assert(SMEM_E <= XS * XST + 2 * WST, "the epilogue leaves the tables for the next row tile");
    static_assert(KB == 4, "a slice's scales are one 16-byte copy per pair");
    static_assert(D >= 1 && D <= SPW, "the next weight stage, in slice m's group, is waited for at slice m + D");
};

// Gate/up SwiGLU (store_tile's formula) to H as Q8 rows: hq [n_pairs][n_ff] int8 at p.h, then
// the scales hd [n_pairs][n_ff / 32] (F32). A warp per (pair, 32 H columns).
template <int TM, int BN, int LDE, int NT>
__device__ __forceinline__ void store_hq(const legion_moe_args & p, const float * E, int r0, int ne, int rt) {
    constexpr int HC = BN / 2, NB = HC / 32;
    static_assert(HC % 32 == 0, "whole 32-value blocks per block");
    int8_t * hq = (int8_t *) p.h;
    float * hd = (float *) (hq + p.n_tok * p.n_used * p.n_ff);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int i = warp; i < ne * NB; i += NT / 32) {
        const int r = i / NB, b = i % NB, c = 32 * b + lane;
        const float2 gu = *(const float2 *) (E + r * LDE + 2 * c);
        float gv = gu.x, u = gu.y;
        if (p.limit > 0.0f) {
            gv = fminf(gv, p.limit);
            u = fminf(fmaxf(u, -p.limit), p.limit);
        }
        const float h = gv / (1.0f + __expf(-gv)) * u;
        float amax = fabsf(h);
#pragma unroll
        for (int o = 16; o > 0; o /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        const int64_t row = (int64_t) (r0 + r);
        hq[row * p.n_ff + (int64_t) rt * HC + c] = (int8_t) (amax == 0.0f ? 0.0f : roundf(h / d));
        if (lane == 0) {
            hd[row * (p.n_ff / 32) + ((int64_t) rt * HC) / 32 + b] = d;
        }
    }
}

template <class W, Kind K, class C, bool HQ>
__global__ void __launch_bounds__(32 * C::NW, 1) gemm_glm_q8_kernel(const legion_moe_args p) {
    q8_pdl_wait();
    using S = Q8Shape<W, K, C>;
    constexpr int RB = W::RB, NT = S::NT, BN = S::BN, SK = S::SK, SPW = S::SPW, D = S::D, XST = S::XST;
    constexpr int TM = S::TM, TE = S::TE, NTN = S::NTN, WMT = S::WMT, XS = S::XS, PF = S::PF, NSC = W::NSC;
    extern __shared__ __align__(16) unsigned char sm[];

    int kt;
    const Tile tl = group_of(p, blockIdx.y, S::TPB, kt);
    if (tl.ne <= 0) {
        return;
    }
    const int r0 = tl.r0, ne = tl.ne;
    const int64_t KD = K == GU ? p.n_embd : p.n_ff;
    const int NS = (int) (KD / SK), NR = (int) (KD / 256);
    constexpr int RPB = C::RPB;

    unsigned char * xbuf = sm;
    unsigned char * wbuf = sm + XS * XST;
    unsigned char * tabs = wbuf + 2 * S::WST;

    // weight row r of row tile rt (gate/up: even rows gate, odd rows up, H column rt * BN / 2 + r / 2)
    auto wrow = [&](int rt, int r) -> const uint8_t * {
        if (K == GU) {
            const int64_t row = (int64_t) rt * (BN / 2) + r / 2;
            return ((r & 1) ? p.up : p.gate) + tl.slot * p.gu_nb2 + row * p.gu_nb1;
        }
        return p.down + tl.slot * p.down_nb2 + ((int64_t) rt * BN + r) * p.down_nb1;
    };
    // the block's row tiles: RPB in a row, one after the other
    int rt = blockIdx.x * RPB;
    // stage R: the aligned RB bytes holding super-block R of each row
    auto issue_w = [&](int R) {
        if (R < NR) {
            unsigned char * dst = wbuf + (R & 1) * S::WST;
            const int64_t a = ((int64_t) R * W::SB) & ~(int64_t) 15;
            for (int f = threadIdx.x; f < BN * (RB / 16); f += NT) {
                const int r = f / (RB / 16), c = f % (RB / 16);
                cp_async16(dst + r * RB + 16 * c, wrow(rt, r) + a + 16 * c);
            }
        }
    };
    // super-blocks V .. V + PF - 1 (counted on through the block's row tiles) of row threadIdx.x
    // into L2 (the GB10 streams the stages' short pieces slowly from DRAM): the 16-byte pieces that
    // cover them, inside the row (a whole number of pieces, checked by the launcher)
    auto prefetch = [&](int V) {
        const int q = V / NR, R = V % NR;
        if (PF > 0 && q < RPB && threadIdx.x < BN) {
            const int64_t a = ((int64_t) R * W::SB) & ~(int64_t) 15;
            const int64_t e = min((int64_t) (R + PF) * W::SB + 15, (int64_t) NR * W::SB) & ~(int64_t) 15;
            bulk_prefetch_l2(wrow(blockIdx.x * RPB + q, threadIdx.x) + a, (uint32_t) (e - a));
        }
    };
    // this thread's copies of each X slice: Q8 rows gathered by token (gate/up) or in list order
    // (down), 8 chunks per row (swizzled), then each row's KB scales
    const int8_t * xq = (const int8_t *) (K == GU ? (const void *) p.xh : (const void *) p.h);
    const int64_t xrows = K == GU ? p.n_tok : p.n_tok * p.n_used;
    const unsigned char * xd = (const unsigned char *) (xq + xrows * KD);
    const unsigned char * xsrc[S::XV];
    int xdst[S::XV], xstep[S::XV];
#pragma unroll
    for (int i = 0; i < S::XV; i++) {
        const int c = threadIdx.x + NT * i;
        const bool chunk = c < TM * (SK / 16);
        const int r = chunk ? c / (SK / 16) : c - TM * (SK / 16);
        xsrc[i] = nullptr;
        xdst[i] = xstep[i] = 0;
        if (c < S::XC && r < ne) {
            const int64_t xr = K == GU ? p.list[r0 + r] / p.n_used : r0 + r;
            if (chunk) {
                const int ch = c % (SK / 16);
                xsrc[i] = (const unsigned char *) (xq + xr * KD) + 16 * ch;
                xdst[i] = r * SK + ((ch ^ (r & 7)) << 4);
                xstep[i] = SK;
            } else {
                xsrc[i] = xd + xr * (KD / 8);
                xdst[i] = S::XQ + 4 * S::KB * r;
                xstep[i] = 4 * S::KB;
            }
        }
    }
    auto issue_x = [&](int m) {
        if (m < NS) {
            unsigned char * dst = xbuf + (m % XS) * XST;
#pragma unroll
            for (int i = 0; i < S::XV; i++) {
                if (xsrc[i]) {
                    cp_async16(dst + xdst[i], xsrc[i] + (int64_t) m * xstep[i]);
                }
            }
        }
    };

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, tg = lane & 3;
    const int nvalid = (ne + 7) / 8;
    // ldmatrix: this lane's row (plus 8 nt) and chunk (plus 2 kk); matrices: n-tile nt, k 0..15
    // and 16..31, then n-tile nt + 1
    const int lrow = 8 * (lane >> 4) + (lane & 7), lch = (lane >> 3) & 1;

    W::tables(tabs, threadIdx.x, NT);  // visible after the first barrier
    prefetch(0);
    for (int q = 0; q < RPB; q++, rt++) {
        float tot[WMT][NTN][4];
#pragma unroll
        for (int i = 0; i < WMT; i++)
#pragma unroll
            for (int n = 0; n < NTN; n++) tot[i][n][0] = tot[i][n][1] = tot[i][n][2] = tot[i][n][3] = 0.0f;

        issue_w(0);
        issue_x(0);
        cp_async_commit();
#pragma unroll
        for (int s = 1; s < D; s++) {
            issue_x(s);
            cp_async_commit();
        }
        for (int m = 0; m < NS; m++) {
            // X slice m (and its weight stage) landed; every warp is past slice m - 1, so its X
            // buffer and, at a stage start, the previous stage's are free
            cp_async_wait<D - 1>();
            __syncthreads();
            issue_x(m + D);
            if (m % SPW == 0) {
                if constexpr (PF > 0) {
                    if ((m / SPW) % PF == 0) {
                        prefetch(q * NR + m / SPW + PF);
                    }
                }
                issue_w(m / SPW + 1);
            }
            cp_async_commit();

            const int R = m / SPW, c = m % SPW;
            const unsigned char * ws = wbuf + (R & 1) * S::WST;
            const int sh = (R * W::SB) & 15;  // the super-block's offset in its slot
            const unsigned char * xs = xbuf + (m % XS) * XST;
            const uint32_t xa = (uint32_t) __cvta_generic_to_shared(xs);
            const float * xsc = (const float *) (xs + S::XQ) + S::KB * 2 * tg;  // pairs 2 tg, 2 tg + 1

            const unsigned char * wr[WMT][2];
            float d0[WMT][2];
#pragma unroll
            for (int i = 0; i < WMT; i++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    wr[i][r] = ws + (16 * (WMT * warp + i) + 8 * r + g) * RB;
                    d0[i][r] = __half2float(__ushort_as_half(*(const uint16_t *) (wr[i][r] + sh)));
                }

#pragma unroll
            for (int kb = 0; kb < S::KB; kb++) {
                const int ib = S::KB * c + kb;
                uint32_t a[WMT][4];
                float dw[WMT][2][NSC], off[WMT][2];
#pragma unroll
                for (int i = 0; i < WMT; i++)
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        W::frag(tabs, wr[i][r], sh, ib, tg, d0[i][r], a[i][r], a[i][2 + r], dw[i][r]);
                        // the offset's share of the row's scaled sum: -1.5 * 2^23 * (dw_lo + dw_hi)
                        off[i][r] = Q8_OFF * (NSC == 1 ? dw[i][r][0] : dw[i][r][0] + dw[i][r][NSC - 1]);
                    }
#pragma unroll
                for (int nt = 0; nt < NTN; nt += 2) {
                    // one branch per n-tile pair (a pair past ne gives sums nobody stores)
                    if (nt >= nvalid) {
                        break;
                    }
                    uint32_t b[2][2];
                    ldsm_x4(xa + (8 * nt + lrow) * SK + (((2 * kb + lch) ^ (lane & 7)) << 4), b[0][0], b[0][1],
                            b[1][0], b[1][1]);
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const int n = nt + j;
                        const float dy[2] = {xsc[S::KB * 8 * n + kb], xsc[S::KB * 8 * n + S::KB + kb]};
#pragma unroll
                        for (int i = 0; i < WMT; i++) {
                            // u = dw * dot (the offset out: off's rounding is under one unit of the
                            // dot), then into the totals times the pair's scale
                            float u[4];
                            if constexpr (NSC == 1) {
                                uint32_t cc[4];
                                mma_s8_16832(cc, a[i], b[j][0], b[j][1], Q8_MAGIC);
#pragma unroll
                                for (int v = 0; v < 4; v++)
                                    u[v] = fmaf(__uint_as_float(cc[v]), dw[i][v >> 1][0], off[i][v >> 1]);
                            } else {
                                uint32_t lo[4], hi[4];
                                mma_s8_16816(lo, a[i][0], a[i][1], b[j][0], Q8_MAGIC);
                                mma_s8_16816(hi, a[i][2], a[i][3], b[j][1], Q8_MAGIC);
#pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    u[v] = fmaf(__uint_as_float(lo[v]), dw[i][v >> 1][0], off[i][v >> 1]);
                                    u[v] = fmaf(__uint_as_float(hi[v]), dw[i][v >> 1][NSC - 1], u[v]);
                                }
                            }
#pragma unroll
                            for (int v = 0; v < 4; v++) tot[i][n][v] = fmaf(u[v], dy[v & 1], tot[i][n][v]);
                        }
                    }
                }
            }
        }
        cp_async_wait<0>();
        __syncthreads();

        // lane (g, tg) holds rows 16 (WMT warp + i) + g (v < 2) and + 8 (v >= 2) of pairs 8 n + 2 tg
        // + (v & 1); through E (pair-major) to store_tile or store_hq, one plan tile at a time
        float * E = (float *) sm;
        constexpr int NE = TE / 8;
#pragma unroll
        for (int e = 0; e < S::TPB; e++) {
            if (e * TE >= ne) {
                break;
            }
#pragma unroll
            for (int i = 0; i < WMT; i++)
#pragma unroll
                for (int n = 0; n < NE; n++) {
                    if (NE * e + n < nvalid) {
#pragma unroll
                        for (int v = 0; v < 4; v++) {
                            E[(8 * n + 2 * tg + (v & 1)) * S::LDE + 16 * (WMT * warp + i) + g + 8 * (v >> 1)] =
                                tot[i][NE * e + n][v];
                        }
                    }
                }
            __syncthreads();
            if constexpr (K == GU && HQ) {
                store_hq<TE, BN, S::LDE, NT>(p, E, r0 + e * TE, min(TE, ne - e * TE), rt);
            } else {
                store_tile<TE, BN, S::LDE, NT, K, true>(p, E, r0 + e * TE, min(TE, ne - e * TE), rt);
            }
            __syncthreads();
        }
    }
}

// One launch over n_rows weight rows (gate and up rows counted apart) through launch_pdl; false
// when the device refuses the shared memory or the rows are not whole blocks.
template <class W, Kind K, class C, bool HQ>
bool glm_q8_ready(const legion_moe_args * a) {
    using S = Q8Shape<W, K, C>;
    static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES];
    const int64_t n_rows = K == GU ? 2 * a->n_ff : a->n_embd;
    const int64_t n_sb = (K == GU ? a->n_embd : a->n_ff) / 256;  // super-blocks per row
    return n_rows % (S::BN * C::RPB) == 0 && (C::PF == 0 || n_sb % C::PF == 0) &&
           legion_device::smem_opt_in(smem, gemm_glm_q8_kernel<W, K, C, HQ>, S::SMEM);
}

template <class W, Kind K, class C, bool HQ>
bool glm_q8_launch(const legion_moe_args * a, unsigned grid_y, void * stream) {
    using S = Q8Shape<W, K, C>;
    if (!glm_q8_ready<W, K, C, HQ>(a)) {
        return false;
    }
    const int64_t n_rows = K == GU ? 2 * a->n_ff : a->n_embd;
    launch_pdl(gemm_glm_q8_kernel<W, K, C, HQ>, dim3((unsigned) (n_rows / S::BN / C::RPB), grid_y), S::NT, S::SMEM,
               (cudaStream_t) stream, *a);
    return true;
}

// The launch shapes (picked on the GB10 with nsys over a 7K-token Maya-S prompt, #603; down from 8 to 16
// warps of one m16 tile each in #594: IQ2_S 6.85 -> 6.40 ms per launch, IQ3_XXS level)
using Q8Gu = Q8Cfg<8, 2, 2, 2, 4, 4>;
using Q8Down = Q8Cfg<16, 1, 3, 1, 4, 4>;
