// fp8_mm.cu, sm_89 and newer: the W8A8 GEMMs on mma.sync m16n8k32 E4M3 x E4M3 -> F32
// (fp8_gemm8, fp8_gemm8_skinny and their tile shapes).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm80/fp8_mm.cuh (cp_async16) and a8_static / a8_scale (the epilogue's scale). The
// kernel bodies are built only where __CUDA_ARCH__ >= 890 and are empty elsewhere, so the file
// compiles for every target architecture; fp8_mma() in fp8_mm.cu (compute capability 8.9+)
// decides per device at run time whether they run.

// More tokens: 64 * WM rows x 128 tokens per CTA, 8 warps as WM x (8 / WM) of 64 rows x
// 128 / (8 / WM) tokens, mma.sync m16n8k32 E4M3 x E4M3 -> F32 (sm_89+) straight from the raw
// bytes through ldmatrix (a 16-byte row of E4M3 is what ldmatrix moves as 8 b16), K steps of BK
// bytes through STAGES cp.async stages (a8::Cfg). WM 2: 128 x 128, warps of 64 x 32. WM 4
// (#749): 256 x 128, warps of 64 x 64, 1.33x the MMAs per byte copied into shared memory: on
// the GB10 the 128 x 128 tile (112 TFLOPS) and the W4A4 GEMM's (179) both move ~800-875 GB/s
// from L2, so the copies bound them, not the FP8 rate (237 TFLOPS of E4M3 mma.sync with F32 sums
// measured there). Every output takes the same MMAs in the same k order in every shape and
// stage layout: the same bits.
namespace a8 {
constexpr int BN = 128;
// A stage layout: STAGES stages of BK bytes of k. SWZ: rows BK bytes apart, each row's 16-byte
// chunks XOR-swizzled by its row so the 8 rows an ldmatrix phase reads hit distinct banks;
// else rows padded to BK + 16 bytes.
template <int WM_, int STAGES_ = 3, int BK_ = 64, bool SWZ_ = false> struct Cfg {
    static constexpr int WM = WM_, STAGES = STAGES_, BK = BK_;
    static constexpr bool SWZ = SWZ_;
    static constexpr int TM = 64 * WM, CH = BK / 16, LD = SWZ ? BK : BK + 16;
    static constexpr size_t TSTAGE = (size_t) (TM + BN) * LD, SMEM = STAGES * TSTAGE;
    static_assert(WM == 2 || WM == 4, "warp rows");
    static_assert(BK % 32 == 0 && (!SWZ || BK == 64 || BK == 128), "stage layout");
    // Byte offset of 16-byte chunk `chunk` of row `row` in a stage's weights or activations.
    static __device__ __forceinline__ int off(int row, int chunk) {
        return row * LD + (SWZ ? chunk ^ (row / (128 / BK) % CH) : chunk) * 16;
    }
};
// WM 4 with a residual: steps before the last that its L2 prefetch is issued (12 or 24 steps
// of 64 bytes measured no faster on the GB10).
constexpr int64_t RES_PF_STEPS = 4;
}  // namespace a8

// RES: p.res added to every output (the residual add fused into the epilogue).
template <bool RES, class C>
__global__ void __launch_bounds__(256) fp8_gemm8(const legion_fp8_mm8 p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using a8::BN;
    constexpr int WM = C::WM, BK = C::BK, LD = C::LD, STAGES = C::STAGES, CH = C::CH;
    constexpr int WN = 8 / WM, NJ = BN / 8 / WN, TM = C::TM;  // NJ: n8 tiles per warp
    constexpr size_t TSTAGE = C::TSTAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM, n0 = tn * BN;
    const int64_t steps = p.k / BK;

    // One step's copies: TM weight rows and BN tokens of BK bytes, 16 bytes per cp.async.
    auto load = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * TSTAGE;
        unsigned char * xs = ws + TM * LD;
        const int64_t k0 = step * BK;
#pragma unroll
        for (int i = 0; i < (TM + BN) * CH / 256; i++) {
            const int c = threadIdx.x + i * 256;
            if (i < TM * CH / 256) {
                const int r = c / CH, ch = c % CH;
                const int64_t row = m0 + r;
                cp_async16(ws + C::off(r, ch), p.w + (row < p.m ? row : 0) * p.k + k0 + ch * 16, row < p.m);
            } else {
                const int t = (c - TM * CH) / CH, ch = c % CH;
                const int64_t tok = n0 + t;
                cp_async16(xs + C::off(t, ch), p.x + (tok < p.n ? tok : 0) * p.k + k0 + ch * 16, tok < p.n);
            }
        }
    };

    float acc[4][NJ][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < steps) {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int64_t step = 0; step < steps; step++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();
        // Refill the stage the previous step used.
        const int64_t next = step + STAGES - 1;
        if (next < steps) {
            load(next, (int) (next % STAGES));
        }
        asm volatile("cp.async.commit_group;\n" ::);
#if __CUDA_ARCH__ >= 900
        if constexpr (RES && WM != 2) {
            // The tile's residual (a token's TM rows are one 16-byte aligned run when m0 + TM
            // <= m) into L2 a few steps before the epilogue, which then loads it a row group
            // at a time from L2 instead of memory.
            constexpr int64_t PF = a8::RES_PF_STEPS;
            if (step == (steps > PF ? steps - PF : 0) && threadIdx.x < BN && n0 + threadIdx.x < p.n &&
                m0 + TM <= p.m) {
                const float * a = p.res + (n0 + threadIdx.x) * p.m + m0;
                if (((uintptr_t) a & 15) == 0) {
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a), "r"((uint32_t) (TM * 4)));
                }
            }
        }
#endif
        const unsigned char * ws = smem + (step % STAGES) * TSTAGE;
        const unsigned char * xs = ws + TM * LD;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            uint32_t a[4][4], b[NJ][2];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned char * src = ws + C::off(wm * 64 + i * 16 + lane % 16, kk / 16 + lane / 16);
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                    : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < NJ / 2; j++) {
                const int tok = wn * NJ * 8 + j * 16 + lane % 8 + (lane / 16) * 8;
                const unsigned char * src = xs + C::off(tok, kk / 16 + (lane / 8) % 2);
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b[2 * j][0]), "=r"(b[2 * j][1]), "=r"(b[2 * j + 1][0]), "=r"(b[2 * j + 1][1])
                    : "r"(addr));
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
                    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const float s = a8_static(p);
    // RES: the residual loaded before the first store (dst may alias it as far as the compiler
    // knows: loads between the stores would each wait out the memory latency). WM 2: the whole
    // tile's at once; WM 4: a row group's (16 rows) before its stores, which keeps the 128
    // accumulators plus 32 residuals in registers.
    float rv[4][NJ][4];
    auto load_res = [&](int i) {
        const int64_t r0 = m0 + wm * 64 + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const int64_t t0 = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
                rv[i][j][e] = row < p.m && tok < p.n ? p.res[tok * p.m + row] : 0.0f;
            }
        }
    };
    if constexpr (RES && WM == 2) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            load_res(i);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if constexpr (RES && WM != 2) {
            load_res(i);
        }
        const int64_t r0 = m0 + wm * 64 + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            const int64_t t0 = n0 + wn * NJ * 8 + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
                if (row < p.m && tok < p.n) {
                    const float v = acc[i][j][e] * a8_scale(p, s, row, tok);
                    p.dst[tok * p.m + row] = RES ? __fadd_rn(v, rv[i][j][e]) : v;
                }
            }
        }
    }
#endif
}

// Few tokens past the GEMV (9..64): memory-bound, so a CTA is W warps of 16 weight rows each
// over all of its BN tokens, so there are enough CTAs (weight streams) to keep the bus busy; K
// steps of 128 bytes through four cp.async stages (three where four would not fit in the 99 KiB
// of shared memory a block gets on sm_86/89/12x), mma.sync m16n8k32 E4M3 as above.
namespace a8s {
constexpr int BK = 128, LD = BK + 16;  // LD in bytes: 144
template <int BN, int W>
__host__ __device__ constexpr int stages() {
    return (size_t) 4 * (16 * W + BN) * LD <= 96 * 1024 ? 4 : 3;
}
template <int BN, int W>
__host__ __device__ constexpr size_t smem() {
    return (size_t) stages<BN, W>() * (16 * W + BN) * LD;
}
}  // namespace a8s

template <int BN, int W>
__global__ void __launch_bounds__(32 * W) fp8_gemm8_skinny(const legion_fp8_mm8 p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    using namespace a8s;
    constexpr int BM = 16 * W, T = 32 * W, NT = BN / 8, STAGES = stages<BN, W>();
    constexpr size_t STAGE = (size_t) (BM + BN) * LD;
    constexpr int WCH = BM * BK / 16 / T, XCH = (BN * BK / 16 + T - 1) / T;
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t m0 = (int64_t) blockIdx.x * BM, n0 = (int64_t) blockIdx.y * BN;
    const int64_t steps = p.k / BK;

    auto load = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * STAGE;
        unsigned char * xs = ws + BM * LD;
        const int64_t k0 = step * BK;
#pragma unroll
        for (int i = 0; i < WCH; i++) {
            const int c = threadIdx.x + i * T;
            const int r = c / (BK / 16), col = c % (BK / 16) * 16;
            const int64_t row = m0 + r;
            cp_async16(ws + r * LD + col, p.w + (row < p.m ? row : 0) * p.k + k0 + col, row < p.m);
        }
#pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int c = threadIdx.x + i * T;
            if (c < BN * BK / 16) {
                const int t = c / (BK / 16), col = c % (BK / 16) * 16;
                const int64_t tok = n0 + t;
                cp_async16(xs + t * LD + col, p.x + (tok < p.n ? tok : 0) * p.k + k0 + col, tok < p.n);
            }
        }
    };

    // GB10's LPDDR5X streams well only in long runs of each row (~1 KiB, see nvfp4_mm/), so
    // each row's next PF_STEPS steps are prefetched into L2 in one bulk request, PF_STEPS ahead
    // of the copies (rows 16-byte aligned only).
    constexpr int PF_STEPS = 8;
    auto prefetch = [&](int64_t step) {
#if __CUDA_ARCH__ >= 900
        if (threadIdx.x < BM && step < steps && m0 + threadIdx.x < p.m && p.k % 16 == 0) {
            const int64_t bytes = min((int64_t) PF_STEPS, steps - step) * BK;
            const unsigned char * a = p.w + (m0 + threadIdx.x) * p.k + step * BK;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a), "r"((uint32_t) bytes));
        }
#endif
    };
    prefetch(0);
    prefetch(PF_STEPS);

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < steps) {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int64_t step = 0; step < steps; step++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();
        const int64_t next = step + STAGES - 1;
        if (next < steps) {
            load(next, (int) (next % STAGES));
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (step % PF_STEPS == 0 && step > 0) {
            prefetch(step + PF_STEPS);
        }
        const unsigned char * ws = smem + (step % STAGES) * STAGE;
        const unsigned char * xs = ws + BM * LD;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            uint32_t a[4];
            {
                const unsigned char * src = ws + (warp * 16 + lane % 16) * LD + kk + (lane / 16) * 16;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                    : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                uint32_t b[2];
                const unsigned char * src = xs + (j * 8 + lane % 8) * LD + kk + ((lane / 8) % 2) * 16;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b[0]), "=r"(b[1])
                    : "r"(addr));
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const float s = a8_static(p);
    const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int64_t t0 = n0 + j * 8 + (lane % 4) * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
            if (row < p.m && tok < p.n) {
                p.dst[tok * p.m + row] = acc[j][e] * a8_scale(p, s, row, tok);
            }
        }
    }
#endif
}
