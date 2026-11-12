// hc_down.cu, sm_80 and newer: the mixer's down GEMM with its B operand rebuilt from the BF16
// streams (cp.async staging, ldmatrix, m16n8k16 BF16 products with F32 sums).
// - one CTA of 16 warps per 32 * NT tokens and all HD_M rows: warp w takes rows
//   (w % 4) * HD_M / 4 and tokens (w / 4) * 8 * NT;
// - the weights' rows, the streams' token rows (rows past n repeat the last token's, never
//   stored) and gamma go through HD_STAGES shared-memory stages of HD_BK k values (64-byte
//   rows, 16-byte chunks swizzled by hd_swz);
// - B fragments rebuilt after ldmatrix as hc_post_norm_b16_kernel makes xnb:
//   __floats2bfloat162_rn(__fmul_rn(__fmul_rn(v, scale), g)) per value;
// - every output is one chain of m16n8k16 over k = 0, 16, 32, ... from a zero sum.
//
// Not a standalone header: hc_down.cu includes it inside its anonymous namespace, after
// legion_hc_down, the HD_* constants, hd_smem and hd_swz. The kernel body is built only where
// __CUDA_ARCH__ >= 800 and traps elsewhere; hc_down.rs checks at run time that this binary has
// it (cudaFuncGetAttributes' binaryVersion).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void hd_cp16(void * smem, const void * g) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g));
}

__device__ __forceinline__ void hd_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void hd_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void hd_ldsm_x4(uint32_t (&r)[4], const void * p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void hd_ldsm_x2(uint32_t (&r)[2], const void * p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void hd_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two BF16 stream values (the lower k in the low half) rebuilt as hc_post_norm_b16_kernel
// writes xnb: each widened, times the stream's norm scale s, times gamma, both multiplies
// rounded, then rounded to BF16 (nearest even).
__device__ __forceinline__ uint32_t hd_rebuild(uint32_t v, float s, float2 g) {
    const float lo = __uint_as_float(v << 16);
    const float hi = __uint_as_float(v & 0xffff0000u);
    const __nv_bfloat162 r = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(lo, s), g.x), __fmul_rn(__fmul_rn(hi, s), g.y));
    return *reinterpret_cast<const uint32_t *>(&r);
}
#endif

// Grid ceil(n / (32 * NT)), HD_THREADS threads, hd_smem(NT) bytes of dynamic shared memory.
// m == HD_M, k a multiple of HD_BK, e a multiple of HD_BK dividing k.
template <int NT>
__global__ void __launch_bounds__(HD_THREADS, 1) hc_down_gemm(const legion_hc_down p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int CPR = HD_BK / 8;               // 16-byte chunks a row (4: hd_swz's rows)
    constexpr int MI = HD_M / 64;                // 16-row tiles of a warp (HD_M / 4 rows)
    constexpr int NC = 32 * NT;                  // tokens of the CTA
    constexpr int A_CH = HD_M * CPR;             // the weights' chunks a stage
    constexpr int B_CH = NC * CPR;               // the streams' chunks a stage
    constexpr int G_CH = HD_BK / 4;              // gamma's chunks a stage
    constexpr int STAGE = A_CH + B_CH + G_CH;    // 16-byte chunks a stage
    static_assert(HD_M % 64 == 0 && CPR == 4, "tile");
    extern __shared__ __align__(16) uint4 hd_sm[];
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int wm = warp % 4, wn = warp / 4;
    const int g = lane / 4, q = lane % 4;
    const int n0 = blockIdx.x * NC;
    const int nk = p.k / HD_BK;
    const int steps = p.e / HD_BK;  // stages a stream

    auto load = [&](int stage, int kt) {
        uint4 * ws = hd_sm + stage * STAGE;
        uint4 * xs = ws + A_CH;
        uint4 * gs = xs + B_CH;
        const int64_t k0 = (int64_t) kt * HD_BK;
#pragma unroll
        for (int i = 0; i < (A_CH + HD_THREADS - 1) / HD_THREADS; i++) {
            const int u = t + i * HD_THREADS;
            if (u < A_CH) {
                const int r = u / CPR, c = u % CPR;
                hd_cp16(ws + hd_swz(r, c), p.w + r * p.w_ld + k0 + c * 8);
            }
        }
#pragma unroll
        for (int i = 0; i < (B_CH + HD_THREADS - 1) / HD_THREADS; i++) {
            const int u = t + i * HD_THREADS;
            if (u < B_CH) {
                const int r = u / CPR, c = u % CPR;
                const int64_t tok = min(n0 + r, p.n - 1);
                hd_cp16(xs + hd_swz(r, c), p.x + tok * p.x_ld + k0 + c * 8);
            }
        }
        if (t < G_CH) {
            hd_cp16(gs + t, p.g + k0 + t * 4);
        }
    };

    float acc[MI][NT][4];
#pragma unroll
    for (int i = 0; i < MI; i++) {
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[i][j][l] = 0.0f;
            }
        }
    }
    // The norm scale of each of the lane's tokens (token g of every n8 tile) for the stream the
    // stage's k values belong to.
    float scl[NT];

#pragma unroll
    for (int s = 0; s < HD_STAGES - 1; s++) {
        if (s < nk) {
            load(s, s);
        }
        hd_commit();
    }
    for (int kt = 0; kt < nk; kt++) {
        if (kt % steps == 0) {
            const int h = kt / steps;
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const int64_t tok = min(n0 + wn * 8 * NT + j * 8 + g, p.n - 1);
                scl[j] = p.sc[tok * p.sc_ld + h];
            }
        }
        hd_wait<HD_STAGES - 2>();
        __syncthreads();
        // The stage read in the step before is free for the step HD_STAGES - 1 ahead.
        if (kt + HD_STAGES - 1 < nk) {
            load((kt + HD_STAGES - 1) % HD_STAGES, kt + HD_STAGES - 1);
        }
        hd_commit();
        const uint4 * ws = hd_sm + (kt % HD_STAGES) * STAGE;
        const uint4 * xs = ws + A_CH;
        const float2 * gs = reinterpret_cast<const float2 *>(xs + B_CH);
#pragma unroll
        for (int kk = 0; kk < HD_BK / 16; kk++) {
            // gamma at the lane's k values: kk * 16 + 2q, + 1 (b[.][0]) and + 8, + 9 (b[.][1])
            const float2 g0 = gs[kk * 8 + q];
            const float2 g1 = gs[kk * 8 + 4 + q];
            uint32_t b[NT][2];
#pragma unroll
            for (int j = 0; j + 1 < NT; j += 2) {
                uint32_t r4[4];
                const int r = wn * 8 * NT + j * 8 + (lane / 16) * 8 + lane % 8;
                hd_ldsm_x4(r4, xs + hd_swz(r, kk * 2 + (lane / 8) % 2));
                b[j][0] = r4[0];
                b[j][1] = r4[1];
                b[j + 1][0] = r4[2];
                b[j + 1][1] = r4[3];
            }
            if constexpr (NT % 2 == 1) {
                uint32_t r2[2];
                const int r = wn * 8 * NT + (NT - 1) * 8 + lane % 8;
                hd_ldsm_x2(r2, xs + hd_swz(r, kk * 2 + (lane / 8) % 2));
                b[NT - 1][0] = r2[0];
                b[NT - 1][1] = r2[1];
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                b[j][0] = hd_rebuild(b[j][0], scl[j], g0);
                b[j][1] = hd_rebuild(b[j][1], scl[j], g1);
            }
            // Per output the k steps of 16 in order: one chain from 0.
#pragma unroll
            for (int i = 0; i < MI; i++) {
                uint32_t a[4];
                const int r = wm * (HD_M / 4) + i * 16 + lane % 16;
                hd_ldsm_x4(a, ws + hd_swz(r, kk * 2 + lane / 16));
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    hd_mma(acc[i][j], a, b[j][0], b[j][1]);
                }
            }
        }
    }
    hd_wait<0>();

    // C fragments: (row g, tokens 2q and 2q + 1) and (row g + 8, the same tokens).
#pragma unroll
    for (int i = 0; i < MI; i++) {
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int row = wm * (HD_M / 4) + i * 16 + g + (l >> 1) * 8;
                const int tok = n0 + wn * 8 * NT + j * 8 + 2 * q + (l & 1);
                if (tok < p.n) {
                    p.dst[(int64_t) tok * p.dst_ld + row] = acc[i][j][l];
                }
            }
        }
    }
#else
    __trap();
#endif
}
