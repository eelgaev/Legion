// head_mm.cu, sm_80 and newer: the GEMM of the head op (cp.async staging, ldmatrix, m16n8k16
// BF16 products with F32 sums), the soft-cap, the logits and each CTA's argmax partials.
// - one CTA of 8 warps per ROWS weight rows and 16 * NT columns (NT 4, 6 or 8): warp w takes
//   rows (w % 4) * ROWS / 4 and columns (w / 4) * 8 * NT;
// - the weights' rows and x's BF16 columns go through STAGES shared-memory stages of BK k
//   values (128-byte rows, 16-byte chunks swizzled by hm_swz); ROWS, BK, STAGES: HM_*;
// - every output is one chain of m16n8k16 over k = 0, 16, 32, ... from a zero sum.
//
// Not a standalone header: head_mm.cu includes it inside its anonymous namespace, after
// legion_head_mm, the HM_* constants, hm_smem, hm_swz and hm_take. The kernel body is built
// only where __CUDA_ARCH__ >= 800 and traps elsewhere; head_mm.rs checks at run time that this
// binary has it (cudaFuncGetAttributes' binaryVersion).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void hm_cp16(void * smem, const void * g) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g));
}

__device__ __forceinline__ void hm_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void hm_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void hm_ldsm_x4(uint32_t (&r)[4], const void * p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void hm_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

// Grid m / HM_ROWS, HM_THREADS threads, hm_smem(NT) bytes of dynamic shared memory.
template <int NT>
__global__ void __launch_bounds__(HM_THREADS, 1) head_mm_gemm(const legion_head_mm p) {
    static_assert(NT % 2 == 0, "column tiles in pairs (ldmatrix x4)");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int ROWS = HM_ROWS;              // rows of the CTA
    constexpr int BK = HM_BK;                  // k values a stage
    constexpr int STAGES = HM_STAGES;
    constexpr int CPR = BK / 8;                // 16-byte chunks a row (8: hm_swz's rows)
    constexpr int MI = ROWS / 64;              // 16-row tiles of a warp (ROWS / 4 rows)
    constexpr int NC = 16 * NT;                // columns of the CTA
    constexpr int STAGE = (ROWS + NC) * CPR;   // 16-byte chunks a stage
    static_assert(ROWS % 64 == 0 && (ROWS * CPR) % HM_THREADS == 0, "tile");
    extern __shared__ __align__(16) uint4 hm_sm[];
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int wm = warp % 4, wn = warp / 4;
    const int64_t row0 = (int64_t) blockIdx.x * ROWS;
    const int nk = (int) (p.k / BK);
    const __nv_bfloat16 * wg = p.w + row0 * p.w_ld;

    auto load = [&](int stage, int kt) {
        uint4 * ws = hm_sm + stage * STAGE;
        uint4 * xs = ws + ROWS * CPR;
        const int64_t k0 = (int64_t) kt * BK;
#pragma unroll
        for (int i = 0; i < ROWS * CPR / HM_THREADS; i++) {
            const int q = t + i * HM_THREADS, r = q / CPR, c = q % CPR;
            hm_cp16(ws + hm_swz(r, c), wg + r * p.w_ld + k0 + c * 8);
        }
#pragma unroll
        for (int i = 0; i < (NC * CPR + HM_THREADS - 1) / HM_THREADS; i++) {
            const int q = t + i * HM_THREADS, r = q / CPR, c = q % CPR;
            if (q < NC * CPR) {
                hm_cp16(xs + hm_swz(r, c), p.xb + r * p.k + k0 + c * 8);
            }
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

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) {
            load(s, s);
        }
        hm_commit();
    }
    for (int kt = 0; kt < nk; kt++) {
        hm_wait<STAGES - 2>();
        __syncthreads();
        // The stage read in the step before is free for the step STAGES - 1 ahead.
        if (kt + STAGES - 1 < nk) {
            load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        }
        hm_commit();
        const uint4 * ws = hm_sm + (kt % STAGES) * STAGE;
        const uint4 * xs = ws + ROWS * CPR;
        // Per output the k steps of 16 in order: one chain from 0.
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            uint32_t a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const int r = wm * (ROWS / 4) + i * 16 + lane % 16;
                hm_ldsm_x4(a[i], ws + hm_swz(r, kk * 2 + lane / 16));
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                uint32_t b[4];
                const int r = wn * 8 * NT + j * 8 + (lane / 16) * 8 + lane % 8;
                hm_ldsm_x4(b, xs + hm_swz(r, kk * 2 + (lane / 8) % 2));
#pragma unroll
                for (int i = 0; i < MI; i++) {
                    hm_mma(acc[i][j], a[i], b[0], b[1]);
                    hm_mma(acc[i][j + 1], a[i], b[2], b[3]);
                }
            }
        }
    }
    hm_wait<0>();
    __syncthreads();

    // Soft-cap, logits, and each column's best of the warp's ROWS / 4 rows.
    float * red_v = (float *) hm_sm;                   // [4][NC]
    uint32_t * red_r = (uint32_t *) (red_v + 4 * NC);  // [4][NC]
    const int g = lane / 4, tq = lane % 4;
#pragma unroll
    for (int j = 0; j < NT; j++) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int col = wn * 8 * NT + j * 8 + 2 * tq + e;
            float bv = -FLT_MAX;
            uint32_t br = PK_NONE;
#pragma unroll
            for (int i = 0; i < MI; i++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int64_t r = row0 + wm * (ROWS / 4) + i * 16 + h * 8 + g;
                    float v = acc[i][j][h * 2 + e];
                    if (p.cap > 0.0f) {
                        v = tanhf(p.scale * v) * p.cap;
                    }
                    if (col < p.n) {
                        p.dst[col * p.dst_ld + r] = v;
                    }
                    hm_take(v, pk_rank(r), bv, br);
                }
            }
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                const float v = __shfl_xor_sync(0xffffffff, bv, o);
                const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
                if (pk_better(v, r, bv, br)) {
                    bv = v;
                    br = r;
                }
            }
            if (g == 0) {
                red_v[wm * NC + col] = bv;
                red_r[wm * NC + col] = br;
            }
        }
    }
    __syncthreads();
    if (t < p.n) {
        float bv = red_v[t];
        uint32_t br = red_r[t];
#pragma unroll
        for (int q = 1; q < 4; q++) {
            if (pk_better(red_v[q * NC + t], red_r[q * NC + t], bv, br)) {
                bv = red_v[q * NC + t];
                br = red_r[q * NC + t];
            }
        }
        p.part[(int64_t) t * p.parts + blockIdx.x] = make_uint2(__float_as_uint(bv), br);
    }
#else
    __trap();
#endif
}
