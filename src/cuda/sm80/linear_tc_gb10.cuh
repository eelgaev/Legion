// glm5next.cu, sm_80 and newer: linear_tc_gb, linear_tc's F16 GEMM with other tile shapes (#862),
// for GLM's dense prompt products on the GB10 (Maya-S's Q6_K weights after wf16, about a quarter
// of its prefill).
//
// Not a standalone header: glm5next.cu includes it inside its anonymous namespace, after
// linear_tc (it uses Lin, lin_scale, tc_cp16, tc_h2 and T_F16).
//
// The same bits as linear_tc: every output element is the F32 sum, in K order, of m16n8k16
// F16 x F16 -> F32 products from zero over 16-deep pieces of the same F16 operands (the weights
// raw F16, or BF16 rounded to F16 by tc_h2; the activations F16 rows (XH) or F32 rounded by
// __floats2half2_rn), then times lin_scale. An mma's output element depends only on its own
// row of A and column of B, so the tile shape, the warp layout, the K step and the pipeline
// depth change the speed only.
//
// linear_tc runs 128 x 128 tiles with K steps of 32: on the GB10 it reaches ~50 TFLOPS of the
// ~118 TFLOPS mma.sync peak. The shapes here: BM rows x BN tokens a block, warps of WM x WN,
// K steps of BK through an S-stage cp.async ring, MINB blocks per SM.

template <bool XH, int BM, int BN, int BK, int S, int WM, int WN, int MINB>
struct TcGb {
    static constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
    static constexpr int MI = WM / 16, NI = WN / 8;
    static constexpr int LW = BK + 8;               // halves per weight row in shared memory
    static constexpr int LX = BK + 8;               // halves (XH) or floats per token row
    static constexpr int WST = BM * LW * 2;         // bytes of a weight stage
    static constexpr int XST = BN * LX * (XH ? 2 : 4);
    static constexpr int SMEM = S * (WST + XST);
    static constexpr int WP = BM * BK / 8;          // 16-byte pieces of a weight stage
    static constexpr int XP = BN * BK / (XH ? 8 : 4);
    static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0, "whole fragments");
    static_assert((WP % NT == 0 || WP < NT) && (XP % NT == 0 || XP < NT), "whole copies per thread");
    static_assert(S >= 2, "a stage in flight while one is read");
};

template <bool XH, int BM, int BN, int BK, int S, int WM, int WN, int MINB, bool OH = false>
__global__ void __launch_bounds__(TcGb<XH, BM, BN, BK, S, WM, WN, MINB>::NT, MINB) linear_tc_gb(const Lin p) {
    pdl_wait();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using G = TcGb<XH, BM, BN, BK, S, WM, WN, MINB>;
    constexpr int MI = G::MI, NI = G::NI, LW = G::LW, LX = G::LX, NT = G::NT;
    extern __shared__ __align__(16) char tg_sm[];
    uint16_t * wsm = (uint16_t *) tg_sm;           // [S][BM][LW]
    char * xsm = tg_sm + S * G::WST;               // [S][BN][LX] halves or floats
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    // linear_tc's order: groups of TCG row tiles, a group's token tiles in turn (grid.x: row
    // tiles * token tiles)
    const int tm = (int) ((p.m + BM - 1) / BM), tn = (int) (gridDim.x / tm), per = TCG * tn;
    const int first = (int) blockIdx.x / per * TCG, gm = min(tm - first, TCG), in = (int) blockIdx.x % per;
    const int64_t m0 = (int64_t) (first + in % gm) * BM, n0 = (int64_t) (in / gm) * BN, i = blockIdx.z;
    const int wm = warp / G::NWN, wn = warp % G::NWN;
    const bool f16 = p.wtype == T_F16;
    const char * wb = p.w + i * p.w_nb2;
    const char * xb = p.x + i * p.x_nbi;
    auto load = [&](int64_t k0, int s) {
#pragma unroll
        for (int j = 0; j < (G::WP + NT - 1) / NT; j++) {
            const int c = tid + NT * j, row = c / (BK / 8), q = c % (BK / 8);
            if (G::WP < NT && c >= G::WP) {
                break;
            }
            const bool ok = m0 + row < p.m;
            tc_cp16(wsm + (s * BM + row) * LW + 8 * q, wb + (ok ? m0 + row : 0) * p.w_nb1 + (k0 + 8 * q) * 2, ok);
        }
        constexpr int PR = XH ? BK / 8 : BK / 4;  // pieces per token row
#pragma unroll
        for (int j = 0; j < (G::XP + NT - 1) / NT; j++) {
            const int c = tid + NT * j, tok = c / PR, q = c % PR;
            if (G::XP < NT && c >= G::XP) {
                break;
            }
            const bool ok = n0 + tok < p.n;
            const char * src = xb + (ok ? n0 + tok : 0) * p.x_nbt + (k0 + (XH ? 8 : 4) * q) * (XH ? 2 : 4);
            tc_cp16(xsm + (s * BN + tok) * LX * (XH ? 2 : 4) + 16 * q, src, ok);
        }
    };
    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++) {
#pragma unroll
        for (int b = 0; b < NI; b++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[a][b][e] = 0.0f;
            }
        }
    }
    const int nk = (int) (p.k / BK);
#pragma unroll
    for (int s = 0; s < S - 1; s++) {
        if (s < nk) {
            load((int64_t) s * BK, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kt = 0; kt < nk; kt++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
        __syncthreads();
        if (kt + S - 1 < nk) {
            load((int64_t) (kt + S - 1) * BK, (kt + S - 1) % S);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        const int s = kt % S;
        const uint16_t * ws = wsm + s * BM * LW;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t fa[MI][4], fb[NI][2];
#pragma unroll
            for (int a = 0; a < MI; a++) {
                const uint16_t * pa = ws + (WM * wm + 16 * a + lane % 16) * LW + kk + 8 * (lane / 16);
                const uint32_t sa = (uint32_t) __cvta_generic_to_shared(pa);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(fa[a][0]), "=r"(fa[a][1]), "=r"(fa[a][2]), "=r"(fa[a][3])
                    : "r"(sa));
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    fa[a][j] = tc_h2(fa[a][j], f16);
                }
            }
            if constexpr (XH) {
                const uint16_t * xs = (const uint16_t *) xsm + s * BN * LX;
#pragma unroll
                for (int b = 0; b < NI; b += 2) {
                    const uint16_t * px =
                        xs + (WN * wn + 8 * b + 8 * (lane >> 4) + (lane & 7)) * LX + kk + 8 * ((lane >> 3) & 1);
                    const uint32_t sx = (uint32_t) __cvta_generic_to_shared(px);
                    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(fb[b][0]), "=r"(fb[b][1]), "=r"(fb[b + 1][0]), "=r"(fb[b + 1][1])
                        : "r"(sx));
                }
            } else {
                const float * xs = (const float *) xsm + s * BN * LX;
#pragma unroll
                for (int b = 0; b < NI; b++) {
                    const float * px = xs + (WN * wn + 8 * b + lane / 4) * LX + kk + 2 * (lane % 4);
                    const float2 x0 = *(const float2 *) px, x1 = *(const float2 *) (px + 8);
                    const __half2 h0 = __floats2half2_rn(x0.x, x0.y), h1 = __floats2half2_rn(x1.x, x1.y);
                    fb[b][0] = *(const uint32_t *) &h0;
                    fb[b][1] = *(const uint32_t *) &h1;
                }
            }
#pragma unroll
            for (int a = 0; a < MI; a++) {
#pragma unroll
                for (int b = 0; b < NI; b++) {
                    float d[4];
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                        "{%8, %9}, {%10, %10, %10, %10};\n"
                        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                        : "r"(fa[a][0]), "r"(fa[a][1]), "r"(fa[a][2]), "r"(fa[a][3]), "r"(fb[b][0]), "r"(fb[b][1]),
                          "f"(0.0f));
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        acc[a][b][e] += d[e];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int a = 0; a < MI; a++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t row = m0 + WM * wm + 16 * a + lane / 4 + 8 * h;
            if (row >= p.m) {
                continue;
            }
            const float sc = lin_scale(p, row);
#pragma unroll
            for (int b = 0; b < NI; b++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int64_t tok = n0 + WN * wn + 8 * b + 2 * (lane % 4) + e;
                    if (tok < p.n) {
                        if constexpr (OH) {
                            ((__half *) p.dst)[(tok * p.h + i) * p.m + row] =
                                __float2half_rn(acc[a][b][2 * h + e] * sc);
                        } else {
                            p.dst[(tok * p.h + i) * p.m + row] = acc[a][b][2 * h + e] * sc;
                        }
                    }
                }
            }
        }
    }
#endif
}

// Launches linear_tc_gb<...> for `p` (its K a multiple of BK); false when the shape does not fit
// or the device refuses the shared memory.
template <bool XH, int BM, int BN, int BK, int S, int WM, int WN, int MINB, bool OH = false>
bool tc_gb_launch(const Lin & p, cudaStream_t st) {
    using G = TcGb<XH, BM, BN, BK, S, WM, WN, MINB>;
    static std::atomic<int8_t> smem[GGML_CUDA_MAX_DEVICES];
    if (p.k % BK != 0 ||
        !legion_device::smem_opt_in(smem, linear_tc_gb<XH, BM, BN, BK, S, WM, WN, MINB, OH>, G::SMEM)) {
        return false;
    }
    const int64_t tiles = ((p.m + BM - 1) / BM) * ((p.n + BN - 1) / BN);
    if (tiles > INT32_MAX) {
        return false;
    }
    const dim3 grid((unsigned) tiles, 1, (unsigned) p.h);
    launch_pdl(linear_tc_gb<XH, BM, BN, BK, S, WM, WN, MINB, OH>, grid, G::NT, G::SMEM, st, p);
    return true;
}

// The shapes measured by `legion lin-bench` (kernel 2 + index); 0 when there is none.
constexpr int TC_GB_SHAPES = 6;

// Launches shape `v` (0 .. TC_GB_SHAPES - 1) of linear_tc_gb with F16 (`xh`) or F32 activations.
// F16 outputs (`oh`, linear_tc's OH): shape 2 with F16 activations only (attn_q_b's, #862).
bool tc_gb_shape(int v, bool xh, const Lin & p, cudaStream_t st, bool oh = false) {
    if (oh) {
        return xh && v == 2 && tc_gb_launch<true, 128, 256, 32, 3, 64, 64, 1, true>(p, st);
    }
    if (xh) {
        switch (v) {
            case 0: return tc_gb_launch<true, 64, 128, 32, 3, 32, 32, 2>(p, st);
            case 1: return tc_gb_launch<true, 256, 128, 32, 3, 64, 64, 1>(p, st);
            case 2: return tc_gb_launch<true, 128, 256, 32, 3, 64, 64, 1>(p, st);
            case 3: return tc_gb_launch<true, 128, 128, 32, 4, 64, 32, 1>(p, st);
            case 4: return tc_gb_launch<true, 128, 128, 32, 2, 64, 32, 2>(p, st);
            case 5: return tc_gb_launch<true, 128, 128, 32, 3, 64, 32, 1>(p, st);
            default: return false;
        }
    }
    switch (v) {
        case 0: return tc_gb_launch<false, 256, 128, 16, 4, 64, 64, 1>(p, st);
        case 1: return tc_gb_launch<false, 64, 128, 16, 3, 32, 32, 2>(p, st);
        case 2: return tc_gb_launch<false, 128, 128, 32, 2, 64, 32, 1>(p, st);
        case 3: return tc_gb_launch<false, 64, 128, 32, 3, 32, 32, 1>(p, st);
        case 4: return tc_gb_launch<false, 128, 128, 32, 3, 32, 64, 1>(p, st);
        case 5: return tc_gb_launch<false, 128, 128, 32, 3, 64, 32, 1>(p, st);
        default: return false;
    }
}

// Whether the current device is a compute capability 12.x part (the GB10 is 12.1): the GPUs
// linear_tc_gb was measured on.
bool tc_gb_device() {
    return legion_device::compute_capability() / 10 == 12;
}

// The device's choice on a cc 12.x GPU; false leaves the product to linear_tc. `legion lin-bench`
// on the GB10 at Maya-S's shapes, 1024 to 4096 tokens: F16 rows (XH) of 4096+ weight rows on
// 128 x 256 tiles (shape 2) 0.86-0.90x linear_tc's time; 2048 rows (the shared expert's gate /
// up) on 256 x 128 tiles (shape 1) 0.82-0.94x from 2048 tokens, but 1.06x at 1024, where
// linear_tc stays. With F32 rows no shape beat linear_tc, which rounds them in its K loop: the
// lone products' x comes in F16 instead on these GPUs (glm5next::ops::xh_one_takes). With F16
// outputs (`oh`) only shape 2 has a variant; the other products stay on linear_tc.
bool tc_gb_auto(bool xh, const Lin & p, cudaStream_t st, bool oh = false) {
    if (!xh || p.h != 1) {
        return false;
    }
    if (p.m > 2048) {
        return tc_gb_shape(2, true, p, st, oh);
    }
    return !oh && p.n >= 2048 && tc_gb_shape(1, true, p, st);
}
