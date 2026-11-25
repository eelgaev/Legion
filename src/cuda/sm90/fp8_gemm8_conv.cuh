// fp8_mm.cu, sm_90 and newer: a gated DeltaNet's qkv projection with its causal conv, SiLU and
// q / k head norm (conv_silu, conv_silu.cuh) in the GEMM's epilogue (#1183).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm90/fp8_gemm8_tma.cuh (gemm8_tma_sums) and conv_silu.cuh. The kernel body is built only
// where __CUDA_ARCH__ >= 900 and traps elsewhere.
//
// In a prompt step the projection's F32 output (10,240 rows a token on Qwen3.8-Flash-Next: 41 KB)
// went to memory only for conv_silu to read it back. Here each CTA computes the wide tile's
// sums (fp8_gemm8_tma's K loop, so every sum is that GEMM's: the same MMAs in the same k order,
// and a token at the same column of its n8 MMA tile), scales them as gemm8_epilogue_stream does
// (the projection's values, bit for bit), and makes conv_silu's outputs from them in registers.
// An output needs its token's last three inputs, so tiles step by CONV_STRIDE = 120 tokens and
// each but the first starts CONV_HALO = 8 tokens early (one n8 tile, so tokens keep their column
// in it): those 8 are recomputed, inputs only. The first tile takes the inputs before the step
// from the conv state. The tile holding the step's last three tokens writes them as the next
// state (conv_silu's hist). The conv's dot, SiLU and the norm's sums run as conv_silu_kernel
// runs them (conv_math's expressions, and the norm's sums added in its warp_sum order), so
// every output and the new state are conv_silu's bits.

namespace a8c {
constexpr int CONV_STRIDE = 120;     // outputs a tile makes
constexpr int CONV_HALO = 8;         // tokens a tile but the first starts early (inputs only)
constexpr int CONV_D = 4;            // the conv's taps
}  // namespace a8c

// Shared memory past the stages' (free after the K loop) the epilogue uses: each wn 0 warp's last
// n8 tile (the inputs before its wn 1 partner's first), 4 x 32 lanes x 8 rows x 2 tokens, then
// the norm's per-warp sums, 4 row warps x 2 halves x 128 tokens.
constexpr int CONV_PREV_FLOATS = 4 * 32 * 8 * 2;
constexpr int CONV_SUM_FLOATS = 4 * 2 * 128;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// The epilogue: acc holds the tile's sums (TM 256 rows from m0, a8::BN tokens from n0, 8 warps
// of 64 x 64: wm = warp / 2, wn = warp % 2); outputs for tile tokens lo.. that exist. TST
// (#1183): the outputs go out by TMA stores into ymap (cv.out's tensor map, gemm8_store_tma).
template <class C, bool TST>
__device__ __forceinline__ void gemm8_conv_epilogue(const legion_fp8_mm8 & p, const legion_conv_silu & cv,
                                                    int64_t m0, int64_t n0, int lo, bool last, int wm, int wn,
                                                    int lane, float (&acc)[4][8][4], float * smem,
                                                    const CUtensorMap * ymap) {
    static_assert(C::WM == 4 && C::WN == 2 && a8::BN == 128, "the wide tile");
    using conv_math::MAX_D;
    constexpr int D = a8c::CONV_D;
    const unsigned FULL = 0xffffffffu;
    const int g = lane / 4, t = lane % 4;
    const int rbase = wm * 64 + g;  // row of i 0, h 0 (from m0)
    const int ubase = wn * 64 + 2 * t;  // tile token of j 0, e 0

    // gemm8_epilogue_stream's values (the projection's output, bit for bit); 0 past the step.
    {
        const float s = a8_static(p);
        const bool uni = p.scale_stride == 0 && p.wscale_stride == 0;
        const bool rowsc = !uni && p.wscale != nullptr;
        float wr[4][2];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = rbase + i * 16 + h * 8;
                wr[i][h] = rowsc ? p.wscale[(m0 + row) * p.wscale_stride] : 0.0f;
            }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                const int64_t tok = n0 + ubase + j * 8 + e2;
                const bool ok = tok < p.n;
                const float sc = !uni && ok ? p.scale[tok * p.scale_stride] : s;
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const float f = uni ? s : rowsc ? sc * wr[i][h] : sc;
                        float & v = acc[i][j][2 * h + e2];
                        v = ok ? v * f : 0.0f;
                    }
            }
    }

    // The next state: the step's last D - 1 inputs, from the tile holding them all.
    if (last) {
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                const int64_t tok = n0 + ubase + j * 8 + e2;
                const int64_t jj = tok - (p.n - (D - 1));
                if (jj >= 0 && tok < p.n) {
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int64_t c = m0 + rbase + i * 16 + h * 8;
                            cv.hist[jj + c * (D - 1)] = acc[i][j][2 * h + e2];
                        }
                }
            }
    }

    // The inputs before this warp's first n8 tile: its wn 0 partner's last (through shared
    // memory, the stages being free once every warp is past the K loop), or in the step's first
    // tile, for the wn 0 warps, the conv state (state[k] the input k - 3 tokens before the
    // step). Elsewhere they only feed the halo's outputs, which are not stored.
    float prev[4][2][2];
    __syncthreads();
    float2 * xch = (float2 *) smem;
    if (wn == 0) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                xch[(wm * 8 + i * 2 + h) * 32 + lane] = make_float2(acc[i][7][2 * h], acc[i][7][2 * h + 1]);
            }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            float a = 0.0f, b = 0.0f;
            if (wn == 1) {
                const float2 v = xch[(wm * 8 + i * 2 + h) * 32 + lane];
                a = v.x;
                b = v.y;
            } else if (n0 == 0) {
                // Tokens 2t - 8 and 2t - 7: lane 2's second is state[0], lane 3's state[1], [2].
                const float * st = cv.state + (m0 + rbase + i * 16 + h * 8) * cv.st_ld;
                if (t == 2) {
                    b = st[0];
                } else if (t == 3) {
                    a = st[1];
                    b = st[2];
                }
            }
            prev[i][h][0] = a;
            prev[i][h][1] = b;
        }

    // The conv and SiLU, row by row, n8 tiles last to first in place (an output replaces its
    // token's input once the later tokens' windows have it). The quad's lanes hold tokens 2t,
    // 2t + 1 of an n8 tile: token 2t's window is (2t - 3 .. 2t), from lanes t - 2 and t - 1
    // (of the tile before for lanes 0 and 1).
    const int src1 = (lane & ~3) | ((t + 3) & 3), src2 = (lane & ~3) | ((t + 2) & 3);
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t c = m0 + rbase + i * 16 + h * 8;
            float w[MAX_D];
#pragma unroll
            for (int k = 0; k < MAX_D; k++) {
                w[k] = k < D ? cv.w[k + c * D] : 0.0f;
            }
#pragma unroll
            for (int j = 7; j >= 0; j--) {
                const float pa = j > 0 ? acc[i][j > 0 ? j - 1 : 0][2 * h] : prev[i][h][0];
                const float pb = j > 0 ? acc[i][j > 0 ? j - 1 : 0][2 * h + 1] : prev[i][h][1];
                const float a = acc[i][j][2 * h], b = acc[i][j][2 * h + 1];
                // Lane t gets lane t - 1's pair (lane 0: lane 3's of the tile before) and lane
                // t - 2's second (lanes 0, 1: lanes 2, 3's of the tile before).
                const float r1a = __shfl_sync(FULL, t == 3 ? pa : a, src1);
                const float r1b = __shfl_sync(FULL, t == 3 ? pb : b, src1);
                const float r2 = __shfl_sync(FULL, t >= 2 ? pb : b, src2);
                float win[MAX_D];
#pragma unroll
                for (int k = 0; k < MAX_D; k++) {
                    win[k] = 0.0f;
                }
                win[0] = r2;
                win[1] = r1a;
                win[2] = r1b;
                win[3] = a;
                const float y0 = conv_math::silu_dot(win, w, D);
                win[0] = r1a;
                win[1] = r1b;
                win[2] = a;
                win[3] = b;
                const float y1 = conv_math::silu_dot(win, w, D);
                acc[i][j][2 * h] = y0;
                acc[i][j][2 * h + 1] = y1;
            }
        }

    // The q and k heads' norm (whole tiles of them): conv_silu_kernel's per token sums over a
    // head of 128 rows, warp w of its block summing rows 32w .. 32w + 31 by warp_sum (xor 16,
    // 8, 4, 2, 1 over its lanes, lane = row % 32), then (S0 + S2) + (S1 + S3). Here row
    // 64 (wm % 2) + 16 i + 8 h + g of a head is that block's lane 16 (i % 2) + 8 h + g of warp
    // 2 (wm % 2) + i / 2: the same adds in the same order, i % 2 and h in registers, g over
    // the quads (xor 16, 8, 4), the four warps' sums through shared memory.
    if (m0 < (int64_t) cv.norm_blocks * cv.norm_head) {
        float * sums = (float *) smem + CONV_PREV_FLOATS;  // [wm][half][tile token]
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    const float a0 = conv_math::square(acc[2 * q][j][e2]) + conv_math::square(acc[2 * q + 1][j][e2]);
                    const float a1 =
                        conv_math::square(acc[2 * q][j][2 + e2]) + conv_math::square(acc[2 * q + 1][j][2 + e2]);
                    float sq = a0 + a1;
                    sq += __shfl_xor_sync(FULL, sq, 16, 32);
                    sq += __shfl_xor_sync(FULL, sq, 8, 32);
                    sq += __shfl_xor_sync(FULL, sq, 4, 32);
                    if (g == 0) {
                        sums[(wm * 2 + q) * 128 + ubase + j * 8 + e2] = sq;
                    }
                }
            }
        __syncthreads();
        const int ev = wm & ~1, od = wm | 1;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                const int u = ubase + j * 8 + e2;
                const float s0 = sums[(ev * 2 + 0) * 128 + u], s1 = sums[(ev * 2 + 1) * 128 + u];
                const float s2 = sums[(od * 2 + 0) * 128 + u], s3 = sums[(od * 2 + 1) * 128 + u];
                const float sum = (s0 + s2) + (s1 + s3);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float & y = acc[i][j][2 * h + e2];
                        y = conv_math::normed(y, sum, cv.norm_head, cv.norm_eps, cv.norm_scale);
                    }
            }
    }

    // The outputs of tokens lo.. of the tile that exist, evict-first (read once, by the delta
    // rule): n8 tiles j0 .. j1 - 1 of them.
    auto store = [&](int j0, int j1) {
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                const int u = ubase + j * 8 + e2;
                const int64_t tok = n0 + u;
                if (j >= j0 && j < j1 && u >= lo && tok < p.n) {
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int64_t c = m0 + rbase + i * 16 + h * 8;
                            __stcs(cv.out + c + tok * cv.C, acc[i][j][2 * h + e2]);
                        }
                }
            }
    };
    if constexpr (TST) {
        // Every warp's reads of the exchange and the norm's sums done: the stages' memory takes
        // the boxes. The halo (tile tokens before lo, all in a wn 0 warp's first quarter) is not
        // stored: that quarter's other tokens go out from the fragments, the rest by TMA.
        static_assert(C::WARPS * a8t::ST_WARP_FLOATS * sizeof(float) <= C::SMEM && a8t::ST_TOKS == 16 &&
                          a8c::CONV_HALO <= a8t::ST_TOKS,
                      "TMA store boxes in the stages; the halo in the first quarter");
        __syncthreads();
        const bool skip0 = wn == 0 && lo > 0;
        if (skip0) {
            store(0, 2);
        }
        gemm8_store_tma(ymap, m0 + wm * 64, n0 + wn * 64, p.n, lane, acc,
                        smem + (wm * C::WN + wn) * a8t::ST_WARP_FLOATS, skip0, [](int) {});
    } else {
        store(0, 8);
    }
}
#endif

// The qkv GEMM with conv_silu in its epilogue (see the top): fp8_gemm8_tma's wide tile (8 warps,
// TM 256) with one CTA a tile of a8c::CONV_STRIDE outputs (grid x), from its first token minus
// a8c::CONV_HALO (but the first). One sequence; p.m rows a multiple of TM; the norm's rows whole
// tiles. TST: the outputs by TMA stores (maps.y: cv.out's).
template <class C, bool TST = false>
__global__ void __launch_bounds__(32 * C::WARPS, 1)
    fp8_gemm8_tma_conv(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps, const a8t::Stagger stagger,
                       const legion_conv_silu cv) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    static_assert(C::WARPS == 8 && C::WM == 4, "the wide tile");
    using a8::BN;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM;
    static_assert(NJ == 8, "8 n8 tiles a warp");
    static_assert((CONV_PREV_FLOATS + CONV_SUM_FLOATS) * sizeof(float) <= C::SMEM, "epilogue memory");
    unsigned char * smem = gemm8_tma_smem_base<C>();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM;
    const int64_t n0 = tn == 0 ? 0 : tn * a8c::CONV_STRIDE - a8c::CONV_HALO;
    const int lo = tn == 0 ? 0 : a8c::CONV_HALO;
    const bool last = tn == (int64_t) gridDim.x - 1;
    float acc[4][NJ][4];
    gemm8_tma_sums<false, C>(p, maps, stagger, m0, n0, smem, wm, wn, lane, acc);
    gemm8_conv_epilogue<C, TST>(p, cv, m0, n0, lo, last, wm, wn, lane, acc, (float *) smem, &maps.y);
#else
    __trap();  // launch_gemm8_conv checks the image is sm_90+ first
#endif
}
