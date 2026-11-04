// fp8_mm.cu, sm_80 and newer: the W8A16 GEMMs on mma.sync m16n8k16 BF16 -> F32 (fp8_gemm,
// fp8_gemm_big and its tile shape), and the cp.async helper sm89/fp8_mm.cuh uses too.
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after the
// legion_fp8_mm declaration and the E4M3 helpers it uses. The kernel bodies are built only where
// __CUDA_ARCH__ >= 800 and are empty elsewhere, so the file compiles for every target
// architecture; bf16_mma() in fp8_mm.cu (compute capability 8.0+) decides per device at run time
// whether they run.

// Tensor-core tile: W warps stacked along m, 16 weight rows each (BM = 16 W), every warp all BN
// tokens of the tile; K steps of BK. mma.sync m16n8k16 BF16 -> F32 (sm_80+), operands from
// shared memory through ldmatrix; the next K step's global loads are issued before the current
// step's MMAs (register double buffering).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // only fp8_gemm uses it
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *(const uint32_t *) &v;
}
#endif

template <int BN, int BK, int W>
__global__ void __launch_bounds__(32 * W) fp8_gemm(const legion_fp8_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int BM = 16 * W;
    constexpr int T  = 32 * W;
    constexpr int LD   = BK + 8;            // smem row stride in bf16 (16-byte rows, no bank conflicts)
    constexpr int WCH  = BM * BK / 16 / T;  // 16-byte weight chunks per thread and step
    constexpr int XCH  = (BN * BK / 4 + T - 1) / T;  // float4 activation chunks per thread
    constexpr int NT   = BN / 8;            // n8 tiles per warp
    static_assert(WCH >= 1 && BM * BK / 16 % T == 0, "weight tile split");
    __shared__ __align__(16) __nv_bfloat16 ws[BM * LD];
    __shared__ __align__(16) __nv_bfloat16 xs[BN * LD];

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t m0 = (int64_t) blockIdx.x * BM;
    const int64_t n0 = (int64_t) blockIdx.y * BN;

    uint4  wreg[WCH];
    float4 xreg[XCH];
    auto load = [&](int64_t k0) {
#pragma unroll
        for (int i = 0; i < WCH; i++) {
            const int c = threadIdx.x + i * T;
            const int r = c / (BK / 16), col = c % (BK / 16) * 16;
            const int64_t row = m0 + r;
            wreg[i] = row < p.m ? __ldg((const uint4 *) (p.w + row * p.k + k0 + col)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int c = threadIdx.x + i * T;
            const int t = c / (BK / 4), col = c % (BK / 4) * 4;
            const int64_t tok = n0 + t;
            xreg[i] = (c < BN * BK / 4 && tok < p.n) ? __ldg((const float4 *) (p.x + tok * p.x_ld + k0 + col))
                                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int i = 0; i < WCH; i++) {
            const int c = threadIdx.x + i * T;
            const int r = c / (BK / 16), col = c % (BK / 16) * 16;
            const uint16_t * pairs = (const uint16_t *) &wreg[i];
            uint32_t out[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float2 f = e4m3x2_to_float2(pairs[j]);
                out[j] = pack_bf16(f.x, f.y);  // exact: every E4M3 value is a BF16 value
            }
            uint4 * d = (uint4 *) (ws + r * LD + col);
            d[0] = make_uint4(out[0], out[1], out[2], out[3]);
            d[1] = make_uint4(out[4], out[5], out[6], out[7]);
        }
#pragma unroll
        for (int i = 0; i < XCH; i++) {
            const int c = threadIdx.x + i * T;
            if (c < BN * BK / 4) {
                const int t = c / (BK / 4), col = c % (BK / 4) * 4;
                *(uint2 *) (xs + t * LD + col) =
                    make_uint2(pack_bf16(xreg[i].x, xreg[i].y), pack_bf16(xreg[i].z, xreg[i].w));
            }
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }

    load(0);
    for (int64_t k0 = 0; k0 < p.k; k0 += BK) {
        store();
        __syncthreads();
        if (k0 + BK < p.k) {
            load(k0 + BK);
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[4];
            {
                const __nv_bfloat16 * src = ws + (warp * 16 + lane % 16) * LD + kk + (lane / 16) * 8;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                uint32_t b[2];
                const __nv_bfloat16 * src = xs + (j * 8 + lane % 8) * LD + kk + ((lane / 8) % 2) * 8;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b[0]), "=r"(b[1])
                             : "r"(addr));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
        __syncthreads();
    }

    // C fragment: c0, c1 at (row lane/4, tokens 2*(lane%4) + 0/1), c2, c3 at row + 8.
    const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int64_t t0 = n0 + j * 8 + (lane % 4) * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int64_t row = r0 + (e / 2) * 8;
            const int64_t tok = t0 + e % 2;
            if (row < p.m && tok < p.n) {
                p.dst[tok * p.m + row] = p.wscale ? acc[j][e] * p.wscale[row * p.wscale_stride] : acc[j][e];
            }
        }
    }
#endif
}

// Large-batch tile (prefill), with BF16 activations (the graph casts them once): 128 weight
// rows x 128 tokens per CTA, 8 warps as 2 (rows) x 4 (tokens), each warp 64 x 32 = 4 m16 x 4 n8
// MMA tiles, so every A and B fragment feeds four MMAs. K steps of 64 through three stages
// filled by cp.async (raw E4M3 weights, BF16 activations) two steps ahead: with one CTA per SM
// and two stages the copies, not the MMAs, set the pace. Each step first widens its weight tile
// to BF16 in shared memory (exact), then runs the MMAs.
namespace big {
constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3;
constexpr int LD  = BK + 8;   // bf16 row stride: 144-byte rows, no ldmatrix bank conflicts
constexpr int LDW = BK;       // raw E4M3 row stride in bytes (unpadded, to fit three stages)
constexpr size_t RAW_W  = BM * LDW;                    // bytes per stage
constexpr size_t X_TILE = BN * LD * 2;                 // bytes per stage
constexpr size_t W_BF16 = BM * LD * 2;                 // one widened weight tile
constexpr size_t SMEM   = STAGES * (RAW_W + X_TILE) + W_BF16;  // 96 KiB
static_assert(SMEM <= 96 * 1024, "shared memory");
}  // namespace big

// The tile of a large GEMM's CTA (grid x: token tiles, y: row tiles), given the bytes of all
// its activations. Up to GROUPED_X_BYTES the token tiles vary fastest: the CTAs resident at
// once share weight rows, so the weights stream from memory once while the activations stay in
// L2. Past it (4096 tokens at k = 5120: 21 MB of E4M3, 42 MB of BF16, against the GB10's
// 24 MiB L2 that the weights pass through too), that order re-read every token's activations
// from memory every few row tiles
// (Qwen3.8-27B: 1.7-2.1x the time per token of 2048-token steps), so the grid is walked in
// groups of GROUP_M row tiles instead, the row tile fastest inside a group, then the token
// tile, and the CTAs resident at once share a few row tiles' weights and a few token tiles'
// activations. GB10, 2,506 tokens at k = 5120: E4M3 (13 MB) 5-8% faster in the first order,
// BF16 (26 MB) 28-40% faster grouped. A tile's sums do not depend on which CTA computes it, or
// when.
__device__ __forceinline__ void grouped_tile(int64_t x_bytes, int64_t & row_tile, int64_t & tok_tile) {
    constexpr int64_t GROUPED_X_BYTES = 16 << 20;
    if (x_bytes <= GROUPED_X_BYTES) {
        row_tile = blockIdx.y;
        tok_tile = blockIdx.x;
        return;
    }
    constexpr int GROUP_M = 8;
    const int tiles_n = (int) gridDim.x, tiles_m = (int) gridDim.y;
    const int pid = (int) blockIdx.y * tiles_n + (int) blockIdx.x, in_group = GROUP_M * tiles_n;
    const int first_m = pid / in_group * GROUP_M;
    const int group = min(tiles_m - first_m, GROUP_M);
    row_tile = first_m + pid % in_group % group;
    tok_tile = pid % in_group / group;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // cp.async: sm_80+, only the sm_80+ GEMMs use it
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool valid) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;  // 0: fill with zeros
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
#endif

__global__ void __launch_bounds__(256) fp8_gemm_big(const legion_fp8_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace big;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char * raw_w[STAGES];
    __nv_bfloat16 * xs[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; s++) {
        raw_w[s] = smem + s * RAW_W;
        xs[s] = (__nv_bfloat16 *) (smem + STAGES * RAW_W + s * X_TILE);
    }
    __nv_bfloat16 * ws = (__nv_bfloat16 *) (smem + STAGES * (RAW_W + X_TILE));
    const __nv_bfloat16 * x = (const __nv_bfloat16 *) p.x;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / 4, wn = warp % 4;
    int64_t tm, tn;
    grouped_tile(p.n * p.k * 2, tm, tn);
    const int64_t m0 = tm * BM, n0 = tn * BN;

    // One step's loads: weights 128 rows x 64 B (2 x 16 B per thread), activations 128 tokens
    // x 128 B (4 x 16 B per thread).
    auto load = [&](int64_t step, int stage) {
        const int64_t k0 = step * BK;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = threadIdx.x + i * 256;
            const int r = c / 4, col = c % 4 * 16;
            const int64_t row = m0 + r;
            const bool ok = row < p.m;
            cp_async16(raw_w[stage] + r * LDW + col, p.w + (ok ? row : 0) * p.k + k0 + col, ok);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = threadIdx.x + i * 256;
            const int t = c / 8, col = c % 8 * 8;
            const int64_t tok = n0 + t;
            const bool ok = tok < p.n;
            cp_async16(xs[stage] + t * LD + col, x + (ok ? tok : 0) * p.x_ld + k0 + col, ok);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // One cp.async group per step (empty past the last), so waiting for all but the newest
    // STAGES - 2 groups waits for exactly this step's copies.
    const int64_t steps = p.k / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < steps) {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % STAGES);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        // Also: every thread is past the previous step's MMAs, so the stage they read (the one
        // loaded next) and the widened tile are free.
        __syncthreads();
        const int64_t next = step + STAGES - 1;
        if (next < steps) {
            load(next, (int) (next % STAGES));
        }
        asm volatile("cp.async.commit_group;\n" ::);
        // Widen this step's weights: 32 bytes per thread.
        {
            const int r = threadIdx.x / 2, col = threadIdx.x % 2 * 32;
            const uint4 * src = (const uint4 *) (raw_w[stage] + r * LDW + col);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 v = src[h];
                const uint16_t * pairs = (const uint16_t *) &v;
                uint32_t out[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float2 f = e4m3x2_to_float2(pairs[j]);
                    out[j] = pack_bf16(f.x, f.y);
                }
                uint4 * d = (uint4 *) (ws + r * LD + col + h * 16);
                d[0] = make_uint4(out[0], out[1], out[2], out[3]);
                d[1] = make_uint4(out[4], out[5], out[6], out[7]);
            }
        }
        __syncthreads();
        const __nv_bfloat16 * xt = xs[stage];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[4][4], b[4][2];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const __nv_bfloat16 * src = ws + (wm * 64 + i * 16 + lane % 16) * LD + kk + (lane / 16) * 8;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                             : "r"(addr));
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int tok = wn * 32 + j * 16 + lane % 8 + (lane / 16) * 8;
                const __nv_bfloat16 * src = xt + tok * LD + kk + ((lane / 8) % 2) * 8;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(b[2 * j][0]), "=r"(b[2 * j][1]), "=r"(b[2 * j + 1][0]), "=r"(b[2 * j + 1][1])
                             : "r"(addr));
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    asm(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[j][0]), "r"(b[j][1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int64_t r0 = m0 + wm * 64 + i * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int64_t t0 = n0 + wn * 32 + j * 8 + (lane % 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t row = r0 + (e / 2) * 8, tok = t0 + e % 2;
                if (row < p.m && tok < p.n) {
                    p.dst[tok * p.m + row] =
                        p.wscale ? acc[i][j][e] * p.wscale[row * p.wscale_stride] : acc[i][j][e];
                }
            }
        }
    }
#endif
}
