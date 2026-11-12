// fp8_mm.cu, sm_80 and newer: the sliced-K W8A16 GEMM for verify-sized batches (5-64 tokens),
// fp8_gemm_sliced, opt-in (`legion serve --fp8-sliced-k`).
//
// Adapted from NInfer's fp8_a16_sliced_k_mma (src/ops/linear/kernels/fp8/
// fp8_a16_sliced_k_mma.cuh and fp8_schedule.cuh, github.com/neroued/ninfer, commit 64d10fb).
// Copyright the NInfer authors. Licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0); this file is a modified version: Legion's
// operand layouts (raw E4M3 rows, F32 output, an F32 scale per tensor or per row), a reduction
// in fixed warp order spread over the CTA's threads, and its own tile configurations.
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm80/fp8_mm.cuh (pack_bf16, cp_async16) and the legion_fp8_mm declaration. The kernel body is
// built only where __CUDA_ARCH__ >= 800 (a trap elsewhere); bf16_mma() decides per device at run
// time.
//
// Why: a verify step's 5-64 tokens leave the prefill-shaped GEMM tiles (fp8_gemm: 16-128
// weight rows per CTA, one K step of 32-64 at a time, register double buffering) short of
// loads in flight, so they read the weights well below the GEMV's bandwidth (GB10: 170-195
// against 235-243 GB/s, #520). Here a CTA owns only 16 or 32 weight rows and splits K across KW
// warps: every K group of KW x 64 columns is one cp.async batch of 16 MT x KW x 64 weight bytes
// plus the tokens' BF16 activations for it, each warp runs m16n8k16 BF16 MMAs over its own 64
// columns, and the warps' partial sums are added in shared memory at the end. Many small CTAs
// per SM keep the weight stream at bandwidth while the tensor cores do the per-token work; 32
// rows per CTA halve the activation bytes per weight byte for 33-64 tokens.
//
// Bits: the weights are widened to BF16 exactly and the activations come in as BF16 (the graph
// rounds them once, as for the large GEMM), F32 accumulation; but K is summed per warp slice
// and the slices are then added in warp order, so the sums round differently from fp8_gemm's
// (one running sum over K). Hence opt-in.
//
// Launch (#689): on compute capability 12.x a programmatic dependent launch (launch_pdl.cuh), so
// a CTA issues its first stages' weight copies (no kernel's output) before it waits for the
// kernel ahead (griddepcontrol.wait) and loads that kernel's activations, and lets the next
// kernel launch early. The same sums.

namespace sk {
constexpr int TILE_K = 64;  // K columns per warp per group

// MT m16 MMA tiles of weight rows per CTA (ROWS = 16 MT).
template <int KW, int TOK, int STAGES, int MT>
struct cfg {
    static constexpr int ROWS = 16 * MT;                                         // weight rows per CTA
    static constexpr int BK = KW * TILE_K;                                       // K per group
    static constexpr size_t CODES = (size_t) ROWS * BK;                          // weight bytes per stage
    static constexpr size_t ACT = (size_t) KW * TOK * TILE_K * 2;                // BF16 bytes per stage
    static constexpr size_t STAGING = STAGES * (CODES + ACT);
    static constexpr size_t PARTIAL = (size_t) KW * MT * (TOK / 8) * 32 * 4 * 4;  // F32 partial sums
    static constexpr size_t SMEM = STAGING > PARTIAL ? STAGING : PARTIAL;
};
}  // namespace sk

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // only fp8_gemm_sliced uses it
// cp.async through L1 (.ca): the CTAs on an SM read the same activations.
__device__ __forceinline__ void cp_async16_ca(void * smem, const void * gmem, bool valid) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;  // 0: fill with zeros
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
#endif

// KW warps split K (KW x 64 columns per group, 64 per warp), MT x 16 weight rows and TOK
// tokens per CTA (a multiple of 8; tokens past n read as zeros and are not stored), STAGES
// cp.async stages, MINB the CTAs per SM to keep resident. Grid: x = row tiles of 16 MT, y =
// token tiles of TOK. Needs BF16 activations (x_bf16), k a multiple of KW x 64 and x_ld a
// multiple of 8.
template <int KW, int TOK, int STAGES, int MT, int MINB>
__global__ void __launch_bounds__(32 * KW, MINB) fp8_gemm_sliced(const legion_fp8_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = sk::cfg<KW, TOK, STAGES, MT>;
    constexpr int T = 32 * KW;
    constexpr int NT = TOK / 8;  // n8 MMA tiles
    constexpr int BK = C::BK;
    constexpr int ROWS = C::ROWS;
    constexpr int CHUNKS = ROWS * BK / 16 / T;  // 16-byte weight chunks per thread and group
    static_assert(TOK % 8 == 0 && STAGES >= 1 && STAGES <= 2 && (MT == 1 || MT == 2), "tile");
    static_assert(CHUNKS * T * 16 == ROWS * BK, "whole weight chunks per thread and group");
    extern __shared__ __align__(16) unsigned char smem[];
    // Stage s: weight codes [ROWS][BK] bytes (16-byte chunks XOR-swizzled by row), then each
    // warp's activations [TOK][64] BF16 (16-byte chunks XOR-swizzled by token).
    auto codes = [&](int s) { return smem + s * (C::CODES + C::ACT); };
    auto acts = [&](int s, int w) {
        return (__nv_bfloat16 *) (smem + s * (C::CODES + C::ACT) + C::CODES) + w * TOK * sk::TILE_K;
    };
    const __nv_bfloat16 * x = (const __nv_bfloat16 *) p.x;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gid = lane / 4, lid = lane % 4;
    const int64_t m0 = (int64_t) blockIdx.x * ROWS;
    const int64_t n0 = (int64_t) blockIdx.y * TOK;
    const int64_t groups = p.k / BK;

    auto load_w = [&](int stage, int64_t g) {
        const int64_t k0 = g * BK;
        unsigned char * cs = codes(stage);
#pragma unroll
        for (int i = 0; i < CHUNKS; i++) {
            const int c = threadIdx.x + i * T;
            const int r = c / (BK / 16), ch = c % (BK / 16);
            // Rows past the last one read the last row again; their sums are not stored.
            const int64_t row = min(m0 + r, p.m - 1);
            cp_async16(cs + r * BK + (ch ^ (r & 7)) * 16, p.w + row * p.k + k0 + ch * 16, true);
        }
    };
    auto load_x = [&](int stage, int64_t g) {
        const int64_t k0 = g * BK;
        __nv_bfloat16 * xs = acts(stage, warp);
        const int64_t kw = k0 + warp * sk::TILE_K;
#pragma unroll
        for (int item = lane; item < TOK * 8; item += 32) {
            const int t = item / 8, c8 = item % 8;
            const int64_t tok = n0 + t;
            const bool ok = tok < p.n;
            cp_async16_ca(xs + t * sk::TILE_K + ((c8 ^ (t & 7)) * 8),
                          x + (ok ? tok : 0) * p.x_ld + kw + c8 * 8, ok);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.0f;
        }
    }

#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
    // The first stages' weights before the wait, their activations after it; group s holds
    // stage s's activations (and group 0 all these weights).
#pragma unroll
    for (int s = 0; s < STAGES; s++) {
        if (s < groups) {
            load_w(s, s);
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#pragma unroll
    for (int s = 0; s < STAGES; s++) {
        if (s < groups) {
            load_x(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    const int wk0 = warp * sk::TILE_K;
    for (int64_t g = 0; g < groups; g++) {
        const int stage = (int) (g % STAGES);
        // One group committed per stage and per later load (empty past the last), so waiting
        // for all but the newest STAGES - 1 waits for exactly this group's copies.
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 1));
        __syncthreads();
        const unsigned char * cs = codes(stage);
        const __nv_bfloat16 * xs = acts(stage, warp);
#pragma unroll
        for (int ks = 0; ks < sk::TILE_K / 16; ks++) {
            // A fragments: rows gid and gid + 8 of each m16 tile, columns 2 lid (+1) and
            // 2 lid + 8 (+1) of this warp's k step, widened from E4M3 to BF16 (exact).
            auto pair = [&](int r, int col) {
                const int kc = wk0 + col;
                const uint16_t two = *(const uint16_t *) (cs + r * BK + (((kc / 16) ^ (r & 7)) * 16) + kc % 16);
                const float2 f = e4m3x2_to_float2(two);
                return pack_bf16(f.x, f.y);
            };
            const int col = ks * 16 + lid * 2;
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                a[mt][0] = pair(mt * 16 + gid, col);
                a[mt][1] = pair(mt * 16 + gid + 8, col);
                a[mt][2] = pair(mt * 16 + gid, col + 8);
                a[mt][3] = pair(mt * 16 + gid + 8, col + 8);
            }
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const int t = j * 8 + lane % 8;
                const int c8 = ks * 2 + (lane / 8) % 2;
                const __nv_bfloat16 * src = xs + t * sk::TILE_K + ((c8 ^ (t & 7)) * 8);
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(src);
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "r"(addr));
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[mt][j][0]), "+f"(acc[mt][j][1]), "+f"(acc[mt][j][2]), "+f"(acc[mt][j][3])
                        : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1));
                }
            }
        }
        // Every warp is done with this stage before it is loaded again.
        __syncthreads();
        if (g + STAGES < groups) {
            load_w(stage, g + STAGES);
            load_x(stage, g + STAGES);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // Partial sums: warp w's fragment (mt, j) at [w][mt][j][lane][4] (the staging space is free
    // now).
    float * part = (float *) smem;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int j = 0; j < NT; j++) {
            *(float4 *) (part + (((warp * MT + mt) * NT + j) * 32 + lane) * 4) =
                make_float4(acc[mt][j][0], acc[mt][j][1], acc[mt][j][2], acc[mt][j][3]);
        }
    }
    __syncthreads();
    // Each (row, token) of the tile by one thread: the warps' sums added in warp order, rows
    // fastest so the stores of a token's rows lie together. C fragment of an m16 tile: e = 0, 1
    // at (row gid, tokens 2 lid + 0/1), e = 2, 3 at row gid + 8.
    for (int i = threadIdx.x; i < ROWS * TOK; i += T) {
        const int r = i % ROWS, t = i / ROWS;
        const int64_t row = m0 + r, tok = n0 + t;
        if (row >= p.m || tok >= p.n) {
            continue;
        }
        const int mt = r / 16, rr = r % 16;
        const int j = t / 8, ln = (rr % 8) * 4 + (t % 8) / 2, e = (rr / 8) * 2 + t % 2;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < KW; w++) {
            s += part[(((w * MT + mt) * NT + j) * 32 + ln) * 4 + e];
        }
        p.dst[tok * p.m + row] = p.wscale ? s * p.wscale[row * p.wscale_stride] : s;
    }
#else
    __trap();
#endif
}
