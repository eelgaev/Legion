// gemm_glu.cu: the grouped fused SwiGLU GEMM (routed experts' gate/up at prompt steps) reading
// its weights from a tiled copy (#1183): per expert, per 128 output rows, per NVFP4 block of k,
// one contiguous 9,216-byte span holding the 128 gate and the 128 up rows' blocks, in the order
// the tile's lanes read them. A stage's weights are then one 9 KiB copy (one TMA bulk copy), and
// each lane's A fragments for the whole block are one 16-byte shared load per row tile.
//
// Not a standalone header: gemm_glu.cu includes it inside an anonymous namespace after
// common.cuh, ../sm90/tma.cuh and gemm_glu_tma.cuh. The kernel body exists only where
// __CUDA_ARCH__ >= 900.
//
// The span of (expert e, row tile mt, block b) sits at ((e * tiles_m + mt) * (k / 64) + b) *
// 9216. Inside it, warp w (rows w * 16 .. w * 16 + 15 of the tile's gate (i = 0) and up (i = 1)
// rows), lane l (g = l / 4, t = l % 4):
// - codes at ((w * 2 + i) * 32 + l) * 16: four 32-bit words, word s = the 2 code bytes at block
//   offset 4 + 8 s + 2 t of row g (low half) and of row g + 8 (high half);
// - scales at 8192 + (w * 32 + l) * 4: the scale byte t of rows (i 0, g), (0, g + 8), (1, g),
//   (1, g + 8).
// Exactly the bytes nvfp4_gemm's lanes read from their shared rows, so the same A fragments, the
// same MMAs in the same order: nvfp4_gemm's bits. Rows past m: zero bytes (never stored).
//
// Modes (how a stage is filled; the micro harness compares them, #1183 r20):
// 0 every thread's cp.async (the span's 576 16-byte words and the token rows), a CTA barrier per
//   step: nvfp4_gemm's pipeline on the tiled layout;
// 1 the span as one TMA bulk copy (thread 0) completing on the stage's mbarrier, the token rows
//   by every thread's cp.async, a CTA barrier per step;
// 2 the span and each token row (128 bytes) as TMA bulk copies issued by the last warp out of the
//   stage, mbarrier stages, no CTA barrier;
// 3 as 2, the token rows by that warp's cp.async (arriving on the stage's mbarrier).

namespace glu_tile {
constexpr int ST = 3, SPAN = 9216, CODES = 8192;
template <int NT> __host__ __device__ constexpr int stage_bytes() {
    return SPAN + 8 * NT * LDX * 2;
}
template <int NT> __host__ __device__ constexpr int smem_bytes() {
    return ST * stage_bytes<NT>() + 128;  // + 128 to align the stages
}
static_assert(smem_bytes<16>() <= 96 * 1024, "shared memory");
static_assert(stage_bytes<8>() % 128 == 0 && stage_bytes<16>() % 128 == 0, "stage alignment");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// `bytes` (a multiple of 16) from global `src` to shared `dst`, completing on `bar`.
__device__ __forceinline__ void bulk(void * dst, const void * src, uint32_t bytes, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
            legion_tma::saddr(dst)),
        "l"((uint64_t) src), "r"(bytes), "r"(legion_tma::saddr(bar))
        : "memory");
}
#endif
}  // namespace glu_tile

template <int NT, int MODE>
__global__ void __launch_bounds__(256, 1)
    nvfp4_glu_tile(const legion_nvfp4_mm p, const uint8_t * __restrict__ packed) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace legion_tma;
    constexpr int BN = 8 * NT, BM_OUT = 128, ST = glu_tile::ST, WARPS = 8;
    constexpr int SPAN = glu_tile::SPAN, STAGE = glu_tile::stage_bytes<NT>();
    constexpr bool SYNC = MODE <= 1;  // a CTA barrier per step (else mbarrier stages only)
    extern __shared__ __align__(128) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 127) & ~(uintptr_t) 127);
    __shared__ int64_t x_off[BN], d_off[BN];
    __shared__ int32_t tile_expert, tile_tokens;
    __shared__ uint64_t full[ST];
    __shared__ uint32_t released[ST];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    // nvfp4_gemm's grouped tile: row tiles vary fastest.
    const int mt = blockIdx.x % p.tiles_m;
    const int64_t m0 = (int64_t) mt * BM_OUT;
    const int tile = blockIdx.x / p.tiles_m;
    constexpr int TI = legion_nvfp4::plan_tile_index(BN);
    if (legion_nvfp4::past_plan_tiles<TI>(p, tile)) {
        const int routed = p.plan[p.n_expert];
        const int skipped = (int) p.n - routed;
        if (skipped > 0 && !p.unread_skips) {
            const int tiles = p.plan[p.n_expert + 1 + p.n + TI];
            const int spare = tile - tiles, n_spare = (int) (gridDim.x / p.tiles_m) - tiles;
            const int64_t rows = min((int64_t) BM_OUT, p.m - m0);
            for (int j = spare; j < skipped; j += n_spare) {
                const int64_t off = (int64_t) p.plan[p.n_expert + 1 + routed + j] * p.m + m0;
                for (int64_t r = threadIdx.x; r < rows; r += 256) {
                    ((__nv_bfloat16 *) p.dst)[off + r] = __float2bfloat16_rn(0.0f);
                }
            }
        }
        return;
    }
    if (warp == 0) {
        int e, first, count;
        if (!find_tile(p.plan, p.n_expert, BN, tile, lane, p.sel_lo, p.sel_hi, e, first, count)) {
            if (lane == 0) {
                tile_expert = -1;
            }
        } else {
            if (lane == 0) {
                tile_expert = e;
                tile_tokens = count;
            }
            for (int i = lane; i < count; i += 32) {
                const int q = p.plan[p.n_expert + 1 + first + i];  // token * n_used + slot
                const int t = q / p.n_used, slot = q % p.n_used;
                x_off[i] = ((int64_t) t * p.n_x + (p.n_x == 1 ? 0 : slot)) * p.x_ld;
                d_off[i] = (int64_t) q * p.m;
            }
        }
    }
    if (MODE >= 1 && threadIdx.x == 32) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            // Mode 1, 2: the issuing lane's expect_tx; mode 3: that and each lane's copies.
            bar_init(full + s, MODE == 3 ? 1 + 32 : 1);
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (tile_expert < 0) {
        return;
    }
    const int expert = tile_expert;
    const float sc = (p.scale ? p.scale[expert] : 1.0f) * 1024.0f;  // 2^10 undoes the decoding's 2^-10
    const float sc2 = (p.scale2 ? p.scale2[expert] : 1.0f) * 1024.0f;
    const int n_tok = tile_tokens;
    const int64_t steps = p.k / BK;
    const uint8_t * wt = packed + ((int64_t) expert * p.tiles_m + mt) * steps * SPAN;

    // The tokens' rows of `step` by every thread's cp.async (modes 0, 1), zeros past the last.
    auto tokens_all = [&](int64_t step, __nv_bfloat16 * xs) {
#pragma unroll
        for (int i = 0; i < (BN * 8 + 255) / 256; i++) {
            const int c = threadIdx.x + i * 256;
            if (BN * 8 % 256 == 0 || c < BN * 8) {
                const int t = c / 8, col = c % 8 * 8;
                const bool ok = t < n_tok;
                cp_async16(xs + t * LDX + col, p.x + (ok ? x_off[t] : 0) + step * BK + col, ok);
            }
        }
    };
    // Modes 0, 1: step `step` into stage `stage`, by the whole CTA.
    auto load_sync = [&](int64_t step, int stage) {
        unsigned char * s = smem + stage * STAGE;
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (s + SPAN);
        const uint8_t * src = wt + step * SPAN;
        if constexpr (MODE == 0) {
#pragma unroll
            for (int i = 0; i < (SPAN / 16 + 255) / 256; i++) {
                const int c = threadIdx.x + i * 256;
                if (c < SPAN / 16) {
                    cp_async16(s + c * 16, src + c * 16, true);
                }
            }
        } else {
            if (threadIdx.x == 0) {
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                bar_expect_tx(full + stage, SPAN);
                glu_tile::bulk(s, src, SPAN, full + stage);
            }
        }
        tokens_all(step, xs);
    };
    // Modes 2, 3: step `step` into stage `stage`, by one whole warp.
    auto issue = [&](int64_t step, int stage) {
        unsigned char * s = smem + stage * STAGE;
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (s + SPAN);
        uint64_t * bar = full + stage;
        if (lane == 0) {
            bar_expect_tx(bar, SPAN + (MODE == 2 ? n_tok * BK * 2 : 0));
            glu_tile::bulk(s, wt + step * SPAN, SPAN, bar);
        }
        if constexpr (MODE == 2) {
            __syncwarp();
            for (int t = lane; t < n_tok; t += 32) {
                glu_tile::bulk(xs + t * LDX, p.x + x_off[t] + step * BK, BK * 2, bar);
            }
        } else {
#pragma unroll 4
            for (int c = lane; c < BN * 8; c += 32) {
                const int t = c / 8, col = c % 8 * 8;
                const bool ok = t < n_tok;
                cp_async16(xs + t * LDX + col, p.x + (ok ? x_off[t] : 0) + step * BK + col, ok);
            }
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(saddr(bar)) : "memory");
        }
    };

    // The L2 bulk prefetch of the next PF spans, PF steps ahead (the tile's spans are one
    // contiguous run: one prefetch per span).
    constexpr int PF = 32;
    auto prefetch = [&](int64_t step) {
        const int r = threadIdx.x;
        if (r < PF && step + r < steps) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"((uint64_t) (wt + (step + r) * SPAN)),
                         "r"(SPAN));
        }
    };
    prefetch(0);
    prefetch(PF);

    if constexpr (SYNC) {
#pragma unroll
        for (int s = 0; s < ST - 1; s++) {
            if (s < steps) {
                load_sync(s, s);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }
    } else if (warp == 0) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            if (s < steps) {
                issue(s, s);
            }
        }
    }

    float acc[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // nvfp4_gemm's stage MMAs on the tiled span: the same A fragments from one 16-byte load per
    // row tile and one 4-byte load of the scales.
    auto mma_stage = [&](const unsigned char * ws, const __nv_bfloat16 * xs, auto full_tile) {
        constexpr bool FULL = decltype(full_tile)::value;
        uint32_t cw[2][4];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 v = *(const uint4 *) (ws + ((warp * 2 + i) * 32 + lane) * 16);
            cw[i][0] = v.x;
            cw[i][1] = v.y;
            cw[i][2] = v.z;
            cw[i][3] = v.w;
        }
        const uint32_t scw = *(const uint32_t *) (ws + glu_tile::CODES + (warp * 32 + lane) * 4);
        uint32_t ssc[2][2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                ssc[i][h] = scale_bf16x2((scw >> (8 * (2 * i + h))) & 0xff);
            }
#pragma unroll
        for (int s = 0; s < 4; s++) {
            uint32_t a[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t d = __shfl_sync(0xffffffffu, ssc[i][h], (lane & ~3) | s);
                    uint32_t lo, hi;
                    codes_bf16x2(__byte_perm(cw[i][s], 0, h ? 0x4342 : 0x4140), lo, hi);
                    a[i][h] = mul_bf16x2(lo, d);
                    a[i][2 + h] = mul_bf16x2(hi, d);
                }
#pragma unroll
            for (int half = 0; half < (NT + 7) / 8; half++) {
                if (!FULL && half * 64 >= n_tok) {
                    break;
                }
                constexpr int JN = NT < 8 ? NT : 8;
                uint32_t b[JN][2];
#pragma unroll
                for (int j = 0; j < JN / 2; j++) {
                    if (!FULL && half * 64 + j * 16 >= n_tok) {
                        break;
                    }
                    const int tok = half * 64 + j * 16 + lane % 8 + (lane / 16) * 8;
                    const __nv_bfloat16 * src = xs + tok * LDX + s * 16 + ((lane / 8) % 2) * 8;
                    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b[2 * j][0]), "=r"(b[2 * j][1]), "=r"(b[2 * j + 1][0]), "=r"(b[2 * j + 1][1])
                        : "r"(saddr(src)));
                }
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < JN; j++) {
                        if (FULL || half * 64 + j * 8 < n_tok) {
                            mma(acc[i][half * 8 + j], a[i], b[j][0], b[j][1]);
                        }
                    }
            }
        }
    };
    auto compute = [&](int stage) {
        const unsigned char * ws = smem + stage * STAGE;
        const __nv_bfloat16 * xs = (const __nv_bfloat16 *) (ws + SPAN);
        if (n_tok == BN) {
            mma_stage(ws, xs, std::true_type{});
        } else {
            mma_stage(ws, xs, std::false_type{});
        }
    };

    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % ST);
        if (step % PF == 0 && step > 0) {
            prefetch(step + PF);
        }
        if constexpr (SYNC) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
            if constexpr (MODE == 1) {
                bar_wait(full + stage, (uint32_t) (step / ST) & 1);
            }
            __syncthreads();
            const int64_t next = step + ST - 1;
            if (next < steps) {
                load_sync(next, (int) (next % ST));
            }
            asm volatile("cp.async.commit_group;\n" ::);
            compute(stage);
        } else {
            bar_wait(full + stage, (uint32_t) (step / ST) & 1);
            compute(stage);
            if (step + ST < steps) {
                // This warp's reads of the stage are done; the last warp out refills it (as
                // nvfp4_glu_tma).
                __syncwarp();
                uint32_t prev = 0;
                if (lane == 0) {
                    asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                                 : "=r"(prev)
                                 : "r"(saddr(released + stage))
                                 : "memory");
                }
                prev = __shfl_sync(0xffffffffu, prev, 0);
                if (prev % WARPS == WARPS - 1) {
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    issue(step + ST, stage);
                }
            }
        }
    }
    if constexpr (SYNC) {
        asm volatile("cp.async.wait_group 0;\n" ::);
    }

    // nvfp4_gemm's SwiGLU epilogue.
    const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
    for (int j = 0; j < NT; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int64_t row = r0 + (e / 2) * 8;
            const int t = j * 8 + (lane % 4) * 2 + e % 2;
            if (row < p.m && t < n_tok) {
                float gate = acc[0][j][e] * sc, up = acc[1][j][e] * sc2;
                if (p.limit > 0.0f) {
                    gate = fminf(gate, p.limit);
                    up = fminf(fmaxf(up, -p.limit), p.limit);
                }
                ((__nv_bfloat16 *) p.dst)[d_off[t] + row] = __float2bfloat16_rn(glu_gate(gate, p.act) * up);
            }
        }
    }
#else
    __trap();  // launched only where glu_tma_code answers yes
#endif
}

// nvfp4_glu_tile for a grouped SwiGLU GEMM of 8 * NT tokens per tile in `mode`; -1 when it does
// not apply (nothing launched).
template <int NT, int MODE> int launch_glu_tile(const legion_nvfp4_mm & p, const uint8_t * packed, cudaStream_t st) {
    if (!p.plan || !packed || ((uintptr_t) packed | (uintptr_t) p.x) % 16 != 0 || p.k % 64 != 0 || p.k <= 0 ||
        p.m <= 0 || p.x_ld % 8 != 0 || p.tiles_m != (int) ((p.m + 127) / 128) || !glu_tma_code()) {
        return -1;
    }
    constexpr int SMEM = glu_tile::smem_bytes<NT>();
    auto kernel = nvfp4_glu_tile<NT, MODE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    kernel<<<(unsigned) p.tiles_m * (unsigned) p.tiles_n, 256, SMEM, st>>>(p, packed);
    return status();
}

template <int NT> int launch_glu_tile_mode(const legion_nvfp4_mm & p, const uint8_t * packed, int mode, cudaStream_t st) {
    switch (mode) {
        case 0: return launch_glu_tile<NT, 0>(p, packed, st);
        case 1: return launch_glu_tile<NT, 1>(p, packed, st);
        case 2: return launch_glu_tile<NT, 2>(p, packed, st);
        case 3: return launch_glu_tile<NT, 3>(p, packed, st);
        default: return -1;
    }
}
