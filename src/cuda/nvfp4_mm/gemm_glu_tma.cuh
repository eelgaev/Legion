// gemm_glu.cu: the grouped fused SwiGLU GEMM (routed experts' gate/up at prompt steps) with its
// stages filled by the tensor memory accelerator (TMA) instead of per-thread cp.async copies
// (#1183; after nvfp4_a4_tma.cuh, #1133).
//
// Not a standalone header: gemm_glu.cu includes it inside an anonymous namespace after
// common.cuh and ../sm90/tma.cuh. The kernel body exists only where __CUDA_ARCH__ >= 900.
//
// nvfp4_gemm<2, NT, true, true> (128 gate and 128 up rows x 8 * NT tokens, one NVFP4 block of k
// per stage, three stages) issues each stage as three 16-byte cp.async per weight row and eight
// per token, then a CTA barrier at every step. Here a stage is two TMA tiles (the 128 gate rows
// and the same rows of the up expert, each row's block as the three aligned 16-byte words
// around it, as the cp.async copies take it) plus the tokens' cp.async copies of one warp, all
// completing on the stage's "full" mbarrier. Each warp counts itself out of a stage when its
// reads of it are done and the last warp to do so refills it, so no warp waits for the others
// except on data.
//
// The tile's fragments, MMAs, k order, L2 prefetch and epilogue are nvfp4_gemm's: the gate's
// rows sit in the stage's first 128 rows and the up rows in the next 128 (nvfp4_gemm
// interleaves them by 16), which moves only where a fragment is read from. Every output takes
// the same MMAs in the same order: nvfp4_gemm's bits.
//
// Needs the weight rows 16-byte aligned (w, w2, w_ld and w_e multiples of 16: k % 256 == 0 for
// packed rows) and the activations as nvfp4_gemm does (x and x_ld * 2 multiples of 16).

namespace glu_tma {
// The tensor maps of a launch: the gate and the up experts, [row bytes, m, n_expert].
struct Maps {
    CUtensorMap w, w2;
};
constexpr int ST = 3, ROWS = 128, W_HALF = ROWS * LDW;
template <int NT> __host__ __device__ constexpr int stage_bytes() {
    return 2 * W_HALF + 8 * NT * LDX * 2;
}
template <int NT> __host__ __device__ constexpr int smem_bytes() {
    return ST * stage_bytes<NT>() + 128;  // + 128 to align the stages
}
static_assert(smem_bytes<16>() <= 99 * 1024, "shared memory");
static_assert(W_HALF % 128 == 0 && stage_bytes<8>() % 128 == 0 && stage_bytes<16>() % 128 == 0, "TMA alignment");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// A 3D tile (c0: byte along the row, c1: row, c2: expert) of `map` into shared memory.
__device__ __forceinline__ void tma_3d(void * dst, const CUtensorMap * map, int c0, int c1, int c2, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];\n" ::"r"(legion_tma::saddr(dst)),
        "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(c2), "r"(legion_tma::saddr(bar))
        : "memory");
}
#endif
}  // namespace glu_tma

// HYB (#1183 r20 micro harness): the weight tiles by TMA as above, but the token rows by every
// thread's cp.async and a CTA barrier per step, nvfp4_gemm's pipeline.
template <int NT, bool HYB = false>
__global__ void __launch_bounds__(256, 1) nvfp4_glu_tma(const legion_nvfp4_mm p, const __grid_constant__ glu_tma::Maps maps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace legion_tma;
    constexpr int MI = 2, BN = 8 * NT, BM_OUT = 128, ST = glu_tma::ST, WARPS = 8;
    constexpr int W_HALF = glu_tma::W_HALF, STAGE = glu_tma::stage_bytes<NT>();
    extern __shared__ __align__(128) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 127) & ~(uintptr_t) 127);
    __shared__ int64_t x_off[BN], d_off[BN];
    __shared__ int32_t tile_expert, tile_tokens;
    __shared__ uint64_t full[ST];
    __shared__ uint32_t released[ST];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    // nvfp4_gemm's grouped tile: row tiles vary fastest.
    const int64_t m0 = (int64_t) (blockIdx.x % p.tiles_m) * BM_OUT;
    const int tile = blockIdx.x / p.tiles_m;
    constexpr int TI = legion_nvfp4::plan_tile_index(BN);
    if (legion_nvfp4::past_plan_tiles<TI>(p, tile)) {
        // Past the routing's last tile: zeros for the plan's skipped pairs, as nvfp4_gemm.
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
    if (threadIdx.x == 32) {
        prefetch_map(&maps.w);
        prefetch_map(&maps.w2);
#pragma unroll
        for (int s = 0; s < ST; s++) {
            bar_init(full + s, HYB ? 1 : 1 + 32);  // the TMA tiles' expect_tx, and each lane's copies
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (tile_expert < 0) {
        return;
    }
    const int expert = tile_expert;
    const uint8_t * w = p.w + (int64_t) expert * p.w_e;
    const uint8_t * w2 = p.w2 + (int64_t) expert * p.w_e;
    const float sc = (p.scale ? p.scale[expert] : 1.0f) * 1024.0f;  // 2^10 undoes the decoding's 2^-10
    const float sc2 = (p.scale2 ? p.scale2[expert] : 1.0f) * 1024.0f;
    const int n_tok = tile_tokens;
    const int64_t steps = p.k / BK;

    // Step `step` into stage `stage`: called by a whole warp. The weights as two TMA tiles; the
    // tokens' rows as nvfp4_gemm's 16-byte cp.async copies (zeros past the tile's last token),
    // each lane's arriving on the barrier once its copies land (a bulk copy per 128-byte row
    // made the kernel twice as slow as nvfp4_gemm at 128 tokens).
    auto issue = [&](int64_t step, int stage) {
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        if (lane == 0) {
            bar_expect_tx(bar, 2 * W_HALF);
            // The block's three aligned 16-byte words (the rows start 16-byte aligned).
            const int c0 = (int) ((step * 36) & ~(int64_t) 15);
            glu_tma::tma_3d(s, &maps.w, c0, (int) m0, expert, bar);
            glu_tma::tma_3d(s + W_HALF, &maps.w2, c0, (int) m0, expert, bar);
        }
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (s + 2 * W_HALF);
#pragma unroll 4
        for (int c = lane; c < BN * 8; c += 32) {
            const int t = c / 8, col = c % 8 * 8;
            const bool ok = t < n_tok;
            cp_async16(xs + t * LDX + col, p.x + (ok ? x_off[t] : 0) + step * BK + col, ok);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(saddr(bar)) : "memory");
    };
    // HYB: step `step` into stage `stage` by the whole CTA (thread 0 the weight tiles).
    auto load_hyb = [&](int64_t step, int stage) {
        unsigned char * s = smem + stage * STAGE;
        uint64_t * bar = full + stage;
        if (threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            bar_expect_tx(bar, 2 * W_HALF);
            const int c0 = (int) ((step * 36) & ~(int64_t) 15);
            glu_tma::tma_3d(s, &maps.w, c0, (int) m0, expert, bar);
            glu_tma::tma_3d(s + W_HALF, &maps.w2, c0, (int) m0, expert, bar);
        }
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (s + 2 * W_HALF);
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

    // nvfp4_gemm's L2 bulk prefetch: each row's next PF blocks, PF blocks ahead of the copies.
    constexpr int PF = 32;
    auto prefetch = [&](int64_t step) {
        if (step < steps) {
            const int r = threadIdx.x;
            const int64_t row = m0 + r / 32 * 16 + r % 16;
            if (row < p.m) {
                const uint8_t * wt = r / 16 % 2 ? w2 : w;
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + step * 36);
                const uintptr_t e = a + min((int64_t) PF, steps - step) * 36;
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((e - a16 + 15) & ~(uintptr_t) 15)));
            }
        }
    };
    prefetch(0);
    prefetch(PF);
    if (HYB) {
#pragma unroll
        for (int s = 0; s < ST - 1; s++) {
            if (s < steps) {
                load_hyb(s, s);
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

    float acc[MI][NT][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    // nvfp4_gemm's stage MMAs; row tile i of this warp: the gate's rows (i = 0) or the up's.
    auto mma_stage = [&](const unsigned char * ws, const __nv_bfloat16 * xs, int64_t step, auto full_tile) {
        constexpr bool FULL = decltype(full_tile)::value;
        const uint32_t in_word = (uint32_t) (step * 36) & 15;
        const unsigned char * rows[MI][2];
        uint32_t ssc[MI][2];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                rows[i][h] = ws + i * W_HALF + (warp * 16 + h * 8 + lane / 4) * LDW + in_word;
                ssc[i][h] = scale_bf16x2(rows[i][h][lane % 4]);
            }
#pragma unroll
        for (int s = 0; s < 4; s++) {
            uint32_t a[MI][4];
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t q = *(const uint16_t *) (rows[i][h] + 4 + 8 * s + 2 * (lane % 4));
                    const uint32_t d = __shfl_sync(0xffffffffu, ssc[i][h], (lane & ~3) | s);
                    uint32_t lo, hi;
                    codes_bf16x2(__byte_perm(q, 0, 0x4140), lo, hi);
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
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < JN; j++) {
                        if (FULL || half * 64 + j * 8 < n_tok) {
                            mma(acc[i][half * 8 + j], a[i], b[j][0], b[j][1]);
                        }
                    }
            }
        }
    };

    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % ST);
        if (step % PF == 0 && step > 0) {
            prefetch(step + PF);
        }
        if (HYB) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
        }
        bar_wait(full + stage, (uint32_t) (step / ST) & 1);
        if (HYB) {
            __syncthreads();
            if (step + ST - 1 < steps) {
                load_hyb(step + ST - 1, (int) ((step + ST - 1) % ST));
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }
        const unsigned char * ws = smem + stage * STAGE;
        const __nv_bfloat16 * xs = (const __nv_bfloat16 *) (ws + 2 * W_HALF);
        if (n_tok == BN) {
            mma_stage(ws, xs, step, std::true_type{});
        } else {
            mma_stage(ws, xs, step, std::false_type{});
        }
        if (!HYB && step + ST < steps) {
            // This warp's reads of the stage are done (their values are in the MMAs above). The
            // acq_rel count orders every warp's reads before the last arrival, whose warp
            // refills the stage with step + ST (after a proxy fence: the copies write through
            // the async proxy).
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

// Whether this device runs nvfp4_glu_tma's sm_90+ body (compute capability 9.0+ and code built
// for it), asked once per device.
inline bool glu_tma_code() {
    static std::atomic<int> state[legion_nvfp4::PDL_DEVICES];  // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= legion_nvfp4::PDL_DEVICES) {
        return false;
    }
    int s = state[dev].load(std::memory_order_relaxed);
    if (s == 0) {
        int major = 0;
        cudaFuncAttributes a{};
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        major >= 9 && cudaFuncGetAttributes(&a, nvfp4_glu_tma<16>) == cudaSuccess &&
                        a.binaryVersion >= 90;
        cudaGetLastError();  // a failed query leaves no error behind
        s = ok ? 1 : 2;
        state[dev].store(s, std::memory_order_relaxed);
    }
    return s == 1;
}

// The 3D tensor map of `n_expert` experts of `m` rows of `row_bytes` bytes (rows w_ld, experts
// w_e bytes apart) at `base`, in boxes of the three words around a block x 128 rows.
inline bool glu_tma_map(CUtensorMap * map, const void * base, int64_t row_bytes, int64_t m, int64_t n_expert,
                        int64_t w_ld, int64_t w_e) {
    const cuuint64_t dims[3]    = { (cuuint64_t) row_bytes, (cuuint64_t) m, (cuuint64_t) n_expert };
    const cuuint64_t strides[2] = { (cuuint64_t) w_ld, (cuuint64_t) w_e };
    const cuuint32_t box[3]     = { (cuuint32_t) LDW, (cuuint32_t) glu_tma::ROWS, 1 };
    const cuuint32_t elem[3]    = { 1, 1, 1 };
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, (void *) base, dims, strides, box, elem,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// nvfp4_glu_tma for a grouped SwiGLU GEMM of 8 * NT tokens per tile; -1 when it does not apply
// (nothing launched).
template <int NT, bool HYB = false> int launch_glu_tma(const legion_nvfp4_mm & p, cudaStream_t st) {
    const uintptr_t align = (uintptr_t) p.w | (uintptr_t) p.w2 | (uintptr_t) p.x;
    const int64_t row_bytes = p.k / 64 * 36;
    if (!p.plan || !p.w2 || p.k % 64 != 0 || p.k <= 0 || p.m <= 0 || p.n_expert <= 0 || align % 16 != 0 ||
        p.w_ld % 16 != 0 || p.w_e % 16 != 0 || p.x_ld % 8 != 0 || row_bytes > p.w_ld || p.m * p.w_ld > p.w_e ||
        p.m > INT32_MAX || !glu_tma_code()) {
        return -1;
    }
    glu_tma::Maps maps;
    if (!glu_tma_map(&maps.w, p.w, row_bytes, p.m, p.n_expert, p.w_ld, p.w_e) ||
        !glu_tma_map(&maps.w2, p.w2, row_bytes, p.m, p.n_expert, p.w_ld, p.w_e)) {
        return -1;
    }
    constexpr int SMEM = glu_tma::smem_bytes<NT>();
    auto kernel = nvfp4_glu_tma<NT, HYB>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    kernel<<<(unsigned) p.tiles_m * (unsigned) p.tiles_n, 256, SMEM, st>>>(p, maps);
    return status();
}
