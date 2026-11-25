// sm_90 and newer: the routed experts' fused SwiGLU GEMM of prompt steps (nvfp4_gemm's grouped GLU
// tile of 128 output rows by 128 tokens, #1183) with a producer warp. Not a standalone header:
// nvfp4_mm/gemm_glu.cu includes it inside its anonymous namespace after common.cuh (BK, LDW,
// LDX, stage_bytes, cp_async16(_n), scale_bf16x2, codes_bf16x2, mul_bf16x2, mma, glu_gate,
// find_tile, gemm, pdl_code, status) and sm90/tma.cuh (the mbarrier helpers).
//
// nvfp4_gemm runs one CTA of 8 warps per SM (its ~250 registers a thread fill the register
// file), each warp both copying a third of the next stage and computing its own 32 rows of the
// current one, with a CTA barrier every k step; so the stream (copies, their waits) and the MMAs
// mostly add up instead of overlapping (#1176 r35: the stream alone 73% of the kernel's time,
// copies left out 64%), and each tile's lookup, first stages and epilogue run with no stream
// under them. Here a ninth warp does all of the copying: it finds the tiles (find_tile, a
// tile's pairs' row offsets into shared memory, two tiles' worth), issues each k step's weight
// and activation copies into a ring of three stages on mbarriers (full: the copies landed; empty:
// the eight compute warps are done with the stage) and keeps the weights' L2 prefetch; the eight
// warps only wait, decode and multiply, and arrive. The CTA is persistent (one per SM, tiles in
// launch order, row tiles fastest, as nvfp4_gemm's grid), so the next tile's lookup and first
// stages are in flight under a tile's last steps and epilogue.
//
// Bits: every compute warp runs nvfp4_gemm's GLU tile code on the same shared-memory layout (the
// weight words are copied by the same 16-byte cp.async copies with the same byte counts; the
// activation rows land at the same places), so each output is the same m16n8k16 chain over k in
// the same order, from zero, with the same A decode, B fragments and epilogue: nvfp4_gemm's bits.
// XBULK: the activation rows by cp.async.bulk (one 128-byte copy per token row and step, on the
// TMA unit) instead of 16-byte cp.async copies; the token rows past a partial tile's last are
// then not zeroed but hold an earlier stage's values, which only feed their own (never stored)
// output columns.
//
// Needs (the launch falls back to nvfp4_gemm otherwise): grouped (plan set), 16-byte aligned
// weights, rows (w_ld) and experts (w_e) and activations, no activation prefetch (xpf 0), and
// sm_90+ code for the device.

constexpr int GWS_ST = 3;                                   // ring stages (k steps of one block)
constexpr int GWS_WARPS = 8;                                // compute warps
constexpr int GWS_THREADS = (GWS_WARPS + 1) * 32;           // and the copy warp
constexpr int GWS_NT = 16, GWS_BN = 8 * GWS_NT;             // 128 tokens a tile
constexpr int GWS_BM = 16 * GWS_WARPS * 2;                  // 256 weight rows: 128 gate, 128 up
constexpr int GWS_W_STAGE = GWS_BM * LDW;                   // 12,288 bytes
constexpr int GWS_STAGE = stage_bytes<2, GWS_NT>();         // 30,720 bytes
constexpr int GWS_SMEM = smem_bytes<2, GWS_NT, GWS_ST>();   // 92,160 bytes
constexpr int GWS_WCOPIES = GWS_BM * 3 / 32;                // weight copies a lane and step: 24
constexpr int GWS_PF = 8;                                   // nvfp4_gemm's prefetch runs (blocks)
static_assert(GWS_BM * 3 % 32 == 0 && GWS_BN * 8 % 32 == 0, "copies");

template <bool XBULK>
__global__ void __launch_bounds__(GWS_THREADS, 1) nvfp4_glu_ws(const legion_nvfp4_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace legion_tma;
    constexpr int BN = GWS_BN, NT = GWS_NT, MI = 2, ST = GWS_ST, BM_OUT = 128;
    extern __shared__ __align__(16) unsigned char smem[];
    // Two tiles' worth: the copy warp fills one while the compute warps finish the other.
    __shared__ int64_t x_off[2][BN], d_off[2][BN];
    __shared__ int32_t info_expert[2], info_tokens[2], info_m0[2];
    __shared__ float info_sc[2], info_sc2[2];
    __shared__ __align__(8) uint64_t full[ST], empty[ST], info_full[2], info_empty[2];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            bar_init(full + s, 33);  // the copy warp's 32 cp.async arrivals and its lane 0's
            bar_init(empty + s, GWS_WARPS);
        }
#pragma unroll
        for (int b = 0; b < 2; b++) {
            bar_init(info_full + b, 32);
            bar_init(info_empty + b, GWS_WARPS);
        }
        bar_init_fence();
    }
    __syncthreads();
    const int steps = (int) (p.k / BK);
    const int64_t total = (int64_t) p.tiles_m * p.tiles_n;

    if (warp == GWS_WARPS) {
        // The copy warp. Tile `ti` takes info slot ti % 2; ring step `it` stage it % ST.
        constexpr int TI = legion_nvfp4::plan_tile_index(BN);
        uint32_t ti = 0;
        int stage = 0;
        uint32_t phase = 0;
        for (int64_t L = blockIdx.x;; L += gridDim.x) {
            const int b = ti & 1;
            int64_t m0 = 0;
            int e = -1, first = 0, count = 0;
            if (L < total) {
                m0 = L % p.tiles_m * BM_OUT;
                const int tile = (int) (L / p.tiles_m);
                if (legion_nvfp4::past_plan_tiles<TI>(p, tile)) {
                    // nvfp4_gemm's zeros for the plan's skipped pairs (ids of -1), split the
                    // same way over the tiles past the plan's last (every later one is too).
                    const int routed = p.plan[p.n_expert];
                    const int skipped = (int) p.n - routed;
                    if (skipped > 0 && !p.unread_skips) {
                        const int tiles = p.plan[p.n_expert + 1 + p.n + TI];
                        const int spare = tile - tiles, n_spare = p.tiles_n - tiles;
                        const int64_t rows = min((int64_t) BM_OUT, p.m - m0);
                        for (int j = spare; j < skipped; j += n_spare) {
                            const int64_t off = (int64_t) p.plan[p.n_expert + 1 + routed + j] * p.m + m0;
                            for (int64_t r = lane; r < rows; r += 32) {
                                ((__nv_bfloat16 *) p.dst)[off + r] = __float2bfloat16_rn(0.0f);
                            }
                        }
                        continue;
                    }
                } else if (!find_tile(p.plan, p.n_expert, BN, tile, lane, p.sel_lo, p.sel_hi, e, first, count)) {
                    e = -1;  // past a pair-count selection's last tile: so is every later one
                }
            }
            // Publish the tile (or the end, e = -1) once the compute warps are done with the
            // tile two back.
            bar_wait(info_empty + b, ((ti >> 1) & 1) ^ 1);
            if (e >= 0) {
                for (int i = lane; i < count; i += 32) {
                    const int q = p.plan[p.n_expert + 1 + first + i];  // token * n_used + slot
                    const int t = q / p.n_used, slot = q % p.n_used;
                    x_off[b][i] = ((int64_t) t * p.n_x + (p.n_x == 1 ? 0 : slot)) * p.x_ld;
                    d_off[b][i] = (int64_t) q * p.m;
                }
            }
            if (lane == 0) {
                info_expert[b] = e;
                info_tokens[b] = count;
                info_m0[b] = (int32_t) m0;
                if (e >= 0) {
                    info_sc[b] = (p.scale ? p.scale[e] : 1.0f) * 1024.0f;  // 2^10 undoes the decoding's 2^-10
                    info_sc2[b] = (p.scale2 ? p.scale2[e] : 1.0f) * 1024.0f;
                }
            }
            __syncwarp();
            bar_arrive(info_full + b);  // each lane's own writes released
            ti++;
            if (e < 0) {
                break;
            }
            // This lane's weight copies, nvfp4_gemm's load(): copy c = lane + 32 i is word c % 3
            // of shared row c / 3's block; rows are 16-byte aligned, so a step's block starts
            // (step * 36) & 15 into its first word and only the third word is partial.
            const uint8_t * wg = p.w + (int64_t) e * p.w_e;
            const uint8_t * wu = p.w2 + (int64_t) e * p.w_e;
            const uint8_t * wsrc[GWS_WCOPIES];
            int wpart[GWS_WCOPIES];  // the word, or -1 for a row past m (zeros)
#pragma unroll
            for (int i = 0; i < GWS_WCOPIES; i++) {
                const int c = lane + 32 * i, r = c / 3, part = c % 3;
                const int64_t row = m0 + r / 32 * 16 + r % 16;
                const uint8_t * wt = r / 16 % 2 ? wu : wg;
                wsrc[i] = wt + (row < p.m ? row : 0) * p.w_ld + part * 16;
                wpart[i] = row < p.m ? part : -1;
            }
            // nvfp4_gemm's L2 prefetch of each weight row's next GWS_PF blocks, GWS_PF ahead.
            auto prefetch = [&](int step) {
                if (step >= steps) {
                    return;
                }
#pragma unroll
                for (int j = 0; j < GWS_BM / 32; j++) {
                    const int r = lane + 32 * j;
                    const int64_t row = m0 + r / 32 * 16 + r % 16;
                    if (row < p.m) {
                        const uint8_t * wt = r / 16 % 2 ? wu : wg;
                        const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + (int64_t) step * 36);
                        const uintptr_t end = a + min(GWS_PF, steps - step) * 36;
                        const uintptr_t a16 = a & ~(uintptr_t) 15;
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                                     "r"((uint32_t) ((end - a16 + 15) & ~(uintptr_t) 15)));
                    }
                }
            };
            prefetch(0);
            prefetch(GWS_PF);
            const int n_tok = count;
            for (int step = 0; step < steps; step++) {
                if (step % GWS_PF == 0 && step > 0) {
                    prefetch(step + GWS_PF);
                }
                bar_wait(empty + stage, phase ^ 1);
                unsigned char * ws = smem + stage * GWS_STAGE;
                __nv_bfloat16 * xs = (__nv_bfloat16 *) (ws + GWS_W_STAGE);
                uint64_t * bar = full + stage;
                if constexpr (XBULK) {
                    // The compute warps' reads of the stage (generic proxy) before the TMA's
                    // writes (async proxy).
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    if (lane == 0) {
                        bar_expect_tx(bar, (uint32_t) n_tok * (BK * 2));
                    }
                    __syncwarp();
                    for (int t = lane; t < n_tok; t += 32) {
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, "
                            "[%3];\n" ::"r"(saddr(xs + t * LDX)),
                            "l"(p.x + x_off[b][t] + (int64_t) step * BK), "r"(BK * 2), "r"(saddr(bar))
                            : "memory");
                    }
                } else {
                    if (lane == 0) {
                        bar_arrive(bar);
                    }
#pragma unroll 8
                    for (int i = 0; i < BN * 8 / 32; i++) {
                        const int c = lane + 32 * i;
                        const int t = c / 8, col = c % 8 * 8;
                        const bool ok = t < n_tok;
                        const int64_t off = ok ? x_off[b][t] : 0;
                        cp_async16(xs + t * LDX + col, p.x + off + (int64_t) step * BK + col, ok);
                    }
                }
                const int lo = step * 36 & 15;
#pragma unroll
                for (int i = 0; i < GWS_WCOPIES; i++) {
                    const int c = lane + 32 * i, r = c / 3;
                    const int bytes = wpart[i] < 0 ? 0 : wpart[i] < 2 ? 16 : lo + 4;
                    cp_async16_n(ws + r * LDW + c % 3 * 16, wsrc[i] + (step * 36 - lo), bytes);
                }
                // Arrives on the stage once this lane's copies above have landed.
                asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(saddr(bar)) : "memory");
                if (++stage == ST) {
                    stage = 0;
                    phase ^= 1;
                }
            }
        }
        return;
    }

    // The compute warps: nvfp4_gemm's GLU tile code on each stage.
    uint32_t ti = 0;
    int stage = 0;
    uint32_t phase = 0;
    for (;; ti++) {
        const int b = ti & 1;
        bar_wait(info_full + b, (ti >> 1) & 1);
        const int e = info_expert[b];
        if (e < 0) {
            return;
        }
        const int n_tok = info_tokens[b];
        const int64_t m0 = info_m0[b];
        const float sc = info_sc[b], sc2 = info_sc2[b];

        float acc[MI][NT][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[i][j][c] = 0.0f;

        // One stage's MMAs, nvfp4_gemm's mma_stage (its rows' offset in their first word: the
        // rows are 16-byte aligned).
        auto mma_stage = [&](const unsigned char * ws, const __nv_bfloat16 * xs, int step, auto full_tile) {
            constexpr bool FULL = decltype(full_tile)::value;
            const unsigned char * rows[MI][2];
            uint32_t ssc[MI][2];
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    rows[i][h] = ws + (warp * 16 * MI + i * 16 + h * 8 + lane / 4) * LDW + ((uint32_t) step * 36 & 15);
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
                for (int half = 0; half < NT / 8; half++) {
                    if (!FULL && half * 64 >= n_tok) {
                        break;
                    }
                    uint32_t bf[8][2];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (!FULL && half * 64 + j * 16 >= n_tok) {
                            break;
                        }
                        const int tok = half * 64 + j * 16 + lane % 8 + (lane / 16) * 8;
                        const __nv_bfloat16 * src = xs + tok * LDX + s * 16 + ((lane / 8) % 2) * 8;
                        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(bf[2 * j][0]), "=r"(bf[2 * j][1]), "=r"(bf[2 * j + 1][0]), "=r"(bf[2 * j + 1][1])
                            : "r"(saddr(src)));
                    }
#pragma unroll
                    for (int i = 0; i < MI; i++)
#pragma unroll
                        for (int j = 0; j < 8; j++) {
                            if (FULL || half * 64 + j * 8 < n_tok) {
                                mma(acc[i][half * 8 + j], a[i], bf[j][0], bf[j][1]);
                            }
                        }
                }
            }
        };

        for (int step = 0; step < steps; step++) {
            bar_wait(full + stage, phase);
            const unsigned char * ws = smem + stage * GWS_STAGE;
            const __nv_bfloat16 * xs = (const __nv_bfloat16 *) (ws + GWS_W_STAGE);
            if (n_tok == BN) {
                mma_stage(ws, xs, step, std::true_type{});
            } else {
                mma_stage(ws, xs, step, std::false_type{});
            }
            // This warp's reads of the stage are done (their values are in the MMAs above).
            __syncwarp();
            if (lane == 0) {
                bar_arrive(empty + stage);
            }
            if (++stage == ST) {
                stage = 0;
                phase ^= 1;
            }
        }

        // nvfp4_gemm's GLU epilogue.
        const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int64_t row = r0 + (c / 2) * 8;
                const int t = j * 8 + (lane % 4) * 2 + c % 2;
                if (row < p.m && t < n_tok) {
                    float gate = acc[0][j][c] * sc, up = acc[1][j][c] * sc2;
                    if (p.limit > 0.0f) {
                        gate = fminf(gate, p.limit);
                        up = fminf(fmaxf(up, -p.limit), p.limit);
                    }
                    ((__nv_bfloat16 *) p.dst)[d_off[b][t] + row] = __float2bfloat16_rn(glu_gate(gate, p.act) * up);
                }
            }
        }
        __syncwarp();
        if (lane == 0) {
            bar_arrive(info_empty + b);
        }
    }
#else
    __trap();  // glu_ws() launches it with sm_90+ code only
#endif
}

// ---------------------------------------------------------------------------------------------
// nvfp4_glu_ws2: the multiply warps copy their own weight rows (#1183, second round).
//
// On the GB10 nvfp4_glu_ws ran at 0.80x nvfp4_gemm's speed with 16-byte activation copies and
// 0.45x with bulk ones: one warp issuing a whole stage (24 weight + 32 activation copies a lane,
// each step) cannot keep 30 KB a step in flight, and the TMA's 128-byte bulk copies (128 a stage)
// are slower still. So here the copies stay spread over many warps:
//  - Weights: each multiply warp copies its own 32 shared rows (nvfp4_gemm's load(), the same
//    words and byte counts: 3 copies a lane and step), two steps ahead into its own part of the
//    ring. Only that warp reads those rows, so it waits for them on its own cp.async groups
//    (wait_group, then __syncwarp), with no cross-warp barrier. The weights' L2 prefetch too
//    (one row a lane, nvfp4_gemm's runs of 8 blocks).
//  - Activations (shared by all eight warps): XW = 0, by the multiply warps as in nvfp4_gemm
//    (4 copies a lane and step), each waiting on `empty` (all eight warps done with the stage it
//    refills, a step back) instead of a CTA barrier every step, the stage's copies tracked by
//    `full` (256 arrivals); XW = 1, 2, 4, by that many copy warps (32, 16 or 8 copies a lane
//    and step) on `full` / `empty`, the multiply warps issuing only their weights.
//  - Warp 8 finds the tiles (find_tile, two tiles of row offsets in shared memory) and stores
//    the skipped pairs' zeros, a tile ahead, so the next tile's first copies go out (from the
//    multiply warps, by their global step) while a tile's last steps and epilogue run. With
//    XW >= 1 it is also the first copy warp, looking up the next tile after its third step.
// One CTA per SM loops over the tiles (row tiles fastest, nvfp4_gemm's launch order). The ring's
// stage of global step j (every warp counts tile * steps + step) is j % 3; the copies of j + 2 go
// out at j, into the stage of j - 1.
//
// Bits: the same shared-memory layout, copies and mma_stage as nvfp4_gemm (and nvfp4_glu_ws), so
// nvfp4_gemm's bits. Needs two steps or more (k >= 128) besides nvfp4_glu_ws's needs.

template <int XW> __host__ __device__ constexpr int gws2_threads() {
    return (GWS_WARPS + (XW > 1 ? XW : 1)) * 32;
}

// One stage's MMAs of a multiply warp: nvfp4_gemm's mma_stage for MI = 2, NT = 16 (rows 16-byte
// aligned, so a step's block starts (step * 36) & 15 into its first word).
template <bool FULL>
__device__ __forceinline__ void gws_mma_stage(float (&acc)[2][GWS_NT][4], const unsigned char * ws,
                                              const __nv_bfloat16 * xs, int step, int n_tok, int warp, int lane) {
    constexpr int MI = 2, NT = GWS_NT;
    const unsigned char * rows[MI][2];
    uint32_t ssc[MI][2];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            rows[i][h] = ws + (warp * 16 * MI + i * 16 + h * 8 + lane / 4) * LDW + ((uint32_t) step * 36 & 15);
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
        for (int half = 0; half < NT / 8; half++) {
            if (!FULL && half * 64 >= n_tok) {
                break;
            }
            uint32_t bf[8][2];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (!FULL && half * 64 + j * 16 >= n_tok) {
                    break;
                }
                const int tok = half * 64 + j * 16 + lane % 8 + (lane / 16) * 8;
                const __nv_bfloat16 * src = xs + tok * LDX + s * 16 + ((lane / 8) % 2) * 8;
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(bf[2 * j][0]), "=r"(bf[2 * j][1]), "=r"(bf[2 * j + 1][0]), "=r"(bf[2 * j + 1][1])
                    : "r"((uint32_t) __cvta_generic_to_shared(src)));
            }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (FULL || half * 64 + j * 8 < n_tok) {
                        mma(acc[i][half * 8 + j], a[i], bf[j][0], bf[j][1]);
                    }
                }
        }
    }
}

template <int XW>
__global__ void __launch_bounds__(gws2_threads<XW>(), 1) nvfp4_glu_ws2(const legion_nvfp4_mm p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace legion_tma;
    static_assert(XW == 0 || XW == 1 || XW == 2 || XW == 4, "copy warps");
    constexpr int BN = GWS_BN, NT = GWS_NT, ST = GWS_ST, BM_OUT = 128;
    constexpr int XLANES = (XW > 0 ? XW : GWS_WARPS) * 32;  // the lanes copying activations
    constexpr int XCOPIES = BN * 8 / XLANES;                 // their copies a lane and step
    static_assert(BN * 8 % XLANES == 0, "activation copies");
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int64_t x_off[2][BN], d_off[2][BN];
    __shared__ int32_t info_expert[2], info_tokens[2], info_m0[2];
    __shared__ float info_sc[2], info_sc2[2];
    __shared__ __align__(8) uint64_t full[ST], empty[ST], info_full[2], info_empty[2];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < ST; s++) {
            bar_init(full + s, XLANES);  // each activation-copying lane's cp.async arrival
            bar_init(empty + s, GWS_WARPS);
        }
#pragma unroll
        for (int b = 0; b < 2; b++) {
            bar_init(info_full + b, 32);
            // The multiply warps and the copy warps past warp 8 (warp 8 writes the slots).
            bar_init(info_empty + b, GWS_WARPS + (XW > 1 ? XW - 1 : 0));
        }
        bar_init_fence();
    }
    __syncthreads();
    const int steps = (int) (p.k / BK);
    const int64_t total = (int64_t) p.tiles_m * p.tiles_n;

    // This lane's activation copies of local step `step` of the tile in info slot `b` into
    // `stage` (nvfp4_gemm's load(): copy c = xl + XLANES i is token row c / 8's 16 bytes c % 8;
    // rows past the tile's tokens zeroed).
    auto copy_x = [&](int xl, int b, int n_tok, int step, int stage) {
        __nv_bfloat16 * xs = (__nv_bfloat16 *) (smem + stage * GWS_STAGE + GWS_W_STAGE);
#pragma unroll
        for (int i = 0; i < XCOPIES; i++) {
            const int c = xl + XLANES * i;
            const int t = c / 8, col = c % 8 * 8;
            const bool ok = t < n_tok;
            const int64_t off = ok ? x_off[b][t] : 0;
            cp_async16(xs + t * LDX + col, p.x + off + (int64_t) step * BK + col, ok);
        }
    };
    auto arrive_copies = [&](int stage) {
        // Arrives on the stage once this lane's copies above have landed.
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(saddr(full + stage)) : "memory");
    };

    if (warp >= GWS_WARPS) {
        constexpr int TI = legion_nvfp4::plan_tile_index(BN);
        int64_t L = blockIdx.x;  // warp 8: the next tile (launch order) to look at
        // Warp 8: the CTA's tile `ti` (its expert, or -1: no more) into info slot ti % 2, once
        // the multiply warps (and the other copy warps) are done with tile ti - 2.
        auto publish = [&](uint32_t ti) -> int {
            const int b = ti & 1;
            int64_t m0 = 0;
            int e = -1, first = 0, count = 0;
            for (; L < total; L += gridDim.x) {
                m0 = L % p.tiles_m * BM_OUT;
                const int tile = (int) (L / p.tiles_m);
                if (legion_nvfp4::past_plan_tiles<TI>(p, tile)) {
                    // nvfp4_gemm's zeros for the plan's skipped pairs (ids of -1), split the same
                    // way over the tiles past the plan's last (every later one is too).
                    const int routed = p.plan[p.n_expert];
                    const int skipped = (int) p.n - routed;
                    if (skipped > 0 && !p.unread_skips) {
                        const int tiles = p.plan[p.n_expert + 1 + p.n + TI];
                        const int spare = tile - tiles, n_spare = p.tiles_n - tiles;
                        const int64_t rows = min((int64_t) BM_OUT, p.m - m0);
                        for (int j = spare; j < skipped; j += n_spare) {
                            const int64_t off = (int64_t) p.plan[p.n_expert + 1 + routed + j] * p.m + m0;
                            for (int64_t r = lane; r < rows; r += 32) {
                                ((__nv_bfloat16 *) p.dst)[off + r] = __float2bfloat16_rn(0.0f);
                            }
                        }
                        continue;
                    }
                    break;
                }
                if (find_tile(p.plan, p.n_expert, BN, tile, lane, p.sel_lo, p.sel_hi, e, first, count)) {
                    L += gridDim.x;
                } else {
                    e = -1;  // past a pair-count selection's last tile: so is every later one
                }
                break;
            }
            bar_wait(info_empty + b, ((ti >> 1) & 1) ^ 1);
            if (e >= 0) {
                for (int i = lane; i < count; i += 32) {
                    const int q = p.plan[p.n_expert + 1 + first + i];  // token * n_used + slot
                    const int t = q / p.n_used, slot = q % p.n_used;
                    x_off[b][i] = ((int64_t) t * p.n_x + (p.n_x == 1 ? 0 : slot)) * p.x_ld;
                    d_off[b][i] = (int64_t) q * p.m;
                }
            }
            if (lane == 0) {
                info_expert[b] = e;
                info_tokens[b] = count;
                info_m0[b] = (int32_t) m0;
                if (e >= 0) {
                    info_sc[b] = (p.scale ? p.scale[e] : 1.0f) * 1024.0f;  // 2^10 undoes the decoding's 2^-10
                    info_sc2[b] = (p.scale2 ? p.scale2[e] : 1.0f) * 1024.0f;
                }
            }
            __syncwarp();
            bar_arrive(info_full + b);  // each lane's own writes released
            return e;
        };
        if constexpr (XW == 0) {
            for (uint32_t ti = 0; publish(ti) >= 0; ti++) {
            }
        } else {
            const int xl = (warp - GWS_WARPS) * 32 + lane;
            const bool lookup = warp == GWS_WARPS;
            int e = lookup ? publish(0) : 0;
            for (uint32_t ti = 0;; ti++) {
                const int b = ti & 1;
                if (!lookup) {
                    bar_wait(info_full + b, (ti >> 1) & 1);
                    e = info_expert[b];
                }
                if (e < 0) {
                    return;
                }
                const int n_tok = info_tokens[b];
                int e_next = -1;
                for (int s = 0; s < steps; s++) {
                    const uint32_t j = ti * (uint32_t) steps + s;
                    const int stage = j % ST;
                    bar_wait(empty + stage, ((j / ST) & 1) ^ 1);
                    copy_x(xl, b, n_tok, s, stage);
                    arrive_copies(stage);
                    if (lookup && s == min(2, steps - 1)) {
                        e_next = publish(ti + 1);  // the next tile, once three stages are out
                    }
                }
                if (lookup) {
                    e = e_next;
                } else {
                    __syncwarp();
                    if (lane == 0) {
                        bar_arrive(info_empty + b);  // done reading the slot's row offsets
                    }
                }
            }
        }
        return;
    }

    // The multiply warps.
    // This lane's weight copies of local step `step` of a tile (expert e, rows from m0) into
    // `stage`: copy c = 96 warp + lane + 32 i (i < 3) is word c % 3 of shared row c / 3, one of
    // this warp's 32 rows (nvfp4_gemm's load(), the same words and byte counts). With the weights'
    // L2 prefetch (row warp * 32 + lane, nvfp4_gemm's) at the tile's first step and every
    // GWS_PF steps.
    auto copy_w = [&](int e, int64_t m0, int step, int stage) {
        unsigned char * ws = smem + stage * GWS_STAGE;
        const uint8_t * wg = p.w + (int64_t) e * p.w_e;
        const uint8_t * wu = p.w2 + (int64_t) e * p.w_e;
        const int lo = step * 36 & 15;
#pragma unroll
        for (int i = 0; i < 3; i++) {
            const int c = warp * 96 + lane + 32 * i, r = c / 3, part = c % 3;
            const int64_t row = m0 + r / 32 * 16 + r % 16;
            const uint8_t * wt = r / 16 % 2 ? wu : wg;
            const bool ok = row < p.m;
            const int bytes = !ok ? 0 : part < 2 ? 16 : lo + 4;
            cp_async16_n(ws + r * LDW + part * 16, wt + (ok ? row : 0) * p.w_ld + (step * 36 - lo) + part * 16, bytes);
        }
        auto prefetch = [&](int at) {
            const int64_t row = m0 + warp * 16 + lane % 16;
            if (at < steps && row < p.m) {
                const uint8_t * wt = lane / 16 ? wu : wg;
                const uintptr_t a = (uintptr_t) (wt + row * p.w_ld + (int64_t) at * 36);
                const uintptr_t end = a + min(GWS_PF, steps - at) * 36;
                const uintptr_t a16 = a & ~(uintptr_t) 15;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a16),
                             "r"((uint32_t) ((end - a16 + 15) & ~(uintptr_t) 15)));
            }
        };
        if (step == 0) {
            prefetch(0);
            prefetch(GWS_PF);
        } else if (step % GWS_PF == 0) {
            prefetch(step + GWS_PF);
        }
    };
    // The copies of global step j (local step `step` of the tile in slot b) into stage j % 3.
    auto copy_step = [&](int e, int64_t m0, int b, int n_tok, int step, uint32_t j) {
        const int stage = j % ST;
        copy_w(e, m0, step, stage);  // this warp's own rows: free once it is done with them
        if constexpr (XW == 0) {
            // The activation rows are every warp's: all eight done with the stage (step j - 3).
            bar_wait(empty + stage, ((j / ST) & 1) ^ 1);
            copy_x(warp * 32 + lane, b, n_tok, step, stage);
            arrive_copies(stage);  // tracks the weight copies above too
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    };

    bar_wait(info_full, 0);
    int e = info_expert[0];
    if (e < 0) {
        return;
    }
    int n_tok = info_tokens[0];
    int64_t m0 = info_m0[0];
    copy_step(e, m0, 0, n_tok, 0, 0);
    copy_step(e, m0, 0, n_tok, 1, 1);
    uint32_t g = 0;  // the global step
    for (uint32_t ti = 0;; ti++) {
        const int b = ti & 1;
        const float sc = info_sc[b], sc2 = info_sc2[b];
        float acc[2][NT][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[i][j][c] = 0.0f;
        int e_next = -1, n_next = 0;
        int64_t m0_next = 0;
        for (int s = 0; s < steps; s++, g++) {
            // The copies two steps ahead: this tile's, or the next one's first two.
            if (s + 2 < steps) {
                copy_step(e, m0, b, n_tok, s + 2, g + 2);
            } else {
                if (s + 2 == steps) {
                    bar_wait(info_full + (b ^ 1), ((ti + 1) >> 1) & 1);
                    e_next = info_expert[b ^ 1];
                    n_next = info_tokens[b ^ 1];
                    m0_next = info_m0[b ^ 1];
                }
                if (e_next >= 0) {
                    copy_step(e_next, m0_next, b ^ 1, n_next, s + 2 - steps, g + 2);
                } else if constexpr (XW > 0) {
                    asm volatile("cp.async.commit_group;\n" ::);  // one group a step
                }
            }
            const int stage = g % ST;
            if constexpr (XW > 0) {
                // This warp's own weight rows of step g (its groups past g + 1 still in flight).
                asm volatile("cp.async.wait_group 2;\n" ::: "memory");
                __syncwarp();
            }
            bar_wait(full + stage, (g / ST) & 1);
            const unsigned char * ws = smem + stage * GWS_STAGE;
            const __nv_bfloat16 * xs = (const __nv_bfloat16 *) (ws + GWS_W_STAGE);
            if (n_tok == BN) {
                gws_mma_stage<true>(acc, ws, xs, s, n_tok, warp, lane);
            } else {
                gws_mma_stage<false>(acc, ws, xs, s, n_tok, warp, lane);
            }
            // This warp's reads of the stage are done (their values are in the MMAs above).
            __syncwarp();
            if (lane == 0) {
                bar_arrive(empty + stage);
            }
        }

        // nvfp4_gemm's GLU epilogue.
        const int64_t r0 = m0 + warp * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < NT; j++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int64_t row = r0 + (c / 2) * 8;
                const int t = j * 8 + (lane % 4) * 2 + c % 2;
                if (row < p.m && t < n_tok) {
                    float gate = acc[0][j][c] * sc, up = acc[1][j][c] * sc2;
                    if (p.limit > 0.0f) {
                        gate = fminf(gate, p.limit);
                        up = fminf(fmaxf(up, -p.limit), p.limit);
                    }
                    ((__nv_bfloat16 *) p.dst)[d_off[b][t] + row] = __float2bfloat16_rn(glu_gate(gate, p.act) * up);
                }
            }
        }
        __syncwarp();
        if (lane == 0) {
            bar_arrive(info_empty + b);
        }
        if (e_next < 0) {
            if constexpr (XW > 0) {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            return;
        }
        e = e_next;
        n_tok = n_next;
        m0 = m0_next;
    }
#else
    __trap();  // glu_ws() launches it with sm_90+ code only
#endif
}

// GLU_WS_CODE + 1000 * v: nvfp4_glu_ws's variants (GemmTune::glu_ws in src/model/nvfp4_mm.rs).
// v 1: activations by cp.async.bulk, one CTA per SM (persistent); 2: activations by 16-byte
// cp.async, persistent; 3: as 1 with a CTA per tile (nvfp4_gemm's grid). v 10 + XW:
// nvfp4_glu_ws2<XW> (the multiply warps copy their own weight rows; XW copy warps for the
// activation rows, 0: the multiply warps copy them too), persistent. The same bits as
// nvfp4_gemm, which takes the launches the kernels do not (see above).
constexpr int GLU_WS_CODE = 98;

// `kernel` (`threads` a CTA, one CTA per SM if `persistent`, else one per tile) where it takes
// the launch, nvfp4_gemm otherwise.
template <typename K>
int glu_ws_launch(const legion_nvfp4_mm & p, K kernel, int threads, bool persistent, int min_steps, cudaStream_t st) {
    const bool fits = p.plan && p.xpf == 0 && p.k / BK >= min_steps && p.w_ld % 16 == 0 && p.w_e % 16 == 0 &&
                      p.x_ld % 8 == 0 && ((uintptr_t) p.w | (uintptr_t) p.w2 | (uintptr_t) p.x) % 16 == 0 &&
                      pdl_code(kernel);
    if (!fits) {
        return gemm<2, GWS_NT, true>(p, st);
    }
    const cudaError_t before = cudaPeekAtLastError();
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GWS_SMEM) != cudaSuccess) {
        if (before == cudaSuccess) {
            cudaGetLastError();
        }
        return gemm<2, GWS_NT, true>(p, st);
    }
    const int64_t total = (int64_t) p.tiles_m * p.tiles_n;
    if (total <= 0) {
        return status();
    }
    const int sms = legion_cuda_sm_count();
    const int64_t grid = persistent && sms > 0 && sms < total ? sms : total;
    kernel<<<(unsigned) grid, threads, GWS_SMEM, st>>>(p);
    return status();
}

template <bool XBULK> int glu_ws1(const legion_nvfp4_mm & p, bool persistent, cudaStream_t st) {
    return glu_ws_launch(p, nvfp4_glu_ws<XBULK>, GWS_THREADS, persistent, 1, st);
}

template <int XW> int glu_ws2(const legion_nvfp4_mm & p, cudaStream_t st) {
    return glu_ws_launch(p, nvfp4_glu_ws2<XW>, gws2_threads<XW>(), true, 2, st);
}

// -2: not an nvfp4_glu_ws variant; -1: no such variant.
int glu_ws(const legion_nvfp4_mm & p, int variant, cudaStream_t st) {
    if (variant < 1000 || variant % 1000 != GLU_WS_CODE) {
        return -2;
    }
    switch (variant / 1000) {
        case 1: return glu_ws1<true>(p, true, st);
        case 2: return glu_ws1<false>(p, true, st);
        case 3: return glu_ws1<true>(p, false, st);
        case 10: return glu_ws2<0>(p, st);
        case 11: return glu_ws2<1>(p, st);
        case 12: return glu_ws2<2>(p, st);
        case 14: return glu_ws2<4>(p, st);
        default: return -1;
    }
}
