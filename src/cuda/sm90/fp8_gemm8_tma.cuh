// fp8_mm.cu, sm_90 and newer: fp8_gemm8 with its stages filled by the tensor memory accelerator
// (TMA) instead of per-thread cp.async copies (#1133).
//
// Not a standalone header: fp8_mm.cu includes it inside its anonymous namespace, after
// sm89/fp8_mm.cuh (a8::Cfg, gemm8_stage, gemm8_epilogue). The kernel body is built only where
// __CUDA_ARCH__ >= 900 and traps elsewhere.
//
// fp8_gemm8 spends a large part of each K step issuing its copies: 12 cp.async of 16 bytes per
// thread for the 256 x 128 tile, each with its own address and bounds check, then a CTA barrier.
// Here two TMA tile copies per stage (the TM weight rows and the BN tokens of BK bytes) complete
// on the stage's "full" mbarrier, and each warp counts itself out of the stage when its ldmatrix
// reads of it are done; the last warp out refills it, so warps no longer wait for each other at
// every step (#1133 r5: the refill was thread 0's, which waited for the slowest warp on an
// "empty" mbarrier). The TMA writes each tile in its 64- or 128-byte swizzle (BK 64 or
// 128), which is exactly a8::Cfg's SWZ layout (16-byte chunk c of row r at c ^ (r / (128 / BK)
// % (BK / 16))) when a stage starts on a 1024-byte boundary, so the MMA code is fp8_gemm8's,
// and rows past m or tokens past n are filled with zeros as cp.async's zero-size copies do:
// every output takes the same MMAs in the same k order, the same bits.

#include "tma.cuh"

namespace a8t {
// The tensor maps of a launch: the weights [m rows of k bytes] and the activations [n tokens
// of k bytes], boxes of BK bytes x TM rows / BN tokens in the stage's swizzle; with the TMA
// store epilogue (gemm8_store_tma) also the F32 outputs y [n tokens of m rows], boxes of
// ST_ROWS rows x ST_TOKS tokens in the 128-byte swizzle (else unset).
struct Maps {
    CUtensorMap w, x, y;
};
// gemm8_store_tma's store box: 32 rows (128 bytes, the 128-byte swizzle's span) of 16 tokens,
// and the boxes of a warp in flight: 3 quarters (16 tokens of its 64 rows, two boxes) of its 64
// tokens, 12 KiB, so the wide tile's 8 warps take its two stages' 96 KiB.
constexpr int ST_ROWS = 32, ST_TOKS = 16, ST_BUFS = 3;
constexpr int ST_QUARTER_FLOATS = 2 * ST_ROWS * ST_TOKS;
constexpr int ST_WARP_FLOATS = ST_BUFS * ST_QUARTER_FLOATS;
// The first wave's staggered start: CTA i < ctas (in launch order) waits i * ns nanoseconds
// before its first copy. ctas 0: none.
struct Stagger {
    int ctas = 0;
    uint32_t ns = 0;
};
}  // namespace a8t

// Shared memory of fp8_gemm8_tma: the stages, 1 KiB to align them, and a full mbarrier and a
// release count per stage. 4 warps (two CTAs per SM in its 100 KB): no alignment slack, as the
// dynamic shared memory of a kernel without static shared memory starts on a 1 KiB boundary
// (checked in the kernel).
template <class C> __host__ __device__ constexpr size_t gemm8_tma_smem() {
    return C::SMEM + (C::WARPS == 4 ? 0 : 1024) + 2 * C::STAGES * sizeof(uint64_t);
}

// legion_fp8_mm8's variant bit 9: the outputs go to mapped host memory (the TP send buffer,
// #1133), through gemm8_epilogue_staged.
constexpr int32_t GEMM8_STAGED = 0x200;
// Bit 11 (#1183): the wide tile without a residual (8 warps, 256 x 128) stores its outputs by
// TMA (gemm8_store_tma) instead of from the fragments; bit 12 the same in the qkv GEMM's conv
// epilogue (fp8_gemm8_tma_conv); bit 13 (`legion fp8-gemm-bench` only) the wide tile's first
// wave starts staggered as the 4-warp tile's does. Every value gives the same bits.
constexpr int32_t GEMM8_TMA_STORE = 0x800;
constexpr int32_t GEMM8_CONV_TMA_STORE = 0x1000;
constexpr int32_t GEMM8_WIDE_STAGGER = 0x2000;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// The tile's outputs without a residual (gemm8_epilogue's values: each sum times a8_scale) by
// whole runs of a token's rows (#1133). The MMA layout stores 4 bytes a lane, 32 bytes of each
// of 4 tokens a store; into device memory L2 merges those pieces into lines, but into mapped
// host memory on the GB10 each piece goes out alone, at about half the speed of whole lines (a
// 168 MB F32 output: ~94 against ~180 GB/s). Here each warp puts one n8 token tile of its 64
// rows at a time into its own piece of shared memory (`stage`, 8 x 68 floats: the stages' memory,
// free once every warp is past the K loop) and writes it 16 bytes a lane, two tokens' 256-byte
// runs a store. The same values at the same places: the same bits.
template <class C, int NJ>
__device__ __forceinline__ void gemm8_epilogue_staged(const legion_fp8_mm8 & p, int64_t m0, int64_t n0, int wm,
                                                      int wn, int lane, const float (&acc)[4][NJ][4],
                                                      float * stage) {
    static_assert(NJ == a8::BN / 8 / C::WN, "n8 tiles per warp");
    constexpr int LDS = 64 + 4;  // a token's 64 rows, padded: the lanes' writes hit 32 banks
    const float s = a8_static(p);
    const int g = lane / 4, t = lane % 4;
    const int64_t rbase = m0 + wm * 64, tbase = n0 + wn * NJ * 8;
    // All 64 rows exist and 16-byte stores are aligned.
    const bool whole = rbase + 64 <= p.m && p.m % 4 == 0 && ((uintptr_t) p.dst & 15) == 0;
#pragma unroll
    for (int j = 0; j < NJ; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = i * 16 + g + (e / 2) * 8, tk = 2 * t + e % 2;
                const int64_t row = rbase + r, tok = tbase + j * 8 + tk;
                stage[tk * LDS + r] = row < p.m && tok < p.n ? acc[i][j][e] * a8_scale(p, s, row, tok) : 0.0f;
            }
        __syncwarp();
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int idx = q * 32 + lane, tk = idx / 16, c = idx % 16;
            const int64_t tok = tbase + j * 8 + tk;
            if (tok < p.n) {
                float * o = p.dst + tok * p.m + rbase + c * 4;
                const float * v = stage + tk * LDS + c * 4;
                if (whole) {
                    __stcs((float4 *) o, *(const float4 *) v);
                } else {
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        if (rbase + c * 4 + u < p.m) {
                            __stcs(o + u, v[u]);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }
}

// The wide tile's sums times their scales in place (#1183, for gemm8_store_tma): each value
// gemm8_epilogue_stream stores (the sum times a8_scale's factor: s0, else the token's scale
// times the row's, or the token's), so the same bits. Tokens past n keep a factor of s0.
template <class C, int NJ>
__device__ __forceinline__ void gemm8_scale_acc(const legion_fp8_mm8 & p, int64_t m0, int64_t n0, int wm, int wn,
                                                int lane, float (&acc)[4][NJ][4]) {
    static_assert(NJ == a8::BN / 8 / C::WN, "n8 tiles per warp");
    const float s = a8_static(p);
    const bool uni = p.scale_stride == 0 && p.wscale_stride == 0;
    const bool rowsc = !uni && p.wscale != nullptr;
    const int64_t rbase = m0 + wm * 64 + lane / 4;  // row of i 0, e 0
    float wr[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int64_t row = rbase + i * 16 + h * 8;
            wr[i][h] = rowsc && row < p.m ? p.wscale[row * p.wscale_stride] : 0.0f;
        }
    const int64_t tbase = n0 + wn * NJ * 8 + (lane % 4) * 2;  // token of j 0, e 0
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
            const int64_t tok = tbase + j * 8 + e2;
            const float sc = !uni && tok < p.n ? p.scale[tok * p.scale_stride] : s;
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const float f = uni ? s : rowsc ? sc * wr[i][h] : sc;
                    acc[i][j][2 * h + e2] = acc[i][j][2 * h + e2] * f;
                }
        }
}

// A warp's 64 x 64 piece of the wide tile's outputs (#1183): v[i][j][e] (the MMA layout) is the
// value of row row0 + 16 i + g + 8 (e / 2) and token tok0 + 8 j + 2 t + e % 2 (g = lane / 4,
// t = lane % 4) in ymap's tensor (F32 [tokens of rows], a8t::Maps::y). The fragments' own stores
// write 4 bytes a lane, 32-byte pieces of 4 tokens' rows a store instruction, 128 of them a warp;
// here the warp puts 16 tokens at a time (a quarter: two boxes of 32 rows) into its own 4 KiB of
// shared memory `buf` (a8t::ST_WARP_FLOATS, the stages' memory, free once every warp is past
// the K loop; 1 KiB aligned) in the TMA's 128-byte swizzle (16-byte chunk c of a box's token u
// at c ^ (u % 8): the lanes' 32 writes of a value hit 32 banks), and one lane hands the boxes to
// the TMA unit, which writes whole 128-byte lines and clips tokens past the tensor's. The warp
// fills the next quarter while those are read; the fourth reuses the first one's memory once
// the TMA has read it. The same values at the same places: the same bits. skip0: the first
// quarter is the caller's to store (not stored here).
__device__ __forceinline__ void gemm8_store_tma(const CUtensorMap * ymap, int64_t row0, int64_t tok0, int64_t n,
                                                int lane, const float (&v)[4][8][4], float * buf, bool skip0) {
    using namespace legion_tma;
    using a8t::ST_BUFS;
    using a8t::ST_QUARTER_FLOATS;
    using a8t::ST_ROWS;
    using a8t::ST_TOKS;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        // Quarters past the tokens are skipped (then so are the later ones: the bulk groups
        // stay in quarter order).
        if ((q == 0 && skip0) || tok0 + q * ST_TOKS >= n) {
            continue;
        }
        float * b = buf + (q % ST_BUFS) * ST_QUARTER_FLOATS;
        if (q >= ST_BUFS) {
            // Quarter q - ST_BUFS's boxes read before their memory takes this one.
            if (lane == 0) {
                bulk_wait_read<ST_BUFS - 1>();
            }
            __syncwarp();
        }
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = (i % 2) * 16 + (e / 2) * 8 + g;  // row in its box
                    const int u = jj * 8 + 2 * t + e % 2;           // token in the box
                    b[(i / 2) * ST_ROWS * ST_TOKS + u * ST_ROWS + ((r / 4) ^ (u % 8)) * 4 + r % 4] =
                        v[i][2 * q + jj][e];
                }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
            const int c1 = (int) (tok0 + q * ST_TOKS);
            tma_store_2d(ymap, b, (int) row0, c1);
            tma_store_2d(ymap, b + ST_ROWS * ST_TOKS, (int) row0 + ST_ROWS, c1);
            bulk_commit();
        }
    }
    // The shared memory stays the CTA's until the TMA has read it.
    if (lane == 0) {
        bulk_wait_read<0>();
    }
    __syncwarp();
}

// fp8_gemm8_tma's dynamic shared memory, aligned to 1 KiB.
template <class C> __device__ __forceinline__ unsigned char * gemm8_tma_smem_base() {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    if constexpr (C::WARPS == 4) {
        if (legion_tma::saddr(smem_raw) % 1024 != 0) {
            __trap();  // gemm8_tma_smem left no room to align
        }
    }
    return smem;
}

// fp8_gemm8_tma's K loop: the sums of the tile of TM rows from m0 and BN tokens from n0 into
// acc (the MMA layout of gemm8_stage), its stages in `smem` (gemm8_tma_smem_base).
template <bool RES, class C>
__device__ __forceinline__ void gemm8_tma_sums(const legion_fp8_mm8 & p, const a8t::Maps & maps,
                                               const a8t::Stagger & stagger, int64_t m0, int64_t n0,
                                               unsigned char * smem, int wm, int wn, int lane,
                                               float (&acc)[4][a8::BN / 8 / C::WN][4]) {
    using a8::BN;
    using namespace legion_tma;
    static_assert(C::SWZ && C::LD == C::BK && (C::BK == 64 || C::BK == 128), "TMA swizzle layout");
    constexpr int BK = C::BK, STAGES = C::STAGES;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM, WARPS = C::WARPS;
    constexpr size_t TSTAGE = C::TSTAGE;
    static_assert(TSTAGE % 1024 == 0 && (TM * BK) % 1024 == 0, "stages on swizzle boundaries");
    uint64_t * full = (uint64_t *) (smem + C::SMEM);
    uint32_t * released = (uint32_t *) (full + STAGES);
    const int64_t steps = p.k / BK;

    const int64_t lin = (int64_t) blockIdx.y * gridDim.x + blockIdx.x;
    if (lin < stagger.ctas) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        const uint64_t until = t + (uint64_t) lin * stagger.ns;
        while (t < until) {
            __nanosleep(256);
            asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        }
    }

    // Step `step`'s weights and activations into stage `stage` (one thread).
    auto issue = [&](int64_t step, int stage) {
        unsigned char * ws = smem + stage * TSTAGE;
        bar_expect_tx(full + stage, (uint32_t) TSTAGE);
        tma_2d(ws, &maps.w, (int) (step * BK), (int) m0, full + stage);
        tma_2d(ws + TM * BK, &maps.x, (int) (step * BK), (int) n0, full + stage);
    };
    if (threadIdx.x == 0) {
        prefetch_map(&maps.w);
        prefetch_map(&maps.x);
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            bar_init(full + s, 1);
            released[s] = 0;
        }
        bar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            if (s < steps) {
                issue(s, s);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    for (int64_t step = 0; step < steps; step++) {
        const int stage = (int) (step % STAGES);
        const uint32_t parity = (uint32_t) (step / STAGES) & 1;
        bar_wait(full + stage, parity);
        gemm8_res_prefetch<RES, C>(p, step, steps, m0, n0);
        const unsigned char * ws = smem + stage * TSTAGE;
        gemm8_stage<C>(ws, ws + TM * BK, wm, wn, lane, acc);
        if (step + STAGES < steps) {
            // This warp's reads of the stage are done. The acq_rel count orders every warp's
            // reads before the last arrival, which refills the stage with step + STAGES (after
            // a proxy fence: the TMA writes through the async proxy).
            __syncwarp();
            if (lane == 0) {
                uint32_t prev;
                asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;\n"
                             : "=r"(prev)
                             : "r"(saddr(released + stage))
                             : "memory");
                if (prev % WARPS == WARPS - 1) {
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    issue(step + STAGES, stage);
                }
            }
            __syncwarp();
        }
    }
}
#endif

// 4 warps (#1133 r11): two 128 x 128 CTAs per SM instead of one 256 x 128 one, so one CTA's
// epilogue (the fused residual's loads and the stores) runs while the other's MMAs do. Started
// together, the two drift into step and finish their tiles at the same time, and the residual
// loads of a whole wave then hit memory at once (on the GB10 that ran 0.94x the wide tile): so
// the first wave starts staggered (a8t::Stagger) over about one tile's time, and each tile's
// residual prefetch is spread over its last steps (gemm8_res_prefetch).
// JG (#1133 r25): 0 gemm8_epilogue, -1 gemm8_epilogue_staged (no residual), -2 (#1183, the
// wide tile without a residual) gemm8_store_tma into maps.y, else gemm8_epilogue_stream with JG
// token tiles at a time.
template <bool RES, class C, int JG = 0>
__global__ void __launch_bounds__(32 * C::WARPS, 8 / C::WARPS)
    fp8_gemm8_tma(const legion_fp8_mm8 p, const __grid_constant__ a8t::Maps maps, const a8t::Stagger stagger) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using a8::BN;
    constexpr int WN = C::WN, NJ = BN / 8 / WN, TM = C::TM;
    unsigned char * smem = gemm8_tma_smem_base<C>();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / WN, wn = warp % WN;
    int64_t tm, tn;
    grouped_tile(p.n * p.k, tm, tn);  // sm80/fp8_mm.cuh
    const int64_t m0 = tm * TM, n0 = tn * BN;
    float acc[4][NJ][4];
    gemm8_tma_sums<RES, C>(p, maps, stagger, m0, n0, smem, wm, wn, lane, acc);
    if constexpr (JG == -2) {
        static_assert(!RES && C::WM == 4 && C::WARPS == 8 && NJ == 8, "TMA store: the wide tile, no residual");
        static_assert(C::WARPS * a8t::ST_WARP_FLOATS * sizeof(float) <= C::SMEM &&
                          a8t::ST_WARP_FLOATS * sizeof(float) % 1024 == 0,
                      "TMA store boxes in the stages, 1 KiB aligned");
        // As for the staged epilogue: the stages' memory is free once every warp is past the
        // K loop.
        __syncthreads();
        gemm8_scale_acc<C>(p, m0, n0, wm, wn, lane, acc);
        gemm8_store_tma(&maps.y, m0 + wm * 64, n0 + wn * NJ * 8, p.n, lane, acc,
                        (float *) smem + warp * a8t::ST_WARP_FLOATS, false);
    } else if constexpr (JG < 0) {
        static_assert(!RES && C::WM == 4, "staged: no residual, 64 rows a warp");
        // Every warp's reads of the stages are done (and every stage's copy landed: each was
        // waited for), so their memory takes the staged outputs.
        __syncthreads();
        gemm8_epilogue_staged<C>(p, m0, n0, wm, wn, lane, acc, (float *) smem + warp * 8 * (64 + 4));
    } else if constexpr (JG == 0) {
        gemm8_epilogue<RES, C>(p, m0, n0, wm, wn, lane, acc);
    } else {
        gemm8_epilogue_stream<RES, C, JG>(p, m0, n0, wm, wn, lane, acc);
    }
#else
    __trap();  // launch_gemm8_tma checks the image is sm_90+ first
#endif
}
