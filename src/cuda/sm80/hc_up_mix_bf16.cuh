// hc_up_mix.cu, sm_80 and newer: the up matmul of qwen4exp's hyper-connection mixer for BF16
// up weights (the NVFP4 checkpoints, #836) on mma.sync m16n8k16 bf16 x bf16 -> f32, with the
// gated mix as its epilogue.
//
// Not a standalone header: hc_up_mix.cu includes it inside its anonymous namespace, after the
// argument struct and the constants (HC, ROWS, WARPS, NT, CHUNK, PAD). The kernel bodies are
// built only where __CUDA_ARCH__ >= 800 and trap elsewhere; mma_code() in hc_up_mix.cu
// checks that the binary has sm_80+ code for the device.
//
// What ggml runs instead: `mul_mat(up, lo)` on cuBLAS (ggml_cuda_mul_mat_cublas: lo converted
// to BF16 with __float2bfloat16, cublasGemmEx BF16 x BF16 -> F32 with CUBLAS_COMPUTE_32F,
// alpha 1, beta 0), which writes the gate (hc * n_embd F32 values a token), then
// dsv4_hc_pre_f32<gated> reading it back. At qwen4exp's shape (rank 320, 10240 rows) cuBLAS
// takes a CUTLASS s16816 tensor-op GEMM, one CTA per output tile (no split-K): each output is
// the m16n8k16 products of K steps 0, 16, 32, ... added in K order into an F32 accumulator that
// starts at zero, and alpha 1 / beta 0 leave it as it is. This kernel adds the same
// instructions in the same order. A first form of the op (one node of another size) moved the
// C=4 det hash on a GB10 through ggml-cuda's layout-dependent MoE fusions, not these sums: with
// those fusions pinned (`--stable-fusions`) the hashes matched, and in ggml's node layout they
// match by default. On by default; `legion serve --no-hc-up-mix-bf16` keeps cuBLAS.
//
// lo_bf16: lo (F32 [n][rank]) rounded to BF16 as ggml's convert does, into the scratch lob.
//
// up_mix_bf16: a CTA owns ROWS mixed columns (i0 .. i0 + ROWS - 1) and CHUNK tokens. Its
// HC x ROWS up rows (row c * e + i for stream c) are staged in shared memory once (rank BF16
// values a row, PAD bytes apart so the fragment loads hit distinct banks). Each warp takes tiles
// of NT x 8 tokens: per K step of 16 one m16n8k16 per stream and token tile. The epilogue is
// up_mix_mma's (ggml's dsv4_hc_pre_f32<gated>): sum += xn[c] * (1 / (1 + exp(-gate[c]))) over
// the streams in order, then scale * sum. RB: xn[c] is rebuilt as hc_post_norm computes it,
// __fmul_rn(__fmul_rn(res[c], sc[c]), g[c]) (the new streams, the stream's norm scale, gamma),
// not read, as up_mix_mma<true> does: the norm then skips writing the F32 xn (40 KB a token at
// qwen4exp's shape), the same bits. With mixedb (#1183) the epilogue also writes mixed rounded
// to BF16 with __float2bfloat16, the rounding of ggml-cuda's F32 -> BF16 copy the BF16
// projections read, and with skip_f32 not mixed itself (the node fusion's mix cast).
//
// MMA fragments (PTX ISA, m16n8k16 .bf16): with g = lane / 4 and q = lane % 4, A holds rows g
// and g + 8 at K values 2q, 2q + 1 and 2q + 8, 2q + 9 (a register each, the lower K in the low
// half); B holds token g at the same K values; C holds (row g, tokens 2q and 2q + 1) and (row
// g + 8, the same tokens), as m16n8k32 .s8 does.

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// lob[i] = __float2bfloat16(lo[i]), two values a thread.
__global__ void lo_bf16(const legion_hc_up_mix p) {
    const int64_t i     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pairs = (int64_t) p.n * p.rank / 2;
    if (i >= pairs) {
        return;
    }
    const float2   v = ((const float2 *) p.lo)[i];
    __nv_bfloat162 r;
    r.x                                = __float2bfloat16(v.x);
    r.y                                = __float2bfloat16(v.y);
    ((__nv_bfloat162 *) p.lob)[i] = r;
}

// Lazy streams (#1183, p.pre_out): the weight stream s of token tok takes the attention block's
// output with, as fused.cu's hc_post_norm kernels compute their wpost (the same expressions).
__device__ __forceinline__ float pre_weight(const legion_hc_up_mix & p, int64_t tok, int s) {
    // ggml: scale(inject, 1 / hc), sigmoid, scale(2)
    const float v = p.pre_inj[tok * p.pre_inj_ld + s] * (1.0f / (float) p.hc);
    const float e = 1.0f / (1.0f + expf(-v));
    return e * 2.0f;
}

// S16 (with RB; #1183, `legion serve --hc-bf16`): the streams are BF16 (hc_post_norm_b16_kernel's),
// each widened to F32 before the rebuild.
// PRE (with RB, F32 streams; #1183, lazy streams): res holds the streams before the attention
// block's combine; each value is combined first, __fmaf_rn(pre_out, pre_weight, res), the
// norm's FMA, so the new streams' bits.
template <bool RB, bool S16 = false, bool PRE = false>
__global__ void __launch_bounds__(WARPS * 32) up_mix_bf16(const legion_hc_up_mix p) {
    static_assert(RB || !S16, "up_mix_bf16: BF16 streams in the rebuilt form only");
    static_assert(!PRE || (RB && !S16), "up_mix_bf16: lazy streams of F32 streams in the rebuilt form only");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ int smem[];
    const int  stride = p.rank / 2 + PAD / 4;  // 32-bit words per staged row
    uint32_t * sa     = (uint32_t *) smem;     // [HC * ROWS][stride]
    const int  i0     = blockIdx.x * ROWS;

    // Stage the up rows (16 bytes at a time): staged row r = c * ROWS + i is weight row
    // c * e + i0 + i.
    const int vecs = p.rank / 8;
    for (int u = threadIdx.x; u < HC * ROWS * vecs; u += WARPS * 32) {
        const int r = u / vecs, v = u % vecs;
        const int c = r / ROWS, i = r % ROWS;
        const int4 * src = (const int4 *) (p.up + (int64_t) (c * p.e + i0 + i) * p.up_nb1) + v;
        *(int4 *) (sa + r * stride + v * 4) = *src;
    }
    __syncthreads();

    const int       warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const int64_t   K    = (int64_t) HC * p.e;
    const int       end  = min(p.n, (int) (blockIdx.y + 1) * CHUNK);
    const uint32_t * lob = (const uint32_t *) p.lob;  // BF16 pairs, [n][rank / 2]
    for (int t0 = blockIdx.y * CHUNK + warp * NT * 8; t0 < end; t0 += WARPS * NT * 8) {
        // The epilogue's xn values (ROWS floats, 64 bytes, per token and stream; RB: the
        // streams') into L2 while the matmul runs: a lane per token.
        if (t0 + lane < end && lane < NT * 8) {
            constexpr int es = S16 ? 2 : 4;
            const char * x = (const char *) (RB ? p.res : p.xn) + ((int64_t) (t0 + lane) * K + i0) * es;
#pragma unroll
            for (int c = 0; c < HC; c++) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(x + (int64_t) c * p.e * es));
            }
        }
        float acc[HC][NT][4];
#pragma unroll
        for (int c = 0; c < HC; c++) {
#pragma unroll
            for (int t = 0; t < NT; t++) {
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    acc[c][t][l] = 0.0f;
                }
            }
        }
        for (int k0 = 0; k0 < p.rank; k0 += 16) {
            // B: lo's BF16 values k0 .. k0 + 15 of token g of each tile (tokens past n: zero).
            uint32_t b[NT][2];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const int tok = t0 + t * 8 + g;
                if (tok < p.n) {
                    const uint32_t * src = lob + (int64_t) tok * (p.rank / 2) + k0 / 2;
                    b[t][0]              = src[q];
                    b[t][1]              = src[4 + q];
                } else {
                    b[t][0] = 0;
                    b[t][1] = 0;
                }
            }
#pragma unroll
            for (int c = 0; c < HC; c++) {
                const uint32_t * ra   = sa + (c * ROWS + g) * stride + k0 / 2;
                const uint32_t * rb   = ra + 8 * stride;
                const uint32_t   a[4] = { ra[q], rb[q], ra[4 + q], rb[4 + q] };
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    mma_bf16(acc[c][t], a, b[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int tok = t0 + t * 8 + 2 * q + (l & 1);
                if (tok < p.n) {
                    const int     i   = i0 + g + (l >> 1) * 8;
                    const int64_t at  = (int64_t) tok * K + i;
                    const float * x   = (RB ? p.res : p.xn) + at;
                    float         sum = 0.0f;
                    [[maybe_unused]] float po = 0.0f;
                    if constexpr (PRE) {
                        po = p.pre_out[(int64_t) tok * p.pre_out_ld + i];
                    }
#pragma unroll
                    for (int c = 0; c < HC; c++) {
                        float xv;
                        if constexpr (S16) {
                            xv = __bfloat162float(((const __nv_bfloat16 *) p.res)[at + (int64_t) c * p.e]);
                        } else {
                            xv = x[(int64_t) c * p.e];
                        }
                        if constexpr (PRE) {
                            xv = __fmaf_rn(po, pre_weight(p, tok, c), xv);
                        }
                        if constexpr (RB) {
                            xv = __fmul_rn(__fmul_rn(xv, p.sc[(int64_t) tok * HC + c]), p.g[c * p.e + i]);
                        }
                        const float wv = 1.0f / (1.0f + expf(-acc[c][t][l]));
                        sum += xv * wv;
                    }
                    const float   v  = p.scale * sum;
                    const int64_t mo = (int64_t) tok * p.e + i;
                    if (p.skip_f32 == 0) {
                        p.mixed[mo] = v;
                    }
                    // The BF16 copy (#1183): ggml-cuda's F32 -> BF16 copy of mixed, made here.
                    if (p.mixedb != nullptr) {
                        ((__nv_bfloat16 *) p.mixedb)[mo] = __float2bfloat16(v);
                    }
                }
            }
        }
    }
#else
    __trap();
#endif
}

// up_mix_bf16_reg (#1183): up_mix_bf16's sums and mix, laid out for the memory floor of a
// prompt step. The mix streams the 4 streams of every token once (F32: 40 KB a token at
// qwen4exp's shape) and its matmul is small (rank 320), so what costs is everything around the
// bytes: up_mix_bf16 stages 40 KB of up rows per 128 tokens with one dependent load per 16
// bytes, has 4 warps on a CTA and 2 CTAs on an SM to cover it, and starts each epilogue on an
// L2 prefetch issued just before its matmul.
//
// Here a warp holds its 16 up rows of one stream in registers (KS = rank / 16 K steps of the
// m16n8k16 A fragment), loaded once; a CTA owns CG groups of 16 columns (CG x HC warps) and a
// run of tokens (the launch's chunk), TT at a time. Per tile: lo's TT rows go to shared memory
// by cp.async one tile ahead (shared by all the CTA's warps), each warp adds its stream's
// products for TT tokens in K order from zero (up_mix_bf16's instructions: the same A, B and C
// fragments, so the same sums), the gates go to shared memory, and then every thread mixes
// TT / 8 outputs of one column with up_mix_bf16's epilogue expressions, the streams in order.
// The streams are prefetched into L2 PD tiles ahead (one bulk prefetch per token and stream on
// sm_90+). For a handful of tokens the up rows are most of the bytes and the register loads of
// A cost more than up_mix_bf16's staging; the launch keeps up_mix_bf16 there.
// PRE: as up_mix_bf16's (lazy streams, #1183); the attention block's output rows are
// prefetched with the streams.
template <int KS, int CG, int TT, int MINB, int PD, bool RB, bool S16, bool PRE = false>
__global__ void __launch_bounds__(CG * HC * 32, MINB) up_mix_bf16_reg(const legion_hc_up_mix p, const int chunk) {
    static_assert(RB || !S16, "up_mix_bf16_reg: BF16 streams in the rebuilt form only");
    static_assert(!PRE || (RB && !S16), "up_mix_bf16_reg: lazy streams of F32 streams in the rebuilt form only");
    static_assert(TT % 8 == 0 && TT >= 8, "up_mix_bf16_reg: whole n8 tiles");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int THREADS = CG * HC * 32;
    constexpr int COLS    = CG * ROWS;      // mixed columns of the CTA
    constexpr int NT      = TT / 8;         // n8 token tiles a tile
    constexpr int LW      = KS * 8 + PAD / 4;  // 32-bit words per staged lo row
    constexpr int GS      = COLS + 4;       // floats per staged gate row (2 GS = 8 mod 32 words)
    constexpr int J       = TT / 8;         // outputs a thread mixes per tile (TT x COLS / THREADS)
    constexpr int ES      = S16 ? 2 : 4;    // bytes per stream value
    extern __shared__ int smem[];
    uint32_t * lb = (uint32_t *) smem;            // [2][TT][LW]: lo's BF16 pairs
    float *    gs = (float *) (lb + 2 * TT * LW);  // [HC][TT][GS]: the gates

    const int     tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int     c = warp % HC, gi = warp / HC;  // the warp's stream and column group
    const int     i0 = blockIdx.x * COLS;
    const int64_t K  = (int64_t) HC * p.e;
    const int     t_begin = blockIdx.y * chunk;
    const int     t_end   = min(p.n, t_begin + chunk);
    if (t_begin >= t_end) {
        return;
    }
    const int tiles = (t_end - t_begin + TT - 1) / TT;
    const void * xs = RB ? (const void *) p.res : (const void *) p.xn;

    // lo's rows of tile `tile` into buffer `buf` (tokens past the run: zeros), one group.
    auto stage = [&](int tile, int buf) {
        constexpr int V    = KS * 2;  // 16-byte pieces a row
        const int     base = t_begin + tile * TT;
        for (int u = tid; u < TT * V; u += THREADS) {
            const int      r = u / V, v = u % V;
            const int      tok = base + r;
            const bool     ok  = tok < t_end;
            const char *   src = (const char *) p.lob + (int64_t) (ok ? tok : t_begin) * (KS * 32) + v * 16;
            const uint32_t dst = (uint32_t) __cvta_generic_to_shared(lb + (buf * TT + r) * LW + v * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // The streams of tile `tile` into L2: COLS values per token and stream.
    auto prefetch = [&](int tile) {
        if (tile >= tiles) {
            return;
        }
        const int base = t_begin + tile * TT;
        for (int u = tid; u < TT * HC; u += THREADS) {
            const int tok = base + u / HC;
            if (tok < t_end) {
                const char * x = (const char *) xs + ((int64_t) tok * K + (int64_t) (u % HC) * p.e + i0) * ES;
#if __CUDA_ARCH__ >= 900
                if (((uintptr_t) x & 15) == 0) {
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(x), "r"((uint32_t) (COLS * ES)));
                    continue;
                }
#endif
#pragma unroll
                for (int b = 0; b < COLS * ES; b += 32) {
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(x + b));
                }
            }
        }
        if constexpr (PRE) {
            // The attention block's output: COLS F32 values per token.
            for (int u = tid; u < TT; u += THREADS) {
                const int tok = base + u;
                if (tok < t_end) {
                    const char * x = (const char *) (p.pre_out + (int64_t) tok * p.pre_out_ld + i0);
#if __CUDA_ARCH__ >= 900
                    if (((uintptr_t) x & 15) == 0) {
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(x), "r"((uint32_t) (COLS * 4)));
                        continue;
                    }
#endif
#pragma unroll
                    for (int b = 0; b < COLS * 4; b += 32) {
                        asm volatile("prefetch.global.L2 [%0];" ::"l"(x + b));
                    }
                }
            }
        }
    };

    stage(0, 0);
#pragma unroll
    for (int d = 0; d < PD; d++) {
        prefetch(d);
    }

    // A: the warp's up rows c * e + i0 + 16 gi + g and + 8, every K step (up_mix_bf16's
    // fragment: words k0 / 2 + q and k0 / 2 + 4 + q of each row).
    uint32_t a[KS][4];
    {
        const uint32_t * r0 = (const uint32_t *) (p.up + (int64_t) (c * p.e + i0 + gi * ROWS + g) * p.up_nb1);
        const uint32_t * r1 = (const uint32_t *) ((const char *) r0 + 8 * p.up_nb1);
#pragma unroll
        for (int k = 0; k < KS; k++) {
            a[k][0] = r0[k * 8 + q];
            a[k][1] = r1[k * 8 + q];
            a[k][2] = r0[k * 8 + 4 + q];
            a[k][3] = r1[k * 8 + 4 + q];
        }
    }
    // The thread's epilogue column and gammas.
    const int ec = tid % COLS, er = tid / COLS;  // er: 0 .. THREADS / COLS - 1 = 7
    const int ei = i0 + ec;
    float     gv[HC];
#pragma unroll
    for (int s = 0; s < HC; s++) {
        gv[s] = RB ? p.g[s * p.e + ei] : 0.0f;
    }

    for (int it = 0; it < tiles; it++) {
        const int buf  = it & 1;
        const int base = t_begin + it * TT;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        // Tile it's lo is in; every thread is past the last tile's matmul (buffer buf ^ 1 free)
        // and its mix (the gates free).
        __syncthreads();
        if (it + 1 < tiles) {
            stage(it + 1, buf ^ 1);
        }
        if (PD > 0) {
            prefetch(it + PD);
        }

        float acc[NT][4];
#pragma unroll
        for (int t = 0; t < NT; t++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[t][l] = 0.0f;
            }
        }
        const uint32_t * lt = lb + buf * TT * LW;
#pragma unroll
        for (int k = 0; k < KS; k++) {
            uint32_t b[NT][2];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const uint32_t * src = lt + (t * 8 + g) * LW + k * 8;
                b[t][0]              = src[q];
                b[t][1]              = src[4 + q];
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                mma_bf16(acc[t], a[k], b[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                gs[(c * TT + t * 8 + 2 * q + (l & 1)) * GS + gi * ROWS + g + (l >> 1) * 8] = acc[t][l];
            }
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < J; j++) {
            const int tt  = er + 8 * j;
            const int tok = base + tt;
            if (tok < t_end) {
                const int64_t at  = (int64_t) tok * K + ei;
                float         sum = 0.0f;
                [[maybe_unused]] float po = 0.0f;
                if constexpr (PRE) {
                    po = p.pre_out[(int64_t) tok * p.pre_out_ld + ei];
                }
#pragma unroll
                for (int s = 0; s < HC; s++) {
                    float xv;
                    if constexpr (S16) {
                        xv = __bfloat162float(((const __nv_bfloat16 *) p.res)[at + (int64_t) s * p.e]);
                    } else {
                        xv = ((const float *) xs)[at + (int64_t) s * p.e];
                    }
                    if constexpr (PRE) {
                        xv = __fmaf_rn(po, pre_weight(p, tok, s), xv);
                    }
                    if constexpr (RB) {
                        xv = __fmul_rn(__fmul_rn(xv, p.sc[(int64_t) tok * HC + s]), gv[s]);
                    }
                    const float wv = 1.0f / (1.0f + expf(-gs[(s * TT + tt) * GS + ec]));
                    sum += xv * wv;
                }
                const float   v  = p.scale * sum;
                const int64_t mo = (int64_t) tok * p.e + ei;
                if (p.skip_f32 == 0) {
                    p.mixed[mo] = v;
                }
                if (p.mixedb != nullptr) {
                    ((__nv_bfloat16 *) p.mixedb)[mo] = __float2bfloat16(v);
                }
            }
        }
    }
#else
    __trap();
#endif
}
