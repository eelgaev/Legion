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
// qwen4exp's shape), the same bits.
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

// S16 (with RB; #1183, `legion serve --hc-bf16`): the streams are BF16 (hc_post_norm_b16_kernel's),
// each widened to F32 before the rebuild.
template <bool RB, bool S16 = false>
__global__ void __launch_bounds__(WARPS * 32) up_mix_bf16(const legion_hc_up_mix p) {
    static_assert(RB || !S16, "up_mix_bf16: BF16 streams in the rebuilt form only");
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
#pragma unroll
                    for (int c = 0; c < HC; c++) {
                        float xv;
                        if constexpr (S16) {
                            xv = __bfloat162float(((const __nv_bfloat16 *) p.res)[at + (int64_t) c * p.e]);
                        } else {
                            xv = x[(int64_t) c * p.e];
                        }
                        if constexpr (RB) {
                            xv = __fmul_rn(__fmul_rn(xv, p.sc[(int64_t) tok * HC + c]), p.g[c * p.e + i]);
                        }
                        const float wv = 1.0f / (1.0f + expf(-acc[c][t][l]));
                        sum += xv * wv;
                    }
                    p.mixed[(int64_t) tok * p.e + i] = p.scale * sum;
                }
            }
        }
    }
#else
    __trap();
#endif
}
