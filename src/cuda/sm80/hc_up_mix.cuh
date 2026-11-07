// hc_up_mix.cu, sm_80 and newer: the up matmul of qwen4exp's hyper-connection mixer on
// mma.sync m16n8k32 s8 x s8 -> s32, with the gated mix as its epilogue.
//
// Not a standalone header: hc_up_mix.cu includes it inside its anonymous namespace, after the
// argument struct and the constants (HC, ROWS, WARPS, NT, CHUNK, PAD). The kernel body is built
// only where __CUDA_ARCH__ >= 800 and traps elsewhere; mma_code() in hc_up_mix.cu checks that
// the binary has it for the device.
//
// A CTA owns ROWS mixed columns (i0 .. i0 + ROWS - 1) and CHUNK tokens. Its HC x ROWS up rows
// (row c * e + i for stream c) are staged in shared memory once: the int8 quants (rank bytes a
// row, PAD bytes apart so the fragment loads hit distinct banks) and the F16 scales as F32.
// Each warp takes tiles of NT x 8 tokens: per Q8_0 block kb (32 values of K) one m16n8k32 per
// stream and token tile gives the block's integer dot products, and
//
//   acc += C * dA * dB
//
// is ggml-cuda's MMQ sum for a Q8_0 weight (mmq-vec-dot.cuh, ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma:
// `sum += C.x[l]*dA*dB`, the blocks in K order), so acc is the gate MMQ writes, bit for bit.
// MMQ's K tiles are 256 values: for rank 320 its second tile also covers blocks 10 .. 15, whose
// activations are zero (scale 0, quants 0), so their terms are +-0 and leave the sums as they
// are; this kernel skips them.
//
// The epilogue is ggml's dsv4_hc_pre_f32<gated> for those columns and tokens, with the gate
// still in registers: sum += xn[c] * (1 / (1 + exp(-gate[c]))) over the streams in order, then
// scale * sum. RB: xn[c] is rebuilt as hc_post_norm computes it, __fmul_rn(__fmul_rn(res[c],
// sc[c]), g[c]) (the new streams, the stream's norm scale, gamma), not read.
//
// MMA fragments (PTX ISA, m16n8k32 .s8): with g = lane / 4 and q = lane % 4, A holds rows g and
// g + 8 at K bytes 4q .. 4q + 3 and 16 + 4q .. 16 + 4q + 3; B holds token g at the same K bytes;
// C holds (row g, tokens 2q and 2q + 1) and (row g + 8, the same tokens).

__device__ __forceinline__ void mma_s8(int (&c)[4], const int (&a)[4], const int (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// B16 (with RB, #826): the streams are BF16, widened to F32 as they are read.
template <bool RB, bool B16 = false>
__global__ void __launch_bounds__(WARPS * 32) up_mix_mma(const legion_hc_up_mix p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ int smem[];
    const int nkb    = p.rank / 32;
    const int stride = p.rank / 4 + PAD / 4;  // ints per staged row
    int *   sq       = smem;                                 // [HC * ROWS][stride]
    float * sd       = (float *) (smem + HC * ROWS * stride);  // [HC * ROWS][nkb]
    const int i0     = blockIdx.x * ROWS;

    // Stage the up rows: staged row r = c * ROWS + i is weight row c * e + i0 + i. A Q8_0 block
    // is 34 bytes (F16 scale, 32 quants), so only 2-byte aligned: read it in halves.
    for (int u = threadIdx.x; u < HC * ROWS * nkb; u += WARPS * 32) {
        const int r = u / nkb, kb = u % nkb;
        const int c = r / ROWS, i = r % ROWS;
        const uint16_t * h =
            (const uint16_t *) (p.up + (int64_t) (c * p.e + i0 + i) * p.up_nb1 + (int64_t) kb * 34);
        sd[r * nkb + kb] = __half2float(__ushort_as_half(h[0]));
#pragma unroll
        for (int j = 0; j < 8; j++) {
            sq[r * stride + kb * 8 + j] = (int) ((uint32_t) h[1 + 2 * j] | ((uint32_t) h[2 + 2 * j] << 16));
        }
    }
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const int64_t K   = (int64_t) HC * p.e;
    const int     end = min(p.n, (int) (blockIdx.y + 1) * CHUNK);
    for (int t0 = blockIdx.y * CHUNK + warp * NT * 8; t0 < end; t0 += WARPS * NT * 8) {
        // The epilogue's xn values (ROWS floats, 64 bytes, per token and stream; RB: the
        // streams') into L2 while the matmul runs: a lane per token.
        if (t0 + lane < end && lane < NT * 8) {
            const int64_t at = (int64_t) (t0 + lane) * K + i0;
#pragma unroll
            for (int c = 0; c < HC; c++) {
                if constexpr (B16) {
                    const __nv_bfloat16 * x = (const __nv_bfloat16 *) p.res + at;
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(x + (int64_t) c * p.e));
                } else {
                    const float * x = (RB ? p.res : p.xn) + at;
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(x + (int64_t) c * p.e));
                }
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
        for (int kb = 0; kb < nkb; kb++) {
            // B: lo's quants of block kb, [kb][token][32] (tokens past n: zero).
            int   b[NT][2];
            float db[NT][2];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const int tok = t0 + t * 8 + g;
                if (tok < p.n) {
                    const int * src = (const int *) (p.lq + ((int64_t) kb * p.n + tok) * 32);
                    b[t][0]         = src[q];
                    b[t][1]         = src[4 + q];
                } else {
                    b[t][0] = 0;
                    b[t][1] = 0;
                }
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int tk = t0 + t * 8 + 2 * q + j;
                    db[t][j]     = tk < p.n ? p.ld[(int64_t) kb * p.n + tk] : 0.0f;
                }
            }
#pragma unroll
            for (int c = 0; c < HC; c++) {
                const int * ra = sq + (c * ROWS + g) * stride + kb * 8;
                const int * rb = ra + 8 * stride;
                const int   a[4] = { ra[q], rb[q], ra[4 + q], rb[4 + q] };
                const float da0  = sd[(c * ROWS + g) * nkb + kb];
                const float da1  = sd[(c * ROWS + g + 8) * nkb + kb];
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    int C[4] = { 0, 0, 0, 0 };
                    mma_s8(C, a, b[t]);
                    acc[c][t][0] += C[0] * da0 * db[t][0];
                    acc[c][t][1] += C[1] * da0 * db[t][1];
                    acc[c][t][2] += C[2] * da1 * db[t][0];
                    acc[c][t][3] += C[3] * da1 * db[t][1];
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
                    const float * x   = (RB ? p.res : p.xn) + (int64_t) tok * K + i;
                    const __nv_bfloat16 * xb = (const __nv_bfloat16 *) p.res + (int64_t) tok * K + i;
                    float         sum = 0.0f;
#pragma unroll
                    for (int c = 0; c < HC; c++) {
                        float xv;
                        if constexpr (B16) {
                            xv = __bfloat162float(xb[(int64_t) c * p.e]);
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
