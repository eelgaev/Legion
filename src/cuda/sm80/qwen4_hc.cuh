// qwen4_hc.cu, sm_80 and newer: the hyper-connection mixer's kernels as the GB10 (sm_121) runs
// them. The same arithmetic in the same order as `norm`, `down`, `quantize_lo` and `up_mix`, so
// the same bits; what changes is how the memory is read and when the kernels start:
// - The down and up weights are staged into shared memory with 16-byte cp.async copies, a CTA's
//   rows as one contiguous run (GB10's LPDDR5X streams well only in long runs; the 2-byte loads
//   of a few Q8_0 blocks per row that `down` and `up_mix` issue read it at ~145 GB/s).
// - Each kernel lets the next one launch at once (programmatic dependent launch, sm_90+): the
//   next stages its weights while the previous runs and waits (griddepcontrol.wait) only before
//   it reads the previous kernel's results, so the up weights stream in as `down`'s CTAs retire.
// - `norm_g` loads gamma before anything else and keeps each element of the stream in a
//   register between the RMS pass and the quantize pass (the two passes take the same elements
//   per thread), instead of reading it twice.
// - The inject rows (F32) come first in `down_g`'s grid and are prefetched into L2 before the
//   wait, so their long per-thread sums do not trail the down rows.
//
// Not a standalone header: qwen4_hc.cu includes it inside its anonymous namespace, after the
// legion_hc_mix declaration, the constants and block_sum / quantize_lane. The kernel bodies are
// built only where __CUDA_ARCH__ >= 800 and trap elsewhere; use_gb10() in qwen4_hc.cu
// decides per device at run time whether they run.

constexpr int G_NORM_M      = 4;   // norm_g: elements per thread, n_embd <= NORM_THREADS * G_NORM_M
constexpr int G_ROWS        = 4;   // down_g: down rows per CTA (4 x 10880 B of shared memory)
constexpr int G_COLS        = 16;  // up_mix_g: columns per CTA
constexpr int G_UNROLL_DOWN = 5;
constexpr int G_UNROLL_UP   = 5;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// `bytes` (a multiple of 16) from `src` to `dst` in shared memory, 16 bytes per copy, the CTA's
// threads on consecutive chunks.
__device__ __forceinline__ void stage_g(char * dst, const char * src, int bytes) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    for (int o = threadIdx.x * 16; o < bytes; o += blockDim.x * 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(d + o), "l"(src + o));
    }
}

__device__ __forceinline__ void stage_wait_g() {
    asm volatile("cp.async.wait_all;" : : : "memory");
    __syncthreads();
}
#endif

// `norm` with gamma loaded first and the stream kept in registers (see the header). Q8: also
// xn as Q8_1 (the Q8_0 kernels' input; the BF16 ones take the F32 xn only).
template <bool Q8>
__global__ void __launch_bounds__(NORM_THREADS) norm_g(const legion_hc_mix p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    ggml_cuda_pdl_lc();
    __shared__ float red[NORM_THREADS / WARP_SIZE];
    const int c = blockIdx.x, t = blockIdx.y, K = p.hc * p.n_embd;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    // Element m of this thread is i = threadIdx.x + m * NORM_THREADS in both passes: in the
    // second, block b = warp + m * (NORM_THREADS / WARP_SIZE), i = b * QK8_1 + lane.
    float g[G_NORM_M];
#pragma unroll
    for (int m = 0; m < G_NORM_M; m++) {
        const int i = threadIdx.x + m * NORM_THREADS;
        g[m] = i < p.n_embd ? p.norm[c * p.n_embd + i] : 0.0f;
    }
    ggml_cuda_pdl_sync();
    const float * xc = (const float *) ((const char *) p.x + t * p.x_nb2 + c * p.x_nb1);
    const float * oc = p.out ? (const float *) ((const char *) p.out + t * p.out_nb1) : nullptr;
    float post = 0.0f;
    if (oc) {
        const float v = (1.0f / p.hc) * ((const float *) ((const char *) p.inject_prev + t * p.inject_prev_nb1))[c] + 0.0f;
        post          = 2.0f * (1.0f / (1.0f + expf(-v))) + 0.0f;
    }
    float * rc = oc ? p.res + ((int64_t) t * p.hc + c) * p.n_embd : nullptr;
    auto stream = [&](int i) {
        if (!oc) {
            return xc[i];
        }
        float sum = oc[i] * post;
        sum += xc[i];
        return sum;
    };
    float xv[G_NORM_M];
    float ss = 0.0f;
#pragma unroll
    for (int m = 0; m < G_NORM_M; m++) {
        const int i = threadIdx.x + m * NORM_THREADS;
        xv[m]       = 0.0f;
        if (i < p.n_embd) {
            const float xi = stream(i);
            xv[m]          = xi;
            if (rc) {
                rc[i] = xi;
            }
            ss += xi * xi;
        }
    }
    const float scale = rsqrtf(block_sum<NORM_THREADS>(ss, red) / p.n_embd + p.eps);
#pragma unroll
    for (int m = 0; m < G_NORM_M; m++) {
        const int b = warp + m * (NORM_THREADS / WARP_SIZE);
        if (b < p.n_embd / QK8_1) {
            const int i = b * QK8_1 + lane, k = c * p.n_embd + i;
            const float v = scale * xv[m] * g[m];
            p.xn[(int64_t) t * K + k] = v;
            if constexpr (Q8) {
                quantize_lane(v, &p.xq[(int64_t) t * (K / QK8_1) + k / QK8_1], lane);
            }
        }
    }
#else
    __trap();
#endif
}

// `down`: the first n_inj CTAs are the inject sums, one CTA per (inject row h, token t); the
// others take R down rows each, staged in shared memory. DOWN_THREADS threads each: a down row's
// sums as `down` splits them (thread i: units i, i + DOWN_THREADS, ...), each unit's activations
// loaded once for the R rows; an inject sum as `down`'s INJ_THREADS threads split it (thread i:
// k = i, i + INJ_THREADS, ...), each thread here running two of those threads' sums.
template <int NT, int R>
__global__ void __launch_bounds__(DOWN_THREADS) down_g(const legion_hc_mix p, const int n_inj) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(INJ_THREADS == 2 * DOWN_THREADS, "two inject sums per thread");
    extern __shared__ __align__(16) char rows[];
    __shared__ float red[R * NT][DOWN_THREADS / WARP_SIZE];
    ggml_cuda_pdl_lc();
    const int K    = p.hc * p.n_embd;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if ((int) blockIdx.x < n_inj) {
        const int h = blockIdx.x / p.n, t = blockIdx.x % p.n;
        const float * w  = (const float *) ((const char *) p.inject_w + h * p.inject_nb1);
        const float * xn = p.xn + (int64_t) t * K;
        for (int k = threadIdx.x * 32; k < K; k += DOWN_THREADS * 32) {
            asm volatile("prefetch.global.L2 [%0];" : : "l"(w + k));
        }
        ggml_cuda_pdl_sync();
        // `down`'s threads threadIdx.x (a) and threadIdx.x + DOWN_THREADS (b).
        float sa = 0.0f, sb = 0.0f;
#pragma unroll 8
        for (int k = threadIdx.x; k < K; k += INJ_THREADS) {
            sa += w[k] * xn[k];
            if (k + DOWN_THREADS < K) {
                sb += w[k + DOWN_THREADS] * xn[k + DOWN_THREADS];
            }
        }
        // block_sum<INJ_THREADS>: `down`'s warp w is warp w here (a) or w - 4 (b).
        sa = warp_reduce_sum(sa);
        sb = warp_reduce_sum(sb);
        static_assert(R * NT * (DOWN_THREADS / WARP_SIZE) >= INJ_THREADS / WARP_SIZE, "red holds the partials");
        float * ired = &red[0][0];
        if (lane == 0) {
            ired[warp]                            = sa;
            ired[warp + DOWN_THREADS / WARP_SIZE] = sb;
        }
        __syncthreads();
        if (warp == 0) {
            float v = lane < INJ_THREADS / WARP_SIZE ? ired[lane] : 0.0f;
            v       = warp_reduce_sum(v);
            if (lane == 0) {
                p.inject[t * p.hc + h] = v;
            }
        }
        return;
    }
    const int r0 = ((int) blockIdx.x - n_inj) * R;
    stage_g(rows, p.down + r0 * p.down_nb1, R * (int) p.down_nb1);
    ggml_cuda_pdl_sync();
    stage_wait_g();
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, lanes_per_block = QI8_0 / VDR;
    float s[R][NT] = {};
#pragma unroll G_UNROLL_DOWN
    for (int c = threadIdx.x; c < K / QK8_0 * lanes_per_block; c += DOWN_THREADS) {
        // vec_dot_q8_0_q8_1(row, &xq[t][kbx], kbx, iqs) for each row and token.
        const int kbx = c / lanes_per_block, iqs = VDR * (c % lanes_per_block);
        int u[NT][VDR];
        half d1[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const block_q8_1 * bq8_1 = &p.xq[(int64_t) t * (K / QK8_1) + kbx];
#pragma unroll
            for (int i = 0; i < VDR; i++) {
                u[t][i] = get_int_b4(bq8_1->qs, iqs + i);
            }
            d1[t] = __low2half(bq8_1->ds);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const block_q8_0 * bq8_0 = (const block_q8_0 *) (rows + r * p.down_nb1) + kbx;
            int v[VDR];
#pragma unroll
            for (int i = 0; i < VDR; i++) {
                v[i] = get_int_b2(bq8_0->qs, iqs + i);
            }
#pragma unroll
            for (int t = 0; t < NT; t++) {
                s[r][t] += vec_dot_q8_0_q8_1_impl<float, VDR>(v, u[t], bq8_0->d, d1[t]);
            }
        }
    }
    // block_sum<DOWN_THREADS> of each (row, token), then * 1/hc and SiLU, as `down`.
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = warp_reduce_sum(s[r][t]);
            if (lane == 0) {
                red[r * NT + t][warp] = v;
            }
        }
    }
    __syncthreads();
    for (int rt = warp; rt < R * NT; rt += DOWN_THREADS / WARP_SIZE) {
        float v = lane < DOWN_THREADS / WARP_SIZE ? red[rt][lane] : 0.0f;
        v       = warp_reduce_sum(v) * (1.0f / p.hc);
        if (lane == 0) {
            const int r = rt / NT, t = rt % NT;
            p.lo[(int64_t) t * p.rank + r0 + r] = v / (1.0f + expf(-v));
        }
    }
#else
    __trap();
#endif
}

// `quantize_lo` after the dependency wait.
__global__ void quantize_lo_g(const legion_hc_mix p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    const int t = blockIdx.y, b = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (b < p.rank / QK8_1) {
        quantize_lane(p.lo[(int64_t) t * p.rank + b * QK8_1 + threadIdx.x % WARP_SIZE],
                      &p.lq[(int64_t) t * (p.rank / QK8_1) + b], threadIdx.x % WARP_SIZE);
    }
#else
    __trap();
#endif
}

// `up_mix` over COLS columns per CTA, their up rows staged in shared memory: row j of the block
// (stream c = j / COLS, column i0 + j % COLS) at j * up_nb1. n_embd is a multiple of COLS. Each
// row's sum and each column's mix as `up_mix` computes them.
//
// QLO (#827): no `quantize_lo_g` ahead of it; each CTA makes lo's Q8_1 itself, in shared memory
// after the rows (hc * COLS * up_nb1 bytes in), right after the wait for `down_g`, with
// quantize_lane on the same values: the same blocks, so the same bits, one launch and one grid
// drain fewer per mixer (as `up_mix_ahead` does on the V100 path, #814).
template <int NT, int COLS, bool QLO>
__global__ void __launch_bounds__(UP_THREADS) up_mix_g(const legion_hc_mix p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) char rows[];
    __shared__ float gate[NT][COLS * MAX_HC];
    constexpr int VDR = VDR_Q8_0_Q8_1_MMVQ, lanes_per_block = QI8_0 / VDR;
    const int K = p.hc * p.n_embd;
    const int lane = threadIdx.x % WARP_SIZE;
    const int i0 = blockIdx.x * COLS, ncols = COLS;
    const int units = p.rank / QK8_0 * lanes_per_block, sub = threadIdx.x % UP_LANES;
    const int chunk = COLS * (int) p.up_nb1;
    for (int c = 0; c < p.hc; c++) {
        stage_g(rows + c * chunk, p.up + (int64_t) (c * p.n_embd + i0) * p.up_nb1, chunk);
    }
    ggml_cuda_pdl_sync();
    const block_q8_1 * lq = p.lq;
    if constexpr (QLO) {
        block_q8_1 * lq_s = (block_q8_1 *) (rows + p.hc * chunk);
        const int nb = p.rank / QK8_1;
        for (int tb = threadIdx.x / WARP_SIZE; tb < NT * nb; tb += UP_THREADS / WARP_SIZE) {
            const int t = tb / nb, b = tb % nb;
            quantize_lane(p.lo[(int64_t) t * p.rank + b * QK8_1 + lane], &lq_s[tb], lane);
        }
        lq = lq_s;
    }
    stage_wait_g();

    for (int j0 = threadIdx.x / WARP_SIZE * (WARP_SIZE / UP_LANES); j0 < ncols * p.hc;
         j0 += UP_THREADS / UP_LANES) {
        const int j = j0 + lane / UP_LANES;
        float s[NT] = {};
        if (j < ncols * p.hc) {
            const char * row = rows + j * p.up_nb1;
#pragma unroll G_UNROLL_UP
            for (int u = sub; u < units; u += UP_LANES) {
                const int kbx = u / lanes_per_block;
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    s[t] += vec_dot_q8_0_q8_1(row, &lq[(int64_t) t * (p.rank / QK8_1) + kbx], kbx,
                                              VDR * (u % lanes_per_block));
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            for (int off = UP_LANES / 2; off > 0; off >>= 1) {
                s[t] += __shfl_xor_sync(0xffffffff, s[t], off, WARP_SIZE);
            }
            if (sub == 0 && j < ncols * p.hc) {
                gate[t][j] = s[t];
            }
        }
    }
    __syncthreads();
    for (int jt = threadIdx.x; jt < ncols * NT; jt += UP_THREADS) {
        const int t = jt / ncols, j = jt % ncols, i = i0 + j;
        float sum = 0.0f;
        for (int c = 0; c < p.hc; c++) {
            const float g = gate[t][c * ncols + j];
            sum += p.xn[(int64_t) t * K + c * p.n_embd + i] * (1.0f / (1.0f + expf(-g)));
        }
        p.mixed[(int64_t) t * p.n_embd + i] = sum * (1.0f / p.hc);
    }
#else
    __trap();
#endif
}
