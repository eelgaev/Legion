// qwen4_hc.cu, sm_80 and newer: the mixer on BF16 down and up matrices (Qwen3.8-Flash-Next-NVFP4
// ships them; opt-in `legion serve --hc-mix-bf16`, #748) as the GB10 runs it. `norm_g` (without
// the Q8_1 copy), then two kernels launched early (PDL, see sm80/qwen4_hc.cuh):
// - `down_bf16_g`: GB_DOWN_ROWS down rows per CTA. Before the dependency wait each thread loads
//   its share of the rows into registers, 16 bytes (8 weights) per load with ld.global.nc, so the
//   rows stream in while `norm_g` still runs; then the sums against the F32 xn, SiLU as `down`.
//   The first n_inj CTAs are the inject sums (F32), one per (row, token), in `down`'s order.
// - `up_mix_bf16_g`: G_COLS columns per CTA, their hc x G_COLS up rows loaded into registers the
//   same way (UP_LANES lanes per row, as `up_mix`), then the sums against the F32 lo and the mix.
// No Q8_1 of xn or lo: the weights meet F32 activations, as ggml's mmvf takes them. The sums run
// in another order than ggml's, so the output moves by rounding (the reason the path is opt-in).
//
// Not a standalone header: qwen4_hc.cu includes it inside its anonymous namespace after
// sm80/qwen4_hc.cuh. The kernel bodies are built only where __CUDA_ARCH__ >= 800 and are empty
// elsewhere; use_gb10_bf16() in qwen4_hc.cu decides per device at run time whether they run.

constexpr int GB_DOWN_THREADS = 256;  // = INJ_THREADS: the inject sums keep `down`'s order
constexpr int GB_DOWN_ROWS    = 2;    // down rows per CTA (the activations read once for both)
constexpr int GB_DOWN_U       = 5;    // 16-byte loads per thread and row: K <= 256 * 8 * 5 = 10240
constexpr int GB_UP_U         = 5;    // 16-byte loads per lane and up row: rank <= 8 * 8 * 5 = 320
constexpr int GB_UP_PASSES    = 2;    // up rows per lane group: hc * G_COLS <= 2 * 32

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// 16 bytes at `p` through the read-only path, not kept in L1 (each weight is read once).
__device__ __forceinline__ uint4 ldg_nc16(const void * p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// s + the dot product of 8 BF16 weights `w` (in memory order) with x[0..8] (16-byte aligned), in
// element order.
__device__ __forceinline__ float dot8_bf16(const uint4 w, const float * x, float s) {
    const float4 a = ((const float4 *) x)[0], b = ((const float4 *) x)[1];
    const uint32_t u[4] = { w.x, w.y, w.z, w.w };
    const float    v[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
    for (int i = 0; i < 4; i++) {
        s += __uint_as_float(u[i] << 16) * v[2 * i];
        s += __uint_as_float(u[i] & 0xffff0000u) * v[2 * i + 1];
    }
    return s;
}
#endif

// See the header. Grid: n_inj inject CTAs, then rank / GB_DOWN_ROWS row CTAs.
template <int NT>
__global__ void __launch_bounds__(GB_DOWN_THREADS) down_bf16_g(const legion_hc_mix p, const int n_inj) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(GB_DOWN_THREADS == INJ_THREADS, "the inject sums in `down`'s order");
    constexpr int NWARPS = GB_DOWN_THREADS / WARP_SIZE;
    __shared__ float red[GB_DOWN_ROWS * NT][NWARPS];
    ggml_cuda_pdl_lc();
    const int K    = p.hc * p.n_embd;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if ((int) blockIdx.x < n_inj) {
        const int h = blockIdx.x / p.n, t = blockIdx.x % p.n;
        const float * w  = (const float *) ((const char *) p.inject_w + h * p.inject_nb1);
        const float * xn = p.xn + (int64_t) t * K;
        for (int k = threadIdx.x * 32; k < K; k += GB_DOWN_THREADS * 32) {
            asm volatile("prefetch.global.L2 [%0];" : : "l"(w + k));
        }
        ggml_cuda_pdl_sync();
        float s = 0.0f;
#pragma unroll 8
        for (int k = threadIdx.x; k < K; k += GB_DOWN_THREADS) {
            s += w[k] * xn[k];
        }
        const float v = block_sum<GB_DOWN_THREADS>(s, &red[0][0]);
        if (threadIdx.x == 0) {
            p.inject[t * p.hc + h] = v;
        }
        return;
    }
    const int r0 = ((int) blockIdx.x - n_inj) * GB_DOWN_ROWS, units = K / 8;
    uint4 w[GB_DOWN_ROWS][GB_DOWN_U];
#pragma unroll
    for (int m = 0; m < GB_DOWN_U; m++) {
        const int u = threadIdx.x + m * GB_DOWN_THREADS;
#pragma unroll
        for (int r = 0; r < GB_DOWN_ROWS; r++) {
            w[r][m] = u < units ? ldg_nc16(p.down + (r0 + r) * p.down_nb1 + u * 16) : make_uint4(0, 0, 0, 0);
        }
    }
    ggml_cuda_pdl_sync();
    float s[GB_DOWN_ROWS][NT] = {};
#pragma unroll
    for (int m = 0; m < GB_DOWN_U; m++) {
        const int u = threadIdx.x + m * GB_DOWN_THREADS;
        if (u < units) {
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const float * x = p.xn + (int64_t) t * K + u * 8;
#pragma unroll
                for (int r = 0; r < GB_DOWN_ROWS; r++) {
                    s[r][t] = dot8_bf16(w[r][m], x, s[r][t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < GB_DOWN_ROWS; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = warp_reduce_sum(s[r][t]);
            if (lane == 0) {
                red[r * NT + t][warp] = v;
            }
        }
    }
    __syncthreads();
    for (int rt = warp; rt < GB_DOWN_ROWS * NT; rt += NWARPS) {
        float v = lane < NWARPS ? red[rt][lane] : 0.0f;
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

// See the header. Row j of the CTA (stream c = j / G_COLS, column i0 + j % G_COLS) is taken by
// the UP_LANES lanes threadIdx.x / UP_LANES = j % 32 of pass j / 32, as in `up_mix`.
template <int NT>
__global__ void __launch_bounds__(UP_THREADS) up_mix_bf16_g(const legion_hc_mix p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int PER_PASS = UP_THREADS / UP_LANES;
    __shared__ float gate[NT][G_COLS * MAX_HC];
    const int K = p.hc * p.n_embd, rows = p.hc * G_COLS, units = p.rank / 8;
    const int i0 = blockIdx.x * G_COLS, sub = threadIdx.x % UP_LANES;
    uint4 w[GB_UP_PASSES][GB_UP_U];
#pragma unroll
    for (int q = 0; q < GB_UP_PASSES; q++) {
        const int j = q * PER_PASS + threadIdx.x / UP_LANES;
        const char * row = p.up + (int64_t) ((j / G_COLS) * p.n_embd + i0 + j % G_COLS) * p.up_nb1;
#pragma unroll
        for (int m = 0; m < GB_UP_U; m++) {
            const int u = sub + m * UP_LANES;
            w[q][m] = j < rows && u < units ? ldg_nc16(row + u * 16) : make_uint4(0, 0, 0, 0);
        }
    }
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int q = 0; q < GB_UP_PASSES; q++) {
        const int j = q * PER_PASS + threadIdx.x / UP_LANES;
        float s[NT] = {};
#pragma unroll
        for (int m = 0; m < GB_UP_U; m++) {
            const int u = sub + m * UP_LANES;
            if (u < units) {
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    s[t] = dot8_bf16(w[q][m], p.lo + (int64_t) t * p.rank + u * 8, s[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            for (int off = UP_LANES / 2; off > 0; off >>= 1) {
                s[t] += __shfl_xor_sync(0xffffffff, s[t], off, WARP_SIZE);
            }
            if (sub == 0 && j < rows) {
                gate[t][j] = s[t];
            }
        }
    }
    __syncthreads();
    for (int jt = threadIdx.x; jt < G_COLS * NT; jt += UP_THREADS) {
        const int t = jt / G_COLS, j = jt % G_COLS, i = i0 + j;
        float sum = 0.0f;
        for (int c = 0; c < p.hc; c++) {
            const float g = gate[t][c * G_COLS + j];
            sum += p.xn[(int64_t) t * K + c * p.n_embd + i] * (1.0f / (1.0f + expf(-g)));
        }
        p.mixed[(int64_t) t * p.n_embd + i] = sum * (1.0f / p.hc);
    }
#else
    __trap();
#endif
}
