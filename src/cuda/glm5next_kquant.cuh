// GLM-5.3-Flash's dense K-quant weights (#559): Project Maya's GGUFs keep attention, the shared
// experts, the dense FFNs, the embeddings and the output head in Q6_K (Maya-S24: Q4_K). Included
// by glm5next.cu inside its anonymous namespace, after `Lin`, `lin_scale` and `warp_sum` (`wval` and `wpiece` reach kq_val
// and kq_piece through declarations above them).
//
// DeepSeek-V4.1-Flash's GGUFs (#584) keep the same kind of dense weights in Q5_K.
//
// Values are ggml's (ggml-quants.c dequantize_row_q6_K / _q4_K / _q5_K, MIT): Q6_K value
// d * scale * (q - 32), Q4_K and Q5_K value d * scale * q - dmin * min, in F32.
//
// - kq_val / kq_piece: one value, or 16 values of a 32-value block, for the generic paths
//   (embed_kernel, hc_mix_kernel, linear_mv, linear_mm).
// - linear_mv_q6k / linear_mv_q4k: the GEMVs of decode steps (up to 8 tokens), a warp per row and
//   a 256-value super-block per warp step, each lane 8 values of it with their scales applied
//   per group (a sum of q x per scale, then times the scale: F32 sums, the CPU reference's values
//   up to summation order). The lanes' loads of a step cover the block's quants contiguously.
//   Their unroll (`KQ_UNROLL`, `KQ_UNROLL_WIDE`) is the knob to tune per GPU.
// - q5k_val / q5k_piece / linear_mv_q5k: the same for Q5_K. The generic paths take Q5_K only in
//   their K5 instantiations (glm5next.cu), so the GLM kernels compile to the same code as before.
// - linear_mv_q6k_st: linear_mv_q6k with each warp's row staged through shared memory in
//   16-byte loads (the same bits; the GB10's choice, #567). The staging idea follows Project
//   Maya's glm_fast.cu (MIT), which streams weight tiles through shared memory before decoding.

// Super-blocks a lane keeps in flight in the K-quant GEMVs (the unroll of their K loops).
constexpr int KQ_UNROLL = 4;
// The same for 5-8 tokens, whose activation loads already fill the registers.
constexpr int KQ_UNROLL_WIDE = 1;
template <int N> struct KqUnroll { static constexpr int v = N > 4 ? KQ_UNROLL_WIDE : KQ_UNROLL; };

__device__ __forceinline__ float h2f(const char * p) { return __half2float(*(const __half *) p); }

// Exact int -> float conversions without I2F (#574). I2F runs on the XU pipe at a quarter of the
// FMA rate, and the Q6_K GEMV spent most of that pipe on it. 0x4B000000 is 2^23, so a 6-bit x in
// the low mantissa bits is the float 2^23 + x exactly, and subtracting 2^23 + 32 leaves x - 32
// exactly (an integer in [-32, 31]). For a signed byte s, the bits 0x4B400000 + s are
// 1.5 * 2^23 + s exactly. Both give the float of (float) (x - 32) and (float) s: the same bits.
__device__ __forceinline__ float q6_f(uint32_t x) { return __int_as_float(0x4B000000 | x) - 8388640.0f; }
__device__ __forceinline__ float i8_f(int s) { return __int_as_float(0x4B400000 + s) - 12582912.0f; }
// q6_f of byte i of w (a 6-bit value): one PRMT puts the byte under the exponent bits.
__device__ __forceinline__ float q6_b(uint32_t w, int i) {
    return __int_as_float(__byte_perm(w, 0x4B000000, 0x7540 | i)) - 8388640.0f;
}

// get_scale_min_k4 of ggml-quants.c on the 12 scale bytes at `s`.
__device__ __forceinline__ void kq_scale_min(int j, const uint8_t * s, int & d, int & m) {
    if (j < 4) {
        d = s[j] & 63;
        m = s[j + 4] & 63;
    } else {
        d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// The same on a Q4_K block's first 16 bytes in registers (the scales in .y, .z, .w).
__device__ __forceinline__ int kq_sbyte(uint4 h, int b) {
    const uint32_t w = b < 4 ? h.y : (b < 8 ? h.z : h.w);
    return (int) ((w >> (8 * (b & 3))) & 0xFF);
}
__device__ __forceinline__ void kq_scale_min_r(int j, uint4 h, int & d, int & m) {
    if (j < 4) {
        d = kq_sbyte(h, j) & 63;
        m = kq_sbyte(h, j + 4) & 63;
    } else {
        d = (kq_sbyte(h, j + 4) & 0xF) | ((kq_sbyte(h, j - 4) >> 6) << 4);
        m = (kq_sbyte(h, j + 4) >> 4) | ((kq_sbyte(h, j) >> 6) << 4);
    }
}

// Q6_K: 210 bytes per 256 values: ql[128], qh[64], scales[16] (int8), d. In half n = v / 128 and
// quarter q = v % 128 / 32, value l < 32 is nibble q / 2 of ql[64 n + 32 (q % 2) + l] with bits
// 2 q .. 2 q + 1 of qh[32 n + l] above it, scale scales[8 n + 2 q + l / 16].
__device__ __forceinline__ float q6k_val(const char * row, int64_t i) {
    const uint8_t * b = (const uint8_t *) row + (i / 256) * 210;
    const int v = (int) (i % 256), n = v / 128, q = v % 128 / 32, l = v % 32;
    const int x = ((b[64 * n + 32 * (q & 1) + l] >> (4 * (q >> 1))) & 0xF) | (((b[128 + 32 * n + l] >> (2 * q)) & 3) << 4);
    return h2f((const char *) b + 208) * (float) (int8_t) b[192 + 8 * n + 2 * q + l / 16] * (float) (x - 32);
}

// Q4_K: 144 bytes per 256: d, dmin, scales[12], qs[128]. Values 64 c .. 64 c + 31 are the low
// nibbles of qs[32 c ..], scale 2 c; the next 32 the high nibbles, scale 2 c + 1.
__device__ __forceinline__ float q4k_val(const char * row, int64_t i) {
    const uint8_t * b = (const uint8_t *) row + (i / 256) * 144;
    const int v = (int) (i % 256), c = v / 64, hi = v % 64 / 32, l = v % 32;
    int sc, m;
    kq_scale_min(2 * c + hi, b + 4, sc, m);
    const float d1 = h2f((const char *) b) * sc, m1 = h2f((const char *) b + 2) * m;
    return d1 * (float) ((b[16 + 32 * c + l] >> (4 * hi)) & 0xF) - m1;
}

// Q5_K (DeepSeek-V4.1-Flash's dense weights, #584): 176 bytes per 256: d, dmin, scales[12],
// qh[32], qs[128]. Q4_K's layout with a fifth bit: value 64 c + 32 hi + l is nibble hi of
// qs[32 c + l] plus bit 2 c + hi of qh[l] times 16 (ggml-quants.c dequantize_row_q5_K, MIT).
__device__ __forceinline__ float q5k_val(const char * row, int64_t i) {
    const uint8_t * b = (const uint8_t *) row + (i / 256) * 176;
    const int v = (int) (i % 256), c = v / 64, hi = v % 64 / 32, l = v % 32;
    int sc, m;
    kq_scale_min(2 * c + hi, b + 4, sc, m);
    const float d1 = h2f((const char *) b) * sc, m1 = h2f((const char *) b + 2) * m;
    const int q = ((b[48 + 32 * c + l] >> (4 * hi)) & 0xF) | (((b[16 + l] >> (2 * c + hi)) & 1) << 4);
    return d1 * (float) q - m1;
}

__device__ __forceinline__ float kq_val(const char * row, int type, int64_t i) {
    return type == T_Q6_K ? q6k_val(row, i) : q4k_val(row, i);
}

// kq_piece of a Q5_K row (the generic paths' K5 instantiations; the others never see Q5_K, so
// their code stays what it was).
__device__ __forceinline__ void q5k_piece(const char * row, int64_t blk, int h, float * v) {
    const int v0 = (int) (blk % 8) * 32 + 16 * h;
    const uint8_t * b = (const uint8_t *) row + (blk / 8) * 176;
    const int c = v0 / 64, hi = v0 % 64 / 32, s = 2 * c + hi;
    int sc, m;
    kq_scale_min(s, b + 4, sc, m);
    const float d1 = h2f((const char *) b) * sc, m1 = h2f((const char *) b + 2) * m;
    // 16-byte aligned: rows of 176-byte blocks (launch_linear checks the base and strides)
    const uint4 u = *(const uint4 *) (b + 48 + 32 * c + 16 * h);
    const uint4 uh = *(const uint4 *) (b + 16 + 16 * h);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w}, wh[4] = {uh.x, uh.y, uh.z, uh.w};
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint32_t sh = 8 * (j % 4);
        const uint32_t q = ((w[j / 4] >> (sh + 4 * hi)) & 0xF) | (((wh[j / 4] >> (sh + s)) & 1) << 4);
        v[j] = d1 * (float) q - m1;
    }
}

// Values 16 h .. 16 h + 15 of the 32-value block `blk` of a Q6_K or Q4_K row.
__device__ __forceinline__ void kq_piece(const char * row, int type, int64_t blk, int h, float * v) {
    const int v0 = (int) (blk % 8) * 32 + 16 * h;
    if (type == T_Q6_K) {
        const uint8_t * b = (const uint8_t *) row + (blk / 8) * 210;
        const int n = v0 / 128, q = v0 % 128 / 32;
        const float ds = h2f((const char *) b + 208) * (float) (int8_t) b[192 + 8 * n + 2 * q + h];
        const uint8_t * ql = b + 64 * n + 32 * (q & 1) + 16 * h;
        const uint8_t * qh = b + 128 + 32 * n + 16 * h;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int x = ((ql[j] >> (4 * (q >> 1))) & 0xF) | (((qh[j] >> (2 * q)) & 3) << 4);
            v[j] = ds * (float) (x - 32);
        }
    } else {
        const uint8_t * b = (const uint8_t *) row + (blk / 8) * 144;
        const int c = v0 / 64, hi = v0 % 64 / 32;
        int sc, m;
        kq_scale_min(2 * c + hi, b + 4, sc, m);
        const float d1 = h2f((const char *) b) * sc, m1 = h2f((const char *) b + 2) * m;
        // 16-byte aligned: rows of 144-byte blocks (launch_linear checks the base and strides)
        const uint4 u = *(const uint4 *) (b + 16 + 32 * c + 16 * h);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 16; j++) {
            v[j] = d1 * (float) ((w[j / 4] >> (8 * (j % 4) + 4 * hi)) & 0xF) - m1;
        }
    }
}

// Q6_K rows for a few tokens: lane = 16 n + l / 2 takes values 128 n + 32 q + l, + 1 (q = 0..3) of
// each super-block: three 16-bit loads (ql[64 n + l], ql[64 n + 32 + l], qh[32 n + l]; blocks are
// 2-byte aligned) and its four scales; the activations in float2s.
// On V100 (#574) at least Q6K_MIN_BLOCKS CTAs per SM: left to itself the compiler took 139
// registers (3 CTAs, 12 warps per SM), too few warps to keep HBM busy (the kernel reached ~48% of
// DRAM bandwidth with no unit saturated). With the word-at-a-time unpack, 1-2 tokens fit 64
// registers without spilling: 8 CTAs (32 warps per SM) there, 7% less time for the GEMVs of MTP
// k = 1 decode steps; 4 tokens spill at 64 and keep 5 CTAs (8 already took 64). The register cap
// changes no arithmetic: the same bits.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
#define Q6K_MIN_BLOCKS(N) ((N) <= 2 ? 8 : 5)
#else
#define Q6K_MIN_BLOCKS(N) 1
#endif
template <int N>
__global__ void __launch_bounds__(128, Q6K_MIN_BLOCKS(N)) linear_mv_q6k(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const char * row = p.w + r * p.w_nb1 + i * p.w_nb2;
    const int n = lane / 16, l = (lane % 16) * 2;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 256;
#pragma unroll KqUnroll<N>::v
    for (int64_t b = 0; b < nb; b++) {
        const char * blk = row + b * 210;
        const uint32_t a = __ldg((const uint16_t *) (blk + 64 * n + l));
        const uint32_t c = __ldg((const uint16_t *) (blk + 64 * n + 32 + l));
        const uint32_t h = __ldg((const uint16_t *) (blk + 128 + 32 * n + l));
        const float d = __half2float(__ldg((const __half *) (blk + 208)));
        const int8_t * sc = (const int8_t *) (blk + 192 + 8 * n + l / 16);
        float ds[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            ds[q] = d * i8_f(__ldg(sc + 2 * q));
        }
        // value (quarter q, byte j): the eight 6-bit values put together a word at a time, bytes
        // (a0, a1, c0, c1) of the low nibbles with the high bits from (h0, h1, h0, h1), then
        // each byte a float through q6_b
        float v[4][2];
        {
            const uint32_t ac = __byte_perm(a, c, 0x5410), hh = __byte_perm(h, 0, 0x1010);
            const uint32_t w01 = (ac & 0x0F0F0F0F) | (__byte_perm(hh << 4, hh << 2, 0x7610) & 0x30303030);
            const uint32_t w23 = ((ac >> 4) & 0x0F0F0F0F) | (__byte_perm(hh, hh >> 2, 0x7610) & 0x30303030);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                v[0][j] = q6_b(w01, j);
                v[1][j] = q6_b(w01, 2 + j);
                v[2][j] = q6_b(w23, j);
                v[3][j] = q6_b(w23, 2 + j);
            }
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            if (t < p.n) {
                const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 256 + 128 * n + l;
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float2 xv = __ldg((const float2 *) (x + 32 * q));
                    s += ds[q] * (v[q][0] * xv.x + v[q][1] * xv.y);
                }
                acc[t] += s;
            }
        }
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// linear_mv_q6k with the weights staged through shared memory (#567, for the GB10's LPDDR5X):
// each warp copies KQ_STAGE super-blocks of its row (1680 bytes) with 16-byte streaming loads,
// the next group's loads in flight while its lanes compute the current one from shared memory
// exactly as linear_mv_q6k does (the same values per lane, the same order: the same bits). The
// row's 2-byte loads of linear_mv_q6k reach ~165 GB/s on a GB10, these ~220. Rows 16-byte
// aligned with a multiple of KQ_STAGE super-blocks (launch_linear checks).
constexpr int KQ_STAGE = 8;
constexpr int KQ_STAGE_V = KQ_STAGE * 210 / 16;  // uint4s per group
constexpr int KQ_STAGE_PER = (KQ_STAGE_V + 31) / 32;

template <int N>
__global__ void __launch_bounds__(128) linear_mv_q6k_st(const Lin p) {
    pdl_wait();
    __shared__ uint4 sm[4][KQ_STAGE_V];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const uint4 * row = (const uint4 *) (p.w + r * p.w_nb1 + i * p.w_nb2);
    const int n = lane / 16, l = (lane % 16) * 2;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t ng = p.k / (256 * KQ_STAGE);
    uint4 reg[KQ_STAGE_PER];
#pragma unroll
    for (int e = 0; e < KQ_STAGE_PER; e++) {
        if (lane + 32 * e < KQ_STAGE_V) {
            reg[e] = __ldcs(row + lane + 32 * e);
        }
    }
    const char * sb = (const char *) sm[warp];
    for (int64_t g = 0; g < ng; g++) {
#pragma unroll
        for (int e = 0; e < KQ_STAGE_PER; e++) {
            if (lane + 32 * e < KQ_STAGE_V) {
                sm[warp][lane + 32 * e] = reg[e];
            }
        }
        __syncwarp();
        if (g + 1 < ng) {
#pragma unroll
            for (int e = 0; e < KQ_STAGE_PER; e++) {
                if (lane + 32 * e < KQ_STAGE_V) {
                    reg[e] = __ldcs(row + (g + 1) * KQ_STAGE_V + lane + 32 * e);
                }
            }
        }
#pragma unroll
        for (int bb = 0; bb < KQ_STAGE; bb++) {
            const int64_t b = g * KQ_STAGE + bb;
            const char * blk = sb + bb * 210;
            const uint32_t a = *(const uint16_t *) (blk + 64 * n + l);
            const uint32_t c = *(const uint16_t *) (blk + 64 * n + 32 + l);
            const uint32_t h = *(const uint16_t *) (blk + 128 + 32 * n + l);
            const float d = __half2float(*(const __half *) (blk + 208));
            const int8_t * sc = (const int8_t *) (blk + 192 + 8 * n + l / 16);
            float ds[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                ds[q] = d * (float) sc[2 * q];
            }
            float v[4][2];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const uint32_t qa = (a >> (8 * j)) & 0xFF, qc = (c >> (8 * j)) & 0xFF, qh = (h >> (8 * j)) & 0xFF;
                v[0][j] = (float) ((int) ((qa & 0xF) | ((qh & 3) << 4)) - 32);
                v[1][j] = (float) ((int) ((qc & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                v[2][j] = (float) ((int) ((qa >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                v[3][j] = (float) ((int) ((qc >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            }
#pragma unroll
            for (int t = 0; t < N; t++) {
                if (t < p.n) {
                    const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 256 + 128 * n + l;
                    float s = 0.0f;
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float2 xv = __ldg((const float2 *) (x + 32 * q));
                        s += ds[q] * (v[q][0] * xv.x + v[q][1] * xv.y);
                    }
                    acc[t] += s;
                }
            }
        }
        __syncwarp();
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// Q4_K rows for a few tokens: lane = 8 g + j / 4 takes values 64 g + j .. + 3 (low nibbles of one
// 32-bit word of qs, scale 2 g) and 64 g + 32 + j .. + 3 (its high nibbles, scale 2 g + 1); the
// block header (d, dmin, scales) in one 16-byte load the warp shares; activations in float4s.
template <int N>
__global__ void __launch_bounds__(128) linear_mv_q4k(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const char * row = p.w + r * p.w_nb1 + i * p.w_nb2;
    const int g = lane / 8, j = (lane % 8) * 4;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 256;
#pragma unroll KqUnroll<N>::v
    for (int64_t b = 0; b < nb; b++) {
        const char * blk = row + b * 144;
        const uint4 hdr = __ldg((const uint4 *) blk);
        const uint32_t qw = __ldg((const uint32_t *) (blk + 16 + 32 * g + j));
        int sc0, m0, sc1, m1;
        kq_scale_min_r(2 * g, hdr, sc0, m0);
        kq_scale_min_r(2 * g + 1, hdr, sc1, m1);
        const float2 dm = __half22float2(*(const __half2 *) &hdr.x);
        const float d0 = dm.x * sc0, mm0 = dm.y * m0, d1 = dm.x * sc1, mm1 = dm.y * m1;
        float lo[4], hi[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            lo[e] = (float) ((qw >> (8 * e)) & 0xF);
            hi[e] = (float) ((qw >> (8 * e + 4)) & 0xF);
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            if (t < p.n) {
                const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 256 + 64 * g + j;
                const float4 xa = __ldg((const float4 *) x), xc = __ldg((const float4 *) (x + 32));
                const float qa = lo[0] * xa.x + lo[1] * xa.y + lo[2] * xa.z + lo[3] * xa.w;
                const float qc = hi[0] * xc.x + hi[1] * xc.y + hi[2] * xc.z + hi[3] * xc.w;
                const float sa = xa.x + xa.y + xa.z + xa.w, sc2 = xc.x + xc.y + xc.z + xc.w;
                acc[t] += d0 * qa - mm0 * sa + d1 * qc - mm1 * sc2;
            }
        }
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}

// Q5_K rows for a few tokens (DeepSeek-V4.1-Flash, #584): linear_mv_q4k's lanes and order, each
// lane's fifth bits from the 32-bit word of qh over its four values (bit 2 g for the low nibbles,
// 2 g + 1 for the high ones).
template <int N>
__global__ void __launch_bounds__(128) linear_mv_q5k(const Lin p) {
    pdl_wait();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * 4 + warp, i = blockIdx.y;
    if (r >= p.m) {
        return;
    }
    const char * row = p.w + r * p.w_nb1 + i * p.w_nb2;
    const int g = lane / 8, j = (lane % 8) * 4;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    const int64_t nb = p.k / 256;
#pragma unroll KqUnroll<N>::v
    for (int64_t b = 0; b < nb; b++) {
        const char * blk = row + b * 176;
        const uint4 hdr = __ldg((const uint4 *) blk);
        const uint32_t qw = __ldg((const uint32_t *) (blk + 48 + 32 * g + j));
        const uint32_t hw = __ldg((const uint32_t *) (blk + 16 + j));
        int sc0, m0, sc1, m1;
        kq_scale_min_r(2 * g, hdr, sc0, m0);
        kq_scale_min_r(2 * g + 1, hdr, sc1, m1);
        const float2 dm = __half22float2(*(const __half2 *) &hdr.x);
        const float d0 = dm.x * sc0, mm0 = dm.y * m0, d1 = dm.x * sc1, mm1 = dm.y * m1;
        float lo[4], hi[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            lo[e] = (float) (((qw >> (8 * e)) & 0xF) | (((hw >> (8 * e + 2 * g)) & 1) << 4));
            hi[e] = (float) (((qw >> (8 * e + 4)) & 0xF) | (((hw >> (8 * e + 2 * g + 1)) & 1) << 4));
        }
#pragma unroll
        for (int t = 0; t < N; t++) {
            if (t < p.n) {
                const float * x = (const float *) (p.x + i * p.x_nbi + t * p.x_nbt) + b * 256 + 64 * g + j;
                const float4 xa = __ldg((const float4 *) x), xc = __ldg((const float4 *) (x + 32));
                const float qa = lo[0] * xa.x + lo[1] * xa.y + lo[2] * xa.z + lo[3] * xa.w;
                const float qc = hi[0] * xc.x + hi[1] * xc.y + hi[2] * xc.z + hi[3] * xc.w;
                const float sa = xa.x + xa.y + xa.z + xa.w, sc2 = xc.x + xc.y + xc.z + xc.w;
                acc[t] += d0 * qa - mm0 * sa + d1 * qc - mm1 * sc2;
            }
        }
    }
    const float sc = lin_scale(p, r);
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < p.n) {
            p.dst[(t * p.h + i) * p.m + r] = s * sc;
        }
    }
}
