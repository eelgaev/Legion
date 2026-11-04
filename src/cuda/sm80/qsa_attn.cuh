// qsa_attn.cu, sm_80 and newer: the prompt kernel on mma.sync.m16n8k16 with a cp.async ring
// (attn_mma_kernel).
//
// Not a standalone header: qsa_attn.cu includes it inside its anonymous namespace, after the
// shared constants (THREADS, ROWS) and sm70/qsa_attn.cuh (pack_h2). The MMA and cp.async keep
// their __CUDA_ARCH__ guards and __trap() stubs, so the file compiles for every target
// architecture; use_mma<HD>() in qsa_attn.cu picks this kernel per device at run time.

// ---------------------------------------------------------------------------------------------
// The prompt kernel for sm_80 and newer: the same op on mma.sync.m16n8k16 (F16 in, F32 out),
// with K / V gathered through a two-stage cp.async ring. Structure adapted from Strata's
// prompt_attn_kernel (src/kernels/cuda/qsa_prompt_attn.cu, MIT, Copyright (c) 2026 Niko1221 and
// the Strata contributors): one CTA of 4 warps per (token, kv head), the G query heads padded
// to 16 rows; scores with warp w on cells 8w..8w+7 of a 32-cell chunk over all dims; an online
// base-2 softmax per row; p.v with warp w on dims [w HD / 4, (w + 1) HD / 4).
//
// Precision as the Volta kernel: q and p are split into F16 hi + lo halves (each its own
// product), K and V are the cache's F16 values. The tensor cores of sm_12x (GB10) truncate
// when they add into an accumulator (see paged_attn.cu), so every product chain is short: a
// score's 16-wide K step (hi then lo) and a chunk's p.v (2 K steps, hi then lo) start from a
// zero fragment and are added to the F32 totals with ordinary adds.
//
// Rows of K / V in shared memory are HD + 8 halves apart (16-byte aligned, banks spread): the
// fragment loads below are conflict-free.

constexpr int MCH = 32;  // cells per chunk (4 warps x 8)

template <int HD>
struct MmaSmem {
    static constexpr int R = HD + 8;  // halves per q / K / V row
    __half qh[ROWS][R];
    __half ql[ROWS][R];
    __half k[2][MCH][R];
    __half v[2][MCH][R];
    float s[ROWS][MCH + 4];
    float qmax[THREADS / 32];
    float alpha[ROWS];
    float lsum[ROWS];
    float mrow[ROWS];
};

__device__ __forceinline__ void mma16816(float * c, const uint32_t * a, const uint32_t * b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

// 16 bytes global -> shared, or zeros (src_bytes 0).
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t dst = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem),
                 "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <int HD>
__global__ void __launch_bounds__(THREADS) attn_mma_kernel(legion_qsa_attn p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    MmaSmem<HD> & S = *reinterpret_cast<MmaSmem<HD> *>(smem_raw);
    constexpr int PIECES = HD * 2 / 16;              // 16-byte pieces per K or V row
    constexpr int ROWS_PER_PASS = THREADS / PIECES;  // rows a pass of the CTA copies
    constexpr int DW = HD / 4;                       // p.v dims per warp
    constexpr int NT = DW / 8;                       // p.v n-tiles per warp
    static_assert(THREADS % PIECES == 0 && MCH % ROWS_PER_PASS == 0, "copy layout");
    const int tok = blockIdx.x, kvh = blockIdx.y;
    const int G = p.n_head / p.n_head_kv;
    const int s = tok / p.n_tok, ti = tok % p.n_tok;
    const char * qb = (const char *) p.q + (size_t) s * p.q_nb3 + (size_t) ti * p.q_nb2 +
                      (size_t) kvh * G * p.q_nb1;
    float * out = p.dst + ((size_t) tok * p.n_head + (size_t) kvh * G) * HD;
    const int32_t * sel = p.sel + (size_t) tok * p.sel_row;
    const int n = sel[0];
    const int32_t * ids = sel + 1;
    const int32_t * table = p.table + (size_t) s * p.table_width;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // Chunk c0's K and V rows into stage st: thread t copies piece t % PIECES of rows
    // t / PIECES, + ROWS_PER_PASS, ...; cells past the selection are zeros.
    auto issue = [&](int c0, int st) {
        const int pc = t % PIECES;
#pragma unroll
        for (int c = t / PIECES; c < MCH; c += ROWS_PER_PASS) {
            const bool valid = c0 + c < n;
            long long r = 0;
            if (valid) {
                const int pos = __ldg(ids + c0 + c);
                const long long cell = (long long) __ldg(table + pos / BLK) * BLK + pos % BLK;
                r = cell * p.kv_row + (long long) kvh * HD;
            }
            cp_async16(&S.k[st][c][pc * 8], reinterpret_cast<const uint4 *>(p.k + r) + pc, valid);
            cp_async16(&S.v[st][c][pc * 8], reinterpret_cast<const uint4 *>(p.v + r) + pc, valid);
        }
        cp_async_commit();
    };
    if (n > 0) {
        issue(0, 0);
    }

    // q: G heads + zero rows, times a power of two that puts its largest value near 2^14
    // (exact, and the lo halves stay out of F16's subnormal range), split into hi + lo.
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) {
        qm = fmaxf(qm, fabsf(*(const float *) (qb + (size_t) (i / HD) * p.q_nb1 + (i % HD) * 4)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    }
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) {
        frexpf(qm, &qe);  // qm < 2^qe
    }
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(p.scale_log2, qe - 14);
    for (int i = t; i < ROWS * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x =
            h < G ? *(const float *) (qb + (size_t) h * p.q_nb1 + d * 4) * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < ROWS) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }

    for (int c0 = 0, st = 0; c0 < n; c0 += MCH, st ^= 1) {
        const int nh = min(MCH, n - c0);
        // Every warp is done with the other stage (the previous chunk): refill it.
        __syncthreads();
        if (c0 + MCH < n) {
            issue(c0 + MCH, st ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        // scores: warp w = cells 8w..8w+7, every dim
        {
            const int cb = warp * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k0 = 0; k0 < HD; k0 += 16) {
                uint32_t ah[4], al[4], b[2];
                ah[0] = *reinterpret_cast<const uint32_t *>(&S.qh[gid][k0 + 2 * tig]);
                ah[1] = *reinterpret_cast<const uint32_t *>(&S.qh[gid + 8][k0 + 2 * tig]);
                ah[2] = *reinterpret_cast<const uint32_t *>(&S.qh[gid][k0 + 2 * tig + 8]);
                ah[3] = *reinterpret_cast<const uint32_t *>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                al[0] = *reinterpret_cast<const uint32_t *>(&S.ql[gid][k0 + 2 * tig]);
                al[1] = *reinterpret_cast<const uint32_t *>(&S.ql[gid + 8][k0 + 2 * tig]);
                al[2] = *reinterpret_cast<const uint32_t *>(&S.ql[gid][k0 + 2 * tig + 8]);
                al[3] = *reinterpret_cast<const uint32_t *>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                b[0] = *reinterpret_cast<const uint32_t *>(&S.k[st][cb + gid][k0 + 2 * tig]);
                b[1] = *reinterpret_cast<const uint32_t *>(&S.k[st][cb + gid][k0 + 2 * tig + 8]);
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
                mma16816(tg, ah, b);
                mma16816(tg, al, b);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sc[i] += tg[i];
                }
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        // online softmax (base 2): row t / 8, MCH / 8 cells per thread
        {
            constexpr int PER = MCH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: p (at most 1) times 2^14 so its lo half stays normal, split into hi + lo
        {
            constexpr float VUP = 16384.0f, VDOWN = 1.0f / 16384.0f;
            float tmp[NT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
            }
#pragma unroll
            for (int k0 = 0; k0 < MCH; k0 += 16) {
                const int ca = k0 + 2 * tig, cbb = ca + 8;
                const float p00 = S.s[gid][ca] * VUP, p01 = S.s[gid][ca + 1] * VUP;
                const float p10 = S.s[gid + 8][ca] * VUP, p11 = S.s[gid + 8][ca + 1] * VUP;
                const float p02 = S.s[gid][cbb] * VUP, p03 = S.s[gid][cbb + 1] * VUP;
                const float p12 = S.s[gid + 8][cbb] * VUP, p13 = S.s[gid + 8][cbb + 1] * VUP;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2 * h = reinterpret_cast<const __half2 *>(ah);
                    float2 f;
                    f = __half22float2(h[0]);
                    al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]);
                    al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]);
                    al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]);
                    al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int d = warp * DW + j * 8 + gid;
                    const __half2 h0 = __halves2half2(S.v[st][ca][d], S.v[st][ca + 1][d]);
                    const __half2 h1 = __halves2half2(S.v[st][cbb][d], S.v[st][cbb + 1][d]);
                    uint32_t b[2];
                    b[0] = *reinterpret_cast<const uint32_t *>(&h0);
                    b[1] = *reinterpret_cast<const uint32_t *>(&h1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * VDOWN);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * VDOWN);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * VDOWN);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * VDOWN);
            }
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int d = warp * DW + j * 8 + 2 * tig;
        if (gid < G) {
            *reinterpret_cast<float2 *>(out + (size_t) gid * HD + d) =
                make_float2(acc[j][0] * i0, acc[j][1] * i0);
        }
        if (gid + 8 < G) {
            *reinterpret_cast<float2 *>(out + (size_t) (gid + 8) * HD + d) =
                make_float2(acc[j][2] * i1, acc[j][3] * i1);
        }
    }
}
