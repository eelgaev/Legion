// q8_mmq.cu, sm_80 and newer: the op's Q8_0 matmul with ggml's MMQ bits from a streaming main
// loop (#858): mma.sync m16n8k32 s8 x s8 -> s32 on weights streamed by 16-byte cp.async
// through a two-stage shared-memory ring and activations read through L1.
//
// Not a standalone header: q8_mmq.cu includes it inside its anonymous namespace, after
// `kstart` and the `legion_q8_fast` struct. The kernel bodies are built only where
// __CUDA_ARCH__ >= 800 (empty elsewhere); src/model/q8_mmq.rs takes them on compute
// capability 12.x only.
//
// The bits. ggml's MMQ (mmq-vec-dot.cuh, ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma) sums, for every
// output, `sum += C * dA * dB` over the Q8_0 blocks of K in order, C the block's exact integer
// dot product, dA the weight's and dB the activation's block scale; stream-k splits a tile's K
// blocks over CTAs at the points `kstart` gives, and the fixup adds the pieces as
// dst = last + (((0 + P[b - 1]) + P[b - 2]) + ...). Here every CTA `b` (of ggml's grid) takes
// ggml's CTA b's K blocks of the same tiles, in the same order, with the same expression per
// block, and the last CTA to finish a split tile adds its pieces in the fixup's order: the
// same sums, so the same bits. How the products are fetched is free:
// - The 32 products of a block are added in integers (exact), so their order within the
//   block is free. A Q8_0 block is 34 bytes; two blocks are 17 aligned words: w0 = (scale,
//   q0, q1), w1 .. w7 = q2 .. q29, w8 = (q30, q31, next scale), w9 .. w16 = the odd block's
//   quants. So the even block's eight 4-byte K slots are w1 .. w7 and (q0, q1, q30, q31), the
//   odd block's w9 .. w16: whole-word loads, one byte permute. The quantization (`fquant`,
//   ggml's quantize_mmq_q8_1 values) writes the activations' quants in the same slot order.
// - The integer dot product starts at 0x4B400000 (1.5 * 2^23 as a float's bits) in the MMA's
//   accumulator: |C| <= 32 * 128 * 128 < 2^22, so the result's bits are the float
//   12582912 + C, and subtracting 12582912 gives C exactly (ggml's I2F, without the
//   conversion pipe).
//
// Layout: a CTA has NW warps, each 16 rows of the tile (rows g and g + 8 of an m16n8k32) times
// all J columns; gridDim.y = 128 / (16 NW) CTAs share a ggml CTA's work by rows. Per K stage
// of 8 blocks (ggml's K iteration, so stream-k's pieces are whole stages): the CTA's rows'
// 272 bytes each (17 16-byte copies, 16-byte aligned for w_ld a multiple of 8) into the ring,
// rows 68 words apart (4 banks: the 32 lanes' fragment reads are conflict-free); the
// activations' quants [8 words] and scales of the J columns by __ldg, prefetched to L1 with
// the stage's copies.

constexpr int FAST_MAGIC_BITS = 0x4B400000;
constexpr float FAST_MAGIC    = 12582912.0f;

// Blocks per stage (four block pairs: 272 bytes of a row, ggml's K iteration).
constexpr int FAST_SB = 8;

__device__ __forceinline__ void fast_cp16(void * smem, const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
#endif
}

__device__ __forceinline__ void fast_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void fast_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void fast_mma(int (&c)[4], const uint32_t (&a)[4], const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// The words of one weight row's block pair that lane q (of 4 on the row) needs: w0, w8 (the
// scales, and the even block's slot 7), the even block's slots q and q + 4 (w[1 + q], and
// w[5 + q] for q < 3), the odd block's slots q and q + 4 (w[9 + q], w[13 + q]).
struct FastPair {
    uint32_t w0, w8, e_lo, e_hi, o_lo, o_hi;
};

// From a row's block pair in shared memory (17 words).
__device__ __forceinline__ FastPair fast_load_pair(const uint32_t * p, const int q) {
    FastPair r;
    r.w0   = p[0];
    r.w8   = p[8];
    r.e_lo = p[1 + q];
    r.e_hi = p[q < 3 ? 5 + q : 8];
    r.o_lo = p[9 + q];
    r.o_hi = p[13 + q];
    return r;
}

__device__ __forceinline__ void fast_prefetch_l1(const void * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("prefetch.global.L1 [%0];\n" ::"l"(gmem));
#endif
}

// The quantization for the fast kernels: ggml's quantize_mmq_q8_1 values (each 32-value block's
// absmax over 8 threads of 4 values, d_inv = 127 / amax, roundf(x * d_inv), d = 1 / d_inv), the
// quants written in the K slot order above to yq [k / 32][n_pad][8 words] (word 2s: slot s,
// word 2s + 1: slot s + 4, so a lane reads its two slots in one 8-byte load), the scales to yd
// [k / 32][n_pad]. Grid (n, k / 256), 64 threads; also zeroes the tile counters.
__global__ void __launch_bounds__(64) fquant(const legion_q8_fast p) {
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int t = threadIdx.x; t < p.ncnt; t += blockDim.x) {
            p.cnt[t] = 0;
        }
    }
    const int i0 = blockIdx.y * 256 + threadIdx.x * 4;
    const int col = blockIdx.x;
    const float * src = p.x + col * p.x_ld + i0;
    const float4 xi = (reinterpret_cast<uintptr_t>(src) & 15) == 0
        ? *reinterpret_cast<const float4 *>(src)
        : make_float4(src[0], src[1], src[2], src[3]);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    const int kb   = i0 / 32;
    const int s    = threadIdx.x % 8;  // this thread's 4 values: 4s .. 4s + 3 of block kb
    const int mine = *reinterpret_cast<const int *>(&q);
    const int lane = threadIdx.x % 32;
    const int next = __shfl_sync(0xFFFFFFFF, mine, (lane & ~7) | ((lane + 1) & 7));
    int word;
    if (kb % 2 == 1) {
        word = mine;                                  // slot s: values 4s .. 4s + 3
    } else if (s < 7) {
        word = __byte_perm(mine, next, 0x5432);       // slot s: values 4s + 2 .. 4s + 5
    } else {
        word = __byte_perm(next, mine, 0x7610);       // slot 7: values 0, 1, 30, 31
    }
    const int64_t at = (int64_t) kb * p.n_pad + col;
    p.yq[at * 8 + (s < 4 ? 2 * s : 2 * (s - 4) + 1)] = word;
    if (s == 0) {
        p.yd[at] = d;
    }
}

// CTA b's work of ggml's grid (gridDim.x = ggml's CTAs), rows blockIdx.y * 16 NW .. of each tile.
template <int J, int NW>
__global__ void __launch_bounds__(NW * 32) fast(const legion_q8_fast p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NT      = J / 8;
    constexpr int RB      = NW * 16;
    constexpr int S       = 128 / RB;
    constexpr int ROW     = FAST_SB * 17 / 2; // words of a row's stage (FAST_SB blocks)
    constexpr int STAGE   = RB * ROW;         // words of a stage
    constexpr int CHUNKS  = ROW / 4;          // 16-byte copies of a row's stage
    constexpr int THREADS = NW * 32;
    static_assert(ROW % 4 == 0 && ROW % 32 == 4, "16-byte copies; rows 4 banks apart");
    static_assert(J % 8 == 0 && 128 % RB == 0, "tile shape");

    extern __shared__ __align__(16) int fsm[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int b = blockIdx.x, sub = blockIdx.y;
    const int bpt = p.k / 32;
    const int64_t total = (int64_t) p.ntiles * bpt;
    const int kbc0 = kstart(b,     p.ctas, total, bpt, 8);
    const int kbc1 = kstart(b + 1, p.ctas, total, bpt, 8);

    for (int t = kbc0 / bpt; kbc0 < kbc1 && t * bpt < kbc1; t++) {
        const int t0 = t * bpt;
        const int s  = max(kbc0, t0) - t0;
        const int e  = min(kbc1, t0 + bpt) - t0;
        const int jt = t % p.ntx, it = t / p.ntx;
        const int col0  = jt * J;
        const int rbase = it * 128 + sub * RB + warp * 16;
        const int nst = (e - s) / FAST_SB;
        const int row0 = it * 128 + sub * RB;  // the CTA's first row

        // Stage st's weights (blocks s + FAST_SB st .. of the CTA's RB rows) into ring slot
        // st % 2 by cp.async, and its activations toward L1.
        auto issue = [&](const int st) {
            if (st < nst) {
                int * dw = fsm + (st % 2) * STAGE;
                const int kb0 = s + st * FAST_SB;
                for (int c = tid; c < RB * CHUNKS; c += THREADS) {
                    const int r = c / CHUNKS, ch = c % CHUNKS;
                    const int gr = min(row0 + r, p.rows - 1);
                    fast_cp16(dw + r * ROW + ch * 4,
                              p.w + ((int64_t) gr * p.w_ld + kb0) * 34 + ch * 16);
                }
                // The activations: J * 32 bytes of quants and J * 4 of scales a block.
                constexpr int LQ = J * 32 / 128 > 0 ? J * 32 / 128 : 1;
                for (int c = tid; c < FAST_SB * (LQ + 1); c += THREADS) {
                    const int bl = c / (LQ + 1), l = c % (LQ + 1);
                    const int64_t at = (int64_t) (kb0 + bl) * p.n_pad + col0;
                    fast_prefetch_l1(l < LQ ? (const void *) (p.yq + at * 8 + l * 32)
                                            : (const void *) (p.yd + at));
                }
            }
            fast_commit();
        };

        float acc[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                acc[nt][l] = 0.0f;
            }
        }

        issue(0);
        for (int st = 0; st < nst; st++) {
            issue(st + 1);
            fast_wait<1>();
            __syncthreads();

            const uint32_t * sw = (const uint32_t *) (fsm + (st % 2) * STAGE + (warp * 16 + g) * ROW);
            const int kb0 = s + st * FAST_SB;
#pragma unroll
            for (int bl = 0; bl < FAST_SB; bl++) {
                const FastPair x0 = fast_load_pair(sw + (bl / 2) * 17, q);
                const FastPair x1 = fast_load_pair(sw + 8 * ROW + (bl / 2) * 17, q);
                uint32_t a[4];
                float da0, da1;
                if (bl % 2 == 0) {
                    a[0] = x0.e_lo;
                    a[1] = x1.e_lo;
                    a[2] = q < 3 ? x0.e_hi : __byte_perm(x0.w0, x0.w8, 0x5432);
                    a[3] = q < 3 ? x1.e_hi : __byte_perm(x1.w0, x1.w8, 0x5432);
                    da0 = __half2float(__ushort_as_half((unsigned short) (x0.w0 & 0xFFFF)));
                    da1 = __half2float(__ushort_as_half((unsigned short) (x1.w0 & 0xFFFF)));
                } else {
                    a[0] = x0.o_lo;
                    a[1] = x1.o_lo;
                    a[2] = x0.o_hi;
                    a[3] = x1.o_hi;
                    da0 = __half2float(__ushort_as_half((unsigned short) (x0.w8 >> 16)));
                    da1 = __half2float(__ushort_as_half((unsigned short) (x1.w8 >> 16)));
                }
                const int64_t at = (int64_t) (kb0 + bl) * p.n_pad + col0;
                const int   * bq = p.yq + at * 8;
                const float * bd = p.yd + at;
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const int2   bv = __ldg(reinterpret_cast<const int2 *>(bq + (nt * 8 + g) * 8 + 2 * q));
                    const float2 dv = __ldg(reinterpret_cast<const float2 *>(bd + nt * 8 + 2 * q));
                    int c[4] = { FAST_MAGIC_BITS, FAST_MAGIC_BITS, FAST_MAGIC_BITS, FAST_MAGIC_BITS };
                    fast_mma(c, a, bv.x, bv.y);
                    const float c0 = __int_as_float(c[0]) - FAST_MAGIC;
                    const float c1 = __int_as_float(c[1]) - FAST_MAGIC;
                    const float c2 = __int_as_float(c[2]) - FAST_MAGIC;
                    const float c3 = __int_as_float(c[3]) - FAST_MAGIC;
                    // ggml's `sum += C.x[l]*dA*dB`.
                    acc[nt][0] += c0 * da0 * dv.x;
                    acc[nt][1] += c1 * da0 * dv.y;
                    acc[nt][2] += c2 * da1 * dv.x;
                    acc[nt][3] += c3 * da1 * dv.y;
                }
            }
            __syncthreads();  // slot st % 2 read before stage st + 2's copies overwrite it
        }
        fast_wait<0>();
        __syncthreads();  // the ring is free (the piece list below, the next tile's stages)

        // The piece: the tile's end in dst (as ggml's non-fixup write back), else this CTA's
        // partial sum in part[b] (ggml's tmp_fixup), [J][128].
        const bool last = e == bpt;
        float * pb = p.part + (int64_t) b * (J * 128);
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = rbase + g + (l >> 1) * 8;
                const int j = col0 + nt * 8 + 2 * q + (l & 1);
                if (last) {
                    if (i < p.rows && j < p.n) {
                        p.dst[(int64_t) j * p.rows + i] = acc[nt][l];
                    }
                } else {
                    pb[(j - col0) * 128 + (i - it * 128)] = acc[nt][l];
                }
            }
        }
        if ((s == 0 && last) || p.sep) {
            continue;  // a whole tile, or the pieces are left for `split_fixup`
        }

        // A split tile: count this piece (per row slice); the CTA that counts last adds the
        // pieces up in ggml's fixup order.
        // [0]: whether this CTA counted last, [1]: pieces before the end, [2], [3]: the first and
        // the last CTA holding a piece, then the CTAs before the last, nearest first (-1: none).
        int * list = fsm;
        __threadfence();
        if (tid == 0) {
            list[1] = 0;
            list[2] = p.ctas;
            list[3] = -1;
        }
        __syncthreads();
        // CTA bb holds a piece of tile t: its K blocks are not empty and overlap the tile's. The
        // CTAs' starts in parallel (a 64-bit division each), not one thread walking them.
        int held = 0;  // bit r: CTA tid + r THREADS holds a piece
        for (int r = 0; r * THREADS < p.ctas; r++) {
            const int bb = tid + r * THREADS;
            if (bb < p.ctas) {
                const int ks = kstart(bb,     p.ctas, total, bpt, 8);
                const int ke = kstart(bb + 1, p.ctas, total, bpt, 8);
                if (ks < ke && ke > t0 && ks < t0 + bpt) {
                    held |= 1 << r;
                    atomicAdd(&list[1], 1);
                    atomicMin(&list[2], bb);
                    atomicMax(&list[3], bb);
                }
            }
        }
        __syncthreads();
        const int bf = list[2], be = list[3], m = list[1] - 1;
        for (int r = 0; r * THREADS < p.ctas; r++) {
            const int bb = tid + r * THREADS;
            if (bb >= bf && bb < be) {
                list[4 + (be - 1 - bb)] = (held >> r) & 1 ? bb : -1;
            }
        }
        if (tid == 0) {
            list[0] = atomicAdd(&p.cnt[t * S + sub], 1) == m ? 1 : 0;
        }
        __syncthreads();
        if (list[0]) {
            __threadfence();
            fixup_add<J, RB, THREADS>(p.part, list + 4, be - bf, p.dst, p.rows, p.n, it, jt, sub * RB);
        }
        __syncthreads();  // the list read before the next tile's stages overwrite it
    }
#else
    NO_DEVICE_CODE;
#endif
}
