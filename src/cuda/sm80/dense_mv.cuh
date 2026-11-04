// dense_mv.cu, sm_80 and newer: the kernel, ggml-cuda's `mul_mat_f` (mmf.cuh, the
// mma path for BF16) in the same work split and order, so the same bits:
// - a CTA of nw warps per tile of 16 * NTA rows; warp p takes the 32-pair chunks p, p + nw, ...
//   of the rows (ggml: the column `col` of thread x of warp p is p * 32 + x + i * nw * 32);
// - per chunk the warp stages its weight rows and the tokens' activations (rounded to BF16, 8
//   columns per tile, zeros past n) in shared memory with ggml's padded layout, loads the A and
//   B fragments with ldmatrix and runs m16n8k16 for k0 = 0, 8, 16, 24 (pairs), F32 sums;
// - the warps' sums go through shared memory and are added in warp order, starting from 0.
// ggml stores 32-row tiles; a tile's rows are independent, so 16-row tiles (NTA = 1) give the
// same sums. What differs from ggml: the weights are read with ld.global.nc, D chunks ahead of
// their use, into registers.
// F32 weights (F, #836: the MoE router): ggml's mul_mat_f<float> in the same way. A chunk is 32
// floats instead of 32 pairs, the activations go to shared memory as they are, and each k0 step
// is one m16n8k8 TF32 product of the raw F32 bits (mma.cuh's tile<16, 8, float>); the staging
// layout in 4-byte words, the ldmatrix addresses and the epilogue are the BF16 kernel's.
//
// Not a standalone header: dense_mv.cu includes it inside its anonymous namespace, after
// legion_dense_mv, PAD and pick. The kernel bodies are built only where __CUDA_ARCH__ >= 800 and
// trap elsewhere; mma_code() in dense_mv.cu checks at run time that this binary has them.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t ldg_nc(const void * p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void * p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The k16 product of mma_bf16 as two m16n8k8 products in k order (a probe of cuBLAS's order).
__device__ __forceinline__ void mma_bf16_k8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[2]), "r"(a[3]), "r"(b[1]));
}

// ggml's F32 tile product (mma.cuh: tile<16, 8, float> x tile<8, 8, float>): the raw F32 bits
// as TF32 operands.
__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#if __CUDA_ARCH__ >= 900
// Every thread of the cluster: the CTAs' shared-memory writes before here are seen after it.
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// The float at `p` (this CTA's shared memory) in CTA `rank` of the cluster.
__device__ __forceinline__ float ld_cluster(const float * p, int rank) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    unsigned r;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r) : "memory");
    return v;
}
#endif

// Chunks of a chain tile's weight rows in flight.
constexpr int CHAIN_D = 4;

// Chain tile `tile` (16 rows of the chain weights, dense_mv.cu) for up to 16 tokens, on one warp:
// cuBLAS's one-warp wmma GEMM (16 x 16 x 16 BF16 steps, F32 sums) in its order. Each output is
// one chain of m16n8k16 products over k = 0, 16, 32, ... starting from 0, written as it is (no
// epilogue add). The staging is mmf's (its layout, ldmatrix, x rounded to BF16 as ggml rounds
// it for cuBLAS), with chunks of 64 k (32 pairs) taken in order by the one warp. `st`: the
// warp's staging area, 32 rows of PAD words.
__device__ __forceinline__ void chain_tile(const legion_dense_mv & p, int64_t tile, bf162 * st) {
    const int lane = threadIdx.x, n = p.n;
    const int wi = pick(p, tile, 16, p.nw, p.nw - p.nchain);
    const int64_t row0 = tile * 16;
    const int64_t ld2 = p.w_ld[wi] / 2, y_ld2 = p.x_ld / 2;
    const uint32_t * xw = (const uint32_t *) p.w[wi] + row0 * ld2 + lane;
    const float2 * y2 = (const float2 *) p.x;
    const int nc = (int) (p.k / 64);
    bf162 * As = st;
    bf162 * Bs = st + 16 * PAD;
    float C[2][4];
#pragma unroll
    for (int b = 0; b < 2; b++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            C[b][l] = 0.0f;
        }
    }
    uint32_t ar[CHAIN_D][16];
    auto load = [&](uint32_t (&r)[16], int c) {
        if (c < nc) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                r[i] = ldg_nc(xw + i * ld2 + c * 32);
            }
        }
    };
#pragma unroll
    for (int d = 0; d < CHAIN_D; ++d) {
        load(ar[d], d);
    }
    for (int base = 0; base < nc; base += CHAIN_D) {
#pragma unroll
        for (int d = 0; d < CHAIN_D; ++d) {
            const int c = base + d;
            if (c >= nc) {
                break;
            }
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                ((uint32_t *) As)[i * PAD + lane] = ar[d][i];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float2 t = j < n ? y2[j * y_ld2 + c * 32 + lane] : make_float2(0.0f, 0.0f);
                Bs[j * PAD + lane] = __float22bfloat162_rn(t);
            }
            load(ar[d], c + CHAIN_D);
            __syncwarp();
            uint32_t A[4][4];
#pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 8) {
                ldsm_x4(A[k0 / 8], As + (lane % 16) * PAD + (lane / 16) * 4 + k0);
            }
            // Per output the k steps in order; the two token halves are separate chains.
#pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 8) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    uint32_t B[2];
                    ldsm_x2(B, Bs + (b * 8 + lane % 8) * PAD + ((lane / 8) * 4) % 8 + k0);
                    mma_bf16(C[b], A[k0 / 8], B);
                }
            }
        }
    }
    float * dst = p.dst[wi] + row0;
    const int64_t dst_ld = p.rows[wi];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int i = lane / 4 + (l / 2) * 8;
            const int j = b * 8 + 2 * (lane % 4) + (l % 2);
            if (j < n) {
                dst[j * dst_ld + i] = C[b][l];
            }
        }
    }
}
#endif

// Tiles of 16 rows over the chain weights (dense_mv.cu): the launch's first CTAs.
__host__ __device__ __forceinline__ int64_t chain_tiles(const legion_dense_mv & p) {
    int64_t t = 0;
    for (int i = p.nw - p.nchain; i < p.nw; i++) {
        t += p.rows[i] / 16;
    }
    return t;
}

// Warps (16-row tiles) per CTA of chain_wide.
constexpr int WIDE_WARPS = 4;
// Chunks of a wide chain tile's weight rows in flight.
constexpr int WIDE_D = 4;

// Chain tiles for 17 to 8 * NT tokens (#836, `legion serve --dense-mv-wide-chain`): every weight
// a chain weight, all of them computed as chain_tile computes one (cuBLAS's wmma order: one chain
// of m16n8k16 products over k = 0, 16, 32, ... from 0 per output, x rounded to BF16, written as it
// is), each warp one 16-row tile. The CTA's WIDE_WARPS warps read the same x, so it is staged once
// per chunk in shared memory for all of them (its next chunk loaded into registers meanwhile);
// each warp keeps WIDE_D chunks of its weight rows in flight. blockDim = (32, WIDE_WARPS), one CTA
// per WIDE_WARPS tiles (warps past the last tile only stage).
// KS (p.kslice > 0): cuBLAS's split-K kernels instead, each slice of p.kslice k values its own
// chain from 0, the slices' sums added in order ((s0 + s1) + s2 ...), all in the warp's registers.
// IA > 1 (a probe of cuBLAS's order, dense-mv-bench): within a slice the k steps of 16 go to
// p.kinter accumulators in turn, p.kgroup k values each, added in order at the slice's end.
template <int NT, bool KS, int IA = 1>
__global__ void __launch_bounds__(32 * WIDE_WARPS) chain_wide(const legion_dense_mv p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NJ = NT * 8;
    constexpr int XR = NJ * 32 / (32 * WIDE_WARPS);  // x pairs a thread stages per chunk
    static_assert(NJ * 32 % (32 * WIDE_WARPS) == 0, "x staging split");
    static_assert(KS || IA == 1, "accumulators only with slices");
    __shared__ __align__(16) bf162 Bs[NJ * PAD];
    __shared__ __align__(16) bf162 Asm[WIDE_WARPS][16 * PAD];
    const int lane = threadIdx.x, wl = threadIdx.y, n = p.n;
    const int t = wl * 32 + lane;
    int64_t tile = (int64_t) blockIdx.x * WIDE_WARPS + wl;
    const bool live = tile < chain_tiles(p);
    const int wi = live ? pick(p, tile, 16, p.nw) : 0;
    const int64_t row0 = tile * 16;
    const int64_t ld2 = p.w_ld[wi] / 2, y_ld2 = p.x_ld / 2;
    const uint32_t * xw = (const uint32_t *) p.w[wi] + (live ? row0 * ld2 : 0) + lane;
    const float2 * y2 = (const float2 *) p.x;
    const int nc = (int) (p.k / 64);
    bf162 * As = Asm[wl];
    float C[IA][NT][4];
    // KS: the sum of the split-K slices before the current one.
    float S[KS ? NT : 1][4];
#pragma unroll
    for (int a = 0; a < IA; a++) {
#pragma unroll
        for (int b = 0; b < NT; b++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                C[a][b][l] = 0.0f;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < (KS ? NT : 1); b++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            S[b][l] = 0.0f;
        }
    }
    // C[0] += the other accumulators, in order (IA > 1).
    auto join = [&]() {
        if (IA > 1 && p.kred == 1) {
            // Pairwise: (a0 + a1) + (a2 + a3), ... (kinter a power of two).
#pragma unroll
            for (int h = 1; h < IA; h *= 2) {
#pragma unroll
                for (int a = 0; a + h < IA; a += 2 * h) {
                    if (a + h < p.kinter) {
#pragma unroll
                        for (int b = 0; b < NT; b++) {
#pragma unroll
                            for (int l = 0; l < 4; l++) {
                                C[a][b][l] += C[a + h][b][l];
                                C[a + h][b][l] = 0.0f;
                            }
                        }
                    }
                }
            }
            return;
        }
        if (IA > 1 && p.kred == 2) {
            // Last to first: a0 + (a1 + (a2 + ...)).
#pragma unroll
            for (int a = IA - 2; a >= 0; a--) {
                if (a + 1 < p.kinter) {
#pragma unroll
                    for (int b = 0; b < NT; b++) {
#pragma unroll
                        for (int l = 0; l < 4; l++) {
                            C[a][b][l] += C[a + 1][b][l];
                            C[a + 1][b][l] = 0.0f;
                        }
                    }
                }
            }
            return;
        }
#pragma unroll
        for (int a = 1; a < IA; a++) {
            if (a < p.kinter) {
#pragma unroll
                for (int b = 0; b < NT; b++) {
#pragma unroll
                    for (int l = 0; l < 4; l++) {
                        C[0][b][l] += C[a][b][l];
                        C[a][b][l] = 0.0f;
                    }
                }
            }
        }
    };
    // Chunks in flight: slices of 6 token tiles keep S too, so fewer to stay in registers.
    constexpr int D = KS && NT >= 6 ? 2 : WIDE_D;
    uint32_t ar[D][16];
    auto load = [&](uint32_t (&r)[16], int c) {
        if (live && c < nc) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                r[i] = ldg_nc(xw + i * ld2 + c * 32);
            }
        }
    };
    float2 xr[XR];
    auto load_x = [&](int c) {
#pragma unroll
        for (int r = 0; r < XR; ++r) {
            const int q = t + r * 32 * WIDE_WARPS, j = q / 32, l = q % 32;
            xr[r] = j < n && c < nc ? y2[j * y_ld2 + c * 32 + l] : make_float2(0.0f, 0.0f);
        }
    };
#pragma unroll
    for (int d = 0; d < D; ++d) {
        load(ar[d], d);
    }
    load_x(0);
    for (int base = 0; base < nc; base += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int c = base + d;
            if (c >= nc) {
                break;
            }
            // The chunk before is done with Bs and the warp's As.
            __syncthreads();
#pragma unroll
            for (int r = 0; r < XR; ++r) {
                const int q = t + r * 32 * WIDE_WARPS;
                Bs[(q / 32) * PAD + q % 32] = __float22bfloat162_rn(xr[r]);
            }
            if (live) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    ((uint32_t *) As)[i * PAD + lane] = ar[d][i];
                }
            }
            load(ar[d], c + D);
            load_x(c + 1);
            __syncthreads();
            if (live) {
                uint32_t A[4][4];
#pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 8) {
                    ldsm_x4(A[k0 / 8], As + (lane % 16) * PAD + (lane / 16) * 4 + k0);
                }
                // Per output the k steps in order; each token tile is its own chain.
#pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 8) {
                    const int kk = c * 64 + k0 * 2;
                    if constexpr (KS) {
                        // A split-K slice starts here: its chain from 0, the slices before
                        // summed in order.
                        if (kk != 0 && kk % p.kslice == 0) {
                            join();
#pragma unroll
                            for (int b = 0; b < NT; ++b) {
#pragma unroll
                                for (int l = 0; l < 4; ++l) {
                                    S[b][l] = kk == p.kslice ? C[0][b][l] : S[b][l] + C[0][b][l];
                                    C[0][b][l] = 0.0f;
                                }
                            }
                        }
                    }
                    const int acc = IA > 1 ? (kk % p.kslice) / p.kgroup % p.kinter : 0;
#pragma unroll
                    for (int b = 0; b < NT; ++b) {
                        uint32_t B[2];
                        ldsm_x2(B, Bs + (b * 8 + lane % 8) * PAD + ((lane / 8) * 4) % 8 + k0);
#pragma unroll
                        for (int a = 0; a < IA; ++a) {
                            if (a == acc) {
                                if (IA > 1 && p.kmma == 8) {
                                    mma_bf16_k8(C[a][b], A[k0 / 8], B);
                                } else {
                                    mma_bf16(C[a][b], A[k0 / 8], B);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (!live) {
        return;
    }
    if constexpr (KS) {
        join();
        // The last slice's sum after the ones before (none for a single slice).
        if (p.k > p.kslice) {
#pragma unroll
            for (int b = 0; b < NT; ++b) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    C[0][b][l] = S[b][l] + C[0][b][l];
                }
            }
        }
    }
    float * dst = p.dst[wi] + row0;
    const int64_t dst_ld = p.rows[wi];
#pragma unroll
    for (int b = 0; b < NT; ++b) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int i = lane / 4 + (l / 2) * 8;
            const int j = b * 8 + 2 * (lane % 4) + (l % 2);
            if (j < n) {
                dst[j * dst_ld + i] = C[0][b][l];
            }
        }
    }
#else
    __trap();
#endif
}

// Dynamic shared memory of mmf<NTB, NTA, *> with nw warps: the per-warp staging tiles, or the
// warps' sums, whichever is larger.
template <int NTB, int NTA>
constexpr int mmf_smem(int nw) {
    const int stage = nw * (NTA * 16 + NTB * 8) * PAD * 4;
    const int sums = NTB * 8 * (nw * NTA * 16 + 4) * 4;
    return stage > sums ? stage : sums;
}

// NTB: tiles of 8 tokens (n <= 8 * NTB); NTA: tiles of 16 rows per CTA; D: chunks in flight per
// warp. F: F32 weights (TF32 products). blockDim = (32, nw).
// S > 1 (sm_90+, BF16, no chain weights): clusters of S CTAs per tile (grid tiles * S, cluster
// width S), blockDim = (32, ceil(nw / S)) with nw = p.warps; CTA r runs the tile's warps
// r * blockDim.y, ... (past nw: idle until the epilogue).
// WB (with F and S == 1, #836: the MoE router read as BF16): the F32 weights' values stored as
// BF16, all of them exact there; each is widened to its F32 bits (`bf16 << 16`) as it is staged,
// so the staged words, and so every product and sum, are the F32 kernel's. Half the bytes: a lane
// loads one 32-bit word (two values) of one of two rows per load, R / 2 loads per chunk.
template <int NTB, int NTA, int D, bool F = false, int S = 1, bool WB = false>
__global__ void __launch_bounds__(MAX_THREADS) mmf(const legion_dense_mv p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(!WB || (F && S == 1), "WB widens F32 values stored as BF16");
#if __CUDA_ARCH__ < 900
    if constexpr (S > 1) {
        __trap();
    }
#endif
    constexpr int R = NTA * 16;
    // nwl: this CTA's warps; nw: the tile's; w: the warp's place among the tile's.
    const int lane = threadIdx.x, wl = threadIdx.y, nwl = blockDim.y, n = p.n;
    const int rank = S > 1 ? (int) (blockIdx.x % S) : 0;
    const int nw = S > 1 ? p.warps : nwl;
    const int w = rank * nwl + wl;
    int64_t tile = S > 1 ? blockIdx.x / S : blockIdx.x;
    extern __shared__ __align__(16) char smem[];
    if constexpr (NTB == 2 && !F && S == 1) {
        // The chain weights' tiles come first; their CTA's first warp does the work.
        const int64_t ct = chain_tiles(p);
        if (tile < ct) {
            if (w == 0) {
                chain_tile(p, tile, (bf162 *) smem);
            }
            return;
        }
        tile -= ct;
    }
    const int wi = pick(p, tile, R, p.nw - p.nchain);
    const int64_t row0 = tile * R;
    // A weight row's 4-byte words (pairs, or floats for F) and an input column's.
    // WB: words of BF16 pairs; lane l reads word l % 16 of a chunk (16 words) of row l / 16 of
    // each row pair.
    const int64_t ld2 = F && !WB ? p.w_ld[wi] : p.w_ld[wi] / 2;
    const int64_t y_ld2 = F ? p.x_ld : p.x_ld / 2;
    const uint32_t * xw = WB ? (const uint32_t *) p.w[wi] + (row0 + lane / 16) * ld2 + lane % 16
                             : (const uint32_t *) p.w[wi] + row0 * ld2 + lane;
    const float2 * y2 = (const float2 *) p.x;
    const int nc = (int) (F ? p.k / 32 : p.k / 64);  // chunks of 32 words
    // The warp's first chunk; none for the idle warps of a cluster's last CTA.
    const int c0 = w < nw ? w : nc;
    bf162 * As = (bf162 *) smem + wl * (NTA * 16 + NTB * 8) * PAD;
    bf162 * Bs = As + NTA * 16 * PAD;
    float C[NTA][NTB][4];
#pragma unroll
    for (int a = 0; a < NTA; a++) {
#pragma unroll
        for (int b = 0; b < NTB; b++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                C[a][b][l] = 0.0f;
            }
        }
    }
    // Words a lane holds per chunk.
    constexpr int LR = WB ? R / 2 : R;
    uint32_t ar[D][LR];
    auto load = [&](uint32_t (&r)[LR], int c) {
        if (c < nc) {
#pragma unroll
            for (int i = 0; i < LR; ++i) {
                r[i] = WB ? ldg_nc(xw + 2 * i * ld2 + c * 16) : ldg_nc(xw + i * ld2 + c * 32);
            }
        }
    };
#pragma unroll
    for (int d = 0; d < D; ++d) {
        load(ar[d], c0 + d * nw);
    }
    for (int base = c0; base < nc; base += D * nw) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int c = base + d * nw;
            if (c >= nc) {
                break;
            }
            __syncwarp();
            if constexpr (WB) {
                // Each value's F32 bits: the low half of a word is the even column.
#pragma unroll
                for (int i = 0; i < LR; ++i) {
                    const uint32_t v = ar[d][i];
                    *(uint2 *) ((uint32_t *) As + (2 * i + lane / 16) * PAD + 2 * (lane % 16)) =
                        make_uint2(v << 16, v & 0xffff0000u);
                }
            } else {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    ((uint32_t *) As)[i * PAD + lane] = ar[d][i];
                }
            }
            if constexpr (F) {
                // ggml stages the F32 activations as they are, zeros past n.
#pragma unroll
                for (int j = 0; j < NTB * 8; ++j) {
                    ((float *) Bs)[j * PAD + lane] = j < n ? p.x[j * y_ld2 + c * 32 + lane] : 0.0f;
                }
            } else {
#pragma unroll
                for (int j = 0; j < NTB * 8; ++j) {
                    const float2 t = j < n ? y2[j * y_ld2 + c * 32 + lane] : make_float2(0.0f, 0.0f);
                    Bs[j * PAD + lane] = __float22bfloat162_rn(t);
                }
            }
            load(ar[d], c + D * nw);
            __syncwarp();
            uint32_t A[NTA][4][4];
#pragma unroll
            for (int a = 0; a < NTA; ++a) {
#pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 8) {
                    ldsm_x4(A[a][k0 / 8], As + (a * 16 + lane % 16) * PAD + (lane / 16) * 4 + k0);
                }
            }
#pragma unroll
            for (int b = 0; b < NTB; ++b) {
#pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 8) {
                    uint32_t B[2];
                    ldsm_x2(B, Bs + (b * 8 + lane % 8) * PAD + ((lane / 8) * 4) % 8 + k0);
#pragma unroll
                    for (int a = 0; a < NTA; ++a) {
                        if constexpr (F) {
                            mma_tf32(C[a][b], A[a][k0 / 8], B);
                        } else {
                            mma_bf16(C[a][b], A[a][k0 / 8], B);
                        }
                    }
                }
            }
        }
    }
    // The warps' sums, added in warp order as ggml's epilogue adds them.
    // (S > 1: each CTA holds its own warps' sums; the cluster's CTAs add them all, every CTA a
    // share of the outputs, in the tile's warp order.)
    float * buf = (float *) smem;
    const int kiw = nwl * R + 4;
    __syncthreads();
#pragma unroll
    for (int b = 0; b < NTB; ++b) {
#pragma unroll
        for (int a = 0; a < NTA; ++a) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = wl * R + a * 16 + lane / 4 + (l / 2) * 8;
                const int j = b * 8 + 2 * (lane % 4) + (l % 2);
                if (j < n) {
                    buf[j * kiw + i] = C[a][b][l];
                }
            }
        }
    }
    float * dst = p.dst[wi] + row0;
    const int64_t dst_ld = p.rows[wi];
#if __CUDA_ARCH__ >= 900
    if constexpr (S > 1) {
        cluster_sync();
        for (int q = w * 32 + lane; q < n * R; q += S * nwl * 32) {
            const int j = q / R, i = q % R;
            float s = 0.0f;
            for (int pw = 0; pw < nw; ++pw) {
                s += ld_cluster(buf + j * kiw + (pw % nwl) * R + i, pw / nwl);
            }
            dst[j * dst_ld + i] = s;
        }
        // No CTA leaves (its shared memory) while another may still read it.
        cluster_sync();
        return;
    }
#endif
    __syncthreads();
    for (int q = w * 32 + lane; q < n * R; q += nw * 32) {
        const int j = q / R, i = q % R;
        float s = 0.0f;
        for (int pw = 0; pw < nw; ++pw) {
            s += buf[j * kiw + pw * R + i];
        }
        dst[j * dst_ld + i] = s;
    }
#else
    __trap();
#endif
}
