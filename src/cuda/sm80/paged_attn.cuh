// paged_attn/common.cuh, sm_80 and newer: the one-pass FlashAttention-2 prefill on mma.sync m16n8k16
// with a cp.async ring (namespace fa, paged_prefill_mma, fa::available and its launcher).
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after the
// shared helpers it uses (BLK, walk, block_cell, merge_row, the KV load helpers). The body is
// built only where __CUDA_ARCH__ >= 800 (LEGION_PAGED_MMA) and is a __trap() elsewhere, so the
// file compiles for every target architecture; fa::available() picks it per device at run time.

// ---------------------------------------------------------------------------------------------
// Prefill on sm_80+: one pass with an online softmax (FlashAttention-2) on mma.sync m16n8k16,
// whose fragment layouts PTX documents, so scores, softmax and output stay in registers.
//
// The queries are the MMA's N dimension, as in llama.cpp's mma flash-attention kernel
// (ggml/src/ggml-cuda/fattn-mma-f16.cuh, MIT): a warp owns 8 query columns (column =
// token_in_tile * G + head_in_group, as the WMMA kernel's rows) and computes S^T = K Q^T for
// them over a whole 64-key tile, K as the A operand straight from shared memory, Q as B
// fragments held in registers. The softmax runs on those registers; movmatrix turns the
// weights into B fragments, and O^T += V^T P accumulates the warp's full-width output in F32
// registers (D / 4 per thread). No partial scores cross warps.
//
// A CTA: 8 warps, 64 query columns of one kv head, the last query tiles (the most keys) first.
// K and V stream through shared memory in chunks of 64 keys x up to 256 dims (XOR-swizzled
// 16-byte units for conflict-free ldmatrix) over a ring of cp.async stages, one barrier per
// chunk; the key tile is a KV block, so a chunk is one table lookup. Keys past the CTA's last
// query are zero-filled, never read. Truncating tensor cores (see paged_attn/common.cuh's header) see
// short chains only: each chunk's scores and each tile's P V start from zero and are added to the
// F32 totals.
//
// Split over the key blocks (SPLIT, --paged-verify-split, #697): small multi-token groups (a
// speculative verify step, 2-16 tokens a sequence) have one or two query tiles per kv head, so
// the plain grid is a few CTAs that each walk the whole context. With SPLIT the grid's z also
// counts splits, as flash-decoding's does: a CTA takes (query tile, kv head, sequence, split), a
// split is a run of whole key blocks of the tile's range, and min(max_split, blocks) splits per
// tile. A CTA writes each of its columns' running (max in log2 units, sum, unnormalized output)
// to p.scratch, laid out [D + 2, max_split, n_head, n_tok * n_seqs], and paged_split_combine
// merges the splits in F32. A tile with one split writes dst directly, as without SPLIT.
//
// Wide prompt tiles (NWT = 16, 256-wide heads, prompt steps only): 16 warps and 128 query columns
// a CTA, so every K/V byte read from L2 feeds twice the columns (the 8-warp kernel ran at about
// 0.9 TB/s of L2 traffic on a GB10 at 32K, its bound). The register file then holds O^T but not Q
// (128 registers a thread), so each thread keeps its Q fragments in shared memory, in fragment
// order (a conflict-free 8-byte read per k-step), and K/V come in 16 KB chunks (128 dims) through
// two stages. A column's scores keep one mma chain over both chunks of its dims, and a warp skips
// only key tiles none of its columns sees (a masked tile adds exactly nothing to a column), so
// every column gets the 8-warp kernel's bits.
// ---------------------------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define LEGION_PAGED_MMA 1
#endif

namespace fa {

// Per head size: dims per K/V chunk, ring stages and CTAs per SM. 256- and 512-wide heads take
// the register file for one CTA (O^T alone is D / 4 registers per thread) and stream 32 KB
// chunks through three stages (96 KB): on GB10 that ran 512-wide heads 17% faster than 16 KB
// chunks through four. 128-wide heads fit two CTAs per SM. NWT = 16: the wide prompt tiles above
// (Q in shared memory after the ring, 64 KB for 256-wide heads, so 96 KB in all).
template <int D, int NWT = 8> struct Shape {
    static constexpr int NW      = NWT;                                    // warps per CTA
    static constexpr int COLS    = 8 * NW;                                 // query columns per CTA
    static constexpr bool QS     = NW > 8;                                 // Q in shared memory
    static constexpr int CH      = D >= 256 && !QS ? 256 : 128;            // dims per chunk
    static constexpr int NS      = QS ? 2 : 3;                             // ring stages
    static constexpr int MINB    = D >= 256 || QS ? 1 : 2;                 // CTAs per SM
    static constexpr int UNITS   = CH / 8;                                 // 16-byte units per row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);          // bytes per chunk
    static constexpr int QBYTES  = QS ? D / 16 * NW * 32 * 8 : 0;          // Q fragments
    static constexpr size_t SMEM = (size_t) NS * CHUNK + QBYTES;
};

// Byte offset of 16-byte unit `u` of chunk row `r` (rows of UNITS units), XOR-swizzled by the
// row so the 8 rows an ldmatrix 8x8 reads fall in distinct bank groups.
template <int UNITS>
__device__ __forceinline__ uint32_t swz(int r, int u) {
    return (uint32_t) (r * (UNITS * 16) + ((u ^ (r & 7)) << 4));
}

#ifdef LEGION_PAGED_MMA
__device__ __forceinline__ void cp_async16(uint32_t dst, const void * src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_t(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// c += a b (m16n8k16, F16 in, F32 accumulate).
__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// An 8x8 F16 matrix in accumulator layout (row lane / 4, columns 2 (lane % 4) + {0, 1}),
// transposed in place: the B-operand layout of its transpose.
__device__ __forceinline__ uint32_t transpose(uint32_t x) {
    uint32_t y;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}
#endif

// SPLIT: the splits of a tile whose key blocks are [*b0, *b1], min(max_split, blocks) of them
// (the count returned), and split `split`'s run of them, which replaces [*b0, *b1].
__device__ __forceinline__ int split_blocks(int max_split, int split, int * b0, int * b1) {
    const int first = *b0, nblk = *b1 - first + 1;
    const int ns = n_splits(max_split, nblk);
    *b0 = first + split * nblk / ns;
    *b1 = first + (split + 1) * nblk / ns - 1;
    return ns;
}

template <int D, int G, bool C = false, bool SPLIT = false, int NWT = 8>
__global__ void __launch_bounds__(Shape<D, NWT>::NW * 32, Shape<D, NWT>::MINB) paged_prefill_mma(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = Shape<D, NWT>;
    constexpr bool QS = SH::QS;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK;
    constexpr int NW = SH::NW, COLS = SH::COLS;
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int KS = D / 16;    // k-steps of K Q^T
    constexpr int QT = COLS / G;  // tokens per CTA
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last query tiles see the most keys: they go first.
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y;
    const int seq = SPLIT ? blockIdx.z / p.max_split : blockIdx.z;
    const int split = SPLIT ? blockIdx.z % p.max_split : 0;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Column c's token. Columns past n_tok, and past QT * G when G does not divide 64 (12:
    // 5 tokens, 60 columns), repeat the tile's last token and are not written.
    auto tok_of = [&](int c) { return min(t0 + min(c / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(COLS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    // SPLIT: this CTA's run of the tile's blocks (split_blocks).
    int ns = 1, w_b0 = b_first, w_b1 = b_last;
    if constexpr (SPLIT) {
        ns = split_blocks(p.max_split, split, &w_b0, &w_b1);
        if (split >= ns) {
            return;
        }
    }
    const Walk wk = walk(p, seq, w_b0, w_b1);
    if (piece_idle(p, wk)) {
        return;
    }

    // This warp's columns (for skipping and unmasked tiles) and this thread's two columns of
    // the S^T / O^T fragments, 2 tq + {0, 1}.
    const int w_pos_lo = pos_of(warp * 8), w_pos_hi = pos_of(warp * 8 + 7);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        c_pos[j]   = pos_of(warp * 8 + 2 * tq + j);
        c_first[j] = first_key(c_pos[j], p.window);
    }

    // Q as B fragments (k = dims, n = columns): column gq, dims 2 tq + {0, 1} and + 8. QS: in
    // shared memory after the ring, k-step ks of this thread at q_at(ks) (only it reads them).
    uint32_t qf[QS ? 1 : KS][2];
    auto q_at = [&](int ks) { return (uint32_t) (NS * CHUNK + ((ks * NW + warp) * 32 + lane) * 8); };
    {
        const int c = warp * 8 + gq;
        const int t = tok_of(c);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            if constexpr (QS) {
                *(uint2 *) (smem + q_at(ks)) = make_uint2(pack(a.x, a.y), pack(b.x, b.y));
            } else {
                qf[ks][0] = pack(a.x, a.y);
                qf[ks][1] = pack(b.x, b.y);
            }
        }
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    auto issue = [&](int j) {
        if (j < n_chunks) {
            const int i = wk.i0 + j / (2 * NC), part = j % (2 * NC);
            const int b = wk.block(i);
            const int ty = part < NC ? p.k_type : p.v_type;
            const int64_t cell_bytes = part < NC ? p.k_cell : p.v_cell;
            const char * base = part < NC ? wk.cells(p.k, p.k_stage, i, cell_bytes) : wk.cells(p.v, p.v_stage, i, cell_bytes);
            const int e0 = head_at<D, C>(p, kvh) + (part % NC) * CH;
            const uint32_t dst = sbase + (uint32_t) (j % NS) * CHUNK;
            if (ty == LEGION_KV_F16) {
                const half * src = (const half *) base + e0;
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    cp_async16(dst + swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + u * 8, b * BLK + r < k_end);
                }
            } else {
                // Quantized: dequantized to F16 on the way in (synchronous loads).
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    uint4 x = make_uint4(0, 0, 0, 0);
                    if (b * BLK + r < k_end) {
                        float f[8];
                        load_row<8>(base + r * cell_bytes, ty, e0 + u * 8, f);
                        x = make_uint4(pack(f[0], f[1]), pack(f[2], f[3]), pack(f[4], f[5]), pack(f[6], f[7]));
                    }
                    *(uint4 *) (smem + (dst - sbase) + swz<UNITS>(r, u)) = x;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // O^T fragments: dims 16 i + gq (+ 8 for [2], [3]), columns 2 tq + {0, 1}.
    float o[D / 16][4];
#pragma unroll
    for (int i = 0; i < D / 16; i++) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    }
    // Running max (log2 units) and this thread's share of the sum, per column.
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float scale = p.scale * LOG2E;

    // ldmatrix addresses. Every chunk row a lane addresses has r % 8 == lane % 8, so unit
    // 8 q + v (v < 8) of it sits at 16 (8 q + (v ^ lane % 8)): a lane offset per row group, an
    // offset per v (4 values in use: K units 2 r + lane / 16, V units 2 r + lane / 8 % 2),
    // and constants.
    constexpr int ROWB = UNITS * 16;
    const uint32_t k_lane = (lane / 8 % 2 * 8 + lane % 8) * ROWB, v_lane = (lane / 16 * 8 + lane % 8) * ROWB;
    uint32_t k_x[4], v_x[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        k_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
        v_x[r] = ((2 * r + lane / 8 % 2) ^ (lane % 8)) << 4;
    }

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        // Whether any column of the warp sees a key of the tile, and whether all see all.
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[4][4];      // S^T: keys 16 i + gq (+ 8), columns 2 tq + {0, 1}
        uint32_t pf[4][2];  // P as B fragments per 16-key step
        float alpha[2];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
            __syncthreads();
            issue(j + NS - 1);
            if (!active) {
                continue;
            }
            const uint32_t st = sbase + (uint32_t) (j % NS) * CHUNK;
            if (part < NC) {
                // S^T += K Q^T over this chunk's dims.
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    // QS: one chain over all the dims (the 8-warp kernel's single 256-dim chunk).
                    float acc[4];
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        acc[e] = QS && part > 0 ? s[i][e] : 0.0f;
                    }
#pragma unroll
                    for (int ks = 0; ks < CH / 16; ks++) {
                        uint32_t a[4];
                        ldsm(a, st + k_lane + k_x[ks % 4] + i * 16 * ROWB + ks / 4 * 128);
                        if constexpr (QS) {
                            const uint2 q = *(const uint2 *) (smem + q_at(part * (CH / 16) + ks));
                            const uint32_t b[2] = {q.x, q.y};
                            mma(acc, a, b);
                        } else {
                            mma(acc, a, qf[part * (CH / 16) + ks]);
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[i][e] = QS || part == 0 ? acc[e] : s[i][e] + acc[e];
                    }
                }
                if (part == NC - 1) {
                    // Online softmax per column; element e is key 16 i + gq + 8 (e / 2),
                    // column 2 tq + e % 2.
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int i = 0; i < 4; i++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float x = s[i][e] * scale;
                            if (!whole) {
                                const int key = k0 + i * 16 + gq + 8 * (e / 2);
                                x = key <= c_pos[e % 2] && key >= c_first[e % 2] ? x : -INFINITY;
                            }
                            s[i][e] = x;
                            mx[e % 2] = fmaxf(mx[e % 2], x);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < 2; c++) {
#pragma unroll
                        for (int off = 4; off < 32; off <<= 1) {
                            mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], off));
                        }
                        const float m_new = fmaxf(m[c], mx[c]);
                        // No visible key yet: exp against 0 gives 0 for the masked scores.
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[c] = exp2f(m[c] - m_use);
                        m[c]  = m_new;
                        mx[c] = m_use;
                    }
                    float sum[2] = {0.0f, 0.0f};
#pragma unroll
                    for (int i = 0; i < 4; i++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            s[i][e] = exp2f(s[i][e] - mx[e % 2]);
                            sum[e % 2] += s[i][e];
                        }
                        pf[i][0] = transpose(pack(s[i][0], s[i][1]));
                        pf[i][1] = transpose(pack(s[i][2], s[i][3]));
                    }
                    l[0] = l[0] * alpha[0] + sum[0];
                    l[1] = l[1] * alpha[1] + sum[1];
                }
            } else {
                // O^T = O^T alpha + V^T P for this chunk's output dims.
                const int c = part - NC;
#pragma unroll
                for (int i = 0; i < CH / 16; i++) {
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_lane + v_x[i % 4] + ks * 16 * ROWB + i / 4 * 128);
                        mma(acc, a, pf[ks]);
                    }
                    float * oi = o[c * (CH / 16) + i];
                    oi[0] = oi[0] * alpha[0] + acc[0];
                    oi[1] = oi[1] * alpha[1] + acc[1];
                    oi[2] = oi[2] * alpha[0] + acc[2];
                    oi[3] = oi[3] * alpha[1] + acc[3];
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int c = 0; c < 2; c++) {
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            l[c] += __shfl_xor_sync(0xffffffff, l[c], off);
        }
        const int col = warp * 8 + 2 * tq + c;
        const int t = t0 + col / G;
        const bool valid = col < QT * G && t < p.n_tok;
        const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + col % G;
        if (SPLIT && ns > 1) {
            // This split's (max, sum, unnormalized output) of the column, for the combine.
            if (valid) {
                float * part = p.scratch + (row * p.max_split + split) * (D + 2);
                if (gq == 0) {
                    part[0] = m[c];
                    part[1] = l[c];
                }
                float * out = part + 2 + gq;
#pragma unroll
                for (int i = 0; i < D / 16; i++) {
                    out[i * 16]     = o[i][c];
                    out[i * 16 + 8] = o[i][2 + c];
                }
            }
        } else if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. The column's 8 lanes read
            // its running (max, sum) before one of them writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[c], l[c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            __syncwarp();
            if (valid) {
                if (gq == 0) {
                    store_ml(p, row, f);
                }
                const float inv = l[c] > 0.0f ? 1.0f / l[c] : 0.0f;
                float * out = p.dst + row * D + gq;
#pragma unroll
                for (int i = 0; i < D / 16; i++) {
                    const float x0 = o[i][c] * inv, x1 = o[i][2 + c] * inv;
                    out[i * 16]     = p.piece > 0 ? f.x * out[i * 16] + f.y * x0 : x0;
                    out[i * 16 + 8] = p.piece > 0 ? f.x * out[i * 16 + 8] + f.y * x1 : x1;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / l[c];
            float * out = p.dst + row * D + gq;
#pragma unroll
            for (int i = 0; i < D / 16; i++) {
                out[i * 16]     = o[i][c] * inv;
                out[i * 16 + 8] = o[i][2 + c] * inv;
            }
        }
    }
#else
    __trap();
#endif
}

// Whether the current device runs paged_prefill_mma (mma.sync m16n8k16, cp.async: sm_80+).
bool available() {
    return legion_device::compute_capability() >= 80;
}

// Prompt steps of at least this many tokens take the wide tiles (256-wide heads, plain cache
// walks): verify and decode-sized groups keep the 8-warp kernel. Same bits either way.
constexpr int WIDE_MIN_TOK = 256;

template <int D, int G, bool C, int NWT>
int launch_nw(const legion_paged_attn & p, cudaStream_t st) {
    using SH = Shape<D, NWT>;
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_mma<D, G, C, false, NWT>, (int) SH::SMEM)) {
        return 2;
    }
    constexpr int QT = SH::COLS / G;
    paged_prefill_mma<D, G, C, false, NWT><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), SH::NW * 32, SH::SMEM, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    if constexpr (D == 256 && !C) {
        if (p.n_tok >= WIDE_MIN_TOK && !p.piece_blk && !p.ml) {
            return launch_nw<D, G, C, 16>(p, st);
        }
    }
    return launch_nw<D, G, C, 8>(p, st);
}

// Merges the splits of paged_prefill_mma<.., SPLIT> for row (token, head) blockIdx (y, x): one
// thread per output dim. The row's tile and its key blocks as the attention kernel takes them;
// a tile with one split was written by that kernel. Max in log2 units, as the kernel keeps it.
template <int D, int G>
__global__ void __launch_bounds__(D) paged_split_combine(const legion_paged_attn p) {
    constexpr int QT = Shape<D>::COLS / G;
    const int h = blockIdx.x, rt = blockIdx.y;  // rt = seq * n_tok + t
    const int seq = rt / p.n_tok, t = rt % p.n_tok;
    const int t0 = t / QT * QT, n_past = p.n_past[seq];
    int b0 = first_key(n_past + t0, p.window) / BLK, b1 = (n_past + min(t0 + QT, p.n_tok) - 1) / BLK;
    const int ns = split_blocks(p.max_split, 0, &b0, &b1);
    if (ns == 1) {
        return;
    }
    const int64_t row = (int64_t) rt * p.n_head + h;
    const float * part = p.scratch + row * p.max_split * (D + 2);
    float mx = -INFINITY;
    for (int s = 0; s < ns; s++) {
        mx = fmaxf(mx, part[s * (D + 2)]);
    }
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < ns; s++) {
        const float w = exp2f(part[s * (D + 2)] - mx);
        l += w * part[s * (D + 2) + 1];
        o += w * part[s * (D + 2) + 2 + threadIdx.x];
    }
    p.dst[row * D + threadIdx.x] = o / l;
}

// A small multi-token group split over the key blocks (SPLIT above): p.max_split > 1 splits per
// tile at most, p.scratch [D + 2, max_split, n_head, n_tok * n_seqs]. Not for pieces of a tiered
// pool (p.piece_blk, p.ml).
template <int D, int G, bool C = false>
int launch_split(const legion_paged_attn & p, cudaStream_t st) {
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_mma<D, G, C, true>, (int) Shape<D>::SMEM)) {
        return 2;
    }
    constexpr int QT = Shape<D>::COLS / G;
    paged_prefill_mma<D, G, C, true><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs * p.max_split), Shape<D>::NW * 32,
                                       Shape<D>::SMEM, st>>>(p);
    paged_split_combine<D, G><<<dim3(p.n_head, p.n_tok * p.n_seqs), D, 0, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace fa
