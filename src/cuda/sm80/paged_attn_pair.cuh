// paged_attn/common.cuh, sm_80 and newer: paged_prefill_mma's prefill for heads of 512 with each
// query column's output split over a warp pair (namespace fa_pair, paged_prefill_pair, #1181). The
// same bits.
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after
// sm80/paged_attn.cuh (namespace fa: swz, cp_async16, ldsm, ldsm_t, mma, transpose, pack).
//
// paged_prefill_mma at heads of 512 gives a warp 8 query columns and their whole output: O^T
// alone is 128 registers a thread and Q's B fragments 64 more, so the kernel sits at 255 and
// ptxas runs its MMA chains one after another, each ldmatrix right before its MMA. Here two warps
// share 8 columns. Each holds Q (64 registers) and half of the output (64), interleaved by V
// chunk: warp h owns output dims c * 256 + h * 128 .. + 127 of chunks c = 0, 1, so both work on
// every V chunk. On a K chunk warp h takes the 16-key blocks 2 h and 2 h + 1 of the 64-key tile;
// the pair swaps its column maxima, the sum and its halves of P through 768 bytes of shared memory
// behind a 64-thread barrier. A CTA: 8 warps, 4 pairs, 32 query columns of one kv head (half of
// paged_prefill_mma's), K and V through the same ring of 64 keys x 256 dims, three stages.
//
// Every value goes through the same operations in the same order as there:
// - A score is one chain of 16 MMAs from zero per K chunk, the two added as s0 + s1; scale and
//   mask as there. A column's maximum is the maximum of the same values (exact in any order).
// - paged_prefill_mma's thread sums its column's weights of key blocks 0 to 3 in order. Warp 0
//   adds those of blocks 0 and 1 from zero, warp 1 goes on from warp 0's partial sum with blocks
//   2 and 3, then (l, alpha) as there; the epilogue's lane tree is the same.
// - Each 16 output dims' P V is one chain of 4 MMAs from zero over the tile's 16-key steps with
//   the same P fragments, added as o alpha + acc.
// Tiles a pair's columns do not see are skipped as there, and a smaller CTA tile covers the
// same keys for each column, so the bits do not change.
//
// Not covered (paged_prefill_mma instead): other head sizes or GQA ratios, the compact cache,
// key-block splits.

namespace fa_pair {

template <int D> struct Shape {
    static_assert(D == 512, "the warp-pair kernel is for heads of 512");
    static constexpr int NW      = 8;                              // warps per CTA
    static constexpr int PAIRS   = NW / 2;                         // warp pairs
    static constexpr int COLS    = 8 * PAIRS;                      // query columns per CTA
    static constexpr int CH      = 256;                            // dims per chunk
    static constexpr int NS      = 3;                              // ring stages
    static constexpr int UNITS   = CH / 8;                         // 16-byte units per row
    static constexpr int CHUNK   = BLK * CH * (int) sizeof(half);  // bytes per chunk
    static constexpr int RING    = NS * CHUNK;
    static constexpr int XB      = 768;  // a pair's exchange: 512 B of P (or maxima), 256 B of sums
    static constexpr size_t SMEM = (size_t) RING + (size_t) PAIRS * XB;
};

#ifdef LEGION_PAGED_MMA
// The 64 threads of warp pair `id - 1`.
__device__ __forceinline__ void pair_sync(int id) {
    asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory");
}
#endif

template <int D, int G>
__global__ void __launch_bounds__(Shape<D>::NW * 32, 1) paged_prefill_pair(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    using SH = Shape<D>;
    using fa::cp_async16;
    using fa::ldsm;
    using fa::ldsm_t;
    using fa::mma;
    using fa::pack;
    using fa::transpose;
    constexpr int NS = SH::NS, CH = SH::CH, UNITS = SH::UNITS, CHUNK = SH::CHUNK;
    constexpr int NW = SH::NW, COLS = SH::COLS;
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int KS = D / 16;    // k-steps of K Q^T
    constexpr int QT = COLS / G;  // tokens per CTA
    constexpr int HB = CH / 32;   // a warp's 16-dim output blocks per V chunk
    constexpr int OB = NC * HB;   // a warp's 16-dim output blocks
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    // The last query tiles see the most keys: they go first.
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int pr = warp / 2, h = warp % 2;   // pair, half
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad
    const int n_past = p.n_past[seq];

    // Column c's token. Columns past n_tok repeat the tile's last token and are not written.
    auto tok_of = [&](int c) { return min(t0 + min(c / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(COLS - 1) + 1;  // keys at or past k_end are never visible
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    // This pair's columns (for skipping and unmasked tiles) and this thread's two columns of the
    // S^T / O^T fragments, 2 tq + {0, 1}.
    const int c_base = pr * 8;
    const int w_pos_lo = pos_of(c_base), w_pos_hi = pos_of(c_base + 7);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2], c_first[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        c_pos[j]   = pos_of(c_base + 2 * tq + j);
        c_first[j] = first_key(c_pos[j], p.window);
    }

    // Q as B fragments (k = dims, n = columns): column gq, dims 2 tq + {0, 1} and + 8.
    uint32_t qf[KS][2];
    {
        const int c = c_base + gq;
        const int t = tok_of(c);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            qf[ks][0] = pack(a.x, a.y);
            qf[ks][1] = pack(b.x, b.y);
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
            const int e0 = head_at<D, false>(p, kvh) + (part % NC) * CH;
            const uint32_t dst = sbase + (uint32_t) (j % NS) * CHUNK;
            if (ty == LEGION_KV_F16) {
                const half * src = (const half *) base + e0;
#pragma unroll
                for (int i = threadIdx.x; i < BLK * UNITS; i += NW * 32) {
                    const int r = i / UNITS, u = i % UNITS;
                    cp_async16(dst + fa::swz<UNITS>(r, u), src + (int64_t) r * p.kv_row + u * 8, b * BLK + r < k_end);
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
                    *(uint4 *) (smem + (dst - sbase) + fa::swz<UNITS>(r, u)) = x;
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // This warp's half of O^T: block ob is dims 16 (ob / HB * 2 HB + h HB + ob % HB) + gq (+ 8 for
    // [2], [3]), columns 2 tq + {0, 1}.
    float o[OB][4];
#pragma unroll
    for (int i = 0; i < OB; i++) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    }
    // Running max (log2 units) per column; this thread's share of the sum (kept by warp 1).
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const float scale = p.scale * LOG2E;

    // The pair's exchange. xp: a lane's uint4 of P fragments (warp 1's, then warp 0's), or warp
    // 0's column maxima; xs: warp 0's partial sums, a lane's float2, or warp 1's column maxima, or
    // at the end warp 1's sums.
    unsigned char * xb = smem + SH::RING + pr * SH::XB;
    uint4 * xp = (uint4 *) xb;
    float * xs = (float *) (xb + 512);
    float * mx_mine = h == 0 ? (float *) xp : xs;
    const float * mx_other = h == 0 ? xs : (const float *) xp;
    const int bar = 1 + pr;

    // ldmatrix addresses, as in paged_prefill_mma.
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
        // Whether any column of the pair sees a key of the tile, and whether all see all.
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[2][4];      // S^T: keys 16 (2 h + ii) + gq (+ 8), columns 2 tq + {0, 1}
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
                // S^T += K Q^T over this chunk's dims, key blocks 2 h and 2 h + 1.
#pragma unroll
                for (int ii = 0; ii < 2; ii++) {
                    const int i = 2 * h + ii;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < CH / 16; ks++) {
                        uint32_t a[4];
                        ldsm(a, st + k_lane + k_x[ks % 4] + i * 16 * ROWB + ks / 4 * 128);
                        mma(acc, a, qf[part * (CH / 16) + ks]);
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[ii][e] = part == 0 ? acc[e] : s[ii][e] + acc[e];
                    }
                }
                if (part == NC - 1) {
                    // Online softmax per column; element e is key 16 (2 h + ii) + gq + 8 (e / 2),
                    // column 2 tq + e % 2.
                    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float x = s[ii][e] * scale;
                            if (!whole) {
                                const int key = k0 + (2 * h + ii) * 16 + gq + 8 * (e / 2);
                                x = key <= c_pos[e % 2] && key >= c_first[e % 2] ? x : -INFINITY;
                            }
                            s[ii][e] = x;
                            mx[e % 2] = fmaxf(mx[e % 2], x);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < 2; c++) {
#pragma unroll
                        for (int off = 4; off < 32; off <<= 1) {
                            mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], off));
                        }
                    }
                    // The other warp's half of the tile.
                    if (gq == 0) {
                        mx_mine[2 * tq]     = mx[0];
                        mx_mine[2 * tq + 1] = mx[1];
                    }
                    pair_sync(bar);
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        mx[c] = fmaxf(mx[c], mx_other[2 * tq + c]);
                        const float m_new = fmaxf(m[c], mx[c]);
                        // No visible key yet: exp against 0 gives 0 for the masked scores.
                        const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                        alpha[c] = exp2f(m[c] - m_use);
                        m[c]  = m_new;
                        mx[c] = m_use;
                    }
                    uint32_t pm[2][2];  // this warp's P fragments, key blocks 2 h + ii
#pragma unroll
                    for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            s[ii][e] = exp2f(s[ii][e] - mx[e % 2]);
                        }
                        pm[ii][0] = transpose(pack(s[ii][0], s[ii][1]));
                        pm[ii][1] = transpose(pack(s[ii][2], s[ii][3]));
                    }
                    // Warp 0: its sum of key blocks 0 and 1 to warp 1, which goes on with 2 and 3;
                    // warp 1: its P to warp 0, then warp 0's P to warp 1.
                    float sum[2] = {0.0f, 0.0f};
                    if (h == 0) {
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                sum[e % 2] += s[ii][e];
                            }
                        }
                        ((float2 *) xs)[lane] = make_float2(sum[0], sum[1]);
                    } else {
                        xp[lane] = make_uint4(pm[0][0], pm[0][1], pm[1][0], pm[1][1]);
                    }
                    pair_sync(bar);
                    if (h == 0) {
                        const uint4 v = xp[lane];
                        xp[lane]      = make_uint4(pm[0][0], pm[0][1], pm[1][0], pm[1][1]);
                        pf[0][0] = pm[0][0];
                        pf[0][1] = pm[0][1];
                        pf[1][0] = pm[1][0];
                        pf[1][1] = pm[1][1];
                        pf[2][0] = v.x;
                        pf[2][1] = v.y;
                        pf[3][0] = v.z;
                        pf[3][1] = v.w;
                    } else {
                        const float2 q = ((const float2 *) xs)[lane];
                        sum[0] = q.x;
                        sum[1] = q.y;
#pragma unroll
                        for (int ii = 0; ii < 2; ii++) {
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                sum[e % 2] += s[ii][e];
                            }
                        }
                        l[0] = l[0] * alpha[0] + sum[0];
                        l[1] = l[1] * alpha[1] + sum[1];
                        pf[2][0] = pm[0][0];
                        pf[2][1] = pm[0][1];
                        pf[3][0] = pm[1][0];
                        pf[3][1] = pm[1][1];
                    }
                    pair_sync(bar);
                    if (h == 1) {
                        const uint4 v = xp[lane];
                        pf[0][0] = v.x;
                        pf[0][1] = v.y;
                        pf[1][0] = v.z;
                        pf[1][1] = v.w;
                    }
                }
            } else {
                // O^T = O^T alpha + V^T P for this warp's 128 of this chunk's output dims.
                const int c = part - NC;
#pragma unroll
                for (int ib = 0; ib < HB; ib++) {
                    const int i = h * HB + ib;
                    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_lane + v_x[i % 4] + ks * 16 * ROWB + i / 4 * 128);
                        mma(acc, a, pf[ks]);
                    }
                    float * oi = o[c * HB + ib];
                    oi[0] = oi[0] * alpha[0] + acc[0];
                    oi[1] = oi[1] * alpha[1] + acc[1];
                    oi[2] = oi[2] * alpha[0] + acc[2];
                    oi[3] = oi[3] * alpha[1] + acc[3];
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // Warp 1's sums (the lane tree as in paged_prefill_mma) to warp 0.
    if (h == 1) {
#pragma unroll
        for (int c = 0; c < 2; c++) {
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                l[c] += __shfl_xor_sync(0xffffffff, l[c], off);
            }
        }
        if (gq == 0) {
            xs[2 * tq]     = l[0];
            xs[2 * tq + 1] = l[1];
        }
    }
    pair_sync(bar);
    if (h == 0) {
        l[0] = xs[2 * tq];
        l[1] = xs[2 * tq + 1];
    }

#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int col = c_base + 2 * tq + c;
        const int t = t0 + col / G;
        const bool valid = col < QT * G && t < p.n_tok;
        const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + col % G;
        float * out = p.dst + row * D + gq;
        if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. The column's 16 lanes over
            // both warps read its running (max, sum) before one of them writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[c], l[c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            pair_sync(bar);
            if (valid) {
                if (h == 0 && gq == 0) {
                    store_ml(p, row, f);
                }
                const float inv = l[c] > 0.0f ? 1.0f / l[c] : 0.0f;
#pragma unroll
                for (int ob = 0; ob < OB; ob++) {
                    const int i = ob / HB * 2 * HB + h * HB + ob % HB;
                    const float x0 = o[ob][c] * inv, x1 = o[ob][2 + c] * inv;
                    out[i * 16]     = p.piece > 0 ? f.x * out[i * 16] + f.y * x0 : x0;
                    out[i * 16 + 8] = p.piece > 0 ? f.x * out[i * 16 + 8] + f.y * x1 : x1;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / l[c];
#pragma unroll
            for (int ob = 0; ob < OB; ob++) {
                const int i = ob / HB * 2 * HB + h * HB + ob % HB;
                out[i * 16]     = o[ob][c] * inv;
                out[i * 16 + 8] = o[ob][2 + c] * inv;
            }
        }
    }
#else
    __trap();
#endif
}

// Where opted in (legion_paged_attn_set_d512_pair) for heads of 512, 8 query heads a kv head, a
// non-compact cache and no key-block split. -1: no shared memory for it on this device (paged_prefill_mma instead).
template <int D, int G>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_pair<D, G>, (int) Shape<D>::SMEM)) {
        return -1;
    }
    constexpr int QT = Shape<D>::COLS / G;
    paged_prefill_pair<D, G><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), Shape<D>::NW * 32, Shape<D>::SMEM, st>>>(p);
    t_d512_pair_launches++;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace fa_pair
