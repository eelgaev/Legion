// sm80/paged_attn.cuh, consumer Blackwell (the GB10): paged_prefill_mma for 256-wide heads with
// warp pairs (#1133), the bits of paged_prefill_mma<256, G, C>.
//
// Not a standalone header: sm80/paged_attn.cuh includes it inside namespace fa, after
// paged_prefill_mma and its helpers (cp_async16, ldsm, ldsm_t, mma, transpose, pack, swz).
//
// paged_prefill_mma reads one 512-byte ldmatrix fragment of K or V per mma: on the GB10 the
// shared-memory reads, not the tensor cores, bound it (~58 of 124 TFLOPS at 256-wide heads). Here
// two warps share 16 query columns (two n8 tiles, the columns of paged_prefill_mma's warps
// 2 p and 2 p + 1), and every K and V fragment feeds both tiles' MMAs:
// - S^T = K Q^T: warp h of the pair takes the keys 32 h .. 32 h + 31 of the 64-key tile, over
//   all D dims (Q of its 16 columns in registers), in the same k order: K comes in chunks of
//   128 dims and the scores carry across them, one mma chain as paged_prefill_mma's 256-dim
//   chunk.
// - Softmax: each warp's column max goes through shared memory to its partner (a max is exact,
//   in any order). Warp 0's per-thread sum of its keys goes to warp 1, which continues the chain
//   with its own keys in paged_prefill_mma's order (keys gq, gq + 8, 16 + gq, ... of the thread),
//   so the running sum is that kernel's thread's. Both warps' P fragments go through shared
//   memory, so each warp holds P of all 64 keys of its 16 columns.
// - O^T += V^T P: warp h takes 64 of each 128-dim V chunk's dims (blocks 4 h .. 4 h + 3), over
//   the 64 keys in the same order as paged_prefill_mma.
// Every score, weight, sum and output takes the same operations in the same order: the same bits.
// Not for split tiles (SPLIT) or the pieces of a tiered pool (p.ml): those take
// paged_prefill_mma.

namespace pair {

constexpr int NW    = 8;                              // warps per CTA (4 pairs)
constexpr int COLS  = 64;                             // query columns per CTA, as paged_prefill_mma
constexpr int CH    = 128;                            // dims per K/V chunk
constexpr int NS    = 5;                              // ring stages
constexpr int UNITS = CH / 8;                         // 16-byte units per chunk row
constexpr int CHUNK = BLK * CH * (int) sizeof(half);  // bytes per chunk (16 KiB)
// Per pair, in 4-byte words: the column maxes [half][nt * 2 + c][tq], warp 0's sums
// [nt][c][lane], the P fragments [half][key block][nt][reg][lane] and the final sums [nt * 2 + c][tq].
constexpr int X_MAX = 0, X_SUM = 32, X_PF = 160, X_L = 672, X_WORDS = 688;
constexpr size_t SMEM = (size_t) NS * CHUNK + (size_t) (NW / 2) * X_WORDS * 4;

template <int D, int G, bool C>
__global__ void __launch_bounds__(NW * 32, 1) paged_prefill_pair(const legion_paged_attn p) {
#ifdef LEGION_PAGED_MMA
    static_assert(D == 256 && BLK == 64, "256-wide heads, 64-key tiles");
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int KS = D / 16;    // k-steps of K Q^T
    constexpr int KC = CH / 16;   // k-steps per chunk
    constexpr int QT = COLS / G;  // tokens per CTA
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(smem);

    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y;
    const int seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int pr = warp / 2, h = warp % 2;
    const int gq = lane / 4, tq = lane % 4;
    const int n_past = p.n_past[seq];
    float * xs = (float *) (smem + NS * CHUNK) + pr * X_WORDS;
    uint32_t * xpf = (uint32_t *) (xs + X_PF);

    auto tok_of = [&](int c) { return min(t0 + min(c / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(COLS - 1) + 1;
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    // The pair's 16 columns, and this thread's columns 16 pr + 8 nt + 2 tq + {0, 1}.
    const int w_pos_lo = pos_of(pr * 16), w_pos_hi = pos_of(pr * 16 + 15);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    int c_pos[2][2], c_first[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            c_pos[nt][j]   = pos_of(pr * 16 + nt * 8 + 2 * tq + j);
            c_first[nt][j] = first_key(c_pos[nt][j], p.window);
        }

    // Q as B fragments per n8 tile, as paged_prefill_mma's.
    uint32_t qf[KS][2][2];
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        const int c = pr * 16 + nt * 8 + gq;
        const int t = tok_of(c);
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const float2 a = *(const float2 *) (qr + ks * 16 + 2 * tq);
            const float2 b = *(const float2 *) (qr + ks * 16 + 8 + 2 * tq);
            qf[ks][nt][0] = pack(a.x, a.y);
            qf[ks][nt][1] = pack(b.x, b.y);
        }
    }

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
    auto pair_sync = [&]() { asm volatile("bar.sync %0, 64;\n" ::"r"(1 + pr) : "memory"); };

    // O^T fragments: chunk c's dims 128 c + 16 (4 h + ib) + gq (+ 8), columns 2 tq + {0, 1} of tile nt.
    float o[NC][4][2][4];
#pragma unroll
    for (int c = 0; c < NC; c++)
#pragma unroll
        for (int ib = 0; ib < 4; ib++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) o[c][ib][nt][0] = o[c][ib][nt][1] = o[c][ib][nt][2] = o[c][ib][nt][3] = 0.0f;
    float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, l[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const float scale = p.scale * LOG2E;

    constexpr int ROWB = UNITS * 16;
    const uint32_t k_lane = (lane / 8 % 2 * 8 + lane % 8) * ROWB, v_lane = (lane / 16 * 8 + lane % 8) * ROWB;
    uint32_t k_x[4], v_x[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        k_x[r] = ((2 * r + lane / 16) ^ (lane % 8)) << 4;
        v_x[r] = ((2 * r + lane / 8 % 2) ^ (lane % 8)) << 4;
    }
    // Warp h's keys 32 h.. of K, and its dim blocks 4 h.. of V (the second 128-byte unit group).
    const uint32_t k_half = (uint32_t) (h * 32 * ROWB), v_half = (uint32_t) (h * 128);

    for (int j = 0; j < NS - 1; j++) {
        issue(j);
    }
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[2][2][4];      // S^T: keys 32 h + 16 il + gq (+ 8), columns 2 tq + {0, 1} of tile nt
        uint32_t pf[4][2][2];  // P as B fragments per 16-key step and tile
        float alpha[2][2];
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
                // S^T += K Q^T over this chunk's dims; the sums carry across the chunks.
#pragma unroll
                for (int il = 0; il < 2; il++) {
                    if (part == 0) {
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) s[il][nt][0] = s[il][nt][1] = s[il][nt][2] = s[il][nt][3] = 0.0f;
                    }
#pragma unroll
                    for (int ks = 0; ks < KC; ks++) {
                        uint32_t a[4];
                        ldsm(a, st + k_half + k_lane + k_x[ks % 4] + il * 16 * ROWB + ks / 4 * 128);
                        mma(s[il][0], a, qf[part * KC + ks][0]);
                        mma(s[il][1], a, qf[part * KC + ks][1]);
                    }
                }
                if (part == NC - 1) {
                    float mx[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
#pragma unroll
                    for (int il = 0; il < 2; il++)
#pragma unroll
                        for (int nt = 0; nt < 2; nt++)
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                float x = s[il][nt][e] * scale;
                                if (!whole) {
                                    const int key = k0 + h * 32 + il * 16 + gq + 8 * (e / 2);
                                    x = key <= c_pos[nt][e % 2] && key >= c_first[nt][e % 2] ? x : -INFINITY;
                                }
                                s[il][nt][e] = x;
                                mx[nt][e % 2] = fmaxf(mx[nt][e % 2], x);
                            }
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int c = 0; c < 2; c++) {
#pragma unroll
                            for (int off = 4; off < 32; off <<= 1) {
                                mx[nt][c] = fmaxf(mx[nt][c], __shfl_xor_sync(0xffffffff, mx[nt][c], off));
                            }
                            if (gq == 0) {
                                xs[X_MAX + (h * 4 + nt * 2 + c) * 4 + tq] = mx[nt][c];
                            }
                        }
                    pair_sync();
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int c = 0; c < 2; c++) {
                            const float other = xs[X_MAX + ((1 - h) * 4 + nt * 2 + c) * 4 + tq];
                            const float mxc = fmaxf(mx[nt][c], other);
                            const float m_new = fmaxf(m[nt][c], mxc);
                            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                            alpha[nt][c] = exp2f(m[nt][c] - m_use);
                            m[nt][c]  = m_new;
                            mx[nt][c] = m_use;
                        }
                    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
                    for (int il = 0; il < 2; il++)
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                s[il][nt][e] = exp2f(s[il][nt][e] - mx[nt][e % 2]);
                            }
                            xpf[(((h * 2 + il) * 2 + nt) * 2 + 0) * 32 + lane] = transpose(pack(s[il][nt][0], s[il][nt][1]));
                            xpf[(((h * 2 + il) * 2 + nt) * 2 + 1) * 32 + lane] = transpose(pack(s[il][nt][2], s[il][nt][3]));
                        }
                    if (h == 0) {
                        // Warp 0's share of the thread's chain (its keys come first).
#pragma unroll
                        for (int il = 0; il < 2; il++)
#pragma unroll
                            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                                for (int e = 0; e < 4; e++) sum[nt][e % 2] += s[il][nt][e];
#pragma unroll
                        for (int nt = 0; nt < 2; nt++)
#pragma unroll
                            for (int c = 0; c < 2; c++) xs[X_SUM + (nt * 2 + c) * 32 + lane] = sum[nt][c];
                    }
                    pair_sync();
                    if (h == 1) {
#pragma unroll
                        for (int nt = 0; nt < 2; nt++)
#pragma unroll
                            for (int c = 0; c < 2; c++) sum[nt][c] = xs[X_SUM + (nt * 2 + c) * 32 + lane];
#pragma unroll
                        for (int il = 0; il < 2; il++)
#pragma unroll
                            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                                for (int e = 0; e < 4; e++) sum[nt][e % 2] += s[il][nt][e];
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
                            l[nt][0] = l[nt][0] * alpha[nt][0] + sum[nt][0];
                            l[nt][1] = l[nt][1] * alpha[nt][1] + sum[nt][1];
                        }
                    }
#pragma unroll
                    for (int ks = 0; ks < 4; ks++)
#pragma unroll
                        for (int nt = 0; nt < 2; nt++)
#pragma unroll
                            for (int r = 0; r < 2; r++) pf[ks][nt][r] = xpf[((ks * 2 + nt) * 2 + r) * 32 + lane];
                }
            } else {
                // O^T = O^T alpha + V^T P for this warp's 64 dims of the chunk.
                const int c = part - NC;
#pragma unroll
                for (int ib = 0; ib < 4; ib++) {
                    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
                    for (int ks = 0; ks < 4; ks++) {
                        uint32_t a[4];
                        ldsm_t(a, st + v_half + v_lane + v_x[ib] + ks * 16 * ROWB);
                        mma(acc[0], a, pf[ks][0]);
                        mma(acc[1], a, pf[ks][1]);
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        float * oi = o[c][ib][nt];
                        oi[0] = oi[0] * alpha[nt][0] + acc[nt][0];
                        oi[1] = oi[1] * alpha[nt][1] + acc[nt][1];
                        oi[2] = oi[2] * alpha[nt][0] + acc[nt][2];
                        oi[3] = oi[3] * alpha[nt][1] + acc[nt][3];
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // Warp 1 holds the columns' sums; both warps write their dims.
    if (h == 1) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
#pragma unroll
                for (int off = 4; off < 32; off <<= 1) {
                    l[nt][c] += __shfl_xor_sync(0xffffffff, l[nt][c], off);
                }
                if (gq == 0) {
                    xs[X_L + (nt * 2 + c) * 4 + tq] = l[nt][c];
                }
            }
    }
    pair_sync();
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const int col = pr * 16 + nt * 8 + 2 * tq + c;
            const int t = t0 + col / G;
            if (col < QT * G && t < p.n_tok) {
                const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + col % G;
                const float inv = 1.0f / xs[X_L + (nt * 2 + c) * 4 + tq];
                float * out = p.dst + row * D + h * 64 + gq;
#pragma unroll
                for (int cc = 0; cc < NC; cc++)
#pragma unroll
                    for (int ib = 0; ib < 4; ib++) {
                        out[cc * CH + ib * 16]     = o[cc][ib][nt][c] * inv;
                        out[cc * CH + ib * 16 + 8] = o[cc][ib][nt][2 + c] * inv;
                    }
            }
        }
#else
    __trap();
#endif
}

// Whether the warp-pair kernel runs here: consumer Blackwell (compute capability 12.x: the GB10,
// where it was measured, #1133), not for a tiered pool's pieces.
bool takes(const legion_paged_attn & p) {
    const int cc = legion_device::compute_capability();
    return cc >= 120 && cc < 130 && !p.ml && !p.piece_blk;
}

// 0: launched; 1: not for this device (paged_prefill_mma instead); 2: a launch error.
template <int D, int G, bool C>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_pair<D, G, C>, (int) SMEM)) {
        return 1;
    }
    constexpr int QT = COLS / G;
    paged_prefill_pair<D, G, C><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), NW * 32, SMEM, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace pair
