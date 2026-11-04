// paged_attn/common.cuh, sm_70 (Volta): the one-pass prefill on mma.sync m8n8k4 (namespace v70,
// paged_prefill_m884 and its launcher).
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace, after the
// shared helpers it uses (BLK, walk, block_cell, merge_row, the KV load helpers,
// t_m884_launches). The kernel compiles for every target architecture (its MMA keeps its
// __CUDA_ARCH__ guard and __trap() stub); paged_attn/common.cuh's launch takes it when use_m884() says so
// (LEGION_PREFILL_M884 on an sm_70 device, or a test's per-thread override).

// ---------------------------------------------------------------------------------------------
// Prefill on sm_70 (Volta): one pass with an online softmax, as paged_prefill_mma, on the one
// MMA Volta has, mma.sync m8n8k4 (F16 in, F32 accumulate). Volta lacks cp.async, ldmatrix,
// movmatrix and m16n8k16, so K, V and P go through shared memory in layouts the m8n8k4
// fragments read directly. The instruction runs on later GPUs too (more slowly), which is how
// the oracle tests check it on sm_80+ (legion_paged_attn_set_prefill); in serving it runs on
// sm_70 only.
//
// m8n8k4 is four independent 8x8x4 products per warp, one per quad pair (lanes 4q..4q+3 and
// 4q+16..4q+19). As in ggml's mma.cuh (MIT, its Volta tile<32, 8> with the permuted row
// order), the four take the same B operand, so one instruction is a 32 x 8 x 4 product: lane l
// holds row l of A (4 k values), and column (l & 3) + 4 (l >> 4) of B; C register i of lane l
// is row (l & ~2) + 2 ((i >> 1) & 1), column (l & 2) + 4 (i >> 2) + (i & 1). (The layout of
// Strata-AC922's Volta prompt attention, ea91c64 / 0ff5b41 (MIT, from Klaus Friedel's Strata
// PR #600), measured on a V100; checked on GB10 as well.)
//
// A warp owns 8 query columns (as paged_prefill_mma: column = token_in_tile * G +
// head_in_group) and, per 64-key tile: S^T = K Q^T as two 32 x 8 products (A = K rows from
// shared memory, B = the warp's Q columns from shared memory), the online softmax on those
// registers (lane l holds keys (l & ~2) + {0, 2} of each half for columns (l & 2) + {0, 1, 4,
// 5}), P to a per-warp shared buffer, then O^T += V^T P (A = V transposed into shared memory
// on its way in, B = P), the warp's full-width output in F32 registers (D / 4 per thread).
// K and V stream in 128-dim chunks through one shared buffer, the next chunk's global loads in
// registers while the current one is computed.
//
// Truncating tensor cores (see paged_attn/common.cuh's header): every accumulator chain is at most 16
// MMAs (64 dims of a score, 64 keys of an output) from zero, added to the F32 totals with
// ordinary adds.
// ---------------------------------------------------------------------------------------------

namespace v70 {

constexpr int NW   = 8;           // warps per CTA
constexpr int COLS = 8 * NW;      // query columns per CTA
constexpr int CH   = 128;         // dims per K / V chunk
constexpr int KLD  = CH + 4;      // halves per K chunk row (a key): 8-byte reads conflict-free
constexpr int VLD  = BLK + 4;     // halves per V^T chunk row (a dim)
constexpr int PLD  = BLK + 4;     // halves per P row (a query column)
constexpr int LOADS = BLK * CH / 8 / (NW * 32);  // 16-byte loads per thread per chunk

template <int D> struct Smem {
    static constexpr int    QLD  = D + 4;  // halves per Q row (a query column)
    static constexpr size_t Q    = 0;
    static constexpr size_t KV   = Q + (size_t) COLS * QLD * 2;
    static constexpr size_t KVB  = (size_t) (BLK * KLD > CH * VLD ? BLK * KLD : CH * VLD) * 2;
    static constexpr size_t P    = KV + KVB;
    static constexpr size_t SIZE = P + (size_t) NW * 8 * PLD * 2;
};
static_assert(Smem<512>::SIZE <= 96 * 1024, "Volta: 96 KB of shared memory per CTA");

// c += a b: rows of a (4 k values) and columns of b as above.
__device__ __forceinline__ void mma884(float c[8], uint2 a, uint2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
#else
    __trap();
#endif
}

template <int D, int G, bool C = false>
__global__ void __launch_bounds__(NW * 32, 1) paged_prefill_m884(const legion_paged_attn p) {
    using L = Smem<D>;
    constexpr int NC = D / CH;    // chunks of K, and of V, per key tile
    constexpr int QT = COLS / G;  // tokens per CTA
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(16) unsigned char smem[];
    half * qs = (half *) (smem + L::Q);
    half * kvs = (half *) (smem + L::KV);

    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    half * pw = (half *) (smem + L::P) + warp * 8 * PLD;
    const int n_past = p.n_past[seq];

    // As paged_prefill_mma: padding columns repeat the tile's last token and are not written.
    auto tok_of = [&](int c) { return min(t0 + min(c / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int c) { return n_past + tok_of(c); };
    const int k_end   = pos_of(COLS - 1) + 1;
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    const int w_pos_lo = pos_of(warp * 8), w_pos_hi = pos_of(warp * 8 + 7);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);
    // This lane's four C columns, (lane & 2) + {0, 1, 4, 5}: column c of [4] is register i with
    // (i & 1) + 2 (i >> 2) == c. Its B column (Q and P reads) is bcol.
    const int bcol = (lane & 3) + 4 * (lane >> 4);
    auto col_of = [&](int c) { return warp * 8 + (lane & 2) + (c & 1) + 4 * (c >> 1); };

    // Q as F16, a row per column.
    for (int i = threadIdx.x; i < COLS * D; i += NW * 32) {
        const int c = i / D, e = i % D;
        const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok_of(c) * p.q_nb2 +
                                            (kvh * G + c % G) * p.q_nb1);
        qs[c * L::QLD + e] = __float2half(qr[e]);
    }

    // Chunk j of the CTA's stream: key tile wk.i0 + j / (2 NC), K chunks then V chunks; the
    // thread's LOADS 16-byte pieces of it. K: piece i is key i / 16, dims 8 (i % 16)..; V: key
    // i % 64, dims 8 (i / 64).. (a warp's 32 keys of one dim range: conflict-free transposed
    // stores). Keys at or past k_end are zero.
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    uint4 pre[LOADS];
    auto fetch = [&](int j) {
        if (j >= n_chunks) {
            return;
        }
        const int wi = wk.i0 + j / (2 * NC), part = j % (2 * NC);
        const int b = wk.block(wi);
        const bool is_k = part < NC;
        const half * src = (const half *) (is_k ? wk.cells(p.k, p.k_stage, wi, p.k_cell) : wk.cells(p.v, p.v_stage, wi, p.v_cell)) +
                           head_at<D, C>(p, kvh) + (part % NC) * CH;
#pragma unroll
        for (int l = 0; l < LOADS; l++) {
            const int i = threadIdx.x + l * NW * 32;
            const int r = is_k ? i / (CH / 8) : i % BLK, u = is_k ? i % (CH / 8) : i / BLK;
            pre[l] = b * BLK + r < k_end ? *(const uint4 *) (src + (int64_t) r * p.kv_row + u * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto store = [&](int j) {
        const bool is_k = j % (2 * NC) < NC;
#pragma unroll
        for (int l = 0; l < LOADS; l++) {
            const int i = threadIdx.x + l * NW * 32;
            if (is_k) {
                const int r = i / (CH / 8), u = i % (CH / 8);
                uint2 * dst = (uint2 *) (kvs + r * KLD + u * 8);
                dst[0] = make_uint2(pre[l].x, pre[l].y);
                dst[1] = make_uint2(pre[l].z, pre[l].w);
            } else {
                const int r = i % BLK, u = i / BLK;
                const uint32_t w[4] = {pre[l].x, pre[l].y, pre[l].z, pre[l].w};
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    kvs[(u * 8 + e) * VLD + r] = __ushort_as_half((unsigned short) (w[e / 2] >> (16 * (e % 2))));
                }
            }
        }
    };

    float o[D / 32][8];
#pragma unroll
    for (int g = 0; g < D / 32; g++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            o[g][i] = 0.0f;
        }
    }
    float m[4], l[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
        m[c] = -INFINITY;
        l[c] = 0.0f;
    }
    const float scale = p.scale * LOG2E;

    fetch(0);
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[2][8];  // S^T: keys 32 h + (lane & ~2) + 2 ((i >> 1) & 1), the lane's columns
        float alpha[4];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            __syncthreads();  // the previous chunk is read (and, the first time, Q written)
            store(j);
            __syncthreads();
            fetch(j + 1);
            if (!active) {
                continue;
            }
            if (part < NC) {
                // S^T += K Q^T over this chunk's dims, 64 at a time.
#pragma unroll
                for (int kk = 0; kk < CH / 64; kk++) {
                    float acc[2][8] = {};
#pragma unroll
                    for (int ks = 0; ks < 16; ks++) {
                        const int e = kk * 64 + ks * 4;
                        const uint2 bq = *(const uint2 *) (qs + (warp * 8 + bcol) * L::QLD + part * CH + e);
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const uint2 a = *(const uint2 *) (kvs + (h * 32 + lane) * KLD + e);
                            mma884(acc[h], a, bq);
                        }
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            s[h][i] = part == 0 && kk == 0 ? acc[h][i] : s[h][i] + acc[h][i];
                        }
                    }
                }
                if (part == NC - 1) {
                    float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
                    for (int h = 0; h < 2; h++) {
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            const int c = (i & 1) + 2 * (i >> 2);
                            float x = s[h][i] * scale;
                            if (!whole) {
                                // Recomputed rather than held: registers are scarce at D = 512.
                                const int key = k0 + h * 32 + (lane & ~2) + 2 * ((i >> 1) & 1);
                                const int pos = pos_of(col_of(c));
                                x = key <= pos && key >= first_key(pos, p.window) ? x : -INFINITY;
                            }
                            s[h][i] = x;
                            mx[c] = fmaxf(mx[c], x);
                        }
                    }
                    float mu[4];
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        // A column's keys: lanes differing in bits 0, 2, 3, 4.
                        mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], 1));
                        mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], 4));
                        mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], 8));
                        mx[c] = fmaxf(mx[c], __shfl_xor_sync(0xffffffff, mx[c], 16));
                        const float m_new = fmaxf(m[c], mx[c]);
                        mu[c]    = m_new == -INFINITY ? 0.0f : m_new;  // no visible key yet
                        alpha[c] = exp2f(m[c] - mu[c]);
                        m[c]     = m_new;
                    }
                    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int h = 0; h < 2; h++) {
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            const int c = (i & 1) + 2 * (i >> 2);
                            const float w = exp2f(s[h][i] - mu[c]);
                            sum[c] += w;
                            const int col = (lane & 2) + (c & 1) + 4 * (c >> 1);
                            pw[col * PLD + h * 32 + (lane & ~2) + 2 * ((i >> 1) & 1)] = __float2half(w);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        l[c] = l[c] * alpha[c] + sum[c];
                    }
                    __syncwarp();
                }
            } else {
                // O^T = O^T alpha + V^T P for this chunk's dims, 64 at a time (two 32-dim
                // products per P fragment read).
                const int pc = part - NC;
#pragma unroll
                for (int g2 = 0; g2 < CH / 64; g2++) {
                    float acc[2][8] = {};
#pragma unroll
                    for (int ks = 0; ks < BLK / 4; ks++) {
                        const uint2 bp = *(const uint2 *) (pw + bcol * PLD + ks * 4);
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const uint2 a = *(const uint2 *) (kvs + ((g2 * 2 + h) * 32 + lane) * VLD + ks * 4);
                            mma884(acc[h], a, bp);
                        }
                    }
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        float * og = o[pc * (CH / 32) + g2 * 2 + h];
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            og[i] = og[i] * alpha[(i & 1) + 2 * (i >> 2)] + acc[h][i];
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < 4; c++) {
        l[c] += __shfl_xor_sync(0xffffffff, l[c], 1);
        l[c] += __shfl_xor_sync(0xffffffff, l[c], 4);
        l[c] += __shfl_xor_sync(0xffffffff, l[c], 8);
        l[c] += __shfl_xor_sync(0xffffffff, l[c], 16);
    }
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int col = col_of(c);
        const int t = t0 + col / G;
        const bool valid = col < QT * G && t < p.n_tok;
        const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + col % G;
        if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. The column's 16 lanes read
            // its running (max, sum) before one of them writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[c], l[c]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            __syncwarp();
            if (valid) {
                if ((lane & ~2) == 0) {
                    store_ml(p, row, f);
                }
                const float inv = l[c] > 0.0f ? 1.0f / l[c] : 0.0f;
                float * out = p.dst + row * D + (lane & ~2);
#pragma unroll
                for (int g = 0; g < D / 32; g++) {
                    const float x0 = o[g][(c & 1) + 4 * (c >> 1)] * inv, x1 = o[g][(c & 1) + 4 * (c >> 1) + 2] * inv;
                    out[g * 32]     = p.piece > 0 ? f.x * out[g * 32] + f.y * x0 : x0;
                    out[g * 32 + 2] = p.piece > 0 ? f.x * out[g * 32 + 2] + f.y * x1 : x1;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / l[c];
            float * out = p.dst + row * D + (lane & ~2);
#pragma unroll
            for (int g = 0; g < D / 32; g++) {
                out[g * 32]     = o[g][(c & 1) + 4 * (c >> 1)] * inv;
                out[g * 32 + 2] = o[g][(c & 1) + 4 * (c >> 1) + 2] * inv;
            }
        }
    }
}

// 0 or 2 as legion_paged_attn_launch; -1 when the device cannot give the kernel its shared
// memory (the caller falls back to the default kernels).
template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    static std::atomic<int8_t> opted_in[GGML_CUDA_MAX_DEVICES];
    if (!legion_device::smem_opt_in(opted_in, paged_prefill_m884<D, G, C>, (int) Smem<D>::SIZE)) {
        return -1;
    }
    constexpr int QT = COLS / G;
    paged_prefill_m884<D, G, C><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), NW * 32, Smem<D>::SIZE, st>>>(p);
    t_m884_launches++;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace v70
