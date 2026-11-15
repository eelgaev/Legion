// paged_attn/common.cuh, Turing without tensor cores (GTX 16xx, MX450, MX550; cc 7.5): the
// one-pass prefill on FP32 FMAs (namespace tu75, paged_prefill_ffma and its launcher).
//
// Not a standalone header: paged_attn/common.cuh includes it inside its anonymous namespace,
// after the shared helpers it uses (BLK, walk, first_key, merge_row, store_ml, head_at,
// warp_sum, t_ffma_launches). The kernel is plain FP32 math and compiles for every target;
// paged_attn/common.cuh's launch takes it when use_ffma() says so (LEGION_PREFILL_FFMA on an
// sm_75 device, or a test's per-thread override).
//
// ---------------------------------------------------------------------------------------------
// Why: these GPUs have no tensor cores. Their HMMA instructions (what WMMA compiles to) run
// emulated at about a quarter of the FP32 FMA rate, so the default WMMA prefill (two passes,
// 1.5x the flops of one) spends most of a long prompt there. This kernel does one pass with an
// online softmax on FFMA: Q, K, V and P in F32 (K and V converted from the F16 cache as they
// are staged), scores, softmax and output in F32.
//
// Not the WMMA kernel's bits (no FMA chain reproduces the emulated HMMA's rounding), so it is
// opt-in (--tu-attn-ffma). It is closer to exact attention than the default: nothing is
// rounded to F16 on the way (WMMA rounds Q and P), and the accumulation rounds to nearest.
//
// A CTA takes 64 query rows (64 / G tokens times the G query heads of one kv head, as the WMMA
// kernel; padding rows repeat the tile's last token and are not written), 8 warps of 8 rows.
// Per 64-key block: S = Q K^T with each lane on keys `lane` and `lane + 32` of its warp's 8
// rows (Q read as broadcasts, K rows float4 at a time, conflict-free), the online softmax on
// those registers (a row's max and sum across the warp), P to the warp's own rows of shared
// memory, then O = O alpha + P V with each lane on dims 2 lane, 2 lane + 1 of every 64-dim
// chunk (P read as broadcasts, V as float2). K and V stream in 64-dim chunks through one
// shared buffer, Q's matching 64 dims (from global memory, F32 as the RoPE wrote it) with each
// K chunk; the next chunk's global loads wait in registers while the current one is computed.
// P takes the Q chunk's buffer once the scores are done (each warp reads only its own rows of
// both). 34,816 bytes of shared memory for every head size.
// ---------------------------------------------------------------------------------------------

namespace tu75 {

constexpr int NW     = 8;              // warps per CTA
constexpr int RW     = 8;              // query rows per warp
constexpr int ROWS   = NW * RW;        // query rows per CTA
constexpr int CH     = 64;             // dims per K / V / Q chunk
constexpr int LD     = CH + 4;         // floats per staged row: float4 reads of 8 rows conflict-free
constexpr int KLOADS = BLK * CH / 8 / (NW * 32);   // 16-byte K / V loads per thread per chunk
constexpr int QLOADS = ROWS * CH / 4 / (NW * 32);  // float4 Q loads per thread per K chunk
constexpr size_t SMEM = (size_t) (ROWS + BLK) * LD * sizeof(float);

static_assert(BLK == 64, "lanes take keys lane and lane + 32");
static_assert(SMEM <= 48 * 1024, "no shared memory opt-in needed");

template <int D, int G, bool C = false>
__global__ void __launch_bounds__(NW * 32, 1) paged_prefill_ffma(const legion_paged_attn p) {
    constexpr int NC = D / CH;      // chunks of K, and of V, per key block
    constexpr int QT = ROWS / G;    // tokens per CTA
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(16) unsigned char smem[];
    float * qp = (float *) smem;    // Q chunk [row][LD], then P [row][LD]
    float * kv = qp + ROWS * LD;    // K or V chunk [key][LD]

    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n_past = p.n_past[seq];

    auto tok_of = [&](int r) { return min(t0 + min(r / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int r) { return n_past + tok_of(r); };
    const int k_end   = pos_of(ROWS - 1) + 1;
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int b_last  = (k_end - 1) / BLK;
    const Walk wk = walk(p, seq, b_first, b_last);
    if (piece_idle(p, wk)) {
        return;
    }

    const int w_pos_lo = pos_of(warp * RW), w_pos_hi = pos_of(warp * RW + RW - 1);
    const int w_first_lo = first_key(w_pos_lo, p.window), w_first_hi = first_key(w_pos_hi, p.window);

    // Chunk j of the CTA's stream: key block wk.i0 + j / (2 NC), K chunks (with Q's matching
    // dims) then V chunks. K / V piece i of a thread: key i / 8, dims 8 (i % 8)..; Q piece i:
    // row i / 16, dims 4 (i % 16)... Keys at or past k_end are zero (the pool may hold anything
    // there, and 0 times a NaN would spread).
    const int n_chunks = (wk.i1 - wk.i0) * 2 * NC;
    uint4  pre[KLOADS];
    float4 qpre[QLOADS];
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
        for (int l = 0; l < KLOADS; l++) {
            const int i = threadIdx.x + l * NW * 32;
            const int r = i / (CH / 8), u = i % (CH / 8);
            pre[l] = b * BLK + r < k_end ? *(const uint4 *) (src + (int64_t) r * p.kv_row + u * 8) : make_uint4(0, 0, 0, 0);
        }
        if (is_k) {
#pragma unroll
            for (int l = 0; l < QLOADS; l++) {
                const int i = threadIdx.x + l * NW * 32;
                const int r = i / (CH / 4), u = i % (CH / 4);
                const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok_of(r) * p.q_nb2 +
                                                    (kvh * G + r % G) * p.q_nb1);
                qpre[l] = *(const float4 *) (qr + part * CH + u * 4);
            }
        }
    };
    auto store = [&](int j) {
#pragma unroll
        for (int l = 0; l < KLOADS; l++) {
            const int i = threadIdx.x + l * NW * 32;
            const int r = i / (CH / 8), u = i % (CH / 8);
            const half2 * h = (const half2 *) &pre[l];
            const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
            const float2 c = __half22float2(h[2]), d = __half22float2(h[3]);
            *(float4 *) (kv + r * LD + u * 8)     = make_float4(a.x, a.y, b.x, b.y);
            *(float4 *) (kv + r * LD + u * 8 + 4) = make_float4(c.x, c.y, d.x, d.y);
        }
        if (j % (2 * NC) < NC) {
#pragma unroll
            for (int l = 0; l < QLOADS; l++) {
                const int i = threadIdx.x + l * NW * 32;
                const int r = i / (CH / 4), u = i % (CH / 4);
                *(float4 *) (qp + r * LD + u * 4) = qpre[l];
            }
        }
    };

    float o[NC][RW][2];
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < RW; r++) {
            o[c][r][0] = 0.0f;
            o[c][r][1] = 0.0f;
        }
    }
    float m[RW], l[RW];  // l: this lane's part of the row's sum
#pragma unroll
    for (int r = 0; r < RW; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    const float scale = p.scale * LOG2E;
    float * qw = qp + warp * RW * LD;  // the warp's rows of Q, then of P

    fetch(0);
    int j = 0;
    for (int wi = wk.i0; wi < wk.i1; wi++) {
        const int k0 = wk.block(wi) * BLK;
        const bool active = k0 <= w_pos_hi && k0 + BLK - 1 >= w_first_lo;
        const bool whole  = k0 >= w_first_hi && k0 + BLK - 1 <= w_pos_lo;
        float s[RW][2];  // scores, then weights: rows of the warp, keys lane and lane + 32
        float alpha[RW];
#pragma unroll
        for (int part = 0; part < 2 * NC; part++, j++) {
            __syncthreads();  // the previous chunk (and P) read
            store(j);
            __syncthreads();
            fetch(j + 1);
            if (!active) {
                continue;
            }
            if (part < NC) {
                if (part == 0) {
#pragma unroll
                    for (int r = 0; r < RW; r++) {
                        s[r][0] = 0.0f;
                        s[r][1] = 0.0f;
                    }
                }
                const float * ka = kv + lane * LD;
                const float * kb = kv + (lane + 32) * LD;
#pragma unroll 2
                for (int e = 0; e < CH; e += 4) {
                    const float4 x = *(const float4 *) (ka + e);
                    const float4 y = *(const float4 *) (kb + e);
#pragma unroll
                    for (int r = 0; r < RW; r++) {
                        const float4 q = *(const float4 *) (qw + r * LD + e);
                        s[r][0] = fmaf(q.x, x.x, s[r][0]);
                        s[r][1] = fmaf(q.x, y.x, s[r][1]);
                        s[r][0] = fmaf(q.y, x.y, s[r][0]);
                        s[r][1] = fmaf(q.y, y.y, s[r][1]);
                        s[r][0] = fmaf(q.z, x.z, s[r][0]);
                        s[r][1] = fmaf(q.z, y.z, s[r][1]);
                        s[r][0] = fmaf(q.w, x.w, s[r][0]);
                        s[r][1] = fmaf(q.w, y.w, s[r][1]);
                    }
                }
                if (part == NC - 1) {
#pragma unroll
                    for (int r = 0; r < RW; r++) {
                        float x0 = s[r][0] * scale, x1 = s[r][1] * scale;
                        if (!whole) {
                            const int pos = pos_of(warp * RW + r), lo = first_key(pos, p.window);
                            const int key0 = k0 + lane, key1 = key0 + 32;
                            x0 = key0 <= pos && key0 >= lo ? x0 : -INFINITY;
                            x1 = key1 <= pos && key1 >= lo ? x1 : -INFINITY;
                        }
                        float mx = fmaxf(x0, x1);
#pragma unroll
                        for (int sh = 16; sh > 0; sh >>= 1) {
                            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, sh));
                        }
                        const float m_new = fmaxf(m[r], mx);
                        const float mu    = m_new == -INFINITY ? 0.0f : m_new;  // no visible key yet
                        alpha[r] = exp2f(m[r] - mu);
                        m[r]     = m_new;
                        s[r][0]  = exp2f(x0 - mu);
                        s[r][1]  = exp2f(x1 - mu);
                        l[r]     = l[r] * alpha[r] + (s[r][0] + s[r][1]);
                    }
                    __syncwarp();  // the warp's lanes are done with its Q rows
#pragma unroll
                    for (int r = 0; r < RW; r++) {
                        qw[r * LD + lane]      = s[r][0];
                        qw[r * LD + lane + 32] = s[r][1];
                    }
                    __syncwarp();
                }
            } else {
                float (&oc)[RW][2] = o[part - NC];
#pragma unroll
                for (int r = 0; r < RW; r++) {
                    oc[r][0] *= alpha[r];
                    oc[r][1] *= alpha[r];
                }
                const float * vp = kv + 2 * lane;
#pragma unroll 2
                for (int k = 0; k < BLK; k += 4) {
                    const float2 v0 = *(const float2 *) (vp + (k + 0) * LD);
                    const float2 v1 = *(const float2 *) (vp + (k + 1) * LD);
                    const float2 v2 = *(const float2 *) (vp + (k + 2) * LD);
                    const float2 v3 = *(const float2 *) (vp + (k + 3) * LD);
#pragma unroll
                    for (int r = 0; r < RW; r++) {
                        const float4 w = *(const float4 *) (qw + r * LD + k);
                        oc[r][0] = fmaf(w.x, v0.x, oc[r][0]);
                        oc[r][1] = fmaf(w.x, v0.y, oc[r][1]);
                        oc[r][0] = fmaf(w.y, v1.x, oc[r][0]);
                        oc[r][1] = fmaf(w.y, v1.y, oc[r][1]);
                        oc[r][0] = fmaf(w.z, v2.x, oc[r][0]);
                        oc[r][1] = fmaf(w.z, v2.y, oc[r][1]);
                        oc[r][0] = fmaf(w.w, v3.x, oc[r][0]);
                        oc[r][1] = fmaf(w.w, v3.y, oc[r][1]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RW; r++) {
        const float lt = warp_sum(l[r]);
        const int row_c = warp * RW + r;
        const int t = t0 + row_c / G;
        const bool valid = row_c < QT * G && t < p.n_tok;  // the same for the whole warp
        const int64_t row = ((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + row_c % G;
        float * out = p.dst + row * D + 2 * lane;
        if (p.ml) {
            // Pieces (#352): merged into the earlier pieces' output. Every lane reads the row's
            // running (max, sum) before lane 0 writes it.
            const float4 f = valid ? merge_row<true>(p, row, m[r], lt) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            __syncwarp();
            if (valid) {
                if (lane == 0) {
                    store_ml(p, row, f);
                }
                const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    float2 x = make_float2(o[c][r][0] * inv, o[c][r][1] * inv);
                    if (p.piece > 0) {
                        const float2 old = *(const float2 *) (out + c * CH);
                        x = make_float2(f.x * old.x + f.y * x.x, f.x * old.y + f.y * x.y);
                    }
                    *(float2 *) (out + c * CH) = x;
                }
            }
        } else if (valid) {
            const float inv = 1.0f / lt;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                *(float2 *) (out + c * CH) = make_float2(o[c][r][0] * inv, o[c][r][1] * inv);
            }
        }
    }
}

// 0 or 2 as legion_paged_attn_launch; -1 when Q's rows are not 16-byte aligned (its float4
// loads; the caller falls back to the default kernels).
template <int D, int G, bool C = false>
int launch(const legion_paged_attn & p, cudaStream_t st) {
    if ((((uintptr_t) p.q) | (uintptr_t) p.q_nb1 | (uintptr_t) p.q_nb2 | (uintptr_t) p.q_nb3) % 16 != 0) {
        return -1;
    }
    constexpr int QT = ROWS / G;
    paged_prefill_ffma<D, G, C><<<dim3((p.n_tok + QT - 1) / QT, p.n_head_kv, p.n_seqs), NW * 32, SMEM, st>>>(p);
    t_ffma_launches++;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace tu75
