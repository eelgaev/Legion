// expert_mv's kernel templates and their launch code per weight type (see common.cuh).
#pragma once

#include "common.cuh"

#include <algorithm>
#include <atomic>

namespace {

// rows_dot with exactly NU users (no per-user predicate) and each unit's weights decoded once
// for all of them. Every (row, user) sum is rows_dot's: the same lanes of a row take the same
// units in the same order and the partial sums meet in the same tree. With SUB, a row rows_dot
// gives a whole warp (L == 32) takes 16 lanes, two rows per warp: lane h keeps the sums of
// rows_dot's lanes h and h + 16 apart and adds them as the first shuffle step would (Strata #783
// PR-h's row_dot_80_sub16, 2715a75, MIT: the same addition tree with every lane busy).
// XS: the activations from xs in shared memory (rows_dot's).
template <ggml_type type, int NU, bool GLU, bool SUB, bool XS = false>
__device__ __forceinline__ void rows_dot_once(const legion_expert_mv & p, const char * rows_at, const char * rows_at2,
                                              int nrows, int r0, const Users & us, const block_q8_1 * xs = nullptr) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int VDR             = vdr<type>();
    constexpr int lanes_per_block = qi / VDR;
    constexpr int M               = GLU ? 2 : 1;
    const int units = units_per_row<type>(p.k);
    const int xcols = p.k / QK8_1;
    const int L     = lanes_per_row(units);  // the lanes whose order the sums keep
    const bool half = SUB && L == WARP_SIZE;
    const int G     = half ? WARP_SIZE / 2 : L;  // the lanes a row takes here
    const int sub   = threadIdx.x % G;
    const block_q8_1 * ys[NU];
#pragma unroll
    for (int u = 0; u < NU; u++) {
        const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
        if constexpr (XS) {
            ys[u] = xs + u * xcols;
        } else {
            ys[u] = p.xq + (int64_t) (t * p.n_x + (p.n_x == 1 ? 0 : j)) * xcols;
        }
    }
    const int lane = threadIdx.x % WARP_SIZE;
    for (int rw = threadIdx.x / WARP_SIZE * (WARP_SIZE / G); rw < nrows; rw += THREADS / G) {
        const int r = rw + lane / G;
        const char * row[M];
        row[0] = rows_at + min(r, nrows - 1) * p.w_nb1;
        if constexpr (GLU) {
            row[1] = rows_at2 + min(r, nrows - 1) * p.w_nb1;
        }
        float acc[M][NU] = {};
        float acc2[M][NU] = {};  // half: the units of rows_dot's lane sub + 16
        auto add = [&](int c, float (&a)[M][NU]) {
            const int kbx = c / lanes_per_block, iqs = VDR * (c % lanes_per_block);
            Unit<type> w[M];
#pragma unroll
            for (int m = 0; m < M; m++) {
                w[m].load(row[m], kbx, iqs);
            }
#pragma unroll
            for (int u = 0; u < NU; u++) {
                typename Unit<type>::Acts x;
                x.load(&ys[u][kbx * (qk / QK8_1)], iqs);
#pragma unroll
                for (int m = 0; m < M; m++) {
                    a[m][u] += w[m].dot(x);
                }
            }
        };
        for (int c = r < nrows ? sub : units; c < units; c += L) {
            add(c, acc);
            if constexpr (SUB) {
                if (half && c + WARP_SIZE / 2 < units) {
                    add(c + WARP_SIZE / 2, acc2);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NU; u++) {
            float s[M];
#pragma unroll
            for (int m = 0; m < M; m++) {
                s[m] = acc[m][u];
                if constexpr (SUB) {
                    if (half) {
                        s[m] += acc2[m][u];
                    }
                }
                s[m] = sum_lanes(s[m], G);
            }
            if (sub == 0 && r < nrows) {
                const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
                float * out = (float *) ((char *) p.dst + t * p.dst_nb2 + j * p.dst_nb1) + r0;
                out[r]      = GLU ? swiglu(s[0], s[M - 1], p.limit) : s[0];
            }
        }
    }
}

// us.n <= MU: only the user counts up to MU are compiled (expert_mv_tile_piped's MU follows the
// step's tokens; registers for the larger counts made the 3-token kernels spill). XS: rows_dot's.
template <ggml_type type, bool GLU, bool SUB, int MU = MAX_USERS, bool XS = false>
__device__ __forceinline__ void rows_dot_once_any(const legion_expert_mv & p, const char * rows_at,
                                                  const char * rows_at2, int nrows, int r0, const Users & us,
                                                  const block_q8_1 * xs = nullptr) {
    if constexpr (MU == 1) {
        rows_dot_once<type, 1, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
        return;
    }
    switch (us.n) {
        case 1: rows_dot_once<type, 1, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); break;
        case 2: rows_dot_once<type, 2, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); break;
        case 3: if constexpr (MU >= 3) { rows_dot_once<type, 3, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        case 4: if constexpr (MU >= 4) { rows_dot_once<type, 4, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        case 5: if constexpr (MU >= 5) { rows_dot_once<type, 5, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        case 6: if constexpr (MU >= 6) { rows_dot_once<type, 6, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        case 7: if constexpr (MU >= 7) { rows_dot_once<type, 7, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        default: if constexpr (MU >= MAX_USERS) { rows_dot_once<type, MAX_USERS, GLU, SUB, XS>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
    }
}

// The tile kernel's dot products by its DOTS: 0 one vec_dot per user (the only code in that
// kernel, so it runs as it did before the other modes existed), 1 decoded once (but, auto, Q4_K's
// SwiGLU pairs at exactly 4 users: 2% faster per user there on V100), 2 decoded once by 16 lanes.
// XS: rows_dot's (not with the IQ types' split activations).
template <ggml_type type, bool GLU, int DOTS, int MU, bool XS = false>
__device__ __forceinline__ void rows_dot_any(const legion_expert_mv & p, const char * rows_at, const char * rows_at2,
                                             int nrows, int r0, const Users & us, const block_q8_1 * xs = nullptr) {
    if constexpr (DOTS == 2) {
        rows_dot_once_any<type, GLU, true, MU, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
        return;
    } else if constexpr (DOTS == 1) {
        if (MU < 4 || !(type == GGML_TYPE_Q4_K && GLU && us.n == 4 && p.dot == DOT_AUTO)) {
            rows_dot_once_any<type, GLU, false, MU, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
            return;
        }
        if constexpr (MU >= 4) {
            rows_dot<type, 4, GLU, 0, false, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
        }
        return;
    }
    constexpr bool S = xq_split<type, MU>();
    static_assert(!(S && XS), "shared-memory activations are block_q8_1 columns");
    if (MU == 1 || us.n == 1) {
        rows_dot<type, 1, GLU, 0, S, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
    } else if (MU == 2 || us.n == 2) {
        rows_dot<type, 2, GLU, 0, S, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
    } else if (MU <= 4 || us.n <= 4) {
        if constexpr (MU >= 3) {
            rows_dot<type, 4, GLU, 0, false, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
        }
    } else {
        if constexpr (MU > 4) {
            rows_dot<type, MAX_USERS, GLU, 0, false, XS>(p, rows_at, rows_at2, nrows, r0, us, xs);
        }
    }
}

// An int product |x| < 2^22 as a float, exact (as the int -> float conversion it replaces): x
// added to the bits of 1.5 * 2^23 is the float 1.5 * 2^23 + x, and the subtraction is exact. An
// IMAD and an FADD instead of an I2F, which V100 runs at a quarter of the FP32 rate.
__device__ __forceinline__ float exact_float(int a, int b) {
    return __fsub_rn(__int_as_float(a * b + 0x4B400000), 12582912.0f);
}

// A unit's activations for rows_dot_q4k_glu: its two sub-blocks' scales and ints q, q + 4 of each.
__device__ __forceinline__ void q4k_acts(const block_q8_1 * y, int q, int (&uu)[4], float (&d8)[2]) {
#pragma unroll
    for (int s = 0; s < 2; s++) {
        d8[s]          = __low2float(y[s].ds);
        const int * q8 = (const int *) y[s].qs + q;
        uu[2 * s + 0]  = q8[0];
        uu[2 * s + 1]  = q8[4];
    }
}

// The activations of a lane's units i < H for NU users, as rows_dot_q4k_glu reads them from
// block_q8_1 columns (#816 round 6): a lane takes the same units in every row and tile, so a pipe
// group of tiles (tiles_piped) loads them once and keeps them in registers instead of once per tile.
template <int NU, int H> struct Q4kHeld {
    int uu[H > 0 ? H : 1][NU][4];
    float d8[H > 0 ? H : 1][NU][2];
};

// rows_dot_q4k_glu's block_q8_1 activation pointers of a lane: user u's unit i at
// ys[u] + i * 2 * (QK_K / QK8_1).
template <int NU, bool XS>
__device__ __forceinline__ void q4k_ys(const legion_expert_mv & p, const Users & us, const block_q8_1 * xs,
                                       const block_q8_1 * (&ys)[NU]) {
    const int xcols = p.k / QK8_1;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int jj    = (lane % 16) / 4;
#pragma unroll
    for (int u = 0; u < NU; u++) {
        const int qq = us.q[u], t = qq / p.n_used, j = qq % p.n_used;
        const int col = t * p.n_x + (p.n_x == 1 ? 0 : j);
        ys[u] = (XS ? xs + u * xcols : p.xq + (int64_t) col * xcols) + lane / 16 * (QK_K / QK8_1) + 2 * jj;
    }
}

template <int NU, int H, bool XS>
__device__ __forceinline__ void q4k_hold(const legion_expert_mv & p, const Users & us, const block_q8_1 * xs,
                                         Q4kHeld<NU, H> & held) {
    const block_q8_1 * ys[NU];
    q4k_ys<NU, XS>(p, us, xs, ys);
    const int q = threadIdx.x % 4;
#pragma unroll
    for (int i = 0; i < H; i++) {
#pragma unroll
        for (int u = 0; u < NU; u++) {
            q4k_acts(ys[u] + i * 2 * (QK_K / QK8_1), q, held.uu[i][u], held.d8[i][u]);
        }
    }
}

// Q4_K gate/up through SwiGLU in the piped tile kernel (#816 round 3), for rows whose units
// (units_per_row) are exactly 32 * NI: rows_dot_once<GGML_TYPE_Q4_K, NU, true, false>'s sums, the
// same bits. Lane l of a row's warp takes units l + 32 i (i < NI) in that order and the warp's
// lanes meet in sum_lanes as there (lanes_per_row is 32 for these rows); each unit's dot product
// is vec_dot_q4_K_q8_1_impl_vmmq's float expression with the same values (ggml's vecdotq.cuh,
// MIT). What differs is only how the values are fetched and the integers decoded:
// - a unit's block, sub-block and quants are lane constants (unit l + 32 i is in block l / 16 +
//   2 i at the same place), so the units' addresses and the scale selection (k_scales's jm and
//   hi) are worked out once, not per unit, and the loop over them is unrolled;
// - the block's dm and its 12 scale bytes come in one 16-byte shared-memory load;
// - the integer products reach float through exact_float, not I2F.
// Activations: block_q8_1 columns at xs (XS) or in xq, as rows_dot's; UL (not with XS): the
// per-unit layout quantize_x wrote (q4k_acts_col), the same values in two 16-byte loads.
// H > 0 (not with UL): the activations of units i < H come from `held` (q4k_hold's, the same
// values), not memory.
template <int NU, int H> struct Q4kHeld;

template <int NU, int NI, bool XS, bool UL = false, int H = 0>
__device__ __forceinline__ void rows_dot_q4k_glu(const legion_expert_mv & p, const char * rows_at,
                                                 const char * rows_at2, int nrows, int r0, const Users & us,
                                                 const block_q8_1 * xs, const Q4kHeld<NU, H> & held = {}) {
    const int xcols = p.k / QK8_1;
    const int lane  = threadIdx.x % WARP_SIZE;
    const int jj    = (lane % 16) / 4;  // vec_dot_q4_K_q8_1's bq8_offset / 2
    const int q     = lane % 4;         // ... (iqs / 2) % 4
    const int jm    = jj & 1;
    const uint32_t hi = (uint32_t) - (int32_t) (jj >= 2);
    // A unit's quants: v[0] at qs + 16 * bq8_offset + 4 * q of its block (qs at byte 16), v[1] 16
    // bytes on; its activations: blocks kbx * 8 + bq8_offset (+ 1), ints q and q + 4.
    const int v_at = 16 + 32 * jj + 4 * q;
    static_assert(!(XS && UL), "the per-unit activations are read from xq");
    static_assert(!(UL && H > 0), "held activations are block_q8_1's");
    const block_q8_1 * ys[NU];
    const int4 * ua[NU];  // UL: user u's unit `lane` (its ds k / 16 units on)
#pragma unroll
    for (int u = 0; u < NU; u++) {
        const int qq = us.q[u], t = qq / p.n_used, j = qq % p.n_used;
        const int col = t * p.n_x + (p.n_x == 1 ? 0 : j);
        if constexpr (UL) {
            ua[u] = q4k_acts_col(p, col) + lane;
        } else if constexpr (XS) {
            ys[u] = xs + u * xcols;
        } else {
            ys[u] = p.xq + (int64_t) col * xcols;
        }
        if constexpr (!UL) {
            ys[u] += lane / 16 * (QK_K / QK8_1) + 2 * jj;
        }
    }
    const int units = p.k / 16;
    for (int rw = threadIdx.x / WARP_SIZE; rw < nrows; rw += THREADS / WARP_SIZE) {
        const int r = rw;  // a warp per row (uniform)
        const char * row[2];
        row[0] = rows_at + r * p.w_nb1 + lane / 16 * (int) sizeof(block_q4_K);
        row[1] = rows_at2 + r * p.w_nb1 + lane / 16 * (int) sizeof(block_q4_K);
        float acc[2][NU] = {};
        // 4 users with the per-unit activations: a rolled unit loop (unrolled, the scheduler
        // hoisted the next unit's loads and spilled at 128 registers). The same order.
#pragma unroll (UL && NU >= 4 ? 1 : NI)
        for (int i = 0; i < NI; i++) {
            const int boff = 2 * i * (int) sizeof(block_q4_K);  // block l / 16 + 2 i
            int v[2][2];
            uint32_t aux0[2], aux1[2];
            half2 dm[2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const char * b  = row[m] + boff;
                const uint4 h   = *(const uint4 *) b;  // dm, scales[0..12)
                v[m][0]         = *(const int *) (b + v_at);
                v[m][1]         = *(const int *) (b + v_at + 16);
                const uint32_t s0 = jm ? h.y >> 16 : h.y & 0xFFFF;
                const uint32_t s2 = jm ? h.z >> 16 : h.z & 0xFFFF;
                const uint32_t s4 = jm ? h.w >> 16 : h.w & 0xFFFF;
                aux0[m] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
                aux1[m] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
                dm[m]   = *(const half2 *) &h.x;
            }
#pragma unroll
            for (int u = 0; u < NU; u++) {
                int uu[4], dot2[2];
                float d8[2];
                if constexpr (UL) {
                    const int4 a = __ldg(ua[u] + WARP_SIZE * i);
                    const int4 b = __ldg(ua[u] + units + WARP_SIZE * i);
                    uu[0]   = a.x;
                    uu[1]   = a.y;
                    uu[2]   = a.z;
                    uu[3]   = a.w;
                    d8[0]   = __int_as_float(b.x);
                    d8[1]   = __int_as_float(b.y);
                    dot2[0] = b.z;
                    dot2[1] = b.w;
                } else {
                    if (i < H) {
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            uu[c] = held.uu[i < H ? i : 0][u][c];
                        }
                        d8[0] = held.d8[i < H ? i : 0][u][0];
                        d8[1] = held.d8[i < H ? i : 0][u][1];
                    } else {
                        q4k_acts(ys[u] + i * 2 * (QK_K / QK8_1), q, uu, d8);
                    }
#pragma unroll
                    for (int s = 0; s < 2; s++) {
                        dot2[s] = ggml_cuda_dp4a(0x01010101, uu[2 * s + 1], ggml_cuda_dp4a(0x01010101, uu[2 * s + 0], 0));
                    }
                }
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    // vec_dot_q4_K_q8_1_impl_vmmq(v, uu, sc, sc + 2, dm, d8)
                    float sumf_d = 0.0f;
                    float sumf_m = 0.0f;
#pragma unroll
                    for (int s = 0; s < 2; s++) {
                        const int v0i  = (v[m][0] >> (4 * s)) & 0x0F0F0F0F;
                        const int v1i  = (v[m][1] >> (4 * s)) & 0x0F0F0F0F;
                        const int dot1 = ggml_cuda_dp4a(v1i, uu[2 * s + 1], ggml_cuda_dp4a(v0i, uu[2 * s + 0], 0));
                        const int sc   = (aux0[m] >> (8 * s)) & 0xFF;
                        const int mn   = (aux1[m] >> (8 * s)) & 0xFF;
                        sumf_d += d8[s] * exact_float(dot1, sc);
                        sumf_m += d8[s] * exact_float(dot2[s], mn);
                    }
                    const float2 dm4f = __half22float2(dm[m]);
                    acc[m][u] += dm4f.x * sumf_d - dm4f.y * sumf_m;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NU; u++) {
            float s[2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                s[m] = sum_lanes(acc[m][u], WARP_SIZE);
            }
            if (lane == 0) {
                const int qq = us.q[u], t = qq / p.n_used, j = qq % p.n_used;
                float * out = (float *) ((char *) p.dst + t * p.dst_nb2 + j * p.dst_nb1) + r0;
                out[r]      = swiglu(s[0], s[1], p.limit);
            }
        }
    }
}

// rows_dot_q4k_glu for us.n <= MU users.
template <int MU, int NI, bool XS, bool UL>
__device__ __forceinline__ void rows_dot_q4k_glu_any(const legion_expert_mv & p, const char * rows_at,
                                                     const char * rows_at2, int nrows, int r0, const Users & us,
                                                     const block_q8_1 * xs) {
    switch (us.n) {
        case 1: rows_dot_q4k_glu<1, NI, XS, UL>(p, rows_at, rows_at2, nrows, r0, us, xs); break;
        case 2: if constexpr (MU >= 2) { rows_dot_q4k_glu<2, NI, XS, UL>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        case 3: if constexpr (MU >= 3) { rows_dot_q4k_glu<3, NI, XS, UL>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
        default: if constexpr (MU >= 4) { rows_dot_q4k_glu<4, NI, XS, UL>(p, rows_at, rows_at2, nrows, r0, us, xs); } break;
    }
}

// Whether pair `pair` (expert `e`, in range) is one this launch reads in place over the link: a
// host expert of `part` without a staging slot (`link_ctas`'s workers take those).
__device__ __forceinline__ bool is_link(const legion_expert_mv & p, int pair, int e) {
    return e >= p.vram_below && in_part(p, e) && p.part != 4 &&
           (p.stage == nullptr || ((const int32_t *) p.stage)[pair] < 0);
}

// Host experts (and, with vram_kernel 0, VRAM ones): a tile (rows tx * rows_per_cta.. of the
// expert of `pair`; both matrices' rows with SwiGLU), all its loads in flight. NI > 0
// (`vram_pf`): an expert in VRAM (cache or staging slot) is read in place instead, each lane's
// loads up front (rows_dot), in the same order: the same bits. MU: the users a CTA takes at once
// (MAX_USERS; a step's token count when that is 1 or 2, #574: a token picks an expert at most
// once, and with only the NU <= MU dot products compiled the 64-register kernel spills less).
// Each user's sums are the same whatever the chunk: the same bits. `leave_link`: a link pair
// (is_link) is left to the link workers (link_worker).
template <ggml_type type, bool GLU, int DOTS, int NI, int MU>
__device__ __forceinline__ void tile_one(const legion_expert_mv & p, char * tile, int pair, int tx,
                                         bool leave_link = false) {
    constexpr int MAX_LOADS = TILE_MAX / (16 * THREADS);  // 16-byte loads per thread

    const int e = expert_of(p, pair);
    if (e < 0 || e >= p.n_w) {
        // A corrupt or padded id, or another rank's expert: nothing to read. dst's rows stay as
        // they are, or with zero_skipped this CTA's rows of the pair are 0.
        if (p.zero_skipped) {
            const int r0 = tx * p.rows_per_cta;
            const int nrows = min(p.rows_per_cta, p.rows - r0);
            float * out = (float *) ((char *) p.dst + (int64_t) (pair / p.n_used) * p.dst_nb2 +
                                     (int64_t) (pair % p.n_used) * p.dst_nb1) + r0;
            for (int r = threadIdx.x; r < nrows; r += blockDim.x) {
                out[r] = 0.0f;
            }
        }
        return;
    }
    if ((p.vram_kernel != 0 && e < p.vram_below) || !in_part(p, e)) {
        return;  // expert_mv_vram's, or the other stream's
    }
    if (leave_link && is_link(p, pair, e)) {
        return;  // the link workers' (uniform: one pair per CTA)
    }
    // Before any load: a duplicate's loads would cross the link a second time.
    Users us = find_users<MU>(p, pair, e, 0);
    if (us.n == 0) {
        return;  // uniform: every warp sees the same ids
    }
    const int r0      = tx * p.rows_per_cta;
    const int nrows   = min(p.rows_per_cta, p.rows - r0);
    const int64_t off = (int64_t) e * p.w_nb2 + (int64_t) r0 * p.w_nb1;
    const char * src  = p.w + off;
    const char * src2 = GLU ? p.w2 + off : nullptr;
    bool in_vram      = e < p.vram_below;
    if (p.stage != nullptr && e >= p.vram_below) {
        const int s = ((const int32_t *) p.stage)[pair];
        if ((p.part == 3 && s >= 0) || (p.part == 4 && s < 0)) {
            return;  // the other launch's (uniform: one pair per CTA)
        }
        if (s >= 0) {  // staged: the same bytes, from VRAM
            const char * b = p.stage + p.stage_hdr + (int64_t) s * p.stage_slot + (int64_t) r0 * p.w_nb1;
            src            = b + p.stage_w;
            src2           = GLU ? b + p.stage_w2 : nullptr;
            in_vram        = true;
        }
    } else if (p.part == 4) {
        return;  // a VRAM expert, or no table: nothing staged
    }
    // The IQ grids: copied past every early return (uniform: one pair per CTA), only by CTAs with
    // work, and on the tile path while its loads are in flight.
    if constexpr (NI > 0) {
        // In place, no shared memory, for an expert one pair uses (a shared one keeps the tile:
        // its rows are read once for all its users).
        if (in_vram && us.total == 1) {
            if constexpr (grid_in_smem<type>()) {
                load_grid<type>(THREADS);
                __syncthreads();
            }
            rows_dot<type, 1, GLU, NI, xq_split<type, MU>()>(p, src, src2, nrows, r0, us);
            return;
        }
    }
    // A multiple of 16 bytes, 16-aligned: rows of a multiple of 4 bytes, at least 4 rows per
    // tile (all but the last), experts a multiple of 16 bytes apart (expert_mv.rs fits()).
    const int chunks  = nrows * (int) p.w_nb1 / 16;
    const int stride  = chunks;  // the second matrix's rows follow the first's in the tile
    char * tile2      = tile + stride * 16;

    // Every load of the tile in flight at once (exactly the tile's rows: nothing past a row).
    uint4 v[MAX_LOADS];
#pragma unroll
    for (int i = 0; i < MAX_LOADS; i++) {
        const int c = threadIdx.x + i * THREADS;
        if (c < chunks) {
            v[i] = __ldg((const uint4 *) src + c);
        } else if (GLU && c >= stride && c - stride < chunks) {
            v[i] = __ldg((const uint4 *) src2 + (c - stride));
        }
    }
    if constexpr (grid_in_smem<type>()) {
        load_grid<type>(THREADS);  // the __syncthreads below covers it
    }
#pragma unroll
    for (int i = 0; i < MAX_LOADS; i++) {
        const int c = threadIdx.x + i * THREADS;
        if (c < chunks || (GLU && c >= stride && c - stride < chunks)) {
            ((uint4 *) tile)[c] = v[i];
        }
    }
    __syncthreads();
    // The users in chunks of MU, each against the tile read once.
    for (int skip = 0;;) {
        rows_dot_any<type, GLU, DOTS, MU>(p, tile, tile2, nrows, r0, us);
        skip += MU;
        if (skip >= us.total) {
            break;
        }
        us = find_users<MU>(p, pair, e, skip);
    }
}

// A tile per CTA (grid.x: row tiles, grid.y: pairs). NI > 0: VRAM experts read in place with
// each lane's loads up front (`vram_pf`, tile_one).
template <ggml_type type, bool GLU, int DOTS, int NI, int MU = MAX_USERS>
__global__ void __launch_bounds__(THREADS, NI > 0 ? PF_MIN_BLOCKS : 1024 / THREADS) expert_mv_tile(const legion_expert_mv p) {
    extern __shared__ __align__(16) char tile[];
    tile_one<type, GLU, DOTS, NI, MU>(p, tile, blockIdx.y, blockIdx.x);
}

// rows_dot for the K half of a down pair (expert_mv_split): the per-user dot products, as
// rows_dot_any's DOTS 0 picks them.
template <ggml_type type, int MU>
__device__ __forceinline__ void rows_dot_half(const legion_expert_mv & p, const char * rows_at, int nrows, int r0,
                                              const Users & us, int kb0, int kk, int64_t stride) {
    constexpr bool S = xq_split<type, MU>();
    if (MU == 1 || us.n == 1) {
        rows_dot<type, 1, false, 0, S>(p, rows_at, nullptr, nrows, r0, us, nullptr, kb0, kk, stride);
    } else if (MU == 2 || us.n == 2) {
        rows_dot<type, 2, false, 0, S>(p, rows_at, nullptr, nrows, r0, us, nullptr, kb0, kk, stride);
    } else if (us.n <= 4) {
        rows_dot<type, 4, false>(p, rows_at, nullptr, nrows, r0, us, nullptr, kb0, kk, stride);
    } else {
        rows_dot<type, MAX_USERS, false>(p, rows_at, nullptr, nrows, r0, us, nullptr, kb0, kk, stride);
    }
}

// `--glm-pair-split` (#735): expert_mv_tile's tile path (the per-user dot products, every pair
// through the tiles) with pairs that are half of their expert: ids with HALF_BIT. The SwiGLU op
// computes only the rows of this GPU's half (`halves` - 1: rows 0..rows/2 or rows/2..rows; the
// other half of dst's column stays unwritten), with the same code and bits as the whole expert's
// rows; the down op every row over the matching half of K (the SwiGLU half's values), from a tile
// of those half rows read with 8-byte loads (a half row of Maya's IQ2_S / IQ3_XXS down is 8- but
// not 16-byte aligned). The NVLink peer computes the other half, and the exchange adds the two
// partial sums. Pairs without the bit are computed whole, as expert_mv_tile does: the same bits.
template <ggml_type type, bool GLU, int MU>
__device__ __forceinline__ void tile_split(const legion_expert_mv & p, char * tile, int pair, int tx) {
    constexpr int MAX_LOADS = TILE_MAX / (16 * THREADS);
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;

    const int id = expert_of(p, pair);
    if (id < 0) {
        return;
    }
    const bool half = (id & HALF_BIT) != 0;
    const int e     = id & ~HALF_BIT;
    if (e >= p.n_w) {
        return;
    }
    const int side = p.halves - 1;
    const int r0   = tx * p.rows_per_cta;
    if (GLU && half && (r0 < side * (p.rows / 2) || r0 >= (side + 1) * (p.rows / 2))) {
        return;  // the peer's rows (uniform: one tile per CTA)
    }
    // Dedup by the id itself: a token's whole and half pairs of one expert never meet (the
    // assignment gives an expert one role per step).
    Users us = find_users<MU>(p, pair, id, 0);
    if (us.n == 0) {
        return;
    }
    const int nrows  = min(p.rows_per_cta, p.rows - r0);
    const char * src = p.w + (int64_t) e * p.w_nb2 + (int64_t) r0 * p.w_nb1;
    if (!GLU && half) {
        // This GPU's half of each row: kh values from block kb0 on, hb bytes (8-byte aligned,
        // checked on the host), packed in the tile.
        const int kh      = p.k / 2;
        const int64_t hb  = p.w_nb1 / 2;
        const int per_row = (int) (hb / 8);
        const int chunks  = nrows * per_row;  // at most half a whole tile's bytes: MAX_LOADS each
        src += side * hb;
        uint2 v[MAX_LOADS];
#pragma unroll
        for (int i = 0; i < MAX_LOADS; i++) {
            const int c = threadIdx.x + i * THREADS;
            if (c < chunks) {
                v[i] = __ldg((const uint2 *) (src + (int64_t) (c / per_row) * p.w_nb1) + c % per_row);
            }
        }
        if constexpr (grid_in_smem<type>()) {
            load_grid<type>(THREADS);
        }
#pragma unroll
        for (int i = 0; i < MAX_LOADS; i++) {
            const int c = threadIdx.x + i * THREADS;
            if (c < chunks) {
                ((uint2 *) tile)[c] = v[i];
            }
        }
        __syncthreads();
        for (int skip = 0;;) {
            rows_dot_half<type, MU>(p, tile, nrows, r0, us, side * (kh / qk), kh, hb);
            skip += MU;
            if (skip >= us.total) {
                break;
            }
            us = find_users<MU>(p, pair, id, skip);
        }
        return;
    }
    // A whole tile (both matrices' rows with SwiGLU), as tile_one loads it.
    const int chunks  = nrows * (int) p.w_nb1 / 16;
    const char * src2 = GLU ? p.w2 + (int64_t) e * p.w_nb2 + (int64_t) r0 * p.w_nb1 : nullptr;
    char * tile2      = tile + chunks * 16;
    uint4 v[MAX_LOADS];
#pragma unroll
    for (int i = 0; i < MAX_LOADS; i++) {
        const int c = threadIdx.x + i * THREADS;
        if (c < chunks) {
            v[i] = __ldg((const uint4 *) src + c);
        } else if (GLU && c >= chunks && c - chunks < chunks) {
            v[i] = __ldg((const uint4 *) src2 + (c - chunks));
        }
    }
    if constexpr (grid_in_smem<type>()) {
        load_grid<type>(THREADS);
    }
#pragma unroll
    for (int i = 0; i < MAX_LOADS; i++) {
        const int c = threadIdx.x + i * THREADS;
        if (c < chunks || (GLU && c >= chunks && c - chunks < chunks)) {
            ((uint4 *) tile)[c] = v[i];
        }
    }
    __syncthreads();
    for (int skip = 0;;) {
        rows_dot_any<type, GLU, 0, MU>(p, tile, tile2, nrows, r0, us);
        skip += MU;
        if (skip >= us.total) {
            break;
        }
        us = find_users<MU>(p, pair, id, skip);
    }
}

template <ggml_type type, bool GLU, int MU>
__global__ void __launch_bounds__(THREADS, 1024 / THREADS) expert_mv_split(const legion_expert_mv p) {
    extern __shared__ __align__(16) char tile[];
    tile_split<type, GLU, MU>(p, tile, blockIdx.y, blockIdx.x);
}

// expert_mv_tile_piped's CTAs per SM: registers for the next tile's loads (MAX_LOADS x 16 bytes a
// thread) beside the dot products' own, which spill at expert_mv_tile's 64 (8 CTAs per SM).
constexpr int PIPE_MIN_BLOCKS = 4;

// expert_mv_tile_piped's CTAs per SM for one instance: PIPE_MIN_BLOCKS (128 registers), or 3
// (168) for the instances that spill at 128 with sm_70 ptxas (CUDA 12.4, #816 round 7): 8 users
// (MAX_USERS) of every type but the per-user dot products of a single matrix (DOTS 0, !GLU), and
// the SwiGLU pairs of Q4_K and Q5_K decoded once at 4 users (Q4_K's 4-token gate/up spilled
// 160 bytes, 114 -> 96 us at 3 CTAs, round 6). All of them are spill-free at 3. Only the
// register budget moves: the same instructions in the same order, so the same output bits.
template <ggml_type type, bool GLU, int DOTS, int MU, bool UL>
__host__ __device__ constexpr int pipe_min_blocks() {
    if (MU > 4) {
        return GLU || DOTS == 1 ? 3 : PIPE_MIN_BLOCKS;
    }
    if (MU == 4 && GLU && DOTS == 1 && !UL && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
        return 3;  // the per-unit activation form (UL) does not spill at 128
    }
    return PIPE_MIN_BLOCKS;
}

// tiles_piped's 16-byte loads a thread with rows_dot_q4k_glu: Qwen3.8-Flash-Next's gate/up tile
// (2 x 4 rows of 1440 bytes, 11520) in 6, not TILE_MAX's 8 (8 registers more for the dot products).
constexpr int Q4K_LOADS = 6;

// tiles_piped's HB for rows_dot_q4k_glu (#816 round 6): (unit, user) pairs of activations a lane
// holds in registers over a pipe group's tiles (5 units at 1 user, 3 at 2, 2 at 3, 1 at 4), and
// the CTAs per SM that kernel is compiled for (168 registers, no spill with sm_70 ptxas). On V100
// against the unheld kernel (3 CTAs at 4 users, round 7) 75.4 -> 68.3 us at 3 tokens and
// 96.4 -> 85.4 at 4 (round 8). Larger holds spilled at 3 CTAs and were no faster; 4 CTAs
// spilled with any hold. On by default (legion_expert_mv::q4k_hold; --no-expert-mv-q4k-hold).
constexpr int Q4K_HOLD        = 6;
constexpr int Q4K_HOLD_BLOCKS = 3;

// tile_one over `pipe_tiles` consecutive tiles (group `gx`) of the expert of `pair`, software
// pipelined: while the CTA computes tile i from shared memory, the loads of tile i + 1 are in flight
// into registers; then a barrier, the registers go to shared memory, and tile i + 1 is computed.
// expert_mv_tile's CTAs load a tile and then compute it, and an SM's CTAs, launched together, do
// both together: HBM idles while they compute (decode verify steps at 39-50% of HBM, #816). Each
// tile is computed by rows_dot_any from the same shared-memory layout as tile_one's: the same bits.
// The pair's checks (its expert, its part, its users) are tile_one's, made once per CTA. Not with
// `vram_pf` (the launcher keeps expert_mv_tile for it).
// XS: the users' activation columns are copied to shared memory after the tile (pipe_xs_bytes),
// once per CTA, and the dot products read them there: the same values. In xq every unit of every
// row went to L1 or L2 for them (the tiles' own loads pass through L1 and push them out), a
// dependent load at the head of each lane's unit loop.
// QNI > 0: Q4_K SwiGLU pairs with 32 * QNI units a row, computed by rows_dot_q4k_glu, and tiles
// of at most Q4K_LOADS 16-byte loads a thread (registers for the dot products); UL: its per-unit
// activations (q4k_units). HB > 0 (with QNI, not XS or UL): when a chunk takes every user, each
// lane's activations of its first units stay in registers over the group's tiles (Q4kHeld, at
// most HB (unit, user) pairs), the same values.
// LD > 0: tiles of at most LD 16-byte loads a thread (legion_expert_mv::half_tiles' kernels).
template <ggml_type type, bool GLU, int DOTS, int MU, bool XS = false, int QNI = 0, bool UL = false, int HB = 0,
          int LD = 0>
__device__ __forceinline__ void tiles_piped(const legion_expert_mv & p, char * tile, int pair, int gx) {
    // 16-byte loads per thread
    constexpr int MAX_LOADS = LD > 0 ? LD : QNI > 0 ? Q4K_LOADS : TILE_MAX / (16 * THREADS);

    const int tiles = (p.rows + p.rows_per_cta - 1) / p.rows_per_cta;
    const int t0    = gx * p.pipe_tiles;
    const int t1    = min(t0 + p.pipe_tiles, tiles);
    const int e     = expert_of(p, pair);
    if (e < 0 || e >= p.n_w) {
        if (p.zero_skipped) {  // tile_one's, for every tile of the group
            const int r0 = t0 * p.rows_per_cta;
            const int nrows = min(t1 * p.rows_per_cta, p.rows) - r0;
            float * out = (float *) ((char *) p.dst + (int64_t) (pair / p.n_used) * p.dst_nb2 +
                                     (int64_t) (pair % p.n_used) * p.dst_nb1) + r0;
            for (int r = threadIdx.x; r < nrows; r += blockDim.x) {
                out[r] = 0.0f;
            }
        }
        return;
    }
    if ((p.vram_kernel != 0 && e < p.vram_below) || !in_part(p, e)) {
        return;
    }
    Users us = find_users<MU>(p, pair, e, 0);
    if (us.n == 0) {
        return;
    }
    const int64_t off = (int64_t) e * p.w_nb2;
    const char * src  = p.w + off;
    const char * src2 = GLU ? p.w2 + off : nullptr;
    if (p.stage != nullptr && e >= p.vram_below) {
        const int s = ((const int32_t *) p.stage)[pair];
        if ((p.part == 3 && s >= 0) || (p.part == 4 && s < 0)) {
            return;
        }
        if (s >= 0) {
            const char * b = p.stage + p.stage_hdr + (int64_t) s * p.stage_slot;
            src            = b + p.stage_w;
            src2           = GLU ? b + p.stage_w2 : nullptr;
        }
    } else if (p.part == 4) {
        return;
    }
    // Tile tx's loads into v (tile_one's: the first matrix's rows, then the second's).
    uint4 v[MAX_LOADS];
    auto load = [&](int tx) {
        const int64_t at = (int64_t) tx * p.rows_per_cta * p.w_nb1;
        const int chunks = min(p.rows_per_cta, p.rows - tx * p.rows_per_cta) * (int) p.w_nb1 / 16;
        const uint4 * a  = (const uint4 *) (src + at);
        const uint4 * a2 = (const uint4 *) (GLU ? src2 + at : src);
#pragma unroll
        for (int i = 0; i < MAX_LOADS; i++) {
            const int c = threadIdx.x + i * THREADS;
            if (c < chunks) {
                v[i] = __ldg(a + c);
            } else if (GLU && c - chunks < chunks) {
                v[i] = __ldg(a2 + (c - chunks));
            }
        }
    };
    load(t0);
    if constexpr (grid_in_smem<type>()) {
        load_grid<type>(THREADS);  // the first __syncthreads below covers it
    }
    // XS: the activation columns of the users in `us` (at most MU), behind the tile.
    const int xcols = p.k / QK8_1;
    block_q8_1 * xs = (block_q8_1 *) (tile + (GLU ? 2 : 1) * p.rows_per_cta * (int) p.w_nb1);
    auto copy_x = [&]() {
        constexpr int XU = 8;  // 4-byte loads in flight per thread and user
        const int words  = xcols * (int) sizeof(block_q8_1) / 4;
        for (int base = threadIdx.x; base < words; base += THREADS * XU) {
            int w[MU][XU];
#pragma unroll
            for (int u = 0; u < MU; u++) {
                const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
                const int * from = (const int *) (p.xq + (int64_t) (t * p.n_x + (p.n_x == 1 ? 0 : j)) * xcols);
#pragma unroll
                for (int i = 0; i < XU; i++) {
                    const int c = base + i * THREADS;
                    if (u < us.n && c < words) {
                        w[u][i] = __ldg(from + c);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < MU; u++) {
#pragma unroll
                for (int i = 0; i < XU; i++) {
                    const int c = base + i * THREADS;
                    if (u < us.n && c < words) {
                        ((int *) (xs + u * xcols))[c] = w[u][i];
                    }
                }
            }
        }
    };
    if constexpr (XS) {
        copy_x();  // the first __syncthreads below covers it
    }
    // The group's tiles in turn, each computed by `compute(tx, r0, nrows, chunks)` from shared memory.
    auto run = [&](auto compute) {
        for (int tx = t0; tx < t1; tx++) {
            const int r0     = tx * p.rows_per_cta;
            const int nrows  = min(p.rows_per_cta, p.rows - r0);
            const int chunks = nrows * (int) p.w_nb1 / 16;
#pragma unroll
            for (int i = 0; i < MAX_LOADS; i++) {
                const int c = threadIdx.x + i * THREADS;
                if (c < chunks || (GLU && c - chunks < chunks)) {
                    ((uint4 *) tile)[c] = v[i];
                }
            }
            __syncthreads();
            if (tx + 1 < t1) {
                load(tx + 1);  // in flight while this tile is computed
            }
            compute(tx, r0, nrows, chunks);
            __syncthreads();  // every warp is done with the tile before the next one overwrites it
        }
    };
    if constexpr (QNI > 0 && HB > 0 && !XS && !UL) {
        // Every user in one chunk: each lane's activations of its first units held in registers over
        // the tiles (Q4kHeld), as many units as HB (unit, user) pairs allow.
        if (us.total <= MU) {
            auto held = [&](auto nu) {
                constexpr int NU = decltype(nu)::value;
                constexpr int H  = HB / NU < QNI ? HB / NU : QNI;
                Q4kHeld<NU, H> a;
                q4k_hold<NU, H, XS>(p, us, xs, a);
                run([&](int, int r0, int nrows, int chunks) {
                    rows_dot_q4k_glu<NU, QNI, XS, false, H>(p, tile, tile + chunks * 16, nrows, r0, us, xs, a);
                });
            };
            switch (us.n) {
                case 1: held(std::integral_constant<int, 1>()); break;
                case 2: if constexpr (MU >= 2) { held(std::integral_constant<int, 2>()); } break;
                case 3: if constexpr (MU >= 3) { held(std::integral_constant<int, 3>()); } break;
                default: if constexpr (MU >= 4) { held(std::integral_constant<int, 4>()); } break;
            }
            return;
        }
    }
    run([&](int tx, int r0, int nrows, int chunks) {
        for (int skip = 0;;) {
            if constexpr (QNI > 0) {
                rows_dot_q4k_glu_any<MU, QNI, XS, UL>(p, tile, tile + chunks * 16, nrows, r0, us, xs);
            } else {
                rows_dot_any<type, GLU, DOTS, MU, XS>(p, tile, tile + chunks * 16, nrows, r0, us, xs);
            }
            skip += MU;
            if (skip >= us.total) {
                break;
            }
            us = find_users<MU>(p, pair, e, skip);
            if constexpr (XS) {  // the next chunk's columns, once every warp is done with these
                __syncthreads();
                copy_x();
                __syncthreads();
            }
        }
        if (us.total > MU && tx + 1 < t1) {
            if constexpr (XS) {
                __syncthreads();
            }
            us = find_users<MU>(p, pair, e, 0);
            if constexpr (XS) {
                copy_x();  // run's __syncthreads after this covers it
            }
        }
    });
}

// Whether expert_mv_tile_piped compiled for MU users copies their activations to shared memory
// (tiles_piped's XS), where it measured faster on V100 (#816): Q5_1 rows at 1-4 users (3 tokens
// 58 -> 50 us), Q8_0 rows at 1-3, Q4_K GLU pairs at 1-2. Q4_K GLU at 3-4 users ran slower with it
// (83 -> 89 us at 3: the copy's registers and shared memory cost the pair tile its occupancy),
// other types unmeasured.
template <ggml_type type, bool GLU, int MU>
__host__ __device__ constexpr bool pipe_xs() {
    return (type == GGML_TYPE_Q5_1 && !GLU && MU <= 4) || (type == GGML_TYPE_Q8_0 && !GLU && MU <= 3) ||
           (type == GGML_TYPE_Q4_K && GLU && MU <= 2);
}

// Pairs (n_used * n_tok) expert_mv_tile_bal takes at most (Qwen3.8-Flash-Next's 4-token steps: 40).
constexpr int BAL_PAIRS = 128;

// find_users with the pairs' experts (expert_of's values) from `sl` in shared memory: the same
// users in the same order.
template <int CAP>
__device__ __forceinline__ Users find_users_sl(const int * sl, int n_pairs, int pair, int e, int skip) {
    const int lane = threadIdx.x % WARP_SIZE;
    Users us;
    us.n     = 0;
    us.total = 0;
#pragma unroll
    for (int u = 0; u < MAX_USERS; u++) {
        us.q[u] = pair;
    }
    for (int base = 0; base < n_pairs; base += WARP_SIZE) {
        const int q = base + lane;
        unsigned m  = __ballot_sync(0xffffffff, q < n_pairs && q >= pair && sl[q] == e);
        while (m) {
            const int b = __ffs(m) - 1;
            m &= m - 1;
            if (us.total >= skip && us.n < CAP) {
#pragma unroll
                for (int u = 0; u < MAX_USERS; u++) {
                    if (u == us.n) {
                        us.q[u] = base + b;
                    }
                }
                us.n++;
            }
            us.total++;
        }
    }
    return us;
}

// Whether tiles_piped computes pair q (expert e) past its checks before find_users (zero_skipped
// aside: expert_mv_tile_bal is not launched with it).
__device__ __forceinline__ bool bal_takes(const legion_expert_mv & p, int q, int e) {
    if (e < 0 || e >= p.n_w || (p.vram_kernel != 0 && e < p.vram_below) || !in_part(p, e)) {
        return false;
    }
    if (p.stage != nullptr && e >= p.vram_below) {
        const int s = ((const int32_t *) p.stage)[q];
        return !((p.part == 3 && s >= 0) || (p.part == 4 && s < 0));
    }
    return p.part != 4;
}

// tiles_piped's tiles balanced over a grid of resident CTAs (`--expert-mv-balance`). tiles_piped
// gives each CTA a fixed group of `pipe_tiles` tiles of one pair, so a step's work comes in
// (distinct experts x groups) CTAs whatever the SMs hold: 3 tokens of Flash-Next's Q5_1 down are
// ~20 experts x 20 groups = 400 CTAs on a V100's 320 slots, and the second wave of 80 CTAs runs
// alone, latency bound (one CTA an SM, a tile's loads in flight at a time). Here every CTA first
// lists the pairs tiles_piped would compute (those passing its checks whose expert no earlier pair
// chose, in pair order), and the (pair, tile) items of that list are cut into gridDim.x equal
// contiguous ranges; a CTA computes its range in order, the next item's loads in flight while the
// current one is computed, across expert boundaries too. Each tile is computed as tiles_piped
// computes it (the same users from find_users_sl, the same rows_dot_any from the same
// shared-memory layout): the same bits. Measured on V100 (#814), only the Q4_K gate/up at one
// token (a tile per CTA there) gains, 48.3 -> 42.5 us; the groups of 4 tiles (Q5_1 down, Q8_0,
// Q4_K at 2-4 tokens) lose 5-50%, so launch_piped_mu only sends that one here (the held-activation
// path, HB > 0, measured 70 -> 93 us at 3 tokens and is not launched; kept for a later try).
template <ggml_type type, bool GLU, int DOTS, int MU, bool XS, int QNI = 0, int HB = 0>
__device__ __forceinline__ void tiles_bal(const legion_expert_mv & p, char * tile) {
    static_assert(!grid_in_smem<type>(), "the types with a Unit");
    constexpr int MAX_LOADS = QNI > 0 ? Q4K_LOADS : TILE_MAX / (16 * THREADS);
    __shared__ int sl[BAL_PAIRS];    // expert_of of every pair
    __shared__ int list[BAL_PAIRS];  // the pairs computed, in pair order
    __shared__ int n_list;
    const int n_pairs = p.n_used * p.n_tok;
    for (int q = threadIdx.x; q < n_pairs; q += THREADS) {
        sl[q] = expert_of(p, q);
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        const int lane = threadIdx.x;
        int n          = 0;
        for (int base = 0; base < n_pairs; base += WARP_SIZE) {
            const int q = base + lane;
            bool take   = false;
            if (q < n_pairs) {
                const int e = sl[q];
                take        = bal_takes(p, q, e);
                for (int r = 0; r < q && take; r++) {
                    take = sl[r] != e;
                }
            }
            const unsigned m = __ballot_sync(0xffffffff, take);
            if (take) {
                list[n + __popc(m & ((1u << lane) - 1))] = q;
            }
            n += __popc(m);
        }
        if (lane == 0) {
            n_list = n;
        }
    }
    __syncthreads();
    const int tiles     = (p.rows + p.rows_per_cta - 1) / p.rows_per_cta;
    const int64_t items = (int64_t) n_list * tiles;
    const int i0        = (int) (items * blockIdx.x / gridDim.x);
    const int i1        = (int) (items * (blockIdx.x + 1) / gridDim.x);
    if (i0 >= i1) {
        return;
    }
    // Item i's loads into v (tiles_piped's load for the pair's tile).
    uint4 v[MAX_LOADS];
    auto load = [&](int i) {
        const int q       = list[i / tiles];
        const int tx      = i % tiles;
        const int e       = sl[q];
        const int64_t off = (int64_t) e * p.w_nb2;
        const char * src  = p.w + off;
        const char * src2 = GLU ? p.w2 + off : nullptr;
        if (p.stage != nullptr && e >= p.vram_below) {
            const int s = ((const int32_t *) p.stage)[q];
            if (s >= 0) {
                const char * b = p.stage + p.stage_hdr + (int64_t) s * p.stage_slot;
                src            = b + p.stage_w;
                src2           = GLU ? b + p.stage_w2 : nullptr;
            }
        }
        const int64_t at = (int64_t) tx * p.rows_per_cta * p.w_nb1;
        const int chunks = min(p.rows_per_cta, p.rows - tx * p.rows_per_cta) * (int) p.w_nb1 / 16;
        const uint4 * a  = (const uint4 *) (src + at);
        const uint4 * a2 = (const uint4 *) (GLU ? src2 + at : src);
#pragma unroll
        for (int k = 0; k < MAX_LOADS; k++) {
            const int c = threadIdx.x + k * THREADS;
            if (c < chunks) {
                v[k] = __ldg(a + c);
            } else if (GLU && c - chunks < chunks) {
                v[k] = __ldg(a2 + (c - chunks));
            }
        }
    };
    load(i0);
    const int xcols = p.k / QK8_1;
    block_q8_1 * xs = (block_q8_1 *) (tile + (GLU ? 2 : 1) * p.rows_per_cta * (int) p.w_nb1);
    Users us;
    // tiles_piped's copy_x.
    auto copy_x = [&]() {
        constexpr int XU = 8;
        const int words  = xcols * (int) sizeof(block_q8_1) / 4;
        for (int base = threadIdx.x; base < words; base += THREADS * XU) {
            int w[MU][XU];
#pragma unroll
            for (int u = 0; u < MU; u++) {
                const int q = us.q[u], t = q / p.n_used, j = q % p.n_used;
                const int * from = (const int *) (p.xq + (int64_t) (t * p.n_x + (p.n_x == 1 ? 0 : j)) * xcols);
#pragma unroll
                for (int k = 0; k < XU; k++) {
                    const int c = base + k * THREADS;
                    if (u < us.n && c < words) {
                        w[u][k] = __ldg(from + c);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < MU; u++) {
#pragma unroll
                for (int k = 0; k < XU; k++) {
                    const int c = base + k * THREADS;
                    if (u < us.n && c < words) {
                        ((int *) (xs + u * xcols))[c] = w[u][k];
                    }
                }
            }
        }
    };
    // Item i: its tile to shared memory, the next item's loads issued, compute(r0, nrows, chunks).
    auto item = [&](int i, auto compute) {
        const int r0     = i % tiles * p.rows_per_cta;
        const int nrows  = min(p.rows_per_cta, p.rows - r0);
        const int chunks = nrows * (int) p.w_nb1 / 16;
#pragma unroll
        for (int k = 0; k < MAX_LOADS; k++) {
            const int c = threadIdx.x + k * THREADS;
            if (c < chunks || (GLU && c - chunks < chunks)) {
                ((uint4 *) tile)[c] = v[k];
            }
        }
        __syncthreads();
        if (i + 1 < i1) {
            load(i + 1);  // in flight while this item is computed
        }
        compute(r0, nrows, chunks);
        __syncthreads();  // every warp is done with the tile (and the columns) before the next item
    };
    // The range a pair at a time (its items up to `end`).
    for (int i = i0; i < i1;) {
        const int q   = list[i / tiles];
        const int e   = sl[q];
        const int end = min(i1, (i / tiles + 1) * tiles);
        us            = find_users_sl<MU>(sl, n_pairs, q, e, 0);
        if constexpr (QNI > 0 && HB > 0 && !XS) {
            // tiles_piped's hold: every user in one chunk, each lane's activations of its first
            // units in registers over the pair's items.
            if (us.total <= MU) {
                auto held = [&](auto nu) {
                    constexpr int NU = decltype(nu)::value;
                    constexpr int H  = HB / NU < QNI ? HB / NU : QNI;
                    Q4kHeld<NU, H> a;
                    q4k_hold<NU, H, false>(p, us, xs, a);
                    for (; i < end; i++) {
                        item(i, [&](int r0, int nrows, int chunks) {
                            rows_dot_q4k_glu<NU, QNI, false, false, H>(p, tile, tile + chunks * 16, nrows, r0, us, xs, a);
                        });
                    }
                };
                switch (us.n) {
                    case 1: held(std::integral_constant<int, 1>()); break;
                    case 2: if constexpr (MU >= 2) { held(std::integral_constant<int, 2>()); } break;
                    case 3: if constexpr (MU >= 3) { held(std::integral_constant<int, 3>()); } break;
                    default: if constexpr (MU >= 4) { held(std::integral_constant<int, 4>()); } break;
                }
                if (i == end) {
                    continue;
                }
            }
        }
        if constexpr (XS) {
            copy_x();  // the last item's closing __syncthreads covers the old columns' readers
        }
        for (; i < end; i++) {
            item(i, [&](int r0, int nrows, int chunks) {
                for (int skip = 0;;) {
                    if constexpr (QNI > 0) {
                        rows_dot_q4k_glu_any<MU, QNI, XS, false>(p, tile, tile + chunks * 16, nrows, r0, us, xs);
                    } else {
                        rows_dot_any<type, GLU, DOTS, MU, XS>(p, tile, tile + chunks * 16, nrows, r0, us, xs);
                    }
                    skip += MU;
                    if (skip >= us.total) {
                        break;
                    }
                    us = find_users_sl<MU>(sl, n_pairs, q, e, skip);
                    if constexpr (XS) {
                        __syncthreads();
                        copy_x();
                        __syncthreads();
                    }
                }
                if (us.total > MU && i + 1 < end) {  // tiles_piped's: the first chunk again
                    if constexpr (XS) {
                        __syncthreads();
                    }
                    us = find_users_sl<MU>(sl, n_pairs, q, e, 0);
                    if constexpr (XS) {
                        copy_x();  // item's __syncthreads after this covers it
                    }
                }
            });
        }
    }
}

// QNI, HB: tiles_piped's (no per-unit activations); MB: the CTAs per SM it is compiled for.
template <ggml_type type, bool GLU, int DOTS, int MU, int QNI = 0, int HB = 0,
          int MB = pipe_min_blocks<type, GLU, DOTS, MU, false>()>
__global__ void __launch_bounds__(THREADS, MB) expert_mv_tile_bal(const legion_expert_mv p) {
    extern __shared__ __align__(16) char tile[];
    tiles_bal<type, GLU, DOTS, MU, pipe_xs<type, GLU, MU>(), QNI, HB>(p, tile);
}

// Shared memory expert_mv_tile_piped takes past the tile for MU users of `k`-wide columns.
template <ggml_type type, bool GLU, int MU>
int pipe_xs_bytes(int k) {
    return pipe_xs<type, GLU, MU>() ? MU * (k / QK8_1) * (int) sizeof(block_q8_1) : 0;
}

// A group of `pipe_tiles` tiles per CTA (grid.x: the groups, grid.y: pairs). QNI, UL, HB:
// tiles_piped's; MB: the CTAs per SM it is compiled for (pipe_min_blocks unless given); LD:
// tiles_piped's.
template <ggml_type type, bool GLU, int DOTS, int MU = MAX_USERS, int QNI = 0, bool UL = false, int HB = 0,
          int MB = pipe_min_blocks<type, GLU, DOTS, MU, UL>(), int LD = 0>
__global__ void __launch_bounds__(THREADS, MB) expert_mv_tile_piped(const legion_expert_mv p) {
    extern __shared__ __align__(16) char tile[];
    tiles_piped<type, GLU, DOTS, MU, pipe_xs<type, GLU, MU>(), QNI, UL, HB, LD>(p, tile, blockIdx.y, blockIdx.x);
}

// half_tiles' loads a thread: tiles of up to 8 KiB (Qwen3.8-Flash-Next's Q5_1 down: 16 rows of
// 480 bytes).
constexpr int HALF_LOADS = 4;

// expert_mv_tile_piped for a single matrix on tiles of half p.rows_per_cta rows, with HALF_LOADS
// loads a thread, in groups of p.pipe_tiles of them: twice as many CTAs as the full tiles take
// (#814). Compiled for the CTAs per SM of the full-tile kernel (pipe_min_blocks): the registers
// hardly drop with the loads (the dot products' dominate), and at 6 or 8 CTAs per SM the kernels
// spilled and ran 1.3-3x slower on V100. Every row is computed by the same lanes from the same
// shared-memory layout (rows_dot*'s row r of a tile is row r0 + r of the expert whatever the
// tile's size): the same bits. Returns false (nothing launched) where the halves do not fit
// (rows_per_cta odd or under 8, a half tile not a 16-byte multiple or over HALF_LOADS loads).
template <ggml_type type, bool GLU, int DOTS, int MU>
bool launch_half(const legion_expert_mv & p, int pairs, int smem_xs, cudaStream_t stream) {
    legion_expert_mv q = p;
    q.rows_per_cta     = p.rows_per_cta / 2;
    const int64_t half = (int64_t) q.rows_per_cta * p.w_nb1;
    if (GLU || p.rows_per_cta % 2 != 0 || q.rows_per_cta < 4 || half % 16 != 0 ||
        half > (int64_t) HALF_LOADS * 16 * THREADS) {
        return false;
    }
    const int tiles  = (p.rows + q.rows_per_cta - 1) / q.rows_per_cta;
    const int groups = (tiles + p.pipe_tiles - 1) / p.pipe_tiles;
    constexpr int MB = pipe_min_blocks<type, GLU, DOTS, MU, false>();
    expert_mv_tile_piped<type, GLU, DOTS, MU, 0, false, 0, MB, HALF_LOADS>
        <<<dim3(groups, pairs), THREADS, (int) half + smem_xs, stream>>>(q);
    return true;
}

// Units a lane takes in rows_dot_q4k_glu (tiles_piped's QNI) for an op, or 0: Q4_K SwiGLU pairs
// whose rows are 32 * 5 units (K = 2560, Qwen3.8-Flash-Next's gate and up) in tiles of at most
// Q4K_LOADS loads a thread, at 3-4 users a CTA (MU). At 1-2 the decoded-once kernel with
// shared-memory activations stays (pipe_xs).
template <ggml_type type, bool GLU, int DOTS, int MU>
int q4k_glu_units(const legion_expert_mv & p, int smem) {
    if constexpr (type == GGML_TYPE_Q4_K && GLU && DOTS == 1 && (MU == 3 || MU == 4)) {
        if (units_per_row<type>(p.k) == 32 * 5 && smem <= Q4K_LOADS * 16 * THREADS) {
            return 5;
        }
    }
    return 0;
}

// Link worker `w` of `link_ctas` (see the top of the file): the link pairs' tiles in pair order,
// rank r's tile tx being item r * tiles + tx, worker w taking the items w, w + link_ctas, ...
// Every warp walks the pairs alike (coalesced id loads, ballots): no shared memory for it, and
// the worker's CTA stays uniform. Pairs repeating an earlier pair's expert count too; their
// tile_one returns at once (the first pair's tiles compute all its users).
template <ggml_type type, bool GLU, int DOTS>
__device__ __forceinline__ void link_worker(const legion_expert_mv & p, char * tile, int w, int tiles) {
    const int lane = threadIdx.x % WARP_SIZE, n_pairs = p.n_used * p.n_tok, W = p.link_ctas;
    int64_t item = 0;  // the first item of the next link pair
    bool first   = true;
    for (int base = 0; base < n_pairs; base += WARP_SIZE) {
        const int q = base + lane;
        bool link   = false;
        if (q < n_pairs) {
            const int e = expert_of(p, q);
            link        = e >= 0 && e < p.n_w && is_link(p, q, e);
        }
        unsigned m = __ballot_sync(0xffffffff, link);
        while (m) {
            const int pair = base + __ffs(m) - 1;
            m &= m - 1;
            for (int tx = (int) (((w - item) % W + W) % W); tx < tiles; tx += W) {
                if (!first) {
                    __syncthreads();  // the last tile's dot products are done with the shared memory
                }
                first = false;
                tile_one<type, GLU, DOTS, 0, MAX_USERS>(p, tile, pair, tx);
            }
            item += tiles;
        }
    }
}

// expert_mv_tile with `link_ctas` workers: grid.y's first rows hold the workers (row-major,
// those past link_ctas idle), the rows after them the tile CTAs, which leave the link pairs.
template <ggml_type type, bool GLU, int DOTS>
__global__ void __launch_bounds__(THREADS, 1024 / THREADS) expert_mv_tile_link(const legion_expert_mv p) {
    extern __shared__ __align__(16) char tile[];
    const int wrows = (p.link_ctas + (int) gridDim.x - 1) / (int) gridDim.x;
    if ((int) blockIdx.y < wrows) {
        const int w = blockIdx.y * gridDim.x + blockIdx.x;
        if (w < p.link_ctas) {
            link_worker<type, GLU, DOTS>(p, tile, w, gridDim.x);
        }
        return;
    }
    tile_one<type, GLU, DOTS, 0, MAX_USERS>(p, tile, blockIdx.y - wrows, blockIdx.x, true);
}

// VRAM experts, read in place as ggml's mul_mat_vec_q_moe reads them (mmvq.cu, MIT): a warp
// per two rows of one (token, slot) pair, lanes along K; no dedup (repeats come from L2). S: split
// activations (xq_split_for).
template <ggml_type type, bool GLU, bool S>
__global__ void __launch_bounds__(THREADS) expert_mv_vram(const legion_expert_mv p) {
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int qi              = ggml_cuda_type_traits<type>::qi;
    constexpr int VDR             = vdr<type>();
    constexpr int lanes_per_block = qi / VDR;
    constexpr int blocks_per_iter = VDR * WARP_SIZE / qi;
    constexpr int M               = GLU ? 2 : 1;
    const int pair = blockIdx.y;
    const int e    = expert_of(p, pair);
    if (e < 0 || e >= p.n_w || e >= p.vram_below || !in_part(p, e)) {
        return;  // expert_mv_tile's, or an id out of range
    }
    const int t = pair / p.n_used, j = pair % p.n_used;
    const int lane = threadIdx.x % WARP_SIZE;
    const int r0   = (blockIdx.x * (THREADS / WARP_SIZE) + threadIdx.x / WARP_SIZE) * VRAM_ROWS;
    if (r0 >= p.rows) {
        return;
    }
    const auto y = xq_col<S>(p, t * p.n_x + (p.n_x == 1 ? 0 : j));
    const char * w[M];
    w[0] = p.w + (int64_t) e * p.w_nb2;
    if constexpr (GLU) {
        w[1] = p.w2 + (int64_t) e * p.w_nb2;
    }
    const int blocks_per_row = p.k / qk;
    float tmp[M][VRAM_ROWS]  = {};
    for (int kbx = lane / lanes_per_block; kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kqs = VDR * (lane % lanes_per_block);
#pragma unroll
        for (int m = 0; m < M; m++) {
#pragma unroll
            for (int i = 0; i < VRAM_ROWS; i++) {
                tmp[m][i] += vec_dot_vram<type>(w[m] + (int64_t) min(r0 + i, p.rows - 1) * p.w_nb1, xq_at(y, kbx * (qk / QK8_1)),
                                           kbx, kqs);
            }
        }
    }
    float * out = (float *) ((char *) p.dst + t * p.dst_nb2 + j * p.dst_nb1);
#pragma unroll
    for (int i = 0; i < VRAM_ROWS; i++) {
        float s[M];
#pragma unroll
        for (int m = 0; m < M; m++) {
            s[m] = warp_reduce_sum(tmp[m][i]);
        }
        if (lane == 0 && r0 + i < p.rows) {
            out[r0 + i] = GLU ? swiglu(s[0], s[M - 1], p.limit) : s[0];
        }
    }
}

// Auto: where the decoded-once kernel was faster on V100 (tests/expert_mv_dot_cuda.rs). The
// K-quants' SwiGLU pairs at every step size (by 16-34% from VRAM, ~1-3% over the host link); Q5_1
// and Q8_0 rows from 3 tokens up (an expert has at most a user per token; at 1-2 both were slower
// decoded once or the same), Q8_0 up to 4 (at 8 and 16, where most experts have one user, 5-8%
// slower). Elsewhere the per-user kernel, unchanged.
template <ggml_type type, bool GLU>
bool auto_once(int n_tok) {
    if constexpr ((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && GLU) {
        return true;
    } else if constexpr (type == GGML_TYPE_Q5_1 && !GLU) {
        return n_tok >= 3;
    } else if constexpr (type == GGML_TYPE_Q8_0 && !GLU) {
        return n_tok >= 3 && n_tok <= 4;
    } else {
        return false;
    }
}

// NI > 0: VRAM experts read in place (`vram_pf`) for NI units per lane.
template <ggml_type type, bool GLU, int DOTS, int NI, int MU = MAX_USERS>
cudaError_t launch_tile(const legion_expert_mv & p, int tiles, int pairs, int smem, cudaStream_t stream) {
    if constexpr (NI == 0 && MU == MAX_USERS && has_unit<type>()) {
        if (p.link_ctas > 0) {  // the link workers' rows first (expert_mv_tile_link)
            if (smem > 48 * 1024) {
                const cudaError_t e = cudaFuncSetAttribute(expert_mv_tile_link<type, GLU, DOTS>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                if (e != cudaSuccess) {
                    return e;
                }
            }
            const int wrows = (p.link_ctas + tiles - 1) / tiles;
            expert_mv_tile_link<type, GLU, DOTS><<<dim3(tiles, pairs + wrows), THREADS, smem, stream>>>(p);
            return cudaSuccess;
        }
    }
    if (smem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(expert_mv_tile<type, GLU, DOTS, NI, MU>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) {
            return e;
        }
    }
    expert_mv_tile<type, GLU, DOTS, NI, MU><<<dim3(tiles, pairs), THREADS, smem, stream>>>(p);
    return cudaSuccess;
}

// expert_mv_tile_bal on as many CTAs as the GPU's SMs hold at once (its occupancy at `smem`),
// kept per instance for the last device and size asked.
template <ggml_type type, bool GLU, int DOTS, int MU, int QNI = 0, int HB = 0,
          int MB = pipe_min_blocks<type, GLU, DOTS, MU, false>()>
cudaError_t launch_bal(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    static std::atomic<uint64_t> last{0};  // device + 1 << 48 | smem << 16 | CTAs
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) {
        return e;
    }
    const uint64_t key = (uint64_t) (dev + 1) << 48 | (uint64_t) smem << 16;
    uint64_t got       = last.load(std::memory_order_relaxed);
    if ((got & ~(uint64_t) 0xFFFF) != key) {
        if (smem > 48 * 1024) {
            e = cudaFuncSetAttribute(expert_mv_tile_bal<type, GLU, DOTS, MU, QNI, HB, MB>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) {
                return e;
            }
        }
        int sms = 0, per_sm = 0;
        e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (e == cudaSuccess) {
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, expert_mv_tile_bal<type, GLU, DOTS, MU, QNI, HB, MB>,
                                                              THREADS, smem);
        }
        if (e != cudaSuccess) {
            return e;
        }
        got = key | (uint64_t) std::min(std::max(sms * per_sm, 1), 0xFFFF);
        last.store(got, std::memory_order_relaxed);
    }
    expert_mv_tile_bal<type, GLU, DOTS, MU, QNI, HB, MB><<<(int) (got & 0xFFFF), THREADS, smem, stream>>>(p);
    return cudaSuccess;
}

// expert_mv_tile_piped over `tiles` tiles in groups of p.pipe_tiles, compiled for MU users.
template <ggml_type type, bool GLU, int DOTS, int MU>
cudaError_t launch_piped_mu(const legion_expert_mv & p, int tiles, int pairs, int smem, cudaStream_t stream) {
    const int groups = (tiles + p.pipe_tiles - 1) / p.pipe_tiles;
    if constexpr (type == GGML_TYPE_Q4_K && GLU && DOTS == 1 && (MU == 3 || MU == 4)) {
        if (q4k_glu_units<type, GLU, DOTS, MU>(p, smem) == 5) {  // no pipe_xs at these MU
            if (p.q4k_units != 0) {
                expert_mv_tile_piped<type, GLU, DOTS, MU, 5, true><<<dim3(groups, pairs), THREADS, smem, stream>>>(p);
            } else if (p.q4k_hold != 0) {
                expert_mv_tile_piped<type, GLU, DOTS, MU, 5, false, Q4K_HOLD, Q4K_HOLD_BLOCKS>
                    <<<dim3(groups, pairs), THREADS, smem, stream>>>(p);
            } else {
                expert_mv_tile_piped<type, GLU, DOTS, MU, 5><<<dim3(groups, pairs), THREADS, smem, stream>>>(p);
            }
            return cudaSuccess;
        }
    }
    // Measured on V100 (#814): half tiles win for the Q5_1 down at 1-3 tokens (Rust's
    // half_tiles_for picks the steps and, at one token, 8 half tiles a CTA); its 4-token steps and
    // the Q8_0 down got slower.
    if constexpr (type == GGML_TYPE_Q5_1 && !GLU && MU <= 3) {
        if (p.half_tiles != 0 && launch_half<type, GLU, DOTS, MU>(p, pairs, pipe_xs_bytes<type, GLU, MU>(p.k), stream)) {
            return cudaSuccess;
        }
    }
    smem += pipe_xs_bytes<type, GLU, MU>(p.k);
    // Measured on V100 (#814): only the Q4_K gate/up at one token (a tile per CTA there) gains;
    // the Q5_1 and Q8_0 groups of 4 tiles and the 2-token Q4_K lose to their own grids.
    if constexpr (type == GGML_TYPE_Q4_K && GLU && MU == 1) {
        if (p.balance != 0 && p.zero_skipped == 0 && p.n_used * p.n_tok <= BAL_PAIRS) {
            return launch_bal<type, GLU, DOTS, MU>(p, smem, stream);
        }
    }
    if (smem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(expert_mv_tile_piped<type, GLU, DOTS, MU>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) {
            return e;
        }
    }
    expert_mv_tile_piped<type, GLU, DOTS, MU><<<dim3(groups, pairs), THREADS, smem, stream>>>(p);
    return cudaSuccess;
}

// launch_piped_mu for the step's tokens: a token picks an expert at most once, so an expert has at
// most n_tok users (a CTA takes them in chunks of MU, each user's sums the same bits whatever the
// chunk).
template <ggml_type type, bool GLU, int DOTS>
cudaError_t launch_piped(const legion_expert_mv & p, int tiles, int pairs, int smem, cudaStream_t stream) {
    switch (p.n_tok) {
        case 1: return launch_piped_mu<type, GLU, DOTS, 1>(p, tiles, pairs, smem, stream);
        case 2: return launch_piped_mu<type, GLU, DOTS, 2>(p, tiles, pairs, smem, stream);
        case 3: return launch_piped_mu<type, GLU, DOTS, 3>(p, tiles, pairs, smem, stream);
        case 4: return launch_piped_mu<type, GLU, DOTS, 4>(p, tiles, pairs, smem, stream);
        default: return launch_piped_mu<type, GLU, DOTS, MAX_USERS>(p, tiles, pairs, smem, stream);
    }
}

// Units a lane computes per row in the unrolled (`vram_pf`) form: the units over the lanes
// sharing a row, rounded up to a compiled count (4, 5 or 8: the surplus ones are predicated off;
// Qwen3.8-Flash-Next's experts all take 5); 0 (the tile path for every pair) past 8, and for
// Q8_0, whose 16-lane rows read in place were slower than tiles on V100 (2560 x 640 down).
template <ggml_type type>
int pf_units(int k) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        return 0;
    }
    const int units = units_per_row<type>(k);
    const int L     = lanes_per_row(units);
    const int ni    = (units + L - 1) / L;
    return ni <= 4 ? 4 : ni == 5 ? 5 : ni <= 8 ? 8 : 0;
}

// launch_tile for `ni` (pf_units, or 0: no pair read in place). Steps of 1 or 2 tokens of the
// per-user formats (GLM-5.3-Flash's Maya quants) take the tile kernel compiled for that many users.
template <ggml_type type, bool GLU, int DOTS>
cudaError_t launch_tile_pf(const legion_expert_mv & p, int ni, int tiles, int pairs, int smem, cudaStream_t stream) {
    switch (ni) {
        case 4: return launch_tile<type, GLU, DOTS, 4>(p, tiles, pairs, smem, stream);
        case 5: return launch_tile<type, GLU, DOTS, 5>(p, tiles, pairs, smem, stream);
        case 8: return launch_tile<type, GLU, DOTS, 8>(p, tiles, pairs, smem, stream);
        default:
            if constexpr (DOTS == 0 && !has_unit<type>()) {
                // Not the IQ types with `vram_pf`: block_q8_1 activations (xq_split_for).
                if (grid_in_smem<type>() && p.vram_pf != 0) {
                    return launch_tile<type, GLU, DOTS, 0>(p, tiles, pairs, smem, stream);
                }
                if (p.n_tok == 1) {
                    return launch_tile<type, GLU, DOTS, 0, 1>(p, tiles, pairs, smem, stream);
                }
                if (p.n_tok == 2) {
                    return launch_tile<type, GLU, DOTS, 0, 2>(p, tiles, pairs, smem, stream);
                }
            }
            return launch_tile<type, GLU, DOTS, 0>(p, tiles, pairs, smem, stream);
    }
}

// The `steps & 2` part of legion_expert_mv_launch (dispatch.cu launches quantize_x before it).
template <ggml_type type, bool GLU>
int launch(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    const int pairs = p.n_used * p.n_tok;
    // The tile kernel first: its link reads take longest.
    const int tiles = (p.rows + p.rows_per_cta - 1) / p.rows_per_cta;
    if (p.halves != 0) {
        // `--glm-pair-split`: every pair through expert_mv_split (the Maya quants' per-user dot
        // products only), the users per CTA as launch_tile_pf picks them without `vram_pf`.
        if constexpr (has_unit<type>()) {
            return cudaErrorInvalidValue;
        } else {
            const dim3 grid(tiles, pairs);
            if (p.n_tok == 1) {
                expert_mv_split<type, GLU, 1><<<grid, THREADS, smem, stream>>>(p);
            } else if (p.n_tok == 2) {
                expert_mv_split<type, GLU, 2><<<grid, THREADS, smem, stream>>>(p);
            } else {
                expert_mv_split<type, GLU, MAX_USERS><<<grid, THREADS, smem, stream>>>(p);
            }
            return cudaGetLastError();
        }
    }
    if (!(p.part == 1 && p.vram_kernel != 0)) {
        // `vram_pf`: something in VRAM to read in place (cache slots not left to expert_mv_vram,
        // or a staging buffer); not with DOT_SUB16 (an opt-in for comparisons); steps of at most
        // PF_MAX_TOKENS tokens (an MTP k = 2 verify step's 3 were slower in place, after #288).
        const bool vram = (p.vram_below > 0 && p.vram_kernel == 0) || p.stage != nullptr;
        const bool pf   = p.vram_pf && vram && p.dot != DOT_SUB16 && p.n_tok <= PF_MAX_TOKENS;
        const int ni    = pf ? pf_units<type>(p.k) : 0;
        cudaError_t e;
        if constexpr (!has_unit<type>()) {
            e = launch_tile_pf<type, GLU, 0>(p, ni, tiles, pairs, smem, stream);
        } else if (p.pipe_tiles > 0 && ni == 0 && p.dot != DOT_SUB16 && p.link_ctas == 0) {
            // The same dot products as below, `pipe_tiles` tiles per CTA (not with link workers,
            // which only expert_mv_tile_link has).
            if (p.dot == DOT_ONCE || (p.dot == DOT_AUTO && auto_once<type, GLU>(p.n_tok))) {
                e = launch_piped<type, GLU, 1>(p, tiles, pairs, smem, stream);
            } else {
                e = launch_piped<type, GLU, 0>(p, tiles, pairs, smem, stream);
            }
        } else if (p.dot == DOT_SUB16) {
            e = launch_tile<type, GLU, 2, 0>(p, tiles, pairs, smem, stream);
        } else if (p.dot == DOT_ONCE || (p.dot == DOT_AUTO && auto_once<type, GLU>(p.n_tok))) {
            e = launch_tile_pf<type, GLU, 1>(p, ni, tiles, pairs, smem, stream);
        } else {
            e = launch_tile_pf<type, GLU, 0>(p, ni, tiles, pairs, smem, stream);
        }
        if (e != cudaSuccess) {
            return e;
        }
    }
    if (p.vram_kernel == 1 && p.vram_below > 0 && p.part != 2 && p.part != 4) {
        const int rpc = THREADS / WARP_SIZE * VRAM_ROWS;
        const dim3 grid((p.rows + rpc - 1) / rpc, pairs);
        bool split = false;
        if constexpr (grid_in_smem<type>()) {
            split = xq_split_for<type>(p);
            if (split) {
                expert_mv_vram<type, GLU, true><<<grid, THREADS, 0, stream>>>(p);
            }
        }
        if (!split) {
            expert_mv_vram<type, GLU, false><<<grid, THREADS, 0, stream>>>(p);
        }
    }
    return cudaGetLastError();
}

template <ggml_type type>
int launch_any(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    return p.w2 ? launch<type, true>(p, smem, stream) : launch<type, false>(p, smem, stream);
}

}  // namespace
