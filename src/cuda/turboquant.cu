// TurboQuant KV caches (#552; Zandieh et al., arXiv 2504.19874, the MSE stage). The format, the
// math and the CPU reference are src/model/core/turboquant.rs; the paged attention's graph side
// is src/model/core/paged/tq.rs.
//
// A cache cell: per kv head, d * bits / 8 bytes of indices (bit planes per group of 32 values:
// `bits` u32 words, bit l of word p = bit p of the index of value 32 * group + l), then the
// heads' norms (F16), padded to 16 bytes. Value i of a head is n * g[idx_i] / sqrt(d) in the
// rotated basis of R = H D / sqrt(d) (H the +-1 Walsh-Hadamard matrix, D the layer's signs).
//
// Kernels:
// - tq_write: a step's K or V rows into the cache, in place of set_rows. One CTA per (head,
//   row): the signs, the norm (32 partial sums, then their sum: the Rust reference's order), an
//   in-place Walsh-Hadamard transform in shared memory, each value's index by its boundaries,
//   the bit planes by warp ballots.
// - tq_rows: cells dequantized to F16 in the original basis (the gather attention path).
// - tq_transform: a query rotated (R q), copied, or an output rotated back (R^T o).
// - tq_decode + tq_combine: flash-decoding over the paged blocks (one query per sequence and
//   token, up to paged::tq::DIRECT_TOK tokens), as paged_attn.cu's paged_decode but reading the
//   indices in place: a key's score is n / sqrt(d) * sum(q_rot * g[idx]), a value's row
//   n / sqrt(d) * g[idx] in the rotated basis.
// - tq_piece_build + tq_stage: a piece of a prefill's blocks dequantized (rotated basis) to F16
//   staging slots, which paged_attn.cu's unchanged F16 prefill kernels read as a tiered pool's
//   pieces (legion_paged_attn::piece_*).
//
// Arithmetic that the Rust reference repeats (norm, transform, scale) uses the _rn intrinsics,
// so --use_fast_math does not contract or approximate it: the writes and the gather path's rows
// are the reference's bits.

#include <cuda_fp16.h>

#include <cstdint>

extern "C" {

// paged::tq / turboquant::Tables.
struct TqTables {
    float g3[8], t3[7];
    float g4[16], t4[15];
};

// turboquant::KernelCache.
struct TqCache {
    const char * data;
    int64_t cell;  // bytes per cell
    int32_t d, n_kv, bits;
    uint32_t seed;
    float rsd;  // 1 / sqrt(d)
};

// paged::tq::TqAttn.
struct TqAttn {
    const float * q;  // contiguous [d, n_head, n_tok, n_seqs], rotated for a TurboQuant K
    const char * k;
    const char * v;
    int64_t k_cell, v_cell;
    int32_t k_code, v_code;  // 0 F16, 1 q8_0, 2 q4_0, 3 tq3, 4 tq4
    const int32_t * table;
    int32_t table_width;
    const int32_t * n_past;
    int32_t d, n_head, n_kv, n_tok, n_seqs;
    float scale, rsd;
    int32_t window, ring;
    uint32_t seed;
    float * out;   // [d, n_head, n_tok, n_seqs]
    float * part;  // decode splits: [d + 2, max_split, n_head, n_tok * n_seqs]
    int32_t max_split;
    int32_t * piece_blk, * piece_ent, * piece_n, * piece_cnt, * src;
    int32_t cap, piece;
    half * k_stage;
    half * v_stage;
};

int legion_tq_write(const TqCache * c, const TqTables * t, const float * x, int64_t x_nb1, const int32_t * cells, int32_t n,
                    void * stream);
int legion_tq_rows(const TqCache * c, const TqTables * t, const int32_t * ids, int32_t n, int32_t per_id, half * out,
                   int64_t out_nb1, void * stream);
int legion_tq_transform(const float * x, int64_t nb1, int64_t nb2, int64_t nb3, int32_t d, int32_t n_head, int32_t n_tok,
                        int32_t n_seqs, int32_t mode, uint32_t seed, float rsd, float * out, void * stream);
int legion_tq_decode(const TqAttn * a, const TqTables * t, void * stream);
int legion_tq_stage(const TqAttn * a, const TqTables * t, void * stream);

}  // extern "C"

namespace {

constexpr int BLK = 64;     // positions per KV block
constexpr int MAX_D = 512;  // largest head

// turboquant::sign.
__device__ __forceinline__ float tq_sign(uint32_t seed, uint32_t j) {
    uint32_t h = (seed * 0x9E3779B1u) ^ (j * 0x85EBCA77u) ^ 0x5bd1e995u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return (h & 1) ? -1.0f : 1.0f;
}

// In-place unnormalized Walsh-Hadamard transform of x[0, n) in shared memory (turboquant::fwht's
// butterflies), by the whole CTA.
__device__ void fwht_smem(float * x, int n) {
    for (int h = 1; h < n; h <<= 1) {
        __syncthreads();
        for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
            const int i = (t / h) * 2 * h + t % h;
            const float a = x[i], b = x[i + h];
            x[i]     = __fadd_rn(a, b);
            x[i + h] = __fsub_rn(a, b);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ const float * centroids(const TqTables & t, int bits) {
    return bits == 3 ? t.g3 : t.g4;
}

// The index of value i (within its group of 32: lane bit `sh`) from a group's bit planes w.
__device__ __forceinline__ int index_of(const uint32_t * w, int bits, int sh) {
    int idx = 0;
#pragma unroll
    for (int p = 0; p < 4; p++) {
        if (p < bits) {
            idx |= ((w[p] >> sh) & 1) << p;
        }
    }
    return idx;
}

// Grid (n_kv, n), 128 threads.
__global__ void __launch_bounds__(128) tq_write(const TqCache c, const TqTables tb, const float * x, int64_t x_nb1,
                                                 const int32_t * cells) {
    __shared__ float v[MAX_D];
    __shared__ float part[32];
    __shared__ float s_inv;
    const int h = blockIdx.x, d = c.d;
    const int64_t cell = cells[blockIdx.y];
    if (cell < 0) {
        return;
    }
    const float * xs = (const float *) ((const char *) x + blockIdx.y * x_nb1) + (int64_t) h * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        v[j] = tq_sign(c.seed, j) * xs[j];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = 0.0f;
        for (int j = threadIdx.x; j < d; j += 32) {
            s = __fadd_rn(s, __fmul_rn(v[j], v[j]));
        }
        part[threadIdx.x] = s;
    }
    __syncthreads();
    const int hb = d * c.bits / 8;
    char * out = (char *) c.data + cell * c.cell;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int l = 0; l < 32; l++) {
            s = __fadd_rn(s, part[l]);
        }
        const float n = __fsqrt_rn(s);
        s_inv = n > 0.0f ? __fdiv_rn(1.0f, n) : 0.0f;
        *(half *) (out + (int64_t) c.n_kv * hb + 2 * h) = __float2half_rn(n);
    }
    fwht_smem(v, d);
    const float * bounds = c.bits == 3 ? tb.t3 : tb.t4;
    const int nb = (1 << c.bits) - 1;
    uint32_t * words = (uint32_t *) (out + (int64_t) h * hb);
    // Whole warps per group of 32 (d and the block size are multiples of 32).
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        const float z = __fmul_rn(v[j], s_inv);
        int idx = 0;
        for (int k = 0; k < nb; k++) {
            idx += z > bounds[k];
        }
        for (int p = 0; p < c.bits; p++) {
            const uint32_t w = __ballot_sync(0xffffffffu, (idx >> p) & 1);
            if (j % 32 == 0) {
                words[(j / 32) * c.bits + p] = w;
            }
        }
    }
}

// Head h of `cell` in the rotated basis into v[0, d) (shared memory), by the CTA.
__device__ __forceinline__ void load_rotated(const TqCache & c, const float * g, const char * cell, int h, float * v) {
    const int hb = c.d * c.bits / 8;
    const float s = __fmul_rn(__half2float(*(const half *) (cell + (int64_t) c.n_kv * hb + 2 * h)), c.rsd);
    const uint32_t * w = (const uint32_t *) (cell + (int64_t) h * hb);
    for (int j = threadIdx.x; j < c.d; j += blockDim.x) {
        v[j] = __fmul_rn(s, g[index_of(w + (j / 32) * c.bits, c.bits, j % 32)]);
    }
}

// Grid (n * per_id, n_kv), 128 threads: cell `ids[i / per_id] * per_id + i % per_id`'s head as F16
// in the original basis, at out row i / per_id, offset (i % per_id) * d * n_kv.
__global__ void __launch_bounds__(128) tq_rows(const TqCache c, const TqTables tb, const int32_t * ids, int per_id,
                                                half * out, int64_t out_nb1) {
    __shared__ float v[MAX_D];
    __shared__ float g[16];
    if (threadIdx.x < 16) {
        g[threadIdx.x] = threadIdx.x < (1 << c.bits) ? centroids(tb, c.bits)[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int i = blockIdx.x, h = blockIdx.y;
    const int id = i / per_id, jj = i % per_id;
    const int64_t cell = (int64_t) ids[id] * per_id + jj;
    load_rotated(c, g, c.data + cell * c.cell, h, v);
    fwht_smem(v, c.d);
    half * o = (half *) ((char *) out + id * out_nb1) + ((int64_t) jj * c.n_kv + h) * c.d;
    for (int j = threadIdx.x; j < c.d; j += blockDim.x) {
        o[j] = __float2half_rn(__fmul_rn(tq_sign(c.seed, j) * v[j], c.rsd));
    }
}

// Grid (n_head, n_tok * n_seqs), 128 threads. mode 0: copy; 1: R x (signs, transform, scale);
// 2: R^T x (transform, signs, scale). x strided (bytes), out contiguous.
__global__ void __launch_bounds__(128) tq_transform(const float * x, int64_t nb1, int64_t nb2, int64_t nb3, int d,
                                                     int n_tok, int mode, uint32_t seed, float rsd, float * out) {
    __shared__ float v[MAX_D];
    const int h = blockIdx.x, r = blockIdx.y;
    const int t = r % n_tok, s = r / n_tok;
    const float * xs = (const float *) ((const char *) x + s * nb3 + t * nb2 + h * nb1);
    float * o = out + ((int64_t) r * gridDim.x + h) * d;
    if (mode == 0) {
        for (int j = threadIdx.x; j < d; j += blockDim.x) {
            o[j] = xs[j];
        }
        return;
    }
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        v[j] = mode == 1 ? tq_sign(seed, j) * xs[j] : xs[j];
    }
    fwht_smem(v, d);
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        o[j] = mode == 1 ? __fmul_rn(v[j], rsd) : __fmul_rn(tq_sign(seed, j) * v[j], rsd);
    }
}

// --- Paged attention ----------------------------------------------------------------------------

__device__ __forceinline__ int first_key(int pos, int window) {
    return window > 0 ? max(0, pos - window + 1) : 0;
}

__device__ __forceinline__ int64_t block_cell(const int32_t * tab, int b, int ring) {
    return (int64_t) tab[ring > 0 ? b % ring : b] * BLK;
}

// Values e .. e + N - 1 (N <= 16, within one 32-value block, aligned) of a cell of a ggml-type
// cache (code 0 F16, 1 q8_0, 2 q4_0), as paged_attn.cu's load_row reads them.
template <int N>
__device__ __forceinline__ void load_std(const char * row, int code, int e, float * out) {
    if (code == 0) {
        const half * h = (const half *) row + e;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = __half2float(h[i]);
        }
    } else if (code == 1) {
        const char * blk = row + (e / 32) * 34;
        const float dd = __half2float(*(const half *) blk);
        const int8_t * q = (const int8_t *) (blk + 2) + e % 32;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = dd * q[i];
        }
    } else {
        const char * blk = row + (e / 32) * 18;
        const float dd = __half2float(*(const half *) blk);
        const int i0 = e % 32;
        const uint8_t * q = (const uint8_t *) (blk + 2) + i0 % 16;
        const int shift = i0 < 16 ? 0 : 4;
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = dd * (float) ((int) ((q[i] >> shift) & 15) - 8);
        }
    }
}

// Bit i of an 8-bit x moved to bit 4 i.
__device__ __forceinline__ uint32_t spread_nibbles(uint32_t x) {
    x = (x | (x << 12)) & 0x000F000Fu;
    x = (x | (x << 6)) & 0x03030303u;
    return (x | (x << 3)) & 0x11111111u;
}

// Values e .. e + N - 1 (within one group of 32) of head h of a TurboQuant cell, rotated basis,
// without the n / sqrt(d) factor (returned).
template <int N>
__device__ __forceinline__ float load_tq(const char * row, int d, int n_kv, int bits, const float * g, int h, int e,
                                         float * out) {
    const int hb = d * bits / 8;
    const uint32_t * w = (const uint32_t *) (row + (int64_t) h * hb) + (e / 32) * bits;
    uint32_t wd[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        wd[p] = p < bits ? w[p] : 0u;
    }
    // Eight values at a time: each plane's byte spread to one bit per nibble, the planes OR-ed
    // into eight 4-bit indices (a few integer ops per value instead of four bit tests).
    const int sh = e % 32;
#pragma unroll
    for (int c = 0; c < (N + 7) / 8; c++) {
        uint32_t pk = 0;
#pragma unroll
        for (int p = 0; p < 4; p++) {
            pk |= spread_nibbles((wd[p] >> (sh + 8 * c)) & 0xFFu) << p;
        }
#pragma unroll
        for (int i = 0; i < 8 && c * 8 + i < N; i++) {
            out[c * 8 + i] = g[(pk >> (4 * i)) & 15u];
        }
    }
    return __half2float(*(const half *) (row + (int64_t) n_kv * hb + 2 * h));
}

// Flash-decoding over the blocks in place, one query per (sequence, token): grid
// (n_kv, n_tok * n_seqs, max_split), 128 threads; paged_attn.cu's paged_decode with the loads
// above.
template <int D, int G>
__global__ void __launch_bounds__(128) tq_decode(const TqAttn p, const TqTables tb) {
    constexpr int NW = 4;
    constexpr int E  = D / 32;
    constexpr int VE = D / (32 * NW);
    const int kvh = blockIdx.x, qi = blockIdx.y, split = blockIdx.z;
    const int seq = qi / p.n_tok, tok = qi % p.n_tok;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    const int len   = p.n_past[seq] + tok + 1;
    const int first = first_key(len - 1, p.window);
    const int bf = first / BLK, be = (len + BLK - 1) / BLK;
    const int nblk = be - bf;
    const int ns   = min(p.max_split, nblk);
    if (split >= ns) {
        return;
    }
    const int b0 = bf + split * nblk / ns, b1 = bf + (split + 1) * nblk / ns;

    __shared__ float s[G][BLK];
    __shared__ float alpha[G], stat_m[G], stat_l[G];
    __shared__ float gk[16], gv[16];
    // A TurboQuant V block's indices, decoded once (eight 4-bit indices per word, value order),
    // and its rows' n / sqrt(d): every thread then reads one word per row instead of the planes.
    __shared__ uint32_t vpk[BLK * D / 8];
    __shared__ float vnrm[BLK];
    const bool k_tq = p.k_code >= 3, v_tq = p.v_code >= 3;
    if (threadIdx.x < 16) {
        const int i = threadIdx.x;
        gk[i] = k_tq && i < (1 << p.k_code) ? centroids(tb, p.k_code)[i] : 0.0f;
        gv[i] = v_tq && i < (1 << p.v_code) ? centroids(tb, p.v_code)[i] : 0.0f;
    }

    float qr[G][E];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const float * qh = p.q + ((int64_t) qi * p.n_head + kvh * G + g) * D;
#pragma unroll
        for (int e = 0; e < E; e++) {
            qr[g][e] = qh[lane * E + e] * p.scale;
        }
    }
    if (threadIdx.x < G) {
        stat_m[threadIdx.x] = -INFINITY;
        stat_l[threadIdx.x] = 0.0f;
    }
    float acc[G][VE];
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
        for (int e = 0; e < VE; e++) {
            acc[g][e] = 0.0f;
        }
    }
    const int32_t * tab = p.table + (int64_t) seq * p.table_width;
    __syncthreads();

    for (int b = b0; b < b1; b++) {
        const int64_t cell0 = block_cell(tab, b, p.ring);
        const int npos = min(BLK, len - b * BLK);
        const int skip = max(0, first - b * BLK);

        if (v_tq) {
            constexpr int NG = D / 32;
            const int bits = p.v_code, hb = D * bits / 8;
            for (int u = threadIdx.x; u < BLK * NG; u += blockDim.x) {
                const int j = u / NG, gi = u % NG;
                if (j < npos && j >= skip) {
                    const uint32_t * w =
                        (const uint32_t *) (p.v + (cell0 + j) * p.v_cell + (int64_t) kvh * hb) + gi * bits;
                    uint32_t wd[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        wd[q] = q < bits ? w[q] : 0u;
                    }
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uint32_t pk = 0;
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            pk |= spread_nibbles((wd[q] >> (8 * c)) & 0xFFu) << q;
                        }
                        vpk[u * 4 + c] = pk;
                    }
                }
            }
            for (int j = threadIdx.x; j < BLK; j += blockDim.x) {
                if (j < npos && j >= skip) {
                    const char * row = p.v + (cell0 + j) * p.v_cell;
                    vnrm[j] = __half2float(*(const half *) (row + (int64_t) p.n_kv * hb + 2 * kvh)) * p.rsd;
                }
            }
        }

        for (int j = warp; j < BLK; j += NW) {
            float dot[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                dot[g] = 0.0f;
            }
            if (j < npos && j >= skip) {
                const char * row = p.k + (cell0 + j) * p.k_cell;
                float kf[E];
                float f = 1.0f;
                if (k_tq) {
                    f = load_tq<E>(row, D, p.n_kv, p.k_code, gk, kvh, lane * E, kf) * p.rsd;
                } else {
                    load_std<E>(row, p.k_code, kvh * D + lane * E, kf);
                }
#pragma unroll
                for (int e = 0; e < E; e++) {
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        dot[g] += qr[g][e] * kf[e];
                    }
                }
#pragma unroll
                for (int g = 0; g < G; g++) {
                    dot[g] *= f;
                }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], o);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < G; g++) {
                    s[g][j] = j < npos && j >= skip ? dot[g] : -INFINITY;
                }
            }
        }
        __syncthreads();

        for (int g = warp; g < G; g += NW) {
            const float a = s[g][lane], c = s[g][lane + 32];
            float mx = fmaxf(a, c);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            const float m_old = stat_m[g];
            const float m_new = fmaxf(m_old, mx);
            const float ea = expf(a - m_new), ec = expf(c - m_new);
            s[g][lane]      = ea;
            s[g][lane + 32] = ec;
            float sum = ea + ec;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            if (lane == 0) {
                const float al = expf(m_old - m_new);
                alpha[g]  = al;
                stat_l[g] = stat_l[g] * al + sum;
                stat_m[g] = m_new;
            }
        }
        __syncthreads();

#pragma unroll
        for (int g = 0; g < G; g++) {
            const float al = alpha[g];
#pragma unroll
            for (int e = 0; e < VE; e++) {
                acc[g][e] *= al;
            }
        }
        const int ve0 = threadIdx.x * VE;
        for (int j = skip; j < npos; j++) {
            float vf[VE];
            float f = 1.0f;
            if (v_tq) {
                const uint32_t pk = vpk[j * (D / 8) + ve0 / 8] >> (4 * (ve0 % 8));
#pragma unroll
                for (int e = 0; e < VE; e++) {
                    vf[e] = gv[(pk >> (4 * e)) & 15u];
                }
                f = vnrm[j];
            } else {
                load_std<VE>(p.v + (cell0 + j) * p.v_cell, p.v_code, kvh * D + ve0, vf);
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                const float w = s[g][j] * f;
#pragma unroll
                for (int e = 0; e < VE; e++) {
                    acc[g][e] += w * vf[e];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < G; g++) {
        const int h = kvh * G + g;
        const int64_t r = (int64_t) qi * p.n_head + h;
        if (ns == 1) {
            const float inv = 1.0f / stat_l[g];
            float * o = p.out + r * D + threadIdx.x * VE;
#pragma unroll
            for (int e = 0; e < VE; e++) {
                o[e] = acc[g][e] * inv;
            }
        } else {
            float * part = p.part + (r * p.max_split + split) * (D + 2);
            if (threadIdx.x == 0) {
                part[0] = stat_m[g];
                part[1] = stat_l[g];
            }
#pragma unroll
            for (int e = 0; e < VE; e++) {
                part[2 + threadIdx.x * VE + e] = acc[g][e];
            }
        }
    }
}

// Grid (n_head, n_tok * n_seqs), D threads: the splits of a query merged.
template <int D>
__global__ void __launch_bounds__(D) tq_combine(const TqAttn p) {
    const int h = blockIdx.x, qi = blockIdx.y;
    const int seq = qi / p.n_tok, tok = qi % p.n_tok;
    const int len = p.n_past[seq] + tok + 1;
    const int bf = first_key(len - 1, p.window) / BLK, be = (len + BLK - 1) / BLK;
    const int ns = min(p.max_split, be - bf);
    if (ns == 1) {
        return;
    }
    const int64_t r = (int64_t) qi * p.n_head + h;
    const float * part = p.part + r * p.max_split * (D + 2);
    float m = -INFINITY;
    for (int s = 0; s < ns; s++) {
        m = fmaxf(m, part[s * (D + 2)]);
    }
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < ns; s++) {
        const float w = expf(part[s * (D + 2)] - m);
        l += w * part[s * (D + 2) + 1];
        o += w * part[s * (D + 2) + 2 + threadIdx.x];
    }
    p.out[r * D + threadIdx.x] = o / l;
}

template <int D, int G>
int decode_launch(const TqAttn & p, const TqTables & t, cudaStream_t st) {
    const int queries = p.n_tok * p.n_seqs;
    tq_decode<D, G><<<dim3(p.n_kv, queries, p.max_split), 128, 0, st>>>(p, t);
    if (p.max_split > 1) {
        tq_combine<D><<<dim3(p.n_head, queries), D, 0, st>>>(p);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int D>
int decode_d(const TqAttn & p, const TqTables & t, cudaStream_t st) {
    switch (p.n_head / p.n_kv) {
        case 1: return decode_launch<D, 1>(p, t, st);
        case 2: return decode_launch<D, 2>(p, t, st);
        case 4: return decode_launch<D, 4>(p, t, st);
        case 6: return decode_launch<D, 6>(p, t, st);
        case 8: return decode_launch<D, 8>(p, t, st);
        case 12: return decode_launch<D, 12>(p, t, st);
        case 16: return decode_launch<D, 16>(p, t, st);
        default: return 1;
    }
}

constexpr int BUILD_THREADS = 1024;

// paged_attn.cu's cta_scan: exclusive prefix sum over the CTA, and the total.
__device__ __forceinline__ int2 cta_scan(int x, int * sw) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int v = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, v, o);
        v += lane >= o ? y : 0;
    }
    if (lane == 31) {
        sw[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        int t = sw[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            t += lane >= o ? y : 0;
        }
        sw[lane] = t;
    }
    __syncthreads();
    const int2 r = make_int2((warp > 0 ? sw[warp - 1] : 0) + v - x, sw[31]);
    __syncthreads();
    return r;
}

// One CTA: piece p.piece's walk entries per sequence, every walked block staged (as
// paged_attn.cu's paged_piece_build with every block in the host tier), through a ring if the
// table is one; src: the pool block of each staging slot.
__global__ void __launch_bounds__(BUILD_THREADS) tq_piece_build(const TqAttn p) {
    __shared__ int sw[32];
    int total = 0;  // blocks so far, over the sequences in order
    for (int seq = 0; seq < p.n_seqs; seq++) {
        const int64_t row = (int64_t) seq * p.table_width;
        const int n_past = p.n_past[seq];
        const int b0 = first_key(n_past, p.window) / BLK;
        int b1 = (n_past + p.n_tok - 1) / BLK;
        if (p.ring == 0) {
            b1 = min(b1, p.table_width - 1);
        }
        int n = 0;
        for (int c = b0; c <= b1; c += BUILD_THREADS) {
            const int b = c + threadIdx.x;
            const bool valid = b <= b1;
            const int2 h = cta_scan(valid, sw);
            const int ord = total + h.x;
            const bool in = valid && ord / p.cap == p.piece;
            const int2 k = cta_scan(in, sw);
            if (in) {
                const int slot = ord - p.piece * p.cap;
                p.piece_blk[row + n + k.x] = b;
                p.piece_ent[row + n + k.x] = -1 - slot;
                p.src[slot] = p.table[row + (p.ring > 0 ? b % p.ring : b)];
            }
            total += h.y;
            n += k.y;
        }
        if (threadIdx.x == 0) {
            p.piece_n[seq] = n;
        }
    }
    if (threadIdx.x == 0) {
        *p.piece_cnt = max(0, min(total - p.piece * p.cap, p.cap));
    }
}

// Grid (cap, n_kv, 2), 256 threads: staging slot blockIdx.x's block, head blockIdx.y, of K
// (z 0) or V (z 1) as F16 rows of d * n_kv: TurboQuant in the rotated basis, ggml types as they
// are.
__global__ void __launch_bounds__(256) tq_stage(const TqAttn p, const TqTables tb) {
    const int slot = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
    if (slot >= *p.piece_cnt) {
        return;
    }
    const int code = z == 0 ? p.k_code : p.v_code;
    const char * pool = z == 0 ? p.k : p.v;
    const int64_t cell_bytes = z == 0 ? p.k_cell : p.v_cell;
    const int d = p.d;
    const int64_t kv_row = (int64_t) d * p.n_kv;
    half * st = (z == 0 ? p.k_stage : p.v_stage) + (int64_t) slot * BLK * kv_row + (int64_t) h * d;
    const char * blk = pool + (int64_t) p.src[slot] * BLK * cell_bytes;
    __shared__ float g[16];
    if (threadIdx.x < 16) {
        g[threadIdx.x] = code >= 3 && threadIdx.x < (1 << code) ? centroids(tb, code)[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    if (code >= 3) {
        const int hb = d * code / 8;
        for (int i = threadIdx.x; i < BLK * d; i += blockDim.x) {
            const int j = i / d, e = i % d;
            const char * row = blk + j * cell_bytes;
            const float n = __half2float(*(const half *) (row + (int64_t) p.n_kv * hb + 2 * h));
            const uint32_t * w = (const uint32_t *) (row + (int64_t) h * hb) + (e / 32) * code;
            st[j * kv_row + e] = __float2half_rn(n * p.rsd * g[index_of(w, code, e % 32)]);
        }
    } else {
        for (int i = threadIdx.x; i < BLK * d; i += blockDim.x) {
            const int j = i / d, e = i % d;
            const char * row = blk + j * cell_bytes;
            float x;
            if (code == 0) {
                st[j * kv_row + e] = ((const half *) row)[(int64_t) h * d + e];
                continue;
            }
            load_std<1>(row, code, h * d + e, &x);
            st[j * kv_row + e] = __float2half_rn(x);
        }
    }
}

}  // namespace

int legion_tq_write(const TqCache * c, const TqTables * t, const float * x, int64_t x_nb1, const int32_t * cells, int32_t n,
                    void * stream) {
    if (c->d > MAX_D || c->d % 32 != 0 || (c->bits != 3 && c->bits != 4)) {
        return 1;
    }
    if (n > 0) {
        tq_write<<<dim3(c->n_kv, n), 128, 0, (cudaStream_t) stream>>>(*c, *t, x, x_nb1, cells);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_tq_rows(const TqCache * c, const TqTables * t, const int32_t * ids, int32_t n, int32_t per_id, half * out,
                   int64_t out_nb1, void * stream) {
    if (c->d > MAX_D || c->d % 32 != 0 || (c->bits != 3 && c->bits != 4)) {
        return 1;
    }
    if (n > 0) {
        tq_rows<<<dim3(n * per_id, c->n_kv), 128, 0, (cudaStream_t) stream>>>(*c, *t, ids, per_id, out, out_nb1);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_tq_transform(const float * x, int64_t nb1, int64_t nb2, int64_t nb3, int32_t d, int32_t n_head, int32_t n_tok,
                        int32_t n_seqs, int32_t mode, uint32_t seed, float rsd, float * out, void * stream) {
    if (d > MAX_D) {
        return 1;
    }
    tq_transform<<<dim3(n_head, n_tok * n_seqs), 128, 0, (cudaStream_t) stream>>>(x, nb1, nb2, nb3, d, n_tok, mode, seed,
                                                                                   rsd, out);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_tq_decode(const TqAttn * a, const TqTables * t, void * stream) {
    if (a->n_kv <= 0 || a->n_head % a->n_kv != 0) {
        return 1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    switch (a->d) {
        case 128: return decode_d<128>(*a, *t, st);
        case 256: return decode_d<256>(*a, *t, st);
        case 512: return decode_d<512>(*a, *t, st);
        default: return 1;
    }
}

int legion_tq_stage(const TqAttn * a, const TqTables * t, void * stream) {
    cudaStream_t st = (cudaStream_t) stream;
    tq_piece_build<<<1, BUILD_THREADS, 0, st>>>(*a);
    tq_stage<<<dim3(a->cap, a->n_kv, 2), 256, 0, st>>>(*a, *t);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
