// qwen4exp's sparse (QSA) attention for big steps: kernel bodies and thin launchers. Shapes,
// grids and the op semantics are in src/model/qwen35/qsa_attn.rs (with the CPU reference).
//
// Layouts (ggml order, dim 0 fastest):
// - vis:       I32 [2, tokens]: each token's visible pools [first, end) in the listing
// - pool mask: F16 [n_pool, tokens]: 0 for visible pools, -inf elsewhere
// - top:       I32 [n_top, tokens] (ggml top_k, any order); score F32 [n_pool, tokens]
// - pool_idxs: I32 [kpool, n_pool] member positions; tail I32 [kpool - 1, tokens]
// - sel:       I32 [1 + cap, tokens]: count, positions ascending, -1
// - q:         F32 [d, n_head, n_tok, n_seqs], any byte strides
// - K, V:      F16, q8_0 or q4_0 pools [d * n_head_kv, cells]; cell = table[s][p / 64] * 64 +
//              p % 64
// - dst:       F32 [d, n_head, n_tok, n_seqs], contiguous
//
// The prompt kernels per architecture live in sm70/qsa_attn.cuh (mma.sync.m8n8k4, attn_kernel),
// sm80/qsa_attn.cuh (mma.sync.m16n8k16, attn_mma_kernel) and sm80/qsa_attn_gb10.cuh
// (attn_gb_kernel: attn_mma_kernel's bits, two CTAs per SM, on compute capability 12.x and
// 8.0-8.7),
// included below; launch_attn picks one per device at run time (use_mma, use_gb). The other
// kernels are portable and stay here.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" {

struct legion_qsa_pool_mask {
    const int32_t * vis;
    int64_t vis_nb1;
    uint16_t * dst;
    int64_t dst_nb1;
    int32_t n_pool, n_tok;
    uint32_t grid[2];
    uint32_t block;
};

struct legion_qsa_select {
    const int32_t * top;
    int64_t top_nb1;
    const float * score;
    int64_t score_nb1;
    const int32_t * pool_idxs;
    int64_t pool_nb1;
    const int32_t * tail;
    int64_t tail_nb1;
    const int32_t * pos;
    int64_t pos_nb0;
    int32_t * dst;
    int64_t dst_nb1;
    int32_t n_top, kpool, row_len, n_tok;
    int32_t sort_n;  // power of two >= n_top, <= 1024
    uint32_t block;
    // select_all (the indexer's early exit, after upstream Strata's #783 series, MIT): each
    // token's visible pools [first, end), at most n_top, instead of top and score; null for
    // the select.
    const int32_t * vis;
    int64_t vis_nb1;
};

struct legion_qsa_attn {
    const float * q;
    int64_t q_nb1, q_nb2, q_nb3;  // bytes
    const half * k;
    const half * v;
    int64_t kv_row;  // elements per cell: d * n_head_kv
    const int32_t * table;
    int32_t table_width;
    const int32_t * sel;
    int32_t sel_row;
    float * dst;
    int32_t head_dim, n_head, n_head_kv, n_tok;
    float scale_log2;  // log2(e) / sqrt(d)
    uint32_t grid[2];  // (tokens, kv heads)
    // Decode and verify steps (a few tokens): each token's selection split into `splits` parts,
    // one CTA each, whose partial softmax states go to `part` and are combined in a fixed order.
    // null: the prompt kernel (one CTA per token and kv head).
    float * part;      // [tokens, kv heads, splits, G, 2 + d]: max, sum, weighted V
    int32_t splits;
    int32_t chain;     // the prompt kernel: hi and lo products in m8n8k4 chains (attn_kernel)
    // The caches' types (#263): 0 F16 (`k` / `v` as halves, `kv_row` apart), 1 q8_0, 2 q4_0
    // (ggml blocks of 32 values: an F16 scale, then 32 int8 or 16 bytes of 4-bit values). K and
    // V are both F16 or both quantized; `k_nb1` / `v_nb1` are the bytes per cell.
    int32_t k_type, v_type;
    int64_t k_nb1, v_nb1;
    // Decode and verify steps (#257): the CTAs each kv head's query heads are split over in
    // decode_fast_kernel; 0 lets the launcher choose. Tests: -1 runs decode_kernel, the
    // reference the fast kernel matches bit for bit.
    int32_t hgroups;
    // The F16 prompt kernel on sm_80+ (#828): 0 the launcher's choice (attn_gb_kernel on compute
    // capability 12.x, attn_mma_kernel on the others), 1 attn_mma_kernel, 2 attn_gb_kernel. Both
    // give the same bits.
    int32_t mma_kernel;
};

int legion_qsa_pool_mask_launch(const legion_qsa_pool_mask * p, void * stream);
int legion_qsa_select_launch(const legion_qsa_select * p, void * stream);
int legion_qsa_attn_launch(const legion_qsa_attn * p, void * stream);

// qsa_scores: the lightning indexer's pool scores for a prompt chunk on Volta tensor cores,
// for indexer heads of 128 dims, a multiple of 4 of them (one launch per 4 heads, each adding to
// the previous launch's sums, so the heads still add in order): score[t][j] = sum_h max(q[t][h] . key[j], 0) * w[t][h] for the
// pools first_t <= j < end_t of vis, -inf for the others (ggml's lightning_indexer with the
// pool mask). Adapted from Strata-AC922 52a35fe (eelgaev/Strata-AC922, MIT): 16 queries per
// CTA share each key read; FP32 values split into hi = fp16(x) and lo = fp16((x - hi) * 2^11),
// three products (lo * lo dropped), hi * hi through a fresh fragment per 16-dim step added with
// FP32 adds (Volta's tensor cores truncate when they accumulate). FP32-level, not bitwise.
struct legion_qsa_scores {
    const float * q;  // [d][n_head][ng]: head h of token t at q + t * q_nb2 + h * q_nb1 (bytes)
    int64_t q_nb1, q_nb2;
    const float * keys;  // [d][n_pool]: pool j at keys + j * k_nb (bytes)
    int64_t k_nb;
    const float * w;  // [n_head][ng]: token t at w + t * w_nb1 (bytes)
    int64_t w_nb1;
    const int32_t * vis;  // [2][ng]
    float * dst;          // [n_pool][ng]: token t at dst + t * dst_nb1 (bytes)
    int64_t dst_nb1;
    int32_t n_pool, ng;
    // Optional, null for none: the pool bias added to the visible scores (ggml's add after the
    // mask: -inf stays -inf), and the key row of each pool (pool j at keys + rows[j] * k_nb:
    // the cache's pooled rows read in place instead of a gathered copy).
    const float * bias;    // [n_pool]
    const int32_t * rows;  // [n_pool]
    int32_t k_f16;         // keys are F16 (the cache's pooled rows), else F32
    int32_t n_head;        // a multiple of 4
};

int legion_qsa_scores_launch(const legion_qsa_scores * p, void * stream);

// qsa_topk: the indexes of the k largest scores of each row (ggml's top_k for the indexer, one
// launch instead of ggml's radix passes). Scores compare as ggml's top_k orders floats (sign-
// flipped bits: -0 below +0, NaN by its bits). The k-th largest key is the threshold; the
// result is every index above it, then the lowest indexes at it, each run ascending. ggml's
// radix top_k picks the same set except among scores tied at the threshold, where it takes
// whichever threads come first (the indexer's pool bias leaves no exact ties among the scores
// that are not -inf; tied -inf pools are dropped by the select).
struct legion_qsa_topk {
    const float * score;  // [n][rows]: row r at score + r * score_nb1 (bytes)
    int64_t score_nb1;
    int32_t * dst;        // [k][rows]
    int64_t dst_nb1;
    int32_t n, k, rows;
};

int legion_qsa_topk_launch(const legion_qsa_topk * p, void * stream);

// qsa_scores_exact: the indexer's pool scores bit for bit as ggml's lightning_indexer (its tile
// kernel for 4 heads of 128 dims, keys rounded to F16, per head an FMA chain over the dims in
// order, then score += relu(qk) * w over the heads in order), with the pool mask added as ggml
// adds it (0 or -inf, from vis instead of an F16 mask tensor) and then the pool bias (ggml's
// add): score[t][j] = (s + m) + bias[j]. More work per thread than ggml's kernel (a 64 x 64
// tile of tokens and pools per CTA, 4 x 4 per thread), and no mask or bias pass of its own.
struct legion_qsa_scores_exact {
    const float * q;  // [128][4][ng]: head h of token t at q + t * q_nb2 + h * q_nb1 (bytes)
    int64_t q_nb1, q_nb2;
    const float * keys;  // [128][n_pool]: pool j at keys + j * k_nb (bytes)
    int64_t k_nb;
    const float * w;  // [4][ng]: token t at w + t * w_nb1 (bytes)
    int64_t w_nb1;
    const int32_t * vis;  // [2][ng]
    const float * bias;   // [n_pool]
    float * dst;          // [n_pool][ng]: token t at dst + t * dst_nb1 (bytes)
    int64_t dst_nb1;
    int32_t n_pool, ng;
};

int legion_qsa_scores_exact_launch(const legion_qsa_scores_exact * p, void * stream);

// 1 when every visible CUDA device has compute capability 7.x (7.0, 7.2 or 7.5: the scores
// kernel's target).
int legion_qsa_scores_sm7x(void);

}  // extern "C"

namespace {

constexpr int BLK = 64;  // positions per KV block (KV_BLOCK on the Rust side)

__global__ void pool_mask_kernel(legion_qsa_pool_mask p) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (j >= p.n_pool) {
        return;
    }
    const int32_t * v = (const int32_t *) ((const char *) p.vis + t * p.vis_nb1);
    uint16_t * row = (uint16_t *) ((char *) p.dst + t * p.dst_nb1);
    row[j] = (j >= v[0] && j < v[1]) ? 0x0000 : 0xfc00;
}

// One CTA per token: the live pools (finite score) sorted by first member, ties by pool index
// (a bitonic sort of (first, pool) pairs in shared memory), their members, then the tail.
__global__ void select_kernel(legion_qsa_select p) {
    __shared__ long long key[1024];
    __shared__ int n_live;
    __shared__ int n_out;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int32_t * top = (const int32_t *) ((const char *) p.top + t * p.top_nb1);
    const float * score = (const float *) ((const char *) p.score + t * p.score_nb1);
    const int dead = INT_MAX;
    if (tid == 0) {
        n_live = 0;
    }
    __syncthreads();
    for (int j = tid; j < p.sort_n; j += blockDim.x) {
        long long k = ((long long) dead << 32) | 0xffffffffll;
        if (p.vis != nullptr) {
            // Every visible pool, as top_k picks them all when they fit its budget (all of
            // them score above -inf).
            const int32_t * v = (const int32_t *) ((const char *) p.vis + t * p.vis_nb1);
            const int pool = v[0] + j;
            if (j < p.n_top && pool < v[1]) {
                const int first =
                    *(const int32_t *) ((const char *) p.pool_idxs + pool * p.pool_nb1);
                k = ((long long) first << 32) | (unsigned) pool;
                atomicAdd(&n_live, 1);
            }
        } else if (j < p.n_top) {
            const int pool = top[j];
            if (score[pool] > -INFINITY) {
                const int first =
                    *(const int32_t *) ((const char *) p.pool_idxs + pool * p.pool_nb1);
                k = ((long long) first << 32) | (unsigned) pool;
                atomicAdd(&n_live, 1);
            }
        }
        key[j] = k;
    }
    __syncthreads();
    for (int size = 2; size <= p.sort_n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < p.sort_n; j += blockDim.x) {
                const int o = j ^ stride;
                if (o > j) {
                    const bool up = (j & size) == 0;
                    const long long a = key[j], b = key[o];
                    if ((a > b) == up) {
                        key[j] = b;
                        key[o] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    int32_t * out = (int32_t *) ((char *) p.dst + t * p.dst_nb1);
    const int n_pool_pos = n_live * p.kpool;
    for (int i = tid; i < n_pool_pos; i += blockDim.x) {
        const int pool = (int) (key[i / p.kpool] & 0xffffffffll);
        const int32_t * m = (const int32_t *) ((const char *) p.pool_idxs + pool * p.pool_nb1);
        out[1 + i] = m[i % p.kpool];
    }
    if (tid == 0) {
        const int32_t * tail = (const int32_t *) ((const char *) p.tail + t * p.tail_nb1);
        const int pos = *(const int32_t *) ((const char *) p.pos + t * p.pos_nb0);
        int n = n_pool_pos;
        // The tail lists the token first, then backwards.
        for (int k = p.kpool - 2; k >= 0; --k) {
            const int x = tail[k];
            if (x <= pos && x >= 0) {
                out[1 + n++] = x;
            }
        }
        out[0] = n;
        n_out = n;
    }
    __syncthreads();
    for (int i = 1 + n_out + tid; i < p.row_len; i += blockDim.x) {
        out[i] = -1;
    }
}

// The prompt kernels' shared shape (sm70/qsa_attn.cuh, sm80/qsa_attn.cuh).
constexpr int CH = 16;        // cells per chunk (32 as Strata-AC922 measured slower here: its
                              // shared memory leaves one CTA per SM)
constexpr int THREADS = 128;  // 4 warps
constexpr int ROWS = 16;      // query rows per CTA (G heads + zero padding)
constexpr int GROUPS = 8;     // q.k dim groups

// ---------------------------------------------------------------------------------------------
// Quantized caches (#263), adapted from Strata-AC922 fdd9b6f (eelgaev/Strata-AC922, MIT,
// Copyright (c) 2026 Niko1221 and the Strata contributors): the Volta prompt kernel reads int8
// K / V with a scale per block of values and runs the same F16 tensor-core products, with
// fewer bytes per cell to fetch. Strata-AC922 keeps its own layout (64-value groups, scales
// apart) and the codes in F16 with the scales applied in F32; here the cache is ggml's q8_0 /
// q4_0 (llama.cpp's --cache-type-k / --cache-type-v, set_rows quantizes), and every 32-value
// block becomes F16 values (scale x code in F32, rounded) where the F16 kernels hold the
// cache's values: the values the gather path reads (get_rows to F32, then F16), so one oracle
// checks both paths.

constexpr int QB = 32;  // values per block
constexpr int QW = 9;   // 32-bit words that cover a q8_0 block (34 bytes) at a 2-byte offset

__device__ __forceinline__ int qblock_bytes(int ty) { return ty == 1 ? 34 : 18; }

// Loads block `b` of a row at `row` (4-byte aligned, an even number of blocks per row and
// head): the words from the 4-byte boundary at or before it (an odd block starts 2 bytes past
// one; an even one's last word reads 2 bytes of the next block). Words past it are 0.
__device__ __forceinline__ void qload(const char * row, int b, int ty, uint32_t (&w)[QW]) {
    const uint32_t * src = reinterpret_cast<const uint32_t *>(row + ((b * qblock_bytes(ty)) & ~3));
    const int nw = ty == 1 ? 9 : 5;
#pragma unroll
    for (int j = 0; j < QW; ++j) {
        w[j] = j < nw ? __ldg(src + j) : 0u;
    }
}

__device__ __forceinline__ uint32_t h2_bits(__half2 h) {
    return *reinterpret_cast<uint32_t *>(&h);
}

// Two biased codes (the bytes of `bytes` that `sel` picks into bytes 0 and 2) as an F16 pair,
// times the scale `d`: 0x64XX is the F16 1024 + XX, so 0x64XX - `bias` is the code exactly, and
// the F16 multiply rounds the exact product once. That is the value scale x code in F32 (exact:
// an 11-bit mantissa times an 8-bit code) rounded to F16 gives, without int-to-float converts.
__device__ __forceinline__ uint32_t qpair(uint32_t bytes, uint32_t sel, __half2 bias, __half2 d) {
    uint32_t x = __byte_perm(bytes, 0x64646464u, sel);
    return h2_bits(__hmul2(__hsub2(*reinterpret_cast<__half2 *>(&x), bias), d));
}

// The 32 values of a TY block (1 q8_0, 2 q4_0) in the words from w (`odd`: it starts 2 bytes
// into w[0]) as F16 pairs, h[i] = values 2i, 2i + 1 as 32 bits: scale x code rounded to F16,
// the values ggml's dequantize (F32) then a cast to F16 give (see `qpair`). All-zero words give
// zeros.
template <int TY>
__device__ __forceinline__ void qunpack_t(const uint32_t * w, bool odd, uint32_t (&h)[16]) {
    const uint32_t sel = odd ? 0x7654u : 0x5432u;
    const __half2 d = __half2half2(__ushort_as_half((unsigned short) (odd ? w[0] >> 16 : w[0] & 0xffffu)));
    if constexpr (TY == 1) {
        // int8 codes c as the bytes c + 128: 1024 + c + 128 - 1152
        const __half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t a = __byte_perm(w[j], w[j + 1], sel) ^ 0x80808080u;  // values 4j .. 4j + 3
            h[2 * j] = qpair(a, 0x4140u, bias, d);
            h[2 * j + 1] = qpair(a, 0x4342u, bias, d);
        }
    } else {
        // 4-bit codes n, values n - 8: 1024 + n - 1032
        const __half2 bias = __float2half2_rn(1032.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // bytes 4j .. 4j + 3: values 4j .. 4j + 3 in the low nibbles, + 16 in the high ones
            const uint32_t a = __byte_perm(w[j], w[j + 1], sel);
            const uint32_t lo = a & 0x0f0f0f0fu, hi = (a >> 4) & 0x0f0f0f0fu;
            h[2 * j] = qpair(lo, 0x4140u, bias, d);
            h[2 * j + 1] = qpair(lo, 0x4342u, bias, d);
            h[8 + 2 * j] = qpair(hi, 0x4140u, bias, d);
            h[8 + 2 * j + 1] = qpair(hi, 0x4342u, bias, d);
        }
    }
}

// A block read by `qload`.
__device__ __forceinline__ void qunpack(const uint32_t (&w)[QW], int ty, bool odd, uint32_t (&h)[16]) {
    if (ty == 1) {
        qunpack_t<1>(w, odd, h);
    } else {
        qunpack_t<2>(w, odd, h);
    }
}

// The prompt kernel's blocks (attn_kernel, quantized): the chunk's K and V rows (HD / 32 blocks
// of 34 or 18 bytes per head: a multiple of LW bytes) are fetched with coalesced LW-byte loads
// and staged in shared memory as they are; then each thread turns run W (one per warp, W < 4)
// of R = HD / 128 consecutive blocks of one row into F16 values, reading the words from the LW
// boundary at or before the run, so the offsets are constants of the warp's code.
template <int LW>
struct LoadPiece {
    using T = uint4;
};
template <>
struct LoadPiece<8> {
    using T = uint2;
};

template <int TY>
__host__ __device__ constexpr int qbytes() {
    return TY == 1 ? 34 : 18;
}

template <int R, int LW>
__host__ __device__ constexpr int qrun_words() {
    return (LW - 2 + R * 34 + LW - 1) / LW * (LW / 4);
}

template <int TY, int R, int W, int LW>
struct QRun {
    static constexpr int START = W * R * qbytes<TY>();
    static constexpr int BASE = START / LW * LW;
    static constexpr int LOADS = (START - BASE + R * qbytes<TY>() + LW - 1) / LW;
    static_assert(LOADS * LW / 4 <= qrun_words<R, LW>(), "the run's words fit");
};

// Run W of the staged row `row` as F16 values into `dst`, the row's first value (16-byte
// aligned).
template <int TY, int R, int W, int LW>
__device__ __forceinline__ void qrun_put(const char * row, __half * dst) {
    using Q = QRun<TY, R, W, LW>;
    uint32_t w[qrun_words<R, LW>()];
#pragma unroll
    for (int k = 0; k < Q::LOADS; ++k) {
        if constexpr (LW == 16) {
            const uint4 x = reinterpret_cast<const uint4 *>(row + Q::BASE)[k];
            w[4 * k] = x.x, w[4 * k + 1] = x.y, w[4 * k + 2] = x.z, w[4 * k + 3] = x.w;
        } else {
            const uint2 x = reinterpret_cast<const uint2 *>(row + Q::BASE)[k];
            w[2 * k] = x.x, w[2 * k + 1] = x.y;
        }
    }
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const int at = Q::START - Q::BASE + k * qbytes<TY>();  // the block's byte in w: even
        uint32_t h[16];
        qunpack_t<TY>(w + at / 4, at % 4 != 0, h);
        uint4 * d = reinterpret_cast<uint4 *>(dst + (W * R + k) * QB);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            d[j] = make_uint4(h[4 * j], h[4 * j + 1], h[4 * j + 2], h[4 * j + 3]);
        }
    }
}

template <int TY, int R, int LW>
__device__ __forceinline__ void qrun_unpack(const char * row, int W, __half * dst) {
    switch (W) {
        case 0: qrun_put<TY, R, 0, LW>(row, dst); break;
        case 1: qrun_put<TY, R, 1, LW>(row, dst); break;
        case 2: qrun_put<TY, R, 2, LW>(row, dst); break;
        default: qrun_put<TY, R, 3, LW>(row, dst); break;
    }
}

// The prompt kernel for sm_70 (Volta) and newer: mma.sync.m8n8k4 (attn_kernel).
#include "sm70/qsa_attn.cuh"

// The prompt kernel for sm_80 and newer: mma.sync.m16n8k16 + cp.async (attn_mma_kernel).
#include "sm80/qsa_attn.cuh"

// The GB10 prompt kernel: attn_mma_kernel's bits, two CTAs per SM (attn_gb_kernel).
#include "sm80/qsa_attn_gb10.cuh"

// --qsa-attn-chain on a GPU running attn_mma_kernel: said once per process.
std::atomic<bool> chain_notice_said{false};

// Whether the current device takes attn_mma_kernel<HD>: compute capability 8.0 or newer, and
// the binary holds code for the kernel built for sm_80+ (a build for older architectures only
// would JIT its PTX, where the MMA is a trap). Decided once per device.
template <int HD>
bool use_mma() {
    static std::atomic<int> state[64];  // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        return false;
    }
    int s = state[dev].load(std::memory_order_acquire);
    if (s == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaFuncAttributes a{};
        const bool code = cudaFuncGetAttributes(&a, attn_mma_kernel<HD>) == cudaSuccess &&
                          a.binaryVersion >= 80 && a.ptxVersion >= 80;
        (void) cudaGetLastError();
        s = major >= 8 && code ? 1 : 2;
        state[dev].store(s, std::memory_order_release);
    }
    return s == 1;
}

// Whether the current device takes attn_gb_kernel<HD> by default: compute capability 12.x (the
// GB10 and the other consumer Blackwell GPUs, 100 KB of shared memory per SM) or Ampere, 8.0 to
// 8.7 (#1212: the sm_86 code took 0.48-0.51x attn_mma_kernel's time at 8K-32K, 4096-token
// steps; not 8.9), running attn_mma_kernel (use_mma). `forced` (legion_qsa_attn::mma_kernel 2)
// takes it on any sm_80+.
template <int HD>
bool use_gb(bool forced) {
    if (!use_mma<HD>()) {
        return false;
    }
    if (forced) {
        return true;
    }
    static std::atomic<int> state[64];  // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        return false;
    }
    int s = state[dev].load(std::memory_order_acquire);
    if (s == 0) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        (void) cudaGetLastError();
        s = major == 12 || (major == 8 && minor <= 7) ? 1 : 2;
        state[dev].store(s, std::memory_order_release);
    }
    return s == 1;
}

// ---------------------------------------------------------------------------------------------
// Decode and verify steps: a few tokens, so one CTA per (token, kv head) would leave the GPU
// idle. Each token's selection is split into p.splits parts of consecutive cells (in selection
// order), one CTA per (token, kv head, part) in chunks of DCH cells: scores with a lane per cell
// (warp w takes heads w, w + 4, ...), an online base-2 softmax per head, P·V with each thread
// owning HD / 128 output dims of every head; all in F32 (q is not rounded, K and V are the
// cache's F16; QUANT: a q8_0 / q4_0 cache's values as F16, as `qunpack` gives them, #263). Each
// part writes its max, sum and weighted V; `combine` adds the parts in part order, so results
// do not depend on scheduling.

constexpr int DCH = 32;        // cells per chunk: one per lane
constexpr int DTHREADS = 128;  // 4 warps
constexpr int DMAXG = 16;      // query heads per kv head

template <int HD, bool QUANT>
__global__ void __launch_bounds__(DTHREADS) decode_kernel(legion_qsa_attn p) {
    constexpr int DPT = HD / DTHREADS;  // output dims per thread
    const int tt = blockIdx.x, kvh = blockIdx.y, part = blockIdx.z;
    const int seq = tt / p.n_tok, t = tt % p.n_tok;
    const int G = p.n_head / p.n_head_kv;
    __shared__ float qs[DMAXG][HD];
    __shared__ float ps[DMAXG][DCH];
    __shared__ int cells[DCH];
    __shared__ float m_s[DMAXG], l_s[DMAXG], a_s[DMAXG];
    const int32_t * row = p.sel + (int64_t) tt * p.sel_row;
    const int count = row[0];
    const int c0 = (int) ((int64_t) part * count / p.splits);
    const int c1 = (int) ((int64_t) (part + 1) * count / p.splits);
    for (int i = threadIdx.x; i < G * HD; i += DTHREADS) {
        const int g = i / HD, d = i % HD;
        const float * qh =
            (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 + (int64_t) (kvh * G + g) * p.q_nb1);
        qs[g][d] = qh[d] * p.scale_log2;
    }
    if (threadIdx.x < DMAXG) {
        m_s[threadIdx.x] = -INFINITY;
        l_s[threadIdx.x] = 0.0f;
    }
    float acc[DMAXG][DPT];
#pragma unroll
    for (int g = 0; g < DMAXG; g++)
#pragma unroll
        for (int j = 0; j < DPT; j++) acc[g][j] = 0.0f;
    const int32_t * tab = p.table + (int64_t) seq * p.table_width;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    for (int base = c0; base < c1; base += DCH) {
        const int nc = min(DCH, c1 - base);
        if (threadIdx.x < DCH) {
            int cell = 0;
            if (threadIdx.x < nc) {
                const int pos = row[1 + base + threadIdx.x];
                cell = tab[pos / BLK] * BLK + pos % BLK;
            }
            cells[threadIdx.x] = cell;
        }
        __syncthreads();
        // scores: lane = cell, warp w = heads w, w + 4, ...
        {
            float sc[DMAXG / 4];
#pragma unroll
            for (int i = 0; i < DMAXG / 4; i++) sc[i] = 0.0f;
            if (QUANT && lane < nc) {
                // The lane's key a block of 32 values at a time.
                const int bytes = qblock_bytes(p.k_type);
                const char * kr = (const char *) p.k + (int64_t) cells[lane] * p.k_nb1 +
                                  (int64_t) kvh * (HD / QB) * bytes;
                for (int b = 0; b < HD / QB; ++b) {
                    uint32_t w[QW], h[16];
                    qload(kr, b, p.k_type, w);
                    qunpack(w, p.k_type, b & 1, h);
#pragma unroll
                    for (int u = 0; u < 16; u++) {
                        const float2 f = __half22float2(*reinterpret_cast<const half2 *>(&h[u]));
                        const int d = b * QB + 2 * u;
#pragma unroll
                        for (int i = 0; i < DMAXG / 4; i++) {
                            const int g = warp + 4 * i;
                            if (g < G) {
                                sc[i] += qs[g][d] * f.x;
                                sc[i] += qs[g][d + 1] * f.y;
                            }
                        }
                    }
                }
            } else if (lane < nc) {
                const half * kr = p.k + (int64_t) cells[lane] * p.kv_row + (int64_t) kvh * HD;
                for (int d = 0; d < HD; d += 8) {
                    const uint4 raw = *(const uint4 *) (kr + d);
                    const half2 * h2 = (const half2 *) &raw;
                    float kf[8];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const float2 f = __half22float2(h2[u]);
                        kf[2 * u] = f.x;
                        kf[2 * u + 1] = f.y;
                    }
#pragma unroll
                    for (int i = 0; i < DMAXG / 4; i++) {
                        const int g = warp + 4 * i;
                        if (g < G) {
#pragma unroll
                            for (int u = 0; u < 8; u++) sc[i] += qs[g][d + u] * kf[u];
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < DMAXG / 4; i++) {
                const int g = warp + 4 * i;
                if (g < G) ps[g][lane] = lane < nc ? sc[i] : -INFINITY;
            }
        }
        __syncthreads();
        // online softmax: thread g < G runs head g's row
        if (threadIdx.x < G) {
            const int g = threadIdx.x;
            float mx = m_s[g];
            for (int c = 0; c < nc; c++) mx = fmaxf(mx, ps[g][c]);
            const float a = m_s[g] == -INFINITY ? 0.0f : exp2f(m_s[g] - mx);
            float sum = 0.0f;
            for (int c = 0; c < DCH; c++) {
                const float e = c < nc ? exp2f(ps[g][c] - mx) : 0.0f;
                ps[g][c] = e;
                sum += e;
            }
            l_s[g] = l_s[g] * a + sum;
            m_s[g] = mx;
            a_s[g] = a;
        }
        __syncthreads();
        // P·V
#pragma unroll
        for (int g = 0; g < DMAXG; g++) {
            if (g < G) {
                const float a = a_s[g];
#pragma unroll
                for (int j = 0; j < DPT; j++) acc[g][j] *= a;
            }
        }
        for (int c = 0; c < nc; c++) {
            float vf[DPT];
            if constexpr (QUANT) {
                // The thread's dims d0 .. d0 + DPT - 1, in one block: its scale and codes.
                const int bytes = qblock_bytes(p.v_type);
                const int d0 = threadIdx.x * DPT, j0 = d0 % QB;
                const unsigned char * blk = (const unsigned char *) p.v +
                                            (int64_t) cells[c] * p.v_nb1 +
                                            ((int64_t) kvh * (HD / QB) + d0 / QB) * bytes;
                const float scale = __half2float(*reinterpret_cast<const half *>(blk));
#pragma unroll
                for (int j = 0; j < DPT; j++) {
                    const int jj = j0 + j;
                    const float x = p.v_type == 1
                                        ? (float) (int8_t) blk[2 + jj]
                                        : (float) ((int) ((blk[2 + (jj & 15)] >> ((jj >> 4) * 4)) & 0xf) - 8);
                    vf[j] = __half2float(__float2half_rn(scale * x));
                }
            } else {
                const half * vr = p.v + (int64_t) cells[c] * p.kv_row + (int64_t) kvh * HD + threadIdx.x * DPT;
#pragma unroll
                for (int j = 0; j < DPT; j++) vf[j] = __half2float(vr[j]);
            }
#pragma unroll
            for (int g = 0; g < DMAXG; g++) {
                if (g < G) {
                    const float pe = ps[g][c];
#pragma unroll
                    for (int j = 0; j < DPT; j++) acc[g][j] += pe * vf[j];
                }
            }
        }
        __syncthreads();
    }
    float * out = p.part + (((int64_t) tt * p.n_head_kv + kvh) * p.splits + part) * G * (2 + HD);
#pragma unroll
    for (int g = 0; g < DMAXG; g++) {
        if (g < G) {
#pragma unroll
            for (int j = 0; j < DPT; j++) out[g * (2 + HD) + 2 + threadIdx.x * DPT + j] = acc[g][j];
        }
    }
    if (threadIdx.x < G) {
        out[threadIdx.x * (2 + HD)] = m_s[threadIdx.x];
        out[threadIdx.x * (2 + HD) + 1] = l_s[threadIdx.x];
    }
}

// The parts of each (token, kv head) in part order: one thread per output dim.
template <int HD>
__global__ void __launch_bounds__(HD) combine_kernel(legion_qsa_attn p) {
    const int tt = blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    const int G = p.n_head / p.n_head_kv;
    const float * base = p.part + ((int64_t) tt * p.n_head_kv + kvh) * p.splits * G * (2 + HD);
    for (int g = 0; g < G; g++) {
        float mx = -INFINITY;
        for (int s = 0; s < p.splits; s++) mx = fmaxf(mx, base[(s * G + g) * (2 + HD)]);
        float l = 0.0f, o = 0.0f;
        for (int s = 0; s < p.splits; s++) {
            const float * q = base + (s * G + g) * (2 + HD);
            if (q[0] == -INFINITY) {
                continue;
            }
            const float w = exp2f(q[0] - mx);
            l += w * q[1];
            o += w * q[2 + d];
        }
        p.dst[((int64_t) tt * p.n_head + kvh * G + g) * HD + d] = l > 0.0f ? o / l : 0.0f;
    }
}

// ---------------------------------------------------------------------------------------------
// The split decode kernel with more of its work in flight (#257): decode_kernel's arithmetic,
// bit for bit. The ideas are those of Strata-AC922's decode attention chunk kernel (250dc42,
// eelgaev/Strata-AC922, MIT; from Strata PR #540): more cells per step, the probabilities
// cell-major so the value loop reads a cell's weights as float4 broadcasts, the V loads of
// several cells issued together. Strata-AC922 sums each score across a warp's lanes (its fewer
// shuffles make that tree cheaper); here a score is one thread's chain over the dims, as in
// decode_kernel, so there are no shuffles to save and the sums keep decode_kernel's order. What
// changes is the schedule:
//
// - a round takes up to HD of the part's cells instead of a 32-cell chunk: the scores of all of
//   the part's chunks at once, their K rows read together;
// - the scores use every thread: the first HD / 2 cells two threads each (half of the heads
//   each), the rest of the round one thread per (cell, head), so a 129-cell part (2,051 cells
//   over 16 parts) does not take a fifth warp for its last cell;
// - the chunks' softmax runs side by side: per head the chunk maxes, the running max and the
//   rescales; the weights per (cell, head); each chunk's sum in its cell order;
// - a kv head's query heads may be split over `hgroups` CTAs (gridDim.z = splits x hgroups),
//   each with its own heads' scores, softmax and P·V: a one-token step launches 2 x 16 CTAs
//   otherwise, under half of a V100's 80 SMs;
// - P·V: each thread one dim of the CTA's heads, the weights cell-major (float4 broadcasts),
//   the V loads of 16 cells issued before their FMAs, the round's V rows prefetched to L2 with
//   the cells.
//
// The chunks stay decode_kernel's (32 cells from the part's first) and every value comes from
// the same expression in the same order: each score a chain over the dims, per chunk the
// running max and the rescale a = exp2(m_old - m_new), the chunk's weight sum in cell order,
// l = l * a + sum, acc = acc * a and then its FMAs in cell order. So the parts, and the
// combined output, are decode_kernel's bit for bit (`cuda_decode_matches_reference`, and
// `cuda_decode_bits` across builds).

// Pulls the 128-byte lines of [`p`, `p` + `bytes`) into L2.
__device__ __forceinline__ void prefetch_l2(const void * p, int bytes) {
    const uintptr_t a = (uintptr_t) p;
    for (uintptr_t x = a & ~(uintptr_t) 127; x < a + bytes; x += 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(x));
    }
}

// Scores of `n` (<= NS) query heads from g_lo for one cell: decode_kernel's chains.
template <int HD, bool QUANT, int NS>
__device__ __forceinline__ void cell_scores(const legion_qsa_attn & p, int kvh, int cell, const float (*qs)[HD],
                                            int g_lo, int n, float (&s)[NS]) {
#pragma unroll
    for (int i = 0; i < NS; i++) s[i] = 0.0f;
    if constexpr (QUANT) {
        const int bytes = qblock_bytes(p.k_type);
        const char * kr = (const char *) p.k + (int64_t) cell * p.k_nb1 + (int64_t) kvh * (HD / QB) * bytes;
        for (int b = 0; b < HD / QB; ++b) {
            uint32_t w[QW], h[16];
            qload(kr, b, p.k_type, w);
            qunpack(w, p.k_type, b & 1, h);
#pragma unroll
            for (int u = 0; u < 16; u++) {
                const float2 f = __half22float2(*reinterpret_cast<const half2 *>(&h[u]));
                const int d = b * QB + 2 * u;
#pragma unroll
                for (int i = 0; i < NS; i++) {
                    if (i < n) {
                        s[i] += qs[g_lo + i][d] * f.x;
                        s[i] += qs[g_lo + i][d + 1] * f.y;
                    }
                }
            }
        }
    } else {
        const half * kr = p.k + (int64_t) cell * p.kv_row + (int64_t) kvh * HD;
#pragma unroll 8
        for (int d = 0; d < HD; d += 8) {
            const uint4 raw = *(const uint4 *) (kr + d);
            const half2 * h2 = (const half2 *) &raw;
            float kf[8];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const float2 f = __half22float2(h2[u]);
                kf[2 * u] = f.x;
                kf[2 * u + 1] = f.y;
            }
#pragma unroll
            for (int i = 0; i < NS; i++) {
                if (i < n) {
#pragma unroll
                    for (int u = 0; u < 8; u++) s[i] += qs[g_lo + i][d + u] * kf[u];
                }
            }
        }
    }
}

template <int HD, bool QUANT, int GM>
__global__ void __launch_bounds__(HD, 2) decode_fast_kernel(legion_qsa_attn p) {
    constexpr int RC = HD;                // cells per round
    constexpr int NCH = RC / DCH;         // chunks per round
    constexpr int PS = (GM + 3) / 4 * 4;  // weights per cell, padded for float4 reads
    constexpr int H1 = (GM + 1) / 2;      // heads per thread in the scores' first pass
    constexpr int VB = 16;                // V loads per batch
    const int tt = blockIdx.x, kvh = blockIdx.y;
    const int hgroups = gridDim.z / p.splits;
    const int part = blockIdx.z / hgroups, hg = blockIdx.z % hgroups;
    const int seq = tt / p.n_tok, tok = tt % p.n_tok;
    const int G = p.n_head / p.n_head_kv;
    const int gh = (G + hgroups - 1) / hgroups;
    const int g0 = hg * gh;
    const int ng = min(gh, G - g0);
    const int t = threadIdx.x;
    __shared__ __align__(16) float qs[PS][HD];
    float (*const pw)[PS] = reinterpret_cast<float (*)[PS]>(&qs[0][0]);
    __shared__ float sc[GM][RC + 1];
    __shared__ int cells[RC];
    __shared__ float cm[GM][NCH];
    __shared__ float ca[GM][NCH];
    __shared__ float cs[GM][NCH];
    __shared__ float m_s[GM], l_s[GM];
    const int32_t * row = p.sel + (int64_t) tt * p.sel_row;
    const int count = row[0];
    const int c0 = (int) ((int64_t) part * count / p.splits);
    const int c1 = (int) ((int64_t) (part + 1) * count / p.splits);
    const int32_t * tab = p.table + (int64_t) seq * p.table_width;
    if (t < GM) {
        m_s[t] = -INFINITY;
        l_s[t] = 0.0f;
    }
    float acc[GM];
#pragma unroll
    for (int g = 0; g < GM; g++) acc[g] = 0.0f;
    // the scores' first pass: thread = (cell, half of the heads), or a cell with one head
    const int nsl = ng >= 2 ? 2 : 1;
    const int pc = HD / nsl;
    const int h1 = (ng + nsl - 1) / nsl;
    for (int base = c0; base < c1; base += RC) {
        const int nr = min(RC, c1 - base);
        const int nch = (nr + DCH - 1) / DCH;
        if (t < nr) {
            const int pos = row[1 + base + t];
            const int cell = tab[pos / BLK] * BLK + pos % BLK;
            cells[t] = cell;
            if constexpr (QUANT) {
                const int bytes = qblock_bytes(p.v_type);
                prefetch_l2((const char *) p.v + (int64_t) cell * p.v_nb1 + (int64_t) kvh * (HD / QB) * bytes,
                            (HD / QB) * bytes);
            } else {
                prefetch_l2(p.v + (int64_t) cell * p.kv_row + (int64_t) kvh * HD, HD * 2);
            }
        }
        for (int i = t; i < ng * HD; i += HD) {
            const int g = i / HD, d = i % HD;
            const float * qh = (const float *) ((const char *) p.q + seq * p.q_nb3 + tok * p.q_nb2 +
                                                (int64_t) (kvh * G + g0 + g) * p.q_nb1);
            qs[g][d] = qh[d] * p.scale_log2;
        }
        __syncthreads();
        // scores: the first min(nr, pc) cells a slice of heads per thread, the rest a head per
        // thread; every (cell, head) decode_kernel's chain
        const int n1 = min(nr, pc);
        {
            const int c = t % pc, glo = (t / pc) * h1;
            if (c < n1 && glo < ng) {
                const int n = min(h1, ng - glo);
                float s[H1];
                cell_scores<HD, QUANT, H1>(p, kvh, cells[c], qs, glo, n, s);
#pragma unroll
                for (int i = 0; i < H1; i++) {
                    if (i < n) sc[glo + i][c] = s[i];
                }
            }
        }
        for (int i = t; i < (nr - n1) * ng; i += HD) {
            const int c = n1 + i / ng, g = i % ng;
            float s[1];
            cell_scores<HD, QUANT, 1>(p, kvh, cells[c], qs, g, 1, s);
            sc[g][c] = s[0];
        }
        __syncthreads();
        // thread = head: each chunk's max (in any order: a max is exact), the running max and
        // the rescale at each chunk's start
        if (t < ng) {
            float m = m_s[t];
            for (int j = 0; j < nch; j++) {
                const int n = min(DCH, nr - j * DCH);
                float mx = m;
#pragma unroll
                for (int c = 0; c < DCH; c++) {
                    if (c < n) mx = fmaxf(mx, sc[t][j * DCH + c]);
                }
                ca[t][j] = m == -INFINITY ? 0.0f : exp2f(m - mx);
                cm[t][j] = mx;
                m = mx;
            }
            m_s[t] = m;
        }
        __syncthreads();
        // weights: thread = cell
        if (t < nr) {
#pragma unroll
            for (int g = 0; g < GM; g++) {
                if (g < ng) pw[t][g] = exp2f(sc[g][t] - cm[g][t / DCH]);
            }
        }
        __syncthreads();
        // each chunk's weight sum in cell order: thread = (head, chunk)
        if (t < ng * nch) {
            const int g = t / nch, j = t % nch;
            const int n = min(DCH, nr - j * DCH);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < DCH; c++) {
                if (c < n) sum += pw[j * DCH + c][g];
            }
            cs[g][j] = sum;
        }
        const auto fold = [&](int c, float v) {
            const float4 * w4 = reinterpret_cast<const float4 *>(pw[c]);
#pragma unroll
            for (int i = 0; i < PS / 4; i++) {
                const float4 w = w4[i];
                const float wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int g = 4 * i + u;
                    if (g < GM && g < ng) acc[g] += wv[u] * v;
                }
            }
        };
        const auto load_v = [&](int c) -> float {
            if constexpr (QUANT) {
                const int bytes = qblock_bytes(p.v_type);
                const unsigned char * blk = (const unsigned char *) p.v + (int64_t) cells[c] * p.v_nb1 +
                                            ((int64_t) kvh * (HD / QB) + t / QB) * bytes;
                const float scale = __half2float(*reinterpret_cast<const half *>(blk));
                const int jj = t % QB;
                const float x = p.v_type == 1 ? (float) (int8_t) blk[2 + jj]
                                              : (float) ((int) ((blk[2 + (jj & 15)] >> ((jj >> 4) * 4)) & 0xf) - 8);
                return __half2float(__float2half_rn(scale * x));
            } else {
                return __half2float(p.v[(int64_t) cells[c] * p.kv_row + (int64_t) kvh * HD + t]);
            }
        };
        for (int j = 0; j < nch; j++) {
#pragma unroll
            for (int g = 0; g < GM; g++) {
                if (g < ng) acc[g] *= ca[g][j];
            }
            const int ce = min(j * DCH + DCH, nr);
            int c = j * DCH;
            for (; c + VB <= ce; c += VB) {
                float v[VB];
#pragma unroll
                for (int u = 0; u < VB; u++) v[u] = load_v(c + u);
#pragma unroll
                for (int u = 0; u < VB; u++) fold(c + u, v[u]);
            }
            for (; c + 8 <= ce; c += 8) {
                float v[8];
#pragma unroll
                for (int u = 0; u < 8; u++) v[u] = load_v(c + u);
#pragma unroll
                for (int u = 0; u < 8; u++) fold(c + u, v[u]);
            }
            for (; c < ce; c++) fold(c, load_v(c));
        }
        __syncthreads();
        if (t < ng) {
            float l = l_s[t];
            for (int j = 0; j < nch; j++) l = l * ca[t][j] + cs[t][j];
            l_s[t] = l;
        }
    }
    __syncthreads();
    float * out = p.part + (((int64_t) tt * p.n_head_kv + kvh) * p.splits + part) * G * (2 + HD);
#pragma unroll
    for (int g = 0; g < GM; g++) {
        if (g < ng) out[(g0 + g) * (2 + HD) + 2 + t] = acc[g];
    }
    if (t < ng) {
        out[(g0 + t) * (2 + HD)] = m_s[t];
        out[(g0 + t) * (2 + HD) + 1] = l_s[t];
    }
}

// combine_kernel's sums with the parts' loads issued together: one CTA per (token, kv head,
// query head), each thread one output dim, every part's max, sum and value loaded before the
// sums, which run in part order as in combine_kernel (the same expressions), so the output is
// combine_kernel's bit for bit. Up to CMAXS parts.
constexpr int CMAXS = 16;

template <int HD>
__global__ void __launch_bounds__(HD) combine_fast_kernel(legion_qsa_attn p) {
    const int tt = blockIdx.x, kvh = blockIdx.y, g = blockIdx.z, d = threadIdx.x;
    const int G = p.n_head / p.n_head_kv;
    const float * base = p.part + ((int64_t) tt * p.n_head_kv + kvh) * p.splits * G * (2 + HD);
    float m[CMAXS], ls[CMAXS], os[CMAXS];
#pragma unroll
    for (int s = 0; s < CMAXS; s++) {
        if (s < p.splits) {
            const float * q = base + (s * G + g) * (2 + HD);
            m[s] = q[0];
            ls[s] = q[1];
            os[s] = q[2 + d];
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < CMAXS; s++) {
        if (s < p.splits) mx = fmaxf(mx, m[s]);
    }
    float l = 0.0f, o = 0.0f;
#pragma unroll
    for (int s = 0; s < CMAXS; s++) {
        if (s < p.splits && m[s] != -INFINITY) {
            const float w = exp2f(m[s] - mx);
            l += w * ls[s];
            o += w * os[s];
        }
    }
    p.dst[((int64_t) tt * p.n_head + kvh * G + g) * HD + d] = l > 0.0f ? o / l : 0.0f;
}

// Multiprocessors of the current device, cached per device.
int device_sms() {
    static std::atomic<int> sms[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return 80;
    }
    int n = sms[dev].load(std::memory_order_relaxed);
    if (n == 0) {
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) {
            (void) cudaGetLastError();
            n = 80;
        }
        sms[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}

// The CTAs a kv head's query heads are split over in decode_fast_kernel: p.hgroups when set,
// else the largest divisor of the heads per kv head that keeps the grid within one CTA per SM.
int decode_hgroups(const legion_qsa_attn & p) {
    const int G = p.n_head / p.n_head_kv;
    if (p.hgroups > 0) {
        return min(p.hgroups, G);
    }
    const int64_t base = (int64_t) p.grid[0] * p.grid[1] * p.splits;
    const int sms = device_sms();
    int hg = 1;
    for (int h = 2; h <= G; h++) {
        if (G % h == 0 && base * h <= sms) {
            hg = h;
        }
    }
    return hg;
}

template <int HD, bool QUANT>
void launch_decode_fast(const legion_qsa_attn & p, int hgroups, cudaStream_t st) {
    const int G = p.n_head / p.n_head_kv;
    const int gh = (G + hgroups - 1) / hgroups;
    const dim3 grid(p.grid[0], p.grid[1], p.splits * hgroups);
    if (gh <= 4) {
        decode_fast_kernel<HD, QUANT, 4><<<grid, HD, 0, st>>>(p);
    } else if (gh <= 6) {
        decode_fast_kernel<HD, QUANT, 6><<<grid, HD, 0, st>>>(p);
    } else if (gh <= 8) {
        decode_fast_kernel<HD, QUANT, 8><<<grid, HD, 0, st>>>(p);
    } else if (gh <= 12) {
        decode_fast_kernel<HD, QUANT, 12><<<grid, HD, 0, st>>>(p);
    } else {
        decode_fast_kernel<HD, QUANT, 16><<<grid, HD, 0, st>>>(p);
    }
}

template <int HD>
int launch_decode(const legion_qsa_attn & p, cudaStream_t st) {
    if (p.hgroups < 0) {
        if (p.k_type != 0) {
            decode_kernel<HD, true><<<dim3(p.grid[0], p.grid[1], p.splits), DTHREADS, 0, st>>>(p);
        } else {
            decode_kernel<HD, false><<<dim3(p.grid[0], p.grid[1], p.splits), DTHREADS, 0, st>>>(p);
        }
    } else {
        const int hg = decode_hgroups(p);
        if (p.k_type != 0) {
            launch_decode_fast<HD, true>(p, hg, st);
        } else {
            launch_decode_fast<HD, false>(p, hg, st);
        }
    }
    if (cudaGetLastError() != cudaSuccess) {
        return 2;
    }
    if (p.hgroups >= 0 && p.splits <= CMAXS) {
        combine_fast_kernel<HD><<<dim3(p.grid[0], p.grid[1], p.n_head / p.n_head_kv), HD, 0, st>>>(p);
    } else {
        combine_kernel<HD><<<dim3(p.grid[0], p.grid[1]), HD, 0, st>>>(p);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// KT / VT: the K / V cache types (0 F16, 1 q8_0, 2 q4_0; both F16 or both quantized).
template <int HD, bool CHAIN, int KT, int VT>
int launch_attn_as(const legion_qsa_attn & p, cudaStream_t st) {
    const int bytes = (int) sizeof(Smem<HD>);
    if (cudaFuncSetAttribute(attn_kernel<HD, CHAIN, KT, VT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) {
        return 2;
    }
    attn_kernel<HD, CHAIN, KT, VT><<<dim3(p.grid[0], p.grid[1]), AT_THREADS, bytes, st>>>(p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int HD, int KT, int VT>
int launch_attn_types(const legion_qsa_attn & p, cudaStream_t st) {
    return p.chain ? launch_attn_as<HD, true, KT, VT>(p, st) : launch_attn_as<HD, false, KT, VT>(p, st);
}

// The indexer scores on Volta's mma.m8n8k4 (PTX: its fragment layout is documented, unlike wmma's).
// A quad pair (QP) of a warp is lanes 4q..4q+3 and 4q+16..4q+19; its thread tq = lane % 4 + 4 * (lane >= 16)
// holds A row tq (k 0..3), B column tq (k 0..3) and eight f32 outputs (see sc_col).
// A warp's 16 pools x 16 queries tile: QP q owns pools 8 * (q & 1) .. + 7 and queries 8 * (q >> 1) .. + 7.
// Each lane reads its pool's key row straight into registers and splits it there: no shared-memory key
// slice and no __syncwarp in the k loop. The products run in wmma 16x16x16's order (four m8n8k4 steps per
// 16 dims, k ascending), so the scores are bitwise those of the earlier wmma kernel and Strata-AC922's.
constexpr int SC_D = 128, SC_HEADS = 4;
constexpr int SC_VQ    = 16;           // queries per CTA
constexpr int SC_ITER  = 4;            // 16-pool tiles per warp
constexpr int SC_GQ    = 16;           // query tiles per CTA group (qsa_scores_kernel's order)
constexpr int SC_QS    = SC_D + 8;     // halves per query row in shared memory
constexpr float SC_LO  = 2048.0f;      // low parts stored x 2^11 (no FP16 underflow)

// x as F16 hi + lo (lo x 2^11). Assumes |x| <= 65504 (F16's largest finite value): the
// indexer's q and keys are normalised activations, far below it. From |x| >= 65520 (round to
// nearest) hi becomes inf and lo -inf, and the cor / acc sums then turn NaN.
__device__ __forceinline__ void sc_split(float x, __half & hi, __half & lo) {
    hi = __float2half_rn(x);
    lo = __float2half_rn((x - __half2float(hi)) * SC_LO);
}

// four floats as the hi and lo A operands (two F16 per register, k ascending)
// Dims 4c .. 4c + 3 of a key row, F32 or F16 (converted exactly, as ggml's get_rows did)
template <bool F16> __device__ __forceinline__ float4 sc_key4(const char * row, int c) {
    if constexpr (F16) {
        const uint2 u = __ldg(reinterpret_cast<const uint2 *>(row) + c);
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
    return __ldg(reinterpret_cast<const float4 *>(row) + c);
}

__device__ __forceinline__ void sc_split4(float4 x, uint2 & hi, uint2 & lo) {
    __half h[4], l[4];
    sc_split(x.x, h[0], l[0]);
    sc_split(x.y, h[1], l[1]);
    sc_split(x.z, h[2], l[2]);
    sc_split(x.w, h[3], l[3]);
    const __half2 h01 = __halves2half2(h[0], h[1]), h23 = __halves2half2(h[2], h[3]);
    const __half2 l01 = __halves2half2(l[0], l[1]), l23 = __halves2half2(l[2], l[3]);
    hi = make_uint2(*reinterpret_cast<const uint32_t *>(&h01), *reinterpret_cast<const uint32_t *>(&h23));
    lo = make_uint2(*reinterpret_cast<const uint32_t *>(&l01), *reinterpret_cast<const uint32_t *>(&l23));
}

// d += a * b on one QP's 8 x 8 x 4 (row-major A, column-major B, F32 accumulators)
__device__ __forceinline__ void sc_mma(float (&d)[8], uint2 a, uint2 b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

// Output element e of thread tq within the QP's 8 x 8: row (pool) (tq & 1) + 2 * ((e >> 1) & 1) +
// 4 * (tq >> 2), column (query) sc_col(tq, e).
__device__ __forceinline__ int sc_col(int tq, int e) { return 2 * ((tq >> 1) & 1) + (e & 1) + 4 * (e >> 2); }

struct ScSmem {
    __half qh[SC_HEADS][SC_VQ][SC_QS], ql[SC_HEADS][SC_VQ][SC_QS];  // queries hi / lo [head][query][dim]
    float w[SC_HEADS][SC_VQ];
    int first[SC_VQ], end[SC_VQ];
};
// Within the 48 KB of dynamic shared memory a launch gets without an opt-in attribute.
static_assert(sizeof(ScSmem) <= 48 * 1024, "ScSmem fits the default shared memory");

// Heads h0 .. h0 + 3. acc_in: the sums start from dst (the previous heads'); fin: the last heads,
// then the mask and the bias (before, dst gets the bare sums of the visible tiles).
template <bool KF16>
__global__ void __launch_bounds__(128) qsa_scores_kernel(const legion_qsa_scores p, int h0, int acc_in, int fin) {
    extern __shared__ __align__(16) unsigned char sc_raw[];
    ScSmem & S = *reinterpret_cast<ScSmem *>(sc_raw);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    // CTA order: groups of SC_GQ query tiles, query tiles fastest within a group. The CTAs resident at once
    // then read a few pool tiles' keys from L2 instead of every key from DRAM per query tile (keys outgrow
    // L2 from ~48K context).
    int qt, pt;
    {
        const int npt = gridDim.x, nqt = gridDim.y;
        const int lin = blockIdx.y * npt + blockIdx.x, g = lin / (SC_GQ * npt), rem = lin % (SC_GQ * npt);
        const int rows = min(SC_GQ, nqt - g * SC_GQ);
        qt = g * SC_GQ + rem % rows;
        pt = rem / rows;
    }
    const int q0 = qt * SC_VQ;
    for (int i = t; i < SC_VQ * SC_HEADS * SC_D / 4; i += 128) {
        const int row = i / (SC_D / 4), c = i % (SC_D / 4);
        const int qr = row / SC_HEADS, h = row % SC_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < p.ng) {
            v = reinterpret_cast<const float4 *>((const char *) p.q + (int64_t) (q0 + qr) * p.q_nb2 + (h0 + h) * p.q_nb1)[c];
        }
        const float x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sc_split(x[j], S.qh[h][qr][c * 4 + j], S.ql[h][qr][c * 4 + j]);
        }
    }
    if (t < SC_VQ * SC_HEADS) {
        const int qr = t / SC_HEADS, h = t % SC_HEADS;
        S.w[h][qr] = q0 + qr < p.ng ? ((const float *) ((const char *) p.w + (int64_t) (q0 + qr) * p.w_nb1))[h0 + h] : 0.f;
    }
    if (t < SC_VQ) {
        const bool in = q0 + t < p.ng;
        S.first[t] = in ? p.vis[2 * (q0 + t)] : 0;
        S.end[t]   = in ? p.vis[2 * (q0 + t) + 1] : 0;
    }
    __syncthreads();
    int hi_end = 0;
#pragma unroll
    for (int i = 0; i < SC_VQ; ++i) {
        hi_end = max(hi_end, S.end[i]);
    }
    const int qp = (lane >> 2) & 3, tq = (lane & 3) + 4 * (lane >> 4);
    const int prow = 8 * (qp & 1) + tq;   // the A row (pool) this lane loads
    const int bcol = 8 * (qp >> 1) + tq;  // the B column (query) it reads
    // the queries of this lane's outputs (columns 0, 1, 4, 5 of its QP plus 2 * bit 1 of tq)
    float wq[SC_HEADS][4];
    int fq[4], eq[4], qq[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int n = 8 * (qp >> 1) + sc_col(tq, (c & 1) + 4 * (c >> 1));
        qq[c] = n;
        fq[c] = S.first[n];
        eq[c] = S.end[n];
#pragma unroll
        for (int h = 0; h < SC_HEADS; ++h) {
            wq[h][c] = S.w[h][n];
        }
    }
    const uint4 * qh = reinterpret_cast<const uint4 *>(&S.qh[0][bcol][0]);
    const uint4 * ql = reinterpret_cast<const uint4 *>(&S.ql[0][bcol][0]);
    constexpr int QH = SC_VQ * SC_QS / 8;  // uint4 per head
    for (int it = 0; it < SC_ITER; ++it) {
        const int b0 = ((pt * SC_ITER + it) * 4 + warp) * 16;
        if (b0 >= p.n_pool) {
            break;  // warp-uniform
        }
        float sc[8];
        if (b0 < hi_end) {
            const int row = b0 + prow;
            const bool rv = row < hi_end && row < p.n_pool;
            const int kr_row = rv ? (p.rows ? p.rows[row] : row) : 0;
            const char * kp = (const char *) p.keys + (int64_t) kr_row * p.k_nb;
            float acc[SC_HEADS][8], cor[SC_HEADS][8];
#pragma unroll
            for (int h = 0; h < SC_HEADS; ++h) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[h][e] = 0.f;
                    cor[h][e] = 0.f;
                }
            }
            // lanes l and l ^ 8 (QPs q and q ^ 2) share the pool row: each loads and splits 8 of every
            // 16 dims (chunks 2 * half, + 1) and they swap the split halves
            const int half = qp >> 1;
            const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 nk[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                nk[j] = rv ? sc_key4<KF16>(kp, 2 * half + j) : z4;
            }
#pragma unroll 1
            for (int kk = 0; kk < SC_D / 16; ++kk) {
                uint2 ah[4], al[4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint2 oh, ol, rh, rl;
                    sc_split4(nk[j], oh, ol);
                    rh.x = __shfl_xor_sync(0xffffffffu, oh.x, 8);
                    rh.y = __shfl_xor_sync(0xffffffffu, oh.y, 8);
                    rl.x = __shfl_xor_sync(0xffffffffu, ol.x, 8);
                    rl.y = __shfl_xor_sync(0xffffffffu, ol.y, 8);
                    ah[j] = half ? rh : oh;
                    al[j] = half ? rl : ol;
                    ah[2 + j] = half ? oh : rh;
                    al[2 + j] = half ? ol : rl;
                }
                if (kk + 1 < SC_D / 16) {  // the next 16 dims' loads behind these products
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        nk[j] = rv ? sc_key4<KF16>(kp, (kk + 1) * 4 + 2 * half + j) : z4;
                    }
                }
#pragma unroll
                for (int h = 0; h < SC_HEADS; ++h) {
                    const uint4 h0 = qh[h * QH + kk * 2], h1 = qh[h * QH + kk * 2 + 1];
                    const uint4 l0 = ql[h * QH + kk * 2], l1 = ql[h * QH + kk * 2 + 1];
                    const uint2 bh[4] = {make_uint2(h0.x, h0.y), make_uint2(h0.z, h0.w), make_uint2(h1.x, h1.y), make_uint2(h1.z, h1.w)};
                    const uint2 bl[4] = {make_uint2(l0.x, l0.y), make_uint2(l0.z, l0.w), make_uint2(l1.x, l1.y), make_uint2(l1.z, l1.w)};
                    // hi * hi from zero per 16 dims, added with FP32 adds (Volta's tensor cores truncate
                    // when they accumulate); hi * lo then lo * hi chained in cor
                    float st[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        sc_mma(st, ah[c], bh[c]);
                    }
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        acc[h][e] += st[e];
                    }
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        sc_mma(cor[h], ah[c], bl[c]);
                    }
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        sc_mma(cor[h], al[c], bh[c]);
                    }
                }
            }
            // the previous heads' sums: element e is pool row 2 * ((e >> 1) & 1) + (tq & 1) of the QP's
            // 4 * (tq >> 2) block, query qq[(e & 1) + 2 * (e >> 2)]
            float start[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int b = b0 + 8 * (qp & 1) + 4 * (tq >> 2) + 2 * ((e >> 1) & 1) + (tq & 1);
                const int qn = q0 + qq[(e & 1) + 2 * (e >> 2)];
                start[e] = acc_in && b < p.n_pool && qn < p.ng ? ((const float *) ((const char *) p.dst + (int64_t) qn * p.dst_nb1))[b] : 0.f;
            }
            // relu per head times its weight, heads added in order (ggml's lightning_indexer)
#pragma unroll
            for (int h = 0; h < SC_HEADS; ++h) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float v = acc[h][e] + cor[h][e] * (1.0f / SC_LO);
                    const float r = (v > 0.f ? v : 0.f) * wq[h][(e & 1) + 2 * (e >> 2)];
                    sc[e] = h > 0 ? sc[e] + r : acc_in ? start[e] + r : r;
                }
            }
        }
        // Lanes tq and tq ^ 1 hold rows (pools) 4 * (tq >> 2) + {0, 2} and + {1, 3} of the same four
        // queries: one exchange gives the even lane four consecutive pools of queries c = 0, 1 and the
        // odd lane those of c = 2, 3, stored as float4 (scalar where the row is not 16-byte aligned or ends).
        const int odd = tq & 1;
        float v4[2][4];
        if (b0 < hi_end) {
#pragma unroll
            for (int ck = 0; ck < 2; ++ck) {
#pragma unroll
                for (int r2 = 0; r2 < 2; ++r2) {
                    // element of column c, row 2 * r2 (+ odd): 2 * r2 + (c & 1) + 4 * (c >> 1), c = ck or 2 + ck
                    const float lo_c = sc[2 * r2 + ck], hi_c = sc[2 * r2 + ck + 4];
                    const float recv = __shfl_xor_sync(0xffffffffu, odd ? lo_c : hi_c, 1);
                    const float own = odd ? hi_c : lo_c;
                    v4[ck][2 * r2] = odd ? recv : own;
                    v4[ck][2 * r2 + 1] = odd ? own : recv;
                }
            }
        }
        const int bb = b0 + 8 * (qp & 1) + 4 * (tq >> 2);
        if (!fin && b0 >= hi_end) {
            continue;  // warp-uniform: no sums to carry
        }
        const bool vec = ((p.dst_nb1 | (int64_t) (uintptr_t) p.dst) & 15) == 0 && bb + 4 <= p.n_pool;
        float bias[4] = {0.f, 0.f, 0.f, 0.f};
        if (fin && b0 < hi_end && p.bias) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                bias[i] = bb + i < p.n_pool ? p.bias[bb + i] : 0.f;
            }
        }
#pragma unroll
        for (int ck = 0; ck < 2; ++ck) {
            // column 2 * odd + ck, by selects (a runtime index puts the arrays in local memory)
            const int qn = odd ? qq[2 + ck] : qq[ck], f = odd ? fq[2 + ck] : fq[ck], en = odd ? eq[2 + ck] : eq[ck];
            if (q0 + qn >= p.ng) {
                continue;
            }
            float o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int b = bb + i;
                const bool vis = b0 < hi_end && b >= f && b < en;
                o[i] = !fin ? v4[ck][i] : vis ? v4[ck][i] + bias[i] : -INFINITY;
            }
            float * d = (float *) ((char *) p.dst + (int64_t) (q0 + qn) * p.dst_nb1) + bb;
            if (vec) {
                *reinterpret_cast<float4 *>(d) = make_float4(o[0], o[1], o[2], o[3]);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (bb + i < p.n_pool) {
                        d[i] = o[i];
                    }
                }
            }
        }
    }
}

template <int HD>
int launch_attn(const legion_qsa_attn & p, cudaStream_t st) {
    // A quantized cache: the m8n8k4 kernel on every device (attn_mma_kernel reads F16).
    if (p.k_type != 0) {
        switch (p.k_type * 4 + p.v_type) {
            case 5: return launch_attn_types<HD, 1, 1>(p, st);
            case 6: return launch_attn_types<HD, 1, 2>(p, st);
            case 9: return launch_attn_types<HD, 2, 1>(p, st);
            case 10: return launch_attn_types<HD, 2, 2>(p, st);
            default: return 1;
        }
    }
    // sm_80+: the mma.m16n8k16 kernel. It has no chain variant: there `chain`
    // (--qsa-attn-chain, a Volta accumulation choice) does not apply, which is said once.
    if (use_mma<HD>()) {
        if (p.chain) {
            if (!chain_notice_said.exchange(true)) {
                fprintf(stderr, "legion: --qsa-attn-chain applies to the Volta QSA prompt kernel "
                                "only; this GPU runs the mma.m16n8k16 kernel\n");
            }
        }
        if (p.mma_kernel != 1 && use_gb<HD>(p.mma_kernel == 2)) {
            // ~39 KB at HD 256: under the 48 KB default, two CTAs per SM with the largest
            // shared-memory carveout.
            const int bytes = (int) sizeof(GbSmem<HD>);
            if (cudaFuncSetAttribute(attn_gb_kernel<HD>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared) != cudaSuccess) {
                return 2;
            }
            attn_gb_kernel<HD><<<dim3(p.grid[0], p.grid[1]), THREADS, bytes, st>>>(p);
            return cudaGetLastError() == cudaSuccess ? 0 : 2;
        }
        const int bytes = (int) sizeof(MmaSmem<HD>);
        if (cudaFuncSetAttribute(attn_mma_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes) != cudaSuccess) {
            return 2;
        }
        attn_mma_kernel<HD><<<dim3(p.grid[0], p.grid[1]), THREADS, bytes, st>>>(p);
        return cudaGetLastError() == cudaSuccess ? 0 : 2;
    }
    return launch_attn_types<HD, 0, 0>(p, st);
}

// ---- qsa_topk: one CTA of TK_T threads per row. Thread t owns the row's entries
// [t * per, (t + 1) * per), in registers (PER >= per) or read again on every pass (PER == 0).
// Three radix passes (11, 11 and 10 bits) find the k-th largest key with warp-aggregated
// shared-memory histograms; one block scan then places each thread's entries above the
// threshold and its tied ones (the lowest indexes first) in index order.
// Selection rule from Strata-AC922's qsa_select.cu (eelgaev/Strata-AC922, MIT): radix
// threshold, ties to the lowest index.

constexpr int TK_T = 512;
constexpr int TK_BINS = 2048;

__device__ __forceinline__ uint32_t topk_key(float v) {
    // ggml's top_k_float_to_ordered.
    const uint32_t bits = __float_as_uint(v);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// Exclusive prefix sum over the CTA's TK_T threads and the total; sw: TK_T / 32 ints.
__device__ __forceinline__ int topk_scan(int x, int * sw, int * total) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int v = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffff, v, o);
        v += lane >= o ? y : 0;
    }
    if (lane == 31) {
        sw[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        int t = lane < TK_T / 32 ? sw[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, t, o);
            t += lane >= o ? y : 0;
        }
        if (lane < TK_T / 32) {
            sw[lane] = t;
        }
    }
    __syncthreads();
    const int r = (warp > 0 ? sw[warp - 1] : 0) + v - x;
    *total = sw[TK_T / 32 - 1];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) topk_kernel(legion_qsa_topk p) {
    __shared__ int hist[TK_BINS];
    __shared__ int sw[TK_T / 32];
    __shared__ uint32_t s_digit;
    __shared__ int s_need;

    const int row = blockIdx.x, t = threadIdx.x, lane = t % 32;
    const float * src = (const float *) ((const char *) p.score + row * p.score_nb1);
    const int per = (p.n + TK_T - 1) / TK_T;
    const int i0 = t * per, i1 = min(p.n, i0 + per);

    uint32_t reg[PER > 0 ? PER : 1];
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < PER; j++) {
            reg[j] = i0 + j < i1 ? topk_key(src[i0 + j]) : 0;
        }
    }
    auto key = [&](int j) -> uint32_t {
        if constexpr (PER > 0) {
            return reg[j];
        } else {
            return topk_key(src[i0 + j]);
        }
    };
    // Loops over this thread's entries j < per; uniform trip counts across the warp (per is the
    // CTA's), so warp intrinsics inside see every lane.
    constexpr int UNROLL = PER > 0 ? PER : 1;

    uint32_t prefix = 0, pmask = 0;
    int need = p.k;
    const int shifts[3] = {21, 10, 0};
    const int widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; pass++) {
        const int shift = shifts[pass];
        const uint32_t nb = 1u << widths[pass];
        for (int i = t; i < TK_BINS; i += TK_T) {
            hist[i] = 0;
        }
        __syncthreads();
        auto count = [&](int j) {
            const uint32_t k = key(j);
            const bool in = i0 + j < i1 && (k & pmask) == prefix;
            const unsigned act = __ballot_sync(0xffffffff, in);
            if (in) {
                const int bin = (int) ((k >> shift) & (nb - 1));
                const unsigned peers = __match_any_sync(act, bin);
                if (lane == __ffs(peers) - 1) {
                    atomicAdd(&hist[bin], __popc(peers));
                }
            }
        };
        if constexpr (PER > 0) {
#pragma unroll
            for (int j = 0; j < UNROLL; j++) {
                if (j < per) {
                    count(j);
                }
            }
        } else {
            for (int j = 0; j < per; j++) {
                count(j);
            }
        }
        __syncthreads();
        // The largest digit d whose bins d.. hold at least `need` entries: suffix sums over
        // the bins, TK_BINS / TK_T per thread from the top.
        constexpr int BPT = TK_BINS / TK_T;
        const int b_hi = (int) nb - 1 - t * BPT;  // this thread's bins b_hi, b_hi - 1, ...
        int mine = 0;
#pragma unroll
        for (int b = 0; b < BPT; b++) {
            mine += b_hi - b >= 0 ? hist[b_hi - b] : 0;
        }
        int total;
        const int before = topk_scan(mine, sw, &total);  // entries in bins above this thread's
        if (before < need && before + mine >= need) {
            int acc = before;
            for (int b = 0; b < BPT; b++) {
                const int c = hist[b_hi - b];
                if (acc + c >= need) {
                    s_digit = (uint32_t) (b_hi - b);
                    s_need = need - acc;
                    break;
                }
                acc += c;
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        pmask |= (nb - 1) << shift;
        need = s_need;
        __syncthreads();
    }
    // prefix is the k-th largest key; take every entry above it and the first `need` at it.
    int above = 0, eq = 0;
    auto tally = [&](int j) {
        const uint32_t k = key(j);
        above += k > prefix;
        eq += k == prefix;
    };
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            if (i0 + j < i1) {
                tally(j);
            }
        }
    } else {
        for (int j = i0; j < i1; j++) {
            tally(j - i0);
        }
    }
    int n_above, n_eq;
    int a = topk_scan(above, sw, &n_above);
    int e = topk_scan(eq, sw, &n_eq);
    int32_t * dst = (int32_t *) ((char *) p.dst + row * p.dst_nb1);
    auto emit = [&](int j) {
        const uint32_t k = key(j);
        if (k > prefix) {
            dst[a++] = i0 + j;
        } else if (k == prefix) {
            if (e < need) {
                dst[n_above + e] = i0 + j;
            }
            e++;
        }
    };
    if constexpr (PER > 0) {
#pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            if (i0 + j < i1) {
                emit(j);
            }
        }
    } else {
        for (int j = i0; j < i1; j++) {
            emit(j - i0);
        }
    }
}

// ---- qsa_scores_exact (see the declaration). A CTA: 64 tokens x 64 pools, 256 threads as 16
// token groups x 16 pool groups; a thread: tokens tg * 4 + i, pools pg + 16 j (i, j < 4), all 4
// heads. The dims stream through shared memory 32 at a time: queries F32, keys as ggml's tile
// kernel holds them (F16, widened on use).

constexpr int SX_T = 64, SX_P = 64, SX_DC = 32;  // tokens, pools, dims per chunk
constexpr int SX_H = 4, SX_D = 128;

__global__ void __launch_bounds__(256) qsa_scores_exact_kernel(legion_qsa_scores_exact p) {
    __shared__ float2 qs[SX_T][SX_H][SX_DC / 2];     // 32 KB
    __shared__ half2 ks[SX_P][SX_DC / 2 + 1];        // padded: pool rows in distinct banks
    const int t0 = blockIdx.y * SX_T, j0 = blockIdx.x * SX_P;
    const int tid = threadIdx.x, pg = tid % 16, tg = tid / 16;

    // Tiles no token of which sees any of the pools: -inf (+ bias) without the products.
    __shared__ int s_any;
    if (tid == 0) {
        s_any = 0;
    }
    __syncthreads();
    if (tid < SX_T && t0 + tid < p.ng) {
        const int32_t * v = p.vis + 2 * (t0 + tid);
        if (v[0] < min(j0 + SX_P, p.n_pool) && v[1] > j0 && v[0] < v[1]) {
            s_any = 1;
        }
    }
    __syncthreads();
    const bool any = s_any != 0;

    float qk[4][4][SX_H];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int h = 0; h < SX_H; h++) {
                qk[i][j][h] = 0.0f;
            }
        }
    }
    if (any) {
        for (int c0 = 0; c0 < SX_D; c0 += SX_DC) {
            __syncthreads();
            // Queries: 64 tokens x 4 heads x 32 dims as float4s; tokens past ng are zero.
            for (int i = tid; i < SX_T * SX_H * (SX_DC / 4); i += 256) {
                const int c4 = i % (SX_DC / 4), h = i / (SX_DC / 4) % SX_H, t = i / (SX_DC / 4 * SX_H);
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (t0 + t < p.ng) {
                    v = *(const float4 *) ((const char *) p.q + (int64_t) (t0 + t) * p.q_nb2 + h * p.q_nb1 +
                                           (c0 + 4 * c4) * sizeof(float));
                }
                qs[t][h][2 * c4]     = make_float2(v.x, v.y);
                qs[t][h][2 * c4 + 1] = make_float2(v.z, v.w);
            }
            // Keys rounded to F16 as ggml's tile kernel stages them; pools past n_pool are zero.
            for (int i = tid; i < SX_P * (SX_DC / 4); i += 256) {
                const int c4 = i % (SX_DC / 4), r = i / (SX_DC / 4);
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (j0 + r < p.n_pool) {
                    v = *(const float4 *) ((const char *) p.keys + (int64_t) (j0 + r) * p.k_nb + (c0 + 4 * c4) * sizeof(float));
                }
                ks[r][2 * c4]     = __floats2half2_rn(v.x, v.y);
                ks[r][2 * c4 + 1] = __floats2half2_rn(v.z, v.w);
            }
            __syncthreads();
#pragma unroll 4
            for (int c = 0; c < SX_DC / 2; c++) {
                float2 kv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    kv[j] = __half22float2(ks[pg + 16 * j][c]);
                }
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int h = 0; h < SX_H; h++) {
                        const float2 qv = qs[tg * 4 + i][h][c];
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            qk[i][j][h] = fmaf(kv[j].x, qv.x, qk[i][j][h]);
                            qk[i][j][h] = fmaf(kv[j].y, qv.y, qk[i][j][h]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int t = t0 + tg * 4 + i;
        if (t >= p.ng) {
            continue;
        }
        const int32_t * v = p.vis + 2 * t;
        const int first = v[0], end = v[1];
        const float * w = (const float *) ((const char *) p.w + (int64_t) t * p.w_nb1);
        float * out = (float *) ((char *) p.dst + (int64_t) t * p.dst_nb1);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int jj = j0 + pg + 16 * j;
            if (jj >= p.n_pool) {
                continue;
            }
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < SX_H; h++) {
                score = fmaf(fmaxf(qk[i][j][h], 0.0f), w[h], score);
            }
            const float m = jj >= first && jj < end ? 0.0f : -INFINITY;
            out[jj] = (score + m) + p.bias[jj];
        }
    }
}

}  // namespace

extern "C" {

int legion_qsa_pool_mask_launch(const legion_qsa_pool_mask * p, void * stream) {
    if (p->grid[0] == 0 || p->grid[1] == 0) {
        return 0;
    }
    pool_mask_kernel<<<dim3(p->grid[0], p->grid[1]), p->block, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_qsa_select_launch(const legion_qsa_select * p, void * stream) {
    if (p->sort_n > 1024) {
        return 1;
    }
    if (p->n_tok == 0) {
        return 0;
    }
    select_kernel<<<p->n_tok, p->block, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_qsa_attn_launch(const legion_qsa_attn * p, void * stream) {
    if (p->grid[0] == 0) {
        return 0;
    }
    if (p->n_head / p->n_head_kv > ROWS) {
        return 1;
    }
    // F16 K and V, or both quantized (q8_0 / q4_0): rows the prompt kernel's loads are aligned
    // to (a head's row is HD / 32 blocks of 34 or 18 bytes: a multiple of 16 bytes at HD 256,
    // of 8 at HD 128).
    const auto quant = [](int t) { return t == 1 || t == 2; };
    if (p->k_type != 0 || p->v_type != 0) {
        const int lw = p->head_dim % 256 == 0 ? 16 : 8;
        if (!quant(p->k_type) || !quant(p->v_type) || p->k_nb1 % lw != 0 || p->v_nb1 % lw != 0 ||
            (uintptr_t) p->k % lw != 0 || (uintptr_t) p->v % lw != 0) {
            return 1;
        }
    }
    if (p->part) {
        if (p->n_head / p->n_head_kv > DMAXG || p->splits < 1) {
            return 1;
        }
        switch (p->head_dim) {
            case 128: return launch_decode<128>(*p, (cudaStream_t) stream);
            case 256: return launch_decode<256>(*p, (cudaStream_t) stream);
            default: return 1;
        }
    }
    switch (p->head_dim) {
        case 128: return launch_attn<128>(*p, (cudaStream_t) stream);
        case 256: return launch_attn<256>(*p, (cudaStream_t) stream);
        default: return 1;
    }
}

int legion_qsa_topk_launch(const legion_qsa_topk * p, void * stream) {
    if (p->rows <= 0) {
        return 0;
    }
    if (p->k <= 0 || p->k > p->n) {
        return 1;
    }
    const int per = (p->n + TK_T - 1) / TK_T;
    const cudaStream_t st = (cudaStream_t) stream;
    if (per <= 8) {
        topk_kernel<8><<<p->rows, TK_T, 0, st>>>(*p);
    } else if (per <= 16) {
        topk_kernel<16><<<p->rows, TK_T, 0, st>>>(*p);
    } else if (per <= 32) {
        topk_kernel<32><<<p->rows, TK_T, 0, st>>>(*p);
    } else if (per <= 64) {
        topk_kernel<64><<<p->rows, TK_T, 0, st>>>(*p);
    } else {
        topk_kernel<0><<<p->rows, TK_T, 0, st>>>(*p);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_qsa_scores_exact_launch(const legion_qsa_scores_exact * p, void * stream) {
    if (p->ng <= 0 || p->n_pool <= 0) {
        return 0;
    }
    if (((uintptr_t) p->q | (uintptr_t) p->keys) % 16 != 0 || p->q_nb1 % 16 != 0 || p->q_nb2 % 16 != 0 ||
        p->k_nb % 16 != 0 || (p->ng + SX_T - 1) / SX_T > 65535) {
        return 1;
    }
    const dim3 grid((p->n_pool + SX_P - 1) / SX_P, (p->ng + SX_T - 1) / SX_T);
    qsa_scores_exact_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(*p);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_qsa_scores_launch(const legion_qsa_scores * p, void * stream) {
    if (p->ng <= 0 || p->n_pool <= 0) {
        return 0;
    }
    if (((uintptr_t) p->q | (uintptr_t) p->keys) % 16 != 0 || p->q_nb1 % 16 != 0 || p->q_nb2 % 16 != 0 ||
        p->k_nb % 16 != 0 || p->ng > 65535 * SC_VQ) {
        return (int) cudaErrorInvalidValue;
    }
    if (p->n_head < SC_HEADS || p->n_head % SC_HEADS != 0) {
        return (int) cudaErrorInvalidValue;
    }
    const int per = 4 * SC_ITER * 16;
    const dim3 grid((p->n_pool + per - 1) / per, (p->ng + SC_VQ - 1) / SC_VQ);
    for (int h0 = 0; h0 < p->n_head; h0 += SC_HEADS) {
        const int acc_in = h0 > 0, fin = h0 + SC_HEADS == p->n_head;
        if (p->k_f16) {
            qsa_scores_kernel<true><<<grid, 128, sizeof(ScSmem), (cudaStream_t) stream>>>(*p, h0, acc_in, fin);
        } else {
            qsa_scores_kernel<false><<<grid, 128, sizeof(ScSmem), (cudaStream_t) stream>>>(*p, h0, acc_in, fin);
        }
    }
    return (int) cudaGetLastError();
}

int legion_qsa_scores_sm7x(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        cudaGetLastError();
        return 0;
    }
    for (int dev = 0; dev < n; ++dev) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();  // no pending error left behind
            return 0;
        }
        if (major != 7) {
            return 0;
        }
    }
    return 1;
}

}  // extern "C"
