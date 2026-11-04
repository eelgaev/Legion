// Q8_0 matmuls of 9-255 tokens (MTP verify steps) as ggml-cuda's MMQ computes them, without its
// stream-k fixup launch (#836). The host side, the shapes and the CPU reference are in
// src/model/q8_mmq.rs, which launches the kernels from `legion_q8_mmq_kernels` with
// cudaLaunchKernel.
//
// ggml_cuda_mul_mat_q runs a Q8_0 weight times more than 8 F32 columns as three launches:
// quantize_mmq_q8_1 (quantize.cu), mul_mat_q (mmq.cuh) and, whenever the tiles do not fill the
// SMs' waves, mul_mat_q_stream_k_fixup, which adds the partial sums of tiles split over several
// CTAs. On a GB10 serving 16 sequences with MTP that fixup is ~300 launches and ~2.3 ms a verify
// round (#836 part 16). Here (third_party/ggml/src/ggml-cuda, MIT):
// - `quantize`: ggml's quantize_mmq_q8_1 for MMQ_Q8_1_DS_LAYOUT_D4 (the layout of a Q8_0 weight),
//   value for value (the arithmetic, the warp reductions and the layout are ggml's), the columns
//   padded to MATRIX_ROW_PADDING with zeros as ggml pads them. It also zeroes the tile counters
//   `mm` uses.
// - `mm<J, fallback, split>`: mul_mat_q's stream-k loop for a dense matmul (one channel, one
//   sample, no ids) around ggml's own mul_mat_q_process_tile, with ggml's grid (host side), so
//   every CTA sums the K blocks ggml's CTA of the same index sums, with ggml's tile loads, MMA
//   and write back. With split tiles, a CTA that leaves a piece of one (a partial sum in `part`,
//   as ggml's fixup buffer, or the tile's last piece in dst) counts it on the tile's counter; the
//   CTA that counts last adds the partial sums to dst in mul_mat_q_stream_k_fixup's order:
//   dst = last + (((0 + P[b - 1]) + P[b - 2]) + ...), walking back from the CTA b holding the
//   tile's end as the fixup kernel does. Same products, same sums: ggml's bits. Grids of a CTA
//   per tile run `mm<J, fallback, false>`, ggml's loop without the split-tile code.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"

#include <cstdint>

extern "C" {
struct legion_q8_mmq {
    const char * w;      // Q8_0 [rows][k], w_ld blocks between rows
    const float * x;     // F32 [n][k], x_ld floats between columns
    void * yq;           // block_q8_1_mmq [k_pad / 128][n] (+ MMQ's read-ahead padding)
    int * cnt;           // [ntiles] split-tile counters, zeroed by `quantize`
    float * part;        // [grid][J * 128] partial sums of split tiles (ggml's tmp_fixup)
    float * dst;         // F32 [n][rows]
    int64_t x_ld;
    int32_t w_ld;
    int32_t k;           // a multiple of 32
    int32_t k_pad;       // k rounded up to 512 (MATRIX_ROW_PADDING)
    int32_t n;           // columns, 9-255
    int32_t rows;
    int32_t ntiles;      // output tiles of the matmul
    int32_t nzero;       // tile counters `quantize` zeroes (the most tiles of the nodes that
                         // reuse its blocks, #858)
    uint32_t bpt_mp;     // ggml's fastdiv values of the K blocks per tile (k / 32)
    uint32_t bpt_l;
    uint32_t ntx_mp;     // and of the column tiles
    uint32_t ntx_l;
};

// The fast kernels' launch (sm80/q8_mmq_fast.cuh): ggml's grid and tiles, ggml's bits.
struct legion_q8_fast {
    const char * w;      // Q8_0 [rows][k], w_ld blocks between rows (even), 4-byte aligned
    const float * x;     // F32 [n][k], x_ld floats between columns
    int * yq;            // [k / 32][n_pad][8] activation quants in the K slot order of `fast`
    float * yd;          // [k / 32][n_pad] activation scales
    int * cnt;           // [ncnt] split-tile counters (tile * row slices), zeroed by `fquant`
    float * part;        // [ctas][J * 128] partial sums of split tiles (ggml's tmp_fixup)
    float * dst;         // F32 [n][rows]
    int64_t x_ld;
    int32_t w_ld;
    int32_t k;           // a multiple of 256
    int32_t n;           // columns
    int32_t n_pad;       // ntx * J
    int32_t rows;
    int32_t ntiles;      // ntx * nty
    int32_t ntx;
    int32_t ctas;        // ggml's grid (gridDim.x)
    int32_t ncnt;        // ntiles * row slices
    int32_t sep;         // 1: split tiles' pieces left in dst and `part` for `split_fixup`
};
}

namespace {

constexpr ggml_type TYPE = GGML_TYPE_Q8_0;
constexpr ggml_prec PREC = GGML_PREC_Q8;

// ggml's CUDA_QUANTIZE_BLOCK_SIZE_MMQ.
constexpr int QUANT_THREADS = 128;

// Most CTAs of a split matmul (the GPU's SMs; the host side keeps to it): a thread per CTA
// lists the pieces of a tile.
constexpr int MAX_CTAS = 256;

// quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false> for a dense, contiguous-row input: grid
// (n, k_pad / (4 * QUANT_THREADS)), QUANT_THREADS threads, 4 values each.
__global__ void __launch_bounds__(QUANT_THREADS) quantize(const legion_q8_mmq p) {
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int t = threadIdx.x; t < p.nzero; t += blockDim.x) {
            p.cnt[t] = 0;
        }
    }
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= p.k_pad) {
        return;
    }
    const float * xc = p.x + blockIdx.x * p.x_ld;
    block_q8_1_mmq * y = (block_q8_1_mmq *) p.yq;

    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    // k is a multiple of 32: the 4 values are all in or all padding. One 16-byte load where
    // aligned (the same values).
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < p.k) {
        const float * src = xc + i0;
        xi = (reinterpret_cast<uintptr_t>(src) & 15) == 0
            ? *reinterpret_cast<const float4 *>(src)
            : make_float4(src[0], src[1], src[2], src[3]);
    }
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

    const int64_t ib = k_block * p.n + blockIdx.x;
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;
    if (iqs % 32 == 0) {
        y[ib].d4[iqs/32] = d;
    }
}

// mul_mat_q's first K block of CTA `b` of `grid` (kbc, moved back to its K iteration).
__device__ __forceinline__ int kstart(const int b, const int grid, const int64_t total, const int bpt, const int bpi) {
    int kbc = int64_t(b) * total / grid;
    kbc -= (kbc % bpt) % bpi;
    return kbc;
}

// Pieces of a split tile the fixup loads at once, per value group (`fixup_add`).
constexpr int FIX_PIECES = 8;
// Groups of 4 values (one 16-byte load from a piece) a thread adds at once.
constexpr int FIX_GROUPS = 2;

// The fixup of a split tile by one CTA (#858): dst[j][i] = dst[j][i] + (((0 + P[list[0]]) +
// P[list[1]]) + ...) for the rows rl0 .. rl0 + RB - 1 of the tile's 128 and its J columns,
// mul_mat_q_stream_k_fixup's sum per value (list: the earlier pieces' CTAs, nearest first;
// -1 for a CTA without a piece, which adds +0: the sum starts at +0 and so is never -0, and
// x + 0 == x for every other x). P[c] is CTA c's partial sum, [J][128] at part + c * J * 128.
// A thread sums FIX_GROUPS groups of 4 rows over FIX_PIECES pieces from 16-byte loads all in
// flight before the adds, which keep the order: the latency of one CTA's reads, not of each
// piece in turn (one value per piece per thread was ~10 GB/s an SM on the GB10).
template <int J, int RB, int THREADS>
__device__ __forceinline__ void fixup_add(
        const float * part, const int * list, const int nlist, float * dst0, const int rows,
        const int ncols, const int it, const int jt, const int rl0) {
    constexpr int GROUPS = RB / 4 * J;  // groups of 4 rows of a column
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int g0 = tid; g0 < GROUPS; g0 += THREADS * FIX_GROUPS) {
        float4 sum[FIX_GROUPS];
#pragma unroll
        for (int v = 0; v < FIX_GROUPS; v++) {
            sum[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        for (int u0 = 0; u0 < nlist; u0 += FIX_PIECES) {
            float4 val[FIX_PIECES][FIX_GROUPS];
#pragma unroll
            for (int u = 0; u < FIX_PIECES; u++) {
                const int c = u0 + u < nlist ? list[u0 + u] : -1;
#pragma unroll
                for (int v = 0; v < FIX_GROUPS; v++) {
                    const int g = g0 + v * THREADS;
                    const int il = rl0 + (g % (RB / 4)) * 4, jl = g / (RB / 4);
                    val[u][v] = c >= 0 && g < GROUPS
                        ? __ldcg(reinterpret_cast<const float4 *>(part + (int64_t) c * (J * 128) + jl * 128 + il))
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
#pragma unroll
            for (int u = 0; u < FIX_PIECES; u++) {
#pragma unroll
                for (int v = 0; v < FIX_GROUPS; v++) {
                    sum[v].x += val[u][v].x;
                    sum[v].y += val[u][v].y;
                    sum[v].z += val[u][v].z;
                    sum[v].w += val[u][v].w;
                }
            }
        }
#pragma unroll
        for (int v = 0; v < FIX_GROUPS; v++) {
            const int g = g0 + v * THREADS;
            if (g >= GROUPS) {
                continue;
            }
            const int i = it * 128 + rl0 + (g % (RB / 4)) * 4, j = jt * J + g / (RB / 4);
            if (j >= ncols) {
                continue;
            }
            float * d = dst0 + (int64_t) j * rows + i;
            const float s[4] = { sum[v].x, sum[v].y, sum[v].z, sum[v].w };
#pragma unroll
            for (int r = 0; r < 4; r++) {
                if (i + r < rows) {
                    d[r] = __ldcg(d + r) + s[r];
                }
            }
        }
    }
}

// A piece of split tile `t` is in memory (this CTA's partial sum in `part`, or the tile's last
// piece in dst): count it; the CTA that counts last adds the pieces up in
// mul_mat_q_stream_k_fixup's order. Its own function (not inlined), so the tile loop's
// registers are allocated as in ggml's kernel. `shared`: the tile area after the column ids,
// free between tiles (no static shared memory: 1 KB more costs the J = 64 kernels their
// second CTA per SM).
template <int J, bool fallback>
static __device__ __noinline__ void counted(
        int * cnt, const float * part, float * dst0, const int rows, const int ncols, const int t,
        const int grid, const int64_t total, const int bpt, const int ntx, int * shared) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = ggml_cuda_mmq_get_nthreads(TYPE, J, fallback, PREC);
    constexpr int nwarps    = nthreads / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<TYPE>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(TYPE, J, fallback, PREC);
    constexpr int bpi       = ggml_cuda_mmq_get_K_vram(TYPE, J, fallback, PREC) / qk;
    static_assert(nthreads == MAX_CTAS, "a thread per CTA when listing a tile's pieces");

    int * masks = shared;          // the warps' ballots of CTAs holding a piece
    int * last  = masks + nwarps;  // whether this CTA counted last
    int * prev  = last + 1;        // the earlier pieces' CTAs, nearest first

    const int tid = threadIdx.y*warp_size + threadIdx.x;
    __threadfence();
    // The CTAs holding a piece of tile t: thread b checks CTA b. The fixup's walk back from the
    // CTA holding the tile's end (the highest), skipping empty CTAs and stopping at the piece
    // that holds the tile's start, visits exactly the others, highest first.
    const int t0 = t * bpt;
    bool mine = false;
    if (tid < grid) {
        const int s0 = kstart(tid,     grid, total, bpt, bpi);
        const int s1 = kstart(tid + 1, grid, total, bpt, bpi);
        mine = s0 < s1 && s1 > t0 && s0 < t0 + bpt;
    }
    const unsigned int ballot = __ballot_sync(0xFFFFFFFF, mine);
    __syncthreads(); // the tile area is free
    if (threadIdx.x == 0) {
        masks[threadIdx.y] = ballot;
    }
    __syncthreads();
    // Pieces in higher CTAs than this thread's (its rank in the fixup's walk, 0: the end).
    int rank = __popc(ballot & ~((2u << threadIdx.x) - 1u));
    int pieces = 0;
#pragma unroll
    for (int w = 0; w < nwarps; w++) {
        const int c = __popc(masks[w]);
        pieces += c;
        rank += w > (int) threadIdx.y ? c : 0;
    }
    if (mine && rank > 0) {
        prev[rank - 1] = tid;
    }
    if (tid == 0) {
        const bool l = atomicAdd(&cnt[t], 1) == pieces - 1;
        if (l) {
            // Every piece is counted: the counter goes back to 0 for the next matmul, which
            // may reuse this one's quantized input without a `quantize` to zero it (#858).
            cnt[t] = 0;
        }
        *last = l ? 1 : 0;
    }
    __syncthreads();
    const bool fix = *last;
    __syncthreads(); // `last` read before the next tile's loads overwrite it
    if (!fix) {
        return;
    }
    __threadfence();
    static_assert(I == 128, "fixup_add's tile rows");
    fixup_add<J, I, nthreads>(part, prev, pieces - 1, dst0, rows, ncols, t / ntx, t % ntx, 0);
    __syncthreads(); // `prev` read before the next tile's loads overwrite it
}

// `count`: the split tiles' pieces are counted and added up by the CTA that counts last
// (`counted`); without it the pieces stay in dst and `part` for `split_fixup`, ggml's mul_mat_q.
template <int J, bool fallback, bool split, bool count = true>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(TYPE, J, fallback, PREC), ggml_cuda_mmq_get_occupancy(TYPE, J, fallback, PREC))
__global__ void mm(const legion_q8_mmq p) {
    if (ggml_cuda_mmq_get_config(TYPE, J, fallback, PREC).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = ggml_cuda_mmq_get_nthreads(TYPE, J, fallback, PREC);
    constexpr int nwarps    = nthreads / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<TYPE>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(TYPE, J, fallback, PREC);
    constexpr int bpi       = ggml_cuda_mmq_get_K_vram(TYPE, J, fallback, PREC) / qk;
    constexpr int sz        = sizeof(block_q8_1_mmq) / sizeof(int);

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int grid   = gridDim.x;
    const int bpt    = p.k / qk;
    const int ntx    = (p.n + J - 1) / J;
    const int nty    = (p.rows + I - 1) / I;
    const int64_t total = int64_t(ntx) * nty * bpt;
    const int * y = (const int *) p.yq;
    // The tile index math as ggml's mul_mat_q does it (fastdiv, #858): the same values.
    const uint3 bpt_fd = make_uint3(p.bpt_mp, p.bpt_l, bpt);
    const uint3 ntx_fd = make_uint3(p.ntx_mp, p.ntx_l, ntx);

    int kbc      = int64_t(blockIdx.x)     * total / grid;
    int kbc_stop = int64_t(blockIdx.x + 1) * total / grid;
    kbc      -= fastmodulo(kbc,      bpt_fd) % bpi;
    kbc_stop -= fastmodulo(kbc_stop, bpt_fd) % bpi;
    int kb0_start = fastmodulo(kbc, bpt_fd);
    int kb0_stop  = min(bpt, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == bpt) {
        const int t  = fastdiv(kbc, bpt_fd);
        const uint2 tj = fast_div_modulo(t, ntx_fd);
        const int jt = tj.y;
        const int it = tj.x;
        const int offset_y   = jt*J*sz;
        const int offset_dst = jt*J*p.rows + it*I;
        const int tile_x_max_i = p.rows - it*I - 1;
        const int tile_y_max_j = p.n    - jt*J - 1;
        const int offset_x = it*I*p.w_ld;

        mul_mat_q_process_tile<TYPE, J, fallback, false, PREC>
            (p.w, offset_x, y + offset_y, ids_dst_shared, p.dst + offset_dst, p.part, nullptr,
             p.w_ld, p.n, p.rows, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
        if constexpr (split && count) {
            if (kb0_start != 0) {
                counted<J, fallback>(p.cnt, p.part, p.dst, p.rows, p.n, t, grid, total, bpt, ntx, ids_dst_shared + J);
            }
        }

        kbc += bpt;
        kbc -= fastmodulo(kbc, bpt_fd);
        kb0_start = 0;
        kb0_stop  = min(bpt, kbc_stop - kbc);
    }
    if constexpr (split) {
        if (kbc >= kbc_stop) {
            return;
        }
        const int t  = fastdiv(kbc, bpt_fd);
        const uint2 tj = fast_div_modulo(t, ntx_fd);
        const int jt = tj.y;
        const int it = tj.x;
        const int offset_y   = jt*J*sz;
        const int offset_dst = jt*J*p.rows + it*I;
        const int tile_x_max_i = p.rows - it*I - 1;
        const int tile_y_max_j = p.n    - jt*J - 1;
        const int offset_x = it*I*p.w_ld;
        // Writes to p.part + blockIdx.x * (J * I), as ggml's to its fixup buffer.
        mul_mat_q_process_tile<TYPE, J, fallback, true, PREC>
            (p.w, offset_x, y + offset_y, ids_dst_shared, p.dst + offset_dst, p.part, nullptr,
             p.w_ld, p.n, p.rows, tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
        if constexpr (count) {
            counted<J, fallback>(p.cnt, p.part, p.dst, p.rows, p.n, t, grid, total, bpt, ntx, ids_dst_shared + J);
        }
    }
}

// The fixup of split tiles as a launch of its own after `mm<J, fallback, true, false>` or
// `fast` without the count (#858): ggml's mul_mat_q_stream_k_fixup (mmq.cuh, MIT) for a dense
// matmul. Grid (CTAs of the matmul, 128 / 32), 32 x 4 threads: the CTA b holding a tile's end
// (its K range starts inside the tile and reaches the tile's end) adds the partial sums of the
// CTAs before it, nearest first, for 32 rows of the tile and its J columns:
// dst = last + (((0 + P[b - 1]) + P[b - 2]) + ...), ggml's order, so ggml's bits. Where a tile
// is split over many CTAs the add-up runs on 4 CTAs per tile at once instead of on the one
// that counts last.
template <int J>
__global__ void __launch_bounds__(128) split_fixup(const legion_q8_mmq p) {
    constexpr int I   = 128;
    constexpr int NW  = 4;
    constexpr int BPI = 8;  // Q8_0 blocks per K iteration (MMQ_ITER_K / QK8_0)
    float sum[J / NW] = {0.0f};
    const int i = blockIdx.y * 32 + threadIdx.x;
    const int grid = gridDim.x;
    const int bpt = p.k / 32;
    const int64_t total = (int64_t) p.ntiles * bpt;
    const int b0 = blockIdx.x;
    const int kbc0      = kstart(b0,     grid, total, bpt, BPI);
    const int kbc0_stop = kstart(b0 + 1, grid, total, bpt, BPI);
    if (kbc0 == kbc0_stop || kbc0 % bpt == 0 || (kbc0 / bpt == kbc0_stop / bpt && kbc0_stop % bpt != 0)) {
        return;  // no data, the tile's start, or not the tile's end
    }
    // ggml walks back one CTA at a time (CTA bidx: no K blocks, skip it; else add its partial
    // sum; stop after the CTA holding the tile's start), each step a K-range division and an
    // L2 read the next step waits for. Here (#858) the 32 lanes of a warp find the next 32
    // CTAs' ranges at once, a ballot picks the CTAs the walk adds (with data, up to the first
    // that holds the tile's start), and up to FB of their partial sums are read at once before
    // they are added, nearest first: ggml's order, so ggml's sums.
    constexpr int FB = J / NW <= 4 ? 8 : J / NW <= 10 ? 4 : J / NW <= 16 ? 2 : 1;
    const int lane = threadIdx.x;
    const int tile0 = kbc0 / bpt;
    bool any = false;
    for (int base = b0 - 1; base >= 0; base -= 32) {
        const int b = base - lane;
        int kbc = 0, kbc_stop = 0;
        if (b >= 0) {
            kbc      = kstart(b,     grid, total, bpt, BPI);
            kbc_stop = kstart(b + 1, grid, total, bpt, BPI);
        }
        const bool data  = b >= 0 && kbc != kbc_stop;
        const bool start = data && (kbc % bpt == 0 || kbc / bpt < tile0);
        const unsigned int has   = __ballot_sync(0xFFFFFFFF, data);
        const unsigned int stops = __ballot_sync(0xFFFFFFFF, start);
        // The lanes up to the first that holds the tile's start (all 32 if none does).
        const unsigned int upto = stops ? ((stops & (0u - stops)) << 1) - 1u : 0xFFFFFFFFu;
        unsigned int take = has & upto;
        any = any || take != 0;
        while (take) {
            int c[FB];
#pragma unroll
            for (int u = 0; u < FB; u++) {
                c[u] = take ? base - (__ffs(take) - 1) : -1;
                take &= take - 1;
            }
            float val[FB][J / NW];
#pragma unroll
            for (int u = 0; u < FB; u++) {
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += NW) {
                    val[u][j0 / NW] = c[u] >= 0
                        ? __ldcg(p.part + (int64_t) c[u] * (J * I) + (j0 + threadIdx.y) * I + i)
                        : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < FB; u++) {
                if (c[u] < 0) {
                    break;
                }
#pragma unroll
                for (int j0 = 0; j0 < J; j0 += NW) {
                    sum[j0 / NW] += val[u][j0 / NW];
                }
            }
        }
        if (stops) {
            break;
        }
    }
    if (!any) {
        return;
    }
    const int ntx = (p.n + J - 1) / J;
    const int t = kbc0 / bpt, jt = t % ntx, it = t / ntx;
    float * dst = p.dst + (int64_t) jt * J * p.rows + it * I;
    if (i > p.rows - it * I - 1) {
        return;
    }
    const int j_max = p.n - jt * J - 1;
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += NW) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) {
            return;
        }
        dst[(int64_t) j * p.rows + i] += sum[j0 / NW];
    }
}

#include "sm80/q8_mmq_fast.cuh"

}  // namespace

// The quantization, then `mm` for J = 8, 16, ..., 128 (index J / 8; ggml has no Q8_0 config
// for 56, 72, 88, 104 or 120), then with the fallback for rows not a multiple of 128 (J = 8,
// 16, 32, 64, 128 at 17-21; ggml has no other widths): the kernels for split tiles; then the
// same 21 for grids of a CTA per tile (22-42), ggml's loop without the split-tile code.
#define LEGION_Q8_MMQ_KERNELS(split, count)                              \
    reinterpret_cast<const void *>(&mm<8, false, split, count>),           \
    reinterpret_cast<const void *>(&mm<16, false, split, count>),          \
    reinterpret_cast<const void *>(&mm<24, false, split, count>),          \
    reinterpret_cast<const void *>(&mm<32, false, split, count>),          \
    reinterpret_cast<const void *>(&mm<40, false, split, count>),          \
    reinterpret_cast<const void *>(&mm<48, false, split, count>),          \
    nullptr,  /* J = 56: no Q8_0 config */                          \
    reinterpret_cast<const void *>(&mm<64, false, split, count>),          \
    nullptr,  /* J = 72 */                                          \
    reinterpret_cast<const void *>(&mm<80, false, split, count>),          \
    nullptr,  /* J = 88 */                                          \
    reinterpret_cast<const void *>(&mm<96, false, split, count>),          \
    nullptr,  /* J = 104 */                                         \
    reinterpret_cast<const void *>(&mm<112, false, split, count>),         \
    nullptr,  /* J = 120 */                                         \
    reinterpret_cast<const void *>(&mm<128, false, split, count>),         \
    reinterpret_cast<const void *>(&mm<8, true, split, count>),            \
    reinterpret_cast<const void *>(&mm<16, true, split, count>),           \
    reinterpret_cast<const void *>(&mm<32, true, split, count>),           \
    reinterpret_cast<const void *>(&mm<64, true, split, count>),           \
    reinterpret_cast<const void *>(&mm<128, true, split, count>)

#define LEGION_Q8_FIXUP_KERNELS                                     \
    reinterpret_cast<const void *>(&split_fixup<8>),                      \
    reinterpret_cast<const void *>(&split_fixup<16>),                     \
    reinterpret_cast<const void *>(&split_fixup<24>),                     \
    reinterpret_cast<const void *>(&split_fixup<32>),                     \
    reinterpret_cast<const void *>(&split_fixup<40>),                     \
    reinterpret_cast<const void *>(&split_fixup<48>),                     \
    reinterpret_cast<const void *>(&split_fixup<56>),                     \
    reinterpret_cast<const void *>(&split_fixup<64>),                     \
    reinterpret_cast<const void *>(&split_fixup<72>),                     \
    reinterpret_cast<const void *>(&split_fixup<80>),                     \
    reinterpret_cast<const void *>(&split_fixup<88>),                     \
    reinterpret_cast<const void *>(&split_fixup<96>),                     \
    reinterpret_cast<const void *>(&split_fixup<104>),                    \
    reinterpret_cast<const void *>(&split_fixup<112>),                    \
    reinterpret_cast<const void *>(&split_fixup<120>),                    \
    reinterpret_cast<const void *>(&split_fixup<128>)

// ... then the 21 split-tile kernels without the count (43-63: ggml's mul_mat_q, the pieces
// left for `split_fixup`), then `split_fixup<J>` for J = 8, 16, ..., 128 (64 + J / 8 - 1).
extern "C" const void * const legion_q8_mmq_kernels[80] = {
    reinterpret_cast<const void *>(&quantize),
    LEGION_Q8_MMQ_KERNELS(true, true),
    LEGION_Q8_MMQ_KERNELS(false, true),
    LEGION_Q8_MMQ_KERNELS(true, false),
    LEGION_Q8_FIXUP_KERNELS,
};

// The fast path (#858): `fquant`, then `fast<J, 8>` for J = 8, 16, ..., 128 (index J / 8, all
// 16 widths: the tile width is the plan's, with or without ggml's fallback), then `fast<J, 4>`
// (two CTAs per ggml CTA, 64 rows each).
#define LEGION_Q8_FAST_KERNELS(nw)                                  \
    reinterpret_cast<const void *>(&fast<8, nw>),                   \
    reinterpret_cast<const void *>(&fast<16, nw>),                  \
    reinterpret_cast<const void *>(&fast<24, nw>),                  \
    reinterpret_cast<const void *>(&fast<32, nw>),                  \
    reinterpret_cast<const void *>(&fast<40, nw>),                  \
    reinterpret_cast<const void *>(&fast<48, nw>),                  \
    reinterpret_cast<const void *>(&fast<56, nw>),                  \
    reinterpret_cast<const void *>(&fast<64, nw>),                  \
    reinterpret_cast<const void *>(&fast<72, nw>),                  \
    reinterpret_cast<const void *>(&fast<80, nw>),                  \
    reinterpret_cast<const void *>(&fast<88, nw>),                  \
    reinterpret_cast<const void *>(&fast<96, nw>),                  \
    reinterpret_cast<const void *>(&fast<104, nw>),                 \
    reinterpret_cast<const void *>(&fast<112, nw>),                 \
    reinterpret_cast<const void *>(&fast<120, nw>),                 \
    reinterpret_cast<const void *>(&fast<128, nw>)

extern "C" const void * const legion_q8_fast_kernels[33] = {
    reinterpret_cast<const void *>(&fquant),
    LEGION_Q8_FAST_KERNELS(8),
    LEGION_Q8_FAST_KERNELS(4),
};
