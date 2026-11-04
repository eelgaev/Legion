// A Q8_0 weight times one F32 column with the bits of ggml-cuda's quantize_q8_1 + mul_mat_vec_q
// pair (#814): the MTP draft pass's LM head at one drafting sequence on a V100. The host side,
// the shapes and the CPU reference are in src/model/core/q8_head.rs.
//
// ggml's mul_mat_vec_q<GGML_TYPE_Q8_0, 1> (mmvq.cu, MIT) on compute capability 7.x runs one CTA
// of 4 warps per row: virtual thread t (0-127) adds vec_dot_q8_0_q8_1 (vecdotq.cuh) of the
// blocks t / 4, t / 4 + 32, t / 4 + 64, ... (ints 2 (t % 4) and 2 (t % 4) + 1 of each) into its
// own F32 sum, from 0, in that order; lane l of warp 0 then adds warps 1, 2 and 3's sums of lane
// l to its own in that order, and warp_reduce_sum (xor 16, 8, 4, 2, 1) gives the output. On the
// 248,320-row head of Qwen3.8-Flash-Next (2,560 values, 680 MB) that is 248,320 CTAs of 2.7 KB
// each, every CTA reading the input's blocks again and each lane its weights with 2-byte loads:
// ~908 us, ~745 GB/s on a V100 (nsys, #814), where the same head at 3 columns (2 rows per CTA)
// runs at ~850.
//
// Here one warp computes a whole row: lane l does the work of virtual threads l, l + 32, l + 64
// and l + 96 (blocks l / 4 + 8 m for m = 0, 1, ..., virtual thread l + 32 (m % 4)), each into a
// sum of its own, then adds them in ggml's warp order and runs ggml's warp_reduce_sum: every sum
// is the same expression over the same values in the same order, so the output has ggml's bits.
// The CTAs stay resident (a grid of SMs x CTAs per SM) and their warps walk the rows. Each lane
// keeps its blocks of the quantized input in registers for the whole kernel (quantized by the
// lane itself, ggml's quantize_q8_1 per value: the block's absolute maximum, which any order
// gives exactly, d = amax / 127 stored as F16, q = roundf(x / d)), and a warp reads its next
// row with 16-byte loads into registers while it computes the row before from shared memory.
//
// Only one column: ggml's 2-8 column kernels already take two rows per CTA and read the head
// near the V100's bandwidth.
//
// The decode matvecs (#814, `xq` set): the same kernel for src/model/core/q8_group.rs's
// one-column matvecs (a DeltaNet layer's qkv and gate, an attention layer's Q, K and V: 512 to
// 12,288 rows of 2,560 values on Qwen3.8-Flash-Next), whose input is quantized already (the
// Q8_1 blocks of q8_group.cu's `quantize`, ggml's quantize_q8_1, or the blocks the
// hyper-connection mixer writes). Each lane loads its blocks' two ints and scale from those
// blocks instead of quantizing x: the same values, so the same bits. ggml's one-column mmvq
// (and q8_group.cu's copy of it) runs a 4-warp CTA a row there too, at 550-700 GB/s on a V100
// (nsys of det C=1, #814). Up to four weights in one launch: their rows one after the other,
// each row read from and written to its own weight's.
//
// The 6,144-value projections (#814, k 6144: an attention layer's output and a DeltaNet layer's
// ssm_out on Qwen3.8-Flash-Next, 2,560 rows, ggml's `mul_mat` in the graph otherwise): 24 blocks
// a lane would take 72 registers for the input, so the CTA quantizes it once into shared memory
// (shared_x below, ggml's quantize_q8_1 values) while each warp's first row loads, and the lanes
// read their two ints and scale of each block from there. ggml runs quantize_q8_1 and a 4-warp
// CTA a row there, ~700 GB/s on a V100 (nsys of det C=1, #814).
//
// The draft pick (#814, src/model/core/q8_head/pick.rs): with `pick` set, every warp also keeps
// the best (value, tie rank) and the largest value of the rows it computed, and each CTA writes
// its warps' into `pick` (one 16-byte partial a CTA), so the 248,320 logits are not read again
// for the argmax. q8_head_argmax_kernel joins the partials into argmax_f32's pick (ggml-cuda's
// tie order, a total order, as src/cuda/sm90/cluster_select.cu's split argmax) and the row's
// largest value; q8_head_pmax_kernel then gives the pick's probability with ggml-cuda's
// soft_max_f32_parallelize_cols (softmax.cu, MIT) sums: one CTA of 256 threads per SM, each
// thread its columns in that kernel's order, its block_reduce, the CTAs' sums block-reduced
// in CTA order by the last CTA to finish, and expf(x[pick] - max) / sum, the value ggml's
// get_rows of its soft_max reads. The largest value is exact in any order, so the probability
// has the bits of ggml's soft_max + get_rows.

#include "device_cache.cuh"
#include "head_pick.cuh"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <atomic>
#include <cstdint>

extern "C" {
struct legion_q8_head {
    const void *  w;     // Q8_0 rows of k values, w_ld bytes apart
    int64_t       w_ld;  // a multiple of 16
    int64_t       rows;
    const float * x;     // F32, k values
    float *       dst;   // F32, rows values
    int32_t       k;     // 1024 or 2560; 6144 (the projections: no xq, no pick)
    void *        pick;  // the draft pick's buffer (PICK_* layout below), or null
    int32_t       pick_cap;  // partials the buffer holds
    // With xq set (no pick): x's Q8_1 blocks (k / 32 of them, ggml's quantize_q8_1), x unread,
    // and nw (1-4) weights: weight i's rows are rows ends[i - 1] (0 for i = 0) to ends[i] - 1
    // of the launch (ends[nw - 1] = rows), read from ws[i] (lds[i] bytes apart, a multiple of
    // 16) and written to dsts[i]; w, w_ld and dst unread.
    const void *  xq;
    int32_t       nw;
    const void *  ws[4];
    int64_t       lds[4];
    float *       dsts[4];
    int64_t       ends[4];
};

// The draft pick's kernels (launched from Rust with cudaLaunchKernel, one argument each).
// Column c (blockIdx.y) of the logits has the buffer's slice c: `stride` bytes from the slice
// before (head_pick.cuh), its own header, sums and partials.
struct legion_q8_head_argmax {
    const void * pick;    // the buffer the head filled
    int32_t *    dst;     // each column's pick, -1 when no value is above -FLT_MAX
    int64_t      stride;  // bytes between the buffer's column slices
};

struct legion_q8_head_pmax {
    const float *   x;       // the logits, n values a column, x_ld floats between columns
    const int32_t * pick;    // q8_head_argmax_kernel's output, one a column
    float *         dst;     // each column's pick's probability (NaN for no pick)
    void *          buf;     // the draft pick's buffer
    int32_t         n;
    int64_t         x_ld;
    int64_t         stride;  // bytes between the buffer's column slices
};

// Enqueues dst = w . x on `stream`. 0 on success; 1 for a shape, an alignment or a device the
// kernel does not take; a CUDA error code for a failed launch.
int legion_q8_head_launch(const legion_q8_head * p, void * stream);

// Whether the current device runs the kernel: compute capability 7.x (ggml's generic mmvq table,
// whose one-column CTA the bits follow).
int legion_q8_head_supported(void);

// Launches every kernel from now on with `warps` per CTA (4, 8 or 16), at most `ctas` CTAs per
// SM (0: as many as fit) and `rows` rows a warp step (1 or 2); `warps` 0 restores the default
// (4, all, 2: the fastest on a V100, #814). Same bits either way, for tuning. 0 on success, 1 for a variant the kernel does
// not have.
int legion_q8_head_force(int warps, int ctas, int rows);
}

namespace {

constexpr int QB = 34;  // bytes of a Q8_0 block: an F16 scale, 32 int8

// The draft pick's buffer layout, tie order and their helpers: head_pick.cuh (shared with
// q4k_head.cu).

// Whether the next launch takes a forced variant: warps | ctas << 8, 0 for the default.
std::atomic<int> forced{0};

// The decode matvecs' (xq) default variant: warps per CTA and rows a warp step, every CTA
// that fits (`legion q8-head-bench --group`, #814).
constexpr int XQ_WARPS = 4;
constexpr int XQ_ROWS  = 1;

// The 6,144-value projections' (k 6,144, MB 24) default variant (`legion q8-head-bench --group
// --k 6144`, #814).
constexpr int SX_WARPS = 4;
constexpr int SX_ROWS  = 1;

// Whether the kernel of MB blocks a lane keeps the input's Q8_1 blocks in shared memory, once per
// CTA (k 6,144), instead of each lane's in registers.
__host__ __device__ constexpr bool shared_x(int mb) {
    return mb > 10;
}

// Bytes of shared memory after the staged rows for the input's blocks (shared_x): each block's
// 8 ints of int8 values, then each block's F32 scale.
__host__ __device__ constexpr int shared_x_bytes(int mb) {
    return shared_x(mb) ? 8 * mb * (32 + 4) : 0;
}

// MB: blocks of each lane per row (the row has 8 MB blocks, 272 MB bytes: 17 MB 16-byte chunks),
// NW: warps per CTA, R: rows a warp takes per step, XQ: x's Q8_1 blocks given (p.xq) and the
// rows of p.nw weights (p.ws).
template <int MB, int NW, int R, bool PICK, bool XQ = false>
__global__ void __launch_bounds__(NW * WARP_SIZE) head_mv(const legion_q8_head p) {
    constexpr int NB    = 8 * MB;                     // blocks per row
    constexpr int CH    = 17 * MB;                    // 16-byte chunks per row
    constexpr int CPL   = (CH + WARP_SIZE - 1) / WARP_SIZE;  // chunks per lane
    constexpr int vdr   = VDR_Q8_0_Q8_1_MMVQ;         // ints per lane and block (2)
    static_assert(NB * QB == 16 * CH, "whole 16-byte chunks per row");
    static_assert(NB >= 32, "ggml's small-K mmvq (more rows per CTA) is another sum");
    static_assert(!(PICK && XQ), "the pick is the draft head's");
    // The input's blocks in shared memory (k 6,144: 24 blocks a lane would not fit in registers).
    constexpr bool SX = shared_x(MB);
    static_assert(!SX || (!PICK && !XQ), "the 6,144-value projections only");
    static_assert(!SX || NB % 4 == 0, "a warp quantizes 4 blocks a pass");
    static_assert(vdr == 2, "two ints of a block a lane");

    // Row `row` of the launch: its weight (XQ: the one whose range holds it) and its row there.
    const auto weight = [&](const int64_t row) -> int {
        int s = 0;
        if constexpr (XQ) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                s += (i + 1 < p.nw && row >= p.ends[i]) ? 1 : 0;
            }
        }
        return s;
    };
    const auto row_src = [&](const int64_t row) -> const uint4 * {
        if constexpr (XQ) {
            const int s = weight(row);
            return (const uint4 *) ((const char *) p.ws[s] + (row - (s > 0 ? p.ends[s - 1] : 0)) * p.lds[s]);
        } else {
            return (const uint4 *) ((const char *) p.w + row * p.w_ld);
        }
    };
    const auto row_dst = [&](const int64_t row) -> float * {
        if constexpr (XQ) {
            const int s = weight(row);
            return p.dsts[s] + (row - (s > 0 ? p.ends[s - 1] : 0));
        } else {
            return p.dst + row;
        }
    };
    (void) weight;

    extern __shared__ uint4 smem[];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    uint4 * buf = smem + warp * CH * R;
    const char * bufc = (const char *) buf;

    const int q = lane % 4;  // ints 2q and 2q + 1 of each block (ggml's kqs = vdr * (t % 4))

    // The input's blocks this lane reads, as ggml's quantize_q8_1 writes them: u the int8 values
    // 8q .. 8q + 7 of block b = lane / 4 + 8m, d8 its scale (F16, as F32). (SX: in shared memory,
    // xs_q and xs_d below.)
    int   u[SX ? 1 : MB][vdr];
    float d8[SX ? 1 : MB];
#pragma unroll
    for (int m = 0; m < (SX ? 0 : MB); ++m) {
        if constexpr (XQ) {
            // The block as quantize_q8_1 wrote it; vec_dot_q8_0_q8_1 reads the same ints and
            // the low half of ds.
            const block_q8_1 * yb = (const block_q8_1 *) p.xq + (lane / 4 + 8 * m);
#pragma unroll
            for (int i = 0; i < vdr; ++i) {
                u[m][i] = get_int_b4(yb->qs, vdr * q + i);
            }
            d8[m] = __low2float(yb->ds);
            continue;
        }
        // Scalar loads: once per kernel, and the input need not be 16-byte aligned.
        const float * xb = p.x + (lane / 4 + 8 * m) * QK8_1;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < QK8_1; ++i) {
            amax = fmaxf(amax, fabsf(xb[i]));
        }
        const float d = amax / 127.0f;
#pragma unroll
        for (int i = 0; i < vdr; ++i) {
            int packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float  xi = xb[4 * (vdr * q + i) + j];
                const int8_t qi = amax == 0.0f ? 0 : roundf(xi / d);
                packed |= ((int) (uint8_t) qi) << (8 * j);
            }
            u[m][i] = packed;
        }
        d8[m] = __half2float(__float2half(d));
    }

    const int64_t gwarp  = (int64_t) blockIdx.x * NW + warp;
    const int64_t nwarps = (int64_t) gridDim.x * NW;

    // A warp step takes R consecutive rows (more loads in flight with R 2).
    uint4 r[R][CPL];
    const auto load = [&](const int64_t row0) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            // (A row past the end is not read: its pointer is only formed.)
            const uint4 * src = row_src(row0 + j);
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH && row0 + j < p.rows) {
                    r[j][i] = __ldg(src + c);
                }
            }
        }
    };
    const int64_t step = nwarps * R;
    // The pick: this warp's best row (-1: none above -FLT_MAX; every lane holds the same sums)
    // and largest value. The tie rank is only needed between equal values.
    float   bv   = -FLT_MAX;
    int64_t brow = -1;
    float   mx   = -INFINITY;
    if (gwarp * R < p.rows) {
        load(gwarp * R);
    }

    // SX: the CTA quantizes the input into shared memory while the first rows load, as ggml's
    // quantize_q8_1 does (quantize.cu, MIT): the block's absolute maximum (exact in any order;
    // here 8 lanes of 4 values each and a butterfly of fmaxf), d = amax / 127 (its F16 rounding
    // is what vec_dot_q8_0_q8_1 reads), q = roundf(x / d), 0 for an all-zero block. A warp takes
    // 4 blocks a pass. xs_q: block b's ints at 4b .. 4b + 3 (int2 each: the ints 2i, 2i + 1 a
    // lane reads, one 8-byte load and no bank conflicts), xs_d: the scales.
    int2 *  xs_q = (int2 *) (smem + NW * R * CH);
    float * xs_d = (float *) (xs_q + 4 * NB);
    if constexpr (SX) {
        const int e = lane % 8;  // values 4e .. 4e + 3 of the block
        for (int b = 4 * warp + lane / 8; b < NB; b += 4 * NW) {
            const float * xb = p.x + b * QK8_1 + 4 * e;
            float xv[4];
            float amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xv[j] = xb[j];
                amax  = fmaxf(amax, fabsf(xv[j]));
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o, WARP_SIZE));
            }
            const float d      = amax / 127.0f;
            int         packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int8_t qi = amax == 0.0f ? 0 : roundf(xv[j] / d);
                packed |= ((int) (uint8_t) qi) << (8 * j);
            }
            ((int *) xs_q)[8 * b + e] = packed;
            if (e == 0) {
                xs_d[b] = __half2float(__float2half(d));
            }
        }
        __syncthreads();
    }
    (void) xs_q;
    (void) xs_d;

    for (int64_t row0 = gwarp * R; row0 < p.rows; row0 += step) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const int c = lane + WARP_SIZE * i;
                if (c < CH) {
                    buf[j * CH + c] = r[j][i];
                }
            }
        }
        __syncwarp();
        if (row0 + step < p.rows) {
            load(row0 + step);
        }

#pragma unroll
        for (int j = 0; j < R; ++j) {
            const int64_t row = row0 + j;
            if (row >= p.rows) {
                break;
            }
            // ggml's per-thread sums: tmp[w] is virtual thread lane + 32 w's.
            float tmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
            for (int m = 0; m < MB; ++m) {
                const block_q8_0 * bq8_0 = (const block_q8_0 *) (bufc + (j * CH * 16 + (lane / 4 + 8 * m) * QB));
                int                v[vdr];
#pragma unroll
                for (int i = 0; i < vdr; ++i) {
                    v[i] = get_int_b2(bq8_0->qs, vdr * q + i);
                }
                if constexpr (SX) {
                    const int2 t     = xs_q[4 * (lane / 4 + 8 * m) + q];
                    const int  uv[2] = { t.x, t.y };
                    tmp[m % 4] += vec_dot_q8_0_q8_1_impl<float, vdr>(v, uv, __half2float(bq8_0->d),
                                                                      xs_d[lane / 4 + 8 * m]);
                } else {
                    tmp[m % 4] += vec_dot_q8_0_q8_1_impl<float, vdr>(v, u[m], __half2float(bq8_0->d), d8[m]);
                }
            }

            // ggml: warp 0's lane adds warps 1, 2, 3 in order, then the warp sum.
            float sum = tmp[0];
            sum += tmp[1];
            sum += tmp[2];
            sum += tmp[3];
            sum = warp_reduce_sum<WARP_SIZE>(sum);
            if (lane == 0) {
                *row_dst(row) = sum;
            }
            if constexpr (PICK) {
                // bv starts at -FLT_MAX: a value at or below it (or NaN) is never taken.
                if (sum > bv) {
                    bv   = sum;
                    brow = row;
                } else if (sum == bv && brow >= 0 && pk_rank(row) < pk_rank(brow)) {
                    brow = row;
                }
                mx = fmaxf(mx, sum);
            }
        }
        __syncwarp();
    }
    if constexpr (PICK) {
        __shared__ float    s_v[NW];
        __shared__ uint32_t s_r[NW];
        __shared__ float    s_m[NW];
        uint32_t br = brow < 0 ? PK_NONE : pk_rank(brow);
        if (lane == 0) {
            s_v[warp] = bv;
            s_r[warp] = br;
            s_m[warp] = mx;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < NW; ++w) {
                if (pk_better(s_v[w], s_r[w], bv, br)) {
                    bv = s_v[w];
                    br = s_r[w];
                }
                mx = fmaxf(mx, s_m[w]);
            }
            uint4 * part = (uint4 *) ((char *) p.pick + PICK_PARTS);
            part[blockIdx.x] = make_uint4(__float_as_uint(bv), br, __float_as_uint(mx), 0);
            if (blockIdx.x == 0) {
                ((pick_header *) p.pick)->count = gridDim.x;
            }
        }
    }
}

// One CTA of 256 threads: the partials' best into dst[0] (argmax_f32's pick) and their largest
// value into the header (soft_max's max, exact in any order: fmaxf skips NaN as ggml's does).
// A CTA per column (blockIdx.y), each on its slice of the buffer.
__global__ void __launch_bounds__(PMAX_T) q8_head_argmax_kernel(const legion_q8_head_argmax p) {
    __shared__ float    s_v[PMAX_T / WARP_SIZE];
    __shared__ uint32_t s_r[PMAX_T / WARP_SIZE];
    __shared__ float    s_m[PMAX_T / WARP_SIZE];
    const int           col  = blockIdx.y;
    const char *        buf  = (const char *) p.pick + col * p.stride;
    const pick_header * h    = (const pick_header *) buf;
    const uint4 *       part = (const uint4 *) (buf + PICK_PARTS);
    const int           t = threadIdx.x, lane = t % WARP_SIZE, warp = t / WARP_SIZE;
    const int           count = (int) h->count;
    float    bv = -FLT_MAX;
    uint32_t br = PK_NONE;
    float    mx = -INFINITY;
    for (int i = t; i < count; i += PMAX_T) {
        const uint4 c = part[i];
        if (pk_better(__uint_as_float(c.x), c.y, bv, br)) {
            bv = __uint_as_float(c.x);
            br = c.y;
        }
        mx = fmaxf(mx, __uint_as_float(c.z));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float    v = __shfl_xor_sync(0xffffffff, bv, o);
        const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
        if (pk_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    if (lane == 0) {
        s_v[warp] = bv;
        s_r[warp] = br;
        s_m[warp] = mx;
    }
    __syncthreads();
    if (t == 0) {
        for (int w = 1; w < PMAX_T / WARP_SIZE; ++w) {
            if (pk_better(s_v[w], s_r[w], bv, br)) {
                bv = s_v[w];
                br = s_r[w];
            }
            mx = fmaxf(mx, s_m[w]);
        }
        p.dst[col] = br == PK_NONE ? -1 : pk_col(br);
        pick_header * hw = (pick_header *) buf;
        hw->max    = mx;
        hw->ticket = 0;  // for q8_head_pmax_kernel, next on the stream
    }
}

// Grid: the device's SM count (ggml's nsm, at most 255), PMAX_T threads: ggml-cuda's
// soft_max_f32_parallelize_cols_single_row's divisor, term for term and in its order, from the
// header's max; the last CTA to finish sums the CTAs' sums as ggml's second block_reduce does
// and writes the pick's probability. Column c is blockIdx.y = c, on its slice of the buffer:
// ggml's kernel loops over the rows of a soft_max with the same grid, each row's sums as one
// row's, so a column's CTAs here sum its terms as ggml's do for that row.
__global__ void __launch_bounds__(PMAX_T) q8_head_pmax_kernel(const legion_q8_head_pmax p) {
    __shared__ float shared_vals_sum[32];
    __shared__ float shared_vals_all[32];
    __shared__ bool  last;
    const int     c    = blockIdx.y;
    char *        slot = (char *) p.buf + c * p.stride;
    pick_header * h    = (pick_header *) slot;
    float *       sums = (float *) (slot + PICK_SUMS);
    const float * x    = p.x + c * p.x_ld;

    const int   tid               = threadIdx.x;
    const int   col_start         = blockIdx.x * blockDim.x + tid;
    const int   n_elem_per_thread = 4;
    const int   step_size         = gridDim.x * blockDim.x;
    const float local_max         = h->max;

    float local_vals[n_elem_per_thread];
    float tmp_expf = 0.0f;
    for (int col = col_start; col < p.n;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.n ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.n) {
                const float tmp = expf(local_vals[i] - local_max);
                tmp_expf += tmp;
            }
        }
        col += step_size * n_elem_per_thread;
    }
    tmp_expf = block_reduce<block_reduce_method::SUM>(tmp_expf, shared_vals_sum);
    if (tid == 0) {
        sums[blockIdx.x] = tmp_expf;
        __threadfence();
        last = atomicAdd(&h->ticket, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    float all = tid < (int) gridDim.x ? __ldcg(sums + tid) : 0.0f;
    all = block_reduce<block_reduce_method::SUM>(all, shared_vals_all);
    if (tid == 0) {
        const int a = p.pick[c];
        // ggml's soft_max stores expf(x - max) and then divides it by the sum.
        p.dst[c] = a < 0 ? __int_as_float(0x7fc00000) : expf(__ldcg(x + a) - local_max) / all;
        h->ticket = 0;
    }
}

template <int MB, int NW, int R, bool PICK, bool XQ>
int launch(const legion_q8_head & p, cudaStream_t s, int ctas_cap) {
    const auto kernel = head_mv<MB, NW, R, PICK, XQ>;
    const int  smem   = NW * R * 17 * MB * 16 + shared_x_bytes(MB);
    static std::atomic<int8_t> opt_in[GGML_CUDA_MAX_DEVICES];
    static std::atomic<int>    per_sm[GGML_CUDA_MAX_DEVICES];
    if (smem > 48 * 1024 && !legion_device::smem_opt_in(opt_in, kernel, smem)) {
        return 1;
    }
    const int dev = legion_device::current();
    const int sms = legion_device::sm_count();
    if (dev < 0 || sms <= 0) {
        return 1;
    }
    int fit = per_sm[dev].load(std::memory_order_relaxed);
    if (fit == 0) {
        const cudaError_t before = cudaPeekAtLastError();
        // The pick kernel takes at most the plain kernel's CTAs an SM: its own register count can
        // differ and fit more, which is slower (more CTAs share the SM for the same rows). (XQ: its
        // own count.)
        int plain = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, NW * WARP_SIZE, smem) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plain, head_mv<MB, NW, R, false, XQ>, NW * WARP_SIZE,
                                                          smem) != cudaSuccess ||
            fit <= 0) {
            legion_device::clear_own_error(before);
            return 1;
        }
        if (plain > 0 && plain < fit) {
            fit = plain;
        }
        per_sm[dev].store(fit, std::memory_order_relaxed);
    }
    if (ctas_cap > 0 && ctas_cap < fit) {
        fit = ctas_cap;
    }
    const int64_t need = (p.rows + NW * R - 1) / (NW * R);
    const int     grid = (int) (need < (int64_t) sms * fit ? need : (int64_t) sms * fit);
    if (PICK && grid > p.pick_cap) {
        return 1;
    }
    head_mv<MB, NW, R, PICK, XQ><<<grid, NW * WARP_SIZE, smem, s>>>(p);
    return 0;
}

template <int MB, int R, bool PICK, bool XQ>
int launch_warps(const legion_q8_head & p, cudaStream_t s, int warps, int ctas) {
    switch (warps) {
        case 4:
            return launch<MB, 4, R, PICK, XQ>(p, s, ctas);
        case 8:
            return launch<MB, 8, R, PICK, XQ>(p, s, ctas);
        case 16:
            return launch<MB, 16, R, PICK, XQ>(p, s, ctas);
        default:
            return 1;
    }
}

template <int MB, bool PICK, bool XQ>
int launch_rows(const legion_q8_head & p, cudaStream_t s, int warps, int ctas, int rows) {
    return rows == 2 ? launch_warps<MB, 2, PICK, XQ>(p, s, warps, ctas) :
                       launch_warps<MB, 1, PICK, XQ>(p, s, warps, ctas);
}

template <int MB>
int launch_variant(const legion_q8_head & p, cudaStream_t s, int warps, int ctas, int rows) {
    if constexpr (shared_x(MB)) {
        // The 6,144-value projections: neither Q8_1 input nor the pick.
        return p.xq != nullptr || p.pick != nullptr ? 1 : launch_rows<MB, false, false>(p, s, warps, ctas, rows);
    } else {
        if (p.xq != nullptr) {
            return launch_rows<MB, false, true>(p, s, warps, ctas, rows);
        }
        return p.pick != nullptr ? launch_rows<MB, true, false>(p, s, warps, ctas, rows) :
                                   launch_rows<MB, false, false>(p, s, warps, ctas, rows);
    }
}

}  // namespace

// The draft pick's kernels as the CUDA runtime knows them (their host stubs), for Rust's
// cudaLaunchKernel (see gate_q8.cu for why they are pointers to anonymous-namespace kernels).
extern "C" const void * const legion_q8_head_argmax_kernel = reinterpret_cast<const void *>(&q8_head_argmax_kernel);
extern "C" const void * const legion_q8_head_pmax_kernel   = reinterpret_cast<const void *>(&q8_head_pmax_kernel);

int legion_q8_head_force(int warps, int ctas, int rows) {
    if (warps == 0) {
        forced.store(0, std::memory_order_relaxed);
        return 0;
    }
    if ((warps != 4 && warps != 8 && warps != 16) || ctas < 0 || ctas > 255 || rows < 1 || rows > 2) {
        return 1;
    }
    forced.store(warps | ctas << 8 | rows << 16, std::memory_order_relaxed);
    return 0;
}

int legion_q8_head_supported(void) {
    return legion_device::compute_capability() / 10 == 7 ? 1 : 0;
}

int legion_q8_head_launch(const legion_q8_head * p, void * stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (p->rows <= 0 || legion_q8_head_supported() == 0) {
        return 1;
    }
    if (p->xq != nullptr) {
        if (p->pick != nullptr || p->nw < 1 || p->nw > 4 || ((uintptr_t) p->xq) % 4 != 0 ||
            p->ends[p->nw - 1] != p->rows) {
            return 1;
        }
        for (int i = 0; i < p->nw; ++i) {
            if (p->lds[i] % 16 != 0 || p->lds[i] < (int64_t) p->k / 32 * QB || ((uintptr_t) p->ws[i]) % 16 != 0 ||
                p->ends[i] <= (i > 0 ? p->ends[i - 1] : 0)) {
                return 1;
            }
        }
    } else if (p->w_ld % 16 != 0 || p->w_ld < (int64_t) p->k / 32 * QB || ((uintptr_t) p->w) % 16 != 0) {
        return 1;
    }
    const int f     = forced.load(std::memory_order_relaxed);
    const bool sx    = p->k == 6144;
    const int  warps = f != 0 ? (f & 0xff) : (sx ? SX_WARPS : p->xq != nullptr ? XQ_WARPS : 4);
    const int  ctas  = (f >> 8) & 0xff;
    const int  rows  = f != 0 ? f >> 16 : (sx ? SX_ROWS : p->xq != nullptr ? XQ_ROWS : 2);
    int r = 1;
    switch (p->k) {
        case 1024:
            r = launch_variant<4>(*p, s, warps, ctas, rows);
            break;
        case 2560:
            r = launch_variant<10>(*p, s, warps, ctas, rows);
            break;
        case 6144:
            r = launch_variant<24>(*p, s, warps, ctas, rows);
            break;
        default:
            return 1;
    }
    if (r != 0) {
        return r;
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}
