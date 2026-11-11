// The target's BF16 LM head in verify steps of many sequences (#1181): `w @ x` for a BF16 head
// [m rows of k] and 49-128 F32 columns, the final logit soft-cap and the argmax of every column in
// one pass over the weights. ggml-cuda runs that head as cuBLAS's BF16 GEMM (x converted to BF16
// first), then the soft-cap kernel and argmax_f32, each reading the logits again; on a GB10 at
// 128 columns cuBLAS picks a 128 x 208 tile (80 of its 208 columns padding) and streams Gemma 4
// 31B's 2.8 GB head at ~206 GB/s.
//
// Here (src/model/core/head_mm.rs launches the kernels from Rust):
// - head_mm_x16: x rounded to BF16 (round to nearest, as ggml's conversion) into the stream's
//   scratch, the columns past n zero;
// - head_mm_gemm<NT> (sm80/head_mm.cuh): one CTA per 128 rows, 16 * NT columns, 8 warps of 32
//   rows x 8 * NT columns (NT 4, 6 or 8), the weights and x staged by cp.async through 3
//   shared-memory stages of 64 k values. Every output is one chain of m16n8k16 BF16 products
//   with F32 sums over k = 0, 16, 32, ... from 0, the order of cuBLAS's kernel there (checked
//   bit for bit by `legion head-mm-bench --check`); then the soft-cap as ggml's fused
//   softcap_f32 computes it (tanhf(scale * v) * cap), written to the logits, and the CTA's best
//   (value, tie rank) of each column into the stream's scratch;
// - head_mm_pick: one CTA per column joins the parts: argmax_f32's pick, ties included (its tie
//   order, head_pick.cuh).

#include "head_pick.cuh"

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

extern "C" {
struct legion_head_mm {
    const __nv_bfloat16 * w;  // [m rows][k], w_ld values apart
    const float * x;          // [n][k], x_ld values apart
    float * dst;              // logits [n][m], dst_ld values apart
    __nv_bfloat16 * xb;       // scratch: [16 * NT][k] BF16
    uint2 * part;             // scratch: [n][parts] (value bits, tie rank)
    int32_t * argmax;         // [n]
    int64_t w_ld, x_ld, dst_ld;
    int64_t m, k;
    int32_t n;      // columns, 1 to 16 * NT
    int32_t ncols;  // 16 * NT: the columns the GEMM computes (x past n is zero)
    int32_t parts;  // m / HM_ROWS
    float scale;    // soft-cap: tanhf(scale * v) * cap where cap > 0; v as it is otherwise
    float cap;
};
}

namespace {

constexpr int HM_THREADS = 256;  // 8 warps: 4 along the rows, 2 along the columns

// The CTA tile: rows per CTA, k values per stage (128-byte rows), shared-memory stages. On a GB10
// at 128 columns 128 x 64 in 3 stages streams the head at ~205 GB/s, 256 x 32 in 4 at ~180.
constexpr int HM_ROWS = 128;
constexpr int HM_BK = 64;
constexpr int HM_STAGES = 3;

// Shared memory of head_mm_gemm<NT>: per stage the weights' rows and x's 16 * NT rows of
// 128 bytes.
constexpr int hm_smem(int nt) {
    return HM_STAGES * (HM_ROWS + 16 * nt) * HM_BK * 2;
}

// A row of 8 16-byte chunks; chunk c of row r sits at c ^ (r & 7), so the 8 rows an ldmatrix
// phase reads hit 32 different banks.
__device__ __forceinline__ int hm_swz(int r, int c) {
    return r * 8 + (c ^ (r & 7));
}

// The better of two (value, tie rank) candidates, argmax_f32's order; a value of -FLT_MAX or
// below (or NaN) is never a candidate.
__device__ __forceinline__ void hm_take(float v, uint32_t r, float & bv, uint32_t & br) {
    if (v > -FLT_MAX && pk_better(v, r, bv, br)) {
        bv = v;
        br = r;
    }
}

// Grid ceil(ncols * k / 2 / 256): x's columns rounded to BF16 pairs, zero past n.
__global__ void __launch_bounds__(256) head_mm_x16(const legion_head_mm p) {
    const int64_t k2 = p.k / 2;
    const int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x;
    if (i >= (int64_t) p.ncols * k2) {
        return;
    }
    const int64_t j = i / k2, c = i % k2;
    const float2 v = j < p.n ? ((const float2 *) (p.x + j * p.x_ld))[c] : make_float2(0.0f, 0.0f);
    ((__nv_bfloat162 *) p.xb)[i] = __float22bfloat162_rn(v);
}

// head_mm_gemm<NT> (sm_80+: cp.async, ldmatrix, m16n8k16).
#include "sm80/head_mm.cuh"

// Grid n, 256 threads: column blockIdx.x's parts joined, its argmax (-1 if none).
__global__ void __launch_bounds__(256) head_mm_pick(const legion_head_mm p) {
    __shared__ float s_v[8];
    __shared__ uint32_t s_r[8];
    const int col = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const uint2 * part = p.part + (int64_t) col * p.parts;
    float bv = -FLT_MAX;
    uint32_t br = PK_NONE;
    for (int i = t; i < p.parts; i += 256) {
        const uint2 q = part[i];
        const float v = __uint_as_float(q.x);
        if (pk_better(v, q.y, bv, br)) {
            bv = v;
            br = q.y;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffff, bv, o);
        const uint32_t r = __shfl_xor_sync(0xffffffff, br, o);
        if (pk_better(v, r, bv, br)) {
            bv = v;
            br = r;
        }
    }
    if (lane == 0) {
        s_v[warp] = bv;
        s_r[warp] = br;
    }
    __syncthreads();
    if (t == 0) {
        for (int w = 1; w < 8; w++) {
            if (pk_better(s_v[w], s_r[w], bv, br)) {
                bv = s_v[w];
                br = s_r[w];
            }
        }
        p.argmax[col] = br == PK_NONE ? -1 : pk_col(br);
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernelExC
// (src/model/core/head_mm.rs): the conversion, the GEMM for NT = 4, 6 and 8, the pick.
extern "C" const void * const legion_head_mm_kernels[5] = {
    reinterpret_cast<const void *>(&head_mm_x16),
    reinterpret_cast<const void *>(&head_mm_gemm<4>),
    reinterpret_cast<const void *>(&head_mm_gemm<6>),
    reinterpret_cast<const void *>(&head_mm_gemm<8>),
    reinterpret_cast<const void *>(&head_mm_pick),
};

// Dynamic shared memory of the GEMM kernels, NT = 4, 6, 8.
extern "C" const int legion_head_mm_smem[3] = { hm_smem(4), hm_smem(6), hm_smem(8) };
