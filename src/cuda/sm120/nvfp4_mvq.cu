// NVFP4 dense matvecs of 1-8 columns with ggml's bits on tensor cores (#1133): ggml-cuda's
// quantize_q8_1 + mul_mat_vec_q<GGML_TYPE_NVFP4> pair (mmvq.cu, vecdotq.cuh, quantize.cu, MIT)
// for `w @ x` with a 2D weight, optionally times a one-value scale (ggml's `mul` after it), bit
// for bit. The host side, the shapes and the CPU reference are in src/model/core/nvfp4_mvq.rs,
// which launches the kernels from `legion_nvfp4_mvq_kernels` with cudaLaunchKernelExC.
//
// ggml's MMVQ spends about five instructions per weight byte at 8 columns on the CUDA cores: the
// E2M1 table lookups (byte permutes), four dp4a per 16-value sub-block and column, then the
// float scale math. Here the integer part runs on the tensor cores and the float part replays
// ggml's order exactly:
//
// - The integer sum of each 16-value sub-block, row and column (ggml's `sumi`, four dp4a) comes
//   from an F16 mma.sync m16n8k16: the weights as F16 E2M1 values (cvt.rn.f16x2.e2m1x2, one
//   instruction per byte), the activations as F16 2 * q of their Q8_1 values (exact; stored as
//   bytes, expanded in registers: the activations are read again for every region). Every
//   product is an integer of at most 1524 in magnitude and every sum one of at most 24384, so
//   the F32 accumulation is exact: C is ggml's `sumi`, as a float. Inside an MMA the K order is
//   free when A and B agree, so a thread takes bytes 2t and 2t + 1 of a sub-block (values 2t,
//   2t + 8, 2t + 1, 2t + 9) and `quantize` stores the activations in that order.
// - One MMA covers two sub-blocks (an even one and the odd one after it) of 4 weight rows: A is
//   the activations, 8 columns of the even sub-block in rows 0-7 and of the odd one in rows
//   8-15; B is the weights, the 4 rows' even sub-block in columns 0-3 and their odd one in 4-7.
//   Half of C pairs a sub-block with itself (the sums wanted); the other half is unused.
// - The float part is ggml's: per (row, column) and per ggml thread (kbx, half h), sum = 0,
//   then sum += d * sumi for the half's two sub-blocks, d = ue4m3 / 2 * yd (exact; yd stored
//   halved, so d = ue4m3 * yd), added to the thread's partial sum in its K order (kbx = r,
//   r + blocks_per_iter, ...); then the partial sums of ggml's warps 1.. added to warp 0's in
//   order, then warp_reduce_sum's butterfly over the 32 lanes (lane 2 * (r % 16) + h). The
//   butterfly is a fixed tree (each add commutes): warp w of the CTA's 8 takes the residues
//   r = w + 8P, so the tree's first level (lanes 16 apart: r % 16 bit 3, P's low bit) is in
//   each thread and the rest (r % 16 bits 2-0, then h) across the warps in shared memory.
//   ggml runs 4 warps (blocks_per_iter 64) up to 4 columns, 2 (32) at 5-8: the template's NWG.
// - The weights stream in whole rows: a row is contiguous, and so are the R rows a CTA works
//   on at once, so each SM reads long linear runs (16 rows x 576-byte pieces at a time read
//   about 20% slower on a GB10). One CTA per SM walks its regions of R rows, the CTAs side by
//   side (region j of CTA b at rows (j G + b) R, the last partial round split evenly); a row
//   too long for the ring goes in p.chunks pieces along K. A ring of ns piece slots in shared
//   memory takes each piece by one bulk copy (cp.async.bulk, completing on the slot's
//   mbarrier), so the next pieces are in flight while a chunk is computed, and a chunk's
//   activations are in registers before its weights are waited for. All of a row's chains
//   end inside the region, which then writes its outputs.
// - The epilogue: times the scale when there is one (ggml's `mul`, one rounding).
// - `quantize` is ggml's quantize_q8_1 value for value (one thread per value, a warp per
//   32-value block, amax by ggml's warp reduction, d = amax / 127, q = roundf(x / d)) into
//   device-global scratch in the layout above: the op's two kernels run back to back on the
//   compute stream, as ggml's pair shares its pool buffer. One scratch per device is enough
//   because only one stream of a device ever runs the op: it takes NVFP4 weights, which only
//   the target model's backend holds (drafters are BF16, Q8_0 or Q4 and run on their own
//   backend between the target's graphs), ggml's concurrent branch streams
//   (GGML_CUDA_GRAPH_OPT) stay off, and tensor parallelism runs one rank per device. A second
//   stream running it at the same time would need its own scratch.
// - On sm_90+ every kernel waits for the kernel ahead of it (griddepcontrol.wait, ggml's
//   ggml_cuda_pdl_sync) before it reads activations, and lets the next one launch early
//   (ggml_cuda_pdl_lc), as ggml-cuda launches its kernels on a GB10. The weights are no other
//   kernel's output: their first rows go out before the wait.
//
// The matvec needs the sm_120a/sm_121a conversion (the FP4 code of nvfp4_a4.cu); elsewhere it
// is a trap, and the host takes the op only where nvfp4_a4's probe says the device runs that
// code.

#include "ggml-cuda/common.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define LEGION_MVQ_MMA 1
#endif

extern "C" {
struct legion_nvfp4_mvq {
    const float * x;        // F32 [n][k]
    int64_t       x_nb1;    // floats between x's columns
    const void *  w;        // NVFP4 [rows][k / 64]
    int64_t       w_nb1;    // bytes between rows
    float *       dst;      // F32 [n][rows]
    int64_t       dst_nb1;  // floats between dst's columns
    const float * scale;    // F32 [1], or null
    int32_t       k;        // a multiple of 256, at most MAX_K
    int32_t       rows;     // a multiple of 4
    int32_t       n;        // 1-8
    int32_t       ns;       // piece slots of the ring (more than 16)
    int32_t       chunks;   // pieces a row (each whole 16-byte units)
    int32_t       exp;      // EXPERIMENT knobs (temporary)
};
}

namespace {

#include "../sm90/tma.cuh"

constexpr int QUANTIZE_THREADS = 256;  // ggml's CUDA_QUANTIZE_BLOCK_SIZE

// src/model/core/nvfp4_mvq.rs's MAX_K and MAX_N.
constexpr int MAX_K = 65536;
constexpr int MAX_N = 8;

// The activations as bytes q + 128: [column][kbx][MMA thread t of 4][sub-block][4 values], 16
// bytes a (column, kbx, t): one uint4 load (expanded to F16 2 * q in registers).
__device__ uint4 scratch_xq[MAX_N * MAX_K / 16];
// The Q8_1 scales yd / 2: [kbx][Q8 block of the kbx][column].
__device__ float scratch_yd[MAX_K / 64 * 2 * MAX_N];

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// ggml's quantize_q8_1 for one value per thread: grid (ceil(k / 256), n).
__global__ void __launch_bounds__(QUANTIZE_THREADS) quantize(const legion_nvfp4_mvq p) {
    pdl_launch_dependents();
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= p.k) {
        return;
    }
    const int t = blockIdx.y;

    pdl_wait();
    const float xi = p.x[t * p.x_nb1 + i0];
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    // Value e of sub-block s of block kbx goes to thread (e % 8) / 2 of the MMA, at its slot
    // (e % 2) * 2 + e / 8 (see the header).
    const int kbx = i0 >> 6;
    const int s   = (i0 >> 4) & 3;
    const int e   = i0 & 15;
    const int th  = (e & 7) >> 1;
    const int pos = ((e & 1) << 1) | (e >> 3);
    uint8_t * xq = reinterpret_cast<uint8_t *>(scratch_xq);
    xq[(((int64_t) t * (p.k >> 6) + kbx) * 4 + th) * 16 + s * 4 + pos] = (uint8_t) (q + 128);

    if ((i0 & (QK8_1 - 1)) > 0) {
        return;
    }
    // ggml's ds.x (make_half2(d, sum)): d rounded to F16; halved here, exactly.
    const float yd = __half2float(__float2half_rn(d)) * 0.5f;
    scratch_yd[(kbx * 2 + ((i0 >> 5) & 1)) * MAX_N + t] = yd;
}

#ifdef LEGION_MVQ_MMA
// Two UE4M3 scales (the low byte, then the high one) as floats, as ggml_cuda_ue4m3_to_fp32
// without its halving (codes 0x7F and 0xFF give 0).
__device__ __forceinline__ void ue4m3x2(uint32_t d, float & lo, float & hi) {
    const uint32_t nan = ((d & 0x7f7fu) + 0x0101u) & 0x8080u;
    d &= ~((nan >> 7) * 0xffu);
    const __half2 h  = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (d & 0xffffu), __NV_E4M3));
    const float2  f  = __half22float2(h);
    lo = f.x;
    hi = f.y;
}

// Two bytes of E2M1 pairs as F16 pairs: the low byte into `lo`, the high one into `hi`, each
// byte's low nibble in the low half.
__device__ __forceinline__ void e2m1x4(uint32_t v, uint32_t & lo, uint32_t & hi) {
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n}"
        : "=r"(lo), "=r"(hi)
        : "r"(v));
}

// Four bytes q + 128 (q in -127..127) as F16 pairs 2 q: bytes 0, 1 into `lo`, 2, 3 into `hi`
// (exact: 0x68 above a byte b is the F16 2048 + 2 b, less 2304).
__device__ __forceinline__ void q8x4(uint32_t v, uint32_t & lo, uint32_t & hi) {
    const uint32_t e = 0x68686868u, bias = 0x68806880u;
    uint32_t       a, b;
    asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(a) : "r"(v), "r"(e));
    asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(b) : "r"(v), "r"(e));
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(bias));
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(hi) : "r"(b), "r"(bias));
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

__device__ __forceinline__ uint32_t lds_u16(uint32_t addr) {
    uint16_t v;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ void bulk_pf_l2(const void * src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src), "r"(bytes) : "memory");
}

// `bytes` (a multiple of 16) from global `src` to shared `dst` (both 16-byte aligned) by the
// TMA unit, completing on `bar`.
__device__ __forceinline__ void bulk_g2s(void * dst, const void * src, uint32_t bytes, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
            legion_tma::saddr(dst)),
        "l"(src), "r"(bytes), "r"(legion_tma::saddr(bar))
        : "memory");
}
#endif

constexpr int MV_WARPS   = 8;
constexpr int MV_THREADS = MV_WARPS * WARP_SIZE;
constexpr int MV_R       = 16;  // rows of a region
constexpr int MV_ITERS   = 8;   // a chunk spans at most MV_ITERS / NWG steps of blocks_per_iter

// `w @ x` with ggml's MMVQ order for NWG of its warps (4: 1-4 columns, 2: 5-8). G CTAs (one per
// SM) take regions of up to MV_R rows (see `start` below), each row in p.chunks pieces along K
// (a chunk of the region's rows at a time, the partial sums carried from chunk to chunk: every
// chain keeps its K order).
// Thread (g, t) of a warp (lane 4g + t) takes column g, the rows 2 (t % 2) and 2 (t % 2) + 1 of
// each group, and half h = t / 2 of each block: the C entries of its own sub-block parity, then
// the other parity's from lane t ^ 2. ggml's warp l (of NWG) walks the residues 16 l + 0..15
// (its lane 2 (r % 16) + h), each over the blocks r + BPI i (BPI = 16 NWG, ggml's
// blocks_per_iter); warp w here holds the residues r = w + 8P, a partial sum for each.
//
// Dynamic shared memory: p.ns piece slots of piece_bytes + 16 (the 4 rows of a group land in
// different banks), the partial sums [warp][h][MV_R][column], p.ns mbarriers. Pieces are read
// in order (region, chunk, row), piece q into slot q % ns.
template <int NWG>
__global__ void __launch_bounds__(MV_THREADS, 1) mv(const legion_nvfp4_mvq p) {
    pdl_launch_dependents();
#ifdef LEGION_MVQ_MMA
    using namespace legion_tma;
    constexpr int R   = MV_R;
    constexpr int BPI = 16 * NWG;
    constexpr int NP  = BPI / MV_WARPS;
    constexpr int G4  = R / 4;
    constexpr int IT  = MV_ITERS / NWG;  // BPI steps a chunk spans at most (the host's plan)
    static_assert(NP % 2 == 0, "residues in pairs of ggml lanes 16 apart");
    extern __shared__ __align__(16) uint8_t smem[];

    const int bpr    = p.k >> 6;
    const int cb     = bpr / p.chunks;  // blocks of a piece
    const int pbytes = cb * 36;
    const int pitch  = pbytes + 16;
    const int ns     = p.ns;
    uint8_t *  ring = smem;
    float *    part = reinterpret_cast<float *>(smem + ns * pitch);
    uint64_t * full = reinterpret_cast<uint64_t *>(part + MV_WARPS * 2 * R * 8);

    const int w    = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int hp   = t >> 1;        // the thread's half and sub-block parity
    const int ra   = 2 * (t & 1);   // its first row of a group

    // Regions: rounds of G R rows, region j of CTA b at rows (j G + b) R, so the CTAs stream
    // neighbouring rows; the last, partial round in nearly equal runs of whole groups.
    const int  G      = gridDim.x;
    const bool contig = p.exp & 16;
    const int  whole  = contig ? 0 : p.rows / (G * R);
    const int  tail   = whole * G * R;
    const int  gl     = (p.rows - tail) >> 2;
    const int  t0     = tail + 4 * (int) (((int64_t) blockIdx.x * gl) / G);
    const int  t1     = tail + 4 * (int) (((int64_t) (blockIdx.x + 1) * gl) / G);
    const int  nreg   = contig ? (t1 - t0 + R - 1) / R : whole + (t1 > t0);
    auto start = [&](int j) { return contig ? t0 + j * R : j < whole ? (j * G + (int) blockIdx.x) * R : t0; };
    auto rows  = [&](int j) { return contig ? min(R, t1 - t0 - j * R) : j < whole ? R : t1 - t0; };
    const int pfd = p.exp & 15;
    auto prefetch_region = [&](int j) {
        if (pfd == 0 || j >= nreg) {
            return;
        }
        const int s = start(j), n = rows(j);
        if (p.exp & 128) {
            for (int i = 0; i < n; ++i) {
                bulk_pf_l2((const uint8_t *) p.w + (int64_t) (s + i) * p.w_nb1, (uint32_t) bpr * 36);
            }
        } else {
            bulk_pf_l2((const uint8_t *) p.w + (int64_t) s * p.w_nb1, (uint32_t) ((n - 1) * p.w_nb1 + bpr * 36));
        }
    };

    // The producer (thread 0): the next piece to issue, as (region, its start and rows, chunk,
    // row).
    int pq = 0, pj = 0, preg = start(0), prr = rows(0), pc = 0, pi = 0;
    auto issue_next = [&]() {
        const int s = pq % ns;
        bar_expect_tx(full + s, pbytes);
        bulk_g2s(ring + s * pitch, (const uint8_t *) p.w + (int64_t) (preg + pi) * p.w_nb1 + pc * pbytes,
                 pbytes, full + s);
        ++pq;
        if (++pi == prr) {
            pi = 0;
            if (++pc == p.chunks) {
                pc = 0;
                ++pj;
                preg = start(pj);
                prr  = rows(pj);
                prefetch_region(pj + pfd);
            }
        }
    };

    if (threadIdx.x == 0) {
        for (int s = 0; s < ns; ++s) {
            bar_init(full + s, 1);
        }
        bar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int d = 1; d <= pfd; ++d) {
            prefetch_region(d);
        }
        while (pq < ns && pj < nreg) {
            issue_next();
        }
    }

    pdl_wait();

    const bool    col_ok = g < p.n;
    const uint4 * xq     = scratch_xq + (int64_t) g * bpr * 4 + t;
    const float * yd     = scratch_yd + hp * MAX_N + g;
    const bool    scaled = p.scale != nullptr;
    const float   sc     = scaled ? *p.scale : 1.0f;

    uint4 xv[IT][NP];  // a chunk's activations (below)
    float yv[IT][NP];
    int   qb = 0;      // the region's first piece
    for (int j = 0; j < nreg; ++j) {
        const int reg = start(j);
        const int rr  = rows(j);
        const int ng = rr >> 2;

        float acc[NP][G4][2];
#pragma unroll
        for (int P = 0; P < NP; ++P) {
#pragma unroll
            for (int c = 0; c < G4; ++c) {
                acc[P][c][0] = 0.0f;
                acc[P][c][1] = 0.0f;
            }
        }

        for (int ch = 0; ch < p.chunks; ++ch) {
            const int q0   = qb + ch * rr;
            const int lo   = ch * cb;
            const int hi   = lo + cb;
            const int base = lo - lo % BPI;
            // The chunk's activations, loaded before its weights are waited for (the warp's
            // blocks of the chunk: kbx = base + BPI it + w + 8P; bytes 128 are zeros); whole
            // rows once.
            if (p.chunks > 1 || j == 0) {
#pragma unroll
                for (int it = 0; it < IT; ++it) {
#pragma unroll
                    for (int P = 0; P < NP; ++P) {
                        const int  kbx = base + BPI * it + w + MV_WARPS * P;
                        const bool ok  = col_ok && kbx >= lo && kbx < hi;
                        xv[it][P]      = ok ? xq[(int64_t) kbx * 4] : make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
                        yv[it][P]      = ok ? yd[kbx * 2 * MAX_N] : 0.0f;
                    }
                }
            }
            if ((p.exp & 32) && p.chunks > 1) {
                const int lo2 = ((ch + 1) % p.chunks) * cb;
                const int nl  = cb / 2;  // 128-byte lines of a column's chunk
                for (int l = threadIdx.x; l < (p.n + 1) * nl; l += MV_THREADS) {
                    const int col = l / nl, li = l % nl;
                    const char * a = col < p.n
                        ? (const char *) (scratch_xq + ((int64_t) col * bpr + lo2) * 4) + li * 128
                        : (const char *) (scratch_yd + lo2 * 2 * MAX_N) + li * 128;
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(a));
                }
            }
            for (int i = 0; i < rr; ++i) {
                bar_wait(full + (q0 + i) % ns, (uint32_t) ((q0 + i) / ns) & 1);
            }
            // Shared addresses: the B fragment's row (g % 4 of each group) at its sub-block
            // (g / 4: the parity) and bytes 2t, 2t + 1; the scale words' rows ra, ra + 1 at the
            // half's two scales.
            uint32_t rb[G4], rs0[G4], rs1[G4];
#pragma unroll
            for (int c = 0; c < G4; ++c) {
                rb[c]  = saddr(ring + ((q0 + 4 * c + (g & 3)) % ns) * pitch) + 4 + 8 * (g >> 2) + 2 * t;
                rs0[c] = saddr(ring + ((q0 + 4 * c + ra) % ns) * pitch) + 2 * hp;
                rs1[c] = saddr(ring + ((q0 + 4 * c + ra + 1) % ns) * pitch) + 2 * hp;
            }
#pragma unroll
            for (int it = 0; it < IT; ++it) {
#pragma unroll
                for (int P = 0; P < NP; ++P) {
                    const int kbx = base + BPI * it + w + MV_WARPS * P;
                    if (kbx < lo || kbx >= hi || (p.exp & 64)) {
                        continue;
                    }
                    // A: sub-block s's values 2t, 2t + 8 (k 2t, 2t + 1) in a[s][0], 2t + 1,
                    // 2t + 9 (k 2t + 8, 2t + 9) in a[s][1].
                    uint32_t    a[4][2];
                    const float y = yv[it][P];
                    q8x4(xv[it][P].x, a[0][0], a[0][1]);
                    q8x4(xv[it][P].y, a[1][0], a[1][1]);
                    q8x4(xv[it][P].z, a[2][0], a[2][1]);
                    q8x4(xv[it][P].w, a[3][0], a[3][1]);
                    const uint32_t o = (kbx - lo) * 36;
#pragma unroll
                    for (int c = 0; c < G4; ++c) {
                        if (c >= ng) {
                            break;
                        }
                        // MMA 0: sub-blocks 0 and 1; MMA 1: 2 and 3.
                        uint32_t b00, b01, b10, b11;
                        e2m1x4(lds_u16(rb[c] + o), b00, b01);
                        e2m1x4(lds_u16(rb[c] + o + 16), b10, b11);
                        float c0[4], c1[4];
                        mma16816(c0, a[0][0], a[1][0], a[0][1], a[1][1], b00, b01);
                        mma16816(c1, a[2][0], a[3][0], a[2][1], a[3][1], b10, b11);
                        // Own parity: rows ra, ra + 1 of sub-block 2 * MMA + hp (C rows g or
                        // g + 8, columns 2t, 2t + 1). The half's MMA is hp; the other one goes
                        // to lane t ^ 2, whose half it is.
                        const float mine0 = hp ? c1[2] : c0[0];
                        const float mine1 = hp ? c1[3] : c0[1];
                        const float send0 = hp ? c0[2] : c1[0];
                        const float send1 = hp ? c0[3] : c1[1];
                        const float recv0 = __shfl_xor_sync(0xffffffffu, send0, 2);
                        const float recv1 = __shfl_xor_sync(0xffffffffu, send1, 2);
                        // Sub-blocks 2 hp (even: lane t & 1's) and 2 hp + 1.
                        const float s_lo[2] = {hp ? recv0 : mine0, hp ? recv1 : mine1};
                        const float s_hi[2] = {hp ? mine0 : recv0, hp ? mine1 : recv1};
                        float dw[2][2];
                        ue4m3x2(lds_u16(rs0[c] + o), dw[0][0], dw[0][1]);
                        ue4m3x2(lds_u16(rs1[c] + o), dw[1][0], dw[1][1]);
#pragma unroll
                        for (int rr2 = 0; rr2 < 2; ++rr2) {
                            // vec_dot_nvfp4_q8_1's half hp: sub-blocks 2 hp and 2 hp + 1.
                            float sum = 0.0f;
                            {
                                const float d = dw[rr2][0] * y;
                                sum += d * s_lo[rr2];
                            }
                            {
                                const float d = dw[rr2][1] * y;
                                sum += d * s_hi[rr2];
                            }
                            acc[P][c][rr2] += sum;
                        }
                    }
                }
            }
            // Every warp's reads of the chunk's slots are done (and, at the region's first
            // chunk, the previous region's outputs are read out of the partial sums).
            __syncthreads();
            if (threadIdx.x == 0) {
                const int lim = q0 + rr + ns;
                while (pq < lim && pj < nreg) {
                    issue_next();
                }
            }
        }
        qb += p.chunks * rr;

#pragma unroll
        for (int c = 0; c < G4; ++c) {
            if (c >= ng) {
                break;
            }
#pragma unroll
            for (int rr2 = 0; rr2 < 2; ++rr2) {
                // ggml's cross-warp adds (warp l = P / 2: warp 0's sum, then 1.., in order),
                // then the butterfly's lanes 16 apart (P's low bit).
                float v[2];
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    v[b] = acc[b][c][rr2];
#pragma unroll
                    for (int l = 1; l < NWG; ++l) {
                        v[b] += acc[2 * l + b][c][rr2];
                    }
                }
                part[((w * 2 + hp) * R + 4 * c + ra + rr2) * 8 + g] = v[0] + v[1];
            }
        }
        __syncthreads();
        if (threadIdx.x < rr * 8) {
            const int row = threadIdx.x >> 3;
            const int col = threadIdx.x & 7;
            if (col < p.n) {
                // The butterfly's lanes 8 apart (warps w and w ^ 4), 4 (w ^ 2), 2 (w ^ 1), then
                // 1 (h).
                float s[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float y[MV_WARPS];
#pragma unroll
                    for (int ww = 0; ww < MV_WARPS; ++ww) {
                        y[ww] = part[((ww * 2 + h) * R + row) * 8 + col];
                    }
                    s[h] = ((y[0] + y[4]) + (y[2] + y[6])) + ((y[1] + y[5]) + (y[3] + y[7]));
                }
                float v = s[0] + s[1];
                if (scaled) {
                    v *= sc;
                }
                p.dst[col * p.dst_nb1 + reg + row] = v;
            }
        }
    }
#else
    __trap();
#endif
}

}  // namespace

// The kernels for the host stubs: the quantization, then mv<NWG> for NWG = 4, 2
// (src/model/core/nvfp4_mvq.rs's kernel index).
extern "C" const void * const legion_nvfp4_mvq_kernels[3] = {
    reinterpret_cast<const void *>(&quantize),
    reinterpret_cast<const void *>(&mv<4>),
    reinterpret_cast<const void *>(&mv<2>),
};
