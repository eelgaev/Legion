// Hopper (sm_90a) paged attention prefill on warpgroup MMA (wgmma) fed by the tensor memory
// accelerator (TMA), FlashAttention-3 style (#1215). UNTESTED ON HARDWARE: written and compiled
// without an H100/H200/GH200; the Rust side (src/model/core/paged/hopper.rs) runs it only under
// `legion serve --hopper-attn`, only on a device of compute capability 9.0 whose binary holds the
// sm_90a bodies, and only for what it covers: prefill (more than one token per sequence) over F16
// caches, heads of 128 or 256, no compact cache, no host tier, no key-block split. Everything else
// keeps the default kernels (paged_attn/).
//
// What it computes is paged_prefill_mma's (sm80/paged_attn.cuh): token t of sequence s (position
// n_past[s] + t) attends to keys first_key(pos, window) ..= pos through the sequence's block
// table (a ring of `ring` entries for a sliding-window layer), Q and P rounded to F16 for the
// MMA, scores, softmax and output in F32, the output normalized by the row's sum.
//
// A CTA: 128 query rows of one kv head (row r: token t0 + r / G, query head kvh * G + r % G; the
// last tiles, which see the most keys, first), three warpgroups (384 threads):
// - warpgroup 0, the producer: one thread looks up each key block in the table and issues its K
//   and V tiles (64 cells x 64 dims per copy, 128-byte swizzle) into separate rings of shared
//   memory stages, each with a "full" mbarrier (the TMA's transaction count) and an "empty" one
//   (one arrival per consumer warp); the other threads exit after giving their registers back
//   with setmaxnreg.
// - warpgroups 1 and 2, the consumers, 64 query rows each: Q from F32 to F16 into shared memory
//   once (the same swizzle), then per key block S = Q K^T (wgmma m64n64k16, both operands from
//   shared memory, K-major), the online softmax on the S registers (log2 units), and
//   O += P V (wgmma m64n128k16 per 128 output dims, P from registers, V from shared memory
//   MN-major: the transpose bit). The S of block j is issued together with the P V of block
//   j - 1, so the tensor cores work while the softmax runs (FA3's intra-warpgroup overlap), and
//   the two consumers take turns issuing (named barriers 1 and 2: FA3's ping-pong), so one's
//   softmax hides under the other's GEMMs.
//
// Every K and V block is read whole (64 cells): cells past a row's last key get an exact 0
// weight, and the pools hold finite numbers in every cell (they are cleared at allocation), as
// for the sm_70 WMMA kernel.
//
// Outputs differ from paged_prefill_mma's in the last bits (other MMA shapes and sum orders):
// an opt-in until it is measured on Hopper.
//
// The wgmma / setmaxnreg bodies exist only where __CUDA_ARCH_FEAT_SM90_ALL is defined (an sm_90a
// target); every other target compiles a __trap(). The Rust side checks wgmma_gemm.cu's probe
// (legion_wgmma_gemm_probe, built with the same flags) before it launches.
//
// Patterns studied: FlashAttention-3 (BSD-3-Clause, Tri Dao et al.) and CUTLASS's sm90 GMMA
// descriptors (BSD-3-Clause, NVIDIA); no code copied verbatim. Kernels only: no host code here,
// Rust launches them from legion_hopper_attn_kernels.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

// The kernel's arguments (src/model/core/paged/hopper.rs, `Args`).
struct legion_hopper_attn {
    const float * q;              // F32 [d, n_head, n_tok, n_seqs], byte strides q_nb1..3
    int64_t q_nb1, q_nb2, q_nb3;
    const int32_t * table;        // [table_width, n_seqs]
    const int32_t * n_past;       // [n_seqs]
    float * dst;                  // F32 [d, n_head, n_tok, n_seqs], contiguous
    uint64_t desc_k;              // fixed bits of a K-major descriptor (Q, K): hopper::desc_k_hi
    uint64_t desc_v;              // fixed bits of V's MN-major descriptor: hopper::desc_v_hi
    int32_t table_width, n_head, n_head_kv, n_tok, window, ring;
    float scale;
    int32_t pad;
};

// The K and V pools as 2D tensors (kv_row elements x cells), boxes of 64 x 64, 128-byte swizzle.
struct legion_hopper_attn_maps {
    CUtensorMap k, v;
};

}  // extern "C"

namespace {

#include "tma.cuh"

constexpr int THREADS = 384;  // producer + two consumer warpgroups
constexpr int ROWS    = 128;  // query rows per CTA (64 per consumer)
constexpr int BLK     = 64;   // cells per KV block (KV_BLOCK on the Rust side)
constexpr int SPAN    = 128;  // bytes of a swizzle span: 64 F16 values

template <int D> struct Cfg {
    static constexpr int STAGES = D == 128 ? 4 : 2;     // K stages, and as many V stages
    static constexpr int Q_SUB  = ROWS * SPAN;          // one 64-dim column of Q (16 KiB)
    static constexpr int KV_SUB = BLK * SPAN;           // one 64-dim column of a block (8 KiB)
    static constexpr int Q      = (D / 64) * Q_SUB;     // Q's bytes
    static constexpr int KV     = (D / 64) * KV_SUB;    // one block of K (or V)
};
// Dynamic shared memory: Q, the K and V stages, 1 KiB to align them, four mbarriers per stage.
template <int D> __host__ __device__ constexpr int smem_bytes() {
    return Cfg<D>::Q + 2 * Cfg<D>::STAGES * Cfg<D>::KV + 1024 + 4 * Cfg<D>::STAGES * 8;
}
static_assert(smem_bytes<128>() == 164992, "paged::hopper::smem_bytes(128)");
static_assert(smem_bytes<256>() == 197696, "paged::hopper::smem_bytes(256)");

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define LEGION_HOPPER_ATTN 1
#else
#define LEGION_HOPPER_ATTN 0
#endif

#if LEGION_HOPPER_ATTN
using namespace legion_tma;

__device__ __forceinline__ uint64_t desc(uint64_t hi, uint32_t a) {
    return hi | (uint64_t) ((a & 0x3FFFF) >> 4);
}
__device__ __forceinline__ void wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// Keeps the compiler from moving reads or writes of registers an in-flight wgmma reads or
// writes across the fences and waits around it, and from reusing them before the wait
// (CUTLASS's warpgroup_fence_operand).
template <int N> __device__ __forceinline__ void fence_regs(float (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        asm volatile("" : "+f"(d[i])::"memory");
    }
}
__device__ __forceinline__ void fence_regs(uint32_t (&a)[4][4]) {
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            asm volatile("" : "+r"(a[i][j])::"memory");
        }
}
__device__ __forceinline__ void named_sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int threads) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

// d (64 x 64 F32) = A B^T + (acc ? d : 0) for one 16-deep slice: A 64 query rows, B 64 keys,
// both K-major in shared memory behind descriptors da / db.
__device__ __forceinline__ void wgmma_s(float (&d)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
        "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]),
          "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(acc));
}

// d (64 x 128 F32) += A B for one 16-key slice: A (64 rows x 16 keys of P) in registers as the
// m64k16 A fragment, B (16 keys x 128 dims of V) MN-major in shared memory behind db.
__device__ __forceinline__ void wgmma_pv(float (&d)[64], const uint32_t (&a)[4], uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
        "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, "
        "%39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
        "%58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]),
          "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]),
          "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
          "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]),
          "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}

__device__ __forceinline__ int first_key(int pos, int window) {
    return window > 0 ? max(0, pos - window + 1) : 0;
}
#endif  // LEGION_HOPPER_ATTN

template <int D>
__global__ void __launch_bounds__(THREADS, 1)
    paged_prefill_wgmma(const legion_hopper_attn p, const __grid_constant__ legion_hopper_attn_maps maps) {
#if LEGION_HOPPER_ATTN
    using C = Cfg<D>;
    constexpr int STAGES = C::STAGES, NH = D / 128;  // NH: 128-dim halves of the output
    constexpr float LOG2E = 1.4426950408889634f;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char * smem = (unsigned char *) (((uintptr_t) smem_raw + 1023) & ~(uintptr_t) 1023);
    unsigned char * sq = smem;
    unsigned char * sk = smem + C::Q;
    unsigned char * sv = sk + STAGES * C::KV;
    uint64_t * full_k  = (uint64_t *) (sv + STAGES * C::KV);
    uint64_t * empty_k = full_k + STAGES;
    uint64_t * full_v  = empty_k + STAGES;
    uint64_t * empty_v = full_v + STAGES;
    const int wg = threadIdx.x / 128, warp = threadIdx.x / 32 % 4, lane = threadIdx.x % 32;

    const int G = p.n_head / p.n_head_kv, QT = ROWS / G;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * QT, kvh = blockIdx.y, seq = blockIdx.z;
    const int n_past = p.n_past[seq];
    // Row r's token. Rows past n_tok, and past QT * G when G does not divide 128, repeat the
    // tile's last token and are not written.
    auto tok_of = [&](int r) { return min(t0 + min(r / G, QT - 1), p.n_tok - 1); };
    auto pos_of = [&](int r) { return n_past + tok_of(r); };
    const int k_end   = pos_of(ROWS - 1) + 1;
    const int b_first = first_key(pos_of(0), p.window) / BLK;
    const int nblk    = (k_end - 1) / BLK - b_first + 1;

    if (threadIdx.x == 0) {
        prefetch_map(&maps.k);
        prefetch_map(&maps.v);
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            bar_init(full_k + s, 1);
            bar_init(empty_k + s, 8);  // the eight consumer warps
            bar_init(full_v + s, 1);
            bar_init(empty_v + s, 8);
        }
        bar_init_fence();
    }
    __syncthreads();  // the last CTA-wide barrier: the producer's idle threads exit below

    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (threadIdx.x == 0) {
            const int32_t * tab = p.table + (int64_t) seq * p.table_width;
            for (int j = 0; j < nblk; j++) {
                const int b = b_first + j;
                const int cell0 = tab[p.ring > 0 ? b % p.ring : b] * BLK;
                const int s = j % STAGES, round = j / STAGES;
                if (round > 0) {
                    bar_wait(empty_k + s, (uint32_t) (round - 1) & 1);
                }
                bar_expect_tx(full_k + s, (uint32_t) C::KV);
#pragma unroll
                for (int c = 0; c < D / 64; c++) {
                    tma_2d(sk + s * C::KV + c * C::KV_SUB, &maps.k, kvh * D + c * 64, cell0, full_k + s);
                }
                if (round > 0) {
                    bar_wait(empty_v + s, (uint32_t) (round - 1) & 1);
                }
                bar_expect_tx(full_v + s, (uint32_t) C::KV);
#pragma unroll
                for (int c = 0; c < D / 64; c++) {
                    tma_2d(sv + s * C::KV + c * C::KV_SUB, &maps.v, kvh * D + c * 64, cell0, full_v + s);
                }
            }
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    const int cw = wg - 1;                   // consumer 0 or 1: query rows 64 * cw ..
    const int tid = threadIdx.x % 128;       // thread in the consumer warpgroup
    const int gq = lane / 4, tq = lane % 4;  // fragment row group, thread in quad

    // Q: this consumer's 64 rows, F32 to F16, into 64-dim columns of the 128-byte swizzle (16-byte
    // unit u of row r at unit u ^ (r % 8)), where wgmma reads them through K-major descriptors.
    {
        constexpr int U = D / 8;  // 16-byte units (8 values) per row
        for (int i = tid; i < 64 * U; i += 128) {
            const int r = 64 * cw + i / U, u = i % U;
            const int t = tok_of(r);
            const float * qr = (const float *) ((const char *) p.q + seq * p.q_nb3 + t * p.q_nb2 +
                                                (kvh * G + r % G) * p.q_nb1) + u * 8;
            const float2 a = *(const float2 *) (qr + 0), b = *(const float2 *) (qr + 2);
            const float2 c = *(const float2 *) (qr + 4), d = *(const float2 *) (qr + 6);
            const uint4 x  = make_uint4(pack(a.x, a.y), pack(b.x, b.y), pack(c.x, c.y), pack(d.x, d.y));
            *(uint4 *) (sq + (u / 8) * C::Q_SUB + r * SPAN + (((u % 8) ^ (r % 8)) << 4)) = x;
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // visible to wgmma
        named_sync(3 + cw, 128);
    }

    // This thread's two rows of the S / O fragments: 16 warp + gq (h = 0) and + 8 (h = 1).
    int pos[2], first[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        pos[h]   = pos_of(64 * cw + 16 * warp + gq + 8 * h);
        first[h] = first_key(pos[h], p.window);
    }

    float s[32];               // S: element i is row h = i / 2 % 2, key 8 (i / 4) + 2 tq + i % 2
    float o[NH][64];           // O, per 128 output dims: element i is row i / 2 % 2, dim 8 (i / 4) + 2 tq + i % 2
    uint32_t pf[4][4];         // P as the A fragment of each 16-key slice
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0.0f;
#pragma unroll
    for (int h = 0; h < NH; h++)
#pragma unroll
        for (int i = 0; i < 64; i++) o[h][i] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) pf[i][j] = 0;
    const float scale = p.scale * LOG2E;
    const uint64_t hk = p.desc_k, hv = p.desc_v;
    const uint32_t q_addr = saddr(sq) + cw * (64 * SPAN);

    // S = Q K^T of the block in K stage `st`: one pipeline stage (fence, wgmmas, commit).
    auto issue_s = [&](int st) {
        const uint32_t k_addr = saddr(sk + st * C::KV);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
            wgmma_s(s, desc(hk, q_addr + (kk / 4) * C::Q_SUB + (kk % 4) * 32),
                    desc(hk, k_addr + (kk / 4) * C::KV_SUB + (kk % 4) * 32), kk > 0);
        }
        wg_commit();
    };
    // O += P V of the block in V stage `sp`: a pipeline stage of its own, so the softmax may read
    // S once S's group is done while this one runs.
    auto issue_pv = [&](int sp) {
        const uint32_t v_addr = saddr(sv + sp * C::KV);
        wg_fence();
#pragma unroll
        for (int h = 0; h < NH; h++)
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                wgmma_pv(o[h], pf[kk], desc(hv, v_addr + h * 2 * C::KV_SUB + kk * 2 * 1024));
            }
        wg_commit();
    };
    // The online softmax of the block at key k0 on S (log2 units): S becomes the weights, m and
    // l move on, alpha the rescale of the output so far.
    auto softmax = [&](int k0, float (&alpha)[2]) {
        const bool whole = k0 >= first[1] && k0 + BLK - 1 <= pos[0];
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int i = 0; i < 32; i++) {
            const int h = i / 2 % 2;
            float x = s[i] * scale;
            if (!whole) {
                const int key = k0 + 8 * (i / 4) + 2 * tq + i % 2;
                x = key <= pos[h] && key >= first[h] ? x : -INFINITY;
            }
            s[i]  = x;
            mx[h] = fmaxf(mx[h], x);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 1));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffff, mx[h], 2));
            const float m_new = fmaxf(m[h], mx[h]);
            // No visible key yet: exp against 0 gives 0 for the masked scores.
            const float m_use = m_new == -INFINITY ? 0.0f : m_new;
            alpha[h] = exp2f(m[h] - m_use);
            m[h]     = m_new;
            mx[h]    = m_use;
        }
        float sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 32; i++) {
            s[i] = exp2f(s[i] - mx[i / 2 % 2]);
            sum[i / 2 % 2] += s[i];
        }
        l[0] = l[0] * alpha[0] + sum[0];
        l[1] = l[1] * alpha[1] + sum[1];
    };
    // P's A fragment of slice kk: rows gq / gq + 8, keys 16 kk + 2 tq (+ 8): S's chunks 2 kk and
    // 2 kk + 1.
    auto to_p = [&]() {
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            pf[kk][0] = pack(s[8 * kk + 0], s[8 * kk + 1]);
            pf[kk][1] = pack(s[8 * kk + 2], s[8 * kk + 3]);
            pf[kk][2] = pack(s[8 * kk + 4], s[8 * kk + 5]);
            pf[kk][3] = pack(s[8 * kk + 6], s[8 * kk + 7]);
        }
    };

    // Ping-pong: consumer c issues its GEMMs between bar.sync on barrier 1 + c and bar.arrive on
    // the other's; consumer 0 goes first, and consumer 1 skips its last arrival, so both
    // barriers see as many arrivals as waits.
    if (cw == 0) {
        named_arrive(1, 256);
    }

    // Block 0: S alone (no P V before it; a block of its own so that every wgmma group count
    // in the loop below is known to the compiler).
    {
        float alpha[2];
        bar_wait(full_k, 0);
        named_sync(1 + cw, 256);
        fence_regs(s);
        issue_s(0);
        if (cw == 0 || nblk > 1) {
            named_arrive(2 - cw, 256);
        }
        wg_wait<0>();
        fence_regs(s);
        if (lane == 0) {
            bar_arrive(empty_k);
        }
        softmax(b_first * BLK, alpha);
        to_p();
    }

    for (int j = 1; j < nblk; j++) {
        const int st = j % STAGES, sp = (j - 1) % STAGES;
        float alpha[2];
        bar_wait(full_k + st, (uint32_t) (j / STAGES) & 1);
        bar_wait(full_v + sp, (uint32_t) ((j - 1) / STAGES) & 1);
        named_sync(1 + cw, 256);
        fence_regs(s);
#pragma unroll
        for (int h = 0; h < NH; h++) fence_regs(o[h]);
        fence_regs(pf);
        issue_s(st);
        issue_pv(sp);
        if (cw == 0 || j < nblk - 1) {
            named_arrive(2 - cw, 256);
        }
        // S done (the older group); the previous block's P V still running.
        wg_wait<1>();
        fence_regs(s);
        if (lane == 0) {
            bar_arrive(empty_k + st);
        }
        softmax((b_first + j) * BLK, alpha);
        // The previous block's P V done: its V stage free, P and O ours again.
        wg_wait<0>();
#pragma unroll
        for (int h = 0; h < NH; h++) fence_regs(o[h]);
        fence_regs(pf);
        if (lane == 0) {
            bar_arrive(empty_v + sp);
        }
#pragma unroll
        for (int h = 0; h < NH; h++)
#pragma unroll
            for (int i = 0; i < 64; i++) o[h][i] *= alpha[i / 2 % 2];
        to_p();
    }

    // The last block's P V.
    {
        const int sp = (nblk - 1) % STAGES;
        bar_wait(full_v + sp, (uint32_t) ((nblk - 1) / STAGES) & 1);
#pragma unroll
        for (int h = 0; h < NH; h++) fence_regs(o[h]);
        fence_regs(pf);
        issue_pv(sp);
        wg_wait<0>();
#pragma unroll
        for (int h = 0; h < NH; h++) fence_regs(o[h]);
    }

    // The rows' sums over the quad, then O / sum.
#pragma unroll
    for (int h = 0; h < 2; h++) {
        l[h] += __shfl_xor_sync(0xffffffff, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffff, l[h], 2);
        const int r = 64 * cw + 16 * warp + gq + 8 * h;
        const int t = t0 + r / G;
        if (r >= QT * G || t >= p.n_tok) {
            continue;
        }
        const float inv = 1.0f / l[h];
        float * out = p.dst + (((int64_t) seq * p.n_tok + t) * p.n_head + kvh * G + r % G) * D + 2 * tq;
#pragma unroll
        for (int hf = 0; hf < NH; hf++)
#pragma unroll
            for (int c = 0; c < 16; c++) {
                *(float2 *) (out + hf * 128 + 8 * c) =
                    make_float2(o[hf][4 * c + 2 * h] * inv, o[hf][4 * c + 2 * h + 1] * inv);
            }
    }
#else
    (void) p;
    (void) maps;
    __trap();  // the Rust side checks legion_wgmma_gemm_probe first
#endif
}

}  // namespace

// The kernels as the CUDA runtime knows them (their host stubs), for Rust's cudaLaunchKernel
// (src/model/core/paged/hopper.rs): heads of 128, 256.
extern "C" const void * const legion_hopper_attn_kernels[2] = {
    reinterpret_cast<const void *>(&paged_prefill_wgmma<128>),
    reinterpret_cast<const void *>(&paged_prefill_wgmma<256>),
};
