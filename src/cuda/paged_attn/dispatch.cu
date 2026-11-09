// paged_attn's extern "C" entry points, the prefill kernel choice and the tiered-pool pieces (see
// common.cuh); the kernels of each head size are compiled in d128.cu, d256.cu and d512.cu.
#include "common.cuh"

namespace legion_paged {

int g_prefill = LEGION_PREFILL_AUTO;
thread_local int t_prefill = -1;
thread_local int64_t t_m884_launches = 0;
thread_local int64_t t_vsplit_launches = 0;

}  // namespace legion_paged

namespace {

// ---------------------------------------------------------------------------------------------
// Pieces of a prefill over a tiered pool (#352). The prefill kernels stream every key block once
// per query tile (64 query columns), which costs nothing extra from VRAM but multiplies a host
// block's link traffic by the tiles of the step (64 for a 512-token step at 8 query heads per kv
// head). A piece instead copies its host blocks into VRAM staging slots once, and every tile then
// reads them there; pieces after the first merge their rows into the first's output through the
// rows' running (max, sum) (merge_row).
// ---------------------------------------------------------------------------------------------

constexpr int BUILD_THREADS = 1024;

// Exclusive prefix sum of x over the CTA (BUILD_THREADS threads) and the total; sw: 32 ints.
__device__ __forceinline__ int2 cta_scan(int x, int * sw) {
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
        int t = sw[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, t, o);
            t += lane >= o ? y : 0;
        }
        sw[lane] = t;
    }
    __syncthreads();
    const int2 r = make_int2((warp > 0 ? sw[warp - 1] : 0) + v - x, sw[31]);
    __syncthreads();
    return r;
}

// One CTA: the piece's walk entries per sequence (piece_blk / piece_ent / piece_n), the pool
// block of each staging slot (src) and the slots in use (piece_cnt).
__global__ void __launch_bounds__(BUILD_THREADS) paged_piece_build(const legion_paged_attn p, int hbm, int cap, int piece, int32_t * src) {
    __shared__ int sw[32];
    int32_t * blk = (int32_t *) p.piece_blk;
    int32_t * ent = (int32_t *) p.piece_ent;
    int host = 0;  // host blocks so far, over the sequences in order
    for (int seq = 0; seq < p.n_seqs; seq++) {
        const int64_t row = (int64_t) seq * p.table_width;
        const int n_past = p.n_past[seq];
        const int b0 = first_key(n_past, p.window) / BLK;
        const int b1 = min((n_past + p.n_tok - 1) / BLK, p.table_width - 1);
        int n = 0;
        for (int c = b0; c <= b1; c += BUILD_THREADS) {
            const int b = c + threadIdx.x;
            const bool valid = b <= b1;
            const int e = valid ? p.table[row + b] : 0;
            const bool is_host = valid && e >= hbm;
            const int2 h = cta_scan(is_host, sw);
            const int ord = host + h.x;
            const bool in = valid && (is_host ? ord / cap == piece : piece == 0);
            const int2 k = cta_scan(in, sw);
            if (in) {
                blk[row + n + k.x] = b;
                if (is_host) {
                    const int slot = ord - piece * cap;
                    ent[row + n + k.x] = -1 - slot;
                    src[slot] = e;
                } else {
                    ent[row + n + k.x] = e;
                }
            }
            host += h.y;
            n += k.y;
        }
        if (threadIdx.x == 0) {
            ((int32_t *) p.piece_n)[seq] = n;
        }
    }
    if (threadIdx.x == 0) {
        *(int32_t *) p.piece_cnt = max(0, min(host - piece * cap, cap));
    }
}

constexpr int COPY_PARTS = 8;

// Grid (cap, COPY_PARTS): slot blockIdx.x's block, K and V, from the pool into its staging slot;
// K's only in Gemma 4's compact cache (C, #551), whose V lies in K's cells.
template <bool C = false>
__global__ void __launch_bounds__(256) paged_piece_copy(const legion_paged_attn p, const int32_t * src) {
    const int slot = blockIdx.x;
    if (slot >= *p.piece_cnt) {
        return;
    }
    const int64_t e = src[slot];
    for (int kv = 0; kv < (C ? 1 : 2); kv++) {
        const int64_t bytes = BLK * (kv == 0 ? p.k_cell : p.v_cell);  // a multiple of 16
        const uint4 * from = (const uint4 *) ((const char *) (kv == 0 ? p.k : p.v) + e * bytes);
        uint4 * to = (uint4 *) ((kv == 0 ? p.k_stage : p.v_stage) + slot * bytes);
        const int64_t n = bytes / 16, per = (n + COPY_PARTS - 1) / COPY_PARTS;
        const int64_t i1 = min(n, (blockIdx.y + 1) * per);
        for (int64_t i = blockIdx.y * per + threadIdx.x; i < i1; i += blockDim.x) {
            to[i] = from[i];
        }
    }
}

}  // namespace

int legion_paged_attn_quant_ok(void) {
    return fa::available() ? 1 : 0;
}

int legion_paged_attn_split_ok(void) {
    return !fa::available() && !use_m884() ? 1 : 0;
}

int64_t legion_paged_attn_m884_launches(void) {
    return t_m884_launches;
}

int64_t legion_paged_attn_vsplit_launches(void) {
    return t_vsplit_launches;
}

void legion_paged_attn_set_prefill(int kernel, int this_thread) {
    if (this_thread > 0) {
        t_prefill = kernel;
    } else {
        __atomic_store_n(&g_prefill, kernel, __ATOMIC_RELAXED);
    }
}

int legion_paged_attn_piece(const legion_paged_attn * p, int32_t hbm_blocks, int32_t cap, int32_t piece,
                            int32_t * src, void * stream) {
    cudaStream_t st = (cudaStream_t) stream;
    paged_piece_build<<<1, BUILD_THREADS, 0, st>>>(*p, hbm_blocks, cap, piece, src);
    if (p->compact) {
        paged_piece_copy<true><<<dim3(cap, COPY_PARTS), 256, 0, st>>>(*p, src);
    } else {
        paged_piece_copy<<<dim3(cap, COPY_PARTS), 256, 0, st>>>(*p, src);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int legion_paged_attn_launch(const legion_paged_attn * p, void * stream) {
    if (p->n_head_kv <= 0 || p->n_head % p->n_head_kv != 0) {
        return 1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (p->compact) {
        return legion_paged::launch_compact(*p, st);
    }
    switch (p->d) {
        case 128: return legion_paged::launch_dim<128>(*p, st);
        case 256: return legion_paged::launch_dim<256>(*p, st);
        case 512: return legion_paged::launch_dim<512>(*p, st);
        default: return 1;
    }
}
