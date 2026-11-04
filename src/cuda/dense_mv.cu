// BF16 skinny GEMM for decode steps (#504): F32 activations of 2-16 tokens, up to four weights
// that read the same input in one launch. The arithmetic and its order are ggml-cuda's
// `mul_mat_f` (mmf.cuh, the mma path), so the results are its bits: warp p of nw takes the
// 32-pair chunks p, p + nw, ... of a row tile, m16n8k16 BF16 products with F32 sums per chunk,
// the warps' sums added in warp order starting from 0; nw as ggml picks it from the row length.
// What differs is how the weights are read. ggml-cuda builds with programmatic dependent launch
// on sm_90+, where it drops `__restrict__` from its kernels (ggml's GGML_CUDA_RESTRICT), so their
// weight loads do not take the read-only path; here they do, and each warp keeps D chunks of its
// weight rows in flight in registers (sm80/dense_mv.cuh). Serving 16 sequences on a GB10, ggml
// streams the large weights at 160-220 GB/s, this kernel at 210-230.
// F32 weights (#836, the MoE router of 4-16 tokens) take ggml's mul_mat_f<float> the same way:
// TF32 products of the raw F32 bits, ggml's warp count for a row of k floats. ggml runs that
// matmul with one 32-row tile per CTA (16 CTAs for a 512-expert router), latency bound; here
// 16-row tiles and the loads in flight. Where the router's F32 values are all exact in BF16 (a
// BF16 checkpoint's router widened by the loader), a BF16 copy is read instead and each value
// widened back to its F32 bits as it is staged (f32 = 2): the same staged words, so the same
// bits, from half the bytes.
// Chain weights (#836 part 29: the GDN alpha and beta of an NVFP4 checkpoint, 48 BF16 rows each)
// ride along at 9-16 tokens. ggml-cuda runs a BF16 matmul whose rows are not a multiple of 32
// on cuBLAS there (mmf takes only full 32-row tiles, mmvf only up to 8 tokens): a one-warp
// wmma 16x16x16 GEMM per 16 x 16 output tile on a GB10, x rounded to BF16, every output one
// chain of m16n8k16 products over k in order from 0, written as it is. Those 18 us launches
// (two per layer) are latency bound; here each 16-row tile is one extra CTA of the launch that
// reads the same x (only its first warp works), in the same order, so the same bits.
// Few row tiles (#836: a hyper-connection mixer's down matrix, 320 rows of 10,240, is 10 CTAs
// at 9 tokens on a 48-SM GB10, ~186 GB/s): each tile's nw warps are spread over a thread-block
// cluster of S CTAs. A warp's chunks and its sums are the same, wherever it runs; the epilogue
// adds the warps' sums in warp order from 0 as before, reading the other CTAs' sums from
// distributed shared memory, so the same bits.
// Wide chains (#836): steps of 17-48 tokens (MTP verify steps of 2-5 sequences), where
// ggml-cuda runs every BF16 matmul on cuBLAS. Every weight is computed as a chain weight, in the
// order of cuBLAS's wmma kernels without split-K, the x chunk staged once per CTA for its warps
// (chain_wide). cuBLAS picks its kernel by shape, and some sum k in slices (a GB10 at 33-48
// tokens: the k = 6,144 projections on an nvjet kernel, 8 slices of 768 added in order); with
// kslice set each slice is its own chain and the slices' sums are added in that order in the
// warp's registers (chain_wide<NT, true>), the same reads. By default src/model/dense_mv.rs routes
// only the shapes and step sizes measured to give cuBLAS's bits here, in their order.
//
// Shapes are checked here and in src/model/dense_mv.rs, which decides where the op runs.

#include "device_cache.cuh"

#include <cuda_bf16.h>
#include <cstdint>

extern "C" {
struct legion_dense_mv {
    const __nv_bfloat16 * w[4];  // BF16 weights: rows of k values, w_ld values apart
    float * dst[4];              // F32 results [rows, n], columns rows apart
    int64_t w_ld[4];
    int64_t rows[4];
    const float * x;  // F32 input: n columns of k values, x_ld values apart
    int64_t x_ld;
    int64_t k;
    int32_t n;   // tokens, 2-16; 17-48 with every weight a chain weight (chain_wide)
    int32_t nw;  // weights, 1-4
    // 1: the weights are F32 (w holds float rows, w_ld in floats; ggml's mul_mat_f<float>
    // with TF32 products, from 4 tokens, k a multiple of 32). 2: F32 weights whose values are
    // stored as BF16 (w holds BF16 rows, w_ld in values, even), computed as 1 computes their F32
    // bits. 0: BF16 weights (ggml's mul_mat_f<nv_bfloat162>).
    int32_t f32;
    // The last `nchain` weights are chain weights (BF16, rows a multiple of 16, 9-16 tokens):
    // cuBLAS's order, one CTA per 16 rows, launched ahead of the others' tiles.
    int32_t nchain;
    // Nonzero: a launch of fewer row tiles than SMs may spread each tile's warps over a cluster
    // of CTAs (BF16, no chain weights, compute capability 9.0+); the same bits.
    int32_t split;
    // Set by the launcher (callers leave 0): the warps a tile has in all (ggml's count).
    int32_t warps;
    // Wide chains only: 0, one chain over k; else the k values of each slice of cuBLAS's
    // split-K kernel (a multiple of 16), each slice its own chain from 0 and the slices' sums
    // added in order (chain_wide<NT, true>).
    int32_t kslice;
    // A probe of cuBLAS's order (dense-mv-bench; 0 or 1 otherwise): within each slice the k
    // steps of 16 go to kinter (1-4) accumulators in turn, kgroup k values each (a multiple of
    // 16), added in order at the slice's end.
    int32_t kinter;
    int32_t kgroup;
    // Probe only (kinter > 1): 8, each k16 step as two m16n8k8 products.
    int32_t kmma;
    // Probe only: how the accumulators are added (0 in order, 1 pairwise, 2 last to first).
    int32_t kred;
};

// Enqueues dst[i] = w[i] . x for every weight on `stream`. 0 on success; 1 for a shape the
// kernels do not take; a CUDA error code for a failed launch.
int legion_dense_mv_launch(const legion_dense_mv * p, void * stream);

// Whether the current device runs the op where it pays: compute capability 9.0 or newer (where
// ggml-cuda's kernels lose `__restrict__`) and a binary with sm_80+ code for the mma kernel.
int legion_dense_mv_supported(void);

// Launches every op from now on with `nta` tiles of 16 rows per CTA and `d` chunks in flight per
// warp (nta 1 or 2, d 2, 4 or 8, nta * d at most 8), or the default choice for nta = 0: for
// tests and tuning, the same bits either way. 0 on success, 1 for a variant the kernels lack.
int legion_dense_mv_force(int nta, int d);
}

namespace {

typedef __nv_bfloat162 bf162;

constexpr int MAX_W = 4;
constexpr int MAX_N = 16;
// Most tokens of a wide chain launch (chain_wide, all weights chain weights).
constexpr int MAX_WIDE_N = 48;
// The kernel's shared-memory row stride in 4-byte words (ggml's 32 + padding 4).
constexpr int PAD = 36;
// ggml's largest CTA for mul_mat_f (NVIDIA): 256 threads.
constexpr int MAX_THREADS = 256;

// The weight among the first `nw` that unit `u` (a row or a row tile of `per` rows) of the launch
// falls in, from weight `i`; `u` becomes its index within that weight.
__device__ __forceinline__ int pick(const legion_dense_mv & p, int64_t & u, int per, int nw, int i = 0) {
    while (i + 1 < nw && u >= p.rows[i] / per) {
        u -= p.rows[i] / per;
        ++i;
    }
    return i;
}

#include "sm80/dense_mv.cuh"

// ggml's choice of warps for mmf (launch_mul_mat_f_cuda), from the row's k / 2 pairs.
int mmf_warps(int64_t k) {
    const int64_t k2 = k / 2;
    int best = 1;
    int64_t niter_best = (k2 + 63) / 64;
    for (int nw = 2; nw <= MAX_THREADS / 32; nw++) {
        const int64_t niter = (k2 + nw * 64 - 1) / (nw * 64);
        if (niter < niter_best) {
            niter_best = niter;
            best = nw;
        }
    }
    return best;
}

// Whether this binary has sm_80+ code for the mma kernels.
bool mma_code() {
    static std::atomic<int8_t> state[GGML_CUDA_MAX_DEVICES];  // 0 not yet, 1 yes, 2 no
    const int dev = legion_device::current();
    if (dev < 0) {
        return false;
    }
    int8_t st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        cudaFuncAttributes fa{};
        const cudaError_t before = cudaPeekAtLastError();
        const bool code = cudaFuncGetAttributes(&fa, mmf<1, 1, 4>) == cudaSuccess && fa.binaryVersion >= 80 &&
                          fa.ptxVersion >= 80;
        if (!code) {
            legion_device::clear_own_error(before);
        }
        st = code ? 1 : 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

// The variant forced by legion_dense_mv_force: 0 none, else nta | d << 8.
std::atomic<int> forced{0};

// Whether mmf<NTB, NTA, D, false, S> launches on the current device: sm_90+ code and its
// cluster width S set (once per device).
template <int NTB, int NTA, int D, int S>
bool cluster_ready() {
    static std::atomic<int8_t> state[GGML_CUDA_MAX_DEVICES];  // 0 not yet, 1 yes, 2 no
    const int dev = legion_device::current();
    if (dev < 0) {
        return false;
    }
    int8_t st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        const auto k = mmf<NTB, NTA, D, false, S>;
        cudaFuncAttributes fa{};
        const cudaError_t before = cudaPeekAtLastError();
        const bool ok = legion_device::compute_capability() >= 90 && cudaFuncGetAttributes(&fa, k) == cudaSuccess &&
                        fa.binaryVersion >= 90 &&
                        cudaFuncSetAttribute(k, cudaFuncAttributeRequiredClusterWidth, S) == cudaSuccess &&
                        cudaFuncSetAttribute(k, cudaFuncAttributeRequiredClusterHeight, 1) == cudaSuccess &&
                        cudaFuncSetAttribute(k, cudaFuncAttributeRequiredClusterDepth, 1) == cudaSuccess;
        if (!ok) {
            legion_device::clear_own_error(before);
        }
        st = ok ? 1 : 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

// CTAs per row tile for `tiles` tiles of nw warps (1: no cluster): from fewer tiles than SMs,
// the fewest of 2, 4, 8 that fill the SMs (at most nw).
int split_of(int64_t tiles, int nw) {
    const int sms = legion_device::sm_count();
    int s = 1;
    while (s < 8 && 2 * s <= nw && tiles * s < sms) {
        s *= 2;
    }
    return s;
}

template <int NTB, int NTA, int D, bool F, int S = 1, bool WB = false>
int launch_mmf(const legion_dense_mv & p, cudaStream_t s) {
    static std::atomic<int8_t> opt[GGML_CUDA_MAX_DEVICES];
    constexpr int R = NTA * 16;
    // ggml counts a row's columns in its element type (pairs for BF16, floats for F32).
    const int nw = mmf_warps(F ? 2 * p.k : p.k);
    if (!legion_device::smem_opt_in(opt, mmf<NTB, NTA, D, F, S, WB>, mmf_smem<NTB, NTA>(MAX_THREADS / 32))) {
        return 1;
    }
    // Only the kernel of two token tiles (9-16 tokens) has the chain tiles.
    if (p.nchain != 0 && (NTB != 2 || F || S != 1)) {
        return 1;
    }
    int64_t tiles = chain_tiles(p);
    for (int i = 0; i < p.nw - p.nchain; i++) {
        tiles += p.rows[i] / R;
    }
    if constexpr (S == 1) {
        mmf<NTB, NTA, D, F, 1, WB><<<(unsigned) tiles, dim3(32, nw), mmf_smem<NTB, NTA>(nw), s>>>(p);
    } else {
        static_assert(!F && !WB, "clusters for BF16 weights only");
        legion_dense_mv q = p;
        q.warps = nw;
        const int nwl = (nw + S - 1) / S;
        mmf<NTB, NTA, D, F, S><<<(unsigned) (tiles * S), dim3(32, nwl), mmf_smem<NTB, NTA>(nwl), s>>>(q);
    }
    return 0;
}

// The default kernel of `NTB` token tiles, BF16: on a cluster of S CTAs per tile where
// split_of says so and the device takes it, else one CTA per tile.
template <int NTB, int NTA, int D>
int launch_bf16(const legion_dense_mv & p, cudaStream_t s) {
    if (p.split != 0 && p.nchain == 0) {
        int64_t tiles = 0;
        for (int i = 0; i < p.nw; i++) {
            tiles += p.rows[i] / (NTA * 16);
        }
        switch (split_of(tiles, mmf_warps(p.k))) {
            case 2:
                if (cluster_ready<NTB, NTA, D, 2>()) {
                    return launch_mmf<NTB, NTA, D, false, 2>(p, s);
                }
                break;
            case 4:
                if (cluster_ready<NTB, NTA, D, 4>()) {
                    return launch_mmf<NTB, NTA, D, false, 4>(p, s);
                }
                break;
            case 8:
                if (cluster_ready<NTB, NTA, D, 8>()) {
                    return launch_mmf<NTB, NTA, D, false, 8>(p, s);
                }
                break;
            default:
                break;
        }
    }
    return launch_mmf<NTB, NTA, D, false>(p, s);
}

// The kernel of `NTB` token tiles with `nta` row tiles and `d` chunks in flight.
template <int NTB, bool F, bool WB = false>
int launch_variant(const legion_dense_mv & p, cudaStream_t s, int nta, int d) {
    switch (nta * 16 + d) {
        case 16 + 2:
            return launch_mmf<NTB, 1, 2, F, 1, WB>(p, s);
        case 16 + 4:
            return launch_mmf<NTB, 1, 4, F, 1, WB>(p, s);
        case 16 + 8:
            return launch_mmf<NTB, 1, 8, F, 1, WB>(p, s);
        case 32 + 2:
            return launch_mmf<NTB, 2, 2, F, 1, WB>(p, s);
        case 32 + 4:
            return launch_mmf<NTB, 2, 4, F, 1, WB>(p, s);
        default:
            return 1;
    }
}

// The default or forced variant for the weights' type. WB (F32 values stored as BF16) holds
// half the words per chunk, so twice the chunks in flight keep the F32 kernel's bytes in flight
// with its registers.
template <bool F, bool WB = false>
int launch_typed(const legion_dense_mv & p, cudaStream_t s, int f) {
    if (f != 0) {
        return p.n <= 8 ? launch_variant<1, F, WB>(p, s, f & 0xff, f >> 8) :
                          launch_variant<2, F, WB>(p, s, f & 0xff, f >> 8);
    }
    if constexpr (WB) {
        return p.n <= 8 ? launch_mmf<1, 1, 8, F, 1, WB>(p, s) : launch_mmf<2, 2, 4, F, 1, WB>(p, s);
    }
    if constexpr (!F) {
        return p.n <= 8 ? launch_bf16<1, 1, 4>(p, s) : launch_bf16<2, 2, 2>(p, s);
    }
    return p.n <= 8 ? launch_mmf<1, 1, 4, F>(p, s) : launch_mmf<2, 2, 2, F>(p, s);
}

// Wide chains (chain_wide): every weight a chain weight (nchain = nw), BF16, 17 to MAX_WIDE_N
// tokens, rows a multiple of 16.
int launch_wide(const legion_dense_mv & p, cudaStream_t s) {
    if (p.f32 != 0 || p.nw < 1 || p.nw > MAX_W || p.nchain != p.nw || p.n <= MAX_N || p.n > MAX_WIDE_N ||
        p.k <= 0 || p.k % 64 != 0 || p.x_ld % 2 != 0) {
        return 1;
    }
    for (int i = 0; i < p.nw; i++) {
        if (p.w_ld[i] % 2 != 0 || p.rows[i] <= 0 || p.rows[i] % 16 != 0) {
            return 1;
        }
    }
    const int64_t tiles = chain_tiles(p);
    if (tiles * 16 > INT32_MAX || !mma_code()) {
        return 1;
    }
    const bool probe = p.kinter > 1 || p.kmma == 8;
    if (p.kslice < 0 || p.kslice % 16 != 0 || p.kinter < 0 || p.kinter > 8 ||
        (probe && (p.kslice == 0 || p.kgroup <= 0 || p.kgroup % 16 != 0))) {
        return 1;
    }
    const dim3 grid((unsigned) ((tiles + WIDE_WARPS - 1) / WIDE_WARPS)), block(32, WIDE_WARPS);
    const int v = probe ? 2 : p.kslice > 0 ? 1 : 0;
    switch ((p.n + 7) / 8 * 4 + v) {
        case 12:
            chain_wide<3, false><<<grid, block, 0, s>>>(p);
            break;
        case 13:
            chain_wide<3, true><<<grid, block, 0, s>>>(p);
            break;
        case 14:
            chain_wide<3, true, 8><<<grid, block, 0, s>>>(p);
            break;
        case 16:
            chain_wide<4, false><<<grid, block, 0, s>>>(p);
            break;
        case 17:
            chain_wide<4, true><<<grid, block, 0, s>>>(p);
            break;
        case 18:
            chain_wide<4, true, 8><<<grid, block, 0, s>>>(p);
            break;
        case 20:
            chain_wide<5, false><<<grid, block, 0, s>>>(p);
            break;
        case 21:
            chain_wide<5, true><<<grid, block, 0, s>>>(p);
            break;
        case 22:
            chain_wide<5, true, 8><<<grid, block, 0, s>>>(p);
            break;
        case 24:
            chain_wide<6, false><<<grid, block, 0, s>>>(p);
            break;
        case 25:
            chain_wide<6, true><<<grid, block, 0, s>>>(p);
            break;
        case 26:
            chain_wide<6, true, 8><<<grid, block, 0, s>>>(p);
            break;
        default:
            return 1;
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}

}  // namespace

int legion_dense_mv_force(int nta, int d) {
    if (nta == 0) {
        forced.store(0, std::memory_order_relaxed);
        return 0;
    }
    if ((nta != 1 && nta != 2) || (d != 2 && d != 4 && d != 8) || nta * d > 8) {
        return 1;
    }
    forced.store(nta | d << 8, std::memory_order_relaxed);
    return 0;
}

int legion_dense_mv_supported(void) {
    return legion_device::compute_capability() >= 90 && mma_code() ? 1 : 0;
}

int legion_dense_mv_launch(const legion_dense_mv * p, void * stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (p->n > MAX_N) {
        return launch_wide(*p, s);
    }
    const bool f32 = p->f32 != 0;
    const bool wb = p->f32 == 2;
    if (p->f32 < 0 || p->f32 > 2 || p->nw < 1 || p->nw > MAX_W || p->n < (f32 ? 4 : 2) || p->n > MAX_N || p->k <= 0 ||
        p->k % (f32 ? 32 : 64) != 0 || p->x_ld % 2 != 0) {
        return 1;
    }
    // Chain weights: BF16 of 9-16 tokens, after at least one other weight.
    if (p->nchain < 0 || p->nchain >= p->nw || (p->nchain > 0 && (f32 || p->n <= 8))) {
        return 1;
    }
    int64_t rows = 0;
    for (int i = 0; i < p->nw; i++) {
        const int tile = i >= p->nw - p->nchain ? 16 : 32;
        if (((!f32 || wb) && p->w_ld[i] % 2 != 0) || p->rows[i] <= 0 || p->rows[i] % tile != 0) {
            return 1;
        }
        rows += p->rows[i];
    }
    if (rows > INT32_MAX) {
        return 1;
    }
    if (!mma_code()) {
        return 1;
    }
    const int f = forced.load(std::memory_order_relaxed);
    const int r = wb ? launch_typed<true, true>(*p, s, f) :
                  f32 ? launch_typed<true>(*p, s, f) :
                        launch_typed<false>(*p, s, f);
    if (r != 0) {
        return r;
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}
