// qwen4exp (Qwen3.8-Flash-Next) hyper-connection mixer, prompt steps: the up matmul and the
// gated mix in one kernel, instead of ggml's mul_mat (MMQ, which writes the gate: hc * n_embd
// F32 values per token) and dsv4_hc_pre (which reads the gate and xn back). The host side, the
// shapes, when it runs and the CPU reference are in src/model/qwen4/hc_up_mix.rs:
//
//   lq, ld = lo quantized as ggml-cuda's quantize_mmq_q8_1 does for a Q8_0 weight
//   gate   = up . lo                          up: Q8_0 [rank -> hc * n_embd], MMQ's sums
//   mixed  = scale * sum_c xn[c] * sigmoid(gate[c])
//
// xn is read, or rebuilt as (res * norm scale) * gamma from the streams and scales hc_post_norm
// writes in its place (up_mix_mma<true>, #526; up_mix_bf16<true> for BF16 up weights).
//
// Two kernels:
// - `quantize_lo`: a warp per (token, group of 32): ggml's expressions (127 / amax, roundf, the
//   scale as 1 / d_inv), into [group][token][32] quants and [group][token] scales.
// - `up_mix_mma` (sm80/hc_up_mix.cuh): the matmul on int8 MMA with MMQ's float sums, then the
//   mix.
// BF16 up weights (the NVFP4 checkpoints, #836) take `lo_bf16` and `up_mix_bf16` instead
// (sm80/hc_up_mix_bf16.cuh, on by default): the BF16 GEMM's m16n8k16 products in K order, then
// the same mix, cuBLAS's sums at the mixer's shape.
//
// The same bits as ggml's two ops wherever ggml's MMQ computes each output tile in one CTA
// (no stream-k split, whose fixup adds the partial sums in another order); the Rust side takes
// this op only there.

#include "device_cache.cuh"
#include "ggml-cuda/common.cuh"

#include <atomic>
#include <cstdint>

extern "C" {

struct legion_hc_up_mix {
    const char *  up;      // Q8_0 [hc * e rows][rank], up_nb1 bytes per row
    int64_t       up_nb1;
    const float * lo;      // F32 [n][rank], rows contiguous
    const float * xn;      // F32 [n][hc][e], contiguous
    float *       mixed;   // F32 [n][e]
    int8_t *      lq;      // scratch: lo's quants, [rank / 32][n][32]
    float *       ld;      // scratch: lo's scales, [rank / 32][n]
    int32_t       e, hc, rank, n;
    float         scale;
    // Null (xn is read), or xn is rebuilt from what hc_post_norm wrote instead of it (fused.cu,
    // scale_at): res F32 [n][hc][e] (the new streams), sc F32 [n][hc] (each stream's norm
    // scale) and g F32 [hc][e] (gamma), xn = (res * sc) * g with the norm kernels' two rounded
    // multiplies, so the same bits.
    const float * res;
    const float * sc;
    const float * g;
    // Nonzero: res holds BF16 values ([n][hc][e] __nv_bfloat16, hc_post_norm's BF16 streams,
    // #826 `legion serve --hc-streams-bf16`, #1183 `--hc-bf16` for BF16 up rows), each widened
    // to F32 before the rebuild.
    int32_t       res_bf16;
    // Nonzero: up holds BF16 rows ([hc * e rows][rank] __nv_bfloat16, the NVFP4 checkpoints'
    // mixers, #836) and the matmul is a BF16 MMA GEMM (sm80/hc_up_mix_bf16.cuh): lo is
    // rounded into lob, lq / ld are unused, and xn is read or rebuilt from the streams (res,
    // sc, g; F32, or BF16 with res_bf16: `legion serve --hc-bf16`, #1183).
    int32_t       up_bf16;
    void *        lob;     // scratch: lo as BF16, [n][rank]
    // Nonzero: lob already holds lo as BF16 (written by legion_hc_up_lob_launch in a node of
    // its own, the place ggml's gate takes, #836), so the launch does not round lo again.
    int32_t       lob_ready;
    // BF16 up only (#1183, the node fusion's mix cast): null, or mixed rounded to BF16 as
    // ggml-cuda's F32 -> BF16 copy rounds it (__float2bfloat16), [n][e] __nv_bfloat16, written
    // beside mixed; with skip_f32 nonzero mixed itself is not written (nothing reads it).
    void *        mixedb;
    int32_t       skip_f32;
};

// Launches the kernels on `stream`; returns a cudaError_t (cudaErrorInvalidValue for shapes the
// kernel does not take, cudaErrorNoKernelImageForDevice without sm_80+ code for the device).
int legion_hc_up_mix_launch(const legion_hc_up_mix * p, void * stream);

// Only lo_bf16: lo (F32 [n][rank]) rounded to BF16 into lob, for a BF16 up_mix node whose
// lob_ready is set.
int legion_hc_up_lob_launch(const legion_hc_up_mix * p, void * stream);

}  // extern "C"

namespace {

constexpr int HC     = 4;    // streams
constexpr int ROWS   = 16;   // mixed columns per CTA: one m16 tile per stream
constexpr int WARPS  = 4;
constexpr int NT     = 4;    // n8 token tiles per warp tile: 32 tokens
constexpr int CHUNK  = 128;  // tokens per CTA: a warp tile per warp
constexpr int PAD    = 16;   // bytes after each staged up row
constexpr int MAX_KB = 16;   // rank up to 512
constexpr int MAX_RANK_BF16 = 368;  // BF16 up rows staged in 48 KiB of shared memory

__global__ void quantize_lo(const legion_hc_up_mix p) {
    const int     nkb  = p.rank / 32;
    const int64_t w    = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;  // t * nkb + kb
    const int     lane = threadIdx.x % 32;
    if (w >= (int64_t) p.n * nkb) {
        return;
    }
    const int64_t t = w / nkb;
    const int     kb = (int) (w % nkb);
    const float   x  = p.lo[w * 32 + lane];
    // fmaxf is exact, so the butterfly gives ggml's amax.
    float amax = fabsf(x);
#pragma unroll
    for (int m = 16; m >= 1; m /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    const float       d_inv = 127.0f / amax;
    const signed char qv    = roundf(x * d_inv);
    const int64_t     at    = (int64_t) kb * p.n + t;
    p.lq[at * 32 + lane]    = qv;
    if (lane == 0) {
        p.ld[at] = 1.0f / d_inv;
    }
}

#include "sm80/hc_up_mix.cuh"
#include "sm80/hc_up_mix_bf16.cuh"

// Whether this binary has sm_80+ code for up_mix_mma on the current device.
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
        const bool code = cudaFuncGetAttributes(&fa, up_mix_mma<false>) == cudaSuccess && fa.binaryVersion >= 80 &&
                          fa.ptxVersion >= 80;
        if (!code) {
            legion_device::clear_own_error(before);
        }
        st = code ? 1 : 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

}  // namespace

extern "C" int legion_hc_up_mix_launch(const legion_hc_up_mix * p, void * stream) {
    auto s = (cudaStream_t) stream;
    if (p->up_bf16 != 0) {
        const bool aligned = (uintptr_t) p->up % 16 == 0 && p->up_nb1 % 16 == 0 && (uintptr_t) p->lob % 16 == 0;
        if (p->hc != HC || p->rank % 16 != 0 || p->rank < 16 || p->rank > MAX_RANK_BF16 || p->e % ROWS != 0 ||
            p->n < 1 || (p->res == nullptr && p->res_bf16 != 0) || !aligned) {
            return cudaErrorInvalidValue;
        }
        if (!mma_code()) {
            return cudaErrorNoKernelImageForDevice;
        }
        if (p->lob_ready == 0) {
            const int64_t pairs = (int64_t) p->n * p->rank / 2;
            lo_bf16<<<(unsigned) ((pairs + 255) / 256), 256, 0, s>>>(*p);
        }
        const int  smem = HC * ROWS * (p->rank / 2 + PAD / 4) * (int) sizeof(uint32_t);
        const dim3 grid(p->e / ROWS, (p->n + CHUNK - 1) / CHUNK);
        if (p->res != nullptr && p->res_bf16 != 0) {
            up_mix_bf16<true, true><<<grid, WARPS * 32, smem, s>>>(*p);
        } else if (p->res != nullptr) {
            up_mix_bf16<true><<<grid, WARPS * 32, smem, s>>>(*p);
        } else {
            up_mix_bf16<false><<<grid, WARPS * 32, smem, s>>>(*p);
        }
        return (int) cudaGetLastError();
    }
    // (The BF16 copy of mixed is the BF16 kernel's only.)
    if (p->mixedb != nullptr || p->skip_f32 != 0) {
        return cudaErrorInvalidValue;
    }
    if (p->hc != HC || p->rank % 32 != 0 || p->rank < 32 || p->rank > 32 * MAX_KB || p->e % ROWS != 0 || p->n < 1) {
        return cudaErrorInvalidValue;
    }
    if (!mma_code()) {
        return cudaErrorNoKernelImageForDevice;
    }
    const int     nkb    = p->rank / 32;
    const int64_t groups = (int64_t) p->n * nkb;
    quantize_lo<<<(unsigned) ((groups + 7) / 8), 8 * 32, 0, s>>>(*p);
    const int smem = HC * ROWS * (p->rank + PAD) + HC * ROWS * nkb * (int) sizeof(float);
    const dim3 grid(p->e / ROWS, (p->n + CHUNK - 1) / CHUNK);
    if (p->res != nullptr && p->res_bf16 != 0) {
        up_mix_mma<true, true><<<grid, WARPS * 32, smem, s>>>(*p);
    } else if (p->res != nullptr) {
        up_mix_mma<true><<<grid, WARPS * 32, smem, s>>>(*p);
    } else {
        up_mix_mma<false><<<grid, WARPS * 32, smem, s>>>(*p);
    }
    return (int) cudaGetLastError();
}

extern "C" int legion_hc_up_lob_launch(const legion_hc_up_mix * p, void * stream) {
    if (p->rank % 16 != 0 || p->rank < 16 || p->n < 1 || (uintptr_t) p->lob % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t pairs = (int64_t) p->n * p->rank / 2;
    lo_bf16<<<(unsigned) ((pairs + 255) / 256), 256, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}
