// Q8_0 skinny matmul for MTP draft passes (#836 part 13): a Q8_0 weight times 9-16 columns of
// activations that ggml's quantize_mmq_q8_1 already quantized (MMQ's block_q8_1_mmq layout,
// MMQ_Q8_1_DS_LAYOUT_D4), with the bits of ggml-cuda's MMQ (mmq.cuh, mmq-vec-dot.cuh, MIT).
// ggml takes MMQ for these column counts (MMVQ up to 8) and computes, for each output, a sum
// over the row's 32-value groups in ascending order of
//     (float) idot * dA * dB        (sum[...] += C.x[l]*dA*dB in ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma)
// where idot is the exact integer dot product of the group's int8 values (an int8 MMA there,
// dp4a here: integer sums, the same value), dA the weight group's F16 scale as F32 and dB the
// activation group's F32 scale. Row and column tiles do not change that order, so any tiling
// gives ggml's bits; the same expression under the same flags (--use_fast_math) compiles to the
// same FMUL + FFMA.
// What differs is how the weight streams in. ggml's MMQ at 16 columns runs one CTA of 128 rows
// per SM, loads each 256-value K tile with 2-byte loads and computes it between barriers: on a
// GB10 it reads the MTP head's 65,536-row LM head at ~188 GB/s. Here each warp owns 32 rows (one
// per lane) and streams them through shared memory with cp.async, CB blocks of each row per
// stage, NS stages ahead; one kernel per column count, so the columns' loads overlap.
//
// Shapes are checked here and in src/model/core/q8_mv.rs, which decides where the op runs.

#include "device_cache.cuh"

#include <cuda_fp16.h>
#include <cstdint>

extern "C" {
struct legion_q8_mv {
    const void * w;     // Q8_0 rows of k values (34-byte blocks), w_ld bytes apart
    int64_t      w_ld;  // a multiple of 4
    int64_t      rows;  // a multiple of 32
    const void * yq;    // n columns quantized by ggml's quantize_mmq_q8_1 (D4 layout)
    float *      dst;   // F32 [rows, n], columns rows apart
    int64_t      k;     // a multiple of 512
    int32_t      n;     // columns, 9-16
};

// Enqueues dst = w . y on `stream`. 0 on success; 1 for a shape or device the kernel does not
// take; a CUDA error code for a failed launch.
int legion_q8_mv_launch(const legion_q8_mv * p, void * stream);

// Whether the current device runs the kernel: compute capability 8.0+ (cp.async) and a binary
// with sm_80+ code for it.
int legion_q8_mv_supported(void);

// Launches every op from now on with stages of `cb` blocks per row, `warps` warps per CTA and
// `stages` stages in flight per warp (see launch_variant; the default (8, 4, 2)), or the
// default for cb = 0: for tests and tuning, the same bits either way. 0 on success, 1 for a
// variant the kernel lacks.
int legion_q8_mv_force(int cb, int warps, int stages);

// The current device's SM count (ggml-cuda's `nsm`, which picks MMQ's stream-k grid); 0 when
// unknown.
int legion_q8_mv_sms(void);
}

namespace {

// Columns the kernel is built for: MMQ's range (ggml-cuda takes MMVQ up to 8).
constexpr int MIN_N = 9;
constexpr int MAX_N = 16;
// Bytes of a Q8_0 block: an F16 scale, 32 int8.
constexpr int QB = 34;
// A stage holds CB blocks of each of a warp's 32 rows (CB = 2, 4 or 8: 68, 136 or 272 bytes, the
// runs of whole blocks that are multiples of 4, 8 and 16 bytes), copied as 17 pieces of U bytes
// per row; rows lie CHUNK bytes apart, which lets a lane read its row with U-byte loads without
// bank conflicts. Small stages keep many warps on an SM, large ones copy in larger pieces.
template <int CB> struct shape {
    static constexpr int CHUNK = CB * QB;
    static constexpr int U = CHUNK / 17;
    static constexpr int WORDS = CHUNK / 4;
    static constexpr int STAGE = 32 * CHUNK;
};
// block_q8_1_mmq: 4 F32 scales, then 128 int8.
constexpr int YB = 144;

#include "sm80/q8_mv.cuh"

// Whether this binary has sm_80+ code for the kernel.
bool sm80_code() {
    static std::atomic<int8_t> state[GGML_CUDA_MAX_DEVICES];  // 0 not yet, 1 yes, 2 no
    const int dev = legion_device::current();
    if (dev < 0) {
        return false;
    }
    int8_t st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        cudaFuncAttributes fa{};
        const cudaError_t before = cudaPeekAtLastError();
        const bool code = cudaFuncGetAttributes(&fa, q8_mv<2, 8, 2, 16>) == cudaSuccess && fa.binaryVersion >= 80 &&
                          fa.ptxVersion >= 80;
        if (!code) {
            legion_device::clear_own_error(before);
        }
        st = code ? 1 : 2;
        state[dev].store(st, std::memory_order_relaxed);
    }
    return st == 1;
}

// The variant forced by legion_q8_mv_force: 0 none, else cb | warps << 8 | stages << 16.
std::atomic<int> forced{0};

template <int CB, int W, int NS, int N>
int launch_n(const legion_q8_mv & p, cudaStream_t s) {
    static std::atomic<int8_t> opt[GGML_CUDA_MAX_DEVICES];
    constexpr int smem = W * NS * shape<CB>::STAGE;
    if (!legion_device::smem_opt_in(opt, q8_mv<CB, W, NS, N>, smem)) {
        return 1;
    }
    const int64_t tiles = p.rows / 32;
    q8_mv<CB, W, NS, N><<<(unsigned) ((tiles + W - 1) / W), dim3(32, W), smem, s>>>(p);
    return 0;
}

// The kernel of CB-block stages, `W` warps and `NS` stages for p.n columns (MIN_N to MAX_N);
// 1 where the weight's rows are not aligned to its copies.
template <int CB, int W, int NS>
int launch(const legion_q8_mv & p, cudaStream_t s) {
    constexpr int U = shape<CB>::U;
    if (p.w_ld % U != 0 || ((uintptr_t) p.w) % U != 0) {
        return 1;
    }
    switch (p.n) {
        case 9:
            return launch_n<CB, W, NS, 9>(p, s);
        case 10:
            return launch_n<CB, W, NS, 10>(p, s);
        case 11:
            return launch_n<CB, W, NS, 11>(p, s);
        case 12:
            return launch_n<CB, W, NS, 12>(p, s);
        case 13:
            return launch_n<CB, W, NS, 13>(p, s);
        case 14:
            return launch_n<CB, W, NS, 14>(p, s);
        case 15:
            return launch_n<CB, W, NS, 15>(p, s);
        case 16:
            return launch_n<CB, W, NS, 16>(p, s);
        default:
            return 1;
    }
}

// The variants: (cb, warps, stages), at most 68 KiB of shared memory per CTA.
int launch_variant(const legion_q8_mv & p, cudaStream_t s, int cb, int w, int ns) {
    switch (cb << 16 | w << 8 | ns) {
        case 8 << 16 | 4 << 8 | 2:
            return launch<8, 4, 2>(p, s);
        case 8 << 16 | 2 << 8 | 2:
            return launch<8, 2, 2>(p, s);
        case 4 << 16 | 8 << 8 | 2:
            return launch<4, 8, 2>(p, s);
        case 4 << 16 | 4 << 8 | 2:
            return launch<4, 4, 2>(p, s);
        case 2 << 16 | 8 << 8 | 2:
            return launch<2, 8, 2>(p, s);
        case 2 << 16 | 16 << 8 | 2:
            return launch<2, 16, 2>(p, s);
        default:
            return -1;
    }
}

// The default variant, and the next smaller stages where the rows are not aligned for it.
int launch_default(const legion_q8_mv & p, cudaStream_t s) {
    int r = launch<8, 4, 2>(p, s);
    if (r == 1) {
        r = launch<4, 8, 2>(p, s);
    }
    if (r == 1) {
        r = launch<2, 8, 2>(p, s);
    }
    return r;
}

}  // namespace

int legion_q8_mv_force(int cb, int warps, int stages) {
    if (cb == 0) {
        forced.store(0, std::memory_order_relaxed);
        return 0;
    }
    // A probe launch would need a device; the variant table is checked by its shape instead.
    const bool known = (cb == 8 && (warps == 4 || warps == 2) && stages == 2) ||
                       (cb == 4 && (warps == 8 || warps == 4) && stages == 2) ||
                       (cb == 2 && (warps == 8 || warps == 16) && stages == 2);
    if (!known) {
        return 1;
    }
    forced.store(cb | warps << 8 | stages << 16, std::memory_order_relaxed);
    return 0;
}

int legion_q8_mv_sms(void) {
    return legion_device::sm_count();
}

int legion_q8_mv_supported(void) {
    return legion_device::compute_capability() >= 80 && sm80_code() ? 1 : 0;
}

int legion_q8_mv_launch(const legion_q8_mv * p, void * stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (p->n < MIN_N || p->n > MAX_N || p->k <= 0 || p->k % 512 != 0 || p->rows <= 0 || p->rows % 32 != 0 ||
        p->w_ld % 4 != 0 || p->w_ld < p->k / 32 * QB || ((uintptr_t) p->w) % 4 != 0 ||
        ((uintptr_t) p->yq) % 16 != 0 || p->rows / 32 > INT32_MAX) {
        return 1;
    }
    if (!sm80_code()) {
        return 1;
    }
    const int f = forced.load(std::memory_order_relaxed);
    const int r = f != 0 ? launch_variant(*p, s, f & 0xff, (f >> 8) & 0xff, f >> 16) : launch_default(*p, s);
    if (r < 0) {
        return 1;
    }
    if (r != 0) {
        return r;
    }
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}
