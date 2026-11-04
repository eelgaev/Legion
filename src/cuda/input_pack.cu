// input_pack: a step's small inputs, written by the host into pinned memory, copied into their
// graph tensors by one kernel instead of one cudaMemcpyAsync each (src/ggml/pinned_io/pack.rs,
// #814). The kernel reads the pinned memory over the host link (the device address of a
// cudaMallocHost buffer). Each thread block copies one piece of at most PACK_CHUNK bytes, each
// thread 16 bytes of it; the bytes are the same as the copies'. Kernel only: Rust launches it with
// cudaLaunchKernel through `legion_input_pack_kernel`.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int PACK_THREADS = 256;
// Bytes a piece holds at most (pack::CHUNK): 16 per thread.
constexpr uint32_t PACK_CHUNK = PACK_THREADS * 16;
// Pieces one launch takes (pack::MAX_PIECES): the arguments stay under 4 KB.
constexpr int PACK_MAX = 200;

// pack::Piece: `len` bytes from offset `src` of the pinned buffer into device address `dst`.
struct Piece {
    uint8_t * dst;
    uint32_t src;
    uint32_t len;
};

// pack::Args.
struct Args {
    const uint8_t * base;
    Piece pieces[PACK_MAX];
};

static_assert(PACK_CHUNK == 4096, "pack::CHUNK");
static_assert(sizeof(Piece) == 16, "pack::Piece");
static_assert(sizeof(Args) == 8 + 16 * PACK_MAX, "pack::Args");

__global__ void __launch_bounds__(PACK_THREADS) input_pack_kernel(const Args a) {
    const Piece p = a.pieces[blockIdx.x];
    const uint32_t at = threadIdx.x * 16u;
    if (at >= p.len) {
        return;
    }
    const uint8_t * s = a.base + p.src + at;
    uint8_t * d = p.dst + at;
    if (at + 16u <= p.len && (((uintptr_t) s | (uintptr_t) d) & 15) == 0) {
        *(uint4 *) d = *(const uint4 *) s;
        return;
    }
    const uint32_t n = min(16u, p.len - at);
    for (uint32_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

}  // namespace

// The kernel as the CUDA runtime knows it (its host stub), for Rust's cudaLaunchKernel. It sits in
// an anonymous namespace: with relocatable device code the device link left extern "C" kernels out
// of the linked image (as deepseek41.cu's table).
extern "C" const void * const legion_input_pack_kernel = reinterpret_cast<const void *>(&input_pack_kernel);
