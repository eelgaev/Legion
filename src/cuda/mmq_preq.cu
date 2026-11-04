// ggml-cuda's MMQ on activations that Legion's kernels have already quantized to Q8_1 in MMQ's
// layout (block_q8_1_mmq), so the matmul skips ggml's own quantize_mmq_q8_1 pass over its F32
// input: qwen4exp's hyper-connection mixer quantizes its normed streams in the kernel that
// writes them (src/cuda/fused.cu, hc_post_norm) for the 10240 -> 320 down matmul (#363).
//
// OWNER-GRANTED EXCEPTION (#363) to "no new C/C++ host code": this file is the minimum glue to
// call ggml's mul_mat_q_case with mmq_args and the backend's pool, which have no C API. Nothing
// else lives here; when to take it and every buffer's size are decided in Rust
// (src/model/core/mmq_preq.rs).
//
// The tile width and the arguments are ggml_cuda_mul_mat_q's for a dense src1 without ids
// (third_party/ggml/src/ggml-cuda/mmq.cu, MIT), so the same MMQ kernel computes the same bits.

#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"

#include <climits>

extern "C" {

// dst ([ne01, ne11] F32, columns nb1 bytes apart) = src0 (a Q8_0 weight [ne00, ne01]) times the
// ne11 columns in yq, ne00 values each, quantized as quantize_mmq_q8_1 does for a Q8_0 src0
// (MMQ_Q8_1_DS_LAYOUT_D4: column j's block of values 128 b .. 128 b + 127 at block b * ne11 + j),
// computed on the backend's stream. yq holds yq_bytes: the blocks and the padding MMQ's tiles
// may read past them. 0 on success; -1 when ggml_cuda_mul_mat would not take MMQ for this
// matmul (another type, ne00 not a multiple of 512, src0 not 2D), -2 when yq's padding is
// short; a CUDA error otherwise.
int legion_mmq_preq(ggml_backend_t backend, const ggml_tensor * src0, const void * yq, size_t yq_bytes,
                    int64_t ne11, float * dst, int64_t nb1) {
    auto & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx.device);
    const cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const size_t smpbo = ggml_cuda_info().devices[ctx.device].smpbo;
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    if (src0->type != GGML_TYPE_Q8_0 || src0->ne[2] != 1 || src0->ne[3] != 1 || ne00 % MATRIX_ROW_PADDING != 0 ||
        ne11 < 1 || !ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return -1;
    }

    // ggml_cuda_mul_mat_q: if src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            if (!ggml_is_contiguously_allocated(src0) || src0->view_src) {
                return -1;
            }
            const cudaError_t e = cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream);
            if (e != cudaSuccess) {
                return (int) e;
            }
        }
    }

    // ggml_cuda_mul_mat_q's tile width for ne11 columns (src1 at Q8_1: Q8_0 never takes Q4).
    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);
    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ntiles_J_best = INT_MAX;
        for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, GGML_PREC_Q8);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }
            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
                continue;
            }
            const int64_t ntiles_x = (ne11 + config.J - 1) / config.J;
            if (ntiles_x < ntiles_J_best) {
                J_best        = J;
                nthreads_best = config.nthreads;
                ntiles_J_best = ntiles_x;
            }
        }
    }
    if (J_best == 0) {
        return -1;
    }
    const size_t chunk   = nthreads_best * sizeof(int);
    const size_t padding = (J_best * sizeof(block_q8_1_mmq) + chunk - 1) / chunk * chunk;
    const size_t data    = (size_t) ne11 * ne00 / QK8_1_MMQ * sizeof(block_q8_1_mmq);
    if (yq_bytes < data + padding) {
        return -2;
    }

    const int64_t ts_src0 = ggml_type_size(src0->type);
    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s1  = nb1 / (int64_t) sizeof(float);
    const int64_t s2  = s1 * ne11;
    const int64_t s3  = s2;
    const int64_t s12 = ne11 * ne00 * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = s12;

    const mmq_args args = {
        (const char *) src0->data, src0->type, (const int *) yq, nullptr, nullptr, dst,
        nullptr,
        ne00, ne01, ne11, s01, ne11, s1,
        1, 1, s02, s12, s2,
        1, 1, s03, s13, s3,
        ne11, J_best};
    mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
    return (int) cudaGetLastError();
}

// ggml's own quantize_mmq_q8_1 of x ([ne10, ne11] F32, columns contiguous, ne10 a multiple of
// 512) for a Q8_0 src0 into vy, on `stream`: what ggml_cuda_mul_mat_q computes before its MMQ.
// For the tests and the check of the fused quantization against it.
int legion_ggml_quantize_mmq_q8_0(const float * x, void * vy, int64_t ne10, int64_t ne11, void * stream) {
    if (ne10 % MATRIX_ROW_PADDING != 0 || ne11 < 1) {
        return -1;
    }
    quantize_mmq_q8_1_cuda(x, nullptr, vy, GGML_TYPE_Q8_0, ne10, ne10, ne10 * ne11, ne10 * ne11, ne10, ne11, 1, 1,
                           (cudaStream_t) stream);
    return (int) cudaGetLastError();
}

}  // extern "C"
