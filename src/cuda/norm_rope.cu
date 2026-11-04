// norm_rope and pool_key: an attention head's RMS norm, its weight and the multi-section RoPE
// in one kernel (upstream Strata #783 PR-f, "fused rms_norm + rope", by stuchapin909, MIT, via
// Strata-AC922), as ggml-cuda computes the chain it replaces, bit for bit:
//
// - rms_norm_f32<256, true> (norm.cu): 256 threads per row, each summing x^2 over its columns
//   (col = tid, tid + 256, ...), warp_reduce_sum, the eight warp sums reduced by warp 0, mean =
//   sum / ncols, rsqrtf(mean + eps), y = scale * x * w;
// - rope_multi (rope.cu) with freq_scale 1, ext_factor 0, attn_factor 1, no frequency factors:
//   the pair (i, i + n_dims / 2) of the rotated window rotated by pos * theta_scale^i (theta_scale
//   = powf(freq_base, -2 / n_dims), computed on the host as ggml does), the section of i picking
//   the position row; columns past n_dims copied.
//
// pool_key also does what comes before the norm in qwen4exp's pool keys: the kpool member rows
// read from the F16 indexer cache (get_rows), summed in order (cont, add, add, ...), scaled by
// 1 / kpool (ggml's scale: scale * x + bias, bias 0), and after the RoPE the F16 store into the
// pool-key half of the last member's cache row (set_rows). One launch instead of about nine.
//
// kv_append and idx_append are the batched K/V and indexer appends of upstream Strata #783
// (same author and licence): a layer's K (normed and rotated here) and V written to the F16
// caches in one launch, and the indexer rows (raw key, zero pool-key half) in one.
//
// Shapes, the CPU references and the parity tests: src/model/core/norm_rope.rs and its tests.rs.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

extern "C" {

struct legion_rope_params {
    int32_t n_dims;       // rotated columns (ggml's n_dims), even, <= ncols
    int32_t sections[4];  // ggml's mrope sections
    int32_t imrope;       // 1: GGML_ROPE_TYPE_IMROPE (interleaved sections), 0: MROPE
    float freq_base;
    float freq_scale;     // 1
    float attn_factor;    // 1
};

struct legion_norm_rope {
    const float * x;      // [ncols, n_head, n_tok]: (c, h, t) at x + c + h * x_s1 + t * x_s2
    int64_t x_s1, x_s2;
    const float * w;      // [ncols]
    const int32_t * pos;  // [4 * n_tok]: section s of token t at pos[t + s * n_tok]
    float * out;          // [ncols, n_head, n_tok] contiguous
    int32_t ncols, n_head, n_tok;
    float eps;
    legion_rope_params rope;
};

int legion_norm_rope_launch(const legion_norm_rope * p, void * stream);

struct legion_pool_key {
    const __half * raw;       // raw keys: row r at raw + r * ld (halves)
    const int32_t * members;  // [kpool * n_new]: pool j's member rows at j * kpool
    const int32_t * pos;      // [4 * n_new]: as norm_rope's
    const int64_t * rep;      // [n_new]: the row pool j's key goes to
    __half * pooled;          // pool keys: row r at pooled + r * ld (halves)
    int64_t ld;
    const float * w;          // [ncols]
    int32_t ncols, kpool, n_new;
    float inv_kpool;          // 1 / kpool as ggml_scale takes it
    float eps;
    legion_rope_params rope;
};

int legion_pool_key_launch(const legion_pool_key * p, void * stream);

// kv_append: a layer's step K and V appended to its F16 caches in one launch, K normed and
// rotated on the way (norm_rope's arithmetic): what rms_norm, mul, rope_multi and two set_rows
// (F32 -> F16, __float2half as ggml-cuda's set_rows converts) do.
struct legion_kv_append {
    const float * k;      // [ncols, n_head, n_tok] before the norm: (c, h, t) at k + c + h * k_s1 + t * k_s2
    int64_t k_s1, k_s2;
    const float * w;      // [ncols]
    const int32_t * pos;  // [4 * n_tok]
    const float * v;      // as k: (c, h, t) at v + c + h * v_s1 + t * v_s2
    int64_t v_s1, v_s2;
    __half * k_cache;     // cell c's row at k_cache + c * k_ld (halves): n_head heads of ncols
    __half * v_cache;
    int64_t k_ld, v_ld;
    const int32_t * cells;  // [n_tok]
    int32_t ncols, n_head, n_tok;
    float eps;
    legion_rope_params rope;
};

int legion_kv_append_launch(const legion_kv_append * p, void * stream);

// idx_append: qwen4exp's indexer cache rows of a step, the raw key then a zero pool-key half
// (ggml's fill, concat and set_rows), in one launch.
struct legion_idx_append {
    const float * raw;      // [d, n_tok]: token t's key at raw + t * raw_s1
    int64_t raw_s1;
    __half * cache;         // cell c's row (2 d halves) at cache + c * ld
    int64_t ld;
    const int32_t * cells;  // [n_tok]
    int32_t d, n_tok;
};

int legion_idx_append_launch(const legion_idx_append * p, void * stream);

}  // extern "C"

namespace {

constexpr int NR_BLOCK = 256;   // ggml's rms_norm block for rows under 1024 columns
constexpr int NR_MAX_COLS = 1023;

__device__ __forceinline__ float nr_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// ggml's block_reduce<SUM, 256>.
__device__ __forceinline__ float nr_block_sum(float v, float * s_sum) {
    v = nr_warp_sum(v);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        s_sum[warp] = v;
    }
    __syncthreads();
    v = 0.0f;
    if (lane < NR_BLOCK / 32) {
        v = s_sum[lane];
    }
    return nr_warp_sum(v);
}

// The device-side rope parameters (theta_scale from the host).
struct nr_rope {
    int n_dims;
    int sections[4];
    int imrope;
    float theta_scale;
    float freq_scale;
    float attn_factor;
};

// Rotates row y (shared, ncols floats, normed) into out (global) as ggml's rope_multi; positions
// of this token: pos[t + s * n_pos].
__device__ __forceinline__ void nr_rotate(const float * y, float * out, int ncols, const nr_rope & r,
                                          const int32_t * pos, int t, int n_pos) {
    const int half_dims = r.n_dims / 2;
    for (int i = threadIdx.x; 2 * i < ncols; i += NR_BLOCK) {
        const int i0 = 2 * i;
        if (i0 >= r.n_dims) {
            out[i0 + 0] = y[i0 + 0];
            out[i0 + 1] = y[i0 + 1];
            continue;
        }
        const int iw = i0;
        const int sect_dims = r.sections[0] + r.sections[1] + r.sections[2] + r.sections[3];
        const int sec_w = r.sections[1] + r.sections[0];
        const int sector = (iw / 2) % sect_dims;
        float theta_base = 0.0;
        if (r.imrope) {
            if (sector % 3 == 1 && sector < 3 * r.sections[1]) {
                theta_base = pos[t + n_pos * 1] * powf(r.theta_scale, iw / 2.0f);
            } else if (sector % 3 == 2 && sector < 3 * r.sections[2]) {
                theta_base = pos[t + n_pos * 2] * powf(r.theta_scale, iw / 2.0f);
            } else if (sector % 3 == 0 && sector < 3 * r.sections[0]) {
                theta_base = pos[t] * powf(r.theta_scale, iw / 2.0f);
            } else {
                theta_base = pos[t + n_pos * 3] * powf(r.theta_scale, iw / 2.0f);
            }
        } else {
            if (sector < r.sections[0]) {
                theta_base = pos[t] * powf(r.theta_scale, iw / 2.0f);
            } else if (sector >= r.sections[0] && sector < sec_w) {
                theta_base = pos[t + n_pos * 1] * powf(r.theta_scale, iw / 2.0f);
            } else if (sector >= sec_w && sector < sec_w + r.sections[2]) {
                theta_base = pos[t + n_pos * 2] * powf(r.theta_scale, iw / 2.0f);
            } else if (sector >= sec_w + r.sections[2]) {
                theta_base = pos[t + n_pos * 3] * powf(r.theta_scale, iw / 2.0f);
            }
        }
        const float freq_factor = 1.0f;
        // rope_yarn, forward, ext_factor 0.
        const float theta_extrap = theta_base / freq_factor;
        const float theta = r.freq_scale * theta_extrap;
        const float cos_theta = cosf(theta) * r.attn_factor;
        const float sin_theta = sinf(theta) * r.attn_factor;

        const float x0 = y[i0 / 2 + 0];
        const float x1 = y[i0 / 2 + half_dims];
        out[i0 / 2 + 0] = x0 * cos_theta - x1 * sin_theta;
        out[i0 / 2 + half_dims] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ggml's rms_norm with the weight, y into shared memory; x read through get(col).
template <typename Get>
__device__ __forceinline__ void nr_norm(Get get, const float * w, float * y, int ncols, float eps,
                                        float * s_sum) {
    float tmp = 0.0f;
    for (int col = threadIdx.x; col < ncols; col += NR_BLOCK) {
        const float xi = get(col);
        tmp += xi * xi;
    }
    tmp = nr_block_sum(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = threadIdx.x; col < ncols; col += NR_BLOCK) {
        y[col] = scale * get(col) * w[col];
    }
    __syncthreads();
}

__global__ void __launch_bounds__(NR_BLOCK) norm_rope_kernel(const legion_norm_rope p, const nr_rope r) {
    __shared__ float s_sum[32];
    __shared__ float y[NR_MAX_COLS + 1];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float * x = p.x + h * p.x_s1 + t * p.x_s2;
    nr_norm([&](int col) { return x[col]; }, p.w, y, p.ncols, p.eps, s_sum);
    float * out = p.out + ((int64_t) t * p.n_head + h) * p.ncols;
    // The rotation writes each output once, from y: no sync needed after.
    __shared__ float o[NR_MAX_COLS + 1];
    nr_rotate(y, o, p.ncols, r, p.pos, t, p.n_tok);
    __syncthreads();
    for (int col = threadIdx.x; col < p.ncols; col += NR_BLOCK) {
        out[col] = o[col];
    }
}

__global__ void __launch_bounds__(NR_BLOCK) pool_key_kernel(const legion_pool_key p, const nr_rope r,
                                                            const float bias) {
    __shared__ float s_sum[32];
    __shared__ float y[NR_MAX_COLS + 1];
    __shared__ float o[NR_MAX_COLS + 1];
    const int j = blockIdx.x;
    const int32_t * m = p.members + (int64_t) j * p.kpool;
    // Mean of the members: get_rows (F16 -> F32), cont, adds in order, scale.
    auto get = [&](int col) {
        float s = __half2float(p.raw[(int64_t) m[0] * p.ld + col]);
        for (int i = 1; i < p.kpool; i++) {
            s = s + __half2float(p.raw[(int64_t) m[i] * p.ld + col]);
        }
        return p.inv_kpool * s + bias;
    };
    nr_norm(get, p.w, y, p.ncols, p.eps, s_sum);
    nr_rotate(y, o, p.ncols, r, p.pos, j, p.n_new);
    __syncthreads();
    __half * dst = p.pooled + p.rep[j] * p.ld;
    for (int col = threadIdx.x; col < p.ncols; col += NR_BLOCK) {
        dst[col] = __float2half(o[col]);
    }
}

// Grid (2 n_head, n_tok): K heads first (normed, rotated), then V heads (copied).
__global__ void __launch_bounds__(NR_BLOCK) kv_append_kernel(const legion_kv_append p, const nr_rope r) {
    const int t = blockIdx.y;
    const int64_t cell = p.cells[t];
    if (blockIdx.x >= (unsigned) p.n_head) {
        const int h = blockIdx.x - p.n_head;
        const float * v = p.v + h * p.v_s1 + t * p.v_s2;
        __half * dst = p.v_cache + cell * p.v_ld + (int64_t) h * p.ncols;
        for (int col = threadIdx.x; col < p.ncols; col += NR_BLOCK) {
            dst[col] = __float2half(v[col]);
        }
        return;
    }
    __shared__ float s_sum[32];
    __shared__ float y[NR_MAX_COLS + 1];
    __shared__ float o[NR_MAX_COLS + 1];
    const int h = blockIdx.x;
    const float * x = p.k + h * p.k_s1 + t * p.k_s2;
    nr_norm([&](int col) { return x[col]; }, p.w, y, p.ncols, p.eps, s_sum);
    nr_rotate(y, o, p.ncols, r, p.pos, t, p.n_tok);
    __syncthreads();
    __half * dst = p.k_cache + cell * p.k_ld + (int64_t) h * p.ncols;
    for (int col = threadIdx.x; col < p.ncols; col += NR_BLOCK) {
        dst[col] = __float2half(o[col]);
    }
}

__global__ void idx_append_kernel(const legion_idx_append p) {
    const int t = blockIdx.x;
    const float * raw = p.raw + t * p.raw_s1;
    __half * dst = p.cache + (int64_t) p.cells[t] * p.ld;
    for (int col = threadIdx.x; col < 2 * p.d; col += blockDim.x) {
        dst[col] = __float2half(col < p.d ? raw[col] : 0.0f);
    }
}

nr_rope nr_rope_of(const legion_rope_params & rp) {
    nr_rope r;
    r.n_dims = rp.n_dims;
    for (int i = 0; i < 4; i++) {
        r.sections[i] = rp.sections[i];
    }
    r.imrope = rp.imrope;
    // As ggml-cuda's rope_multi_cuda (host side).
    r.theta_scale = powf(rp.freq_base, -2.0f / rp.n_dims);
    r.freq_scale = rp.freq_scale;
    r.attn_factor = rp.attn_factor;
    return r;
}

bool nr_rope_ok(const legion_rope_params & rp, int ncols) {
    const int sect = rp.sections[0] + rp.sections[1] + rp.sections[2] + rp.sections[3];
    return rp.n_dims > 0 && rp.n_dims % 2 == 0 && rp.n_dims <= ncols && ncols % 2 == 0 && sect > 0;
}

}  // namespace

extern "C" int legion_norm_rope_launch(const legion_norm_rope * p, void * stream) {
    if (p->ncols < 2 || p->ncols > NR_MAX_COLS || p->n_head < 1 || p->n_tok < 1 || p->n_tok > 65535 ||
        !nr_rope_ok(p->rope, p->ncols)) {
        return -1;
    }
    norm_rope_kernel<<<dim3(p->n_head, p->n_tok), NR_BLOCK, 0, (cudaStream_t) stream>>>(*p, nr_rope_of(p->rope));
    return (int) cudaGetLastError();
}

extern "C" int legion_pool_key_launch(const legion_pool_key * p, void * stream) {
    if (p->ncols < 2 || p->ncols > NR_MAX_COLS || p->kpool < 1 || p->n_new < 1 || !nr_rope_ok(p->rope, p->ncols)) {
        return -1;
    }
    pool_key_kernel<<<p->n_new, NR_BLOCK, 0, (cudaStream_t) stream>>>(*p, nr_rope_of(p->rope), 0.0f);
    return (int) cudaGetLastError();
}

extern "C" int legion_kv_append_launch(const legion_kv_append * p, void * stream) {
    if (p->ncols < 2 || p->ncols > NR_MAX_COLS || p->n_head < 1 || 2 * p->n_head > 65535 || p->n_tok < 1 ||
        p->n_tok > 65535 || !nr_rope_ok(p->rope, p->ncols)) {
        return -1;
    }
    kv_append_kernel<<<dim3(2 * p->n_head, p->n_tok), NR_BLOCK, 0, (cudaStream_t) stream>>>(*p, nr_rope_of(p->rope));
    return (int) cudaGetLastError();
}

extern "C" int legion_idx_append_launch(const legion_idx_append * p, void * stream) {
    if (p->d < 1 || p->n_tok < 1 || p->n_tok > 2147483647) {
        return -1;
    }
    idx_append_kernel<<<p->n_tok, 256, 0, (cudaStream_t) stream>>>(*p);
    return (int) cudaGetLastError();
}
