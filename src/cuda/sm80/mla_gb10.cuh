// glm5next.cu, sm_80 and newer: GLM-5.3-Flash's sparse MLA prompt steps as the GB10 runs them
// (mla_gb_kernel, #862). mla_tiled_kernel's arithmetic, every sum with the same terms in the same
// order (the same chunks of MC positions, the same online softmax): the same bits.
//
// Not a standalone header: glm5next.cu includes it inside its anonymous namespace, after
// mla_tiled_kernel (MD, MC, MT_H) and the cp.async helpers (xcp16, xcp_commit, xcp_wait_all).
// The body is behind a __CUDA_ARCH__ >= 800 guard (a __trap() stub below), so the file compiles
// for every target; the launcher takes it on compute capability 12.x (GB10, RTX 50) by
// legion_device::sm12x_device, or on any sm_80+ when asked (ops::MlaKernel).
//
// mla_tiled (one CTA of 8 warps per SM, every warp scoring and then summing values in turn) is
// bound by shared-memory loads: each warp-step of its score loop takes 4 wavefronts for 64
// products (32 positions' rows, and two query rows broadcast), ~5.6K wavefronts a chunk.
// Here the CTA's 8 warps split into two roles that run a chunk apart, synchronised by named
// barriers:
// - score warps (0-3): a lane scores 2 heads x 2 positions (rows read as 16-byte pieces, 8 rows
//   an instruction; queries 4 heads an instruction), 1.5K wavefronts a chunk;
// - value warps (4-7): the online softmax of the chunk the score warps finished (a warp per 4
//   heads, lane = position, as mla_tiled's), then a lane sums 8 heads x 8 latent columns, and
//   the chunk after next copied into the freed buffer by cp.async.
// Shared memory (100,736 bytes): two chunk buffers of 32 rows x 1 KB (16-byte pieces swizzled
// by the row, piece c of row r at c ^ (r % 8): conflict-free for both roles), the 16 query rows
// in F32 (516 words apart), the chunk's scores (then its weights) [position][head] and per head
// alpha and 1 / sum.

constexpr int MG_T = 256;                     // 4 score warps, then 4 value warps
constexpr int MG_QW = MD + 4;                 // query row stride in words
constexpr int MG_KB = MC * MD * 2;            // one chunk buffer: MC rows of MD halves
constexpr int MG_READY = 1, MG_FREE = 2, MG_FULL = 3, MG_VSYNC = 5;  // named barriers (MG_FULL + buffer)

constexpr int mla_gb_smem() {
    return 2 * MG_KB + MT_H * MG_QW * 4 + MC * MT_H * 4 + 2 * MT_H * 4;
}

__device__ __forceinline__ void mg_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mg_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

// The 8 values of a 16-byte piece of F16 latents.
__device__ __forceinline__ void mg_unpack(const uint4 & u, float (&k)[8]) {
    const float2 a = __half22float2(*(const __half2 *) &u.x), b = __half22float2(*(const __half2 *) &u.y);
    const float2 c = __half22float2(*(const __half2 *) &u.z), d = __half22float2(*(const __half2 *) &u.w);
    k[0] = a.x;
    k[1] = a.y;
    k[2] = b.x;
    k[3] = b.y;
    k[4] = c.x;
    k[5] = c.y;
    k[6] = d.x;
    k[7] = d.y;
}

// mla_tiled_kernel's two steps of a score: 4 latent values a step, a pair sum each, in order.
__device__ __forceinline__ void mg_dot(float & d, const float4 & x0, const float4 & x1, const float (&k)[8]) {
    d += x0.x * k[0] + x0.y * k[1];
    d += x0.z * k[2] + x0.w * k[3];
    d += x1.x * k[4] + x1.y * k[5];
    d += x1.z * k[6] + x1.w * k[7];
}

__global__ void __launch_bounds__(MG_T, 1) mla_gb_kernel(const A a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    pdl_wait();
    extern __shared__ __align__(16) char mla_sm[];
    char * sk = mla_sm;                               // [2][MC][MD] F16, pieces swizzled
    float * sq = (float *) (mla_sm + 2 * MG_KB);      // [MT_H][MG_QW]
    float * ss = sq + MT_H * MG_QW;                   // [MC][MT_H] scores, then weights
    float * sa = ss + MC * MT_H;                      // [MT_H] this chunk's alpha, [MT_H] 1 / sum
    const int64_t h = a.ne[1], t = blockIdx.y, h0 = (int64_t) blockIdx.x * MT_H;
    const int64_t n_tok = a.params[0], block = a.params[2];
    const float scale = pf(a, 1);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int64_t s = t / n_tok;
    const char * cache = S(a, 1);
    const int64_t cnb1 = a.src_nb[1][1];
    const int32_t * table = (const int32_t *) (S(a, 2) + s * a.src_nb[2][1]);
    const int32_t * sel = (const int32_t *) (S(a, 3) + t * a.src_nb[3][1]);
    const int count = sel[0];
    const int nch = (count + MC - 1) / MC;
    // Chunk c's rows into buffer c % 2 by 128 threads (i = 0..127): row i / 4, pieces i % 4 + 4 u
    // (zeros past the selection).
    auto load = [&](int c, int i) {
        const int r = i / 4, row = c * MC + r;
        char * dst = sk + (c & 1) * MG_KB + r * (MD * 2);
        const char * src = cache;
        if (row < count) {
            const int64_t pos = sel[1 + row];
            src = cache + ((int64_t) table[pos / block] * block + pos % block) * cnb1;
        }
#pragma unroll
        for (int u = 0; u < MD / 32; u++) {
            const int pc = i % 4 + 4 * u;
            xcp16(dst + 16 * (pc ^ (r & 7)), src + (row < count ? 16 * pc : 0), row < count ? 16 : 0);
        }
        xcp_commit();
    };
    for (int i = tid; i < MT_H * MD; i += MG_T) {
        const int hh = i / MD, c = i % MD;
        sq[hh * MG_QW + c] = ((const float *) (S(a, 0) + t * a.src_nb[0][2] + (h0 + hh) * a.src_nb[0][1]))[c];
    }
    if (tid < 128 && nch > 0) {
        load(0, tid);
    } else if (tid >= 128 && nch > 1) {
        load(1, tid - 128);
    }
    xcp_wait_all();
    __syncthreads();
    if (warp < 4) {
        // Scores: heads ha and ha + 4, positions pa and pa + 8 (both rows' pieces at c ^ pq).
        const int hp = lane / 8, pq = lane % 8;
        const int ha = 8 * (warp / 2) + hp, pa = 16 * (warp % 2) + pq;
        const float4 * qa = (const float4 *) (sq + ha * MG_QW);
        const float4 * qb = (const float4 *) (sq + (ha + 4) * MG_QW);
        for (int c = 0; c < nch; c++) {
            if (c >= 2) {
                mg_sync(MG_FULL + (c & 1), MG_T);
            }
            const uint4 * k0 = (const uint4 *) (sk + (c & 1) * MG_KB + pa * (MD * 2));
            const uint4 * k1 = k0 + 8 * (MD / 8);
            float d00 = 0.0f, d01 = 0.0f, d10 = 0.0f, d11 = 0.0f;  // [head a, b][position 0, 1]
#pragma unroll 2
            for (int pc = 0; pc < MD / 8; pc++) {
                float v0[8], v1[8];
                mg_unpack(k0[pc ^ pq], v0);
                mg_unpack(k1[pc ^ pq], v1);
                const float4 xa0 = qa[2 * pc], xa1 = qa[2 * pc + 1];
                const float4 xb0 = qb[2 * pc], xb1 = qb[2 * pc + 1];
                mg_dot(d00, xa0, xa1, v0);
                mg_dot(d01, xa0, xa1, v1);
                mg_dot(d10, xb0, xb1, v0);
                mg_dot(d11, xb0, xb1, v1);
            }
            mg_sync(MG_FREE, MG_T);
            ss[pa * MT_H + ha] = d00;
            ss[(pa + 8) * MT_H + ha] = d01;
            ss[pa * MT_H + ha + 4] = d10;
            ss[(pa + 8) * MT_H + ha + 4] = d11;
            mg_arrive(MG_READY, MG_T);
        }
        return;
    }
    // Values: the softmax of heads 4 vw .. 4 vw + 3 (lane = position), then the sums of heads
    // 8 hg .. 8 hg + 7 over latent columns 8 vc .. 8 vc + 7.
    const int vw = warp - 4, hg = vw / 2, vc = 32 * (vw % 2) + lane;
    float mx[4], ls[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        mx[j] = -INFINITY;
        ls[j] = 0.0f;
    }
    float acc[8][8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
            acc[j][k] = 0.0f;
        }
    }
    if (nch > 0) {
        mg_arrive(MG_FREE, MG_T);
    }
    bool pending = false;  // chunk c + 1's copies, issued last chunk
    for (int c = 0; c < nch; c++) {
        const int nc = min(MC, count - c * MC);
        mg_sync(MG_READY, MG_T);
        if (pending) {
            xcp_wait_all();
            mg_arrive(MG_FULL + ((c + 1) & 1), MG_T);
            pending = false;
        }
        {
            float * sr = ss + lane * MT_H + 4 * vw;
            const float4 raw = *(const float4 *) sr;
            const float r[4] = {raw.x, raw.y, raw.z, raw.w};
            float pw[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float sc = lane < nc ? r[j] * scale : -INFINITY;
                float cm = sc;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, o));
                }
                const float mn = fmaxf(mx[j], cm);
                const float alpha = expf(mx[j] - mn);
                const float p = lane < nc ? expf(sc - mn) : 0.0f;
                ls[j] = ls[j] * alpha + warp_sum(p);
                mx[j] = mn;
                pw[j] = p;
                if (lane == 0) {
                    sa[4 * vw + j] = alpha;
                }
            }
            *(float4 *) sr = make_float4(pw[0], pw[1], pw[2], pw[3]);
        }
        mg_sync(MG_VSYNC, 128);
        {
            const float4 a0 = ((const float4 *) (sa + 8 * hg))[0], a1 = ((const float4 *) (sa + 8 * hg))[1];
            const float al[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    acc[j][k] = acc[j][k] * al[j];
                }
            }
        }
        const char * kb = sk + (c & 1) * MG_KB;
        for (int pp = 0; pp < nc; pp++) {
            float kv[8];
            mg_unpack(((const uint4 *) (kb + pp * (MD * 2)))[vc ^ (pp & 7)], kv);
            const float4 w0 = ((const float4 *) (ss + pp * MT_H + 8 * hg))[0];
            const float4 w1 = ((const float4 *) (ss + pp * MT_H + 8 * hg))[1];
            const float wj[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    acc[j][k] += wj[j] * kv[k];
                }
            }
        }
        if (c + 1 < nch) {
            mg_arrive(MG_FREE, MG_T);
        }
        if (c + 2 < nch) {
            mg_sync(MG_VSYNC, 128);  // every value warp is done with this buffer
            load(c + 2, tid - 128);
            pending = true;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            sa[MT_H + 4 * vw + j] = ls[j] > 0.0f ? 1.0f / ls[j] : 0.0f;
        }
    }
    mg_sync(MG_VSYNC, 128);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float inv = sa[MT_H + 8 * hg + j];
        float4 * out = (float4 *) ((float *) a.dst + (t * h + h0 + 8 * hg + j) * MD) + 2 * vc;
        out[0] = make_float4(acc[j][0] * inv, acc[j][1] * inv, acc[j][2] * inv, acc[j][3] * inv);
        out[1] = make_float4(acc[j][4] * inv, acc[j][5] * inv, acc[j][6] * inv, acc[j][7] * inv);
    }
#else
    __trap();
#endif
}
