// q8_mv.cu, sm_80 and newer: the kernel. A CTA of W warps; warp t of the grid owns rows
// 32 t .. 32 t + 31, one per lane, and walks them CB blocks (a chunk) at a time:
// - each chunk's 32 rows go to the warp's shared memory with U-byte cp.async copies
//   (consecutive lanes on consecutive pieces), NS chunks in flight;
// - a lane reads its row's chunk into registers (U-byte loads, no bank conflicts), the warp
//   refills the stage with chunk c + NS, and the lane, for each group in order and each column,
//   adds (float) idot * dA * dB to the column's sum, idot from 8 dp4a over the group's int8
//   values and the column's (uniform loads, L1 hits);
// - the sums go straight to dst, consecutive lanes on consecutive rows.
//
// Not a standalone header: q8_mv.cu includes it inside its anonymous namespace, after
// legion_q8_mv and the constants. The kernel body is built only where __CUDA_ARCH__ >= 800 and
// traps elsewhere; sm80_code() in q8_mv.cu checks at run time that this binary has it.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// A U-byte copy (4, 8: through L1; 16: L2 only).
template <int U>
__device__ __forceinline__ void cp_async(void * smem, const void * gmem) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(smem);
    if constexpr (U == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(gmem));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(a), "l"(gmem), "n"(U));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// The F32 value of the F16 in the low half of `h`.
__device__ __forceinline__ float half_lo(uint32_t h) {
    return __half2float(__ushort_as_half((unsigned short) (h & 0xffff)));
}
#endif

// CB blocks per stage, W warps per CTA, NS chunks per warp in shared memory, N columns (p.n: a
// compile-time count, so that the columns' loads and products overlap instead of each waiting
// behind a branch). blockDim = (32, W).
template <int CB, int W, int NS, int N>
__global__ void __launch_bounds__(W * 32) q8_mv(const legion_q8_mv p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int CHUNK = shape<CB>::CHUNK, U = shape<CB>::U, WORDS = shape<CB>::WORDS;
    constexpr int STAGE = shape<CB>::STAGE;
    const int lane = threadIdx.x;
    const int64_t tile = (int64_t) blockIdx.x * W + threadIdx.y;
    if (tile * 32 >= p.rows) {
        return;
    }
    extern __shared__ __align__(16) char smem[];
    char * const stages = smem + threadIdx.y * (NS * STAGE);
    const char * const wt = (const char *) p.w + tile * 32 * p.w_ld;
    const int nc = (int) (p.k / (32 * CB));
    const char * const yq = (const char *) p.yq;

    // Piece i * 32 + lane of a stage: row (i * 32 + lane) / 17, piece (i * 32 + lane) % 17 of it
    // (32 rows of 17 pieces, rows w_ld bytes apart: offsets within 32 rows fit an int).
    int src_off[17];
#pragma unroll
    for (int i = 0; i < 17; ++i) {
        const int q = i * 32 + lane;
        src_off[i] = (q / 17) * (int) p.w_ld + (q % 17) * U;
    }
    // Copies chunk c of the warp's 32 rows into its stage, as one cp.async group (an empty one
    // past the last chunk, so the waits below count the same).
    auto issue = [&](int c) {
        if (c < nc) {
            char * const dst = stages + (c % NS) * STAGE;
            const char * const src = wt + (int64_t) c * CHUNK;
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                cp_async<U>(dst + (i * 32 + lane) * U, src + src_off[i]);
            }
        }
        cp_async_commit();
    };

    float s[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        s[j] = 0.0f;
    }

#pragma unroll
    for (int c = 0; c < NS; ++c) {
        issue(c);
    }
    for (int c = 0; c < nc; ++c) {
        // Chunks c .. c + NS - 1 are in flight; wait for c.
        cp_async_wait<NS - 1>();
        __syncwarp();
        uint32_t wv[WORDS];
        {
            const char * const row = stages + (c % NS) * STAGE + lane * CHUNK;
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                if constexpr (U == 16) {
                    const uint4 v = *(const uint4 *) (row + 16 * i);
                    wv[4 * i] = v.x;
                    wv[4 * i + 1] = v.y;
                    wv[4 * i + 2] = v.z;
                    wv[4 * i + 3] = v.w;
                } else if constexpr (U == 8) {
                    const uint2 v = *(const uint2 *) (row + 8 * i);
                    wv[2 * i] = v.x;
                    wv[2 * i + 1] = v.y;
                } else {
                    wv[i] = *(const uint32_t *) (row + 4 * i);
                }
            }
        }
        // The stage is in registers: refill it while they are used.
        __syncwarp();
        issue(c + NS);
#pragma unroll
        for (int g = 0; g < CB; ++g) {
            // A pair of blocks is 17 words: the even block's scale in the low half of word 0 and
            // its values from byte 2; the odd block's scale in the high half of word 8 and its
            // values in words 9-16.
            const int b0 = 17 * (g / 2);
            int wq[8];
            float dA;
            if (g % 2 == 0) {
                dA = half_lo(wv[b0]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    wq[i] = (int) __byte_perm(wv[b0 + i], wv[b0 + i + 1], 0x5432);
                }
            } else {
                dA = half_lo(wv[b0 + 8] >> 16);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    wq[i] = (int) wv[b0 + 9 + i];
                }
            }
            // The group's index in the row, its 128-value block and its place there.
            const int B = c * CB + g;
            const char * const yb = yq + (int64_t) (B / 4) * N * YB;
            const int gi = B % 4;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const char * const yj = yb + j * YB;
                const float dB = __ldg((const float *) yj + gi);
                const int4 y0 = __ldg((const int4 *) (yj + 16 + 32 * gi));
                const int4 y1 = __ldg((const int4 *) (yj + 32 + 32 * gi));
                // Two chains of integer sums: the same exact value, half the latency.
                int i0 = __dp4a(wq[0], y0.x, 0);
                int i1 = __dp4a(wq[4], y1.x, 0);
                i0 = __dp4a(wq[1], y0.y, i0);
                i1 = __dp4a(wq[5], y1.y, i1);
                i0 = __dp4a(wq[2], y0.z, i0);
                i1 = __dp4a(wq[6], y1.z, i1);
                i0 = __dp4a(wq[3], y0.w, i0);
                i1 = __dp4a(wq[7], y1.w, i1);
                const int idot = i0 + i1;
                // ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma: sum += C.x[l]*dA*dB.
                s[j] += (float) idot * dA * dB;
            }
        }
    }
    float * const dst = p.dst + tile * 32 + lane;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        dst[(int64_t) j * p.rows] = s[j];
    }
#else
    __trap();
#endif
}
