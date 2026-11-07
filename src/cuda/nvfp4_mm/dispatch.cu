// nvfp4_mm's extern "C" entry points, the grouped plan and the read probe (see common.cuh); the
// GEMM and mv kernels are compiled in the other parts.
#include "common.cuh"

namespace legion_nvfp4 {

std::atomic<int> pdl_state[PDL_DEVICES];  // zero-initialized: 0 unknown

}  // namespace legion_nvfp4

namespace {

// The grouped plan: one CTA counts the pairs of each expert, scans the counts into offsets and
// scatters the pairs (in no particular order within an expert: results do not depend on it).
// rows (or null): pair (t, slot) goes to the expert of (rows[t], slot), a verify step's padding
// tokens to the experts of real ones (pad_rows.rs). skip_pad (rows set): a padding token's
// pairs (rows[t] != t) go to no expert; so do the pairs whose id is -1 (--tp-moe-ep's other
// ranks' experts, #1176). Such pairs are listed after every expert's pairs instead
// (plan[n_expert] counts the experts' pairs only), and the grouped kernels zero their results.
// After the pairs: the token tiles of 8, 16, 32, 64 and 128 pairs over all experts (PLAN_TILES,
// the totals find_tile counts with no pair-count selection), so the grouped mv's and GEMM's CTAs
// past the last tile exit before the plan scan (#836).
__global__ void __launch_bounds__(1024) nvfp4_plan(const int32_t * ids, int64_t ids_ld, const int32_t * rows,
                                                   int32_t skip_pad, int32_t n_used, int64_t n_tokens,
                                                   int32_t n_expert, int32_t * plan) {
    __shared__ int32_t count[1024], cursor[1024];
    __shared__ int32_t routed, pad_cursor, tiles[legion_nvfp4::PLAN_TILES];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        count[e] = 0;
    }
    if (threadIdx.x == 0) {
        pad_cursor = 0;
    }
    if (threadIdx.x < legion_nvfp4::PLAN_TILES) {
        tiles[threadIdx.x] = 0;
    }
    __syncthreads();
    const int64_t pairs = n_tokens * n_used;
    auto expert = [&](int64_t q) {
        const int64_t t = q / n_used;
        return ids[(rows ? (int64_t) rows[t] : t) * ids_ld + q % n_used];
    };
    // A pair goes to no expert: a skipped padding token's, or one whose id is -1 (an expert
    // another rank holds under --tp-moe-ep: such ids come with the node's skip flag only).
    auto padding = [&](int64_t q) {
        const int64_t t = q / n_used;
        return (skip_pad && rows && (int64_t) rows[t] != t) || expert(q) < 0;
    };
    for (int64_t q = threadIdx.x; q < pairs; q += blockDim.x) {
        if (!padding(q)) {
            atomicAdd(&count[expert(q)], 1);
        }
    }
    __syncthreads();
    // The offsets: a block-wide exclusive scan of the counts, thread e holding expert e's (1024
    // threads, n_expert <= 1024; one thread walking 512 experts took most of the launch).
    {
        __shared__ int32_t warp_sum[32];
        const int e = threadIdx.x, lane = e % 32, warp = e / 32;
        const int c = e < n_expert ? count[e] : 0;
        int incl = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, incl, o);
            incl += lane >= o ? v : 0;
        }
        if (lane == 31) {
            warp_sum[warp] = incl;
        }
        __syncthreads();
        if (warp == 0) {
            int s = lane < (int) (blockDim.x / 32) ? warp_sum[lane] : 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, s, o);
                s += lane >= o ? v : 0;
            }
            warp_sum[lane] = s;
        }
        __syncthreads();
        const int excl = incl - c + (warp > 0 ? warp_sum[warp - 1] : 0);
        if (e < n_expert) {
            plan[e] = excl;
            cursor[e] = excl;
        }
        if (e == max(n_expert - 1, 0)) {
            plan[n_expert] = excl + c;
            routed         = excl + c;
        }
        // The token tiles: ceil(c / bn) per expert, summed over the warp, then the block.
#pragma unroll
        for (int j = 0; j < legion_nvfp4::PLAN_TILES; j++) {
            const int bn = 8 << j;
            int t = (c + bn - 1) / bn;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                t += __shfl_down_sync(0xffffffffu, t, o);
            }
            if (lane == 0 && t > 0) {
                atomicAdd(&tiles[j], t);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < legion_nvfp4::PLAN_TILES) {
        plan[n_expert + 1 + pairs + threadIdx.x] = tiles[threadIdx.x];
    }
    for (int64_t q = threadIdx.x; q < pairs; q += blockDim.x) {
        if (padding(q)) {
            plan[n_expert + 1 + routed + atomicAdd(&pad_cursor, 1)] = (int32_t) q;
        } else {
            const int e = expert(q);
            plan[n_expert + 1 + atomicAdd(&cursor[e], 1)] = (int32_t) q;
        }
    }
}

__global__ void __launch_bounds__(256) nvfp4_read_probe(const uint8_t * w0, const uint8_t * w1, const uint8_t * w2,
                                                       int64_t w_e, const int32_t * ranges, int64_t chunk,
                                                       uint32_t * sink) {
    const int t = ranges[2 * blockIdx.x], e = ranges[2 * blockIdx.x + 1];
    const uint8_t * base = (t == 0 ? w0 : t == 1 ? w1 : w2) + (int64_t) e * w_e;
    const int64_t lo = (int64_t) blockIdx.y * chunk, hi = min(lo + chunk, w_e);
    constexpr int U = 4, STEP = 256 * 16;
    uint32_t acc = 0;
    for (int64_t o = lo + threadIdx.x * 16; o < hi; o += U * STEP) {
        uint4 v[U];
#pragma unroll
        for (int j = 0; j < U; j++) {
            const int64_t q = o + j * STEP;
            v[j] = q < hi ? __ldcs((const uint4 *) (base + q)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < U; j++) {
            acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
        }
    }
    if (acc == 0x9e3779b9u) {
        sink[0] = acc;  // never in practice; keeps the loads
    }
}

}  // namespace

int legion_nvfp4_gemm_launch(const legion_nvfp4_mm * p, int mi, int nt, int variant, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    return mi == 1   ? legion_nvfp4::launch_gemm<1>(*p, nt, variant, st)
           : mi == 2 ? legion_nvfp4::launch_gemm<2>(*p, nt, variant, st)
                     : -1;
}

int legion_nvfp4_mv_launch(const legion_nvfp4_mm * p, int nt, int rtw, int kw, int rpt, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    switch (nt) {
        case 1: return legion_nvfp4::launch_mv<1>(*p, rtw, kw, rpt, st);
        case 2: return legion_nvfp4::launch_mv<2>(*p, rtw, kw, rpt, st);
        case 4: return legion_nvfp4::launch_mv<4>(*p, rtw, kw, rpt, st);
        case 8: return legion_nvfp4::launch_mv<8>(*p, rtw, kw, rpt, st);
        default: return -1;
    }
}

int legion_nvfp4_mvg_launch(const legion_nvfp4_mm * p, int nt, int rtw, int rpt, int sb, int pft, void * stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    if (!p->plan) {
        return -1;
    }
    const bool glu = p->w2 != nullptr;
    switch (nt) {
        case 1: return legion_nvfp4::launch_mvg<1>(*p, glu, rtw, rpt, sb, pft, st);
        case 2: return legion_nvfp4::launch_mvg<2>(*p, glu, rtw, rpt, sb, pft, st);
        case 4: return legion_nvfp4::launch_mvg<4>(*p, glu, rtw, rpt, sb, pft, st);
        default: return -1;
    }
}

int legion_nvfp4_swiglu_gemm_launch(const legion_nvfp4_mm * p, int nt, int variant, void * stream) {
    return legion_nvfp4::launch_swiglu_gemm(*p, nt, variant, (cudaStream_t) stream);
}

int legion_nvfp4_swiglu_mv_launch(const legion_nvfp4_mm * p, int nt, void * stream) {
    return legion_nvfp4::launch_swiglu_mv(*p, nt, (cudaStream_t) stream);
}

int legion_nvfp4_plan_launch(const int32_t * ids, int64_t ids_ld, const int32_t * rows, int32_t skip_pad,
                             int32_t n_used, int64_t n_tokens, int32_t n_expert, int32_t * plan, void * stream) {
    if (n_expert > 1024) {
        return -1;
    }
    nvfp4_plan<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, ids_ld, rows, skip_pad, n_used, n_tokens, n_expert,
                                                      plan);
    return status();
}

int legion_nvfp4_read_probe_launch(const uint8_t * w0, const uint8_t * w1, const uint8_t * w2, int64_t w_e,
                                   const int32_t * ranges, int32_t n, int64_t chunk, uint32_t * sink, void * stream) {
    if (n <= 0 || chunk <= 0 || w_e % 16 != 0) {
        return -1;
    }
    const dim3 grid((unsigned) n, (unsigned) ((w_e + chunk - 1) / chunk));
    nvfp4_read_probe<<<grid, 256, 0, (cudaStream_t) stream>>>(w0, w1, w2, w_e, ranges, chunk, sink);
    return status();
}

int legion_cuda_sm_count(void) {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n;
}
