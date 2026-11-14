// expert_mv's extern "C" entry points and stage_copy (see common.cuh); the kernels of each weight
// type are compiled in the type's own .cu.
#include "common.cuh"

namespace {

constexpr int STAGE_THREADS = 256;
constexpr int STAGE_UNROLL  = 8;    // 16-byte loads in flight per thread
constexpr int STAGE_PAIRS   = 512;  // most pairs a step has here (64 tokens x 8, 8 x 64)
constexpr int PF_SLOTS      = 8;    // prefetch slots of a half at most (the peer prefetch)
constexpr int PF_CTAS       = 64;   // CTAs of a prefetch copy at most

// Copies a step's host experts into the staging buffer on few CTAs (`ctas`): each thread keeps
// STAGE_UNROLL loads in flight, enough for a few CTAs to fill the link while the VRAM experts
// keep the rest of the SMs (as Strata-AC922's fetch_blobs does, 756ff34, MIT). Every CTA works
// out the slots alike: a host expert's first pair takes the next slot while slots last.
__global__ void __launch_bounds__(STAGE_THREADS) stage_copy(const legion_expert_stage s) {
    __shared__ int slot[STAGE_PAIRS];
    const int n_pairs = s.n_used * s.n_tok;
    auto id = [&](int q) {
        return *(const int32_t *) ((const char *) s.ids + (q / s.n_used) * s.ids_nb1 + (q % s.n_used) * 4);
    };
    for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
        const int e = id(q);
        bool first  = e >= s.vram_below && e < s.n_w;
        for (int r = 0; r < q && first; r++) {
            first = id(r) != e;
        }
        slot[q] = first;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int rank = 0;
        for (int q = 0; q < n_pairs; q++) {
            int sl = -1;
            if (slot[q]) {
                const int r = rank++;
                if (r % s.t_every < s.t_take) {
                    sl = r / s.t_every * s.t_take + r % s.t_every;
                }
            }
            slot[q] = sl < s.cap ? sl : -1;
        }
    }
    __syncthreads();
    if (s.table && blockIdx.x == 0) {
        for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
            ((int32_t *) s.stage)[q] = slot[q];
        }
    }
    const int64_t step = (int64_t) gridDim.x * STAGE_THREADS * STAGE_UNROLL;
    for (int q = 0; q < n_pairs; q++) {
        if (slot[q] < 0 || (slot[q] % s.c_every < s.c_take) == (s.c_flip != 0)) {
            continue;
        }
        const int64_t e = id(q);
        for (int m = 0; m < s.n_seg; m++) {
            const uint4 * src = (const uint4 *) (s.src[m] + e * s.nb2[m]);
            uint4 * dst       = (uint4 *) (s.stage + s.hdr + slot[q] * s.slot + s.off[m]);
            const int64_t n   = s.nb2[m] / 16;
            for (int64_t base = (int64_t) blockIdx.x * STAGE_THREADS * STAGE_UNROLL + threadIdx.x; base < n;
                 base += step) {
                uint4 v[STAGE_UNROLL];
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < n) {
                        v[i] = __ldg(src + c);
                    }
                }
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < n) {
                        dst[c] = v[i];
                    }
                }
            }
        }
    }
}

// One CTA (see legion_expert_prepare): which of the step's host experts the peer prefetched, the
// slot table for the layer's launches, the ids left for the peer's copy inside the layer and the
// step's ids for the next step's guess. The ranks and slots of the experts left are stage_copy's
// over `rest` (same order, same rule), so the table says what the peer's stage_copy fills.
__global__ void __launch_bounds__(STAGE_THREADS) stage_prepare(const legion_expert_prepare s) {
    __shared__ int slot[STAGE_PAIRS];
    __shared__ int held[PF_SLOTS];  // the tiered id a done prefetch slot holds, else -1
    const int n_pairs = s.n_used * s.n_tok;
    auto id = [&](int q) {
        return *(const int32_t *) ((const char *) s.ids + (q / s.n_used) * s.ids_nb1 + (q % s.n_used) * 4);
    };
    if (threadIdx.x == 0) {
        // The peer's copies for this layer (and any older one) stop here.
        *(volatile uint64_t *) s.progress = s.now;
        __threadfence_system();
    }
    if (threadIdx.x < PF_SLOTS) {
        const int j = threadIdx.x;
        bool done   = s.stamps != nullptr && j < s.pf_slots;
        for (int c = 0; c < s.pf_ctas && done; c++) {
            done = ((const volatile uint64_t *) s.stamps)[j * s.pf_ctas + c] == s.seq;
        }
        // The id is read only once every stamp is (the copy wrote it before its fences).
        __threadfence();
        held[j] = done ? ((const volatile int32_t *) s.pf_ids)[j] : -1;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
        const int e     = id(q);
        const bool host = e >= s.vram_below && e < s.n_w;
        int pf          = -1;
        for (int j = 0; j < PF_SLOTS && host && pf < 0; j++) {
            if (held[j] == e) {
                pf = j;
            }
        }
        bool first = host;
        for (int r = 0; r < q && first; r++) {
            first = id(r) != e;
        }
        // -1: no slot; -2: an expert to rank; else its prefetch slot.
        slot[q]   = !first ? -1 : pf < 0 ? -2 : s.pf_base + pf;
        s.rest[q] = pf < 0 ? e : -1;
    }
    if (s.x_keep != nullptr) {
        for (int i = threadIdx.x; i < s.n_tok * s.k; i += blockDim.x) {
            s.x_keep[i] = *(const float *) (s.x + (i / s.k) * s.x_nb2 + (int64_t) (i % s.k) * 4);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int rank = 0, host = 0, hit = 0, filled = 0;
        for (int q = 0; q < n_pairs; q++) {
            if (slot[q] == -1) {
                continue;
            }
            host++;
            if (slot[q] >= 0) {
                hit++;
                continue;
            }
            int sl      = -1;
            const int r = rank++;
            if (r % s.t_every < s.t_take) {
                sl = r / s.t_every * s.t_take + r % s.t_every;
            }
            slot[q] = sl < s.cap ? sl : -1;
        }
        for (int j = 0; j < PF_SLOTS; j++) {
            filled += held[j] >= 0;
        }
        s.count[0] += host;
        s.count[1] += hit;
        s.count[2] += filled;
        s.count[3] += 1;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < n_pairs; q += blockDim.x) {
        s.table[q] = slot[q];
    }
}

constexpr int GUESS_ROWS    = 8;     // router rows per CTA, a warp each
constexpr int GUESS_X       = 8192;  // floats of x a CTA holds at once
constexpr int GUESS_TOKENS  = 8;     // tokens per pass at most
constexpr int GUESS_EXPERTS = 1024;

// The guess's logits (see legion_expert_guess): a warp per router row, x in shared memory a few
// tokens at a time, so the router's rows cross the link once per pass.
__global__ void __launch_bounds__(GUESS_ROWS * 32) guess_logits(const legion_expert_guess s) {
    __shared__ float xs[GUESS_X];
    const int lane  = threadIdx.x & 31;
    const int row   = blockIdx.x * GUESS_ROWS + (threadIdx.x >> 5);
    const int chunk = min(GUESS_X / s.k, GUESS_TOKENS);
    for (int t0 = 0; t0 < s.n_tok; t0 += chunk) {
        const int nt = min(chunk, s.n_tok - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * s.k; i += blockDim.x) {
            xs[i] = __ldg(s.x + (int64_t) t0 * s.k + i);
        }
        __syncthreads();
        if (row >= s.n_expert) {
            continue;
        }
        float acc[GUESS_TOKENS] = {};
        const int64_t wr        = (int64_t) row * s.k;
#pragma unroll 8
        for (int i = lane; i < s.k; i += 32) {
            const float wv = s.w_f16 ? __half2float(__ldg((const __half *) s.w + wr + i)) :
                                       __ldg((const float *) s.w + wr + i);
            for (int t = 0; t < nt; t++) {
                acc[t] += wv * xs[t * s.k + i];
            }
        }
        for (int t = 0; t < nt; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o);
            }
            if (lane == 0) {
                s.logits[(int64_t) (t0 + t) * s.n_expert + row] = v;
            }
        }
    }
}

// The guess's ids: per token the n_used largest logits (ties: the lower expert), as tiered ids,
// rank-major (a copy of the first few takes every token's likeliest experts).
__global__ void __launch_bounds__(STAGE_THREADS) guess_topk(const legion_expert_guess s) {
    __shared__ float l[GUESS_EXPERTS];
    for (int t = 0; t < s.n_tok; t++) {
        __syncthreads();
        for (int e = threadIdx.x; e < s.n_expert; e += blockDim.x) {
            l[e] = s.logits[(int64_t) t * s.n_expert + e];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < s.n_expert; e += blockDim.x) {
            const float v = l[e];
            int rank      = 0;
            for (int j = 0; j < s.n_expert && rank < s.n_used; j++) {
                rank += l[j] > v || (l[j] == v && j < e);
            }
            if (rank < s.n_used) {
                // Rank-major: every token's first choice, then their second ones...
                s.guess[rank * s.n_tok + t] = s.table[e];
            }
        }
    }
}

// The peer's copy of a guess into a prefetch half (see legion_expert_prefetch). Every CTA works
// out the slots alike (the first `cap` distinct host ids), copies its share of each slot a chunk
// at a time and stamps the slot once its share is in. Thread 0 reads the running GPU's progress
// once per chunk, while the chunk's loads are in flight; once it reaches `seq` the copy stops
// (stamps already written stay valid: their bytes are in). Before each slot thread 0 checks the
// layer's learned times: a slot that would end after the next FORK is not started. CTA 0 then
// watches for that FORK (at most PF_WATCH_NS) and learns the times for the next copy.
__device__ __forceinline__ uint64_t pf_clock() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

constexpr uint64_t PF_WATCH_NS  = 1000000;  // CTA 0 waits for the next FORK at most this long
constexpr uint64_t PF_MARGIN_NS = 4000;     // a slot must end this long before the FORK

__global__ void __launch_bounds__(STAGE_THREADS) prefetch_copy(const legion_expert_prefetch s) {
    __shared__ int ids[PF_SLOTS];
    __shared__ int n;
    __shared__ int stop;  // 1: the next layer has started; 2: no time for the next slot
    __shared__ int skipped;
    __shared__ uint64_t t0, seen, window, per_slot, slot_ns;
    if (threadIdx.x == 0) {
        t0       = pf_clock();
        seen     = 0;
        slot_ns  = 0;
        skipped  = 0;
        window   = ((const volatile uint64_t *) s.timing)[0];
        per_slot = ((const volatile uint64_t *) s.timing)[1];
        int m    = 0;
        for (int r = 0; r < s.n_guess && m < s.cap; r++) {
            const int e = s.guess[r];
            bool fresh  = e >= s.vram_below && e < s.n_w;
            for (int j = 0; j < m && fresh; j++) {
                fresh = ids[j] != e;
            }
            if (fresh) {
                ids[m++] = e;
            }
        }
        n    = m;
        stop = *(const volatile uint64_t *) s.progress >= s.seq;
        if (stop) {
            seen = t0;
        }
        if (blockIdx.x == 0) {
            for (int j = 0; j < m; j++) {
                ((volatile int32_t *) s.pf_ids)[j] = ids[j];
            }
            s.count[0] += m;
        }
    }
    __syncthreads();
    const int64_t step = (int64_t) gridDim.x * STAGE_THREADS * STAGE_UNROLL;
    for (int j = 0; j < n; j++) {
        uint64_t ts = 0;
        if (threadIdx.x == 0 && !stop) {
            ts = pf_clock();
            if (window > 0 && per_slot > 0 && ts - t0 + per_slot + PF_MARGIN_NS > window) {
                stop    = 2;
                skipped = n - j;
            }
        }
        // Every thread reads `stop` only after this barrier (thread 0 wrote it before it).
        __syncthreads();
        if (stop) {
            break;
        }
        const int64_t e = ids[j];
        for (int m = 0; m < s.n_seg && !stop; m++) {
            const uint4 * src = (const uint4 *) (s.src[m] + e * s.nb2[m]);
            uint4 * dst       = (uint4 *) (s.slots + j * s.slot + s.off[m]);
            const int64_t nv  = s.nb2[m] / 16;
            for (int64_t c0 = (int64_t) blockIdx.x * STAGE_THREADS * STAGE_UNROLL; c0 < nv; c0 += step) {
                const int64_t base = c0 + threadIdx.x;
                uint4 v[STAGE_UNROLL];
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < nv) {
                        v[i] = __ldg(src + c);
                    }
                }
                bool late = false;
                if (threadIdx.x == 0) {
                    late = *(const volatile uint64_t *) s.progress >= s.seq;
                    if (late) {
                        seen = pf_clock();
                    }
                }
#pragma unroll
                for (int i = 0; i < STAGE_UNROLL; i++) {
                    const int64_t c = base + i * STAGE_THREADS;
                    if (c < nv) {
                        dst[c] = v[i];
                    }
                }
                // `stop` is written only between these two barriers and read only outside them.
                __syncthreads();
                if (late) {
                    stop = 1;
                }
                __syncthreads();
                if (stop) {
                    break;
                }
            }
        }
        // This CTA's share of slot j is in: a system fence by every thread, then the stamp.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0 && !stop) {
            ((volatile uint64_t *) s.stamps)[j * gridDim.x + blockIdx.x] = s.seq;
            slot_ns = pf_clock() - ts;
        }
    }
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    // CTA 0 learns the layer's times: when the next FORK came (watched for if the copy ended
    // first) and how long its last whole slot took.
    uint64_t at = seen;
    while (at == 0) {
        const uint64_t now = pf_clock();
        if (*(const volatile uint64_t *) s.progress >= s.seq) {
            at = now;
        } else if (now - t0 > PF_WATCH_NS) {
            break;
        } else {
#if __CUDA_ARCH__ >= 700
            __nanosleep(1000);
#endif
        }
    }
    volatile uint64_t * timing = s.timing;
    if (at != 0) {
        // Quick to shrink, slow to grow: a slot started too late costs the next layer.
        const uint64_t w = at - t0, old = timing[0];
        timing[0]        = old == 0 ? w : w < old ? (old + w) / 2 : old + (w - old) / 8;
    }
    if (slot_ns != 0) {
        const uint64_t old = timing[1];
        timing[1]          = old == 0 ? slot_ns : (3 * old + slot_ns) / 4;
    }
    s.count[1] += stop == 1;
    s.count[2] += skipped;
}

// quantize_x (`steps & 1`), then the type's kernels (`steps & 2`, launch_type in the type's .cu).
template <ggml_type type>
int launch_steps(const legion_expert_mv & p, int smem, cudaStream_t stream) {
    if (p.steps & 1) {
        const dim3 grid((p.k + 127) / 128, p.n_x * p.n_tok);
        if (xq_split_for<type>(p)) {
            quantize_x<true><<<grid, 128, 0, stream>>>(p);
        } else {
            quantize_x<false><<<grid, 128, 0, stream>>>(p);
        }
    }
    if (!(p.steps & 2)) {
        return cudaGetLastError();
    }
    return legion_emv::launch_type<type>(p, smem, stream);
}

}  // namespace

extern "C" int legion_expert_mv_launch(int type, const legion_expert_mv * p, int smem, void * stream) {
    auto s = (cudaStream_t) stream;
    switch (type) {
        case GGML_TYPE_Q4_K: return launch_steps<GGML_TYPE_Q4_K>(*p, smem, s);
        case GGML_TYPE_Q5_K: return launch_steps<GGML_TYPE_Q5_K>(*p, smem, s);
        case GGML_TYPE_Q5_1: return launch_steps<GGML_TYPE_Q5_1>(*p, smem, s);
        case GGML_TYPE_Q6_K: return launch_steps<GGML_TYPE_Q6_K>(*p, smem, s);
        case GGML_TYPE_Q8_0: return launch_steps<GGML_TYPE_Q8_0>(*p, smem, s);
        case GGML_TYPE_Q2_K: return launch_steps<GGML_TYPE_Q2_K>(*p, smem, s);
        case GGML_TYPE_Q3_K: return launch_steps<GGML_TYPE_Q3_K>(*p, smem, s);
        case GGML_TYPE_IQ2_XXS: return launch_steps<GGML_TYPE_IQ2_XXS>(*p, smem, s);
        case GGML_TYPE_IQ2_S: return launch_steps<GGML_TYPE_IQ2_S>(*p, smem, s);
        case GGML_TYPE_IQ3_XXS: return launch_steps<GGML_TYPE_IQ3_XXS>(*p, smem, s);
        case GGML_TYPE_IQ3_S: return launch_steps<GGML_TYPE_IQ3_S>(*p, smem, s);
        case GGML_TYPE_IQ4_XS: return launch_steps<GGML_TYPE_IQ4_XS>(*p, smem, s);
        default: return cudaErrorInvalidValue;
    }
}

extern "C" int legion_expert_stage_launch(const legion_expert_stage * s, void * stream) {
    if (s->n_used * s->n_tok > STAGE_PAIRS || s->n_seg < 0 || s->n_seg > 3 || s->ctas < 1 || s->t_every < 1 ||
        s->t_take < 1 || s->t_take > s->t_every || s->c_every < 1) {
        return cudaErrorInvalidValue;
    }
    stage_copy<<<s->ctas, STAGE_THREADS, 0, (cudaStream_t) stream>>>(*s);
    return cudaGetLastError();
}

extern "C" int legion_expert_prepare_launch(const legion_expert_prepare * s, void * stream) {
    if (s->n_used * s->n_tok > STAGE_PAIRS || s->pf_slots < 0 || s->pf_slots > PF_SLOTS || s->pf_ctas < 1 ||
        s->pf_ctas > PF_CTAS || s->progress == nullptr || s->t_every < 1 || s->t_take < 1 || s->t_take > s->t_every) {
        return cudaErrorInvalidValue;
    }
    stage_prepare<<<1, STAGE_THREADS, 0, (cudaStream_t) stream>>>(*s);
    return cudaGetLastError();
}

extern "C" int legion_expert_guess_launch(const legion_expert_guess * s, void * stream) {
    if (s->k < 1 || s->k > GUESS_X || s->n_expert < 1 || s->n_expert > GUESS_EXPERTS || s->n_tok < 1 ||
        s->n_used < 1 || s->n_used > s->n_expert || s->n_used * s->n_tok > STAGE_PAIRS) {
        return cudaErrorInvalidValue;
    }
    const int ctas = (s->n_expert + GUESS_ROWS - 1) / GUESS_ROWS;
    guess_logits<<<ctas, GUESS_ROWS * 32, 0, (cudaStream_t) stream>>>(*s);
    guess_topk<<<1, STAGE_THREADS, 0, (cudaStream_t) stream>>>(*s);
    return cudaGetLastError();
}

extern "C" int legion_expert_prefetch_launch(const legion_expert_prefetch * s, void * stream) {
    if (s->n_guess < 0 || s->n_guess > STAGE_PAIRS || s->cap < 0 || s->cap > PF_SLOTS || s->n_seg < 0 ||
        s->n_seg > 3 || s->ctas < 1 || s->ctas > PF_CTAS || s->slot % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    prefetch_copy<<<s->ctas, STAGE_THREADS, 0, (cudaStream_t) stream>>>(*s);
    return cudaGetLastError();
}
