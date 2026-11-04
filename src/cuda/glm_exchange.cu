// GLM-5.3-Flash's parallel decode (#558): the exchange between the GPUs of one step. Every GPU
// runs the same graph; where a tensor is split over them (rows of a split matmul, the routed
// experts each GPU owns), this kernel hands each GPU's part to the others. The Rust side
// (src/model/glm5next/par/exchange.rs) launches it and owns the memory; the protocol, the
// layouts and a CPU twin of it are documented there.
//
// Each GPU has an inbox in pinned host memory on its own NUMA node: per parity (the exchange
// count's low bit), writer and chunk a flag, per parity and writer a payload slot. A GPU whose
// peers write to it over NVLink (#671: the other GPU of its NVLink pair) also has an inbox in
// its own VRAM with a slot per such writer, used for every payload that fits. A payload goes in
// chunks: the writer copies chunk c into the reader's slot, fences, and sets the chunk's flag to
// the exchange's value `v`; the reader waits for chunk c's flag and copies (or adds) it. The
// blocks of a launch take the chunks in turn, all their sends first: no block waits before its
// own sends are out, and the grid is small enough to be resident at once (no deadlock between
// the GPUs' spinning blocks). A GPU's k-th exchange can only overwrite a slot of the same parity
// (k + 2) after every reader finished exchange k: it needs the readers' payloads of exchange
// k + 1 first, which they write only once their kernel k is done (a stream runs its kernels in
// order).
//
// Gather, bcast and select do no arithmetic: the values are copied as they are, so the result is
// bitwise what one GPU would have computed. A reduce (the tensor-parallel layers' partial sums)
// adds every GPU's part in GPU order, the same bits on every GPU, whichever chunk, block or link
// the parts came by. A reduce of a chunk or more per GPU (`scatter`, #686) sends each chunk to
// one GPU, its owner, which adds it up the same way and sends the sum to the others: half the
// bytes on four GPUs, the same bits.

#include <cuda_runtime.h>

#include <cstdint>

extern "C" {

// Must match src/model/glm5next/par/exchange.rs (`Kernel`).
struct legion_exchange {
    char * inbox[8];      // per GPU: its host inbox
    char * peer_in[8];    // per GPU: its VRAM inbox (null: none)
    const char * src[4];  // gather: this GPU's rows of each segment, [rows, n] F32 contiguous;
                          // select: the experts' outputs [e, n_used, n], the shared expert's [e, n]
    const int32_t * ids;  // select: the router's choices, [n_used, n] I32 contiguous
    char * dst;
    int64_t slot_bytes;
    int64_t peer_slot;  // bytes of a VRAM inbox slot
    int64_t chunk;      // bytes of a chunk (a multiple of 16)
    uint64_t v;
    int64_t total[4];  // gather / bcast: rows of each segment
    int64_t n;         // tokens
    int64_t e;         // select: values per column; reduce: values of the part
    int64_t timeout_ns;
    int32_t bounds[9];       // select: GPU d owns experts bounds[d]..bounds[d + 1]
    uint32_t peer_mask[8];   // per GPU: bit w set when GPU w writes into its VRAM inbox
    int32_t n_dev, me, mode, n_seg, root, sh, n_used, sub;  // sub: a reduce's sum blocks per chunk
    int32_t scatter;  // a reduce as a reduce-scatter and an all-gather (#686)
};
}

namespace {

constexpr int MODE_GATHER = 0;
constexpr int MODE_BCAST  = 1;
constexpr int MODE_SELECT = 2;
constexpr int MODE_REDUCE = 3;

constexpr int MAX_DEV       = 8;
constexpr int MAX_CHUNKS    = 64;
constexpr int64_t FLAG_SEP  = 128;                                    // bytes between flags
constexpr int64_t FLAGS     = 2 * MAX_DEV * MAX_CHUNKS * FLAG_SEP;    // per parity, writer, chunk
constexpr int64_t WAIT_AT   = FLAGS;                                  // per writer: ns waited (u64)
constexpr int64_t ERR_AT    = FLAGS + 512;                            // u32: a wait timed out
constexpr int64_t HDR       = FLAGS + 4096;                           // the host inbox's header
constexpr int64_t PHDR      = FLAGS;                                  // the VRAM inbox's header
constexpr int THREADS       = 512;
constexpr int UNROLL        = 4;

__device__ __forceinline__ volatile uint64_t * flag(char * base, int parity, int writer, int c) {
    return (volatile uint64_t *) (base + ((int64_t) (parity * MAX_DEV + writer) * MAX_CHUNKS + c) * FLAG_SEP);
}

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ int64_t padded(int64_t n) {
    return (n + 3) / 4 * 4;
}

// Where writer w's payload of `bytes` for reader r goes: r's VRAM inbox when w writes there
// (peer_mask) and it fits, else r's host inbox. `flags`: the base of the chunks' flags.
struct Route {
    char * flags;
    char * slot;
};

__device__ __forceinline__ Route route(const legion_exchange & p, int w, int r, int parity, int64_t bytes) {
    const uint32_t m = p.peer_mask[r];
    if (p.mode != MODE_SELECT && ((m >> w) & 1u) && p.peer_in[r] != nullptr && bytes <= p.peer_slot) {
        const int idx = __popc(m & ((1u << w) - 1u));
        const int np  = __popc(m);
        return { p.peer_in[r], p.peer_in[r] + PHDR + (int64_t) (parity * np + idx) * p.peer_slot };
    }
    return { p.inbox[r], p.inbox[r] + HDR + (int64_t) (parity * p.n_dev + w) * p.slot_bytes };
}

// Copies `n` floats with the block's threads; `remote`: `src` is an inbox, read past the caches
// (another GPU wrote it).
__device__ void copy_f32(float * dst, const float * src, int64_t n, bool remote) {
    const bool vec = ((((uintptr_t) dst) | ((uintptr_t) src)) % 16) == 0;
    int64_t done = 0;
    if (vec) {
        const int64_t n4 = n / 4;
        const uint4 * s = (const uint4 *) src;
        uint4 * d       = (uint4 *) dst;
        for (int64_t base = threadIdx.x; base < n4; base += (int64_t) THREADS * UNROLL) {
            uint4 r[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                const int64_t i = base + (int64_t) u * THREADS;
                if (i < n4) {
                    r[u] = remote ? __ldcv(s + i) : s[i];
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                const int64_t i = base + (int64_t) u * THREADS;
                if (i < n4) {
                    d[i] = r[u];
                }
            }
        }
        done = n4 * 4;
    }
    for (int64_t i = done + threadIdx.x; i < n; i += THREADS) {
        dst[i] = remote ? __ldcv(src + i) : src[i];
    }
}

// Rows of a gather / bcast segment of `total` rows that GPU `d` holds: [start, start + len).
__device__ __forceinline__ void rows_of(const legion_exchange & p, int64_t total, int d, int64_t & start,
                                        int64_t & len) {
    if (p.mode == MODE_BCAST) {
        start = 0;
        len   = d == p.root ? total : 0;
        return;
    }
    const int64_t q = total / p.n_dev, r = total % p.n_dev;
    start           = d * q + (d < r ? d : r);
    len             = q + (d < r ? 1 : 0);
}

// The floats of GPU w's payload (gather / bcast / reduce): per segment its rows of every token,
// each segment padded to 4 floats; a reduce's part.
__device__ int64_t payload_floats(const legion_exchange & p, int w) {
    if (p.mode == MODE_REDUCE) {
        return p.e;
    }
    int64_t at = 0;
    for (int s = 0; s < p.n_seg; s++) {
        int64_t start, len;
        rows_of(p, p.total[s], w, start, len);
        at += padded(len * p.n);
    }
    return at;
}

__device__ __forceinline__ int n_chunks(int64_t floats, int64_t cf) {
    return floats <= 0 ? 1 : (int) ((floats + cf - 1) / cf);
}

// This GPU's payload's floats lo..hi into `out` (a reader's slot, at the same offsets).
__device__ void pack_range(const legion_exchange & p, float * out, int64_t lo, int64_t hi) {
    if (p.mode == MODE_REDUCE) {
        copy_f32(out + lo, (const float *) p.src[0] + lo, hi - lo, false);
        return;
    }
    int64_t at = 0;
    for (int s = 0; s < p.n_seg; s++) {
        int64_t start, len;
        rows_of(p, p.total[s], p.me, start, len);
        const int64_t L = len * p.n;
        const int64_t a = lo > at ? lo : at;
        const int64_t b = hi < at + L ? hi : at + L;
        if (b > a) {
            copy_f32(out + a, (const float *) p.src[s] + (a - at), b - a, false);
        }
        at += padded(L);
    }
}

// GPU w's payload's floats lo..hi into the result: `from` its slot in this GPU's inbox, or null
// for this GPU's own sources. Segment by segment, token row by token row.
__device__ void unpack_range(const legion_exchange & p, int w, const float * from, int64_t lo, int64_t hi) {
    float * dst = (float *) p.dst;
    int64_t at  = 0;  // floats into the payload
    int64_t out = 0;  // floats into dst: the segments one after another
    for (int s = 0; s < p.n_seg; s++) {
        int64_t start, len;
        rows_of(p, p.total[s], w, start, len);
        const int64_t L = len * p.n;
        const int64_t a = (lo > at ? lo : at) - at;
        const int64_t b = (hi < at + L ? hi : at + L) - at;
        if (b > a) {
            const float * base = from ? from + at : (const float *) p.src[s];
            for (int64_t t = a / len; t * len < b; t++) {
                const int64_t k0 = a > t * len ? a : t * len;
                const int64_t k1 = b < (t + 1) * len ? b : (t + 1) * len;
                copy_f32(dst + out + t * p.total[s] + start + (k0 - t * len), base + k0, k1 - k0, from != nullptr);
            }
        }
        at += padded(L);
        out += p.total[s] * p.n;
    }
}

// Thread 0: waits for one flag of writer w (`fl`), adding the wait to this GPU's count for w
// when `count`. Once one wait of the step timed out (the error word), the rest give up at once:
// the step fails anyway, and a whole graph of timeouts would take minutes.
__device__ void await_one(const legion_exchange & p, volatile uint64_t * fl, int w, bool count) {
    char * mine             = p.inbox[p.me];
    volatile uint32_t * err = (volatile uint32_t *) (mine + ERR_AT);
    const uint64_t t0       = now_ns();
    uint64_t t              = t0;
    while (*fl != p.v) {
        t = now_ns();
        if (*err != 0 || (int64_t) (t - t0) > p.timeout_ns) {
            *err = 1;
            break;
        }
    }
    if (count) {
        volatile uint64_t * wt = (volatile uint64_t *) (mine + WAIT_AT) + w;
        *wt += t - t0;
    }
    __threadfence_system();
}

// Thread 0: waits for chunk c's flag of every other writer of a reduce, counting each writer's
// wait (until its flag was seen) when `count`.
__device__ void await_all(const legion_exchange & p, int parity, int c, bool count) {
    char * mine             = p.inbox[p.me];
    volatile uint32_t * err = (volatile uint32_t *) (mine + ERR_AT);
    volatile uint64_t * fl[MAX_DEV];
    uint32_t pending = 0;
    for (int w = 0; w < p.n_dev; w++) {
        if (w != p.me) {
            fl[w] = flag(route(p, w, p.me, parity, p.e * 4).flags, parity, w, c);
            pending |= 1u << w;
        }
    }
    const uint64_t t0 = now_ns();
    uint64_t t        = t0;
    for (;;) {
        for (int w = 0; w < p.n_dev; w++) {
            if (((pending >> w) & 1u) && *fl[w] == p.v) {
                pending &= ~(1u << w);
                if (count) {
                    *((volatile uint64_t *) (mine + WAIT_AT) + w) += t - t0;
                }
            }
        }
        if (pending == 0) {
            break;
        }
        t = now_ns();
        if (*err != 0 || (int64_t) (t - t0) > p.timeout_ns) {
            *err = 1;
            break;
        }
    }
    __threadfence_system();
}

// Sends chunk c of this GPU's payload (`mine` floats) to GPU b, then raises the chunk's flag.
__device__ void send_chunk(const legion_exchange & p, int b, int c, int parity, int64_t mine) {
    const int64_t cf = p.chunk / 4;
    const Route r    = route(p, p.me, b, parity, mine * 4);
    const int64_t lo = (int64_t) c * cf;
    const int64_t hi = lo + cf < mine ? lo + cf : mine;
    if (hi > lo) {
        pack_range(p, (float *) r.slot, lo, hi);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag(r.flags, parity, p.me, c) = p.v;
    }
}

// A reduce's sum block j: chunk j / sub's slice j % sub, once every writer's chunk is in; the
// parts added in GPU order. FAN (a scattered reduce's owner, sub 1): the sum also goes into
// every other GPU's slot of this GPU, at the chunk's place, and the chunk's flag goes up there.
template <bool FAN> __device__ void reduce_item(const legion_exchange & p, int j, int parity) {
    const int64_t cf = p.chunk / 4;
    const int c      = j / p.sub;
    const int s      = j % p.sub;
    const int64_t lo = (int64_t) c * cf;
    const int64_t hi = lo + cf < p.e ? lo + cf : p.e;
    const int64_t per = padded((hi - lo + p.sub - 1) / p.sub);
    const int64_t a   = lo + s * per;
    const int64_t b   = a + per < hi ? a + per : hi;
    if (threadIdx.x == 0) {
        // The first sum block counts the waits (one count per writer and exchange); an owner
        // its first chunk's.
        await_all(p, parity, c, FAN ? c == p.me : j == 0);
    }
    __syncthreads();
    const float * parts[MAX_DEV];
    float * fan[MAX_DEV];
    bool vec = ((((uintptr_t) p.dst) | ((uintptr_t) p.src[0])) % 16) == 0;
    for (int w = 0; w < p.n_dev; w++) {
        parts[w] = w == p.me ? (const float *) p.src[0] : (const float *) route(p, w, p.me, parity, p.e * 4).slot;
        vec      = vec && ((uintptr_t) parts[w]) % 16 == 0;
        if (FAN) {
            fan[w] = w == p.me ? (float *) p.dst : (float *) route(p, p.me, w, parity, p.e * 4).slot;
            vec    = vec && ((uintptr_t) fan[w]) % 16 == 0;
        }
    }
    float * dst = (float *) p.dst;
    int64_t i0  = a;
    if (vec && b > a) {
        // `a` is a multiple of 4 (chunks and slices are): float4 loads, the same additions.
        const int64_t n4 = (b - a) / 4;
        for (int64_t q = threadIdx.x; q < n4; q += THREADS) {
            const int64_t i = a / 4 + q;
            float4 acc      = p.me == 0 ? ((const float4 *) parts[0])[i] : __ldcv((const float4 *) parts[0] + i);
            for (int w = 1; w < p.n_dev; w++) {
                const float4 x = w == p.me ? ((const float4 *) parts[w])[i] : __ldcv((const float4 *) parts[w] + i);
                acc.x += x.x;
                acc.y += x.y;
                acc.z += x.z;
                acc.w += x.w;
            }
            if (FAN) {
                for (int w = 0; w < p.n_dev; w++) {
                    ((float4 *) fan[w])[i] = acc;
                }
            } else {
                ((float4 *) dst)[i] = acc;
            }
        }
        i0 = a + n4 * 4;
    }
    for (int64_t i = i0 + threadIdx.x; i < b; i += THREADS) {
        float acc = p.me == 0 ? parts[0][i] : __ldcv(parts[0] + i);
        for (int w = 1; w < p.n_dev; w++) {
            acc += w == p.me ? parts[w][i] : __ldcv(parts[w] + i);
        }
        if (FAN) {
            for (int w = 0; w < p.n_dev; w++) {
                fan[w][i] = acc;
            }
        } else {
            dst[i] = acc;
        }
    }
    if (FAN) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < p.n_dev; w++) {
                if (w != p.me) {
                    *flag(route(p, p.me, w, parity, p.e * 4).flags, parity, p.me, c) = p.v;
                }
            }
        }
    }
}

// The k-th chunk of `cm` that a scattered reduce's GPU `me` does not own (chunk c is GPU
// c % n's): k's group of n - 1, `me` skipped in it.
__device__ __forceinline__ int other_chunk(int k, int n, int me) {
    const int r = k % (n - 1);
    return k / (n - 1) * n + (r >= me ? r + 1 : r);
}

// A reduce as a reduce-scatter and an all-gather (#686; the Rust side's `scatter_cpu` is its
// twin, `Exchange::scatters` says which reduces go this way): every GPU sends the chunks of its
// part it does not own to their owners; each owner
// adds its chunks' parts in GPU order (the bits of every GPU adding everything) and sends the
// sums back out, into the places of the readers' slots its own sends leave alone; every GPU
// copies the others' sums. All sends come first, then the sums (they wait only for sends),
// then the copies (they wait only for sums).
__device__ void scatter_reduce(const legion_exchange & p, int parity) {
    const int64_t cf  = p.chunk / 4;
    const int n       = p.n_dev;
    const int cm      = n_chunks(p.e, cf);
    const int own     = (cm - p.me + n - 1) / n;
    const int others  = cm - own;
    for (int i = blockIdx.x; i < others; i += gridDim.x) {
        const int c = other_chunk(i, n, p.me);
        send_chunk(p, c % n, c, parity, p.e);
    }
    for (int i = blockIdx.x; i < own; i += gridDim.x) {
        __syncthreads();
        reduce_item<true>(p, p.me + i * n, parity);
    }
    float * dst = (float *) p.dst;
    for (int i = blockIdx.x; i < others; i += gridDim.x) {
        const int c   = other_chunk(i, n, p.me);
        const int o   = c % n;
        const Route r = route(p, o, p.me, parity, p.e * 4);
        __syncthreads();
        if (threadIdx.x == 0) {
            await_one(p, flag(r.flags, parity, o, c), o, c == o);
        }
        __syncthreads();
        const int64_t lo = (int64_t) c * cf;
        const int64_t hi = lo + cf < p.e ? lo + cf : p.e;
        if (hi > lo) {
            copy_f32(dst + lo, (const float *) r.slot + lo, hi - lo, true);
        }
    }
}

// --- select: one block per GPU, one chunk, host inboxes (as before #671) ---

__device__ __forceinline__ char * host_slot(const legion_exchange & p, char * inbox, int parity, int writer) {
    return inbox + HDR + (int64_t) (parity * p.n_dev + writer) * p.slot_bytes;
}

__device__ __forceinline__ int owner_of(const legion_exchange & p, int32_t e) {
    for (int d = 0; d < p.n_dev; d++) {
        if (e < p.bounds[d + 1]) {
            return d;
        }
    }
    return p.n_dev - 1;
}

// GPU w's select payload from `from` (an inbox slot, or null for this GPU's own sources) into
// dst: the columns (token t, choice j) whose expert w owns, in (t, j) order, then the shared
// expert's columns if w runs it.
__device__ void select_unpack(const legion_exchange & p, int w, const char * from) {
    const bool remote   = from != nullptr;
    float * dst         = (float *) p.dst;
    const int64_t pairs = (int64_t) p.n_used * p.n;
    int64_t k           = 0;
    for (int64_t q = 0; q < pairs; q++) {
        if (owner_of(p, p.ids[q]) != w) {
            continue;
        }
        const float * s = remote ? (const float *) from + k * p.e : (const float *) p.src[0] + q * p.e;
        copy_f32(dst + q * p.e, s, p.e, remote);
        k++;
    }
    if (w == p.sh) {
        const float * s = remote ? (const float *) from + k * p.e : (const float *) p.src[1];
        copy_f32(dst + pairs * p.e, s, p.n * p.e, remote);
    }
}

// This GPU's select payload into `to` (another GPU's inbox slot).
__device__ void select_pack(const legion_exchange & p, char * to) {
    float * out         = (float *) to;
    const int64_t pairs = (int64_t) p.n_used * p.n;
    int64_t k           = 0;
    for (int64_t q = 0; q < pairs; q++) {
        if (owner_of(p, p.ids[q]) != p.me) {
            continue;
        }
        copy_f32(out + k * p.e, (const float *) p.src[0] + q * p.e, p.e, false);
        k++;
    }
    if (p.me == p.sh) {
        copy_f32(out + k * p.e, (const float *) p.src[1], p.n * p.e, false);
    }
}

// Block b != me sends this GPU's payload to GPU b, then copies b's out; block me its own.
__device__ void select_block(const legion_exchange & p, int b, int parity) {
    if (b == p.me) {
        select_unpack(p, p.me, nullptr);
        return;
    }
    select_pack(p, host_slot(p, p.inbox[b], parity, p.me));
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag(p.inbox[b], parity, p.me, 0) = p.v;
    }
    char * mine = p.inbox[p.me];
    if (threadIdx.x == 0) {
        await_one(p, flag(mine, parity, b, 0), b, true);
    }
    __syncthreads();
    select_unpack(p, b, host_slot(p, mine, parity, b));
}

__global__ void __launch_bounds__(THREADS) exchange_kernel(const legion_exchange p) {
    const int parity = (int) (p.v & 1);
    if (p.mode == MODE_SELECT) {
        if ((int) blockIdx.x < p.n_dev) {
            select_block(p, blockIdx.x, parity);
        }
        return;
    }
    if (p.mode == MODE_REDUCE && p.scatter) {
        scatter_reduce(p, parity);
        return;
    }
    const int64_t cf   = p.chunk / 4;
    const int64_t mine = payload_floats(p, p.me);
    const int cm       = n_chunks(mine, cf);
    // A gather to one GPU (`root`, mode gather): the others only send to it.
    const bool to_root = p.mode == MODE_GATHER && p.root >= 0;
    // Sends: every chunk to every reader, the readers from the next GPU on.
    const int readers = to_root ? (p.me == p.root ? 0 : 1) : p.n_dev - 1;
    for (int i = blockIdx.x; i < readers * cm; i += gridDim.x) {
        const int b = to_root ? p.root : (p.me + 1 + i / cm) % p.n_dev;
        send_chunk(p, b, i % cm, parity, mine);
    }
    if (to_root && p.me != p.root) {
        return;
    }
    if (p.mode == MODE_REDUCE) {
        for (int j = blockIdx.x; j < cm * p.sub; j += gridDim.x) {
            __syncthreads();
            reduce_item<false>(p, j, parity);
        }
        return;
    }
    // Receives: every writer's chunks (this GPU's own from its sources).
    int first[MAX_DEV + 1];
    first[0] = 0;
    for (int w = 0; w < p.n_dev; w++) {
        first[w + 1] = first[w] + n_chunks(payload_floats(p, w), cf);
    }
    for (int i = blockIdx.x; i < first[p.n_dev]; i += gridDim.x) {
        int w = 0;
        while (i >= first[w + 1]) {
            w++;
        }
        const int c       = i - first[w];
        const int64_t all = payload_floats(p, w);
        const float * from = nullptr;
        __syncthreads();
        if (w != p.me) {
            const Route r = route(p, w, p.me, parity, all * 4);
            if (threadIdx.x == 0) {
                await_one(p, flag(r.flags, parity, w, c), w, c == 0);
            }
            from = (const float *) r.slot;
        }
        __syncthreads();
        const int64_t lo = (int64_t) c * cf;
        const int64_t hi = lo + cf < all ? lo + cf : all;
        if (hi > lo) {
            unpack_range(p, w, from, lo, hi);
        }
    }
}

// Expert parallelism: the tiered ids `mapped` of the router's choices `sel` (both `n` I32), -1
// where the chosen expert is not this GPU's (outside lo..hi), which the expert matvec skips.
__global__ void ep_mask_kernel(const int32_t * mapped, const int32_t * sel, int32_t * out, int32_t n, int32_t lo,
                               int32_t hi) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int32_t e = sel[i];
        out[i]          = e >= lo && e < hi ? mapped[i] : -1;
    }
}

// Expert parallelism's share of a MoE sum: out[i, t] = sum over the choices j of token t whose
// expert sel[j, t] is in lo..hi of w[j, t] * y[i, j, t], in choice order, plus add[i, t] (if
// any). y's other columns are never read (the expert matvec left them unwritten). Grid
// (ceil(e / 256), n), 256 threads.
__global__ void ep_sum_kernel(const float * y, const float * w, const int32_t * sel, const float * add, float * out,
                              int64_t e, int32_t n_used, int32_t lo, int32_t hi) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (i >= e) {
        return;
    }
    float acc = 0.0f;
    for (int32_t j = 0; j < n_used; j++) {
        const int32_t s = sel[t * n_used + j];
        if (s >= lo && s < hi) {
            acc += w[t * n_used + j] * y[(t * n_used + j) * e + i];
        }
    }
    if (add) {
        acc += add[t * e + i];
    }
    out[t * e + i] = acc;
}

// `--glm-pair-balance` (#729): which GPU of the pair owning experts lo..hi (the first GPU owns
// lo..mid) computes each chosen expert of the step: a cached one (cached[e] != 0) its owner, an
// uncached one, in ascending id order, the GPU with less work so far (hit / miss per expert),
// the owner on a tie (src/residency/ep_balance.rs's assign_pair). out[i] is `mine` where the
// choice sel[i] goes to this GPU (`side` of the pair), else -1. One block, 256 threads,
// 2 * (hi - lo) ints of shared memory: who computes each expert (-1 not chosen), then the
// uncached chosen ones in id order. The first warp gives the cached ones to their owners and
// lists the uncached ones (ballots, 32 ids at a time); its first lane assigns the list.
__global__ void ep_assign_kernel(const int32_t * sel, const int32_t * cached, int32_t * out, int32_t n, int32_t lo,
                                 int32_t mid, int32_t hi, int32_t side, int32_t mine, int32_t hit, int32_t miss) {
    extern __shared__ int32_t who[];
    const int32_t span = hi - lo;
    int32_t *     list = who + span;
    for (int32_t k = threadIdx.x; k < span; k += blockDim.x) {
        who[k] = -1;
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t e = sel[i];
        if (e >= lo && e < hi) {
            who[e - lo] = 2;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const uint32_t lane  = threadIdx.x;
        int32_t        hits0 = 0;
        int32_t        hits1 = 0;
        int32_t        count = 0;
        for (int32_t base = 0; base < span; base += 32) {
            const int32_t  k      = base + (int32_t) lane;
            const bool     chosen = k < span && who[k] == 2;
            const bool     in0    = lo + k < mid;
            const bool     is_hit = chosen && cached[lo + k] != 0;
            const bool     is_mis = chosen && !is_hit;
            const uint32_t h0     = __ballot_sync(0xffffffffu, is_hit && in0);
            const uint32_t h1     = __ballot_sync(0xffffffffu, is_hit && !in0);
            const uint32_t m      = __ballot_sync(0xffffffffu, is_mis);
            if (is_hit) {
                who[k] = in0 ? 0 : 1;
            }
            if (is_mis) {
                list[count + __popc(m & ((1u << lane) - 1u))] = k;
            }
            hits0 += __popc(h0);
            hits1 += __popc(h1);
            count += __popc(m);
        }
        __syncwarp();
        if (lane == 0) {
            int32_t cost[2] = { hits0 * hit, hits1 * hit };
            for (int32_t j = 0; j < count; j++) {
                const int32_t k     = list[j];
                const int32_t owner = lo + k < mid ? 0 : 1;
                const int32_t best  = cost[1 - owner] < cost[owner] ? 1 - owner : owner;
                who[k]              = best;
                cost[best] += miss;
            }
        }
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t e = sel[i];
        out[i]          = e >= lo && e < hi && who[e - lo] == side ? mine : -1;
    }
}

// The experts each GPU owns for ep_assign_cross_kernel: GPU d owns b[d]..b[d + 1], b[0] = 0.
#define EP_MAX_DEV 8
struct ep_bounds {
    int32_t b[EP_MAX_DEV + 1];
};

// `--glm-cross-balance` (#731): ep_assign_kernel over all n_dev GPUs. A cached chosen expert
// goes to its owner; an uncached one, in ascending id order, to the GPU where it finishes first:
// its work so far plus `miss` in the owner's pair (GPUs 2k, 2k + 1) or `cross` on another pair
// (read across the sockets); on a tie the owner, then the pair's other GPU, then the lowest
// index (src/residency/ep_balance.rs's assign_cross). out[i] is b[me] where sel[i] goes to GPU
// `me`, else -1. One block, 256 threads, 2 * b[n_dev] ints of shared memory as in
// ep_assign_kernel; the first warp counts each GPU's hits with ballots and lists the misses,
// its first lane assigns them.
__global__ void ep_assign_cross_kernel(const int32_t * sel, const int32_t * cached, int32_t * out, int32_t n,
                                       int32_t n_dev, int32_t me, int32_t hit, int32_t miss, int32_t cross,
                                       ep_bounds bounds) {
    extern __shared__ int32_t who[];
    const int32_t span = bounds.b[n_dev];
    int32_t *     list = who + span;
    for (int32_t k = threadIdx.x; k < span; k += blockDim.x) {
        who[k] = -1;
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t e = sel[i];
        if (e >= 0 && e < span) {
            who[e] = -2;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const uint32_t lane = threadIdx.x;
        int32_t        hits[EP_MAX_DEV];
#pragma unroll
        for (int32_t d = 0; d < EP_MAX_DEV; d++) {
            hits[d] = 0;
        }
        int32_t count = 0;
        for (int32_t base = 0; base < span; base += 32) {
            const int32_t k      = base + (int32_t) lane;
            const bool    chosen = k < span && who[k] == -2;
            int32_t       owner  = 0;
            while (owner + 1 < n_dev && k >= bounds.b[owner + 1]) {
                owner++;
            }
            const bool     is_hit = chosen && cached[k] != 0;
            const bool     is_mis = chosen && !is_hit;
            const uint32_t m      = __ballot_sync(0xffffffffu, is_mis);
#pragma unroll
            for (int32_t d = 0; d < EP_MAX_DEV; d++) {
                hits[d] += __popc(__ballot_sync(0xffffffffu, is_hit && owner == d));
            }
            if (is_hit) {
                who[k] = owner;
            }
            if (is_mis) {
                list[count + __popc(m & ((1u << lane) - 1u))] = k;
            }
            count += __popc(m);
        }
        __syncwarp();
        if (lane == 0) {
            int32_t cost[EP_MAX_DEV];
#pragma unroll
            for (int32_t d = 0; d < EP_MAX_DEV; d++) {
                cost[d] = hits[d] * hit;
            }
            for (int32_t j = 0; j < count; j++) {
                const int32_t k     = list[j];
                int32_t       owner = 0;
                while (owner + 1 < n_dev && k >= bounds.b[owner + 1]) {
                    owner++;
                }
                const int32_t p0   = owner & ~1;
                const int32_t p1   = p0 + 2 < n_dev ? p0 + 2 : n_dev;
                int32_t       best = owner;
                int32_t       at   = cost[owner] + miss;
                for (int32_t d = p0; d < p1; d++) {
                    if (cost[d] + miss < at) {
                        best = d;
                        at   = cost[d] + miss;
                    }
                }
                for (int32_t d = 0; d < n_dev; d++) {
                    if ((d < p0 || d >= p1) && cost[d] + cross < at) {
                        best = d;
                        at   = cost[d] + cross;
                    }
                }
                who[k]     = best;
                cost[best] = at;
            }
        }
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t e = sel[i];
        out[i]          = e >= 0 && e < span && who[e] == me ? bounds.b[me] : -1;
    }
}

// `--glm-pair-split` (#735): the expert matvec's ids for GPU `side` of the pair owning experts
// lo..hi (the first GPU owns lo..mid). For the router's choice sel[i] (tiered id mapped[i]): an
// expert its owner caches (cached[e] != 0) stays whole with its owner (mapped[i] there, -1 on
// the other GPU); an uncached one is split over both GPUs of the pair (mapped[i] | half on both,
// each computing its half); other pairs' experts -1 (src/model/glm5next/par/split.rs's
// split_host).
__global__ void ep_split_kernel(const int32_t * sel, const int32_t * cached, const int32_t * mapped, int32_t * out,
                                int32_t n, int32_t lo, int32_t mid, int32_t hi, int32_t side, int32_t half) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int32_t e = sel[i];
        int32_t       v = -1;
        if (e >= lo && e < hi) {
            if (cached[e] == 0) {
                v = mapped[i] | half;
            } else if ((e >= mid) == (side != 0)) {
                v = mapped[i];
            }
        }
        out[i] = v;
    }
}

}  // namespace

// The kernels as the CUDA runtime knows them, for Rust's cudaLaunchKernel (see gate_q8.cu).
extern "C" const void * const legion_exchange_kernel = reinterpret_cast<const void *>(&exchange_kernel);
extern "C" const void * const legion_ep_mask_kernel  = reinterpret_cast<const void *>(&ep_mask_kernel);
extern "C" const void * const legion_ep_sum_kernel   = reinterpret_cast<const void *>(&ep_sum_kernel);
extern "C" const void * const legion_ep_assign_kernel = reinterpret_cast<const void *>(&ep_assign_kernel);
extern "C" const void * const legion_ep_assign_cross_kernel =
    reinterpret_cast<const void *>(&ep_assign_cross_kernel);
extern "C" const void * const legion_ep_split_kernel  = reinterpret_cast<const void *>(&ep_split_kernel);
