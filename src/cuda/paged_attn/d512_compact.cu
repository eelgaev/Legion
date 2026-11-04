// paged_attn's kernels for Gemma 4's compact global KV cache (see common.cuh): their own
// translation unit, compiled in parallel with the head sizes'.
#include "common.cuh"

int legion_paged::launch_compact(const legion_paged_attn & p, cudaStream_t st) {
    if (p.d != 512) {
        return 1;
    }
    switch (p.n_head / p.n_head_kv) {
        case 4: return launch<512, 4, true>(p, st);
        case 8: return launch<512, 8, true>(p, st);  // 31B: 32 query heads, 4 kv heads
        case 16: return launch<512, 16, true>(p, st);
        default: return 1;
    }
}
