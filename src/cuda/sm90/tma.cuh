// sm_90 and newer: the mbarrier and tensor-memory-accelerator (TMA) helpers of the TMA-fed GEMMs
// (fp8_gemm8_tma, nvfp4_a4_gemm_tma; #1133).
//
// Not a standalone header: included inside the including file's anonymous namespace after the
// CUDA headers, so each translation unit has its own copy. The helpers exist only where
// __CUDA_ARCH__ >= 900.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
namespace legion_tma {
__device__ __forceinline__ uint32_t saddr(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void bar_init(uint64_t * bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint64_t * bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(saddr(bar)), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void bar_arrive(uint64_t * bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(saddr(bar)) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t * bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra WAIT;\n"
        "}\n" ::"r"(saddr(bar)),
        "r"(parity)
        : "memory");
}
// The barriers' initialization visible to the TMA unit (async proxy); after bar_init, before
// the CTA barrier that publishes them.
__device__ __forceinline__ void bar_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void prefetch_map(const void * map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"((uint64_t) map) : "memory");
}
// A 2D tile (c0: element along the row, c1: row) of `map` into shared memory, completing on `bar`.
__device__ __forceinline__ void tma_2d(void * dst, const CUtensorMap * map, int c0, int c1, uint64_t * bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
            saddr(dst)),
        "l"((uint64_t) map), "r"(c0), "r"(c1), "r"(saddr(bar))
        : "memory");
}
}  // namespace legion_tma
#endif
