# Legion's CUDA kernels

The kernels Legion runs between ggml graph segments, and the shim into ggml's CUDA backend.
build.rs compiles the `.cu` files listed in `build_cuda_kernels` with `cc`, for the architectures
in `CMAKE_CUDA_ARCHITECTURES` (default: the build machine's GPUs), into `liblegion_cuda.a`.
The Rust side reaches them through the `extern "C"` launchers each file declares.

## Layout

| Where | What |
|---|---|
| `*.cu` | One translation unit per op: argument structs, portable kernels, the `extern "C"` launchers and the per-device dispatch. |
| `device_cache.cuh` | Per-device facts the launchers cache: compute capability, SM count, shared-memory opt-in. |
| `sm70/` | Volta and newer: `mma.sync.m8n8k4` code (valid PTX on every sm_70+ device). |
| `sm80/` | Ampere and newer: `mma.sync.m16n8k16`, `ldmatrix`, `cp.async`. |
| `sm89/` | Ada and newer: `mma.sync.m16n8k32` on FP8 E4M3. |
| `sm90/` | Hopper and newer: thread-block clusters and distributed shared memory. Whole translation units (listed in build.rs) launched from Rust with `cudaLaunchKernelExC`; older targets compile a trap. |
| `sm120/` | Consumer Blackwell (sm_120a / sm_121a, GB10): block-scaled FP4 MMA. |
| `ptx/` | Hand-written PTX (not built by nvcc): embedded in the binary and JIT-loaded per device by `src/ptx.rs` (`cuModuleLoadDataEx`, launched with `cuLaunchKernel` from Rust). `expert_mv_vram.ptx` (sm_70): `--expert-mv-ptx`'s Q4_K gate/up and Q5_1 down VRAM-expert kernels, `expert_mv/`'s bits (#703), with `_users` forms for steps of several tokens (#710); written by `gen_expert_mv_vram.py`. |

A directory is named for the oldest architecture its instructions need, not for the GPUs that
end up running the code: the sm70 kernels also run on newer GPUs when the dispatch picks them.

| File | Arch-specific parts |
|---|---|
| `moe_gemm.cu` | `sm70/moe_gemm.cuh` (m8n8k4 K step), `sm80/moe_gemm.cuh` (m16n8k16 K step), `sm80/moe_gemm_raw.cuh` (cp.async raw-staged kernel), `sm80/moe_gemm_frag.cuh` (Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down with B fragments dequantized into registers; picked on compute capability 12.x; their L2 bulk prefetch is sm_90 PTX behind its own guard, a no-op on older targets). `gemm_kernel` itself stays in `moe_gemm.cu`: one template holds both K-step paths. |
| `paged_attn/` | Split by head size (`d128.cu`, `d256.cu`, `d512.cu`, `d512_compact.cu`, `dispatch.cu`; #955); `common.cuh` includes `sm80/paged_attn.cuh` (prefill on m16n8k16, and split over the key blocks with a combine kernel for `--paged-verify-split`, `namespace fa`), `sm70/paged_attn.cuh` (prefill on m8n8k4, `namespace v70`), `sm70/paged_split.cuh` (the WMMA prefill's last-token rows split over the key blocks, `namespace split`), `sm70/paged_vsplit.cuh` (every row of a 2-16-token group, e.g. a verify step, split over the key blocks with the WMMA prefill's bits, `namespace vsplit`, #1211). The WMMA prefill (`paged_prefill`, `common.cuh`) takes 32 query rows a CTA instead of 64 where 64 rows' tiles pass the device's shared memory a block (head size 512 on Turing's 64 KB; #1214), with the same bits. |
| `qsa_attn.cu` | `sm70/qsa_attn.cuh` (prompt kernel on m8n8k4), `sm80/qsa_attn.cuh` (prompt kernel on m16n8k16), `sm80/qsa_attn_gb10.cuh` (the same kernel's bits from ~39 KB of shared memory, two CTAs per SM; picked on compute capability 12.x and Ampere (8.0-8.7, #1212) by `use_gb()`, #828). |
| `fp8_mm.cu` | `sm80/fp8_mm.cuh` (BF16 W8A16 GEMMs), `sm89/fp8_mm.cuh` (E4M3 W8A8 GEMMs). |
| `qwen4_hc.cu` | `sm80/qwen4_hc.cuh` (the hyper-connection mixer as the GB10 runs it: weights staged by `cp.async`, kernels chained by PDL; same bits as the portable kernels, picked on compute capability 12.x (GB10, RTX 50) by `use_gb10()`). |
| `dense_mv.cu` | `sm80/dense_mv.cuh` (the kernel on m16n8k16 and ldmatrix; `mma_code()` checks the binary has it). |
| `glm5next.cu` | `sm80/mla_gb10.cuh` (GLM's sparse MLA prompt kernel as the GB10 runs it: score warps and value warps a chunk apart on named barriers, swizzled `cp.async` chunk buffers; `mla_tiled_kernel`'s bits, picked on compute capability 12.x by `legion_device::sm12x_device()`, #862, and on Ampere by `legion_device::ampere_device()`, #1212), `sm80/linear_tc_gb10.cuh` (`linear_tc_gb`: linear_tc's F16 prompt GEMM and bits on other tile shapes; F16-row products on compute capability 12.x by `tc_gb_auto`, #862; Ampere's own picks by `tc_gb_auto_ampere`, #1212). |
| `sm120/nvfp4_a4.cu` | The whole file (its own translation unit, listed in build.rs). |
| `sm120/nvfp4_mvq.cu` | The matvec kernels (`cvt.rn.f16x2.e2m1x2`, sm_120a/sm_121a; a trap elsewhere, and `nvfp4_mvq::takes` asks `nvfp4_a4::supported_on` first); its activation quantization is plain SIMT code. |
| `sm90/cluster_select.cu` | The whole file: decode-sized top-k (`qsa_attn.cu`'s `topk_kernel` bits) and argmax (ggml-cuda's `argmax_f32` bits) on clusters of 8 CTAs a row (`src/model/cluster_select.rs`). |

The other files have no architecture-specific code (`fused.cu`, `expert_mv/` (split by weight type into its own translation units, #955),
`dequant.cu`, `shim.cu`, `gdn_scan.cu`) or one family only (`nvfp4_mm/` (split into the GEMM,
mv and grouped-mv parts by tile width, #955), `gdn_mma.cu`, sm_80+), and stay whole. `gdn_mma.cu` is Legion's port of ggml-cuda's chunked gated delta rule
(`--gdn-mma`), kernels only and without any ggml-cuda header: `src/model/core/gdn_mma.rs`
picks the schedule, gives the kernels their shared memory and launches them from the
`legion_gdn_mma_kernels` table. `gdn_scan.cu` is plain SIMT code that the Rust side picks on 7.x GPUs only (ggml's bits
there). `deepseek41.cu` (DeepSeek-V4.1-Flash's own ops, `src/model/deepseek41/ops.rs`) is
plain SIMT code too, kernels only: Rust launches them with `cudaLaunchKernel` from the
`legion_ds4_kernels` table.

`mmq_preq.cu` has no kernel: it is the one piece of C++ host code the owner allowed for #363,
the glue that runs ggml-cuda's MMQ (`mul_mat_q_case`, which has no C API) on activations that
`fused.cu`'s `hc_post_norm` already quantized in MMQ's layout. Keep it that small; the Rust
side (`src/model/core/mmq_preq.rs`) decides when it runs.

## The `.cuh` files are not standalone headers

Each `smXX/*.cuh` is a piece of its op's translation unit: the `.cu` file `#include`s it inside
its anonymous namespace, at the place the code sits in the op's order, after the shared helpers
it uses (the header comment of each names them). Keeping one translation unit per op keeps
internal linkage, inlining and the compiled code exactly as when the op was one file. Do not
include them anywhere else, and do not make them separate `.cu` files.

## One binary, every architecture, the path chosen at run time

Every file is compiled for every target architecture of the build (for example
`70;80;89;121a` gives one binary for V100, A100, RTX 40 and GB10). So code that uses newer
instructions keeps its `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800` (or `>= 890`, or
`__CUDA_ARCH_FEAT_SM120_ALL` / `SM121_ALL`) guard, and a kernel whose body is guarded ends in
`#else __trap(); #endif`: a launch the host checks got wrong stops with an error instead of
leaving its output unwritten (an empty body is silent garbage). Device functions may keep an
older fallback. The launchers then pick a path per device, decided once per device:

- `legion_device::compute_capability()` (`device_cache.cuh`): `fa::available()`, `use_m884()`
  (paged_attn), `bf16_mma()`, `fp8_mma()` (fp8_mm).
- `cudaFuncGetAttributes` on the sm_80 kernel (`binaryVersion` and `ptxVersion` >= 80), so a
  binary without sm_80 code never takes a path that would JIT to a trap: `use_mma16()`
  (moe_gemm), `use_mma<HD>()` (qsa_attn), `legion_device::sm12x_device()` (`device_cache.cuh`,
  together with compute capability 12.x: qwen4_hc's and glm5next's GB10 kernels and
  `launch_pdl`'s programmatic dependent launch; `legion_device::ampere_device()`, together with
  compute capability 8.0-8.7: the GB10 kernels that also measured faster on Ampere, #1212).
- Ampere is 8.0 to 8.7 (`legion_device::ampere_cc()`, `crate::hw::per_device::ampere`), never
  `major == 8`: 8.9 (Ada) has picks of its own.
- Compute capability 12.x (GB10 and RTX 50) is one SM: a pick that depends on instructions,
  shared memory or PDL takes the whole family (`cc / 10 == 12`, `sm12x_device()`), never
  `== 121` (#1200). Only choices measured against the GB10's memory (its bandwidth, its 48 SMs)
  or tied to ggml's GB10-only tables (`__CUDA_ARCH__ == 1210`: q4k_head, q8_group's one-column
  launches) key on 12.1 itself, or on `cudaDevAttrIntegrated`.
- `legion_nvfp4_a4_supported()`: a probe kernel that reports whether the current device runs
  code built with the FP4 MMA (`nvfp4_a4::supported_on` asks it once per device).

Answers about the GPU are kept per device, never once per process: one box may hold GPUs of
different architectures. On the Rust side `crate::hw::per_device` keeps them (compute
capability, SM count, a probe's answer) by device ordinal, and an op picks by the device of
the weight it multiplies (`Tensor::cuda_device`). Before any weight is read, the engine checks
each GPU of the layout against the code the binary holds for it (`crate::hw::cuda_arch`: the
`ptxVersion` of one kernel there, every file being built for the same architectures) and
refuses one whose architecture level (7.0, 8.0, 8.9, 9.0+) the code is older than, with the
`CMAKE_CUDA_ARCHITECTURES` target to build: the compute-capability checks above assume the
binary has that architecture's code.

When adding a kernel for one architecture: put it in the matching directory, keep the guard and
stub so the other targets still compile, and add the run-time check to the launcher in the
`.cu` file; a new `.cu` file also needs its line in build.rs.
