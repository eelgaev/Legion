# Environment variables

Features and tuning values are command-line flags of `legion serve` (and `legion run`, which
shares the model flags); see `legion serve --help` and [serving.md](serving.md). New switches
are flags too, never environment variables. What the environment still does is listed here.
`tests/env_reads.rs` fails when the code reads a `LEGION_*` variable this page and its
allow-list do not name.

## Variables that mirror a flag

Most `legion serve` flags take their value from a variable when the flag is not given
(`--model` from `LEGION_MODEL`, `--port` from `LEGION_PORT`, ...). `--help` shows each one as
`[env: LEGION_...]`. A flag on the command line wins.

## Secrets

| Variable | Flag | Use |
|---|---|---|
| `LEGION_API_KEY` | `legion serve --api-key`, `legion chat --api-key` | API keys, kept out of the process list. |
| `LEGION_BENCH_API_KEY` | `legion bench --api-key` | The key `legion bench` sends. |
| `LEGION_TLS_CERT`, `LEGION_TLS_KEY` | `--tls-cert`, `--tls-key` | HTTPS certificate and key paths. |

## Removed

These worked as environment variables before #421 and as deprecated aliases until #737. They
are no longer read: setting one does nothing. Use the flag.

| Variable | Use instead |
|---|---|
| `LEGION_MOE_RAW=0` | `--no-moe-raw-staging` |
| `LEGION_MOE_MMA16=0` | none: the fused MoE op runs `mma.m16n8k16` wherever the GPU and the binary have it |
| `LEGION_QSA_MMA=0` | none: qwen4exp's sparse-attention prompt kernel runs `mma.m16n8k16` wherever the GPU and the binary have it |
| `LEGION_CPU_ISA=<tier>` | none: the CPU kernels take the best tier the CPU has (`legion cpu-bench` times every tier) |
| `LEGION_GATED_NORM=1` | `--gated-norm` |
| `LEGION_MOE_COMBINE=1` | `--moe-combine` |
| `LEGION_DELTA_PREP=0` | `--delta-prep-ggml` |
| `LEGION_MOE_PAIR_OUT=f16\|bf16` | `--moe-pair-out f16\|bf16` |
| `LEGION_EXPERT_MV_VRAM=<mode>` | `--expert-mv-vram <mode>` |
| `LEGION_EXPERT_MV_DEVICE=1` | `--expert-mv-device` |
| `LEGION_EXPERT_MV_MAX_TOKENS=<n>` | `--expert-mv-max-tokens <n>` |
| `LEGION_EXPERT_MV_GLU=0` | `--expert-mv-split-glu` |
| `LEGION_QSA_SCORES_TC=1` | `--qsa-scores-tc` |
| `LEGION_QSA_ATTN_CHAIN=1` | `--qsa-attn-chain` |
| `LEGION_QWEN4_HC=0` | `--hc-mix-ggml` |
| `LEGION_QWEN4_HC_POST_NORM=0` | `--hc-post-norm-ggml` |
| `LEGION_AFTER_OVERLAP=0` | `--no-after-overlap` |
| `LEGION_LOAD_THREADS=<n>` | `--load-threads <n>` |
| `LEGION_LOAD_PARALLEL_DEVICE=1` | `--load-parallel-device` |
| `LEGION_NO_MMAP=1` | `--no-mmap` |
| `LEGION_SPEC_DRAFT_WINDOW=<n>` | `--spec-draft-window <n>` |
| `LEGION_EXPERT_PROFILE_SEED=<x>` | `--expert-profile-seed <x>` |
| `LEGION_MTP_HEAD_TYPE=q8_0` | `--spec-draft-head-type q8_0` |

## Debug and development hooks

For people working on Legion's kernels and tests. They are not settings: they may change or go
without notice.

| Variable | Effect |
|---|---|
| `LEGION_OP_PROFILE=1` | Per-op GPU time profile (CUDA): every graph node timed between CUDA events, a table on stderr now and then. Disables ggml's CUDA graphs and fusions. |
| `LEGION_DEBUG_NGRAM_OLD_POLICY=1` | An n-gram table read from the SSD with an `--ngram-ram` budget: the row cache replaces round-robin, without a hot set (the policy before #1209), for A/B runs. Output does not change. |
| `LEGION_META_DEBUG=1` | Tensor parallelism: print how each tensor is split over the GPUs. |
| `LEGION_DEBUG_FAIL_CUMEMCREATE_AFTER=<n>[,<count>]` | The expert cache's slot allocations (CUDA): after the first `n` `cuMemCreate` calls for VRAM slots, the next `count` (1 if left out) fail as out of memory, to try the load's shrink to fit (#1208: the cache plans that GPU again in less and the memory plan says how many slots it kept). |
| `LEGION_EXPERT_COUNTS=<file>` | On exit, write the raw routing counts per layer and expert as JSON, for offline expert-cache studies. |
| `LEGION_SSD_EXPERT_TRACE=<file>` | The SSD expert tier: write every step's routed expert ids per layer and the router lookahead's predictions of them (GLM-5.3-Flash: the next layer's router on the post-attention and post-MoE states, and on the MTP draft's state under `--spec mtp`; binary, `src/residency/ssd_experts/trace.rs`), for `legion ssd-replay <file>` to compare slot policies, prediction recall and prefetches offline. Small steps run one extra router per layer while it is set. |
| `LEGION_CPU_BENCH=<rows>,<columns>` | The shape of the ignored CPU kernel micro-benchmark (`cpu::bench`). |
| `LEGION_MOE_SPEED_TYPES=<gu>,<down>` | GGML type numbers of the experts in the ignored fused-MoE speed test. |
| `LEGION_TINY=1` | The ignored sparse-attention GPU tests: smallest cases only (for compute-sanitizer runs). |
| `LEGION_TEST_DS41_HF=<dir>` | The ignored `released_token_map` test: the DeepSeek-V4.1-Flash directory holding the checkpoint's `tokenizer.json`, whose Engram token map is compared with the released GGUF's. |
| `LEGION_TEST_PAIR_*`, `LEGION_TEST_HC_*` | Model, text and sizes of the ignored KL measurements in `src/engine/lm/tests/` (see their file headers). |

## Build time

`LEGION_GIT_HASH`, `LEGION_GGML_HASH` and `LEGION_CUDA` are set by `build.rs` for `legion
--version` and `/health`; setting them at run time does nothing.
