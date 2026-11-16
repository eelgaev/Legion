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
| `LEGION_OP_PROFILE=1` | Per-op GPU time profile (CUDA): every graph node timed between CUDA events, a table on stderr now and then (`=dev`: per GPU, every row). Disables ggml's CUDA graphs and fusions. A node that reads a weight of 1 MiB or more is keyed `op weight [type] ne0xne1xne2 x<tokens>` and shows `GB/s`: the weight bytes it read per second, its achieved bandwidth (caches and states do not count; an expert stack counts every routed choice as a different expert, so its GB/s above 1 token is an upper bound). External ops take their token count from their input, so a flat output still lands in the right `x<tokens>` row. |
| `LEGION_DEBUG_NGRAM_OLD_POLICY=1` | An n-gram table read from the SSD with an `--ngram-ram` budget: the row cache replaces round-robin, without a hot set (the policy before #1209), for A/B runs. Output does not change. |
| `LEGION_DEBUG_NODE_GROUP_DRAFTS=1` | A model over two nodes (`--node-rank`) with `--spec mtp`: the depth policy may draft for decoding groups in flight (the behaviour before #560 round 3, which drafts nothing there), for A/B runs. Greedy output does not change. |
| `LEGION_DEBUG_NODE_MTP_BF16=1` | A model over two nodes (`--node-rank`) with `--spec mtp`: GLM-5.3-Flash's MTP experts stay BF16 as shipped instead of loading as NVFP4 (both ranks), for A/B runs. Only the drafts change; greedy output does not. |
| `LEGION_DEBUG_NODE_SPLIT_HEAD=1` | A model over two nodes with `--spec mtp` and no `--node-split`: the automatic split counts the MTP head's bytes on rank 1 (the split before #560 round 4, which gives rank 0 more layers); give it to both ranks. Output does not change. |
| `LEGION_DEBUG_NODE_TAKE_IN_NOW=1` | A model over two nodes with `--spec mtp`: rank 1 runs every MTP take-in when it comes, instead of letting decode steps' take-ins wait for the sequence's next draft round (#560), for A/B runs. Set it on rank 1. Greedy output does not change. |
| `LEGION_DEBUG_HOST_MEM_GIB=<n>` | The host expert tier's sizing (#37, a discrete GPU): take `n` GiB as the machine's total and available memory instead of what `/proc/meminfo` says, to try a smaller PC's RAM slot count on a bigger one. `<total>:<available>` gives the two apart (`32:27`: a 32 GB machine with 27 GiB available, which keeps the margin of a 32 GB machine; `27` keeps the smaller margin of a 27 GiB one). The process is not held to it. |
| `LEGION_DEBUG_TIER_FETCH=hostfunc` | The host expert tier (#37): run each layer's fetch of a step's routed experts as a host function on the CUDA stream (the way before) instead of on the launching thread after a wait for the ids, for A/B runs. Output does not change. |
| `LEGION_DEBUG_TIER_READ_PIECE_KIB=<n>` | The host expert tier (#37): a small step reads each missed expert's tensors from the checkpoint in pieces of `n` KiB over the reader threads instead of the default 256; `0` reads every tensor in one read (the way before), for A/B runs. Output does not change. |
| `LEGION_DEBUG_SWAP_COPIES_FIRST=1` | The expert cache on a CUDA device (no bounded host tier): start the after-step's swap copies before the expert tables are written (the way before #37's round 11), for A/B runs; the table write then waits on the copy engine behind the swaps' expert bytes. Output does not change. |
| `LEGION_DEBUG_SWAP_PACE=0` | The expert cache on a CUDA device with the exact `--cpu-expert-share` (#37, no bounded host tier): queue the after-step's swap copies all at once (the way before round 14) instead of letting the next step's share layers start them a few at a time, each behind its layer's pool job, for A/B runs. Output does not change. |
| `LEGION_DEBUG_SHARE_ALL_UPTO=<n>` | `--cpu-expert-share auto` (#37): the pool computes every host expert of a layer's step with at most `n` of them, and the race between pool and link only splits bigger steps (`0`: every step races, the way before round 16). The default is 3 with every routed expert in host memory and 0 on the bounded host tier. Output does not change. |
| `LEGION_DEBUG_EMBD_DEVICE=1` | The bounded host tier's machine and model (#37, one discrete GPU): keep the token embeddings in VRAM (the way before) instead of pinned host memory, for A/B runs. Output does not change. |
| `LEGION_DEBUG_SHARE_STREAM_WAIT=1` | `--cpu-expert-share` on CUDA (#37): the stream waits for the host's "gate/up done" and "done" flags before the `activate` and `collect` kernels (`cuStreamWaitValue32`, the way before) instead of those kernels' host-pair blocks polling the flags, for A/B runs. Output does not change. |
| `LEGION_DEBUG_EXPERT_DUMP=<file>[,every=<n>][,full=<n>][,max=<n>]` | The decode expert matvec (CUDA, #37): write to `file` what the kernels read and wrote in steps of up to 8 tokens (each op's ids, the first 64 bytes of each picked expert, the Q8_1 activation blocks, a digest of each expert's output rows; a SwiGLU op's raw gate and up sums, from two more launches of the same kernels), for `legion expert-cpu-proof check` to recompute on the host and compare bit for bit. `every`: one op in `n`; `full`: the first `n` records carry the rows themselves; `max`: stop after `n` records. Stage graphs leave the expert ops out and recorded steps are slower. Output does not change. |
| `LEGION_DEBUG_NVFP4_PF32=1` | NVFP4 checkpoints' routed experts at prefill (CUDA): the SwiGLU GEMM's 128-token tile prefetches its weights into L2 in runs of 32 blocks (the way before #1176 round 35) instead of 8, for A/B runs. Output does not change. |
| `LEGION_DEBUG_QWEN4_QSA_SHARE=0` | Qwen3.8-Flash-Next over machines with `--tp-seq-par` (CUDA): every rank scores the sparse attention's pools and picks the best for every token of a big step (the way before #1176 round 37) instead of for its share of the tokens, then exchanging the picks, for A/B runs. Set it on every rank. Output does not change. |
| `LEGION_DEBUG_TP_DEVICE_ROWS=0` | Qwen3.8-Flash-Next over machines with MTP (CUDA): a prompt chunk's step reads all its captured rows back on every rank and the MTP head's take-in pieces upload them again (the way before #1176 round 39) instead of copying them on the device, for A/B runs. Set it on rank 0. Output does not change. |
| `LEGION_DEBUG_TP_ROW_HINT=0` | Tensor parallelism over machines with an n-gram table: rank 0 no longer sends the other ranks a prompt's next chunk to read ahead (the way before #1176's PR #104); they read each chunk's rows in its own step, for A/B runs. Set it on rank 0. Output does not change. |
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
