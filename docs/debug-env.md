# Environment variables

Features and tuning values are command-line flags of `legion serve` (and `legion run`, which
shares the model flags); see `legion serve --help` and [serving.md](serving.md). New switches
are flags too, never environment variables. What the environment still does is listed here.
`tests/env_reads.rs` fails when the code reads a `LEGION_*` variable this page and its
allow-list do not name. `LEGION_DEBUG_*` switches need no allow-list entry, but each one needs
a row of its own in the table of debug hooks below, and a row whose switch nothing reads any
more fails the test too.

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
without notice. Most `LEGION_DEBUG_*` switches put back the code from before a change, for A/B
runs; unless a row says so, output does not change.

| Variable | Effect |
|---|---|
| `LEGION_OP_PROFILE=1` | Per-op GPU time profile (CUDA): every graph node timed between CUDA events, a table on stderr now and then (`=dev`: per GPU, every row). Disables ggml's CUDA graphs and fusions. A node that reads a weight of 1 MiB or more is keyed `op weight [type] ne0xne1xne2 x<tokens>` and shows `GB/s`: the weight bytes it read per second, its achieved bandwidth (caches and states do not count; an expert stack counts every routed choice as a different expert, so its GB/s above 1 token is an upper bound). External ops take their token count from their input, so a flat output still lands in the right `x<tokens>` row. |
| `LEGION_DEBUG_NGRAM_OLD_POLICY=1` | An n-gram table read from the SSD with an `--ngram-ram` budget: the row cache replaces round-robin, without a hot set (the policy before #1209), for A/B runs. Output does not change. |
| `LEGION_DEBUG_NODE_GROUP_DRAFTS=1` | A model over two nodes (`--node-rank`) with `--spec mtp`: the depth policy may draft for decoding groups in flight (the behaviour before #560 round 3, which drafts nothing there), for A/B runs. Greedy output does not change. |
| `LEGION_DEBUG_NODE_MTP_BF16=1` | A model over two nodes (`--node-rank`) with `--spec mtp`: GLM-5.3-Flash's MTP experts stay BF16 as shipped instead of loading as NVFP4 (both ranks), for A/B runs. Only the drafts change; greedy output does not. |
| `LEGION_DEBUG_NODE_SPLIT_HEAD=1` | A model over two nodes with `--spec mtp` and no `--node-split`: the automatic split counts the MTP head's bytes on rank 1 (the split before #560 round 4, which gives rank 0 more layers); give it to both ranks. Output does not change. |
| `LEGION_DEBUG_NODE_TAKE_IN_NOW=1` | A model over two nodes with `--spec mtp`: rank 1 runs every MTP take-in when it comes, instead of letting decode steps' take-ins wait for the sequence's next draft round (#560), for A/B runs. Set it on rank 1. Greedy output does not change. |
| `LEGION_DEBUG_GLM_KDA_SPLIT=<0-2>` | GLM-5.3-Flash on a DGX Spark (#1172 round 8): how many blocks the KDA scan's decode kernel (a verify step's scan and every MTP commit's) gives each of the 64 heads, as a log2: 0 one (the kernel before round 8), 1 two, 2 four (the default). Each block computes the token's norms itself from the same values in the same order, so greedy output does not change. For A/B runs and `legion lin-bench --kda`. |
| `LEGION_DEBUG_HOST_MEM_GIB=<n>` | The host expert tier's sizing (#37, a discrete GPU): take `n` GiB as the machine's total and available memory instead of what `/proc/meminfo` says, to try a smaller PC's RAM slot count on a bigger one. `<total>:<available>` gives the two apart (`32:27`: a 32 GB machine with 27 GiB available, which keeps the margin of a 32 GB machine; `27` keeps the smaller margin of a 27 GiB one). The process is not held to it. |
| `LEGION_DEBUG_TIER_FETCH=hostfunc` | The host expert tier (#37): run each layer's fetch of a step's routed experts as a host function on the CUDA stream (the way before) instead of on the launching thread after a wait for the ids, for A/B runs. Output does not change. |
| `LEGION_DEBUG_TIER_READ_PIECE_KIB=<n>` | The host expert tier (#37): a small step reads each missed expert's tensors from the checkpoint in pieces of `n` KiB over the reader threads instead of the default 256; `0` reads every tensor in one read (the way before), for A/B runs. Output does not change. |
| `LEGION_DEBUG_SWAP_COPIES_FIRST=1` | The expert cache on a CUDA device (no bounded host tier): start the after-step's swap copies before the expert tables are written (the way before #37's round 11), for A/B runs; the table write then waits on the copy engine behind the swaps' expert bytes. Output does not change. |
| `LEGION_DEBUG_SWAP_PACE=0` | The expert cache on a CUDA device with the exact `--cpu-expert-share` (#37, no bounded host tier): queue the after-step's swap copies all at once (the way before round 14) instead of letting the next step's share layers start them a few at a time, each behind its layer's pool job, for A/B runs. Output does not change. |
| `LEGION_DEBUG_SHARE_ALL_UPTO=<n>` | `--cpu-expert-share auto` (#37): the pool computes every host expert of a layer's step with at most `n` of them, and the race between pool and link only splits bigger steps (`0`: every step races, the way before round 16). The default is 3 with every routed expert in host memory and 0 on the bounded host tier. Output does not change. |
| `LEGION_DEBUG_EMBD_DEVICE=1` | The bounded host tier's machine and model (#37, one discrete GPU): keep the token embeddings in VRAM (the way before) instead of pinned host memory, for A/B runs. Output does not change. |
| `LEGION_DEBUG_HEAT_SIZES=0` | The host expert tier with every routed expert in host memory (#37): give every layer the same number of VRAM expert slots instead of sizing the layers by the saved routing heat (the default). Added by PR #95. |
| `LEGION_DEBUG_SHARE_STREAM_WAIT=1` | `--cpu-expert-share` on CUDA (#37): the stream waits for the host's "gate/up done" and "done" flags before the `activate` and `collect` kernels (`cuStreamWaitValue32`, the way before) instead of those kernels' host-pair blocks polling the flags, for A/B runs. Output does not change. |
| `LEGION_DEBUG_EXPERT_DUMP=<file>[,every=<n>][,full=<n>][,max=<n>]` | The decode expert matvec (CUDA, #37): write to `file` what the kernels read and wrote in steps of up to 8 tokens (each op's ids, the first 64 bytes of each picked expert, the Q8_1 activation blocks, a digest of each expert's output rows; a SwiGLU op's raw gate and up sums, from two more launches of the same kernels), for `legion expert-cpu-proof check` to recompute on the host and compare bit for bit. `every`: one op in `n`; `full`: the first `n` records carry the rows themselves; `max`: stop after `n` records. Stage graphs leave the expert ops out and recorded steps are slower. Output does not change. |
| `LEGION_DEBUG_GDN_PF=<MiB>` | The DeltaNet z prefetch into L2 on a paced side stream (#689; CUDA, compute capability 12.x, stage graphs, dense models, steps of up to 8 tokens): prefetch this many MiB (up to 64, `0` turns it off) instead of the default 8. Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_FORK=0` | The DeltaNet z prefetch: run it on the compute stream, unpaced, instead of on the forked side stream (the default). Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_PACE=<ns>` | The DeltaNet z prefetch: one step every `ns` nanoseconds (up to 1 ms, `0`: all at once) instead of every 1000. Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_SHAPE=<KiB>x<threads>` | The DeltaNet z prefetch: bytes per bulk prefetch (a multiple of 4 KiB, up to 1024 KiB) and threads per CTA (1-32) instead of 4 KiB x 32. Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_FAR=1` | The DeltaNet z prefetch: prefetch as many bytes of the LM head instead of z's rows (a probe: the DRAM traffic without the data). Off by default. Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_LATE=1` | The DeltaNet z prefetch: the prefetch kernel issues its prefetches before it waits for the qkv projection instead of after. Off by default. Added by PR #105. |
| `LEGION_DEBUG_GDN_PF_CONV=1` | The DeltaNet z prefetch: no prefetch node; the layer's `conv_silu` kernel issues the prefetch at its start. Off by default. Added by PR #105. |
| `LEGION_DEBUG_GDN_Z_SIDE=0` | A DeltaNet layer with an FP8 z projection where the z prefetch applies (#689): go back to the paced z prefetch instead of running the z projection itself on the side stream, beside the layer's small kernels (the default). Added by PR #107. |
| `LEGION_DEBUG_SIDE_PRIO=0` | A stage graph with a kernel on the side stream (the z projection above): leave all its kernels at one priority instead of giving the compute stream's kernels the greatest one (the default). Added by PR #107. |
| `LEGION_DEBUG_MVF_ROWS=<n>` | The op for small BF16 matmuls of 1-8 tokens (a DeltaNet layer's alpha and beta gates, #689): `1` or `2` output rows a CTA instead of 4. Added by PR #100. |
| `LEGION_DEBUG_MVQ_SWIGLU=0` | The NVFP4 decode matvec of a SwiGLU FFN's down projection (#689): run the SwiGLU as a kernel of its own instead of inside the matvec's quantization kernel (the default). Added by PR #103. |
| `LEGION_DEBUG_DRAFT_Q4K_MV=<mode>` | The DFlash drafter's Q4_K matmuls (`--spec-draft-weights q4`, `--spec-draft-lm-head q4`, #689): `0` runs them all on ggml's `mul_mat` instead of Legion's Q4_K matvec op, `head` takes the op for the head only, `layers` for the layers only; by default both. Added by PR #102. |
| `LEGION_DEBUG_DRAFT_FUSE=0` | The DFlash drafter's quantized weights (#689): store them one tensor each instead of fused (the default). Added by PR #100. |
| `LEGION_DEBUG_SPEC_WINDOWS=0` | The commit graph of speculative decoding's DeltaNet states (#689): copy each layer's conv window in a launch of its own instead of every layer's in one launch (the default). Added by PR #100. |
| `LEGION_DEBUG_HC_GB10=<mode>` | Qwen3.8-Flash-Next's hyper-connection mixer on a GB10 (#1176): `whole` stages the down rows whole and launches the norm plainly (the kernels before), `ring` or `early` keeps only one of the two; by default both (the down rows through a ring in shared memory, the norm launched early). Added by PR #82. |
| `LEGION_DEBUG_TP_ROWS=0` | Tensor parallelism with an n-gram table read through a store (#1176): the ranks above 0 read the rows their cache misses themselves instead of asking rank 0 for them over the TP link (the default). Added by PR #106. |
| `LEGION_DEBUG_TP_ROW_HINT=0` | Tensor parallelism with an n-gram table (#1176): rank 0 does not send the other ranks a prompt's next chunk to read ahead (the default); they read each chunk's rows in its own step. Added by PR #104. |
| `LEGION_DEBUG_NGRAM_FIFO=1` | An n-gram table read directly from the SSD (#1176): the I/O threads take every read in asking order (the old queue) instead of a step's reads ahead of queued read-ahead. Off by default. Added by PR #104. |
| `LEGION_DEBUG_TP_EP_GET_ROWS=1` | `--tp-moe-ep` (#1176): remap the skipped experts' ids and weights with ggml's `get_rows` (the way before) instead of one remap kernel per table. Off by default. Added by PR #109. |
| `LEGION_DEBUG_GLM_BF16_GEMV=<n>` | GLM-5.3-Flash's BF16 decode GEMV of 8 tokens or fewer where the staged GEMV is on (not V100, #560): `0` linear_mv, `1` linear_mv_bf16, `2` (the default) linear_mv_bf16 at 1 token and linear_mv_bf16x at 2-8; `3` also takes 9-16 tokens on matrices of 2048+-value rows, which moves output. Added by PR #80. |
| `LEGION_DEBUG_GLM_Q6K_MV=<bits>` | GLM-5.3-Flash's Q6_K decode GEMV of 8 tokens or fewer where the staged GEMV is on (#1172): `0` the kernels before; bit 0 (the default, `1`) linear_mv_q6k_rw for MLA's short rows; bit 1 two stages in flight; bit 2 the next kernel launched early; bit 3 the output head's rows on `linear_mv_q6k_rw` too (#1172 round 8); bits 4-6 log2 of more rows a warp. Every mode gives the same bits. For A/B runs and `legion lin-bench --q6k`. Added by PR #108. |
| `LEGION_DEBUG_GLM_SMALL_MV=<bits>` | GLM-5.3-Flash's small decode products of 1-4 token steps where the staged GEMV is on (#1172), a sum of bits: 1 the router's F32 GEMV with the inputs in shared memory; 2 the 128-value Q8_0 rows (`ssm_f_b`, `ssm_g_b`) two a warp; 4 `hc_pre`'s mix weights read once; 8 `hc_pre`'s finish launched early by the mix, its streams and norm weight loaded before the wait; 16 the shared expert's gate, up and GLU in one launch; 32 early triggers in hc_finish_wide, pool_keys and glu_mv_q6k_sp; 64 (opt-in) the kv_a, q_a and shared expert down rows loaded before the wait; 128 the routed experts' decode kernels back to back with PDL. `191` (all but 64) by default, `31` the kernels of PR #111, `7` those of PR #110, `0` the kernels before. Added by PR #110. |
| `LEGION_DEBUG_GLM_SHARE_LEVEL=<0-4>` | GLM-5.3-Flash with `--tp-seq-par` (#560): how much of a step's per-token work runs on the token share, each level adding to the ones below; 4 (all) by default. Set it on every rank alike. Added by PR #97. |
| `LEGION_DEBUG_GLM_DRAFT_PREFIX=1` | GLM-5.3-Flash with `--spec-draft-vocab` (#1172): draft over the head's first rows only, without the number tokens. Off by default. Only the drafts change; greedy output does not. Added by PR #84. |
| `LEGION_DEBUG_NVFP4_PF32=1` | NVFP4 checkpoints' routed experts at prefill (CUDA): the SwiGLU GEMM's 128- and 64-token tiles prefetch their weights into L2 in runs of 32 blocks (the way before #1176 rounds 35 and 36) instead of 8, for A/B runs. Output does not change. |
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
