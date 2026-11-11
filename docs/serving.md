# Serving with Legion

This guide explains how to run the Legion server and connect clients to it.

## Quick start

### Mock engine
To run the server with the GPU-free mock engine:
```sh
cargo run --release -- serve
```

### Real GGUF model
To serve a real GGUF model (Qwen3.5 or Qwen3.6, dense or MoE), specify the model path and the device:
```sh
cargo run --release -- serve --model /path/to/model.gguf --device cpu
```
Supported devices: `cpu` or `cuda`.

## Flags

| Flag | Env Var | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `--host` | `LEGION_HOST` | `0.0.0.0` | The host to bind the server to. |
| `--log-format` | `LEGION_LOG_FORMAT` | `pretty` | Log format: `pretty` (human-readable) or `json` (for log collectors). |
| `--color` | - | `auto` | ANSI colour in log lines (every command) and in the startup report: `auto` colours a terminal unless `TERM=dumb` or `NO_COLOR` is set, `always`, `never`. On a terminal (or with `always`) `serve` prints a startup report: hardware, the model's metadata and settings before the load, then where the load put the weights, KV cache, expert slots, host arenas, n-gram table and Engram tables, per GPU and per NUMA node. |
| `--status-line` | - | `auto` | One live status line under the logs, redrawn every 350 ms (sequences in prefill/decode, queued, prompt and generation tok/s, KV use, expert cache hits, MTP acceptance): `auto` on a terminal unless `TERM=dumb`, `on`, `off`. Never with `--log-format json`. |
| `--stats-interval` | - | off | Without the live line: log a `server stats` event every N seconds. |
| `--port` | `LEGION_PORT` | `8090` | The port to listen on. |
| `--engine` | - | `mock` (if neither `--model` nor `--hfmodel` is given) | Engine to use: `mock` or `model`. |
| `--model` | `LEGION_MODEL` | - | GGUF model file to serve (Qwen3.5 / 3.6: `qwen35` or `qwen35moe`; Qwen3.8-Flash-Next (Qwen4 architecture): `qwen4exp`; Nemotron-H: `nemotron_h` or `nemotron_h_moe`, incl. Nemotron 3.5 Bolt 74B-A9B's blocks with their own expert counts and latent MoE; Gemma 4: `gemma4` (incl. Nemotron 3.5 Bolt E4B), or a Gemma 4 Hugging Face checkpoint directory). |
| `--hfmodel` (`--hf-model`) | - | - | A Hugging Face repo to load from the local cache instead of `--model`: `<org/name>[@<revision>]` (revision: a branch, tag or commit hash; default `main`). Resolves offline — no download, no tokens — to the snapshot under the cache root (`$HF_HUB_CACHE`, else `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`) that a previous `huggingface-cli download <org/name>` (or `hf download`) fetched, and then loads it exactly as `--model <dir>` would: the safetensors checkpoint, or the snapshot's GGUF files (a split model: the first shard; several quantizations, or quantizations in subdirectories: an error listing them, pass one with `--model <path>`). A download that stopped part way (shards missing, `.incomplete` blobs) is reported before loading. The model id in `/v1/models` defaults to the repo id `org/name`. Example: `legion serve --hfmodel nvidia/Qwen3.8-Flash-Next-NVFP4 --device cuda`. |
| `--mmproj` | - | - | A vision projector (mmproj GGUF, `qwen3vl_merger`, e.g. unsloth's `mmproj-F16.gguf`) for a Qwen3.5 / 3.6 model (`qwen35`, `qwen35moe`): requests may carry images through all three APIs and the web UI, and `/v1/models` lists the `vision` capability. The tower runs on the CPU (up to 8 threads; ~7 s for a 640 × 480 image on a laptop). Not with `--tp-rank` or `--node-rank`. |
| `--device` | - | `cpu` | Where the model runs: `cpu` or `cuda`. |
| `--ctx-size` | - | `8192` | Max tokens per sequence (prompt + output). |
| `--threads` | - | `8` | CPU threads for the CPU backend. |
| `--nvfp4-act` | `LEGION_NVFP4_ACT` | `a8` | Activations of NVFP4 matmuls: `a8` keeps them at 8 bits or wider (as W4A16 checkpoints are meant to run); `a4` lets CUDA quantize them to 4 bits on Blackwell for weights the GGUF does not list in `prec_a4`, as llama.cpp does (faster prefill; on Nemotron 3.5, KL 2.0e-2 vs 2.1e-3 against BF16 activations). |
| `--fp8-act` | `LEGION_FP8_ACT` | `a8` | Activations of FP8 matmuls (ModelOpt FP8 layers kept raw): `a8` quantizes them to E4M3 with the checkpoint's static per-tensor `input_scale`, the W8A8 numerics the checkpoint declares and vLLM serves (E4M3 tensor cores, about 2x the prefill GEMM throughput); `a16` keeps them F32/BF16 (W8A16), more precise than shipped (KL against BF16 activations, prefill/decode, a16 vs a8: Qwen3.6-27B 2.5e-4/1.8e-4 vs 3.0e-3/1.5e-3; Nemotron 3.5 3.6e-3/3.5e-3 vs 7.3e-3/1.9e-2). |
| `--model-name` | - | `mock`, the `--hfmodel` repo id, or GGUF name | Model id reported by `/v1/models`. |
| `--mock-step-ms` | - | `5` | Simulated decode step time of the mock engine. |
| `--max-batch` | - | `64` | Sequences decoded together. |
| `--max-waiting` | - | `1024` | Requests allowed to wait for a batch slot before new ones get 429 (OpenAI) / 529 (Anthropic). |
| `--scheduling-policy` | - | `fcfs` | Order waiting requests are admitted in: `fcfs` = arrival (the request's `priority` is accepted and ignored); `priority` = lowest `priority` first (vLLM's convention; SGLang's default is the opposite), ties by arrival, prefill in the same order; running requests are preempted only by `--kv-admission on-demand`. See [Priority and fair share](#priority-and-fair-share). |
| `--kv-admission` | - | `reserve` | Model engine: how admission sizes a request's KV cache. `reserve` = the whole prompt plus `max_tokens` at admission (a running request never runs short); `on-demand` = the prompt plus a step's headroom, sequences grow as they decode and the cheapest one is preempted and recomputed when the cache runs short. See [KV admission on demand](#kv-admission-on-demand). |
| `--itl-target` | - | off | Model engine: inter-token latency target in ms. While sequences decode, each step takes only the prompt tokens a step-time model (fitted online) puts under the target. See [Latency targets](#latency-targets). |
| `--ttft-target` | - | off | Model engine: time-to-first-token target in ms. Prompts that would miss it read first; with `--itl-target`, a step may take more prompt tokens to save one. See [Latency targets](#latency-targets). |
| `--max-seqs` | - | `16` | Model engine: sequences generated at once (continuous batching). |
| `--pipeline-groups` | - | `0` (off) | Model engine, several GPUs as a layer split: the decoding sequences as this many groups whose steps are in flight at once, each GPU running one group's step while the next GPU runs another's (#415). Prompts drain the groups first. Qwen3.8-Flash-Next UD-Q4_K_XL on four V100s, `--deterministic`, 8 fixed prompts (#752), `4` against off: MTP k=2 C=4 88 -> 110 tok/s, C=8 107 -> 155; without MTP C=8 133 -> 155; C=1 and 8K prefill unchanged. |
| `--pipeline-min-group` | - | `2` | With `--pipeline-groups`: the fewest sequences per group; fewer sequences make fewer groups, below two whole steps run. `1`: each sequence may be a group of its own. Same setup (#752), `1` against `2`: MTP C=3 79 -> 95 tok/s (three groups instead of whole steps), C=4 equal, C=2 -4%; without MTP C=4 7% below whole steps, because each round's fixed host time outweighs the stages it fills. |
| `--pipeline-mtp-split` | - | off | With `--pipeline-groups`, the MTP head and no `--layer-split`: after loading, decode a short synthetic run as groups, measure a layer's time and the last stage's extra work per round (output head, sampling, commit, MTP drafts), and load again with fewer layers on the last stage if that shortens the rounds (#765; the log shows what it measured and chose; ~25 s more startup). Same setup, `--pipeline-min-group 1`: it measured 0.57 ms a layer and 4.8 ms extra and chose 14/14/14/6; against the even split C=2 91 -> 94, C=4 129 -> 137, C=8 171 -> 188 tok/s, but C=1 -3%, chat -2%, 8K prefill -1% and 32K prefill 5239 -> 3509 tok/s. |
| `--split-last-extra` | - | `0` (off) | Several GPUs, no `--layer-split`: split the layers as if the last GPU also did this many layers' worth of other work per step (output head, MTP drafts), so it gets fewer layers (48 layers over four GPUs: `2` gives 13/12/12/11, `3` gives 13/13/12/10). The static form of `--pipeline-mtp-split` (its log's `extra_layers`), without the second load (#814). Whole steps and prompts cross the stages in turn, so it can pay only with `--pipeline-groups`. On four V100s (Qwen3.8-Flash-Next, MTP k=2, `--deterministic`, no groups) `2` moved the stages' decode busy time from 37.7/34.0/34.5/37.0 s to 42.0/34.6/34.6/34.4 s (the first GPU's 13 layers share its expert cache: 37% more host-expert bytes there), C=1 75.9 -> 75.8, C=4 97.3 -> 94.8, C=16 143.4 -> 141.1 tok/s, 32K prefill 5802 -> 5636, and the greedy texts differ from the even split's. |
| `--tp-replicas` | - | `1` | Model engine, with `--tp`: run this many copies of the model in one server, replica r on the `--tp` group from CUDA r×tp (on an AC922 with `--tp 2`: one NVLink pair each), each with its own `--max-seqs`, KV cache (sized from its own GPUs), prefix cache and, with `--numa auto`, its threads and host memory on its GPUs' NUMA node. A new request goes to the replica with the fewest requests in flight (ties: the most free KV), or with prefix caching on to the one that most likely holds at least half of its prompt (256+ tokens) while it has room, and stays there. Each replica decodes exactly as a single `--tp` server on its GPUs would. `auto` = as many groups as the visible GPUs hold. `/metrics` adds `legion_engine_replica_{requests,running,waiting,kv_blocks_used,kv_blocks_total}{replica="r"}`. |
| `--grammar-threads` | - | `2` | Model engine: threads that compute the token masks of grammar-constrained requests (`response_format` JSON, `strict` tools) for the next step while the model runs the current one; `0` computes each mask on the engine thread when sampling. Started only once a request is constrained. See [Constrained output](#constrained-output-json-and-strict-tool-calls). |
| `--max-batch-tokens` | - | `1024` (`4096` on a GPU sharing the host's memory, e.g. the GB10) | Model engine: tokens per forward pass, one per decoding sequence plus prompt chunks. On the GB10 the bigger default changes greedy output for prompts longer than 512 tokens; `--max-batch-tokens 1024 --prefill-chunk 512` gives other devices' chunking. |
| `--prefill-chunk` | - | `512` (on a GPU sharing the host's memory: `--max-batch-tokens`) | Model engine: prompt tokens per prefill chunk. |
| `--prefill-even` | - | off | Model engine: a prompt that needs several steps takes them in equal parts (multiples of 64 tokens) instead of full steps and a short last one (#758). Each MoE step reads every routed expert, so a short last step costs nearly what a full one's weights cost; with a slightly bigger `--max-batch-tokens` (e.g. 4608) an 8,240-token prompt runs as two steps instead of 4,096 + 4,096 + 48 (two DGX Sparks under `--tp-rank`, `--tp-wire bf16 --tp-seq-par 1024 --max-batch-tokens 4608 --prefill-even`: 8K prefill 2,392 -> 2,503 tok/s, 32K and decode unchanged). Prompts of whole steps are split as before. |
| `--warmup-prompt <TOKENS>` | - | 0 (off) | Model engine: before taking requests, run one greedy prompt of about this many tokens (#760). A step larger than any before it allocates its compute buffer, ggml-cuda's pool and, under `--tp-rank`, the pinned exchange buffers (registered with the NIC again) anew at their full size, so the first long prompt after a start was slower. Use 3 x `--max-batch-tokens` (three full steps, one per kept prompt-step graph's buffer); smaller warmup steps leave the next larger step to grow them again. Two DGX Sparks under `--tp-rank`, `--max-batch-tokens 4608 --prefill-even --warmup-prompt 13824`: the first 8K prompt after decode traffic 2,023 -> 2,452 tok/s, 32K and decode unchanged, greedy output identical; startup +6.4 s. |
| `--warmup-shapes <TOKENS,...>` | - | `1024,256,128` | Model engine: before taking requests (and before `--warmup-prompt`), run one greedy one-token prompt of each size (#836). The first prompt step of a new short size loads the CUDA kernels only it uses and grows ggml-cuda's pool. One DGX Spark (Qwen3.8-Flash-Next NVFP4 + MTP): the first request after start, 53 tokens, prefill 0.54-0.58 s -> 0.17 s; 1K/8K prefill and decode unchanged, greedy output identical; startup +2.5 s. Sizes above half the context are skipped, a failed warmup only warns. `0` turns it off. |
| `--moe-kernel` | `LEGION_MOE_KERNEL` | `auto` | Routed experts of big steps: `auto` = Legion's fused op on CUDA for 64+ tokens (on sm_80+ its GEMMs run on `mma.m16n8k16`; a binary without sm_80 code keeps Volta's `m8n8k4`, and then a GPU sharing the host's memory with every expert in device memory takes ggml's MMQ); `graph` = always ggml's `mul_mat_id`; `fused` = always the op, small steps too. Prompt logits differ between the op (F16 activations, F32 sums) and ggml's MMQ (Q8_1 activations): since #370, prompts of 64+ tokens on sm_80+ give other greedy continuations than before; `--moe-kernel graph` gives the MMQ results. |
| `--hc-inject-f32` | - | off | Opt-in, **moves output** for GGUFs that store the hyper-connection inject weights in BF16 (the IQ2_XS Qwen3.8-Flash-Next): those 2560 x 4 x 4 values a layer load as F32 (exactly), so prompt steps take the fused mixer norm (`hc_post_norm`) instead of ggml's chain, whose BF16 matmul rounds the activations to BF16. GB10, IQ2_XS: prefill +7-9%; teacher-forced KL against the weights as stored 2.5e-3 (two passes as stored: 1.1e-3). Files with F32 inject weights (UD-Q4_K_XL) are unaffected. |
| `--ngram-io` | `LEGION_NGRAM_IO` | `auto` | qwen4exp's n-gram table (on a GPU) and DeepSeek-V4.1-Flash's Engram tables (always; never VRAM, never the weights' read-ahead): `ram` (locked in memory), `direct` (unbuffered SSD reads plus a RAM row cache), `mmap`, or `auto` (`ram` when it fits with a margin and may be locked, else `direct`; always `direct` on a GPU sharing the host's memory, which falls back to `mmap` where the file system has no O_DIRECT). |
| `--ngram-ram <auto\|all\|SIZE>` | - | `auto` | How much of the n-gram (or Engram) table lives in RAM; the rest is read from the SSD (#1209). `auto`: as `--ngram-io` decides (`direct` keeps a row cache of a quarter of the free memory, at most 4 GiB, at most 1 on a GPU sharing the host's memory). `all`: the whole table, locked (`--ngram-io ram`). SIZE: binary units with an optional K/M/G/T suffix (`8G` = 8 GiB, no suffix GiB, decimals allowed) or a share of the memory available at load (`30%`). A SIZE the table fits in takes it whole (`ram`); a smaller one reads it directly from the SSD with exactly that much RAM: up to half of it pins the rows hit most since start (a row hit twice joins; one hit more often takes the place of the hot row hit least), the rest replaces by use (CLOCK), so rows read once do not push out rows read again. Nothing assumes a row is a token, so Engram's hashed rows work the same. The load printout shows the budget and the hot set's size; `/metrics` has `legion_engine_ngram_rows`, `_cache_hits`, `_cache_misses`, `_cache_hit_rate` and `_hot_rows`, `/v1/stats` `engine.ngram_rows`, `ngram_cache_hits` and `ngram_hot_rows`. Refused: `--ngram-io mmap` with `all` or a size, `--ngram-io direct` with `all`, `--ngram-io ram` with a size below the table. The same rows either way: output does not change. `--ngram-ram-gib N` (and `LEGION_NGRAM_RAM_GIB`) still works as `--ngram-ram NG`. |
| `--kv-cache-tokens` | - | - | Model engine: KV cache size in tokens, shared by all sequences and the prefix cache. |
| `--uma-free-gib <GIB>` | - | `16` | Model engine, on a GPU sharing the host's memory (GB10): GiB the automatic KV pool leaves free for the OS and other programs, on top of what it keeps for the step buffers (2 + 2 GiB + 1.5 GiB per 1024 tokens of `--max-batch-tokens`). A `--ctx-size` that does not fit is refused at load. GLM-5.3-Flash Maya-S with MTP takes 89.8 GiB of a DGX Spark's ~116 GiB, so 16 sequences at 131072 need `--uma-free-gib 8` (#1172; with a pool 0.5 GiB smaller, at least 9 GiB stayed available through an 86K-token prompt beside 16 decoding requests). Below ~4 GiB the machine may swap. No effect on the output. |
| `--kv-host-tokens` | `LEGION_KV_HOST_TOKENS` | automatic | Model engine: KV cache tokens past what fits in GPU memory, in pinned memory on each GPU's NUMA node, read over the link (0: none). Automatic: a tier only when the context does not fit in a discrete GPU's memory, sized to the rest of the cache and capped by each node's free memory less a margin of 16 GiB or 10% of the node, whichever is more. A node's free memory is `MemFree` plus the page cache the kernel can drop for the pinned tier, from `/sys/devices/system/node/nodeN/meminfo`: `Active(file)` + `Inactive(file)` less `Mapped` (files mapped by a process, e.g. a model loaded by mmap), `Dirty` and `Writeback` (pages that need a disk write first), at least 0, the sum at most `MemTotal` (#541). The startup log and the error when the context does not fit print each node's figure and how much of it is page cache. Every host block is read once per decoded token (~68.5 GB/s on an AC922's NVLink, PCIe about 3x slower). |
| `--prefix-cache-states` | - | `8` | Model engine: saved recurrent states for prefix caching; 0 turns prefix caching off. |
| `--deterministic` | - | off | Model engine: the same bytes for the same request from run to run, whatever the server served before (greedy or seeded). Turns prefix caching off, fixes the draft depth as `--spec-fixed` does and plans every step graph's memory afresh (#459); refuses `--expert-mv-vram` 1 and 2 and a nonzero `--prefix-cache-states` (`0` is accepted). Admits requests in waves (`--deterministic-gather-ms`), so requests sent together give the same bytes too. See [Deterministic output](#deterministic-output). |
| `--deterministic-gather-ms` | - | `200` | With `--deterministic`: an idle server gathers requests until none has arrived for this many milliseconds (at most 20 windows), admits them together ordered by prompt and limits, and holds requests that arrive meanwhile until they are all done (#836). A client that sends its next request when one finishes must make it within the window (~50 ms seen on a POWER9 host). `0`: admit requests as they arrive; requests decoded at the same time then follow their arrival timing. |
| `--prefix-states-vram` | - | off | Model engine, `--experts cache` on discrete GPUs: keep the saved prefix-caching states in VRAM. By default they live in pinned host memory there, and the expert cache takes their VRAM (Qwen3.8-Flash-Next on two V100s: 0.44 GiB per GPU, 5,050 -> 5,346 slots); a save or resume then copies one state over the host link. Outputs are the same either way. |
| `--spec` | - | `none` | Model engine: speculative decoding drafter: `none`, `mtp` (the model's own MTP layers), `ngram`, `suffix`, or a drafter checkpoint given with `--spec-draft`: `draft` (a smaller model of the same vocabulary), `eagle3` (an EAGLE-3 head), `dflash` (a DFlash block drafter, GGUF or Hugging Face `DFlashDraftModel`; also DFlash 2, `DFlash2DraftModel` such as `z-lab/Qwen3.8-27B-DFlash2`, recognized by its config: convolutions and a candidate selector, always the whole block, one draft pass for all the sequences of a step) or `dspark` (a DSpark drafter: DFlash plus a Markov head and an optional confidence head). The checkpoint drafters read the target's hidden states, so they need an architecture that hands them out (not qwen4exp). DFlash/DSpark settings: `--spec-draft-temperature`, `--spec-min-confidence`, `--spec-draft-window`, `--spec-draft-weights`. |
| `--spec-k` | - | per model | Model engine: max draft tokens per sequence and step (the depth policy picks fewer where drafts do not pay). Default: 2 for qwen4exp, 8 on an integrated GPU (GB10); 8 for qwen35 MTP on an integrated GPU; 8 for Gemma 4; 4 otherwise. See [tuning-integrated-gpus.md](tuning-integrated-gpus.md). |
| `--spec-draft` | - | - | Model engine: drafter weights; for `mtp`, the MTP head's GGUF when the model's own GGUF has none (Qwen3.8-Flash-Next: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`; Gemma 4: its `gemma4-assistant` GGUF, e.g. `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`, which reads the target's KV cache. The default depth policy picks 2-3 drafts on the Spark's 26B-A4B and 6-7 on the 31B, as fast as the best `--spec-fixed` depth). |
| `--spec-p-min` | - | `0.5` | Model engine, `--spec mtp`: stop a sequence's drafts at the first one the head gives less than this probability. Without `--spec-fixed` a ceiling: the depth policy lowers it to a draft pass's cost against the time per token. |
| `--spec-draft-vocab` | `LEGION_SPEC_DRAFT_VOCAB` | `0` (all) | Model engine, the MTP heads of qwen4exp and qwen35 (`--spec mtp`) and DFlash drafters that borrow the target's output head (`--spec dflash`; a warning elsewhere): the head scores only the first N tokens of the vocabulary when it drafts, reading N of the output head's rows. Drafts only: the output does not change. Not a default on any device; recommended `65536` with MTP (GB10: Flash-Next ~2 ms less per draft pass; Qwen3.8-27B cycle 117-127 -> 108-113 ms), `131072` with DFlash. Smaller values cut acceptance. The head's probability is renormalized over those N tokens, so `--spec-p-min` and the depth policy's confidence see higher values. See [tuning-integrated-gpus.md](tuning-integrated-gpus.md). |
| `--mtp-window` | `LEGION_MTP_WINDOW` | `0` (all); `32768` for qwen4exp on an integrated GPU | Model engine, `--spec mtp`: the MTP head attends to the last N positions only (0..=2147483647; an idea from Strata's `--mtp-window`, src/program/generate.cpp, default 32768). For heads with dense paged attention: qwen4exp, qwen35 and Nemotron-H heads (CUDA, or CPU with `--attention paged`); not Gemma 4's assistant, a gather path or a sparse head layer (a warning; the flag is then ignored). Measured on qwen4exp on the GB10 only: at 64K-token contexts 32768 lifts greedy decode from 34-35 to 40-42 tok/s (accepted length 2.5 -> 3.1) and halves the head's dense attention (1.4 ms per draft pass at 62K without a window). Verification stays exact. |
| `--spec-draft-head-type` | - | `auto` | Model engine, `--spec mtp`: which weights the MTP head drafts with: `model` (as shipped), `q8_0` (a Q8_0 copy of its 16/32-bit matrices made at load: Qwen3.8-27B-NVFP4's BF16 head 810 -> 430 MiB per draft pass, +5-10% decode on the GB10), `f16` (an F16 copy of its BF16 and F32 matrices made at load, for GPUs without BF16 tensor cores such as the V100; the hyper-connection inject weights stay F32; on 2x V100 with Flash-Next's Q8_0 head, whose only such matrices are its F32 router and BF16 indexer, decode and acceptance did not change), `auto` (`q8_0` on an integrated GPU for qwen35 / qwen35moe heads and for a qwen4exp head shipped 16-bit, Flash-Next-NVFP4's own: 175 -> 92 MiB per draft pass; else `model`). Drafts only: the output does not change. Replaces `LEGION_MTP_HEAD_TYPE` (no longer read). See [tuning-integrated-gpus.md](tuning-integrated-gpus.md). |
| `--spec-draft-temperature` | `LEGION_SPEC_DRAFT_TEMPERATURE` | `0` | DFlash/DSpark: draft temperature, at least 0. `0` drafts each position's most likely token (a point mass); above 0 each draft is sampled and its full distribution goes to the exact rejection sampler. Drafts only: verification keeps the output the target's own. |
| `--spec-min-confidence` | `LEGION_SPEC_DRAFT_MIN_CONFIDENCE` | unset (no cut) | DSpark: stop a block's drafts before the first one whose confidence-head score is below this, in [0, 1]. |
| `--spec-draft-window` | - | `4096` | DFlash/DSpark: positions of context a sequence keeps for drafting, a ring of the newest. Replaces `LEGION_SPEC_DRAFT_WINDOW`. |
| `--spec-draft-weights` | - | `model` | DFlash/DSpark: the weights the drafter drafts with: `model` (as stored) or `q8_0` (Q8_0 copies of a Hugging Face checkpoint's 16/32-bit layer matrices, `fc` and DFlash 2's selector projection, made at load; norms, convolution kernels, embeddings and head as stored; z-lab's Qwen3.8-27B-DFlash2 3.3 -> 1.75 GB a draft pass reads) or `q4` (4-bit copies of the same matrices: Q4_K, Q4_0 for rows not a multiple of 256 wide). A GGUF drafter is read as stored. An FP8 Hugging Face drafter (`quant_method: fp8`, F8_E4M3 with block scales, such as `tcclaviger/Qwen3.8-27B-DFlash2-FP8`) has its FP8 matrices read as Q8_0 with every value. Drafts only: the output does not change. |
| `--spec-draft-lm-head` | - | `auto` | The MTP heads of qwen4exp and qwen35 (`--spec mtp`), and DFlash with a drafter that borrows the target's output head: `auto` (`q4` for the qwen4exp MTP head on an integrated GPU when the target's output head is 16- or 32-bit, Flash-Next-NVFP4's BF16 head; else, and for DFlash, `model`), `model` (the head's rows as stored: all, or the first `--spec-draft-vocab`) or `q4` (a 4-bit copy of those rows made at load, Q4_K with the head's scales folded in; Qwen3.8-27B's FP8 head over 131,072 rows 671 -> 377 MB per draft pass, over 65,536 rows 320 -> 180 MiB per MTP draft token). Drafts only: the output does not change, only acceptance can. Ignored for a drafter with its own head. See docs/tuning-integrated-gpus.md. |
| `--spec-fixed` | - | off | Model engine: always draft `--spec-k` tokens. By default the draft depth adapts, and drafting stops at `--spec-max-batch` decoding sequences: with `--spec mtp` to the measured cost of verify steps and head passes and each sequence's accepted drafts (the depth with the most tokens per millisecond, `src/spec/depth.rs`), with other drafters per sequence to its acceptance rate. |
| `--spec-max-batch` | - | `8` | Model engine: no drafts while this many sequences or more decode in a step (compute-bound: drafts only cost). With `--spec mtp` the measured-cost depth already drops to 0 drafts where they do not pay, so a higher value lets it keep drafting at many users: one GB10, Qwen3.8-Flash-Next UD-Q4_K_XL with the Q8_0 MTP head, `--spec-max-batch 64` gives 8 users 81.6 -> 100.1 tok/s and 16 users 107-109 -> 146-147 (at 8 and more users the default drafts nothing), 1-4 users unchanged, prefill unchanged. It moves greedy text at 8+ users (#827). |
| `--spec-gate-cost-model` | - | `on` (MTP on a V100 layer split of 2 GPUs: gated only up to 8 sequences) | Model engine: `off` exempts the drafters whose depth follows measured step costs (`--spec mtp`, and block drafters such as DFlash 2; `src/spec/depth.rs`) from `--spec-max-batch`: they keep drafting at any number of decoding sequences, and the cost model picks the depth (0 where drafts do not pay). N-gram drafters (including `--spec-mtp-ngram`'s drafts) and other model drafters keep the gate. Below `--spec-max-batch` sequences nothing changes. At `--spec-max-batch` or more sequences greedy text changes (drafted and verified steps batch differently); `--spec-fixed` and `--deterministic` ignore the gate. Left out on an AC922 with 2 GPUs (V100, `--gpus 2 --experts cache`, `--spec mtp` without `--spec-fixed`, `--spec-max-batch` at most 8, #814), the gate stops MTP drafts only from `--spec-max-batch` to 8 sequences: steps of up to 8 rows run as matrix-vector products, which drafts would leave (8 users ungated: 98 -> 93 tok/s), while past 8 every step runs the tiled kernels and the cost model's drafts pay (12 users 71 -> 83 tok/s, 16 users 80 -> 89); C=1/4/8 and prefill unchanged. `on` gates every batch size from `--spec-max-batch` there too. Measured with Qwen3.8-Flash-Next UD-Q4_K_XL GGUF and its Q8_0 MTP head (det, temperature 0): one GB10 C=8 81.9 -> 103.9 tok/s (+27%), C=16 108.9 -> 148.7 (+37%); 4x V100 (`--experts cache --spec-k 2`) no gain, C=8 124.3 -> 121.0, C=16 132.4 -> 134.2 (within the rounds' spread). C=1/4 and prefill unchanged on both. |
| `--spec-row-cost` | - | `on` | Model engine, `--spec mtp` without `--spec-fixed`/`--deterministic`: the depth policy charges each further draft, next to its draft pass, what its row in the verify step costs more than a padding row (the verify shape is the chosen depth's; a MoE reads experts for a drafted row that padding, a repeat of the last token, mostly does not). Measured per verify shape by fitting step time to the rows that carried drafts, so late, rarely accepted drafts stop. `off`: the pass alone (before #836 part 27). Greedy text changes with it, as with any depth change. |
| `--spec-pad-longest` | - | `off` | Speculative decoding: a verify step pads its sequences only to the step's longest draft instead of `1 + --spec-k` (`--spec-fixed`, `--deterministic`) or `1 +` the depth policy's depth. When every draft stopped early (`--spec-p-min`) the step runs fewer rows of attention, DeltaNet, dense matmuls and the output head; each row count is a verify graph of its own (`--verify-kept-graphs`). |
| `--spec-p-floor` | - | `0.5` | Model engine, drafters other than `mtp`: the adaptive depth is the largest one whose last draft is still accepted with at least this probability, in (0, 1). |
| `--default-max-tokens` | - | `4096` | `max_tokens` for OpenAI chat requests that set none. |
| `--api-key` | `LEGION_API_KEY` | - | Require this API key (`Authorization: Bearer <key>` or `x-api-key: <key>`) on every route except GET /health. |
| `--cors` | - | - | Allow this origin in CORS responses, e.g. `http://localhost:3000`. |
| `--no-webui` | - | `false` | Do not serve the built-in web UI at `/`. |
| `--max-request-mb` | - | `64` | Refuse request bodies larger than this many megabytes. |
| `--shutdown-timeout-secs` | - | `30` | On SIGINT/SIGTERM, wait this many seconds for in-flight requests. |
| `--served-model-name` | - | - | Name to serve the model under; replaces the engine id in `/v1/models`. |
| `--sampling-defaults` | `LEGION_SAMPLING_DEFAULTS` | `model` | Defaults for the sampling parameters a request leaves out (see [Sampling defaults](#sampling-defaults)): `model` = the model's own, then OpenAI's; `openai` = OpenAI's (temperature 1, top_p 1, no top-k / min-p, no penalties). |
| `--default-temperature`, `--default-top-p`, `--default-top-k`, `--default-min-p`, `--default-presence-penalty`, `--default-frequency-penalty`, `--default-repetition-penalty` | - | - | The operator's default for that parameter, over the model's (requests that set it still win). |
| `--registry-api` | - | off | Register with an LLM proxy (simple-llm-proxy): POST this server's address, port and first `--api-key` to `<URL>/register` (the URL may end in `/register` or not) once the model is loaded and the server listens, then every `--registry-interval` seconds. Repeatable, one registration per proxy. See [Registering with a proxy](#registering-with-a-proxy). |
| `--registry-host` | - | `--host` if a specific IP, else the local address of the route to the proxy | With `--registry-api`: the IP address the proxy reaches this server on. |
| `--registry-interval` | - | `300` | With `--registry-api`: seconds between registrations after a success. |
| `--fair-share` | - | `off` | Share batch slots between clients: `api-key` (the request's API key; with `--api-key` only accepted keys) or `user` (OpenAI `user`, Anthropic `metadata.user_id`). See [Priority and fair share](#priority-and-fair-share). |

### Sampling defaults

A request that leaves a sampling parameter out gets, in this order of precedence:

1. the operator's `--default-<parameter>`;
2. the model file's declared default: a GGUF's `general.sampling.temp`, `.top_k`, `.top_p`, `.min_p`, `.penalty_repeat` keys, or a safetensors checkpoint's `generation_config.json` (`temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`; `"do_sample": false` is temperature 0);
3. only when the file declares none: the architecture's published values (Gemma 4: temperature 1.0, top_k 64, top_p 0.95; Qwen3.8-Flash-Next: temperature 1.0, top_k 20, top_p 0.95; Qwen3: temperature 0.6, top_k 20, top_p 0.95);
4. OpenAI's defaults.

A field the request sets always wins, `temperature: 0` included (greedy). This holds on `/v1/chat/completions`, `/v1/completions`, `/v1/messages` and `/v1/responses`. The startup log prints each model's effective defaults and where each value came from (`operator`, `model file`, `architecture`, `openai`). So a client that sends no temperature (llama-benchy, for one) gets a sampled run with the model's defaults; comparisons with engines that treat a missing temperature as greedy must send `temperature: 0` (`legion bench` does).

### Kernel and load switches

Opt-in kernels, kernels kept for comparisons and loader tuning (`legion serve --help` lists them
under "Kernel and load switches"; `legion run` takes them too). Without them nothing changes.
They replace `LEGION_*` environment variables, which are no longer read (#737; see
[debug-env.md](debug-env.md) for the old names and the remaining debug hooks).

| Flag | Default | Meaning |
|---|---|---|
| `--gated-norm` | off | CUDA, gated DeltaNet layers: the output norm and its gate as one op instead of ggml's chain. |
| `--moe-combine` | off | CUDA, MoE layers: the weighted expert sum as one op, rounded as ggml-cuda's fused reduction whatever the memory layout (#410). |
| `--stable-fusions` | off | CUDA, MoE layers: keep the inputs of ggml-cuda's fused weighted expert sum (and of ggml's top-k router chain, where a model builds it) allocated until the chain's output, as llama.cpp's scheduler does, so ggml-cuda fuses the sum in every layer whatever the memory layout. Greedy text then no longer moves with unrelated graph changes (`--kv-append`, `--ple-ops`, `--shexp-gate`, `--gated-norm`, a server's history). No extra kernels; the text moves once from the default and equals `--moe-combine`'s. See [Deterministic output](#deterministic-output) (#472). |
| `--delta-prep-ggml` | off | CUDA, gated DeltaNet layers: ggml's chain for the recurrent prep instead of Legion's one op (the same bits). For comparisons. |
| `--node-fuse-ggml` | off | CUDA: ggml's own small kernels instead of the Legion ops that take over their chains once a graph is allocated (the same bits): each MoE layer's weighted expert sum where ggml-cuda runs it unfused (a mul and an add per expert) with the shared expert's sigmoid gate, the gate alone behind the sums ggml-cuda fuses, and the expert cache's id remap (#814). For comparisons. |
| `--gdn-scan-ggml` | off | CUDA, gated DeltaNet layers, prompt steps on compute capability 7.x GPUs (V100): ggml's `gated_delta_net` kernel instead of Legion's `gdn_scan` (the same bits, about 1.5x faster per call; other GPUs always run ggml's). For comparisons. |
| `--gdn-decode-ggml` | off | CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50), gated DeltaNet layers in decode steps (one token per sequence): ggml's `gated_delta_net` instead of Legion's op (the default there: `src/cuda/gdn.cu`'s decode kernel, the same bits; it moves the recurrent states, which is what these steps spend their time on, in 16-byte vectors through shared memory, #836). For comparisons. |
| `--gdn-replay-smem` | off | CUDA, compute capability 9.0+, gated DeltaNet layers in MTP verify steps that run the deferred commit first (`--gdn-commit-in-verify`, the default): the tokens' k, q, v, g and beta in shared memory, filled 8 tokens at a time (5 CTAs per SM), instead of each lane holding its tokens' q and k rows in registers (4 CTAs per SM). The same bits; about 3% faster in these steps on GB10 but flat end to end (#836). |
| `--gdn-commit-runs` | off | CUDA, gated DeltaNet layers, MTP replay commits: a commit graph per set of runs of slots keeping the same count (a delta-rule launch and a conv window copy per run and layer) instead of the default one kept graph for every slot (one launch of each per layer, each slot's kept count a graph input; the same kernel code per slot, so the same states, #836). For comparisons. |
| `--no-gdn-commit-in-verify` | on | CUDA, gated DeltaNet layers, MTP on one stage: each commit's delta rule runs inside the next verify step's kernel (#836 parts 14, 31, 38; `--gdn-commit-in-verify` is still accepted). A verify step stores its tokens' inputs packed for that replay alone, so the KV cache is as large as without it; the verify kernel advances each state by them, writes it and computes its rows from it, so a round reads and writes each state once less. Same kernel code per state, so the same bits; any other use of a waiting state (a prompt or decode step, a state copy, a reset) runs its commit first. |
| `--gdn-mma` | off | CUDA sm_80+, gated DeltaNet layers, prompt groups of 64+ tokens whose head counts ggml-cuda keeps on its per-token kernel (its chunked tensor-core kernel takes only 16 key and 16/32/48/64 value heads): a chunked kernel on them anyway (Legion's port of ggml-cuda's, `src/cuda/gdn_mma.cu`, the same bits). A tensor parallel rank's share is such a layer (Qwen3.8-Flash-Next: 8 key and 24 value heads on each of 2 ranks). Tensor parallel output moves (the chunked kernel's rounding). Decode and verify steps unchanged. For TP prompts on two DGX Sparks (#781: plain TP 8K prefill 2,433/2,482 -> 2,517/2,577 tok/s, 32K 2,493/2,503 -> 2,590/2,600; TP+MTP 32K 2,432/2,447 -> 2,510/2,513). |
| `--no-joined-matmuls` | off | CUDA, steps of up to 3 tokens: two matmuls per layer for DeltaNet alpha / beta (with `--shexp-gate`, also the MoE router / shared-expert gate) instead of one over both weights (the same bits). For comparisons. |
| `--joined-prompt-min-tokens N` | 1024 | CUDA GPUs with BF16 tensor cores, prompt steps of at least N tokens: BF16 weight pairs that lie back to back in memory (an NVFP4 checkpoint's DeltaNet alpha / beta, a shared expert's gate / up) as one cuBLAS GEMM instead of two (#836; no split-K at these sizes, so the same bits). 0 keeps two matmuls. |
| `--no-joined-prompt-z` | off | Those prompt steps: a DeltaNet layer's z gate projection on a GEMM of its own instead of one GEMM with alpha and beta (an NVFP4 checkpoint stores the three back to back in BF16; z's consumers read its rows strided in place). For comparisons. |
| `--no-joined-prompt-qkv` | off | Those prompt steps: a full-attention layer's query + gate, key and value projections as three GEMMs instead of one (an NVFP4 checkpoint stores the three back to back in BF16; the norm + RoPE and K/V append kernels read their rows strided in place, the same bits). For comparisons. |
| `--dense-mv-ggml` | off | CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50): BF16 matmuls of 2-16 tokens (weights of 16 MiB or more) on ggml's `mul_mat_f` instead of Legion's dense matmul op, which gives ggml's bits with the weights read on the read-only path that ggml's sm_90+ builds give up (#504). For comparisons. |
| `--dense-mv-small` | with the stage graphs | CUDA, compute capability 9.0+: the small BF16 matmuls of 2-16 tokens that Legion's dense matmul op reads faster than ggml's `mul_mat_f` (a shared expert's gate and up as one launch, the hyper-connection mixers' down projections) on the op too; ggml's bits either way (#860). On by default where the stage graphs are (`--stage-graphs`), which capture the op with the rest (#836): GB10, Qwen3.8-Flash-Next NVFP4 at 16 sequences, ~2.5 ms less GPU time per decode step of ~130 (nsys). The flag turns it on without them too, where each op splits ggml's CUDA graph (decode tok/s within noise there, #860). |
| `--no-dense-mv-small` | off | Keeps the small BF16 matmuls of `--dense-mv-small` on ggml's `mul_mat_f` with the stage graphs on: the same bits, a little slower decode. For comparisons. |
| `--no-dense-mv-ab` | off | Keeps a GDN layer's alpha and beta projections (BF16, 48 rows each: NVFP4 checkpoints of Qwen3.8-Flash-Next) on ggml's cuBLAS GEMM in decode steps of 9-16 tokens. By default (CUDA, compute capability 9.0+) they ride along in the layer's qkv+z dense op as chain weights computed in cuBLAS's order: the same bits, without the two ~18 us latency-bound cuBLAS launches per layer (#836). For comparisons. |
| `--no-router-mv` | off | Keeps a MoE layer's router in steps of 1-3 tokens (decode at 1-3 sequences, MTP verify steps at one sequence, draft passes) as three launches: ggml's F32 matmul, then the top-k and the weights ops. By default (CUDA, compute capability 9.0+) one kernel computes the logits as ggml's `mul_mat_vec_f` does and its last CTA selects the experts as the two ops do: the same bits, two launches fewer per MoE layer; on a GB10 the router's weight rows load while the kernel ahead still runs (#836). For comparisons. |
| `--no-router-join` | off | Runs a MoE layer's expert selection as two CUDA launches, the top-k and then the weights. By default (CUDA, routers outside `--no-router-mv`'s one-op router, without `--nvfp4-once`'s routing plan or `--decode-fuse`) one launch picks the experts and makes their weights from the logits it picked: the same bits, one launch fewer per MoE layer (#814). For comparisons. |
| `--no-router-bf16` | off | Keeps reading the MoE routers' F32 weights in decode and verify steps (1-16 tokens). By default (CUDA, compute capability 9.0+) a router whose F32 values are all exact in BF16 (a BF16 checkpoint's router widened to F32) gets a BF16 copy that the 1-3 token router kernel and the 4-16 token F32 matmul read instead, widening each value back to its F32 bits: the same bits from half the bytes (~126 MB more GPU memory for 49 routers of 512 x 2560). Routers not exact in BF16 keep the F32 reads (#836). For comparisons, or to save that memory. |
| `--no-ple-early-reads` | off | Qwen3.8-Flash-Next with the n-gram table on the SSD (direct reads), verify steps that leave their n-gram rows to their compute: issues the rows' reads only when the step writes them, after its first layers are launched. By default the step's input fill issues them, so they run through the launch and the first layers, and the write collects them. The same bytes either way (#836). For comparisons. |
| `--dense-mv-f32` | off | CUDA, compute capability 9.0+: F32 matmuls of 4-16 tokens (the MoE router of a decode step) on Legion's dense matmul op instead of ggml's `mul_mat_f<float>`: ggml's bits (TF32 products in ggml's order) with the weights' loads in flight. GB10, Qwen3.8-Flash-Next NVFP4 at 16 sequences: the router 1.73 ms per step against ggml's 1.87 (nsys), decode tok/s within noise, so off by default (#836). For tuning. |
| `--no-cluster-select` | off | CUDA, compute capability 9.0+ (GB10, Hopper, RTX 50): decode-sized selections (the QSA indexer's top-k for up to 4 rows of 8,192+ scores, about 32K+ tokens of context, or 8 rows of 16,384+, the greedy argmax over the vocabulary for up to 16 rows) on one CTA per row instead of a thread-block cluster of 8 per row (the same outputs, ties included; #554). For comparisons. |
| `--fp8-small-batch-gemv` | off | CUDA, compute capability 8.0+, channel-wise FP8 W8A16 weights (compressed-tensors, such as unsloth's Qwen3.8-27B-NVFP4: attention, linear attention, the last MLPs, the lm_head): matmuls of 5-9 tokens (a `--spec mtp` verify step of 4-8 drafts, a decode step of 5-9 sequences) on the GEMV that reads each weight once for all tokens, instead of the BF16 tensor-core GEMM. Every token then gets the bits it gets alone in a one-token decode step, whatever the verify depth, so greedy text can differ from the default's where such steps occur. GB10, 27B unsloth: lm_head (5120 -> 248320) at 8 tokens 6.60 -> 5.52 ms; verify step at a fixed depth (`--deterministic`, code prompt) 5 drafts 128.5 -> 116.2 ms, 7 drafts 132.6 -> 123.3 ms, 8 drafts 134.7 -> 128.5 ms (#520). Opt-in since it changes those steps' bits. |
| `--fp8-sliced-k` | off | CUDA, compute capability 8.0+, FP8 W8A16 weights (as `--fp8-small-batch-gemv`): matmuls of 5-64 tokens (speculative verify steps, decode steps of that many sequences, short prompt chunks) on a sliced-K tensor-core GEMM after NInfer's (16-32 weight rows per CTA, K split over its warps), which streams the weights near the GEMV's bandwidth, instead of the prefill-shaped GEMMs. With `--fp8-small-batch-gemv` too, the GEMV keeps 5-9 tokens. The sums round in another order, so greedy text can differ from the default's where such steps occur. GB10, 27B unsloth: 5120 -> 17408 at 16 tokens 448 -> 385 us, at 64 519 -> 421 us, lm_head at 36 tokens 6.82 -> 5.59 ms; `--spec mtp --spec-fixed` (8 drafts) verify step 147.6 -> 133.7 ms at C=1 (20.5 -> 24.2 tok/s), 60.4 -> 54.2 ms per sequence at C=4 (42.7 -> 47.0 tok/s); default adaptive MTP at C=4 (with `--nvfp4-wide-mv`) 53.6 -> 58.8 tok/s. Quality probe: 384 of 384 greedy tokens the default's, top-5 KL 1.3e-5 (#664). |
| `--nvfp4-expert-mv-max <PAIRS>` | 32 | CUDA, NVFP4 routed experts on Legion's W4A16 kernels: the most tokens per expert (pairs over the experts a step touches) that still run on the grouped mv kernel; steps with more take the grouped GEMM. Lower values move short prompt steps onto the GEMM, whose sums round in another order, so greedy text can differ from the default's where such steps occur; 32 (the default) keeps the default bits. A DGX Spark (GB10), Qwen3.8-Flash-Next NVFP4 with MTP, deterministic, one request each, with 8: TTFT at 1,080 prompt tokens 1.11 -> 0.92 s (970 -> 1,170 tok/s), at 572 0.67 -> 0.64 s, at 312 unchanged, 8K prefill and decode unchanged (#836). |
| `--nvfp4-expert-mv-tile <NT>` | 4 | CUDA, NVFP4 routed experts: the grouped mv kernel's widest token tile, 8 x NT tokens of one expert per CTA (1, 2 or 4). Each token's sums are the same at any tile, so greedy text keeps the default's bits. On GB10 4 is the best or ties at 256-1,536 prompt tokens, 2 is up to 20% slower on the down projection at 1,024+ and 1 up to 94% slower, so this is for tuning other GPUs (#836). |
| `--nvfp4-expert-mv-split <MODE>` | `step` | CUDA, NVFP4 routed experts: how `--nvfp4-expert-mv-max`'s cutoff applies. `step`: a step's pairs per used expert put all its experts on one kernel. `expert`: prompt steps of 256+ tokens with up to 32 pairs per expert run the experts with more pairs than the cutoff on the grouped GEMM and the rest on the grouped mv kernel (two launches on one plan). `down` / `expert-down`: `step` / `expert` for the down projections only. Any mode but `step` can change greedy text where an expert moves to the GEMM; decode and verify steps keep the default kernels. A DGX Spark (GB10), Qwen3.8-Flash-Next NVFP4 with MTP, server prefill time, with `--nvfp4-expert-mv-max 8`: `expert` 1,080 tokens 0.74-0.82 -> 0.62 s (`step` 0.65-0.68 s), 312 and 572 tokens 10-12% faster in one of two runs (the other run had a transient slowdown that also hit its 8K warm-up), 8K and decode unchanged; `down` and `expert-down` not measured yet (#836). |
| `--nvfp4-expert-rows-bf16` | off | CUDA, NVFP4 routed experts on Legion's W4A16 kernels, steps where every sequence is in its prompt with more than 64 tokens in all: the down projection stores each (token, expert) row as BF16 instead of F32 and the weighted sum reads it back (on qwen4exp the hyper-connection norm the MoE tail folds into reads the BF16 rows itself). Half the bytes of those rows. On a DGX Spark, Qwen3.8-Flash-Next NVFP4 prefill is a few percent faster; elsewhere (no fold) the BF16 rows are cast back to F32 and it does not pay. Each row is rounded to BF16 before the sum, so output moves (opt-in). Decode and verify steps, and steps that batch a prompt with them, keep F32 rows. Not on the `--tp-moe-ep` skip path. |
| `--nvfp4-wide-mv` | off | CUDA, compute capability 8.0+, dense NVFP4 weights on Legion's W4A16 kernels (`--nvfp4-mm auto` or `legion`; e.g. the MLPs of unsloth's Qwen3.8-27B-NVFP4): matmuls and fused SwiGLUs of 33-64 tokens (verify steps of deep drafts at 4+ users, DFlash, short prompt chunks) on the decode mv kernel, 64 tokens per CTA (16 rows up to 40 tokens, 32 above), instead of the GEMM, whose tiles read the weights at about half the bandwidth there. Each token gets the bits the mv kernel gives it at 1-32 tokens, so greedy text can differ from the default's where such steps occur. GB10, 27B unsloth: SwiGLU 5120 -> 2 x 17408 at 33 tokens 870 -> 548 us, at 64 870 -> 570 us; 17408 -> 5120 at 64 413 -> 397 us; `--spec mtp --spec-fixed` at C=4 verify step 60.4 -> 55.9 ms per sequence. MTP's adaptive depth picks 2-3 drafts at C=4 (12-16 tokens), where it does not apply (#664). |
| `--paged-verify-split` | off | CUDA, compute capability 8.0+ (ignored with a warning elsewhere), pools without a host tier: the attention of groups of 2-16 tokens per sequence (speculative verify steps, short prompt chunks) split over the key blocks as decode steps are (flash-decoding: as many splits as a decode step of as many sequences, up to 32, then a combine kernel), instead of one CTA per query tile and kv head walking the whole context. The sums take another order, so greedy text can differ from the default's where such steps occur. GB10, Qwen3.8-27B attention (24 heads over 4), one sequence, per layer: 8 tokens over 64K keys 3.07 -> 1.19 ms, over 16K 0.78 -> 0.34 ms, over 128 keys 15 -> 18 us (the combine launch); DFlash 2 + the #689 stack, C=1 decode at 64K depth 21.7 -> 27.0 tok/s per request, at 16K 31.0 -> 32.5, depth 0 unchanged (#697). |
| `--paged-verify-split-wmma` | off | CUDA, compute capability 7.0 / 7.5 (V100, Turing): the attention of groups of 2-16 tokens per sequence (speculative verify steps, short prompt chunks) split over the key blocks across the GPU, at up to 32 CTAs of the WMMA prefill kernel (one per query tile and kv head, each walking the whole context twice; #1211, `src/cuda/sm70/paged_vsplit.cuh`). Every value the WMMA kernel computes for a row is computed in the same order, so the attention's output bits are the same (with `--spec none` the text is too). With `--spec`, cheaper verify steps keep the adaptive draft length up (Qwen3.8-27B, MTP k=3, 32K context: 17 verify steps for 256 tokens without it, 112 with it; decode 20.0 to 34.9 tok/s), so tokens are computed in other steps and greedy text can differ. One V100, Qwen3.8-27B's 4-token verify step, one sequence: 26.6 ms to 2.4 ms per layer at 64K keys, 13.4 to 1.3 at 32K; at 128 keys with 4 sequences 0.10 to 0.13 ms (more launches), so it stays off by default. |
| `--tu-attn-ffma` | off | CUDA, Turing without tensor cores (GTX 16xx, MX450, MX550; compute capability 7.5), F16 KV caches: prompt attention through a one-pass kernel on FP32 FMAs instead of the WMMA kernel, whose matrix instructions these GPUs emulate. GTX 1660 Ti, Qwen3.5-4B UD-Q4_K_XL: attention 2.73 -> 0.82 s over the first five 1K steps of a 6K prompt, 8K prefill 536 -> 770 tok/s, decode unchanged. Opt-in, **moves output**: Q and P are not rounded to F16, one pass, another summation order; in the measured chats (this GPU, Qwen3.5-4B, Gemma 4 E2B and Fara1.5-4B) greedy text kept the default's first tokens and parted ways after 21 to 247 tokens, and any prompt can part earlier, from its first token on a near tie. |
| `--shexp-gate` | off | CUDA, MoE layers: the shared expert's gate (sigmoid, scaling, sum) as one op, and on steps of up to 3 tokens the router's and the gate's logits from one matmul (one matmul and two kernels fewer per layer). Each op gives ggml's bits, but the new graph layout can flip ggml-cuda's layout-dependent MoE reduction fusion, so MTP outputs can move unless `--moe-combine` is on too (#410). |
| `--moe-pair-out` | `f32` | `f16` / `bf16`: opt-in, **moves output**: the fused MoE op keeps each (token, expert) output in 16 bits between the down GEMM and the per-token sum, halving the op's largest memory stream. GB10, Qwen3.8-Flash-Next UD-Q4_K_XL: prefill +2-2.5%; teacher-forced KL against F32 on 16K tokens of text 4.2e-3 (F16) / 3.4e-3 (BF16), against 5.2e-3 between the default op and ggml's MMQ path and up to 1.6e-3 between two F32 passes; greedy chats diverge after a few hundred characters. |
| `--no-moe-raw-staging` | off | The fused MoE op on sm_80+: the GEMM that loads each K step's weight bytes straight from global memory instead of copying row blocks into shared memory with `cp.async` first (the default's `gemm_raw_kernel` for Q4_K, Q5_1, IQ2_S and IQ2_XXS weights, and on the GB10 the fragment kernels for Q4_K gate/up and Q5_1 down). The same output bits. For comparisons; was `LEGION_MOE_RAW=0`. |
| `--moe-down-sum` | off | The fused MoE op on the GB10, prompt steps whose down GEMM runs the fragment kernel (Q5_1 / Q8_0 down, F32 pair outputs): the down GEMM makes the per-token sums itself instead of the separate sum kernel (#826: the block that finishes a token's last pair sums its pairs in slot order). The same output bits; prefill within noise (8K -0.2%, 32K +2.2%), so opt-in. |
| `--expert-mv-vram` | `0` | Decode expert matvec: VRAM cache slots read through tiles (0), by an mmvq-style kernel in place (1), or that kernel for gate/up only (2). 1 and 2 are not byte-identical and may not reproduce from run to run. For tuning. |
| `--expert-mv-prefetch` | off | Decode expert matvec: VRAM experts (cache and staging slots) read in place inside the tile kernel, with each lane's weight loads issued before its first dot product (after Strata-AC922's S28). Host experts, experts several tokens of a step share, Q8_0 weights, and steps of more than 2 tokens (speculative verify) keep the tiles. The same bits as the default. About +1.6% plain decode on 4 V100s; no gain with MTP. |
| `--expert-mv-link-ctas` | `0` | Decode expert matvec over an expert cache: that many CTAs of the tile kernel, dispatched ahead of the rest, take the cache misses read in place over the host link; the other CTAs take the cached experts only. Experimental (#710): 64 on 2 V100s with MTP decoded greedy chats about 5% slower, and their text differed from the default's (each tile is computed by the same code; MTP's adaptive depth follows timing, the likely cause). For tuning. |
| `--expert-mv-device` | off | Decode expert matvec: experts in GPU memory take the SwiGLU op in verify steps of 2-8 tokens. Same bits per layer; generated text can differ from the first multi-token verify step (the new graph layout can flip ggml-cuda's layout-dependent fusions, #410 / #472), so it is not byte-identical end to end. GB10, Qwen3.8-Flash-Next: about 2.7% shorter verify step at fixed MTP k=2, a tie at k=4 and with the adaptive depth. Opt-in; see [tuning-integrated-gpus.md](tuning-integrated-gpus.md). |
| `--expert-mv-max-tokens` | `8` | Decode expert matvec: the most tokens of a step it takes (1-64). For A/B runs. |
| `--glm-verify-mv-max-tokens` | `64` | GLM-5.3-Flash over an expert cache (`--experts cache`): the most tokens of a speculative verify step whose routed experts run through the expert matvec (1-64, never below `--expert-mv-max-tokens`). Bigger verify steps, and prompt steps past `--expert-mv-max-tokens`, run the fused MoE op, which reads the uncached experts in place over the host link: on an AC922 ~2 s per 12-16 row verify step (MTP `--spec-fixed` k = 2-3 at 4 sequences) against ~90 ms through the matvec (#569). The matvec gives each token the same bits at any step size. `8` is the behaviour before #569, for A/B runs. |
| `--expert-mv-split-glu` | off | Decode expert matvec: gate and up as two ops instead of the SwiGLU op. For comparisons. |
| `--expert-mv-dot` | `auto` | Decode expert matvec, tile kernel: `auto` decodes each weight unit once for all users of an expert with 3 or more in the step (else one vec_dot per user), `per-user` / `once` / `sub16` (16 lanes per row, two rows per warp) force one. All give the same bits. For comparisons. |
| `--expert-peer-fetch`, `--no-expert-peer-fetch` | automatic | Decode over an expert cache (`--experts cache`) on a layer split: while one GPU runs its layers, its idle NVLink peer copies half of each MoE layer's missed experts from host memory into the running GPU's VRAM over its own host link (after Strata-AC922's `partner_fetch`, #252). It stages into a ~25 MB VRAM buffer per GPU outside the step graph, so the graphs keep the plain ops' memory layout: the same output bits, and `--deterministic` text byte-identical with it on or off. Prompt steps and steps where both GPUs work run as without it. Automatic (#522): on for a layer split over exactly the two GPUs of one NVLink pair on an AC922 (GPUs 0, 1 or 2, 3; POWER9 + V100 with ATS), off elsewhere. `--expert-peer-fetch` forces it on (e.g. on four GPUs), `--no-expert-peer-fetch` off. Measured on Qwen3.8-Flash-Next UD-Q4_K_XL, MTP k=2, temperature 0, against the old default (off): GPUs 2,3 decode C=1 52.6 -> 58.6, C=4 70.0 -> 80.8 (all 5 runs above all 5 off runs), C=8 70.9 -> 80.5, C=16 64.7 -> 65.5 tok/s, chats 57.5 -> 64.2, same-text `--deterministic` chats 57-58 -> 63-64, prefill 8K/32K/123K a tie (123K 3917 -> 3927); four GPUs with it forced on: C=1 a tie (63.0 -> 63.2, same-text chats a tie), C=4/C=8 medians 96.4 -> 94.7 and 92.3 -> 87.1 (inside the spread, not shown to be a tie), 123K 7002 -> 7038, so it stays off there. |
| `--qsa-scores-tc` | off | qwen4exp prompt steps on Volta: the sparse-attention indexer's scores on the tensor cores (FP32-level, may shift the selected pools). |
| `--qsa-attn-chain` | off | qwen4exp prompt steps, Volta kernel: accumulate the hi / lo products in chains across K steps, 17% faster at 32K; changes the rounding. sm_80+ GPUs run the `mma.m16n8k16` kernel, which has no chain; the server says so once. |
| `--hc-mix-ggml` | off | qwen4exp, CUDA: small steps' hyper-connection mixer from ggml's nodes instead of Legion's op. For comparisons. |
| `--hc-mix-bf16` | off | qwen4exp, CUDA: steps of up to 8 tokens also run Legion's mixer op when the mixer's down and up matrices are BF16 (Qwen3.8-Flash-Next-NVFP4), instead of ggml's dozen nodes. Different summation order, so greedy output moves by rounding (opt-in). On a GB10 the op loads its weights before the PDL wait: C=1 decode 19.1 -> 19.7 tok/s (+3.1%), prefill unchanged; Q8_0 mixers (the GGUF) take the op either way (#748). |
| `--hc-post-norm-ggml` | off | qwen4exp, CUDA: ggml's hyper-connection combine and a separate norm instead of the fused op. For comparisons. |
| `--hc-moe-fold` | on for an integrated GPU (GB10), else off | qwen4exp prompt steps of 256+ tokens on an sm_80+ GPU with BF16 mixers (the NVFP4 checkpoints), CUDA: the next combine-and-norm makes each MoE layer's tail (the routed experts' weighted sum and the shared expert's gate) from the tail's inputs, rounded as the tail's kernels round it, and the tail's kernels go (#836: the same bits, the layer output no longer written and read back). Prompt graphs are planned twice (as built, to read which sums ggml-cuda fuses, then folded). On a DGX Spark (GB10), Qwen3.8-Flash-Next-NVFP4 with its MTP head: bit-identical (46 of 48 tails folded), the fold kernel 4.3 ms per 4096-token chunk against 4.8 ms for the three kernels it replaces (all near the memory roof), so server prefill time drops about 1.2% at 4K-8K (3.97 -> 3.92 s at 8K) and 0.2% at 32K. The prompt compute buffer stays the same size. An earlier 5-6% slowdown was measured before the prompt compute buffers were reserved at load and does not reproduce on later builds. No effect with `--node-fuse-ggml` or on decode steps. |
| `--no-hc-moe-fold` | off | Keeps the MoE tails as their own kernels (see `--hc-moe-fold`). For comparisons. |
| `--hc-fuse` | off | qwen4exp prompt steps on a V100 (7.0) or an sm_80+ GPU (no effect elsewhere), experts on the fused MoE op: the gated shared expert is added in the op's per-token sum (ggml's sigmoid, mul and add bits, as on the GB10, #526), and the next combine-and-norm makes that sum from the op's per-expert rows itself (`hc_post_norm_moe_kernel`) instead of the op's sum kernel writing it and the norm reading it back (#774). The same bits per op; the graph layout changes, which moves greedy output through ggml-cuda's layout-dependent fusions (#472), so opt-in; identical text under `--deterministic --stable-fusions`. Steps under 64 tokens keep the default graph. AC922 (4 GPUs) record flags (6 servers): 123K 15.82 -> 15.51 s; 8K-64K and decode unchanged. On sm_80+ GPUs (GB10, #826) too, there also in the norm's rebuilt form with the quantized `xq`: greedy text unchanged on a Spark (chats and det C=1/C=4 hashes), the fused kernel 4.26 ms per 4096-token call against 2.27 + 2.08 ms for the norm and the sum kernel apart (nsys, 32K), prefill 8K -1%, 32K flat, 64K +3% (one request each, noise level), so opt-in there as well. |
| `--hc-xn-f32` | off | qwen4exp prompt steps of 64+ tokens on a V100 (7.0) with Q8_0 mixer down weights: the down projection converts its inputs inside ggml's matmul again instead of taking the F16 `xn` and F16 down weight the fused combine-and-norm writes. Same bits either way (ggml's own conversions, the same cuBLAS GEMM); for comparisons (#377). |
| `--hc-quant-ggml` | off | qwen4exp prompt steps of 256+ tokens on compute capability 12.x (GB10) with Q8_0 mixer down weights: ggml-cuda quantizes the down projection's input itself again instead of taking the blocks the fused combine-and-norm writes for its MMQ. Same bytes, so the same output either way; for comparisons (#363). |
| `--hc-up-mix-ggml` | off | qwen4exp prompt steps on compute capability 12.x (GB10) with Q8_0 mixer up weights: ggml's up matmul (MMQ, writing the gate) and gated `hc_pre` (reading it back) instead of Legion's one kernel (`src/cuda/hc_up_mix.cu`) that keeps the gate in registers. The kernel adds MMQ's sums in MMQ's order and runs only for step sizes where ggml's MMQ computes whole tiles, so the same output either way; for comparisons (#526). |
| `--no-hc-up-mix-bf16` | off | qwen4exp prompt steps of 256+ tokens on compute capability 12.x (GB10) with BF16 mixer up weights (the NVFP4 checkpoints): ggml's cuBLAS BF16 GEMM (writing the gate) and gated `hc_pre` instead of Legion's BF16 MMA kernel (`src/cuda/sm80/hc_up_mix_bf16.cuh`, the default), which runs the mixer's up matmul and gated mix as one op that keeps the gate in registers. The kernel adds the GEMM's m16n8k16 products in K order, and its two nodes (`lo` as BF16 in the gate's place, then the mix) have the sizes and places of ggml's two, so ggml's allocator lays the graph out as for ggml's ops. A first one-node form moved the C=4 det hash through ggml-cuda's layout-dependent MoE fusions (with `--stable-fusions` the hashes matched). On one GB10 (Qwen3.8-Flash-Next-NVFP4 + MTP, `--deterministic`) every det hash (C=1, C=4, depth 16K) and the chats equal ggml's ops'; prefill 8K/32K 1,729/1,811 -> 1,844/1,889 tok/s (+6.6%/+4.3%), decode unchanged (#836). For comparisons; no effect with `--hc-up-mix-ggml`, which keeps ggml's ops for every up-mix. |
| `--hc-xn-written` | off | qwen4exp prompt steps on compute capability 12.x (GB10) where that up-mix kernel runs, and of 64+ tokens on a V100 where the F16 mix (`--hc-mix-f16`) runs: the combine-and-norm writes the normed streams (4 x n_embd F32 values a token) and the kernel reads them, instead of the norm writing each stream's scale and the kernel rebuilding the normed values from the streams as the norm computes them (the same two rounded multiplies, so the same output); for comparisons (#526). With BF16 mixers (the NVFP4 checkpoints) the norm then still writes the BF16 normed values for the down projection, but not the F32 ones: on a DGX Spark (GB10) the norm kernel takes 2.26 ms per 4096-token call instead of 3.12 ms, and the up-mix kernel is no slower (#836). |
| `--hc-streams-bf16` | off | qwen4exp (4 x 2560 streams) prompt steps on compute capability 12.x (GB10) where that up-mix kernel runs with the rebuilt norm: opt-in, **moves output**: the residual streams between those mixers are kept in BF16, not F32 (each new stream value rounded to BF16 before it is normed, quantized and injected), so the combine-and-norm reads and writes them and the up-mix kernel reads them at half the bytes (#826). Every other reader (decode steps, the PLE layer, the output head, `--tp-seq-par`'s exchange) gets them converted back to F32. No effect with `--hc-xn-written` or `--hc-up-mix-ggml`. Measured on one GB10 (Qwen3.8-Flash-Next UD-Q4_K_XL, 32K prompt, nsys): the up-mix kernel 912 -> 667 ms, but the BF16 combine-and-norm kernel 1,727 -> 1,839 ms (it is bound by loads in flight, not bytes), so prefill is unchanged within noise (8K/32K/64K 1,665/1,902/1,883 -> 1,692/1,914/1,895 tok/s); greedy logprobs against the default: KL 5.4e-6, top-1 1096/1096. |
| `--hc-bf16` | off | qwen4exp (4 x 2560 streams) passes where every sequence is in its prompt (steps of 256+ tokens), on compute capability 8.0+ with BF16 mixer weights (the NVFP4 checkpoints) where the BF16 up-mix kernel runs with the rebuilt norm: opt-in, **moves output**: the residual streams between those mixers are kept in BF16, not F32 (each new stream value rounded to BF16 before it is normed, rounded into the down projection's input and injected; the math inside each kernel stays F32), so the combine-and-norm (with the MoE tail folded into it) and the up-mix kernel read and write half the stream bytes (#1183). Decode, verify and mixed steps keep F32 streams; every other reader (the PLE layer, the output head, the MTP capture) gets them converted back to F32. No effect with `--hc-xn-written`, `--no-hc-up-mix-bf16` or when `--tp-seq-par` splits the step. Combines with `--nvfp4-expert-rows-bf16`. |
| `--hc-mix-f16` | off | qwen4exp prompt steps of 64+ tokens on a V100 (7.0): the hyper-connection mix as Legion's op (`src/cuda/hc_mix16.cu`) that also writes the block input as F16, which its Q8_0 projections' cuBLAS GEMMs then take with an F16 copy of their weight made by ggml's own converter, instead of ggml's gated `hc_pre` and ggml-cuda converting the input to F16 again inside each projection. The same GEMMs on the same F16 values (greedy text equal under `--deterministic --stable-fusions`), but the smaller graph moves where ggml's allocator places tensors, which can flip ggml-cuda's layout-gated MoE fusions and change greedy output, so it is opt-in (#774). |
| `--gate-quant-ggml` | off | Prompt steps of 256+ tokens on compute capability 12.x (GB10) with Q8_0 output projections behind a sigmoid gate (gated attention's `attn_output`, a sigmoid-gated DeltaNet layer's `ssm_out`, as in qwen4exp): ggml's fused sigmoid + mul (writing the gated values as F32) and the matmul's own quantization pass (reading them back) instead of Legion's one op (`src/cuda/gate_q8.cu`) that gates and quantizes for ggml's MMQ. The same arithmetic, so the same output either way; for comparisons (#526). `--hc-quant-ggml` turns it off too. |
| `--gate-norm-ggml` | off | Where that op runs for a sigmoid-gated DeltaNet layer's `ssm_out` (qwen4exp): ggml's output RMS norm and its weight (`rms_norm` + `mul`, writing the normed values as F32) before the op, instead of the op norming each 128-value row itself, summing the squares in ggml-cuda's order. The same output either way; for comparisons (#526). |
| `--shared-quant-ggml` | off | Prompt steps of 256+ tokens on compute capability 12.x (GB10): every Q8_0 projection of a layer input that several read (a DeltaNet layer's `attn_qkv` and `attn_gate`, an attention layer's `attn_q`, `attn_k` and `attn_v`, the shared expert's gate and up) quantizes it in its own pass, as ggml's `mul_mat` does, instead of once for all of them with ggml's own quantization kernel and ggml's MMQ on those blocks. The same bytes, so the same output either way; for comparisons (#526). `--hc-quant-ggml` turns it off too. |
| `--shared-quant-small-ggml` | off | The same for steps of 9-255 tokens (multi-sequence decode, MTP verify) inside stage graphs on compute capability 12.x (GB10): every such projection quantizes its input on its own, instead of the first quantizing it once into a scratch buffer with ggml's kernel and each running ggml's MMQ on those blocks (the default since #858, ~100 fewer quantize launches a C=16 verify step). The same bytes and kernels, so the same output either way; for comparisons (#836, #858). Projections the Q8_0 MMQ op runs keep its own quantization. |
| `--q8-mmq-ggml` | off | Q8_0 matmuls of 9-255 tokens (multi-sequence decode, MTP verify) inside stage graphs on compute capability 12.x (GB10): all on ggml's MMQ (quantize, tiles, stream-k fixup), instead of the shapes Legion's op runs faster on it (the default since #858: ggml's tile code on ggml's grid, split tiles added in ggml's fixup order, the input quantized once for the matmuls that read it; today a 10240-value weight of 320 rows at 9-16 tokens, `legion q8-mmq-check`). The same output either way; for comparisons (#836, #858). |
| `--shexp-sum-ggml` | off | Compute capability 12.x (GB10), steps on the fused MoE op (`--moe-kernel`), models with a sigmoid-gated shared expert (qwen4exp, Qwen3.5 MoE): ggml's sigmoid, mul and add after the op (each a pass over the step's hidden states) instead of the op's per-token sum adding the shared expert, with ggml's sigmoid and the product and the sum rounded apart as ggml's two ops round them. The same output either way; for comparisons (#526). Not with `--shexp-gate`, whose op it leaves as it is. |
| `--hc-norm-fma` | off | qwen4exp prompt steps on compute capability 7.x (V100, Turing; no effect elsewhere): opt-in, **moves output**: the separate hyper-connection norm (layer 0's mixer, the MTP head, every layer with `--hc-post-norm-ggml`) adds each square with one FMA as the fused combine-and-norm does there, so the two round alike and `--hc-post-norm-ggml` gives the fused op's bits (#409, #376). By default nvcc adds every fourth square with a separate multiply and add, an ulp apart at times. Measured on a V100 (Qwen3.8-Flash-Next UD-Q4_K_XL, 16K tokens teacher-forced, 512 scored positions): KL 3.4e-3 mean against the default, top-1 99.8%; `--hc-post-norm-ggml` alone is KL 2.7e-3 from the default, and with this flag on it changes no row (512 of 512 bit-identical). |
| `--ple-ops` | off | qwen4exp prompt steps on a GPU: the per-layer n-gram embedding after its two matmuls as Legion's two ops (`src/cuda/ple.cu`, #439) instead of ggml's chain of about thirty nodes: fewer passes over the `[e · hc, n]` streams on the first pipeline stage (~10 ms per 4,096-token chunk on a V100). Each op gives the chain's bits, but the smaller graph moves where ggml's allocator places tensors, and ggml-cuda fuses an MoE layer's weighted expert sum (with FMA adds) only where its output does not overlap the router weights: a layer that flips changes greedy output. |
| `--no-after-overlap` | off | With an expert cache: the cache's work after a pass runs before the next pass, with the GPUs idle, instead of beside GPU work (with MTP the next MTP pass; without MTP, over several GPUs, each stage's share beside the next stage's compute). For comparisons. |
| `--no-stage-launch-ahead` | off | A decode or verify step over several GPUs: each next stage's residual copy and graph are queued after the host waited for the stage before it, instead of the default: queued while that stage computes, the next GPU waiting on an event (`src/model/core/exec/launch_ahead.rs`, #814). The same graphs on the same inputs in the same order, so the same output either way; for comparisons. |
| `--no-stage-graph-words-skip` | off | CUDA stage graphs: every run of a kept step or MTP graph compares its words (each node's op, shape, strides, parameters and data pointers, and its sources') with its last run's, instead of the default: a kept graph that ran unchanged skips the compare (`src/ggml/stage_graph/kept_words.rs`, #814; ~40-45 µs of host time per graph on a Qwen3.8-Flash-Next V100 stage, on the critical path at a step's first stage and each MTP draft pass), since a kept graph is never written after its build; every 64th run in a row compares anyway. Off also turns off the rest of a kept graph's host shortcuts: the late input's first reader found once per kept graph, and an unchanged run's cache entry kept with the graph instead of the shared cache (no lock, lookups or clock reads). The same kernels in the same order, so the same output either way; for comparisons. |
| `--stage-graph-spare-execs N` | 8 | CUDA stage graphs: big captures (256+ CUDA graph nodes) dropped from the store stay as up to N spare executable graphs per GPU (`src/ggml/stage_graph/spare.rs`, #836), and a new capture of the same topology updates one (`cudaGraphExecUpdate`) instead of instantiating its graph: each new MTP verify shape as the contexts grow (~18 ms of instantiation each on a DGX Spark, ~11 per 256 tokens at 4 parallel requests). The updated graph runs the new capture's kernels with its arguments, so the same output; a refused update falls back to an instantiation. `spares_updated` / `spare_updates_refused` in `--stage-graph-stats`. 0 turns it off. |
| `--stage-graph-exec-steal-ms MS` | 1000 | CUDA stage graphs: a new big capture with no spare of its topology takes over the executable graph of a kept capture of the same topology not launched for MS milliseconds and updates it (`execs_taken_over` in `--stage-graph-stats`). The kept graphs of the verify shapes a context grew out of hold their captures until evicted, so early on there are no spares; a taken-over capture is captured again if its graph runs again. The same output. 0: never. |
| `--stage-graph-hold-kept` | off | CUDA stage graphs: every run of a kept step or MTP graph keeps its captures with the graph for as long as it is kept (`src/ggml/stage_graph/kept_words.rs`, #858), instead of the default, where a graph's first runs put its captures into the shared store (`--stage-graph-cache`, 64 sets a GPU, sets unused for 10 s dropped) and only a run on the unchanged path keeps them with the graph. With hundreds of kept graphs (the MTP take-in shapes at 4+ parallel requests) graphs that stayed kept were captured again (`captures_lost` in `--stage-graph-stats`). The same kernels in the same order, so the same output; costs the kept graphs' captures' memory. |
| `--no-input-pack` | off | CUDA: one `cudaMemcpyAsync` per input of a decode or verify step and of an MTP draft pass (tokens, positions, slot and block tables, n-gram rows, ...), instead of the default: the host writes them all into the stage's pinned buffer and one kernel (`src/cuda/input_pack.cu`) copies them into their tensors ahead of the graph (#814). The same bytes and the same graphs, so the same output either way; for comparisons. |
| `--mtp-ahead` | off | `--spec mtp`: opt-in, each draft pass queues the next pass's inputs while it computes (its picks and output residual copied on the GPU, the rest from pinned memory, #827) wherever nothing else runs between a round's passes (no `--experts cache`, one machine, a pass within the gather budget); the next pass then starts its graph without filling anything if every row went on. Off, the host builds them after the pass with the GPU idle. The same output either way, at any number of parallel requests (#858: the take-in pass keeps its sort by rows); measured no faster on a GB10 (#881) and, with `--mtp-pipe-rows 16`, 2-8% slower at 1, 4 and 16 requests (#858). |
| `--mtp-pipe-rows N` | `0` | `--spec mtp` with `--mtp-ahead`: opt-in, a draft pass of at most N rows is queued while the pass before it still computes, before the host has read that pass's picks (#827), so the GPU does not wait for the host between passes. On CUDA 12.3+ the queued pass is the body of a conditional graph node: a one-thread kernel checks that every row of the pass before reached `--spec-p-min` and the GPU skips the pass otherwise (a few µs instead of a whole pass). Where CUDA refuses that graph, the pass runs and the host drops it. The same output either way. |
| `--glm-mla-tc` | off | GLM-5.3-Flash, CUDA sm_80+ (GB10), prompt steps of 64+ tokens: opt-in, **moves output**: MLA attention on tensor cores (F16 queries and softmax weights, F32 sums). Decode unchanged. Part of the fast prefill set below. |
| `--glm-expert-q8` | off | GLM-5.3-Flash, CUDA sm_80+ (GB10), prompt steps of 17+ tokens on IQ2_XXS gate/up and IQ2_S or IQ3_XXS down experts (Maya-S; IQ3_XXS down since #594): opt-in, **moves output**: the routed experts' GEMMs on int8 tensor cores, llama.cpp MMQ-style, with the expert inputs and the SwiGLU rows quantized to 8 bits with an F32 scale per 32 values (#603). Decode unchanged. One GB10, Maya-S: prefill +7% (4K-32K tokens) to +11% (1K); against the default over 3 raw ~4-5K-token prompts, top-5 KL mean 9.3e-4 (max 4.8e-2) on the 107 positions with the same context, top-1 104/107. Part of the fast prefill set below. |
| `--load-threads` | `4` | Threads that read the model files (1-8). |
| `--load-parallel-device` | off | Every load thread writes GPU tensors (unverified on a GPU). |
| `--no-mmap` | off | Read CPU-resident tensors instead of mapping them from the file. |
| `--expert-profile-seed` | `16` | With `--expert-profile`: the routing heat the profile's hottest expert starts with. For tuning. |
| `--expert-profile-split-only` | off | With `--expert-profile`: the profile sizes each layer's slots only (no first contents, no heat seed). For testing a measured per-layer split. |
| `--expert-split-misses PATH` | off | Sizes each layer's slots from the hits and misses a previous run saved to PATH (written on exit; a missing file splits as usual): each GPU's slot bytes go where the next slot saves the most misses. Placement only, same outputs. |

**Fast prefill (GLM-5.3-Flash on a GB10, opt-in).** `--glm-mla-tc --glm-expert-q8` together are the
fast prefill set for Maya-S on one DGX Spark: both move output, so neither is a default. Measured on
one GB10 (`--max-seqs 4 --ctx-size 40960 --max-batch-tokens 2048`, temperature 0, #603), prompt
tokens per second to the first token:

| Prompt tokens | 736 | 4,304 | 5,746 | 8,198 | 31,545 |
|---|---|---|---|---|---|
| default | 551 | 553 | 579 | 569 | 510 |
| `--glm-mla-tc` | 556 | 645 | 694 | 685 | 621 |
| `--glm-expert-q8` | 614 | 592 | 621 | 607 | 546 |
| both | 635 | 700 | 753 | 741 | 667 |

Project Maya's own engine does about 730 tokens/s at ~6K on one Spark. Decode is unchanged (both
only touch prompt steps). Against the default over 3 raw ~4-5K-token prompts (128 greedy tokens
each, top-5 log-probs), both together give a top-5 KL mean of 8.5e-4 (max 5.5e-2) on the 64
positions with the same context, top-1 61/64; `--glm-mla-tc` alone 3.1e-3 (max 8.4e-2), 199/202.

Since #594 `--glm-expert-q8` also covers Maya-S's 8 layers with IQ3_XXS down, and the down GEMMs
run with 16 warps per block. Both flags on, against #603 on the same GB10 (2 rounds, the standard bench
prompts): 736 tokens 636 -> 651, 4,304 tokens 680 -> 703, 5,746 tokens 754 -> 760, 22,950 tokens
686 -> 688 tok/s; nsys of one 7,321-token prompt, GPU busy 10.0 -> 9.7 s (-3%). `--glm-expert-q8`
alone against the default, on the 50 positions all three runs share: top-5 KL mean 3.2e-3 (max
5.0e-2), where #603's version gave 2.1e-3 (max 4.0e-2).

**Fast decode (GLM-5.3-Flash on the AC922, four V100s, opt-in).** With `--glm-parallel-decode`
every GPU runs its share of every layer and the four GPUs sum their parts twice per layer. Each
GPU's routed experts sit in host memory on its own socket, and since #720 the loader also writes
them from that socket's CPUs: on POWER9 a GPU reads memory last written by the other socket at
about two thirds of the speed, which made GPUs 2 and 3 the GPUs the others waited for every step.

`--glm-parallel-mtp` (opt-in, needs `--glm-parallel-decode` and `--spec mtp`) runs the MTP draft
head's FFN on every GPU: each GPU its own share of the head's routed experts and shared-expert
rows, one all-reduce, then each GPU its rows of the draft logits, gathered on the last GPU for the
argmax. The last GPU no longer keeps a full copy of the LM head, so that memory goes to its expert
cache. The head's embedding, projection and attention still run on the last GPU. Greedy output is
the same as without the flag (with either split); with `--glm-expert-split interleave` the flag is
ignored with a warning.

`--glm-expert-shares` (one number of the 288 routed experts per GPU in CUDA order; needs the
default contiguous `--glm-expert-split`) gives GPUs fewer experts. It moves output (the routed sum
is added in another order, so greedy text differs at near-ties), so the even split stays the
default. Before #720, 78,78,70,62 made up for the slow reads on GPUs 2 and 3; now it overcorrects.

The fastest set measured so far:

```bash
legion serve --model GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --device cuda --gpus 4 --experts cache --max-seqs 4 --ctx-size 40960 \
  --spec mtp --spec-k 2 --glm-parallel-decode --glm-parallel-mtp
```

Measured on 4x V100 16 GB with Maya-S v2 (det suite: 8 fixed prompts, temperature 0,
256 tokens each, 2 rounds; decode tok/s without the time to the first token). Before #720 (#716,
#719):

| | C=1 per request | C=4 per request | C=4 total |
|---|---|---|---|
| even split | 53.7 | 14.5 | 47.2 |
| `--glm-expert-shares 78,78,70,62` | 55.8 | 17.4 | 53.6 |
| `--glm-expert-shares 78,78,70,62 --glm-parallel-mtp` | 55.6 | 17.5 | 54.5 |
| even split + `--glm-parallel-mtp` | 54.0 | 15.4 | 48.9 |

Prefill is the same with every flag (8K / 32K: about 889 / 948 tokens/s). With #720, both with
`--glm-parallel-mtp`:

| | C=1 per request | C=4 per request | C=4 total | the others wait for GPU 0/1/2/3, ms per step |
|---|---|---|---|---|
| even split, before #720 | 54.0 | 16.3 | 49.9 | 6 / 7 / 25-29 / 32-39 |
| even split | 58.1 | 20.5 | 66.8 | 8-9 / 8-10 / 8-10 / 10-12 |
| `--glm-expert-shares 78,78,70,62` | 58.5 | 19.9 | 62.6 | 12-15 / 14-18 / 7-9 / 5-6 |

#720 leaves the output byte-identical.

The #723 overlap (needs `--glm-parallel-mtp`) shortens the host gap between steps: every GPU
queues its MTP pass first and its verify commit right after it (the commits used to be launched
from the main thread on the last GPU), and the expert-cache work of the step runs while the passes
launch. It wins for one sequence and loses at C=4 (the adaptive depth drafts the same, about one
token per sequence, in both orders; the overlap order itself is slower there). Since #725 a step
takes it only when it has at most `--glm-mtp-overlap-seqs` sequences: default 1, `0` turns it
off, and `--glm-mtp-overlap` applies it to every step (for comparisons). #725 also keeps 32 verify
commit graphs per GPU instead of 8 for GLM, so C=4 rebuilds them less (the `commit` part of the
C=4 gap 4.2 -> 2.0 ms). Greedy output is the same. `--glm-parallel-decode --spec-k 2
--glm-parallel-mtp`, even split, det suite medians (tok/s):

| | C=1 total \| per request | C=4 total \| per request | greedy chats |
|---|---|---|---|
| before #725 (overlap off) | 43.2 \| 56.9 | 62.8 \| 19.1 | 56.6 |
| #725 defaults | 45.1 \| 60.8 | 67.8 \| 20.1 | 59.8 |

`--glm-pair-balance` (#729, opt-in, needs `--glm-parallel-decode`) evens out the routed experts
inside each NVLink pair (GPUs 0,1 and 2,3). In a decode step the exchanges wait at every MoE
layer for the GPU with the most expert work. Most of that wait comes from where the cache misses
fall: a GPU reads an uncached expert from host memory at about 12 times the time of a cached
one. The cached hits are even across the GPUs. The server logs "expert parallelism: routed work
per step" every 512 logged steps. At C=1 with main, the busiest GPU has about 35 more misses per
step than the mean (summed over the layers), about 4 ms of the 4.4 ms imbalance #728 measured.

With the flag, an expert the owner caches stays with the owner. An uncached one goes to whichever
GPU of the owner's pair has less work so far in that layer; both GPUs read the same socket's
host memory. Both GPUs of a pair compute the same assignment on the GPU, from the router's choices
and a per-layer list of the experts their owners cache, so they need no extra exchange. Moving
work across pairs would need cross-socket reads at about 2/3 speed, so it is not done. The log's
`pair_cost_excess` is what this assignment leaves, about 35% less than `cost_excess`. Prompts
are unchanged.

The routed sum adds in another order, and the order follows the cache's contents. So with the
flag, greedy output can change from run to run at near-ties (see the table below). Same command
as above, det suite medians (tok/s):

| | C=1 total \| per request | C=4 total \| per request | exchange wait, ms per step |
|---|---|---|---|
| main 90a2dec | 45.0 \| 60.5 | 65.5 \| 19.5 | 12.2-13.5 |
| `--glm-pair-balance` | 46.3 \| 63.0 | 68.5 \| 19.6 | 8.5 |

C=4 rounds vary by about ±5% in every arm. The waits are the mean over the run's 512-step
windows, mixing C=1, C=4 and prompt steps.

`--glm-prefetch-stats` (#745, measurement only, needs `--glm-parallel-decode`) measures how well
the next layer's cache misses could be foreseen, the first step toward prefetching them while
the current layer computes. Each MoE layer also applies the next layer's router to its own FFN
input. Every 512 logged steps the server logs "expert prefetch predictors": per GPU and step,
the misses, how many of them that guess or the previous step's choices name, and how many
uncached experts each would copy. The output is unchanged, but the extra router costs about
5% at C=1 and 9% at C=4, so leave it off for serving. Same command as above:

| | C=1 | C=4 |
|---|---|---|
| misses per GPU and step | 28-37 | 114-125 |
| named by the next layer's router | 64-67% | 66-67% |
| named by the previous step | 12-16% | 22-24% |
| uncached experts the router guess names | about 32 | about 125 |

The next layer's router is the predictor to build on: it would leave about 10 misses per GPU
and step at C=1. `--glm-prefetch` (#747) builds it; see below.

Where the misses go after #729 (#731, C=1, from the same log lines). Each GPU reads 28-42 of its
own experts per step from host memory, about 115 µs each, so 3.5-5 ms of the 32 ms step. Averaged
over the step that is only 5.4-6.6 GiB/s per GPU and 11.5-13 GiB/s per socket: the links idle
most of the step, and the reads wait in line inside each layer. The cache holds about 46% of the
experts and hits about 80% of the routed ones; the swaps move about 15 experts per step.

`--glm-cross-balance` (#731, opt-in, implies `--glm-pair-balance`) also lets an uncached expert go
to a GPU of the other pair, at a cost of 18 against 12 inside the pair (the other socket's host
memory reads at about 2/3 speed). The log's `cross_cost_excess` is what it leaves, only about 6%
less than `pair_cost_excess`: after the pair balance, what is left in most layers is one miss,
and moving a single miss to the other socket costs more than it saves. Measured, it does not
gain. A slower expert heat decay (`--expert-heat-decay 0.99`, half the swaps) does not raise the
hit rate either. Same command, det suite medians (tok/s), with Xid 41 before and after each run:

| | C=1 total \| per request | C=4 total \| per request | chats |
|---|---|---|---|
| `--glm-pair-balance` (#729) | 46.3 \| 63.0 | 68.5 \| 19.6 | |
| `--glm-cross-balance` | 44.8 \| 59.9 | 61.3 \| 18.5 | 57.5, 61.1, 61.4 |
| `--glm-pair-balance --expert-heat-decay 0.99` | 45.7 \| 60.7 | 69.6 \| 21.4 | 59.5, 65.0, 57.5 |

What would cut the rest: split one uncached expert's matvec over the GPUs of a pair along its
intermediate rows, so each reads half of it (the exchange already adds the partial sums), or
prefetch the next layer's misses while this layer computes (`--glm-prefetch`, below).

`--glm-prefetch N` (#747, opt-in, needs `--glm-parallel-decode`) builds that prefetch. At every
MoE layer of a decode step of at most 16 tokens, a side stream applies the next layer's router to
this layer's FFN input on the GPU. It then copies up to N of the GPU's guessed experts that miss
its cache into a VRAM ring of 2 x N expert slots, taken from the cache's share of VRAM, not from
the 1 GiB reserve. The next layer reads those of its real choices from the ring with the same
bits instead of from host memory. Without `--glm-pair-balance`, the output is byte-identical to
the default. With it, staged experts count as cached, so the output follows that, as pair balance
does. Same command as above:

| | C=1 per request | C=4 total |
|---|---|---|
| default | 60.5 | 63.8 |
| `--glm-prefetch 4` | 57.2 | 64.5 |
| `--glm-pair-balance --glm-prefetch 4` | 56.3 | 65.4 |

It does not pay yet. Each layer's experts wait for the whole copy. That copy shares the host link
with the current layer's own misses, and about a third of it is wrong guesses. Leave it off. The
next step, a per-slot ready flag so the read falls back to host memory instead of waiting, is on
#747.

`--glm-pair-split` (#735, opt-in, needs `--glm-parallel-decode`) splits each uncached expert
over both GPUs of the owner's pair, instead of giving it whole to one of them. Each GPU computes
the SwiGLU rows of its half of the intermediate and the down projection over that half of K.
The exchange adds the two partial sums. Each GPU reads half the expert from its node's host
memory. A cached expert stays whole with its owner. Where the split kernels take the expert
types (Maya-S's IQ2_XXS, IQ2_S and IQ3_XXS; the MTP head's Q2_K and Q3_K), the flag takes
precedence over `--glm-pair-balance`. Like the pair balance, the order of the routed sum follows
the cache's contents, so greedy output can change from run to run. Same command, det suite
medians (tok/s), one session:

| | C=1 total \| per request | C=4 total \| per request | C=1 exchange wait, ms per step |
|---|---|---|---|
| main 4962fe1 | 44.9 \| 60.2 | 63.2 \| 18.9 | 8.7 |
| `--glm-pair-balance` | 45.8 \| 61.9 | 66.5 \| 20.5 | 7.0 |
| `--glm-pair-split` | 46.2 \| 62.4 | 69.1 \| 20.7 | 6.9 |

At C=1 the split cuts the same share of the wait as the pair balance. There are fewer than one
miss per layer and GPU, and a split puts the same bytes on each GPU of a pair as a balanced
whole expert does. The C=4 gain over the balance (+4%) is inside the ±5% spread between rounds.

### Deterministic output

A greedy (`temperature: 0`) or seeded request gives the same bytes from run to run when every
step that computes it has the same shape. With the default kernels the expert cache does not
matter: how many slots it got (it follows a few MB of free memory) and which experts it holds or
swaps during the run change where an expert's weights are read from, not the sums (all experts go
through the same tiles). What does change the output:

| Source | Default | Deterministic with |
|---|---|---|
| Prefix cache: a prompt that shares a prefix with an earlier request starts from a saved state, so the rest runs in other chunks, with other sums. The output depends on what the server served before. | on (`--prefix-cache-states 8`) | `--prefix-cache-states 0` |
| Adaptive draft depth (`--spec mtp`): picked from measured step times, so which tokens a verify step holds follows the clock. | on | `--spec-fixed` |
| Step graph memory layout (#459): ggml's allocator gives a graph the plan (every intermediate's offset) it made for the graph it allocated before, whenever the new one has as many nodes and each fits; kept graphs hand their allocator on to the next shape. A single chat's steps then get the offsets of, say, the batch of 8 the server ran before, and ggml-cuda fuses an MoE layer's weighted expert sum only where its output does not overlap its inputs (the fused kernel rounds with FMAs), so the sums follow what the server served before. | the last plan that fits | planned afresh per graph |
| Requests decoded at the same time share steps; a step's size changes the order of the sums (#49), so concurrent output follows which step each request joined, i.e. arrival timing. | admitted as they arrive | admission in waves (`--deterministic-gather-ms`, #836): requests sent together are admitted together in a fixed order |
| `--expert-mv-vram 1` / `2`: a cached expert is summed in another order than an uncached one, so the output follows the cache's contents. | off (`0`) | `0` |
| `--glm-pair-balance`: which GPU of a pair computes an uncached expert follows the cache's contents, and so does the order of the routed sum. The contents follow the cache's size (the auto size follows a few MB of free VRAM), the requests served before, and when a swap's copy or a refill after a long prompt is seen finished. | off | `--deterministic` (each step waits for the copies and the refill, #1165) plus a fixed `--expert-cache-gib`, and the same requests served before. Maya-S v2 on 4x V100: the same bytes in two runs at C=1/4 and at depth 1536/4096; C=1 at depth 1536 -7.5%, at 4096 -2%, depth 0 unchanged. |
| `--glm-cross-balance`: as `--glm-pair-balance`, over all four GPUs. | off | as `--glm-pair-balance` (not measured) |
| `--glm-pair-split`: which experts are computed in two halves (added in the exchange) follows the cache's contents. | off | as `--glm-pair-balance` (not measured) |
| `--qsa-topk-ggml`: ggml's `top_k` picks between scores tied exactly at the threshold in the order its threads arrive. | off | off |

`--deterministic` sets the first four, refuses `--expert-mv-vram` 1 and 2, and with the GLM
pair balance or split makes each step wait for the expert cache's copies. Its waves: an
idle server gathers requests until none has arrived for `--deterministic-gather-ms` (200 ms),
orders them by prompt tokens and limits (equal requests in arrival order) and admits them
together; a request that arrives while a wave runs waits until the wave is done. So a client
that sends its requests together (a batch, the bench det client's C=16) gets the same steps,
and the same bytes, every run, and one that keeps C requests in flight gets waves of C. The
first request of a wave waits the window, and a wave's free slots stay idle until it ends.
Measured on one GB10 (Qwen3.8-Flash-Next UD-Q4_K_XL + Q8_0 MTP, `--deterministic
--spec-gate-cost-model off`, bench det client): C=1, C=4 and C=16 rounds and the three chats give
the same texts in every round and after a server restart; with `--deterministic-gather-ms 0`
every C=16 round differs. On 4 V100s (POWER9) with a 50 ms window the bench client's C=4
rounds still split one wave differently from round to round: the request sent when the last of a
wave finished reached the server 51-56 ms later in two rounds and 7 ms later in the other, hence
the 200 ms default. Two identical requests sent together may still get each other's text
(rows of one step can round differently), since the server cannot tell them apart. A
different
`--max-batch-tokens`, `--prefill-chunk`, `--moe-kernel`, GPU count or device gives other (equally
valid) bytes; keep them equal between runs. `scripts/determinism.py` sends a fixed request
sequence and byte-compares runs.

Measured on Qwen3.8-Flash-Next UD-Q4_K_XL, two V100s (#327), one request at a time, 10K and 123K
prompts plus three chats, server restarted between runs: byte-identical with
`--prefix-cache-states 0`, MTP off (also between runs whose caches got 5629 and 5631 slots) and
MTP at a fixed depth of 2; with the adaptive depth the chats after the 123K prompt drifted
apart. With prefix caching on, the 10K prompt gave other text after a request holding its first
half than on a fresh server. Prefix caching also places chunk boundaries at its save points, so
`--deterministic` output differs from the default's, as any other chunking does.
Cost of `--deterministic` there: a repeated prompt is read again (10K: 4.2 s instead of 0.27 s
to the first token; 123K: 33.7 s instead of 0.64 s); MTP at the fixed depth 2 decodes chats at
55-57 tok/s (adaptive 51-56) and after the 10K prompt at 42-43 tok/s (adaptive 45-46); without
drafts decode is unchanged.

#459 (Qwen3.8-Flash-Next UD-Q4_K_XL, two V100s, MTP k=2, 4,639 cache slots): before step graphs were
planned afresh, the three chats run on a server that had first served a batch of 4 and one of 8
(16 and 24 tokens) diverged from a fresh server's after 722, 933 and 1,295 characters. That is
what looked like `--pipeline-groups 2` changing the output: the groups' steps drop the whole steps'
kept graphs, so after that warm-up its single chats got fresh plans and matched a fresh server,
while the default did not. With thinking off the same warm-ups moved the other side (the
`--pipeline-groups 2` server's logprobs bit-identical up to one step, then different). With
`GGML_CUDA_DISABLE_FUSION=1`, with `--moe-combine` (the MoE sum rounded one way whatever the
layout) or with fresh plans, default, `--pipeline-groups 2`, warm and fresh servers give the same
bytes. Fresh plans cost nothing measurable (decode set, 10K and 123K prefill within noise).
Without `--deterministic` the plans stay as they were (and so the default's bytes), and a server's
output can follow what it served before this way.

Even with fresh plans, the layout itself follows the graph: a change elsewhere in it (`--kv-append`,
`--ple-ops`, `--shexp-gate`, `--gated-norm`, a new op) can flip which layers ggml-cuda fuses the MoE
weighted sum in, so greedy text moves with changes that compute the same bits per op (#472). llama.cpp
does not see this: ggml-cuda asks ggml's scheduler to keep the fusions' inputs allocated until their
outputs. Legion allocates its graphs itself; `--stable-fusions` adds the same dependency nodes, so the
fusion fires in every layer whatever the layout. Measured (#472, the setup above, 4,639 slots): the
text moves once from the default (the chats differ after 203, 933 and 167 characters) and is then
`--moe-combine`'s text, logprobs bit for bit; with each of the four switches above added it stays the
same, while the default moves under each. No kernels added: decode and 10K / 123K prefill within noise
(prompt chunks take the fused MoE op and build no chain). The step-buffer sizing grows by about 10 MiB
per GPU (16 expert cache slots of 5,363). Off by default, so the default keeps its bytes.

### Constrained output: JSON and strict tool calls

The model engine constrains sampling with Legion's grammar engine (`src/grammar/`, Rust, no new
dependency) for two kinds of request:

- **`response_format`** `{"type":"json_object"}` or `{"type":"json_schema",...}` (chat and legacy
  completions; `text.format` on `/v1/responses`): every token, from the first, until the document is
  complete; then the sequence ends. See [JSON output](api.md#json-output) for the request shape.
- **Strict tool calls**: a tool with `"strict": true` (`tools[].function.strict` on chat completions,
  `tools[].strict` on `/v1/responses`). The model writes freely, reasoning included, until it samples
  the `<tool_call>` token outside its reasoning; from there the call follows the Qwen3-Coder format
  (`<function=NAME>`, then `<parameter=KEY>` blocks) up to the `</tool_call>` token, and the model is
  free again (further calls are constrained the same way). `NAME` must be one of the request's tools.
  A strict tool's parameters follow its schema: its `properties` in declaration order, the
  `required` ones always and the others optionally, nothing else (as OpenAI's strict mode requires
  `additionalProperties: false`). A parameter whose `type` is or includes `"string"` (or that has no
  `type`) is raw text, or one of its `enum` / `const` strings; any other parameter is JSON for its
  schema, with the same supported subset as `response_format`. Tools without `strict` in the same
  request only have their name checked. Every call that completes parses into arguments that follow
  the schema. Limit: a JSON-typed parameter (an object or array) whose strings contain the text
  `</parameter>` or `</tool_call>` would still cut the call short; raw text cannot contain them.

A request with neither runs exactly the engine's usual path: no constraint, no mask, the same bytes.
Constraints are per request: in a batch, only constrained sequences pay for them. While a sequence is
constrained (a response format, or inside a strict call) it does not draft (speculative decoding) and
the step returns its full logits row instead of the argmax. Drafts of a sequence waiting for a strict
call stop before the `<tool_call>` token, so a call's tokens are never accepted unconstrained.

**Off the critical path.** A token's mask is a bitset over the vocabulary, found from the grammar's
state (a table lookup once that state was seen, a walk of the vocabulary's trie the first time).
Right after a sequence's token is accepted, a mask thread (`--grammar-threads`, default 2, started
the first time a request is constrained, off the engine thread's core) finds the next state's mask
while the engine thread plans and runs the next forward pass; sampling then applies the ready mask,
32 tokens a word. `--grammar-threads 0` finds each mask on the engine thread at sampling time
instead. Compiled grammars are cached per model (64, least recently used first), so a repeated schema
or tool set compiles once.

**Unsupported cases.** A `response_format` schema the grammar cannot express returns HTTP 400. A
strict tool whose schema it cannot express (for example a `pattern`, `anyOf` or `minLength` on a
string parameter, or `parameters` that are not an object) leaves that request's calls unconstrained,
as before strict was enforced, with a warning in the log. Gemma 4's tool-call format, and models
whose vocabulary has no `<tool_call>` / `</tool_call>` tokens, are not constrained (`strict` is then
accepted and ignored). `response_format` takes precedence over strict tools in the same request.

GPU throughput of constrained requests (for example 32 concurrent JSON schema requests against the
same batch unconstrained) has not been measured yet.

## Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Built-in web UI |
| `GET` | `/health` | Server health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/v1/models` | List available models |
| `GET` | `/models` | The same list as `/v1/models` (llama.cpp's path; LLM proxies discover models here) |
| `GET` | `/v1/models/{id}` | Get information for a specific model |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/completions` | OpenAI-compatible completions |
| `POST` | `/v1/messages` | Anthropic-compatible messages |
| `POST` | `/v1/messages/count_tokens` | Anthropic-compatible token counting |
| `POST` | `/v1/responses` | Anthropic-compatible responses |
| `GET` | `/v1/stats` | The live dashboard's snapshot (JSON, [dashboard.md](dashboard.md)) |
| `GET` | `/v1/stats/stream` | The snapshot as server-sent events (`?interval_ms=` 100-5000, default 250) |
| `GET` | `/v1/logs/stream` | The last 2,000 log lines and new ones as server-sent events (`?level=`) |

## Clients

### curl
Non-streaming:
```sh
curl -s localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}'
```

Streaming:
```sh
curl -s localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

### OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="your-api-key" # Only if --api-key is set on server
)
```

### Anthropic Python SDK
```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8090",
    api_key="your-api-key" # Only if --api-key is set on server
)
```

### Claude Code
```sh
ANTHROPIC_BASE_URL=http://localhost:8090 claude
```

If `--api-key` is set on the server, clients must send it as `Authorization: Bearer <key>` or `x-api-key: <key>`.

### Registering with a proxy

`--registry-api <URL>` makes the server announce itself to an LLM proxy that has a dynamic
`POST /register` route, such as [simple-llm-proxy](https://github.com/eelgaev/simple-llm-proxy):

```sh
legion serve --model ... --api-key "$KEY" --registry-api http://192.0.2.5:8000
```

- Once the model is loaded and the listener is bound, a background task POSTs
  `{"host": "<IP>", "port": <bound port>, "api_key": "<first --api-key, or empty>"}` to
  `<URL>/register`. The proxy then checks `GET /health` and `GET /models` with that key (https
  first, then http) and answers `201`; the server logs one info line with the proxy, the URL it
  chose and the model count. The key itself is never logged.
- The advertised IP is `--registry-host`, else `--host` when it is a specific address, else the
  local address of the route to the proxy (the interface the server would reach it from).
  Behind NAT, or when the proxy must use another interface, pass `--registry-host`.
- Failures (proxy down, a refused check, a non-2xx answer) are logged with the proxy's message
  and retried with a doubling backoff from 1 s to 60 s; they never stop or delay the server.
- After a success the server registers again every `--registry-interval` seconds (default 300),
  so a proxy that restarted without its saved host list picks it up again. Re-registering the
  same address and port replaces the proxy's entry.
- The proxy lists the models as `<IP>/<model id>`.
- simple-llm-proxy queues each registered host behind a semaphore of 1: it sends one request at
  a time to this server, so the engine's continuous batching sees one sequence per proxy. That is
  the proxy's limit, not the server's.

### Priority and fair share

By default the engine admits waiting requests in arrival order, so one client that sends a
burst of long jobs makes everyone behind it wait. Two opt-in flags change the order; they only
reorder the queue (running requests are preempted only under `--kv-admission on-demand`, below,
and then the lowest priority first).

- `--scheduling-policy priority`: requests carry an integer `priority` (OpenAI chat,
  completions and responses, Anthropic messages; default 0). The lowest value is admitted
  first, ties in arrival order, and prompts being prefilled get the step's token budget in the
  same order. Lower-first is vLLM's convention; SGLang's `--enable-priority-scheduling`
  schedules higher values first unless `--schedule-low-priority-values-first` is set, so
  clients written for SGLang must negate their values. Under the default `fcfs` the field is
  accepted and ignored.
- `--fair-share api-key|user`: among waiting requests of the best priority, the client
  (API key or `user`) holding the fewest running sequences goes next, round robin between
  clients holding the same number. While several clients wait, none holds more than
  `ceil(max sequences / clients)` slots (a slot is never left idle for it: when every waiting
  client is at its share, the next one is admitted anyway). Requests without a key count as
  one client. Slots are the shared resource here because each running sequence gets one token
  per decode step: holding k of the n slots is a k/n share of decode throughput. Counting
  tokens instead (deficit round robin) needs each request's output length, which is not known
  at admission. Keys are never logged; the request's log line shows an 8-digit hash that is
  stable within one server process.

The live status line adds the waiting requests per priority class and the number of waiting
clients after the queued count, e.g. `queued 45 (p-1:5 p10:40, 2 keys)`, only when something
other than priority 0 or more than one client waits. `/metrics` has
`legion_engine_waiting_by_priority{priority}` and `legion_engine_waiting_keys`, and every
OpenAI response's `timings` and the request log line carry `queue_ms`, the time the request
waited for a slot.

### KV admission on demand

By default (`--kv-admission reserve`) a request is admitted only when the KV cache has room for
its prompt plus its whole `max_tokens`, and those blocks stay reserved until it ends. Clients
often send `max_tokens` far above what they use (an agent asking for 32K tokens that stops
after 300), so the cache holds few sequences and requests queue while most of it sits unused.

`--kv-admission on-demand` admits on the prompt alone, as SGLang and vLLM do:

- A request is admitted when the cache has room for its prompt plus a step's headroom (and one
  free block per decoding sequence for their growth). A request whose prompt plus `max_tokens`
  is larger than the whole cache is still refused, so a request running alone always finishes.
- Before each step, every decoding sequence gets the blocks its next step writes (the token,
  and with speculative decoding the drafts and their verify rows).
- When those do not fit, the prefix cache and finished sequences kept warm give theirs back
  first. If that is not enough, the cheapest running sequence is preempted: the lowest
  priority under `--scheduling-policy priority`, then the fewest tokens in the cache (the
  cheapest to recompute), then the latest arrival. Its cache is freed, its stream stays open,
  and it goes back to the queue ahead of every other request of its priority.
- When it is admitted again, its prompt and the tokens it had generated are read again as one
  prompt (prefix cache hits count), and it continues from the token it was about to feed. The
  client sees only a pause. Its text state (detokenizer, parsers, grammar) is kept, on the
  engine thread or the stream thread (`--stream-thread-seqs`), wherever it was.

Recompute rather than swapping to host memory: prompt steps are the engine's fastest path, and
recurrent state (Qwen's DeltaNet, Nemotron's Mamba) is rebuilt by the same prefill with nothing
to save. With greedy sampling on the CPU reference kernels a preempted request's text is
identical to an uninterrupted run. On GPUs a prompt step and a decode step round differently,
so a preempted request can continue differently (as in SGLang and vLLM); requests that are not
preempted keep their output. Not available with `--pipeline-groups` or KV slots (`--kv-slots`),
where the engine falls back to `reserve` with a warning. Preemptions are logged at `info` and
counted in `legion_engine_kv_preemptions`.

### Latency targets

Each step runs every decoding sequence plus prompt chunks up to `--max-batch-tokens`. A long
prompt arriving while chats decode therefore makes every chat wait one long step per chunk:
their inter-token latency (ITL) jumps for as long as the prompt is read. `--itl-target <ms>` and
`--ttft-target <ms>` (both off by default) size each step's prompt budget instead
(`src/engine/lm/slo.rs`, #628):

- **Step-time model.** The engine fits the time of steps with decoding rows online (least
  squares, recent steps weighing most) as a linear function of the decode / verify rows, the
  positions they attend to, whether the step reads a prompt chunk, its tokens and the positions
  they attend to. Every 30 s it logs the model's out-of-sample error (`step budget: step-time
  model`: mean absolute error in ms and percent, RMS, steps over the ITL target).
- **`--itl-target`.** While sequences decode, a step takes the most prompt tokens (whole KV
  blocks) whose predicted step stays under the target, and at least one block. Until the model
  has seen a few steps with prompt chunks, a probe: two blocks, doubled after a step under the
  target and halved after one over. With nothing decoding a prompt gets the whole
  `--max-batch-tokens`, so a prompt alone reads as fast as without targets.
- **`--ttft-target`.** Deadline: arrival plus the target. A prompt is at risk when, read in
  arrival order at the planned pace, it would finish after its deadline. Prompts at risk read
  first, the fewest tokens left first, within their `--scheduling-policy priority` class (classes
  still come first). With both targets a step takes more than the ITL budget when an at-risk
  prompt can still make its deadline that way; one no budget could save takes nothing extra. A
  prompt past its deadline reads after the others (arrival order among the late ones): once a
  prompt has missed, letting it go first would make the prompts behind it miss too. A preempted
  decoding sequence (`--kv-admission on-demand`) being read again is due now (its stream
  stalls meanwhile), so it is at risk and reads first.

Only step boundaries move: every token is computed as without targets, in other chunks, so
greedy text can differ only as other chunking does on GPUs (see [Deterministic
output](#deterministic-output)); on the CPU reference kernels it is identical
(`tests/slo_budget.rs`). Not with decoding groups in flight (`--pipeline-groups` above 1, the
default over two machines): their steps are not whole, and the server refuses the flags there.
A layer split, `--tp` and `--tp-rank` (rank 0 schedules) run whole steps and are measured as
they are.

## Monitoring

### Live dashboard
`legion serve --tui` draws a live dashboard in the server's terminal; `legion top --url ...`
draws the same from another terminal or machine (`--api-key` or `LEGION_API_KEY` when the server
has keys). See [dashboard.md](dashboard.md).

### Health Check
The `/health` endpoint returns the server status and version.
Example output:
```json
{"status":"ok","version":"0.1.0","git":"783614b","models":["mock"]}
```

### Metrics
The `/metrics` endpoint provides Prometheus-compatible metrics for monitoring the server's performance.
The memory-bandwidth families (`legion_engine_model_*`, `legion_engine_stage_*`, and `legion_gpu_*` with `--hw-counters`) are described in [bandwidth.md](bandwidth.md).
