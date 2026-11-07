# Gemma 4 31B QAT UD-Q4_K_XL: IBM AC922 2x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Two GPUs of one NVLink pair, the process bound to their NUMA node. MTP k=3:

```sh
legion serve \
  --model gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --tp 2 \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-gemma-4-31B-it.gguf \
  --spec-k 3 \
  --spec-gate-cost-model off \
  --spec-max-rows 63
```

Without a drafter, per-GPU executors:

```sh
legion serve \
  --model gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --tp 2 \
  --max-seqs 4 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --tp-per-gpu
```

The README row is the MTP command.

- `--max-seqs 16 --ctx-size 262144` gives each of the 16 sequences its own context of up to 262,144 tokens.
  That is 4,194,304 tokens of KV cache in all.
- The cache lives in a KV host tier in pinned memory on the GPUs' NUMA node, backed as the sequences grow.
  Part of the sliding-window rings is in host memory too.
  The ring the MTP assistant reads in every draft pass is always one of those kept in VRAM.
- With `--max-seqs 4` the four slots don't fit in VRAM either, but the VRAM left past the rings does hold
  about 49K tokens (`shared_device_tokens` in the startup log). The slots share it in 4096-token chunks,
  the running sequence first: one user at 32K reads its whole context from VRAM, at 64K three quarters of it.
  A finished (warm) sequence's chunks move to the host tier when a running one needs them.
- llama.cpp's `-np 16 -kvu -c 262144` (see [other engines](../other-engines.md)) is 262,144 tokens in all, shared by its 16 slots.
- Prompt steps copy each sequence's host-tier keys and values into GPU memory once per layer before attention.
  This happens for contexts up to the length whose copy fits in the compute buffers: 150,720 tokens here
  (`prompt_copy_tokens` in the startup log). Longer contexts read the host tier in place.
  `--kv-slot-stage-rows 0` turns the copy off.
- `--spec-gate-cost-model off --spec-max-rows 63` keep MTP drafting at 8 or more users with verify steps of at most 63 rows:
  16 users draft up to 2 tokens each, 12 or fewer up to 3. At 64 rows a V100 runs the quantized matmuls and the LM head
  as cuBLAS F16 GEMMs over F16 copies. Without the two flags drafting stops at 8 users: 16 users decoded 194.8 instead
  of 247.2, 4 users 137.7 instead of 141.1, 1 user and prefill the same. Greedy text at 8 or more users moves with them.
- With `--deterministic` or `--spec-fixed`, all 16 sequences draft, so a verify step asks the LM head for 64 rows.
  A V100 runs that through ggml's F16 copy of the head (2.6 GiB per GPU, `head_pool_mib` in the startup log).
  That room is kept free, so fewer sliding-window rings stay in VRAM than with the command above.

With `--max-seqs 4 --ctx-size 40960` prefill was 8K 1,064 / 32K 734.

MTP row: deterministic client, C=1/4/16, 2 rounds. The 1-user cell is the range of per-request speeds over
both rounds. Prefill: one request per size. `--tp-per-gpu`: llama-benchy C=1/4/16 (C=16 queues on the 4
sequences), 3 runs.

## What each flag does

- `--tp` (`2`): Tensor parallelism over this many GPUs in one process: every layer's heads and FFN rows are split, with one all-reduce after attention and one after the FFN. For dense models such as Gemma 4 31B.
- `--max-seqs` (`16`; `4` with `--tp-per-gpu`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). With 16 sequences each one gets a slot of this size, mostly in the KV host tier, backed only as it fills.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-gemma-4-31B-it.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-k` (`3`): Maximum draft tokens per sequence and step. The depth policy drafts fewer when drafts do not pay off.
- `--spec-gate-cost-model` (`off`): Past `--spec-max-batch` (8 sequences) the MTP head keeps drafting and its cost model picks the depth, instead of drafting stopping there. Output-moving at 8 or more users.
- `--spec-max-rows` (`63`): The most rows a verify step takes (each sequence's token and its drafts); the depth is lowered at large batches to stay within it. 63 keeps V100 steps below ggml's 64-row cuBLAS path and its F16 copy of the LM head (2.6 GiB per GPU). Output-moving where it lowers the depth.
- `--tp-per-gpu`: With `--tp 2`: runs each GPU as its own rank (own executor, launch thread and CUDA graphs per step shape) instead of one ggml device spanning both; the partial sums are added over NVLink. Faster at higher concurrency.
