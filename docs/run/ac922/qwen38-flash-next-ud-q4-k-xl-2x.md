# Qwen3.8-Flash-Next UD-Q4_K_XL: IBM AC922 2x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Two GPUs of one NVLink pair, layer split, the process bound to their NUMA node. The 4-GPU record set's flags
(all apply to a 2-GPU split), with `--gpus 2` (2026-11-01: C=1 76.9, C=4 86.4; C=16 88.5 with MTP drafting past
8 users (2026-11-02, #814; 76.0 before); prefill 8K 2,789, 32K 4,094, 123K 4,302):

```sh
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --gpus 2 \
  --experts cache \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-k 2 \
  --qsa-attn-chain \
  --qsa-scores-tc \
  --ple-ops \
  --long-dataflow \
  --weights-fp16 load \
  --weights-fp16-gates \
  --long-pipeline-micro 3072 \
  --hc-fuse \
  --long-dataflow-peer
```

On a V100 layer split of 2 GPUs with the expert cache, the KV pools' tail is lent to the expert cache by default
(`--kv-vmm-keep 32768`: each GPU keeps VRAM for the first 32,768 tokens of its pool and gives the rest, 1,440 MiB,
to about 6,550 expert slots in all). A sequence past 32K maps the tail again; `--no-kv-vmm` turns the lend off.
When the tail goes back, the chats cached on it move to pinned host memory and are copied back when resumed, with
no flag (up to an eighth of the free host memory; the load log's "KV cache" line prints the split).

Measured 2026-11-01 with this exact command on three server starts. Deterministic client at C=1/4/16, 2 rounds per
start, the median of the 6 rounds (C=1 per request; 66.7 tok/s total over the round). Prefill 8192/32768/122880
tokens, one request each per start after a throwaway 8K warm-up, the median of the three starts (the 123K prompt
takes 28.5 s).

MTP drafting stops at 8 decoding sequences (`--spec-max-batch 8`) and, on this layout by default, starts again
past 8 (`--spec-gate-cost-model`, #814): a step of up to 8 rows runs its matmuls as matrix-vector products, which
drafts would leave, but past 8 every step runs the tiled kernels and the drafts pay. Against the gate at every
size from 8 (2026-11-02, one start each, 2 rounds, deterministic client): C=12 70.9 -> 82.5 tok/s, C=16 79.7 ->
88.5; C=1 67.4 -> 67.8, C=4 86.6 -> 87.5, C=8 98.0 -> 98.4, C=1 at 1.5K/4K context and prefill 8K/32K unchanged.
Gated, 12 and 16 users were below C=4. Drafting at exactly 8 users is slower (98.0 -> 93.0), so the gate stays there.
`--spec-fixed` is no better at 16 users (78) and much slower at 4 (56): fixed depth k=2 puts 4 users' verify steps
past 8 rows.

## What each flag does

- `--gpus` (`2`): Splits the layers over this many GPUs, one pipeline stage each. Used because the model does not fit in one 16 GB V100.
- `--experts` (`cache`): Where a MoE model's routed experts live. `cache` keeps them in host memory and caches the most used ones in VRAM; used because the experts are far larger than the GPUs' VRAM, and on the AC922 the GPUs read host memory over NVLink 2.0.
- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-k` (`2`): Maximum draft tokens per sequence and step. The depth policy drafts fewer when drafts do not pay off.
- `--qsa-attn-chain`: V100 prompt steps: accumulates the sparse-attention products in chains across K steps, 17% faster at 32K. Changes the rounding (opt-in).
- `--qsa-scores-tc`: V100 prompt steps: computes the sparse-attention indexer's scores on the tensor cores. FP32-level accuracy, may shift the selected pools (opt-in).
- `--ple-ops`: Computes the per-layer n-gram embedding with two Legion ops instead of ggml's chain of about thirty nodes, saving about 10 ms per 4,096-token chunk on a V100.
- `--long-dataflow`: Long prompts over several GPUs: each GPU starts its next micro-chunk as soon as its own inputs are ready instead of waiting for the slowest GPU each round. Same outputs.
- `--weights-fp16` (`load`): `load` converts BF16 weights to FP16 while loading, because the V100 has FP16 tensor cores but no BF16 ones. Output shifts slightly (opt-in).
- `--weights-fp16-gates`: With `--weights-fp16 load`: the F32 router and gate weights become FP16 too. Routing shifts slightly (opt-in).
- `--long-pipeline-micro` (`3072`): Micro-chunk size (tokens) of the multi-GPU prefill pipeline. The default is 4,096; the 4-GPU record runs use 3,072, kept here. See [tuning-multi-gpu-prefill.md](../../tuning-multi-gpu-prefill.md).
- `--hc-fuse`: V100 prompt steps: folds the gated shared expert and the next combine-and-norm into the fused MoE op, saving passes over the hidden states.
- `--long-dataflow-peer`: With `--long-dataflow`: turns on peer access between the two GPUs, so a stage's residual goes straight to the next GPU over their NVLink instead of through host memory. Same outputs.
