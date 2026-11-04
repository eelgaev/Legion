# Qwen3.8-Flash-Next UD-Q4_K_XL: IBM AC922 4x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Record set + `--hc-fuse --long-dataflow-peer` (2026-11-01: C=1 93.0, C=4 118.6, C=16 136.2; prefill 8K 4,370,
32K 6,271, 123K 7,997):

```sh
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --gpus 4 \
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

Measured 2026-11-01 with this exact command on two server starts. Deterministic client at C=1/4/16, 2 rounds per
start, the median of the 4 rounds (C=1 per request; 81.4 tok/s total over the round). Prefill 8192/32768/122880
tokens, one request each per start after a throwaway 8K warm-up, the median of the two starts (8K 4,471 / 4,268,
32K 6,152 / 6,390, 123K 8,085 / 7,908; the 123K prompt takes 15.2-15.5 s). The README keeps the earlier 32K 6,316
and 123K ~8,050 records, which these runs match within 1%. Every flag in the command is still an opt-in; none has
become a default.

## What each flag does

- `--gpus` (`4`): Splits the layers over this many GPUs, one pipeline stage each. Used because the model does not fit in one 16 GB V100.
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
- `--long-pipeline-micro` (`3072`): Micro-chunk size (tokens) of the multi-GPU prefill pipeline. The default is 4,096; this record run used 3,072 (smaller chunks, shorter pipeline fill and drain at the start and end of a prompt). See [tuning-multi-gpu-prefill.md](../../tuning-multi-gpu-prefill.md).
- `--hc-fuse`: V100 prompt steps: folds the gated shared expert and the next combine-and-norm into the fused MoE op, saving passes over the hidden states.
- `--long-dataflow-peer`: With `--long-dataflow`: turns on peer access between neighbouring GPUs, so a stage's residual goes straight to the next GPU instead of through host memory. Same outputs.
