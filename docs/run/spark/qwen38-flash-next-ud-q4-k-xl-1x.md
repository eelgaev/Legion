# Qwen3.8-Flash-Next UD-Q4_K_XL: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

```sh
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 131072 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-draft-vocab 65536
```

- Decode (C=1 49.2, C=4 83.8, C=16 109.4): deterministic client at C=1/4/16, 2 rounds on each of two servers,
  the median of the two servers' medians.
- Prefill (8K 1,872, 32K 2,008, 64K 1,969): 8192/32768/65536 tokens, one request each; 32K and 64K (and an 8K
  1,828) were measured with `--deterministic` added, on an earlier build. The same two servers gave 8K 1,953 /
  1,681 and 32K 1,979 / 1,894 (one request each after an 8K warm-up).
- 123K prefill (1,916): a 122,880-token prompt, one request after an 8K warm-up, the median of two servers
  (1,913 / 1,920, 2026-11-04).
- The C=8 103.9 / C=16 148.7 run adds `--spec-gate-cost-model off` (an earlier build, C=1/4/8/16).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`131072`): Maximum tokens per sequence (prompt plus output). The full native 262,144 does not fit next to the weights on one Spark (the server stops at start: room for about 241,000 tokens). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-draft-vocab` (`65536`): The MTP head scores only the first N vocabulary entries while drafting, reading N rows of the output head instead of all of them. Drafting gets cheaper; the verify step still checks against the full vocabulary, so output does not change.

## Optional

- `--ngram-ram <auto|all|SIZE>` (not set: `auto`): How much of the 26.8 GiB n-gram table stays in RAM. On a Spark `auto` reads it from the SSD with a row cache of up to 1 GiB (less when little memory is free at load), because the GPU needs the rest of the shared memory. A SIZE such as `4G` sets that RAM exactly, with its most-hit rows pinned ([docs/serving.md](../../serving.md)). These benchmarks touch about 21 MB of distinct rows, so the 1 GiB cache these runs had already holds every row read twice: 2G, 4G and 8G gave the same hit rate (87.2%), the same speed and the same output as the default. A bigger budget only helps traffic whose distinct rows outgrow the default cache, and it takes KV-cache room.
