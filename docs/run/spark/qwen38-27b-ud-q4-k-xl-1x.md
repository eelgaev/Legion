# Qwen3.8-27B UD-Q4_K_XL: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

```sh
legion serve \
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-k 8 \
  --spec-draft-vocab 65536
```

One round of the deterministic client (2026-11-03): 1 user 23.0 (story) to 40.4 tok/s (code), median 29.0,
mean accepted length 3.18; 4 users 80.1, 16 users 121.7 (the `--spec-max-batch` gate stops drafting at 8 users);
prefill 880 / 822 / 652 tok/s at 8K / 32K / 123K.

With `--spec-k 2` and without `--spec-draft-vocab` (every row of the MTP head): 1 user 21.3-25.9 (median 23.8), 4 users 70.2, 16 users 120.4; prefill 879 / 825 / 657.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks.
- `--spec-draft` (`mtp-Qwen3.8-27B-Q4_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-k` (`8`): Lets the depth policy draft up to 8 tokens a step; it picks the depth from measured step costs. Output-moving (other drafts, other verify shapes); every token is still verified by the full model.
- `--spec-draft-vocab` (`65536`): Each draft pass scores only the first 65,536 tokens of the MTP head's output layer, a cheaper pass over the 248,320-token vocabulary. Output-moving, as `--spec-k`.
