# Qwen3.8-27B UD-Q4_K_XL: IBM AC922 2x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Two GPUs of one NVLink pair, the layers split between them, the process bound to their NUMA node. MTP drafting:

```sh
legion serve \
  --model Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --gpus 2 \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-k 3 \
  --spec-draft-vocab 65536
```

One round of the deterministic client (2026-11-03): 1 user 58.5 (story) to 79.2 tok/s (code), median 69.1,
mean accepted length 2.72; 4 users 122.1, 16 users 194.6 (the `--spec-max-batch` gate stops drafting at 8 users);
prefill 1,375 / 1,023 / 509 tok/s at 8K / 32K / 123K. The 1-user cell is the range of the 8 per-request speeds;
prefill is one request per size.

With `--spec-k 2` and without `--spec-draft-vocab` (every row of the MTP head): 1 user 52.8-72.0 (median 63.4),
4 users 109.9, 16 users 195.4; prefill 1,374 / 1,024 / 509. Without `--spec*`: 1 user 37.7, 4 users 102.2,
16 users 198.5; prefill 1,381 / 1,028.

- `--max-seqs 16 --ctx-size 262144` gives each of the 16 sequences its own context of up to 262,144 tokens.
  About 90K tokens of KV cache fit in VRAM next to the weights; the rest is a KV host tier in pinned memory on
  the GPUs' NUMA node (the startup log says how large). llama.cpp's `-np 16 -kvu -c 262144` (see
  [other engines](../other-engines.md)) is 262,144 tokens in all, shared by its 16 slots.
- `--tp` supports dense Gemma 4 only, so this model runs as a layer split (`--gpus 2`): at one user the two GPUs
  take turns. llama.cpp splits every layer over both GPUs (`-sm tensor`).

## What each flag does

- `--gpus` (`2`): Splits the layers over this many GPUs, one pipeline stage each. Used because the model and 16 sequences' cache do not fit in one 16 GB V100.
- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). With 16 sequences each one gets a slot of this size, mostly in the KV host tier, backed only as it fills.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks.
- `--spec-draft` (`mtp-Qwen3.8-27B-Q4_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-k` (`3`): Lets the depth policy draft up to 3 tokens a step; it picks the depth from measured step costs. On two V100s a verify step grows with its rows, so deeper caps lose (4: 67.2, 8: 64.8 tok/s median at 1 user). Output-moving (other drafts, other verify shapes); every token is still verified by the full model.
- `--spec-draft-vocab` (`65536`): Each draft pass scores only the first 65,536 tokens of the MTP head's output layer, a cheaper pass over the 248,320-token vocabulary (with `--spec-k 3` alone: 65.7 tok/s median at 1 user). Output-moving, as `--spec-k`.
