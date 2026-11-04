# Gemma 4 31B QAT UD-Q4_K_XL: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

```sh
legion serve \
  --model gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-gemma-4-31B-it-Q8_0.gguf \
  --spec-gate-cost-model off
```

Deterministic client: 8 fixed prompts (prose, code, lists, chats, a story; thinking off and on), temperature 0,
256 tokens each, 2 rounds at 1, 4 and 16 users (2026-11-03). At 1 user the speed depends on how many drafts the
prompt accepts: 24 to 49 tok/s per prompt, 34.8 in all, mean accepted length 3.58. 93.0 tok/s at 4 users, 254.9
at 16 users. Prefill 8K 791, 32K 704 and 123K 532 tok/s, one request each, from the same server.

Without `--spec-gate-cost-model off` (the same build, the run before): 16 users 137.0 tok/s, as MTP stops drafting
from 8 decoding sequences; 1 and 4 users 35.1 / 97.1, prefill 871 / 767 / 544 (a cooler GPU; the flag does not
touch prompt steps).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-gemma-4-31B-it-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-gate-cost-model` (`off`): lets MTP keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
