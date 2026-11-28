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
  --spec-gate-cost-model off \
  --mmproj mmproj-F16.gguf
```

Deterministic client: 8 fixed prompts (prose, code, lists, chats, a story; thinking off and on), temperature 0,
256 tokens each, 2 rounds at 1, 4 and 16 users on each of two servers, medians of the 4 rounds (2026-11-28). At 1
user the speed depends on how many drafts the prompt accepts: 23 to 49 tok/s per prompt, 35.0 in all, mean accepted
length 3.6. 93.5 tok/s at 4 users, 250.5 at 16 users (241.6-253.0 over the rounds). Prefill 8K 1,306 and 32K 1,122
tok/s, one request each from the same servers (mean of the two). 123K 738 tok/s, one request from a third server
started on a cool GPU (38 C; it peaks at 80 C).

The run of 2026-11-03, before the Q4_0 prompt matmuls: 24-49 per prompt, 93.0 at 4 users, 254.9 at 16 users,
prefill 791 / 704 / 532.

Without `--spec-gate-cost-model off` (the 2026-11-03 build, the run before it): 16 users 137.0 tok/s, as MTP stops drafting
from 8 decoding sequences; 1 and 4 users 35.1 / 97.1, prefill 871 / 767 / 544 (a cooler GPU; the flag does not
touch prompt steps).

### Images

With `--mmproj` requests may carry images, and `/v1/models` lists `vision`. The tower runs on the GPU
(2026-11-19): a 1024 × 768 image (336 image tokens) encodes in 0.39-0.40 s and its first token comes 0.8-1.0 s
after the request (encode 13.0 s, TTFT 13.4 s on the CPU tower, `LEGION_DEBUG_VISION_CPU=1`). Smaller images
(102-221 tokens) encode in 66-194 ms, TTFT 0.27-0.74 s. An image sent again takes its rows from a cache (TTFT
0.43-0.63 s). Text-only requests give the same output as without `--mmproj`, at the same speed (`--deterministic`
4 users 82.2 tok/s against 82.1, 1 user 30.2 against 30.2, 8K prefill 880 against 884).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-gemma-4-31B-it-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-gate-cost-model` (`off`): lets MTP keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
- `--mmproj` (`mmproj-F16.gguf`): The vision tower (the `mmproj-F16.gguf` file in the same Hugging Face repo), so requests may carry images. It runs on the GPU. Leave it out for text only.
