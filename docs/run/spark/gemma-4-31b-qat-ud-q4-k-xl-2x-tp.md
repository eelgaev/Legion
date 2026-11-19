# Gemma 4 31B QAT UD-Q4_K_XL: 2x DGX Spark (tensor parallel)

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

```sh
# rank 0
legion serve \
  --model gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft mtp-gemma-4-31B-it-Q8_0.gguf \
  --tp-profile 50 \
  --mmproj mmproj-F16.gguf \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port and --mmproj, with --tp-rank 1
```

Deterministic client at C=1/4/16, 2 rounds (2026-11-04): 1 user 37-66 tok/s by prompt (median 49.4, mean
accepted length 3.53), 4 users 119.9, 16 users 202.6. Prefill 8,201 / 32,705 / 122,601 tokens, one request each
after an 8K warm-up: 975 / 925 / 744 tok/s. `--tp-profile 50` only logs step times.

### Images

With `--mmproj` on rank 0 requests may carry images: rank 0 encodes on its GPU and rank 1 gets each image's
rows from rank 0, once per request. 2026-11-19: a 1024 × 768 image (336 image tokens) encodes in 0.39 s and its
first token comes 0.75-0.96 s after the request (0.35-0.54 s when the same image is sent again); smaller images
(102-221 tokens) encode in 65-192 ms, TTFT 0.25-0.67 s. Greedy answers to the six test images equal one Spark's.
Text-only requests give the same output as without `--mmproj`, at the same speed (`--deterministic` 4 users
109.4 tok/s against 111.7, 1 user 45.8 against 45.8).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-gemma-4-31B-it-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--tp-profile` (`50`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--mmproj` (`mmproj-F16.gguf`): The vision tower (the `mmproj-F16.gguf` file in the same Hugging Face repo), so requests may carry images. Rank 0 loads it and encodes on its GPU; rank 1 gets each image's rows from rank 0. Leave it out for text only.
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
