# Qwen3.8-Flash-Next NVFP4: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `nvidia/Qwen3.8-Flash-Next-NVFP4`. MTP with the checkpoint's own head:

```sh
legion serve \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --kv-cache-tokens 262144 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-draft-vocab 65536
```

Deterministic client: 8 fixed prompts (prose, code, lists, chats, a story; thinking off and on), temperature 0,
256 tokens each, 2 rounds at 1, 4 and 16 users. 2026-11-03, this command: 1 user 34.4 tok/s (story) to 60.1
(code), median 42.2, mean accepted length 3.21; 4 users 90.4; 16 users 121.0. Prefill 8,240, 32,738 and
122,528 tokens, one request each, from the same server: 2,049 / 2,088 / 1,976 tok/s.

The gate behind `--spec-max-batch` stops the MTP drafts at 8 or more decoding sequences, so 16 users decode
without drafts. Without `--spec mtp` (an earlier build): 1 user 19.7, 4 users 56.3, 16 users 121.9, prefill
8K 1,977 / 32K 1,972.

### 16 users: one output-moving opt-in

Add this to the command above:

```sh
  --spec-gate-cost-model off
```

The table's row is this command (2026-11-03, the same session as the numbers above): 16 users decode at
172.1 tok/s total, against 121.0 without the flag; MTP keeps drafting at the depth its step-cost model picks,
mean accepted length 3.07. 1 user 34.8 (story) to 57.7 (code), median 41.9, mean accepted length 3.13;
4 users 92.0. Prefill 8,240 / 32,738 / 122,528 tokens: 2,024 / 2,065 / 1,946 tok/s. Greedy text at 8 or
more users changes (other drafts, other verify shapes).

- The opt-in prefill numbers (8K 1,749 / 32K 1,800) add `--nvfp4-a4 prefill --prefill-even` to the command
  without `--spec` (an earlier build).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--kv-cache-tokens` (`262144`): Size of the shared KV cache in tokens (all sequences plus the prefix cache). At least `--ctx-size`, so one request can use the full context; shorter requests share it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output. The NVFP4 checkpoint ships its head, so no `--spec-draft` file is needed.
- `--spec-draft-vocab` (`65536`): The MTP head scores only the first N vocabulary entries while drafting, reading N rows of the output head instead of all of them. Drafting gets cheaper; the verify step still checks against the full vocabulary, so output does not change.
- `--spec-gate-cost-model` (`off`, opt-in above): lets MTP keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
