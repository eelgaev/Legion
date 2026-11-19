# Gemma 4 31B NVFP4 (RedHatAI): 2x DGX Spark (tensor parallel)

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `RedHatAI/gemma-4-31B-it-NVFP4`, `--spec-draft` the downloaded snapshot of
`RedHatAI/gemma-4-31B-it-speculator.dflash2` (both read as shipped), on each Spark. One process per Spark. Rank 0
serves HTTP; `--tp-peers` lists the two ranks' addresses on the QSFP link (rank 0 first) and is the same on both.
The command is the [1x one](gemma-4-31b-nvfp4-1x.md) plus the tensor-parallel flags:

```sh
# rank 0
legion serve \
  --model RedHatAI/gemma-4-31B-it-NVFP4 \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --spec dflash \
  --spec-draft RedHatAI/gemma-4-31B-it-speculator.dflash2 \
  --spec-gate-cost-model off \
  --nvfp4-a4 prefill \
  --spec-draft-weights q8_0 \
  --spec-draft-lm-head q4 \
  --head-rope-fused \
  --spec-draft-batched-attn \
  --tp-profile 1 \
  --tp-head-split \
  --tp-rails 2 \
  --tp-wire bf16 \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port, with --tp-rank 1
```

Deterministic client: the same 8 fixed prompts as the 1x row, temperature 0, 256 tokens each (2026-11-19, one
session). 1 user 32.3 (story) to 58.2 (code), median 42.2, mean accepted length 3.32, the same tokens as the 1x
command over two Sparks without the last three flags; 113.9 tok/s at 4 users; 323.7 at 16 users (one round).
Prefill 8K 2,373 and 32K 2,166 tok/s, one request each after an 8K warm-up, from the same server. 123K was not
measured (one Spark ran too hot during the long prefills).

The 1x command over two Sparks as it is (only `--tp-rank`/`--tp-peers` and `--tp-profile 1` added), same day:
1 user 29.6-53.8 (median 38.7-39.1 in five runs), 4 users 105.2-109.4, 16 users 259.9-274.9, prefill 8K
1,736-1,752 and 32K 1,629-1,636. Adding `--tp-head-split --tp-rails 2` (both exact), two sessions in ABBA order:
1 user 42.4 / 42.4 median vs 39.1 / 38.8, 4 users 112.8 / 117.2 vs 107.2 / 105.2, 16 users 289.2 / 279.6 vs
262.8 / 265.5, prefill 8K 1,970 / 1,954 vs 1,751 / 1,752, 32K 1,804 / 1,804 vs 1,634 / 1,636. With
`--deterministic` the text is the same with and without them at 1, 4 and 16 users.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`) and the next six flags: as in the [1x command](gemma-4-31b-nvfp4-1x.md#what-each-flag-does). Under tensor parallelism the DFlash drafter runs on rank 0 only, between verify steps.
- `--tp-profile` (`1`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--tp-head-split`: Gemma's output head is tied to the embeddings (262,144 rows, 2.8 GB in BF16, soft-capped) and both Sparks ran all of it every step (~12 ms of a ~65 ms 1-user verify step). With this, a greedy decode or verify step computes it as two halves of the vocabulary, one on each Spark, each half soft-capped before its candidates, and rank 0 keeps the better of the two per row. Same output (the pick and its tie order are the whole head's). Requests with sampling, log probabilities or `min_tokens` keep the whole head.
- `--tp-rails` (`2`): Each Spark's QSFP port is two RDMA devices, one per PCIe link of about 108 Gb/s. With `2`, sums of at least 1 MiB go in 1 MiB pieces over both devices instead of only the one on the `--tp-peers` address (the second needs no IP address). The same bytes arrive, so output does not change.
- `--tp-wire` (`bf16`): sums of at least 1 MiB travel in BF16, half the bytes. Here that is prompt steps and the 16-user verify steps (128 rows of 5,376); 1 and 4 users stay F32. Prefill 8K 1,962 -> 2,373, 32K 1,804 -> 2,166 tok/s. Output-moving for prompts and 16-user batches (all ranks still agree).
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
