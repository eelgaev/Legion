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
  --spec-draft-weights q4 \
  --spec-draft-lm-head q4 \
  --head-rope-fused \
  --spec-draft-batched-attn \
  --tp-profile 1 \
  --tp-head-split \
  --tp-rails 2 \
  --tp-wire bf16 \
  --tp-draft-split \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port, with --tp-rank 1
```

Deterministic client: the same 8 fixed prompts as the 1x row, temperature 0, 256 tokens each (2026-11-20, two
sessions, 2 rounds each). 1 user 36.5 to 66.0 tok/s by prompt, median 46.6 over the 4 rounds, mean accepted length
3.27-3.32, the same tokens in every round; 126.4 tok/s at 4 users (the median of 4 rounds, 120.7-129.1); 363.8 at
16 users (the median of 4 rounds, 353.4-381.5). Prefill 8K 2,765 / 2,789 (mean 2,777) and 32K 2,496 / 2,500
(2,498) tok/s, one request each per session after an 8K warm-up, from the same servers. 123K was not measured
(heat, below).

`--tp-draft-split` and `--spec-draft-weights q4` (2026-11-20): the drafter's FFN split over the two Sparks and its
matrices in 4 bits instead of Q8_0 cut the time between verify steps (the draft pass and host work, on rank 0)
from 19 / 23 / 33 ms to 11 / 14.5 / 21-22 ms at 1 / 4 / 16 users; the verify step stays at 58 / 67-69 / 86 ms.
Against the command before (`--spec-draft-weights q8_0`, no `--tp-draft-split`), two sessions in ABBA order
(before / after / after / before): 1 user median 41.9 / 46.4 / 46.8 / 41.8, 4 users 113.4 / 126.4 / 124.9 /
115.4, 16 users 326.3 / 369.2 / 362.1 / 323.2, prefill 8K 2,797 / 2,765 / 2,789 / 2,787 and 32K 2,524 / 2,496 /
2,500 / 2,515. The 1-user tokens are the same as before; the 4-bit drafter accepts a little less (1 user 3.28
vs 3.34 with the split, one session each). With `--deterministic`, `--tp-draft-split` keeps the text at 1 and 4
users (8-bit drafter, one session); at 16 users the text differs.

Tried on the command before, one session each, and slower: the drafter as shipped (`--spec-draft-weights model
--spec-draft-lm-head model`, BF16) accepts more (1 user 3.38 vs 3.32) but its draft pass takes 38.5 ms instead of
19: 1 user 34.5 vs 42.2, 4 users 95.5 vs 116.0. A fixed depth of 7 (`--spec-fixed`, vLLM's 7 speculative tokens)
accepts more at 4 users (3.36 vs 3.23) but the 32-row verify step takes 74 instead of 69 ms: 4 users 114.3 vs
116.0, 1 user the same. `--tp-rails 1` instead of 2: 4 users 111.0 vs 116.0, 1 user the same.

The command before, the same day (one session, after the prompt-step halves below): 1 user 31.8 (story) to 58.2
(code), median 41.8 over 2 rounds, mean accepted length 3.30, the same tokens in both rounds; 117.6 tok/s at 4 users
(2 rounds, 117.4 / 117.8); 329.6 at 16 users (the median of 4 rounds, 272.0-339.4; mean accepted length 3.10).
Prefill 8K 2,787 and 32K 2,525 tok/s. 123K was not measured: it took both Sparks from 69 to 80 and 86 C within a
minute, and the run stopped at 85 C.

The 2026-11-19 numbers before the halves (one session): 1 user 32.3-58.2, median 42.2 (the same tokens as the 1x
command over two Sparks without `--tp-head-split`, `--tp-rails 2` and `--tp-wire bf16`), 113.9 tok/s at 4 users,
323.7 at 16 users (one round), prefill 8K 2,373 and 32K 2,166.

The 1x command over two Sparks as it is (only `--tp-rank`/`--tp-peers` and `--tp-profile 1` added), same day:
1 user 29.6-53.8 (median 38.7-39.1 in five runs), 4 users 105.2-109.4, 16 users 259.9-274.9, prefill 8K
1,736-1,752 and 32K 1,629-1,636. Adding `--tp-head-split --tp-rails 2` (both exact), two sessions in ABBA order:
1 user 42.4 / 42.4 median vs 39.1 / 38.8, 4 users 112.8 / 117.2 vs 107.2 / 105.2, 16 users 289.2 / 279.6 vs
262.8 / 265.5, prefill 8K 1,970 / 1,954 vs 1,751 / 1,752, 32K 1,804 / 1,804 vs 1,634 / 1,636. With
`--deterministic` the text is the same with and without them at 1, 4 and 16 users.

Prompt steps of 2,048 tokens or more run each layer in two halves of their tokens, so one half's sums travel
while the GPU computes the other half (#619, no flag). With the command above plus `--deterministic`, four
sessions in ABBA order (whole / halves / halves / whole): prefill 8K 2,274 / 2,660 / 2,659 / 2,266 and 32K
2,130 / 2,466 / 2,466 / 2,138 tok/s (each includes `--deterministic`'s 200 ms gather), a 32K prompt step on rank 0
1,886 -> 1,634 ms. Decode does not change (1, 4 and 16 users within 1%), and the output is the same bits:
the same text and log probabilities at 1, 4 and 16 users, at 8K depth and for 8K and 32K prompts.

`--tp-draft-split` alone, 2026-11-19 (8-bit drafter), on both ranks: the DFlash drafter's FFN is split over the two Sparks like the target's,
so rank 1 no longer waits while rank 0 drafts. The time between verify steps drops from about 19 / 25 / 33 ms to
15-18 / 19 / 25 ms at 1 / 4 / 16 users. On top of the command above, two sessions in ABBA order: 1 user 45.3 / 45.6
vs 42.4 / 42.6, 4 users 117.6 / 117.7 vs 114.8 / 113.6, 16 users 339.7 / 336.3 vs 320.0 / 319.9; prefill does
not use the drafter's split (8K 2,328 vs 2,387, 32K 2,146 vs 2,155, one request each). The sums run in a different
order, so the drafts can differ in bits. With `--deterministic` the text is still the same at 1 and 4 users and the
mean accepted length about the same; without it, at 4 users the depth policy drafts less per step (4.5-4.9 vs
6.3-6.8 tokens, mean accepted length 3.03-3.07 vs 3.31-3.33) and still comes out ahead.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`) and the next six flags: as in the [1x command](gemma-4-31b-nvfp4-1x.md#what-each-flag-does), except `--spec-draft-weights`. Under tensor parallelism the DFlash drafter runs on rank 0, between verify steps (`--tp-draft-split` below splits its FFN over both).
- `--spec-draft-weights` (`q4`, opt-in): 4-bit copies of the drafter's BF16 matrices made at load (the 1x command's `q8_0` reads about twice as much a draft pass). Drafts only, every token is still verified by the full model; the drafts shape the verify batches, so at 4 and 16 users the text can move.
- `--tp-profile` (`1`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--tp-head-split`: Gemma's output head is tied to the embeddings (262,144 rows, 2.8 GB in BF16, soft-capped) and both Sparks ran all of it every step (~12 ms of a ~65 ms 1-user verify step). With this, a greedy decode or verify step computes it as two halves of the vocabulary, one on each Spark, each half soft-capped before its candidates, and rank 0 keeps the better of the two per row. Same output (the pick and its tie order are the whole head's). Requests with sampling, log probabilities or `min_tokens` keep the whole head.
- `--tp-rails` (`2`): Each Spark's QSFP port is two RDMA devices, one per PCIe link of about 108 Gb/s. With `2`, sums of at least 1 MiB go in 1 MiB pieces over both devices instead of only the one on the `--tp-peers` address (the second needs no IP address), and sums of 128 KiB up to 1 MiB (the 1- and 4-user verify steps) in two halves, one per device. The same bytes arrive, so output does not change.
- `--tp-wire` (`bf16`): sums of at least 1 MiB travel in BF16, half the bytes. Here that is prompt steps and the 16-user verify steps (128 rows of 5,376); 1 and 4 users stay F32. Prefill 8K 1,962 -> 2,373, 32K 1,804 -> 2,166 tok/s. Output-moving for prompts and 16-user batches (all ranks still agree).
- `--tp-draft-split`: each Spark holds half of every drafter layer's FFN rows and both run each draft round (one sum per drafter layer), so rank 1 no longer waits while rank 0 drafts; the drafter's head stays on rank 0. The sums run in another order, so the drafts can differ in bits (output-moving at 4 and 16 users, as above). Both ranks pass it.
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
