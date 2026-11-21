# Qwen3.8-27B NVFP4: 2x DGX Spark (tensor parallel)

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `unsloth/Qwen3.8-27B-NVFP4`, `--spec-draft` the downloaded snapshot of
`z-lab/Qwen3.8-27B-DFlash2` (both read as shipped), on each Spark. One process per Spark. Rank 0 serves HTTP;
`--tp-peers` lists the two ranks' addresses on the QSFP link (rank 0 first) and is the same on both. The command
is the [1x one](qwen38-27b-nvfp4-1x.md) with its two opt-in blocks plus the tensor-parallel flags:

```sh
# rank 0
legion serve \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --kv-cache-tokens 262144 \
  --prefix-cache-states 0 \
  --spec dflash \
  --spec-draft z-lab/Qwen3.8-27B-DFlash2 \
  --spec-draft-weights q4 \
  --spec-draft-vocab 131072 \
  --spec-draft-lm-head q4 \
  --fp8-sliced-k \
  --nvfp4-wide-mv \
  --fp8-act a8-dynamic \
  --nvfp4-a4 prefill \
  --spec-gate-cost-model off \
  --max-batch-tokens 16384 \
  --prefill-chunk 16384 \
  --tp-profile 1 \
  --tp-rails 2 \
  --tp-draft-split \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port, with --tp-rank 1
```

Deterministic client: the same 8 fixed prompts as the 1x row, temperature 0, 256 tokens each (2026-11-21, two
sessions, 2 rounds each, with everything below that has no flag). 1 user 37.8 (story) to 83.5 (code), median
51.4-51.7 per round, mean accepted length 3.37, the same tokens in every round; 140.1 tok/s at 4 users (the median
of 4 rounds, 139.1-142.9); 301.7 at 16 users (298.3-304.4). Prefill 8K 3,031-3,064 (median 3,048) and 32K
2,838-2,847 (2,842) tok/s, two requests each per session after an 8K warm-up, from the same servers. 123K was not
measured. vLLM 0.33.0 TP2 (2026-11-20, [its command](../other-engines.md#qwen38-27b)): 1 user 37.5-75.5 (median
49.0), 125.6 at 4 users, 248.0 at 16 users, prefill 8K 2,913 and 32K 2,757.

`--tp-draft-split` (2026-11-21): the drafter's FFN and head split over the two Sparks. Against the command without
it (same build, two sessions in ABBA order): 1 user median 49.4 -> 51.7 (+4.7%, the same 1-user text), 4 users
139.2 -> 142.6, 16 users 287.9 -> 298.7; prefill ties since prompt steps read back only the rows the drafter takes
in (#1133). Before `--tp-draft-split` (2026-11-21 morning): 1 user median 49.2-49.8, 136.9 at 4 users, 295.9 at
16 users, prefill 8K 3,002 and 32K 2,780.

The first measurement (2026-11-19, one session, the build before the halves below): 1 user 36.6 (story) to 80.4 (code), median 50.2, mean accepted length 3.36; 136.1 tok/s at 4 users
(2 rounds: 136.4 / 135.8); 283.1 at 16 users (282.9 / 283.4). Prefill 8K 2,074 / 2,008 and 32K 1,794 / 1,939
tok/s, one request each after an 8K warm-up, from the same server. 123K was not measured. Against the 1x row,
decode gains 47% at 1 user (median 34.1), 40% at 4 users (97.5) and 20% at 16 users (235.2), but prefill is
slower than on one Spark (8K 2,697, 32K 2,309): each of a prompt step's 128 sums (two per layer) carries the
whole step's hidden states, and a 8K step waits about 1.2 of its 4 s for them.

`--tp-head-split` does not apply: this checkpoint's output head is FP8 with one scale per row, and the split head
takes only a plain matmul of the head.

Prompt steps of 2,048 tokens or more run each layer's sums in two halves of their tokens, so half 1's mixer sum
travels during half 0's FFN and half 0's FFN sum during half 1's (#1133, no flag; the mixer runs whole, and the
FP8 MLPs of layers 56-63 stay whole). With the command above plus `--deterministic`, two sessions in ABBA order
(whole / halves, halves / whole; whole = `LEGION_DEBUG_QWEN35_TP_HALVES=0` on both ranks): prefill 8K 1,885-1,926 / 2,156-2,195, 2,156-2,202 / 1,909-1,929 and 32K
1,755 / 1,847, 1,831 / 1,789 tok/s (each includes `--deterministic`'s 200 ms gather), an 8K prompt step on
rank 0 about 4,100 -> 3,560 ms. Decode does not change (1, 4 and 16 users within 1%), and the output is the same
bits: the same text at 1, 4 and 16 users and the same log probabilities for 8K and 32K prompts. Prefill is still
below one Spark's: the halves hide about half of the sums' wire time.

Each sum also adds the layer's residual in the same kernel (one Spark adds it in the GEMM epilogue), the halves
write their FFN sums straight into one output, and the post-attention and DeltaNet input norms ride in the
quantization as on one Spark (#1133, no flag, same bits). That takes about 380 ms of kernels out of an 8K prompt
step on each rank. With the command above plus `--deterministic`, two sessions in ABBA order (before / after,
after / before): prefill 8K 2,167-2,198 / 2,415-2,453, 2,413-2,458 / 2,161-2,186 and 32K 1,767 / 2,016,
2,030 / 1,820 tok/s. Decode does not change (1, 4 and 16 users within 1%), and text and log probabilities are
the same bits.

A rank holds 8 of the 16 key heads and 24 of the 48 value heads of each DeltaNet layer, a head count ggml-cuda
runs token after token instead of on its chunked kernel. Its prompt steps now take Legion's `gdn_scan` (the V100
kernel, ggml-cuda's arithmetic, #1133, no flag, `--gdn-scan-ggml` goes back): 285 -> 128 ms of an 8K prompt step
on each rank. With the command above plus `--deterministic`, two sessions in ABBA order (before / after,
after / before): prefill 8K 2,418-2,447 / 2,542-2,575, 2,535-2,583 / 2,404-2,451 and 32K 2,025-2,398 /
2,111-2,525, 2,112-2,529 / 2,000-2,401 tok/s (two requests each, the second 32K one is the faster). Decode does
not change (1 user 48.5-48.7, 4 users 126.8-127.7, 16 users 270.3-273.1), and text and log probabilities are the
same bits.

Optional, output-moving: `--tp-wire bf16` on both ranks sends sums of at least 1 MiB in BF16, half the bytes:
prompt steps, the 16-user verify steps and short prompts (all ranks still agree). On top of the command above
(same session day, one session): prefill 8K 2,676 / 2,728 and 32K 2,252 / 2,539 tok/s (an 8K prompt step on rank 0
about 3,020 ms), 1 user 36.4 (story) to 78.2 (code), median 57.8 (other text: a short prompt's step sums move, so
the drafts accept differently, mean accepted length 3.54), 4 users 139.6 / 137.6, 16 users 310.6 / 303.7.
With `gdn_scan` on the ranks (the build before its final launch shape, which is ~30 ms faster per 8K step) and
`--deterministic`, one session: prefill 8K 2,963 / 3,026 and 32K 2,584 / 2,920 tok/s (vLLM 0.33.0 TP2: 2,913 and
2,757), 1 user median 57.7, 4 users 129.5, 16 users 303.3.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`), `--kv-cache-tokens` (`262144`), `--prefix-cache-states` (`0`): as in the [1x command](qwen38-27b-nvfp4-1x.md#what-each-flag-does).
- `--spec` (`dflash`) and the next ten flags: as in the [1x command](qwen38-27b-nvfp4-1x.md#what-each-flag-does). Under tensor parallelism the DFlash drafter runs between verify steps, split over both ranks with `--tp-draft-split` (below).
- `--tp-profile` (`1`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--tp-rails` (`2`): Each Spark's QSFP port is two RDMA devices, one per PCIe link of about 108 Gb/s. With `2`, sums of at least 1 MiB go in 1 MiB pieces over both devices instead of only the one on the `--tp-peers` address (the second needs no IP address). The same bytes arrive, so output does not change.
- `--tp-draft-split`: each Spark holds half of every drafter layer's FFN rows and both run each draft round (one sum per drafter layer); with greedy drafts and the Q4 head (`--spec-draft-weights q4`) the head is split too, each rank taking the top candidates of half the vocabulary. Rank 1 no longer waits while rank 0 drafts. The sums run in another order, so drafts can differ in bits (output-moving at 4 and 16 users; the 1-user text above is the same). Both ranks pass it.
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
