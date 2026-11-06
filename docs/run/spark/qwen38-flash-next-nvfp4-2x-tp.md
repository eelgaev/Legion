# Qwen3.8-Flash-Next NVFP4: 2x DGX Spark (tensor parallel)

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `nvidia/Qwen3.8-Flash-Next-NVFP4` on each Spark. One process per Spark.
Rank 0 serves HTTP; `--tp-peers` lists the two ranks' addresses on the QSFP link (rank 0 first) and is the same on
both. MTP uses the checkpoint's own head, so there is no `--spec-draft` file. Decode (38-51 tok/s at 1 user, 111.5 at 4,
231.7 at 16) and prefill (2,701 / 2,816 / 2,685 tok/s at 8K / 32K / 123K):

```sh
# rank 0
legion serve \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --kv-cache-tokens 262144 \
  --prefix-cache-states 0 \
  --max-batch-tokens 4608 \
  --prefill-even \
  --warmup-prompt 13824 \
  --tp-wire bf16 \
  --tp-seq-par 1024 \
  --tp-profile 1 \
  --gdn-mma \
  --spec mtp \
  --spec-draft-vocab 65536 \
  --spec-k 2 \
  --tp-mtp-chain \
  --tp-mtp-local-rows \
  --spec-fixed \
  --tp-moe-ep \
  --prefill-slots 16 \
  --tp-rails 2 \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port, with --tp-rank 1
```

Measured against vLLM 0.33.0 with tensor parallel 2, MTP and CUDA graphs on the same two Sparks (2026-11-04:
43-58 / 116.8 / 248.5 tok/s decode, 3,174 / 3,257 / 3,135 tok/s prefill); its command is in
[other engines](../other-engines.md).

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--kv-cache-tokens` (`262144`): Size of the shared KV cache in tokens (all sequences plus the prefix cache). At least `--ctx-size`, so one request can use the full context; shorter requests share it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--max-batch-tokens` (`4608`): Tokens per forward pass (decoding sequences plus prompt chunks). A bigger step amortizes each MoE step's expert reads over more prompt tokens.
- `--prefill-even`: Splits a long prompt into equal steps instead of full steps plus a short last one, because a short MoE step still reads every routed expert.
- `--warmup-prompt` (`13824`): Runs one prompt of about this many tokens before taking requests, so compute buffers (and, with `--tp-rank`, the network exchange buffers) are allocated at full size up front instead of during the first real prompt.
- `--tp-wire` (`bf16`): `bf16` sends large prompt-step sums at half the bytes over the single QSFP link; decode steps stay F32. Prompt output drifts slightly from F32 (all ranks still agree).
- `--tp-seq-par` (`1024`): Prompt steps of at least this many tokens also split the per-token work by tokens (sequence parallelism), so each Spark runs the hyper-connection mixers on only its share of the tokens.
- `--tp-profile` (`1`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--gdn-mma`: Gated DeltaNet layers of long prompts run on Legion's chunked tensor-core kernel even for head counts where ggml uses its per-token kernel. Same bits, faster prefill.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output. The NVFP4 checkpoint ships its head, so no `--spec-draft` file is needed.
- `--spec-draft-vocab` (`65536`): The MTP head scores only the first N vocabulary entries while drafting, reading N rows of the output head instead of all of them. Drafting gets cheaper; the verify step still checks against the full vocabulary, so output does not change.
- `--spec-k` (`2`): Maximum draft tokens per sequence and step.
- `--tp-mtp-chain`: Rank 0 sends a whole draft round as one request and every rank runs the round's further MTP passes on its own, instead of one network round trip per pass.
- `--tp-mtp-local-rows`: The other ranks read the MTP head's input rows from their own copy of the step output instead of receiving them from rank 0.
- `--spec-fixed`: Always drafts `--spec-k` tokens: turns off the depth policy's acceptance adaptation and its batch gate.
- `--tp-moe-ep`: Expert parallelism inside tensor parallelism: each expert lives on one rank, so its fixed cost is paid once instead of on both Sparks. Each rank reads and repacks only its own NVFP4 experts at load. The checkpoint's per-expert scales keep its experts
  on Legion's grouped NVFP4 kernels, which skip the other rank's (token, expert) pairs, so each rank computes only
  its own experts' pairs (steps of up to 8 tokens, such as 1-user decode, still compute them on the rank's first
  expert with a zero router weight, so they add nothing). In prompt steps the weighted expert sum and the shared
  expert's gate then run as one kernel that skips the other rank's zero-weight rows (prefill +4-5%, same bits).
- `--prefill-slots` (`16`): How many prompts are read in one step (default 4). At 16, sixteen requests arriving
  together start decoding in the same step instead of four per step (16 users: 234.9 instead of about 223 tok/s in
  the same session without it; 1 and 4 users and prefill are unchanged). Batches change, so with several users
  greedy text can differ from the default's (as it already does between runs).
- `--tp-rails` (`2`): Each Spark's QSFP port is two RDMA devices, one per PCIe link of about 108 Gb/s. With
  `2`, prompt-step sums of at least 1 MiB go in 1 MiB pieces over both devices instead of only the one on the
  `--tp-peers` address (the second needs no IP address). Prefill +2% (2,523 / 2,609 / 2,506 tok/s before);
  decode steps stay on the first device, and the same bytes arrive, so output does not change.
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
