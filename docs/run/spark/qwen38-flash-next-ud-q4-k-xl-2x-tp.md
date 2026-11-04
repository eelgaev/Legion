# Qwen3.8-Flash-Next UD-Q4_K_XL: 2x DGX Spark (tensor parallel)

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

One process per Spark. Rank 0 serves HTTP; `--tp-peers` lists the two ranks' addresses on the QSFP link
(rank 0 first) and is the same on both. Prefill (8K 2,933, 32K 2,940, 123K 2,758), plain TP:

```sh
# rank 0
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --max-batch-tokens 4608 \
  --prefill-even \
  --warmup-prompt 13824 \
  --tp-wire bf16 \
  --tp-seq-par 1024 \
  --tp-profile 1 \
  --gdn-mma \
  --tp-moe-ep \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
# rank 1: the same without --port, with --tp-rank 1
```

Decode (C=1 55-70, C=4 112.6, C=16 216.9) adds MTP on both ranks:

```sh
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 262144 \
  --prefix-cache-states 0 \
  --max-batch-tokens 4608 \
  --prefill-even \
  --warmup-prompt 13824 \
  --tp-wire bf16 \
  --tp-seq-par 1024 \
  --tp-profile 1 \
  --gdn-mma \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-draft-vocab 65536 \
  --spec-k 2 \
  --tp-mtp-chain \
  --tp-mtp-local-rows \
  --spec-fixed \
  --tp-moe-ep \
  --tp-rank 0 \
  --tp-peers <rank0-ip>:7619,<rank1-ip>:7619
```

- Prefill: 8192/32768/122880 tokens, 2 distinct prompts each after one unrecorded 8K warm-up request, the
  median of the two (8K 2,935 / 2,931, 32K 2,942 / 2,938, 123K 2,758 / 2,757). The MTP command's server in the
  same session gave 8K 2,853, 32K 2,827 and 123K 2,685 (one request each).
- Decode: deterministic client at C=1/4/16, 2 rounds (2026-11-04). At 1 user the speed depends on the prompt:
  55-70 tok/s, median 60.2, mean accepted length 2.52.
- `--tp-profile 1` only logs step times.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--max-batch-tokens` (`4608`): Tokens per forward pass (decoding sequences plus prompt chunks). A bigger step amortizes each MoE step's expert reads over more prompt tokens.
- `--prefill-even`: Splits a long prompt into equal steps instead of full steps plus a short last one, because a short MoE step still reads every routed expert.
- `--warmup-prompt` (`13824`): Runs one prompt of about this many tokens before taking requests, so compute buffers (and, with `--tp-rank`, the network exchange buffers) are allocated at full size up front instead of during the first real prompt.
- `--tp-wire` (`bf16`): `bf16` sends large prompt-step sums at half the bytes over the single QSFP link; decode steps stay F32. A sum of 4,096 Qwen3.8-Flash-Next prompt tokens is 42 MB, about 3.4 ms one way at F32; BF16 halves that. Prompt output drifts slightly from F32 (all ranks still agree).
- `--tp-seq-par` (`1024`): Prompt steps of at least this many tokens also split the per-token work by tokens (sequence parallelism), so each Spark runs the hyper-connection mixers on only its share of the tokens.
- `--tp-profile` (`1`): Logs where step time goes (compute, waiting for the other rank, sends) every N steps. Measurement only, not needed for normal use.
- `--gdn-mma`: Gated DeltaNet layers of long prompts run on Legion's chunked tensor-core kernel even for head counts where ggml uses its per-token kernel. Same bits, faster prefill.
- `--tp-moe-ep`: Expert parallelism inside tensor parallelism: each expert lives on one rank, so its fixed cost is paid once instead of on both Sparks. Measured on two Sparks with MTP: decode 48.6 to 51.2 tok/s at 1 user, 8K prefill 2,475 to 2,667.
- `--tp-rank` (`0`): This process's rank in tensor parallelism across machines (one process per DGX Spark). Rank 0 serves HTTP.
- `--tp-peers`: Every rank's `ip:port` on the QSFP link, in rank order, identical on all ranks.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-draft` (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`): The MTP head's weights, for models whose main GGUF does not include it (the `mtp-*.gguf` file in the same Hugging Face repo).
- `--spec-draft-vocab` (`65536`): The MTP head scores only the first N vocabulary entries while drafting, reading N rows of the output head instead of all of them. Drafting gets cheaper; the verify step still checks against the full vocabulary, so output does not change.
- `--spec-k` (`2`): Maximum draft tokens per sequence and step. The depth policy drafts fewer when drafts do not pay off.
- `--tp-mtp-chain`: Rank 0 sends a whole draft round as one request and every rank runs the round's further MTP passes on its own, instead of one network round trip per pass.
- `--tp-mtp-local-rows`: The other ranks read the MTP head's input rows from their own copy of the step output instead of receiving them from rank 0, saving about 42 MB per 1,024 prompt tokens. Measured: the gap between 4,096-token prompt steps 208 to 35 ms, 32K prefill 2,237 to 2,436 tok/s; same output.
- `--spec-fixed`: Always drafts `--spec-k` tokens: turns off the depth policy's acceptance adaptation and its batch gate. With `--spec-k 2` the depth stays fixed at two drafts on both ranks.
