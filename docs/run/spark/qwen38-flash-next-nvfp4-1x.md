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

### The table's row: three output-moving opt-ins

Add these to the command above:

```sh
  --spec-gate-cost-model off \
  --nvfp4-expert-rows-bf16 \
  --hc-bf16
```

`--spec-gate-cost-model off` alone (2026-11-03, the same session as the numbers above): 16 users decode at
172.1 tok/s total, against 121.0 without the flag; MTP keeps drafting at the depth its step-cost model picks,
mean accepted length 3.07. 1 user 34.8 (story) to 57.7 (code), median 41.9, mean accepted length 3.13;
4 users 92.0. Prefill 8,240 / 32,738 / 122,528 tokens: 2,024 / 2,065 / 1,946 tok/s. Greedy text at 8 or
more users changes (other drafts, other verify shapes).

All three (2026-11-12): prefill 8,240 / 32,738 / 122,528 tokens at 2,353 / 2,450 / 2,276 tok/s, against
2,146 / 2,205 / 2,071 with `--spec-gate-cost-model off` alone from the same build in the same session (the
rest of the gain since 2026-11-03 is the default prompt-step kernels, same output). The two prefill flags
leave decode alone: 1 user 36.2 (story) to 58.6 (code), median 43.9; 4 users 91.7; 16 users 172.9 (without
them 34.8-58.5, 94.2 and 179.5; 4 and 16 users swing a few percent between rounds). Both round prompt-step
values to BF16, so greedy text can move after prompts of over 64 tokens. The table's row is this
2026-11-12 run, decode and prefill.

- The opt-in prefill numbers (8K 1,749 / 32K 1,800) add `--nvfp4-a4 prefill --prefill-even` to the command
  without `--spec` (an earlier build).

### Images

The checkpoint ships its vision tower (`model.visual.*`), and the command above loads it: requests may carry
images, and `/v1/models` lists `vision`. The tower runs on the GPU (2026-11-19): a 1024 × 768 image (768
image tokens) encodes in 0.35-0.44 s and its first token comes 1.1-1.4 s after the request (11.8-12.0 s and
12.4-12.7 s on the CPU tower, `LEGION_DEBUG_VISION_CPU=1`); 800 × 600 takes 0.15 s (TTFT 0.6-0.8 s),
640 × 480 0.08 s (0.4-0.6 s). An image sent again takes its rows from a cache (TTFT 0.55-0.73 s). Text-only
requests give the same output as with `--no-mmproj`, at the same speed (8K prefill 2,316 tok/s against
2,323, 1 user 41.3 tok/s against 40.8).

An encode leaves room for the other users' decode: its kernels run on capped grids (one GEMM block per SM).
With 4 users decoding, they keep about 72% of their rate while a 1024 × 768 image encodes (56% with one block
per tile), their longest stall drops from 0.37 s to 0.24 s, and the encode takes 0.75 s instead of 0.53 s
(TTFT 1.44 s against 1.22 s). The image's own prefill step (0.6 s for its 805 tokens) still pauses them.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--kv-cache-tokens` (`262144`): Size of the shared KV cache in tokens (all sequences plus the prefix cache). At least `--ctx-size`, so one request can use the full context; shorter requests share it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output. The NVFP4 checkpoint ships its head, so no `--spec-draft` file is needed.
- `--spec-draft-vocab` (`65536`): The MTP head scores only the first N vocabulary entries while drafting, reading N rows of the output head instead of all of them. Drafting gets cheaper; the verify step still checks against the full vocabulary, so output does not change.
- `--spec-gate-cost-model` (`off`, opt-in above): lets MTP keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
- `--nvfp4-expert-rows-bf16` (opt-in above): on prompt steps of more than 64 tokens, the routed experts' down projection stores each (token, expert) row as BF16 instead of F32 and the weighted sum reads it back (here the hyper-connection norm the MoE output folds into reads them), half the bytes of those rows. Output-moving: each row is rounded to BF16 before the sum. Decode and verify steps keep F32 rows.
- `--hc-bf16` (opt-in above): on prompt steps of 256 or more tokens on the GB10, the hyper-connection residual streams stay BF16 between the mixers instead of F32, so the combine-and-norm and the up-mix kernel move half the stream bytes. Math stays F32 inside each kernel; each new stream value is rounded to BF16, so output moves. Decode, verify and mixed steps keep F32 streams.
- `--max-batch-tokens` (opt-in, not in the table's row): prompt tokens per step, 4096 by default here (`--prefill-chunk` follows it). Beside the 262,144-token pool, three kept prompt graphs' compute buffers do not fit at bigger steps, so the load keeps fewer and logs `prompt steps keep fewer graphs`: two at `8192` (about 4.4 GiB each at the end of the context, against three of 2.2 GiB at 4K steps), one at `12288` (about 6.7 GiB). With one kept graph the next chunk's graph is built after the current chunk instead of during it. 16K steps do not fit beside this pool. Against the table's row command (one Spark, 2026-11-23, a fresh server per arm): `12288` prefill 8K 2,710 -> 2,897-2,965 tok/s (+7-9%), 32K 2,785 -> 2,968-2,978 (+7%), 64K 2,698 -> 2,887 (+7%); `8192` +4.7% / +4.5% / +5.1%. Fixed-prompt decode with `12288` ties (1 user 42.7 -> 43.1, 4 users 93.1 -> 93.0, 16 users 173.4 -> 174.4), and about 1.2 GB more memory stays free after load than at 4K steps. Output-moving: prompts over 4,096 tokens run in other step sizes (other sums), so greedy text can change.
- `--dense-fp8` (opt-in, not in the table's row yet): the checkpoint ships its dense projections (attention, DeltaNet and shared-expert weights) in BF16; this keeps an FP8 E4M3 copy of them, made at load with one F32 scale per output row (`amax / 448`, llm-compressor's `FP8_DYNAMIC` weights), and decode and verify steps of up to 64 tokens read the copy with 16-bit activations: half the bytes those steps stream for the dense projections. Prompt steps keep the BF16 weights, with the same kernels and output as without the flag; a step that batches prompt rows with decode rows reads BF16 for all of them. The one exception is a prompt chunk of one token per sequence (64 tokens or fewer in all): it has a decode step's shape and reads the copy. The copy takes about 2.7 GiB more memory; the MTP layer gets none and drafts from its BF16 weights. Output-moving: decoded tokens come from 8-bit weights. One machine, without `--tp`. Against the command above (one Spark, 2026-11-23): fixed-prompt decode 43.2 -> 47.3 tok/s at 1 user (+9.5%), 93.4 -> 96.1 at 4 users, 172.9 -> 171.5 at 16 users; prefill unchanged. Without MTP, the first token after each of 40 prompts is the same as without the flag, and 100 exact-answer tasks score 92% (94% without).
- `--dense-fp8-prompt` (opt-in, with `--dense-fp8`, not in the table's row yet): prompt steps read the FP8 copies too, as W8A8 GEMMs on the GB10's FP8 tensor cores with one FP8 scale per token. The copies of query + gate / key / value and of the shared expert's gate / up are stored back to back and run as one GEMM each, as their BF16 weights do; the activations' 8-bit codes come from the BF16 rows the hyper-connection mixers and the DeltaNet output norm write anyway, or from the ops that make the attention output and the shared expert's SwiGLU. Output-moving: prompt activations are rounded to 8 bits too. Against the command above (one Spark, 2026-11-23, a fresh server per arm): prefill 8K 2,716 -> 2,872 tok/s (+5.7%), 32K 2,779 -> 2,951 (+6.2%), 64K 2,694 -> 2,859 (+6.1%). Without MTP, the first token after 40 prompts of 0.3-8K tokens matches 86% of the time (mean KL 0.040 nats over the top 20; `--nvfp4-a4 prefill` gives 91% and 0.026), and 100 exact-answer tasks score 92% (94% without).
