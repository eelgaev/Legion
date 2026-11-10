# Gemma 4 31B NVFP4 (RedHatAI): 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `RedHatAI/gemma-4-31B-it-NVFP4`, `--spec-draft` the downloaded snapshot of
`RedHatAI/gemma-4-31B-it-speculator.dflash2` (both read as shipped):

```sh
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
  --spec-draft-batched-attn
```

Deterministic client: 8 fixed prompts (prose, code, lists, chats, a story; thinking off and on), temperature 0,
256 tokens each, 2 rounds at 1, 4 and 16 users (2026-11-10). At 1 user the speed depends on how many drafts
the prompt accepts: 22.0 tok/s (story) to 39.4 (code), 28.7 in all (median), mean accepted length 3.30, the
same tokens in both rounds (the same as without `--spec-draft-batched-attn`). 83.9 tok/s at 4 users. 16 users
swing a few percent between rounds: 243.8 is the median of 8 rounds in three sessions that day (234.1-247.0);
without `--spec-draft-batched-attn`, same sessions, 235.4 (231.2-240.5). Prefill 8K 2,041, 32K
1,630 and 123K (122,880 tokens) 907 tok/s, one request each, from the same server.

The 2026-11-06 session, before `--spec-draft-batched-attn` and the drafter step changes that came with it:
22.1-39.5 at 1 user (29.0 median), 81.0 at 4 users, 235.1 at 16, prefill 8K 1,845, 32K 1,540, 123K 874. The
build before `--head-rope-fused`
(and the captured-rows change that came with it), same session, without the flag: 21.9-40.2 at 1 user (28.3
median), 81.9 at 4 users, 221.9 at 16, prefill 8K 1,727, 32K 1,458, 123K 848.

An earlier session (2026-11-03) without the last five flags (all of them opt-in): 18-33 tok/s at 1 user (24.4 median,
accepted length 3.36), 70.2 at 4 users, 154.3 at 16, prefill 8K 897 and 32K 710 (123K 587). `--nvfp4-a4 prefill` alone
gives most of the prefill and 16-user gain (16 users ~187); the 8-bit drafter and its 4-bit head make each
draft pass cheaper (16 users ~212 with both). `--spec-draft-vocab 131072` on top cuts the accepted length
to ~2.3 and is slower at every concurrency.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`): Speculative decoding drafter. `dflash` drafts a block of tokens in one pass of a small block-diffusion model that reads the target's hidden states; one verify step then checks them, so greedy output does not change.
- `--spec-draft` (`RedHatAI/gemma-4-31B-it-speculator.dflash2`): The drafter: RedHatAI's DFlash 2 speculator for Gemma 4 31B, a Hugging Face checkpoint read as shipped.
- `--spec-gate-cost-model` (`off`): lets DFlash keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
- `--nvfp4-a4` (`prefill`, opt-in): runs every NVFP4 matmul of batches over 64 tokens with 4-bit activations on the GB10's block-scaled FP4 tensor cores: prompt steps, and the 128-token verify steps of 16 users. The RedHatAI checkpoint is calibrated for 4-bit activations (vLLM runs it W4A4). For Gemma 4, q, k and v share one activation quantization with the attention norm folded into it, and gate, up and the GELU run as one kernel with the FFN norm folded in. Faster prefill and 16-user decode, output-moving (prompts and batches of over 64 tokens).
- `--spec-draft-weights` (`q8_0`, opt-in): the drafter's BF16 layers (~4.8 GB read per draft pass) quantized to Q8_0 at load. Drafts only, every token is still verified by the full model; at 4 and 16 users the drafts shape the verify batches, so the text can move.
- `--spec-draft-lm-head` (`q4`, opt-in): the drafter reads a 4-bit copy of Gemma's tied 262,144-row head (2.8 GB in BF16) made at load. Drafts only, as above.
- `--head-rope-fused` (opt-in): every layer's RoPE reads one shared view of the positions, so ggml-cuda runs the q and k heads' norm, weight and RoPE as its one fused kernel instead of two. Prefill ~4% faster at 8K and 32K, 16 users a few percent. Output-moving: the fused kernel rounds differently.
- `--spec-draft-batched-attn` (opt-in): when several sequences draft together, every drafter layer attends all of them in one flash attention over a shared pool of their drafter caches, instead of a few small kernels per sequence (about 800 a step at 16 users). 16 users ~3.5% faster. Drafts only, as above; one sequence drafts exactly as without it. The pool is allocated once, when the first sequence drafts: one slot per sequence the server can hold, each sized for a full drafter cache (here 20 slots, ~3.8 GB, about what the per-sequence caches would use at full load). If it can't be allocated, or its slots run out, the server warns and falls back to per-sequence caches.
