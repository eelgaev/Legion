# Qwen3.8-27B NVFP4: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

`--model` is the downloaded snapshot of `unsloth/Qwen3.8-27B-NVFP4`, `--spec-draft` the downloaded snapshot of
`z-lab/Qwen3.8-27B-DFlash2` (BF16, read as shipped):

```sh
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
  --nvfp4-a4 prefill
```

Deterministic client: 8 fixed prompts (prose, code, lists, chats, a story; thinking off and on), temperature 0,
256 tokens each, 2 rounds at 1, 4 and 16 users. At 1 user the speed depends on how many drafts the prompt
accepts: 23.4 tok/s (story) to 51.1 (code), median 34.3, mean accepted length 3.45. At a context depth of
16K the 1-user median is 36.1, at 64K 22.8. Prefill 8,240 and 32,738 tokens, one request each, from the same
server. Without drafting (the command without the `--spec*`, `--fp8-sliced-k` and `--nvfp4-wide-mv` flags):
11.5 tok/s at 1 user, 41.3 at 4, prefill 2,205 / 1,925.

16 users: 124.9 tok/s total with this command (2026-11-03; the 1- and 4-user numbers are unchanged with
16 sequences, same 1-user tokens). The gate behind `--spec-max-batch` stops DFlash at 8 or more decoding
sequences, so those 16 users decode without drafts.

### 16 users and 8K prefill: two output-moving opt-ins

Add these to the command above:

```sh
  --spec-gate-cost-model off \
  --max-batch-tokens 16384 \
  --prefill-chunk 16384
```

With these flags (2026-11-03, one round each), 16 users decode at 229.5 tok/s total, against 124.9 without
them: DFlash keeps drafting at depth 7, mean accepted length 3.48. An 8,227-token prompt prefills in one step
at 2,282 tok/s and a 32,746-token prompt at 1,900, against 2,145 / 1,888 from the same binary without them.
A 122,528-token prompt prefills at 1,218 tok/s (2026-11-04, median of two servers: 1,223 / 1,212).
1 and 4 users decoded at 33.8 (same 1-user text) and 95.4. Greedy text at 8 or more users changes (other
drafts, other verify shapes), and so does a prompt longer than 4,096 tokens (other step sums). Below 8 users,
and for prompts up to 4,096 tokens, the output is the same as without the flags.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`262144`): Maximum tokens per sequence (prompt plus output). The model's full native context (also the default when the flag is left out). The KV cache is shared, so short requests do not reserve it.
- `--kv-cache-tokens` (`262144`): Size of the shared KV cache in tokens (all sequences plus the prefix cache). At least `--ctx-size`, so one request can use the full context; shorter requests share it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`): Speculative decoding drafter. `dflash` drafts a block of tokens in one pass of a small block-diffusion model that reads the target's hidden states; one verify step then checks them, so greedy output does not change.
- `--spec-draft` (`z-lab/Qwen3.8-27B-DFlash2`): The drafter: z-lab's DFlash 2 model for Qwen3.8-27B, a Hugging Face checkpoint read as shipped.
- `--spec-draft-weights` (`q4`): The drafter's matrices as 4-bit copies made at load, so a draft pass reads less. Drafts only: every token is verified, so the output does not change; only acceptance can.
- `--spec-draft-vocab` (`131072`): Drafts are scored over the first N vocabulary entries only, reading N rows of the output head. The verify step still uses the full vocabulary.
- `--spec-draft-lm-head` (`q4`): A 4-bit copy of those output head rows for drafting (the FP8 head's 131,072 rows: 671 to 377 MB a pass). Drafts only.
- `--fp8-sliced-k`: FP8 matmuls of 5-64 tokens (the verify steps) on a sliced-K tensor-core GEMM that streams the weights near the matrix-vector kernel's bandwidth. Output-moving, opt-in.
- `--nvfp4-wide-mv`: Dense NVFP4 matmuls of 33-64 tokens on the decode matrix-vector kernel instead of the GEMM. Output-moving, opt-in.
- `--fp8-act` (`a8-dynamic`): `a8-dynamic` quantizes the activations of FP8 matmuls to E4M3 per step, so prefill uses the FP8 tensor cores at about twice the BF16 throughput. Output-moving, opt-in.
- `--nvfp4-a4` (`prefill`): `prefill` runs every NVFP4 matmul of batches over 64 tokens with 4-bit activations on the GB10's block-scaled FP4 tensor cores. Faster prefill, output-moving, opt-in.
- `--spec-gate-cost-model` (`off`, opt-in above): lets DFlash keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks. Output-moving at 8 or more users.
- `--max-batch-tokens` / `--prefill-chunk` (`16384`, opt-in above): prompt tokens per prefill step (4,096 by default on the GB10). Fewer, larger steps; output-moving for prompts over 4,096 tokens.
