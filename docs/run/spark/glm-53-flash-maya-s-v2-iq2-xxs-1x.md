# GLM-5.3-Flash Maya-S v2 (IQ2_XXS): 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

The MTP layer is inside the GGUF:

```sh
legion serve \
  --model GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 131072 \
  --uma-free-gib 8 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-k 2 \
  --spec-gate-cost-model off
```

- Decode (C=1 26.3 total, 27.5 per request, 24-29 over the eight prompts; C=4 42.5; C=16 69.3): deterministic
  client at C=1/4/16, 2 rounds, one server.
- Prefill (8K 689, 32K 633, 123K 498): one request each from the same server, after the decode rounds. The
  bench's 8192/32768/122880-token prompts are 5,746/22,950/85,878 GLM tokens.
- Without `--spec-gate-cost-model off` (no output-moving flags): C=1 26.2, C=4 42.3, C=16 49.7; prefill 661 / 611
  / 493.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`131072`): Maximum tokens per sequence (prompt plus output). Covers the 123K prefill request (about 86K GLM tokens). The KV cache is shared, so short requests do not reserve it.
- `--uma-free-gib` (`8`): Memory the KV pool leaves free for the OS beside the step buffers on a GPU that shares the host's memory (default 16). The weights and the MTP head take 89.8 GiB, which leaves about 26 GiB; with the default no context fits beside 16 sequences' states and the step buffers.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-k` (`2`): Maximum draft tokens per sequence and step. The depth policy drafts fewer when drafts do not pay off.
- `--spec-gate-cost-model` (`off`, opt-in): lets MTP keep drafting at `--spec-max-batch` (8) or more decoding sequences, at the depth its step-cost model picks (16 users 49.7 -> 69.3 tok/s). Output-moving at 8 or more users; 1 and 4 users and prefill are unchanged.
