# GLM-5.3-Flash UD-Q4_K_XL: IBM AC922 4x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

The unsloth UD-Q4_K_XL GGUF (141 GB, 6 shards); the MTP layer is inside the GGUF:

```sh
legion serve \
  --model GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --port 8090 \
  --device cuda \
  --gpus 4 \
  --experts cache \
  --max-seqs 16 \
  --ctx-size 131072 \
  --prefix-cache-states 0 \
  --spec mtp \
  --spec-k 2 \
  --glm-parallel-decode \
  --glm-parallel-mtp \
  --glm-pair-balance \
  --glm-decode-mv \
  --spec-gate-cost-model off \
  --kv-vmm-keep 32768
```

Deterministic client at C=1/4/16, two rounds: C=1 per request 41.1 / 41.9 (38.6-43.6 over the 8 prompts), C=4 total
44.3 / 41.0, C=16 total 55.6 / 55.2; prefill 8K 1,198, 32K 1,171, 123K 1,120 (one request each, 5,746 / 22,950 / 85,878
prompt tokens, same command). The expert cache is 35.2 GiB. Without the last three flags: C=1 38.8, C=4 39.5, C=16
11.0, prefill 1,200 / 1,169 / 1,132, a 24.0 GiB cache (the 123K prompt maps the rest of the KV pool when it starts,
0.55 s).

## What each flag does

- `--gpus` (`4`): Splits the model over this many GPUs. Used because the model does not fit in one 16 GB V100.
- `--experts` (`cache`): Where a MoE model's routed experts live. `cache` keeps them in host memory and caches the most used ones in VRAM; used because the experts (about 130 GB) are far larger than the GPUs' 64 GB, and on the AC922 the GPUs read host memory over NVLink 2.0.
- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`131072`): Maximum tokens per sequence (prompt plus output); covers the 123K prefill request. The KV cache is shared by the sequences and sized to one context (every GPU keeps every layer's latent cache); the rest of the VRAM goes to the expert cache. With the native 1048576 (the default when the flag is left out) the KV cache would take half of each GPU's free VRAM, so fewer experts would be cached.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks.
- `--spec-k` (`2`): Maximum draft tokens per sequence and step.
- `--glm-parallel-decode`: GLM-5.3-Flash on several GPUs: every GPU works on every token (tensor parallel) instead of one pipeline stage at a time, each with its own heads, FFN rows and cached experts.
- `--glm-parallel-mtp`: With `--glm-parallel-decode`: the MTP head's drafts also run on every GPU instead of only the last one.
- `--glm-pair-balance`: Within each NVLink GPU pair, an expert that is not cached goes to whichever GPU has less work in that step, so the exchanges wait less for the slowest GPU.
- `--glm-decode-mv`: Plain decode steps of up to 64 sequences run through the expert matvec, which reads each selected expert once, instead of the fused MoE op, which reads the uncached experts in place over the host link, slow for steps of more than 8 rows (verify steps already take the matvec). Each sequence decodes to the bits it gets alone, so greedy output at more than 8 sequences differs from the command without this flag.
- `--spec-gate-cost-model` (`off`): Past `--spec-max-batch` (8 sequences) the MTP head keeps drafting and its cost model picks the depth, instead of drafting stopping there.
- `--kv-vmm-keep` (`32768`): Only the first 32,768 tokens of each GPU's KV pool keep VRAM at load; the rest (2.9 GiB a GPU) goes to the expert cache. A prompt past them maps the rest again when it starts (its long-prompt lend frees the VRAM) and gives it back once no sequence uses it.
