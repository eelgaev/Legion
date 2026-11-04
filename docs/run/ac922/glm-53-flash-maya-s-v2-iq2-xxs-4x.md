# GLM-5.3-Flash Maya-S v2 (IQ2_XXS): IBM AC922 4x V100 SXM2 16 GB

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Decode (C=1 ~64, C=4 ~71, C=16 ~84) and prefill (8K ~1,300, 32K ~1,300, 123K ~1,205); the MTP layer is inside the GGUF:

```sh
legion serve \
  --model GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
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
  --glm-prompt-step 8192 \
  --glm-decode-mv \
  --spec-gate-cost-model off \
  --kv-vmm-keep 32768
```

Deterministic client, two rounds: C=1 per request 63.9 / 64.8 (61.3-67.9 over the 8 prompts), C=4 total 74.0 /
68.8, C=16 total 84.4 / 83.6. The expert cache is 37.6 GiB (26.2 without `--kv-vmm-keep`). With
`--kv-vmm-keep 16384`, 16 users get 75.0 without `--spec-gate-cost-model off` and 18.6 without `--glm-decode-mv` too. Prefill, one request per size
(the GLM tokenizer makes the prompts 5,746 / 22,950 / 85,878 tokens): 8K 1,299, 32K 1,302, 123K 1,205. The 8K and
32K prompts fit the kept 32,768 tokens; the 123K one maps the rest of the KV pool when it starts (0.55 s; 1,233
without the three opt-in flags).

## What each flag does

- `--gpus` (`4`): Splits the layers over this many GPUs, one pipeline stage each. Used because the model does not fit in one 16 GB V100.
- `--experts` (`cache`): Where a MoE model's routed experts live. `cache` keeps them in host memory and caches the most used ones in VRAM; used because the experts are far larger than the GPUs' VRAM, and on the AC922 the GPUs read host memory over NVLink 2.0.
- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`131072`): Maximum tokens per sequence (prompt plus output); covers the 123K prefill request. The KV cache is shared by the sequences and sized to one context (3.8 GiB per GPU here, since every GPU keeps every layer's latent cache); the rest of the VRAM goes to the expert cache. With the native 1048576 (the default when the flag is left out) the KV cache would take half of each GPU's free VRAM, so fewer experts would be cached.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`mtp`): Speculative decoding drafter. `mtp` uses the model's own multi-token-prediction head to draft tokens that one verify step then checks, which raises decode speed without changing greedy output.
- `--spec-k` (`2`): Maximum draft tokens per sequence and step. The depth policy drafts fewer when drafts do not pay off.
- `--glm-parallel-decode`: GLM-5.3-Flash on several GPUs: every GPU works on every token (tensor parallel) instead of one pipeline stage at a time, each with its own heads, FFN rows and cached experts.
- `--glm-parallel-mtp`: With `--glm-parallel-decode`: the MTP head's drafts also run on every GPU instead of only the last one.
- `--glm-pair-balance`: Within each NVLink GPU pair, an expert that is not cached goes to whichever GPU has less work in that step, so the exchanges wait less for the slowest GPU. Measured 60.5 to 63 tok/s at 1 user.
- `--glm-prompt-step` (`8192`): With `--glm-parallel-decode`: tokens per parallel step of a prompt (default 2048, at most 8192). Each GPU stages its uncached experts of a layer once a step, so bigger steps read long prompts faster: 32K prefill 1,118 at 2048, 1,288 at 8192. Costs 4 GiB of pinned host memory per GPU (512 KiB a token), and a sequence decoding beside a long prompt waits for whole steps.
- `--glm-decode-mv`: Plain decode steps of up to 64 sequences run through the expert matvec, which reads each selected expert once, instead of the fused MoE op, which reads the uncached experts in place over the host link, slow for steps of more than 8 rows (verify steps already take the matvec). Each sequence decodes to the bits it gets alone, so greedy output at more than 8 sequences differs from the command without this flag.
- `--spec-gate-cost-model` (`off`): Past `--spec-max-batch` (8 sequences) the MTP head keeps drafting and its cost model picks the depth, instead of drafting stopping there.
- `--kv-vmm-keep` (`32768`): Only the first 32,768 tokens of each GPU's KV pool keep VRAM at load; the rest (2.9 GiB a GPU) goes to the expert cache. A prompt past them maps the rest again when it starts (its long-prompt lend frees the VRAM) and gives it back once no sequence uses it.
