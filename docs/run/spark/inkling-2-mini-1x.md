# Inkling-2-Mini: 1x DGX Spark

[Back to the run docs](../README.md), which say how these runs were measured.

Text only for now: no images and no DSpark / MTP drafter yet.

Inkling-2-Mini is a looped MoE (`inkling2_loop`): 24 physical layers, the middle 12 run three times (48 layer
slots), with sliding-window (1024) and global attention, 128 experts (8 picked per token), depth-mixed residuals
and per-pass controllers. KV: 196,608 bytes per token over all 48 slots, 32,768 bytes per token for the 8 global
slots plus a 1024-token ring for each sliding-window slot. A short token-convolution history per sequence makes the
model stateful, so prefix caching needs saved states.

## UD-Q4_K_XL GGUF

`Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf` from `unsloth/Inkling-2-Mini-GGUF` on Hugging Face (renamed from
`.gguf_file`, as the repo says). The root `Inkling-2-Mini-UD-Q4_K_XL.gguf` (Unsloth's own looped layout) loads too
and holds the same tensors.

```sh
legion serve \
  --model Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 65536 \
  --prefix-cache-states 0
```

## NVFP4 safetensors

`--model` is the downloaded `thinkingmachines/Inkling-2-Mini` directory, loaded as shipped: the NVFP4 experts
(layers 0-19) and the MXFP8 experts (layers 20-23) keep their bytes and scales. The vision tower and
`mtp.safetensors` are skipped.

```sh
legion serve \
  --model Inkling-2-Mini \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 65536 \
  --prefix-cache-states 0
```

## Chat

The chat template is the model's own (`tokenizer_config.json` / the GGUF's `tokenizer.chat_template`). Thinking
comes back as `reasoning_content`, tool calls as `tool_calls`. `reasoning_effort` goes to the template as sent
(`none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`; a number from 0 to 0.99 goes in
`chat_template_kwargs`); without it the template uses its own default, 0.9. The model card recommends temperature 1.0 with no truncation; the benchmarks use 0.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`65536`): Maximum tokens per sequence (prompt plus output). The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved token-convolution states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
