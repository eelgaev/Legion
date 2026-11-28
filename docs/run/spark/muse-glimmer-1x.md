# Muse Glimmer 30B UD-Q4_K_XL: 1x DGX Spark

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

Muse Glimmer 30B (`muse-glimmer`) is a dense 28B model: 52 layers, three sliding-window (2048, RoPE) layers to
one global (NoPE) layer, 2 KV heads, QK norms and a sigmoid gate on the attention output. `--model` is
`Muse-Glimmer-30B-UD-Q4_K_XL.gguf` from `unsloth/Muse-Glimmer-30B-GGUF` on Hugging Face.

```sh
legion serve \
  --model Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 131072 \
  --prefix-cache-states 0
```

One round of the deterministic client (2026-11-28): 1 user 13.0 tok/s on every prompt, 4 users 50.8, 16 users
157.4; prefill 927 / 890 / 776 tok/s at 8.5K / 33.7K / 126K tokens. Muse's tokenizer makes about half as many
tokens as Qwen's from the benchmark's prompt text, so the prompts were sized to give Muse 8K / 32K / 123K-class
token counts (the counts above are the server's own).

## With DFlash

`--spec-draft` is `dflash-kquant.gguf` from the same Hugging Face repo (the model's DFlash drafter: 5 layers,
blocks of 16):

```sh
legion serve \
  --model Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 16 \
  --ctx-size 131072 \
  --prefix-cache-states 0 \
  --spec dflash \
  --spec-draft dflash-kquant.gguf
```

One round of the deterministic client (2026-11-28, same prompts): 1 user 16.9 (chat-explain) to 24.1 tok/s
(code), 23.3 median, mean accepted length 2.46; 4 users 51.9; 16 users 138.4 (157.4 without drafting, so leave
it off for many users); prefill 927 / 896 / 780 tok/s at 8.5K / 33.7K / 126K tokens, the same as without it.
Greedy text can differ from the run without drafting at near ties, since the verify steps run the batched
matmuls.

## Chat

The chat template is the model's own (the GGUF's `tokenizer.chat_template`). The model writes its reasoning to
itself first and then answers the user; Legion returns the reasoning as `reasoning_content`, the answer as
`content` and tool calls as `tool_calls`, streamed or not. The model always reasons; how much is set by the
template's `reasoning_strength` (`high` by default; `low` gives a few lines of reasoning), sent in
`chat_template_kwargs`:

```json
{"messages": [...], "chat_template_kwargs": {"reasoning_strength": "low"}}
```

`reasoning_effort` and `enable_thinking` do not change this template's output.

## What each flag does

- `--max-seqs` (`16`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`131072`): Maximum tokens per sequence (prompt plus output). The KV cache is shared, so short requests do not reserve it; the sliding-window layers keep only their 2048-token rings.
- `--prefix-cache-states` (`0`): Saved states for prefix caching (here the sliding-window rings); `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`, DFlash command): Speculative decoding drafter. `dflash` drafts a block of tokens in one pass of a small block-diffusion model that reads the target's hidden states; one verify step then checks them.
- `--spec-draft` (`dflash-kquant.gguf`, DFlash command): The drafter: the DFlash model shipped in the same Hugging Face repo as the target.
