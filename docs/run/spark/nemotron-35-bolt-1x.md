# Nemotron 3.5 Bolt (E4B, 74B-A9B): 1x DGX Spark

[Back to the run docs](../README.md), which say how these runs were measured.

## Bolt E4B UD-Q4_K_XL

Bolt E4B is a Gemma 4 E4B (`gemma4`): 42 layers, sliding-window and global attention, per-layer token
embeddings, the last 18 layers reading earlier layers' KV.

```sh
legion serve \
  --model NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 4 \
  --ctx-size 40960 \
  --prefix-cache-states 0
```

DFlash (opt-in): the same plus the drafter from the same repo.

```sh
legion serve \
  --model NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 4 \
  --ctx-size 40960 \
  --prefix-cache-states 0 \
  --spec dflash \
  --spec-draft dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf
```

The BF16 checkpoint loads as shipped: `--model` is the downloaded `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`
directory (the text model; the shared-KV layers' unused `k_proj` / `v_proj` / `k_norm` are skipped).

One round of the deterministic client (2026-11-07): 1 user 51.2 tok/s, 4 users 191.2; prefill 3,942 / 3,668
tok/s at 8K / 32K. With DFlash: 1 user 59.7 (mean accepted length 2.08), 4 users 147.9, so it only pays at 1
user; DFlash verify steps run other shapes than plain decode, so long greedy answers can drift after a few dozen
tokens.

## Bolt 74B-A9B UD-Q4_K_XL

Bolt 74B-A9B is a Nemotron-H MoE (`nemotron_h_moe`, 88 layers: Mamba-2, 8 attention layers, MoE) whose MoE
blocks differ: 240 to 496 experts and up to 9 picked per token, read from each block's router and the per-layer
`expert_used_count`. The routed experts run in a 1024-wide latent space (`ffn_latent_down` before them,
`ffn_latent_up` after); the router and the shared expert read the full 4096-wide input.

```sh
legion serve \
  --model NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00001-of-00003.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 4 \
  --ctx-size 40960 \
  --prefix-cache-states 0
```

One round of the deterministic client (2026-11-07): 1 user 23.2 tok/s, 4 users 61.3; prefill 1,183 / 1,165
tok/s at 8K / 32K. The weights take 47.9 GB, all on the GPU.

## What each flag does

- `--max-seqs` (`4`): How many requests are decoded at once (continuous batching). Set to the highest concurrency the benchmark uses.
- `--ctx-size` (`40960`): Maximum tokens per sequence (prompt plus output), room for the 32K prefill. The KV cache is shared, so short requests do not reserve it.
- `--prefix-cache-states` (`0`): Saved recurrent states for prefix caching; `0` turns prefix caching off. Measurement only, so repeated benchmark prompts are not answered from the cache. Drop it for normal use.
- `--spec` (`dflash`, opt-in): Speculative decoding drafter. `dflash` drafts a block of tokens in one pass of a small block-diffusion model that reads the target's hidden states; one verify step then checks them. Output-moving: the verify step runs other shapes than plain decode.
- `--spec-draft` (`dflash-NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf`): The drafter: unsloth's Q8_0 GGUF of Bolt E4B's DFlash model (5 layers, blocks of 8 tokens), in the same Hugging Face repo as the model.
