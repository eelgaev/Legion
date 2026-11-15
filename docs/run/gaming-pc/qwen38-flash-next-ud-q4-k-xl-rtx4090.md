# Qwen3.8-Flash-Next UD-Q4_K_XL: gaming PC, one RTX 4090

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

One RTX 4090 (24 GB, PCIe 4.0) with a desktop session on it (1.2 GB of VRAM used), a Ryzen 9 7950X3D and 128 GB
of RAM. The model's experts do not fit the card: Legion keeps them in RAM, caches the hot ones in VRAM by routing
heat (saved between runs) and swaps them while it decodes. The tier switches on by itself when the experts do not
fit. `--cpu-expert-share auto` is opt-in: the CPU computes part of the experts that miss VRAM, with the same bits as
the GPU (the output does not change), and the share per layer follows the measured GPU and CPU times.

```sh
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --port 8090 \
  --device cuda \
  --max-seqs 4 \
  --ctx-size 40960 \
  --prefix-cache-states 0 \
  --ngram-io direct \
  --cpu-expert-share auto
```

`--ngram-io direct` reads the model's n-gram table from the SSD instead of holding it in RAM.

64 and 32 GB: the same command on the same PC, run in a systemd scope with `MemoryMax=58G` / `27G` and no swap, with
the host tier sized to the limit (`LEGION_DEBUG_HOST_MEM_GIB=58` / `32:27`; on a machine with that much RAM it sizes
itself).

Measured 2026-11-15. Decode is the tok/s of three single-user chats (prose, code, a list; temperature 0) right after the
load; prefill is one request each of 8,192 and 32,768 tokens, tok/s to the first token, in the same run. 4 and 16
users and 123K were not measured. Every chat is byte-identical to the run without `--cpu-expert-share`.

| RAM | Decode, 3 chats | Prefill 8K | 32K |
|---|---|--:|--:|
| 128 GB | 60.5 / 59.9 / 60.8 | 2,709 | 4,616 |
| 64 GB | 46.3 / 46.8 / 48.7 | 1,660 | 2,336 |
| 32 GB | 25.6 / 24.5 / 24.1 | 888 | 1,215 |

Strata v0.2.13.1 on the same PC: [other-engines.md](../other-engines.md#qwen38-flash-next).
