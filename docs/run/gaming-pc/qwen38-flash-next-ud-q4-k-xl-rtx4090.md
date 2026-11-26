# Qwen3.8-Flash-Next UD-Q4_K_XL: gaming PC, one RTX 4090

[Back to the performance tables](../../../README.md#performance). How these runs were measured: [docs/run/README.md](../README.md).

One RTX 4090 (24 GB, PCIe 4.0) with a desktop session on it (1.3-1.6 GB of VRAM used), a Ryzen 9 7950X3D and 128 GB of DDR5-4800
RAM. The model's experts do not fit the card: Legion keeps them in RAM, caches the hot ones in VRAM by routing
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
  --cpu-expert-share auto \
  --spec mtp \
  --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-k 1
```

`--ngram-io direct` reads the model's n-gram table from the SSD instead of holding it in RAM.

`--spec mtp` drafts with the model's own multi-token-prediction head and checks the draft in the next step;
`--spec-draft` is the head's weights (the `mtp-*.gguf` files in the same Hugging Face repo; the Q4_K_M one is 0.9 GB
smaller than the Q8_0 one, which leaves that VRAM to the expert cache). `--spec-k 1` drafts one token a step: each
extra token in a step reads its own experts, most of them from RAM here, and `--spec-k 2` was no faster.
Greedy chats come out byte-identical to the command without the three `--spec` flags.

64 and 32 GB: the same command without the three `--spec` flags (`--spec mtp` does not load yet when the experts
do not all fit in RAM: the head's layer is not part of the bounded host tier), on the same PC, run in a systemd
scope with `MemoryMax=58G` / `27G` and no swap, with the host tier sized to the limit
(`LEGION_DEBUG_HOST_MEM_GIB=58` / `32:27`; on a machine with that much RAM it sizes itself).

Measured 2026-11-26. Decode is the tok/s of three single-user chats (prose, code, a list; temperature 0) right after the
load; prefill is one request each of 8,192 and 32,768 tokens, tok/s to the first token, in the same run. 4 and 16
users and 123K were not measured. Every chat is byte-identical to the run without `--cpu-expert-share` and, at
128 GB, without the `--spec` flags. 128 GB: two runs; 64 and 32 GB: one each.

| RAM | Decode, 3 chats | Prefill 8K | 32K |
|---|---|--:|--:|
| 128 GB (MTP) | 69.1 / 66.6 / 69.6, 68.8 / 66.5 / 70.0 | 3,257 / 3,258 | 5,318 / 5,326 |
| 64 GB | 47.5 / 48.3 / 49.7 | 2,101 | 2,942 |
| 32 GB | 27.2 / 26.8 / 26.5 | 1,312 | 1,271 |

Strata v0.2.13.1 on the same PC: [other-engines.md](../other-engines.md#qwen38-flash-next).
