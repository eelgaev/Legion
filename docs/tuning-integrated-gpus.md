# Tuning speculative decoding on integrated GPUs (DGX Spark / GB10)

This page explains the speculative-decoding settings that made single-stream decode faster on the
NVIDIA GB10 (DGX Spark: 121 GB of memory shared by the CPU and GPU, ~233 GB/s of read bandwidth).
For each setting it says what it does, when it helps, when it does not, the measurements, and a
command line you can copy for each model.

**Defaults and recommendations.** On an integrated GPU (the GB10), Legion picks some of these
settings by itself; the rest are recommendations you pass yourself. Any value you give on the
command line wins.

| Setting | Default on an integrated GPU | Default elsewhere | Recommended on the GB10 |
|---|---|---|---|
| `--spec-k` | 8 for qwen4exp (Flash-Next) and for qwen35 / qwen35moe drafting with their MTP head | 2 for qwen4exp, 8 for Gemma 4, 4 otherwise | 4 for chat and prose, 8 when code or repetitive text dominates |
| `--mtp-window` | 32768 for qwen4exp's head | 0 (all positions) | 32768 |
| `--spec-draft-head-type` | `auto`: `q8_0` for qwen35 / qwen35moe heads and for a qwen4exp head shipped 16-bit (Flash-Next-NVFP4's own, #823), as shipped otherwise | `auto`: as shipped | `auto` |
| `--spec-draft-vocab` | **0 (all rows) on every device** | 0 | **65536** for MTP, **131072** for DFlash, as a flag |
| `--spec-draft-lm-head` | `auto`: `q4` for the qwen4exp MTP head when the output head is 16-bit (Flash-Next-NVFP4, #823), `model` otherwise | `auto`: `model` | `auto`; **`q4`** on Qwen3.8-27B (DFlash and MTP), as a flag |

## Why it matters on the GB10

Decoding one token reads every weight it uses once. On the GB10 that read time is most of the
step: Qwen3.8-Flash-Next (UD-Q4_K_XL) reads ~6.2 GB per token (26 ms of a 38 ms step), the dense
Qwen3.8-27B (NVFP4) ~21 GB (81 ms). Speculative decoding (`--spec mtp`) lets the model's own
MTP head guess several tokens, and the model then checks them all in one step. Each accepted
guess is a token that did not need a step of its own.

Three things decide the gain:
- how many tokens a step can check (`--spec-k`);
- how much each guess costs, i.e. one pass of the head (`--spec-draft-vocab`, the Q8_0 head copy,
  `--mtp-window`);
- how often the guesses are right. This depends on the text: code and repeated text are easy,
  prose is hard.

None of these settings changes what the model outputs. Only the guesses change; every token is
checked against the full model. (On a GPU, the text can still part from plain decoding after a
while: a step that checks several tokens computes in batches and rounds slightly differently.
This happens with or without these settings.)

## The settings, in plain words

| Setting | What it does | Helps when | Does not help when |
|---|---|---|---|
| `--spec-k N` | The most tokens the head may guess per step. The depth policy picks the actual number each step from measured costs and how often guesses are accepted; N is only the ceiling. | Text is predictable (code, lists, repeating or quoting text), and checking more tokens is cheap: dense models, where a step reads the same weights for 1 row or 9. | Prose and chat with thinking. The policy then stays at 3-5 guesses, and a higher ceiling changes nothing or costs a little. On MoE models (Flash-Next) each extra checked token also reads its own experts. |
| `--spec-draft-vocab N` | The head scores only the first N tokens of the vocabulary when guessing (the common ones), so it reads N rows of the output head instead of all 248,320. Guesses only: the output is unchanged. | The head's output layer is a big part of a guess. 65,536 is the sweet spot on both Qwen models with MTP; 131,072 with a DFlash drafter that borrows the target's head (NInfer's shortlist size). | Too small a set (32,768 and below) cuts acceptance more than it saves. |
| `--spec-draft-lm-head q4` | The MTP heads of qwen4exp and qwen35, and DFlash drafters that borrow the target's output head: makes a 4-bit copy (Q4_K, the head's scales folded in) of the head rows they score with (all, or the first `--spec-draft-vocab`) at load, and drafts with that. Guesses only: checking uses the target's own head, so the output is unchanged. After NInfer's 4-bit shortlist head. | The head is big next to the drafter: Qwen3.8-27B's FP8 head over 131,072 rows is 671 MB, 377 MB as Q4_K. With z-lab's DFlash 2 the identical-text chats went from 28.8 / 23.3 / 25.5 to 30.4 / 25.0 / 25.8 tok/s; with the MTP head (65,536 rows, 320 -> 180 MiB per guess) from 25.3 / 28.9 / 21.6 to 26.6 / 30.1 / 21.6. | The drafter has its own head (a `d2t` vocabulary): nothing to do. Flash-Next: its Q8_0 head is small next to its MoE step, and the chats did not move. |
| `--spec-draft-head-type q8_0` | Makes a Q8_0 copy (~half the size) of the head's BF16/F16 matrices at load (~1 s) and guesses with that. Guesses only: checking uses the model as shipped, so the output is unchanged. | The head ships in BF16 (Qwen3.8-27B-NVFP4: 810 -> 430 MiB per guess). Acceptance did not change measurably. | The head is already quantized (Flash-Next's `mtp-...-shared-Q8_0.gguf`): nothing to gain. |
| `--mtp-window N` | The head looks back at the last N positions only. Guesses only. | Long contexts on Flash-Next: at 64K tokens, 32768 lifts decode from 34-35 to 40-42 tok/s. | Short contexts (below N): no effect. |
| `--spec-mtp-ngram N` with `--spec-mtp-ngram-k K` | Where the last N or more tokens occurred earlier in the conversation, the continuation of that earlier occurrence is checked instead of the head's guesses, up to K tokens (the head still guesses at most `--spec-k`). It needs no head passes. The depth policy picks it only where it is expected to give more tokens per millisecond. Not byte-identical: other guesses in a check step can flip near-tied tokens (#327). | The output copies its input: repeating, quoting or editing text. On Flash-Next (GB10), N=4 and K=16 took recall from 74 to 135 tok/s, editing code from 78 to 113 and quoting a 13K-token document from 77 to 138 (`2026-10-21_qwen3.8-flash-next_mtp-ngram-387`). Opt-in. | Prose, code written from scratch, and chats: these rarely repeat 4+ tokens of their context, so speed stays the same (45 tok/s on the chats). |
| `--expert-mv-device` (was `LEGION_EXPERT_MV_DEVICE=1`) | Flash-Next (and other MoE models with experts in GPU memory): check steps run the experts' gate and up projections in one kernel. Same bits per layer; generated text can differ from the first multi-token verify step (layout-dependent ggml-cuda fusions, #410 / #472), so greedy output is not byte-identical to the default. | A few % on the check step's MoE (-13% of its gate/up time): end to end about 2.7% shorter check step at a fixed k=2, a tie at k=4 and with the adaptive depth (`2026-10-20_qwen3.8-flash-next_expert-mv-device-ab`). Opt-in. | Only check steps of 2-8 tokens; plain decoding is unchanged. |

## Measurements

All on DGX Spark (GB10, driver 580, CUDA 13), greedy, single stream. The runs (kept
outside this repository) are named next to each table.

### Qwen3.8-27B (RadixArk/Qwen3.8-27B-NVFP4, the checkpoint's own MTP head)

`2026-10-19_qwen3.8-27b-nvfp4_legion-qwen35-mtp`. 256 tokens per prompt. The prose / code /
list / recall prompts run with thinking off; the "chat-" prompts and the 3 x 384-token chats with
thinking on (their output is reasoning text). tok/s:

| Settings | cycle | prose | code | list | recall | chat-prose | chat-code | chat-list | chats (3 x 384) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| no speculation | | 12.3 | 12.3 | 12.3 | 12.3 | 12.3 | 12.2 | 12.2 | 12.3 / 12.3 / 12.3 |
| k=4 | 117-127 ms | 25.4 | 33.1 | 27.4 | 41.8 | 28.6 | 24.8 | 22.7 | 26.5 / 22.2 / 23.6 |
| k=6 | 117-139 ms | 25.0 | 34.3 | 27.9 | 50.6 | 25.4 | 24.2 | 23.9 | 25.6 / 22.4 / 23.8 |
| k=4, draft vocab 65,536 | 108-113 ms | 26.2 | 33.7 | 29.8 | 42.7 | 29.4 | 32.0 | 27.6 | 28.2 / 22.5 / 24.0 |
| k=4, draft vocab 32,768 | 107-119 ms | 24.0 | 33.9 | 25.6 | 37.8 | 26.8 | 23.8 | 20.8 | 27.3 / 24.5 / 20.6 |
| k=4, 65,536, Q8_0 head | 102-106 ms | 28.8 | 37.2 | 31.6 | 45.1 | 28.5 | 27.5 | 28.1 | 30.3 / 23.4 / 26.1 |
| k=8, 65,536, Q8_0 head | 107-120 ms | 29.6 | 44.6 | 29.6 | 73.2 | 30.8 | 26.6 | 24.9 | 28.4 / 27.6 / 24.9 |
| k=12, 65,536, Q8_0 head | 104-130 ms | 29.2 | 45.8 | 32.0 | 75.3 | 27.9 | 26.7 | 25.5 | 29.7 / 26.5 / 24.2 |

Reading the table:
- The draft vocabulary and the Q8_0 head make every guess cheaper: the cycle drops from ~120 to
  ~104 ms at k=4, and every prompt class gains.
- With cheap guesses, a deeper ceiling pays on code (37 -> 45) and recall (45 -> 73-75).
- It does **not** pay on lists and the thinking chats. Lists 31.6 -> 29.6 at k=8 (32.0 at k=12);
  chat-code 27.5 -> 26.6 and chat-list 28.1 -> 24.9 at k=8. These moves are within the ±2 tok/s
  run-to-run noise or slightly worse.
- `legion bench` c=1 (512-token prompt, 256 out, its own request defaults): 12.2 -> 26.6 (k=2),
  31.6 (k=3), 34.0 (k=4), 34.2 (k=6), all with the full vocabulary and the BF16 head.
- Prefill with the head's take-in: 2,648 tokens 927 -> 878 tok/s, 10,425 tokens 773 -> 723 (k=4).

### Qwen3.8-Flash-Next (unsloth UD-Q4_K_XL + `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`)

**Draft depth** (`2026-10-19_qwen3.8-flash-next_legion-deep-draft`). Draft vocabulary
65,536 and `--mtp-window 32768` in every row; the default `--spec-p-min` 0.5, plus 0.6 and 0 (no
cutoff). 256 tokens; tok/s (accepted length, cycle):

| Settings | prose | code | list | recall | chat-prose | chat-code | chat-list |
|---|---:|---:|---:|---:|---:|---:|---:|
| no speculation | 25.5 | 25.8 | 25.8 | 26.5 | 25.9 | 25.7 | 25.8 |
| k=4 | 43.9 (2.52, 58 ms) | 51.9 (3.34, 65 ms) | 46.3 (3.14, 68 ms) | 64.7 (4.62, 72 ms) | 47.3 (3.11, 66 ms) | 42.5 (2.84, 69 ms) | 37.2 (2.37, 69 ms) |
| k=4, p-min 0.6 | 45.2 (2.83, 63 ms) | 52.5 (3.30, 63 ms) | 48.4 (3.00, 62 ms) | 64.7 (4.38, 68 ms) | 50.4 (3.34, 67 ms) | 42.6 (2.68, 64 ms) | 44.7 (2.80, 63 ms) |
| k=4, p-min 0 | 45.0 (2.83, 63 ms) | 52.4 (3.30, 63 ms) | 48.5 (3.00, 62 ms) | 62.8 (4.16, 67 ms) | 48.6 (3.23, 66 ms) | 44.0 (2.83, 64 ms) | 40.4 (2.63, 65 ms) |
| k=6 | 44.8 (2.83, 63 ms) | 53.3 (3.45, 65 ms) | 47.5 (2.93, 62 ms) | 71.3 (5.64, 80 ms) | 46.5 (3.27, 70 ms) | 43.1 (2.92, 69 ms) | 40.3 (2.66, 67 ms) |
| k=6, p-min 0.6 | 44.7 (2.83, 63 ms) | 52.7 (3.39, 64 ms) | 48.5 (3.00, 62 ms) | 66.5 (4.72, 71 ms) | 47.0 (3.34, 71 ms) | 41.3 (2.69, 66 ms) | 42.5 (2.79, 67 ms) |
| k=6, p-min 0 | 43.5 (2.52, 58 ms) | 53.5 (3.45, 64 ms) | 49.4 (3.04, 61 ms) | 67.8 (4.98, 74 ms) | 46.0 (3.26, 71 ms) | 42.5 (2.83, 67 ms) | 39.1 (2.55, 65 ms) |
| k=8 | 44.7 (2.83, 63 ms) | 52.1 (3.45, 66 ms) | 47.2 (2.93, 62 ms) | 70.5 (5.64, 80 ms) | 46.1 (3.23, 70 ms) | 42.4 (2.91, 70 ms) | 39.4 (2.62, 67 ms) |
| k=8, p-min 0.6 | 45.0 (2.83, 63 ms) | 54.5 (3.75, 69 ms) | 50.0 (3.27, 65 ms) | 71.2 (6.22, 87 ms) | 44.4 (3.14, 71 ms) | 41.1 (2.60, 65 ms) | 39.7 (2.57, 66 ms) |
| k=8, p-min 0 | 43.5 (2.60, 60 ms) | 56.0 (3.92, 70 ms) | 48.9 (3.36, 69 ms) | 70.7 (5.29, 75 ms) | 45.1 (3.23, 72 ms) | 41.8 (2.77, 66 ms) | 37.5 (2.26, 60 ms) |
| k=12 | 43.8 (2.63, 60 ms) | 55.1 (4.32, 78 ms) | 51.2 (3.53, 69 ms) | 71.4 (6.54, 92 ms) | 46.1 (2.84, 62 ms) | 40.1 (2.58, 64 ms) | 40.6 (2.62, 65 ms) |
| k=12, p-min 0.6 | 44.6 (2.83, 64 ms) | 54.2 (3.64, 67 ms) | 48.6 (3.00, 62 ms) | 67.3 (5.43, 81 ms) | 48.8 (3.26, 67 ms) | 41.4 (2.77, 67 ms) | 41.2 (2.70, 66 ms) |
| k=12, p-min 0 | 44.8 (2.83, 63 ms) | 51.2 (3.45, 67 ms) | 50.4 (3.11, 62 ms) | 68.7 (5.54, 81 ms) | 46.1 (3.02, 66 ms) | 41.0 (2.77, 68 ms) | 42.4 (2.82, 67 ms) |

Reading the table:
- On this MoE model a deeper ceiling helps recall (65 -> 71) and code a little (52 -> 54-56).
- It does **not** help prose, lists or the thinking chats: they stay within noise, and chat-list
  is lower at k=8 (39.4-37.5 vs 44.7 in one k=4 run). Each extra checked token reads ~10 experts
  of its own (~1.5 GB), so the cycle grows 6-9 ms per extra guess.
- `--spec-p-min` 0.6, 0.5 and 0 make no consistent difference: the depth policy lowers the cutoff
  from measured costs anyway.
- Without speculation Flash-Next decodes at 25.8 tok/s. Most of the remaining step time is reading
  the checkpoint's Q8_0 dense layers and output head (~4.75 GB per token); no flag changes that.

**Draft vocabulary** (`2026-10-18_qwen3.8-flash-next_legion-gb10-defaults-qsa-mma`,
dv-* runs, 3 chats x 384 tokens, thinking on):

| `--spec-k` | `--spec-draft-vocab` | chats tok/s | accepted length |
|---|---|---|---|
| 4 | all | 38.8 / 37.1 / 37.0 | 2.78 |
| 4 | 65,536 | 40.8 / 40.5 / 41.8 | 2.79 |
| 4 | 32,768 | 39.1 / 36.8 / 37.2 | 2.51 |
| 4 | 16,384 | 39.0 / 35.6 / 34.0 | 2.31 |
| 2 | all | 39.0 / 36.5 / 37.5 | 2.34 |
| 2 | 65,536 | 42.1 / 38.1 / 38.5 | 2.29 |
| 3 | 65,536 | 41.3 / 38.3 / 38.9 | 2.45 |

**MTP window** (same folder, g8 -> g9, 64K-token contexts, k=4, draft vocabulary 65,536):
decode at 2,048 / 8,192 new tokens 34.1 / 35.1 -> 42.4 / 40.0 tok/s with `--mtp-window 32768`
(8,192 gave 39.8 / 39.1).

**The owner's grid** with k=4, draft vocabulary 65,536, window 32768 and `--expert-mv-device`
(`2026-10-19_qwen3.8-flash-next_legion-decode-verify`, llama-benchy, temperature 0):
mean decode 41.7 tok/s over pp 2K/8K x depth 0/16K/64K (Strata on the same box: 40.0), against
40.6 for k=4 without the expert option.

### Qwen3.8-27B (unsloth/Qwen3.8-27B-NVFP4, FP8 head) with z-lab's DFlash 2 drafter

`2026-10-27_qwen3.8-27b-nvfp4-unsloth_665-dflash2-head`. `--spec dflash --spec-draft
<z-lab/Qwen3.8-27B-DFlash2> --spec-draft-weights q8_0`; the drafter borrows the target's FP8
output head (248,320 rows, 1.27 GB). 3 greedy chats x 384 tokens with thinking on, then
llama-benchy pp 512 / tg 256. tok/s:

| Draft head | chats, same text (`--deterministic`) | chats, mean over runs | C=1 (mean) | C=4 (mean) | accepted length (same text) |
|---|---|---:|---:|---:|---:|
| all rows, FP8 | 28.8 / 23.3 / 25.5 | 25.7 (4 runs) | 25.2 | 63.8 | 3.17 |
| first 131,072, FP8 | 29.4 / 23.8 / 25.3 | 26.1 (2 runs) | 27.0 | 67.3 | 3.36 |
| first 131,072, `q4` | 30.4 / 25.0 / 25.8 | 26.7 (4 runs) | 26.3 | 65.5 | 3.26 |

Reading the table:
- The three `--deterministic` runs emit the same text (hash fa25400400): the head only changes
  the guesses. On that text, the shortlist plus Q4 head is 4.6% faster over the three chats.
- Without `--deterministic` it is 3.9% faster, in every run: chat 1 30.2-30.4 vs 28.9-29.1 tok/s.
- llama-benchy's C=1 and C=4 move within their noise. What it generates differs from run to
  run, and single runs of one setting range from 23.0 to 31.3 (C=1) and from 55.3 to 73.1 (C=4).
- Prefill is unchanged (8K ~1,090, 32K ~1,018 tok/s): the head is not used there.

With the opt-in matmul switches as well (`2026-10-27_qwen3.8-27b-nvfp4-unsloth_689-stack`,
the det suite: 8 fixed prompts, temperature 0, 256 tokens; per-request median at C=1,
total at C=4): `--spec-draft-vocab 131072 --spec-draft-lm-head q4 --fp8-sliced-k --nvfp4-wide-mv
--fp8-act a8-dynamic --nvfp4-a4 prefill` gives 32.5 / 89.3 tok/s, against 27.0 / 80.0 with
`--spec-draft-weights q8_0` alone. `--spec-draft-weights q4` instead of `q8_0` makes it 33.7 /
93.3, with the same text. `--fp8-small-batch-gemv` on top costs 1-3% (31.5 / 90.3): it takes
the 5-9-token verify steps from the sliced-K GEMM. The prefill switches move the logits (see
their entries).

### Qwen3.8-27B (unsloth/Qwen3.8-27B-NVFP4, FP8 head) with its MTP head

`2026-10-27_qwen3.8-27b-nvfp4-unsloth_665-mtp-q4-head`. `--spec mtp --spec-draft-vocab
65536`, with and without `--spec-draft-lm-head q4` (the head's 65,536 FP8 rows, 320 MiB, as a
180 MiB Q4_K copy). 3 greedy chats x 384 tokens with thinking on; tok/s:

| Draft head | chats, same text (`--deterministic --spec-p-min 0`) | chats, 3 runs | C=1 (runs) | C=4 (runs) |
|---|---|---|---|---|
| FP8 rows | 25.3 / 28.9 / 21.6 | 25.2/21.8/23.6, 25.6/29.2/23.3, 24.9/22.0/22.6 | 22.5, 20.9, 22.6 | 63.8, 67.5, 61.2 |
| `q4` | 26.6 / 30.1 / 21.6 | 26.9/30.0/23.8, 26.9/23.2/24.2, 26.7/30.8/24.0 | 26.1, 21.7, 23.3 | 61.7, 65.0, 68.2 |

- Both deterministic runs emit the same text (hash 472916cb5b): +5% and +4% on chats 1 and 2,
  and chat 3 is unchanged. Without `--deterministic`, chat 1 is 24.9-25.6 -> 26.7-26.9 tok/s
  and chat 3 22.6-23.6 -> 23.8-24.2. Chat 2's text differs between runs (22 or 29-31 tok/s).
- The same-text check needs `--spec-p-min 0`. With the default 0.5, the drafts the head stops at
  depend on its probabilities, so the verify sizes, and with them the GPU's rounding, differ.
- llama-benchy C=1 / C=4 / C=16 (one full-suite run: 63.8 -> 67.8) move within their noise;
  prefill is unchanged (8K ~1,120, 32K 1,038 tok/s).

## Recommended command lines (GB10)

Qwen3.8-Flash-Next, UD-Q4_K_XL. `--spec-k 8` and `--mtp-window 32768` are what an integrated GPU
picks anyway; they are spelled out here. Use `--spec-k 4` for chat and prose:

```bash
legion serve \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --device cuda --expert-mv-device \
  --ctx-size 131072 --kv-cache-tokens 131072 \
  --spec mtp --spec-draft mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-k 8 --spec-draft-vocab 65536 --mtp-window 32768
```

Add `--spec-mtp-ngram 4 --spec-mtp-ngram-k 16` when requests repeat, quote or edit their input
(it changes nothing measurable on other text).

Qwen3.8-27B, RadixArk/Qwen3.8-27B-NVFP4 (the safetensors folder; the head is in the checkpoint).
`--spec-k 8` and the Q8_0 draft head are what an integrated GPU picks; `--spec-k 4` suits lists
and chat better:

```bash
legion serve \
  --model ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/<rev> --device cuda \
  --ctx-size 32768 --kv-cache-tokens 65536 \
  --spec mtp --spec-k 8 --spec-draft-vocab 65536 --spec-draft-head-type q8_0
```

With unsloth/Qwen3.8-27B-NVFP4 (FP8 output head), add `--spec-draft-lm-head q4`.

Qwen3.8-27B, unsloth/Qwen3.8-27B-NVFP4 (FP8 head) with z-lab's DFlash 2 drafter:

```bash
legion serve \
  --model ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/<rev> --device cuda \
  --ctx-size 32768 --kv-cache-tokens 131072 \
  --spec dflash --spec-draft ~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/<rev> \
  --spec-draft-weights q4 --spec-draft-vocab 131072 --spec-draft-lm-head q4
```

On the GB10 always give `--kv-cache-tokens` explicitly: the KV cache shares memory with the
weights, the OS and the page cache.

The load drops the model files' pages from the page cache on an integrated GPU, before it
allocates the weights and again before the KV cache and step buffers (#836). With 20-35 GiB of
checkpoint pages left in memory, free memory sat near the kernel's watermarks: the weights and
caches got scattered pages, and reclaim and compaction kept running during requests. About one
server start in three then ran an 8K-token Qwen3.8-Flash-Next prompt in 4.4-5.4 s instead of
4.2 s, every prompt kernel slower, for the server's whole life. `--keep-load-cache` keeps the
pages (for comparisons); the output is the same either way.

## What these numbers do not cover

- One stream only. At several concurrent streams the steps get compute-bound, and speculation
  stops at `--spec-max-batch` (8) decoding sequences by default. With `--spec mtp`, raising it
  (e.g. `--spec-max-batch 64`) lets the measured-cost depth keep drafting: Qwen3.8-Flash-Next on
  one GB10 went 81.6 -> 100.1 tok/s at 8 users and 107-109 -> 146-147 at 16 (#827). `--spec-gate-cost-model off`
  does the same for MTP alone (n-gram drafters keep the gate; see docs/serving.md).
- Other GPUs. On a discrete GPU with much faster memory the balance between guess cost and check
  cost differs; measure before copying these settings.
- Sampled (temperature > 0) requests accept fewer guesses than greedy ones.
