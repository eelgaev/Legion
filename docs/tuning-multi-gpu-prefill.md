# Long prompts on several discrete GPUs: layer split and stage balance

This page covers how a long prompt runs over a pipeline of discrete GPUs (one stage per GPU), where
its time goes, and the settings that change it. The measurements are Qwen3.8-Flash-Next
(UD-Q4_K_XL, `qwen4exp`) on an IBM AC922 with four V100s (NVLink 2.0 to the host), `--experts
cache`, `--spec mtp --spec-k 2`, `--max-seqs 16`, `--ctx-size 131072`, greedy requests (#462).

**Recommendation.** Keep the defaults: the automatic split and 4,096-token micro-chunks.

| Setting | Default | Recommended | Why |
|---|---|---|---|
| `--layer-split` | automatic (balances the bytes per layer): `12,24,36` for Flash-Next on 4 GPUs | the default | It is the best split at whole-layer granularity (measured below), and the split Strata-AC922 runs (`12,24,36` with `STRATA_SPLIT_BALANCE=1` and its stage trim; Legion's stage trim is #266) |
| `--long-pipeline-micro` | `4096` | the default | 6,144 gives the same pipeline time |
| `--long-micro-even` | off | off, unless output may move | +2-3% at 123K, but other chunk boundaries give other (equally valid) greedy bytes |

## How a long prompt runs

A prompt with at least `--long-prompt-tokens` (2,048) tokens left borrows the expert cache's VRAM.
Read alone, it goes through the stages in steps of up to `--long-alone-tokens` tokens, cut into
micro-chunks of `--long-pipeline-micro` tokens (`LongPrompts`, `src/engine/lm.rs`). Each round,
every stage runs the chunk it holds and passes it on, and the round ends when the slowest stage is
done (`Executor` round loop, `src/model/core/exec.rs`; the
`legion::model::core::exec=debug` log prints a `step round` line per round with each stage's
`stage_done_ms`). A 123K prompt on 4 GPUs is 31 micro-chunks in 34 rounds:

- **Fill and drain.** The first 3 and the last 3 rounds have idle stages: 6 rounds, 2.9-3.0 s of a
  ~18 s pipeline. A stage-round never takes less than ~300 ms, because each micro-chunk stages all
  of its stage's experts over the host link (~19 GB at ~64 GB/s). So with these chunk sizes the
  bubble is (stages - 1) rounds of the largest chunk at either end. Changing the chunk size does
  not remove it: 6,144-token chunks give 24 rounds of 779 ms (median) and 4.1 s of fill and drain,
  and the same 18.3 s pipeline.
- **Steady rounds.** The median full round is 533-540 ms: stage 0 is done at 517-524 ms, stages 1-3
  at 490-504 ms, and the round adds 13-15 ms of host work. Stage 0 is the slowest stage in 25-27 of the
  28 full rounds.

## Why stage 0 is the slowest

Every stage holds 12 layers: 3 sparse-attention layers and 9 DeltaNet layers. Per 4,096-token
chunk, an attention layer takes ~64-68 ms and a DeltaNet layer ~30-35 ms (from moving one layer
between stages, below). Only stage 0 also runs the token embedding, layer 0's hyper-connection
setup and the per-layer n-gram embedding in layer 1 (gathered rows, three grouped norms, the gate
and the dilated conv over 10,240-wide rows). An nsys trace of a 123K prompt gives GPU 0 0.7 s
more kernel time than GPUs 1 and 2 over the prompt: the extra is in elementwise ops (+258 ms), copies
(+138 ms), norms, `concat`, `get_rows` and the hyper-connection ops: ~20 ms per round. The output
head and the MTP head are on the last stage; that stage is not the limit.

Since #462 the n-gram embedding's conv reads a prompt chunk's taps as strided views of the history
and the chunk in token order, instead of transposing the chunk and then each tap back (verify
steps and one-token steps keep the old graph). Stage 0's median drops by 3-4 ms per round with
the same bits, which is within run-to-run noise on the 123K figure.

Perfect balance would put every stage at ~503 ms, at most ~3% off the round. The rest of the
gap to Strata-AC922 (~7,700 tok/s against Legion's 6,600-6,700 at the 123K peak) is per-layer time, not
balance.

## The split

At whole-layer granularity the default split is the best one:

| `--layer-split` | Stage done medians (ms) | Pipeline | 123K tok/s (2nd prompt) |
|---|---|---|---|
| `12,24,36` (default; Strata-AC922's) | 517-524, 495-504, 493-502, 490-499 | 17.7-18.4 s | 6,176-6,445 |
| `12,24,37` | 521, 500, 529, 462 | 18.58 s | 6,172 |
| `11,23,35` | 451, 494, 491, 561 | 19.62 s | 5,870 |

Moving a DeltaNet layer off stage 0 (30-35 ms) overshoots the ~20 ms imbalance, so the next stage
becomes the limit. Strata's own scorer picks `11,24,36` on this machine, 10% slower than
`12,24,36` there; Legion's byte balance picks `12,24,36`.

## Chunk size and `--long-micro-even`

| Setting | Rounds | Full-round median | Fill + drain | Pipeline | 123K tok/s (2nd prompt) |
|---|---|---|---|---|---|
| default (4,096) | 34 | 533-540 ms | 2.9 s | 17.7-18.4 s | 6,176-6,445 |
| `--long-pipeline-micro 6144` | 24 | 779 ms | 4.1 s | 18.29 s | 6,262 |
| `--long-micro-even` | 34 | 535 ms | 3.0 s | 17.91 s | 6,383 |

`--long-micro-even` spreads the prompt evenly over the chunks instead of leaving a short last one,
so the last rounds carry real work. It is not a default because other chunk boundaries change the
greedy bytes.

## Measuring

- The first long prompt after a start is ~10% slower (21.0-21.9 s against 18.3-19.7 s for the next
  ones): judge long-prompt throughput by the third 123K prompt of a run (the "peak"), with prompts
  that differ so the prefix cache does not serve them.
- Byte-identical checks: `--deterministic --expert-cache-gib <n>` fixes the expert-cache slot
  count, so both runs get the same slots.
