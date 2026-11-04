//! Sizing tensor parallelism's KV cache next to measured compute buffers: the largest
//! context, at most the asked one, whose cache fits beside the buffers measured at that same
//! context. Pure, the measurement injected (see `fit_tp_cache` in `lm.rs`).

use crate::model::core::{Executor, KV_BLOCK, StepPart};

/// What the scheduler may put in one step (see `LmEngine::step`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sched {
    /// Sequences decoding and prefilling at once.
    pub max_seqs: usize,
    /// Drafts per verify step (0 without speculation).
    pub spec_k: usize,
    /// The most prompt tokens one sequence takes per step (with rings, the ring step).
    pub chunk: usize,
    /// The most tokens per step (`--max-batch-tokens`).
    pub max_batch: usize,
    /// The most sequences prefilling at once.
    pub prefill_slots: usize,
}

/// The steps measured for context `ctx`, each as its groups in the scheduler's order (the
/// decode or verify group first, then one group per prompt chunk). No step gathers more keys
/// than one sequence at the full context (the executor's gather budget splits the others),
/// so these are the corners of that budget: one long sequence, or all of them sharing it; a
/// gather step's buffer grows with sequences × span and with tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Corners {
    /// Steps with more than one token per sequence (prompt chunks, verify batches): one kept
    /// graph's buffer.
    pub mixed: Vec<Vec<StepPart>>,
    /// Decode steps (one token per sequence): another.
    pub decode: Vec<Vec<StepPart>>,
    /// The MTP head's draft passes, `(sequences, span)` of one token each.
    pub drafts: Vec<(usize, usize)>,
}

/// The gather budget for context `ctx`: one sequence's span at the end of it, the verify
/// step's drafts included.
pub fn budget(ctx: usize, spec_k: usize) -> usize {
    Executor::gather_span(ctx + spec_k)
}

/// [`Corners`] of the steps `s` allows at context `ctx`.
pub fn corners(s: &Sched, ctx: usize) -> Corners {
    let budget = budget(ctx, s.spec_k);
    let pad = Executor::gather_span(1);
    let max_seqs = s.max_seqs.max(1);
    // The span each of `n` sequences has when they share `keys`.
    let share = |keys: usize, n: usize| (keys / n.max(1) / pad * pad).max(pad);
    let part = |n_seqs, n_tok: usize, span: usize, verify| StepPart {
        n_seqs,
        n_tok,
        n_past: span.saturating_sub(n_tok),
        verify,
    };
    let run = s.spec_k + 1;
    let verify = s.spec_k > 0;
    let batch = |n_seqs, span| part(n_seqs, run, span, verify);
    let chunk = s.chunk.min(s.max_batch).max(1);
    let mut c = Corners::default();
    // A prompt chunk alone at the end of the context.
    c.mixed.push(vec![part(1, chunk, budget, false)]);
    c.decode
        .push(vec![part(max_seqs, 1, share(budget, max_seqs), false)]);
    c.decode.push(vec![part(1, 1, budget, false)]);
    if verify {
        c.mixed.push(vec![batch(max_seqs, share(budget, max_seqs))]);
        c.mixed.push(vec![batch(1, budget)]);
    }
    // Prompt chunks: one, or as many as the batch and the prefill slots take, beside the
    // other sequences' verify (or decode) group.
    let most = s
        .max_batch
        .div_ceil(chunk)
        .min(s.prefill_slots.max(1))
        .min(max_seqs);
    for n_prompts in [1, most] {
        let others = max_seqs - n_prompts;
        let first = Executor::gather_span(chunk);
        let prompts = |span: usize| vec![part(1, chunk, span.max(chunk), false); n_prompts];
        let with = |others_span: usize, prompts: Vec<StepPart>| {
            let mut step = Vec::new();
            if others > 0 {
                step.push(batch(others, others_span));
            }
            step.extend(prompts);
            step
        };
        // The chunks at the end of the budget (sharing it) beside short sequences, or at
        // their start beside sequences sharing the rest.
        let rest = budget.saturating_sub(others * pad);
        c.mixed.push(with(pad, prompts(share(rest, n_prompts))));
        if others > 0 {
            let rest = budget.saturating_sub(n_prompts * first);
            c.mixed.push(with(share(rest, others), prompts(chunk)));
        }
    }
    c.mixed.dedup();
    c.drafts.push((max_seqs, share(budget, max_seqs)));
    c.drafts.push((1, budget));
    c
}

/// [`corners`] for a cache in sequence slots (`Executor::set_kv_slot_blocks`): nothing is
/// gathered there, so no budget splits the sequences' spans, and every sequence of a decode
/// step or a draft may read its own slot up to the full context. Those steps are measured with
/// every sequence at the full context: the shapes they reach, and past the sliding-window
/// layers' saturation (their windows' reads stop growing there), so the buffers grow with the
/// masks only. A shared span below the windows grew with the windows' reads instead, and the
/// line through it overestimated the buffers at long contexts several times over (#1159). The
/// mixed steps keep [`corners`]' shapes: their buffer is also the room that prompt steps'
/// copies of a sequence's span are sized to (`slot_stage_span` in `lm.rs`).
pub fn slot_corners(s: &Sched, ctx: usize) -> Corners {
    let mut c = corners(s, ctx);
    let budget = budget(ctx, s.spec_k);
    let n_seqs = s.max_seqs.max(1);
    c.decode = vec![vec![StepPart {
        n_seqs,
        n_tok: 1,
        n_past: budget.saturating_sub(1),
        verify: false,
    }]];
    c.drafts = vec![(n_seqs, budget)];
    c
}

/// A context the cache fits at, with what was measured at it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    /// The context, at most the asked one; the compute buffers were measured at it.
    pub ctx: usize,
    /// KV blocks: at least `ctx + spec_k` tokens' worth, at most `max_seqs` contexts'.
    pub n_blocks: usize,
    /// The compute bytes measured at `ctx`.
    pub compute: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FitError<E> {
    Measure(E),
    /// Not even two blocks' context fits: the last context tried and its compute bytes.
    NoRoom {
        ctx: usize,
        compute: usize,
    },
}

/// Measurements spent narrowing the context between one that fits and one that does not:
/// each halves the gap (on block boundaries), so the result is within a sixteenth of the
/// first gap (and a block) of the largest context that fits.
const BISECT_STEPS: usize = 4;

/// The largest context up to `asked` whose cache fits: `measure(ctx)` gives the compute bytes
/// of the steps kept at context `ctx`, `room(compute)` the bytes left for the KV cache next to
/// them, `block_bytes` what one block takes. Every candidate is measured at itself, so the
/// returned context is always one the buffers were measured at. From `asked` down, a context
/// that does not fit is lowered to what its cache left room for, at most halved (the buffers
/// shrink with the context: a lower one may leave more); once one fits, the gap to the
/// lowest that did not is narrowed by bisection.
pub fn fit_context<E>(
    asked: usize,
    spec_k: usize,
    max_seqs: usize,
    block_bytes: usize,
    room: impl Fn(usize) -> usize,
    mut measure: impl FnMut(usize) -> Result<usize, E>,
) -> Result<Fitted, FitError<E>> {
    let block_bytes = block_bytes.max(1);
    let mut try_ctx = |ctx: usize| -> Result<Result<Fitted, (usize, usize)>, E> {
        let compute = measure(ctx)?;
        let want = max_seqs.max(1) * (ctx + spec_k).div_ceil(KV_BLOCK);
        let n_blocks = (room(compute) / block_bytes).min(want);
        let holds = (n_blocks * KV_BLOCK).saturating_sub(spec_k);
        Ok(if holds >= ctx {
            Ok(Fitted {
                ctx,
                n_blocks,
                compute,
            })
        } else {
            Err((compute, holds))
        })
    };
    let down = |ctx: usize| ctx / KV_BLOCK * KV_BLOCK;
    let mut ctx = asked;
    // The lowest context that did not fit.
    let mut over = asked;
    let mut fit = loop {
        match try_ctx(ctx).map_err(FitError::Measure)? {
            Ok(f) => break f,
            Err((compute, holds)) => {
                over = ctx;
                let next = down(holds).max(down(ctx / 2)).min(down(ctx - 1));
                if next < 2 * KV_BLOCK {
                    return Err(FitError::NoRoom { ctx, compute });
                }
                ctx = next;
            }
        }
    };
    if fit.ctx == asked {
        return Ok(fit);
    }
    for _ in 0..BISECT_STEPS {
        let mid = down((fit.ctx + over) / 2);
        if mid <= fit.ctx {
            break;
        }
        match try_ctx(mid).map_err(FitError::Measure)? {
            Ok(f) => fit = f,
            Err(_) => over = mid,
        }
    }
    Ok(fit)
}

#[cfg(test)]
mod tests;
