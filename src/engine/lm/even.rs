//! `--prefill-even` (#758): a prompt that needs several steps takes them in equal parts instead
//! of full steps and a short last one. Each step of a big MoE reads every routed expert, so a
//! last step of a few dozen tokens costs about what a full one's weights cost (two DGX Sparks
//! under `--tp-rank`, Qwen3.8-Flash-Next: an 8,240-token prompt was 4096 + 4096 + 48 tokens).

/// The tokens a prompt with `left` tokens to go takes in a step that allows `take`: with
/// `left > take`, the fewest steps of at most `take` that hold `left`, made equal and rounded
/// up to `align` tokens (never past `take`); else `take` (the prompt ends in this step).
pub(crate) fn even_take(left: usize, take: usize, align: usize) -> usize {
    if take == 0 || left <= take {
        return take;
    }
    let steps = left.div_ceil(take);
    let even = left.div_ceil(steps);
    even.next_multiple_of(align.max(1)).min(take)
}

#[cfg(test)]
mod tests;
