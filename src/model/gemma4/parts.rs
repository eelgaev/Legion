//! The parts of a Gemma 4 prompt step's output projection under tensor parallelism over
//! machines (#1181), for [`super::Model::gemma4_layers_halves`]: half 0's tokens in two parts,
//! a third and the rest, then half 1 whole.
//!
//! With the halves, half 0's attention-output sum has only half 1's output projection (and its
//! sum's start) to hide behind before half 0's FFN needs it: over two DGX Sparks (an 8K-class
//! prompt step of 4,096 tokens, BF16 on the wire) about 0.7 ms of each sliding-window layer's
//! ~1.5 ms transfer is left in sight, most of the GPU's idle time in the step. With half 0's
//! projection in two parts, the first part's sum (a third of half 0's bytes) travels while the
//! second part's projection runs, and the second's while half 1's runs.
//!
//! Every part's work is per token: the output projection's W4A4 GEMM at every part's batch
//! (more than 64 tokens, the same kernel and tiles, [`crate::model::nvfp4_a4::tokens_tile`]),
//! and each sum gives every value the plain all-reduce's bits
//! ([`crate::tp_net::micro::all_reduce_start`]), so the output is the halves' bit for bit.

#[cfg(test)]
mod tests;

/// Tokens the smallest part needs ([`mixer_plan`]): the W4A4 GEMM's batches of more than 64
/// tokens run the same kernel and tiles; a smaller part would add launches for little wire time.
pub(super) const TP_PART_MIN: i64 = 512;

/// The most output-projection parts a step takes ([`parts`]).
pub(super) const TP_PARTS_MAX: usize = 3;

/// Parts by default ([`parts`]): half 0 in two, half 1 whole.
const TP_PARTS: usize = 3;

/// The most output-projection parts a step's layers take: `LEGION_DEBUG_GEMMA4_TP_PARTS` (2 the
/// halves, as before #1181; up to [`TP_PARTS_MAX`]), else [`TP_PARTS`]. Read once.
pub(super) fn parts() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        parts_of(
            std::env::var("LEGION_DEBUG_GEMMA4_TP_PARTS")
                .ok()
                .as_deref(),
        )
    })
}

/// [`parts`] for the variable's value `v`.
pub(super) fn parts_of(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .map_or(TP_PARTS, |p| p.clamp(2, TP_PARTS_MAX))
}

/// The output projection's parts of a step of `n` tokens with at most `parts` of them,
/// `(first token, tokens)` in token order, half 1 (`n - n / 2` tokens) last and whole: with
/// three, half 0 (`n / 2` tokens) in `n / 2 / 3` and the rest, unless that first part would be
/// under [`TP_PART_MIN`] tokens (then the halves). The same answer on every rank.
pub(super) fn mixer_plan(n: i64, parts: usize) -> Vec<(i64, i64)> {
    let (n0, n1) = (n / 2, n - n / 2);
    let a = n0 / 3;
    if parts < 3 || a < TP_PART_MIN {
        return vec![(0, n0), (n0, n1)];
    }
    vec![(0, a), (a, n0 - a), (n0, n1)]
}

/// One step of [`mixer_order`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MixerOp {
    /// Part `j`'s output projection, its sum started right behind it.
    Proj(usize),
    /// Part `j`'s sum finished (half 0's parts, into one tensor).
    Finish(usize),
}

/// The order of the output projection's `k` parts ([`mixer_plan`], the last one half 1): each
/// part's matmul and start, the oldest exchange finished before a matmul whose start would make
/// a third in flight (so that no finish lies between a matmul and its start), and half 0's
/// parts finished at the end. Half 1's sum is not finished here (it travels during half 0's
/// FFN). Two parts give the halves' order: both starts, then half 0's finish.
pub(super) fn mixer_order(k: usize) -> Vec<MixerOp> {
    let mut out = Vec::new();
    let mut oldest = 0;
    for j in 0..k {
        if j - oldest >= 2 {
            out.push(MixerOp::Finish(oldest));
            oldest += 1;
        }
        out.push(MixerOp::Proj(j));
    }
    while oldest + 1 < k {
        out.push(MixerOp::Finish(oldest));
        oldest += 1;
    }
    out
}
