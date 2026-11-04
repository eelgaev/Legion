//! `--mtp-moe-fused-rows` (#858): the routed experts of MTP draft passes on the fused expert op.
//!
//! An MTP pass runs its head's MoE on the output rows only, one per drafting sequence, so at
//! 16 sequences its steps are 1-16 tokens: ggml's MMVQ up to 8 and MMQ above, while the target's
//! verify steps (16 x 9 rows) take the fused op ([`super::moe_ffn`], on a GB10 its fragment
//! kernels). With a row floor set, head passes of at least that many rows take the fused op too
//! (where it supports the head's experts and they are in device memory, as for the target).
//!
//! Opt-in: the fused op computes in FP16 on tensor cores where MMQ quantizes the activations to
//! Q8_1, so the head's picks can differ and with them the drafts and the acceptance. The target
//! verifies every draft, but the steps' sizes follow the acceptance, so greedy output can move.

use std::sync::atomic::{AtomicI64, Ordering};

use super::MoeKernel;

#[cfg(test)]
mod tests;

/// The row floor (0: off, the head's passes keep the target's rule).
static ROWS: AtomicI64 = AtomicI64::new(0);

/// Sets the fewest rows of an MTP pass whose routed experts take the fused op, for graphs built
/// from now on (0, the default: off).
pub fn set_rows(rows: usize) {
    ROWS.store(rows as i64, Ordering::Relaxed);
}

/// The configured floor ([`set_rows`]).
pub fn rows() -> i64 {
    ROWS.load(Ordering::Relaxed)
}

/// Whether layer `il` of a model with `n_layer` target layers is an MTP head's block (the
/// heads Legion loads number their block `n_layer`).
pub fn is_head(il: usize, n_layer: usize) -> bool {
    il >= n_layer
}

/// The kernel choice for an MTP head's MoE of `n` tokens: `target` (the rule for the target's
/// steps) unless `kernel` is auto, `floor` is set and `n` reaches it, with CUDA built in.
pub fn choose(
    kernel: MoeKernel,
    target: Option<MoeKernel>,
    n: i64,
    floor: i64,
    cuda: bool,
) -> Option<MoeKernel> {
    if target.is_none() && kernel == MoeKernel::Auto && cuda && floor > 0 && n >= floor {
        Some(MoeKernel::Auto)
    } else {
        target
    }
}
