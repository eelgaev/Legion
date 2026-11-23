//! `--dense-fp8` (#1183): decode-sized steps read the FP8 copies of a Qwen3.8-Flash-Next
//! checkpoint's dense projections ([`crate::hf::dense_fp8`]) in place of their BF16 weights.
//!
//! [`super::Model::w`] gives the copy's codes for the weight's name, and its row scales for
//! the weight's `.scale` (which a BF16 weight has none of), while a graph of at most
//! [`STEP_MAX_TOKENS`] tokens is built ([`DenseTwins::step`]). Every path that reads a weight
//! by name then sees an FP8 weight with a per-row scale, as for a channel-wise FP8 checkpoint:
//! the BF16 paths step aside and the FP8 matmul's W8A16 route runs it
//! ([`super::linear::Linear::mul_mat`]). Larger steps (prompt steps) and everything outside a
//! graph build read the BF16 weights, so they keep every kernel, joined run and fusion they
//! take without the flag. Without copies in the file nothing changes.

use std::cell::Cell;

use crate::gguf::GgufFile;
use crate::hf::dense_fp8;

/// The largest steps that read the copies: decode and verify steps (16 sequences of 4 tokens),
/// [`super::linear::NVFP4_A4_MIN_N`], the size up to which FP8 matmuls without a static input
/// scale take 16-bit activations.
pub(crate) const STEP_MAX_TOKENS: i64 = super::linear::NVFP4_A4_MIN_N;

/// Whether graphs being built read the FP8 copies ([`Self::step`]).
#[derive(Debug, Default)]
pub(crate) struct DenseTwins {
    /// The file has copies ([`dense_fp8::KEY`]).
    on: bool,
    /// A graph of a decode-sized step is being built.
    small: Cell<bool>,
}

impl DenseTwins {
    pub(crate) fn from_gguf(g: &GgufFile) -> Self {
        Self {
            on: g.get(dense_fp8::KEY).is_some(),
            small: Cell::new(false),
        }
    }

    /// While the guard lives, weights of a step of `n` tokens: the copies where
    /// [`takes_step`] and the file has them. The guard puts back the previous state.
    pub(crate) fn step(&self, n: i64) -> StepGuard<'_> {
        let prev = self.small.replace(self.on && takes_step(n));
        StepGuard {
            small: &self.small,
            prev,
        }
    }

    /// The tensor to read for `name` now: its copy's ([`copy_name`]) in a decode-sized step's
    /// graph, else `None` (the tensor itself).
    pub(crate) fn copy_of(&self, name: &str) -> Option<String> {
        if !self.small.get() {
            return None;
        }
        copy_name(name)
    }
}

/// Restores [`DenseTwins`]' state when a graph build ends ([`DenseTwins::step`]).
pub(crate) struct StepGuard<'a> {
    small: &'a Cell<bool>,
    prev: bool,
}

impl Drop for StepGuard<'_> {
    fn drop(&mut self) {
        self.small.set(self.prev);
    }
}

/// Whether a step of `n` tokens reads the copies ([`STEP_MAX_TOKENS`]).
pub(crate) fn takes_step(n: i64) -> bool {
    n <= STEP_MAX_TOKENS
}

/// The copy's tensor for `name`: the codes for a copied weight `<base>.weight`, the row scales
/// for its `<base>.scale` ([`dense_fp8::copy_names`]); `None` for any other name.
pub(crate) fn copy_name(name: &str) -> Option<String> {
    if dense_fp8::takes(name) {
        return Some(dense_fp8::copy_names(name).0);
    }
    let weight = format!("{}.weight", name.strip_suffix(".scale")?);
    dense_fp8::takes(&weight).then(|| dense_fp8::copy_names(&weight).1)
}

#[cfg(test)]
mod tests;
