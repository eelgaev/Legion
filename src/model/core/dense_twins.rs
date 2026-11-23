//! `--dense-fp8` (#1183): decode and verify steps read the FP8 copies of a Qwen3.8-Flash-Next
//! checkpoint's dense projections ([`crate::hf::dense_fp8`]) in place of their BF16 weights.
//!
//! [`super::Model::w`] gives the copy's codes for the weight's name, and its row scales for
//! the weight's `.scale` (which a BF16 weight has none of), while the graph of a step without
//! prompt rows and of at most [`STEP_MAX_TOKENS`] tokens is built ([`DenseTwins::step`]).
//! Every path that reads a weight by name then sees an FP8 weight with a per-row scale, as for
//! a channel-wise FP8 checkpoint: the BF16 paths step aside and the FP8 matmul's W8A16 route
//! runs it ([`super::linear::Linear::mul_mat`]). Steps with prompt rows (any length, alone or
//! batched with decode rows, which then read BF16 too: a weight is one matrix per graph), larger
//! steps and everything outside a graph build read the BF16 weights, so they keep every kernel,
//! joined run and fusion they take without the flag. Without copies in the file nothing
//! changes.
//!
//! Prompt rows: a group of more than one token per sequence that is not a verify group
//! ([`prompt_rows`]). A prompt chunk of one token per sequence has a decode step's shape, whose
//! graph it shares (graphs are reused by shape), and reads the copies.
//!
//! `--dense-fp8-prompt` ([`DenseTwins::set_prompt`]): prompt steps read the copies too, as
//! W8A8 GEMMs on E4M3 tensor cores with per-token activation scales (twice BF16's rate on a
//! GB10). The activations' codes come from BF16 rows the step's ops write already where it has
//! them ([`super::linear::Linear::mul_mat`]), and a joined run's copies, stored back to back
//! ([`dense_fp8::RUNS`]), are one GEMM as the BF16 run is ([`Runs`]).

use std::cell::Cell;

use crate::ggml::{Context, Tensor, types};
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
    /// Prompt steps read the copies too ([`Self::set_prompt`]).
    prompt: bool,
    /// A graph of a step that reads the copies is being built.
    active: Cell<bool>,
}

impl DenseTwins {
    pub(crate) fn from_gguf(g: &GgufFile) -> Self {
        Self {
            on: g.get(dense_fp8::KEY).is_some(),
            prompt: false,
            active: Cell::new(false),
        }
    }

    /// Sets whether graphs built from now on read the copies in prompt steps too
    /// (`--dense-fp8-prompt`); whether they do (the file has copies).
    pub(crate) fn set_prompt(&mut self, on: bool) -> bool {
        self.prompt = on;
        self.on && on
    }

    /// While the guard lives, weights of a step of `n` tokens, with prompt rows or not
    /// ([`prompt_rows`]): the copies where the file has them and the step is a decode or verify
    /// step that [`takes_step`], or any step under [`Self::set_prompt`]. The guard puts back
    /// the previous state.
    pub(crate) fn step(&self, n: i64, prompt_rows: bool) -> StepGuard<'_> {
        let prev = self
            .active
            .replace(self.on && (self.prompt || (!prompt_rows && takes_step(n))));
        StepGuard {
            active: &self.active,
            prev,
        }
    }

    /// The tensor to read for `name` now: its copy's ([`copy_name`]) in the graph of a step
    /// that reads them, else `None` (the tensor itself).
    pub(crate) fn copy_of(&self, name: &str) -> Option<String> {
        if !self.active.get() {
            return None;
        }
        copy_name(name)
    }
}

/// Restores [`DenseTwins`]' state when a graph build ends ([`DenseTwins::step`]).
pub(crate) struct StepGuard<'a> {
    active: &'a Cell<bool>,
    prev: bool,
}

impl Drop for StepGuard<'_> {
    fn drop(&mut self) {
        self.active.set(self.prev);
    }
}

/// The copies of [`dense_fp8::RUNS`]' runs read as one matrix by `--dense-fp8-prompt`'s prompt
/// steps: per layer and run, a tensor placed over the members' codes (I8 `[k, sum(m)]`, named
/// as the first member's codes with a `+`) and one over their row scales (F32 `[sum(m)]`), as
/// [`super::joined`] reads the BF16 runs. Nothing is copied.
pub(crate) struct Runs {
    ctx: Context,
    /// How many runs are joined.
    runs: usize,
}

/// The address of `t`'s bytes.
fn addr(t: Tensor<'_>) -> usize {
    // SAFETY: a live tensor; reads a field.
    unsafe { (*t.raw()).data as usize }
}

/// Whether `ts` (two or more, in storage order) lie back to back in device memory (`host`: host
/// memory too, for tests), each of type `ty`, contiguous, 2-D (codes, with one row length) or
/// 1-D (scales).
fn adjacent(ts: &[Tensor<'_>], ty: crate::ggml::ffi::ggml_type, codes: bool, host: bool) -> bool {
    let fits = |t: &Tensor<'_>| {
        let shape = if codes {
            t.ne()[0] == ts[0].ne()[0]
        } else {
            t.ne()[1] == 1
        };
        t.ty() == ty
            && t.is_contiguous()
            && (host || !t.is_host())
            && !t.is_meta()
            && t.ne()[2..] == [1, 1]
            && shape
    };
    ts.len() >= 2
        && ts.iter().all(fits)
        && ts
            .windows(2)
            .all(|w| addr(w[0]) + w[0].nbytes() == addr(w[1]))
}

impl Runs {
    /// Joins every one of `layers`' [`dense_fp8::RUNS`] runs whose copies `get` finds back to
    /// back in GPU memory (`host`: host memory too, for tests), codes and scales each; `None`
    /// if none.
    pub(crate) fn build<'w>(
        layers: usize,
        get: impl Fn(&str) -> Option<Tensor<'w>>,
        host: bool,
    ) -> Option<Self> {
        let mut found: Vec<(Vec<Tensor<'w>>, Vec<Tensor<'w>>)> = Vec::new();
        for il in 0..layers {
            for run in dense_fp8::RUNS {
                let names: Vec<(String, String)> = run
                    .iter()
                    .map(|p| dense_fp8::copy_names(&format!("blk.{il}.{p}.weight")))
                    .collect();
                let codes: Option<Vec<_>> = names.iter().map(|n| get(&n.0)).collect();
                let scales: Option<Vec<_>> = names.iter().map(|n| get(&n.1)).collect();
                let (Some(codes), Some(scales)) = (codes, scales) else {
                    continue;
                };
                if adjacent(&codes, types::I8, true, host)
                    && adjacent(&scales, types::F32, false, host)
                {
                    found.push((codes, scales));
                }
            }
        }
        if found.is_empty() {
            return None;
        }
        let ctx = Context::new(2 * found.len(), 0);
        let mut runs = 0;
        for (codes, scales) in found {
            let rows: i64 = codes.iter().map(|t| t.ne()[1]).sum();
            let c = ctx
                .new_tensor(types::I8, &[codes[0].ne()[0], rows])
                .named(&joined_name(&codes[0].name()));
            let s = ctx
                .new_tensor(types::F32, &[rows])
                .named(&joined_name(&scales[0].name()));
            // SAFETY: weights, which graphs only read, and live: the model drops the joined
            // tensors before its weights.
            if unsafe { c.place_over(&codes) && s.place_over(&scales) } {
                runs += 1;
            }
        }
        ctx.index_names();
        (runs > 0).then_some(Self { ctx, runs })
    }

    /// How many runs are joined.
    pub(crate) fn len(&self) -> usize {
        self.runs
    }

    /// The joined codes and scales of the run whose members' copies are `codes` and `scales`
    /// (in storage order), if they are the very tensors the joined ones were placed over.
    pub(crate) fn get(
        &self,
        codes: &[Tensor<'_>],
        scales: &[Tensor<'_>],
    ) -> Option<(Tensor<'_>, Tensor<'_>)> {
        let joined = |ts: &[Tensor<'_>]| {
            let j = self.ctx.tensor(&joined_name(&ts.first()?.name()))?;
            let same = addr(j) == addr(ts[0])
                && ts.len() >= 2
                && ts
                    .windows(2)
                    .all(|w| addr(w[0]) + w[0].nbytes() == addr(w[1]))
                && j.nbytes() == ts.iter().map(|t| t.nbytes()).sum::<usize>();
            same.then_some(j)
        };
        Some((joined(codes)?, joined(scales)?))
    }
}

/// The name of the tensor joining a run that starts with `first`.
fn joined_name(first: &str) -> String {
    format!("{first}+")
}

/// Whether a step of groups of `(tokens per sequence, verify)` has prompt rows: a group of
/// several tokens per sequence that is not a verify group (a decode step's groups have one
/// token per sequence, a verify step's are marked).
pub(crate) fn prompt_rows(groups: impl IntoIterator<Item = (usize, bool)>) -> bool {
    groups
        .into_iter()
        .any(|(n_tok, verify)| n_tok > 1 && !verify)
}

/// Whether a decode or verify step of `n` tokens reads the copies ([`STEP_MAX_TOKENS`]).
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
