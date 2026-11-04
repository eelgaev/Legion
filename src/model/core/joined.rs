//! Weight matrices that lie back to back in GPU memory, read as one matrix by a decode step
//! (#256): one matvec launch instead of two, and the input read once.
//!
//! - `ssm_alpha` and `ssm_beta`: a gated DeltaNet layer's two gate projections
//!   (`[n_embd, n_v]` each), after Strata-AC922's `gdn_ab_multi` (MIT), which computes both
//!   in one kernel.
//! - `ffn_gate_inp` and `ffn_gate_inp_shexp`: a MoE layer's router (`[n_embd, n_expert]`) and
//!   the shared expert's gate (`[n_embd]`), after Strata-AC922's shared-expert gate path.
//!
//! GGUF files store both members of each pair next to each other, and the loader keeps the
//! file's order in a buffer, so their bytes form one `[n_embd, rows_a + rows_b]` matrix: a
//! tensor placed over them ([`Tensor::place_over`]) reads both. Nothing is copied.
//!
//! The result is exact: ggml-cuda runs an F32 matmul of at most [`MAX_TOKENS`] tokens with its
//! vector kernel (`mmvf`) on every NVIDIA GPU, one block per output row whose size depends only
//! on the row length, so each row comes out as it does from the separate matmul. Larger steps
//! of F32 pairs go to cuBLAS, whose rounding can depend on the matrix shape: they keep the
//! separate matmuls. The CPU graph keeps them too (the reference).
//!
//! Prompt steps (#836, [`takes_prompt`]): BF16 pairs (the NVFP4 checkpoints' `ssm_alpha` /
//! `ssm_beta`, and `ffn_gate_shexp` / `ffn_up_shexp`, a shared expert's gate and up, stored in
//! that order) run as one cuBLAS GEMM from [`prompt_min_tokens`] tokens on: one launch instead
//! of two, and a GEMM twice as tall, which the small projections fill the GPU better with
//! (GB10: shared-expert gate and up 48 TFLOPS each, alpha and beta 7.5). In prompt steps
//! cuBLAS runs these shapes without split-K, so each output is one F32 sum over K in order
//! and the halves keep the separate GEMMs' bits; `--joined-prompt-min-tokens 0` keeps the
//! separate matmuls. A DeltaNet layer's `z` gate projection lies right before alpha and beta
//! in those checkpoints, so prompt steps read the three as one GEMM ([`RUNS`]); `z`'s
//! consumers (the gated norm, ggml's sigmoid and mul) read its rows strided in place. A
//! full-attention layer's query + gate, key and value projections lie back to back too: one
//! GEMM of the three (key and value are 512 rows each in Qwen3.8-Flash-Next, too few to fill
//! the GPU alone), whose rows the norm + RoPE and K/V append kernels read strided.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use crate::ggml::{Context, Tensor, ffi, types};

/// Whether graphs built from now on read joined pairs ([`set_joined_matmuls`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on read joined weight pairs where they fit (A/B tests).
/// On by default; `legion serve --no-joined-matmuls` turns it off.
pub fn set_joined_matmuls(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Joined matmuls built into graphs so far ([`joined_matmuls_built`]).
static BUILT: AtomicUsize = AtomicUsize::new(0);

/// How many joined matmuls graphs built so far contain, in the whole process (for tests that
/// check the joined path runs).
pub fn joined_matmuls_built() -> usize {
    BUILT.load(Ordering::Relaxed)
}

/// The pairs (weight names within a layer, the first stored first).
pub(super) const PAIRS: [(&str, &str); 3] = [
    ("ssm_alpha.weight", "ssm_beta.weight"),
    ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"),
    ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
];

/// Longer runs of weights read as one matrix, in storage order: a gated DeltaNet layer's `z`
/// gate projection (`attn_gate`, `[n_embd, n_v * head_v]`) followed by alpha and beta, which
/// the NVFP4 checkpoints keep in BF16 back to back (#836). Prompt steps only
/// ([`takes_prompt`]); the alpha / beta pair inside it stays a pair of its own. A
/// full-attention layer's query + output gate (`attn_q`, heads interleaved), key and value
/// projections, which the checkpoints store in that order (BF16 in the NVFP4 ones).
pub(super) const RUNS: [&[&str]; 2] = [
    &["attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"],
    &["attn_q.weight", "attn_k.weight", "attn_v.weight"],
];

/// [`RUNS`]' DeltaNet z, alpha and beta.
pub(super) const RUN_Z: usize = 0;

/// [`RUNS`]' attention query + gate, key and value.
pub(super) const RUN_QKV: usize = 1;

/// Most tokens a joined matmul takes: ggml-cuda's F32 vector kernel takes up to 3 tokens on
/// every NVIDIA GPU (Volta, Ampere and later: 3; Turing: 4).
pub(super) const MAX_TOKENS: i64 = 3;

/// [`prompt_min_tokens`]'s default: prompt chunks, well above the step sizes where cuBLAS
/// splits K over CTAs (decode and verify steps, short prompt tails).
pub const PROMPT_MIN_TOKENS: u32 = 1024;

/// Fewest tokens of a prompt step that reads joined BF16 pairs; 0: none
/// ([`set_prompt_min_tokens`]).
static PROMPT_MIN: AtomicU32 = AtomicU32::new(PROMPT_MIN_TOKENS);

/// Sets the fewest tokens of a prompt step that reads joined BF16 pairs as one GEMM in graphs
/// built from now on (`legion serve --joined-prompt-min-tokens`, default
/// [`PROMPT_MIN_TOKENS`]); 0 turns it off.
pub fn set_prompt_min_tokens(n: u32) {
    PROMPT_MIN.store(n, Ordering::Relaxed);
}

/// [`set_prompt_min_tokens`]'s value.
pub fn prompt_min_tokens() -> u32 {
    PROMPT_MIN.load(Ordering::Relaxed)
}

/// Whether prompt steps read [`RUNS`] as one GEMM ([`set_prompt_z`]).
static PROMPT_Z: AtomicBool = AtomicBool::new(true);

/// Sets whether prompt steps of graphs built from now on read a DeltaNet layer's z, alpha and
/// beta ([`RUNS`]) as one GEMM where they are joined. On by default; `legion serve
/// --no-joined-prompt-z` keeps z apart (alpha and beta stay a pair).
pub fn set_prompt_z(on: bool) {
    PROMPT_Z.store(on, Ordering::Relaxed);
}

/// [`set_prompt_z`]'s value.
pub(super) fn prompt_z() -> bool {
    PROMPT_Z.load(Ordering::Relaxed)
}

/// Whether prompt steps read [`RUNS`]' attention projections as one GEMM ([`set_prompt_qkv`]).
static PROMPT_QKV: AtomicBool = AtomicBool::new(true);

/// Sets whether prompt steps of graphs built from now on read a full-attention layer's query +
/// gate, key and value projections ([`RUN_QKV`]) as one GEMM where they are joined. On by
/// default; `legion serve --no-joined-prompt-qkv` keeps three GEMMs.
pub fn set_prompt_qkv(on: bool) {
    PROMPT_QKV.store(on, Ordering::Relaxed);
}

/// [`set_prompt_qkv`]'s value.
pub(super) fn prompt_qkv() -> bool {
    PROMPT_QKV.load(Ordering::Relaxed)
}

/// The joined tensors of a model's weights.
pub(super) struct Joined {
    ctx: Context,
    /// The joined tensors' names (the first member's name and [`SUFFIX`]).
    names: Vec<String>,
}

/// What a joined tensor's name adds to its first member's.
const SUFFIX: &str = "+";

/// The name of the tensor joining `first` and the weight after it.
pub(super) fn joined_name(first: &str) -> String {
    format!("{first}{SUFFIX}")
}

/// The address of `t`'s bytes.
fn addr(t: Tensor<'_>) -> usize {
    // SAFETY: a live tensor; reads a field.
    unsafe { (*t.raw()).data as usize }
}

/// Whether `a` and `b` (stored in that order) have the shapes of a pair read as one matrix:
/// F32 (small steps) or BF16 (prompt steps) rows of one type and the same even length (the
/// vector kernel reads pairs of values), at most 2-D, contiguous.
fn shapes_fit(a: Tensor<'_>, b: Tensor<'_>) -> bool {
    let fits = |t: Tensor<'_>| {
        matches!(t.ty(), types::F32 | types::BF16)
            && t.ne()[2] == 1
            && t.ne()[3] == 1
            && t.is_contiguous()
    };
    fits(a) && fits(b) && a.ty() == b.ty() && a.ne()[0] == b.ne()[0] && a.ne()[0] % 2 == 0
}

/// Whether `a` and `b` can be joined: [`shapes_fit`], in GPU memory that is not split over
/// devices (`host`: host memory too, for tests), back to back in one buffer.
pub(super) fn joinable(a: Tensor<'_>, b: Tensor<'_>, host: bool) -> bool {
    let placed = |t: Tensor<'_>| (host || !t.is_host()) && !t.is_meta();
    shapes_fit(a, b) && placed(a) && placed(b) && addr(a) + a.nbytes() == addr(b)
}

/// Whether the run `ts` (two or more, stored in that order) can be read as one matrix: each
/// member [`joinable`] with the next.
fn run_joinable(ts: &[Tensor<'_>], host: bool) -> bool {
    ts.len() >= 2 && ts.windows(2).all(|w| joinable(w[0], w[1], host))
}

/// The members after the first of the pair or run that starts with `name`'s weight, named as
/// `name` is (its layer's prefix): from [`PAIRS`] and [`RUNS`].
fn rest_of(name: &str) -> Option<Vec<String>> {
    let groups = PAIRS.iter().map(|(f, s)| vec![*f, *s]);
    let mut groups = groups.chain(RUNS.iter().map(|r| r.to_vec()));
    groups.find_map(|g| {
        let prefix = name.strip_suffix(g[0])?;
        (prefix.is_empty() || prefix.ends_with('.'))
            .then(|| g[1..].iter().map(|s| format!("{prefix}{s}")).collect())
    })
}

impl Joined {
    /// Joins every pair of [`PAIRS`] and run of [`RUNS`] among `tensors` (the other members
    /// looked up by name with `get`) that [`joinable`] takes; `None` if none.
    pub(super) fn build<'w>(
        tensors: impl Iterator<Item = Tensor<'w>>,
        get: impl Fn(&str) -> Option<Tensor<'w>>,
        host: bool,
    ) -> Option<Self> {
        let runs: Vec<Vec<Tensor<'w>>> = tensors
            .filter_map(|a| {
                let rest = rest_of(&a.name())?;
                let mut run = vec![a];
                for name in &rest {
                    run.push(get(name)?);
                }
                run_joinable(&run, host).then_some(run)
            })
            .collect();
        if runs.is_empty() {
            return None;
        }
        let ctx = Context::new(runs.len(), 0);
        let mut names = Vec::new();
        for run in runs {
            let a = run[0];
            let rows = run.iter().map(|t| t.ne()[1]).sum::<i64>();
            let name = joined_name(&a.name());
            let t = ctx.new_tensor(a.ty(), &[a.ne()[0], rows]).named(&name);
            // SAFETY: weights, which graphs only read, and live: the model drops the joined
            // tensors before its weights.
            if unsafe { t.place_over(&run) } {
                names.push(name);
            }
        }
        // Graph builds look a joined tensor up per layer.
        ctx.index_names();
        (!names.is_empty()).then_some(Self { ctx, names })
    }

    /// The joined tensors.
    pub(super) fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> + '_ {
        self.names.iter().filter_map(|n| self.ctx.tensor(n))
    }

    /// The tensor joining `a` and `b`, if they are the very tensors it was placed over (the
    /// MTP head's matrices may have been replaced by copies since, see `Model::requant_mtp`).
    /// The model reads pairs through [`Self::get_run`].
    #[cfg(test)]
    pub(super) fn get(&self, a: Tensor<'_>, b: Tensor<'_>) -> Option<Tensor<'_>> {
        self.get_run(&[a, b])
    }

    /// [`Self::get`] for a run of two or more members (a pair or one of [`RUNS`]), in storage
    /// order.
    pub(super) fn get_run(&self, ts: &[Tensor<'_>]) -> Option<Tensor<'_>> {
        let first = *ts.first()?;
        let j = self.ctx.tensor(&joined_name(&first.name()))?;
        let same = addr(j) == addr(first)
            && ts.len() >= 2
            && ts
                .windows(2)
                .all(|w| addr(w[0]) + w[0].nbytes() == addr(w[1]) && shapes_fit(w[0], w[1]))
            && j.nbytes() == ts.iter().map(|t| t.nbytes()).sum::<usize>();
        same.then_some(j)
    }
}

/// Whether a step of `n` tokens reads joined pairs: switched on ([`set_joined_matmuls`]), on
/// CUDA, at most [`MAX_TOKENS`] tokens.
pub(super) fn takes(n: i64) -> bool {
    ON.load(Ordering::Relaxed) && cfg!(feature = "cuda") && (1..=MAX_TOKENS).contains(&n)
}

/// Whether a prompt step of `n` tokens reads joined BF16 pairs: on CUDA, at least
/// [`prompt_min_tokens`] tokens (not 0).
pub(super) fn takes_prompt(n: i64) -> bool {
    let min = prompt_min_tokens();
    cfg!(feature = "cuda") && min > 0 && n >= i64::from(min)
}

/// Which steps a joined tensor of type `ty` serves: F32 pairs small steps ([`takes`]), BF16
/// pairs prompt steps ([`takes_prompt`]).
pub(super) fn serves(ty: ffi::ggml_type, n: i64) -> bool {
    match ty {
        types::F32 => takes(n),
        types::BF16 => takes_prompt(n),
        _ => false,
    }
}

/// A joined matmul's result `r` (`[sum(rows), n]`) split back into its members' results, one
/// `[rows[i], n]` view each (rows `r.nb()[1]` bytes apart). Counts the joined matmuls built
/// ([`joined_matmuls_built`]).
pub(super) fn split_run<'a>(ctx: &'a Context, r: Tensor<'a>, rows: &[i64]) -> Vec<Tensor<'a>> {
    BUILT.fetch_add(1, Ordering::Relaxed);
    let [total, n, _, _] = r.ne();
    assert_eq!(rows.iter().sum::<i64>(), total, "split_run: member rows");
    let nb1 = r.nb()[1];
    let mut off = 0;
    rows.iter()
        .map(|&m| {
            let v = ctx.view_2d(r, [m, n], nb1, off as usize * r.element_size());
            off += m;
            v
        })
        .collect()
}

#[cfg(test)]
mod tests;
