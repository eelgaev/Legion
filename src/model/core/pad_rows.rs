//! A verify step's padding rows (#836): every decoding sequence of a verify step feeds the
//! same number of rows (`1 +` the step's most drafts), its own drafts followed by padding that
//! nothing reads back. Their routed experts are read like any other row's: at one sequence
//! with 3-4 drafts of 8 that is about a third of the step's expert bytes.
//!
//! The engine registers each sequence's fed rows and how many of them are real for the
//! duration of its step ([`register`]); the model's MoE layers then route every padding row's
//! expert work as its sequence's first row (which is always real, see [`map`]), so the
//! routed-expert kernels read only the experts of the real rows. A padding row's output is then
//! another expert set's sums on its own input: finite and unread. Every real row's sums are
//! the same instructions on the same values (each pair's column is independent of the other
//! pairs of its expert tile, and the kernels are picked from the step's pair count, which does
//! not change), so the output is the same bits. On by default where the routed experts take
//! Legion's NVFP4 kernels; `legion serve --no-moe-pad-remap` turns it off. On the decode mv
//! kernels the padding rows' pairs are skipped outright (zero results, no room in the experts'
//! tiles; [`crate::model::nvfp4_mm::set_pad_skip`], `--no-moe-pad-skip` keeps the remap).
//!
//! With `--moe-pad-remap-tiered` ([`set_tiered`], opt-in), tiered GGUF experts (an expert
//! cache) on Legion's expert matvec take it too ([`routed_ids`]: the cache then also looks up,
//! logs and learns only the real rows' experts). That is not byte-identical end to end: on 4x
//! V100 the greedy text differed from the default's (the matvec's per-pair bits are meant not to
//! depend on the other pairs of an expert, so either some variant does, or the cache's changed
//! contents do), and with MTP k=2 it saved only ~1.4% of the decode expert bytes (the drafter
//! fills nearly every step), so it stays off. `--moe-pad-remap-probe` splits the remap's two
//! uses (the kernels' ids, the cache's log) for finding out which one moves the text.
//!
//! The opt-in alternative ([`set_skip_tiered`], `--moe-pad-skip-tiered`): the ids stay the
//! router's, and the expert matvec gets the row map as a source. It skips a padding row's pair
//! only when no real row's pair chose the same expert (a dead padding pair; its rows are written
//! 0). Every expert a real row chose is computed as without the map, with the same users (its
//! padding pairs included), and the cache looks up and learns the same ids. It still moved the
//! greedy text on 4x V100, so it stays off. `--moe-pad-skip-layout-only` attaches the map but
//! skips nothing: on an AC922 (4x V100) it gave the same text as the skip, so the skip itself
//! changes no real row; the extra graph source does (it moves ggml's buffer layout, and
//! ggml-cuda's layout-dependent paths change bits). It saves the bytes of the experts only
//! padding rows chose, which at MTP k=2 bought no decode speed.

use crate::ggml::{Context, Tensor};

#[cfg(test)]
mod tests;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static ON: AtomicBool = AtomicBool::new(true);

/// Whether verify steps route their padding rows' experts as their sequences' first rows
/// (see the module docs). Set before the model's graphs are built.
pub fn set_pad_remap(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

static TIERED: AtomicBool = AtomicBool::new(false);

/// Whether the remap also covers tiered GGUF experts on Legion's expert matvec (off by default,
/// see the module docs; only with [`on`]). Set before the model's graphs are built.
pub fn set_tiered(on: bool) {
    TIERED.store(on, Ordering::Relaxed);
}

pub fn tiered() -> bool {
    on() && TIERED.load(Ordering::Relaxed)
}

static SKIP_TIERED: AtomicBool = AtomicBool::new(false);

/// Whether a verify step whose routed FFN runs on Legion's expert matvec over an expert cache
/// hands the matvec its row map, which then skips the dead padding pairs (padding rows' pairs
/// of experts no real row chose; see the module docs). Off by default (only with [`on`]); set
/// before the model's graphs are built.
pub fn set_skip_tiered(on: bool) {
    SKIP_TIERED.store(on, Ordering::Relaxed);
}

pub fn skip_tiered() -> bool {
    on() && SKIP_TIERED.load(Ordering::Relaxed)
}

static SKIP_LAYOUT_ONLY: AtomicBool = AtomicBool::new(false);

/// A diagnostic of [`skip_tiered`]: the ops keep the row map as a source (the graph and its
/// layout as with the skip) but the kernels skip nothing.
pub fn set_skip_layout_only(on: bool) {
    SKIP_LAYOUT_ONLY.store(on, Ordering::Relaxed);
}

pub fn skip_layout_only() -> bool {
    SKIP_LAYOUT_ONLY.load(Ordering::Relaxed)
}

/// `--moe-pad-remap-probe` (a diagnostic of [`tiered`]'s remap): which of the tiered lookup's
/// two uses take the remapped ids. `Both` is the remap itself; `Kernels`: the expert kernels
/// read the remapped ids, the cache logs and learns the router's own; `Log`: the other way
/// round. Each of the probes adds one table lookup to the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Probe {
    /// The remap as it is: kernels and cache both.
    Both,
    /// The kernels read the remapped ids; the cache logs the router's own.
    Kernels,
    /// The cache logs the remapped ids; the kernels read the router's own.
    Log,
}

static PROBE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_probe(p: Probe) {
    PROBE.store(p as u8, Ordering::Relaxed);
}

pub fn probe() -> Probe {
    match PROBE.load(Ordering::Relaxed) {
        1 => Probe::Kernels,
        2 => Probe::Log,
        _ => Probe::Both,
    }
}

/// Registered rows: the address and length of a sequence's fed tokens, and its real rows.
static ROWS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// Registers the fed rows of a verify step's sequences (`fed[i]`, of which the first `real[i]`
/// are the sequence's own) until the returned guard drops. A step's [`super::SeqInput`] whose
/// `tokens` is one of these slices (the same address and length) has its padding known;
/// any other slice (a copy, a part) has none, so its rows keep their own experts.
pub fn register<'f>(fed: &'f [Vec<u32>], real: impl IntoIterator<Item = usize>) -> Guard<'f> {
    let mut keys = Vec::new();
    if on() {
        let mut rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
        for (f, r) in fed.iter().zip(real) {
            if !f.is_empty() && r >= 1 && r < f.len() {
                let key = (f.as_ptr() as usize, f.len(), r);
                rows.push(key);
                keys.push(key);
            }
        }
    }
    Guard {
        keys,
        _fed: std::marker::PhantomData,
    }
}

/// Keeps a step's registered rows (see [`register`]); dropping it removes them. It borrows the
/// fed rows, so they stay where they were registered while it lives.
pub struct Guard<'f> {
    keys: Vec<(usize, usize, usize)>,
    _fed: std::marker::PhantomData<&'f [Vec<u32>]>,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let mut rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
        for k in &self.keys {
            if let Some(i) = rows.iter().position(|r| r == k) {
                rows.swap_remove(i);
            }
        }
    }
}

/// The real rows of a sequence fed `tokens`, when registered; `None`: all of them.
pub fn real_rows(tokens: &[u32]) -> Option<usize> {
    if tokens.is_empty() {
        return None;
    }
    let (a, n) = (tokens.as_ptr() as usize, tokens.len());
    let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
    rows.iter()
        .find(|&&(ra, rn, _)| ra == a && rn == n)
        .map(|&(_, _, r)| r)
}

/// The router's choices `selected` (I32 `[n_used, n]`) with every row `t` taking row `rows[t]`'s
/// (`rows` I32 `[n]`, [`map`]): a verify step's padding rows choose their sequences' first rows'
/// experts, every real row its own. One get_rows, contiguous (where a lookup would `cont` the
/// choices anyway). Expert kernels fed these read only the real rows' experts.
pub fn routed_ids<'a>(ctx: &'a Context, selected: Tensor<'a>, rows: Tensor<'a>) -> Tensor<'a> {
    ctx.get_rows(selected, rows)
}

/// The MoE row map of a group of sequences `tokens` (`n_tok` rows each, in order): for every
/// row, the row whose experts it takes. A padding row (past its sequence's real rows) takes its
/// sequence's first row; every other row its own.
pub fn map<'t>(seqs: impl IntoIterator<Item = &'t [u32]>, n_tok: usize) -> Vec<i32> {
    let mut out = Vec::new();
    for (si, tokens) in seqs.into_iter().enumerate() {
        let real = if tokens.len() == n_tok {
            real_rows(tokens).unwrap_or(n_tok)
        } else {
            n_tok
        };
        let first = si * n_tok;
        out.extend((0..n_tok).map(|t| (first + if t < real { t } else { 0 }) as i32));
    }
    out
}
