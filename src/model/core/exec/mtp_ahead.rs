//! The next MTP draft pass's inputs, queued while the pass before it computes (#827).
//!
//! A draft round ([`Executor::mtp_chain`]) runs the head pass after pass: each further pass
//! feeds every row still drafting its last pick, at the next position, with the head's own
//! output residual. Before, the host waited for a pass, read its picks and only then built the
//! next pass's inputs (token ids, positions, KV cells, the attention's block table or masks)
//! and queued them; on the GB10 the GPU sat idle for ~110-130 µs between two passes, ~40-60 µs
//! of it that host work and its copies.
//!
//! When the pass after this one is known but for its picks (every row of this pass goes on
//! unless its pick misses the confidence floor: none is at its cap), this pass queues the next
//! pass's inputs behind its own reads, on the same stream: its picks and output residual
//! copied on the device into the next graph's token ids and residual input, the rest written
//! from pinned memory while this pass computes. If every row did go on, the next pass starts
//! its graph at once, without filling anything; else it fills its inputs as before, over the
//! staged ones. The bytes are the same either way, so outputs do not change.
//!
//! Only where nothing else runs between two passes of a round and the copies queue: no expert
//! cache (its work runs beside the passes), one machine, no GLM parallel MTP, no sparse head or
//! head that reads the target's cache, a backend that queues copies (CUDA, not a meta
//! backend), and passes within the gather budget. Opt-in: `--mtp-ahead`.
//!
//! [`Executor::mtp_chain`]: super::Executor::mtp_chain

use std::ops::Range;

use super::mtp_window::Queued;
use super::{Executor, Group, GroupShape, Logits, SeqInput};
use crate::ggml::pinned_io::PinnedIo;
use crate::ggml::{Backend, KeptGraph, Tensor, ffi};
use crate::model::core::{InputSink, Inputs, KV_BLOCK};

#[cfg(test)]
mod tests;

/// The token id a predicted pass's sequences carry: never written (the picks are copied on
/// the device).
static PLACEHOLDER: [u32; 1] = [0];

/// What [`Executor::mtp_pass_run`] takes besides its groups for staging.
#[derive(Clone, Copy, Default)]
pub(super) struct Ahead<'a, 'g> {
    /// The pass after this one should every row go on ([`predict`]): queue its inputs.
    pub next: Option<&'a [Group<'g>]>,
    /// This pass may run on inputs the pass before staged for it, if they are its own.
    pub staged: bool,
    /// Leave this pass running if it stages the next one ([`super::mtp_pipe`]).
    pub pipe: bool,
    /// This pass runs on the staged inputs alone, queued before the pass that staged them
    /// landed: its groups' token ids are not known (placeholders); if the inputs staged are
    /// not its own, it does not run ([`super::mtp_pipe`]).
    pub spec: bool,
    /// The confidence floor: with `spec`, the pass runs on the GPU only if every row of the
    /// pass before reached it (`crate::ggml::stage_graph::cond`).
    pub p_min: f32,
}

/// The next pass's inputs, queued by a pass ([`prestage`]).
pub(super) struct Staged {
    shape: Vec<GroupShape>,
    /// Its graph (the address of its `ggml_cgraph`): the one whose inputs were written.
    graph: usize,
    key: Vec<GroupKey>,
    /// The token ids it feeds (copied on the device): the staging pass's picks.
    pub(super) tokens: Vec<u32>,
    /// The staging pass queued the device's stop check of its picks before the inputs
    /// (`crate::ggml::stage_graph::cond::decide`): the pass may run behind it.
    pub(super) decided: bool,
}

/// What a group's inputs depend on besides its token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GroupKey {
    slot: usize,
    verify: bool,
    seqs: Vec<SeqKey>,
}

/// What a sequence's inputs depend on besides its token ids.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SeqKey {
    n_past: usize,
    /// Its block list: address and length.
    blocks: (usize, usize),
    ring: usize,
    n_tokens: usize,
    logits: Logits,
}

/// `groups` as [`GroupKey`]s.
pub(super) fn key(groups: &[Group<'_>]) -> Vec<GroupKey> {
    groups
        .iter()
        .map(|g| GroupKey {
            slot: g.slot,
            verify: g.verify,
            seqs: (g.seqs.iter())
                .map(|s| SeqKey {
                    n_past: s.n_past,
                    blocks: (s.blocks.as_ptr() as usize, s.blocks.len()),
                    ring: s.ring,
                    n_tokens: s.tokens.len(),
                    logits: s.logits,
                })
                .collect(),
        })
        .collect()
}

impl Staged {
    /// Whether a pass of `groups` (`shape`) running `kept` is the one staged: then its inputs
    /// hold what filling them would write. `any_tokens`: but for the token ids, which the
    /// staging pass copied on the device from its picks (a pass queued before they landed).
    pub(super) fn matches(
        &self,
        shape: &[GroupShape],
        groups: &[Group<'_>],
        kept: &KeptGraph,
        any_tokens: bool,
    ) -> bool {
        let ok = self.shape == shape
            && self.graph == graph_id(kept)
            && (any_tokens
                || (groups.iter().flat_map(|g| &g.seqs))
                    .flat_map(|s| s.tokens)
                    .eq(self.tokens.iter()))
            && self.key == key(groups);
        if ok {
            static FIRST: std::sync::Once = std::sync::Once::new();
            FIRST.call_once(|| tracing::info!("mtp passes start on inputs staged ahead"));
        }
        ok
    }
}

/// The device address of `t`'s data and its F32 values: a pass's pick probabilities, for the
/// next pass's device condition.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn f32_span(t: Tensor<'_>) -> Option<(usize, u32)> {
    let n = u32::try_from(t.nbytes() / 4).ok()?;
    (t.ty() == ffi::ggml_type_GGML_TYPE_F32).then(|| (span(t).start, n))
}

fn graph_id(kept: &KeptGraph) -> usize {
    kept.graph().raw() as usize
}

/// The pass after this one should every row go on: one group of one-token sequences, a row's
/// at `n_past` (the position of its next draft) with its blocks and ring, as
/// [`Executor::mtp_chain`] feeds them. Each item is a row of this pass, `None` for a row that
/// does not go on whatever its pick (at its cap). `None` if a row does not go on, there is no
/// row, or a row's blocks do not reach its next position.
///
/// [`Executor::mtp_chain`]: super::Executor::mtp_chain
pub(super) fn predict<'b>(
    rows: impl IntoIterator<Item = Option<(usize, &'b [u32], usize)>>,
) -> Option<Group<'b>> {
    let mut seqs = Vec::new();
    for row in rows {
        let (n_past, blocks, ring) = row?;
        if (n_past + 1).div_ceil(KV_BLOCK) > blocks.len() {
            return None;
        }
        seqs.push(SeqInput {
            tokens: &PLACEHOLDER,
            n_past,
            blocks,
            ring,
            logits: Logits::Last,
            media: None,
        });
    }
    (!seqs.is_empty()).then_some(Group {
        slot: 0,
        seqs,
        verify: false,
    })
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

/// An order to run the device copies `(src, dst)` (byte ranges) in, so that no copy writes
/// over what a later one reads; a copy onto itself is left out (its bytes are in place).
/// `None` if there is none, a copy overlaps itself in part, or two copies' destinations
/// overlap.
pub(super) fn copy_order(copies: &[(Range<usize>, Range<usize>)]) -> Option<Vec<usize>> {
    for (i, (s, d)) in copies.iter().enumerate() {
        if s != d && overlaps(s, d) {
            return None;
        }
        if copies[..i].iter().any(|(_, e)| overlaps(d, e)) {
            return None;
        }
    }
    let mut todo: Vec<usize> = (0..copies.len())
        .filter(|&i| copies[i].0 != copies[i].1)
        .collect();
    let mut order = Vec::with_capacity(todo.len());
    while !todo.is_empty() {
        let at = todo.iter().position(|&i| {
            (todo.iter())
                .filter(|&&j| j != i)
                .all(|&j| !overlaps(&copies[i].1, &copies[j].0))
        })?;
        order.push(todo.remove(at));
    }
    Some(order)
}

/// The bytes of `t`'s data.
fn span(t: Tensor<'_>) -> Range<usize> {
    // SAFETY: a live tensor; the field is read only.
    let at = unsafe { (*t.raw()).data } as usize;
    at..at + t.nbytes()
}

fn same_layout(a: Tensor<'_>, b: Tensor<'_>) -> bool {
    a.ty() == b.ty() && a.ne() == b.ne() && a.nb() == b.nb()
}

/// Notes what [`Inputs::fill_into`] would write (but the token ids): whether every write
/// queues on the backend, the sizes and where they go.
struct Measure<'b> {
    backend: &'b Backend,
    skip: *mut ffi::ggml_tensor,
    queued: bool,
    lens: Vec<usize>,
    spans: Vec<Range<usize>>,
}

impl Measure<'_> {
    fn add(&mut self, t: Tensor<'_>, len: usize) {
        if t.raw() == self.skip {
            return;
        }
        self.queued &= self.backend.queues_copies_of(t);
        self.lens.push(len);
        self.spans.push(span(t));
    }
}

impl InputSink for Measure<'_> {
    fn i32(&mut self, t: Tensor<'_>, values: &[i32]) {
        self.add(t, std::mem::size_of_val(values));
    }

    fn f32(&mut self, t: Tensor<'_>, values: &[f32]) {
        self.add(t, std::mem::size_of_val(values));
    }

    fn bytes(&mut self, t: Tensor<'_>, bytes: &[u8]) {
        self.add(t, bytes.len());
    }
}

/// [`Queued`] but for one tensor.
struct Skip<'p> {
    inner: Queued<'p>,
    skip: *mut ffi::ggml_tensor,
}

impl InputSink for Skip<'_> {
    fn i32(&mut self, t: Tensor<'_>, values: &[i32]) {
        if t.raw() != self.skip {
            self.inner.i32(t, values);
        }
    }

    fn f32(&mut self, t: Tensor<'_>, values: &[f32]) {
        if t.raw() != self.skip {
            self.inner.f32(t, values);
        }
    }

    fn bytes(&mut self, t: Tensor<'_>, bytes: &[u8]) {
        if t.raw() != self.skip {
            self.inner.bytes(t, bytes);
        }
    }
}

/// What [`prestage`] stages: the next pass (as [`predict`] made it), its shape, its graph.
pub(super) struct NextPass<'n, 'g> {
    pub groups: &'n [Group<'g>],
    pub shape: &'n [GroupShape],
    pub kept: &'n KeptGraph,
    /// The model's sliding window ([`Inputs::from_list`]).
    pub swa: usize,
    /// [`Inputs::exclusive`].
    pub exclusive: bool,
}

/// Queues the inputs of `next` on `backend`'s stream, after the pass running on it now
/// (whose reads are queued already): its token ids from `picks` and its residual rows from
/// `h_out` (that pass's outputs) as device copies, the other inputs from pinned memory through
/// `io`, in this pass's transfers. Every copy queues, or nothing is queued: `None`.
///
/// The next pass's graph may share its compute buffer with the running one: its inputs are
/// written after the running pass, and the copies run in an order that writes nothing a later
/// copy reads ([`copy_order`]); the pinned writes go last, onto none of the copies'
/// destinations.
pub(super) fn prestage(
    io: &mut PinnedIo,
    backend: &mut Backend,
    picks: Tensor<'_>,
    h_out: Tensor<'_>,
    next: NextPass<'_, '_>,
) -> Option<Staged> {
    let kept = next.kept;
    let mut inputs =
        Inputs::from_list(next.shape, 0, false, 0, false, next.swa, |i| kept.tensor(i));
    inputs.exclusive = next.exclusive;
    let tokens = inputs.tokens;
    let h = kept.tensor(inputs.list().len());
    let pairs = [(h_out, h), (picks, tokens)];
    if !(pairs.iter()).all(|&(s, d)| same_layout(s, d))
        || !(pairs.iter()).all(|&(s, d)| backend.queues_copies_of(s) && backend.queues_copies_of(d))
    {
        return None;
    }
    let copies = pairs.map(|(s, d)| (span(s), span(d)));
    let order = copy_order(&copies)?;
    let mut m = Measure {
        backend: &*backend,
        skip: tokens.raw(),
        queued: true,
        lens: Vec::new(),
        spans: Vec::new(),
    };
    inputs.fill_into(next.groups, next.shape, &mut m);
    let onto_copy = (m.spans.iter()).any(|w| copies.iter().any(|(_, d)| overlaps(w, d)));
    if !m.queued || onto_copy || !io.has_room(&m.lens) {
        return None;
    }
    {
        let b: &Backend = backend;
        for i in order {
            let (s, d) = pairs[i];
            b.copy_to(b, s, d);
        }
    }
    inputs.fill_into(
        next.groups,
        next.shape,
        &mut Skip {
            inner: Queued { io, backend },
            skip: tokens.raw(),
        },
    );
    Some(Staged {
        shape: next.shape.to_vec(),
        graph: graph_id(kept),
        key: key(next.groups),
        tokens: Vec::new(),
        decided: false,
    })
}

impl Executor {
    /// Whether an MTP draft round stages each next pass's inputs while a pass computes (see
    /// the module docs; opt-in, `--mtp-ahead`): on, and nothing else runs between a round's
    /// passes.
    pub fn mtp_ahead(&self) -> bool {
        let arch = &self.model.hp.arch;
        self.mtp_ahead
            && self.mtp_enabled
            && self.model.experts().is_none()
            && self.tp.is_none()
            && self.model.tp_comm.is_none()
            && self.model.tp_pair.is_none()
            && self.remote.is_none()
            && !self.par.every_step
            && !self.mtp_overlap()
            && !arch.mtp_shares_kv()
            && !arch.mtp_sparse()
            && !self.last().backend.is_meta()
    }

    /// See [`Self::mtp_ahead`].
    pub fn set_mtp_ahead(&mut self, on: bool) {
        self.mtp_ahead = on;
    }
}
