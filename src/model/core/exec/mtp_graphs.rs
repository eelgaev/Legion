//! The MTP head's kept graphs, per kind of pass (#289). A drafter step's take-in pass has one
//! row per position the target kept plus the bonus token, so its shape follows how many drafts
//! the last verify accepted, and the drafts' passes follow how many sequences still draft. With
//! one kept graph per kind, every change of the accepted count rebuilt the take-in graph (built,
//! planned, allocated and, on CUDA, captured again). Like Strata's drafter, which captures a
//! round per window size (`MtpDrafter::capture_round`, MIT), the head now keeps the last few
//! shapes of each kind.
//!
//! The kept graphs of a kind share one compute buffer: they run one at a time (each pass waits
//! for its outputs) and set their inputs every run, so the buffer is what the single kept graph's
//! was, the largest pass of the kind, and keeping more shapes costs no device memory.

use crate::ggml::{BufferType, ComputeAllocator, Context, GgmlError, KeptGraph, SharedAllocator};
use crate::model::core::GroupShape;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[cfg(test)]
mod tests;

/// Shapes kept per kind of pass: the take-in rows of every accepted count up to a few drafts,
/// a prompt's pieces and its tail.
pub(super) const MTP_KEPT: usize = 8;

/// The executor's MTP head's default for [`KEPT`] (#858): the take-in shapes of 4 sequences
/// drafting together, ordered by their rows (`MtpDrafter::step`), are the 15 multisets of their
/// 2-4 rows at `--spec-k 2`, which 8 kept shapes cycled through with a build and a capture for
/// most passes. The kept graphs of a kind share one compute buffer: more costs host memory only.
pub(super) const MTP_KEPT_HEAD: usize = 16;

/// Shapes the MTP head keeps per kind of pass from now on (`legion serve
/// --mtp-kept-graphs`, #858; [`MTP_KEPT_HEAD`] by default). Several sequences drafting together
/// take in rows by each one's accepted count, so their take-in shapes are many more than one
/// sequence's few.
static KEPT: AtomicUsize = AtomicUsize::new(MTP_KEPT_HEAD);

/// Sets [`KEPT`] for the executors made from now on (at least 1).
pub fn set_kept(n: usize) {
    KEPT.store(n.max(1), Relaxed);
}

/// See [`set_kept`].
pub(super) fn kept() -> usize {
    KEPT.load(Relaxed)
}

/// Whether an MTP pass of `shape` waits for its graph's second run to capture it
/// ([`crate::ggml::stage_graph::ReuseScope`], #836): a take-in pass (rows other than one a
/// sequence) of several sequences, whose shape follows each one's accepted count and so seldom
/// comes back. One sequence's take-in shapes are its few accepted counts, and the draft
/// passes' follow how many sequences draft: those are captured behind their first run.
pub(crate) fn take_in_waits_for_reuse(shape: &[GroupShape]) -> bool {
    let take_in = shape.iter().any(|g| g.n_tok != 1);
    take_in && shape.iter().map(|g| g.n_seqs).sum::<usize>() > 1
}

/// One kind's kept MTP graphs, most recently used last, in one shared compute buffer.
pub(super) struct MtpGraphs {
    kept: Vec<(Vec<GroupShape>, KeptGraph)>,
    shared: Option<SharedAllocator>,
    cap: usize,
    /// Graphs built so far (for tests and logs).
    builds: u64,
}

impl MtpGraphs {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            kept: Vec::new(),
            shared: None,
            cap: cap.max(1),
            builds: 0,
        }
    }

    /// Drops every kept graph and the compute buffer.
    pub(super) fn clear(&mut self) {
        self.kept.clear();
        self.shared = None;
    }

    /// Takes every kept graph and the compute buffer out, as [`Self::clear`] drops them, for the
    /// caller to drop on another thread (#466).
    pub(super) fn take(&mut self) -> impl Send + 'static {
        (std::mem::take(&mut self.kept), self.shared.take())
    }

    /// Drops every kept graph and takes the compute buffer out (to keep, #491).
    pub(super) fn take_buffer(&mut self) -> Option<ComputeAllocator> {
        self.kept.clear();
        let shared = self.shared.take()?;
        // The kept graphs held the only other references.
        let a = std::sync::Arc::try_unwrap(shared).ok()?;
        Some(a.into_inner().unwrap_or_else(|e| e.into_inner()))
    }

    /// With no graph kept: the next graphs allocate from `allocator`'s buffer (#491).
    pub(super) fn start_with(&mut self, allocator: ComputeAllocator) {
        debug_assert!(self.kept.is_empty(), "graphs kept in the old buffer");
        self.kept.clear();
        self.shared = Some(std::sync::Arc::new(std::sync::Mutex::new(allocator)));
    }

    /// The kept graph of `shape`, if any.
    pub(super) fn get(&self, shape: &[GroupShape]) -> Option<&KeptGraph> {
        self.kept.iter().find(|(s, _)| s == shape).map(|(_, g)| g)
    }

    /// Marks the kept graph of `shape` most recently used; false if there is none.
    pub(super) fn touch(&mut self, shape: &[GroupShape]) -> bool {
        let Some(i) = self.kept.iter().position(|(s, _)| s == shape) else {
            return false;
        };
        let g = self.kept.remove(i);
        self.kept.push(g);
        true
    }

    /// Makes room for one more graph: drops the least recently used beyond the cap.
    pub(super) fn evict_for_one(&mut self) {
        while self.kept.len() >= self.cap {
            self.kept.remove(0);
        }
    }

    /// Allocates `graph` (built in `ctx`) from the kind's shared buffer, created on `buft` at
    /// first use. Should the buffer have to grow (it moves), the graphs kept so far point into
    /// the old one: they are dropped.
    pub(super) fn alloc(
        &mut self,
        ctx: &Context,
        graph: &crate::ggml::Graph<'_>,
        buft: BufferType,
    ) -> Result<(), GgmlError> {
        let shared = self
            .shared
            .get_or_insert_with(|| {
                std::sync::Arc::new(std::sync::Mutex::new(ComputeAllocator::new(buft)))
            })
            .clone();
        if ctx.alloc_graph_shared(graph, &shared)? {
            self.kept.clear();
        }
        Ok(())
    }

    /// Keeps `kept` as the graph of `shape` (built and allocated by [`Self::alloc`] after
    /// [`Self::evict_for_one`]), most recently used.
    pub(super) fn push(&mut self, shape: Vec<GroupShape>, kept: KeptGraph) -> &KeptGraph {
        self.builds += 1;
        crate::ggml::stage_graph::stats::count_mtp_build();
        kept.mark_mtp();
        self.kept.push((shape, kept));
        &self.kept.last().expect("just pushed").1
    }

    /// Bytes of the shared compute buffer (0 before the first graph).
    pub(super) fn buffer_size(&self) -> usize {
        self.shared.as_ref().map_or(0, |a| {
            a.lock().unwrap_or_else(|e| e.into_inner()).buffer_size()
        })
    }

    /// Graphs built since the start.
    pub(super) fn builds(&self) -> u64 {
        self.builds
    }

    /// Shapes kept now.
    pub(super) fn len(&self) -> usize {
        self.kept.len()
    }

    /// The shapes kept now, least recently used first.
    pub(super) fn shapes(&self) -> impl Iterator<Item = &[GroupShape]> {
        self.kept.iter().map(|(s, _)| s.as_slice())
    }

    /// Shapes kept at most.
    pub(super) fn cap(&self) -> usize {
        self.cap
    }
}
