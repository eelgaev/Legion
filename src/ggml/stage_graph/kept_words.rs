//! A kept graph's words check skipped while nothing it reads can have changed (#814).
//!
//! Every stage-graph run compares the graph's words with its last run's ([`super::props_same`]):
//! each node's op, shape, strides, parameters and data, and each source's. On a
//! Qwen3.8-Flash-Next decode step over four V100s that is ~40-45 µs of host time per graph,
//! paid on the critical path where nothing runs ahead of the launch: the first stage of a step
//! and each MTP draft pass (#814's hand-off profile).
//!
//! A kept graph ([`crate::ggml::KeptGraph`]) is built, allocated and then never written: its
//! nodes stay where they are with the same fields, and the tensors of other contexts it reads
//! (weights, caches) must stay live at the same memory for as long as it runs (its `new`'s
//! contract). So once a run of a kept graph compared its words in the steady (shorter) form,
//! the next run of the same kept graph (by its serial, unique even where a dropped graph's
//! address is reused) on the same backend has the same words, as long as the late input is
//! first read at the same node. The run then takes the unchanged path without walking them.
//!
//! What could still change them is code that rewrites tensors' fields in place outside a
//! build: [`words_may_change`] marks that (a new epoch), and the next run of every graph
//! compares in full again. As a guard, every [`VERIFY_EVERY`]th run in a row compares in full
//! anyway and logs a warning if the words changed without either.
//!
//! The same holds for every way a kept graph runs ([`crate::ggml::Backend::compute_kept`], its
//! `_with` and `_async` forms and [`crate::ggml::Backend::launcher`]'s threads), and it cuts the
//! rest of such a run's host work before its first launch too: the late input's first reader is
//! found once per kept graph and input ([`LateAt`]), and a run on the unchanged path leaves the
//! graph's cache entry with the graph ([`KeptSlot`]) rather than in the shared, locked cache.
//! On by default; `legion serve --no-stage-graph-words-skip` turns all of it off (every run
//! compares, searches and goes through the shared cache).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(test)]
mod tests;

/// Whether unchanged kept graphs skip their words check (on by default).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether a kept graph's runs keep its entry and captures with the graph whatever they did
/// ([`hold`]; off by default).
static HOLD_ALL: AtomicBool = AtomicBool::new(false);

/// Bumped by [`words_may_change`].
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Runs in a row of an unchanged graph between full compares while skipping.
pub(crate) const VERIFY_EVERY: u32 = 64;

/// Turns the skip on or off for the runs from now on (`--no-stage-graph-words-skip`: off).
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_enabled`].
pub(crate) fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Makes every run of a kept graph keep its entry and set of captures in the graph's
/// [`KeptSlot`] (`legion serve --stage-graph-hold-kept`, #858), not only the runs on the
/// unchanged path: a kept graph's captures then last as long as the graph, rather than going
/// into the shared store after its first runs, where the store's cap (64 sets a device) and its
/// 10 s sweep dropped them while the graph stayed kept (hundreds of kept MTP take-in graphs at 4
/// parallel requests, each run every few seconds), and it captured again. Only while the skip
/// is on ([`enabled`]).
pub fn set_hold_all(on: bool) {
    HOLD_ALL.store(on, Ordering::Relaxed);
}

/// See [`set_hold_all`].
pub(crate) fn hold_all() -> bool {
    HOLD_ALL.load(Ordering::Relaxed)
}

/// Marks that tensors kept graphs read may have changed in place (not by a new kept graph):
/// every graph's next run compares its words in full.
pub fn words_may_change() {
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// See [`words_may_change`].
pub(crate) fn epoch() -> u64 {
    EPOCH.load(Ordering::Relaxed)
}

/// What a graph's last full compare ran under: the kept graph's serial and the epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Seen {
    pub serial: u64,
    pub epoch: u64,
}

/// What a run records after comparing its words in full: `kept` (its kept graph's serial,
/// `None` for a graph built for this run alone) at `epoch`.
pub(crate) fn seen(kept: Option<u64>, epoch: u64) -> Option<Seen> {
    kept.map(|serial| Seen { serial, epoch })
}

/// What a run knows before its check, for [`skip`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Check {
    /// The skip is on ([`enabled`]).
    pub on: bool,
    /// The graph's kept serial this run (`None`: not a kept graph).
    pub kept: Option<u64>,
    /// The epoch now ([`epoch`]).
    pub epoch: u64,
    /// The entry's last full compare ([`seen`]).
    pub seen: Option<Seen>,
    /// The entry's words are in the steady form: compared again unchanged since they last
    /// changed (its `weighed`), so the unchanged path does nothing but count the run.
    pub weighed: bool,
    /// Runs in a row with the same words so far (its `same`).
    pub same: u32,
    /// The last run's late-input place (its words' last word), and this run's.
    pub last_tail: Option<u64>,
    pub tail: u64,
}

/// Whether this run may take the unchanged path without comparing its words (see the module
/// docs): the same kept graph as the last full compare, under the same epoch, in the steady
/// form, with the late input at the same place, and not a run due for a full compare.
pub(crate) fn skip(c: Check) -> bool {
    let Some(serial) = c.kept else {
        return false;
    };
    c.on && c.seen
        == Some(Seen {
            serial,
            epoch: c.epoch,
        })
        && c.weighed
        && c.last_tail == Some(c.tail)
        && !c.same.saturating_add(1).is_multiple_of(VERIFY_EVERY)
}

/// A kept graph's own place for its stage-graph cache entry ([`crate::ggml::KeptGraph`] holds
/// one): between two runs that both took the unchanged path ([`skip`]), the entry and its set
/// of captures stay here instead of going back into the shared cache, so such a run takes them
/// without the cache's lock, hash lookups and clock reads. Any other run takes the shared
/// cache's way, which keeps its sweep of unused entries and its cap on sets of captures; a
/// graph dropped with its entry held here puts its captures back into the store
/// ([`super::store`]), where a graph built again with the same words finds them.
#[derive(Default)]
pub(crate) struct KeptSlot {
    #[cfg(feature = "cuda")]
    pub(super) held: std::sync::Mutex<Option<super::run::Held>>,
    /// The graph ran before (for `--stage-graph-stats`' [`super::stats::Why`]).
    #[cfg(feature = "cuda")]
    pub(super) ran: std::sync::atomic::AtomicBool,
    /// The graph is the MTP head's (for `--stage-graph-stats`' [`super::stats::Split`]).
    pub(crate) mtp: std::sync::atomic::AtomicBool,
}

/// A kept graph as a run of it sees it: its serial ([`crate::ggml::KeptGraph::serial`]) and its
/// [`KeptSlot`].
#[derive(Clone, Copy)]
pub(crate) struct KeptRun<'k> {
    pub serial: u64,
    pub slot: &'k KeptSlot,
}

/// Whether a run keeps its entry in its kept graph's [`KeptSlot`] afterwards: a kept graph's
/// run that took the unchanged path (`skipped`) and made no new set of captures (`made`; a new
/// set goes into the store, under its cap), or any run of a kept graph with `all`
/// ([`set_hold_all`]).
pub(crate) fn hold(kept: bool, skipped: bool, made: bool, all: bool) -> bool {
    kept && (all || (skipped && !made))
}

/// Where a kept graph's late input was first read on an earlier run, so the next run with the
/// same input need not walk the graph's nodes again for it (#814: an O(nodes) scan every run of
/// a step's first stage when the n-gram rows arrive late).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LateAt {
    /// The kept graph's serial and the epoch ([`epoch`]) of the run that found it.
    pub serial: u64,
    pub epoch: u64,
    /// The late input (its tensor's address) and the graph's node count then.
    pub tensor: usize,
    pub nodes: i32,
    /// Its first reader's node index (0 if none reads it).
    pub at: i32,
}

impl LateAt {
    /// `at` again for a run of kept graph `kept` (`None`: not a kept graph) of `nodes` nodes
    /// under `epoch` with late input `tensor`, if `last` found it for the same, with the skip
    /// on ([`enabled`]).
    pub(crate) fn again(
        last: Option<LateAt>,
        on: bool,
        kept: Option<u64>,
        epoch: u64,
        tensor: usize,
        nodes: i32,
    ) -> Option<i32> {
        let l = last?;
        (on && kept == Some(l.serial) && l.epoch == epoch && l.tensor == tensor && l.nodes == nodes)
            .then_some(l.at)
    }
}
