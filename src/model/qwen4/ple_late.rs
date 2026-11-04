//! A verify step's late n-gram rows, read from the store from the step's fill on (#836).
//!
//! With the n-gram table on the SSD (direct reads, a GPU sharing the host's memory), a verify
//! step's fill leaves its rows to the step's compute (qwen4exp's `Model::set_ple_late`): the
//! stage graph's head layers run while the host reads and
//! dequantizes them, then the tail waits for the copy. A DGX Spark (GB10) profile at one
//! sequence with MTP showed the head done ~0.4 ms before the rows were (the reads, then the
//! dequantization, all started only once the head was launched). Here the fill issues the
//! reads ([`LookupTable::issue`]) and the late write collects them, so they run from the fill
//! on: through the step's launch and its head. The same rows' bytes either way, so the same
//! inputs. On by default; `legion serve --no-ple-early-reads` issues them at the write.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::residency::lookup_table::{LookupTable, Ticket};

#[cfg(test)]
mod tests;

/// Whether fills issue their late rows' reads ([`set_ple_early_reads`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether a fill that leaves its n-gram rows to the step's compute issues their store reads
/// at once (on by default) or leaves them to the write (`legion serve --no-ple-early-reads`,
/// for comparisons). The same bytes either way.
pub fn set_ple_early_reads(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_ple_early_reads`].
pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The n-gram rows a fill left to its step's compute, and their reads if the fill issued them.
pub(crate) struct LateRows {
    pub(crate) rows: Vec<i32>,
    reads: Option<Ticket>,
}

impl LateRows {
    /// `rows` for the step's compute, their reads issued on `store` now where [`on`].
    pub(crate) fn new(rows: Vec<i32>, store: &LookupTable) -> Self {
        let reads = on().then(|| store.issue(&ids(&rows)));
        Self { rows, reads }
    }

    /// The rows' bytes from `store` (the one [`Self::new`] issued on): the issued reads'
    /// results, or reads issued now.
    pub(crate) fn bytes(self, store: &LookupTable) -> std::io::Result<Vec<u8>> {
        match self.reads {
            Some(t) => store.collect(t),
            None => store.fetch(&ids(&self.rows)),
        }
    }

    /// Whether the fill issued the reads.
    #[cfg(test)]
    pub(crate) fn issued(&self) -> bool {
        self.reads.is_some()
    }
}

/// The store's row ids of `rows`.
pub(crate) fn ids(rows: &[i32]) -> Vec<u32> {
    rows.iter().map(|&r| r as u32).collect()
}
