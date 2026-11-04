//! Kept graphs dropped on another thread (#466). An expert-cache restore drops the long
//! prompt's kept graphs, and with them compute buffers of 1.5-2 GiB per stage: a `cudaFree` of
//! one usually takes ~3 ms but sometimes 40-60 ms (the device was idle: the time is the free
//! itself), which held up the first decode step after the prompt by up to ~110 ms. The restore's
//! slot thread now drops them ([`ExpertResidency::restore_after`]) before it maps the cache's
//! VRAM again. Until it has, a graph built on the engine thread may not find the memory: on an
//! allocation failure it waits for them ([`Freeing::wait`]) and allocates again.
//!
//! [`ExpertResidency::restore_after`]: crate::residency::experts::ExpertResidency::restore_after

use std::sync::mpsc;

#[cfg(test)]
mod tests;

/// Values to drop elsewhere; dropping it (all of it, wherever) wakes [`Freeing::wait`].
pub(super) struct Doomed {
    /// Declared first: dropped before the sender, so a waiter wakes after the values are gone.
    values: Vec<Box<dyn Send>>,
    _done: mpsc::Sender<()>,
}

impl Doomed {
    pub(super) fn len(&self) -> usize {
        self.values.len()
    }
}

/// Values handed to [`doomed`] and not dropped yet, if any.
#[derive(Default)]
pub(super) struct Freeing(Option<mpsc::Receiver<()>>);

impl Freeing {
    /// Waits until the last [`Doomed`] is dropped; false (at once) when none was pending.
    pub(super) fn wait(&mut self) -> bool {
        match self.0.take() {
            // Nothing is ever sent: `recv` returns once the sender is gone.
            Some(rx) => {
                let _ = rx.recv();
                true
            }
            None => false,
        }
    }
}

/// `values` to drop on another thread, and the handle to wait for it with.
pub(super) fn doomed(values: Vec<Box<dyn Send>>) -> (Doomed, Freeing) {
    let (tx, rx) = mpsc::channel();
    (Doomed { values, _done: tx }, Freeing(Some(rx)))
}
