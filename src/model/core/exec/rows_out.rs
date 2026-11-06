//! A step's captured rows that are not deferred ([`super::rows`]) to the host (#363).
//!
//! A one-micro-chunk prompt step (one stage, as on the GB10) reads its captured rows (the MTP
//! head's inputs, 168 MB per 4,096-token chunk of Qwen3.8-Flash-Next) back through the pinned
//! bounce buffer with the GPU idle: the DMA takes ~3 ms, but copying them into a new vector on
//! the engine thread took ~50 ms (page faults plus the copy), and the step then copied that
//! vector once more into its output (another ~55 ms). Now the copy out of the bounce buffer
//! runs on four threads and the output takes a micro-chunk's vector as it is when it is the
//! only one with rows. The same bytes, in the same order.
//!
//! A fresh 168 MB vector costs more than its copy on the GB10: faulting its pages in (~60 ms
//! for one thread, against ~7 ms to copy into touched memory) and unmapping them when the
//! engine drops the rows after the MTP take-in (~35 ms, with the GPU idle before the next
//! chunk). The engine hands the rows back ([`super::Executor::recycle_rows`]) and the next
//! chunk's copy goes into them ([`Spare`]).

/// Vectors kept for the next copy are at most this big, and all of them together: a
/// 4,096-token chunk's rows of Flash-Next are 168 MB; a several-stage prompt's whole output
/// (gigabytes, put together by [`super::rows`]) is not kept.
const SPARE_MAX_BYTES: usize = 1 << 30;

/// The most vectors kept. A drafter reading several of the target's layers (DFlash 2 on Gemma
/// 4 31B: five, 44 MB each per 2,048-token chunk) has that many big copies per chunk; with one
/// vector kept the other four were new vectors (their pages faulted in, ~2.5 ms each on the
/// GB10) and were unmapped again once read, with the GPU idle before the next chunk.
const SPARE_KEPT: usize = 8;

/// Captured-rows vectors the engine handed back, for the next big copies ([`copy_out`]).
#[derive(Default)]
pub(super) struct Spare(Vec<Vec<f32>>);

impl Spare {
    /// Keeps the vectors of `rows` of at least `min_bytes` and at most [`SPARE_MAX_BYTES`] of
    /// capacity, the biggest ones when more than [`SPARE_KEPT`] or more than
    /// [`SPARE_MAX_BYTES`] together would be kept; drops the others.
    pub(super) fn keep(&mut self, rows: impl IntoIterator<Item = Vec<f32>>, min_bytes: usize) {
        for v in rows {
            let bytes = v.capacity() * 4;
            if bytes >= min_bytes && bytes <= SPARE_MAX_BYTES {
                self.0.push(v);
            }
        }
        // Biggest first; the smallest go while over either limit.
        self.0.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        let mut total = 0;
        let fit = self
            .0
            .iter()
            .take_while(|v| {
                total += v.capacity() * 4;
                total <= SPARE_MAX_BYTES
            })
            .count();
        self.0.truncate(fit.min(SPARE_KEPT));
    }

    /// An empty vector with room for `n` floats: the smallest kept one that has the room.
    fn take(&mut self, n: usize) -> Vec<f32> {
        match self.0.iter().rposition(|v| v.capacity() >= n) {
            Some(i) => {
                let mut v = self.0.remove(i);
                v.clear();
                v
            }
            None => Vec::with_capacity(n),
        }
    }

    /// Capacity of the biggest kept vector, in floats.
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.0.iter().map(Vec::capacity).max().unwrap_or(0)
    }

    /// How many vectors are kept.
    #[cfg(test)]
    fn kept(&self) -> usize {
        self.0.len()
    }
}

/// `n` floats at `src` (memory the caller keeps alive and unchanged) in a vector (`spare`'s when
/// it has room, else a new one), copied on four threads off the engine thread's core.
pub(super) fn copy_out(spare: &mut Spare, src: usize, n: usize) -> Vec<f32> {
    const THREADS: usize = 4;
    let mut v = spare.take(n);
    let dst = v.as_mut_ptr() as usize;
    let pin = &crate::hw::host_core::off_host_core(&[]);
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let (lo, hi) = (n * t / THREADS, n * (t + 1) / THREADS);
            scope.spawn(move || {
                crate::hw::host_core::pin_helper(pin);
                // SAFETY: disjoint ranges of the `n` floats of capacity at `dst`, from the `n`
                // floats at `src` the caller keeps alive and unchanged.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (src as *const f32).add(lo),
                        (dst as *mut f32).add(lo),
                        hi - lo,
                    );
                }
            });
        }
    });
    // SAFETY: all `n` floats were written above (the scope joined every thread).
    unsafe { v.set_len(n) };
    v
}

/// Captured layer `layer`'s rows of every micro-chunk in `hidden`, one after another, taken out
/// of it: the one non-empty vector itself when there is at most one, else a new vector of them
/// all. Every micro-chunk captures the layer.
pub(super) fn take_layer(hidden: &mut [Vec<(usize, Vec<f32>)>], layer: usize) -> Vec<f32> {
    let mut parts: Vec<&mut Vec<f32>> = hidden
        .iter_mut()
        .map(|h| {
            &mut h
                .iter_mut()
                .find(|(l, _)| *l == layer)
                .expect("every captured layer is in a stage")
                .1
        })
        .collect();
    if parts.iter().filter(|v| !v.is_empty()).count() <= 1 {
        return parts
            .iter_mut()
            .find(|v| !v.is_empty())
            .map_or_else(Vec::new, |v| std::mem::take(*v));
    }
    // The micro-chunks' rows in one vector of the final size: growing it chunk by chunk copied
    // a prompt's multi-gigabyte rows several times over.
    let total = parts.iter().map(|v| v.len()).sum();
    let mut rows = Vec::with_capacity(total);
    for v in parts {
        rows.extend_from_slice(v);
        *v = Vec::new();
    }
    rows
}

#[cfg(test)]
mod tests;
