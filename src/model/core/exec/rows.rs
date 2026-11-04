//! Big captured rows (the MTP head's inputs, ~134 MB per 4,096-token round) whose copy out of
//! the pinned bounce buffer runs on threads beside the next round's compute: the DMA is quick,
//! but a fresh 134 MB `Vec` (page faults plus the copy) took ~29 ms of every round with the GPUs
//! idle. Two bounce buffers alternate; a buffer is joined before it is read into again, and
//! everything is joined before the step returns.
//!
//! #435: the rows go straight to their place in the step's output (one vector of every
//! micro-chunk's rows, one after another), so the step does not assemble that vector at its end
//! (a 123K prompt's 4.9 GB copied serially into a fresh vector took ~0.75 s with the GPUs idle).
//! A micro-chunk's place is known up front from the micro-chunks' token counts and the width of
//! the first rows copied out; rows of another size, or of a layer that was not deferred, fall
//! back to the per-micro-chunk vectors and the copy at the end.

/// A running row copy: the rows in a new `Vec`, or `None` when they went to [`Direct`].
type Copying = std::thread::JoinHandle<Option<Vec<f32>>>;

/// The step's deferred row copies.
pub(super) struct DeferredRows {
    /// Every micro-chunk's tokens.
    tokens: Vec<usize>,
    /// `(micro-chunk, layer, bounce parity, the copy)`.
    pending: Vec<(usize, usize, usize, Copying)>,
    direct: Option<Direct>,
}

/// One captured layer's output vector, filled in place.
struct Direct {
    layer: usize,
    /// Floats per token.
    width: usize,
    /// Capacity `width * Σ tokens`; written by the copies, read once they are joined.
    buf: Vec<f32>,
    /// Which micro-chunks' rows were copied into `buf`.
    placed: Vec<bool>,
}

impl Direct {
    /// Micro-chunk `k`'s place in `buf`: `(offset, floats)`.
    fn place(&self, tokens: &[usize], k: usize) -> (usize, usize) {
        let before: usize = tokens[..k].iter().sum();
        (before * self.width, tokens[k] * self.width)
    }
}

impl DeferredRows {
    /// For a step of micro-chunks of `tokens` tokens each.
    pub(super) fn new(tokens: Vec<usize>) -> Self {
        Self {
            tokens,
            pending: Vec::new(),
            direct: None,
        }
    }

    /// Starts copying micro-chunk `k`'s rows of `layer`, `n_bytes` at `src` (pinned memory that
    /// stays untouched until this is joined), on four threads: into the layer's output vector
    /// when they fit their place there, else into a new `Vec`.
    pub(super) fn start(
        &mut self,
        k: usize,
        layer: usize,
        parity: usize,
        src: usize,
        n_bytes: usize,
    ) {
        let n = n_bytes / 4;
        if self.direct.is_none() && n_bytes.is_multiple_of(4) && self.tokens[k] > 0 {
            let width = n / self.tokens[k];
            if width * self.tokens[k] == n {
                let total = width * self.tokens.iter().sum::<usize>();
                self.direct = Some(Direct {
                    layer,
                    width,
                    buf: Vec::with_capacity(total),
                    placed: vec![false; self.tokens.len()],
                });
            }
        }
        let dst = match self.direct.as_mut() {
            Some(d) if d.layer == layer && !d.placed[k] && n == self.tokens[k] * d.width => {
                let (at, _) = d.place(&self.tokens, k);
                d.placed[k] = true;
                // SAFETY: `at + n` is within the capacity; the vector is neither moved nor
                // dropped before the copy is joined (`assemble`, `Drop`).
                Some(unsafe { d.buf.as_mut_ptr().add(at) } as usize)
            }
            _ => None,
        };
        // Off the engine thread's core, with the four copy threads it starts (#250).
        let pin = crate::hw::host_core::off_host_core(&[]);
        let handle = std::thread::spawn(move || {
            crate::hw::host_core::pin_helper(&pin);
            let mut v: Vec<f32> = Vec::new();
            let to = match dst {
                Some(to) => to,
                None => {
                    v.reserve_exact(n);
                    v.as_mut_ptr() as usize
                }
            };
            copy_threads(src, to, n);
            if dst.is_some() {
                return None;
            }
            // SAFETY: all `n` floats were written above.
            unsafe { v.set_len(n) };
            Some(v)
        });
        self.pending.push((k, layer, parity, handle));
    }

    /// Waits for the copies out of bounce buffer `parity` (all for `None`) and puts the rows that
    /// got a new `Vec` in `hidden`.
    pub(super) fn join(&mut self, parity: Option<usize>, hidden: &mut [Vec<(usize, Vec<f32>)>]) {
        let mut rest = Vec::new();
        for (k, layer, p, h) in std::mem::take(&mut self.pending) {
            if parity.is_none_or(|x| x == p) {
                let rows = h.join().expect("a row copy panicked");
                if let (Some(rows), Some(e)) =
                    (rows, hidden[k].iter_mut().find(|(l, _)| *l == layer))
                {
                    e.1 = rows;
                }
            } else {
                rest.push((k, layer, p, h));
            }
        }
        self.pending = rest;
    }

    /// Captured layer `layer`'s rows of every micro-chunk, one after another, taking them out
    /// of `hidden`: the vector the copies filled in place, completed with the rows that were
    /// not deferred. `None` when no copy went in place for the layer. Joins every copy first.
    pub(super) fn assemble(
        &mut self,
        layer: usize,
        hidden: &mut [Vec<(usize, Vec<f32>)>],
    ) -> Option<Vec<f32>> {
        self.join(None, hidden);
        if self.direct.as_ref()?.layer != layer {
            return None;
        }
        let mut d = self.direct.take()?;
        let mut rows: Vec<&mut Vec<f32>> = hidden
            .iter_mut()
            .map(|h| {
                &mut h
                    .iter_mut()
                    .find(|(l, _)| *l == layer)
                    .expect("every micro-chunk captures the layer")
                    .1
            })
            .collect();
        assert_eq!(rows.len(), self.tokens.len(), "a row list per micro-chunk");
        let fits =
            (0..rows.len()).all(|k| d.placed[k] || rows[k].len() == d.place(&self.tokens, k).1);
        if fits {
            for (k, r) in rows.iter_mut().enumerate() {
                if !d.placed[k] {
                    let (at, n) = d.place(&self.tokens, k);
                    // SAFETY: micro-chunk `k`'s place, within the capacity.
                    unsafe {
                        std::ptr::copy_nonoverlapping(r.as_ptr(), d.buf.as_mut_ptr().add(at), n)
                    };
                }
                **r = Vec::new();
            }
            // SAFETY: every micro-chunk's place was written (in place, joined above, or just
            // now), and the places tile the capacity.
            unsafe { d.buf.set_len(d.buf.capacity()) };
            return Some(d.buf);
        }
        // Rows of an unexpected size: one after another, in place or not.
        let mut out = Vec::new();
        for (k, r) in rows.iter_mut().enumerate() {
            if d.placed[k] {
                let (at, n) = d.place(&self.tokens, k);
                // SAFETY: written by a joined copy.
                out.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(d.buf.as_ptr().add(at), n)
                });
            } else {
                out.extend_from_slice(r);
            }
            **r = Vec::new();
        }
        Some(out)
    }
}

impl Drop for DeferredRows {
    fn drop(&mut self) {
        // An early return: the copies read pinned memory the stages still own, and write the
        // output vector.
        for (.., h) in self.pending.drain(..) {
            let _ = h.join();
        }
    }
}

/// Copies `n` floats from `src` to `dst` on four threads.
fn copy_threads(src: usize, dst: usize, n: usize) {
    const THREADS: usize = 4;
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let (lo, hi) = (n * t / THREADS, n * (t + 1) / THREADS);
            scope.spawn(move || {
                // SAFETY: disjoint ranges of `n` floats at `dst` (capacity the caller keeps
                // alive and nobody else writes), from `n` floats at `src` the caller keeps alive
                // and unchanged.
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
}

/// Writes `parts` one after another into `dst` (exactly their total size), on four threads when
/// big: an MTP take-in piece's rows (a 1,024-token piece of Flash-Next is 40 MB) go straight
/// from the step's output into the pinned bounce buffer, instead of first into a new vector
/// (#435).
pub(super) fn gather(parts: &[&[f32]], dst: &mut [f32]) {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    assert_eq!(total, dst.len(), "the parts fill the destination");
    const THREADS: usize = 4;
    if total * 4 < super::BOUNCE_MIN_BYTES {
        gather_range(parts, 0, dst);
        return;
    }
    let per = total.div_ceil(THREADS);
    let pin = &crate::hw::host_core::off_host_core(&[]);
    std::thread::scope(|scope| {
        for (t, chunk) in dst.chunks_mut(per).enumerate() {
            scope.spawn(move || {
                crate::hw::host_core::pin_helper(pin);
                gather_range(parts, t * per, chunk)
            });
        }
    });
}

/// `dst` = the floats `[from, from + dst.len())` of `parts` one after another.
fn gather_range(parts: &[&[f32]], from: usize, dst: &mut [f32]) {
    let (mut at, mut skip) = (0, from);
    for p in parts {
        if at == dst.len() {
            break;
        }
        if skip >= p.len() {
            skip -= p.len();
            continue;
        }
        let n = (p.len() - skip).min(dst.len() - at);
        dst[at..at + n].copy_from_slice(&p[skip..skip + n]);
        at += n;
        skip = 0;
    }
}

#[cfg(test)]
mod tests;
