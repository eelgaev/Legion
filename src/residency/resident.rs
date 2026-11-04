//! Keeping a file-mapped range of host memory resident: faulted in at load by several threads
//! on one NUMA node, then locked, so reads from it never wait on the disk. qwen4exp's per-layer
//! n-gram table (`per_layer_token_embd.weight`, ~27 GiB, gathered row by row on the CPU every
//! step) stays in host RAM this way: a row missing from the page cache would otherwise be a
//! disk read inside a decode step.
//!
//! Adapted from Strata-AC922's `PleTable::open` (`src/kernels/ngram.cpp`, `--ple-io ram`; MIT):
//! `madvise(MADV_WILLNEED)`, a pool of up to 16 threads touching one byte per 4 KiB page of 64
//! MiB pieces taken from a shared counter (one toucher, or `mlock` alone, faults a cold file in
//! at ~0.5 GB/s), then `mlock` of the whole range. Unlike Strata the touchers run on one node's
//! CPUs, so the page cache lands on the node of the GPU the rows feed.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::hw::numa::{Topology, pin_current_thread};

/// Bytes each toucher takes at a time.
const PIECE: usize = 64 << 20;
/// Page size the touchers step by (the smallest Linux page; larger pages are touched more than
/// once, which costs nothing).
const PAGE: usize = 4096;
/// Most touching threads.
const MAX_THREADS: usize = 16;

/// What [`make_resident`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct Resident {
    /// Bytes faulted in (the range rounded out to whole pages).
    pub bytes: usize,
    /// Whether the whole range is locked. A partial lock is given back: the rest could still be
    /// reclaimed, so it would not be a guarantee.
    pub locked: bool,
    /// Why locking failed, if it did.
    pub lock_error: Option<String>,
    pub seconds: f64,
}

/// Faults `range` in with up to 16 threads (pinned to `node`'s CPUs when given; page-cache
/// pages are then allocated there, unless the process's memory policy says otherwise) and
/// locks it.
pub fn make_resident(range: &[u8], node: Option<(&Topology, u32)>) -> Resident {
    let t = std::time::Instant::now();
    if range.is_empty() {
        return Resident {
            bytes: 0,
            locked: true,
            lock_error: None,
            seconds: 0.0,
        };
    }
    let a0 = range.as_ptr() as usize & !(PAGE - 1);
    let a1 = range.as_ptr() as usize + range.len();
    let len = a1 - a0;
    let base = NonNull::new(a0 as *mut c_void).expect("non-null mapping");
    // SAFETY: an advice on pages of a live mapping that `range` borrows.
    let _ =
        unsafe { nix::sys::mman::madvise(base, len, nix::sys::mman::MmapAdvise::MADV_WILLNEED) };
    let cpus: Option<Vec<u32>> = node.and_then(|(topo, id)| {
        topo.nodes
            .iter()
            .find(|n| n.id == id && !n.cpus.is_empty())
            .map(|n| n.cpus.clone())
    });
    let threads = cpus
        .as_ref()
        .map_or_else(
            || std::thread::available_parallelism().map_or(1, |n| n.get()),
            |c| c.len(),
        )
        .clamp(1, MAX_THREADS);
    let pieces = len.div_ceil(PIECE);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, cpus) = (&next, cpus.as_deref());
            scope.spawn(move || {
                if let Some(cpus) = cpus {
                    let _ = pin_current_thread(cpus);
                }
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pieces {
                        break;
                    }
                    let (p0, p1) = (a0 + i * PIECE, (a0 + (i + 1) * PIECE).min(a1));
                    let mut sink = 0u8;
                    for p in (p0..p1).step_by(PAGE) {
                        // SAFETY: a byte inside the borrowed mapping (rounded down to its page,
                        // which the mapping covers).
                        sink =
                            sink.wrapping_add(unsafe { std::ptr::read_volatile(p as *const u8) });
                    }
                    std::hint::black_box(sink);
                }
            });
        }
    });
    // SAFETY: locking and unlocking pages of a live mapping that `range` borrows.
    let lock_error = match unsafe { nix::sys::mman::mlock(base, len) } {
        Ok(()) => None,
        Err(e) => {
            let _ = unsafe { nix::sys::mman::munlock(base, len) };
            Some(e.to_string())
        }
    };
    Resident {
        bytes: len,
        locked: lock_error.is_none(),
        lock_error,
        seconds: t.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests;
