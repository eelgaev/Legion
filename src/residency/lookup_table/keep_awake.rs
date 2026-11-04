//! Keeps the direct tier's SSD out of its power-saving states while the table is in use (#835).
//!
//! An NVMe drive with APST on drops to a low-power state after tens of ms without a command;
//! the first read after that pays the exit, ~2 ms on a DGX Spark's SSD (`default_ps_max_latency_us`
//! 100000). A C=16 decode step takes ~137 ms, so every step's reads met a sleeping drive and the
//! GPU waited ~1 ms per step for its n-gram rows. With `--ngram-keep-awake-ms N`, a helper thread
//! reads one 4 KiB page of the table (discarded) whenever no read went out for N ms, as long as
//! the last [`KeepAwake::touch`] (a fetch) is under [`WINDOW`] old; then it sleeps until the next
//! fetch. One page per N ms is ~100 KB/s at N = 40. The rows are unchanged: only I/O timing.

use std::fs::File;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::{Aligned, PAGE, read_full};
use crate::hw::numa::pin_current_thread;

/// How long after the last fetch the drive is kept awake.
pub(super) const WINDOW: Duration = Duration::from_secs(1);

struct State {
    /// The last fetch.
    fetch: Option<Instant>,
    /// The last read of the table (a fetch's or this thread's).
    read: Option<Instant>,
    reads: u64,
    stop: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    window: Duration,
}

/// The helper thread; stopped and joined on drop.
pub(super) struct KeepAwake {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl KeepAwake {
    /// Starts a thread that reads a page of `file`'s bytes `start..start + len` every `interval`
    /// without other reads, until `window` after the last fetch.
    pub(super) fn start(
        file: Arc<File>,
        start: u64,
        len: u64,
        interval: Duration,
        window: Duration,
        cpus: &[u32],
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                fetch: None,
                read: None,
                reads: 0,
                stop: false,
            }),
            wake: Condvar::new(),
            window,
        });
        let first = start - start % PAGE;
        let pages = ((start + len).next_multiple_of(PAGE) - first) / PAGE;
        let (s, cpus) = (shared.clone(), cpus.to_vec());
        let thread = std::thread::Builder::new()
            .name("table-keep-awake".into())
            .spawn(move || {
                if !cpus.is_empty() {
                    let _ = pin_current_thread(&cpus);
                }
                run(&file, &s, first, pages.max(1), interval)
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// A fetch: `read` when it reads the file (not only the row cache).
    pub(super) fn touch(&self, read: bool) {
        let now = Instant::now();
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let idle = st.fetch.is_none_or(|f| now - f >= self.shared.window);
        st.fetch = Some(now);
        if read {
            st.read = Some(now);
        }
        drop(st);
        if idle {
            self.shared.wake.notify_all();
        }
    }

    /// The pages read to keep the drive awake.
    pub(super) fn reads(&self) -> u64 {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reads
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stop = true;
        self.shared.wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn run(file: &File, s: &Shared, first: u64, pages: u64, interval: Duration) {
    let mut buf = Aligned::new(PAGE as usize);
    // Pages visited in a fixed scattered order, so the drive's own cache does not answer them.
    let mut k = 0u64;
    let mut st = s.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if st.stop {
            return;
        }
        let now = Instant::now();
        let Some(fetch) = st.fetch.filter(|&f| now - f < s.window) else {
            // Idle: sleep until a fetch.
            st = s.wake.wait(st).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let last = st.read.map_or(fetch, |r| r.max(fetch));
        let since = now.saturating_duration_since(last);
        if since < interval {
            st = s
                .wake
                .wait_timeout(st, interval - since)
                .unwrap_or_else(|e| e.into_inner())
                .0;
            continue;
        }
        drop(st);
        let page = first + (k.wrapping_mul(0x9E37_79B9) % pages) * PAGE;
        k += 1;
        // An error or a short read changes nothing: the next fetch reports its own.
        let _ = read_full(file, buf.slice(PAGE as usize), page);
        st = s.state.lock().unwrap_or_else(|e| e.into_inner());
        st.read = Some(Instant::now());
        st.reads += 1;
    }
}
