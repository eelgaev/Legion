//! How the node link's threads wait for the next frame (#1176): spinning, then napping, then
//! blocking. A thread blocked in a read or a channel's receive is woken by the scheduler, which
//! on two DGX Sparks cost 0.5-1 ms a message once the cores idle between steps: a step's request
//! reached rank 1 that late, and rank 0 waited for it at the step's first all-reduce, twice a
//! decode round (the verify step, the draft round). A wait that spins sees the frame within
//! microseconds; the data connections' readers wait the same way
//! ([`crate::tp_net`]'s `next_job`).
//!
//! `LEGION_DEBUG_NODE_LINK_SPIN=0` blocks at once (the waits before), for A/B runs. A wait
//! that the other thread knows will end soon is rung ([`Bell`]) and spins again.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, mpsc};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

/// How a wait goes: spin for `spin`, then sleep `nap` at a time until `nap_for` has passed
/// since it began; after that the caller blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pace {
    pub spin: Duration,
    pub nap: Duration,
    pub nap_for: Duration,
}

/// The link's waits: a decode round's requests come within a few ms of each other (spun), a
/// verify step's commit some 100 ms after its step (napped, ~0.1 ms late at most, unless the
/// wait was rung: [`Bell`]); a link
/// with nothing to carry for a second blocks.
pub const LINK: Pace = Pace {
    spin: Duration::from_millis(5),
    nap: Duration::from_micros(50),
    nap_for: Duration::from_secs(1),
};

/// Whether the link's waits spin ([`LINK`]); `LEGION_DEBUG_NODE_LINK_SPIN=0` blocks at once
/// (for A/B runs only).
pub fn spins() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_NODE_LINK_SPIN").is_none_or(|v| v != "0"))
}

/// Whether the link's threads ring each other's [`Bell`] (#1176 r51);
/// `LEGION_DEBUG_NODE_LINK_BELL=0` never rings (the waits before, for A/B runs only).
pub fn bells() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_NODE_LINK_BELL").is_none_or(|v| v != "0"))
}

/// The timer slack a link thread asks for ([`tight_slack`]): 1 µs instead of Linux's default
/// 50 µs.
pub const SLACK_NS: u64 = 1_000;

/// Gives the calling thread (a link thread) a 1 µs timer slack (#1176 r51), so a 50 µs nap
/// lasts ~50 µs instead of ~0.1 ms; with `LEGION_DEBUG_NODE_LINK_BELL=0` (or no spinning) it
/// keeps the default. Only the calling thread's timers change. True when it was set.
pub fn tight_slack() -> bool {
    if !(spins() && bells()) {
        return false;
    }
    set_slack(SLACK_NS)
}

/// Sets the calling thread's timer slack to `ns` (Linux; elsewhere it does nothing).
pub fn set_slack(ns: u64) -> bool {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: PR_SET_TIMERSLACK takes the slack in ns by value and changes only the
        // calling thread.
        let none: libc::c_ulong = 0;
        unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, ns as libc::c_ulong, none, none, none) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = ns;
        false
    }
}

/// A wait's doorbell (#1176 r51): another thread rings it when a frame is about to come, and
/// the wait spins again from then on, woken at once if it was napping. A verify step's commit
/// comes some 40-100 ms after the step's request, long after the wait stopped spinning, and a
/// nap (50 µs plus the kernel's timer slack, ~0.1 ms) made rank 1 start its commit 0.1-0.2 ms
/// after rank 0 on two Sparks; rank 0 then waited for it at the draft round's first
/// all-reduce. Rank 1 rings its reader when it answers a request (the next ones come right
/// after: the commit and the draft round follow the verify step's reply), and rank 0 its
/// writer when it queues a frame.
#[derive(Debug, Default)]
pub struct Bell {
    rung: AtomicBool,
    napping: AtomicBool,
    waiter: OnceLock<std::thread::Thread>,
}

impl Bell {
    /// Rings: the wait spins again, and stops napping now if it was.
    pub fn ring(&self) {
        self.rung.store(true, Ordering::SeqCst);
        if self.napping.load(Ordering::SeqCst)
            && let Some(t) = self.waiter.get()
        {
            t.unpark();
        }
    }

    /// Whether it rang since the last call.
    fn take(&self) -> bool {
        self.rung.swap(false, Ordering::SeqCst)
    }

    /// Naps up to `nap`, less if it rings: the waiting thread calls this.
    fn nap(&self, nap: Duration) {
        self.waiter.get_or_init(std::thread::current);
        self.napping.store(true, Ordering::SeqCst);
        // A ring before `napping` was set did not unpark: it is seen here instead.
        if !self.rung.load(Ordering::SeqCst) {
            std::thread::park_timeout(nap);
        }
        self.napping.store(false, Ordering::SeqCst);
    }
}

/// Calls `ready` until it is true, as `pace` says: true once it was, false when the time to
/// nap is over (the caller then blocks).
pub fn until(pace: Pace, ready: impl FnMut() -> bool) -> bool {
    until_rung(pace, None, ready)
}

/// [`until`], spinning again from each ring of `bell`.
pub fn until_rung(pace: Pace, bell: Option<&Bell>, mut ready: impl FnMut() -> bool) -> bool {
    let mut t0 = Instant::now();
    let mut tries = 0u32;
    loop {
        if ready() {
            return true;
        }
        tries = tries.wrapping_add(1);
        // The clock is read every few tries while spinning (a try is a syscall or less).
        if !tries.is_multiple_of(16) {
            std::hint::spin_loop();
            continue;
        }
        if bell.is_some_and(Bell::take) {
            t0 = Instant::now();
        }
        let waited = t0.elapsed();
        if waited >= pace.nap_for {
            return false;
        }
        if waited < pace.spin {
            std::hint::spin_loop();
        } else if let Some(b) = bell {
            b.nap(pace.nap);
        } else {
            std::thread::sleep(pace.nap);
        }
    }
}

/// The next item on `rx`: waited for as [`until_rung`] says when `pace` is given, then
/// blocked for. `None` once the channel is closed.
pub fn recv<T>(rx: &mpsc::Receiver<T>, pace: Option<Pace>, bell: Option<&Bell>) -> Option<T> {
    if let Some(pace) = pace {
        let mut got = None;
        let mut closed = false;
        until_rung(pace, bell, || match rx.try_recv() {
            Ok(v) => {
                got = Some(v);
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => {
                closed = true;
                true
            }
        });
        if got.is_some() || closed {
            return got;
        }
    }
    rx.recv().ok()
}

/// Whether a read of `s` would not block: bytes (or its end, or an error) are there to read.
fn readable(s: &std::net::TcpStream) -> bool {
    use std::os::fd::AsRawFd;
    let mut b = 0u8;
    // SAFETY: a one-byte peek into `b`, which outlives the call, on an open socket.
    let n = unsafe {
        libc::recv(
            s.as_raw_fd(),
            (&raw mut b).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    n >= 0
        || !matches!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EAGAIN | libc::EINTR)
        )
}

/// Before a blocking read of `r`: waits as [`until_rung`] says for its next bytes, unless some
/// are buffered already. Only a hint: the read that follows blocks if they did not come.
pub fn for_frame(
    r: &std::io::BufReader<std::net::TcpStream>,
    pace: Option<Pace>,
    bell: Option<&Bell>,
) {
    if let Some(pace) = pace
        && r.buffer().is_empty()
    {
        until_rung(pace, bell, || readable(r.get_ref()));
    }
}
