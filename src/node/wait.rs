//! How the node link's threads wait for the next frame (#1176): spinning, then napping, then
//! blocking. A thread blocked in a read or a channel's receive is woken by the scheduler, which
//! on two DGX Sparks cost 0.5-1 ms a message once the cores idle between steps: a step's request
//! reached rank 1 that late, and rank 0 waited for it at the step's first all-reduce, twice a
//! decode round (the verify step, the draft round). A wait that spins sees the frame within
//! microseconds; the data connections' readers wait the same way
//! ([`crate::tp_net`]'s `next_job`).
//!
//! `LEGION_DEBUG_NODE_LINK_SPIN=0` blocks at once (the waits before), for A/B runs.

use std::sync::mpsc;
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
/// verify step's commit some 100 ms after its step (napped, ~0.1 ms late at most); a link
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

/// Calls `ready` until it is true, as `pace` says: true once it was, false when the time to
/// nap is over (the caller then blocks).
pub fn until(pace: Pace, mut ready: impl FnMut() -> bool) -> bool {
    let t0 = Instant::now();
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
        let waited = t0.elapsed();
        if waited >= pace.nap_for {
            return false;
        }
        if waited >= pace.spin {
            std::thread::sleep(pace.nap);
        } else {
            std::hint::spin_loop();
        }
    }
}

/// The next item on `rx`: waited for as [`until`] says when `pace` is given, then blocked for.
/// `None` once the channel is closed.
pub fn recv<T>(rx: &mpsc::Receiver<T>, pace: Option<Pace>) -> Option<T> {
    if let Some(pace) = pace {
        let mut got = None;
        let mut closed = false;
        until(pace, || match rx.try_recv() {
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

/// Before a blocking read of `r`: waits as [`until`] says for its next bytes, unless some are
/// buffered already. Only a hint: the read that follows blocks if they did not come.
pub fn for_frame(r: &std::io::BufReader<std::net::TcpStream>, pace: Option<Pace>) {
    if let Some(pace) = pace
        && r.buffer().is_empty()
    {
        until(pace, || readable(r.get_ref()));
    }
}
