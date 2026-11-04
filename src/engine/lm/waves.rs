//! `--deterministic` admission (#836): requests are admitted in waves, in a fixed order.
//!
//! Requests decoded together share steps, and a step's size and rows move the sums (#49), so
//! concurrent output follows which step each request joined, i.e. arrival timing. Under
//! `--deterministic` the engine therefore starts a wave only when it is idle: it waits until
//! no request has arrived for the gather window, orders what it holds by content (prompt,
//! then the request's limits; arrival order among equal ones), and admits those. Requests
//! that arrive while a wave runs are held until the engine is idle again. A client that sends
//! its requests together (the det client's C=16) then gets the same steps every run.

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// The gather window's cap, in windows: a client that keeps sending at shorter gaps than the
/// window still starts after this many.
pub(super) const MAX_WINDOWS: u32 = 20;

/// Receives from `rx` into `held` until nothing arrived for `quiet` (or `quiet` times
/// [`MAX_WINDOWS`] passed). False once the sender is gone.
pub(super) fn gather<T>(rx: &Receiver<T>, quiet: Duration, held: &mut Vec<T>) -> bool {
    let end = Instant::now() + quiet * MAX_WINDOWS;
    loop {
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return true;
        }
        match rx.recv_timeout(quiet.min(left)) {
            Ok(item) => held.push(item),
            Err(RecvTimeoutError::Timeout) => return true,
            Err(RecvTimeoutError::Disconnected) => return false,
        }
    }
}

/// What orders a wave: the prompt's tokens first, then the request's limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct WaveKey<'a> {
    pub prompt: &'a [u32],
    pub max_tokens: u32,
    pub min_tokens: u32,
    pub ignore_eos: bool,
}

/// Sorts a wave by `key`, stably (equal keys keep their arrival order).
pub(super) fn order<T>(held: &mut [T], key: impl Fn(&T) -> WaveKey<'_>) {
    held.sort_by(|a, b| key(a).cmp(&key(b)));
}

#[cfg(test)]
mod tests;
