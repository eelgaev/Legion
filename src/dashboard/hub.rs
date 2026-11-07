//! The few counters the engine feeds the dashboard directly. Without a dashboard attached
//! (`legion serve --tui`, or a `/v1/stats` client in the last minute) [`wanted`] is false and
//! the engine skips them: one relaxed atomic load per step. The tensor-parallel exchange's bytes
//! are counted always (one relaxed add per frame).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static WANTED: AtomicBool = AtomicBool::new(false);

/// Draft positions tracked by [`record_spec`].
pub const SPEC_POSITIONS: usize = 16;

static SPEC_DRAFTED: [AtomicU64; SPEC_POSITIONS] = [const { AtomicU64::new(0) }; SPEC_POSITIONS];
static SPEC_ACCEPTED: [AtomicU64; SPEC_POSITIONS] = [const { AtomicU64::new(0) }; SPEC_POSITIONS];

/// The engine thread's step phases, as `engine::lm::host_time` names them.
pub const PHASES: [&str; 7] = ["loop", "plan", "pack", "exec", "sample", "device", "finish"];

static PHASE_NS: [AtomicU64; PHASES.len()] = [const { AtomicU64::new(0) }; PHASES.len()];
static PHASE_STEPS: AtomicU64 = AtomicU64::new(0);

static TP_TX: AtomicU64 = AtomicU64::new(0);
static TP_RX: AtomicU64 = AtomicU64::new(0);
static TP_USED: AtomicBool = AtomicBool::new(false);

/// Whether a dashboard is attached.
#[inline]
pub fn wanted() -> bool {
    WANTED.load(Ordering::Relaxed)
}

/// Marks a dashboard attached (or gone).
pub fn set_wanted(on: bool) {
    WANTED.store(on, Ordering::Relaxed);
}

/// One sequence's verify step: `drafted` drafts, the first `accepted` of them accepted.
#[inline]
pub fn record_spec(drafted: usize, accepted: usize) {
    if !wanted() {
        return;
    }
    for i in 0..drafted.min(SPEC_POSITIONS) {
        SPEC_DRAFTED[i].fetch_add(1, Ordering::Relaxed);
        if i < accepted {
            SPEC_ACCEPTED[i].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// `[drafted, accepted]` per draft position, up to the last position ever drafted.
pub fn spec_positions() -> Vec<[u64; 2]> {
    let mut out: Vec<[u64; 2]> = (0..SPEC_POSITIONS)
        .map(|i| {
            [
                SPEC_DRAFTED[i].load(Ordering::Relaxed),
                SPEC_ACCEPTED[i].load(Ordering::Relaxed),
            ]
        })
        .collect();
    while out.last().is_some_and(|p| p[0] == 0) {
        out.pop();
    }
    out
}

/// Adds one step's time by phase (nanoseconds, in [`PHASES`] order).
pub fn add_step(ns: &[u64; PHASES.len()]) {
    for (c, &n) in PHASE_NS.iter().zip(ns) {
        if n > 0 {
            c.fetch_add(n, Ordering::Relaxed);
        }
    }
    PHASE_STEPS.fetch_add(1, Ordering::Relaxed);
}

/// Cumulative microseconds per phase and the steps they cover.
pub fn step_phases() -> (Vec<(&'static str, u64)>, u64) {
    let phases = PHASES
        .iter()
        .zip(&PHASE_NS)
        .map(|(&name, c)| (name, c.load(Ordering::Relaxed) / 1000))
        .collect();
    (phases, PHASE_STEPS.load(Ordering::Relaxed))
}

/// Bytes the tensor-parallel exchange sent to other machines.
#[inline]
pub fn tp_sent(bytes: usize) {
    TP_TX.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Bytes the tensor-parallel exchange received from other machines.
#[inline]
pub fn tp_received(bytes: usize) {
    TP_RX.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Marks a network exchange in use (a communicator over TCP or RDMA was set up).
pub fn tp_network_used() {
    TP_USED.store(true, Ordering::Relaxed);
}

/// `(sent, received)` bytes of the network exchange, `None` when there is none.
pub fn tp_bytes() -> Option<(u64, u64)> {
    let (tx, rx) = (TP_TX.load(Ordering::Relaxed), TP_RX.load(Ordering::Relaxed));
    (TP_USED.load(Ordering::Relaxed) || tx > 0 || rx > 0).then_some((tx, rx))
}
