//! `--tp-profile N`: where a rank's steps went, logged every `N` steps. A step's wall time
//! splits into the all-reduces' stalls (this rank's partial ready, a peer's not there yet:
//! the network and the other ranks' lag), the step's send to the other ranks (rank 0), and the
//! rest, which is this rank's own work (compute, launches, the step's output).

use std::time::Duration;

use super::{ArStats, Comm};

#[cfg(test)]
mod tests;

/// One window of steps, as logged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub steps: u64,
    pub step_ms: f64,
    pub send_ms: f64,
    pub all_reduces: f64,
    /// Per step: the all-reduces' stalls.
    pub stall_ms: f64,
    /// Per step: the rest of the step (`step - stall - send`).
    pub own_ms: f64,
    /// The mean signed wait per all-reduce (µs; see [`ArStats`]): over two ranks the two
    /// ranks' means add up to twice the one-way transport time.
    pub wait_us: f64,
    /// The mean stall of a step's first all-reduce (µs), which also waits for the ranks that
    /// started the step later.
    pub first_stall_us: f64,
}

impl Window {
    /// The window between the stats `a` and `b` over `steps` steps of `step` and `send` in all.
    pub fn new(a: ArStats, b: ArStats, steps: u64, step: Duration, send: Duration) -> Self {
        let per = |x: f64| if steps == 0 { 0.0 } else { x / steps as f64 };
        let count = b.count - a.count;
        let firsts = b.firsts - a.firsts;
        let step_ms = per(step.as_secs_f64() * 1e3);
        let send_ms = per(send.as_secs_f64() * 1e3);
        let stall_ms = per((b.stall_ns - a.stall_ns) as f64 / 1e6);
        Self {
            steps,
            step_ms,
            send_ms,
            all_reduces: per(count as f64),
            stall_ms,
            own_ms: (step_ms - stall_ms - send_ms).max(0.0),
            wait_us: if count == 0 {
                0.0
            } else {
                (b.wait_ns - a.wait_ns) as f64 / 1e3 / count as f64
            },
            first_stall_us: if firsts == 0 {
                0.0
            } else {
                (b.first_stall_ns - a.first_stall_ns) as f64 / 1e3 / firsts as f64
            },
        }
    }
}

/// A rank's step timing, logged every `every` steps (0: never).
pub struct StepProfile {
    every: usize,
    steps: u64,
    step: Duration,
    send: Duration,
    last: ArStats,
}

impl StepProfile {
    pub fn new(every: usize) -> Self {
        Self {
            every,
            steps: 0,
            step: Duration::ZERO,
            send: Duration::ZERO,
            last: ArStats::default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.every > 0
    }

    /// Records a step of `step` (wall time, `send` of it sending the step to the other ranks)
    /// and logs the window every `every` steps.
    pub fn record(&mut self, comm: &Comm, step: Duration, send: Duration) {
        if !self.enabled() {
            return;
        }
        self.steps += 1;
        self.step += step;
        self.send += send;
        if self.steps < self.every as u64 {
            return;
        }
        let now = comm.stats();
        let w = Window::new(self.last, now, self.steps, self.step, self.send);
        tracing::info!(
            rank = comm.rank(),
            steps = w.steps,
            step_ms = format!("{:.2}", w.step_ms),
            own_ms = format!("{:.2}", w.own_ms),
            stall_ms = format!("{:.2}", w.stall_ms),
            send_ms = format!("{:.3}", w.send_ms),
            all_reduces = format!("{:.1}", w.all_reduces),
            wait_us = format!("{:.1}", w.wait_us),
            first_stall_us = format!("{:.1}", w.first_stall_us),
            "tensor parallel: step profile"
        );
        *self = Self {
            last: now,
            ..Self::new(self.every)
        };
    }
}
