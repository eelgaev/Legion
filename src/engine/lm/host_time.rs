//! The engine thread's time per step by phase (#47): what the host does between the device's
//! steps (admission, the step's plan, packing its inputs, sampling and streaming, bookkeeping)
//! next to the step itself and the other device calls. Summed over a window and logged at
//! debug level (`RUST_LOG=legion::engine::lm::host_time=debug`) as microseconds per step, a
//! few `Instant::now` calls per step otherwise.

use std::time::{Duration, Instant};

/// Where the engine thread's time goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    /// The loop around the steps: taking new requests, admission, the status counters.
    Loop,
    /// [`super::Worker::plan_step`]: drafts of model-free drafters, block tables, prompt chunks.
    Plan,
    /// The step's inputs as the executor takes them (groups, fed tokens, n-gram prefetch).
    Pack,
    /// The executor's step: its own input fill and graph launch, and the device's compute.
    Exec,
    /// Sampling and streaming: every sequence's tokens picked, detokenized, parsed and sent.
    Sample,
    /// Device calls after the step: the verify commit, model drafts, state copies.
    Device,
    /// The rest of a step's end: prompts joining the batch, finished sequences leaving.
    Finish,
}

const PHASES: usize = 7;

impl Phase {
    const ALL: [Phase; PHASES] = [
        Phase::Loop,
        Phase::Plan,
        Phase::Pack,
        Phase::Exec,
        Phase::Sample,
        Phase::Device,
        Phase::Finish,
    ];

    /// Time the device may be waiting on: everything but the step and the device calls.
    fn host(self) -> bool {
        !matches!(self, Phase::Exec | Phase::Device)
    }
}

/// How often the summary is logged.
const WINDOW: Duration = Duration::from_secs(5);

/// Per-phase sums over the current window.
pub(super) struct HostClock {
    sums: [Duration; PHASES],
    steps: u64,
    /// Decoding sequences summed over the window's steps.
    seqs: u64,
    since: Instant,
    /// Whether the summary is logged at all (debug level on): otherwise nothing is summed.
    on: bool,
}

impl Default for HostClock {
    fn default() -> Self {
        Self {
            sums: [Duration::ZERO; PHASES],
            steps: 0,
            seqs: 0,
            since: Instant::now(),
            on: tracing::enabled!(tracing::Level::DEBUG),
        }
    }
}

impl HostClock {
    /// Adds the time from `*at` to now to `phase` and moves `*at` to now.
    pub(super) fn lap(&mut self, phase: Phase, at: &mut Instant) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        self.add(phase, now.saturating_duration_since(*at));
        *at = now;
    }

    /// Adds `d` to `phase`.
    pub(super) fn add(&mut self, phase: Phase, d: Duration) {
        if self.on {
            self.sums[phase as usize] += d;
        }
    }

    /// A step of `seqs` decoding sequences ended; logs the window's summary when it is due.
    pub(super) fn step(&mut self, seqs: usize) {
        if !self.on {
            return;
        }
        self.steps += 1;
        self.seqs += seqs as u64;
        if self.since.elapsed() < WINDOW {
            return;
        }
        let s = self.summary();
        tracing::debug!(
            steps = self.steps,
            seqs_per_step = s.seqs_per_step,
            host_us = s.host_us,
            loop_us = s.us[Phase::Loop as usize],
            plan_us = s.us[Phase::Plan as usize],
            pack_us = s.us[Phase::Pack as usize],
            exec_us = s.us[Phase::Exec as usize],
            sample_us = s.us[Phase::Sample as usize],
            device_us = s.us[Phase::Device as usize],
            finish_us = s.us[Phase::Finish as usize],
            "host time per step"
        );
        *self = Self {
            on: self.on,
            ..Self::default()
        };
    }

    /// The window so far, per step.
    pub(super) fn summary(&self) -> Summary {
        let per = |d: Duration| {
            if self.steps == 0 {
                0.0
            } else {
                (d.as_secs_f64() * 1e6 / self.steps as f64 * 10.0).round() / 10.0
            }
        };
        let mut us = [0.0; PHASES];
        let mut host = Duration::ZERO;
        for p in Phase::ALL {
            us[p as usize] = per(self.sums[p as usize]);
            if p.host() {
                host += self.sums[p as usize];
            }
        }
        Summary {
            us,
            host_us: per(host),
            seqs_per_step: if self.steps == 0 {
                0.0
            } else {
                self.seqs as f64 / self.steps as f64
            },
        }
    }
}

/// [`HostClock::summary`]: microseconds per step by [`Phase`], and the host's share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Summary {
    pub(super) us: [f64; PHASES],
    pub(super) host_us: f64,
    pub(super) seqs_per_step: f64,
}

#[cfg(test)]
mod tests;
