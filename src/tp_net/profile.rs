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

/// A step's host laps on this rank (the executor's step of one stage and one micro-batch, such
/// as a decode or verify step; [`crate::model::core::Executor::take_step_laps`]), in order.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepLaps {
    /// From the step's start to its round: held commits, deferred cache work, shapes.
    pub before: Duration,
    /// The stage's graph found or built.
    pub prepare: Duration,
    /// Its inputs filled.
    pub fill: Duration,
    /// Its launches queued.
    pub launch: Duration,
    /// The next verify shape's graph built while this one computes (`verify_ahead`).
    pub ahead: Duration,
    /// Other host work beside the GPU, up to the stage's sync.
    pub beside: Duration,
    /// The sync: waiting for the GPU.
    pub wait: Duration,
    /// The step's output read.
    pub output: Duration,
}

impl StepLaps {
    fn add(&mut self, o: &Self) {
        self.before += o.before;
        self.prepare += o.prepare;
        self.fill += o.fill;
        self.launch += o.launch;
        self.ahead += o.ahead;
        self.beside += o.beside;
        self.wait += o.wait;
        self.output += o.output;
    }

    /// Per step of `n`, in ms, as logged: `before/prepare/fill/launch/ahead/beside/wait/output`.
    pub fn per_step(&self, n: u64) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / n.max(1) as f64;
        format!(
            "{:.2}/{:.2}/{:.2}/{:.2}/{:.2}/{:.2}/{:.2}/{:.2}",
            ms(self.before),
            ms(self.prepare),
            ms(self.fill),
            ms(self.launch),
            ms(self.ahead),
            ms(self.beside),
            ms(self.wait),
            ms(self.output)
        )
    }
}

/// The calling thread's scheduling counters (`getrusage(RUSAGE_THREAD)`): context switches it
/// was forced into and gave up, and page faults (minor, major).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThreadCounts {
    pub forced: u64,
    pub yielded: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

impl ThreadCounts {
    /// Now, for the calling thread (zeros where the counters cannot be read).
    pub fn now() -> Self {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: rusage is plain integers (all zeros is valid); getrusage fills it with the
            // calling thread's counters (RUSAGE_THREAD).
            let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
            // SAFETY: as above, a valid rusage for the call to fill.
            if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) } == 0 {
                return Self {
                    forced: ru.ru_nivcsw as u64,
                    yielded: ru.ru_nvcsw as u64,
                    minor_faults: ru.ru_minflt as u64,
                    major_faults: ru.ru_majflt as u64,
                };
            }
        }
        Self::default()
    }

    /// The counts from `a` to `self`.
    pub fn since(&self, a: &Self) -> Self {
        Self {
            forced: self.forced.saturating_sub(a.forced),
            yielded: self.yielded.saturating_sub(a.yielded),
            minor_faults: self.minor_faults.saturating_sub(a.minor_faults),
            major_faults: self.major_faults.saturating_sub(a.major_faults),
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
    /// The steps' host laps, and how many steps had them.
    laps: StepLaps,
    lapped: u64,
    /// From each step's start to its first partial ready, from its last all-reduce to its end,
    /// and how many steps had all-reduces.
    before_first: Duration,
    after_last: Duration,
    marked: u64,
    /// The stepping thread's counters at the window's start.
    counts: Option<ThreadCounts>,
}

impl StepProfile {
    pub fn new(every: usize) -> Self {
        Self {
            every,
            steps: 0,
            step: Duration::ZERO,
            send: Duration::ZERO,
            last: ArStats::default(),
            laps: StepLaps::default(),
            lapped: 0,
            before_first: Duration::ZERO,
            after_last: Duration::ZERO,
            marked: 0,
            counts: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.every > 0
    }

    /// Records a step of `step` (wall time, `send` of it sending the step to the other ranks,
    /// `laps` its host laps if the executor kept them) and logs the window every `every` steps.
    /// Called on the thread that ran the step, at its end.
    pub fn record(&mut self, comm: &Comm, step: Duration, send: Duration, laps: Option<StepLaps>) {
        if !self.enabled() {
            return;
        }
        let marks = comm.step_marks();
        let counts = ThreadCounts::now();
        let start = *self.counts.get_or_insert(counts);
        self.steps += 1;
        self.step += step;
        self.send += send;
        if let Some(l) = laps {
            self.laps.add(&l);
            self.lapped += 1;
        }
        if let (Some(a), Some(b)) = (marks.before_first, marks.after_last) {
            self.before_first += a;
            self.after_last += b;
            self.marked += 1;
        }
        if self.steps < self.every as u64 {
            return;
        }
        let now = comm.stats();
        let w = Window::new(self.last, now, self.steps, self.step, self.send);
        let per = |d: Duration, n: u64| d.as_secs_f64() * 1e3 / n.max(1) as f64;
        let c = counts.since(&start);
        let cpu = |c: Option<u32>| c.map_or(-1, |c| c as i64);
        let laps = if self.lapped > 0 {
            self.laps.per_step(self.lapped)
        } else {
            "-".to_string()
        };
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
            to_first_ms = format!("{:.2}", per(self.before_first, self.marked)),
            after_last_ms = format!("{:.2}", per(self.after_last, self.marked)),
            host_laps = %laps,
            cpu = cpu(nix::sched::sched_getcpu().ok().map(|c| c as u32)),
            tx_cpu = cpu(marks.tx_cpu),
            rx_cpu = cpu(marks.rx_cpu),
            forced_csw = c.forced,
            yielded_csw = c.yielded,
            faults = c.minor_faults + c.major_faults,
            "tensor parallel: step profile"
        );
        *self = Self {
            last: now,
            counts: Some(counts),
            ..Self::new(self.every)
        };
    }
}
