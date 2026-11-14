//! How long the running GPU waits for its peer's copies (#519), and the peer's share picked
//! from that wait.
//!
//! One FORK op of every [`SAMPLE_EVERY`] on a device is timed with its JOIN op: three timed
//! events on the running GPU's stream per op, where the op starts (F), where its own pairs are
//! queued and it reaches the wait for the peer's event (A), and after that wait (B). B - A is
//! the join wait (the time the stream stood still for the peer), B - F the op's time up to its
//! staged pairs. The events are read only once the JOIN's last one is done, from a later timed
//! FORK op (`cudaEventQuery`, never a wait), so the counter costs a few event records a step.
//!
//! [`ShareControl`] moves the peer's share of the missed experts along [`LEVELS`] from the
//! waits: up (more for the peer) while the running GPU hardly waits, down when it waits. The
//! share decides only which GPU brings which missed expert's bytes; every pair is computed by
//! the same tile code from the same bytes, so the outputs are the same bits at every share.
//! `--expert-peer-share` fixes the share (the waits are still logged); so does
//! `LEGION_DEBUG_PEER_SHARE=fixed` (A/B runs).
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

/// The shares (take, every) the controller moves between, the peer's least first.
pub const LEVELS: [(usize, usize); 6] = [(1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1)];

/// [`LEVELS`]' index of the starting share, (1, 2): the share #252 tuned (Strata-AC922's every
/// other one), [`super::SHARE`].
pub const START: usize = 2;

/// One FORK op of this many on a device is timed.
pub const SAMPLE_EVERY: u64 = 8;

/// Timed op pairs per decision of [`ShareControl`].
pub const DECIDE_EVERY: u32 = 128;

/// Above this mean wait per timed op pair (microseconds) the share moves down...
pub const WAIT_HIGH_US: f32 = 6.0;

/// ...and below this one it moves up.
pub const WAIT_LOW_US: f32 = 2.0;

/// Decisions a share that made the running GPU wait stays barred, at first; doubled each time
/// it is barred again, up to [`BAR_MAX`].
pub const BAR_FIRST: u32 = 8;
pub const BAR_MAX: u32 = 256;

/// One timed pair of ops, microseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sample {
    /// The FORK op's wait for the peer's gate and up rows.
    pub fork_wait: f32,
    /// The JOIN op's wait for the peer's down rows.
    pub join_wait: f32,
    /// The FORK op from its start to its staged pairs.
    pub fork_op: f32,
    /// The JOIN op from its start to its staged pairs.
    pub join_op: f32,
}

impl Sample {
    pub fn wait(&self) -> f32 {
        self.fork_wait + self.join_wait
    }
}

/// Sums of samples since the last decision or report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sums {
    pub n: u32,
    pub fork_wait: f64,
    pub join_wait: f64,
    pub fork_op: f64,
    pub join_op: f64,
    /// Samples whose wait was above [`WAIT_LOW_US`].
    pub waited: u32,
}

impl Sums {
    pub fn add(&mut self, s: Sample) {
        self.n += 1;
        self.fork_wait += s.fork_wait as f64;
        self.join_wait += s.join_wait as f64;
        self.fork_op += s.fork_op as f64;
        self.join_op += s.join_op as f64;
        self.waited += u32::from(s.wait() > WAIT_LOW_US);
    }

    /// Mean of a sum per sample (0 without samples).
    pub fn mean(&self, x: f64) -> f64 {
        x / self.n.max(1) as f64
    }
}

/// The peer's share of one device, moved by the waits ([`Self::add`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ShareControl {
    level: usize,
    sums: Sums,
    /// Per level: decisions it stays barred for, and how long its next bar is.
    barred: [u32; LEVELS.len()],
    bar: [u32; LEVELS.len()],
    /// Whether the share moves at all (false: it stays at [`START`]).
    auto: bool,
}

impl ShareControl {
    pub fn new(auto: bool) -> Self {
        Self {
            level: START,
            sums: Sums::default(),
            barred: [0; LEVELS.len()],
            bar: [BAR_FIRST; LEVELS.len()],
            auto,
        }
    }

    /// The share now, (take, every).
    pub fn share(&self) -> (usize, usize) {
        LEVELS[self.level]
    }

    pub fn level(&self) -> usize {
        self.level
    }

    /// Adds a timed pair of ops; every [`DECIDE_EVERY`] of them the share may move one level.
    /// Returns whether it moved.
    pub fn add(&mut self, s: Sample) -> bool {
        self.sums.add(s);
        if self.sums.n < DECIDE_EVERY {
            return false;
        }
        let wait = self.sums.mean(self.sums.fork_wait + self.sums.join_wait) as f32;
        self.sums = Sums::default();
        for b in &mut self.barred {
            *b = b.saturating_sub(1);
        }
        if !self.auto {
            return false;
        }
        if wait > WAIT_HIGH_US && self.level > 0 {
            // This share keeps the running GPU waiting: bar it for a while, longer each time.
            let l = self.level;
            self.barred[l] = self.bar[l];
            self.bar[l] = (self.bar[l] * 2).min(BAR_MAX);
            self.level -= 1;
            return true;
        }
        if wait <= WAIT_HIGH_US {
            // A share that held without a long wait: its next bar starts short again.
            self.bar[self.level] = BAR_FIRST;
        }
        if wait < WAIT_LOW_US && self.level + 1 < LEVELS.len() && self.barred[self.level + 1] == 0 {
            self.level += 1;
            return true;
        }
        false
    }
}

/// Whether the share follows the waits: not with `--expert-peer-share` (`fixed_by_flag`), nor
/// with `LEGION_DEBUG_PEER_SHARE=fixed` (`debug`: the variable's value).
pub fn auto_share(fixed_by_flag: bool, debug: Option<&str>) -> bool {
    !fixed_by_flag && debug.is_none_or(|v| v.trim() != "fixed")
}

#[cfg(feature = "cuda")]
pub(in super::super) use cuda::{fork_start, mark, share_of};

#[cfg(feature = "cuda")]
mod cuda {
    use super::{SAMPLE_EVERY, Sample, ShareControl, Sums};
    use std::os::raw::{c_int, c_uint, c_void};
    use std::sync::Mutex;

    unsafe extern "C" {
        fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
        fn cudaEventQuery(e: *mut c_void) -> c_int;
        fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    /// Timed pairs a device has in flight at most (a sample is skipped while all are busy).
    const SETS: usize = 8;

    /// Logged every this many timed pairs of a device.
    const REPORT_EVERY: u32 = 2048;

    /// The events of one timed pair: F, A, B of the FORK op, then of the JOIN op.
    const EVENTS: usize = 6;

    #[derive(Clone, Copy)]
    struct Set {
        ev: [usize; EVENTS],
        /// Events recorded so far: 0 free, [`EVENTS`] complete and waiting to be read.
        recorded: usize,
    }

    struct Device {
        device: c_int,
        forks: u64,
        sets: Vec<Set>,
        control: ShareControl,
        report: Sums,
        moves: u32,
    }

    static DEVICES: Mutex<Vec<Device>> = Mutex::new(Vec::new());

    thread_local! {
        /// The set the last timed FORK op on this thread records into, for the rest of its
        /// events (that op's A and B, its JOIN op's F, A and B).
        static OPEN: std::cell::Cell<Option<(c_int, usize)>> = const { std::cell::Cell::new(None) };
    }

    fn device_mut(devs: &mut Vec<Device>, device: c_int) -> &mut Device {
        let i = match devs.iter().position(|d| d.device == device) {
            Some(i) => i,
            None => {
                devs.push(Device {
                    device,
                    forks: 0,
                    sets: Vec::new(),
                    control: ShareControl::new(super::super::share_is_auto()),
                    report: Sums::default(),
                    moves: 0,
                });
                devs.len() - 1
            }
        };
        &mut devs[i]
    }

    fn round(x: f64) -> f64 {
        (x * 100.0).round() / 100.0
    }

    /// Reads `d`'s complete sets whose last event is done (without waiting) and frees them.
    fn harvest(d: &mut Device) {
        for i in 0..d.sets.len() {
            let s = d.sets[i];
            if s.recorded != EVENTS {
                continue;
            }
            // SAFETY: a live event of this process.
            if unsafe { cudaEventQuery(s.ev[EVENTS - 1] as *mut c_void) } != 0 {
                // Not done yet (cudaErrorNotReady): look again later.
                // SAFETY: plain runtime call.
                unsafe { cudaGetLastError() };
                continue;
            }
            let el = |a: usize, b: usize| -> f32 {
                let mut ms = 0.0f32;
                // SAFETY: two recorded, completed timing events of the same device.
                let e = unsafe {
                    cudaEventElapsedTime(&mut ms, s.ev[a] as *mut c_void, s.ev[b] as *mut c_void)
                };
                if e != 0 {
                    // SAFETY: plain runtime call.
                    unsafe { cudaGetLastError() };
                    return 0.0;
                }
                (ms * 1000.0).max(0.0)
            };
            let sample = Sample {
                fork_wait: el(1, 2),
                join_wait: el(4, 5),
                fork_op: el(0, 2),
                join_op: el(3, 5),
            };
            d.sets[i].recorded = 0;
            d.report.add(sample);
            if d.control.add(sample) {
                d.moves += 1;
            }
            if d.report.n >= REPORT_EVERY {
                let r = d.report;
                let (take, every) = d.control.share();
                tracing::info!(
                    device = d.device,
                    timed_layers = r.n,
                    fork_wait_us = round(r.mean(r.fork_wait)),
                    join_wait_us = round(r.mean(r.join_wait)),
                    fork_op_us = round(r.mean(r.fork_op)),
                    join_op_us = round(r.mean(r.join_op)),
                    waited = round(r.waited as f64 / r.n.max(1) as f64),
                    share = format!("{take}/{every}"),
                    share_moves = d.moves,
                    "expert peer fetch: join wait (the running GPU waiting for its peer's copies)"
                );
                d.report = Sums::default();
            }
        }
    }

    /// The share (take, every) device `device`'s FORK ops use now.
    pub(in super::super) fn share_of(device: c_int) -> (usize, usize) {
        let mut devs = DEVICES.lock().unwrap();
        device_mut(&mut devs, device).control.share()
    }

    /// Records `s`'s next event on `main`; false (and the set freed) when that fails.
    ///
    /// # Safety
    /// `main` is a live stream of the set's device, the current device.
    unsafe fn record(s: &mut Set, main: *mut c_void) -> bool {
        if s.recorded >= EVENTS {
            return false;
        }
        // SAFETY: a live event of the stream's device (the caller's).
        if unsafe { cudaEventRecord(s.ev[s.recorded] as *mut c_void, main) } != 0 {
            // SAFETY: plain runtime call.
            unsafe { cudaGetLastError() };
            s.recorded = 0;
            return false;
        }
        s.recorded += 1;
        true
    }

    /// A FORK op with the peer starts on `main` (device `device`): one of [`SAMPLE_EVERY`]
    /// reads the finished samples and opens a set, recording its F event. A sample that cannot
    /// be taken is skipped; nothing here fails the op.
    ///
    /// # Safety
    /// `main` is a live stream of `device`, the current device.
    pub(in super::super) unsafe fn fork_start(device: c_int, main: *mut c_void) {
        let mut devs = DEVICES.lock().unwrap();
        // A set left open (its JOIN op never came): free it.
        if let Some((dev, i)) = OPEN.with(|o| o.take())
            && let Some(o) = devs.iter_mut().find(|x| x.device == dev)
            && o.sets[i].recorded < EVENTS
        {
            o.sets[i].recorded = 0;
        }
        let d = device_mut(&mut devs, device);
        d.forks += 1;
        if !d.forks.is_multiple_of(SAMPLE_EVERY) {
            return;
        }
        harvest(d);
        let i = match d.sets.iter().position(|s| s.recorded == 0) {
            Some(i) => i,
            None if d.sets.len() < SETS => {
                let mut ev = [0usize; EVENTS];
                for e in &mut ev {
                    let mut p = std::ptr::null_mut();
                    // SAFETY: an out-parameter; flags 0: a timing event.
                    if unsafe { cudaEventCreateWithFlags(&mut p, 0) } != 0 {
                        // SAFETY: plain runtime call.
                        unsafe { cudaGetLastError() };
                        return;
                    }
                    *e = p as usize;
                }
                d.sets.push(Set { ev, recorded: 0 });
                d.sets.len() - 1
            }
            None => return,
        };
        // SAFETY: as the caller promises.
        if unsafe { record(&mut d.sets[i], main) } {
            OPEN.with(|o| o.set(Some((device, i))));
        }
    }

    /// Records the open set's next event on `main`, if this pair of ops is timed: the FORK op's
    /// A (before its wait for the peer) and B (after it), then the JOIN op's F, A and B.
    ///
    /// # Safety
    /// `main` is a live stream of the open set's device, the current device.
    pub(in super::super) unsafe fn mark(main: *mut c_void) {
        let Some((device, i)) = OPEN.with(|o| o.get()) else {
            return;
        };
        let mut devs = DEVICES.lock().unwrap();
        let Some(d) = devs.iter_mut().find(|x| x.device == device) else {
            return;
        };
        let s = &mut d.sets[i];
        // SAFETY: as the caller promises.
        if !unsafe { record(s, main) } || s.recorded == EVENTS {
            OPEN.with(|o| o.set(None));
        }
    }
}

#[cfg(test)]
mod tests;
