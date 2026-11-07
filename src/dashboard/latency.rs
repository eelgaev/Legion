//! Latency percentiles over the last five minutes without a lock or a list: one histogram per
//! minute (the last [`MINUTES`] kept), log-spaced buckets of relaxed atomic counters. Recording
//! is a bucket index and one atomic add.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Buckets per doubling.
const PER_OCTAVE: f64 = 4.0;
/// Bucket 0 ends here (seconds); bucket `i` ends at `FIRST * 2^((i + 1) / 4)`.
const FIRST: f64 = 1e-4;
/// 0.1 ms .. ~105 s.
pub const BUCKETS: usize = 80;
/// Minutes kept (five reported plus the one being filled out).
const MINUTES: usize = 6;
/// Minutes a percentile covers.
pub const WINDOW_MINUTES: u64 = 5;

struct Minute {
    /// The minute (since the window's start) this slot holds, `u64::MAX` before any.
    at: AtomicU64,
    counts: [AtomicU32; BUCKETS],
}

pub struct LatencyWindow {
    start: Instant,
    minutes: [Minute; MINUTES],
}

impl Default for LatencyWindow {
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

/// The bucket `seconds` falls in.
pub fn bucket(seconds: f64) -> usize {
    if seconds.is_nan() || seconds <= FIRST {
        return 0;
    }
    let i = ((seconds / FIRST).log2() * PER_OCTAVE).floor() as i64;
    i.clamp(0, BUCKETS as i64 - 1) as usize
}

/// A bucket's upper bound, in seconds.
pub fn bucket_end(i: usize) -> f64 {
    FIRST * 2f64.powf((i + 1) as f64 / PER_OCTAVE)
}

impl LatencyWindow {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            minutes: std::array::from_fn(|_| Minute {
                at: AtomicU64::new(u64::MAX),
                counts: std::array::from_fn(|_| AtomicU32::new(0)),
            }),
        }
    }

    fn minute(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.start).as_secs() / 60
    }

    /// Counts one latency of `seconds`, observed at `now`.
    pub fn record(&self, now: Instant, seconds: f64) {
        let m = self.minute(now);
        let slot = &self.minutes[(m % MINUTES as u64) as usize];
        if slot.at.load(Ordering::Relaxed) != m && slot.at.swap(m, Ordering::Relaxed) != m {
            // A new minute takes over the slot: what it held is six minutes old. Racing
            // recorders may lose a count or two; a dashboard does not mind.
            for c in &slot.counts {
                c.store(0, Ordering::Relaxed);
            }
        }
        slot.counts[bucket(seconds)].fetch_add(1, Ordering::Relaxed);
    }

    /// The counts per bucket over the last [`WINDOW_MINUTES`] minutes at `now`.
    pub fn counts(&self, now: Instant) -> [u64; BUCKETS] {
        let m = self.minute(now);
        let mut out = [0u64; BUCKETS];
        for slot in &self.minutes {
            let at = slot.at.load(Ordering::Relaxed);
            if at == u64::MAX || at > m || m - at >= WINDOW_MINUTES {
                continue;
            }
            for (o, c) in out.iter_mut().zip(&slot.counts) {
                *o += u64::from(c.load(Ordering::Relaxed));
            }
        }
        out
    }

    /// `(p50, p95, samples)` in milliseconds over the last five minutes; `None`s without
    /// samples.
    pub fn percentiles(&self, now: Instant) -> (Option<f64>, Option<f64>, u64) {
        let counts = self.counts(now);
        let n: u64 = counts.iter().sum();
        (quantile(&counts, n, 0.50), quantile(&counts, n, 0.95), n)
    }

    /// The window's span, for labels.
    pub fn span() -> Duration {
        Duration::from_secs(WINDOW_MINUTES * 60)
    }
}

/// The `q` quantile's bucket end in milliseconds (an upper bound within one bucket, 19%).
pub fn quantile(counts: &[u64; BUCKETS], n: u64, q: f64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let want = ((n as f64 * q).ceil() as u64).max(1);
    let mut seen = 0;
    for (i, &c) in counts.iter().enumerate() {
        seen += c;
        if seen >= want {
            return Some(bucket_end(i) * 1e3);
        }
    }
    Some(bucket_end(BUCKETS - 1) * 1e3)
}
