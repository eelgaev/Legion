//! `legion serve --prompt-step-stats` (#836): one log line per long prompt step (1,024 tokens
//! or more), for finding where a slow prefill's time goes.
//!
//! On a DGX Spark (GB10) an 8K prefill of the same prompt takes ~4.2 s in one run and ~7 s in
//! the next. The step timings show one chunk ~2.5-3 s slower than the others, whether its graph
//! was built or kept. So the line puts the step's host phases (prepare: graph build and plan;
//! inputs; launch; wait for the GPU; outputs) next to what the host did meanwhile: the kernel's
//! memory counters (direct reclaim, compaction stalls, major faults, transparent huge pages
//! allocated on fault), the process's page faults and CPU time, and the n-gram table's rows and
//! SSD reads. Off by default; no effect on what runs.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Instant;

use crate::residency::lookup_table::{LookupTable, TableStats};

#[cfg(test)]
mod tests;

static ON: AtomicBool = AtomicBool::new(false);

/// Whether long prompt steps log their stats line (`--prompt-step-stats`; off by default).
pub fn set_logging(on: bool) {
    ON.store(on, Relaxed);
}

/// See [`set_logging`].
pub fn logging() -> bool {
    ON.load(Relaxed)
}

/// The steps that log: prompt steps of at least this many tokens.
pub const MIN_TOKENS: usize = 1024;

/// The kernel's memory counters a stalled step may show (`/proc/vmstat`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VmStat {
    /// Direct reclaim entered by an allocation (`allocstall_*`, all zones).
    pub allocstall: u64,
    /// Pages scanned by direct reclaim (`pgscan_direct`).
    pub pgscan_direct: u64,
    /// Allocations stalled on compaction (`compact_stall`).
    pub compact_stall: u64,
    /// Major page faults, system wide (`pgmajfault`).
    pub pgmajfault: u64,
    /// Transparent huge pages allocated on a fault (`thp_fault_alloc`).
    pub thp_fault_alloc: u64,
}

impl VmStat {
    /// The counters in `text`, `/proc/vmstat`'s format (`name value` lines); missing ones 0.
    pub fn parse(text: &str) -> Self {
        let mut v = Self::default();
        for line in text.lines() {
            let mut it = line.split_ascii_whitespace();
            let (Some(name), Some(n)) = (it.next(), it.next()) else {
                continue;
            };
            let Ok(n) = n.parse::<u64>() else { continue };
            match name {
                "pgscan_direct" => v.pgscan_direct += n,
                "compact_stall" => v.compact_stall += n,
                "pgmajfault" => v.pgmajfault += n,
                "thp_fault_alloc" => v.thp_fault_alloc += n,
                _ if name.starts_with("allocstall_") => v.allocstall += n,
                _ => {}
            }
        }
        v
    }

    fn read() -> Self {
        std::fs::read_to_string("/proc/vmstat")
            .map(|t| Self::parse(&t))
            .unwrap_or_default()
    }

    /// What grew from `before` to `self`.
    pub fn since(&self, before: &Self) -> Self {
        Self {
            allocstall: self.allocstall.saturating_sub(before.allocstall),
            pgscan_direct: self.pgscan_direct.saturating_sub(before.pgscan_direct),
            compact_stall: self.compact_stall.saturating_sub(before.compact_stall),
            pgmajfault: self.pgmajfault.saturating_sub(before.pgmajfault),
            thp_fault_alloc: self.thp_fault_alloc.saturating_sub(before.thp_fault_alloc),
        }
    }
}

/// The process's own counters (`getrusage`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Usage {
    minflt: u64,
    majflt: u64,
    /// User and system CPU time, all threads.
    cpu_ms: f64,
}

impl Usage {
    fn read() -> Self {
        // SAFETY: getrusage only writes the struct it is given.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
            return Self::default();
        }
        let ms = |t: libc::timeval| t.tv_sec as f64 * 1e3 + t.tv_usec as f64 / 1e3;
        Self {
            minflt: ru.ru_minflt as u64,
            majflt: ru.ru_majflt as u64,
            cpu_ms: ms(ru.ru_utime) + ms(ru.ru_stime),
        }
    }
}

/// The counters when a long prompt step starts ([`Snap::take`]), to log what grew by its end.
pub struct Snap {
    at: Instant,
    vm: VmStat,
    usage: Usage,
    ngram: Option<TableStats>,
}

impl Snap {
    pub fn take(ngram: Option<&LookupTable>) -> Self {
        Self {
            at: Instant::now(),
            vm: VmStat::read(),
            usage: Usage::read(),
            ngram: ngram.map(|t| t.stats()),
        }
    }

    /// Logs the step's line: `tokens` its prompt tokens, `n_pool` its sparse-attention pools,
    /// `built` whether it built its graph, `ahead_ms` the next chunk's graph built meanwhile
    /// (`prompt_ahead`), the phases in ms since the step started, and the counters' growth.
    #[allow(clippy::too_many_arguments)]
    pub fn log(
        &self,
        ngram: Option<&LookupTable>,
        tokens: usize,
        n_pool: usize,
        built: bool,
        prepare_ms: f64,
        inputs_ms: f64,
        launched_ms: f64,
        synced_ms: f64,
        ahead_ms: f64,
        output_ms: f64,
    ) {
        let vm = VmStat::read().since(&self.vm);
        let u = Usage::read();
        let t = ngram.map(|t| t.stats()).unwrap_or_default();
        let t0 = self.ngram.unwrap_or_default();
        tracing::info!(
            tokens,
            n_pool,
            built,
            prepare_ms,
            inputs_ms,
            launched_ms,
            synced_ms,
            ahead_ms,
            output_ms,
            total_ms = self.at.elapsed().as_secs_f64() * 1e3,
            cpu_ms = u.cpu_ms - self.usage.cpu_ms,
            minflt = u.minflt.saturating_sub(self.usage.minflt),
            majflt = u.majflt.saturating_sub(self.usage.majflt),
            allocstall = vm.allocstall,
            pgscan_direct = vm.pgscan_direct,
            compact_stall = vm.compact_stall,
            pgmajfault = vm.pgmajfault,
            thp_fault_alloc = vm.thp_fault_alloc,
            ngram_rows = t.rows.saturating_sub(t0.rows),
            ngram_hits = t.cache_hits.saturating_sub(t0.cache_hits),
            ngram_reads = t.reads.saturating_sub(t0.reads),
            ngram_mib = t.read_bytes.saturating_sub(t0.read_bytes) as f64 / (1u64 << 20) as f64,
            "prompt step stats"
        );
    }
}
