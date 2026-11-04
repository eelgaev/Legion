//! Helper threads leave the host thread's core (#250). The engine thread (the host thread: it
//! schedules, prepares every step and samples) runs on its GPU's node off CPU 0 when the load
//! places the stages' threads ([`super::numa::helper_cpus`]); the threads it starts (the stages'
//! launches, the n-gram and captured-row gathers, the expert cache's refill) would otherwise
//! inherit its whole CPU set, including the core it is on. A Linux thread starts with its
//! creator's affinity, so the engine works out [`off_host_core`] (the set without its current
//! core, SMT siblings included: four CPUs on a POWER9) just before it starts them, and each
//! pins itself there first ([`pin_helper`]).
//!
//! Only a thread marked as a placed host thread ([`mark_host_thread`]) does this: anywhere the
//! load placed nothing (one node, `--numa off` without an expert arena, the CPU), threads are
//! started as before.
//!
//! Adapted from Strata-AC922 (eelgaev/Strata-AC922, commits 4f00ead and 46ccb94, MIT): its
//! helper, stage, PLE-gather and cache threads leave the host thread's core.

use std::cell::Cell;
use std::fs;
use std::sync::OnceLock;

use nix::sched::sched_getcpu;

use super::numa::{copy_worker_cpus, current_affinity, parse_cpulist, pin_current_thread};

thread_local! {
    static HOST: Cell<bool> = const { Cell::new(false) };
}

/// Marks the calling thread as a placed host thread: the helpers it starts leave its core.
pub fn mark_host_thread() {
    HOST.with(|h| h.set(true));
}

/// Every CPU's SMT siblings (its core's CPUs), read from sysfs once; empty when unknown.
pub(crate) fn siblings(cpu: u32) -> &'static [u32] {
    static ALL: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    let all = ALL.get_or_init(|| {
        let n = fs::read_to_string("/sys/devices/system/cpu/possible")
            .map(|s| parse_cpulist(&s))
            .ok()
            .and_then(|c| c.last().copied())
            .map_or(0, |last| last as usize + 1);
        (0..n)
            .map(|c| {
                fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
                ))
                .map(|s| parse_cpulist(&s))
                .unwrap_or_default()
            })
            .collect()
    });
    all.get(cpu as usize).map_or(&[], Vec::as_slice)
}

/// Called on the host thread just before it starts helpers: the CPUs they should run on,
/// `cpus` (their own set; empty: the host's affinity) without the core the host is on now.
/// `cpus` as given when the caller is not a marked host thread or when that would leave no
/// CPU (empty: the helpers keep what they inherit).
pub fn off_host_core(cpus: &[u32]) -> Vec<u32> {
    if !HOST.with(Cell::get) {
        return cpus.to_vec();
    }
    let Ok(host) = sched_getcpu() else {
        return cpus.to_vec();
    };
    let host = host as u32;
    let own;
    let cpus = if cpus.is_empty() {
        own = current_affinity().unwrap_or_default();
        own.as_slice()
    } else {
        cpus
    };
    let off = copy_worker_cpus(cpus, host, siblings(host));
    if off.is_empty() { cpus.to_vec() } else { off }
}

/// Called first by a helper with the set [`off_host_core`] gave its creator (empty: left alone).
pub fn pin_helper(cpus: &[u32]) {
    if !cpus.is_empty() {
        // Best effort: a failed pin leaves the inherited affinity.
        let _ = pin_current_thread(cpus);
    }
}

#[cfg(test)]
mod tests;
