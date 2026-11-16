//! `--deterministic`'s parts one at a time, for A/B runs (#1176). Each switch, when set, decides
//! its part whatever `--deterministic` says, so a plain server can take one part alone and a
//! `--deterministic` one can drop one; unset, `--deterministic` decides as before.
//!
//! - `LEGION_DEBUG_FRESH_PLANS` (`0` / `1`): every step graph's memory planned afresh
//!   (`legion::ggml::set_fresh_plans`, #459).
//! - `LEGION_DEBUG_WAVES` (`0` / gather milliseconds): wave admission (#836) off, or on with
//!   that gather window.
//!
//! The other parts need no switch: the draft depth is fixed by `--spec-fixed` and prefix
//! caching is turned off by `--prefix-cache-states 0`, both plain flags, and the expert cache's
//! settling (#1165) only runs with GLM's pair balance or split.

use std::time::Duration;

/// Whether step graphs are planned afresh: `LEGION_DEBUG_FRESH_PLANS`'s value (`env`), else
/// `--deterministic`.
pub(super) fn fresh_plans(deterministic: bool, env: Option<&str>) -> bool {
    match env {
        Some("0") => false,
        Some("1") => true,
        _ => deterministic,
    }
}

/// The wave admission's gather window: `LEGION_DEBUG_WAVES`'s milliseconds (`env`, 0: none),
/// else `--deterministic-gather-ms` with `--deterministic`.
pub(super) fn waves(deterministic: bool, gather_ms: u64, env: Option<&str>) -> Option<Duration> {
    let ms = match env.and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(ms) => ms,
        None if deterministic => gather_ms,
        None => 0,
    };
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// A debug switch's value, if set.
pub(super) fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
mod tests;
