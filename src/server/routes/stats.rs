//! The live dashboard's routes (#1198): `GET /v1/stats` (one [`Snapshot`]), `GET
//! /v1/stats/stream` (the same as server-sent events, every `interval_ms`) and `GET
//! /v1/logs/stream` (the last log lines, then new ones). Each request keeps the hardware sampler
//! running; it stops a minute after the last one ([`crate::server::stats::IDLE_STOP`]).

use std::convert::Infallible;
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::response::sse::{KeepAlive, Sse};
use futures::stream::{self, StreamExt};
use serde::Deserialize;

use crate::dashboard::logs::{self, level_rank};
use crate::dashboard::snapshot::Snapshot;
use crate::server::AppState;
use crate::server::routes::json_event;

/// The stream's default and allowed interval.
pub const DEFAULT_INTERVAL_MS: u64 = 250;
pub const MIN_INTERVAL_MS: u64 = 100;
pub const MAX_INTERVAL_MS: u64 = 5000;
/// How often the log stream looks for new lines.
const LOG_POLL: Duration = Duration::from_millis(200);

/// A snapshot now, the sampler started if it is not running.
async fn take(state: &AppState) -> Snapshot {
    let hub = state.stats.clone();
    let engine = state.engine.clone();
    // Starting NVML can take a moment: off the async threads.
    tokio::task::spawn_blocking(move || {
        let hw = hub.attach();
        hub.snapshot(&engine.stats(), hw, Instant::now())
    })
    .await
    .unwrap_or_default()
}

pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    Json(take(&state).await)
}

#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    pub interval_ms: Option<u64>,
}

/// `?interval_ms=` clamped to 100-5000 (250 without).
pub fn interval(q: Option<u64>) -> Duration {
    Duration::from_millis(
        q.unwrap_or(DEFAULT_INTERVAL_MS)
            .clamp(MIN_INTERVAL_MS, MAX_INTERVAL_MS),
    )
}

pub async fn stats_stream(
    State(state): State<AppState>,
    Query(q): Query<StreamQuery>,
) -> impl IntoResponse {
    let mut tick = tokio::time::interval(interval(q.interval_ms));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let events = stream::unfold((state, tick), |(state, mut tick)| async move {
        tick.tick().await;
        let snap = take(&state).await;
        Some((json_event(&snap), (state, tick)))
    })
    .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default())
}

#[derive(Debug, Deserialize)]
pub struct LogQuery {
    /// `error`, `warn`, `info`, `debug`, `trace` or 1-5: that level and the ones above it.
    pub level: Option<String>,
}

/// The highest [`level_rank`] `?level=` lets through (all without it).
pub fn max_rank(level: Option<&str>) -> u8 {
    match level.map(str::trim) {
        None | Some("") => 5,
        Some(l) => match l.parse::<u8>() {
            Ok(n) => n.clamp(1, 5),
            Err(_) => level_rank(l),
        },
    }
}

pub async fn logs_stream(Query(q): Query<LogQuery>) -> impl IntoResponse {
    let rank = max_rank(q.level.as_deref());
    // The kept lines first, then new ones as they come.
    let events = stream::unfold((0u64, true), move |(seq, first)| async move {
        if !first {
            tokio::time::sleep(LOG_POLL).await;
        }
        loop {
            let lines = logs::ring().since(seq, rank);
            if let Some(last) = lines.last().map(|l| l.seq) {
                let batch: Vec<_> = lines.iter().map(|l| json_event(l)).collect();
                return Some((stream::iter(batch), (last, false)));
            }
            tokio::time::sleep(LOG_POLL).await;
        }
    })
    .flatten()
    .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests;
