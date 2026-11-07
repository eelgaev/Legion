//! The live dashboard (#1198): `legion serve --tui` draws it in the server's terminal, `legion
//! top` in another one from `GET /v1/stats/stream` and `GET /v1/logs/stream`. Both feed the same
//! [`term::Feed`] and draw the same [`ui`].
//!
//! - [`snapshot`]: the stats JSON (schema version 1) and the rates between two snapshots.
//! - [`hub`]: the few counters the engine feeds directly, skipped without a dashboard.
//! - [`latency`]: TTFT and inter-token percentiles over five minutes, lock-free.
//! - [`logs`]: the in-memory ring of the last 2,000 log lines.
//! - [`scale`]: each gauge's full scale from what the hardware reports.
//! - [`gauge`]: the link gauges (`LineGauge`); [`speedo`]: the memory speedometer.
//! - [`ui`]: the layout and panels.
//! - [`term`]: the terminal (raw mode, restore on exit, error and panic) and the key loop.
//! - [`client`]: `legion top`'s HTTP client (SSE, reconnect with backoff).

pub mod client;
pub mod gauge;
pub mod hub;
pub mod latency;
pub mod logs;
pub mod scale;
pub mod snapshot;
pub mod speedo;
pub mod term;
pub mod ui;

#[cfg(test)]
mod tests;
