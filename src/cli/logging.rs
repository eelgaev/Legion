//! Log setup for the CLI: `--log-format pretty|json` picks the subscriber, `RUST_LOG`
//! still picks what is logged in both modes.

use clap::ValueEnum;
use tracing_subscriber::EnvFilter;

/// Output format chosen by `--log-format` (env `LEGION_LOG_FORMAT`).
#[derive(Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum LogFormat {
    /// Human-readable text, as before.
    #[default]
    Pretty,
    /// One JSON object per line, for systemd and log collectors.
    Json,
}

/// Installs the global subscriber for `format`.
pub(crate) fn init(format: LogFormat, ansi: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    match format {
        // No module path on every line (`legion::server::middleware::request_id:`); RUST_LOG
        // still filters by it.
        LogFormat::Pretty => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_writer(super::statusline::stdout_writer)
            .with_ansi(ansi)
            .init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .with_ansi(false)
            .init(),
    }
}

impl LogFormat {
    pub(crate) fn is_json(self) -> bool {
        self == Self::Json
    }
}

/// Like [`init`], but for commands whose stdout is their output: logs go to stderr and only
/// warnings show unless `RUST_LOG` says otherwise.
pub(crate) fn init_stderr(format: LogFormat, ansi: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into());
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .with_writer(std::io::stderr);
    match format {
        LogFormat::Pretty => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
}
