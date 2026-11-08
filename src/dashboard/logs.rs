//! The last [`CAPACITY`] log lines in memory, for the dashboard's log widget and
//! `GET /v1/logs/stream`: a `tracing` layer ([`RingLayer`]) next to the normal output, and a
//! switch ([`set_muted`]) that keeps the normal output off the terminal while `serve --tui` draws
//! on it.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// Lines kept.
pub const CAPACITY: usize = 2000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogLine {
    /// Increases by one per line since the start; a client asks for lines after the last it has.
    pub seq: u64,
    /// Unix milliseconds.
    pub time_ms: u64,
    /// `ERROR`, `WARN`, `INFO`, `DEBUG` or `TRACE`.
    pub level: String,
    pub target: String,
    /// The message, then the event's other fields as `key=value`.
    pub message: String,
}

/// A level's rank: 1 = ERROR ... 5 = TRACE (the log widget's keys `1`-`5`).
pub fn level_rank(level: &str) -> u8 {
    match level.to_ascii_uppercase().as_str() {
        "ERROR" => 1,
        "WARN" | "WARNING" => 2,
        "INFO" => 3,
        "DEBUG" => 4,
        _ => 5,
    }
}

#[derive(Default)]
pub struct LogRing {
    lines: Mutex<(u64, VecDeque<LogLine>)>,
}

impl LogRing {
    pub fn push(&self, mut line: LogLine) {
        let mut g = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += 1;
        line.seq = g.0;
        if g.1.len() == CAPACITY {
            g.1.pop_front();
        }
        g.1.push_back(line);
    }

    /// Lines after `seq` (all kept lines for 0), at most `max_rank` ([`level_rank`]).
    pub fn since(&self, seq: u64, max_rank: u8) -> Vec<LogLine> {
        let g = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        g.1.iter()
            .filter(|l| l.seq > seq && level_rank(&l.level) <= max_rank)
            .cloned()
            .collect()
    }

    /// The last line's sequence number (0 before any).
    pub fn last_seq(&self) -> u64 {
        self.lines.lock().map(|g| g.0).unwrap_or(0)
    }
}

/// The process's ring.
pub fn ring() -> &'static LogRing {
    static RING: OnceLock<LogRing> = OnceLock::new();
    RING.get_or_init(LogRing::default)
}

static MUTED: AtomicBool = AtomicBool::new(false);

/// Whether the normal log output is held back (the dashboard owns the terminal).
pub fn muted() -> bool {
    MUTED.load(Ordering::Relaxed)
}

pub fn set_muted(on: bool) {
    MUTED.store(on, Ordering::Relaxed);
}

/// A `tracing` layer that copies every event (that passes the filter) into [`ring`].
pub struct RingLayer;

struct Fields {
    message: String,
    rest: String,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.rest, " {}={}", field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.rest, " {}={:?}", field.name(), value);
        }
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Whether an event goes into the ring: Legion's own lines (which never carry prompt or
/// completion text), and other crates' at INFO and above only. Debug and trace lines of the HTTP
/// and TLS stacks can print request headers and body bytes (API keys, prompts), and
/// `GET /v1/logs/stream` serves the ring over the network.
pub fn kept(target: &str, level: &Level) -> bool {
    target == "legion" || target.starts_with("legion::") || *level <= Level::INFO
}

impl<S: Subscriber> Layer<S> for RingLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        if !kept(meta.target(), meta.level()) {
            return;
        }
        let mut f = Fields {
            message: String::new(),
            rest: String::new(),
        };
        event.record(&mut f);
        let level = match *meta.level() {
            Level::ERROR => "ERROR",
            Level::WARN => "WARN",
            Level::INFO => "INFO",
            Level::DEBUG => "DEBUG",
            Level::TRACE => "TRACE",
        };
        ring().push(LogLine {
            seq: 0,
            time_ms: now_ms(),
            level: level.to_string(),
            target: meta.target().to_string(),
            message: format!("{}{}", f.message, f.rest),
        });
    }
}

/// A stdout writer that drops what it is given while [`muted`].
pub struct MutableStdout;

impl std::io::Write for MutableStdout {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if muted() {
            return Ok(buf.len());
        }
        std::io::stdout().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}
