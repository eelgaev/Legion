//! A single, redraw-in-place server status line and a tracing writer that protects it.

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use legion::engine::{Engine, EngineStats};

const CLEAR_LINE: &str = "\r\x1b[2K";
const REFRESH: Duration = Duration::from_millis(350);
const AVERAGE_WINDOW: Duration = Duration::from_secs(5);

#[derive(Default)]
struct LineState {
    line: Option<String>,
}

#[derive(Default)]
struct Terminal {
    state: Mutex<LineState>,
}

fn terminal() -> Arc<Terminal> {
    static TERMINAL: OnceLock<Arc<Terminal>> = OnceLock::new();
    TERMINAL
        .get_or_init(|| Arc::new(Terminal::default()))
        .clone()
}

/// A tracing-event writer. Each complete line it is given goes out in one critical section
/// under the live line's lock: the line cleared, the log text, the line redrawn, as a single
/// write, so neither another event nor a status refresh can land in between. A trailing
/// partial line waits for its newline or for the writer's drop (the end of the event).
pub(super) struct StatusWriter<W: Write> {
    inner: W,
    terminal: Arc<Terminal>,
    pending: Vec<u8>,
}

impl<W: Write> StatusWriter<W> {
    fn new(inner: W, terminal: Arc<Terminal>) -> Self {
        Self {
            inner,
            terminal,
            pending: Vec::new(),
        }
    }

    /// Writes `text` around the live line, holding its lock throughout.
    fn emit(&mut self, text: &[u8]) -> io::Result<()> {
        // The dashboard owns the terminal: the line is in its log ring.
        if text.is_empty() || legion::dashboard::logs::muted() {
            return Ok(());
        }
        let state = self
            .terminal
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match &state.line {
            Some(line) => {
                let mut out = Vec::with_capacity(text.len() + line.len() + 2 * CLEAR_LINE.len());
                out.extend_from_slice(CLEAR_LINE.as_bytes());
                out.extend_from_slice(text);
                out.extend_from_slice(CLEAR_LINE.as_bytes());
                out.extend_from_slice(line.as_bytes());
                self.inner.write_all(&out)?;
            }
            None => self.inner.write_all(text)?,
        }
        self.inner.flush()
    }
}

impl<W: Write> Write for StatusWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        if let Some(end) = self.pending.iter().rposition(|&b| b == b'\n') {
            let rest = self.pending.split_off(end + 1);
            let lines = std::mem::replace(&mut self.pending, rest);
            self.emit(&lines)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: Write> Drop for StatusWriter<W> {
    fn drop(&mut self) {
        let rest = std::mem::take(&mut self.pending);
        let _ = self.emit(&rest);
    }
}

pub(super) fn stdout_writer() -> StatusWriter<io::Stdout> {
    StatusWriter::new(io::stdout(), terminal())
}

pub(super) fn stdout_is_terminal() -> bool {
    io::stdout().is_terminal()
}

/// Writes a startup report without racing a live line.
pub(super) fn write_report(report: &str) -> io::Result<()> {
    let terminal = terminal();
    let state = terminal.state.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = io::stdout().lock();
    if state.line.is_some() {
        out.write_all(CLEAR_LINE.as_bytes())?;
    }
    out.write_all(report.as_bytes())?;
    if let Some(line) = &state.line {
        write!(out, "{CLEAR_LINE}{line}")?;
    }
    out.flush()
}

/// Owns the refresh task and clears the terminal line when the server exits.
pub(super) struct StatusGuard {
    task: tokio::task::JoinHandle<()>,
    live: bool,
}

impl Drop for StatusGuard {
    fn drop(&mut self) {
        self.task.abort();
        if self.live {
            set_line(None);
        }
    }
}

/// Starts either the interactive line, or periodic structured stats for non-interactive output.
pub(super) fn start(
    engine: Arc<dyn Engine>,
    live: bool,
    stats_interval: Option<Duration>,
) -> Option<StatusGuard> {
    if !live && stats_interval.is_none() {
        return None;
    }
    let task = tokio::spawn(async move {
        if live {
            run_live(engine).await;
        } else if let Some(interval) = stats_interval {
            run_periodic(engine, interval).await;
        }
    });
    Some(StatusGuard { task, live })
}

async fn run_periodic(engine: Arc<dyn Engine>, interval: Duration) {
    let mut timer = tokio::time::interval(interval);
    timer.tick().await;
    loop {
        timer.tick().await;
        let s = engine.stats();
        // Only when requests of other priorities or several fair-share keys wait (#625).
        let queued_by_priority = s.waiting_classes.summary();
        let waiting_keys = (s.waiting_keys > 1).then_some(s.waiting_keys);
        tracing::info!(
            prefill = s.prefilling,
            decode = s.decoding,
            queued = s.waiting,
            queued_by_priority = queued_by_priority.as_deref(),
            waiting_keys,
            prompt_tokens = s.prompt_tokens,
            generated_tokens = s.generated_tokens,
            kv_used = s.kv_blocks_used,
            kv_total = s.kv_blocks_total,
            expert_cache_hit_rate = expert_hit_rate(s),
            mtp_accepted_length = s.spec.mean_accepted_length(),
            "server stats"
        );
    }
}

async fn run_live(engine: Arc<dyn Engine>) {
    let mut timer = tokio::time::interval(REFRESH);
    let mut history = VecDeque::new();
    loop {
        timer.tick().await;
        let now = Instant::now();
        let stats = engine.stats();
        let mem = device_bytes(&stats);
        history.push_back((now, stats.prompt_tokens, stats.generated_tokens, mem));
        while history
            .front()
            .is_some_and(|(at, ..)| now.duration_since(*at) > AVERAGE_WINDOW)
        {
            history.pop_front();
        }
        if stats.running == 0 && stats.waiting == 0 {
            set_line(None);
            continue;
        }
        let previous = history.iter().rev().nth(1).copied();
        let oldest = history.front().copied();
        let prompt_now = rate(now, stats.prompt_tokens, previous.map(|p| (p.0, p.1)));
        let gen_now = rate(now, stats.generated_tokens, previous.map(|p| (p.0, p.2)));
        let prompt_avg = rate(now, stats.prompt_tokens, oldest.map(|p| (p.0, p.1)));
        let gen_avg = rate(now, stats.generated_tokens, oldest.map(|p| (p.0, p.2)));
        let mut line = format!(
            "prefill {} | decode {} | queued {}{}  prompt {:.1}/{:.1} tok/s  gen {:.1}/{:.1} tok/s",
            stats.prefilling,
            stats.decoding,
            stats.waiting,
            queue_text(&stats).unwrap_or_default(),
            prompt_now,
            prompt_avg,
            gen_now,
            gen_avg
        );
        if stats.running == 1 {
            line.push_str(&format!("  seq {:.1} tok/s", gen_avg));
        }
        if stats.kv_blocks_total > 0 {
            line.push_str(&format!(
                "  KV {:.0}%",
                100.0 * f64::from(stats.kv_blocks_used) / f64::from(stats.kv_blocks_total)
            ));
        }
        if let Some(hit) = expert_hit_rate(stats) {
            line.push_str(&format!("  experts {:.0}%", hit * 100.0));
        }
        if let Some(accepted) = stats.spec.mean_accepted_length() {
            line.push_str(&format!("  MTP {accepted:.2}"));
        }
        if let Some(mem) = oldest.and_then(|p| mem_text(p.3, mem, stats.roofline().memory_gb_s)) {
            line.push_str(&mem);
        }
        set_line(Some(truncate(&line, terminal_width())));
    }
}

/// " (p-1:5 p10:40, 2 keys)" after the queued count (#625): waiting requests per priority
/// class and the fair-share keys with waiting work. `None`, the line as before, while every
/// waiting request has priority 0 and at most one key waits.
fn queue_text(s: &EngineStats) -> Option<String> {
    let classes = s.waiting_classes.summary();
    let keys = (s.waiting_keys > 1).then(|| format!("{} keys", s.waiting_keys));
    let parts: Vec<String> = classes.into_iter().chain(keys).collect();
    (!parts.is_empty()).then(|| format!(" ({})", parts.join(", ")))
}

/// The engine's model bytes from device memory and the time of the steps that read them, since
/// start (#391): every kind of step together.
fn device_bytes(s: &EngineStats) -> (u64, f64) {
    let m = &s.model_bytes;
    [m.prefill, m.decode, m.draft]
        .iter()
        .fold((0, 0.0), |(b, t), p| (b + p.device(), t + p.seconds))
}

/// "  mem 187 GB/s (80%)": device-memory model bytes over the steps' time between two
/// [`device_bytes`] snapshots, and their share of the roofline (`memory_gb_s`, 0: unknown).
/// `None` when no step ran in between.
fn mem_text(old: (u64, f64), now: (u64, f64), memory_gb_s: f64) -> Option<String> {
    let seconds = now.1 - old.1;
    if seconds <= 0.0 {
        return None;
    }
    let gb_s = now.0.saturating_sub(old.0) as f64 / seconds / 1e9;
    Some(if memory_gb_s > 0.0 {
        format!("  mem {gb_s:.0} GB/s ({:.0}%)", gb_s / memory_gb_s * 100.0)
    } else {
        format!("  mem {gb_s:.0} GB/s")
    })
}

fn expert_hit_rate(s: EngineStats) -> Option<f64> {
    let total = s.expert_cache_hits + s.expert_cache_misses;
    (total > 0).then(|| s.expert_cache_hits as f64 / total as f64)
}

fn rate(now: Instant, value: u64, before: Option<(Instant, u64)>) -> f64 {
    let Some((at, old)) = before else { return 0.0 };
    let seconds = now.duration_since(at).as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        value.saturating_sub(old) as f64 / seconds
    }
}

fn set_line(line: Option<String>) {
    let terminal = terminal();
    let mut state = terminal.state.lock().unwrap_or_else(|e| e.into_inner());
    if state.line == line {
        return;
    }
    state.line = line;
    let mut out = io::stdout().lock();
    let _ = out.write_all(CLEAR_LINE.as_bytes());
    if let Some(line) = &state.line {
        let _ = out.write_all(line.as_bytes());
    }
    let _ = out.flush();
}

fn terminal_width() -> usize {
    let mut size = std::mem::MaybeUninit::<nix::libc::winsize>::zeroed();
    // SAFETY: `size` points to writable storage for the `TIOCGWINSZ` result, and stdout is a
    // process-owned descriptor. A failed ioctl simply falls through to COLUMNS/the default.
    if unsafe {
        nix::libc::ioctl(
            nix::libc::STDOUT_FILENO,
            nix::libc::TIOCGWINSZ,
            size.as_mut_ptr(),
        )
    } == 0
    {
        // SAFETY: a successful ioctl initialized the winsize structure.
        let columns = unsafe { size.assume_init() }.ws_col as usize;
        if columns > 0 {
            return columns;
        }
    }
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(80)
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }
    let mut out: String = s.chars().take(width - 1).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests;
