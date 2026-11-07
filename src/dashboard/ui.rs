//! The dashboard's state ([`App`]: the last snapshots, the rates between them, a minute and
//! ten minutes of history, the log lines, the keys' state) and its drawing ([`draw`]).
//!
//! Two layouts: the full one (120 x 46 and up) with a row of link gauges and memory dials and
//! the sparklines, and a compact one where Tab picks the panel next to the request table.
//! Small terminals drop the sparklines first (under 36 rows), then the step bar (under 28);
//! 80 x 24 works.

use std::collections::{HashMap, VecDeque};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Sparkline, Table, Wrap};

use super::gauge::{self, LinkGauge};
use super::logs::{LogLine, level_rank};
use super::snapshot::{LinkKind, Rates, Snapshot};
use super::speedo::{Speedo, gb};

/// Log lines kept on the client (the server's ring size).
pub const LOG_LINES: usize = super::logs::CAPACITY;
/// One-second history points kept: ten minutes.
pub const HISTORY: usize = 600;
/// A peak holds this long (seconds of server time).
pub const PEAK_HOLD_S: f64 = 10.0;

/// Where the data comes from, and whether it is coming.
#[derive(Debug, Clone, PartialEq)]
pub enum Conn {
    /// `serve --tui`: in-process.
    Local,
    Connecting,
    Connected,
    /// The last error, and the seconds until the next try.
    Disconnected {
        error: String,
        retry_s: f64,
    },
}

/// The panels Tab cycles through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Requests,
    Throughput,
    Links,
    Memory,
    Gpu,
    Spec,
    Kv,
    Moe,
    Steps,
    Latency,
    Logs,
}

pub const PANELS: [Panel; 11] = [
    Panel::Requests,
    Panel::Throughput,
    Panel::Links,
    Panel::Memory,
    Panel::Gpu,
    Panel::Spec,
    Panel::Kv,
    Panel::Moe,
    Panel::Steps,
    Panel::Latency,
    Panel::Logs,
];

impl Panel {
    pub fn title(self) -> &'static str {
        match self {
            Panel::Requests => "Requests",
            Panel::Throughput => "Throughput",
            Panel::Links => "Links",
            Panel::Memory => "Memory bandwidth",
            Panel::Gpu => "GPUs",
            Panel::Spec => "Speculative decoding",
            Panel::Kv => "KV cache",
            Panel::Moe => "MoE experts",
            Panel::Steps => "Step breakdown",
            Panel::Latency => "Latency (5 min)",
            Panel::Logs => "Logs",
        }
    }
}

/// One second of history.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point {
    pub prefill: f64,
    pub decode: f64,
    pub accepted: Option<f64>,
}

/// The counters a history point is taken from.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Anchor {
    uptime: f64,
    prompt: u64,
    generated: u64,
    accepted: u64,
    steps: u64,
}

impl Anchor {
    fn of(s: &Snapshot) -> Self {
        Self {
            uptime: s.uptime_s,
            prompt: s.engine.prompt_tokens,
            generated: s.engine.generated_tokens,
            accepted: s.engine.spec.accepted,
            steps: s.engine.spec.steps,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Peak {
    tx: f64,
    tx_at: f64,
    rx: f64,
    rx_at: f64,
}

impl Peak {
    fn update(&mut self, tx: Option<f64>, rx: Option<f64>, now: f64) {
        if let Some(v) = tx
            && (v >= self.tx || now - self.tx_at > PEAK_HOLD_S)
        {
            self.tx = v;
            self.tx_at = now;
        }
        if let Some(v) = rx
            && (v >= self.rx || now - self.rx_at > PEAK_HOLD_S)
        {
            self.rx = v;
            self.rx_at = now;
        }
    }
}

/// What a key asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    Quit,
    /// Ctrl-C.
    Interrupt,
}

/// Prefill tokens over the engine's prefill time. A long prompt lands one chunk per step (2,048
/// tokens every second or two), so tokens per snapshot interval read 0 or a spike; prompt tokens
/// and prefill seconds both move once per step, so their ratio since the last change is the rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrefillRate {
    prompt: u64,
    seconds: f64,
    pub tok_s: f64,
}

impl PrefillRate {
    /// `first`: the first snapshot from this server (no rate before it).
    fn update(&mut self, s: &Snapshot, first: bool) {
        let e = &s.engine;
        let (prompt, seconds) = (e.prompt_tokens, e.prefill.seconds);
        if first || prompt < self.prompt || seconds < self.seconds {
            *self = Self {
                prompt,
                seconds,
                tok_s: 0.0,
            };
            return;
        }
        let (dp, ds) = (prompt - self.prompt, seconds - self.seconds);
        if dp > 0 && ds > 0.0 {
            *self = Self {
                prompt,
                seconds,
                tok_s: dp as f64 / ds,
            };
        } else if e.prefilling == 0 && dp == 0 {
            // Idle: no prompt in flight. (A step that is half published waits for its other half.)
            self.tok_s = 0.0;
        }
    }
}

pub struct App {
    pub snap: Option<Snapshot>,
    pub rates: Option<Rates>,
    /// The prefill figure shown (see `PrefillRate`).
    pub prefill: PrefillRate,
    anchor: Option<Anchor>,
    pub history: VecDeque<Point>,
    peaks: HashMap<(u32, usize), Peak>,
    pub logs: VecDeque<LogLine>,
    pub conn: Conn,
    /// `serve --tui` or the URL `legion top` reads.
    pub source: String,
    pub paused: bool,
    pub focus: Panel,
    pub help: bool,
    /// The log widget shows this level and above (1 = ERROR ... 5 = TRACE).
    pub level: u8,
    pub search: String,
    pub searching: bool,
    /// `l`: the logs full screen.
    pub full_logs: bool,
    /// Lines up from the newest (0 follows the tail).
    pub scroll: usize,
}

impl App {
    pub fn new(source: impl Into<String>, conn: Conn) -> Self {
        Self {
            snap: None,
            rates: None,
            prefill: PrefillRate::default(),
            anchor: None,
            history: VecDeque::new(),
            peaks: HashMap::new(),
            logs: VecDeque::new(),
            conn,
            source: source.into(),
            paused: false,
            focus: Panel::Requests,
            help: false,
            level: 3,
            search: String::new(),
            searching: false,
            full_logs: false,
            scroll: 0,
        }
    }

    /// A new snapshot (ignored while paused).
    pub fn push_snapshot(&mut self, snap: Snapshot) {
        if self.paused {
            return;
        }
        if let Some(prev) = &self.snap
            && snap.uptime_s < prev.uptime_s
        {
            // The server restarted: its counters did too.
            self.history.clear();
            self.peaks.clear();
            self.anchor = None;
            self.snap = None;
            self.rates = None;
        }
        if let Some(prev) = &self.snap {
            let rates = Rates::between(prev, &snap);
            let now = snap.uptime_s;
            for (g, links) in snap.gpus.iter().zip(&rates.links) {
                for (i, r) in links.iter().enumerate() {
                    self.peaks
                        .entry((g.index, i))
                        .or_default()
                        .update(r.tx_gb_s, r.rx_gb_s, now);
                }
            }
            if let Some(n) = &rates.network {
                self.peaks
                    .entry((u32::MAX, 0))
                    .or_default()
                    .update(Some(n.legion_gb_s), None, now);
            }
            self.rates = Some(rates);
        }
        self.prefill.update(&snap, self.snap.is_none());
        let cur = Anchor::of(&snap);
        match self.anchor {
            None => self.anchor = Some(cur),
            Some(a) if cur.uptime - a.uptime >= 1.0 => {
                let dt = cur.uptime - a.uptime;
                let d_steps = cur.steps.saturating_sub(a.steps);
                self.history.push_back(Point {
                    prefill: cur.prompt.saturating_sub(a.prompt) as f64 / dt,
                    decode: cur.generated.saturating_sub(a.generated) as f64 / dt,
                    accepted: (d_steps > 0)
                        .then(|| cur.accepted.saturating_sub(a.accepted) as f64 / d_steps as f64),
                });
                while self.history.len() > HISTORY {
                    self.history.pop_front();
                }
                self.anchor = Some(cur);
            }
            Some(_) => {}
        }
        self.snap = Some(snap);
    }

    /// New log lines (`reset`: the source sends its whole ring again).
    pub fn push_logs(&mut self, lines: Vec<LogLine>, reset: bool) {
        if reset {
            self.logs.clear();
            self.scroll = 0;
        }
        for l in lines {
            if self.scroll > 0 && self.shown(&l) {
                // Scrolled up: keep the view where it is.
                self.scroll += 1;
            }
            self.logs.push_back(l);
            if self.logs.len() > LOG_LINES {
                self.logs.pop_front();
            }
        }
    }

    fn shown(&self, l: &LogLine) -> bool {
        if level_rank(&l.level) > self.level {
            return false;
        }
        if self.search.is_empty() {
            return true;
        }
        let q = self.search.to_lowercase();
        l.message.to_lowercase().contains(&q) || l.target.to_lowercase().contains(&q)
    }

    /// The log lines the filter and the search let through, oldest first.
    pub fn visible_logs(&self) -> Vec<&LogLine> {
        self.logs.iter().filter(|l| self.shown(l)).collect()
    }

    pub fn handle_key(&mut self, k: KeyEvent) -> Action {
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            return Action::Interrupt;
        }
        if self.searching {
            match k.code {
                KeyCode::Enter => self.searching = false,
                KeyCode::Esc => {
                    self.searching = false;
                    self.search.clear();
                }
                KeyCode::Backspace => {
                    self.search.pop();
                }
                KeyCode::Char(c) => self.search.push(c),
                _ => {}
            }
            self.scroll = 0;
            return Action::None;
        }
        match k.code {
            KeyCode::Char('q') => return Action::Quit,
            KeyCode::Char('p') => {
                self.paused = !self.paused;
                if !self.paused {
                    // Rates after a pause span it: start over from the next snapshot.
                    self.snap = None;
                    self.rates = None;
                    self.anchor = None;
                }
            }
            KeyCode::Tab => self.cycle(1),
            KeyCode::BackTab => self.cycle(PANELS.len() - 1),
            KeyCode::Char('?') => self.help = !self.help,
            KeyCode::Esc => {
                self.help = false;
                self.search.clear();
            }
            KeyCode::Char(c @ '1'..='5') => {
                self.level = c as u8 - b'0';
                self.scroll = 0;
            }
            KeyCode::Char('/') => {
                self.searching = true;
                self.search.clear();
            }
            KeyCode::Char('l') => self.full_logs = !self.full_logs,
            KeyCode::Up => self.scroll += 1,
            KeyCode::Down => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::PageUp => self.scroll += 10,
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::Home => self.scroll = self.logs.len(),
            KeyCode::End => self.scroll = 0,
            _ => {}
        }
        Action::None
    }

    fn cycle(&mut self, by: usize) {
        let i = PANELS.iter().position(|&p| p == self.focus).unwrap_or(0);
        self.focus = PANELS[(i + by) % PANELS.len()];
    }

    /// The last 60 s and the last 10 min (10-second means) of `f` over the history.
    pub fn series(&self, f: impl Fn(&Point) -> f64) -> (Vec<u64>, Vec<u64>) {
        let n = self.history.len();
        let minute = self
            .history
            .iter()
            .skip(n.saturating_sub(60))
            .map(|p| f(p).round().max(0.0) as u64)
            .collect();
        let all: Vec<f64> = self.history.iter().map(&f).collect();
        let ten = all
            .chunks(10)
            .map(|c| (c.iter().sum::<f64>() / c.len() as f64).round().max(0.0) as u64)
            .collect();
        (minute, ten)
    }

    /// Every link gauge (per GPU, then the network), in the snapshot's order.
    pub fn links(&self) -> Vec<LinkGauge> {
        let Some(s) = &self.snap else {
            return Vec::new();
        };
        let rates = self.rates.as_ref();
        let mut links = Vec::new();
        for (gi, g) in s.gpus.iter().enumerate() {
            for (i, l) in g.links.iter().enumerate() {
                let r = rates.and_then(|r| r.links.get(gi)).and_then(|v| v.get(i));
                let peak = self.peaks.get(&(g.index, i));
                let measured = l.tx_bytes.is_some() || l.rx_bytes.is_some();
                let rate = |v: Option<f64>| measured.then_some(v.unwrap_or(0.0));
                links.push(LinkGauge {
                    name: format!("GPU{} {}", g.index, l.kind.label()),
                    short: format!("GPU{} {}", g.index, short_kind(l.kind)),
                    full_scale: l.full_scale_gb_s,
                    rx: rate(r.and_then(|r| r.rx_gb_s)),
                    tx: rate(r.and_then(|r| r.tx_gb_s)),
                    // Peer copies are not in Legion's model bytes.
                    legion: (l.kind != LinkKind::NvlinkPeer)
                        .then(|| r.map_or(0.0, |r| r.legion_gb_s)),
                    peak: peak.filter(|_| measured).map(|p| p.tx.max(p.rx)),
                    why: (!measured).then(|| {
                        if l.kind == LinkKind::C2c {
                            "no C2C counter".to_string()
                        } else {
                            "no NVML counter".to_string()
                        }
                    }),
                    detail: (!l.detail.is_empty()).then(|| l.detail.clone()),
                });
            }
        }
        if let Some(n) = &s.network {
            let r = rates.and_then(|r| r.network);
            links.push(LinkGauge {
                name: LinkKind::Network.label().into(),
                short: short_kind(LinkKind::Network).into(),
                full_scale: n.full_scale_gb_s,
                rx: None,
                tx: None,
                legion: Some(r.map_or(0.0, |r| r.legion_gb_s)),
                peak: self.peaks.get(&(u32::MAX, 0)).map(|p| p.tx),
                why: Some("no NIC counter".into()),
                detail: (!n.detail.is_empty()).then(|| n.detail.clone()),
            });
        }
        links
    }

    /// Every memory dial: GPUs with GPM or unified memory.
    pub fn memory_dials(&self) -> Vec<Speedo> {
        let Some(s) = &self.snap else {
            return Vec::new();
        };
        let rates = self.rates.as_ref();
        let mut memory = Vec::new();
        for (gi, g) in s.gpus.iter().enumerate() {
            let m = &g.memory;
            let mr = rates.and_then(|r| r.memory.get(gi));
            if g.unified || m.gpm_dram_pct.is_some() {
                let gpm = m.gpm_dram_pct.is_some();
                memory.push(Speedo {
                    title: format!(
                        "GPU{} memory{}",
                        g.index,
                        if gpm { " (GPM)" } else { " (Legion bytes)" }
                    ),
                    full_scale: m.peak_gb_s,
                    measured: mr.and_then(|r| r.measured_gb_s),
                    legion: mr.map_or(0.0, |r| r.legion_gb_s),
                    note: (!gpm).then(|| match g.mem_busy_pct {
                        Some(p) => format!("busy {p:.0}%"),
                        None => "no GPM".into(),
                    }),
                });
            }
        }
        memory
    }
}

/// A link's kind in a narrow label: `CPU`, `peer`, `PCIe`, `C2C`, `TP net`.
fn short_kind(kind: LinkKind) -> &'static str {
    match kind {
        LinkKind::NvlinkCpu => "CPU",
        LinkKind::NvlinkPeer => "peer",
        k => k.label(),
    }
}

/// `1h02m`, `3m05s`, `2d04h`.
pub fn duration(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 86_400 {
        format!("{}d{:02}h", s / 86_400, s % 86_400 / 3600)
    } else if s >= 3600 {
        format!("{}h{:02}m", s / 3600, s % 3600 / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// `850 ms`, `1.24 s`; `n/a`.
pub fn ms(v: Option<f64>) -> String {
    match v {
        None => "n/a".into(),
        Some(v) if v >= 1000.0 => format!("{:.2} s", v / 1e3),
        Some(v) if v >= 10.0 => format!("{v:.0} ms"),
        Some(v) => format!("{v:.1} ms"),
    }
}

/// The draft depth: `k3`, or `k default` without `--spec-k` (the engine picks the model's
/// default after the header is made).
pub fn spec_depth(k: u32) -> String {
    if k == 0 {
        "k default".into()
    } else {
        format!("k{k}")
    }
}

fn na<T>(v: Option<T>, f: impl Fn(T) -> String) -> String {
    v.map_or_else(|| "n/a".into(), f)
}

fn pct(v: Option<f64>) -> String {
    na(v, |v| format!("{v:.0}%"))
}

fn tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 10_000 {
        format!("{}k", n / 1000)
    } else {
        n.to_string()
    }
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

fn panel_block(app: &App, panel: Panel, title: String) -> Block<'static> {
    let focused = app.focus == panel;
    Block::bordered()
        .title(Span::styled(title, bold()))
        .border_style(if focused {
            Style::default().fg(Color::White)
        } else {
            dim()
        })
}

/// Which parts the terminal's size leaves room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode {
    pub full: bool,
    pub sparklines: bool,
    pub step_bar: bool,
}

impl Mode {
    pub fn of(area: Rect) -> Self {
        let full = area.width >= 120 && area.height >= 46;
        Self {
            full,
            sparklines: full || area.height >= 36,
            step_bar: area.height >= 28,
        }
    }
}

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let mode = Mode::of(area);
    if app.full_logs {
        let [h, logs, foot] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(area);
        draw_header(f, h, app);
        draw_logs(f, logs, app);
        draw_footer(f, foot, app);
    } else if mode.full {
        draw_full(f, area, app, mode);
    } else {
        draw_compact(f, area, app, mode);
    }
    if app.help {
        draw_help(f, area);
    }
}

fn draw_full(f: &mut Frame, area: Rect, app: &App, mode: Mode) {
    let [h, row1, bandwidth, row3, steps, logs, foot] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(13),
        Constraint::Length(bandwidth_height(app)),
        Constraint::Length(9),
        Constraint::Length(if mode.step_bar { 4 } else { 0 }),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(area);
    draw_header(f, h, app);
    let [req, right] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(row1);
    draw_requests(f, req, app);
    let [tp, lat] = Layout::vertical([Constraint::Min(5), Constraint::Length(4)]).areas(right);
    draw_throughput(f, tp, app, mode.sparklines);
    draw_latency(f, lat, app);
    draw_bandwidth(f, bandwidth, app);
    let [gpu, spec, kv, moe] = Layout::horizontal([
        Constraint::Percentage(34),
        Constraint::Percentage(24),
        Constraint::Percentage(21),
        Constraint::Percentage(21),
    ])
    .areas(row3);
    draw_gpus(f, gpu, app);
    draw_spec(f, spec, app, mode.sparklines);
    draw_kv(f, kv, app);
    draw_moe(f, moe, app);
    if mode.step_bar {
        draw_steps(f, steps, app);
    }
    draw_logs(f, logs, app);
    draw_footer(f, foot, app);
}

fn draw_compact(f: &mut Frame, area: Rect, app: &App, mode: Mode) {
    let logs_h = (area.height / 5).max(3);
    let [h, summary, body, steps, logs, foot] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Min(6),
        Constraint::Length(if mode.step_bar { 3 } else { 0 }),
        Constraint::Length(logs_h),
        Constraint::Length(1),
    ])
    .areas(area);
    draw_header(f, h, app);
    draw_summary(f, summary, app);
    // The request table on the left, the panel Tab picked on the right.
    let side = match app.focus {
        Panel::Requests | Panel::Logs => Panel::Throughput,
        p => p,
    };
    if area.width >= 60 {
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(body);
        draw_requests(f, left, app);
        draw_panel(f, right, app, side, mode);
    } else {
        draw_panel(f, body, app, app.focus, mode);
    }
    if mode.step_bar {
        draw_steps(f, steps, app);
    }
    draw_logs(f, logs, app);
    draw_footer(f, foot, app);
}

/// Draws `panel` on its own in `area`.
pub fn draw_panel(f: &mut Frame, area: Rect, app: &App, panel: Panel, mode: Mode) {
    match panel {
        Panel::Requests => draw_requests(f, area, app),
        Panel::Throughput => draw_throughput(f, area, app, mode.sparklines),
        Panel::Links => draw_links(f, area, app),
        Panel::Memory => draw_memory_bars(f, area, app),
        Panel::Gpu => draw_gpus(f, area, app),
        Panel::Spec => draw_spec(f, area, app, mode.sparklines),
        Panel::Kv => draw_kv(f, area, app),
        Panel::Moe => draw_moe(f, area, app),
        Panel::Steps => draw_steps(f, area, app),
        Panel::Latency => draw_latency(f, area, app),
        Panel::Logs => draw_logs(f, area, app),
    }
}

fn conn_span(app: &App) -> Span<'static> {
    let s = match &app.conn {
        Conn::Local => Span::styled("serve --tui", dim()),
        Conn::Connecting => Span::styled(
            format!("connecting to {}", app.source),
            Style::default().fg(Color::Yellow),
        ),
        Conn::Connected => Span::styled(
            format!("live {}", app.source),
            Style::default().fg(Color::Green),
        ),
        Conn::Disconnected { error, retry_s } => Span::styled(
            format!("disconnected: {error} (retry in {retry_s:.0}s)"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
    };
    if app.paused {
        Span::styled(
            format!("{}  PAUSED", s.content),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        s
    }
}

pub fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let Some(s) = &app.snap else {
        let lines = vec![
            Line::from(vec![Span::styled("Legion ", bold()), conn_span(app)]),
            Line::styled("waiting for the first snapshot", dim()),
        ];
        f.render_widget(Paragraph::new(lines), area);
        return;
    };
    let h = &s.header;
    let layout = if h.tp > 1 {
        format!("TP{}", h.tp)
    } else if h.stages > 1 {
        format!("{} stages", h.stages)
    } else {
        "1 stage".into()
    };
    let spec = if h.spec.is_empty() || h.spec == "none" {
        "none".to_string()
    } else {
        format!("{} {}", h.spec, spec_depth(h.spec_k))
    };
    // The connection before the uptime: "disconnected" must fit at 80 columns.
    let line1 = Line::from(vec![
        Span::styled(format!("{} ", h.model), bold()),
        Span::styled(
            format!("{} ", h.quant.as_deref().unwrap_or("quant n/a")),
            Style::default().fg(Color::Cyan),
        ),
        conn_span(app),
        Span::raw(format!("  up {}", duration(s.uptime_s))),
    ]);
    let line2 = Line::styled(
        format!(
            "{} GPU {layout} [{}]  ctx {}  seqs {}  spec {spec}  {}",
            h.gpus,
            h.devices.join(", "),
            tokens(u64::from(h.ctx_size)),
            h.max_seqs,
            h.version
        ),
        dim(),
    );
    f.render_widget(Paragraph::new(vec![line1, line2]), area);
}

fn draw_summary(f: &mut Frame, area: Rect, app: &App) {
    let (Some(s), r) = (&app.snap, app.rates.as_ref()) else {
        return;
    };
    let e = &s.engine;
    let line = Line::from(vec![
        Span::styled(
            format!("decode {:.1} tok/s  ", r.map_or(0.0, |r| r.decode_tok_s)),
            Style::default().fg(Color::Green),
        ),
        Span::styled(
            format!("prefill {:.0} tok/s  ", app.prefill.tok_s),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(format!(
            "active {} queued {}  TTFT p50 {}",
            e.running,
            e.waiting,
            ms(s.latency.ttft_p50_ms)
        )),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

pub fn draw_requests(f: &mut Frame, area: Rect, app: &App) {
    let Some(s) = &app.snap else {
        f.render_widget(panel_block(app, Panel::Requests, "Requests".into()), area);
        return;
    };
    let active = s.requests.iter().filter(|r| r.state != "queued").count() as u32;
    let queued = s.requests.len() as u32 - active;
    let block = panel_block(
        app,
        Panel::Requests,
        format!("Requests ({active} active, {queued} queued)"),
    );
    let header =
        Row::new(["#", "state", "api", "prompt", "gen", "tok/s", "TTFT", "age"]).style(dim());
    let state_style = |st: &str| match st {
        "decode" => Style::default().fg(Color::Green),
        "prefill" => Style::default().fg(Color::Cyan),
        "queued" => Style::default().fg(Color::Yellow),
        _ => dim(),
    };
    let mut rows: Vec<Row> = s
        .requests
        .iter()
        .map(|r| {
            Row::new(vec![
                Cell::from(r.id.to_string()),
                Cell::from(r.state.clone()).style(state_style(&r.state)),
                Cell::from(r.endpoint.clone()),
                Cell::from(na(r.prompt_tokens, |p| tokens(u64::from(p)))),
                Cell::from(r.generated.to_string()),
                Cell::from(na(r.tok_s, |t| format!("{t:.1}"))),
                Cell::from(ms(r.ttft_ms.map(|t| t as f64))),
                Cell::from(duration(r.age_ms as f64 / 1e3)),
            ])
        })
        .collect();
    let free = s.header.max_seqs.saturating_sub(active);
    for _ in 0..free.min(area.height as u32) {
        rows.push(Row::new(["", "free", "", "", "", "", "", ""]).style(dim()));
    }
    let widths = [
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Length(11),
        Constraint::Length(6),
        Constraint::Length(5),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(6),
    ];
    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .block(block)
            .column_spacing(1),
        area,
    );
}

pub fn draw_throughput(f: &mut Frame, area: Rect, app: &App, sparklines: bool) {
    let block = panel_block(app, Panel::Throughput, "Throughput".into());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(s) = &app.snap else { return };
    let e = &s.engine;
    let r = app.rates.as_ref();
    let decode = r.map_or(0.0, |r| r.decode_tok_s);
    let prefill = app.prefill.tok_s;
    let decoding: Vec<f64> = s
        .requests
        .iter()
        .filter(|q| q.state == "decode")
        .filter_map(|q| q.tok_s)
        .collect();
    let per_decode =
        (!decoding.is_empty()).then(|| decoding.iter().sum::<f64>() / decoding.len() as f64);
    let per_prefill = (e.prefilling > 0).then(|| prefill / f64::from(e.prefilling));
    let lines = vec![
        Line::from(vec![
            Span::styled("decode  ", Style::default().fg(Color::Green)),
            Span::raw(format!(
                "{decode:>8.1} tok/s  {} per request",
                na(per_decode, |v| format!("{v:.1}"))
            )),
        ]),
        Line::from(vec![
            Span::styled("prefill ", Style::default().fg(Color::Cyan)),
            Span::raw(format!(
                "{prefill:>8.0} tok/s  {} per request",
                na(per_prefill, |v| format!("{v:.0}"))
            )),
        ]),
        Line::raw(format!(
            "active {} (prefill {}, decode {})  queued {}",
            e.running, e.prefilling, e.decoding, e.waiting
        )),
    ];
    let text_h = (lines.len() as u16).min(inner.height);
    let [text, rest] =
        Layout::vertical([Constraint::Length(text_h), Constraint::Min(0)]).areas(inner);
    f.render_widget(Paragraph::new(lines), text);
    if !sparklines || rest.height == 0 {
        return;
    }
    let (dec60, dec10) = app.series(|p| p.decode);
    let (pre60, pre10) = app.series(|p| p.prefill);
    let rows = [
        ("dec 60s", dec60, Color::Green),
        ("dec 10m", dec10, Color::Green),
        ("pre 60s", pre60, Color::Cyan),
        ("pre 10m", pre10, Color::Cyan),
    ];
    for (i, (label, data, color)) in rows.into_iter().enumerate() {
        if i as u16 >= rest.height {
            break;
        }
        let row = Rect {
            y: rest.y + i as u16,
            height: 1,
            ..rest
        };
        let [l, spark] = Layout::horizontal([Constraint::Length(8), Constraint::Min(1)]).areas(row);
        f.render_widget(Paragraph::new(Span::styled(label, dim())), l);
        let skip = data.len().saturating_sub(spark.width as usize);
        f.render_widget(
            Sparkline::default()
                .data(data.into_iter().skip(skip))
                .style(Style::default().fg(color)),
            spark,
        );
    }
}

pub fn draw_latency(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Latency, "Latency (5 min)".into());
    let Some(s) = &app.snap else {
        f.render_widget(block, area);
        return;
    };
    let l = &s.latency;
    let lines = vec![
        Line::raw(format!(
            "TTFT p50 {:>8}  p95 {:>8}  ({} req)",
            ms(l.ttft_p50_ms),
            ms(l.ttft_p95_ms),
            l.ttft_samples
        )),
        Line::raw(format!(
            "ITL  p50 {:>8}  p95 {:>8}  ({} tok)",
            ms(l.itl_p50_ms),
            ms(l.itl_p95_ms),
            l.itl_samples
        )),
    ];
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn hw_missing(app: &App) -> Option<String> {
    let s = app.snap.as_ref()?;
    if !s.hw.sampling {
        return Some(format!(
            "hardware counters n/a: {}",
            s.hw.reason.as_deref().unwrap_or("not started")
        ));
    }
    s.gpus.is_empty().then(|| "no GPU sampled".into())
}

/// Link gauges the full layout's row gives a row each; more go into columns.
pub const LINK_ROWS: u16 = 10;
/// A memory dial's size in the full layout.
pub const DIAL_W: u16 = 34;
pub const DIAL_H: u16 = 12;

/// The full layout's link and memory row: one row per link gauge, as tall as a memory dial
/// where there is one.
pub fn bandwidth_height(app: &App) -> u16 {
    if hw_missing(app).is_some() {
        return 4;
    }
    let links = (app.links().len() as u16).clamp(1, LINK_ROWS) + 2;
    if app.memory_dials().is_empty() {
        links
    } else {
        links.max(DIAL_H)
    }
}

/// The full layout's row: the link gauges, then the memory dials on the right.
pub fn draw_bandwidth(f: &mut Frame, area: Rect, app: &App) {
    let memory = if hw_missing(app).is_some() {
        Vec::new()
    } else {
        app.memory_dials()
    };
    // The dials take at most half the row.
    let fit = memory.len().min(usize::from(area.width / 2 / DIAL_W));
    let [links, dials] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(fit as u16 * DIAL_W)])
            .areas(area);
    draw_links(f, links, app);
    for (i, d) in memory.iter().take(fit).enumerate() {
        let r = Rect {
            x: dials.x + i as u16 * DIAL_W,
            width: DIAL_W,
            ..dials
        };
        d.render(f, r, app.focus == Panel::Memory);
    }
    if memory.len() > fit {
        let more = format!(" +{} more (Tab: Memory) ", memory.len() - fit);
        let r = Rect {
            x: area.x + area.width.saturating_sub(more.len() as u16 + 1),
            y: area.y + area.height.saturating_sub(1),
            width: (more.len() as u16).min(area.width),
            height: 1.min(area.height),
        };
        f.render_widget(Paragraph::new(Span::styled(more, dim())), r);
    }
}

/// The links panel: a gauge per link, one per row (both layouts).
pub fn draw_links(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Links, Panel::Links.title().into());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if let Some(why) = hw_missing(app) {
        f.render_widget(
            Paragraph::new(Line::styled(why, dim())).wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }
    gauge::render_stack(f, inner, &app.links());
}

/// The memory dials as bars, for the compact layout.
pub fn draw_memory_bars(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Memory, Panel::Memory.title().into());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if let Some(why) = hw_missing(app) {
        f.render_widget(
            Paragraph::new(Line::styled(why, dim())).wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }
    let dials = app.memory_dials();
    let mut lines = Vec::new();
    if dials.is_empty() {
        lines.push(Line::styled(
            "no gauge: needs GPM (Hopper and newer) or unified memory",
            dim(),
        ));
        if let Some(s) = &app.snap {
            for g in &s.gpus {
                lines.push(Line::raw(format!(
                    "GPU{} busy {} (time, not bandwidth)",
                    g.index,
                    pct(g.mem_busy_pct)
                )));
            }
        }
    }
    for d in dials {
        lines.push(Line::styled(d.title.clone(), bold()));
        lines.extend(d.bar_lines(inner.width as usize));
    }
    // The first GPU's dial first: what does not fit is cut at the bottom.
    f.render_widget(Paragraph::new(lines), inner);
}

fn gib(b: u64) -> String {
    format!("{:.1}", b as f64 / (1u64 << 30) as f64)
}

pub fn draw_gpus(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Gpu, "GPUs".into());
    if let Some(why) = hw_missing(app) {
        f.render_widget(
            Paragraph::new(Line::styled(why, dim()))
                .block(block)
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    let Some(s) = &app.snap else {
        f.render_widget(block, area);
        return;
    };
    let header = Row::new(["GPU", "SM", "mem", "VRAM GiB", "W", "SM MHz", "mem MHz"]).style(dim());
    let rows: Vec<Row> = s
        .gpus
        .iter()
        .map(|g| {
            let vram = match (g.vram_used, g.vram_total) {
                (Some(u), Some(t)) => format!("{}/{}", gib(u), gib(t)),
                _ if g.unified => "unified".into(),
                _ => "n/a".into(),
            };
            Row::new(vec![
                Cell::from(g.index.to_string()),
                Cell::from(pct(g.sm_busy_pct)),
                Cell::from(pct(g.mem_busy_pct)),
                Cell::from(vram),
                Cell::from(na(g.power_w, |w| format!("{w:.0}"))),
                Cell::from(na(g.sm_clock_mhz, |m| m.to_string())),
                Cell::from(na(g.mem_clock_mhz, |m| m.to_string())),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Length(4),
        Constraint::Length(10),
        Constraint::Length(4),
        Constraint::Length(6),
        Constraint::Length(7),
    ];
    f.render_widget(Table::new(rows, widths).header(header).block(block), area);
}

pub fn draw_spec(f: &mut Frame, area: Rect, app: &App, sparklines: bool) {
    let block = panel_block(app, Panel::Spec, "Speculative".into());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(s) = &app.snap else { return };
    let h = &s.header;
    let sp = &s.engine.spec;
    if h.spec.is_empty() || h.spec == "none" {
        f.render_widget(
            Paragraph::new(Line::styled("off (no --spec)", dim())),
            inner,
        );
        return;
    }
    let live = app.rates.as_ref().and_then(|r| r.accepted_per_step);
    let total = (sp.steps > 0).then(|| sp.accepted as f64 / sp.steps as f64);
    let positions: Vec<String> = sp
        .by_position
        .iter()
        .enumerate()
        .map(|(i, [d, a])| {
            if *d == 0 {
                format!("{}:-", i + 1)
            } else {
                format!("{}:{:.0}%", i + 1, *a as f64 * 100.0 / *d as f64)
            }
        })
        .collect();
    let mut lines = vec![
        Line::raw(format!(
            "{} depth {} (mean {:.1})",
            h.spec,
            spec_depth(h.spec_k),
            sp.mean_k
        )),
        Line::from(vec![
            Span::raw("accepted/step "),
            Span::styled(
                na(live, |v| format!("{v:.2}")),
                Style::default().fg(Color::Green),
            ),
            Span::raw(format!("  all {}", na(total, |v| format!("{v:.2}")))),
        ]),
        Line::raw(if positions.is_empty() {
            "by position: n/a yet".to_string()
        } else {
            format!("by position {}", positions.join(" "))
        }),
    ];
    if !sparklines || inner.height <= lines.len() as u16 {
        lines.truncate(inner.height as usize);
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
        return;
    }
    let [text, spark] =
        Layout::vertical([Constraint::Length(lines.len() as u16), Constraint::Min(1)]).areas(inner);
    f.render_widget(Paragraph::new(lines), text);
    // Accepted per step x 10, so the sparkline has steps to show.
    let (minute, _) = app.series(|p| p.accepted.unwrap_or(0.0) * 10.0);
    let skip = minute.len().saturating_sub(spark.width as usize);
    f.render_widget(
        Sparkline::default()
            .data(minute.into_iter().skip(skip))
            .style(Style::default().fg(Color::Green)),
        spark,
    );
}

pub fn draw_kv(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Kv, "KV cache".into());
    let Some(s) = &app.snap else {
        f.render_widget(block, area);
        return;
    };
    let (h, e) = (&s.header, &s.engine);
    let used = (e.kv_blocks_total > 0)
        .then(|| f64::from(e.kv_blocks_used) * 100.0 / f64::from(e.kv_blocks_total));
    let hit_rate = {
        let all = e.prompt_tokens + e.prefix_cache_hit_tokens;
        (all > 0).then(|| e.prefix_cache_hit_tokens as f64 * 100.0 / all as f64)
    };
    let lines = vec![
        Line::raw(format!(
            "VRAM {} tok  host {} tok",
            tokens(h.kv_device_tokens),
            tokens(h.kv_host_tokens)
        )),
        Line::raw(format!(
            "blocks {}/{} ({})",
            e.kv_blocks_used,
            e.kv_blocks_total,
            pct(used)
        )),
        Line::raw(format!(
            "prefix hits {} ({} tok)",
            pct(hit_rate),
            tokens(e.prefix_cache_hit_tokens)
        )),
        Line::raw(format!(
            "types {}  warm n/a",
            h.kv_types.as_deref().unwrap_or("n/a")
        )),
    ];
    f.render_widget(Paragraph::new(lines).block(block), area);
}

pub fn draw_moe(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Moe, "MoE experts".into());
    let Some(s) = &app.snap else {
        f.render_widget(block, area);
        return;
    };
    let (h, e) = (&s.header, &s.engine);
    if h.expert_mode.is_none() && e.expert_cache_slots == 0 {
        f.render_widget(
            Paragraph::new(Line::styled("dense model", dim())).block(block),
            area,
        );
        return;
    }
    let r = app.rates.as_ref();
    let all = e.expert_cache_hits + e.expert_cache_misses;
    let total = (all > 0).then(|| e.expert_cache_hits as f64 * 100.0 / all as f64);
    let lines = vec![
        Line::raw(format!(
            "experts {}",
            h.expert_mode.as_deref().unwrap_or("n/a")
        )),
        Line::raw(if e.expert_cache_slots > 0 {
            format!(
                "resident {} slots, {} layers",
                e.expert_cache_slots, e.expert_cache_layers
            )
        } else {
            "resident: all (no cache)".into()
        }),
        Line::raw(format!(
            "hit rate {} (all {})",
            pct(r.and_then(|r| r.expert_hit_rate).map(|v| v * 100.0)),
            pct(total)
        )),
        Line::raw(format!(
            "streamed {} GB/s",
            gb(r.map_or(0.0, |r| r.expert_stream_gb_s))
        )),
    ];
    f.render_widget(Paragraph::new(lines).block(block), area);
}

/// The step breakdown's parts: the engine thread's phases grouped as the bar shows them.
pub fn step_parts(step_us: &[(String, f64)]) -> Vec<(&'static str, f64, Color)> {
    let get = |n: &str| {
        step_us
            .iter()
            .find(|(name, _)| name == n)
            .map_or(0.0, |(_, us)| *us)
    };
    vec![
        ("step", get("exec"), Color::Green),
        ("draft+verify", get("device"), Color::Magenta),
        ("sampling", get("sample"), Color::Cyan),
        (
            "host gaps",
            get("loop") + get("plan") + get("pack") + get("finish"),
            Color::Yellow,
        ),
    ]
}

pub fn draw_steps(f: &mut Frame, area: Rect, app: &App) {
    let block = panel_block(app, Panel::Steps, "Step (engine thread, us/step)".into());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(r) = &app.rates else {
        return;
    };
    let parts = step_parts(&r.step_us);
    let total: f64 = parts.iter().map(|p| p.1).sum();
    if r.step_us.is_empty() || total <= 0.0 {
        f.render_widget(
            Paragraph::new(Line::styled(
                "n/a: no steps timed yet (the engine times them while a dashboard is attached)",
                dim(),
            )),
            inner,
        );
        return;
    }
    let width = inner.width as usize;
    let mut bar = Vec::new();
    let mut legend = Vec::new();
    let mut used = 0;
    for (i, (name, us, color)) in parts.iter().enumerate() {
        let w = if i + 1 == parts.len() {
            width.saturating_sub(used)
        } else {
            ((us / total) * width as f64).round() as usize
        };
        used += w;
        bar.push(Span::styled("█".repeat(w), Style::default().fg(*color)));
        legend.push(Span::styled(
            format!("{name} {us:.0}  "),
            Style::default().fg(*color),
        ));
    }
    legend.push(Span::styled(format!("= {total:.0} us"), dim()));
    let lines = if inner.height >= 2 {
        vec![Line::from(bar), Line::from(legend)]
    } else {
        vec![Line::from(legend)]
    };
    f.render_widget(Paragraph::new(lines), inner);
}

fn level_style(level: &str) -> Style {
    match level_rank(level) {
        1 => Style::default().fg(Color::Red),
        2 => Style::default().fg(Color::Yellow),
        3 => Style::default().fg(Color::Green),
        4 => Style::default().fg(Color::Blue),
        _ => dim(),
    }
}

/// `HH:MM:SS` (UTC) of Unix milliseconds.
pub fn clock(time_ms: u64) -> String {
    let s = time_ms / 1000 % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
}

const LEVELS: [&str; 5] = ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"];

pub fn draw_logs(f: &mut Frame, area: Rect, app: &App) {
    let mut title = format!(
        "Logs [{} and up]",
        LEVELS[(app.level as usize).clamp(1, 5) - 1]
    );
    if app.searching || !app.search.is_empty() {
        title.push_str(&format!(" /{}", app.search));
        if app.searching {
            title.push('_');
        }
    }
    if app.scroll > 0 {
        title.push_str(&format!(" (scrolled {})", app.scroll));
    }
    let block = panel_block(app, Panel::Logs, title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let lines = app.visible_logs();
    let h = inner.height as usize;
    let end = lines.len().saturating_sub(app.scroll.min(lines.len()));
    let start = end.saturating_sub(h);
    let text: Vec<Line> = lines[start..end]
        .iter()
        .map(|l| {
            let target = l.target.rsplit("::").next().unwrap_or(&l.target);
            Line::from(vec![
                Span::styled(format!("{} ", clock(l.time_ms)), dim()),
                Span::styled(format!("{:<5} ", l.level), level_style(&l.level)),
                Span::styled(format!("{target}: "), dim()),
                Span::raw(l.message.clone()),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(text), inner);
}

pub fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let keys = "q quit  p pause  Tab panel  ? help  1-5 level  / search  l logs";
    let line = Line::from(vec![
        Span::styled(keys, dim()),
        Span::raw("  "),
        Span::styled(format!("[{}]", app.focus.title()), bold()),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// The help screen's text: the keys, and why a value can be n/a.
pub const HELP: &[&str] = &[
    "Keys",
    "  q  quit (serve --tui: the server keeps running, plain logs)   Ctrl-C  stop",
    "  p  pause   Tab / Shift-Tab  cycle panels   ?  this help   Esc  close",
    "  1-5  log level (ERROR .. TRACE)   /  search logs   l  full-screen logs",
    "  Up/Down PgUp/PgDn Home/End  scroll the logs",
    "",
    "Link bars: the busier of RX and TX over the link's per-direction full scale, green",
    "  under 60%, yellow under 85%, red above; no counter: Legion's bytes. pk: 10 s peak.",
    "Memory dial needles: measured cyan, Legion's own bytes yellow.",
    "Legion bytes are model bytes (what the steps must read), not measured traffic.",
    "",
    "Why n/a",
    "  hardware panels: NVML is missing or sees no GPU (the reason shows in place).",
    "  PCIe TX/RX: NVML has no PCIe throughput counter on this GPU (GB10).",
    "  C2C TX/RX: no NVML C2C counter (GPM); the bar shows Legion's bytes.",
    "  memory gauge: measured = GPM DRAM bandwidth x peak; without GPM NVML only",
    "    reports the time memory was busy, not its bandwidth. GB10: no GPM, and NVML's",
    "    memory busy reads 0 under any load (mem n/a); no CPU DRAM PMU either, so the",
    "    shared memory's traffic is not counted. The dial shows Legion's bytes only.",
    "  VRAM: unified memory (GB10) has no VRAM of its own.",
    "  quant: the model's file name does not say.",
    "  warm sequences: not counted by the engine yet.",
    "  step bar: attention and the TP exchange are inside `step` (no device timers).",
    "  TP net: Legion's own frames only (no NIC counters); scale from the NIC speed.",
];

fn draw_help(f: &mut Frame, area: Rect) {
    let w = area.width.saturating_sub(4).min(92);
    let h = area.height.saturating_sub(2).min(HELP.len() as u16 + 2);
    let r = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, r);
    let lines: Vec<Line> = HELP.iter().map(|l| Line::raw(*l)).collect();
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::bordered().title(Span::styled("Help", bold())))
            .alignment(Alignment::Left),
        r,
    );
}
