//! The link gauges: two solid block bars per link (PCIe, NVLink to the CPU and to the peer
//! GPUs, NVLink-C2C, the TP exchange), RX over TX. Each bar fills to its direction's share of
//! the link's per-direction full scale, coloured by load on a green, yellow, orange, red
//! gradient; a dimmer "ghost" fill of the same hue runs on to the direction's 10 s peak; the
//! value and the peak sit centred inside. A title line above the pair carries the link's name,
//! full scale and Legion bytes and gets shorter on narrow panels; where the panel has no room
//! for it, the bars' rows carry a short name instead (`G0 CPU RX`). A link without a counter
//! gets one bar of Legion's own bytes and the reason.
//!
//! ratatui's `Gauge` draws one fill level, so [`Bar`] is a small widget of its own with the
//! same look: unicode eighths at the edge and a centred label that stays readable.

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::block;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use super::snapshot::LinkKind;
use super::speedo::gb;

pub const RX: Color = Color::Cyan;
pub const TX: Color = Color::Magenta;
pub const LEGION: Color = Color::Yellow;

/// Title forms, longest first (see [`LinkGauge::title`]).
pub const LEVELS: usize = 5;
/// Value forms inside a bar, longest first (see [`value_label`]).
pub const VALUE_LEVELS: usize = 6;
/// A column narrower than this gets the next link instead of another column.
pub const MIN_COL: u16 = 56;
/// Titles need at least this much width; narrower columns put short names in the bar rows.
pub const MIN_TITLED_COL: u16 = 30;
/// The ghost fill's glyph: shaded, between the solid fill and the empty background.
pub const GHOST: &str = "▓";

/// What colours the terminal shows: 24-bit, the 256-colour palette, or the 16 basic ones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorDepth {
    #[default]
    Ansi16,
    Ansi256,
    Rgb,
}

impl ColorDepth {
    /// From `COLORTERM` and `TERM` (what terminals advertise; ssh forwards `TERM`).
    pub fn detect() -> Self {
        Self::from_env(
            std::env::var("COLORTERM").ok().as_deref(),
            std::env::var("TERM").ok().as_deref(),
        )
    }

    pub fn from_env(colorterm: Option<&str>, term: Option<&str>) -> Self {
        const RGB_TERMS: [&str; 6] = [
            "xterm-kitty",
            "xterm-ghostty",
            "alacritty",
            "foot",
            "wezterm",
            "contour",
        ];
        let ct = colorterm.unwrap_or("").to_ascii_lowercase();
        let term = term.unwrap_or("").to_ascii_lowercase();
        if ct == "truecolor"
            || ct == "24bit"
            || term.ends_with("-direct")
            || RGB_TERMS.iter().any(|t| term.starts_with(t))
        {
            ColorDepth::Rgb
        } else if term.contains("256color") {
            ColorDepth::Ansi256
        } else {
            ColorDepth::Ansi16
        }
    }

    /// `(r, g, b)` in this depth: itself, the nearest of the 6x6x6 cube, or `basic`.
    fn pick(self, (r, g, b): (u8, u8, u8), basic: Color) -> Color {
        match self {
            ColorDepth::Rgb => Color::Rgb(r, g, b),
            ColorDepth::Ansi256 => Color::Indexed(cube(r, g, b)),
            ColorDepth::Ansi16 => basic,
        }
    }
}

/// The xterm 256-colour cube index nearest to `(r, g, b)` (levels 0, 95, 135, 175, 215, 255).
pub fn cube(r: u8, g: u8, b: u8) -> u8 {
    let level = |v: u8| match v {
        0..48 => 0,
        48..115 => 1,
        _ => (v - 35) / 40,
    };
    16 + 36 * level(r) + 6 * level(g) + level(b)
}

/// The load gradient's stops: green, yellow at 60%, orange at 80%, red at 100%.
const STOPS: [(f64, (u8, u8, u8)); 4] = [
    (0.0, (40, 200, 80)),
    (0.6, (235, 210, 40)),
    (0.8, (255, 140, 30)),
    (1.0, (235, 45, 45)),
];
/// The fill without a full scale.
const GREY: (u8, u8, u8) = (150, 150, 150);
/// The empty part's background in RGB.
const EMPTY: (u8, u8, u8) = (48, 52, 64);

fn mix(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let m = |x: u8, y: u8| (f64::from(x) + (f64::from(y) - f64::from(x)) * t).round() as u8;
    (m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// The gradient at `ratio` (0..=1) as RGB.
pub fn gradient(ratio: f64) -> (u8, u8, u8) {
    let r = if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.0
    };
    for w in STOPS.windows(2) {
        let ((a, ca), (b, cb)) = (w[0], w[1]);
        if r <= b {
            return mix(ca, cb, (r - a) / (b - a));
        }
    }
    STOPS[STOPS.len() - 1].1
}

/// A bar's fill colour: the gradient (16 colours: green under 60%, yellow under 85%, red
/// above); grey without a full scale.
pub fn load_color(ratio: f64, scaled: bool, depth: ColorDepth) -> Color {
    if !scaled {
        return depth.pick(GREY, Color::Gray);
    }
    let basic = if ratio < 0.6 {
        Color::Green
    } else if ratio < 0.85 {
        Color::Yellow
    } else {
        Color::Red
    };
    depth.pick(gradient(ratio), basic)
}

/// The bars' empty part: a dim background, visible on a black terminal.
pub fn empty_color(depth: ColorDepth) -> Color {
    match depth {
        ColorDepth::Rgb => Color::Rgb(EMPTY.0, EMPTY.1, EMPTY.2),
        ColorDepth::Ansi256 => Color::Indexed(237),
        ColorDepth::Ansi16 => Color::DarkGray,
    }
}

/// The value's text where it sits on the solid fill.
pub fn on_fill_color(depth: ColorDepth) -> Color {
    match depth {
        ColorDepth::Rgb => Color::Rgb(16, 16, 16),
        ColorDepth::Ansi256 => Color::Indexed(233),
        ColorDepth::Ansi16 => Color::Black,
    }
}

/// A link title's colour by the link's type.
pub fn kind_color(kind: LinkKind) -> Color {
    match kind {
        LinkKind::NvlinkCpu | LinkKind::NvlinkPeer => Color::LightGreen,
        LinkKind::Pcie => Color::LightBlue,
        LinkKind::C2c => Color::LightRed,
        LinkKind::Network => Color::White,
    }
}

/// A bar's colours at a load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarColors {
    pub fill: Color,
    /// The ghost glyph's colour: the fill's hue, dimmer (16 colours: the same colour, dim).
    pub ghost: Color,
    pub ghost_dim: bool,
    /// Behind the label and the edge cell over the ghost fill (16 colours: the empty colour).
    pub ghost_bg: Color,
    pub empty: Color,
    pub on_fill: Color,
}

impl BarColors {
    pub fn of(ratio: f64, scaled: bool, depth: ColorDepth) -> Self {
        let rgb = if scaled { gradient(ratio) } else { GREY };
        let dim = mix(EMPTY, rgb, 0.45);
        let fill = load_color(ratio, scaled, depth);
        let ghost = depth.pick(dim, fill);
        let empty = empty_color(depth);
        let basic = depth == ColorDepth::Ansi16;
        Self {
            fill,
            ghost,
            ghost_dim: basic,
            ghost_bg: if basic { empty } else { ghost },
            empty,
            on_fill: on_fill_color(depth),
        }
    }
}

/// One solid bar: filled to `ratio`, the ghost fill on to `peak`, then the empty background;
/// `label` centred on the middle row, dark on the fill and in the fill's colour elsewhere.
/// Ratios outside 0..=1 (and NaN) clamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar<'a> {
    pub ratio: f64,
    pub peak: f64,
    pub label: &'a str,
    pub colors: BarColors,
}

fn unit(r: f64) -> f64 {
    if r.is_finite() {
        r.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn eighth(n: usize) -> &'static str {
    [
        " ",
        block::ONE_EIGHTH,
        block::ONE_QUARTER,
        block::THREE_EIGHTHS,
        block::HALF,
        block::FIVE_EIGHTHS,
        block::THREE_QUARTERS,
        block::SEVEN_EIGHTHS,
    ][n % 8]
}

impl Bar<'_> {
    /// The cells: full blocks before `.0`, the edge's eighths at `.0` (when `.1` > 0), the
    /// ghost fill up to `.2`.
    pub fn cells(&self, width: u16) -> (u16, usize, u16) {
        let filled = f64::from(width) * unit(self.ratio);
        let mut full = filled.floor() as u16;
        let mut eighths = ((filled - filled.floor()) * 8.0).round() as usize;
        if eighths == 8 {
            full += 1;
            eighths = 0;
        }
        let full = full.min(width);
        let solid_end = full + u16::from(eighths > 0 && full < width);
        let ghost_end = ((f64::from(width) * unit(self.peak.max(self.ratio))).round() as u16)
            .clamp(solid_end, width);
        (full, eighths, ghost_end)
    }
}

impl Widget for Bar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let area = area.intersection(buf.area);
        if area.is_empty() {
            return;
        }
        let c = self.colors;
        let (full, eighths, ghost_end) = self.cells(area.width);
        let label: Vec<char> = self.label.chars().collect();
        let lw = (label.len() as u16).min(area.width);
        let label_col = (area.width - lw) / 2;
        let label_row = area.height / 2;
        for dy in 0..area.height {
            for i in 0..area.width {
                let cell = &mut buf[(area.x + i, area.y + dy)];
                cell.reset();
                let solid = i < full;
                let edge = i == full && eighths > 0;
                // The edge cell sits over the ghost fill only when the ghost runs past it.
                let ghost = !solid && i < ghost_end && !(edge && ghost_end == full + 1);
                if dy == label_row && i >= label_col && i < label_col + lw {
                    let (fg, bg) = if solid {
                        (c.on_fill, c.fill)
                    } else if ghost {
                        (c.fill, c.ghost_bg)
                    } else {
                        (c.fill, c.empty)
                    };
                    cell.set_char(label[usize::from(i - label_col)])
                        .set_fg(fg)
                        .set_bg(bg);
                    cell.modifier = Modifier::BOLD;
                } else if solid {
                    cell.set_symbol(block::FULL).set_fg(c.fill).set_bg(c.empty);
                } else if edge {
                    // The edge: the fill's eighths over the ghost or the empty part.
                    let bg = if ghost { c.ghost_bg } else { c.empty };
                    cell.set_symbol(eighth(eighths)).set_fg(c.fill).set_bg(bg);
                } else if ghost {
                    cell.set_symbol(GHOST).set_fg(c.ghost).set_bg(c.empty);
                    if c.ghost_dim {
                        cell.modifier = Modifier::DIM;
                    }
                } else {
                    cell.set_symbol(" ").set_fg(c.fill).set_bg(c.empty);
                }
            }
        }
    }
}

/// Which figure a bar shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Rx,
    Tx,
    /// Legion's own bytes, on a link without a counter.
    Legion,
}

impl Dir {
    pub fn tag(self) -> &'static str {
        match self {
            Dir::Rx => "RX",
            Dir::Tx => "TX",
            Dir::Legion => "L",
        }
    }

    pub fn color(self) -> Color {
        match self {
            Dir::Rx => RX,
            Dir::Tx => TX,
            Dir::Legion => LEGION,
        }
    }
}

/// One row under a link's title.
#[derive(Debug, Clone, PartialEq)]
pub enum BarRow {
    /// A direction's value and its 10 s peak.
    Bar(Dir, Option<f64>, Option<f64>),
    /// Why there is no RX and TX bar.
    Note(String),
}

/// One link's figures, in GB/s.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkGauge {
    pub kind: LinkKind,
    /// The GPU the link belongs to (`None`: the TP exchange); a GPU's links share a column.
    pub gpu: Option<u32>,
    /// `GPU0 NVLink CPU`, and `G0 CPU` for narrow panels.
    pub name: String,
    pub short: String,
    /// The link's bandwidth per direction; `None`: nothing reports it (the bars stay empty).
    pub full_scale: Option<f64>,
    /// Measured received / sent; `None`: no hardware counter.
    pub rx: Option<f64>,
    pub tx: Option<f64>,
    /// Legion's own bytes; `None`: not counted for this link.
    pub legion: Option<f64>,
    /// Each figure's peak over the last 10 s.
    pub rx_peak: Option<f64>,
    pub tx_peak: Option<f64>,
    pub legion_peak: Option<f64>,
    /// Why RX and TX are n/a (`no NVML counter`).
    pub why: Option<String>,
    /// `PCIe 4.0 x16`, `3 links NVLink 2`: the first thing a narrow title drops.
    pub detail: Option<String>,
}

/// The text inside a bar at `level`, 0 the longest: `14.5 GB/s  19%  pk 22.0`, then without
/// the peak `14.5 GB/s  19%`, `14.5 GB/s 19%`, `14.5 19%`, `14.5`, nothing. Without a full
/// scale there is no percentage.
pub fn value_label(
    v: Option<f64>,
    peak: Option<f64>,
    ratio: f64,
    scaled: bool,
    level: usize,
) -> String {
    let Some(v) = v.filter(|v| v.is_finite()) else {
        return if level + 1 < VALUE_LEVELS {
            "n/a".into()
        } else {
            String::new()
        };
    };
    let pct = (unit(ratio) * 100.0).round() as u32;
    let peak = peak.filter(|p| p.is_finite());
    let v = gb(v);
    match (level, scaled, peak) {
        (0, true, Some(p)) => format!("{v} GB/s  {pct}%  pk {}", gb(p)),
        (0, false, Some(p)) => format!("{v} GB/s  pk {}", gb(p)),
        (0 | 1, true, _) => format!("{v} GB/s  {pct}%"),
        (2, true, _) => format!("{v} GB/s {pct}%"),
        (0..=2, false, _) => format!("{v} GB/s"),
        (3, true, _) => format!("{v} {pct}%"),
        (3 | 4, _, _) => v,
        _ => String::new(),
    }
}

impl LinkGauge {
    pub fn measured(&self) -> bool {
        self.rx.is_some() || self.tx.is_some()
    }

    fn scaled(&self) -> bool {
        self.full_scale.is_some_and(|f| f > 0.0)
    }

    /// A bar's fill for `v`, 0..=1; NaN and nothing read 0.
    pub fn ratio(&self, v: Option<f64>) -> f64 {
        match (v, self.full_scale) {
            (Some(v), Some(full)) if full > 0.0 && v.is_finite() => (v / full).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    /// The rows under the title: RX and TX, or Legion's bytes and why there is no counter.
    pub fn rows(&self) -> Vec<BarRow> {
        if self.measured() {
            return vec![
                BarRow::Bar(Dir::Rx, self.rx, self.rx_peak),
                BarRow::Bar(Dir::Tx, self.tx, self.tx_peak),
            ];
        }
        let mut rows = Vec::with_capacity(2);
        if self.legion.is_some() {
            rows.push(BarRow::Bar(Dir::Legion, self.legion, self.legion_peak));
        }
        let why = self.why.as_deref().unwrap_or("no counter");
        rows.push(BarRow::Note(format!("RX/TX n/a ({why})")));
        rows
    }

    /// The title at `level`, 0 the longest and [`LEVELS`] - 1 the shortest.
    ///
    /// - 0: `GPU0 NVLink CPU  / 75.0 GB/s  Legion 12.1  3 links NVLink 2`
    /// - 1: without the link's detail.
    /// - 2: `G0 CPU / 75.0 GB/s L 12.1`; 3: `G0 CPU /75.0 L 12.1`.
    /// - 4: without the full scale (the bars show it).
    pub fn title(&self, level: usize) -> Line<'static> {
        let sep = if level <= 1 { "  " } else { " " };
        let dim = Style::default().fg(Color::DarkGray);
        let mut parts: Vec<(String, Style)> = Vec::new();
        let name = if level <= 1 { &self.name } else { &self.short };
        parts.push((
            name.clone(),
            Style::default()
                .fg(kind_color(self.kind))
                .add_modifier(Modifier::BOLD),
        ));
        match (level, self.full_scale) {
            (0..=2, Some(f)) => parts.push((format!("/ {} GB/s", gb(f)), dim)),
            (0..=2, None) => parts.push(("full scale n/a".into(), dim)),
            (3, Some(f)) => parts.push((format!("/{}", gb(f)), dim)),
            (3, None) => parts.push(("/n/a".into(), dim)),
            _ => {}
        }
        if let Some(l) = self.legion {
            let word = if level <= 1 { "Legion" } else { "L" };
            parts.push((format!("{word} {}", gb(l)), Style::default().fg(LEGION)));
        }
        if level == 0
            && let Some(d) = &self.detail
        {
            parts.push((d.clone(), dim));
        }
        let mut spans = Vec::with_capacity(parts.len() * 2);
        for (i, (text, style)) in parts.into_iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(sep));
            }
            spans.push(Span::styled(text, style));
        }
        Line::from(spans)
    }

    /// The start of a row: `RX` under a title, `G0 CPU RX` without one.
    pub fn tag(&self, row: &BarRow, titled: bool) -> Line<'static> {
        let mut spans = Vec::new();
        if !titled {
            spans.push(Span::styled(
                self.short.clone(),
                Style::default()
                    .fg(kind_color(self.kind))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if let BarRow::Bar(dir, ..) = row {
            if !titled {
                spans.push(Span::raw(" "));
            }
            spans.push(Span::styled(
                dir.tag(),
                Style::default()
                    .fg(dir.color())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        Line::from(spans)
    }

    /// The link in `area`: the title (when `look` has one), then its rows.
    pub fn render(&self, f: &mut Frame, area: Rect, look: &Look) {
        let mut y = area.y;
        if look.titled && y < area.bottom() {
            f.buffer_mut()
                .set_line(area.x, y, &self.title(look.title_level), area.width);
            y += 1;
        }
        for row in self.rows() {
            if y >= area.bottom() {
                break;
            }
            let r = Rect {
                y,
                height: 1,
                ..area
            };
            self.render_row(f.buffer_mut(), r, &row, look);
            y += 1;
        }
    }

    fn render_row(&self, buf: &mut Buffer, area: Rect, row: &BarRow, look: &Look) {
        let (x, _) = buf.set_line(area.x, area.y, &self.tag(row, look.titled), area.width);
        match row {
            BarRow::Note(why) => {
                let x = if look.titled { area.x } else { x + 1 };
                if x < area.right() {
                    buf.set_stringn(
                        x,
                        area.y,
                        why,
                        usize::from(area.right() - x),
                        Style::default().fg(Color::DarkGray),
                    );
                }
            }
            BarRow::Bar(_, v, peak) => {
                let skip = look.tag_w.saturating_add(1).min(area.width);
                let bar = Rect {
                    x: area.x + skip,
                    width: area.width - skip,
                    ..area
                };
                let ratio = self.ratio(*v);
                let label = value_label(*v, *peak, ratio, self.scaled(), look.value_level);
                Bar {
                    ratio,
                    peak: self.ratio(*peak),
                    label: &label,
                    colors: BarColors::of(ratio, self.scaled(), look.depth),
                }
                .render(bar, buf);
            }
        }
    }
}

/// How a column of links is drawn: with titles or short names, which title and value forms,
/// and how wide the row tags are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Look {
    pub titled: bool,
    pub title_level: usize,
    pub value_level: usize,
    pub tag_w: u16,
    pub depth: ColorDepth,
}

/// The longest title form whose widest title fits `width`; the shortest when none does.
pub fn fit_title(gauges: &[LinkGauge], width: u16) -> usize {
    (0..LEVELS)
        .find(|&level| {
            gauges
                .iter()
                .all(|g| g.title(level).width() <= usize::from(width))
        })
        .unwrap_or(LEVELS - 1)
}

/// The longest value form that fits every bar of `width` cells with a cell to spare each side.
pub fn fit_value(gauges: &[LinkGauge], width: u16) -> usize {
    (0..VALUE_LEVELS)
        .find(|&level| {
            gauges.iter().all(|g| {
                g.rows().iter().all(|r| match r {
                    BarRow::Bar(_, v, peak) => {
                        let label = value_label(*v, *peak, g.ratio(*v), g.scaled(), level);
                        label.is_empty() || label.chars().count() + 2 <= usize::from(width)
                    }
                    BarRow::Note(_) => true,
                })
            })
        })
        .unwrap_or(VALUE_LEVELS - 1)
}

/// How a panel of `width` x `rows` cells holds the links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// A title line above each link's bars.
    pub titled: bool,
    pub cols: usize,
    /// Links per column.
    pub per_col: usize,
}

impl Plan {
    /// Rows per link: the title and two bars, or the two bars.
    pub fn block(self) -> u16 {
        if self.titled { 3 } else { 2 }
    }

    /// Rows the tallest column takes.
    pub fn rows(self) -> u16 {
        (self.per_col as u16).saturating_mul(self.block())
    }
}

/// Titled when every link fits with its title in columns at least [`MIN_TITLED_COL`] wide,
/// else two rows a link. As many columns as the width gives at [`MIN_COL`] each (and as the
/// height needs), with a GPU's links kept in one column where that still fits.
pub fn plan(gauges: &[LinkGauge], width: u16, rows: u16) -> Plan {
    let n = gauges.len().max(1);
    let group = match gauges.first().and_then(|g| g.gpu) {
        Some(first) => gauges
            .iter()
            .take_while(|g| g.gpu == Some(first))
            .count()
            .max(1),
        None => 1,
    };
    let by_w = usize::from(width / MIN_COL).max(1);
    let layout = |block: u16| {
        let by_h = usize::from(rows / block);
        (by_h > 0).then(|| {
            let cols = by_w.max(n.div_ceil(by_h)).min(n);
            let mut per = n.div_ceil(cols);
            let grouped = per.next_multiple_of(group).min(n);
            if grouped <= by_h {
                per = grouped;
            }
            (n.div_ceil(per), per)
        })
    };
    if let Some((cols, per_col)) = layout(3)
        && width / cols as u16 >= MIN_TITLED_COL
    {
        return Plan {
            titled: true,
            cols,
            per_col,
        };
    }
    // Under two rows: one column, cut at the bottom.
    let (cols, per_col) = layout(2).unwrap_or((1, n));
    Plan {
        titled: false,
        cols,
        per_col,
    }
}

/// `gauges` in `area`: a block per link (see [`plan`]), side-by-side columns when they do not
/// fit one, the bars of a column starting in the same cell.
pub fn render_stack(f: &mut Frame, area: Rect, gauges: &[LinkGauge], depth: ColorDepth) {
    if gauges.is_empty() || area.is_empty() {
        return;
    }
    let p = plan(gauges, area.width, area.height);
    let col_w = area.width / p.cols as u16;
    // Every column picks its forms for the narrowest one, so they look alike.
    let fit_w = if p.cols > 1 {
        col_w.saturating_sub(1)
    } else {
        area.width
    };
    for (c, chunk) in gauges.chunks(p.per_col).enumerate() {
        let x = area.x + c as u16 * col_w;
        // One blank cell between columns; the last one takes what is left.
        let width = if c + 1 < p.cols {
            col_w.saturating_sub(1)
        } else {
            area.right() - x
        };
        if width == 0 {
            continue;
        }
        let tag_w = chunk
            .iter()
            .flat_map(|g| {
                g.rows()
                    .iter()
                    .filter(|r| matches!(r, BarRow::Bar(..)))
                    .map(|r| g.tag(r, p.titled).width())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap_or(0)
            .min(usize::from(width)) as u16;
        let fit_w = fit_w.min(width);
        let look = Look {
            titled: p.titled,
            title_level: fit_title(chunk, fit_w),
            value_level: fit_value(chunk, fit_w.saturating_sub(tag_w + 1)),
            tag_w,
            depth,
        };
        for (i, g) in chunk.iter().enumerate() {
            let y = area.y + (i as u16).saturating_mul(p.block());
            if y >= area.bottom() {
                break;
            }
            let r = Rect {
                x,
                y,
                width,
                height: p.block().min(area.bottom() - y),
            };
            g.render(f, r, &look);
        }
    }
}
