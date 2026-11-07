//! The link gauges: one ratatui [`LineGauge`] per link (PCIe, NVLink to the CPU and to the
//! peer GPUs, NVLink-C2C, the TP exchange), one row each. The bar fills to the busier
//! direction's share of the link's per-direction full scale, or to Legion's own bytes where the
//! link has no counter, coloured by load. The label on its left carries the figures and gets
//! shorter on narrow panels instead of wrapping.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::LineGauge;

use super::speedo::gb;

pub const TX: Color = Color::Cyan;
pub const RX: Color = Color::Magenta;
pub const LEGION: Color = Color::Yellow;

/// The bar keeps at least this many cells: the label gets shorter first.
pub const MIN_BAR: u16 = 6;
/// Label forms, longest first (see [`LinkGauge::label`]).
pub const LEVELS: usize = 6;

/// One link's figures, in GB/s.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinkGauge {
    /// `GPU0 NVLink CPU`, and `GPU0 CPU` for narrow panels.
    pub name: String,
    pub short: String,
    /// The link's bandwidth per direction; `None`: nothing reports it (the bar stays empty).
    pub full_scale: Option<f64>,
    /// Measured received / sent; `None`: no hardware counter.
    pub rx: Option<f64>,
    pub tx: Option<f64>,
    /// Legion's own bytes; `None`: not counted for this link.
    pub legion: Option<f64>,
    /// The higher of the two directions' peaks over the last seconds.
    pub peak: Option<f64>,
    /// Why RX and TX are n/a (`no NVML counter`).
    pub why: Option<String>,
    /// `PCIe 4.0 x16`, `3 links NVLink 2`: the first thing a narrow panel drops.
    pub detail: Option<String>,
}

/// Green under 60% of the full scale, yellow under 85%, red above; grey without a full scale.
pub fn load_color(ratio: f64, scaled: bool) -> Color {
    if !scaled {
        Color::DarkGray
    } else if ratio < 0.6 {
        Color::Green
    } else if ratio < 0.85 {
        Color::Yellow
    } else {
        Color::Red
    }
}

impl LinkGauge {
    fn measured(&self) -> bool {
        self.rx.is_some() || self.tx.is_some()
    }

    /// What the bar shows: the busier direction, or Legion's bytes without a counter.
    pub fn value(&self) -> Option<f64> {
        if self.measured() {
            Some(self.rx.unwrap_or(0.0).max(self.tx.unwrap_or(0.0)))
        } else {
            self.legion
        }
    }

    /// The bar's fill, 0..=1 (`LineGauge` panics outside it, so NaN reads 0).
    pub fn ratio(&self) -> f64 {
        match (self.value(), self.full_scale) {
            (Some(v), Some(full)) if full > 0.0 && v.is_finite() => (v / full).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    /// The label at `level`, 0 the longest and [`LEVELS`] - 1 the shortest.
    ///
    /// - 0: `GPU0 NVLink CPU  RX 8.00  TX 2.00  / 75.0 GB/s  pk 9.00  Legion 5.00  3 links NVLink 2`
    /// - 1: without the link's detail; 2: without the peak.
    /// - 3: `GPU0 CPU RX 8.00 TX 2.00 / 75.0 GB/s L 5.00`; 4: `... /75.0 L 5.00`.
    /// - 5: without the full scale (the bar shows it).
    ///
    /// Without a counter RX and TX read `n/a (<why>)`, and from level 4 `no counter`.
    pub fn label(&self, level: usize) -> Line<'static> {
        let sep = if level <= 2 { "  " } else { " " };
        let dim = Style::default().fg(Color::DarkGray);
        let mut parts: Vec<(String, Style)> = Vec::new();
        let name = if level <= 2 { &self.name } else { &self.short };
        parts.push((name.clone(), Style::default().add_modifier(Modifier::BOLD)));
        if self.measured() {
            let v = |x: Option<f64>| x.map_or_else(|| "n/a".into(), gb);
            parts.push((format!("RX {}", v(self.rx)), Style::default().fg(RX)));
            parts.push((format!("TX {}", v(self.tx)), Style::default().fg(TX)));
        } else if level <= 3 {
            parts.push(("RX n/a".into(), Style::default().fg(RX)));
            let why = self.why.as_deref().unwrap_or("no counter");
            parts.push((format!("TX n/a ({why})"), Style::default().fg(TX)));
        } else {
            parts.push(("no counter".into(), dim));
        }
        match (level, self.full_scale) {
            (0..=3, Some(f)) => parts.push((format!("/ {} GB/s", gb(f)), dim)),
            (0..=3, None) => parts.push(("full scale n/a".into(), dim)),
            (4, Some(f)) => parts.push((format!("/{}", gb(f)), dim)),
            (4, None) => parts.push(("/n/a".into(), dim)),
            _ => {}
        }
        if level <= 1
            && let Some(p) = self.peak
        {
            parts.push((format!("pk {}", gb(p)), dim));
        }
        if let Some(l) = self.legion {
            let word = if level <= 2 { "Legion" } else { "L" };
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

    /// The gauge in a one-row `area`, its label padded to `label_w` so the bars of a stack
    /// start in the same column.
    pub fn render(&self, f: &mut Frame, area: Rect, level: usize, label_w: u16) {
        let mut label = self.label(level);
        let pad = usize::from(label_w).saturating_sub(label.width());
        if pad > 0 {
            label.spans.push(Span::raw(" ".repeat(pad)));
        }
        let ratio = self.ratio();
        let color = load_color(ratio, self.full_scale.is_some_and(|f| f > 0.0));
        f.render_widget(
            LineGauge::default()
                .ratio(ratio)
                .label(label)
                .line_set(symbols::line::THICK)
                .filled_style(Style::default().fg(color))
                .unfilled_style(Style::default().fg(Color::DarkGray)),
            area,
        );
    }
}

/// The longest label form whose widest label leaves [`MIN_BAR`] cells in `width`, and that
/// width; the shortest form when none does.
pub fn fit(gauges: &[LinkGauge], width: u16) -> (usize, u16) {
    let widest = |level| {
        gauges
            .iter()
            .map(|g| g.label(level).width())
            .max()
            .unwrap_or(0)
            .min(usize::from(u16::MAX)) as u16
    };
    for level in 0..LEVELS {
        let w = widest(level);
        if u32::from(w) + 1 + u32::from(MIN_BAR) <= u32::from(width) {
            return (level, w);
        }
    }
    (LEVELS - 1, widest(LEVELS - 1))
}

/// `gauges` one per row in `area`; more gauges than rows go into side-by-side columns.
pub fn render_stack(f: &mut Frame, area: Rect, gauges: &[LinkGauge]) {
    if gauges.is_empty() || area.is_empty() {
        return;
    }
    let rows = usize::from(area.height);
    let cols = gauges.len().div_ceil(rows);
    let col_w = area.width / cols as u16;
    for (c, chunk) in gauges.chunks(rows).enumerate() {
        let x = area.x + c as u16 * col_w;
        // One blank cell between columns; the last one takes what is left.
        let width = if c + 1 < cols {
            col_w.saturating_sub(1)
        } else {
            area.right() - x
        };
        let (level, label_w) = fit(chunk, width);
        for (i, g) in chunk.iter().enumerate() {
            let row = Rect {
                x,
                y: area.y + i as u16,
                width,
                height: 1,
            };
            g.render(f, row, level, label_w);
        }
    }
}
