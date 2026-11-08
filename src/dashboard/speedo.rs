//! The memory-bandwidth speedometer: a half dial whose full scale is the memory's peak
//! bandwidth, with a needle for the measured bandwidth (GPM) and one for Legion's own bytes.
//! Where nothing gives a full scale the dial is not drawn; the figures still are. Small panels
//! draw [`Speedo::bar_lines`] instead. The links have [`super::gauge`] bars.

use std::f64::consts::PI;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CLine};
use ratatui::widgets::{Block, Paragraph};

use super::gauge::{LEGION, RX as MEASURED};

/// One dial's data, in GB/s.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Speedo {
    pub title: String,
    pub full_scale: Option<f64>,
    /// Measured bandwidth; `None`: no hardware counter (the needle is left out).
    pub measured: Option<f64>,
    /// Legion's own bytes.
    pub legion: f64,
    /// What the dial cannot show, in a few words (`no GPM`).
    pub note: Option<String>,
    /// The 10 s peak of the measured figure (Legion's where nothing measures): a tick on the
    /// arc and `pk` in the figures.
    pub peak: Option<f64>,
    /// The same figure's highest value since the dashboard started (or `r`): a second tick
    /// and `max`.
    pub max: Option<f64>,
}

/// The peak tick's colour.
pub const PEAK: Color = Color::LightRed;
/// The max tick's colour.
pub const MAX: Color = Color::White;

/// A GB/s figure, short: `0.42`, `12.3`, `450`.
pub fn gb(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

fn opt_gb(v: Option<f64>) -> String {
    v.map_or_else(|| "n/a".into(), gb)
}

/// The needle's angle for `v` of `full`: pi (left, 0) to 0 (right, full scale).
pub fn angle(v: f64, full: f64) -> f64 {
    if full <= 0.0 || !v.is_finite() {
        return PI;
    }
    PI * (1.0 - (v / full).clamp(0.0, 1.0))
}

/// A text bar of `width` cells for `v` of `full`.
pub fn bar(v: Option<f64>, full: Option<f64>, width: usize) -> String {
    let (Some(v), Some(full)) = (v, full) else {
        return "·".repeat(width);
    };
    let frac = if full > 0.0 {
        (v / full).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eighths = (frac * width as f64 * 8.0).round() as usize;
    let mut s = "█".repeat(eighths / 8);
    if eighths % 8 > 0 && s.chars().count() < width {
        s.push([' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'][eighths % 8]);
    }
    let n = s.chars().count();
    s.push_str(&"·".repeat(width.saturating_sub(n)));
    s
}

impl Speedo {
    fn values_line(&self) -> Line<'static> {
        let mut spans = Vec::new();
        if let Some(m) = self.measured {
            spans.push(Span::styled(
                format!("meas {} ", gb(m)),
                Style::default().fg(MEASURED),
            ));
        }
        spans.push(Span::styled(
            format!("L {}", gb(self.legion)),
            Style::default().fg(LEGION),
        ));
        spans.extend(self.peak_spans());
        Line::from(spans)
    }

    /// ` pk 210` and ` max 250`, when there are a peak and a max (a narrow panel cuts the max
    /// first).
    fn peak_spans(&self) -> Vec<Span<'static>> {
        let fig = |v: Option<f64>, name: &str, color: Color| {
            v.filter(|v| v.is_finite())
                .map(|v| Span::styled(format!(" {name} {}", gb(v)), Style::default().fg(color)))
        };
        fig(self.peak, "pk", PEAK)
            .into_iter()
            .chain(fig(self.max, "max", MAX))
            .collect()
    }

    fn scale_line(&self) -> Line<'static> {
        let mut s = match self.full_scale {
            Some(f) => format!("/{} GB/s", gb(f)),
            None => "full scale n/a".into(),
        };
        if let Some(n) = &self.note {
            s = format!("{s} {n}");
        }
        Line::styled(s, Style::default().fg(Color::DarkGray))
    }

    /// The dial in `area` (with its border): at least 18 x 7 for a dial, smaller areas get
    /// the bars.
    pub fn render(&self, f: &mut Frame, area: Rect, focused: bool) {
        let block = Block::bordered()
            .title(Span::styled(
                self.title.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ))
            .border_style(if focused {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::DarkGray)
            });
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 5 || inner.width < 16 || self.full_scale.is_none() {
            let lines = self.bar_lines(inner.width as usize);
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        let [dial, text] =
            Layout::vertical([Constraint::Min(3), Constraint::Length(2)]).areas(inner);
        let full = self.full_scale.unwrap_or(1.0);
        let (measured, legion, peak, max) = (self.measured, self.legion, self.peak, self.max);
        let canvas = Canvas::default()
            .marker(Marker::Braille)
            .x_bounds([-1.15, 1.15])
            .y_bounds([-0.1, 1.1])
            .paint(move |ctx| {
                // The dial: an arc and ticks every quarter.
                let steps = 48;
                for i in 0..steps {
                    let (a0, a1) = (
                        PI * i as f64 / steps as f64,
                        PI * (i + 1) as f64 / steps as f64,
                    );
                    ctx.draw(&CLine {
                        x1: a0.cos(),
                        y1: a0.sin(),
                        x2: a1.cos(),
                        y2: a1.sin(),
                        color: Color::DarkGray,
                    });
                }
                for q in 0..=4 {
                    let a = PI * q as f64 / 4.0;
                    ctx.draw(&CLine {
                        x1: 0.85 * a.cos(),
                        y1: 0.85 * a.sin(),
                        x2: a.cos(),
                        y2: a.sin(),
                        color: Color::Gray,
                    });
                }
                // The max and the peak: ticks across the arc, the max's longer so it still
                // shows where the peak sits on it.
                for (v, r0, color) in [(max, 0.7, MAX), (peak, 0.8, PEAK)] {
                    if let Some(v) = v.filter(|v| v.is_finite()) {
                        let a = angle(v, full);
                        ctx.draw(&CLine {
                            x1: r0 * a.cos(),
                            y1: r0 * a.sin(),
                            x2: 1.1 * a.cos(),
                            y2: 1.1 * a.sin(),
                            color,
                        });
                    }
                }
                let mut needle = |v: f64, r: f64, color: Color| {
                    let a = angle(v, full);
                    ctx.draw(&CLine {
                        x1: 0.0,
                        y1: 0.0,
                        x2: r * a.cos(),
                        y2: r * a.sin(),
                        color,
                    });
                };
                needle(legion, 0.6, LEGION);
                if let Some(v) = measured {
                    needle(v, 0.92, MEASURED);
                }
            });
        f.render_widget(canvas, dial);
        f.render_widget(
            Paragraph::new(vec![self.values_line(), self.scale_line()]),
            text,
        );
    }

    /// Two or three lines of bars for small panels: `meas ███··· 163`, `L ...`, the scale.
    pub fn bar_lines(&self, width: usize) -> Vec<Line<'static>> {
        let w = width.saturating_sub(12).clamp(4, 40);
        let row = |label: &str, v: Option<f64>, color: Color| {
            Line::from(vec![
                Span::raw(format!("{label:<4}")),
                Span::styled(bar(v, self.full_scale, w), Style::default().fg(color)),
                Span::raw(format!(" {}", opt_gb(v))),
            ])
        };
        let mut lines = Vec::new();
        if self.measured.is_some() {
            lines.push(row("meas", self.measured, MEASURED));
        }
        lines.push(row("L", Some(self.legion), LEGION));
        let mut scale = self.scale_line();
        scale.spans.extend(self.peak_spans());
        lines.push(scale);
        lines
    }
}
