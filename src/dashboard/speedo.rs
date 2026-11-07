//! A speedometer: a half dial whose full scale is the link's (or memory's) bandwidth per
//! direction, with up to three needles (TX, RX, Legion's own bytes) and a peak-hold tick per
//! direction. Where nothing gives a full scale the dial is not drawn; the figures still are.
//! Small panels draw [`Speedo::bar_lines`] instead.

use std::f64::consts::PI;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CLine};
use ratatui::widgets::{Block, Paragraph};

pub const TX: Color = Color::Cyan;
pub const RX: Color = Color::Magenta;
pub const LEGION: Color = Color::Yellow;

/// One dial's data, in GB/s.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Speedo {
    pub title: String,
    pub full_scale: Option<f64>,
    /// Measured sent / received; `None`: no hardware counter (the needle is left out).
    pub tx: Option<f64>,
    pub rx: Option<f64>,
    /// Legion's own bytes; `None`: not counted for this link.
    pub legion: Option<f64>,
    /// The highest TX and RX over the last seconds.
    pub peak_tx: Option<f64>,
    pub peak_rx: Option<f64>,
    /// The values' names (`TX`, `RX`; a memory dial has `meas` and no second, empty).
    pub tx_label: &'static str,
    pub rx_label: &'static str,
    /// What the dial cannot show, in a few words (`no PCIe counter`).
    pub note: Option<String>,
}

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
        if self.tx.is_some() || self.rx.is_some() || self.legion.is_none() {
            spans.push(Span::styled(
                format!("{} {} ", self.tx_label, opt_gb(self.tx)),
                Style::default().fg(TX),
            ));
            if !self.rx_label.is_empty() {
                spans.push(Span::styled(
                    format!("{} {} ", self.rx_label, opt_gb(self.rx)),
                    Style::default().fg(RX),
                ));
            }
        }
        if let Some(l) = self.legion {
            spans.push(Span::styled(
                format!("L {}", gb(l)),
                Style::default().fg(LEGION),
            ));
        }
        Line::from(spans)
    }

    fn scale_line(&self) -> Line<'static> {
        let mut s = match self.full_scale {
            Some(f) => format!("/{} GB/s", gb(f)),
            None => "full scale n/a".into(),
        };
        if let (Some(t), Some(r)) = (self.peak_tx, self.peak_rx) {
            s.push_str(&format!(" pk {}/{}", gb(t), gb(r)));
        }
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
        let me = self.clone();
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
                if let Some(v) = me.legion {
                    needle(v, 0.6, LEGION);
                }
                if let Some(v) = me.rx {
                    needle(v, 0.8, RX);
                }
                if let Some(v) = me.tx {
                    needle(v, 0.92, TX);
                }
                for (p, color) in [(me.peak_tx, TX), (me.peak_rx, RX)] {
                    if let Some(p) = p {
                        let a = angle(p, full);
                        ctx.draw(&CLine {
                            x1: 0.95 * a.cos(),
                            y1: 0.95 * a.sin(),
                            x2: 1.1 * a.cos(),
                            y2: 1.1 * a.sin(),
                            color,
                        });
                    }
                }
            });
        f.render_widget(canvas, dial);
        f.render_widget(
            Paragraph::new(vec![self.values_line(), self.scale_line()]),
            text,
        );
    }

    /// Two or three lines of bars for small panels: `TX ███··· 12.3`, `RX ...`, the scale.
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
        if self.tx.is_some() || self.rx.is_some() || self.legion.is_none() {
            lines.push(row(self.tx_label, self.tx, TX));
            if !self.rx_label.is_empty() {
                lines.push(row(self.rx_label, self.rx, RX));
            }
        }
        if self.legion.is_some() {
            lines.push(row("L", self.legion, LEGION));
        }
        lines.push(self.scale_line());
        lines
    }
}
