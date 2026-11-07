//! The dashboard without a terminal or a server: full-scale math, rates, latency windows, the
//! log ring, the client's event parsing, the snapshot's JSON, the keys, and every widget drawn
//! on ratatui's `TestBackend` at 80 x 24, 120 x 46 and 200 x 60.

use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier};
use ratatui::widgets::Widget;
use tracing::Level;

use super::client::{base_url, next_backoff, push_utf8, take_events};
use super::gauge::{
    self, Bar, BarColors, BarRow, ColorDepth, Dir, GHOST, LEVELS, LinkGauge, Plan, VALUE_LEVELS,
    cube, empty_color, gradient, kind_color, load_color, on_fill_color, value_label,
};
use super::latency::{BUCKETS, LatencyWindow, bucket, bucket_end, quantile};
use super::logs::{CAPACITY, LogLine, LogRing, kept, level_rank};
use super::scale::*;
use super::snapshot::*;
use super::speedo::{Speedo, angle, bar, gb};
use super::term::{Feed, Inbox};
use super::ui::{self, Action, App, Conn, Mode, PANELS, Panel, PeakOf};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6 * b.abs().max(1.0)
}

// ---- full scales ----

#[test]
fn pcie_full_scale_is_lane_rate_times_width() {
    assert!(close(pcie_gb_s(3, 16).unwrap(), 15.76));
    assert!(close(pcie_gb_s(4, 16).unwrap(), 31.504));
    assert!(close(pcie_gb_s(5, 16).unwrap(), 63.008));
    assert!(close(pcie_gb_s(4, 8).unwrap(), 15.752));
    assert!(close(pcie_gb_s(1, 1).unwrap(), 0.25));
    assert_eq!(pcie_gb_s(0, 16), None);
    assert_eq!(pcie_gb_s(9, 16), None);
    assert_eq!(pcie_label(4, 16), "PCIe 4.0 x16");
}

#[test]
fn nvlink_full_scale_is_links_times_version_speed() {
    assert_eq!(nvlink_gb_s(3, 2), Some(75.0));
    assert_eq!(nvlink_gb_s(6, 2), Some(150.0));
    assert_eq!(nvlink_gb_s(1, 1), Some(20.0));
    assert_eq!(nvlink_gb_s(18, 4), Some(450.0));
    assert_eq!(nvlink_gb_s(2, 5), Some(100.0));
    assert_eq!(nvlink_gb_s(0, 2), None);
    assert_eq!(nvlink_gb_s(3, 0), None);
}

#[test]
fn links_are_classed_by_remote_type_then_by_bus_id() {
    assert_eq!(classify(Some("gpu"), None), LinkClass::Peer);
    assert_eq!(classify(Some("switch"), None), LinkClass::Peer);
    assert_eq!(classify(Some("cpu"), Some(true)), LinkClass::Cpu);
    assert_eq!(classify(Some("unknown"), Some(true)), LinkClass::Peer);
    assert_eq!(classify(None, Some(false)), LinkClass::Cpu);
    // An AC922's POWER9 NPU reports nothing useful: the CPU.
    assert_eq!(classify(None, None), LinkClass::Cpu);
}

#[test]
fn c2c_full_scale_from_nvml_else_the_grace_spec() {
    let (gb, src) = c2c_gb_s(Some(10), Some(45_000), "NVIDIA GH200 480GB");
    assert_eq!(gb, Some(450.0));
    assert!(src.starts_with("NVML"));
    assert_eq!(c2c_gb_s(None, None, "NVIDIA GH200 480GB").0, Some(450.0));
    assert_eq!(c2c_gb_s(Some(0), Some(1), "NVIDIA GB300").0, Some(450.0));
    let (none, why) = c2c_gb_s(None, None, "NVIDIA GB10");
    assert_eq!(none, None);
    assert_eq!(why, "not reported by NVML");
}

#[test]
fn memory_peak_from_bus_width_and_clock() {
    // V100 SXM2: 4096 bit HBM2 at 877 MHz.
    let (v100, src) = memory_peak_gb_s(Some(4096), Some(877), "Tesla V100-SXM2-16GB");
    assert!(close(v100.unwrap(), 898.048));
    assert!(src.contains("4096-bit"));
    // RTX 4090: 384 bit GDDR6X, NVML's 10501 MHz.
    let (rtx, _) = memory_peak_gb_s(Some(384), Some(10501), "NVIDIA GeForce RTX 4090");
    assert!(close(rtx.unwrap(), 1008.096));
    // GB10: the LPDDR5X spec, whatever NVML says.
    assert_eq!(
        memory_peak_gb_s(Some(0), Some(0), "NVIDIA GB10").0,
        Some(273.0)
    );
    assert_eq!(
        memory_peak_gb_s(Some(256), Some(9999), "NVIDIA GB10").0,
        Some(273.0)
    );
    // Nothing reported, an unknown part.
    assert_eq!(memory_peak_gb_s(None, None, "x").0, None);
    // A placeholder value is not a bandwidth.
    assert_eq!(memory_peak_gb_s(Some(1), Some(1), "x").0, None);
}

#[test]
fn nic_speed_from_sysfs_text() {
    assert_eq!(nic_gb_s_from_rate("200 Gb/sec (4X HDR)"), Some(25.0));
    assert_eq!(nic_gb_s_from_rate("100 Gb/sec (4X EDR)"), Some(12.5));
    assert_eq!(nic_gb_s_from_rate("100000"), Some(12.5));
    assert_eq!(nic_gb_s_from_rate("-1"), None);
    assert_eq!(nic_gb_s_from_rate(""), None);
    assert_eq!(nic_gb_s_from_rate("fast"), None);
}

// ---- gauges ----

#[test]
fn needle_angles_run_left_to_right_and_clamp() {
    use std::f64::consts::PI;
    assert!(close(angle(0.0, 10.0), PI));
    assert!(close(angle(5.0, 10.0), PI / 2.0));
    assert!(close(angle(10.0, 10.0), 0.0));
    assert!(close(angle(30.0, 10.0), 0.0));
    assert!(close(angle(-1.0, 10.0), PI));
    assert!(close(angle(1.0, 0.0), PI));
    assert!(close(angle(f64::NAN, 10.0), PI));
}

#[test]
fn bars_fill_in_eighths() {
    assert_eq!(bar(Some(0.0), Some(10.0), 4), "····");
    assert_eq!(bar(Some(10.0), Some(10.0), 4), "████");
    assert_eq!(bar(Some(5.0), Some(10.0), 4), "██··");
    assert_eq!(bar(Some(20.0), Some(10.0), 4), "████");
    // 1/8 of 4 cells = half a cell = 4 eighths.
    assert_eq!(bar(Some(1.25), Some(10.0), 4), "▌···");
    assert_eq!(bar(None, Some(10.0), 3), "···");
    assert_eq!(bar(Some(1.0), None, 3), "···");
    for v in [0.0, 0.1, 3.3, 7.7, 9.99, 10.0] {
        assert_eq!(bar(Some(v), Some(10.0), 7).chars().count(), 7, "{v}");
    }
}

#[test]
fn gb_figures_are_short() {
    assert_eq!(gb(0.4242), "0.42");
    assert_eq!(gb(12.34), "12.3");
    assert_eq!(gb(450.2), "450");
}

fn line_text(l: &ratatui::text::Line) -> String {
    l.spans.iter().map(|s| s.content.as_ref()).collect()
}

#[test]
fn bar_lines_show_what_a_memory_dial_has() {
    let d = Speedo {
        title: "GPU0 memory (GPM)".into(),
        full_scale: Some(273.0),
        measured: Some(163.8),
        legion: 120.0,
        note: None,
        peak: Some(210.0),
    };
    let text: Vec<String> = d.bar_lines(40).iter().map(line_text).collect();
    assert_eq!(text.len(), 3);
    assert!(text[0].starts_with("meas") && text[0].ends_with("164"));
    assert!(text[1].starts_with('L') && text[1].ends_with("120"));
    assert_eq!(text[2], "/273 GB/s pk 210");
    // Unified memory without GPM: Legion's bytes only.
    let m = Speedo {
        measured: None,
        full_scale: None,
        note: Some("no GPM".into()),
        peak: None,
        ..d
    };
    let text: Vec<String> = m.bar_lines(40).iter().map(line_text).collect();
    assert_eq!(text.len(), 2);
    assert!(text[0].starts_with('L'));
    assert_eq!(text[1], "full scale n/a no GPM");
}

fn nvlink_cpu(rx: Option<f64>, tx: Option<f64>) -> LinkGauge {
    LinkGauge {
        kind: LinkKind::NvlinkCpu,
        gpu: Some(0),
        name: "GPU0 NVLink CPU".into(),
        short: "G0 CPU".into(),
        full_scale: Some(75.0),
        rx,
        tx,
        legion: Some(12.1),
        rx_peak: Some(22.0),
        tx_peak: Some(6.1),
        legion_peak: Some(20.0),
        why: (rx.is_none() && tx.is_none()).then(|| "no NVML counter".into()),
        detail: Some("3 links NVLink 2".into()),
    }
}

#[test]
fn a_link_has_an_rx_and_a_tx_bar_each_over_the_full_scale() {
    let g = nvlink_cpu(Some(14.5), Some(3.2));
    assert_eq!(
        g.rows(),
        vec![
            BarRow::Bar(Dir::Rx, Some(14.5), Some(22.0)),
            BarRow::Bar(Dir::Tx, Some(3.2), Some(6.1))
        ]
    );
    assert!(close(g.ratio(g.rx), 14.5 / 75.0));
    assert!(close(g.ratio(g.tx), 3.2 / 75.0));
    // Over the full scale clamps; nothing, NaN, no full scale or a zero one read 0.
    assert_eq!(g.ratio(Some(100.0)), 1.0);
    assert_eq!(g.ratio(None), 0.0);
    assert_eq!(g.ratio(Some(f64::NAN)), 0.0);
    assert_eq!(g.ratio(Some(f64::INFINITY)), 0.0);
    let unscaled = LinkGauge {
        full_scale: None,
        ..g.clone()
    };
    assert_eq!(unscaled.ratio(Some(10.0)), 0.0);
    let zero = LinkGauge {
        full_scale: Some(0.0),
        ..g
    };
    assert_eq!(zero.ratio(Some(10.0)), 0.0);
    // No counter: one bar of Legion's bytes and the reason; without Legion bytes the reason.
    let n = nvlink_cpu(None, None);
    assert_eq!(
        n.rows(),
        vec![
            BarRow::Bar(Dir::Legion, Some(12.1), Some(20.0)),
            BarRow::Note("RX/TX n/a (no NVML counter)".into())
        ]
    );
    assert!(close(n.ratio(n.legion), 12.1 / 75.0));
    let bare = LinkGauge {
        legion: None,
        why: None,
        ..n
    };
    assert_eq!(
        bare.rows(),
        vec![BarRow::Note("RX/TX n/a (no counter)".into())]
    );
}

#[test]
fn the_fill_follows_a_gradient_with_a_16_colour_fallback() {
    // The stops exactly; NaN and out-of-range clamp.
    assert_eq!(gradient(0.0), (40, 200, 80));
    assert_eq!(gradient(0.6), (235, 210, 40));
    assert_eq!(gradient(0.8), (255, 140, 30));
    assert_eq!(gradient(1.0), (235, 45, 45));
    assert_eq!(gradient(f64::NAN), gradient(0.0));
    assert_eq!(gradient(-1.0), gradient(0.0));
    assert_eq!(gradient(7.0), gradient(1.0));
    // Smooth: between the stops, red rises to orange and green falls after yellow.
    let (r3, g3, _) = gradient(0.3);
    assert!(r3 > 40 && r3 < 235 && g3 > 200 && g3 < 210, "{r3} {g3}");
    let (_, g7, _) = gradient(0.7);
    assert!(g7 > 140 && g7 < 210, "{g7}");
    for i in 0..100 {
        let (a, b) = (
            gradient(f64::from(i) / 100.0),
            gradient(f64::from(i + 1) / 100.0),
        );
        let step = |x: u8, y: u8| (i32::from(x) - i32::from(y)).abs();
        assert!(step(a.0, b.0) <= 12 && step(a.1, b.1) <= 12 && step(a.2, b.2) <= 12);
    }
    // 16 colours: green under 60%, yellow under 85%, red above.
    let basic = |r| load_color(r, true, ColorDepth::Ansi16);
    assert_eq!(basic(0.0), Color::Green);
    assert_eq!(basic(0.59), Color::Green);
    assert_eq!(basic(0.6), Color::Yellow);
    assert_eq!(basic(0.84), Color::Yellow);
    assert_eq!(basic(0.85), Color::Red);
    assert_eq!(basic(1.0), Color::Red);
    assert_eq!(load_color(0.9, false, ColorDepth::Ansi16), Color::Gray);
    // RGB and the 256-colour cube.
    let (r, g, b) = gradient(0.3);
    assert_eq!(load_color(0.3, true, ColorDepth::Rgb), Color::Rgb(r, g, b));
    assert_eq!(
        load_color(0.3, true, ColorDepth::Ansi256),
        Color::Indexed(cube(r, g, b))
    );
    assert_eq!(cube(0, 0, 0), 16);
    assert_eq!(cube(255, 255, 255), 231);
    assert_eq!(cube(255, 0, 0), 196);
    assert_eq!(cube(0, 255, 0), 46);
    assert_eq!(cube(255, 135, 0), 208);
    // The empty part is visible and differs from every fill; the text on the fill is dark.
    for depth in [ColorDepth::Ansi16, ColorDepth::Ansi256, ColorDepth::Rgb] {
        assert_ne!(empty_color(depth), Color::Reset);
        assert_ne!(empty_color(depth), Color::Black);
        for i in 0..=20 {
            let r = f64::from(i) / 20.0;
            assert_ne!(
                load_color(r, true, depth),
                empty_color(depth),
                "{depth:?} {r}"
            );
        }
        assert_ne!(load_color(0.0, false, depth), empty_color(depth));
    }
    assert_eq!(on_fill_color(ColorDepth::Ansi16), Color::Black);
    // 256 colours: the empty part and the dark text from the grey ramp, not the cube.
    assert_eq!(empty_color(ColorDepth::Ansi256), Color::Indexed(237));
    assert_eq!(on_fill_color(ColorDepth::Ansi256), Color::Indexed(233));
    // The ghost fill: the fill's hue, dimmer, apart from the fill and the empty part; in 16
    // colours the fill's own colour, dim, over the empty part.
    for depth in [ColorDepth::Ansi256, ColorDepth::Rgb] {
        for i in 0..=20 {
            let r = f64::from(i) / 20.0;
            let c = BarColors::of(r, true, depth);
            assert_ne!(c.ghost, c.fill, "{depth:?} {r}");
            assert_ne!(c.ghost, c.empty, "{depth:?} {r}");
            assert_eq!((c.ghost_bg, c.ghost_dim), (c.ghost, false));
        }
    }
    let c = BarColors::of(0.3, true, ColorDepth::Rgb);
    let ((fr, fg, fb), Color::Rgb(gr, gg, gb)) = (gradient(0.3), c.ghost) else {
        panic!("{c:?}")
    };
    let sum = |r: u8, g: u8, b: u8| u32::from(r) + u32::from(g) + u32::from(b);
    assert!(sum(gr, gg, gb) < sum(fr, fg, fb), "{c:?}");
    let c = BarColors::of(0.9, true, ColorDepth::Ansi16);
    assert_eq!(
        (c.fill, c.ghost, c.ghost_dim),
        (Color::Red, Color::Red, true)
    );
    assert_eq!(c.ghost_bg, Color::DarkGray);
    // Link titles by type.
    assert_eq!(
        kind_color(LinkKind::NvlinkCpu),
        kind_color(LinkKind::NvlinkPeer)
    );
    let kinds = [
        LinkKind::NvlinkCpu,
        LinkKind::Pcie,
        LinkKind::C2c,
        LinkKind::Network,
    ];
    for (i, a) in kinds.iter().enumerate() {
        for b in &kinds[i + 1..] {
            assert_ne!(kind_color(*a), kind_color(*b), "{a:?} {b:?}");
        }
    }
}

#[test]
fn colour_depth_from_the_terminal() {
    let d = |ct: Option<&str>, term: Option<&str>| ColorDepth::from_env(ct, term);
    assert_eq!(
        d(Some("truecolor"), Some("xterm-256color")),
        ColorDepth::Rgb
    );
    assert_eq!(d(Some("24bit"), None), ColorDepth::Rgb);
    assert_eq!(d(None, Some("xterm-kitty")), ColorDepth::Rgb);
    assert_eq!(d(None, Some("xterm-ghostty")), ColorDepth::Rgb);
    assert_eq!(d(None, Some("xterm-direct")), ColorDepth::Rgb);
    assert_eq!(d(None, Some("tmux-256color")), ColorDepth::Ansi256);
    assert_eq!(d(None, Some("xterm-256color")), ColorDepth::Ansi256);
    assert_eq!(d(Some(""), Some("xterm")), ColorDepth::Ansi16);
    assert_eq!(d(None, Some("linux")), ColorDepth::Ansi16);
    assert_eq!(d(None, None), ColorDepth::Ansi16);
}

#[test]
fn values_inside_a_bar_get_shorter_level_by_level() {
    let r = 14.5 / 75.0;
    let text: Vec<String> = (0..VALUE_LEVELS)
        .map(|l| value_label(Some(14.5), Some(22.0), r, true, l))
        .collect();
    assert_eq!(
        text,
        [
            "14.5 GB/s  19%  pk 22.0",
            "14.5 GB/s  19%",
            "14.5 GB/s 19%",
            "14.5 19%",
            "14.5",
            ""
        ]
    );
    // No peak yet: the first form without it.
    assert_eq!(value_label(Some(14.5), None, r, true, 0), "14.5 GB/s  19%");
    assert_eq!(
        value_label(Some(14.5), Some(f64::NAN), r, true, 0),
        "14.5 GB/s  19%"
    );
    // No full scale: no percentage.
    assert_eq!(
        value_label(Some(1.0), Some(2.0), 0.0, false, 0),
        "1.00 GB/s  pk 2.00"
    );
    assert_eq!(
        value_label(Some(1.0), Some(2.0), 0.0, false, 1),
        "1.00 GB/s"
    );
    assert_eq!(value_label(Some(1.0), None, 0.0, false, 3), "1.00");
    assert_eq!(
        value_label(Some(70.0), Some(6.1), 70.0 / 75.0, true, 0),
        "70.0 GB/s  93%  pk 6.10"
    );
    assert_eq!(value_label(None, Some(1.0), 0.0, true, 0), "n/a");
    assert_eq!(value_label(Some(f64::NAN), None, 0.0, true, 1), "n/a");
    assert_eq!(value_label(None, None, 0.0, true, VALUE_LEVELS - 1), "");
}

#[test]
fn link_titles_get_shorter_level_by_level() {
    let g = nvlink_cpu(Some(14.5), Some(3.2));
    let text: Vec<String> = (0..LEVELS).map(|l| line_text(&g.title(l))).collect();
    assert_eq!(
        text,
        [
            "GPU0 NVLink CPU  / 75.0 GB/s  Legion 12.1  3 links NVLink 2",
            "GPU0 NVLink CPU  / 75.0 GB/s  Legion 12.1",
            "G0 CPU / 75.0 GB/s L 12.1",
            "G0 CPU /75.0 L 12.1",
            "G0 CPU L 12.1",
        ]
    );
    for w in text.windows(2) {
        assert!(w[1].chars().count() < w[0].chars().count(), "{w:?}");
    }
    // The name in the link type's colour, Legion's figure yellow.
    let t = g.title(0);
    assert_eq!(t.spans[0].style.fg, Some(kind_color(LinkKind::NvlinkCpu)));
    assert!(t.spans[0].style.add_modifier.contains(Modifier::BOLD));
    let legion = t
        .spans
        .iter()
        .find(|s| s.content.starts_with("Legion"))
        .unwrap();
    assert_eq!(legion.style.fg, Some(gauge::LEGION));
    // Peer links have no Legion bytes; no full scale says so.
    let p = LinkGauge {
        kind: LinkKind::NvlinkPeer,
        gpu: Some(1),
        name: "GPU1 NVLink peer".into(),
        short: "G1 peer".into(),
        legion: None,
        full_scale: None,
        detail: None,
        ..nvlink_cpu(Some(1.0), Some(1.0))
    };
    assert_eq!(line_text(&p.title(0)), "GPU1 NVLink peer  full scale n/a");
    assert_eq!(line_text(&p.title(3)), "G1 peer /n/a");
    // Row tags: `RX` under a title, the short name and `RX` without one.
    let rx = BarRow::Bar(Dir::Rx, Some(1.0), None);
    assert_eq!(line_text(&p.tag(&rx, true)), "RX");
    assert_eq!(line_text(&p.tag(&rx, false)), "G1 peer RX");
    assert_eq!(p.tag(&rx, true).spans[0].style.fg, Some(gauge::RX));
    let tx = BarRow::Bar(Dir::Tx, Some(1.0), None);
    assert_eq!(p.tag(&tx, true).spans[0].style.fg, Some(gauge::TX));
    assert_ne!(gauge::RX, gauge::TX);
    assert_eq!(
        line_text(&p.tag(&BarRow::Note("x".into()), false)),
        "G1 peer"
    );
}

#[test]
fn a_column_picks_the_longest_titles_and_values_that_fit() {
    let gauges = vec![nvlink_cpu(Some(14.5), Some(3.2)), nvlink_cpu(None, None)];
    assert_eq!(gauge::fit_title(&gauges, 200), 0);
    for width in [20u16, 30, 38, 50, 60, 70, 90] {
        let level = gauge::fit_title(&gauges, width);
        let widest = |l: usize| gauges.iter().map(|g| g.title(l).width()).max().unwrap();
        if level + 1 < LEVELS {
            assert!(
                widest(level) <= usize::from(width),
                "{width}: level {level}"
            );
        }
        if level > 0 {
            // The next longer form would not have fit.
            assert!(
                widest(level - 1) > usize::from(width),
                "{width}: level {level}"
            );
        }
    }
    // Too narrow for any form: the shortest, clipped.
    assert_eq!(gauge::fit_title(&gauges, 5), LEVELS - 1);
    // Values keep a cell each side: `14.5 GB/s  19%  pk 22.0` needs 25 cells, and the peak
    // goes first.
    assert_eq!(gauge::fit_value(&gauges, 25), 0);
    assert_eq!(gauge::fit_value(&gauges, 24), 1);
    assert_eq!(gauge::fit_value(&gauges, 16), 1);
    assert_eq!(gauge::fit_value(&gauges, 15), 2);
    assert_eq!(gauge::fit_value(&gauges, 10), 3);
    assert_eq!(gauge::fit_value(&gauges, 9), 4);
    assert_eq!(gauge::fit_value(&gauges, 6), 4);
    assert_eq!(gauge::fit_value(&gauges, 3), VALUE_LEVELS - 1);
}

#[test]
fn links_go_into_columns_and_drop_their_titles_when_short_of_room() {
    let four = app_with(ac922_4, 2).links();
    let two = app_with(ac922, 2).links();
    let plan = |titled, cols, per_col| Plan {
        titled,
        cols,
        per_col,
    };
    // Four V100s: a GPU's two links per column, two columns, titled where there is room.
    assert_eq!(gauge::plan(&four, 198, 26), plan(true, 2, 4));
    assert_eq!(gauge::plan(&four, 118, 12), plan(true, 2, 4));
    assert_eq!(gauge::plan(&four, 118, 8), plan(false, 2, 4));
    // The compact layout's links panel (80 x 24): no room for titles with four GPUs.
    assert_eq!(gauge::plan(&four, 38, 14), plan(false, 2, 4));
    assert_eq!(gauge::plan(&two, 198, 26), plan(true, 2, 2));
    assert_eq!(gauge::plan(&two, 118, 12), plan(true, 2, 2));
    assert_eq!(gauge::plan(&two, 38, 14), plan(true, 1, 4));
    // No room at all: one column, cut at the bottom.
    assert_eq!(gauge::plan(&four, 38, 1), plan(false, 1, 8));
    assert_eq!(plan(true, 2, 4).rows(), 12);
    assert_eq!(plan(false, 2, 4).rows(), 8);
    // The GB10: C2C and TP net side by side.
    assert_eq!(
        gauge::plan(&app_with(gb10, 2).links(), 164, 10),
        plan(true, 2, 1)
    );
}

fn stack(gauges: &[LinkGauge], w: u16, h: u16, depth: ColorDepth) -> ratatui::buffer::Buffer {
    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
    t.draw(|f| {
        let area = f.area();
        gauge::render_stack(f, area, gauges, depth);
    })
    .unwrap();
    t.backend().buffer().clone()
}

fn row_text(buf: &ratatui::buffer::Buffer, y: u16, x0: u16, x1: u16) -> String {
    (x0..x1)
        .map(|x| buf.cell((x, y)).unwrap().symbol())
        .collect()
}

#[test]
fn block_gauges_fill_by_direction_with_the_value_readable_inside() {
    // RX at 10% with a 40% peak, TX at 93% with a lower peak: a title row and two bars from
    // column 3 (`RX `), 57 cells wide.
    let g = LinkGauge {
        rx_peak: Some(30.0),
        ..nvlink_cpu(Some(7.5), Some(70.0))
    };
    for depth in [ColorDepth::Ansi16, ColorDepth::Ansi256, ColorDepth::Rgb] {
        let buf = stack(std::slice::from_ref(&g), 60, 3, depth);
        let all = (0..3)
            .map(|y| row_text(&buf, y, 0, 60))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            row_text(&buf, 0, 0, 59),
            "GPU0 NVLink CPU  / 75.0 GB/s  Legion 12.1  3 links NVLink 2",
            "{all}"
        );
        assert!(row_text(&buf, 1, 0, 3).starts_with("RX"), "{all}");
        assert!(row_text(&buf, 2, 0, 3).starts_with("TX"), "{all}");
        assert_eq!(buf.cell((0, 1)).unwrap().fg, gauge::RX);
        assert_eq!(buf.cell((0, 2)).unwrap().fg, gauge::TX);
        assert_eq!(
            buf.cell((0, 0)).unwrap().fg,
            kind_color(LinkKind::NvlinkCpu)
        );
        let rx = BarColors::of(g.ratio(g.rx), true, depth);
        let tx = BarColors::of(g.ratio(g.tx), true, depth);
        assert_ne!(rx.fill, tx.fill, "{depth:?}");
        let empty = empty_color(depth);
        // RX: 57 x 0.1 = 5.7 cells: five full blocks (3..8) and a 3/4 block at 8.
        for x in 3..8 {
            let c = buf.cell((x, 1)).unwrap();
            assert_eq!(
                (c.symbol(), c.fg, c.bg),
                ("█", rx.fill, empty),
                "{x} {depth:?}"
            );
        }
        let c = buf.cell((8, 1)).unwrap();
        // The edge's eighths over the ghost fill.
        assert_eq!(
            (c.symbol(), c.fg, c.bg),
            ("▊", rx.fill, rx.ghost_bg),
            "{all}"
        );
        // The ghost fill to the 40% peak (57 x 0.4 = 22.8, so 23 cells: 9..26), the value
        // over its end.
        for x in 9..20 {
            let c = buf.cell((x, 1)).unwrap();
            assert_eq!(
                (c.symbol(), c.fg, c.bg),
                (GHOST, rx.ghost, empty),
                "{x} {depth:?}"
            );
            assert_eq!(
                c.modifier.contains(Modifier::DIM),
                depth == ColorDepth::Ansi16
            );
        }
        // The value, centred: 23 cells from 3 + (57 - 23) / 2 = 20, bold, in the fill's
        // colour over the ghost (20..26) and the empty part (26..43).
        assert_eq!(
            row_text(&buf, 1, 20, 43),
            "7.50 GB/s  10%  pk 30.0",
            "{all}"
        );
        for x in 20..43 {
            let c = buf.cell((x, 1)).unwrap();
            let bg = if x < 26 { rx.ghost_bg } else { empty };
            assert_eq!((c.fg, c.bg), (rx.fill, bg), "{x} {depth:?}");
            assert!(c.modifier.contains(Modifier::BOLD));
        }
        // The rest is the dim background, to the panel's edge.
        for x in 43..60 {
            let c = buf.cell((x, 1)).unwrap();
            assert_eq!((c.symbol(), c.bg), (" ", empty), "{x} {depth:?}");
        }
        // TX: 57 x 0.933 = 53.2 cells, so the value sits on the fill: dark text on it. The
        // peak is below the value: no ghost.
        assert_eq!(
            row_text(&buf, 2, 20, 43),
            "70.0 GB/s  93%  pk 6.10",
            "{all}"
        );
        for x in 20..43 {
            let c = buf.cell((x, 2)).unwrap();
            assert_eq!(
                (c.fg, c.bg),
                (on_fill_color(depth), tx.fill),
                "{x} {depth:?}"
            );
        }
        let c = buf.cell((10, 2)).unwrap();
        assert_eq!((c.symbol(), c.fg), ("█", tx.fill));
        let c = buf.cell((56, 2)).unwrap();
        assert_eq!((c.symbol(), c.bg), ("▎", empty), "{all}");
        assert!(!row_text(&buf, 2, 0, 60).contains(GHOST), "{all}");
        assert_eq!(buf.cell((57, 2)).unwrap().bg, empty);
    }
    // 16 colours: green and red, the ghost green and dim; RGB: the gradient.
    let buf = stack(std::slice::from_ref(&g), 60, 3, ColorDepth::Ansi16);
    assert_eq!(buf.cell((3, 1)).unwrap().fg, Color::Green);
    assert_eq!(buf.cell((10, 1)).unwrap().fg, Color::Green);
    assert_eq!(buf.cell((3, 2)).unwrap().fg, Color::Red);
    assert_eq!(buf.cell((59, 2)).unwrap().bg, Color::DarkGray);
    let buf = stack(std::slice::from_ref(&g), 60, 3, ColorDepth::Rgb);
    let (r, gr, b) = gradient(0.1);
    assert_eq!(buf.cell((3, 1)).unwrap().fg, Color::Rgb(r, gr, b));
    assert_ne!(buf.cell((10, 1)).unwrap().fg, Color::Rgb(r, gr, b));
    // A link without a counter: Legion's bar (`L`, yellow) and the reason.
    let n = nvlink_cpu(None, None);
    let buf = stack(std::slice::from_ref(&n), 60, 3, ColorDepth::Ansi16);
    assert!(
        row_text(&buf, 1, 0, 60).starts_with("L "),
        "{}",
        row_text(&buf, 1, 0, 60)
    );
    assert_eq!(buf.cell((0, 1)).unwrap().fg, gauge::LEGION);
    assert!(row_text(&buf, 1, 0, 60).contains("12.1 GB/s  16%  pk 20.0"));
    assert!(row_text(&buf, 2, 0, 60).starts_with("RX/TX n/a (no NVML counter)"));
}

#[test]
fn a_bar_draws_its_fill_ghost_and_edge_like_a_gauge() {
    let colors = BarColors::of(0.5, true, ColorDepth::Rgb);
    let draw = |ratio: f64, peak: f64, label: &str| {
        let mut buf = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 10, 1));
        Bar {
            ratio,
            peak,
            label,
            colors,
        }
        .render(buf.area, &mut buf);
        buf
    };
    let text = |b: &ratatui::buffer::Buffer| row_text(b, 0, 0, 10);
    assert_eq!(text(&draw(0.0, 0.0, "")), " ".repeat(10));
    assert_eq!(text(&draw(1.0, 1.0, "")), "█".repeat(10));
    // Half a cell is a half block; ratios outside 0..=1 and NaN clamp.
    assert_eq!(text(&draw(0.05, 0.0, "")), format!("▌{}", " ".repeat(9)));
    assert_eq!(text(&draw(7.0, f64::NAN, "")), "█".repeat(10));
    assert_eq!(text(&draw(f64::NAN, -1.0, "")), " ".repeat(10));
    // The ghost from the value to the peak.
    assert_eq!(text(&draw(0.5, 0.8, "")), "█████▓▓▓  ");
    // Over the ghost, the edge cell sits on the ghost's colour; without one, on the empty one.
    let b = draw(0.25, 0.8, "");
    assert_eq!(text(&b), "██▌▓▓▓▓▓  ");
    assert_eq!(b.cell((2, 0)).unwrap().bg, colors.ghost_bg);
    let b = draw(0.25, 0.0, "");
    assert_eq!(text(&b), "██▌       ");
    assert_eq!(b.cell((2, 0)).unwrap().bg, colors.empty);
    // The label: dark on the fill, the fill's colour on the ghost and the empty part.
    let fg_bg = |b: &ratatui::buffer::Buffer, x: u16| {
        let c = b.cell((x, 0)).unwrap();
        (c.fg, c.bg)
    };
    let b = draw(0.2, 0.6, "abcdef");
    assert_eq!(text(&b), "██abcdef  ");
    assert_eq!(fg_bg(&b, 2), (colors.fill, colors.ghost_bg));
    assert_eq!(fg_bg(&b, 6), (colors.fill, colors.empty));
    assert_eq!(
        fg_bg(&draw(0.9, 0.9, "abcdef"), 2),
        (colors.on_fill, colors.fill)
    );
    // A label wider than the bar is cut, never past it.
    let b = draw(0.0, 0.0, "0123456789abc");
    assert_eq!(text(&b), "0123456789");
    // Drawing into a smaller area leaves the rest alone.
    let mut buf = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 10, 1));
    Bar {
        ratio: 1.0,
        peak: 1.0,
        label: "",
        colors,
    }
    .render(Rect::new(2, 0, 4, 1), &mut buf);
    assert_eq!(row_text(&buf, 0, 0, 10), "  ████    ");
    assert_eq!(buf.cell((0, 0)).unwrap().bg, Color::Reset);
}

#[test]
fn stacked_links_keep_their_bars_aligned_in_columns() {
    // Four V100s in 118 x 12: two columns (GPU0-1, GPU2-3) of four titled links.
    let links = app_with(ac922_4, 2).links();
    let buf = stack(&links, 118, 12, ColorDepth::Ansi16);
    let rows: Vec<String> = (0..12).map(|y| row_text(&buf, y, 0, 118)).collect();
    let all = rows.join("\n");
    for (y, name) in [
        (0, "GPU0 NVLink CPU"),
        (3, "GPU0 NVLink peer"),
        (6, "GPU1 NVLink CPU"),
    ] {
        assert!(rows[y].starts_with(name), "{all}");
    }
    assert!(
        row_text(&buf, 0, 59, 118).starts_with("GPU2 NVLink CPU"),
        "{all}"
    );
    assert!(
        row_text(&buf, 9, 59, 118).starts_with("GPU3 NVLink peer"),
        "{all}"
    );
    // Both columns take the forms that fit the narrower one (58 cells): no link detail.
    assert_eq!(
        row_text(&buf, 0, 0, 58).trim_end(),
        "GPU0 NVLink CPU  / 75.0 GB/s  Legion 5.00"
    );
    assert_eq!(
        row_text(&buf, 0, 59, 118).trim_end(),
        "GPU2 NVLink CPU  / 75.0 GB/s  Legion 5.00"
    );
    assert!(!all.contains("3 links"), "{all}");
    assert!(rows[1].contains("8.00 GB/s  11%  pk 8.00"), "{all}");
    // Every bar row starts with its tag and its bar in the column's same cell.
    for y in [1u16, 2, 4, 5, 7, 8, 10, 11] {
        let tag = if y % 3 == 1 { "RX" } else { "TX" };
        for x0 in [0u16, 59] {
            assert_eq!(row_text(&buf, y, x0, x0 + 2), tag, "{all}");
            let c = buf.cell((x0 + 3, y)).unwrap();
            assert_ne!(c.bg, Color::Reset, "{x0},{y}\n{all}");
        }
    }
    // The blank cell between the columns.
    assert_eq!(buf.cell((58, 1)).unwrap().bg, Color::Reset);
    // Without room for titles: short names in the rows, still aligned.
    let buf = stack(&links, 38, 14, ColorDepth::Ansi16);
    let rows: Vec<String> = (0..14).map(|y| row_text(&buf, y, 0, 38)).collect();
    let all = rows.join("\n");
    assert!(rows[0].starts_with("G0 CPU RX"), "{all}");
    assert!(rows[1].starts_with("G0 CPU TX"), "{all}");
    assert!(rows[3].starts_with("G0 peer TX"), "{all}");
    assert!(rows[7].starts_with("G1 peer TX"), "{all}");
    assert!(row_text(&buf, 0, 19, 38).starts_with("G2 CPU RX"), "{all}");
    assert!(row_text(&buf, 7, 19, 38).starts_with("G3 peer TX"), "{all}");
    // The bars start after the widest tag (`G0 peer RX` and a blank).
    for y in 0..8 {
        assert_ne!(buf.cell((11, y)).unwrap().bg, Color::Reset, "{y}\n{all}");
        assert_eq!(buf.cell((10, y)).unwrap().bg, Color::Reset, "{y}\n{all}");
    }
    assert!(!all.contains("NVLink"), "{all}");
    // Seven-cell bars: the value alone (`G0 CPU RX  ▊8.00  `).
    assert_eq!(row_text(&buf, 0, 0, 18), "G0 CPU RX  ▊8.00  ", "{all}");
    assert_eq!(row_text(&buf, 1, 0, 18), "G0 CPU TX  ▏2.00  ", "{all}");
    assert!(!all.contains("pk"), "{all}");
}

// ---- snapshots and rates ----

fn link(
    kind: LinkKind,
    full: Option<f64>,
    tx: Option<f64>,
    rx: Option<f64>,
    legion: f64,
) -> LinkSnap {
    LinkSnap {
        kind,
        detail: match kind {
            LinkKind::Pcie => "PCIe 4.0 x16".into(),
            LinkKind::C2c => String::new(),
            _ => "3 links NVLink 2".into(),
        },
        full_scale_gb_s: full,
        scale_source: "test".into(),
        tx_bytes: tx,
        rx_bytes: rx,
        legion_rx_bytes: legion,
        legion_tx_bytes: 0.0,
    }
}

fn header(model: &str, gpus: u32) -> Header {
    Header {
        model: model.into(),
        quant: quant_from_name(model),
        gpus,
        tp: 0,
        stages: gpus,
        devices: (0..gpus).map(|i| format!("CUDA{i}")).collect(),
        ctx_size: 131_072,
        max_seqs: 4,
        spec: "mtp".into(),
        spec_k: 3,
        kv_types: Some("q8_0/q8_0".into()),
        kv_device_tokens: 262_144,
        kv_host_tokens: 0,
        expert_mode: Some("resident".into()),
        version: "0.0.0-test".into(),
    }
}

fn engine(t: f64) -> EngineSnap {
    let n = t as u64;
    EngineSnap {
        running: 3,
        prefilling: 1,
        decoding: 2,
        waiting: 1,
        prompt_tokens: 4000 * n,
        generated_tokens: 50 * n,
        kv_blocks_used: 300,
        kv_blocks_total: 1000,
        prefix_cache_hit_tokens: 1000 * n,
        expert_cache_hits: 90 * n,
        expert_cache_misses: 10 * n,
        expert_cache_slots: 512,
        expert_cache_layers: 40,
        spec: SpecSnap {
            drafted: 30 * n,
            accepted: 20 * n,
            steps: 10 * n,
            mean_k: 2.8,
            by_position: vec![[10 * n, 9 * n], [10 * n, 7 * n], [10 * n, 4 * n]],
        },
        prefill: PhaseBytes {
            device: 1_000_000_000 * n,
            link: 500_000_000 * n,
            experts_streamed: 200_000_000 * n,
            seconds: t,
            steps: n,
            ..PhaseBytes::default()
        },
        step_phases: [
            ("loop", 20u64),
            ("plan", 30),
            ("pack", 10),
            ("exec", 9000),
            ("sample", 200),
            ("device", 700),
            ("finish", 40),
        ]
        .iter()
        .map(|(name, us)| PhaseTime {
            name: (*name).into(),
            us: us * 100 * n,
        })
        .collect(),
        steps: 100 * n,
        ..EngineSnap::default()
    }
}

fn requests() -> Vec<RequestRow> {
    vec![
        RequestRow {
            id: 7,
            state: "decode".into(),
            endpoint: "chat".into(),
            prompt_tokens: Some(1200),
            generated: 345,
            age_ms: 9000,
            ttft_ms: Some(420),
            tok_s: Some(24.5),
        },
        RequestRow {
            id: 8,
            state: "prefill".into(),
            endpoint: "completions".into(),
            prompt_tokens: Some(32_768),
            generated: 0,
            age_ms: 1500,
            ttft_ms: None,
            tok_s: None,
        },
        RequestRow {
            id: 9,
            state: "queued".into(),
            endpoint: "messages".into(),
            prompt_tokens: None,
            ..RequestRow::default()
        },
    ]
}

fn latency() -> Latency {
    Latency {
        ttft_p50_ms: Some(410.0),
        ttft_p95_ms: Some(1900.0),
        ttft_samples: 42,
        itl_p50_ms: Some(38.0),
        itl_p95_ms: Some(61.0),
        itl_samples: 9000,
    }
}

/// An AC922 with two V100s at `t` seconds: NVLink to the CPU and to the peer.
fn ac922(t: f64) -> Snapshot {
    let gpu = |index: u32| GpuSnap {
        index,
        name: "Tesla V100-SXM2-16GB".into(),
        sm_busy_pct: Some(87.0),
        mem_busy_pct: Some(55.0),
        vram_used: Some(14 << 30),
        vram_total: Some(16 << 30),
        power_w: Some(240.0),
        sm_clock_mhz: Some(1530),
        mem_clock_mhz: Some(877),
        unified: false,
        links: vec![
            link(
                LinkKind::NvlinkCpu,
                Some(75.0),
                Some(2e9 * t),
                Some(8e9 * t),
                5e9 * t,
            ),
            link(
                LinkKind::NvlinkPeer,
                Some(75.0),
                Some(1e9 * t),
                Some(1e9 * t),
                0.0,
            ),
        ],
        memory: MemorySnap {
            peak_gb_s: Some(898.0),
            peak_source: "NVML".into(),
            gpm_dram_pct: None,
            legion_bytes: (500e9 * t) as u64,
            legion_seconds: t,
        },
    };
    Snapshot {
        version: VERSION,
        uptime_s: t,
        time_ms: 1_700_000_000_000 + (t * 1e3) as u64,
        header: header("gemma-4-31b-it-qat-q4_0", 2),
        engine: engine(t),
        requests: requests(),
        latency: latency(),
        hw: HwStatus {
            sampling: true,
            reason: None,
        },
        gpus: vec![gpu(0), gpu(1)],
        network: None,
    }
}

/// An AC922 with four V100s: eight NVLink gauges.
fn ac922_4(t: f64) -> Snapshot {
    let mut s = ac922(t);
    s.header = header("Qwen3.8-Flash-Next-UD-Q4_K_XL", 4);
    for index in 2..4 {
        let mut g = s.gpus[0].clone();
        g.index = index;
        s.gpus.push(g);
    }
    s
}

/// A DGX Spark (GB10) at `t` seconds: unified memory, C2C without counters, GPM DRAM
/// bandwidth, and the TP exchange to a second Spark.
fn gb10(t: f64) -> Snapshot {
    let mut s = ac922(t);
    s.header = header("Qwen3.8-Flash-Next-NVFP4", 1);
    s.header.tp = 2;
    s.gpus = vec![GpuSnap {
        index: 0,
        name: "NVIDIA GB10".into(),
        sm_busy_pct: Some(93.0),
        mem_busy_pct: Some(70.0),
        vram_used: None,
        vram_total: None,
        power_w: Some(60.0),
        sm_clock_mhz: Some(2400),
        mem_clock_mhz: None,
        unified: true,
        links: vec![link(LinkKind::C2c, None, None, None, 1e9 * t)],
        memory: MemorySnap {
            peak_gb_s: Some(273.0),
            peak_source: "LPDDR5X spec (273 GB/s)".into(),
            gpm_dram_pct: Some(60.0),
            legion_bytes: (150e9 * t) as u64,
            legion_seconds: t,
        },
    }];
    s.network = Some(LinkSnap {
        kind: LinkKind::Network,
        detail: "rocep1s0f0 200 Gb/sec (4X HDR)".into(),
        full_scale_gb_s: Some(25.0),
        scale_source: "NIC link speed (sysfs)".into(),
        tx_bytes: None,
        rx_bytes: None,
        legion_rx_bytes: 1e9 * t,
        legion_tx_bytes: 1e9 * t,
    });
    s
}

/// An RTX 4090 on PCIe at `t` seconds.
fn pcie(t: f64) -> Snapshot {
    let mut s = ac922(t);
    s.header = header("Qwen3.8-27B-Q4_K_M.gguf", 1);
    s.header.spec = "none".into();
    s.header.expert_mode = None;
    s.engine.expert_cache_slots = 0;
    s.gpus.truncate(1);
    s.gpus[0].name = "NVIDIA GeForce RTX 4090".into();
    s.gpus[0].links = vec![link(
        LinkKind::Pcie,
        Some(31.504),
        Some(1e9 * t),
        Some(3e9 * t),
        2e9 * t,
    )];
    s
}

#[test]
fn rates_between_two_snapshots() {
    let r = Rates::between(&ac922(10.0), &ac922(12.0));
    assert!(close(r.seconds, 2.0));
    assert!(close(r.prefill_tok_s, 4000.0));
    assert!(close(r.decode_tok_s, 50.0));
    assert_eq!(r.accepted_per_step, Some(2.0));
    assert_eq!(r.expert_hit_rate, Some(0.9));
    assert!(close(r.expert_stream_gb_s, 0.2));
    assert!(close(r.prefix_hit_tok_s, 1000.0));
    let cpu = r.links[0][0];
    assert!(close(cpu.tx_gb_s.unwrap(), 2.0));
    assert!(close(cpu.rx_gb_s.unwrap(), 8.0));
    assert!(close(cpu.legion_gb_s, 5.0));
    let peer = r.links[1][1];
    assert!(close(peer.tx_gb_s.unwrap(), 1.0));
    assert!(close(r.memory[0].legion_gb_s, 500.0));
    assert_eq!(r.memory[0].measured_gb_s, None);
    let exec = r.step_us.iter().find(|(n, _)| n == "exec").unwrap().1;
    assert!(close(exec, 9000.0));
}

#[test]
fn gpm_memory_rate_is_utilization_times_peak() {
    let r = Rates::between(&gb10(1.0), &gb10(2.0));
    assert!(close(r.memory[0].measured_gb_s.unwrap(), 0.6 * 273.0));
    assert!(close(r.memory[0].legion_gb_s, 150.0));
    // C2C has no counter: no TX/RX rate, Legion's bytes only.
    assert_eq!(r.links[0][0].tx_gb_s, None);
    assert!(close(r.links[0][0].legion_gb_s, 1.0));
    assert!(close(r.network.unwrap().legion_gb_s, 2.0));
}

#[test]
fn pcie_rates_from_fake_counters() {
    let r = Rates::between(&pcie(100.0), &pcie(101.0));
    let l = r.links[0][0];
    assert!(close(l.tx_gb_s.unwrap(), 1.0));
    assert!(close(l.rx_gb_s.unwrap(), 3.0));
    assert!(close(l.legion_gb_s, 2.0));
}

#[test]
fn counters_going_backwards_read_zero() {
    let r = Rates::between(&ac922(20.0), &ac922(10.0));
    assert_eq!(r.prefill_tok_s, 0.0);
    assert_eq!(r.decode_tok_s, 0.0);
    let same = Rates::between(&ac922(5.0), &ac922(5.0));
    assert_eq!(same.decode_tok_s, 0.0);
    assert_eq!(same.accepted_per_step, None);
}

#[test]
fn a_link_that_changed_kind_starts_over() {
    let a = ac922(1.0);
    let mut b = ac922(2.0);
    b.gpus[0].links.swap(0, 1);
    let r = Rates::between(&a, &b);
    assert_eq!(r.links[0][0].tx_gb_s, Some(0.0));
    assert_eq!(r.links[0][0].legion_gb_s, 0.0);
}

#[test]
fn the_snapshot_survives_json() {
    for s in [ac922(3.5), gb10(7.0), pcie(1.0), Snapshot::default()] {
        let json = serde_json::to_string(&s).unwrap();
        let back: Snapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
    let json = serde_json::to_value(gb10(1.0)).unwrap();
    assert_eq!(json["version"], VERSION);
    assert_eq!(json["gpus"][0]["links"][0]["kind"], "c2c");
    assert_eq!(json["network"]["kind"], "network");
    assert!(json["gpus"][0]["vram_total"].is_null());
    assert_eq!(
        serde_json::to_value(LinkKind::NvlinkCpu).unwrap(),
        "nvlink_cpu"
    );
}

#[test]
fn quant_from_the_models_name() {
    assert_eq!(
        quant_from_name("/models/gemma-4-31b-it-qat-q4_0/model.gguf").as_deref(),
        Some("QAT Q4_0")
    );
    assert_eq!(
        quant_from_name("Qwen3.8-Flash-Next-NVFP4").as_deref(),
        Some("NVFP4")
    );
    assert_eq!(
        quant_from_name("GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00003.gguf").as_deref(),
        Some("UD-Q4_K_XL")
    );
    assert_eq!(quant_from_name("model-bf16.gguf").as_deref(), Some("BF16"));
    assert_eq!(quant_from_name("x/Qwen-FP8/").as_deref(), Some("FP8"));
    assert_eq!(quant_from_name("llama"), None);
    assert_eq!(quant_from_name(""), None);
    // A quant name inside a word is not one.
    assert_eq!(quant_from_name("QATAR-Q8_0X"), None);
}

#[test]
fn a_hugging_face_snapshot_shows_its_repository_not_its_hash() {
    let hash = "fc694b54fb0174e0913e6adf86691ef85a4ead47";
    let path = format!("/cache/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/{hash}");
    let name = display_model(hash, Some(&path));
    assert_eq!(name, "Qwen3.8-Flash-Next-NVFP4");
    assert_eq!(quant_from_name(&name).as_deref(), Some("NVFP4"));
    // A named model keeps its name; a hash without a cache path stays as it is.
    assert_eq!(
        display_model("Gemma-4 31B IT", Some(&path)),
        "Gemma-4 31B IT"
    );
    assert_eq!(display_model(hash, Some("/models/x.gguf")), hash);
    assert_eq!(display_model(hash, None), hash);
}

#[test]
fn the_draft_depth_says_default_without_spec_k() {
    assert_eq!(ui::spec_depth(3), "k3");
    assert_eq!(ui::spec_depth(0), "k default");
}

// ---- latency ----

#[test]
fn latency_buckets_are_log_spaced() {
    assert_eq!(bucket(0.0), 0);
    assert_eq!(bucket(f64::NAN), 0);
    assert_eq!(bucket(1e9), BUCKETS - 1);
    for s in [0.0005, 0.003, 0.04, 0.2, 1.7, 30.0] {
        let b = bucket(s);
        assert!(s < bucket_end(b) * (1.0 + 1e-9), "{s}");
        assert!(b == 0 || s >= bucket_end(b - 1) * (1.0 - 1e-9), "{s}");
    }
    // Four buckets per doubling: a bucket's end is within 19% of what fell in it.
    assert!(close(bucket_end(4) / bucket_end(0), 2.0));
}

#[test]
fn quantiles_from_counts() {
    let mut counts = [0u64; BUCKETS];
    assert_eq!(quantile(&counts, 0, 0.5), None);
    counts[bucket(0.010)] = 90;
    counts[bucket(1.0)] = 10;
    let p50 = quantile(&counts, 100, 0.5).unwrap();
    let p95 = quantile(&counts, 100, 0.95).unwrap();
    assert!((10.0..12.0).contains(&p50), "{p50}");
    assert!((1000.0..1190.0).contains(&p95), "{p95}");
}

#[test]
fn the_latency_window_forgets_after_five_minutes() {
    let start = Instant::now();
    let w = LatencyWindow::new(start);
    for _ in 0..20 {
        w.record(start, 0.05);
    }
    let at = |s: u64| start + Duration::from_secs(s);
    w.record(at(130), 0.5);
    let (p50, _, n) = w.percentiles(at(130));
    assert_eq!(n, 21);
    assert!(p50.unwrap() < 60.0);
    // Minute 0 leaves the window at minute 5.
    let (p50, p95, n) = w.percentiles(at(5 * 60 + 1));
    assert_eq!(n, 1);
    assert_eq!(p50, p95);
    assert!(p50.unwrap() > 500.0);
    // Minute 6 reuses minute 0's slot: the old counts are gone.
    w.record(at(6 * 60), 0.002);
    assert_eq!(w.percentiles(at(6 * 60)).2, 2);
    assert_eq!(w.percentiles(at(20 * 60)).2, 0);
}

// ---- logs ----

fn line(level: &str, message: &str) -> LogLine {
    LogLine {
        seq: 0,
        time_ms: 1_700_000_000_000,
        level: level.into(),
        target: "legion::server".into(),
        message: message.into(),
    }
}

#[test]
fn level_ranks() {
    assert_eq!(level_rank("ERROR"), 1);
    assert_eq!(level_rank("warn"), 2);
    assert_eq!(level_rank("INFO"), 3);
    assert_eq!(level_rank("DEBUG"), 4);
    assert_eq!(level_rank("TRACE"), 5);
}

#[test]
fn the_ring_numbers_filters_and_drops_the_oldest() {
    let r = LogRing::default();
    assert_eq!(r.last_seq(), 0);
    r.push(line("INFO", "a"));
    r.push(line("DEBUG", "b"));
    r.push(line("ERROR", "c"));
    assert_eq!(r.last_seq(), 3);
    let all = r.since(0, 5);
    assert_eq!(all.iter().map(|l| l.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(r.since(0, 3).len(), 2);
    assert_eq!(r.since(1, 5).len(), 2);
    assert_eq!(r.since(3, 5).len(), 0);
    for i in 0..CAPACITY + 10 {
        r.push(line("INFO", &i.to_string()));
    }
    let kept_lines = r.since(0, 5);
    assert_eq!(kept_lines.len(), CAPACITY);
    assert_eq!(kept_lines.last().unwrap().seq, (CAPACITY + 13) as u64);
    assert_eq!(kept_lines[0].seq, 14);
}

#[test]
fn only_legions_own_lines_below_info_enter_the_ring() {
    assert!(kept("legion", &Level::DEBUG));
    assert!(kept("legion::engine::lm", &Level::TRACE));
    assert!(kept("hyper::proto", &Level::INFO));
    assert!(kept("h2", &Level::WARN));
    // Third-party debug lines carry headers and bodies: never.
    assert!(!kept("hyper::proto::h1", &Level::DEBUG));
    assert!(!kept("rustls::conn", &Level::TRACE));
    assert!(!kept("legionnaire", &Level::DEBUG));
}

// ---- the client ----

#[test]
fn server_sent_events_split_on_blank_lines() {
    let mut buf = String::from("data: {\"a\":1}\n\n: keep-alive\n\ndata: x\ndata: y\n\ndata: part");
    let ev = take_events(&mut buf);
    assert_eq!(ev, vec!["{\"a\":1}".to_string(), "x\ny".to_string()]);
    assert_eq!(buf, "data: part");
    buf.push_str("ial\r\n\r\n");
    assert_eq!(take_events(&mut buf), vec!["partial".to_string()]);
    assert!(buf.is_empty());
    let mut empty = String::new();
    assert!(take_events(&mut empty).is_empty());
    let mut no_space = String::from("data:z\n\n");
    assert_eq!(take_events(&mut no_space), vec!["z".to_string()]);
}

#[test]
fn a_character_split_across_chunks_arrives_whole() {
    let text = "data: {\"m\":\"GB10 – μ\"}\n\n".as_bytes();
    let dash = text.iter().position(|&b| b == 0xE2).unwrap();
    let (mut buf, mut pending) = (String::new(), Vec::new());
    // Split inside the three-byte dash, then inside the two-byte mu.
    push_utf8(&mut buf, &mut pending, &text[..dash + 1]);
    assert_eq!(pending.len(), 1);
    let mu = text.iter().position(|&b| b == 0xCE).unwrap();
    push_utf8(&mut buf, &mut pending, &text[dash + 1..mu + 1]);
    push_utf8(&mut buf, &mut pending, &text[mu + 1..]);
    assert!(pending.is_empty());
    assert_eq!(
        take_events(&mut buf),
        vec!["{\"m\":\"GB10 – μ\"}".to_string()]
    );
    // Bytes that can never be UTF-8 are replaced, not held.
    let (mut buf, mut pending) = (String::new(), Vec::new());
    push_utf8(&mut buf, &mut pending, b"a\xFFb");
    assert_eq!(buf, "a\u{FFFD}b");
    assert!(pending.is_empty());
}

#[test]
fn urls_and_backoff() {
    assert_eq!(base_url("192.0.2.10:8080"), "http://192.0.2.10:8080");
    assert_eq!(
        base_url("https://192.0.2.10:8443/"),
        "https://192.0.2.10:8443"
    );
    assert_eq!(base_url(" http://127.0.0.1:8080 "), "http://127.0.0.1:8080");
    let mut d = Duration::from_millis(500);
    let mut seen = Vec::new();
    for _ in 0..8 {
        d = next_backoff(d);
        seen.push(d.as_millis());
    }
    assert_eq!(
        seen,
        vec![1000, 2000, 4000, 8000, 10_000, 10_000, 10_000, 10_000]
    );
}

#[test]
fn the_feed_keeps_the_newest_snapshots_and_resets_logs() {
    let feed = Feed::default();
    for t in 0..20 {
        feed.snapshot(ac922(f64::from(t)));
    }
    feed.logs(vec![line("INFO", "old")], false);
    feed.logs(vec![line("INFO", "new")], true);
    feed.conn(Conn::Connected);
    let inbox: Inbox = feed.take();
    assert_eq!(inbox.snaps.len(), 8);
    assert_eq!(inbox.snaps.last().unwrap().uptime_s, 19.0);
    assert!(inbox.logs_reset);
    assert_eq!(inbox.logs.len(), 1);
    assert_eq!(inbox.conn, Some(Conn::Connected));
    let mut app = App::new("http://192.0.2.1:8080", Conn::Connecting);
    app.push_logs(vec![line("INFO", "stale")], false);
    app.ingest(inbox);
    assert_eq!(app.conn, Conn::Connected);
    assert_eq!(app.logs.len(), 1);
    assert_eq!(app.logs[0].message, "new");
    assert!(app.rates.is_some());
    // Taken: empty until something new arrives.
    let again = feed.take();
    assert!(again.snaps.is_empty() && again.conn.is_none() && !again.logs_reset);
}

// ---- the app ----

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn app_with(snaps: impl Fn(f64) -> Snapshot, seconds: u32) -> App {
    let mut app = App::new("http://192.0.2.1:8080", Conn::Connected);
    for i in 0..=seconds * 4 {
        app.push_snapshot(snaps(f64::from(i) * 0.25));
    }
    app.push_logs(
        vec![
            line("INFO", "listening"),
            line("WARN", "a long prompt waits for KV blocks"),
            line("ERROR", "request failed: test"),
            line("DEBUG", "admitted sequence"),
        ],
        false,
    );
    app
}

#[test]
fn history_takes_one_point_per_second() {
    let app = app_with(ac922, 30);
    assert_eq!(app.history.len(), 30);
    let p = app.history.back().unwrap();
    assert!(close(p.prefill, 4000.0));
    assert!(close(p.decode, 50.0));
    assert_eq!(p.accepted, Some(2.0));
    let (minute, ten) = app.series(|p| p.decode);
    assert_eq!(minute.len(), 30);
    assert_eq!(ten.len(), 3);
    assert!(minute.iter().all(|&v| v == 50));
}

#[test]
fn chunked_prefill_reads_its_rate_between_chunks_not_zero_or_spikes() {
    // A 32K prompt in 2,048-token chunks, one per 2.5 s step; snapshots every 250 ms.
    let at = |t: f64, prefilling: u32| {
        let mut s = ac922(t);
        let chunks = (t / 2.5).floor();
        s.engine.prompt_tokens = 2048 * chunks as u64;
        s.engine.prefill.seconds = 2.5 * chunks;
        s.engine.prefilling = prefilling;
        s
    };
    let mut app = App::new("x", Conn::Local);
    app.push_snapshot(at(0.0, 1));
    assert_eq!(app.prefill.tok_s, 0.0);
    let mut seen = Vec::new();
    for i in 1..=40 {
        app.push_snapshot(at(f64::from(i) * 0.25, 1));
        seen.push(app.prefill.tok_s);
    }
    // 0 until the first chunk lands, then 2048 / 2.5 s in every frame.
    assert!(seen[..9].iter().all(|&v| v == 0.0));
    assert!(seen[9..].iter().all(|&v| close(v, 819.2)), "{seen:?}");
    // The prompt is done: no prefill in flight reads 0.
    app.push_snapshot(at(10.25, 0));
    assert_eq!(app.prefill.tok_s, 0.0);
}

#[test]
fn the_first_snapshot_shows_no_prefill_rate_from_the_servers_lifetime() {
    let mut app = App::new("x", Conn::Local);
    let mut s = ac922(100.0);
    s.engine.prefilling = 0;
    app.push_snapshot(s);
    assert_eq!(app.prefill.tok_s, 0.0);
}

#[test]
fn history_is_capped_and_a_restart_clears_it() {
    let mut app = App::new("x", Conn::Local);
    for i in 0..=700 {
        app.push_snapshot(ac922(f64::from(i)));
    }
    assert_eq!(app.history.len(), ui::HISTORY);
    let (minute, ten) = app.series(|p| p.prefill);
    assert_eq!(minute.len(), 60);
    assert_eq!(ten.len(), 60);
    app.push_snapshot(ac922(1.0));
    assert!(app.history.is_empty());
    assert!(app.rates.is_none());
}

#[test]
fn peaks_hold_then_fall() {
    let mut app = App::new("x", Conn::Local);
    let at = |t: f64, tx: f64, rx: f64| {
        let mut s = pcie(t);
        s.gpus[0].links[0].tx_bytes = Some(tx);
        s.gpus[0].links[0].rx_bytes = Some(rx);
        s.gpus[0].sm_busy_pct = Some(if t < 1.5 { 99.0 } else { 20.0 });
        s.gpus[0].power_w = Some(if t < 1.5 { 300.0 } else { 90.0 });
        s
    };
    // TX: 0 -> 10 GB in one second, then flat; RX: 4 GB/s throughout. Each its own peak.
    app.push_snapshot(at(0.0, 0.0, 0.0));
    app.push_snapshot(at(1.0, 10e9, 4e9));
    app.push_snapshot(at(2.0, 10e9, 8e9));
    let links = app.links();
    assert!(close(links[0].tx.unwrap(), 0.0));
    assert!(close(links[0].tx_peak.unwrap(), 10.0));
    assert!(close(links[0].rx.unwrap(), 4.0));
    assert!(close(links[0].rx_peak.unwrap(), 4.0));
    assert!(close(links[0].legion_peak.unwrap(), 2.0));
    assert_eq!(app.peak(PeakOf::Sm(0)), Some(99.0));
    assert_eq!(app.peak(PeakOf::Power(0)), Some(300.0));
    assert!(close(app.peak(PeakOf::Decode).unwrap(), 50.0));
    // Still within the hold: the peaks stay.
    app.push_snapshot(at(2.0 + ui::PEAK_HOLD_S - 1.0, 10e9, 8e9 + 4e9 * 9.0));
    assert!(close(app.links()[0].tx_peak.unwrap(), 10.0));
    assert_eq!(app.peak(PeakOf::Sm(0)), Some(99.0));
    // Past it: the current value.
    app.push_snapshot(at(2.0 + ui::PEAK_HOLD_S + 1.0, 10e9, 8e9 + 4e9 * 11.0));
    let links = app.links();
    assert!(close(links[0].tx_peak.unwrap(), 0.0));
    assert!(close(links[0].rx_peak.unwrap(), 4.0));
    assert_eq!(app.peak(PeakOf::Sm(0)), Some(20.0));
    assert_eq!(app.peak(PeakOf::Power(0)), Some(90.0));
    // A restart clears them (the first snapshot after it sets SM and power again).
    app.push_snapshot(at(1.6, 0.0, 0.0));
    assert_eq!(app.peak(PeakOf::Sm(0)), Some(20.0));
    assert_eq!(app.links()[0].tx_peak, None);
    assert_eq!(app.peak(PeakOf::Decode), None);
}

#[test]
fn link_gauges_and_memory_dials_per_machine() {
    let app = app_with(ac922, 2);
    let links = app.links();
    let names: Vec<&str> = links.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "GPU0 NVLink CPU",
            "GPU0 NVLink peer",
            "GPU1 NVLink CPU",
            "GPU1 NVLink peer"
        ]
    );
    let short: Vec<&str> = links.iter().map(|d| d.short.as_str()).collect();
    assert_eq!(short, ["G0 CPU", "G0 peer", "G1 CPU", "G1 peer"]);
    let gpus: Vec<Option<u32>> = links.iter().map(|d| d.gpu).collect();
    assert_eq!(gpus, [Some(0), Some(0), Some(1), Some(1)]);
    assert_eq!(links[0].kind, LinkKind::NvlinkCpu);
    assert!(close(links[0].rx.unwrap(), 8.0));
    assert!(close(links[0].tx.unwrap(), 2.0));
    assert!(close(links[0].ratio(links[0].rx), 8.0 / 75.0));
    // Steady traffic: each peak is its own direction's value.
    assert!(close(links[0].rx_peak.unwrap(), 8.0));
    assert!(close(links[0].tx_peak.unwrap(), 2.0));
    assert!(links[0].legion.is_some());
    assert!(links[1].legion.is_none());
    assert_eq!(links[0].why, None);
    assert_eq!(links[0].detail.as_deref(), Some("3 links NVLink 2"));
    // V100: no GPM and not unified: no memory dial.
    assert!(app.memory_dials().is_empty());
    assert_eq!(app_with(ac922_4, 2).links().len(), 8);

    let app = app_with(gb10, 2);
    let links = app.links();
    assert_eq!(links[0].name, "GPU0 C2C");
    assert_eq!(links[0].tx, None);
    assert_eq!(links[0].why.as_deref(), Some("no C2C counter"));
    assert!(close(links[0].legion.unwrap(), 1.0));
    assert_eq!(links[1].name, "TP net");
    assert_eq!((links[1].kind, links[1].gpu), (LinkKind::Network, None));
    assert_eq!(links[1].why.as_deref(), Some("no NIC counter"));
    assert_eq!(links[1].full_scale, Some(25.0));
    // No counter: the bar shows Legion's bytes (2 GB/s of the NIC's 25).
    assert!(close(links[1].ratio(links[1].legion), 2.0 / 25.0));
    assert!(close(links[1].legion_peak.unwrap(), 2.0));
    assert_eq!((links[1].rx_peak, links[1].tx_peak), (None, None));
    let memory = app.memory_dials();
    assert_eq!(memory.len(), 1);
    assert_eq!(memory[0].title, "GPU0 memory (GPM)");
    assert!(close(memory[0].measured.unwrap(), 163.8));
    // The dial's peak follows the measured figure.
    assert!(close(memory[0].peak.unwrap(), 163.8));

    let app = app_with(pcie, 2);
    let links = app.links();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].name, "GPU0 PCIe");
    assert_eq!(links[0].short, "G0 PCIe");
    assert!(close(links[0].tx.unwrap(), 1.0));
    assert!(close(links[0].rx.unwrap(), 3.0));
    assert!(close(links[0].legion.unwrap(), 2.0));
    assert_eq!(links[0].full_scale, Some(31.504));
    assert!(app.memory_dials().is_empty());
}

#[test]
fn the_link_row_follows_the_link_count() {
    // The links' plan + borders; a memory dial keeps its 12 rows. At 200 x 60 and 120 x 46
    // the row may take 28 and 14 rows.
    let h = |app: &App, w, max| ui::bandwidth_height(app, w, max);
    assert_eq!(h(&app_with(pcie, 2), 200, 28), 5);
    assert_eq!(h(&app_with(ac922, 2), 200, 28), 8);
    assert_eq!(h(&app_with(ac922_4, 2), 200, 28), 14);
    assert_eq!(h(&app_with(ac922_4, 2), 120, 14), 14);
    assert_eq!(h(&app_with(gb10, 2), 200, 28), ui::DIAL_H);
    // Short of rows: the titles go.
    assert_eq!(h(&app_with(ac922_4, 2), 120, 10), 10);
}

#[test]
fn a_link_without_counters_says_so() {
    let mut app = App::new("x", Conn::Local);
    let at = |t: f64| {
        let mut s = pcie(t);
        s.gpus[0].links[0].tx_bytes = None;
        s.gpus[0].links[0].rx_bytes = None;
        s
    };
    app.push_snapshot(at(0.0));
    app.push_snapshot(at(1.0));
    let links = app.links();
    assert_eq!(links[0].tx, None);
    assert_eq!(links[0].rx, None);
    assert_eq!((links[0].rx_peak, links[0].tx_peak), (None, None));
    assert_eq!(links[0].why.as_deref(), Some("no NVML counter"));
    assert_eq!(
        links[0].rows()[1],
        BarRow::Note("RX/TX n/a (no NVML counter)".into())
    );
    let mut app = app;
    app.focus = Panel::Links;
    let s = render(&app, 80, 24);
    assert!(s.contains("RX/TX n/a (no NVML counter)"), "{s}");
}

#[test]
fn keys() {
    let mut app = app_with(ac922, 2);
    assert_eq!(app.handle_key(key(KeyCode::Char('q'))), Action::Quit);
    assert_eq!(
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Action::Interrupt
    );
    // Tab cycles every panel and comes back.
    for p in PANELS.iter().skip(1) {
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.focus, *p);
    }
    app.handle_key(key(KeyCode::Tab));
    assert_eq!(app.focus, Panel::Requests);
    app.handle_key(key(KeyCode::BackTab));
    assert_eq!(app.focus, Panel::Logs);
    app.handle_key(key(KeyCode::Char('?')));
    assert!(app.help);
    app.handle_key(key(KeyCode::Esc));
    assert!(!app.help);
    // p pauses: snapshots are ignored, and rates start over after it.
    app.handle_key(key(KeyCode::Char('p')));
    assert!(app.paused);
    let before = app.snap.clone();
    app.push_snapshot(ac922(99.0));
    assert_eq!(app.snap, before);
    app.handle_key(key(KeyCode::Char('p')));
    assert!(!app.paused && app.snap.is_none() && app.rates.is_none());
    app.handle_key(key(KeyCode::Char('l')));
    assert!(app.full_logs);
}

#[test]
fn log_levels_and_search() {
    let mut app = app_with(ac922, 1);
    // INFO and up by default.
    assert_eq!(app.visible_logs().len(), 3);
    app.handle_key(key(KeyCode::Char('1')));
    assert_eq!(app.visible_logs().len(), 1);
    app.handle_key(key(KeyCode::Char('5')));
    assert_eq!(app.visible_logs().len(), 4);
    app.handle_key(key(KeyCode::Char('/')));
    assert!(app.searching);
    for c in "KV".chars() {
        // Typing `q` or a digit while searching is text, not a key.
        assert_eq!(app.handle_key(key(KeyCode::Char(c))), Action::None);
    }
    assert_eq!(app.handle_key(key(KeyCode::Char('q'))), Action::None);
    app.handle_key(key(KeyCode::Backspace));
    app.handle_key(key(KeyCode::Enter));
    assert!(!app.searching);
    assert_eq!(app.search, "KV");
    let v = app.visible_logs();
    assert_eq!(v.len(), 1);
    assert!(v[0].message.contains("KV blocks"));
    app.handle_key(key(KeyCode::Esc));
    assert!(app.search.is_empty());
}

#[test]
fn scrolled_logs_stay_put() {
    let mut app = app_with(ac922, 1);
    app.handle_key(key(KeyCode::Up));
    assert_eq!(app.scroll, 1);
    app.push_logs(vec![line("INFO", "more")], false);
    assert_eq!(app.scroll, 2);
    // A hidden level does not move the view.
    app.push_logs(vec![line("TRACE", "hidden")], false);
    assert_eq!(app.scroll, 2);
    app.handle_key(key(KeyCode::End));
    assert_eq!(app.scroll, 0);
    app.push_logs(Vec::new(), true);
    assert!(app.logs.is_empty());
}

#[test]
fn layout_modes_by_size() {
    let m = |w, h| Mode::of(Rect::new(0, 0, w, h));
    assert_eq!(
        m(80, 24),
        Mode {
            full: false,
            sparklines: false,
            step_bar: false
        }
    );
    assert!(m(80, 30).step_bar);
    assert!(m(100, 40).sparklines && !m(100, 40).full);
    assert!(m(120, 46).full);
    assert!(m(200, 60).full);
    assert!(!m(119, 60).full);
}

#[test]
fn step_parts_group_the_phases() {
    let us: Vec<(String, f64)> = [
        ("loop", 2.0),
        ("plan", 3.0),
        ("pack", 1.0),
        ("exec", 90.0),
        ("sample", 20.0),
        ("device", 7.0),
        ("finish", 4.0),
    ]
    .iter()
    .map(|(n, v)| ((*n).to_string(), *v))
    .collect();
    let parts = ui::step_parts(&us);
    let get = |name: &str| parts.iter().find(|p| p.0 == name).unwrap().1;
    assert_eq!(get("step"), 90.0);
    assert_eq!(get("draft+verify"), 7.0);
    assert_eq!(get("sampling"), 20.0);
    assert_eq!(get("host gaps"), 10.0);
    assert!(ui::step_parts(&[]).iter().all(|p| p.1 == 0.0));
}

#[test]
fn small_formatters() {
    assert_eq!(ui::duration(5.0), "5s");
    assert_eq!(ui::duration(185.0), "3m05s");
    assert_eq!(ui::duration(3720.0), "1h02m");
    assert_eq!(ui::duration(187_200.0), "2d04h");
    assert_eq!(ui::ms(None), "n/a");
    assert_eq!(ui::ms(Some(4.3)), "4.3 ms");
    assert_eq!(ui::ms(Some(850.0)), "850 ms");
    assert_eq!(ui::ms(Some(1240.0)), "1.24 s");
    assert_eq!(ui::clock(3_723_000), "01:02:03");
}

// ---- drawing ----

fn render(app: &App, w: u16, h: u16) -> String {
    let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
    t.draw(|f| ui::draw(f, app)).unwrap();
    screen(t.backend())
}

fn screen(b: &TestBackend) -> String {
    let buf = b.buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf.cell((x, y)).map_or(" ", |c| c.symbol()));
        }
        out.push('\n');
    }
    out
}

const SIZES: [(u16, u16); 3] = [(80, 24), (120, 46), (200, 60)];

/// The cell at `(x, y)` of a drawn screen.
fn cell_at(t: &Terminal<TestBackend>, x: u16, y: u16) -> ratatui::buffer::Cell {
    t.backend().buffer().cell((x, y)).unwrap().clone()
}

/// Where `want` starts on the drawn screen (column, row), by characters.
fn find(s: &str, want: &str) -> Option<(u16, u16)> {
    s.lines().enumerate().find_map(|(y, l)| {
        l.find(want)
            .map(|i| (l[..i].chars().count() as u16, y as u16))
    })
}

#[test]
fn the_full_layout_at_200_by_60() {
    let app = app_with(ac922, 3);
    let mut t = Terminal::new(TestBackend::new(200, 60)).unwrap();
    t.draw(|f| ui::draw(f, &app)).unwrap();
    let s = screen(t.backend());
    for want in [
        "gemma-4-31b-it-qat-q4_0",
        "QAT Q4_0",
        "Requests (2 active, 1 queued)",
        "Throughput",
        "tok/s  pk ",
        "per request",
        "Latency (5 min)",
        "Links",
        "GPU0 NVLink CPU  / 75.0 GB/s  Legion 5.00  3 links NVLink 2",
        "GPU1 NVLink peer  / 75.0 GB/s  3 links NVLink 2",
        "8.00 GB/s  11%  pk 8.00",
        "2.00 GB/s  3%  pk 2.00",
        "1.00 GB/s  1%  pk 1.00",
        "GPUs",
        "87%  87  55%",
        "240  240",
        "SM MHz",
        "Speculative",
        "KV cache",
        "MoE experts",
        "Step (engine thread, us/step)",
        "Logs [INFO and up]",
        "q quit",
        "free",
        "decode",
        "prefill",
        "queued",
        "dec 60s",
        "mtp depth k3",
    ] {
        assert!(s.contains(want), "missing {want:?} in\n{s}");
    }
    // The ERROR and WARN lines show at INFO, DEBUG does not.
    assert!(s.contains("request failed: test"));
    assert!(!s.contains("admitted sequence"));
    // A GPU's two links in one column, the second GPU's next to it; colours by link type,
    // direction and load (16 colours before a terminal says more).
    let (x, y) = find(&s, "GPU0 NVLink CPU").unwrap();
    assert_eq!(find(&s, "GPU0 NVLink peer"), Some((x, y + 3)), "{s}");
    let (x1, y1) = find(&s, "GPU1 NVLink CPU").unwrap();
    assert!(y1 == y && x1 > x + 90, "{s}");
    assert_eq!(cell_at(&t, x, y).fg, kind_color(LinkKind::NvlinkCpu));
    assert_eq!(cell_at(&t, x, y + 1).fg, gauge::RX);
    assert_eq!(cell_at(&t, x, y + 2).fg, gauge::TX);
    let bar = cell_at(&t, x + 3, y + 1);
    assert_eq!(
        (bar.fg, bar.bg),
        (Color::Green, empty_color(ColorDepth::Ansi16))
    );
    assert_eq!(cell_at(&t, x + 60, y + 1).bg, Color::DarkGray);
}

#[test]
fn the_full_layout_at_120_by_46_holds_four_gpus() {
    let app = app_with(ac922_4, 3);
    let s = render(&app, 120, 46);
    assert_eq!(s.lines().count(), 46);
    for g in 0..4 {
        for want in [
            format!("GPU{g} NVLink CPU  / 75.0 GB/s  Legion 5.00"),
            format!("GPU{g} NVLink peer  / 75.0 GB/s"),
        ] {
            assert!(s.contains(&want), "missing {want:?} in\n{s}");
        }
    }
    // Two columns of four, titled; the details go.
    let (x0, y0) = find(&s, "GPU0 NVLink CPU").unwrap();
    assert_eq!(find(&s, "GPU1 NVLink peer"), Some((x0, y0 + 9)), "{s}");
    let (x2, y2) = find(&s, "GPU2 NVLink CPU").unwrap();
    assert!(y2 == y0 && x2 > x0 + 50, "{s}");
    assert!(!s.contains("3 links"), "{s}");
    assert_eq!(s.matches("8.00 GB/s  11%  pk 8.00").count(), 4, "{s}");
    assert_eq!(s.matches("1.00 GB/s  1%  pk 1.00").count(), 8, "{s}");
    // The rest of the layout still fits: GPUs (without the clocks), logs, footer.
    for want in [
        "GPUs",
        "Speculative",
        "Logs [INFO and up]",
        "q quit",
        "pk",
        "VRAM GiB",
    ] {
        assert!(s.contains(want), "missing {want:?} in\n{s}");
    }
    assert!(!s.contains("SM MHz"), "{s}");
}

#[test]
fn the_compact_layout_at_80_by_24() {
    let app = app_with(ac922, 3);
    let s = render(&app, 80, 24);
    for want in [
        "gemma-4-31b-it-qat-q4_0",
        "Requests",
        "Throughput",
        "Logs",
        "q quit",
        "decode",
    ] {
        assert!(s.contains(want), "missing {want:?} in\n{s}");
    }
    // Every line fits: nothing is cut mid-frame by a panic or an overflow.
    assert_eq!(s.lines().count(), 24);
    assert!(s.lines().all(|l| l.chars().count() == 80));
}

#[test]
fn every_link_gauge_fits_at_every_size() {
    // Four V100s at 80 x 24: eight links in two columns, two rows each with a short name.
    let mut app = app_with(ac922_4, 3);
    app.focus = Panel::Links;
    let mut t = Terminal::new(TestBackend::new(80, 24)).unwrap();
    t.draw(|f| ui::draw(f, &app)).unwrap();
    let s = screen(t.backend());
    for g in 0..4 {
        for want in [
            format!("G{g} CPU RX "),
            format!("G{g} CPU TX "),
            format!("G{g} peer RX "),
            format!("G{g} peer TX "),
        ] {
            assert_eq!(s.matches(&want).count(), 1, "missing {want:?} in\n{s}");
        }
    }
    // Bars of seven cells (left) and eight (right): the value alone, no `pk`.
    assert!(s.contains("G0 CPU RX  ▊8.00   G2 CPU RX  ▉ 8.00  "), "{s}");
    assert!(s.contains("G1 peer TX ▏1.00   G3 peer TX ▏ 1.00  "), "{s}");
    assert_eq!(s.lines().count(), 24);
    assert!(s.lines().all(|l| l.chars().count() == 80));
    // The short names in the link type's colour, the tags in the direction's.
    let (x, y) = find(&s, "G0 CPU RX").unwrap();
    assert_eq!(cell_at(&t, x, y).fg, kind_color(LinkKind::NvlinkCpu));
    assert_eq!(cell_at(&t, x + 7, y).fg, gauge::RX);
    assert_eq!(cell_at(&t, x + 7, y + 1).fg, gauge::TX);
    // The value over the empty part, in the fill's colour.
    let v = cell_at(&t, x + 12, y);
    assert_eq!(
        (v.symbol(), v.fg, v.bg),
        ("8", Color::Green, Color::DarkGray)
    );
    for (w, h) in [(120, 46), (200, 60)] {
        let s = render(&app, w, h);
        for g in 0..4 {
            assert!(
                s.contains(&format!("GPU{g} NVLink CPU  / 75.0 GB/s")),
                "{s}"
            );
            assert!(
                s.contains(&format!("GPU{g} NVLink peer  / 75.0 GB/s")),
                "{s}"
            );
        }
        assert_eq!(s.matches("RX ").count(), 8, "{w}x{h}\n{s}");
    }
    // The two-GPU AC922 in the compact layout: titled, short.
    let mut app = app_with(ac922, 3);
    app.focus = Panel::Links;
    let s = render(&app, 80, 24);
    assert!(s.contains("G0 CPU / 75.0 GB/s L 5.00"), "{s}");
    assert!(s.contains("G1 peer / 75.0 GB/s"), "{s}");
    assert!(s.contains("8.00 GB/s  11%  pk 8.00"), "{s}");
    // The GB10: Legion's bars and the reasons.
    let mut app = app_with(gb10, 3);
    app.focus = Panel::Links;
    let s = render(&app, 80, 24);
    // No full scale for the C2C fixture.
    assert!(s.contains("GPU0 C2C  full scale n/a  Legion 1.00"), "{s}");
    assert!(s.contains("TP net  / 25.0 GB/s  Legion 2.00"), "{s}");
    assert!(s.contains("1.00 GB/s  pk 1.00"), "{s}");
    assert!(s.contains("2.00 GB/s  8%  pk 2.00"), "{s}");
    assert!(s.contains("RX/TX n/a (no C2C counter)"), "{s}");
    assert!(s.contains("RX/TX n/a (no NIC counter)"), "{s}");
}

#[test]
fn every_panel_draws_in_the_compact_layout() {
    for snaps in [ac922 as fn(f64) -> Snapshot, gb10, pcie] {
        let mut app = app_with(snaps, 2);
        for p in PANELS {
            app.focus = p;
            for (w, h) in SIZES {
                let s = render(&app, w, h);
                assert_eq!(s.lines().count(), h as usize);
            }
            let s = render(&app, 80, 24);
            if !matches!(p, Panel::Requests | Panel::Logs) {
                assert!(
                    s.contains(p.title()) || s.contains("Speculative") || s.contains("Step"),
                    "{p:?}\n{s}"
                );
            }
        }
    }
}

#[test]
fn each_widget_draws_alone_at_every_size() {
    let app = app_with(gb10, 2);
    for (w, h) in SIZES {
        for p in PANELS {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| {
                let area = f.area();
                ui::draw_panel(f, area, &app, p, Mode::of(area));
            })
            .unwrap();
            let s = screen(t.backend());
            assert!(!s.trim().is_empty(), "{p:?} at {w}x{h}");
        }
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| {
            let area = f.area();
            ui::draw_bandwidth(f, area, &app);
        })
        .unwrap();
        let s = screen(t.backend());
        assert!(s.contains("GPU0 C2C"), "{w}x{h}\n{s}");
        assert!(s.contains("GPU0 memory (GPM)"), "{w}x{h}\n{s}");
        assert!(s.contains('█'), "{w}x{h}\n{s}");
        // The dial's peak: `pk` after its figures.
        assert!(s.contains("meas 164 L 150 pk 164"), "{w}x{h}\n{s}");
    }
}

#[test]
fn a_gb10_at_200_by_60_shows_c2c_memory_and_the_network() {
    let s = render(&app_with(gb10, 3), 200, 60);
    for want in [
        "Qwen3.8-Flash-Next-NVFP4",
        "NVFP4",
        "TP2",
        "GPU0 C2C  full scale n/a  Legion 1.00",
        "TP net  / 25.0 GB/s  Legion 2.00  rocep1s0f0 200 Gb/sec (4X HDR)",
        "RX/TX n/a (no C2C counter)",
        "RX/TX n/a (no NIC counter)",
        "1.00 GB/s  pk 1.00",
        "2.00 GB/s  8%  pk 2.00",
        "GPU0 memory (GPM)",
        "meas 164 L 150 pk 164",
        "unified",
    ] {
        assert!(s.contains(want), "missing {want:?} in\n{s}");
    }
    // C2C and TP net side by side, the title in each link type's colour.
    let mut t = Terminal::new(TestBackend::new(200, 60)).unwrap();
    let app = app_with(gb10, 3);
    t.draw(|f| ui::draw(f, &app)).unwrap();
    let (x, y) = find(&s, "GPU0 C2C").unwrap();
    let (xn, yn) = find(&s, "TP net  /").unwrap();
    assert!(yn == y && xn > x + 60, "{s}");
    assert_eq!(cell_at(&t, x, y).fg, kind_color(LinkKind::C2c));
    assert_eq!(cell_at(&t, xn, yn).fg, kind_color(LinkKind::Network));
    // Legion's bar: `L` in yellow; the C2C has no full scale, so its value is grey.
    assert_eq!(cell_at(&t, x, y + 1).fg, gauge::LEGION);
    assert_eq!(cell_at(&t, xn + 2, yn + 1).fg, Color::Green);
    let (xv, yv) = find(&s, "1.00 GB/s  pk 1.00").unwrap();
    assert_eq!(yv, y + 1);
    assert_eq!(cell_at(&t, xv, yv).fg, Color::Gray);
}

#[test]
fn a_pcie_card_at_every_size() {
    let app = app_with(pcie, 3);
    for (w, h) in [(200, 60), (120, 46)] {
        let s = render(&app, w, h);
        for want in [
            "GPU0 PCIe  / 31.5 GB/s  Legion 2.00  PCIe 4.0 x16",
            "3.00 GB/s  10%  pk 3.00",
            "1.00 GB/s  3%  pk 1.00",
            "off (no --spec)",
            "dense model",
        ] {
            assert!(s.contains(want), "missing {want:?} at {w}x{h} in\n{s}");
        }
    }
    let mut app = app;
    app.focus = Panel::Links;
    let mut t = Terminal::new(TestBackend::new(80, 24)).unwrap();
    t.draw(|f| ui::draw(f, &app)).unwrap();
    let s = screen(t.backend());
    assert!(s.contains("GPU0 PCIe  / 31.5 GB/s  Legion 2.00"), "{s}");
    assert!(!s.contains("PCIe 4.0 x16"), "{s}");
    assert!(s.contains("3.00 GB/s  10%  pk 3.00"), "{s}");
    let (x, y) = find(&s, "GPU0 PCIe").unwrap();
    assert_eq!(cell_at(&t, x, y).fg, kind_color(LinkKind::Pcie));
}

#[test]
fn before_the_first_snapshot_and_while_disconnected() {
    for (w, h) in SIZES {
        let app = App::new("http://192.0.2.1:8080", Conn::Connecting);
        let s = render(&app, w, h);
        assert!(s.contains("connecting to http://192.0.2.1:8080"), "{s}");
        assert!(s.contains("waiting for the first snapshot"));
        let mut app = app_with(ac922, 1);
        app.conn = Conn::Disconnected {
            error: "cannot connect".into(),
            retry_s: 2.0,
        };
        let s = render(&app, w, h);
        assert!(
            s.contains("disconnected: cannot connect (retry in 2s)"),
            "{s}"
        );
    }
}

#[test]
fn hardware_panels_say_why_they_are_empty() {
    let mut app = App::new("x", Conn::Local);
    let at = |t: f64| {
        let mut s = ac922(t);
        s.gpus.clear();
        s.hw = HwStatus {
            sampling: false,
            reason: Some("NVML: library not found".into()),
        };
        s
    };
    app.push_snapshot(at(0.0));
    app.push_snapshot(at(1.0));
    let s = render(&app, 200, 60);
    assert!(
        s.contains("hardware counters n/a: NVML: library not found"),
        "{s}"
    );
    app.focus = Panel::Gpu;
    let s = render(&app, 80, 24);
    assert!(s.contains("hardware counters n/a"), "{s}");
}

#[test]
fn help_and_full_screen_logs() {
    for (w, h) in SIZES {
        let mut app = app_with(ac922, 1);
        app.help = true;
        let s = render(&app, w, h);
        assert!(s.contains("Help"));
        assert!(s.contains("Keys"));
        app.help = false;
        app.full_logs = true;
        app.level = 5;
        let s = render(&app, w, h);
        assert!(s.contains("admitted sequence"), "{s}");
        assert!(!s.contains("Throughput"));
    }
    let mut app = app_with(ac922, 1);
    app.help = true;
    assert!(render(&app, 200, 60).contains("Why n/a"));
    assert!(ui::HELP.iter().any(|l| l.contains("no CPU DRAM PMU")));
}

#[test]
fn tiny_terminals_do_not_panic() {
    let app = app_with(gb10, 2);
    for (w, h) in [(1, 1), (10, 5), (40, 10), (59, 20), (80, 12), (300, 100)] {
        render(&app, w, h);
        let mut help = app_with(ac922, 1);
        help.help = true;
        render(&help, w, h);
    }
}

#[test]
fn a_memory_dial_draws_on_a_canvas_when_it_fits() {
    let d = Speedo {
        title: "GPU0 memory (GPM)".into(),
        full_scale: Some(273.0),
        measured: Some(160.0),
        legion: 120.0,
        note: None,
        peak: Some(250.0),
    };
    let draw = |d: &Speedo| {
        let mut t = Terminal::new(TestBackend::new(30, 11)).unwrap();
        t.draw(|f| {
            let area = f.area();
            d.render(f, area, true);
        })
        .unwrap();
        t
    };
    let t = draw(&d);
    let s = screen(t.backend());
    assert!(s.contains("GPU0 memory (GPM)"));
    assert!(s.contains("meas 160 L 120 pk 250"), "{s}");
    assert!(s.contains("/273 GB/s"), "{s}");
    // Braille dots: the arc and the needles.
    assert!(
        s.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)),
        "{s}"
    );
    // The peak tick, in its own colour on the arc; none without a peak.
    let peak_cells = |t: &Terminal<TestBackend>| {
        let buf = t.backend().buffer();
        (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .filter(|&p| {
                let c = buf.cell(p).unwrap();
                let braille = c
                    .symbol()
                    .chars()
                    .all(|ch| ('\u{2801}'..='\u{28ff}').contains(&ch));
                c.fg == super::speedo::PEAK && braille
            })
            .count()
    };
    assert!(peak_cells(&t) > 0, "{s}");
    let none = draw(&Speedo {
        peak: None,
        ..d.clone()
    });
    assert_eq!(peak_cells(&none), 0);
    assert!(!screen(none.backend()).contains("pk"));
    // Too small for a dial: the bars.
    let mut t = Terminal::new(TestBackend::new(14, 6)).unwrap();
    t.draw(|f| {
        let area = f.area();
        d.render(f, area, false);
    })
    .unwrap();
    let s = screen(t.backend());
    assert!(s.contains('█'), "{s}");
}
