use super::*;

#[test]
fn window_splits_a_step() {
    let a = ArStats {
        count: 100,
        wait_ns: -5_000,
        stall_ns: 1_000_000,
        first_stall_ns: 10_000,
        firsts: 1,
    };
    let b = ArStats {
        count: 100 + 4 * 120,
        wait_ns: -5_000 + 480 * 200_000,
        stall_ns: 1_000_000 + 4 * 30_000_000,
        first_stall_ns: 10_000 + 4 * 500_000,
        firsts: 5,
    };
    let w = Window::new(
        a,
        b,
        4,
        Duration::from_millis(4 * 100),
        Duration::from_micros(4 * 250),
    );
    assert_eq!(w.steps, 4);
    assert!((w.step_ms - 100.0).abs() < 1e-9);
    assert!((w.send_ms - 0.25).abs() < 1e-9);
    assert!((w.all_reduces - 120.0).abs() < 1e-9);
    assert!((w.stall_ms - 30.0).abs() < 1e-9);
    assert!((w.own_ms - 69.75).abs() < 1e-9);
    assert!((w.wait_us - 200.0).abs() < 1e-9);
    assert!((w.first_stall_us - 500.0).abs() < 1e-9);
}

#[test]
fn an_empty_window_is_zero() {
    let s = ArStats::default();
    let w = Window::new(s, s, 0, Duration::ZERO, Duration::ZERO);
    assert_eq!(w.step_ms, 0.0);
    assert_eq!(w.wait_us, 0.0);
    assert_eq!(w.first_stall_us, 0.0);
    assert_eq!(w.own_ms, 0.0);
}
