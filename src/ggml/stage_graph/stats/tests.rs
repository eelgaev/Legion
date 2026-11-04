use super::*;

fn t(runs: u64, captures: u64, builds: u64) -> Totals {
    Totals {
        runs,
        captures,
        capture_ns: captures * 1000,
        check_ns: runs * 50,
        builds,
        reuses: captures / 2,
        // Linear in `builds`: a window's counts are differences of totals, and `builds / 2`
        // rounded 3 -> 1 but 4 -> 2.
        precaptures: builds * 2,
        updates: captures,
        update_fails: builds,
        steals: runs,
        mtp_builds: runs / 5,
        rewords: builds,
        why: [captures, 0, 0, 0],
        split: [runs * 2, runs * 3, captures * 3, builds * 7, builds, runs],
    }
}

/// The window logs the counts of the last [`EVERY`] once it has passed, only when something
/// was captured or built, and starts again either way.
#[test]
fn window_reports_what_changed_once_due() {
    let mut w = Window::new();
    let t0 = Instant::now();
    assert_eq!(
        w.due(t0, t(5, 2, 1)),
        None,
        "the first call starts the window"
    );
    assert_eq!(w.due(t0 + EVERY / 2, t(9, 4, 1)), None, "not due yet");
    let (d, took) = w
        .due(t0 + EVERY, t(20, 6, 3))
        .expect("due, captures happened");
    assert_eq!(d, t(15, 4, 2));
    assert_eq!(took, EVERY);
    // Runs only: due, but nothing to log; the window still moves on.
    assert_eq!(w.due(t0 + 2 * EVERY, t(40, 6, 3)), None);
    let (d, _) = w
        .due(t0 + 3 * EVERY, t(41, 6, 4))
        .expect("a build is enough");
    assert_eq!(d, t(1, 0, 1));
}

/// #858: a shape's first build is first once per kind and stage.
#[test]
fn first_build_is_first_once_per_kind_and_stage() {
    let h = 0x05ee_d858;
    assert!(first_build("verify", 0, h));
    assert!(!first_build("verify", 0, h), "built before");
    assert!(first_build("verify", 1, h), "another stage");
    assert!(first_build("take-in", 0, h), "another kind");
}
