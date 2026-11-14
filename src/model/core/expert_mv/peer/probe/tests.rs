use super::*;

fn waiting(us: f32) -> Sample {
    Sample {
        fork_wait: us / 2.0,
        join_wait: us / 2.0,
        fork_op: 60.0,
        join_op: 40.0,
    }
}

/// Feeds one decision's worth of samples; returns whether the share moved.
fn decide(c: &mut ShareControl, us: f32) -> bool {
    let mut moved = false;
    for _ in 0..DECIDE_EVERY {
        moved |= c.add(waiting(us));
    }
    moved
}

#[test]
fn starts_at_the_tuned_share() {
    assert_eq!(LEVELS[START], (1, 2));
    assert_eq!(LEVELS[START], super::super::SHARE);
    assert_eq!(ShareControl::new(true).share(), (1, 2));
    // The peer's share grows along the levels.
    for w in LEVELS.windows(2) {
        assert!(w[0].0 * w[1].1 < w[1].0 * w[0].1, "{w:?}");
    }
}

#[test]
fn moves_once_per_decision() {
    let mut c = ShareControl::new(true);
    for _ in 0..DECIDE_EVERY - 1 {
        assert!(!c.add(waiting(0.0)));
    }
    assert_eq!(c.level(), START);
    assert!(c.add(waiting(0.0)));
    assert_eq!(c.level(), START + 1);
}

#[test]
fn climbs_while_the_running_gpu_does_not_wait() {
    let mut c = ShareControl::new(true);
    for _ in 0..LEVELS.len() + 2 {
        decide(&mut c, 0.5);
    }
    assert_eq!(c.level(), LEVELS.len() - 1);
    assert_eq!(c.share(), (1, 1));
}

#[test]
fn stays_between_the_thresholds() {
    let mut c = ShareControl::new(true);
    for _ in 0..10 {
        assert!(!decide(&mut c, (WAIT_LOW_US + WAIT_HIGH_US) / 2.0));
    }
    assert_eq!(c.level(), START);
}

#[test]
fn backs_off_and_bars_a_share_that_waits() {
    let mut c = ShareControl::new(true);
    assert!(decide(&mut c, 0.0));
    assert_eq!(c.level(), START + 1);
    // The new share waits: back down, and it stays barred for BAR_FIRST decisions.
    assert!(decide(&mut c, 20.0));
    assert_eq!(c.level(), START);
    for _ in 0..BAR_FIRST - 1 {
        assert!(!decide(&mut c, 0.0));
        assert_eq!(c.level(), START);
    }
    assert!(decide(&mut c, 0.0));
    assert_eq!(c.level(), START + 1);
    // Barred again: twice as long.
    assert!(decide(&mut c, 20.0));
    let mut n = 0;
    while !decide(&mut c, 0.0) {
        n += 1;
        assert!(n < 1000);
    }
    assert_eq!(n, 2 * BAR_FIRST - 1);
}

#[test]
fn never_below_the_least_share() {
    let mut c = ShareControl::new(true);
    for _ in 0..10 {
        decide(&mut c, 50.0);
    }
    assert_eq!(c.level(), 0);
    assert_eq!(c.share(), LEVELS[0]);
}

#[test]
fn fixed_share_never_moves() {
    let mut c = ShareControl::new(false);
    for us in [0.0, 50.0, 0.0] {
        assert!(!decide(&mut c, us));
    }
    assert_eq!(c.share(), (1, 2));
    assert!(!auto_share(false, None));
    assert!(auto_share(false, Some("auto")));
    assert!(!auto_share(true, Some("auto")));
    assert!(!auto_share(false, Some("log")));
    assert!(!auto_share(false, Some("")));
}

#[test]
fn sums_mean_and_waited() {
    let mut s = Sums::default();
    assert_eq!(s.mean(1.0), 1.0);
    s.add(waiting(1.0));
    s.add(waiting(9.0));
    assert_eq!(s.n, 2);
    assert_eq!(s.mean(s.fork_wait + s.join_wait), 5.0);
    assert_eq!(s.waited, 1);
}
