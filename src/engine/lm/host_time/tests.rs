use super::*;

fn clock() -> HostClock {
    HostClock {
        on: true,
        ..HostClock::default()
    }
}

#[test]
fn host_time_leaves_out_the_step_and_device_calls() {
    let mut c = clock();
    c.add(Phase::Loop, Duration::from_micros(10));
    c.add(Phase::Plan, Duration::from_micros(20));
    c.add(Phase::Pack, Duration::from_micros(30));
    c.add(Phase::Exec, Duration::from_micros(1000));
    c.add(Phase::Sample, Duration::from_micros(40));
    c.add(Phase::Device, Duration::from_micros(500));
    c.add(Phase::Finish, Duration::from_micros(50));
    c.steps = 2;
    c.seqs = 6;
    let s = c.summary();
    assert_eq!(s.host_us, 75.0);
    assert_eq!(s.us[Phase::Exec as usize], 500.0);
    assert_eq!(s.us[Phase::Device as usize], 250.0);
    assert_eq!(s.us[Phase::Sample as usize], 20.0);
    assert_eq!(s.seqs_per_step, 3.0);
}

#[test]
fn a_lap_moves_the_mark() {
    let mut c = clock();
    let start = Instant::now() - Duration::from_millis(3);
    let mut at = start;
    c.lap(Phase::Plan, &mut at);
    assert!(at > start);
    assert!(c.sums[Phase::Plan as usize] >= Duration::from_millis(3));
}

#[test]
fn an_off_clock_sums_nothing() {
    let mut c = HostClock {
        on: false,
        ..HostClock::default()
    };
    let start = Instant::now() - Duration::from_millis(3);
    let mut at = start;
    c.lap(Phase::Plan, &mut at);
    c.add(Phase::Loop, Duration::from_millis(1));
    c.step(4);
    assert_eq!(at, start);
    assert_eq!(c.steps, 0);
    assert_eq!(c.summary().host_us, 0.0);
}

#[test]
fn an_empty_window_reports_zeros() {
    let s = clock().summary();
    assert_eq!(s.host_us, 0.0);
    assert_eq!(s.seqs_per_step, 0.0);
}
