use super::Timeline;

/// Two stages, two micro-chunks of 10 ms each, the host 1 ms late on every start.
#[test]
fn fill_drain_and_lateness() {
    let mut t = Timeline::new(2, 2);
    t.launched(0, 0, 1.0);
    t.finished(0, 0, 11.0);
    t.launched(0, 1, 12.0);
    t.launched(1, 0, 12.0);
    t.finished(0, 1, 22.0);
    t.finished(1, 0, 22.0);
    t.launched(1, 1, 23.0);
    t.finished(1, 1, 33.0);
    assert_eq!(t.fill(), vec![1.0, 12.0]);
    assert_eq!(t.drain(33.0), vec![11.0, 0.0]);
    assert_eq!(t.late(), &[2.0, 2.0]);
}

/// A stage that waits on the residual is not late: its start counts from the later of its own
/// previous graph and the stage before's.
#[test]
fn waiting_on_the_stage_before_is_not_late() {
    let mut t = Timeline::new(2, 2);
    t.launched(0, 0, 0.0);
    t.finished(0, 0, 10.0);
    t.launched(1, 0, 10.0);
    t.launched(0, 1, 10.0);
    t.finished(1, 0, 15.0);
    t.finished(0, 1, 30.0);
    t.launched(1, 1, 30.5);
    assert_eq!(t.late(), &[0.0, 0.5]);
}

/// A stage that never ran drains nothing.
#[test]
fn empty_stage() {
    let t = Timeline::new(2, 1);
    assert_eq!(t.fill(), vec![0.0, 0.0]);
    assert_eq!(t.drain(5.0), vec![0.0, 0.0]);
}

/// A graph queued while its own previous one or the residual it takes still runs
/// (`--long-dataflow-async`) is not late.
#[test]
fn queued_ahead_is_not_late() {
    let mut t = Timeline::new(2, 2);
    t.launched(0, 0, 0.0);
    t.launched(0, 1, 1.0);
    t.launched(1, 0, 2.0);
    t.finished(0, 0, 10.0);
    t.finished(0, 1, 20.0);
    t.finished(1, 0, 20.0);
    t.launched(1, 1, 21.0);
    assert_eq!(t.late(), &[0.0, 1.0]);
}
