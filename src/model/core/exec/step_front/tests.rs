use super::*;

fn clock(on: bool) -> FrontClock {
    FrontClock {
        on: Some(on),
        ..Default::default()
    }
}

#[test]
fn off_takes_no_time_and_keeps_no_laps() {
    let mut c = clock(false);
    c.picked();
    c.returned();
    assert!(c.picked.is_none() && c.returned.is_none());
    let mut f = c.step();
    f.lap("fill");
    f.done();
    assert!(f.laps.is_empty());
    assert!(f.start.is_none());
}

#[test]
fn on_records_laps_in_order_and_logs_once() {
    let mut c = clock(true);
    c.picked();
    c.returned();
    let mut f = c.step();
    // The pass's marks belong to this step only.
    assert!(c.picked.is_none() && c.returned.is_none());
    f.lap("held");
    f.lap("fill");
    f.done();
    let names: Vec<_> = f.laps.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["held", "fill", "launch"]);
    // Laps after the launch (later rounds of the step) add nothing.
    f.lap("prepare");
    f.done();
    assert_eq!(f.laps.len(), 3);
}

#[test]
fn a_step_without_a_pass_before_it_has_no_tail() {
    let mut c = clock(true);
    let f = c.step();
    assert!(f.picked.is_none() && f.returned.is_none());
}
