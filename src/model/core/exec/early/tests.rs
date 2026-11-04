use super::*;

/// Runs [`next_work`] to the end like `Executor::early_inputs`, returning the work in order.
fn drain(p: EarlyProgress<'_>, early: &mut [Early]) -> Vec<Work> {
    let mut done = Vec::new();
    while let Some(w) = next_work(p, early) {
        match w {
            Work::Fill(s) => early[s] = Early::Filled,
            Work::HandOff(s) => early[s] = Early::Ready,
        }
        done.push(w);
    }
    done
}

#[test]
fn rounds_are_a_wavefront() {
    // 3 micro-chunks over 2 stages: 4 rounds.
    assert_eq!(round_active(0, 2, 3), vec![(0, 0)]);
    assert_eq!(round_active(1, 2, 3), vec![(0, 1), (1, 0)]);
    assert_eq!(round_active(2, 2, 3), vec![(0, 2), (1, 1)]);
    assert_eq!(round_active(3, 2, 3), vec![(1, 2)]);
    assert_eq!(round_active(4, 2, 3), vec![]);
}

#[test]
fn nothing_before_the_slowest_stage_it_needs() {
    // Round 1 of 2 stages, stage 1 done, stage 0 still running: stage 1's next inputs only
    // (its hand-off copies stage 0's residual).
    let next = round_active(2, 2, 3);
    let ran = [true, true];
    let mut early = [Early::No; 2];
    let p = EarlyProgress {
        next: &next,
        ran: &ran,
        finished: &[false, true],
    };
    assert_eq!(drain(p, &mut early), vec![Work::Fill(1)]);
    // Stage 0 done: the hand-off, then stage 0's own inputs, then its (empty) hand-off.
    let p = EarlyProgress {
        finished: &[true, true],
        ..p
    };
    assert_eq!(
        drain(p, &mut early),
        vec![Work::HandOff(1), Work::Fill(0), Work::HandOff(0)]
    );
    assert_eq!(early, [Early::Ready; 2]);
}

#[test]
fn a_stage_waits_for_the_stage_after_it() {
    // 4 stages, all running; stage 1 is the slowest. Stage 3's inputs and hand-off (stage 2's
    // residual) go; stage 2 waits for stage 1's residual; stages 1 and 0 wait behind it.
    let next = round_active(5, 4, 8);
    let ran = [true; 4];
    let mut early = [Early::No; 4];
    let p = EarlyProgress {
        next: &next,
        ran: &ran,
        finished: &[true, false, true, true],
    };
    assert_eq!(
        drain(p, &mut early),
        vec![Work::Fill(3), Work::HandOff(3), Work::Fill(2)]
    );
    assert_eq!(early, [Early::No, Early::No, Early::Filled, Early::Ready]);
}

#[test]
fn a_running_stage_keeps_its_graph() {
    // Stage 0 still runs: its next micro-chunk is not built even with stage 1 ready.
    let next = round_active(2, 2, 3);
    let ran = [true, true];
    let mut early = [Early::No, Early::Ready];
    let p = EarlyProgress {
        next: &next,
        ran: &ran,
        finished: &[false, true],
    };
    assert_eq!(next_work(p, &early), None);
    assert!(drain(p, &mut early).is_empty());
}

#[test]
fn a_stage_starting_its_first_micro_chunk() {
    // Round 0 → 1 of 2 stages: stage 1 did not run, so its inputs go at once; its hand-off
    // waits for stage 0.
    let next = round_active(1, 2, 3);
    let ran = [true, false];
    let mut early = [Early::No; 2];
    let p = EarlyProgress {
        next: &next,
        ran: &ran,
        finished: &[false, false],
    };
    assert_eq!(drain(p, &mut early), vec![Work::Fill(1)]);
}

#[test]
fn the_last_round_has_nothing_next() {
    let next = round_active(4, 2, 3);
    let p = EarlyProgress {
        next: &next,
        ran: &[false, true],
        finished: &[false, true],
    };
    assert_eq!(next_work(p, &[Early::No; 2]), None);
}
