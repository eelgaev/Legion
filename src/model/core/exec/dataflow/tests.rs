use super::{Prep, Progress, Work, next_work};

/// Plays a step with the host doing all it can, later stages first, and graphs finishing in
/// the order `finish` picks among the running ones; returns each launch as (stage, micro).
fn play(
    n_stages: usize,
    n_micro: usize,
    finish: impl Fn(&[usize]) -> usize,
) -> Vec<(usize, usize)> {
    let mut p = Progress::new(n_stages, 0);
    let mut prep = vec![Prep::No; n_stages];
    let mut launches = Vec::new();
    let mut running = Vec::new();
    loop {
        let mut moved = true;
        while moved {
            moved = false;
            for s in (0..n_stages).rev() {
                let Some(w) = next_work(&p, prep[s], s, n_micro) else {
                    continue;
                };
                moved = true;
                let m = p.launched[s];
                match w {
                    Work::Fill => prep[s] = Prep::Filled,
                    Work::HandOff => {
                        if s > 0 {
                            // The residual is the previous stage's graph output of `m`.
                            assert!(p.finished[s - 1] > m);
                            p.taken[s - 1] = m + 1;
                        }
                        prep[s] = Prep::Ready;
                    }
                    Work::Launch => {
                        launches.push((s, m));
                        p.launched[s] += 1;
                        prep[s] = Prep::No;
                        running.push(s);
                    }
                }
            }
        }
        if p.finished.iter().all(|&n| n == n_micro) {
            return launches;
        }
        assert!(!running.is_empty(), "stuck: {p:?} {prep:?}");
        let s = running.remove(finish(&running));
        p.finished[s] += 1;
    }
}

#[test]
fn every_micro_chunk_runs_once_per_stage_in_order() {
    for n_stages in 1..=4 {
        for n_micro in 1..=6 {
            for pick in [0usize, 1, 2] {
                let launches = play(n_stages, n_micro, |r| match pick {
                    0 => 0,
                    1 => r.len() - 1,
                    _ => r.len() / 2,
                });
                assert_eq!(launches.len(), n_stages * n_micro);
                for s in 0..n_stages {
                    let ms: Vec<usize> =
                        launches.iter().filter(|l| l.0 == s).map(|l| l.1).collect();
                    assert_eq!(ms, (0..n_micro).collect::<Vec<_>>());
                }
            }
        }
    }
}

#[test]
fn a_stage_runs_one_graph_at_a_time() {
    let mut p = Progress::new(2, 0);
    p.launched[0] = 1;
    assert_eq!(next_work(&p, Prep::No, 0, 4), None);
    assert_eq!(next_work(&p, Prep::Ready, 0, 4), None);
    p.finished[0] = 1;
    // Its residual of micro-chunk 0 not taken by stage 1 yet.
    assert_eq!(next_work(&p, Prep::No, 0, 4), None);
    p.taken[0] = 1;
    assert_eq!(next_work(&p, Prep::No, 0, 4), Some(Work::Fill));
}

#[test]
fn a_hand_off_waits_for_the_stage_before() {
    let mut p = Progress::new(3, 0);
    assert_eq!(next_work(&p, Prep::No, 1, 4), Some(Work::Fill));
    assert_eq!(next_work(&p, Prep::Filled, 1, 4), None);
    p.launched[0] = 1;
    assert_eq!(next_work(&p, Prep::Filled, 1, 4), None);
    p.finished[0] = 1;
    assert_eq!(next_work(&p, Prep::Filled, 1, 4), Some(Work::HandOff));
    // The last stage's residual is nobody's input.
    p.launched[2] = 1;
    p.finished[2] = 1;
    assert_eq!(next_work(&p, Prep::No, 2, 4), Some(Work::Fill));
    // Done with the step.
    p.launched[2] = 4;
    p.finished[2] = 4;
    assert_eq!(next_work(&p, Prep::No, 2, 4), None);
}

/// `--long-dataflow-async` with mailboxes: plays a step where the host does all it can and
/// the stage threads and GPUs move in an order `seed` picks. A graph is queued by its thread
/// after its launch, and done once queued, after its own previous graph and after the
/// stage before's graph of the same micro-chunk (its take waits for that one's copy). Checks
/// the order the events and buffers of `hand_off` need.
fn play_async(n_stages: usize, n_micro: usize, seed: u64) {
    let mut p = Progress::new(n_stages, 2);
    p.early = true;
    for s in 0..n_stages {
        p.ahead[s] = s + 1 < n_stages;
    }
    let mut prep = vec![Prep::No; n_stages];
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = |n: usize| {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as usize) % n
    };
    // Per stage, and per micro-chunk: when its take of it was queued (host order).
    let mut took_at = vec![vec![None; n_micro]; n_stages];
    let mut clock = 0usize;
    loop {
        let mut moved = true;
        while moved {
            moved = false;
            for s in 0..n_stages {
                let Some(w) = next_work(&p, prep[s], s, n_micro) else {
                    continue;
                };
                moved = true;
                clock += 1;
                let m = p.launched[s];
                let in_flight = p.launched[s] - p.finished[s];
                match w {
                    Work::Fill => prep[s] = Prep::Filled,
                    Work::HandOff => {
                        if s > 0 {
                            // The stage before queued its copy of `m`.
                            assert!(p.queued[s - 1] > m);
                            p.taken[s - 1] = m + 1;
                            took_at[s][m] = Some(clock);
                        }
                        prep[s] = Prep::Ready;
                    }
                    Work::Launch => {
                        // At most one graph runs under it, and it is queued.
                        assert!(in_flight == 0 || (in_flight == 1 && p.queued[s] == m));
                        assert!(in_flight == 0 || p.ahead[s]);
                        if s + 1 < n_stages && m >= 2 {
                            // The mailbox entry (and `taken`) of `m - 2` was taken.
                            assert!(took_at[s + 1][m - 2].is_some());
                        }
                        p.launched[s] += 1;
                        prep[s] = Prep::No;
                    }
                }
            }
        }
        if p.finished.iter().all(|&n| n == n_micro) {
            return;
        }
        // What the threads and GPUs may do: queue a launched graph, or finish a queued one.
        let mut can = Vec::new();
        for s in 0..n_stages {
            let sends = s + 1 < n_stages;
            if sends && p.queued[s] < p.launched[s] {
                can.push((s, true));
            }
            let m = p.finished[s];
            let queued = if sends {
                p.queued[s] > m
            } else {
                p.launched[s] > m
            };
            let up_done = s == 0 || p.finished[s - 1] > m;
            if queued && up_done {
                can.push((s, false));
            }
        }
        assert!(!can.is_empty(), "stuck: {p:?} {prep:?}");
        let (s, queue) = can[next(can.len())];
        if queue {
            p.queued[s] += 1;
        } else {
            p.finished[s] += 1;
        }
    }
}

#[test]
fn async_hand_off_keeps_the_order() {
    for n_stages in 1..=4 {
        for n_micro in 1..=7 {
            for seed in 0..40 {
                play_async(n_stages, n_micro, seed);
            }
        }
    }
}

/// Without a kept graph for its next micro-chunk (`ahead` off), a stage waits for its graph.
#[test]
fn no_launch_ahead_without_ahead() {
    let mut p = Progress::new(2, 2);
    p.early = true;
    p.launched[0] = 1;
    p.queued[0] = 1;
    assert_eq!(next_work(&p, Prep::No, 0, 4), None);
    p.ahead[0] = true;
    assert_eq!(next_work(&p, Prep::No, 0, 4), Some(Work::Fill));
    // Not before its thread queued the running one.
    p.queued[0] = 0;
    assert_eq!(next_work(&p, Prep::No, 0, 4), None);
    // Never two ahead.
    p.queued[0] = 2;
    p.launched[0] = 2;
    assert_eq!(next_work(&p, Prep::No, 0, 4), None);
}
