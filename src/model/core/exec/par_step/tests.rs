use super::par_refusal;
use crate::model::glm5next::par::MAX_TOKENS;

/// A parallel step runs decode and verify steps of up to `MAX_TOKENS` tokens, capturing the
/// last layer's head input (the MTP head's) or nothing (#669).
#[test]
fn par_runs_verify_steps_with_replay_buffers() {
    assert!(par_refusal(1, MAX_TOKENS, false, &[], 78).is_none());
    assert!(par_refusal(MAX_TOKENS, MAX_TOKENS, false, &[], 78).is_none());
    assert!(par_refusal(3, MAX_TOKENS, false, &[77], 78).is_none());
}

#[test]
fn par_refuses_what_it_cannot_run() {
    assert!(par_refusal(MAX_TOKENS + 1, MAX_TOKENS, false, &[], 78).is_some());
    // Verify rows without replay buffers could not be rolled back.
    assert!(par_refusal(3, MAX_TOKENS, true, &[], 78).is_some());
    // Only the last layer's head input is captured.
    assert!(par_refusal(3, MAX_TOKENS, false, &[76], 78).is_some());
    assert!(par_refusal(3, MAX_TOKENS, false, &[77, 77], 78).is_some());
}

/// Long prompt steps (`--glm-prompt-step`, #686) run up to the exchange's limit, no further.
#[test]
fn par_runs_prompt_steps_up_to_the_slots_tokens() {
    assert!(par_refusal(2048, 2048, false, &[77], 78).is_none());
    assert!(par_refusal(2049, 2048, false, &[77], 78).is_some());
}

/// Rows: how long each GPU waited; columns (without the diagonal): how long the others waited
/// for each GPU.
#[test]
fn wait_sums_give_each_gpu_its_wait_and_its_lateness() {
    let m = vec![vec![1, 10, 20], vec![0, 2, 30], vec![5, 0, 3]];
    let (waits, late) = super::wait_sums(&m);
    assert_eq!(waits, vec![31, 32, 8]);
    assert_eq!(late, vec![5, 10, 50]);
    assert_eq!(super::wait_sums(&[]), (vec![], vec![]));
}

/// The launches of a step run at once (each waits for all the others here), on threads kept
/// across steps, their results in order.
#[test]
fn launch_pool_runs_every_launch_at_once_and_keeps_its_threads() {
    use super::pool::{Launch, LaunchPool};
    use std::sync::{Arc, Barrier, Mutex};
    let pool = LaunchPool::new(3).unwrap();
    assert_eq!(pool.len(), 3);
    let ids = Arc::new(Mutex::new(Vec::new()));
    for step in 0..2 {
        let barrier = Arc::new(Barrier::new(3));
        let jobs = (0..3)
            .map(|d| {
                let barrier = barrier.clone();
                let ids = ids.clone();
                let job: Launch = Box::new(move || {
                    barrier.wait();
                    ids.lock()
                        .unwrap()
                        .push((step, d, std::thread::current().id()));
                    if d == 1 {
                        Err(crate::ggml::GgmlError::Compute(0))
                    } else {
                        Ok(())
                    }
                });
                (Vec::new(), job)
            })
            .collect();
        let results = pool.run(jobs);
        assert!(results[0].is_ok() && results[1].is_err() && results[2].is_ok());
    }
    let ids = ids.lock().unwrap();
    let thread_of = |step, d| {
        ids.iter()
            .find(|&&(s, e, _)| (s, e) == (step, d))
            .unwrap()
            .2
    };
    for d in 0..3 {
        assert_eq!(thread_of(0, d), thread_of(1, d));
        assert_ne!(thread_of(0, d), std::thread::current().id());
    }
}

/// A launch that panics fails the step once the others are done, as a joined thread would.
#[test]
fn launch_pool_reports_a_panicked_launch_after_the_others() {
    use super::pool::{Launch, LaunchPool};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let pool = LaunchPool::new(2).unwrap();
    let other_done = Arc::new(AtomicBool::new(false));
    let done = other_done.clone();
    let jobs: Vec<(Vec<u32>, Launch)> = vec![
        (Vec::new(), Box::new(|| panic!("launch failed"))),
        (
            Vec::new(),
            Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                done.store(true, Ordering::SeqCst);
                Ok(())
            }),
        ),
    ];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.run(jobs)));
    assert!(r.is_err());
    assert!(other_done.load(Ordering::SeqCst));
}

/// Started launches run while the caller goes on; a join waits for them, results in thread
/// order, and only for the threads started (#714).
#[test]
fn launch_pool_starts_launches_and_joins_them_later() {
    use super::pool::{Launch, LaunchPool};
    use std::sync::mpsc::channel;
    let pool = LaunchPool::new(3).unwrap();
    assert!(pool.join().is_empty());
    let (go, wait) = channel::<()>();
    let job: Launch = Box::new(move || {
        // Runs only once the caller, who went on, says so.
        wait.recv().unwrap();
        Err(crate::ggml::GgmlError::Compute(2))
    });
    pool.start(2, (Vec::new(), job));
    pool.start(0, (Vec::new(), Box::new(|| Ok(()))));
    assert!(pool.pending());
    go.send(()).unwrap();
    let results = pool.join();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok() && results[1].is_err());
    assert!(!pool.pending());
    // A run after the join: every thread is free again.
    assert_eq!(
        pool.run(vec![(Vec::new(), Box::new(|| Ok(())) as Launch)])
            .len(),
        1
    );
}

/// `join_next` hands back whichever started launch is done first, not thread order (#756: a
/// round of groups in flight reads each stage's output once that stage is done), then the
/// rest, then nothing.
#[test]
fn launch_pool_joins_the_first_launch_done() {
    use super::pool::{Launch, LaunchPool};
    use std::sync::mpsc::channel;
    let pool = LaunchPool::new(3).unwrap();
    assert!(pool.join_next().is_none());
    let (go, wait) = channel::<()>();
    let job: Launch = Box::new(move || {
        // Done only once thread 2's result was taken.
        wait.recv().unwrap();
        Err(crate::ggml::GgmlError::Compute(0))
    });
    pool.start(0, (Vec::new(), job));
    pool.start(2, (Vec::new(), Box::new(|| Ok(()))));
    let (d, r) = pool.join_next().unwrap();
    assert_eq!(d, 2);
    assert!(r.is_ok());
    assert!(pool.pending());
    go.send(()).unwrap();
    let (d, r) = pool.join_next().unwrap();
    assert_eq!(d, 0);
    assert!(r.is_err());
    assert!(!pool.pending());
    assert!(pool.join_next().is_none());
    assert!(pool.join().is_empty());
}

/// The caller's work runs while the threads launch, not after them (#723): a launch that
/// waits for the work ends, and the results come back in thread order with the work's.
#[test]
fn launch_pool_runs_work_beside_the_launches() {
    use super::pool::{Launch, LaunchPool};
    use std::sync::mpsc::channel;
    let pool = LaunchPool::new(3).unwrap();
    let (go, wait) = channel::<()>();
    let (started, seen) = channel::<()>();
    let jobs: Vec<(Vec<u32>, Launch)> = vec![
        (Vec::new(), Box::new(|| Ok(()))),
        (
            Vec::new(),
            Box::new(move || {
                // Started before the work, done only once the work says so.
                started.send(()).unwrap();
                wait.recv().unwrap();
                Err(crate::ggml::GgmlError::Compute(1))
            }),
        ),
    ];
    let (results, out) = pool.run_beside(jobs, || {
        seen.recv().unwrap();
        go.send(()).unwrap();
        7
    });
    assert_eq!(out, 7);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok() && results[1].is_err());
    assert!(!pool.pending());
    // Nothing started: the work alone.
    let (results, out) = pool.run_beside(Vec::new(), || 3);
    assert!(results.is_empty() && out == 3);
}

/// The engine thread spins for a launch thread's result until the deadline, then sleeps for it
/// (#676); a gone thread gives none either way.
#[test]
fn launch_results_are_polled_then_waited_for() {
    use super::pool::recv_spinning;
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};
    let far = Instant::now() + Duration::from_secs(60);
    // Sent while spinning.
    let (tx, rx) = channel();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        tx.send(5).unwrap();
    });
    assert_eq!(recv_spinning(&rx, far), Some(5));
    t.join().unwrap();
    // Sent after the deadline: the blocking wait takes it.
    let (tx, rx) = channel();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        tx.send(6).unwrap();
    });
    assert_eq!(recv_spinning(&rx, Instant::now()), Some(6));
    t.join().unwrap();
    // Already there, deadline or not.
    let (tx, rx) = channel();
    tx.send(7).unwrap();
    assert_eq!(recv_spinning(&rx, Instant::now()), Some(7));
    // The sender gone: none, spinning or sleeping.
    drop(tx);
    assert_eq!(recv_spinning(&rx, far), None);
    assert_eq!(recv_spinning(&rx, Instant::now()), None::<i32>);
}

/// Held commit launches start only when asked (the MTP pass is queued), a join starts any
/// still held, and a failed launch's error is given once, to the next call that can return
/// it (#714).
#[test]
fn held_commits_start_late_and_keep_their_error() {
    use super::ParStats;
    use super::pool::{Launch, LaunchPool};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let ran = Arc::new(AtomicUsize::new(0));
    let job = |fail: bool| -> Launch {
        let ran = ran.clone();
        Box::new(move || {
            ran.fetch_add(1, Ordering::SeqCst);
            if fail {
                Err(crate::ggml::GgmlError::Compute(1))
            } else {
                Ok(())
            }
        })
    };
    let mut p = ParStats {
        pool: Some(LaunchPool::new(3).unwrap()),
        ..Default::default()
    };
    p.held.jobs = vec![(0, Vec::new(), job(false)), (1, Vec::new(), job(false))];
    // Held: nothing runs until started.
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(ran.load(Ordering::SeqCst), 0);
    p.start_held();
    assert!(p.held.jobs.is_empty() && p.pool.as_ref().unwrap().pending());
    p.join_held();
    assert_eq!(ran.load(Ordering::SeqCst), 2);
    assert!(p.launch_result().is_ok());
    // A join with launches still held starts them first; the failure is kept for one call.
    p.held.jobs = vec![(1, Vec::new(), job(true))];
    p.join_held();
    assert_eq!(ran.load(Ordering::SeqCst), 3);
    assert!(!p.pool.as_ref().unwrap().pending());
    assert!(p.launch_result().is_err());
    assert!(p.launch_result().is_ok());
    // Nothing held, nothing started: a no-op.
    p.join_held();
    assert!(p.launch_result().is_ok());
    // Taken by GPU for a parallel MTP pass (#717): no longer held, nothing started.
    p.held.jobs = vec![(2, Vec::new(), job(false)), (0, Vec::new(), job(false))];
    let mut taken = p.take_held(3);
    assert_eq!(
        taken.iter().map(Option::is_some).collect::<Vec<_>>(),
        [true, false, true]
    );
    assert!(p.held.jobs.is_empty());
    p.join_held();
    assert_eq!(ran.load(Ordering::SeqCst), 3);
    for launch in taken.iter_mut().filter_map(Option::take) {
        assert!(launch().is_ok());
    }
    assert_eq!(ran.load(Ordering::SeqCst), 5);
}

/// The gap between steps runs from one step's graphs done to the next step's launches; what
/// no phase measured is the engine's. A gap not seen from its start, or next to a prompt step,
/// does not count.
#[test]
fn gap_clock_splits_the_gap_between_steps_by_phase() {
    use super::{GapClock, GapPhase};
    use std::time::{Duration, Instant};
    let ms = Duration::from_millis;
    let t = Instant::now();
    let mut c = GapClock::default();
    // The first step: its launches close no gap.
    c.add(GapPhase::Prepare, ms(5));
    c.launch(t, true);
    c.done(t + ms(20), true);
    c.add(GapPhase::Output, ms(1));
    c.add(GapPhase::Mtp, ms(2));
    c.add(GapPhase::Fill, ms(1));
    c.launch(t + ms(26), true);
    let (gaps, gap, phases) = c.take_ms();
    assert_eq!((gaps, gap), (1, 6.0));
    let of = |p: GapPhase| phases[p as usize].1;
    assert_eq!(of(GapPhase::Output), 1.0);
    assert_eq!(of(GapPhase::Mtp), 2.0);
    assert_eq!(of(GapPhase::Fill), 1.0);
    assert_eq!(of(GapPhase::Prepare), 0.0);
    assert_eq!(of(GapPhase::Engine), 2.0);
    // Taken: the sums start over.
    assert_eq!(c.take_ms().1, 0.0);
    // A prompt step: neither the gap before it nor the one after counts.
    c.done(t + ms(40), true);
    c.add(GapPhase::Prepare, ms(3));
    c.launch(t + ms(45), false);
    c.done(t + ms(60), false);
    c.add(GapPhase::Mtp, ms(2));
    c.launch(t + ms(64), true);
    let (gaps, gap, phases) = c.take_ms();
    assert_eq!((gaps, gap), (0, 0.0));
    assert!(phases.iter().all(|&(_, v)| v == 0.0));
    // The next decode gap counts again.
    c.done(t + ms(80), true);
    c.add(GapPhase::Commit, ms(1));
    c.launch(t + ms(83), true);
    let (gaps, gap, phases) = c.take_ms();
    assert_eq!((gaps, gap), (1, 3.0));
    assert_eq!(phases[GapPhase::Commit as usize].1, 1.0);
    assert_eq!(phases[GapPhase::Engine as usize].1, 2.0);
}
