use super::*;

#[test]
fn a_layer_split_with_mtp_holds_its_commit() {
    assert!(wanted(true, true, 2, false));
    assert!(wanted(true, true, 4, false));
}

#[test]
fn one_stage_commits_at_once() {
    assert!(!wanted(true, true, 1, false));
}

#[test]
fn without_mtp_or_the_switch_it_commits_at_once() {
    assert!(!wanted(true, false, 4, false));
    assert!(!wanted(false, true, 4, false));
}

#[test]
fn commits_run_elsewhere_are_not_held() {
    // Tensor or GLM parallel steps, a remote half, pipelined rounds, deferred commits.
    assert!(!wanted(true, true, 4, true));
}

fn held(stages: &[usize]) -> HeldCommit {
    HeldCommit {
        commits: vec![(0, 2), (3, 1)],
        stages: stages.to_vec(),
        error: None,
    }
}

#[test]
fn the_other_stages_go_first_in_order() {
    let mut h = held(&[0, 1, 2, 3]);
    let (stages, commits) = h.take(|s| s != 3);
    assert_eq!(stages, [0, 1, 2]);
    assert_eq!(commits, [(0, 2), (3, 1)]);
    assert!(!h.is_empty());
    // The last stage keeps the kept counts.
    let (stages, commits) = h.take(|_| true);
    assert_eq!(stages, [3]);
    assert_eq!(commits, [(0, 2), (3, 1)]);
    assert!(h.is_empty());
    assert!(h.commits.is_empty());
}

#[test]
fn nothing_held_takes_nothing() {
    let mut h = HeldCommit::default();
    assert!(h.is_empty());
    let (stages, commits) = h.take(|_| true);
    assert!(stages.is_empty() && commits.is_empty());
}

#[test]
fn the_first_error_is_kept() {
    let mut h = held(&[]);
    h.fail(GgmlError::External("first".into()));
    h.fail(GgmlError::External("second".into()));
    assert!(matches!(&h.error, Some(GgmlError::External(e)) if e == "first"));
}

#[test]
fn late_the_last_commit_waits_for_the_second_pass() {
    // The first pass queued the other stages' commits: the last one waits.
    assert!(!last_now(true, true));
    // The second pass (nothing else left to queue) takes it behind its reads.
    assert!(last_now(true, false));
}

#[test]
fn not_late_the_first_pass_takes_the_last_commit() {
    assert!(last_now(false, true));
    assert!(last_now(false, false));
}
