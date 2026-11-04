use super::*;

/// The steady round: a commit uploads its counts, the verify step after it reads them as they
/// are (no upload) and consumes them, and the next commit uploads again.
#[test]
fn commit_then_verify_uploads_once_per_round() {
    let mut p = Pending::new(4);
    assert!(!p.any());
    assert!(p.can_commit(&[(0, 2), (1, 3)]));
    assert_eq!(p.commit(&[(0, 2), (1, 3)]), vec![2, 3, 0, 0]);
    assert!(p.any() && p.touches(1..2) && !p.touches(2..4));
    assert_eq!(p.verify(0..2, 4), None);
    assert!(!p.any());
    assert!(p.can_commit(&[(0, 1), (1, 1)]));
    assert_eq!(p.commit(&[(0, 1), (1, 1)]), vec![1, 1, 0, 0]);
    assert_eq!(p.verify(0..2, 4), None);
}

/// A verify step of a slot whose device count is stale (its commit was consumed, the slot
/// reset and reused without a commit in between) uploads zeros first.
#[test]
fn stale_device_counts_are_uploaded_before_a_verify() {
    let mut p = Pending::new(3);
    p.commit(&[(1, 4)]);
    assert_eq!(p.verify(1..2, 4), None);
    // The device still holds 4 for slot 1; a new sequence's first verify there must not
    // replay it.
    assert_eq!(p.verify(1..2, 4), Some(vec![0, 0, 0]));
    assert_eq!(p.verify(1..2, 4), None);
}

/// A reset slot's deferred commit is dropped, and a verify there uploads the zero.
#[test]
fn a_dropped_slot_is_not_replayed() {
    let mut p = Pending::new(2);
    p.commit(&[(0, 2), (1, 1)]);
    p.drop_slot(0);
    assert!(!p.touches(0..1) && p.touches(1..2));
    assert_eq!(p.verify(0..1, 4), Some(vec![0, 1]));
    assert!(p.touches(1..2));
}

/// A flush replays every deferred commit at once and leaves none.
#[test]
fn flush_takes_everything_once() {
    let mut p = Pending::new(3);
    assert_eq!(p.flush(), None);
    p.commit(&[(2, 5)]);
    assert!(!p.can_commit(&[(2, 1)]));
    assert_eq!(p.flush(), Some(vec![0, 0, 5]));
    assert_eq!(p.flush(), None);
    assert!(p.can_commit(&[(2, 1)]));
    // The device holds the flushed counts: a verify of slot 2 must upload zeros.
    assert_eq!(p.verify(2..3, 4), Some(vec![0, 0, 0]));
}

/// A pool whose verify steps may leave the states (#836): kept tokens wait over rounds while the
/// packed rows have room for the next step's tokens after them, and each commit extends the
/// waiting run; the device's words need no upload in between.
#[test]
fn leaving_verify_steps_extend_the_waiting_run() {
    let w = |keep, start| count_word(keep, start, true);
    let mut p = Pending::with_room(2, 9, true);
    assert_eq!(p.commit(&[(0, 3), (1, 1)]), vec![w(3, 0), w(1, 0)]);
    // 3 kept + 4 tokens fit 9 rows: the state stays, the tokens go after the kept ones.
    assert_eq!(p.verify(0..2, 4), None);
    assert!(p.touches(0..1) && p.touches(1..2));
    assert!(p.can_commit(&[(0, 2), (1, 4)]));
    assert_eq!(p.commit(&[(0, 2), (1, 4)]), vec![w(5, 0), w(5, 0)]);
    // 5 + 4 = 9 still fits.
    assert_eq!(p.verify(0..1, 4), None);
    assert_eq!(p.commit(&[(0, 4)]), vec![w(9, 0), w(5, 0)]);
    // 9 + 4 does not: the step writes the state and stores from row 0 again.
    assert_eq!(p.verify(0..1, 4), None);
    assert!(!p.touches(0..1));
    assert_eq!(p.commit(&[(0, 1)]), vec![w(1, 0), w(5, 0)]);
}

/// A leaving pool with a width floor (#858): a verify step of fewer sequences clears the leave
/// bit of its slots' words (one upload) and writes the states; a wide one sets it again (one
/// upload), and wide steps after it need none.
#[test]
fn narrow_verify_steps_write_the_states() {
    let w = |keep, start| count_word(keep, start, true);
    let mut p = Pending::with_room(4, 9, true).leave_from(3);
    assert_eq!(p.commit(&[(0, 3), (1, 1)]), vec![w(3, 0), w(1, 0), 0, 0]);
    // Two sequences: below the floor, the plain counts go up and the states are written.
    assert_eq!(p.verify(0..2, 4), Some(vec![3, 1, 0, 0]));
    assert!(!p.any());
    // The commit after it predicts another narrow step.
    assert_eq!(p.commit(&[(0, 2), (1, 1)]), vec![2, 1, 0, 0]);
    // Four sequences: the leave bit goes up, and the states stay.
    assert_eq!(p.verify(0..4, 4), Some(vec![w(2, 0), w(1, 0), 0, 0]));
    assert!(p.touches(0..1) && p.touches(1..2));
    assert_eq!(p.commit(&[(0, 1), (1, 1)]), vec![w(3, 0), w(2, 0), 0, 0]);
    assert_eq!(p.verify(0..4, 4), None);
    assert!(p.touches(0..1));
}

/// A flush between a leaving verify step and its commit: the waiting tokens are replayed, and
/// the commit's kept tokens are where the step stored them, after those.
#[test]
fn a_commit_after_a_flush_starts_where_the_verify_step_stored() {
    let w = |keep, start| count_word(keep, start, true);
    let mut p = Pending::with_room(1, 9, true);
    p.commit(&[(0, 3)]);
    assert_eq!(p.verify(0..1, 4), None);
    assert_eq!(p.flush(), Some(vec![w(3, 0)]));
    assert!(!p.any());
    assert_eq!(p.commit(&[(0, 2)]), vec![w(2, 3)]);
    // 3 + 2 + 4 = 9 rows: left again, stored from row 5.
    assert_eq!(p.verify(0..1, 4), None);
    assert_eq!(p.commit(&[(0, 1)]), vec![w(3, 3)]);
    // 3 + 3 + 4 > 9: written; the next commit starts at row 0.
    assert_eq!(p.verify(0..1, 4), None);
    assert_eq!(p.commit(&[(0, 2)]), vec![w(2, 0)]);
}

/// A reset slot of a leaving pool starts over at row 0 and uploads its zero before a verify.
#[test]
fn a_dropped_slot_of_a_leaving_pool_starts_over() {
    let mut p = Pending::with_room(1, 9, true);
    p.commit(&[(0, 3)]);
    assert_eq!(p.verify(0..1, 2), None);
    p.drop_slot(0);
    assert!(!p.any() && p.can_commit(&[(0, 1)]));
    assert_eq!(p.verify(0..1, 2), Some(vec![0]));
    assert_eq!(p.commit(&[(0, 1)]), vec![count_word(1, 0, true)]);
}

/// The count words: kept tokens, start row, the leave bit; a word leaves its state only where
/// the rows have room.
#[test]
fn count_words_and_the_leave_rule() {
    assert_eq!(count_word(5, 0, false), 5);
    assert_eq!(count_word(2, 3, true), 2 | 3 << 8 | 1 << 16);
    assert!(leaves(count_word(3, 2, true), 4, 9));
    assert!(!leaves(count_word(3, 3, true), 4, 9));
    assert!(!leaves(count_word(3, 0, false), 1, 9));
}
