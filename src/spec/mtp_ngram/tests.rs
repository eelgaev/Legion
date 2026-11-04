use super::*;

#[test]
fn longest_match_prefers_length_then_recency() {
    // Suffix [7, 1, 2]: [1, 2] occurs at 0 and 5, [7, 1, 2] only at 4.
    let t = [1, 2, 9, 9, 7, 1, 2, 3, 8, 7, 1, 2];
    assert_eq!(longest_match(&t, 2), Some(Match { len: 3, follow: 7 }));
    // Equally long: the most recent.
    let t = [1, 2, 3, 1, 2, 4, 1, 2];
    assert_eq!(longest_match(&t, 2), Some(Match { len: 2, follow: 5 }));
    // Too short a match, no match, too short a sequence.
    assert_eq!(longest_match(&[1, 2, 3, 1, 2], 3), None);
    assert_eq!(longest_match(&[1, 2, 3], 1), None);
    assert_eq!(longest_match(&[1, 1], 2), None);
    assert_eq!(longest_match(&[1, 2, 1], 0), None);
}

#[test]
fn longest_match_may_overlap_and_stops_at_max() {
    // A run of one token: the occurrence ending just before the last token overlaps the suffix.
    let run = vec![5u32; 200];
    assert_eq!(
        longest_match(&run, 4),
        Some(Match {
            len: MAX_MATCH,
            follow: 199
        })
    );
    // A passage repeated: the match runs back to the passage's length, capped at MAX_MATCH.
    let passage: Vec<u32> = (100..180).collect();
    let mut t = passage.clone();
    t.extend_from_slice(&passage[..70]);
    let m = longest_match(&t, 8).unwrap();
    assert_eq!(
        m,
        Match {
            len: MAX_MATCH,
            follow: 70
        }
    );
}

#[test]
fn lookup_cuts_the_continuation_and_classes_by_length() {
    let ng = MtpNgram::new(2);
    let passage: Vec<u32> = (10..30).collect();
    let mut t = passage.clone();
    t.extend_from_slice(&passage[..10]);
    // A 10-token match: class 2 (>= 4 x 2).
    let c = ng.lookup(&t, 4).unwrap();
    assert_eq!(
        c,
        Candidate {
            class: 2,
            tokens: vec![20, 21, 22, 23]
        }
    );
    // The continuation ends where the sequence does.
    let c = ng.lookup(&[1, 2, 3, 1, 2], 8).unwrap();
    assert_eq!(
        c,
        Candidate {
            class: 0,
            tokens: vec![3, 1, 2]
        }
    );
    // A 4-token match is class 1 (>= 2 x 2, < 4 x 2).
    let t4 = [6, 7, 1, 2, 3, 9, 6, 7, 1, 2];
    assert_eq!(ng.lookup(&t4, 1).unwrap().class, 1);
    assert_eq!(ng.lookup(&t, 0), None);
}

#[test]
fn shadow_learns_while_the_head_drafts() {
    let mut ng = MtpNgram::new(2);
    let before = ng.gain(0, 4);
    // Right for 2 tokens over two steps, then wrong: a chain of 2 accepted out of 4.
    ng.follow(Candidate {
        class: 0,
        tokens: vec![1, 2, 3, 4],
    });
    ng.appended(&[1]);
    assert!(ng.shadow.is_some(), "not resolved yet");
    // A second candidate while one is followed is ignored.
    ng.follow(Candidate {
        class: 1,
        tokens: vec![9],
    });
    ng.appended(&[2, 7, 8]);
    assert!(ng.shadow.is_none());
    let mut want = AdaptiveState::default();
    want.record_chain(4, 4, 2);
    assert_eq!(ng.gain(0, 4), want.gain(4));
    assert_ne!(ng.gain(0, 4), before);
    // Class 1 untouched, no real steps counted.
    assert_eq!(ng.gain(1, 4), AdaptiveState::default().gain(4));
    assert_eq!(ng.counters, SpecCounters::default());
    // All matched: a chain that reached its cap.
    ng.follow(Candidate {
        class: 2,
        tokens: vec![5, 6],
    });
    ng.appended(&[5, 6, 7]);
    assert!(ng.shadow.is_none());
    want = AdaptiveState::default();
    want.record_chain(2, 2, 2);
    assert_eq!(ng.gain(2, 6), want.gain(6));
    // Nothing appended: nothing learned.
    ng.follow(Candidate {
        class: 0,
        tokens: vec![1],
    });
    ng.appended(&[]);
    assert!(ng.shadow.is_some());
}

#[test]
fn a_real_step_replaces_the_shadow() {
    let mut ng = MtpNgram::new(2);
    ng.follow(Candidate {
        class: 0,
        tokens: vec![1, 2, 3],
    });
    ng.record(1, 4, 3);
    assert!(ng.shadow.is_none());
    assert_eq!(
        ng.counters,
        SpecCounters {
            drafted: 4,
            accepted: 3,
            steps: 1
        }
    );
    let mut want = AdaptiveState::default();
    want.record_chain(4, 4, 3);
    assert_eq!(ng.gain(1, 4), want.gain(4));
    assert_eq!(ng.gain(0, 4), AdaptiveState::default().gain(4));
}

/// A depth model whose target step costs `base + row` ms per row and whose pass costs `pass`.
fn costs(base: f64, row: f64, pass: f64, rows: usize) -> DepthModel {
    let mut d = DepthModel::new(true);
    for r in 1..=rows {
        d.record_step(r, base + row * r as f64);
    }
    d.record_passes(1, 1, pass);
    d
}

#[test]
fn choose_takes_the_ngram_where_it_pays() {
    // MoE-like rows (GB10 Flash-Next: ~40 ms + ~6 ms per row, ~2 ms per head pass).
    let d = costs(40.0, 6.0, 2.0, 17);
    let mtp = Choice {
        batch: 1,
        take_in: true,
        mtp_k: 3,
        mtp_gain: 2.6,
        shape: 0,
        avail: 8,
    };
    // A continuation that is almost always right: as deep as offered.
    assert_eq!(choose(&d, mtp, |k| 1.0 + 0.97 * k as f32), Some(8));
    // One that is rarely right past its first token: not worth a step over the head's.
    assert_eq!(choose(&d, mtp, |k| 1.0 + 0.5f32.min(k as f32)), None);
    // A fair continuation of only two tokens against a good head: the head.
    let short = Choice { avail: 2, ..mtp };
    assert_eq!(choose(&d, short, |k| 1.0 + 0.5 * k as f32), None);
    // When the head has nothing (depth 0), a fair continuation is better than nothing.
    let idle = Choice {
        mtp_k: 0,
        mtp_gain: 1.0,
        ..mtp
    };
    assert!(choose(&d, idle, |k| AdaptiveState::default().gain(k)).is_some());
    // Nothing measured: the head.
    assert_eq!(
        choose(&DepthModel::new(true), mtp, |k| 1.0 + k as f32),
        None
    );
}

#[test]
fn choose_cuts_the_depth_where_rows_get_dear() {
    // Cheap rows up to 5, then a jump (another kernel past some row count).
    let mut d = DepthModel::new(false);
    for r in 1..=5 {
        d.record_step(r, 40.0 + r as f64);
    }
    for r in 6..=9 {
        d.record_step(r, 120.0 + r as f64);
    }
    d.record_passes(1, 1, 2.0);
    let c = Choice {
        batch: 1,
        take_in: false,
        mtp_k: 2,
        mtp_gain: 2.0,
        shape: 0,
        avail: 8,
    };
    assert_eq!(choose(&d, c, |k| 1.0 + 0.9 * k as f32), Some(4));
    // A batch verified 7 deep anyway: the n-gram's rows up to 7 cost the same.
    let deep = Choice {
        shape: 7,
        mtp_k: 7,
        mtp_gain: 2.0,
        ..c
    };
    assert_eq!(choose(&d, deep, |k| 1.0 + 0.9 * k as f32), Some(8));
}
