use super::*;

fn at(v: &[f32]) -> (usize, usize) {
    (v.as_ptr() as usize, v.len())
}

fn check(v: &[f32]) -> u64 {
    if cfg!(debug_assertions) {
        crate::node::rows_check(v)
    } else {
        0
    }
}

#[test]
fn rows_of_the_last_steps_are_named_others_sent() {
    let layer0: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let layer1: Vec<f32> = (0..12).map(|i| -(i as f32)).collect();
    let before: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let pending = vec![7.5f32; 4];
    let last = [at(&layer0), at(&layer1)];
    let prev = [at(&before)];
    let mut h = Vec::new();
    let parts = rows_parts(
        &[&last, &prev],
        &[
            &pending,
            &layer0[8..36],
            &[],
            &before[4..12],
            &layer1[4..],
            &layer0[..],
        ],
        &mut h,
    );
    let step = |back, layer, start, v: &[f32]| RowsPart::Step {
        back,
        layer,
        start,
        len: v.len() as u64,
        check: check(v),
    };
    assert_eq!(
        parts,
        vec![
            RowsPart::Sent { len: 4 },
            step(0, 0, 8, &layer0[8..36]),
            RowsPart::Sent { len: 0 },
            step(1, 0, 4, &before[4..12]),
            step(0, 1, 4, &layer1[4..]),
            step(0, 0, 0, &layer0),
        ]
    );
    assert_eq!(h, pending);
}

#[test]
fn without_a_step_every_part_is_sent() {
    let rows = [1.0f32, 2.0, 3.0];
    let mut h = vec![9.0];
    let parts = rows_parts(&[&[], &[]], &[&rows[1..], &rows[..1]], &mut h);
    assert_eq!(
        parts,
        vec![RowsPart::Sent { len: 2 }, RowsPart::Sent { len: 1 }]
    );
    assert_eq!(h, vec![9.0, 2.0, 3.0, 1.0]);
}

/// One rank's control link over a cycle of 16-user decoding with `--tp-head-split` and
/// `--tp-mtp-chain`: the verify step (its answer read by the split head's join), the commit's
/// mirror, the draft round, and the next verify step's send. Returns the answers read before
/// that send, oldest first.
fn read_before_next_step(ahead: bool) -> Vec<&'static str> {
    struct Link {
        owed: Vec<&'static str>,
        sent: usize,
        read: Vec<&'static str>,
        ahead: bool,
    }
    impl Link {
        fn run(&mut self, name: &'static str) {
            self.read.clear();
            if let Some(left) = settle_left(self.owed.len(), self.sent, self.ahead) {
                let n = self.owed.len() - left;
                self.read.extend(self.owed.drain(..n));
            }
            self.sent = 1;
            self.owed.push(name);
        }
    }
    let mut l = Link {
        owed: Vec::new(),
        sent: 0,
        read: Vec::new(),
        ahead,
    };
    l.run("verify");
    // The join reads every answer.
    l.owed.clear();
    // The commit's mirror.
    l.owed.push("commit");
    l.sent += 1;
    l.run("round");
    l.run("next verify");
    l.read
}

#[test]
fn a_step_goes_out_without_waiting_for_the_draft_round() {
    // Before: both answers, the round's too (rank 1 sends it as it ends the round, so the
    // step went out a round trip later).
    assert_eq!(read_before_next_step(false), vec!["commit", "round"]);
    // Now only the commit's, which rank 1 sent before it started the round.
    assert_eq!(read_before_next_step(true), vec!["commit"]);
}

#[test]
fn the_answers_left_unread_are_the_last_steps_and_after() {
    // Nothing owed beyond the last step and the mirrors after it: nothing to read.
    assert_eq!(settle_left(0, 0, true), None);
    assert_eq!(settle_left(1, 1, true), None);
    assert_eq!(settle_left(3, 3, true), None);
    // Older ones are read, those left.
    assert_eq!(settle_left(2, 1, true), Some(1));
    assert_eq!(settle_left(5, 2, true), Some(2));
    // A wait that read everything since leaves fewer owed than sent.
    assert_eq!(settle_left(0, 4, true), None);
    // The way before: every answer once two are owed.
    assert_eq!(settle_left(1, 0, false), None);
    assert_eq!(settle_left(2, 1, false), Some(0));
    assert_eq!(settle_left(7, 7, false), Some(0));
}
