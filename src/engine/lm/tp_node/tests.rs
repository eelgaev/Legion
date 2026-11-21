use super::*;

fn step(back: u32, layer: u32, start: u64, len: u64, check: u64) -> RowsPart {
    RowsPart::Step {
        back,
        layer,
        start,
        len,
        check,
    }
}

#[test]
fn mtp_rows_come_from_the_request_and_the_last_steps() {
    let last = vec![(0..10).map(|i| i as f32).collect::<Vec<f32>>()];
    let before = vec![vec![-1.0f32; 4], vec![7.0f32, 8.0, 9.0]];
    let steps = [last, before];
    let h = [0.5f32, 0.25, -1.0];
    let parts = [
        RowsPart::Sent { len: 2 },
        step(0, 0, 3, 4, crate::node::rows_check(&steps[0][0][3..7])),
        step(1, 1, 1, 2, crate::node::rows_check(&steps[1][1][1..])),
        RowsPart::Sent { len: 1 },
        step(0, 0, 0, 10, 0),
    ];
    let got = rows_of(&parts, &h, &steps).unwrap();
    assert_eq!(
        got,
        vec![
            &h[..2],
            &steps[0][0][3..7],
            &steps[1][1][1..],
            &h[2..],
            &steps[0][0][..]
        ]
    );
}

#[test]
fn mtp_rows_outside_the_steps_or_the_request_fail() {
    let steps = [vec![vec![1.0f32; 8]], vec![]];
    let h = [2.0f32; 3];
    // Past the captured rows, another layer, an overflowing range, a step without rows, a step
    // not kept.
    for p in [
        step(0, 0, 5, 4, 0),
        step(0, 1, 0, 1, 0),
        step(0, 0, u64::MAX, 2, 0),
        step(1, 0, 0, 1, 0),
        step(2, 0, 0, 1, 0),
    ] {
        assert!(rows_of(&[p], &[], &steps).is_err(), "{p:?}");
    }
    // More sent than the request holds, or fewer used.
    assert!(rows_of(&[RowsPart::Sent { len: 4 }], &h, &steps).is_err());
    assert!(rows_of(&[RowsPart::Sent { len: 2 }], &h, &steps).is_err());
    // No step yet.
    assert!(rows_of(&[step(0, 0, 0, 1, 0)], &[], &[vec![], vec![]]).is_err());
}

#[test]
fn mtp_rows_whose_bits_differ_from_rank_0s_fail() {
    let steps = [vec![vec![0.0f32, 1.0, 2.0]], vec![]];
    let theirs = crate::node::rows_check(&[-0.0, 1.0, 2.0]);
    let e = rows_of(&[step(0, 0, 0, 3, theirs)], &[], &steps).unwrap_err();
    assert!(e.contains("differ from rank 0's"), "{e}");
}

#[test]
fn a_take_ins_device_rows_are_its_last_part_only() {
    let dev = RowsPart::Device { start: 7, rows: 3 };
    let parts = [RowsPart::Sent { len: 2 }, dev];
    let (before, on) = split_device(&parts).unwrap();
    assert_eq!(before, &parts[..1]);
    assert_eq!(on, Some((7, 3)));
    // None: all parts go to `rows_of`.
    let host = [RowsPart::Sent { len: 2 }, step(0, 0, 0, 1, 0)];
    assert_eq!(split_device(&host).unwrap(), (&host[..], None));
    assert_eq!(split_device(&[]).unwrap(), (&[][..], None));
    // Anywhere but last, `rows_of` refuses it.
    let steps = [vec![vec![0.0f32; 4]], vec![]];
    let e = rows_of(&[dev, RowsPart::Sent { len: 0 }], &[], &steps).unwrap_err();
    assert!(e.contains("last part"), "{e}");
}

fn take_in(seq: u64, off: u32, start: u32, ran: u32) -> crate::node::WireDraft {
    crate::node::WireDraft {
        seq,
        base: start + ran,
        tokens: vec![1],
        start,
        ran,
        off,
        max: 0,
        blocks: Vec::new(),
        ring: 0,
    }
}

#[test]
fn a_draft_round_reading_rows_its_step_skipped_is_named() {
    // A 32K prompt's second 16K chunk: rank 0 read back from row 12,288 on.
    let reqs = [take_in(4, 0, 16_384, 16_384)];
    // This rank's drafter reads the same rows (the same ring): nothing to say.
    assert_eq!(first_unread(&reqs, 12_288, |_| 12_288), None);
    assert_eq!(first_unread(&reqs, 12_288, |_| 14_336), None);
    // A bigger ring here reads rows the step did not read back.
    assert_eq!(first_unread(&reqs, 12_288, |_| 8_192), Some(4));
    // A chunk the ring overwrites: nothing read back, nothing taken in.
    let first = [take_in(4, 0, 0, 16_384)];
    assert_eq!(first_unread(&first, 16_384, |r| r.ran as usize), None);
    // Every row read back (decode steps, or the switch off): never.
    assert_eq!(first_unread(&reqs, 0, |_| 0), None);
    // A request that takes nothing in reads no rows.
    let none = [take_in(9, 0, 16_384, 0)];
    assert_eq!(first_unread(&none, 12_288, |_| 0), None);
}

#[test]
fn a_step_hands_back_the_rows_no_request_names() {
    let rows = |x: f32| vec![vec![x; 4], vec![x; 2]];
    let mut r = FollowRows::default();
    // MTP passes name the last two steps' rows: the step before the last's go back.
    r.host[0] = rows(1.0);
    assert!(r.start_step(false).is_empty());
    r.host[0] = rows(2.0);
    assert_eq!(r.start_step(false), rows(1.0));
    assert_eq!(r.host[1], rows(2.0));
    // A draft round reads the last step's only (`--tp-draft-split`): those go back too.
    r.host[0] = rows(3.0);
    let gone = r.start_step(true);
    assert_eq!(gone, [rows(3.0), rows(2.0)].concat());
    assert!(r.host.iter().all(Vec::is_empty));
    r.host[0] = rows(4.0);
    assert_eq!(r.start_step(true), rows(4.0));
    assert!(r.host.iter().all(Vec::is_empty));
}

#[test]
fn a_released_sequence_leaves_the_held_rounds() {
    let step = |x: f32| vec![vec![x; 4]];
    let mut held: Vec<HeldRound> = vec![
        (
            vec![take_in(1, 0, 4_096, 4_096), take_in(2, 4_096, 0, 9)],
            step(1.0),
        ),
        (vec![take_in(1, 0, 8_192, 9)], step(2.0)),
    ];
    // A sequence no round names: nothing changes.
    release_held(&mut held, 7);
    assert_eq!(held.len(), 2);
    // Its requests go, and the round it alone was in.
    release_held(&mut held, 1);
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0.len(), 1);
    assert_eq!(held[0].0[0].seq, 2);
    assert_eq!(held[0].1, step(1.0));
    release_held(&mut held, 2);
    assert!(held.is_empty());
}
