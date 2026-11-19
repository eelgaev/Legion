use super::*;

fn group(slot: usize, n_seqs: usize) -> GroupShape {
    GroupShape {
        slot,
        n_seqs,
        n_tok: 1,
        n_kv: 0,
        table: 4,
        ring: 0,
        logits: vec![1; n_seqs],
        last_only: true,
        verify: false,
        n_pool: 0,
        n_new: 0,
        pick_all: false,
        kv_slots: Vec::new(),
        par: false,
        pair: false,
        head_half: 0,
        img: 0,
        spans: false,
        captures: Vec::new(),
    }
}

/// A decode step stages from the lend's floor on, counting every group's sequences.
#[test]
fn decode_steps_stage_from_the_floor() {
    assert!(!may_stage(&[group(0, 4)], 9));
    assert!(!may_stage(&[group(0, 4), group(4, 4)], 9));
    assert!(may_stage(&[group(0, 4), group(4, 5)], 9));
    assert!(may_stage(&[group(0, 9)], 9));
    assert!(!may_stage(&[group(0, 16)], 128));
    // No lend floor: nothing stages.
    assert!(!may_stage(&[group(0, 64)], i64::MAX));
}

/// A short lend's change or a restore takes only the graphs that may stage; the others stay
/// kept, most recent last as before, and the taken shapes are built again on their next step.
#[test]
fn only_graphs_that_may_stage_are_taken() {
    let mut kept: Vec<(Vec<GroupShape>, u32)> = vec![
        (vec![group(0, 2)], 0),
        (vec![group(0, 12)], 1),
        (vec![group(0, 4)], 2),
        (vec![group(0, 8), group(8, 8)], 3),
    ];
    let gone = take_staging(&mut kept, 9);
    assert_eq!(gone.iter().map(|g| g.1).collect::<Vec<_>>(), [1, 3]);
    assert_eq!(kept.iter().map(|g| g.1).collect::<Vec<_>>(), [0, 2]);
    // The taken shapes are no longer kept: a step of either builds its graph again.
    for (shape, _) in &gone {
        assert!(!kept.iter().any(|(s, _)| s == shape));
    }
    // Again: nothing left to take.
    assert!(take_staging(&mut kept, 9).is_empty());
    assert_eq!(kept.len(), 2);
}

/// The 4-sequence decode steps of a default one-GPU setup stay kept through a short lend
/// (floor `short_stage_min`): no rebuild.
#[test]
fn small_decode_steps_stay_kept_through_a_short_lend() {
    let floor = crate::residency::experts::short_stage_min();
    let mut kept: Vec<(Vec<GroupShape>, ())> = (1..=4).map(|n| (vec![group(0, n)], ())).collect();
    assert!(take_staging(&mut kept, floor).is_empty());
    assert_eq!(kept.len(), 4);
}
