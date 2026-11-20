use super::*;

const B: [usize; 3] = [100, 100, 60];

/// A row of `n` staged experts with ids `first..first + n` (expert `e` = entry `e`), the
/// down kind padded, host addresses `1_000_000 * (k + 1) + e * b`.
fn row(n: usize, first: usize, heat: &[f32]) -> GateRow {
    let kinds = std::array::from_fn(|k| Kind {
        staged: 10_000_000 * (k + 1),
        bytes: B[k],
        pad: if k == 2 { 16 } else { 0 },
    });
    let entries = (0..n)
        .map(|e| Entry {
            expert: e as u32,
            id: first + e,
            src: std::array::from_fn(|k| 1_000_000 * (k + 1) + e * B[k]),
        })
        .collect();
    let end = first + n;
    let fixed = vec![(30_000_000 + end * B[2], 0, 16)];
    GateRow::new(kinds, entries, n, fixed, Vec::new(), end, heat)
}

#[test]
fn guarded_id_zeroed_unless_copied() {
    let mut r = row(3, 5, &[]);
    r.guard = vec![5, 2];
    let pads = |want: &[bool]| -> Vec<usize> {
        let c = r.copies(want, &[], true);
        c.iter()
            .filter(|c| c.1 == 0)
            .map(|c| (c.0 - 30_000_000) / 60)
            .collect()
    };
    // Id 5 is entry 0: copied, no pad there; id 2 (a VRAM slot's neighbour) always.
    assert_eq!(pads(&[true, false, false]), vec![8, 2, 6]);
    assert_eq!(pads(&[false, true, false]), vec![8, 5, 2, 7]);
}

#[test]
fn routed_names_staged_entries_only() {
    let r = row(6, 4, &[]);
    let want = r.routed(&[1, 3, 3, -1, 99, 5]);
    assert_eq!(want, vec![false, true, false, true, false, true]);
}

#[test]
fn all_routed_is_the_full_copy() {
    let r = row(4, 2, &[]);
    let want = vec![true; 4];
    let c = r.copies(&want, &[], true);
    // One joined copy per kind, then the fixed pad; no pad between copied neighbours.
    assert_eq!(
        c,
        vec![
            (10_000_000 + 2 * 100, 1_000_000, 400),
            (20_000_000 + 2 * 100, 2_000_000, 400),
            (30_000_000 + 2 * 60, 3_000_000, 240),
            (30_000_000 + 6 * 60, 0, 16),
        ]
    );
    assert_eq!(copied_bytes(&c), 1040);
}

#[test]
fn holes_get_zeroed_pads_and_split_runs() {
    let r = row(5, 0, &[]);
    let want = vec![true, true, false, true, false];
    let c = r.copies(&want, &[], true);
    let down: Vec<Copy> = c.iter().copied().filter(|c| c.0 >= 30_000_000).collect();
    assert_eq!(
        down,
        vec![
            (30_000_000, 3_000_000, 120),
            (30_000_000 + 3 * 60, 3_000_000 + 3 * 60, 60),
            (30_000_000 + 5 * 60, 0, 16),
            // after entry 1 (id 2 not copied) and after entry 3 (id 4 not copied)
            (30_000_000 + 2 * 60, 0, 16),
            (30_000_000 + 4 * 60, 0, 16),
        ]
    );
    // No pads for the unpadded kinds.
    assert!(c.iter().all(|c| c.1 != 0 || c.0 >= 30_000_000));
}

#[test]
fn ahead_then_rest_copies_each_entry_once() {
    let heat = [0.0, 9.0, 1.0, 8.0, 0.5];
    let r = row(5, 0, &heat);
    // Two entries' bytes: the two hottest, experts 1 and 3.
    let ahead = r.ahead(2 * r.entry_bytes() + 10);
    assert_eq!(ahead, vec![false, true, false, true, false]);
    let first = r.copies(&ahead, &[], false);
    assert!(first.iter().all(|c| c.1 != 0), "no pads ahead");
    let want = r.routed(&[0, 1, 2]);
    let rest = r.copies(&want, &ahead, true);
    let mut moved: Vec<(usize, usize)> = first
        .iter()
        .chain(&rest)
        .filter(|c| c.1 != 0)
        .flat_map(|&(dst, _, n)| {
            let k = dst / 10_000_000 - 1;
            let at = dst % 10_000_000 / B[k];
            (at..at + n / B[k]).map(move |id| (k, id))
        })
        .collect();
    moved.sort_unstable();
    let mut expect: Vec<(usize, usize)> = (0..3)
        .flat_map(|k| [0, 1, 2, 3].map(|id| (k, id)))
        .collect();
    expect.sort_unstable();
    assert_eq!(moved, expect);
    // Entry 3 (ahead, not routed) is followed by unrouted entry 4: padded.
    assert!(rest.contains(&(30_000_000 + 4 * 60, 0, 16)));
}

#[test]
fn runs_stop_at_a_piece() {
    let kinds = [Kind {
        staged: 0,
        bytes: PIECE / 2,
        pad: 0,
    }; 3];
    let entries = (0..3)
        .map(|e| Entry {
            expert: e,
            id: e as usize,
            src: [PIECE * 8 + e as usize * PIECE / 2; 3],
        })
        .collect();
    let r = GateRow::new(kinds, entries, 3, Vec::new(), Vec::new(), 3, &[]);
    let c = r.copies(&[true; 3], &[], true);
    assert_eq!(c.len(), 6);
    assert!(c.iter().all(|c| c.2 <= PIECE));
}

/// [`row`] with the bounded host tier's sources: entry `e` from `5_000_000 * (k + 1) + e * b`,
/// read in `batch[e]` (`None`: a RAM slot).
fn tier_row(n: usize, heat: &[f32], batch: Vec<Option<usize>>) -> GateRow {
    let experts: Vec<u32> = (0..n as u32).collect();
    let src = (0..n)
        .map(|e| std::array::from_fn(|k| 5_000_000 * (k + 1) + e * B[k]))
        .collect();
    let ram = std::array::from_fn(|k| 7_000_000 * (k + 1));
    row(n, 0, heat)
        .with_reads(&experts, src, batch, ram)
        .unwrap()
}

#[test]
fn with_reads_checks_the_entries() {
    let ok = |experts: &[u32], n_src: usize| {
        let r = row(3, 0, &[]);
        r.with_reads(experts, vec![[0; 3]; n_src], vec![None; 3], [0; 3])
            .is_some()
    };
    assert!(!ok(&[0, 1], 3));
    assert!(!ok(&[0, 2, 1], 3));
    assert!(!ok(&[0, 1, 2], 2));
    assert!(ok(&[0, 1, 2], 3));
}

#[test]
fn read_entries_wait_for_their_batch() {
    const CALL: usize = usize::MAX;
    let r = tier_row(5, &[], vec![None, Some(3), Some(3), None, Some(4)]);
    let want = vec![true, true, false, true, false];
    // From RAM: entries 0 and 3, plus the pads of every copied entry, read ones included.
    let ram = r.copies(&want, &[], true);
    let srcs: Vec<usize> = ram.iter().map(|c| c.1).filter(|&s| s != 0).collect();
    let expect: Vec<usize> = (1..=3)
        .flat_map(|k| [0, 3].map(|e| 5_000_000 * k + e * B[k - 1]))
        .collect();
    assert_eq!(srcs, expect);
    let pads: Vec<usize> = ram
        .iter()
        .filter(|c| c.1 == 0)
        .map(|c| (c.0 - 30_000_000) / 60)
        .collect();
    assert_eq!(pads, vec![5, 2, 4]);
    // Read: each batch between its wait and its release, the unrouted batch 4 released too.
    let reads = r.read_copies(&want, &[], |b, rel| (CALL, b * 2 + rel as usize, 0));
    assert_eq!(
        reads,
        vec![
            (CALL, 6, 0),
            (10_000_000 + 100, 5_000_000 + 100, 100),
            (20_000_000 + 100, 10_000_000 + 100, 100),
            (30_000_000 + 60, 15_000_000 + 60, 60),
            (CALL, 7, 0),
            (CALL, 8, 0),
            (CALL, 9, 0),
        ]
    );
    // Without the tier's reads: none.
    let plain = row(3, 0, &[]).read_copies(&[true; 3], &[], |b, _| (CALL, b, 0));
    assert!(plain.is_empty());
}

#[test]
fn a_read_entry_sent_to_a_ram_slot_copies_from_it_after_its_batch() {
    const CALL: usize = usize::MAX;
    let mut r = tier_row(3, &[], vec![None, Some(2), Some(2)]);
    // Entry 2 goes to RAM slot 4; entry 0 (a RAM slot already) keeps its source.
    r.to_slot(2, 4);
    r.to_slot(0, 9);
    let want = vec![true, false, true];
    let ram: Vec<usize> = r.copies(&want, &[], false).iter().map(|c| c.1).collect();
    assert_eq!(ram, vec![5_000_000, 10_000_000, 15_000_000]);
    let reads = r.read_copies(&want, &[], |b, rel| (CALL, b * 2 + rel as usize, 0));
    assert_eq!(
        reads,
        vec![
            (CALL, 4, 0),
            (10_000_000 + 200, 7_000_000 + 4 * 100, 100),
            (20_000_000 + 200, 14_000_000 + 4 * 100, 100),
            (30_000_000 + 120, 21_000_000 + 4 * 60, 60),
            (CALL, 5, 0),
        ]
    );
}

#[test]
fn ahead_takes_ram_entries_only() {
    let heat = [0.0, 9.0, 1.0, 8.0, 0.5];
    let r = tier_row(5, &heat, vec![None, Some(0), None, None, None]);
    // Expert 1 is the hottest but read: experts 3 and 2 instead.
    let ahead = r.ahead(2 * r.entry_bytes() + 10);
    assert_eq!(ahead, vec![false, false, true, true, false]);
    // Every one fits: still not the read one.
    assert_eq!(r.ahead(usize::MAX / 2), vec![true, false, true, true, true]);
}

#[test]
fn env_values() {
    assert_eq!(parse_env(None, 7), 7);
    assert_eq!(parse_env(Some("0".into()), 7), 0);
    assert_eq!(parse_env(Some(" 2048 ".into()), 7), 2048);
    assert_eq!(parse_env(Some("x".into()), 7), 7);
}
