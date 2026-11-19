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

#[test]
fn env_values() {
    assert_eq!(parse_env(None, 7), 7);
    assert_eq!(parse_env(Some("0".into()), 7), 0);
    assert_eq!(parse_env(Some(" 2048 ".into()), 7), 2048);
    assert_eq!(parse_env(Some("x".into()), 7), 7);
}
