use super::*;

#[test]
fn groups_are_balanced_larger_first() {
    assert_eq!(group_sizes(16, 2), vec![8, 8]);
    assert_eq!(group_sizes(7, 2), vec![4, 3]);
    assert_eq!(group_sizes(10, 4), vec![3, 3, 2, 2]);
    assert_eq!(group_sizes(3, 3), vec![1, 1, 1]);
    assert_eq!(group_sizes(5, 0), Vec::<usize>::new());
    for n in 1..40 {
        for k in 1..=n {
            let sizes = group_sizes(n, k);
            assert_eq!(sizes.iter().sum::<usize>(), n);
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        }
    }
}

/// Applies `moves` in order to slots holding their own numbers.
fn apply(slots: usize, moves: &[(usize, usize)]) -> Vec<usize> {
    let mut s: Vec<usize> = (0..slots).collect();
    for &(from, to) in moves {
        s[to] = s[from];
    }
    s
}

#[test]
fn compacting_never_overwrites_a_source() {
    for old in [
        vec![0, 1, 2],
        vec![0, 2, 3, 6, 7],
        vec![3, 4, 5],
        vec![1, 5, 9, 10],
    ] {
        let moves = compact_moves(&old);
        assert!(moves.iter().all(|&(f, t)| f != t));
        let s = apply(16, &moves);
        assert_eq!(&s[..old.len()], &old[..], "{old:?}");
    }
    assert!(compact_moves(&[0, 1, 2, 3]).is_empty());
}

#[test]
fn groups_hold_at_least_two_sequences() {
    let m = DEFAULT_MIN_GROUP;
    assert_eq!(m, 2);
    assert_eq!(groups_of(1, 2, m), 0);
    assert_eq!(groups_of(2, 2, m), 0);
    assert_eq!(groups_of(3, 2, m), 0);
    assert_eq!(groups_of(4, 2, m), 2);
    assert_eq!(groups_of(16, 2, m), 2);
    assert_eq!(groups_of(5, 3, m), 2);
    assert_eq!(groups_of(6, 3, m), 3);
    assert_eq!(groups_of(4, 4, m), 2);
    assert_eq!(groups_of(16, 1, m), 0);
    assert_eq!(groups_of(16, 0, m), 0);
}

#[test]
fn groups_of_one_fill_the_stages() {
    // `--pipeline-min-group 1` (#752): four sequences over four stages are four groups.
    assert_eq!(groups_of(4, 4, 1), 4);
    assert_eq!(groups_of(3, 4, 1), 3);
    assert_eq!(groups_of(2, 4, 1), 2);
    assert_eq!(groups_of(16, 4, 1), 4);
    // One sequence never forms groups: C=1 runs whole steps as before.
    assert_eq!(groups_of(1, 4, 1), 0);
    assert_eq!(groups_of(4, 1, 1), 0);
    // 0 counts as 1.
    assert_eq!(groups_of(4, 4, 0), 4);
    // Larger minimums: fewer groups.
    assert_eq!(groups_of(8, 4, 3), 2);
    assert_eq!(groups_of(5, 4, 3), 0);
}

/// #814: groups in flight by default only for Qwen3.8-Flash-Next with MTP over 4 V100s with the
/// expert cache, when `--pipeline-groups` is left out and nothing needs whole steps.
#[test]
fn groups_default_on_four_v100s_with_mtp_only() {
    let on = || GroupsSetup {
        asked: 0,
        auto: true,
        qwen4exp: true,
        mtp: true,
        volta: true,
        cache: true,
        stages: 4,
        slo: false,
        on_demand: false,
    };
    let with = |f: fn(&mut GroupsSetup)| {
        let mut s = on();
        f(&mut s);
        default_groups(s)
    };
    assert_eq!(default_groups(on()), 4);
    // Given: as given (0 or 1 is off).
    assert_eq!(with(|s| s.asked = 2), 2);
    assert_eq!(with(|s| s.asked = 1), 1);
    assert_eq!(with(|s| s.auto = false), 0);
    // Anything else: off.
    assert_eq!(with(|s| s.qwen4exp = false), 0);
    assert_eq!(with(|s| s.mtp = false), 0);
    assert_eq!(with(|s| s.volta = false), 0);
    assert_eq!(with(|s| s.cache = false), 0);
    assert_eq!(with(|s| s.stages = 2), 0);
    assert_eq!(with(|s| s.stages = 3), 0);
    assert_eq!(with(|s| s.slo = true), 0);
    assert_eq!(with(|s| s.on_demand = true), 0);
}
