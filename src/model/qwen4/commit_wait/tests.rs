use super::*;

/// Unset or unreadable: rank 0 only; `0`: none; a mask: its ranks.
#[test]
fn the_switch_is_a_mask_of_ranks() {
    assert_eq!(ranks_from(None), DEFAULT_RANKS);
    assert_eq!(ranks_from(Some("junk")), DEFAULT_RANKS);
    assert_eq!(ranks_from(Some("")), DEFAULT_RANKS);
    assert_eq!(ranks_from(Some("0")), 0);
    assert_eq!(ranks_from(Some(" 3 ")), 3);
    assert_eq!(ranks_from(Some("0x2")), 2);
    let on = |mask: u32| (0..3).map(|r| takes_in(mask, r)).collect::<Vec<_>>();
    assert_eq!(on(DEFAULT_RANKS), [true, false, false]);
    assert_eq!(on(0), [false, false, false]);
    assert_eq!(on(3), [true, true, false]);
    assert_eq!(on(2), [false, true, false]);
    assert!(!takes_in(u32::MAX, 32));
}

/// Only verify groups count, and a step with more than a sum can run gets none.
#[test]
fn verify_groups_of_a_step() {
    let step = [(true, 0, 16), (false, 16, 1), (true, 17, 2), (true, 19, 0)];
    assert_eq!(verify_groups(step, 8), [(0, 16), (17, 2)]);
    assert!(verify_groups(step, 1).is_empty());
    assert!(verify_groups([(false, 0, 4)], 8).is_empty());
    assert!(verify_groups(std::iter::empty(), 8).is_empty());
}
