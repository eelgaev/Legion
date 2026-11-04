use super::*;

#[test]
fn offset_inside_a_group_is_aligned_and_leaves_a_full_piece() {
    // A group of 1000 tokens after 4000: a save point at 4096 is 96 in, not a multiple of 64.
    assert_eq!(offset(4000, 1000, 4096), None);
    // After 4032 it is 64 in.
    assert_eq!(offset(4032, 1000, 4096), Some(64));
    // Too close to the start or the end of the group.
    assert_eq!(offset(4096, 1000, 4096), None);
    assert_eq!(offset(0, 4096 + 32, 4096), None);
    assert_eq!(offset(0, 4096 + 64, 4096), Some(4096));
    // Before the group or past it.
    assert_eq!(offset(5000, 1000, 4096), None);
    assert_eq!(offset(0, 4000, 4096), None);
}

#[test]
fn offset_at_the_group_end_needs_no_alignment() {
    assert_eq!(offset(4000, 96, 4096), Some(96));
    assert_eq!(offset(4000, 100, 4100), Some(100));
    // A group shorter than one piece is never split, not even at its end.
    assert_eq!(offset(4090, 6, 4096), None);
}

#[test]
fn pieces_split_at_inner_captures() {
    let c = |at| Capture { at, saved: 0 };
    assert_eq!(pieces(&[], 300), vec![(0, 300)]);
    assert_eq!(pieces(&[c(128)], 300), vec![(0, 128), (128, 172)]);
    assert_eq!(
        pieces(&[c(64), c(128), c(128), c(300)], 300),
        vec![(0, 64), (64, 64), (128, 172)]
    );
}
