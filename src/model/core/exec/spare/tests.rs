use super::{SPARE_BYTES, SPARE_MAX_BYTES, pick};

const MIB: usize = 1 << 20;

#[test]
fn nothing_kept_allocates_one() {
    assert_eq!(pick(&[]), None);
}

/// One sequence's verify buffers (6-12 MiB) are too small to be the spare: a step after a long
/// prompt needs more than before it.
#[test]
fn small_buffers_are_not_kept() {
    assert_eq!(pick(&[7 * MIB, 12 * MIB, SPARE_BYTES - 1]), None);
}

/// The last lend's spare, still kept by a verify graph, is the next spare: no new allocation.
#[test]
fn the_last_spare_is_kept_again() {
    assert_eq!(pick(&[9 * MIB, SPARE_BYTES, 12 * MIB]), Some(1));
}

#[test]
fn the_biggest_fitting_one_is_kept() {
    let sizes = [
        SPARE_BYTES,
        100 * MIB,
        SPARE_MAX_BYTES + 1,
        90 * MIB,
        100 * MIB,
    ];
    // Not the one over the cap; of two equal ones the older (it would be evicted first anyway).
    assert_eq!(pick(&sizes), Some(1));
    assert_eq!(pick(&[SPARE_MAX_BYTES]), Some(0));
}

/// A short prompt's buffer is too big to hold through the lend.
#[test]
fn a_prompt_sized_buffer_is_not_kept() {
    assert_eq!(pick(&[SPARE_MAX_BYTES + 1, 1 << 30]), None);
}
