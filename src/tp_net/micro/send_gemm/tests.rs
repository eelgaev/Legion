use super::{FP8_MIN_COLS, covers, on_of};

/// [`covers`] for an FP8 matmul's column ranges.
fn fp8(ranges: &mut [(usize, usize)], n: usize) -> bool {
    covers(ranges, n, false, FP8_MIN_COLS)
}

#[test]
fn on_unless_zero() {
    assert!(on_of(None));
    assert!(on_of(Some("1")));
    assert!(on_of(Some("")));
    assert!(!on_of(Some("0")));
    assert!(!on_of(Some(" 0 ")));
}

#[test]
fn halves_cover_the_columns() {
    // A Qwen3.8-27B 8K step's two halves of the output projection, in either order.
    assert!(fp8(&mut [(4096, 4096), (0, 4096)], 8192));
    // An odd step: the second half one column larger.
    assert!(fp8(&mut [(0, 4096), (4096, 4097)], 8193));
    // A W4A4 matmul runs whole.
    assert!(covers(&mut [(0, 4096)], 4096, true, 0));
}

#[test]
fn gaps_overlaps_and_small_ranges_refused() {
    // A gap, an overlap, a range past the end, a missing half.
    assert!(!fp8(&mut [(0, 4000), (4096, 4096)], 8192));
    assert!(!fp8(&mut [(0, 4100), (4096, 4096)], 8192));
    assert!(!fp8(&mut [(0, 4096), (4096, 4097)], 8192));
    assert!(!fp8(&mut [(0, 4096)], 8192));
    // A range the large-batch GEMM would not run (other kernels, other bits).
    assert!(!fp8(&mut [(0, 64), (64, 4096)], 4160));
    // A W4A4 matmul in two ranges, or only part of it.
    assert!(!covers(&mut [(0, 2048), (2048, 2048)], 4096, true, 0));
    assert!(!covers(&mut [(0, 2048)], 4096, true, 0));
}
