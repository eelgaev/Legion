use super::*;

/// The scratch of a `gx` × `gy` patch grid as an encode lays it out: its pixels and its
/// rotation table of one row per position on the longer side.
fn grid_scratch(dims: &Dims, gx: usize, gy: usize, patch: usize, d_head: usize) -> usize {
    let n = gx * gy;
    dims.scratch(n, 3 * n * patch * patch, gx.max(gy) * d_head / 2)
        .total
}

/// Every grid of an even width and height within `max_px` pixels' patches.
fn every_grid(max_px: usize, patch: usize) -> Vec<(usize, usize)> {
    let n_max = max_px / (patch * patch);
    let mut grids = Vec::new();
    for gx in (2..=n_max / 2).step_by(2) {
        for gy in (2..=n_max / gx).step_by(2) {
            grids.push((gx, gy));
        }
    }
    grids
}

/// A small tower whose attention dominates its scratch: 6-pixel patches, a width of 16, two
/// heads, 3,364 patches at most. The largest grid's chunk (3,368 keys) floors to 3,200
/// queries, a 56 × 60 grid's (3,360 keys) to 3,328, so that grid needs more scratch than the
/// largest one; the bound must cover it and every other grid.
#[test]
fn scratch_bound_covers_a_smaller_grid_with_a_larger_chunk() {
    let dims = Dims {
        e: 16,
        heads: 2,
        patch_k: 112,
        n_ff: 32,
        merger: 32,
        mw: 32,
        mff: 32,
        n_out: 32,
    };
    let (patch, d_head, max_px) = (6, 8, 3364 * 36);
    let largest = dims.scratch(3364, 3 * max_px, 3364 * d_head / 2);
    let smaller = dims.scratch(3360, 3 * 3360 * 36, 60 * d_head / 2);
    assert_eq!((largest.chunk, smaller.chunk), (3200, 3328));
    assert!(smaller.total > largest.total);
    let bound = dims.scratch_bound(max_px, patch, d_head);
    assert!(bound >= smaller.total);
    for (gx, gy) in every_grid(max_px, patch) {
        let need = grid_scratch(&dims, gx, gy, patch, d_head);
        assert!(need <= bound, "{gx} x {gy}: {need} > {bound}");
    }
}

/// GLM-5.3-Flash's real tower (1024 wide, 16 heads, 14-pixel patches, 8,000 tokens at most):
/// its largest grid needs the most scratch, so the bound is exactly that grid's (what the
/// planner reserved before), and it covers every grid.
#[test]
fn scratch_bound_of_the_real_tower_is_its_largest_grid() {
    let dims = Dims {
        e: 1024,
        heads: 16,
        patch_k: 592,
        n_ff: 4096,
        merger: 4096,
        mw: 4096,
        mff: 10240,
        n_out: 4096,
    };
    let (patch, d_head, max_px) = (14, 64, 8000 * 28 * 28);
    let n_max = max_px / (patch * patch);
    let largest = dims.scratch(n_max, 3 * max_px, n_max * d_head / 2).total;
    let bound = dims.scratch_bound(max_px, patch, d_head);
    assert_eq!(bound, largest);
    assert_eq!(bound, 1_522_176_000);
    for (gx, gy) in every_grid(max_px, patch) {
        let need = grid_scratch(&dims, gx, gy, patch, d_head);
        assert!(need <= bound, "{gx} x {gy}: {need} > {bound}");
    }
}

/// A pixel budget under one patch (the hparams refuse it) still gives a bound, the scratch of
/// one merged patch, and no chunk divides by zero keys.
#[test]
fn scratch_bound_under_one_patch_is_one_merged_patch() {
    let dims = Dims {
        e: 16,
        heads: 2,
        patch_k: 112,
        n_ff: 32,
        merger: 32,
        mw: 32,
        mff: 32,
        n_out: 32,
    };
    assert!(dims.chunk(0, 0) >= 1);
    let bound = dims.scratch_bound(10, 6, 8);
    assert_eq!(bound, dims.scratch_bound(4 * 36, 6, 8));
    assert!(bound >= dims.scratch(4, 3 * 4 * 36, 2 * 8 / 2).total);
}
