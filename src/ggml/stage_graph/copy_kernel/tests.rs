use super::*;

fn copy(dst: u64, dpitch: u64, src: u64, spitch: u64, width: u64, rows: u64) -> Copy2d {
    Copy2d {
        dst,
        dpitch,
        src,
        spitch,
        width,
        rows,
    }
}

/// The PTX declares the three kernels, for a target every supported GPU runs.
#[test]
fn module_has_the_kernels() {
    assert_eq!(MODULE.entries(), vec![B16, B4, B1]);
    assert_eq!(MODULE.target(), Some(70));
}

/// The widest element every address, the width and (with several rows) the pitches allow.
#[test]
fn element_from_alignment() {
    let p = plan(copy(0x1000, 4096, 0x2000, 8192, 1024, 4)).unwrap();
    assert_eq!((p.entry, p.n), (B16, 64));
    let p = plan(copy(0x1004, 4096, 0x2000, 8192, 1024, 4)).unwrap();
    assert_eq!((p.entry, p.n), (B4, 256));
    let p = plan(copy(0x1000, 4100, 0x2000, 8192, 1024, 4)).unwrap();
    assert_eq!(p.entry, B4, "a pitch that is 4-aligned only");
    let p = plan(copy(0x1000, 4096, 0x2001, 8192, 1024, 4)).unwrap();
    assert_eq!((p.entry, p.n), (B1, 1024));
    let p = plan(copy(0x1000, 4096, 0x2000, 8192, 1022, 4)).unwrap();
    assert_eq!(p.entry, B1, "an odd width");
}

/// One row: the pitches neither limit the element nor are checked against the width.
#[test]
fn one_row_ignores_pitches() {
    let p = plan(copy(0x1000, 0, 0x2000, 3, 256, 1)).unwrap();
    assert_eq!((p.entry, p.n), (B16, 16));
    assert_eq!(p.grid, [1, 1, 1]);
}

/// Empty copies and rows wider than their pitch have no launch.
#[test]
fn refuses_empty_and_overlapping_rows() {
    assert_eq!(plan(copy(0, 16, 0, 16, 0, 4)), None);
    assert_eq!(plan(copy(0, 16, 0, 16, 16, 0)), None);
    assert_eq!(plan(copy(0, 8, 0, 16, 16, 2)), None);
    assert_eq!(plan(copy(0, 16, 0, 8, 16, 2)), None);
}

/// The grid covers a row with up to MAX_X CTAs and the rows with the rest of MAX_CTAS (at least
/// one row a CTA row, at most MAX_Y).
#[test]
fn grid_sizes() {
    let p = plan(copy(0, 0, 0, 0, 16 * 256 * 3, 1)).unwrap();
    assert_eq!(p.grid, [3, 1, 1]);
    assert_eq!(p.block, [THREADS, 1, 1]);
    let p = plan(copy(0, 0, 0, 0, 1 << 30, 1)).unwrap();
    assert_eq!(p.grid, [1024, 1, 1]);
    let p = plan(copy(0, 1 << 20, 0, 1 << 20, 1 << 20, 100)).unwrap();
    assert_eq!(p.grid, [256, 16, 1]);
    let p = plan(copy(0, 64, 0, 64, 64, 1 << 20)).unwrap();
    assert_eq!(p.grid, [1, 4096, 1]);
    let p = plan(copy(0, 1 << 30, 0, 1 << 30, 1 << 30, 7)).unwrap();
    assert_eq!(p.grid, [1024, 4, 1]);
}
