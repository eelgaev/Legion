use super::*;
use crate::ggml::Backend;
use crate::vision::tiny;

fn encoder(dir: &Path, deepstack: bool) -> Result<Encoder, VisionError> {
    let path = dir.join("mmproj.gguf");
    tiny::write(&path, 64, 7, deepstack).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    Encoder::load(&path, cpu)
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    tiny::bmp(w, h, |x, y| {
        [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
    })
}

/// A 16 × 16 image grows to 24 × 24 (the fewest pixels, 8 merged patches of 8 × 8 pixels), a
/// 6 × 6 patch grid whose positions are resampled from the 8 × 8 table; a 32 × 32 image is the
/// table's own grid. Each merged 2 × 2 block is one row of the language model's width, the
/// same bits every time, and other pixels give other rows.
#[test]
fn encodes_an_image_to_one_row_per_merged_patch() {
    let dir = tempfile::tempdir().unwrap();
    let mut enc = encoder(dir.path(), false).unwrap();
    assert_eq!(enc.config().n_out, 64);
    let small = enc.encode_bytes(&gradient(16, 16)).unwrap();
    assert_eq!((small.nx, small.ny, small.n_embd), (3, 3, 64));
    assert_eq!(small.data.len(), 9 * 64);
    assert!(small.data.iter().all(|v| v.is_finite()));
    let again = enc.encode_bytes(&gradient(16, 16)).unwrap();
    assert_eq!(small, again);
    let other = enc
        .encode_bytes(&tiny::bmp(16, 16, |x, _| [x as u8 * 16, 0, 255]))
        .unwrap();
    assert_ne!(small.data, other.data);
    let grid = enc.encode_bytes(&gradient(32, 32)).unwrap();
    assert_eq!((grid.nx, grid.ny), (4, 4));
    assert!(grid.data.iter().all(|v| v.is_finite()));
    // A wide image: more columns than rows.
    let wide = enc.encode_bytes(&gradient(64, 16)).unwrap();
    assert!(wide.nx > wide.ny, "{} × {}", wide.nx, wide.ny);
    assert_eq!(wide.row(wide.n_tokens() - 1).len(), 64);
}

/// A projector with deepstack layers is refused when loading, not when encoding.
#[test]
fn deepstack_projectors_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let e = encoder(dir.path(), true).err().expect("refused");
    assert!(e.to_string().contains("deepstack"), "{e}");
}

/// mtmd's position order: each merged 2 × 2 block's four patches together (row-major within the
/// block), blocks row by row; rows, columns, rows, columns.
#[test]
fn patch_positions_are_in_merge_order() {
    let p = patch_positions(4, 2);
    let n = 8;
    assert_eq!(&p[..n], &[0, 0, 1, 1, 0, 0, 1, 1]);
    assert_eq!(&p[n..2 * n], &[0, 1, 0, 1, 2, 3, 2, 3]);
    assert_eq!(&p[2 * n..3 * n], &p[..n]);
    assert_eq!(&p[3 * n..], &p[n..2 * n]);
}
