//! The node digests' row split.

use super::row_digests;

/// A `[2, 3, 4]` F32 tensor whose token dimension is 1 or 2: equal rows give equal digests,
/// and one changed value moves only its own token's digest.
#[test]
fn rows_split_by_the_token_dimension() {
    let ne = [2i64, 3, 4, 1];
    let nb = [4usize, 8, 24, 96];
    let vals: Vec<f32> = (0..24).map(|i| (i % 6) as f32).collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Dimension 2: every index holds the same 3 rows.
    let d2 = row_digests(&bytes, ne, nb, 8, 2);
    assert_eq!(d2.len(), 4);
    assert!(d2.iter().all(|&d| d == d2[0]));
    // Dimension 1: three different rows.
    let d1 = row_digests(&bytes, ne, nb, 8, 1);
    assert_eq!(d1.len(), 3);
    assert_ne!(d1[0], d1[1]);
    let mut changed = vals.clone();
    changed[2 * 6 + 3] = 9.0; // index 2 of dimension 2, row 1 of dimension 1
    let bytes2: Vec<u8> = changed.iter().flat_map(|v| v.to_le_bytes()).collect();
    let e2 = row_digests(&bytes2, ne, nb, 8, 2);
    for t in 0..4 {
        assert_eq!(e2[t] == d2[t], t != 2, "token {t}");
    }
    let e1 = row_digests(&bytes2, ne, nb, 8, 1);
    for t in 0..3 {
        assert_eq!(e1[t] == d1[t], t != 1, "token {t}");
    }
}
