use super::{buffers_for, reduce_reserve};

const MIB: usize = 1 << 20;

/// An odd step's halves (#619): 2,049 tokens of 4,096 BF16 values, half 0 1,024 tokens (8 MiB
/// exactly), half 1 1,025 (8 MiB + 8,192 bytes). Half 0's buffers are reserved for half 1, so
/// half 1 starts while half 0 is in flight without growing them.
#[test]
fn odd_halves_reserve_the_larger() {
    let (row, n, eb) = (4096, 2049, 2);
    let (c0, c1) = (n / 2, n - n / 2);
    let (b0, b1) = (row * c0 * eb, row * c1 * eb);
    assert_eq!((b0, b1), (8 * MIB, 8 * MIB + 8192));
    assert_eq!(reduce_reserve(row, c0, n), row * c1);
    assert_eq!(reduce_reserve(row, c1, n), row * c1);
    // Half 0, nothing in flight, no buffers yet: 4 sets, room for half 1.
    let want0 = reduce_reserve(row, c0, n) * eb;
    let g0 = buffers_for(None, b0, want0, 4, false).unwrap().unwrap();
    assert_eq!(g0, (9 * MIB, 4));
    assert!(g0.0 >= b1);
    // Half 1 with half 0 in flight: as they are.
    let want1 = reduce_reserve(row, c1, n) * eb;
    assert_eq!(buffers_for(Some(g0), b1, want1, 4, true), Ok(None));
    // Sized for half 0 alone (as before the fix), half 1 could not start.
    let alone = buffers_for(None, b0, b0, 4, false).unwrap().unwrap();
    assert_eq!(alone, (8 * MIB, 4));
    assert_eq!(buffers_for(Some(alone), b1, want1, 4, true), Err(()));
}

/// Buffers sized for half 0 alone (an earlier, smaller step) grow to the reserve at half 0's
/// start when nothing is in flight; never shrink; and an exchange that fits starts on them as
/// they are while another is in flight even below its reserve.
#[test]
fn buffers_grow_to_the_reserve_when_idle() {
    let have = Some((8 * MIB, 4));
    assert_eq!(
        buffers_for(have, 8 * MIB, 8 * MIB + 8192, 4, false),
        Ok(Some((9 * MIB, 4)))
    );
    assert_eq!(
        buffers_for(have, 8 * MIB, 8 * MIB + 8192, 4, true),
        Ok(None)
    );
    assert_eq!(
        buffers_for(Some((64 * MIB, 4)), 3 * MIB, 5 * MIB, 4, false),
        Ok(None)
    );
    // Plain exchanges reserve their own bytes: they grow only when short, as before.
    assert_eq!(
        buffers_for(Some((8 * MIB, 2)), 8 * MIB, 8 * MIB, 2, false),
        Ok(None)
    );
    assert_eq!(
        buffers_for(Some((8 * MIB, 2)), 8 * MIB + 1, 8 * MIB + 1, 2, false),
        Ok(Some((9 * MIB, 2)))
    );
    // More sets: grown even when the bytes fit; the cap kept.
    assert_eq!(
        buffers_for(Some((16 * MIB, 2)), MIB, MIB, 4, false),
        Ok(Some((16 * MIB, 4)))
    );
    assert_eq!(buffers_for(Some((16 * MIB, 2)), MIB, MIB, 4, true), Err(()));
}
