use super::{
    buffers_for, decode_planned, encode_planned, fold_need, merge_planned, reduce_reserve,
};

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

/// The buffers sized at load (#1133) for the exchanges of the graphs measured there: a long
/// prompt step's micro-batch halves (4 sets, an odd step's larger half reserved), the plain
/// all-reduce of its whole-step layers (2 sets) and a decode step's small exchange. In any
/// order they fold to the largest bytes rounded to a MiB and the most sets, and on those
/// buffers no exchange of the step grows them, in flight beside another or not.
#[test]
fn planned_buffers_fit_every_exchange_of_the_steps() {
    let (row, n, eb) = (5120, 16383, 2);
    let (c0, c1) = (n / 2, n - n / 2);
    // (bytes, want, sets) per exchange.
    let half0 = (row * c0 * eb, reduce_reserve(row, c0, n) * eb, 4);
    let half1 = (row * c1 * eb, reduce_reserve(row, c1, n) * eb, 4);
    let whole = (row * n * eb, row * n * eb, 2);
    let decode = (row * 4 * 4, row * 4 * 4, 2);
    let steps = [decode, half0, half1, whole, half0];
    let fold = |order: &[(usize, usize, usize)]| {
        order
            .iter()
            .fold(None, |have, &(b, w, s)| fold_need(have, b, w, s))
    };
    let planned = fold(&steps).unwrap();
    assert_eq!(planned, ((row * n * eb).next_multiple_of(MIB), 4));
    let mut back = steps;
    back.reverse();
    assert_eq!(fold(&back), Some(planned));
    for (b, w, s) in steps {
        assert_eq!(buffers_for(Some(planned), b, w, s, false), Ok(None));
        assert_eq!(buffers_for(Some(planned), b, w, s, true), Ok(None));
    }
    // A step bigger than any measured still grows them (the fallback).
    let longer = row * (n + 4096) * eb;
    assert_eq!(
        buffers_for(Some(planned), longer, longer, 2, false),
        Ok(Some((longer.next_multiple_of(MIB), 4)))
    );
    // Nothing noted: nothing planned.
    assert_eq!(fold(&[]), None);
}

/// The ranks' plans meet at the larger cap and the most sets, and travel as they are.
#[test]
fn planned_buffers_merge_and_travel() {
    let a = Some((160 * MIB, 2));
    let b = Some((96 * MIB, 4));
    assert_eq!(merge_planned(a, b), Some((160 * MIB, 4)));
    assert_eq!(merge_planned(b, a), Some((160 * MIB, 4)));
    assert_eq!(merge_planned(a, None), a);
    assert_eq!(merge_planned(None, b), b);
    assert_eq!(merge_planned(None, None), None);
    for p in [a, b, None, Some((usize::MAX >> 1, 4))] {
        assert_eq!(decode_planned(&encode_planned(p)), Ok(p));
    }
    // Anything else on the data connection: out of step.
    let mut other = encode_planned(a);
    other[0] ^= 1;
    assert!(decode_planned(&other).is_err());
}
