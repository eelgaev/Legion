use super::*;

/// Every column of every rank's share is in exactly one micro-batch's piece, in order.
#[test]
fn natural_pieces_split_every_share() {
    for (n, ranks) in [(4096, 2), (4097, 2), (17, 3), (5, 2), (1, 1), (1024, 4)] {
        for r in 0..ranks {
            let (a, m) = share(n, r, ranks);
            let (b0, c0) = piece(n, r, ranks, 0, false);
            let (b1, c1) = piece(n, r, ranks, 1, false);
            assert_eq!(
                (b0, b0 + c0, b1 + c1),
                (a, b1, a + m),
                "n {n} rank {r}/{ranks}"
            );
        }
    }
}

/// The compact layout puts every rank's piece of a micro-batch after the other, in rank order,
/// and spans the micro-batch.
#[test]
fn compact_pieces_follow_rank_order() {
    for (n, ranks) in [(4096, 2), (4099, 2), (17, 3), (1024, 4)] {
        for j in 0..MICRO {
            let mut at = 0;
            for r in 0..ranks {
                let (b, c) = piece(n, r, ranks, j, true);
                assert_eq!(b, at, "n {n} rank {r}/{ranks} micro {j}");
                assert_eq!(c, piece(n, r, ranks, j, false).1);
                at += c;
            }
            assert_eq!(span_cols(n, ranks, j, true), at);
            assert_eq!(span_cols(n, ranks, j, false), n);
        }
        let both: usize = (0..MICRO).map(|j| span_cols(n, ranks, j, true)).sum();
        assert_eq!(both, n);
    }
}

#[test]
fn params_round_trip() {
    let p = Params {
        comm: 3,
        gather: true,
        n: 4096,
        j: 1,
        compact: true,
        reduce: false,
    };
    assert_eq!(p.list(), [3, 1, 4096, 1, 1, 0]);
    let r = Params { reduce: true, ..p };
    assert_eq!(r.list()[5], 1);
}

/// A micro-batch all-reduce (#619) of each half of a tensor gives the halves of the whole
/// tensor's all-reduce bit for bit: the wire type goes by the whole tensor's values, even
/// where a half alone would travel in F32.
#[test]
fn halves_all_reduce_as_whole() {
    use super::super::Wire;
    let (row, cols) = (5, 12);
    let len = row * cols;
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(10));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                c.set_collectives(Wire::Bf16, 0);
                // BF16 for the whole tensor's exchange, F32 for a half's alone.
                c.set_bf16_min(len);
                assert_eq!((c.wire(len), c.wire(len / 2)), (Wire::Bf16, Wire::F32));
                let src: Vec<f32> = (0..len)
                    .map(|i| 1.0 + (i * 7 + r * 31) as f32 / 1024.0 + r as f32 * 1e-3)
                    .collect();
                let mut whole = vec![0f32; len];
                c.all_reduce_host(&src, &mut whole).unwrap();
                let mut halves = vec![0f32; len];
                let cut = row * (cols / 2);
                let (a, b) = halves.split_at_mut(cut);
                c.all_reduce_host_as(&src[..cut], a, len).unwrap();
                c.all_reduce_host_as(&src[cut..], b, len).unwrap();
                (whole, halves)
            })
        })
        .collect();
    for t in threads {
        let (whole, halves) = t.join().unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&whole), bits(&halves));
    }
}

/// An odd step's halves (#619): half 1 has one more column than half 0; the sums are still the
/// whole all-reduce's bits.
#[test]
fn odd_halves_all_reduce_as_whole() {
    use super::super::Wire;
    let (row, cols) = (5, 13);
    let len = row * cols;
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(10));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                c.set_collectives(Wire::Bf16, 0);
                c.set_bf16_min(len);
                let src: Vec<f32> = (0..len)
                    .map(|i| 1.0 + (i * 7 + r * 31) as f32 / 1024.0 + r as f32 * 1e-3)
                    .collect();
                let mut whole = vec![0f32; len];
                c.all_reduce_host(&src, &mut whole).unwrap();
                let mut halves = vec![0f32; len];
                let cut = row * (cols / 2);
                assert_eq!((cut, len - cut), (row * 6, row * 7));
                let (a, b) = halves.split_at_mut(cut);
                c.all_reduce_host_as(&src[..cut], a, len).unwrap();
                c.all_reduce_host_as(&src[cut..], b, len).unwrap();
                (whole, halves)
            })
        })
        .collect();
    for t in threads {
        let (whole, halves) = t.join().unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&whole), bits(&halves));
    }
}

/// `--tp-overlap` splits only steps whose every micro-batch has at least `MIN_COLS` columns
/// (#1176): a 1,536-token step stays whole, 4,096 and up split.
#[test]
fn short_steps_stay_whole() {
    assert_eq!(MIN_COLS, 1024);
    for (n, ranks, split) in [
        (1536, 2, false),
        (4095, 2, false),
        (4096, 2, true),
        (4608, 2, true),
        (8192, 4, true),
        (8191, 4, false),
    ] {
        assert_eq!(worth(n, ranks), split, "n {n} ranks {ranks}");
    }
}

/// A caller's lower floor (#560 r12, GLM's 256): a long prompt's 1,650- and 2,470-token tail
/// steps split over two ranks, a 1,023-token one does not; above `MIN_COLS` it changes nothing.
#[test]
fn lower_floor_splits_tail_steps() {
    for (n, split) in [(1650, true), (2470, true), (1024, true), (1023, false)] {
        assert_eq!(worth_at(n, 2, 256), split, "n {n}");
    }
    assert!(!worth_at(4095, 2, 4096));
    assert_eq!(worth_at(4095, 2, 2048), worth(4095, 2));
}
