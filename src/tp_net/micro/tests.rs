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
    };
    assert_eq!(p.list(), [3, 1, 4096, 1, 1]);
}
