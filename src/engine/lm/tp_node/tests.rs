use super::*;

fn step(back: u32, layer: u32, start: u64, len: u64, check: u64) -> RowsPart {
    RowsPart::Step {
        back,
        layer,
        start,
        len,
        check,
    }
}

#[test]
fn mtp_rows_come_from_the_request_and_the_last_steps() {
    let last = vec![(0..10).map(|i| i as f32).collect::<Vec<f32>>()];
    let before = vec![vec![-1.0f32; 4], vec![7.0f32, 8.0, 9.0]];
    let steps = [last, before];
    let h = [0.5f32, 0.25, -1.0];
    let parts = [
        RowsPart::Sent { len: 2 },
        step(0, 0, 3, 4, crate::node::rows_check(&steps[0][0][3..7])),
        step(1, 1, 1, 2, crate::node::rows_check(&steps[1][1][1..])),
        RowsPart::Sent { len: 1 },
        step(0, 0, 0, 10, 0),
    ];
    let got = rows_of(&parts, &h, &steps).unwrap();
    assert_eq!(
        got,
        vec![
            &h[..2],
            &steps[0][0][3..7],
            &steps[1][1][1..],
            &h[2..],
            &steps[0][0][..]
        ]
    );
}

#[test]
fn mtp_rows_outside_the_steps_or_the_request_fail() {
    let steps = [vec![vec![1.0f32; 8]], vec![]];
    let h = [2.0f32; 3];
    // Past the captured rows, another layer, an overflowing range, a step without rows, a step
    // not kept.
    for p in [
        step(0, 0, 5, 4, 0),
        step(0, 1, 0, 1, 0),
        step(0, 0, u64::MAX, 2, 0),
        step(1, 0, 0, 1, 0),
        step(2, 0, 0, 1, 0),
    ] {
        assert!(rows_of(&[p], &[], &steps).is_err(), "{p:?}");
    }
    // More sent than the request holds, or fewer used.
    assert!(rows_of(&[RowsPart::Sent { len: 4 }], &h, &steps).is_err());
    assert!(rows_of(&[RowsPart::Sent { len: 2 }], &h, &steps).is_err());
    // No step yet.
    assert!(rows_of(&[step(0, 0, 0, 1, 0)], &[], &[vec![], vec![]]).is_err());
}

#[test]
fn mtp_rows_whose_bits_differ_from_rank_0s_fail() {
    let steps = [vec![vec![0.0f32, 1.0, 2.0]], vec![]];
    let theirs = crate::node::rows_check(&[-0.0, 1.0, 2.0]);
    let e = rows_of(&[step(0, 0, 0, 3, theirs)], &[], &steps).unwrap_err();
    assert!(e.contains("differ from rank 0's"), "{e}");
}
