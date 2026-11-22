use super::*;

/// Three parts by default; `LEGION_DEBUG_GEMMA4_TP_PARTS=2` the halves, at most three.
#[test]
fn parts_default_to_three_and_stay_in_range() {
    assert_eq!(parts_of(None), 3);
    assert_eq!(parts_of(Some("2")), 2);
    assert_eq!(parts_of(Some(" 3 ")), 3);
    assert_eq!(parts_of(Some("1")), 2);
    assert_eq!(parts_of(Some("9")), TP_PARTS_MAX);
    assert_eq!(parts_of(Some("x")), 3);
    assert_eq!(TP_PART_MIN, 512);
}

/// Half 0 in a third and the rest, half 1 whole and last; the halves with `2` or where the
/// third would be under 512 tokens; the parts cover the step once, in order.
#[test]
fn plan_splits_half_0_in_a_third_and_the_rest() {
    assert_eq!(mixer_plan(4096, 2), [(0, 2048), (2048, 2048)]);
    assert_eq!(mixer_plan(4096, 3), [(0, 682), (682, 1366), (2048, 2048)]);
    assert_eq!(mixer_plan(8192, 3), [(0, 1365), (1365, 2731), (4096, 4096)]);
    // Odd steps: half 1 has the extra token.
    assert_eq!(mixer_plan(4097, 3), [(0, 682), (682, 1366), (2048, 2049)]);
    // A third under 512 tokens: the halves (an odd 3,071-token step has 1,535 in half 0).
    assert_eq!(mixer_plan(3071, 3), [(0, 1535), (1535, 1536)]);
    assert_eq!(mixer_plan(3072, 3), [(0, 512), (512, 1024), (1536, 1536)]);
    assert_eq!(mixer_plan(2048, 3), [(0, 1024), (1024, 1024)]);
    for n in [2048, 2049, 3071, 3072, 4095, 4096, 4097, 6000, 8192, 16385] {
        for p in 2..=TP_PARTS_MAX {
            let ps = mixer_plan(n, p);
            assert!(ps.len() >= 2 && ps.len() <= p, "n {n} parts {p}: {ps:?}");
            assert_eq!(ps[ps.len() - 1], (n / 2, n - n / 2), "n {n} parts {p}");
            let mut at = 0;
            for &(a, c) in &ps {
                assert_eq!(a, at, "n {n} parts {p}: {ps:?}");
                assert!(c >= TP_PART_MIN, "n {n} parts {p}: {ps:?}");
                at += c;
            }
            assert_eq!(at, n, "n {n} parts {p}");
        }
    }
}

/// The halves' order as before #1181; with three parts the first finished before half 1's
/// matmul, never a third exchange in flight, finishes in start order, half 1's not finished.
#[test]
fn order_keeps_two_in_flight_and_each_start_behind_its_matmul() {
    use MixerOp::*;
    assert_eq!(mixer_order(2), [Proj(0), Proj(1), Finish(0)]);
    assert_eq!(
        mixer_order(3),
        [Proj(0), Proj(1), Finish(0), Proj(2), Finish(1)]
    );
    for k in 2..=5 {
        let ops = mixer_order(k);
        let mut flight = std::collections::VecDeque::new();
        let mut done = Vec::new();
        for &op in &ops {
            match op {
                Proj(j) => {
                    assert!(flight.len() < 2, "k {k}: a third in flight: {ops:?}");
                    flight.push_back(j);
                }
                Finish(j) => {
                    assert_eq!(flight.pop_front(), Some(j), "k {k}: finish order");
                    done.push(j);
                }
            }
        }
        assert_eq!(done, (0..k - 1).collect::<Vec<_>>(), "k {k}: {ops:?}");
        assert_eq!(flight, [k - 1], "k {k}: half 1 left in flight");
    }
}
