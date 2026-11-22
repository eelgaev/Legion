use super::*;

#[test]
fn halves_are_on_unless_the_debug_variable_is_0() {
    assert!(on_of(None));
    assert!(on_of(Some("1")));
    assert!(!on_of(Some("0")));
    assert_eq!(TP_HALF_MIN, 1024);
    assert_eq!(FFN, ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
}

/// Four FFN parts by default; `LEGION_DEBUG_QWEN35_TP_PARTS=2` the halves, at most four.
#[test]
fn parts_default_to_three_and_stay_in_range() {
    assert_eq!(parts_of(None), 4);
    assert_eq!(parts_of(Some("3")), 3);
    assert_eq!(parts_of(Some("2")), 2);
    assert_eq!(parts_of(Some(" 4 ")), 4);
    assert_eq!(parts_of(Some("1")), 2);
    assert_eq!(parts_of(Some("9")), TP_PARTS_MAX);
    assert_eq!(parts_of(Some("x")), 4);
}

/// Half 0 whole, half 1 with its smallest part last; the parts cover the step once, in order.
#[test]
fn plan_splits_the_second_half_smallest_last() {
    assert_eq!(plan(8192, 2), [(0, 4096), (4096, 4096)]);
    assert_eq!(plan(8192, 3), [(0, 4096), (4096, 3072), (7168, 1024)]);
    // An 8K step's half 1 in three would end in 512 tokens: allowed, the floor.
    assert_eq!(
        plan(8192, 4),
        [(0, 4096), (4096, 2560), (6656, 1024), (7680, 512)]
    );
    assert_eq!(plan(16384, 3), [(0, 8192), (8192, 6144), (14336, 2048)]);
    assert_eq!(
        plan(16384, 4),
        [(0, 8192), (8192, 5120), (13312, 2048), (15360, 1024)]
    );
    // Odd steps: half 1 has the extra token, in its first part.
    assert_eq!(plan(8195, 3), [(0, 4097), (4097, 3074), (7171, 1024)]);
    // Too short for a part of `TP_PART_MIN` tokens: the halves.
    assert_eq!(plan(4094, 3), [(0, 2047), (2047, 2047)]);
    assert_eq!(plan(4096, 3), [(0, 2048), (2048, 1536), (3584, 512)]);
    assert_eq!(plan(6000, 4), [(0, 3000), (3000, 2250), (5250, 750)]);
    for n in [
        2048, 2049, 4095, 4096, 6143, 8191, 8192, 12345, 16384, 16385,
    ] {
        for p in 2..=TP_PARTS_MAX {
            let ps = plan(n, p);
            assert!(
                ps.len() >= 2 && ps.len() <= p.max(2),
                "n {n} parts {p}: {ps:?}"
            );
            assert_eq!(ps[0], (0, n / 2), "n {n} parts {p}");
            let mut at = 0;
            for &(a, c) in &ps {
                assert_eq!(a, at, "n {n} parts {p}: {ps:?}");
                assert!(c >= TP_PART_MIN, "n {n} parts {p}: {ps:?}");
                at += c;
            }
            assert_eq!(at, n, "n {n} parts {p}");
            // Half 1's parts shrink towards its end.
            assert!(ps[1..].windows(2).all(|w| w[0].1 >= w[1].1), "{ps:?}");
        }
    }
}

/// The halves' order as before (#1133 r1), and with half 1 in more parts every part's start
/// after its half's mixer sum, at most two exchanges in flight, finishes in start order.
#[test]
fn order_keeps_two_in_flight_and_the_starts_order() {
    use Op::*;
    assert_eq!(
        order(&[0, 1]),
        [
            MixerFinish(0),
            Ffn(0),
            MixerFinish(1),
            Ffn(1),
            FfnFinish(0),
            FfnFinish(1)
        ]
    );
    assert_eq!(
        order(&[0, 1, 1]),
        [
            MixerFinish(0),
            Ffn(0),
            MixerFinish(1),
            Ffn(1),
            FfnFinish(0),
            Ffn(2),
            FfnFinish(1),
            FfnFinish(2)
        ]
    );
    for half in [
        vec![0, 1],
        vec![0, 1, 1],
        vec![0, 1, 1, 1],
        vec![0, 0, 1, 1],
    ] {
        let ops = order(&half);
        // Replay: the two mixer starts first, then each part's start right after it.
        let mut flight = std::collections::VecDeque::from([MixerFinish(0), MixerFinish(1)]);
        let mut mixed = [false; 2];
        let mut ran = vec![false; half.len()];
        for op in &ops {
            match *op {
                Ffn(j) => {
                    assert!(mixed[half[j]], "{half:?}: part {j} before its mixer sum");
                    assert!(flight.len() <= 1, "{half:?}: a third exchange in flight");
                    ran[j] = true;
                    flight.push_back(FfnFinish(j));
                }
                f => {
                    assert_eq!(flight.pop_front(), Some(f), "{half:?}: finish order");
                    if let MixerFinish(h) = f {
                        mixed[h] = true;
                    }
                }
            }
        }
        assert!(
            flight.is_empty() && ran.iter().all(|&r| r),
            "{half:?}: {ops:?}"
        );
    }
}

/// The FP8-MLP layers take as many parts as the others by default; `0` keeps them whole,
/// `2` to `TP_PARTS_MAX` caps them on their own (#1133 r3).
#[test]
fn fp8_parts_follow_the_others_unless_set() {
    assert_eq!(fp8_parts_of(None, 4), Some(4));
    assert_eq!(fp8_parts_of(None, 2), Some(2));
    assert_eq!(fp8_parts_of(Some("0"), 4), None);
    assert_eq!(fp8_parts_of(Some(" 0 "), 4), None);
    assert_eq!(fp8_parts_of(Some("2"), 4), Some(2));
    assert_eq!(fp8_parts_of(Some("3"), 2), Some(3));
    assert_eq!(fp8_parts_of(Some("1"), 4), Some(2));
    assert_eq!(fp8_parts_of(Some("9"), 4), Some(TP_PARTS_MAX));
    assert_eq!(fp8_parts_of(Some("x"), 3), Some(3));
}
