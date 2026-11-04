use super::even_take;

#[test]
fn a_prompt_that_fits_takes_the_step() {
    assert_eq!(even_take(48, 4096, 64), 4096);
    assert_eq!(even_take(4096, 4096, 64), 4096);
    assert_eq!(even_take(8240, 8448, 64), 8448);
}

#[test]
fn a_short_tail_spreads_over_the_steps() {
    // 4096 + 4096 + 48 becomes three steps of about 2747, aligned to 64.
    assert_eq!(even_take(8240, 4096, 64), 2752);
    // The steps after it keep the split: 5488 left in two steps, then the rest in one.
    assert_eq!(even_take(8240 - 2752, 4096, 64), 2752);
    assert_eq!(even_take(8240 - 2 * 2752, 4096, 64), 4096);
}

#[test]
fn whole_steps_stay_whole() {
    assert_eq!(even_take(8192, 4096, 64), 4096);
    assert_eq!(even_take(32768, 4096, 64), 4096);
}

#[test]
fn never_past_the_step_and_never_more_steps() {
    for take in [1usize, 7, 64, 500, 4096] {
        for left in 1..3 * take + 5 {
            let t = even_take(left, take, 64);
            assert!(t >= 1 && t <= take, "left {left} take {take}: {t}");
            if left > take {
                // As many steps as full steps would need.
                let mut rest = left;
                let mut steps = 0;
                while rest > 0 {
                    rest -= even_take(rest, take, 64).min(rest);
                    steps += 1;
                }
                assert_eq!(steps, left.div_ceil(take), "left {left} take {take}");
            }
        }
    }
}

#[test]
fn no_step_allowed_takes_nothing() {
    assert_eq!(even_take(100, 0, 64), 0);
}
