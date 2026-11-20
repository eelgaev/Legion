use super::*;

#[test]
fn parses_a_share_list() {
    let s = parse("0, 1,1/2 ,2").unwrap();
    assert_eq!(
        s,
        vec![
            Share::FULL,
            DEFAULT,
            Share { num: 1, den: 2 },
            Share { num: 2, den: 1 },
        ]
    );
    assert_eq!(
        s.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ["0", "1", "1/2", "2"]
    );
    for bad in ["", "x", "1/0", "1/", "-1", "1p", ","] {
        assert!(parse(bad).is_none(), "{bad:?}");
    }
}

#[test]
fn caps_by_the_sm_count() {
    assert_eq!(Share::FULL.gemm_ctas(48), 0);
    assert_eq!(Share::FULL.small_blocks(48), 0);
    assert_eq!(DEFAULT.gemm_ctas(48), 48);
    assert_eq!(DEFAULT.small_blocks(48), 96);
    let half = parse("1/2").unwrap()[0];
    assert_eq!(half.gemm_ctas(48), 24);
    assert_eq!(half.gemm_ctas(1), 1);
    assert_eq!(parse("1/4").unwrap()[0].gemm_ctas(48), 12);
}

#[test]
fn turns_give_each_encode_its_caps() {
    let mut t = Turns::new(48);
    t.set(parse("1/2").unwrap()[0]);
    let half = Caps {
        gemm: 24,
        small: 48,
    };
    // A set share holds whether or not anything else runs.
    assert_eq!(t.next_caps(false), half);
    assert_eq!(t.next_caps(true), half);
    t.set(Share::FULL);
    assert_eq!(t.next_caps(false), Caps::default());
}

#[test]
fn an_encode_alone_runs_uncapped() {
    let half = Share { num: 1, den: 2 };
    // Nothing set: uncapped alone, the default share beside other work.
    assert_eq!(choose(None, true, true), Share::FULL);
    assert_eq!(choose(None, true, false), DEFAULT);
    // The idle rule off: the default share either way.
    assert_eq!(choose(None, false, true), DEFAULT);
    assert_eq!(choose(None, false, false), DEFAULT);
    // A share set for A/B runs wins in every case, the default share too.
    for idle_full in [false, true] {
        for alone in [false, true] {
            assert_eq!(choose(Some(half), idle_full, alone), half);
            assert_eq!(choose(Some(DEFAULT), idle_full, alone), DEFAULT);
            assert_eq!(choose(Some(Share::FULL), idle_full, alone), Share::FULL);
        }
    }
}

#[test]
fn turns_follow_the_engine_when_nothing_is_set() {
    let mut t = Turns {
        sms: 48,
        shares: Vec::new(),
        idle_full: true,
        encodes: 0,
    };
    let capped = Caps {
        gemm: 48,
        small: 96,
    };
    assert_eq!(t.next_caps(true), Caps::default());
    assert_eq!(t.next_caps(false), capped);
    assert_eq!(t.next_caps(true), Caps::default());
    t.idle_full = false;
    assert_eq!(t.next_caps(true), capped);
}
