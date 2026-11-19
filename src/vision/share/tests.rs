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
