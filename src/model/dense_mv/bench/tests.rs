//! The dense matmul bench's group syntax and its defaults.

use super::*;

#[test]
fn parses_groups() {
    let g = Group::parse("2560:10240+6144").unwrap();
    assert_eq!(
        g,
        Group {
            k: 2560,
            ms: vec![10240, 6144],
            f32: false,
        }
    );
    assert_eq!(g.to_string(), "2560:10240+6144");
    assert_eq!(g.bytes(), 2560 * (10240 + 6144) * 2);
    assert_eq!(Group::parse("6144:2560").unwrap().ms, [2560]);
    let router = Group::parse("f32:2560:512").unwrap();
    assert!(router.f32);
    assert_eq!(router.to_string(), "f32:2560:512");
    assert_eq!(router.bytes(), 2560 * 512 * 4);
    for bad in [
        "2560",
        "2560:",
        "x:64",
        "100:64",
        "2560:100",
        "2560:0",
        "64:32+32+32+32+32",
        "f32:",
        "f32:2560",
        "2560:48",
        "2560:48+6144",
        "2560:6144+80",
        "f32:2560:512+48",
    ] {
        assert!(Group::parse(bad).is_err(), "{bad}");
    }
}

/// Chain weights (16 or 48 rows) ride along after the others, from 9 tokens.
#[test]
fn parses_chain_groups() {
    let g = Group::parse("2560:10240+6144+48+48").unwrap();
    assert_eq!(g.chain(), 2);
    assert_eq!(g.to_string(), "2560:10240+6144+48+48");
    assert!(!g.takes(8) && g.takes(9) && g.takes(16) && !g.takes(17));
    assert_eq!(Group::parse("2560:6144+16").unwrap().chain(), 1);
    let plain = Group::parse("2560:10240+6144").unwrap();
    assert_eq!(plain.chain(), 0);
    assert!(plain.takes(2));
}

#[test]
fn default_groups_fit_the_op() {
    let cfg = Config::default();
    assert!(!cfg.groups.is_empty());
    for g in &cfg.groups {
        assert_eq!(Group::parse(&g.to_string()).as_ref(), Ok(g));
    }
    assert_eq!(VARIANTS[0].kind, Kind::Ggml, "the reference first");
}
