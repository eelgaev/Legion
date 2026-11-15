use super::*;

#[test]
fn the_switch_is_off_only_for_zero() {
    assert!(switch_from(None));
    assert!(switch_from(Some("1")));
    assert!(!switch_from(Some("0")));
    assert!(!switch_from(Some(" 0 ")));
}
