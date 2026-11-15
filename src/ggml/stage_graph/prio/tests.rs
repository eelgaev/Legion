use super::*;

#[test]
fn the_switch_is_on_only_for_one() {
    assert!(!switch_from(None));
    assert!(!switch_from(Some("0")));
    assert!(!switch_from(Some("yes")));
    assert!(switch_from(Some("1")));
    assert!(switch_from(Some(" 1 ")));
}
