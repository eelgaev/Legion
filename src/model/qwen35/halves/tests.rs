use super::*;

#[test]
fn halves_are_on_unless_the_debug_variable_is_0() {
    assert!(on_of(None));
    assert!(on_of(Some("1")));
    assert!(!on_of(Some("0")));
    assert_eq!(TP_HALF_MIN, 1024);
    assert_eq!(FFN, ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
}
