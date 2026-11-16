use std::time::Duration;

use super::*;

#[test]
fn fresh_plans_follow_deterministic_unless_set() {
    assert!(!fresh_plans(false, None));
    assert!(fresh_plans(true, None));
    assert!(fresh_plans(false, Some("1")));
    assert!(!fresh_plans(true, Some("0")));
    // Anything else: as --deterministic says.
    assert!(fresh_plans(true, Some("yes")));
    assert!(!fresh_plans(false, Some("")));
}

#[test]
fn waves_follow_deterministic_unless_set() {
    assert_eq!(waves(false, 200, None), None);
    assert_eq!(waves(true, 200, None), Some(Duration::from_millis(200)));
    assert_eq!(waves(true, 0, None), None);
    assert_eq!(
        waves(false, 200, Some("150")),
        Some(Duration::from_millis(150))
    );
    assert_eq!(waves(true, 200, Some("0")), None);
    // Not a number: as --deterministic says.
    assert_eq!(
        waves(true, 200, Some("on")),
        Some(Duration::from_millis(200))
    );
    assert_eq!(waves(false, 200, Some("on")), None);
}
