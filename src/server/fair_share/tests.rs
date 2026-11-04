use super::*;

#[test]
fn off_gives_no_key() {
    let caller = Some(CallerKey(ShareKey::of("sk-a")));
    assert_eq!(share_key(FairShare::Off, caller, Some("alice")), None);
}

#[test]
fn api_key_mode_takes_the_caller_not_the_user() {
    let caller = Some(CallerKey(ShareKey::of("sk-a")));
    assert_eq!(
        share_key(FairShare::ApiKey, caller, Some("alice")),
        Some(ShareKey::of("sk-a"))
    );
    assert_eq!(share_key(FairShare::ApiKey, None, Some("alice")), None);
}

#[test]
fn user_mode_takes_the_user_not_the_caller() {
    let caller = Some(CallerKey(ShareKey::of("sk-a")));
    assert_eq!(
        share_key(FairShare::User, caller, Some("alice")),
        Some(ShareKey::of("alice"))
    );
    assert_eq!(share_key(FairShare::User, caller, Some("")), None);
    assert_eq!(share_key(FairShare::User, caller, None), None);
}
