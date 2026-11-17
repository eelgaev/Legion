//! `l2_prefetch::tp`: its switches and its node.

use super::*;

#[test]
fn the_switch_takes_a_token_count_or_off() {
    assert_eq!(min_tokens_from(None), MIN_TOKENS);
    assert_eq!(min_tokens_from(Some("0")), 0);
    assert_eq!(min_tokens_from(Some(" 1 ")), 1);
    assert_eq!(min_tokens_from(Some("48")), 48);
    assert_eq!(min_tokens_from(Some("65")), MIN_TOKENS);
    assert_eq!(min_tokens_from(Some("-1")), MIN_TOKENS);
    assert_eq!(min_tokens_from(Some("x")), MIN_TOKENS);
}

#[test]
fn verify_steps_over_eight_tokens_take_it_by_default() {
    // 16 users with 2 drafts each: 48 tokens; 4 users: 12; one user: 3.
    assert!(takes_tokens(48, MIN_TOKENS));
    assert!(takes_tokens(12, MIN_TOKENS));
    assert!(takes_tokens(MAX_TOKENS, MIN_TOKENS));
    assert!(!takes_tokens(3, MIN_TOKENS));
    assert!(!takes_tokens(8, MIN_TOKENS));
    // Prompt steps never.
    assert!(!takes_tokens(MAX_TOKENS + 1, MIN_TOKENS));
    assert!(!takes_tokens(4608, MIN_TOKENS));
    // Off.
    assert!(!takes_tokens(48, 0));
    // From one token on.
    assert!(takes_tokens(1, 1) && takes_tokens(3, 1));
}

#[test]
fn the_pace_is_all_at_once_unless_set() {
    assert_eq!(pace_from(None), 0);
    assert_eq!(pace_from(Some("500")), 500);
    assert_eq!(pace_from(Some("1000001")), 0);
    assert_eq!(pace_from(Some("x")), 0);
}

#[test]
fn the_exchange_reads_the_partial_then_the_weights() {
    use crate::ggml::{Context, types};
    let ctx = Context::new(16, 0);
    let down = ctx.new_tensor(types::BF16, &[10240, 320]);
    let up = ctx.new_tensor(types::BF16, &[320, 10240]);
    let x = ctx.new_tensor(types::F32, &[2560, 48]);
    let partial = ctx.scale(x, 2.0);
    let sum = crate::tp_net::all_reduce_ahead(&ctx, 0, partial, &[down, up]);
    let plain = crate::tp_net::all_reduce(&ctx, 0, partial);
    assert_eq!(sum.ne(), partial.ne());
    // SAFETY: nodes of the live context.
    let (src, op, op_plain) = unsafe { ((*sum.raw()).src, (*sum.raw()).op, (*plain.raw()).op) };
    assert_eq!(op, op_plain);
    assert!(std::ptr::eq(src[0], partial.raw()));
    assert!(std::ptr::eq(src[1], down.raw()));
    assert!(std::ptr::eq(src[2], up.raw()));
    assert!(src[3].is_null());
    // The plain exchange carries nothing else.
    // SAFETY: as above.
    assert!(unsafe { (*plain.raw()).src[1] }.is_null());
}
