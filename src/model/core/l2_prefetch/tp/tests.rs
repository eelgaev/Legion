//! `l2_prefetch::tp`: its switches, the budget's split and the exchange node.

use super::*;

const MIB: usize = 1 << 20;

#[test]
fn the_switch_takes_mib_or_off() {
    assert_eq!(budget_from(None), DEFAULT_MIB * MIB);
    assert_eq!(budget_from(Some("0")), 0);
    assert_eq!(budget_from(Some(" 8 ")), 8 * MIB);
    assert_eq!(budget_from(Some("20")), 20 * MIB);
    assert_eq!(budget_from(Some("21")), DEFAULT_MIB * MIB);
    assert_eq!(budget_from(Some("-1")), DEFAULT_MIB * MIB);
    assert_eq!(budget_from(Some("x")), DEFAULT_MIB * MIB);
}

#[test]
fn verify_steps_take_it_prompt_steps_do_not() {
    let b = DEFAULT_MIB * MIB;
    // 4 users (32 rows), 16 users (128); not one user (8).
    assert!(takes_tokens(32, b) && takes_tokens(128, b));
    assert!(takes_tokens(MIN_TOKENS, b) && takes_tokens(MAX_TOKENS, b));
    assert!(!takes_tokens(8, b) && !takes_tokens(1, b) && !takes_tokens(0, b));
    assert!(!takes_tokens(MAX_TOKENS + 1, b) && !takes_tokens(4096, b));
    // Off, or less than a chunk.
    assert!(!takes_tokens(128, 0) && !takes_tokens(128, super::super::CHUNK - 1));
}

#[test]
fn the_pace_is_all_at_once_unless_set() {
    assert_eq!(pace_from(None), 0);
    assert_eq!(pace_from(Some("500")), 500);
    assert_eq!(pace_from(Some("1000001")), 0);
    assert_eq!(pace_from(Some("x")), 0);
}

#[test]
fn in_order_each_weight_is_read_whole_before_the_next() {
    let c = super::super::CHUNK;
    // q 12 MiB, k and v 3 MiB: 12 MiB is q's; 16 MiB q, k and 1 MiB of v.
    let qkv = [12 * MIB, 3 * MIB, 3 * MIB];
    assert_eq!(split(&qkv, 12 * MIB, false), vec![12 * MIB, 0, 0]);
    assert_eq!(split(&qkv, 16 * MIB, false), vec![12 * MIB, 3 * MIB, MIB]);
    assert_eq!(split(&qkv, 40 * MIB, false), qkv.to_vec());
    // Whole chunks only: the budget's and every weight's tail is left out.
    assert_eq!(split(&[3 * c + 5], 10 * c + 7, false), vec![3 * c]);
    assert_eq!(split(&[c - 1, 2 * c], 4 * c, false), vec![0, 2 * c]);
    assert_eq!(split(&qkv, 0, false), vec![0, 0, 0]);
}

#[test]
fn together_the_budget_is_shared_evenly() {
    let c = super::super::CHUNK;
    // Gate and up, 31 MiB each: half the budget each.
    let glu = [31 * MIB, 31 * MIB];
    assert_eq!(split(&glu, 12 * MIB, true), vec![6 * MIB, 6 * MIB]);
    // A small weight's leftover goes to the others.
    assert_eq!(
        split(&[MIB, 31 * MIB, 31 * MIB], 13 * MIB, true),
        vec![MIB, 6 * MIB, 6 * MIB]
    );
    // An odd count of chunks: every chunk spent, none twice.
    let s = split(&glu, 5 * c, true);
    assert_eq!(s.iter().sum::<usize>(), 5 * c);
    assert!(s.iter().all(|b| b % c == 0 && *b >= 2 * c));
    // More budget than bytes: each weight whole, no more.
    assert_eq!(split(&[2 * c, 3 * c], 100 * c, true), vec![2 * c, 3 * c]);
    assert_eq!(split(&[c - 1, 3 * c], 100 * c, true), vec![0, 3 * c]);
}

#[test]
fn the_exchange_carries_its_weights_and_their_chunks() {
    use crate::ggml::{Context, external_params, types};
    let c = super::super::CHUNK;
    let ctx = Context::new(16, 0);
    let gate = ctx.new_tensor(types::F32, &[5376, 512]);
    let up = ctx.new_tensor(types::F32, &[5376, 512]);
    let x = ctx.new_tensor(types::F32, &[5376, 128]);
    let partial = ctx.scale(x, 2.0);
    let sum = crate::tp_net::all_reduce_ahead(&ctx, 0, partial, None, &[(gate, 6 * c), (up, c)]);
    let plain = crate::tp_net::all_reduce(&ctx, 0, partial);
    assert_eq!(sum.ne(), partial.ne());
    // SAFETY: nodes of the live context.
    let (n, p) = unsafe { (&*sum.raw(), &*plain.raw()) };
    assert_eq!(n.op, p.op);
    assert!(std::ptr::eq(n.src[0], partial.raw()));
    assert!(std::ptr::eq(n.src[1], gate.raw()));
    assert!(std::ptr::eq(n.src[2], up.raw()));
    assert!(n.src[3].is_null());
    assert_eq!(external_params(n)[..5], [0, 0, 2, 6, 1]);
    // The plain exchange carries nothing else.
    assert!(p.src[1].is_null());
    assert_eq!(external_params(p)[2], 0);
    // With a residual: it comes first, then the weights.
    let res = ctx.new_tensor(types::F32, &[5376, 128]);
    let sum = crate::tp_net::all_reduce_ahead(&ctx, 0, partial, Some(res), &[(gate, 2 * c)]);
    // SAFETY: as above.
    let n = unsafe { &*sum.raw() };
    assert!(std::ptr::eq(n.src[1], res.raw()));
    assert!(std::ptr::eq(n.src[2], gate.raw()));
    assert_eq!(external_params(n)[..4], [0, 1, 1, 2]);
}
