//! The early commit's end test against a token-by-token run of `Active::advance`'s checks,
//! its pairs, and the end save it skips.

use super::*;
use crate::spec::Draft;
use crate::spec::accept::Outcome;

const STOP: u32 = 7;

/// `Active::advance`'s ends for a plain sequence, token by token: a stop token, then the length
/// or context cap after counting the token.
fn advance_ends(
    appended: &[u32],
    ignore_eos: bool,
    mut generated: u32,
    max_tokens: u32,
    mut n_past: usize,
    max_ctx: usize,
) -> bool {
    for &t in appended {
        n_past += 1;
        if !ignore_eos && t == STOP {
            return true;
        }
        generated += 1;
        if at_limit(generated, max_tokens, n_past, max_ctx) {
            return true;
        }
    }
    false
}

#[test]
fn ends_on_matches_advance_token_by_token() {
    let runs: [&[u32]; 6] = [&[], &[1], &[1, 2], &[1, 2, 3], &[STOP], &[1, 2, STOP]];
    for appended in runs {
        for ignore_eos in [false, true] {
            for generated in 0..6 {
                for max_tokens in 0..8 {
                    for start in 0..6 {
                        for max_ctx in 0..9 {
                            let stop = |t: u32| !ignore_eos && t == STOP;
                            assert_eq!(
                                ends_on(appended, stop, generated, max_tokens, start, max_ctx),
                                advance_ends(
                                    appended, ignore_eos, generated, max_tokens, start, max_ctx
                                ),
                                "{appended:?} ignore_eos {ignore_eos} generated {generated} \
                                 max_tokens {max_tokens} start {start} max_ctx {max_ctx}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_last_token_under_the_caps_does_not_end() {
    // 3 generated, 3 more of 7: the sixth; positions 10..13 of a 14-token context.
    assert!(!ends_on(&[1, 2, 3], |_| false, 3, 7, 10, 14));
    // The seventh token reaches max_tokens, position 14 the context.
    assert!(ends_on(&[1, 2, 3, 4], |_| false, 3, 7, 10, 20));
    assert!(ends_on(&[1, 2, 3, 4], |_| false, 3, 9, 10, 14));
    // A stop token ends it; ignored, it does not.
    assert!(ends_on(&[1, STOP], |t| t == STOP, 0, 100, 0, 100));
    assert!(!ends_on(&[1, STOP], |_| false, 0, 100, 0, 100));
}

fn verified(appended: &[u32]) -> Verified {
    Verified {
        drafts: Draft::default(),
        o: Outcome {
            appended: appended.to_vec(),
            accepted: appended.len().saturating_sub(1),
        },
        raw: Vec::new(),
    }
}

#[test]
fn pairs_keep_every_appended_token_from_the_base_slot() {
    let v = [verified(&[5]), verified(&[5, 6, 8]), verified(&[9, 9])];
    assert_eq!(pairs(4, &v), vec![(4, 1), (5, 3), (6, 2)]);
    assert!(pairs(0, &[]).is_empty());
}

#[test]
fn sent_pairs_cover_the_commits_of_the_sequences_left() {
    let sent = [(0, 1), (1, 3), (2, 2)];
    // Every sequence went on.
    assert!(covers(&sent, &sent));
    // The second ended after all: its pair is not among the engine's commits.
    assert!(covers(&sent, &[(0, 1), (2, 2)]));
    assert!(covers(&sent, &[]));
    // Another count or slot is not what went out.
    assert!(!covers(&sent, &[(1, 2)]));
    assert!(!covers(&sent, &[(3, 1)]));
}

#[test]
fn an_early_committed_sequence_that_ends_saves_no_state() {
    // A decode step advanced the state; a verify step left it at the start unless committed.
    assert_eq!(end_state(false, false, 10, 11), Some(11));
    assert_eq!(end_state(false, true, 10, 11), Some(11));
    assert_eq!(end_state(true, false, 10, 13), Some(10));
    assert_eq!(end_state(true, true, 10, 13), None);
}
