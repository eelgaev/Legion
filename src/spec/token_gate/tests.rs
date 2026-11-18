use super::*;
use crate::spec::TokenDrafter;
use crate::spec::ngram::NgramDrafter;
use crate::spec::suffix::{SuffixCache, SuffixDrafter};

#[test]
fn classes_by_match_length() {
    let got: Vec<usize> = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 1000]
        .iter()
        .map(|&l| class(l))
        .collect();
    assert_eq!(got, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]);
}

fn offer(class: usize, drafts: usize) -> Offer {
    Offer { class, drafts }
}

/// A gate whose steps of `r` rows cost `ms(r)` (each measured a few times).
fn gate(ms: impl Fn(usize) -> f64) -> TokenGate {
    let mut g = TokenGate::default();
    for _ in 0..4 {
        for r in 1..=8 {
            g.record_step(100, r, ms(r));
        }
    }
    g
}

#[test]
fn plain_step_first_then_unmeasured_windows_go_through() {
    let mut g = TokenGate::default();
    assert_eq!(g.choose(100, &[offer(0, 2)]), 0);
    assert_eq!(g.choose(100, &[offer(0, 0)]), 0);
    // Only the plain step measured: a window costs as its nearest measured neighbour.
    g.record_step(100, 1, 16.0);
    g.record_step(100, 1, 16.0);
    assert_eq!(g.choose(100, &[offer(0, 2)]), 2);
    // A batch of two: its own plain step first.
    assert_eq!(g.choose(100, &[offer(0, 2), offer(0, 2)]), 0);
    g.record_step(100, 2, 18.0);
    assert_eq!(g.choose(100, &[offer(0, 2), offer(0, 2)]), 1);
}

#[test]
fn windows_stay_within_max_rows() {
    let mut g = gate(|r| 16.0 + 0.1 * r as f64);
    assert_eq!(g.choose(100, &[offer(0, 5)]), MAX_ROWS - 1);
    assert_eq!(g.choose(100, &[offer(0, 5), offer(0, 5)]), MAX_ROWS / 2 - 1);
    assert_eq!(g.choose(100, &[offer(0, 5); 3]), 0);
}

#[test]
fn cheap_windows_draft_dear_ones_do_not() {
    // A window of 1 + k rows at 1 + 0.2 k plain steps: the prior (0.7 a position) pays.
    let mut g = gate(|r| 16.0 * (1.0 + 0.2 * (r - 1) as f64));
    assert_eq!(g.choose(100, &[offer(0, 2)]), 2);
    // At 1 + 0.7 k the prior's 1.7 tokens for 1.7 steps is no 3% better than a plain step.
    let mut g = gate(|r| 16.0 * (1.0 + 0.7 * (r - 1) as f64));
    assert_eq!(g.choose(100, &[offer(0, 2)]), 0);
}

#[test]
fn acceptance_by_class_decides() {
    let mut g = gate(|r| 16.0 * (1.0 + 0.45 * (r - 1) as f64));
    // Class 0 learns that its drafts fail, class 4 that they hold.
    for _ in 0..200 {
        g.learn(Chain {
            class: 0,
            drafted: 2,
            accepted: 0,
        });
        g.learn(Chain {
            class: 4,
            drafted: 2,
            accepted: 2,
        });
    }
    assert!(g.rate(0) < 0.05 && g.rate(4) > 0.95);
    assert_eq!(g.choose(100, &[offer(0, 2)]), 0);
    assert_eq!(g.choose(100, &[offer(4, 2)]), 2);
    // A batch shares one shape (at most 4 rows, so one draft each): two reliable sequences pay
    // for it, one beside a failing one does not (3 tokens for 4 rows' 37.6 ms against 2 for 2
    // rows' 23.2 ms).
    assert_eq!(g.choose(100, &[offer(4, 2), offer(4, 2)]), 1);
    assert_eq!(g.choose(100, &[offer(0, 2), offer(4, 2)]), 0);
}

#[test]
fn context_buckets_keep_their_own_costs() {
    let mut g = TokenGate::default();
    for _ in 0..4 {
        g.record_step(100, 1, 16.0);
        g.record_step(100, 3, 18.0);
        g.record_step(20_000, 1, 20.0);
        g.record_step(20_000, 3, 60.0);
    }
    assert_eq!(g.choose(100, &[offer(0, 2)]), 2);
    assert_eq!(g.choose(20_000, &[offer(0, 2)]), 0);
}

#[test]
fn verified_and_shadow_chains() {
    let mut s = SeqGate::default();
    assert_eq!(s.verified(2, 1), None);
    s.drafting(3);
    assert_eq!(
        s.verified(2, 1),
        Some(Chain {
            class: 3,
            drafted: 2,
            accepted: 1
        })
    );
    assert_eq!(s.verified(2, 1), None);
    // A shadow ends at the first differing token, or once it is matched in full.
    s.follow(1, &[5, 6, 7]);
    s.follow(2, &[9]);
    assert_eq!(s.appended(&[5]), None);
    assert_eq!(
        s.appended(&[6, 8]),
        Some(Chain {
            class: 1,
            drafted: 3,
            accepted: 2
        })
    );
    s.follow(2, &[9]);
    assert_eq!(
        s.appended(&[9, 1]),
        Some(Chain {
            class: 2,
            drafted: 1,
            accepted: 1
        })
    );
    // Drafting drops a shadow.
    s.follow(2, &[4]);
    s.drafting(0);
    assert_eq!(s.appended(&[4]), None);
}

/// The suffix drafter's proposal, from the raw sequences: for each draft, the longest suffix
/// (up to `depth` tokens) followed by a token somewhere in `corpus` or `tokens`, its most
/// frequent next token (ties to the smaller id), stopping below a 10% share.
fn suffix_by_scan(
    corpus: &[Vec<u32>],
    tokens: &[u32],
    depth: usize,
    max: usize,
) -> (Vec<u32>, usize) {
    let mut seq = tokens.to_vec();
    let (mut out, mut matched) = (Vec::new(), 0);
    for i in 0..max {
        let mut found = None;
        for l in (1..=depth.min(seq.len())).rev() {
            let ctx = &seq[seq.len() - l..];
            let mut counts = std::collections::BTreeMap::<u32, u32>::new();
            for s in corpus.iter().map(|s| &s[..]).chain([tokens]) {
                for p in 0..s.len().saturating_sub(l) {
                    if s[p..p + l] == *ctx {
                        *counts.entry(s[p + l]).or_default() += 1;
                    }
                }
            }
            if !counts.is_empty() {
                found = Some((l, counts));
                break;
            }
        }
        let Some((l, counts)) = found else { break };
        let total: u32 = counts.values().sum();
        let (&tok, &count) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .unwrap();
        if (count as f64) / (total as f64) < 0.1 {
            break;
        }
        if i == 0 {
            matched = l;
        }
        out.push(tok);
        seq.push(tok);
    }
    (out, matched)
}

#[test]
fn suffix_drafts_equal_a_scan_of_the_sequences() {
    let mut x = 12345u64;
    let mut next = |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 33) % m) as u32
    };
    for round in 0..300 {
        let alphabet = 2 + (round % 4) as u64;
        let corpus: Vec<Vec<u32>> = (0..3)
            .map(|_| (0..10 + next(40)).map(|_| next(alphabet)).collect())
            .collect();
        let tokens: Vec<u32> = (0..1 + next(30)).map(|_| next(alphabet)).collect();
        let cache = SuffixCache::new(8, 1000);
        for s in &corpus {
            cache.insert(s);
        }
        let mut d = SuffixDrafter::new(cache);
        let mut out = Vec::new();
        d.propose(&tokens, 4, &mut out);
        assert_eq!(
            (out, d.matched()),
            suffix_by_scan(&corpus, &tokens, 8, 4),
            "round {round}"
        );
    }
}

#[test]
fn drafters_report_their_match() {
    let cache = SuffixCache::new(8, 1000);
    cache.insert(&[10, 11, 12, 13, 14, 15]);
    let mut d = SuffixDrafter::new(cache);
    let mut out = Vec::new();
    d.propose(&[99, 10, 11, 12], 2, &mut out);
    assert_eq!((out.as_slice(), d.matched()), (&[13, 14][..], 3));
    out.clear();
    d.propose(&[77, 78], 2, &mut out);
    assert_eq!((out.len(), d.matched()), (0, 0));

    let mut n = NgramDrafter::default();
    out.clear();
    n.propose(&[1, 2, 3, 4, 1, 2, 3], 2, &mut out);
    assert_eq!((out.as_slice(), n.matched()), (&[4, 1][..], 3));
    out.clear();
    n.propose(&[1, 2, 3], 2, &mut out);
    assert_eq!((out.len(), n.matched()), (0, 0));
}
