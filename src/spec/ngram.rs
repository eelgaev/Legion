//! N-gram / prompt-lookup drafter: copies continuations from the sequence itself.
//!
//! Very effective when the model repeats its input (code edits, agent tool loops). The
//! implementation is a plain scan, O(`len` * `max_n`) per call; no index is built.

use super::TokenDrafter;

/// Looks up the suffix as an n-gram in the sequence so far and proposes what followed the most
/// recent earlier occurrence of it. Longer keys are tried first, so a 4-gram match beats a
/// 3-gram match of the same suffix, and so on down to `min_n`.
pub struct NgramDrafter {
    min_n: usize,
    max_n: usize,
}

impl NgramDrafter {
    /// Panics if `min_n == 0` or `min_n > max_n`.
    pub fn new(min_n: usize, max_n: usize) -> Self {
        assert!(min_n > 0 && min_n <= max_n, "require 0 < min_n <= max_n");
        Self { min_n, max_n }
    }
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self::new(2, 4)
    }
}

impl TokenDrafter for NgramDrafter {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        if max == 0 {
            return;
        }
        let len = tokens.len();
        for n in (self.min_n..=self.max_n).rev() {
            if len <= n {
                continue;
            }
            let key = &tokens[len - n..];
            // Most recent earlier occurrence: the suffix itself is not a candidate, so
            // candidate starts run only up to `len - n` (exclusive).
            let Some(start) = (0..len - n).rev().find(|&s| tokens[s..s + n] == *key) else {
                continue;
            };
            let follow = start + n;
            out.extend_from_slice(&tokens[follow..(follow + max).min(len)]);
            return;
        }
    }
    // `accepted` stays the default no-op: the drafter is stateless.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn propose(tokens: &[u32], max: usize) -> Vec<u32> {
        let mut d = NgramDrafter::default();
        let mut out = Vec::new();
        d.propose(tokens, max, &mut out);
        out
    }

    #[test]
    fn copies_continuation_after_shortest_key() {
        assert_eq!(propose(&[1, 2, 3, 4, 1, 2], 3), vec![3, 4, 1]);
    }

    #[test]
    fn two_gram_match() {
        // Only the 2-gram [1, 2] matches, at index 1.
        assert_eq!(propose(&[5, 1, 2, 9, 7, 2, 8, 1, 2], 3), vec![9, 7, 2]);
    }

    #[test]
    fn prefers_longer_key() {
        // Suffix [1, 2]: the 3-gram [9, 1, 2] matches at index 1 (continuation [6, 3, ..]),
        // the 2-gram [1, 2] matches at index 6 (continuation [7, 9, ..]). The 3-gram wins.
        let tokens = [5, 9, 1, 2, 6, 3, 1, 2, 7, 9, 1, 2];
        assert_eq!(propose(&tokens, 3), vec![6, 3, 1]);
    }

    #[test]
    fn most_recent_occurrence_wins() {
        // [1, 2] occurs at index 0 (continuation [3, ..]) and at index 3 (continuation [4, ..]).
        assert_eq!(propose(&[1, 2, 3, 1, 2, 4, 1, 2], 3), vec![4, 1, 2]);
    }

    #[test]
    fn no_match_proposes_nothing() {
        assert!(propose(&[1, 2, 3], 3).is_empty());
    }

    #[test]
    fn max_zero_proposes_nothing() {
        assert!(propose(&[1, 2, 3, 4, 1, 2], 0).is_empty());
    }

    #[test]
    fn draft_truncated_at_end_of_tokens() {
        // The match at index 0 is followed by only three tokens.
        assert_eq!(propose(&[1, 2, 3, 1, 2], 10), vec![3, 1, 2]);
    }

    #[test]
    fn out_is_appended_not_cleared() {
        let mut d = NgramDrafter::default();
        let mut out = vec![99];
        d.propose(&[1, 2, 3, 4, 1, 2], 2, &mut out);
        assert_eq!(out, vec![99, 3, 4]);
    }

    #[test]
    fn overlapping_match_is_allowed() {
        // Suffix [2, 2] also occurs at index 0, overlapping it; the token after is [2].
        assert_eq!(propose(&[2, 2, 2], 3), vec![2]);
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_min() {
        NgramDrafter::new(0, 4);
    }

    #[test]
    #[should_panic]
    fn new_rejects_min_above_max() {
        NgramDrafter::new(5, 4);
    }
}
