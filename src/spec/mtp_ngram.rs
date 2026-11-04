//! MTP + context n-gram drafting (#387, `--spec-mtp-ngram`): a sequence whose recent tokens
//! occurred earlier in its own context (a prompt it repeats or quotes, code it edits, a list
//! that comes back) drafts the continuation of that occurrence, verified in one step like the
//! MTP head's drafts but without head passes, and up to `--spec-k` deep. Elsewhere the MTP head
//! drafts as before.
//!
//! The idea is llama.cpp's speculative type list (MIT, `common/speculative.cpp`,
//! `common_speculative_gen_draft`: the n-gram drafters come first, the MTP head drafts only
//! where they have nothing) with its `ngram-mod` drafter (`common/ngram-mod.cpp`, a hash of the
//! last 24 tokens). Here the lookup is exact (no hash), over the sequence's own tokens, and the
//! choice between the two is made with the depth policy's measured costs ([`super::depth`]):
//! the n-gram draft is taken where its expected tokens per millisecond beat the MTP head's at
//! the depth the policy chose, and cut to the depth that pays most.
//!
//! The n-gram's acceptance is learned per sequence and by match length (a 64-token match is
//! more reliable than an 8-token one). A continuation the policy turned down is followed as a
//! shadow against the tokens the sequence then appends, so the estimate keeps learning while
//! the head drafts, and an unlucky stretch does not shut the n-gram out for good.

use super::SpecCounters;
use super::adaptive::AdaptiveState;
use super::depth::{DepthModel, HYSTERESIS};

/// Longest suffix match looked for, in tokens: past this, longer matches are no more reliable.
pub const MAX_MATCH: usize = 64;
/// Match-length classes the acceptance is estimated in: `[min, 2 min)`, `[2 min, 4 min)` and
/// `4 min` or more.
const CLASSES: usize = 3;

/// An earlier occurrence of a suffix of the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Tokens matched (at most [`MAX_MATCH`]).
    pub len: usize,
    /// Where the occurrence's continuation starts.
    pub follow: usize,
}

/// The longest earlier occurrence of a suffix of `tokens` of at least `min` tokens (at most
/// [`MAX_MATCH`] are compared), the most recent among equally long ones. The suffix itself is
/// not a candidate, but an occurrence may overlap it. One pass over the sequence, extending
/// only where the last token occurs.
pub fn longest_match(tokens: &[u32], min: usize) -> Option<Match> {
    let n = tokens.len();
    if min == 0 || n <= min {
        return None;
    }
    let last = tokens[n - 1];
    let mut best: Option<Match> = None;
    // `e`: the last token of an earlier occurrence, so its continuation `e + 1` exists.
    for e in (0..n - 1).rev() {
        if tokens[e] != last {
            continue;
        }
        let mut len = 1;
        while len < MAX_MATCH && len <= e && tokens[e - len] == tokens[n - 1 - len] {
            len += 1;
        }
        if len >= min && best.is_none_or(|b| len > b.len) {
            best = Some(Match { len, follow: e + 1 });
            if len == MAX_MATCH {
                break;
            }
        }
    }
    best
}

/// A continuation the policy turned down, compared with what the sequence appends instead.
#[derive(Debug, Clone)]
struct Shadow {
    class: usize,
    tokens: Vec<u32>,
    matched: usize,
}

/// One sequence's n-gram drafting: the lookup, the acceptance estimates and the counters.
#[derive(Debug, Clone)]
pub struct MtpNgram {
    min: usize,
    /// Per match-length class: the tokens a step of `k` n-gram drafts gains.
    chains: [AdaptiveState; CLASSES],
    shadow: Option<Shadow>,
    /// Verify steps that drafted from the n-gram, and their drafts.
    pub counters: SpecCounters,
}

/// A continuation to draft from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The match-length class its acceptance is estimated in.
    pub class: usize,
    pub tokens: Vec<u32>,
}

impl MtpNgram {
    /// Matches of at least `min` tokens (`--spec-mtp-ngram`). Panics if `min` is 0.
    pub fn new(min: usize) -> Self {
        assert!(min > 0, "the n-gram needs a match of at least one token");
        Self {
            min,
            chains: Default::default(),
            shadow: None,
            counters: SpecCounters::default(),
        }
    }

    fn class(&self, len: usize) -> usize {
        if len >= 4 * self.min {
            2
        } else if len >= 2 * self.min {
            1
        } else {
            0
        }
    }

    /// The continuation of `tokens`' longest earlier match, at most `cap` tokens.
    pub fn lookup(&self, tokens: &[u32], cap: usize) -> Option<Candidate> {
        if cap == 0 {
            return None;
        }
        let m = longest_match(tokens, self.min)?;
        let end = (m.follow + cap).min(tokens.len());
        Some(Candidate {
            class: self.class(m.len),
            tokens: tokens[m.follow..end].to_vec(),
        })
    }

    /// Expected tokens from a step of up to `k` drafts of a class-`class` continuation, the
    /// bonus token included.
    pub fn gain(&self, class: usize, k: usize) -> f32 {
        self.chains[class].gain(k)
    }

    /// The step drafted `drafted` tokens of a class-`class` continuation and the first
    /// `accepted` were accepted.
    pub fn record(&mut self, class: usize, drafted: usize, accepted: usize) {
        self.shadow = None;
        self.counters.record(drafted, accepted);
        self.chains[class].record_chain(drafted, drafted, accepted);
    }

    /// The policy turned `c` down: follow it against what the sequence appends (unless a
    /// shadow is being followed already).
    pub fn follow(&mut self, c: Candidate) {
        if self.shadow.is_none() && !c.tokens.is_empty() {
            self.shadow = Some(Shadow {
                class: c.class,
                tokens: c.tokens,
                matched: 0,
            });
        }
    }

    /// The sequence appended `appended`: a shadow that now ends (a token differs, or all of it
    /// was matched) counts as a step of its own drafts.
    pub fn appended(&mut self, appended: &[u32]) {
        let Some(s) = &mut self.shadow else {
            return;
        };
        let mut differed = false;
        for &t in appended {
            if s.matched == s.tokens.len() {
                break;
            }
            if s.tokens[s.matched] != t {
                differed = true;
                break;
            }
            s.matched += 1;
        }
        if differed || s.matched == s.tokens.len() {
            let s = self.shadow.take().expect("checked above");
            let n = s.tokens.len();
            self.chains[s.class].record_chain(n, n, s.matched);
        }
    }
}

/// What [`choose`] compares, for one sequence.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    /// Sequences in the step.
    pub batch: usize,
    /// Whether the head takes in the verified positions every step (one pass even without
    /// drafts).
    pub take_in: bool,
    /// The MTP head's depth for this sequence and what it is expected to gain.
    pub mtp_k: usize,
    pub mtp_gain: f32,
    /// The step verifies at least this many drafts' rows whatever the sequence drafts (the
    /// depth chosen for the whole batch: one verify shape).
    pub shape: usize,
    /// Tokens the n-gram continuation offers.
    pub avail: usize,
}

/// The n-gram depth (`1..=avail`) to draft at, if one beats the MTP head's depth by
/// [`HYSTERESIS`] in expected tokens per millisecond, with step and pass costs from `depth`
/// (an n-gram step runs no draft pass but the head's take-in; either verifies at least
/// [`Choice::shape`] drafts' rows). `ngram_gain(k)`: what `k`
/// n-gram drafts are expected to gain. `None` also before any step is measured.
pub fn choose(depth: &DepthModel, c: Choice, ngram_gain: impl Fn(usize) -> f32) -> Option<usize> {
    let b = c.batch.max(1);
    let per_ms = |k: usize, passes: usize, gain: f32| -> Option<f64> {
        let step = depth.step_ms(b * (1 + k.max(c.shape)))?;
        let pass = depth.pass_ms(b).unwrap_or(step / 10.0);
        Some(gain as f64 / (step + pass * passes as f64))
    };
    let mtp_passes = if c.take_in { c.mtp_k.max(1) } else { c.mtp_k };
    let mtp = per_ms(c.mtp_k, mtp_passes, c.mtp_gain)?;
    let passes = usize::from(c.take_in);
    let (k, best) = (1..=c.avail)
        .filter_map(|k| per_ms(k, passes, ngram_gain(k)).map(|t| (k, t)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    (best > mtp * HYSTERESIS).then_some(k)
}

#[cfg(test)]
mod tests;
