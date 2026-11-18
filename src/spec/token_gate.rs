//! A cost-aware gate for the model-free drafters (`--spec ngram|suffix`) on a target that streams
//! its experts (#37): a verify step there costs more than a plain step with every row it adds
//! (each row routes its own experts, and each new expert is a host read or a share of the
//! host's pool), so a draft pays only where its expected tokens beat that. Drafting changes
//! no output (greedy acceptance keeps the target's picks), only the time per token.
//!
//! After upstream Strata v0.2.13.1's draft policy (MIT, `draft_policy.cpp`,
//! `DraftPolicy::choose`; the idea, no code): the expected accepted tokens come from
//! the acceptance by the drafter's match length, the cost of a window of `1 + k` rows from
//! measured steps of that many rows, and a draft is taken only when its tokens per millisecond
//! beat a plain step's by [`MARGIN`].
//!
//! Here:
//! - Step costs are [`DepthModel`]'s means by rows and context bucket (the MTP depth policy's
//!   own, [`super::depth`]): every decode and verify step without a prompt chunk is one sample.
//!   A row count never measured costs as its nearest measured neighbour, so the first drafts
//!   go through and measure it.
//! - Acceptance is learned by match-length class ([`class`]) for the whole engine, as
//!   [`super::mtp_ngram`] learns it per sequence: [`AdaptiveState`]'s chain gains, whose prior
//!   (0.7 a position) lets every class draft until it has shown otherwise. A draft the gate
//!   turned down is followed as a shadow against what the sequence appends instead
//!   ([`SeqGate::appended`]), so a class shut out keeps learning and can come back.
//! - The batch shares one verify shape (`1 + K` rows a sequence): the gate picks the `K` that
//!   gives the batch the most expected tokens per millisecond, each sequence drafting
//!   `min(K, its drafts)` and the step padding to the longest of them only.
//!
//! Off unless `LEGION_DEBUG_TOKEN_GATE=1`: on an RTX 4090 with the experts in host memory a
//! verify window of 2 or 3 rows reads 2.4-6x the host experts of a plain step (1.5-3x its
//! time), so the gate drafts nothing, where acceptance alone gains in repetitive chats.

use super::adaptive::AdaptiveState;
use super::depth::DepthModel;

/// Match-length classes: 1, 2-3, 4-7, 8-15, 16-31, and 32 or more tokens of context matched.
pub const CLASSES: usize = 6;
/// Factor by which a draft's expected tokens per millisecond must beat a plain step's
/// (Strata's 3%).
pub const MARGIN: f64 = 1.03;
/// The most rows a verify step may have (sequences x (1 + drafts)). On an RTX 4090 with the
/// exact CPU expert share, Flash-Next's greedy outputs kept every bit with windows of 1-4 rows
/// and changed with 5 (the share takes no 5-row step, so its host experts take another path)
/// and with a 12-row step of four sequences. A window past 3 rows did not pay there either.
pub const MAX_ROWS: usize = 4;

/// The class of a draft whose first token matched `len` tokens of context.
pub fn class(len: usize) -> usize {
    (len.max(1).ilog2() as usize).min(CLASSES - 1)
}

/// One sequence's part: the class of its drafts in flight, and a turned-down draft followed as
/// a shadow.
#[derive(Debug, Clone, Default)]
pub struct SeqGate {
    pending: Option<usize>,
    shadow: Option<Shadow>,
}

#[derive(Debug, Clone)]
struct Shadow {
    class: usize,
    tokens: Vec<u32>,
    matched: usize,
}

/// A chain to learn from: `drafted` drafts of class `class`, the first `accepted` right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chain {
    pub class: usize,
    pub drafted: usize,
    pub accepted: usize,
}

impl SeqGate {
    /// The step verifies this sequence's drafts of class `class`.
    pub fn drafting(&mut self, class: usize) {
        self.pending = Some(class);
        self.shadow = None;
    }

    /// The step verifies no drafts of this sequence (a shadow followed goes on).
    pub fn plain(&mut self) {
        self.pending = None;
    }

    /// The gate turned down `tokens` (class `class`): follow them against what the sequence
    /// appends, unless a shadow is followed already.
    pub fn follow(&mut self, class: usize, tokens: &[u32]) {
        if self.shadow.is_none() && !tokens.is_empty() {
            self.shadow = Some(Shadow {
                class,
                tokens: tokens.to_vec(),
                matched: 0,
            });
        }
    }

    /// After the verify step of the drafts in flight: their chain (`drafted` verified, the first
    /// `accepted` right), if [`Self::drafting`] named them.
    pub fn verified(&mut self, drafted: usize, accepted: usize) -> Option<Chain> {
        let class = self.pending.take()?;
        (drafted > 0).then_some(Chain {
            class,
            drafted,
            accepted: accepted.min(drafted),
        })
    }

    /// The sequence appended `appended`: a shadow that ends now (a token differs, or all of it
    /// was matched) is a chain of its own.
    pub fn appended(&mut self, appended: &[u32]) -> Option<Chain> {
        let s = self.shadow.as_mut()?;
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
        if !differed && s.matched < s.tokens.len() {
            return None;
        }
        let s = self.shadow.take()?;
        Some(Chain {
            class: s.class,
            drafted: s.tokens.len(),
            accepted: s.matched,
        })
    }
}

/// The engine's gate: step costs and acceptance by class.
#[derive(Debug)]
pub struct TokenGate {
    costs: DepthModel,
    chains: [AdaptiveState; CLASSES],
}

impl Default for TokenGate {
    fn default() -> Self {
        Self {
            costs: DepthModel::new(false),
            chains: Default::default(),
        }
    }
}

/// One sequence's offer to [`TokenGate::choose`]: its drafts' class and how many it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offer {
    pub class: usize,
    pub drafts: usize,
}

impl TokenGate {
    /// A decode or verify step of `rows` rows (no prompt chunk), reading a context of `span`
    /// positions, took `ms`.
    pub fn record_step(&mut self, span: usize, rows: usize, ms: f64) {
        self.costs.set_context(span);
        self.costs.record_step(rows, ms);
    }

    /// Learn from a chain ([`SeqGate::verified`], [`SeqGate::appended`]).
    pub fn learn(&mut self, c: Chain) {
        if let Some(a) = self.chains.get_mut(c.class) {
            a.record_chain(c.drafted, c.drafted, c.accepted);
        }
    }

    /// Expected tokens from `k` drafts of class `class`, the bonus token included.
    pub fn gain(&self, class: usize, k: usize) -> f32 {
        if k == 0 {
            return 1.0;
        }
        self.chains[class.min(CLASSES - 1)].gain(k)
    }

    /// Per-position acceptance of class `class` as learned so far.
    pub fn rate(&self, class: usize) -> f32 {
        self.chains[class.min(CLASSES - 1)].rate()
    }

    /// Predicted time of a step of `rows` rows at the context set last.
    pub fn step_ms(&self, rows: usize) -> Option<f64> {
        self.costs.step_ms(rows)
    }

    /// The drafts per sequence the next step verifies (`K`: each sequence verifies
    /// `min(K, offer.drafts)`), for a batch reading a context of `span` positions: the `K` with
    /// the most expected tokens per millisecond, if that beats a plain step by [`MARGIN`], else
    /// 0. Windows stay within [`MAX_ROWS`] rows, and nothing is drafted before a plain step of
    /// the batch's size is measured (with a sequence or more always drafting, it never would be).
    pub fn choose(&mut self, span: usize, offers: &[Offer]) -> usize {
        let b = offers.len();
        let most = (offers.iter().map(|o| o.drafts).max().unwrap_or(0))
            .min((MAX_ROWS / b.max(1)).saturating_sub(1));
        if most == 0 {
            return 0;
        }
        self.costs.set_context(span);
        let Some(plain) = self.costs.step_ms(b).filter(|_| self.costs.measured(b)) else {
            return 0;
        };
        let tokens = |k: usize| -> f64 {
            (offers.iter())
                .map(|o| self.gain(o.class, k.min(o.drafts)) as f64)
                .sum()
        };
        let base = b as f64 / plain;
        let best = (1..=most)
            .filter_map(|k| {
                let ms = self.costs.step_ms(b * (1 + k))?;
                Some((k, tokens(k) / ms))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((k, per_ms)) if per_ms > base * MARGIN => k,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests;
