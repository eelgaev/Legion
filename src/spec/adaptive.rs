//! Load and acceptance based draft depth, independent of the executor and drafter.

use super::SpecConfig;

/// Largest draft depth whose last token has at least `p_floor` acceptance probability.
/// The batch gate takes precedence, even for a perfect drafter.
pub fn adaptive_k(alpha: f32, k_max: usize, batch: usize, cfg: &SpecConfig) -> usize {
    if batch >= cfg.max_batch {
        return 0;
    }
    if alpha >= 0.999 {
        return k_max;
    }
    if alpha <= 0.0 || !alpha.is_finite() {
        return 0;
    }
    k_max.min((cfg.p_floor.ln() / alpha.ln()).floor().max(0.0) as usize)
}

/// Whether a drafter whose depth follows measured costs ([`super::depth`]) drafts nothing at
/// `batch` decoding sequences: the `--spec-max-batch` gate, unless `--spec-gate-cost-model off`
/// lets the cost model weigh large batches itself, and only up to `gate_until` sequences when
/// that is set (#814).
pub fn cost_model_gated(batch: usize, cfg: &SpecConfig) -> bool {
    cfg.gate_cost_model
        && batch >= cfg.max_batch
        && (cfg.gate_until == 0 || batch <= cfg.gate_until)
}

/// Whether the gate leaves a drafter whose depth follows measured costs drafting at large
/// batches, up to a full one of `max_seqs` sequences, so its buffers there must be sized: with
/// `--spec-gate-cost-model off`, or past `gate_until` (see [`cost_model_gated`]).
pub fn cost_model_ungated_large(max_seqs: usize, cfg: &SpecConfig) -> bool {
    !cfg.gate_cost_model || (cfg.gate_until > 0 && max_seqs > cfg.gate_until)
}

/// The deepest draft `batch` drafting sequences may take under `--spec-max-rows` (`max_rows`;
/// 0: no cap) when the depth policy would allow `k`: every sequence's token and its drafts
/// within `max_rows` rows, so `max_rows / batch - 1` (0 once one row each fills it).
pub fn rows_depth_cap(batch: usize, k: usize, max_rows: usize) -> usize {
    if max_rows == 0 {
        return k;
    }
    k.min((max_rows / batch.max(1)).saturating_sub(1))
}

/// Per verify step, the weight the chain estimates ([`AdaptiveState::rate`],
/// [`AdaptiveState::gain`]) keep of what they knew before.
const CHAIN_DECAY: f32 = 0.95;
/// Prior per-position acceptance of a new sequence.
const PRIOR_RATE: f32 = 0.7;
/// Deepest cap [`AdaptiveState::gain`] estimates.
pub const MAX_GAIN_DEPTH: usize = 16;

/// Acceptance estimates and consecutive low-acceptance idle steps of one sequence.
#[derive(Debug, Clone)]
pub struct AdaptiveState {
    alpha: f32,
    idle: usize,
    /// Per-position acceptance ([`Self::rate`]): accepted drafts, and draft positions tried
    /// (the accepted ones plus the one that ended the chain before its cap), both decayed.
    hits: f32,
    tries: f32,
    /// `gains[k]`: tokens a step with a cap of `k` drafts would have gained, averaged (decayed)
    /// over the steps so far ([`Self::gain`]).
    gains: [f32; MAX_GAIN_DEPTH + 1],
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            idle: 0,
            // The prior rate, worth two positions.
            hits: 2.0 * PRIOR_RATE,
            tries: 2.0,
            gains: std::array::from_fn(|k| geometric_gain(PRIOR_RATE, k)),
        }
    }
}

/// `1 + a + a^2 + ... + a^k`: tokens a chain of `k` drafts gains at per-position acceptance `a`.
pub fn geometric_gain(a: f32, k: usize) -> f32 {
    let (mut sum, mut p) = (1.0, 1.0);
    for _ in 0..k {
        p *= a;
        sum += p;
    }
    sum
}

impl AdaptiveState {
    /// Update only after a step with drafts; plain decode steps carry no new evidence.
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        if drafted > 0 {
            self.alpha = 0.8 * self.alpha + 0.2 * (accepted as f32 / drafted as f32);
        }
    }

    /// After verifying `drafted` drafts made under cap `cap`, of which the first `accepted`
    /// were accepted. A chain that ended before its cap (a rejection, or a draft the head was
    /// not confident enough to make) counts one failed position; one that reached its cap
    /// says nothing about the position after it.
    ///
    /// The step also tells what every other cap would have gained: drafts do not depend on the
    /// cap, so a cap `k <= cap` keeps `min(accepted, k)` of them, and a chain that ended before
    /// its cap would have ended there under any larger one. Only a chain that reached its cap
    /// leaves the larger caps unknown: they get the cap's tokens plus a geometric tail at
    /// [`Self::rate`]. So [`Self::gain`] follows how acceptance falls along a chain, and drafts
    /// the head declines to make, from the steps themselves.
    pub fn record_chain(&mut self, cap: usize, drafted: usize, accepted: usize) {
        if cap == 0 {
            return;
        }
        let accepted = accepted.min(drafted);
        self.hits = CHAIN_DECAY * self.hits + accepted as f32;
        self.tries = CHAIN_DECAY * self.tries + (accepted + usize::from(accepted < cap)) as f32;
        let rate = self.rate();
        for (k, g) in self.gains.iter_mut().enumerate() {
            let now = if k <= cap || accepted < cap {
                (accepted.min(k) + 1) as f32
            } else {
                cap as f32 + geometric_gain(rate, k - cap)
            };
            *g = CHAIN_DECAY * *g + (1.0 - CHAIN_DECAY) * now;
        }
    }

    /// Expected tokens from a step with a cap of `k` drafts, the bonus token included (`k` at
    /// most [`MAX_GAIN_DEPTH`]).
    pub fn gain(&self, k: usize) -> f32 {
        self.gains[k.min(MAX_GAIN_DEPTH)]
    }

    /// The probability that a draft is accepted given the drafts before it were: what a
    /// chain of `k` drafts gains is `rate + rate^2 + ... + rate^k` ([`crate::spec::depth`]).
    pub fn rate(&self) -> f32 {
        self.hits / self.tries
    }

    /// Select once per drafting opportunity. Grammar-constrained sequences must not call
    /// this: their zero depth is unrelated to acceptance and must not advance the probe.
    pub fn select(&mut self, k_max: usize, batch: usize, cfg: &SpecConfig) -> usize {
        if cfg.fixed {
            self.idle = 0;
            return k_max;
        }
        let k = adaptive_k(self.alpha, k_max, batch, cfg);
        if batch >= cfg.max_batch || k_max == 0 || k > 0 {
            self.idle = 0;
            return k;
        }
        self.idle += 1;
        if self.idle == 16 {
            self.idle = 0;
            1
        } else {
            0
        }
    }
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            kind: Default::default(),
            k: 0,
            fixed: false,
            max_batch: 8,
            gate_cost_model: true,
            gate_cost_model_given: false,
            gate_until: 0,
            row_cost: true,
            pad_longest: false,
            max_rows: 0,
            p_floor: 0.5,
            draft: None,
            token_drafter: None,
            model_drafter: None,
            p_min: 0.0,
            draft_temperature: 0.0,
            min_confidence: None,
            draft_window: 0,
            draft_weights: Default::default(),
            draft_lm_head: Default::default(),
            draft_batched_attn: false,
            mtp_ngram: 0,
            mtp_ngram_k: 0,
        }
    }
}

#[cfg(test)]
mod tests;
