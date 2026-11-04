//! Offline measures of the router lookahead (#611) over a trace with predictions
//! ([`super::trace::Prediction`]): how much of each small step's routing a predictor saw
//! coming, and what prefetching by it would have read and saved.
//!
//! The predictor is the PILOT idea of project colibri (Apache-2.0, github.com/JustVugg/colibri):
//! a layer's router applied to an earlier hidden state ranks the experts the layer will route
//! to; its best `n_used + m` (m the margin) are read ahead.
//!
//! A step's prediction is every prediction of its layer (by the chosen predictor) made since
//! the layer's previous fetch; sets count each step's distinct experts, as the tier does.

use std::collections::HashSet;

use super::cache::SlotCache;
use super::trace::{Prediction, Predictor, Trace};

/// The routed experts per token of the trace's small steps (0 without any).
pub fn n_used(trace: &Trace) -> usize {
    trace
        .records
        .iter()
        .find(|r| !r.big && r.n_tokens > 0)
        .map_or(0, |r| r.ids.len() / r.n_tokens)
}

/// Per fetch record: the predictions of its layer by `p` made since the layer's previous fetch.
pub fn step_predictions(trace: &Trace, p: Predictor) -> Vec<Vec<&Prediction>> {
    let mut out = vec![Vec::new(); trace.records.len()];
    // Per layer, the indices of its fetches.
    let mut fetches = vec![Vec::new(); trace.header.layers.len()];
    for (i, r) in trace.records.iter().enumerate() {
        fetches[r.ord].push(i);
    }
    for pr in trace.predictions.iter().filter(|x| x.predictor == p) {
        let f = &fetches[pr.ord];
        let at = f.partition_point(|&i| i < pr.before);
        if let Some(&i) = f.get(at) {
            out[i].push(pr);
        }
    }
    out
}

/// The distinct experts `preds` rank in their best `top` per token, rank by rank.
fn predicted(preds: &[&Prediction], top: usize) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for p in preds {
        for e in p.top(top) {
            if seen.insert(e) {
                out.push(e);
            }
        }
    }
    out
}

/// The distinct experts of `ids` in first-seen order.
fn distinct(ids: &[u16]) -> Vec<usize> {
    let mut seen = HashSet::new();
    ids.iter()
        .map(|&e| e as usize)
        .filter(|&e| seen.insert(e))
        .collect()
}

/// How well a predictor saw the small steps coming, at one margin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Recall {
    /// Small steps, and those with a prediction.
    pub steps: u64,
    pub predicted_steps: u64,
    /// The predicted steps' distinct routed experts, and those the prediction named.
    pub routed: u64,
    pub covered: u64,
    /// Distinct experts the predictions named.
    pub named: u64,
    /// Of the predicted steps' routed experts: those LRU (the tier today) would have read, and
    /// those of them the prediction named.
    pub misses: u64,
    pub misses_covered: u64,
}

impl Recall {
    /// Covered over routed, of the predicted steps.
    pub fn recall(&self) -> f64 {
        self.covered as f64 / self.routed.max(1) as f64
    }

    /// Covered over the LRU misses.
    pub fn miss_recall(&self) -> f64 {
        self.misses_covered as f64 / self.misses.max(1) as f64
    }

    /// Steps with a prediction, of all small steps.
    pub fn coverage(&self) -> f64 {
        self.predicted_steps as f64 / self.steps.max(1) as f64
    }
}

/// The recall of predictor `p` naming each token's best `n_used + margin`, with LRU at `n_slots`
/// slots as the reference for which routed experts are misses.
pub fn recall(trace: &Trace, p: Predictor, margin: usize, n_slots: usize) -> Recall {
    let k = n_used(trace);
    let preds = step_predictions(trace, p);
    let h = &trace.header;
    let mut caches = vec![SlotCache::new(h.n_expert, n_slots); h.layers.len()];
    let mut r = Recall::default();
    for (i, rec) in trace.records.iter().enumerate() {
        let experts = distinct(&rec.ids);
        let c = &mut caches[rec.ord];
        if rec.big {
            for &e in &experts {
                if c.slot(e).is_none() && c.take_empty(e).is_none() {
                    break;
                }
            }
            continue;
        }
        let missing: Vec<usize> = experts
            .iter()
            .copied()
            .filter(|&e| c.slot(e).is_none())
            .collect();
        c.plan(&experts);
        r.steps += 1;
        if preds[i].is_empty() {
            continue;
        }
        r.predicted_steps += 1;
        let named: HashSet<usize> = predicted(&preds[i], k + margin).into_iter().collect();
        r.named += named.len() as u64;
        r.routed += experts.len() as u64;
        r.covered += experts.iter().filter(|e| named.contains(e)).count() as u64;
        r.misses += missing.len() as u64;
        r.misses_covered += missing.iter().filter(|e| named.contains(e)).count() as u64;
    }
    r
}

/// One lookahead replay's settings: LRU slots, and before each small step the prefetch of what
/// `predictor` names (each token's best `n_used + margin`, at most `ranks` of them), at most
/// `budget` reads per layer and step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lookahead {
    pub predictor: Predictor,
    pub margin: usize,
    pub n_slots: usize,
    pub budget: usize,
    /// Ranks per token read ahead at most (`usize::MAX`: all `n_used + margin`).
    pub ranks: usize,
}

/// What a lookahead replay served.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LookOutcome {
    /// Small steps' distinct experts found in a slot (prefetched ones included), and read when
    /// the step needed them.
    pub hits: u64,
    pub misses: u64,
    /// Tokens of the small steps (of one layer).
    pub tokens: u64,
    /// Experts read ahead, and of them those their step did not route to.
    pub prefetched: u64,
    pub wasted: u64,
}

impl LookOutcome {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }

    pub fn per_token(&self, n: u64) -> f64 {
        n as f64 / self.tokens.max(1) as f64
    }

    /// Every expert read, ahead or not.
    pub fn reads(&self) -> u64 {
        self.misses + self.prefetched
    }
}

/// Replays `trace` with prefetches as `la` sets.
pub fn replay(trace: &Trace, la: &Lookahead) -> LookOutcome {
    let k = n_used(trace);
    let preds = step_predictions(trace, la.predictor);
    let h = &trace.header;
    let mut caches = vec![SlotCache::new(h.n_expert, la.n_slots); h.layers.len()];
    let mut o = LookOutcome::default();
    for (i, rec) in trace.records.iter().enumerate() {
        let experts = distinct(&rec.ids);
        let c = &mut caches[rec.ord];
        if rec.big {
            for &e in &experts {
                if c.slot(e).is_none() && c.take_empty(e).is_none() {
                    break;
                }
            }
            continue;
        }
        let ahead = if preds[i].is_empty() {
            Vec::new()
        } else {
            let mut named = predicted(&preds[i], (k + la.margin).min(la.ranks));
            named.truncate(la.n_slots);
            c.prefetch(&named, la.budget)
        };
        let used: HashSet<usize> = experts.iter().copied().collect();
        o.prefetched += ahead.len() as u64;
        o.wasted += ahead.iter().filter(|(e, _)| !used.contains(e)).count() as u64;
        let p = c.plan(&experts);
        o.hits += p.hits as u64;
        o.misses += p.loads.len() as u64;
        if rec.ord == 0 {
            o.tokens += rec.n_tokens as u64;
        }
    }
    o
}
