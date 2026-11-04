//! `--pipeline-mtp-split` (#765): a layer split for decoding groups in flight with the MTP head.
//!
//! With `--pipeline-groups` a round lasts as long as its slowest stage. The last stage also
//! runs the output head, and since #762 the exiting group's sampling, commit and MTP drafts
//! run on its GPU while the other stages compute, so the last stage's share of a round is its
//! layers plus that work. The byte-balanced split gives it as many layers as the others, which
//! leaves stages 0..n-1 idle for that extra time each round.
//!
//! The [`Probe`] measures, over rounds where every stage had a piece, each stage's piece time
//! and the exiting step's work on the engine thread. From those it takes a layer's cost (the
//! stages before the last, which run layers only) and the last stage's extra work in layers
//! ([`SplitHint::extra_layers`]); [`mtp_split`] then moves layers off the last stage until the
//! round's longest share is smallest. The server measures this on a short synthetic run after
//! loading and loads again with the split if it differs (`src/cli/serve/pipe_split.rs`).
//!
//! The split applies to everything: whole steps, prompts and C=1 decoding cross the stages
//! one after another, so for them only the total matters, apart from the expert cache's share
//! per layer, which follows the layers.

use std::time::Duration;

/// Rounds the probe ignores first: the groups' graphs are built in them.
const SKIP_ROUNDS: u64 = 8;

/// Fewest measured rounds for a hint.
pub const MIN_ROUNDS: u64 = 24;

/// Per-stage piece times and the exiting step's work, summed over full rounds.
#[derive(Debug, Clone, Default)]
pub struct Probe {
    /// Layers per local stage.
    layers: Vec<usize>,
    /// Full rounds seen (with the skipped ones).
    seen: u64,
    /// Full rounds summed.
    rounds: u64,
    /// Per stage: summed piece seconds.
    piece: Vec<f64>,
    /// Summed seconds of the exiting step's sampling, commit and drafts.
    exit: f64,
}

/// What a [`Probe`] measured, and the split it suggests.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitHint {
    /// The suggested split (first layer of every stage after the first).
    pub cuts: Vec<usize>,
    /// The split measured.
    pub current: Vec<usize>,
    /// A layer's mean piece time over the stages before the last, in ms.
    pub layer_ms: f64,
    /// The last stage's work per round beyond its layers at `layer_ms` each (the output head,
    /// sampling, commit, MTP drafts), in ms.
    pub extra_ms: f64,
    /// `extra_ms` in layers.
    pub extra_layers: f64,
    /// Rounds measured.
    pub rounds: u64,
}

impl Probe {
    /// A probe for stages of `layers` layers each.
    pub fn new(layers: Vec<usize>) -> Self {
        let n = layers.len();
        Self {
            layers,
            piece: vec![0.0; n],
            ..Self::default()
        }
    }

    /// Records one round: per local stage its piece time (`None`: no piece), whether its
    /// exiting step was handed out while other stages computed, and the time the engine took
    /// for that step (sampling, commit, drafts). Only rounds with a piece on every stage and a
    /// step handed out early count.
    pub fn record(&mut self, pieces: &[Option<Duration>], early: bool, exit: Duration) {
        if !early || pieces.len() != self.layers.len() || pieces.iter().any(Option::is_none) {
            return;
        }
        self.seen += 1;
        if self.seen <= SKIP_ROUNDS {
            return;
        }
        self.rounds += 1;
        for (sum, p) in self.piece.iter_mut().zip(pieces) {
            *sum += p.map_or(0.0, |d| d.as_secs_f64());
        }
        self.exit += exit.as_secs_f64();
    }

    /// Rounds measured.
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// The measured costs and the split they suggest, once [`MIN_ROUNDS`] rounds are in (and
    /// with at least two stages).
    pub fn hint(&self) -> Option<SplitHint> {
        let n = self.layers.len();
        if n < 2 || self.rounds < MIN_ROUNDS {
            return None;
        }
        let r = self.rounds as f64;
        let before: usize = self.layers[..n - 1].iter().sum();
        if before == 0 {
            return None;
        }
        let layer_s = self.piece[..n - 1].iter().sum::<f64>() / r / before as f64;
        if layer_s <= 0.0 {
            return None;
        }
        let last = self.piece[n - 1] / r + self.exit / r;
        let extra_s = last - self.layers[n - 1] as f64 * layer_s;
        let extra_layers = extra_s / layer_s;
        let n_layer: usize = self.layers.iter().sum();
        let current = cuts_of(&self.layers);
        Some(SplitHint {
            cuts: mtp_split(n_layer, n, extra_layers),
            current,
            layer_ms: layer_s * 1e3,
            extra_ms: extra_s * 1e3,
            extra_layers,
            rounds: self.rounds,
        })
    }
}

/// Split points of stages of `layers` layers each.
fn cuts_of(layers: &[usize]) -> Vec<usize> {
    let mut at = 0;
    let mut cuts = Vec::new();
    for &l in &layers[..layers.len().saturating_sub(1)] {
        at += l;
        cuts.push(at);
    }
    cuts
}

/// The split of `n_layer` equal layers over `n_stages` stages whose last also does
/// `extra_layers` layers' worth of other work each round: the last stage's layer count that
/// makes the longest stage shortest (the most layers among equals, nearest the even split
/// that whole steps and prompts prefer), the other stages' layers at most one apart, the
/// larger first. Returns the first layer of every stage after the first.
pub fn mtp_split(n_layer: usize, n_stages: usize, extra_layers: f64) -> Vec<usize> {
    if n_stages < 2 || n_layer < n_stages {
        return Vec::new();
    }
    let others = n_stages - 1;
    let extra = if extra_layers.is_finite() {
        extra_layers
    } else {
        0.0
    };
    let mut best = (f64::INFINITY, 1);
    for last in 1..=n_layer - others {
        let rest = (n_layer - last).div_ceil(others) as f64;
        let round = rest.max(last as f64 + extra);
        // Ties go to the larger last stage (scanned upwards).
        if round <= best.0 {
            best = (round, last);
        }
    }
    let rest = n_layer - best.1;
    let mut layers: Vec<usize> = (0..others)
        .map(|i| rest / others + usize::from(i < rest % others))
        .collect();
    layers.push(best.1);
    cuts_of(&layers)
}

#[cfg(test)]
mod tests;
