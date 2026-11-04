//! EAGLE-3 drafter (`--spec eagle3 --spec-draft <dir>`): a light head that reads the target's
//! low, middle and high layer outputs and proposes the next tokens autoregressively.
//!
//! Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
//! Training-Time Test", 2025. The head is one Llama-style decoder layer whose attention reads
//! `cat(norm(embed(token)), norm(g))` (twice the model width), `g` being a linear map (`fc`) of
//! the target's three hidden states; its own `lm_head` predicts over a smaller vocabulary that
//! `d2t` maps back to the target's tokens. The forward pass is in [`graph`], after vLLM's
//! `llama_eagle3.py` (Apache-2.0; SGLang's and SafeAILab's are the same computation, also
//! Apache-2.0). The checkpoint is safetensors + `config.json`, read through
//! [`crate::safetensors::Checkpoint`] ([`weights`], [`config`]).
//!
//! # As a [`ModelDrafter`]
//!
//! [`ModelDrafter::hidden_layers`] is the three target layers of the config
//! ([`config::Eagle3Config::target_layers`]). A call takes in `ran` target positions: position
//! `p` pairs the fused hidden rows of `p` with `tokens[p + 1]` (the token the target went on to
//! pick) and appends one K/V cell to the head's per-sequence cache; the last of them is the
//! head's first step, whose output predicts the first draft. Further drafts are a chain of
//! single-token steps that feed back the head's own pre-norm output and the previous draft
//! (no `fc`: `h` is already the head's width). No trees.
//!
//! The head's cache holds committed cells (positions the target kept) and, past them, the
//! chain's speculative ones. Both are rolled back the same way: a call whose `start` is below
//! the committed end overwrites from `start`, so rejected drafts' cells (and any positions a
//! rejected verify row had put there) are simply reused. A `start` past the committed end (the
//! prefix cache skipped positions, so the target has no hidden states for them) restarts the
//! cache there: the head then attends only to what it has seen.
//!
//! # Draft distributions
//!
//! [`DraftMode::Greedy`] (the default) drafts the head's argmax and returns no probabilities:
//! the draft is a point mass, which the acceptance rule handles exactly for greedy and sampled
//! targets alike (vLLM does the same). [`DraftMode::Sample`] draws each draft from the head's
//! softmax and returns that distribution as [`crate::spec::Draft::probs`] (over the target
//! vocabulary, zero outside the draft vocabulary), for rejection sampling with `q`. The drafter does not know
//! the request's temperature, so a greedy request then loses some acceptance to the sampling.
//!
//! Sequences are drafted one after another, each with its own small graphs.

pub mod config;
mod graph;
pub mod tiny;
mod weights;

use std::collections::HashMap;
use std::ops::Range;

use crate::engine::sampling::Rng;
use crate::ggml::{Backend, gpu_device};
use crate::spec::{DraftRequest, ModelDrafter, SeqId, TargetInfo};

use config::{Eagle3Config, VocabMap};
use graph::{Head, Input, KvCache};
use weights::Weights;

/// How drafts are picked from the head's distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DraftMode {
    /// Argmax; point-mass drafts, no [`crate::spec::Draft::probs`].
    #[default]
    Greedy,
    /// Draw from the softmax with this seed; the distribution goes to
    /// [`crate::spec::Draft::probs`].
    Sample { seed: u64 },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Eagle3Options {
    pub mode: DraftMode,
}

/// One sequence's cache: cells `0..len` hold the target-kept positions `first..first + len`.
struct SeqState {
    kv: KvCache,
    first: usize,
    len: usize,
}

pub struct Eagle3Drafter {
    head: Head,
    layers: Vec<usize>,
    n_vocab: usize,
    threads: usize,
    mode: DraftMode,
    rng: Rng,
    seqs: HashMap<SeqId, SeqState>,
}

impl Eagle3Drafter {
    /// Loads `info.draft` on the target's device.
    pub fn load(info: &TargetInfo, opts: Eagle3Options) -> Result<Self, String> {
        let mut backend = if info.gpu {
            gpu_device().ok_or("no GPU for the EAGLE-3 head")?.init()
        } else {
            Backend::cpu()
        }
        .map_err(|e| e.to_string())?;
        backend.set_threads(info.threads.clamp(1, 8));
        Self::load_on(info, opts, backend)
    }

    /// Loads `info.draft` on `backend`.
    pub fn load_on(
        info: &TargetInfo,
        opts: Eagle3Options,
        backend: Backend,
    ) -> Result<Self, String> {
        let dir = info
            .draft
            .as_deref()
            .ok_or("--spec eagle3 needs --spec-draft <checkpoint directory>")?;
        let w = Weights::load(dir, info, &backend)
            .map_err(|e| format!("EAGLE-3 head {}: {e}", dir.display()))?;
        let layers = w.cfg.target_layers(info.n_layer)?;
        let seed = match opts.mode {
            DraftMode::Sample { seed } => seed,
            DraftMode::Greedy => 0,
        };
        Ok(Self {
            head: Head::new(w, backend),
            layers,
            n_vocab: info.n_vocab,
            threads: info.threads,
            mode: opts.mode,
            rng: Rng::new(seed),
            seqs: HashMap::new(),
        })
    }

    pub fn config(&self) -> &Eagle3Config {
        self.head.cfg()
    }

    pub fn vocab_map(&self) -> &VocabMap {
        &self.head.w.vocab
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The head's embedding of `token`.
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>, String> {
        let mut row = Vec::new();
        self.head.w.embed(token, &mut row)?;
        Ok(row)
    }

    /// The positions a sequence's cache holds (committed cells), if it has one.
    pub fn cached(&self, seq: SeqId) -> Option<Range<usize>> {
        self.seqs.get(&seq).map(|s| s.first..s.first + s.len)
    }

    /// Takes in and drafts for one sequence.
    fn step_seq(&mut self, r: &mut DraftRequest<'_>) -> Result<(), String> {
        let cfg = self.head.cfg().clone();
        let (h, ht) = (cfg.hidden, cfg.target_hidden);
        if r.hidden.len() != cfg.n_aux {
            return Err(format!(
                "{} hidden layers for a head that fuses {}",
                r.hidden.len(),
                cfg.n_aux
            ));
        }
        if let Some(bad) = r.hidden.iter().find(|x| x.len() < r.ran * ht) {
            return Err(format!(
                "{} hidden values for {} positions of {ht}",
                bad.len(),
                r.ran
            ));
        }
        // Position p needs tokens[p + 1].
        let n = r.ran.min(r.tokens.len().saturating_sub(r.start + 1));
        if !self.seqs.contains_key(&r.seq) {
            let kv = KvCache::new(self.head.backend(), cfg.n_kv_head * cfg.head_dim, n + r.max)?;
            let state = SeqState {
                kv,
                first: r.start,
                len: 0,
            };
            self.seqs.insert(r.seq, state);
        }
        let st = self.seqs.get_mut(&r.seq).expect("just inserted");
        // Rollback, or a gap the head has no hidden states for.
        let end = st.first + st.len;
        if r.start < st.first || r.start > end {
            st.first = r.start;
            st.len = 0;
        } else {
            st.len = r.start - st.first;
        }
        if n == 0 {
            return Ok(());
        }
        st.kv
            .reserve(self.head.backend(), st.len + n + r.max, st.len)?;

        // The `ran` positions in one pass: embeddings of tokens[p + 1], fused hidden rows.
        let mut emb = Vec::with_capacity(n * h);
        for p in 0..n {
            self.head.w.embed(r.tokens[r.start + p + 1], &mut emb)?;
        }
        let mut hid = Vec::with_capacity(n * cfg.n_aux * ht);
        for p in 0..n {
            for layer in &r.hidden {
                hid.extend_from_slice(&layer[p * ht..(p + 1) * ht]);
            }
        }
        let want = r.max > 0 && n == r.ran;
        let out = self.head.forward(
            &st.kv,
            &Input {
                emb: &emb,
                hid: &hid,
                fused: true,
                cells: st.len,
                pos0: r.start,
                want_out: want,
            },
        )?;
        st.len += n;
        let Some(mut out) = out else { return Ok(()) };

        // The chain: cells past `len` are speculative and never committed.
        for i in 0..r.max {
            let (token, q) = pick(
                &out.logits,
                &self.head.w.vocab,
                self.mode,
                &mut self.rng,
                self.n_vocab,
            );
            r.out.tokens.push(token);
            if let Some(q) = q {
                r.out.probs.push(q);
            }
            if i + 1 == r.max {
                break;
            }
            let mut emb = Vec::with_capacity(h);
            self.head.w.embed(token, &mut emb)?;
            let next = self.head.forward(
                &st.kv,
                &Input {
                    emb: &emb,
                    hid: &out.hidden,
                    fused: false,
                    cells: st.len + i,
                    pos0: r.start + n + i,
                    want_out: true,
                },
            )?;
            out = next.expect("requested output");
        }
        Ok(())
    }
}

/// The draft token (a target token) and, when sampling, its distribution over `n_vocab` tokens.
fn pick(
    logits: &[f32],
    vocab: &VocabMap,
    mode: DraftMode,
    rng: &mut Rng,
    n_vocab: usize,
) -> (u32, Option<Vec<f32>>) {
    match mode {
        DraftMode::Greedy => {
            let best = crate::engine::sampling::argmax(logits);
            (vocab.to_target(best), None)
        }
        DraftMode::Sample { .. } => {
            let probs = softmax(logits);
            let mut u = rng.next_f64();
            let mut pick = None;
            for (i, &p) in probs.iter().enumerate() {
                if p <= 0.0 {
                    continue;
                }
                pick = Some(i);
                if u < f64::from(p) {
                    break;
                }
                u -= f64::from(p);
            }
            let d = pick.expect("a softmax has a positive entry");
            (
                vocab.to_target(d),
                Some(vocab.to_target_probs(&probs, n_vocab)),
            )
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits.iter().map(|&l| f64::from(l - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.iter().map(|&e| (e / total) as f32).collect()
}

impl ModelDrafter for Eagle3Drafter {
    fn hidden_layers(&self) -> Vec<usize> {
        self.layers.clone()
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        for r in batch.iter_mut() {
            self.step_seq(r)?;
        }
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        self.seqs.remove(&seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_normalizes() {
        let p = softmax(&[1.0, 2.0, 3.0, f32::NEG_INFINITY]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(p[3], 0.0);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }
}
