//! Speculative decoding in the Qwen3.5 engine's worker: which drafter runs, the acceptance
//! target over a sequence's verify rows, and draft-length caps. The step itself is in
//! `Worker::step` (see `crate::spec` for the design).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::sampling::{Dist, Sampler};
use crate::model::core::{Executor, Group, Logits, SeqInput};
use crate::spec::accept::Target;
use crate::spec::dflash::{DflashDrafter, DflashOptions};
use crate::spec::draft::DraftModel;
use crate::spec::dspark::{DsparkDrafter, DsparkOptions};
use crate::spec::eagle3::{Eagle3Drafter, Eagle3Options};
use crate::spec::{DraftRequest, SeqId};
use crate::spec::{
    ModelDrafter, SpecConfig, SpecKind, TargetInfo, TokenDrafter, TokenDrafterFactory,
};

/// The engine's drafter.
pub(super) enum Drafting {
    Off,
    /// A model-free drafter per sequence.
    Tokens(TokenDrafterFactory),
    /// One model drafter for all sequences, fed the step's hidden states.
    Model(Box<dyn ModelDrafter>),
    /// The target's own MTP head, run by the executor.
    Mtp(MtpDrafter),
    /// Tensor parallelism over machines, a rank above 0: rank 0 alone drafts with a model
    /// drafter (`--spec dflash|dspark`, fed the target's hidden states, which every rank has
    /// whole after the all-reduces), and this rank runs the verify steps it sends. The drafter's
    /// whole block, if it drafts one ([`crate::spec::ModelDrafter::block_drafts`]).
    Rank0(Option<usize>),
}

impl Drafting {
    /// The drafter `cfg` asks for, or why there is none yet. A DFlash drafter scores its drafts
    /// with the first `dflash_vocab` rows of the target's head (`--spec-draft-vocab`; 0: all).
    pub(super) fn new(
        cfg: &SpecConfig,
        target: &TargetInfo,
        dflash_vocab: usize,
    ) -> Result<Self, String> {
        if !cfg.enabled() {
            return Ok(Drafting::Off);
        }
        if let Some(f) = &cfg.model_drafter {
            return Ok(Drafting::Model(f(target)?));
        }
        if let Some(f) = &cfg.token_drafter {
            return Ok(Drafting::Tokens(f.clone()));
        }
        if let Some(f) = cfg.kind.token_drafter() {
            return Ok(Drafting::Tokens(f));
        }
        if cfg.kind == SpecKind::Mtp {
            return Ok(Drafting::Mtp(MtpDrafter::new(cfg.p_min)));
        }
        if cfg.kind == SpecKind::Dflash {
            let mut opts = DflashOptions {
                temperature: cfg.draft_temperature,
                min_confidence: cfg.min_confidence.unwrap_or(0.0),
                weights: cfg.draft_weights,
                draft_vocab: dflash_vocab,
                lm_head: cfg.draft_lm_head,
                ..Default::default()
            };
            if cfg.draft_window > 0 {
                opts.window = cfg.draft_window;
            }
            return Ok(Drafting::Model(Box::new(DflashDrafter::load(
                target, opts,
            )?)));
        }
        if cfg.kind == SpecKind::Dspark {
            let mut opts = DsparkOptions {
                temperature: cfg.draft_temperature,
                min_confidence: cfg.min_confidence.unwrap_or(0.0),
                weights: cfg.draft_weights,
                ..Default::default()
            };
            if cfg.draft_window > 0 {
                opts.window = cfg.draft_window;
            }
            return Ok(Drafting::Model(Box::new(DsparkDrafter::load(
                target, opts,
            )?)));
        }
        if cfg.kind == SpecKind::Draft {
            return Ok(Drafting::Model(Box::new(DraftModel::load(target, cfg.k)?)));
        }
        if cfg.kind == SpecKind::Eagle3 {
            let head = Eagle3Drafter::load(target, Eagle3Options::default())?;
            return Ok(Drafting::Model(Box::new(head)));
        }
        unreachable!("--spec {} has a drafter above", cfg.kind.name())
    }

    pub(super) fn is_on(&self) -> bool {
        !matches!(self, Drafting::Off)
    }

    /// A model drafter's whole block ([`crate::spec::ModelDrafter::block_drafts`]).
    pub(super) fn block_drafts(&self) -> Option<usize> {
        match self {
            Drafting::Model(d) => d.block_drafts(),
            Drafting::Rank0(n) => *n,
            _ => None,
        }
    }

    /// A new sequence's own drafter, if drafters are per sequence.
    pub(super) fn for_seq(&self) -> Option<Box<dyn TokenDrafter>> {
        match self {
            Drafting::Tokens(f) => Some(f()),
            _ => None,
        }
    }
}

/// Drafts worth verifying for a sequence: at most `k`, and no more than can still be emitted
/// (`max_tokens - generated - 1`, the bonus token being one) or cached (`max_ctx - n_past - 1`).
pub(super) fn draft_cap(
    k: usize,
    max_tokens: u32,
    generated: u32,
    max_ctx: usize,
    n_past: usize,
) -> usize {
    let by_tokens = (max_tokens.saturating_sub(generated) as usize).saturating_sub(1);
    let by_ctx = max_ctx.saturating_sub(n_past + 1);
    k.min(by_tokens).min(by_ctx)
}

/// A sequence's verify rows as [`crate::spec::accept`] sees them: the sampler's distribution
/// at each row, with the penalty counts of the tokens accepted before it.
pub(super) struct SeqTarget<'s> {
    pub sampler: &'s mut Sampler,
    /// Per row: the device argmax, and the logits (empty for argmax-only rows).
    pub rows: Vec<(u32, Vec<f32>)>,
    /// Keep each used row's raw logits (for log probabilities).
    pub keep_raw: bool,
    /// A full row's device argmax is the greedy pick unless it is a masked stop token, so the
    /// row is not scanned (#695): the sampler is greedy, no log probabilities are asked for and
    /// the model's device argmax is the host's (`Arch::argmax_first_of_ties`).
    pub device_picks: bool,
    pub raw: Vec<Option<Vec<f32>>>,
    /// Rows `0..mask_rows` are below `min_tokens`: `stop_tokens` cannot be picked there.
    pub mask_rows: usize,
    pub stop_tokens: &'s [u32],
}

impl Target for SeqTarget<'_> {
    fn dist(&mut self, i: usize) -> Dist {
        let (argmax, row) = &mut self.rows[i];
        let masked = i < self.mask_rows && self.stop_tokens.contains(argmax);
        if row.is_empty() || (self.device_picks && !masked) {
            self.raw.push(None);
            return Dist::Greedy(*argmax);
        }
        self.raw.push(self.keep_raw.then(|| row.clone()));
        if i < self.mask_rows {
            for &t in self.stop_tokens {
                if let Some(l) = row.get_mut(t as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        self.sampler.dist(row)
    }

    fn uniform(&mut self) -> f64 {
        self.sampler.uniform()
    }

    fn push(&mut self, token: u32) {
        self.sampler.accept(token);
    }
}

/// A longer take-in (a prompt chunk) goes in pieces without logits of at most this many
/// positions (the head's compute buffers grow with them), which only write the head's K/V, and
/// then a pass over the last [`TAKE_IN_TAIL`] positions or fewer.
pub(super) const TAKE_IN_PIECE: usize = 1024;
/// See [`TAKE_IN_PIECE`].
pub(super) const TAKE_IN_TAIL: usize = 8;

/// [`MtpDrafter::prefetch_rows`] for a sequence fed `fed` (up to its bonus token) and drafted
/// `drafts` so far.
fn prefetch_after(exec: &Executor, fed: &[u32], drafts: &[(u32, f32)]) {
    let all: Vec<u32> = fed[fed.len().saturating_sub(8)..]
        .iter()
        .copied()
        .chain(drafts.iter().map(|d| d.0))
        .collect();
    let (tok, before) = all.split_last().expect("the bonus token at least");
    exec.model()
        .prefetch_ngram(before, std::slice::from_ref(tok));
}

/// Whether an MTP take-in pass orders its sequences by their rows ([`MtpDrafter::step`], #858;
/// on by default, `legion serve --no-mtp-take-in-sort` keeps the batch's order).
static TAKE_IN_SORTED: AtomicBool = AtomicBool::new(true);

/// Sets [`TAKE_IN_SORTED`] for the passes run from now on.
pub fn set_take_in_sorted(on: bool) {
    TAKE_IN_SORTED.store(on, Ordering::Relaxed);
}

fn take_in_sorted() -> bool {
    TAKE_IN_SORTED.load(Ordering::Relaxed)
}

/// Drafts with the target's MTP head (`crate::model::core`'s `mtp.rs`), as llama.cpp's
/// `common_speculative_impl_draft_mtp` (MIT, `common/speculative.cpp`) drives it: the head's
/// position `p` pairs the token at `p` with the target's residual at `p - 1`.
///
/// Each step it first takes in the positions the target kept (`start..start + ran`, each with
/// the residual of the position before; the first with the last residual of the previous
/// step), plus, when drafting, the bonus token after them, whose output gives draft 1. Then one
/// pass per further draft, feeding back the head's own residual. A sequence stops drafting at a
/// draft the head gives less than `p_min`. Draft positions past the kept ones are rewritten by
/// the next step's take-in, so a rejection needs no rollback.
pub(super) struct MtpDrafter {
    /// The head confidence below which a sequence stops drafting (`--spec-p-min`, or what the
    /// depth policy lowers it to, [`Self::set_p_min`]).
    p_min: f32,
    /// Per sequence: the target's residual at its last position taken in.
    pending: HashMap<SeqId, Vec<f32>>,
}

impl MtpDrafter {
    pub(super) fn new(p_min: f32) -> Self {
        Self {
            p_min,
            pending: HashMap::new(),
        }
    }

    pub(super) fn set_p_min(&mut self, p_min: f32) {
        self.p_min = p_min;
    }

    pub(super) fn release(&mut self, seq: SeqId) {
        self.pending.remove(&seq);
    }

    /// Starts reading the per-layer n-gram rows of `r`'s newest token for the verify step
    /// (qwen4exp's table on the SSD: `--ngram-io direct`): the bonus token before any draft,
    /// else the last draft. The verify step feeds all of them; read here, their reads overlap
    /// the head's passes instead of delaying the verify step's input fill (~1.5 ms on the GB10).
    /// A hint: models without the table ignore it.
    fn prefetch_rows(exec: &Executor, r: &DraftRequest<'_>) {
        let fed = &r.tokens[..r.start + r.ran + 1];
        let all: Vec<u32> = fed[fed.len().saturating_sub(8)..]
            .iter()
            .chain(&r.out.tokens)
            .copied()
            .collect();
        let (tok, before) = all.split_last().expect("the bonus token at least");
        exec.model()
            .prefetch_ngram(before, std::slice::from_ref(tok));
    }

    /// Returns the head passes it ran after any long take-in's pieces: the take-in (with the
    /// first drafts) and one per further draft.
    pub(super) fn step(
        &mut self,
        exec: &mut Executor,
        batch: &mut [DraftRequest<'_>],
        blocks: &[&[u32]],
        rings: &[usize],
    ) -> Result<usize, String> {
        if exec.mtp_shares_kv() {
            return self.step_shared(exec, batch, blocks, rings);
        }
        let w = exec.mtp_width();
        for r in batch.iter() {
            assert_eq!(
                r.hidden[0].len(),
                r.ran * w,
                "one residual row per position taken in"
            );
        }
        // A long take-in (a prompt chunk) goes first in pieces without logits.
        let mut skip = vec![0usize; batch.len()];
        let t0 = std::time::Instant::now();
        for (i, r) in batch.iter().enumerate() {
            let hidden = r.hidden[0];
            while r.ran - skip[i] > TAKE_IN_TAIL {
                let at = skip[i];
                let n = TAKE_IN_PIECE.min(r.ran - at - TAKE_IN_TAIL);
                // The pending row, then the piece's rows but its last, straight from the step's
                // output into the head's input (#435: no 40 MB vector per piece).
                let zeros;
                let first: &[f32] = match self.pending.get(&r.seq) {
                    Some(p) => p,
                    None => {
                        zeros = vec![0.0; w];
                        &zeros
                    }
                };
                let rest = &hidden[at * w..(at + n - 1) * w];
                let group = Group {
                    slot: 0,
                    seqs: vec![SeqInput {
                        tokens: &r.tokens[r.start + at..r.start + at + n],
                        n_past: r.start + at,
                        blocks: blocks[i],
                        ring: rings[i],
                        logits: Logits::None,
                        media: None,
                    }],
                    verify: false,
                };
                exec.mtp_step_parts(&[group], &[first, rest])
                    .map_err(|e| e.to_string())?;
                self.pending
                    .insert(r.seq, hidden[(at + n - 1) * w..(at + n) * w].to_vec());
                skip[i] += n;
            }
        }
        if skip.iter().any(|&n| n > 0) {
            tracing::debug!(
                positions = skip.iter().sum::<usize>(),
                ms = t0.elapsed().as_secs_f64() * 1e3,
                "mtp take-in pieces"
            );
        }
        // The bonus tokens' n-gram rows, read while the head runs.
        for r in batch.iter().filter(|r| r.max > 0) {
            Self::prefetch_rows(exec, r);
        }
        // Take in, and the first draft. The sequences go in order of their rows (stable), so
        // the pass's shape is the multiset of their take-in rows, not their order: at 4
        // sequences 15 shapes instead of 81, which the head keeps rather than building and
        // capturing a graph for most passes (#858). Each sequence's rows are the same; only
        // where they sit in the pass moves, but that moves bits with many rows (at 16 sequences
        // batch order gives other drafts than this order, #858). `--mtp-ahead` keeps this
        // order (its further passes go in batch order, as below); a round sent to other
        // machines (`--tp-mtp-chain`) keeps the batch's.
        let tp_chain = exec.mtp_chains();
        let chained = tp_chain || exec.mtp_ahead();
        let rows_of = |i: usize| batch[i].ran - skip[i] + usize::from(batch[i].max > 0);
        let mut order: Vec<usize> = (0..batch.len()).collect();
        if take_in_sorted() && !tp_chain {
            order.sort_by_key(|&i| rows_of(i));
        }
        let mut fed: Vec<&[u32]> = Vec::with_capacity(batch.len());
        let mut h = Vec::new();
        for &i in &order {
            let r = &batch[i];
            let start = r.start + skip[i];
            let rows = rows_of(i);
            let tokens = r.tokens;
            fed.push(&tokens[start..start + rows]);
            let hidden = &r.hidden[0][skip[i] * w..];
            match self.pending.get(&r.seq) {
                Some(p) => h.extend_from_slice(p),
                // A new sequence (or one resumed from the prefix cache): nothing before.
                None => h.resize(h.len() + w, 0.0),
            }
            h.extend_from_slice(&hidden[..(rows - 1) * w]);
        }
        let groups: Vec<Group<'_>> = order
            .iter()
            .zip(&fed)
            .map(|(&i, tokens)| {
                let r = &batch[i];
                Group {
                    slot: 0,
                    seqs: vec![SeqInput {
                        tokens,
                        n_past: r.start + skip[i],
                        blocks: blocks[i],
                        ring: rings[i],
                        logits: if r.max > 0 {
                            Logits::Last
                        } else {
                            Logits::None
                        },
                        media: None,
                    }],
                    verify: false,
                }
            })
            .collect();
        // The same passes as below, each staging the next one where it can (#827).
        if chained {
            return self.chain(exec, batch, &groups, &order, &h);
        }
        let out = exec.mtp_step(&groups, &h).map_err(|e| e.to_string())?;
        let mut passes = 1;
        drop(groups);
        for r in batch.iter() {
            let hidden = r.hidden[0];
            self.pending
                .insert(r.seq, hidden[(r.ran - 1) * w..r.ran * w].to_vec());
        }
        // Each drafting request's output row of the pass (the rows of the groups with logits, in
        // the pass's order).
        let mut out_row = vec![usize::MAX; batch.len()];
        for (k, &i) in order.iter().filter(|&&i| batch[i].max > 0).enumerate() {
            out_row[i] = k;
        }
        // Per drafting request: (index in batch, its residual for the next draft).
        let mut live: Vec<(usize, Vec<f32>)> = Vec::new();
        for (i, r) in batch.iter_mut().enumerate() {
            if r.max == 0 {
                continue;
            }
            let k = out_row[i];
            if out.pmax[k] >= self.p_min {
                r.out.tokens.push(out.argmax[k]);
                r.out.confidence.push(out.pmax[k]);
                Self::prefetch_rows(exec, r);
                if r.max > 1 {
                    live.push((i, out.hidden[k * w..(k + 1) * w].to_vec()));
                }
            }
        }
        // Further drafts, one pass each.
        while !live.is_empty() {
            let toks: Vec<u32> = live
                .iter()
                .map(|(i, _)| *batch[*i].out.tokens.last().unwrap())
                .collect();
            let h: Vec<f32> = live.iter().flat_map(|(_, h)| h.iter().copied()).collect();
            let group = Group {
                slot: 0,
                seqs: live
                    .iter()
                    .zip(&toks)
                    .map(|((i, _), t)| {
                        let r = &batch[*i];
                        SeqInput {
                            tokens: std::slice::from_ref(t),
                            n_past: r.start + r.ran + r.out.tokens.len(),
                            blocks: blocks[*i],
                            ring: rings[*i],
                            logits: Logits::Last,
                            media: None,
                        }
                    })
                    .collect(),
                verify: false,
            };
            let out = exec.mtp_step(&[group], &h).map_err(|e| e.to_string())?;
            passes += 1;
            let mut next = Vec::new();
            for (k, (i, _)) in live.into_iter().enumerate() {
                if out.pmax[k] < self.p_min {
                    continue;
                }
                let r = &mut batch[i];
                r.out.tokens.push(out.argmax[k]);
                r.out.confidence.push(out.pmax[k]);
                Self::prefetch_rows(exec, r);
                if r.out.tokens.len() < r.max {
                    next.push((i, out.hidden[k * w..(k + 1) * w].to_vec()));
                }
            }
            live = next;
        }
        Ok(passes)
    }

    /// [`Self::step`]'s first pass (over `groups`, residual rows `h`) and further drafts as one
    /// draft round on every rank of tensor parallelism over machines (`--tp-mtp-chain`,
    /// [`Executor::mtp_chain`]): the same passes, sent as one request. `order`: the batch
    /// index of each of `groups`' sequences.
    fn chain(
        &mut self,
        exec: &mut Executor,
        batch: &mut [DraftRequest<'_>],
        groups: &[Group<'_>],
        order: &[usize],
        h: &[f32],
    ) -> Result<usize, String> {
        let w = exec.mtp_width();
        // The drafting requests, in batch order: the rows of the round, whose further passes
        // go in this order (as `step`'s do).
        let drafting: Vec<usize> = (0..batch.len()).filter(|&i| batch[i].max > 0).collect();
        // Per output row of the first pass (its drafting sequences, in `order`): its row.
        let mut row_of = vec![usize::MAX; batch.len()];
        for (k, &i) in drafting.iter().enumerate() {
            row_of[i] = k;
        }
        let first: Vec<usize> = (order.iter())
            .filter(|&&i| batch[i].max > 0)
            .map(|&i| row_of[i])
            .collect();
        let caps: Vec<u32> = drafting.iter().map(|&i| batch[i].max as u32).collect();
        let fed: Vec<&[u32]> = (drafting.iter())
            .map(|&i| &batch[i].tokens[..batch[i].start + batch[i].ran + 1])
            .collect();
        let out = exec
            .mtp_chain(
                groups,
                &[h],
                &caps,
                &first,
                self.p_min,
                |exec, k, drafts| prefetch_after(exec, fed[k], drafts),
            )
            .map_err(|e| e.to_string())?;
        for r in batch.iter() {
            let hidden = r.hidden[0];
            self.pending
                .insert(r.seq, hidden[(r.ran - 1) * w..r.ran * w].to_vec());
        }
        for (k, &i) in drafting.iter().enumerate() {
            let r = &mut batch[i];
            for &(t, p) in &out.drafts[k] {
                r.out.tokens.push(t);
                r.out.confidence.push(p);
            }
        }
        Ok(out.passes)
    }

    /// [`Self::step`] for a head that reads the target's KV cache (Gemma 4's assistant, as
    /// llama.cpp's `draft-mtp` drives a head with shared memory): nothing to take in. Every
    /// draft runs at the position of the bonus token (the target has not written it yet, so
    /// the head sees the positions before it): the first pairs the bonus token with the
    /// target's residual of the position before, each further one the last draft with the
    /// head's own output residual.
    fn step_shared(
        &mut self,
        exec: &mut Executor,
        batch: &mut [DraftRequest<'_>],
        blocks: &[&[u32]],
        rings: &[usize],
    ) -> Result<usize, String> {
        let w = exec.mtp_width();
        let mut passes = 0;
        // Per drafting request: (index in batch, the token to feed, its residual).
        let mut live: Vec<(usize, u32, Vec<f32>)> = Vec::new();
        for (i, r) in batch.iter().enumerate() {
            if r.max == 0 {
                continue;
            }
            let hidden = r.hidden[0];
            assert_eq!(hidden.len(), r.ran * w, "one residual row per position run");
            let bonus = r.tokens[r.start + r.ran];
            live.push((i, bonus, hidden[(r.ran - 1) * w..r.ran * w].to_vec()));
        }
        while !live.is_empty() {
            let toks: Vec<u32> = live.iter().map(|(_, t, _)| *t).collect();
            let h: Vec<f32> = live
                .iter()
                .flat_map(|(_, _, h)| h.iter().copied())
                .collect();
            let group = Group {
                slot: 0,
                seqs: live
                    .iter()
                    .zip(&toks)
                    .map(|((i, _, _), t)| {
                        let r = &batch[*i];
                        SeqInput {
                            tokens: std::slice::from_ref(t),
                            n_past: r.start + r.ran,
                            blocks: blocks[*i],
                            ring: rings[*i],
                            logits: Logits::Last,
                            media: None,
                        }
                    })
                    .collect(),
                verify: false,
            };
            let out = exec.mtp_step(&[group], &h).map_err(|e| e.to_string())?;
            passes += 1;
            let mut next = Vec::new();
            for (k, (i, _, _)) in live.into_iter().enumerate() {
                if out.pmax[k] < self.p_min {
                    continue;
                }
                let r = &mut batch[i];
                r.out.tokens.push(out.argmax[k]);
                r.out.confidence.push(out.pmax[k]);
                if r.out.tokens.len() < r.max {
                    next.push((i, out.argmax[k], out.hidden[k * w..(k + 1) * w].to_vec()));
                }
            }
            live = next;
        }
        Ok(passes)
    }
}

#[cfg(test)]
mod tests;
