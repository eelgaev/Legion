//! DFlash 2 (`DFlash2DraftModel`, e.g. `z-lab/Qwen3.8-27B-DFlash2`): DFlash plus two dynamic
//! causal convolutions per layer and a candidate selector over the block.
//!
//! - **Convolutions.** Around the attention and around the MLP of every layer, the input is
//!   convolved before and the output after, causally along the noise block (position `t` mixes
//!   `t` and the `kernel - 1` positions before it; the first block position has nothing before
//!   it, the context does not take part). Each tap's coefficient is a per-channel base kernel
//!   plus a dynamic coefficient per group of `group_size` channels, which a linear projection
//!   of the layer input gives per position: `kernel_projection(x)` is `[2 sides][kernel][groups]`
//!   per position, side 0 for the input convolution ("prepare"), side 1 for the output one
//!   ("finish"), both from the same `x`.
//! - **Selector.** The output head's top `top_k` tokens of every draft position are the
//!   candidates, their logits the unary scores. A pairwise score between the previous draft
//!   (the last token for the first) and each candidate is added:
//!   `sum_r predecessor[prev][r] * hidden_projection(h)[r] * successor[candidate][r]`, with two
//!   `vocab × rank` codebooks. The drafts are chosen in order: the best candidate (or one
//!   sampled at the draft temperature from the softmax over the candidates' scores), which is
//!   the next position's predecessor.
//!
//! The candidates, their logits and the hidden projection come out of the block pass; the walk
//! over positions runs on the host with the codebooks there (`top_k` rows of `rank` per
//! position: nothing to gain on the device).
//!
//! Adapted from z-lab's DFlash reference, `dflash/model.py` (`GroupedDynamicCausalConv`,
//! `CandidateSelector`; MIT, <https://github.com/z-lab/dflash>; the checkpoints are
//! Apache-2.0), and llama.cpp's `src/models/dflash.cpp` (`build_dflash2_conv`, the GGUF tensor
//! and metadata names; MIT).

use crate::engine::sampling::Rng;
use crate::ggml::{Context, Tensor, ffi};

use super::model::floats;

/// DFlash 2's hyperparameters beyond DFlash's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dflash2 {
    /// Taps of each convolution (the position itself and the ones before it).
    pub conv_kernel: usize,
    /// Channels that share one dynamic coefficient.
    pub conv_group: usize,
    /// Rank of the selector's codebooks and hidden projection.
    pub selector_rank: usize,
    /// Candidates per draft position.
    pub selector_top_k: usize,
}

impl Dflash2 {
    /// Groups of channels of a hidden size of `n_embd`.
    pub fn groups(&self, n_embd: usize) -> usize {
        n_embd / self.conv_group
    }

    /// Rows of a `kernel_projection` output: two sides of `kernel` taps of `groups` each.
    pub fn projected(&self, n_embd: usize) -> usize {
        2 * self.conv_kernel * self.groups(n_embd)
    }
}

/// `x` (`[n_embd, n]`, the `n` block positions) through one side of a grouped dynamic causal
/// convolution: `out[c, t] = sum_tap (base[side][tap][c] + dynamic[t][side][tap][c / group])
/// * x[c, t - tap]`, with `x[., t] = 0` for `t < 0`.
///
/// `dynamic` is the layer's `kernel_projection` output `[2 · kernel · groups, n]`; `base` is the
/// F32 base kernel `[n_embd, kernel, 2]`.
///
/// `x` may hold several sequences' blocks of `block` positions each (a batched draft pass):
/// each block is convolved on its own (its first position sees only padding before it).
pub fn conv<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    dynamic: Tensor<'a>,
    base: Tensor<'a>,
    side: usize,
    v2: &Dflash2,
    block: i64,
) -> Tensor<'a> {
    let [e, n, _, _] = x.ne();
    assert!(
        block > 0 && n % block == 0,
        "{n} positions in blocks of {block}"
    );
    let blocks = n / block;
    let (k, group) = (v2.conv_kernel as i64, v2.conv_group as i64);
    let groups = e / group;
    let coeffs = ctx.reshape_4d(dynamic, [groups, k, 2, n]);
    let cnb = coeffs.nb();
    let es = base.element_size();
    let mut out: Option<Tensor<'a>> = None;
    // Taps at or past the block length read only the zero padding.
    for tap in 0..k.min(block) {
        let values = if tap == 0 {
            x
        } else if blocks == 1 {
            let before = ctx.view_2d(x, [e, n - tap], x.nb()[1], 0);
            // SAFETY: a contiguous view of this context; pads `tap` zero columns in front.
            ctx.wrap(unsafe {
                ffi::ggml_pad_ext(ctx.raw(), before.raw(), 0, 0, tap as i32, 0, 0, 0, 0, 0)
            })
        } else {
            // Every block's first `block - tap` positions, shifted by `tap` within the block.
            let x3 = ctx.reshape_3d(x, e, block, blocks);
            let nb = x3.nb();
            let before = ctx.cont(ctx.view_3d(x3, [e, block - tap, blocks], [nb[1], nb[2]], 0));
            // SAFETY: a contiguous tensor of this context; pads `tap` zero columns in front of
            // every block.
            let padded = ctx.wrap(unsafe {
                ffi::ggml_pad_ext(ctx.raw(), before.raw(), 0, 0, tap as i32, 0, 0, 0, 0, 0)
            });
            ctx.reshape_2d(padded, e, n)
        };
        let values = ctx.reshape_3d(values, group, groups, n);
        // The dynamic coefficient of every group and position, `[1, groups, n]`.
        let dynamic = ctx.view_2d(
            coeffs,
            [groups, n],
            cnb[3],
            side * cnb[2] + tap as usize * cnb[1],
        );
        let dynamic = ctx.reshape_3d(ctx.cont(dynamic), 1, groups, n);
        // The base kernel of this side and tap, `[group, groups, 1]`.
        let at = (side * k as usize + tap as usize) * e as usize * es;
        let base = ctx.reshape_3d(
            ctx.view_2d(base, [e, 1], e as usize * es, at),
            group,
            groups,
            1,
        );
        let term = ctx.add(ctx.mul(values, dynamic), ctx.mul(values, base));
        out = Some(match out {
            Some(o) => ctx.add(o, term),
            None => term,
        });
    }
    ctx.reshape_2d(out.expect("at least one tap"), e, n)
}

/// The selector's codebooks, on the host: one row of each is read per candidate and position.
pub struct Selector {
    pub rank: usize,
    pub top_k: usize,
    ty: ffi::ggml_type,
    predecessor: Vec<u8>,
    successor: Vec<u8>,
}

/// What [`Selector::walk`] picked: the draft of every position, and with a temperature the
/// distribution over the candidates each was drawn from.
pub struct Walk {
    pub tokens: Vec<u32>,
    /// Per position, `(candidate token, probability)` pairs.
    pub probs: Option<Vec<Vec<(u32, f32)>>>,
}

impl Selector {
    /// Codebooks `[rank, vocab]` (ggml order) of type `ty` as their raw bytes.
    pub fn new(
        rank: usize,
        top_k: usize,
        ty: ffi::ggml_type,
        predecessor: Vec<u8>,
        successor: Vec<u8>,
    ) -> Self {
        Self {
            rank,
            top_k,
            ty,
            predecessor,
            successor,
        }
    }

    fn row(&self, book: &[u8], token: u32) -> Vec<f32> {
        let bytes = crate::ggml::row_size(self.ty, self.rank);
        let at = token as usize * bytes;
        floats(self.ty, &book[at..at + bytes], self.rank)
    }

    /// The pairwise scores of `candidates` after `prev`, given a position's hidden projection.
    fn pair_scores(&self, prev: u32, proj: &[f32], candidates: &[u32]) -> Vec<f32> {
        let gate: Vec<f32> = self
            .row(&self.predecessor, prev)
            .iter()
            .zip(proj)
            .map(|(a, b)| a * b)
            .collect();
        candidates
            .iter()
            .map(|&c| {
                let s = self.row(&self.successor, c);
                gate.iter().zip(&s).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// Chooses the drafts of `n` positions in order, from the last token `anchor`:
    /// `candidates` and `unary` are `[n][top_k]` (the head's top tokens and their logits),
    /// `proj` is `[n][rank]`. With `temperature` 0 each draft is the best-scoring candidate
    /// (the first on a tie); above 0 it is drawn from `softmax(scores / temperature)`.
    pub fn walk(
        &self,
        anchor: u32,
        candidates: &[u32],
        unary: &[f32],
        proj: &[f32],
        temperature: f32,
        rng: &mut Rng,
    ) -> Walk {
        let k = self.top_k;
        let n = candidates.len() / k;
        let mut prev = anchor;
        let mut tokens = Vec::with_capacity(n);
        let mut probs = (temperature > 0.0).then(Vec::new);
        for p in 0..n {
            // The device's top-k comes in no particular order: best logit first (lower token
            // on a tie), so that ties between scores resolve the same on every backend.
            let mut order: Vec<usize> = (p * k..(p + 1) * k).collect();
            order.sort_by(|&a, &b| {
                unary[b]
                    .total_cmp(&unary[a])
                    .then(candidates[a].cmp(&candidates[b]))
            });
            let cand: Vec<u32> = order.iter().map(|&i| candidates[i]).collect();
            let pair = self.pair_scores(prev, &proj[p * self.rank..(p + 1) * self.rank], &cand);
            let scores: Vec<f32> = pair
                .iter()
                .zip(order.iter().map(|&i| unary[i]))
                .map(|(a, b)| a + b)
                .collect();
            let pick = match probs.as_mut() {
                None => {
                    let mut best = 0;
                    for (i, &s) in scores.iter().enumerate() {
                        if s > scores[best] {
                            best = i;
                        }
                    }
                    best
                }
                Some(all) => {
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut q: Vec<f32> = scores
                        .iter()
                        .map(|&s| ((s - max) / temperature).exp())
                        .collect();
                    let sum: f32 = q.iter().sum();
                    q.iter_mut().for_each(|x| *x /= sum);
                    let pick = super::sample(&q, rng.next_f64());
                    all.push(cand.iter().copied().zip(q).collect());
                    pick
                }
            };
            prev = cand[pick];
            tokens.push(prev);
        }
        Walk { tokens, probs }
    }
}
