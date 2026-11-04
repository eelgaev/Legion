//! A draft round of MTP passes as one request to the other ranks of tensor parallelism over
//! machines (`--tp-mtp-chain`, #771).
//!
//! Without it, rank 0 sends every pass of a round ([`Executor::mtp_step`]) on its control
//! connection, and each pass's first all-reduce waits until the other ranks have received and
//! started it. But a pass's outputs (the head's pick, its probability and the output residual)
//! are the same bits on every rank (the sums add every rank's partial in rank order, and the
//! output head is mirrored), so every rank can decide the round's further passes itself: rank 0
//! sends the first pass with each output row's cap and the confidence floor, and every rank runs
//! [`Executor::mtp_chain`] to the same end, passes in step, without another request.
//!
//! The passes are those `MtpDrafter::step` (`crate::engine::lm`) runs: after the first pass,
//! each row whose pick reaches `p_min` keeps it as a draft and, below its cap, runs one more
//! pass with that draft at the next position, fed the head's own output residual.

use std::time::Instant;

use super::mtp_ahead::{self, Ahead};
use super::mtp_pipe::Pass;
use super::{Executor, Group, Logits, ModelError, SeqInput};
use crate::node::Request;

#[cfg(test)]
mod tests;

/// What a draft round ran ([`Executor::mtp_chain`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChainOutput {
    /// Per row of the round (the order of its caps, [`Executor::mtp_chain`]): the drafts and
    /// the head's probability for each.
    pub drafts: Vec<Vec<(u32, f32)>>,
    /// The head passes run, the first included.
    pub passes: usize,
}

/// One output row of the first pass: where its next draft goes.
#[derive(Clone, Copy)]
struct Row<'b> {
    /// The position of the first pass's last token (the bonus token): draft `d` (from 1) is
    /// fed at `at + d`.
    at: usize,
    blocks: &'b [u32],
    ring: usize,
}

/// The pass after one over the rows `(k, drafts)` (row `k` of `rows` with `drafts` drafts
/// before that pass), should every row's pick reach the floor (`mtp_ahead`): `None` if a row
/// would be at its cap of `caps`.
fn next_pass<'b>(
    rows: &[Row<'b>],
    caps: &[u32],
    drafted: impl Iterator<Item = (usize, usize)>,
) -> Option<Group<'b>> {
    mtp_ahead::predict(drafted.map(|(k, d)| {
        let d = d + 1;
        (d < caps[k] as usize).then_some((rows[k].at + d, rows[k].blocks, rows[k].ring))
    }))
}

/// Where each row of the round goes in the first pass, given `first` (per output row of the
/// first pass, in its order, the row's index in the round's order): `None` unless `first` is
/// a permutation of `0..first.len()`.
fn first_pass_rows(first: &[usize]) -> Option<Vec<usize>> {
    let mut at = vec![usize::MAX; first.len()];
    for (j, &k) in first.iter().enumerate() {
        if *at.get(k)? != usize::MAX {
            return None;
        }
        at[k] = j;
    }
    Some(at)
}

impl Executor {
    /// Whether this executor leads other ranks that run draft rounds as chains
    /// (`--tp-mtp-chain`): [`Self::mtp_chain`] then sends one request per round.
    pub fn mtp_chains(&self) -> bool {
        self.tp.as_ref().is_some_and(|l| l.chains())
    }

    /// A draft round: the first pass over `groups` (their residual rows in `parts`, as
    /// [`Self::mtp_step_parts`]), then, per row `k` of the round, further passes while the
    /// head's pick reaches `p_min` and the row has fewer than `caps[k]` drafts. `first`: per
    /// output row of the first pass (the sequences with logits, in order), its row `k`; the
    /// further passes run their rows in the order of `k`, so a first pass whose sequences are
    /// ordered otherwise (`MtpDrafter`'s take-in sort, `crate::engine::lm`) is followed by the
    /// passes of a round without that order. `on_draft(exec, k, drafts)` sees each row's
    /// drafts so far after each new one (rank 0 starts reading the drafts' n-gram rows there).
    /// Every pass runs on this rank only, except that a rank 0 of [`Self::mtp_chains`] first
    /// sends the round to the other ranks, which run the same passes (there `first` must be
    /// `0..`: the request carries no order).
    pub fn mtp_chain(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
        caps: &[u32],
        first: &[usize],
        p_min: f32,
        on_draft: impl FnMut(&Executor, usize, &[(u32, f32)]),
    ) -> Result<ChainOutput, ModelError> {
        if !self.mtp_chains() {
            return self.mtp_chain_here(groups, parts, caps, first, p_min, on_draft);
        }
        if first.iter().enumerate().any(|(j, &k)| j != k) {
            return Err(ModelError::Node(
                "an MTP chain over machines with its first pass in another order".into(),
            ));
        }
        let req = Request::TpMtpChain {
            groups: super::remote::to_wire(groups),
            gather_budget: self.gather_budget as u64,
            h: parts.concat(),
            caps: caps.to_vec(),
            p_min: p_min.to_bits(),
        };
        self.tp_run(&req, |e| {
            e.mtp_chain_here(groups, parts, caps, first, p_min, on_draft)
        })
    }

    /// [`Self::mtp_chain`] on this rank.
    fn mtp_chain_here(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
        caps: &[u32],
        first: &[usize],
        p_min: f32,
        mut on_draft: impl FnMut(&Executor, usize, &[(u32, f32)]),
    ) -> Result<ChainOutput, ModelError> {
        let in_pass: Vec<Row<'_>> = (groups.iter())
            .flat_map(|g| &g.seqs)
            .filter(|s| !matches!(s.logits, Logits::None))
            .map(|s| Row {
                at: s.n_past + s.tokens.len() - 1,
                blocks: s.blocks,
                ring: s.ring,
            })
            .collect();
        if in_pass.len() != caps.len() || first.len() != caps.len() {
            return Err(ModelError::Node(format!(
                "an MTP chain of {} output rows with {} caps and {} rows in order",
                in_pass.len(),
                caps.len(),
                first.len()
            )));
        }
        let Some(at) = first_pass_rows(first) else {
            return Err(ModelError::Node(format!(
                "an MTP chain's first pass order {first:?} is not one of its rows"
            )));
        };
        // The rows in the round's order.
        let rows: Vec<Row<'_>> = at.iter().map(|&j| in_pass[j]).collect();
        let in_order = first.iter().enumerate().all(|(j, &k)| j == k);
        let w = self.mtp_width();
        // Each pass stages the next one, should every row go on (`mtp_ahead`), and one of few
        // enough rows is left running while that one is queued (`mtp_pipe`).
        let ahead = self.mtp_ahead();
        let pipe = self.mtp_pipe_rows();
        let piped = |g: Option<&Group<'_>>| g.is_some_and(|g| g.seqs.len() <= pipe);
        // A first pass in another order than the round's does not stage the second pass: the
        // staging copies its output rows as they lie, and the second pass takes them in the
        // round's order (the order the passes run in without staging, so the same bits).
        let mut pred = (ahead && in_order)
            .then(|| next_pass(&rows, caps, (0..rows.len()).map(|k| (k, 0))))
            .flatten();
        let mut since = Instant::now();
        let mut cur = self.mtp_launch(
            groups,
            parts,
            Ahead {
                next: pred.as_ref().map(std::slice::from_ref),
                staged: false,
                pipe: piped(pred.as_ref()),
                spec: false,
                p_min,
            },
        )?;
        let mut out = ChainOutput {
            drafts: vec![Vec::new(); rows.len()],
            passes: 1,
        };
        // The rows of the pass `cur` (indices into `rows`), in its order.
        let mut cur_rows: Vec<usize> = first.to_vec();
        loop {
            // Left running, it staged `pred`: queue that before reading its picks.
            let mut spec = None;
            if let (Pass::Flying(_), Some(p)) = (&cur, &pred) {
                let drafted = cur_rows.iter().map(|&k| (k, out.drafts[k].len() + 1));
                let after = next_pass(&rows, caps, drafted);
                let s = self.mtp_launch(
                    std::slice::from_ref(p),
                    &[],
                    Ahead {
                        next: after.as_ref().map(std::slice::from_ref),
                        staged: true,
                        pipe: piped(after.as_ref()),
                        spec: true,
                        p_min,
                    },
                )?;
                if !matches!(s, Pass::Skipped) {
                    spec = Some((s, after));
                }
            }
            let o = self.mtp_output(cur, &mut since)?;
            // Per row going on: its index and its residual for the next pass.
            let mut live: Vec<(usize, &[f32])> = Vec::new();
            let mut all_on = true;
            for (j, &k) in cur_rows.iter().enumerate() {
                if o.pmax[j] < p_min {
                    all_on = false;
                    continue;
                }
                out.drafts[k].push((o.argmax[j], o.pmax[j]));
                on_draft(self, k, &out.drafts[k]);
                if out.drafts[k].len() < caps[k] as usize {
                    live.push((k, &o.hidden[j * w..(j + 1) * w]));
                } else {
                    all_on = false;
                }
            }
            // The further passes in the round's order (the first pass's may differ; after it
            // the rows are in order already).
            live.sort_by_key(|&(k, _)| k);
            if let Some((s, after)) = spec {
                if all_on {
                    out.passes += 1;
                    // The pass queued is the one to run now: the same graph on the same input
                    // bytes (`pred` with these picks and residual rows).
                    cur = s;
                    pred = after;
                    cur_rows = live.into_iter().map(|(k, _)| k).collect();
                    continue;
                }
                // A row stopped: the queued pass is not the next one; its outputs are dropped
                // (the GPU skipped it, where its condition ran: `mtp_pipe`). Not counted.
                self.mtp_output(s, &mut since)?;
                self.mtp_staged = None;
            }
            if live.is_empty() {
                break;
            }
            let toks: Vec<u32> = live
                .iter()
                .map(|&(k, _)| out.drafts[k].last().expect("a live row drafted").0)
                .collect();
            let h: Vec<f32> = live.iter().flat_map(|(_, h)| h.iter().copied()).collect();
            let group = Group {
                slot: 0,
                seqs: live
                    .iter()
                    .zip(&toks)
                    .map(|(&(k, _), t)| SeqInput {
                        tokens: std::slice::from_ref(t),
                        n_past: rows[k].at + out.drafts[k].len(),
                        blocks: rows[k].blocks,
                        ring: rows[k].ring,
                        logits: Logits::Last,
                        media: None,
                    })
                    .collect(),
                verify: false,
            };
            pred = ahead
                .then(|| {
                    let drafted = live.iter().map(|&(k, _)| (k, out.drafts[k].len()));
                    next_pass(&rows, caps, drafted)
                })
                .flatten();
            cur = self.mtp_launch(
                &[group],
                &[&h],
                Ahead {
                    next: pred.as_ref().map(std::slice::from_ref),
                    staged: ahead,
                    pipe: piped(pred.as_ref()),
                    spec: false,
                    p_min,
                },
            )?;
            out.passes += 1;
            cur_rows = live.into_iter().map(|(k, _)| k).collect();
        }
        Ok(out)
    }
}
