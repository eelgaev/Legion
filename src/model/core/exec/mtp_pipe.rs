//! MTP draft passes queued one ahead of the host's stop decision (#827).
//!
//! A draft round stops a row at the first pick the head gives less than `p_min`, so the host
//! read each pass's picks before it ran the next one, and the GPU sat idle for that round
//! trip and the next pass's launch (~0.1-0.2 ms a pass on the GB10). With
//! [`super::mtp_ahead`] a pass already stages the inputs of the pass after it on the device
//! (its picks and output residual copied into the next graph's inputs), as if every row went
//! on. Here a pass that staged them is left running ([`Pass::Flying`]: its reads queued behind
//! a [`Fence`]) and the staged pass is queued right behind it, before the host has seen a
//! pick. Then the host reads the first pass's picks and decides as before: if every row went
//! on, the queued pass is the one it would run now, on the same graph and the same input bytes,
//! so drafts and outputs do not change; else its outputs are dropped (the cells it wrote are
//! past the kept drafts, which the next round's take-in rewrites before any pass reads them,
//! as for a rejected draft) and the pass the host decides on runs after it.
//!
//! A queued pass runs behind a CUDA conditional graph node where it can
//! ([`crate::ggml::stage_graph::cond`]): a one-thread kernel checks that every row of the pass
//! before reached `p_min`, so a pass the host will drop is skipped on the GPU (a few µs) rather
//! than run (~1-1.5 ms on a GB10), and writes no cell. Where that graph is refused (the pass's
//! first runs, before its stage graph is captured, or CUDA before 12.3) it runs plainly and a
//! dropped pass costs a pass. `--mtp-pipe-rows` sets the most rows a pass queued this way may
//! have (0, the default: off).

use std::time::Instant;

use super::mtp_ahead::Ahead;
use super::{Executor, Group, GroupShape, ModelError};
use crate::ggml::model_bytes::Kind;
use crate::ggml::pinned_io::{Fence, Read};
use crate::model::core::mtp::MtpOutput;

/// What [`Executor::mtp_launch`] left.
pub(super) enum Pass {
    /// Ran, its outputs read.
    Done(MtpOutput),
    /// Running, the next pass's inputs staged behind it ([`Executor::mtp_land`] reads it).
    Flying(InFlight),
    /// A pass on staged inputs that were not its own: nothing ran.
    Skipped,
}

/// An MTP pass left running.
pub(super) struct InFlight {
    /// After its reads.
    pub fence: Fence,
    /// Its picks, their probabilities and its output residual, queued into pinned memory.
    pub reads: [Read<'static>; 3],
    /// The target step's cache work it took, as its kind of step.
    pub credit: Option<Kind>,
    /// Its tokens.
    pub n: usize,
    pub shape: Vec<GroupShape>,
}

impl Executor {
    /// The most rows of an MTP draft pass queued before the host has the picks of the pass
    /// before it (`--mtp-pipe-rows`, see the module docs; 0 for none, the default). Only
    /// where the passes stage each other ([`Self::mtp_ahead`]).
    pub fn set_mtp_pipe_rows(&mut self, rows: usize) {
        self.mtp_pipe_rows = rows;
    }

    /// See [`Self::set_mtp_pipe_rows`]: 0 where passes do not stage each other.
    pub(super) fn mtp_pipe_rows(&self) -> usize {
        if self.mtp_ahead() {
            self.mtp_pipe_rows
        } else {
            0
        }
    }

    /// An MTP pass of `groups` (residual rows `parts`, as [`Self::mtp_step_parts`]), left
    /// running where `ahead.pipe` lets it and it stages the next pass; with `ahead.spec`, a
    /// pass on the staged inputs alone (else [`Pass::Skipped`]).
    pub(super) fn mtp_launch(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
        ahead: Ahead<'_, '_>,
    ) -> Result<Pass, ModelError> {
        if self.budget_chunks(groups).len() > 1 {
            if ahead.spec {
                self.mtp_staged = None;
                return Ok(Pass::Skipped);
            }
            return self.mtp_step_ahead(groups, parts, ahead).map(Pass::Done);
        }
        let pass = self.mtp_pass_fly(groups, parts, ahead)?;
        if ahead.spec && !matches!(pass, Pass::Skipped) {
            static FIRST: std::sync::Once = std::sync::Once::new();
            FIRST.call_once(|| tracing::info!("mtp passes queued before the last one's picks"));
        }
        Ok(pass)
    }

    /// Waits for a pass left running ([`Pass::Flying`]) and returns its outputs; `since`: when
    /// its time on the GPU began, as far as the host can tell (its launch, or the last pass's
    /// landing), for the model bytes.
    pub(super) fn mtp_land(
        &mut self,
        f: InFlight,
        since: Instant,
    ) -> Result<MtpOutput, ModelError> {
        let InFlight {
            fence,
            reads: [argmax, pmax, hidden],
            credit,
            n,
            shape,
        } = f;
        let io = &mut self.stages.last_mut().expect("at least one stage").mtp_io;
        io.wait(fence);
        let out = MtpOutput {
            argmax: (io.take_i32(argmax).into_iter())
                .map(|x| x as u32)
                .collect(),
            pmax: io.take_f32(pmax),
            hidden: io.take_f32(hidden),
        };
        self.mtp_pass_after(credit, n)?;
        // Model bytes (#391), as `mtp_pass` counts them.
        let slot = if shape.iter().all(|g| g.n_tok == 1) {
            &self.mtp_decode
        } else {
            &self.mtp_mixed
        };
        if let Some(kept) = slot.get(&shape) {
            self.bytes
                .draft
                .add(kept.model_bytes(), since.elapsed().as_secs_f64());
        }
        Ok(out)
    }

    /// The outputs of a pass [`Self::mtp_launch`] ran or left running (landing it); `since`
    /// as [`Self::mtp_land`] takes it, then now.
    pub(super) fn mtp_output(
        &mut self,
        pass: Pass,
        since: &mut Instant,
    ) -> Result<MtpOutput, ModelError> {
        let out = match pass {
            Pass::Done(out) => out,
            Pass::Flying(f) => self.mtp_land(f, *since)?,
            Pass::Skipped => unreachable!("a skipped pass has no outputs"),
        };
        *since = Instant::now();
        Ok(out)
    }
}
