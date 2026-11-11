//! A decode or verify step's next stage queued while the stage before it computes (#814).
//!
//! A step of one micro-batch runs its stages one after another, one round each. The next
//! stage's inputs are already filled while the stage before computes (`step_fill`), but its
//! graph was launched only once the host had seen the stage before finish: the host woke from
//! that stage's synchronize, read its rows, copied the residual over (a copy it waited for),
//! started the expert cache's pass, checked the stage graph's words (~40-45 µs) and launched
//! it, with every GPU idle meanwhile. On Qwen3.8-Flash-Next over four V100s with MTP drafting
//! (C=1, `--deterministic`) those hand-offs were about 0.4-0.5 ms of each ~30 ms verify step
//! (#814's profile: `handoff copy->graph`, ~120 µs per hand-off, all of it host time).
//!
//! Here, once stage `s` is launched and stage `s + 1`'s inputs are queued, the host also
//! queues the residual's copy on stage `s`'s stream (the next GPU's stream waits for it on an
//! event, [`Executor::hand_off_queued`]), starts the expert cache's pass for stage `s + 1`'s
//! layers and launches its graph, all before it waits for stage `s`. The GPUs then go from
//! one stage to the next without the host in between. Every stage runs the same graph on the
//! same inputs and residual, in the same order on its stream, so the outputs are
//! byte-identical.
//!
//! Only for steps whose stages take their inputs as queued copies (`step_fill::queued`: no
//! prompt, no pipelined micro-chunks, no pair or split-model help), and not with
//! `--expert-peer-fetch`, whose idle-peer help belongs to the stage that runs alone. On by
//! default; `legion serve --no-stage-launch-ahead` keeps the launch after the wait.

use super::{Executor, GroupShape, Kind, ModelError};

#[cfg(test)]
mod tests;

/// Whether stage `s + 1` of a step of `n_stages` is launched while stage `s` computes (see
/// the module docs): the setting (`on`), queued inputs (`queued`), a next stage, and no idle
/// peer help (`peer_fetch`).
pub(super) fn wanted(on: bool, queued: bool, s: usize, n_stages: usize, peer_fetch: bool) -> bool {
    on && queued && s + 1 < n_stages && !peer_fetch
}

impl Executor {
    /// See [`wanted`] (with this executor's setting and the peer fetch as it is now).
    pub(super) fn launches_ahead(&self, queued: bool, s: usize) -> bool {
        wanted(
            self.launch_ahead,
            queued,
            s,
            self.stages.len(),
            super::super::expert_mv::peer::enabled(),
        )
    }

    /// Queues stage `s`'s residual hand-off, starts the expert cache's pass over its layers and
    /// launches its graph for `shape`, while stage `s - 1` (launched, not waited for) computes.
    /// Its inputs must be filled (queued) already. On an error the caller synchronizes both
    /// stages before it returns.
    pub(super) fn launch_ahead(
        &mut self,
        s: usize,
        shape: &[GroupShape],
        kind: Kind,
    ) -> Result<(), ModelError> {
        // The last stage's share of the last step's cache work runs before its graph is queued
        // (the round loop otherwise runs it at the start of each later round).
        self.flush_deferred_tail()?;
        self.hand_off_queued(s, shape);
        // The stages that run nothing now: neither this one nor the one still computing.
        let n_stages = self.stages.len();
        if let Some(r) = self.model.experts_mut() {
            r.set_idle((0..n_stages).filter(|&i| i != s && i + 1 != s).collect());
        }
        let layers = self.stages[s].layers.clone();
        let n = shape.iter().map(|g| g.tokens()).sum();
        let begun = self.begin_pass(n, |il| layers.contains(&il), kind);
        if let Some(r) = self.model.experts_mut() {
            r.set_idle(Vec::new());
        }
        begun?;
        self.launch_stage(s, shape, n_stages == 1)
    }

    /// Launches stage `s`'s kept graph for `shape`, to completion when `sync` (one stage, as
    /// before #836's verify graphs built ahead) or without waiting for it, its late n-gram rows
    /// (if the fill left any) written while its first layers run.
    pub(super) fn launch_stage(
        &mut self,
        s: usize,
        shape: &[GroupShape],
        sync: bool,
    ) -> Result<(), ModelError> {
        let parts = self.parts(s);
        let late_rows = self.model.take_ple_late();
        let model = &self.model;
        let super::Stage {
            decode,
            mixed,
            backend,
            ..
        } = &mut self.stages[s];
        let kept = super::kept_in(decode, mixed, shape).expect("prepared");
        // The n-gram rows the fill left: read while the first layer computes.
        let took = std::cell::Cell::new(std::time::Duration::ZERO);
        let took_ref = &took;
        let mut write = late_rows.map(|rows| {
            let t = super::Inputs::from_parts(shape, parts, |i| kept.tensor(i))
                .ple
                .expect("a step with n-gram rows has their input");
            let mut rows = Some(rows);
            (t, move || {
                if let Some(rows) = rows.take() {
                    let t0 = std::time::Instant::now();
                    model.write_ple(t, rows);
                    took_ref.set(t0.elapsed());
                }
            })
        });
        let late = write.as_mut().map(|(t, f)| crate::ggml::LateInput {
            tensor: *t,
            fill: f,
        });
        let r = backend.compute_kept_with(kept, sync, late);
        drop(write);
        self.late_rows_took = took.get();
        r?;
        Ok(())
    }
}
