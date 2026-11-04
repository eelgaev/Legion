//! Pair parallelism's steps (`--pair-tp`, #668, [`crate::model::core::pair`]): a decode or
//! verify step of one micro-batch runs its stages one after another, so while stage `s` computes,
//! the other GPU of its NVLink pair (stage `s ^ 1`) runs a *helper* graph beside it: half of
//! each of the stage's MoE layers' routed experts.
//!
//! The executor marks such steps' shapes ([`GroupShape::pair`]), launches the helper before the
//! stage's own graph (both asynchronous: each waits on the other only on its GPU), and waits for
//! both. The expert cache learns a copy's routing (its route log, written by the helper) once
//! per step that ran it, with its layer's ([`PairState::widen`]).

use super::*;

/// Helper graphs each stage keeps (one per token count: decode, and verify steps' few sizes).
const HELPS_KEPT: usize = 8;

/// The executor's side of pair parallelism.
#[derive(Default)]
pub(super) struct PairState {
    on: bool,
    /// The link's rounds so far ([`crate::model::core::pair::link::PairLink::begin`]).
    round: u64,
    /// Per expert-cache entry: the model layer it is a copy of (pair parallelism's copies).
    copy_of: Vec<Option<usize>>,
    /// Per entry: a copy whose route log holds a paired step's routing not learned yet.
    fresh: Vec<bool>,
    /// Per stage: the helper graphs it keeps for its pair's steps, by token count, most recent
    /// last. Each has its own compute buffer.
    helps: Vec<Vec<(usize, KeptGraph)>>,
}

impl PairState {
    /// `ran` (expert-cache entries a pass ran) and the copies of the layers it selects whose
    /// routing a paired step logged since they last learned.
    pub(super) fn widen<'a>(&'a self, ran: impl Fn(usize) -> bool + 'a) -> impl Fn(usize) -> bool {
        move |e| {
            ran(e)
                || (self.fresh.get(e).copied().unwrap_or(false)
                    && self.copy_of[e].is_some_and(&ran))
        }
    }

    /// After the cache learned [`Self::widen`]`(ran)`: those copies' routing is learned.
    pub(super) fn learned(&mut self, ran: impl Fn(usize) -> bool) {
        for (f, c) in self.fresh.iter_mut().zip(&self.copy_of) {
            if *f && c.is_some_and(&ran) {
                *f = false;
            }
        }
    }

    /// Drops the helper graphs (the stages' graphs are dropped).
    pub(super) fn forget_graphs(&mut self) {
        for h in &mut self.helps {
            h.clear();
        }
    }
}

impl Executor {
    /// Runs decode and verify steps of one micro-batch with pair parallelism from now on (the
    /// model's [`Model::set_pair_tp`] and its expert cache's copies set up first), or not.
    pub fn set_pair_tp(&mut self, on: bool) {
        self.forget_graphs();
        let n_stages = self.stages.len();
        let on = on
            && self.model.pair_tp().is_some()
            && self.model.experts().is_some_and(|r| r.ep_real().is_some())
            && n_stages >= 2;
        let copy_of: Vec<Option<usize>> = match self.model.experts().and_then(|r| r.ep_real()) {
            Some(real) if on => real
                .iter()
                .enumerate()
                .map(|(e, &il)| (il != e).then_some(il))
                .collect(),
            _ => Vec::new(),
        };
        self.pair = PairState {
            on,
            round: self.pair.round,
            fresh: vec![false; copy_of.len()],
            copy_of,
            helps: (0..n_stages).map(|_| Vec::new()).collect(),
        };
    }

    /// Whether a step of `n_micro` micro-batches of `n_tok` tokens in all pairs its stages.
    pub(super) fn pair_fits(&self, n_micro: usize, prompt: bool, n_tok: usize) -> bool {
        self.pair.on
            && n_micro == 1
            && !prompt
            && n_tok as i64 <= crate::model::core::pair::MAX_TOKENS
            && self.remote.is_none()
    }

    /// Launches the helper of stage `s`'s step of `n` tokens on its pair's GPU (asynchronously),
    /// before stage `s`'s own graph. Returns the pair's stage, or `None` when the stage pairs no
    /// layer of it.
    pub(super) fn pair_launch(&mut self, s: usize, n: usize) -> Result<Option<usize>, ModelError> {
        let p = crate::model::core::pair::partner(s);
        if p >= self.stages.len() {
            return Ok(None);
        }
        let layers = self.stages[s].layers.clone();
        if self.model.pair_layers(layers.clone(), n as i64).is_empty() {
            return Ok(None);
        }
        let helps = &mut self.pair.helps[p];
        match helps.iter().position(|(k, _)| *k == n) {
            Some(i) => {
                let g = helps.remove(i);
                helps.push(g);
            }
            None => {
                let old = (helps.len() >= HELPS_KEPT).then(|| helps.remove(0));
                let buft = self.stages[p].backend.buffer_type();
                let allocator = old
                    .and_then(|(_, g)| g.into_allocator())
                    .unwrap_or_else(|| ComputeAllocator::new(buft));
                let ctx = Context::new(GRAPH_SIZE, GRAPH_SIZE);
                let parts = {
                    let Some(graph) = self
                        .model
                        .pair_helper(&ctx, GRAPH_SIZE, p, layers, n as i64)
                    else {
                        return Ok(None);
                    };
                    ctx.alloc_graph(&graph, allocator)?;
                    KeptParts::new(&graph, &[])
                };
                // SAFETY: the graph references the weights, the expert cache and the link, all
                // owned by the model, which outlives the kept graphs (dropped first).
                let kept = unsafe { KeptGraph::new(ctx, parts) };
                self.pair.helps[p].push((n, kept));
            }
        }
        self.pair.round += 1;
        let link = &self.model.pair_tp().expect("pair parallelism set").link;
        link.begin(self.pair.round);
        let kept = &self.pair.helps[p].last().expect("kept above").1;
        self.stages[p].backend.compute_kept_async(kept)?;
        Ok(Some(p))
    }

    /// Waits for the helper [`Self::pair_launch`] started on stage `p` for stage `s`'s step of
    /// `n` tokens (once stage `s` is done), and marks its copies' routing to learn. An error
    /// when the exchange timed out (the step's results are not to be used).
    pub(super) fn pair_wait(
        &mut self,
        helped: Option<usize>,
        s: usize,
        n: usize,
    ) -> Result<(), ModelError> {
        let Some(p) = helped else {
            return Ok(());
        };
        self.stages[p].backend.synchronize();
        let failed = self.model.pair_tp().is_some_and(|t| t.link.failed());
        if failed {
            return Err(ModelError::Ggml(GgmlError::External(format!(
                "pair parallelism: the exchange between stages {s} and {p} timed out"
            ))));
        }
        let layers = self
            .model
            .pair_layers(self.stages[s].layers.clone(), n as i64);
        for (f, c) in self.pair.fresh.iter_mut().zip(&self.pair.copy_of) {
            if c.is_some_and(|il| layers.contains(&il)) {
                *f = true;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
