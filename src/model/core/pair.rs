//! Pair parallelism for the routed experts of decode and verify steps (`--pair-tp`, #668).
//!
//! The layers stay split over the stages as before (stage `s` on CUDA device `s`), but while a
//! stage runs a step of a few tokens, the other GPU of its NVLink pair (`s ^ 1`, idle in such a
//! step otherwise) computes half of each MoE layer's routed experts:
//!
//! - the stage's graph (the *owner*) routes the layer's tokens, hands the rows, the choices and
//!   the weights to its pair ([`link::send_task`]), computes the experts of its own half
//!   ([`half`]) and the shared expert, then adds the pair's partial sum to its own
//!   ([`link::recv_add`]);
//! - the pair runs a *helper* graph ([`Model::pair_helper`]) beside it: per layer, it takes the
//!   task, computes the experts of the other half and sends their weighted sum back.
//!
//! Each GPU's expert cache serves its half of the experts: the owner's entry of a layer its own
//! half, a copy of the layer on the pair the other ([`ep_layers`]). A pair's GPUs share a socket,
//! so both read their misses from that socket's memory, and the exchange goes over NVLink.
//!
//! The routed sum is two partial sums (each in choice order) added: other bits than one GPU's
//! sum of all the choices, so greedy text may differ at near-ties. Graphs without the flag are
//! unchanged.

pub mod link;
pub mod sum;

use std::ops::Range;
use std::sync::Arc;

use super::{ModelError, Pass, expert_mv, fused};
use crate::ggml::{Context, Graph, Tensor, types};
use crate::model::core::Model;
use crate::model::glm5next::par::exchange::{ep_mask, ep_sum};

/// The most tokens of a paired step: the expert matvec computes up to this many exactly.
pub const MAX_TOKENS: i64 = expert_mv::EXACT_MAX_TOKENS;

/// The pair parallelism of a loaded model ([`Model::set_pair_tp`]).
pub struct PairTp {
    /// Per model layer: the CUDA device of its stage when the layer is paired.
    pub owner: Vec<Option<usize>>,
    pub n_expert: u32,
    pub link: Arc<link::PairLink>,
}

/// The other GPU of CUDA device `cuda`'s NVLink pair.
pub fn partner(cuda: usize) -> usize {
    cuda ^ 1
}

/// The experts CUDA device `cuda` computes of a paired layer of `n_expert`: the first half on
/// an even device, the rest on an odd one.
pub fn half(cuda: usize, n_expert: u32) -> Range<u32> {
    let mid = n_expert / 2;
    if cuda.is_multiple_of(2) {
        0..mid
    } else {
        mid..n_expert
    }
}

/// Which model layers of `n_layer` a pair parallelism over stages on CUDA devices
/// `stage_of(il)` pairs: all but the last, whose rows a step crops to the logits rows before
/// its MoE (the helper would not know them).
pub fn owners(n_layer: usize, stage_of: impl Fn(usize) -> usize) -> Vec<Option<usize>> {
    (0..n_layer)
        .map(|il| (il + 1 < n_layer).then(|| stage_of(il)))
        .collect()
}

/// The entries of the expert cache's layout ([`crate::residency::EpLayers`]) for pair
/// parallelism over `owner` ([`owners`]): the cache's `n_entries` layers first (a paired
/// layer's serving its owner's half, the others and the MTP head all experts), then a copy of
/// each paired layer on its owner's pair, serving the other half. Also each entry's CUDA device
/// (`stage_of`, as for a cache without copies).
pub fn ep_layers(
    owner: &[Option<usize>],
    n_entries: usize,
    n_expert: u32,
    stage_of: impl Fn(usize) -> usize,
) -> (crate::residency::EpLayers, Vec<usize>) {
    let mut real: Vec<usize> = (0..n_entries).collect();
    let mut owned: Vec<Option<Range<u32>>> = (0..n_entries)
        .map(|il| owner.get(il).copied().flatten().map(|c| half(c, n_expert)))
        .collect();
    let mut cuda: Vec<usize> = (0..n_entries).map(&stage_of).collect();
    for (il, c) in owner.iter().enumerate() {
        if let Some(c) = *c {
            real.push(il);
            owned.push(Some(half(partner(c), n_expert)));
            cuda.push(partner(c));
        }
    }
    (crate::residency::EpLayers { real, owned }, cuda)
}

/// A paired layer's expert tensors on its owner and on its pair.
struct Sides<'a> {
    owner: usize,
    own: crate::residency::experts::TieredLayer<'a>,
    copy: crate::residency::experts::TieredLayer<'a>,
}

impl Model {
    /// Sets pair parallelism up (graphs built from now on with paired steps,
    /// [`super::GroupShape::pair`]).
    pub fn set_pair_tp(&mut self, pair: Option<PairTp>) {
        self.pair = pair;
    }

    /// The pair parallelism set up, if any.
    pub fn pair_tp(&self) -> Option<&PairTp> {
        self.pair.as_ref()
    }

    /// Layer `il`'s expert tensors on both GPUs when a step of `n` tokens pairs it: the owner's
    /// graph and the helper's decide alike.
    fn pair_sides(&self, il: usize, n: i64) -> Option<Sides<'_>> {
        let p = self.pair.as_ref()?;
        let owner = (*p.owner.get(il)?)?;
        if n > MAX_TOKENS.min(p.link.max_tokens()) || self.hp.n_expert > fused::ROUTER_MAX_EXPERTS {
            return None;
        }
        let own = self.expert_tier_on(il, owner)?;
        let copy = self.expert_tier_on(il, partner(owner))?;
        if own.gathers(n) || copy.gathers(n) {
            return None;
        }
        // The expert matvec takes the stacks (the same types and shapes on both GPUs).
        let hp = &self.hp;
        let scratch = Context::new(3, 0);
        let (gate, up, down) = (own.weight(1), own.weight(0), own.weight(2));
        let x = scratch.new_tensor(types::F32, &[hp.n_embd, 1, n]);
        let ids = scratch.new_tensor(types::I32, &[hp.n_expert_used, n]);
        let act = scratch.new_tensor(types::F32, &[gate.ne()[1], hp.n_expert_used, n]);
        let fits = [gate, up, down]
            .iter()
            .all(|w| !expert_mv::maya_type(w.ty()))
            && expert_mv::glu_fits_narrow_up_to(gate, up, x, ids, MAX_TOKENS)
            && expert_mv::fits_glm_up_to(down, act, ids, MAX_TOKENS);
        fits.then_some(Sides { owner, own, copy })
    }

    /// The layers of `layers` a paired step of `n` tokens pairs (whose copies its helper runs).
    pub(crate) fn pair_layers(&self, layers: Range<usize>, n: i64) -> Vec<usize> {
        layers
            .filter(|&il| self.pair_sides(il, n).is_some())
            .collect()
    }

    /// Layer `il`'s MoE of `h` (`[n_embd, n]`) in a paired step ([`super::GroupShape::pair`]):
    /// the owner's side (see the module). `None` where the layer is not paired: the caller
    /// builds [`Model::moe`] then.
    pub(crate) fn pair_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
        n: i64,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let hp = &self.hp;
        if pass.tap.on || !pass.shape.first().is_some_and(|g| g.pair) {
            return Ok(None);
        }
        let (Some(p), Some(sides)) = (self.pair.as_ref(), self.pair_sides(il, n)) else {
            return Ok(None);
        };
        let ctx = pass.ctx;
        let me = sides.owner;
        let t = sides.own;
        let used = hp.n_expert_used;
        // The router as `Model::moe` builds it (no taps in a paired step).
        let gate_w = self.lw(il, "ffn_gate_inp_shexp.weight")?;
        let gate_op = fused::shexp_gate_on(gate_w);
        let users = [
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        ];
        let hq = self.share_q8(ctx, false, il, &users, h);
        let (logits, gate_logit) = match gate_op
            .then(|| {
                let pair = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight");
                self.lmm_joined(ctx, il, pair.0, pair.1, h)
            })
            .flatten()
        {
            Some((l, g)) => (l, Some(g)),
            None => (self.lmm_q(ctx, il, "ffn_gate_inp.weight", h, hq)?, None),
        };
        let (ids, w) = fused::router(ctx, logits, used);
        let ids = if ids.is_contiguous() {
            ids
        } else {
            ctx.cont(ids)
        };
        let x = if h.is_contiguous() { h } else { ctx.cont(h) };
        // The task first, so the pair starts while this GPU computes its half.
        let send = link::send_task(ctx, &p.link, me, partner(me), x, ids, w);
        let mine = half(me, p.n_expert);
        let mapped = t.remap(ctx, &pass.graph, ids, n);
        let masked = ep_mask(ctx, mapped, ids, mine.clone());
        let x3 = ctx.reshape_3d(x, hp.n_embd, 1, n);
        let slots = t.n_slots();
        let act = expert_mv::glu_tiered(ctx, t.weight(1), t.weight(0), x3, masked, slots);
        let y = expert_mv::mul_mat_id_tiered(ctx, t.weight(2), act, masked, slots);
        let own = ep_sum(ctx, y, w, ids, mine, None);
        let (sh, g) = self.shexp(ctx, il, h, hq, gate_logit)?;
        // The wait last: the shared expert computes while the pair still works.
        let out = link::recv_add(ctx, &p.link, me, &[send], own, &[sh, g]);
        if gate_op && fused::shexp_gate_fits(out, sh, g, gate_w) {
            return Ok(Some(fused::shexp_gate(ctx, out, sh, g)));
        }
        let g = if g.is_contiguous() { g } else { ctx.cont(g) };
        let sh = ctx.mul(sh, ctx.sigmoid(g));
        Ok(Some(ctx.add(out, sh)))
    }

    /// The helper's graph on CUDA device `me` for a paired step of `n` tokens of the stage over
    /// `layers` on its pair: per paired layer, the task in, the experts of `me`'s half, their
    /// weighted sum out. `None` when no layer of them is paired.
    pub(crate) fn pair_helper<'a>(
        &'a self,
        ctx: &'a Context,
        graph_size: usize,
        me: usize,
        layers: Range<usize>,
        n: i64,
    ) -> Option<Graph<'a>> {
        let p = self.pair.as_ref()?;
        let graph = std::cell::RefCell::new(ctx.new_graph(graph_size));
        let mut prev: Option<Tensor<'a>> = None;
        let mine = half(me, p.n_expert);
        for il in layers {
            let Some(sides) = self.pair_sides(il, n) else {
                continue;
            };
            debug_assert_eq!(partner(sides.owner), me);
            let t = sides.copy;
            let ids = link::recv_task(ctx, &p.link, me, n, prev.as_slice());
            let x = ctx.reshape_3d(p.link.task_x(ctx, me, n), self.hp.n_embd, 1, n);
            let w = p.link.task_w(ctx, me, n);
            let mapped = t.remap(ctx, &graph, ids, n);
            let masked = ep_mask(ctx, mapped, ids, mine.clone());
            let slots = t.n_slots();
            let act = expert_mv::glu_tiered(ctx, t.weight(1), t.weight(0), x, masked, slots);
            let y = expert_mv::mul_mat_id_tiered(ctx, t.weight(2), act, masked, slots);
            let part = ep_sum(ctx, y, w, ids, mine.clone(), None);
            let sent = link::send_part(ctx, &p.link, me, sides.owner, part);
            graph.borrow_mut().expand(sent);
            prev = Some(sent);
        }
        prev.map(|_| graph.into_inner())
    }
}

#[cfg(test)]
mod tests;
