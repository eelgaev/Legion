//! `--tp-moe-ep` (#798): expert-parallel routed experts under tensor parallelism over machines.
//!
//! Rank `r` of `R` holds experts `[r·E/R, (r+1)·E/R)` at full width
//! ([`crate::tp_net::shard::qwen4::Experts::Whole`]), or with `--tp-moe-ep-place` (#803) on
//! the layers its placement names, the `E/R` experts it gives the rank, in ascending order
//! ([`crate::tp_net::ep_place`]; such a layer has slot tables of its own, and the expert
//! matvec maps the router's ids through them); the router, the shared expert and
//! attention keep their split. Every rank routes all tokens, computes only the (token, expert)
//! pairs of its own experts, weighted by the router's probabilities, and the layer's existing
//! all-reduce adds the ranks' partial sums into the full MoE output.
//!
//! Four ways to leave the other experts out:
//! - Legion's expert matvec ([`expert_mv`]) takes the router's ids with [`TpEp::share`]: it
//!   skips the other experts' pairs and writes their rows 0. On CUDA, steps of up to
//!   [`expert_mv::max_tokens`] tokens take it (decode and MTP verify steps), as main's row split
//!   takes ggml's mmvq there: the fused op is slower for 1-3 tokens;
//! - the fused expert op ([`crate::model::moe_ffn`]) takes [`TpEp::table`]: expert `e`'s slot in
//!   the rank's stack, -1 for the others, whose pairs the op skips (no work, no sum). On CUDA
//!   the bigger steps of a cut layer take it;
//! - NVFP4 experts with per-expert scales on Legion's grouped W4A16 kernels take the slot
//!   table's ids ([`TpEp::remap_skips`], #1176): the grouped plan lists the -1 pairs after the
//!   experts' pairs, no tile computes them and the kernels write their rows 0;
//! - on the CPU, ggml's `mul_mat_id` path gets the ids remapped into the stack (the others to
//!   slot 0) and the others' router weights zeroed ([`TpEp::remap`]), so their products are
//!   computed but add nothing. Not on CUDA: a token then names slot 0 several times, and
//!   ggml-cuda's `mul_mat_id` grouping (`mmid.cu`) counts an expert once per token.

#[cfg(test)]
mod tests;

use crate::ggml::{BufferType, Context, Tensor, ffi, types};

use super::{ModelError, expert_mv};
use crate::tp_net::ep_place::Placement;

static SKIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether NVFP4 experts with per-expert scales on Legion's grouped kernels take the slot
/// table's ids ([`TpEp::remap_skips`]: the other ranks' pairs skipped) instead of
/// [`TpEp::remap`]'s (computed on slot 0 with a zero weight). The same bits; on by default,
/// `legion serve --no-tp-moe-ep-skip` turns it off for comparisons (#1176).
pub fn set_skip(on: bool) {
    SKIP.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// [`set_skip`]'s setting.
pub(crate) fn skip() -> bool {
    SKIP.load(std::sync::atomic::Ordering::Relaxed)
}

pub(super) static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a model of this process runs an expert-parallel share (`--tp-moe-ep`): the rows of
/// the experts other ranks hold are +0.0 with zero router weights, so a MoE tail may skip
/// reading rows of zero weights for the same bits (`super::node_fuse`'s fused tail, #1176).
pub(crate) fn active() -> bool {
    ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Names of the tables in [`TpEp`]'s context.
const SLOT: &str = "tp_ep.slot";
const IDS: &str = "tp_ep.ids";
const KEEP: &str = "tp_ep.keep";

/// A rank's expert-parallel share and the tables that map the router's choices onto it.
pub(crate) struct TpEp {
    /// The tables, in the experts' buffer type.
    ctx: Context,
    n_expert: usize,
    /// The first expert this rank holds.
    lo: usize,
    /// Experts this rank holds.
    n_local: usize,
    /// The layers placed by `--tp-moe-ep-place` (#803), which have tables of their own; the
    /// others take the contiguous split's.
    placed: std::collections::BTreeSet<usize>,
}

/// The tables for experts `[lo, lo + n_local)` of `n_expert`: the fused op's slot table (-1 for
/// experts held elsewhere), the remapped ids (0 for those) and the router weight masks (0.0 for
/// those, 1.0 for the rank's own).
pub(crate) fn tables(n_expert: usize, lo: usize, n_local: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let own = |e: usize| (lo..lo + n_local).contains(&e);
    let slot: Vec<i32> = (0..n_expert)
        .map(|e| if own(e) { (e - lo) as i32 } else { -1 })
        .collect();
    tables_of(slot)
}

/// [`tables`] from a slot table (expert `e`'s slot, -1 for experts held elsewhere): a placed
/// layer's (`--tp-moe-ep-place`, #803).
pub(crate) fn tables_of(slot: Vec<i32>) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let ids = slot.iter().map(|&s| s.max(0)).collect();
    let keep = slot
        .iter()
        .map(|&s| if s >= 0 { 1.0 } else { 0.0 })
        .collect();
    (slot, ids, keep)
}

/// A layer's tables in [`TpEp`]'s context: the contiguous split's (`None`) or layer `il`'s.
fn name(what: &str, il: Option<usize>) -> String {
    match il {
        Some(il) => format!("{what}.{il}"),
        None => what.to_string(),
    }
}

fn le_bytes<T: Copy>(v: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

impl TpEp {
    /// Rank `rank` of `ranks`' share of `n_expert` routed experts. `first` is a layer's
    /// `ffn_up_exps` as loaded: it must hold exactly the rank's experts, and the tables go in
    /// its buffer type. With `place` (`--tp-moe-ep-place`, #803), the layers it names hold
    /// the experts it gives this rank.
    pub(crate) fn new(
        first: Tensor<'_>,
        n_expert: usize,
        rank: usize,
        ranks: usize,
        place: Option<&Placement>,
    ) -> Result<Self, ModelError> {
        if let Some(p) = place {
            p.fits(n_expert, ranks)
                .map_err(|e| ModelError::Layout(format!("--tp-moe-ep-place: {e}")))?;
        }
        let (lo, n_local) = crate::tp_net::shard::qwen4::expert_range(n_expert as u64, ranks, rank)
            .map_err(|e| ModelError::Layout(format!("--tp-moe-ep: {e}")))?;
        let (lo, n_local) = (lo as usize, n_local as usize);
        if first.ne()[2] as usize != n_local {
            return Err(ModelError::Layout(format!(
                "--tp-moe-ep: `{}` holds {} experts, rank {rank} of {ranks} should hold {n_local}",
                first.name(),
                first.ne()[2]
            )));
        }
        let e = n_expert as i64;
        let placed: std::collections::BTreeSet<usize> =
            place.map_or_else(Default::default, |p| p.layers.keys().copied().collect());
        let ctx = Context::new(3 * (1 + placed.len()), 0);
        for il in std::iter::once(None).chain(placed.iter().map(|&il| Some(il))) {
            ctx.new_tensor(types::I32, &[e]).named(&name(SLOT, il));
            ctx.new_tensor(types::I32, &[1, e]).named(&name(IDS, il));
            ctx.new_tensor(types::F32, &[1, e]).named(&name(KEEP, il));
        }
        // SAFETY: an allocated weight's buffer; reads its type.
        let buft = unsafe {
            BufferType(
                std::ptr::NonNull::new(ffi::ggml_backend_buffer_get_type((*first.raw()).buffer))
                    .ok_or_else(|| ModelError::Tensor(format!("{}: no buffer", first.name())))?,
            )
        };
        ctx.alloc_tensors(buft)?;
        let get = |name: &str| ctx.tensor(name).expect("created above");
        let fill = |il: Option<usize>, (slot, ids, keep): (Vec<i32>, Vec<i32>, Vec<f32>)| {
            get(&name(SLOT, il)).set_bytes(&le_bytes(&slot, i32::to_le_bytes));
            get(&name(IDS, il)).set_bytes(&le_bytes(&ids, i32::to_le_bytes));
            get(&name(KEEP, il)).set_bytes(&le_bytes(&keep, f32::to_le_bytes));
        };
        fill(None, tables(n_expert, lo, n_local));
        if let Some(p) = place {
            for &il in &placed {
                fill(Some(il), tables_of(p.slot_table(il, rank)));
            }
        }
        Ok(Self {
            ctx,
            n_expert,
            lo,
            n_local,
            placed,
        })
    }

    /// Layer `il`'s tables' names' suffix: its own if placed, else the contiguous split's.
    fn at(&self, il: usize) -> Option<usize> {
        self.placed.contains(&il).then_some(il)
    }

    /// This rank's experts of layer `il` for Legion's expert matvec ([`expert_mv::Share`]):
    /// steps of a few tokens take the router's ids as they are there, with the others' pairs
    /// skipped. A placed layer's ids go through its slot table.
    pub(crate) fn share(&self, il: usize) -> expert_mv::Share<'_> {
        match self.at(il) {
            Some(_) => expert_mv::Share {
                first: 0,
                map: Some(self.table(il)),
            },
            None => expert_mv::Share {
                first: self.lo,
                map: None,
            },
        }
    }

    /// Whether a layer whose `ffn_up_exps` is `up` holds this rank's share (an MTP head cut
    /// the same way does; one left whole would not).
    pub(crate) fn cuts(&self, up: Tensor<'_>) -> bool {
        self.n_local < self.n_expert && up.ne()[2] as usize == self.n_local
    }

    /// Layer `il`'s fused expert op table: I32 `[n_expert]`, expert `e`'s slot or -1.
    pub(crate) fn table(&self, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&name(SLOT, self.at(il)))
            .expect("created in new")
    }

    /// Layer `il`'s router choices `selected` (`[n_used, n]`) remapped into the rank's stack
    /// and the router weights `w` (`[n_used, n]`) with the other ranks' experts zeroed.
    pub(crate) fn remap<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        selected: Tensor<'a>,
        w: Tensor<'a>,
        n: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let used = selected.ne()[0];
        let flat = ctx.reshape_1d(ctx.cont(selected), used * n);
        let at = self.at(il);
        let table = |what: &str| self.ctx.tensor(&name(what, at)).expect("created in new");
        let ids = ctx.reshape_2d(ctx.get_rows(table(IDS), flat), used, n);
        let keep = ctx.reshape_2d(ctx.get_rows(table(KEEP), flat), used, n);
        let w = if w.is_contiguous() { w } else { ctx.cont(w) };
        (ids, ctx.mul(w, keep))
    }

    /// [`Self::remap`] with the other ranks' experts' ids -1 instead of 0 (the slot table's,
    /// [`Self::table`]): for Legion's grouped NVFP4 kernels, which skip such pairs and write
    /// their rows 0 ([`crate::model::nvfp4_mm::skips`], #1176). The router weights are masked
    /// as there.
    pub(crate) fn remap_skips<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        selected: Tensor<'a>,
        w: Tensor<'a>,
        n: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let used = selected.ne()[0];
        let flat = ctx.reshape_1d(ctx.cont(selected), used * n);
        let at = self.at(il);
        let table = |what: &str| self.ctx.tensor(&name(what, at)).expect("created in new");
        let slot = ctx.reshape_2d(table(SLOT), 1, self.n_expert as i64);
        let ids = ctx.reshape_2d(ctx.get_rows(slot, flat), used, n);
        let keep = ctx.reshape_2d(ctx.get_rows(table(KEEP), flat), used, n);
        let w = if w.is_contiguous() { w } else { ctx.cont(w) };
        (ids, ctx.mul(w, keep))
    }
}
