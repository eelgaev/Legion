//! The expert matvec over experts staged a layer ahead (`--glm-prefetch`, #747).
//!
//! GLM's parallel decode copies the next layer's guessed uncached experts into a VRAM staging
//! ring on a side stream while a layer computes (src/model/glm5next/par/prefetch.rs). The next
//! layer's SwiGLU and down ops then read each pair whose expert was staged from its slot, as
//! the #253 split's staged launch reads them ([`super::overlap`]): an i32 slot per pair (`-1`:
//! read in place), the slot's bytes at the slots' start plus `slot * slot_bytes`, the matrix at
//! its offset in the slot. The same rows, the same code: the same bits as reading them in
//! place.

use super::{EXPERT_MV, glu_node, q8_1_scratch};
use crate::ggml::{Context, Tensor, types};

#[cfg(test)]
mod tests;

/// [`super::Operands`]' `split` of a SwiGLU op reading staged experts (its sixth source the
/// ring).
pub(super) const GLU: i32 = 3;
/// ... and of a down op (its fifth source the ring, no `w2`).
pub(super) const DOWN: i32 = 4;

/// The params past the role: the slot table's offset in the ring, ...
pub(super) const P_TABLE: usize = 2;
/// ... a slot's bytes, ...
pub(super) const P_SLOT: usize = 4;
/// ... the slots' offset from the table, the matrix's offset in a slot, `w2`'s.
pub(super) const P_HDR: usize = 6;
pub(super) const P_W: usize = 7;
pub(super) const P_W2: usize = 8;

/// Where a layer's staged experts are.
#[derive(Clone, Copy)]
pub struct Ahead<'a> {
    /// The device's staging ring (`I8`).
    pub ring: Tensor<'a>,
    /// The node that wrote the slot table (a source for the order only).
    pub after: Tensor<'a>,
    /// Bytes from the ring's start to the slot table (one i32 per pair of the step), and to
    /// the slots.
    pub table: usize,
    pub slots: usize,
    /// A slot's bytes, and where its gate, up and down rows start in it.
    pub slot: usize,
    pub off: [usize; 3],
}

impl Ahead<'_> {
    fn params(&self, vram_slots: usize, role: i32, limit: f32, w: usize, w2: usize) -> [i32; 9] {
        let i = |v: usize| i32::try_from(v).expect("a staging ring under 2 GiB");
        [
            vram_slots as i32,
            role,
            i(self.table),
            0,
            i(self.slot),
            limit.to_bits() as i32,
            i(self.slots - self.table),
            i(w),
            i(w2),
        ]
    }
}

/// [`super::glu_tiered_clamped`] reading the pairs `a` staged from their slots.
pub fn glu_clamped<'a>(
    ctx: &'a Context,
    [gate, up]: [Tensor<'a>; 2],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    (vram_slots, limit): (usize, f32),
    a: &Ahead<'a>,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, gate, x, ids);
    let params = a.params(vram_slots, GLU, limit, a.off[0], a.off[1]);
    glu_node(ctx, [gate, up], x, ids, xq, &[a.ring, a.after], &params)
}

/// [`super::mul_mat_id_tiered`] of the down projection `w`, reading the pairs `a` staged from
/// their slots.
pub fn down<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    a: &Ahead<'a>,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, w, x, ids);
    let (rows, n_used, n_tok) = (w.ne()[1], ids.ne()[0], ids.ne()[1]);
    ctx.external_with(
        &EXPERT_MV,
        types::F32,
        [rows, n_used, n_tok, 1],
        &[w, x, ids, xq, a.ring, a.after],
        &a.params(vram_slots, DOWN, 0.0, a.off[2], 0),
    )
}
