//! Causal attention with sliding-window layers, for architectures whose layers alternate
//! sliding-window and global attention (Gemma 4, Muse Glimmer): the gather, cell-gather,
//! sequence-slot and paged paths, each windowed on the sliding layers and reading a ring
//! where the layer's cache is one. The attention scale is the architecture's.

use super::{AttnInputs, Model, ModelError, Pass, concat_cols, gather_mask, paged};
use crate::ggml::Tensor;

impl Model {
    /// Causal attention of every group's queries `q` (`[d, n_head, n]`) over the cache layer
    /// `il` reads, softmax scale `scale`, through the sliding-window mask (window
    /// [`crate::model::core::Arch::swa_window`]) when `swa`; `[d × n_head, n]`. The gather path
    /// is llama.cpp's `build_attn_mha` without flash attention.
    pub(crate) fn window_attend<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        swa: bool,
        n_head: i64,
        q: Tensor<'a>,
        scale: f32,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let (d, _) = self.hp.kv_dims(il);
        let mut outs = Vec::new();
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let qnb = q.nb();
            let q = ctx.view_3d(
                q,
                [d, n_head, nt * ns],
                [qnb[1], qnb[2]],
                off as usize * qnb[2],
            );
            let q = ctx.reshape_4d(q, [d, n_head, nt, ns]);
            // The mask, the keys' entries, and the keys per sequence.
            let (mask, blocks, n_kv) = match pass.attn[gi] {
                // A ring layer gathers its positions through the rings, with its own mask:
                // per cell its window only (`swa_span`).
                AttnInputs::Gather {
                    mask,
                    blocks,
                    ring,
                    swa: swa_mask,
                } => match (ring, swa_mask) {
                    (Some(r), Some(m)) if pass.kv.is_ring(il) => (m, r, m.ne()[0]),
                    _ => (gather_mask(ctx, mask, ns, swa), blocks, sh.n_kv as i64),
                },
                AttnInputs::Paged {
                    table,
                    n_past,
                    scratch,
                    ring,
                } => {
                    // The blocks in place: the cache this layer reads, windowed; through the
                    // sequences' rings if it is one.
                    let (k, v) = pass.kv.pools(il);
                    // Exclusive queries (an MTP assistant) run as the position before theirs
                    // (`Inputs::fill`): a window one shorter ends where theirs does.
                    let window = match (swa, pass.exclusive) {
                        (false, _) => 0,
                        (true, ex) => self.hp.arch.swa_window() - usize::from(ex),
                    };
                    let (table, ring) = match ring {
                        Some(r) if pass.kv.is_ring(il) => (r, sh.ring),
                        _ => (table, 0),
                    };
                    // TurboQuant caches (#552): their own op. A compact cache (Gemma 4, #551):
                    // granules `d + r` wide (never TurboQuant, `turboquant::check`).
                    let o = match (pass.kv.tq(il), pass.kv.compact(il)) {
                        (Some(tq), _) => paged::tq::attention(
                            ctx, q, k, v, table, n_past, tq, scale, window, ring,
                        ),
                        (None, 0) => paged::attention_tiered(
                            ctx,
                            q,
                            k,
                            v,
                            table,
                            n_past,
                            scratch,
                            scale,
                            window,
                            ring,
                            pass.kv.paged_tier(il),
                        ),
                        (None, r) => paged::attention_granules(
                            ctx,
                            q,
                            k,
                            v,
                            table,
                            n_past,
                            scratch,
                            scale,
                            window,
                            pass.kv.paged_tier(il),
                            (d + r) as usize,
                        ),
                    };
                    outs.push(ctx.reshape_2d(o, d * n_head, nt * ns));
                    continue;
                }
            };
            if !sh.kv_slots.is_empty() && !pass.kv.is_ring(il) {
                // Sequence slots (`KvPool::slot_span`): each sequence's span read in place,
                // wherever it lives, by its own flash attention (split by heads as below).
                let mask = ctx.cast(mask, crate::ggml::types::F16);
                let (qnb, mnb) = (q.nb(), mask.nb());
                for (si, &slot) in sh.kv_slots.iter().enumerate() {
                    // The MTP head's passes (exclusive queries) through the alias buffer.
                    let (k, v) = pass
                        .kv
                        .slot_span(ctx, il, slot as usize, n_kv, pass.exclusive);
                    // A prompt step whose span reaches the host tier reads a VRAM copy (#1153).
                    let (k, v) = if slot_stage_copies(self.slot_stage, sh.n_tok, n_kv as usize) {
                        (ctx.cont(k), ctx.cont(v))
                    } else {
                        (k, v)
                    };
                    let qs =
                        ctx.view_4d(q, [d, n_head, nt, 1], [qnb[1], qnb[2], qnb[3]], si * qnb[3]);
                    let m = ctx.view_4d(
                        mask,
                        [n_kv, nt, 1, 1],
                        [mnb[1], mnb[2], mnb[3]],
                        si * mnb[3],
                    );
                    let o = ctx.flash_attn_ext(
                        ctx.permute(qs, [0, 2, 1, 3]),
                        ctx.permute(k, [0, 2, 1, 3]),
                        ctx.permute(v, [0, 2, 1, 3]),
                        m,
                        scale,
                    );
                    outs.push(ctx.reshape_2d(o, d * n_head, nt));
                }
                continue;
            }
            if blocks.ne()[0] == n_kv * ns {
                // Cells (tensor parallelism, `Model::set_cell_gather`): flash attention, which
                // a meta device splits by heads with GQA (its mul_mat does not).
                let (k, v) = pass.kv.read_cells(ctx, il, blocks, n_kv, ns);
                let o = ctx.flash_attn_ext(
                    ctx.permute(q, [0, 2, 1, 3]),
                    ctx.permute(k, [0, 2, 1, 3]),
                    ctx.permute(v, [0, 2, 1, 3]),
                    ctx.cast(mask, crate::ggml::types::F16),
                    scale,
                );
                outs.push(ctx.reshape_2d(o, d * n_head, nt * ns));
                continue;
            }
            let (k, vt) = pass.kv.read(ctx, il, blocks, n_kv, ns);
            let q = ctx.permute(q, [0, 2, 1, 3]);
            let k = ctx.permute(k, [0, 2, 1, 3]);
            let kq = ctx.mul_mat(k, q);
            ctx.prec_set_acc(kq, crate::ggml::ffi::ggml_prec_GGML_PREC_F32);
            let kq = ctx.soft_max_ext(kq, mask, scale, 0.0);
            let kqv = ctx.mul_mat(vt, kq);
            outs.push(ctx.cont_2d(ctx.permute(kqv, [0, 2, 1, 3]), d * n_head, nt * ns));
        }
        Ok(concat_cols(ctx, outs))
    }
}

/// Whether sequence-slot attention copies a sequence's span of `n_kv` keys into the compute
/// buffers before flash attention in a step of `n_tok` query rows per sequence, under
/// [`Model::set_slot_stage`]'s `(rows, from, upto)`: a prompt step (at least `rows` rows) whose
/// span reaches past the slot's first `from` tokens, the ones in VRAM, and is at most `upto`
/// long. Flash attention reads the keys once per tile of query rows, so a span in the host tier
/// would cross the link that many times; the copy crosses it once.
pub(crate) fn slot_stage_copies(
    (rows, from, upto): (usize, usize, usize),
    n_tok: usize,
    n_kv: usize,
) -> bool {
    rows > 0 && n_tok >= rows && n_kv > from && n_kv <= upto
}

#[cfg(test)]
mod tests;
