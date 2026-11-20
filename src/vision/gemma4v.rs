//! The Gemma 4 vision tower of a `gemma4v` mmproj (#1217): one image's preprocessed pixels to
//! the language model's input embeddings, one row per 3×3 block of patches.
//!
//! The graph is mtmd's, op for op, so that on the same backend it computes the same values:
//! 1. **Patches.** The pixels (`byte / 255`) scaled to `[-1, 1]`, one 16×16 convolution, no
//!    bias; patches row by row.
//! 2. **Positions.** Two learned tables, one row per column and one per row of the patch grid,
//!    both added. Attention rotates the first half of each head by the patch's column and the
//!    second half by its row (NEOX pairs, base 100).
//! 3. **Blocks.** RMS-normed ViT blocks: separate `q`, `k`, `v`; `q` and `k` RMS-normed per head
//!    with weights, `v` without; full (non-causal) attention at scale 1; the output RMS-normed
//!    before the residual; a GELU-gated MLP, its output RMS-normed before the residual.
//! 4. **Pooler.** Each 3×3 block of patches averaged, scaled by `sqrt(width)`, standardized
//!    (`(x - std_bias) * std_scale`), RMS-normed without a weight and projected to the language
//!    model's width.
//!
//! The clipped linears some Gemma 4 towers have (`*.input_min` and friends) are refused: the
//! 31B's have none.
//!
//! Ported from llama.cpp's `tools/mtmd/models/gemma4v.cpp` and `clip.cpp` (`build_vit`,
//! `build_attn` without flash attention, `build_norm`, `build_ffn`, the `GEMMA4V` hparams)
//! (MIT).

use crate::ggml::{ComputeAllocator, Context, ROPE_NEOX, Tensor, types};
use crate::gguf::GgufFile;

use super::qwen3vl::{Encoder, EncoderConfig, ImageEmbedding, weight};
use super::{Preprocessed, VisionError, VisionHparams};

/// Graph nodes per ViT block, with room.
const NODES_PER_LAYER: usize = 64;
const NODES_EXTRA: usize = 256;
/// The rotation's base (`hparams.rope_theta`).
pub(super) const ROPE_THETA: f32 = 100.0;

/// The tower's shape (`n_layer` read already), from the mmproj's keys and tensors.
pub(super) fn config(
    g: &GgufFile,
    hp: &VisionHparams,
    n_layer: usize,
) -> Result<EncoderConfig, VisionError> {
    let meta = |what: &str| VisionError::Metadata(what.to_string());
    let int = |k: &str| {
        g.get_u64(&format!("clip.vision.{k}"))
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .ok_or_else(|| meta(&format!("clip.vision.{k}")))
    };
    let eps = g
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let pos = g
        .tensor("v.position_embd.weight")
        .ok_or_else(|| meta("v.position_embd.weight"))?;
    if pos.shape.len() != 3 || pos.shape[2] != 2 {
        return Err(meta("v.position_embd.weight: not two tables"));
    }
    let n_out = g
        .tensor("mm.input_projection.weight")
        .and_then(|t| t.shape.get(1).copied())
        .ok_or_else(|| meta("mm.input_projection.weight"))? as usize;
    if g.tensors()
        .iter()
        .any(|t| t.name.ends_with(".input_min") || t.name.ends_with(".output_min"))
    {
        return Err(VisionError::Unsupported(
            "gemma4v with clipped linears".into(),
        ));
    }
    let cfg = EncoderConfig {
        n_embd: int("embedding_length")?,
        n_head: int("attention.head_count")?,
        n_layer,
        eps,
        patch_size: hp.patch_size as usize,
        merge: hp.spatial_merge_size as usize,
        pos_side: pos.shape[1] as usize,
        n_out,
    };
    if !cfg.n_embd.is_multiple_of(cfg.n_head) || !cfg.d_head().is_multiple_of(4) {
        return Err(meta("a head width divisible by 4"));
    }
    Ok(cfg)
}

/// Every tensor the graph reads.
pub(super) fn names(n_layer: usize) -> Vec<String> {
    let mut v: Vec<String> = [
        "v.patch_embd.weight",
        "v.position_embd.weight",
        "v.std_bias",
        "v.std_scale",
        "mm.input_projection.weight",
    ]
    .map(String::from)
    .to_vec();
    for il in 0..n_layer {
        for t in [
            "ln1",
            "ln2",
            "attn_post_norm",
            "ffn_post_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_out",
            "attn_q_norm",
            "attn_k_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            v.push(format!("v.blk.{il}.{t}.weight"));
        }
    }
    v
}

/// The patches' columns and rows, row by row (the tables' and the rotation's positions).
pub(super) fn patch_positions(px: usize, py: usize) -> (Vec<i32>, Vec<i32>) {
    let n = px * py;
    let xs = (0..n).map(|i| (i % px) as i32).collect();
    let ys = (0..n).map(|i| (i / px) as i32).collect();
    (xs, ys)
}

/// The tower's output for `p` (see the module docs).
pub(super) fn encode(enc: &mut Encoder, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
    let c = enc.cfg.clone();
    let (px, py) = (p.grid_w as usize, p.grid_h as usize);
    let k = c.merge;
    if px % k != 0 || py % k != 0 || px == 0 || py == 0 || px.max(py) > c.pos_side {
        return Err(VisionError::Dimensions);
    }
    let n_pos = (px * py) as i64;
    let (n_embd, d_head, n_head) = (c.n_embd as i64, c.d_head() as i64, c.n_head as i64);
    let size = NODES_EXTRA + NODES_PER_LAYER * c.n_layer;
    let ctx = Context::new(size, size);
    let mut graph = ctx.new_graph(size);
    let raw = ctx
        .new_tensor(types::F32, &[p.width as i64, p.height as i64, 3, 1])
        .named("inp_raw")
        .as_input();
    let pos_x = ctx
        .new_tensor(types::I32, &[n_pos])
        .named("pos_x")
        .as_input();
    let pos_y = ctx
        .new_tensor(types::I32, &[n_pos])
        .named("pos_y")
        .as_input();
    let weights = &enc.weights;
    let w = |name: &str| weight(weights, name);
    let ps = c.patch_size as i32;

    // Patches of the pixels in [-1, 1].
    let inp = ctx.scale_bias(raw, 2.0, -1.0);
    let inp = ctx.conv_2d(w("v.patch_embd.weight")?, inp, [ps, ps], [0, 0], [1, 1]);
    let inp = ctx.reshape_3d(inp, n_pos, n_embd, 1);
    let inp = ctx.cont(ctx.transpose(inp));

    // The column and row tables.
    let tbl = w("v.position_embd.weight")?;
    let rows = c.pos_side as i64;
    let nb1 = tbl.nb()[1];
    let tbl_x = ctx.view_2d(tbl, [n_embd, rows], nb1, 0);
    let tbl_y = ctx.view_2d(tbl, [n_embd, rows], nb1, rows as usize * nb1);
    let inp = ctx.add(inp, ctx.get_rows(tbl_x, pos_x));
    let mut x = ctx.add(inp, ctx.get_rows(tbl_y, pos_y));

    let norm = |t, name: &str| -> Result<Tensor<'_>, VisionError> {
        Ok(ctx.mul(ctx.rms_norm(t, c.eps), w(name)?))
    };
    let half = (d_head / 2) as i32;
    // Columns rotate the first half of each head, rows the second.
    let rope = |t| {
        let t = ctx.rope_ext(t, pos_x, None, half, ROPE_NEOX, 0, ROPE_THETA, 1.0);
        let t = ctx.rope_ext(t, pos_y, None, half, ROPE_NEOX, 0, ROPE_THETA, 1.0);
        ctx.rope_offset(t, half)
    };
    for il in 0..c.n_layer {
        let blk = |s: &str| format!("v.blk.{il}.{s}.weight");
        let cur = norm(x, &blk("ln1"))?;
        let head = |name: &str| -> Result<Tensor<'_>, VisionError> {
            Ok(ctx.reshape_4d(ctx.mul_mat(w(&blk(name))?, cur), [d_head, n_head, n_pos, 1]))
        };
        let q = rope(norm(head("attn_q")?, &blk("attn_q_norm"))?);
        let k = rope(norm(head("attn_k")?, &blk("attn_k_norm"))?);
        let v = ctx.rms_norm(head("attn_v")?, c.eps);
        graph.expand(q);
        graph.expand(k);
        graph.expand(v);
        let q = ctx.permute(q, [0, 2, 1, 3]);
        let k = ctx.permute(k, [0, 2, 1, 3]);
        let v = ctx.cont(ctx.permute(v, [1, 2, 0, 3]));
        let kq = ctx.soft_max_scaled(ctx.mul_mat(k, q), 1.0);
        let kqv = ctx.permute(ctx.mul_mat(v, kq), [0, 2, 1, 3]);
        let attn = ctx.cont_2d(kqv, n_embd, n_pos);
        let attn = ctx.mul_mat(w(&blk("attn_out"))?, attn);
        x = ctx.add(norm(attn, &blk("attn_post_norm"))?, x);
        let cur = norm(x, &blk("ln2"))?;
        let gate = ctx.mul_mat(w(&blk("ffn_gate"))?, cur);
        let up = ctx.mul_mat(w(&blk("ffn_up"))?, cur);
        let cur = ctx.mul_mat(w(&blk("ffn_down"))?, ctx.geglu_split(gate, up));
        x = ctx.add(x, norm(cur, &blk("ffn_post_norm"))?);
    }

    // The pooler: [n_embd, n] -> [px, py, n_embd] -> 3×3 averages -> [n_embd, n / 9].
    let (ox, oy) = ((px / k) as i64, (py / k) as i64);
    let t = ctx.cont_4d(ctx.transpose(x), [px as i64, py as i64, n_embd, 1]);
    let t = ctx.avg_pool_2d(t, k as i32);
    let t = ctx.reshape_3d(t, ox * oy, n_embd, 1);
    let t = ctx.cont(ctx.transpose(t));
    let t = ctx.scale(t, (n_embd as f32).sqrt());
    let t = ctx.mul(ctx.sub(t, w("v.std_bias")?), w("v.std_scale")?);
    let t = ctx.rms_norm(t, c.eps);
    let out = ctx.mul_mat(w("mm.input_projection.weight")?, t).as_output();
    graph.expand(out);

    let allocator = enc
        .alloc
        .take()
        .unwrap_or_else(|| ComputeAllocator::new(enc.backend.buffer_type()));
    ctx.alloc_graph(&graph, allocator)
        .map_err(|e| VisionError::Compute(e.to_string()))?;
    raw.set_f32(&p.pixels);
    let (xs, ys) = patch_positions(px, py);
    pos_x.set_i32(&xs);
    pos_y.set_i32(&ys);
    enc.backend
        .compute(&graph)
        .map_err(|e| VisionError::Compute(e.to_string()))?;
    let data = out.to_f32();
    enc.alloc = ctx.into_allocator();
    Ok(ImageEmbedding {
        nx: px / k,
        ny: py / k,
        n_embd: c.n_out,
        data,
    })
}

#[cfg(test)]
mod tests;
