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
//!    (`(x - std_bias) * std_scale`) when the tower has `v.std_bias` and `v.std_scale` (the 31B's;
//!    the E2B/E4B-sized towers, Nemotron 3.5 Bolt's among them, have none), RMS-normed without a
//!    weight and projected to the language model's width.
//!
//! **Clipped linears** (#1220). The smaller towers' block matrices carry four scalars each,
//! `<matrix>.input_min`, `.input_max`, `.output_min` and `.output_max` (HF's
//! `Gemma4ClippableLinear`): the input is clamped to the first two, multiplied, and the product
//! clamped to the last two; a missing scalar does not clamp that side (mtmd's `±FLT_MAX`). A
//! matrix without any (all of the 31B's) is a plain product, so that tower's graph is unchanged.
//! mtmd's input clamp is in place, on the normed rows `q`, `k` and `v` (or `gate` and `up`)
//! share, so towers whose matrices of one input differ in their input bounds are refused (none
//! do: the checkpoints share them), and clips only on the blocks' seven matrices are taken.
//!
//! Ported from llama.cpp's `tools/mtmd/models/gemma4v.cpp` and `clip.cpp` (`build_vit`,
//! `build_attn` without flash attention, `build_norm`, `build_ffn`, the `GEMMA4V` hparams)
//! (MIT).

use crate::ggml::{ComputeAllocator, Context, ROPE_NEOX, Tensor, types};
use crate::gguf::GgufFile;
use crate::weights::Weights;

use super::qwen3vl::{Encoder, EncoderConfig, ImageEmbedding, weight};
use super::{Preprocessed, VisionError, VisionHparams};

/// Graph nodes per ViT block, with room.
const NODES_PER_LAYER: usize = 96;
const NODES_EXTRA: usize = 256;
/// The rotation's base (`hparams.rope_theta`).
pub(super) const ROPE_THETA: f32 = 100.0;
/// A clipped linear's scalars, after its matrix's name without `.weight`.
pub(super) const CLIP_SCALARS: [&str; 4] =
    [".input_min", ".input_max", ".output_min", ".output_max"];
/// The block matrices that may be clipped.
pub(super) const CLIPPED: [&str; 7] = [
    "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down",
];

/// A clipped linear's bounds: the input clamped to `inp`, the product to `out` (`±f32::MAX`
/// where a scalar is missing, as mtmd's defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Clip {
    pub inp: (f32, f32),
    pub out: (f32, f32),
}

/// The bounds of block `il`'s matrix `m` (`attn_q` ...), `None` when it has no scalars (a
/// plain product).
pub(super) fn clip(weights: &Weights, il: usize, m: &str) -> Result<Option<Clip>, VisionError> {
    let mut v = [-f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
    let mut any = false;
    for (slot, s) in v.iter_mut().zip(CLIP_SCALARS) {
        let name = format!("v.blk.{il}.{m}{s}");
        let Some(t) = weights.get(&name) else {
            continue;
        };
        let x = if t.ty() == types::F32 {
            t.to_f32()
        } else if t.ty() == types::F16 {
            t.bytes()
                .chunks_exact(2)
                .map(|h| crate::cpu::gpu_exact::f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
                .collect()
        } else {
            return Err(VisionError::Metadata(format!("`{name}`: not F32 or F16")));
        };
        let [x] = x[..] else {
            return Err(VisionError::Metadata(format!("`{name}`: not one value")));
        };
        if x.is_nan() {
            return Err(VisionError::Metadata(format!("`{name}`: NaN")));
        }
        *slot = x;
        any = true;
    }
    Ok(any.then_some(Clip {
        inp: (v[0], v[1]),
        out: (v[2], v[3]),
    }))
}

/// The bounds of a side no scalar clamps (mtmd's defaults).
pub(super) const UNCLAMPED: (f32, f32) = (-f32::MAX, f32::MAX);

/// Checks a loaded tower: `v.std_bias` and `v.std_scale` both or neither, every clip readable,
/// and the matrices of one input (`q`, `k`, `v`; `gate`, `up`) clamping it the same (a matrix
/// without clips, or with output bounds only, leaves it [`UNCLAMPED`]).
pub(super) fn check(weights: &Weights, n_layer: usize) -> Result<(), VisionError> {
    if weights.get("v.std_bias").is_some() != weights.get("v.std_scale").is_some() {
        return Err(VisionError::Metadata(
            "gemma4v: one of `v.std_bias` and `v.std_scale` without the other".into(),
        ));
    }
    for il in 0..n_layer {
        for group in [&["attn_q", "attn_k", "attn_v"][..], &["ffn_gate", "ffn_up"]] {
            let inp: Vec<_> = group
                .iter()
                .map(|m| Ok(clip(weights, il, m)?.map_or(UNCLAMPED, |c| c.inp)))
                .collect::<Result<_, VisionError>>()?;
            if inp.windows(2).any(|p| p[0] != p[1]) {
                return Err(VisionError::Unsupported(format!(
                    "gemma4v block {il}: {group:?} clamp their shared input differently"
                )));
            }
        }
        for m in ["attn_out", "ffn_down"] {
            clip(weights, il, m)?;
        }
    }
    Ok(())
}

/// Whether the tower standardizes its pooled rows (`v.std_bias`, `v.std_scale`).
pub(super) fn standardizes(weights: &Weights) -> bool {
    weights.get("v.std_bias").is_some()
}

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
    // Clips only on the blocks' matrices (an mmproj's audio tower has its own, not read here).
    for t in g.tensors() {
        let Some(base) = CLIP_SCALARS.iter().find_map(|s| t.name.strip_suffix(s)) else {
            continue;
        };
        if base.starts_with("a.") || base.starts_with("mm.a.") {
            continue;
        }
        let clippable = base
            .strip_prefix("v.blk.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(il, m)| {
                il.parse::<usize>().is_ok_and(|il| il < n_layer) && CLIPPED.contains(&m)
            });
        if !clippable {
            return Err(VisionError::Unsupported(format!(
                "gemma4v with a clipped `{base}`"
            )));
        }
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

/// Every tensor the graph reads but the optional ones (the standardization's, the clips).
pub(super) fn names(n_layer: usize) -> Vec<String> {
    let mut v: Vec<String> = [
        "v.patch_embd.weight",
        "v.position_embd.weight",
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
        // mtmd's `build_mm`: a clipped linear clamps its input (in place) and its product.
        let mm = |m: &str, t| -> Result<Tensor<'_>, VisionError> {
            let wt = w(&blk(m))?;
            Ok(match clip(weights, il, m)? {
                None => ctx.mul_mat(wt, t),
                Some(cl) => ctx.clamp(
                    ctx.mul_mat(wt, ctx.clamp(t, cl.inp.0, cl.inp.1)),
                    cl.out.0,
                    cl.out.1,
                ),
            })
        };
        let cur = norm(x, &blk("ln1"))?;
        let head = |name: &str| -> Result<Tensor<'_>, VisionError> {
            Ok(ctx.reshape_4d(mm(name, cur)?, [d_head, n_head, n_pos, 1]))
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
        let attn = mm("attn_out", attn)?;
        x = ctx.add(norm(attn, &blk("attn_post_norm"))?, x);
        let cur = norm(x, &blk("ln2"))?;
        let gate = mm("ffn_gate", cur)?;
        let up = mm("ffn_up", cur)?;
        let cur = mm("ffn_down", ctx.geglu_split(gate, up))?;
        x = ctx.add(x, norm(cur, &blk("ffn_post_norm"))?);
    }

    // The pooler: [n_embd, n] -> [px, py, n_embd] -> 3×3 averages -> [n_embd, n / 9].
    let (ox, oy) = ((px / k) as i64, (py / k) as i64);
    let t = ctx.cont_4d(ctx.transpose(x), [px as i64, py as i64, n_embd, 1]);
    let t = ctx.avg_pool_2d(t, k as i32);
    let t = ctx.reshape_3d(t, ox * oy, n_embd, 1);
    let t = ctx.cont(ctx.transpose(t));
    let t = ctx.scale(t, (n_embd as f32).sqrt());
    let t = if standardizes(weights) {
        ctx.mul(ctx.sub(t, w("v.std_bias")?), w("v.std_scale")?)
    } else {
        t
    };
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
