//! GLM-5.3-Flash's vision tower of a `glm5v` mmproj (#1219): one image's preprocessed pixels to
//! the language model's input embeddings, one row per 2×2 block of patches.
//!
//! The graph is mtmd's, op for op, so that on the same backend it computes the same values:
//! 1. **Patches.** Two 14×14 convolutions of the image (the checkpoint's two temporal kernels;
//!    a still image is both of its frames), added; patches put in merge order (each 2×2 block's
//!    four patches next to each other, blocks row by row), plus the patch bias. No learned
//!    positions: attention rotates each head by the patch's (row, column) (`ROPE_VISION`, base
//!    10000), as Qwen3-VL's.
//! 2. **Blocks.** RMS-normed ViT blocks: one `qkv` matrix with a bias, `q` and `k` RMS-normed per
//!    head with weights before the rotation, full (non-causal) attention at `1 / sqrt(d)`, the
//!    output projection into the residual; an RMS norm and a SwiGLU MLP with biases whose gate is
//!    clamped from above and up from both sides at `swiglu_clamp`.
//! 3. **Merger.** The final RMS norm; each 2×2 block's four patches through a 2×2 stride-2
//!    convolution (with a bias) to the projector's width; a bias-free linear, a LayerNorm (eps
//!    1e-5), an exact GELU and a clamped SwiGLU MLP (bias-free in GLM-5.3-Flash) to the language
//!    model's width.
//!
//! Ported from llama.cpp's `tools/mtmd/models/glm4v.cpp` (`PROJECTOR_TYPE_GLM5V` takes its
//! graph) and `clip.cpp` (`build_vit` with `NORM_TYPE_RMS` and per-head q / k norms, `build_attn`
//! without flash attention, `build_ffn` with the SwiGLU clamp, the `GLM5V` hparams) (MIT).

use crate::ggml::{ComputeAllocator, Context, ROPE_VISION, Tensor, types};
use crate::gguf::GgufFile;

use super::qwen3vl::{
    Encoder, EncoderConfig, ImageEmbedding, merge_order, patch_positions, weight,
};
use super::{Preprocessed, VisionError, VisionHparams};

/// Graph nodes per ViT block, with room.
const NODES_PER_LAYER: usize = 64;
const NODES_EXTRA: usize = 256;
/// The rotation's base (`hparams.rope_theta`).
pub(super) const ROPE_THETA: f32 = 10000.0;
/// The merger's LayerNorm epsilon (llama.cpp's, the checkpoint's `nn.LayerNorm` default).
pub(super) const MERGER_EPS: f32 = 1e-5;

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
    let float = |k: &str| {
        g.get(&format!("clip.vision.{k}"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| meta(&format!("clip.vision.{k}")))
    };
    let n_out = g
        .tensor("mm.down.weight")
        .and_then(|t| t.shape.get(1).copied())
        .ok_or_else(|| meta("mm.down.weight"))? as usize;
    let cfg = EncoderConfig {
        n_embd: int("embedding_length")?,
        n_head: int("attention.head_count")?,
        n_layer,
        eps: float("attention.layer_norm_epsilon")?,
        patch_size: hp.patch_size as usize,
        merge: hp.spatial_merge_size as usize,
        pos_side: 0,
        n_out,
        swiglu_limit: float("swiglu_clamp")?,
    };
    if cfg.merge != 2
        || !cfg.n_embd.is_multiple_of(cfg.n_head)
        || !cfg.d_head().is_multiple_of(4)
        || !(cfg.swiglu_limit > 0.0 && cfg.swiglu_limit.is_finite())
    {
        return Err(meta(
            "spatial merge 2, a head width divisible by 4 and a positive swiglu_clamp",
        ));
    }
    Ok(cfg)
}

/// Every tensor the graph reads (the projector MLP's biases, absent in GLM-5.3-Flash, are
/// read when there).
pub(super) fn names(n_layer: usize) -> Vec<String> {
    let mut v: Vec<String> = [
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
        "v.patch_embd.bias",
        "v.post_ln.weight",
        "mm.patch_merger.weight",
        "mm.patch_merger.bias",
        "mm.model.fc.weight",
        "mm.post_norm.weight",
        "mm.post_norm.bias",
        "mm.gate.weight",
        "mm.up.weight",
        "mm.down.weight",
    ]
    .map(String::from)
    .to_vec();
    for il in 0..n_layer {
        for t in ["ln1", "ln2", "attn_q_norm", "attn_k_norm"] {
            v.push(format!("v.blk.{il}.{t}.weight"));
        }
        for t in ["attn_qkv", "attn_out", "ffn_gate", "ffn_up", "ffn_down"] {
            for p in ["weight", "bias"] {
                v.push(format!("v.blk.{il}.{t}.{p}"));
            }
        }
    }
    v
}

/// The tower's output for `p` (see the module docs).
pub(super) fn encode(enc: &mut Encoder, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
    encode_tapped(enc, p, None)
}

/// [`encode`], and with `taps` each stage's rows as well (tests: where the GPU tower parts from
/// this one), named as [`crate::vision::GpuEncoder`]'s GLM stages: `patches`, `blk.N.attn`
/// (the residual after block N's attention), `blk.N` (after its MLP), `post_ln`, `downsample`,
/// `fc`, `ln_gelu`, `out`; rows in merge order (blocks of four patches), one per row of `x`.
pub(super) fn encode_tapped(
    enc: &mut Encoder,
    p: &Preprocessed,
    taps: Option<&mut Vec<(String, Vec<f32>)>>,
) -> Result<ImageEmbedding, VisionError> {
    let c = enc.cfg.clone();
    let (px, py) = (p.grid_w as usize, p.grid_h as usize);
    if px % c.merge != 0 || py % c.merge != 0 || px == 0 || py == 0 {
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
    let positions = ctx
        .new_tensor(types::I32, &[4 * n_pos])
        .named("positions")
        .as_input();
    let weights = &enc.weights;
    let w = |name: &str| weight(weights, name);
    let ps = c.patch_size as i32;
    // Stage rows to read back (`taps`): copies kept as outputs, so nothing reuses them.
    let tapping = taps.is_some();
    let mut marks = Vec::new();
    let mut mark = |name: String, t| {
        if tapping {
            marks.push((name, ctx.cont(t).as_output()));
        }
    };

    // Patches: both temporal kernels over the one frame, in merge order, plus the bias.
    let conv = |k| ctx.conv_2d(k, raw, [ps, ps], [0, 0], [1, 1]);
    let inp = ctx.add(
        conv(w("v.patch_embd.weight")?),
        conv(w("v.patch_embd.weight.1")?),
    );
    let inp = merge_order(
        &ctx,
        ctx.permute(inp, [1, 2, 0, 3]),
        n_embd,
        px as i64,
        py as i64,
    );
    let mut x = ctx.add(inp, w("v.patch_embd.bias")?);
    mark("patches".into(), x);

    let rms = |t, name: &str| -> Result<Tensor<'_>, VisionError> {
        Ok(ctx.mul(ctx.rms_norm(t, c.eps), w(name)?))
    };
    let linear = |name: &str, t| -> Result<Tensor<'_>, VisionError> {
        let y = ctx.mul_mat(w(&format!("{name}.weight"))?, t);
        Ok(match weights.get(&format!("{name}.bias")) {
            Some(b) => ctx.add(y, b),
            None => y,
        })
    };
    // build_ffn's order: up, gate, the clamped SwiGLU, down.
    let ffn = |prefix: &str, sep: &str, t| -> Result<Tensor<'_>, VisionError> {
        let up = linear(&format!("{prefix}{sep}up"), t)?;
        let gate = linear(&format!("{prefix}{sep}gate"), t)?;
        linear(
            &format!("{prefix}{sep}down"),
            ctx.swiglu_clamp(gate, up, c.swiglu_limit),
        )
    };
    let quarter = (d_head / 4) as i32;
    let rope = |t| {
        ctx.rope_multi(
            t,
            positions,
            (d_head / 2) as i32,
            [quarter; 4],
            ROPE_VISION,
            32768,
            ROPE_THETA,
            1.0,
            0.0,
            1.0,
            32.0,
            1.0,
        )
    };
    let kq_scale = 1.0 / (d_head as f32).sqrt();
    for il in 0..c.n_layer {
        let blk = format!("v.blk.{il}");
        let cur = rms(x, &format!("{blk}.ln1.weight"))?;
        let qkv = linear(&format!("{blk}.attn_qkv"), cur)?;
        let es = qkv.element_size();
        let view = |at: i64| {
            ctx.view_3d(
                qkv,
                [d_head, n_head, n_pos],
                [d_head as usize * es, qkv.nb()[1]],
                (at * n_embd) as usize * es,
            )
        };
        let q = rope(rms(view(0), &format!("{blk}.attn_q_norm.weight"))?);
        let k = rope(rms(view(1), &format!("{blk}.attn_k_norm.weight"))?);
        let v = view(2);
        graph.expand(q);
        graph.expand(k);
        graph.expand(v);
        let q = ctx.permute(q, [0, 2, 1, 3]);
        let k = ctx.permute(k, [0, 2, 1, 3]);
        let v = ctx.cont(ctx.permute(v, [1, 2, 0, 3]));
        let kq = ctx.soft_max_scaled(ctx.mul_mat(k, q), kq_scale);
        let kqv = ctx.permute(ctx.mul_mat(v, kq), [0, 2, 1, 3]);
        let attn = ctx.cont_2d(kqv, n_embd, n_pos);
        x = ctx.add(linear(&format!("{blk}.attn_out"), attn)?, x);
        mark(format!("blk.{il}.attn"), x);
        let cur = rms(x, &format!("{blk}.ln2.weight"))?;
        x = ctx.add(x, ffn(&blk, ".ffn_", cur)?);
        mark(format!("blk.{il}"), x);
    }
    let x = rms(x, "v.post_ln.weight")?;
    mark("post_ln".into(), x);

    // The 2×2 convolution over each block's four patches: [n_embd, 2, 2, n / 4] as images
    // [2, 2, n_embd, n / 4].
    let merged = (c.merge * c.merge) as i64;
    let n_tok = n_pos / merged;
    let m = c.merge as i64;
    let t = ctx.reshape_4d(x, [n_embd, m, m, n_tok]);
    let t = ctx.cont(ctx.permute(t, [2, 0, 1, 3]));
    let t = ctx.conv_2d(
        w("mm.patch_merger.weight")?,
        t,
        [m as i32, m as i32],
        [0, 0],
        [1, 1],
    );
    let t = ctx.reshape_2d(t, t.ne()[2], n_tok);
    let t = ctx.add(t, w("mm.patch_merger.bias")?);
    mark("downsample".into(), t);

    // The projector: linear, LayerNorm, exact GELU, then the clamped SwiGLU MLP.
    let t = ctx.mul_mat(w("mm.model.fc.weight")?, t);
    mark("fc".into(), t);
    let t = ctx.add(
        ctx.mul(ctx.norm(t, MERGER_EPS), w("mm.post_norm.weight")?),
        w("mm.post_norm.bias")?,
    );
    let t = ctx.gelu_erf(t);
    mark("ln_gelu".into(), t);
    for &(_, t) in &marks {
        graph.expand(t);
    }
    let out = ffn("mm", ".", t)?.as_output();
    graph.expand(out);

    let allocator = enc
        .alloc
        .take()
        .unwrap_or_else(|| ComputeAllocator::new(enc.backend.buffer_type()));
    ctx.alloc_graph(&graph, allocator)
        .map_err(|e| VisionError::Compute(e.to_string()))?;
    raw.set_f32(&p.pixels);
    positions.set_i32(&patch_positions(px, py));
    enc.backend
        .compute(&graph)
        .map_err(|e| VisionError::Compute(e.to_string()))?;
    let data = out.to_f32();
    let stages: Vec<_> = marks
        .into_iter()
        .map(|(name, t)| (name, t.to_f32()))
        .collect();
    if let Some(taps) = taps {
        taps.extend(stages);
        taps.push(("out".into(), data.clone()));
    }
    enc.alloc = ctx.into_allocator();
    Ok(ImageEmbedding {
        nx: px / c.merge,
        ny: py / c.merge,
        n_embd: c.n_out,
        data,
    })
}

/// Whether [`encode_tapped`]'s stage `name` has a row per patch (in merge order), not one per
/// merged block.
#[cfg(test)]
pub(crate) fn per_patch(name: &str) -> bool {
    name == "patches" || name == "post_ln" || name.starts_with("blk.")
}

/// Rows of `width` in merge order (blocks of four patches) for a `gx × gy` grid, row by row
/// (tests).
#[cfg(test)]
pub(crate) fn merge_to_rows(v: &[f32], gx: usize, gy: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for (m, row) in v.chunks(width).enumerate().take(gx * gy) {
        let (blk, q) = (m / 4, m % 4);
        let (by, bx) = (blk / (gx / 2), blk % (gx / 2));
        let i = (by * 2 + q / 2) * gx + bx * 2 + q % 2;
        out[i * width..][..width].copy_from_slice(row);
    }
    out
}

/// How far rows `got` are from rows `want` (`rows` of each; tests): the values that are not
/// finite, the largest absolute difference, the largest `want` value, the smallest cosine of
/// a row with its row (and which row), the mean cosine.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Diff {
    pub(crate) bad: usize,
    pub(crate) max_abs: f32,
    pub(crate) scale: f32,
    pub(crate) min_cos: f64,
    pub(crate) worst: usize,
    pub(crate) mean_cos: f64,
}

#[cfg(test)]
impl Diff {
    pub(crate) fn new(got: &[f32], want: &[f32], rows: usize) -> Self {
        assert_eq!(got.len(), want.len());
        let width = want.len() / rows.max(1);
        let bad = got.iter().chain(want).filter(|v| !v.is_finite()).count();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max_abs = got
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let (mut min_cos, mut worst, mut sum) = (f64::INFINITY, 0, 0.0);
        for (r, (a, b)) in got.chunks(width).zip(want.chunks(width)).enumerate() {
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            let cos = if na == 0.0 && nb == 0.0 {
                1.0
            } else {
                dot / (na * nb).sqrt()
            };
            let cos = if cos.is_nan() { -2.0 } else { cos };
            sum += cos;
            if cos < min_cos {
                (min_cos, worst) = (cos, r);
            }
        }
        Self {
            bad,
            max_abs,
            scale,
            min_cos,
            worst,
            mean_cos: sum / rows.max(1) as f64,
        }
    }
}

#[cfg(test)]
impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max abs {:.3e} (rel {:.3e} of {:.3e}), min cos {:.6} (row {}), mean cos {:.6}",
            self.max_abs,
            self.max_abs / self.scale.max(f32::MIN_POSITIVE),
            self.scale,
            self.min_cos,
            self.worst,
            self.mean_cos
        )?;
        if self.bad > 0 {
            write!(f, ", {} values not finite", self.bad)?;
        }
        Ok(())
    }
}

/// The mean row cosine two GLM-5.3-Flash towers' rows `out` must reach on a checkpoint's
/// real weights (with every value finite and no row under [`MIN_ROW_COS`]). Not a high floor
/// on every row: from about block 10-14 on, a few rows carry a residual tens of times the
/// others' and the model amplifies rounding on just those. Measured on the GPU vs the CPU
/// tower, mean / min: chart 0.99951 / 0.877, colors 0.99810 / 0.929, gradient 0.99847 / 0.852,
/// scene 0.99853 / 0.622, shapes 0.99901 / 0.910, word 0.99900 / 0.890, and on the two small
/// images smooth 0.99974 / 0.997, stripes 0.99798. Against F64 loops both towers drift about
/// equally (mean, GPU / CPU: smooth 0.99612 / 0.99586, stripes 0.99873 / 0.99952; min row
/// smooth 0.963 / 0.965, stripes 0.989 / 0.996), so neither is wrong; the stage tables show
/// where it starts.
#[cfg(test)]
pub(crate) const MEAN_ROW_COS: f64 = 0.995;

/// The smallest row cosine allowed with [`MEAN_ROW_COS`].
#[cfg(test)]
pub(crate) const MIN_ROW_COS: f64 = 0.5;

/// Whether two towers' rows `out` agree as [`MEAN_ROW_COS`] says.
#[cfg(test)]
pub(crate) fn rows_agree(d: &Diff) -> bool {
    d.bad == 0 && d.mean_cos >= MEAN_ROW_COS && d.min_cos >= MIN_ROW_COS
}

#[cfg(test)]
pub(crate) mod tests;
