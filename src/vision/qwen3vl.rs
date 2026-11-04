//! The Qwen3-VL vision tower of a `qwen3vl_merger` mmproj (Qwen3.5 / Qwen3.6): one image's
//! preprocessed pixels to the language model's input embeddings, one row per merged 2×2 patch.
//!
//! The graph is mtmd's, op for op, so that on the same backend it computes the same values:
//! 1. **Patches.** Two 16×16 convolutions of the image (the checkpoint's two temporal kernels;
//!    a still image is both of its frames), added, plus the patch bias. Patches are then put in
//!    merge order: each 2×2 block's four patches next to each other, blocks row by row.
//! 2. **Positions.** The learned position table (a square grid) resampled bilinearly, with
//!    corners aligned, to the image's patch grid, and added. Attention rotates each head by the
//!    patch's (row, column) ([`ROPE_VISION`]).
//! 3. **Blocks.** Pre-norm ViT blocks: LayerNorm, full (non-causal) attention with one `qkv`
//!    matrix, LayerNorm, a GELU MLP.
//! 4. **Merger.** The final LayerNorm, the four patches of each block concatenated and a GELU
//!    MLP to the language model's width.
//!
//! Deepstack layers (Qwen3-VL's side outputs injected into the language model's first layers)
//! are refused: no `qwen35` checkpoint has them, and the language model would need them too.
//!
//! Ported from llama.cpp's `tools/mtmd/models/qwen3vl.cpp`, `models/qwen2vl.cpp`
//! (`build_inp_with_temporal_merge`) and `clip.cpp` (`build_attn` without flash attention,
//! `build_norm`, `build_ffn`, `resize_position_embeddings`, the `QWEN3VL` positions) (MIT).

use std::path::Path;

use crate::ggml::{
    Backend, ComputeAllocator, Context, ROPE_VISION, SCALE_ALIGN_CORNERS, SCALE_BILINEAR, Tensor,
    types,
};
use crate::gguf::GgufFile;
use crate::weights::Weights;

use super::{Preprocessed, VisionError, VisionHparams};

/// Graph nodes per ViT block, with room.
const NODES_PER_LAYER: usize = 48;
const NODES_EXTRA: usize = 256;

/// The tower's shape, from the mmproj's `clip.vision.*` keys and its tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub eps: f32,
    pub patch_size: usize,
    pub merge: usize,
    /// Side of the learned position grid (`v.position_embd.weight` has `side²` rows).
    pub pos_side: usize,
    /// Width of an output row: the language model's embedding width.
    pub n_out: usize,
}

impl EncoderConfig {
    pub fn from_gguf(g: &GgufFile, hp: &VisionHparams) -> Result<Self, VisionError> {
        let meta = |what: &str| VisionError::Metadata(what.to_string());
        let int = |k: &str| {
            g.get_u64(&format!("clip.vision.{k}"))
                .filter(|&v| v > 0)
                .map(|v| v as usize)
                .ok_or_else(|| meta(&format!("clip.vision.{k}")))
        };
        let n_layer = int("block_count")?;
        if let Some(v) = g.get("clip.vision.is_deepstack_layers")
            && v.as_array()
                .is_some_and(|a| a.iter().any(|x| x.as_bool() == Some(true)))
        {
            return Err(VisionError::Unsupported(
                "qwen3vl_merger with deepstack layers".into(),
            ));
        }
        let eps = g
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| meta("clip.vision.attention.layer_norm_epsilon"))?
            as f32;
        let pos_rows = g
            .tensor("v.position_embd.weight")
            .and_then(|t| t.shape.get(1).copied())
            .ok_or_else(|| meta("v.position_embd.weight"))? as usize;
        let pos_side = pos_rows.isqrt();
        if pos_side * pos_side != pos_rows {
            return Err(meta("v.position_embd.weight: not a square grid"));
        }
        let n_out = g
            .tensor("mm.2.bias")
            .and_then(|t| t.shape.first().copied())
            .ok_or_else(|| meta("mm.2.bias"))? as usize;
        let cfg = Self {
            n_embd: int("embedding_length")?,
            n_head: int("attention.head_count")?,
            n_layer,
            eps,
            patch_size: hp.patch_size as usize,
            merge: hp.spatial_merge_size as usize,
            pos_side,
            n_out,
        };
        if cfg.merge != 2
            || !cfg.n_embd.is_multiple_of(cfg.n_head)
            || !(cfg.n_embd / cfg.n_head).is_multiple_of(4)
        {
            return Err(meta("spatial merge 2 and a head width divisible by 4"));
        }
        Ok(cfg)
    }

    fn d_head(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// One image's language-model input: `nx × ny` rows of [`EncoderConfig::n_out`] values, row by
/// row (`nx` merged patches across).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageEmbedding {
    pub nx: usize,
    pub ny: usize,
    pub n_embd: usize,
    pub data: Vec<f32>,
}

impl ImageEmbedding {
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }

    /// The row of token `i` (`< nx × ny`).
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_embd..][..self.n_embd]
    }
}

/// The tower's weights on a backend, and the backend.
pub struct Encoder {
    cfg: EncoderConfig,
    hp: VisionHparams,
    weights: Weights,
    backend: Backend,
    alloc: Option<ComputeAllocator>,
}

impl Encoder {
    /// Loads `path` (an mmproj GGUF) onto `backend`.
    pub fn load(path: &Path, backend: Backend) -> Result<Self, VisionError> {
        let g = GgufFile::open(path)
            .map_err(|e| VisionError::Metadata(format!("{}: {e}", path.display())))?;
        let hp = VisionHparams::from_gguf(&g)?;
        let cfg = EncoderConfig::from_gguf(&g, &hp)?;
        let buft = backend.buffer_type();
        let weights = Weights::load(&g, |_| buft)
            .map_err(|e| VisionError::Metadata(format!("{}: {e}", path.display())))?;
        let enc = Self {
            cfg,
            hp,
            weights,
            backend,
            alloc: None,
        };
        // Every tensor the graph reads, checked once.
        for name in enc.names() {
            enc.w(&name)?;
        }
        Ok(enc)
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    pub fn hparams(&self) -> &VisionHparams {
        &self.hp
    }

    /// Decodes, preprocesses and encodes an image file's bytes.
    pub fn encode_bytes(&mut self, bytes: &[u8]) -> Result<ImageEmbedding, VisionError> {
        let p = super::preprocess(&self.hp, bytes)?;
        self.encode(&p)
    }

    fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = [
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.post_ln.weight",
            "v.post_ln.bias",
            "mm.0.weight",
            "mm.0.bias",
            "mm.2.weight",
            "mm.2.bias",
        ]
        .map(String::from)
        .to_vec();
        for il in 0..self.cfg.n_layer {
            for t in ["ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down"] {
                for p in ["weight", "bias"] {
                    v.push(format!("v.blk.{il}.{t}.{p}"));
                }
            }
        }
        v
    }

    fn w(&self, name: &str) -> Result<Tensor<'_>, VisionError> {
        weight(&self.weights, name)
    }

    /// The tower's output for `p` (see the module docs).
    pub fn encode(&mut self, p: &Preprocessed) -> Result<ImageEmbedding, VisionError> {
        let c = &self.cfg;
        let (px, py) = (p.grid_w as usize, p.grid_h as usize);
        if px % c.merge != 0 || py % c.merge != 0 || px == 0 || py == 0 {
            return Err(VisionError::Dimensions);
        }
        let n_pos = px * py;
        let (n_embd, d_head, n_head) = (c.n_embd as i64, c.d_head() as i64, c.n_head as i64);
        let size = NODES_EXTRA + NODES_PER_LAYER * c.n_layer;
        let ctx = Context::new(size, size);
        let mut graph = ctx.new_graph(size);
        let raw = ctx
            .new_tensor(types::F32, &[p.width as i64, p.height as i64, 3, 1])
            .named("inp_raw")
            .as_input();
        let positions = ctx
            .new_tensor(types::I32, &[4 * n_pos as i64])
            .named("positions")
            .as_input();
        let weights = &self.weights;
        let w = |name: &str| weight(weights, name);
        let ps = c.patch_size as i32;

        // Patches: both temporal kernels over the one frame.
        let conv = |k| ctx.conv_2d(k, raw, [ps, ps], [0, 0], [1, 1]);
        let inp = ctx.add(
            conv(w("v.patch_embd.weight")?),
            conv(w("v.patch_embd.weight.1")?),
        );
        let merge_order = |t| merge_order(&ctx, t, n_embd, px as i64, py as i64);
        let inp = merge_order(ctx.permute(inp, [1, 2, 0, 3]));
        let inp = ctx.add(inp, w("v.patch_embd.bias")?);

        // Learned positions, resampled to the patch grid.
        let mut pos = w("v.position_embd.weight")?;
        if px != c.pos_side || py != c.pos_side {
            let side = c.pos_side as i64;
            pos = ctx.reshape_3d(pos, n_embd, side, side);
            pos = ctx.permute(pos, [2, 0, 1, 3]);
            pos = ctx.interpolate(
                pos,
                [px as i64, py as i64, n_embd, 1],
                SCALE_BILINEAR | SCALE_ALIGN_CORNERS,
            );
            pos = ctx.permute(pos, [1, 2, 0, 3]);
            pos = ctx.cont_2d(pos, n_embd, n_pos as i64);
        }
        let mut x = ctx.add(inp, merge_order(pos));

        let norm = |t, il: Option<usize>, name: &str| -> Result<Tensor<'_>, VisionError> {
            let prefix = match il {
                Some(il) => format!("v.blk.{il}.{name}"),
                None => format!("v.{name}"),
            };
            Ok(ctx.add(
                ctx.mul(ctx.norm(t, c.eps), w(&format!("{prefix}.weight"))?),
                w(&format!("{prefix}.bias"))?,
            ))
        };
        let linear = |name: &str, t| -> Result<Tensor<'_>, VisionError> {
            Ok(ctx.add(
                ctx.mul_mat(w(&format!("{name}.weight"))?, t),
                w(&format!("{name}.bias"))?,
            ))
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
                10000.0,
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
            let cur = norm(x, Some(il), "ln1")?;
            let qkv = linear(&format!("{blk}.attn_qkv"), cur)?;
            let es = qkv.element_size();
            let view = |at: i64| {
                ctx.view_3d(
                    qkv,
                    [d_head, n_head, n_pos as i64],
                    [d_head as usize * es, qkv.nb()[1]],
                    (at * n_embd) as usize * es,
                )
            };
            let (q, k, v) = (rope(view(0)), rope(view(1)), view(2));
            graph.expand(q);
            graph.expand(k);
            graph.expand(v);
            let q = ctx.permute(q, [0, 2, 1, 3]);
            let k = ctx.permute(k, [0, 2, 1, 3]);
            let v = ctx.cont(ctx.permute(v, [1, 2, 0, 3]));
            let kq = ctx.soft_max_scaled(ctx.mul_mat(k, q), kq_scale);
            let kqv = ctx.permute(ctx.mul_mat(v, kq), [0, 2, 1, 3]);
            let attn = ctx.cont_2d(kqv, n_embd, n_pos as i64);
            let attn = linear(&format!("{blk}.attn_out"), attn)?;
            x = ctx.add(attn, x);
            let cur = norm(x, Some(il), "ln2")?;
            let cur = ctx.gelu(linear(&format!("{blk}.ffn_up"), cur)?);
            let cur = linear(&format!("{blk}.ffn_down"), cur)?;
            x = ctx.add(x, cur);
        }
        let x = norm(x, None, "post_ln")?;
        let merged = (c.merge * c.merge) as i64;
        let x = ctx.reshape_3d(x, n_embd * merged, n_pos as i64 / merged, 1);
        let x = ctx.gelu(linear("mm.0", x)?);
        let out = linear("mm.2", x)?.as_output();
        graph.expand(out);

        let allocator = self
            .alloc
            .take()
            .unwrap_or_else(|| ComputeAllocator::new(self.backend.buffer_type()));
        ctx.alloc_graph(&graph, allocator)
            .map_err(|e| VisionError::Compute(e.to_string()))?;
        raw.set_f32(&p.pixels);
        positions.set_i32(&patch_positions(px, py));
        self.backend
            .compute(&graph)
            .map_err(|e| VisionError::Compute(e.to_string()))?;
        let data = out.to_f32();
        self.alloc = ctx.into_allocator();
        Ok(ImageEmbedding {
            nx: px / c.merge,
            ny: py / c.merge,
            n_embd: c.n_out,
            data,
        })
    }
}

fn weight<'w>(weights: &'w Weights, name: &str) -> Result<Tensor<'w>, VisionError> {
    weights
        .get(name)
        .ok_or_else(|| VisionError::Metadata(format!("the mmproj has no tensor `{name}`")))
}

/// Patch rows `[n_embd, px, py]` (any layout) in merge order: each 2×2 block's four patches
/// next to each other, blocks row by row; `[n_embd, px × py]`.
fn merge_order<'a>(ctx: &'a Context, t: Tensor<'a>, n_embd: i64, px: i64, py: i64) -> Tensor<'a> {
    let t = ctx.cont(t);
    let t = ctx.reshape_4d(t, [n_embd * 2, px / 2, py, 1]);
    let t = ctx.reshape_4d(t, [n_embd * 2, px / 2, 2, py / 2]);
    let t = ctx.permute(t, [0, 2, 1, 3]);
    let t = ctx.cont(t);
    ctx.reshape_3d(t, n_embd, px * py, 1)
}

/// The patches' (row, column) for [`ROPE_VISION`], in merge order, as four sections of
/// `px × py` entries: rows, columns, rows, columns.
pub(super) fn patch_positions(px: usize, py: usize) -> Vec<i32> {
    let n = px * py;
    let mut pos = vec![0i32; 4 * n];
    let mut i = 0;
    for y in (0..py).step_by(2) {
        for x in (0..px).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, c) = ((y + dy) as i32, (x + dx) as i32);
                    pos[i] = r;
                    pos[n + i] = c;
                    pos[2 * n + i] = r;
                    pos[3 * n + i] = c;
                    i += 1;
                }
            }
        }
    }
    pos
}

#[cfg(test)]
mod tests;
