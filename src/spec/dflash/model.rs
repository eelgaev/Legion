//! The DFlash network as ggml graphs.
//!
//! Two graphs, per sequence or, for DFlash 2 (whose block length is the same for every
//! sequence), for several sequences at once: the matmuls run over all their rows, the
//! attention per sequence (the caches differ in length).
//!
//! - **ingest**: target hidden states of the fused layers, `[n_layers × n_embd, n]`, go through
//!   `fc` and an RMS norm (`hidden_norm`); every drafter layer projects the result to K and V
//!   (K: per-head RMS norm, then RoPE at the row's position) and writes them to the cache.
//! - **draft**: the noise block `[last token, <mask> × (n - 1)]` at positions `p, p + 1, ...`
//!   (`p`: the last token's position) runs through the layers. Attention is not causal: every
//!   block position attends to the whole cached context and to the whole block. The final norm
//!   and the output head give a distribution for every block position after the first; each is
//!   a draft token. One pass drafts the block.
//!
//! The layers are Qwen3's: RMS-normed input, GQA attention with per-head q/k norms and NeoX
//! RoPE, SwiGLU FFN. Adapted from llama.cpp's `src/models/dflash.cpp` (MIT) and the DFlash
//! paper, <https://arxiv.org/abs/2602.06036>.

use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, ffi, types};

use super::cache::{CachePool, SeqCache};
use super::config::{Config, Flavor};
use super::v2::{self, Selector};
use super::weights::{HostTensor, Store};

#[cfg(test)]
mod tests;

const GRAPH_SIZE: usize = 4096;
/// Graph nodes a batched draft pass adds per sequence after the first (its attentions).
const GRAPH_PER_SEQ: usize = 512;
/// Rows ingested per graph.
pub const INGEST_CHUNK: usize = 512;
const ROPE_NEOX: i32 = 2;

/// What the draft pass returns for the block positions it was asked for.
pub struct Rows {
    /// The most likely token (a draft-vocabulary row) of every position (not for DFlash 2).
    pub argmax: Vec<u32>,
    /// Logits `[position][draft vocabulary]`, when asked for.
    pub logits: Option<Vec<f32>>,
    /// The final-normed hidden states `[position][n_embd]`, when asked for.
    pub hidden: Option<Vec<f32>>,
    /// DFlash 2: the selector's inputs, `[position][top_k]` candidates and their logits, and
    /// `[position][rank]` hidden projections.
    pub selector: Option<(Vec<u32>, Vec<f32>, Vec<f32>)>,
}

/// One sequence of a draft pass: its cache, and its last token at `anchor_pos`.
pub struct BlockSeq<'c> {
    pub cache: &'c SeqCache,
    pub last: u32,
    pub anchor_pos: usize,
}

/// A draft pass's attention over a [`CachePool`] (`--spec-draft-batched-attn`): per layer, one
/// write of every block's K/V to its slot's scratch cells and one flash attention over the
/// slots `0..n_view` (`n_view`: the highest slot in the pass, plus one), each slot's first
/// `n_kv` cells (the longest cache's, rounded up to whole 256-cell tiles; the rest of a slot's
/// are masked, and ggml-cuda skips its fully masked tiles). Slots the pass does not use attend
/// to their first cell only and are dropped. The queries are gathered into slot order and the
/// outputs back into the pass's order, unless the sequences hold slots `0..n_seq` in order.
struct Pooled<'a> {
    pool: &'a CachePool,
    n_kv: usize,
    n_view: usize,
    /// `[n_seq · n]`: where each block row goes in the pool.
    block_cells: Tensor<'a>,
    /// `[n_view]`: the pass's sequence of each slot (gather of the queries).
    q_idx: Option<Tensor<'a>>,
    /// `[n_seq]`: each sequence's slot (gather of the outputs).
    o_idx: Option<Tensor<'a>>,
    /// The masks as they are set, F32 `[n_kv, n, 1, n_view]` (full attention, sliding), and
    /// the F16 copies flash attention reads.
    masks_f32: (Option<Tensor<'a>>, Option<Tensor<'a>>),
    mask_full: Option<Tensor<'a>>,
    mask_swa: Option<Tensor<'a>>,
}

impl<'a> Pooled<'a> {
    fn new(
        ctx: &'a Context,
        pool: &'a CachePool,
        seqs: &[BlockSeq<'_>],
        n: usize,
        any_full: bool,
        any_swa: bool,
    ) -> Self {
        let slots: Vec<usize> = seqs
            .iter()
            .map(|s| s.cache.slot().expect("pooled"))
            .collect();
        let n_view = slots.iter().max().expect("a sequence") + 1;
        let used = seqs
            .iter()
            .map(|s| s.cache.used_cells())
            .max()
            .expect("a sequence");
        let n_kv = used
            .next_multiple_of(super::cache::POOL_CELLS)
            .min(pool.cells());
        let in_order = n_view == seqs.len() && slots.iter().enumerate().all(|(i, &s)| i == s);
        let input = |ty, ne: &[i64], name: &str| ctx.new_tensor(ty, ne).named(name).as_input();
        let block_cells = input(types::I32, &[(n * seqs.len()) as i64], "pool_cells");
        let q_idx = (!in_order).then(|| input(types::I32, &[n_view as i64], "pool_q"));
        let o_idx = (!in_order).then(|| input(types::I32, &[seqs.len() as i64], "pool_o"));
        let ne = [n_kv as i64, n as i64, 1, n_view as i64];
        let f32_mask = |name: &str| input(types::F32, &ne, name);
        let masks_f32 = (
            any_full.then(|| f32_mask("pool_mask")),
            any_swa.then(|| f32_mask("pool_mask_swa")),
        );
        Self {
            pool,
            n_kv,
            n_view,
            block_cells,
            q_idx,
            o_idx,
            mask_full: masks_f32.0.map(|m| ctx.cast(m, types::F16)),
            mask_swa: masks_f32.1.map(|m| ctx.cast(m, types::F16)),
            masks_f32,
        }
    }

    /// Sets the inputs, after the graph is allocated.
    fn set_inputs(&self, model: &Model, seqs: &[BlockSeq<'_>], n: usize) {
        let cells = self.pool.cells();
        let slot = |s: &BlockSeq<'_>| s.cache.slot().expect("pooled");
        let at: Vec<i32> = seqs
            .iter()
            .flat_map(|s| (0..n).map(move |j| (slot(s) * cells + j) as i32))
            .collect();
        self.block_cells.set_i32(&at);
        if let (Some(q), Some(o)) = (self.q_idx, self.o_idx) {
            let mut of_slot = vec![0i32; self.n_view];
            for (i, s) in seqs.iter().enumerate() {
                of_slot[slot(s)] = i as i32;
            }
            q.set_i32(&of_slot);
            o.set_i32(&seqs.iter().map(|s| slot(s) as i32).collect::<Vec<_>>());
        }
        let window = model.cfg.sliding_window.as_ref().map(|(w, _)| *w);
        for (t, window) in [(self.masks_f32.0, None), (self.masks_f32.1, window)] {
            let Some(t) = t else { continue };
            // A slot no sequence of the pass holds sees its first cell (no row without keys).
            let mut idle = vec![f32::NEG_INFINITY; self.n_kv * n];
            idle.iter_mut().step_by(self.n_kv).for_each(|x| *x = 0.0);
            let mut all = Vec::with_capacity(self.n_kv * n * self.n_view);
            for v in 0..self.n_view {
                match seqs.iter().find(|s| slot(s) == v) {
                    Some(s) => all.extend(model.mask(s.cache, s.anchor_pos, n, self.n_kv, window)),
                    None => all.extend_from_slice(&idle),
                }
            }
            t.set_f32(&all);
        }
    }
}

/// DSpark's Markov head: the logits of a draft position get the bias
/// `markov_w2 · markov_w1[previous token]`, a function of the token before it (the last
/// token for the first draft, then each draft in turn). `markov_w1` is kept on the host
/// (one row is read per drafted position); `markov_w2` runs on the backend.
struct Markov {
    rank: usize,
    w1: Vec<u8>,
    w1_type: ffi::ggml_type,
    /// Confidence head: weights over `[hidden; w1 row]` and a bias.
    conf: Option<(Vec<f32>, f32)>,
}

pub struct Model {
    pub cfg: Config,
    store: Store,
    /// Rows of the output head: the draft vocabulary.
    pub n_vocab: usize,
    /// Target token id of every draft-vocabulary row (reduced-vocabulary drafters).
    pub d2t: Option<Vec<u32>>,
    pub window: usize,
    /// The first block position that yields a draft: 1 for DFlash (position 0 is the last
    /// token), 0 for a DSpark that samples from the anchor.
    pub first_row: usize,
    markov: Option<Markov>,
    /// DFlash 2's selector codebooks.
    selector: Option<Selector>,
}

/// `n` values of `ty` in `bytes` as floats (F32 has no conversion function in ggml's table).
pub(super) fn floats(ty: ffi::ggml_type, bytes: &[u8], n: usize) -> Vec<f32> {
    if ty == types::F32 {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect()
    } else {
        crate::ggml::dequantize(ty, bytes, n)
    }
}

/// The drafter's compute allocator, with a little headroom. Its graphs each need a few KiB more
/// than the one before (every sequence's cache grows by its accepted tokens per step), and a
/// plain allocator freed and allocated its buffer again nearly every step: a `cudaFree` and a
/// `cudaMalloc` of ~5 ms in a step of ~177 ms (Gemma 4 31B, DFlash 2, 16 users, a DGX Spark).
fn drafter_allocator(backend: &Backend) -> ComputeAllocator {
    ComputeAllocator::headroom(backend.buffer_type())
}

/// Part `s` of `n` equal parts of `v` (one sequence's rows of a batched pass's output).
fn part<T: Clone>(v: &[T], n: usize, s: usize) -> Vec<T> {
    let len = v.len() / n;
    v[s * len..(s + 1) * len].to_vec()
}

/// Sequence `s`'s block of `n` positions in a batched `[a, b, n_seq · n]` tensor: a contiguous
/// view of its columns `s · n..(s + 1) · n`.
fn block_of<'a>(ctx: &'a Context, t: Tensor<'a>, n: usize, s: usize) -> Tensor<'a> {
    let [a, b, _, _] = t.ne();
    let nb = t.nb();
    ctx.view_3d(t, [a, b, n as i64], [nb[1], nb[2]], s * n * nb[2])
}

/// `w @ x` for each of `ws` (the same input), as the target's linears run their BF16 weights:
/// steps of 2-16 rows (a draft block of one or two sequences, a short ingest) on the dense
/// matmul op, ggml-cuda's bits from a kernel that reads the weights faster, in one launch for
/// all of `ws`; 17-48 rows on its wide chain where that is on (`--dense-mv-wide-chain`, whose
/// sums can differ from cuBLAS's: drafts only, every token is verified); else ggml's matmul.
fn mul_mats<'a>(ctx: &'a Context, ws: &[Tensor<'a>], x: Tensor<'a>) -> Vec<Tensor<'a>> {
    use crate::model::dense_mv;
    if dense_mv::takes_all(ws, x) {
        dense_mv::mul_mat(ctx, ws, x)
    } else if dense_mv::takes_wide_chains(ws, x) {
        dense_mv::mul_mat_wide_chains(ctx, ws, x)
    } else {
        ws.iter().map(|&w| mul_mat(ctx, w, x)).collect()
    }
}

/// [`mul_mats`] of one weight.
fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    use crate::model::dense_mv;
    if dense_mv::takes(w, x) {
        dense_mv::mul_mat(ctx, &[w], x)[0]
    } else if dense_mv::takes_wide_chain(w, x) {
        dense_mv::mul_mat_wide_chain(ctx, w, x)
    } else {
        ctx.mul_mat(w, x)
    }
}

fn host_f32(t: Tensor<'_>) -> Vec<f32> {
    floats(t.ty(), &t.bytes(), t.elements())
}

fn rope<'a>(
    ctx: &'a Context,
    a: Tensor<'a>,
    pos: Tensor<'a>,
    n_rot: usize,
    base: f32,
) -> Tensor<'a> {
    // SAFETY: valid tensors of one context; no frequency factors, no YaRN.
    ctx.wrap(unsafe {
        ffi::ggml_rope_ext(
            ctx.raw(),
            a.raw(),
            pos.raw(),
            std::ptr::null_mut(),
            n_rot as i32,
            ROPE_NEOX,
            0,
            base,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    })
}

/// [`Model::first_row`] of a `flavor` drafter with `cfg`.
pub fn first_row(flavor: Flavor, cfg: &Config) -> usize {
    usize::from(!(flavor == Flavor::Dspark && cfg.sample_from_anchor))
}

impl Model {
    /// Checks that every tensor the network needs is there with the right shape. `codebooks`
    /// are DFlash 2's selector codebooks (predecessor, successor), read on the host.
    /// `shortlist`: the store's `target.output.weight` is the first rows of the target's head
    /// only (`--spec-draft-vocab`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: Config,
        store: Store,
        n_target_embd: usize,
        n_target_vocab: usize,
        window: usize,
        flavor: Flavor,
        codebooks: Option<(HostTensor, HostTensor)>,
        shortlist: bool,
    ) -> Result<Self, String> {
        let c = &cfg;
        let e = c.n_embd;
        let (q, kv) = (c.head_dim * c.n_head, c.head_dim * c.n_head_kv);
        store.require("fc.weight", &[c.target_layers.len() * n_target_embd, e])?;
        store.require("enc.output_norm.weight", &[e])?;
        store.require("output_norm.weight", &[e])?;
        for il in 0..c.n_layer {
            let t = |n: &str| format!("blk.{il}.{n}");
            store.require(&t("attn_norm.weight"), &[e])?;
            store.require(&t("attn_q.weight"), &[e, q])?;
            store.require(&t("attn_k.weight"), &[e, kv])?;
            if store.get(&t("attn_v.weight")).is_some() {
                store.require(&t("attn_v.weight"), &[e, kv])?;
            }
            store.require(&t("attn_output.weight"), &[q, e])?;
            store.require(&t("attn_q_norm.weight"), &[c.head_dim])?;
            store.require(&t("attn_k_norm.weight"), &[c.head_dim])?;
            store.require(&t("ffn_norm.weight"), &[e])?;
            store.require(&t("ffn_gate.weight"), &[e, c.n_ff])?;
            store.require(&t("ffn_up.weight"), &[e, c.n_ff])?;
            store.require(&t("ffn_down.weight"), &[c.n_ff, e])?;
            if let Some(sinks) = Self::sinks_of(&store, il) {
                store.require(&sinks, &[c.n_head])?;
            }
            if let Some(v2) = &c.v2 {
                for part in ["attn", "ffn"] {
                    store.require(&t(&format!("{part}_conv_base")), &[e, v2.conv_kernel, 2])?;
                    store.require(
                        &t(&format!("{part}_conv_proj.weight")),
                        &[e, v2.projected(e)],
                    )?;
                }
            }
        }
        let embd = store
            .get("token_embd.weight")
            .or_else(|| store.get("target.token_embd.weight"))
            .ok_or("the drafter has no token embeddings and neither has the target model")?;
        if embd.shape().first() != Some(&(e as i64)) {
            return Err(format!(
                "token embeddings are {:?} wide, the drafter's hidden size is {e}",
                embd.shape().first()
            ));
        }
        let head = store
            .get("output.weight")
            .or_else(|| store.get("target.output.weight"))
            .or_else(|| store.get("target.token_embd.weight"))
            .unwrap_or(embd);
        if head.shape().first() != Some(&(e as i64)) {
            return Err(format!(
                "the output head is {:?} wide, the drafter's hidden size is {e}",
                head.shape().first()
            ));
        }
        let n_vocab = head.shape().get(1).copied().unwrap_or(1) as usize;
        let d2t = match store.get("d2t") {
            Some(t) => {
                let ids: Vec<u32> = t
                    .bytes()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_ne_bytes(*b) as u32)
                    .collect();
                if ids.len() != n_vocab {
                    return Err(format!(
                        "`d2t` has {} entries for a draft vocabulary of {n_vocab}",
                        ids.len()
                    ));
                }
                if ids.iter().any(|&i| i as usize >= n_target_vocab) {
                    return Err("`d2t` names tokens outside the target's vocabulary".into());
                }
                Some(ids)
            }
            // The first rows of the target's own head (`--spec-draft-vocab`): drafts only
            // come from them.
            None if shortlist
                && n_vocab < n_target_vocab
                && store.get("output.weight").is_none()
                && store.get("target.output.weight").is_some() =>
            {
                None
            }
            None => {
                if n_vocab != n_target_vocab {
                    return Err(format!(
                        "the drafter's vocabulary has {n_vocab} tokens, the target's \
                         {n_target_vocab}, and there is no `d2t` mapping"
                    ));
                }
                None
            }
        };
        if cfg.mask_token as usize >= n_target_vocab {
            return Err(format!(
                "mask token {} is outside the target's vocabulary",
                cfg.mask_token
            ));
        }
        let markov = match flavor {
            Flavor::Dflash => None,
            Flavor::Dspark => {
                let w1 = store
                    .get("markov_w1.weight")
                    .ok_or("the drafter has no Markov head (`markov_w1.weight`): it is a DFlash")?;
                let rank = w1.shape().first().copied().unwrap_or(0) as usize;
                store.require("markov_w1.weight", &[rank, n_target_vocab])?;
                store.require("markov_w2.weight", &[rank, n_vocab])?;
                if store.get("markov_w2.scale").is_some() {
                    store.require("markov_w2.scale", &[1])?;
                }
                let conf = match store.get("conf_proj.weight") {
                    Some(_) => {
                        let w = host_f32(store.require("conf_proj.weight", &[e + rank, 1])?);
                        let b = match store.get("conf_proj.bias") {
                            Some(_) => host_f32(store.require("conf_proj.bias", &[1])?)[0],
                            None => 0.0,
                        };
                        Some((w, b))
                    }
                    None => None,
                };
                Some(Markov {
                    rank,
                    w1_type: w1.ty(),
                    w1: w1.bytes(),
                    conf,
                })
            }
        };
        // An FP8 head kept raw (an `I8` tensor of E4M3 bytes) runs on Legion's FP8 matmul, which
        // takes an F32 scale of one value or one per row.
        if head.ty() == types::I8 {
            let scale = if store.get("output.weight").is_some() {
                store.get("output.scale")
            } else {
                store.get("target.output.scale")
            };
            if let Some(s) = scale
                && !(s.ty() == types::F32 && (s.elements() == 1 || s.elements() == n_vocab))
            {
                return Err(format!(
                    "the FP8 output head's scale is {:?} of type {}, expected F32 of 1 or \
                     {n_vocab} values",
                    s.shape(),
                    crate::ggml::type_name(s.ty())
                ));
            }
            if !e.is_multiple_of(32) {
                return Err(format!(
                    "an FP8 output head needs a hidden size that is a multiple of 32, not {e}"
                ));
            }
        }
        let selector = match (&cfg.v2, codebooks) {
            (None, _) => None,
            (Some(_), None) => return Err("DFlash 2 drafter without selector codebooks".into()),
            (Some(v2), Some((pred, succ))) => {
                let rank = v2.selector_rank;
                if d2t.is_some() {
                    return Err("a DFlash 2 drafter with a reduced vocabulary (`d2t`)".into());
                }
                if v2.selector_top_k > n_vocab {
                    return Err(format!(
                        "selector top-k {} of a vocabulary of {n_vocab}",
                        v2.selector_top_k
                    ));
                }
                store.require("selector_hidden.weight", &[e, rank])?;
                for (name, t) in [("predecessor", &pred), ("successor", &succ)] {
                    // Every target token has codebook rows (the walk starts from the last
                    // token), even when the head is a shortlist.
                    if t.shape != [rank, n_target_vocab] {
                        return Err(format!(
                            "selector {name} codebook has shape {:?}, expected {:?}",
                            t.shape,
                            [rank, n_target_vocab]
                        ));
                    }
                }
                if pred.ty != succ.ty {
                    return Err("the selector codebooks differ in type".into());
                }
                Some(Selector::new(
                    rank,
                    v2.selector_top_k,
                    pred.ty,
                    pred.bytes,
                    succ.bytes,
                ))
            }
        };
        let first_row = first_row(flavor, &cfg);
        Ok(Self {
            cfg,
            store,
            n_vocab,
            d2t,
            window,
            first_row,
            markov,
            selector,
        })
    }

    /// DFlash 2's selector.
    pub fn selector(&self) -> Option<&Selector> {
        self.selector.as_ref()
    }

    /// The tensor name of layer `il`'s attention sinks: `blk.N.attn_sinks.weight`, or
    /// `blk.N.attn_sinks` as some conversions leave it.
    fn sinks_of(store: &Store, il: usize) -> Option<String> {
        [
            format!("blk.{il}.attn_sinks.weight"),
            format!("blk.{il}.attn_sinks"),
        ]
        .into_iter()
        .find(|n| store.get(n).is_some())
    }

    fn w(&self, name: &str) -> Tensor<'_> {
        self.store
            .get(name)
            .unwrap_or_else(|| panic!("tensor `{name}` was checked at load"))
    }

    fn lw(&self, il: usize, name: &str) -> Tensor<'_> {
        self.w(&format!("blk.{il}.{name}"))
    }

    fn embeddings(&self) -> Tensor<'_> {
        self.store
            .get("token_embd.weight")
            .or_else(|| self.store.get("target.token_embd.weight"))
            .expect("checked at load")
    }

    fn head(&self) -> Tensor<'_> {
        self.store
            .get("output.weight")
            .or_else(|| self.store.get("target.output.weight"))
            .or_else(|| self.store.get("target.token_embd.weight"))
            .unwrap_or_else(|| self.embeddings())
    }

    /// The per-tensor scale of the output head (an NVFP4 head's `weight_scale_2`), if any:
    /// the logits are the matmul times it.
    fn head_scale(&self) -> Option<Tensor<'_>> {
        if self.store.get("output.weight").is_some() {
            self.store.get("output.scale")
        } else if self.store.get("target.output.weight").is_some() {
            self.store.get("target.output.scale")
        } else {
            None
        }
    }

    /// DFlash 2: layer `il`'s base kernel of `part` (`attn` or `ffn`) in F32.
    fn conv_base<'a>(&'a self, ctx: &'a Context, il: usize, part: &str) -> Tensor<'a> {
        let t = self.lw(il, &format!("{part}_conv_base"));
        if t.ty() == types::F32 {
            t
        } else {
            ctx.cast(t, types::F32)
        }
    }

    /// DFlash 2: `x` through the input side of layer `il`'s convolution around `part`, and the
    /// dynamic coefficients of its output side ([`Self::conv_out`]). Without DFlash 2: `x`.
    /// `x` holds blocks of `block` positions (one per sequence).
    fn conv_in<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        part: &str,
        x: Tensor<'a>,
        block: i64,
    ) -> (Tensor<'a>, Option<Tensor<'a>>) {
        match &self.cfg.v2 {
            Some(v2) => {
                let dynamic = ctx.mul_mat(self.lw(il, &format!("{part}_conv_proj.weight")), x);
                let base = self.conv_base(ctx, il, part);
                (v2::conv(ctx, x, dynamic, base, 0, v2, block), Some(dynamic))
            }
            None => (x, None),
        }
    }

    /// DFlash 2: `y` (the attention's or the MLP's output) through the output side of the
    /// convolution [`Self::conv_in`] began. Without DFlash 2: `y`.
    fn conv_out<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        part: &str,
        y: Tensor<'a>,
        dynamic: Option<Tensor<'a>>,
        block: i64,
    ) -> Tensor<'a> {
        match (&self.cfg.v2, dynamic) {
            (Some(v2), Some(dynamic)) => {
                let base = self.conv_base(ctx, il, part);
                v2::conv(ctx, y, dynamic, base, 1, v2, block)
            }
            _ => y,
        }
    }

    /// K and V of the rows `x` (`[n_embd, n]`, after the layer's input norm for the noise
    /// block, the fused features as they are for the context) in layer `il`.
    fn kv<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        x: Tensor<'a>,
        pos: Tensor<'a>,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let c = &self.cfg;
        let n = x.ne()[1];
        let (d, kvh) = (c.head_dim as i64, c.n_head_kv as i64);
        let wk = self.lw(il, "attn_k.weight");
        let (k, v) = match self.store.get(&format!("blk.{il}.attn_v.weight")) {
            Some(wv) => {
                let kv = mul_mats(ctx, &[wk, wv], x);
                (
                    ctx.reshape_3d(kv[0], d, kvh, n),
                    ctx.reshape_3d(kv[1], d, kvh, n),
                )
            }
            // No V projection: V is the K projection, normed without a weight.
            None => {
                let k = ctx.reshape_3d(mul_mat(ctx, wk, x), d, kvh, n);
                (k, ctx.rms_norm(k, c.rms_eps))
            }
        };
        let k = ctx.mul(
            ctx.rms_norm(k, c.rms_eps),
            self.lw(il, "attn_k_norm.weight"),
        );
        (rope(ctx, k, pos, c.n_rot, c.rope_base), v)
    }

    /// Runs the fused target features `feats` (`n_layers × n_embd` per row) through the
    /// drafter's `fc` and layers and writes K/V of rows at `positions` to `cache`.
    pub fn ingest(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        cache: &mut SeqCache,
        feats: &[f32],
        positions: &[usize],
    ) -> Result<(), String> {
        self.ingest_many(
            backend,
            alloc,
            &mut [cache],
            None,
            feats,
            &[positions],
            true,
        )
    }

    /// [`Self::ingest`] for several sequences in one graph (#521): `feats` holds their rows in
    /// order, `positions[s]` sequence `s`'s positions (at most a window each). One sequence is
    /// exactly [`Self::ingest`]'s graph. With `pool`, when the caches are all its slots, several
    /// sequences' K/V go to it in one write per layer and kind, the same values to the same
    /// cells.
    ///
    /// Without `wait` the graph may still run on the backend's stream when this returns: the
    /// next graph on that stream runs after it, and `alloc`'s buffer stays its own until the
    /// backend is synchronized (a graph built meanwhile takes another allocator).
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_many(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        caches: &mut [&mut SeqCache],
        pool: Option<&CachePool>,
        feats: &[f32],
        positions: &[&[usize]],
        wait: bool,
    ) -> Result<(), String> {
        let c = &self.cfg;
        assert_eq!(caches.len(), positions.len());
        let n: usize = positions.iter().map(|p| p.len()).sum();
        let n_in = feats.len() / n.max(1);
        assert!(n > 0 && feats.len() == n * n_in);
        for (cache, p) in caches.iter().zip(positions) {
            assert!(!p.is_empty() && p.len() <= cache.window());
        }
        let n_seq = caches.len();
        let size = GRAPH_SIZE + GRAPH_PER_SEQ * (n_seq - 1);
        let ctx = Context::new(size, size);
        let input = |ty, ne: &[i64], name: &str| ctx.new_tensor(ty, ne).named(name).as_input();
        let x_in = input(types::F32, &[n_in as i64, n as i64], "features");
        let pos = input(types::I32, &[n as i64], "pos");
        // A sequence without a slot (every slot taken) has its cache apart: one write each.
        let pool = pool.filter(|_| n_seq > 1 && caches.iter().all(|c| c.slot().is_some()));
        // Pooled: every row's cell in the pool. Else each sequence's cells in its cache.
        let pool_cells = pool.map(|_| input(types::I32, &[n as i64], "pool_cells"));
        let cells: Vec<_> = positions
            .iter()
            .filter(|_| pool.is_none())
            .map(|p| input(types::I32, &[p.len() as i64], "cells"))
            .collect();
        let mut graph = ctx.new_graph(size);

        let g = mul_mat(&ctx, self.w("fc.weight"), x_in);
        let g = ctx.mul(ctx.rms_norm(g, c.rms_eps), self.w("enc.output_norm.weight"));
        for il in 0..c.n_layer {
            let (k, v) = self.kv(&ctx, il, g, pos);
            let row = (c.head_dim * c.n_head_kv) as i64;
            for (kind, x) in [("k", k), ("v", v)] {
                let x = ctx.reshape_2d(x, row, n as i64);
                if let (Some(pool), Some(pc)) = (pool, pool_cells) {
                    graph.expand(ctx.set_rows(pool.tensor(kind, il), x, pc));
                    continue;
                }
                let mut at = 0;
                for ((cache, p), &cells) in caches.iter().zip(positions).zip(&cells) {
                    let rows = if n_seq == 1 {
                        x
                    } else {
                        ctx.view_2d(x, [row, p.len() as i64], x.nb()[1], at * x.nb()[1])
                    };
                    graph.expand(ctx.set_rows(cache.tensor(kind, il), rows, cells));
                    at += p.len();
                }
            }
        }
        let allocator = alloc.take().unwrap_or_else(|| drafter_allocator(backend));
        ctx.alloc_graph(&graph, allocator)
            .map_err(|e| e.to_string())?;
        x_in.set_f32(feats);
        let all: Vec<i32> = positions
            .iter()
            .flat_map(|p| p.iter())
            .map(|&p| p as i32)
            .collect();
        pos.set_i32(&all);
        if let (Some(pool), Some(pc)) = (pool, pool_cells) {
            let at: Vec<i32> = caches
                .iter()
                .zip(positions)
                .flat_map(|(cache, p)| {
                    let base = cache.slot().expect("checked above") * pool.cells();
                    p.iter().map(move |&p| (base + cache.cell(p)) as i32)
                })
                .collect();
            pc.set_i32(&at);
        }
        for ((cache, p), cells) in caches.iter().zip(positions).zip(&cells) {
            cells.set_i32(&p.iter().map(|&p| cache.cell(p) as i32).collect::<Vec<_>>());
        }
        if wait {
            backend.compute(&graph)
        } else {
            backend.compute_async(&graph)
        }
        .map_err(|e| e.to_string())?;
        *alloc = ctx.into_allocator();
        for (cache, p) in caches.iter_mut().zip(positions) {
            for &p in p.iter() {
                cache.store(p);
            }
        }
        Ok(())
    }

    /// Runs the noise block of `n` positions (`last` at `anchor_pos`, then masks) and returns
    /// the output for block positions `first_row..n`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        cache: &SeqCache,
        last: u32,
        anchor_pos: usize,
        n: usize,
        want_logits: bool,
        want_hidden: bool,
    ) -> Result<Rows, String> {
        let one = [BlockSeq {
            cache,
            last,
            anchor_pos,
        }];
        let mut rows = self.draft_many(backend, alloc, &one, None, n, want_logits, want_hidden)?;
        Ok(rows.pop().expect("one sequence"))
    }

    /// [`Self::draft`] for several sequences in one pass, `n` block positions each (#521): the
    /// matmuls read the weights once for all of them; the attention is per sequence (each reads
    /// its own cache). One sequence is exactly [`Self::draft`]'s graph. Returns one [`Rows`] per
    /// sequence, in order.
    ///
    /// With `pool` (`--spec-draft-batched-attn`), when every cache is one of its slots, each
    /// layer's attention is one flash attention over all the sequences ([`Pooled`]) instead of
    /// a few small kernels per sequence; its sums round differently (drafts only).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_many(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        seqs: &[BlockSeq<'_>],
        pool: Option<&CachePool>,
        n: usize,
        want_logits: bool,
        want_hidden: bool,
    ) -> Result<Vec<Rows>, String> {
        let c = &self.cfg;
        let first = self.first_row;
        let n_seq = seqs.len();
        assert!(n_seq > 0);
        for s in seqs {
            assert!((first + 1..=s.cache.scratch()).contains(&n));
        }
        let total = n * n_seq;
        let size = GRAPH_SIZE + GRAPH_PER_SEQ * (n_seq - 1);
        let ctx = Context::new(size, size);
        let input = |ty, ne: &[i64], name: &str| ctx.new_tensor(ty, ne).named(name).as_input();
        let tokens = input(types::I32, &[total as i64], "tokens");
        let pos = input(types::I32, &[total as i64], "pos");
        let is_swa = |il: usize| c.sliding_window.as_ref().is_some_and(|(_, p)| p[il]);
        let any_full = (0..c.n_layer).any(|il| !is_swa(il));
        let any_swa = (0..c.n_layer).any(is_swa);
        let pooled = pool
            .filter(|_| n_seq > 1 && seqs.iter().all(|s| s.cache.slot().is_some()))
            .map(|pool| Pooled::new(&ctx, pool, seqs, n, any_full, any_swa));
        // Every sequence's block goes to the same scratch cells of its own cache.
        let cells = pooled
            .is_none()
            .then(|| input(types::I32, &[n as i64], "cells"));
        // A mask that no layer uses is not part of the graph and has no memory.
        let masks: Vec<_> = seqs
            .iter()
            .filter(|_| pooled.is_none())
            .map(|s| {
                let n_att = s.cache.used_cells() as i64;
                (
                    any_full.then(|| input(types::F32, &[n_att, n as i64], "mask")),
                    any_swa.then(|| input(types::F32, &[n_att, n as i64], "mask_swa")),
                )
            })
            .collect();
        let mut graph = ctx.new_graph(size);

        let (d, nh, kvh) = (c.head_dim as i64, c.n_head as i64, c.n_head_kv as i64);
        let row = d * kvh;
        let nb = n as i64;
        let scale = c.attn_scale.unwrap_or(1.0 / (c.head_dim as f32).sqrt());
        let mut h = ctx.get_rows(self.embeddings(), tokens);
        if c.embedding_scale != 1.0 {
            h = ctx.scale(h, c.embedding_scale);
        }
        for il in 0..c.n_layer {
            let x = ctx.mul(ctx.rms_norm(h, c.rms_eps), self.lw(il, "attn_norm.weight"));
            let (x, attn_dynamic) = self.conv_in(&ctx, il, "attn", x, nb);
            let q = ctx.reshape_3d(
                mul_mat(&ctx, self.lw(il, "attn_q.weight"), x),
                d,
                nh,
                total as i64,
            );
            let q = ctx.mul(
                ctx.rms_norm(q, c.rms_eps),
                self.lw(il, "attn_q_norm.weight"),
            );
            let q_all = rope(&ctx, q, pos, c.n_rot, c.rope_base);
            let (k_all, v_all) = self.kv(&ctx, il, x, pos);
            let mut out = None;
            // Several sequences: each one's attention output is copied into its columns of one
            // tensor (a chain of concats copied everything before it again, ~0.9 ms a step at 16
            // sequences). The values are the same.
            let out_all = (n_seq > 1 && pooled.is_none())
                .then(|| ctx.new_tensor(types::F32, &[d * nh, total as i64]));
            if let Some(pa) = &pooled {
                let mask = if is_swa(il) {
                    pa.mask_swa
                } else {
                    pa.mask_full
                };
                let o = self.pooled_attention(&ctx, il, pa, [q_all, k_all, v_all], mask, scale);
                out = Some(o);
            }
            for (s, seq) in seqs.iter().enumerate().filter(|_| pooled.is_none()) {
                let cells = cells.expect("made above");
                let cache = seq.cache;
                let n_att = cache.used_cells();
                let (q, k, v) = if n_seq == 1 {
                    (q_all, k_all, v_all)
                } else {
                    (
                        block_of(&ctx, q_all, n, s),
                        block_of(&ctx, k_all, n, s),
                        block_of(&ctx, v_all, n, s),
                    )
                };
                // The block's K/V go to the scratch cells; the attention below reads them there.
                for (kind, t) in [("k", k), ("v", v)] {
                    let t = ctx.reshape_2d(t, row, nb);
                    graph.expand(ctx.set_rows(cache.tensor(kind, il), t, cells));
                }
                let (ck, cv) = (cache.tensor("k", il), cache.tensor("v", il));
                let es = 2usize; // F16
                let ka = ctx.view_3d(
                    ck,
                    [d, kvh, n_att as i64],
                    [d as usize * es, row as usize * es],
                    0,
                );
                let ka = ctx.permute(ka, [0, 2, 1, 3]); // [d, n_att, kvh]
                let va = ctx.view_2d(cv, [row, n_att as i64], row as usize * es, 0);
                let va = ctx.cont(ctx.transpose(va)); // [n_att, row]
                let va = ctx.reshape_3d(va, n_att as i64, d, kvh);
                let q = ctx.permute(q, [0, 2, 1, 3]); // [d, n, nh]
                let kq = ctx.mul_mat(ka, q); // [n_att, n, nh]
                let (mask_full, mask_swa) = masks[s];
                let mask = if is_swa(il) { mask_swa } else { mask_full }.expect("made above");
                let kq = ctx.soft_max_ext(kq, mask, scale, 0.0);
                if let Some(sinks) = Self::sinks_of(&self.store, il) {
                    // Each head has one more logit in the softmax, a key with no value.
                    let sinks = ctx.cast(self.w(&sinks), types::F32);
                    // SAFETY: `kq` is a soft_max node of this context, `sinks` an F32 `[n_head]`.
                    unsafe { ffi::ggml_soft_max_add_sinks(kq.raw(), sinks.raw()) };
                }
                let o = ctx.mul_mat(va, kq); // [d, n, nh]
                let o = ctx.permute(o, [0, 2, 1, 3]);
                match out_all {
                    Some(all) => {
                        let row = all.nb()[1];
                        let cols = ctx.view_2d(all, [d * nh, nb], row, s * n * row);
                        graph.expand(ctx.cpy(o, cols));
                    }
                    None => out = Some(ctx.cont_2d(o, d * nh, nb)),
                }
            }
            let o = out_all.or(out).expect("at least one sequence");
            let o = if c.value_scale != 1.0 {
                ctx.scale(o, c.value_scale)
            } else {
                o
            };
            let attn = mul_mat(&ctx, self.lw(il, "attn_output.weight"), o);
            let attn = self.conv_out(&ctx, il, "attn", attn, attn_dynamic, nb);
            let h_attn = ctx.add(attn, h);
            let f = ctx.mul(
                ctx.rms_norm(h_attn, c.rms_eps),
                self.lw(il, "ffn_norm.weight"),
            );
            let (f, ffn_dynamic) = self.conv_in(&ctx, il, "ffn", f, nb);
            let gu = mul_mats(
                &ctx,
                &[self.lw(il, "ffn_gate.weight"), self.lw(il, "ffn_up.weight")],
                f,
            );
            let act = ctx.mul(ctx.silu(gu[0]), gu[1]);
            let down = mul_mat(&ctx, self.lw(il, "ffn_down.weight"), act);
            let down = self.conv_out(&ctx, il, "ffn", down, ffn_dynamic, nb);
            h = ctx.add(down, h_attn);
        }
        let x = ctx.mul(ctx.rms_norm(h, c.rms_eps), self.w("output_norm.weight"));
        // Output rows: every sequence's block positions `first..n`, in order.
        let n_out = ((n - first) * n_seq) as i64;
        let es = x.element_size();
        let e = c.n_embd as i64;
        let x = if n_seq == 1 {
            ctx.view_2d(x, [e, n_out], c.n_embd * es, first * c.n_embd * es)
        } else {
            let x = ctx.reshape_3d(x, e, nb, n_seq as i64);
            let nbs = x.nb();
            let x = ctx.view_3d(
                x,
                [e, (n - first) as i64, n_seq as i64],
                [nbs[1], nbs[2]],
                first * nbs[1],
            );
            ctx.cont_2d(x, e, n_out)
        };
        let hidden = want_hidden.then(|| ctx.cont(x).as_output());
        let head = self.head();
        let logits = if head.ty() == types::I8 {
            crate::model::fp8::mul_mat(&ctx, head, x, self.head_scale())
        } else {
            let logits = mul_mat(&ctx, head, x);
            match self.head_scale() {
                Some(s) => ctx.mul(logits, s),
                None => logits,
            }
        };
        // DFlash 2: the head's top-k tokens of every position, their logits and the selector's
        // hidden projection; the selector itself runs on the host (`Selector::walk`).
        let selector = self.selector.as_ref().map(|s| {
            let top = ctx.top_k(logits, s.top_k as i32);
            let unary = ctx.get_rows(ctx.reshape_3d(logits, 1, self.n_vocab as i64, n_out), top);
            let proj = ctx.mul_mat(self.w("selector_hidden.weight"), x);
            (top.as_output(), unary.as_output(), proj.as_output())
        });
        let logits = logits.as_output();
        let argmax = selector
            .is_none()
            .then(|| crate::model::cluster_select::argmax(&ctx, logits).as_output());
        match selector {
            Some((top, unary, proj)) => {
                graph.expand(top);
                graph.expand(unary);
                graph.expand(proj);
            }
            None => graph.expand(argmax.expect("made above")),
        }
        if want_logits {
            graph.expand(logits);
        }
        if let Some(h) = hidden {
            graph.expand(h);
        }

        let allocator = alloc.take().unwrap_or_else(|| drafter_allocator(backend));
        ctx.alloc_graph(&graph, allocator)
            .map_err(|e| e.to_string())?;
        let mut ids = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        for s in seqs {
            ids.push(s.last as i32);
            ids.extend(std::iter::repeat_n(c.mask_token as i32, n - 1));
            positions.extend((0..n).map(|j| (s.anchor_pos + j) as i32));
        }
        tokens.set_i32(&ids);
        pos.set_i32(&positions);
        if let Some(cells) = cells {
            cells.set_i32(&(0..n as i32).collect::<Vec<_>>());
        }
        if let Some(pa) = &pooled {
            pa.set_inputs(self, seqs, n);
        }
        for (s, &(mask_full, mask_swa)) in seqs.iter().zip(&masks) {
            let n_att = s.cache.used_cells();
            if let Some(t) = mask_full {
                t.set_f32(&self.mask(s.cache, s.anchor_pos, n, n_att, None));
            }
            if let (Some(t), Some((w, _))) = (mask_swa, &c.sliding_window) {
                t.set_f32(&self.mask(s.cache, s.anchor_pos, n, n_att, Some(*w)));
            }
        }
        backend.compute(&graph).map_err(|e| e.to_string())?;
        let ids =
            |t: Tensor<'_>| -> Vec<u32> { t.to_i32().into_iter().map(|i| i as u32).collect() };
        let all = Rows {
            argmax: argmax.map_or_else(Vec::new, ids),
            logits: want_logits.then(|| logits.to_f32()),
            hidden: hidden.map(|h| h.to_f32()),
            selector: selector.map(|(top, unary, proj)| (ids(top), unary.to_f32(), proj.to_f32())),
        };
        *alloc = ctx.into_allocator();
        if n_seq == 1 {
            return Ok(vec![all]);
        }
        Ok((0..n_seq)
            .map(|s| Rows {
                argmax: part(&all.argmax, n_seq, s),
                logits: all.logits.as_deref().map(|l| part(l, n_seq, s)),
                hidden: all.hidden.as_deref().map(|h| part(h, n_seq, s)),
                selector: all
                    .selector
                    .as_ref()
                    .map(|(t, u, p)| (part(t, n_seq, s), part(u, n_seq, s), part(p, n_seq, s))),
            })
            .collect())
    }

    /// Whether the drafter has a Markov head (DSpark).
    pub fn has_markov(&self) -> bool {
        self.markov.is_some()
    }

    /// Whether the drafter has a confidence head.
    pub fn has_confidence(&self) -> bool {
        self.markov.as_ref().is_some_and(|m| m.conf.is_some())
    }

    /// The Markov head after `prev`: the bias to add to the logits of the draft position that
    /// follows token `prev` (over the draft vocabulary), and the row `markov_w1[prev]` (which
    /// the confidence head reads too).
    pub fn markov_step(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        prev: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let m = self.markov.as_ref().expect("a DSpark drafter");
        let bytes = crate::ggml::row_size(m.w1_type, m.rank);
        let at = prev as usize * bytes;
        let row = floats(m.w1_type, &m.w1[at..at + bytes], m.rank);
        let ctx = Context::new(64, 64);
        let input = ctx.new_tensor(types::F32, &[m.rank as i64]).as_input();
        let mut bias = ctx.mul_mat(self.w("markov_w2.weight"), input);
        if let Some(scale) = self.store.get("markov_w2.scale") {
            bias = ctx.mul(bias, scale);
        }
        let bias = bias.as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(bias);
        let allocator = alloc.take().unwrap_or_else(|| drafter_allocator(backend));
        ctx.alloc_graph(&graph, allocator)
            .map_err(|e| e.to_string())?;
        input.set_f32(&row);
        backend.compute(&graph).map_err(|e| e.to_string())?;
        let out = bias.to_f32();
        *alloc = ctx.into_allocator();
        Ok((out, row))
    }

    /// The confidence head's estimate that a draft is accepted, from the position's final-normed
    /// hidden state and the Markov row of its previous token.
    pub fn confidence(&self, hidden: &[f32], w1_row: &[f32]) -> f32 {
        let (w, b) = self
            .markov
            .as_ref()
            .and_then(|m| m.conf.as_ref())
            .expect("a confidence head");
        let x = hidden.iter().chain(w1_row);
        let z: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + b;
        1.0 / (1.0 + (-z).exp())
    }

    /// Layer `il`'s attention over the pool ([`Pooled`]): `[q, k, v]` are the pass's
    /// `[d, n_head(_kv), n_seq · n]` projections, `mask` the F16 mask of the layer's kind.
    /// Returns `[d · n_head, n_seq · n]`, as the per-sequence attentions' outputs side by side.
    fn pooled_attention<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        pa: &Pooled<'a>,
        [q, k, v]: [Tensor<'a>; 3],
        mask: Option<Tensor<'a>>,
        scale: f32,
    ) -> Tensor<'a> {
        let c = &self.cfg;
        let (d, nh, kvh) = (c.head_dim as i64, c.n_head as i64, c.n_head_kv as i64);
        let row = d * kvh;
        let total = q.ne()[2];
        let n_view = pa.n_view as i64;
        let n_seq = pa.o_idx.map_or(n_view, |o| o.ne()[0]);
        let n = total / n_seq;
        let es = 2usize; // F16
        let slot_bytes = pa.pool.cells() * row as usize * es;
        // The blocks' K/V go to their slots' scratch cells; the attention reads them there.
        let read = |kind: &str, t: Tensor<'a>| {
            let t = ctx.reshape_2d(t, row, total);
            let all = ctx.set_rows(pa.pool.tensor(kind, il), t, pa.block_cells);
            let t = ctx.view_4d(
                all,
                [d, kvh, pa.n_kv as i64, n_view],
                [d as usize * es, row as usize * es, slot_bytes],
                0,
            );
            ctx.permute(t, [0, 2, 1, 3]) // [d, n_kv, kvh, n_view]
        };
        let (k, v) = (read("k", k), read("v", v));
        let block = d * nh * n;
        let q = match pa.q_idx {
            Some(idx) => ctx.get_rows(ctx.reshape_2d(q, block, n_seq), idx),
            None => q,
        };
        let q = ctx.permute(ctx.reshape_4d(q, [d, nh, n, n_view]), [0, 2, 1, 3]); // [d, n, nh, n_view]
        let o = ctx.flash_attn_ext(q, k, v, mask.expect("made for this kind"), scale);
        if let Some(sinks) = Self::sinks_of(&self.store, il) {
            let sinks = ctx.cast(self.w(&sinks), types::F32);
            // SAFETY: `o` is a flash_attn_ext node of this context, `sinks` an F32 `[n_head]`.
            unsafe { ffi::ggml_flash_attn_ext_add_sinks(o.raw(), sinks.raw()) };
        }
        // [d, n_head, n, n_view]: a slot's block is a sequence's columns.
        let o = match pa.o_idx {
            Some(idx) => ctx.get_rows(ctx.reshape_2d(o, block, n_view), idx),
            None => o,
        };
        ctx.reshape_2d(o, d * nh, total)
    }

    /// The attention mask of the block queries `[n_att, n]` (0 or -inf). `n_att` may exceed the
    /// cache's cells (a pooled attention's padding).
    fn mask(
        &self,
        cache: &SeqCache,
        anchor_pos: usize,
        n: usize,
        n_att: usize,
        window: Option<usize>,
    ) -> Vec<f32> {
        let scratch = cache.scratch();
        let mut m = vec![f32::NEG_INFINITY; n_att * n];
        for q in 0..n {
            let qpos = anchor_pos + q;
            let row = &mut m[q * n_att..(q + 1) * n_att];
            // The block: all of it, or what is not after the query.
            let in_window = |kpos: usize| window.is_none_or(|w| qpos < kpos + w);
            for (s, cell) in row.iter_mut().enumerate().take(n) {
                let visible = (!self.cfg.causal || s <= q) && in_window(anchor_pos + s);
                if visible {
                    *cell = 0.0;
                }
            }
            // A pooled attention reads past the slot's ring (`n_att` above `scratch + window`):
            // those cells stay masked.
            for i in 0..(n_att - scratch).min(cache.window()) {
                if let Some(p) = cache.held(i)
                    && p < anchor_pos
                    && in_window(p)
                {
                    row[scratch + i] = 0.0;
                }
            }
        }
        m
    }
}
