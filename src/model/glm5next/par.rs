//! GLM-5.3-Flash over several GPUs, every GPU on every token (`--glm-parallel-decode`, #558):
//! tensor parallelism for attention and the dense / shared FFNs, expert parallelism for the
//! routed experts, instead of one pipeline stage at a time.
//!
//! Every GPU runs every layer ([`graph`]) and keeps the whole residual streams. Per layer:
//!
//! - **The hyper-connection mixers, the norms and the router** are mirrored: every GPU runs
//!   them over the same streams (its own copies, `<name>@m<d>` where it is not the layer's
//!   pipeline owner, whose originals they are), so every GPU computes the same bits. A mixer's
//!   input norm is nonlinear in the whole stream, so the streams must be whole before every
//!   sublayer: two exchanges per layer is the least there can be.
//! - **KDA by heads** (Megatron's column / row split): GPU `d` has its heads' rows of q, k, v,
//!   the gates' second products (`f_b`, `g_b`), beta, the convolutions, `dt` and `A`, and the
//!   matching columns of the output projection (`<name>@s<d>`, [`Cut`]); `f_a`, `g_a` and the
//!   head norm are mirrored. Its conv and delta-rule states are its heads' only (a stage's
//!   state pool built for [`shard_hparams`]). Its output projection gives a partial sum of the
//!   layer's output.
//! - **MLA by heads**: the query and latent down-projections, their norms and the indexer are
//!   mirrored, and every GPU caches the latents and indexer rows of every layer (one latent
//!   row serves all heads: the cache is mirrored, not split). GPU `d` has its heads' `q_b`,
//!   `k_b`, `v_b` and output projection columns; the indexer's selection is the same on every
//!   GPU.
//! - **The dense FFN and the shared expert** split by their hidden rows: gate and up by rows,
//!   down by columns (in units of [`FFN_UNIT`], whole quant blocks).
//! - **The routed experts** expert-parallel: GPU `d` owns the experts `bounds[d]..bounds[d + 1]`
//!   of every MoE layer ([`expert_bounds`]) and caches and computes only those (its own entry
//!   of the expert cache per layer, [`ep_layers`]); its share of the routed sum and of the
//!   shared expert is one partial sum.
//! - **One all-reduce per sublayer** ([`exchange::reduce`]): the GPUs' partial sums added in
//!   GPU order, the same bits on every GPU, then the mixer's post step on every GPU. Two per
//!   layer, plus the embedding's broadcast and the LM head's gather.
//! - **The LM head split by rows** (vocabulary): each GPU computes its share of the logits
//!   (`output.weight@s<d>`; the last GPU reads its rows of its full copy), gathered on the
//!   last GPU, which takes the argmax. These logits are bitwise the unsplit head's.
//!
//! Every step runs this way, prompts included (the states are split by heads, which the
//! pipeline cannot use): up to [`MAX_TOKENS`] tokens a step, and a prompt borrowing the expert
//! cache up to [`step_tokens`] (`--glm-prompt-step`, #686). The partial sums are added in
//! another order than one GPU's matmul adds them, so the output is not bitwise the pipeline's.
//! The originals of the split weights are not loaded ([`ParPlan::tensors`]).

pub mod assign;
pub mod balance;
pub mod exchange;
mod graph;
pub mod mtp;
pub mod prefetch;
pub mod split;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use super::Glm5Params;
use crate::gguf::{GgufFile, Recipe, TensorInfo};
use crate::model::core::Hparams;

/// The suffix of a shard's name: `<name>@s<device>`.
const SHARD: &str = "@s";
/// The suffix of a mirror's name: `<name>@m<device>`.
const MIRROR: &str = "@m";

/// GPU `d`'s shard of weight `name`.
pub fn shard_name(name: &str, d: usize) -> String {
    format!("{name}{SHARD}{d}")
}

/// GPU `d`'s mirror of weight `name`.
pub fn mirror_name(name: &str, d: usize) -> String {
    format!("{name}{MIRROR}{d}")
}

/// The GPU whose memory holds a shard or mirror (by its name), if `name` is one.
pub fn copy_device(name: &str) -> Option<usize> {
    let at = name.rfind('@')?;
    let rest = &name[at..];
    rest.strip_prefix(SHARD)
        .or_else(|| rest.strip_prefix(MIRROR))?
        .parse()
        .ok()
}

/// GPU `d`'s rows of `total` split over `n_dev` GPUs: even shares, the first `total % n_dev`
/// GPUs one more.
pub fn rows(total: i64, n_dev: usize, d: usize) -> Range<i64> {
    let n = n_dev as i64;
    let (q, r) = (total / n, total % n);
    let start = d as i64 * q + (d as i64).min(r);
    start..start + q + i64::from((d as i64) < r)
}

/// The experts each GPU owns: GPU `d` the ids `bounds[d]..bounds[d + 1]`, an even split in
/// GPU order (GPUs of one NUMA node are next to each other in CUDA order on an AC922, so each
/// socket's GPUs own one contiguous half).
pub fn expert_bounds(n_expert: usize, n_dev: usize) -> Vec<u32> {
    (0..=n_dev).map(|d| (d * n_expert / n_dev) as u32).collect()
}

/// The experts each GPU owns with GPU `d` taking `shares[d]` parts of them
/// (`--glm-expert-shares`, #714): bounds at the rounded cumulative shares, every GPU at least
/// one expert. Empty `shares` (or one per GPU missing): [`expert_bounds`]. The last GPU also
/// runs the MTP head over all the experts and keeps the drafts' full LM head, so fewer experts
/// of its own leave it less of the step's routed work and more cache per expert.
pub fn expert_bounds_shared(n_expert: usize, n_dev: usize, shares: &[u32]) -> Vec<u32> {
    let total: u64 = shares.iter().map(|&s| u64::from(s)).sum();
    if shares.len() != n_dev || total == 0 || n_expert < n_dev {
        return expert_bounds(n_expert, n_dev);
    }
    let mut bounds = vec![0u32];
    let mut cum = 0u64;
    for (d, &s) in shares.iter().enumerate() {
        cum += u64::from(s);
        let at = ((cum * n_expert as u64 + total / 2) / total) as usize;
        // At least one expert for this GPU and for each after it.
        let lo = bounds[d] as usize + 1;
        let hi = n_expert - (n_dev - 1 - d);
        bounds.push(at.clamp(lo, hi) as u32);
    }
    bounds
}

/// The GPU owning expert `e` under `bounds`.
pub fn expert_owner(bounds: &[u32], e: u32) -> usize {
    bounds[1..]
        .iter()
        .position(|&b| e < b)
        .unwrap_or(bounds.len() - 2)
}

/// The FFNs' split unit (their hidden rows, the down projection's columns): one K-quant
/// block, so every column shard is whole blocks.
pub const FFN_UNIT: i64 = 256;

/// How a layer's weight splits over the GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    /// Every GPU reads all of it: the owner its original, the others a copy each.
    Mirror,
    /// Split along its last dimension (a matrix's rows, a batched matmul's matrices, a
    /// vector's values) in units of `unit`: one contiguous range of the file per GPU.
    Outer { unit: i64 },
    /// Split along its first dimension (a matrix's columns, the input of an output
    /// projection) in units of `unit`: a slice of every row.
    Cols { unit: i64 },
}

/// The weights of decoder layer `il` and how each splits.
pub fn layer_cuts(p: &Glm5Params, il: usize) -> Vec<(String, Cut)> {
    let mut v: Vec<(&str, Cut)> = Vec::new();
    for w in [
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_attn_base.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
        "hc_ffn_base.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
    ] {
        v.push((w, Cut::Mirror));
    }
    if p.mla[il] {
        for w in [
            "attn_q_a.weight",
            "attn_q_a_norm.weight",
            "attn_kv_a_mqa.weight",
            "attn_kv_a_norm.weight",
            "indexer.attn_k.weight",
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
            "indexer.attn_q_b.weight",
            "indexer.proj.weight",
            "indexer_compressor_gate.weight",
            "indexer_compressor_ape.weight",
        ] {
            v.push((w, Cut::Mirror));
        }
        let dh = p.head_mla;
        v.push(("attn_q_b.weight", Cut::Outer { unit: dh }));
        v.push(("attn_k_b.weight", Cut::Outer { unit: 1 }));
        v.push(("attn_v_b.weight", Cut::Outer { unit: 1 }));
        v.push(("attn_output.weight", Cut::Cols { unit: dh }));
    } else {
        for w in ["ssm_f_a.weight", "ssm_g_a.weight", "ssm_norm.weight"] {
            v.push((w, Cut::Mirror));
        }
        let d = p.kda_head;
        for w in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "ssm_conv1d_q.weight",
            "ssm_conv1d_k.weight",
            "ssm_conv1d_v.weight",
            "ssm_f_b.weight",
            "ssm_g_b.weight",
            "ssm_dt.bias",
        ] {
            v.push((w, Cut::Outer { unit: d }));
        }
        v.push(("ssm_beta.weight", Cut::Outer { unit: 1 }));
        v.push(("ssm_a", Cut::Outer { unit: 1 }));
        v.push(("attn_output.weight", Cut::Cols { unit: d }));
    }
    let ffn = if il < p.n_dense {
        ""
    } else {
        v.push(("ffn_gate_inp.weight", Cut::Mirror));
        v.push(("exp_probs_b.bias", Cut::Mirror));
        "_shexp"
    };
    let mut out: Vec<(String, Cut)> = v
        .into_iter()
        .map(|(w, c)| (format!("blk.{il}.{w}"), c))
        .collect();
    for (w, c) in [
        ("gate", Cut::Outer { unit: FFN_UNIT }),
        ("up", Cut::Outer { unit: FFN_UNIT }),
        ("down", Cut::Cols { unit: FFN_UNIT }),
    ] {
        out.push((format!("blk.{il}.ffn_{w}{ffn}.weight"), c));
    }
    out
}

/// The LM head, split by rows (vocabulary).
pub const HEAD: &str = "output.weight";
/// The tensors past the layers every GPU reads whole.
pub const GLOBAL_MIRRORED: &[&str] = &["output_norm.weight"];

/// GPU `d`'s share (`d` of `n_dev`) of tensor `t` cut by `cut` ([`Cut::Outer`] or
/// [`Cut::Cols`]), named [`shard_name`]: its info and, for a column slice, how to read it (a
/// slice of every row of the file's range). Errors name a tensor that does not split so.
pub fn shard_of(
    t: &TensorInfo,
    cut: Cut,
    n_dev: usize,
    d: usize,
    path: &std::path::Path,
) -> Result<(TensorInfo, Option<Recipe>), String> {
    let dims = t.shape.len();
    let name = shard_name(&t.name, d);
    match cut {
        Cut::Mirror => Err(format!("`{}`: a mirror is no shard", t.name)),
        Cut::Outer { unit } => {
            let last = t.shape[dims - 1] as i64;
            if last % unit != 0 {
                return Err(format!("`{}`: {last} not in units of {unit}", t.name));
            }
            if !t.size.is_multiple_of(last as u64) {
                return Err(format!("`{}`: uneven slices", t.name));
            }
            let slice = t.size / last as u64;
            let r = rows(last / unit, n_dev, d);
            let (r0, n) = ((r.start * unit) as u64, ((r.end - r.start) * unit) as u64);
            let mut shape = t.shape.clone();
            shape[dims - 1] = n;
            Ok((
                TensorInfo {
                    name,
                    ggml_type: t.ggml_type,
                    shape,
                    file_index: t.file_index,
                    offset: t.offset + r0 * slice,
                    size: n * slice,
                },
                None,
            ))
        }
        Cut::Cols { unit } => {
            if dims != 2 {
                return Err(format!("`{}`: {dims} dims, not a matrix", t.name));
            }
            let (k, m) = (t.shape[0] as i64, t.shape[1]);
            if k % unit != 0 {
                return Err(format!("`{}`: {k} columns not in units of {unit}", t.name));
            }
            let r = rows(k / unit, n_dev, d);
            let (c0, len) = (r.start * unit, (r.end - r.start) * unit);
            let blck = crate::ggml::block_size(t.ggml_type) as i64;
            if c0 % blck != 0 || len % blck != 0 {
                return Err(format!(
                    "`{}`: columns {c0}..{} cut its blocks of {blck}",
                    t.name,
                    c0 + len
                ));
            }
            let row = crate::ggml::row_size(t.ggml_type, k as usize);
            let at = crate::ggml::row_size(t.ggml_type, c0 as usize);
            let take = crate::ggml::row_size(t.ggml_type, len as usize);
            if t.size != (row as u64) * m {
                return Err(format!("`{}`: rows of uneven size", t.name));
            }
            let (path, offset, size) = (path.to_path_buf(), t.offset, t.size);
            let read = move || -> Result<Vec<u8>, String> {
                use std::os::unix::fs::FileExt;
                let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                let mut whole = vec![0u8; size as usize];
                f.read_exact_at(&mut whole, offset)
                    .map_err(|e| e.to_string())?;
                let mut out = Vec::with_capacity(m as usize * take);
                for r in whole.chunks_exact(row) {
                    out.extend_from_slice(&r[at..at + take]);
                }
                Ok(out)
            };
            Ok((
                TensorInfo {
                    name,
                    ggml_type: t.ggml_type,
                    shape: vec![len as u64, m],
                    file_index: t.file_index,
                    offset: t.offset,
                    size: m * take as u64,
                },
                Some(Recipe::Computed(Arc::new(read))),
            ))
        }
    }
}

/// The hyperparameters of GPU `d`'s share of the model (`n_dev` GPUs): its KDA heads (its
/// state pool's sizes). Everything else as the model's.
pub fn shard_hparams(hp: &Hparams, n_dev: usize) -> Hparams {
    let mut s = hp.clone();
    let n = n_dev as i64;
    s.ssm_d_inner /= n;
    s.ssm_n_group /= n;
    s.ssm_dt_rank /= n;
    s
}

/// What the GPUs load for tensor parallelism beside (or instead of) the model's tensors.
#[derive(Default)]
pub struct TpTensors {
    /// The shards and mirrors, each placed on its GPU ([`copy_device`]).
    pub extra: Vec<TensorInfo>,
    /// How to read the column shards.
    pub recipes: HashMap<String, Recipe>,
    /// The split weights' originals, left out of the load.
    pub skip: Vec<String>,
}

/// Who does what in a parallel step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParPlan {
    pub n_dev: usize,
    /// Per decoder layer: its pipeline owner (the GPU its originals load on).
    pub owner: Vec<usize>,
    /// [`expert_bounds`].
    pub bounds: Vec<u32>,
    /// The file's expert behind each of the step's expert ids ([`assign::expert_order`]):
    /// `None` for the file's own order.
    pub order: Option<Vec<u32>>,
    /// The MTP head's passes run on every GPU (`--glm-parallel-mtp`, #717, [`mtp`]): its
    /// routed experts expert-parallel, its shared expert by rows, its draft logits by rows of
    /// the LM head (whose full copy on the last GPU is not loaded).
    pub mtp: bool,
    /// A decode step's routed experts balanced within each GPU pair (`--glm-pair-balance`,
    /// #729, [`balance`]).
    pub pair_balance: bool,
    /// Every MoE layer also logs the next layer's router applied to its own input, for the
    /// prefetch predictors' report (`--glm-prefetch-stats`, #745,
    /// [`crate::residency::ep_prefetch`]).
    pub predict_next: bool,
    /// With [`Self::pair_balance`]: over every GPU, uncached experts also moving across the
    /// pairs (`--glm-cross-balance`, #731, [`balance::ep_assign_cross`]).
    pub cross_balance: bool,
    /// The next layer's guessed uncached experts each GPU copies into VRAM ahead, at most this
    /// many per layer, 0 none (`--glm-prefetch`, #747, [`prefetch`]).
    pub prefetch: usize,
    /// A decode step's uncached routed experts split over both GPUs of their pair by rows
    /// (`--glm-pair-split`, #735, [`split`]); ahead of [`Self::pair_balance`].
    pub pair_split: bool,
}

impl ParPlan {
    /// The plan for layers owned as `owner` says, over `n_dev` GPUs, `n_expert` experts.
    pub fn new(n_dev: usize, owner: Vec<usize>, n_expert: usize) -> Self {
        assert!(n_dev > 0 && owner.iter().all(|&d| d < n_dev));
        Self {
            n_dev,
            owner,
            bounds: expert_bounds(n_expert, n_dev),
            order: None,
            mtp: false,
            pair_balance: false,
            predict_next: false,
            cross_balance: false,
            prefetch: 0,
            pair_split: false,
        }
    }

    /// The plan with a decode step's routed experts balanced within each GPU pair or not
    /// (`--glm-pair-balance`, [`balance`]).
    pub fn balance_pairs(mut self, on: bool) -> Self {
        self.pair_balance = on;
        self
    }

    /// The plan with every MoE layer logging the next layer's router guess or not
    /// (`--glm-prefetch-stats`, #745).
    pub fn predict_next(mut self, on: bool) -> Self {
        self.predict_next = on;
        self
    }

    /// The plan with a decode step's routed experts balanced over every GPU or not
    /// (`--glm-cross-balance`, [`balance::ep_assign_cross`]); on implies [`Self::pair_balance`]'s
    /// machinery (the owners' cached experts on every device).
    pub fn balance_cross(mut self, on: bool) -> Self {
        self.cross_balance = on;
        self.pair_balance |= on;
        self
    }

    /// The plan with each GPU copying up to `n` guessed uncached experts of the next layer
    /// into VRAM ahead per layer, 0 none (`--glm-prefetch`, #747).
    pub fn prefetch(mut self, n: usize) -> Self {
        self.prefetch = n;
        self
    }

    /// The plan with a decode step's uncached routed experts split over both GPUs of their
    /// pair by rows or not (`--glm-pair-split`, [`split`]).
    pub fn split_pairs(mut self, on: bool) -> Self {
        self.pair_split = on;
        self
    }

    /// The plan with GPU `d` owning `shares[d]` parts of the routed experts
    /// ([`expert_bounds_shared`]; empty: the even split).
    pub fn share_experts(mut self, shares: &[u32]) -> Self {
        let n_expert = *self.bounds.last().expect("bounds") as usize;
        self.bounds = expert_bounds_shared(n_expert, self.n_dev, shares);
        self
    }

    /// The plan with the MTP head's passes on every GPU or not (`--glm-parallel-mtp`, [`mtp`]).
    pub fn split_mtp(mut self, on: bool) -> Self {
        self.mtp = on;
        self
    }

    /// The plan with the routed experts split over the GPUs as `split` says
    /// (`--glm-expert-split`, [`assign`]).
    pub fn split_experts(mut self, split: assign::ExpertSplit) -> Self {
        let n_expert = *self.bounds.last().expect("bounds") as usize;
        self.order = assign::expert_order(split, n_expert, self.n_dev);
        self
    }

    /// The last GPU: it takes the logits' argmax.
    pub fn last(&self) -> usize {
        self.n_dev - 1
    }

    /// The experts GPU `d` owns.
    pub fn owned(&self, d: usize) -> Range<u32> {
        self.bounds[d]..self.bounds[d + 1]
    }

    /// The suffix of GPU `d`'s mirrored weights of layer `il`: none on the layer's owner.
    pub fn mirror_sfx(&self, il: usize, d: usize) -> String {
        if self.owner[il] == d {
            String::new()
        } else {
            mirror_name("", d)
        }
    }

    /// Whether the model splits evenly over the GPUs: the KDA heads (each GPU's state pool is
    /// one size) and the MLA heads.
    pub fn check(&self, hp: &Hparams, p: &Glm5Params) -> Result<(), String> {
        let n = self.n_dev as i64;
        let kda = hp.ssm_d_inner / p.kda_head;
        if kda % n != 0 || hp.n_head % n != 0 {
            return Err(format!(
                "{kda} KDA and {} MLA heads do not split over {n} GPUs",
                hp.n_head
            ));
        }
        Ok(())
    }

    /// The tensors the GPUs load besides the model's: per decoder layer, every GPU's shards of
    /// the split weights (whose originals stay unloaded) and mirrors of the mirrored ones
    /// (none for the layer's owner); the head's row shards (the last GPU reads its rows of its
    /// full copy) and the output norm's mirrors. With renumbered experts ([`Self::order`]),
    /// gather recipes that read every decoder MoE layer's [`assign::ROUTED`] tensors (and the
    /// router's mirrors) in that order. Errors name a tensor that does not split.
    pub fn tensors(&self, g: &GgufFile, p: &Glm5Params) -> Result<TpTensors, String> {
        let mut out = TpTensors::default();
        let tensor = |name: &str| -> Result<TensorInfo, String> {
            if g.recipe(name).is_some() {
                return Err(format!("`{name}` is synthesized, not a range of the file"));
            }
            let base = name.strip_suffix(".weight").unwrap_or(name);
            if g.tensor(&format!("{base}.scale")).is_some() {
                return Err(format!("`{name}` has a checkpoint scale"));
            }
            g.tensor(name).ok_or_else(|| format!("no tensor `{name}`"))
        };
        let path = |t: &TensorInfo| -> Result<std::path::PathBuf, String> {
            g.shard_paths()
                .nth(t.file_index)
                .map(|p| p.to_path_buf())
                .ok_or_else(|| format!("`{}`: no file {}", t.name, t.file_index))
        };
        let mirror = |out: &mut TpTensors, name: &str, d: usize| -> Result<(), String> {
            let t = tensor(name)?;
            out.extra.push(TensorInfo {
                name: mirror_name(name, d),
                ..t
            });
            Ok(())
        };
        let n_layer = self.owner.len();
        // With the MTP head on every GPU, its layer's FFN after the decoder layers (owned by
        // the last GPU, which runs the rest of the head).
        let mtp = self.mtp.then_some((n_layer, self.last()));
        for (il, own) in self.owner.iter().copied().enumerate().chain(mtp) {
            let cuts = if il < n_layer {
                layer_cuts(p, il)
            } else {
                mtp::mtp_cuts(il)
            };
            for (name, cut) in cuts {
                if cut == Cut::Mirror {
                    for d in (0..self.n_dev).filter(|&d| d != own) {
                        mirror(&mut out, &name, d)?;
                    }
                    continue;
                }
                let t = tensor(&name)?;
                let file = path(&t)?;
                for d in 0..self.n_dev {
                    let (info, recipe) = shard_of(&t, cut, self.n_dev, d, &file)?;
                    if let Some(r) = recipe {
                        out.recipes.insert(info.name.clone(), r);
                    }
                    out.extra.push(info);
                }
                out.skip.push(name);
            }
            // Renumbered experts: the router's rows (and its mirrors), its bias and the
            // experts' matrices read in the plan's order.
            if let Some(order) = self.order.as_deref()
                && il >= p.n_dense
                && il < n_layer
            {
                assign::check_layer(g, il)?;
                for w in assign::ROUTED {
                    let name = format!("blk.{il}.{w}");
                    let recipe = assign::permuted(&tensor(&name)?, order)?;
                    // The router is mirrored ([`layer_cuts`]); the experts are not.
                    let mirrored = !w.contains("_exps");
                    for d in (0..self.n_dev).filter(|&d| mirrored && d != own) {
                        out.recipes.insert(mirror_name(&name, d), recipe.clone());
                    }
                    out.recipes.insert(name, recipe);
                }
            }
        }
        let head = tensor(HEAD)?;
        let file = path(&head)?;
        // With the MTP head on every GPU, its drafts read the head by rows too: the last GPU
        // its own shard, the full copy not loaded.
        for d in (0..self.n_dev).filter(|&d| self.mtp || d != self.last()) {
            let (info, _) = shard_of(&head, Cut::Outer { unit: 1 }, self.n_dev, d, &file)?;
            out.extra.push(info);
        }
        if self.mtp {
            out.skip.push(HEAD.to_string());
        }
        for d in (0..self.n_dev).filter(|&d| d != self.last()) {
            for name in GLOBAL_MIRRORED {
                mirror(&mut out, name, d)?;
            }
        }
        Ok(out)
    }
}

/// The most tokens a parallel step takes while no prompt borrows the expert cache (decode and
/// verify steps, a short prompt's chunks; a prompt's chunk and the decoding sequences' tokens
/// together).
pub const MAX_TOKENS: usize = 512;

/// The default tokens of a prompt's parallel step while it borrows the expert cache
/// (`--glm-prompt-step`, #686): each GPU stages its uncached experts of a layer once per step,
/// and an expert's matmul reads its weights once for all of the step's tokens routed to it, so
/// a 2048-token step reads them once per 2048 tokens instead of four times.
pub const PROMPT_TOKENS: usize = 2048;

/// The most tokens `--glm-prompt-step` takes: the exchange's host slots grow by 64 KiB a token
/// (GLM-5.3-Flash's four residual streams), a GPU's inbox has two per writer (4 GiB a GPU at
/// 8192 on four GPUs).
pub const MAX_PROMPT_TOKENS: usize = 8192;

/// The most tokens a parallel step takes with prompt steps of `prompt_step` tokens
/// (`--glm-prompt-step`): at least [`MAX_TOKENS`], at most [`MAX_PROMPT_TOKENS`]. The exchange's
/// slots hold this many tokens' residual streams.
pub fn step_tokens(prompt_step: usize) -> usize {
    prompt_step.clamp(MAX_TOKENS, MAX_PROMPT_TOKENS)
}

/// The expert cache's entry of model layer `il`'s copy on GPU `c` (not its owner `owner`), in
/// a cache of `n_entries` model layers (the MTP head's included): the copies follow the model's
/// layers, `n_dev - 1` per layer.
pub fn copy_entry(n_entries: usize, n_dev: usize, il: usize, owner: usize, c: usize) -> usize {
    debug_assert!(c != owner && c < n_dev && owner < n_dev);
    n_entries + il * (n_dev - 1) + if c < owner { c } else { c - 1 }
}

/// The entries of the expert cache's expert-parallel layout ([`crate::residency::EpLayers`]):
/// the model's `n_entries` layers first (each decoder layer serving its owner's experts, the
/// MTP head all of them), then per decoder layer a copy on every other GPU serving that GPU's
/// ([`copy_entry`]). Also each entry's GPU.
pub fn ep_layers(plan: &ParPlan, n_entries: usize) -> (crate::residency::EpLayers, Vec<usize>) {
    let n_layer = plan.owner.len();
    let mut real: Vec<usize> = (0..n_entries).collect();
    // The MTP head on every GPU (`--glm-parallel-mtp`): its entry the last GPU's experts.
    let head = plan.mtp.then(|| plan.owned(plan.last()));
    let mut owned: Vec<Option<Range<u32>>> = (0..n_entries)
        .map(|il| match plan.owner.get(il) {
            Some(&d) => Some(plan.owned(d)),
            None => head.clone(),
        })
        .collect();
    let mut cuda: Vec<usize> = (0..n_entries)
        .map(|il| plan.owner.get(il).copied().unwrap_or(plan.last()))
        .collect();
    for il in 0..n_layer {
        let own = plan.owner[il];
        for c in (0..plan.n_dev).filter(|&c| c != own) {
            debug_assert_eq!(real.len(), copy_entry(n_entries, plan.n_dev, il, own, c));
            real.push(il);
            owned.push(Some(plan.owned(c)));
            cuda.push(c);
        }
    }
    // Then the MTP head's copies on the other GPUs, each serving that GPU's experts
    // ([`mtp_copies`]).
    if plan.mtp && n_entries > n_layer {
        for c in 0..plan.last() {
            debug_assert!(mtp_copies(plan, n_entries).contains(&real.len()));
            real.push(n_layer);
            owned.push(Some(plan.owned(c)));
            cuda.push(c);
        }
    }
    (crate::residency::EpLayers { real, owned }, cuda)
}

/// The expert cache's entries of the MTP head's copies (`--glm-parallel-mtp`, GPU `c` the
/// `c`-th, every GPU but the last) in [`ep_layers`]' layout of `n_entries` model layers: after
/// the decoder layers' copies. Empty without them.
pub fn mtp_copies(plan: &ParPlan, n_entries: usize) -> Range<usize> {
    let n_layer = plan.owner.len();
    let start = n_entries + n_layer * (plan.n_dev - 1);
    if plan.mtp && n_entries > n_layer {
        start..start + plan.n_dev - 1
    } else {
        start..start
    }
}

/// The exchange's slot size for steps of up to `max_tokens` tokens of a model of width
/// `n_embd` with `hc` residual streams and a vocabulary of `n_vocab` over `n_dev` GPUs (at
/// most `max_out` logits rows): the largest payload of any exchange (the embedding's
/// broadcast of the streams, a reduce, the head's gather).
pub fn slot_bytes(
    n_embd: i64,
    hc: i64,
    n_vocab: i64,
    n_dev: usize,
    max_tokens: usize,
    max_out: usize,
) -> usize {
    let n = max_tokens as i64;
    let streams = (n_embd * hc * n * 4) as usize;
    let reduce = exchange::reduce_bytes(n_embd, n);
    let head = exchange::gather_bytes(&[n_vocab], max_out as i64, n_dev, 0);
    streams.max(reduce).max(head).next_multiple_of(4096)
}

/// The exchange's VRAM inbox slot size ([`exchange::PeerWrites`]): a reduce of a whole step of
/// `max_tokens` tokens of width `n_embd`. The broadcast of the streams is bigger and takes the
/// host inboxes; the reduces, 90 a step, go GPU to GPU. Called with [`MAX_TOKENS`]: a longer
/// prompt step's reduces (`--glm-prompt-step`) take the host inboxes too, as 32 MiB slots
/// would cost 48 MiB more of the expert cache's VRAM per GPU for 15% faster reduces (#686).
pub fn peer_slot_bytes(n_embd: i64, max_tokens: usize) -> usize {
    exchange::reduce_bytes(n_embd, max_tokens as i64).next_multiple_of(4096)
}

/// GLM's parallel steps on a loaded model ([`crate::model::core::Model::set_glm_par`]).
pub struct GlmPar {
    pub plan: ParPlan,
    pub ex: Arc<exchange::Exchange>,
    /// The most tokens of a step ([`step_tokens`]): the exchange's slots hold them.
    pub max_tokens: usize,
}
