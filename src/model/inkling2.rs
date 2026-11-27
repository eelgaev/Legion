//! `inkling2_loop` (Inkling-2-Mini, text), following llama.cpp's `src/models/inkling2.cpp` and
//! `inkling2-loop.cpp` (MIT, ggml-org/llama.cpp PRs #32694 and #32704) op for op:
//!
//! - **Looped core**: 24 physical blocks; the core (blocks 6..18) runs 3 passes, so a token goes
//!   through 48 layer slots (prelude 0..6, the core three times, coda 18..24). The slots share
//!   their block's weights and keep their own KV cache and conv history (slot `il` reads block
//!   [`Ink2Params::phys`]`(il)`). Each pass starts again from the prelude's output; the passes
//!   after the first take a message from the pass before (its depth blocks after the first core
//!   block, and its output, mixed per layer by four queries), which a rank-64 controller per
//!   input (Q, K, V, FFN) blends in: `x - tanh(forget) · x + tanh(gain) · msg`.
//! - **Depth residuals** ("attention residuals"): every `block_size`-th layer saves its input as
//!   a block and restarts the residual from its attention output; the attention inputs (one, or
//!   one each for Q, K, V on the global layers) and the FFN input are softmax mixes over the
//!   blocks and the residual, scored by learned queries ([`ops::mix`]).
//! - **Attention**: no RoPE. A learned relative-position bias (rank 16, distances below the
//!   bias table's extent), a per-token sink logit, Q and K clipped to an L2 norm, V RMS-normed
//!   per head, a sigmoid output gate. K, V and the attention and FFN outputs go through a 4-tap
//!   causal short conv with a residual ([`ops::conv`]; K and V of the global layers also with
//!   input-dependent taps per kv head). Sliding-window layers (1024) and global layers (every
//!   6th), whose Q and relative weights scale with the log of the position.
//! - **MoE**: 128 experts, top 8 by `sigmoid + bias`, weights renormalized times 8, SITU
//!   activation (`4 tanh(g / 4) σ(g) · 25 tanh(u / 25)`), no shared expert.
//! - **Head**: a depth mix, the output norm, the logits soft-capped at 30 and cut to the
//!   unpadded vocabulary.
//!
//! The KV cache: one F16 row of K and of V per slot and position (the sliding slots' as rings).
//! The conv history: [`Arch::token_conv`] state (F32, `K - 1` inputs of every slot's four convs).
//!
//! Two GGUF layouts load: llama.cpp's `inkling2_loop` (24 blocks, the loop in its own keys and
//! per-block `loop_*` tensors) and the earlier `inkling2` one (48 slots in the arrays, global
//! cross-pass queries, other controller names). [`crate::hf`] loads the NVFP4 checkpoint as
//! shipped under the `inkling2_loop` names.
//!
//! Every computation is a Legion op: [`ops`], GLM-5.3-Flash's [`glm_ops`] and the NVFP4 expert
//! matmul ([`nvfp4_mm`]). Not yet: the vision tower, the MTP / DSpark drafter, pipeline
//! stages, speculative verify steps.

pub mod ops;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod tiny;

use std::ops::Range;

use crate::ggml::{Context, Tensor, types};
use crate::gguf::GgufFile;
use crate::model::core::arch::same_as;
use crate::model::core::state::KvPool;
use crate::model::core::{
    Arch, ArchRef, AttnInputs, Group, Hparams, Model, ModelError, Pass, TokenConv, cols,
};
use crate::model::deepseek41::ops as ds41_ops;
use crate::model::glm5next::ops as glm_ops;
use crate::model::nvfp4_mm;

use ops::LogScale;

#[cfg(test)]
thread_local! {
    /// Tests only: the routed experts' ids of every slot (I32 `[n_expert_used, n]` per slot,
    /// slot after slot) that a one-group step on this thread takes in place of the router's,
    /// so two backends' experts are compared on one routing where a near tie in the router's
    /// scores makes their own choices differ.
    pub(crate) static FORCED_ROUTE: std::cell::RefCell<Option<Vec<i32>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn forced_route() -> Option<Vec<i32>> {
    FORCED_ROUTE.with(|r| r.borrow().clone())
}

/// [`experts_flags`]: the NVFP4 down projection's decode mv takes whole row tiles into L2
/// ([`nvfp4_mm::mul_mat_id_with`]'s `tile`).
const EXPERTS_TILE: u32 = 1;
/// [`experts_flags`]: the NVFP4 down projection reads the gate and up rows' plan of the pairs
/// and starts while SITU ends ([`nvfp4_mm::mul_mat_id_with`]'s `plan_of`).
const EXPERTS_PLAN: u32 = 2;
/// [`experts_flags`]: SITU stores an NVFP4 down projection's input as BF16 ([`ops::situ`]).
const EXPERTS_BF16_ACT: u32 = 4;
/// [`experts_flags`]: [`EXPERTS_TILE`] for the gate and up rows.
const EXPERTS_TILE_GU: u32 = 8;
/// [`experts_flags`]: without [`EXPERTS_TILE`], the NVFP4 down projection's decode mv takes
/// whole row tiles into L2 one tile ahead ([`EXPERT_MV_T1`], #1221 r9).
const EXPERTS_DOWN_T1: u32 = 16;
/// [`experts_flags`] by default: the shared plan, the BF16 SITU and the down projection's
/// one-ahead tile prefetch. On a DGX Spark (GB10) the two-ahead prefetches ([`EXPERTS_TILE`],
/// [`EXPERTS_TILE_GU`]) gained nothing on Inkling-2-Mini (the down's cost 0.3-0.4 tok/s at
/// decode); one ahead, the down's calls take 60 us instead of 65 (r9).
const EXPERTS_DEFAULT: u32 = EXPERTS_PLAN | EXPERTS_BF16_ACT | EXPERTS_DOWN_T1;

/// The down projection's decode mv by default ([`EXPERTS_DOWN_T1`], `t1` of [`EXPERT_MVS`]):
/// 8 row tiles of 16 rows a CTA, as the plain mv's, each whole tile prefetched into L2 one tile
/// ahead.
const EXPERT_MV_T1: nvfp4_mm::GroupedMv = nvfp4_mm::GroupedMv {
    rtw: 1,
    rpt: 8,
    sb: 16,
    pft: 1,
};

/// How the routed NVFP4 experts run (#1221 r7), all the same bits: [`EXPERTS_TILE`],
/// [`EXPERTS_PLAN`], [`EXPERTS_BF16_ACT`], [`EXPERTS_TILE_GU`] and [`EXPERTS_DOWN_T1`]
/// (r9); [`EXPERTS_DEFAULT`] unless
/// `LEGION_DEBUG_INK_EXPERTS=<mask>` takes others for A/B runs (0: as round 6).
fn experts_flags() -> u32 {
    static FLAGS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FLAGS.get_or_init(|| {
        let env = std::env::var("LEGION_DEBUG_INK_EXPERTS").ok();
        experts_flags_of(env.as_deref())
    })
}

/// Whether `Model::ink_mm_set` takes BF16 weights (#1221 r7, r9 at 2-8 tokens): unless
/// `LEGION_DEBUG_INK_BF16_SET=0` (A/B runs: a node a product, as round 6).
fn bf16_set() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGION_DEBUG_INK_BF16_SET").map_or(true, |v| v.trim() != "0"))
}

/// The Q8_0 attention sets', lone products' and head's warp layout on CUDA's deeper-load set
/// kernel (#1221 r10, [`glm_ops::linear_set_deep`]): one row a warp and four loads a lane
/// (about +1% GGUF decode on a GB10) unless `LEGION_DEBUG_INK_Q8_DEEP` names another for A/B
/// runs (`r1u2`, `r2u2`, `r2u4`) or is `0` (round 9's linear_set_q8 sets and linear_mv_q8
/// products). The same bits either way.
fn q8_deep() -> Option<glm_ops::DeepQ8> {
    static DEEP: std::sync::OnceLock<Option<glm_ops::DeepQ8>> = std::sync::OnceLock::new();
    *DEEP.get_or_init(|| q8_deep_of(std::env::var("LEGION_DEBUG_INK_Q8_DEEP").ok().as_deref()))
}

/// [`q8_deep`]'s layout from the variable's value: unset or not a name, `r1u4`.
fn q8_deep_of(v: Option<&str>) -> Option<glm_ops::DeepQ8> {
    let deep = |rows, loads| Some(glm_ops::DeepQ8 { rows, loads });
    match v.map(str::trim) {
        Some("0") => None,
        Some("r1u2") => deep(1, 2),
        Some("r2u2") => deep(2, 2),
        Some("r2u4") => deep(2, 4),
        _ => deep(1, 4),
    }
}

/// The grouped mv variants of the NVFP4 experts' one-token tiles an A/B run may pick (#1221 r9,
/// [`expert_mv`]), all compiled for them (`mvg<1>` without the SwiGLU) and all the same bits:
/// `pR` R row tiles of 16 rows a CTA (the default's p8), `wWpR` tiles of 16 * W rows, `sS`
/// stages of S blocks (16 otherwise), `tT` whole row tiles prefetched T tiles ahead (t2 is
/// `nvfp4_mm::DOWN_MV_TILE`, [`EXPERTS_TILE`]'s).
pub(crate) const EXPERT_MVS: [(&str, nvfp4_mm::GroupedMv); 15] = {
    const fn g(rtw: i32, rpt: i32, sb: i32, pft: i32) -> nvfp4_mm::GroupedMv {
        nvfp4_mm::GroupedMv { rtw, rpt, sb, pft }
    }
    [
        ("p1", g(1, 1, 16, 0)),
        ("p2", g(1, 2, 16, 0)),
        ("p4", g(1, 4, 16, 0)),
        ("p8", g(1, 8, 16, 0)),
        ("p16", g(1, 16, 16, 0)),
        ("w2p2", g(2, 2, 16, 0)),
        ("w2p4", g(2, 4, 16, 0)),
        ("w4p1", g(4, 1, 16, 0)),
        ("w4p2", g(4, 2, 16, 0)),
        ("s24", g(1, 8, 24, 0)),
        ("s40", g(1, 8, 40, 0)),
        ("t1", g(1, 8, 16, 1)),
        ("t2", g(1, 8, 16, 2)),
        ("t1p4", g(1, 4, 16, 1)),
        ("t1p16", g(1, 16, 16, 1)),
    ]
};

/// The routed NVFP4 experts' grouped mv variant at one-token tiles for A/B runs (#1221 r9): the
/// [`EXPERT_MVS`] name in `LEGION_DEBUG_INK_DOWN_MV` for the down projection,
/// `LEGION_DEBUG_INK_GU_MV` for the gate and up rows; unset (or not a name), the default's
/// ([`EXPERTS_DOWN_T1`]'s `t1` for the down, the plain `p8` for the gate and up rows).
fn expert_mv(down: bool) -> Option<nvfp4_mm::GroupedMv> {
    static MV: std::sync::OnceLock<[Option<nvfp4_mm::GroupedMv>; 2]> = std::sync::OnceLock::new();
    MV.get_or_init(|| {
        ["LEGION_DEBUG_INK_GU_MV", "LEGION_DEBUG_INK_DOWN_MV"]
            .map(|var| expert_mv_of(std::env::var(var).ok().as_deref()))
    })[down as usize]
}

/// [`expert_mv`] given the environment's value.
pub(crate) fn expert_mv_of(env: Option<&str>) -> Option<nvfp4_mm::GroupedMv> {
    let name = env?.trim();
    EXPERT_MVS.iter().find(|(n, _)| *n == name).map(|&(_, g)| g)
}

/// [`experts_flags`] given the environment's value.
pub(crate) fn experts_flags_of(env: Option<&str>) -> u32 {
    const ALL: u32 =
        EXPERTS_TILE | EXPERTS_PLAN | EXPERTS_BF16_ACT | EXPERTS_TILE_GU | EXPERTS_DOWN_T1;
    env.and_then(|v| v.trim().parse::<u32>().ok())
        .map_or(EXPERTS_DEFAULT, |v| v & ALL)
}

/// The GGUF architecture name (llama.cpp's).
pub const ARCH: &str = "inkling2_loop";
/// The earlier GGUF layout's architecture name.
pub const ARCH_OLD: &str = "inkling2";

/// The RMS norms' epsilon of the depth mixes and the controllers (llama.cpp's constant).
const MIX_EPS: f32 = 1e-6;
/// Head sizes the attention kernels take.
const HEAD_DIMS: [i64; 3] = [32, 64, 128];

/// A looped layer's controllers, in llama.cpp's order (the message queries' columns).
const CTRL_NAMES: [&str; 4] = ["q", "k", "v", "ffn"];
/// The earlier layout's names for them.
const CTRL_NAMES_OLD: [&str; 4] = ["attn", "k", "v", "mlp"];

/// Inkling-2's hyperparameters beyond [`Hparams`].
#[derive(Debug, Clone, PartialEq)]
pub struct Ink2Params {
    /// The earlier `inkling2` GGUF layout.
    pub old_layout: bool,
    /// Physical blocks (weights).
    pub n_phys: usize,
    /// The looped blocks.
    pub core: Range<usize>,
    /// Passes over the core.
    pub passes: usize,
    /// Sliding-window attention, per physical block.
    pub swa: Vec<bool>,
    pub window: usize,
    /// Layers per depth block.
    pub block: usize,
    /// Rank of the relative-position bias.
    pub n_rel: i64,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale: LogScale,
    /// Conv taps.
    pub conv_k: i64,
    /// Conv channels per slot: K, V, the attention output, the FFN output.
    pub slot_width: i64,
    pub weights_scale: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub softcap: f32,
    /// The vocabulary without the embedding's padding rows.
    pub n_vocab: i64,
}

impl Ink2Params {
    /// Layer slots: the prelude and coda once, the core `passes` times.
    pub fn n_slots(&self) -> usize {
        self.n_phys + (self.passes - 1) * self.core.len()
    }

    /// The physical block slot `il` runs.
    pub fn phys(&self, il: usize) -> usize {
        if il < self.n_phys {
            il
        } else {
            self.core.start + (il - self.n_phys) % self.core.len()
        }
    }

    /// The slot of core block `ip` in pass `pass`.
    pub fn slot(&self, pass: usize, ip: usize) -> usize {
        if pass == 0 {
            ip
        } else {
            self.n_phys + (pass - 1) * self.core.len() + (ip - self.core.start)
        }
    }

    /// The weight name of controller `c` (0..4: Q, K, V, FFN) of block `ip`, part `part`
    /// (`down`, `gain` or `forget`).
    fn ctrl_name(&self, ip: usize, c: usize, part: &str) -> String {
        if self.old_layout {
            format!("blk.{ip}.loop_{}_{part}.weight", CTRL_NAMES_OLD[c])
        } else {
            format!("blk.{ip}.loop_{}_ctrl_{part}.weight", CTRL_NAMES[c])
        }
    }
}

/// The Inkling-2 hyperparameters of `hp`, if it is one.
pub fn params(hp: &Hparams) -> Option<&Ink2Params> {
    hp.arch.as_any().downcast_ref::<Ink2Params>()
}

/// The numbers of the array `key` (or its one value).
fn numbers(g: &GgufFile, key: &str) -> Option<Vec<f64>> {
    let v = g.get(key)?;
    match v.as_array() {
        Some(a) => a.iter().map(|x| x.as_f64()).collect(),
        None => v.as_f64().map(|x| vec![x]),
    }
}

/// The hyperparameters of an `inkling2_loop` or `inkling2` (`arch`) GGUF.
pub(crate) fn parse(g: &GgufFile, arch: &str) -> Result<Hparams, ModelError> {
    let old = arch == ARCH_OLD;
    let key = |k: &str| format!("{arch}.{k}");
    let meta = |k: &str| ModelError::Metadata(key(k));
    let u = |k: &str| -> Result<u64, ModelError> { g.get_u64(&key(k)).ok_or_else(|| meta(k)) };
    let f = |k: &str| -> Result<f32, ModelError> {
        g.get(&key(k))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| meta(k))
    };
    let unsupported = |what: &str| Err(ModelError::Unsupported(format!("Inkling-2 {what}")));
    // A per-layer key: every entry the same (the kernels take one shape), or one value.
    let uniform = |k: &str| -> Result<i64, ModelError> {
        let v = numbers(g, &key(k)).ok_or_else(|| meta(k))?;
        match v.first() {
            Some(&x) if v.iter().all(|&y| y == x) => Ok(x as i64),
            _ => Err(ModelError::Unsupported(format!(
                "Inkling-2 with layers of different `{}`",
                key(k)
            ))),
        }
    };

    if g.tensors()
        .iter()
        .any(|t| t.name.starts_with("v.") || t.name.starts_with("mm."))
    {
        return unsupported("vision tower (run the text GGUF; images are not in Legion yet)");
    }
    let n_blocks = u("block_count")? as usize;
    let count_key = if old { "num_loops" } else { "loop.count" };
    let (passes, start, end) = if old {
        (u("num_loops")?, u("loop.core_start")?, u("loop.core_end")?)
    } else {
        (u("loop.count")?, u("loop.start")?, u("loop.end")?)
    };
    let (passes, start, end) = (passes as usize, start as usize, end as usize);
    if g.get(&key("loop.varying")).and_then(|v| v.as_bool()) == Some(true) {
        return unsupported("with a variable number of loop passes");
    }
    let block = u("attn_res.block_size")? as usize;
    let core_len = end.saturating_sub(start);
    // The extra passes' slots, checked: a huge loop count must not wrap.
    let extra = passes
        .saturating_sub(1)
        .checked_mul(core_len)
        .ok_or_else(|| meta(count_key))?;
    let n_phys = if old {
        n_blocks
            .checked_sub(extra)
            .ok_or_else(|| meta("block_count"))?
    } else {
        n_blocks
    };
    if passes < 2
        || block == 0
        || start == 0
        || start >= end
        || end > n_phys
        || start % block != 0
        || end % block != 0
    {
        return Err(ModelError::Layout(format!(
            "inkling2: invalid loop core [{start}, {end}) with block size {block}, {passes} \
             passes for {n_phys} blocks"
        )));
    }
    let n_slots = n_phys.checked_add(extra).ok_or_else(|| meta(count_key))?;
    let pattern = g
        .get(&key("attention.sliding_window_pattern"))
        .and_then(|v| {
            v.as_array()
                .and_then(|a| a.iter().map(|x| x.as_bool()).collect::<Option<Vec<_>>>())
        })
        .ok_or_else(|| meta("attention.sliding_window_pattern"))?;
    if pattern.len() != n_blocks {
        return Err(meta("attention.sliding_window_pattern"));
    }
    let swa = pattern[..n_phys].to_vec();

    let n_embd = u("embedding_length")? as i64;
    let n_head = uniform("attention.head_count")?;
    let n_head_kv = uniform("attention.head_count_kv")?;
    let head = u("attention.key_length")? as i64;
    let other = |name: &str| g.get_u64(&key(name)).is_some_and(|v| v != head as u64);
    if other("attention.value_length") || other("attention.key_length_swa") {
        return unsupported("with different key, value or sliding-window head sizes");
    }
    if !HEAD_DIMS.contains(&head) {
        return unsupported(&format!("heads of {head} (the kernels take 32, 64 or 128)"));
    }
    if n_head_kv == 0 || n_head % n_head_kv != 0 || n_head / n_head_kv > 4 {
        return unsupported(&format!(
            "with {n_head} query heads over {n_head_kv} kv heads (the kernel takes up to 4 a kv \
             head)"
        ));
    }
    let n_rel = u("attention.relative_length")? as i64;
    if !(1..=16).contains(&n_rel) {
        return unsupported(&format!("relative bias of rank {n_rel} (up to 16)"));
    }
    let conv_k = u("shortconv.l_cache")? as i64;
    if !(2..=ops::MAX_TAPS as i64).contains(&conv_k) {
        return unsupported(&format!("short convs of {conv_k} taps"));
    }
    let n_vocab = u("vocab_size_unpadded")? as i64;
    let embd = g
        .tensor("token_embd.weight")
        .ok_or_else(|| ModelError::Tensor("token_embd.weight".into()))?;
    // The reader trims trailing unit dimensions: a missing row count is one row.
    let embd_rows = embd.shape.get(1).copied().unwrap_or(1);
    if n_vocab == 0 || n_vocab as u64 > embd_rows {
        return Err(meta("vocab_size_unpadded"));
    }
    // The head's soft-cap crops the logits to `n_vocab`: the head must have that many rows.
    if let Some(out) = g.tensor("output.weight") {
        let rows = out.shape.get(1).copied().unwrap_or(1);
        if (n_vocab as u64) > rows {
            return Err(ModelError::Layout(format!(
                "inkling2: output.weight has {rows} rows for a vocabulary of {n_vocab}"
            )));
        }
    }
    let blocks_max = n_phys / block + 2;
    if blocks_max > ops::MAX_STATES {
        return unsupported(&format!("with {blocks_max} depth blocks"));
    }
    let params = Ink2Params {
        old_layout: old,
        n_phys,
        core: start..end,
        passes,
        swa,
        window: u("attention.sliding_window")? as usize,
        block,
        n_rel,
        qk_clip: f("attention.qk_clip")?,
        rel_clip: f("attention.relative_clip")?,
        log_scale: LogScale {
            floor: u("attention.log_scale_floor")? as f32,
            alpha: f("attention.log_scale_alpha")?,
        },
        conv_k,
        slot_width: 2 * head * n_head_kv + 2 * n_embd,
        weights_scale: f("expert_weights_scale").unwrap_or(1.0),
        situ_beta: f("activation.situ_beta")?,
        situ_lb: f("activation.situ_linear_beta")?,
        softcap: f("final_logit_softcapping").unwrap_or(0.0),
        n_vocab,
    };
    if params.window == 0 || params.log_scale.floor <= 0.0 {
        return Err(meta("attention.sliding_window"));
    }
    let n_expert = u("expert_count")? as i64;
    let n_expert_used = u("expert_used_count")? as i64;
    if !(1..=n_expert).contains(&n_expert_used) {
        return Err(meta("expert_used_count"));
    }
    Ok(Hparams {
        n_layer: n_slots,
        n_embd,
        n_head,
        n_head_kv,
        head_dim: head,
        n_rot: 0,
        rope_sections: [0; 4],
        rope_freq_base: 10000.0,
        n_ctx_train: u("context_length")?.min(i32::MAX as u64) as i32,
        rms_eps: f("attention.layer_norm_rms_epsilon")?,
        moe: true,
        n_expert,
        n_expert_used,
        // No recurrent layers.
        full_attention_interval: 1,
        ssm_d_conv: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 1,
        ssm_d_inner: 0,
        n_vocab,
        arch: ArchRef::new(params),
    })
}

impl Arch for Ink2Params {
    fn name(&self) -> &'static str {
        if self.old_layout { ARCH_OLD } else { ARCH }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, _hp: &Hparams, _il: usize) -> bool {
        false
    }

    fn sliding_layer(&self, _hp: &Hparams, il: usize) -> bool {
        self.swa[self.phys(il)]
    }

    fn swa_window(&self) -> usize {
        self.window
    }

    /// The short convs' history: `K - 1` inputs of every slot's K, V, attention output and FFN
    /// output (slot `il` at channels `il · slot_width ..`).
    fn token_conv(&self, _hp: &Hparams) -> Option<TokenConv> {
        Some(TokenConv {
            layer: 0,
            history: self.conv_k - 1,
            width: self.n_slots() as i64 * self.slot_width,
            n_hist: 0,
        })
    }

    /// The attention reads the cache in place through the block tables.
    fn paged(&self, _m: &Model, _kv: &KvPool) -> bool {
        true
    }

    fn always_paged(&self) -> bool {
        true
    }

    fn paged_for(&self, _m: &Model, _kv: &KvPool, _groups: &[Group<'_>]) -> bool {
        true
    }

    /// The embeddings RMS-normed times `token_embd_norm`.
    fn embed_tokens<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        tokens: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let x = glm_ops::embed(ctx, m.w("token_embd.weight")?, tokens, 1);
        let x = ctx.reshape_2d(x, m.hp.n_embd, tokens.ne()[0]);
        Ok(Some(ops::rms(
            ctx,
            x,
            Some(m.w("token_embd_norm.weight")?),
            m.hp.rms_eps,
        )))
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        if !capture.is_empty() {
            return Err(ModelError::Unsupported(format!(
                "hidden-state captures {capture:?} of Inkling-2"
            )));
        }
        Ok((m.ink_layers(pass, x, layers)?, Vec::new()))
    }

    /// The last layer ends in the head's depth mix, normed, at the output rows.
    fn output_ready(&self) -> bool {
        true
    }

    /// The logits soft-capped and cut to the unpadded vocabulary. The greedy pick over
    /// [`glm_ops::argmax_wide`]'s CTAs (#1221): one CTA a token read the 200K logits slowly;
    /// the same index as `argmax` (the first of the largest) for NaN-free logits. The soft cap
    /// passes a NaN through (`tanh(NaN)` is NaN), so that rests on the output projection.
    /// The projection reads only the unpadded vocabulary's rows (#1221 r9: the padding's
    /// logits were cut anyway), a Q8_0 head at decode steps on the deeper-load set kernel
    /// ([`glm_ops::linear_set_deep`], [`q8_deep`]; r10): the same bits.
    fn head<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        x: Tensor<'a>,
    ) -> Result<Option<(Tensor<'a>, Tensor<'a>)>, ModelError> {
        let name = "output.weight";
        let w = m.w(name)?;
        let ws = m.w("output.scale").ok();
        let w = if w.ne()[1] > self.n_vocab && w.ne()[2] == 1 {
            ctx.view_2d(w, [w.ne()[0], self.n_vocab], w.nb()[1], 0)
        } else {
            w
        };
        let n: i64 = x.ne()[1..].iter().product();
        let deep = q8_deep().filter(|_| {
            ws.is_none()
                && x.ty() == types::F32
                && x.is_contiguous()
                && glm_ops::linear_set_takes(&[w], n)
        });
        let logits = match deep {
            Some(d) => {
                glm_ops::linear_set_deep(ctx, &[(w, x, glm_ops::SetEpi::Plain)], d).remove(0)
            }
            None => m.glm_mm_w(ctx, name, w, ws, x, None)?,
        };
        let logits = ops::softcap(ctx, logits, self.softcap, self.n_vocab);
        Ok(Some((logits, glm_ops::argmax_wide(ctx, logits))))
    }

    fn external_drafters(&self) -> bool {
        false
    }

    /// The head's [`glm_ops::argmax`] takes the first of the largest.
    fn argmax_first_of_ties(&self) -> bool {
        true
    }

    /// Tests only: the [`FORCED_ROUTE`] input, when set.
    #[cfg(test)]
    fn group_inputs_len(&self) -> usize {
        forced_route().is_some() as usize
    }

    #[cfg(test)]
    fn group_inputs<'a>(
        &self,
        ctx: &'a Context,
        _g: &crate::model::core::GroupShape,
        i: usize,
    ) -> Vec<Tensor<'a>> {
        let Some(ids) = forced_route() else {
            return Vec::new();
        };
        assert_eq!(i, 0, "a forced route takes one group");
        let t = ctx.new_tensor(types::I32, &[ids.len() as i64]);
        vec![t.named("forced_route").as_input()]
    }

    #[cfg(test)]
    fn fill_inputs(
        &self,
        _m: &Model,
        inputs: &crate::model::core::Inputs<'_>,
        _groups: &[Group<'_>],
        _shape: &[crate::model::core::GroupShape],
        _rec: &crate::model::core::StatePool,
        _kv: &KvPool,
        _spec: Option<&crate::model::core::SpecPool>,
        sink: &mut dyn crate::model::core::InputSink,
    ) {
        if let (Some(ids), Some(&t)) =
            (forced_route(), inputs.group.first().and_then(|g| g.first()))
        {
            sink.i32(t, &ids);
        }
    }
}

/// The states a depth mix reads: the blocks, then the residual `cur`.
fn states<'a>(blocks: &[Tensor<'a>], cur: Tensor<'a>) -> Vec<Tensor<'a>> {
    let mut s = blocks.to_vec();
    s.push(cur);
    s
}

/// The BF16 rows of the inputs of a layer's NVFP4 products (#1221): made once per input
/// ([`Model::glm_mm_shared`]) for every product reading it, which [`Model::glm_mm`] would each
/// round again. The same bits; an NVFP4 checkpoint's sliding layer rounds its normed input
/// once for Q, K, V, the relative weights, the sinks and the gate instead of six times.
#[derive(Default)]
struct Bf16Rows<'a>(Vec<(*mut crate::ggml::ffi::ggml_tensor, Option<Tensor<'a>>)>);

impl<'a> Bf16Rows<'a> {
    /// The slot of `x`'s rows (empty until a product makes them).
    fn of(&mut self, x: Tensor<'a>) -> &mut Option<Tensor<'a>> {
        let i = match self.0.iter().position(|(t, _)| *t == x.raw()) {
            Some(i) => i,
            None => {
                self.0.push((x.raw(), None));
                self.0.len() - 1
            }
        };
        &mut self.0[i].1
    }
}

/// Slice `j` (`[e, n]`) of a mix of several queries (`[e, n, nq]`, [`ops::mix`]).
fn slice<'a>(ctx: &'a Context, m: Tensor<'a>, j: usize) -> Tensor<'a> {
    let [e, n, _, _] = m.ne();
    ctx.view_2d(m, [e, n], m.nb()[1], j * m.nb()[2])
}

impl Model {
    /// The Inkling-2 hyperparameters of this model.
    fn ink(&self) -> &Ink2Params {
        params(&self.hp).expect("an inkling2 model")
    }

    /// [`Self::glm_mm`] of block `ip`'s weight `name`.
    fn ink_mm<'a>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm(ctx, &format!("blk.{ip}.{name}"), x)
    }

    /// [`Self::ink_mm`] with an NVFP4 weight's BF16 rows of `x` shared through `rows`.
    fn ink_mm_b<'a>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        name: &str,
        x: Tensor<'a>,
        rows: &mut Bf16Rows<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm_shared(ctx, &format!("blk.{ip}.{name}"), x, rows.of(x))
    }

    /// [`Self::ink_mm_b`] of block `ip`'s weights `names`, all of `x`: one glm_linear_set node
    /// ([`glm_ops::linear_set`], linear_mv_q8's sums row by row: the same bits) where it takes
    /// them, Q8_0 weights without a scale over up to 8 tokens (decode steps; #1221 r6) or BF16
    /// ones (the NVFP4 checkpoint's) on the GB10 kernels, at one token (linear_mv_bf16's sums,
    /// [`glm_ops::linear_set_bf16_takes`]; r7) and at 2-8 (linear_mv_bf16x's; r9), unless `own`
    /// (traced runs keep a node a product); else a node each. Q8_0 sets run CUDA's deeper-load
    /// kernel ([`glm_ops::linear_set_deep`], [`q8_deep`]; r10), and then a single Q8_0 product
    /// (`N` 1) takes it too.
    fn ink_mm_set<'a, const N: usize>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        names: [&str; N],
        x: Tensor<'a>,
        rows: &mut Bf16Rows<'a>,
        own: bool,
    ) -> Result<[Tensor<'a>; N], ModelError> {
        self.ink_mm_setx(ctx, ip, names, [x; N], rows, own)
    }

    /// [`Self::ink_mm_set`] with an input a product (`xs[j]` for `names[j]`, all of the same
    /// tokens): a global layer's K and V read different inputs (#1221 r9).
    fn ink_mm_setx<'a, const N: usize>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        names: [&str; N],
        xs: [Tensor<'a>; N],
        rows: &mut Bf16Rows<'a>,
        own: bool,
    ) -> Result<[Tensor<'a>; N], ModelError> {
        let ws = names
            .iter()
            .map(|n| self.w(&format!("blk.{ip}.{n}")))
            .collect::<Result<Vec<_>, _>>()?;
        let tokens = |x: &Tensor<'_>| -> i64 { x.ne()[1..].iter().product() };
        let n = tokens(&xs[0]);
        let deep = q8_deep().filter(|_| glm_ops::linear_set_takes(&ws, n));
        let set = !own
            && ((2..=glm_ops::SET_MAX).contains(&N) || (N == 1 && deep.is_some()))
            && xs
                .iter()
                .all(|x| x.ty() == types::F32 && x.is_contiguous() && tokens(x) == n)
            && (glm_ops::linear_set_takes(&ws, n)
                || (bf16_set() && glm_ops::linear_set_bf16_takes(&ws, n)))
            && names.iter().all(|name| {
                name.strip_suffix(".weight")
                    .is_some_and(|b| self.w(&format!("blk.{ip}.{b}.scale")).is_err())
            });
        let out = if set {
            let set: Vec<_> = ws
                .iter()
                .zip(xs)
                .map(|(&w, x)| (w, x, glm_ops::SetEpi::Plain))
                .collect();
            match deep {
                Some(d) => glm_ops::linear_set_deep(ctx, &set, d),
                None => glm_ops::linear_set(ctx, &set),
            }
        } else {
            names
                .iter()
                .zip(xs)
                .map(|(name, x)| self.ink_mm_b(ctx, ip, name, x, rows))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(out
            .try_into()
            .unwrap_or_else(|_| unreachable!("a product a name")))
    }

    /// `x` RMS-normed times block `ip`'s weight `name`.
    fn ink_rms<'a>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        Ok(ops::rms(ctx, x, Some(self.lw(ip, name)?), self.hp.rms_eps))
    }

    /// The layers over the normed embeddings `x` (`[n_embd, n]`); ends in the head's input.
    fn ink_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        layers: Range<usize>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ink();
        let hp = &self.hp;
        if layers != (0..hp.n_layer) {
            return Err(ModelError::Unsupported(format!(
                "Inkling-2 in pipeline stages (layers {layers:?})"
            )));
        }
        let types = pass.kv.types();
        if types.k.ggml() != types::F16 || types.v.ggml() != types::F16 {
            return Err(ModelError::Unsupported(
                "Inkling-2 with a KV cache type other than F16".into(),
            ));
        }
        if pass.shape.iter().any(|g| g.verify) {
            return Err(ModelError::Unsupported(
                "speculative verify steps of Inkling-2".into(),
            ));
        }
        let mut cur = ctx.reshape_2d(x, hp.n_embd, pass.n);
        tap.tap("inp_norm", 0, cur);
        let mut blocks: Vec<Tensor<'a>> = Vec::new();
        for ip in 0..p.core.start {
            cur = self.ink_layer(pass, ip, ip, cur, &mut blocks, None)?;
        }
        // Every pass starts from the prelude's output.
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        for pi in 0..p.passes {
            let mut bank = Vec::new();
            if pi > 0 {
                // The previous pass's blocks after its first core block, and its output.
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank.push(cur);
                cur = prelude;
                blocks = prelude_blocks.clone();
            }
            for ip in p.core.clone() {
                let il = p.slot(pi, ip);
                let msgs = if pi == 0 {
                    None
                } else {
                    let q = self.ink_msg_query(ctx, ip)?;
                    let m = ops::mix(ctx, &bank, q, MIX_EPS);
                    tap.tap("loop_msgs", il, m);
                    Some(m)
                };
                cur = self.ink_layer(pass, il, ip, cur, &mut blocks, msgs)?;
            }
        }
        for ip in p.core.end..p.n_phys {
            cur = self.ink_layer(pass, ip, ip, cur, &mut blocks, None)?;
        }
        blocks.push(cur);
        let x = ops::mix(ctx, &blocks, self.w("output_res_query.weight")?, MIX_EPS);
        let x = ctx.reshape_2d(x, hp.n_embd, pass.n);
        tap.tap("result_mix", 0, x);
        let x = match pass.out_rows {
            Some(rows) => glm_ops::get_cols(ctx, x, rows),
            None => x,
        };
        let x = ops::rms(ctx, x, Some(self.w("output_norm.weight")?), hp.rms_eps);
        tap.tap("result_norm", 0, x);
        Ok(x)
    }

    /// Core block `ip`'s message queries, F32 `[n_embd, 4]` (Q, K, V, FFN).
    fn ink_msg_query<'a>(&'a self, ctx: &'a Context, ip: usize) -> Result<Tensor<'a>, ModelError> {
        let p = self.ink();
        if !p.old_layout {
            return self.lw(ip, "loop_msg_query.weight");
        }
        // The earlier layout keeps them in two tables, two columns a core block each
        // (llama.cpp's converter stacks [q 2i, kv 2i, kv 2i + 1, q 2i + 1]).
        let q = self.w("loop_res_query.weight")?;
        let kv = self.w("loop_kv_res_query.weight")?;
        let i = (ip - p.core.start) as i64;
        Ok(glm_ops::join_cols(
            ctx,
            &[
                cols(ctx, q, 2 * i, 1),
                cols(ctx, kv, 2 * i, 1),
                cols(ctx, kv, 2 * i + 1, 1),
                cols(ctx, q, 2 * i + 1, 1),
            ],
        ))
    }

    /// Controller `c` of core block `ip` over `cur` and its message `msg` (F32 `[n_embd, n]`).
    fn ink_control<'a>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        c: usize,
        cur: Tensor<'a>,
        msg: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let p = self.ink();
        let g = ops::ctrl_in(ctx, cur, msg, MIX_EPS);
        let g = self.glm_mm(ctx, &p.ctrl_name(ip, c, "down"), g)?;
        // Small steps: the gain and forget products and the output in one node of the same bits
        // (#1221 r5), when the weights suit it and carry no scales (glm_mm's products of these
        // are then scale 1).
        let names = ["gain", "forget"].map(|part| p.ctrl_name(ip, c, part));
        let unscaled = names.iter().all(|nm| {
            nm.strip_suffix(".weight")
                .is_some_and(|b| self.w(&format!("{b}.scale")).is_err())
        });
        if let (true, Ok(gw), Ok(fw)) = (unscaled, self.w(&names[0]), self.w(&names[1]))
            && cur.ne()[2..] == [1, 1]
            && ops::ctrl_gate_takes(gw, fw, cur.ne()[1])
        {
            return Ok(ops::ctrl_gate(ctx, cur, msg, g, gw, fw));
        }
        let mut rows = Bf16Rows::default();
        let gain = self.glm_mm_shared(ctx, &p.ctrl_name(ip, c, "gain"), g, rows.of(g))?;
        let forget = self.glm_mm_shared(ctx, &p.ctrl_name(ip, c, "forget"), g, rows.of(g))?;
        Ok(ops::ctrl_out(ctx, cur, msg, gain, forget))
    }

    /// Slot `il`'s short conv `name` (of block `ip`) over `x` (F32 `[c, n]`) with the
    /// input-dependent taps `dynamic`, its history at channels `off ..` of the slot's.
    #[allow(clippy::too_many_arguments)]
    fn ink_conv<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        ip: usize,
        name: &str,
        x: Tensor<'a>,
        dynamic: Option<Tensor<'a>>,
        off: i64,
    ) -> Result<Tensor<'a>, ModelError> {
        self.ink_conv_add(pass, il, ip, name, x, dynamic, off, None)
    }

    /// [`Self::ink_conv`] plus the residual `res` (F32 `[c, n]`) when given: the sum of
    /// `glm_add(res, conv)` in the conv's own node, one launch fewer (#1221 r6).
    #[allow(clippy::too_many_arguments)]
    fn ink_conv_add<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        ip: usize,
        name: &str,
        x: Tensor<'a>,
        dynamic: Option<Tensor<'a>>,
        off: i64,
        res: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let p = self.ink();
        let w = self.lw(ip, name)?;
        let k = w.ne()[0];
        let taps = if dynamic.is_some() { 2 * k } else { k };
        let scale = 1.0 / ((taps + 1) as f32).sqrt();
        let base = il as i64 * p.slot_width + off;
        let mut outs = Vec::new();
        for (at, sh) in pass.groups() {
            let ng = sh.tokens() as i64;
            let state = pass.rec.ple_slots(ctx, 0, sh.slot, sh.n_seqs);
            outs.push(ops::conv_add(
                ctx,
                cols(ctx, x, at, ng),
                w,
                state,
                dynamic.map(|d| cols(ctx, d, at, ng)),
                sh.n_tok as i64,
                base,
                scale,
                res.map(|r| cols(ctx, r, at, ng)),
            ));
        }
        Ok(glm_ops::join_cols(ctx, &outs))
    }

    /// Slot `il` (block `ip`) over the residual `cur`, with the depth blocks so far `blocks`
    /// (this layer may add one) and, in a pass after the first, the messages `msgs` (F32
    /// `[n_embd, n, 4]`).
    fn ink_layer<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        ip: usize,
        cur: Tensor<'a>,
        blocks: &mut Vec<Tensor<'a>>,
        msgs: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ink();
        let hp = &self.hp;
        let (n, e) = (pass.n, hp.n_embd);
        let swa = p.swa[ip];

        // The attention inputs: one mix (sliding layers) or one each for Q, K, V.
        let inp_sa = cur;
        let mut inputs = [cur; 3];
        // Block 0 has no query (one state); every later block must have one.
        let query = if ip == 0 {
            self.lw(ip, "attn_res_query.weight").ok()
        } else {
            Some(self.lw(ip, "attn_res_query.weight")?)
        };
        if let Some(q) = query {
            let mixed = ops::mix(ctx, &states(blocks, cur), q, MIX_EPS);
            tap.tap("attn_mix", il, mixed);
            for (j, x) in inputs.iter_mut().enumerate() {
                *x = slice(ctx, mixed, if swa { 0 } else { j });
            }
        }
        if let Some(m) = msgs {
            for (j, x) in inputs.iter_mut().enumerate() {
                *x = self.ink_control(ctx, ip, j, *x, slice(ctx, m, j))?;
                tap.tap(["attn_ctrl_q", "attn_ctrl_k", "attn_ctrl_v"][j], il, *x);
            }
        }
        let inputs = if swa && msgs.is_some() {
            let mut normed = inputs;
            for x in normed.iter_mut() {
                *x = self.ink_rms(ctx, ip, "attn_norm.weight", *x)?;
            }
            normed
        } else if swa {
            [self.ink_rms(ctx, ip, "attn_norm.weight", inputs[0])?; 3]
        } else {
            [
                self.ink_rms(ctx, ip, "attn_q_inp_norm.weight", inputs[0])?,
                self.ink_rms(ctx, ip, "attn_k_inp_norm.weight", inputs[1])?,
                self.ink_rms(ctx, ip, "attn_v_inp_norm.weight", inputs[2])?,
            ]
        };
        for (name, x) in ["attn_in_q", "attn_in_k", "attn_in_v"]
            .into_iter()
            .zip(inputs)
        {
            tap.tap(name, il, x);
        }

        // A block's first layer starts a new residual; its input joins the depth blocks.
        // Otherwise the residual add is the post conv's own, unless a trace wants the
        // attention's output by itself.
        let starts = ip % p.block == 0;
        let fold = !starts && !tap.on;
        let out = self.ink_attn(pass, il, ip, &inputs, fold.then_some(inp_sa))?;
        tap.tap("attn_out", il, out);
        let cur = if starts {
            blocks.push(inp_sa);
            out
        } else if fold {
            out
        } else {
            glm_ops::add(ctx, inp_sa, out)
        };

        let ffn_inp = cur;
        let h = ops::mix(
            ctx,
            &states(blocks, cur),
            self.lw(ip, "ffn_res_query.weight")?,
            MIX_EPS,
        );
        let mut h = ctx.reshape_2d(h, e, n);
        tap.tap("ffn_mix", il, h);
        if let Some(m) = msgs {
            h = self.ink_control(ctx, ip, 3, h, slice(ctx, m, 3))?;
            tap.tap("ffn_ctrl", il, h);
        }
        let h = self.ink_rms(ctx, ip, "ffn_norm.weight", h)?;
        tap.tap("ffn_inp_norm", il, h);
        let y = self.ink_moe(pass, il, ip, h)?;
        tap.tap("ffn_moe_out", il, y);
        let kvw = hp.head_dim * hp.n_head_kv;
        let name = "ffn_post_conv.weight";
        let out = if tap.on {
            let y = self.ink_conv(pass, il, ip, name, y, None, 2 * kvw + e)?;
            tap.tap("ffn_out", il, y);
            glm_ops::add(ctx, ffn_inp, y)
        } else {
            self.ink_conv_add(pass, il, ip, name, y, None, 2 * kvw + e, Some(ffn_inp))?
        };
        tap.tap("l_out", il, out);
        Ok(out)
    }

    /// Slot `il`'s attention (block `ip`) over its Q, K and V inputs (normed, `[n_embd, n]`):
    /// gated, projected and through the post conv, plus the residual `res` when given.
    fn ink_attn<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        ip: usize,
        inputs: &[Tensor<'a>; 3],
        res: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ink();
        let hp = &self.hp;
        let n = pass.n;
        let (d, nh, nkv) = (hp.head_dim, hp.n_head, hp.n_head_kv);
        let kvw = d * nkv;
        let swa = p.swa[ip];
        let es = pass.pos.element_size();
        let pos_all = ctx.view_2d(pass.pos, [n, 1], n as usize * es, 0);
        let pos_of =
            |off: i64, ng: i64| ctx.view_2d(pass.pos, [ng, 1], ng as usize * es, off as usize * es);
        let [xq, xk, xv] = *inputs;

        let mut rows = Bf16Rows::default();
        // Q, the relative weights and the gate read xq, a sliding layer's K and V one input:
        // a node a set where it takes them (ink_mm_set), unless traced.
        let [q, r, g] = self.ink_mm_set(
            ctx,
            ip,
            ["attn_q.weight", "attn_rel.weight", "attn_gate.weight"],
            xq,
            &mut rows,
            tap.on,
        )?;
        // K and V one set, from one input or (a global layer's) two (#1221 r9), and so the
        // global layers' conv projections.
        let [kp, vp] = self.ink_mm_setx(
            ctx,
            ip,
            ["attn_k.weight", "attn_v.weight"],
            [xk, xv],
            &mut rows,
            tap.on,
        )?;
        let (kdyn, vdyn) = if swa {
            (None, None)
        } else {
            let [kd, vd] = self.ink_mm_setx(
                ctx,
                ip,
                ["attn_k_conv_proj.weight", "attn_v_conv_proj.weight"],
                [xk, xv],
                &mut rows,
                tap.on,
            )?;
            (Some(kd), Some(vd))
        };
        for (name, x) in [("q_proj", q), ("k_proj", kp), ("v_proj", vp)] {
            tap.tap(name, il, x);
        }
        for (name, x) in [("k_dyn", kdyn), ("v_dyn", vdyn)] {
            if let Some(x) = x {
                tap.tap(name, il, x);
            }
        }
        // A decode step (one group, a token a sequence) makes K and V and writes them to the
        // caches in one node, ops::kv, of the bits of the nodes below (#1221 r6). Traced
        // runs keep those for their taps.
        let (kcw, vcw) = (
            self.lw(ip, "attn_k_conv.weight")?,
            self.lw(ip, "attn_v_conv.weight")?,
        );
        let mut gs = pass.groups();
        let decode = match (gs.next(), gs.next()) {
            (Some((_, sh)), None)
                if !tap.on
                    && sh.n_tok == 1
                    && d <= 128
                    && kvw % d == 0
                    && kcw.ne()[0] == vcw.ne()[0]
                    && kcw.ne()[0] <= ops::MAX_TAPS as i64 =>
            {
                Some((sh.slot, sh.n_seqs))
            }
            _ => None,
        };
        let convs = match decode {
            Some(_) => None,
            None => {
                let k = self.ink_conv(pass, il, ip, "attn_k_conv.weight", kp, kdyn, 0)?;
                let v = self.ink_conv(pass, il, ip, "attn_v_conv.weight", vp, vdyn, kvw)?;
                tap.tap("k_conv", il, k);
                tap.tap("v_conv", il, v);
                Some((k, v))
            }
        };
        tap.tap("r_proj", il, r);

        // The global layers' Q and relative weights scale with the log of the position.
        let tau = (!swa).then_some((pos_all, p.log_scale));
        let kv = convs.map(|(k, v)| {
            (
                ops::clip(ctx, ctx.reshape_3d(k, d, nkv, n), None, p.qk_clip, None),
                ops::rms(ctx, ctx.reshape_3d(v, d, nkv, n), None, hp.rms_eps),
            )
        });
        // Q's and the relative weights' clips: one node of their bits, unless traced.
        let q = (
            ctx.reshape_3d(q, d, nh, n),
            self.lw(ip, "attn_q_clip.weight")?,
            p.qk_clip,
        );
        let r = (
            ctx.reshape_3d(r, p.n_rel, nh, n),
            self.lw(ip, "attn_rel_clip.weight")?,
            p.rel_clip,
        );
        let [q, r] = if tap.on || d > 128 || p.n_rel > 128 {
            [q, r].map(|(x, s, thr)| ops::clip(ctx, x, Some(s), thr, tau))
        } else {
            ops::clip2(ctx, [q, r], tau)
        };
        let [sinks] = self.ink_mm_set(ctx, ip, ["attn_sink_proj.weight"], xq, &mut rows, tap.on)?;
        tap.tap("r_clip", il, r);
        tap.tap("sinks", il, sinks);
        tap.tap("Qcur", il, q);

        let (kc, vc) = pass.kv.pools(il);
        let cells = pass.write_cells(il);
        let written = match (kv, decode) {
            (Some((k, v)), _) => {
                tap.tap("Kcur", il, k);
                tap.tap("Vcur", il, v);
                let kw = ds41_ops::cache_write(
                    ctx,
                    kc,
                    ctx.reshape_2d(k, kvw, n),
                    cells,
                    pos_all,
                    0,
                    1,
                    &[],
                );
                ds41_ops::cache_write(
                    ctx,
                    vc,
                    ctx.reshape_2d(v, kvw, n),
                    cells,
                    pos_all,
                    0,
                    1,
                    &[kw],
                )
            }
            (None, Some((slot, n_seqs))) => {
                let k = kcw.ne()[0];
                let taps = if kdyn.is_some() { 2 * k } else { k };
                ops::kv(
                    ctx,
                    kp,
                    vp,
                    kcw,
                    vcw,
                    pass.rec.ple_slots(ctx, 0, slot, n_seqs),
                    [kc, vc],
                    cells,
                    pos_all,
                    kdyn.zip(vdyn).map(|(a, b)| [a, b]),
                    il as i64 * p.slot_width,
                    d,
                    1.0 / ((taps + 1) as f32).sqrt(),
                    p.qk_clip,
                    hp.rms_eps,
                )
                .1
            }
            (None, None) => unreachable!("K and V come from the convs or ops::kv"),
        };
        pass.graph.borrow_mut().expand(written);

        let rel = self.lw(ip, "attn_rel_bias.weight")?;
        let scale = 1.0 / (d as f32).sqrt();
        let window = if swa { p.window } else { 0 };
        let mut outs = Vec::new();
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let ng = sh.tokens() as i64;
            let (table, ring) = match pass.attn[gi] {
                AttnInputs::Paged { table, ring, .. } => (table, ring),
                _ => {
                    return Err(ModelError::Unsupported(
                        "Inkling-2's attention on the gather path".into(),
                    ));
                }
            };
            // A ring layer's window through the sequences' rings.
            let ring = ring.filter(|_| pass.kv.is_ring(il)).map(|r| (r, sh.ring));
            let view = |t: Tensor<'a>, rows: i64| {
                let nb = t.nb();
                ctx.view_3d(t, [rows, nh, ng], [nb[1], nb[2]], off as usize * nb[2])
            };
            let o = ops::attn(
                ctx,
                view(q, d),
                kc,
                vc,
                table,
                ring,
                pos_of(off, ng),
                view(r, p.n_rel),
                rel,
                cols(ctx, sinks, off, ng),
                sh.n_tok as i64,
                scale,
                window,
                written,
            );
            outs.push(ctx.reshape_2d(o, d * nh, ng));
        }
        let o = glm_ops::join_cols(ctx, &outs);
        tap.tap("kqv_out", il, o);
        tap.tap("attn_gate", il, g);
        let o = ops::sigmul(ctx, o, g);
        tap.tap("attn_gated", il, o);
        let [o] = self.ink_mm_set(
            ctx,
            ip,
            ["attn_output.weight"],
            o,
            &mut Bf16Rows::default(),
            tap.on,
        )?;
        tap.tap("attn_proj", il, o);
        self.ink_conv_add(pass, il, ip, "attn_post_conv.weight", o, None, 2 * kvw, res)
    }

    /// The routed experts of block `ip` over the normed `h` (`[n_embd, n]`).
    fn ink_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        ip: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ink();
        let hp = &self.hp;
        let n = h.ne()[1];
        let logits = self.ink_mm(ctx, ip, "ffn_gate_inp.weight", h)?;
        tap.tap("ffn_moe_logits", il, logits);
        let bias = self.lw(ip, "exp_probs_b.bias")?;
        // The router's top-k needs `n_expert` logits and biases (`parse` checks k against it).
        if logits.ne()[0] != hp.n_expert || bias.elements() as i64 != hp.n_expert {
            return Err(ModelError::Layout(format!(
                "inkling2: block {ip}'s router has {} experts and {} biases, want {}",
                logits.ne()[0],
                bias.elements(),
                hp.n_expert
            )));
        }
        let ids = glm_ops::router_ids(ctx, logits, bias, hp.n_expert_used);
        // Tests only: slot `il`'s ids from the forced route (see [`FORCED_ROUTE`]), the
        // router's own traced as `ffn_moe_router`.
        #[cfg(test)]
        let ids = match pass.group.first().and_then(|g| g.first()) {
            Some(&f) => {
                tap.tap("ffn_moe_router", il, ids);
                let k = hp.n_expert_used;
                assert_eq!(
                    f.ne()[0],
                    p.n_slots() as i64 * k * n,
                    "forced route: length"
                );
                ctx.view_2d(f, [k, n], 4 * k as usize, il * 4 * (k * n) as usize)
            }
            None => ids,
        };
        tap.tap("ffn_moe_topk", il, ids);
        let w = glm_ops::router_w(ctx, logits, ids, true, p.weights_scale);
        tap.tap("ffn_moe_weights", il, w);
        let x = ctx.reshape_3d(h, hp.n_embd, 1, n);
        let gu = self.ink_experts(ctx, ip, "ffn_gate_up_exps", x, ids, None)?;
        tap.tap("ffn_moe_gate_up", il, gu);
        // An NVFP4 down projection rounds its input to BF16: SITU stores it so (the same
        // bits, one cast kernel less).
        let flags = experts_flags();
        let bf16 = flags & EXPERTS_BF16_ACT != 0
            && self
                .w(&format!("blk.{ip}.ffn_down_exps.weight"))
                .is_ok_and(|w| w.ty() == types::NVFP4);
        let a = ops::situ(ctx, gu, p.situ_beta, p.situ_lb, bf16);
        // Traced as F32, as before.
        if bf16 && tap.on {
            tap.tap("ffn_moe_act", il, ctx.cast(a, types::F32));
        } else {
            tap.tap("ffn_moe_act", il, a);
        }
        let y = self.ink_experts(ctx, ip, "ffn_down_exps", a, ids, Some(gu))?;
        tap.tap("ffn_moe_down", il, y);
        Ok(glm_ops::weighted_sum(ctx, y, w))
    }

    /// `mul_mat_id` of block `ip`'s expert stack `base` (NVFP4, or what [`ops::mm_id`] takes)
    /// over `x` (F32 `[k, 1 or n_used, n]`, or BF16 for NVFP4) for the choices `ids`: F32
    /// `[m, n_used, n]`. `gate_up`: the down projection's, the gate and up rows of the same
    /// routing it follows. NVFP4 experts take [`nvfp4_mm::mul_mat_id_mv`] as
    /// [`experts_flags`] and [`expert_mv`] say, the down projection with `plan_of` those rows.
    fn ink_experts<'a>(
        &'a self,
        ctx: &'a Context,
        ip: usize,
        base: &str,
        x: Tensor<'a>,
        ids: Tensor<'a>,
        gate_up: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let name = format!("blk.{ip}.{base}.weight");
        let w = self.w(&name)?;
        let scale = self.w(&format!("blk.{ip}.{base}.scale")).ok();
        if w.ty() == types::NVFP4 {
            // `mul_mat_id` asserts an F32 `[n_expert, 1, 1, 1]` scale: refuse anything else.
            let bad_scale = |s: Tensor<'_>| s.ty() != types::F32 || s.ne() != [w.ne()[2], 1, 1, 1];
            if !nvfp4_mm::supports(w) || scale.is_some_and(bad_scale) {
                return Err(ModelError::Unsupported(format!(
                    "`{name}`: NVFP4 experts {:?} on this device",
                    w.ne()
                )));
            }
            let flags = experts_flags();
            let tile = match gate_up {
                Some(_) => flags & EXPERTS_TILE != 0,
                None => flags & EXPERTS_TILE_GU != 0,
            };
            let plan_of = gate_up.filter(|_| flags & EXPERTS_PLAN != 0);
            // An A/B pick, else the down's one-ahead prefetch (unless EXPERTS_TILE's two).
            let down_t1 =
                gate_up.is_some() && flags & (EXPERTS_TILE | EXPERTS_DOWN_T1) == EXPERTS_DOWN_T1;
            let mv = expert_mv(gate_up.is_some()).or(down_t1.then_some(EXPERT_MV_T1));
            return Ok(nvfp4_mm::mul_mat_id_mv(
                ctx, w, x, ids, scale, tile, plan_of, mv,
            ));
        }
        if !ops::mm_id_takes(w, scale) {
            return Err(ModelError::Unsupported(format!(
                "`{name}`: experts of type {} {:?}",
                crate::ggml::type_name(w.ty()),
                w.ne()
            )));
        }
        Ok(ops::mm_id(ctx, w, scale, x, ids))
    }
}
