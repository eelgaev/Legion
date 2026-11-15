//! One rank's share of a dense Qwen3.5+ (`qwen35`) GGUF, or of its MTP head, for tensor
//! parallelism (`--tp 2` over a GPU pair, #1163): the layers' heads, DeltaNet head groups and
//! FFN rows of this rank, with the metadata of the smaller model they make, so the qwen35 graph
//! runs on them unchanged and sums the attention's (or DeltaNet's) and the FFN's partial
//! outputs (`Model::tp_sum`, two all-reduces per layer).
//!
//! - **Gated attention and gated DeltaNet:** cut as Qwen3.8-Flash-Next's
//!   ([`super::qwen4::piece`]): the same blocks of the same Qwen3-Next stack.
//! - **FFN (SwiGLU):** `ffn_gate`/`ffn_up` by rows, `ffn_down` by the same columns, in whole
//!   [`super::ffn_rows`] (256-row granules when they divide evenly, else whole blocks of the
//!   down projection's type).
//! - Everything else (embeddings, norms, the output head, the MTP head's `nextn.*`): mirrored.
//!
//! The mixture-of-experts sibling (`qwen35moe`) is refused: its experts would need expert
//! parallelism.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;

use super::qwen4::{self, Dims, Piece};
use crate::ggml::ffi;
use crate::gguf::{GgufFile, GgufWriter, TensorInfo, Value};
use crate::model::core::Hparams;

const ARCH: &str = "qwen35";

/// Rank `rank`'s share of `gguf`, a dense `qwen35` model, over `n` ranks (see the module docs).
pub fn view(gguf: &GgufFile, rank: usize, n: usize) -> Result<GgufFile, String> {
    arch(gguf)?;
    let d = Dims::of_arch(gguf, ARCH)?;
    let here = d.share(n, rank)?;
    let mut meta = copy_meta(gguf)?;
    let u = |v: u64| Value::U32(v as u32);
    set(&mut meta, "attention.head_count", u(here.n_head))?;
    set(&mut meta, "attention.head_count_kv", u(here.n_head_kv))?;
    set(&mut meta, "ssm.group_count", u(here.n_k))?;
    set(&mut meta, "ssm.time_step_rank", u(here.n_v))?;
    set(&mut meta, "ssm.inner_size", u(here.n_v * here.head_v))?;
    cut(gguf, &d, meta, rank, n)
}

/// Rank `rank`'s share over `n` ranks of `gguf`, the MTP head of its own file (one more block
/// of the target's shape) of a target whose share has hyperparameters `hp`: cut as the
/// target's layers are. The head's file carries no shapes of its own (the target's apply), so
/// they come from `hp` (all ranks': this rank's times `n`).
pub fn head_view(gguf: &GgufFile, hp: &Hparams, rank: usize, n: usize) -> Result<GgufFile, String> {
    arch(gguf)?;
    let d = full_dims(hp, n);
    d.share(n, rank)?;
    let meta = copy_meta(gguf)?;
    cut(gguf, &d, meta, rank, n)
}

/// All ranks' shapes of a model whose rank's share over `n` ranks has hyperparameters `hp`.
pub fn full_dims(hp: &Hparams, n: usize) -> Dims {
    let n = n as u64;
    let n_v = hp.ssm_dt_rank.max(1) as u64;
    Dims {
        n_head: hp.n_head as u64 * n,
        n_head_kv: hp.n_head_kv as u64 * n,
        head_dim: hp.head_dim as u64,
        n_k: hp.ssm_n_group as u64 * n,
        head_k: hp.ssm_d_state as u64,
        n_v: n_v * n,
        head_v: hp.ssm_d_inner as u64 / n_v,
    }
}

fn arch(gguf: &GgufFile) -> Result<(), String> {
    match gguf.architecture().unwrap_or_default().as_str() {
        ARCH => Ok(()),
        "qwen35moe" => Err(
            "architecture \"qwen35moe\": a mixture-of-experts model: experts need expert \
             parallelism"
                .into(),
        ),
        a => Err(format!("architecture {a:?}: not {ARCH}")),
    }
}

fn copy_meta(gguf: &GgufFile) -> Result<GgufWriter, String> {
    let mut meta = GgufWriter::new();
    for (k, v) in gguf.metadata() {
        meta.set(&k, &v).map_err(|e| format!("{k}: {e}"))?;
    }
    Ok(meta)
}

/// Sets `qwen35.{k}` in `meta`.
fn set(meta: &mut GgufWriter, k: &str, v: Value) -> Result<(), String> {
    meta.set(&format!("{ARCH}.{k}"), &v)
        .map(|_| ())
        .map_err(|e| format!("{k}: {e}"))
}

/// How rank `rank` of `n` cuts tensor `t` of a model of shapes `d`; `down` gives each layer's
/// FFN down projection's type, whose blocks the FFN rows go in.
pub fn piece(
    d: &Dims,
    t: &TensorInfo,
    down: &dyn Fn(usize) -> Option<ffi::ggml_type>,
    n: usize,
    rank: usize,
) -> Result<Piece, String> {
    let layer = t.name.strip_prefix("blk.").and_then(|r| {
        let (il, name) = r.split_once('.')?;
        Some((il.parse::<usize>().ok()?, name))
    });
    if let Some((il, name)) = layer
        && matches!(
            name,
            "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight"
        )
    {
        let ty = down(il).ok_or_else(|| format!("{}: no blk.{il}.ffn_down.weight", t.name))?;
        let block = unsafe { ffi::ggml_blck_size(ty) } as u64;
        let ne0 = t.shape.first().copied().unwrap_or(1);
        let ne1 = t.shape.get(1).copied().unwrap_or(1);
        return Ok(if name == "ffn_down.weight" {
            Piece::Cols(vec![super::ffn_rows(ne0, block, n, rank)?])
        } else {
            Piece::Rows(vec![super::ffn_rows(ne1, block, n, rank)?])
        });
    }
    qwen4::piece(d, t, &|_, _| None, n, rank)
}

/// Every tensor of `gguf` cut by `d` into rank `rank`'s share over `n`, with metadata `meta`
/// (and this rank's `feed_forward_length`).
fn cut(
    gguf: &GgufFile,
    d: &Dims,
    mut meta: GgufWriter,
    rank: usize,
    n: usize,
) -> Result<GgufFile, String> {
    let tensors = gguf.tensors();
    let types: HashMap<String, ffi::ggml_type> = tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type))
        .collect();
    let down = |il: usize| types.get(&format!("blk.{il}.ffn_down.weight")).copied();
    let paths: Vec<PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut out = Vec::with_capacity(tensors.len());
    let mut recipes = HashMap::new();
    let mut n_ff = None;
    for t in tensors {
        let p = piece(d, &t, &down, n, rank)?;
        let s = qwen4::apply(gguf, &paths, &t, p, &mut recipes)?;
        if t.name.ends_with(".ffn_down.weight") {
            n_ff.get_or_insert(s.shape[0]);
        }
        out.push(s);
    }
    if let Some(f) = n_ff
        && gguf.get(&format!("{ARCH}.feed_forward_length")).is_some()
    {
        set(&mut meta, "feed_forward_length", Value::U32(f as u32))?;
    }
    let mut share = GgufFile::synthetic(meta, gguf.path().to_path_buf(), paths, out, recipes)
        .map_err(|e| e.to_string())?;
    share.set_checkpoint(gguf.checkpoint());
    Ok(share)
}
