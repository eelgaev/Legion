//! One rank's share of a model for tensor parallelism over machines (#619): a synthetic
//! [`GgufFile`] over the same files whose layers have this rank's heads and FFN rows, so the
//! model loads (and the executor runs) as an ordinary smaller model and reads only its share
//! of the split weights. The Megatron layout of [`crate::model::core::tp`]:
//!
//! - Q, K and V and the FFN's gate and up: by output rows (axis 1), one contiguous byte range
//!   of the file each: whole heads, whole GQA groups (a KV head's query heads with it), FFN
//!   rows in quantization blocks.
//! - The attention output and the FFN's down projection: by input columns (axis 0), in whole
//!   quantization blocks, read and cut row by row at load.
//! - Everything else (embeddings, norms, per-head norm weights, the output head): mirrored.
//!
//! The metadata says what each rank has: `attention.head_count`, `attention.head_count_kv`
//! (per layer) and `feed_forward_length` (per layer) are this rank's.
//!
//! A Gemma 4 MTP assistant ([`assistant_view`]) is cut the same way: its query heads go with
//! the target's kv heads they attend over (each rank's share of the target's cache), its
//! attention output by those columns, its FFN as the target's.
//!
//! Gemma 4 dense models (31B): every layer's KV heads must divide among the ranks. Gemma's
//! mixture-of-experts layers would need expert parallelism, and per-layer embeddings a split of
//! their projections. Qwen3.8-Flash-Next (`qwen4exp`) and its MTP head: [`qwen4`].
//!
//! A safetensors checkpoint's tensors that are computed at load (NVFP4
//! repacks) are made whole on every rank and then cut, so each rank reads and repacks the
//! whole of them at load (the plain ones it reads only its share of); per-row weight scales
//! are refused.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ggml::{self, ffi};
use crate::gguf::{Compute, GgufFile, GgufWriter, Recipe, TensorInfo, Value};

pub mod qwen4;

/// FFN rows go in multiples of this when they divide evenly (kernels' preferred tile).
const FFN_GRANULE: u64 = 256;

/// How one tensor is cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    Mirrored,
    /// Rows `start..start + len` (axis 1).
    Rows {
        start: u64,
        len: u64,
    },
    /// Columns `start..start + len` of every row (axis 0).
    Cols {
        start: u64,
        len: u64,
    },
}

/// Per layer: `(head_dim, n_head_kv)`, and the model's query heads.
pub struct Heads {
    pub layers: Vec<(u64, u64)>,
    pub n_head: u64,
    pub n_ff: Vec<u64>,
}

fn even(what: &str, total: u64, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let n = n as u64;
    if total == 0 || !total.is_multiple_of(n) {
        return Err(format!("{what}: {total} do not divide among {n} ranks"));
    }
    Ok((rank as u64 * total / n, total / n))
}

/// Rank `rank`'s FFN rows of `n_ff` over `n` ranks: whole [`FFN_GRANULE`]s when they divide
/// evenly, else in whole blocks of `block` (which every rank must get some of).
pub fn ffn_rows(n_ff: u64, block: u64, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let g = if n_ff.is_multiple_of(FFN_GRANULE * n as u64) {
        FFN_GRANULE
    } else {
        block.max(1)
    };
    let shares = ggml::meta::shares(n_ff as i64, n, g as i64);
    let start: i64 = shares[..rank].iter().sum();
    let len = shares[rank];
    if shares.iter().any(|&s| s <= 0) {
        return Err(format!(
            "feed_forward_length: {n_ff} rows do not split among {n} ranks in blocks of {g}"
        ));
    }
    Ok((start as u64, len as u64))
}

/// How rank `rank` of `n` cuts tensor `t` (see the module docs).
pub fn cut(heads: &Heads, t: &TensorInfo, n: usize, rank: usize) -> Result<Cut, String> {
    let Some(rest) = t.name.strip_prefix("blk.") else {
        return Ok(Cut::Mirrored);
    };
    let mut parts = rest.splitn(2, '.');
    let (Some(il), Some(name)) = (
        parts.next().and_then(|s| s.parse::<usize>().ok()),
        parts.next(),
    ) else {
        return Ok(Cut::Mirrored);
    };
    let Some(&(d, n_kv)) = heads.layers.get(il) else {
        return Err(format!("{}: no such layer", t.name));
    };
    let block = unsafe { ffi::ggml_blck_size(t.ggml_type) } as u64;
    let ne0 = t.shape.first().copied().unwrap_or(1);
    let ne1 = t.shape.get(1).copied().unwrap_or(1);
    let want = |ok: bool, what: &str| -> Result<(), String> {
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{}: shape {:?} is not the {what} this layer's heads give",
                t.name, t.shape
            ))
        }
    };
    let n_ff = heads.n_ff.get(il).copied().unwrap_or(0);
    let cut = match name {
        "attn_q.weight" => {
            want(ne1 == heads.n_head * d, "queries")?;
            let (start, len) = even(&format!("layer {il} query heads"), heads.n_head, n, rank)?;
            even(&format!("layer {il} kv heads"), n_kv, n, rank)?;
            Cut::Rows {
                start: start * d,
                len: len * d,
            }
        }
        "attn_k.weight" | "attn_v.weight" => {
            want(ne1 == n_kv * d, "keys and values")?;
            let (start, len) = even(&format!("layer {il} kv heads"), n_kv, n, rank)?;
            Cut::Rows {
                start: start * d,
                len: len * d,
            }
        }
        "attn_output.weight" => {
            want(ne0 == heads.n_head * d, "attention output")?;
            let (start, len) = even(&format!("layer {il} query heads"), heads.n_head, n, rank)?;
            Cut::Cols {
                start: start * d,
                len: len * d,
            }
        }
        "ffn_gate.weight" | "ffn_up.weight" => {
            want(ne1 == n_ff, "FFN rows")?;
            let (start, len) = ffn_rows(n_ff, block, n, rank)?;
            Cut::Rows { start, len }
        }
        "ffn_down.weight" => {
            want(ne0 == n_ff, "FFN rows")?;
            let (start, len) = ffn_rows(n_ff, block, n, rank)?;
            Cut::Cols { start, len }
        }
        // A weight's quantization scale (a safetensors checkpoint's): one per tensor is the
        // same for every share, one per row (channel-wise FP8) would need cutting with it.
        s if (s.ends_with(".scale") || s.ends_with(".input_scale"))
            && t.shape.iter().product::<u64>() > 1 =>
        {
            return Err(format!(
                "{}: per-row weight scales cannot be split yet (one scale per tensor only)",
                t.name
            ));
        }
        _ if t.shape.len() <= 1 => Cut::Mirrored,
        _ => {
            return Err(format!(
                "{}: a layer tensor tensor parallelism over machines cannot split yet (dense \
                 Gemma 4 layers only: no experts or per-layer embeddings)",
                t.name
            ));
        }
    };
    if let Cut::Cols { start, len } = cut
        && (!start.is_multiple_of(block) || !len.is_multiple_of(block))
    {
        return Err(format!(
            "{}: columns {start}..{} are not whole blocks of {block}",
            t.name,
            start + len
        ));
    }
    Ok(cut)
}

/// The model's heads as the shard needs them (Gemma 4 metadata).
fn heads_of(gguf: &GgufFile) -> Result<Heads, String> {
    let hp = crate::model::core::Hparams::from_gguf(gguf).map_err(|e| e.to_string())?;
    let g = crate::model::gemma4::params(&hp).ok_or("not a Gemma 4 model")?;
    Ok(Heads {
        layers: (0..hp.n_layer)
            .map(|il| {
                let (d, n_kv) = g.kv_dims(il);
                (d as u64, n_kv as u64)
            })
            .collect(),
        n_head: hp.n_head as u64,
        n_ff: g.n_ff.iter().map(|&f| f as u64).collect(),
    })
}

/// What tensor parallelism over machines cannot take (yet), before anything is cut.
fn check(gguf: &GgufFile) -> Result<(), String> {
    let arch = gguf.architecture().unwrap_or_default();
    if arch != "gemma4" {
        return Err(format!(
            "architecture {arch:?}: dense Gemma 4 and qwen4exp only for now"
        ));
    }
    let n = |k: &str| gguf.get_u64(&format!("gemma4.{k}")).unwrap_or(0);
    if n("expert_count") > 0 {
        return Err("a mixture-of-experts model: experts need expert parallelism".into());
    }
    if n("embedding_length_per_layer_input") > 0 {
        return Err("per-layer embeddings: not split yet".into());
    }
    Ok(())
}

/// Sets `{arch}.{k}` in `meta`.
fn set(meta: &mut GgufWriter, arch: &str, k: &str, v: Value) -> Result<(), String> {
    meta.set(&format!("{arch}.{k}"), &v)
        .map(|_| ())
        .map_err(|e| format!("{k}: {e}"))
}

/// Rank `rank`'s share of `gguf` over `n` ranks (see the module docs).
pub fn view(gguf: &GgufFile, rank: usize, n: usize) -> Result<GgufFile, String> {
    if gguf.architecture().as_deref() == Some("qwen4exp") {
        return qwen4::view(gguf, rank, n);
    }
    check(gguf)?;
    let heads = heads_of(gguf)?;
    let kv: Vec<Value> = heads
        .layers
        .iter()
        .enumerate()
        .map(|(il, &(_, n_kv))| {
            even(&format!("layer {il} kv heads"), n_kv, n, rank).map(|r| Value::I32(r.1 as i32))
        })
        .collect::<Result<_, _>>()?;
    share(gguf, &heads, "gemma4", Some(kv), rank, n)
}

/// Rank `rank`'s share over `n` ranks of `gguf`, a Gemma 4 MTP assistant
/// (`gemma4-assistant`) of a target whose last sliding-window and last global layers have
/// `swa` and `global` (`(head_dim, n_head_kv)`, all ranks' kv heads): its query heads by the
/// target's kv heads (each rank's queries over its share of the target's cache) and its FFN
/// rows, as the target's layers are cut. The assistant's layers read the target's cache, so
/// the target must be split (by [`view`]) over the same ranks.
pub fn assistant_view(
    gguf: &GgufFile,
    swa: (u64, u64),
    global: (u64, u64),
    rank: usize,
    n: usize,
) -> Result<GgufFile, String> {
    const ARCH: &str = "gemma4-assistant";
    let arch = gguf.architecture().unwrap_or_default();
    if arch != ARCH {
        return Err(format!(
            "architecture {arch:?}: not a Gemma 4 assistant ({ARCH})"
        ));
    }
    let key = |k: &str| format!("{ARCH}.{k}");
    let n_layer = gguf
        .get_u64(&key("block_count"))
        .ok_or_else(|| key("block_count"))? as usize;
    let pattern: Vec<bool> = gguf
        .get(&key("attention.sliding_window_pattern"))
        .and_then(|v| {
            v.as_array()
                .map(|a| a.iter().map(|x| x.as_bool()).collect())
        })
        .and_then(|v: Option<Vec<bool>>| v)
        .filter(|v| v.len() >= n_layer)
        .ok_or_else(|| key("attention.sliding_window_pattern"))?;
    let n_ff: Vec<u64> = match gguf.get(&key("feed_forward_length")) {
        Some(v) => match v.as_array() {
            Some(a) => a.iter().filter_map(|x| x.as_u64()).collect(),
            None => vec![v.as_u64().ok_or_else(|| key("feed_forward_length"))?; n_layer],
        },
        None => Vec::new(),
    };
    if n_ff.len() < n_layer {
        return Err(key("feed_forward_length"));
    }
    let heads = Heads {
        layers: pattern[..n_layer]
            .iter()
            .map(|&s| if s { swa } else { global })
            .collect(),
        n_head: gguf
            .get_u64(&key("attention.head_count"))
            .ok_or_else(|| key("attention.head_count"))?,
        n_ff: n_ff[..n_layer].to_vec(),
    };
    share(gguf, &heads, ARCH, None, rank, n)
}

/// Rank `rank`'s share of `gguf` (architecture `arch`) over `n` ranks, cut by `heads`, with
/// `kv` its per-layer kv heads (the target's own layers; `None`: none, an assistant's).
fn share(
    gguf: &GgufFile,
    heads: &Heads,
    arch: &str,
    kv: Option<Vec<Value>>,
    rank: usize,
    n: usize,
) -> Result<GgufFile, String> {
    let n_layer = heads.layers.len();
    let mut meta = GgufWriter::new();
    for (k, v) in gguf.metadata() {
        meta.set(&k, &v).map_err(|e| format!("{k}: {e}"))?;
    }
    let (_, heads_here) = even("query heads", heads.n_head, n, rank)?;
    set(
        &mut meta,
        arch,
        "attention.head_count",
        Value::U32(heads_here as u32),
    )?;
    if let Some(kv) = kv {
        set(&mut meta, arch, "attention.head_count_kv", Value::Array(kv))?;
    }
    let paths: Vec<PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut tensors = Vec::new();
    let mut recipes = HashMap::new();
    let mut ffn_here = vec![0u64; n_layer];
    for t in gguf.tensors() {
        let c = cut(heads, &t, n, rank)?;
        let mut s = t.clone();
        let recipe = gguf.recipe(&t.name);
        match c {
            Cut::Mirrored => {
                if let Some(r) = recipe {
                    recipes.insert(t.name.clone(), r.clone());
                }
            }
            Cut::Rows { start, len } => {
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize) as u64;
                s.shape[1] = len;
                s.size = len * row;
                if recipe.is_none() {
                    s.offset = t.offset + start * row;
                } else {
                    // A safetensors checkpoint's computed tensor (a repacked NVFP4 matrix):
                    // made whole, then this rank's rows kept.
                    let whole = whole_bytes(gguf, &paths, &t);
                    let (at, take) = ((start * row) as usize, s.size as usize);
                    let read = move || -> Result<Vec<u8>, String> {
                        let mut b = whole()?;
                        b.truncate(at + take);
                        Ok(b.split_off(at))
                    };
                    recipes.insert(t.name.clone(), Recipe::Computed(Arc::new(read)));
                }
            }
            Cut::Cols { start, len } => {
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
                let at = ggml::row_size(t.ggml_type, start as usize);
                let take = ggml::row_size(t.ggml_type, len as usize);
                let rows = (t.size as usize) / row;
                s.shape[0] = len;
                s.size = (rows * take) as u64;
                let whole = whole_bytes(gguf, &paths, &t);
                let read = move || -> Result<Vec<u8>, String> {
                    let whole = whole()?;
                    let mut out = Vec::with_capacity(rows * take);
                    for r in whole.chunks_exact(row) {
                        out.extend_from_slice(&r[at..at + take]);
                    }
                    Ok(out)
                };
                recipes.insert(t.name.clone(), Recipe::Computed(Arc::new(read)));
            }
        }
        if let Some(il) = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
            && t.name.ends_with(".ffn_down.weight")
        {
            ffn_here[il] = s.shape[0];
        }
        tensors.push(s);
    }
    // This rank's FFN rows per layer, as the down projections were cut.
    if let Some(il) = ffn_here.iter().position(|&f| f == 0) {
        return Err(format!("layer {il}: no blk.{il}.ffn_down.weight"));
    }
    let ffn: Vec<Value> = ffn_here.iter().map(|&f| Value::I32(f as i32)).collect();
    set(&mut meta, arch, "feed_forward_length", Value::Array(ffn))?;
    let mut out = GgufFile::synthetic(meta, gguf.path().to_path_buf(), paths, tensors, recipes)
        .map_err(|e| e.to_string())?;
    out.set_checkpoint(gguf.checkpoint());
    Ok(out)
}

/// All of `t`'s bytes in `gguf` (backed by `paths`), as the loader would make them: its recipe
/// (a safetensors checkpoint's), else its byte range. Checked against its size.
fn whole_bytes(gguf: &GgufFile, paths: &[PathBuf], t: &TensorInfo) -> Compute {
    use std::os::unix::fs::FileExt;
    let read = |path: &PathBuf, offset: u64, len: u64| -> Result<Vec<u8>, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut b = vec![0u8; len as usize];
        f.read_exact_at(&mut b, offset)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(b)
    };
    let (name, size) = (t.name.clone(), t.size as usize);
    let make: Compute = match gguf.recipe(&t.name) {
        Some(Recipe::Computed(f)) => f.clone(),
        Some(Recipe::Gather(parts)) => {
            let (parts, paths) = (parts.clone(), paths.to_vec());
            Arc::new(move || {
                let mut out = Vec::with_capacity(size);
                for p in &parts {
                    out.extend(read(&paths[p.file], p.offset, p.len)?);
                }
                Ok(out)
            })
        }
        None => {
            let (path, offset) = (paths[t.file_index].clone(), t.offset);
            Arc::new(move || read(&path, offset, size as u64))
        }
    };
    Arc::new(move || {
        let b = make()?;
        if b.len() != size {
            return Err(format!("{name}: {} bytes, expected {size}", b.len()));
        }
        Ok(b)
    })
}
