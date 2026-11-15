//! One rank's share of a GLM-5.3-Flash (`glm5-next`) model, its MTP block included, for tensor
//! parallelism over machines (#560): the layers' attention heads, KDA heads and FFN rows of this
//! rank, with the metadata of the smaller model they make, so the GLM graph runs on them
//! unchanged and sums the attention's and the FFN's partial outputs (`Model::tp_sum`, two
//! all-reduces per layer). The cuts are those of the in-process head split
//! (`--glm-parallel-decode`, [`crate::model::glm5next::par::layer_cuts`]), with the routed
//! experts split by rows too:
//!
//! - **KDA by heads:** rank `r` of `n` has heads `r·H/n ..` of q, k, v, their convolutions, the
//!   gates' second products (`f_b`, `g_b`), beta, `dt` and `A`, and those heads' columns of the
//!   output projection; `f_a`, `g_a` and the head norm are mirrored. Its recurrent states are
//!   its heads' only (the KDA width comes from `attn_q`'s rows).
//! - **MLA by heads:** the query and latent down-projections, their norms and the indexer are
//!   mirrored, so every rank caches the same latents and indexer rows and selects the same
//!   positions; `q_b`, `k_b`, `v_b` (one matrix per head) and the output projection's columns
//!   go by heads (`attention.head_count` is this rank's).
//! - **FFNs by rows:** the dense FFN, the shared expert and every routed expert: gate and up by
//!   rows, down by columns, in whole blocks of the down projection's type. The router and its
//!   bias are mirrored, so every rank routes alike and the work is even whatever the routing.
//!   A checkpoint's NVFP4 experts are read and repacked per expert, this rank's rows only for
//!   gate and up ([`nvfp4_rows`]); the down projection's columns are a slice of every row, so
//!   each expert's whole down matrix is read and its columns repacked ([`nvfp4_cols`]).
//! - Everything else (embeddings, hyper-connection mixers, norms, the output head, the MTP
//!   block's `nextn.*`): mirrored.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ggml::{self, ffi};
use crate::gguf::{
    Compute, FileRange, GgufFile, GgufWriter, Nvfp4Stack, Recipe, TensorInfo, Value,
};

const ARCH: &str = "glm5-next";

/// How one tensor is cut: `(start, len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Mirrored,
    /// Items of the outermost axis: a matrix's rows, a stack's matrices (one per head), a
    /// vector's values.
    Outer(u64, u64),
    /// Rows (axis 1) of every matrix of a stack of experts.
    Rows(u64, u64),
    /// Columns (axis 0) of every row.
    Cols(u64, u64),
}

/// The model's shapes the cut goes by (all ranks').
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dims {
    /// MLA query heads and their key and value widths.
    pub n_head: u64,
    pub head_k: u64,
    pub head_v: u64,
    /// KDA head width.
    pub kda_head: u64,
    /// Per layer (the MTP block's included): an MLA layer, else KDA.
    pub mla: Vec<bool>,
}

impl Dims {
    /// From a glm5-next GGUF's metadata.
    pub fn of(gguf: &GgufFile) -> Result<Self, String> {
        let key = |k: &str| format!("{ARCH}.{k}");
        let u = |k: &str| {
            gguf.get_u64(&key(k))
                .ok_or_else(|| format!("{}: missing", key(k)))
        };
        let n_all = u("block_count")? as usize;
        let kv = gguf
            .get(&key("attention.head_count_kv"))
            .ok_or_else(|| format!("{}: missing", key("attention.head_count_kv")))?;
        let mla: Vec<bool> = match kv.as_array() {
            Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(0.0) > 0.0).collect(),
            None => vec![kv.as_f64().unwrap_or(0.0) > 0.0; n_all],
        };
        if mla.len() < n_all {
            return Err(format!(
                "{}: {} entries for {n_all} layers",
                key("attention.head_count_kv"),
                mla.len()
            ));
        }
        Ok(Self {
            n_head: u("attention.head_count")?,
            head_k: u("attention.key_length_mla")?,
            head_v: u("attention.value_length_mla")?,
            kda_head: u("kda.head_dim")?,
            mla,
        })
    }
}

/// Rank `rank`'s `(start, len)` of `total` items over `n` ranks: an even split.
fn even(what: &str, total: u64, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let n = n as u64;
    if total == 0 || !total.is_multiple_of(n) {
        return Err(format!("{what}: {total} do not divide among {n} ranks"));
    }
    Ok((rank as u64 * total / n, total / n))
}

/// Rank `rank`'s share of an FFN of `n_ff` rows over `n` ranks whose down projection is of
/// type `down` (its columns must be whole blocks).
fn ffn_share(n_ff: u64, down: ffi::ggml_type, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let block = unsafe { ffi::ggml_blck_size(down) } as u64;
    let (start, per) = even("FFN rows", n_ff, n, rank)?;
    if !per.is_multiple_of(block) {
        return Err(format!(
            "{n_ff} FFN rows do not split among {n} ranks in whole blocks of {block} \
             (the down projection's {})",
            ggml::type_name(down)
        ));
    }
    Ok((start, per))
}

/// How rank `rank` of `n` cuts tensor `t` (see the module docs). `kda_heads(il)`: layer `il`'s
/// KDA heads; `down(il, name)`: the type of layer `il`'s tensor `name` (an FFN's down
/// projection, whose blocks the FFN rows go in).
pub fn piece(
    d: &Dims,
    t: &TensorInfo,
    kda_heads: &dyn Fn(usize) -> Option<u64>,
    down: &dyn Fn(usize, &str) -> Option<ffi::ggml_type>,
    n: usize,
    rank: usize,
) -> Result<Piece, String> {
    let Some(rest) = t.name.strip_prefix("blk.") else {
        return Ok(Piece::Mirrored);
    };
    let mut parts = rest.splitn(2, '.');
    let (Some(il), Some(name)) = (
        parts.next().and_then(|s| s.parse::<usize>().ok()),
        parts.next(),
    ) else {
        return Ok(Piece::Mirrored);
    };
    let Some(&mla) = d.mla.get(il) else {
        return Err(format!("{}: no such layer", t.name));
    };
    let ne0 = t.shape.first().copied().unwrap_or(1);
    let ne1 = t.shape.get(1).copied().unwrap_or(1);
    let last = t.shape.last().copied().unwrap_or(1);
    let want = |ok: bool, what: &str| -> Result<(), String> {
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{}: shape {:?} is not the {what} the metadata gives",
                t.name, t.shape
            ))
        }
    };
    // `items` heads of `unit` along the outermost axis.
    let outer = |what: &str, items: u64, unit: u64| -> Result<Piece, String> {
        want(last == items * unit, what)?;
        let (s, l) = even(&format!("layer {il} {what}"), items, n, rank)?;
        Ok(Piece::Outer(s * unit, l * unit))
    };
    let cols = |what: &str, items: u64, unit: u64| -> Result<Piece, String> {
        want(ne0 == items * unit, what)?;
        let (s, l) = even(&format!("layer {il} {what}"), items, n, rank)?;
        Ok(Piece::Cols(s * unit, l * unit))
    };
    let kda = || kda_heads(il).ok_or_else(|| format!("{}: no blk.{il}.attn_q.weight", t.name));
    // An FFN's rows: `n_ff` (the gate's and up's rows, the down's columns) in whole blocks of
    // the layer's down projection `down_name`.
    let ffn = |n_ff: u64, down_name: &str| -> Result<(u64, u64), String> {
        let ty =
            down(il, down_name).ok_or_else(|| format!("{}: no blk.{il}.{down_name}", t.name))?;
        ffn_share(n_ff, ty, n, rank).map_err(|e| format!("{}: {e}", t.name))
    };
    let (dk, dv, dh) = (d.head_k, d.head_v, d.kda_head);
    Ok(match name {
        "attn_q_b.weight" if mla => outer("query heads", d.n_head, dk)?,
        "attn_k_b.weight" | "attn_v_b.weight" if mla => outer("heads", d.n_head, 1)?,
        "attn_output.weight" if mla => cols("attention output heads", d.n_head, dv)?,
        "attn_q.weight"
        | "attn_k.weight"
        | "attn_v.weight"
        | "ssm_conv1d_q.weight"
        | "ssm_conv1d_k.weight"
        | "ssm_conv1d_v.weight"
        | "ssm_f_b.weight"
        | "ssm_g_b.weight"
        | "ssm_dt.bias"
            if !mla =>
        {
            outer("KDA heads", kda()?, dh)?
        }
        "ssm_beta.weight" | "ssm_a" if !mla => outer("KDA heads", kda()?, 1)?,
        "attn_output.weight" => cols("KDA output heads", kda()?, dh)?,
        "ffn_gate.weight" | "ffn_up.weight" => {
            want(t.shape.len() == 2, "dense FFN")?;
            let (s, l) = ffn(ne1, "ffn_down.weight")?;
            Piece::Outer(s, l)
        }
        "ffn_down.weight" => {
            let (s, l) = ffn(ne0, "ffn_down.weight")?;
            Piece::Cols(s, l)
        }
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => {
            want(t.shape.len() == 2, "shared expert")?;
            let (s, l) = ffn(ne1, "ffn_down_shexp.weight")?;
            Piece::Outer(s, l)
        }
        "ffn_down_shexp.weight" => {
            let (s, l) = ffn(ne0, "ffn_down_shexp.weight")?;
            Piece::Cols(s, l)
        }
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
            want(t.shape.len() == 3, "routed experts")?;
            let (s, l) = ffn(ne1, "ffn_down_exps.weight")?;
            Piece::Rows(s, l)
        }
        "ffn_down_exps.weight" => {
            let (s, l) = ffn(ne0, "ffn_down_exps.weight")?;
            Piece::Cols(s, l)
        }
        s if s.starts_with("hc_")
            || s.starts_with("indexer")
            || s.starts_with("nextn.")
            || s.starts_with("attn_q_a")
            || s.starts_with("attn_kv_a")
            || s.starts_with("ffn_gate_inp")
            || s.starts_with("exp_probs_b")
            || s.starts_with("ssm_f_a")
            || s.starts_with("ssm_g_a") =>
        {
            Piece::Mirrored
        }
        // Norms, per-tensor scales and other vectors not listed above.
        _ if t.shape.len() <= 1 => Piece::Mirrored,
        _ => {
            return Err(format!(
                "{}: a glm5-next layer tensor tensor parallelism over machines cannot split",
                t.name
            ));
        }
    })
}

/// Sets `glm5-next.{k}` in `meta`.
fn set(meta: &mut GgufWriter, k: &str, v: Value) -> Result<(), String> {
    meta.set(&format!("{ARCH}.{k}"), &v)
        .map(|_| ())
        .map_err(|e| format!("{k}: {e}"))
}

/// The bytes `ranges` (`(offset, len)`, in order) of `whole`, back to back.
fn take(whole: Compute, ranges: Vec<(u64, u64)>) -> Compute {
    Arc::new(move || {
        let w = whole()?;
        let mut b = Vec::with_capacity(ranges.iter().map(|r| r.1 as usize).sum());
        for &(at, len) in &ranges {
            b.extend_from_slice(&w[at as usize..(at + len) as usize]);
        }
        Ok(b)
    })
}

/// Rank `rank`'s share of `gguf`, a glm5-next model (its MTP block included, if it has one),
/// over `n` ranks (see the module docs).
pub fn view(gguf: &GgufFile, rank: usize, n: usize) -> Result<GgufFile, String> {
    let arch = gguf.architecture().unwrap_or_default();
    if arch != ARCH {
        return Err(format!("architecture {arch:?}: not {ARCH}"));
    }
    let d = Dims::of(gguf)?;
    let (_, heads) = even("attention.head_count", d.n_head, n, rank)?;
    let mut meta = GgufWriter::new();
    for (k, v) in gguf.metadata() {
        meta.set(&k, &v).map_err(|e| format!("{k}: {e}"))?;
    }
    let u = |v: u64| Value::U32(v as u32);
    set(&mut meta, "attention.head_count", u(heads))?;

    let tensors = gguf.tensors();
    let types: HashMap<String, ffi::ggml_type> = tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type))
        .collect();
    let kda: HashMap<usize, u64> = tensors
        .iter()
        .filter_map(|t| {
            let il = t
                .name
                .strip_prefix("blk.")?
                .strip_suffix(".attn_q.weight")?
                .parse::<usize>()
                .ok()?;
            Some((il, t.shape.get(1).copied()? / d.kda_head.max(1)))
        })
        .collect();
    let kda_heads = |il: usize| kda.get(&il).copied();
    let down = |il: usize, name: &str| types.get(&format!("blk.{il}.{name}")).copied();
    let paths: Vec<PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut out = Vec::with_capacity(tensors.len());
    let mut recipes = HashMap::new();
    let mut ff: [Option<u64>; 3] = [None; 3];
    for t in tensors {
        let p = piece(&d, &t, &kda_heads, &down, n, rank)?;
        let recipe = gguf.recipe(&t.name);
        let stack = gguf.nvfp4_stack(&t.name).filter(|st| {
            t.shape.len() == 3
                && st.experts.len() as u64 == t.shape[2]
                && st.k as u64 == t.shape[0]
                && st.rows as u64 == t.shape[1]
        });
        let mut s = t.clone();
        match p {
            Piece::Mirrored => {
                if let Some(r) = recipe {
                    recipes.insert(t.name.clone(), r.clone());
                }
            }
            Piece::Outer(start, len) => {
                let dims = t.shape.len().max(1);
                let items = t.shape.last().copied().unwrap_or(1);
                if len == 0 || start + len > items || !t.size.is_multiple_of(items) {
                    return Err(format!("{}: items {start}+{len} of {items}", t.name));
                }
                let per = t.size / items;
                if s.shape.is_empty() {
                    s.shape.push(1);
                }
                s.shape[dims - 1] = len;
                s.size = len * per;
                let (at, take_len) = (start * per, len * per);
                let r = match recipe {
                    None => Recipe::Gather(vec![FileRange {
                        file: t.file_index,
                        offset: t.offset + at,
                        len: take_len,
                    }]),
                    Some(_) => Recipe::Computed(take(
                        super::whole_bytes(gguf, &paths, &t),
                        vec![(at, take_len)],
                    )),
                };
                recipes.insert(t.name.clone(), r);
            }
            Piece::Rows(start, len) => {
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize) as u64;
                let (ne1, slices) = (t.shape[1], t.shape[2]);
                if len == 0 || start + len > ne1 {
                    return Err(format!("{}: rows {start}+{len} of {ne1}", t.name));
                }
                s.shape[1] = len;
                s.size = len * row * slices;
                // Byte ranges of the tensor, in order.
                let bytes: Vec<(u64, u64)> = (0..slices)
                    .map(|sl| ((sl * ne1 + start) * row, len * row))
                    .collect();
                let r = match (recipe, stack) {
                    (None, _) => Recipe::Gather(
                        bytes
                            .iter()
                            .map(|&(at, l)| FileRange {
                                file: t.file_index,
                                offset: t.offset + at,
                                len: l,
                            })
                            .collect(),
                    ),
                    (Some(_), Some(st)) => {
                        Recipe::Computed(nvfp4_rows(st, &paths, start as usize, len as usize))
                    }
                    (Some(_), None) => {
                        Recipe::Computed(take(super::whole_bytes(gguf, &paths, &t), bytes))
                    }
                };
                recipes.insert(t.name.clone(), r);
            }
            Piece::Cols(start, len) => {
                let block = unsafe { ffi::ggml_blck_size(t.ggml_type) } as u64;
                if !start.is_multiple_of(block) || !len.is_multiple_of(block) {
                    return Err(format!(
                        "{}: columns {start}+{len} are not whole blocks of {block}",
                        t.name
                    ));
                }
                if len == 0 || start + len > t.shape[0] {
                    return Err(format!(
                        "{}: columns {start}+{len} of {}",
                        t.name, t.shape[0]
                    ));
                }
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
                let at = ggml::row_size(t.ggml_type, start as usize);
                let take_len = ggml::row_size(t.ggml_type, len as usize);
                let rows = (t.size as usize) / row;
                s.shape[0] = len;
                s.size = (rows * take_len) as u64;
                let r = match (recipe, stack) {
                    (Some(_), Some(st)) => nvfp4_cols(st, &paths, start as usize, len as usize),
                    _ => {
                        let whole = super::whole_bytes(gguf, &paths, &t);
                        Arc::new(move || {
                            let w = whole()?;
                            let mut b = Vec::with_capacity(rows * take_len);
                            for r in w.chunks_exact(row) {
                                b.extend_from_slice(&r[at..at + take_len]);
                            }
                            Ok(b)
                        })
                    }
                };
                recipes.insert(t.name.clone(), Recipe::Computed(r));
            }
        }
        for (i, sfx) in [
            ".ffn_down.weight",
            ".ffn_down_exps.weight",
            ".ffn_down_shexp.weight",
        ]
        .iter()
        .enumerate()
        {
            if t.name.ends_with(sfx) {
                ff[i].get_or_insert(s.shape[0]);
            }
        }
        out.push(s);
    }
    for (f, k) in ff.iter().zip([
        "feed_forward_length",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
    ]) {
        if let Some(f) = f
            && gguf.get(&format!("{ARCH}.{k}")).is_some()
        {
            set(&mut meta, k, u(*f))?;
        }
    }
    let mut cut = GgufFile::synthetic(meta, gguf.path().to_path_buf(), paths, out, recipes)
        .map_err(|e| e.to_string())?;
    cut.set_checkpoint(gguf.checkpoint());
    Ok(cut)
}

/// Reads `r` of `paths`.
fn read(paths: &[PathBuf], r: &FileRange) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::FileExt;
    let path = paths
        .get(r.file)
        .ok_or_else(|| format!("NVFP4 expert: no file {}", r.file))?;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut b = vec![0u8; r.len as usize];
    f.read_exact_at(&mut b, r.offset)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b)
}

/// Checks expert `e`'s sources of `stack`: `rows` x `k` codes and scales.
fn sources(stack: &Nvfp4Stack, e: usize) -> Result<(FileRange, FileRange), String> {
    let src = stack
        .experts
        .get(e)
        .ok_or_else(|| format!("NVFP4 expert {e}: not in the stack"))?;
    let (rows, k) = (stack.rows as u64, stack.k as u64);
    if src.weight.len != rows * k / 2 || src.scale.len != rows * k / 16 {
        return Err(format!(
            "NVFP4 expert {e}: {} code and {} scale bytes for {rows} x {k}",
            src.weight.len, src.scale.len
        ));
    }
    Ok((src.weight, src.scale))
}

/// Rows `start..start + len` of every expert of the stacked NVFP4 tensor whose per-expert
/// sources are `stack` (file indices into `paths`): each expert's rows of codes and sub-block
/// scales read from the checkpoint and repacked ([`crate::hf::nvfp4::repack_nvfp4`], row by
/// row), the same bytes as the whole stack's repack cut to those rows.
pub fn nvfp4_rows(stack: Arc<Nvfp4Stack>, paths: &[PathBuf], start: usize, len: usize) -> Compute {
    let paths = paths.to_vec();
    Arc::new(move || {
        let k = stack.k;
        if start + len > stack.rows {
            return Err(format!("NVFP4 rows {start}+{len} of {}", stack.rows));
        }
        let mut out = Vec::new();
        for e in 0..stack.experts.len() {
            let (w, s) = sources(&stack, e)?;
            let part = |r: FileRange, per: usize| FileRange {
                file: r.file,
                offset: r.offset + (start * per) as u64,
                len: (len * per) as u64,
            };
            let w = read(&paths, &part(w, k / 2))?;
            let s = read(&paths, &part(s, k / 16))?;
            out.extend(crate::hf::nvfp4::repack_nvfp4(&w, &s, len, k));
        }
        Ok(out)
    })
}

/// Columns `start..start + len` (whole 64-element blocks) of every row of every expert of the
/// stacked NVFP4 tensor whose per-expert sources are `stack`: each expert's codes and scales
/// read whole, those columns' kept and repacked, the same bytes as the whole stack's repack
/// cut to those columns (the repack is block by block).
pub fn nvfp4_cols(stack: Arc<Nvfp4Stack>, paths: &[PathBuf], start: usize, len: usize) -> Compute {
    let paths = paths.to_vec();
    Arc::new(move || {
        let (rows, k) = (stack.rows, stack.k);
        if !start.is_multiple_of(64) || !len.is_multiple_of(64) || start + len > k {
            return Err(format!(
                "NVFP4 columns {start}+{len} of {k} in blocks of 64"
            ));
        }
        let mut out = Vec::new();
        let (mut wc, mut sc) = (Vec::new(), Vec::new());
        for e in 0..stack.experts.len() {
            let (w, s) = sources(&stack, e)?;
            let (w, s) = (read(&paths, &w)?, read(&paths, &s)?);
            wc.clear();
            sc.clear();
            for (wr, sr) in w.chunks_exact(k / 2).zip(s.chunks_exact(k / 16)) {
                wc.extend_from_slice(&wr[start / 2..(start + len) / 2]);
                sc.extend_from_slice(&sr[start / 16..(start + len) / 16]);
            }
            out.extend(crate::hf::nvfp4::repack_nvfp4(&wc, &sc, rows, len));
        }
        Ok(out)
    })
}
