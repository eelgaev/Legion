//! One rank's share of a Qwen3.8-Flash-Next (`qwen4exp`) GGUF, or of its MTP head, for tensor
//! parallelism over machines (#733): the layers' heads, DeltaNet head groups and FFN rows of
//! this rank, with the metadata of the smaller model they make, so the qwen4exp graph runs on
//! them unchanged and sums the attention's (or DeltaNet's) and the MoE's partial outputs
//! (`Model::tp_sum`, two all-reduces per layer).
//!
//! - **Gated attention:** `attn_q` holds `[q_h | gate_h]` per query head, so a rank's query
//!   heads are one range of rows (`2 × head_dim` each); `attn_k`/`attn_v` by kv heads (a GQA
//!   group's queries go with their kv head); `attn_output` by the same heads' columns. The QSA
//!   indexer is mirrored: every rank scores and selects the same pools.
//! - **Gated DeltaNet:** value head `h` reads key head `h % n_k` (ggml's `gated_delta_net`), so
//!   rank `r` of `n` takes the key heads `r × n_k/n ..` and the value heads of every tile
//!   `t × n_k + r × n_k/n ..` (`n_v / n_k` tiles); its local value head `j` then reads its
//!   local key head `j % (n_k/n)`, as the op expects. `attn_qkv` and `ssm_conv1d` by those q,
//!   k and v channels, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a` and `ssm_dt.bias` by the
//!   value heads, `ssm_out` by the value heads' columns. `ssm_norm` (one head's width) is
//!   mirrored.
//! - **MoE:** every routed expert's rows (`ffn_gate_exps`/`ffn_up_exps` rows, `ffn_down_exps`
//!   columns of every expert) and the shared expert's rows: the router and the shared-expert
//!   gate are mirrored, so every rank routes alike and the work is even whatever the routing.
//!   With `--tp-moe-ep` ([`Experts::Whole`], #798) the routed experts go by expert index
//!   instead: rank `r` of `n` holds experts `r·E/n .. (r+1)·E/n` whole (their bytes as
//!   shipped, one range of each stack) and runs only the pairs routed to them; the shared
//!   expert keeps its rows.
//! - Everything else (embeddings, hyper-connection mixers, the per-layer n-gram embedding and
//!   its table, norms, the output head, the MTP head's `nextn.*`): mirrored.
//!
//! Row ranges of plain file tensors are gathered straight from the file; column cuts (and any
//! tensor a safetensors checkpoint computes at load) read the whole tensor and keep this
//! rank's columns.

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::ggml::{self, ffi};
use crate::gguf::{
    Compute, FileRange, GgufFile, GgufWriter, Nvfp4Stack, Recipe, TensorInfo, Value,
};
use crate::tp_net::ep_place::{self, Placement};

const ARCH: &str = "qwen4exp";

/// How one tensor is cut: `(start, len)` ranges, back to back in the share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Mirrored,
    /// Rows (axis 1) of every matrix of the tensor (each expert's, for stacked experts).
    Rows(Vec<(u64, u64)>),
    /// Columns (axis 0) of every row; a 1-D tensor's elements.
    Cols(Vec<(u64, u64)>),
    /// Whole experts along the outermost axis, as runs `(first, count)` of expert ids
    /// (`--tp-moe-ep`, #798; one run unless placed, #803).
    Experts(Vec<(u64, u64)>),
}

/// How the routed experts are shared out among the ranks.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Experts {
    /// Every expert's rows (the default): each rank holds every expert at `1/n` of its width.
    #[default]
    Rows,
    /// Whole experts (`--tp-moe-ep`, #798): rank `r` of `n` holds experts
    /// [`expert_range`]`(E, n, r)` at full width. Its MoE then runs only the (token, expert)
    /// pairs of those experts and its partial sum is over them (the router is mirrored), so
    /// the same all-reduce adds the ranks' partials.
    Whole,
    /// [`Experts::Whole`] with the experts placed per layer (`--tp-moe-ep-place`, #803): rank
    /// `r` holds [`Placement::experts_of`]`(layer, r)`, ascending (the contiguous split for
    /// layers the table does not name).
    Placed(Arc<Placement>),
}

impl Experts {
    /// Whether the routed experts go by expert index ([`Experts::Whole`] or placed).
    pub fn whole(&self) -> bool {
        !matches!(self, Experts::Rows)
    }
}

/// The routed experts rank `rank` of `n` holds under [`Experts::Whole`]: `(first, count)`.
pub fn expert_range(n_expert: u64, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let n64 = n as u64;
    if n_expert == 0 || !n_expert.is_multiple_of(n64) || rank >= n {
        return Err(format!(
            "expert_count: {n_expert} experts do not divide among {n} ranks"
        ));
    }
    let per = n_expert / n64;
    Ok((rank as u64 * per, per))
}

/// Whether `name` (a layer tensor's name after `blk.{il}.`) is one of the routed experts'
/// stacks or their per-expert scales, which [`Experts::Whole`] cuts by expert.
fn routed(name: &str) -> bool {
    ["ffn_gate_exps.", "ffn_up_exps.", "ffn_down_exps."]
        .iter()
        .any(|p| name.starts_with(p))
}

/// The model's shapes the cut goes by (all ranks').
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub n_head: u64,
    pub n_head_kv: u64,
    pub head_dim: u64,
    /// DeltaNet key heads and their width.
    pub n_k: u64,
    pub head_k: u64,
    /// DeltaNet value heads and their width.
    pub n_v: u64,
    pub head_v: u64,
}

impl Dims {
    /// From a qwen4exp GGUF's (or MTP head's) metadata.
    pub fn of(gguf: &GgufFile) -> Result<Self, String> {
        let u = |k: &str| {
            gguf.get_u64(&format!("{ARCH}.{k}"))
                .ok_or_else(|| format!("{ARCH}.{k}: missing"))
        };
        let n_v = u("ssm.time_step_rank")?;
        let inner = u("ssm.inner_size")?;
        let d = Self {
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_k: u("ssm.group_count")?,
            head_k: u("ssm.state_size")?,
            n_v,
            head_v: inner.checked_div(n_v).unwrap_or(0),
        };
        if d.n_k == 0 || d.n_v == 0 || !d.n_v.is_multiple_of(d.n_k) || d.head_v * n_v != inner {
            return Err(format!(
                "DeltaNet heads: {} value heads of {inner} channels over {} key heads",
                d.n_v, d.n_k
            ));
        }
        Ok(d)
    }

    /// The metadata of rank `rank`'s share over `n` ranks, checked: every split count divides.
    fn share(&self, n: usize, rank: usize) -> Result<Dims, String> {
        let n64 = n as u64;
        for (what, v) in [
            ("attention.head_count", self.n_head),
            ("attention.head_count_kv", self.n_head_kv),
            ("ssm.group_count (DeltaNet key heads)", self.n_k),
        ] {
            if v == 0 || !v.is_multiple_of(n64) {
                return Err(format!("{what}: {v} do not divide among {n} ranks"));
            }
        }
        debug_assert!(rank < n);
        Ok(Dims {
            n_head: self.n_head / n64,
            n_head_kv: self.n_head_kv / n64,
            n_k: self.n_k / n64,
            n_v: self.n_v / n64,
            ..*self
        })
    }
}

/// `count` items of `unit` each from item `first`: one range.
fn span(first: u64, count: u64, unit: u64) -> (u64, u64) {
    (first * unit, count * unit)
}

/// Rank `rank`'s DeltaNet value heads over `n` ranks, as head ranges (see the module docs).
fn value_heads(d: &Dims, n: usize, rank: usize) -> Vec<(u64, u64)> {
    let kq = d.n_k / n as u64;
    (0..d.n_v / d.n_k)
        .map(|t| (t * d.n_k + rank as u64 * kq, kq))
        .collect()
}

/// Rank `rank`'s share of an FFN of `n_ff` rows over `n` ranks whose down projection is of
/// type `down` (its columns must be whole blocks).
fn ffn_share(n_ff: u64, down: ffi::ggml_type, n: usize, rank: usize) -> Result<(u64, u64), String> {
    let block = unsafe { ffi::ggml_blck_size(down) } as u64;
    let per = n_ff / n as u64;
    if n_ff == 0 || !n_ff.is_multiple_of(n as u64) || !per.is_multiple_of(block) {
        return Err(format!(
            "{n_ff} FFN rows do not split among {n} ranks in whole blocks of {block} \
             (the down projection's {})",
            ggml::type_name(down)
        ));
    }
    Ok((rank as u64 * per, per))
}

/// How rank `rank` of `n` cuts tensor `t` (see the module docs); `down` is the type of each
/// layer's FFN down projections by name (`ffn_down_exps.weight` / `ffn_down_shexp.weight`),
/// whose blocks the FFN rows go in.
pub fn piece(
    d: &Dims,
    t: &TensorInfo,
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
    let ne0 = t.shape.first().copied().unwrap_or(1);
    let ne1 = t.shape.get(1).copied().unwrap_or(1);
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
    let n64 = n as u64;
    let r = rank as u64;
    let (qh, kvh) = (d.n_head / n64, d.n_head_kv / n64);
    let kq = d.n_k / n64;
    let vh = value_heads(d, n, rank);
    let v_of = |unit: u64| {
        vh.iter()
            .map(|&(h, c)| span(h, c, unit))
            .collect::<Vec<_>>()
    };
    // An FFN's rows: `n_ff` (the gate's and up's rows, the down's columns) in whole blocks of
    // the layer's down projection `down_name`.
    let ffn = |n_ff: u64, down_name: &str| -> Result<(u64, u64), String> {
        let ty =
            down(il, down_name).ok_or_else(|| format!("{}: no blk.{il}.{down_name}", t.name))?;
        ffn_share(n_ff, ty, n, rank).map_err(|e| format!("{}: {e}", t.name))
    };
    Ok(match name {
        "attn_q.weight" => {
            want(ne1 == d.n_head * 2 * d.head_dim, "gated queries")?;
            Piece::Rows(vec![span(r * qh, qh, 2 * d.head_dim)])
        }
        "attn_k.weight" | "attn_v.weight" => {
            want(ne1 == d.n_head_kv * d.head_dim, "keys and values")?;
            Piece::Rows(vec![span(r * kvh, kvh, d.head_dim)])
        }
        "attn_output.weight" => {
            want(ne0 == d.n_head * d.head_dim, "attention output")?;
            Piece::Cols(vec![span(r * qh, qh, d.head_dim)])
        }
        "attn_qkv.weight" | "ssm_conv1d.weight" => {
            let k_all = d.n_k * d.head_k;
            want(ne1 == 2 * k_all + d.n_v * d.head_v, "DeltaNet q, k and v")?;
            let mut rows = vec![span(r * kq, kq, d.head_k)];
            rows.push((k_all + r * kq * d.head_k, kq * d.head_k));
            rows.extend(v_of(d.head_v).into_iter().map(|(s, l)| (2 * k_all + s, l)));
            Piece::Rows(rows)
        }
        "attn_gate.weight" => {
            want(ne1 == d.n_v * d.head_v, "DeltaNet output gate")?;
            Piece::Rows(v_of(d.head_v))
        }
        "ssm_alpha.weight" | "ssm_beta.weight" => {
            want(ne1 == d.n_v, "DeltaNet value heads")?;
            Piece::Rows(v_of(1))
        }
        "ssm_a" | "ssm_dt.bias" => {
            want(t.shape.len() == 1 && ne0 == d.n_v, "DeltaNet value heads")?;
            Piece::Cols(v_of(1))
        }
        "ssm_out.weight" => {
            want(ne0 == d.n_v * d.head_v, "DeltaNet output")?;
            Piece::Cols(v_of(d.head_v))
        }
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
            Piece::Rows(vec![ffn(ne1, "ffn_down_exps.weight")?])
        }
        "ffn_down_exps.weight" => Piece::Cols(vec![ffn(ne0, "ffn_down_exps.weight")?]),
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => {
            Piece::Rows(vec![ffn(ne1, "ffn_down_shexp.weight")?])
        }
        "ffn_down_shexp.weight" => Piece::Cols(vec![ffn(ne0, "ffn_down_shexp.weight")?]),
        s if s.starts_with("hc_")
            || s.starts_with("ple_")
            || s.starts_with("indexer.")
            || s.starts_with("nextn.")
            || s.starts_with("ffn_gate_inp") =>
        {
            Piece::Mirrored
        }
        // Norms and other per-channel vectors not listed above.
        _ if t.shape.len() <= 1 => Piece::Mirrored,
        _ => {
            return Err(format!(
                "{}: a qwen4exp layer tensor tensor parallelism over machines cannot split",
                t.name
            ));
        }
    })
}

/// Sets `qwen4exp.{k}` in `meta`.
fn set(meta: &mut GgufWriter, k: &str, v: Value) -> Result<(), String> {
    meta.set(&format!("{ARCH}.{k}"), &v)
        .map(|_| ())
        .map_err(|e| format!("{k}: {e}"))
}

/// How [`Experts::Whole`] (or [`Experts::Placed`]) cuts layer `il`'s tensor `name` (after
/// `blk.{il}.`) of shape `shape` of a model of `n_expert` routed experts: whole experts along
/// the outermost axis for the routed stacks and their per-expert scales; `None` for everything
/// else (cut by [`piece`]), a per-tensor scale of the stacks included.
#[allow(clippy::too_many_arguments)]
fn expert_piece(
    full: &str,
    il: usize,
    name: &str,
    shape: &[u64],
    n_expert: u64,
    n: usize,
    rank: usize,
    experts: &Experts,
) -> Result<Option<Piece>, String> {
    if !routed(name) {
        return Ok(None);
    }
    match shape.last() {
        Some(&e) if shape.len() >= 2 || name.ends_with("scale") => {
            if e != n_expert {
                if name.ends_with(".weight") {
                    return Err(format!(
                        "{full}: shape {shape:?} does not end in the {n_expert} experts"
                    ));
                }
                // A per-tensor scale: the same for every expert.
                return Ok(Some(Piece::Mirrored));
            }
            let runs = match experts {
                Experts::Placed(p) => ep_place::runs(&p.experts_of(il, rank)),
                _ => vec![expert_range(n_expert, n, rank)?],
            };
            Ok(Some(Piece::Experts(runs)))
        }
        _ => Ok(Some(Piece::Mirrored)),
    }
}

/// Rank `rank`'s share of `gguf`, a qwen4exp model or MTP head, over `n` ranks (see the
/// module docs).
pub fn view(gguf: &GgufFile, rank: usize, n: usize) -> Result<GgufFile, String> {
    view_with(gguf, rank, n, Experts::Rows)
}

/// [`view`] with the routed experts shared out as `experts` says (the shared expert, the
/// router, attention and DeltaNet are cut as [`view`] cuts them either way).
pub fn view_with(
    gguf: &GgufFile,
    rank: usize,
    n: usize,
    experts: Experts,
) -> Result<GgufFile, String> {
    let arch = gguf.architecture().unwrap_or_default();
    if arch != ARCH {
        return Err(format!("architecture {arch:?}: not {ARCH}"));
    }
    let n_expert = gguf.get_u64(&format!("{ARCH}.expert_count")).unwrap_or(0);
    if experts.whole() {
        expert_range(n_expert, n, rank).map_err(|e| format!("--tp-moe-ep: {e}"))?;
    }
    if let Experts::Placed(p) = &experts {
        p.fits(n_expert as usize, n)
            .map_err(|e| format!("--tp-moe-ep-place: {e}"))?;
    }
    let d = Dims::of(gguf)?;
    let here = d.share(n, rank)?;
    let mut meta = GgufWriter::new();
    for (k, v) in gguf.metadata() {
        meta.set(&k, &v).map_err(|e| format!("{k}: {e}"))?;
    }
    let u = |v: u64| Value::U32(v as u32);
    set(&mut meta, "attention.head_count", u(here.n_head))?;
    set(&mut meta, "attention.head_count_kv", u(here.n_head_kv))?;
    set(&mut meta, "ssm.group_count", u(here.n_k))?;
    set(&mut meta, "ssm.time_step_rank", u(here.n_v))?;
    set(&mut meta, "ssm.inner_size", u(here.n_v * here.head_v))?;

    let tensors = gguf.tensors();
    let types: HashMap<String, ffi::ggml_type> = tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type))
        .collect();
    let down = |il: usize, name: &str| types.get(&format!("blk.{il}.{name}")).copied();
    let paths: Vec<PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut out = Vec::with_capacity(tensors.len());
    let mut recipes = HashMap::new();
    let (mut ff_exp, mut ff_sh) = (None, None);
    for t in tensors {
        let whole_experts = match t.name.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
            Some((il, name)) if experts.whole() => {
                let il = il
                    .parse::<usize>()
                    .map_err(|_| format!("{}: no layer index", t.name))?;
                expert_piece(&t.name, il, name, &t.shape, n_expert, n, rank, &experts)?
            }
            _ => None,
        };
        let p = match whole_experts {
            Some(p) => p,
            None => piece(&d, &t, &down, n, rank)?,
        };
        let recipe = gguf.recipe(&t.name);
        let mut s = t.clone();
        match p {
            Piece::Mirrored => {
                if let Some(r) = recipe {
                    recipes.insert(t.name.clone(), r.clone());
                }
            }
            Piece::Experts(runs) => {
                let last = t.shape.len() - 1;
                let per = t.size / t.shape[last];
                let len: u64 = runs.iter().map(|r| r.1).sum();
                // Byte ranges of the tensor, one per run of consecutive experts, in order.
                let bytes: Vec<(u64, u64)> =
                    runs.iter().map(|&(s, l)| (s * per, l * per)).collect();
                s.shape[last] = len;
                s.size = len * per;
                let stack = gguf
                    .nvfp4_stack(&t.name)
                    .filter(|st| st.experts.len() as u64 == t.shape[last]);
                let r = match (recipe, stack) {
                    (None, _) => Recipe::Gather(
                        bytes
                            .iter()
                            .map(|&(at, take)| FileRange {
                                file: t.file_index,
                                offset: t.offset + at,
                                len: take,
                            })
                            .collect(),
                    ),
                    // A ModelOpt NVFP4 stack: only this rank's experts read and repacked, as
                    // the checkpoint's recipe repacks each of the whole stack's (#1150).
                    (Some(_), Some(st)) => Recipe::Computed(nvfp4_experts(st, &paths, &runs, per)),
                    // A tensor a safetensors checkpoint computes at load: made whole, then
                    // this rank's experts kept (as shipped: their bytes, no repack).
                    (Some(_), None) => {
                        let whole = super::whole_bytes(gguf, &paths, &t);
                        Recipe::Computed(Arc::new(move || {
                            let w = whole()?;
                            let mut b = Vec::new();
                            for &(at, take) in &bytes {
                                b.extend_from_slice(&w[at as usize..(at + take) as usize]);
                            }
                            Ok(b)
                        }))
                    }
                };
                recipes.insert(t.name.clone(), r);
            }
            Piece::Rows(ranges) => {
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize) as u64;
                let ne1 = t.shape.get(1).copied().unwrap_or(1);
                let slices: u64 = t.shape.iter().skip(2).product();
                let take: u64 = ranges.iter().map(|r| r.1).sum();
                if ranges.iter().any(|&(s, l)| l == 0 || s + l > ne1) {
                    return Err(format!("{}: rows {ranges:?} out of {ne1}", t.name));
                }
                // Byte ranges of the tensor, in order.
                let bytes: Vec<(u64, u64)> = (0..slices)
                    .flat_map(|sl| {
                        ranges
                            .iter()
                            .map(move |&(st, l)| ((sl * ne1 + st) * row, l * row))
                    })
                    .collect();
                if s.shape.len() < 2 {
                    s.shape.push(1);
                }
                s.shape[1] = take;
                s.size = take * row * slices;
                let r = match recipe {
                    None => Recipe::Gather(
                        bytes
                            .iter()
                            .map(|&(at, len)| FileRange {
                                file: t.file_index,
                                offset: t.offset + at,
                                len,
                            })
                            .collect(),
                    ),
                    Some(_) => {
                        let whole = super::whole_bytes(gguf, &paths, &t);
                        Recipe::Computed(Arc::new(move || {
                            let w = whole()?;
                            let mut b = Vec::new();
                            for &(at, len) in &bytes {
                                b.extend_from_slice(&w[at as usize..(at + len) as usize]);
                            }
                            Ok(b)
                        }))
                    }
                };
                recipes.insert(t.name.clone(), r);
            }
            Piece::Cols(ranges) => {
                let block = unsafe { ffi::ggml_blck_size(t.ggml_type) } as u64;
                if ranges
                    .iter()
                    .any(|&(s, l)| !s.is_multiple_of(block) || !l.is_multiple_of(block))
                {
                    return Err(format!(
                        "{}: columns {ranges:?} are not whole blocks of {block}",
                        t.name
                    ));
                }
                if ranges.iter().any(|&(s, l)| l == 0 || s + l > t.shape[0]) {
                    return Err(format!(
                        "{}: columns {ranges:?} out of {}",
                        t.name, t.shape[0]
                    ));
                }
                let row = ggml::row_size(t.ggml_type, t.shape[0] as usize);
                let cuts: Vec<(usize, usize)> = ranges
                    .iter()
                    .map(|&(st, l)| {
                        (
                            ggml::row_size(t.ggml_type, st as usize),
                            ggml::row_size(t.ggml_type, l as usize),
                        )
                    })
                    .collect();
                let take_cols: u64 = ranges.iter().map(|r| r.1).sum();
                let take = ggml::row_size(t.ggml_type, take_cols as usize);
                let rows = (t.size as usize) / row;
                s.shape[0] = take_cols;
                s.size = (rows * take) as u64;
                let whole = super::whole_bytes(gguf, &paths, &t);
                recipes.insert(
                    t.name.clone(),
                    Recipe::Computed(Arc::new(move || {
                        let w = whole()?;
                        let mut b = Vec::with_capacity(rows * take);
                        for r in w.chunks_exact(row) {
                            for &(at, len) in &cuts {
                                b.extend_from_slice(&r[at..at + len]);
                            }
                        }
                        Ok(b)
                    })),
                );
            }
        }
        if t.name.ends_with(".ffn_down_exps.weight") {
            ff_exp.get_or_insert(s.shape[0]);
        } else if t.name.ends_with(".ffn_down_shexp.weight") {
            ff_sh.get_or_insert(s.shape[0]);
        }
        out.push(s);
    }
    if let Some(f) = ff_exp {
        set(&mut meta, "expert_feed_forward_length", u(f))?;
    }
    if let Some(f) = ff_sh {
        set(&mut meta, "expert_shared_feed_forward_length", u(f))?;
    }
    let mut cut = GgufFile::synthetic(meta, gguf.path().to_path_buf(), paths, out, recipes)
        .map_err(|e| e.to_string())?;
    cut.set_checkpoint(gguf.checkpoint());
    Ok(cut)
}

/// Experts `runs` (`(first, count)` of expert ids, in order) of the stacked NVFP4 tensor whose
/// per-expert sources are `stack` (file indices into `paths`), each `per` bytes once repacked:
/// every expert's codes and sub-block scales read from the checkpoint and repacked
/// ([`crate::hf::nvfp4::repack_nvfp4`]) as the whole stack's recipe does, so the bytes are the
/// whole stack's for those experts, and a rank reads only its own experts' share of the file.
fn nvfp4_experts(
    stack: Arc<Nvfp4Stack>,
    paths: &[PathBuf],
    runs: &[(u64, u64)],
    per: u64,
) -> Compute {
    let paths = paths.to_vec();
    let ids: Vec<usize> = runs
        .iter()
        .flat_map(|&(first, count)| first as usize..(first + count) as usize)
        .collect();
    Arc::new(move || {
        use std::os::unix::fs::FileExt;
        let read = |r: &FileRange| -> Result<Vec<u8>, String> {
            let path = paths
                .get(r.file)
                .ok_or_else(|| format!("NVFP4 expert: no file {}", r.file))?;
            let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut b = vec![0u8; r.len as usize];
            f.read_exact_at(&mut b, r.offset)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(b)
        };
        let mut out = Vec::with_capacity(ids.len() * per as usize);
        for &e in &ids {
            let src = stack
                .experts
                .get(e)
                .ok_or_else(|| format!("NVFP4 expert {e}: not in the stack"))?;
            let (w, s) = (read(&src.weight)?, read(&src.scale)?);
            if w.len() != stack.rows * stack.k / 2 || s.len() != stack.rows * stack.k / 16 {
                return Err(format!(
                    "NVFP4 expert {e}: {} code and {} scale bytes for {} x {}",
                    w.len(),
                    s.len(),
                    stack.rows,
                    stack.k
                ));
            }
            out.extend(crate::hf::nvfp4::repack_nvfp4(&w, &s, stack.rows, stack.k));
        }
        if out.len() as u64 != ids.len() as u64 * per {
            return Err(format!(
                "NVFP4 experts: {} bytes, expected {}",
                out.len(),
                ids.len() as u64 * per
            ));
        }
        Ok(out)
    })
}
