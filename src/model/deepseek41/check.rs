//! A CPU-only check of a DeepSeek-V4.1-Flash GGUF before a GPU load (`legion gguf check`):
//! every tensor the graph reads, by name, shape and type, against the parsed hyperparameters,
//! and the Engram tables (in the model's files or the companion `engram.file` names) against
//! the hash parameters, with a few hashed rows read and dequantized.
//!
//! It reads metadata, the small hash tensors and those few rows (positioned reads, no mapping
//! and no readahead), so it costs nothing next to the 100+ GB files.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::FileExt;

use crate::ggml::{Context, ffi, types};
use crate::gguf::{GgufFile, Recipe, TensorInfo};
use crate::model::core::state::NO_TOKEN;
use crate::model::core::{Hparams, ModelError};
use crate::model::moe_ffn::{self, ExpertRefs};
use crate::residency::lookup_table::{LookupTable, TableOptions};

use super::{ARCH, Ds41Params, EngramParams, Hash, params};

/// Weight types the CUDA dense kernels take (`launch_linear` and the embedding in
/// `src/cuda/glm5next.cu`); others run on the CPU backend only.
const CUDA_LINEAR: [ffi::ggml_type; 7] = [
    types::F32,
    types::F16,
    types::BF16,
    types::Q8_0,
    types::Q4_K,
    types::Q5_K,
    types::Q6_K,
];

/// How the graph uses a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A matrix product or the embedding (the CUDA dense kernels' types).
    Linear,
    /// A routed expert stack (the fused expert op's types).
    Experts,
    /// Norms, mixers, sinks, biases, hash parameters, Engram tables: read as they are.
    Other,
}

/// What [`check`] found.
#[derive(Debug, Default)]
pub struct Report {
    /// Tensors the graph reads.
    pub checked: usize,
    /// Tensors the graph reads that no file has.
    pub missing: Vec<String>,
    /// Tensors of another shape than the hyperparameters give.
    pub shapes: Vec<String>,
    /// Layers whose routed experts the fused expert op does not take.
    pub experts: Vec<String>,
    /// Dense weights of a type the CUDA kernels do not take (the CPU backend does).
    pub cuda_types: Vec<String>,
    /// Dense FP8 weights with 32 × 32 block scales (the safetensors checkpoint as shipped),
    /// which no matmul takes yet (the parse refuses them, [`super::block_fp8`]).
    pub block_fp8: Vec<String>,
    /// Problems with the Engram tables or hash parameters.
    pub engram: Vec<String>,
    /// What the Engram check read (rows, files, sample values).
    pub engram_notes: Vec<String>,
    /// Tensors in the files that the graph never reads (the MTP / DSpark head, say).
    pub unread: Vec<String>,
    /// Tensor count per type, over all files.
    pub types: BTreeMap<&'static str, usize>,
}

impl Report {
    /// Whether the model loads and builds: nothing missing, no wrong shape, every dense weight,
    /// expert stack and the Engram tables usable. [`Self::cuda_types`] and [`Self::unread`]
    /// are notes.
    pub fn ok(&self) -> bool {
        self.missing.is_empty()
            && self.shapes.is_empty()
            && self.experts.is_empty()
            && self.block_fp8.is_empty()
            && self.engram.is_empty()
    }
}

/// Checks the DeepSeek-V4.1-Flash GGUF `g` (opened with its companions); `samples` tokens are
/// hashed through each Engram layer and their rows read and dequantized.
pub fn check(g: &GgufFile, samples: usize) -> Result<Report, ModelError> {
    check_with(g, samples, None)
}

/// [`check`], and with `table` the sampled rows read again through the Engram lookup table the
/// engine opens ([`super::engram`]) with those options, which must return the file's bytes.
pub fn check_with(
    g: &GgufFile,
    samples: usize,
    table: Option<&TableOptions>,
) -> Result<Report, ModelError> {
    if g.architecture().as_deref() != Some(ARCH) {
        return Err(ModelError::Unsupported(format!(
            "`gguf check` knows {ARCH} only, not {}",
            g.architecture().unwrap_or_default()
        )));
    }
    // The weights the parse refuses ([`super::block_fp8`]) are reported below.
    let hp = super::hparams(g)?;
    let p = params(&hp).expect("deepseek41 hyperparameters");
    let n_ff = g
        .get_u64(&format!("{ARCH}.expert_feed_forward_length"))
        .ok_or_else(|| ModelError::Metadata(format!("{ARCH}.expert_feed_forward_length")))?;
    let mut r = Report::default();
    let all: BTreeMap<String, TensorInfo> = g
        .tensors()
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    for t in all.values() {
        *r.types.entry(t.type_name()).or_default() += 1;
    }
    let want = expected(&hp, p, n_ff, read_engram_rows(g, p, &mut r));
    let mut read = BTreeSet::new();
    for (name, shape, role) in &want {
        r.checked += 1;
        read.insert(name.as_str());
        let Some(t) = all.get(name) else {
            r.missing.push(name.clone());
            continue;
        };
        if let Some(shape) = shape
            && &t.shape != shape
        {
            r.shapes
                .push(format!("{name}: {:?}, expected {shape:?}", t.shape));
        }
        if *role == Role::Linear {
            // A checkpoint's weight scale is read with its weight.
            let scale = name
                .strip_suffix(".weight")
                .and_then(|b| all.get(&format!("{b}.scale")));
            if let Some(s) = scale {
                read.insert(s.name.as_str());
            }
            if let Some(s) = scale.filter(|s| super::block_scaled(t, Some(s))) {
                r.block_fp8
                    .push(format!("{name} (FP8, {:?} block scales)", s.shape));
            } else if !CUDA_LINEAR.contains(&t.ggml_type) {
                r.cuda_types.push(format!("{name} ({})", t.type_name()));
            }
        }
    }
    r.unread = all
        .keys()
        .filter(|n| !read.contains(n.as_str()))
        .cloned()
        .collect();
    check_experts(&all, hp.n_layer, &mut r);
    if r.engram.is_empty()
        && let Some(e) = &p.engram
    {
        let rows = sample_engram(g, e, samples, &mut r);
        if let (Some(rows), Some(opts)) = (rows, table) {
            engram_table(g, &hp, e, &rows, opts, &mut r);
        }
    }
    Ok(r)
}

/// Every tensor the graph reads: name, shape (`None`: not known from the metadata) and role.
fn expected(
    hp: &Hparams,
    p: &Ds41Params,
    n_ff: u64,
    engram_rows: Option<Vec<u64>>,
) -> Vec<(String, Option<Vec<u64>>, Role)> {
    let u = |v: i64| v as u64;
    let (e, nv, ne) = (u(hp.n_embd), u(hp.n_vocab), u(hp.n_expert));
    let (hc, head, nh) = (u(p.hc), u(p.head), u(hp.n_head));
    let (ql, og, ol) = (u(p.q_lora), u(p.o_groups), u(p.o_lora));
    let (ih, id) = (u(p.idx_heads), u(p.idx_dim));
    let mut out = Vec::new();
    let mut add = |name: String, shape: Option<Vec<u64>>, role| out.push((name, shape, role));
    add("token_embd.weight".into(), Some(vec![e, nv]), Role::Linear);
    add("output_norm.weight".into(), Some(vec![e]), Role::Other);
    add("output.weight".into(), Some(vec![e, nv]), Role::Linear);
    for il in 0..hp.n_layer {
        let mut t =
            |s: &str, shape: Vec<u64>, role| add(format!("blk.{il}.{s}"), Some(shape), role);
        t("attn_norm.weight", vec![e], Role::Other);
        t("ffn_norm.weight", vec![e], Role::Other);
        t("attn_q_a.weight", vec![e, ql], Role::Linear);
        t("attn_q_a_norm.weight", vec![ql], Role::Other);
        t("attn_q_b.weight", vec![ql, nh * head], Role::Linear);
        t("attn_kv.weight", vec![e, head], Role::Linear);
        t("attn_kv_a_norm.weight", vec![head], Role::Other);
        t("attn_sinks.weight", vec![nh], Role::Other);
        t(
            "attn_output_a.weight",
            vec![nh * head / og, ol, og],
            Role::Linear,
        );
        t("attn_output_b.weight", vec![ol * og, e], Role::Linear);
        for which in ["attn", "ffn"] {
            let n_mix = (2 + hc) * hc;
            t(
                &format!("hc_{which}_fn.weight"),
                vec![e * hc, n_mix],
                Role::Other,
            );
            t(&format!("hc_{which}_base.weight"), vec![n_mix], Role::Other);
            t(&format!("hc_{which}_scale.weight"), vec![3], Role::Other);
        }
        t("ffn_gate_inp.weight", vec![e, ne], Role::Linear);
        t("exp_probs_b.bias", vec![ne], Role::Other);
        t("ffn_gate_shexp.weight", vec![e, n_ff], Role::Linear);
        t("ffn_up_shexp.weight", vec![e, n_ff], Role::Linear);
        t("ffn_down_shexp.weight", vec![n_ff, e], Role::Linear);
        t("ffn_gate_exps.weight", vec![e, n_ff, ne], Role::Experts);
        t("ffn_up_exps.weight", vec![e, n_ff, ne], Role::Experts);
        t("ffn_down_exps.weight", vec![n_ff, e, ne], Role::Experts);
        if p.kv_source[il] {
            t("attn_compressor_kv.weight", vec![e, head], Role::Linear);
            if p.ratios[il] > 1 {
                t("attn_compressor_gate.weight", vec![e, head], Role::Linear);
            }
            t("attn_compressor_norm.weight", vec![head], Role::Other);
            t("indexer.attn_k.weight", vec![head, id], Role::Linear);
            t("indexer.k_norm.weight", vec![id], Role::Other);
        }
        if p.index_source[il] {
            t("indexer.proj.weight", vec![e, ih], Role::Linear);
            t("indexer.attn_q_b.weight", vec![ql, id * ih], Role::Linear);
        }
    }
    if let Some(en) = &p.engram {
        let width = u(en.width());
        for (li, &il) in en.layers.iter().enumerate() {
            let mut t = |s: &str, shape| add(format!("blk.{il}.{s}"), shape, Role::Other);
            t("engram_q.weight", Some(vec![e, hc]));
            t("engram_k.weight", Some(vec![e, hc]));
            let rows = engram_rows.as_ref().map(|r| vec![u(en.dim), r[li]]);
            t("engram_embd.weight", rows);
            if en.tables[li].0 == types::I8 {
                // FP8 rows' scales (shape checked by the parse).
                t("engram_embd.scale", None);
            }
            let wkv = Some(vec![width, (hc + 1) * e]);
            add(format!("blk.{il}.engram_wkv.weight"), wkv, Role::Linear);
        }
        // Shapes checked by the parse.
        for name in [
            "engram_token_map.weight",
            "engram_primes.weight",
            "engram_offsets.weight",
            "engram_multipliers.weight",
        ] {
            add(name.into(), None, Role::Other);
        }
    }
    out
}

/// Layers whose routed experts [`moe_ffn::supported_glm`] refuses (as the graph would).
fn check_experts(all: &BTreeMap<String, TensorInfo>, n_layer: usize, r: &mut Report) {
    for il in 0..n_layer {
        let get = |s: &str| all.get(&format!("blk.{il}.ffn_{s}_exps.weight"));
        let (Some(up), Some(gate), Some(down)) = (get("up"), get("gate"), get("down")) else {
            continue;
        };
        let shape = |t: &TensorInfo| -> Vec<i64> { t.shape.iter().map(|&d| d as i64).collect() };
        // Descriptors only: nothing is allocated.
        let ctx = Context::new(3, 0);
        let refs = ExpertRefs {
            up: ctx.new_tensor(up.ggml_type, &shape(up)),
            gate: ctx.new_tensor(gate.ggml_type, &shape(gate)),
            down: ctx.new_tensor(down.ggml_type, &shape(down)),
            table: None,
        };
        if !moe_ffn::supported_glm(&refs) {
            r.experts.push(format!(
                "layer {il}: gate {} / up {} / down {}",
                gate.type_name(),
                up.type_name(),
                down.type_name()
            ));
        }
    }
}

/// `t`'s bytes, read from its file; a whole tensor of a safetensors checkpoint that is not one
/// range in it (the Engram hash parameters, say) as its recipe makes it.
fn read(g: &GgufFile, t: &TensorInfo) -> std::io::Result<Vec<u8>> {
    let whole = g
        .tensor(&t.name)
        .is_some_and(|w| w.offset == t.offset && w.size == t.size);
    match g.recipe(&t.name).filter(|_| whole) {
        Some(Recipe::Computed(make)) => return make().map_err(std::io::Error::other),
        Some(Recipe::Gather(parts)) => {
            let mut out = Vec::with_capacity(t.size as usize);
            for p in parts {
                let at = TensorInfo {
                    file_index: p.file,
                    offset: p.offset,
                    size: p.len,
                    name: String::new(),
                    ..t.clone()
                };
                out.extend(read(g, &at)?);
            }
            return Ok(out);
        }
        None => {}
    }
    let path = g
        .shard_paths()
        .nth(t.file_index)
        .expect("a tensor's file")
        .to_path_buf();
    let mut buf = vec![0; t.size as usize];
    std::fs::File::open(&path)?.read_exact_at(&mut buf, t.offset)?;
    Ok(buf)
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|x| i64::from_ne_bytes(*x))
        .collect()
}

/// The hash parameters of `p`'s Engram, read from the file.
struct HashData {
    map: Vec<i64>,
    primes: Vec<i64>,
    offsets: Vec<i64>,
    mult: Vec<i64>,
}

fn hash_data(g: &GgufFile) -> std::io::Result<HashData> {
    let get = |name: &str| read(g, &g.tensor(name).expect("checked by the parse"));
    Ok(HashData {
        map: get("engram_token_map.weight")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_ne_bytes(*b) as i64)
            .collect(),
        primes: i64s(&get("engram_primes.weight")?),
        offsets: i64s(&get("engram_offsets.weight")?),
        mult: i64s(&get("engram_multipliers.weight")?),
    })
}

/// The table rows per Engram layer: the sum of its primes, every column's rows
/// (`offset .. offset + prime`) inside them. `None` without an Engram or when the hash
/// parameters cannot be read (recorded in `r`).
fn read_engram_rows(g: &GgufFile, p: &Ds41Params, r: &mut Report) -> Option<Vec<u64>> {
    let e = p.engram.as_ref()?;
    let h = match hash_data(g) {
        Ok(h) => h,
        Err(err) => {
            r.engram.push(format!("reading the hash parameters: {err}"));
            return None;
        }
    };
    let cols = e.cols();
    let mut rows = Vec::new();
    for (li, &il) in e.layers.iter().enumerate() {
        let at = li * cols..(li + 1) * cols;
        let (primes, offsets) = (&h.primes[at.clone()], &h.offsets[at]);
        if primes.iter().any(|&p| p <= 0) || offsets.iter().any(|&o| o < 0) {
            r.engram.push(format!(
                "layer {il}: primes {primes:?}, offsets {offsets:?}"
            ));
            return None;
        }
        let n: i64 = primes.iter().sum();
        if let Some(c) = (0..cols).find(|&c| offsets[c] + primes[c] > n) {
            r.engram.push(format!(
                "layer {il}: column {c}'s rows {}..{} past the {n} rows",
                offsets[c],
                offsets[c] + primes[c]
            ));
        }
        rows.push(n as u64);
    }
    if let Some(&bad) = h.map.iter().find(|&&t| t < 0) {
        r.engram.push(format!("a negative token map entry ({bad})"));
    }
    Some(rows)
}

/// Hashes `samples` tokens through every Engram layer and reads and dequantizes their rows:
/// the rows read, as (Engram layer, row) with their bytes, unless a problem stopped it.
fn sample_engram(
    g: &GgufFile,
    e: &EngramParams,
    samples: usize,
    r: &mut Report,
) -> Option<Vec<(usize, i64, Vec<u8>)>> {
    let h = match hash_data(g) {
        Ok(h) => h,
        Err(err) => {
            r.engram.push(format!("reading the hash parameters: {err}"));
            return None;
        }
    };
    let hash = Hash {
        map: &h.map,
        primes: h.primes.clone(),
        offsets: h.offsets.clone(),
        mult: h.mult.clone(),
        e,
    };
    let n_vocab = h.map.len().max(1) as u64;
    // Tokens spread over the vocabulary, the first after the sequence start.
    let tokens: Vec<u32> = (0..samples as u64)
        .map(|i| (i.wrapping_mul(0x9E37_79B9) % n_vocab) as u32)
        .collect();
    let mut prev = vec![NO_TOKEN; e.ngram - 1];
    let mut ids = Vec::new();
    let mut files = BTreeSet::new();
    let mut rows = Vec::new();
    let mut max_abs = 0f32;
    for &tok in &tokens {
        for (li, &il) in e.layers.iter().enumerate() {
            let name = format!("blk.{il}.engram_embd.weight");
            let t = g.tensor(&name)?;
            files.insert(t.file_index);
            hash.ids(li, tok, &prev, &mut ids);
            let row_bytes = crate::ggml::row_size(t.ggml_type, e.dim as usize) as u64;
            let n_rows = t.shape.get(1).copied().unwrap_or(0) as i64;
            for &id in &ids {
                if !(0..n_rows).contains(&id) {
                    r.engram
                        .push(format!("{name}: row {id} outside its {n_rows} rows"));
                    return None;
                }
                let at = TensorInfo {
                    offset: t.offset + id as u64 * row_bytes,
                    size: row_bytes,
                    ..t.clone()
                };
                let b = match read(g, &at) {
                    Ok(b) => b,
                    Err(err) => {
                        r.engram.push(format!("{name}: reading row {id}: {err}"));
                        return None;
                    }
                };
                let v = if t.ggml_type == types::I8 {
                    // FP8 rows: with their scale bytes (one per 32).
                    let s = g.tensor(&format!("blk.{il}.engram_embd.scale"))?;
                    let n = e.dim as u64 / 32;
                    let at = TensorInfo {
                        offset: s.offset + id as u64 * n,
                        size: n,
                        ..s
                    };
                    match read(g, &at) {
                        Ok(scale) => super::fp8_row_f32(&b, &scale),
                        Err(err) => {
                            r.engram
                                .push(format!("{name}: reading row {id}'s scales: {err}"));
                            return None;
                        }
                    }
                } else {
                    super::row_f32(t.ggml_type, &b, e.dim as usize)
                };
                if let Some(x) = v.iter().find(|x| !x.is_finite()) {
                    r.engram.push(format!("{name}: row {id} holds {x}"));
                    return None;
                }
                max_abs = v.iter().fold(max_abs, |m, x| m.max(x.abs()));
                rows.push((li, id, b));
            }
        }
        prev.remove(0);
        prev.push(tok);
    }
    let paths: Vec<_> = g.shard_paths().collect();
    for f in files {
        r.engram_notes
            .push(format!("tables read from {}", paths[f].display()));
    }
    r.engram_notes.push(format!(
        "{} rows of {} tokens hashed, read and dequantized: all finite, max |x| {max_abs}",
        rows.len(),
        tokens.len()
    ));
    Some(rows)
}

/// Opens the Engram lookup table over `g`'s tables with `opts` as the engine does and reads
/// `rows` (from [`sample_engram`], with the file's bytes) through it.
fn engram_table(
    g: &GgufFile,
    hp: &Hparams,
    e: &EngramParams,
    rows: &[(usize, i64, Vec<u8>)],
    opts: &TableOptions,
    r: &mut Report,
) {
    let Some((parts, _)) = super::engram::lookup_table(g, hp) else {
        r.engram
            .push("no lookup table takes the Engram tables: they load as weights".into());
        return;
    };
    let bytes: u64 = parts.iter().map(|p| p.bytes()).sum();
    let table = match LookupTable::open_parts(parts, opts, None) {
        Ok(t) => t,
        Err(err) => {
            r.engram
                .push(format!("opening the Engram lookup table: {err}"));
            return;
        }
    };
    let c = table.choice();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    r.engram_notes.push(format!(
        "lookup table over {} parts, {:.1} GiB: {:?}{}, {:.1} GiB row cache, opened in {:.2} s ({})",
        e.layers.len(),
        gib(bytes),
        c.io,
        if c.locked { ", locked" } else { "" },
        gib(c.ram_cache_bytes),
        c.seconds,
        c.reason
    ));
    let bases = super::engram::bases(e);
    let ids: Vec<u32> = rows
        .iter()
        .map(|&(li, id, _)| (bases[li] + id as u64) as u32)
        .collect();
    let got = match table.fetch(&ids) {
        Ok(b) => b,
        Err(err) => {
            r.engram
                .push(format!("reading rows through the lookup table: {err}"));
            return;
        }
    };
    let want: Vec<u8> = rows
        .iter()
        .flat_map(|(_, _, b)| b.iter().copied())
        .collect();
    if got == want {
        r.engram_notes.push(format!(
            "{} rows read through the lookup table: the file's bytes",
            rows.len()
        ));
    } else {
        r.engram
            .push("rows read through the lookup table differ from the file's".into());
    }
}
