//! ModelOpt NVFP4 / FP8 checkpoints (`quant_method: modelopt`) as llama.cpp's converter writes
//! them (`conversion/base.py`, MIT: `_nvfp4_pack`, `_repack_nvfp4`, `_generate_nvfp4_tensors`,
//! `dequant_model`):
//!
//! - An NVFP4 layer (`weight` U8 `[rows, k/2]` of packed E2M1 pairs, `weight_scale` E4M3
//!   `[rows, k/16]`, F32 `weight_scale_2` and `input_scale`) becomes a `GGML_TYPE_NVFP4` tensor
//!   (per 64 elements: the four sub-block scales with the sign bit cleared, then 32 bytes of codes
//!   with element `j` in the low and `j + 8` in the high nibble of each 16-element sub-block),
//!   plus F32 `<name>.scale` (`weight_scale_2`) and `<name>.input_scale` tensors of shape `[1]`,
//!   each left out when it is 1. Layers quantized as `W4A16_NVFP4` are listed in
//!   `general.tensor_extra.prec_a4` as `false` (sorted by name).
//! - An FP8 layer (`weight` E4M3, F32 per-tensor `weight_scale`) stays FP8 by default
//!   ([`super::Fp8Mode::Raw`]): its bytes (rows selected or permuted like any weight's) in an
//!   `I8` tensor of the weight's shape, the F32 `<name>.scale` (`weight_scale`, `[1]`, always
//!   written), the F32 `<name>.input_scale` (`[1]`, when the checkpoint has one: the static
//!   activation scale of its W8A8 scheme) and its name in `legion.tensor_extra.fp8_e4m3`. The
//!   graph multiplies the matmul result by `.scale` as for NVFP4, so an FP8 matmul computes
//!   `W_e4m3 @ x` only ([`crate::model::fp8`]).
//!   With [`super::Fp8Mode::Bf16`] the weight is dequantized as llama.cpp's converter does,
//!   `f32(weight) * weight_scale` stored as BF16 (round to nearest even, as gguf-py does).
//!
//! compressed-tensors checkpoints (`quant_method: compressed-tensors`, llm-compressor's
//! `nvfp4-pack-quantized` and `float-quantized` groups) load the same way, as shipped:
//!
//! - NVFP4 weights are `weight_packed` (the same bytes as ModelOpt's `weight`) with E4M3
//!   `weight_scale`; their F32 `weight_global_scale` and `input_global_scale` are what a value
//!   is divided by, so `.scale` and `.input_scale` are their reciprocals.
//! - FP8 weights may have one `weight_scale` per output row (`strategy: channel`, `[rows, 1]`):
//!   `.scale` is then F32 `[rows]` (rows selected as the weight's), applied per row by the FP8
//!   matmul's epilogue. Activations scaled per token at run time (`dynamic: true`) have no
//!   `input_scale`, so those matmuls run W8A16.
//!
//! The mapping code (`qwen35.rs`) does not know about either: [`Builder::direct`],
//! [`Builder::gather_rows`] and [`Builder::computed_raw`] hand quantized sources to this module,
//! which applies the same row selection or byte transform to the quantized data first.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::Value as Json;

use super::Builder;
use crate::ggml::{ffi, types};
use crate::gguf::{Compute, Recipe, Value};
use crate::safetensors::{Checkpoint, Dtype};

/// Elements per NVFP4 sub-block (one E4M3 scale each) and per ggml block.
const SUB: usize = 16;
const BLOCK: usize = 64;
/// Bytes of one ggml `block_nvfp4`.
const BLOCK_BYTES: usize = 36;

/// The checkpoint's ModelOpt quantization: the global algorithm and the per-module ones.
#[derive(Debug, Clone, Default)]
pub(super) struct ModelOpt {
    global: Option<String>,
    layers: HashMap<String, String>,
    /// GGUF tensor names of NVFP4 weights and whether they may use 4-bit activations.
    prec_a4: BTreeMap<String, bool>,
    /// How FP8 weights load, and the GGUF names of those kept raw.
    fp8_mode: super::Fp8Mode,
    fp8_raw: Vec<String>,
    /// compressed-tensors names and conventions ([`Self::compressed`]) instead of ModelOpt's.
    compressed: bool,
}

/// The metadata key listing the FP8 E4M3 weights kept raw (`I8` tensors).
pub const KEY_FP8_E4M3: &str = "legion.tensor_extra.fp8_e4m3";

/// A byte transform of a source tensor (a column permutation), applied before dequantizing.
pub(super) type Transform = Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// How the converter treats one source weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Nvfp4,
    Fp8,
}

impl ModelOpt {
    /// The ModelOpt settings of `ckpt`, from `config.json`'s `quantization_config` or
    /// `hf_quant_config.json`; `None` for checkpoints that are not ModelOpt-quantized.
    pub(super) fn detect(
        ckpt: &Checkpoint,
        config: &Json,
        fp8_mode: super::Fp8Mode,
    ) -> Result<Option<Self>, String> {
        let mut qc = config
            .get("quantization_config")
            .cloned()
            .unwrap_or(Json::Null);
        let mut method = qc
            .get("quant_method")
            .and_then(Json::as_str)
            .map(str::to_string);
        let file = ckpt.dir().join("hf_quant_config.json");
        let needs_file = qc.get("quant_algo").is_none() || qc.get("quantized_layers").is_none();
        if needs_file && file.is_file() {
            let text = std::fs::read_to_string(&file).map_err(|e| format!("{file:?}: {e}"))?;
            let hf: Json = serde_json::from_str(&text).map_err(|e| format!("{file:?}: {e}"))?;
            if method.is_none() {
                method = hf
                    .pointer("/producer/name")
                    .and_then(Json::as_str)
                    .map(str::to_lowercase);
            }
            if let Some(q) = hf.get("quantization") {
                qc = q.clone();
            }
        }
        if method.as_deref() == Some("compressed-tensors") {
            return Ok(Some(Self::compressed(&qc, fp8_mode)));
        }
        if !matches!(method.as_deref(), Some("modelopt" | "modelopt_mixed")) {
            return Ok(None);
        }
        let layers = qc
            .get("quantized_layers")
            .and_then(Json::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        Some((k.clone(), v.get("quant_algo")?.as_str()?.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Some(Self {
            global: qc
                .get("quant_algo")
                .and_then(Json::as_str)
                .map(str::to_string),
            layers,
            prec_a4: BTreeMap::new(),
            fp8_mode,
            fp8_raw: Vec::new(),
            compressed: false,
        }))
    }

    /// A compressed-tensors checkpoint (`quant_method: compressed-tensors`, as llm-compressor
    /// writes them). Its tensors say what each layer is (see [`Builder::quant_kind`]); the
    /// config only says whether NVFP4 layers were quantized for 4-bit activations: when an
    /// `nvfp4-pack-quantized` group has none, every NVFP4 weight is listed as `W4A16_NVFP4`.
    fn compressed(qc: &Json, fp8_mode: super::Fp8Mode) -> Self {
        let groups = qc.get("config_groups").and_then(Json::as_object);
        let w4a16 = groups.is_some_and(|g| {
            g.values().any(|g| {
                let bits = |k: &str| g.pointer(&format!("/{k}/num_bits")).and_then(Json::as_u64);
                let fp4 = g.get("format").and_then(Json::as_str) == Some("nvfp4-pack-quantized")
                    || bits("weights") == Some(4);
                fp4 && bits("input_activations") != Some(4)
            })
        });
        Self {
            global: w4a16.then(|| "W4A16_NVFP4".to_string()),
            fp8_mode,
            compressed: true,
            ..Self::default()
        }
    }

    /// The quantization algorithm of module `name` (`...mlp.up_proj`): its own entry, else its
    /// nearest listed parent's, else the global one (the converter's `_tag_prec_a4`).
    fn algo(&self, module: &str) -> Option<&str> {
        let mut m = module;
        loop {
            if let Some(a) = self.layers.get(m) {
                return Some(a);
            }
            match m.rfind('.') {
                Some(i) => m = &m[..i],
                None => return self.global.as_deref(),
            }
        }
    }

    /// The `general.tensor_extra.*` keys the converter writes for the NVFP4 tensors, and the
    /// list of raw FP8 tensors.
    pub(super) fn metadata(&self) -> Vec<(&'static str, Value)> {
        let mut kv = Vec::new();
        if !self.prec_a4.is_empty() {
            let names = self.prec_a4.keys().cloned().map(Value::String).collect();
            let values = self.prec_a4.values().map(|&v| Value::Bool(v)).collect();
            kv.push(("general.tensor_extra.name", Value::Array(names)));
            kv.push(("general.tensor_extra.prec_a4", Value::Array(values)));
        }
        if !self.fp8_raw.is_empty() {
            let names = self.fp8_raw.iter().cloned().map(Value::String).collect();
            kv.push((KEY_FP8_E4M3, Value::Array(names)));
        }
        kv
    }
}

/// `weight` -> `weight_scale` etc.: the sibling tensor of the weight `src`.
fn sibling(src: &str, suffix: &str) -> Option<String> {
    src.strip_suffix(".weight").map(|b| format!("{b}.{suffix}"))
}

/// An FP8 E4M3 (`float8_e4m3fn`) value as f32: no infinities, `0x7f`/`0xff` are NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let man = (b & 7) as f32;
    if b & 0x7f == 0x7f {
        return f32::NAN;
    }
    sign * if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// f32 to BF16 bits as gguf-py's `quants.quantize(.., BF16)` rounds: to nearest even, NaNs kept
/// quiet.
pub fn f32_to_bf16(v: f32) -> u16 {
    let mut n = v.to_bits();
    if n & 0x7fff_ffff > 0x7f80_0000 {
        n = (n & 0xffff_0000) | (64 << 16);
    }
    ((n as u64 + (0x7fff + ((n as u64 >> 16) & 1))) >> 16) as u16
}

/// FP8 E4M3 bytes times the per-tensor `scale`, as BF16 bytes (`dequant_simple` + BF16 output).
pub fn dequant_fp8_bf16(bytes: &[u8], scale: f32) -> Vec<u8> {
    let lut: Vec<f32> = (0..=255u8).map(e4m3_to_f32).collect();
    bytes
        .iter()
        .flat_map(|&b| f32_to_bf16(lut[b as usize] * scale).to_le_bytes())
        .collect()
}

/// ModelOpt NVFP4 rows (`weight`: `rows` x `k/2` packed bytes, element `2i` in the low nibble;
/// `scale`: `rows` x `k/16` E4M3 bytes) as ggml `block_nvfp4` rows (`_nvfp4_pack`).
pub fn repack_nvfp4(weight: &[u8], scale: &[u8], rows: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0; rows * k / BLOCK * BLOCK_BYTES];
    repack_nvfp4_into(weight, scale, rows, k, &mut out);
    out
}

/// [`repack_nvfp4`] into `out` (`rows * k / 64 * 36` bytes).
pub fn repack_nvfp4_into(weight: &[u8], scale: &[u8], rows: usize, k: usize, out: &mut [u8]) {
    assert_eq!(k % BLOCK, 0, "NVFP4 rows hold whole 64-element blocks");
    assert_eq!(weight.len(), rows * k / 2);
    assert_eq!(scale.len(), rows * k / SUB);
    assert_eq!(out.len(), rows * k / BLOCK * BLOCK_BYTES);
    let blocks = weight
        .as_chunks::<{ BLOCK / 2 }>()
        .0
        .iter()
        .zip(scale.as_chunks::<{ BLOCK / SUB }>().0);
    for ((wb, sb), o) in blocks.zip(out.as_chunks_mut::<BLOCK_BYTES>().0) {
        for (d, &s) in o.iter_mut().zip(sb) {
            *d = s & 0x7f;
        }
        let (subs, _) = wb.as_chunks::<{ SUB / 2 }>();
        for (sub, o) in subs
            .iter()
            .zip(o[BLOCK / SUB..].as_chunks_mut::<{ SUB / 2 }>().0)
        {
            // Element e of the sub-block is nibble e % 2 of byte e / 2.
            let code = |e: usize| (sub[e / 2] >> (4 * (e % 2))) & 0xf;
            for (j, d) in o.iter_mut().enumerate() {
                *d = code(j) | code(j + SUB / 2) << 4;
            }
        }
    }
}

/// A scalar F32 source tensor's value.
pub(super) fn scalar(ckpt: &Checkpoint, name: &str) -> Result<f32, String> {
    let (_, t) = ckpt
        .locate(name)
        .ok_or_else(|| format!("missing `{name}`"))?;
    let bytes = ckpt.data(name).map_err(|e| e.to_string())?;
    match (&t.dtype, bytes.len()) {
        (Dtype::F32, 4) => Ok(f32::from_le_bytes(bytes.try_into().unwrap())),
        (Dtype::BF16, 2) => Ok(f32::from_bits(
            (u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16,
        )),
        _ => Err(format!(
            "`{name}`: expected one F32 value, got {} {:?}",
            t.dtype.name(),
            t.shape
        )),
    }
}

/// A float source tensor's values, whatever its shape.
fn floats(ckpt: &Checkpoint, name: &str) -> Result<Vec<f32>, String> {
    let (_, t) = ckpt
        .locate(name)
        .ok_or_else(|| format!("missing `{name}`"))?;
    let bytes = ckpt.data(name).map_err(|e| e.to_string())?;
    super::to_f32(&t.dtype, bytes).ok_or_else(|| format!("`{name}`: not a float tensor"))
}

impl Builder {
    /// How the converter treats the weight `src`, when the checkpoint is ModelOpt-quantized and
    /// `src` is one of its quantized weights.
    fn quant_kind(&self, src: &str) -> Option<Kind> {
        let opt = self.modelopt.as_ref()?;
        if opt.compressed {
            // NVFP4 weights are `weight_packed` (no `weight`); FP8 ones keep `weight`, E4M3.
            let has = |suffix: &str| sibling(src, suffix).is_some_and(|s| self.has(&s));
            if !self.has(src) {
                let packed =
                    has("weight_packed") && has("weight_scale") && has("weight_global_scale");
                return packed.then_some(Kind::Nvfp4);
            }
            let (_, w) = self.ckpt.locate(src)?;
            return (w.dtype == Dtype::F8E4M3 && has("weight_scale")).then_some(Kind::Fp8);
        }
        let (_, scale) = self.ckpt.locate(&sibling(src, "weight_scale")?)?;
        let (_, w) = self.ckpt.locate(src)?;
        match w.dtype {
            Dtype::F8E4M3 => Some(Kind::Fp8),
            _ if scale.shape.len() >= 2 => Some(Kind::Nvfp4),
            _ => None,
        }
    }

    /// Handles `src` when it is quantized: `name` (ggml `shape` of the unquantized tensor; by
    /// default the source's) is the source rows `rows` (`None`: all) with `bytes` (if any)
    /// applied to the quantized data. Returns `None` for plain sources.
    pub(super) fn quantized(
        &mut self,
        name: &str,
        src: &str,
        rows: Option<&[(u64, u64)]>,
        bytes: Option<Transform>,
        shape: Option<Vec<u64>>,
    ) -> Option<Result<(), String>> {
        let kind = self.quant_kind(src)?;
        let (_, t) = self.ckpt.locate(&self.packed_src(src))?;
        let shape = shape.unwrap_or_else(|| match kind {
            // Two E2M1 codes per byte along the row.
            Kind::Nvfp4 => vec![t.shape[1] as u64 * 2, t.shape[0] as u64],
            Kind::Fp8 => super::ggml_shape(&t.shape),
        });
        Some(match kind {
            Kind::Fp8 => self.fp8(name, src, rows, bytes, shape),
            Kind::Nvfp4 if bytes.is_some() => Err(format!(
                "`{src}`: NVFP4 weights with a column transform are not supported yet"
            )),
            Kind::Nvfp4 => self.nvfp4(name, src, rows, shape),
        })
    }

    /// The tensor holding the quantized data of the weight `src`: compressed-tensors' NVFP4
    /// `weight_packed` when there is no `weight`.
    fn packed_src(&self, src: &str) -> String {
        match sibling(src, "weight_packed") {
            Some(p) if !self.has(src) && self.has(&p) => p,
            _ => src.to_string(),
        }
    }

    fn fp8(
        &mut self,
        name: &str,
        src: &str,
        rows: Option<&[(u64, u64)]>,
        bytes: Option<Transform>,
        shape: Vec<u64>,
    ) -> Result<(), String> {
        let (_, t) = self.source(src)?;
        // One scale for the tensor (ModelOpt), or one per output row (compressed-tensors'
        // channel-wise FP8, `[rows, 1]`), the rows selected as the weight's.
        let row = t.nbytes() / t.shape[0];
        let scale = floats(&self.ckpt, &sibling(src, "weight_scale").unwrap())?;
        let scale = match (scale.len(), rows) {
            (1, _) => scale,
            (n, None) if n == t.shape[0] => scale,
            (n, Some(r)) if n == t.shape[0] => r
                .iter()
                .flat_map(|&(first, count)| &scale[first as usize..(first + count) as usize])
                .copied()
                .collect(),
            (n, _) => {
                return Err(format!(
                    "`{src}`: {n} weight scales for {} rows (expected 1 or one per row)",
                    t.shape[0]
                ));
            }
        };
        // The static per-tensor activation scale of the checkpoint's W8A8 scheme, for the FP8
        // matmul's `--fp8-act a8` path.
        let input_scale = match sibling(src, "input_scale").filter(|s| self.has(s)) {
            Some(input) => Some(scalar(&self.ckpt, &input)?),
            None => None,
        };
        let raw = self.modelopt.as_ref().unwrap().fp8_mode == super::Fp8Mode::Raw;
        let rows = rows.map(<[_]>::to_vec);
        let ckpt = self.ckpt.clone();
        let src = src.to_string();
        let scales = scale.clone();
        let compute: Compute = Arc::new(move || {
            let data = ckpt.data(&src).map_err(|e| e.to_string())?;
            let gathered;
            let data = match &rows {
                Some(rows) => {
                    gathered = gather(data, rows, row);
                    &gathered[..]
                }
                None => data,
            };
            let data = match &bytes {
                Some(f) => f(data),
                None => data.to_vec(),
            };
            Ok(if raw {
                data
            } else if let [s] = scales[..] {
                dequant_fp8_bf16(&data, s)
            } else {
                data.chunks_exact(row)
                    .zip(&scales)
                    .flat_map(|(r, &s)| dequant_fp8_bf16(r, s))
                    .collect()
            })
        });
        if !raw {
            return self.add_computed(name, types::BF16, shape, compute);
        }
        self.add_computed(name, types::I8, shape, compute)?;
        let out = sibling(name, "scale").ok_or_else(|| format!("`{name}`: no .weight"))?;
        let n = scale.len() as u64;
        let s: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_computed(&out, types::F32, vec![n], Arc::new(move || Ok(s.clone())))?;
        if let Some(v) = input_scale {
            let out =
                sibling(name, "input_scale").ok_or_else(|| format!("`{name}`: no .weight"))?;
            let b = v.to_le_bytes().to_vec();
            self.add_computed(&out, types::F32, vec![1], Arc::new(move || Ok(b.clone())))?;
        }
        self.modelopt
            .as_mut()
            .unwrap()
            .fp8_raw
            .push(name.to_string());
        Ok(())
    }

    fn nvfp4(
        &mut self,
        name: &str,
        src: &str,
        rows: Option<&[(u64, u64)]>,
        shape: Vec<u64>,
    ) -> Result<(), String> {
        let scale_src = sibling(src, "weight_scale").unwrap();
        let packed = self.packed_src(src);
        let (_, w) = self.source(&packed)?;
        let (_, s) = self.source(&scale_src)?;
        let k = w.shape[1] * 2;
        if s.shape != [w.shape[0], k / SUB] || k % BLOCK != 0 {
            return Err(format!(
                "`{src}`: NVFP4 weight {:?} and scales {:?} do not match",
                w.shape, s.shape
            ));
        }
        if shape.first() != Some(&(k as u64)) {
            return Err(format!("`{src}`: {k} columns, expected shape {shape:?}"));
        }
        let rows = rows.map(<[_]>::to_vec);
        let ckpt = self.ckpt.clone();
        let (src_w, src_s) = (packed, scale_src);
        let compute: Compute = Arc::new(move || {
            let w = ckpt.data(&src_w).map_err(|e| e.to_string())?;
            let s = ckpt.data(&src_s).map_err(|e| e.to_string())?;
            Ok(match &rows {
                Some(r) => {
                    let (w, s) = (gather(w, r, k / 2), gather(s, r, k / SUB));
                    repack_nvfp4(&w, &s, w.len() / (k / 2), k)
                }
                None => repack_nvfp4(w, s, w.len() / (k / 2), k),
            })
        });
        self.add_computed(name, types::NVFP4, shape, compute)?;

        // Per-tensor scales, written only when they are not 1 (`_write_scale_tensor`).
        // compressed-tensors stores their reciprocals, the `*_global_scale` a value is divided by.
        let compressed = self.modelopt.as_ref().unwrap().compressed;
        let scales = if compressed {
            [
                ("weight_global_scale", "scale"),
                ("input_global_scale", "input_scale"),
            ]
        } else {
            [("weight_scale_2", "scale"), ("input_scale", "input_scale")]
        };
        for (suffix, gguf_suffix) in scales {
            let Some(s) = sibling(src, suffix).filter(|s| self.has(s)) else {
                continue;
            };
            let v = scalar(&self.ckpt, &s)?;
            let v = if compressed { 1.0 / v } else { v };
            if (v - 1.0).abs() < 1e-6 {
                continue;
            }
            let out = sibling(name, gguf_suffix).ok_or_else(|| format!("`{name}`: no .weight"))?;
            let bytes = v.to_le_bytes().to_vec();
            self.add_computed(
                &out,
                types::F32,
                vec![1],
                Arc::new(move || Ok(bytes.clone())),
            )?;
        }
        let module = src.strip_suffix(".weight").unwrap_or(src);
        let opt = self.modelopt.as_mut().unwrap();
        if opt.algo(module) == Some("W4A16_NVFP4") {
            opt.prec_a4.insert(name.to_string(), false);
        }
        Ok(())
    }

    pub(super) fn add_computed(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: Vec<u64>,
        compute: Compute,
    ) -> Result<(), String> {
        self.push(name, ty, shape, 0, 0);
        self.recipes
            .insert(name.to_string(), Recipe::Computed(compute));
        Ok(())
    }
}

/// The row ranges `(first, count)` of `data` (rows of `row` bytes), back to back.
fn gather(data: &[u8], rows: &[(u64, u64)], row: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for &(r, n) in rows {
        let (a, b) = (r as usize * row, (r + n) as usize * row);
        out.extend_from_slice(&data[a..b]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Made by llama.cpp's converter (f336359) with `tests/data/nvfp4/make_vectors.py`:
    // `ModelBase._nvfp4_pack` on a random 3 x 128 ModelOpt weight and its E4M3 scales (sign bits
    // set on some), and `dequant_simple` + `gguf.quants.quantize(.., BF16)` on every E4M3 value
    // (NaNs replaced by 0) times 0.0123456789.
    const W: &str = include_str!("../../tests/data/nvfp4/w.hex");
    const S: &str = include_str!("../../tests/data/nvfp4/s.hex");
    const OUT: &str = include_str!("../../tests/data/nvfp4/out.hex");
    const SCALE: &str = include_str!("../../tests/data/nvfp4/scale.hex");
    const BF16: &str = include_str!("../../tests/data/nvfp4/bf16.hex");

    #[test]
    fn repack_matches_the_converter() {
        let got = repack_nvfp4(&hex(W.trim()), &hex(S.trim()), 3, 128);
        assert_eq!(got, hex(OUT.trim()));
    }

    #[test]
    fn fp8_dequant_matches_the_converter() {
        let scale = f32::from_le_bytes(hex(SCALE.trim()).try_into().unwrap());
        let mut codes: Vec<u8> = (0..=255).collect();
        codes[0x7f] = 0;
        codes[0xff] = 0;
        assert_eq!(dequant_fp8_bf16(&codes, scale), hex(BF16.trim()));
    }

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert!(e4m3_to_f32(0x7f).is_nan());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80); // tie, even stays
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82); // tie, odd rounds up
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(f32_to_bf16(f32::NAN) & 0x7fc0, 0x7fc0);
    }

    #[test]
    fn algo_falls_back_to_parents_then_global() {
        let m = ModelOpt {
            global: Some("MIXED_PRECISION".into()),
            layers: [
                (
                    "model.layers.0.mlp.up_proj".to_string(),
                    "W4A16_NVFP4".into(),
                ),
                ("model.layers.1".to_string(), "NVFP4".into()),
            ]
            .into(),
            ..Default::default()
        };
        assert_eq!(m.algo("model.layers.0.mlp.up_proj"), Some("W4A16_NVFP4"));
        assert_eq!(m.algo("model.layers.1.mlp.up_proj"), Some("NVFP4"));
        assert_eq!(m.algo("lm_head"), Some("MIXED_PRECISION"));
    }
}
