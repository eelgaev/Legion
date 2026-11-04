//! Hugging Face safetensors checkpoints, opened as the GGUF llama.cpp's converter would write.
//!
//! [`open`] reads a checkpoint directory (`config.json`, the safetensors shards, `tokenizer.json`,
//! `tokenizer_config.json`, `generation_config.json` for the sampling defaults) and builds a synthetic [`GgufFile`]: metadata keys, the vocabulary and
//! tensor infos are what `convert_hf_to_gguf.py` (llama.cpp, MIT) would put in a GGUF of the
//! same checkpoint, so everything downstream (hyperparameters, tokenizer, chat template, the model
//! graphs) reads it unchanged. Tensor data is never converted ahead of time:
//!
//! - A tensor the converter only renames (or reshapes) points at its range in the shard; plain-CPU
//!   loads map it like any GGUF tensor.
//! - A tensor whose rows the converter reorders, or that stacks several source tensors (MoE
//!   experts), is a [`Recipe::Gather`] of shard ranges, streamed in like a plain range.
//! - Small elementwise transforms (`-exp(A_log)`, `norm + 1`, BF16 -> F32) and column
//!   permutations are [`Recipe::Computed`] at load time.
//!
//! Matching the converter tensor for tensor is the contract: `tests/hf_checkpoint.rs` checks it
//! on synthetic checkpoints, and the real-model test compares a converted GGUF byte for byte.

mod deepseek41;
mod gemma4;
mod glm5next;
mod nemotron_h;
pub mod nvfp4;
mod qwen35;
mod qwen4;
mod vocab;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value as Json;

use crate::ggml::{ffi, types};
use crate::gguf::{Compute, FileRange, GgufError, GgufFile, GgufWriter, Recipe, TensorInfo, Value};
use crate::safetensors::{Checkpoint, Dtype};

/// How FP8 weights of a quantized checkpoint are loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Fp8Mode {
    /// As shipped: the E4M3 bytes in an `I8` tensor plus the per-tensor `<name>.scale`, for
    /// Legion's FP8 matmul (see [`nvfp4`]).
    #[default]
    Raw,
    /// Dequantized to BF16 at load, as llama.cpp's converter does. A temporary escape hatch
    /// until the FP8 matmul lands: twice the memory and bandwidth.
    Bf16,
}

/// Options for opening a checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenOptions {
    pub fp8: Fp8Mode,
}

/// Opens the checkpoint in `dir` as a synthetic GGUF model.
pub(crate) fn open(dir: &Path, opts: &OpenOptions) -> Result<GgufFile, GgufError> {
    let err = |reason: String| GgufError::Checkpoint {
        path: dir.to_path_buf(),
        reason,
    };
    let ckpt = Checkpoint::open(dir).map_err(|e| err(e.to_string()))?;
    let config = ckpt
        .config()
        .cloned()
        .ok_or_else(|| err("no config.json".into()))?;
    let model_type = config
        .get("model_type")
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    let mut b = Builder::new(Arc::new(ckpt));
    b.modelopt = nvfp4::ModelOpt::detect(&b.ckpt, &config, opts.fp8).map_err(err)?;
    match model_type.as_str() {
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
            qwen35::build(&mut b, &config).map_err(err)?
        }
        "nemotron_h" => nemotron_h::build(&mut b, &config).map_err(err)?,
        "gemma4" | "gemma4_text" => gemma4::build(&mut b, &config).map_err(err)?,
        "qwen4_exp" | "qwen4_exp_text" => qwen4::build(&mut b, &config).map_err(err)?,
        "glm5_next" | "glm5_next_text" => glm5next::build(&mut b, &config).map_err(err)?,
        "deepseek_v41" | "deepseek_v41_text" => deepseek41::build(&mut b, &config).map_err(err)?,
        other => {
            return Err(err(format!(
                "model_type `{other}` is not supported from safetensors yet (supported: qwen3_5, qwen3_5_moe, qwen4_exp, nemotron_h, gemma4, glm5_next, deepseek_v41)"
            )));
        }
    }
    // The sampling defaults the checkpoint declares, as `general.sampling.*` keys.
    if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) {
        let j: Json =
            serde_json::from_str(&text).map_err(|e| err(format!("generation_config.json: {e}")))?;
        let layer = crate::engine::sampling_defaults::SamplingLayer::from_generation_config(&j);
        for (k, v) in layer.metadata() {
            b.set(k, v).map_err(err)?;
        }
    }
    b.finish(dir.to_path_buf())
}

/// Opens the MTP head of the checkpoint in `dir` as a synthetic head GGUF of its own, as
/// llama.cpp's converter writes it for `--spec-draft` (Qwen3.8-Flash-Next's `mtp.*`; see
/// [`qwen4::build_mtp`]). Its tensors stay as shipped.
pub fn open_mtp(dir: &Path, opts: &OpenOptions) -> Result<GgufFile, GgufError> {
    let err = |reason: String| GgufError::Checkpoint {
        path: dir.to_path_buf(),
        reason,
    };
    let ckpt = Checkpoint::open(dir).map_err(|e| err(e.to_string()))?;
    let config = ckpt
        .config()
        .cloned()
        .ok_or_else(|| err("no config.json".into()))?;
    let model_type = config
        .get("model_type")
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    let mut b = Builder::new(Arc::new(ckpt));
    b.modelopt = nvfp4::ModelOpt::detect(&b.ckpt, &config, opts.fp8).map_err(err)?;
    match model_type.as_str() {
        "qwen4_exp" | "qwen4_exp_text" => qwen4::build_mtp(&mut b, &config).map_err(err)?,
        other => {
            return Err(err(format!(
                "model_type `{other}`: an MTP head from safetensors is supported for qwen4_exp only"
            )));
        }
    }
    b.finish(dir.to_path_buf())
}

/// Collects metadata, tensor infos and recipes for one synthetic model.
pub(crate) struct Builder {
    ckpt: Arc<Checkpoint>,
    meta: GgufWriter,
    tensors: Vec<TensorInfo>,
    recipes: HashMap<String, Recipe>,
    /// ModelOpt NVFP4 / FP8 quantization, repacked or dequantized as the converter does.
    modelopt: Option<nvfp4::ModelOpt>,
    /// Stacked NVFP4 experts' per-expert sources ([`GgufFile::nvfp4_stack`]).
    stacks: HashMap<String, Arc<crate::gguf::Nvfp4Stack>>,
}

/// The ggml type that holds a safetensors dtype unchanged.
fn ggml_type(d: &Dtype) -> Option<ffi::ggml_type> {
    Some(match d {
        Dtype::F32 => types::F32,
        Dtype::F16 => types::F16,
        Dtype::BF16 => types::BF16,
        _ => return None,
    })
}

impl Builder {
    fn new(ckpt: Arc<Checkpoint>) -> Self {
        Self {
            ckpt,
            meta: GgufWriter::new(),
            tensors: Vec::new(),
            recipes: HashMap::new(),
            modelopt: None,
            stacks: HashMap::new(),
        }
    }

    fn set(&mut self, key: &str, value: Value) -> Result<(), String> {
        self.meta
            .set(key, &value)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn has(&self, src: &str) -> bool {
        self.ckpt.locate(src).is_some()
    }

    /// The source tensor's shard, dtype and shape (row-major, outermost first).
    fn source(&self, src: &str) -> Result<(usize, crate::safetensors::TensorInfo), String> {
        self.ckpt
            .locate(src)
            .map(|(i, t)| (i, t.clone()))
            .ok_or_else(|| format!("missing tensor `{src}`"))
    }

    fn push(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        mut shape: Vec<u64>,
        file: usize,
        offset: u64,
    ) {
        // GGUF readers see trailing size-1 dims trimmed; match them.
        while shape.len() > 1 && shape.last() == Some(&1) {
            shape.pop();
        }
        let size = crate::ggml::row_size(ty, shape[0] as usize) as u64
            * shape[1..].iter().product::<u64>();
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            ggml_type: ty,
            shape,
            file_index: file,
            offset,
            size,
        });
    }

    /// `name` is `src` as it is in the file, with its dims reversed into ggml order; `shape`
    /// (ggml order) may differ only by dropped size-1 dims.
    fn direct(&mut self, name: &str, src: &str, shape: Option<Vec<u64>>) -> Result<(), String> {
        // First: a compressed-tensors NVFP4 weight has no `src` tensor, only `weight_packed`.
        if let Some(r) = self.quantized(name, src, None, None, shape.clone()) {
            return r;
        }
        let (file, t) = self.source(src)?;
        let ty = ggml_type(&t.dtype)
            .ok_or_else(|| format!("`{src}`: dtype {} is not loadable as is", t.dtype.name()))?;
        let shape = shape.unwrap_or_else(|| ggml_shape(&t.shape));
        let numel: u64 = shape.iter().product();
        if numel != t.numel() as u64 {
            return Err(format!(
                "`{src}`: shape {:?} does not match {:?}",
                shape, t.shape
            ));
        }
        self.push(name, ty, shape, file, t.data_range.start as u64);
        Ok(())
    }

    /// `name` (ggml `shape`, the dtype of `src`) is the given row ranges of `src` in order; each
    /// range is `(first_row, rows)`, a row being `src`'s innermost dimension (all outer dims
    /// flattened, so a 3-D `[experts, rows, cols]` tensor has `experts * rows` rows).
    fn gather_rows(
        &mut self,
        name: &str,
        src: &str,
        rows: &[(u64, u64)],
        shape: Vec<u64>,
    ) -> Result<(), String> {
        if let Some(r) = self.quantized(name, src, Some(rows), None, Some(shape.clone())) {
            return r;
        }
        let (file, t) = self.source(src)?;
        let ty = ggml_type(&t.dtype)
            .ok_or_else(|| format!("`{src}`: dtype {} is not loadable as is", t.dtype.name()))?;
        let row_bytes = (t.nbytes() / t.numel() * t.shape.last().copied().unwrap_or(1)) as u64;
        let parts: Vec<FileRange> = merge(rows.iter().map(|&(r, n)| FileRange {
            file,
            offset: t.data_range.start as u64 + r * row_bytes,
            len: n * row_bytes,
        }));
        self.gather(name, ty, shape, parts)
    }

    /// `name` (ggml `shape`) is the given source tensors, whole, back to back (all one dtype).
    fn stack(&mut self, name: &str, srcs: &[String], shape: Vec<u64>) -> Result<(), String> {
        let mut parts = Vec::with_capacity(srcs.len());
        let mut ty = None;
        for src in srcs {
            let (file, t) = self.source(src)?;
            let tt = ggml_type(&t.dtype).ok_or_else(|| {
                format!("`{src}`: dtype {} is not loadable as is", t.dtype.name())
            })?;
            if ty.is_some_and(|x| x != tt) {
                return Err(format!("`{name}`: stacked tensors differ in dtype"));
            }
            ty = Some(tt);
            parts.push(FileRange {
                file,
                offset: t.data_range.start as u64,
                len: t.nbytes() as u64,
            });
        }
        let ty = ty.ok_or_else(|| format!("`{name}`: nothing to stack"))?;
        self.gather(name, ty, shape, merge(parts.into_iter()))
    }

    /// `name` is the given ranges back to back.
    fn gather(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: Vec<u64>,
        parts: Vec<FileRange>,
    ) -> Result<(), String> {
        self.push(name, ty, shape, 0, 0);
        let want = self.tensors.last().unwrap().size;
        let got: u64 = parts.iter().map(|p| p.len).sum();
        if got != want {
            return Err(format!(
                "`{name}`: gathered {got} bytes, the tensor has {want}"
            ));
        }
        self.recipes.insert(name.to_string(), Recipe::Gather(parts));
        Ok(())
    }

    /// `name` (F32, ggml `shape`) is `f` applied to `src` read as f32, then made F32 bytes.
    fn computed_f32(
        &mut self,
        name: &str,
        src: &str,
        shape: Vec<u64>,
        f: impl Fn(Vec<f32>) -> Vec<f32> + Send + Sync + 'static,
    ) -> Result<(), String> {
        let (_, t) = self.source(src)?;
        let numel: u64 = shape.iter().product();
        if numel != t.numel() as u64 {
            return Err(format!(
                "`{src}`: shape {:?} does not match {:?}",
                shape, t.shape
            ));
        }
        let ckpt = self.ckpt.clone();
        let src = src.to_string();
        let compute: Compute = Arc::new(move || {
            let (_, t) = ckpt.locate(&src).ok_or("tensor disappeared")?;
            let bytes = ckpt.data(&src).map_err(|e| e.to_string())?;
            let x =
                to_f32(&t.dtype, bytes).ok_or_else(|| format!("`{src}`: not a float tensor"))?;
            Ok(f(x).iter().flat_map(|v| v.to_le_bytes()).collect())
        });
        self.push(name, types::F32, shape, 0, 0);
        self.recipes
            .insert(name.to_string(), Recipe::Computed(compute));
        Ok(())
    }

    /// `name` (F32, 1-D) is `values`, made up by the converter rather than read from a shard.
    fn constant_f32(&mut self, name: &str, values: Vec<f32>) -> Result<(), String> {
        let shape = vec![values.len() as u64];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compute: Compute = Arc::new(move || Ok(bytes.clone()));
        self.push(name, types::F32, shape, 0, 0);
        self.recipes
            .insert(name.to_string(), Recipe::Computed(compute));
        Ok(())
    }

    /// `name` (`src`'s dtype, ggml `shape`) is `f` applied to `src`'s raw bytes.
    fn computed_raw(
        &mut self,
        name: &str,
        src: &str,
        shape: Vec<u64>,
        f: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
    ) -> Result<(), String> {
        let f = Arc::new(f);
        let transform: nvfp4::Transform = f.clone();
        if let Some(r) = self.quantized(name, src, None, Some(transform), Some(shape.clone())) {
            return r;
        }
        let (_, t) = self.source(src)?;
        let ty = ggml_type(&t.dtype)
            .ok_or_else(|| format!("`{src}`: dtype {} is not loadable as is", t.dtype.name()))?;
        let ckpt = self.ckpt.clone();
        let src = src.to_string();
        let compute: Compute = Arc::new(move || {
            let bytes = ckpt.data(&src).map_err(|e| e.to_string())?;
            Ok((*f)(bytes))
        });
        self.push(name, ty, shape, 0, 0);
        self.recipes
            .insert(name.to_string(), Recipe::Computed(compute));
        Ok(())
    }

    fn finish(mut self, path: PathBuf) -> Result<GgufFile, GgufError> {
        for (k, v) in self
            .modelopt
            .as_ref()
            .map(nvfp4::ModelOpt::metadata)
            .unwrap_or_default()
        {
            self.meta.set(k, &v)?;
        }
        let files = self
            .ckpt
            .shards()
            .iter()
            .map(|s| s.path().to_path_buf())
            .collect();
        let mut out = GgufFile::synthetic(self.meta, path, files, self.tensors, self.recipes)?;
        out.set_nvfp4_stacks(self.stacks);
        out.set_checkpoint(Some(self.ckpt));
        Ok(out)
    }
}

/// Row-major safetensors dims (outermost first) as ggml dims (innermost first).
fn ggml_shape(shape: &[usize]) -> Vec<u64> {
    shape.iter().rev().map(|&d| d as u64).collect()
}

/// Adjacent ranges joined, so a reorder that keeps runs together reads them in one go.
fn merge(parts: impl Iterator<Item = FileRange>) -> Vec<FileRange> {
    let mut out: Vec<FileRange> = Vec::new();
    for p in parts {
        match out.last_mut() {
            Some(l) if l.file == p.file && l.offset + l.len == p.offset => l.len += p.len,
            _ => out.push(p),
        }
    }
    out
}

/// Float tensor bytes widened to f32, exactly (every BF16/F16 value is an f32).
fn to_f32(dtype: &Dtype, bytes: &[u8]) -> Option<Vec<f32>> {
    Some(match dtype {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        _ => return None,
    })
}

/// IEEE half to f32 (exact).
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize.
            let shift = m.leading_zeros() - 21;
            let m = (m << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (0x1f, m) => sign | (0xff << 23) | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// A JSON number as u64 / f64, with a readable error.
fn num_u(cfg: &Json, key: &str) -> Result<u64, String> {
    cfg.get(key)
        .and_then(Json::as_u64)
        .ok_or_else(|| format!("config.json: missing or invalid `{key}`"))
}

fn num_f(cfg: &Json, key: &str) -> Result<f64, String> {
    cfg.get(key)
        .and_then(Json::as_f64)
        .ok_or_else(|| format!("config.json: missing or invalid `{key}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_is_exact() {
        for (h, f) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x0400, 6.103_515_6e-5),
        ] {
            assert_eq!(half_to_f32(h), f, "{h:#06x}");
        }
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn adjacent_ranges_merge() {
        let r = |offset, len| FileRange {
            file: 0,
            offset,
            len,
        };
        let got = merge([r(0, 4), r(4, 4), r(16, 2), r(18, 2), r(8, 1)].into_iter());
        assert_eq!(got, vec![r(0, 8), r(16, 4), r(8, 1)]);
    }
}
