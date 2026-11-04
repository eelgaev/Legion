//! FP16 load mode (`--weights-fp16 load`, opt-in): weights GPUs without BF16 tensor cores (V100)
//! would run on slow paths are converted to F16 as they load, so ggml multiplies them with FP16
//! cuBLAS on tensor cores. BF16 matmuls on a V100 run as F32 GEMMs (BF16 converted to F32), F32
//! ones as FP32 SGEMMs: 5-8x slower than FP16 at prefill sizes.
//!
//! - BF16 tensors become F16 (BF16's 8 mantissa bits fit F16's 10, so only values outside F16's
//!   range would change; such a tensor keeps its type, logged).
//! - With `gates`, the F32 router and gate weights too (`ffn_gate_inp`, `ffn_gate_inp_shexp`,
//!   `ssm_alpha`, `ssm_beta`): rounded to F16, so routing and gating shift slightly (hence its
//!   own switch). The hyper-connection inject weights stay F32: the fused mixer norm reads them
//!   ([`crate::model`]'s `hc_norm`) and costs no matmul.
//!
//! The converted bytes replace the tensor's data from the file (no second copy of the original).
//! Adapted from Strata-AC922's FP16 load mode (`STRATA_FP16=load`, `STRATA_FP16_GATES=1`;
//! `src/core/load_converted.cpp`, MIT, github.com/eelgaev/Strata-AC922).

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;

use crate::ggml::{ffi, types};
use crate::gguf::{GgufFile, TensorInfo};

/// The FP16 load mode's choice of weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum WeightsFp16 {
    /// Weights as stored.
    #[default]
    Off,
    /// BF16 weights as F16 (plus the F32 gates with `--weights-fp16-gates`).
    Load,
}

/// What the loader converts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fp16Load {
    pub mode: WeightsFp16,
    /// Also the F32 router and gate weights.
    pub gates: bool,
    /// BF16 hyper-connection inject weights (`hc_attn_inject`, `hc_ffn_inject`: 2560 x 4 x 4
    /// values a layer) as F32, exactly (`--hc-inject-f32`, opt-in, any mode): the fused
    /// prompt-step mixer norm (`hc_post_norm`) takes F32 inject weights only, so a GGUF that
    /// stores them in BF16 (an IQ2_XS Qwen3.8-Flash-Next) otherwise runs ggml's chain, whose
    /// BF16 matmul rounds the activations to BF16. Output-moving for such files.
    pub hc_inject_f32: bool,
}

/// The hyper-connection inject weights by name suffix.
const HC_INJECT: [&str; 2] = ["hc_attn_inject.weight", "hc_ffn_inject.weight"];

/// F32 gate weights by name suffix (`--weights-fp16-gates`).
const GATES: [&str; 4] = [
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
];

/// Largest finite F16 value.
const F16_MAX: f32 = 65504.0;

impl Fp16Load {
    /// The type the loader converts `info` to, if any: F32 for a BF16 inject weight with
    /// [`Self::hc_inject_f32`], else F16 for what [`Self::wants`].
    pub fn target(&self, info: &TensorInfo) -> Option<ffi::ggml_type> {
        if self.hc_inject_f32
            && info.ggml_type == types::BF16
            && HC_INJECT.iter().any(|s| base(&info.name).ends_with(s))
        {
            return Some(types::F32);
        }
        self.wants(info).then_some(types::F16)
    }

    /// Whether the tensor `info` is one to convert.
    pub fn wants(&self, info: &TensorInfo) -> bool {
        if self.mode == WeightsFp16::Off {
            return false;
        }
        match info.ggml_type {
            types::BF16 => true,
            types::F32 => self.gates && GATES.iter().any(|g| base(&info.name).ends_with(g)),
            _ => false,
        }
    }
}

/// Whether `name` is a hyper-connection inject weight (kept F32 or made F32, never F16: the
/// fused mixer norm reads F32 ones).
pub fn is_hc_inject(name: &str) -> bool {
    HC_INJECT.iter().any(|s| base(name).ends_with(s))
}

/// A tensor's name without a parallel-decode copy suffix (`<name>@m<d>`, `<name>@s<d>`): a
/// GPU's copy converts like the original.
fn base(name: &str) -> &str {
    name.split_once('@').map_or(name, |(b, _)| b)
}

/// Whether every one of `values` is finite and within F16's range.
pub fn fits_f16(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite() && v.abs() <= F16_MAX)
}

/// `values` as F16 bytes, or `None` when one is out of F16's range (or not finite).
pub fn to_f16(values: &[f32]) -> Option<Vec<u8>> {
    if !fits_f16(values) {
        return None;
    }
    let mut h = vec![0u16; values.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(values.as_ptr(), h.as_mut_ptr(), values.len() as i64) };
    Some(h.iter().flat_map(|x| x.to_ne_bytes()).collect())
}

/// Raw bytes of a BF16 or F32 tensor as F32 values.
fn as_f32(ty: ffi::ggml_type, bytes: &[u8]) -> Vec<f32> {
    match ty {
        types::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_ne_bytes(*b) as u32) << 16))
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect(),
    }
}

/// Converts the tensors `load` wants: their infos in `infos` become F16 (type and size), and
/// their F16 bytes are returned by name. Tensors with a recipe (synthetic models) are left as
/// they are, and so is any with a value out of F16's range.
pub fn convert(
    gguf: &GgufFile,
    infos: &mut [TensorInfo],
    load: Fp16Load,
) -> std::io::Result<HashMap<String, Vec<u8>>> {
    let mut out = HashMap::new();
    if load.mode == WeightsFp16::Off && !load.hc_inject_f32 {
        return Ok(out);
    }
    let paths: Vec<_> = gguf.shard_paths().collect();
    let mut files: HashMap<usize, File> = HashMap::new();
    let (mut before, mut after, mut to_f32) = (0u64, 0u64, 0usize);
    for info in infos.iter_mut() {
        let Some(to) = load.target(info) else {
            continue;
        };
        if gguf.recipe(&info.name).is_some() {
            continue;
        }
        let file = match files.entry(info.file_index) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(File::open(paths[info.file_index])?)
            }
        };
        let mut raw = vec![0u8; info.size as usize];
        file.read_exact_at(&mut raw, info.offset)?;
        let values = as_f32(info.ggml_type, &raw);
        let h = if to == types::F32 {
            to_f32 += 1;
            values.iter().flat_map(|v| v.to_ne_bytes()).collect()
        } else if let Some(h) = to_f16(&values) {
            h
        } else {
            tracing::warn!(tensor = %info.name, "weights-fp16: a value is out of F16's range; kept as stored");
            continue;
        };
        before += info.size;
        after += h.len() as u64;
        info.ggml_type = to;
        info.size = h.len() as u64;
        out.insert(info.name.clone(), h);
    }
    tracing::info!(
        tensors = out.len(),
        mib_before = before as f64 / (1u64 << 20) as f64,
        mib_after = after as f64 / (1u64 << 20) as f64,
        gates = load.gates,
        hc_inject_f32 = to_f32,
        "weights-fp16: converted at load (to F16; hyper-connection inject weights to F32)"
    );
    Ok(out)
}

#[cfg(test)]
mod tests;
