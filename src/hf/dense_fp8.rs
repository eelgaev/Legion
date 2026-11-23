//! `--dense-fp8`: FP8 E4M3 copies of a Qwen3.8-Flash-Next (`qwen4_exp`) checkpoint's BF16 dense
//! projections, made as they load, which decode-sized steps (decode and verify, up to 64
//! tokens) multiply instead of the BF16 weights: half the bytes for the steps that stream them.
//! Prompt steps keep the BF16 weights and every kernel, join and fusion they take today.
//! Opt-in: decode outputs move.
//!
//! Why not prompt steps too (#1183): with the FP8 weights in their place, W8A8 prompt steps
//! measured 2-3% slower at 8K and 32K on a DGX Spark (GB10). The FP8 weights leave the BF16
//! prompt paths: the weight runs read as one GEMM (query + gate / key / value, `z` / alpha /
//! beta, shared-expert gate / up: 38 of 134 stayed joined), the hyper-connection mixers writing
//! the BF16 copy their readers take instead of the F32 stream (28 of 75 casts, none of 48
//! skipped F32 streams left), and they add one quantization pass of the F32 stream per input:
//! together more than the FP8 tensor cores saved on these shapes.
//!
//! The scheme is llm-compressor's `FP8_DYNAMIC` weights (channel-wise): each output row `r`
//! gets the F32 scale `s_r = amax_r / 448` (`amax_r = max |w_rj|`; 1 for an all-zero row) and
//! the codes `e4m3(w_rj / s_r)`, IEEE division, rounded to nearest even and saturated at ±448
//! ([`e4m3`]). A copy is what a channel-wise FP8 checkpoint loads as (see [`super::nvfp4`]): an
//! `I8` tensor of the weight's shape and an F32 scale per row, named by [`copy_names`] and
//! stored after every other tensor (the weights keep their order, which the joined runs read).
//! [`KEY`] lists the weights that have one; [`crate::model::core`] reads the copy in place of
//! the weight in decode-sized steps, through the FP8 matmul's W8A16 route.
//!
//! Which weights ([`takes`]): attention Q (with its gate), K, V and output, the DeltaNet `qkv`,
//! `z` and output projections, and the shared expert's gate, up and down. The rest has no copy:
//! the hyper-connection mixers and the n-gram projections (their own kernels), the DeltaNet
//! alpha / beta and the indexer (small), the router, the embeddings and the output head, the
//! routed experts (NVFP4) and the MTP head (its own file). Nor does a weight whose row length
//! is not a multiple of 64.
//!
//! After Infernix's `qwen3_8_flash_next_nvfp4_dense8` recipe (Apache-2.0), which stores the same
//! classes of dense weights in 8 bits (int8 in groups of 32 there); no Infernix code is used.

use std::sync::{Arc, OnceLock};

use super::Builder;
use super::nvfp4::{Transform, gather, par_items};
use crate::gguf::{Compute, Recipe};
use crate::safetensors::{Checkpoint, Dtype, TensorInfo};

/// GGUF metadata: the names of the weights with an FP8 copy ([`copy_names`]).
pub const KEY: &str = "legion.dense_fp8";

/// The per-layer weights `--dense-fp8` copies (`blk.<il>.<name>.weight`).
pub const PROJECTIONS: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_qkv",
    "attn_gate",
    "ssm_out",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
];

/// Whether the GGUF tensor `name` is one of a layer's [`PROJECTIONS`].
pub fn takes(name: &str) -> bool {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .is_some_and(|(il, rest)| {
            !il.is_empty()
                && il.bytes().all(|c| c.is_ascii_digit())
                && rest
                    .strip_suffix(".weight")
                    .is_some_and(|p| PROJECTIONS.contains(&p))
        })
}

/// The names of the FP8 copy of the weight `name` (`<base>.weight`): its codes
/// `<base>.fp8.weight` and its row scales `<base>.fp8.scale`.
pub fn copy_names(name: &str) -> (String, String) {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    (format!("{base}.fp8.weight"), format!("{base}.fp8.scale"))
}

/// The nearest E4M3 (`fn`) code of `x`: round to nearest even, saturating at ±448, NaN to
/// `0x7f`. The bits of [`crate::model::fp8::f32_to_e4m3`] (CUDA's `__NV_SATFINITE`
/// conversion), from the F32 bits instead of logarithms: billions of weights go through it.
pub fn e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    let code = if a < 1.0 / 64.0 {
        // Subnormals (and zero): steps of 2^-9, exact in F32; a rounding up to 2^-6 gives code
        // 8, the smallest normal.
        (a * 512.0).round_ties_even() as u32
    } else {
        // Normals: exponent -6..=8, the 23-bit mantissa rounded to 3 bits. A carry out of the
        // mantissa moves into the exponent field; below 448 the result stays at most 0x7e.
        let bits = a.to_bits();
        let exp = (bits >> 23) as i32 - 127;
        let man = bits & 0x7f_ffff;
        let (mut q, rem) = (man >> 20, man & 0xf_ffff);
        if rem > 0x8_0000 || (rem == 0x8_0000 && q & 1 == 1) {
            q += 1;
        }
        (((exp + 7) as u32) << 3) + q
    };
    sign | code as u8
}

/// One output row's scale, `amax / 448` (1 for an all-zero row); an error for a row with a
/// value that is not finite.
pub fn row_scale(row: &[f32]) -> Result<f32, String> {
    let mut amax = 0f32;
    for &v in row {
        if !v.is_finite() {
            return Err(format!("a weight value is {v}"));
        }
        amax = amax.max(v.abs());
    }
    Ok(if amax > 0.0 { amax / 448.0 } else { 1.0 })
}

/// `row`'s codes with scale `s` ([`row_scale`]): `e4m3(w / s)` each, into `out`.
pub fn quantize_row(row: &[f32], s: f32, out: &mut [u8]) {
    for (o, &v) in out.iter_mut().zip(row) {
        *o = e4m3(v / s);
    }
}

/// The values of a weight's source rows as F32 (`rows` of `src` selected, then `bytes` applied:
/// the same row selection or column permutation the unquantized tensor would get), handed to
/// `f` as row-major bytes of the source dtype.
fn with_rows<R>(
    ckpt: &Checkpoint,
    src: &str,
    rows: Option<&[(u64, u64)]>,
    bytes: Option<&Transform>,
    row_bytes: usize,
    f: impl FnOnce(&[u8]) -> Result<R, String>,
) -> Result<R, String> {
    let data = ckpt.data(src).map_err(|e| e.to_string())?;
    match (rows, bytes) {
        (None, None) => f(data),
        (Some(r), None) => f(&gather(data, r, row_bytes)),
        (None, Some(t)) => f(&t(data)),
        (Some(r), Some(t)) => f(&t(&gather(data, r, row_bytes))),
    }
}

impl Builder {
    /// With `--dense-fp8`, the FP8 copy of `name` (from `src`, ggml `shape` or the source's,
    /// `rows` and `bytes` as the weight takes them) when [`takes`] picks it and its source is
    /// a 2-D float tensor with rows of a multiple of 64: its recipes, its tensors pushed by
    /// [`Self::finish`]. The weight itself loads as usual.
    pub(super) fn dense_fp8(
        &mut self,
        name: &str,
        src: &str,
        rows: Option<&[(u64, u64)]>,
        bytes: Option<Transform>,
        shape: Option<&[u64]>,
    ) -> Result<(), String> {
        if !self.dense_fp8_on || !takes(name) {
            return Ok(());
        }
        let Some((_, t)) = self.ckpt.locate(src) else {
            return Ok(());
        };
        let t: TensorInfo = t.clone();
        let float = matches!(t.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32);
        if !float || t.shape.len() != 2 || t.shape[1] % 64 != 0 {
            return Ok(());
        }
        let k = t.shape[1];
        let shape = shape.map_or_else(|| super::ggml_shape(&t.shape), <[u64]>::to_vec);
        if shape.len() != 2 || shape[0] != k as u64 {
            return Ok(());
        }
        let m = shape[1] as usize;
        let selected: u64 = rows.map_or(t.shape[0] as u64, |r| r.iter().map(|p| p.1).sum());
        if selected != m as u64 {
            return Err(format!(
                "`{src}`: {selected} rows selected for a [{k}, {m}] tensor"
            ));
        }
        let elem = t.nbytes() / t.numel();
        let row_bytes = k * elem;
        let dtype = t.dtype.clone();
        let rows: Option<Arc<Vec<(u64, u64)>>> = rows.map(|r| Arc::new(r.to_vec()));
        let ckpt = self.ckpt.clone();
        let src = src.to_string();
        // The rows' scales, read once for both tensors.
        let memo: Arc<OnceLock<Result<Vec<f32>, String>>> = Arc::default();
        let scales = {
            let (ckpt, src, rows, bytes, dtype) = (
                ckpt.clone(),
                src.clone(),
                rows.clone(),
                bytes.clone(),
                dtype.clone(),
            );
            Arc::new(move || -> Result<Vec<f32>, String> {
                memo.get_or_init(|| {
                    with_rows(
                        &ckpt,
                        &src,
                        rows.as_deref().map(Vec::as_slice),
                        bytes.as_ref(),
                        row_bytes,
                        |data| {
                            let mut out = vec![0f32; m];
                            par_items(&mut out, 1, &|r, o| {
                                let x = row_values(&dtype, data, r, row_bytes)?;
                                o[0] =
                                    row_scale(&x).map_err(|e| format!("`{src}` row {r}: {e}"))?;
                                Ok(())
                            })?;
                            Ok(out)
                        },
                    )
                })
                .clone()
            })
        };
        let codes: Compute = {
            let scales = scales.clone();
            Arc::new(move || {
                let s = scales()?;
                with_rows(
                    &ckpt,
                    &src,
                    rows.as_deref().map(Vec::as_slice),
                    bytes.as_ref(),
                    row_bytes,
                    |data| {
                        let mut out = vec![0u8; m * k];
                        par_items(&mut out, k, &|r, o| {
                            quantize_row(&row_values(&dtype, data, r, row_bytes)?, s[r], o);
                            Ok(())
                        })?;
                        Ok(out)
                    },
                )
            })
        };
        let (codes_name, scale_name) = copy_names(name);
        self.recipes.insert(codes_name, Recipe::Computed(codes));
        self.recipes.insert(
            scale_name,
            Recipe::Computed(Arc::new(move || {
                Ok(scales()?.iter().flat_map(|v| v.to_le_bytes()).collect())
            })),
        );
        self.dense_fp8_names.push((name.to_string(), shape));
        Ok(())
    }
}

/// Row `r` (`row_bytes` bytes of `dtype`) of `data` as F32, exactly.
fn row_values(dtype: &Dtype, data: &[u8], r: usize, row_bytes: usize) -> Result<Vec<f32>, String> {
    let row = data
        .get(r * row_bytes..(r + 1) * row_bytes)
        .ok_or_else(|| format!("row {r} is past the source's end"))?;
    super::to_f32(dtype, row).ok_or_else(|| format!("dtype {} is not a float", dtype.name()))
}

#[cfg(test)]
mod tests;
