//! A 4-bit copy of an output head for drafting (`--spec-draft-lm-head q4`): DFlash's borrowed
//! head ([`super::dflash`]) and the MTP head's ([`crate::model::core::Model::quantize_draft_head`]).
//! Also the 4-bit copies of a DFlash drafter's matrices (`--spec-draft-weights q4`).
//! `--spec-draft-lm-head nvfp4`: an NVFP4 copy of DFlash's borrowed head ([`to_nvfp4`]).
//! A draft only changes what the target verifies, never what it emits.

use crate::ggml::{ffi, types};

/// The 4-bit type rows of `k` values are stored as: Q4_K, or Q4_0 when `k` is not a multiple of
/// 256; none when it is not a multiple of 32 either.
pub(crate) fn q4_type(k: usize) -> Option<ffi::ggml_type> {
    if k.is_multiple_of(256) {
        Some(types::Q4_K)
    } else if k.is_multiple_of(32) {
        Some(types::Q4_0)
    } else {
        None
    }
}

/// Rows of `k` values of type `ty` (an `I8` type is FP8 E4M3 bytes) times `scale` (one value,
/// or one per row) as Q4_K, or Q4_0 when `k` is not a multiple of 256. The rows are split over
/// threads.
pub(crate) fn to_q4(
    ty: ffi::ggml_type,
    w: &[u8],
    scale: Option<&[f32]>,
    k: usize,
) -> Result<(ffi::ggml_type, Vec<u8>), String> {
    let q4 = q4_type(k)
        .ok_or_else(|| format!("a 4-bit head needs rows of a multiple of 32 values, not {k}"))?;
    if ty != types::I8 && ty != types::F32 && !crate::ggml::can_dequantize(ty) {
        return Err(format!(
            "a {} head cannot be converted to 4 bits",
            crate::ggml::type_name(ty)
        ));
    }
    let row = crate::ggml::row_size(ty, k);
    let rows = w.len() / row;
    if let Some(s) = scale
        && s.len() != 1
        && s.len() != rows
    {
        return Err(format!("{} scales for {rows} rows", s.len()));
    }
    let lut: [f32; 256] = std::array::from_fn(|b| crate::model::fp8::e4m3_to_f32(b as u8));
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let per = rows.div_ceil(threads).max(1);
    // Rows converted at a time by one thread: bounded F32 scratch.
    const BATCH: usize = 256;
    let parts: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let lut = &lut;
        let hs: Vec<_> = (0..rows)
            .step_by(per)
            .map(|r0| {
                scope.spawn(move || {
                    let r1 = (r0 + per).min(rows);
                    let mut out = Vec::with_capacity(crate::ggml::row_size(q4, k) * (r1 - r0));
                    let mut f = Vec::with_capacity(BATCH * k);
                    for b0 in (r0..r1).step_by(BATCH) {
                        f.clear();
                        for r in b0..(b0 + BATCH).min(r1) {
                            let s = scale.map_or(1.0, |s| s[if s.len() == 1 { 0 } else { r }]);
                            push_row(ty, &w[r * row..(r + 1) * row], s, k, lut, &mut f);
                        }
                        out.extend(crate::ggml::quantize(q4, &f, k));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("quantize thread"))
            .collect()
    });
    Ok((q4, parts.concat()))
}

/// Appends one row of `k` values of type `ty` (`I8`: FP8 E4M3 bytes through `lut`) times `s`
/// to `f` as F32.
fn push_row(
    ty: ffi::ggml_type,
    bytes: &[u8],
    s: f32,
    k: usize,
    lut: &[f32; 256],
    f: &mut Vec<f32>,
) {
    if ty == types::I8 {
        f.extend(bytes.iter().map(|&b| lut[b as usize] * s));
    } else if ty == types::F32 {
        let v = bytes.as_chunks::<4>().0;
        f.extend(v.iter().map(|c| f32::from_ne_bytes(*c) * s));
    } else {
        let v = crate::ggml::dequantize(ty, bytes, k);
        f.extend(v.into_iter().map(|v| v * s));
    }
}

/// The largest E2M1 code (6) times the largest UE4M3 scale (448): the per-tensor scale maps the
/// head's largest magnitude there, so the sub-block scales use UE4M3's whole (normal) range.
const NVFP4_MAX: f32 = 6.0 * 448.0;

/// Rows of `k` values of type `ty` (as [`to_q4`] takes them) times `scale` as NVFP4 (`k` a
/// multiple of 64): the rows times one per-tensor factor `g`, quantized with UE4M3 scales per
/// 16 values, and the decode scale `1 / g` the matmul multiplies its output by. Two passes over
/// the rows (their largest magnitude, then the codes), split over threads.
pub(crate) fn to_nvfp4(
    ty: ffi::ggml_type,
    w: &[u8],
    scale: Option<&[f32]>,
    k: usize,
) -> Result<(Vec<u8>, f32), String> {
    if !k.is_multiple_of(64) {
        return Err(format!(
            "an NVFP4 head needs rows of a multiple of 64 values, not {k}"
        ));
    }
    if ty != types::I8 && ty != types::F32 && !crate::ggml::can_dequantize(ty) {
        return Err(format!(
            "a {} head cannot be converted to NVFP4",
            crate::ggml::type_name(ty)
        ));
    }
    let row = crate::ggml::row_size(ty, k);
    let rows = w.len() / row;
    if let Some(s) = scale
        && s.len() != 1
        && s.len() != rows
    {
        return Err(format!("{} scales for {rows} rows", s.len()));
    }
    let lut: [f32; 256] = std::array::from_fn(|b| crate::model::fp8::e4m3_to_f32(b as u8));
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let per = rows.div_ceil(threads).max(1);
    const BATCH: usize = 256;
    // Rows r0..r1 as F32 (scale folded in), BATCH at a time, handed to `each`.
    let pass = |r0: usize, r1: usize, each: &mut dyn FnMut(&[f32])| {
        let mut f = Vec::with_capacity(BATCH * k);
        for b0 in (r0..r1).step_by(BATCH) {
            f.clear();
            for r in b0..(b0 + BATCH).min(r1) {
                let s = scale.map_or(1.0, |s| s[if s.len() == 1 { 0 } else { r }]);
                push_row(ty, &w[r * row..(r + 1) * row], s, k, &lut, &mut f);
            }
            each(&f);
        }
    };
    let amax = std::thread::scope(|scope| {
        let hs: Vec<_> = (0..rows)
            .step_by(per)
            .map(|r0| {
                let pass = &pass;
                scope.spawn(move || {
                    let mut m = 0f32;
                    pass(r0, (r0 + per).min(rows), &mut |f| {
                        m = f.iter().fold(m, |m, v| m.max(v.abs()));
                    });
                    m
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("amax thread"))
            .fold(0f32, f32::max)
    });
    if !(amax.is_finite() && amax > 0.0) {
        return Err(format!("the head's largest magnitude is {amax}"));
    }
    let g = NVFP4_MAX / amax;
    let parts: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let hs: Vec<_> = (0..rows)
            .step_by(per)
            .map(|r0| {
                let pass = &pass;
                scope.spawn(move || {
                    let r1 = (r0 + per).min(rows);
                    let mut out =
                        Vec::with_capacity(crate::ggml::row_size(types::NVFP4, k) * (r1 - r0));
                    let mut scaled = Vec::with_capacity(BATCH * k);
                    pass(r0, r1, &mut |f| {
                        scaled.clear();
                        scaled.extend(f.iter().map(|v| v * g));
                        out.extend(crate::ggml::quantize(types::NVFP4, &scaled, k));
                    });
                    out
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("quantize thread"))
            .collect()
    });
    Ok((parts.concat(), 1.0 / g))
}
