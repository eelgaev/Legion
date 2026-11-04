//! A 4-bit copy of an output head for drafting (`--spec-draft-lm-head q4`): DFlash's borrowed
//! head ([`super::dflash`]) and the MTP head's ([`crate::model::core::Model::quantize_draft_head`]).
//! Also the 4-bit copies of a DFlash drafter's matrices (`--spec-draft-weights q4`).
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
                            let bytes = &w[r * row..(r + 1) * row];
                            let s = scale.map_or(1.0, |s| s[if s.len() == 1 { 0 } else { r }]);
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
