use super::*;
use crate::ggml::dequantize;
use crate::model::fp8::e8m0_to_f32;

/// E2M1 code -> value (OCP: sign, 2 exponent bits, 1 mantissa bit).
fn e2m1(c: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

#[test]
fn mxfp4_repack_keeps_every_value() {
    let (rows, k) = (3usize, 64usize);
    // Every code in every position, and scales around 2^0.
    let w: Vec<u8> = (0..rows * k / 2).map(|i| (i * 37 + 11) as u8).collect();
    let s: Vec<u8> = (0..rows * k / 32).map(|i| 124 + i as u8).collect();
    let out = repack_mxfp4(&w, &s, rows, k);
    assert_eq!(out.len(), crate::ggml::row_size(types::MXFP4, k) * rows);
    let got = dequantize(types::MXFP4, &out, rows * k);
    for r in 0..rows {
        for e in 0..k {
            let byte = w[r * k / 2 + e / 2];
            let code = if e % 2 == 0 { byte & 15 } else { byte >> 4 };
            let want = e2m1(code) * e8m0_to_f32(s[r * k / 32 + e / 32]);
            assert_eq!(got[r * k + e], want, "row {r} element {e}");
        }
    }
}
