use super::*;

#[test]
fn takes_the_layer_projections_only() {
    for name in [
        "blk.0.attn_q.weight",
        "blk.3.attn_output.weight",
        "blk.12.attn_qkv.weight",
        "blk.12.attn_gate.weight",
        "blk.12.ssm_out.weight",
        "blk.47.ffn_down_shexp.weight",
    ] {
        assert!(takes(name), "{name}");
    }
    for name in [
        "blk.0.attn_q.scale",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.hc_attn_down.weight",
        "blk.0.ple_key.weight",
        "blk.0.indexer.q_proj.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.x.attn_q.weight",
        "blk..attn_q.weight",
        "output.weight",
        "token_embd.weight",
    ] {
        assert!(!takes(name), "{name}");
    }
}

/// The fast conversion has the reference's bits: every E4M3 value, every midpoint between
/// neighbours (ties to even) and its two F32 neighbours, and a sweep of F32 bit patterns.
#[test]
fn e4m3_matches_the_reference() {
    use crate::model::fp8::{e4m3_to_f32, f32_to_e4m3};
    let check = |x: f32| assert_eq!(e4m3(x), f32_to_e4m3(x), "{x:e} ({:#010x})", x.to_bits());
    let values: Vec<f32> = (0u8..0x7f).map(e4m3_to_f32).collect();
    for w in values.windows(2) {
        let mid = (w[0] + w[1]) / 2.0;
        for x in [
            w[0],
            mid,
            f32::from_bits(mid.to_bits() - 1),
            f32::from_bits(mid.to_bits() + 1),
        ] {
            check(x);
            check(-x);
        }
    }
    for x in [
        0.0,
        -0.0,
        448.0,
        449.0,
        463.9,
        464.0,
        1e9,
        f32::INFINITY,
        1e-30,
    ] {
        check(x);
        check(-x);
    }
    assert_eq!(e4m3(f32::NAN), 0x7f);
    let mut bits = 0u32;
    while bits < 0x7f80_0000 {
        let x = f32::from_bits(bits);
        check(x);
        check(-x);
        bits += 0x1357;
    }
}

#[test]
fn rows_scale_by_their_largest_magnitude() {
    let row = [0.5f32, -3.5, 1.0e-3, 0.0, 2.0, -7.0e-6, 3.25, -0.125];
    let s = row_scale(&row).unwrap();
    assert_eq!(s, 3.5 / 448.0);
    let mut codes = [0u8; 8];
    quantize_row(&row, s, &mut codes);
    // The largest magnitude is the largest code.
    assert_eq!(codes[1], 0x80 | 0x7e);
    for (&c, &v) in codes.iter().zip(&row) {
        assert_eq!(c, crate::model::fp8::f32_to_e4m3(v / s));
        // Dequantized within half a step of 3 mantissa bits (or of the subnormal step).
        let back = crate::model::fp8::e4m3_to_f32(c) * s;
        let tol = (v.abs() / 16.0).max(s * 2f32.powi(-10));
        assert!((back - v).abs() <= tol, "{v} -> {back}");
    }
    assert_eq!(row_scale(&[0.0; 4]).unwrap(), 1.0);
    assert!(row_scale(&[1.0, f32::NAN]).is_err());
    assert!(row_scale(&[f32::INFINITY]).is_err());
}
