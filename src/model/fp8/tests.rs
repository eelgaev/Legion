use super::*;

#[test]
fn e4m3_values() {
    assert_eq!(e4m3_to_f32(0x00), 0.0);
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    assert_eq!(e4m3_to_f32(0xb8), -1.0);
    assert_eq!(e4m3_to_f32(0x7e), 448.0);
    assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
    assert_eq!(e4m3_to_f32(0x3c), 1.5);
    assert!(e4m3_to_f32(0x7f).is_nan() && e4m3_to_f32(0xff).is_nan());
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
    assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
    assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
    assert!(e8m0_to_f32(255).is_nan());
    // Every finite code round-trips.
    for b in (0..=255u8).filter(|b| b & 0x7f != 0x7f) {
        let back = f32_to_e4m3(e4m3_to_f32(b));
        assert!(
            back == b || (b == 0x80 && back == 0x80),
            "{b:#04x} -> {back:#04x}"
        );
    }
    assert_eq!(f32_to_e4m3(1000.0), 0x7e);
    assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
    assert_eq!(f32_to_e4m3(1.06), 0x38);
    // Ties to even: 1.0625 is between 1.0 (0x38) and 1.125 (0x39).
    assert_eq!(f32_to_e4m3(1.0625), 0x38);
    assert_eq!(f32_to_e4m3(1.1875), 0x3a);
    // Subnormals and the carry into the next binade.
    assert_eq!(f32_to_e4m3(3.0 * 2f32.powi(-10)), 0x02);
    assert_eq!(f32_to_e4m3(0.0156), 0x08);
    assert_eq!(f32_to_e4m3(15.9), 0x58);
}

/// Per-token quantization: the largest magnitude maps to 448 (0x7e / 0xfe), the scale is
/// `amax / 448`, every code is within E4M3's rounding of `x / s`, and an all-zero token gets
/// scale 0 and zero codes.
#[test]
fn quantize_token_scales_by_amax() {
    let x = [0.5f32, -3.5, 1.0, 0.0, 2.25, -0.001];
    let (q, s) = quantize_token(&x);
    assert_eq!(s, 3.5 / 448.0);
    assert_eq!(q[1], 0xfe, "-amax is -448");
    for (&v, &b) in x.iter().zip(&q) {
        let back = e4m3_to_f32(b) * s;
        assert!(
            (back - v).abs() <= v.abs() / 16.0 + s / 512.0,
            "{v} -> {back}"
        );
    }
    assert_eq!(q[3], 0);
    let (q, s) = quantize_token(&[0.0; 8]);
    assert_eq!(s, 0.0);
    assert!(q.iter().all(|&b| b == 0));
    // A token of one tiny value: still exactly +-448 at the top.
    let (q, s) = quantize_token(&[1e-30, -2e-30]);
    assert!(s > 0.0);
    assert_eq!(q[1], 0xfe);
}

/// The packed layout: codes first, then one F32 scale per token, 4-byte aligned for `k` a
/// multiple of 4.
#[test]
fn dynamic_layout_packs_codes_then_scales() {
    assert_eq!(dynamic_layout(64, 3), (192, 204));
    assert_eq!(
        dynamic_layout(5120, 4096),
        (5120 * 4096, 5120 * 4096 + 4 * 4096)
    );
    assert_eq!(dynamic_layout(4, 0), (0, 0));
}

/// The sliced-K GEMM takes 5-64 tokens when asked, leaves 5-9 to the GEMV when that is on too,
/// and needs rows a multiple of 256 long.
#[test]
fn sliced_k_split_by_width() {
    let off = SmallBatch::default();
    let sliced = SmallBatch {
        sliced_k: true,
        ..off
    };
    let both = SmallBatch {
        gemv: true,
        ..sliced
    };
    let gemv = SmallBatch { gemv: true, ..off };
    for n in 1..=80 {
        assert!(!sliced_k_takes(off, n, 5120), "n {n}, off");
        assert!(!sliced_k_takes(gemv, n, 5120), "n {n}, GEMV only");
        assert_eq!(
            sliced_k_takes(sliced, n, 5120),
            (5..=64).contains(&n),
            "n {n}, sliced-K"
        );
        assert_eq!(
            sliced_k_takes(both, n, 5120),
            (10..=64).contains(&n),
            "n {n}, both"
        );
    }
    for (k, takes) in [
        (17408, true),
        (6144, true),
        (2688, false),
        (4160, false),
        (256, true),
    ] {
        assert_eq!(sliced_k_takes(sliced, 16, k), takes, "k {k}");
    }
}
