use super::*;

#[test]
fn e4m3_round_trips_every_finite_code() {
    for b in 0..=0x7eu8 {
        assert_eq!(e4m3(e4m3_value(b)), b, "code {b:#04x}");
    }
    assert_eq!(e4m3_value(0x7e), 448.0);
    assert_eq!(e4m3_value(0x01), 2f32.powi(-9));
    assert_eq!(e4m3_value(0x08), 2f32.powi(-6));
}

#[test]
fn e4m3_rounds_to_nearest_even_and_saturates() {
    // 1 + 1/16 lies between 1 (0x38) and 1.125 (0x39): the even mantissa wins.
    assert_eq!(e4m3(1.0625), 0x38);
    assert_eq!(e4m3(1.1875), 0x3a);
    assert_eq!(e4m3(1.07), 0x39);
    // Subnormals in steps of 2^-9: half a step rounds to 0, one and a half to 2.
    assert_eq!(e4m3(2f32.powi(-10)), 0);
    assert_eq!(e4m3(3.0 * 2f32.powi(-10)), 2);
    // The largest subnormal rounds up into the normals.
    assert_eq!(e4m3(7.75 * 2f32.powi(-9)), 0x08);
    // Just below a power of two rounds up to it.
    assert_eq!(e4m3(255.99998), e4m3(256.0));
    assert_eq!(e4m3(447.0), 0x7e);
    assert_eq!(e4m3(1.0e9), 0x7e);
    assert_eq!(e4m3(0.0), 0);
}

#[test]
fn e2m1_rounds_to_nearest_even_and_saturates() {
    let cases = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.26, 0.5),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.3, 1.5),
        (1.75, 2.0),
        (2.5, 2.0),
        (2.6, 3.0),
        (3.5, 4.0),
        (5.0, 4.0),
        (5.1, 6.0),
        (100.0, 6.0),
    ];
    for (v, want) in cases {
        assert_eq!(e2m1_value(e2m1(v)), want, "{v}");
        assert_eq!(e2m1_value(e2m1(-v)), -want, "-{v}");
    }
    assert_eq!(e2m1(-0.1), 8, "negative values keep their sign at zero");
}

#[test]
fn quantize_packs_like_ggml_nvfp4() {
    // Element j of a sub-block in the low, j + 8 in the high nibble of byte j.
    // amax 2.625 = 2688 / 1024: decode scale 2^-10, the sub-block's scale 448, so 2.625 is
    // code 6 and -1.3125 code -3, all exact.
    let x: Vec<f32> = (0..16)
        .map(|i| if i < 8 { 2.625 } else { -1.3125 })
        .collect();
    let q = quantize(&x);
    assert_eq!(q.scales, vec![0x7e]);
    assert_eq!(q.dec, 2f32.powi(-10));
    // 6 is code 7, -3 is code 13 (sign bit and 5).
    assert_eq!(q.codes, vec![0xd7; 8]);
    assert_eq!(dequantize(&q), x);
}

#[test]
fn quantize_zero_rows_and_sub_blocks() {
    let q = quantize(&[0.0; 64]);
    assert!(q.codes.iter().all(|&c| c == 0) && q.scales.iter().all(|&s| s == 0));
    assert_eq!(q.dec, 0.0);
    let mut x = vec![0.0f32; 32];
    x[20] = 2.625;
    let q = quantize(&x);
    assert_eq!(q.scales[0], 0);
    assert_eq!(dequantize(&q), x);
}

#[test]
fn quantize_error_is_small_and_the_scale_search_helps() {
    // Each value is within two units of its sub-block's scale (half an E2M1 step, or a
    // saturated largest element when a smaller scale wins), the row's RMS error is a few
    // percent, and no sub-block is worse than with its scale rounded to nearest.
    let mut s = 0x1234_5678u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..1024)
        .map(|i| next() * if i % 64 < 16 { 50.0 } else { 0.3 })
        .collect();
    let q = quantize(&x);
    let y = dequantize(&q);
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (enc, enc16) = (2688.0 / amax, 448.0 / amax);
    let (mut err2, mut sum2) = (0f64, 0f64);
    for (sb, (xs, ys)) in x.chunks(16).zip(y.chunks(16)).enumerate() {
        let unit = e4m3_value(q.scales[sb]) * q.dec;
        let mut chosen = 0f64;
        for (a, b) in xs.iter().zip(ys) {
            assert!((a - b).abs() <= 2.0 * unit, "{a} vs {b}");
            chosen += ((a - b) as f64).powi(2);
            err2 += ((a - b) as f64).powi(2);
            sum2 += (*a as f64).powi(2);
        }
        // The same sub-block with the scale rounded to nearest.
        let bmax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sf = e4m3_value(e4m3(bmax * enc16));
        let nearest: f64 = xs
            .iter()
            .map(|&a| {
                let qs = if sf > 0.0 { enc / sf } else { 0.0 };
                let b = e2m1_value(e2m1(a * qs)) * sf * q.dec;
                ((a - b) as f64).powi(2)
            })
            .sum();
        assert!(
            chosen <= nearest * 1.0001 + 1e-12,
            "sub-block {sb}: {chosen} > {nearest}"
        );
    }
    assert!(
        (err2 / sum2).sqrt() < 0.1,
        "RMS error {}",
        (err2 / sum2).sqrt()
    );
}

#[test]
fn scratch_layout_is_aligned() {
    assert_eq!(q_strides(64), (128, 16));
    assert_eq!(q_strides(2688), (1408, 176));
    let (offs, total) = layout(1856 * 4 * 6, 300, 2688, 129 + 1800);
    for o in offs {
        assert_eq!(o % 256, 0);
    }
    assert!(offs[0] >= 1856 * 4 * 6);
    assert_eq!(offs[1] - offs[0], (300 * 1408usize).div_ceil(256) * 256);
    assert_eq!(total, offs[3] + (129 + 1800) * 4);
}

#[test]
fn token_tiles() {
    assert_eq!(tokens_tile(1), 2);
    assert_eq!(tokens_tile(32), 2);
    assert_eq!(tokens_tile(33), 4);
    assert_eq!(tokens_tile(64), 4);
    assert_eq!(tokens_tile(65), 8);
    assert_eq!(tokens_tile(4096), 8);
}

#[test]
fn decode_kernel_tiles_and_shapes() {
    assert_eq!(mv_tokens_tile(1), 1);
    assert_eq!(mv_tokens_tile(8), 1);
    assert_eq!(mv_tokens_tile(9), 2);
    assert_eq!(mv_tokens_tile(16), 2);
    assert_eq!(mv_tokens_tile(17), 4);
    assert_eq!(mv_tokens_tile(64), 4);
    // The SwiGLU's row tiles pair the gate's rows with the up's: 2 or 4; otherwise 1 or 2. The
    // kernel splits k over 2, 4 or 8 warps, the same for a given k (its sums depend on k alone).
    for k in [256, 640, 1024, 1088, 2560, 4096] {
        let Grouped::Mv { rt, kw } = mv_shape(k, true) else {
            panic!()
        };
        assert!([2, 4].contains(&rt) && [2, 4, 8].contains(&kw));
        let Grouped::Mv { rt: rt1, kw: kw1 } = mv_shape(k, false) else {
            panic!()
        };
        assert!([1, 2].contains(&rt1) && kw1 == kw, "kw depends on k only");
    }
    assert_eq!(mv_shape(640, false), Grouped::Mv { rt: 1, kw: 2 });
    assert_eq!(mv_shape(2560, true), Grouped::Mv { rt: 2, kw: 8 });
}
