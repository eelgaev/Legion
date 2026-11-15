use super::*;

#[test]
fn f16_round_trips_every_finite_pattern() {
    for h in 0..=u16::MAX {
        if h & 0x7c00 == 0x7c00 {
            continue;
        }
        assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#06x}");
    }
}

#[test]
fn f16_known_values() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x7bff), 65504.0);
    assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(f32_to_f16(65520.0), 0x7c00); // the tie at the top rounds to infinity
    assert_eq!(f32_to_f16(65519.0), 0x7bff);
    assert_eq!(f32_to_f16(1.0e-10), 0);
    assert_eq!(f32_to_f16(-1.0e-10), 0x8000);
}

#[test]
fn f16_rounds_ties_to_even() {
    // 1 + 2^-11 is halfway between 1 and the next f16: to the even mantissa (1).
    assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
    // 1 + 3 * 2^-11: halfway between mantissas 1 and 2, to 2.
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
    // Subnormal ties: 1.5 units of 2^-24 to 2, 2.5 to 2, 0.5 to 0.
    assert_eq!(f32_to_f16(1.5 * 2.0f32.powi(-24)), 2);
    assert_eq!(f32_to_f16(2.5 * 2.0f32.powi(-24)), 2);
    assert_eq!(f32_to_f16(0.5 * 2.0f32.powi(-24)), 0);
    assert_eq!(f32_to_f16(0.75 * 2.0f32.powi(-24)), 1);
    // The largest subnormal's upper neighbourhood rounds into the normals.
    assert_eq!(f32_to_f16(1023.75 * 2.0f32.powi(-24)), 0x0400);
}

#[test]
fn lanes_follow_the_kernels_rule() {
    assert_eq!(units_per_row(QuantFormat::Q4K, 2560), Some(160));
    assert_eq!(units_per_row(QuantFormat::Q5_1, 640), Some(40));
    assert_eq!(units_per_row(QuantFormat::Q8_0, 640), Some(80));
    assert_eq!(units_per_row(QuantFormat::Q6K, 2560), None);
    assert_eq!(lanes_per_row(160), 32);
    assert_eq!(lanes_per_row(80), 16);
    assert_eq!(lanes_per_row(40), 8);
    assert_eq!(lanes_per_row(16), 4);
    assert_eq!(lanes_per_row(3), 1);
}

#[test]
fn rejects_wrong_shapes() {
    let mut out = [0.0f32; 2];
    assert_eq!(
        matvec(QuantFormat::Q6K, &[], &[], 256, &mut out),
        Err(Error::Format(QuantFormat::Q6K))
    );
    assert_eq!(
        matvec(QuantFormat::Q8_0, &[0; 68], &[0; 36], 33, &mut out),
        Err(Error::Columns { k: 33 })
    );
    assert_eq!(
        matvec(QuantFormat::Q8_0, &[0; 67], &[0; 36], 32, &mut out),
        Err(Error::Weights {
            actual: 67,
            row: 34
        })
    );
    assert_eq!(
        matvec(QuantFormat::Q8_0, &[0; 68], &[0; 35], 32, &mut out),
        Err(Error::Activations {
            actual: 35,
            expected: 36
        })
    );
}

/// One Q8_0 row of one block against hand arithmetic: 4 units on 1 lane, fused in order.
#[test]
fn q8_0_row_is_the_fused_sum() {
    let mut w = vec![0u8; 34];
    w[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
    let mut xq = vec![0u8; 36];
    xq[0..2].copy_from_slice(&0x3400u16.to_le_bytes()); // 0.25
    for v in 0..32 {
        w[2 + v] = (v as i8 - 16) as u8;
        xq[4 + v] = (3 - v as i8) as u8;
    }
    let mut out = [0.0f32];
    matvec_scalar(QuantFormat::Q8_0, &w, &xq, 32, &mut out).unwrap();
    let mut want = 0.0f32;
    for q in 0..4 {
        let sumi: i32 = (8 * q..8 * q + 8).map(|v| (v - 16) * (3 - v)).sum();
        want = 0.125f32.mul_add(sumi as f32, want);
    }
    assert_eq!(out[0].to_bits(), want.to_bits());
}

/// A Q5_1 block whose quants are all 31 with a zero minimum: the row is d5 * d8 * 31 * sum(x).
#[test]
fn q5_1_row_decodes_the_fifth_bit() {
    let mut w = vec![0xffu8; 24];
    w[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
    w[2..4].copy_from_slice(&0u16.to_le_bytes()); // m = 0
    let mut xq = vec![1u8; 36];
    xq[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    xq[2..4].copy_from_slice(&0x4c00u16.to_le_bytes());
    let mut out = [0.0f32];
    matvec_scalar(QuantFormat::Q5_1, &w, &xq, 32, &mut out).unwrap();
    assert_eq!(out[0], 31.0 * 32.0);
}

/// A Q4_K block with one scale and no minimum is its nibbles times d * scale * d8.
#[test]
fn q4_k_row_matches_dequantized_dot() {
    let mut w = vec![0u8; 144];
    w[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
    w[2..4].copy_from_slice(&0x3c00u16.to_le_bytes()); // dmin = 1
    for j in 0..4 {
        w[4 + j] = 2; // scales of sub-blocks 0..4
        w[8 + j] = 0; // their minimums
        w[12 + j] = 2; // sub-blocks 4..8: scale 2, minimum 0
    }
    let mut want = 0i32;
    let mut xq = vec![0u8; 8 * 36];
    for b in 0..8 {
        xq[b * 36..][..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    for v in 0..256 {
        let q = (v * 7 + 3) % 16;
        let x = (v as i32 * 5) % 23 - 11;
        let (sub, idx) = (v / 32, v % 32);
        w[16 + 32 * (sub / 2) + idx] |= (q as u8) << (4 * (sub % 2));
        xq[sub * 36 + 4 + idx] = x as i8 as u8;
        want += 2 * q as i32 * x;
    }
    let mut out = [0.0f32];
    matvec_scalar(QuantFormat::Q4K, &w, &xq, 256, &mut out).unwrap();
    assert_eq!(out[0], want as f32);
}

#[test]
fn vector_kernels_equal_the_reference() {
    // On a CPU without the vector form this compares the reference with itself.
    let compared = self_check(6, 0x5eed).unwrap();
    assert_eq!(compared, 6 * 4 * 24);
}

#[test]
fn flush_to_zero_restores_the_mode() {
    let tiny = std::hint::black_box(f32::MIN_POSITIVE);
    {
        let _ftz = FlushToZero::new();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(std::hint::black_box(tiny) * 0.5, 0.0);
    }
    assert!(std::hint::black_box(tiny) * 0.5 > 0.0);
}
