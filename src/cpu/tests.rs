use super::*;
use crate::ggml::{self, ffi, types};

/// xorshift64*: deterministic test data without a dependency.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "powerpc64", target_endian = "little")
    ))]
    pub(super) fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 56) as u8).collect()
    }

    /// A random finite f16 (normal or subnormal, either sign).
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "powerpc64", target_endian = "little")
    ))]
    pub(super) fn finite_f16(&mut self) -> u16 {
        let h = (self.next() >> 48) as u16;
        if h & 0x7c00 == 0x7c00 { h & !0x4000 } else { h }
    }

    /// Uniform in [-1, 1).
    pub(super) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

pub(super) fn ggml_type(format: QuantFormat) -> ffi::ggml_type {
    format.ggml_type()
}

/// Byte offsets of the f16 scale fields inside one block.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "powerpc64", target_endian = "little")
))]
pub(super) fn f16_fields(format: QuantFormat) -> &'static [usize] {
    match format {
        QuantFormat::Q8_0 | QuantFormat::Iq4Xs => &[0],
        QuantFormat::Q5_1 | QuantFormat::Q4K | QuantFormat::Q5K => &[0, 2],
        QuantFormat::Q6K => &[208],
    }
}

/// `rows` rows of random block bytes (every quant and sub-scale bit pattern is possible) with
/// finite f16 scales.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "powerpc64", target_endian = "little")
))]
pub(super) fn random_weights(
    rng: &mut Rng,
    format: QuantFormat,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let n = rows * format.row_bytes(columns).unwrap();
    let mut w = rng.bytes(n);
    for blk in w.chunks_exact_mut(format.block_bytes()) {
        for &at in f16_fields(format) {
            blk[at..at + 2].copy_from_slice(&rng.finite_f16().to_le_bytes());
        }
    }
    w
}

/// Random Q8_0 activations: any quant byte (including -128) and finite f16 scales.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "powerpc64", target_endian = "little")
))]
pub(super) fn random_acts(rng: &mut Rng, columns: usize, tokens: usize) -> Q8Activations {
    let mut packed = rng.bytes(columns / GROUP * tokens * Q8_0_BLOCK_BYTES);
    for blk in packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
        blk[..2].copy_from_slice(&rng.finite_f16().to_le_bytes());
    }
    Q8Activations::from_q8_0(&packed, columns, tokens).unwrap()
}

pub(super) fn run(
    isa: Isa,
    format: QuantFormat,
    w: &[u8],
    x: &Q8Activations,
    rows: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; rows * x.tokens()];
    matmul_q8_with_isa(isa, format, w, x, rows, &mut out).unwrap();
    out
}

#[test]
fn half_to_f32_matches_ggml_for_every_f16() {
    for h in 0..=u16::MAX {
        let got = scalar::half_to_f32(h);
        // SAFETY: pure conversion function.
        let want = unsafe { ffi::ggml_fp16_to_fp32(h) };
        if want.is_nan() {
            assert!(got.is_nan(), "{h:#06x}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}");
        }
    }
}

#[test]
fn f32_to_f16_rounds_like_ggml() {
    let mut rng = Rng::new(7);
    let specials = [
        0.0f32,
        -0.0,
        65504.0,
        65519.99,
        65520.0,
        1e10,
        5.96e-8,
        2.98e-8,
        2.99e-8,
        1e-10,
        1.0,
        -1.0,
        0.1,
        6.1035e-5,
        6.1e-5,
        f32::INFINITY,
        -f32::INFINITY,
    ];
    let randoms = (0..1_000_000).map(|_| f32::from_bits((rng.next() >> 32) as u32));
    for v in specials.into_iter().chain(randoms).filter(|v| !v.is_nan()) {
        // SAFETY: pure conversion function.
        let want = unsafe { ffi::ggml_fp32_to_fp16(v) };
        assert_eq!(scalar::f32_to_f16(v), want, "{v:e} ({:#010x})", v.to_bits());
    }
    assert!(scalar::half_to_f32(scalar::f32_to_f16(f32::NAN)).is_nan());
}

#[test]
fn activation_quantization_is_bit_identical_to_ggml() {
    let mut rng = Rng::new(11);
    let columns = 256;
    let mut x: Vec<f32> = (0..columns * 5).map(|_| rng.unit() * 3.0).collect();
    x[..32].fill(0.0); // a zero block: d = 0, all quants 0
    x[32..64].iter_mut().for_each(|v| *v *= 1e-7); // subnormal f16 scale
    x[64..96].iter_mut().for_each(|v| *v *= 1e8); // f16 scale overflows to inf, as in ggml
    x[96] = -5.0; // the block's amax is negative
    let ours = Q8Activations::quantize(&x, columns, 5).unwrap();
    assert_eq!(ours.to_q8_0(), ggml::quantize(types::Q8_0, &x, columns));
    let back = Q8Activations::from_q8_0(&ours.to_q8_0(), columns, 5).unwrap();
    assert_eq!(back.to_q8_0(), ours.to_q8_0());
}

/// f64 dot of ggml's dequantization of `w` with the dequantized activations, and the sum of
/// `|w_i * x_i|` for the error bound.
fn exact_dot(format: QuantFormat, wrow: &[u8], x: &Q8Activations, t: usize) -> (f64, f64) {
    let columns = x.columns();
    let wd = ggml::dequantize(ggml_type(format), wrow, columns);
    let mut dot = 0.0f64;
    let mut mag = 0.0f64;
    for (i, &wv) in wd.iter().enumerate() {
        let xv = f64::from(x.scales(t)[i / GROUP]) * f64::from(x.qs(t)[i]);
        dot += f64::from(wv) * xv;
        mag += (f64::from(wv) * xv).abs();
    }
    (dot, mag)
}

/// The reference agrees with ggml's dequantization of every format (layouts, scales, mins,
/// codebook): within `2^-16` of `sum |w_i * x_i|`, which covers f32 rounding of the group terms
/// (and of ggml's own dequantized values) with margin.
#[test]
fn scalar_reference_matches_ggml_dequantization() {
    let mut rng = Rng::new(3);
    let (rows, columns, tokens) = (5, 512, 3);
    let xf: Vec<f32> = (0..tokens * columns).map(|_| rng.unit()).collect();
    let x = Q8Activations::quantize(&xf, columns, tokens).unwrap();
    for format in QuantFormat::ALL {
        let wf: Vec<f32> = (0..rows * columns).map(|_| rng.unit() * 0.1).collect();
        let w = ggml::quantize(ggml_type(format), &wf, columns);
        let got = run(Isa::Scalar, format, &w, &x, rows);
        let rb = format.row_bytes(columns).unwrap();
        for t in 0..tokens {
            for r in 0..rows {
                let (want, mag) = exact_dot(format, &w[r * rb..][..rb], &x, t);
                let err = (f64::from(got[t * rows + r]) - want).abs();
                assert!(
                    err <= mag * 2f64.powi(-16),
                    "{format:?} row {r} token {t}: {} vs {want} (err {err:e}, mag {mag:e})",
                    got[t * rows + r]
                );
            }
        }
    }
}

/// Q8_0 x Q8_0 is the one case where ggml's CPU `vec_dot` takes the same inputs.
#[test]
fn q8_0_matches_ggml_vec_dot_within_rounding() {
    // SAFETY: initializes ggml's CPU tables; idempotent.
    unsafe { ffi::ggml_cpu_init() };
    let mut rng = Rng::new(5);
    let (rows, columns) = (4, 1024);
    let xf: Vec<f32> = (0..columns).map(|_| rng.unit()).collect();
    let wf: Vec<f32> = (0..rows * columns).map(|_| rng.unit()).collect();
    let x = Q8Activations::quantize(&xf, columns, 1).unwrap();
    let xp = x.to_q8_0();
    let w = ggml::quantize(types::Q8_0, &wf, columns);
    let got = run(Isa::Scalar, QuantFormat::Q8_0, &w, &x, rows);
    // SAFETY: static traits table.
    let vec_dot = unsafe { (*ffi::ggml_get_type_traits_cpu(types::Q8_0)).vec_dot }.unwrap();
    let rb = QuantFormat::Q8_0.row_bytes(columns).unwrap();
    for (r, &g) in got.iter().enumerate() {
        let mut s = 0.0f32;
        // SAFETY: one row of `columns` Q8_0 values each; `s` receives one float.
        unsafe {
            vec_dot(
                columns as i32,
                &mut s,
                0,
                w[r * rb..].as_ptr().cast(),
                0,
                xp.as_ptr().cast(),
                0,
                1,
            )
        };
        let (_, mag) = exact_dot(QuantFormat::Q8_0, &w[r * rb..][..rb], &x, 0);
        assert!(
            f64::from((g - s).abs()) <= mag * 2f64.powi(-18),
            "row {r}: {g} vs ggml {s}"
        );
    }
}

/// Every supported ISA's BF16 dot is within `gamma_n * sum |a_i b_i|` of the exact dot.
#[test]
fn bf16_matmul_is_within_the_stated_bound() {
    let mut rng = Rng::new(9);
    let (rows, columns, tokens) = (3, 1000, 2);
    let bf = |v: f32| crate::hf::nvfp4::f32_to_bf16(v);
    let w: Vec<u16> = (0..rows * columns).map(|_| bf(rng.unit() * 4.0)).collect();
    let x: Vec<u16> = (0..tokens * columns).map(|_| bf(rng.unit())).collect();
    let u = 2f64.powi(-24);
    let gamma = columns as f64 * u / (1.0 - columns as f64 * u);
    for isa in supported_isas() {
        let mut out = vec![0.0; rows * tokens];
        matmul_bf16_with_isa(isa, &w, &x, rows, columns, &mut out).unwrap();
        for t in 0..tokens {
            for r in 0..rows {
                let (mut exact, mut mag) = (0.0f64, 0.0f64);
                for i in 0..columns {
                    let p = f64::from(scalar::bf16_to_f32(w[r * columns + i]))
                        * f64::from(scalar::bf16_to_f32(x[t * columns + i]));
                    exact += p;
                    mag += p.abs();
                }
                let err = (f64::from(out[t * rows + r]) - exact).abs();
                assert!(
                    err <= gamma * mag,
                    "{isa:?}: err {err:e} > {:e}",
                    gamma * mag
                );
            }
        }
    }
}

#[test]
fn rejects_bad_shapes() {
    let x = Q8Activations::from_q8_0(&[0; 34], 32, 1).unwrap();
    let mut out = [0.0f32; 1];
    assert_eq!(
        matmul_q8(QuantFormat::Q4K, &[], &x, 1, &mut out),
        Err(MatmulError::Columns {
            columns: 32,
            block: 256
        })
    );
    assert!(matches!(
        matmul_q8(QuantFormat::Q8_0, &[0; 33], &x, 1, &mut out),
        Err(MatmulError::Weights { .. })
    ));
    assert!(matches!(
        matmul_q8(QuantFormat::Q8_0, &[0; 34], &x, 1, &mut [0.0; 2]),
        Err(MatmulError::Output { .. })
    ));
    assert!(matches!(
        Q8Activations::from_q8_0(&[0; 35], 32, 1),
        Err(MatmulError::Activations { .. })
    ));
    assert!(matches!(
        Q8Activations::quantize(&[0.0; 48], 48, 1),
        Err(MatmulError::Columns { .. })
    ));
    assert!(matches!(
        matmul_bf16(&[0; 3], &[0; 5], 1, 3, &mut out),
        Err(MatmulError::Activations { .. })
    ));
    assert_eq!(
        matmul_q8(QuantFormat::Q8_0, &[0; 34], &x, 1, &mut out),
        Ok(())
    );
}

#[test]
fn isa_names_round_trip_and_scalar_is_always_supported() {
    for isa in Isa::ALL {
        assert_eq!(Isa::from_name(isa.name()), Some(isa));
    }
    assert_eq!(Isa::from_name(" AVX2 "), Some(Isa::Avx2));
    assert_eq!(Isa::from_name(" DoTpRoD "), Some(Isa::NeonDotprod));
    let supported = supported_isas();
    assert_eq!(supported.first(), Some(&Isa::Scalar));
    let automatic = supported.iter().copied().filter(|i| !i.opt_in());
    assert_eq!(Some(detected_isa()), automatic.max());
}

/// The word-mask scale unpack the SIMD kernels use equals the reference `get_scale_min_k4` port.
#[cfg(target_arch = "x86_64")]
#[test]
fn packed_k_scales_match_the_reference() {
    let mut rng = Rng::new(12);
    for i in 0..20_000 {
        let q = if i < 2 {
            vec![[0u8, 0xff][i]; 12]
        } else {
            rng.bytes(12)
        };
        let (sc, m) = scalar::k_scales(&q);
        let packed = scalar::k_scales_packed(&q);
        for j in 0..8 {
            assert_eq!(i32::from(packed[j]), sc[j], "{q:?} scale {j}");
            assert_eq!(i32::from(packed[8 + j]), m[j], "{q:?} min {j}");
        }
    }
}
