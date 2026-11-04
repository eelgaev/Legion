use super::super::tests::{Rng, random_acts, random_weights, run};
use super::super::{GROUP, Q8_0_BLOCK_BYTES, Q8Activations, QuantFormat};
use super::*;

const SIMD: [Isa; 5] = [
    Isa::Avx2,
    Isa::AvxVnni,
    Isa::Avx512Vnni,
    Isa::Avx512Bf16,
    Isa::Avx512Zmm,
];

/// The SIMD tiers this CPU runs; the others are reported as skipped.
fn runnable() -> Vec<Isa> {
    SIMD.into_iter()
        .filter(|&isa| {
            let ok = kernels_for(isa).is_some();
            if !ok {
                eprintln!("skipping {isa:?}: not supported by this CPU");
            }
            ok
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn assert_all_isas_match(
    format: QuantFormat,
    w: &[u8],
    x: &Q8Activations,
    rows: usize,
    what: &str,
) {
    let want = bits(&run(Isa::Scalar, format, w, x, rows));
    for isa in runnable() {
        let got = bits(&run(isa, format, w, x, rows));
        assert_eq!(
            got,
            want,
            "{format:?} on {isa:?} ({what}, {} cols, {} tokens)",
            x.columns(),
            x.tokens()
        );
    }
}

/// Random block bytes and activations, column counts with and without a partial group of eight
/// blocks, and token counts around the kernels' token chunk (4): bit-identical to the reference.
#[test]
fn random_blocks_are_bit_exact_on_every_tier() {
    let mut rng = Rng::new(306);
    for format in QuantFormat::ALL {
        let blocks: &[usize] = if format.block_len() == 32 {
            &[1, 7, 8, 13, 24]
        } else {
            &[1, 3]
        };
        for &nb in blocks {
            let columns = nb * format.block_len();
            for tokens in [1, 4, 5, 9] {
                for _ in 0..3 {
                    let rows = 3;
                    let w = random_weights(&mut rng, format, rows, columns);
                    let x = random_acts(&mut rng, columns, tokens);
                    assert_all_isas_match(format, &w, &x, rows, "random");
                }
            }
        }
    }
}

/// Extreme bytes: all-zero and all-0xff quants and sub-scales, Q8_0 weights of -128 against
/// activations of -128 and 127, where saturating or sign-trick dots would go wrong.
#[test]
fn extreme_blocks_are_bit_exact_on_every_tier() {
    let mut rng = Rng::new(1);
    for format in QuantFormat::ALL {
        // Nine blocks for Q8_0/Q5_1 (one SIMD chunk of eight and a scalar tail), two superblocks
        // otherwise.
        let columns = 256 + format.block_len();
        let rows = 2;
        let n = rows * format.row_bytes(columns).unwrap();
        for wbyte in [0x00u8, 0xff, 0x80, 0x7f] {
            for xbyte in [0x80u8, 0x7f, 0x81, 0x00] {
                let mut w = vec![wbyte; n];
                for blk in w.chunks_exact_mut(format.block_bytes()) {
                    for &at in super::super::tests::f16_fields(format) {
                        blk[at..at + 2].copy_from_slice(&rng.finite_f16().to_le_bytes());
                    }
                }
                let tokens = 5;
                let mut packed = vec![xbyte; columns / GROUP * tokens * Q8_0_BLOCK_BYTES];
                for blk in packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
                    blk[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1.0
                }
                let x = Q8Activations::from_q8_0(&packed, columns, tokens).unwrap();
                let what = format!("w {wbyte:#04x} x {xbyte:#04x}");
                assert_all_isas_match(format, &w, &x, rows, &what);
            }
        }
    }
}

/// Exhaustive single-group dots: one Q8_0 weight byte against one activation byte at every
/// position class, catching any lane/pair mix-up in the integer primitives.
#[test]
fn q8_0_byte_pairs_are_exact() {
    let columns = 9 * GROUP; // eight SIMD blocks and one scalar tail block
    let mut w = vec![0u8; 9 * 34];
    let mut packed = vec![0u8; 9 * 34];
    for blk in w
        .as_chunks_mut::<34>()
        .0
        .iter_mut()
        .chain(packed.as_chunks_mut::<34>().0)
    {
        blk[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    for a in (0..=255u8).step_by(3).chain([0x80, 0x7f, 0xff]) {
        for b in (0..=255u8).step_by(5).chain([0x80, 0x7f, 0x01]) {
            for (i, (wb, xb)) in w
                .as_chunks_mut::<34>()
                .0
                .iter_mut()
                .zip(packed.as_chunks_mut::<34>().0)
                .enumerate()
            {
                let pos = 2 + (i * 7 + usize::from(a)) % 32;
                wb[2..].fill(0);
                xb[2..].fill(0);
                wb[pos] = a;
                xb[pos] = b;
            }
            let x = Q8Activations::from_q8_0(&packed, columns, 1).unwrap();
            let want = 9.0 * f32::from(a as i8) * f32::from(b as i8);
            for isa in std::iter::once(Isa::Scalar).chain(runnable()) {
                let got = run(isa, QuantFormat::Q8_0, &w, &x, 1);
                assert_eq!(got[0], want, "{isa:?}: {a:#04x} x {b:#04x}");
            }
        }
    }
}

#[test]
fn detection_follows_cpu_features() {
    assert_eq!(kernels_for(Isa::Scalar).map(|k| k.isa), None);
    for (isa, has) in [
        (Isa::Avx2, has_avx2()),
        (Isa::AvxVnni, has_avx_vnni()),
        (Isa::Avx512Vnni, has_avx512_vnni()),
        (Isa::Avx512Bf16, has_avx512_bf16()),
        (Isa::Avx512Zmm, has_avx512_vnni()),
    ] {
        assert_eq!(kernels_for(isa).map(|k| k.isa), has.then_some(isa));
    }
    // Each tier implies AVX2 and the AVX-512 BF16 tier implies the VNNI one.
    assert!(!has_avx512_bf16() || has_avx512_vnni());
    assert!(!has_avx512_vnni() || has_avx2());
    assert!(!has_avx_vnni() || has_avx2());
}

/// The 512-bit tier forms the same BF16 dot as the tier the CPU would pick (`vdpbf16ps` where
/// present), and it is never the automatic choice.
#[test]
fn zmm_tier_is_opt_in() {
    assert!(Isa::Avx512Zmm.opt_in());
    assert_ne!(super::super::detected_isa(), Isa::Avx512Zmm);
    let Some(zmm) = kernels_for(Isa::Avx512Zmm) else {
        eprintln!("skipping Avx512Zmm: not supported by this CPU");
        return;
    };
    let best = if has_avx512_bf16() {
        Isa::Avx512Bf16
    } else {
        Isa::Avx512Vnni
    };
    let best = kernels_for(best).unwrap();
    let mut rng = Rng::new(512);
    for n in [1, 31, 32, 129, 1000] {
        let bf16 = |rng: &mut Rng| -> Vec<u16> {
            (0..n)
                .map(|_| (rng.unit().to_bits() >> 16) as u16)
                .collect()
        };
        let (a, b) = (bf16(&mut rng), bf16(&mut rng));
        assert_eq!(
            (zmm.bf16_dot)(&a, &b).to_bits(),
            (best.bf16_dot)(&a, &b).to_bits(),
            "{n} values"
        );
    }
}
