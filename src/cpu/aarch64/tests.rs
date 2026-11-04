//! AArch64 NEON and dotprod kernels against the scalar reference, bit for bit.

use super::super::tests::{Rng, random_acts, random_weights, run};
use super::super::{GROUP, Q8_0_BLOCK_BYTES, Q8Activations, QuantFormat};
use super::*;

fn runnable() -> Vec<Isa> {
    [Isa::Neon, Isa::NeonDotprod]
        .into_iter()
        .filter(|&isa| kernels_for(isa).is_some())
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn assert_matches_scalar(
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

#[test]
fn detection_follows_cpu_features() {
    assert_eq!(kernels_for(Isa::Scalar).map(|k| k.isa), None);
    assert_eq!(
        kernels_for(Isa::Neon).map(|k| k.isa),
        has_neon().then_some(Isa::Neon)
    );
    assert_eq!(
        kernels_for(Isa::NeonDotprod).map(|k| k.isa),
        has_dotprod().then_some(Isa::NeonDotprod)
    );
    assert!(!has_dotprod() || has_neon());
}

#[test]
fn random_blocks_are_bit_exact() {
    let mut rng = Rng::new(307);
    for format in QuantFormat::ALL {
        let blocks: &[usize] = if format.block_len() == GROUP {
            &[1, 7, 8, 13, 24]
        } else {
            &[1, 3]
        };
        for &blocks in blocks {
            let columns = blocks * format.block_len();
            for tokens in [1, 3, 4, 9] {
                for _ in 0..3 {
                    let rows = 3;
                    let w = random_weights(&mut rng, format, rows, columns);
                    let x = random_acts(&mut rng, columns, tokens);
                    assert_matches_scalar(format, &w, &x, rows, "random");
                }
            }
        }
    }
}

#[test]
fn extreme_blocks_are_bit_exact() {
    let mut rng = Rng::new(19);
    for format in QuantFormat::ALL {
        let columns = 256 + format.block_len();
        let rows = 2;
        let bytes = rows * format.row_bytes(columns).unwrap();
        for wbyte in [0x00u8, 0xff, 0x80, 0x7f] {
            for xbyte in [0x80u8, 0x7f, 0x81, 0x00] {
                let mut w = vec![wbyte; bytes];
                for block in w.chunks_exact_mut(format.block_bytes()) {
                    for &at in super::super::tests::f16_fields(format) {
                        block[at..at + 2].copy_from_slice(&rng.finite_f16().to_le_bytes());
                    }
                }
                let tokens = 3;
                let mut packed = vec![xbyte; columns / GROUP * tokens * Q8_0_BLOCK_BYTES];
                for block in packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
                    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
                let x = Q8Activations::from_q8_0(&packed, columns, tokens).unwrap();
                assert_matches_scalar(
                    format,
                    &w,
                    &x,
                    rows,
                    &format!("w {wbyte:#04x} x {xbyte:#04x}"),
                );
            }
        }
    }
}

#[test]
fn q8_0_byte_positions_pair_up() {
    let columns = 9 * GROUP;
    let mut w = vec![0u8; 9 * Q8_0_BLOCK_BYTES];
    let mut packed = vec![0u8; 9 * Q8_0_BLOCK_BYTES];
    for block in w
        .as_chunks_mut::<Q8_0_BLOCK_BYTES>()
        .0
        .iter_mut()
        .chain(packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0)
    {
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    for a in (0..=255u8).step_by(7).chain([0x80, 0x7f, 0xff]) {
        for b in (0..=255u8).step_by(11).chain([0x80, 0x7f, 0x01]) {
            for (i, (wb, xb)) in w
                .as_chunks_mut::<Q8_0_BLOCK_BYTES>()
                .0
                .iter_mut()
                .zip(packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0)
                .enumerate()
            {
                let at = 2 + (i * 7 + usize::from(a)) % GROUP;
                wb[2..].fill(0);
                xb[2..].fill(0);
                wb[at] = a;
                xb[at] = b;
            }
            let x = Q8Activations::from_q8_0(&packed, columns, 1).unwrap();
            assert_matches_scalar(QuantFormat::Q8_0, &w, &x, 1, "one byte");
        }
    }
}
