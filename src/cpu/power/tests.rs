//! The POWER kernels against the scalar reference, bit for bit (run on a ppc64le host).

use super::super::tests::{Rng, random_acts, random_weights, run};
use super::super::{GROUP, Q8_0_BLOCK_BYTES, Q8Activations, QuantFormat};
use super::*;

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
    let got = bits(&run(Isa::Vsx, format, w, x, rows));
    assert_eq!(
        got,
        want,
        "{format:?} ({what}, {} cols, {} tokens)",
        x.columns(),
        x.tokens()
    );
}

#[test]
fn every_ppc64le_cpu_has_the_kernels() {
    assert_eq!(kernels_for(Isa::Vsx).map(|k| k.isa), Some(Isa::Vsx));
    assert!(kernels_for(Isa::Scalar).is_none());
    assert!(kernels_for(Isa::Avx2).is_none());
}

/// Random block bytes and activations, several block and token counts.
#[test]
fn random_blocks_are_bit_exact() {
    let mut rng = Rng::new(308);
    for format in QuantFormat::ALL {
        let blocks: &[usize] = if format.block_len() == 32 {
            &[1, 7, 8, 13, 24]
        } else {
            &[1, 3]
        };
        for &nb in blocks {
            let columns = nb * format.block_len();
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

/// The expert shapes of Qwen3.8-Flash-Next (gate / up 2560 columns of K-quants, down 640 of
/// Q5_1 / Q8_0) at decode and verify-step token counts: the kernels' block loops end to end.
#[test]
fn expert_shapes_are_bit_exact() {
    let mut rng = Rng::new(637);
    for (format, columns) in [
        (QuantFormat::Q4K, 2560),
        (QuantFormat::Q5K, 2560),
        (QuantFormat::Q5_1, 640),
        (QuantFormat::Q8_0, 640),
        (QuantFormat::Q5_1, 2560),
        (QuantFormat::Q8_0, 2560),
    ] {
        for tokens in 1..=5 {
            let rows = 4;
            let w = random_weights(&mut rng, format, rows, columns);
            let x = random_acts(&mut rng, columns, tokens);
            assert_matches_scalar(format, &w, &x, rows, "expert shape");
        }
    }
}

/// Extreme bytes: all-zero and all-0xff quants and scales, Q8_0 weights of -128 against
/// activations of -128 and 127.
#[test]
fn extreme_blocks_are_bit_exact() {
    let mut rng = Rng::new(9);
    for format in QuantFormat::ALL {
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
                let tokens = 3;
                let mut packed = vec![xbyte; columns / GROUP * tokens * Q8_0_BLOCK_BYTES];
                for blk in packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
                    blk[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
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

/// One nonzero weight byte against one activation byte at every position of a block: a lane or
/// pairing mix-up in the `lxvw4x` loads would pair the wrong bytes.
#[test]
fn byte_positions_pair_up() {
    for format in [
        QuantFormat::Q8_0,
        QuantFormat::Q5_1,
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
        QuantFormat::Iq4Xs,
    ] {
        let columns = format.block_len();
        let bb = format.block_bytes();
        let mut rng = Rng::new(17);
        for _ in 0..40 {
            let mut w = random_weights(&mut rng, format, 1, columns);
            // Sparse activations: one random position nonzero per group.
            let mut packed = vec![0u8; columns / GROUP * Q8_0_BLOCK_BYTES];
            for blk in packed.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0 {
                blk[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                let at = 2 + (rng.bytes(1)[0] as usize) % 32;
                blk[at] = rng.bytes(1)[0] | 1;
            }
            w.truncate(bb);
            let x = Q8Activations::from_q8_0(&packed, columns, 1).unwrap();
            assert_matches_scalar(format, &w, &x, 1, "one byte");
        }
    }
}
