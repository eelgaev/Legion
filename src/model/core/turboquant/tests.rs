//! TurboQuant's format and math: the Lloyd-Max centroids, the randomized Hadamard transform, the
//! MSE per unit vector against the theory, the bit planes, the cell sizes and the startup checks.

use super::*;

/// Standard normal values (Box-Muller over an xorshift).
fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
        })
        .collect()
}

#[test]
fn centroids() {
    // b = 1: ±E|X| = ±sqrt(2 / pi).
    let g1 = lloyd_max(1);
    let e = (2.0 / std::f64::consts::PI).sqrt();
    assert!(
        (g1[0] + e).abs() < 1e-9 && (g1[1] - e).abs() < 1e-9,
        "{g1:?}"
    );
    assert!((e - 0.7979).abs() < 1e-4);
    // b = 2: Max's ±0.4528, ±1.5104.
    let g2 = lloyd_max(2);
    for (g, want) in g2.iter().zip([-1.5104, -0.4528, 0.4528, 1.5104]) {
        assert!((g - want).abs() < 1e-4, "{g2:?}");
    }
    for bits in 1..=4 {
        let g = lloyd_max(bits);
        assert_eq!(g.len(), 1 << bits);
        // Symmetric and ascending.
        for i in 0..g.len() {
            assert!(
                (g[i] + g[g.len() - 1 - i]).abs() < 1e-9,
                "{bits} bits: {g:?}"
            );
        }
        assert!(g.windows(2).all(|w| w[0] < w[1]));
        // The fixed point to 1e-9: each centroid the mean of its cell (between the midpoints).
        let t: Vec<f64> = std::iter::once(f64::NEG_INFINITY)
            .chain(g.windows(2).map(|w| 0.5 * (w[0] + w[1])))
            .chain(std::iter::once(f64::INFINITY))
            .collect();
        for i in 0..g.len() {
            let mean = (pdf(t[i]) - pdf(t[i + 1])) / (cdf(t[i + 1]) - cdf(t[i]));
            assert!((mean - g[i]).abs() < 1e-9, "{bits} bits, centroid {i}");
        }
    }
    // The F32 tables are the F64 centroids and their midpoints.
    let t = Tables::get();
    let g3 = lloyd_max(3);
    assert_eq!(t.g3[7], g3[7] as f32);
    assert!(t.t4[7].abs() < 1e-12);
    assert_eq!(t.of(4).0.len(), 16);
}

#[test]
fn cdf_is_accurate() {
    // Φ(1) and Φ(-2) to 1e-12.
    assert!((cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-12);
    assert!((cdf(-2.0) - 0.022_750_131_948_179_2).abs() < 1e-12);
    assert_eq!(cdf(f64::INFINITY), 1.0);
}

#[test]
fn theory_mse() {
    // The Lloyd-Max MSE of a standard normal (Max, 1960): what TurboQuant's MSE per unit vector
    // is, d large. The paper's Table / Theorem 1 rounds these to 0.03 and 0.009.
    assert!((lloyd_max_mse(1) - (1.0 - 2.0 / std::f64::consts::PI)).abs() < 1e-9);
    assert!((lloyd_max_mse(2) - 0.117_48).abs() < 1e-4);
    assert!((lloyd_max_mse(3) - 0.034_54).abs() < 1e-4);
    assert!((lloyd_max_mse(4) - 0.009_497).abs() < 1e-5);
}

#[test]
fn hadamard_is_orthogonal() {
    for d in [64usize, 128, 256, 512] {
        // Twice the unnormalized transform: d times the identity.
        let x = gaussian(d, d as u64);
        let mut y = x.clone();
        fwht(&mut y);
        fwht(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a * d as f32 - b).abs() < 1e-3 * d as f32, "d {d}");
        }
        // R's columns (R e_j) orthonormal; R^T R = I.
        if d <= 128 {
            let cols: Vec<Vec<f32>> = (0..d)
                .map(|j| {
                    let mut e = vec![0f32; d];
                    e[j] = 1.0;
                    rotate(7, &e)
                })
                .collect();
            for i in 0..d {
                for j in 0..d {
                    let dot: f32 = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-5,
                        "d {d}: <R e_{i}, R e_{j}> = {dot}"
                    );
                }
            }
        }
        // Round trip, and the norm kept.
        for seed in [0u32, 1, 61] {
            let r = rotate(seed, &x);
            let back = unrotate(seed, &r);
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() < 1e-5, "d {d}, seed {seed}");
            }
            let n2 = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
            assert!((n2(&x) - n2(&r)).abs() < 1e-4 * n2(&x));
        }
    }
    // The signs depend on the seed and are balanced.
    let plus = (0..4096).filter(|&j| sign(3, j) > 0.0).count();
    assert!((1900..2200).contains(&plus), "{plus}");
    assert!((0..64).any(|j| sign(0, j) != sign(1, j)));
}

#[test]
fn mse_per_unit_vector() {
    // 10k random unit vectors of 256 values: E ||x - x̂||² against the Lloyd-Max MSE (the paper
    // rounds the 3- and 4-bit values to 0.030 / 0.009; the theory is 0.03454 / 0.009497). The F16
    // norm adds ~1e-7.
    let d = 256;
    let n = 10_000;
    let all = gaussian(d * n, 11);
    for bits in [3u32, 4] {
        let want = lloyd_max_mse(bits);
        let mut words = vec![0u32; d * bits as usize / 32];
        let mut total = 0f64;
        for x in all.chunks(d) {
            let len = x.iter().map(|a| a * a).sum::<f32>().sqrt();
            let u: Vec<f32> = x.iter().map(|a| a / len).collect();
            let nb = encode_head(5, bits, &u, &mut words);
            let back = decode_head(5, bits, &words, nb, d);
            total += u
                .iter()
                .zip(&back)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>();
        }
        let mse = total / n as f64;
        eprintln!("{bits} bits: MSE per unit vector {mse:.6}, theory {want:.6}");
        assert!(
            (mse - want).abs() < 0.03 * want,
            "{bits} bits: MSE {mse}, theory {want}"
        );
    }
}

#[test]
fn bit_planes_and_scale() {
    let d = 128;
    let x = gaussian(d, 2);
    for bits in [3u32, 4] {
        let mut words = vec![0u32; d * bits as usize / 32];
        let nb = encode_head(9, bits, &x, &mut words);
        // The norm, as F16.
        let len = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((f32_of(nb) - len).abs() < 1e-3 * len);
        // Each value's index: the cell of its rotated coordinate (scaled to unit variance).
        let (g, t) = Tables::get().of(bits);
        let z = rotate(9, &x);
        let s = (d as f32).sqrt() / len;
        for (i, zi) in z.iter().map(|v| v * s).enumerate() {
            let idx = index_of(&words, bits, i);
            assert!(idx < g.len());
            if idx > 0 {
                assert!(zi > t[idx - 1] - 1e-4, "{bits} bits, value {i}");
            }
            if idx < t.len() {
                assert!(zi <= t[idx] + 1e-4, "{bits} bits, value {i}");
            }
        }
        // Rotated decode: n · g[idx] / sqrt(d).
        let r = decode_head_rotated(bits, &words, nb, d);
        assert_eq!(
            r[5],
            f32_of(nb) * inv_sqrt(d) * g[index_of(&words, bits, 5)]
        );
    }
    // A zero head: norm 0, decodes to zeros.
    let mut words = vec![0u32; 12];
    let nb = encode_head(0, 3, &[0.0; 128], &mut words);
    assert_eq!(f32_of(nb), 0.0);
    assert!(decode_head(0, 3, &words, nb, 128).iter().all(|&v| v == 0.0));
}

#[test]
fn cell_layout() {
    assert_eq!(head_bytes(128, 3), 48);
    assert_eq!(head_bytes(512, 4), 256);
    // Indices of all heads, the norms, padded to 16 bytes.
    assert_eq!(cell_bytes(128, 2, 3), 112);
    assert_eq!(cell_bytes(256, 16, 3), 16 * 98);
    assert_eq!(cell_bytes(256, 16, 4), 16 * 130);
    assert_eq!(cell_bytes(512, 4, 4), 1040);
    assert_eq!(cell_bytes(128, 1, 4), 80);
    assert!(head_ok(64) && head_ok(512) && !head_ok(96) && !head_ok(1024));
    // A cell of 2 heads: encode, read a head back, decode.
    let l = Layout {
        d: 128,
        n_kv: 2,
        bits: 4,
    };
    let x = gaussian(256, 4);
    let mut cell = vec![0u8; l.bytes()];
    l.encode(3, &x, &mut cell);
    let mut w = vec![0u32; 16];
    let n1 = encode_head(3, 4, &x[128..], &mut w);
    assert_eq!(l.head(&cell, 1), (w.clone(), n1));
    assert_eq!(
        &l.decode(3, &cell, false)[128..],
        &decode_head(3, 4, &w, n1, 128)[..]
    );
    assert_eq!(
        &l.decode(3, &cell, true)[128..],
        &decode_head_rotated(4, &w, n1, 128)[..]
    );
    // The padding stays zero.
    assert!(cell[2 * 64 + 4..].iter().all(|&b| b == 0));
}

/// One attention layer of `head_dim`-wide heads, 8 query and 2 kv heads.
fn hparams(head_dim: i64) -> crate::model::core::Hparams {
    crate::model::core::Hparams {
        n_layer: 1,
        n_embd: head_dim * 8,
        n_head: 8,
        n_head_kv: 2,
        head_dim,
        n_rot: head_dim as i32,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        full_attention_interval: 1,
        ssm_d_conv: 4,
        ssm_d_state: 16,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

#[test]
fn startup_checks() {
    use crate::model::core::{KvType, KvTypes};
    let hp = hparams(128);
    let tq = KvTypes {
        k: KvType::Tq4,
        v: KvType::Tq3,
    };
    assert!(check(&hp, tq, 1, false).is_ok());
    assert!(check(&hp, KvTypes::default(), 2, true).is_ok());
    let tp = check(&hp, tq, 2, false).unwrap_err();
    assert!(tp.contains("--tp 2"), "{tp}");
    assert!(check(&hp, tq, 1, true).unwrap_err().contains("--kv-slots"));
    let small = hparams(64);
    assert!(
        check(&small, tq, 1, false)
            .unwrap_err()
            .contains("head size is 64")
    );
}
