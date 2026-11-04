//! Portable reference kernels: the oracle every SIMD kernel is compared with bit for bit.
//!
//! Block layouts and dequantization formulas follow ggml's MIT-licensed `ggml-common.h` and
//! `ggml-quants.c` (`dequantize_row_*`), pinned under `third_party/ggml`. The float evaluation
//! order is this module's own (see the module docs of `cpu`): the group terms below are written
//! so a SIMD kernel can compute eight of them lane-wise with the same IEEE operations.

use super::{GROUP, Q8Activations, QuantFormat};

/// Accumulator lanes: group `g` of a row adds its term to accumulator `g % LANES`.
pub(super) const LANES: usize = 8;

/// The fixed final sum of the accumulators. A 256-bit kernel stores its accumulator vector and
/// calls this, so the order is shared by construction.
pub(super) fn reduce8(a: [f32; LANES]) -> f32 {
    ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]))
}

// Group terms (f32, evaluated left to right, never fused).

/// Q8_0: `dw * xd * dot`.
pub(super) fn q8_0_term(dw: f32, xd: f32, dot: i32) -> f32 {
    (dw * xd) * dot as f32
}

/// Q5_1: weights are `d * q + m`, so the term is `xd * (d * dot + m * sum(x))`.
pub(super) fn q5_1_term(d: f32, m: f32, xd: f32, dot: i32, sum: i32) -> f32 {
    xd * (d * dot as f32 + m * sum as f32)
}

/// Q4_K / Q5_K: weights are `d * sc * q - dmin * m`.
pub(super) fn kmin_term(d: f32, dmin: f32, sc: i32, m: i32, xd: f32, dot: i32, sum: i32) -> f32 {
    xd * (d * (sc * dot) as f32 - dmin * (m * sum) as f32)
}

/// Q6_K / IQ4_XS: the block scale times an integer sum that already includes the sub-scales.
pub(super) fn scaled_term(d: f32, xd: f32, isum: i32) -> f32 {
    (d * xd) * isum as f32
}

/// IQ4_NL / IQ4_XS non-linear codebook (`kvalues_iq4nl` in ggml-common.h).
pub(super) const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The eight 6-bit scales and mins of a Q4_K/Q5_K superblock (`get_scale_min_k4`).
pub(super) fn k_scales(scales: &[u8]) -> ([i32; 8], [i32; 8]) {
    let q = &scales[..12];
    let mut sc = [0; 8];
    let mut m = [0; 8];
    for j in 0..8 {
        let (s, mn) = if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        };
        sc[j] = i32::from(s);
        m[j] = i32::from(mn);
    }
    (sc, m)
}

/// [`k_scales`] as bytes, computed with whole-word masks (ggml's `utmp`/`kmask` trick in
/// `ggml_vec_dot_q4_K_q8_K`): bytes 0..8 are the scales, 8..16 the mins.
#[cfg(target_arch = "x86_64")]
pub(super) fn k_scales_packed(scales: &[u8]) -> [u8; 16] {
    const K1: u32 = 0x3f3f_3f3f;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x0303_0303;
    let q = &scales[..12];
    let word = |i: usize| u32::from_le_bytes([q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]]);
    let (u0, u1, u2) = (word(0), word(1), word(2));
    let parts = [
        u0 & K1,
        (u2 & K2) | (((u0 >> 6) & K3) << 4),
        u1 & K1,
        ((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4),
    ];
    let mut out = [0; 16];
    for (o, p) in out.as_chunks_mut::<4>().0.iter_mut().zip(parts) {
        o.copy_from_slice(&p.to_le_bytes());
    }
    out
}

/// The eight 6-bit IQ4_XS sub-scales minus 32.
pub(super) fn iq4_xs_scales(block: &[u8]) -> [i32; 8] {
    let high = u16::from_le_bytes([block[2], block[3]]);
    let low = &block[4..8];
    std::array::from_fn(|g| {
        let ls = ((low[g / 2] >> (4 * (g % 2))) & 0xf) | ((((high >> (2 * g)) & 3) as u8) << 4);
        i32::from(ls) - 32
    })
}

/// Q5_1 quants (0..=31) of one 24-byte block: `d`, `m`, `qh[4]`, `qs[16]`.
fn decode_q5_1(b: &[u8]) -> [u8; 32] {
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    std::array::from_fn(|i| {
        let nib = if i < 16 { qs[i] & 0xf } else { qs[i - 16] >> 4 };
        nib | ((((qh >> i) & 1) as u8) << 4)
    })
}

/// Q4_K quants (0..=15) of one superblock: `d`, `dmin`, `scales[12]`, `qs[128]`.
fn decode_q4_k(b: &[u8]) -> [u8; 256] {
    let qs = &b[16..144];
    std::array::from_fn(|i| {
        let (g, l) = (i / 32, i % 32);
        let packed = qs[(g / 2) * 32 + l];
        if g % 2 == 0 {
            packed & 0xf
        } else {
            packed >> 4
        }
    })
}

/// Q5_K quants (0..=31) of one superblock: `d`, `dmin`, `scales[12]`, `qh[32]`, `qs[128]`.
fn decode_q5_k(b: &[u8]) -> [u8; 256] {
    let (qh, qs) = (&b[16..48], &b[48..176]);
    std::array::from_fn(|i| {
        let (g, l) = (i / 32, i % 32);
        let packed = qs[(g / 2) * 32 + l];
        let nib = if g % 2 == 0 {
            packed & 0xf
        } else {
            packed >> 4
        };
        nib | (((qh[l] >> g) & 1) << 4)
    })
}

/// Q6_K quants (-32..=31) of one superblock: `ql[128]`, `qh[64]`, `scales[16]`, `d`.
fn decode_q6_k(b: &[u8]) -> [i8; 256] {
    let (ql, qh) = (&b[..128], &b[128..192]);
    std::array::from_fn(|i| {
        let (n, k, l) = (i / 128, (i % 128) / 32, i % 32);
        let lo = ql[64 * n + 32 * (k % 2) + l];
        let lo = if k < 2 { lo & 0xf } else { lo >> 4 };
        let hi = (qh[32 * n + l] >> (2 * k)) & 3;
        (lo | (hi << 4)) as i8 - 32
    })
}

/// IQ4_XS values of one superblock: `d`, `scales_h`, `scales_l[4]`, `qs[128]`.
fn decode_iq4_xs(b: &[u8]) -> [i8; 256] {
    let qs = &b[8..136];
    std::array::from_fn(|i| {
        let (g, l) = (i / 32, i % 32);
        let packed = qs[16 * g + l % 16];
        IQ4_VALUES[usize::from(if l < 16 { packed & 0xf } else { packed >> 4 })]
    })
}

/// The term of one 34-byte Q8_0 weight block against 32 activation quants with scale `xd`.
pub(super) fn q8_0_block(blk: &[u8], x: &[i8], xd: f32) -> f32 {
    let wq: [i8; GROUP] = std::array::from_fn(|i| blk[2 + i] as i8);
    q8_0_term(f16_at(blk, 0), xd, dot(&wq, &x[..GROUP]))
}

/// The term of one 24-byte Q5_1 weight block (`sum` is the sum of the 32 activation quants).
pub(super) fn q5_1_block(blk: &[u8], x: &[i8], xd: f32, sum: i32) -> f32 {
    let d = dot(&decode_q5_1(blk), &x[..GROUP]);
    q5_1_term(f16_at(blk, 0), f16_at(blk, 2), xd, d, sum)
}

fn dot<W: Copy + Into<i32>>(w: &[W], x: &[i8]) -> i32 {
    w.iter()
        .zip(x)
        .map(|(&a, &b)| a.into() * i32::from(b))
        .sum()
}

/// The reference matmul for every quantized format. Callers validated the shapes.
pub(super) fn matmul(
    format: QuantFormat,
    w: &[u8],
    acts: &Q8Activations,
    rows: usize,
    out: &mut [f32],
) {
    let columns = acts.columns();
    let Some(row_bytes) = format.row_bytes(columns) else {
        return;
    };
    let bb = format.block_bytes();
    for t in 0..acts.tokens() {
        let (xq, xd, xs) = (acts.qs(t), acts.scales(t), acts.sums(t));
        for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
            let mut acc = [0.0f32; LANES];
            let mut add = |g: usize, term: f32| acc[g % LANES] += term;
            for (b, blk) in row.chunks_exact(bb).enumerate() {
                match format {
                    QuantFormat::Q8_0 => add(b, q8_0_block(blk, &xq[GROUP * b..], xd[b])),
                    QuantFormat::Q5_1 => {
                        add(b, q5_1_block(blk, &xq[GROUP * b..], xd[b], xs[b]));
                    }
                    QuantFormat::Q4K | QuantFormat::Q5K => {
                        let q = if format == QuantFormat::Q4K {
                            decode_q4_k(blk)
                        } else {
                            decode_q5_k(blk)
                        };
                        let (sc, m) = k_scales(&blk[4..16]);
                        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                        for j in 0..8 {
                            let g = 8 * b + j;
                            let dt = dot(&q[GROUP * j..][..GROUP], &xq[GROUP * g..][..GROUP]);
                            add(g, kmin_term(d, dmin, sc[j], m[j], xd[g], dt, xs[g]));
                        }
                    }
                    QuantFormat::Q6K => {
                        let q = decode_q6_k(blk);
                        let sc = &blk[192..208];
                        let d = f16_at(blk, 208);
                        for j in 0..8 {
                            let g = 8 * b + j;
                            let (wq, x) = (&q[GROUP * j..][..GROUP], &xq[GROUP * g..][..GROUP]);
                            let isum = i32::from(sc[2 * j] as i8) * dot(&wq[..16], &x[..16])
                                + i32::from(sc[2 * j + 1] as i8) * dot(&wq[16..], &x[16..]);
                            add(g, scaled_term(d, xd[g], isum));
                        }
                    }
                    QuantFormat::Iq4Xs => {
                        let q = decode_iq4_xs(blk);
                        let ls = iq4_xs_scales(blk);
                        let d = f16_at(blk, 0);
                        for j in 0..8 {
                            let g = 8 * b + j;
                            let dt = dot(&q[GROUP * j..][..GROUP], &xq[GROUP * g..][..GROUP]);
                            add(g, scaled_term(d, xd[g], ls[j] * dt));
                        }
                    }
                }
            }
            out[t * rows + r] = reduce8(acc);
        }
    }
}

macro_rules! scalar_kernel {
    ($($name:ident => $format:expr),* $(,)?) => {$(
        pub(super) fn $name(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            matmul($format, w, acts, rows, out)
        }
    )*};
}

scalar_kernel! {
    q8_0 => QuantFormat::Q8_0,
    q5_1 => QuantFormat::Q5_1,
    q4_k => QuantFormat::Q4K,
    q5_k => QuantFormat::Q5K,
    q6_k => QuantFormat::Q6K,
    iq4_xs => QuantFormat::Iq4Xs,
}

pub(super) fn bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| bf16_to_f32(x) * bf16_to_f32(y))
        .sum()
}

pub(super) fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// The little-endian f16 at `bytes[at..at + 2]`, converted exactly.
pub(super) fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// IEEE binary16 to f32 (exact; NaN payloads are kept).
pub(super) fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            let shift = m.leading_zeros() - 21;
            let m = (m << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (0x1f, m) => sign | (0xff << 23) | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE binary16, rounding to nearest even (what ggml's `GGML_FP32_TO_FP16` and F16C do).
/// Adapted from the `half` crate's `f32_to_f16_fallback` (MIT/Apache-2.0).
pub(super) fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7f80_0000;
    let man = x & 0x007f_ffff;
    let half_sign = sign >> 16;
    if exp == 0x7f80_0000 {
        let nan_bit = if man == 0 { 0 } else { 0x0200 };
        return (half_sign | 0x7c00 | nan_bit | (man >> 13)) as u16;
    }
    let half_exp = ((exp >> 23) as i32) - 127 + 15;
    if half_exp >= 0x1f {
        return (half_sign | 0x7c00) as u16;
    }
    if half_exp <= 0 {
        if 14 - half_exp > 24 {
            return half_sign as u16;
        }
        let man = man | 0x0080_0000;
        let mut half_man = man >> (14 - half_exp);
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        return (half_sign | half_man) as u16;
    }
    let bits = half_sign | ((half_exp as u32) << 10) | (man >> 13);
    let round_bit = 0x0000_1000;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        (bits + 1) as u16
    } else {
        bits as u16
    }
}
