//! AArch64 kernels: baseline Advanced SIMD (NEON) and the optional dot-product extension,
//! selected by `dispatch`.
//!
//! The kernels are written once (`quant_kernels!`) and compiled per tier with that tier's
//! `#[target_feature]` set, so the per-tier integer-dot primitive inlines. Only that primitive
//! differs between tiers:
//!
//! - `dot16(acc, w, x)`: adds the dot of 16 signed weight bytes and 16 signed activation bytes
//!   into the i32 lanes of `acc`. NEON widens to i16 products (`smull`, which cannot overflow:
//!   `|-128 * -128|` fits i16) and pairwise-accumulates them (`sadalp`); dotprod uses `sdot`.
//!   Both are exact for any bytes.
//!
//! Packed weights are expanded in registers the way ggml's MIT-licensed
//! `ggml-cpu/arch/arm/quants.c` (pinned under `third_party/ggml`) does: a 32-byte Q4_K/Q5_K load
//! yields two groups (low nibbles via `vandq_u8`, high nibbles via `vshrq_n_u8`), Q5_K/Q6_K high
//! bits are masked and shifted into bit 4, Q6_K subtracts 32 in i8, Q5_1 spreads its 32 `qh` bits
//! over the bytes with `vtstq_u8`, and IQ4_XS looks its codebook up with `vqtbl1q_s8`. Q4_K, Q5_K
//! and Q5_1 values are at most 31, so they are already positive `i8` and need no unsigned dot (as
//! in ggml). Every group's float term is computed exactly as in `scalar`, in the same order, so
//! results match the scalar reference bit for bit.
//!
//! BF16 (both tiers) widens to f32, which is exact, and multiply-adds in four vector accumulators,
//! within the bound stated on `matmul_bf16`.

use std::arch::is_aarch64_feature_detected;

use std::arch::aarch64::*;

use super::dispatch::{Isa, Kernels};

fn has_neon() -> bool {
    is_aarch64_feature_detected!("neon")
}

fn has_dotprod() -> bool {
    has_neon() && is_aarch64_feature_detected!("dotprod")
}

/// The kernels of `isa`, after checking at run time that the CPU has every feature they use.
pub(super) fn kernels_for(isa: Isa) -> Option<Kernels> {
    // SAFETY (both arms): the `has_*` checks detect exactly the features each set of kernels is
    // compiled for.
    match isa {
        Isa::Neon => has_neon().then(|| unsafe { neon::kernels(isa) }),
        Isa::NeonDotprod => has_dotprod().then(|| unsafe { dotprod::kernels(isa) }),
        _ => None,
    }
}

/// Safe entry points for the `#[target_feature]` kernels of one tier.
macro_rules! entry {
    ($($name:ident => $imp:ident),*) => {$(
        fn $name(w: &[u8], x: &Q8Activations, rows: usize, out: &mut [f32]) {
            // SAFETY: these functions are reachable only through the tier's `kernels`, whose
            // caller detected every feature the kernel is compiled for.
            unsafe { $imp(w, x, rows, out) }
        }
    )*};
}

macro_rules! quant_kernels {
    ($feat:literal) => {
        use std::arch::aarch64::*;

        use super::super::dispatch::{Isa, Kernels};
        use super::super::scalar::{
            IQ4_VALUES, LANES, f16_at, iq4_xs_scales, k_scales, kmin_term, q5_1_term, q8_0_term,
            reduce8, scaled_term,
        };
        use super::super::{GROUP, Q8Activations, QuantFormat};

        /// The kernels of this tier.
        ///
        /// # Safety
        /// The CPU must support every feature in this module's `target_feature` set.
        pub(super) unsafe fn kernels(isa: Isa) -> Kernels {
            Kernels {
                isa,
                q8_0,
                q5_1,
                q4_k,
                q5_k,
                q6_k,
                iq4_xs,
                bf16_dot: super::bf16_dot,
            }
        }

        entry!(q8_0 => q8_0_imp, q5_1 => q5_1_imp, q4_k => q4_k_imp, q5_k => q5_k_imp,
               q6_k => q6_k_imp, iq4_xs => iq4_xs_imp);

        #[target_feature(enable = $feat)]
        #[inline]
        fn load16(b: &[u8]) -> uint8x16_t {
            let b = &b[..16];
            // SAFETY: `b` holds 16 readable bytes; `ld1` has no alignment requirement.
            unsafe { vld1q_u8(b.as_ptr()) }
        }

        /// Two 16-byte halves of 32 bytes at the start of `b` (bounds-checked).
        #[target_feature(enable = $feat)]
        #[inline]
        fn load32(b: &[u8]) -> (uint8x16_t, uint8x16_t) {
            let b = &b[..32];
            // SAFETY: `b` holds 32 readable bytes; `ld1` has no alignment requirement.
            unsafe { (vld1q_u8(b.as_ptr()), vld1q_u8(b.as_ptr().add(16))) }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        fn load32i(b: &[i8]) -> (int8x16_t, int8x16_t) {
            let b = &b[..32];
            // SAFETY: as in `load32`.
            unsafe { (vld1q_s8(b.as_ptr()), vld1q_s8(b.as_ptr().add(16))) }
        }

        /// The exact dot of a 32-value group: weights `w`, activations `x`.
        #[target_feature(enable = $feat)]
        #[inline]
        fn dot32(w: (int8x16_t, int8x16_t), x: (int8x16_t, int8x16_t)) -> i32 {
            vaddvq_s32(dot16(dot16(vdupq_n_s32(0), w.0, x.0), w.1, x.1))
        }

        /// Q4_K/Q5_K values (0..=31) as signed bytes.
        #[target_feature(enable = $feat)]
        #[inline]
        fn signed(v: (uint8x16_t, uint8x16_t)) -> (int8x16_t, int8x16_t) {
            (vreinterpretq_s8_u8(v.0), vreinterpretq_s8_u8(v.1))
        }

        #[target_feature(enable = $feat)]
        fn q8_0_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            let row_bytes = QuantFormat::Q8_0.row_bytes(acts.columns()).expect("validated");
            for t in 0..acts.tokens() {
                let (xq, xd) = (acts.qs(t), acts.scales(t));
                for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
                    let mut acc = [0.0f32; LANES];
                    for (g, blk) in row.chunks_exact(QuantFormat::Q8_0.block_bytes()).enumerate() {
                        let wv = signed(load32(&blk[2..]));
                        let dot = dot32(wv, load32i(&xq[GROUP * g..]));
                        acc[g % LANES] += q8_0_term(f16_at(blk, 0), xd[g], dot);
                    }
                    out[t * rows + r] = reduce8(acc);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn q5_1_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            let format = QuantFormat::Q5_1;
            let row_bytes = format.row_bytes(acts.columns()).expect("validated");
            let (mask, b4) = (vdupq_n_u8(0x0f), vdupq_n_u8(0x10));
            // Byte `i` tests bit `i % 8` of the `qh` byte broadcast to its half.
            let bit = vcombine_u8(
                vcreate_u8(0x8040_2010_0804_0201),
                vcreate_u8(0x8040_2010_0804_0201),
            );
            // Bit 4 of values `8 * lo..` and `8 * hi..`: from `qh` bytes `lo` and `hi`.
            let high = |lo: u8, hi: u8| {
                vandq_u8(vtstq_u8(vcombine_u8(vdup_n_u8(lo), vdup_n_u8(hi)), bit), b4)
            };
            for t in 0..acts.tokens() {
                let (xq, xd, xs) = (acts.qs(t), acts.scales(t), acts.sums(t));
                for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
                    let mut acc = [0.0f32; LANES];
                    for (g, blk) in row.chunks_exact(format.block_bytes()).enumerate() {
                        // Values 0..16 are the low nibbles of `qs`, 16..32 the high ones; bit
                        // `i` of the little-endian `qh` word is bit 4 of value `i`.
                        let qs = load16(&blk[8..]);
                        let wv = (
                            vorrq_u8(vandq_u8(qs, mask), high(blk[4], blk[5])),
                            vorrq_u8(vshrq_n_u8::<4>(qs), high(blk[6], blk[7])),
                        );
                        let dot = dot32(signed(wv), load32i(&xq[GROUP * g..]));
                        acc[g % LANES] +=
                            q5_1_term(f16_at(blk, 0), f16_at(blk, 2), xd[g], dot, xs[g]);
                    }
                    out[t * rows + r] = reduce8(acc);
                }
            }
        }

        /// Q4_K (`q5 == false`) and Q5_K.
        #[target_feature(enable = $feat)]
        #[inline]
        fn kmin_imp(q5: bool, w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            let format = if q5 { QuantFormat::Q5K } else { QuantFormat::Q4K };
            let row_bytes = format.row_bytes(acts.columns()).expect("validated");
            let (qs_at, mask) = (if q5 { 48 } else { 16 }, vdupq_n_u8(0x0f));
            for t in 0..acts.tokens() {
                let (xq, xd, xs) = (acts.qs(t), acts.scales(t), acts.sums(t));
                for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
                    let mut acc = [0.0f32; LANES];
                    for (block, blk) in row.chunks_exact(format.block_bytes()).enumerate() {
                        let (scales, mins) = k_scales(&blk[4..16]);
                        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                        // Q5_K: bit `j` of each of these 32 bytes is bit 4 of group `j` (unused
                        // for Q4_K).
                        let qh = load32(&blk[16..]);
                        for pair in 0..4 {
                            // One 32-byte load holds groups `2 * pair` (low nibbles) and
                            // `2 * pair + 1` (high nibbles).
                            let q = load32(&blk[qs_at + GROUP * pair..]);
                            for half in 0..2 {
                                let j = 2 * pair + half;
                                let nib = |v| {
                                    if half == 0 {
                                        vandq_u8(v, mask)
                                    } else {
                                        vshrq_n_u8::<4>(v)
                                    }
                                };
                                let mut wv = (nib(q.0), nib(q.1));
                                if q5 {
                                    // Isolate bit `j` and move it to bit 4.
                                    let (bit, by) =
                                        (vdupq_n_u8(1 << j), vdupq_n_s8(4 - j as i8));
                                    let high = |h| vshlq_u8(vandq_u8(h, bit), by);
                                    wv = (vorrq_u8(wv.0, high(qh.0)), vorrq_u8(wv.1, high(qh.1)));
                                }
                                let group = 8 * block + j;
                                let dot = dot32(signed(wv), load32i(&xq[GROUP * group..]));
                                acc[group % LANES] += kmin_term(
                                    d, dmin, scales[j], mins[j], xd[group], dot, xs[group],
                                );
                            }
                        }
                    }
                    out[t * rows + r] = reduce8(acc);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn q4_k_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            kmin_imp(false, w, acts, rows, out)
        }

        #[target_feature(enable = $feat)]
        fn q5_k_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            kmin_imp(true, w, acts, rows, out)
        }

        #[target_feature(enable = $feat)]
        fn q6_k_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            let format = QuantFormat::Q6K;
            let row_bytes = format.row_bytes(acts.columns()).expect("validated");
            let mask = vdupq_n_u8(0x0f);
            for t in 0..acts.tokens() {
                let (xq, xd) = (acts.qs(t), acts.scales(t));
                for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
                    let mut acc = [0.0f32; LANES];
                    for (block, blk) in row.chunks_exact(format.block_bytes()).enumerate() {
                        let d = f16_at(blk, 208);
                        for half in 0..2 {
                            // Groups `4 * half + quarter`: low bits from `ql` bytes
                            // `64 * half + 32 * (quarter % 2)` (low nibbles for quarters 0-1,
                            // high for 2-3), high bits `2 * quarter` of `qh` bytes `32 * half`.
                            let ql = [
                                load32(&blk[64 * half..]),
                                load32(&blk[64 * half + 32..]),
                            ];
                            let qh = load32(&blk[128 + 32 * half..]);
                            for quarter in 0..4 {
                                let l = ql[quarter % 2];
                                let nib = |v| {
                                    if quarter < 2 {
                                        vandq_u8(v, mask)
                                    } else {
                                        vshrq_n_u8::<4>(v)
                                    }
                                };
                                // Isolate bits `2 * quarter..+2` and move them to bits 4-5.
                                let (bits, by) = (
                                    vdupq_n_u8(3 << (2 * quarter)),
                                    vdupq_n_s8(4 - 2 * quarter as i8),
                                );
                                let m32 = vdupq_n_s8(32);
                                let value = |v, h| {
                                    let q = vorrq_u8(nib(v), vshlq_u8(vandq_u8(h, bits), by));
                                    vsubq_s8(vreinterpretq_s8_u8(q), m32)
                                };
                                let (w0, w1) = (value(l.0, qh.0), value(l.1, qh.1));
                                let j = 4 * half + quarter;
                                let group = 8 * block + j;
                                let x = load32i(&xq[GROUP * group..]);
                                // Q6_K has a separate signed scale for each half-group.
                                let lo = vaddvq_s32(dot16(vdupq_n_s32(0), w0, x.0));
                                let hi = vaddvq_s32(dot16(vdupq_n_s32(0), w1, x.1));
                                let sum = i32::from(blk[192 + 2 * j] as i8) * lo
                                    + i32::from(blk[193 + 2 * j] as i8) * hi;
                                acc[group % LANES] += scaled_term(d, xd[group], sum);
                            }
                        }
                    }
                    out[t * rows + r] = reduce8(acc);
                }
            }
        }

        #[target_feature(enable = $feat)]
        fn iq4_xs_imp(w: &[u8], acts: &Q8Activations, rows: usize, out: &mut [f32]) {
            let format = QuantFormat::Iq4Xs;
            let row_bytes = format.row_bytes(acts.columns()).expect("validated");
            let table = vreinterpretq_s8_u8(load16(&IQ4_VALUES.map(|v| v as u8)));
            let mask = vdupq_n_u8(0x0f);
            for t in 0..acts.tokens() {
                let (xq, xd) = (acts.qs(t), acts.scales(t));
                for (r, row) in w.chunks_exact(row_bytes).take(rows).enumerate() {
                    let mut acc = [0.0f32; LANES];
                    for (block, blk) in row.chunks_exact(format.block_bytes()).enumerate() {
                        let d = f16_at(blk, 0);
                        for (j, ls) in iq4_xs_scales(blk).into_iter().enumerate() {
                            // Group `j`: the low nibbles of `qs[16j..16j + 16]`, then the high.
                            let q = load16(&blk[8 + 16 * j..]);
                            let wv = (
                                vqtbl1q_s8(table, vandq_u8(q, mask)),
                                vqtbl1q_s8(table, vshrq_n_u8::<4>(q)),
                            );
                            let group = 8 * block + j;
                            let dot = dot32(wv, load32i(&xq[GROUP * group..]));
                            acc[group % LANES] += scaled_term(d, xd[group], ls * dot);
                        }
                    }
                    out[t * rows + r] = reduce8(acc);
                }
            }
        }
    };
}

/// Baseline Advanced SIMD.
mod neon {
    quant_kernels!("neon");

    /// `acc` plus the dot of `w` and `x`, via widening i16 products.
    #[target_feature(enable = "neon")]
    #[inline]
    fn dot16(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
        let lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
        let hi = vmull_high_s8(w, x);
        vpadalq_s16(vpadalq_s16(acc, lo), hi)
    }
}

/// Advanced SIMD plus the ARMv8.2-A dot-product extension.
mod dotprod {
    quant_kernels!("neon,dotprod");

    /// `acc` plus the dot of `w` and `x`, via `sdot`.
    #[target_feature(enable = "neon,dotprod")]
    #[inline]
    fn dot16(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
        vdotq_s32(acc, w, x)
    }
}

fn bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: this function is only placed in `Kernels` by `kernels_for` after `has_neon`.
    unsafe { bf16_dot_imp(a, b) }
}

/// Eight BF16 values widened to f32 (exact: BF16 is the top half of an f32).
#[target_feature(enable = "neon")]
#[inline]
fn bf16x8(v: &[u16; 8]) -> (float32x4_t, float32x4_t) {
    // SAFETY: `v` is 16 readable bytes; `ld1` has no alignment requirement.
    let h = unsafe { vld1q_u16(v.as_ptr()) };
    (
        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h))),
        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h)),
    )
}

/// Four independent accumulators hide the FMA latency. BF16 products are exact in f32, so each
/// `fmla` rounds only the sum, as an add would.
#[target_feature(enable = "neon")]
fn bf16_dot_imp(a: &[u16], b: &[u16]) -> f32 {
    let mut acc = [vdupq_n_f32(0.0); 4];
    let ((a8, a_tail), (b8, b_tail)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let ((a2, a_rest), (b2, b_rest)) = (a8.as_chunks::<2>(), b8.as_chunks::<2>());
    for (x, y) in a2.iter().zip(b2) {
        for k in 0..2 {
            let ((xl, xh), (yl, yh)) = (bf16x8(&x[k]), bf16x8(&y[k]));
            acc[2 * k] = vfmaq_f32(acc[2 * k], xl, yl);
            acc[2 * k + 1] = vfmaq_f32(acc[2 * k + 1], xh, yh);
        }
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        let ((xl, xh), (yl, yh)) = (bf16x8(x), bf16x8(y));
        acc[0] = vfmaq_f32(acc[0], xl, yl);
        acc[1] = vfmaq_f32(acc[1], xh, yh);
    }
    let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
    vaddvq_f32(sum) + super::scalar::bf16_dot(a_tail, b_tail)
}

#[cfg(test)]
mod tests;
