//! Host kernels with the bits of the GPU's decode expert matvec (#37).
//!
//! `src/cuda/expert_mv/` multiplies a routed expert's packed rows (Q4_K, Q5_K, Q5_1, Q8_0) by
//! Q8_1 activation blocks. [`matvec`] computes the same row sums on the CPU, from the same two
//! byte strings, with every float operation the kernels' machine code performs, in its order,
//! so the result is the kernel's bit for bit. That makes a missed expert computable on the host
//! without changing a token.
//!
//! # What the kernels do (read from the SASS nvcc makes of them, `--use_fast_math`)
//!
//! A row of `k` values is `units` vec_dot units ([`units_per_row`]). `L` lanes share the row
//! ([`lanes_per_row`], a power of two): lane `l` adds units `l, l + L, l + 2L, ...` in that
//! order to an accumulator that starts at `+0.0`, then the lanes meet in a butterfly
//! (`s += shfl_xor(s, off)` for `off = L/2 .. 1`) and lane 0 holds the row's sum. Per unit
//! (ggml's `vec_dot_*_q8_1`, vecdotq.cuh; all integer dot products are exact):
//!
//! - **Q4_K, Q5_K** (16 values of two 32-value sub-blocks `i = 0, 1`):
//!   `sumf_d = fma(d8[i], float(dot1[i] * sc[i]), sumf_d)` and
//!   `sumf_m = fma(d8[i], float(dot2[i] * m[i]), sumf_m)` from `+0.0`, then
//!   `unit = fma(sumf_d, d, -(sumf_m * dmin))` and `acc = unit + acc`.
//! - **Q5_1** (16 values of one block): `(d5d8, m5s8) = half(d5 * d8), half(m5 * s8)` (a
//!   half-precision multiply, rounded to nearest even in f16), then
//!   `unit = fma(float(d5d8), float(sumi), float(m5s8) * 0.5)` and `acc = unit + acc`.
//! - **Q8_0** (8 values): `acc = fma(float(d_w) * float(d_x), float(sumi), acc)`.
//!
//! Every f32 operation rounds to nearest even and flushes subnormal results to zero (FTZ), so
//! the kernels here run with MXCSR's FTZ and DAZ set ([`FlushToZero`]).
//!
//! What is **not** here: the kernels' activation quantization (`quantize_x`: an approximate
//! reciprocal division) and SwiGLU (`MUFU.EX2`, `MUFU.RCP`), which the GPU's function unit
//! computes in ways no CPU instruction reproduces. A host share therefore takes the Q8_1 blocks
//! the GPU quantized and hands the raw gate and up sums back for the GPU to activate.
//!
//! [`matvec_scalar`] is the reference; the AVX2 + FMA + F16C form computes the same operations
//! lane-wise and is checked against it ([`self_check`], the unit tests).

use super::QuantFormat;

#[cfg(test)]
mod tests;

/// Bytes of a `block_q8_1`: the f16 scale, the f16 sum (scale times the quants' sum), 32 quants.
pub const Q8_1_BLOCK_BYTES: usize = 36;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("{0:?} experts are not computed by these kernels")]
    Format(QuantFormat),
    #[error("{k} values a row are not a whole number of blocks")]
    Columns { k: usize },
    #[error("weights hold {actual} bytes, not a whole number of {row}-byte rows for the output")]
    Weights { actual: usize, row: usize },
    #[error("activations hold {actual} bytes, expected {expected}")]
    Activations { actual: usize, expected: usize },
}

/// vec_dot units per row of `k` values (the kernels' `units_per_row`: `qi / vdr` per block).
pub fn units_per_row(format: QuantFormat, k: usize) -> Option<usize> {
    match format {
        QuantFormat::Q4K | QuantFormat::Q5K => Some(k / 256 * 16),
        QuantFormat::Q5_1 => Some(k / 32 * 2),
        QuantFormat::Q8_0 => Some(k / 32 * 4),
        _ => None,
    }
}

/// Lanes sharing a row: 32, halved while a lane would get fewer than 4 units (the kernels'
/// `lanes_per_row`).
pub fn lanes_per_row(units: usize) -> usize {
    let mut l = 32;
    while l > 1 && units / l < 4 {
        l /= 2;
    }
    l
}

/// An f16 bit pattern as f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = u32::from(h >> 10) & 0x1f;
    let man = u32::from(h) & 0x3ff;
    match exp {
        0 => {
            // A subnormal: man * 2^-24, a normal f32.
            let v = man as f32 * f32::from_bits(0x3380_0000);
            f32::from_bits(sign | v.to_bits())
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (man << 13)),
        e => f32::from_bits(sign | ((e + 112) << 23) | (man << 13)),
    }
}

/// `x` rounded to f16, nearest even, subnormal results kept (IEEE 754, as the GPU's `HMUL2`
/// and F16C's `vcvtps2ph` round).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    if e == 0 {
        return sign; // zero, or an f32 subnormal: far below half of f16's smallest
    }
    let exp = e - 127;
    let sig = u64::from(m | 0x80_0000); // the value is sig * 2^(exp - 23)
    if exp > 15 {
        return sign | 0x7c00;
    }
    // Bits dropped: 13 for a normal f16, more below 2^-14 (a unit of 2^-24).
    let shift = if exp >= -14 { 13 } else { (-(exp + 1)) as u32 };
    if shift > 25 {
        return sign;
    }
    let kept = sig >> shift;
    let rem = sig & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let kept = kept + u64::from(rem > half || (rem == half && kept & 1 == 1));
    let h = if exp >= -14 {
        // `kept` holds the leading 1 at bit 10; a carry out of the mantissa bumps the exponent
        // (to infinity from the largest).
        (((exp + 14) as u64) << 10) + kept
    } else {
        kept
    };
    sign | h as u16
}

/// The GPU's f16 multiply (`__hmul2` on a lane): the exact product rounded once to f16, as f32.
/// The product of two f16 values is exact in f32 (22 significant bits), so one rounding to f16
/// is the IEEE result.
#[inline]
fn hmul(a: u16, b: u16) -> f32 {
    f16_to_f32(f32_to_f16(f16_to_f32(a) * f16_to_f32(b)))
}

/// MXCSR with flush-to-zero and denormals-are-zero set for a scope: the GPU's `.FTZ` float
/// operations. No effect on other architectures.
pub struct FlushToZero {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    saved: u32,
}

impl FlushToZero {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let mut saved = 0u32;
            // SAFETY: stmxcsr and ldmxcsr on a u32 of this frame; SSE2 is x86-64's baseline.
            // The mode holds only while the guard lives, around the kernels below.
            unsafe {
                std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut saved, options(nostack));
                let on = saved | 0x8040;
                std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &on, options(nostack, readonly));
            }
            Self { saved }
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self { saved: 0 }
    }
}

impl Default for FlushToZero {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlushToZero {
    fn drop(&mut self) {
        // SAFETY: restores the value `new` read.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &self.saved, options(nostack, readonly));
        }
    }
}

/// A K-quant block's 6-bit scale and minimum of sub-block `j` (ggml's `get_scale_min_k4`).
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        (i32::from(q[j] & 63), i32::from(q[j + 4] & 63))
    } else {
        (
            i32::from((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4)),
            i32::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
        )
    }
}

#[inline]
fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// Unit `c` of a Q4_K (`Q5`: Q5_K) row against the activation blocks `xq`.
fn unit_k<const Q5: bool>(row: &[u8], xq: &[u8], c: usize) -> f32 {
    let bytes = if Q5 { 176 } else { 144 };
    let (kbx, j) = (c / 16, c % 16);
    let (p, jj) = (j / 4, j % 4);
    let b = &row[kbx * bytes..][..bytes];
    let (d, dmin) = (f16_to_f32(le16(&b[0..])), f16_to_f32(le16(&b[2..])));
    let scales = &b[4..16];
    let (qh, qs) = if Q5 {
        (&b[16..48], &b[48..176])
    } else {
        (&b[..0], &b[16..144])
    };
    let (mut sumf_d, mut sumf_m) = (0.0f32, 0.0f32);
    for i in 0..2 {
        let sub = 2 * p + i;
        let (sc, m) = scale_min_k4(sub, scales);
        let xb = &xq[(kbx * 8 + sub) * Q8_1_BLOCK_BYTES..][..Q8_1_BLOCK_BYTES];
        let d8 = f16_to_f32(le16(xb));
        let (mut dot1, mut dot2) = (0i32, 0i32);
        for half in 0..2 {
            for t in 0..4 {
                let idx = 16 * half + 4 * jj + t;
                let mut q = i32::from((qs[32 * p + idx] >> (4 * i)) & 0xf);
                if Q5 {
                    q |= i32::from((qh[idx] >> sub) & 1) << 4;
                }
                let x = i32::from(xb[4 + idx] as i8);
                dot1 += q * x;
                dot2 += x;
            }
        }
        sumf_d = d8.mul_add((dot1 * sc) as f32, sumf_d);
        sumf_m = d8.mul_add((dot2 * m) as f32, sumf_m);
    }
    sumf_d.mul_add(d, -(sumf_m * dmin))
}

/// Unit `c` of a Q5_1 row.
fn unit_q5_1(row: &[u8], xq: &[u8], c: usize) -> f32 {
    let (kbx, h) = (c / 2, c % 2);
    let b = &row[kbx * 24..][..24];
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    let xb = &xq[kbx * Q8_1_BLOCK_BYTES..][..Q8_1_BLOCK_BYTES];
    let mut sumi = 0i32;
    for half in 0..2 {
        for t in 0..8 {
            let v = 16 * half + 8 * h + t;
            let nib = if half == 0 {
                qs[v] & 0xf
            } else {
                qs[v - 16] >> 4
            };
            let q = i32::from(nib) | (((qh >> v) & 1) as i32) << 4;
            sumi += q * i32::from(xb[4 + v] as i8);
        }
    }
    let d5d8 = hmul(le16(&b[0..]), le16(&xb[0..]));
    let m5s8 = hmul(le16(&b[2..]), le16(&xb[2..]));
    d5d8.mul_add(sumi as f32, m5s8 * 0.5)
}

/// Unit `c` of a Q8_0 row: the scale product and the integer dot (the kernel fuses their
/// product into the lane's sum).
fn unit_q8_0(row: &[u8], xq: &[u8], c: usize) -> (f32, f32) {
    let (kbx, q) = (c / 4, c % 4);
    let b = &row[kbx * 34..][..34];
    let xb = &xq[kbx * Q8_1_BLOCK_BYTES..][..Q8_1_BLOCK_BYTES];
    let mut sumi = 0i32;
    for t in 0..8 {
        let v = 8 * q + t;
        sumi += i32::from(b[2 + v] as i8) * i32::from(xb[4 + v] as i8);
    }
    (f16_to_f32(le16(b)) * f16_to_f32(le16(xb)), sumi as f32)
}

/// The lanes' butterfly: lane 0 after `s += shfl_xor(s, off)` for `off = lanes / 2 .. 1`.
fn butterfly(mut acc: [f32; 32], lanes: usize) -> f32 {
    let mut off = lanes / 2;
    while off > 0 {
        let prev = acc;
        for l in 0..lanes {
            acc[l] = prev[l] + prev[l ^ off];
        }
        off /= 2;
    }
    acc[0]
}

fn row_scalar(format: QuantFormat, row: &[u8], xq: &[u8], units: usize, lanes: usize) -> f32 {
    let mut acc = [0.0f32; 32];
    for c in 0..units {
        let a = &mut acc[c % lanes];
        *a = match format {
            QuantFormat::Q4K => unit_k::<false>(row, xq, c) + *a,
            QuantFormat::Q5K => unit_k::<true>(row, xq, c) + *a,
            QuantFormat::Q5_1 => unit_q5_1(row, xq, c) + *a,
            _ => {
                let (d, s) = unit_q8_0(row, xq, c);
                d.mul_add(s, *a)
            }
        };
    }
    butterfly(acc, lanes)
}

/// A checked call's row bytes, units and lanes.
fn shape(
    format: QuantFormat,
    w: &[u8],
    xq: &[u8],
    k: usize,
    rows: usize,
) -> Result<(usize, usize, usize), Error> {
    let units = units_per_row(format, k).ok_or(Error::Format(format))?;
    let row = format
        .row_bytes(k)
        .filter(|_| k > 0)
        .ok_or(Error::Columns { k })?;
    if w.len() != rows * row {
        return Err(Error::Weights {
            actual: w.len(),
            row,
        });
    }
    let expected = k / 32 * Q8_1_BLOCK_BYTES;
    if xq.len() != expected {
        return Err(Error::Activations {
            actual: xq.len(),
            expected,
        });
    }
    Ok((row, units, lanes_per_row(units)))
}

/// `out[r]` = row `r` of `w` (`out.len()` packed rows of `k` values, as in a GGUF) times the
/// activation column `xq` (`k / 32` `block_q8_1`), with the GPU kernel's bits. The reference:
/// one unit at a time.
pub fn matvec_scalar(
    format: QuantFormat,
    w: &[u8],
    xq: &[u8],
    k: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let (row, units, lanes) = shape(format, w, xq, k, out.len())?;
    let _ftz = FlushToZero::new();
    for (r, o) in out.iter_mut().enumerate() {
        *o = row_scalar(format, &w[r * row..][..row], xq, units, lanes);
    }
    Ok(())
}

/// Whether [`matvec`] has a vector form on this CPU (AVX2, FMA and F16C).
pub fn simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// [`matvec_scalar`]'s result through the vector kernels where the CPU has them (detected at
/// run time) and the row's lanes fill whole vectors (the shapes the kernels' lane rule gives
/// Qwen3.8-Flash-Next's experts and any like them); the reference otherwise.
pub fn matvec(
    format: QuantFormat,
    w: &[u8],
    xq: &[u8],
    k: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    #[cfg(target_arch = "x86_64")]
    {
        let (row, _, lanes) = shape(format, w, xq, k, out.len())?;
        let blocks = k / format.block_len();
        let whole = match format {
            QuantFormat::Q4K | QuantFormat::Q5K => lanes == 32 && blocks.is_multiple_of(2),
            QuantFormat::Q5_1 => lanes == 8 && blocks.is_multiple_of(4),
            QuantFormat::Q8_0 => lanes == 16 && blocks.is_multiple_of(4),
            _ => false,
        };
        if whole && simd() {
            let _ftz = FlushToZero::new();
            // SAFETY: `simd` detected AVX2, FMA and F16C.
            unsafe { x86::matvec(format, w, xq, row, blocks, out) };
            return Ok(());
        }
    }
    matvec_scalar(format, w, xq, k, out)
}

/// Compares [`matvec`] with [`matvec_scalar`] on `rounds` random matrices per format, of the
/// decode shapes (2560-value rows for Q4_K and Q5_K, 640 for Q5_1 and Q8_0), with scales that
/// reach f16's subnormals and quants that reach -128. Returns the first difference.
pub fn self_check(rounds: usize, seed: u64) -> Result<usize, String> {
    let mut s = seed | 1;
    let mut next = move || {
        // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut compared = 0;
    for round in 0..rounds {
        for (format, k) in [
            (QuantFormat::Q4K, 2560),
            (QuantFormat::Q5K, 2560),
            (QuantFormat::Q5_1, 640),
            (QuantFormat::Q8_0, 640),
        ] {
            let rows = 24;
            let row = format.row_bytes(k).expect("whole blocks");
            let mut w: Vec<u8> = (0..rows * row).map(|_| (next() >> 32) as u8).collect();
            let mut xq: Vec<u8> = (0..k / 32 * Q8_1_BLOCK_BYTES)
                .map(|_| (next() >> 32) as u8)
                .collect();
            // Finite f16 scales: small ones (products in f16's subnormal range) in odd rounds.
            let mut scale = |tiny: bool| {
                let exp = if tiny {
                    1 + (next() >> 40) % 8
                } else {
                    4 + (next() >> 40) % 14
                };
                let sign = ((next() >> 50) & 1) << 15;
                (sign | (exp << 10) | ((next() >> 30) & 0x3ff)) as u16
            };
            let tiny = round % 2 == 1;
            let block = format.block_bytes();
            for b in w.chunks_exact_mut(block) {
                b[0..2].copy_from_slice(&scale(tiny).to_le_bytes());
                if format != QuantFormat::Q8_0 {
                    b[2..4].copy_from_slice(&scale(tiny).to_le_bytes());
                }
            }
            for b in xq.chunks_exact_mut(Q8_1_BLOCK_BYTES) {
                b[0..2].copy_from_slice(&scale(tiny).to_le_bytes());
                b[2..4].copy_from_slice(&scale(false).to_le_bytes());
            }
            let (mut a, mut b) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            matvec_scalar(format, &w, &xq, k, &mut a).map_err(|e| e.to_string())?;
            matvec(format, &w, &xq, k, &mut b).map_err(|e| e.to_string())?;
            for r in 0..rows {
                if a[r].to_bits() != b[r].to_bits() {
                    return Err(format!(
                        "{format:?} round {round} row {r}: scalar {:#010x} ({}), vector {:#010x} ({})",
                        a[r].to_bits(),
                        a[r],
                        b[r].to_bits(),
                        b[r]
                    ));
                }
            }
            compared += rows;
        }
    }
    Ok(compared)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{Q8_1_BLOCK_BYTES, QuantFormat, le16, scale_min_k4};

    const RN: i32 = _MM_FROUND_TO_NEAREST_INT;

    /// An activation column unpacked once for all the rows: the quants, each block's scale and
    /// sum as f32 and as f16 bits, and (the K-quants' `dot2`) the quants' sums per unit.
    struct Acts {
        qs: Vec<i8>,
        d: Vec<f32>,
        s: Vec<f32>,
        /// Per block: for `jj` in 0..4, the sum of quants `4jj..4jj + 4` and `16 + 4jj..`.
        sums: Vec<[i32; 4]>,
    }

    impl Acts {
        fn new(xq: &[u8]) -> Self {
            let n = xq.len() / Q8_1_BLOCK_BYTES;
            let mut a = Self {
                qs: Vec::with_capacity(n * 32),
                d: Vec::with_capacity(n),
                s: Vec::with_capacity(n),
                sums: Vec::with_capacity(n),
            };
            for b in xq.chunks_exact(Q8_1_BLOCK_BYTES) {
                a.d.push(super::f16_to_f32(le16(&b[0..])));
                a.s.push(super::f16_to_f32(le16(&b[2..])));
                let q = &b[4..];
                a.qs.extend(q.iter().map(|&v| v as i8));
                let sum4 = |at: usize| q[at..at + 4].iter().map(|&v| i32::from(v as i8)).sum();
                let mut sums = [0i32; 4];
                for (jj, s) in sums.iter_mut().enumerate() {
                    let (lo, hi): (i32, i32) = (sum4(4 * jj), sum4(16 + 4 * jj));
                    *s = lo + hi;
                }
                a.sums.push(sums);
            }
            a
        }
    }

    /// # Safety
    /// The CPU has AVX2, FMA and F16C; the shapes are [`super::matvec`]'s (`w` is `out.len()`
    /// rows of `row` bytes, `blocks` weight blocks each, the lanes fill whole vectors).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn matvec(
        format: QuantFormat,
        w: &[u8],
        xq: &[u8],
        row: usize,
        blocks: usize,
        out: &mut [f32],
    ) {
        let x = Acts::new(xq);
        for (r, o) in out.iter_mut().enumerate() {
            let rw = &w[r * row..][..row];
            *o = match format {
                QuantFormat::Q4K => row_k::<false>(rw, &x, blocks),
                QuantFormat::Q5K => row_k::<true>(rw, &x, blocks),
                QuantFormat::Q5_1 => row_q5_1(rw, &x, blocks),
                _ => row_q8_0(rw, &x, blocks),
            };
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn load32(b: &[u8]) -> __m256i {
        let b = &b[..32];
        // SAFETY: 32 readable bytes; unaligned load.
        unsafe { _mm256_loadu_si256(b.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn load32i(b: &[i8]) -> __m256i {
        let b = &b[..32];
        // SAFETY: 32 readable bytes; unaligned load.
        unsafe { _mm256_loadu_si256(b.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn load4(v: &[i32; 4]) -> __m128i {
        // SAFETY: 16 readable bytes; unaligned load.
        unsafe { _mm_loadu_si128(v.as_ptr().cast()) }
    }

    /// Lanes 0..4 `a`, lanes 4..8 `b`.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn pair_i32(a: i32, b: i32) -> __m256i {
        _mm256_set_epi32(b, b, b, b, a, a, a, a)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn pair_f32(a: f32, b: f32) -> __m256 {
        _mm256_set_ps(b, b, b, b, a, a, a, a)
    }

    /// The f16 at `b[0..2]`, converted exactly and broadcast.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn f16_splat(b: &[u8]) -> __m256 {
        _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(le16(b)))))
    }

    /// Eight i32 lanes: lane `m` the dot of bytes `4m..4m + 4` of unsigned `u` (at most 31) and
    /// signed `s` (any byte): `vpmaddubsw`'s i16 pair sums cannot saturate.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn dot_us(u: __m256i, s: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1))
    }

    /// The low half plus the high half: lane `m` + lane `m + 4`.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn fold(v: __m256i) -> __m128i {
        _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
    }

    /// Lane 0 of the butterfly over 8 lanes (offsets 4, 2, 1).
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn reduce8(v: __m256) -> f32 {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<1>(s2, s2));
        _mm_cvtss_f32(s1)
    }

    /// Byte `i` is 0x10 where bit `i` of `bits` is set (ggml's `bytes_from_bits_32`, MIT).
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    fn fifth_bits(bits: u32) -> __m256i {
        let t = _mm256_set1_epi32(bits as i32);
        let shuf = _mm256_setr_epi64x(
            0,
            0x0101_0101_0101_0101,
            0x0202_0202_0202_0202,
            0x0303_0303_0303_0303,
        );
        let b = _mm256_shuffle_epi8(t, shuf);
        let b = _mm256_or_si256(b, _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe_u64 as i64));
        _mm256_and_si256(
            _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1)),
            _mm256_set1_epi8(0x10),
        )
    }

    /// A Q4_K (`Q5`: Q5_K) row: 32 lanes in four vectors, two blocks a step.
    #[target_feature(enable = "avx2,fma,f16c")]
    fn row_k<const Q5: bool>(row: &[u8], x: &Acts, blocks: usize) -> f32 {
        let bytes = if Q5 { 176 } else { 144 };
        let m4 = _mm256_set1_epi8(0xf);
        let one = _mm256_set1_epi8(1);
        let zero = _mm256_setzero_ps();
        let mut acc = [zero; 4];
        for kbx in 0..blocks {
            let b = &row[kbx * bytes..][..bytes];
            let (d, dmin) = (f16_splat(&b[0..]), f16_splat(&b[2..]));
            let scales = &b[4..16];
            let (qh, qs) = if Q5 {
                (load32(&b[16..48]), &b[48..176])
            } else {
                (_mm256_setzero_si256(), &b[16..144])
            };
            let mut sc = [0i32; 8];
            let mut mn = [0i32; 8];
            for j in 0..8 {
                (sc[j], mn[j]) = scale_min_k4(j, scales);
            }
            for quarter in 0..2 {
                // dot1 of sub-block i of pair pp, lanes jj.
                let mut d1 = [_mm_setzero_si128(); 4];
                for pp in 0..2 {
                    let p = 2 * quarter + pp;
                    let q = load32(&qs[32 * p..]);
                    let mut lo = _mm256_and_si256(q, m4);
                    let mut hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4);
                    if Q5 {
                        let bit = |n: usize| {
                            let c = _mm_cvtsi32_si128(n as i32);
                            _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srl_epi16(qh, c), one))
                        };
                        lo = _mm256_or_si256(lo, bit(2 * p));
                        hi = _mm256_or_si256(hi, bit(2 * p + 1));
                    }
                    let sub = kbx * 8 + 2 * p;
                    d1[2 * pp] = fold(dot_us(lo, load32i(&x.qs[sub * 32..])));
                    d1[2 * pp + 1] = fold(dot_us(hi, load32i(&x.qs[(sub + 1) * 32..])));
                }
                let p = 2 * quarter;
                let sub = kbx * 8 + 2 * p;
                let dot1 = [
                    _mm256_set_m128i(d1[2], d1[0]),
                    _mm256_set_m128i(d1[3], d1[1]),
                ];
                let dot2 = [
                    _mm256_set_m128i(load4(&x.sums[sub + 2]), load4(&x.sums[sub])),
                    _mm256_set_m128i(load4(&x.sums[sub + 3]), load4(&x.sums[sub + 1])),
                ];
                let (mut sumf_d, mut sumf_m) = (zero, zero);
                for i in 0..2 {
                    let d8 = pair_f32(x.d[sub + i], x.d[sub + 2 + i]);
                    let scv = pair_i32(sc[2 * p + i], sc[2 * p + 2 + i]);
                    let mnv = pair_i32(mn[2 * p + i], mn[2 * p + 2 + i]);
                    let vd = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot1[i], scv));
                    let vm = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dot2[i], mnv));
                    sumf_d = _mm256_fmadd_ps(d8, vd, sumf_d);
                    sumf_m = _mm256_fmadd_ps(d8, vm, sumf_m);
                }
                let unit = _mm256_fmsub_ps(sumf_d, d, _mm256_mul_ps(sumf_m, dmin));
                let a = &mut acc[2 * (kbx % 2) + quarter];
                *a = _mm256_add_ps(unit, *a);
            }
        }
        // Offsets 16 and 8, then the 8 lanes left.
        let a = _mm256_add_ps(acc[0], acc[2]);
        let b = _mm256_add_ps(acc[1], acc[3]);
        reduce8(_mm256_add_ps(a, b))
    }

    /// A Q5_1 row: 8 lanes, four blocks a step.
    #[target_feature(enable = "avx2,fma,f16c")]
    fn row_q5_1(row: &[u8], x: &Acts, blocks: usize) -> f32 {
        let m4 = _mm_set1_epi8(0xf);
        let half = _mm256_set1_ps(0.5);
        let mut acc = _mm256_setzero_ps();
        for step in 0..blocks / 4 {
            let mut t = [_mm_setzero_si128(); 4];
            let (mut d5, mut m5) = ([0u16; 8], [0u16; 8]);
            let (mut d8, mut s8) = ([0.0f32; 8], [0.0f32; 8]);
            for i in 0..4 {
                let kbx = 4 * step + i;
                let b = &row[kbx * 24..][..24];
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                // SAFETY: 16 readable bytes; unaligned load.
                let qs = unsafe { _mm_loadu_si128(b[8..24].as_ptr().cast()) };
                let lo = _mm_and_si128(qs, m4);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(qs), m4);
                let v = _mm256_or_si256(_mm256_set_m128i(hi, lo), fifth_bits(qh));
                t[i] = fold(dot_us(v, load32i(&x.qs[kbx * 32..])));
                (d5[2 * i], d5[2 * i + 1]) = (le16(&b[0..]), le16(&b[0..]));
                (m5[2 * i], m5[2 * i + 1]) = (le16(&b[2..]), le16(&b[2..]));
                (d8[2 * i], d8[2 * i + 1]) = (x.d[kbx], x.d[kbx]);
                (s8[2 * i], s8[2 * i + 1]) = (x.s[kbx], x.s[kbx]);
            }
            // A block's two units: lanes 0 + 1 and 2 + 3 of its folded dots.
            let sumi = _mm256_set_m128i(_mm_hadd_epi32(t[2], t[3]), _mm_hadd_epi32(t[0], t[1]));
            // SAFETY: 16 readable bytes (eight u16), 32 (eight f32); unaligned loads.
            let (d5, m5, d8, s8) = unsafe {
                (
                    _mm256_cvtph_ps(_mm_loadu_si128(d5.as_ptr().cast())),
                    _mm256_cvtph_ps(_mm_loadu_si128(m5.as_ptr().cast())),
                    _mm256_loadu_ps(d8.as_ptr()),
                    _mm256_loadu_ps(s8.as_ptr()),
                )
            };
            // The f16 products: exact in f32, rounded once to f16.
            let d5d8 = _mm256_cvtph_ps(_mm256_cvtps_ph::<RN>(_mm256_mul_ps(d5, d8)));
            let m5s8 = _mm256_cvtph_ps(_mm256_cvtps_ph::<RN>(_mm256_mul_ps(m5, s8)));
            let unit = _mm256_fmadd_ps(d5d8, _mm256_cvtepi32_ps(sumi), _mm256_mul_ps(m5s8, half));
            acc = _mm256_add_ps(unit, acc);
        }
        reduce8(acc)
    }

    /// A Q8_0 row: 16 lanes in two vectors, four blocks a step. Signed bytes widened to i16, so
    /// any byte is exact (-128 too).
    #[target_feature(enable = "avx2,fma,f16c")]
    fn row_q8_0(row: &[u8], x: &Acts, blocks: usize) -> f32 {
        let mut acc = [_mm256_setzero_ps(); 2];
        for step in 0..blocks / 4 {
            for (j, a) in acc.iter_mut().enumerate() {
                let mut u = [_mm_setzero_si128(); 2];
                let mut dw = [0u16; 8];
                let mut dx = [0.0f32; 8];
                for i in 0..2 {
                    let kbx = 4 * step + 2 * j + i;
                    let b = &row[kbx * 34..][..34];
                    let w = load32(&b[2..34]);
                    let xv = load32i(&x.qs[kbx * 32..]);
                    let wide = |v: __m256i, hi: bool| {
                        _mm256_cvtepi8_epi16(if hi {
                            _mm256_extracti128_si256::<1>(v)
                        } else {
                            _mm256_castsi256_si128(v)
                        })
                    };
                    // Lane m: values 2m, 2m + 1 (p0), 16 + 2m, 17 + 2m (p1).
                    let p0 = _mm256_madd_epi16(wide(w, false), wide(xv, false));
                    let p1 = _mm256_madd_epi16(wide(w, true), wide(xv, true));
                    let h = _mm256_hadd_epi32(p0, p1);
                    let h = _mm256_hadd_epi32(h, h);
                    // The low half holds units 0 and 2, the high half 1 and 3.
                    u[i] = _mm_unpacklo_epi32(
                        _mm256_castsi256_si128(h),
                        _mm256_extracti128_si256::<1>(h),
                    );
                    dw[4 * i..4 * i + 4].fill(le16(b));
                    dx[4 * i..4 * i + 4].fill(x.d[kbx]);
                }
                let sumi = _mm256_set_m128i(u[1], u[0]);
                // SAFETY: 16 and 32 readable bytes; unaligned loads.
                let d = unsafe {
                    _mm256_mul_ps(
                        _mm256_cvtph_ps(_mm_loadu_si128(dw.as_ptr().cast())),
                        _mm256_loadu_ps(dx.as_ptr()),
                    )
                };
                *a = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), *a);
            }
        }
        reduce8(_mm256_add_ps(acc[0], acc[1]))
    }
}
