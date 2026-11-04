//! Host CPU kernels selected at runtime (#306).
//!
//! The quantized kernels multiply ggml-packed weight rows (Q4_K, Q5_K, Q6_K, Q8_0, Q5_1, IQ4_XS,
//! exactly as stored in GGUF files, so mapped weights need no repacking) by Q8_0-quantized
//! activations ([`Q8Activations`]), for any number of tokens. Outputs are token-major
//! (`out[t * rows + r]`). A BF16 path ([`matmul_bf16`]) serves BF16 weights and router dots.
//!
//! # Semantics
//!
//! [`scalar`] is the reference and the oracle. Every row is a sequence of 32-value groups. Each
//! group's integer dot product is exact; the group's float term (block scales times the integer
//! dot) uses a fixed sequence of f32 multiplies and adds (no FMA); group `g`'s term is added to
//! accumulator `g % 8`, and the eight accumulators are summed in a fixed tree ([`scalar::reduce8`]).
//! The SIMD kernels compute the same terms lane-wise in the same order, so **every ISA returns
//! bit-identical results** for the quantized formats, for any quant bytes (including -128) and
//! any non-NaN f16 scales (a NaN scale gives NaN everywhere, but F16C may quiet its payload).
//! Results are therefore independent of the host CPU. They are not bit-identical to ggml's `vec_dot` (which quantizes
//! activations to Q8_K for the K-quants and sums in its own order); the tests bound the difference
//! against the exact dot of the dequantized values.
//!
//! BF16 sums in vector-lane order, so its result differs from the scalar reference by float
//! rounding, within the bound stated on [`matmul_bf16`].
//!
//! Kernels are single-threaded; callers split rows across their own (NUMA-pinned) pools.

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod dispatch;
pub mod microbench;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
mod power;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use dispatch::{Isa, detected_isa, supported_isas};

use dispatch::Kernels;

/// Values per Q8_0 activation block (and per accumulation group).
pub const GROUP: usize = 32;
/// Bytes per packed Q8_0 block: an f16 scale and 32 signed bytes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Packed GGUF weight formats supported by the host kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Q5_1,
    Iq4Xs,
}

impl QuantFormat {
    pub const ALL: [QuantFormat; 6] = [
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8_0,
        Self::Q5_1,
        Self::Iq4Xs,
    ];

    /// Values per block (ggml's `QK_K` = 256 for the K-quants and IQ4_XS, 32 otherwise).
    pub const fn block_len(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q5_1 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K | Self::Iq4Xs => 256,
        }
    }

    /// Bytes per block, as `sizeof(block_*)` in ggml-common.h.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8_0 => 34,
            Self::Q5_1 => 24,
            Self::Iq4Xs => 136,
        }
    }

    /// Bytes per packed row of `columns` values, if `columns` is a whole number of blocks.
    pub fn row_bytes(self, columns: usize) -> Option<usize> {
        columns
            .is_multiple_of(self.block_len())
            .then(|| (columns / self.block_len()).checked_mul(self.block_bytes()))
            .flatten()
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MatmulError {
    #[error("{columns} columns are not a whole number of {block}-value blocks")]
    Columns { columns: usize, block: usize },
    #[error("weights contain {actual} elements, expected {expected}")]
    Weights { actual: usize, expected: usize },
    #[error("activations contain {actual} elements, expected {expected}")]
    Activations { actual: usize, expected: usize },
    #[error("output holds {actual} values, expected {expected}")]
    Output { actual: usize, expected: usize },
    #[error("{0:?} is not supported by this CPU")]
    UnsupportedIsa(Isa),
    #[error("matrix dimensions overflow usize")]
    Overflow,
}

/// Q8_0 activation rows, unpacked once for every matmul that consumes them.
///
/// Per 32-value group it keeps the f32 scale (the f16 scale converted exactly) and the exact sum
/// of the quants, which the formats with a minimum or offset need.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8Activations {
    columns: usize,
    tokens: usize,
    qs: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
}

impl Q8Activations {
    /// Quantizes `tokens x columns` f32 values exactly as ggml's `quantize_row_q8_0_ref`
    /// (`d = amax / 127` stored as f16, `q = round(x / d)` with the unrounded `d`).
    pub fn quantize(x: &[f32], columns: usize, tokens: usize) -> Result<Self, MatmulError> {
        let n = Self::shape(columns, tokens)?;
        if x.len() != n {
            return Err(MatmulError::Activations {
                actual: x.len(),
                expected: n,
            });
        }
        let groups = n / GROUP;
        let mut qs = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(groups);
        for block in x.as_chunks::<GROUP>().0 {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            scales.push(scalar::half_to_f32(scalar::f32_to_f16(d)));
            // `as i8` saturates; |v * id| <= 127 up to rounding, as in ggml.
            qs.extend(block.iter().map(|&v| (v * id).round() as i8));
        }
        Ok(Self::finish(columns, tokens, qs, scales))
    }

    /// Unpacks `tokens` rows of packed Q8_0 blocks (34 bytes per 32 values). Any byte values are
    /// accepted, including quants of -128.
    pub fn from_q8_0(packed: &[u8], columns: usize, tokens: usize) -> Result<Self, MatmulError> {
        let n = Self::shape(columns, tokens)?;
        let groups = n / GROUP;
        let expected = groups
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or(MatmulError::Overflow)?;
        if packed.len() != expected {
            return Err(MatmulError::Activations {
                actual: packed.len(),
                expected,
            });
        }
        let mut qs = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(groups);
        for block in packed.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
            scales.push(scalar::f16_at(block, 0));
            qs.extend(block[2..].iter().map(|&b| b as i8));
        }
        Ok(Self::finish(columns, tokens, qs, scales))
    }

    /// Packs back to ggml's Q8_0 layout (the inverse of [`Self::from_q8_0`]).
    pub fn to_q8_0(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.scales.len() * Q8_0_BLOCK_BYTES);
        for (q, &d) in self.qs.as_chunks::<GROUP>().0.iter().zip(&self.scales) {
            out.extend(scalar::f32_to_f16(d).to_le_bytes());
            out.extend(q.iter().map(|&v| v as u8));
        }
        out
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Returns `columns * tokens` after checking that `columns` is a whole number of groups.
    fn shape(columns: usize, tokens: usize) -> Result<usize, MatmulError> {
        if columns == 0 || !columns.is_multiple_of(GROUP) {
            return Err(MatmulError::Columns {
                columns,
                block: GROUP,
            });
        }
        columns.checked_mul(tokens).ok_or(MatmulError::Overflow)
    }

    fn finish(columns: usize, tokens: usize, qs: Vec<i8>, scales: Vec<f32>) -> Self {
        let sums = qs
            .as_chunks::<GROUP>()
            .0
            .iter()
            .map(|g| g.iter().map(|&v| i32::from(v)).sum())
            .collect();
        Self {
            columns,
            tokens,
            qs,
            scales,
            sums,
        }
    }

    /// Quants of token `t` (`columns` values). Panics if `t >= tokens`.
    pub(crate) fn qs(&self, t: usize) -> &[i8] {
        &self.qs[t * self.columns..][..self.columns]
    }

    /// Group scales of token `t` (`columns / 32` values).
    pub(crate) fn scales(&self, t: usize) -> &[f32] {
        let g = self.columns / GROUP;
        &self.scales[t * g..][..g]
    }

    /// Group quant sums of token `t` (`columns / 32` values).
    pub(crate) fn sums(&self, t: usize) -> &[i32] {
        let g = self.columns / GROUP;
        &self.sums[t * g..][..g]
    }
}

/// Multiplies packed `rows x columns` weights by `acts` (`tokens x columns`) into `out`
/// (`tokens x rows`, token-major), with the best kernels for this CPU.
pub fn matmul_q8(
    format: QuantFormat,
    weights: &[u8],
    acts: &Q8Activations,
    rows: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    matmul_q8_inner(dispatch::kernels(), format, weights, acts, rows, out)
}

/// As [`matmul_q8`] with a forced implementation (exactness checks, per-host tuning). Returns
/// [`MatmulError::UnsupportedIsa`] instead of running instructions the CPU lacks.
pub fn matmul_q8_with_isa(
    isa: Isa,
    format: QuantFormat,
    weights: &[u8],
    acts: &Q8Activations,
    rows: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    let kernels = dispatch::kernels_for(isa).ok_or(MatmulError::UnsupportedIsa(isa))?;
    matmul_q8_inner(kernels, format, weights, acts, rows, out)
}

fn matmul_q8_inner(
    kernels: Kernels,
    format: QuantFormat,
    weights: &[u8],
    acts: &Q8Activations,
    rows: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    let columns = acts.columns;
    let row_bytes = format.row_bytes(columns).ok_or(MatmulError::Columns {
        columns,
        block: format.block_len(),
    })?;
    let expected = rows.checked_mul(row_bytes).ok_or(MatmulError::Overflow)?;
    if weights.len() != expected {
        return Err(MatmulError::Weights {
            actual: weights.len(),
            expected,
        });
    }
    check_output(out, rows, acts.tokens)?;
    (kernels.quant(format))(weights, acts, rows, out);
    Ok(())
}

/// BF16 `rows x columns` weights times BF16 `tokens x columns` activations (raw bit patterns),
/// f32 accumulation, token-major output.
///
/// Products of two BF16 values are exact in f32; only the summation order differs between ISAs.
/// Each output is within `n * 2^-24 / (1 - n * 2^-24) * sum(|w_i * x_i|)` of the exact dot
/// (`n = columns`), plus, on the AVX-512 BF16 path, the flush of subnormal products and partial
/// sums to zero that `vdpbf16ps` performs.
pub fn matmul_bf16(
    weights: &[u16],
    acts: &[u16],
    rows: usize,
    columns: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    matmul_bf16_inner(dispatch::kernels(), weights, acts, rows, columns, out)
}

/// As [`matmul_bf16`] with a forced implementation.
pub fn matmul_bf16_with_isa(
    isa: Isa,
    weights: &[u16],
    acts: &[u16],
    rows: usize,
    columns: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    let kernels = dispatch::kernels_for(isa).ok_or(MatmulError::UnsupportedIsa(isa))?;
    matmul_bf16_inner(kernels, weights, acts, rows, columns, out)
}

fn matmul_bf16_inner(
    kernels: Kernels,
    weights: &[u16],
    acts: &[u16],
    rows: usize,
    columns: usize,
    out: &mut [f32],
) -> Result<(), MatmulError> {
    if columns == 0 {
        return Err(MatmulError::Columns { columns, block: 1 });
    }
    let expected = rows.checked_mul(columns).ok_or(MatmulError::Overflow)?;
    if weights.len() != expected {
        return Err(MatmulError::Weights {
            actual: weights.len(),
            expected,
        });
    }
    if !acts.len().is_multiple_of(columns) {
        return Err(MatmulError::Activations {
            actual: acts.len(),
            expected: acts.len().next_multiple_of(columns),
        });
    }
    let tokens = acts.len() / columns;
    check_output(out, rows, tokens)?;
    for (t, x) in acts.chunks_exact(columns).enumerate() {
        for (r, w) in weights.chunks_exact(columns).enumerate() {
            out[t * rows + r] = (kernels.bf16_dot)(w, x);
        }
    }
    Ok(())
}

fn check_output(out: &[f32], rows: usize, tokens: usize) -> Result<(), MatmulError> {
    let expected = rows.checked_mul(tokens).ok_or(MatmulError::Overflow)?;
    if out.len() != expected {
        return Err(MatmulError::Output {
            actual: out.len(),
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;
