//! KV cache element types, chosen separately for K and V (`--cache-type-k` / `--cache-type-v`,
//! llama.cpp's names). A cache row (one position of one layer, `head_dim × n_head_kv` values)
//! is stored in the type's ggml format: the step writes it with `set_rows` (which quantizes),
//! and the gather attention path dequantizes the rows it reads (`get_rows` of whole blocks to
//! F32, then F16 for the matmuls, as llama.cpp's attention reads a quantized cache). The paged
//! kernels read q8_0 and q4_0 on sm_80+ (`paged::reads_types`), qwen4exp's sparse attention
//! kernels on every GPU when K and V are both (`qsa_attn::reads_types`, #263, the values the
//! gather path reads); other caches take the gather path.
//!
//! `tq3` / `tq4` are TurboQuant caches ([`super::turboquant`], arXiv 2504.19874): not ggml
//! types. Their cache tensors are raw bytes (I8 rows of [`super::turboquant::cell_bytes`] per
//! cell), written and read by Legion's own kernels only.

use crate::ggml::{ffi, types};

/// One cache's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum KvType {
    #[default]
    F16,
    Bf16,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "iq4_nl")]
    Iq4Nl,
    /// TurboQuant (arXiv 2504.19874), 3 bits per value + an F16 norm per head; head_dim a
    /// power of two.
    #[value(name = "tq3")]
    Tq3,
    /// TurboQuant (arXiv 2504.19874), 4 bits per value + an F16 norm per head; head_dim a
    /// power of two.
    #[value(name = "tq4")]
    Tq4,
}

impl KvType {
    /// The cache tensor's ggml type: raw bytes (I8) for the TurboQuant types.
    pub fn ggml(self) -> ffi::ggml_type {
        match self {
            KvType::F16 => types::F16,
            KvType::Bf16 => types::BF16,
            KvType::Q8_0 => types::Q8_0,
            KvType::Q4_0 => ffi::ggml_type_GGML_TYPE_Q4_0,
            KvType::Q4_1 => ffi::ggml_type_GGML_TYPE_Q4_1,
            KvType::Q5_0 => ffi::ggml_type_GGML_TYPE_Q5_0,
            KvType::Q5_1 => types::Q5_1,
            KvType::Iq4Nl => types::IQ4_NL,
            KvType::Tq3 | KvType::Tq4 => ffi::ggml_type_GGML_TYPE_I8,
        }
    }

    /// The TurboQuant bits per value, `None` for a ggml type.
    pub fn tq_bits(self) -> Option<u32> {
        match self {
            KvType::Tq3 => Some(3),
            KvType::Tq4 => Some(4),
            _ => None,
        }
    }

    /// Bytes of a cache row of `n` values in heads of `d` (`n` a multiple of the type's block
    /// size, or of `d` for the TurboQuant types).
    pub fn bytes(self, d: usize, n: usize) -> usize {
        match self.tq_bits() {
            Some(bits) => super::turboquant::cell_bytes(d, n / d.max(1), bits),
            None => crate::ggml::row_size(self.ggml(), n),
        }
    }

    /// The cache tensor's dim 0 for rows of `n` values in heads of `d`: `n`, or the bytes of a
    /// TurboQuant cell (its I8 tensor).
    pub fn row_ne0(self, d: usize, n: usize) -> usize {
        if self.tq_bits().is_some() {
            self.bytes(d, n)
        } else {
            n
        }
    }

    /// Whether `n`-wide rows of heads of `d` can be stored (a whole number of the type's
    /// blocks; TurboQuant: whole heads of a power-of-two size, [`super::turboquant::head_ok`]).
    pub fn fits(self, d: usize, n: usize) -> bool {
        if self.tq_bits().is_some() {
            return super::turboquant::head_ok(d) && n.is_multiple_of(d);
        }
        // SAFETY: a pure lookup.
        let block = unsafe { ffi::ggml_blck_size(self.ggml()) } as usize;
        n.is_multiple_of(block)
    }
}

/// The K and V caches' types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvTypes {
    pub k: KvType,
    pub v: KvType,
}

impl KvTypes {
    /// Both F16 (the paged kernels' type).
    pub fn is_f16(&self) -> bool {
        self.k == KvType::F16 && self.v == KvType::F16
    }

    /// Bytes of one position of a cache row `row` wide in heads of `d`, K plus V.
    pub fn row_bytes(&self, d: usize, row: usize) -> usize {
        self.k.bytes(d, row) + self.v.bytes(d, row)
    }

    /// Whether K or V is a TurboQuant cache.
    pub fn has_tq(&self) -> bool {
        self.k.tq_bits().is_some() || self.v.tq_bits().is_some()
    }
}

#[cfg(test)]
mod tests;
