//! Ops for Gemma 4: NEOX RoPE with per-dimension frequency factors, the GELU-gated FFN and
//! tanh (final logit soft-capping). Thin wrappers, as in `ops.rs`.

use super::context::{Context, Tensor};
use super::ffi;

/// `GGML_ROPE_TYPE_NEOX`: the two halves of each head rotate together.
pub const ROPE_NEOX: i32 = 2;

impl Context {
    /// RoPE of `a` (`[head_dim, heads, n]`) at positions `pos` (I32 `[n]`), rotating the first
    /// `n_dims` dimensions. `freq_factors` (F32 `[n_dims / 2]`) divide the per-pair frequencies
    /// (`ggml_rope_ext`'s `c`). No YaRN: the extension factor is 0.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_ext<'a>(
        &'a self,
        a: Tensor<'a>,
        pos: Tensor<'a>,
        freq_factors: Option<Tensor<'a>>,
        n_dims: i32,
        mode: i32,
        n_ctx_orig: i32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Tensor<'a> {
        let c = freq_factors.map_or(std::ptr::null_mut(), |t| t.raw());
        self.wrap(unsafe {
            ffi::ggml_rope_ext(
                self.raw(),
                a.raw(),
                pos.raw(),
                c,
                n_dims,
                mode,
                n_ctx_orig,
                freq_base,
                freq_scale,
                0.0,
                1.0,
                32.0,
                1.0,
            )
        })
    }

    /// `gelu(a) * b` (tanh approximation), one fused op as llama.cpp's `build_ffn` runs a
    /// parallel GELU gate. `a` and `b` have the same shape.
    pub fn geglu_split<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_geglu_split(self.raw(), a.raw(), b.raw()) })
    }

    /// GELU, tanh approximation (`ggml_gelu`).
    pub fn gelu<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_gelu(self.raw(), a.raw()) })
    }

    pub fn tanh<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_tanh(self.raw(), a.raw()) })
    }

    /// The first `ne0` elements of `a` from byte `offset`.
    pub fn view_1d<'a>(&'a self, a: Tensor<'a>, ne0: i64, offset: usize) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_view_1d(self.raw(), a.raw(), ne0, offset) })
    }
}
