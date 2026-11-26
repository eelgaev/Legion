//! Ops for vision encoders (`crate::vision`): the patch convolution, the resampling of learned
//! position embeddings and an unmasked scaled softmax. Thin wrappers, as in `ops.rs`.

use super::context::{Context, Tensor};
use super::ffi;

/// `GGML_SCALE_MODE_BILINEAR`.
pub const SCALE_BILINEAR: u32 = 1;
/// `GGML_SCALE_FLAG_ALIGN_CORNERS`: the corner samples of input and output line up.
pub const SCALE_ALIGN_CORNERS: u32 = 1 << 8;
/// `GGML_ROPE_TYPE_VISION`: a head's dimensions rotate by two positions (rows, then columns).
pub const ROPE_VISION: i32 = 24;

impl Context {
    /// 2D convolution of `b` (`[w, h, c_in, n]`) with kernel `a` (`[kw, kh, c_in, c_out]`):
    /// `[w', h', c_out, n]`, at strides `s`, padding `p` and dilation `d` (x, then y).
    pub fn conv_2d<'a>(
        &'a self,
        a: Tensor<'a>,
        b: Tensor<'a>,
        s: [i32; 2],
        p: [i32; 2],
        d: [i32; 2],
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_conv_2d(
                self.raw(),
                a.raw(),
                b.raw(),
                s[0],
                s[1],
                p[0],
                p[1],
                d[0],
                d[1],
            )
        })
    }

    /// `a` resampled to `ne` with `mode` ([`SCALE_BILINEAR`], optionally with
    /// [`SCALE_ALIGN_CORNERS`]).
    pub fn interpolate<'a>(&'a self, a: Tensor<'a>, ne: [i64; 4], mode: u32) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_interpolate(self.raw(), a.raw(), ne[0], ne[1], ne[2], ne[3], mode)
        })
    }

    /// `silu(min(a, limit)) * clamp(b, -limit, limit)` (`ggml_swiglu_clamp`): GLM-5.3-Flash's
    /// vision MLPs.
    pub fn swiglu_clamp<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>, limit: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_swiglu_clamp(self.raw(), a.raw(), b.raw(), limit) })
    }

    /// GELU, exact (`ggml_gelu_erf`).
    pub fn gelu_erf<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_gelu_erf(self.raw(), a.raw()) })
    }

    /// `soft_max(a * scale)` over rows, with no mask.
    pub fn soft_max_scaled<'a>(&'a self, a: Tensor<'a>, scale: f32) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_soft_max_ext(self.raw(), a.raw(), std::ptr::null_mut(), scale, 0.0)
        })
    }

    /// The RoPE node `a` (a `rope_ext` result) rotating the dimensions from `offset` on
    /// instead of from 0 (`ggml_rope_set_offset`); `a` itself.
    pub fn rope_offset<'a>(&'a self, a: Tensor<'a>, offset: i32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_rope_set_offset(a.raw(), offset) })
    }

    /// Average pooling of `a` (`[w, h, c, n]`) over `k × k` windows at stride `k`, no padding.
    pub fn avg_pool_2d<'a>(&'a self, a: Tensor<'a>, k: i32) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_pool_2d(
                self.raw(),
                a.raw(),
                ffi::ggml_op_pool_GGML_OP_POOL_AVG,
                k,
                k,
                k,
                k,
                0.0,
                0.0,
            )
        })
    }

    /// `a` made contiguous as `[ne0, ne1, ne2, ne3]`.
    pub fn cont_4d<'a>(&'a self, a: Tensor<'a>, ne: [i64; 4]) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_cont_4d(self.raw(), a.raw(), ne[0], ne[1], ne[2], ne[3]) })
    }
}
