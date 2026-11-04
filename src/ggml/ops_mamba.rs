//! Ops for the Mamba-2 hybrids (Nemotron-H): the selective scan, layer norm, the squared-ReLU and
//! SwiGLU activations. Thin wrappers, as in `ops.rs`.

use super::context::{Context, Tensor};
use super::ffi;

impl Context {
    pub fn relu<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_relu(self.raw(), a.raw()) })
    }

    /// Layer norm without weight or bias: `(a - mean) / sqrt(var + eps)` over dim 0.
    pub fn norm<'a>(&'a self, a: Tensor<'a>, eps: f32) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_norm(self.raw(), a.raw(), eps) })
    }

    pub fn sqr<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_sqr(self.raw(), a.raw()) })
    }

    /// `silu(a) * b`, one fused op (not the same rounding as `mul(silu(a), b)` on every
    /// backend). `a` and `b` have the same shape.
    pub fn swiglu_split<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        self.wrap(unsafe { ffi::ggml_swiglu_split(self.raw(), a.raw(), b.raw()) })
    }

    /// Mamba-2 selective scan over a chunk of tokens. `s`: the states `[d_state, head_dim,
    /// n_head, slots]` (contiguous), read at the slots `ids` (I32 `[n_seqs]`); `x`
    /// `[head_dim, n_head, n_tok, n_seqs]`; `dt` `[n_head, n_tok, n_seqs]` (before softplus);
    /// `a` `[1, n_head]`; `b`, `c` `[d_state, n_group, n_tok, n_seqs]`. Result (F32, 1-D):
    /// every token's `y` (`x`'s shape), then `snapshots` × `n_seqs` states (the first set is the
    /// final state; the others the states before the last 1.., for rollback).
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_scan<'a>(
        &'a self,
        s: Tensor<'a>,
        x: Tensor<'a>,
        dt: Tensor<'a>,
        a: Tensor<'a>,
        b: Tensor<'a>,
        c: Tensor<'a>,
        ids: Tensor<'a>,
        snapshots: i64,
    ) -> Tensor<'a> {
        self.wrap(unsafe {
            ffi::ggml_ssm_scan(
                self.raw(),
                s.raw(),
                x.raw(),
                dt.raw(),
                a.raw(),
                b.raw(),
                c.raw(),
                ids.raw(),
                snapshots,
            )
        })
    }
}
