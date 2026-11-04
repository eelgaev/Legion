//! One quantization of a layer input that several Q8_0 projections read, prompt steps on a GB10
//! (#526). ggml-cuda's `mul_mat` quantizes its F32 input for MMQ (`quantize_mmq_q8_1`) in every
//! matmul, so the same input is quantized once per projection, to the same bytes each time: a
//! DeltaNet layer's input twice (`attn_qkv`, `attn_gate`), an attention layer's three times
//! (`attn_q`, `attn_k`, `attn_v`) and an FFN's input twice (`ffn_gate_shexp`, `ffn_up_shexp`).
//!
//! Here an external op ([`crate::ggml::ExternalOp`]) runs ggml's own quantization kernel once,
//! through the glue in `src/cuda/mmq_preq.cu` (`legion_ggml_quantize_mmq_q8_0`, called as
//! `ggml_cuda_mul_mat_q` calls it for a contiguous input), and every projection runs ggml's MMQ
//! on those blocks with ggml's arguments ([`mmq_preq::mul_mat`]): ggml's kernels, ggml's bits.
//!
//! Chosen where [`mmq_preq::fits`] (compute capability 12.x, 256+ tokens) for at least two of an
//! input's projections; `legion serve --shared-quant-ggml` keeps ggml's quantization in each
//! matmul.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::mmq_preq;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static SHARED_Q8: ExternalOp = ExternalOp {
    name: "shared_q8",
    cpu: cpu_shared_q8,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_shared_q8),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs built from now on may share an input's quantization (`legion serve
/// --shared-quant-ggml` turns it off for comparisons).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on quantize an input that several Q8_0 projections read
/// once for all of them where it fits (on by default); `false` keeps ggml's quantization in
/// each matmul.
pub fn set_shared_q8(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Fewest projections an input is quantized for: with one, the op would only move ggml's pass.
pub(crate) const MIN_USERS: usize = 2;

/// An input `x` (F32 `[k, n]`) and its blocks in MMQ's layout ([`mmq_preq::q8_1_bytes`]), for
/// the Q8_0 matmuls that read it ([`Self::mul_mat`]): a node of the graph, or (`None`, steps of
/// [`small::MIN_TOKENS`] to [`small::MAX_TOKENS`] tokens) the scratch buffer [`small`]'s nodes
/// share.
#[derive(Clone, Copy)]
pub(crate) struct Shared<'a> {
    x: Tensor<'a>,
    xq: Option<Tensor<'a>>,
}

pub(crate) mod small;

/// Whether [`quantize`] takes `x`: switched on, a contiguous F32 `[k, n]` with `k` a multiple of
/// 512 (ggml-cuda's row padding: its blocks are then exactly `x`'s).
pub(crate) fn input_fits(x: Tensor<'_>) -> bool {
    let [k, _, n2, n3] = x.ne();
    ON.load(Ordering::Relaxed)
        && x.ty() == types::F32
        && x.is_contiguous()
        && k % 512 == 0
        && n2 == 1
        && n3 == 1
}

/// `x` (see [`input_fits`]) quantized as ggml-cuda's `mul_mat` quantizes it for MMQ.
pub(crate) fn quantize<'a>(ctx: &'a Context, x: Tensor<'a>) -> Shared<'a> {
    assert!(input_fits(x), "shared_q8: x {:?}", x.ne());
    Shared {
        x,
        xq: Some(blocks(ctx, x)),
    }
}

/// `x` (see [`small::takes_input`]) shared by its projections in a small step: each reads the
/// blocks the first of them writes to the scratch buffer ([`small::mul_mat`]).
pub(crate) fn small<'a>(x: Tensor<'a>) -> Shared<'a> {
    assert!(small::input_fits(x), "shared_q8: x {:?}", x.ne());
    Shared { x, xq: None }
}

/// `x`'s blocks as [`quantize`] makes them, whether or not sharing is switched on: for other ops
/// that read MMQ's layout ([`super::q8_mv`]). `x` a contiguous F32 `[k, n]`, `k` a multiple of
/// 512.
pub(crate) fn blocks<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let [k, n, n2, n3] = x.ne();
    assert!(
        x.ty() == types::F32 && x.is_contiguous() && k % 512 == 0 && n2 == 1 && n3 == 1,
        "shared_q8: x {:?}",
        x.ne()
    );
    let floats = mmq_preq::q8_1_bytes(k, n).div_ceil(4) as i64;
    ctx.external(&SHARED_Q8, types::F32, [floats, 1, 1, 1], &[x])
}

impl<'a> Shared<'a> {
    /// The number of tokens.
    pub(crate) fn n(&self) -> i64 {
        self.x.ne()[1]
    }

    /// Whether [`Self::mul_mat`] takes `w`: a weight and step [`mmq_preq::fits`] takes, with
    /// rows of `x`'s length.
    pub(crate) fn takes(&self, w: Tensor<'_>) -> bool {
        let k = self.x.ne()[0];
        match self.xq {
            Some(_) => w.ne()[0] == k && mmq_preq::fits(w, self.n()),
            None => small::takes_weight(w, self.x),
        }
    }

    /// `mul_mat(w, x)` (F32 `[rows, n]`) for `w` Q8_0 `[k, rows]` (see [`Self::takes`]) from the
    /// shared blocks: ggml's bytes.
    pub(crate) fn mul_mat(&self, ctx: &'a Context, w: Tensor<'a>) -> Tensor<'a> {
        match self.xq {
            Some(xq) => mmq_preq::mul_mat(ctx, w, xq, self.x),
            None => small::mul_mat(ctx, w, self.x),
        }
    }
}

unsafe fn cpu_shared_q8(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // SAFETY: a node made by `quantize`: a contiguous host `x`, the blocks' bytes behind `dst`.
    unsafe {
        if ith != 0 {
            return;
        }
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let xs = std::slice::from_raw_parts(x.data as *const f32, k * n);
        let bytes = mmq_preq::q8_1_bytes(k as i64, n as i64);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut u8, bytes);
        mmq_preq::quantize_ref(xs, k, n, out);
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_shared_q8(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `quantize`: `x` contiguous in the stream's device memory, room for
    // its blocks behind `dst`.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0], x.ne[1]);
        if ffi::ggml_nbytes(dst) < mmq_preq::q8_1_bytes(k, n) {
            return Err("shared_q8: output short".into());
        }
        match mmq_preq::legion_ggml_quantize_mmq_q8_0(x.data as *const f32, dst.data, k, n, stream)
        {
            0 => Ok(()),
            e => Err(format!("shared_q8: error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
