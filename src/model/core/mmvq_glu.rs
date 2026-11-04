//! NVFP4 routed experts at decode and verify steps (#836): gate, up and their SwiGLU built as
//! the chain ggml-cuda fuses into one MMVQ launch.
//!
//! Steps of up to 8 tokens run the NVFP4 experts on ggml's `mul_mat_vec_q` (MoE variant), each
//! matmul times its expert's scale (`mul_mat_id`, then `reshape`, `repeat`, `get_rows`, `mul`).
//! Built as `mul(silu(gate), up)`, that is two matvec launches (each quantizing `x` to Q8_1),
//! then a SiLU and a multiply. Built as `swiglu_split(gate, up)` with the gate chain first, the
//! nodes match ggml-cuda's gate + up + GLU fusion (`ggml_cuda_should_fuse_mul_mat`): one launch
//! computes both rows from one quantized `x` and writes `up * silu(gate)`, each product scaled
//! as before, with the SiLU of `ggml_cuda_op_silu_single`: the bits of the four launches it
//! replaces (FP32 products commute). Where ggml does not fuse, its SwiGLU kernel computes the
//! same `silu(gate) * up`.
//!
//! Opt-in (`legion serve --nvfp4-mmvq-glu`; CUDA, experts in device memory). On a GB10 the
//! fused launch is slower than the two it replaces (two accumulators per thread: about 0.42 ms
//! against 2 x 0.17 ms per layer at 6-7 verify rows), so C=1 decode with MTP stays flat.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Tensor, ffi, types};

#[cfg(test)]
mod tests;

/// Largest step (tokens) that takes the chain: ggml's NVFP4 MMVQ bound (`NVFP4_GGML_MAX_N`);
/// bigger steps take Legion's grouped NVFP4 SwiGLU.
pub(crate) const MAX_TOKENS: i64 = 8;

static ON: AtomicBool = AtomicBool::new(false);

/// `legion serve --nvfp4-mmvq-glu` turns the fused chain on (`true`).
pub fn set_mmvq_glu(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// True if a step of `n_tokens` builds its NVFP4 gate and up experts as the fusable chain: both
/// NVFP4 of one shape and layout, in device memory (CUDA builds only).
pub(crate) fn takes(gate: Tensor<'_>, up: Tensor<'_>, n_tokens: i64) -> bool {
    on() && cfg!(feature = "cuda") && fits(gate, up, n_tokens)
}

/// [`takes`] without the switch and the build: the tensors' side of it.
pub(crate) fn fits(gate: Tensor<'_>, up: Tensor<'_>, n_tokens: i64) -> bool {
    pair_fits(
        (gate.ty(), gate.ne(), gate.nb()),
        (up.ty(), up.ne(), up.nb()),
        n_tokens,
    ) && on_device(gate)
        && on_device(up)
}

/// Type, dims and strides of an expert weight.
pub(crate) type Layout = (ffi::ggml_type, [i64; 4], [usize; 4]);

/// Both NVFP4, of one shape and layout, for a step of 1 to [`MAX_TOKENS`] tokens.
pub(crate) fn pair_fits(gate: Layout, up: Layout, n_tokens: i64) -> bool {
    (1..=MAX_TOKENS).contains(&n_tokens) && gate.0 == types::NVFP4 && gate == up
}

/// Allocated outside host memory (a device buffer).
fn on_device(t: Tensor<'_>) -> bool {
    // SAFETY: reads the tensor's buffer pointer.
    let allocated = unsafe { !(*t.raw()).buffer.is_null() };
    allocated && !t.is_host()
}
