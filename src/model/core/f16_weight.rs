//! A Q8_0 weight dequantized to F16 by ggml-cuda's own converter, as an external op
//! ([`crate::ggml::ExternalOp`]), for a matmul whose input an op already wrote as F16 (#774).
//!
//! On a V100 ggml-cuda runs a Q8_0 `mul_mat` of 64 or more tokens as one cuBLAS F16 GEMM with
//! F32 output (`ggml_cuda_mul_mat_cublas`): it dequantizes the weight to F16
//! (`ggml_get_to_fp16_cuda(Q8_0)`), rounds the F32 input to F16, then calls `cublasGemmEx`. An
//! F16 input on a Q8_0 weight is not a `mul_mat` ggml-cuda takes, but F16 on F16 is, and runs
//! that same GEMM without either conversion. So `mul_mat(dequant(w), x16)`, with `x16` the
//! input's F16 rounding written by the op that makes the input, is ggml's `mul_mat(w, x)` bit for
//! bit: the weight's conversion moves into this op (the same kernel), the input's per-matmul
//! conversion is gone. See [`super::linear::f16_joined`] for where it is used.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static F16_WEIGHT: ExternalOp = ExternalOp {
    name: "f16_weight",
    cpu: cpu_f16_weight,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_f16_weight),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// `w` (Q8_0 `[k, m]`, contiguous) as F16 `[k, m]`, each value `d · q` rounded to F16 as
/// ggml-cuda's converter rounds it.
pub(crate) fn dequant<'a>(ctx: &'a Context, w: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(w.ty(), types::Q8_0, "f16_weight: Q8_0 weights");
    assert!(w.is_contiguous(), "f16_weight: contiguous weights");
    let [k, m, m2, m3] = w.ne();
    assert_eq!([m2, m3], [1, 1], "f16_weight: 2D weights");
    ctx.external(&F16_WEIGHT, types::F16, [k, m, 1, 1], &[w])
}

unsafe fn cpu_f16_weight(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `dequant`, host tensors. Q8_0 blocks of 32: an F16 scale, then 32
    // int8 values; d · q is exact in F32, so one rounding to F16.
    unsafe {
        const QK: usize = 32;
        const BLOCK: usize = 2 + QK;
        let w = &*dst.src[0];
        let blocks = (w.ne[0] * w.ne[1]) as usize / QK;
        let (q8, out) = (w.data as *const u8, dst.data as *mut u16);
        for b in (ith..blocks).step_by(nth) {
            let block = q8.add(b * BLOCK);
            let d = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*block, *block.add(1)]));
            for j in 0..QK {
                let q = *(block.add(2 + j) as *const i8);
                *out.add(b * QK + j) = ffi::ggml_fp32_to_fp16(f32::from(q) * d);
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// ggml-cuda's `to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type)` (`ggml-cuda/convert.cuh`,
    /// a C++ symbol; `--wrap`ped as for ggml's own calls, see `build/cuda_wrap.rs`).
    #[link_name = "_Z21ggml_get_to_fp16_cuda9ggml_type"]
    fn ggml_get_to_fp16_cuda(ty: ffi::ggml_type) -> Option<ToF16>;
}

/// ggml-cuda's `to_fp16_cuda_t`: converts `k` values at `x` to F16 at `y` on `stream`.
#[cfg(feature = "cuda")]
type ToF16 = unsafe extern "C" fn(x: *const c_void, y: *mut u16, k: i64, stream: *mut c_void);

#[cfg(feature = "cuda")]
unsafe fn cuda_f16_weight(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `dequant`, device tensors of the stream's device.
    unsafe {
        let w = &*dst.src[0];
        let Some(to_f16) = ggml_get_to_fp16_cuda(w.type_) else {
            return Err("f16_weight: no F16 conversion".into());
        };
        to_f16(w.data, dst.data as *mut u16, w.ne[0] * w.ne[1], stream);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
