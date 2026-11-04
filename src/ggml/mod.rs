//! ggml bindings. `ffi` is the raw bindgen output; the rest is a small safe layer over it:
//! [`Device`]/[`Backend`] to compute with, [`Context`]/[`Tensor`]/[`Graph`] to describe
//! computations, and quantization helpers.

mod backend;
mod context;
mod external;
mod graph_uid;
pub mod input_audit;
mod keep;
pub mod meta;
pub mod model_bytes;
#[cfg(feature = "cuda")]
mod op_profile;
mod ops;
mod ops_gemma4;
mod ops_hc;
mod ops_mamba;
mod ops_vision;
mod padded;
pub mod pinned_io;
pub mod post_alloc;
pub mod stage_graph;
mod uma_memory;

pub use backend::{
    Backend, BufferType, Device, DeviceKind, Event, LateInput, Launcher, backend_by_name,
    device_by_kind, devices, gpu_device,
};
pub use context::{
    ComputeAllocator, Context, Graph, KeptGraph, KeptParts, SharedAllocator, Tensor, TensorSlot,
    mapped_tail_padding, set_fresh_plans,
};
#[cfg(feature = "cuda")]
pub(crate) use external::cuda::current_backend as current_cuda_backend;
#[cfg(feature = "cuda")]
pub(crate) use external::cuda::current_graph as current_cuda_graph;
pub use external::{EXTERNAL_PARAMS, ExternalOp, external_params};
pub(crate) use external::{make_external, op_of};
pub use ops_gemma4::ROPE_NEOX;
pub use ops_vision::{ROPE_VISION, SCALE_ALIGN_CORNERS, SCALE_BILINEAR};

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unsafe_op_in_unsafe_fn,
    unnecessary_transmutes,
    clippy::all
)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ggml_bindings.rs"));
}

/// Tensor types, re-exported under short names.
pub mod types {
    use super::ffi;
    pub const F32: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_F32;
    pub const F16: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_F16;
    pub const BF16: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_BF16;
    /// Also the carrier of raw FP8 E4M3 weights (`crate::model::fp8`).
    pub const I8: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_I8;
    pub const I32: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_I32;
    pub const Q8_0: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q8_0;
    pub const Q4_0: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q4_0;
    pub const Q4_K: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q4_K;
    pub const Q5_K: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q5_K;
    pub const Q6_K: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q6_K;
    pub const Q5_1: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q5_1;
    pub const IQ4_NL: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ4_NL;
    pub const IQ2_XXS: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ2_XXS;
    pub const IQ2_S: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ2_S;
    pub const IQ2_XS: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ2_XS;
    pub const IQ1_M: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ1_M;
    pub const Q2_0: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q2_0;
    // Project Maya's GLM-5.3-Flash formats (#559).
    pub const Q2_K: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q2_K;
    pub const Q3_K: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_Q3_K;
    pub const IQ3_XXS: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ3_XXS;
    pub const IQ3_S: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ3_S;
    pub const IQ4_XS: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_IQ4_XS;
    pub const I64: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_I64;
    pub const NVFP4: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_NVFP4;
    /// OCP MXFP4: DeepSeek-V4.1-Flash's FP4 routed experts, repacked (`crate::hf`).
    pub const MXFP4: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_MXFP4;
}

#[derive(Debug, thiserror::Error)]
pub enum GgmlError {
    #[error("could not start ggml backend `{0}`")]
    Init(String),
    #[error("could not allocate {bytes} bytes in `{buffer}`")]
    Alloc { bytes: usize, buffer: String },
    #[error("could not map the file: {0}")]
    Map(String),
    #[error("graph compute failed (ggml status {0})")]
    Compute(i32),
    #[error("this context already allocated a graph; build the next one in a new context")]
    GraphAlreadyAllocated,
    #[error("the graph belongs to another context")]
    ForeignGraph,
    #[error("external op failed: {0}")]
    External(String),
}

/// Name of a ggml tensor type ("f32", "q4_K", ...).
pub fn type_name(t: ffi::ggml_type) -> &'static str {
    // SAFETY: ggml_type_name returns a pointer into a static table.
    unsafe {
        let p = ffi::ggml_type_name(t);
        if p.is_null() {
            return "unknown";
        }
        std::ffi::CStr::from_ptr(p).to_str().unwrap_or("unknown")
    }
}

/// Bytes for `n` elements of `ty` in one row (`n` must be a multiple of the block size).
pub fn row_size(ty: ffi::ggml_type, n: usize) -> usize {
    unsafe { ffi::ggml_row_size(ty, n as i64) }
}

/// Elements per block of `ty` (1 for plain types).
pub fn block_size(ty: ffi::ggml_type) -> usize {
    unsafe { ffi::ggml_blck_size(ty) as usize }
}

/// Quantizes `rows × n_per_row` f32 values to `ty` (CPU, reference quality).
pub fn quantize(ty: ffi::ggml_type, src: &[f32], n_per_row: usize) -> Vec<u8> {
    assert!(n_per_row > 0 && src.len().is_multiple_of(n_per_row));
    let rows = src.len() / n_per_row;
    let mut out = vec![0u8; row_size(ty, n_per_row) * rows];
    // SAFETY: out holds exactly rows × row_size bytes; no importance matrix.
    let written = unsafe {
        ffi::ggml_quantize_chunk(
            ty,
            src.as_ptr(),
            out.as_mut_ptr().cast(),
            0,
            rows as i64,
            n_per_row as i64,
            std::ptr::null(),
        )
    };
    assert_eq!(written, out.len());
    out
}

/// Whether [`dequantize`] takes `ty`.
pub fn can_dequantize(ty: ffi::ggml_type) -> bool {
    // SAFETY: type traits are static.
    unsafe { (*ffi::ggml_get_type_traits(ty)).to_float.is_some() }
}

/// Converts `n` elements of `ty` back to f32 (any type ggml can convert, quantized or not).
pub fn dequantize(ty: ffi::ggml_type, src: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(
        src.len(),
        row_size(ty, n),
        "size of {n} {} elements",
        type_name(ty)
    );
    let mut out = vec![0f32; n];
    // SAFETY: type traits are static; to_float reads row_size(ty, n) bytes and writes n floats.
    unsafe {
        let traits = ffi::ggml_get_type_traits(ty);
        let to_float = (*traits).to_float.expect("type has no to_float");
        to_float(src.as_ptr().cast(), out.as_mut_ptr(), n as i64);
    }
    out
}

/// Number of CUDA devices ggml sees (0 without the `cuda` feature).
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        // SAFETY: plain query, no arguments.
        unsafe { ffi::ggml_backend_cuda_get_device_count().max(0) as usize }
    }
    #[cfg(not(feature = "cuda"))]
    0
}
