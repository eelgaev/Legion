//! CUDA symbols Legion replaces without modifying the pinned ggml submodule.

/// C++ ABI symbols of ggml's contiguous CUDA conversion dispatchers.
///
/// The functions are declared in `ggml-cuda/convert.cuh`; GNU ld's `--wrap` sends every
/// reference from ggml-cuda through the implementations in `src/cuda/dequant.cu`.
pub const DEQUANT_SYMBOLS: [&str; 3] = [
    "_Z21ggml_get_to_fp16_cuda9ggml_type",
    "_Z21ggml_get_to_bf16_cuda9ggml_type",
    "_Z21ggml_get_to_fp32_cuda9ggml_type",
];

pub fn link_args() -> impl Iterator<Item = String> {
    DEQUANT_SYMBOLS
        .into_iter()
        .map(|symbol| format!("-Wl,--wrap={symbol}"))
}
