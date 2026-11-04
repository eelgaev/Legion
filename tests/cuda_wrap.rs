//! The build-time symbol interposition that installs Legion's dequantizers.

#[path = "../build/cuda_wrap.rs"]
mod cuda_wrap;

#[test]
fn wraps_all_contiguous_dequant_dispatchers() {
    assert_eq!(
        cuda_wrap::link_args().collect::<Vec<_>>(),
        [
            "-Wl,--wrap=_Z21ggml_get_to_fp16_cuda9ggml_type",
            "-Wl,--wrap=_Z21ggml_get_to_bf16_cuda9ggml_type",
            "-Wl,--wrap=_Z21ggml_get_to_fp32_cuda9ggml_type",
        ]
    );
}
