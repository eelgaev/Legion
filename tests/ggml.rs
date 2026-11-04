#[test]
#[cfg_attr(
    feature = "cuda",
    ignore = "touches the GPU: run with --ignored when allowed"
)]
fn cuda_devices_match_feature() {
    let n = legion::ggml::cuda_device_count();
    if cfg!(feature = "cuda") {
        // Needs a GPU on the test machine.
        assert!(n > 0, "built with cuda but ggml sees no CUDA device");
    } else {
        assert_eq!(n, 0);
    }
}
