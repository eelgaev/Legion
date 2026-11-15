//! The tensor-core NVFP4 dense matvec op (`legion::model::core::nvfp4_mvq`, #1133) against
//! ggml's quantize_q8_1 + `mul_mat_vec_q` (ignored, `--features cuda`): every output of 1-8
//! columns on Qwen3.8-27B's FFN shapes, with and without the one-value scale, has ggml's bits.
//! The timed version is `legion nvfp4-mvq-bench`.
//!
//! `cargo test --release --features cuda --test nvfp4_mvq -- --include-ignored --nocapture --test-threads 1`

#[cfg(feature = "cuda")]
mod cuda {
    use legion::ggml::{Backend, DeviceKind, device_by_kind};
    use legion::model::core::nvfp4_mvq::bench;

    fn gpu() -> Option<Backend> {
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU")
            .init()
            .unwrap();
        if !legion::model::nvfp4_a4::device_supported() {
            println!("skipped: the op needs the sm_120a/sm_121a FP4 conversions on this GPU");
            return None;
        }
        Some(gpu)
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_op_has_ggmls_mmvq_bits() {
        let Some(mut gpu) = gpu() else { return };
        for scale in [true, false] {
            let cfg = bench::Config {
                layers: 2,
                rounds: 0,
                scale,
                ..bench::Config::default()
            };
            let rows = bench::run(&mut gpu, &cfg).unwrap();
            for r in &rows {
                println!(
                    "scale {scale} k {} rows {} cols {} {}: {} of {} differ, first {:?}",
                    r.k, r.rows, r.cols, r.variant, r.differ, r.outputs, r.first
                );
            }
            assert!(rows.iter().all(|r| r.differ == 0 && r.outputs > 0));
        }
    }
}
