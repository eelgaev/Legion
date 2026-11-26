use super::*;

fn run(form: usize, warps: u32, chains: u32, tflops: f64) -> Run {
    Run {
        form,
        warps,
        chains,
        ms: 20.0,
        tflops,
        mhz: 2000.0,
        sms_used: 48,
    }
}

#[test]
fn tflops_counts_two_flops_a_multiply_add() {
    // 48 CTAs of 4 warps, one chain, 1000 trips of m16n8k16 in 1 ms:
    // 48 * 4 * 1000 MMAs * 4096 FLOPs = 786.432 GFLOP in 1 ms.
    let t = tflops(48, 4, 1, 1000, 16 * 8 * 16, 1.0);
    assert!((t - 0.786432).abs() < 1e-9, "{t}");
    assert_eq!(tflops(48, 4, 1, 1000, 2048, 0.0), 0.0);
}

#[test]
fn scaled_iters_aims_at_the_target() {
    assert_eq!(scaled_iters(1024, 0.5, 20.0), 40960);
    // Never fewer than the calibration's, never past 2^30, and a zero time keeps it.
    assert_eq!(scaled_iters(1024, 100.0, 20.0), 1024);
    assert_eq!(scaled_iters(1024, 1e-9, 20.0), 1 << 30);
    assert_eq!(scaled_iters(1024, 0.0, 20.0), 1024);
}

#[test]
fn best_keeps_each_forms_fastest_in_order() {
    let runs = [
        run(3, 4, 1, 50.0),
        run(0, 4, 1, 30.0),
        run(0, 16, 4, 110.0),
        run(3, 8, 2, 220.0),
        run(0, 32, 8, 100.0),
    ];
    let b = best(&runs);
    assert_eq!(b.len(), 2);
    assert_eq!((b[0].form, b[0].warps, b[0].chains), (0, 16, 4));
    assert_eq!((b[1].form, b[1].warps, b[1].chains), (3, 8, 2));
}

#[test]
fn report_names_peaks_and_missing_forms() {
    let r = report(48, &[run(0, 16, 4, 96.0)], &[6]);
    assert!(r.contains("m16n8k16 BF16 -> F32"), "{r}");
    // 96 TFLOPS at 2 GHz on 48 SMs is 1000 FLOPs a clock an SM.
    assert!(r.contains("1000 FLOP/clk/SM"), "{r}");
    assert!(r.contains("mxf4nvf4") && r.contains("needs sm_120a"), "{r}");
}

#[test]
fn forms_match_the_kernels() {
    // src/cuda/mma_peak.cu's table rows and K of each form.
    assert_eq!(FORMS.len(), 7);
    let k: Vec<u64> = FORMS.iter().map(|f| f.mnk / (16 * 8)).collect();
    assert_eq!(k, [16, 16, 16, 32, 32, 32, 64]);
    assert_eq!(CHAINS, [1, 2, 4, 8]);
}

/// The peak table on GPU 0 (run it with `--ignored --nocapture`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU microbench; run with --ignored --nocapture"]
fn cuda_mma_peak() {
    let (sms, runs, missing) = measure(0, 20.0).expect("mma_peak sweep");
    println!("{}", report(sms, &runs, &missing));
    assert!(runs.iter().all(|r| r.tflops > 0.0 && r.sms_used > 0));
}
