//! Micro-benchmark of the host kernels against the scalar reference and ggml's CPU `vec_dot`.
//!
//! `cargo test --release --lib cpu::bench -- --ignored --nocapture`
//! (`LEGION_CPU_BENCH=rows,columns` changes the shape; default 4096,4096). The quantized
//! formats run [`microbench::kernels`] (also `legion cpu-bench kernels`); BF16 is here.
//!
//! Single thread on every path, ggml built for this host by `GGML_NATIVE`.

use std::time::Duration;

use super::microbench::{self, time};
use super::tests::Rng;
use super::*;
use crate::ggml::{self, ffi};

#[test]
#[ignore = "micro-benchmark; run with --release --ignored --nocapture"]
fn bench_host_kernels() {
    // A test input of this ignored micro-benchmark (docs/debug-env.md), not a setting, so an
    // environment variable rather than a flag (#737).
    let (rows, columns) = std::env::var("LEGION_CPU_BENCH")
        .ok()
        .and_then(|s| {
            let (r, c) = s.split_once(',')?;
            Some((r.trim().parse().ok()?, c.trim().parse().ok()?))
        })
        .unwrap_or((4096usize, 4096usize));
    let budget = Duration::from_millis(400);
    microbench::kernels(
        rows,
        columns,
        &QuantFormat::ALL,
        &[1, 4],
        budget,
        &mut |l| println!("{l}"),
    );
    bench_bf16(rows, columns, budget, &mut Rng::new(42));
}

/// One BF16 token (a router or BF16 projection matvec).
fn bench_bf16(rows: usize, columns: usize, budget: Duration, rng: &mut Rng) {
    let bf = crate::hf::nvfp4::f32_to_bf16;
    let w: Vec<u16> = (0..rows * columns).map(|_| bf(rng.unit())).collect();
    let x: Vec<u16> = (0..columns).map(|_| bf(rng.unit())).collect();
    let gb = (w.len() * 2) as f64 / 1e9;
    let mut out = vec![0.0f32; rows];
    // SAFETY: static traits table.
    let vec_dot = unsafe { (*ffi::ggml_get_type_traits_cpu(ggml::types::BF16)).vec_dot }.unwrap();
    let ggml_t = time(budget, 3, || {
        for (r, o) in out.iter_mut().enumerate() {
            // SAFETY: one BF16 row of `columns` values each for weights and activations.
            unsafe {
                vec_dot(
                    columns as i32,
                    o,
                    0,
                    w[r * columns..].as_ptr().cast(),
                    0,
                    x.as_ptr().cast(),
                    0,
                    1,
                )
            };
        }
        std::hint::black_box(&out);
    });
    let mut scalar_t = Duration::ZERO;
    for isa in supported_isas() {
        let t = time(budget, 1, || {
            matmul_bf16_with_isa(isa, &w, &x, rows, columns, &mut out).unwrap();
            std::hint::black_box(&out);
        });
        if isa == Isa::Scalar {
            scalar_t = t;
        }
        println!(
            "BF16         1  {:<11} {:>8.3} {:>9.2} {:>9.2}x {:>7.2}x",
            isa.name(),
            t.as_secs_f64() * 1e3,
            gb / t.as_secs_f64(),
            scalar_t.as_secs_f64() / t.as_secs_f64(),
            ggml_t.as_secs_f64() / t.as_secs_f64(),
        );
    }
    println!(
        "BF16         1  {:<11} {:>8.3} {:>9.2}",
        "ggml",
        ggml_t.as_secs_f64() * 1e3,
        gb / ggml_t.as_secs_f64(),
    );
}
