//! The FP32 FMA prefill kernel (`paged_prefill_ffma`, written for Turing GPUs without tensor
//! cores) against the f64 oracle: the cases of `tests.rs` and `ring_tests.rs`, with the kernel
//! forced for the test's thread. It is plain FP32 math, so these check its indexing on any GPU;
//! its speed only means something on a GTX 16xx. It rounds neither the query nor the softmax
//! weights to F16, so the oracle doesn't either ([`super::tests::EXACT_ORACLE`]).

use super::ring_tests::{ring_case, seams};
use super::tests::{cuda_long_context, cuda_prefill_edges};
use super::*;
use crate::ggml::{Backend, DeviceKind, device_by_kind};

/// The kernel and the exact oracle for this thread until dropped.
struct Ffma;

impl Ffma {
    fn on() -> Self {
        set_ffma_prefill_here(true);
        super::tests::EXACT_ORACLE.set(true);
        Ffma
    }
}

impl Drop for Ffma {
    fn drop(&mut self) {
        set_ffma_prefill_here(false);
        super::tests::EXACT_ORACLE.set(false);
    }
}

fn gpu() -> Backend {
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// Several sequences over shuffled blocks, ragged tiles, mid-block windows, GQA 1 to 16,
/// heads of 128, 256 and 512.
#[test]
#[ignore = "needs a GPU"]
fn ffma_prefill_edges_match_oracle() {
    let _ffma = Ffma::on();
    let before = ffma_launches();
    cuda_prefill_edges();
    assert!(ffma_launches() > before, "the FFMA kernel did not run");
}

/// 8K keys: rounding noise and no bias.
#[test]
#[ignore = "needs a GPU"]
fn ffma_long_context_matches_oracle() {
    let _ffma = Ffma::on();
    let before = ffma_launches();
    for (n_head, n_head_kv) in [(16, 4), (24, 2), (24, 4)] {
        cuda_long_context(256, n_head, n_head_kv, 1.0 / 16.0, 0);
    }
    cuda_long_context(256, 16, 8, 1.0 / 16.0, 1024);
    cuda_long_context(512, 32, 4, 1.0 / 22.6, 0);
    cuda_long_context(128, 32, 2, 1.0 / 11.3, 0);
    assert!(ffma_launches() > before, "the FFMA kernel did not run");
}

/// Sliding-window rings, Gemma 4's sliding shape and window.
#[test]
#[ignore = "needs a GPU"]
fn ffma_ring_matches_full_cache_and_oracle() {
    let _ffma = Ffma::on();
    let before = ffma_launches();
    let mut backend = gpu();
    seams(&mut backend, 5e-4);
    for n_past in [[1024, 1536, 2047], [2048, 2560, 3071]] {
        ring_case(&mut backend, 256, 16, 8, 1024, 512, &n_past, 512, 5e-4);
    }
    assert!(ffma_launches() > before, "the FFMA kernel did not run");
}

/// The flag for all threads (`--tu-attn-ffma`) leaves GPUs other than sm_75 on their own
/// kernels.
#[test]
#[ignore = "needs a GPU"]
fn ffma_flag_for_all_threads_skips_other_gpus() {
    let mut backend = gpu();
    // Off again however the test ends: other tests of this process read the flag.
    struct Off;
    impl Drop for Off {
        fn drop(&mut self) {
            set_ffma_prefill(false);
        }
    }
    set_ffma_prefill(true);
    let _off = Off;
    let before = ffma_launches();
    ring_case(&mut backend, 256, 16, 8, 100, 37, &[190], 37, 5e-4);
    let sm75 = crate::hw::per_device::compute_capability(0) == Some(75);
    assert_eq!(ffma_launches() > before, sm75, "sm_75: {sm75}");
}
