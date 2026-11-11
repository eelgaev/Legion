//! The m8n8k4 prefill kernel (`paged_prefill_m884`, written for Volta) against the f64 oracle:
//! the cases of `tests.rs` and `ring_tests.rs`, with the kernel forced for the test's thread.
//! m8n8k4 runs on every GPU from sm_70, so these check its indexing anywhere; its speed only
//! means something on sm_70.

use super::ring_tests::{ring_case, seams};
use super::tests::{cuda_long_context, cuda_prefill_edges};
use super::*;
use crate::ggml::{Backend, DeviceKind, device_by_kind};

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
fn m884_prefill_edges_match_oracle() {
    set_m884_prefill_here(true);
    let before = m884_launches();
    cuda_prefill_edges();
    assert!(m884_launches() > before, "the m8n8k4 kernel did not run");
}

/// 8K keys: rounding noise and no bias from the truncating accumulators (chains of 16 MMAs).
#[test]
#[ignore = "needs a GPU"]
fn m884_long_context_matches_oracle() {
    set_m884_prefill_here(true);
    let before = m884_launches();
    for (n_head, n_head_kv) in [(16, 4), (24, 2), (24, 4)] {
        cuda_long_context(256, n_head, n_head_kv, 1.0 / 16.0, 0);
    }
    cuda_long_context(256, 16, 8, 1.0 / 16.0, 1024);
    cuda_long_context(512, 32, 4, 1.0 / 22.6, 0);
    assert!(m884_launches() > before, "the m8n8k4 kernel did not run");
}

/// Sliding-window rings, Gemma 4's sliding shape and window.
#[test]
#[ignore = "needs a GPU"]
fn m884_ring_matches_full_cache_and_oracle() {
    set_m884_prefill_here(true);
    let before = m884_launches();
    let mut backend = gpu();
    seams(&mut backend, 5e-4);
    for n_past in [[1024, 1536, 2047], [2048, 2560, 3071]] {
        ring_case(&mut backend, 256, 16, 8, 1024, 512, &n_past, 512, 5e-4);
    }
    assert!(m884_launches() > before, "the m8n8k4 kernel did not run");
}

/// The flag for all threads (`--paged-prefill-m884`) leaves sm_80+ GPUs on their own kernel
/// (m8n8k4 runs emulated there).
#[test]
#[ignore = "needs a GPU"]
fn m884_flag_for_all_threads_skips_sm80() {
    let mut backend = gpu();
    if !quant_ok() {
        return; // sm_70/75: the flag applies on sm_70 (and sm_75 falls back by its shared memory)
    }
    // The choice for every thread as this thread alone reads it: the GPU tests running beside
    // this one keep their default kernel. Off again however the test ends.
    struct Off;
    impl Drop for Off {
        fn drop(&mut self) {
            set_prefill_for_all_here(-1);
        }
    }
    set_prefill_for_all_here(1);
    let _off = Off;
    let before = m884_launches();
    ring_case(&mut backend, 256, 16, 8, 100, 37, &[190], 37, 5e-4);
    assert_eq!(m884_launches(), before, "the m8n8k4 kernel ran on sm_80+");
}
