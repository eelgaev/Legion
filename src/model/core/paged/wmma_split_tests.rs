//! [`WMMA_SPLIT`] (#1211): every row of a 2-16-token group split over the key blocks on sm_70 /
//! sm_75, bit for bit the WMMA kernel's ([`wmma_bench`]); and the scratch's size and plan.

use super::wmma_bench::{self, Group};
use super::*;
use crate::ggml::Backend;

/// Qwen3.8-27B's verify step (24 heads of 256 over 4 kv heads, MTP k = 3: 4 tokens): one query
/// tile of 24 rows (padded to 32) per kv head; G = 12 and 16 take several tiles at 16 tokens.
#[test]
fn rows_and_tiles() {
    assert_eq!(wmma_rows(6, 4), (1, 32));
    assert_eq!(wmma_rows(6, 16), (2, 64));
    assert_eq!(wmma_rows(1, 2), (1, 16));
    assert_eq!(wmma_rows(12, 16), (4, 64));
    assert_eq!(wmma_rows(16, 3), (1, 48));
}

/// The scratch: the whole table at once where it fits the budget, else the budget; none past
/// the CTA cap or outside 2-16 tokens; and the plan read back from it.
#[test]
fn scratch_and_plan() {
    let lane = |chunk| wmma_lane_floats(256, 32, 4096, chunk);
    // One sequence of 27B at a 256K table needs more than the budget: two chunks of weights.
    assert_eq!(wmma_split_floats(256, 24, 4, 4, 1, 4096), WMMA_SPLIT_BUDGET);
    let (seqs, chunk) = wmma_split_plan(WMMA_SPLIT_BUDGET, 256, 24, 4, 4, 1, 4096).unwrap();
    assert_eq!(seqs, 1);
    assert!(chunk >= 4096 / 2 && chunk < 4096);
    assert!(4 * lane(chunk) <= WMMA_SPLIT_BUDGET);
    // A short table fits whole.
    let full = wmma_split_floats(256, 24, 4, 4, 2, 64);
    assert_eq!(full, 2 * 4 * wmma_lane_floats(256, 32, 64, 64));
    assert_eq!(wmma_split_plan(full, 256, 24, 4, 4, 2, 64), Some((2, 64)));
    // Too small for one sequence with the smallest chunk: the WMMA kernel.
    assert_eq!(wmma_split_plan(4 * lane(63), 256, 24, 4, 4, 1, 4096), None);
    assert_eq!(
        wmma_split_plan(4 * lane(64), 256, 24, 4, 4, 1, 4096),
        Some((1, 64))
    );
    // Four sequences of 27B: 16 CTAs; nine: 36, past the cap.
    assert!(wmma_split_floats(256, 24, 4, 4, 4, 4096) > 0);
    assert_eq!(wmma_split_floats(256, 24, 4, 4, 9, 4096), 0);
    assert_eq!(wmma_split_plan(1 << 30, 256, 24, 4, 4, 9, 4096), None);
    // Decode and long chunks keep their kernels.
    assert_eq!(wmma_split_floats(256, 24, 4, 1, 1, 4096), 0);
    assert_eq!(wmma_split_floats(256, 24, 4, 17, 1, 4096), 0);
    assert_eq!(wmma_split_plan(1 << 30, 256, 24, 4, 1, 1, 4096), None);
    // Without a device that runs the WMMA kernel, graphs keep the placeholder.
    if !split_ok() {
        assert_eq!(wmma_split_len(&[(256, 4)], 24, 4, 1, 4096), 0);
    }
}

/// The plumbing on the CPU (which ignores the scratch): the same result with it.
#[test]
fn cpu_split_scratch_is_ignored() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let (cases, total, differ) = wmma_bench::edges(&mut backend, &[(128, 8, 2)], &[2, 7]);
    assert!(cases > 0 && total > 0);
    assert_eq!(differ, 0);
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// The edge cases at every GQA ratio and head size, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_split_rows_bitwise() {
    println!("split path on this device: {}", split_ok());
    let shapes = [
        (128, 32, 2),
        (128, 8, 8),
        (128, 16, 4),
        (256, 16, 8),
        (256, 24, 4),
        (256, 24, 2),
        (512, 16, 2),
        (512, 32, 4),
    ];
    let (_, total, differ) = wmma_bench::edges(&mut gpu(), &shapes, &[2, 3, 4, 7, 16]);
    assert_eq!(differ, 0, "{differ} of {total} values differ");
}

/// A long context at 27B's shape, bit for bit, with the time of either.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_long_context_bitwise_and_timing() {
    let mut backend = gpu();
    let heads = (256, 24, 4);
    let n_blocks = 1100;
    let kv = wmma_bench::pool(&backend, heads, n_blocks);
    let table: Vec<u32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as u32).collect();
    for n_ctx in [1000usize, 65536] {
        let n_past = [n_ctx];
        let tables = [table[..(n_ctx + 4).div_ceil(KV_BLOCK)].to_vec()];
        let g = Group {
            heads,
            n_tok: 4,
            n_past: &n_past,
            tables: &tables,
            width: 4096,
            window: 0,
            ring: 0,
            scale: 1.0 / 16.0,
        };
        let (differ, total) = wmma_bench::compare(&mut backend, &kv, &g, None);
        assert_eq!(differ, 0, "{n_ctx} keys: {differ} of {total} values differ");
        let wmma = wmma_bench::time(&mut backend, &kv, &g, false, 10, 2);
        let split = wmma_bench::time(&mut backend, &kv, &g, true, 10, 2);
        println!("{n_ctx} keys, 4 tokens: {wmma:.3} ms -> {split:.3} ms");
    }
}
