//! Which NVFP4 expert pairs and steps build the fused chain, and that tensors without device
//! memory never do.

use super::*;
use crate::ggml::Context;

/// An NVFP4 expert tensor's layout: `[k, rows, n_expert]`, 36 bytes per 64 weights.
fn nvfp4(k: i64, rows: i64, n_expert: i64) -> Layout {
    let row = (k as usize / 64) * 36;
    (
        types::NVFP4,
        [k, rows, n_expert, 1],
        [
            1,
            row,
            row * rows as usize,
            row * (rows * n_expert) as usize,
        ],
    )
}

#[test]
fn nvfp4_pairs_of_one_layout_take_the_chain_up_to_eight_tokens() {
    let g = nvfp4(2560, 640, 512);
    for n in 1..=MAX_TOKENS {
        assert!(pair_fits(g, g, n), "{n} tokens");
    }
    assert!(!pair_fits(g, g, 0));
    assert!(!pair_fits(g, g, MAX_TOKENS + 1));
    assert!(!pair_fits(g, g, 512));
}

#[test]
fn other_types_and_mismatched_pairs_keep_the_separate_launches() {
    let g = nvfp4(2560, 640, 512);
    // Another type, same dims.
    let q = (types::Q4_K, g.1, g.2);
    assert!(!pair_fits(q, q, 1));
    assert!(!pair_fits(g, q, 1));
    // Other dims or strides.
    assert!(!pair_fits(g, nvfp4(2560, 768, 512), 1));
    let mut strided = g;
    strided.2[2] += 36;
    assert!(!pair_fits(g, strided, 1));
}

#[test]
fn tensors_without_device_memory_never_take_the_chain() {
    // Unallocated (a plain context's tensors): no device buffer, so neither `fits` nor `takes`.
    let ctx = Context::new(4, 4);
    let gate = ctx.new_tensor(types::NVFP4, &[128, 4, 2]);
    let up = ctx.new_tensor(types::NVFP4, &[128, 4, 2]);
    assert!(!fits(gate, up, 1));
    assert!(!takes(gate, up, 1));
}
