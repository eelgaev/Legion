//! [`verify_grid`]: which groups with heads of 256 at 6 query heads a kv head take the wide-grid
//! kernel (#689).

use super::*;

/// A DGX Spark's GB10: 48 SMs. Qwen3.8-27B's attention layers: 4 kv heads.
const SMS: usize = 48;
const KV: usize = 4;

#[test]
fn verify_steps_whose_grid_fits_one_wave_take_the_wide_grid() {
    // One sequence of 8 tokens: 48 columns, 3 CTAs a kv head, 12 CTAs.
    assert!(verify_grid(8, 1, KV, SMS));
    // One of 16: 6 a kv head, 24.
    assert!(verify_grid(16, 1, KV, SMS));
    // One of 2 (12 columns): 1 a kv head.
    assert!(verify_grid(2, 1, KV, SMS));
    // Two of 16: 48.
    assert!(verify_grid(16, 2, KV, SMS));
    // Four of 8: 48.
    assert!(verify_grid(8, 4, KV, SMS));
    // Eight of 2: 32.
    assert!(verify_grid(2, 8, KV, SMS));
}

#[test]
fn grids_over_one_wave_keep_paged_prefill_mma() {
    // Four sequences of 16 tokens: 96 CTAs.
    assert!(!verify_grid(16, 4, KV, SMS));
    // Eight of 4 (24 columns, 2 CTAs a kv head): 64.
    assert!(!verify_grid(4, 8, KV, SMS));
    // Sixteen of 2: 64.
    assert!(!verify_grid(2, 16, KV, SMS));
    // Fewer SMs: one sequence of 16 tokens on 16.
    assert!(!verify_grid(16, 1, KV, 16));
}

#[test]
fn prompt_chunks_single_tokens_and_unknown_sm_counts_keep_the_old_kernels() {
    assert!(!verify_grid(17, 1, KV, SMS));
    assert!(!verify_grid(2048, 1, KV, SMS));
    // One token a sequence is the decode kernel's.
    assert!(!verify_grid(1, 1, KV, SMS));
    assert!(!verify_grid(8, 1, KV, 0));
}
