//! [`d512_pair`]: which groups with heads of 512 take the warp-pair kernel (#1181).

use super::*;

/// A DGX Spark's GB10: 48 SMs. Gemma 4 31B's global layers: 4 kv heads.
const SMS: usize = 48;
const KV: usize = 4;

#[test]
fn verify_steps_of_a_few_sequences_take_the_pair_kernel() {
    // One sequence, 2-16 tokens: 4-16 pair CTAs.
    for n_tok in [2, 4, 8, 16] {
        assert!(d512_pair(n_tok, 1, KV, SMS), "{n_tok} tokens");
    }
    // Two sequences of 16 tokens: 32; four of 8: 32; eight of 4: 32.
    assert!(d512_pair(16, 2, KV, SMS));
    assert!(d512_pair(8, 4, KV, SMS));
    assert!(d512_pair(4, 8, KV, SMS));
    // Twelve of 4: 48, one wave.
    assert!(d512_pair(4, 12, KV, SMS));
}

#[test]
fn grids_over_one_wave_keep_paged_prefill_mma() {
    // Four sequences of 16 tokens and eight of 8: 64 pair CTAs.
    assert!(!d512_pair(16, 4, KV, SMS));
    assert!(!d512_pair(8, 8, KV, SMS));
    // Sixteen sequences of 2 tokens: 64.
    assert!(!d512_pair(2, 16, KV, SMS));
    // Prompt chunks.
    assert!(!d512_pair(2048, 1, KV, SMS));
    assert!(!d512_pair(96, 1, KV, SMS));
}

#[test]
fn single_tokens_and_unknown_sm_counts_keep_the_old_kernels() {
    // One token a sequence is the decode kernel's.
    assert!(!d512_pair(1, 1, KV, SMS));
    assert!(!d512_pair(8, 1, KV, 0));
}
