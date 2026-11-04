use super::*;

#[test]
fn depths() {
    let cfg = SpecConfig::default();
    assert_eq!(adaptive_k(1.0, 10, 8, &cfg), 0);
    assert_eq!(adaptive_k(0.7, 10, 128, &cfg), 0);
    assert_eq!(adaptive_k(1.0, 10, 1, &cfg), 10);
    assert_eq!(adaptive_k(0.999, 10, 1, &cfg), 10);
    assert_eq!(adaptive_k(0.5, 10, 1, &cfg), 1);
    assert_eq!(adaptive_k(0.9, 10, 1, &cfg), 6);
    assert_eq!(adaptive_k(0.9, 4, 1, &cfg), 4);
    assert_eq!(adaptive_k(0.1, 10, 1, &cfg), 0);
    assert_eq!(adaptive_k(0.0, 10, 1, &cfg), 0);
    assert_eq!(adaptive_k(1.0, 0, 1, &cfg), 0);
}

#[test]
fn estimate_and_probe_cadence() {
    let cfg = SpecConfig::default();
    let mut s = AdaptiveState::default();
    s.record(0, 0);
    assert_eq!(s.alpha, 0.7);
    s.record(4, 2);
    assert!((s.alpha - 0.66).abs() < 1e-6);
    s.record(4, 0);
    s.record(4, 0);
    for step in 1..=48 {
        assert_eq!(s.select(4, 1, &cfg), usize::from(step % 16 == 0));
        s.record(0, 0);
    }
    // A batch-gated step breaks the consecutive low-alpha streak, and never probes.
    for _ in 0..15 {
        assert_eq!(s.select(4, 1, &cfg), 0);
    }
    for _ in 0..32 {
        assert_eq!(s.select(4, 8, &cfg), 0);
    }
    for _ in 0..15 {
        assert_eq!(s.select(4, 1, &cfg), 0);
    }
    assert_eq!(s.select(4, 1, &cfg), 1);
    s.record(1, 1);
    assert!(s.select(4, 1, &cfg) > 0, "successful probes recover");
}

#[test]
fn fixed_bypasses_acceptance_and_load() {
    let cfg = SpecConfig {
        fixed: true,
        ..Default::default()
    };
    let mut s = AdaptiveState::default();
    for _ in 0..32 {
        s.record(4, 0);
        assert_eq!(s.select(4, 128, &cfg), 4);
    }
}

#[test]
fn cost_model_gate_follows_its_switch() {
    let cfg = SpecConfig::default();
    assert!(!cost_model_gated(7, &cfg));
    assert!(cost_model_gated(8, &cfg));
    assert!(cost_model_gated(64, &cfg));
    let off = SpecConfig {
        gate_cost_model: false,
        ..Default::default()
    };
    for batch in [1, 8, 16, 64] {
        assert!(!cost_model_gated(batch, &off), "{batch}");
    }
    // Off keeps the gate for drafters that pick their depth by acceptance alone.
    assert_eq!(adaptive_k(1.0, 4, 8, &off), 0);
    assert_eq!(adaptive_k(1.0, 4, 7, &off), 4);
    assert!(!cost_model_ungated_large(16, &cfg));
    assert!(cost_model_ungated_large(16, &off));
}

/// #814: with `gate_until`, the cost-model drafters stop only from `max_batch` to it and draft
/// again past it (so a full batch's buffers count); others keep the whole gate.
#[test]
fn cost_model_gate_ends_at_gate_until() {
    let band = SpecConfig {
        gate_until: 8,
        ..Default::default()
    };
    assert!(!cost_model_gated(7, &band));
    assert!(cost_model_gated(8, &band));
    for batch in [9, 12, 16, 64] {
        assert!(!cost_model_gated(batch, &band), "{batch}");
    }
    assert!(cost_model_ungated_large(16, &band));
    assert!(!cost_model_ungated_large(8, &band));
    assert_eq!(adaptive_k(1.0, 4, 16, &band), 0);
}

/// `--spec-max-rows 63` (V100s, below ggml's cuBLAS path at 64 rows): 16 sequences draft 2
/// each (48 rows), 8 or 12 keep `--spec-k 3`, 32 and more draft nothing; 0 caps nothing.
#[test]
fn max_rows_lowers_the_depth_of_large_batches() {
    assert_eq!(rows_depth_cap(16, 3, 0), 3);
    assert_eq!(rows_depth_cap(1, 3, 63), 3);
    assert_eq!(rows_depth_cap(8, 3, 63), 3);
    assert_eq!(rows_depth_cap(12, 3, 63), 3);
    assert_eq!(rows_depth_cap(16, 3, 63), 2);
    assert_eq!(rows_depth_cap(21, 3, 63), 2);
    assert_eq!(rows_depth_cap(22, 3, 63), 1);
    assert_eq!(rows_depth_cap(32, 3, 63), 0);
    assert_eq!(rows_depth_cap(64, 3, 63), 0);
    for batch in 1..=64 {
        let d = rows_depth_cap(batch, 3, 63);
        assert!(d == 0 || batch * (1 + d) <= 63, "{batch}");
    }
}

#[test]
fn chain_rate_counts_the_position_that_ended_a_chain() {
    let mut s = AdaptiveState::default();
    assert!((s.rate() - 0.7).abs() < 1e-6);
    // No cap: nothing drafted on purpose, no evidence.
    s.record_chain(0, 0, 0);
    assert!((s.rate() - 0.7).abs() < 1e-6);
    // A chain that reached its cap says nothing about the next position: 4 of 4.
    s.record_chain(4, 4, 4);
    let (h, t) = (0.95 * 1.4 + 4.0, 0.95 * 2.0 + 4.0);
    assert!((s.rate() - h / t).abs() < 1e-6);
    // Two accepted, the third rejected: 2 of 3. The head stopping early counts the same.
    for (drafted, accepted) in [(4, 2), (2, 2)] {
        let mut s = AdaptiveState::default();
        s.record_chain(4, drafted, accepted);
        let (h, t) = (0.95 * 1.4 + 2.0, 0.95 * 2.0 + 3.0);
        assert!((s.rate() - h / t).abs() < 1e-6, "{drafted} drafted");
    }
    // Every draft rejected for a while: the rate falls towards 0, and recovers.
    let mut s = AdaptiveState::default();
    for _ in 0..100 {
        s.record_chain(3, 3, 0);
    }
    assert!(s.rate() < 0.01);
    for _ in 0..100 {
        s.record_chain(3, 3, 3);
    }
    assert!(s.rate() > 0.95);
}

#[test]
fn geometric_gain_is_the_chain_sum() {
    assert_eq!(geometric_gain(0.5, 0), 1.0);
    assert_eq!(geometric_gain(0.5, 3), 1.875);
    assert_eq!(geometric_gain(1.0, 6), 7.0);
    assert_eq!(geometric_gain(0.0, 4), 1.0);
}

#[test]
fn one_step_tells_every_cap_its_gain() {
    // Cap 4, all drafted, 2 accepted: caps 0, 1 and 2 or more would have gained 1, 2 and 3
    // tokens (the chain ended at the third draft under any larger cap).
    let mut s = AdaptiveState::default();
    let before: Vec<f32> = (0..=6).map(|k| s.gain(k)).collect();
    s.record_chain(4, 4, 2);
    for (k, now) in [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0].into_iter().enumerate() {
        let want = 0.95 * before[k] + 0.05 * now;
        assert!((s.gain(k) - want).abs() < 1e-6, "cap {k}");
    }
    // The head stopped after 1 draft of 3, accepted: as a rejection of the second.
    let mut s = AdaptiveState::default();
    s.record_chain(3, 1, 1);
    assert!((s.gain(5) - (0.95 * geometric_gain(0.7, 5) + 0.05 * 2.0)).abs() < 1e-6);
    // A chain that reached its cap: caps above it get the cap's tokens plus a geometric tail
    // at the per-position rate.
    let mut s = AdaptiveState::default();
    s.record_chain(2, 2, 2);
    let r = s.rate();
    for k in 0..=5 {
        let now = if k <= 2 {
            (k + 1) as f32
        } else {
            2.0 + geometric_gain(r, k - 2)
        };
        let want = 0.95 * geometric_gain(0.7, k) + 0.05 * now;
        assert!((s.gain(k) - want).abs() < 1e-5, "cap {k}");
    }
    // Deeper than estimated: the deepest estimate.
    assert_eq!(s.gain(MAX_GAIN_DEPTH + 3), s.gain(MAX_GAIN_DEPTH));
}
