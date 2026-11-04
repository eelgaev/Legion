use super::*;
use crate::spec::adaptive::{AdaptiveState, geometric_gain};

/// One sequence gaining `1 + a + ... + a^k` from `k` drafts, for `k` in `0..=k_max`.
fn geo(a: f32, k_max: usize) -> Vec<Vec<f32>> {
    vec![(0..=k_max).map(|k| geometric_gain(a, k)).collect()]
}

#[test]
fn unmeasured_rows_are_interpolated_or_held() {
    let mut m = DepthModel::new(false);
    assert_eq!(m.step_ms(3), None);
    m.record_step(2, 10.0);
    m.record_step(6, 18.0);
    assert_eq!(m.step_ms(2), Some(10.0));
    assert_eq!(m.step_ms(4), Some(14.0));
    // Held at the nearest measured value outside the measured range.
    assert_eq!(m.step_ms(1), Some(10.0));
    assert_eq!(m.step_ms(9), Some(18.0));
    // A higher second sample does not count (the first step of a shape builds its graph, so
    // only a lower one replaces it); then EWMA, with spikes clipped at SPIKE x the mean.
    m.record_step(2, 20.0);
    assert_eq!(m.step_ms(2), Some(10.0));
    m.record_step(2, 11.0);
    assert!((m.step_ms(2).unwrap() - 10.2).abs() < 1e-9);
    m.record_step(2, 40.0);
    assert!((m.step_ms(2).unwrap() - (10.2 + EWMA * (SPIKE - 1.0) * 10.2)).abs() < 1e-9);
    let mut first_high = DepthModel::new(false);
    first_high.record_step(3, 30.0);
    first_high.record_step(3, 20.0);
    assert_eq!(first_high.step_ms(3), Some(20.0));
    // Per pass.
    m.record_passes(1, 4, 8.0);
    assert_eq!(m.pass_ms(1), Some(2.0));
    assert_eq!(m.pass_ms(3), Some(2.0));
}

#[test]
fn first_choice_is_moderate_then_follows_costs() {
    let mut m = DepthModel::new(false);
    assert_eq!(m.choose(&geo(0.8, 8), 8), FIRST_DEPTH);
    assert_eq!(m.choose(&[], 8), 0);
    assert_eq!(m.choose(&geo(0.8, 0), 0), 0);
    // A flat verify step and cheap passes: as deep as allowed, one step up per choice.
    for r in 1..=9 {
        m.record_step(r, 80.0);
    }
    m.record_passes(1, 1, 1.0);
    let climb: Vec<usize> = (0..8).map(|_| m.choose(&geo(0.9, 8), 8)).collect();
    assert_eq!(climb, [3, 4, 5, 6, 7, 8, 8, 8]);
    // Each row as expensive as the bonus token's whole step: no drafts, at once.
    let mut m = DepthModel::new(false);
    m.choose(&geo(0.5, 4), 4);
    for r in 1..=5 {
        m.record_step(r, 10.0 * r as f64);
    }
    m.record_passes(1, 1, 1.0);
    assert_eq!(m.choose(&geo(0.5, 4), 4), 0);
}

#[test]
fn a_head_that_takes_in_pays_a_pass_at_depth_zero() {
    for (take_in, want) in [(false, 0.1), (true, 1.0 / 12.0)] {
        let mut m = DepthModel::new(take_in);
        m.record_step(1, 10.0);
        m.record_passes(1, 1, 2.0);
        let t = m.throughput(&geo(0.5, 0), 0).unwrap();
        assert!((t - want).abs() < 1e-12, "take_in {take_in}: {t}");
    }
}

#[test]
fn a_block_drafter_pays_one_pass_at_every_depth() {
    let mut m = DepthModel::block();
    for r in 1..=8 {
        m.record_step(r, 80.0 + r as f64);
    }
    m.record_passes(1, 1, 12.0);
    for k in 0..=7 {
        let t = m.throughput(&geo(0.7, 7), k).unwrap();
        let want = geometric_gain(0.7, k) as f64 / (80.0 + (1 + k) as f64 + 12.0);
        assert!((t - want).abs() < 1e-9, "k {k}: {t} vs {want}");
    }
    // A flat verify step: as deep as allowed, where a head paying per draft stays shallower.
    let mut per_pass = DepthModel::new(true);
    for r in 1..=8 {
        per_pass.record_step(r, 80.0 + r as f64);
    }
    per_pass.record_passes(1, 1, 12.0);
    let settle = |m: &mut DepthModel| (0..12).map(|_| m.choose(&geo(0.8, 7), 7)).last();
    assert_eq!(settle(&mut m), Some(7));
    assert!(settle(&mut per_pass).unwrap() < 7);
}

#[test]
fn a_batch_adds_its_sequences_gains() {
    let mut m = DepthModel::new(false);
    m.record_step(2, 10.0);
    m.record_step(6, 20.0);
    m.record_passes(2, 1, 1.0);
    let gains = vec![vec![1.0, 1.5, 2.0], vec![1.0, 1.9, 2.7]];
    // Two sequences of three rows each: 6 rows.
    let t = m.throughput(&gains, 2).unwrap();
    assert!((t - (2.0 + 2.7) / (20.0 + 2.0)).abs() < 1e-6, "{t}");
}

#[test]
fn probes_alternate_around_the_best_depth() {
    let mut m = DepthModel::new(false);
    m.choose(&geo(0.9, 8), 8);
    for r in 1..=9 {
        m.record_step(r, 50.0);
    }
    m.record_passes(1, 1, 5.0);
    // Climbs while a step up gains more than HYSTERESIS (G(k) / (50 + 5 k) at 0.8: 5 gives
    // 0.0492, 6 0.0494), then probes.
    let gains = geo(0.8, 8);
    let picks: Vec<usize> = (0..3 * PROBE_EVERY).map(|_| m.choose(&gains, 8)).collect();
    assert_eq!(picks[..5], [3, 4, 5, 5, 5]);
    for (i, &k) in picks.iter().enumerate().skip(4) {
        let want = match (i + 1) % (2 * PROBE_EVERY) {
            0 => 4,
            n if n == PROBE_EVERY => 6,
            _ => 5,
        };
        assert_eq!(k, want, "choice {i}: {picks:?}");
    }
}

#[test]
fn small_differences_keep_the_current_depth() {
    let gains = geo(0.75, 6);
    let mut m = DepthModel::new(false);
    m.choose(&gains, 6);
    m.record_step(3, 20.0);
    m.record_passes(1, 1, 2.0);
    // Flat at 20 ms: the policy goes deep first, up to where a step gains less than
    // HYSTERESIS (G(k) / (20 + 2 k) at 0.75: 4 gives 0.1090, 5 0.1095).
    let mut deep = 0;
    for _ in 0..4 {
        deep = m.choose(&gains, 6);
    }
    assert_eq!(deep, 4);
    // Costs that make a neighbour 1% better do not move it.
    let here = m.throughput(&gains, deep).unwrap();
    let step = gains[0][deep + 1] as f64 / (here * 1.01) - 2.0 * (deep + 1) as f64;
    m.record_step(2 + deep, step);
    assert!(m.throughput(&gains, deep + 1).unwrap() > here);
    assert_eq!(m.choose(&gains, 6), deep);
}

/// The whole policy (gain estimates from accepted chains, measured costs) against costs
/// measured on GB10 (Gemma 4 QAT Q4_0 with its Q8_0 assistant: verify step medians at
/// `--spec-fixed` depths, one assistant pass), with drafts accepted at rates that fall along
/// the chain: after a warm-up the depths it picks give within 3% of the best fixed depth's
/// tokens per millisecond, with the measurements jittered by +-3%.
#[test]
fn settles_within_three_percent_of_the_best_fixed_depth() {
    // Verify step (ms) by rows 1..=9, and one assistant pass.
    let moe_26b = ([12.5, 16.0, 17.9, 20.15, 22.1, 24.0, 25.8, 27.6, 29.4], 2.2);
    let dense_31b = ([82.0, 85.0, 85.3, 85.6, 85.3, 86.2, 87.1, 88.0, 102.7], 2.5);
    // Acceptance of draft i given drafts before it were accepted.
    let falling = [0.85, 0.8, 0.78, 0.74, 0.7, 0.68, 0.65, 0.62];
    let good = [0.95, 0.93, 0.92, 0.9, 0.9, 0.88, 0.86, 0.85];
    let poor = [0.5, 0.4, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3];
    for (name, (verify, pass), accept) in [
        ("26B", moe_26b, falling),
        ("31B", dense_31b, falling),
        ("26B, good drafts", moe_26b, good),
        ("26B, poor drafts", moe_26b, poor),
        ("31B, good drafts", dense_31b, good),
        ("31B, poor drafts", dense_31b, poor),
    ] {
        // Expected tokens of a fixed depth, and the best fixed depth's tokens per ms.
        let gain = |k: usize| {
            (0..k)
                .scan(1.0f64, |p, i| {
                    *p *= accept[i];
                    Some(*p)
                })
                .sum::<f64>()
                + 1.0
        };
        let cost = |k: usize| verify[k] + pass * k as f64;
        let best = (0..=8).map(|k| gain(k) / cost(k)).fold(0.0, f64::max);
        let mut m = DepthModel::new(false);
        let mut seq = AdaptiveState::default();
        let mut x = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f64 / (1u64 << 24) as f64
        };
        let (mut tokens, mut ms) = (0.0, 0.0);
        for step in 0..1500 {
            let gains = vec![(0..=8).map(|k| seq.gain(k)).collect::<Vec<f32>>()];
            let k = m.choose(&gains, 8);
            let accepted = (0..k).take_while(|&i| uniform() < accept[i]).count();
            seq.record_chain(k, k, accepted);
            let jitter = 1.0 + (uniform() - 0.5) * 0.06;
            m.record_step(1 + k, verify[k] * jitter);
            if k > 0 {
                m.record_passes(1, k, pass * k as f64 * jitter);
            }
            if step >= 100 {
                tokens += gain(k);
                ms += cost(k);
            }
        }
        let got = tokens / ms;
        assert!(
            got >= 0.97 * best,
            "{name}: {got:.4} tokens/ms against the best fixed depth's {best:.4}"
        );
    }
}

#[test]
fn drafts_go_on_while_the_head_is_confident_enough_to_pay_for_a_pass() {
    let gains = geo(0.8, 4);
    let mut m = DepthModel::new(false);
    // Nothing measured: the ceiling.
    assert_eq!(m.confidence_floor(&gains, 2, 0.5), 0.5);
    m.record_step(3, 20.0);
    m.record_passes(1, 1, 2.0);
    // 2.44 tokens per 24 ms: a pass costs 0.203 of a token's time. Calibrated at first (a draft
    // at confidence c is accepted with probability c): from the bucket whose middle beats it.
    assert!((2.0 * m.throughput(&gains, 2).unwrap() - 0.2033).abs() < 1e-3);
    assert_eq!(m.confidence_floor(&gains, 2, 0.5), 0.2);
    // Clamped to the ceiling and to MIN_CONFIDENCE.
    assert_eq!(m.confidence_floor(&gains, 2, 0.1), 0.1);
    let mut dense = DepthModel::new(false);
    dense.record_step(3, 2000.0);
    dense.record_passes(1, 1, 0.1);
    assert_eq!(dense.confidence_floor(&gains, 2, 0.5), MIN_CONFIDENCE);
    // Drafts at 0.2-0.3 turn out to be rejected: the floor rises above them.
    for _ in 0..40 {
        m.record_drafts(&[0.9, 0.25], 1);
    }
    assert_eq!(m.confidence_floor(&gains, 2, 0.5), 0.3);
    // Only drafts up to the first rejection are judged: the third one here is not.
    let mut c = DepthModel::new(false);
    c.record_drafts(&[0.95, 0.05, 0.05], 0);
    assert_eq!(
        c.calibration.judged[9],
        PRIOR_DRAFTS * CALIBRATION_DECAY + 1.0
    );
    assert_eq!(c.calibration.judged[0], PRIOR_DRAFTS);
}

/// Steps of 7 rows (one sequence at depth 6) costing `base + row` per drafted row, for 1 to 6
/// drafts in turn, `n` steps in all.
fn fill(m: &mut DepthModel, base: f64, row: f64, n: usize) {
    for i in 0..n {
        let drafted = 1 + i % 6;
        let ms = base + row * drafted as f64;
        m.record_step(7, ms);
        m.record_fill(7, drafted, ms);
    }
}

#[test]
fn a_drafted_row_is_priced_from_steps_of_one_shape() {
    let mut m = DepthModel::new(true);
    assert_eq!(m.row_ms(7), None);
    // The first two steps of a shape are left out (graph builds), then a few more are needed.
    fill(&mut m, 60.0, 2.0, 6);
    assert_eq!(m.row_ms(7), None);
    fill(&mut m, 60.0, 2.0, 24);
    assert!(
        (m.row_ms(7).unwrap() - 2.0).abs() < 1e-6,
        "{:?}",
        m.row_ms(7)
    );
    // Other shapes are not priced from it.
    assert_eq!(m.row_ms(5), None);
    // Steps that all drafted the same say nothing.
    let mut same = DepthModel::new(true);
    for _ in 0..30 {
        same.record_step(7, 70.0);
        same.record_fill(7, 4, 70.0);
    }
    assert_eq!(same.row_ms(7), None);
    // Padding that costs more than drafts (noise) prices a row at 0, not below.
    let mut neg = DepthModel::new(true);
    fill(&mut neg, 80.0, -1.0, 30);
    assert_eq!(neg.row_ms(7), Some(0.0));
    // A spike (a graph rebuilt) does not count.
    let mut spiky = DepthModel::new(true);
    fill(&mut spiky, 60.0, 2.0, 30);
    spiky.record_step(7, 200.0);
    spiky.record_fill(7, 6, 200.0);
    assert!((spiky.row_ms(7).unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn a_priced_row_raises_the_confidence_floor() {
    let gains = geo(0.8, 6);
    let mut m = DepthModel::new(true);
    m.record_passes(1, 1, 0.5);
    // Free drafted rows: the pass alone (72 ms steps and six 0.5 ms passes for 3.95 tokens: a
    // pass is 0.026 of a token's time, under the lowest bucket).
    fill(&mut m, 72.0, 0.0, 30);
    // 0 up to the fit's rounding (2e-14 on aarch64).
    let row = m.row_ms(7).expect("priced");
    assert!(row.abs() < 1e-9, "{row}");
    let free = m.confidence_floor(&gains, 6, 0.5);
    assert_eq!(free, MIN_CONFIDENCE);
    // Rows of 3 ms more than padding: a draft now costs 3.5 ms of a ~19 ms token.
    let mut dear = DepthModel::new(true);
    dear.record_passes(1, 1, 0.5);
    fill(&mut dear, 60.0, 3.0, 30);
    let t = dear.throughput(&gains, 6).unwrap();
    let row = dear.row_ms(7).unwrap();
    let want = Calibration::default().floor((0.5 * t + row * t) as f32);
    assert_eq!(dear.confidence_floor(&gains, 6, 0.5), want);
    assert!(want >= 0.2, "{want}");
    // Without the fit (--spec-row-cost off records none): the pass alone, as before.
    let mut off = DepthModel::new(true);
    off.record_passes(1, 1, 0.5);
    for i in 0..30 {
        off.record_step(7, 60.0 + 3.0 * (1 + i % 6) as f64);
    }
    assert_eq!(off.confidence_floor(&gains, 6, 0.5), MIN_CONFIDENCE);
}

#[test]
fn context_buckets() {
    assert_eq!(context_bucket(0), 0);
    assert_eq!(context_bucket(4095), 0);
    assert_eq!(context_bucket(4096), 1);
    assert_eq!(context_bucket(8191), 1);
    assert_eq!(context_bucket(8192), 2);
    assert_eq!(context_bucket(13000), 2);
    assert_eq!(context_bucket(32768), 4);
}

/// Costs are kept per context: a long prompt's dear steps (a 13K-token needle) do not change
/// the short chats' costs or depth after it, and a bucket not yet measured borrows the nearest
/// one's costs (the shorter first).
#[test]
fn a_long_context_does_not_price_short_steps() {
    let mut m = DepthModel::new(false);
    m.set_context(500);
    m.choose(&geo(0.8, 3), 3);
    // Short: a verify row costs little.
    for r in 1..=4 {
        m.record_step(r, 30.0 + r as f64);
    }
    m.record_passes(1, 1, 1.4);
    let picks: Vec<usize> = (0..4).map(|_| m.choose(&geo(0.8, 3), 3)).collect();
    assert_eq!(*picks.last().unwrap(), 3, "{picks:?}");
    // A needle at 13K: every step dearer, a pass three times dearer.
    m.set_context(13000);
    assert_eq!(
        m.step_ms(4),
        Some(34.0),
        "an empty bucket borrows the nearest"
    );
    for _ in 0..20 {
        for r in 1..=4 {
            m.record_step(r, 40.0 + 10.0 * r as f64);
        }
        m.record_passes(1, 1, 4.5);
    }
    // Back to short chats: their own costs, their own depth.
    m.set_context(600);
    assert_eq!(m.step_ms(4), Some(34.0));
    assert!((m.pass_ms(1).unwrap() - 1.4).abs() < 1e-9);
    assert_eq!(m.choose(&geo(0.8, 3), 3), 3);
    // A context between them borrows the shorter bucket.
    m.set_context(5000);
    assert_eq!(m.step_ms(4), Some(34.0));
}
