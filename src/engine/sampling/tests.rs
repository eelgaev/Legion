use super::*;

fn params(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        seed: Some(7),
        ..Default::default()
    }
}

/// Draw counts per token over `n` samples.
fn histogram(p: &SamplingParams, logits: &[f32], n: usize) -> Vec<usize> {
    let mut s = Sampler::new(p);
    let mut h = vec![0; logits.len()];
    for _ in 0..n {
        h[s.sample(&mut logits.to_vec()) as usize] += 1;
    }
    h
}

#[test]
fn rng_is_deterministic_and_uniform() {
    let a: Vec<u64> = (0..4)
        .scan(Rng::new(1), |r, _| Some(r.next_u64()))
        .collect();
    let b: Vec<u64> = (0..4)
        .scan(Rng::new(1), |r, _| Some(r.next_u64()))
        .collect();
    assert_eq!(a, b);
    let mut r = Rng::new(3);
    let mean = (0..100_000).map(|_| r.next_f64()).sum::<f64>() / 100_000.0;
    assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
}

#[test]
fn temperature_zero_is_greedy() {
    let logits = [0.5, 3.0, 2.9, -1.0];
    let mut s = Sampler::new(&SamplingParams {
        seed: None,
        ..params(0.0)
    });
    for _ in 0..10 {
        assert_eq!(s.sample(&mut logits.to_vec()), 1);
    }
}

#[test]
fn same_seed_same_tokens() {
    let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
    let run = || {
        let mut s = Sampler::new(&params(1.0));
        (0..32)
            .map(|_| s.sample(&mut logits.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
    let mut other = Sampler::new(&SamplingParams {
        seed: Some(8),
        ..params(1.0)
    });
    let third: Vec<u32> = (0..32).map(|_| other.sample(&mut logits.clone())).collect();
    assert_ne!(run(), third);
}

#[test]
fn samples_follow_softmax() {
    // p = softmax([0, ln 3]) = [0.25, 0.75].
    let logits = [0.0, 3f32.ln()];
    let h = histogram(&params(1.0), &logits, 20_000);
    let frac = h[1] as f64 / 20_000.0;
    assert!((frac - 0.75).abs() < 0.02, "{frac}");
    // Temperature 2 flattens to softmax([0, ln 3 / 2]) = [0.366, 0.634].
    let h = histogram(&params(2.0), &logits, 20_000);
    let frac = h[1] as f64 / 20_000.0;
    assert!((frac - 0.634).abs() < 0.02, "{frac}");
}

#[test]
fn top_k_keeps_the_k_best() {
    let logits = [1.0, 4.0, 3.0, 2.0, 0.0];
    let p = SamplingParams {
        top_k: 2,
        ..params(5.0)
    };
    let h = histogram(&p, &logits, 2000);
    assert_eq!(h[0] + h[3] + h[4], 0, "{h:?}");
    assert!(h[1] > 0 && h[2] > 0);
}

#[test]
fn top_p_keeps_the_nucleus() {
    // probs ~ [0.6, 0.3, 0.1]: top_p 0.8 keeps the first two.
    let logits = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
    let p = SamplingParams {
        top_p: 0.8,
        ..params(1.0)
    };
    let h = histogram(&p, &logits, 3000);
    assert_eq!(h[2], 0, "{h:?}");
    let frac = h[0] as f64 / 3000.0;
    assert!((frac - 2.0 / 3.0).abs() < 0.03, "{frac}");
    // A tiny top_p still keeps the best token.
    let p = SamplingParams {
        top_p: 1e-6,
        ..params(1.0)
    };
    assert_eq!(histogram(&p, &logits, 100)[0], 100);
}

#[test]
fn top_p_over_a_large_vocabulary() {
    // Flat over 5000 tokens: the first pass (1024) cannot cover 0.5, the full sort must.
    let logits = vec![0.0f32; 5000];
    let p = SamplingParams {
        top_p: 0.5,
        ..params(1.0)
    };
    let s = Sampler::new(&p);
    assert_eq!(s.candidates(&logits).len(), 2500);
    let s = Sampler::new(&SamplingParams { top_p: 0.6, ..p });
    // Peaked (p ~ [0.525, 0.475, tiny...]): the first pass covers it, both are needed.
    let mut logits = vec![-20.0f32; 5000];
    logits[10] = 0.0;
    logits[20] = -0.1;
    let n = s.candidates(&logits).len();
    assert_eq!(n, 2, "{n}");
}

#[test]
fn min_p_drops_unlikely_tokens() {
    // probs ∝ [1, 0.5, 0.05]: min_p 0.1 drops the last.
    let logits = [0.0, 0.5f32.ln(), 0.05f32.ln()];
    let p = SamplingParams {
        min_p: 0.1,
        ..params(1.0)
    };
    let h = histogram(&p, &logits, 3000);
    assert_eq!(h[2], 0, "{h:?}");
    assert!(h[1] > 0);
}

#[test]
fn penalties_push_seen_tokens_down() {
    let mut s = Sampler::new(&SamplingParams {
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        ..params(0.0)
    });
    s.accept(0);
    s.accept(0);
    s.accept(1);
    let mut logits = vec![2.0, 2.0, 1.5];
    assert_eq!(s.sample(&mut logits), 2);
    assert_eq!(logits, vec![2.0 - 0.5 - 0.5, 2.0 - 0.5 - 0.25, 1.5]);

    let mut s = Sampler::new(&SamplingParams {
        repetition_penalty: 2.0,
        ..params(0.0)
    });
    s.accept(0);
    s.accept(1);
    let mut logits = vec![3.0, -1.0, 1.6];
    assert_eq!(s.sample(&mut logits), 2);
    assert_eq!(logits, vec![1.5, -2.0, 1.6]);
}

#[test]
fn dist_is_what_sample_draws_from() {
    let logits = [1.0f32, 4.0, 3.0, 2.0, 0.0];
    let s = Sampler::new(&SamplingParams {
        top_k: 3,
        ..params(2.0)
    });
    let Dist::Probs(p) = s.dist(&mut logits.to_vec()) else {
        panic!("not a distribution");
    };
    assert_eq!(p.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert!((p.iter().map(|c| c.1).sum::<f64>() - 1.0).abs() < 1e-12);
    // softmax([4, 3, 2] / 2)
    let z = 1.0 + (-0.5f64).exp() + (-1.0f64).exp();
    assert!((p[0].1 - 1.0 / z).abs() < 1e-9);
    let h = histogram(
        &SamplingParams {
            top_k: 3,
            ..params(2.0)
        },
        &logits,
        20_000,
    );
    assert!((h[1] as f64 / 20_000.0 - p[0].1).abs() < 0.02);
    assert_eq!(Dist::Probs(p.clone()).prob(4), 0.0);

    // Temperature 0 with a penalty: the argmax after the penalty.
    let mut s = Sampler::new(&SamplingParams {
        presence_penalty: 2.0,
        ..params(0.0)
    });
    s.accept(1);
    assert_eq!(s.dist(&mut logits.to_vec()), Dist::Greedy(2));
    assert_eq!(Dist::Greedy(2).prob(2), 1.0);
}

#[test]
fn logprobs_are_normalized() {
    let logits = [1.0f32, 2.0, 3.0];
    let (lp, top) = logprobs(&logits, 1, 2);
    let lse = (1f64.exp() + 2f64.exp() + 3f64.exp()).ln();
    assert!((lp as f64 - (2.0 - lse)).abs() < 1e-6);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, 2);
    assert_eq!(top[1].0, 1);
    assert!(logprobs(&logits, 0, 0).1.is_empty());
}

/// The one-pass top-k path picks the same candidates, in the same order of probability, as
/// the general path (top-k 0 = off, then the k best kept by hand), with min-p and top-p.
#[test]
fn small_top_k_matches_the_general_path() {
    let mut rng = Rng::new(3);
    let logits: Vec<f32> = (0..5000)
        .map(|_| (rng.next_f64() * 20.0 - 10.0) as f32)
        .collect();
    for (k, min_p, top_p) in [
        (20, 0.0, 1.0),
        (20, 0.0, 0.95),
        (64, 0.05, 0.8),
        (1, 0.0, 1.0),
    ] {
        let fast = Sampler::new(&SamplingParams {
            top_k: k,
            min_p,
            top_p,
            ..params(1.0)
        });
        let got = fast.candidates(&logits);
        // The general path over the k best only.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let mut masked = vec![f32::NEG_INFINITY; logits.len()];
        for &i in &order[..k as usize] {
            masked[i] = logits[i];
        }
        let slow = Sampler::new(&SamplingParams {
            top_k: 0,
            min_p,
            top_p,
            ..params(1.0)
        });
        let want: Vec<(u32, f64)> = slow
            .candidates(&masked)
            .into_iter()
            .filter(|&(_, p)| p > 0.0)
            .collect();
        assert_eq!(got.len(), want.len(), "k {k} min_p {min_p} top_p {top_p}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.0, w.0);
            assert!((g.1 - w.1).abs() < 1e-12);
        }
    }
}

/// Ties at the top-k boundary keep the lower token ids.
#[test]
fn small_top_k_breaks_ties_by_token_id() {
    let s = Sampler::new(&SamplingParams {
        top_k: 2,
        ..params(1.0)
    });
    let ids: Vec<u32> = s
        .candidates(&[1.0, 3.0, 3.0, 3.0])
        .iter()
        .map(|c| c.0)
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

/// The radix sort orders a whole vocabulary as a comparison sort does (largest first, ties by
/// ascending id), including negative, zero, infinite and repeated logits.
#[test]
fn radix_sort_matches_a_comparison_sort() {
    let mut rng = Rng::new(9);
    let mut logits: Vec<f32> = (0..20_000)
        .map(|_| ((rng.next_f64() * 40.0 - 20.0) as f32 * 8.0).round() / 8.0)
        .collect();
    logits[3] = f32::NEG_INFINITY;
    logits[7] = f32::INFINITY;
    logits[11] = 0.0;
    logits[12] = -0.0;
    let mut got: Vec<u32> = (0..logits.len() as u32).collect();
    sort_by_logit(&mut got, &logits);
    let mut want: Vec<u32> = (0..logits.len() as u32).collect();
    want.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
    assert_eq!(got, want);
}

/// Sampling a whole vocabulary (the API's defaults): the same token for the same seed through
/// the radix path as through a comparison sort of the candidates.
#[test]
fn full_vocabulary_sampling_is_unchanged() {
    let mut rng = Rng::new(5);
    let logits: Vec<f32> = (0..50_000)
        .map(|_| (rng.next_f64() * 12.0 - 6.0) as f32)
        .collect();
    let p = SamplingParams {
        seed: Some(3),
        ..params(1.0)
    };
    let s = Sampler::new(&p);
    let got = s.candidates(&logits);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
    let want: Vec<(u32, f64)> = ids
        .into_iter()
        .map(|id| (id, (logits[id as usize] as f64 - max).exp()))
        .collect();
    assert_eq!(got, want);
}

/// The vectorized `exp` of the top-p fast path stays within the error its slack assumes.
#[test]
fn approx_exp_is_within_3e_13_of_libm() {
    let mut worst = 0.0f64;
    let mut x = -700.0f64;
    while x <= 0.0 {
        let want = x.exp();
        worst = worst.max(((approx_exp(x) - want) / want).abs());
        x += 0.000_937;
    }
    assert!(worst < 3e-13, "{worst:e}");
    assert_eq!(approx_exp(0.0), 1.0);
    assert!(approx_exp(f64::NEG_INFINITY) < 1e-300);
    assert!(approx_exp(f64::NAN).is_nan());
    // The lanes and the tail add up to the libm sum.
    let logits: Vec<f32> = (0..1_000).map(|i| -(i as f32) / 37.0).collect();
    let want: f64 = logits.iter().map(|&l| (l as f64).exp()).sum();
    let got = approx_exp_sum(&logits, 0.0);
    assert!(((got - want) / want).abs() < 1e-12, "{got} {want}");
}

/// A row of `n` logits: a normal bulk of spread `sd`, plus `peaks` tokens near the top.
fn row(rng: &mut Rng, n: usize, sd: f64, peaks: usize) -> Vec<f32> {
    let mut l: Vec<f32> = (0..n)
        .map(|_| {
            let (u, v) = (rng.next_f64().max(1e-300), rng.next_f64());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * sd) as f32
        })
        .collect();
    for _ in 0..peaks {
        let at = (rng.next_u64() % n as u64) as usize;
        l[at] = (8.0 * sd + rng.next_f64() * 4.0) as f32;
    }
    l
}

/// Top-p alone takes the fast path, which keeps the general path's tokens and probabilities
/// bit for bit, over peaked and flat rows, small and GLM-sized vocabularies, and the top-p
/// values in use; and it does decide almost every row itself.
#[test]
fn top_p_fast_path_matches_the_general_path() {
    let mut rng = Rng::new(695);
    let (mut rows, mut fast) = (0, 0);
    for n in [40usize, 3_000, 154_880] {
        for (sd, peaks) in [(3.0, 1), (3.0, 4), (1.0, 0), (0.3, 0), (6.0, 2)] {
            let logits = row(&mut rng, n, sd, peaks);
            for top_p in [1e-6f32, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999] {
                let s = Sampler::new(&SamplingParams {
                    top_p,
                    ..params(1.0)
                });
                let want = s.general_candidates(&logits);
                let got = s.candidates(&logits);
                assert_eq!(got.len(), want.len(), "n {n} sd {sd} top_p {top_p}");
                for (g, w) in got.iter().zip(&want) {
                    assert_eq!(g.0, w.0, "n {n} sd {sd} top_p {top_p}");
                    assert_eq!(g.1.to_bits(), w.1.to_bits(), "n {n} sd {sd} top_p {top_p}");
                }
                rows += 1;
                fast += usize::from(top_p_candidates(&logits, top_p as f64).is_some());
            }
        }
    }
    assert!(
        fast * 10 >= rows * 9,
        "{fast} of {rows} rows on the fast path"
    );
}

/// Masked (-inf) tokens, a NaN, or a run of equal logits at the cut: the same candidates.
#[test]
fn top_p_fast_path_edge_rows() {
    let mut rng = Rng::new(17);
    let s = Sampler::new(&SamplingParams {
        top_p: 0.95,
        ..params(1.0)
    });
    let mut masked = row(&mut rng, 20_000, 3.0, 2);
    for t in [0, 5, 19_999] {
        masked[t] = f32::NEG_INFINITY;
    }
    assert!(top_p_candidates(&masked, 0.95).is_some());
    assert_eq!(s.candidates(&masked), s.general_candidates(&masked));
    let mut nan = masked.clone();
    nan[7] = f32::NAN;
    assert!(top_p_candidates(&nan, 0.95).is_none());
    // Flat: the cut lands exactly on the target, which the general path decides.
    let flat = vec![0.0f32; 5_000];
    let s = Sampler::new(&SamplingParams {
        top_p: 0.5,
        ..params(1.0)
    });
    assert!(top_p_candidates(&flat, 0.5).is_none());
    assert_eq!(s.candidates(&flat).len(), 2_500);
    let all_masked = vec![f32::NEG_INFINITY; 100];
    assert!(top_p_candidates(&all_masked, 0.5).is_none());
}
