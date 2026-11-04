//! Exact acceptance of draft tokens against the target model's rows of a verify step.
//!
//! [`verify`] walks the verify rows in order. Row `i` is the target's prediction after the
//! sequence plus drafts `d1..di`; its distribution `p_i` comes from [`Target::dist`], which the
//! caller computes with the sampler state of the tokens appended so far (penalties apply per
//! position). Greedy rows accept a draft while it equals the row's pick; sampled rows use
//! speculative rejection sampling, so the appended tokens are distributed exactly as plain
//! sampling would produce them (Leviathan et al., "Fast Inference from Transformers via
//! Speculative Decoding", 2023, Algorithm 1; Chen et al., "Accelerating Large Language Model
//! Decoding with Speculative Sampling", 2023).

pub use crate::engine::sampling::Dist;

/// The target side of one sequence's verify step.
pub trait Target {
    /// The distribution at verify row `i` (0: after the sequence's last token; `i`: after
    /// draft `i` too), given every token [`Self::push`]ed so far.
    fn dist(&mut self, i: usize) -> Dist;
    /// A uniform draw in `[0, 1)`.
    fn uniform(&mut self) -> f64;
    /// A token was appended to the sequence (an accepted draft, or the final pick).
    fn push(&mut self, token: u32);
}

/// What one verify step appended.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// Accepted drafts, then the token picked at the first rejected row (or the bonus token
    /// after the last draft), unless the sequence stopped on an accepted draft.
    pub appended: Vec<u32>,
    /// Drafts accepted (a prefix of the drafts).
    pub accepted: usize,
}

/// Verifies `drafts` against `target`'s rows `0..=drafts.len()`. `probs` is the drafter's
/// distribution at each draft (full vocabulary rows; empty: point masses). Stops after a token
/// for which `stop` holds (an end-of-generation token: nothing after it is used).
pub fn verify(
    target: &mut impl Target,
    drafts: &[u32],
    probs: &[Vec<f32>],
    stop: impl Fn(u32) -> bool,
) -> Outcome {
    assert!(
        probs.is_empty() || probs.len() == drafts.len(),
        "one draft distribution per draft"
    );
    let mut out = Outcome::default();
    for i in 0..=drafts.len() {
        let p = target.dist(i);
        let draft = drafts.get(i).copied();
        let (token, accepted) = match (&p, draft) {
            (Dist::Greedy(t), d) => (*t, d == Some(*t)),
            (Dist::Probs(p), None) => (draw(p.iter().copied(), target.uniform()), false),
            (Dist::Probs(p), Some(d)) => {
                let q = probs.get(i).map(|q| &q[..]);
                let qd = q.map_or(1.0, |q| {
                    f64::from(q.get(d as usize).copied().unwrap_or(0.0))
                });
                let pd = p
                    .iter()
                    .find(|&&(id, _)| id == d)
                    .map_or(0.0, |&(_, pr)| pr);
                // Accept with probability min(1, p(d) / q(d)).
                if qd > 0.0 && target.uniform() * qd < pd {
                    (d, true)
                } else {
                    (residual(p, q, d, target.uniform()), false)
                }
            }
        };
        target.push(token);
        out.appended.push(token);
        if accepted {
            out.accepted += 1;
        }
        if !accepted || stop(token) {
            break;
        }
    }
    out
}

/// Draws from `norm(max(0, p - q))` with the uniform `u` (`q`: a point mass at `d` when `None`).
/// If that is empty (only when `p <= q` everywhere, in which case the draft is always accepted
/// and this is never reached except by rounding), draws from `p`.
fn residual(p: &[(u32, f64)], q: Option<&[f32]>, d: u32, u: f64) -> u32 {
    let rest = |&(id, pr): &(u32, f64)| {
        let qi = match q {
            Some(q) => f64::from(q.get(id as usize).copied().unwrap_or(0.0)),
            None => f64::from(u8::from(id == d)),
        };
        (id, (pr - qi).max(0.0))
    };
    let total: f64 = p.iter().map(|c| rest(c).1).sum();
    if total > 0.0 {
        draw(p.iter().map(rest).map(|(id, w)| (id, w / total)), u)
    } else {
        draw(p.iter().copied(), u)
    }
}

/// The token at which the cumulative probability passes `u` (the last one with weight on
/// rounding).
fn draw(p: impl Iterator<Item = (u32, f64)>, u: f64) -> u32 {
    let mut last = None;
    let mut r = u;
    for (id, pr) in p {
        if pr <= 0.0 {
            continue;
        }
        if r < pr {
            return id;
        }
        r -= pr;
        last = Some(id);
    }
    last.expect("an empty distribution")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sampling::Rng;

    /// Rows with fixed distributions (independent of what was appended), seeded draws.
    struct Fixed {
        rows: Vec<Dist>,
        rng: Rng,
        pushed: Vec<u32>,
    }

    impl Target for Fixed {
        fn dist(&mut self, i: usize) -> Dist {
            self.rows[i].clone()
        }
        fn uniform(&mut self) -> f64 {
            self.rng.next_f64()
        }
        fn push(&mut self, token: u32) {
            self.pushed.push(token);
        }
    }

    fn fixed(rows: Vec<Dist>, seed: u64) -> Fixed {
        Fixed {
            rows,
            rng: Rng::new(seed),
            pushed: Vec::new(),
        }
    }

    #[test]
    fn greedy_takes_the_longest_matching_prefix_and_the_bonus() {
        let rows = vec![
            Dist::Greedy(5),
            Dist::Greedy(6),
            Dist::Greedy(7),
            Dist::Greedy(8),
        ];
        let never = |_| false;
        let mut t = fixed(rows.clone(), 0);
        let o = verify(&mut t, &[5, 6, 7], &[], never);
        assert_eq!(o.appended, vec![5, 6, 7, 8]);
        assert_eq!(o.accepted, 3);
        assert_eq!(t.pushed, o.appended);

        let o = verify(&mut fixed(rows.clone(), 0), &[5, 9, 7], &[], never);
        assert_eq!((o.appended, o.accepted), (vec![5, 6], 1));
        let o = verify(&mut fixed(rows.clone(), 0), &[1, 6, 7], &[], never);
        assert_eq!((o.appended, o.accepted), (vec![5], 0));
        let o = verify(&mut fixed(rows.clone(), 0), &[], &[], never);
        assert_eq!((o.appended, o.accepted), (vec![5], 0));
        // An accepted stop token ends the step.
        let o = verify(&mut fixed(rows, 0), &[5, 6, 7], &[], |t| t == 6);
        assert_eq!((o.appended, o.accepted), (vec![5, 6], 2));
    }

    fn probs(p: &[f64]) -> Dist {
        Dist::Probs(p.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect())
    }

    /// Asserts that `counts` (over `n` draws) matches `p` within 5 standard deviations per
    /// token, and that the chi-square statistic is plausible.
    fn assert_matches(counts: &[usize], p: &[f64], n: usize, what: &str) {
        let mut chi2 = 0.0;
        let mut dof = 0;
        for (i, (&c, &pi)) in counts.iter().zip(p).enumerate() {
            let expect = pi * n as f64;
            let sd = (n as f64 * pi * (1.0 - pi)).sqrt().max(1.0);
            assert!(
                (c as f64 - expect).abs() < 5.0 * sd,
                "{what}: token {i}: {c} draws, expected {expect:.0} (counts {counts:?})"
            );
            if expect > 0.0 {
                chi2 += (c as f64 - expect).powi(2) / expect;
                dof += 1;
            } else {
                assert_eq!(c, 0, "{what}: token {i} has probability 0");
            }
        }
        // chi2 with dof - 1 degrees of freedom: mean dof - 1, far tail below ~ 3 * dof + 20.
        assert!(
            chi2 < (3 * dof + 20) as f64,
            "{what}: chi2 {chi2:.1}, dof {dof}"
        );
    }

    #[test]
    fn rejection_sampling_with_point_drafts_keeps_the_target_distribution() {
        // Two verify rows with different distributions; drafts proposed by a fixed rule (as
        // a model-free drafter would). The first appended token must follow p0; given that
        // the first draft was accepted, the second must follow p1.
        let p0 = [0.5, 0.3, 0.15, 0.05, 0.0];
        let p1 = [0.1, 0.1, 0.2, 0.25, 0.35];
        let p2 = [0.2, 0.2, 0.2, 0.2, 0.2];
        let n = 200_000;
        for drafts in [[1u32, 4], [0, 0], [3, 2], [4, 1]] {
            let mut first = [0usize; 5];
            let mut second = [0usize; 5];
            let mut n_second = 0;
            let mut t = fixed(vec![probs(&p0), probs(&p1), probs(&p2)], 11);
            for _ in 0..n {
                let o = verify(&mut t, &drafts, &[], |_| false);
                first[o.appended[0] as usize] += 1;
                if o.accepted >= 1 {
                    second[o.appended[1] as usize] += 1;
                    n_second += 1;
                }
            }
            assert_matches(&first, &p0, n, &format!("drafts {drafts:?}, row 0"));
            assert_matches(&second, &p1, n_second, &format!("drafts {drafts:?}, row 1"));
        }
    }

    #[test]
    fn rejection_sampling_with_draft_distributions_keeps_the_target_distribution() {
        // The drafter samples its draft from q; acceptance is min(1, p/q), the residual
        // norm(max(0, p - q)).
        let p = [0.4, 0.05, 0.3, 0.25, 0.0];
        let p_next = [0.2, 0.2, 0.2, 0.2, 0.2];
        for q in [
            [0.1f32, 0.6, 0.1, 0.1, 0.1],
            [0.4, 0.05, 0.3, 0.25, 0.0], // q == p: always accepted
            [0.0, 0.0, 0.0, 0.0, 1.0],   // always rejected
        ] {
            let n = 200_000;
            let mut rng = Rng::new(5);
            let mut t = fixed(vec![probs(&p), probs(&p_next)], 3);
            let mut counts = [0usize; 5];
            let mut accepted = 0;
            for _ in 0..n {
                let d = draw(
                    q.iter().enumerate().map(|(i, &x)| (i as u32, f64::from(x))),
                    rng.next_f64(),
                );
                let o = verify(&mut t, &[d], &[q.to_vec()], |_| false);
                counts[o.appended[0] as usize] += 1;
                accepted += o.accepted;
            }
            assert_matches(&counts, &p, n, &format!("q {q:?}"));
            // Acceptance probability is sum_x min(p(x), q(x)).
            let expect: f64 = p.iter().zip(&q).map(|(&a, &b)| a.min(f64::from(b))).sum();
            let rate = accepted as f64 / n as f64;
            assert!(
                (rate - expect).abs() < 0.01,
                "q {q:?}: rate {rate}, expected {expect}"
            );
        }
    }

    #[test]
    fn residual_without_mass_falls_back_to_p() {
        // p == q, the draft rejected only by rounding: still a valid token from p.
        let p = [(0u32, 0.5), (1, 0.5)];
        let q = [0.5f32, 0.5];
        assert_eq!(residual(&p, Some(&q), 0, 0.7), 1);
        assert_eq!(residual(&p, None, 0, 0.99), 1);
        assert_eq!(residual(&p, None, 1, 0.01), 0);
    }
}
