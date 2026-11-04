use super::*;

/// A step time linear in the model's terms: 30 ms, 2 ms per decode row, 1 ms per 100K row
/// positions, 8 ms for having a prompt chunk, 0.5 ms per prompt token, 1 ms per 10M prompt
/// positions.
fn truth(s: &Shape) -> f64 {
    30.0 + 2.0 * s.rows as f64
        + s.rows_ctx / 1e5
        + if s.prefill > 0 { 8.0 } else { 0.0 }
        + 0.5 * s.prefill as f64
        + s.prefill_ctx / 1e7
}

/// A deterministic sequence in `0..n`.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

fn random_shape(r: &mut Lcg) -> Shape {
    let rows = 1 + r.below(24);
    let ctx = 200 + r.below(20000);
    let prefill = if r.below(3) == 0 { 0 } else { r.below(2048) };
    let past = r.below(30000);
    Shape {
        rows,
        rows_ctx: (rows * ctx) as f64,
        prefill,
        prefill_ctx: Shape::prompt_ctx(past, prefill),
    }
}

fn run(slo: &mut Slo, s: Shape, ms: f64) {
    slo.planned(s);
    slo.finished(Duration::from_secs_f64(ms / 1e3), true);
}

/// A model trained on `steps` random steps of [`truth`].
fn trained(targets: SloTargets, steps: usize) -> Slo {
    let mut slo = Slo::new(targets);
    let mut r = Lcg(7);
    for _ in 0..steps {
        let s = random_shape(&mut r);
        run(&mut slo, s, truth(&s));
    }
    slo
}

fn itl(ms: f64) -> SloTargets {
    SloTargets {
        itl_ms: Some(ms),
        ttft_ms: None,
    }
}

fn both(itl_ms: f64, ttft_ms: f64) -> SloTargets {
    SloTargets {
        itl_ms: Some(itl_ms),
        ttft_ms: Some(ttft_ms),
    }
}

fn prompt(n_past: usize, left: usize, due_ms: f64, arrival: u64) -> Prompt {
    Prompt {
        n_past,
        left,
        take: left,
        due_ms,
        order: (0, arrival),
    }
}

#[test]
fn the_model_recovers_a_linear_step_time() {
    let slo = trained(itl(100.0), 400);
    let m = slo.model();
    assert!(m.ready());
    let mut r = Lcg(99);
    for _ in 0..50 {
        let s = random_shape(&mut r);
        let p = m.predict(&s).unwrap();
        assert!((p - truth(&s)).abs() < 0.05, "{s:?}: {p} vs {}", truth(&s));
    }
    // Exact data: the out-of-sample error falls to nothing once the fit has settled.
    let e = m.error().unwrap();
    assert!(e.mean_rel < 0.05, "{e:?}");
}

#[test]
fn the_model_waits_for_steps_with_prompts() {
    let mut slo = Slo::new(itl(100.0));
    for i in 0..20 {
        let s = Shape {
            rows: 4 + i % 3,
            rows_ctx: 4000.0,
            ..Shape::default()
        };
        run(&mut slo, s, truth(&s));
    }
    assert!(!slo.model().ready(), "no step with a prompt chunk yet");
    let s = Shape {
        rows: 4,
        rows_ctx: 4000.0,
        prefill: 256,
        prefill_ctx: Shape::prompt_ctx(0, 256),
    };
    for _ in 0..3 {
        run(&mut slo, s, truth(&s));
    }
    // Three decayed steps weigh just under three.
    assert!(!slo.model().ready());
    run(&mut slo, s, truth(&s));
    assert!(slo.model().ready());
}

#[test]
fn an_outlier_moves_the_fit_little() {
    let mut slo = trained(itl(100.0), 400);
    let s = Shape {
        rows: 8,
        rows_ctx: 16000.0,
        prefill: 512,
        prefill_ctx: Shape::prompt_ctx(1000, 512),
    };
    let before = slo.model().predict(&s).unwrap();
    // A graph built or the expert cache lent: one step 2 s long.
    run(&mut slo, s, 2000.0);
    let after = slo.model().predict(&s).unwrap();
    assert!((after - before).abs() < 5.0, "{before} -> {after}");
}

#[test]
fn collinear_steps_still_solve() {
    // Every step the same shape: the ridge term keeps the fit solvable, and it predicts that
    // shape.
    let mut slo = Slo::new(itl(100.0));
    let s = Shape {
        rows: 4,
        rows_ctx: 4000.0,
        prefill: 256,
        prefill_ctx: Shape::prompt_ctx(0, 256),
    };
    for _ in 0..20 {
        run(&mut slo, s, 200.0);
    }
    assert!((slo.model().predict(&s).unwrap() - 200.0).abs() < 1.0);
}

#[test]
fn the_itl_budget_is_the_most_whole_blocks_under_the_target() {
    let mut slo = trained(itl(200.0), 400);
    let rows = 4;
    let prompts = [prompt(0, 16384, f64::INFINITY, 0)];
    let plan = slo.plan(rows, 4000.0, &prompts, 4096);
    assert_eq!(plan.budget % KV_BLOCK, 0);
    let at = |b: usize| {
        truth(&with_prompts(
            &Shape {
                rows,
                rows_ctx: 4000.0,
                ..Shape::default()
            },
            &prompts,
            &[0],
            b,
        ))
    };
    assert!(at(plan.budget) <= 200.0 + 0.1, "{}", at(plan.budget));
    assert!(at(plan.budget + KV_BLOCK) > 200.0 - 0.1);
    // 30 + 8 + 8 + 0.04 + 0.5 b (+ attention) <= 200: b = 256 (320 would be 200.0+).
    assert_eq!(plan.budget, 256);
}

#[test]
fn nothing_decoding_keeps_the_whole_budget() {
    let mut slo = trained(both(50.0, 1.0), 400);
    let prompts = [prompt(0, 100_000, -5.0, 0)];
    assert_eq!(slo.plan(0, 0.0, &prompts, 4096).budget, 4096);
    // A short prompt: no more than it takes.
    let prompts = [prompt(0, 700, 1e9, 0)];
    assert_eq!(slo.plan(0, 0.0, &prompts, 4096).budget, 700);
}

#[test]
fn a_target_below_the_decode_step_leaves_no_budget() {
    // The step without prompts already takes 30 + 2*16 ms: the planner leaves the step's
    // minimum progress to the caller.
    let mut slo = trained(itl(50.0), 400);
    let prompts = [prompt(0, 5000, f64::INFINITY, 0)];
    assert_eq!(slo.plan(16, 16000.0, &prompts, 4096).budget, 0);
}

#[test]
fn the_probe_doubles_under_the_target_and_halves_over_it() {
    let mut slo = Slo::new(itl(100.0));
    let prompts = [prompt(0, 10_000, f64::INFINITY, 0)];
    let mixed = |b: usize| Shape {
        rows: 4,
        rows_ctx: 400.0,
        prefill: b,
        prefill_ctx: Shape::prompt_ctx(0, b),
    };
    assert_eq!(slo.plan(4, 400.0, &prompts, 4096).budget, PROBE_START);
    run(&mut slo, mixed(PROBE_START), 50.0);
    assert_eq!(slo.plan(4, 400.0, &prompts, 4096).budget, 2 * PROBE_START);
    run(&mut slo, mixed(2 * PROBE_START), 150.0);
    assert_eq!(slo.plan(4, 400.0, &prompts, 4096).budget, PROBE_START);
    // Between 80% and 100% of the target: kept.
    run(&mut slo, mixed(PROBE_START), 90.0);
    assert!(!slo.model().ready());
    assert_eq!(slo.plan(4, 400.0, &prompts, 4096).budget, PROBE_START);
}

#[test]
fn without_ttft_target_the_order_stays() {
    let mut slo = trained(itl(200.0), 400);
    let prompts = [prompt(0, 16000, -1.0, 0), prompt(0, 100, -1.0, 1)];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.order, vec![0, 1]);
    assert_eq!(plan.at_risk, 0);
}

#[test]
fn a_short_prompt_behind_a_long_one_goes_first() {
    // ITL budget 256 tokens at about 200 ms a step: 1.3 tokens per ms. The 16K prompt that
    // arrived first needs 12.6 s; the chat behind it would wait as long.
    let mut slo = trained(both(200.0, 3000.0), 400);
    let prompts = [prompt(0, 16384, 2500.0, 0), prompt(0, 512, 2900.0, 1)];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.order, vec![1, 0]);
    assert_eq!(plan.at_risk, 2);
    // The chat can make it at the ITL pace once first; the 16K prompt cannot make 2.5 s at
    // any budget: nothing raises the budget.
    assert_eq!(plan.budget, 256);
}

#[test]
fn an_at_risk_prompt_that_can_make_it_raises_the_budget() {
    // 4,000 tokens left in 2 s: 2 tokens per ms (at 90% of the time left), above the ITL pace
    // (256 tokens in ~200 ms) but below the fastest step (~1.9-2 tokens per ms at 4096).
    let mut slo = trained(both(200.0, 30_000.0), 400);
    let prompts = [prompt(0, 3000, 2000.0, 0)];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.at_risk, 1);
    assert!(plan.budget > 256, "{}", plan.budget);
    let s = with_prompts(
        &Shape {
            rows: 4,
            rows_ctx: 4000.0,
            ..Shape::default()
        },
        &prompts,
        &[0],
        plan.budget,
    );
    let pace = s.prefill as f64 / truth(&s);
    assert!(pace * 2000.0 * DEADLINE_SHARE >= 3000.0 * 0.99, "{pace}");
    // The smallest budget that does: one block less is too slow.
    let less = with_prompts(
        &Shape {
            rows: 4,
            rows_ctx: 4000.0,
            ..Shape::default()
        },
        &prompts,
        &[0],
        plan.budget - KV_BLOCK,
    );
    assert!(less.prefill as f64 / truth(&less) * 2000.0 * DEADLINE_SHARE < 3000.0 * 1.01);
}

#[test]
fn a_prompt_past_its_deadline_raises_nothing() {
    let mut slo = trained(both(200.0, 1000.0), 400);
    let prompts = [prompt(0, 3000, -10.0, 0)];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.at_risk, 1);
    assert_eq!(plan.budget, 256);
}

#[test]
fn a_prompt_past_its_deadline_waits_behind_those_still_in_time() {
    // Ahead of the 3,000-token prompt that can make 2 s at a raised budget, the late one's
    // 2,000 tokens would leave it no budget fast enough.
    let mut slo = trained(both(200.0, 30_000.0), 400);
    let prompts = [
        prompt(0, 2000, -10.0, 0),
        prompt(0, 3000, 2000.0, 1),
        prompt(0, 512, 1e6, 2),
    ];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.order, vec![1, 2, 0]);
    assert_eq!(plan.at_risk, 2);
    assert!(plan.budget > 256, "{}", plan.budget);
}

#[test]
fn priority_classes_come_before_risk() {
    let mut slo = trained(both(200.0, 3000.0), 400);
    let mut low = prompt(0, 512, -1.0, 1);
    low.order.0 = 5;
    let high = prompt(0, 16384, 1e9, 0);
    let plan = slo.plan(4, 4000.0, &[low, high], 4096);
    assert_eq!(plan.order, vec![1, 0]);
}

#[test]
fn ttft_alone_orders_but_keeps_the_budget() {
    let mut slo = trained(
        SloTargets {
            itl_ms: None,
            ttft_ms: Some(1000.0),
        },
        400,
    );
    let prompts = [prompt(0, 16384, 900.0, 0), prompt(0, 64, 950.0, 1)];
    let plan = slo.plan(4, 4000.0, &prompts, 4096);
    assert_eq!(plan.order, vec![1, 0]);
    assert_eq!(plan.budget, 4096);
}

#[test]
fn candidates_are_whole_blocks_then_the_cap() {
    assert_eq!(candidates(200).collect::<Vec<_>>(), vec![64, 128, 192, 200]);
    assert_eq!(candidates(128).collect::<Vec<_>>(), vec![64, 128]);
    assert_eq!(candidates(10).collect::<Vec<_>>(), vec![10]);
    assert_eq!(candidates(0).count(), 0);
}

#[test]
fn prompts_share_a_budget_in_order_up_to_their_take() {
    let base = Shape::default();
    let mut a = prompt(100, 1000, 0.0, 0);
    a.take = 300;
    let b = prompt(0, 50, 0.0, 1);
    let s = with_prompts(&base, &[a, b], &[0, 1], 1000);
    assert_eq!(s.prefill, 350);
    assert_eq!(
        s.prefill_ctx,
        Shape::prompt_ctx(100, 300) + Shape::prompt_ctx(0, 50)
    );
    assert_eq!(with_prompts(&base, &[a, b], &[1, 0], 100).prefill, 100);
}

#[test]
fn the_targets_need_whole_steps() {
    let off = SloTargets::default();
    assert!(check(off, 4, true).is_ok());
    let on = itl(100.0);
    assert!(check(on, 0, false).is_ok());
    assert!(check(on, 1, false).is_ok());
    assert!(check(on, 1, true).is_ok());
    let e = check(on, 2, false).unwrap_err();
    assert!(e.contains("--pipeline-groups"), "{e}");
    assert!(
        check(on, 0, true).is_err(),
        "two machines default to groups"
    );
    assert!(check(itl(0.0), 0, false).is_err());
}
