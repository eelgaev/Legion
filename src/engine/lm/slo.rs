//! `--itl-target` / `--ttft-target` (#628, part of #51): each step's prompt-token budget sized
//! from an online model of the step time, so decoding sequences keep their inter-token latency
//! while long prompts are read, and prompts whose time to first token is at risk go first.
//!
//! Without targets a step runs every decoding sequence plus prompt chunks up to
//! `--max-batch-tokens`: a long prompt arriving costs every decoding stream one long step per
//! chunk. With them ([`Slo`]):
//!
//! - **Step-time model** ([`StepModel`]): the time of a step with decoding rows, fitted online
//!   by least squares from the measured steps (each step's weight decays, so the fit follows
//!   the operating point), as a linear function of the decode / verify rows, the positions they
//!   attend to, whether a prompt chunk is in the step, its tokens and the positions they attend
//!   to. Its out-of-sample error (each step against the prediction made before it) is logged.
//! - **ITL budget**: with decoding sequences in the step, the most prompt tokens (in whole KV
//!   blocks) whose predicted step stays under `--itl-target`. Until the model has seen enough
//!   steps with prompt chunks, a probe: a few blocks, doubled after a step under the target and
//!   halved after one over. With nothing decoding the budget stays the full `max_batch_tokens`,
//!   so a prompt alone is read as fast as without targets.
//! - **TTFT** (`--ttft-target`): a prompt is at risk when, read in arrival order at the
//!   planned pace, it would not finish before its deadline (arrival plus the target). Prompts
//!   at risk get the budget first (the fewest tokens left first), within their
//!   `--scheduling-policy priority` class. With both targets, the budget is raised above the ITL
//!   budget as far as the at-risk prompts that can still make their deadline need; a prompt
//!   past its deadline (or one no budget can save) keeps its place but raises nothing.
//!
//! Only step boundaries move: every prompt token and decode row is computed as without
//! targets, in other chunks (see the determinism notes in `docs/serving.md`).

use std::time::{Duration, Instant};

use super::{Seq, Worker};
use crate::engine::queue::SchedulingPolicy;
use crate::model::core::KV_BLOCK;

/// `--itl-target` / `--ttft-target` in milliseconds (`None`: off).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SloTargets {
    pub itl_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
}

impl SloTargets {
    pub fn is_on(&self) -> bool {
        self.itl_ms.is_some() || self.ttft_ms.is_some()
    }
}

/// Refuses the targets where steps are not whole: decoding groups in flight over pipeline
/// stages (`--pipeline-groups` above 1, and the default over two machines, `--node-rank`) run
/// a group's step a stage at a time, beside other groups' steps, so no step time measures a
/// decoding stream's inter-token latency. A layer split without groups, tensor parallelism in
/// one box (`--tp`) and over machines (`--tp-rank`; rank 0 schedules every rank's steps) run
/// whole steps, which the model measures as they are.
pub(super) fn check(
    targets: SloTargets,
    pipeline_groups: usize,
    two_machines: bool,
) -> Result<(), String> {
    if !targets.is_on() {
        return Ok(());
    }
    if let Some(ms) = targets
        .itl_ms
        .into_iter()
        .chain(targets.ttft_ms)
        .find(|ms| !ms.is_finite() || *ms <= 0.0)
    {
        return Err(format!(
            "--itl-target / --ttft-target must be above 0 ms (got {ms})"
        ));
    }
    if pipeline_groups > 1 || (two_machines && pipeline_groups == 0) {
        return Err(
            "--itl-target / --ttft-target need whole steps: decoding groups in flight over the \
             pipeline stages (--pipeline-groups, two by default over two machines) are not \
             supported; pass --pipeline-groups 1 or drop the targets"
                .into(),
        );
    }
    Ok(())
}

/// The step budget's state for `targets` (`None`: off), announced in the log.
pub(super) fn start(targets: SloTargets) -> Option<Slo> {
    if !targets.is_on() {
        return None;
    }
    tracing::info!(
        itl_target_ms = targets.itl_ms,
        ttft_target_ms = targets.ttft_ms,
        "step budget from a step-time model: prompt tokens per step sized for the targets"
    );
    Some(Slo::new(targets))
}

/// What a step runs, as the step-time model reads it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Shape {
    /// Decode and verify rows: decoding sequences times the tokens each feeds.
    pub rows: usize,
    /// Over those rows, the positions each attends to.
    pub rows_ctx: f64,
    /// Prompt tokens.
    pub prefill: usize,
    /// Over the prompt tokens, the positions each attends to.
    pub prefill_ctx: f64,
}

impl Shape {
    /// `n` prompt tokens after `n_past` cached ones: the positions they attend to.
    pub fn prompt_ctx(n_past: usize, n: usize) -> f64 {
        n as f64 * (n_past as f64 + (n as f64 + 1.0) / 2.0)
    }
}

/// The model's terms.
const N: usize = 6;

/// A step's terms, scaled to about 1 at typical sizes so the fit stays well conditioned.
fn features(s: &Shape) -> [f64; N] {
    [
        1.0,
        s.rows as f64 / 16.0,
        s.rows_ctx / 1e6,
        if s.prefill > 0 { 1.0 } else { 0.0 },
        s.prefill as f64 / 1024.0,
        s.prefill_ctx / 1e8,
    ]
}

/// Each step's weight in the fit decays by this per later step (about the last 200 steps).
const DECAY: f64 = 0.995;
/// Ridge term: keeps the fit solvable while some terms have not varied yet.
const RIDGE: f64 = 1e-4;
/// Steps the model needs before it predicts: all, and with a prompt chunk (decayed counts).
const MIN_STEPS: f64 = 8.0;
const MIN_PREFILL_STEPS: f64 = 3.0;
/// A step further than this many RMS errors from the prediction weighs less (Huber), so a
/// one-off (a graph built, the expert cache lent) does not skew the fit.
const HUBER: f64 = 3.0;
/// The error scale [`HUBER`] multiplies is at least this share of the predicted step.
const HUBER_FLOOR: f64 = 0.05;

/// The step-time model: an online, exponentially weighted least-squares fit of the step time
/// in milliseconds to [`features`].
#[derive(Debug, Clone)]
pub(super) struct StepModel {
    /// Weighted sums of `x xᵀ` and `x y`.
    xx: [[f64; N]; N],
    xy: [f64; N],
    /// Decayed step counts: all, and those with a prompt chunk.
    n: f64,
    n_prefill: f64,
    coef: Option<[f64; N]>,
    /// The prediction errors made before each step's update: decayed sums of `|e|`, `|e| / y`,
    /// `e²`, and their count.
    err_abs: f64,
    err_rel: f64,
    err_sq: f64,
    err_n: f64,
}

impl Default for StepModel {
    fn default() -> Self {
        Self {
            xx: [[0.0; N]; N],
            xy: [0.0; N],
            n: 0.0,
            n_prefill: 0.0,
            coef: None,
            err_abs: 0.0,
            err_rel: 0.0,
            err_sq: 0.0,
            err_n: 0.0,
        }
    }
}

/// The model's fit error, out of sample (each step against the prediction made before it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FitError {
    pub mean_abs_ms: f64,
    pub mean_rel: f64,
    pub rms_ms: f64,
}

impl StepModel {
    /// Whether it has seen enough steps to predict.
    pub fn ready(&self) -> bool {
        self.coef.is_some() && self.n >= MIN_STEPS && self.n_prefill >= MIN_PREFILL_STEPS
    }

    /// The predicted time of a step of `s`, in milliseconds (once [`Self::ready`]).
    pub fn predict(&self, s: &Shape) -> Option<f64> {
        if !self.ready() {
            return None;
        }
        let x = features(s);
        let c = self.coef.as_ref()?;
        Some((0..N).map(|i| c[i] * x[i]).sum())
    }

    /// The fitted coefficients (milliseconds per scaled term), for the log.
    pub fn coef(&self) -> Option<[f64; N]> {
        self.coef
    }

    pub fn error(&self) -> Option<FitError> {
        (self.err_n > 0.0).then(|| FitError {
            mean_abs_ms: self.err_abs / self.err_n,
            mean_rel: self.err_rel / self.err_n,
            rms_ms: (self.err_sq / self.err_n).sqrt(),
        })
    }

    /// Adds a step of `s` that took `ms`.
    pub fn record(&mut self, s: &Shape, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let mut w = 1.0;
        if let Some(p) = self.predict(s) {
            let e = ms - p;
            // The error so far, at least a few percent of the step (exact fits would weigh
            // every new step down).
            let scale = self
                .error()
                .map_or(0.0, |f| f.rms_ms)
                .max(HUBER_FLOOR * p.abs())
                .max(1e-3);
            self.err_abs = self.err_abs * DECAY + e.abs();
            self.err_rel = self.err_rel * DECAY + e.abs() / ms.max(1e-3);
            self.err_sq = self.err_sq * DECAY + e * e;
            self.err_n = self.err_n * DECAY + 1.0;
            if e.abs() > HUBER * scale {
                w = HUBER * scale / e.abs();
            }
        }
        let x = features(s);
        for i in 0..N {
            for j in 0..N {
                self.xx[i][j] = self.xx[i][j] * DECAY + w * x[i] * x[j];
            }
            self.xy[i] = self.xy[i] * DECAY + w * x[i] * ms;
        }
        self.n = self.n * DECAY + 1.0;
        self.n_prefill = self.n_prefill * DECAY + if s.prefill > 0 { 1.0 } else { 0.0 };
        self.coef = solve(&self.xx, &self.xy);
    }
}

/// `(xx + RIDGE·I) c = xy` by Gaussian elimination with partial pivoting.
fn solve(xx: &[[f64; N]; N], xy: &[f64; N]) -> Option<[f64; N]> {
    let mut a = *xx;
    let mut b = *xy;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += RIDGE;
    }
    for col in 0..N {
        let pivot = (col..N).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let (top, rest) = a.split_at_mut(col + 1);
        let prow = &top[col];
        for (i, row) in rest.iter_mut().enumerate() {
            let f = row[col] / prow[col];
            if f != 0.0 {
                for (x, p) in row[col..].iter_mut().zip(&prow[col..]) {
                    *x -= f * p;
                }
                b[col + 1 + i] -= f * b[col];
            }
        }
    }
    let mut c = [0.0; N];
    for i in (0..N).rev() {
        let s: f64 = (i + 1..N).map(|k| a[i][k] * c[k]).sum();
        c[i] = (b[i] - s) / a[i][i];
    }
    c.iter().all(|v| v.is_finite()).then_some(c)
}

/// A prompt being read, as the planner sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Prompt {
    /// Tokens in its cache.
    pub n_past: usize,
    /// Tokens left to read.
    pub left: usize,
    /// The most it takes in one step (its chunk, up to its next save point).
    pub take: usize,
    /// Milliseconds to its TTFT deadline (negative: past it).
    pub due_ms: f64,
    /// `(priority class, arrival)`: its place in the queue's order (class 0 under FCFS).
    pub order: (i32, u64),
}

/// What [`Slo::plan`] decided for a step.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Plan {
    /// The prompts' order for this step (indices into the planner's input).
    pub order: Vec<usize>,
    /// Prompt tokens for the step.
    pub budget: usize,
    /// Prompts at risk of missing their TTFT target.
    pub at_risk: usize,
}

/// The first ITL probe budget, before the model is ready.
const PROBE_START: usize = 2 * KV_BLOCK;
/// The probe doubles after a step under this share of the ITL target.
const PROBE_GROW: f64 = 0.8;
/// TTFT deadlines are planned for this share of the time left (the model's error).
const DEADLINE_SHARE: f64 = 0.9;
/// The fit error and the budgets are logged this often while steps run.
const LOG_EVERY: Duration = Duration::from_secs(30);

/// The SLO-driven step budget (see the module docs).
#[derive(Debug)]
pub(super) struct Slo {
    targets: SloTargets,
    model: StepModel,
    /// Until the model is ready: the budget of steps with decoding rows.
    probe: usize,
    /// The step in flight's shape, set when it is planned.
    planned: Option<Shape>,
    /// Prompt reading pace over every step with a prompt chunk: decayed tokens and ms.
    pace_tokens: f64,
    pace_ms: f64,
    /// Since the last log line: steps with decoding rows, those with prompt tokens, those
    /// over the ITL target, their summed prompt budget and the most prompts at risk.
    logged: Instant,
    steps: u64,
    mixed: u64,
    over: u64,
    budget_sum: u64,
    at_risk_max: usize,
}

impl Slo {
    pub fn new(targets: SloTargets) -> Self {
        Self {
            targets,
            model: StepModel::default(),
            probe: PROBE_START,
            planned: None,
            pace_tokens: 0.0,
            pace_ms: 0.0,
            logged: Instant::now(),
            steps: 0,
            mixed: 0,
            over: 0,
            budget_sum: 0,
            at_risk_max: 0,
        }
    }

    pub fn targets(&self) -> SloTargets {
        self.targets
    }

    #[cfg(test)]
    pub(super) fn model(&self) -> &StepModel {
        &self.model
    }

    /// The prompt-token budget and the prompts' order for a step with `rows` decode / verify
    /// rows attending to `rows_ctx` positions, `prompts` in their current order, at most `cap`
    /// prompt tokens (the budget without targets).
    pub fn plan(&mut self, rows: usize, rows_ctx: f64, prompts: &[Prompt], cap: usize) -> Plan {
        let mut order: Vec<usize> = (0..prompts.len()).collect();
        if prompts.is_empty() {
            return Plan {
                order,
                budget: cap,
                at_risk: 0,
            };
        }
        let base = Shape {
            rows,
            rows_ctx,
            ..Shape::default()
        };
        let total: usize = prompts.iter().map(|p| p.take).sum();
        let cap = cap.min(total.max(1));
        let itl = match self.targets.itl_ms {
            Some(target) if rows > 0 => {
                if self.model.ready() {
                    self.itl_budget(&base, prompts, &order, cap, target)
                } else {
                    self.probe.min(cap)
                }
            }
            _ => cap,
        };
        let mut at_risk = 0;
        let mut budget = itl;
        if self.targets.ttft_ms.is_some() {
            let pace = self.pace(&base, prompts, &order, itl, rows);
            let mut by_arrival = order.clone();
            by_arrival.sort_by_key(|&i| prompts[i].order);
            let mut risky = vec![false; prompts.len()];
            let mut ahead = 0usize;
            for &i in &by_arrival {
                ahead += prompts[i].left;
                risky[i] = prompts[i].due_ms <= 0.0
                    || pace.is_some_and(|tok_per_ms| ahead as f64 / tok_per_ms > prompts[i].due_ms);
            }
            at_risk = risky.iter().filter(|&&r| r).count();
            // At risk and still in time first (the fewest tokens left first), then the rest in
            // arrival order, then those past their target: one that missed it already must not
            // make the next ones miss theirs too.
            order.sort_by_key(|&i| {
                let p = &prompts[i];
                let late = p.due_ms <= 0.0;
                let first = risky[i] && !late;
                (
                    p.order.0,
                    if first {
                        0
                    } else if late {
                        2
                    } else {
                        1
                    },
                    if first { p.left } else { 0 },
                    p.order.1,
                )
            });
            if rows > 0 && self.targets.itl_ms.is_some() && self.model.ready() && at_risk > 0 {
                budget = self.ttft_budget(&base, prompts, &order, &risky, itl, cap);
            }
        }
        if rows > 0 {
            self.at_risk_max = self.at_risk_max.max(at_risk);
        }
        Plan {
            order,
            budget,
            at_risk,
        }
    }

    /// The largest whole-block budget up to `cap` whose step the model puts under `target` ms
    /// (0 when even one block is over).
    fn itl_budget(
        &self,
        base: &Shape,
        prompts: &[Prompt],
        order: &[usize],
        cap: usize,
        target: f64,
    ) -> usize {
        let mut best = 0;
        for b in candidates(cap) {
            match self.model.predict(&with_prompts(base, prompts, order, b)) {
                Some(ms) if ms <= target => best = b,
                _ => break,
            }
        }
        best
    }

    /// Prompt tokens per millisecond at a budget of `budget`: the model's with decoding rows,
    /// else the pace measured over recent steps with a prompt chunk.
    fn pace(
        &self,
        base: &Shape,
        prompts: &[Prompt],
        order: &[usize],
        budget: usize,
        rows: usize,
    ) -> Option<f64> {
        if rows > 0 && budget > 0 {
            let s = with_prompts(base, prompts, order, budget);
            if let Some(ms) = self.model.predict(&s)
                && ms > 0.0
                && s.prefill > 0
            {
                return Some(s.prefill as f64 / ms);
            }
        }
        (self.pace_ms > 0.0 && self.pace_tokens > 0.0).then(|| self.pace_tokens / self.pace_ms)
    }

    /// The smallest budget from `itl` up to `cap` at which the at-risk prompts that can still
    /// make their deadlines (in `order`) read fast enough; `itl` when none can.
    fn ttft_budget(
        &self,
        base: &Shape,
        prompts: &[Prompt],
        order: &[usize],
        risky: &[bool],
        itl: usize,
        cap: usize,
    ) -> usize {
        let tries: Vec<(usize, f64)> = candidates(cap)
            .filter(|&b| b >= itl)
            .filter_map(|b| {
                let s = with_prompts(base, prompts, order, b);
                let ms = self.model.predict(&s)?;
                (ms > 0.0).then(|| (b, s.prefill as f64 / ms))
            })
            .collect();
        let fastest = tries.iter().map(|t| t.1).fold(0.0, f64::max);
        let mut need = 0.0f64;
        let mut ahead = 0usize;
        for &i in order {
            ahead += prompts[i].left;
            if !risky[i] || prompts[i].due_ms <= 0.0 {
                continue;
            }
            let rate = ahead as f64 / (prompts[i].due_ms * DEADLINE_SHARE);
            if rate <= fastest {
                need = need.max(rate);
            }
        }
        if need <= 0.0 {
            return itl;
        }
        tries
            .iter()
            .find(|t| t.1 >= need)
            .map_or(itl, |t| t.0.max(itl))
    }

    /// Sets the shape of the step about to run.
    pub fn planned(&mut self, s: Shape) {
        self.planned = Some(s);
    }

    /// The planned step ran in `elapsed` (`ok`: without an error): fits the model (steps with
    /// decoding rows), steers the probe, and logs the fit now and then.
    pub fn finished(&mut self, elapsed: Duration, ok: bool) {
        let Some(s) = self.planned.take() else {
            return;
        };
        if !ok {
            return;
        }
        let ms = elapsed.as_secs_f64() * 1e3;
        if s.prefill > 0 {
            self.pace_tokens = self.pace_tokens * DECAY + s.prefill as f64;
            self.pace_ms = self.pace_ms * DECAY + ms;
        }
        if s.rows > 0 {
            if let Some(target) = self.targets.itl_ms {
                if s.prefill > 0 && !self.model.ready() {
                    if ms <= target * PROBE_GROW {
                        self.probe = (self.probe * 2).min(1 << 20);
                    } else if ms > target {
                        self.probe = (self.probe / 2).max(KV_BLOCK);
                    }
                }
                if ms > target {
                    self.over += 1;
                }
            }
            self.model.record(&s, ms);
            self.steps += 1;
            if s.prefill > 0 {
                self.mixed += 1;
                self.budget_sum += s.prefill as u64;
            }
        }
        if self.logged.elapsed() >= LOG_EVERY {
            self.log();
        }
    }

    fn log(&mut self) {
        if self.steps > 0 {
            let err = self.model.error();
            tracing::info!(
                steps = self.steps,
                steps_with_prompts = self.mixed,
                over_itl_target = self.over,
                mean_prompt_tokens = (self.mixed > 0).then(|| self.budget_sum / self.mixed),
                prompts_at_risk_max = self.at_risk_max,
                model_ready = self.model.ready(),
                fit_mean_abs_ms = err.map(|e| round2(e.mean_abs_ms)),
                fit_mean_rel_pct = err.map(|e| round2(e.mean_rel * 100.0)),
                fit_rms_ms = err.map(|e| round2(e.rms_ms)),
                coef_ms = ?self.model.coef().map(|c| c.map(round2)),
                "step budget: step-time model (out-of-sample error of steps with decoding rows)"
            );
        }
        self.logged = Instant::now();
        self.steps = 0;
        self.mixed = 0;
        self.over = 0;
        self.budget_sum = 0;
        self.at_risk_max = 0;
    }
}

impl Worker {
    /// With targets: puts `prefilling` in the step's order and returns its prompt budget, at
    /// most `cap` (the budget without targets). Decoding sequences feed `n_tok` tokens each;
    /// a prompt takes at most `chunk` tokens.
    pub(super) fn slo_budget(
        &mut self,
        decoding: &[Seq<'_>],
        prefilling: &mut Vec<Seq<'_>>,
        n_tok: usize,
        chunk: usize,
        cap: usize,
    ) -> usize {
        let policy = self.scheduling;
        let Some(ttft) = self.slo.as_ref().map(|slo| slo.targets().ttft_ms) else {
            return cap;
        };
        if prefilling.is_empty() {
            return cap;
        }
        let now = Instant::now();
        let prompts: Vec<Prompt> = prefilling
            .iter()
            .map(|s| {
                let left = s.prompt.len() - s.n_past;
                // A save point the step saves inside the chunk does not end it (#46).
                let take = self.save_cut(s, s.n_past + chunk.min(left)) - s.n_past;
                // A preempted decoding sequence read again has streamed its first token: its
                // stream stalls until it is read, so it is due now (at risk, not late).
                let due_ms = ttft.map_or(f64::INFINITY, |t| {
                    if s.resuming {
                        f64::MIN_POSITIVE
                    } else {
                        t - now
                            .saturating_duration_since(s.active.arrived())
                            .as_secs_f64()
                            * 1e3
                    }
                });
                let (priority, arrival) = s.ticket.order();
                let class = match policy {
                    SchedulingPolicy::Fcfs => 0,
                    SchedulingPolicy::Priority => priority,
                };
                Prompt {
                    n_past: s.n_past,
                    left,
                    take,
                    due_ms,
                    order: (class, arrival),
                }
            })
            .collect();
        let rows = decoding.len() * n_tok;
        let rows_ctx = decoding
            .iter()
            .map(|s| Shape::prompt_ctx(s.n_past, n_tok))
            .sum();
        let slo = self.slo.as_mut().expect("checked above");
        let plan = slo.plan(rows, rows_ctx, &prompts, cap);
        if plan.order.iter().enumerate().any(|(i, &j)| i != j) {
            let mut seqs: Vec<Option<Seq<'_>>> = prefilling.drain(..).map(Some).collect();
            prefilling.extend(
                plan.order
                    .iter()
                    .map(|&i| seqs[i].take().expect("a permutation")),
            );
        }
        tracing::trace!(
            rows,
            prompts = prompts.len(),
            cap,
            budget = plan.budget,
            at_risk = plan.at_risk,
            "step budget"
        );
        plan.budget
    }

    /// With targets: the shape of the step just planned, for the step-time model.
    pub(super) fn slo_planned(
        &mut self,
        decoding: &[Seq<'_>],
        prefilling: &[Seq<'_>],
        n_tok: usize,
        chunks: &[(usize, usize)],
    ) {
        let Some(slo) = &mut self.slo else {
            return;
        };
        let mut s = Shape {
            rows: decoding.len() * n_tok,
            rows_ctx: decoding
                .iter()
                .map(|q| Shape::prompt_ctx(q.n_past, n_tok))
                .sum(),
            ..Shape::default()
        };
        for &(i, len) in chunks {
            s.prefill += len;
            s.prefill_ctx += Shape::prompt_ctx(prefilling[i].n_past, len);
        }
        slo.planned(s);
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Whole KV blocks up to `cap`, then `cap`.
fn candidates(cap: usize) -> impl Iterator<Item = usize> {
    (1..)
        .map(|k| k * KV_BLOCK)
        .take_while(move |&b| b < cap)
        .chain(std::iter::once(cap).filter(|&c| c > 0))
}

/// `base` plus the prompt chunks a budget of `budget` gives `prompts` in `order` (each takes up
/// to its `take`, as the step's planner hands them out).
fn with_prompts(base: &Shape, prompts: &[Prompt], order: &[usize], budget: usize) -> Shape {
    let mut s = *base;
    let mut left = budget;
    for &i in order {
        if left == 0 {
            break;
        }
        let p = &prompts[i];
        let n = p.take.min(left);
        s.prefill += n;
        s.prefill_ctx += Shape::prompt_ctx(p.n_past, n);
        left -= n;
    }
    s
}

#[cfg(test)]
mod tests;
