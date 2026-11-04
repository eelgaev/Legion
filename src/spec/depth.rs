//! Draft depth for MTP heads (#280) and block drafters (DFlash 2, #521) from measured step
//! costs.
//!
//! A step that drafts `k` tokens per sequence verifies `1 + k` rows per sequence and runs the
//! draft head up to `k` times (a head with its own cache also takes in the positions just
//! verified, so it runs once at `k = 0` too). Each sequence `i` would gain `G_i(k)` tokens
//! (estimated from its own steps, [`crate::spec::adaptive::AdaptiveState::gain`]); the depth for
//! the next step maximizes the batch's expected tokens per millisecond,
//!
//! `sum_i G_i(k) / (V(b (1 + k)) + D(b) k)`,
//!
//! where `V(r)` is the measured time of a target step of `r` rows and `D(b)` that of one draft
//! pass over `b` sequences. Both are measured on the fly (exponential averages), so the depth
//! follows the model and the device: a dense target whose verify step costs the same for 1 or
//! 7 rows drafts deep, a MoE whose verify step grows with every row (more experts) drafts
//! shallow, and a jump in `V` (another matmul kernel past some row count) stops the depth below
//! it. A block drafter ([`DepthModel::block`]) drafts any depth in one pass: its term is `D(b)`.
//!
//! Rows never measured are interpolated between measured neighbours, or held at the nearest
//! measured value (optimistic above the largest: the policy goes there and measures it). Every
//! [`PROBE_EVERY`]th choice tries a depth next to the best one, so neighbours get measured and
//! drifting costs (a growing context) are followed. A new choice must beat the current one by
//! [`HYSTERESIS`], so noise does not flip depths (and the verify graph's shape) every step, and
//! the depth climbs one step per choice (it drops at once): a brief run of accepted drafts does
//! not buy a deep, expensive step.
//!
//! Step and pass costs grow with the context (attention reads every cached key): they are
//! kept per context bucket ([`DepthModel::set_context`]: under 4K positions, then doubling),
//! so a long prompt's steps do not make every depth look dear to the short chats after it
//! (and a depth not visited since would stay dear until probed).
//!
//! Within the chosen cap, the head keeps drafting while its probability for the draft pays for
//! the pass ([`DepthModel::confidence_floor`]): `--spec-p-min` becomes a ceiling. A dense
//! target's tokens are expensive next to a pass, so its drafts go on at low confidence.
//!
//! A draft the head stops short of leaves a padding row in the verify step (its shape is the
//! chosen depth's), and padding repeats the sequence's last token, so on a MoE it reads few
//! experts the real rows did not. A drafted row costs more than a padding row: at each shape,
//! the step time is fitted against the rows that carry drafts ([`DepthModel::record_fill`]), and
//! the slope is charged to every further draft next to the pass (`--spec-row-cost`, #836).

/// Relative weight of a new sample in the step and pass averages.
const EWMA: f64 = 0.2;
/// Every this many choices, a neighbouring depth instead of the best one.
pub const PROBE_EVERY: usize = 16;
/// Factor by which a depth's predicted throughput must beat the current depth's to replace it.
pub const HYSTERESIS: f64 = 1.02;
/// Depth before anything is measured.
const FIRST_DEPTH: usize = 2;
/// Most a sample counts above the mean (see [`Mean::add`]).
const SPIKE: f64 = 1.25;
/// Lowest head confidence [`DepthModel::confidence_floor`] lets a draft through at.
pub const MIN_CONFIDENCE: f32 = 0.05;
/// Head confidence buckets of [`Calibration`].
const BUCKETS: usize = 10;
/// Per draft judged, the weight a confidence bucket keeps of what it knew before.
const CALIBRATION_DECAY: f32 = 0.995;

/// How often the head's drafts are accepted, by its confidence for them (buckets of 0.1),
/// counting only drafts whose predecessors were accepted. Starts calibrated (acceptance = the
/// bucket's middle, worth [`PRIOR_DRAFTS`] drafts): buckets below the floor are never tried, so a
/// head found less reliable than it claims raises the floor for good, while one more reliable
/// than its prior only lowers it as far as the floor has been.
#[derive(Debug, Clone)]
struct Calibration {
    accepted: [f32; BUCKETS],
    judged: [f32; BUCKETS],
}

/// Weight of [`Calibration`]'s prior, in drafts per bucket.
const PRIOR_DRAFTS: f32 = 4.0;

impl Default for Calibration {
    fn default() -> Self {
        Self {
            accepted: std::array::from_fn(|b| PRIOR_DRAFTS * (b as f32 + 0.5) / BUCKETS as f32),
            judged: [PRIOR_DRAFTS; BUCKETS],
        }
    }
}

impl Calibration {
    fn bucket(confidence: f32) -> usize {
        ((confidence.clamp(0.0, 1.0) * BUCKETS as f32) as usize).min(BUCKETS - 1)
    }

    fn record(&mut self, confidence: f32, accepted: bool) {
        let b = Self::bucket(confidence);
        self.accepted[b] = CALIBRATION_DECAY * self.accepted[b] + f32::from(u8::from(accepted));
        self.judged[b] = CALIBRATION_DECAY * self.judged[b] + 1.0;
    }

    /// The lowest confidence from which every bucket's acceptance is at least `p`.
    fn floor(&self, p: f32) -> f32 {
        let mut b = BUCKETS;
        while b > 0 && self.accepted[b - 1] >= p * self.judged[b - 1] {
            b -= 1;
        }
        b as f32 / BUCKETS as f32
    }
}

/// The costs measured in one context bucket.
#[derive(Debug, Clone, Default)]
struct Costs {
    step: Vec<Mean>,
    pass: Vec<Mean>,
    /// By verify shape (rows), step time over the rows that carry drafts.
    fill: Vec<Fill>,
}

/// Per sample, the weight a [`Fill`] keeps of what it knew before (about 50 steps).
const FILL_DECAY: f64 = 0.98;
/// Samples (decayed) a [`Fill`] needs before its slope counts.
const FILL_MIN: f64 = 8.0;
/// Least variance of the drafted rows a [`Fill`]'s slope needs (rows squared): steps that all
/// drafted the same tell nothing about a row.
const FILL_MIN_VAR: f64 = 0.1;

/// Least squares of a verify shape's step time `y` over its drafted rows `x`, with older steps
/// decayed by [`FILL_DECAY`]: the slope is what a drafted row costs more than a padding row.
#[derive(Debug, Clone, Copy, Default)]
struct Fill {
    w: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
}

impl Fill {
    fn add(&mut self, x: f64, y: f64) {
        self.w = FILL_DECAY * self.w + 1.0;
        self.x = FILL_DECAY * self.x + x;
        self.y = FILL_DECAY * self.y + y;
        self.xx = FILL_DECAY * self.xx + x * x;
        self.xy = FILL_DECAY * self.xy + x * y;
    }

    /// Milliseconds per drafted row (at least 0), once enough steps with different numbers of
    /// drafts are in.
    fn slope(&self) -> Option<f64> {
        if self.w < FILL_MIN {
            return None;
        }
        let (mx, my) = (self.x / self.w, self.y / self.w);
        let var = self.xx / self.w - mx * mx;
        if var < FILL_MIN_VAR {
            return None;
        }
        Some(((self.xy / self.w - mx * my) / var).max(0.0))
    }
}

/// Context buckets: spans under 4096 positions, then 4K-8K, 8K-16K, and so on.
pub fn context_bucket(span: usize) -> usize {
    if span < 4096 {
        0
    } else {
        (span / 2048).ilog2() as usize
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Mean {
    ms: f64,
    n: u32,
}

impl Mean {
    /// A new shape's first step includes building its graph, and a shape the executor no longer
    /// keeps is rebuilt: the second sample replaces a higher first one, and later samples count
    /// at most [`SPIKE`] times the mean.
    fn add(&mut self, ms: f64) {
        self.ms = match self.n {
            0 => ms,
            1 => self.ms.min(ms),
            _ => self.ms + EWMA * (ms.min(SPIKE * self.ms) - self.ms),
        };
        self.n = self.n.saturating_add(1);
    }
}

/// Measured costs and the current depth of one engine.
#[derive(Debug)]
pub struct DepthModel {
    /// Whether the head runs a pass at depth 0 too (a head with its own cache takes in the
    /// verified positions every step; one reading the target's cache does not).
    take_in: bool,
    /// Whether one pass drafts any depth (a block drafter, [`Self::block`]).
    block: bool,
    /// Per context bucket ([`context_bucket`]): target step time by rows (sequences x tokens
    /// per sequence), and draft pass time by sequences in the pass.
    buckets: Vec<Costs>,
    /// The bucket of the steps being measured and chosen for ([`Self::set_context`]).
    active: usize,
    /// The last depth chosen by throughput (not a probe).
    current: Option<usize>,
    choices: usize,
    calibration: Calibration,
}

impl DepthModel {
    pub fn new(take_in: bool) -> Self {
        Self {
            take_in,
            block: false,
            buckets: vec![Costs::default()],
            active: 0,
            current: None,
            choices: 0,
            calibration: Calibration::default(),
        }
    }

    /// For a drafter that drafts a whole block in one pass whatever the depth and takes in
    /// every step's positions (DFlash 2, [`crate::spec::ModelDrafter::block_drafts`]): one pass
    /// at every depth, 0 included, so only the verify step's rows price the depth.
    pub fn block() -> Self {
        Self {
            block: true,
            ..Self::new(true)
        }
    }

    /// The head drafted `confidence.len()` tokens with these probabilities and the first
    /// `accepted` were accepted: every draft up to the first rejected one is judged.
    pub fn record_drafts(&mut self, confidence: &[f32], accepted: usize) {
        for (i, &c) in confidence.iter().enumerate().take(accepted + 1) {
            self.calibration.record(c, i < accepted);
        }
    }

    /// The steps measured and chosen for from now on read a context of `span` positions (the
    /// batch's longest): their costs go to, and come from, its bucket ([`context_bucket`]).
    pub fn set_context(&mut self, span: usize) {
        self.active = context_bucket(span);
        if self.buckets.len() <= self.active {
            self.buckets.resize(self.active + 1, Costs::default());
        }
    }

    /// A target step of `rows` rows (decode or verify, no prompt chunk) took `ms`.
    pub fn record_step(&mut self, rows: usize, ms: f64) {
        if rows == 0 || !ms.is_finite() {
            return;
        }
        let step = &mut self.buckets[self.active].step;
        if step.len() <= rows {
            step.resize(rows + 1, Mean::default());
        }
        step[rows].add(ms);
    }

    /// The verify step of `rows` rows just given to [`Self::record_step`] carried `drafted`
    /// drafts (the other rows past each sequence's first padded its shape). Steps the step
    /// average does not trust (a shape's first two, which may build its graph, and spikes past
    /// [`SPIKE`] times its mean) are left out.
    pub fn record_fill(&mut self, rows: usize, drafted: usize, ms: f64) {
        let costs = &mut self.buckets[self.active];
        let Some(mean) = costs.step.get(rows).filter(|m| m.n > 2) else {
            return;
        };
        if !ms.is_finite() || ms > SPIKE * mean.ms {
            return;
        }
        if costs.fill.len() <= rows {
            costs.fill.resize(rows + 1, Fill::default());
        }
        costs.fill[rows].add(drafted as f64, ms);
    }

    /// What a drafted row costs more than a padding row in a verify step of `rows` rows, in
    /// the active bucket (or the nearest one that has it): `None` until measured.
    pub fn row_ms(&self, rows: usize) -> Option<f64> {
        self.nearest(|c| c.fill.get(rows).and_then(Fill::slope))
    }

    /// `passes` draft passes over (at most) `seqs` sequences took `ms` in all.
    pub fn record_passes(&mut self, seqs: usize, passes: usize, ms: f64) {
        if seqs == 0 || passes == 0 || !ms.is_finite() {
            return;
        }
        let pass = &mut self.buckets[self.active].pass;
        if pass.len() <= seqs {
            pass.resize(seqs + 1, Mean::default());
        }
        pass[seqs].add(ms / passes as f64);
    }

    /// Predicted time of a target step of `rows` rows, in the active bucket (or, before it has
    /// any, the nearest one that has).
    pub fn step_ms(&self, rows: usize) -> Option<f64> {
        self.nearest(|c| predict(&c.step, rows))
    }

    /// Predicted time of one draft pass over `seqs` sequences (as [`Self::step_ms`]).
    pub fn pass_ms(&self, seqs: usize) -> Option<f64> {
        self.nearest(|c| predict(&c.pass, seqs))
    }

    /// `f` of the active bucket, else of the nearest bucket where it has a value (the shorter
    /// one first).
    fn nearest(&self, f: impl Fn(&Costs) -> Option<f64>) -> Option<f64> {
        let a = self.active;
        (0..self.buckets.len()).find_map(|d| {
            let below = a
                .checked_sub(d)
                .and_then(|b| self.buckets.get(b))
                .and_then(&f);
            below.or_else(|| self.buckets.get(a + d).and_then(&f))
        })
    }

    /// Predicted tokens per millisecond for depth `k`, where `gains[i][k]` is what sequence `i`
    /// of the batch is expected to gain from a step with up to `k` drafts.
    pub fn throughput(&self, gains: &[Vec<f32>], k: usize) -> Option<f64> {
        let b = gains.len();
        let step = self.step_ms(b * (1 + k))?;
        // Before the first draft pass, a tenth of the step (the heads are a layer or a few).
        let pass = self.pass_ms(b).unwrap_or(step / 10.0);
        let tokens: f64 = gains.iter().map(|g| g[k] as f64).sum();
        let passes = match (self.block, self.take_in) {
            (true, _) => 1,
            (false, true) => k.max(1),
            (false, false) => k,
        };
        Some(tokens / (step + pass * passes as f64))
    }

    /// The head confidence below which a further draft does not pay at depth `k`: a draft
    /// within the cap costs one more pass (shared by the batch) and what its verify row costs
    /// more than padding ([`Self::row_ms`], 0 until measured), and is worth a token's time if
    /// accepted, so drafting on pays where drafts are accepted more often than that cost over
    /// time per token (acceptance by confidence: [`Self::record_drafts`]). At most `ceiling`
    /// (`--spec-p-min`), at least [`MIN_CONFIDENCE`].
    pub fn confidence_floor(&self, gains: &[Vec<f32>], k: usize, ceiling: f32) -> f32 {
        let b = gains.len().max(1);
        let floor = match (self.pass_ms(b), self.throughput(gains, k)) {
            // Per sequence: its share of the pass, and its own row, in the batch's time.
            (Some(pass), Some(tokens_per_ms)) => {
                let row = self.row_ms(b * (1 + k)).unwrap_or(0.0);
                self.calibration
                    .floor((pass * tokens_per_ms / b as f64 + row * tokens_per_ms) as f32)
            }
            _ => ceiling,
        };
        floor.clamp(MIN_CONFIDENCE.min(ceiling), ceiling)
    }

    /// The depth for the next step, at most `k_max`, of the batch whose sequences gain
    /// `gains[i][k]` tokens from a step of `k` drafts (`k` in `0..=k_max`).
    pub fn choose(&mut self, gains: &[Vec<f32>], k_max: usize) -> usize {
        if gains.is_empty() || k_max == 0 {
            return 0;
        }
        let Some(cur) = self.current.map(|c| c.min(k_max)) else {
            // Nothing measured yet: a moderate depth, measured from the next step on.
            let k = FIRST_DEPTH.min(k_max);
            self.current = Some(k);
            return k;
        };
        let score = |k: usize| self.throughput(gains, k).unwrap_or(0.0);
        let top = (0..=k_max)
            .max_by(|&x, &y| score(x).total_cmp(&score(y)))
            .unwrap_or(cur);
        // Down at once, up one step at a time: a short run of accepted drafts lifts the
        // acceptance estimate for a step or two, and a deep step it does not hold up costs.
        let best = if score(top) <= score(cur) * HYSTERESIS {
            cur
        } else if top > cur {
            cur + 1
        } else {
            top
        };
        self.current = Some(best);
        self.choices += 1;
        if self.choices.is_multiple_of(PROBE_EVERY) {
            // A neighbour, alternating up and down, so both sides stay measured.
            let up = (self.choices / PROBE_EVERY) % 2 == 1;
            return match (up, best) {
                (true, b) if b < k_max => b + 1,
                (_, b) if b > 0 => b - 1,
                (_, b) => b + 1,
            };
        }
        best
    }
}

/// The average at `x`, or else interpolated between measured neighbours, or else the nearest
/// measured value.
fn predict(m: &[Mean], x: usize) -> Option<f64> {
    if let Some(v) = m.get(x).filter(|v| v.n > 0) {
        return Some(v.ms);
    }
    let below = (0..x.min(m.len()))
        .rev()
        .find(|&i| m[i].n > 0)
        .map(|i| (i, m[i].ms));
    let above = (x + 1..m.len()).find(|&i| m[i].n > 0).map(|i| (i, m[i].ms));
    match (below, above) {
        (Some((i, a)), Some((j, b))) => Some(a + (b - a) * (x - i) as f64 / (j - i) as f64),
        (Some((_, a)), None) => Some(a),
        (None, Some((_, b))) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests;
