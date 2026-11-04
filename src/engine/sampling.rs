//! Token sampling from logits: penalties, temperature, top-k, top-p, min-p and a seeded RNG.
//!
//! Order (as in vLLM and HF transformers): penalties on the raw logits, then temperature, then
//! the filters on the tempered distribution (min-p, top-k, then top-p over what is left), then
//! one draw. Temperature
//! 0 is greedy: the argmax after penalties, no randomness.
//!
//! Penalties look at the tokens generated so far (not the prompt), as OpenAI defines
//! `presence_penalty` and `frequency_penalty`: `logit -= presence * [count > 0] + frequency *
//! count`. `repetition_penalty` (HF / vLLM style) divides positive logits and multiplies
//! negative ones of the tokens seen; 1.0 is off.

use std::collections::HashMap;

use super::SamplingParams;

/// A small, fast, seedable RNG: SplitMix64 (Steele, Lea, Flood 2014, public domain reference
/// code). Deterministic for a given seed on every platform.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Seeded from the OS's per-process hash keys and the clock.
    pub fn from_entropy() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        Self(h.finish())
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A sampling distribution over token ids, as [`Sampler::dist`] gives it.
#[derive(Debug, Clone, PartialEq)]
pub enum Dist {
    /// Temperature 0: always this token.
    Greedy(u32),
    /// `(token, probability)`, most likely first, summing to 1; tokens not listed have
    /// probability 0.
    Probs(Vec<(u32, f64)>),
}

impl Dist {
    /// The probability of `token`.
    pub fn prob(&self, token: u32) -> f64 {
        match self {
            Dist::Greedy(t) => f64::from(u8::from(*t == token)),
            Dist::Probs(p) => p
                .iter()
                .find(|&&(id, _)| id == token)
                .map_or(0.0, |&(_, p)| p),
        }
    }
}

/// Picks tokens for one sequence. Keeps the RNG and the counts of generated tokens.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    counts: HashMap<u32, u32>,
}

/// top-p first looks at this many most likely tokens, and sorts the whole vocabulary only if
/// they do not cover `top_p`.
const TOP_P_FIRST_PASS: usize = 1024;

impl Sampler {
    pub fn new(params: &SamplingParams) -> Self {
        let rng = match params.seed {
            Some(seed) => Rng::new(seed),
            None => Rng::from_entropy(),
        };
        Self {
            params: params.clone(),
            rng,
            counts: HashMap::new(),
        }
    }

    /// Whether [`Self::sample`] is a plain argmax of the model's logits (temperature 0, no
    /// penalties), which the engine can then compute on the device.
    pub fn is_greedy(&self) -> bool {
        let p = &self.params;
        p.temperature <= 0.0
            && p.presence_penalty == 0.0
            && p.frequency_penalty == 0.0
            && p.repetition_penalty == 1.0
    }

    /// Records a generated token for the penalties.
    pub fn accept(&mut self, token: u32) {
        *self.counts.entry(token).or_default() += 1;
    }

    /// Picks the next token. `logits` is modified (penalties, temperature).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "no logits");
        self.apply_penalties(logits);
        let p = &self.params;
        if p.temperature <= 0.0 {
            return argmax(logits) as u32;
        }
        if p.temperature != 1.0 {
            let inv = 1.0 / p.temperature;
            logits.iter_mut().for_each(|l| *l *= inv);
        }
        let candidates = self.candidates(logits);
        let total: f64 = candidates.iter().map(|&(_, p)| p).sum();
        let mut r = self.rng.next_f64() * total;
        for &(id, p) in &candidates {
            if r < p {
                return id;
            }
            r -= p;
        }
        candidates.last().expect("at least one candidate").0
    }

    /// The distribution [`Self::sample`] draws from, for speculative acceptance: penalties
    /// (counting the tokens [`Self::accept`]ed so far), then temperature and the filters,
    /// normalized; a point mass at the argmax for temperature 0. `logits` is modified.
    pub fn dist(&self, logits: &mut [f32]) -> Dist {
        assert!(!logits.is_empty(), "no logits");
        self.apply_penalties(logits);
        let p = &self.params;
        if p.temperature <= 0.0 {
            return Dist::Greedy(argmax(logits) as u32);
        }
        if p.temperature != 1.0 {
            let inv = 1.0 / p.temperature;
            logits.iter_mut().for_each(|l| *l *= inv);
        }
        let mut candidates = self.candidates(logits);
        let total: f64 = candidates.iter().map(|&(_, p)| p).sum();
        candidates.iter_mut().for_each(|c| c.1 /= total);
        Dist::Probs(candidates)
    }

    /// A uniform draw in `[0, 1)` from this sampler's RNG.
    pub fn uniform(&mut self) -> f64 {
        self.rng.next_f64()
    }

    fn apply_penalties(&self, logits: &mut [f32]) {
        let p = &self.params;
        let rep = p.repetition_penalty;
        if p.presence_penalty == 0.0 && p.frequency_penalty == 0.0 && rep == 1.0 {
            return;
        }
        for (&id, &count) in &self.counts {
            let Some(l) = logits.get_mut(id as usize) else {
                continue;
            };
            if rep != 1.0 {
                *l = if *l > 0.0 { *l / rep } else { *l * rep };
            }
            *l -= p.presence_penalty + p.frequency_penalty * count as f32;
        }
    }

    /// The tokens that survive top-k / top-p / min-p, with unnormalized probabilities,
    /// most likely first.
    fn candidates(&self, logits: &[f32]) -> Vec<(u32, f64)> {
        let p = &self.params;
        let k = p.top_k as usize;
        if k > 0 && k <= SMALL_TOP_K && k < logits.len() {
            return self.top_k_candidates(logits, k);
        }
        if (k == 0 || k >= logits.len())
            && p.min_p <= 0.0
            && p.top_p < 1.0
            && let Some(c) = top_p_candidates(logits, p.top_p as f64)
        {
            return c;
        }
        self.general_candidates(logits)
    }

    /// [`Self::candidates`] over the whole vocabulary, for any filters.
    fn general_candidates(&self, logits: &[f32]) -> Vec<(u32, f64)> {
        let p = &self.params;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let prob = |l: f32| (l as f64 - max).exp();
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);

        // min-p: keep tokens with p >= min_p * p_max (p_max is 1 before normalization).
        if p.min_p > 0.0 {
            let floor = p.min_p as f64;
            ids.retain(|&id| prob(logits[id as usize]) >= floor);
        }
        let k = p.top_k as usize;
        if k > 0 && k < ids.len() {
            ids.select_nth_unstable_by(k - 1, by_logit);
            ids.truncate(k);
        }
        if p.top_p < 1.0 {
            let total: f64 = ids.iter().map(|&id| prob(logits[id as usize])).sum();
            let target = p.top_p as f64 * total;
            if ids.len() > TOP_P_FIRST_PASS {
                let mut head = ids.clone();
                head.select_nth_unstable_by(TOP_P_FIRST_PASS - 1, by_logit);
                head.truncate(TOP_P_FIRST_PASS);
                let covered: f64 = head.iter().map(|&id| prob(logits[id as usize])).sum();
                if covered >= target {
                    ids = head;
                }
            }
            sort_by_logit(&mut ids, logits);
            let mut cum = 0.0;
            let mut keep = ids.len();
            for (i, &id) in ids.iter().enumerate() {
                cum += prob(logits[id as usize]);
                if cum >= target {
                    keep = i + 1;
                    break;
                }
            }
            ids.truncate(keep.max(1));
        } else {
            sort_by_logit(&mut ids, logits);
        }
        ids.into_iter()
            .map(|id| (id, prob(logits[id as usize])))
            .collect()
    }
}

impl Sampler {
    /// [`Self::candidates`] for a small top-k, in one pass over the vocabulary: the `k` best
    /// logits kept in a sorted buffer (ties: the lower token id first), then min-p and top-p
    /// over them. The same tokens as the general path (which sorts or selects over the whole
    /// vocabulary: 248K entries per row on Qwen3.8, three rows per speculative verify step).
    /// min-p keeps every token above a threshold, so it keeps a prefix of the k best.
    fn top_k_candidates(&self, logits: &[f32], k: usize) -> Vec<(u32, f64)> {
        let p = &self.params;
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for (id, &l) in logits.iter().enumerate() {
            if best.len() == k && l.total_cmp(&best[k - 1].0).is_le() {
                continue;
            }
            let at = best.partition_point(|&(b, _)| b.total_cmp(&l).is_ge());
            best.insert(at, (l, id as u32));
            best.truncate(k);
        }
        let max = best[0].0 as f64;
        let prob = |l: f32| (l as f64 - max).exp();
        if p.min_p > 0.0 {
            let floor = p.min_p as f64;
            best.retain(|&(l, _)| prob(l) >= floor);
        }
        if p.top_p < 1.0 {
            let total: f64 = best.iter().map(|&(l, _)| prob(l)).sum();
            let target = p.top_p as f64 * total;
            let mut cum = 0.0;
            let mut keep = best.len();
            for (i, &(l, _)) in best.iter().enumerate() {
                cum += prob(l);
                if cum >= target {
                    keep = i + 1;
                    break;
                }
            }
            best.truncate(keep.max(1));
        }
        best.into_iter().map(|(l, id)| (id, prob(l))).collect()
    }
}

/// [`Sampler::candidates`] for top-p alone (no top-k, no min-p) without the work over the whole
/// vocabulary that made it 3-4 ms per row on the AC922 (#695): the general path sums `exp` of
/// every logit with libm (3.4 ms for GLM-5.3-Flash's 155K-token rows on POWER9) and selects the
/// 1024 best out of all of them (0.7 ms).
///
/// Here the softmax total is estimated with [`approx_exp_sum`] (vectorized, within
/// [`TOTAL_SLACK`] of the general path's sum, the bound for every row), which brackets the
/// general path's target `top_p * total` between `lo` and `hi`. The tokens whose logits are
/// within `cut` of the best are sorted and summed with libm, in the general path's order,
/// until the running sum reaches `lo`; if it is then also at or past `hi`, the general path
/// would cut at the same token, because its running sums are the same numbers (which exact
/// ties come first is the only freedom; here the lower token id). `cut` leaves out at most
/// `n e^-cut` of the mass, too little to keep the running sum from `hi`. When the sum falls between
/// `lo` and `hi`, or a row has no finite logits or NaNs, the general path decides (`None`).
/// The same tokens with the same probabilities, bit for bit.
fn top_p_candidates(logits: &[f32], top_p: f64) -> Option<Vec<(u32, f64)>> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let p_hi = top_p * (1.0 + TOTAL_SLACK);
    if !max.is_finite() || p_hi >= 1.0 || top_p.is_nan() {
        return None;
    }
    let cut = (logits.len() as f64 * p_hi / (1.0 - p_hi)).ln().max(0.0) + 1.0;
    let floor = (max - cut) as f32;
    let mut ids: Vec<u32> = Vec::new();
    for (id, &l) in logits.iter().enumerate() {
        if l >= floor {
            ids.push(id as u32);
        }
    }
    let total = approx_exp_sum(logits, max);
    if !total.is_finite() {
        return None;
    }
    let lo = top_p * total * (1.0 - TOTAL_SLACK);
    let hi = top_p * total * (1.0 + TOTAL_SLACK);
    sort_by_logit(&mut ids, logits);
    let mut out = Vec::new();
    let mut cum = 0.0;
    for id in ids {
        let p = (logits[id as usize] as f64 - max).exp();
        out.push((id, p));
        cum += p;
        if cum >= lo {
            return (cum >= hi).then_some(out);
        }
    }
    None
}

/// How far [`approx_exp_sum`] may be from the sequential libm sum of the general top-p path,
/// relative: its `exp` is within 3e-13 of libm's, and the two orders of summation add at most
/// `n` rounding errors of 1.1e-16 each (2e-11 for a 155K vocabulary), far inside this.
const TOTAL_SLACK: f64 = 1e-8;

/// `sum(exp(l - max))` over `logits`, `max` at least every logit, with [`approx_exp`] in 16
/// independent lanes (they vectorize). NaN when a logit is NaN.
fn approx_exp_sum(logits: &[f32], max: f64) -> f64 {
    const LANES: usize = 16;
    let mut acc = [0.0f64; LANES];
    let (chunks, rest) = logits.as_chunks::<LANES>();
    for c in chunks {
        for (a, &l) in acc.iter_mut().zip(c) {
            *a += approx_exp(l as f64 - max);
        }
    }
    let tail: f64 = rest.iter().map(|&l| approx_exp(l as f64 - max)).sum();
    acc.iter().sum::<f64>() + tail
}

/// `e^x` for `x <= 0`, within 3e-13 relative of libm down to -700 (below that, `e^-700`; a
/// NaN stays NaN): `x = k ln 2 + r` with `|r| <= ln 2 / 2` (Cody-Waite, fdlibm's split of
/// ln 2), `e^r` by its Taylor series to `r^10` (truncation below 2e-13), times `2^k` made from
/// the exponent bits. Plain operations, no table, so a loop over it vectorizes.
#[inline(always)]
fn approx_exp(x: f64) -> f64 {
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    // 1.5 * 2^52: adding it rounds to an integer, kept in the low mantissa bits.
    const ROUND: f64 = 6_755_399_441_055_744.0;
    const INV_FACT: [f64; 11] = [
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5_040.0,
        1.0 / 40_320.0,
        1.0 / 362_880.0,
        1.0 / 3_628_800.0,
    ];
    let x = if x < -700.0 { -700.0 } else { x };
    let y = x * std::f64::consts::LOG2_E + ROUND;
    let k = y - ROUND;
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = INV_FACT[10];
    for &c in INV_FACT[..10].iter().rev() {
        p = p * r + c;
    }
    // k + 1023 (in 13..=1023) into the exponent field; the low 12 bits of y's bits are k's.
    p * f64::from_bits(y.to_bits().wrapping_add(1023) << 52)
}

/// Below this many ids [`sort_by_logit`] compares; above it, it radix-sorts.
const RADIX_MIN: usize = 4096;

/// Sorts `ids` by their logit, largest first (`f32::total_cmp`), ties by the order they come in
/// (ascending ids from [`Sampler::candidates`]). Sampling a whole vocabulary (no top-k, top-p 1:
/// the API's defaults) sorted all 248K ids of a Qwen3.8 row with comparisons per sampled row,
/// ~4.5 ms on the GB10; a stable radix sort on the logits' order-preserving bits gives the same
/// order (any order of exact ties was allowed before) in a few passes.
fn sort_by_logit(ids: &mut [u32], logits: &[f32]) {
    if ids.len() < RADIX_MIN {
        ids.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        return;
    }
    // total_cmp's order as unsigned keys, then inverted for largest first.
    let key = |id: u32| {
        let b = logits[id as usize].to_bits();
        let up = if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        };
        !up
    };
    let mut keyed: Vec<(u32, u32)> = ids.iter().map(|&id| (key(id), id)).collect();
    let mut tmp = vec![(0u32, 0u32); keyed.len()];
    for shift in [0u32, 16] {
        let mut count = vec![0usize; 1 << 16];
        for &(k, _) in &keyed {
            count[((k >> shift) & 0xffff) as usize] += 1;
        }
        let mut at = 0;
        for c in count.iter_mut() {
            let n = *c;
            *c = at;
            at += n;
        }
        for &e in &keyed {
            let d = ((e.0 >> shift) & 0xffff) as usize;
            tmp[count[d]] = e;
            count[d] += 1;
        }
        std::mem::swap(&mut keyed, &mut tmp);
    }
    for (dst, (_, id)) in ids.iter_mut().zip(keyed) {
        *dst = id;
    }
}

/// Largest top-k taken by [`Sampler::top_k_candidates`]; beyond it the buffer's inserts cost
/// more than the general path's selection.
const SMALL_TOP_K: usize = 256;

/// Index of the largest value (the first one on ties).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// Log probability of `token` under `softmax(logits)`, and the `n` most likely tokens with
/// theirs (most likely first). For OpenAI-style `logprobs`, taken from the model's raw
/// distribution, before penalties and temperature.
pub fn logprobs(logits: &[f32], token: u32, n: usize) -> (f32, Vec<(u32, f32)>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = logits
        .iter()
        .map(|&l| (l as f64 - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    let lp = |id: u32| (logits[id as usize] as f64 - lse) as f32;
    let mut top = Vec::new();
    if n > 0 {
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        let n = n.min(ids.len());
        ids.select_nth_unstable_by(n - 1, by_logit);
        ids.truncate(n);
        ids.sort_unstable_by(by_logit);
        top = ids.into_iter().map(|id| (id, lp(id))).collect();
    }
    (lp(token), top)
}

#[cfg(test)]
mod tests;
