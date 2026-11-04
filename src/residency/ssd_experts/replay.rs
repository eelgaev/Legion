//! Offline replays of slot policies over a routing trace ([`super::trace`]): how many of each
//! small step's experts a policy would have had in a slot. `legion ssd-replay` prints the table.
//!
//! Every policy sees the steps the way the tier treats them: a small step needs all its
//! experts in slots (each one not there is read, into an empty slot or the victim's), a big
//! step reads its experts elsewhere and only fills empty slots. Hits and misses count each
//! small step's distinct experts, as the tier's log does.
//!
//! The pinned split (a hot set kept apart from an LRU part sharing the budget, repinned live
//! when an outsider is hotter than the coldest pinned expert by 25%, at most 4 swaps per step)
//! follows project colibri's learning cache (Apache-2.0, github.com/JustVugg/colibri,
//! `docs/tuning.md`, "The learning cache").

use super::cache::SlotCache;
use super::trace::Trace;
use crate::residency::profile::Profile;

/// How the replay picks the slot a missing expert takes when none is empty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// The least recently used slot ([`SlotCache`] as the tier runs it).
    Lru,
    /// The coldest slot (least recently used among equals): routing heat, +1 per routed pair,
    /// times `decay` every small step; with `prompts`, big steps' pairs count too (without a
    /// decay step).
    Heat { decay: f32, prompts: bool },
    /// A pinned hot set of `frac` of the slots, LRU over the rest; the pinned set follows the
    /// heat (decayed as in [`Policy::Heat`]).
    Pinned { frac: f32, decay: f32 },
    /// The slot whose expert the layer's small steps need again last (Belady's optimum).
    Belady,
    /// LRU-2: the slot whose expert's second-to-last small step is oldest (an expert used once
    /// goes first), most recent use breaking ties.
    Lru2,
}

/// Colibri's repinning: an outsider must be this much hotter than the coldest pinned expert...
const REPIN_MARGIN: f32 = 0.25;
/// ...and at most this many trade places per step.
const REPIN_SWAPS: usize = 4;

/// One replay's settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Setup {
    pub policy: Policy,
    pub n_slots: usize,
    /// Per tier layer (ordinal): starting heat per expert, e.g. from an earlier run's profile
    /// ([`seed_heat`]). The hottest experts fill the slots before the first step.
    pub seed: Option<Vec<Vec<f32>>>,
}

/// What a replay served.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Outcome {
    /// Small steps' distinct experts found in a slot, and read.
    pub hits: u64,
    pub misses: u64,
    /// Tokens of the small steps (of one layer).
    pub tokens: u64,
}

impl Outcome {
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }

    /// Experts read per token, over all layers.
    pub fn misses_per_token(&self) -> f64 {
        self.misses as f64 / self.tokens.max(1) as f64
    }
}

/// The distinct experts of `ids` in first-seen order, and the routed pairs of each.
fn distinct(ids: &[u16], n_expert: usize) -> (Vec<usize>, Vec<u32>) {
    let mut count = vec![0u32; n_expert];
    let mut out = Vec::new();
    for &e in ids {
        let e = e as usize;
        if count[e] == 0 {
            out.push(e);
        }
        count[e] += 1;
    }
    (out, count)
}

/// Replays `trace` under `setup`.
pub fn replay(trace: &Trace, setup: &Setup) -> Outcome {
    let h = &trace.header;
    let mut out = Outcome::default();
    for ord in 0..h.layers.len() {
        let steps: Vec<(bool, usize, &[u16])> = trace
            .records
            .iter()
            .filter(|r| r.ord == ord)
            .map(|r| (r.big, r.n_tokens, r.ids.as_slice()))
            .collect();
        let seed = setup.seed.as_ref().and_then(|s| s.get(ord));
        let o = match setup.policy {
            Policy::Lru => run_lru(&steps, h.n_expert, setup.n_slots, seed),
            Policy::Heat { decay, prompts } => {
                run_heat(&steps, h.n_expert, setup.n_slots, decay, prompts, seed)
            }
            Policy::Pinned { frac, decay } => {
                run_pinned(&steps, h.n_expert, setup.n_slots, frac, decay, seed)
            }
            Policy::Belady => run_belady(&steps, h.n_expert, setup.n_slots, seed),
            Policy::Lru2 => run_lru2(&steps, h.n_expert, setup.n_slots, seed),
        };
        out.hits += o.hits;
        out.misses += o.misses;
        if ord == 0 {
            out.tokens = o.tokens;
        }
    }
    out
}

/// The experts `seed` ranks hottest (ties: lower id), at most `n`.
fn seed_order(seed: &[f32], n: usize) -> Vec<usize> {
    let mut r: Vec<usize> = (0..seed.len()).filter(|&e| seed[e] > 0.0).collect();
    r.sort_by(|&a, &b| seed[b].total_cmp(&seed[a]).then(a.cmp(&b)));
    r.truncate(n);
    r
}

/// Puts `experts` into `c`'s empty slots, in order, while there are any.
fn fill_empty(c: &mut SlotCache, experts: &[usize]) {
    for &e in experts {
        if c.slot(e).is_none() && c.take_empty(e).is_none() {
            break;
        }
    }
}

/// LRU, through the tier's own [`SlotCache`].
fn run_lru(
    steps: &[(bool, usize, &[u16])],
    n_expert: usize,
    n_slots: usize,
    seed: Option<&Vec<f32>>,
) -> Outcome {
    let mut c = SlotCache::new(n_expert, n_slots);
    if let Some(sd) = seed {
        fill_empty(&mut c, &seed_order(sd, n_slots));
    }
    let mut o = Outcome::default();
    for &(big, n_tokens, ids) in steps {
        let (experts, _) = distinct(ids, n_expert);
        if big {
            fill_empty(&mut c, &experts);
        } else {
            let p = c.plan(&experts);
            o.hits += p.hits as u64;
            o.misses += p.loads.len() as u64;
            o.tokens += n_tokens as u64;
        }
    }
    o
}

/// Decaying routing heat: the coldest slot this step does not use goes (least recently used,
/// then lowest slot, among equals).
fn run_heat(
    steps: &[(bool, usize, &[u16])],
    n_expert: usize,
    n_slots: usize,
    decay: f32,
    prompts: bool,
    seed: Option<&Vec<f32>>,
) -> Outcome {
    let mut s = Slots::new(n_expert, n_slots);
    let mut heat = vec![0f32; n_expert];
    if let Some(sd) = seed {
        heat.clone_from(sd);
        s.fill(&seed_order(sd, n_slots));
    }
    let mut o = Outcome::default();
    let mut tick = 0;
    for &(big, n_tokens, ids) in steps {
        let (experts, count) = distinct(ids, n_expert);
        if big {
            if prompts {
                for &e in &experts {
                    heat[e] += count[e] as f32;
                }
            }
            s.fill_big(&experts, tick);
            continue;
        }
        tick += 1;
        for h in &mut heat {
            *h *= decay;
        }
        for &e in &experts {
            heat[e] += count[e] as f32;
        }
        let (h, m) = s.step(&experts, tick, |s, free| {
            *free
                .iter()
                .min_by(|&&a, &&b| {
                    heat[s.owner[a].unwrap()]
                        .total_cmp(&heat[s.owner[b].unwrap()])
                        .then(s.used[a].cmp(&s.used[b]))
                        .then(a.cmp(&b))
                })
                .expect("a free slot")
        });
        o.hits += h;
        o.misses += m;
        o.tokens += n_tokens as u64;
    }
    o
}

/// One layer's slots for the replays that are not [`SlotCache`]: expert per slot, last use.
struct Slots {
    owner: Vec<Option<usize>>,
    slot_of: Vec<Option<usize>>,
    used: Vec<u64>,
}

impl Slots {
    fn new(n_expert: usize, n_slots: usize) -> Self {
        Self {
            owner: vec![None; n_slots],
            slot_of: vec![None; n_expert],
            used: vec![0; n_slots],
        }
    }

    fn empty(&self) -> Option<usize> {
        self.owner.iter().position(Option::is_none)
    }

    fn put(&mut self, e: usize, s: usize, tick: u64) {
        if let Some(old) = self.owner[s] {
            self.slot_of[old] = None;
        }
        self.owner[s] = Some(e);
        self.slot_of[e] = Some(s);
        self.used[s] = tick;
    }

    fn fill(&mut self, experts: &[usize]) {
        for &e in experts {
            if let Some(s) = self.empty() {
                self.put(e, s, 0);
            }
        }
    }

    /// A big step: its experts take empty slots, while there are any.
    fn fill_big(&mut self, experts: &[usize], tick: u64) {
        for &e in experts {
            if self.slot_of[e].is_some() {
                continue;
            }
            let Some(s) = self.empty() else { break };
            self.put(e, s, tick);
        }
    }

    /// A small step at `tick`: marks the cached ones, then each missing one takes an empty slot
    /// or the one `victim` picks among those this step does not use.
    fn step(
        &mut self,
        experts: &[usize],
        tick: u64,
        mut victim: impl FnMut(&Self, &[usize]) -> usize,
    ) -> (u64, u64) {
        let mut missing = Vec::new();
        for &e in experts {
            match self.slot_of[e] {
                Some(s) => self.used[s] = tick,
                None => missing.push(e),
            }
        }
        let hits = (experts.len() - missing.len()) as u64;
        for &e in &missing {
            let free: Vec<usize> = (0..self.owner.len())
                .filter(|&s| self.used[s] < tick)
                .collect();
            let s = self.empty().unwrap_or_else(|| victim(self, &free));
            self.put(e, s, tick);
        }
        (hits, missing.len() as u64)
    }
}

/// Colibri's split: a pinned hot set, LRU over the other slots.
fn run_pinned(
    steps: &[(bool, usize, &[u16])],
    n_expert: usize,
    n_slots: usize,
    frac: f32,
    decay: f32,
    seed: Option<&Vec<f32>>,
) -> Outcome {
    let n_pin = ((frac * n_slots as f32).round() as usize).min(n_slots);
    let mut s = Slots::new(n_expert, n_slots);
    let mut heat = vec![0f32; n_expert];
    let mut pinned = vec![false; n_expert];
    if let Some(sd) = seed {
        heat.clone_from(sd);
        s.fill(&seed_order(sd, n_slots));
        for e in seed_order(sd, n_pin) {
            pinned[e] = true;
        }
    }
    let mut o = Outcome::default();
    let mut tick = 0;
    for &(big, n_tokens, ids) in steps {
        let (experts, count) = distinct(ids, n_expert);
        if big {
            for &e in &experts {
                heat[e] += count[e] as f32;
            }
            s.fill_big(&experts, tick);
            continue;
        }
        tick += 1;
        for h in &mut heat {
            *h *= decay;
        }
        for &e in &experts {
            heat[e] += count[e] as f32;
        }
        repin(&heat, &mut pinned, n_pin);
        let (h, m) = s.step(&experts, tick, |s, free| {
            let lru = |only_unpinned: bool| {
                free.iter()
                    .copied()
                    .filter(|&x| !only_unpinned || !pinned[s.owner[x].unwrap()])
                    .min_by_key(|&x| (s.used[x], x))
            };
            lru(true).or_else(|| lru(false)).expect("a free slot")
        });
        o.hits += h;
        o.misses += m;
        o.tokens += n_tokens as u64;
    }
    o
}

/// Tops the pinned set up to `n_pin` with the hottest experts, then trades up to
/// [`REPIN_SWAPS`] of the coldest pinned ones for hotter outsiders.
fn repin(heat: &[f32], pinned: &mut [bool], n_pin: usize) {
    let mut out: Vec<usize> = (0..heat.len())
        .filter(|&e| !pinned[e] && heat[e] > 0.0)
        .collect();
    out.sort_by(|&a, &b| heat[b].total_cmp(&heat[a]).then(a.cmp(&b)));
    let mut n = pinned.iter().filter(|&&p| p).count();
    let mut next = out.into_iter().peekable();
    while n < n_pin {
        let Some(e) = next.next() else { return };
        pinned[e] = true;
        n += 1;
    }
    let mut cold: Vec<usize> = (0..heat.len()).filter(|&e| pinned[e]).collect();
    cold.sort_by(|&a, &b| heat[a].total_cmp(&heat[b]).then(a.cmp(&b)));
    for c in cold.into_iter().take(REPIN_SWAPS) {
        let Some(&h) = next.peek() else { return };
        if heat[h] <= heat[c] * (1.0 + REPIN_MARGIN) {
            return;
        }
        pinned[c] = false;
        pinned[h] = true;
        next.next();
    }
}

/// LRU-2 over the small steps (history kept for evicted experts too).
fn run_lru2(
    steps: &[(bool, usize, &[u16])],
    n_expert: usize,
    n_slots: usize,
    seed: Option<&Vec<f32>>,
) -> Outcome {
    let mut s = Slots::new(n_expert, n_slots);
    if let Some(sd) = seed {
        s.fill(&seed_order(sd, n_slots));
    }
    // Per expert: its last and second-to-last small step.
    let mut hist = vec![(0u64, 0u64); n_expert];
    let mut o = Outcome::default();
    let mut tick = 0;
    for &(big, n_tokens, ids) in steps {
        let (experts, _) = distinct(ids, n_expert);
        if big {
            s.fill_big(&experts, tick);
            continue;
        }
        tick += 1;
        for &e in &experts {
            hist[e] = (tick, hist[e].0);
        }
        let (h, m) = s.step(&experts, tick, |s, free| {
            *free
                .iter()
                .min_by_key(|&&x| {
                    let (last, prev) = hist[s.owner[x].unwrap()];
                    (prev, last, x)
                })
                .expect("a free slot")
        });
        o.hits += h;
        o.misses += m;
        o.tokens += n_tokens as u64;
    }
    o
}

/// Belady: the victim is the expert needed again last (or never) by the layer's small steps.
fn run_belady(
    steps: &[(bool, usize, &[u16])],
    n_expert: usize,
    n_slots: usize,
    seed: Option<&Vec<f32>>,
) -> Outcome {
    let sets: Vec<Vec<usize>> = steps
        .iter()
        .map(|(_, _, ids)| distinct(ids, n_expert).0)
        .collect();
    // next[i][e]: the first small step after step i that needs e (usize::MAX: none).
    let mut next = vec![vec![usize::MAX; n_expert]; steps.len()];
    let mut later = vec![usize::MAX; n_expert];
    for i in (0..steps.len()).rev() {
        next[i].clone_from(&later);
        if !steps[i].0 {
            for &e in &sets[i] {
                later[e] = i;
            }
        }
    }
    let mut s = Slots::new(n_expert, n_slots);
    if let Some(sd) = seed {
        s.fill(&seed_order(sd, n_slots));
    }
    let mut o = Outcome::default();
    let mut tick = 0;
    for (i, &(big, n_tokens, _)) in steps.iter().enumerate() {
        if big {
            s.fill_big(&sets[i], tick);
            continue;
        }
        tick += 1;
        let nx = &next[i];
        let (h, m) = s.step(&sets[i], tick, |s, free| {
            *free
                .iter()
                .max_by_key(|&&x| (nx[s.owner[x].unwrap()], std::cmp::Reverse(x)))
                .expect("a free slot")
        });
        o.hits += h;
        o.misses += m;
        o.tokens += n_tokens as u64;
    }
    o
}

/// The routing counts of `trace` (small and big steps) as a profile over the model's layers.
pub fn profile(trace: &Trace) -> Profile {
    let h = &trace.header;
    let n_layers = h.layers.iter().max().map_or(0, |&l| l + 1);
    let mut heat = vec![vec![0f32; h.n_expert]; n_layers];
    for r in &trace.records {
        let l = h.layers[r.ord];
        for &e in &r.ids {
            heat[l][e as usize] += 1.0;
        }
    }
    Profile::from_heat(&heat)
}

/// A profile's ranking as starting heat per tier layer (ordinal), the hottest pair `top`.
pub fn seed_heat(profile: &Profile, layers: &[usize], n_expert: usize, top: f32) -> Vec<Vec<f32>> {
    let h = profile.heat(top);
    layers
        .iter()
        .map(|&l| {
            let mut v = h.get(l).cloned().unwrap_or_default();
            v.resize(n_expert, 0.0);
            v
        })
        .collect()
}
