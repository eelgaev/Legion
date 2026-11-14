//! Which routed experts each MoE layer keeps in its VRAM slots, and when to swap.
//!
//! Every layer has a fixed number of slots (set at load from the VRAM budget and the profile,
//! see [`allocate_slots`]). The graph looks experts up in a per-layer table: a slot index for
//! an expert whose copy in VRAM is complete, `n_slots + expert` (its host bytes, read in place)
//! for any other. After each logged step the engine feeds the looked-up ids back
//! ([`LayerSlots::record`]); they say which routed (token, expert) pairs VRAM served and add
//! to each expert's routing heat, a count that decays every step. [`plan_swaps`] then trades
//! the coldest slot holders for hotter host experts, the biggest gains first, within a budget
//! of copies per step.
//!
//! A swap never lets the graph read a slot while it is being written: the old expert's table
//! entry goes back to its host bytes before the copy starts (the caller writes the table before
//! the next step), and the new expert's entry points at the slot only once the copy is done
//! ([`LayerSlots::complete`]).
//!
//! Sizing the slots by a global ranking of (layer, expert) pairs and the learned ranking follow
//! Strata's expert cache (MIT, eelgaev/Strata-AC922 `src/core/expert_cache.cpp`).

/// A slot's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Empty,
    /// The copy of this expert is complete; the table points here.
    Ready(u32),
    /// This expert is being copied in; nothing reads the slot.
    Filling(u32),
}

/// One layer's slots and routing statistics.
#[derive(Debug, Clone)]
pub struct LayerSlots {
    slots: Vec<Slot>,
    /// Per expert: the slot holding it (ready or filling).
    slot_of: Vec<Option<u32>>,
    heat: Vec<f32>,
    /// Routed (token, expert) pairs served from VRAM / from host memory since the start.
    pub hits: u64,
    pub misses: u64,
    /// The experts this layer's slots serve (expert parallelism: its GPU's share), `None`:
    /// all. The others take no heat, no counts and no slots ([`Self::set_owned`]).
    owned: Option<std::ops::Range<u32>>,
}

/// One planned copy: `new` goes into `slot` of `layer`, replacing `old`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Swap {
    pub layer: usize,
    pub slot: u32,
    pub old: Option<u32>,
    pub new: u32,
}

impl LayerSlots {
    pub fn new(n_expert: usize, n_slots: usize) -> Self {
        Self {
            slots: vec![Slot::Empty; n_slots],
            slot_of: vec![None; n_expert],
            heat: vec![0.0; n_expert],
            hits: 0,
            misses: 0,
            owned: None,
        }
    }

    /// Serves only the experts `owned` from now on (set before the first fill): routings of the
    /// others are not this layer's, so they take no heat nor counts, and never a slot.
    pub fn set_owned(&mut self, owned: std::ops::Range<u32>) {
        for (e, h) in self.heat.iter_mut().enumerate() {
            if !owned.contains(&(e as u32)) {
                *h = 0.0;
            }
        }
        self.owned = Some(owned);
    }

    /// Whether expert `e` is this layer's ([`Self::set_owned`]).
    pub fn owns(&self, e: u32) -> bool {
        self.owned.as_ref().is_none_or(|r| r.contains(&e))
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn n_expert(&self) -> usize {
        self.slot_of.len()
    }

    /// Puts `experts` in slots `0..` as complete copies (the initial fill; the caller copies
    /// the bytes before the first step). At most `n_slots` of them; duplicates are skipped.
    pub fn fill(&mut self, experts: &[u32]) {
        let mut s = 0;
        for &e in experts {
            if s == self.slots.len() {
                break;
            }
            if self.slot_of[e as usize].is_some() {
                continue;
            }
            self.slots[s] = Slot::Ready(e);
            self.slot_of[e as usize] = Some(s as u32);
            s += 1;
        }
    }

    /// Empties every slot (the table then reads every expert from host memory) and returns
    /// the experts the slots held or were filling, in slot order, for [`Self::refill`].
    pub fn clear(&mut self) -> Vec<u32> {
        self.clear_from(0)
    }

    /// [`Self::clear`] of the slots from `first` on: the others keep their experts.
    pub fn clear_from(&mut self, first: usize) -> Vec<u32> {
        let mut held = Vec::new();
        for s in self.slots.iter_mut().skip(first) {
            if let Slot::Ready(e) | Slot::Filling(e) = *s {
                held.push(e);
                self.slot_of[e as usize] = None;
            }
            *s = Slot::Empty;
        }
        held
    }

    /// Re-packs the slots `0..keep` that a lend keeps (`lend_spread`) before it clears the
    /// rest: when the experts they hold are scattered over the expert ids, so that the experts
    /// a long prompt copies in (all the others) fall into many runs of consecutive ids, each a
    /// copy per kind and pass (#778: after decode's swaps, ~130 runs a layer instead of 1-2),
    /// they take the window of `keep` consecutive experts ([`kept_window`]) instead: the
    /// prompt's copies are then at most two runs a layer. A window expert already in a slot
    /// from `keep` on (`tail`: those slots still hold their bytes) is copied from there, the
    /// others from host memory. Returns the copies `(slot, from)` to make before the slots'
    /// VRAM is lent, `from` a slot (`< n_slots`) or `n_slots + expert`, in slot order; empty
    /// when the window would not leave fewer runs, or while a kept slot is filling.
    pub fn repack_kept(&mut self, keep: usize, tail: bool) -> Vec<(u32, u32)> {
        let n_slots = self.slots.len();
        let keep = keep.min(n_slots);
        if keep == 0
            || self.slots[..keep]
                .iter()
                .any(|s| matches!(s, Slot::Filling(_)))
        {
            return Vec::new();
        }
        let n = self.slot_of.len();
        let owned = self.owned.clone().unwrap_or(0..n as u32);
        let at = |e: usize| match self.slot_of[e] {
            Some(s) if self.slots[s as usize] == Slot::Ready(e as u32) => Some(s as usize),
            _ => None,
        };
        let kept: Vec<bool> = (0..n).map(|e| at(e).is_some_and(|s| s < keep)).collect();
        let cached: Vec<bool> = (0..n)
            .map(|e| at(e).is_some_and(|s| s < keep || tail))
            .collect();
        let Some(first) = kept_window(&cached, &kept, owned.clone(), keep) else {
            return Vec::new();
        };
        let window = first as usize..first as usize + keep;
        let in_window: Vec<bool> = (0..n).map(|e| window.contains(&e)).collect();
        if miss_runs(&in_window, owned.clone()) >= miss_runs(&kept, owned) {
            return Vec::new();
        }
        let free: Vec<usize> = (0..keep)
            .filter(|&s| !matches!(self.slots[s], Slot::Ready(e) if in_window[e as usize]))
            .collect();
        let needed: Vec<usize> = window.clone().filter(|&e| !kept[e]).collect();
        debug_assert_eq!(free.len(), needed.len());
        let mut out = Vec::with_capacity(free.len());
        for (&s, &e) in free.iter().zip(&needed) {
            if let Slot::Ready(old) = self.slots[s] {
                self.slot_of[old as usize] = None;
            }
            let from = match self.slot_of[e] {
                Some(t) => {
                    // A slot past `keep`: its bytes if they are there, else host memory's.
                    let ready = self.slots[t as usize] == Slot::Ready(e as u32);
                    self.slots[t as usize] = Slot::Empty;
                    if ready && tail {
                        t
                    } else {
                        (n_slots + e) as u32
                    }
                }
                None => (n_slots + e) as u32,
            };
            self.slots[s] = Slot::Ready(e as u32);
            self.slot_of[e] = Some(s as u32);
            out.push((s as u32, from));
        }
        out
    }

    /// Every expert, the most routed first (ties: lower id first); `None` while none was.
    pub fn ranked(&self) -> Option<Vec<u32>> {
        if !self.heat.iter().any(|&h| h > 0.0) {
            return None;
        }
        let mut r: Vec<u32> = (0..self.heat.len() as u32)
            .filter(|&e| self.owns(e))
            .collect();
        r.sort_by(|a, b| {
            self.heat[*b as usize]
                .total_cmp(&self.heat[*a as usize])
                .then(a.cmp(b))
        });
        Some(r)
    }

    /// Starts copies of `experts` into the empty slots `0..` (after [`Self::clear`]): they
    /// are filling until [`Self::complete`]. Returns the `(slot, expert)` copies to make.
    pub fn refill(&mut self, experts: &[u32]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let mut s = 0;
        for &e in experts {
            if self.slot_of[e as usize].is_some() {
                continue;
            }
            while s < self.slots.len() && self.slots[s] != Slot::Empty {
                s += 1;
            }
            if s == self.slots.len() {
                break;
            }
            self.slots[s] = Slot::Filling(e);
            self.slot_of[e as usize] = Some(s as u32);
            out.push((s as u32, e));
        }
        out
    }

    /// Slots holding a ready expert.
    pub fn ready_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Ready(_)))
            .count()
    }

    /// The experts in slots, per slot (`None`: empty or still filling).
    pub fn ready(&self) -> Vec<Option<u32>> {
        self.slots
            .iter()
            .map(|s| match s {
                Slot::Ready(e) => Some(*e),
                _ => None,
            })
            .collect()
    }

    /// The lookup table the graph reads: per expert, its slot if ready, else
    /// `n_slots + expert`.
    pub fn table(&self) -> Vec<i32> {
        let n = self.slots.len() as i32;
        (0..self.slot_of.len())
            .map(|e| match self.slot_of[e] {
                Some(s) if self.slots[s as usize] == Slot::Ready(e as u32) => s as i32,
                _ => n + e as i32,
            })
            .collect()
    }

    /// Whether a ready slot holds expert `e` (the table reads it from VRAM).
    pub fn serves(&self, e: u32) -> bool {
        matches!(self.slot_of.get(e as usize), Some(&Some(s)) if self.slots[s as usize] == Slot::Ready(e))
    }

    /// Which expert a looked-up id stands for, and whether VRAM served it. `None` for an id
    /// that names no expert (a slot that was not ready: cannot happen with a current table).
    pub fn decode(&self, id: i32) -> Option<(u32, bool)> {
        let n = self.slots.len() as i32;
        if id < 0 {
            None
        } else if id < n {
            match self.slots[id as usize] {
                Slot::Ready(e) => Some((e, true)),
                _ => None,
            }
        } else {
            let e = (id - n) as usize;
            (e < self.slot_of.len()).then_some((e as u32, false))
        }
    }

    /// Adds a step's looked-up ids (the table entries the graph used, one per routed
    /// (token, expert) pair) to the heat and the hit counts.
    pub fn record(&mut self, ids: &[i32]) {
        for &id in ids {
            if let Some((e, hit)) = self.decode(id).filter(|&(e, _)| self.owns(e)) {
                self.heat[e as usize] += 1.0;
                if hit {
                    self.hits += 1;
                } else {
                    self.misses += 1;
                }
            }
        }
    }

    /// [`Self::record`] of one pass's ids with each expert's heat rising by at most `cap` for
    /// the whole pass (`--expert-heat-pass-cap`, #814): an expert that the tokens of one MTP
    /// verify step all chose (consecutive tokens, often the same experts) then counts about as
    /// one routing, so one step's burst alone does not earn it a copy into VRAM; an expert
    /// needs routings in more than one pass to beat a slot holder by the swap margin. The hit
    /// counts are the same as `record`'s. `cap` 0 (or less) is `record` itself.
    pub fn record_capped(&mut self, ids: &[i32], cap: f32) {
        if cap <= 0.0 {
            return self.record(ids);
        }
        let mut seen: Vec<u32> = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some((e, hit)) = self.decode(id).filter(|&(e, _)| self.owns(e)) {
                if hit {
                    self.hits += 1;
                } else {
                    self.misses += 1;
                }
                seen.push(e);
            }
        }
        seen.sort_unstable();
        for run in seen.chunk_by(|a, b| a == b) {
            self.heat[run[0] as usize] += (run.len() as f32).min(cap);
        }
    }

    /// Adds `heat` (per expert, e.g. a saved profile's counts) to the current heat.
    pub fn add_heat(&mut self, heat: &[f32]) {
        for (e, (h, &x)) in self.heat.iter_mut().zip(heat).enumerate() {
            if self.owned.as_ref().is_none_or(|r| r.contains(&(e as u32))) {
                *h += x;
            }
        }
    }

    pub fn heat(&self) -> &[f32] {
        &self.heat
    }

    /// Multiplies every expert's heat by `f` (in 0..=1): older routing counts less.
    pub fn decay(&mut self, f: f32) {
        for h in &mut self.heat {
            *h *= f;
        }
    }

    /// Marks the copy into `slot` complete: its expert's table entry points there from now on.
    pub fn complete(&mut self, slot: u32) {
        if let Slot::Filling(e) = self.slots[slot as usize] {
            self.slots[slot as usize] = Slot::Ready(e);
        }
    }

    /// Candidate swaps `(gain, slot, old, new)`: the hottest host experts against the coldest
    /// slots (empty ones first), while the newcomer is hotter than the holder by more than
    /// `margin` (absolute) and `ratio` (relative). At most `limit` of them: only the `limit`
    /// hottest host experts and coldest slots are ranked (a full sort of every layer after
    /// every step cost ~1 ms of host time on the decode path), with ties broken by expert id
    /// and slot index, as a stable sort of both lists would break them.
    fn candidates(
        &self,
        margin: f32,
        ratio: f32,
        limit: usize,
    ) -> Vec<(f32, u32, Option<u32>, u32)> {
        let mut hot: Vec<u32> = (0..self.slot_of.len() as u32)
            .filter(|&e| self.slot_of[e as usize].is_none() && self.heat[e as usize] > 0.0)
            .collect();
        let by_heat = |a: &u32, b: &u32| {
            self.heat[*b as usize]
                .total_cmp(&self.heat[*a as usize])
                .then(a.cmp(b))
        };
        top(&mut hot, limit, by_heat);
        let mut cold: Vec<(f32, u32, Option<u32>)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, slot)| match *slot {
                Slot::Empty => Some((f32::NEG_INFINITY, s as u32, None)),
                Slot::Ready(e) => Some((self.heat[e as usize], s as u32, Some(e))),
                Slot::Filling(_) => None,
            })
            .collect();
        top(&mut cold, limit, |a, b| {
            a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
        });
        hot.iter()
            .zip(&cold)
            .map_while(|(&new, &(h_old, slot, old))| {
                let h_new = self.heat[new as usize];
                let gain = h_new - h_old.max(0.0);
                (old.is_none() || (gain > margin && h_new > h_old * (1.0 + ratio)))
                    .then_some((gain, slot, old, new))
            })
            .collect()
    }

    fn start(&mut self, slot: u32, old: Option<u32>, new: u32) {
        if let Some(o) = old {
            self.slot_of[o as usize] = None;
        }
        self.slots[slot as usize] = Slot::Filling(new);
        self.slot_of[new as usize] = Some(slot);
    }
}

/// The first expert of the window of `keep` consecutive experts inside `owned` that holds the
/// most `cached` experts (fewest to copy in from host memory), then the most `kept` ones
/// (fewest to move), then the lowest; `None` when no window fits (or `keep` is 0).
pub fn kept_window(
    cached: &[bool],
    kept: &[bool],
    owned: std::ops::Range<u32>,
    keep: usize,
) -> Option<u32> {
    let (lo, hi) = (owned.start as usize, (owned.end as usize).min(cached.len()));
    if keep == 0 || hi < lo + keep {
        return None;
    }
    let count = |v: &[bool], r: std::ops::Range<usize>| v[r].iter().filter(|&&b| b).count();
    let (mut c, mut k) = (count(cached, lo..lo + keep), count(kept, lo..lo + keep));
    let mut best = (c, k, lo);
    for a in lo + 1..=hi - keep {
        let (out, add) = (a - 1, a + keep - 1);
        c = c + cached[add] as usize - cached[out] as usize;
        k = k + kept[add] as usize - kept[out] as usize;
        if (c, k) > (best.0, best.1) {
            best = (c, k, a);
        }
    }
    Some(best.2 as u32)
}

/// Runs of consecutive experts inside `owned` that are not `kept`: what a long prompt copies
/// in, one copy per run and kind.
pub fn miss_runs(kept: &[bool], owned: std::ops::Range<u32>) -> usize {
    let (lo, hi) = (owned.start as usize, (owned.end as usize).min(kept.len()));
    (lo..hi)
        .filter(|&e| !kept[e] && (e == lo || kept[e - 1]))
        .count()
}

/// Keeps the first `n` of `v` in the order `cmp` (a total order) gives, sorted: the first `n`
/// of a full sort, at the cost of a selection.
fn top<T>(v: &mut Vec<T>, n: usize, mut cmp: impl FnMut(&T, &T) -> std::cmp::Ordering) {
    if n == 0 {
        v.clear();
        return;
    }
    if v.len() > n {
        v.select_nth_unstable_by(n - 1, &mut cmp);
        v.truncate(n);
    }
    v.sort_unstable_by(cmp);
}

/// When to swap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwapRule {
    /// Copies per planning round over all layers given to it.
    pub budget: usize,
    /// A newcomer must be hotter than the slot holder by this much (routings)...
    pub margin: f32,
    /// ...and by this fraction of the holder's heat.
    pub ratio: f32,
}

/// The default's margin and ratio follow Strata-AC922's adaptive tier (eelgaev/Strata-AC922,
/// MIT): the `adapt` rule in src/program/generate.cpp at 8eef05e lets a missed expert in when
/// its usage beats the victim's by 1.5, with no ratio. Budget and decay are tuned for Legion's
/// planning, which runs after every pass (MTP passes included) with 24 copies per device and
/// heat x0.95 per pass, where Strata plans every 4 windows with up to 96 swaps and x0.7 per
/// round (`--adapt-decay`, c38dc71); Legion also keeps its strict gain test and a floor of
/// 0 instead of Strata's minimum usage of 2. Measured on 2 V100s it serves 4-7 points more
/// routings from VRAM than margin 2 / ratio 0.25 / 8 copies / x0.995.
impl Default for SwapRule {
    fn default() -> Self {
        Self {
            budget: 24,
            margin: 1.5,
            ratio: 0.0,
        }
    }
}

impl SwapRule {
    /// This rule with device `d`'s budget taken from `per_device` (`--expert-swaps-device`,
    /// index = the cache's device order: the device of the first expert layers first) when it
    /// names one; otherwise the rule's own.
    pub fn on_device(self, per_device: &[usize], d: usize) -> Self {
        Self {
            budget: per_device.get(d).copied().unwrap_or(self.budget),
            ..self
        }
    }
}

/// Plans up to `rule.budget` swaps over `layers` (index = layer id), the largest heat gains
/// first, and marks them started (old entries leave the table now; new ones join on
/// [`LayerSlots::complete`]). The caller copies each new expert into its slot.
pub fn plan_swaps(layers: &mut [Option<&mut LayerSlots>], rule: SwapRule) -> Vec<Swap> {
    let mut all: Vec<(f32, Swap)> = Vec::new();
    for (layer, l) in layers.iter().enumerate() {
        let Some(l) = l else { continue };
        for (gain, slot, old, new) in l.candidates(rule.margin, rule.ratio, rule.budget) {
            all.push((
                gain,
                Swap {
                    layer,
                    slot,
                    old,
                    new,
                },
            ));
        }
    }
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    all.truncate(rule.budget);
    for (_, s) in &all {
        if let Some(l) = layers[s.layer].as_mut() {
            l.start(s.slot, s.old, s.new);
        }
    }
    all.into_iter().map(|(_, s)| s).collect()
}

/// Splits a VRAM budget into per-layer slot counts and picks each layer's first experts:
/// walks `ranked` (`(layer, expert)` pairs, hottest first) and takes every pair whose layer has
/// a slot size (`slot_bytes[layer] > 0`: the layer's experts are on this device) while the
/// bytes fit in `budget`. A layer the ranking does not name at all (e.g. an MTP head's, past a
/// profile of the main layers) is first given an even share of the budget. Layers are then
/// filled up in round-robin order with the experts the ranking does not name (lowest ids
/// first), while bytes remain, so a short or missing ranking still uses the whole budget.
/// Returns each layer's experts in slot order.
pub fn allocate_slots(
    ranked: &[(usize, u32)],
    slot_bytes: &[u64],
    n_expert: usize,
    budget: u64,
) -> Vec<Vec<u32>> {
    allocate_slots_in(ranked, slot_bytes, &[], n_expert, budget)
}

/// The VRAM mapping granule the slot plan rounds to (2 MiB).
pub const SLOT_GRANULE: u64 = 2 << 20;

/// How often a device's slots may fail to find VRAM at load before the load fails (#1208).
/// [`shrunk_budget`] reaches 0 well before.
pub const MAX_FIT_TRIES: u32 = 48;

/// A device's slot budget after its slots found no VRAM at load (#1208): `budget` less what
/// other processes took since the plan (`drift`: the free VRAM the plan saw less what is free
/// with the device's slots given back) and a granule, so the room kept beside the cache stays
/// free; and less at least one slot of the failed layer (`slot` bytes), doubled for each of the
/// device's `failures` before, so repeated failures where the free figure tells nothing (a
/// GB10's shared memory) still shrink the cache to 0 in a few tries.
pub fn shrunk_budget(budget: u64, drift: u64, slot: u64, failures: u32) -> u64 {
    let step = slot
        .max(SLOT_GRANULE)
        .saturating_mul(1u64 << failures.min(40));
    budget.saturating_sub(step.max(drift.saturating_add(SLOT_GRANULE)))
}

/// [`allocate_slots`] with each layer's slots for its `owned` experts only (expert
/// parallelism; a layer past `owned`'s end, or `None`, any): the others are never picked, nor
/// counted in a layer's even share.
pub fn allocate_slots_in(
    ranked: &[(usize, u32)],
    slot_bytes: &[u64],
    owned: &[Option<std::ops::Range<u32>>],
    n_expert: usize,
    budget: u64,
) -> Vec<Vec<u32>> {
    let n_layer = slot_bytes.len();
    let mut out = vec![Vec::new(); n_layer];
    // Experts a layer may not take start out taken.
    let mut taken: Vec<Vec<bool>> = (0..n_layer)
        .map(|l| match owned.get(l).cloned().flatten() {
            Some(r) => (0..n_expert as u32).map(|e| !r.contains(&e)).collect(),
            None => vec![false; n_expert],
        })
        .collect();
    let mine = |l: usize| -> usize { taken[l].iter().filter(|&&t| !t).count() };
    let mine: Vec<usize> = (0..n_layer).map(mine).collect();
    let mut left = budget;
    let here = slot_bytes.iter().filter(|&&b| b > 0).count() as u64;
    if !ranked.is_empty() && here > 0 {
        let named: std::collections::HashSet<usize> = ranked.iter().map(|&(l, _)| l).collect();
        for l in (0..n_layer).filter(|l| slot_bytes[*l] > 0 && !named.contains(l)) {
            let n = ((budget / here) / slot_bytes[l]).min(mine[l] as u64) as usize;
            let first: Vec<u32> = (0..n_expert as u32)
                .filter(|&e| !taken[l][e as usize])
                .take(n)
                .collect();
            for &e in &first {
                taken[l][e as usize] = true;
            }
            out[l].extend(first);
            left -= n as u64 * slot_bytes[l];
        }
    }
    for &(l, e) in ranked {
        if l >= n_layer || e as usize >= n_expert || slot_bytes[l] == 0 || taken[l][e as usize] {
            continue;
        }
        if slot_bytes[l] > left {
            break;
        }
        left -= slot_bytes[l];
        taken[l][e as usize] = true;
        out[l].push(e);
    }
    let mut next = vec![0usize; n_layer];
    loop {
        let mut any = false;
        for l in 0..n_layer {
            if slot_bytes[l] == 0 || slot_bytes[l] > left {
                continue;
            }
            while next[l] < n_expert && taken[l][next[l]] {
                next[l] += 1;
            }
            if next[l] == n_expert {
                continue;
            }
            taken[l][next[l]] = true;
            out[l].push(next[l] as u32);
            left -= slot_bytes[l];
            any = true;
        }
        if !any {
            break;
        }
    }
    out
}

/// Each layer's slot count of `alloc` ([`allocate_slots_in`]'s result) with the layer's lowest
/// experts it may take (`owned`, as there) in slot order instead: the ranking's per-layer
/// split with the contents a cache without a profile starts with
/// (`--expert-profile-split-only`).
pub fn lowest_experts(
    alloc: &[Vec<u32>],
    owned: &[Option<std::ops::Range<u32>>],
    n_expert: usize,
) -> Vec<Vec<u32>> {
    alloc
        .iter()
        .enumerate()
        .map(|(l, v)| {
            let r = owned.get(l).cloned().flatten();
            (0..n_expert as u32)
                .filter(|e| r.as_ref().is_none_or(|r| r.contains(e)))
                .take(v.len())
                .collect()
        })
        .collect()
}

/// Each layer's slot count of `alloc` with the layer's most routed experts first (`order`,
/// per layer, hottest first), then the experts `alloc` gave it that are not among them: the
/// bounded host tier's saved heat picks the first contents, the layers' sizes stay.
pub fn hottest_experts(alloc: &[Vec<u32>], order: &[Vec<u32>]) -> Vec<Vec<u32>> {
    alloc
        .iter()
        .enumerate()
        .map(|(l, v)| {
            let hot = order.get(l).map_or(&[][..], Vec::as_slice);
            let mut taken = std::collections::HashSet::new();
            hot.iter()
                .chain(v)
                .copied()
                .filter(|e| taken.insert(*e))
                .take(v.len())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests;

/// How many slots each staged layer of a staging device keeps through a lend when nothing sizes
/// it (`ResidencyConfig::lend_need`). `layers`: each staged layer's slots and one slot's bytes
/// (its three kinds); `unstaged`: the slot bytes of the device's layers that lend them all
/// and stage nothing (an MTP head's). Every staged layer keeps the same share of its slots,
/// the biggest whose lent rest holds the staging ring (`ring` buffers per kind, each the
/// biggest layer's uncached experts: at most `n_expert` less its kept slots, and a pad) and
/// `need` more. `n_expert`: the most experts a layer stages (an expert-parallel entry's own
/// share). All zero if even lending every slot leaves less.
pub fn auto_keep(
    layers: &[(usize, u64)],
    unstaged: u64,
    n_expert: usize,
    ring: usize,
    need: u64,
) -> Vec<usize> {
    auto_keep_where(layers, unstaged, n_expert, ring, need, |_| true)
}

/// Each kind's range splits at a mapping granule, and each staging buffer pads to one.
const GRANULE: u64 = 2 << 20;

/// [`auto_keep`]'s search, where a share must also pass `also`.
fn auto_keep_where(
    layers: &[(usize, u64)],
    unstaged: u64,
    n_expert: usize,
    ring: usize,
    need: u64,
    mut also: impl FnMut(&[usize]) -> bool,
) -> Vec<usize> {
    let keep = |share: u64| -> Vec<usize> {
        layers
            .iter()
            .map(|&(n, _)| (n as u64 * share / 1024) as usize)
            .collect()
    };
    let fits = |keep: &[usize]| {
        let lent: u64 = layers
            .iter()
            .zip(keep)
            .map(|(&(n, b), &k)| (n - k) as u64 * b)
            .sum::<u64>()
            + unstaged;
        let pads = 3 * GRANULE * layers.len() as u64;
        let biggest = layers
            .iter()
            .zip(keep)
            .map(|(&(_, b), &k)| (n_expert.saturating_sub(k) + 1) as u64 * b + 3 * GRANULE)
            .max()
            .unwrap_or(0);
        lent.saturating_sub(pads) >= ring as u64 * biggest + need
    };
    // Neighbouring shares often round to the same slots: `also` sees each keep once.
    let mut last: Option<(Vec<usize>, bool)> = None;
    (1..=1024)
        .rev()
        .map(keep)
        .find(|k| {
            fits(k) && {
                if last.as_ref().is_none_or(|(l, _)| l != k) {
                    last = Some((k.clone(), also(k)));
                }
                last.as_ref().is_some_and(|(_, ok)| *ok)
            }
        })
        .unwrap_or_else(|| vec![0; layers.len()])
}

/// A layer of a staging device for [`auto_keep_carved`]: its slots, one slot's bytes per
/// expert kind, the bytes the quantized matmuls read past a slot's last row per kind (see
/// [`kept_slots`]), and whether it stages (an MTP head's layer lends all of its slots and stages
/// nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LendLayer {
    pub slots: usize,
    pub bytes: [usize; 3],
    pub over: [usize; 3],
    pub staged: bool,
}

/// The slots of a tiered tensor kept while its tail is lent: those that end, with the `over`
/// bytes the quantized matmuls read past a slot's last row
/// ([`crate::ggml::mapped_tail_padding`]), by `split`, where the tail starts. Slot 0 starts at
/// `first`; slots are `slot_bytes` each. With the tail lent nothing is mapped from `split` on,
/// so a slot ending right at it (the split is cut at a granule, and a slot can end on one)
/// would read unmapped memory (#814).
pub fn kept_slots(
    first: usize,
    split: usize,
    slot_bytes: usize,
    over: usize,
    n_slots: usize,
) -> usize {
    (split.saturating_sub(first + over) / slot_bytes.max(1)).min(n_slots)
}

/// [`auto_keep`] over `layers` (in the device's layer order), checked against how a lend makes
/// its staging buffers: of whole lent allocations (each staged layer's slot tail per kind, an
/// unstaged layer's slots per kind), the biggest buffers first, each from the parts that cover
/// it with the least left over ([`super::experts::plan_staging`], as `Staging::new` does;
/// `cover`: [`super::ResidencyConfig::stage_reuse`], else the parts that fit and new VRAM for
/// the rest). A part bigger than its buffer stays with it whole, so [`auto_keep`]'s byte count
/// can leave far less free than it assumes: on a stage of few layers that keeps nearly all of
/// its slots, the ring takes the MTP head's whole allocations and needs new VRAM besides, and
/// the prompt's compute buffers spill into host memory (#767). When the parts the buffers leave
/// (less the new VRAM they take) fall short of `need` by more than the rounding [`auto_keep`]
/// allows for, the share goes down until they do not.
///
/// Which of a device's layers a restore leaves without VRAM so that `need` bytes stay free
/// (`--kv-vmm-keep`'s mapped KV tail, #814): `lent[i]`, in the device's layer order, is the
/// bytes of layer `i`'s lent slots when it may stay without (`None`: it is refilled). From the
/// last layer down, until the bytes left reach `need` (all candidates when they never do).
pub fn layers_left_for(lent: &[Option<u64>], need: u64) -> Vec<bool> {
    let mut left = vec![false; lent.len()];
    let mut sum = 0u64;
    for (i, bytes) in lent.iter().enumerate().rev() {
        if sum >= need {
            break;
        }
        if let Some(b) = bytes {
            sum += b;
            left[i] = true;
        }
    }
    left
}

/// [`layers_left_for`] for the bounded host tier, where a slot without VRAM costs a copy over
/// PCIe per routed token: the candidates whose lent slots held the least routing heat
/// (`heat[i]`) go first, the later layer of two equal ones before the earlier.
pub fn layers_left_coldest(lent: &[Option<u64>], heat: &[f32], need: u64) -> Vec<bool> {
    let mut left = vec![false; lent.len()];
    let mut order: Vec<usize> = (0..lent.len()).filter(|&i| lent[i].is_some()).collect();
    let h = |i: usize| heat.get(i).copied().unwrap_or(0.0);
    order.sort_by(|&a, &b| h(a).total_cmp(&h(b)).then(b.cmp(&a)));
    let mut sum = 0u64;
    for i in order {
        if sum >= need {
            break;
        }
        sum += lent[i].unwrap_or(0);
        left[i] = true;
    }
    left
}

/// Returns the keep, and [`auto_keep`]'s byte-count keep (the same unless it was lowered).
pub fn auto_keep_carved(
    layers: &[LendLayer],
    n_expert: usize,
    ring: usize,
    need: u64,
    cover: bool,
) -> (Vec<usize>, Vec<usize>) {
    let slot = |l: &LendLayer| l.bytes.iter().sum::<usize>() as u64;
    let staged: Vec<(usize, u64)> = layers
        .iter()
        .filter(|l| l.staged)
        .map(|l| (l.slots, slot(l)))
        .collect();
    let unstaged = layers
        .iter()
        .filter(|l| !l.staged)
        .map(|l| l.slots as u64 * slot(l))
        .sum();
    let plain = auto_keep(&staged, unstaged, n_expert, ring, need);
    if plain.iter().all(|&k| k == 0) {
        return (plain.clone(), plain);
    }
    // `Staging::new` makes at least two buffers per kind.
    let bufs = ring.max(super::PREFILL_RING);
    let pads = 3 * GRANULE * staged.len() as u64;
    let fits = |keep: &[usize]| {
        let (parts, bytes) = lend_parts(layers, keep, n_expert, GRANULE as usize);
        let plan = super::experts::plan_staging(&parts, bytes, bufs, cover);
        let mut taken = vec![false; parts.len()];
        let mut new = 0u64;
        for (i, picked) in &plan {
            let have: usize = picked.iter().map(|&p| parts[p]).sum();
            let want = bytes[i / bufs].max(1).next_multiple_of(GRANULE as usize);
            new += want.saturating_sub(have) as u64;
            for &p in picked {
                taken[p] = true;
            }
        }
        let left: u64 = parts
            .iter()
            .zip(&taken)
            .filter(|(_, t)| !**t)
            .map(|(&p, _)| p as u64)
            .sum();
        left.saturating_sub(new) + pads >= need
    };
    if fits(&plain) {
        return (plain.clone(), plain);
    }
    let keep = auto_keep_where(&staged, unstaged, n_expert, ring, need, fits);
    (keep, plain)
}

/// What a lend with `keep` slots kept per staged layer (in order) unmaps, in the order it
/// unmaps them (per layer, per kind; the sizes a tiered range of `gran` granules gives its
/// slot tail, or all of an unstaged layer's slots), and the staging buffer bytes per kind
/// (the most a staged layer packs: its kept slots' ids, every other expert of `n_expert`, and
/// one expert more; as `Layer::pack`).
fn lend_parts(
    layers: &[LendLayer],
    keep: &[usize],
    n_expert: usize,
    gran: usize,
) -> (Vec<usize>, [usize; 3]) {
    let mut parts = Vec::new();
    let mut bytes = [0usize; 3];
    let mut keep = keep.iter();
    for l in layers {
        // Per kind: the slots' range (rounded up to a granule, the slots at its end) and
        // where its tail starts.
        let range = |b: usize, kept: Option<usize>| {
            let vram = (l.slots * b).next_multiple_of(gran);
            let first = vram - l.slots * b;
            let split = match kept {
                Some(k) if k < l.slots => (first + k * b) / gran * gran,
                _ => vram,
            };
            (vram, first, split)
        };
        if !l.staged {
            for &b in &l.bytes {
                parts.push(range(b, None).0);
            }
            continue;
        }
        let kept = keep.next().copied().unwrap_or(0);
        let geo = l.bytes.map(|b| range(b, Some(kept)));
        for &(vram, _, split) in &geo {
            parts.push(vram - split);
        }
        // Per kind: the bytes of slots before the tail, and one slot's.
        let head = |k: usize| (geo[k].2.saturating_sub(geo[k].1), l.bytes[k].max(1));
        let n_kept = (0..3)
            .map(|k| kept_slots(geo[k].1, geo[k].2, l.bytes[k], l.over[k], l.slots))
            .min()
            .unwrap_or(0);
        let stage_first = (0..3)
            .map(|k| head(k).0.div_ceil(head(k).1))
            .max()
            .unwrap_or(0);
        let n_ids = stage_first + n_expert.saturating_sub(n_kept) + 1;
        for k in 0..3 {
            let (_, first, split) = geo[k];
            bytes[k] = bytes[k].max((first + n_ids * l.bytes[k]).saturating_sub(split));
        }
    }
    parts.retain(|&p| p > 0);
    (parts, bytes)
}
