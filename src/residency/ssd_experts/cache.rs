//! One MoE layer's expert slots: which expert each slot holds, least recently used out first.

/// The slots of one layer: `n_slots` of them over `n_expert` experts.
#[derive(Debug, Clone)]
pub struct SlotCache {
    /// Expert -> its slot, or -1.
    slot_of: Vec<i32>,
    /// Slot -> its expert, or -1 (never filled).
    owner: Vec<i32>,
    /// Slot -> the step that last used it.
    used: Vec<u64>,
    /// Slot -> a read into it is in flight: never a victim ([`Self::pin`]).
    pinned: Vec<bool>,
    /// The current step (starts at 1, so 0 is "never").
    tick: u64,
}

/// What a step must do to have all its experts in slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    /// Experts already in a slot.
    pub hits: usize,
    /// `(expert, slot)`: experts to read into a slot (whatever it held is evicted).
    pub loads: Vec<(usize, usize)>,
}

impl SlotCache {
    pub fn new(n_expert: usize, n_slots: usize) -> Self {
        Self {
            slot_of: vec![-1; n_expert],
            owner: vec![-1; n_slots],
            used: vec![0; n_slots],
            pinned: vec![false; n_slots],
            tick: 0,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.owner.len()
    }

    /// The slot holding `expert`.
    pub fn slot(&self, expert: usize) -> Option<usize> {
        usize::try_from(self.slot_of[expert]).ok()
    }

    /// Slots that hold an expert.
    pub fn filled(&self) -> usize {
        self.owner.iter().filter(|&&e| e >= 0).count()
    }

    /// A new step that needs `experts` (distinct, at most [`Self::n_slots`]): marks the cached
    /// ones used and gives every other one a slot, an empty one first, else the least recently
    /// used one this step does not need (lowest slot on ties).
    pub fn plan(&mut self, experts: &[usize]) -> Plan {
        assert!(
            experts.len() <= self.n_slots(),
            "{} experts for {} slots",
            experts.len(),
            self.n_slots()
        );
        self.tick += 1;
        let mut plan = Plan::default();
        let mut missing = Vec::new();
        for &e in experts {
            match self.slot(e) {
                Some(s) => {
                    self.used[s] = self.tick;
                    plan.hits += 1;
                }
                None => missing.push(e),
            }
        }
        for e in missing {
            let s = self.victim().expect("a slot this step does not use");
            self.assign(e, s);
            plan.loads.push((e, s));
        }
        plan
    }

    /// [`Self::plan`] for a piece of a chunked step (#1205), `next` per expert the first row of
    /// the step's later pieces that routes to it (`u32::MAX`: none): an empty slot first, else
    /// the expert the step needs again latest or never, the least recently used of those (the
    /// lowest slot on ties). With every `next` at `u32::MAX` the same slots as [`Self::plan`].
    pub fn plan_ahead(&mut self, experts: &[usize], next: &[u32]) -> Plan {
        assert!(
            experts.len() <= self.n_slots(),
            "{} experts for {} slots",
            experts.len(),
            self.n_slots()
        );
        self.tick += 1;
        let mut plan = Plan::default();
        let mut missing = Vec::new();
        for &e in experts {
            match self.slot(e) {
                Some(s) => {
                    self.used[s] = self.tick;
                    plan.hits += 1;
                }
                None => missing.push(e),
            }
        }
        for e in missing {
            let s = self
                .victim_ahead(next)
                .expect("a slot this step does not use");
            self.assign(e, s);
            plan.loads.push((e, s));
        }
        plan
    }

    /// [`Self::plan`] for a step that may need more experts than the slots hold: the experts
    /// no slot is left for (every slot is this step's or pinned) come back second, in order,
    /// and are not cached.
    pub fn plan_some(&mut self, experts: &[usize]) -> (Plan, Vec<usize>) {
        self.tick += 1;
        let mut plan = Plan::default();
        let mut missing = Vec::new();
        for &e in experts {
            match self.slot(e) {
                Some(s) => {
                    self.used[s] = self.tick;
                    plan.hits += 1;
                }
                None => missing.push(e),
            }
        }
        let mut over = Vec::new();
        for e in missing {
            match self.victim() {
                Some(s) => {
                    self.assign(e, s);
                    plan.loads.push((e, s));
                }
                None => over.push(e),
            }
        }
        (plan, over)
    }

    /// A prefetch of `experts` (distinct, best first) for the layer's next step: marks the
    /// cached ones used, then gives each other one, while `budget` lasts, an empty slot or the
    /// least recently used one the prediction does not keep. Returns the `(expert, slot)` loads.
    pub fn prefetch(&mut self, experts: &[usize], budget: usize) -> Vec<(usize, usize)> {
        self.tick += 1;
        let mut missing = Vec::new();
        for &e in experts {
            match self.slot(e) {
                Some(s) => self.used[s] = self.tick,
                None => missing.push(e),
            }
        }
        let mut loads = Vec::new();
        for e in missing.into_iter().take(budget) {
            let Some(s) = self.victim() else { break };
            self.assign(e, s);
            loads.push((e, s));
        }
        loads
    }

    /// Empties slot `s` (a prefetch into it was dropped before it completed), and unpins it.
    pub fn drop_slot(&mut self, s: usize) {
        if let Ok(old) = usize::try_from(self.owner[s]) {
            self.slot_of[old] = -1;
        }
        self.owner[s] = -1;
        self.used[s] = 0;
        self.pinned[s] = false;
    }

    /// Pins slot `s` (a read into it is in flight: no step or prefetch may take it for another
    /// expert) or unpins it. Its expert still counts as cached.
    pub fn pin(&mut self, s: usize, on: bool) {
        self.pinned[s] = on;
    }

    /// Puts `expert` (not cached) into an empty slot, if there is one; used by steps that read
    /// their experts elsewhere and only fill the slots no expert holds yet.
    pub fn take_empty(&mut self, expert: usize) -> Option<usize> {
        if self.slot(expert).is_some() {
            return None;
        }
        let s = self.owner.iter().position(|&o| o < 0)?;
        self.assign(expert, s);
        Some(s)
    }

    /// Takes a slot for bytes that arrive later (the bounded host tier's write-back of an
    /// expert a VRAM slot drops): an empty one first, else the least recently used one, whose
    /// expert is evicted. The slot is left empty and pinned until [`Self::put`] or
    /// [`Self::drop_slot`]. `None`: every slot is pinned or the current step's.
    pub fn reserve(&mut self) -> Option<usize> {
        let s = self.victim()?;
        self.drop_slot(s);
        self.pinned[s] = true;
        Some(s)
    }

    /// Puts `expert` (not cached) into slot `s` that [`Self::reserve`] took, and unpins it.
    pub fn put(&mut self, expert: usize, s: usize) {
        debug_assert!(self.slot(expert).is_none() && self.owner[s] < 0);
        self.assign(expert, s);
        self.pinned[s] = false;
    }

    fn victim(&self) -> Option<usize> {
        if let Some(s) = (0..self.n_slots()).find(|&s| self.owner[s] < 0 && !self.pinned[s]) {
            return Some(s);
        }
        (0..self.n_slots())
            .filter(|&s| self.used[s] < self.tick && !self.pinned[s])
            .min_by_key(|&s| self.used[s])
    }

    /// [`Self::victim`] that keeps the experts needed soonest (`next`, [`Self::plan_ahead`]).
    fn victim_ahead(&self, next: &[u32]) -> Option<usize> {
        if let Some(s) = (0..self.n_slots()).find(|&s| self.owner[s] < 0 && !self.pinned[s]) {
            return Some(s);
        }
        (0..self.n_slots())
            .filter(|&s| self.used[s] < self.tick && !self.pinned[s])
            .min_by_key(|&s| {
                let e = self.owner[s] as usize;
                (
                    std::cmp::Reverse(next.get(e).copied().unwrap_or(u32::MAX)),
                    self.used[s],
                )
            })
    }

    fn assign(&mut self, expert: usize, s: usize) {
        if let Ok(old) = usize::try_from(self.owner[s]) {
            self.slot_of[old] = -1;
        }
        self.owner[s] = expert as i32;
        self.slot_of[expert] = s as i32;
        self.used[s] = self.tick;
    }
}
