//! Routing-gated staging (#37): a prompt step copies only the uncached experts its tokens
//! route to, not every uncached expert of the layer.
//!
//! A short prompt leaves many of a layer's experts without a token (on Qwen3.8-Flash-Next about
//! 42% at 2K tokens, 29% at 8K), and a layer's staging copy is the whole cost of such a step
//! when the experts come over PCIe. So each layer's staging op reads the router's choices back
//! (a small copy and a wait), and queues the copies of the chosen experts only; the expert op
//! reads the same bytes for every expert it uses, so the output does not change.
//!
//! The wait for the router leaves the copy stream idle while the previous layer's experts and
//! this layer's attention run. Strata's staged walk hides its readback behind the shared
//! expert (eelgaev/Strata-AC922, `src/prefill/prefill.cpp`, MIT; the idea only); here the
//! copy stream instead runs ahead into the next layer once a layer's chosen experts are queued:
//! its hottest staged experts by routing heat, up to a byte budget per token of the step
//! ([`GateRow::ahead`]). The next layer's own gate then copies the chosen experts that are
//! still missing.
//!
//! With the bounded host tier (#37 round 40, a machine with less RAM than the experts) a row's
//! experts come from RAM slots or from the checkpoint, read into pinned staging places by the
//! pass's reader thread ([`super::host_tier`]). There the gate also tells the reader which
//! experts the layer's tokens route to, so it reads only those from the SSD (the reader reads
//! the layer's hottest ones ahead while the router runs), and the row's copies out of the
//! places wait for the reads as before ([`GateRow::with_reads`]).
//!
//! This module plans the copies (plain functions, CPU-tested); [`super::staging`] queues them.

#[cfg(test)]
mod tests;

/// `(dst, src, bytes)` as [`super::staging`]'s copies: `src == 0` zeroes `bytes` at `dst`.
pub type Copy = (usize, usize, usize);

/// Bytes per copy at most (as the full plans' pieces).
pub const PIECE: usize = 32 << 20;

/// The largest step (tokens) that gates by default: the keeping lend's pass size on a 24 GB
/// GPU, past which most experts get a token anyway.
pub const GATE_UPTO: usize = 8704;

/// Default bytes copied ahead into the next layer per token of the step: about what the link
/// moves while a layer computes on an RTX 4090 (~3.6 us per token and layer at ~26 GB/s).
pub const AHEAD_PER_TOKEN: usize = 96 << 10;

/// One expert kind of a gated row: the staged tensor's data address, one expert's bytes, and
/// the bytes zeroed past a copied expert whose neighbour is not copied (0: none; the quantized
/// matmuls read up to a padded row past an expert's last row, see `Layer::stage_copies`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kind {
    pub staged: usize,
    pub bytes: usize,
    pub pad: usize,
}

/// A staged expert: its expert id, its id in the staged tensors and its host address per kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    pub expert: u32,
    pub id: usize,
    pub src: [usize; 3],
}

/// What a gated row's copies are made from.
#[derive(Debug, Clone, PartialEq)]
pub struct GateRow {
    kinds: [Kind; 3],
    /// In increasing staged id order.
    entries: Vec<Entry>,
    /// Per expert, its index in `entries` (`u32::MAX`: not staged).
    of: Vec<u32>,
    /// The zeroed pads every pass queues whatever the routing (where the packed ids start, past
    /// the last id).
    fixed: Vec<Copy>,
    /// Ids whose first bytes must stay finite though no copied entry precedes them (the id
    /// after the last VRAM slot): their pad is zeroed unless they are copied.
    guard: Vec<usize>,
    /// `entries` indices, hottest first.
    hot: Vec<u32>,
    /// The staged id past the last one: `fixed` zeroes its pad.
    end: usize,
    /// With the bounded host tier, per entry the read batch that fills its source (`None`: a
    /// RAM slot); empty: every source is in host memory ([`Self::with_reads`]).
    batch: Vec<Option<usize>>,
    /// With the bounded host tier, per kind the first RAM slot's address ([`Self::to_slot`]).
    ram: [usize; 3],
}

impl GateRow {
    /// `entries` in increasing id order; `heat` per expert (missing: 0).
    pub fn new(
        kinds: [Kind; 3],
        entries: Vec<Entry>,
        n_expert: usize,
        fixed: Vec<Copy>,
        guard: Vec<usize>,
        end: usize,
        heat: &[f32],
    ) -> Self {
        debug_assert!(entries.windows(2).all(|w| w[0].id < w[1].id));
        let mut of = vec![u32::MAX; n_expert];
        for (i, e) in entries.iter().enumerate() {
            if let Some(o) = of.get_mut(e.expert as usize) {
                *o = i as u32;
            }
        }
        let h = |i: u32| {
            heat.get(entries[i as usize].expert as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let mut hot: Vec<u32> = (0..entries.len() as u32).collect();
        hot.sort_by(|&a, &b| h(b).total_cmp(&h(a)));
        Self {
            kinds,
            entries,
            of,
            fixed,
            guard,
            hot,
            end,
            batch: Vec::new(),
            ram: [0; 3],
        }
    }

    /// The row with the bounded host tier's sources: entry `i` (expert `experts[i]`) copies
    /// from `src[i]`, once read batch `batch[i]` is in (`None`: a RAM slot, nothing to wait
    /// for); `ram` per kind the first RAM slot's address. `None` when the lists do not name
    /// this row's entries in order.
    pub fn with_reads(
        mut self,
        experts: &[u32],
        src: Vec<[usize; 3]>,
        batch: Vec<Option<usize>>,
        ram: [usize; 3],
    ) -> Option<Self> {
        let n = self.entries.len();
        let same = experts.len() == n
            && experts
                .iter()
                .zip(&self.entries)
                .all(|(&e, x)| x.expert == e);
        if !same || src.len() != n || batch.len() != n {
            return None;
        }
        for (e, s) in self.entries.iter_mut().zip(src) {
            e.src = s;
        }
        // Only RAM sources are copied ahead ([`Self::ahead`]): a read one waits for its batch.
        self.hot.retain(|&i| batch[i as usize].is_none());
        self.batch = batch;
        self.ram = ram;
        Some(self)
    }

    /// Entry `i`, read from the checkpoint, is read into RAM slot `slot` instead of a place
    /// ([`super::host_tier::PassRoute`]): its copies read it there, after its batch as before.
    /// Nothing without the tier's reads, or for an entry not read.
    pub fn to_slot(&mut self, i: usize, slot: usize) {
        if !self.read(i) {
            return;
        }
        let e = &mut self.entries[i];
        for (k, kind) in self.kinds.iter().enumerate() {
            e.src[k] = self.ram[k] + slot * kind.bytes;
        }
    }

    /// Whether the entry's source is read from the checkpoint first ([`Self::with_reads`]).
    fn read(&self, i: usize) -> bool {
        self.batch.get(i).is_some_and(Option::is_some)
    }

    /// Staged experts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// One expert's bytes over the three kinds.
    pub fn entry_bytes(&self) -> usize {
        self.kinds.iter().map(|k| k.bytes).sum()
    }

    /// The entries the router's choices `ids` (expert ids; others ignored) name.
    pub fn routed(&self, ids: &[i32]) -> Vec<bool> {
        let mut want = vec![false; self.entries.len()];
        for &e in ids {
            if let Some(&i) = usize::try_from(e).ok().and_then(|e| self.of.get(e))
                && i != u32::MAX
            {
                want[i as usize] = true;
            }
        }
        want
    }

    /// The hottest entries whose bytes fit in `budget` (with the tier's reads, of those in
    /// RAM slots).
    pub fn ahead(&self, budget: usize) -> Vec<bool> {
        let mut want = vec![false; self.entries.len()];
        let n = (budget / self.entry_bytes().max(1)).min(self.hot.len());
        for &i in &self.hot[..n] {
            want[i as usize] = true;
        }
        want
    }

    /// The copies of the entries in `want` and not in `done` (both by entry; shorter: false),
    /// kind by kind, neighbours in the staged tensors and in host memory joined up to
    /// [`PIECE`]. With `pads`, then the fixed pads, a zeroed pad at every guarded id not
    /// copied, and one at the id after every copied entry (`want` or `done`) whose next id is
    /// neither copied nor the end. Entries read from the checkpoint are not among them
    /// ([`Self::read_copies`]), but their pads are.
    pub fn copies(&self, want: &[bool], done: &[bool], pads: bool) -> Vec<Copy> {
        let at = |v: &[bool], i: usize| v.get(i).copied().unwrap_or(false);
        let mut out: Vec<Copy> = Vec::new();
        self.kind_copies(&mut out, |i| at(want, i) && !at(done, i) && !self.read(i));
        if !pads {
            return out;
        }
        out.extend_from_slice(&self.fixed);
        let copied = |i: usize| at(want, i) || at(done, i);
        for kind in self.kinds.iter().filter(|k| k.pad > 0) {
            for &g in &self.guard {
                let hit = self.entries.binary_search_by_key(&g, |e| e.id);
                if !hit.is_ok_and(copied) {
                    out.push((kind.staged + g * kind.bytes, 0, kind.pad));
                }
            }
            for (i, e) in self.entries.iter().enumerate() {
                let next = e.id + 1;
                if !copied(i) || next == self.end {
                    continue;
                }
                let follows = self
                    .entries
                    .get(i + 1)
                    .is_some_and(|n| n.id == next && copied(i + 1));
                if !follows {
                    out.push((kind.staged + next * kind.bytes, 0, kind.pad));
                }
            }
        }
        out
    }

    /// The copies of the entries read from the checkpoint ([`Self::with_reads`]) in `want` and
    /// not in `done`, batch by batch: `call(wait)` before a batch's copies and `call(release)`
    /// after them, for every batch of the row (one with nothing routed gives its places back
    /// too). `call(batch, release)` makes the marker. Empty without the tier's reads.
    pub fn read_copies(
        &self,
        want: &[bool],
        done: &[bool],
        call: impl Fn(usize, bool) -> Copy,
    ) -> Vec<Copy> {
        let at = |v: &[bool], i: usize| v.get(i).copied().unwrap_or(false);
        let mut batches: Vec<usize> = self.batch.iter().flatten().copied().collect();
        batches.sort_unstable();
        batches.dedup();
        let mut out = Vec::new();
        for b in batches {
            out.push(call(b, false));
            let of = |i: usize| self.batch.get(i).copied().flatten() == Some(b);
            self.kind_copies(&mut out, |i| of(i) && at(want, i) && !at(done, i));
            out.push(call(b, true));
        }
        out
    }

    /// Kind by kind, the copies of the entries `pick` takes, neighbours in the staged tensors
    /// and in host memory joined up to [`PIECE`].
    fn kind_copies(&self, out: &mut Vec<Copy>, pick: impl Fn(usize) -> bool) {
        for (k, kind) in self.kinds.iter().enumerate() {
            let b = kind.bytes;
            let start = out.len();
            for (i, e) in self.entries.iter().enumerate() {
                if !pick(i) {
                    continue;
                }
                let (dst, src) = (kind.staged + e.id * b, e.src[k]);
                match out[start..].last_mut() {
                    Some(c) if c.0 + c.2 == dst && c.1 + c.2 == src && c.2 + b <= PIECE => {
                        c.2 += b;
                    }
                    _ => out.push((dst, src, b)),
                }
            }
        }
    }
}

/// Bytes the copies move over the link (zeroed pads not counted).
pub fn copied_bytes(copies: &[Copy]) -> usize {
    copies.iter().filter(|c| c.1 != 0).map(|c| c.2).sum()
}

/// `LEGION_DEBUG_ROUTE_GATE`: unset, the largest gated step is [`GATE_UPTO`] tokens; `0` turns
/// the gate off (every uncached expert copied, as before); another number sets that size.
pub fn gate_upto() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_env(std::env::var("LEGION_DEBUG_ROUTE_GATE").ok(), GATE_UPTO))
}

/// `LEGION_DEBUG_ROUTE_GATE_AHEAD`: KiB copied ahead into the next layer per token of the step
/// (unset: [`AHEAD_PER_TOKEN`]; 0: none).
pub fn ahead_per_token() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let kib = parse_env(
            std::env::var("LEGION_DEBUG_ROUTE_GATE_AHEAD").ok(),
            AHEAD_PER_TOKEN >> 10,
        );
        kib << 10
    })
}

/// Default bytes the bounded host tier's reader reads ahead per token of the step, of a layer
/// whose router has not run yet: its hottest experts in the checkpoint, while it would
/// otherwise wait (the router's choices drop the reads not routed that have not started).
pub const READ_AHEAD_PER_TOKEN: usize = 256 << 10;

/// With the bounded host tier, a pass is gated only when it would read at least this many bytes
/// per token from the checkpoint: below that the SSD does not hold the pass up (an RTX 4090 with
/// 64 GB of RAM: 2.5 MiB a token at 8K, a tie gated; 5 MiB at 4K, 28% faster gated).
pub const TIER_READ_PER_TOKEN: u64 = 4 << 20;

/// `LEGION_DEBUG_ROUTE_GATE_TIER`: with the bounded host tier, the largest gated step (unset:
/// as [`gate_upto`]; `0`: the tier's passes read and copy every expert, as before).
pub fn tier_gate_upto() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        parse_env(
            std::env::var("LEGION_DEBUG_ROUTE_GATE_TIER").ok(),
            gate_upto(),
        )
    })
}

/// `LEGION_DEBUG_ROUTE_GATE_READ_AHEAD`: KiB per token of the step the bounded host tier's
/// reader reads ahead of a layer's router (unset: [`READ_AHEAD_PER_TOKEN`]; 0: none).
pub fn read_ahead_per_token() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let kib = parse_env(
            std::env::var("LEGION_DEBUG_ROUTE_GATE_READ_AHEAD").ok(),
            READ_AHEAD_PER_TOKEN >> 10,
        );
        kib << 10
    })
}

/// A debug variable's number, `default` when unset or not a number.
fn parse_env(v: Option<String>, default: usize) -> usize {
    v.and_then(|s| s.trim().parse().ok()).unwrap_or(default)
}
