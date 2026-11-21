//! Reads into the RAM slots a restore left holding copies of VRAM experts (#37 round 43).
//!
//! A lend drops the VRAM cache's tail: those experts are in the checkpoint only. The restore
//! refills the tail slots from the RAM slots' hottest experts, copies over the link that
//! leave each RAM slot holding its expert too ([`HostTier::pin_fills`]). So right after a lend
//! the two tiers hold as many fewer distinct experts as the tail had slots (on a 32 GB 4090
//! about 160 a layer, not 214), and decode reads from the checkpoint what the RAM slots held
//! before, until its misses have filled them again (long1k decode 18.8 tok/s after the lend,
//! 20.1 without).
//!
//! Once the refill's copies are done, [`HostTier::refill_dups`] reads the layer's hottest
//! experts that are in neither tier (what the prompt and the decode before it routed to most)
//! into those RAM slots, as [`Class::Extra`] reads: the readers take them only when no step
//! waits for a read. A slot stays empty and pinned until its read lands; a step that wants the
//! expert meanwhile waits for that read (promoted to a demand read), not for a second one, and
//! a staged pass drops the reads not started. The same bytes land wherever an expert is read
//! from, so the outputs do not change. Only where the RAM slots hold at least half of each
//! layer's experts ([`dup_refill_ram`]); `LEGION_DEBUG_DUP_REFILL=0` leaves the copies as before.
//!
//! The same reads carry a VRAM swap's new expert that no RAM slot holds (#37 round 45,
//! [`HostTier::read_for_swap`]). Such a swap used to read it into a staging place and wait for
//! the read and the copy in the after-step, one swap after the other: right after a lend on a
//! 32 GB 4090 that was 30-90 ms in each of the first decode steps (the lend drops the VRAM
//! tail, whose experts no RAM slot holds). Now the read runs in the background while the next
//! step computes, into a RAM slot, and the swap's copies start at the start of the next
//! after-step, from that slot (waiting for the read only if it has not landed). Those copies
//! land on their own, as any copy from a RAM slot: the slot fills an after-step later than it
//! did, and the steps in between leave it out of the tables, as any slot whose copy is under way.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{HostTier, Inner, State, To, raw_pieces, read_piece};
use crate::residency::ssd_experts::cache::SlotCache;
use crate::residency::ssd_experts::io::{Class, Group, Job};

/// A read under way into an empty RAM slot, pinned until it lands ([`Inner::land_refills`]).
pub(super) struct Refill {
    pub expert: usize,
    pub slot: usize,
    pub group: Arc<Group>,
    /// A VRAM swap's read ([`HostTier::read_for_swap`]), not a refill of a copy's slot.
    pub swap: bool,
}

/// `LEGION_DEBUG_SWAP_OVERLAP`: `0` reads a VRAM swap's new expert that no RAM slot holds in
/// the after-step, waiting for it, as before round 45; `1` reads it in the background as by
/// default but waits for the swap's copies at the next after-step (the slot fills there, as
/// before); anything else (the default): [`HostTier::read_for_swap`], and the copies land on
/// their own, as any copy from a RAM slot (the slot fills an after-step later).
pub fn swap_overlap() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(
        || match std::env::var("LEGION_DEBUG_SWAP_OVERLAP").as_deref() {
            Ok("0") => 0,
            Ok("1") => 1,
            _ => 2,
        },
    )
}

/// `LEGION_DEBUG_DUP_REFILL=0`: the RAM slots a restore's fill left holding copies of VRAM
/// experts keep them until decode misses evict them, as before round 43.
pub fn dup_refill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LEGION_DEBUG_DUP_REFILL").is_none_or(|v| v != "0"))
}

/// Whether a tier with `slots` RAM slots per layer of `experts` refills them after a lend: where
/// they hold at least half of each layer's experts. With fewer, decode after a lend waits on
/// checkpoint reads most steps, and a refill read a reader has started holds up the next step's
/// own (#37 round 43 on an RTX 4090: 111 of 512 slots a layer, long1k decode a tie, the chats
/// after it 0.1-0.5 tok/s slower; 320 of 512, long1k decode 34.9/35.2 -> 35.7/35.9 tok/s).
pub(crate) fn dup_refill_ram(slots: usize, experts: usize) -> bool {
    2 * slots >= experts
}

/// Which experts go into which of `cache`'s slots: `ranked` (hottest first) that no VRAM slot
/// holds (`in_vram`), no RAM slot holds and no read is under way for (`flying`), each into the
/// next slot whose expert a VRAM slot holds too and no read or copy pins, coldest first
/// ([`SlotCache::evict_order`]). `(expert, slot)`.
pub(crate) fn plan_dup_refill(
    cache: &SlotCache,
    in_vram: &[bool],
    flying: &[bool],
    ranked: &[u32],
) -> Vec<(usize, usize)> {
    let mut dup = vec![false; cache.n_slots()];
    for (e, _) in in_vram.iter().enumerate().filter(|&(_, &v)| v) {
        if let Some(s) = cache.slot(e) {
            dup[s] = true;
        }
    }
    let victims = cache.evict_order().into_iter().filter(|&s| dup[s]);
    let incoming = ranked.iter().map(|&e| e as usize).filter(|&e| {
        !in_vram.get(e).copied().unwrap_or(true)
            && !flying.get(e).copied().unwrap_or(true)
            && cache.slot(e).is_none()
    });
    incoming.zip(victims).collect()
}

/// Where [`HostTier::read_for_swap`] reads a swap's `new` expert into `cache` (`flying`: a read
/// of it is under way already): `None`, nowhere (a slot holds it, or every slot is pinned or
/// the step's: the swap copies now); `Some(None)`, the read under way; `Some(Some(s))`, into
/// slot `s`, taken empty and pinned ([`SlotCache::reserve`]).
pub(crate) fn plan_swap_read(
    cache: &mut SlotCache,
    new: usize,
    flying: bool,
) -> Option<Option<usize>> {
    if cache.slot(new).is_some() {
        return None;
    }
    if flying {
        return Some(None);
    }
    cache.reserve().map(Some)
}

impl Inner {
    /// Lands layer `il`'s refills that are done, after waiting for those of the experts `need`
    /// (moved ahead of every other queued read first): each expert goes into its RAM slot
    /// unless a VRAM slot or another RAM slot took it meanwhile, or the read was dropped (the
    /// slot is empty again). The first read error, if any.
    pub(super) fn land_refills(
        &self,
        st: &mut State,
        il: usize,
        need: &[usize],
    ) -> Result<(), String> {
        if st.refills[il].is_empty() {
            return Ok(());
        }
        for r in st.refills[il].iter().filter(|r| need.contains(&r.expert)) {
            if !r.group.done() {
                self.pool.promote(&r.group);
            }
        }
        for r in st.refills[il].iter().filter(|r| need.contains(&r.expert)) {
            r.group.wait_landed();
        }
        let mut err = None;
        for r in std::mem::take(&mut st.refills[il]) {
            if !r.group.done() {
                st.refills[il].push(r);
                continue;
            }
            let read = match r.group.wait() {
                Ok(()) => !r.group.skipped(),
                Err(e) => {
                    err.get_or_insert(e);
                    false
                }
            };
            let cache = &mut st.caches[il];
            if read && cache.slot(r.expert).is_none() && !st.in_vram[il][r.expert] {
                cache.put(r.expert, r.slot);
                if !r.swap {
                    self.counts.dup_refills.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                cache.drop_slot(r.slot);
            }
        }
        err.map_or(Ok(()), Err)
    }

    /// Drops every refill read not started, waits for the others and lands them.
    pub(super) fn cancel_refills(&self, st: &mut State) -> Result<(), String> {
        for r in st.refills.iter().flatten() {
            r.group.cancel();
        }
        let mut err = Ok(());
        for il in 0..st.refills.len() {
            for r in &st.refills[il] {
                r.group.wait_landed();
            }
            let r = self.land_refills(st, il, &[]);
            if err.is_ok() {
                err = r;
            }
        }
        err
    }
}

impl HostTier {
    /// Whether a restore's RAM fill is followed by [`Self::refill_dups`]: [`dup_refill`] and
    /// [`dup_refill_ram`].
    pub fn refills_dups(&self) -> bool {
        dup_refill() && dup_refill_ram(self.plan.slots, self.inner.n_expert)
    }

    /// After a restore's RAM fill is done (its copies landed, the tables point at the filled
    /// VRAM slots): starts reads of layer `il`'s `ranked` experts (hottest first) that are in
    /// neither tier into its RAM slots that hold copies of VRAM experts ([`plan_dup_refill`]),
    /// behind every other read. Returns how many it started.
    pub fn refill_dups(&self, il: usize, ranked: &[u32]) -> Result<usize, String> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let st = &mut *st;
        inner.take_ahead(st, None)?;
        inner.land_refills(st, il, &[])?;
        let mut flying = vec![false; inner.n_expert];
        for r in &st.refills[il] {
            flying[r.expert] = true;
        }
        let pairs = plan_dup_refill(&st.caches[il], &st.in_vram[il], &flying, ranked);
        let piece = read_piece();
        for &(e, s) in &pairs {
            let cache = &mut st.caches[il];
            cache.drop_slot(s);
            cache.pin(s, true);
            let jobs: Vec<Job> = inner
                .reads(il, e, To::Slot(s))
                .into_iter()
                .flat_map(|j| match j {
                    Job::Raw { src, dst } => raw_pieces(src, dst, piece),
                    j => vec![j],
                })
                .collect();
            st.refills[il].push(Refill {
                expert: e,
                slot: s,
                group: inner.pool.submit_as(jobs, Class::Extra),
                swap: false,
            });
        }
        Ok(pairs.len())
    }

    /// A VRAM swap of layer `il` that brings in `new` and drops `old` (#37 round 45): when no
    /// RAM slot holds `new`, starts its read from the checkpoint into a RAM slot (an empty one,
    /// else the coldest one no step or copy pins, [`SlotCache::reserve`]) behind every step's
    /// own read, and returns `true`: the caller starts the swap's copies at the next
    /// after-step, where [`Self::swap`] takes `new` from that slot (a step that wants `new`
    /// meanwhile waits for this read instead of reading it again; a staged pass drops it if it
    /// has not started, and then the swap reads `new` as before). `false`: `new` is in a RAM
    /// slot, or every slot is pinned: swap now. Where the experts come from does not change
    /// what the steps compute: the same bytes land in the same VRAM slot.
    pub fn read_for_swap(&self, il: usize, new: u32, old: Option<u32>) -> Result<bool, String> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let st = &mut *st;
        inner.take_ahead(st, None)?;
        inner.land_refills(st, il, &[])?;
        let new = new as usize;
        let flying = st.refills[il].iter().any(|r| r.expert == new);
        let Some(to) = plan_swap_read(&mut st.caches[il], new, flying) else {
            return Ok(false);
        };
        if let Some(s) = to {
            let piece = read_piece();
            let jobs: Vec<Job> = inner
                .reads(il, new, To::Slot(s))
                .into_iter()
                .flat_map(|j| match j {
                    Job::Raw { src, dst } => raw_pieces(src, dst, piece),
                    j => vec![j],
                })
                .collect();
            st.refills[il].push(Refill {
                expert: new,
                slot: s,
                group: inner.pool.submit_as(jobs, Class::Ahead),
                swap: true,
            });
            inner.counts.swap_reads.fetch_add(1, Ordering::Relaxed);
        }
        // As [`Self::swap`] has it: the dropped expert is no VRAM slot's from now on (its slot
        // is out of the tables). `new` counts as in VRAM only once its copy is queued, so the
        // read above lands in its RAM slot.
        if let Some(o) = old {
            st.in_vram[il][o as usize] = false;
        }
        Ok(true)
    }
}
