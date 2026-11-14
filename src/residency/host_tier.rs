//! The bounded host expert tier (#37): routed experts of a plain GGUF on a discrete GPU whose
//! machine has less RAM than the experts take (a gaming PC).
//!
//! Without it every expert not in a VRAM slot sits in pinned host memory
//! ([`super::experts::ExpertResidency`]: `[VRAM slots | all the experts]`), which a 32 or 64 GB
//! machine cannot hold. With it a layer's tiered tensor is
//!
//! ```text
//! [ S VRAM slots | R pinned RAM slots | gap | staging: two sets of places ]
//! ```
//!
//! - The **RAM slots** hold the experts used last (least recently used out first,
//!   [`SlotCache`]); `R` comes from the memory the machine has ([`size`]). An expert a VRAM
//!   slot drops goes to a RAM slot, and the RAM copy of one a VRAM slot takes is dropped
//!   ([`HostTier::swap`]): the two tiers hold different experts.
//! - The **staging** part is one buffer that every layer maps: a step that routes to more
//!   experts than the RAM slots hold (a prompt chunk) reads the rest into its places
//!   ([`stage_sets`]). The layers take its two sets in turn, so a big step reads the next
//!   layer's experts (those its last big step read) while this layer computes.
//! - Experts in neither are read from the GGUF, around the page cache (`O_DIRECT` through the
//!   SSD tier's readers, [`super::ssd_experts::io`]), as the step routes to them.
//!
//! The graph keeps the VRAM cache's lookup (`slot`, or `S + expert` for a miss) and its route
//! log; a fetch op ([`FETCH`]) after it turns every miss into the index of the RAM slot or
//! staging place that holds the expert, reading it first where needed. On CUDA the op copies
//! the ids to pinned memory, runs as a host function on the backend's stream (so the layer
//! before is done with the staging buffer and the matmuls after it see the bytes) and copies
//! the new ids back.
//!
//! The VRAM cache's own copies (its first fill and its swaps) take their bytes from a RAM slot
//! or, read first, from the expert's staging place ([`HostTier::sources`]).
//!
//! A long prompt lends the VRAM cache as without the tier; its big steps skip the fetch and
//! stage every layer's experts in VRAM from the RAM slots and, through the staging places,
//! from the checkpoint, read layers ahead ([`passes`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use crate::ggml::{BufferType, Context, ExternalOp, Tensor, external_params, ffi, types};
use crate::gguf::{FileRange, GgufFile};

use super::lookup_table::MemInfo;
use super::ssd_experts::KINDS;
use super::ssd_experts::cache::SlotCache;
use super::ssd_experts::io::{Group, Job, Pool};

mod passes;
pub use passes::{PassHook, PassSrc, release_arg, wait_arg};

#[cfg(test)]
mod tests;

/// Experts' worth of room between the RAM slots and the staging part: the RAM slots' arena
/// piece ends on a mapping granule (2 MiB) past its tensor's padded end, and the staging
/// places start at the next whole expert after that.
pub const GAP: usize = 4;

/// Reader threads (an NVMe drive's queue; a decode step misses a few experts per layer).
const READERS: usize = 8;

/// Fewest RAM slots per layer worth running with: a full decode batch's experts of a few
/// sequences.
pub const MIN_SLOTS: usize = 32;

/// Staging places kept beyond the experts the RAM slots cannot hold: RAM slots that VRAM
/// swaps under way keep from a step.
const SPARE_PLACES: usize = 16;

/// Experts of a layer its VRAM slots hold at least, where the staging has two sets: a set
/// need not hold those. A layer with fewer runs its big steps without reads ahead.
const AHEAD_IN_VRAM: usize = 64;

/// A step of at least this many routed (token, expert) pairs per expert of a layer is a big
/// one (a prompt chunk): it routes to about every expert of every layer, and the next
/// layer's are read ahead.
const AHEAD_PAIRS_PER_EXPERT: usize = 8;

/// The staging part for `slots` RAM slots: `(places per set, sets)`. A step needs at most a
/// place for every expert the RAM slots cannot hold. Two sets, which the layers take in turn
/// so that a big step reads the next layer's experts while this one computes: each as big as
/// a big step's layer needs with its VRAM slots, together at least what any step needs and
/// at most a place per expert (a layer that needs more than its set takes the other too, and
/// the next layer is not read ahead).
pub(crate) fn stage_sets(n_expert: usize, slots: usize) -> (usize, usize) {
    let need = (n_expert.saturating_sub(slots) + SPARE_PLACES).min(n_expert);
    let places = need
        .saturating_sub(AHEAD_IN_VRAM)
        .min(n_expert / 2)
        .max(need.div_ceil(2))
        .max(1);
    (places, 2)
}

/// Fetches between two reports.
const REPORT_EVERY: u64 = 48 * 256;

/// Host memory kept free beside the tier: at least this much ...
const MARGIN_MIN: u64 = 8 << 30;
/// ... or a tenth of the machine's, if more.
const MARGIN_SHARE: u64 = 10;
/// A machine too small for [`MARGIN_MIN`] to be a sixth of its memory (under 48 GiB) keeps
/// that sixth free instead, and never less than this.
const MARGIN_SMALL_SHARE: u64 = 6;
const MARGIN_FLOOR: u64 = 4 << 30;

/// The host memory kept free beside the tier on a machine with `total` bytes: what is
/// available when the tier is sized already leaves out what the desktop holds, so this is
/// only room to grow into, and the tier's memory is pinned and cannot give way. A 32 GiB
/// machine keeps 5.3 GiB (2.7 GiB more for RAM slots than with 8: about 19 slots a layer of
/// Qwen3.8-Flash-Next, a quarter more), a 24 GiB one 4 GiB, from 48 GiB on it is 8 GiB as
/// before and a tenth of the memory past 80 GiB.
pub(crate) fn margin(total: u64) -> u64 {
    (total / MARGIN_SMALL_SHARE)
        .clamp(MARGIN_FLOOR, MARGIN_MIN)
        .max(total / MARGIN_SHARE)
}

/// The direct n-gram tier's row cache at most ([`super::lookup_table`]'s default cap).
const TABLE_CACHE: u64 = 4 << 30;

/// The n-gram table beside the experts, for [`size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Table {
    pub bytes: u64,
    /// `--ngram-io ram` or `mmap`: the table is in RAM whole whatever the experts get.
    pub in_ram: bool,
    /// `--ngram-io direct`: never in RAM whole.
    pub direct: bool,
}

/// What [`size`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sizing {
    /// Every expert fits in host memory: no tier, the load is as without this module.
    Whole,
    /// The tier with this many RAM slots per layer.
    Bounded(Plan),
    /// Not even [`MIN_SLOTS`] per layer fit.
    TooSmall(Plan),
}

/// The tier's memory, as planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// RAM slots per layer.
    pub slots: usize,
    /// Staging places per set, and the sets ([`stage_sets`]).
    pub places: usize,
    pub sets: usize,
    /// Their bytes over all layers.
    pub slot_bytes: u64,
    /// The staging buffer's bytes.
    pub stage_bytes: u64,
    /// What was kept free: the margin, the n-gram table's share and the KV spill.
    pub margin: u64,
    pub table: u64,
    pub kv_spill: u64,
    /// The memory the plan was made against.
    pub available: u64,
}

/// The pinned host memory the routed experts get on a machine with `mem`: what is available
/// less a margin ([`margin`]), the n-gram table's share and
/// `kv_spill` (host memory the KV cache may take). `per_slot`: one expert of every kind of
/// every layer (the layers' types may differ), the experts whole taking `n_expert` of that;
/// `stage_one`: one expert of every staging buffer.
///
/// The table and the experts are decided together: when both fit whole, or the experts fit
/// beside the table's direct tier (its row cache), nothing changes ([`Sizing::Whole`]: the
/// table's own `auto` then picks ram or direct as before). Otherwise the table reads direct and
/// the experts get RAM slots from what is left.
pub fn size(
    mem: MemInfo,
    per_slot: u64,
    stage_one: u64,
    n_expert: usize,
    table: Option<Table>,
    kv_spill: u64,
) -> Sizing {
    let margin = margin(mem.total);
    let room = mem.available.saturating_sub(margin + kv_spill);
    let all = per_slot * n_expert as u64;
    // What the table takes at least: all of it when asked for in RAM, else its row cache.
    let table_min = table.map_or(0, |t| {
        if t.in_ram {
            t.bytes
        } else {
            TABLE_CACHE.min(t.bytes)
        }
    });
    if room >= all + table_min {
        return Sizing::Whole;
    }
    // The most slots that fit with the staging they leave to do ([`stage_sets`]).
    let left = room.saturating_sub(table_min);
    let stage = |slots: usize| {
        let (places, sets) = stage_sets(n_expert, slots);
        stage_one * (GAP + places * sets + 1) as u64
    };
    let mut slots = ((left / per_slot.max(1)) as usize).min(n_expert);
    while slots > 0 && slots as u64 * per_slot + stage(slots) > left {
        slots -= 1;
    }
    let stage_bytes = stage(slots);
    let (places, sets) = stage_sets(n_expert, slots);
    let plan = Plan {
        slots,
        places,
        sets,
        slot_bytes: slots as u64 * per_slot,
        stage_bytes,
        margin,
        table: table_min,
        kv_spill,
        available: mem.available,
    };
    if slots < MIN_SLOTS {
        Sizing::TooSmall(plan)
    } else {
        Sizing::Bounded(plan)
    }
}

/// A debug cap for emulating a smaller machine (`LEGION_DEBUG_HOST_MEM_GIB`, see
/// `docs/debug-env.md`): the tier is sized as on a machine with that much memory, all of it
/// available, or as `total:available` says (a 32 GB machine has some 27 GiB available: "32:27"
/// gives it its own margin, "27" the smaller one of a 27 GiB machine). Not a tuning knob: real
/// machines are sized from `/proc/meminfo`.
pub fn mem_info() -> MemInfo {
    let real = MemInfo::read();
    match std::env::var("LEGION_DEBUG_HOST_MEM_GIB")
        .ok()
        .and_then(|v| debug_mem(&v))
    {
        Some((total, available)) => MemInfo {
            total,
            available: available.min(real.available.max(1)),
        },
        None => real,
    }
}

/// [`mem_info`]'s debug value as `(total, available)` bytes: `"<GiB>"` (both) or
/// `"<total GiB>:<available GiB>"`; `None` when it is neither (or not positive, or more is
/// available than there is).
pub(crate) fn debug_mem(v: &str) -> Option<(u64, u64)> {
    let gib = |s: &str| {
        s.trim()
            .parse::<f64>()
            .ok()
            .filter(|g| *g > 0.0 && g.is_finite())
            .map(|g| (g * (1u64 << 30) as f64) as u64)
    };
    match v.split_once(':') {
        Some((t, a)) => gib(t).zip(gib(a)).filter(|(t, a)| a <= t),
        None => gib(v).map(|b| (b, b)),
    }
}

/// The routed expert tensors of layers `0..n_layer`, when every one is a plain range of the
/// GGUF with `n_expert` experts last (no per-expert scales): `[up, gate, down]` per layer.
pub fn sources(
    gguf: &GgufFile,
    n_layer: usize,
    n_expert: usize,
) -> Option<Vec<[crate::gguf::TensorInfo; 3]>> {
    (0..n_layer)
        .map(|il| {
            let one = |k: &str| {
                let name = format!("blk.{il}.{k}.weight");
                let scale = format!("blk.{il}.{k}.scale");
                let t = gguf.tensor(&name)?;
                (gguf.recipe(&name).is_none()
                    && gguf.tensor(&scale).is_none()
                    && t.shape.len() == 3
                    && t.shape[2] == n_expert as u64
                    && t.size.is_multiple_of(n_expert as u64))
                .then_some(t)
            };
            Some([one(KINDS[0])?, one(KINDS[1])?, one(KINDS[2])?])
        })
        .collect()
}

/// The names [`sources`]' tensors have: left out of the weights' load.
pub fn skipped_tensors(n_layer: usize) -> Vec<String> {
    (0..n_layer)
        .flat_map(|il| KINDS.map(|k| format!("blk.{il}.{k}.weight")))
        .collect()
}

/// The tier's counters since the load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostStats {
    /// Routed (token, expert) pairs by where the step found the expert.
    pub vram_pairs: u64,
    pub ram_pairs: u64,
    pub ssd_pairs: u64,
    /// Experts read into a RAM slot, and into the staging buffer.
    pub ram_loads: u64,
    pub stage_loads: u64,
    /// Experts big steps read into the staging buffer a layer ahead.
    pub ahead_loads: u64,
    /// Experts a VRAM slot dropped that went to a RAM slot.
    pub write_backs: u64,
    /// Bytes read from the GGUF (steps and the VRAM cache's copies).
    pub read_bytes: u64,
    /// Time steps waited in the fetch.
    pub fetch_ns: u64,
    pub fetches: u64,
}

#[derive(Default)]
struct Counters {
    vram_pairs: AtomicU64,
    ram_pairs: AtomicU64,
    ssd_pairs: AtomicU64,
    ram_loads: AtomicU64,
    stage_loads: AtomicU64,
    ahead_loads: AtomicU64,
    write_backs: AtomicU64,
    fetch_ns: AtomicU64,
    fetches: AtomicU64,
}

struct LayerMem {
    /// The layer's expert tensors in the GGUF, whole.
    src: [FileRange; 3],
    /// Host addresses of the RAM slots.
    ram: [usize; 3],
    /// One expert's bytes, per kind.
    bytes: [usize; 3],
    /// Host address of expert 0's staging place, per kind.
    stage: [usize; 3],
    /// The layer's staging tensors: [`classes`]' index per kind.
    class: [usize; 3],
}

struct Inner {
    n_expert: usize,
    /// RAM slots per layer.
    slots: usize,
    /// Staging places per set, and the sets ([`stage_sets`]): layer `il`'s steps take set
    /// `il % sets` first.
    places: usize,
    sets: usize,
    layers: Vec<LayerMem>,
    /// Pinned ids in and out of the fetch (CUDA), `cap` ids each.
    xchg: Option<(usize, usize)>,
    cap: usize,
    pool: Pool,
    state: Mutex<State>,
    /// The staged prompt pass whose reads are under way ([`HostTier::stage_pass`]).
    pass: Mutex<Option<passes::Running>>,
    counts: Counters,
}

/// The RAM slots of every layer, and the VRAM swaps whose copies are under way.
struct State {
    caches: Vec<SlotCache>,
    /// Batches of swaps ([`HostTier::swap`], closed by [`HostTier::swaps_queued`]), oldest
    /// first: one per [`HostTier::swaps_landed`].
    flights: VecDeque<Vec<Flight>>,
    /// The batch being queued.
    open: Vec<Flight>,
    /// Per layer, the experts the VRAM cache took ([`HostTier::sources`], [`HostTier::swap`])
    /// and has not dropped: a read ahead leaves them out.
    in_vram: Vec<Vec<bool>>,
    /// Per layer, the experts its last big step read from the checkpoint: what the next big
    /// step's read ahead takes (prompts route alike from chunk to chunk).
    last_big: Vec<Vec<usize>>,
    /// The read ahead under way, if any.
    ahead: Option<Ahead>,
}

/// Experts of a layer being read into staging places ahead of the big step that is one layer
/// before it.
struct Ahead {
    layer: usize,
    group: Arc<Group>,
    /// `(expert, place)`.
    at: Vec<(usize, usize)>,
}

/// The experts a big step's read ahead takes for a layer: those its last big step read
/// (`last`) that are still in no VRAM slot (`in_vram`) and no RAM slot of `cache`, as many as
/// a set has `places`.
pub(crate) fn plan_ahead(
    cache: &SlotCache,
    in_vram: &[bool],
    last: &[usize],
    places: usize,
) -> Vec<usize> {
    last.iter()
        .copied()
        .filter(|&e| !in_vram[e] && cache.slot(e).is_none())
        .take(places)
        .collect()
}

/// Gives the experts a step stages their places: `taken` places of the step's own set (`set`
/// of `sets`, `places` each) hold experts read ahead; the others follow there, then in the
/// other sets from their first place. `None`: more experts than places. Returns the places
/// and whether another set was used (its places then hold this step's experts: no read ahead
/// into it).
pub(crate) fn assign_places(
    n: usize,
    set: usize,
    sets: usize,
    places: usize,
    taken: usize,
) -> Option<(Vec<usize>, bool)> {
    let own = places.saturating_sub(taken);
    if n > own + (sets - 1) * places {
        return None;
    }
    let at = (0..n)
        .map(|i| {
            if i < own {
                set * places + taken + i
            } else {
                let j = i - own;
                ((set + 1 + j / places) % sets) * places + j % places
            }
        })
        .collect();
    Some((at, n > own))
}

/// One VRAM swap's RAM slots while its copies are under way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Flight {
    pub layer: usize,
    /// The RAM slot the new expert's bytes are copied from (pinned; emptied when it lands:
    /// the expert is in VRAM then).
    pub from: Option<usize>,
    /// The expert the VRAM slot dropped and the RAM slot its bytes are copied to (reserved;
    /// the expert's when it lands).
    pub back: Option<(usize, usize)>,
}

/// Where a VRAM swap takes and leaves its experts ([`HostTier::swap`]), as indices past the
/// layer's VRAM slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapAt {
    /// The new expert's bytes: its RAM slot, or its staging place (read first).
    pub src: usize,
    /// The RAM slot the dropped expert's bytes go to, before the slot is overwritten.
    pub back: Option<usize>,
    /// `src` is a staging place: its bytes only stay until the next step or swap, so the
    /// copy must be done by then.
    pub staged: bool,
}

/// Plans one VRAM swap in a layer's RAM slots `cache`: `new` comes from its RAM slot (pinned
/// until the swap lands), `old` (what the VRAM slot held) goes to a reserved RAM slot unless
/// a RAM slot holds it already. The swap's [`Flight`] of layer `il`.
pub(crate) fn plan_swap(
    cache: &mut SlotCache,
    il: usize,
    new: usize,
    old: Option<usize>,
) -> Flight {
    let from = cache.slot(new);
    if let Some(r) = from {
        cache.pin(r, true);
    }
    let back = old
        .filter(|&o| cache.slot(o).is_none())
        .and_then(|o| cache.reserve().map(|s| (o, s)));
    Flight {
        layer: il,
        from,
        back,
    }
}

/// A swap's copies are done: the new expert leaves its RAM slot (it is in VRAM), the dropped
/// one takes the slot its bytes went to (unless a step read it into another meanwhile).
pub(crate) fn land_swap(cache: &mut SlotCache, f: &Flight) {
    if let Some(r) = f.from {
        cache.drop_slot(r);
    }
    if let Some((o, d)) = f.back {
        if cache.slot(o).is_some() {
            cache.drop_slot(d);
        } else {
            cache.put(o, d);
        }
    }
}

// SAFETY: the addresses point into host buffers the owning `HostTier` keeps alive; every write
// through them is ordered by the state lock and the backend's stream.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

static REGISTRY: Mutex<Vec<Option<Arc<Inner>>>> = Mutex::new(Vec::new());
/// The live tiers, for [`HostTier::of`].
static TIERS: Mutex<Vec<Weak<HostTier>>> = Mutex::new(Vec::new());

fn lookup(id: i32) -> Arc<Inner> {
    REGISTRY.lock().unwrap()[id as usize]
        .clone()
        .expect("a live host expert tier")
}

/// The tier of one model.
pub struct HostTier {
    inner: Arc<Inner>,
    id: i32,
    plan: Plan,
    /// The RAM slot tensors (named as the model's expert tensors), the staging tensors and the
    /// exchange buffers. Last: dropped after the readers.
    ctx: Context,
}

// SAFETY: the context's tensors are made once in `new` and only read after; their bytes are
// written through `Inner`'s addresses (see there).
unsafe impl Send for HostTier {}
unsafe impl Sync for HostTier {}

impl HostTier {
    /// The tier over `srcs` ([`sources`]) with `plan.slots` RAM slots per layer, its memory in
    /// `buft` (a pinned host arena the GPU maps), `max_tokens` the biggest step.
    pub fn new(
        gguf: &GgufFile,
        srcs: &[[crate::gguf::TensorInfo; 3]],
        n_expert: usize,
        n_used: usize,
        plan: Plan,
        buft: BufferType,
        max_tokens: usize,
    ) -> Result<Arc<Self>, String> {
        if srcs.is_empty() {
            return Err("no MoE layers".into());
        }
        let slots = plan.slots;
        let n_place = plan.places * plan.sets;
        if plan.places == 0 || plan.sets == 0 {
            return Err("the host expert tier needs staging places".into());
        }
        // The layers' experts differ in type (a mixed quantization): one staging tensor per
        // kind and type, shared by the layers that have it.
        let (classes, class_of) = classes(srcs);
        let ctx = Context::new(srcs.len() * 3 + classes.len() + 2, 0);
        let shape = |t: &crate::gguf::TensorInfo, n: usize| {
            [t.shape[0] as i64, t.shape[1] as i64, n as i64]
        };
        for (il, l) in srcs.iter().enumerate() {
            for (k, kind) in KINDS.iter().enumerate() {
                ctx.new_tensor(l[k].ggml_type, &shape(&l[k], slots))
                    .named(&format!("blk.{il}.{kind}.weight"));
            }
        }
        for (c, &(il, k)) in classes.iter().enumerate() {
            let t = &srcs[il][k];
            ctx.new_tensor(t.ggml_type, &shape(t, n_place + GAP + 1))
                .named(&format!("host_tier.stage.{c}"));
        }
        let cap = n_used * max_tokens.max(1);
        ctx.new_tensor(types::I32, &[cap as i64])
            .named("host_tier.ids_in");
        ctx.new_tensor(types::I32, &[cap as i64])
            .named("host_tier.ids_out");
        ctx.alloc_tensors(buft)
            .map_err(|e| format!("host expert tier: {e}"))?;
        let t = |name: &str| ctx.tensor(name).expect("tier tensor");
        let addr = |name: &str| -> usize {
            // SAFETY: an allocated tensor of this context, in host memory.
            unsafe { (*t(name).raw()).data as usize }
        };
        let mut layers = Vec::with_capacity(srcs.len());
        for (il, l) in srcs.iter().enumerate() {
            let bytes = [0, 1, 2].map(|k| (l[k].size / n_expert as u64) as usize);
            // A staging place's index past the VRAM slots is `slots + GAP + place`: its bytes
            // start that many experts after the RAM slots' start, of which the RAM tensor's own
            // pieces cover `span`.
            let mut stage = [0usize; 3];
            for (k, kind) in KINDS.iter().enumerate() {
                let ram = t(&format!("blk.{il}.{kind}.weight"));
                let name = format!("host_tier.stage.{}", class_of[il][k]);
                let (span, tail_span) = spans(ram, t(&name))?;
                let at = ((slots + GAP) * bytes[k])
                    .checked_sub(span)
                    .ok_or("the RAM slots' piece runs past the gap")?;
                // The last place's bytes and a matmul's reads past them stay inside the buffer.
                if at + (n_place + 1) * bytes[k] > tail_span {
                    return Err("the staging buffer is too short".into());
                }
                stage[k] = addr(&name) + at;
            }
            layers.push(LayerMem {
                src: [0, 1, 2].map(|k| FileRange {
                    file: l[k].file_index,
                    offset: l[k].offset,
                    len: l[k].size,
                }),
                ram: [0, 1, 2].map(|k| addr(&format!("blk.{il}.{}.weight", KINDS[k]))),
                bytes,
                stage,
                class: class_of[il],
            });
        }
        let cuda = cfg!(feature = "cuda") && !buft.is_plain_cpu();
        let xchg = cuda.then(|| (addr("host_tier.ids_in"), addr("host_tier.ids_out")));
        let files: Vec<std::path::PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
        let pool = Pool::new(&files, READERS, true)?;
        let inner = Arc::new(Inner {
            n_expert,
            slots,
            places: plan.places,
            sets: plan.sets,
            state: Mutex::new(State {
                caches: vec![SlotCache::new(n_expert, slots); layers.len()],
                flights: VecDeque::new(),
                open: Vec::new(),
                in_vram: vec![vec![false; n_expert]; layers.len()],
                last_big: vec![Vec::new(); layers.len()],
                ahead: None,
            }),
            layers,
            xchg,
            cap,
            pool,
            pass: Mutex::new(None),
            counts: Counters::default(),
        });
        let id = {
            let mut r = REGISTRY.lock().unwrap();
            r.push(Some(inner.clone()));
            (r.len() - 1) as i32
        };
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        tracing::info!(
            layers = srcs.len(),
            ram_slots_per_layer = slots,
            experts = n_expert,
            ram_gib = gib(plan.slot_bytes),
            staging_gib = gib(plan.stage_bytes),
            direct = inner.pool.direct(),
            "host expert tier: experts in no VRAM or RAM slot are read from the GGUF"
        );
        let me = Arc::new(Self {
            inner,
            id,
            plan,
            ctx,
        });
        let mut tiers = TIERS.lock().unwrap();
        tiers.retain(|w| w.strong_count() > 0);
        tiers.push(Arc::downgrade(&me));
        Ok(me)
    }

    /// The tier `host` is a RAM slot tensor of, with its layer.
    pub fn of(host: Tensor<'_>) -> Option<(Arc<Self>, usize)> {
        // SAFETY: reading the data pointer of a live tensor.
        let data = unsafe { (*host.raw()).data as usize };
        if data == 0 {
            return None;
        }
        let tiers = TIERS.lock().unwrap();
        tiers.iter().filter_map(Weak::upgrade).find_map(|t| {
            let il = t.inner.layers.iter().position(|l| l.ram.contains(&data))?;
            Some((t, il))
        })
    }

    /// The tensor `name` of the tier: a layer's RAM slots under the model's name for its
    /// experts (`blk.N.ffn_*_exps.weight`).
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        name.starts_with("blk.")
            .then(|| self.ctx.tensor(name))
            .flatten()
    }

    /// The staging tensor of layer `il`'s expert kind `k`.
    pub fn stage_tensor(&self, il: usize, k: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!(
                "host_tier.stage.{}",
                self.inner.layers[il].class[k]
            ))
            .expect("staging tensor")
    }

    /// RAM slots per layer.
    pub fn slots(&self) -> usize {
        self.inner.slots
    }

    /// The experts of a layer's tiered tensor past its VRAM slots: RAM slots, the gap and
    /// the staging places.
    pub fn n_host(&self) -> usize {
        self.inner.slots + GAP + self.places()
    }

    /// The staging places: the most experts one [`Self::sources`] call can read.
    pub fn places(&self) -> usize {
        self.inner.places * self.inner.sets
    }

    pub fn plan(&self) -> Plan {
        self.plan
    }

    /// `ids` (I32 `[n_used, n]`: a VRAM slot below `n_slots`, or `n_slots + expert`) as the
    /// indices the matmuls read, after the reads that takes.
    pub fn fetch<'a>(
        &self,
        ctx: &'a Context,
        il: usize,
        ids: Tensor<'a>,
        n_slots: usize,
    ) -> Result<Tensor<'a>, String> {
        let [n_used, n_tok, _, _] = ids.ne();
        let pairs = (n_used * n_tok) as usize;
        if self.inner.xchg.is_some() && pairs > self.inner.cap {
            return Err(format!(
                "a step of {n_tok} tokens: the host expert tier takes at most {}",
                self.inner.cap / n_used.max(1) as usize
            ));
        }
        let ids = if ids.is_contiguous() {
            ids
        } else {
            ctx.cont(ids)
        };
        Ok(ctx.external_with(
            &FETCH,
            types::I32,
            ids.ne(),
            &[ids],
            &[self.id, il as i32, n_slots as i32],
        ))
    }

    /// Where the VRAM cache copies `experts` of layer `il` from, as indices past the layer's
    /// VRAM slots: an expert's RAM slot, or its staging place, read first. The bytes stay
    /// until the next step or the next call: the copies must be done by then.
    pub fn sources(&self, il: usize, experts: &[u32]) -> Result<Vec<usize>, String> {
        self.inner.sources(il, experts)
    }

    /// A VRAM slot of layer `il` swaps `old` (what it held, if anything) for `new`: where
    /// the copies go. The caller queues, in this order, the slot's bytes to
    /// [`SwapAt::back`] and [`SwapAt::src`]'s to the slot, then closes its batch of swaps
    /// with [`Self::swaps_queued`] and tells when the batch's copies are done
    /// ([`Self::swaps_landed`], batches in order). Until then the RAM slots involved are
    /// pinned: `new` reads from its RAM slot, `old` from the checkpoint if a step wants it.
    /// So an expert a VRAM slot drops stays in RAM, and RAM holds no copy of an expert that
    /// is in VRAM: the two tiers hold different experts.
    pub fn swap(&self, il: usize, new: u32, old: Option<u32>) -> Result<SwapAt, String> {
        self.inner.swap(il, new as usize, old.map(|o| o as usize))
    }

    /// Closes the batch of [`Self::swap`]s queued since the last call.
    pub fn swaps_queued(&self) {
        self.inner.swaps_queued();
    }

    /// The routing heat of `layers`' experts (`(layer, per expert)`, the VRAM cache's own
    /// decayed counts) as of the step just learned: their RAM slots give up the coldest
    /// expert first from now on ([`SlotCache::set_heat`]), not the least recently used. An
    /// expert a VRAM slot holds counts as cold: no step reads its RAM copy (one the refill
    /// after a long prompt left), so that slot goes first.
    pub fn set_heat<'a>(&self, layers: impl Iterator<Item = (usize, &'a [f32])>) {
        let mut st = self.inner.state.lock().unwrap();
        let st = &mut *st;
        let mut cold = Vec::new();
        for (il, heat) in layers {
            cold.clear();
            cold.extend(
                heat.iter()
                    .zip(&st.in_vram[il])
                    .map(|(&h, &v)| if v { 0.0 } else { h }),
            );
            st.caches[il].set_heat(&cold);
        }
    }

    /// At load: reads `experts` of layer `il` (best first, none of them in a VRAM slot) from
    /// the checkpoint into the layer's empty RAM slots, as many as there are. Returns how many
    /// were read.
    pub fn prefill(&self, il: usize, experts: &[u32]) -> Result<usize, String> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let st = &mut *st;
        inner.take_ahead(st, None)?;
        let want: Vec<usize> = experts.iter().map(|&e| e as usize).collect();
        let loads = plan_prefill(&mut st.caches[il], &want);
        let jobs: Vec<Job> = loads
            .iter()
            .flat_map(|&(e, s)| inner.reads(il, e, To::Slot(s)))
            .collect();
        inner.pool.run(jobs)?;
        Ok(loads.len())
    }

    /// The oldest batch's copies are done.
    pub fn swaps_landed(&self) {
        self.inner.swaps_landed();
    }

    pub fn stats(&self) -> HostStats {
        let c = &self.inner.counts;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        HostStats {
            vram_pairs: l(&c.vram_pairs),
            ram_pairs: l(&c.ram_pairs),
            ssd_pairs: l(&c.ssd_pairs),
            ram_loads: l(&c.ram_loads),
            stage_loads: l(&c.stage_loads),
            ahead_loads: l(&c.ahead_loads),
            write_backs: l(&c.write_backs),
            read_bytes: self.inner.pool.bytes_read(),
            fetch_ns: l(&c.fetch_ns),
            fetches: l(&c.fetches),
        }
    }
}

impl Drop for HostTier {
    fn drop(&mut self) {
        self.inner.settle();
        self.inner.report();
        REGISTRY.lock().unwrap()[self.id as usize] = None;
    }
}

/// The staging tensors `srcs` need: one per kind and tensor type and shape, as the `(layer,
/// kind)` of its first user, and every layer's index into them per kind.
pub(crate) fn classes(
    srcs: &[[crate::gguf::TensorInfo; 3]],
) -> (Vec<(usize, usize)>, Vec<[usize; 3]>) {
    let mut classes: Vec<(usize, usize)> = Vec::new();
    let of = srcs
        .iter()
        .enumerate()
        .map(|(il, l)| {
            [0, 1, 2].map(|k| {
                let same = |&(jl, j): &(usize, usize)| {
                    let (a, b) = (&srcs[jl][j], &l[k]);
                    j == k && a.ggml_type == b.ggml_type && a.shape == b.shape && a.size == b.size
                };
                classes.iter().position(same).unwrap_or_else(|| {
                    classes.push((il, k));
                    classes.len() - 1
                })
            })
        })
        .collect();
    (classes, of)
}

/// One expert of every kind of every layer of `srcs`, and one expert of every staging tensor
/// ([`classes`]): what [`size`] counts in.
pub fn expert_bytes(srcs: &[[crate::gguf::TensorInfo; 3]], n_expert: usize) -> (u64, u64) {
    let one = |t: &crate::gguf::TensorInfo| t.size / n_expert as u64;
    let per_slot = srcs.iter().flatten().map(one).sum();
    let stage_one = classes(srcs)
        .0
        .iter()
        .map(|&(il, k)| one(&srcs[il][k]))
        .sum();
    (per_slot, stage_one)
}

/// The bytes a tiered range maps for the RAM slot tensor `ram` and the staging tensor `tail`.
fn spans(ram: Tensor<'_>, tail: Tensor<'_>) -> Result<(usize, usize), String> {
    #[cfg(feature = "cuda")]
    {
        super::cuda::arena_span(ram)
            .zip(super::cuda::arena_span(tail))
            .ok_or_else(|| "the host expert tier needs a pinned host arena".to_string())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ram, tail);
        Err("the host expert tier needs a CUDA build".to_string())
    }
}

/// What a step's fetch does for its misses: per expert the index past the VRAM slots, and the
/// reads `(expert, into a RAM slot or None for its staging place)`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Targets {
    /// Expert -> index past the VRAM slots (`usize::MAX`: not routed to).
    pub index: Vec<usize>,
    pub loads: Vec<(usize, Option<usize>)>,
    /// Expert -> it was in a RAM slot before this step.
    pub cached: Vec<bool>,
}

/// Plans a step that reads `need` (distinct experts no VRAM slot holds) of a layer with
/// `cache`: cached ones from their RAM slots, the others read into the slots this step does
/// not use, least recently used first, and past those into their staging places.
pub(crate) fn plan_step(cache: &mut SlotCache, n_expert: usize, need: &[usize]) -> Targets {
    let slots = cache.n_slots();
    let mut t = Targets {
        index: vec![usize::MAX; n_expert],
        loads: Vec::new(),
        cached: vec![false; n_expert],
    };
    for &e in need {
        t.cached[e] = cache.slot(e).is_some();
    }
    let (plan, over) = cache.plan_some(need);
    t.loads
        .extend(plan.loads.iter().map(|&(e, s)| (e, Some(s))));
    for &e in need {
        if let Some(s) = cache.slot(e) {
            t.index[e] = s;
        }
    }
    for e in over {
        t.index[e] = slots + GAP + e;
        t.loads.push((e, None));
    }
    t
}

/// The `(expert, slot)` reads that put `experts` (in order) into `cache`'s empty slots, until
/// none is left; the ones cached already are passed over.
pub(crate) fn plan_prefill(cache: &mut SlotCache, experts: &[usize]) -> Vec<(usize, usize)> {
    let mut loads = Vec::new();
    for &e in experts {
        if cache.slot(e).is_some() {
            continue;
        }
        match cache.take_empty(e) {
            Some(s) => loads.push((e, s)),
            None => break,
        }
    }
    loads
}

/// Where a read puts an expert: a RAM slot, or a staging place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum To {
    Slot(usize),
    Place(usize),
}

impl Inner {
    /// The reads of expert `e` of layer `il` into RAM slot `to`, or into staging place `to`.
    fn reads(&self, il: usize, e: usize, to: To) -> [Job; 3] {
        let mem = &self.layers[il];
        [0, 1, 2].map(|k| {
            let n = mem.bytes[k];
            Job::Raw {
                src: FileRange {
                    file: mem.src[k].file,
                    offset: mem.src[k].offset + (e * n) as u64,
                    len: n as u64,
                },
                dst: match to {
                    To::Slot(s) => mem.ram[k] + s * n,
                    To::Place(p) => mem.stage[k] + p * n,
                },
            }
        })
    }

    /// Waits for the read ahead under way, if any, and gives its `(expert, place)` when it is
    /// layer `il`'s. Anything that writes or reads staging places starts with this: another
    /// layer's read ahead is dropped, its places free again. So are those of a staged prompt
    /// pass, done by now ([`Self::settle`]).
    fn take_ahead(&self, st: &mut State, il: Option<usize>) -> Result<Vec<(usize, usize)>, String> {
        self.settle();
        let Some(a) = st.ahead.take() else {
            return Ok(Vec::new());
        };
        a.group.wait()?;
        Ok(if Some(a.layer) == il {
            a.at
        } else {
            Vec::new()
        })
    }

    /// The fetch of layer `il` for a step's `ids` into `out`.
    fn fetch(&self, il: usize, n_slots: usize, ids: &[i32], out: &mut [i32]) -> Result<(), String> {
        let t0 = Instant::now();
        let mut seen = vec![false; self.n_expert];
        let mut need = Vec::new();
        for &id in ids {
            let Some(e) = (id as usize).checked_sub(n_slots) else {
                continue;
            };
            if id < 0 || e >= self.n_expert {
                return Err(format!("layer {il}: id {id} is no slot or expert"));
            }
            if !seen[e] {
                seen[e] = true;
                need.push(e);
            }
        }
        let c = &self.counts;
        if need.is_empty() {
            out.copy_from_slice(ids);
            c.vram_pairs.fetch_add(ids.len() as u64, Ordering::Relaxed);
            return Ok(());
        }
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        // The experts read ahead for this layer are in their places.
        let ready = self.take_ahead(st, Some(il))?;
        let mut place = vec![usize::MAX; self.n_expert];
        for &(e, p) in &ready {
            place[e] = p;
        }
        let cache = &mut st.caches[il];
        need.retain(|&e| place[e] == usize::MAX || cache.slot(e).is_some());
        let mut t = plan_step(cache, self.n_expert, &need);
        // The experts the RAM slots cannot hold: places after those read ahead.
        let over: Vec<usize> = t
            .loads
            .iter()
            .filter(|l| l.1.is_none())
            .map(|l| l.0)
            .collect();
        let set = il % self.sets;
        let (at, spilled) = assign_places(over.len(), set, self.sets, self.places, ready.len())
            .ok_or_else(|| format!("layer {il}: {} experts to stage, no place", over.len()))?;
        let mut jobs: Vec<Job> = Vec::new();
        for (&e, &p) in over.iter().zip(&at) {
            place[e] = p;
            jobs.extend(self.reads(il, e, To::Place(p)));
        }
        for &(e, s) in &t.loads {
            if let Some(s) = s {
                jobs.extend(self.reads(il, e, To::Slot(s)));
            }
        }
        for (e, &p) in place.iter().enumerate() {
            if p != usize::MAX && cache.slot(e).is_none() {
                t.index[e] = self.slots + GAP + p;
            }
        }
        let group = self.pool.submit(jobs);
        // A big step: the next layer's experts start now, into the other set, which the layer
        // before is done with (the stream runs this after its kernels).
        let next = il + 1;
        let big = ids.len() >= AHEAD_PAIRS_PER_EXPERT * self.n_expert;
        if big {
            // What this step took from the checkpoint: the layer's next read ahead.
            let read = ready.iter().map(|r| r.0).filter(|&e| seen[e]);
            st.last_big[il] = read.chain(t.loads.iter().map(|l| l.0)).collect();
        }
        if big && self.sets > 1 && !spilled && next < self.layers.len() {
            let last = &st.last_big[next];
            let experts = plan_ahead(&st.caches[next], &st.in_vram[next], last, self.places);
            let base = (next % self.sets) * self.places;
            let at: Vec<(usize, usize)> = experts
                .iter()
                .enumerate()
                .map(|(i, &e)| (e, base + i))
                .collect();
            let jobs: Vec<Job> = at
                .iter()
                .flat_map(|&(e, p)| self.reads(next, e, To::Place(p)))
                .collect();
            c.ahead_loads.fetch_add(at.len() as u64, Ordering::Relaxed);
            st.ahead = Some(Ahead {
                layer: next,
                group: self.pool.submit(jobs),
                at,
            });
        }
        group.wait()?;
        let (mut vram, mut ram, mut ssd) = (0u64, 0u64, 0u64);
        for (o, &id) in out.iter_mut().zip(ids) {
            match (id as usize).checked_sub(n_slots) {
                None => {
                    *o = id;
                    vram += 1;
                }
                Some(e) => {
                    // An id past the layer's RAM slots and places would send the matmul
                    // outside the tensor: an error here, not an illegal access there.
                    if t.index[e] >= self.slots + GAP + self.places * self.sets {
                        return Err(format!("layer {il}: expert {e} has no RAM slot or place"));
                    }
                    *o = (n_slots + t.index[e]) as i32;
                    if t.cached[e] {
                        ram += 1;
                    } else {
                        ssd += 1;
                    }
                }
            }
        }
        let staged = t.loads.iter().filter(|l| l.1.is_none()).count() as u64;
        c.vram_pairs.fetch_add(vram, Ordering::Relaxed);
        c.ram_pairs.fetch_add(ram, Ordering::Relaxed);
        c.ssd_pairs.fetch_add(ssd, Ordering::Relaxed);
        c.stage_loads.fetch_add(staged, Ordering::Relaxed);
        c.ram_loads
            .fetch_add(t.loads.len() as u64 - staged, Ordering::Relaxed);
        c.fetch_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if (c.fetches.fetch_add(1, Ordering::Relaxed) + 1).is_multiple_of(REPORT_EVERY) {
            self.report();
        }
        Ok(())
    }

    fn sources(&self, il: usize, experts: &[u32]) -> Result<Vec<usize>, String> {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        self.take_ahead(st, None)?;
        let mut jobs = Vec::new();
        let mut used = 0;
        let mut at = Vec::with_capacity(experts.len());
        for &e in experts {
            let e = e as usize;
            st.in_vram[il][e] = true;
            at.push(match st.caches[il].slot(e) {
                Some(s) => s,
                None => {
                    if used == self.places * self.sets {
                        return Err(format!("layer {il}: more experts than staging places"));
                    }
                    jobs.extend(self.reads(il, e, To::Place(used)));
                    used += 1;
                    self.slots + GAP + used - 1
                }
            });
        }
        self.pool.run(jobs)?;
        Ok(at)
    }

    fn swap(&self, il: usize, new: usize, old: Option<usize>) -> Result<SwapAt, String> {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        self.take_ahead(st, None)?;
        let f = plan_swap(&mut st.caches[il], il, new, old);
        st.in_vram[il][new] = true;
        if let Some(o) = old {
            st.in_vram[il][o] = false;
        }
        // Read first, the new expert is in the first staging place until its copy is done.
        let at = SwapAt {
            src: f.from.unwrap_or(self.slots + GAP),
            back: f.back.map(|b| b.1),
            staged: f.from.is_none(),
        };
        st.open.push(f);
        if at.staged {
            self.pool.run(self.reads(il, new, To::Place(0)).into())?;
        }
        if at.back.is_some() {
            self.counts.write_backs.fetch_add(1, Ordering::Relaxed);
        }
        Ok(at)
    }

    fn swaps_queued(&self) {
        let mut st = self.state.lock().unwrap();
        let batch = std::mem::take(&mut st.open);
        st.flights.push_back(batch);
    }

    fn swaps_landed(&self) {
        let mut st = self.state.lock().unwrap();
        for f in st.flights.pop_front().unwrap_or_default() {
            land_swap(&mut st.caches[f.layer], &f);
        }
    }

    fn report(&self) {
        let c = &self.counts;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let (vram, ram, ssd) = (l(&c.vram_pairs), l(&c.ram_pairs), l(&c.ssd_pairs));
        let pairs = (vram + ram + ssd).max(1);
        let pct = |n: u64| format!("{:.1}%", 100.0 * n as f64 / pairs as f64);
        tracing::info!(
            vram_hit = pct(vram),
            ram_hit = pct(ram),
            ssd = pct(ssd),
            vram_pairs = vram,
            ram_pairs = ram,
            ssd_pairs = ssd,
            ram_loads = l(&c.ram_loads),
            stage_loads = l(&c.stage_loads),
            ahead_loads = l(&c.ahead_loads),
            write_backs = l(&c.write_backs),
            read_gib = self.pool.bytes_read() as f64 / (1u64 << 30) as f64,
            fetch_wait_s = l(&c.fetch_ns) as f64 / 1e9,
            fetches = l(&c.fetches),
            "host expert tier"
        );
    }
}

/// The fetch op: `src[0]` the step's ids (`[n_used, n_tokens]` I32), out the indices the
/// matmuls read; params `[tier id, layer, the layer's VRAM slots]`.
pub static FETCH: ExternalOp = ExternalOp {
    name: "host_expert_fetch",
    cpu: fetch_cpu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::fetch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// # Safety
/// `dst` must be a node made by [`HostTier::fetch`], in host memory.
unsafe fn fetch_cpu(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    let inner = lookup(p[0]);
    let n = (dst.ne[0] * dst.ne[1]) as usize;
    // SAFETY: the op's I32 source (the step's ids, contiguous) and output (`n` ids), in host
    // memory.
    let (ids, out) = unsafe {
        let s = &*dst.src[0];
        (
            std::slice::from_raw_parts(s.data as *const i32, n),
            std::slice::from_raw_parts_mut(dst.data as *mut i32, n),
        )
    };
    if let Err(e) = inner.fetch(p[1] as usize, p[2] as usize, ids, out) {
        fatal(&e);
    }
}

/// A read failed inside a step: the op has no way to fail the step, and the slot's bytes are
/// not the expert's.
fn fatal(e: &str) -> ! {
    tracing::error!(error = e, "host expert tier: a read failed");
    eprintln!("host expert tier: {e}");
    std::process::abort();
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};
    use std::sync::{Arc, OnceLock};

    use super::{Inner, external_params, fatal, lookup};
    use crate::ggml::ffi;

    unsafe extern "C" {
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: *mut c_void,
        ) -> c_int;
        fn cudaLaunchHostFunc(
            s: *mut c_void,
            f: unsafe extern "C" fn(*mut c_void),
            user: *mut c_void,
        ) -> c_int;
        fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    }

    /// `cudaMemcpyDefault`: the direction from the pointers (unified addressing).
    const DEFAULT: c_int = 4;

    struct Job {
        inner: Arc<Inner>,
        layer: usize,
        n_slots: usize,
        n: usize,
    }

    unsafe extern "C" fn host(user: *mut c_void) {
        // SAFETY: the box `fetch` leaked for this call.
        let job = unsafe { Box::from_raw(user as *mut Job) };
        let (ids_in, ids_out) = job.inner.xchg.expect("exchange buffers");
        // SAFETY: the pinned exchange buffers, `cap >= n` ids each; the stream orders this
        // between the copy in and the copy out.
        let (ids, out) = unsafe {
            (
                std::slice::from_raw_parts(ids_in as *const i32, job.n),
                std::slice::from_raw_parts_mut(ids_out as *mut i32, job.n),
            )
        };
        if let Err(e) = job.inner.fetch(job.layer, job.n_slots, ids, out) {
            fatal(&e);
        }
    }

    /// Whether the fetch runs as a host function on the stream (`LEGION_DEBUG_TIER_FETCH=
    /// hostfunc`, the way before) instead of on the launching thread.
    fn as_host_function() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("LEGION_DEBUG_TIER_FETCH").is_some_and(|v| v == "hostfunc")
        })
    }

    /// Copies the ids to pinned memory, runs the fetch, and copies the result into the op's
    /// output. The launching thread waits for the ids and runs the fetch itself: a host
    /// function on the stream starts only once a driver thread has woken up for it, at every
    /// layer of every step, and that wake-up is longer than the wait here (the thread polls).
    pub(super) unsafe fn fetch(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let p = external_params(dst);
        let inner = lookup(p[0]);
        let n = (dst.ne[0] * dst.ne[1]) as usize;
        // SAFETY: the op's source, made by `HostTier::fetch`: the step's ids.
        let src = unsafe { &*dst.src[0] };
        let (ids_in, ids_out) = inner.xchg.ok_or("no exchange buffers")?;
        if n > inner.cap {
            return Err(format!("{n} ids, room for {}", inner.cap));
        }
        if !as_host_function() {
            // SAFETY: device and pinned buffers of `n` ids; after the wait the stream is done
            // with everything before this op (the copy in, the layer before and the copy out
            // of its fetch), so this thread reads and writes the exchange buffers alone.
            unsafe {
                let e = cudaMemcpyAsync(ids_in as *mut c_void, src.data, n * 4, DEFAULT, stream);
                if e != 0 {
                    return Err(format!("ids to host: CUDA error {e}"));
                }
                let e = cudaStreamSynchronize(stream);
                if e != 0 {
                    return Err(format!("waiting for the ids: CUDA error {e}"));
                }
                let ids = std::slice::from_raw_parts(ids_in as *const i32, n);
                let out = std::slice::from_raw_parts_mut(ids_out as *mut i32, n);
                if let Err(e) = inner.fetch(p[1] as usize, p[2] as usize, ids, out) {
                    fatal(&e);
                }
                let e = cudaMemcpyAsync(dst.data, ids_out as *const c_void, n * 4, DEFAULT, stream);
                if e != 0 {
                    return Err(format!("ids to device: CUDA error {e}"));
                }
            }
            return Ok(());
        }
        let job = Box::into_raw(Box::new(Job {
            inner: inner.clone(),
            layer: p[1] as usize,
            n_slots: p[2] as usize,
            n,
        }));
        // SAFETY: device and pinned buffers of `n` ids; the host function owns `job`.
        unsafe {
            let e = cudaMemcpyAsync(ids_in as *mut c_void, src.data, n * 4, DEFAULT, stream);
            if e != 0 {
                drop(Box::from_raw(job));
                return Err(format!("ids to host: CUDA error {e}"));
            }
            let e = cudaLaunchHostFunc(stream, host, job.cast());
            if e != 0 {
                drop(Box::from_raw(job));
                return Err(format!("host function: CUDA error {e}"));
            }
            let e = cudaMemcpyAsync(dst.data, ids_out as *const c_void, n * 4, DEFAULT, stream);
            if e != 0 {
                return Err(format!("ids to device: CUDA error {e}"));
            }
        }
        Ok(())
    }
}
