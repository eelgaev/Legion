//! The SSD expert tier (#560): a model whose routed experts do not fit in memory keeps a cache
//! of them and reads the others from the checkpoint as steps route to them.
//!
//! GLM-5.3-Flash NVFP4 is 191 GB as shipped, 171 GB of it routed experts; a DGX Spark (GB10)
//! has 121 GB for everything. The tier loads every other weight as usual, and for each MoE layer
//! keeps `n_slots` experts in slots in host memory (on the GB10 the GPU reads it in place). Like
//! Strata's `--ple-io` and Legion's n-gram table it is automatic: a GB10 whose experts do not fit
//! turns it on without any setup.
//!
//! Per MoE layer and step, after the router, an external op ([`FETCH`]) gets the step's expert
//! ids on the host (on CUDA: copied to pinned memory, then a host function on the backend's
//! stream, so it runs once the layers before it are done and the stream waits for it):
//!
//! - **Small steps** (decode, verify: at most `n_slots` routed pairs): every routed expert gets a
//!   slot ([`cache::SlotCache`], least recently used out), the missing ones are read from the
//!   checkpoint into their slots, and the op outputs the ids mapped to slots. The layer's
//!   matmuls read the slot tensors.
//! - **Chunked steps** (decode and verify steps of more pairs than slots, #1205): the model
//!   splits the step's rows into pieces of at most `n_slots` pairs ([`SsdExperts::pieces`]),
//!   each a fetch in slot mode and its own matmuls, one after the other. A piece's fetch sees
//!   the whole step's ids and evicts the experts the later pieces need last.
//! - **Big steps** (prompts): the layer's routed experts go into a staging buffer of all
//!   `n_expert` experts: the cached ones copied from their slots, the others read from the
//!   checkpoint; slots no expert holds yet take a copy. The op outputs the ids unchanged and the
//!   matmuls read the staging buffer. Two staging buffers alternate between layers: while the
//!   GPU computes a layer, the reader threads already read the next layer's uncached experts
//!   into the other one (a prompt step routes to nearly every expert).
//!
//! Reads go around the page cache (see [`io`]). Results are bit-identical to the experts loaded
//! whole: the same NVFP4 bytes, scales and kernels, only other addresses.
//!
//! GLM only (the model wires it; the checkpoint loader records every stacked NVFP4 expert
//! tensor's per-expert sources, [`crate::gguf::Nvfp4Stack`]); single stage.

pub mod cache;
pub mod io;
pub mod lookahead;
pub mod replay;
#[cfg(test)]
mod tests;
pub mod trace;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::ggml::{BufferType, Context, ExternalOp, Tensor, external_params, ffi, types};
use crate::gguf::{GgufFile, Nvfp4Stack};

use cache::SlotCache;
use io::{Group, Job, Pool};

/// The expert tensors of a MoE layer, in the order [`SsdLayer::weights`] holds them.
pub const KINDS: [&str; 3] = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"];

/// Slot-mode steps of the first layer between hit-rate reports.
const REPORT_EVERY: u64 = 128;

/// Decode and verify steps of more routed pairs than slots go through the slots in at most
/// this many pieces (#1205); bigger ones stage. A staged step moves every distinct routed
/// expert once (a copy from its slot or a read), at most `n_expert`, and never updates the
/// cache; a chunked one reads only its pieces' misses and keeps the hot experts cached, but
/// each piece is a host round trip and, once the step's distinct experts pass the slots, a
/// piece can evict what a later one reads again. Eight cover every decode or verify step of 16
/// sequences at the default 4 drafts (80 rows: 7 pieces at 109 slots and 8 used), and keep the
/// pairs of a chunked step under `8 * n_slots` (872 at 109 slots: the staged step's kernels too
/// take one token at a time there, the same sums).
const MAX_PIECES: usize = 8;

/// When the tier is used (`--expert-ssd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ExpertSsd {
    /// On a GPU that shares the host's memory (GB10), when the experts do not fit.
    #[default]
    Auto,
    /// Always (where the model supports it), sized by `--expert-ssd-cache-gib`.
    On,
    /// Never.
    Off,
}

/// What the tier reads ahead of a small step (`--ssd-prefetch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum SsdPrefetch {
    /// Nothing: each step reads its missing experts when it needs them.
    #[default]
    Off,
    /// The router lookahead: while layer L computes, the experts layer L+1's router ranks
    /// best on layer L's post-attention state (each token's `n_used` + margin).
    Router,
    /// The same from layer L's output (after its MoE), through layer L+1's FFN mixer and norm:
    /// a better prediction (#611's replay: 77.8% recall at margin 0 against 72.0%) made later,
    /// with only layer L+1's attention left to hide the reads behind.
    RouterPost,
    /// Both: reads start from the post-attention prediction, and the post-MoE one then cancels
    /// the reads it does not name that have not started and adds the experts it names.
    RouterBoth,
}

/// The tier's settings (`--expert-ssd*`, `--ssd-prefetch*`).
#[derive(Debug, Clone, PartialEq)]
pub struct SsdArgs {
    pub mode: ExpertSsd,
    /// Memory for slots and staging buffers; `None`: what is free after the other weights, the
    /// KV cache and the step buffers, less [`Self::reserve`].
    pub cache_bytes: Option<u64>,
    /// What an automatic size leaves free besides the KV cache and the step buffers.
    pub reserve: u64,
    /// Reader threads.
    pub readers: usize,
    /// Read around the page cache (`O_DIRECT`).
    pub direct: bool,
    /// Reads ahead of small steps.
    pub prefetch: SsdPrefetch,
    /// Experts per token the lookahead reads beyond the `n_used` it ranks best.
    pub prefetch_margin: usize,
    /// Experts per token the lookahead reads ahead at most, best first (0: no cap): fewer
    /// than `n_used` reads only the surest of the routed count.
    pub prefetch_ranks: usize,
    /// The post-attention lookahead of layer L+1 is queued before layer L's fetch instead of
    /// after it (its reads wait behind the fetch's own).
    pub prefetch_early: bool,
}

impl Default for SsdArgs {
    fn default() -> Self {
        Self {
            mode: ExpertSsd::Auto,
            cache_bytes: None,
            reserve: 2 << 30,
            readers: 16,
            direct: true,
            prefetch: SsdPrefetch::Off,
            prefetch_margin: 0,
            prefetch_ranks: 0,
            prefetch_early: false,
        }
    }
}

/// The margin a trace's predictions record (each token's best `n_used` plus this many).
pub const TRACE_MARGIN: usize = 16;

/// The routed expert tensor names of layer `il` (per kind: weight, per-expert scale).
pub fn layer_tensors(il: usize) -> [(String, String); 3] {
    KINDS.map(|k| {
        (
            format!("blk.{il}.{k}.weight"),
            format!("blk.{il}.{k}.scale"),
        )
    })
}

/// The layers of `gguf` (of `n_layer`) whose routed experts the tier can read: all three
/// kinds stacked NVFP4 with per-expert sources. `None` if some layer has routed experts the
/// tier cannot read, or none has any.
pub fn tier_layers(gguf: &GgufFile, n_layer: usize) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for il in 0..n_layer {
        let names = layer_tensors(il);
        let present = names
            .iter()
            .filter(|(w, _)| gguf.tensor(w).is_some())
            .count();
        if present == 0 {
            continue;
        }
        if present < 3 || names.iter().any(|(w, _)| gguf.nvfp4_stack(w).is_none()) {
            return None;
        }
        out.push(il);
    }
    (!out.is_empty()).then_some(out)
}

/// The tensors the tier reads itself, left out of the weights' load.
pub fn skipped_tensors(layers: &[usize]) -> Vec<String> {
    layers
        .iter()
        .flat_map(|&il| layer_tensors(il))
        .flat_map(|(w, s)| [w, s])
        .collect()
}

/// Bytes of one expert matrix of `stack` as ggml NVFP4 rows.
fn matrix_bytes(s: &Nvfp4Stack) -> usize {
    s.rows * s.k / 64 * 36
}

/// What the tier holds and how fast it reads.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SsdStats {
    pub layers: usize,
    pub slots: usize,
    pub cache_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    /// Tokens of the small steps (per layer).
    pub tokens: u64,
    /// Bytes read for small steps' misses, and the time their reads took.
    pub miss_bytes: u64,
    pub miss_ns: u64,
    /// Bytes read for big steps (staging and read ahead).
    pub stage_bytes: u64,
    /// `--ssd-prefetch`: experts read ahead of small steps, those their step used, those it
    /// did not (read in full), those dropped before their read completed; the bytes read
    /// ahead, and the time steps waited for reads ahead still running.
    pub prefetched: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    pub prefetch_dropped: u64,
    pub prefetch_bytes: u64,
    pub prefetch_wait_ns: u64,
    /// The time small steps' fetches took in all (the stream waits for them): the demand
    /// reads' latency, with or without reads ahead.
    pub fetch_ns: u64,
}

impl SsdStats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

/// One MoE layer's slot memory and sources.
struct LayerMem {
    src: [Arc<Nvfp4Stack>; 3],
    /// Host addresses of the slot tensors (kind-major) and their per-slot scales.
    slots: [usize; 3],
    slot_scale: [usize; 3],
}

/// What a step's fetch does (the op's mode parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Slots,
    Stage(usize),
}

impl Mode {
    fn param(self) -> i32 {
        match self {
            Mode::Slots => 0,
            Mode::Stage(b) => 1 + b as i32,
        }
    }

    fn from_param(p: i32) -> Self {
        match p {
            0 => Mode::Slots,
            p => Mode::Stage((p - 1) as usize),
        }
    }
}

/// Reads of the next layer's experts into a staging buffer, ahead of its step.
struct Ahead {
    ord: usize,
    buf: usize,
    experts: Vec<usize>,
    group: Arc<Group>,
}

/// A read of a predicted expert into a layer's slot ahead of its small step (`--ssd-prefetch`).
/// The slot is pinned until the read ends; its expert counts as cached meanwhile.
struct Load {
    e: usize,
    s: usize,
    group: Arc<Group>,
    /// Whether the step it was read for routes to it: `None` until that step's fetch (or a
    /// later prediction that no longer names it: `Some(false)`, cancelled).
    used: Option<bool>,
}

struct State {
    caches: Vec<SlotCache>,
    /// Per staging buffer: the layer (ordinal) whose experts it holds and which.
    held: [Option<(usize, Vec<bool>)>; 2],
    ahead: Option<Ahead>,
    /// Per layer (ordinal): its reads ahead not yet ended and counted, those for its next step
    /// and those cancelled ones that may still run.
    inflight: Vec<Vec<Load>>,
    /// `LEGION_SSD_EXPERT_TRACE`: every fetch's routing, for offline replays.
    trace: Option<trace::Writer>,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    tokens: AtomicU64,
    steps: AtomicU64,
    miss_bytes: AtomicU64,
    miss_ns: AtomicU64,
    stage_bytes: AtomicU64,
    prefetched: AtomicU64,
    prefetch_used: AtomicU64,
    prefetch_wasted: AtomicU64,
    prefetch_dropped: AtomicU64,
    prefetch_bytes: AtomicU64,
    prefetch_wait_ns: AtomicU64,
    fetch_ns: AtomicU64,
}

/// What the fetch op reaches through its registry id.
struct Inner {
    n_expert: usize,
    n_slots: usize,
    layers: Vec<LayerMem>,
    /// Per staging buffer: the three kinds' bases and per-expert scales.
    stage: [[usize; 3]; 2],
    stage_scale: [[usize; 3]; 2],
    /// Pinned exchange buffers for the ids (CUDA), `cap` ids each.
    xchg: Option<(usize, usize)>,
    cap: usize,
    n_used: usize,
    /// Predictions ([`SsdExperts::predict`]) are made: for the trace, the prefetch or both.
    predicts: bool,
    /// The trace was opened (its predictions rank [`TRACE_MARGIN`] more experts).
    traced: bool,
    prefetch: SsdPrefetch,
    margin: usize,
    /// Ranks per token read ahead at most (`usize::MAX`: no cap).
    ranks: usize,
    early: bool,
    /// The pinned buffer predictions come through (CUDA), `pcap` ids.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pxchg: Option<usize>,
    pcap: usize,
    pool: Pool,
    state: Mutex<State>,
    counts: Counters,
}

// SAFETY: the addresses point into host buffers the owning `SsdExperts` keeps alive; every
// write through them is ordered by the state lock and the backend's stream.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

static REGISTRY: Mutex<Vec<Option<Arc<Inner>>>> = Mutex::new(Vec::new());

fn lookup(id: i32) -> Arc<Inner> {
    REGISTRY.lock().unwrap()[id as usize]
        .clone()
        .expect("a live SSD expert tier")
}

/// The tier: slot and staging tensors plus the state the fetch op works on.
pub struct SsdExperts {
    inner: Arc<Inner>,
    id: i32,
    /// Layer -> ordinal among the tier's layers.
    ord: Vec<Option<usize>>,
    /// Whether each layer's experts have per-expert scales (as the loaded tensors would).
    scaled: Vec<bool>,
    n_used: usize,
    /// Slot tensors, staging tensors and their scales. Last: dropped after the reader threads.
    ctx: Context,
}

/// A layer's experts for one step.
pub struct SsdLayer<'a> {
    /// up, gate, down (as [`KINDS`]).
    pub weights: [Tensor<'a>; 3],
    pub scales: Option<[Tensor<'a>; 3]>,
    /// The step's ids into `weights`.
    pub ids: Tensor<'a>,
}

impl SsdExperts {
    /// The tier over `layers` of `gguf` ([`tier_layers`]) with `bytes` for slots and staging,
    /// its memory in `buft` (host-addressable), `max_tokens` the biggest step.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gguf: &GgufFile,
        layers: &[usize],
        n_layer: usize,
        n_expert: usize,
        n_used: usize,
        bytes: u64,
        buft: BufferType,
        max_tokens: usize,
        args: &SsdArgs,
    ) -> Result<Self, String> {
        if !buft.is_host() {
            return Err(format!("{}: not host memory", buft.name()));
        }
        let mut srcs = Vec::with_capacity(layers.len());
        let mut scaled = vec![false; n_layer + 1];
        for &il in layers {
            let names = layer_tensors(il);
            let src = names.clone().map(|(w, _)| gguf.nvfp4_stack(&w));
            let [Some(u), Some(g), Some(d)] = src else {
                return Err(format!("layer {il}: no per-expert sources"));
            };
            if [&u, &g, &d].iter().any(|s| s.experts.len() != n_expert) {
                return Err(format!("layer {il}: not {n_expert} experts"));
            }
            scaled[il] = gguf.tensor(&names[0].1).is_some();
            srcs.push([u, g, d]);
        }
        let first = srcs.first().ok_or("no layers")?;
        let kind_bytes: [usize; 3] = [0, 1, 2].map(|k| matrix_bytes(&first[k]));
        let expert_bytes: usize = kind_bytes.iter().sum();
        let stage_bytes = 2 * n_expert * expert_bytes;
        let per_slot = layers.len() * expert_bytes;
        let n_slots = (bytes as usize).saturating_sub(stage_bytes) / per_slot;
        let n_slots = n_slots.min(n_expert);
        if n_slots < n_used {
            return Err(format!(
                "{:.1} GiB holds {n_slots} expert slots per layer, fewer than the {n_used} a token \
                 routes to (staging takes {:.1} GiB)",
                bytes as f64 / (1u64 << 30) as f64,
                stage_bytes as f64 / (1u64 << 30) as f64
            ));
        }
        let trace = open_trace(layers, n_expert, n_slots);
        let predicts = trace.is_some() || args.prefetch != SsdPrefetch::Off;
        // Predictions come from small steps: at most `n_slots / n_used` tokens of ranked ids.
        let width = n_used + TRACE_MARGIN.max(args.prefetch_margin);
        let pcap = (n_slots / n_used.max(1)).max(1) * width;
        let ctx = Context::new(layers.len() * 6 + 12 + 3, 0);
        let shape = |s: &Nvfp4Stack, n: usize| [s.k as i64, s.rows as i64, n as i64];
        for (i, &il) in layers.iter().enumerate() {
            for (k, kind) in KINDS.iter().enumerate() {
                ctx.new_tensor(types::NVFP4, &shape(&srcs[i][k], n_slots))
                    .named(&format!("ssd.{il}.{kind}"));
                ctx.new_tensor(types::F32, &[n_slots as i64])
                    .named(&format!("ssd.{il}.{kind}.scale"));
            }
        }
        for b in 0..2 {
            for (k, kind) in KINDS.iter().enumerate() {
                ctx.new_tensor(types::NVFP4, &shape(&first[k], n_expert))
                    .named(&format!("ssd.stage{b}.{kind}"));
                ctx.new_tensor(types::F32, &[n_expert as i64])
                    .named(&format!("ssd.stage{b}.{kind}.scale"));
            }
        }
        let cap = n_used * max_tokens.max(1);
        let cuda = cfg!(feature = "cuda") && !buft.is_plain_cpu();
        if cuda {
            ctx.new_tensor(types::I32, &[cap as i64])
                .named("ssd.ids_in");
            ctx.new_tensor(types::I32, &[cap as i64])
                .named("ssd.ids_out");
            if predicts {
                ctx.new_tensor(types::I32, &[pcap as i64])
                    .named("ssd.pred_in");
            }
        }
        ctx.alloc_tensors(buft)
            .map_err(|e| format!("SSD expert tier: {e}"))?;
        let addr = |name: &str| -> usize {
            // SAFETY: an allocated tensor of this context, in host memory.
            unsafe { (*ctx.tensor(name).expect("tier tensor").raw()).data as usize }
        };
        let mut mems = Vec::with_capacity(layers.len());
        for (i, &il) in layers.iter().enumerate() {
            mems.push(LayerMem {
                src: srcs[i].clone(),
                slots: KINDS.map(|k| addr(&format!("ssd.{il}.{k}"))),
                slot_scale: KINDS.map(|k| addr(&format!("ssd.{il}.{k}.scale"))),
            });
        }
        // Slot scales start at 1 (unused slots are never read; the values only keep NaNs out).
        for il in layers {
            for k in KINDS {
                let t = ctx.tensor(&format!("ssd.{il}.{k}.scale")).unwrap();
                t.set_f32(&vec![1.0; n_slots]);
            }
        }
        let stage = [0, 1].map(|b| KINDS.map(|k| addr(&format!("ssd.stage{b}.{k}"))));
        let stage_scale = [0, 1].map(|b| KINDS.map(|k| addr(&format!("ssd.stage{b}.{k}.scale"))));
        let xchg = cuda.then(|| (addr("ssd.ids_in"), addr("ssd.ids_out")));
        let pxchg = (cuda && predicts).then(|| addr("ssd.pred_in"));
        let files: Vec<PathBuf> = gguf.shard_paths().map(|p| p.to_path_buf()).collect();
        let pool = Pool::new(&files, args.readers, args.direct)?;
        let inner = Arc::new(Inner {
            n_expert,
            n_slots,
            layers: mems,
            stage,
            stage_scale,
            xchg,
            cap,
            n_used,
            predicts,
            traced: trace.is_some(),
            prefetch: args.prefetch,
            margin: args.prefetch_margin,
            ranks: match args.prefetch_ranks {
                0 => usize::MAX,
                r => r,
            },
            early: args.prefetch_early,
            pxchg,
            pcap,
            pool,
            state: Mutex::new(State {
                caches: vec![SlotCache::new(n_expert, n_slots); layers.len()],
                held: [None, None],
                ahead: None,
                inflight: (0..layers.len()).map(|_| Vec::new()).collect(),
                trace,
            }),
            counts: Counters::default(),
        });
        let id = {
            let mut r = REGISTRY.lock().unwrap();
            r.push(Some(inner.clone()));
            (r.len() - 1) as i32
        };
        let mut ord = vec![None; n_layer + 1];
        for (i, &il) in layers.iter().enumerate() {
            ord[il] = Some(i);
        }
        tracing::info!(
            layers = layers.len(),
            slots_per_layer = n_slots,
            experts = n_expert,
            cache_gib = (per_slot * n_slots) as f64 / (1u64 << 30) as f64,
            staging_gib = stage_bytes as f64 / (1u64 << 30) as f64,
            readers = args.readers,
            direct = inner.pool.direct(),
            prefetch = ?args.prefetch,
            "SSD expert tier: experts not in a slot are read from the checkpoint"
        );
        Ok(Self {
            inner,
            id,
            ord,
            scaled,
            n_used,
            ctx,
        })
    }

    /// Whether layer `il`'s experts come from the tier.
    pub fn has_layer(&self, il: usize) -> bool {
        self.ord.get(il).is_some_and(|o| o.is_some())
    }

    pub fn n_slots(&self) -> usize {
        self.inner.n_slots
    }

    /// Layer `il`'s experts for a step routing `ids` (`[n_used, n_tokens]` I32): the fetch op
    /// and the tensors the step's matmuls read.
    pub fn fetch<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        ids: Tensor<'a>,
    ) -> Result<SsdLayer<'a>, String> {
        let ord = self.ord[il].ok_or_else(|| format!("layer {il} is not in the SSD tier"))?;
        let [n_used, n_tok, _, _] = ids.ne();
        let pairs = (n_used * n_tok) as usize;
        let mode = if pairs <= self.inner.n_slots {
            Mode::Slots
        } else {
            Mode::Stage(ord % 2)
        };
        self.fetch_op(ctx, il, ord, ids, mode, None)
    }

    /// The pieces a step of `n_tok` tokens takes through the slots when its routed pairs
    /// exceed them (#1205): `decoding`, every sequence of the step decodes or verifies, in at
    /// most [`MAX_PIECES`] pieces. `None`: one [`Self::fetch`] (the slots when the step fits
    /// them, else a staging buffer).
    pub fn pieces(&self, n_tok: usize, decoding: bool) -> Option<usize> {
        let i = &self.inner;
        if !decoding || i.n_used * n_tok <= i.n_slots {
            return None;
        }
        let k = n_tok.div_ceil(i.n_slots / i.n_used);
        (k <= MAX_PIECES).then_some(k)
    }

    /// Layer `il`'s experts for rows `rows` of a step routing `ids` (`[n_used, n_tokens]`
    /// I32), at most `n_slots` pairs, in slot mode: a piece of a chunked step
    /// ([`Self::pieces`], [`piece_rows`]). `after`: the previous piece's result, so this
    /// fetch runs once its matmuls have read their slots. The fetch sees every row of `ids`
    /// and keeps the experts of the rows after `rows` cached where it can.
    pub fn fetch_rows<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        ids: Tensor<'a>,
        rows: std::ops::Range<usize>,
        after: Option<Tensor<'a>>,
    ) -> Result<SsdLayer<'a>, String> {
        let ord = self.ord[il].ok_or_else(|| format!("layer {il} is not in the SSD tier"))?;
        let [_, n_tok, _, _] = ids.ne();
        if rows.is_empty() || rows.end > n_tok as usize || rows.len() * self.n_used > self.n_slots()
        {
            return Err(format!(
                "rows {rows:?} of {n_tok}: a piece takes 1 to {} rows",
                self.n_slots() / self.n_used
            ));
        }
        self.fetch_op(ctx, il, ord, ids, Mode::Slots, Some((rows, after)))
    }

    /// The fetch op of layer `il` (ordinal `ord`) in `mode` for `ids`, or for `piece`'s rows
    /// of them, after its tensor if given.
    fn fetch_op<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        ord: usize,
        ids: Tensor<'a>,
        mode: Mode,
        piece: Option<(std::ops::Range<usize>, Option<Tensor<'a>>)>,
    ) -> Result<SsdLayer<'a>, String> {
        let [n_used, n_tok, _, _] = ids.ne();
        let pairs = (n_used * n_tok) as usize;
        if self.inner.xchg.is_some() && pairs > self.inner.cap {
            return Err(format!(
                "a step of {n_tok} tokens: the SSD expert tier takes at most {}",
                self.inner.cap / self.n_used
            ));
        }
        let ids = if ids.is_contiguous() {
            ids
        } else {
            ctx.cont(ids)
        };
        let out = match piece {
            None => ctx.external_with(
                &FETCH,
                types::I32,
                ids.ne(),
                &[ids],
                &[self.id, ord as i32, mode.param()],
            ),
            Some((rows, after)) => {
                let mut src = vec![ids];
                src.extend(after);
                ctx.external_with(
                    &FETCH,
                    types::I32,
                    [n_used, rows.len() as i64, 1, 1],
                    &src,
                    &[
                        self.id,
                        ord as i32,
                        mode.param(),
                        rows.start as i32,
                        rows.len() as i32,
                    ],
                )
            }
        };
        let name = |kind: &str, scale: bool| match mode {
            Mode::Slots if scale => format!("ssd.{il}.{kind}.scale"),
            Mode::Slots => format!("ssd.{il}.{kind}"),
            Mode::Stage(b) if scale => format!("ssd.stage{b}.{kind}.scale"),
            Mode::Stage(b) => format!("ssd.stage{b}.{kind}"),
        };
        let t = |n: String| self.ctx.tensor(&n).expect("tier tensor");
        Ok(SsdLayer {
            weights: KINDS.map(|k| t(name(k, false))),
            scales: self.scaled[il].then(|| KINDS.map(|k| t(name(k, true)))),
            ids: out,
        })
    }

    /// The experts per token a prediction of layer `il`'s routing by `p` should rank, for a step
    /// of `n_tokens`; `None` when none is wanted (no trace and no prefetch by `p`, a big step, a
    /// layer outside the tier).
    pub fn predict_width(&self, il: usize, p: trace::Predictor, n_tokens: usize) -> Option<usize> {
        let i = &self.inner;
        if !i.predicts || !self.has_layer(il) || n_tokens * i.n_used > i.n_slots {
            return None;
        }
        let width = if i.traced {
            i.n_used + TRACE_MARGIN.max(i.margin)
        } else if i.prefetches(p) {
            i.n_used + i.margin
        } else {
            return None;
        };
        (n_tokens * width <= i.pcap).then_some(width)
    }

    /// Whether the post-attention lookahead of the next layer goes before a layer's fetch
    /// (`--ssd-prefetch-early`, when it reads ahead) rather than after it.
    pub fn prefetch_early(&self) -> bool {
        self.inner.early && self.inner.prefetches(trace::Predictor::PostAttn)
    }

    /// The op handing a prediction of layer `il`'s routing to the tier: `ids` (`[width,
    /// n_tokens]` I32, each token's experts best first, [`Self::predict_width`]) by `p`, made
    /// after `after` (the fetch before it, so its reads go first). The trace records it; with
    /// `--ssd-prefetch router` a post-attention prediction reads its experts ahead. A node the
    /// caller expands into the graph; nothing reads its output.
    pub fn predict<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        ids: Tensor<'a>,
        p: trace::Predictor,
        after: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, String> {
        let ord = self.ord[il].ok_or_else(|| format!("layer {il} is not in the SSD tier"))?;
        let [w, n_tok, _, _] = ids.ne();
        if (w * n_tok) as usize > self.inner.pcap {
            return Err(format!("a prediction of {w} x {n_tok} ids"));
        }
        let ids = if ids.is_contiguous() {
            ids
        } else {
            ctx.cont(ids)
        };
        let mut src = vec![ids];
        src.extend(after);
        Ok(ctx.external_with(
            &PREDICT,
            types::I32,
            [1, 1, 1, 1],
            &src,
            &[self.id, ord as i32, p.code() as i32],
        ))
    }

    /// What the prediction op runs, on the host. For tests.
    pub fn predict_host(
        &self,
        il: usize,
        p: trace::Predictor,
        ids: &[i32],
        n_tokens: usize,
    ) -> Result<(), String> {
        let ord = self.ord[il].ok_or("not a tier layer")?;
        self.inner.predict(ord, p, ids, n_tokens)
    }

    pub fn stats(&self) -> SsdStats {
        let c = &self.inner.counts;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        SsdStats {
            layers: self.inner.layers.len(),
            slots: self.inner.n_slots,
            cache_bytes: self.ctx_bytes(),
            hits: l(&c.hits),
            misses: l(&c.misses),
            tokens: l(&c.tokens),
            miss_bytes: l(&c.miss_bytes),
            miss_ns: l(&c.miss_ns),
            stage_bytes: l(&c.stage_bytes),
            prefetched: l(&c.prefetched),
            prefetch_used: l(&c.prefetch_used),
            prefetch_wasted: l(&c.prefetch_wasted),
            prefetch_dropped: l(&c.prefetch_dropped),
            prefetch_bytes: l(&c.prefetch_bytes),
            prefetch_wait_ns: l(&c.prefetch_wait_ns),
            fetch_ns: l(&c.fetch_ns),
        }
    }

    fn ctx_bytes(&self) -> u64 {
        let i = &self.inner;
        let per: usize = i.layers[0].src.iter().map(|s| matrix_bytes(s)).sum();
        (per * (i.layers.len() * i.n_slots + 2 * i.n_expert)) as u64
    }

    /// The fetch for a step on the host (what the op runs): `ids` routed, `out` the ids its
    /// matmuls read. For tests.
    pub fn fetch_host(&self, il: usize, ids: &[i32], n_tokens: usize) -> Result<Vec<i32>, String> {
        let ord = self.ord[il].ok_or("not a tier layer")?;
        let mode = if ids.len() <= self.inner.n_slots {
            Mode::Slots
        } else {
            Mode::Stage(ord % 2)
        };
        let mut out = vec![0; ids.len()];
        self.inner.fetch(ord, mode, ids, &[], &mut out, n_tokens)?;
        Ok(out)
    }

    /// The fetch of a piece on the host ([`Self::fetch_rows`]): rows `rows` of a step routing
    /// `ids` (`n_used` per row), in slot mode. For tests.
    pub fn fetch_host_rows(
        &self,
        il: usize,
        ids: &[i32],
        rows: std::ops::Range<usize>,
    ) -> Result<Vec<i32>, String> {
        let ord = self.ord[il].ok_or("not a tier layer")?;
        if rows.is_empty() || rows.len() * self.n_used > self.n_slots() {
            return Err(format!("rows {rows:?}: more pairs than slots"));
        }
        let (mine, later) = piece_ids(ids, self.n_used, rows.start, rows.len())?;
        let mut out = vec![0; mine.len()];
        self.inner
            .fetch(ord, Mode::Slots, mine, later, &mut out, rows.len())?;
        Ok(out)
    }
}

/// `n` rows split into `k` pieces of consecutive rows, as even as they go (the first `n % k`
/// one row longer): the rows of a chunked step's pieces ([`SsdExperts::pieces`]).
pub fn piece_rows(n: usize, k: usize) -> Vec<std::ops::Range<usize>> {
    let k = k.clamp(1, n.max(1));
    let (base, extra) = (n / k, n % k);
    let mut at = 0;
    (0..k)
        .map(|i| {
            let len = base + usize::from(i < extra);
            at += len;
            at - len..at
        })
        .collect()
}

/// The ids of rows `row0..row0 + rows` of a step's `ids` (`n_used` per row), and those of the
/// rows after them.
fn piece_ids(
    ids: &[i32],
    n_used: usize,
    row0: usize,
    rows: usize,
) -> Result<(&[i32], &[i32]), String> {
    let (a, b) = (row0 * n_used, (row0 + rows) * n_used);
    if b > ids.len() {
        return Err(format!("rows {row0}..{} of {} ids", row0 + rows, ids.len()));
    }
    Ok((&ids[a..b], &ids[b..]))
}

impl Drop for SsdExperts {
    fn drop(&mut self) {
        if let Ok(mut st) = self.inner.state.lock() {
            if let Some(a) = st.ahead.take() {
                a.group.cancel();
                let _ = a.group.wait();
            }
            for l in st.inflight.iter_mut().flat_map(std::mem::take) {
                self.inner.pool.cancel(&l.group);
                let _ = l.group.wait();
            }
            // Flushes the trace.
            st.trace = None;
        }
        REGISTRY.lock().unwrap()[self.id as usize] = None;
    }
}

impl Inner {
    /// The work of one fetch: makes every expert of `ids` readable where `mode` puts it and
    /// writes the ids the matmuls read into `out`. `later`: a chunked step's ids of the pieces
    /// after this one (slot mode), whose experts are the last this fetch evicts.
    fn fetch(
        &self,
        ord: usize,
        mode: Mode,
        ids: &[i32],
        later: &[i32],
        out: &mut [i32],
        n_tokens: usize,
    ) -> Result<(), String> {
        let mut distinct = Vec::new();
        let mut seen = vec![false; self.n_expert];
        for &e in ids {
            let e = usize::try_from(e)
                .ok()
                .filter(|&e| e < self.n_expert)
                .ok_or_else(|| format!("expert id {e} of {}", self.n_expert))?;
            if !seen[e] {
                seen[e] = true;
                distinct.push(e);
            }
        }
        let t0 = Instant::now();
        let mut st = self.state.lock().unwrap();
        if let Some(t) = st.trace.as_mut() {
            t.record(ord, matches!(mode, Mode::Stage(_)), n_tokens, ids);
        }
        if mode != Mode::Slots {
            // A big step copies cached experts out of the slots: every read ahead ends first.
            for o in 0..self.layers.len() {
                self.reap(&mut st, o, true);
            }
        }
        let layer = &self.layers[ord];
        match mode {
            Mode::Slots => {
                // Reads ahead only pay for prompts; a small step needs the readers now.
                if let Some(a) = st.ahead.take() {
                    a.group.cancel();
                    a.group.wait()?;
                    st.held = [None, None];
                }
                let claimed = self.claim(&mut st, ord, &seen);
                if st.inflight[ord].len() + distinct.len() > self.n_slots {
                    // Pinned slots could leave the step short of victims.
                    self.reap(&mut st, ord, true);
                }
                let plan = if later.is_empty() {
                    st.caches[ord].plan(&distinct)
                } else {
                    // Each expert's first row among the later pieces': the furthest go first.
                    let mut next = vec![u32::MAX; self.n_expert];
                    for (i, &e) in later.iter().enumerate() {
                        if let Some(n) = usize::try_from(e).ok().and_then(|e| next.get_mut(e)) {
                            *n = (*n).min((i / self.n_used) as u32);
                        }
                    }
                    st.caches[ord].plan_ahead(&distinct, &next)
                };
                let mut jobs = Vec::with_capacity(plan.loads.len() * 3);
                let mut bytes = 0;
                for &(e, s) in &plan.loads {
                    let (j, b) = self.reads(ord, e, s);
                    jobs.extend(j);
                    bytes += b;
                }
                // The step's own reads go in behind the claimed reads ahead, before any other.
                let t = Instant::now();
                let demand = self.pool.submit(jobs);
                demand.wait()?;
                let c = &self.counts;
                c.miss_ns
                    .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.wait_claimed(&mut st, ord, claimed)?;
                c.fetch_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                c.miss_bytes.fetch_add(bytes, Ordering::Relaxed);
                c.hits.fetch_add(plan.hits as u64, Ordering::Relaxed);
                c.misses
                    .fetch_add(plan.loads.len() as u64, Ordering::Relaxed);
                let cache = &st.caches[ord];
                for (o, &e) in out.iter_mut().zip(ids) {
                    *o = cache.slot(e as usize).expect("planned") as i32;
                }
                if ord == 0 {
                    c.tokens.fetch_add(n_tokens as u64, Ordering::Relaxed);
                    if c.steps.fetch_add(1, Ordering::Relaxed) % REPORT_EVERY == REPORT_EVERY - 1 {
                        self.report();
                    }
                }
            }
            Mode::Stage(b) => {
                // The experts already read into this buffer for this layer.
                let mut have = vec![false; self.n_expert];
                if let Some(a) = st.ahead.take() {
                    if a.ord == ord && a.buf == b {
                        a.group.wait()?;
                        for e in a.experts {
                            have[e] = true;
                        }
                    } else {
                        a.group.cancel();
                        a.group.wait()?;
                    }
                }
                st.held = [None, None];
                for k in 0..3 {
                    let scales = (0..self.n_expert).map(|e| layer.src[k].experts[e].scale2);
                    // SAFETY: the buffer's `n_expert` per-expert scales (host memory).
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            self.stage_scale[b][k] as *mut f32,
                            self.n_expert,
                        )
                    };
                    for (d, s) in dst.iter_mut().zip(scales) {
                        *d = s;
                    }
                }
                let cache = &st.caches[ord];
                let mut jobs = Vec::new();
                let mut read = Vec::new();
                for &e in &distinct {
                    for k in 0..3 {
                        let src = &layer.src[k];
                        let mb = matrix_bytes(src);
                        let dst = self.stage[b][k] + e * mb;
                        match cache.slot(e) {
                            Some(s) => jobs.push(Job::Copy {
                                src: layer.slots[k] + s * mb,
                                dst,
                                len: mb,
                            }),
                            None if have[e] => {}
                            None => jobs.push(Job::Read {
                                src: src.experts[e],
                                rows: src.rows,
                                k: src.k,
                                dst,
                            }),
                        }
                    }
                    if cache.slot(e).is_none() {
                        read.push(e);
                    }
                }
                let bytes: u64 = jobs
                    .iter()
                    .map(|j| match j {
                        Job::Read { src, .. } => src.weight.len + src.scale.len,
                        Job::Copy { .. } => 0,
                    })
                    .sum();
                self.pool.run(jobs)?;
                self.counts.stage_bytes.fetch_add(bytes, Ordering::Relaxed);
                // Slots no expert holds yet take copies (the first prompt fills the cache).
                let cache = &mut st.caches[ord];
                let mut fill = Vec::new();
                for e in read {
                    let Some(s) = cache.take_empty(e) else { break };
                    for k in 0..3 {
                        let src = &layer.src[k];
                        let mb = matrix_bytes(src);
                        fill.push(Job::Copy {
                            src: self.stage[b][k] + e * mb,
                            dst: layer.slots[k] + s * mb,
                            len: mb,
                        });
                        // SAFETY: as above.
                        unsafe {
                            *(layer.slot_scale[k] as *mut f32).add(s) = src.experts[e].scale2
                        };
                    }
                }
                self.pool.run(fill)?;
                out.copy_from_slice(ids);
                // Read the next layer's uncached experts into the other buffer while this
                // layer computes (a prompt step routes to nearly all of them). Its buffer was
                // last read by the layer before this one, whose step is done.
                let next = (ord + 1) % self.layers.len();
                let nb = next % 2;
                if nb != b && 2 * distinct.len() > self.n_expert {
                    let nl = &self.layers[next];
                    let cache = &st.caches[next];
                    let experts: Vec<usize> = (0..self.n_expert)
                        .filter(|&e| cache.slot(e).is_none())
                        .collect();
                    let mut jobs = Vec::with_capacity(experts.len() * 3);
                    let mut bytes = 0;
                    for &e in &experts {
                        for k in 0..3 {
                            let src = &nl.src[k];
                            bytes += src.experts[e].weight.len + src.experts[e].scale.len;
                            jobs.push(Job::Read {
                                src: src.experts[e],
                                rows: src.rows,
                                k: src.k,
                                dst: self.stage[nb][k] + e * matrix_bytes(src),
                            });
                        }
                    }
                    self.counts.stage_bytes.fetch_add(bytes, Ordering::Relaxed);
                    st.ahead = Some(Ahead {
                        ord: next,
                        buf: nb,
                        experts,
                        group: self.pool.submit(jobs),
                    });
                }
            }
        }
        Ok(())
    }

    /// Whether predictions by `p` are read ahead.
    fn prefetches(&self, p: trace::Predictor) -> bool {
        use trace::Predictor::{PostAttn, PostMoe};
        matches!(
            (self.prefetch, p),
            (SsdPrefetch::Router, PostAttn)
                | (SsdPrefetch::RouterPost, PostMoe)
                | (SsdPrefetch::RouterBoth, PostAttn | PostMoe)
        )
    }

    /// The jobs reading expert `e` of layer `ord` into slot `s`, and the bytes they read; sets
    /// the slot's per-expert scales.
    fn reads(&self, ord: usize, e: usize, s: usize) -> (Vec<Job>, u64) {
        let layer = &self.layers[ord];
        let mut jobs = Vec::with_capacity(3);
        let mut bytes = 0;
        for k in 0..3 {
            let src = &layer.src[k];
            let x = src.experts[e];
            bytes += x.weight.len + x.scale.len;
            jobs.push(Job::Read {
                src: x,
                rows: src.rows,
                k: src.k,
                dst: layer.slots[k] + s * matrix_bytes(src),
            });
            // SAFETY: slot `s` of this layer's per-slot scales (host memory); the layer's
            // matmuls run after its fetch, which waits for this read.
            unsafe { *(layer.slot_scale[k] as *mut f32).add(s) = x.scale2 };
        }
        (jobs, bytes)
    }

    /// The work of one prediction op: layer `ord`'s routing predicted by `p` (`ids`, each of
    /// `n_tokens` tokens' experts best first) goes into the trace, and with the prefetch on,
    /// each token's best `n_used` + margin (at most `ranks`) not in a slot are queued for reading into the
    /// slots the least recently used experts the prediction does not name hold: rank by rank
    /// (every token's best first), the routed count as [`io::Class::Ahead`] and the margin as
    /// [`io::Class::Extra`], so a step's own reads never wait behind them and the reads that
    /// matter most start first. Reads ahead of the layer an earlier prediction queued that this
    /// one does not name are cancelled. Nothing waits for the reads here: the layer's fetch
    /// does ([`Self::claim`]).
    fn predict(
        &self,
        ord: usize,
        p: trace::Predictor,
        ids: &[i32],
        n_tokens: usize,
    ) -> Result<(), String> {
        let mut st = self.state.lock().unwrap();
        if let Some(t) = st.trace.as_mut() {
            t.predict(ord, p, n_tokens, ids);
        }
        if !self.prefetches(p) || n_tokens == 0 {
            return Ok(());
        }
        let width = ids.len() / n_tokens;
        // Under router-both the cap is the early (post-attention) lookahead's: the later,
        // surer post-MoE one reads every rank.
        let ranks = match (self.prefetch, p) {
            (SsdPrefetch::RouterBoth, trace::Predictor::PostMoe) => usize::MAX,
            _ => self.ranks,
        };
        let top = (self.n_used + self.margin).min(ranks).min(width);
        let mut seen = vec![false; self.n_expert];
        let mut named = Vec::new();
        let mut core = 0;
        for r in 0..top {
            for t in 0..n_tokens {
                let e = ids[t * width + r];
                if let Some(e) = usize::try_from(e).ok().filter(|&e| e < self.n_expert)
                    && !seen[e]
                {
                    seen[e] = true;
                    named.push(e);
                }
            }
            if r < self.n_used {
                core = named.len();
            }
        }
        for &e in named.iter().skip(self.n_slots) {
            seen[e] = false;
        }
        named.truncate(self.n_slots);
        self.reap(&mut st, ord, false);
        for l in st.inflight[ord].iter_mut() {
            if l.used.is_none() && !seen[l.e] {
                self.pool.cancel(&l.group);
                l.used = Some(false);
            }
        }
        let loads = st.caches[ord].prefetch(&named, usize::MAX);
        let mut is_core = vec![false; self.n_expert];
        for &e in &named[..core.min(named.len())] {
            is_core[e] = true;
        }
        let mut bytes = 0;
        let n = loads.len();
        for (e, s) in loads {
            let (jobs, b) = self.reads(ord, e, s);
            bytes += b;
            st.caches[ord].pin(s, true);
            let class = if is_core[e] {
                io::Class::Ahead
            } else {
                io::Class::Extra
            };
            st.inflight[ord].push(Load {
                e,
                s,
                group: self.pool.submit_as(jobs, class),
                used: None,
            });
        }
        let c = &self.counts;
        c.prefetched.fetch_add(n as u64, Ordering::Relaxed);
        c.prefetch_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// The reads ahead of layer `ord`'s small step about to run (`need`: per expert, whether
    /// the step routes to it). Those it routes to move to the front of the readers' queue and
    /// are returned for the step to wait for ([`Self::wait_claimed`]); the others are
    /// cancelled: their queued pieces dropped at once, a piece already running ends in the
    /// background with its slot pinned until then. A read cancelled earlier that the step
    /// routes to after all is waited for here, and its slot emptied if it did not complete
    /// (the step reads it again).
    fn claim(&self, st: &mut State, ord: usize, need: &[bool]) -> Vec<Load> {
        let mut claimed = Vec::new();
        for mut l in std::mem::take(&mut st.inflight[ord]) {
            match (need[l.e], l.used) {
                (true, None) => {
                    self.pool.promote(&l.group);
                    l.used = Some(true);
                    claimed.push(l);
                }
                (true, Some(_)) => {
                    l.used = Some(true);
                    self.end(&mut st.caches[ord], l);
                }
                (false, None) => {
                    self.pool.cancel(&l.group);
                    l.used = Some(false);
                    st.inflight[ord].push(l);
                }
                (false, Some(_)) => st.inflight[ord].push(l),
            }
        }
        self.reap(st, ord, false);
        claimed
    }

    /// Waits for the reads ahead a step claimed ([`Self::claim`]) and unpins their slots. The
    /// step's plan counted them cached: one that failed is read again now, into its slot.
    fn wait_claimed(&self, st: &mut State, ord: usize, claimed: Vec<Load>) -> Result<(), String> {
        if claimed.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        let c = &self.counts;
        for l in claimed {
            let r = l.group.wait();
            st.caches[ord].pin(l.s, false);
            if r.is_ok() && !l.group.skipped() {
                c.prefetch_used.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if let Err(e) = r {
                tracing::debug!(error = e, "SSD expert tier: a read ahead failed");
            }
            c.prefetch_dropped.fetch_add(1, Ordering::Relaxed);
            let (jobs, bytes) = self.reads(ord, l.e, l.s);
            self.pool.run(jobs)?;
            c.miss_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
        c.prefetch_wait_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Ends a read ahead of layer cache `cache` once its jobs ended (it waits for them) and
    /// counts it; its slot is unpinned, or emptied if the read did not complete (cancelled
    /// before it ran, or failed). Whether it completed.
    fn end(&self, cache: &mut SlotCache, l: Load) -> bool {
        let r = l.group.wait();
        let complete = r.is_ok() && !l.group.skipped();
        let c = &self.counts;
        let n = if !complete {
            cache.drop_slot(l.s);
            &c.prefetch_dropped
        } else {
            cache.pin(l.s, false);
            if l.used == Some(true) {
                &c.prefetch_used
            } else {
                &c.prefetch_wasted
            }
        };
        n.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = r {
            tracing::debug!(error = e, "SSD expert tier: a read ahead failed");
        }
        complete
    }

    /// Ends layer `ord`'s cancelled reads ahead that have ended ([`Self::end`]); with
    /// `block`, all of them, cancelling the others and waiting.
    fn reap(&self, st: &mut State, ord: usize, block: bool) {
        if st.inflight[ord].is_empty() {
            return;
        }
        let loads = std::mem::take(&mut st.inflight[ord]);
        if block {
            for l in &loads {
                if l.used.is_none() {
                    self.pool.cancel(&l.group);
                }
            }
        }
        for mut l in loads {
            if block {
                l.used.get_or_insert(false);
            } else if l.used.is_none() || !l.group.done() {
                st.inflight[ord].push(l);
                continue;
            }
            self.end(&mut st.caches[ord], l);
        }
    }

    fn report(&self) {
        let c = &self.counts;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let (hits, misses) = (l(&c.hits), l(&c.misses));
        let tokens = l(&c.tokens).max(1);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let mib_per_token = |b: u64| b as f64 / tokens as f64 / (1u64 << 20) as f64;
        let ms_per_token = |ns: u64| ns as f64 / tokens as f64 / 1e6;
        if self.prefetch == SsdPrefetch::Off {
            tracing::info!(
                hit_rate = format!(
                    "{:.1}%",
                    100.0 * hits as f64 / (hits + misses).max(1) as f64
                ),
                miss_mib_per_token = mib_per_token(l(&c.miss_bytes)),
                read_gb_s = l(&c.miss_bytes) as f64 / l(&c.miss_ns).max(1) as f64,
                fetch_wait_ms_per_token = ms_per_token(l(&c.fetch_ns)),
                read_gib = gib(self.pool.bytes_read()),
                tokens,
                hits,
                misses,
                "SSD expert tier"
            );
            return;
        }
        tracing::info!(
            hit_rate = format!(
                "{:.1}%",
                100.0 * hits as f64 / (hits + misses).max(1) as f64
            ),
            miss_mib_per_token = mib_per_token(l(&c.miss_bytes)),
            prefetch_mib_per_token = mib_per_token(l(&c.prefetch_bytes)),
            prefetched = l(&c.prefetched),
            prefetch_used = l(&c.prefetch_used),
            prefetch_wasted = l(&c.prefetch_wasted),
            prefetch_dropped = l(&c.prefetch_dropped),
            fetch_wait_ms_per_token = ms_per_token(l(&c.fetch_ns)),
            miss_wait_ms_per_token = ms_per_token(l(&c.miss_ns)),
            prefetch_wait_ms_per_token = ms_per_token(l(&c.prefetch_wait_ns)),
            read_gib = gib(self.pool.bytes_read()),
            tokens,
            hits,
            misses,
            "SSD expert tier"
        );
    }
}

/// The routing trace `LEGION_SSD_EXPERT_TRACE` asks for (a debug hook, docs/debug-env.md).
fn open_trace(layers: &[usize], n_expert: usize, n_slots: usize) -> Option<trace::Writer> {
    let path = std::env::var("LEGION_SSD_EXPERT_TRACE").ok()?;
    let header = trace::Header {
        layers: layers.to_vec(),
        n_expert,
        n_slots,
    };
    match trace::Writer::create(std::path::Path::new(&path), &header) {
        Ok(w) => {
            tracing::info!(path, "SSD expert tier: tracing the routing");
            Some(w)
        }
        Err(e) => {
            tracing::warn!(path, error = %e, "SSD expert tier: cannot write the trace");
            None
        }
    }
}

/// The fetch op: `src[0]` the step's ids (`[n_used, n_tokens]` I32), out the ids the matmuls
/// read; params `[tier id, layer ordinal, mode]`, and for a piece of a chunked step
/// ([`SsdExperts::fetch_rows`]) its first row and rows (out `[n_used, rows]`; `src[1]` the
/// node it runs after).
pub static FETCH: ExternalOp = ExternalOp {
    name: "ssd_expert_fetch",
    cpu: fetch_cpu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::fetch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// # Safety
/// `dst` must be a node made by [`SsdExperts::fetch`], in host memory.
unsafe fn fetch_cpu(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    let inner = lookup(p[0]);
    let n = (dst.ne[0] * dst.ne[1]) as usize;
    // SAFETY: the op's I32 source (the step's ids, contiguous) and output (`n` ids), in host
    // memory.
    let (all, out) = unsafe {
        let s = &*dst.src[0];
        (
            std::slice::from_raw_parts(s.data as *const i32, (s.ne[0] * s.ne[1]) as usize),
            std::slice::from_raw_parts_mut(dst.data as *mut i32, n),
        )
    };
    if let Err(e) = run_fetch(&inner, &p, dst.ne[0] as usize, all, out) {
        fatal(&e);
    }
}

/// A fetch op's work ([`FETCH`], params `p`) on the step's ids `all` (`n_used` per row) into
/// `out`: the whole step, or the piece its params name.
fn run_fetch(
    inner: &Inner,
    p: &[i32],
    n_used: usize,
    all: &[i32],
    out: &mut [i32],
) -> Result<(), String> {
    let (ord, mode) = (p[1] as usize, Mode::from_param(p[2]));
    let rows = out.len() / n_used.max(1);
    if p[4] == 0 {
        return inner.fetch(ord, mode, all, &[], out, rows);
    }
    let (mine, later) = piece_ids(all, n_used, p[3] as usize, p[4] as usize)?;
    if mine.len() != out.len() {
        return Err(format!("a piece of {} ids into {}", mine.len(), out.len()));
    }
    inner.fetch(ord, mode, mine, later, out, rows)
}

/// The prediction op: `src[0]` a prediction's ids (`[width, n_tokens]` I32, best first),
/// `src[1]` (optional) the node it runs after; params `[tier id, layer ordinal, predictor]`.
pub static PREDICT: ExternalOp = ExternalOp {
    name: "ssd_expert_predict",
    cpu: predict_cpu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::predict),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// # Safety
/// `dst` must be a node made by [`SsdExperts::predict`], its ids in host memory.
unsafe fn predict_cpu(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    let inner = lookup(p[0]);
    // SAFETY: the op's I32 source, contiguous, in host memory.
    let (ids, n_tokens) = unsafe {
        let s = &*dst.src[0];
        let n = (s.ne[0] * s.ne[1]) as usize;
        (
            std::slice::from_raw_parts(s.data as *const i32, n),
            s.ne[1] as usize,
        )
    };
    let Some(pred) = trace::Predictor::from_code(p[2] as u8) else {
        return;
    };
    if let Err(e) = inner.predict(p[1] as usize, pred, ids, n_tokens) {
        fatal(&e);
    }
}

/// A read the model cannot do without: the step's output would be garbage.
fn fatal(e: &str) -> ! {
    tracing::error!(error = e, "SSD expert tier: a read failed");
    eprintln!("SSD expert tier: {e}");
    std::process::abort();
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};
    use std::sync::Arc;

    use super::trace::Predictor;
    use super::{Inner, external_params, fatal, lookup, run_fetch};
    use crate::ggml::{EXTERNAL_PARAMS, ffi};

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
    }

    /// `cudaMemcpyDefault`: the direction from the pointers (unified addressing).
    const DEFAULT: c_int = 4;

    struct Job {
        inner: Arc<Inner>,
        /// The op's params.
        p: [i32; EXTERNAL_PARAMS],
        /// Ids out (a fetch) or in (a prediction).
        n: usize,
        /// A fetch: the step's ids in, `n_used` per row.
        n_in: usize,
        n_used: usize,
        n_tokens: usize,
        /// A prediction's ([`super::PREDICT`]), not a fetch.
        predictor: Option<Predictor>,
    }

    unsafe extern "C" fn host(user: *mut c_void) {
        // SAFETY: the box `fetch` leaked for this call.
        let job = unsafe { Box::from_raw(user as *mut Job) };
        if let Some(p) = job.predictor {
            let at = job.inner.pxchg.expect("a prediction buffer");
            // SAFETY: the pinned prediction buffer, `pcap >= n` ids; the stream orders this
            // after the copy in.
            let ids = unsafe { std::slice::from_raw_parts(at as *const i32, job.n) };
            if let Err(e) = job.inner.predict(job.p[1] as usize, p, ids, job.n_tokens) {
                fatal(&e);
            }
            return;
        }
        let (ids_in, ids_out) = job.inner.xchg.expect("exchange buffers");
        // SAFETY: the pinned exchange buffers, `cap >= n_in >= n` ids each; the stream orders
        // this between the copy in and the copy out.
        let (ids, out) = unsafe {
            (
                std::slice::from_raw_parts(ids_in as *const i32, job.n_in),
                std::slice::from_raw_parts_mut(ids_out as *mut i32, job.n),
            )
        };
        if let Err(e) = run_fetch(&job.inner, &job.p, job.n_used, ids, out) {
            fatal(&e);
        }
    }

    /// Copies the ids to pinned memory, runs the fetch as a host function on the stream, and
    /// copies the result into the op's output.
    pub(super) unsafe fn fetch(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let p = external_params(dst);
        let inner = lookup(p[0]);
        let n = (dst.ne[0] * dst.ne[1]) as usize;
        // SAFETY: the op's source, made by `SsdExperts::fetch_op`: the step's ids (a piece's
        // fetch sees every row).
        let src = unsafe { &*dst.src[0] };
        let n_in = (src.ne[0] * src.ne[1]) as usize;
        let (ids_in, ids_out) = inner.xchg.ok_or("no exchange buffers")?;
        if n_in > inner.cap || n > n_in {
            return Err(format!("{n_in} ids in, {n} out, room for {}", inner.cap));
        }
        let job = Box::into_raw(Box::new(Job {
            inner: inner.clone(),
            p,
            n,
            n_in,
            n_used: dst.ne[0] as usize,
            n_tokens: dst.ne[1] as usize,
            predictor: None,
        }));
        // SAFETY: device and pinned buffers of `n_in` ids; the host function owns `job`.
        unsafe {
            let e = cudaMemcpyAsync(ids_in as *mut c_void, src.data, n_in * 4, DEFAULT, stream);
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

    /// Copies a prediction's ids to pinned memory and hands them to the tier in a host function
    /// on the stream (which only queues reads: the stream goes on at once).
    pub(super) unsafe fn predict(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let p = external_params(dst);
        let inner = lookup(p[0]);
        // SAFETY: the op's source, made by `SsdExperts::predict`.
        let src = unsafe { &*dst.src[0] };
        let n = (src.ne[0] * src.ne[1]) as usize;
        let at = inner.pxchg.ok_or("no prediction buffer")?;
        if n > inner.pcap {
            return Err(format!("{n} predicted ids, room for {}", inner.pcap));
        }
        let predictor =
            Predictor::from_code(p[2] as u8).ok_or_else(|| format!("predictor {}", p[2]))?;
        let job = Box::into_raw(Box::new(Job {
            inner: inner.clone(),
            p,
            n,
            n_in: n,
            n_used: src.ne[0] as usize,
            n_tokens: src.ne[1] as usize,
            predictor: Some(predictor),
        }));
        // SAFETY: a device buffer and the pinned one of `n` ids; the host function owns `job`.
        unsafe {
            let e = cudaMemcpyAsync(at as *mut c_void, src.data, n * 4, DEFAULT, stream);
            if e != 0 {
                drop(Box::from_raw(job));
                return Err(format!("predicted ids to host: CUDA error {e}"));
            }
            let e = cudaLaunchHostFunc(stream, host, job.cast());
            if e != 0 {
                drop(Box::from_raw(job));
                return Err(format!("host function: CUDA error {e}"));
            }
        }
        Ok(())
    }
}
