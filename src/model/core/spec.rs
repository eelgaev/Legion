//! Recurrent-state rollback for speculative verify steps, by replay (see `crate::spec` for the
//! design and the measurements behind it; adapted from Strata's verify/commit split, MIT,
//! Niko1221/Strata `src/core/verify.cpp`).
//!
//! A verify group (`Group::verify`) does not advance its sequences' DeltaNet state slots.
//! Instead every recurrent layer stores, per slot and fed token, what the state update needs:
//! the conv window (`[history | this step's conv inputs]`) and the delta rule's inputs (`q`,
//! `k` after the L2 norm, `v`, gate, beta). [`SpecPool::commit`] then advances each slot by
//! the tokens it keeps: the conv history becomes the matching window, and `gated_delta_net`
//! runs over the kept tokens' stored inputs from the unchanged state, which is the same
//! kernel on the same inputs as decoding them one by one, so the result is bit-identical.
//!
//! Nemotron-H's Mamba-2 layers store the same conv window and the selective scan's inputs in
//! the same buffers (`x` in `v`, `dt` in `g`, `B` in `q`, `C` in `k`; the shapes line up:
//! `d_state × n_group`, `head_dim × n_head`, `n_head`), and the commit runs `ssm_scan` over
//! the kept tokens instead.
//!
//! GLM-5.3-Flash's KDA layers run ops of their own: a verify step stores each token's conv
//! inputs (q, k and v before the convs), per-channel decay and beta, and leaves the conv and
//! delta-rule states as they were; the commit runs the same convs and scan over the kept
//! tokens' stored inputs ([`Arch::spec_replay`](super::Arch::spec_replay)), which advance the
//! states as decoding them one by one would.
//!
//! qwen4exp's per-layer embedding is rolled back the same way: its conv history becomes the
//! window `[history | this step's inputs]` at the kept length, and the n-gram token history
//! (host side) takes only the kept tokens.

use crate::ggml::{
    Backend, BufferType, ComputeAllocator, Context, GgmlError, Graph, KeptGraph, KeptParts,
    Launcher, Tensor, types,
};

/// Commit graphs kept per set of runs (`(slot, slots, keep)`), most recent last: a decode
/// batch commits the same few shapes over and over, so their graphs are built once.
pub type CommitGraphs = Vec<(Vec<(usize, usize, usize)>, KeptGraph)>;

/// Commit graphs kept per stage ([`CommitGraphs`]): with one sequence, one per kept count.
const COMMIT_KEPT: usize = 8;

/// [`COMMIT_KEPT`] of a pool of KDA layers (GLM, #725). Four sequences verifying one or two
/// drafts each keep their tokens in about thirty sets of runs, so eight kept graphs missed
/// one commit in ten or more (a rebuild of every GPU's graph on the main thread, ~3 ms a
/// step at C=4). A kept graph holds a few hundred KiB to ~2 MiB of intermediates.
const COMMIT_KEPT_KDA: usize = 32;

/// How many commit graphs a pool keeps: [`COMMIT_KEPT_KDA`] for KDA layers, else
/// [`COMMIT_KEPT`].
pub(super) fn commit_kept(kda: bool) -> usize {
    if kda { COMMIT_KEPT_KDA } else { COMMIT_KEPT }
}

mod pending;
mod window;

/// Whether gated DeltaNet pools made from now on defer each commit's delta rule into the next
/// verify step's (#836 part 14): each verify step's delta rule stores its tokens' inputs in
/// packed rows (part 39), the commit copies the conv windows, and the next verify step of the
/// slot replays the kept ones before its rows in the same pass over the state
/// ([`super::gdn::rows_replay`]). Anything else that reads
/// a slot's state first replays its deferred commit on its own ([`SpecPool::flush`]).
/// On by default since part 38 (the packed rows cost no memory beyond the default's).
static DEFER: AtomicBool = AtomicBool::new(true);

/// Sets whether pools made from now on defer their commits into the next verify step (`legion
/// serve --gdn-commit-in-verify`, `--no-gdn-commit-in-verify`).
pub fn set_commit_in_verify(on: bool) {
    DEFER.store(on, Ordering::Relaxed);
}

/// Whether deferring pools made from now on let their verify steps leave the states as they are
/// where the packed rows have room (#836, [`super::gdn::leaves`]): 0 the default (off: the same
/// bits, but 1-4% slower decode on GB10, where the kept tokens' second replay costs more than the
/// state write it saves), 1 on, 2 off.
static STATE_DEFER: AtomicU8 = AtomicU8::new(0);

/// Packed replay rows per slot beyond the verify width that such pools add, so that the kept
/// tokens of more rounds wait there (#836).
static STATE_DEFER_ROWS: AtomicUsize = AtomicUsize::new(DEFAULT_STATE_DEFER_ROWS);

/// [`STATE_DEFER_ROWS`] unless `--gdn-state-defer-rows` says otherwise.
pub const DEFAULT_STATE_DEFER_ROWS: usize = 8;

/// The most [`STATE_DEFER_ROWS`] a pool takes: the kernel's count words hold rows below 64 in
/// all.
pub const MAX_STATE_DEFER_ROWS: usize = 32;

/// Sets whether deferring pools made from now on let verify steps leave the states (`legion
/// serve --gdn-state-defer`, `--no-gdn-state-defer`; `None`: the default, off).
pub fn set_state_defer(on: Option<bool>) {
    STATE_DEFER.store(
        match on {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        },
        Ordering::Relaxed,
    );
}

/// Sets the packed rows per slot such pools add (`legion serve --gdn-state-defer-rows`; at most
/// [`MAX_STATE_DEFER_ROWS`]).
pub fn set_state_defer_rows(rows: usize) {
    STATE_DEFER_ROWS.store(rows.min(MAX_STATE_DEFER_ROWS), Ordering::Relaxed);
}

/// The fewest sequences a verify step of a leaving pool ([`STATE_DEFER`]) leaves the states in
/// (#858); narrower steps write them. 0 or 1: every step may leave them.
static STATE_DEFER_MIN_SEQS: AtomicUsize = AtomicUsize::new(0);

/// Sets the fewest sequences a verify step of pools made from now on leaves the states in
/// (`legion serve --gdn-state-defer-min-seqs`; 0: every step).
pub fn set_state_defer_min_seqs(n: usize) {
    STATE_DEFER_MIN_SEQS.store(n, Ordering::Relaxed);
}

/// The fewest sequences a verify step of a rank of tensor parallelism over machines has for
/// its commit to be deferred into the next verify step ([`DEFER`], #1176); 0, the default: such
/// ranks defer none.
static TP_DEFER_MIN_SEQS: AtomicUsize = AtomicUsize::new(0);

/// Sets the fewest sequences a verify step of the ranks of tensor parallelism over machines
/// made from now on has for its commit to be deferred (`legion serve
/// --tp-commit-in-verify-min-seqs`; 0: none).
pub fn set_tp_commit_in_verify_min_seqs(n: usize) {
    TP_DEFER_MIN_SEQS.store(n, Ordering::Relaxed);
}

/// What [`set_tp_commit_in_verify_min_seqs`] set.
pub fn tp_commit_in_verify_min_seqs() -> usize {
    TP_DEFER_MIN_SEQS.load(Ordering::Relaxed)
}

/// `commits` (`(slot, keep)`) as those of the slots whose last verify step stored its inputs in
/// the packed replay rows (`in_rows[slot]`), which a pool with both stores
/// ([`SpecPool::new_layers_wide`]) defers, and the others, which it runs in the commit.
fn split_by_store(
    in_rows: &[bool],
    commits: &[(usize, usize)],
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    commits
        .iter()
        .copied()
        .partition(|&(slot, _)| in_rows[slot])
}

/// Whether deferring pools made now let verify steps leave the states ([`STATE_DEFER`]).
fn state_defer() -> bool {
    STATE_DEFER.load(Ordering::Relaxed) == 1
}

/// The key of a deferring pool's commit graph in [`CommitGraphs`] (stash and windows).
const DEFER_KEY: [(usize, usize, usize); 1] = [(usize::MAX, 0, 0)];
/// The key of a deferring pool's flush graph in [`CommitGraphs`].
const FLUSH_KEY: [(usize, usize, usize); 1] = [(usize::MAX, 1, 0)];

/// Whether gated DeltaNet pools commit through one graph for every slot (#836): per layer one
/// delta-rule launch ([`super::gdn::commit_slots`]) and one conv window copy
/// ([`window::window`]) over all slots, each slot's kept count a graph input (0: left alone),
/// instead of a graph per set of runs with an op per run. The same kernels per slot, so the same
/// states.
static COMMIT_SLOTS: AtomicBool = AtomicBool::new(true);

/// Sets whether commits from now on use the one all-slot graph where it fits (A/B tests). On by
/// default; `legion serve --gdn-commit-runs` keeps the graphs per set of runs.
pub fn set_commit_slots(on: bool) {
    COMMIT_SLOTS.store(on, Ordering::Relaxed);
}

/// The key of the all-slot graph in a pool's [`CommitGraphs`] (a set of runs is never empty).
const SLOTS_KEY: Vec<(usize, usize, usize)> = Vec::new();

/// Host copies of the keep counts an all-slot commit uploads, used in turn: an upload is queued
/// on the backend's stream and reads its source when it runs, which is before the backend's next
/// synchronize; every verify step synchronizes, and fewer commits than this run between two.
const KEEP_RING: usize = 16;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use super::state::StatePool;
use super::{Hparams, Model, Recurrent};

/// What a commit hands [`super::Arch::spec_replay`] for one recurrent layer and one run of
/// `n` consecutive slots keeping `keep` tokens each ([`Recurrent::Kda`]).
#[derive(Clone, Copy)]
pub struct Replay<'a> {
    /// The kept tokens' stored conv inputs (q, k, v before the convs), F32 `[c, keep, n]`
    /// views (rows contiguous).
    pub x: [Tensor<'a>; 3],
    /// Their decay logs, F32 `[c, keep, n]` view.
    pub g: Tensor<'a>,
    /// Their betas, F32 `[n_v, keep, n]` view.
    pub beta: Tensor<'a>,
    /// The slots' conv states and delta-rule states ([`StatePool::conv_slots`],
    /// [`StatePool::ssm_slots`]), advanced in place.
    pub conv: Tensor<'a>,
    pub ssm: Tensor<'a>,
    pub keep: i64,
    /// GLM's parallel steps ([`SpecPool::set_shard`]): the GPU whose share of the heads the
    /// states hold, whose shards of the weights the replay reads; `None` for the whole layer.
    pub shard: Option<usize>,
}

/// `src` viewed as `[a, rows, n]` (strides `nb`, at `offset`), copied into `dst`, a contiguous
/// tensor of that shape: a commit's conv window.
///
/// With one slot, ggml's CUDA copy turns `[a, rows, 1]` into a 2D memcpy of `rows` rows of `a`
/// elements, which runs at ~10 GB/s for rows this short (12 µs for a DeltaNet layer's conv
/// window on GB10). Viewed as `[a, rows / 2, 2, n]` it runs as ggml's element-wise copy kernel
/// instead, in ~6 µs. The same bytes land either way.
fn window_copy<'a>(
    ctx: &'a Context,
    src: Tensor<'a>,
    [a, rows, n]: [i64; 3],
    [nb1, nb2]: [usize; 2],
    offset: usize,
    dst: Tensor<'a>,
) -> Tensor<'a> {
    if rows % 2 != 0 {
        return ctx.cpy(ctx.view_3d(src, [a, rows, n], [nb1, nb2], offset), dst);
    }
    let half = rows / 2;
    let src = ctx.view_4d(
        src,
        [a, half, 2, n],
        [nb1, nb1 * half as usize, nb2],
        offset,
    );
    ctx.cpy(src, ctx.reshape_4d(dst, [a, half, 2, n]))
}

/// Per recurrent layer, per state slot, room for `max_tokens` verify tokens' replay inputs.
pub struct SpecPool {
    ctx: Context,
    n_slots: usize,
    max_tokens: usize,
    layers: Vec<usize>,
    d_conv: i64,
    conv_dim: i64,
    head_k: i64,
    n_k: i64,
    head_v: i64,
    n_v: i64,
    /// The per-token conv ([`super::TokenConv`]): its layer, conv history and width.
    ple: Option<(usize, i64, i64)>,
    /// Per slot: the tokens the last verify step fed it (for the n-gram history).
    fed: std::sync::Mutex<Vec<Vec<u32>>>,
    /// Mamba-2 layers rather than gated DeltaNet ([`super::Recurrent`]).
    mamba: bool,
    /// KDA layers ([`Recurrent::Kda`]): no conv window; `q`, `k`, `v` hold the conv inputs
    /// and `g` one decay per channel.
    kda: bool,
    /// See [`Self::set_shard`].
    shard: Option<usize>,
    /// The all-slot commit graph ([`COMMIT_SLOTS`]) does not fit this pool or its backend (found
    /// once, by a build): commits keep the graphs per set of runs.
    slots_unfit: AtomicBool,
    /// [`KEEP_RING`] host keep-count arrays and the next one to use.
    keep_host: Mutex<(usize, Vec<Vec<i32>>)>,
    /// Commits are deferred into the next verify step ([`DEFER`]): the pool has the packed
    /// replay rows (`spec_r-{il}`) and the device's deferred counts (`spec_pending`).
    defer: bool,
    /// The layers' q, k, v, g and beta are views of the packed replay rows (a pool made to
    /// defer, #836 part 38), not tensors of their own.
    packed: bool,
    /// The deferred commits ([`DEFER`]).
    pending: Mutex<pending::Pending>,
    /// The deferred commit graph does not fit (found once, by a build): commits run as usual.
    defer_unfit: AtomicBool,
    /// A pool with both stores ([`Self::new_layers_wide`]): the fewest sequences a verify step
    /// has to replay and store in the packed rows. 0 for any other pool (every verify step of a
    /// deferring pool does).
    replay_min: usize,
    /// Per slot of a pool with both stores: its last verify step stored its inputs in the packed
    /// rows (its commit is deferred), not in the layers' own tensors (its commit runs at once).
    in_rows: Mutex<Vec<bool>>,
}

/// The stored tensors of one layer.
struct Layer<'a> {
    /// `[d_conv - 1 + max_tokens, conv_dim, slots]`: the conv input window (none for KDA).
    conv: Option<Tensor<'a>>,
    /// `[head_k * n_k, max_tokens, slots]`.
    q: Tensor<'a>,
    k: Tensor<'a>,
    /// `[head_v * n_v, max_tokens, slots]`.
    v: Tensor<'a>,
    /// `[n_v, max_tokens, slots]` (KDA: `g` `[head_v * n_v, ...]`).
    g: Tensor<'a>,
    beta: Tensor<'a>,
}

impl<'a> Layer<'a> {
    /// The conv window of a DeltaNet or Mamba-2 layer.
    fn conv(&self) -> Tensor<'a> {
        self.conv.expect("a KDA layer stores no conv window")
    }
}

impl SpecPool {
    pub fn new(
        hp: &Hparams,
        buft: BufferType,
        n_slots: usize,
        max_tokens: usize,
    ) -> Result<Self, GgmlError> {
        Self::new_layers(hp, buft, n_slots, max_tokens, 0..hp.n_layer, true)
    }

    /// A pool for the layers `layers` only (a pipeline stage's). `may_defer`: its commits may be
    /// deferred into the next verify step ([`DEFER`]; the executor runs them on one stage here).
    pub fn new_layers(
        hp: &Hparams,
        buft: BufferType,
        n_slots: usize,
        max_tokens: usize,
        stage: std::ops::Range<usize>,
        may_defer: bool,
    ) -> Result<Self, GgmlError> {
        Self::new_layers_wide(hp, buft, n_slots, max_tokens, stage, may_defer, 0)
    }

    /// [`Self::new_layers`] for a rank of tensor parallelism over machines (#1176). With
    /// `replay_min` above 0, a pool that would defer has both stores: the layers' own q, k, v, g
    /// and beta, which verify steps of fewer than `replay_min` sequences store and whose commits
    /// run at once (graphs and kernels of a pool that does not defer), and the packed replay
    /// rows, which wider verify steps replay and store ([`Self::replays`]) and whose commits are
    /// deferred. Each slot's commit follows its last verify step ([`Self::split_commits`]).
    pub fn new_layers_wide(
        hp: &Hparams,
        buft: BufferType,
        n_slots: usize,
        max_tokens: usize,
        stage: std::ops::Range<usize>,
        may_defer: bool,
        replay_min: usize,
    ) -> Result<Self, GgmlError> {
        let n_slots = n_slots.max(1);
        let max_tokens = max_tokens.max(1);
        let layers: Vec<usize> = stage
            .clone()
            .filter(|&il| il < hp.n_layer && hp.is_recurrent(il))
            .collect();
        let ctx = Context::new(7 * layers.len().max(1) + 3, 0);
        let conv_dim = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
        let n_v = hp.ssm_dt_rank;
        let head_v = hp.ssm_d_inner / n_v;
        let (t, s) = (max_tokens as i64, n_slots as i64);
        let kda = hp.arch.recurrent() == Recurrent::Kda;
        let defer = may_defer && Self::defers_layers(hp) && !layers.is_empty();
        // Both stores: the own tensors beside the packed rows.
        let both = defer && replay_min > 0;
        // Verify steps that may leave the states keep the waiting tokens of more rounds in the
        // packed rows (#836).
        let extra = if defer {
            Self::leave_rows(hp, max_tokens)
        } else {
            0
        };
        let leave = defer && Self::leaves_layers(hp);
        let rows = t + extra as i64;
        for &il in &layers {
            let new = |name: &str, ne: &[i64]| {
                ctx.new_tensor(types::F32, ne)
                    .named(&format!("spec_{name}-{il}"));
            };
            if kda {
                // The conv inputs, one decay per channel.
                for name in ["q", "k", "v", "g"] {
                    new(name, &[hp.ssm_d_inner, t, s]);
                }
                new("beta", &[n_v, t, s]);
                continue;
            }
            new("conv", &[hp.ssm_d_conv - 1 + t, conv_dim, s]);
            let a_qk = hp.ssm_d_state * hp.ssm_n_group;
            let widths = [a_qk, a_qk, head_v * n_v, n_v, n_v];
            if !defer || both {
                for (name, w) in ["q", "k", "v", "g", "beta"].into_iter().zip(widths) {
                    new(name, &[w, t, s]);
                }
            }
            if !defer {
                continue;
            }
            // Deferred commits keep the inputs packed per token only (#836 part 38), as the
            // verify steps' delta rule stores and replays them (`gdn::rows_replay`); q, k, v, g
            // and beta are views of those rows (what a commit that does not defer reads; the
            // verify steps do not store q there, as no commit's states depend on it).
            let row = super::gdn::replay_row(hp.ssm_n_group, n_v);
            let r = ctx
                .new_tensor(types::F32, &[row, rows, s])
                .named(&format!("spec_r-{il}"));
            if both {
                continue;
            }
            let nb = r.nb();
            let mut off = 0;
            for (name, w) in ["q", "k", "v", "g", "beta"].into_iter().zip(widths) {
                ctx.view_3d(r, [w, t, s], [nb[1], nb[2]], off * 4)
                    .named(&format!("spec_{name}-{il}"));
                off += w as usize;
            }
        }
        if defer {
            ctx.new_tensor(types::I32, &[s]).named("spec_pending");
            // The verify steps' delta-rule CTAs count themselves here (part 39); zeros between
            // launches. One set for every layer: the layers run one after another.
            ctx.new_tensor(types::I32, &[hp.ssm_n_group * s])
                .named("spec_arrive");
        }
        let ple = match hp.arch.token_conv(hp) {
            Some(c) if stage.contains(&c.layer) => {
                let (hist, width) = (c.history, c.width);
                ctx.new_tensor(types::F32, &[hist + t, width, s])
                    .named("spec_ple");
                Some((c.layer, hist, width))
            }
            _ => None,
        };
        ctx.alloc_tensors(buft)?;
        // Committing never reads rows a verify step did not write, but keep them finite.
        ctx.clear_buffers(0);
        // The verify step's delta rule reads them on a GPU only.
        let packed = defer && !both;
        let defer = defer
            && ctx
                .tensor("spec_pending")
                .is_some_and(|t| !t.is_host() && !t.is_meta());
        Ok(Self {
            ctx,
            n_slots,
            max_tokens,
            layers,
            d_conv: hp.ssm_d_conv,
            conv_dim,
            head_k: hp.ssm_d_state,
            n_k: hp.ssm_n_group,
            head_v,
            n_v,
            ple,
            fed: std::sync::Mutex::new(vec![Vec::new(); n_slots]),
            mamba: hp.arch.recurrent() == Recurrent::Mamba2,
            kda,
            shard: None,
            slots_unfit: AtomicBool::new(false),
            keep_host: Mutex::new((0, vec![vec![0; n_slots]; KEEP_RING])),
            defer,
            packed,
            pending: Mutex::new(
                pending::Pending::with_room(n_slots, rows as usize, leave && defer)
                    .leave_from(STATE_DEFER_MIN_SEQS.load(Ordering::Relaxed)),
            ),
            defer_unfit: AtomicBool::new(false),
            replay_min: if both && defer { replay_min } else { 0 },
            in_rows: Mutex::new(vec![false; n_slots]),
        })
    }

    /// Whether deferring pools of `hp`'s layers made now let their verify steps leave the
    /// states ([`STATE_DEFER`]).
    fn leaves_layers(hp: &Hparams) -> bool {
        Self::defers_layers(hp) && state_defer()
    }

    /// The packed rows per slot beyond the verify width (`max_tokens`) a deferring pool of
    /// `hp`'s layers made now adds ([`STATE_DEFER_ROWS`] where [`Self::leaves_layers`], within
    /// the 63 rows the verify step's op takes).
    fn leave_rows(hp: &Hparams, max_tokens: usize) -> usize {
        if Self::leaves_layers(hp) {
            STATE_DEFER_ROWS
                .load(Ordering::Relaxed)
                .min(63usize.saturating_sub(max_tokens))
        } else {
            0
        }
    }

    /// Whether pools of `hp`'s layers made now defer their commits ([`DEFER`]): switched on,
    /// gated DeltaNet layers of the delta-rule kernel's head size, its op and the all-slot
    /// commit on, a CUDA build.
    fn defers_layers(hp: &Hparams) -> bool {
        let n_v = hp.ssm_dt_rank.max(1);
        DEFER.load(Ordering::Relaxed)
            && cfg!(feature = "cuda")
            && hp.arch.recurrent() == Recurrent::DeltaNet
            && hp.ssm_d_state == 128
            && hp.ssm_d_inner / n_v == 128
            && super::gdn::on()
            && COMMIT_SLOTS.load(Ordering::Relaxed)
    }

    /// Makes the pool GPU `d`'s of GLM's parallel steps (built for its share of the heads,
    /// [`crate::model::glm5next::par::shard_hparams`]): its commits replay through that GPU's
    /// shards of the weights ([`Replay::shard`]).
    pub fn set_shard(&mut self, d: usize) {
        self.shard = Some(d);
    }

    /// Bytes per slot over all layers, for `max_tokens` tokens.
    pub fn slot_bytes(hp: &Hparams, max_tokens: usize) -> usize {
        Self::slot_bytes_layers(hp, max_tokens, 0..hp.n_layer)
    }

    /// Bytes per slot for the layers `layers`.
    pub fn slot_bytes_layers(
        hp: &Hparams,
        max_tokens: usize,
        layers: std::ops::Range<usize>,
    ) -> usize {
        let n_rec = layers
            .clone()
            .filter(|&il| il < hp.n_layer && hp.is_recurrent(il))
            .count() as i64;
        let conv_dim = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
        let t = max_tokens as i64;
        let per_token = 2 * hp.ssm_d_state * hp.ssm_n_group + hp.ssm_d_inner + 2 * hp.ssm_dt_rank;
        let ple = match hp.arch.token_conv(hp) {
            Some(c) if layers.contains(&c.layer) => (c.history + t) * c.width,
            _ => 0,
        };
        if hp.arch.recurrent() == Recurrent::Kda {
            // q, k, v and the decay per channel, beta per head; no conv window.
            return (4 * (n_rec * t * (4 * hp.ssm_d_inner + hp.ssm_dt_rank) + ple)) as usize;
        }
        // Deferred commits ([`DEFER`]): the inputs packed per token instead (the same floats, #836
        // part 38) and one count.
        let pending = i64::from(Self::defers_layers(hp) && n_rec > 0);
        // Verify steps that may leave the states: rows for more rounds' kept tokens (#836).
        let extra = Self::leave_rows(hp, max_tokens) as i64;
        // A pool with both stores ([`Self::new_layers_wide`]): the inputs once more.
        let both = i64::from(pending == 1 && tp_commit_in_verify_min_seqs() > 0);
        let rows = (t + extra + both * t) * per_token;
        (4 * (n_rec * ((hp.ssm_d_conv - 1 + t) * conv_dim + rows) + ple + pending)) as usize
    }

    /// Adds to `graph` the store of a verify group's per-layer-embedding conv window
    /// (`padded`: `[history + n_tok, hc × n_embd, n]`) for slots `slot..slot + n`.
    pub fn store_ple<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        slot: usize,
        padded: Tensor<'a>,
    ) {
        let t = self.ctx.tensor("spec_ple").expect("a per-layer embedding");
        let [len, width, n, _] = padded.ne();
        assert!(
            slot + n as usize <= self.n_slots && len <= t.ne()[0],
            "per-layer-embedding window of {n} x {len} at slot {slot} does not fit"
        );
        let nb = t.nb();
        let dst = ctx.view_3d(t, [len, width, n], [nb[1], nb[2]], slot * nb[2]);
        graph.expand(ctx.cpy(padded, dst));
    }

    /// Records the tokens a verify step fed `slot` (for the n-gram history the commit keeps).
    pub fn record_fed(&self, slot: usize, tokens: &[u32]) {
        let mut fed = self.fed.lock().unwrap();
        fed[slot].clear();
        fed[slot].extend_from_slice(tokens);
    }

    /// Verify tokens per sequence this pool has room for.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn layer(&self, il: usize) -> Layer<'_> {
        let t = |name: &str| {
            self.ctx
                .tensor(&format!("spec_{name}-{il}"))
                .unwrap_or_else(|| panic!("layer {il} has no replay buffers"))
        };
        Layer {
            conv: (!self.kda).then(|| t("conv")),
            q: t("q"),
            k: t("k"),
            v: t("v"),
            g: t("g"),
            beta: t("beta"),
        }
    }

    /// `t` (`[a, b, n_tok, n]` with `a * b` elements per token) as stored for slots
    /// `slot..slot + n`, tokens `0..n_tok`: a view of `[a, b, n_tok, n]` into the pool tensor
    /// `pool` (`[a * b, max_tokens, slots]`).
    fn token_view<'a>(
        ctx: &'a Context,
        pool: Tensor<'a>,
        ab: [i64; 2],
        slot: usize,
        n_tok: i64,
        n: i64,
    ) -> Tensor<'a> {
        let nb = pool.nb();
        let es = pool.element_size();
        ctx.view_4d(
            pool,
            [ab[0], ab[1], n_tok, n],
            [ab[0] as usize * es, nb[1], nb[2]],
            slot * nb[2],
        )
    }

    /// Adds to `graph` the stores of one verify group's layer-`il` replay inputs, for slots
    /// `slot..slot + n` and `n_tok` tokens each: `conv_in` `[d_conv - 1 + n_tok, conv_dim, n]`,
    /// `q`, `k` `[head_k, n_k, n_tok, n]`, `v` `[head_v, n_v, n_tok, n]`, `g`, `beta`
    /// `[1, n_v, n_tok, n]`.
    #[allow(clippy::too_many_arguments)]
    pub fn store<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        slot: usize,
        conv_in: Tensor<'a>,
        [q, k, v, g, beta]: [Tensor<'a>; 5],
    ) {
        let [_, _, n_tok, n] = q.ne();
        self.store_window(ctx, graph, il, slot, conv_in);
        for (src, dst) in [q, k, v, g, beta]
            .into_iter()
            .zip(self.token_views(ctx, il, slot, n_tok, n))
        {
            graph.expand(ctx.cpy(src, dst));
        }
    }

    /// Adds to `graph` the store of one verify group's layer-`il` conv window `conv_in`
    /// (`[d_conv - 1 + n_tok, conv_dim, n]`) for slots `slot..slot + n`.
    pub fn store_window<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        slot: usize,
        conv_in: Tensor<'a>,
    ) {
        let [_, _, n, _] = conv_in.ne();
        let n_tok = conv_in.ne()[0] - (self.d_conv - 1);
        self.check(slot, n_tok, n);
        let l = self.layer(il);
        let nb = l.conv().nb();
        let conv = ctx.view_3d(
            l.conv(),
            [self.d_conv - 1 + n_tok, self.conv_dim, n],
            [nb[1], nb[2]],
            slot * nb[2],
        );
        graph.expand(ctx.cpy(conv_in, conv));
    }

    /// Where one verify group's layer-`il` q, k, v, g and beta go, for slots `slot..slot + n`
    /// and `n_tok` tokens each: views `[a, b, n_tok, n]` of the pool (see [`Self::store`]).
    pub fn token_views<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n_tok: i64,
        n: i64,
    ) -> [Tensor<'a>; 5] {
        self.check(slot, n_tok, n);
        let l = self.layer(il);
        let (hk, nk, hv, nv) = (self.head_k, self.n_k, self.head_v, self.n_v);
        [
            (l.q, [hk, nk]),
            (l.k, [hk, nk]),
            (l.v, [hv, nv]),
            (l.g, [1, nv]),
            (l.beta, [1, nv]),
        ]
        .map(|(pool, ab)| Self::token_view(ctx, pool, ab, slot, n_tok, n))
    }

    fn check(&self, slot: usize, n_tok: i64, n: i64) {
        assert!(
            slot + n as usize <= self.n_slots && n_tok as usize <= self.max_tokens,
            "verify of {n} x {n_tok} tokens at slot {slot} does not fit the replay pool"
        );
    }

    /// Where one verify group's KDA layer-`il` inputs go ([`Recurrent::Kda`]), for slots
    /// `slot..slot + n` and `n_tok` tokens each: the conv inputs q, k, v and the decay `g`
    /// (`[c, n_tok, n]` views), beta (`[n_v, n_tok, n]`). The verify step's ops write them.
    pub fn kda_views<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n_tok: i64,
        n: i64,
    ) -> [Tensor<'a>; 5] {
        assert!(self.kda, "KDA views of a pool of other recurrent layers");
        self.check(slot, n_tok, n);
        let l = self.layer(il);
        [l.q, l.k, l.v, l.g, l.beta].map(|t| Self::slots_view(ctx, t, slot, n_tok, n))
    }

    /// `[w, n_tok, n]` of the pool tensor `t` (`[w, max_tokens, slots]`) at slot `slot`.
    fn slots_view<'a>(
        ctx: &'a Context,
        t: Tensor<'a>,
        slot: usize,
        n_tok: i64,
        n: i64,
    ) -> Tensor<'a> {
        let nb = t.nb();
        ctx.view_3d(t, [t.ne()[0], n_tok, n], [nb[1], nb[2]], slot * nb[2])
    }

    /// Adds to `graph` the stores of one verify group's Mamba-2 layer-`il` replay inputs, for
    /// slots `slot..slot + n` and `n_tok` tokens each: `conv_in` `[d_conv - 1 + n_tok,
    /// conv_dim, n]`, `x` `[head_dim, n_head, n_tok, n]`, `dt` `[n_head, n_tok, n]` (bias
    /// added, before softplus), `b`, `c` `[d_state, n_group, n_tok, n]`.
    pub fn store_mamba<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        slot: usize,
        conv_in: Tensor<'a>,
        [x, dt, b, c]: [Tensor<'a>; 4],
    ) {
        let [_, _, n_tok, n] = x.ne();
        assert!(
            slot + n as usize <= self.n_slots && n_tok as usize <= self.max_tokens,
            "verify of {n} x {n_tok} tokens at slot {slot} does not fit the replay pool"
        );
        let l = self.layer(il);
        let nb = l.conv().nb();
        let conv = ctx.view_3d(
            l.conv(),
            [self.d_conv - 1 + n_tok, self.conv_dim, n],
            [nb[1], nb[2]],
            slot * nb[2],
        );
        graph.expand(ctx.cpy(conv_in, conv));
        let (hk, nk, hv, nv) = (self.head_k, self.n_k, self.head_v, self.n_v);
        for (src, pool, ab) in [
            (b, l.q, [hk, nk]),
            (c, l.k, [hk, nk]),
            (x, l.v, [hv, nv]),
            (dt, l.g, [1, nv]),
        ] {
            let dst = Self::token_view(ctx, pool, ab, slot, n_tok, n);
            graph.expand(ctx.cpy(src, dst));
        }
    }

    /// Advances the states of `rec` by the tokens each slot keeps of the last verify step:
    /// `(slot, keep)` with `1 <= keep <=` the tokens that step fed. Runs as one graph on
    /// `backend`, reusing `alloc` for its intermediates. `model`: its weights a commit reads (a
    /// Mamba-2 layer's `A`; a KDA layer's convs, [`super::Arch::spec_replay`]).
    ///
    /// The graph is only started: the caller synchronizes `backend` before the states are read
    /// (so a layer split's stages commit at the same time, as Strata-AC922's async commit does,
    /// eelgaev/Strata-AC922 d775be3, MIT).
    /// # Safety
    /// The graphs kept in `kept` read `rec`, this pool and `model`'s weights: the caller drops
    /// them before any of those.
    pub unsafe fn commit(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
        commits: &[(usize, usize)],
        model: &Model,
    ) -> Result<(), GgmlError> {
        if commits.is_empty() {
            return Ok(());
        }
        if let Some(i) = self.slots_graph(backend, kept, rec)? {
            self.upload_keep(backend, &kept[i].1, commits);
            let res = backend.compute_kept_async(&kept[i].1);
            if res.is_err() {
                backend.synchronize();
            }
            res?;
            self.push_fed(rec, commits);
            return Ok(());
        }
        let runs = self.commit_runs(commits);
        if let Some(i) = kept.iter().position(|(r, _)| *r == runs) {
            let g = kept.remove(i);
            let res = backend.compute_kept_async(&g.1);
            kept.push(g);
            if res.is_err() {
                backend.synchronize();
            }
            res?;
        } else {
            self.build_commit(backend, kept, rec, runs, model)?;
        }
        self.push_fed(rec, commits);
        Ok(())
    }

    /// [`Self::commit`] whose kept graph another thread starts: the bookkeeping now, and the
    /// graph's [`Launcher`] when a kept one fits (its [`Launcher::start`] queues it). A graph
    /// built now is started now, and `None` returned.
    ///
    /// # Safety
    /// [`Self::commit`]'s, and [`Backend::launcher`]'s until the launcher has started.
    pub unsafe fn commit_launcher(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
        commits: &[(usize, usize)],
        model: &Model,
    ) -> Result<Option<Launcher>, GgmlError> {
        if commits.is_empty() {
            return Ok(None);
        }
        if let Some(i) = self.slots_graph(backend, kept, rec)? {
            self.upload_keep(backend, &kept[i].1, commits);
            // SAFETY: the caller's.
            let l = unsafe { backend.launcher(&kept[i].1) }?;
            self.push_fed(rec, commits);
            return Ok(Some(l));
        }
        let runs = self.commit_runs(commits);
        let launcher = if let Some(i) = kept.iter().position(|(r, _)| *r == runs) {
            let g = kept.remove(i);
            // SAFETY: the caller's.
            let l = unsafe { backend.launcher(&g.1) };
            kept.push(g);
            Some(l?)
        } else {
            self.build_commit(backend, kept, rec, runs, model)?;
            None
        };
        self.push_fed(rec, commits);
        Ok(launcher)
    }

    /// Whether [`Self::commit`] of `commits` would reuse a graph in `kept` (not build one).
    pub fn is_kept(&self, kept: &CommitGraphs, commits: &[(usize, usize)]) -> bool {
        commits.is_empty() || {
            let runs = self.commit_runs(commits);
            kept.iter()
                .any(|(r, _)| *r == runs || r.is_empty() || r[..] == DEFER_KEY)
        }
    }

    /// Whether this pool's commits may use the all-slot graph ([`COMMIT_SLOTS`]): gated DeltaNet
    /// layers (not Mamba-2 or KDA), switched on, not found unfit before.
    fn slots_wanted(&self) -> bool {
        COMMIT_SLOTS.load(Ordering::Relaxed)
            && cfg!(feature = "cuda")
            && !self.mamba
            && !self.kda
            && !self.slots_unfit.load(Ordering::Relaxed)
    }

    /// The index in `kept` of the all-slot commit graph, built (not run) if there is none yet;
    /// `None` when commits keep the graphs per set of runs: [`Self::slots_wanted`] does not hold,
    /// or this build finds a layer the ops do not take or a backend that cannot queue the keep
    /// counts' upload (CPU, tensor parallel meta backends), which it then remembers.
    fn slots_graph(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
    ) -> Result<Option<usize>, GgmlError> {
        if !self.slots_wanted() {
            return Ok(None);
        }
        if let Some(i) = kept.iter().position(|(r, _)| r.is_empty()) {
            return Ok(Some(i));
        }
        if backend.is_meta() {
            self.slots_unfit.store(true, Ordering::Relaxed);
            return Ok(None);
        }
        let Some(g) = self.build_slots(backend, rec, false)? else {
            tracing::info!("replay commits: per-run graphs (the all-slot graph does not fit)");
            self.slots_unfit.store(true, Ordering::Relaxed);
            return Ok(None);
        };
        tracing::info!("replay commits: one all-slot graph");
        // Graphs per set of runs built before (the switch flipped) stay until dropped.
        kept.push((SLOTS_KEY, g));
        Ok(Some(kept.len() - 1))
    }

    /// The all-slot commit graph: per layer the conv window copy and the delta rule over every
    /// slot, and the per-layer embedding's window, all reading one I32 keep count per slot (the
    /// kept graph's tensor 0). `None` if a layer does not fit the ops or the backend cannot queue
    /// the keep counts' upload. `defer` ([`DEFER`]): the kept tokens' inputs are stashed for the
    /// next verify step instead of the delta rule running now.
    fn build_slots(
        &self,
        backend: &mut Backend,
        rec: &StatePool,
        defer: bool,
    ) -> Result<Option<KeptGraph>, GgmlError> {
        // The views count too (as in `build_commit`, one run's worth).
        let n_nodes = (self.layers.len() + 1) * 24 + 16;
        let ctx = Context::new(n_nodes, n_nodes);
        let mut graph = ctx.new_graph(n_nodes);
        let (hk, nk, hv, nv) = (self.head_k, self.n_k, self.head_v, self.n_v);
        let (n, t) = (self.n_slots, self.max_tokens as i64);
        // An output too, so the allocator never places anything over it.
        let keep = ctx
            .new_tensor(types::I32, &[n as i64])
            .as_input()
            .as_output();
        for &il in &self.layers {
            let l = self.layer(il);
            let conv = rec.conv_slots(&ctx, il, 0, n);
            let window_src = l.conv();
            if !window::fits(window_src, conv, keep) {
                return Ok(None);
            }
            graph.expand(window::window(&ctx, window_src, conv, keep));
            if defer {
                // The kept tokens' inputs stay in the packed replay rows, where the verify step
                // stored them, for the next verify step's replay (#836 part 38).
                continue;
            }
            let view = |pool, ab| Self::token_view(&ctx, pool, ab, 0, t, n as i64);
            let q = view(l.q, [hk, nk]);
            let k = view(l.k, [hk, nk]);
            let v = view(l.v, [hv, nv]);
            let g = view(l.g, [1, nv]);
            let beta = view(l.beta, [1, nv]);
            let state = rec.ssm_slots(&ctx, il, 0, n);
            if !super::gdn::fits(q, k, v, g, beta, state) {
                return Ok(None);
            }
            graph.expand(super::gdn::commit_slots(
                &ctx, q, k, v, g, beta, state, keep,
            ));
        }
        if let Some((il, _, _)) = self.ple {
            let t = self.ctx.tensor("spec_ple").expect("a per-layer embedding");
            let dst = rec.ple_slots(&ctx, il, 0, n);
            if !window::fits(t, dst, keep) {
                return Ok(None);
            }
            graph.expand(window::window(&ctx, t, dst, keep));
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))?;
        if !backend.queues_copies_of(keep) {
            return Ok(None);
        }
        let parts = KeptParts::new(&graph, &[keep]);
        // SAFETY: the graph reads `rec`, this pool and nothing else; the caller keeps them
        // alive while it keeps the graph ([`Self::commit`]'s contract).
        Ok(Some(unsafe { KeptGraph::new(ctx, parts) }))
    }

    /// Queues the upload of `commits`' keep counts (0 for the other slots) into the all-slot
    /// graph `g`'s input, on `backend`'s stream ahead of the graph.
    fn upload_keep(&self, backend: &mut Backend, g: &KeptGraph, commits: &[(usize, usize)]) {
        let mut ring = self.keep_host.lock().unwrap();
        let i = ring.0;
        ring.0 = (i + 1) % KEEP_RING;
        let host = &mut ring.1[i];
        host.fill(0);
        for &(slot, keep) in commits {
            assert!(
                keep >= 1 && keep <= self.max_tokens && slot < self.n_slots,
                "commit of {keep} tokens at slot {slot}"
            );
            assert!(host[slot] == 0, "slot {slot} committed twice");
            host[slot] = keep as i32;
        }
        let t = g.tensor(0);
        // SAFETY: a GPU backend's own buffer (`build_slots` checked); the ring entry is not
        // written again before this backend synchronizes ([`KEEP_RING`]); `n_slots` values fit.
        unsafe { backend.set_async(t, host.as_ptr().cast(), 4 * host.len()) };
    }

    /// Whether this pool defers its commits into the next verify step ([`DEFER`]): its verify
    /// steps' delta rule replays the deferred commits ([`Self::replay_views`]), whether or not a
    /// commit was deferred.
    pub fn defers(&self) -> bool {
        self.defer
    }

    /// Whether the layers' q, k, v, g and beta are strided views of the packed replay rows (a
    /// pool made to defer, #836 part 38): stored by copies, never by the fused delta-rule prep.
    pub fn packed(&self) -> bool {
        self.packed
    }

    /// Whether a verify step of `n` sequences replays its slots' deferred commits and stores
    /// its inputs in the packed replay rows ([`Self::replay_views`]): any of a pool that defers,
    /// those of at least its floor where it has both stores ([`Self::new_layers_wide`]).
    pub fn replays(&self, n: usize) -> bool {
        self.defer && n >= self.replay_min
    }

    /// Before a verify step over `slots` that does not replay ([`Self::replays`]): their
    /// commits run at once, from the layers' own tensors.
    pub fn plain_verify(&self, slots: std::ops::Range<usize>) {
        if self.replay_min > 0 {
            self.in_rows.lock().unwrap()[slots].fill(false);
        }
    }

    /// For a pool with both stores ([`Self::new_layers_wide`]): `commits` as those to defer
    /// ([`Self::commit_deferred`]) and those to run at once ([`Self::commit`]), by each slot's
    /// last verify step. `None` for any other pool, whose commits all go one way.
    pub fn split_commits(
        &self,
        commits: &[(usize, usize)],
    ) -> Option<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
        (self.replay_min > 0).then(|| split_by_store(&self.in_rows.lock().unwrap(), commits))
    }

    /// Layer `il`'s packed replay rows ([`DEFER`]), `[row, max_tokens, slots]`.
    fn replay_tensor(&self, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("spec_r-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no replay rows"))
    }

    /// The device's deferred counts ([`DEFER`]), I32 `[slots]`.
    fn pending_tensor(&self) -> Tensor<'_> {
        self.ctx
            .tensor("spec_pending")
            .expect("a pool that defers its commits")
    }

    /// A verify group's view of the deferred commits of slots `slot..slot + n` in layer `il`
    /// ([`super::gdn::rows_replay`]): their counts (I32 `[n]`), packed rows (`[row,
    /// max_tokens, n]`), and the pool's counters with the group's first slot (the tensor itself,
    /// not a view: a view would be one more graph node). `None` when the pool does not defer, or
    /// not for a step this narrow ([`Self::replays`]).
    pub fn replay_views<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n: usize,
    ) -> Option<(Tensor<'a>, Tensor<'a>, (Tensor<'a>, usize))> {
        if !self.replays(n) {
            return None;
        }
        self.check(slot, 1, n as i64);
        let p = self.pending_tensor();
        let keep = ctx.view_1d(p, n as i64, slot * p.nb()[0]);
        let r = self.replay_tensor(il);
        let nb = r.nb();
        let rows = ctx.view_3d(
            r,
            [r.ne()[0], r.ne()[1], n as i64],
            [nb[1], nb[2]],
            slot * nb[2],
        );
        let arrive = self
            .ctx
            .tensor("spec_arrive")
            .expect("a pool that defers its commits");
        Some((keep, rows, (arrive, slot)))
    }

    /// Queues the upload of `counts` (one per slot) into the device's deferred counts.
    fn upload_counts(&self, backend: &mut Backend, counts: &[i32]) {
        let mut ring = self.keep_host.lock().unwrap();
        let i = ring.0;
        ring.0 = (i + 1) % KEEP_RING;
        let host = &mut ring.1[i];
        host.copy_from_slice(counts);
        // SAFETY: the pool's own GPU buffer (`defer` holds); the ring entry is not written again
        // before this backend synchronizes ([`KEEP_RING`]); `n_slots` values fit.
        unsafe { backend.set_async(self.pending_tensor(), host.as_ptr().cast(), 4 * host.len()) };
    }

    /// [`Self::commit`] deferred into the next verify step ([`DEFER`]): the conv windows now
    /// and the kept tokens' inputs stashed; each slot's delta rule waits for its next verify step
    /// or a [`Self::flush`]. `false` (nothing done) when the pool does not defer or its graph
    /// does not fit: the caller commits as usual.
    ///
    /// # Safety
    /// [`Self::commit`]'s.
    pub unsafe fn commit_deferred(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
        commits: &[(usize, usize)],
    ) -> Result<bool, GgmlError> {
        if commits.is_empty() {
            return Ok(true);
        }
        if !self.defer
            || !self.slots_wanted()
            || backend.is_meta()
            || self.defer_unfit.load(Ordering::Relaxed)
        {
            return Ok(false);
        }
        if !self.pending.lock().unwrap().can_commit(commits) {
            // A slot committed again without its verify step in between: not a verify round.
            // SAFETY: the caller's.
            unsafe { self.flush(backend, kept, rec) }?;
        }
        let i = match kept.iter().position(|(r, _)| r[..] == DEFER_KEY) {
            Some(i) => i,
            None => match self.build_slots(backend, rec, true)? {
                Some(g) => {
                    kept.push((DEFER_KEY.to_vec(), g));
                    kept.len() - 1
                }
                None => {
                    self.defer_unfit.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
            },
        };
        self.upload_keep(backend, &kept[i].1, commits);
        let counts = self.pending.lock().unwrap().commit(commits);
        self.upload_counts(backend, &counts);
        let res = backend.compute_kept_async(&kept[i].1);
        if res.is_err() {
            backend.synchronize();
        }
        res?;
        self.push_fed(rec, commits);
        Ok(true)
    }

    /// Whether a slot of `slots` waits for a deferred commit ([`DEFER`]).
    pub fn defers_slots(&self, slots: std::ops::Range<usize>) -> bool {
        self.defer && self.pending.lock().unwrap().touches(slots)
    }

    /// Whether any slot waits for a deferred commit ([`DEFER`]).
    pub fn defers_any(&self) -> bool {
        self.defer && self.pending.lock().unwrap().any()
    }

    /// Before a verify step of `n_tok` tokens over slots `slots`, queued ahead of it: makes the
    /// device's deferred counts of those slots the host's, and counts their commits as done (the
    /// step replays them), or as still waiting where the step leaves the state
    /// ([`super::gdn::leaves`]).
    pub fn before_verify(
        &self,
        backend: &mut Backend,
        slots: std::ops::Range<usize>,
        n_tok: usize,
    ) {
        if !self.defer {
            return;
        }
        if self.replay_min > 0 {
            self.in_rows.lock().unwrap()[slots.clone()].fill(true);
        }
        let up = self.pending.lock().unwrap().verify(slots, n_tok);
        if let Some(counts) = up {
            self.upload_counts(backend, &counts);
        }
    }

    /// Slot `slot`'s state was reset: its deferred commit is void.
    pub fn drop_deferred(&self, slot: usize) {
        if self.defer {
            self.pending.lock().unwrap().drop_slot(slot);
        }
    }

    /// Replays every deferred commit on its own (one launch per layer for every slot, the bits
    /// of [`super::gdn::commit_slots`]), queued on `backend`: before anything but a verify step
    /// reads or writes a slot's state.
    ///
    /// # Safety
    /// [`Self::commit`]'s.
    pub unsafe fn flush(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
    ) -> Result<(), GgmlError> {
        if !self.defer {
            return Ok(());
        }
        let Some(counts) = self.pending.lock().unwrap().flush() else {
            return Ok(());
        };
        let i = match kept.iter().position(|(r, _)| r[..] == FLUSH_KEY) {
            Some(i) => i,
            None => {
                let g = self.build_flush(backend, rec)?;
                kept.push((FLUSH_KEY.to_vec(), g));
                kept.len() - 1
            }
        };
        self.upload_counts(backend, &counts);
        let res = backend.compute_kept_async(&kept[i].1);
        if res.is_err() {
            backend.synchronize();
        }
        res
    }

    /// The flush graph ([`Self::flush`]): per layer the deferred delta rule of every slot from
    /// its packed replay rows, reading the device's deferred counts.
    fn build_flush(&self, backend: &mut Backend, rec: &StatePool) -> Result<KeptGraph, GgmlError> {
        // Per layer the state's view, five views of the replay rows and the op.
        let n_nodes = 12 * self.layers.len() + 32;
        let ctx = Context::new(n_nodes, n_nodes);
        let mut graph = ctx.new_graph(n_nodes);
        let n = self.n_slots;
        let keep = self.pending_tensor();
        for &il in &self.layers {
            let state = rec.ssm_slots(&ctx, il, 0, n);
            let r = self.replay_tensor(il);
            graph.expand(super::gdn::commit_replay(&ctx, r, state, keep, self.n_k));
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))?;
        let parts = KeptParts::new(&graph, &[]);
        // SAFETY: the graph reads `rec` and this pool; the caller keeps them alive while it
        // keeps the graph ([`Self::commit`]'s contract).
        Ok(unsafe { KeptGraph::new(ctx, parts) })
    }

    /// `commits` as runs of consecutive slots keeping the same count, `(slot, n, keep)`: a
    /// run shares one op per layer.
    fn commit_runs(&self, commits: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
        let mut sorted = commits.to_vec();
        sorted.sort_unstable();
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for (slot, keep) in sorted {
            assert!(
                keep >= 1 && keep <= self.max_tokens && slot < self.n_slots,
                "commit of {keep} tokens at slot {slot}"
            );
            match runs.last_mut() {
                Some((s, n, k)) if *s + *n == slot && *k == keep => *n += 1,
                Some((s, n, _)) if *s + *n > slot => panic!("slot {slot} committed twice"),
                _ => runs.push((slot, 1, keep)),
            }
        }
        runs
    }

    /// The per-layer embedding's token history: the kept tokens of each commit.
    fn push_fed(&self, rec: &StatePool, commits: &[(usize, usize)]) {
        if self.ple.is_some() {
            let fed = self.fed.lock().unwrap();
            for &(slot, keep) in commits {
                assert!(
                    fed[slot].len() >= keep,
                    "slot {slot} kept tokens it was not fed"
                );
                rec.push_tokens(slot, &fed[slot][..keep]);
            }
        }
    }

    /// Builds, keeps and launches the commit graph of `runs` ([`Self::commit`]).
    fn build_commit(
        &self,
        backend: &mut Backend,
        kept: &mut CommitGraphs,
        rec: &StatePool,
        runs: Vec<(usize, usize, usize)>,
        model: &Model,
    ) -> Result<(), GgmlError> {
        // Up to 17 nodes per layer and run (a run of several slots copies g and beta to
        // contiguous memory), and the per-layer embedding's few.
        let n_nodes = (self.layers.len() + 1) * runs.len() * 24 + 16;
        let ctx = Context::new(n_nodes, n_nodes);
        let mut graph = ctx.new_graph(n_nodes);
        let (hk, nk, hv, nv) = (self.head_k, self.n_k, self.head_v, self.n_v);
        // Mamba-2: the selective scan reads each run's slots through ids 0..n. Set once, after
        // the allocation, and the graph is kept and run again: an output too, so the allocator
        // never places a node over it once its last scan ran (#836).
        let most = runs.iter().map(|r| r.1).max().unwrap_or(1);
        let ids = self.mamba.then(|| {
            ctx.new_tensor(types::I32, &[most as i64])
                .as_input()
                .as_output()
        });
        for &il in &self.layers {
            let l = self.layer(il);
            for &(slot, n, keep) in &runs {
                let (n_i, keep_i) = (n as i64, keep as i64);
                if self.kda {
                    let v = |t| Self::slots_view(&ctx, t, slot, keep_i, n_i);
                    let r = Replay {
                        x: [v(l.q), v(l.k), v(l.v)],
                        g: v(l.g),
                        beta: v(l.beta),
                        conv: rec.conv_slots(&ctx, il, slot, n),
                        ssm: rec.ssm_slots(&ctx, il, slot, n),
                        keep: keep_i,
                        shard: self.shard,
                    };
                    let nodes = model
                        .hp
                        .arch
                        .spec_replay(model, &ctx, il, r)
                        .map_err(|e| GgmlError::External(e.to_string()))?;
                    for t in nodes {
                        graph.expand(t);
                    }
                    continue;
                }
                // Conv history: the last d_conv - 1 inputs after the kept tokens.
                let nb = l.conv().nb();
                graph.expand(window_copy(
                    &ctx,
                    l.conv(),
                    [self.d_conv - 1, self.conv_dim, n_i],
                    [nb[1], nb[2]],
                    slot * nb[2] + keep * l.conv().element_size(),
                    rec.conv_slots(&ctx, il, slot, n),
                ));

                let view = |pool, ab| Self::token_view(&ctx, pool, ab, slot, keep_i, n_i);
                if let Some(ids) = ids {
                    let x = view(l.v, [hv, nv]);
                    let b = view(l.q, [hk, nk]);
                    let c = view(l.k, [hk, nk]);
                    let dt = ctx.reshape_3d(ctx.cont(view(l.g, [1, nv])), nv, keep_i, n_i);
                    let state = rec.ssm_slots(&ctx, il, slot, n);
                    let ids = ctx.view_2d(ids, [n_i, 1], ids.nb()[1], 0);
                    let a = model
                        .w(&format!("blk.{il}.ssm_a"))
                        .expect("a Mamba-2 layer's A");
                    let out = ctx.ssm_scan(state, x, dt, a, b, c, ids, 1);
                    let es = out.element_size();
                    let sn = state.ne();
                    let final_state = ctx.view_4d(
                        out,
                        sn,
                        [
                            sn[0] as usize * es,
                            (sn[0] * sn[1]) as usize * es,
                            (sn[0] * sn[1] * sn[2]) as usize * es,
                        ],
                        (hv * nv * keep_i * n_i) as usize * es,
                    );
                    graph.expand(ctx.cpy(final_state, state));
                    continue;
                }
                let contiguous = |t: Tensor<'_>| t.is_contiguous();
                let (mut q, mut k, mut v) = (
                    view(l.q, [hk, nk]),
                    view(l.k, [hk, nk]),
                    view(l.v, [hv, nv]),
                );
                if self.packed {
                    // A deferring pool's packed rows (stored by `gdn::rows_replay`; q left at
                    // zeros, which no state reads) when its commit does not defer after all.
                    (q, k, v) = (ctx.cont(q), ctx.cont(k), ctx.cont(v));
                }
                let mut g = view(l.g, [1, nv]);
                let mut beta = view(l.beta, [1, nv]);
                if !contiguous(g) {
                    g = ctx.cont(g);
                    beta = ctx.cont(beta);
                }
                let state = rec.ssm_slots(&ctx, il, slot, n);
                if super::gdn::fits(q, k, v, g, beta, state) {
                    // The final states straight into the slots, no rows (nothing reads them).
                    graph.expand(super::gdn::commit(&ctx, q, k, v, g, beta, state));
                    continue;
                }
                let out = ctx.gated_delta_net(q, k, v, g, beta, state, 1);
                let es = out.element_size();
                let final_state = ctx.view_4d(
                    out,
                    state.ne(),
                    [
                        hv as usize * es,
                        (hv * hv) as usize * es,
                        (hv * hv * nv) as usize * es,
                    ],
                    (hv * nv * keep_i * n_i) as usize * es,
                );
                graph.expand(ctx.cpy(final_state, state));
            }
        }
        if let Some((il, hist, width)) = self.ple {
            let t = self.ctx.tensor("spec_ple").expect("a per-layer embedding");
            let nb = t.nb();
            for &(slot, n, keep) in &runs {
                graph.expand(window_copy(
                    &ctx,
                    t,
                    [hist, width, n as i64],
                    [nb[1], nb[2]],
                    slot * nb[2] + keep * t.element_size(),
                    rec.ple_slots(&ctx, il, slot, n),
                ));
            }
        }
        // The oldest kept graph's allocator, once there are enough of them.
        let allocator = (kept.len() >= commit_kept(self.kda))
            .then(|| kept.remove(0).1.into_allocator())
            .flatten()
            .unwrap_or_else(|| ComputeAllocator::new(backend.buffer_type()));
        ctx.alloc_graph(&graph, allocator)?;
        if let Some(ids) = ids {
            let v: Vec<i32> = (0..most as i32).collect();
            ids.set_i32(&v);
        }
        let parts = KeptParts::new(&graph, &ids.into_iter().collect::<Vec<_>>());
        // SAFETY: the graph reads `rec`, this pool and the weights, which the caller keeps
        // alive while it keeps the graph (this function's contract).
        let g = unsafe { KeptGraph::new(ctx, parts) };
        let res = backend.compute_kept_async(&g);
        kept.push((runs, g));
        if res.is_err() {
            backend.synchronize();
        }
        res
    }
}

#[cfg(test)]
mod tests;
