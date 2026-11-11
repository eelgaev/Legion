//! The gated delta rule of a gated DeltaNet layer for the few tokens of an MTP verify window or
//! a replay commit, as an external op ([`crate::ggml::ExternalOp`], kernel in `src/cuda/gdn.cu`)
//! that produces only the output its caller uses:
//!
//! - [`rows`] (a verify step): the tokens' attention rows. ggml's `gated_delta_net` also writes
//!   every sequence's final state (64 KiB per head) to its output's tail, which a verify step
//!   never reads: its states advance in the commit.
//! - [`commit`] (a replay commit): the final states, written in place over the state slots.
//!   ggml's op computes the rows too and puts the states in its tail, and a copy then moves them
//!   into the slots.
//!
//! The CUDA kernel is ggml-cuda's autoregressive kernel per state column, operation for
//! operation, so both outputs are ggml's bit for bit (`gdn/tests.rs` compares them for 1 to 9
//! tokens); its token loop is unrolled for 1..8 tokens with the next token's inputs loaded
//! ahead (the exact-T kernels of upstream Strata #783, PR-b, by stuchapin909, MIT, via
//! Strata-AC922). On by default; `legion serve --gdn-ggml` keeps ggml's op.
//!
//! - [`decode`] (a one-token step, #836): ggml's whole output (the rows, then the final states
//!   in its tail, which a decode step places over the state slots). The arithmetic is the same
//!   per column; the kernel moves the state between global and shared memory in 16-byte vectors
//!   instead of ggml's 4-byte loads and stores, which is what a decode step's time is spent on.
//!   On by default on compute capability 9.0+ ([`set_gpu`]); `legion serve --gdn-decode-ggml`
//!   keeps ggml's op.
//! - [`rows_replay`] (a verify step that runs the deferred commit first) can read its tokens' k,
//!   q, v, g and beta from shared memory on compute capability 9.0+, filled 8 tokens at a time by
//!   the CTA (fewer registers, 5 CTAs per SM), with each lane's arithmetic unchanged (the same
//!   bits): `legion serve --gdn-replay-smem` (opt-in; ~3% faster kernel, flat end to end).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static GDN: ExternalOp = ExternalOp {
    name: "gdn",
    cpu: cpu_gdn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gdn),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The head size the kernel is written for (ggml's S_v, every qwen35 / qwen4exp model's).
const S: i64 = 128;

/// The op's modes (its first param).
const MODE_ROWS: i32 = 1;
const MODE_STATE: i32 = 2;
/// With [`MODE_STATE`]: the final states go to the output's tail (ggml's layout) instead of over
/// the input states.
const MODE_TAIL: i32 = 4;
/// With [`MODE_STATE`] alone: sequence `s` advances by its `keep[s]` tokens (a seventh source,
/// I32 [`count_word`]s: from the word's start token on), 0 leaving it alone ([`commit_slots`]).
const MODE_KEEP: i32 = 8;
/// With [`MODE_ROWS`] alone: before its rows, sequence `s` advances by the `keep[s]` tokens of a
/// deferred commit (seventh source, I32 [`count_word`]s), read from packed replay rows (eighth
/// source), and that state goes over the slot (unless the step [`leaves`] it); then the step's
/// own inputs go over those rows (after the kept ones where it leaves the state; the ninth source
/// holds the CTAs' counters, I32, from the second param on), for the next step ([`rows_replay`]).
const MODE_REPLAY: i32 = 16;
/// With [`MODE_REPLAY`]: a verify step's MoE row map (I32 `[n_tokens * n_seqs]`,
/// [`super::pad_rows::map`]) marks its padding tokens, whose rows are zeros and whose inputs are
/// not stored ([`rows_replay`]). ggml's custom ops take at most nine sources, so the map is the
/// ninth and the counters' address (already at the group's first counter) goes in the third and
/// fourth params; the counters are a pool's own buffer, at a fixed address.
const MODE_PAD: i32 = 32;

/// Whether graphs built from now on use [`rows`] and [`commit`].
static GDN_ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on use [`rows`] and [`commit`] where they fit (A/B
/// tests). On by default (ggml's op bit for bit); `legion serve --gdn-ggml` turns it off.
pub fn set_gdn(on: bool) {
    GDN_ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on use [`rows`] and [`commit`] where they fit.
pub(super) fn on() -> bool {
    GDN_ON.load(Ordering::Relaxed)
}

/// Whether one-token groups use [`decode`] (switched on and a GPU it is chosen for).
static DECODE_ON: AtomicBool = AtomicBool::new(true);
static DECODE_GPU: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on use [`decode`] where it fits (A/B tests). On by
/// default (ggml's op bit for bit); `legion serve --gdn-decode-ggml` turns it off.
pub fn set_gdn_decode(on: bool) {
    DECODE_ON.store(on, Ordering::Relaxed);
}

/// Whether [`rows_replay`] stages its tokens' inputs through shared memory on a GPU [`gpu_for`]
/// chose (the same bits as the per-lane kernel).
static REPLAY_STAGED: AtomicBool = AtomicBool::new(false);

/// Sets whether launches from now on stage [`rows_replay`]'s inputs (A/B tests). Off by default;
/// `legion serve --gdn-replay-smem` turns it on (compute capability 9.0+ only).
pub fn set_gdn_replay_staged(on: bool) {
    REPLAY_STAGED.store(on, Ordering::Relaxed);
}

/// Whether [`rows_replay`] skips a verify step's padding tokens where it is given their row map.
static PAD_SKIP: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on pass verify steps' row maps to [`rows_replay`], which
/// then skips their padding tokens (#836: zero rows, no delta rule, no stores; the real tokens'
/// rows and stores are the same bits). On by default; `legion serve --no-gdn-pad-skip` turns it
/// off.
pub fn set_gdn_pad_skip(on: bool) {
    PAD_SKIP.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on pass verify steps' row maps to [`rows_replay`].
pub(super) fn pad_skip() -> bool {
    PAD_SKIP.load(Ordering::Relaxed)
}

/// Whether [`rows_replay`] takes `pad` as the row map of `v`'s tokens (`v`: F32 `[S, H,
/// n_tokens, n_seqs]`): I32 `[n_tokens * n_seqs]`, contiguous, in device memory.
pub(super) fn pad_fits(pad: Tensor<'_>, v: Tensor<'_>) -> bool {
    let vn = v.ne();
    pad.ty() == types::I32
        && pad.is_contiguous()
        && pad.elements() as i64 == vn[2] * vn[3]
        && !pad.is_host()
        && !pad.is_meta()
}

/// The engine's GPU: [`decode`] (and, when asked for, the staged [`rows_replay`] kernel) are
/// used on compute capability 9.0+ (`cc_major`), where they were measured (GB10, #836); other GPUs
/// keep ggml's op and the per-lane replay kernel.
pub fn set_gpu(cc_major: Option<i32>) {
    DECODE_GPU.store(gpu_takes(cc_major), Ordering::Relaxed);
}

/// Whether a GPU of compute capability major `cc_major` takes [`decode`] ([`set_gpu`]).
pub(super) fn gpu_takes(cc_major: Option<i32>) -> bool {
    cc_major.is_some_and(|m| m >= 9)
}

/// Whether the GPU of device `dev` (the state's, or a launch's) takes [`decode`] and the staged
/// replay, asked per device (#1201: a box may hold GPUs of different architectures); the
/// engine's answer ([`set_gpu`]) where `dev` is unknown.
fn gpu_for(dev: Option<usize>) -> bool {
    gpu_for_by(
        dev,
        DECODE_GPU.load(Ordering::Relaxed),
        crate::hw::per_device::cc,
    )
}

/// [`gpu_for`] with the engine's answer `engine` and the compute capabilities `cc_of` gives.
pub(super) fn gpu_for_by(
    dev: Option<usize>,
    engine: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, |cc| gpu_takes(cc.map(|c| c.0)), cc_of)
}

/// Whether [`decode`] takes ggml's `gated_delta_net(q, k, v, g, beta, state, 1)`: as [`fits`],
/// for one token per sequence, the state on a GPU it is chosen for ([`gpu_for`]).
pub(super) fn decode_fits(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    DECODE_ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && v.ne()[2] == 1
        && [q, k, v, g, beta, state]
            .iter()
            .all(|t| !t.is_host() && !t.is_meta())
        && shapes_fit(q, k, v, g, beta, state)
        && gpu_for(state.cuda_device())
}

/// Whether the op takes ggml's `gated_delta_net(q, k, v, g, beta, state, 1)`: on a GPU only (the
/// CPU graph keeps ggml's op; no tensor-parallel meta buffers), switched on, heads of 128, a scalar gate (not KDA) and the
/// layouts ggml-cuda asserts (rows contiguous, q and k of the same strides, g, beta and the
/// state contiguous), in device memory.
pub(super) fn fits(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    GDN_ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && [q, k, v, g, beta, state]
            .iter()
            .all(|t| !t.is_host() && !t.is_meta())
        && shapes_fit(q, k, v, g, beta, state)
}

/// [`fits`] without the switch, the build and the memory: the shapes and layouts the kernel
/// takes.
fn shapes_fit(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    let (qn, vn) = (q.ne(), v.ne());
    let all = [q, k, v, g, beta, state];
    all.iter().all(|t| t.ty() == types::F32)
        && qn[0] == S
        && vn[0] == S
        && k.ne() == qn
        && k.nb() == q.nb()
        && [q, v].iter().all(|t| t.nb()[0] == 4)
        && qn[2] == vn[2]
        // ggml-cuda's chunked (MMA) paths start at 64 tokens.
        && vn[2] < 64
        && qn[3] > 0
        && vn[3] % qn[3] == 0
        && vn[1] % qn[1] == 0
        && g.ne() == [1, vn[1], vn[2], vn[3]]
        && beta.ne() == g.ne()
        && g.is_contiguous()
        && beta.is_contiguous()
        && state.is_contiguous()
        && state.elements() as i64 == S * S * vn[1] * vn[3]
}

fn node<'a>(ctx: &'a Context, mode: i32, ne: [i64; 4], args: [Tensor<'a>; 6]) -> Tensor<'a> {
    let [q, k, v, g, beta, state] = args;
    assert!(shapes_fit(q, k, v, g, beta, state), "gdn: shapes");
    ctx.external_with(&GDN, types::F32, ne, &args, &[mode])
}

/// ggml's `gated_delta_net(q, k, v, g, beta, state, 1)`'s token rows (F32 `[S * H, n_tokens *
/// n_seqs]`, ggml's first rows), without the final states.
pub(super) fn rows<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
) -> Tensor<'a> {
    let [_, h, nt, ns] = v.ne();
    node(
        ctx,
        MODE_ROWS,
        [S * h, nt * ns, 1, 1],
        [q, k, v, g, beta, state],
    )
}

/// ggml's `gated_delta_net(q, k, v, g, beta, state, 1)`'s final states, written over `state`
/// in place (each state is read before it is written, by the same thread). The returned node
/// (one float, never written) is what the graph expands.
pub(super) fn commit<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
) -> Tensor<'a> {
    node(ctx, MODE_STATE, [1, 1, 1, 1], [q, k, v, g, beta, state])
}

/// [`commit`] of every sequence at once, each over its own number of tokens: sequence `s`
/// advances by its first `keep[s]` tokens of `q`, `k`, `v`, `g`, `beta` (`keep`: I32
/// `[n_seqs]`, a graph input; 0 leaves the state alone; `v.ne()[2]` is only the most there
/// are). Per sequence the same kernel code and bits as [`commit`] of its kept tokens alone, so
/// a replay commit's graph is one launch per layer whatever the slots keep (#836).
#[allow(clippy::too_many_arguments)]
pub(super) fn commit_slots<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
    keep: Tensor<'a>,
) -> Tensor<'a> {
    assert!(shapes_fit(q, k, v, g, beta, state), "gdn: shapes");
    assert!(
        keep.ty() == types::I32 && keep.elements() as i64 == v.ne()[3] && keep.is_contiguous(),
        "gdn: one keep count per sequence"
    );
    ctx.external_with(
        &GDN,
        types::F32,
        [1, 1, 1, 1],
        &[q, k, v, g, beta, state, keep],
        &[MODE_STATE | MODE_KEEP],
    )
}

/// Floats per token of a deferred commit's packed replay rows (#836 part 14): `[q | k | v | g
/// | beta]` of `n_k` q / k heads and `n_v` value heads of [`S`].
pub(super) fn replay_row(n_k: i64, n_v: i64) -> i64 {
    2 * S * n_k + S * n_v + 2 * n_v
}

/// A deferred commit's count word as [`rows_replay`] and [`commit_replay`] read it (#836):
/// `keep` kept tokens of the packed replay rows from row `start` on, and with `leave` the verify
/// step may leave the slot's state as it is ([`leaves`]). A plain count is the word of start 0
/// without `leave`.
pub(super) fn count_word(keep: usize, start: usize, leave: bool) -> i32 {
    assert!(keep < 256 && start < 256, "count word of {keep} at {start}");
    keep as i32 | (start as i32) << 8 | i32::from(leave) << 16
}

/// [`count_word`]'s parts: `(keep, start, leave)`.
fn count_parts(w: i32) -> (usize, usize, bool) {
    (
        (w & 0xff) as usize,
        ((w >> 8) & 0xff) as usize,
        w & (1 << 16) != 0,
    )
}

/// Whether a [`rows_replay`] step of `n_tok` tokens leaves the slot's state as it is (#836):
/// its count word allows it and the packed rows (`room` per slot) hold the step's tokens after
/// the kept ones. It then advances the state by the kept tokens in registers only, computes its
/// rows from that, and stores its own tokens after the kept ones instead of over them, so the
/// next replay advances the unchanged state by both runs, token by token as before: the same
/// bits, and a state write (64 KiB per head and sequence) fewer. `src/cuda/gdn.cu`'s
/// `gdn_leaves` is the same rule.
pub(super) fn leaves(w: i32, n_tok: usize, room: usize) -> bool {
    let (keep, start, leave) = count_parts(w);
    leave && start + keep + n_tok <= room
}

/// Whether [`rows_replay`] takes these: [`fits`], one q / k sequence per value sequence, `keep`
/// I32 `[n_seqs]`, `replay` F32 `[replay_row, max_t, n_seqs]` with contiguous rows and room for
/// the step's tokens, `arrive` I32 with `n_k` counters per sequence from the group's first
/// slot on, in device memory.
#[allow(clippy::too_many_arguments)]
pub(super) fn replay_fits(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
    keep: Tensor<'_>,
    replay: Tensor<'_>,
    arrive: (Tensor<'_>, usize),
) -> bool {
    fits(q, k, v, g, beta, state)
        && replay_shapes(q, v, keep, replay, arrive)
        && replay.ne()[1] < 64
        && [replay, arrive.0]
            .iter()
            .all(|t| !t.is_host() && !t.is_meta())
}

/// The shapes [`rows_replay`] asserts, past [`shapes_fit`].
fn replay_shapes(
    q: Tensor<'_>,
    v: Tensor<'_>,
    keep: Tensor<'_>,
    replay: Tensor<'_>,
    (arrive, slot): (Tensor<'_>, usize),
) -> bool {
    let (qn, vn, rn) = (q.ne(), v.ne(), replay.ne());
    qn[3] == vn[3]
        && keep.ty() == types::I32
        && keep.is_contiguous()
        && keep.elements() as i64 == vn[3]
        && replay.ty() == types::F32
        && replay.nb()[0] == 4
        && rn == [replay_row(qn[1], vn[1]), rn[1], vn[3], 1]
        && rn[1] >= vn[2]
        && arrive.ty() == types::I32
        && arrive.is_contiguous()
        && arrive.elements() as i64 >= qn[1] * (slot as i64 + vn[3])
        && qn[1] * slot as i64 <= i32::MAX as i64
}

/// [`rows`] of a verify step that first runs the last commit (#836 part 14): sequence `s`'s
/// state advances by its `keep[s]` tokens of `replay` (a deferred commit's packed rows, see
/// [`replay_row`]; a [`count_word`], 0: none), goes over its slot (unless the step [`leaves`]
/// it, #836: then the slot keeps its state and the kept tokens wait for the next replay), and
/// this step's rows start from it. Per sequence the same kernel code and bits as
/// [`commit_slots`] and then [`rows`], with one pass over the state instead of two.
///
/// Then the step's own tokens go over `replay` for the next step (part 39; part 38 stored them
/// with an op of its own after this one), from row 0 on, or after the kept ones where the step
/// leaves the state: their k, v, g and beta, as copies. q is not stored, as
/// no commit reads it (its delta rule computes no rows; ggml's, where it runs, drops them).
/// `arrive` is a pool's I32 counters (zeros) and the group's first slot, whose `n_k` counters
/// per sequence on are where the kernel's CTAs count themselves to find the last reader of the
/// values they share; they hold zeros again after the op. The tensor itself and an offset, not
/// a view: a view would be one more graph node.
///
/// With `pad` (a verify step's MoE row map, [`super::pad_rows::map`]; see [`pad_fits`]), a
/// sequence's tokens from its first padding one on are skipped (#836): their rows are zeros and
/// their inputs are not stored over `replay`. Nothing reads either: a padding row's output is
/// never sampled, and a commit keeps only accepted tokens, which are real ones. The real tokens'
/// rows and stores are the same instructions on the same values, so the same bits.
#[allow(clippy::too_many_arguments)]
pub(super) fn rows_replay<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
    keep: Tensor<'a>,
    replay: Tensor<'a>,
    arrive: (Tensor<'a>, usize),
    pad: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        shapes_fit(q, k, v, g, beta, state) && replay_shapes(q, v, keep, replay, arrive),
        "gdn: replay shapes"
    );
    let [_, h, nt, ns] = v.ne();
    let first = (q.ne()[1] * arrive.1 as i64) as i32;
    let mut srcs = vec![q, k, v, g, beta, state, keep, replay];
    let mut params = vec![MODE_ROWS | MODE_REPLAY, first];
    match pad {
        Some(pad) => {
            assert!(
                pad.ty() == types::I32 && pad.is_contiguous() && pad.elements() as i64 == nt * ns,
                "gdn: one row map entry per token"
            );
            // SAFETY: a pool's allocated tensor (`replay_fits`: not host, not meta); the offset
            // stays within it (`replay_shapes`).
            let at = unsafe { ((*arrive.0.raw()).data as *mut i32).add(first as usize) } as u64;
            srcs.push(pad);
            params[0] |= MODE_PAD;
            params.extend([at as u32 as i32, (at >> 32) as u32 as i32]);
        }
        None => srcs.push(arrive.0),
    }
    ctx.external_with(&GDN, types::F32, [S * h, nt * ns, 1, 1], &srcs, &params)
}

/// A [`MODE_PAD`] node's real tokens of sequence `seq` (`n_tokens` each): up to its first
/// padding token of the row map (token 0 is always real); all of them without the mode.
///
/// SAFETY: `dst` is a node made by [`rows_replay`], its sources in host memory.
unsafe fn live_tokens(dst: &ffi::ggml_tensor, mode: i32, seq: usize, n_tokens: usize) -> usize {
    if mode & MODE_PAD == 0 {
        return n_tokens;
    }
    // SAFETY: the ninth source is the I32 row map of `n_tokens` per sequence (`rows_replay`).
    unsafe {
        let map = (*dst.src[8]).data as *const i32;
        let first = seq * n_tokens;
        (1..n_tokens)
            .find(|&t| *map.add(first + t) != (first + t) as i32)
            .unwrap_or(n_tokens)
    }
}

/// The deferred commit of every sequence on its own (no verify step after it): sequence `s`
/// advances by its `keep[s]` tokens of `replay` (a [`count_word`]: from its start row on;
/// packed rows, [`replay_row`] floats per token of `n_k` q / k heads; F32 `[row, max_t,
/// n_seqs]`), in place, 0 leaving it alone. The
/// same kernel and bits as [`commit_slots`] over the tokens' unpacked inputs.
pub(super) fn commit_replay<'a>(
    ctx: &'a Context,
    replay: Tensor<'a>,
    state: Tensor<'a>,
    keep: Tensor<'a>,
    n_k: i64,
) -> Tensor<'a> {
    let [row, t, ns, _] = replay.ne();
    let h = state.ne()[2];
    assert!(
        replay.ty() == types::F32
            && replay.nb()[0] == 4
            && row == replay_row(n_k, h)
            && state.ne() == [S, S, h, ns]
            && keep.ty() == types::I32
            && keep.elements() as i64 == ns,
        "gdn: deferred commit shapes"
    );
    let es = 4usize;
    let nb = replay.nb();
    let view = |ne: [i64; 4], nb0: usize, off: i64| {
        ctx.view_4d(replay, ne, [nb0, nb[1], nb[2]], off as usize * es)
    };
    let q = view([S, n_k, t, ns], S as usize * es, 0);
    let k = view([S, n_k, t, ns], S as usize * es, S * n_k);
    let v = view([S, h, t, ns], S as usize * es, 2 * S * n_k);
    let g = view([1, h, t, ns], es, 2 * S * n_k + S * h);
    let beta = view([1, h, t, ns], es, 2 * S * n_k + S * h + h);
    ctx.external_with(
        &GDN,
        types::F32,
        [1, 1, 1, 1],
        &[q, k, v, g, beta, state, keep],
        &[MODE_STATE | MODE_KEEP],
    )
}

/// ggml's `gated_delta_net(q, k, v, g, beta, state, 1)` for one token per sequence: the same
/// node shape (F32 `[S * H, n_seqs + S * n_seqs]`: the rows, then the final states) and bits.
/// The input states are only read, so the output may be placed over them (`decode_output`):
/// each CTA reads its part of the states before it writes it.
pub(super) fn decode<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
) -> Tensor<'a> {
    let [_, h, nt, ns] = v.ne();
    assert_eq!(nt, 1, "gdn decode: one token per sequence");
    node(
        ctx,
        MODE_ROWS | MODE_STATE | MODE_TAIL,
        [S * h, ns + S * ns, 1, 1],
        [q, k, v, g, beta, state],
    )
}

/// The op's sources and their float strides, as ggml's CUDA op reads them.
struct Args {
    s_v: usize,
    h: usize,
    n_tokens: usize,
    n_seqs: usize,
    neqk1: usize,
    rq3: usize,
    sq: [usize; 3],
    sv: [usize; 3],
    sb: [usize; 3],
}

/// SAFETY: `dst` is a node made by [`node`].
unsafe fn args(dst: &ffi::ggml_tensor) -> Args {
    unsafe {
        let (q, v, b) = (&*dst.src[0], &*dst.src[2], &*dst.src[4]);
        let f = |nb: usize| nb / 4;
        Args {
            s_v: v.ne[0] as usize,
            h: v.ne[1] as usize,
            n_tokens: v.ne[2] as usize,
            n_seqs: v.ne[3] as usize,
            neqk1: q.ne[1] as usize,
            rq3: (v.ne[3] / q.ne[3]) as usize,
            sq: [f(q.nb[1]), f(q.nb[2]), f(q.nb[3])],
            sv: [f(v.nb[1]), f(v.nb[2]), f(v.nb[3])],
            sb: [f(b.nb[1]), f(b.nb[2]), f(b.nb[3])],
        }
    }
}

/// Where one op's token inputs are read from (float pointers and strides, as [`Args`]).
struct Inputs {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    sq: [usize; 3],
    sv: [usize; 3],
    sb: [usize; 3],
}

impl Inputs {
    /// These inputs from token `t` on.
    ///
    /// SAFETY: the pointers stay inside the inputs' tensors for the tokens read.
    unsafe fn from(&self, t: usize) -> Inputs {
        unsafe {
            Inputs {
                q: self.q.add(t * self.sq[1]),
                k: self.k.add(t * self.sq[1]),
                v: self.v.add(t * self.sv[1]),
                g: self.g.add(t * self.sb[1]),
                beta: self.beta.add(t * self.sb[1]),
                ..*self
            }
        }
    }
}

/// One head's state `s` (transposed, `n * n`) of sequence `seq` advanced over the first
/// `n_tokens` tokens of `x`; with `rows`, each token's attention row `t` goes to
/// `rows(t)` (`n` floats).
///
/// SAFETY: the pointers of `x` cover those tokens.
#[allow(clippy::too_many_arguments)]
unsafe fn cpu_steps(
    s: &mut [f32],
    n: usize,
    x: &Inputs,
    (h, seq, iq1, iq3): (usize, usize, usize, usize),
    n_tokens: usize,
    scale: f32,
    mut rows: Option<&mut dyn FnMut(usize) -> *mut f32>,
) {
    unsafe {
        for t in 0..n_tokens {
            let qt = x.q.add(iq3 * x.sq[2] + t * x.sq[1] + iq1 * x.sq[0]);
            let kt = x.k.add(iq3 * x.sq[2] + t * x.sq[1] + iq1 * x.sq[0]);
            let vt = x.v.add(seq * x.sv[2] + t * x.sv[1] + h * x.sv[0]);
            let gb = seq * x.sb[2] + t * x.sb[1] + h * x.sb[0];
            let (gv, bv) = ((*x.g.add(gb)).exp(), *x.beta.add(gb));
            let out = rows.as_mut().map(|r| r(t));
            for c in 0..n {
                let m = &mut s[c * n..(c + 1) * n];
                let kv: f32 = (0..n).map(|i| m[i] * *kt.add(i)).sum();
                let delta = (*vt.add(c) - gv * kv) * bv;
                let mut attn = 0f32;
                for (i, x) in m.iter_mut().enumerate() {
                    *x = gv * *x + *kt.add(i) * delta;
                    attn += *x * *qt.add(i);
                }
                if let Some(o) = out {
                    *o.add(c) = attn * scale;
                }
            }
        }
    }
}

unsafe fn cpu_gdn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `node`, host tensors. The CPU graphs keep ggml's op; this is the
    // op's reference for tests (the same recurrence, sums in f32 in row order).
    unsafe {
        let mode = crate::ggml::external_params(dst)[0];
        let a = args(dst);
        let src = |i: usize| (*dst.src[i]).data as *const f32;
        let x = Inputs {
            q: src(0),
            k: src(1),
            v: src(2),
            g: src(3),
            beta: src(4),
            sq: a.sq,
            sv: a.sv,
            sb: a.sb,
        };
        let state = (*dst.src[5]).data as *mut f32;
        let rows = dst.data as *mut f32;
        let n = a.s_v;
        // The final states: over the input states, or in the output's tail.
        let out = if mode & MODE_TAIL != 0 {
            rows.add(n * a.h * a.n_tokens * a.n_seqs)
        } else {
            state
        };
        let scale = 1.0 / (n as f32).sqrt();
        let keep =
            (mode & (MODE_KEEP | MODE_REPLAY) != 0).then(|| (*dst.src[6]).data as *const i32);
        let mut s = vec![0f32; n * n];
        if mode & MODE_REPLAY != 0 {
            // A sequence per job, so its heads all replay before its rows are written over.
            let (r, max_t) = replay_inputs(dst, &a);
            for seq in (ith..a.n_seqs).step_by(nth) {
                let w = keep.map_or(0, |kp| *kp.add(seq));
                let (kp, from, _) = count_parts(w);
                let from = from.min(max_t);
                let kp = kp.min(max_t - from);
                let leave = leaves(w, a.n_tokens, max_t);
                let live = live_tokens(dst, mode, seq, a.n_tokens);
                let r = r.from(from);
                for h in 0..a.h {
                    let at = (h, seq, h % a.neqk1, seq / a.rq3);
                    let st = state.add((seq * a.h + h) * n * n);
                    // The deferred commit's tokens first, its state over the slot (unless the
                    // step leaves it), the rows.
                    std::ptr::copy_nonoverlapping(st, s.as_mut_ptr(), n * n);
                    if kp > 0 {
                        cpu_steps(&mut s, n, &r, (h, seq, at.2, seq), kp, scale, None);
                        if !leave {
                            std::ptr::copy_nonoverlapping(s.as_ptr(), st, n * n);
                        }
                    }
                    let mut row = |t: usize| rows.add(((seq * a.n_tokens + t) * a.h + h) * n);
                    cpu_steps(&mut s, n, &x, at, live, scale, Some(&mut row));
                    // The padding tokens' rows (`MODE_PAD`).
                    for t in live..a.n_tokens {
                        std::ptr::write_bytes(row(t), 0, n);
                    }
                }
                cpu_store(dst, &a, &x, seq, if leave { from + kp } else { 0 }, live);
            }
            return;
        }
        for job in (ith..a.h * a.n_seqs).step_by(nth) {
            let (h, seq) = (job % a.h, job / a.h);
            let at = (h, seq, h % a.neqk1, seq / a.rq3);
            let st = state.add((seq * a.h + h) * n * n);
            // A count word (`count_word`): the kept tokens from its start token on.
            let (from, n_tokens) = match keep {
                Some(kp) => {
                    let (k, from, _) = count_parts(*kp.add(seq));
                    let from = from.min(a.n_tokens);
                    (from, k.min(a.n_tokens - from))
                }
                None => (0, a.n_tokens),
            };
            if n_tokens == 0 {
                continue;
            }
            // Transposed, as ggml stores it: column c at c * n.
            std::ptr::copy_nonoverlapping(st, s.as_mut_ptr(), n * n);
            let mut row = |t: usize| rows.add(((seq * a.n_tokens + t) * a.h + h) * n);
            let rows_to =
                (mode & MODE_ROWS != 0).then_some(&mut row as &mut dyn FnMut(usize) -> *mut f32);
            cpu_steps(&mut s, n, &x.from(from), at, n_tokens, scale, rows_to);
            if mode & MODE_STATE != 0 {
                let o = out.add((seq * a.h + h) * n * n);
                std::ptr::copy_nonoverlapping(s.as_ptr(), o, n * n);
            }
        }
    }
}

/// A [`MODE_REPLAY`] node's packed replay rows (its eighth source, `[row, max_t, n_seqs]`, per
/// token `[q | k | v | g | beta]`) as [`Inputs`], and `max_t`.
///
/// SAFETY: `dst` is a node made by [`rows_replay`].
unsafe fn replay_inputs(dst: &ffi::ggml_tensor, a: &Args) -> (Inputs, usize) {
    unsafe {
        let r = &*dst.src[7];
        let base = r.data as *const f32;
        let (p1, p2) = (r.nb[1] / 4, r.nb[2] / 4);
        let qk = a.s_v * a.neqk1;
        let v = 2 * qk;
        let g = v + a.s_v * a.h;
        let x = Inputs {
            q: base,
            k: base.add(qk),
            v: base.add(v),
            g: base.add(g),
            beta: base.add(g + a.h),
            sq: [a.s_v, p1, p2],
            sv: [a.s_v, p1, p2],
            sb: [1, p1, p2],
        };
        (x, r.ne[1] as usize)
    }
}

/// A [`MODE_REPLAY`] node's store of sequence `seq`'s first `live` tokens of `x` (the step's own
/// inputs; the real ones, see [`live_tokens`]) over its packed replay rows from row `at` on: k,
/// v, g and beta; q is left as it is ([`rows_replay`]).
///
/// SAFETY: `dst` is a node made by [`rows_replay`], `x` its inputs, `at` leaves room for them;
/// no other thread reads or writes this sequence's rows.
unsafe fn cpu_store(
    dst: &ffi::ggml_tensor,
    a: &Args,
    x: &Inputs,
    seq: usize,
    at: usize,
    live: usize,
) {
    unsafe {
        let r = &*dst.src[7];
        let (p1, p2) = (r.nb[1] / 4, r.nb[2] / 4);
        let (n, qk) = (a.s_v, a.s_v * a.neqk1);
        let iq3 = seq / a.rq3;
        for t in 0..live {
            let row = (r.data as *mut f32).add(seq * p2 + (at + t) * p1);
            for j in 0..a.neqk1 {
                let k = x.k.add(iq3 * x.sq[2] + t * x.sq[1] + j * x.sq[0]);
                std::ptr::copy_nonoverlapping(k, row.add(qk + j * n), n);
            }
            for h in 0..a.h {
                let v = x.v.add(seq * x.sv[2] + t * x.sv[1] + h * x.sv[0]);
                std::ptr::copy_nonoverlapping(v, row.add(2 * qk + h * n), n);
                let gb = seq * x.sb[2] + t * x.sb[1] + h * x.sb[0];
                *row.add(2 * qk + n * a.h + h) = *x.g.add(gb);
                *row.add(2 * qk + n * a.h + a.h + h) = *x.beta.add(gb);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LegionGdn {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    state: *const f32,
    rows: *mut f32,
    state_out: *mut f32,
    h: i64,
    n_tokens: i64,
    n_seqs: i64,
    sq: [i64; 3],
    sv: [i64; 3],
    sb: [i64; 3],
    neqk1: i64,
    rq3: i64,
    scale: f32,
    num_warps: i32,
    keep: *const i32,
    replay: *mut f32,
    rp1: i64,
    rp2: i64,
    rp_t: i64,
    arrive: *mut i32,
    replay_variant: i32,
    pad: *const i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_gdn_launch(p: *const LegionGdn, stream: *mut c_void) -> i32;
    fn legion_gdn_decode_launch(p: *const LegionGdn, variant: i32, stream: *mut c_void) -> i32;
}

/// The [`decode`] kernel's variant (-1: the default), for the microbenchmark.
#[cfg(feature = "cuda")]
static DECODE_VARIANT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Sets the [`decode`] kernel's variant (`src/cuda/gdn.cu`'s `GDN_DECODE_*` bits; -1: the
/// default) for graphs computed from now on.
#[cfg(feature = "cuda")]
pub(crate) fn set_decode_variant(v: i32) {
    DECODE_VARIANT.store(v, Ordering::Relaxed);
}

/// The [`rows_replay`] kernel's variant (`src/cuda/gdn.cu`'s `replay_variant`; -1: as
/// [`set_gdn_replay_staged`] and [`set_gpu`] choose), for the microbenchmark and tests.
#[cfg(feature = "cuda")]
static REPLAY_VARIANT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Sets the [`rows_replay`] kernel's variant (-1: the engine's choice) for graphs computed from
/// now on.
#[cfg(feature = "cuda")]
pub(crate) fn set_replay_variant(v: i32) {
    REPLAY_VARIANT.store(v, Ordering::Relaxed);
}

/// The variant a [`rows_replay`] launch passes: the forced one, else the kernel's default (-1)
/// where staging is on and the launch's GPU (the current device) chosen, else the per-lane
/// kernel (0).
#[cfg(feature = "cuda")]
fn replay_variant() -> i32 {
    let v = REPLAY_VARIANT.load(Ordering::Relaxed);
    if v >= 0 {
        v
    } else if REPLAY_STAGED.load(Ordering::Relaxed) && gpu_for(crate::hw::per_device::current()) {
        -1
    } else {
        0
    }
}

/// CTA warps for the kernel (0: ggml's choice), for the microbenchmark.
#[cfg(feature = "cuda")]
static GDN_WARPS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(all(test, feature = "cuda"))]
pub(super) fn force_warps(n: i32) {
    GDN_WARPS.store(n, Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gdn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `node`, device tensors of the stream's device.
    unsafe {
        let mode = crate::ggml::external_params(dst)[0];
        let a = args(dst);
        let src = |i: usize| (*dst.src[i]).data as *const f32;
        let state = (*dst.src[5]).data as *mut f32;
        let i = |x: [usize; 3]| x.map(|v| v as i64);
        let mut p = LegionGdn {
            q: src(0),
            k: src(1),
            v: src(2),
            g: src(3),
            beta: src(4),
            state,
            rows: if mode & MODE_ROWS != 0 {
                dst.data as *mut f32
            } else {
                std::ptr::null_mut()
            },
            state_out: if mode & MODE_TAIL != 0 {
                (dst.data as *mut f32).add(a.s_v * a.h * a.n_tokens * a.n_seqs)
            } else if mode & (MODE_STATE | MODE_REPLAY) != 0 {
                // A replaying verify step writes the committed state over the slot.
                state
            } else {
                std::ptr::null_mut()
            },
            h: a.h as i64,
            n_tokens: a.n_tokens as i64,
            n_seqs: a.n_seqs as i64,
            sq: i(a.sq),
            sv: i(a.sv),
            sb: i(a.sb),
            neqk1: a.neqk1 as i64,
            rq3: a.rq3 as i64,
            // As ggml-cuda computes it (host side, correctly rounded sqrt and division).
            scale: 1.0 / (a.s_v as f32).sqrt(),
            num_warps: GDN_WARPS.load(Ordering::Relaxed),
            keep: if mode & (MODE_KEEP | MODE_REPLAY) != 0 {
                (*dst.src[6]).data as *const i32
            } else {
                std::ptr::null()
            },
            replay: std::ptr::null_mut(),
            rp1: 0,
            rp2: 0,
            rp_t: 0,
            arrive: std::ptr::null_mut(),
            replay_variant: 0,
            pad: std::ptr::null(),
        };
        if mode & MODE_REPLAY != 0 {
            let r = &*dst.src[7];
            p.replay = r.data as *mut f32;
            p.rp1 = r.nb[1] as i64 / 4;
            p.rp2 = r.nb[2] as i64 / 4;
            p.rp_t = r.ne[1];
            let params = crate::ggml::external_params(dst);
            p.replay_variant = replay_variant();
            if mode & MODE_PAD != 0 {
                let at = params[2] as u32 as u64 | (params[3] as u32 as u64) << 32;
                p.arrive = at as usize as *mut i32;
                p.pad = (*dst.src[8]).data as *const i32;
            } else {
                p.arrive = ((*dst.src[8]).data as *mut i32).add(params[1] as usize);
            }
        }
        let r = if mode & MODE_TAIL != 0 {
            legion_gdn_decode_launch(&p, DECODE_VARIANT.load(Ordering::Relaxed), stream)
        } else {
            legion_gdn_launch(&p, stream)
        };
        match r {
            0 => Ok(()),
            -1 => Err(format!(
                "gdn: {} heads x {} sequences x {} tokens unsupported",
                a.h, a.n_seqs, a.n_tokens
            )),
            e => Err(format!("gdn: CUDA error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(test)]
mod tests;
