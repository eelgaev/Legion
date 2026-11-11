//! BF16 skinny GEMM for decode steps (#504): `w @ x` for BF16 weights and F32 activations of 2
//! to [`MAX_TOKENS`] tokens as one external op ([`crate::ggml::ExternalOp`]) that gives
//! ggml-cuda's bits, read faster.
//!
//! ggml-cuda runs these matmuls with `mul_mat_f` (BF16 tensor cores). Built with programmatic
//! dependent launch, on compute capability 9.0 and newer it drops `__restrict__` from it, so its
//! weight loads miss the read-only path: on a GB10, serving 16 sequences, it streams the large
//! weights at 160-220 GB/s of the ~230 the GPU reads. The op's kernel (`src/cuda/dense_mv.cu`)
//! does the same arithmetic in the same order (ggml's warp split, tensor-core tiles and
//! warp-order sums), so every output has ggml's bits; it reads the weights with `ld.global.nc`,
//! a few chunks ahead into registers.
//!
//! One token stays on ggml's `mul_mat_vec_f`: it reads at the GPU's limit already in a decode
//! step, and ggml fuses a GLU's two matmuls there. An external op splits ggml's CUDA graph (a few
//! microseconds per split), so the op only takes weights large enough to gain more than that
//! ([`MIN_BYTES`]); a stage graph captures it with the rest (#836), and smaller weights where
//! the op reads faster go to it there too ([`takes_small`]). So do F32 weights of
//! [`MIN_TOKENS_F32`] tokens or more (the MoE router, ggml's `mul_mat_f<float>`: [`set_f32`]).
//!
//! On by default where it applies (CUDA, compute capability 9.0+); `legion serve --dense-mv-ggml`
//! keeps ggml's kernels. The CPU implementation is the reference (F32 sums in row order).

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static DENSE_MV: ExternalOp = ExternalOp {
    name: "dense_mv",
    cpu: cpu_dense_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::dense_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest tokens the op takes (one token stays on ggml's `mul_mat_vec_f`).
pub const MIN_TOKENS: i64 = 2;

/// Fewest tokens the op takes with F32 weights: ggml-cuda runs up to 3 on `mul_mat_vec_f`
/// (its F32 rule on tensor-core GPUs), from 4 on `mul_mat_f<float>`.
pub const MIN_TOKENS_F32: i64 = 4;

/// Most tokens the op takes (ggml-cuda's `mul_mat_f` limit).
pub const MAX_TOKENS: i64 = 16;

/// Most weights one op reads (all with the same input).
pub const MAX_WEIGHTS: usize = 4;

/// Smallest weight the op takes from [`takes`]: 16 MiB, ~70 us of reading on a GB10, of which
/// the op saves more than the graph split costs.
pub const MIN_BYTES: usize = 16 << 20;

/// Whether graphs built from now on may take the op ([`set_dense_mv`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether [`takes_small`] takes weights below [`MIN_BYTES`] ([`set_small`]): [`SMALL_AUTO`],
/// [`SMALL_ON`] or [`SMALL_OFF`].
static SMALL: AtomicU8 = AtomicU8::new(SMALL_AUTO);

/// [`SMALL`]: where stage graphs are on ([`crate::ggml::stage_graph::enabled`]).
const SMALL_AUTO: u8 = 0;
/// [`SMALL`]: always.
const SMALL_ON: u8 = 1;
/// [`SMALL`]: never.
const SMALL_OFF: u8 = 2;

/// Whether [`takes_all`] takes F32 weights ([`set_f32`]).
static F32: AtomicBool = AtomicBool::new(false);

/// Fewest tokens of a chain weight ([`fits_chain`]): ggml-cuda runs BF16 matmuls whose rows
/// `mul_mat_f` does not take on `mul_mat_vec_f` up to 8 tokens, on cuBLAS from 9.
pub const CHAIN_MIN_TOKENS: i64 = 9;

/// Most rows of a chain weight ([`fits_chain`]): the shapes whose cuBLAS kernel was checked
/// (a GDN layer's alpha or beta, 48 rows; on a GB10 cuBLAS takes a one-warp 16 x 16 wmma tile).
pub const CHAIN_MAX_ROWS: i64 = 48;

/// Whether launches of few row tiles spread each tile over a thread-block cluster ([`set_split`]).
static SPLIT: AtomicBool = AtomicBool::new(false);

/// Most tokens of a wide chain ([`fits_wide_chain`]).
pub const WIDE_CHAIN_MAX_TOKENS: i64 = 48;

/// Which matmuls graphs built from now on take as wide chains ([`set_wide_chain`]):
/// [`WIDE_BITWISE`], [`WIDE_ALL`] or [`WIDE_OFF`].
static WIDE_CHAIN: AtomicU8 = AtomicU8::new(WIDE_BITWISE);

/// [`WIDE_CHAIN`]: the shapes and step sizes of cuBLAS's bits ([`wide_chain_bitwise`]).
const WIDE_BITWISE: u8 = 0;
/// [`WIDE_CHAIN`]: every one that fits.
const WIDE_ALL: u8 = 1;
/// [`WIDE_CHAIN`]: none.
const WIDE_OFF: u8 = 2;

/// Whether graphs built from now on run every BF16 matmul of [`MAX_TOKENS`] + 1 to
/// [`WIDE_CHAIN_MAX_TOKENS`] tokens (MTP verify steps of 2-5 sequences, short prompt steps) as a
/// wide chain (`on`), or none (`!on`) ([`takes_wide_chain`], #836): every output one chain of
/// m16n8k16 products over k in order, the order of cuBLAS's wmma kernels without split-K, the
/// weights read with `ld.global.nc`. On a GB10 at 36 tokens ggml-cuda's cuBLAS reads a DeltaNet
/// layer's `qkv` and `z` at ~170 GB/s. cuBLAS sums some shapes in other orders, known ones taken
/// ([`wide_chain_order`]), others not, so all of them can move output (`legion serve
/// --dense-mv-wide-chain`). Until this is
/// called, only the shapes and step sizes of cuBLAS's bits take it ([`wide_chain_bitwise`]);
/// `legion serve --no-dense-mv-wide-chain` keeps them all on cuBLAS.
pub fn set_wide_chain(on: bool) {
    let mode = if on { WIDE_ALL } else { WIDE_OFF };
    WIDE_CHAIN.store(mode, Ordering::Relaxed);
}

/// The compute capability (10 * major + minor) [`WIDE_CHAIN_BITWISE`] was measured on: a GB10.
pub const WIDE_CHAIN_CC: i32 = 121;

/// The BF16 weight shapes `(k, m)` (`w` is `[k, m]`) whose matmuls a wide chain computes with
/// the bits of ggml-cuda's cuBLAS on a GB10 ([`WIDE_CHAIN_CC`]), the step sizes (bit `n - 17` for
/// `n` tokens) and the order: 0, one chain over `k` (cuBLAS's wmma kernels); else the length of
/// the `k` slices whose sums cuBLAS adds in order (its nvjet kernel of 64 x 56 tiles at 33-48
/// tokens sums `k` = 6,144 in 8 slices of 768, [`WideOrder::kslice`]). A shape may be listed
/// more than once, with other steps. Measured with `legion dense-mv-bench --check --tokens
/// 17,18,...,48` (random weights and inputs, every output's bits compared; `--wide-probe` finds
/// the order); the shapes are Qwen3.8-Flash-Next's (NVFP4 checkpoint) BF16 matmuls of 16 MiB and
/// more. cuBLAS picks its kernel by shape, so a cuBLAS that picks others needs that check again.
/// The output projections (`k` = 6,144) at 17-25 tokens run an nvjet kernel of 64 x 32 tiles
/// whose order no probed one matches, so they stay on cuBLAS. The LM head (`[2560, 248320]`)
/// has cuBLAS's bits at every step too but reads no faster (1-4% slower from 20 tokens), so it
/// stays on cuBLAS. On a GB10 the listed ones take 25-50% less time alone than cuBLAS.
pub const WIDE_CHAIN_BITWISE: &[(i64, i64, u32, i32)] = &[
    // DeltaNet in_proj_qkv, the PLE key projection.
    (2560, 10240, u32::MAX, 0),
    // DeltaNet in_proj_z.
    (2560, 6144, u32::MAX, 0),
    // Attention q_proj.
    (2560, 12288, u32::MAX, 0),
    // DeltaNet out_proj, attention o_proj: 26-32 tokens one chain, 33-48 slices of 768.
    (6144, 2560, 0x0000_FE00, 0),
    (6144, 2560, 0xFFFF_0000, 768),
];

/// The order in which a wide chain gives cuBLAS's bits for `[k, m]` weights at `n` tokens on a
/// GB10 ([`WIDE_CHAIN_BITWISE`]), if it does.
pub fn wide_chain_order(k: i64, m: i64, n: i64) -> Option<WideOrder> {
    if !(MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS).contains(&n) {
        return None;
    }
    WIDE_CHAIN_BITWISE
        .iter()
        .find(|&&(wk, wm, steps, _)| wk == k && wm == m && steps >> (n - MAX_TOKENS - 1) & 1 != 0)
        .map(|&(_, _, _, kslice)| WideOrder {
            kslice,
            ..WideOrder::default()
        })
}

/// Whether a wide chain gives cuBLAS's bits for `[k, m]` weights at `n` tokens on a GB10
/// ([`WIDE_CHAIN_BITWISE`]).
pub fn wide_chain_bitwise(k: i64, m: i64, n: i64) -> bool {
    wide_chain_order(k, m, n).is_some()
}

/// Whether [`takes_all`] takes chain weights ([`set_chain`]).
static CHAIN: AtomicBool = AtomicBool::new(true);

/// Whether graphs built from now on let small BF16 weights that ggml-cuda runs on cuBLAS ride
/// along in a dense op of larger weights with the same input ([`fits_chain`], #836): a GDN
/// layer's alpha and beta (48 rows each) in its `qkv` and `z` launch at 9-16 tokens, in
/// cuBLAS's order, so with its bits. On by default; `legion serve --no-dense-mv-ab` keeps them
/// on cuBLAS (for comparisons). Never on compute capability 9.x ([`chain_cc`]).
pub fn set_chain(on: bool) {
    CHAIN.store(on, Ordering::Relaxed);
}

/// Whether chain weights run on a device of compute capability `cc` (#1215): not on 9.x
/// (Hopper), whose cuBLAS picks other kernels than the GB10's, against which the chain's order
/// was matched, so a chain there would not give cuBLAS's bits; those weights stay on cuBLAS.
pub fn chain_cc(cc: Option<i32>) -> bool {
    !cc.is_some_and(|cc| cc / 10 == 9)
}

/// Whether launches from now on that have fewer tiles of rows than the GPU has SMs spread each
/// tile's warps over a thread-block cluster of 2-8 CTAs (#836; compute capability 9.0+, BF16
/// weights without chain weights). Each warp computes the same sums wherever it runs, and the
/// tile's sums are added in ggml's warp order from the cluster's distributed shared memory, so
/// the same bits. On a GB10 at 9 tokens a hyper-connection mixer's down matrix (320 rows of
/// 10,240) is 10 tiles, otherwise 10 SMs of 48: 37.6 against 35.2 us a launch, but C=1 MTP
/// decode tok/s within noise, so off by default; `legion serve --dense-mv-split` turns it on.
pub fn set_split(on: bool) {
    SPLIT.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on run F32 matmuls of [`MIN_TOKENS_F32`] to [`MAX_TOKENS`]
/// tokens on the op, at any size (#836): ggml-cuda's `mul_mat_f<float>` bits (TF32 products)
/// with the loads in flight. On a GB10 at 16 tokens that is qwen4exp's MoE router (2560 x 512,
/// 5 MiB), which ggml reads with 16 CTAs at ~125 GB/s. Off by default (`legion serve
/// --dense-mv-f32` turns it on): in a GB10 decode step of 16 sequences the router took 1.73 ms
/// on the op against ggml's 1.87, and decode tok/s did not move (117.8 vs 117.95, one A/B each).
/// Without the stage graphs each op also splits ggml's CUDA graph.
pub fn set_f32(on: bool) {
    F32.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on run the small matmuls that the op reads faster than ggml on
/// the op too ([`takes_small`], #860): the same bits. `legion serve --dense-mv-small` turns it
/// on, `--no-dense-mv-small` off; by default it follows the stage graphs (#836). Inside a
/// stage graph the op costs no graph split, and on a GB10 decode step of 16 sequences the
/// small matmuls take ~2.5 ms less GPU time (of ~130); between ggml's own CUDA graphs each op
/// adds a split, and the gain stayed within noise (#860).
pub fn set_small(on: bool) {
    SMALL.store(if on { SMALL_ON } else { SMALL_OFF }, Ordering::Relaxed);
}

/// Whether [`takes_small`] takes small weights now ([`set_small`]).
pub fn small() -> bool {
    match SMALL.load(Ordering::Relaxed) {
        SMALL_ON => true,
        SMALL_OFF => false,
        _ => crate::ggml::stage_graph::enabled(),
    }
}

/// Whether graphs built from now on run BF16 matmuls of 2-16 tokens on the op where it applies
/// (on by default); off keeps ggml's kernels (`legion serve --dense-mv-ggml`, for comparisons).
pub fn set_dense_mv(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether `w @ x` takes the op: [`fits`], the op on ([`set_dense_mv`]), a weight of at least
/// [`MIN_BYTES`] in a GPU's memory (not host memory, not split over GPUs), and a GPU where the
/// op runs ([`device_supported`]).
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    takes_all(&[w], x)
}

/// Whether `w @ x` for every weight of `ws` (1 to [`MAX_WEIGHTS`]) takes one op ([`mul_mat`]):
/// as [`takes`], with [`MIN_BYTES`] counted over all of them. A group's smaller weights (an
/// attention layer's K and V beside its Q) ride along in the launch the large one pays for,
/// with ggml's bits all the same. F32 weights take it at any size, where [`f32`] says so.
pub fn takes_all(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    if ws.iter().any(|w| w.ty() == types::F32) {
        return f32() && takes_any(ws, x);
    }
    ws.iter().map(|w| w.nbytes()).sum::<usize>() >= MIN_BYTES && takes_any(ws, x)
}

/// Whether [`takes_all`] takes F32 weights now (`--dense-mv-f32`, [`set_f32`]).
pub fn f32() -> bool {
    F32.load(Ordering::Relaxed)
}

/// As [`takes_all`] at any size, where [`small`] says so (stage graphs on, or `--dense-mv-small`;
/// [`set_small`]): for the call
/// sites whose small matmuls the op reads faster than ggml's `mul_mat_f` on a GB10 (#860): a
/// shared expert's gate and up joined (2560 x 640 each, ~9% faster at 16 tokens) and the
/// hyper-connection mixers' down projections (10240 x 320, ~10%). Others lose in the op (short
/// `k`, one weight of ~3 MB) and stay on [`takes`].
pub fn takes_small(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    small() && takes_any(ws, x)
}

/// [`takes_all`] without its size floor.
fn takes_any(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && (1..=MAX_WEIGHTS).contains(&ws.len())
        && chain_count(ws, x).is_some_and(|c| {
            c == 0 || (CHAIN.load(Ordering::Relaxed) && chain_cc(compute_capability()))
        })
        && ws
            .iter()
            .all(|&w| w.ty() == ws[0].ty() && !w.is_host() && !w.is_meta())
        && device_supported()
}

/// How many of `ws`'s last weights are chain weights ([`fits_chain`]) when all the ones before
/// them, at least one, [`fits`]; `None` otherwise. The op runs them in cuBLAS's order.
pub fn chain_count(ws: &[Tensor<'_>], x: Tensor<'_>) -> Option<usize> {
    let main = ws.iter().take_while(|&&w| fits(w, x)).count();
    (main > 0 && ws[main..].iter().all(|&w| fits_chain(w, x))).then_some(ws.len() - main)
}

/// Whether the op computes `w @ x` as ggml-cuda's cuBLAS path does (a chain weight, #836): a BF16
/// matrix `w` (`[k, m]`) whose `m` (16 or 48, at most [`CHAIN_MAX_ROWS`]) is a multiple of 16
/// but not of 32, so that ggml's `mul_mat_f` does not take it, with an F32 `x` of
/// [`CHAIN_MIN_TOKENS`] to [`MAX_TOKENS`] tokens; layout as in [`fits`]. Only beside weights
/// that [`fits`] takes ([`chain_count`]).
pub fn fits_chain(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [k, m, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let (wnb, xnb) = (w.nb(), x.nb());
    w.ty() == types::BF16
        && x.ty() == types::F32
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && k % 64 == 0
        && m % 32 == 16
        && m <= CHAIN_MAX_ROWS
        && wnb[0] == 2
        && wnb[1..].iter().all(|b| b % 4 == 0)
        && xnb[0] == 4
        && xnb[1] % 8 == 0
        && (CHAIN_MIN_TOKENS..=MAX_TOKENS).contains(&n)
}

/// Whether the op computes `w @ x` as ggml-cuda's `mul_mat_f` would: a BF16 matrix `w` (`[k, m]`,
/// rows contiguous and 4-byte aligned) and an F32 `x` (`[k, n]`, values contiguous, columns 8-byte
/// aligned) of [`MIN_TOKENS`] to [`MAX_TOKENS`] tokens, `k` a multiple of 64 and `m` of 32; or an
/// F32 matrix `w` (ggml's `mul_mat_f<float>`, rows 8-byte aligned) from [`MIN_TOKENS_F32`] tokens.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [k, m, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let (wnb, xnb) = (w.nb(), x.nb());
    let (elem, min_n) = match w.ty() {
        types::BF16 => (2, MIN_TOKENS),
        types::F32 => (4, MIN_TOKENS_F32),
        _ => return false,
    };
    x.ty() == types::F32
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && k % 64 == 0
        && m % 32 == 0
        && wnb[0] == elem
        && wnb[1..].iter().all(|b| b % (2 * elem) == 0)
        && xnb[0] == 4
        && xnb[1] % 8 == 0
        && (min_n..=MAX_TOKENS).contains(&n)
}

/// Whether the op computes `w @ x` as a wide chain ([`set_wide_chain`]): a BF16 matrix `w`
/// (`[k, m]`, rows contiguous and 4-byte aligned, `k` a multiple of 64, `m` of 16) and an F32 `x`
/// (`[k, n]`, values contiguous, columns 8-byte aligned) of [`MAX_TOKENS`] + 1 to
/// [`WIDE_CHAIN_MAX_TOKENS`] tokens.
pub fn fits_wide_chain(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [k, m, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let (wnb, xnb) = (w.nb(), x.nb());
    w.ty() == types::BF16
        && x.ty() == types::F32
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && k > 0
        && k % 64 == 0
        && m > 0
        && m % 16 == 0
        && wnb[0] == 2
        && wnb[1..].iter().all(|b| b % 4 == 0)
        && xnb[0] == 4
        && xnb[1] % 8 == 0
        && (MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS).contains(&n)
}

/// Whether `w @ x` takes a wide chain ([`mul_mat_wide_chain`]): [`fits_wide_chain`], the op on
/// ([`set_dense_mv`]), a weight of at least [`MIN_BYTES`] in a GPU's memory, a GPU where the op
/// runs ([`device_supported`]), and wide chains on for every shape ([`set_wide_chain`]) or, by
/// default, a shape and step size of cuBLAS's bits on a GB10 ([`wide_chain_bitwise`]).
pub fn takes_wide_chain(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let mode = WIDE_CHAIN.load(Ordering::Relaxed);
    let [k, m, _, _] = w.ne();
    mode != WIDE_OFF
        && ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && fits_wide_chain(w, x)
        && (mode == WIDE_ALL || wide_chain_bitwise(k, m, x.ne()[1]))
        && w.nbytes() >= MIN_BYTES
        && !w.is_host()
        && !w.is_meta()
        && device_supported()
        && (mode == WIDE_ALL || compute_capability() == Some(WIDE_CHAIN_CC))
}

/// The current device's compute capability (10 * major + minor), read once per device; `None`
/// without CUDA or when the queries fail.
pub(crate) fn compute_capability() -> Option<i32> {
    #[cfg(feature = "cuda")]
    {
        cuda::compute_capability()
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// The order in which a wide chain sums each output's products over `k`: one chain of m16n8k16
/// products from 0 (the default), or slices of `kslice` values, each its own chain, their sums
/// added in order. `kinter` > 1 (a probe for `legion dense-mv-bench`): within a slice the steps
/// of 16 go to `kinter` accumulators in turn, `kgroup` values each, added in order at its end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideOrder {
    pub kslice: i32,
    pub kinter: i32,
    pub kgroup: i32,
    /// 8 (probe): each step of 16 as two m16n8k8 products; 0 or 16: one m16n8k16.
    pub kmma: i32,
    /// Probe: how the accumulators are added (0 in order, 1 pairwise, 2 last to first).
    pub kred: i32,
}

/// `w @ x` (F32 `[m, n]`) as a wide chain ([`fits_wide_chain`]), in cuBLAS's order on a GB10
/// where [`wide_chain_order`] knows it, else one chain over `k`.
pub fn mul_mat_wide_chain<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    let [k, m, _, _] = w.ne();
    let o = wide_chain_order(k, m, x.ne()[1]).unwrap_or_default();
    mul_mat_wide_chain_order(ctx, w, x, o)
}

/// [`mul_mat_wide_chain`] in the order `o`.
pub fn mul_mat_wide_chain_order<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    o: WideOrder,
) -> Tensor<'a> {
    assert!(fits_wide_chain(w, x), "dense_mv: wide chain shapes");
    let [_, m, _, _] = w.ne();
    let n = x.ne()[1];
    let params = [0, o.kslice, o.kinter, o.kgroup, o.kmma, o.kred];
    ctx.external_with(&DENSE_MV, types::F32, [m, n, 1, 1], &[x, w], &params)
}

/// Whether every weight of `ws` (1 to [`MAX_WEIGHTS`], the same input) takes a wide chain
/// ([`takes_wide_chain`]) in the same order, so that one launch runs them all
/// ([`mul_mat_wide_chains`]): a DeltaNet layer's qkv and z at 17-48 tokens.
pub fn takes_wide_chains(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    let order = |w: Tensor<'_>| {
        let [k, m, _, _] = w.ne();
        wide_chain_order(k, m, x.ne()[1]).unwrap_or_default()
    };
    (1..=MAX_WEIGHTS).contains(&ws.len())
        && ws.iter().all(|&w| takes_wide_chain(w, x))
        && ws.iter().all(|&w| order(w) == order(ws[0]))
}

/// [`mul_mat_wide_chain`] for each weight of `ws` ([`takes_wide_chains`], or each
/// [`fits_wide_chain`] with `x` in one order) in one launch: F32 `[m_i, n]` results, contiguous.
/// Each output is computed as one weight's launch computes it, so the same bits; one launch
/// instead of one per weight, and the weights' row tiles spread over the GPU together.
pub fn mul_mat_wide_chains<'a>(
    ctx: &'a Context,
    ws: &[Tensor<'a>],
    x: Tensor<'a>,
) -> Vec<Tensor<'a>> {
    assert!(
        (1..=MAX_WEIGHTS).contains(&ws.len()),
        "dense_mv: {} weights",
        ws.len()
    );
    if let [w] = ws {
        return vec![mul_mat_wide_chain(ctx, *w, x)];
    }
    let n = x.ne()[1];
    let order = |w: Tensor<'_>| {
        let [k, m, _, _] = w.ne();
        wide_chain_order(k, m, n).unwrap_or_default()
    };
    let o = order(ws[0]);
    for &w in ws {
        assert!(fits_wide_chain(w, x), "dense_mv: wide chain shapes");
        assert!(order(w) == o, "dense_mv: wide chain orders");
    }
    let params = [0, o.kslice, o.kinter, o.kgroup, o.kmma, o.kred];
    let mut args = vec![x];
    args.extend_from_slice(ws);
    let total: i64 = ws.iter().map(|w| w.ne()[1]).sum();
    let node = ctx.external_with(&DENSE_MV, types::F32, [total * n, 1, 1, 1], &args, &params);
    let mut off = 0;
    ws.iter()
        .map(|w| {
            let m = w.ne()[1];
            let v = ctx.view_2d(node, [m, n], m as usize * 4, off);
            off += (m * n) as usize * 4;
            v
        })
        .collect()
}

/// The op parameter of [`mul_mat_wide`]'s node: its BF16 weight holds an F32 matrix's values.
const WIDE: i32 = 1;

/// Whether [`mul_mat_wide`] takes `w @ x`: [`fits_wide`], the op on ([`set_dense_mv`]), the
/// weight in a GPU's memory and a GPU where the op runs ([`device_supported`]). Any size: the
/// F32 matmuls it stands for are small (the MoE router).
pub fn takes_wide(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && fits_wide(w, x)
        && !w.is_host()
        && !w.is_meta()
        && device_supported()
}

/// Whether the op computes `wf @ x` as ggml-cuda's `mul_mat_f<float>` would from `w`, the BF16
/// copy of an F32 matrix `wf` whose values are all exact in BF16 (#836: a BF16 checkpoint's MoE
/// router, which the loader widens to F32): a BF16 `w` (`[k, m]`, rows 4-byte aligned) laid out
/// as [`fits`] takes a BF16 weight, with an F32 `x` of [`MIN_TOKENS_F32`] to [`MAX_TOKENS`]
/// tokens (where ggml runs the F32 matmul on `mul_mat_f<float>`).
pub fn fits_wide(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    w.ty() == types::BF16 && fits(w, x) && x.ne()[1] >= MIN_TOKENS_F32
}

/// `wf @ x` (F32 `[m, n]`) for the F32 matrix `wf` whose BF16 copy is `w` ([`fits_wide`]):
/// ggml-cuda's `mul_mat_f<float>` bits (TF32 products of `wf`'s raw F32 bits in its order), the
/// weight read from `w`, half the bytes. The kernel widens each value back to its F32 bits as
/// it stages it, so it computes on the same words as the F32 kernel.
pub fn mul_mat_wide<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits_wide(w, x), "dense_mv: wide shapes");
    let [_, m, _, _] = w.ne();
    let n = x.ne()[1];
    ctx.external_with(&DENSE_MV, types::F32, [m, n, 1, 1], &[x, w], &[WIDE])
}

/// Whether the current GPU runs the op: compute capability 9.0 or newer, with the kernel in this
/// build. Older GPUs keep ggml's kernels, which keep `__restrict__` there.
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_dense_mv_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_dense_mv_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Launches every op from now on with `nta` tiles of 16 rows per CTA and `d` chunks of each
/// warp's weight rows in flight (`(nta, d)` of (1, 2), (1, 4), (1, 8), (2, 2), (2, 4)), or the
/// default choice for `None`: for tests and tuning, the same bits either way. False where the
/// kernels lack the variant (or without CUDA).
#[doc(hidden)]
pub fn force_variant(v: Option<(i32, i32)>) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_dense_mv_force(nta: i32, d: i32) -> i32;
        }
        let (nta, d) = v.unwrap_or((0, 0));
        // SAFETY: stores the choice in an atomic.
        unsafe { legion_dense_mv_force(nta, d) == 0 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = v;
        false
    }
}

/// `w @ x` for each weight of `ws` (1 to [`MAX_WEIGHTS`], each [`fits`] with `x`, the last ones
/// possibly chain weights: [`chain_count`]) in one op: F32 `[m_i, n]` results, contiguous.
pub fn mul_mat<'a>(ctx: &'a Context, ws: &[Tensor<'a>], x: Tensor<'a>) -> Vec<Tensor<'a>> {
    assert!(
        (1..=MAX_WEIGHTS).contains(&ws.len()),
        "dense_mv: {} weights",
        ws.len()
    );
    assert!(chain_count(ws, x).is_some(), "dense_mv: shapes");
    for &w in ws {
        assert!(w.ty() == ws[0].ty(), "dense_mv: types");
    }
    let n = x.ne()[1];
    let mut args = vec![x];
    args.extend_from_slice(ws);
    if let [w] = ws {
        return vec![ctx.external(&DENSE_MV, types::F32, [w.ne()[1], n, 1, 1], &args)];
    }
    let total: i64 = ws.iter().map(|w| w.ne()[1]).sum();
    let node = ctx.external(&DENSE_MV, types::F32, [total * n, 1, 1, 1], &args);
    let mut off = 0;
    ws.iter()
        .map(|w| {
            let m = w.ne()[1];
            let v = ctx.view_2d(node, [m, n], m as usize * 4, off);
            off += (m * n) as usize * 4;
            v
        })
        .collect()
}

/// The `[m, n]` result of weight `i` of a node made by [`mul_mat`]: its offset in `dst.data`.
fn out_offset(dst: &ffi::ggml_tensor, i: usize) -> usize {
    // SAFETY: a node made by `mul_mat`: src[0] the input, src[1..] the weights.
    unsafe {
        let n = (*dst.src[0]).ne[1] as usize;
        (1..=i).map(|j| (*dst.src[j]).ne[1] as usize * n * 4).sum()
    }
}

/// The weights of a node made by [`mul_mat`].
fn weights(dst: &ffi::ggml_tensor) -> usize {
    dst.src[1..]
        .iter()
        .take(MAX_WEIGHTS)
        .take_while(|s| !s.is_null())
        .count()
}

unsafe fn cpu_dense_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        for i in 0..weights(dst) {
            let w = &*dst.src[1 + i];
            let m = w.ne[1] as usize;
            let out = (dst.data as *mut u8).add(out_offset(dst, i)) as *mut f32;
            let f32_rows = w.type_ == types::F32;
            for r in (ith..m).step_by(nth) {
                let row = (w.data as *const u8).add(r * w.nb[1]);
                for t in 0..n {
                    let col = (x.data as *const u8).add(t * x.nb[1]) as *const f32;
                    let mut s = 0f32;
                    for c in 0..k {
                        let wv = if f32_rows {
                            *(row as *const f32).add(c)
                        } else {
                            f32::from_bits(u32::from(*(row as *const u16).add(c)) << 16)
                        };
                        s += wv * *col.add(c);
                    }
                    *out.add(t * m + r) = s;
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::c_void;

    use super::*;

    /// `legion_dense_mv` of `src/cuda/dense_mv.cu`.
    #[repr(C)]
    struct Params {
        w: [*const u16; MAX_WEIGHTS],
        dst: [*mut f32; MAX_WEIGHTS],
        w_ld: [i64; MAX_WEIGHTS],
        rows: [i64; MAX_WEIGHTS],
        x: *const f32,
        x_ld: i64,
        k: i64,
        n: i32,
        nw: i32,
        f32: i32,
        nchain: i32,
        split: i32,
        warps: i32,
        kslice: i32,
        kinter: i32,
        kgroup: i32,
        kmma: i32,
        kred: i32,
    }

    unsafe extern "C" {
        fn legion_dense_mv_launch(p: *const Params, stream: *mut c_void) -> i32;
        fn cudaGetDevice(device: *mut i32) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
        fn cudaGetLastError() -> i32;
    }

    /// `cudaDevAttrComputeCapabilityMajor` and `Minor`.
    const DEV_ATTR_CC_MAJOR: i32 = 75;
    const DEV_ATTR_CC_MINOR: i32 = 76;

    /// Devices [`CC`] covers.
    const DEVICES: usize = 64;

    /// Each device's compute capability; 0 not yet read, -1 unknown.
    static CC: [std::sync::atomic::AtomicI32; DEVICES] =
        [const { std::sync::atomic::AtomicI32::new(0) }; DEVICES];

    /// The current device's compute capability, read once per device.
    pub(super) fn compute_capability() -> Option<i32> {
        let mut dev = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as i32).contains(&dev) {
                cudaGetLastError();
                return None;
            }
            let slot = &CC[dev as usize];
            let mut cc = slot.load(Ordering::Relaxed);
            if cc == 0 {
                let (mut major, mut minor) = (0, 0);
                cc = if cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0
                    && cudaDeviceGetAttribute(&mut minor, DEV_ATTR_CC_MINOR, dev) == 0
                {
                    10 * major + minor
                } else {
                    cudaGetLastError();
                    -1
                };
                slot.store(cc, Ordering::Relaxed);
            }
            (cc > 0).then_some(cc)
        }
    }

    pub(super) unsafe fn dense_mv(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`, device tensors of the stream's device.
        unsafe {
            let x = &*dst.src[0];
            let nw = weights(dst);
            let f32 = (*dst.src[1]).type_ == types::F32;
            let elem = if f32 { 4 } else { 2 };
            // 1: F32 weights; 2: BF16 copies of F32 ones (`mul_mat_wide`); 0: BF16 weights.
            let params = crate::ggml::external_params(dst);
            let mode = if f32 {
                1
            } else if params[0] == WIDE {
                2
            } else {
                0
            };
            let mut p = Params {
                w: [std::ptr::null(); MAX_WEIGHTS],
                dst: [std::ptr::null_mut(); MAX_WEIGHTS],
                w_ld: [0; MAX_WEIGHTS],
                rows: [0; MAX_WEIGHTS],
                x: x.data as *const f32,
                x_ld: (x.nb[1] / 4) as i64,
                k: x.ne[0],
                n: x.ne[1] as i32,
                nw: nw as i32,
                f32: mode,
                // The chain weights (`chain_count`): the only ones whose rows are not a
                // multiple of 32; all of them past MAX_TOKENS (a wide chain).
                nchain: if x.ne[1] > MAX_TOKENS {
                    nw as i32
                } else {
                    (0..nw)
                        .filter(|&i| (*dst.src[1 + i]).ne[1] % 32 != 0)
                        .count() as i32
                },
                split: i32::from(SPLIT.load(Ordering::Relaxed)),
                warps: 0,
                // Wide chains: the order of `mul_mat_wide_chain_order`'s node.
                kslice: params[1],
                kinter: params[2],
                kgroup: params[3],
                kmma: params[4],
                kred: params[5],
            };
            for i in 0..nw {
                let w = &*dst.src[1 + i];
                p.w[i] = w.data as *const u16;
                p.dst[i] = (dst.data as *mut u8).add(out_offset(dst, i)) as *mut f32;
                p.w_ld[i] = (w.nb[1] / elem) as i64;
                p.rows[i] = w.ne[1];
            }
            match legion_dense_mv_launch(&p, stream) {
                0 => Ok(()),
                e => Err(format!("error {e}")),
            }
        }
    }
}

pub mod bench;

#[cfg(test)]
mod tests;
