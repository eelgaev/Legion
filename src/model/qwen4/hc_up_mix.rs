//! qwen4exp's hyper-connection mixer after the down projection, prompt steps on a GB10: the up
//! matmul and the gated mix as one external op ([`crate::ggml::ExternalOp`], CPU reference plus
//! the kernels in `src/cuda/hc_up_mix.cu`), instead of ggml's `mul_mat(up, lo)` and
//! `hc_pre_gated(xn, gate, scale)`. The gate (`hc · n_embd` F32 values per token) stays in
//! registers instead of being written by the matmul and read back by the mix (#526).
//!
//! The same bits as ggml's two ops: the kernel quantizes `lo` with ggml-cuda's expressions,
//! adds MMQ's per-block products in MMQ's order and then mixes as `dsv4_hc_pre_f32` does. That
//! holds where ggml's MMQ computes every output tile in one CTA; with a stream-k split
//! (`mul_mat_q_stream_k_fixup`) ggml adds partial sums in another order, so [`fits`] keeps
//! ggml's ops for those step sizes ([`mmq_whole_tiles`]).
//!
//! After [`super::hc_norm::hc_post_norm_rebuilt`] the op also rebuilds `xn` itself
//! ([`up_mix_rebuilt`]): the norm then writes each stream's scale instead of `xn` (4 floats a
//! token, not `hc · n_embd`), and `xn = (res · scale) · gamma` here is the norm's own two
//! rounded multiplies, so the same bits; `legion serve --hc-xn-written` keeps `xn` written.
//!
//! BF16 `up` weights (the NVFP4 checkpoints' mixers, #836), where ggml runs the up matmul on
//! cuBLAS (`lo` rounded to BF16, a BF16 GEMM with F32 sums), can take a BF16 MMA kernel
//! instead: the GEMM's m16n8k16 products added in K order from zero, then the same mix
//! ([`fits_up_bf16`]), as two nodes in the places of ggml's two (`HC_UP_LOB`), so ggml's
//! allocator plans the graph as it does for ggml's ops. On by default (`legion serve
//! --no-hc-up-mix-bf16` keeps ggml's ops): its first form (one node of another size) moved
//! greedy output at C=4 on a GB10 through ggml-cuda's layout-dependent MoE fusions; with the
//! two nodes the greedy output equals ggml's ops' (and it did under `--stable-fusions` too).
//!
//! Chosen on compute capability 12.x GPUs (GB10), where it was measured; `legion serve
//! --hc-up-mix-ggml` keeps ggml's two ops everywhere.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static HC_UP_MIX: ExternalOp = ExternalOp {
    name: "hc_up_mix",
    cpu: cpu_hc_up_mix,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_up_mix),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// For a BF16 `up`: `lo` rounded to BF16 into a node of the gate's shape (`[hc · e, n]` F32,
/// the place ggml's `mul_mat(up, lo)` takes), which the mix node then reads (#836). The two
/// nodes have the sizes, sources and order of ggml's matmul and `hc_pre`, so ggml's allocator
/// places every tensor of the graph where it does for ggml's two ops (a graph of other sizes
/// moves ggml-cuda's layout-dependent MoE fusions, `crate::model::core::stable_fusions`).
static HC_UP_LOB: ExternalOp = ExternalOp {
    name: "hc_up_lob",
    cpu: cpu_hc_up_lob,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_up_lob),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Streams the kernel takes.
const HC: i64 = 4;

/// Mixed columns per CTA: `n_embd` must be a multiple.
const ROWS: i64 = 16;

/// Values per Q8_0 block and per scale of the quantized `lo`.
const GROUP: usize = 32;

/// Largest low rank the kernel takes (16 blocks).
const MAX_RANK: i64 = 512;

/// Fewest tokens [`fits`] takes. The kernel is ggml's bits from 9 tokens (below, ggml-cuda runs
/// the up matmul as MMVQ, its own sums), but an external op splits the step's CUDA graph: on
/// decode and verify steps (up to 16 sequences x 9 tokens with the GB10's default draft depth
/// of 8) the extra segments cost more than the op saves (GB10, `--deterministic` 9-token verify
/// steps: +0.5 ms per token). Prompt steps are far larger.
pub(super) const MIN_TOKENS: i64 = 256;

/// Whether graphs built from now on may use [`up_mix`] (`legion serve --hc-up-mix-ggml` turns
/// it off for comparisons).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether graphs built from now on may use [`up_mix`] with BF16 `up` weights (on by default;
/// `legion serve --no-hc-up-mix-bf16` keeps ggml's two ops for them).
static BF16_ON: AtomicBool = AtomicBool::new(true);

/// Largest low rank the BF16 kernel takes (its `up` rows staged in 48 KiB of shared memory).
const MAX_RANK_BF16: i64 = 368;

/// Sets whether graphs built from now on run the mixer's up matmul and gated mix as one op for
/// BF16 `up` weights where it fits (on by default; [`set_up_mix`] off turns it off too); `false`
/// keeps ggml's cuBLAS `mul_mat` and `hc_pre`.
pub fn set_up_mix_bf16(on: bool) {
    BF16_ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on may use [`up_mix_rebuilt`] (`legion serve --hc-xn-written`
/// turns it off for comparisons).
static REBUILD: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on have the norm write the streams' scales instead of
/// `xn` where [`up_mix`] follows and rebuild `xn` there (on by default); `false` keeps `xn`
/// written and read.
pub fn set_rebuild(on: bool) {
    REBUILD.store(on, Ordering::Relaxed);
}

/// See [`set_rebuild`].
pub(super) fn rebuild() -> bool {
    REBUILD.load(Ordering::Relaxed)
}

/// The engine's GPU's streaming multiprocessors when it is one the op is chosen for (compute
/// capability 12.x), else 0; set when the engine loads ([`set_gpu`]).
static GPU_SMS: AtomicI32 = AtomicI32::new(0);

/// Sets whether graphs built from now on run the mixer's up matmul and gated mix as one op where
/// it fits (on by default); `false` keeps ggml's `mul_mat` and `hc_pre`.
pub fn set_up_mix(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: [`up_mix`] is used on compute capability 12.x (`cc_major`: GB10) with
/// `sms` streaming multiprocessors (what ggml-cuda's stream-k decomposition splits over).
pub fn set_gpu(cc_major: Option<i32>, sms: i32) {
    GPU_SMS.store(
        if cc_major == Some(12) { sms.max(0) } else { 0 },
        Ordering::Relaxed,
    );
}

/// The SMs of `up`'s GPU where the op runs there, else 0: the engine's GPU is one it is chosen
/// for ([`set_gpu`]) and so is the weight's own device (asked once per device: a box may hold
/// GPUs of different architectures, #1201), whose SM count the split is planned for.
fn sms_for(up: Tensor<'_>) -> i32 {
    let engine = GPU_SMS.load(Ordering::Relaxed);
    if engine <= 0 {
        return 0;
    }
    up.cuda_device()
        .map_or(engine, crate::hw::per_device::sms_if_cc12)
}

/// ggml-cuda's MMQ tile widths for a Q8_0 weight whose rows are a multiple of 128
/// (`mmq-config-ampere.cuh`, which Blackwell and Ampere take; no `fallback`).
const MMQ_J: [i64; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Rows per MMQ tile (`I`).
const MMQ_I: i64 = 128;

/// Q8_0 blocks per MMQ K iteration (`MMQ_ITER_K / QK8_0`).
const MMQ_BLOCKS_PER_ITER: i64 = 8;

/// Whether ggml-cuda's MMQ of a Q8_0 weight of `rows` rows of `k` values times `n` columns,
/// on a GPU with `sms` multiprocessors, computes every output tile in one CTA, from the first
/// K block to the last: then each output is MMQ's sum in K order, as [`up_mix`] computes it.
/// It mirrors `ggml_cuda_mul_mat_q` (the tile width) and `launch_mul_mat_q` (stream-k: one CTA
/// per tile when the tiles fill the waves to 90%, else `sms` CTAs splitting the tiles' K
/// blocks evenly, each share's start moved back to its K iteration's start; a tile is whole
/// when no share starts inside it).
pub(super) fn mmq_whole_tiles(rows: i64, k: i64, n: i64, sms: i64) -> bool {
    if rows % MMQ_I != 0 || k % GROUP as i64 != 0 || n < 1 || sms < 1 {
        return false;
    }
    // ggml takes the first width with the fewest column tiles.
    let mut best = i64::MAX;
    for j in MMQ_J {
        best = best.min((n + j - 1) / j);
    }
    let ntiles = best * (rows / MMQ_I);
    let waves = (ntiles + sms - 1) / sms;
    if 100 * ntiles / (sms * waves) >= 90 {
        return true;
    }
    let per_tile = k / GROUP as i64;
    let total = ntiles * per_tile;
    (1..sms).all(|b| {
        let kbc = b * total / sms;
        let kbc = kbc - (kbc % per_tile) % MMQ_BLOCKS_PER_ITER;
        kbc % per_tile == 0
    })
}

/// Whether [`up_mix`] takes the mixer's `up` (Q8_0 or BF16 `[rank, hc · e]`) times `lo` (F32
/// `[rank, n]`) mixed with `xn` (F32 `[hc · e, n]`): switched on, a 12.x GPU ([`set_gpu`]), 4
/// streams, an `up` in one device's memory, contiguous F32 operands, at least [`MIN_TOKENS`]
/// tokens and, for Q8_0, a step size whose MMQ ggml computes in whole tiles
/// ([`mmq_whole_tiles`]; BF16: [`fits_up_bf16`]).
pub(super) fn fits(up: Tensor<'_>, lo: Tensor<'_>, xn: Tensor<'_>, hc: i64) -> bool {
    let [rank, rows, _, _] = up.ne();
    let n = lo.ne()[1];
    (fits_up(up, n, hc) || fits_up_bf16(up, n, hc))
        && lo.ty() == types::F32
        && lo.ne() == [rank, n, 1, 1]
        && lo.is_contiguous()
        && xn.ty() == types::F32
        && xn.ne() == [rows, n, 1, 1]
        && xn.is_contiguous()
}

/// The part of [`fits`] about `up` and the step's `n` tokens, before `lo` and `xn` exist (as
/// [`up_mix_rebuilt`] needs to know before the norm runs).
pub(super) fn fits_up(up: Tensor<'_>, n: i64, hc: i64) -> bool {
    let sms = sms_for(up) as i64;
    let [rank, rows, u2, u3] = up.ne();
    let e = rows / hc.max(1);
    ON.load(Ordering::Relaxed)
        && sms > 0
        && cfg!(feature = "cuda")
        && hc == HC
        && up.ty() == types::Q8_0
        && u2 == 1
        && u3 == 1
        && up.is_contiguous()
        && !up.is_host()
        && !up.is_meta()
        && rank % GROUP as i64 == 0
        && (GROUP as i64..=MAX_RANK).contains(&rank)
        && rows == hc * e
        && e % ROWS == 0
        && n >= MIN_TOKENS
        && mmq_whole_tiles(rows, rank, n, sms)
}

/// The part of [`fits`] for a BF16 `up` (the NVFP4 checkpoints, #836): switched on (both
/// switches), a 12.x GPU, 4 streams, contiguous rows of a multiple of 16 values up to
/// [`MAX_RANK_BF16`] in one device's memory, and at least [`MIN_TOKENS`] tokens. ggml runs this
/// matmul on cuBLAS, whose GEMM at the mixer's shape (K = rank, at most a few hundred, and
/// `hc · e` rows) computes each output tile in one CTA, the products in K order: the sums the
/// kernel adds.
pub(super) fn fits_up_bf16(up: Tensor<'_>, n: i64, hc: i64) -> bool {
    let sms = sms_for(up);
    let [rank, rows, u2, u3] = up.ne();
    let e = rows / hc.max(1);
    ON.load(Ordering::Relaxed)
        && BF16_ON.load(Ordering::Relaxed)
        && sms > 0
        && cfg!(feature = "cuda")
        && hc == HC
        && up.ty() == types::BF16
        && u2 == 1
        && u3 == 1
        && up.is_contiguous()
        && !up.is_host()
        && !up.is_meta()
        && rank % 16 == 0
        && (16..=MAX_RANK_BF16).contains(&rank)
        && rows == hc * e
        && e % ROWS == 0
        && n >= MIN_TOKENS
        // The kernel stages the rows 16 bytes at a time.
        // SAFETY: a live tensor; reads a field.
        && unsafe { (*up.raw()).data as usize }.is_multiple_of(16)
}

/// Where [`up_mix`]'s parts sit in its F32 node, in bytes: `[mixed | lq | ld]`, the scratch
/// parts 256 bytes aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Layout {
    /// `lo`'s quants, `[rank / 32][n][32]` int8.
    pub lq: usize,
    /// `lo`'s scales, `[rank / 32][n]` F32.
    pub ld: usize,
    /// The node's size in floats.
    pub floats: usize,
}

impl Layout {
    pub(super) fn new(e: usize, rank: usize, n: usize) -> Self {
        let lq = (e * n * 4).next_multiple_of(256);
        let ld = (lq + rank * n).next_multiple_of(256);
        Layout {
            lq,
            ld,
            floats: (ld + rank / GROUP * n * 4).div_ceil(4),
        }
    }
}

/// `mixed = scale · Σ_c xn[c] · sigmoid(gate[c])` with `gate = mul_mat(up, lo)` (F32 `[e, n]`),
/// as ggml's `mul_mat` and `hc_pre_gated(xn, gate, scale)` compute it: `up` Q8_0 or BF16
/// `[rank, hc · e]`, `lo` F32 `[rank, n]`, `xn` F32 `[hc · e, n]`, both contiguous (see [`fits`]).
pub(super) fn up_mix<'a>(
    ctx: &'a Context,
    up: Tensor<'a>,
    lo: Tensor<'a>,
    xn: Tensor<'a>,
    hc: i64,
    scale: f32,
) -> Tensor<'a> {
    let [rank, rows, _, _] = up.ne();
    let n = lo.ne()[1];
    let e = rows / hc;
    let bf16 = up.ty() == types::BF16;
    assert!(up.ty() == types::Q8_0 || bf16, "hc_up_mix: Q8_0 or BF16 up");
    assert_eq!(rows, hc * e, "hc_up_mix: up rows");
    let k_unit = if bf16 { 16 } else { GROUP as i64 };
    assert_eq!(rank % k_unit, 0, "hc_up_mix: rank");
    assert_eq!(lo.ne(), [rank, n, 1, 1], "hc_up_mix: lo");
    assert_eq!(xn.ne(), [rows, n, 1, 1], "hc_up_mix: xn");
    for t in [lo, xn] {
        assert_eq!(t.ty(), types::F32, "hc_up_mix: F32 operands");
        assert!(t.is_contiguous(), "hc_up_mix: contiguous operands");
    }
    if bf16 {
        // ggml's two nodes' places: the gate's for `lo` as BF16, then `[e, n]` (see HC_UP_LOB).
        let lob = ctx.external(&HC_UP_LOB, types::F32, [rows, n, 1, 1], &[lo]);
        return ctx.external_with(
            &HC_UP_MIX,
            types::F32,
            [e, n, 1, 1],
            &[up, lob, xn],
            &[scale.to_bits() as i32, hc as i32],
        );
    }
    node(ctx, &[up, lo, xn], e, rank, n, hc, scale)
}

/// The F32 `lo` (`[rank, n]`) a BF16 mix node reads: its `lob` node's source.
///
/// # Safety
/// `lo` is a node's `src[1]`, made by [`up_mix`].
unsafe fn lo_of(up: &ffi::ggml_tensor, lo: &ffi::ggml_tensor) -> *const ffi::ggml_tensor {
    if up.type_ == types::BF16 {
        lo.src[0]
    } else {
        lo
    }
}

unsafe fn cpu_hc_up_lob(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `up_mix`: src[0] F32 `[rank, n]` contiguous, dst at least
    // `rank · n` BF16 values, host memory.
    unsafe {
        let lo = &*dst.src[0];
        let m = ffi::ggml_nelements(lo) as usize;
        let (x, y) = (lo.data as *const f32, dst.data as *mut u16);
        for i in (ith..m).step_by(nth) {
            *y.add(i) = crate::model::nvfp4_mm::bf16_bits(*x.add(i));
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_up_lob(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `up_mix`, device tensors of the stream's device.
    unsafe {
        let lo = &*dst.src[0];
        let mut p = HcUpMix::empty();
        p.lo = lo.data as *const f32;
        (p.rank, p.n) = (lo.ne[0] as i32, lo.ne[1] as i32);
        p.lob = dst.data as *mut u16;
        match legion_hc_up_lob_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("hc_up_lob: error {e}")),
        }
    }
}

/// [`up_mix`] with `xn` rebuilt from what [`super::hc_norm::hc_post_norm_rebuilt`] writes in
/// its place, `xn = (res · scales) · gamma` per value as the norm computes it (the same bits):
/// `res` the new streams (F32 `[e, hc, n]`), `scales` each stream's norm scale (F32 `[hc, n]`),
/// `gamma` the norm's weight (F32 `[e · hc]`), all contiguous. Saves the norm writing `xn` and
/// this op reading it back: `res` is read instead (#526). `res` may also be BF16, the streams
/// of `legion serve --hc-streams-bf16` (#826) or, for a BF16 `up`, `--hc-bf16` (#1183): each
/// value is widened to F32 first.
#[allow(clippy::too_many_arguments)]
pub(super) fn up_mix_rebuilt<'a>(
    ctx: &'a Context,
    up: Tensor<'a>,
    lo: Tensor<'a>,
    res: Tensor<'a>,
    scales: Tensor<'a>,
    gamma: Tensor<'a>,
    hc: i64,
    scale: f32,
) -> Tensor<'a> {
    let [rank, rows, _, _] = up.ne();
    let n = lo.ne()[1];
    let e = rows / hc;
    let bf16 = up.ty() == types::BF16;
    assert!(up.ty() == types::Q8_0 || bf16, "hc_up_mix: Q8_0 or BF16 up");
    assert_eq!(rows, hc * e, "hc_up_mix: up rows");
    let k_unit = if bf16 { 16 } else { GROUP as i64 };
    assert_eq!(rank % k_unit, 0, "hc_up_mix: rank");
    assert_eq!(lo.ne(), [rank, n, 1, 1], "hc_up_mix: lo");
    assert_eq!(res.ne(), [e, hc, n, 1], "hc_up_mix: streams");
    assert_eq!(scales.ne(), [hc, n, 1, 1], "hc_up_mix: scales");
    assert_eq!(gamma.elements(), rows as usize, "hc_up_mix: gamma");
    assert!(
        res.ty() == types::F32 || res.ty() == types::BF16,
        "hc_up_mix: F32 or BF16 streams"
    );
    assert!(res.is_contiguous(), "hc_up_mix: contiguous operands");
    for t in [lo, scales, gamma] {
        assert_eq!(t.ty(), types::F32, "hc_up_mix: F32 operands");
        assert!(t.is_contiguous(), "hc_up_mix: contiguous operands");
    }
    if bf16 {
        // The two nodes of `up_mix`'s BF16 form, the mix node reading the streams, scales and
        // gamma in place of `xn`. `res` and the scales are views of the norm's node, as `xn`
        // is, so ggml's allocator frees and places every tensor as it does for `up_mix` (#836).
        // BF16 streams: `legion serve --hc-bf16` (#1183).
        let lob = ctx.external(&HC_UP_LOB, types::F32, [rows, n, 1, 1], &[lo]);
        return ctx.external_with(
            &HC_UP_MIX,
            types::F32,
            [e, n, 1, 1],
            &[up, lob, res, scales, gamma],
            &[scale.to_bits() as i32, hc as i32],
        );
    }
    node(ctx, &[up, lo, res, scales, gamma], e, rank, n, hc, scale)
}

/// The op's node for `src` (`[up, lo, xn]` or `[up, lo, res, scales, gamma]`) and its mixed
/// values' view.
fn node<'a>(
    ctx: &'a Context,
    src: &[Tensor<'a>],
    e: i64,
    rank: i64,
    n: i64,
    hc: i64,
    scale: f32,
) -> Tensor<'a> {
    let at = Layout::new(e as usize, rank as usize, n as usize);
    let node = ctx.external_with(
        &HC_UP_MIX,
        types::F32,
        [at.floats as i64, 1, 1, 1],
        src,
        &[scale.to_bits() as i32, hc as i32],
    );
    ctx.view_2d(node, [e, n], e as usize * 4, 0)
}

/// Quantizes the 32 values of `x` as ggml's `quantize_mmq_q8_1` does (scale `1 / (127 /
/// amax)`); an all-zero group gets scale 0 and quants 0, as on the GPU.
pub(super) fn quantize_group(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d_inv = 127.0 / amax;
    for (o, v) in q.iter_mut().zip(x) {
        // NaN (an all-zero group) converts to 0.
        *o = (v * d_inv).round() as i8;
    }
    1.0 / d_inv
}

unsafe fn cpu_hc_up_mix(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `up_mix`, host tensors. The reference: MMQ's sums per block of 32
    // (integer products times both scales) in K order, then ggml's gated mix.
    unsafe {
        let (up, xn) = (&*dst.src[0], &*dst.src[2]);
        let lo = &*lo_of(up, &*dst.src[1]);
        let params = crate::ggml::external_params(dst);
        let scale = f32::from_bits(params[0] as u32);
        let hc = params[1] as usize;
        let (rank, rows) = (up.ne[0] as usize, up.ne[1] as usize);
        let n = lo.ne[1] as usize;
        let e = rows / hc;
        let nkb = rank / GROUP;
        let lop = lo.data as *const f32;
        // `xn`, or (`up_mix_rebuilt`) the streams with the scales and gamma to rebuild it from;
        // BF16 streams (#826) widened to F32.
        let xp = xn.data as *const f32;
        let x16 = (xn.type_ == types::BF16).then_some(xn.data as *const u16);
        let xv_at = |at: usize| match x16 {
            Some(p) => f32::from_bits(u32::from(*p.add(at)) << 16),
            None => *xp.add(at),
        };
        let rebuilt = rebuilt(dst).map(|(s, g)| ((*s).data as *const f32, (*g).data as *const f32));
        let out = dst.data as *mut f32;
        // block_q8_0: an F16 scale, 32 int8.
        const Q8_0: usize = 34;
        let up_bf16 = up.type_ == types::BF16;
        let bf16 = |b: u16| f32::from_bits(u32::from(b) << 16);
        // BF16: `lo` as BF16 from the `hc_up_lob` node (src[1]), as on CUDA. The F32 `lo` is
        // no source of this node: the allocator may have handed its memory on (to `dst`).
        let lob = (*dst.src[1]).data as *const u16;
        let mut q = vec![0i8; rank];
        let mut d = vec![0f32; nkb];
        let mut xb = vec![0f32; rank];
        let mut gate = vec![0f32; rows];
        for t in (ith..n).step_by(nth) {
            if up_bf16 {
                // ggml's cuBLAS path: `lo` rounded to BF16, F32 sums of the products in K order.
                for (k, o) in xb.iter_mut().enumerate() {
                    *o = bf16(*lob.add(t * rank + k));
                }
                for (r, g) in gate.iter_mut().enumerate() {
                    let row = (up.data as *const u8).add(r * up.nb[1]) as *const u16;
                    let mut sum = 0f32;
                    for (k, v) in xb.iter().enumerate() {
                        sum += bf16(row.add(k).read_unaligned()) * v;
                    }
                    *g = sum;
                }
            } else {
                let x = std::slice::from_raw_parts(lop.add(t * rank), rank);
                for b in 0..nkb {
                    d[b] = quantize_group(&x[b * GROUP..][..GROUP], &mut q[b * GROUP..][..GROUP]);
                }
                for (r, g) in gate.iter_mut().enumerate() {
                    let row = (up.data as *const u8).add(r * up.nb[1]);
                    let mut sum = 0f32;
                    for b in 0..nkb {
                        let wb = row.add(b * Q8_0);
                        let dw = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*wb, *wb.add(1)]));
                        let mut isum = 0i32;
                        for i in 0..GROUP {
                            isum += (*wb.add(2 + i) as i8 as i32) * q[b * GROUP + i] as i32;
                        }
                        sum += isum as f32 * dw * d[b];
                    }
                    *g = sum;
                }
            }
            for i in 0..e {
                let mut sum = 0f32;
                for c in 0..hc {
                    let mut xv = xv_at(t * rows + c * e + i);
                    if let Some((sc, g)) = rebuilt {
                        xv = (xv * *sc.add(t * hc + c)) * *g.add(c * e + i);
                    }
                    sum += xv * (1.0 / (1.0 + (-gate[c * e + i]).exp()));
                }
                *out.add(t * e + i) = scale * sum;
            }
        }
    }
}

/// For a node made by [`up_mix_rebuilt`], its scales and gamma (`src[3]`, `src[4]`; `src[2]`
/// is then the streams); `None` for [`up_mix`]'s (`src[2]` is `xn`).
fn rebuilt(dst: &ffi::ggml_tensor) -> Option<(*const ffi::ggml_tensor, *const ffi::ggml_tensor)> {
    (!dst.src[3].is_null()).then(|| (dst.src[3] as *const _, dst.src[4] as *const _))
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct HcUpMix {
    up: *const u8,
    up_nb1: i64,
    lo: *const f32,
    xn: *const f32,
    mixed: *mut f32,
    lq: *mut i8,
    ld: *mut f32,
    e: i32,
    hc: i32,
    rank: i32,
    n: i32,
    scale: f32,
    /// Null, or (with `sc` and `g`) the streams `xn` is rebuilt from ([`up_mix_rebuilt`]).
    res: *const f32,
    sc: *const f32,
    g: *const f32,
    /// Nonzero: `res` holds BF16 values (#826).
    res_bf16: i32,
    /// Nonzero: `up` holds BF16 rows (#836); `lob` is then the scratch for `lo` as BF16.
    up_bf16: i32,
    lob: *mut u16,
    /// Nonzero: `lob` already holds `lo` as BF16 (the `hc_up_lob` node).
    lob_ready: i32,
}

#[cfg(feature = "cuda")]
impl HcUpMix {
    fn empty() -> Self {
        HcUpMix {
            up: std::ptr::null(),
            up_nb1: 0,
            lo: std::ptr::null(),
            xn: std::ptr::null(),
            mixed: std::ptr::null_mut(),
            lq: std::ptr::null_mut(),
            ld: std::ptr::null_mut(),
            e: 0,
            hc: 0,
            rank: 0,
            n: 0,
            scale: 0.0,
            res: std::ptr::null(),
            sc: std::ptr::null(),
            g: std::ptr::null(),
            res_bf16: 0,
            up_bf16: 0,
            lob: std::ptr::null_mut(),
            lob_ready: 0,
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_up_mix_launch(p: *const HcUpMix, stream: *mut c_void) -> i32;
    fn legion_hc_up_lob_launch(p: *const HcUpMix, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_up_mix(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `up_mix`, device tensors of the stream's device.
    unsafe {
        let (up, xn) = (&*dst.src[0], &*dst.src[2]);
        let lo = &*lo_of(up, &*dst.src[1]);
        let params = crate::ggml::external_params(dst);
        let hc = params[1];
        let (rank, rows, n) = (up.ne[0], up.ne[1], lo.ne[1]);
        let e = rows / hc as i64;
        let up_bf16 = up.type_ == types::BF16;
        let at = Layout::new(e as usize, rank as usize, n as usize);
        let base = dst.data as *mut u8;
        let null = std::ptr::null::<f32>();
        let (res, sc, g) = match rebuilt(dst) {
            Some((s, g)) => (
                xn.data as *const f32,
                (*s).data as *const f32,
                (*g).data as *const f32,
            ),
            None => (null, null, null),
        };
        let p = HcUpMix {
            up: up.data as *const u8,
            up_nb1: up.nb[1] as i64,
            lo: lo.data as *const f32,
            xn: if res.is_null() {
                xn.data as *const f32
            } else {
                null
            },
            mixed: dst.data as *mut f32,
            // Unused for BF16 (the node is the mixed values only).
            lq: if up_bf16 {
                std::ptr::null_mut()
            } else {
                base.add(at.lq) as *mut i8
            },
            ld: if up_bf16 {
                std::ptr::null_mut()
            } else {
                base.add(at.ld) as *mut f32
            },
            e: e as i32,
            hc,
            rank: rank as i32,
            n: n as i32,
            scale: f32::from_bits(params[0] as u32),
            res,
            sc,
            g,
            res_bf16: i32::from(!res.is_null() && xn.type_ == types::BF16),
            up_bf16: i32::from(up_bf16),
            // BF16: the `hc_up_lob` node (src[1]) holds `lo` as BF16 already.
            lob: if up_bf16 {
                (*dst.src[1]).data as *mut u16
            } else {
                std::ptr::null_mut()
            },
            lob_ready: i32::from(up_bf16),
        };
        match legion_hc_up_mix_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("hc_up_mix: error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
