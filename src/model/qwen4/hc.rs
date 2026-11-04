//! qwen4exp's hyper-connection mixer as one external op ([`crate::ggml::ExternalOp`]) for small
//! steps on CUDA (`src/cuda/qwen4_hc.cu`): what [`super`]'s `hc_mix` builds from about a dozen
//! ggml nodes (grouped RMS norm, the down matvec, scale and SiLU, the up matvec, the gated mix,
//! the inject matvec), in two kernels. A decode step runs 96 of these mixers; as nodes they are
//! mostly launches of tiny kernels and matvecs too small to stream their weights.
//!
//! Numerics follow ggml's: the matvecs quantize their activations to Q8_1 as ggml's mmvq does
//! and use its Q8_0 dot products, the norm and the mix use the same formulas. The reduction
//! orders differ, so results agree to rounding, not bit for bit.
//!
//! BF16 down and up matrices (the NVFP4 checkpoint's mixers) take the op only with
//! [`set_mix_bf16`] (`legion serve --hc-mix-bf16`): F32 activations against the BF16 weights as
//! ggml's mmvf does, in another summation order, so it is opt-in (#748).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::q8_group;

static HC_MIX: ExternalOp = ExternalOp {
    name: "qwen4_hc_mix",
    cpu: cpu_hc_mix,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_mix),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most tokens in a step the op takes.
pub const MAX_TOKENS: i64 = 8;

/// Whether [`fits`] may take a step (on by default).
static MIX_OP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether [`fits`] also takes BF16 down and up matrices (off by default).
static MIX_BF16: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the V100 Q8_0 kernels issue their weight loads ahead (on by default, #814).
static LOADS_AHEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Which weights the V100 Q8_0 kernels read into L2 for the kernel after them ([`L2Prefetch`]).
static L2_AHEAD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Which mixer weights the kernel before their reader fetches into L2 on the Q8_0 path with its
/// loads ahead (the V100's; `legion serve --hc-l2-prefetch`, #814). Extra CTAs of the norm read
/// the down matrix while the norm runs, extra CTAs of `down_ahead` the up matrix while it runs;
/// their reads only warm L2, so every mode gives the same bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum L2Prefetch {
    /// No reads ahead (the kernels before #814's L2 prefetch).
    #[default]
    Off,
    /// The norm reads the down matrix into L2.
    Down,
    /// `down_ahead` reads the up matrix into L2.
    Up,
    /// Both.
    Both,
}

impl L2Prefetch {
    /// The kernels' `l2_ahead` bits (`L2_DOWN` 1, `L2_UP` 2).
    pub fn code(self) -> i32 {
        match self {
            L2Prefetch::Off => 0,
            L2Prefetch::Down => 1,
            L2Prefetch::Up => 2,
            L2Prefetch::Both => 3,
        }
    }
}

/// A kernel variant forced on the Q8_0 path with its loads ahead ([`force_variant`]; negative:
/// the defaults, [`NORM_REGS`], [`DOWN_SPLIT`], [`DOWN_HOIST`] and [`UP_STAGE`]).
static VARIANT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Whether the norm keeps gamma and the stream in registers on compute capability 7.x (on by
/// default, #814; the same bits).
static NORM_REGS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the down kernel splits each row's units over twice the threads at 1-2 tokens on
/// compute capability 7.x (on by default, #814; the same bits).
static DOWN_SPLIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the down kernel runs with its address math out of the loop on compute capability
/// 7.x, in the split's place (on by default, #814; the same bits).
static DOWN_HOIST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the up kernel stages its rows in shared memory on compute capability 7.x (on by
/// default, #814; the same bits).
static UP_STAGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The kernels' variant bits: `V_NORM_REGS` 2, `V_DOWN_SPLIT` 4, `V_DEFAULTS` 8, `V_DOWN_HOIST`
/// 16, `V_UP_STAGE` 32 (with `V_DEFAULTS` the bits apply only where the defaults' measurements
/// hold: compute capability 7.x, the split at <= 2 tokens and only without the hoist).
pub fn variant_code(
    forced: i32,
    norm_regs: bool,
    down_split: bool,
    down_hoist: bool,
    up_stage: bool,
) -> i32 {
    if forced >= 0 {
        return forced;
    }
    8 | (2 * i32::from(norm_regs))
        | (4 * i32::from(down_split))
        | (16 * i32::from(down_hoist))
        | (32 * i32::from(up_stage))
}

/// Sets whether the V100 norm keeps the stream in registers (`legion serve --no-hc-norm-regs`
/// turns it off; the same bits either way).
pub fn set_norm_regs(on: bool) {
    NORM_REGS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Sets whether the V100 down kernel splits its rows over twice the threads at 1-2 tokens
/// (`legion serve --no-hc-down-split` turns it off; the same bits either way).
pub fn set_down_split(on: bool) {
    DOWN_SPLIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Sets whether the V100 down kernel runs with its address math out of the loop, in the split's
/// place (`legion serve --no-hc-down-hoist` turns it off; the same bits either way).
pub fn set_down_hoist(on: bool) {
    DOWN_HOIST.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Sets whether the V100 up kernel stages its rows in shared memory (`legion serve
/// --no-hc-up-stage` turns it off; the same bits either way).
pub fn set_up_stage(on: bool) {
    UP_STAGE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`combine_mix`] writes the block input's Q8_1 blocks wherever the Q8_0 kernels with
/// their loads ahead run, not only in stage graphs ([`bench`], so that it runs the decode's
/// kernels outside a server).
static FORCE_QUANT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Forces a kernel variant of the Q8_0 path with its loads ahead ([`bench`]): 0 the kernels
/// before #814's variants, bits as [`variant_code`]'s, negative the defaults.
pub fn force_variant(v: i32) {
    VARIANT.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// See [`FORCE_QUANT`].
pub fn force_quant(on: bool) {
    FORCE_QUANT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Sets which weights the mixer's V100 kernels read into L2 ahead of their reader.
pub fn set_mix_l2_prefetch(mode: L2Prefetch) {
    L2_AHEAD.store(mode.code(), std::sync::atomic::Ordering::Relaxed);
}

/// Whether the op's Q8_0 kernels (not the GB10's) load each thread's weights before its dot
/// products and make lo's Q8_1 inside the up kernel (on, the default), or one block at a time
/// with a separate quantize kernel (off). The same sums in the same order either way, so the
/// same bits: off is for checking that.
pub fn set_mix_loads_ahead(on: bool) {
    LOADS_AHEAD.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether graphs built from now on may take [`mix`]; off keeps the ggml graph (for A/B runs,
/// `legion serve --hc-mix-ggml`).
pub fn set_mix_op(on: bool) {
    MIX_OP.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether graphs built from now on may take [`mix`] with BF16 down and up matrices too (the
/// NVFP4 checkpoint's mixers; `legion serve --hc-mix-bf16`, #748). Off by default: the op's
/// sums run in another order than ggml's mmvf, so the outputs move by rounding.
pub fn set_mix_bf16(on: bool) {
    MIX_BF16.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the down and up matrices are ones the op reads: both Q8_0 (the kernels' tiles: a
/// down row in 16 KiB, the up rows of 8 columns in 12 KiB), or with [`set_mix_bf16`] both BF16.
fn weights_fit(down: Tensor<'_>, up: Tensor<'_>, k: i64, hc: i64) -> bool {
    let rank = down.ne()[1];
    if down.ty() == types::Q8_0 && up.ty() == types::Q8_0 {
        return k / 32 * 34 <= 16 * 1024 && hc * 8 * (rank / 32 * 34) <= 12 * 1024;
    }
    MIX_BF16.load(std::sync::atomic::Ordering::Relaxed)
        && down.ty() == types::BF16
        && up.ty() == types::BF16
}

/// Whether [`mix`] takes these weights and this input: Q8_0 (or, opted in, BF16) down and up
/// matrices in device memory (on the CPU the graph is the reference), an F32 norm and inject,
/// rows of the input contiguous, at most [`MAX_TOKENS`] tokens.
pub fn fits(
    x: Tensor<'_>,
    norm: Tensor<'_>,
    down: Tensor<'_>,
    up: Tensor<'_>,
    inject: Option<Tensor<'_>>,
) -> bool {
    let [e, hc, n, _] = x.ne();
    let k = e * hc;
    MIX_OP.load(std::sync::atomic::Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && !down.is_host()
        && weights_fit(down, up, k, hc)
        && x.ty() == types::F32
        && x.nb()[0] == 4
        && norm.ty() == types::F32
        && norm.elements() as i64 == k
        && down.ne()[0] == k
        && up.ne()[1] == k
        && up.ne()[0] == down.ne()[1]
        && down.ne()[1] % 64 == 0
        && down.ne()[1] <= 2048
        && k % 32 == 0
        && (1..=8).contains(&hc)
        && n <= MAX_TOKENS
        && inject.is_none_or(|w| w.ty() == types::F32 && w.ne()[0] == k && w.ne()[1] == hc)
}

/// The mixer for `x` (`[n_embd, hc, n]`): `(mixed [n_embd, n], inject [hc, n] if inject_w)`.
pub fn mix<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    norm: Tensor<'a>,
    down: Tensor<'a>,
    up: Tensor<'a>,
    inject_w: Option<Tensor<'a>>,
    eps: f32,
) -> (Tensor<'a>, Option<Tensor<'a>>) {
    let (mixed, inject, _) = combine_mix(ctx, x, None, norm, down, up, inject_w, eps);
    (mixed, inject)
}

/// [`mix`] after `hc_combine` of the previous block's output: with `prev = (out [n_embd, n],
/// inject_prev [hc, n])` the residual streams become `out * 2 sigmoid(inject_prev / hc) + x`,
/// as `hc_combine` computes them, and are returned as well (`[n_embd, hc, n]`); without, `x`
/// itself. One op instead of the combine's four kernels and the mixer's dozen.
#[allow(clippy::too_many_arguments)]
pub fn combine_mix<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    prev: Option<(Tensor<'a>, Tensor<'a>)>,
    norm: Tensor<'a>,
    down: Tensor<'a>,
    up: Tensor<'a>,
    inject_w: Option<Tensor<'a>>,
    eps: f32,
) -> (Tensor<'a>, Option<Tensor<'a>>, Tensor<'a>) {
    let [e, hc, n, _] = x.ne();
    let rank = down.ne()[1];
    // One scratch buffer: xn and lo in F32, then both as Q8_1 (see `Scratch`).
    let scratch = ctx
        .new_tensor(types::I8, &[Scratch::new(e, hc, rank, n).bytes as i64])
        .named("hc_mix_scratch")
        .as_input();
    let mut srcs = vec![x, norm, down, up, scratch];
    srcs.extend(inject_w);
    if let Some((out, inj)) = prev {
        srcs.extend([out, inj]);
    }
    let n_inj = if inject_w.is_some() { hc } else { 0 };
    let n_res = if prev.is_some() { e * hc } else { 0 };
    // The block input's Q8_1 blocks too, for its Q8_0 matvecs (#819).
    let quant = quant_fits(e, down, up);
    let n_q = if quant {
        q8_group::quantized_elements(e, n)
    } else {
        0
    };
    let node = ctx.external_with(
        &HC_MIX,
        types::F32,
        [(e + n_inj + n_res) * n + n_q, 1, 1, 1],
        &srcs,
        &[
            eps.to_bits() as i32,
            inject_w.is_some() as i32,
            prev.is_some() as i32,
            quant as i32,
        ],
    );
    let f = 4usize;
    let mixed = ctx.view_2d(node, [e, n], e as usize * f, 0);
    if quant {
        let at = ((e + n_inj + n_res) * n) as usize * f;
        q8_group::set_quantized(ctx, mixed, ctx.view_1d(node, n_q, at));
    }
    let inject =
        inject_w.map(|_| ctx.view_2d(node, [hc, n], hc as usize * f, (e * n) as usize * f));
    let res = if prev.is_some() {
        let at = ((e + n_inj) * n) as usize * f;
        ctx.view_3d(
            node,
            [e, hc, n],
            [e as usize * f, (e * hc) as usize * f],
            at,
        )
    } else {
        x
    };
    (mixed, inject, res)
}

/// The op's scratch: byte offsets of xn (F32), lo (F32), xn as Q8_1 and lo as Q8_1.
/// Whether the op writes the block input's Q8_1 blocks beside it ([`q8_group::set_quantized`],
/// #819): where the block input's Q8_0 matvecs take [`q8_group`] (on, a V100-class GPU, stage
/// graphs) and the mixer runs its Q8_0 kernels with their loads ahead, whose up_mix writes them.
fn quant_fits(e: i64, down: Tensor<'_>, up: Tensor<'_>) -> bool {
    down.ty() == types::Q8_0
        && up.ty() == types::Q8_0
        && e % 32 == 0
        && LOADS_AHEAD.load(std::sync::atomic::Ordering::Relaxed)
        && (FORCE_QUANT.load(std::sync::atomic::Ordering::Relaxed)
            || (q8_group::enabled() && q8_group::part("qwen4exp", q8_group::PART_HC)))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Scratch {
    lo: usize,
    xq: usize,
    lq: usize,
    bytes: usize,
}

impl Scratch {
    fn new(e: i64, hc: i64, rank: i64, n: i64) -> Self {
        let (k, rank, n) = ((e * hc) as usize, rank as usize, n as usize);
        let lo = k * n * 4;
        let xq = lo + rank * n * 4;
        let lq = xq + k / 32 * 36 * n;
        Self {
            lo,
            xq,
            lq,
            bytes: lq + rank / 32 * 36 * n,
        }
    }
}

/// What the kernels read from a node made by [`combine_mix`].
struct Operands<'a> {
    x: &'a ffi::ggml_tensor,
    norm: &'a ffi::ggml_tensor,
    down: &'a ffi::ggml_tensor,
    up: &'a ffi::ggml_tensor,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    scratch: &'a ffi::ggml_tensor,
    inject_w: Option<&'a ffi::ggml_tensor>,
    prev: Option<(&'a ffi::ggml_tensor, &'a ffi::ggml_tensor)>,
    /// Whether the node ends with the block input's Q8_1 blocks ([`quant_fits`]).
    quant: bool,
    e: usize,
    hc: usize,
    n: usize,
    rank: usize,
    eps: f32,
}

impl<'a> Operands<'a> {
    /// # Safety
    /// `dst` must be a node made by [`combine_mix`].
    unsafe fn of(dst: &'a ffi::ggml_tensor) -> Self {
        unsafe {
            let src = |i: usize| &*dst.src[i];
            let params = crate::ggml::external_params(dst);
            let (x, down) = (src(0), src(2));
            let mut next = 5;
            let inject_w = (params[1] != 0).then(|| {
                next += 1;
                src(next - 1)
            });
            let prev = (params[2] != 0).then(|| (src(next), src(next + 1)));
            Self {
                x,
                norm: src(1),
                down,
                up: src(3),
                scratch: src(4),
                inject_w,
                prev,
                quant: params[3] != 0,
                e: x.ne[0] as usize,
                hc: x.ne[1] as usize,
                n: x.ne[2] as usize,
                rank: down.ne[1] as usize,
                eps: f32::from_bits(params[0] as u32),
            }
        }
    }

    /// Where the output's parts start, in floats: (inject, res).
    fn out_at(&self) -> (usize, usize) {
        let inj = self.e * self.n;
        let n_inj = if self.inject_w.is_some() { self.hc } else { 0 };
        (inj, inj + n_inj * self.n)
    }

    /// Where the block input's Q8_1 blocks start in the node (F32 elements), with `quant`.
    fn q_at(&self) -> usize {
        let (_, res_at) = self.out_at();
        res_at
            + if self.prev.is_some() {
                self.e * self.hc * self.n
            } else {
                0
            }
    }
}

/// The CPU reference: the same formulas in F32 with dequantized weights (no Q8_1 activations).
unsafe fn cpu_hc_mix(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `mix`, every tensor in host memory.
    unsafe {
        let o = Operands::of(dst);
        let k = o.e * o.hc;
        let f32s =
            |t: &ffi::ggml_tensor, n: usize| std::slice::from_raw_parts(t.data as *const f32, n);
        let rows = |t: &ffi::ggml_tensor, r: usize, len: usize| {
            let b = std::slice::from_raw_parts((t.data as *const u8).add(r * t.nb[1]), t.nb[1]);
            crate::ggml::dequantize(t.type_, b, len)
        };
        let norm = f32s(o.norm, k);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut f32, dst.ne[0] as usize);
        let (inj_at, res_at) = o.out_at();
        for t in 0..o.n {
            let mut xn = vec![0f32; k];
            for c in 0..o.hc {
                let mut xc = std::slice::from_raw_parts(
                    (o.x.data as *const u8).add(t * o.x.nb[2] + c * o.x.nb[1]) as *const f32,
                    o.e,
                )
                .to_vec();
                if let Some((po, pi)) = o.prev {
                    let po = std::slice::from_raw_parts(
                        (po.data as *const u8).add(t * po.nb[1]) as *const f32,
                        o.e,
                    );
                    let v = *((pi.data as *const u8).add(t * pi.nb[1]) as *const f32).add(c)
                        * (1.0 / o.hc as f32);
                    let post = 2.0 / (1.0 + (-v).exp());
                    for (i, x) in xc.iter_mut().enumerate() {
                        *x += po[i] * post;
                        out[res_at + (t * o.hc + c) * o.e + i] = *x;
                    }
                }
                let ss: f32 = xc.iter().map(|v| v * v).sum();
                let s = 1.0 / (ss / o.e as f32 + o.eps).sqrt();
                for i in 0..o.e {
                    xn[c * o.e + i] = s * xc[i] * norm[c * o.e + i];
                }
            }
            let lo: Vec<f32> = (0..o.rank)
                .map(|r| {
                    let w = rows(o.down, r, k);
                    let v = w.iter().zip(&xn).map(|(a, b)| a * b).sum::<f32>() / o.hc as f32;
                    v / (1.0 + (-v).exp())
                })
                .collect();
            for i in 0..o.e {
                let mut sum = 0f32;
                for c in 0..o.hc {
                    let w = rows(o.up, c * o.e + i, o.rank);
                    let g: f32 = w.iter().zip(&lo).map(|(a, b)| a * b).sum();
                    sum += xn[c * o.e + i] / (1.0 + (-g).exp());
                }
                out[t * o.e + i] = sum / o.hc as f32;
            }
            if let Some(w) = o.inject_w {
                for h in 0..o.hc {
                    let row = std::slice::from_raw_parts(
                        (w.data as *const u8).add(h * w.nb[1]) as *const f32,
                        k,
                    );
                    out[inj_at + t * o.hc + h] = row.iter().zip(&xn).map(|(a, b)| a * b).sum();
                }
            }
        }
        if o.quant {
            let mixed = out[..o.e * o.n].to_vec();
            let q = (dst.data as *mut f32).add(o.q_at()) as *mut u8;
            q8_group::write_q8_1(|t| &mixed[t * o.e..(t + 1) * o.e], o.e, o.n, q);
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LaunchParams {
    x: *const f32,
    x_nb1: i64,
    x_nb2: i64,
    out: *const f32,
    out_nb1: i64,
    inject_prev: *const f32,
    inject_prev_nb1: i64,
    res: *mut f32,
    norm: *const f32,
    down: *const u8,
    down_nb1: i64,
    up: *const u8,
    up_nb1: i64,
    inject_w: *const f32,
    inject_nb1: i64,
    xn: *mut f32,
    xq: *mut u8,
    lo: *mut f32,
    lq: *mut u8,
    mixed: *mut f32,
    inject: *mut f32,
    n_embd: i32,
    hc: i32,
    rank: i32,
    n: i32,
    eps: f32,
    bf16: i32,
    ahead: i32,
    mixed_q: *mut u8,
    l2_ahead: i32,
    variant: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_mix_launch(p: *const LaunchParams, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_mix(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mix`; its tensors in device memory.
    let o = unsafe { Operands::of(dst) };
    let out = dst.data as *mut f32;
    let (inj_at, res_at) = o.out_at();
    let sc = Scratch::new(o.e as i64, o.hc as i64, o.rank as i64, o.n as i64);
    let base = o.scratch.data as *mut u8;
    // SAFETY: offsets inside the op's own output and scratch.
    let (res, inject, lo, xq, lq) = unsafe {
        (
            out.add(res_at),
            out.add(inj_at),
            base.add(sc.lo) as *mut f32,
            base.add(sc.xq),
            base.add(sc.lq),
        )
    };
    let p = LaunchParams {
        x: o.x.data as *const f32,
        x_nb1: o.x.nb[1] as i64,
        x_nb2: o.x.nb[2] as i64,
        out: o
            .prev
            .map_or(std::ptr::null(), |(t, _)| t.data as *const f32),
        out_nb1: o.prev.map_or(0, |(t, _)| t.nb[1] as i64),
        inject_prev: o
            .prev
            .map_or(std::ptr::null(), |(_, t)| t.data as *const f32),
        inject_prev_nb1: o.prev.map_or(0, |(_, t)| t.nb[1] as i64),
        res,
        norm: o.norm.data as *const f32,
        down: o.down.data as *const u8,
        down_nb1: o.down.nb[1] as i64,
        up: o.up.data as *const u8,
        up_nb1: o.up.nb[1] as i64,
        inject_w: o
            .inject_w
            .map_or(std::ptr::null(), |w| w.data as *const f32),
        inject_nb1: o.inject_w.map_or(0, |w| w.nb[1] as i64),
        xn: base as *mut f32,
        xq,
        lo,
        lq,
        mixed: out,
        inject,
        n_embd: o.e as i32,
        hc: o.hc as i32,
        rank: o.rank as i32,
        n: o.n as i32,
        eps: o.eps,
        bf16: (o.down.type_ == types::BF16) as i32,
        ahead: LOADS_AHEAD.load(std::sync::atomic::Ordering::Relaxed) as i32,
        mixed_q: if o.quant {
            // SAFETY: the Q8_1 blocks at the end of the op's own output.
            unsafe { out.add(o.q_at()) as *mut u8 }
        } else {
            std::ptr::null_mut()
        },
        l2_ahead: L2_AHEAD.load(std::sync::atomic::Ordering::Relaxed),
        variant: variant_code(
            VARIANT.load(std::sync::atomic::Ordering::Relaxed),
            NORM_REGS.load(std::sync::atomic::Ordering::Relaxed),
            DOWN_SPLIT.load(std::sync::atomic::Ordering::Relaxed),
            DOWN_HOIST.load(std::sync::atomic::Ordering::Relaxed),
            UP_STAGE.load(std::sync::atomic::Ordering::Relaxed),
        ),
    };
    match unsafe { legion_hc_mix_launch(&p, stream) } {
        0 => Ok(()),
        e => Err(format!(
            "CUDA error {e} (hc mix: n_embd {} hc {} rank {} tokens {})",
            o.e, o.hc, o.rank, o.n
        )),
    }
}

pub mod bench;

#[cfg(test)]
mod tests;
