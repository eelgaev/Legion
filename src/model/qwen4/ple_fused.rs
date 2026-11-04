//! qwen4exp's per-layer n-gram embedding (PLE) after its key and value matmuls, for prompt
//! steps on a GPU, as two external ops ([`crate::ggml::ExternalOp`], CPU references plus CUDA
//! kernels in `src/cuda/ple.cu`) instead of ggml's chain in [`super::Model::ple`] (#439): about
//! thirty nodes, most of them passes over `[e · hc, n]` F32 tensors (168 MB each for a
//! 4,096-token chunk of Qwen3.8-Flash-Next), on the first pipeline stage of a long prompt, the
//! stage the others wait for.
//!
//! - [`gate_norm`]: per token, the key's and the query's (the streams') grouped RMS norms, their
//!   per-stream dot product, the gate (`sigmoid(sgn(s) · sqrt(max(|s| / sqrt(e), 1e-6)))`), the
//!   value times the gate and its grouped RMS norm (the conv's input).
//! - [`conv_out`]: per group of a prompt step, the depthwise causal conv dilated by the n-gram
//!   size over `[history | tokens]`, `silu`, plus the gated value, plus the streams; the
//!   history the next step starts from goes back to the state.
//!
//! On CUDA the kernels compute what ggml-cuda's kernels for the chain compute, bit for bit
//! (`ple_fused/tests.rs`, and over a real model with ggml-cuda's fusions off:
//! `engine/lm/tests/ple_ops_kl.rs`). Opt-in all the same (`legion serve --ple-ops`): the smaller
//! graph places tensors elsewhere, and ggml-cuda fuses an MoE layer's weighted expert sum only
//! where its output does not overlap the router weights; the fused sum adds with FMAs, so a
//! layer near the PLE that flips between the two moves greedy output.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static PLE_GATE: ExternalOp = ExternalOp {
    name: "ple_gate",
    cpu: cpu_ple_gate,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ple_gate),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static PLE_CONV: ExternalOp = ExternalOp {
    name: "ple_conv",
    cpu: cpu_ple_conv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ple_conv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The streams the CUDA kernels take.
pub(super) const PLE_HC: i64 = 4;
/// The narrowest stream the CUDA kernels take (ggml-cuda's rms_norm takes 1,024 threads per
/// row from there; narrower rows have another order of sums).
pub(super) const PLE_MIN_E: i64 = 1024;

/// Whether prompt steps take the two ops where they fit (off by default).
static FUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether graphs built from now on take the PLE ops for prompt steps on a GPU
/// (`legion serve --ple-ops`); off keeps ggml's chain.
pub fn set_fused_ple(on: bool) {
    FUSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// See [`set_fused_ple`].
pub(super) fn fused_ple() -> bool {
    FUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The gate and the conv's input for `n` tokens: `key` (F32 `[e · hc, n]`, rows contiguous),
/// the streams `res` (F32 `[e, hc, n]`, streams contiguous), `value` (F32 `[e, n]`, rows
/// contiguous) and the gammas `gk`, `gq`, `gc` (F32, `e · hc` each). As the chain: `kn =
/// rms_norm(key) · gk`, `qn = rms_norm(res) · gq` per stream, `s = sum(kn · qn) · inv_sqrt_e`,
/// `gate = sigmoid(sgn(s) · sqrt(clamp(|s|, clamp_min, inf)))`, `xn = rms_norm(value · gate) ·
/// gc`. Returns `xn` (`[e · hc, n]`) and `gate` (`[hc, n]`).
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_norm<'a>(
    ctx: &'a Context,
    key: Tensor<'a>,
    res: Tensor<'a>,
    value: Tensor<'a>,
    gammas: [Tensor<'a>; 3],
    eps: f32,
    inv_sqrt_e: f32,
    clamp_min: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    let [e, hc, n, _] = res.ne();
    assert_eq!(key.ne()[..2], [e * hc, n], "ple_gate: key");
    assert_eq!(value.ne()[..2], [e, n], "ple_gate: value");
    for t in [key, res, value] {
        assert_eq!(t.ty(), types::F32, "ple_gate: F32 inputs");
        assert_eq!(t.nb()[0], 4, "ple_gate: rows contiguous");
    }
    assert_eq!(res.nb()[1], e as usize * 4, "ple_gate: streams contiguous");
    for g in gammas {
        assert!(
            g.ty() == types::F32 && g.is_contiguous() && g.elements() == (e * hc) as usize,
            "ple_gate: gammas"
        );
    }
    let node = ctx.external_with(
        &PLE_GATE,
        types::F32,
        [e * hc * n + hc * n, 1, 1, 1],
        &[key, res, value, gammas[0], gammas[1], gammas[2]],
        &[
            eps.to_bits() as i32,
            inv_sqrt_e.to_bits() as i32,
            clamp_min.to_bits() as i32,
        ],
    );
    let xn = ctx.view_2d(node, [e * hc, n], (e * hc) as usize * 4, 0);
    let gate = ctx.view_2d(node, [hc, n], hc as usize * 4, (e * hc * n) as usize * 4);
    (xn, gate)
}

/// One group's conv and output: `x` (F32 `[e · hc, nt · ns]`, [`gate_norm`]'s `xn` for the
/// group's columns), its sequences' conv history `state` (F32 `[hist, e · hc, ns]`, read and
/// then overwritten with the last `hist` inputs), the conv weight `w` (F32 `[kern, e · hc]`,
/// `hist = (kern - 1) · dil`), and the group's `value` (`[e, nt · ns]`), `gate` (`[hc, nt ·
/// ns]`) and streams `res` (`[e · hc, nt · ns]`). Returns `res + (value · gate + silu(conv))`,
/// F32 `[e · hc, nt · ns]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_out<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    state: Tensor<'a>,
    w: Tensor<'a>,
    value: Tensor<'a>,
    gate: Tensor<'a>,
    res: Tensor<'a>,
    nt: i64,
    dil: i64,
) -> Tensor<'a> {
    let [width, m, _, _] = x.ne();
    let [hist, sw, ns, _] = state.ne();
    let [kern, ww, _, _] = w.ne();
    let [e, _, _, _] = value.ne();
    let hc = gate.ne()[0];
    assert!(
        nt >= 1 && m == nt * ns,
        "ple_conv: {m} columns, {nt} x {ns}"
    );
    assert!(
        sw == width && ww == width && e * hc == width,
        "ple_conv: widths"
    );
    assert_eq!(hist, (kern - 1) * dil, "ple_conv: history");
    for (t, cols) in [(value, e), (gate, hc), (res, width)] {
        assert_eq!(t.ne()[..2], [cols, m], "ple_conv: group columns");
    }
    for t in [x, state, w, value, gate, res] {
        assert_eq!(t.ty(), types::F32, "ple_conv: F32 tensors");
        assert_eq!(t.nb()[0], 4, "ple_conv: rows contiguous");
    }
    assert!(w.is_contiguous(), "ple_conv: weight contiguous");
    assert_eq!(gate.nb()[1], hc as usize * 4, "ple_conv: gates contiguous");
    ctx.external_with(
        &PLE_CONV,
        types::F32,
        [width, m, 1, 1],
        &[x, state, w, value, gate, res],
        &[nt as i32, dil as i32],
    )
}

/// The `i`-th element of row `row` of `t` (rows `nb[1]` apart, elements contiguous).
///
/// # Safety
/// `t` is a host F32 tensor holding that element.
unsafe fn at(t: &ffi::ggml_tensor, row: usize, i: usize) -> *mut f32 {
    unsafe { ((t.data as *mut u8).add(row * t.nb[1]) as *mut f32).add(i) }
}

/// RMS norm scale of `x` with `eps`.
fn rms_scale(x: impl Iterator<Item = f32>, n: usize, eps: f32) -> f32 {
    let ss: f32 = x.map(|v| v * v).sum();
    1.0 / (ss / n as f32 + eps).sqrt()
}

unsafe fn cpu_ple_gate(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `gate_norm`, host tensors.
    unsafe {
        let (key, res, value) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (gk, gq, gc) = (
            (*dst.src[3]).data as *const f32,
            (*dst.src[4]).data as *const f32,
            (*dst.src[5]).data as *const f32,
        );
        let (e, hc, n) = (res.ne[0] as usize, res.ne[1] as usize, res.ne[2] as usize);
        let p = crate::ggml::external_params(dst);
        let [eps, inv_sqrt_e, clamp_min] = [0, 1, 2].map(|i| f32::from_bits(p[i] as u32));
        let k = e * hc;
        let xn = dst.data as *mut f32;
        let gate = xn.add(k * n);
        for t in (ith..n).step_by(nth) {
            let kr = std::slice::from_raw_parts(at(key, t, 0), k);
            let rr = (res.data as *const u8).add(t * res.nb[2]) as *const f32;
            let qr = std::slice::from_raw_parts(rr, k);
            let vr = std::slice::from_raw_parts(at(value, t, 0), e);
            for h in 0..hc {
                let span = h * e..(h + 1) * e;
                let sk = rms_scale(kr[span.clone()].iter().copied(), e, eps);
                let sq = rms_scale(qr[span.clone()].iter().copied(), e, eps);
                let s: f32 = span
                    .clone()
                    .map(|i| (kr[i] * sk * *gk.add(i)) * (qr[i] * sq * *gq.add(i)))
                    .sum();
                let s = s * inv_sqrt_e;
                let mag = s.abs().max(clamp_min).sqrt();
                let x = if s > 0.0 {
                    mag
                } else if s < 0.0 {
                    -mag
                } else {
                    0.0
                };
                let g = 1.0 / (1.0 + (-x).exp());
                *gate.add(t * hc + h) = g;
                let sc = rms_scale(vr.iter().map(|v| v * g), e, eps);
                for (i, v) in vr.iter().enumerate() {
                    let at = h * e + i;
                    *xn.add(t * k + at) = (sc * (v * g)) * *gc.add(at);
                }
            }
        }
    }
}

unsafe fn cpu_ple_conv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `conv_out`, host tensors. Each thread takes whole channels, so the
    // history it overwrites is read by no other thread.
    unsafe {
        let (x, state, w, value, gate, res) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
            &*dst.src[5],
        );
        let p = crate::ggml::external_params(dst);
        let (nt, dil) = (p[0] as usize, p[1] as usize);
        let (width, m) = (x.ne[0] as usize, x.ne[1] as usize);
        let (hist, ns) = (state.ne[0] as usize, state.ne[2] as usize);
        let (kern, e) = (w.ne[0] as usize, value.ne[0] as usize);
        let wp = w.data as *const f32;
        let out = dst.data as *mut f32;
        let st = |j: usize, c: usize, s: usize| {
            (state.data as *mut u8).add(j * 4 + c * state.nb[1] + s * state.nb[2]) as *mut f32
        };
        // Input j of [history | tokens] of channel c, sequence s.
        let input = |j: usize, c: usize, s: usize| {
            if j < hist {
                *st(j, c, s)
            } else {
                *at(x, j - hist + s * nt, c)
            }
        };
        for c in (ith..width).step_by(nth) {
            let h = c / e;
            for s in 0..ns {
                for t in 0..nt {
                    let mut acc = 0f32;
                    for k in 0..kern {
                        let term =
                            input(hist - (kern - 1 - k) * dil + t, c, s) * *wp.add(k + c * kern);
                        acc = if k == 0 { term } else { acc + term };
                    }
                    let silu = acc / (1.0 + (-acc).exp());
                    let i = t + s * nt;
                    let gated = *at(value, i, c - h * e) * *at(gate, i, h);
                    *out.add(c + i * width) = *at(res, i, c) + (gated + silu);
                }
                for j in 0..hist {
                    let v = input(nt + j, c, s);
                    *st(j, c, s) = v;
                }
            }
        }
        debug_assert_eq!(m, nt * ns);
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct PleGate {
    key: *const f32,
    key_ld: i64,
    res: *const f32,
    res_ld: i64,
    value: *const f32,
    value_ld: i64,
    gk: *const f32,
    gq: *const f32,
    gc: *const f32,
    xn: *mut f32,
    gate: *mut f32,
    e: i32,
    hc: i32,
    n_tok: i32,
    eps: f32,
    inv_sqrt_e: f32,
    clamp_min: f32,
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct PleConv {
    x: *const f32,
    x_ld: i64,
    x_sd: i64,
    state: *mut f32,
    st_ld: i64,
    st_sd: i64,
    w: *const f32,
    value: *const f32,
    value_ld: i64,
    gate: *const f32,
    res: *const f32,
    res_ld: i64,
    out: *mut f32,
    width: i32,
    e: i32,
    hc: i32,
    nt: i32,
    ns: i32,
    kern: i32,
    dil: i32,
    hist: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_ple_gate_launch(p: *const PleGate, stream: *mut c_void) -> i32;
    fn legion_ple_conv_launch(p: *const PleConv, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ple_gate(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gate_norm`, device tensors of the stream's device.
    unsafe {
        let (key, res, value) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (e, hc, n) = (res.ne[0], res.ne[1], res.ne[2]);
        let prm = crate::ggml::external_params(dst);
        let xn = dst.data as *mut f32;
        let p = PleGate {
            key: key.data as *const f32,
            key_ld: (key.nb[1] / 4) as i64,
            res: res.data as *const f32,
            res_ld: (res.nb[2] / 4) as i64,
            value: value.data as *const f32,
            value_ld: (value.nb[1] / 4) as i64,
            gk: (*dst.src[3]).data as *const f32,
            gq: (*dst.src[4]).data as *const f32,
            gc: (*dst.src[5]).data as *const f32,
            xn,
            gate: xn.add((e * hc * n) as usize),
            e: e as i32,
            hc: hc as i32,
            n_tok: n as i32,
            eps: f32::from_bits(prm[0] as u32),
            inv_sqrt_e: f32::from_bits(prm[1] as u32),
            clamp_min: f32::from_bits(prm[2] as u32),
        };
        match legion_ple_gate_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("ple_gate: error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ple_conv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `conv_out`, device tensors of the stream's device.
    unsafe {
        let (x, state, w, value, gate, res) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
            &*dst.src[5],
        );
        let prm = crate::ggml::external_params(dst);
        let (nt, dil) = (prm[0], prm[1]);
        let ns = state.ne[2] as i32;
        let x_ld = (x.nb[1] / 4) as i64;
        let p = PleConv {
            x: x.data as *const f32,
            x_ld,
            x_sd: x_ld * nt as i64,
            state: state.data as *mut f32,
            st_ld: (state.nb[1] / 4) as i64,
            st_sd: (state.nb[2] / 4) as i64,
            w: w.data as *const f32,
            value: value.data as *const f32,
            value_ld: (value.nb[1] / 4) as i64,
            gate: gate.data as *const f32,
            res: res.data as *const f32,
            res_ld: (res.nb[1] / 4) as i64,
            out: dst.data as *mut f32,
            width: x.ne[0] as i32,
            e: value.ne[0] as i32,
            hc: gate.ne[0] as i32,
            nt,
            ns,
            kern: w.ne[0] as i32,
            dil,
            hist: state.ne[0] as i32,
        };
        match legion_ple_conv_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("ple_conv: error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
