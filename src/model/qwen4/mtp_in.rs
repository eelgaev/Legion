//! qwen4exp's MTP head input as one external op (#1183): `[enorm(tok) per stream ; hnorm(h)]`,
//! the `eh_proj` matmul's input, for the head's prompt take-ins under `--dense-fp8-prompt`.
//!
//! ggml's chain in [`super::Model::build_mtp_qwen4`] is six nodes: the embedding's `rms_norm`
//! and `mul` by `enorm`, its `repeat` to the hc streams, the streams' `rms_norm` and `mul` by
//! `hnorm`, and the `concat` (on a GB10 about 1.7 ms per 1,024-token take-in, ~50 ms of a
//! 32K prompt). The op reads the embedding and the streams once and writes the concat's output
//! (CUDA kernel in `src/cuda/mtp_in.cu`). On CUDA it computes what ggml-cuda's kernels for the
//! chain compute, bit for bit (`mtp_in/tests.rs`): rows of 1,024 values or more, the width
//! ggml-cuda's `rms_norm` takes 1,024 threads a row from. Taken by a prompt's take-in pieces
//! only (passes without output rows of more than
//! [`crate::model::core::dense_twins::STEP_MAX_TOKENS`] tokens), as the PLE ops
//! ([`super::ple_fused`]) are by prompt steps: draft passes and the take-ins after decode and
//! verify steps keep the chain.
//!
//! Quantized ([`input_q8`]): where the `eh_proj` weight is Q8_0 and ggml-cuda would run its
//! matmul through MMQ ([`mmq_preq::fits`]), the op writes its values already quantized in
//! MMQ's layout, as ggml-cuda's `quantize_mmq_q8_1` would from the F32 values (its arithmetic,
//! the same compiler flags), and the matmul runs ggml's MMQ on those blocks
//! ([`mmq_preq::mul_mat_cols`]): ggml's product bit for bit, without the F32 tensor (84 MB a
//! 1,024-token take-in) written and read back by ggml's quantize pass (about 16 ms of a 32K
//! prompt on a GB10; 24 MB of blocks instead).

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::mmq_preq;

static MTP_IN: ExternalOp = ExternalOp {
    name: "mtp_in",
    cpu: cpu_mtp_in,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mtp_in),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The streams the CUDA kernel takes.
pub(super) const MTP_IN_HC: i64 = 4;
/// The narrowest row the CUDA kernel takes (ggml-cuda's rms_norm takes 1,024 threads per row
/// from there; narrower rows have another order of sums).
pub(super) const MTP_IN_MIN_E: i64 = 1024;

/// Whether `node` is an [`input`] op (counted on the node fuse log line).
///
/// # Safety
/// `node` is a valid tensor.
pub(crate) unsafe fn is_mtp_in(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe { crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &MTP_IN)) }
}

/// Whether the [`is_mtp_in`] `node` writes its values quantized ([`input_q8`]).
///
/// # Safety
/// `node` is a valid tensor.
pub(crate) unsafe fn is_quantized(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe { is_mtp_in(node) && quantized(&*node) }
}

/// The [`P_Q8`] parameter of an op's node.
fn quantized(dst: &ffi::ggml_tensor) -> bool {
    crate::ggml::external_params(dst)[P_Q8] == 1
}

/// The op's parameter set where it writes its values quantized ([`input_q8`]).
const P_Q8: usize = 1;

/// Why [`input_q8`] and [`mmq_preq::mul_mat_cols`] do not take the `eh_proj` weight `w` for
/// `n` tokens of `hc` streams of width `e` (the first reason found), `None` where they do:
/// a weight [`mmq_preq::fits`] takes for `hc · n` columns, `2e` values a row, `e` a multiple
/// of 256 (whole MMQ blocks and 32-value groups in each half).
pub(super) fn why_not_q8(w: Tensor<'_>, e: i64, hc: i64, n: i64) -> Option<String> {
    if w.ne()[0] != 2 * e || e % 256 != 0 {
        Some(format!(
            "eh_proj is {:?} for streams of {e} (rows of 2e, e a multiple of 256 wanted)",
            w.ne()
        ))
    } else if !mmq_preq::fits(w, hc * n) {
        Some(format!(
            "eh_proj is {} {:?} for {} columns (a Q8_0 weight on a compute capability 12.x \
             GPU, rows a multiple of 512 values and 256 columns or more wanted; off with \
             --hc-quant-ggml)",
            crate::ggml::type_name(w.ty()),
            w.ne(),
            hc * n
        ))
    } else {
        None
    }
}

/// Why [`input`] does not take these tensors (the first reason found), `None` where it does:
/// it takes `tok` (`[e, n]`) and `h` (`[e · hc, n]`) F32 with contiguous rows, the gammas `ge`
/// (`[e]`) and `gh` (`[e · hc]`) F32 and contiguous, all in GPU memory, `hc` the kernel's and
/// `e` at least [`MTP_IN_MIN_E`].
pub(super) fn why_not(
    tok: Tensor<'_>,
    h: Tensor<'_>,
    ge: Tensor<'_>,
    gh: Tensor<'_>,
    hc: i64,
) -> Option<String> {
    let on_gpu = |t: Tensor<'_>| !t.is_host() && !t.is_meta();
    let f32_rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
    let [e, n, ..] = tok.ne();
    let ty = |t: Tensor<'_>| crate::ggml::type_name(t.ty());
    if !cfg!(feature = "cuda") {
        Some("not a CUDA build".to_string())
    } else if hc != MTP_IN_HC || e < MTP_IN_MIN_E {
        Some(format!(
            "{hc} streams of {e} (the kernel takes 4 of 1,024 or more)"
        ))
    } else if h.ne()[..2] != [e * hc, n] {
        Some(format!(
            "the streams are {:?} for {n} tokens of {e}",
            h.ne()
        ))
    } else if let Some(t) = [tok, h].into_iter().find(|&t| !(f32_rows(t) && on_gpu(t))) {
        Some(format!(
            "{} is {} with strides {:?} (F32 rows in GPU memory wanted)",
            t.name(),
            ty(t),
            t.nb()
        ))
    } else if let Some((g, len)) = [(ge, e), (gh, e * hc)].into_iter().find(|&(g, len)| {
        !(g.ty() == types::F32 && g.is_contiguous() && g.elements() == len as usize && on_gpu(g))
    }) {
        Some(format!(
            "{} is {} {:?} (contiguous F32 of {len} in GPU memory wanted)",
            g.name(),
            ty(g),
            g.ne()
        ))
    } else {
        None
    }
}

/// `concat(repeat(rms_norm(tok) · ge, hc), rms_norm(h) · gh, 0)` for `n` tokens: `tok` (F32
/// `[e, n]`, rows contiguous), the streams `h` (F32 `[e · hc, n]`, rows contiguous) and the
/// gammas `ge` (`[e]`) and `gh` (`[e · hc]`, one `e` per stream). Returns F32 `[2e, hc, n]`:
/// per token and stream, the normed embedding then the stream's norm.
pub(super) fn input<'a>(
    ctx: &'a Context,
    tok: Tensor<'a>,
    h: Tensor<'a>,
    ge: Tensor<'a>,
    gh: Tensor<'a>,
    hc: i64,
    eps: f32,
) -> Tensor<'a> {
    let [e, n, ..] = tok.ne();
    check(tok, h, ge, gh, hc);
    ctx.external_with(
        &MTP_IN,
        types::F32,
        [2 * e, hc, n, 1],
        &[tok, h, ge, gh],
        &[eps.to_bits() as i32, 0],
    )
}

/// [`input`]'s values quantized for ggml-cuda's MMQ of a Q8_0 `eh_proj` weight (see the module
/// doc): `hc · n` columns of `2e` values (column `t · hc + c`: token `t`, stream `c`) in
/// [`mmq_preq`]'s layout, an F32 node of [`mmq_preq::q8_1_bytes`] for
/// [`mmq_preq::mul_mat_cols`]. `e` a multiple of 256 ([`why_not_q8`]).
pub(super) fn input_q8<'a>(
    ctx: &'a Context,
    tok: Tensor<'a>,
    h: Tensor<'a>,
    ge: Tensor<'a>,
    gh: Tensor<'a>,
    hc: i64,
    eps: f32,
) -> Tensor<'a> {
    let [e, n, ..] = tok.ne();
    check(tok, h, ge, gh, hc);
    assert!(e % 256 == 0, "mtp_in: {e} values a stream");
    let floats = mmq_preq::q8_1_bytes(2 * e, hc * n).div_ceil(4) as i64;
    ctx.external_with(
        &MTP_IN,
        types::F32,
        [floats, 1, 1, 1],
        &[tok, h, ge, gh],
        &[eps.to_bits() as i32, 1],
    )
}

/// [`input`]'s operands as it takes them.
fn check(tok: Tensor<'_>, h: Tensor<'_>, ge: Tensor<'_>, gh: Tensor<'_>, hc: i64) {
    let [e, n, ..] = tok.ne();
    assert_eq!(h.ne()[..2], [e * hc, n], "mtp_in: streams");
    for t in [tok, h] {
        assert_eq!(t.ty(), types::F32, "mtp_in: F32 inputs");
        assert_eq!(t.nb()[0], 4, "mtp_in: rows contiguous");
    }
    for (g, len) in [(ge, e), (gh, e * hc)] {
        assert!(
            g.ty() == types::F32 && g.is_contiguous() && g.elements() == len as usize,
            "mtp_in: gammas"
        );
    }
}

/// RMS norm scale of `x` with `eps`.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    1.0 / (ss / x.len() as f32 + eps).sqrt()
}

unsafe fn cpu_mtp_in(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `input`, host tensors.
    unsafe {
        let (tok, h) = (&*dst.src[0], &*dst.src[1]);
        let (ge, gh) = (
            (*dst.src[2]).data as *const f32,
            (*dst.src[3]).data as *const f32,
        );
        let (e, n) = (tok.ne[0] as usize, tok.ne[1] as usize);
        let hc = h.ne[0] as usize / e;
        let eps = f32::from_bits(crate::ggml::external_params(dst)[0] as u32);
        let q8 = quantized(dst);
        let out = dst.data as *mut f32;
        // Quantized: one column's values, then its blocks ([`mmq_preq::quantize_ref`]).
        let len = if q8 { 2 * e } else { 0 };
        let mut col = vec![0f32; len];
        let mut blocks = vec![0u8; len / mmq_preq::BLOCK as usize * mmq_preq::BLOCK_BYTES];
        for t in (ith..n).step_by(nth) {
            let row = |x: &ffi::ggml_tensor, len: usize| {
                std::slice::from_raw_parts(
                    (x.data as *const u8).add(t * x.nb[1]) as *const f32,
                    len,
                )
            };
            let (tr, hr) = (row(tok, e), row(h, e * hc));
            let se = rms_scale(tr, eps);
            for c in 0..hc {
                let o = if q8 {
                    col.as_mut_ptr()
                } else {
                    out.add((t * hc + c) * 2 * e)
                };
                let hs = &hr[c * e..(c + 1) * e];
                let sh = rms_scale(hs, eps);
                for i in 0..e {
                    *o.add(i) = (se * tr[i]) * *ge.add(i);
                    *o.add(e + i) = (sh * hs[i]) * *gh.add(c * e + i);
                }
                if q8 {
                    mmq_preq::quantize_ref(&col, 2 * e, 1, &mut blocks);
                    let j = t * hc + c;
                    for (b, blk) in blocks.chunks_exact(mmq_preq::BLOCK_BYTES).enumerate() {
                        let at = (b * n * hc + j) * mmq_preq::BLOCK_BYTES;
                        std::ptr::copy_nonoverlapping(
                            blk.as_ptr(),
                            (out as *mut u8).add(at),
                            mmq_preq::BLOCK_BYTES,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The kernel's launch: `cudaLaunchKernel` from Rust (no host code in the `.cu`).

    use std::os::raw::{c_int, c_void};

    use super::ffi;

    /// `legion_mtp_in` (src/cuda/mtp_in.cu).
    #[repr(C)]
    struct Params {
        tok: *const f32,
        tok_ld: i64,
        h: *const f32,
        h_ld: i64,
        ge: *const f32,
        gh: *const f32,
        dst: *mut f32,
        e: i32,
        n: i32,
        eps: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// Threads of a block (`MI_THREADS`).
    const THREADS: u32 = 1024;

    unsafe extern "C" {
        static legion_mtp_in_kernels: [*const c_void; 2];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
    }

    pub(super) unsafe fn mtp_in(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `input`, device tensors of the stream's device.
        unsafe {
            let (tok, h) = (&*dst.src[0], &*dst.src[1]);
            let (e, n) = (tok.ne[0], tok.ne[1]);
            let hc = h.ne[0] / e;
            let q8 = super::quantized(dst);
            let fits = if q8 {
                e % 256 == 0 && ffi::ggml_nbytes(dst) >= super::mmq_preq::q8_1_bytes(2 * e, hc * n)
            } else {
                dst.ne[..3] == [2 * e, hc, n]
            };
            if !fits
                || hc != super::MTP_IN_HC
                || e < super::MTP_IN_MIN_E
                || n < 1
                || n > i32::MAX as i64
            {
                return Err(format!(
                    "mtp_in: shape the kernel does not take (e {e}, hc {hc}, n {n}, q8 {q8})"
                ));
            }
            let mut p = Params {
                tok: tok.data as *const f32,
                tok_ld: (tok.nb[1] / 4) as i64,
                h: h.data as *const f32,
                h_ld: (h.nb[1] / 4) as i64,
                ge: (*dst.src[2]).data as *const f32,
                gh: (*dst.src[3]).data as *const f32,
                dst: dst.data as *mut f32,
                e: e as i32,
                n: n as i32,
                eps: f32::from_bits(crate::ggml::external_params(dst)[0] as u32),
            };
            let grid = Dim3 {
                x: n as u32,
                y: 1,
                z: 1,
            };
            let block = Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            };
            let mut args = [(&mut p as *mut Params).cast::<c_void>()];
            // SAFETY: the kernel takes one argument of type `Params`; the launch copies it.
            match cudaLaunchKernel(
                legion_mtp_in_kernels[usize::from(q8)],
                grid,
                block,
                args.as_mut_ptr(),
                0,
                stream,
            ) {
                0 => Ok(()),
                e => Err(format!("mtp_in: CUDA error {e}")),
            }
        }
    }
}

#[cfg(test)]
mod tests;
