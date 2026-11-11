//! The gated delta rule of a prompt step's DeltaNet layers as an external op
//! ([`crate::ggml::ExternalOp`]) with a CUDA kernel of Legion's (`src/cuda/gdn_scan.cu`): ggml's
//! `gated_delta_net` token after token, with ggml-cuda's arithmetic and its bits, faster where
//! that kernel waits on memory each token (compute capability 7.x: V100).
//!
//! ggml-cuda runs the recurrence one token at a time per state column, eight lanes a column, and
//! loads each token's inputs inside that token's step. `gdn_scan` keeps the lanes, the sums'
//! order and the expressions (so the outputs and states are ggml's, bit for bit, on the GPUs it
//! is chosen for) and stages the next tokens' inputs into shared memory while it computes the
//! current ones (as upstream Strata 0.1.41 pipelines its prompt recurrence, MIT).
//!
//! It takes prompt groups only: a decode step's state update in place and an MTP verify step's
//! replay keep ggml's op. GPUs from sm_80 on keep ggml's op too: ggml runs a chunked MMA kernel
//! on several of them, with other bits.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static GDN_SCAN: ExternalOp = ExternalOp {
    name: "gdn_scan",
    cpu: cpu_gdn_scan,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gdn_scan),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The head size (state rows and columns) the kernel takes.
pub(super) const HEAD: i64 = 128;

/// Fewest tokens of a group the op takes: shorter groups (decode, MTP verify) keep ggml's op.
pub(super) const MIN_TOKENS: i64 = 16;

/// Whether graphs built from now on may use [`gdn_scan`] (`legion serve --gdn-scan-ggml` turns
/// it off for comparisons).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether the engine's GPU is one the op is chosen for (compute capability 7.x), set when the
/// engine loads ([`set_gpu`]): the answer for a group whose state names no device ([`gpu_for`]).
static GPU: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on use [`gdn_scan`] where it fits (on by default; ggml's
/// `gated_delta_net` with `false`).
pub fn set_gdn_scan(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: [`gdn_scan`] is used on compute capability 7.x (`cc_major`), where it gives
/// ggml-cuda's bits faster; elsewhere ggml's op stays (sm_80+ GPUs may run ggml's chunked
/// kernel, other bits).
pub fn set_gpu(cc_major: Option<i32>) {
    GPU.store(gpu_takes(cc_major), Ordering::Relaxed);
}

/// Whether a GPU of compute capability major `cc_major` takes the op ([`set_gpu`]).
pub(super) fn gpu_takes(cc_major: Option<i32>) -> bool {
    cc_major == Some(7)
}

/// Whether the GPU of device `dev` (the state's) takes the op, asked per device (#1201: a box
/// may hold GPUs of different architectures); the engine's answer where `dev` is unknown.
fn gpu_for(dev: Option<usize>) -> bool {
    gpu_for_by(dev, GPU.load(Ordering::Relaxed), crate::hw::per_device::cc)
}

/// [`gpu_for`] with the engine's answer `engine` and the compute capabilities `cc_of` gives.
pub(super) fn gpu_for_by(
    dev: Option<usize>,
    engine: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, |cc| gpu_takes(cc.map(|c| c.0)), cc_of)
}

/// Whether [`gdn_scan`] takes this group: switched on, a prompt group (at least [`MIN_TOKENS`]
/// tokens, not a verify group: `prompt`), heads of [`HEAD`] with a scalar gate, F32, and the
/// state in the memory of one 7.x GPU ([`gpu_for`]).
pub(super) fn fits(
    q: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    state: Tensor<'_>,
    n_tok: i64,
    prompt: bool,
) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && prompt
        && n_tok >= MIN_TOKENS
        && q.ne()[0] == HEAD
        && v.ne()[0] == HEAD
        && g.ne()[0] == 1
        && [q, v, g, state].iter().all(|t| t.ty() == types::F32)
        && !state.is_host()
        && !state.is_meta()
        && gpu_for(state.cuda_device())
}

/// ggml's `gated_delta_net(q, k, v, g, beta, state, 1)` as [`GDN_SCAN`]: the same shapes in
/// and out (`[S_v * H, n_tokens * n_seqs + S_v * n_seqs]`: the tokens' outputs, then the final
/// states), so the caller's views of the result do not change.
pub(super) fn gdn_scan<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
) -> Tensor<'a> {
    let [s_v, h, nt, ns] = v.ne();
    assert_eq!(s_v, HEAD, "gdn_scan: head size");
    assert_eq!(q.ne(), k.ne(), "gdn_scan: q and k");
    assert_eq!(q.nb(), k.nb(), "gdn_scan: q and k strides");
    assert_eq!(q.ne()[0], s_v, "gdn_scan: q head size");
    assert_eq!(q.ne()[2], nt, "gdn_scan: q tokens");
    assert!(
        h % q.ne()[1] == 0 && ns % q.ne()[3] == 0,
        "gdn_scan: heads {} of {h}, sequences {} of {ns}",
        q.ne()[1],
        q.ne()[3]
    );
    assert_eq!(g.ne(), [1, h, nt, ns], "gdn_scan: g");
    assert_eq!(beta.ne(), [1, h, nt, ns], "gdn_scan: beta");
    assert_eq!(state.ne(), [s_v, s_v, h, ns], "gdn_scan: state");
    for t in [q, k, v] {
        assert_eq!(t.nb()[0], 4, "gdn_scan: rows contiguous");
    }
    for t in [g, beta, state] {
        assert!(t.is_contiguous(), "gdn_scan: g, beta, state contiguous");
    }
    ctx.external(
        &GDN_SCAN,
        types::F32,
        [s_v * h, nt * ns + s_v * ns, 1, 1],
        &[q, k, v, g, beta, state],
    )
}

/// Element strides in floats of dims 1, 2 and 3.
pub(super) fn strides(t: &ffi::ggml_tensor) -> [i64; 3] {
    [1, 2, 3].map(|d| (t.nb[d] / 4) as i64)
}

pub(super) unsafe fn cpu_gdn_scan(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `gdn_scan`, host tensors. The CPU graphs keep ggml's op; this is the
    // reference for tests, in ggml-cuda's order of operations (its rounding may differ by the
    // compiler's contractions).
    unsafe {
        let [q, k, v, g, beta, s0] = [0, 1, 2, 3, 4, 5].map(|i| &*dst.src[i]);
        let (s_v, h_n, nt, ns) = (
            v.ne[0] as usize,
            v.ne[1] as usize,
            v.ne[2] as usize,
            v.ne[3] as usize,
        );
        let (n_qk, rq3) = (q.ne[1] as usize, ns / q.ne[3] as usize);
        let [sq1, sq2, sq3] = strides(q);
        let [sv1, sv2, sv3] = strides(v);
        let [sb1, sb2, sb3] = strides(beta);
        let (qp, kp, vp) = (
            q.data as *const f32,
            k.data as *const f32,
            v.data as *const f32,
        );
        let (gp, bp, sp) = (
            g.data as *const f32,
            beta.data as *const f32,
            s0.data as *const f32,
        );
        let out = dst.data as *mut f32;
        let states = out.add(s_v * h_n * nt * ns);
        let scale = 1.0 / (s_v as f32).sqrt();
        // One (sequence, head) per task: its own state and output columns.
        for task in (ith..h_n * ns).step_by(nth) {
            let (seq, h) = (task / h_n, task % h_n);
            let qk_off = ((h % n_qk) as i64 * sq1 + (seq / rq3) as i64 * sq3) as usize;
            // st[col][i]: S[i][col], as ggml stores it
            let mut st: Vec<f32> =
                std::slice::from_raw_parts(sp.add((seq * h_n + h) * s_v * s_v), s_v * s_v).to_vec();
            for t in 0..nt {
                let qt = qp.add(qk_off + t * sq2 as usize);
                let kt = kp.add(qk_off + t * sq2 as usize);
                let vt = vp.add((h as i64 * sv1 + t as i64 * sv2 + seq as i64 * sv3) as usize);
                let gb = (h as i64 * sb1 + t as i64 * sb2 + seq as i64 * sb3) as usize;
                let (g_val, beta_val) = ((*gp.add(gb)).exp(), *bp.add(gb));
                let o = out.add(((seq * nt + t) * h_n + h) * s_v);
                for col in 0..s_v {
                    let s = &mut st[col * s_v..(col + 1) * s_v];
                    let mut kv = 0f32;
                    for (i, x) in s.iter().enumerate() {
                        kv += x * *kt.add(i);
                    }
                    let delta = (*vt.add(col) - g_val * kv) * beta_val;
                    let mut attn = 0f32;
                    for (i, x) in s.iter_mut().enumerate() {
                        *x = g_val * *x + *kt.add(i) * delta;
                        attn += *x * *qt.add(i);
                    }
                    *o.add(col) = attn * scale;
                }
            }
            std::ptr::copy_nonoverlapping(
                st.as_ptr(),
                states.add((seq * h_n + h) * s_v * s_v),
                s_v * s_v,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct GdnScan {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    s0: *const f32,
    out: *mut f32,
    state: *mut f32,
    sq: [i64; 3],
    sv: [i64; 3],
    sb: [i64; 3],
    h: i32,
    n_tokens: i32,
    n_seqs: i32,
    n_qk: i32,
    rq3: i32,
    scale: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_gdn_scan_launch(p: *const GdnScan, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gdn_scan(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gdn_scan`, device tensors of the stream's device.
    unsafe {
        let [q, k, v, g, beta, s0] = [0, 1, 2, 3, 4, 5].map(|i| &*dst.src[i]);
        let (s_v, h, nt, ns) = (v.ne[0], v.ne[1], v.ne[2], v.ne[3]);
        let out = dst.data as *mut f32;
        let p = GdnScan {
            q: q.data as *const f32,
            k: k.data as *const f32,
            v: v.data as *const f32,
            g: g.data as *const f32,
            beta: beta.data as *const f32,
            s0: s0.data as *const f32,
            out,
            state: out.add((s_v * h * nt * ns) as usize),
            sq: strides(q),
            sv: strides(v),
            sb: strides(beta),
            h: h as i32,
            n_tokens: nt as i32,
            n_seqs: ns as i32,
            n_qk: q.ne[1] as i32,
            rq3: (ns / q.ne[3]) as i32,
            // ggml: 1.0f / sqrtf((float) S_v)
            scale: 1.0 / (s_v as f32).sqrt(),
        };
        match legion_gdn_scan_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err(format!("gdn_scan: shape {:?} unsupported", v.ne)),
            e => Err(format!("gdn_scan: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
