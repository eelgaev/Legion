//! The shared quantization of [`super`] for steps of [`MIN_TOKENS`] to [`MAX_TOKENS`] tokens
//! (multi-sequence decode, MTP verify) on a GB10 (#836), where ggml-cuda runs each Q8_0
//! projection as MMQ with a `quantize_mmq_q8_1` launch of its own: a DeltaNet layer's input is
//! quantized twice (`attn_qkv`, `attn_gate`), an attention layer's three times and a shared
//! expert's input twice, to the same bytes each time.
//!
//! The prompt path's node of quantized blocks would change the graph's layout here: it lives
//! from the first projection to the last, and a DeltaNet layer's `attn_gate` result is read late
//! (by the gated norm), so ggml-alloc would place the layer's later tensors elsewhere, and
//! ggml-cuda turns some of its fusions on or off by whether outputs overlap inputs (#819). So
//! each projection here is one node of ggml's `mul_mat`'s shape and sources (the weight and the
//! input, `[rows, n]` F32), and the blocks go to a scratch buffer of the GPU outside the graphs
//! ([`SCRATCH_BYTES`] per device and stream, as ggml's pool is kept): a node quantizes its input
//! into it with ggml's kernel unless the node before it in the graph's order of these nodes read
//! the same input tensor, then runs ggml's MMQ on the blocks with ggml's arguments
//! ([`super::mmq_preq`]'s glue): ggml's kernels, ggml's bits, one quantize launch per input.
//!
//! Which node quantizes follows from the graph alone ([`reuses`]: the nodes run in the graph's
//! order on the backend's stream, directly or captured in a stage graph), so a capture records
//! the same launches a direct run makes. Taken inside stage graphs on compute capability 12.x
//! (GB10), on by default since #858 (~100 fewer quantize launches a verify step, a few
//! microseconds each; the same bits); `legion serve --shared-quant-small-ggml` keeps ggml's
//! quantization in each matmul. Projections the Q8_0 MMQ op runs
//! ([`crate::model::q8_mmq::takes`]) stay there.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::mmq_preq;

#[cfg(test)]
mod tests;

static SHARED_Q8_SMALL: ExternalOp = ExternalOp {
    name: "shared_q8_small",
    cpu: cpu_shared_q8_small,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::shared_q8_small),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest tokens: ggml-cuda runs up to 8 on MMVQ (`MMVQ_MAX_BATCH_SIZE`).
pub(crate) const MIN_TOKENS: i64 = 9;

/// Most tokens: from 256 the prompt path ([`super::quantize`]) takes over.
pub(crate) const MAX_TOKENS: i64 = 255;

/// Bytes of a scratch buffer: the blocks of the widest input, with MMQ's padding.
pub(crate) const SCRATCH_BYTES: usize = 4 << 20;

/// Whether graphs built from now on may share an input's quantization in small steps (`legion
/// serve --shared-quant-small-ggml` turns it off).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on quantize an input of [`MIN_TOKENS`] to [`MAX_TOKENS`]
/// tokens that several Q8_0 projections read once for them where it fits (on by default);
/// `false` keeps ggml's quantization in each matmul.
pub fn set_shared_q8_small(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether [`mul_mat`] may take `x` in graphs built now: switched on, a GB10
/// ([`mmq_preq::gb10`]) inside stage graphs, and [`input_fits`].
pub(crate) fn takes_input(x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && mmq_preq::gb10()
        && crate::ggml::stage_graph::enabled()
        && input_fits(x)
}

/// Whether the shapes of `x` fit: a contiguous F32 `[k, n]`, `k` a multiple of 512 (ggml-cuda's
/// row padding: the blocks are then exactly `x`'s), [`MIN_TOKENS`] to [`MAX_TOKENS`] columns,
/// its blocks within a scratch buffer.
pub(crate) fn input_fits(x: Tensor<'_>) -> bool {
    let [k, n, n2, n3] = x.ne();
    x.ty() == types::F32
        && x.is_contiguous()
        && k > 0
        && k % 512 == 0
        && n2 == 1
        && n3 == 1
        && (MIN_TOKENS..=MAX_TOKENS).contains(&n)
        && mmq_preq::q8_1_bytes(k, n) <= SCRATCH_BYTES
}

/// Whether [`mul_mat`] takes the weight `w` for the input `x`: [`weight_fits`], and not a
/// matmul the Q8_0 MMQ op runs ([`crate::model::q8_mmq::takes`]), which quantizes on its
/// own.
pub(crate) fn takes_weight(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    weight_fits(w, x.ne()[0]) && !crate::model::q8_mmq::takes(w, x)
}

/// Whether [`mul_mat`] takes the weight `w` with an input of `k` values a token: a plain Q8_0
/// matrix `[k, rows]` in one GPU's memory.
pub(crate) fn weight_fits(w: Tensor<'_>, k: i64) -> bool {
    let [wk, rows, w2, w3] = w.ne();
    w.ty() == types::Q8_0
        && wk == k
        && rows > 0
        && rows <= i32::MAX as i64
        && w2 == 1
        && w3 == 1
        && w.is_contiguous()
        && !w.is_host()
        && !w.is_meta()
}

/// `w @ x` (F32 `[rows, n]`, ggml's `mul_mat`'s shape and sources) for `w` and `x` that fit
/// ([`weight_fits`], [`input_fits`]), from `x`'s blocks in the scratch buffer (see the module
/// docs).
pub(crate) fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    let k = x.ne()[0];
    assert!(
        input_fits(x) && weight_fits(w, k),
        "shared_q8_small: shapes"
    );
    ctx.external(
        &SHARED_Q8_SMALL,
        types::F32,
        [w.ne()[1], x.ne()[1], 1, 1],
        &[w, x],
    )
}

/// Whether the node of `op` at `nodes[at]` finds its input's blocks in the scratch buffer: the
/// last node of `op` before it in `nodes` (the graph's order) read the same input (`src[1]`).
/// Each node of `op` leaves its input's blocks there, so the first of a run of nodes with one
/// input quantizes it and the others reuse it.
///
/// # Safety
/// Every pointer of `nodes` is a valid tensor; `at < nodes.len()`.
#[cfg(any(feature = "cuda", test))]
pub(crate) unsafe fn reuses(
    nodes: &[*mut ffi::ggml_tensor],
    at: usize,
    op: &'static ExternalOp,
) -> bool {
    // SAFETY: the caller's.
    unsafe {
        let x = (*nodes[at]).src[1];
        nodes[..at]
            .iter()
            .rev()
            .find(|&&n| is_op(n, op))
            .is_some_and(|&n| (*n).src[1] == x)
    }
}

/// Whether `node` is a node of `op`.
///
/// # Safety
/// `node` is a valid tensor.
#[cfg(any(feature = "cuda", test))]
unsafe fn is_op(node: *const ffi::ggml_tensor, op: &'static ExternalOp) -> bool {
    // SAFETY: the caller's.
    unsafe { crate::ggml::op_of(node).is_some_and(|o| std::ptr::eq(o, op)) }
}

/// The CPU reference: `x` quantized as ggml's `quantize_mmq_q8_1` does
/// ([`mmq_preq::quantize_ref`]), then each row's sum of its groups' products, in row order.
unsafe fn cpu_shared_q8_small(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`: a contiguous host `x`, a Q8_0 `w`.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, n, rows) = (x.ne[0] as usize, x.ne[1] as usize, w.ne[1] as usize);
        let xs = std::slice::from_raw_parts(x.data as *const f32, k * n);
        let mut xq = vec![0u8; mmq_preq::q8_1_bytes(k as i64, n as i64)];
        mmq_preq::quantize_ref(xs, k, n, &mut xq);
        let out = dst.data as *mut f32;
        for r in (ith..rows).step_by(nth) {
            let row = std::slice::from_raw_parts(
                (w.data as *const u8).add(r * w.nb[1]),
                k / 32 * Q8_0_BYTES,
            );
            for j in 0..n {
                *out.add(j * rows + r) = row_dot(row, &xq, k, n, j);
            }
        }
    }
}

/// Bytes of a Q8_0 block: an F16 scale, 32 int8.
const Q8_0_BYTES: usize = 34;

/// One Q8_0 row (`k / 32` blocks) times column `j` of `n` in MMQ's layout (`xq`): the groups'
/// integer products times both scales, summed in row order.
fn row_dot(row: &[u8], xq: &[u8], k: usize, n: usize, j: usize) -> f32 {
    let per = mmq_preq::BLOCK as usize;
    let mut sum = 0f32;
    for g in 0..k / 32 {
        let wb = &row[g * Q8_0_BYTES..][..Q8_0_BYTES];
        // SAFETY: a pure conversion.
        let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([wb[0], wb[1]])) };
        let blk = &xq[((g * 32 / per) * n + j) * mmq_preq::BLOCK_BYTES..][..mmq_preq::BLOCK_BYTES];
        let gi = g % (per / 32);
        let dx = f32::from_ne_bytes(blk[gi * 4..gi * 4 + 4].try_into().unwrap());
        let isum: i32 = (0..32)
            .map(|i| wb[2 + i] as i8 as i32 * blk[16 + gi * 32 + i] as i8 as i32)
            .sum();
        sum += dw * dx * isum as f32;
    }
    sum
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::os::raw::c_int;
    use std::sync::Mutex;

    use super::*;

    unsafe extern "C" {
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    /// The scratch buffer of each (device, stream), allocated on first use and kept: captured
    /// graphs replay its address. Nodes on one stream run one after the other, so they share it.
    static SCRATCH: Mutex<Option<HashMap<(c_int, usize), usize>>> = Mutex::new(None);

    fn scratch(stream: *mut c_void) -> Result<*mut c_void, String> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries and an allocation on the current device (the stream's).
        unsafe {
            if cudaGetDevice(&mut dev) != 0 {
                cudaGetLastError();
                return Err("no current device".into());
            }
            let mut map = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            let map = map.get_or_insert_with(HashMap::new);
            if let Some(&p) = map.get(&(dev, stream as usize)) {
                return Ok(p as *mut c_void);
            }
            let mut p = std::ptr::null_mut();
            if cudaMalloc(&mut p, SCRATCH_BYTES) != 0 {
                cudaGetLastError();
                return Err(format!("cudaMalloc of {SCRATCH_BYTES} bytes"));
            }
            map.insert((dev, stream as usize), p as usize);
            Ok(p)
        }
    }

    thread_local! {
        /// The graph and index of the last node of the op this thread launched: the next one
        /// is usually a little after it, so [`reuses_in`] looks there first.
        static LAST: Cell<(usize, usize)> = const { Cell::new((0, usize::MAX)) };
    }

    /// [`reuses`] for `node` in graph `g` (`false` when `g` is null or does not hold it).
    ///
    /// # Safety
    /// `g` is null or a valid graph; `node` a valid tensor.
    unsafe fn reuses_in(g: *mut ffi::ggml_cgraph, node: *const ffi::ggml_tensor) -> bool {
        if g.is_null() {
            return false;
        }
        // SAFETY: the caller's: the graph's node array holds `n` valid tensor pointers.
        unsafe {
            let n = ffi::ggml_graph_n_nodes(g) as usize;
            let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(g), n);
            let (lg, li) = LAST.with(Cell::get);
            // Past the last launched node first (it is one of the op's while the graph is the
            // same), else from the start.
            let from = if lg == g as usize && li < n && is_op(nodes[li], &SHARED_Q8_SMALL) {
                li
            } else {
                0
            };
            let at = (from..n)
                .find(|&i| std::ptr::eq(nodes[i], node))
                .or_else(|| (0..from).find(|&i| std::ptr::eq(nodes[i], node)));
            let Some(at) = at else { return false };
            LAST.with(|c| c.set((g as usize, at)));
            reuses(nodes, at, &SHARED_Q8_SMALL)
        }
    }

    pub(super) unsafe fn shared_q8_small(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let backend = crate::ggml::current_cuda_backend();
        if backend.is_null() {
            return Err("no CUDA backend computing".into());
        }
        // SAFETY: a node made by `mul_mat` (its shapes passed `input_fits` and `weight_fits`),
        // device tensors of the backend's device, the current one; the glue enqueues on the
        // backend's stream, `stream`.
        unsafe {
            let (w, x) = (&*dst.src[0], &*dst.src[1]);
            let (k, n) = (x.ne[0], x.ne[1]);
            let yq = scratch(stream)?;
            if !reuses_in(crate::ggml::current_cuda_graph(), dst) {
                let e =
                    mmq_preq::legion_ggml_quantize_mmq_q8_0(x.data as *const f32, yq, k, n, stream);
                if e != 0 {
                    return Err(format!("quantize: CUDA error {e}"));
                }
            }
            match mmq_preq::legion_mmq_preq(
                backend,
                w,
                yq,
                SCRATCH_BYTES,
                n,
                dst.data as *mut f32,
                dst.nb[1] as i64,
            ) {
                0 => Ok(()),
                -1 => Err("not a matmul ggml-cuda runs with MMQ".into()),
                -2 => Err("MMQ's padding short".into()),
                e => Err(format!("CUDA error {e}")),
            }
        }
    }
}
