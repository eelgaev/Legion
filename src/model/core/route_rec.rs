//! `--moe-route-record` (#803): every MoE layer's router choices, appended to a file for
//! `legion ep-place` ([`crate::tp_net::ep_place`]).
//!
//! With a file open ([`open`]), [`Model::moe`](super::Model) adds [`record`] after the
//! router's top-k: an external op that copies the step's ids to the host (on CUDA: on the
//! backend's stream, then waits for it) and appends them in
//! [`crate::tp_net::ep_place::write_record`]'s format. The wait splits the step's CUDA graph
//! at every MoE layer, so recording slows the server down: calibration runs only. Without a
//! file nothing is added to the graphs.

#[cfg(test)]
mod tests;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

/// Open record files; a node names its file by index (so a test's records stay its own).
static SINKS: Mutex<Vec<BufWriter<File>>> = Mutex::new(Vec::new());
/// The file [`open`] opened: the one [`record`] writes to.
static ACTIVE: OnceLock<usize> = OnceLock::new();

/// Opens `path` to append records to; its index in [`SINKS`].
pub(crate) fn sink(path: &Path) -> Result<usize, String> {
    let f = File::options()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut s = SINKS.lock().unwrap_or_else(|e| e.into_inner());
    s.push(BufWriter::new(f));
    Ok(s.len() - 1)
}

/// Starts recording every MoE layer to `path` (appended to) for the rest of the process.
pub fn open(path: &Path) -> Result<(), String> {
    if on() {
        return Err("already recording".into());
    }
    let at = sink(path)?;
    ACTIVE.set(at).map_err(|_| "already recording".to_string())
}

/// Whether [`open`] was called.
pub fn on() -> bool {
    ACTIVE.get().is_some()
}

fn append(sink: usize, il: usize, n_expert: usize, n_used: usize, ids: &[i32]) {
    let mut s = SINKS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(w) = s.get_mut(sink) else {
        return;
    };
    let r = crate::tp_net::ep_place::write_record(w, il, n_expert, n_used, ids)
        .and_then(|()| w.flush());
    if let Err(e) = r {
        tracing::warn!("--moe-route-record: {e}");
    }
}

static OP: ExternalOp = ExternalOp {
    name: "moe_route_record",
    cpu: cpu_record,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::record),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Records layer `il`'s router choices `selected` (I32 `[n_used, n]`, maybe a view) of
/// `n_expert` experts to [`open`]'s file when this step is computed: a node to expand into the
/// step's graph. Its output (one I32) holds nothing. Only with [`on`].
pub(crate) fn record<'a>(
    ctx: &'a Context,
    il: usize,
    n_expert: usize,
    selected: Tensor<'a>,
) -> Tensor<'a> {
    let at = *ACTIVE.get().expect("--moe-route-record is on");
    record_to(ctx, at, il, n_expert, selected)
}

/// [`record`] to [`sink`] `at`.
pub(crate) fn record_to<'a>(
    ctx: &'a Context,
    at: usize,
    il: usize,
    n_expert: usize,
    selected: Tensor<'a>,
) -> Tensor<'a> {
    let ids = if selected.is_contiguous() {
        selected
    } else {
        ctx.cont(selected)
    };
    let params = [at as i32, il as i32, n_expert as i32];
    ctx.external_with(&OP, types::I32, [1, 1, 1, 1], &[ids], &params)
}

/// The node's file, layer, expert count, ids per token and ids.
struct Shape {
    at: usize,
    il: usize,
    n_expert: usize,
    n_used: usize,
    n: usize,
}

fn shape(dst: &ffi::ggml_tensor) -> Shape {
    let p = external_params(dst);
    // SAFETY: the op's one source, made by `record_to`.
    let ids = unsafe { &*dst.src[0] };
    let n_used = ids.ne[0] as usize;
    Shape {
        at: p[0] as usize,
        il: p[1] as usize,
        n_expert: p[2] as usize,
        n_used,
        n: n_used * (ids.ne[1] * ids.ne[2] * ids.ne[3]) as usize,
    }
}

unsafe fn cpu_record(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let s = shape(dst);
    // SAFETY: the contiguous I32 source in host memory, `n` ids.
    let ids = unsafe { std::slice::from_raw_parts((*dst.src[0]).data as *const i32, s.n) };
    append(s.at, s.il, s.n_expert, s.n_used, ids);
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};

    use super::{append, ffi, shape};

    unsafe extern "C" {
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: *mut c_void,
        ) -> c_int;
        fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    }

    /// `cudaMemcpyDefault`: the direction from the pointers (unified addressing).
    const DEFAULT: c_int = 4;

    /// Copies the ids to the host after the work before them on `stream`, waits, appends.
    pub(super) unsafe fn record(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let s = shape(dst);
        let mut ids = vec![0i32; s.n];
        // SAFETY: the op's contiguous device source of `n` ids into a host buffer of as many.
        unsafe {
            let e = cudaMemcpyAsync(
                ids.as_mut_ptr().cast(),
                (*dst.src[0]).data,
                s.n * 4,
                DEFAULT,
                stream,
            );
            if e != 0 {
                return Err(format!("moe_route_record: ids to host: CUDA error {e}"));
            }
            let e = cudaStreamSynchronize(stream);
            if e != 0 {
                return Err(format!("moe_route_record: CUDA error {e}"));
            }
        }
        append(s.at, s.il, s.n_expert, s.n_used, &ids);
        Ok(())
    }
}
