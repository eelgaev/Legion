//! A per-op GPU time profile for development, on with `LEGION_OP_PROFILE=1` (CUDA only): every
//! node of a graph runs as a segment of its own between two CUDA events, and the times add up
//! per op and tensor name (layer numbers stripped). The cumulative table goes to stderr every
//! [`REPORT_EVERY`] profiled graphs. Running node by node disables ggml's CUDA graphs and
//! fusions, so totals are somewhat higher than in a normal run; the split is what it is for.

use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

use super::external::{cuda, op_of};
use super::{GgmlError, ffi};

/// Profiled graphs between two reports.
const REPORT_EVERY: usize = 8;

unsafe extern "C" {
    fn cudaEventCreate(e: *mut *mut c_void) -> c_int;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> c_int;
    fn cudaEventDestroy(e: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    fn cudaGetDevice(d: *mut c_int) -> c_int;
}

/// Whether `LEGION_OP_PROFILE=dev` asks for the times per device (`cuda<N>` before each key).
fn per_device() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_OP_PROFILE").is_some_and(|v| v == "dev"))
}

/// Whether `LEGION_OP_PROFILE` asks for the profile.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_OP_PROFILE").is_some_and(|v| v != "0"))
}

#[derive(Default)]
struct Totals {
    /// (milliseconds, calls) per key.
    by_key: HashMap<String, (f64, u64)>,
    graphs: usize,
}

static TOTALS: Mutex<Option<Totals>> = Mutex::new(None);

/// `name` without a trailing `-<layer>` (and `(view)` suffixes), so layers add up.
fn key_of(op: &str, name: &str) -> String {
    if name.is_empty() || name.starts_with("node_") || name.starts_with("leaf_") {
        return op.to_string();
    }
    if op == "MUL_MAT" || op == "MUL_MAT_ID" || op == "glm_linear" {
        return format!("{op} {name}");
    }
    let base = name.split(" (").next().unwrap_or(name);
    let base = match base.rsplit_once('-') {
        Some((b, n)) if n.chars().all(|c| c.is_ascii_digit()) => b,
        _ => base,
    };
    format!("{op} {base}")
}

/// Runs `graph` on the CUDA backend `raw` node by node, timing each, and synchronizes.
///
/// # Safety
/// `raw` is a CUDA backend and `g` an allocated graph for it.
pub(crate) unsafe fn compute(
    raw: ffi::ggml_backend_t,
    g: *mut ffi::ggml_cgraph,
) -> Result<(), GgmlError> {
    unsafe {
        cuda::legion_cuda_set_device(raw);
        let stream = cuda::legion_cuda_stream(raw);
        let _backend = cuda::BackendScope::enter(raw, g);
        let n = ffi::ggml_graph_n_nodes(g);
        let dev = if per_device() {
            let mut d = 0;
            cudaGetDevice(&mut d);
            format!("cuda{d} ")
        } else {
            String::new()
        };
        let mut marks: Vec<(*mut c_void, *mut c_void, String)> = Vec::new();
        let new_event = || {
            let mut e = std::ptr::null_mut();
            cudaEventCreate(&mut e);
            e
        };
        for i in 0..n {
            let node = ffi::ggml_graph_node(g, i);
            let op = match op_of(node) {
                Some(ext) => ext.name.to_string(),
                None => std::ffi::CStr::from_ptr(ffi::ggml_op_desc(node))
                    .to_string_lossy()
                    .into_owned(),
            };
            // A matmul is known by its weight (its type too: the kernels differ); GLM's linear
            // also by the bytes it reads, for GB/s per shape.
            let glm_linear = op == "glm_linear";
            let name = if (glm_linear
                || matches!(
                    (*node).op,
                    ffi::ggml_op_GGML_OP_MUL_MAT | ffi::ggml_op_GGML_OP_MUL_MAT_ID
                ))
                && !(*node).src[0].is_null()
            {
                let w = &*(*node).src[0];
                let wn = std::ffi::CStr::from_ptr(w.name.as_ptr()).to_string_lossy();
                let wn = match wn.strip_prefix("blk.") {
                    Some(r) => r.split_once('.').map_or(r, |(_, t)| t).to_string(),
                    None => wn.into_owned(),
                };
                let ty = std::ffi::CStr::from_ptr(ffi::ggml_type_name(w.type_)).to_string_lossy();
                if glm_linear {
                    let wn = wn.split(" (").next().unwrap_or(&wn);
                    let bytes = ffi::ggml_nbytes(w);
                    format!(
                        "{wn} [{ty}] {}x{}x{} {bytes}B x{}",
                        w.ne[0],
                        w.ne[1],
                        w.ne[2],
                        (*node).ne[1]
                    )
                } else {
                    format!("{wn} [{ty}] x{}", (*node).ne[1])
                }
            } else {
                std::ffi::CStr::from_ptr((*node).name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            let (a, b) = (new_event(), new_event());
            cudaEventRecord(a, stream);
            match op_of(node) {
                Some(ext) => {
                    let launch = ext.cuda.ok_or_else(|| {
                        GgmlError::External(format!("{}: no CUDA kernel", ext.name))
                    })?;
                    launch(&*node, stream)
                        .map_err(|e| GgmlError::External(format!("{}: {e}", ext.name)))?;
                }
                None => {
                    let status = cuda::legion_cuda_compute_range(raw, g, i, i + 1);
                    if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                        return Err(GgmlError::Compute(status));
                    }
                }
            }
            cudaEventRecord(b, stream);
            marks.push((a, b, format!("{dev}{}", key_of(&op, &name))));
        }
        cudaStreamSynchronize(stream);
        let mut guard = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        let totals = guard.get_or_insert_with(Totals::default);
        for (a, b, key) in marks {
            let mut ms = 0f32;
            cudaEventElapsedTime(&mut ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            let e = totals.by_key.entry(key).or_default();
            e.0 += ms as f64;
            e.1 += 1;
        }
        totals.graphs += 1;
        if totals.graphs.is_multiple_of(REPORT_EVERY) {
            let mut rows: Vec<_> = totals.by_key.iter().collect();
            rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            let all: f64 = rows.iter().map(|r| r.1.0).sum();
            let mut out = format!(
                "op profile after {} graphs: {:.1} ms GPU in total\n",
                totals.graphs, all
            );
            // With `dev`, every key: a per-device diff needs the small ops too.
            let shown = if per_device() { rows.len() } else { 60 };
            for (k, (ms, calls)) in rows.iter().take(shown) {
                out += &format!(
                    "  {:>10.1} ms {:>5.1}% {:>7} calls  {k}\n",
                    ms,
                    100.0 * ms / all.max(1e-9),
                    calls
                );
            }
            eprint!("{out}");
        }
        Ok(())
    }
}
