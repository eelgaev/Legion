//! A per-op GPU time profile for development, on with `LEGION_OP_PROFILE=1` (CUDA only): every
//! node of a graph runs as a segment of its own between two CUDA events, and the times add up
//! per op and tensor name (layer numbers stripped). The cumulative table goes to stderr every
//! [`REPORT_EVERY`] profiled graphs. Running node by node disables ggml's CUDA graphs and
//! fusions, so totals are somewhat higher than in a normal run; the split is what it is for.
//!
//! A node that reads a weight (a matmul, or an external op taking one) is known by that weight,
//! its type, shape and the node's columns (tokens, `x<n>`), and its row shows the weight bytes
//! it read per millisecond: GB/s, a matvec's achieved weight bandwidth (#1176). An expert stack
//! counts at most every routed choice a different expert, so its GB/s at several tokens is an
//! upper bound. Caches and states are not weights ([`weight_name`]), and an external op's
//! columns come from its activation, not its (often flat) output ([`act_tokens`]).

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
    /// (milliseconds, calls, weight bytes read) per key.
    by_key: HashMap<String, (f64, u64, u64)>,
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

/// Whether a leaf named `name` (in a weights buffer) can be a weight. Model tensors carry
/// GGUF-style dotted names (`blk.3.attn_q.weight`, `output.weight`, `ssd.3.gate` for streamed
/// experts); the KV caches and recurrent states that share their kind of buffer do not
/// (`cache_k-3`, `host_cache_k-3`, `ssm_state-3`, `conv_state-3`, `pool_k-3`, `spec_q-3`), so
/// their reads never count as weight bytes.
fn weight_name(name: &[u8]) -> bool {
    name.contains(&b'.')
}

/// The tokens of an activation of shape `ne` that a weight with rows of `k` elements reads:
/// the dimensions after the leading ones that make up one row of `k` (`[k, n]` and
/// `[k / 4, 4, n]` are `n` tokens). `None` if no leading run of dimensions is `k` long.
fn act_tokens(ne: [i64; 4], k: i64) -> Option<i64> {
    let mut lead = 1i64;
    for (d, &n) in ne.iter().enumerate() {
        lead = lead.saturating_mul(n.max(1));
        if lead == k {
            return Some(ne[d + 1..].iter().map(|&n| n.max(1)).product());
        }
        if lead > k {
            return None;
        }
    }
    None
}

/// The tensor a view (or a view of a view) reads from.
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn root(t: *const ffi::ggml_tensor) -> *const ffi::ggml_tensor {
    unsafe {
        let mut r = t;
        while !(*r).view_src.is_null() {
            r = (*r).view_src;
        }
        r
    }
}

/// Whether leaf `t` lives in a weights buffer (weights, and also caches and states).
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn in_weights_buffer(t: *const ffi::ggml_tensor) -> bool {
    unsafe {
        (*t).op == ffi::ggml_op_GGML_OP_NONE
            && !(*t).buffer.is_null()
            && ffi::ggml_backend_buffer_get_usage((*t).buffer)
                == ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS
    }
}

/// Whether leaf `t` is a weight: in a weights buffer, or a layer tensor (`blk.*`: expert cache
/// slots live in buffers of their own), and named like one ([`weight_name`]: not a cache or a
/// state).
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn is_weight(t: *const ffi::ggml_tensor) -> bool {
    unsafe {
        if (*t).op != ffi::ggml_op_GGML_OP_NONE {
            return false;
        }
        let name = std::ffi::CStr::from_ptr((*t).name.as_ptr()).to_bytes();
        weight_name(name) && (in_weights_buffer(t) || name.starts_with(b"blk."))
    }
}

/// The columns (tokens) of external op `node` reading weight `w`: from the first activation
/// source whose leading dimensions make up one of `w`'s rows ([`act_tokens`]). An external op's
/// output is often one flat buffer (`qwen4_hc_mix`, a `dense_mv` of several weights), whose
/// `ne[1]` is 1 at any step size. `None` if no source fits.
///
/// # Safety
/// `node` must be a valid tensor whose sources are live; `w` one of them.
unsafe fn external_tokens(
    node: *const ffi::ggml_tensor,
    w: *const ffi::ggml_tensor,
) -> Option<i64> {
    unsafe {
        (*node).src.iter().copied().find_map(|a| {
            if a.is_null() || (*a).type_ == ffi::ggml_type_GGML_TYPE_I32 {
                return None;
            }
            let r = root(a);
            if is_weight(r) || in_weights_buffer(r) {
                return None;
            }
            act_tokens((*a).ne, (*w).ne[0])
        })
    }
}

/// The biggest weight `node` reads (a view's root is the weight), the weight bytes it reads in
/// all (an expert stack with routing ids at most every routed choice a different expert;
/// embedding lookups not counted) and its columns (tokens: the routing ids' for an expert
/// stack, [`external_tokens`] for an external op, else the node's `ne[1]`). `None` without a
/// weight of 1 MiB.
///
/// # Safety
/// `node` must be a valid tensor whose sources are live.
unsafe fn weight_read(
    node: *const ffi::ggml_tensor,
) -> Option<(*const ffi::ggml_tensor, u64, i64)> {
    unsafe {
        if (*node).op == ffi::ggml_op_GGML_OP_GET_ROWS {
            return None;
        }
        let ids = (*node).src.iter().copied().find(|&s| {
            !s.is_null() && (*s).type_ == ffi::ggml_type_GGML_TYPE_I32 && (*s).ne[0] <= 64
        });
        let mut best: *const ffi::ggml_tensor = std::ptr::null();
        let (mut best_bytes, mut total) = (0u64, 0u64);
        // The weights read, in source order, for the columns of an external op.
        let mut weights: Vec<*const ffi::ggml_tensor> = Vec::new();
        for &src in (*node).src.iter() {
            if src.is_null() {
                continue;
            }
            if !is_weight(root(src)) {
                continue;
            }
            weights.push(src);
            let bytes = match ids {
                Some(ids) if (*src).ne[2] > 1 => {
                    let pairs = (*ids).ne[0].max(1) as u64 * (*ids).ne[1].max(1) as u64;
                    (*src).nb[2] as u64 * pairs.min((*src).ne[2] as u64)
                }
                _ => ffi::ggml_nbytes(src) as u64,
            };
            total += bytes;
            if bytes > best_bytes {
                best = src;
                best_bytes = bytes;
            }
        }
        if best_bytes < 1 << 20 {
            return None;
        }
        let cols = match ids {
            Some(ids) if (*best).ne[2] > 1 => (*ids).ne[1],
            _ if op_of(node).is_some() => std::iter::once(best)
                .chain(weights.iter().copied().filter(|&w| w != best))
                .find_map(|w| external_tokens(node, w))
                .unwrap_or((*node).ne[1]),
            _ => (*node).ne[1],
        };
        Some((best, total, cols))
    }
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
        let mut marks: Vec<(*mut c_void, *mut c_void, String, u64)> = Vec::new();
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
            let read = weight_read(node);
            let key = if let (false, Some((w, _, cols))) = (glm_linear, read) {
                // Any node reading a weight (#1176): its weight, type, shape and columns.
                let w = &*w;
                let wn = std::ffi::CStr::from_ptr(w.name.as_ptr()).to_string_lossy();
                let wn = match wn.strip_prefix("blk.") {
                    Some(r) => r.split_once('.').map_or(r, |(_, t)| t).to_string(),
                    None => wn.into_owned(),
                };
                let wn = wn.split(" (").next().unwrap_or(&wn).to_string();
                let ty = std::ffi::CStr::from_ptr(ffi::ggml_type_name(w.type_)).to_string_lossy();
                Some(format!(
                    "{op} {wn} [{ty}] {}x{}x{} x{cols}",
                    w.ne[0], w.ne[1], w.ne[2]
                ))
            } else {
                None
            };
            let name = if key.is_some() {
                String::new()
            } else if (glm_linear
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
            let key = key.unwrap_or_else(|| key_of(&op, &name));
            marks.push((a, b, format!("{dev}{key}"), read.map_or(0, |r| r.1)));
        }
        cudaStreamSynchronize(stream);
        let mut guard = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        let totals = guard.get_or_insert_with(Totals::default);
        for (a, b, key, bytes) in marks {
            let mut ms = 0f32;
            cudaEventElapsedTime(&mut ms, a, b);
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            let e = totals.by_key.entry(key).or_default();
            e.0 += ms as f64;
            e.1 += 1;
            e.2 += bytes;
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
            for (k, (ms, calls, bytes)) in rows.iter().take(shown) {
                // Weight bytes per millisecond: GB/s.
                let rate = if *bytes > 0 {
                    format!("{:>6.1} GB/s", *bytes as f64 / ms.max(1e-9) / 1e6)
                } else {
                    " ".repeat(11)
                };
                out += &format!(
                    "  {:>10.1} ms {:>5.1}% {:>7} calls {rate}  {k}\n",
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

#[cfg(test)]
mod tests;
