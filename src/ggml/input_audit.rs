//! A debug check of the kept graphs' inputs (#836, after the stale bench input of #1028):
//! ggml's graph allocator frees an input after its last consumer like any other tensor (only
//! outputs are never freed), so a later node of the same graph may be placed over it. A graph
//! that runs again then reads whatever that node left there, unless every run fills the input
//! again first or the input is an output too.
//!
//! With the check on (`legion serve --audit-graph-inputs`, hidden), every graph that is kept
//! ([`super::KeptGraph::new`]) is scanned once it is allocated: each input it reads (and each
//! leaf) in the allocator's compute buffers whose memory a computed node of the graph overlaps
//! is listed, as one of
//!
//! - output: marked an output too (the allocator places nothing over it);
//! - refilled: in the kept graph's tensor list, the tensors its caller addresses (and fills
//!   before every run: the check cannot see that, the audit of each caller does);
//! - unread: only ever the shape source of a FILL (`ctx.fill(ctx.new_tensor(..), c)`), whose
//!   data no op reads;
//! - scratch: named `*scratch*`, memory an op writes before it reads it, every run;
//! - **not refilled**: none of these, a value set once would be stale on the next run.
//!
//! The last kind is logged as a warning; a summary line per distinct graph says how many
//! there were of each. Each distinct line is logged once. No effect on what runs.
//!
//! Graphs kept in one compute buffer are checked against each other too ([`cross`]): a run of
//! one may write over another's input set once.
//!
//! Memory outside the compute buffers (weights, the KV and state pools) is not the
//! allocator's to recycle: a node placed there is placed on purpose (`Tensor::place_in`, e.g.
//! a gated-delta-net output landing on its state slots), so it is not checked.

use super::ffi;
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

mod cross;
pub(crate) use cross::{Registration, on_run, register};

#[cfg(test)]
mod tests;

static ON: AtomicBool = AtomicBool::new(false);

/// Lines logged so far (each distinct one once).
static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Turns the check on or off for the graphs kept from now on.
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_enabled`].
pub(crate) fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Why an overlapped input is (or is not) safe to run again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Filled by the caller before every run (in the kept graph's tensor list).
    Refilled,
    /// An output too.
    Output,
    /// Only a FILL's shape source: no op reads its data.
    Unread,
    /// Scratch an op writes before it reads it.
    Scratch,
    /// Neither: a value set once is overwritten by the first run.
    NotRefilled,
}

/// An input (or compute-buffer leaf) some computed node of its graph overlaps.
#[derive(Debug, Clone)]
pub struct Finding {
    pub name: String,
    pub bytes: usize,
    pub kind: Kind,
    /// The overlapping nodes: op, name, offset from the input's start and size.
    pub over: Vec<String>,
}

fn name_of(t: *const ffi::ggml_tensor) -> String {
    // SAFETY: a live tensor; its name is a NUL-terminated array.
    let s = unsafe { std::ffi::CStr::from_ptr((*t).name.as_ptr()) };
    let s = s.to_string_lossy();
    if s.is_empty() {
        "(unnamed)".into()
    } else {
        s.into_owned()
    }
}

/// The tensor whose memory `t` is (a view's source, down the chain).
///
/// # Safety
/// A live tensor whose view sources are live.
unsafe fn root(mut t: *mut ffi::ggml_tensor) -> *mut ffi::ggml_tensor {
    unsafe {
        while !(*t).view_src.is_null() {
            t = (*t).view_src;
        }
    }
    t
}

/// The inputs of `graph` (allocated) that a computed node of it overlaps; `refilled`: the
/// tensors its runs fill before every run.
///
/// # Safety
/// `graph` and every tensor it references (and `refilled`) are live, and the graph is
/// allocated.
pub unsafe fn scan(
    graph: *mut ffi::ggml_cgraph,
    refilled: &[*mut ffi::ggml_tensor],
) -> Vec<Finding> {
    let input = ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32;
    let output = ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32;
    let none = ffi::ggml_op_GGML_OP_NONE;
    let fill = ffi::ggml_op_GGML_OP_FILL;
    let compute_usage = ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    // SAFETY: the caller's.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph);
        let nodes: Vec<*mut ffi::ggml_tensor> =
            (0..n).map(|i| ffi::ggml_graph_node(graph, i)).collect();
        // Computed nodes with memory of their own in the allocator's compute buffers: what the
        // allocator placed.
        let placed: Vec<*mut ffi::ggml_tensor> = (nodes.iter().copied())
            .filter(|&t| (*t).op != none && (*t).view_src.is_null() && !(*t).data.is_null())
            .filter(|&t| {
                !(*t).buffer.is_null()
                    && ffi::ggml_backend_buffer_get_usage((*t).buffer) == compute_usage
            })
            .collect();
        let compute: HashSet<usize> = placed.iter().map(|&t| (*t).buffer as usize).collect();
        // The tensors some op reads the data of (all sources but a FILL's, whose shape only
        // counts).
        let mut read = HashSet::new();
        for &node in &nodes {
            for (i, &src) in (*node).src.iter().enumerate() {
                if !src.is_null() && !((*node).op == fill && i == 0 && (*src).view_src.is_null()) {
                    read.insert(root(src) as usize);
                }
            }
        }
        // The candidates: inputs (by flag) and leaves in a compute buffer, read by a node or
        // listed by the caller; each once, in the order met.
        let mut seen = HashSet::new();
        let mut cands = Vec::new();
        let mut consider = |t: *mut ffi::ggml_tensor| {
            if t.is_null() {
                return;
            }
            let t = root(t);
            if (*t).data.is_null() || !seen.insert(t as usize) {
                return;
            }
            if !compute.contains(&((*t).buffer as usize)) {
                return;
            }
            if (*t).flags & input != 0 || (*t).op == none {
                cands.push(t);
            }
        };
        for &t in refilled {
            consider(t);
        }
        for &node in &nodes {
            for &src in &(*node).src {
                consider(src);
            }
        }
        let listed: HashSet<usize> = refilled.iter().map(|&t| root(t) as usize).collect();
        let mut out = Vec::new();
        for t in cands {
            let start = (*t).data as usize;
            let bytes = ffi::ggml_nbytes(t);
            let end = start + bytes;
            let over: Vec<String> = (placed.iter().copied())
                .filter(|&p| p != t)
                .filter(|&p| {
                    let s = (*p).data as usize;
                    s < end && start < s + ffi::ggml_nbytes(p)
                })
                .map(|p| {
                    let op = std::ffi::CStr::from_ptr(ffi::ggml_op_desc(p)).to_string_lossy();
                    let at = (*p).data as isize - start as isize;
                    format!(
                        "{op} '{}' at {at:+} B, {} B",
                        name_of(p),
                        ffi::ggml_nbytes(p)
                    )
                })
                .collect();
            if over.is_empty() {
                continue;
            }
            let name = name_of(t);
            let kind = if (*t).flags & output != 0 {
                Kind::Output
            } else if listed.contains(&(t as usize)) {
                Kind::Refilled
            } else if !read.contains(&(t as usize)) {
                Kind::Unread
            } else if name.contains("scratch") {
                Kind::Scratch
            } else {
                Kind::NotRefilled
            };
            out.push(Finding {
                name,
                bytes,
                kind,
                over,
            });
        }
        out
    }
}

/// A finding's name with its trailing `-<group>` index cut, for the summary.
fn base_name(name: &str) -> &str {
    match name.rsplit_once('-') {
        Some((b, i)) if i.chars().all(|c| c.is_ascii_digit()) => b,
        _ => name,
    }
}

/// The summary line of a graph's findings: counts per kind and the inputs' names by kind.
pub fn summary(n_nodes: usize, findings: &[Finding]) -> String {
    let mut parts = Vec::new();
    for (kind, label) in [
        (Kind::Refilled, "refilled"),
        (Kind::Output, "output"),
        (Kind::Unread, "unread"),
        (Kind::Scratch, "scratch"),
        (Kind::NotRefilled, "NOT refilled"),
    ] {
        let mut names: Vec<&str> = (findings.iter())
            .filter(|f| f.kind == kind)
            .map(|f| base_name(&f.name))
            .collect();
        if names.is_empty() {
            continue;
        }
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        parts.push(format!("{label} {count} [{}]", names.join(", ")));
    }
    if parts.is_empty() {
        parts.push("none".into());
    }
    format!(
        "graph input audit: {n_nodes} nodes, inputs overlapped by a node: {}",
        parts.join("; ")
    )
}

/// Whether `line` is new (and notes it).
fn first_time(line: &str) -> bool {
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    seen.get_or_insert_with(HashSet::new)
        .insert(line.to_string())
}

/// [`scan`] and logs it, when the check is on.
///
/// # Safety
/// [`scan`]'s.
pub(crate) unsafe fn check(graph: *mut ffi::ggml_cgraph, refilled: &[*mut ffi::ggml_tensor]) {
    if !enabled() {
        return;
    }
    // SAFETY: the caller's.
    let (findings, n) = unsafe { (scan(graph, refilled), ffi::ggml_graph_n_nodes(graph)) };
    for f in findings.iter().filter(|f| f.kind == Kind::NotRefilled) {
        let line = format!(
            "graph input audit: input '{}' ({} B) is neither refilled before every run nor an \
             output, and a node is placed over it: {}",
            f.name,
            f.bytes,
            f.over.join("; ")
        );
        if first_time(&line) {
            tracing::warn!("{line}");
        }
    }
    let line = summary(n as usize, &findings);
    if first_time(&line) {
        tracing::info!("{line}");
    }
}
