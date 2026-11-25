//! Where a compute buffer's plan peaks (#1183): the tensors alive in the allocator's buffer at
//! the node whose output ends highest in it, for each plan of a graph (as built, then as the
//! relayout rewrote it: ggml keeps the bigger buffer of the two, unless the plan as built was
//! out of the buffer, "as built, unbacked": [`super::ComputeAllocator::headroom`]). A load logs
//! it for the prompt step whose buffers it reserves, so a log names the tensors that size those
//! buffers. It only reads the plan: nothing it logs changes where tensors are or what runs.

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};

use super::ffi;

#[cfg(test)]
mod tests;

/// Whether allocations log their plans' peaks ([`set_report`]).
static REPORT: AtomicBool = AtomicBool::new(false);

/// Tensors a peak's log line lists, biggest first.
const LISTED: usize = 12;

/// Makes the graphs allocated from now on log where their plans peak (on while a load measures
/// the prompt step whose buffers it reserves, off after).
pub fn set_report(on: bool) {
    REPORT.store(on, Ordering::Relaxed);
}

/// Whether allocations log their plans' peaks now.
pub(super) fn reporting() -> bool {
    REPORT.load(Ordering::Relaxed)
}

/// A tensor's memory in a buffer and the nodes it is alive across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    /// The node making it (0 for a tensor the graph's nodes only read: an input).
    pub first: usize,
    /// The last node reading it (or a view of it); the graph's node count for one ggml never
    /// frees (an output, or one an unread node is or views: [`spans`]).
    pub last: usize,
    /// Its bytes in the buffer, `start..end`.
    pub start: usize,
    pub end: usize,
}

/// The peak of one buffer's `spans`: the span ending highest in it (the first of equals), and
/// the spans alive at the node making it (`first <= node <= last`), biggest first. None
/// without spans.
pub(crate) fn peak(spans: &[Span]) -> Option<(usize, Vec<usize>)> {
    let top = (0..spans.len()).max_by_key(|&i| (spans[i].end, std::cmp::Reverse(i)))?;
    let at = spans[top].first;
    let mut live: Vec<usize> = (0..spans.len())
        .filter(|&i| spans[i].first <= at && at <= spans[i].last)
        .collect();
    live.sort_by_key(|&i| (std::cmp::Reverse(spans[i].end - spans[i].start), i));
    Some((top, live))
}

/// The bytes the spans cover together (each byte once: an in-place node shares its source's).
pub(crate) fn covered(spans: impl Iterator<Item = Span>) -> usize {
    let mut r: Vec<(usize, usize)> = spans
        .map(|s| (s.start, s.end))
        .filter(|&(a, b)| b > a)
        .collect();
    r.sort_unstable();
    let mut total = 0;
    let mut cur: Option<(usize, usize)> = None;
    for (a, b) in r {
        cur = match cur {
            Some((ca, cb)) if a <= cb => Some((ca, cb.max(b))),
            Some((ca, cb)) => {
                total += cb - ca;
                Some((a, b))
            }
            None => Some((a, b)),
        };
    }
    total + cur.map_or(0, |(a, b)| b - a)
}

/// The tensor `t` is a view of (through any views), or `t`.
///
/// # Safety
/// A valid tensor.
unsafe fn root(mut t: *mut ffi::ggml_tensor) -> *mut ffi::ggml_tensor {
    // SAFETY: the caller's; a view's source is a valid tensor.
    unsafe {
        while !(*t).view_src.is_null() {
            t = (*t).view_src;
        }
    }
    t
}

/// `t` for a log line: its op (a Legion op's name), name, type, shape and MiB.
///
/// # Safety
/// A valid tensor.
unsafe fn label(t: *mut ffi::ggml_tensor, bytes: usize) -> String {
    // SAFETY: the caller's; ggml's names are NUL-terminated.
    unsafe {
        let op = match super::op_of(t) {
            Some(e) => e.name.to_string(),
            None => CStr::from_ptr(ffi::ggml_op_desc(t))
                .to_string_lossy()
                .into_owned(),
        };
        let name = CStr::from_ptr((*t).name.as_ptr()).to_string_lossy();
        let ty = CStr::from_ptr(ffi::ggml_type_name((*t).type_)).to_string_lossy();
        let ne = (*t).ne;
        format!(
            "{op} '{name}' {ty} [{}, {}, {}, {}] {:.1} MiB",
            ne[0],
            ne[1],
            ne[2],
            ne[3],
            bytes as f64 / (1u64 << 20) as f64
        )
    }
}

/// The tensors with memory in the buffers holding `placed` (see [`log`]) and their spans, in
/// `graph` (allocated): with those buffers, in the order first met. A tensor is alive from the
/// node making it to the last node reading it or a view of it, or to the graph's end (its node
/// count) where ggml's allocator never frees it: an output, or a tensor that a node no node
/// reads is or views (an output view, say; not a `GGML_OP_NONE` view, which ggml does not
/// count).
///
/// # Safety
/// `graph` is a live, allocated graph; `placed` its tensors.
unsafe fn spans(
    graph: *mut ffi::ggml_cgraph,
    placed: &[*mut ffi::ggml_tensor],
) -> (
    Vec<ffi::ggml_backend_buffer_t>,
    Vec<(ffi::ggml_backend_buffer_t, *mut ffi::ggml_tensor, Span)>,
) {
    // SAFETY: the caller's; nodes, their sources and leafs are valid tensors.
    unsafe {
        let mut buffers: Vec<ffi::ggml_backend_buffer_t> = Vec::new();
        for &t in placed {
            let b = (*root(t)).buffer;
            if !b.is_null() && !buffers.contains(&b) {
                buffers.push(b);
            }
        }
        let n = ffi::ggml_graph_n_nodes(graph).max(0) as usize;
        let nodes: Vec<*mut ffi::ggml_tensor> = (0..n)
            .map(|i| ffi::ggml_graph_node(graph, i as i32))
            .collect();
        // The last node reading each tensor (by its root).
        let mut last: HashMap<usize, usize> = HashMap::new();
        let mut made: HashMap<usize, usize> = HashMap::new();
        let mut read: HashSet<usize> = HashSet::new();
        for (i, &t) in nodes.iter().enumerate() {
            made.entry(t as usize).or_insert(i);
            if !(*t).view_src.is_null() {
                last.insert(root(t) as usize, i);
            }
            for &s in (*t).src.iter().filter(|s| !s.is_null()) {
                last.insert(root(s) as usize, i);
                read.insert(s as usize);
            }
        }
        // Kept to the end: ggml frees a tensor once its readers and its views' readers have
        // run, so never one that a node no node reads is or views.
        let unread_roots: HashSet<usize> = (nodes.iter())
            .filter(|&&t| {
                !read.contains(&(t as usize))
                    && ((*t).view_src.is_null() || (*t).op != ffi::ggml_op_GGML_OP_NONE)
            })
            .map(|&t| root(t) as usize)
            .collect();
        let mut seen: HashSet<usize> = HashSet::new();
        let mut spans: Vec<(ffi::ggml_backend_buffer_t, *mut ffi::ggml_tensor, Span)> = Vec::new();
        let all = nodes
            .iter()
            .copied()
            .chain(super::graph_uid::leafs(graph).iter().copied());
        for t in all.flat_map(|t| std::iter::once(t).chain((*t).src.iter().copied())) {
            if t.is_null() || !(*t).view_src.is_null() || (*t).data.is_null() {
                continue;
            }
            let b = (*t).buffer;
            if b.is_null() || !buffers.contains(&b) || !seen.insert(t as usize) {
                continue;
            }
            let start = (*t).data as usize - ffi::ggml_backend_buffer_get_base(b) as usize;
            let first = made.get(&(t as usize)).copied().unwrap_or(0);
            let output = (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0;
            let kept = output || unread_roots.contains(&(t as usize));
            let read = last.get(&(t as usize)).copied().unwrap_or(first);
            let span = Span {
                first,
                last: if kept { n } else { read.max(first) },
                start,
                end: start + ffi::ggml_backend_buffer_get_alloc_size(b, t),
            };
            spans.push((b, t, span));
        }
        (buffers, spans)
    }
}

/// Logs where the plan of `graph` (`plan`: which one), allocated just now, peaks in the
/// buffers holding `placed` (the tensors the plan gave memory: nodes and inputs without memory
/// before it), `buffer` the bytes of the allocator that made the plan.
///
/// # Safety
/// `graph` is a live, allocated graph; `placed` its tensors.
pub(super) unsafe fn log(
    graph: *mut ffi::ggml_cgraph,
    placed: &[*mut ffi::ggml_tensor],
    plan: &str,
    buffer: usize,
) {
    // SAFETY: the caller's; nodes, their sources and leafs are valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph).max(0) as usize;
        let (buffers, spans) = spans(graph, placed);
        for (k, &b) in buffers.iter().enumerate() {
            let mine: Vec<(*mut ffi::ggml_tensor, Span)> = (spans.iter())
                .filter(|s| s.0 == b)
                .map(|s| (s.1, s.2))
                .collect();
            let only: Vec<Span> = mine.iter().map(|s| s.1).collect();
            let Some((top, live)) = peak(&only) else {
                continue;
            };
            let mib = |x: usize| x >> 20;
            let listed: Vec<String> = (live.iter().take(LISTED))
                .map(|&i| label(mine[i].0, only[i].end - only[i].start))
                .collect();
            tracing::info!(
                plan,
                buffer = k,
                buffer_mib = mib(buffer),
                peak_mib = mib(only[top].end),
                node = only[top].first,
                nodes = n,
                at = %label(mine[top].0, only[top].end - only[top].start),
                live = live.len(),
                live_mib = mib(covered(live.iter().map(|&i| only[i]))),
                biggest = %listed.join("; "),
                "compute plan peak"
            );
        }
    }
}
