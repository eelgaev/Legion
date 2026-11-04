//! The input check across kept graphs (#836): graphs kept in one compute buffer (the MTP
//! head's per kind, the piped step graphs, a meta backend's decode and mixed graphs:
//! [`crate::ggml::Context::alloc_graph_shared`]) overlap in it, so a run of one writes over
//! the others' tensors. An input of graph A set once (not refilled before every run: not in
//! its tensor list, not scratch) is then stale on A's next run if graph B ran in between and
//! B writes over it, with a node it computes or an input its caller fills. An output too
//! keeps it safe from A's own nodes only (see [`super::scan`]), not from B's.
//!
//! With `--audit-graph-inputs`, every kept graph is registered once allocated: the memory its
//! set-once inputs hold, and the memory its runs write in the allocator's compute buffers.
//! A new graph is compared with the graphs still kept: each pair where one writes over the
//! other's set-once input is logged (a possible hit). Every run of a kept graph is then noted
//! ([`on_run`]): a graph that runs after a graph writing over its set-once inputs ran since
//! its own last run (or its keeping) reads stale bytes, and that is logged as a warning (a hit).
//! Dropping a kept graph removes it. Each distinct line is logged once. No effect on what
//! runs; off by default it costs one relaxed load per run.

use super::ffi;
use super::{enabled, first_time, name_of, root};
use std::collections::HashSet;
use std::sync::Mutex;

#[cfg(test)]
mod tests;

/// Memory a tensor holds: its name, start and end address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub name: String,
    pub start: usize,
    pub end: usize,
}

impl Span {
    fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// What a kept graph keeps and writes in the compute buffers.
#[derive(Debug, Clone, Default)]
pub struct Spans {
    /// Inputs (and leaves) read by a node whose value must last from one run to the next: not
    /// refilled before every run, not scratch.
    pub held: Vec<Span>,
    /// What a run writes: the computed nodes placed in the buffers, and the tensors its caller
    /// fills (the kept graph's tensor list).
    pub writes: Vec<Span>,
}

/// The [`Spans`] of `graph` (allocated); `refilled`: the tensors its runs fill before every run.
///
/// # Safety
/// As for [`super::scan`].
pub unsafe fn spans(graph: *mut ffi::ggml_cgraph, refilled: &[*mut ffi::ggml_tensor]) -> Spans {
    let input = ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_INPUT as i32;
    let none = ffi::ggml_op_GGML_OP_NONE;
    let fill = ffi::ggml_op_GGML_OP_FILL;
    let compute_usage = ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    // SAFETY: the caller's.
    unsafe {
        let in_compute = |t: *mut ffi::ggml_tensor| {
            !(*t).data.is_null()
                && !(*t).buffer.is_null()
                && ffi::ggml_backend_buffer_get_usage((*t).buffer) == compute_usage
        };
        let span = |t: *mut ffi::ggml_tensor| {
            let start = (*t).data as usize;
            Span {
                name: name_of(t),
                start,
                end: start + ffi::ggml_nbytes(t),
            }
        };
        let n = ffi::ggml_graph_n_nodes(graph);
        let nodes: Vec<*mut ffi::ggml_tensor> =
            (0..n).map(|i| ffi::ggml_graph_node(graph, i)).collect();
        let mut out = Spans::default();
        let mut seen = HashSet::new();
        for &t in &nodes {
            if (*t).op != none
                && (*t).view_src.is_null()
                && in_compute(t)
                && seen.insert(t as usize)
            {
                out.writes.push(span(t));
            }
        }
        let listed: HashSet<usize> = refilled.iter().map(|&t| root(t) as usize).collect();
        for &t in refilled {
            let t = root(t);
            if in_compute(t) && seen.insert(t as usize) {
                out.writes.push(span(t));
            }
        }
        // The tensors some op reads the data of (all sources but a FILL's shape source), as in
        // `scan`.
        let mut held = HashSet::new();
        for &node in &nodes {
            for (i, &src) in (*node).src.iter().enumerate() {
                if src.is_null() || ((*node).op == fill && i == 0 && (*src).view_src.is_null()) {
                    continue;
                }
                let t = root(src);
                let leaf = (*t).flags & input != 0 || (*t).op == none;
                if !leaf || listed.contains(&(t as usize)) || !in_compute(t) {
                    continue;
                }
                if !held.insert(t as usize) {
                    continue;
                }
                let s = span(t);
                if !s.name.contains("scratch") {
                    out.held.push(s);
                }
            }
        }
        out
    }
}

/// A kept graph as the check knows it.
struct Entry {
    serial: u64,
    /// Its node count, to tell graphs apart in the log.
    nodes: usize,
    spans: Spans,
    /// The epoch of its last run, or of its keeping before it ran.
    mark: u64,
    /// The epoch of its last run (0: none yet).
    last_run: u64,
    /// The kept graphs that write over its set-once inputs.
    partners: Vec<u64>,
}

#[derive(Default)]
struct Registry {
    epoch: u64,
    entries: Vec<Entry>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

/// Logs `line` (a warning when `warn`) the first time it is met.
fn log(line: String, warn: bool) {
    if !first_time(&line) {
        return;
    }
    if warn {
        tracing::warn!("{line}");
    } else {
        tracing::info!("{line}");
    }
}

/// The names of `held`'s spans that a span of `writes` overlaps, sorted and deduplicated.
fn hits(held: &[Span], writes: &[Span]) -> Vec<String> {
    let mut names: Vec<String> = (held.iter())
        .filter(|h| writes.iter().any(|w| w.overlaps(h)))
        .map(|h| super::base_name(&h.name).to_string())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// The line for graph `a`'s set-once inputs `names` that graph `b` writes over.
fn pair_line(a: (u64, usize), b: (u64, usize), names: &[String]) -> String {
    format!(
        "graph input audit (across graphs): set-once input(s) [{}] of a kept graph of {} nodes \
         (#{}) share memory with what a kept graph of {} nodes (#{}) writes",
        names.join(", "),
        a.1,
        a.0,
        b.1,
        b.0
    )
}

/// Registers the kept graph `serial` with its spans; returns the pair lines it adds (for tests).
pub fn register_spans(serial: u64, nodes: usize, spans: Spans) -> Vec<String> {
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.get_or_insert_with(Registry::default);
    let mut entry = Entry {
        serial,
        nodes,
        spans,
        mark: reg.epoch,
        last_run: 0,
        partners: Vec::new(),
    };
    let mut lines = Vec::new();
    for other in &mut reg.entries {
        let mine = hits(&entry.spans.held, &other.spans.writes);
        if !mine.is_empty() {
            entry.partners.push(other.serial);
            lines.push(pair_line(
                (serial, nodes),
                (other.serial, other.nodes),
                &mine,
            ));
        }
        let theirs = hits(&other.spans.held, &entry.spans.writes);
        if !theirs.is_empty() {
            other.partners.push(serial);
            lines.push(pair_line(
                (other.serial, other.nodes),
                (serial, nodes),
                &theirs,
            ));
        }
    }
    reg.entries.push(entry);
    lines
}

/// The summary line of the kept graph `serial` (registered): its set-once inputs, and how
/// many of the graphs still kept when it was kept write where it writes (share its memory), so
/// a log without pairs shows what was compared.
pub fn summary_line(serial: u64) -> Option<String> {
    let reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let reg = reg.as_ref()?;
    let me = reg.entries.iter().find(|e| e.serial == serial)?;
    let shares = (reg.entries.iter())
        .filter(|e| e.serial != serial)
        .filter(|e| (e.spans.writes.iter()).any(|w| me.spans.writes.iter().any(|m| m.overlaps(w))))
        .count();
    let mut held: Vec<&str> = (me.spans.held.iter())
        .map(|h| super::base_name(&h.name))
        .collect();
    held.sort_unstable();
    held.dedup();
    Some(format!(
        "graph input audit (across graphs): a kept graph of {} nodes: {} set-once input(s) \
         [{}], shares memory with {} graph(s) still kept, {} of them writing over its set-once \
         inputs",
        me.nodes,
        me.spans.held.len(),
        held.join(", "),
        shares,
        me.partners.len()
    ))
}

/// Removes the kept graph `serial` (dropped).
pub fn unregister(serial: u64) {
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(reg) = reg.as_mut() else {
        return;
    };
    reg.entries.retain(|e| e.serial != serial);
    for e in &mut reg.entries {
        e.partners.retain(|&p| p != serial);
    }
}

/// Notes a run of the kept graph `serial`; returns the hit lines (for tests): for each graph
/// that writes over its set-once inputs and ran since its last run.
pub fn note_run(serial: u64) -> Vec<String> {
    let mut reg = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(reg) = reg.as_mut() else {
        return Vec::new();
    };
    reg.epoch += 1;
    let epoch = reg.epoch;
    let Some(i) = reg.entries.iter().position(|e| e.serial == serial) else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    let me = &reg.entries[i];
    for &p in &me.partners {
        let Some(other) = reg.entries.iter().find(|e| e.serial == p) else {
            continue;
        };
        if other.last_run > me.mark {
            let names = hits(&me.spans.held, &other.spans.writes);
            lines.push(format!(
                "graph input audit (across graphs): STALE: set-once input(s) [{}] of a kept \
                 graph of {} nodes (#{}) were written over by a run of a kept graph of {} nodes \
                 (#{}) since it last ran",
                names.join(", "),
                me.nodes,
                me.serial,
                other.nodes,
                other.serial
            ));
        }
    }
    let me = &mut reg.entries[i];
    me.mark = epoch;
    me.last_run = epoch;
    lines
}

/// Registers a kept graph when the check is on: [`spans`], [`register_spans`] and the lines
/// logged. The returned guard removes it when dropped.
///
/// # Safety
/// As for [`super::scan`].
pub(crate) unsafe fn register(
    serial: u64,
    graph: *mut ffi::ggml_cgraph,
    refilled: &[*mut ffi::ggml_tensor],
) -> Registration {
    if !enabled() {
        return Registration(None);
    }
    // SAFETY: the caller's.
    let (spans, n) = unsafe { (spans(graph, refilled), ffi::ggml_graph_n_nodes(graph)) };
    for line in register_spans(serial, n as usize, spans) {
        log(line, false);
    }
    if let Some(line) = summary_line(serial) {
        log(line, false);
    }
    Registration(Some(serial))
}

/// Notes a run of the kept graph `serial` when the check is on, warning about each hit
/// ([`note_run`]).
pub(crate) fn on_run(serial: u64) {
    if !enabled() {
        return;
    }
    for line in note_run(serial) {
        log(line, true);
    }
}

/// A kept graph's registration with the check ([`register`]); removes it when dropped.
#[derive(Default)]
pub(crate) struct Registration(Option<u64>);

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(serial) = self.0 {
            unregister(serial);
        }
    }
}
