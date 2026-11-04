//! Why a stage-graph exec update was refused (#836; `legion serve --stage-graph-stats`).
//!
//! A new big capture (at least [`super::spare::MIN_NODES`] CUDA graph nodes) updates a spare
//! executable graph of its node count when CUDA accepts it ([`super::spare`]); otherwise it is
//! instantiated (~18 ms of host time for a ~3,400-node verify graph on a GB10). While the log is
//! on, each big capture's topology is noted as one id per node (its type, and a kernel's
//! function name), compared with the newest big capture of the same device of the same node
//! count, else of the nearest one, and logged: how many nodes the two share at the front and at
//! the back, and which kernels the differing middle holds more or fewer of. A refused update
//! logs CUDA's `cudaGraphExecUpdateResult` and the error node (its index in the new capture, its
//! type and kernel). Nothing of this runs with the log off.

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

#[cfg(test)]
mod tests;

/// Big captures remembered per device for the comparison.
const RECENT: usize = 8;

/// Kernels named in a difference line, the largest count changes first.
const NAMED: usize = 8;

/// Node kinds by id (a node type and, for kernels, the function's name).
#[derive(Default)]
pub(crate) struct Names {
    ids: HashMap<String, u32>,
    list: Vec<String>,
}

impl Names {
    /// The id of `name`, made the first time.
    pub(crate) fn id(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.ids.get(name) {
            return i;
        }
        let i = self.list.len() as u32;
        self.list.push(name.to_string());
        self.ids.insert(name.to_string(), i);
        i
    }

    /// The name of id `i` (`?` for an unknown id).
    pub(crate) fn name(&self, i: u32) -> &str {
        self.list.get(i as usize).map_or("?", String::as_str)
    }
}

/// How two captures' node lists differ ([`compare`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TopoDiff {
    /// Nodes equal at the front.
    pub prefix: usize,
    /// Nodes equal at the back (not overlapping `prefix` in either list).
    pub suffix: usize,
    /// Of the differing middles, each kind whose count differs: (kind, count in the new list
    /// minus count in the old one), the largest changes first, then by kind.
    pub counts: Vec<(u32, i64)>,
}

impl TopoDiff {
    /// Whether the two lists are the same kinds in the same order.
    pub(crate) fn same(&self, old: usize, new: usize) -> bool {
        old == new && self.prefix == new
    }
}

/// Compares node list `new` with `old`.
pub(crate) fn compare(old: &[u32], new: &[u32]) -> TopoDiff {
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let room = old.len().min(new.len()) - prefix;
    let suffix = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take(room)
        .take_while(|(a, b)| a == b)
        .count();
    let mut d: HashMap<u32, i64> = HashMap::new();
    for &k in &old[prefix..old.len() - suffix] {
        *d.entry(k).or_default() -= 1;
    }
    for &k in &new[prefix..new.len() - suffix] {
        *d.entry(k).or_default() += 1;
    }
    let mut counts: Vec<(u32, i64)> = d.into_iter().filter(|&(_, n)| n != 0).collect();
    counts.sort_by_key(|&(k, n)| (std::cmp::Reverse(n.unsigned_abs()), k));
    TopoDiff {
        prefix,
        suffix,
        counts,
    }
}

/// The big captures of each device, newest last.
#[derive(Default)]
pub(crate) struct Recent {
    list: VecDeque<(i32, Vec<u32>)>,
}

impl Recent {
    /// The newest capture of device `dev` with `nodes` nodes, else the newest of the nearest
    /// node count (the smaller one on a tie).
    pub(crate) fn nearest(&self, dev: i32, nodes: usize) -> Option<&[u32]> {
        self.list
            .iter()
            .enumerate()
            .filter(|(_, (d, _))| *d == dev)
            .min_by_key(|(i, (_, s))| {
                (
                    s.len().abs_diff(nodes),
                    s.len() > nodes,
                    std::cmp::Reverse(*i),
                )
            })
            .map(|(_, (_, s))| s.as_slice())
    }

    /// Remembers `sig` as device `dev`'s newest; the oldest past [`RECENT`] go.
    pub(crate) fn push(&mut self, dev: i32, sig: Vec<u32>) {
        self.list.push_back((dev, sig));
        let mine = self.list.iter().filter(|(d, _)| *d == dev).count();
        if mine > RECENT
            && let Some(i) = self.list.iter().position(|(d, _)| *d == dev)
        {
            self.list.remove(i);
        }
    }
}

/// The name of a `cudaGraphExecUpdateResult` (driver_types.h).
pub(crate) fn result_name(code: i32) -> &'static str {
    match code {
        0 => "success",
        1 => "error",
        2 => "topology_changed",
        3 => "node_type_changed",
        4 => "function_changed",
        5 => "parameters_changed",
        6 => "not_supported",
        7 => "unsupported_function_change",
        8 => "attributes_changed",
        _ => "unknown",
    }
}

/// The name of a `CUgraphNodeType` (cuda.h).
pub(crate) fn node_type_name(t: u32) -> &'static str {
    match t {
        0 => "kernel",
        1 => "memcpy",
        2 => "memset",
        3 => "host",
        4 => "graph",
        5 => "empty",
        6 => "event_wait",
        7 => "event_record",
        8 => "ext_semaphore_signal",
        9 => "ext_semaphore_wait",
        10 => "mem_alloc",
        11 => "mem_free",
        12 => "batch_mem_op",
        13 => "conditional",
        _ => "unknown",
    }
}

struct State {
    names: Names,
    recent: Recent,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        names: Names::default(),
        recent: Recent::default(),
    })
});

/// The id of node kind `name` ([`Names::id`]).
pub(crate) fn kind_id(name: &str) -> u32 {
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .names
        .id(name)
}

/// `diff`'s count changes as `name:+n` / `name:-n`, at most [`NAMED`] of them.
fn counts_text(names: &Names, diff: &TopoDiff) -> String {
    let mut out = String::new();
    for &(k, n) in diff.counts.iter().take(NAMED) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&format!("{}:{n:+}", names.name(k)));
    }
    if diff.counts.len() > NAMED {
        out.push_str(&format!(" (+{} more)", diff.counts.len() - NAMED));
    }
    out
}

/// Logs how big capture `sig` (device `dev`) differs from the nearest one before it and
/// remembers it. Only called while the log is on.
pub(crate) fn note_capture(dev: i32, sig: Vec<u32>) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match st.recent.nearest(dev, sig.len()) {
        Some(old) => {
            let d = compare(old, &sig);
            let at = |s: &[u32], i: usize| s.get(i).map_or("-", |&k| st.names.name(k));
            tracing::info!(
                dev,
                nodes = sig.len(),
                old_nodes = old.len(),
                same = d.same(old.len(), sig.len()),
                prefix = d.prefix,
                suffix = d.suffix,
                first_old = at(old, d.prefix),
                first_new = at(&sig, d.prefix),
                counts = counts_text(&st.names, &d),
                "stage graph big capture vs nearest (--stage-graph-stats)"
            );
        }
        None => tracing::info!(
            dev,
            nodes = sig.len(),
            "stage graph big capture, first (--stage-graph-stats)"
        ),
    }
    st.recent.push(dev, sig);
}

/// Logs how many memcpy nodes of a big capture (device `dev`, `nodes` CUDA graph nodes) were
/// made copy kernel nodes ([`super::copy_kernel`]) and how many were left as they were.
pub(crate) fn log_copies(dev: i32, nodes: usize, made: usize, left: usize) {
    tracing::info!(
        dev,
        nodes,
        made,
        left,
        "stage graph memcpy nodes as copy kernels (--stage-graph-stats)"
    );
}

/// What a refused update's log line says about the error node ([`log_refused`]).
#[derive(Debug, Default)]
pub(crate) struct ErrorNode {
    /// Its index in the new capture's node list (`None`: no error node, or not found).
    pub index: Option<usize>,
    /// Its kind (type, and kernel name).
    pub kind: String,
    /// Its parameters, briefly (a kernel's grid, block and shared memory; a memset's or a
    /// memcpy's sizes).
    pub params: String,
    /// The index of the mismatched dependency (`errorFromNode`), if any.
    pub from: Option<usize>,
}

/// Logs a refused update of a spare (`stolen`: taken over from a kept capture): CUDA's status,
/// its result and the error node.
pub(crate) fn log_refused(status: i32, result: i32, stolen: bool, node: &ErrorNode) {
    tracing::info!(
        status,
        result = result_name(result),
        stolen,
        error_node = node.index.map_or(-1, |i| i as i64),
        error_kind = node.kind,
        error_params = node.params,
        error_from = node.from.map_or(-1, |i| i as i64),
        "stage graph exec update refused (--stage-graph-stats)"
    );
}
