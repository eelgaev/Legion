//! Spare executable graphs (#836): big captures dropped from the store are kept a while, so
//! a new capture of the same topology updates one (`cudaGraphExecUpdate`) instead of
//! instantiating its graph again.
//!
//! With MTP at several sequences the verify step's shape follows the sequences' padded pool
//! counts, which grow with the contexts: on a DGX Spark (GB10) at 4 parallel requests about 11
//! new verify shapes per 256 generated tokens, each a new kept graph whose capture took ~18 ms
//! of host time, nearly all of it `cudaGraphInstantiate` of ~3,400 kernel nodes. The shapes
//! differ in sizes and addresses, not in which kernels run in which order, and the kept graphs
//! they replace drop their captures soon after. CUDA can update an instantiated graph with the
//! node parameters of a topologically identical one; a launch of the updated exec runs the new
//! graph's nodes with the new graph's parameters, as an instantiation of it would. Where the
//! update is refused (another topology, a kernel change CUDA cannot apply) the spare is
//! destroyed and the graph instantiated as before.
//!
//! Captures of at least [`MIN_NODES`] CUDA graph nodes are kept at most `cap` per device, the
//! newest first ([`Spares::put`]); a new capture takes the newest spare of its node count.
//!
//! Smaller captures are kept in a list of their own, at most [`SMALL_SPARE_EXECS`] per device
//! (#1176). One instantiates in well under a millisecond, but a verify step over tensor
//! parallelism is ~100 of them (the pieces between two exchanges): on two DGX Sparks at 16
//! sequences each new verify shape (every ~0.9 s) instantiated 98 of them, 19-33 ms of host
//! time, partly in the step's tail where the GPU waits for the host. Kept, the sets of the
//! shapes left behind (dropped once unused, [`super::store`]) hand theirs to the next shape's
//! captures. `LEGION_DEBUG_SMALL_SPARES=0` keeps none of them (the way before, an A/B switch;
//! [`small_enabled`]).
//!
//! The kept graphs of the shapes a context grew out of hold their captures until they are
//! evicted, so at first there are no spares. A new capture with none then takes over the
//! executable graph of a capture of its node count still held but not launched for a while
//! (`--stage-graph-exec-steal-ms`, [`pick_idle`]): that set captures its piece again if it
//! ever runs again, as after a refused capture's change. Either way every launch runs the
//! graph last captured into it.
//!
//! A small capture's copies become kernel nodes and its cluster layout is read as a big one's
//! (`copy_node`, `cluster_nodes`), with small spares on. Nothing else changes: what a launch
//! runs is the captured graph, updated or instantiated.
//!
//! An update pairs only captures whose kernels on thread-block clusters sit at the same nodes
//! with the same cluster shapes ([`cluster_key`], #1132). CUDA accepts an update that puts
//! another kernel into a node of the same topology, but a node instantiated without a cluster
//! keeps running without one: with Gemma 4's MTP on a DGX Spark (GB10) a verify step of 16 rows
//! or fewer (the cluster argmax) and one of more rows (ggml's argmax, one CTA a row) are
//! captures of one node count, and the cluster argmax run as plain CTAs picked garbage tokens
//! (an illegal memory access a step later).

use std::collections::VecDeque;
use std::sync::OnceLock;

#[cfg(test)]
mod tests;

/// Captures of fewer CUDA graph nodes than this are small ones: kept in a list of their own
/// ([`SMALL_SPARE_EXECS`]) and never taken over from a kept capture ([`pick_idle`]).
pub(crate) const MIN_NODES: usize = 256;

/// Small spares (captures of fewer than [`MIN_NODES`] nodes) each device keeps: about three
/// verify shapes' pieces over tensor parallelism (~100 a shape).
pub(crate) const SMALL_SPARE_EXECS: usize = 320;

/// Whether small captures are kept as spares: yes unless `LEGION_DEBUG_SMALL_SPARES=0` (read
/// once).
pub(crate) fn small_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| switch_from(std::env::var("LEGION_DEBUG_SMALL_SPARES").ok().as_deref()))
}

/// [`small_enabled`] for the switch's value `v`.
pub(crate) fn switch_from(v: Option<&str>) -> bool {
    !v.is_some_and(|v| v.trim() == "0")
}

/// The small-spare cap with spares on (`cap` above 0) and small ones kept (`small_on`):
/// [`SMALL_SPARE_EXECS`], else 0.
pub(crate) fn small_cap(cap: usize, small_on: bool) -> usize {
    if cap > 0 && small_on {
        SMALL_SPARE_EXECS
    } else {
        0
    }
}

/// Of captures unlaunched for `idle` (ns each), the index of the one unlaunched longest, if
/// that is at least `min` (ties: the first).
pub(crate) fn pick_idle(idle: impl Iterator<Item = u64>, min: u64) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;
    for (i, t) in idle.enumerate() {
        if t >= min && best.is_none_or(|(_, b)| t > b) {
            best = Some((i, t));
        }
    }
    best.map(|(i, _)| i)
}

/// Where a capture's kernels on thread-block clusters are: a hash of each such node's index
/// (in the capture's node order) and cluster shape, from `(index, [x, y, z])` of its kernel
/// nodes; 0 when none has a cluster of more than one CTA. Captures of one node count update
/// each other only when this matches too.
pub(crate) fn cluster_key(nodes: impl IntoIterator<Item = (usize, [u32; 3])>) -> u64 {
    // FNV-1a over the clustered nodes' words.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut any = false;
    for (i, dims) in nodes {
        if dims.iter().map(|&d| u64::from(d.max(1))).product::<u64>() <= 1 {
            continue;
        }
        any = true;
        for w in [
            i as u64,
            u64::from(dims[0]),
            u64::from(dims[1]),
            u64::from(dims[2]),
        ] {
            for b in w.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    match (any, h) {
        (false, _) => 0,
        (true, 0) => 1,
        (true, h) => h,
    }
}

/// (device, node count, [`cluster_key`], spare), newest last.
type List<T> = VecDeque<(i32, usize, u64, T)>;

/// The spares of each device: big captures ([`MIN_NODES`] nodes or more) and small ones, each
/// list newest last.
pub(crate) struct Spares<T> {
    list: List<T>,
    small: List<T>,
}

impl<T> Default for Spares<T> {
    fn default() -> Self {
        Self {
            list: VecDeque::new(),
            small: VecDeque::new(),
        }
    }
}

impl<T> Spares<T> {
    /// Keeps `x` (a capture of `nodes` CUDA graph nodes on device `dev`, its clustered kernels
    /// at `clusters`, [`cluster_key`]) as `dev`'s newest spare of its size: a big one if `cap`
    /// is not 0, a small one (under [`MIN_NODES`], at least one node) if `small_cap` is not 0;
    /// then drops `dev`'s oldest spares of that size past its cap. What was not kept is
    /// returned, for the caller to destroy (outside its lock).
    pub(crate) fn put(
        &mut self,
        dev: i32,
        nodes: usize,
        clusters: u64,
        x: T,
        cap: usize,
        small_cap: usize,
    ) -> Vec<T> {
        let (list, cap) = if nodes >= MIN_NODES {
            (&mut self.list, cap)
        } else {
            (&mut self.small, if nodes == 0 { 0 } else { small_cap })
        };
        if cap == 0 {
            return vec![x];
        }
        list.push_back((dev, nodes, clusters, x));
        let mut mine = list.iter().filter(|(d, _, _, _)| *d == dev).count();
        let mut out = Vec::new();
        let mut i = 0;
        while mine > cap && i < list.len() {
            if list[i].0 == dev {
                let (_, _, _, x) = list.remove(i).expect("in range");
                out.push(x);
                mine -= 1;
            } else {
                i += 1;
            }
        }
        out
    }

    /// Takes `dev`'s newest spare of `nodes` CUDA graph nodes and clustered kernels at
    /// `clusters`, if any.
    pub(crate) fn take(&mut self, dev: i32, nodes: usize, clusters: u64) -> Option<T> {
        let list = if nodes >= MIN_NODES {
            &mut self.list
        } else {
            &mut self.small
        };
        let i = list
            .iter()
            .rposition(|(d, n, c, _)| *d == dev && *n == nodes && *c == clusters)?;
        list.remove(i).map(|(_, _, _, x)| x)
    }

    /// Spares kept on `dev`, big and small.
    #[cfg(test)]
    pub(crate) fn len(&self, dev: i32) -> usize {
        (self.list.iter().chain(&self.small))
            .filter(|(d, _, _, _)| *d == dev)
            .count()
    }

    /// Small spares kept on all devices and their CUDA graph nodes (for the memory they hold).
    pub(crate) fn small_held(&self) -> (usize, usize) {
        (
            self.small.len(),
            self.small.iter().map(|(_, n, _, _)| n).sum(),
        )
    }
}
