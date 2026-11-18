//! A share layer's shared expert while the CPU pool works (#37).
//!
//! In the step graph as built, a MoE layer's shared expert runs after the routed experts' down
//! op, whose `collect` waits for the pool's down rows. It reads only the layer's input, so on
//! a GPU it can run while the pool computes: between `publish` and the SwiGLU op (whose
//! `activate` waits for the pool's gate and up sums) or between the SwiGLU op and the down op.
//!
//! [`place_shexp`] moves it there after the graph is allocated (`crate::ggml::post_alloc`, after
//! the node fusion pass), so the graph is planned as built: every other tensor keeps its memory
//! and ggml-cuda's layout-dependent fusions (#410, #472) choose as without the move. Moved before
//! planning, the shared expert's tensors took other memory and the text changed. The moved
//! nodes' outputs cannot stay where the plan put them (the allocator reuses that memory for the
//! nodes they now run ahead of), so they get memory of their own: a device region outside the
//! graph's buffer, a few KiB a layer, which no other node uses. Views of them follow. Their
//! readers, the gate's `sigmoid`, `mul` and the sum, stay where they were, so the layer's tail
//! and its fusions are as built. A node writing in place over a moved output still writes the
//! memory the plan gave it, which nothing else uses then either.
//!
//! Nodes move only where none of the nodes they then run ahead of writes their sources' memory
//! and they read none of those nodes' outputs; otherwise the layer stays as built. Nodes are
//! only reordered, never added or removed, and each computes what it computed before.

use super::ops::{PUBLISH, SHARE_DOWN, SHARE_GLU, SHARE_GLU_PREQ};
use crate::ggml::{ExternalOp, ffi, op_of};

/// Where a share layer's shared expert runs in the step graph ([`shexp_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShexpAt {
    /// After the routed experts' output (the model's own order).
    After,
    /// Between the SwiGLU op (whose `activate` hands the pool its down rows) and the down op
    /// (whose `collect` waits for them): while the pool runs the down rows.
    BeforeDown,
    /// Between `publish` and the SwiGLU op: while the pool runs the gate and up rows.
    BeforeGlu,
    /// Split over both waits: its gate and up (with their SwiGLU) between `publish` and the
    /// SwiGLU op, the rest (its down and its gate's matvec) between the SwiGLU op and the down
    /// op, so `activate` hands the pool its down rows sooner.
    Split,
}

/// [`ShexpAt`] of the share's layers: [`ShexpAt::BeforeGlu`]; `LEGION_DEBUG_SHARE_SHEXP=0`
/// (as built), `1` (before the down op), `2` (before the SwiGLU op) or `3` (split), for A/B
/// runs.
pub fn shexp_at() -> ShexpAt {
    static AT: std::sync::OnceLock<ShexpAt> = std::sync::OnceLock::new();
    *AT.get_or_init(
        || match std::env::var("LEGION_DEBUG_SHARE_SHEXP").as_deref() {
            Ok("0") => ShexpAt::After,
            Ok("1") => ShexpAt::BeforeDown,
            Ok("2") => ShexpAt::BeforeGlu,
            Ok("3") => ShexpAt::Split,
            _ => ShexpAt::BeforeGlu,
        },
    )
}

/// Nodes after a share layer's down op [`plan`] looks at for its shared expert.
const SCAN: usize = 64;
/// The alignment of each moved output in the region ([`place`]).
pub(super) const ALIGN: usize = 256;

/// Whether a node of op `op` computes and may move: matvecs (ggml's or external ops) and the
/// SwiGLU. Its gate's `sigmoid`, scaling and sum stay with the layer's tail.
fn moves(op: ffi::ggml_op) -> bool {
    matches!(
        op,
        ffi::ggml_op_GGML_OP_MUL_MAT | ffi::ggml_op_GGML_OP_CUSTOM | ffi::ggml_op_GGML_OP_GLU
    )
}

fn view_op(op: ffi::ggml_op) -> bool {
    matches!(
        op,
        ffi::ggml_op_GGML_OP_VIEW
            | ffi::ggml_op_GGML_OP_RESHAPE
            | ffi::ggml_op_GGML_OP_PERMUTE
            | ffi::ggml_op_GGML_OP_TRANSPOSE
    )
}

/// The tensor whose memory `t` is (itself unless a view).
///
/// # Safety
/// A valid tensor.
unsafe fn base(t: *mut ffi::ggml_tensor) -> *mut ffi::ggml_tensor {
    // SAFETY: the caller's.
    let v = unsafe { (*t).view_src };
    if v.is_null() { t } else { v }
}

/// The bytes `t`'s memory takes: its buffer type's allocation size (padded rows), else its
/// size.
///
/// # Safety
/// A valid tensor.
unsafe fn bytes(t: *mut ffi::ggml_tensor) -> usize {
    // SAFETY: the caller's.
    unsafe {
        let b = (*t).buffer;
        if b.is_null() {
            ffi::ggml_nbytes(t)
        } else {
            ffi::ggml_backend_buft_get_alloc_size(ffi::ggml_backend_buffer_get_type(b), t)
        }
    }
}

/// The bytes `t` spans (empty without memory).
///
/// # Safety
/// A valid tensor.
unsafe fn span(t: *mut ffi::ggml_tensor) -> (usize, usize) {
    // SAFETY: the caller's.
    unsafe {
        let p = (*t).data as usize;
        if p == 0 {
            return (0, 0);
        }
        (p, p + bytes(t))
    }
}

/// Whether `a` and `b` share memory.
///
/// # Safety
/// Valid tensors.
unsafe fn overlap(a: *mut ffi::ggml_tensor, b: *mut ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    let ((a0, a1), (b0, b1)) = unsafe { (span(a), span(b)) };
    a0 < a1 && b0 < b1 && a0 < b1 && b0 < a1
}

/// One layer's move: `nodes[at..=last]` becomes the nodes `order` names (the `moving` nodes
/// and the ones they jump, each part in graph order).
struct Move {
    at: usize,
    last: usize,
    moving: Vec<usize>,
    order: Vec<usize>,
}

/// What [`place`] did: layers moved, layers left as built, the region's bytes, and the moved
/// nodes that compute (not views) before the SwiGLU op and after it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    pub moved: usize,
    pub left: usize,
    pub bytes: usize,
    pub before_glu: usize,
    pub after_glu: usize,
}

/// The share layer's move whose down op is `nodes[d]`, `None` when it has no shared expert to
/// move, `Some(None)` when it has one that cannot move.
///
/// # Safety
/// Valid tensors.
unsafe fn plan(nodes: &[*mut ffi::ggml_tensor], d: usize, mode: ShexpAt) -> Option<Option<Move>> {
    let is = |t: *mut ffi::ggml_tensor, op: &'static ExternalOp| {
        // SAFETY: the caller's.
        unsafe { op_of(t) }.is_some_and(|o| std::ptr::eq(o, op))
    };
    let share_op =
        |t| is(t, &PUBLISH) || is(t, &SHARE_DOWN) || is(t, &SHARE_GLU) || is(t, &SHARE_GLU_PREQ);
    // SAFETY: the caller's.
    unsafe {
        // The down op reads the SwiGLU op's output (source 1), which reads the published ids.
        let glu = (*nodes[d]).src[1];
        let a = nodes[..d].iter().rposition(|&t| t == glu)?;
        let p = nodes[..a].iter().rposition(|&t| is(t, &PUBLISH))?;
        // The layer's input: what `publish` copies for the pool (`x`, source 1).
        let x = (*nodes[p]).src[1];
        if x.is_null() {
            return None;
        }
        let input = base(x);
        let at = if mode == ShexpAt::BeforeDown {
            a + 1
        } else {
            a
        };
        // Nodes the moved ones would run ahead of and must not read: the routed experts' work
        // and what reads it, or a node not moved.
        let mut stay: std::collections::HashSet<*mut ffi::ggml_tensor> =
            nodes[at..=d].iter().copied().collect();
        let mut moved: std::collections::HashSet<*mut ffi::ggml_tensor> = Default::default();
        let mut moving = Vec::new();
        let mut last = d;
        for (j, &t) in nodes
            .iter()
            .enumerate()
            .take((d + 1 + SCAN).min(nodes.len()))
            .skip(d + 1)
        {
            if share_op(t) {
                break;
            }
            let srcs = (*t).src.iter().copied().filter(|s| !s.is_null());
            let vs = (*t).view_src;
            let reads_stay = srcs
                .clone()
                .any(|s| stay.contains(&s) || stay.contains(&base(s)))
                || (!vs.is_null() && stay.contains(&vs));
            // The shared expert: reads the layer's input or another moved node.
            let ours = srcs
                .clone()
                .any(|s| base(s) == input || moved.contains(&s) || moved.contains(&base(s)));
            let op = (*t).op;
            if !reads_stay && ours && (moves(op) || view_op(op)) {
                moving.push(j);
                moved.insert(t);
                last = j;
            } else {
                stay.insert(t);
            }
        }
        if !moving.iter().any(|&j| !view_op((*nodes[j]).op)) {
            return None;
        }
        let jumped: Vec<usize> = (at..=last).filter(|j| !moving.contains(j)).collect();
        // The moved nodes' outputs get memory of their own; what they read must not be written
        // by a node they now run ahead of (the memory of their sources, and of the views they
        // are, that is not moved).
        let mut read = Vec::new();
        for &m in &moving {
            let u = nodes[m];
            let srcs = (*u).src.iter().copied().filter(|s| !s.is_null());
            for s in srcs.chain(std::iter::once(u).filter(|&u| view_op((*u).op))) {
                let b = base(s);
                if !moved.contains(&b) && !read.contains(&b) {
                    read.push(b);
                }
            }
        }
        let safe = read.iter().all(|&s| {
            jumped.iter().all(|&k| {
                let t = nodes[k];
                view_op((*t).op) || (*t).op == ffi::ggml_op_GGML_OP_NONE || !overlap(t, s)
            })
        });
        // Split: the moving nodes another moving node reads (the gate and up with their SwiGLU,
        // and views) run before the SwiGLU op `nodes[a]` (`jumped[0]`), the others (the down
        // and the gate's matvec, read only by the tail) right after it. Those jump fewer nodes
        // than the check above covers, and none they did not jump as built.
        let order = if mode == ShexpAt::Split {
            let mut fed = std::collections::HashSet::new();
            for &m in &moving {
                for s in (*nodes[m]).src.iter().copied().filter(|s| !s.is_null()) {
                    fed.insert(s);
                    fed.insert(base(s));
                }
            }
            let (first, rest): (Vec<usize>, Vec<usize>) =
                moving.iter().partition(|&&j| fed.contains(&nodes[j]));
            first
                .iter()
                .chain(&jumped[..1])
                .chain(&rest)
                .chain(&jumped[1..])
                .copied()
                .collect()
        } else {
            moving.iter().chain(&jumped).copied().collect()
        };
        Some(safe.then_some(Move {
            at,
            last,
            moving,
            order,
        }))
    }
}

/// Moves each share layer's shared expert in an allocated graph to `mode` (see the module doc),
/// its outputs into a region of [`Placed::bytes`] bytes that `region` gives (called once, with
/// the bytes, when a layer moves; `None`: nothing moves).
///
/// # Safety
/// A live, allocated graph none of whose nodes has run; the region `region` gives is memory of
/// the graph's device that no other tensor uses while the graph runs.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) unsafe fn place(
    graph: *mut ffi::ggml_cgraph,
    mode: ShexpAt,
    region: &mut dyn FnMut(*mut ffi::ggml_tensor, usize) -> Option<usize>,
) -> Placed {
    let mut placed = Placed::default();
    if mode == ShexpAt::After {
        return placed;
    }
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors, none run.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        if n == 0 {
            return placed;
        }
        let nodes = std::slice::from_raw_parts_mut(ffi::ggml_graph_nodes(graph), n);
        let mut moves = Vec::new();
        let mut d = 0;
        while d < n {
            let down = op_of(nodes[d]).is_some_and(|o| std::ptr::eq(o, &SHARE_DOWN));
            match down.then(|| plan(nodes, d, mode)).flatten() {
                Some(Some(m)) => {
                    d = m.last + 1;
                    moves.push(m);
                }
                Some(None) => {
                    placed.left += 1;
                    d += 1;
                }
                None => d += 1,
            }
        }
        // The moved outputs' places in the region, in graph order.
        let mut outs = Vec::new();
        for m in &moves {
            for &j in &m.moving {
                let t = nodes[j];
                if !view_op((*t).op) {
                    outs.push((t, placed.bytes));
                    placed.bytes += bytes(t).div_ceil(ALIGN) * ALIGN;
                }
            }
        }
        let Some(at) = outs.first().and_then(|&(t, _)| region(t, placed.bytes)) else {
            placed.left += moves.len();
            placed.bytes = 0;
            return placed;
        };
        let mut new = std::collections::HashMap::new();
        for &(t, off) in &outs {
            (*t).data = (at + off) as *mut std::ffi::c_void;
            new.insert(t, (*t).data);
        }
        // Views of the moved outputs (nodes, their sources, the graph's leafs) follow them.
        let leafs = crate::ggml::graph_leafs(graph);
        for &t in nodes.iter().chain(leafs) {
            for v in std::iter::once(t).chain((*t).src.iter().copied().filter(|s| !s.is_null())) {
                let vs = (*v).view_src;
                if !vs.is_null() && new.contains_key(&vs) {
                    (*v).data = ((*vs).data as usize + (*v).view_offs) as *mut std::ffi::c_void;
                }
            }
        }
        for m in &moves {
            // The SwiGLU op is `nodes[m.at]` unless the nodes move past it.
            let glu = if mode == ShexpAt::BeforeDown {
                0
            } else {
                m.order.iter().position(|&j| j == m.at).unwrap_or(0)
            };
            for (i, &j) in m.order.iter().enumerate() {
                if m.moving.contains(&j) && !view_op((*nodes[j]).op) {
                    if i < glu {
                        placed.before_glu += 1;
                    } else {
                        placed.after_glu += 1;
                    }
                }
            }
            let seg: Vec<_> = m.order.iter().map(|&j| nodes[j]).collect();
            nodes[m.at..=m.last].copy_from_slice(&seg);
        }
        placed.moved = moves.len();
        placed
    }
}

/// Graphs whose placement was logged (the first few).
#[cfg(feature = "cuda")]
static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`place`] on a CUDA graph ([`crate::model::core::node_fuse::rewrite`]): the region is a
/// device buffer of the graph's device ([`cuda::region`]).
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
#[cfg(feature = "cuda")]
pub(crate) unsafe fn place_shexp(graph: *mut ffi::ggml_cgraph) {
    let mode = shexp_at();
    if mode == ShexpAt::After {
        return;
    }
    // SAFETY: the caller's; the region is the device's, used by no graph's tensors but the
    // moved ones, and a device's graphs run one after another in its stream.
    let p = unsafe { place(graph, mode, &mut |t, n| cuda::region(t, n)) };
    if p.moved + p.left > 0 && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        tracing::info!(
            ?mode,
            moved = p.moved,
            left = p.left,
            bytes = p.bytes,
            before_glu = p.before_glu,
            after_glu = p.after_glu,
            "cpu share: the shared expert runs while the pool works (layers moved, left as built)"
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::{c_int, c_void};
    use std::sync::Mutex;

    use crate::ggml::ffi;

    unsafe extern "C" {
        fn cudaPointerGetAttributes(attr: *mut PointerAttributes, ptr: *const c_void) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, size: usize) -> c_int;
    }

    /// `cudaPointerAttributes` (driver_types.h), with room to spare past its end.
    #[repr(C)]
    struct PointerAttributes {
        kind: c_int,
        device: c_int,
        device_pointer: *mut c_void,
        host_pointer: *mut c_void,
        spare: [u64; 8],
    }

    /// `cudaMemoryTypeDevice`.
    const MEMORY_DEVICE: c_int = 2;

    /// Per device: its region's address and bytes. A region that grows is allocated again and
    /// the old one kept (graphs allocated before still use it).
    static REGIONS: Mutex<Vec<(c_int, usize, usize)>> = Mutex::new(Vec::new());

    /// A region of at least `bytes` on the device whose memory `t` (an allocated tensor) is.
    /// Every graph of the device gets the same one: they run one after another in its stream.
    /// Sized to the next power of two of the ask, at least 1 MiB.
    pub(super) fn region(t: *mut ffi::ggml_tensor, bytes: usize) -> Option<usize> {
        // SAFETY: zeroed plain data, written by CUDA.
        let mut a: PointerAttributes = unsafe { std::mem::zeroed() };
        // SAFETY: an attribute query of an allocated tensor's address; an error is cleared.
        if unsafe { cudaPointerGetAttributes(&mut a, (*t).data) } != 0 {
            // SAFETY: clears the query's error.
            unsafe { cudaGetLastError() };
            return None;
        }
        if a.kind != MEMORY_DEVICE {
            return None;
        }
        let mut regions = REGIONS.lock().unwrap();
        if let Some(r) = regions.iter().find(|r| r.0 == a.device && r.2 >= bytes) {
            return Some(r.1);
        }
        let size = bytes.next_power_of_two().max(1 << 20);
        let mut p = std::ptr::null_mut();
        let mut was = 0;
        // SAFETY: plain runtime calls; the device is restored.
        let ok = unsafe {
            cudaGetDevice(&mut was) == 0
                && cudaSetDevice(a.device) == 0
                && cudaMalloc(&mut p, size) == 0
                && cudaSetDevice(was) == 0
        };
        if !ok || p.is_null() {
            // SAFETY: clears the failed call's error.
            unsafe { cudaGetLastError() };
            return None;
        }
        regions.retain(|r| r.0 != a.device);
        regions.push((a.device, p as usize, size));
        // The old region is never freed: graphs allocated before still use it.
        Some(p as usize)
    }
}
