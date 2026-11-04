//! "Model bytes" of a graph (#391): the weight bytes a run of it must read, counted from the
//! graph's nodes when it is built, so that a step's achieved bandwidth (bytes over its time) can
//! be put next to the device's roofline. Required traffic, not measured traffic: activations,
//! caches and anything an L2 serves twice are left out.
//!
//! Per graph, each weight a node reads counts once, by what the node reads of it:
//! - a whole weight, or the rows of a view of one (a draft head over part of the vocabulary):
//!   its bytes;
//! - `get_rows` of a weight (an embedding table): the rows looked up;
//! - an expert stack (`mul_mat_id`, or an external op taking a 3-D weight with routing ids):
//!   recorded as an [`ExpertRead`], whose bytes depend on the routing of the step: at most
//!   `min(pairs, n_expert)` experts ([`GraphBytes::upper`]).
//!
//! Which tensors are weights is the caller's (a predicate over a leaf tensor), so caches,
//! states and inputs, which live in buffers of the same kind, are never counted.
//!
//! What a graph cannot know is counted by the executor per step ([`Phase`]): the experts an
//! expert cache routed to (from its route log, [`ExpertTraffic`]), the KV attention reads
//! (from the sequences' lengths), and each pipeline stage's part ([`StageTotals`]).

use std::collections::HashSet;

use super::ffi;

/// An expert stack read by a step: `pairs` (token, slot) choices among `n_expert` experts of
/// `bytes_per_expert` each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRead {
    pub bytes_per_expert: u64,
    pub n_expert: u64,
    pub pairs: u64,
}

impl ExpertRead {
    /// Bytes if every choice is a different expert (at most every expert once).
    pub fn upper(&self) -> u64 {
        self.bytes_per_expert * self.pairs.min(self.n_expert)
    }
}

/// The model bytes of one graph ([`count`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphBytes {
    /// Dense weights, embedding rows and views, each counted once.
    pub dense: u64,
    /// Expert stacks, each counted once.
    pub experts: Vec<ExpertRead>,
}

impl GraphBytes {
    /// Dense bytes plus the experts' [`ExpertRead::upper`].
    pub fn upper(&self) -> u64 {
        self.dense + self.experts.iter().map(ExpertRead::upper).sum::<u64>()
    }

    /// Expert bytes at most ([`ExpertRead::upper`] summed).
    pub fn expert_upper(&self) -> u64 {
        self.experts.iter().map(ExpertRead::upper).sum()
    }
}

/// Bytes of routed experts read through an expert cache (tiered experts, see
/// [`crate::residency::experts`]), counted by the cache from the route log after each pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertTraffic {
    /// Experts read from VRAM: cache slots, or the staging buffers of a staged prompt pass.
    pub device: u64,
    /// Experts read in place from host memory, over the link.
    pub host: u64,
    /// Experts copied from host memory into VRAM staging buffers (prefill staging), over the
    /// link: read again from VRAM by the pass (in `device`).
    pub staged: u64,
    /// Passes the route log covers that read at least one expert in place from host memory
    /// (one per layer and pass): how the misses spread over the layers' matvecs (#677).
    pub host_passes: u64,
}

impl ExpertTraffic {
    /// `self` minus an earlier snapshot of the same counters.
    pub fn since(&self, earlier: &ExpertTraffic) -> ExpertTraffic {
        ExpertTraffic {
            device: self.device - earlier.device,
            host: self.host - earlier.host,
            staged: self.staged - earlier.staged,
            host_passes: self.host_passes - earlier.host_passes,
        }
    }

    /// Adds `other` to `self`.
    pub fn add(&mut self, other: &ExpertTraffic) {
        self.device += other.device;
        self.host += other.host;
        self.staged += other.staged;
        self.host_passes += other.host_passes;
    }
}

/// Model bytes and time of the steps of one kind ([`Totals`]), or of one pipeline stage's part
/// of them ([`StageTotals`], whose `seconds` are the stage's busy time).
///
/// What the steps must read: dense weights, routed experts and KV, split by where they are read
/// from (device memory, or host memory over the link), plus the copies into device memory that
/// prefill staging makes. Not measured traffic: activations, writes and L2 hits are left out.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Phase {
    /// Steps (graph runs) counted.
    pub steps: u64,
    /// Dense bytes ([`GraphBytes::dense`]) summed.
    pub dense: u64,
    /// Expert bytes at most ([`GraphBytes::expert_upper`]) summed: experts in the weights (not
    /// tiered), as if every routed choice were a different expert.
    pub experts_upper: u64,
    /// Tiered experts read from device memory and in place from host memory
    /// ([`ExpertTraffic`]): the distinct experts routed for passes the route log covers, at most
    /// every routed choice a different expert for bigger ones.
    pub experts_device: u64,
    pub experts_host: u64,
    /// Experts copied into device memory by prefill staging ([`ExpertTraffic::staged`]).
    pub staged: u64,
    /// KV cache bytes attention reads from device memory and from the KV host tier.
    pub kv: u64,
    pub kv_host: u64,
    /// Wall time of those steps, seconds (launch to results).
    pub seconds: f64,
}

impl Phase {
    /// Adds one step of `bytes` that took `seconds`.
    pub fn add(&mut self, bytes: &GraphBytes, seconds: f64) {
        self.steps += 1;
        self.add_graph(bytes);
        self.seconds += seconds;
    }

    /// Adds a graph's weight bytes (no step, no time).
    pub fn add_graph(&mut self, bytes: &GraphBytes) {
        self.dense += bytes.dense;
        self.experts_upper += bytes.expert_upper();
    }

    /// Adds tiered-expert traffic (no step, no time).
    pub fn add_experts(&mut self, t: &ExpertTraffic) {
        self.experts_device += t.device;
        self.experts_host += t.host;
        self.staged += t.staged;
    }

    /// Adds every counter of `other`.
    pub fn merge(&mut self, other: &Phase) {
        self.steps += other.steps;
        self.dense += other.dense;
        self.experts_upper += other.experts_upper;
        self.experts_device += other.experts_device;
        self.experts_host += other.experts_host;
        self.staged += other.staged;
        self.kv += other.kv;
        self.kv_host += other.kv_host;
        self.seconds += other.seconds;
    }

    /// Bytes read from device memory (HBM, LPDDR5X): weights, experts and KV there.
    pub fn device(&self) -> u64 {
        self.dense + self.experts_upper + self.experts_device + self.kv
    }

    /// Bytes read in place from host memory over the link (experts and KV host tier).
    pub fn host(&self) -> u64 {
        self.experts_host + self.kv_host
    }

    /// Bytes over the host link: in-place host reads plus staging copies.
    pub fn link(&self) -> u64 {
        self.host() + self.staged
    }

    /// Model bytes: everything the steps read, from device or host memory.
    pub fn total(&self) -> u64 {
        self.device() + self.host()
    }

    /// `bytes` over the time, GB/s; 0 without time.
    fn rate(&self, bytes: u64) -> f64 {
        if self.seconds > 0.0 {
            bytes as f64 / self.seconds / 1e9
        } else {
            0.0
        }
    }

    /// Model bytes ([`Self::total`]) over the time: GB/s, 0 without time.
    pub fn gb_s(&self) -> f64 {
        self.rate(self.total())
    }

    /// Device-memory bytes ([`Self::device`]) over the time, GB/s.
    pub fn device_gb_s(&self) -> f64 {
        self.rate(self.device())
    }

    /// Link bytes ([`Self::link`]) over the time, GB/s.
    pub fn link_gb_s(&self) -> f64 {
        self.rate(self.link())
    }

    /// `self` minus an earlier snapshot of the same counters.
    pub fn since(&self, earlier: &Phase) -> Phase {
        Phase {
            steps: self.steps - earlier.steps,
            dense: self.dense - earlier.dense,
            experts_upper: self.experts_upper - earlier.experts_upper,
            experts_device: self.experts_device - earlier.experts_device,
            experts_host: self.experts_host - earlier.experts_host,
            staged: self.staged - earlier.staged,
            kv: self.kv - earlier.kv,
            kv_host: self.kv_host - earlier.kv_host,
            seconds: self.seconds - earlier.seconds,
        }
    }
}

/// A kind of step ([`Totals`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Prefill,
    Decode,
    Draft,
}

impl Kind {
    /// The kind of a target step: a prompt step when a sequence has more than
    /// [`DECODE_MAX_TOKENS`] tokens in it.
    pub fn of_step(prompt: bool) -> Kind {
        if prompt { Kind::Prefill } else { Kind::Decode }
    }
}

/// Pipeline stages counted apart ([`Totals::stages`]); later stages count with the last.
pub const MAX_STAGES: usize = 8;

/// One pipeline stage's part of the target steps ([`Totals::stages`]): its graphs' weights, its
/// layers' experts and KV, and its busy time (launch to synchronized, per round).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTotals {
    pub prefill: Phase,
    pub decode: Phase,
}

impl StageTotals {
    /// `self` minus an earlier snapshot.
    pub fn since(&self, earlier: &StageTotals) -> StageTotals {
        StageTotals {
            prefill: self.prefill.since(&earlier.prefill),
            decode: self.decode.since(&earlier.decode),
        }
    }
}

/// An engine's model bytes since start, by kind of step: prompt steps (more than
/// [`DECODE_MAX_TOKENS`] tokens of one sequence), decode and verify steps, and draft passes (an
/// MTP head's), and the target steps by pipeline stage.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Totals {
    pub prefill: Phase,
    pub decode: Phase,
    pub draft: Phase,
    /// Per pipeline stage (the first [`Self::n_stages`]): its part of the prefill and decode
    /// steps.
    pub stages: [StageTotals; MAX_STAGES],
    pub n_stages: usize,
}

impl Totals {
    /// `self` minus an earlier snapshot.
    pub fn since(&self, earlier: &Totals) -> Totals {
        let mut stages = [StageTotals::default(); MAX_STAGES];
        for (s, (a, b)) in stages
            .iter_mut()
            .zip(self.stages.iter().zip(&earlier.stages))
        {
            *s = a.since(b);
        }
        Totals {
            prefill: self.prefill.since(&earlier.prefill),
            decode: self.decode.since(&earlier.decode),
            draft: self.draft.since(&earlier.draft),
            stages,
            n_stages: self.n_stages,
        }
    }

    /// The steps of kind `k`.
    pub fn phase_mut(&mut self, k: Kind) -> &mut Phase {
        match k {
            Kind::Prefill => &mut self.prefill,
            Kind::Decode => &mut self.decode,
            Kind::Draft => &mut self.draft,
        }
    }

    /// Stage `s`'s part of the target steps of kind `k` (`None` for draft passes).
    pub fn stage_mut(&mut self, s: usize, k: Kind) -> Option<&mut Phase> {
        let st = &mut self.stages[s.min(MAX_STAGES - 1)];
        match k {
            Kind::Prefill => Some(&mut st.prefill),
            Kind::Decode => Some(&mut st.decode),
            Kind::Draft => None,
        }
    }

    /// Decode and verify steps plus draft passes, as one phase (a request's decoding).
    pub fn decoding(&self) -> Phase {
        let mut p = self.decode;
        p.merge(&self.draft);
        p
    }
}

/// The most tokens of one sequence a step counted as decode has (a verify step's rows).
pub const DECODE_MAX_TOKENS: usize = 16;

/// The tensor a view (or a view of a view) reads from.
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn root(t: *const ffi::ggml_tensor) -> *const ffi::ggml_tensor {
    unsafe {
        let mut t = t;
        while !(*t).view_src.is_null() {
            t = (*t).view_src;
        }
        t
    }
}

/// The byte offset of a view (of views) into its root tensor; 0 for the root itself.
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn view_offs(t: *const ffi::ggml_tensor) -> usize {
    unsafe {
        let mut t = t;
        let mut offs = 0;
        while !(*t).view_src.is_null() {
            offs += (*t).view_offs;
            t = (*t).view_src;
        }
        offs
    }
}

/// Elements of `t`.
///
/// # Safety
/// `t` must be a valid tensor.
unsafe fn elements(t: *const ffi::ggml_tensor) -> u64 {
    unsafe { (*t).ne.iter().map(|&n| n.max(1) as u64).product() }
}

/// The routing ids among a node's sources: an I32 matrix of at most 64 choices per token.
///
/// # Safety
/// `node` must be a valid tensor.
unsafe fn ids_of(node: *const ffi::ggml_tensor) -> Option<*const ffi::ggml_tensor> {
    unsafe {
        (*node)
            .src
            .iter()
            .copied()
            .find(|&s| {
                !s.is_null() && (*s).type_ == ffi::ggml_type_GGML_TYPE_I32 && (*s).ne[0] <= 64
            })
            .map(|s| s as *const _)
    }
}

/// Counts the model bytes of `graph`: `is_weight` says whether a leaf tensor (a view's root) is
/// a weight.
///
/// # Safety
/// `graph` must be a valid graph whose tensors are live.
pub unsafe fn count(
    graph: *mut ffi::ggml_cgraph,
    is_weight: &dyn Fn(*const ffi::ggml_tensor) -> bool,
) -> GraphBytes {
    let mut out = GraphBytes::default();
    // (weight, view offset, bytes) already counted: a weight several nodes read counts once.
    // Keyed by the tensor, not its data address: a meta (`--tp`) buffer gives every weight
    // the same placeholder address.
    let mut seen: HashSet<(usize, usize, u64)> = HashSet::new();
    let mut stacks: HashSet<(usize, usize)> = HashSet::new();
    unsafe {
        for i in 0..ffi::ggml_graph_n_nodes(graph) {
            let node = ffi::ggml_graph_node(graph, i);
            // Views read nothing: what reads through them counts.
            if matches!(
                (*node).op,
                ffi::ggml_op_GGML_OP_VIEW
                    | ffi::ggml_op_GGML_OP_RESHAPE
                    | ffi::ggml_op_GGML_OP_PERMUTE
                    | ffi::ggml_op_GGML_OP_TRANSPOSE
                    | ffi::ggml_op_GGML_OP_NONE
            ) {
                continue;
            }
            for (j, &src) in (*node).src.iter().enumerate() {
                if src.is_null() {
                    continue;
                }
                let r = root(src);
                if !is_weight(r) {
                    continue;
                }
                let stack = (*src).ne[2] > 1 && (*node).op != ffi::ggml_op_GGML_OP_GET_ROWS;
                if stack && let Some(ids) = ids_of(node) {
                    // An expert stack: what is read depends on the routing.
                    if stacks.insert((r as usize, view_offs(src))) {
                        out.experts.push(ExpertRead {
                            bytes_per_expert: (*src).nb[2] as u64,
                            n_expert: (*src).ne[2] as u64,
                            pairs: (*ids).ne[0].max(1) as u64 * (*ids).ne[1].max(1) as u64,
                        });
                    }
                    continue;
                }
                let bytes = if (*node).op == ffi::ggml_op_GGML_OP_GET_ROWS && j == 0 {
                    // The rows looked up, each a row of the table.
                    let rows = (*node).src[1];
                    if rows.is_null() {
                        continue;
                    }
                    elements(rows) * (*src).nb[1] as u64
                } else {
                    ffi::ggml_nbytes(src) as u64
                };
                // Lookups of one table add up; whole reads of one tensor count once.
                let key = if (*node).op == ffi::ggml_op_GGML_OP_GET_ROWS && j == 0 {
                    (node as usize, 0, bytes)
                } else {
                    (r as usize, view_offs(src), bytes)
                };
                if seen.insert(key) {
                    out.dense += bytes;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
