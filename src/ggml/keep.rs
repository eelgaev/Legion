//! Allocation dependencies (#472): tensors a graph keeps allocated until a later node.
//!
//! ggml's graph allocator frees an intermediate after its last reader and gives the memory to
//! the next node that fits. ggml-cuda fuses some op chains (a MoE layer's weighted expert sum,
//! the top-k router) only when the chain's output does not overlap the chain's inputs
//! (`ggml_cuda_check_fusion_memory_ranges`), and the fused kernels round differently from the
//! ops they replace. Whether an input's freed memory went to the output follows the rest of
//! the graph, so a graph change elsewhere could flip a fusion and move the bits.
//!
//! ggml's scheduler avoids this with dependency nodes: ggml-cuda's `graph_optimize` asks it to
//! keep such a chain's inputs allocated until the chain's output (`add_alloc_dep`), and the
//! scheduler adds a `GGML_OP_NONE` node reading them after that output (`ggml-backend.cpp`,
//! MIT). The allocator counts the node's sources as readers, so the inputs live until it; the
//! backends compute nothing for it. Legion allocates its graphs itself, so it adds the same
//! nodes where it builds those chains ([`Graph::keep_until`]).

use super::context::{Graph, Tensor};
use super::ffi;

impl<'a> Graph<'a> {
    /// Keeps `keep` allocated until `until` has been computed: adds `until` (and what it depends
    /// on) to the graph now, then one dependency node per 10 tensors right after it, reading
    /// them. Leaves (weights, caches, inputs: `GGML_OP_NONE` without a view source) are
    /// skipped; their memory is never the allocator's to reuse. Returns how many dependency
    /// nodes it added.
    ///
    /// The dependency node changes no tensor's value and no other node's readers (the graph's
    /// use counts, which ggml-cuda's fusion matching reads, stay as they were); it only moves
    /// where the allocator frees `keep`.
    pub fn keep_until(&mut self, keep: &[Tensor<'a>], until: Tensor<'a>) -> usize {
        self.expand(until);
        let mut kept: Vec<*mut ffi::ggml_tensor> = Vec::with_capacity(keep.len());
        for t in keep {
            let raw = t.raw();
            // SAFETY: a live tensor of this graph's tensors.
            let leaf =
                unsafe { (*raw).op == ffi::ggml_op_GGML_OP_NONE && (*raw).view_src.is_null() };
            if leaf || raw == until.raw() || kept.contains(&raw) {
                continue;
            }
            kept.push(raw);
        }
        let mut added = 0;
        for chunk in kept.chunks(ffi::GGML_MAX_SRC as usize) {
            // SAFETY: the graph's context makes the node (as ggml's scheduler does: a
            // full-tensor view, op NONE); its sources are live tensors; the graph has room for a
            // node more or ggml aborts, as it does for any node added past its size.
            unsafe {
                let dep = ffi::ggml_view_tensor(self.ctx_raw(), chunk[0]);
                (*dep).src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
                for (s, &t) in chunk.iter().enumerate() {
                    (*dep).src[s] = t;
                }
                ffi::ggml_graph_add_node(self.raw(), dep);
            }
            added += 1;
        }
        added
    }

    /// Adds `inputs`, tensors no op of the graph reads (inputs a caller fills all the same), as
    /// sources of dependency nodes at the end of the graph rather than as bare leaves.
    ///
    /// ggml's graph allocator reuses its last plan for a graph whose nodes and their sources
    /// still fit it, without looking at other leaves' sizes: a bare leaf that grew (an MTP
    /// take-in piece's attention mask, which widens with the context while the piece's nodes
    /// stay as they were) kept its old offset and ran past the buffer, and
    /// `ggml_backend_tensor_alloc` aborted (#527). Read by a node, its size is checked. The
    /// inputs are leaves either way, allocated before every node; the dependency nodes compute
    /// nothing and take no memory, and come last, so the plan lays out every tensor as before.
    /// Inputs a node reads already are only expanded (a dependency node would keep them
    /// allocated longer). Returns how many dependency nodes it added.
    pub fn expand_unread(&mut self, inputs: &[Tensor<'a>]) -> usize {
        // SAFETY: the graph's nodes are live tensors; their sources live or null.
        let read: Vec<*mut ffi::ggml_tensor> = unsafe {
            (0..self.nodes())
                .flat_map(|i| (*ffi::ggml_graph_node(self.raw(), i as i32)).src)
                .filter(|s| !s.is_null())
                .collect()
        };
        let mut unread: Vec<*mut ffi::ggml_tensor> = Vec::with_capacity(inputs.len());
        for &t in inputs {
            self.expand(t);
            if !read.contains(&t.raw()) && !unread.contains(&t.raw()) {
                unread.push(t.raw());
            }
        }
        let mut added = 0;
        for chunk in unread.chunks(ffi::GGML_MAX_SRC as usize) {
            // SAFETY: as in `keep_until`: a full-tensor view of op NONE made in the graph's
            // context, reading live tensors; ggml aborts if the graph has no room for it.
            unsafe {
                let dep = ffi::ggml_view_tensor(self.ctx_raw(), chunk[0]);
                (*dep).src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
                for (s, &t) in chunk.iter().enumerate() {
                    (*dep).src[s] = t;
                }
                ffi::ggml_graph_add_node(self.raw(), dep);
            }
            added += 1;
        }
        added
    }
}

impl<'a> Tensor<'a> {
    /// The node's sources (`src[..]`, the non-null ones in order).
    pub fn srcs(self) -> Vec<Tensor<'a>> {
        // SAFETY: a live tensor; its sources are live tensors of the same lifetime.
        unsafe {
            (*self.raw())
                .src
                .iter()
                .filter_map(|&s| std::ptr::NonNull::new(s))
                .map(|s| self.with_raw(s))
                .collect()
        }
    }

    /// Whether the node is a ggml `mul`.
    pub fn is_mul(self) -> bool {
        // SAFETY: a live tensor.
        unsafe { (*self.raw()).op == ffi::ggml_op_GGML_OP_MUL }
    }
}

#[cfg(test)]
mod tests;
