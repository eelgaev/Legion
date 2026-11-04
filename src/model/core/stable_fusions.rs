//! `--stable-fusions` (#472): ggml-cuda's layout-dependent MoE fusions always taken.
//!
//! ggml-cuda fuses a MoE layer's weighted expert sum (a `mul` by the router weights, a view
//! per expert and the adds) into one kernel that rounds with one FMA per expert, and ggml's
//! top-k router chain into one kernel, but only when the chain's output does not overlap its
//! inputs in memory. Where ggml's allocator puts the output follows the whole graph, so an
//! unrelated change (a fused op elsewhere, a smaller graph) flipped some layers' sums between
//! the two roundings and moved greedy text (#410, #459, #495).
//!
//! llama.cpp never sees this: ggml-cuda's `graph_optimize` asks ggml's scheduler to keep those
//! inputs allocated until the chain's output, so the output never takes their memory. Legion
//! allocates its graphs without the scheduler; with the switch on, the graph builders add the
//! same dependencies where they build the chains ([`Graph::keep_until`]). The fusions then
//! fire in every layer whatever the layout, with no extra kernels: the text moves once from
//! the default (the layers that ran unfused before now run fused) and then stays put.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Graph, Tensor};

/// Whether graphs built from now on pin the fusions' inputs.
static ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on keep the MoE fusions' inputs allocated until their
/// outputs (off by default; `legion serve --stable-fusions` turns it on).
pub fn set_stable_fusions(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// With the switch on, keeps a MoE layer's weighted-sum inputs allocated until `sum` (the
/// chain's last node; added to the graph now): the experts' outputs and the weights `weighted`
/// (their `mul`) reads, and for the scaled form (`mul(mul(experts, scale), weights)`) the inner
/// `mul`'s too. Those are what ggml-cuda keeps for the fusion (`graph_optimize`).
pub(crate) fn pin_weighted_sum<'a>(
    graph: &RefCell<Graph<'a>>,
    weighted: Tensor<'a>,
    sum: Tensor<'a>,
) {
    if on() {
        graph
            .borrow_mut()
            .keep_until(&weighted_sum_inputs(weighted), sum);
    }
}

/// What [`pin_weighted_sum`] keeps: `weighted`'s sources, and a source `mul`'s sources.
fn weighted_sum_inputs(weighted: Tensor<'_>) -> Vec<Tensor<'_>> {
    let mut keep = weighted.srcs();
    for s in weighted.srcs() {
        if s.is_mul() {
            keep.extend(s.srcs());
        }
    }
    keep
}

/// With the switch on, keeps a router chain's logits allocated until its weights `w` (added
/// to the graph now), so neither the selected ids nor the weights take their memory: the chain
/// from the softmax (or sigmoid) to the weights is then one fused top-k kernel whatever the
/// layout. ggml-cuda keeps every node of the chain; the logits are the only input not made
/// inside it that the allocator could free early.
pub(crate) fn pin_router<'a>(graph: &RefCell<Graph<'a>>, logits: Tensor<'a>, w: Tensor<'a>) {
    let mut g = graph.borrow_mut();
    if on() {
        g.keep_until(&[logits], w);
    } else {
        g.expand(w);
    }
}

#[cfg(test)]
mod tests;
