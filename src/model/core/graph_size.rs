//! How many tensors and nodes a step's graph context holds, by its group count (#1176).
//!
//! A graph's tensors grow about linearly with its groups (each group's attention, recurrent
//! state and output rows, per layer). [`GRAPH_SIZE`] holds every step at the default four
//! prefill slots; with `--prefill-slots 16`, sixteen prompt groups and a verify group overflow
//! it (`ggml_new_object: not enough space in the context's memory pool`).

use super::GRAPH_SIZE;

#[cfg(test)]
mod tests;

/// Groups a graph of [`GRAPH_SIZE`] tensors is known to hold: the most a step has at the
/// default four prefill slots (four prompt groups and the decode or verify group).
pub(crate) const GRAPH_SIZE_GROUPS: usize = 5;

/// Tensors and nodes for the graph of a step (or MTP pass) of `n_groups` groups: [`GRAPH_SIZE`]
/// up to [`GRAPH_SIZE_GROUPS`] groups, so every step at the default `--prefill-slots` is built
/// in a context of the same size as before, and a whole multiple of it above (a few sizes, so
/// big contexts reuse each other's pooled metadata memory). The size only bounds the
/// context's metadata: the graph and its results are the same in any context it fits in.
pub(crate) fn graph_size(n_groups: usize) -> usize {
    GRAPH_SIZE * n_groups.div_ceil(GRAPH_SIZE_GROUPS).max(1)
}
