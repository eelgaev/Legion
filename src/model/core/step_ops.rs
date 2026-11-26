//! Tests only: a one-sequence step's graph walked for its external ops (#1221), to check
//! which of them a stage graph's capture holds.

use crate::ggml::{Context, ExternalOp, ffi, op_of};

use super::state::SeqState;
use super::{GRAPH_SIZE, Group, Logits, Model, ModelError, SeqInput, StepOpts, batch_shape};

impl Model {
    /// Builds the graph of `tokens` at `state`'s next positions (as [`Model::forward`] would
    /// run them, last logits only) and calls `f` with each external op node, in graph order.
    pub(crate) fn step_external_ops(
        &self,
        state: &SeqState,
        tokens: &[u32],
        mut f: impl FnMut(&ExternalOp, &ffi::ggml_tensor),
    ) -> Result<(), ModelError> {
        let groups = [Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past: state.n_past(),
                blocks: &state.blocks,
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
            verify: false,
        }];
        let table = self
            .paged_for(&state.kv, &groups)
            .then_some(state.blocks.len());
        let shape = batch_shape(&groups, table, Some(&*self.hp.arch));
        let ctx = Context::new(GRAPH_SIZE, GRAPH_SIZE);
        let (graph, _, _) = self.build(
            &ctx,
            &state.kv,
            &state.rec,
            StepOpts::default(),
            &shape,
            false,
        )?;
        for i in 0..graph.nodes() {
            // SAFETY: a node of the graph just built; `op_of` reads its params.
            unsafe {
                let node = ffi::ggml_graph_node(graph.raw(), i as i32);
                if let Some(op) = op_of(node) {
                    f(op, &*node);
                }
            }
        }
        Ok(())
    }
}
