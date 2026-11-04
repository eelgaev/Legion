//! What the pins keep. The switch itself is process-wide, so the model-level check (the same
//! logits with it on) is an integration test of its own (`tests/stable_fusions.rs`).

use super::weighted_sum_inputs;
use crate::ggml::{Context, Tensor, types};

fn raws<'a>(ts: &[Tensor<'a>]) -> Vec<*mut crate::ggml::ffi::ggml_tensor> {
    ts.iter().map(|t| t.raw()).collect()
}

#[test]
fn plain_form_keeps_experts_and_weights() {
    let ctx = Context::new(16, 16);
    let experts = ctx.scale(ctx.new_tensor(types::F32, &[8, 4, 2]), 1.0);
    let w = ctx.scale(ctx.new_tensor(types::F32, &[1, 4, 2]), 1.0);
    let weighted = ctx.mul(experts, w);
    assert_eq!(raws(&weighted_sum_inputs(weighted)), raws(&[experts, w]));
}

/// `mul(mul(experts, scale), weights)`: ggml-cuda fuses both muls, so the raw experts and the
/// scale are inputs too.
#[test]
fn scaled_form_keeps_the_inner_mul_inputs() {
    let ctx = Context::new(16, 16);
    let experts = ctx.scale(ctx.new_tensor(types::F32, &[8, 4, 2]), 1.0);
    let scale = ctx.scale(ctx.new_tensor(types::F32, &[1, 4, 2]), 1.0);
    let w = ctx.scale(ctx.new_tensor(types::F32, &[1, 4, 2]), 1.0);
    let scaled = ctx.mul(experts, scale);
    let weighted = ctx.mul(scaled, w);
    assert_eq!(
        raws(&weighted_sum_inputs(weighted)),
        raws(&[scaled, w, experts, scale])
    );
}
