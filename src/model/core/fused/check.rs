//! The gated norm op's BF16 rows ([`super::gated_norm_bf16`]) against ggml's chain on one GPU
//! (#689): for each head layout and step size, one graph holds ggml's
//! `cast(mul(mul(rms_norm(o), w), gate(z)), BF16)` (the nodes a gated DeltaNet layer builds,
//! then the cast the FP8 sliced-K GEMM makes of its input) and the op's result. Used by the
//! hidden `legion gated-norm-check` subcommand (a build tree's binary on the GPU box).

use super::gated_norm_bf16;
use crate::ggml::{Backend, ComputeAllocator, Context, types};

/// One layout at one step size: BF16 values compared and how many differ in their bits.
#[derive(Clone, Debug)]
pub struct Check {
    pub head_v: usize,
    pub n_v: usize,
    pub cols: usize,
    /// SiLU gate (else sigmoid).
    pub silu: bool,
    pub outputs: usize,
    pub differing: usize,
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Every `(head_v, n_v)` of `heads` at every step size of `cols` with both gates, on `gpu`.
pub fn check(
    gpu: &mut Backend,
    heads: &[(usize, usize)],
    cols: &[usize],
    eps: f32,
) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for (hi, &(head_v, n_v)) in heads.iter().enumerate() {
        for &n in cols {
            for silu in [true, false] {
                let c = head_v * n_v;
                let ctx = Context::new(32, 64);
                let (ci, ni, hv) = (c as i64, n as i64, head_v as i64);
                let o = ctx.new_tensor(types::F32, &[ci, ni]).as_input();
                let z = ctx.new_tensor(types::F32, &[ci, ni]).as_input();
                let w = ctx.new_tensor(types::F32, &[hv]).as_input();
                let shape = [hv, n_v as i64, ni, 1];
                let normed = ctx.mul(ctx.rms_norm(ctx.reshape_4d(o, shape), eps), w);
                let z4 = ctx.reshape_4d(z, shape);
                let g = if silu { ctx.silu(z4) } else { ctx.sigmoid(z4) };
                let chain = ctx.reshape_2d(ctx.mul(normed, g), ci, ni);
                let reference = ctx.cast(chain, types::BF16).as_output();
                let op = gated_norm_bf16(&ctx, o, w, z, hv, eps, silu).as_output();
                let mut graph = ctx.new_graph(64);
                graph.expand(reference);
                graph.expand(op);
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                let seed = (hi * 1000 + n * 2 + usize::from(silu)) as u64;
                // A delta rule's outputs (small) with a few large values, one all-zero head
                // (the norm's eps alone), the gate's input over its whole range.
                let mut os: Vec<f32> = noise(c * n, 10 + seed).iter().map(|v| 0.05 * v).collect();
                os[3] = 4.0;
                os[c / 2 + 1] = -7.5;
                os[head_v..2 * head_v].fill(0.0);
                let zs: Vec<f32> = noise(c * n, 20 + seed).iter().map(|v| 12.0 * v).collect();
                let ws: Vec<f32> = noise(head_v, 30 + seed)
                    .iter()
                    .map(|v| 1.0 + 0.5 * v)
                    .collect();
                o.set_f32(&os);
                z.set_f32(&zs);
                w.set_f32(&ws);
                gpu.compute(&graph).map_err(|e| e.to_string())?;
                let (a, b) = (op.bytes(), reference.bytes());
                if a.len() != b.len() || a.len() != c * n * 2 {
                    return Err(format!("sizes {} and {} for {c} x {n}", a.len(), b.len()));
                }
                let differing = a
                    .chunks_exact(2)
                    .zip(b.chunks_exact(2))
                    .filter(|(p, q)| p != q)
                    .count();
                out.push(Check {
                    head_v,
                    n_v,
                    cols: n,
                    silu,
                    outputs: c * n,
                    differing,
                });
            }
        }
    }
    Ok(out)
}
