//! The mixer chain's ops against ggml's: on the CPU (the references, within libm's `expf` and
//! summation order) and, ignored (needs a GPU), bit for bit against ggml-cuda's ops.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;

#[test]
fn chain_defaults_on() {
    assert!(on());
}

/// Which op a case builds.
#[derive(Clone, Copy, Debug)]
enum Op {
    LoAct,
    Combine,
    RmsMul,
}

/// `op` as this module's node (`ours`) or ggml's ops on `backend`, `e` columns, 4 streams, `n`
/// tokens (for [`Op::LoAct`] `e · n` values): the result's bytes. The streams of
/// [`Op::RmsMul`] are a strided view (every other token of a wider tensor), as a mixer may get.
fn case(backend: &mut Backend, op: Op, ours: bool, e: i64, n: i64) -> Vec<u8> {
    let hc = 4i64;
    let ctx = Context::new(32, 32);
    let a = ctx.new_tensor(types::F32, &[e, hc, 2 * n]).as_input();
    let b = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let c = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[e * hc]).as_input();
    let y = match op {
        Op::LoAct => {
            // A plain node to act on, as the down matmul's result is.
            let lo = ctx.cont(b);
            if ours {
                ctx.external_inplace(&HC_LO_ACT, lo, &[], &[0.25f32.to_bits() as i32])
            } else {
                ctx.silu(ctx.scale(lo, 0.25))
            }
        }
        Op::Combine => {
            let res = ctx.view_3d(a, [e, hc, n], [e as usize * 4, (e * hc) as usize * 4], 0);
            let inj = ctx.cont(c);
            if ours {
                combine(&ctx, b, res, inj)
            } else {
                let w = ctx.scale(ctx.sigmoid(ctx.scale(inj, 1.0 / hc as f32)), 2.0);
                ctx.hc_post(b, res, w)
            }
        }
        Op::RmsMul => {
            // Every other token: a token stride of two tokens' streams.
            let x = ctx.view_3d(
                a,
                [e, hc, n],
                [e as usize * 4, 2 * (e * hc) as usize * 4],
                0,
            );
            if ours {
                rms_mul(&ctx, x, g, 1e-6)
            } else {
                ctx.mul(ctx.rms_norm(x, 1e-6), ctx.reshape_2d(g, e, hc))
            }
        }
    };
    let out = ctx.cont(y).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Only the inputs `op` reads are in the graph and get memory.
    let (uses_a, uses_b, uses_c, uses_g) = match op {
        Op::LoAct => (false, true, false, false),
        Op::Combine => (true, true, true, false),
        Op::RmsMul => (true, false, false, true),
    };
    if uses_a {
        a.set_f32(&noise((e * hc * 2 * n) as usize, 71));
    }
    if uses_b {
        b.set_f32(
            &noise((e * n) as usize, 72)
                .iter()
                .map(|v| 8.0 * v)
                .collect::<Vec<_>>(),
        );
    }
    if uses_c {
        c.set_f32(
            &noise((hc * n) as usize, 73)
                .iter()
                .map(|v| 4.0 * v)
                .collect::<Vec<_>>(),
        );
    }
    if uses_g {
        g.set_f32(&noise((e * hc) as usize, 74));
    }
    backend.compute(&graph).unwrap();
    out.bytes()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn cpu_ops_are_ggml_ops() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for op in [Op::LoAct, Op::Combine, Op::RmsMul] {
        for (e, n) in [(256, 1), (2560, 3), (96, 9)] {
            let (a, b) = (
                floats(&case(&mut cpu, op, true, e, n)),
                floats(&case(&mut cpu, op, false, e, n)),
            );
            assert_eq!(a.len(), b.len());
            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                    "{op:?} {e}x{n}, value {i}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn fits_checks() {
    let ctx = Context::new(16, 16);
    let x = ctx.new_tensor(types::F32, &[2560, 4, 3]);
    let g = ctx.new_tensor(types::F32, &[2560 * 4]);
    let short = ctx.new_tensor(types::F32, &[2560]);
    // Never off a GPU.
    assert!(!rms_mul_fits(x, g, false));
    assert!(!rms_mul_fits(x, short, true));
    let out = ctx.new_tensor(types::F32, &[2560, 3]);
    let inj = ctx.new_tensor(types::F32, &[4, 3]);
    // A leaf inject (not a node ggml would take over in place) is not taken.
    assert!(!combine_fits(out, x, inj, true));
    assert!(!plain_node(inj));
    assert!(plain_node(ctx.cont(inj)));
}

/// On a GPU each op gives ggml-cuda's bits at qwen4exp's shape (2560 columns, 4 streams) and
/// the step sizes the chain runs (1-8 tokens) and a prompt's.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ops_are_ggml_ops_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for op in [Op::LoAct, Op::Combine, Op::RmsMul] {
        for (e, n) in [
            (2560, 1),
            (2560, 3),
            (2560, 8),
            (2560, 300),
            (320, 2048),
            (96, 5),
        ] {
            let ours = case(&mut gpu, op, true, e, n);
            let ggml = case(&mut gpu, op, false, e, n);
            let diff = ours
                .chunks(4)
                .zip(ggml.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(diff, 0, "{op:?} {e}x{n}: {diff} values moved");
        }
    }
}
