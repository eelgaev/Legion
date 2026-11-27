//! The combine of BF16 streams (#1183, `--hc-bf16`) against ggml's copy, scales, sigmoid and
//! `hc_post`: on the CPU the op made directly (within libm's `expf`), and, ignored (needs a
//! GPU), `node_fuse`'s rewrite of ggml's chain bit for bit.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;
use crate::model::core::node_fuse::rewrite_graph;

/// How a case's combine runs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// ggml's nodes.
    Chain,
    /// `node_fuse`'s pass over the allocated graph.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Rewrite,
    /// [`make_combine16`] on the allocated `hc_post` node.
    Direct,
}

/// ggml's combine of `[e, 4, n]` BF16 streams, the inject a view of a wider node (as the
/// flush's is of the mixer's norm): the result's bytes and how many combines the pass folded.
fn case(backend: &mut Backend, mode: Mode, e: i64, n: i64) -> (Vec<u8>, usize) {
    let hc = 4i64;
    let ctx = Context::new(32, 32);
    let res16 = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let wide = ctx.new_tensor(types::F32, &[hc + 3, n]).as_input();
    let node = ctx.cont(wide);
    let inject = ctx.view_2d(node, [hc, n], node.nb()[1], 4);
    let s1 = 1.0 / hc as f32;
    let w = ctx.scale(ctx.sigmoid(ctx.scale(inject, s1)), 2.0);
    let y = ctx.hc_post(out, ctx.cast(res16, types::F32), w).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let folded = match mode {
        Mode::Chain => 0,
        // SAFETY: an allocated graph that has not run.
        Mode::Rewrite => unsafe { rewrite_graph(graph.raw(), true).combine16 },
        Mode::Direct => {
            // SAFETY: the allocated `hc_post` node and its sources, of the shapes the op takes.
            unsafe { make_combine16(y.raw(), out.raw(), res16.raw(), inject.raw(), s1, 2.0) };
            0
        }
    };
    let bits: Vec<u8> = noise((e * hc * n) as usize, 81)
        .iter()
        .flat_map(|v| ((4.0 * v).to_bits() >> 16).to_le_bytes()[..2].to_vec())
        .collect();
    res16.set_bytes(&bits);
    out.set_f32(
        &noise((e * n) as usize, 82)
            .iter()
            .map(|v| 8.0 * v)
            .collect::<Vec<_>>(),
    );
    wide.set_f32(
        &noise(((hc + 3) * n) as usize, 83)
            .iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>(),
    );
    backend.compute(&graph).unwrap();
    (y.bytes(), folded)
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn cpu_combine16_matches_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (e, n) in [(256, 1), (2560, 3), (96, 9)] {
        let (a, _) = case(&mut cpu, Mode::Direct, e, n);
        let (b, _) = case(&mut cpu, Mode::Chain, e, n);
        let (a, b) = (floats(&a), floats(&b));
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "{e}x{n}, value {i}: {a} vs {b}"
            );
        }
    }
    // The pass folds only on CUDA graphs.
    let ctx = Context::new(16, 16);
    let res16 = ctx.new_tensor(types::BF16, &[96, 4, 2]).as_input();
    let out = ctx.new_tensor(types::F32, &[96, 2]).as_input();
    let inj = ctx.cont(ctx.new_tensor(types::F32, &[4, 2]).as_input());
    let w = ctx.scale(ctx.sigmoid(ctx.scale(inj, 0.25)), 2.0);
    let y = ctx.hc_post(out, ctx.cast(res16, types::F32), w).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run.
    assert_eq!(unsafe { rewrite_graph(graph.raw(), false) }.combine16, 0);
    // SAFETY: a valid node.
    assert!(!unsafe { is_combine16(y.raw()) });
}

/// On a GPU the pass folds ggml's chain into one op with ggml-cuda's bits, at qwen4exp's
/// shape (2560 columns, 4 streams) for a few step sizes and a prompt chunk's (unless
/// `LEGION_DEBUG_HC_COMBINE16=0` keeps the chain).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_combine16_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let off = std::env::var_os("LEGION_DEBUG_HC_COMBINE16").is_some_and(|v| v == "0");
    for (e, n) in [(2560, 1), (2560, 8), (2560, 300), (2560, 4096), (96, 5)] {
        let (ggml, _) = case(&mut gpu, Mode::Chain, e, n);
        let (ours, folded) = case(&mut gpu, Mode::Rewrite, e, n);
        assert_eq!(folded, usize::from(!off), "{e}x{n}: combines folded");
        let diff = ours
            .chunks(4)
            .zip(ggml.chunks(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff, 0, "{e}x{n}: {diff} values moved");
    }
}
