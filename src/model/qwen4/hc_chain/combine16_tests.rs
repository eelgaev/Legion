//! The combine of BF16 streams (#1183, `--hc-bf16`) against ggml's copy, scales, sigmoid and
//! `hc_post`: on the CPU within libm's `expf`, and, ignored (needs a GPU), bit for bit.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;

/// ggml's combine of `[e, 4, n]` BF16 streams (`ours`: [`combine16`] instead), the inject a
/// view of a wider node (as the flush's is of the mixer's): the result's bytes.
fn case(backend: &mut Backend, ours: bool, e: i64, n: i64) -> Vec<u8> {
    let hc = 4i64;
    let ctx = Context::new(32, 32);
    let res16 = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let wide = ctx.new_tensor(types::F32, &[hc + 3, n]).as_input();
    let node = ctx.cont(wide);
    let inject = ctx.view_2d(node, [hc, n], node.nb()[1], 4);
    let y = if ours {
        combine16(&ctx, out, res16, inject)
    } else {
        let w = ctx.scale(ctx.sigmoid(ctx.scale(inject, 1.0 / hc as f32)), 2.0);
        ctx.hc_post(out, ctx.cast(res16, types::F32), w)
    }
    .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
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
    y.bytes()
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
        let a = floats(&case(&mut cpu, true, e, n));
        let b = floats(&case(&mut cpu, false, e, n));
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "{e}x{n}, value {i}: {a} vs {b}"
            );
        }
    }
}

/// The op takes BF16 streams of the combine's rows on one GPU only (a CUDA build, unless
/// `LEGION_DEBUG_HC_COMBINE16=0` or `--hc-chain-ggml`).
#[test]
fn combine16_fits_bf16_rows_on_a_gpu() {
    let ctx = Context::new(16, 16);
    let res16 = ctx.new_tensor(types::BF16, &[96, 4, 2]);
    let res32 = ctx.new_tensor(types::F32, &[96, 4, 2]);
    let out = ctx.new_tensor(types::F32, &[96, 2]);
    let wide = ctx.new_tensor(types::F32, &[7, 2]);
    let inject = ctx.view_2d(wide, [4, 2], wide.nb()[1], 4);
    let off = std::env::var_os("LEGION_DEBUG_HC_COMBINE16").is_some_and(|v| v == "0");
    let gpu_on = cfg!(feature = "cuda") && on() && !off;
    assert_eq!(combine16_fits(out, res16, inject, true), gpu_on);
    assert!(!combine16_fits(out, res16, inject, false));
    assert!(!combine16_fits(out, res32, inject, true));
    assert!(!combine16_fits(inject, res16, inject, true));
    assert!(!combine16_fits(out, res16, out, true));
}

/// On a GPU the op gives ggml-cuda's bits, at qwen4exp's shape (2560 columns, 4 streams) for a
/// few step sizes and a prompt chunk's.
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
    for (e, n) in [(2560, 1), (2560, 8), (2560, 300), (2560, 4096), (96, 5)] {
        let ggml = case(&mut gpu, false, e, n);
        let ours = case(&mut gpu, true, e, n);
        let diff = ours
            .chunks(4)
            .zip(ggml.chunks(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff, 0, "{e}x{n}: {diff} values moved");
    }
}
