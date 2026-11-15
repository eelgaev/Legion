//! The F16-writing gated mix: the node layout, the CPU reference against ggml's `hc_pre_gated`,
//! the rebuilt chain against the one with `xn` on the CPU, and (ignored, needs a V100) both bit
//! for bit against ggml's ops on CUDA.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn layout_puts_the_f16_part_after_mixed() {
    for (e, n) in [(2560usize, 4097usize), (64, 1), (2, 3)] {
        let at = Layout::new(e, n);
        assert!(at.mixed16 >= e * n * 4 && at.mixed16.is_multiple_of(256));
        assert!(at.floats * 4 >= at.mixed16 + e * n * 2);
        assert!(at.floats * 4 < at.mixed16 + e * n * 2 + 4);
    }
}

#[test]
fn switch_defaults_off() {
    assert!(!on());
    assert!(!rebuild_bf16());
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// [`mix`] (`ours`) or ggml's `hc_pre_gated` and an F16 cast on `backend`, for `n` tokens of
/// `e` columns and 4 streams: the bytes of the mixed values and of their F16 rounding.
fn case(backend: &mut Backend, ours: bool, e: i64, n: i64) -> [Vec<u8>; 2] {
    let hc = 4;
    let ctx = Context::new(16, 16);
    let gate = ctx.new_tensor(types::F32, &[hc * e, n]).as_input();
    let xn = ctx.new_tensor(types::F32, &[hc * e, n]).as_input();
    let (m, m16) = if ours {
        mix(&ctx, gate, xn, hc, 1.0 / hc as f32)
    } else {
        let m = ctx.hc_pre_gated(
            ctx.reshape_3d(xn, e, hc, n),
            ctx.reshape_3d(gate, e, hc, n),
            1.0 / hc as f32,
        );
        (m, ctx.cast(m, types::F16))
    };
    let outs = [m, m16].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(16);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // The gate as the up matmul writes it: mostly small, some large (saturated sigmoids).
    let mut g = noise((hc * e * n) as usize, 81);
    g.iter_mut().step_by(7).for_each(|v| *v *= 40.0);
    gate.set_f32(&g);
    let mut x = noise((hc * e * n) as usize, 82);
    x[32..64].iter_mut().for_each(|v| *v *= 1e-38);
    x[96..128].iter_mut().for_each(|v| *v *= 1e5);
    xn.set_f32(&x);
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

/// On the CPU the reference is ggml's mix up to libm's `expf` and contraction (close, not
/// always equal), and its F16 part is its own F32 values rounded as ggml rounds them.
#[test]
fn cpu_mix_matches_ggml_hc_pre() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for n in [1, 9] {
        let [ours, ours16] = case(&mut cpu, true, 64, n);
        let [ggml, _] = case(&mut cpu, false, 64, n);
        let (ours, ggml) = (floats(&ours), floats(&ggml));
        for (i, (a, b)) in ours.iter().zip(&ggml).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "{n} tokens, value {i}: {a} vs {b}"
            );
        }
        let want: Vec<u8> = ours
            .iter()
            // SAFETY: a plain conversion.
            .flat_map(|&v| unsafe { ffi::ggml_fp32_to_fp16(v) }.to_le_bytes())
            .collect();
        assert!(
            ours16 == want,
            "{n} tokens: the F16 part is not the F32 part rounded"
        );
    }
}

/// The mixer of a prompt step after the combine-and-norm, from the same inputs on `backend`:
/// with `rebuilt`, [`super::super::hc_norm::hc_post_norm_rebuilt16`] (scales, no `xn`) and
/// [`mix_rebuilt`]; else `hc_post_norm` writing `xn` and its F16 copies, and [`mix`]. Both take
/// the down matmul from the F16 copies. `e` columns, 4 streams, rank `rank`, `n` tokens: the
/// bytes of the new streams, the inject weights, the mixed values and their F16 rounding.
fn chain_case(backend: &mut Backend, rebuilt: bool, e: i64, rank: i64, n: i64) -> [Vec<u8>; 4] {
    use super::super::hc_norm::{hc_post_norm, hc_post_norm_rebuilt16};
    let (hc, n_inj) = (4i64, 4i64);
    let k = hc * e;
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let w = ctx.new_tensor(types::F32, &[k, n_inj]).as_input();
    let down = ctx.new_tensor(types::Q8_0, &[k, rank]).as_input();
    let up = ctx.new_tensor(types::Q8_0, &[rank, k]).as_input();
    let scale = 1.0 / hc as f32;
    let gate = |down16, xn16| {
        let lo = ctx.silu(ctx.scale(ctx.mul_mat(down16, xn16), scale));
        ctx.mul_mat(up, lo)
    };
    let (r, ij, (m, m16)) = if rebuilt {
        let p = hc_post_norm_rebuilt16(&ctx, res, out, inj, g, w, 1e-6, down);
        let gate = gate(p.down16, p.xn16);
        let mixed = mix_rebuilt(&ctx, gate, p.res, p.scales, g, hc, scale);
        (p.res, p.inject, mixed)
    } else {
        let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
        let (down16, xn16) = p.down16.zip(p.xn16).expect("the F16 copies");
        (
            p.res,
            p.inject,
            mix(&ctx, gate(down16, xn16), p.xn, hc, scale),
        )
    };
    let outs = [r, ij, m, m16].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((k * n) as usize, 71));
    out.set_f32(&noise((e * n) as usize, 72));
    inj.set_f32(&noise((hc * n) as usize, 73));
    // gamma: noise with an all-zero group, tiny values (flushed to zero on the GPU) and huge ones.
    let mut gamma = noise(k as usize, 74);
    gamma[64..96].fill(0.0);
    gamma[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    gamma[256..288].iter_mut().for_each(|v| *v *= 1e3);
    g.set_f32(&gamma);
    w.set_f32(&noise((k * n_inj) as usize, 75));
    down.set_bytes(&q8_0_matrix(k as usize, rank as usize, 76));
    up.set_bytes(&q8_0_matrix(rank as usize, k as usize, 77));
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

/// On the CPU (the references) the rebuilt chain gives the bytes of the one with `xn`.
#[test]
fn cpu_rebuilt_chain_is_the_xn_chain() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for n in [1, 9] {
        let ours = chain_case(&mut cpu, true, 128, 64, n);
        let xn = chain_case(&mut cpu, false, 128, 64, n);
        for (i, what) in ["streams", "inject", "mixed", "mixed16"].iter().enumerate() {
            assert!(
                ours[i] == xn[i],
                "{n} tokens: the rebuilt chain's {what} moved"
            );
        }
    }
}

/// A V100, or `None` (the comparisons below hold where ggml-cuda runs the projections as cuBLAS
/// F16 GEMMs and the norm has its V100 kernel).
#[cfg(feature = "cuda")]
fn v100() -> Option<Backend> {
    use crate::ggml::{DeviceKind, device_by_kind};
    let gpu = device_by_kind(DeviceKind::Gpu).expect("no GPU device");
    let cc = crate::residency::cuda::caps(0).map(|c| (c.cc_major, c.cc_minor));
    if cc != Some((7, 0)) {
        println!("not a V100 ({cc:?}): not compared");
        return None;
    }
    Some(gpu.init().unwrap())
}

/// On a V100 the op is ggml's `hc_pre_gated` and F16 cast, bit for bit, at qwen4exp's shape
/// (2560 columns, 4 streams); and the time of each path.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a V100"]
fn cuda_mix_is_ggml_bit_for_bit() {
    let Some(mut gpu) = v100() else { return };
    for n in [64, 300, 1000, 4096] {
        let t0 = std::time::Instant::now();
        let ours = case(&mut gpu, true, 2560, n);
        let t1 = std::time::Instant::now();
        let ggml = case(&mut gpu, false, 2560, n);
        println!(
            "{n} tokens: ours {:.1} ms, ggml {:.1} ms (with setup)",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        for (i, what) in ["mixed", "mixed16"].iter().enumerate() {
            let diff = ours[i].iter().zip(&ggml[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "{n} tokens: {diff} {what} bytes moved");
        }
    }
}

/// On a V100 the rebuilt chain (the norm's scales kernel, the op rebuilding `xn`) gives the
/// bytes of the one with `xn` at qwen4exp's shape (2560 columns, rank 320).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a V100"]
fn cuda_rebuilt_chain_is_the_xn_chain() {
    let Some(mut gpu) = v100() else { return };
    for n in [64, 300, 1000, 4096] {
        let t0 = std::time::Instant::now();
        let ours = chain_case(&mut gpu, true, 2560, 320, n);
        let t1 = std::time::Instant::now();
        let xn = chain_case(&mut gpu, false, 2560, 320, n);
        println!(
            "{n} tokens: rebuilt {:.1} ms, xn {:.1} ms (with setup)",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        for (i, what) in ["streams", "inject", "mixed", "mixed16"].iter().enumerate() {
            let diff = ours[i].iter().zip(&xn[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "{n} tokens: {diff} {what} bytes moved");
        }
    }
}

/// The mixer of a prompt step with BF16 mixer weights (the NVFP4 checkpoints) on `backend`:
/// with `rebuilt`, [`super::super::hc_norm::hc_post_norm_rebuilt_xnb`] (scales and the BF16
/// `xn`, no F32 `xn`) and [`mix_rebuilt_f32`]; else `hc_post_norm` writing `xn` and the BF16
/// copy and ggml's `hc_pre_gated` on `xn` (main's chain). Both take the down matmul from the
/// BF16 `xn`. `e` columns, 4 streams, rank `rank`, `n` tokens: the bytes of the new streams,
/// the inject weights and the mixed values (#836).
fn xnb_chain_case(backend: &mut Backend, rebuilt: bool, e: i64, rank: i64, n: i64) -> [Vec<u8>; 3] {
    use super::super::hc_norm::{hc_post_norm, hc_post_norm_rebuilt_xnb};
    let (hc, n_inj) = (4i64, 4i64);
    let k = hc * e;
    let ctx = Context::new(64, 64);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let w = ctx.new_tensor(types::F32, &[k, n_inj]).as_input();
    let down = ctx.new_tensor(types::BF16, &[k, rank]).as_input();
    let up = ctx.new_tensor(types::BF16, &[rank, k]).as_input();
    let scale = 1.0 / hc as f32;
    let gate = |xnb| ctx.mul_mat(up, ctx.silu(ctx.scale(ctx.mul_mat(down, xnb), scale)));
    let (r, ij, m) = if rebuilt {
        let p = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, g, w, 1e-6, down, false);
        let mixed = mix_rebuilt_f32(&ctx, gate(p.xnb), p.res, p.scales, g, hc, scale);
        assert_eq!((mixed.ty(), mixed.ne()), (types::F32, [e, n, 1, 1]));
        (p.res, p.inject, mixed)
    } else {
        let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
        let gate = gate(p.xnb.expect("the BF16 xn"));
        let mixed = ctx.hc_pre_gated(
            ctx.reshape_3d(p.xn, e, hc, n),
            ctx.reshape_3d(gate, e, hc, n),
            scale,
        );
        (p.res, p.inject, mixed)
    };
    let outs = [r, ij, m].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((k * n) as usize, 61));
    out.set_f32(&noise((e * n) as usize, 62));
    inj.set_f32(&noise((hc * n) as usize, 63));
    // gamma: noise with an all-zero group, tiny values (flushed to zero on the GPU) and huge ones.
    let mut gamma = noise(k as usize, 64);
    gamma[64..96].fill(0.0);
    gamma[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    gamma[256..288].iter_mut().for_each(|v| *v *= 1e3);
    g.set_f32(&gamma);
    w.set_f32(&noise((k * n_inj) as usize, 65));
    let bf16 = |len: i64, seed: u64, mul: f32| -> Vec<u8> {
        noise(len as usize, seed)
            .iter()
            .flat_map(|&v| crate::model::nvfp4_mm::bf16_bits(mul * v).to_ne_bytes())
            .collect()
    };
    down.set_bytes(&bf16(k * rank, 66, 0.05));
    up.set_bytes(&bf16(rank * k, 67, 0.5));
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

/// The F32-only op's node is `mixed` alone.
#[test]
fn f32_node_is_mixed_alone() {
    assert_eq!(node_floats(2560, 4097, false), 2560 * 4097);
    assert_eq!(node_floats(64, 3, true), Layout::new(64, 3).floats);
}

/// On the CPU (the references) the rebuilt BF16 chain gives the streams and inject weights of
/// main's chain, and mixed values within libm's `expf` and contraction of ggml's `hc_pre`.
#[test]
fn cpu_rebuilt_xnb_chain_is_the_xn_chain() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for n in [1, 9] {
        let ours = xnb_chain_case(&mut cpu, true, 128, 64, n);
        let xn = xnb_chain_case(&mut cpu, false, 128, 64, n);
        for (i, what) in ["streams", "inject"].iter().enumerate() {
            assert!(
                ours[i] == xn[i],
                "{n} tokens: the rebuilt chain's {what} moved"
            );
        }
        let (a, b) = (floats(&ours[2]), floats(&xn[2]));
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                "{n} tokens, value {i}: {a} vs {b}"
            );
        }
    }
}

/// On sm_80+ (BF16 tensor cores: the GB10) the rebuilt BF16 chain (the register kernel writing
/// scales and the BF16 xn, the op rebuilding xn) is main's chain bit for bit at qwen4exp's shape
/// (2560 columns, rank 320) and the step sizes the model hands it (128+ tokens); and the time
/// of each.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cuda_rebuilt_xnb_chain_is_the_xn_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() {
        println!("no BF16 tensor cores: skipped");
        return;
    }
    for n in [128, 300, 2058, 4096] {
        let t0 = std::time::Instant::now();
        let ours = xnb_chain_case(&mut gpu, true, 2560, 320, n);
        let t1 = std::time::Instant::now();
        let xn = xnb_chain_case(&mut gpu, false, 2560, 320, n);
        println!(
            "{n} tokens: rebuilt {:.1} ms, xn {:.1} ms (with setup)",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        for (i, what) in ["streams", "inject", "mixed"].iter().enumerate() {
            let diff = ours[i].iter().zip(&xn[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "{n} tokens: {diff} {what} bytes moved");
        }
    }
}

#[test]
fn hc_pre_defaults_on() {
    assert!(pre_on());
}

/// [`hc_pre`] (`ours`) or ggml's `hc_pre_gated` on `backend`, for `n` tokens of `e` columns and
/// 4 streams, the operands as the mixer hands them over (`[e, hc, n]` views): the bytes of the
/// mixed values.
fn pre_case(backend: &mut Backend, ours: bool, e: i64, n: i64) -> Vec<u8> {
    let hc = 4;
    let ctx = Context::new(16, 16);
    let gate = ctx.new_tensor(types::F32, &[hc * e, n]).as_input();
    let xn = ctx.new_tensor(types::F32, &[hc * e, n]).as_input();
    let (x3, g3) = (ctx.reshape_3d(xn, e, hc, n), ctx.reshape_3d(gate, e, hc, n));
    let m = if ours {
        hc_pre(&ctx, x3, g3, 1.0 / hc as f32)
    } else {
        ctx.hc_pre_gated(x3, g3, 1.0 / hc as f32)
    };
    let out = ctx.cont(m).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let mut g = noise((hc * e * n) as usize, 91);
    g.iter_mut().step_by(7).for_each(|v| *v *= 40.0);
    gate.set_f32(&g);
    let mut x = noise((hc * e * n) as usize, 92);
    x[32..64].iter_mut().for_each(|v| *v *= 1e-38);
    x[96..128].iter_mut().for_each(|v| *v *= 1e5);
    xn.set_f32(&x);
    backend.compute(&graph).unwrap();
    out.bytes()
}

/// The node is ggml's: its shape, type and sources in ggml's order, so the graph and its
/// allocation do not move.
#[test]
fn hc_pre_node_is_ggml_shaped() {
    let ctx = Context::new(16, 16);
    let (e, hc, n) = (64, 4, 9);
    let x = ctx.new_tensor(types::F32, &[e, hc, n]);
    let g = ctx.new_tensor(types::F32, &[e, hc, n]);
    let ours = hc_pre(&ctx, x, g, 0.25);
    let ggml = ctx.hc_pre_gated(x, g, 0.25);
    assert_eq!(ours.ne(), ggml.ne());
    assert_eq!(ours.nb(), ggml.nb());
    assert_eq!(ours.ty(), ggml.ty());
    // SAFETY: live nodes of `ctx`.
    unsafe {
        let (a, b) = (&*ours.raw(), &*ggml.raw());
        assert_eq!(a.src[..3], b.src[..3]);
    }
    let cuda = cfg!(feature = "cuda");
    assert_eq!(pre_fits(x, g, true), cuda, "a GPU graph takes the op");
    assert!(!pre_fits(x, g, false), "the CPU graph keeps ggml's op");
}

/// On the CPU the op is ggml's mix up to libm's `expf` and contraction.
#[test]
fn cpu_hc_pre_matches_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (e, n) in [(64, 1), (64, 9), (66, 3)] {
        let ours = floats(&pre_case(&mut cpu, true, e, n));
        let ggml = floats(&pre_case(&mut cpu, false, e, n));
        assert_eq!(ours.len(), ggml.len());
        for (i, (a, b)) in ours.iter().zip(&ggml).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "{e} x {n}, value {i}: {a} vs {b}"
            );
        }
    }
}

/// On any CUDA GPU the op is ggml-cuda's hc_pre bit for bit: the four-a-thread kernel at
/// qwen4exp's shape (2560 columns) for decode and prompt step sizes, the one-a-thread kernel
/// for rows of a width not a multiple of 4; and the time of each.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_pre_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let ns = [1, 8, 300, 2048, 4097];
    for (e, n) in ns.map(|n| (2560, n)).into_iter().chain([(2562, 37)]) {
        let t0 = std::time::Instant::now();
        let ours = pre_case(&mut gpu, true, e, n);
        let t1 = std::time::Instant::now();
        let ggml = pre_case(&mut gpu, false, e, n);
        println!(
            "{e} x {n}: ours {:.1} ms, ggml {:.1} ms (with setup)",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        let diff = ours.iter().zip(&ggml).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{e} x {n}: {diff} bytes moved");
    }
}
