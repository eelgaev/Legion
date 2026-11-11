//! The fused up matmul and gated mix: ggml's stream-k split rule, the node layout and the
//! quantization, the CPU reference against ggml's `mul_mat` + `hc_pre_gated` on the CPU, and
//! (ignored, needs a GPU) bit for bit against ggml's two ops on CUDA.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn mmq_whole_tiles_follows_ggml_stream_k() {
    // qwen4exp's up matmul (10240 rows, K 320) on a GB10 (48 SMs): 4096 tokens are 2560
    // tiles, 98% of 54 waves, a CTA per tile.
    assert!(mmq_whole_tiles(10240, 320, 4096, 48));
    // 9 tokens: one column tile (J 16), 80 tiles in 2 waves (83%), so 48 CTAs share the 800
    // K blocks; every share starts at a tile (17 -> 10, 33 -> 30, 50, 66 -> 60, ...).
    assert!(mmq_whole_tiles(10240, 320, 9, 48));
    // On 46 SMs the fourth share starts at block 69, moved back to 68: inside tile 6.
    assert!(!mmq_whole_tiles(10240, 320, 9, 46));
    // On 48 SMs every step size of this shape is whole.
    assert!((9..=8192).all(|n| mmq_whole_tiles(10240, 320, n, 48)));
    // Rows ggml tiles with the fallback configs, and nothing to tile.
    assert!(!mmq_whole_tiles(10240 + 64, 320, 4096, 48));
    assert!(!mmq_whole_tiles(10240, 320, 0, 48));
    assert!(!mmq_whole_tiles(10240, 320, 4096, 0));
}

#[test]
fn layout_parts_follow_each_other() {
    let (e, rank, n) = (2560usize, 320usize, 4097usize);
    let at = Layout::new(e, rank, n);
    assert!(at.lq >= e * n * 4 && at.lq.is_multiple_of(256));
    assert!(at.ld >= at.lq + rank * n && at.ld.is_multiple_of(256));
    assert!(at.floats * 4 >= at.ld + rank / 32 * n * 4);
    assert!(at.floats * 4 < at.ld + rank / 32 * n * 4 + 4);
}

#[test]
fn bf16_nodes_are_ggmls_nodes() {
    // A BF16 `up`: `lo` as BF16 in a node of the gate's shape, then the mixed values as a plain
    // `[e, n]` node reading it, as ggml's `mul_mat` and `hc_pre` (#836).
    let (e, rank, n) = (256i64, 320i64, 300i64);
    // Four tensors here and the op's nodes.
    let ctx = Context::new(16, 4);
    let up = ctx.new_tensor(types::BF16, &[rank, 4 * e]);
    let lo = ctx.cont(ctx.new_tensor(types::F32, &[rank, n]));
    let xn = ctx.new_tensor(types::F32, &[4 * e, n]);
    let mixed = up_mix(&ctx, up, lo, xn, 4, 0.25);
    assert_eq!(mixed.ne(), [e, n, 1, 1]);
    let srcs = mixed.srcs();
    assert_eq!(srcs.len(), 3);
    assert_eq!(srcs[1].ne(), [4 * e, n, 1, 1]);
    assert_eq!(srcs[1].srcs()[0].ne(), lo.ne());
    // SAFETY: live tensors; fields read only.
    unsafe {
        assert!((*mixed.raw()).view_src.is_null());
        assert!((*srcs[1].raw()).view_src.is_null());
        // The BF16 `lo` fits the gate's place.
        assert!((rank * n * 2) as usize <= srcs[1].elements() * 4);
    }
}

#[test]
fn quantize_group_is_ggmls() {
    let mut x = noise(32, 7);
    x[5] = -2.0;
    let mut q = [0i8; 32];
    let d = quantize_group(&x, &mut q);
    assert_eq!(d, 1.0 / (127.0 / 2.0));
    assert_eq!(q[5], -127);
    for (v, q) in x.iter().zip(q) {
        assert!((f32::from(q) * d - v).abs() <= d * 0.5 + 1e-6);
    }
    // An all-zero group: scale 0, quants 0 (NaN converts to 0).
    let d = quantize_group(&[0.0; 32], &mut q);
    assert_eq!((d, q), (0.0, [0i8; 32]));
}

#[test]
fn bf16_up_mix_defaults_on() {
    // On by default (#836): in ggml's two-node layout the greedy output equals ggml's ops'.
    assert!(BF16_ON.load(Ordering::Relaxed));
}

#[test]
fn fits_needs_the_gpu() {
    // On no GPU (the default until the engine sets one) nothing fits.
    let ctx = Context::new(4, 4);
    let up = ctx.new_tensor(types::Q8_0, &[320, 10240]);
    let lo = ctx.new_tensor(types::F32, &[320, 4096]);
    let xn = ctx.new_tensor(types::F32, &[10240, 4096]);
    assert!(!fits(up, lo, xn, 4));
    let up = ctx.new_tensor(types::BF16, &[320, 10240]);
    assert!(!fits(up, lo, xn, 4));
    assert!(!fits_up_bf16(up, 4096, 4));
}

/// A BF16 `[k, rows]` matrix of noise, as bytes.
fn bf16_matrix(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    noise(k * rows, seed)
        .into_iter()
        .flat_map(|v| crate::model::nvfp4_mm::bf16_bits(v * 0.25).to_ne_bytes())
        .collect()
}

/// `up_mix` (`ours`) or ggml's `mul_mat` + `hc_pre_gated` on `backend`, for `n` tokens of
/// `e` columns, 4 streams and rank `rank`: the mixed values' bytes.
fn case(backend: &mut Backend, ours: bool, e: i64, rank: i64, n: i64, up: &[u8]) -> Vec<u8> {
    case_typed(backend, types::Q8_0, ours, e, rank, n, up)
}

/// [`case`] with an `up` of type `ty` (Q8_0 or BF16).
fn case_typed(
    backend: &mut Backend,
    ty: ffi::ggml_type,
    ours: bool,
    e: i64,
    rank: i64,
    n: i64,
    up: &[u8],
) -> Vec<u8> {
    let hc = 4;
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(ty, &[rank, hc * e]).as_input();
    let lo = ctx.new_tensor(types::F32, &[rank, n]).as_input();
    let xn = ctx.new_tensor(types::F32, &[hc * e, n]).as_input();
    let y = if ours {
        up_mix(&ctx, w, lo, xn, hc, 1.0 / hc as f32)
    } else {
        let gate = ctx.mul_mat(w, lo);
        ctx.hc_pre_gated(
            ctx.reshape_3d(xn, e, hc, n),
            ctx.reshape_3d(gate, e, hc, n),
            1.0 / hc as f32,
        )
    };
    let y = ctx.cont(y).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(up);
    // lo as a SiLU output would be: mostly small, some larger.
    lo.set_f32(
        &noise((rank * n) as usize, 62)
            .iter()
            .map(|v| v * 1.5)
            .collect::<Vec<_>>(),
    );
    xn.set_f32(&noise((hc * e * n) as usize, 63));
    backend.compute(&graph).unwrap();
    y.bytes()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// On the CPU the reference quantizes `lo` to Q8_1-style F32 scales, ggml's CPU `mul_mat` to
/// Q8_0 (F16 scales): close, not equal.
#[test]
fn cpu_up_mix_matches_ggml_chain() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (e, rank) = (64i64, 320i64);
    let up = q8_0_matrix(rank as usize, (4 * e) as usize, 64);
    for n in [1, 9] {
        let ours = floats(&case(&mut cpu, true, e, rank, n, &up));
        let ggml = floats(&case(&mut cpu, false, e, rank, n, &up));
        let scale = ggml.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = ours
            .iter()
            .zip(&ggml)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 2e-3 * scale, "{n} tokens: {worst:e} of {scale:e}");
    }
}

/// With a BF16 `up` the CPU reference (F32 sums of `lo` rounded to BF16) is close to ggml's CPU
/// `mul_mat` + `hc_pre_gated` (its own BF16 dot products).
#[test]
fn cpu_bf16_up_mix_matches_ggml_chain() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (e, rank) = (64i64, 320i64);
    let up = bf16_matrix(rank as usize, (4 * e) as usize, 65);
    for n in [1, 9] {
        let ours = floats(&case_typed(&mut cpu, types::BF16, true, e, rank, n, &up));
        let ggml = floats(&case_typed(&mut cpu, types::BF16, false, e, rank, n, &up));
        let scale = ggml.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "nothing mixed");
        let worst = ours
            .iter()
            .zip(&ggml)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 1e-4 * scale, "{n} tokens: {worst:e} of {scale:e}");
    }
}

/// On CUDA the BF16 op is close to ggml's `mul_mat` (cuBLAS) + `hc_pre_gated` at qwen4exp's
/// shape (2560 columns, 4 streams, rank 320) for prompt step sizes, and how many values differ
/// in their bits (cuBLAS picks its kernel per shape, so not always none: the op is opt-in); and
/// the time of each path.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_bf16_up_mix_is_close_to_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, rank) = (2560i64, 320i64);
    let up = bf16_matrix(rank as usize, (4 * e) as usize, 66);
    for n in [256, 300, 1000, 2048, 4096] {
        let mut timed = |ours: bool| {
            let t0 = std::time::Instant::now();
            let y = case_typed(&mut gpu, types::BF16, ours, e, rank, n, &up);
            (y, t0.elapsed().as_secs_f64() * 1e3)
        };
        let (ours, ours_ms) = timed(true);
        let (ggml, ggml_ms) = timed(false);
        println!("{n} tokens: up_mix {ours_ms:.1} ms, ggml {ggml_ms:.1} ms (with setup)");
        let (a, b) = (floats(&ours), floats(&ggml));
        let diff = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!(
            "{n} tokens: {diff} of {} values differ, worst {worst:e}",
            a.len()
        );
        assert!(worst <= 1e-4 * scale, "{n} tokens: {worst:e} of {scale:e}");
    }
}

/// The mixer of a prompt step after the combine-and-norm, from the same inputs on `backend`:
/// with `rebuilt`, [`super::super::hc_norm::hc_post_norm_rebuilt`] (scales, no `xn`) and
/// [`up_mix_rebuilt`]; else `hc_post_norm` writing `xn` and [`up_mix`]. Both take the down
/// matmul from the quantized `xn`. `e` columns, 4 streams, rank `rank`, `n` tokens: the bytes of
/// the new streams, the inject weights and the mixed values.
fn chain_case(backend: &mut Backend, rebuilt: bool, e: i64, rank: i64, n: i64) -> [Vec<u8>; 3] {
    use super::super::hc_norm::{hc_post_norm, hc_post_norm_rebuilt};
    use crate::model::core::mmq_preq;
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
    let silu = |lo| ctx.silu(ctx.scale(lo, 1.0 / hc as f32));
    let (r, ij, mixed) = if rebuilt {
        let p = hc_post_norm_rebuilt(&ctx, res, out, inj, g, w, 1e-6, false);
        let lo = silu(mmq_preq::mul_mat_cols(&ctx, down, p.xq, n));
        let m = up_mix_rebuilt(&ctx, up, lo, p.res, p.scales, g, hc, 1.0 / hc as f32);
        (p.res, p.inject, m)
    } else {
        let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, None, true);
        let lo = silu(mmq_preq::mul_mat(&ctx, down, p.xq.expect("xq"), p.xn));
        (
            p.res,
            p.inject,
            up_mix(&ctx, up, lo, p.xn, hc, 1.0 / hc as f32),
        )
    };
    let outs = [r, ij, mixed].map(|t| ctx.cont(t).as_output());
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
    gamma[256..288].iter_mut().for_each(|v| *v *= 1e30);
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
        assert!(ours == xn, "{n} tokens: the rebuilt chain moved");
    }
}

/// `up_mix_rebuilt` from BF16 streams (#826) gives the bytes of the op from F32 streams
/// holding the same values (on the CPU, the reference).
#[test]
fn cpu_rebuilt_bf16_streams_are_the_f32_ones() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_bf16_streams(&mut cpu, 128, 64, 9);
}

/// The same on CUDA at qwen4exp's shape (`up_mix_mma` with BF16 streams against F32 ones).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rebuilt_bf16_streams_are_the_f32_ones() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for n in [9, 300, 4096] {
        check_bf16_streams(&mut gpu, 2560, 320, n);
    }
}

/// `up_mix_rebuilt` on `backend` from F32 streams and from BF16 ones with the same values, `e`
/// columns, 4 streams, rank `rank`, `n` tokens: the same bytes.
fn check_bf16_streams(backend: &mut Backend, e: i64, rank: i64, n: i64) {
    use crate::model::nvfp4_mm::bf16_bits;
    let hc = 4i64;
    let k = hc * e;
    let ctx = Context::new(64, 64);
    let up = ctx.new_tensor(types::Q8_0, &[rank, k]).as_input();
    let lo = ctx.new_tensor(types::F32, &[rank, n]).as_input();
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let res16 = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let sc = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let outs = [res, res16].map(|r| {
        let m = up_mix_rebuilt(&ctx, up, lo, r, sc, g, hc, 1.0 / hc as f32);
        ctx.cont(m).as_output()
    });
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let bits: Vec<u16> = noise((k * n) as usize, 61)
        .into_iter()
        .map(bf16_bits)
        .collect();
    let widened: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    res.set_f32(&widened);
    res16.set_bytes(
        &bits
            .iter()
            .flat_map(|b| b.to_ne_bytes())
            .collect::<Vec<u8>>(),
    );
    up.set_bytes(&q8_0_matrix(rank as usize, k as usize, 62));
    lo.set_f32(&noise((rank * n) as usize, 63));
    sc.set_f32(&noise((hc * n) as usize, 64));
    g.set_f32(&noise(k as usize, 65));
    backend.compute(&graph).unwrap();
    let [f, b] = outs.map(|t| t.bytes());
    assert!(floats(&f).iter().any(|v| *v != 0.0), "nothing mixed");
    assert!(f == b, "the BF16 streams moved the op");
}

/// On CUDA the rebuilt chain gives the bytes of the one with `xn` at qwen4exp's shape, with
/// each of the norm's kernels (the default one, the register one, the generic one).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rebuilt_chain_is_the_xn_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for mode in [0, 2, 1] {
        super::super::hc_norm::force_post_norm_kernel(mode);
        for n in [9, 256, 300, 1000, 4096] {
            let t0 = std::time::Instant::now();
            let ours = chain_case(&mut gpu, true, 2560, 320, n);
            let t1 = std::time::Instant::now();
            let xn = chain_case(&mut gpu, false, 2560, 320, n);
            println!(
                "kernel {mode}, {n} tokens: rebuilt {:.1} ms, xn {:.1} ms (with setup)",
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
            for (i, what) in ["streams", "inject", "mixed"].iter().enumerate() {
                let diff = floats(&ours[i])
                    .iter()
                    .zip(floats(&xn[i]))
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    diff, 0,
                    "kernel {mode}, {n} tokens: {diff} {what} values moved"
                );
            }
        }
    }
    super::super::hc_norm::force_post_norm_kernel(0);
}

/// On CUDA the op is ggml's `mul_mat` (MMQ) + `hc_pre_gated`, bit for bit, at qwen4exp's shape
/// (2560 columns, 4 streams, rank 320) for step sizes ggml computes in whole tiles, from 9
/// tokens (ggml's first MMQ size) up; and the time of each path.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_up_mix_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let sms = crate::residency::cuda::caps(0).map_or(0, |c| c.sms) as i64;
    let (e, rank) = (2560i64, 320i64);
    let up = q8_0_matrix(rank as usize, (4 * e) as usize, 61);
    for n in [9, 16, 100, 128, 129, 300, 790, 1000, 2048, 4096] {
        if !mmq_whole_tiles(4 * e, rank, n, sms) {
            println!("{n} tokens: ggml splits tiles on {sms} SMs, not compared");
            continue;
        }
        let mut timed = |ours: bool| {
            let t0 = std::time::Instant::now();
            let y = case(&mut gpu, ours, e, rank, n, &up);
            (y, t0.elapsed().as_secs_f64() * 1e3)
        };
        let (ours, ours_ms) = timed(true);
        let (ggml, ggml_ms) = timed(false);
        println!("{n} tokens: up_mix {ours_ms:.1} ms, ggml {ggml_ms:.1} ms (with setup)");
        if ours != ggml {
            let (a, b) = (floats(&ours), floats(&ggml));
            let diff = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            panic!(
                "{n} tokens: {diff} of {} values differ from ggml's",
                a.len()
            );
        }
    }
}

#[test]
fn bf16_rebuilt_nodes_are_the_xn_nodes() {
    // A BF16 `up` rebuilding `xn` (#836): the same two nodes as `up_mix`'s, the mix node
    // reading the streams, scales and gamma in place of `xn`.
    let (e, rank, n) = (256i64, 320i64, 300i64);
    let ctx = Context::new(16, 8);
    let up = ctx.new_tensor(types::BF16, &[rank, 4 * e]);
    let lo = ctx.cont(ctx.new_tensor(types::F32, &[rank, n]));
    let xn = ctx.new_tensor(types::F32, &[4 * e, n]);
    let res = ctx.new_tensor(types::F32, &[e, 4, n]);
    let sc = ctx.new_tensor(types::F32, &[4, n]);
    let g = ctx.new_tensor(types::F32, &[4 * e]);
    let plain = up_mix(&ctx, up, lo, xn, 4, 0.25);
    let ours = up_mix_rebuilt(&ctx, up, lo, res, sc, g, 4, 0.25);
    assert_eq!(ours.ne(), plain.ne());
    let (a, b) = (ours.srcs(), plain.srcs());
    assert_eq!(a.len(), 5);
    assert_eq!(a[1].ne(), b[1].ne());
    assert_eq!(a[1].srcs()[0].ne(), lo.ne());
    assert_eq!(ours.nbytes(), plain.nbytes(), "the mix node's size");
    // SAFETY: a live tensor; a field read only.
    unsafe {
        assert!((*ours.raw()).view_src.is_null());
    }
}

/// The mixer of a prompt step with BF16 mixer weights (the NVFP4 checkpoints) where the BF16
/// up-mix takes it, from the same inputs on `backend`: with `rebuilt`,
/// [`super::super::hc_norm::hc_post_norm_rebuilt_xnb`] (scales and the BF16 `xn`, no F32 `xn`)
/// and [`up_mix_rebuilt`]; else `hc_post_norm` writing `xn` and the BF16 copy and [`up_mix`]
/// (main's chain). Both take the down matmul from the BF16 `xn`. `e` columns, 4 streams, rank
/// `rank`, `n` tokens: the bytes of the new streams, the inject weights and the mixed values.
fn bf16_chain_case(
    backend: &mut Backend,
    rebuilt: bool,
    e: i64,
    rank: i64,
    n: i64,
) -> [Vec<u8>; 3] {
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
    let lo = |xnb| ctx.silu(ctx.scale(ctx.mul_mat(down, xnb), scale));
    let (r, ij, m) = if rebuilt {
        let p = hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, g, w, 1e-6, down, false);
        let mixed = up_mix_rebuilt(&ctx, up, lo(p.xnb), p.res, p.scales, g, hc, scale);
        (p.res, p.inject, mixed)
    } else {
        let p = hc_post_norm(&ctx, res, out, inj, g, w, 1e-6, Some(down), false);
        let mixed = up_mix(&ctx, up, lo(p.xnb.expect("the BF16 xn")), p.xn, hc, scale);
        (p.res, p.inject, mixed)
    };
    let outs = [r, ij, m].map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(&noise((k * n) as usize, 81));
    out.set_f32(&noise((e * n) as usize, 82));
    inj.set_f32(&noise((hc * n) as usize, 83));
    // gamma: noise with an all-zero group, tiny values (flushed to zero on the GPU) and huge ones.
    let mut gamma = noise(k as usize, 84);
    gamma[64..96].fill(0.0);
    gamma[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    gamma[256..288].iter_mut().for_each(|v| *v *= 1e3);
    g.set_f32(&gamma);
    w.set_f32(&noise((k * n_inj) as usize, 85));
    down.set_bytes(&bf16_matrix(k as usize, rank as usize, 86));
    up.set_bytes(&bf16_matrix(rank as usize, k as usize, 87));
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

/// On the CPU (the references) the rebuilt BF16 chain gives the bytes of the one with `xn`.
#[test]
fn cpu_rebuilt_bf16_chain_is_the_xn_chain() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for n in [1, 9] {
        let ours = bf16_chain_case(&mut cpu, true, 128, 64, n);
        let xn = bf16_chain_case(&mut cpu, false, 128, 64, n);
        assert!(floats(&xn[2]).iter().any(|v| *v != 0.0), "nothing mixed");
        assert!(ours == xn, "{n} tokens: the rebuilt BF16 chain moved");
    }
}

/// On sm_80+ (BF16 tensor cores: the GB10) the rebuilt BF16 chain (the register kernel writing
/// scales and the BF16 xn, `up_mix_bf16<true>` rebuilding xn) is main's chain bit for bit at
/// qwen4exp's shape (2560 columns, rank 320) and the step sizes the BF16 up-mix takes (256+
/// tokens); and the time of each.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cuda_rebuilt_bf16_chain_is_the_xn_chain() {
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
    for n in [256, 300, 2058, 4096] {
        let t0 = std::time::Instant::now();
        let ours = bf16_chain_case(&mut gpu, true, 2560, 320, n);
        let t1 = std::time::Instant::now();
        let xn = bf16_chain_case(&mut gpu, false, 2560, 320, n);
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

/// The BF16 up-mix rebuilding xn from BF16 streams (`--hc-bf16`, #1183) and from F32 streams
/// holding the same values, on `backend`: the mixed values' bytes of each.
fn bf16_streams_case(backend: &mut Backend, e: i64, rank: i64, n: i64) -> [Vec<u8>; 2] {
    let hc = 4i64;
    let k = hc * e;
    let ctx = Context::new(64, 64);
    let up = ctx.new_tensor(types::BF16, &[rank, k]).as_input();
    let lo = ctx.new_tensor(types::F32, &[rank, n]).as_input();
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let res16 = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
    let sc = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let outs = [res, res16].map(|r| up_mix_rebuilt(&ctx, up, lo, r, sc, g, hc, 0.25).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    up.set_bytes(&bf16_matrix(rank as usize, k as usize, 71));
    lo.set_f32(&noise((rank * n) as usize, 72));
    let r: Vec<u16> = noise((k * n) as usize, 73)
        .into_iter()
        .map(crate::model::nvfp4_mm::bf16_bits)
        .collect();
    let wide: Vec<f32> = r
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    res.set_f32(&wide);
    res16.set_bytes(&r.iter().flat_map(|b| b.to_ne_bytes()).collect::<Vec<_>>());
    sc.set_f32(
        &noise((hc * n) as usize, 74)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<_>>(),
    );
    g.set_f32(&noise(k as usize, 75));
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

/// BF16 streams widen exactly, so the mix is the F32 streams' bit for bit (the CPU reference).
#[test]
fn cpu_bf16_streams_up_mix_is_the_f32_one() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for n in [1, 9] {
        let [f, b] = bf16_streams_case(&mut cpu, 128, 64, n);
        assert!(floats(&f).iter().any(|v| *v != 0.0), "nothing mixed");
        assert!(f == b, "{n} tokens: BF16 streams moved the mix");
    }
}

/// The same on sm_80+ at qwen4exp's shape: `up_mix_bf16<true, true>` against
/// `up_mix_bf16<true>`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cuda_bf16_streams_up_mix_is_the_f32_one() {
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
    for n in [256, 300, 2058] {
        let [f, b] = bf16_streams_case(&mut gpu, 2560, 320, n);
        assert!(f == b, "{n} tokens: BF16 streams moved the mix");
    }
}
