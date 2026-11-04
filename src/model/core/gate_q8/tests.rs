//! The gate-and-quantize op: how it reads its operands, the CPU reference, and (ignored, needs
//! a GPU) bit for bit against ggml's sigmoid + mul, quantization and MMQ on CUDA.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{ggml_quantize, noise, q8_0_matrix};

#[test]
fn rows_of_reads_rows_and_head_views() {
    let f32 = types::F32;
    // [k, n] rows, contiguous or with a longer stride.
    let r = rows_of(f32, [6144, 300, 1, 1], [4, 6144 * 4, 0, 0], 6144, 300);
    assert_eq!(
        r,
        Some(Rows {
            ld: 6144,
            run: 6144,
            stride: 6144
        })
    );
    let r = rows_of(f32, [6144, 300, 1, 1], [4, 8192 * 4, 0, 0], 6144, 300);
    assert_eq!(r.map(|r| (r.ld, r.run)), Some((8192, 6144)));
    // The attention's gate: head h of token t at t · 2 d n_head + h · 2 d + d.
    let (d, h) = (256i64, 24i64);
    let nb = [4, (2 * d * 4) as usize, (2 * d * h * 4) as usize, 0];
    let r = rows_of(f32, [d, h, 300, 1], nb, d * h, 300);
    assert_eq!(
        r,
        Some(Rows {
            ld: 2 * d * h,
            run: d,
            stride: 2 * d
        })
    );
    // Heads that follow each other are one run.
    let nb = [4, (d * 4) as usize, (d * h * 4) as usize, 0];
    let r = rows_of(f32, [d, h, 300, 1], nb, d * h, 300);
    assert_eq!(r.map(|r| (r.ld, r.run)), Some((d * h, d * h)));
    // Runs that split a group of 32, other types, other shapes: not taken.
    let nb = [4, 2 * 48 * 4, 2 * 48 * 128 * 4, 0];
    assert_eq!(rows_of(f32, [48, 128, 300, 1], nb, 48 * 128, 300), None);
    let f16 = [2, 6144 * 2, 0, 0];
    assert_eq!(rows_of(types::F16, [6144, 300, 1, 1], f16, 6144, 300), None);
    assert_eq!(
        rows_of(f32, [6144, 299, 1, 1], [4, 6144 * 4, 0, 0], 6144, 300),
        None
    );
    assert_eq!(rows_of(f32, [4, 1, 1, 1], [3, 12, 12, 12], 4, 1), None);
}

#[test]
fn fits_needs_the_gpu() {
    // On no GPU (the default until the engine sets one) nothing fits.
    let ctx = Context::new(4, 4);
    let w = ctx.new_tensor(types::Q8_0, &[6144, 2560]);
    let x = ctx.new_tensor(types::F32, &[6144, 4096]);
    let g = ctx.new_tensor(types::F32, &[6144, 4096]);
    assert!(!fits(w, x, g, 4096));
}

/// The attention's layout on `backend`: `o` F32 `[d · h, n]` and the gate inside an
/// interleaved `[q_h | gate_h] × h` projection. With `ours`, [`gate_quantized`] of `o` and the
/// gate view; else ggml's `mul(o, sigmoid(cont(gate)))` quantized by ggml
/// ([`ggml_quantize`]). Then (with `w`, a Q8_0 `[d · h, rows]`) the matmul: ours
/// [`mmq_preq::mul_mat_cols`], ggml's `mul_mat`. The quantized bytes (without MMQ's padding)
/// and the product's.
fn case(backend: &mut Backend, ours: bool, d: i64, h: i64, rows: i64, n: i64) -> [Vec<u8>; 2] {
    let k = d * h;
    let ctx = Context::new(16, 16);
    let o = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let qg = ctx.new_tensor(types::F32, &[2 * k, n]).as_input();
    let w = ctx.new_tensor(types::Q8_0, &[k, rows]).as_input();
    let nb = [(2 * d * 4) as usize, (2 * d * h * 4) as usize];
    let gate = ctx.view_3d(qg, [d, h, n], nb, d as usize * 4);
    let (xq, y) = if ours {
        let xq = gate_quantized(&ctx, o, gate, None, k, n);
        (xq, mmq_preq::mul_mat_cols(&ctx, w, xq, n))
    } else {
        let x = ctx.mul(o, ctx.sigmoid(ctx.cont_2d(gate, k, n)));
        (ggml_quantize(&ctx, x), ctx.mul_mat(w, x))
    };
    let outs = [xq, y].map(|t| t.as_output());
    let mut graph = ctx.new_graph(16);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // o: noise with an all-zero group and tiny values (flushed to zero on the GPU).
    let mut ov = noise((k * n) as usize, 81);
    ov[64..96].fill(0.0);
    ov[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    o.set_f32(&ov);
    // Gate logits over a wide range: sigmoid saturates at both ends.
    qg.set_f32(
        &noise((2 * k * n) as usize, 82)
            .iter()
            .map(|v| v * 24.0)
            .collect::<Vec<_>>(),
    );
    w.set_bytes(&q8_0_matrix(k as usize, rows as usize, 83));
    backend.compute(&graph).unwrap();
    let data = (k / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    let [xq, y] = outs.map(|t| t.bytes());
    [xq[..data].to_vec(), y]
}

/// On the CPU the op's blocks are the reference quantization of `o · sigmoid(gate)`, read
/// through the gate view.
#[test]
fn cpu_gate_quantized_is_the_reference() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let (d, h, n) = (64i64, 8i64, 5i64);
    let k = d * h;
    let ours = case(&mut cpu, true, d, h, 128, n);
    let ov = {
        let mut v = noise((k * n) as usize, 81);
        v[64..96].fill(0.0);
        v[160..192].iter_mut().for_each(|v| *v *= 1e-38);
        v
    };
    let qg: Vec<f32> = noise((2 * k * n) as usize, 82)
        .iter()
        .map(|v| v * 24.0)
        .collect();
    let mut y = vec![0f32; (k * n) as usize];
    for t in 0..n {
        for i in 0..k {
            let z = qg[(t * 2 * k + i / d * 2 * d + d + i % d) as usize];
            let at = (t * k + i) as usize;
            y[at] = (1.0 / (1.0 + (-z).exp())) * ov[at];
        }
    }
    let mut want = vec![0u8; mmq_preq::q8_1_bytes(k, n)];
    mmq_preq::quantize_ref(&y, k as usize, n as usize, &mut want);
    assert!(ours[0] == want[..ours[0].len()], "the blocks moved");
}

#[test]
fn norm_fits_takes_rows_of_128() {
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[128, 48, 300]);
    let w = ctx.new_tensor(types::F32, &[128]);
    assert!(norm_fits(x, w));
    // Rows of another length, a weight of another shape or type: not taken.
    let x64 = ctx.new_tensor(types::F32, &[64, 96, 300]);
    let flat = ctx.new_tensor(types::F32, &[6144, 300]);
    assert!(!norm_fits(x64, ctx.new_tensor(types::F32, &[64])));
    assert!(!norm_fits(flat, w));
    assert!(!norm_fits(x, ctx.new_tensor(types::F32, &[128, 2])));
    assert!(!norm_fits(x, ctx.new_tensor(types::F16, &[128])));
}

/// ggml-cuda's `rms_norm_f32<256>` sums a row of 128 squares with `block_reduce`: each warp's
/// butterfly, then a butterfly over the 8 warp sums (warps 4-7 hold zeros) padded with zeros to
/// 32 lanes. That comes to [`norm_row`]'s `(S0 + S2) + (S1 + S3)`, bit for bit.
#[test]
fn norm_row_sums_as_ggml_block_reduce() {
    for seed in 0..64 {
        // Squares over many magnitudes, so the order of the adds shows.
        let x: Vec<f32> = noise(128, 300 + seed)
            .iter()
            .enumerate()
            .map(|(i, v)| v * 10f32.powi((i % 9) as i32 - 4))
            .collect();
        let sq = |c: usize| if c < 128 { x[c] * x[c] } else { 0.0 };
        let warps: Vec<f32> = (0..8)
            .map(|w| warp_sum(std::array::from_fn(|l| sq(32 * w + l))))
            .collect();
        let block = warp_sum(std::array::from_fn(|l| if l < 8 { warps[l] } else { 0.0 }));
        // norm_row with a unit weight and eps 0: scale = 1 / sqrt(block / 128).
        let w = [1f32; 128];
        let mut y = x.clone();
        norm_row(&mut y, &w, 0.0);
        let scale = 1.0 / (block / 128.0).sqrt();
        let want: Vec<f32> = x.iter().map(|v| scale * v).collect();
        assert_eq!(y, want, "seed {seed}");
    }
}

/// A gated DeltaNet layer's output on `backend`: `o` and `z` F32 `[128, h, n]`, the norm weight
/// `[128]`, `w` a Q8_0 `[128 h, rows]`. With `ours`, [`gate_quantized`] with the [`Norm`];
/// else ggml's `mul(mul(rms_norm(o), nw), sigmoid(z))` quantized by ggml. Then the matmul as in
/// [`case`]. The quantized bytes (without MMQ's padding) and the product's.
fn case_norm(backend: &mut Backend, ours: bool, h: i64, rows: i64, n: i64) -> [Vec<u8>; 2] {
    let (d, eps) = (128i64, 1e-6f32);
    let k = d * h;
    let ctx = Context::new(16, 16);
    let o = ctx.new_tensor(types::F32, &[d, h, n]).as_input();
    let z = ctx.new_tensor(types::F32, &[d, h, n]).as_input();
    let nw = ctx.new_tensor(types::F32, &[d]).as_input();
    let w = ctx.new_tensor(types::Q8_0, &[k, rows]).as_input();
    let (xq, y) = if ours {
        let xq = gate_quantized(&ctx, o, z, Some(Norm { w: nw, eps }), k, n);
        (xq, mmq_preq::mul_mat_cols(&ctx, w, xq, n))
    } else {
        let normed = ctx.mul(ctx.rms_norm(o, eps), nw);
        let x = ctx.reshape_2d(ctx.mul(normed, ctx.sigmoid(z)), k, n);
        (ggml_quantize(&ctx, x), ctx.mul_mat(w, x))
    };
    let outs = [xq, y].map(|t| t.as_output());
    let mut graph = ctx.new_graph(16);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    o.set_f32(&norm_case_o(k, n));
    z.set_f32(
        &noise((k * n) as usize, 92)
            .iter()
            .map(|v| v * 24.0)
            .collect::<Vec<_>>(),
    );
    nw.set_f32(&noise(d as usize, 93));
    w.set_bytes(&q8_0_matrix(k as usize, rows as usize, 94));
    backend.compute(&graph).unwrap();
    let data = (k / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    let [xq, y] = outs.map(|t| t.bytes());
    [xq[..data].to_vec(), y]
}

/// [`case_norm`]'s `o`: noise over several magnitudes, an all-zero row (the norm of zeros is
/// zero) and one of tiny values.
fn norm_case_o(k: i64, n: i64) -> Vec<f32> {
    let mut v: Vec<f32> = noise((k * n) as usize, 91)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 4f32.powi((i / 128 % 5) as i32 - 2))
        .collect();
    v[128..256].fill(0.0);
    v[384..512].iter_mut().for_each(|v| *v *= 1e-20);
    v
}

/// On the CPU the normed op's blocks are the reference quantization of
/// `norm_row(o) · sigmoid(z)`.
#[test]
fn cpu_gate_quantized_normed_is_the_reference() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let (h, n) = (4i64, 5i64);
    let k = 128 * h;
    let ours = case_norm(&mut cpu, true, h, 128, n);
    let nw = noise(128, 93);
    let z: Vec<f32> = noise((k * n) as usize, 92)
        .iter()
        .map(|v| v * 24.0)
        .collect();
    let mut y = norm_case_o(k, n);
    for row in y.chunks_mut(128) {
        norm_row(row, &nw, 1e-6);
    }
    for (v, z) in y.iter_mut().zip(&z) {
        *v *= 1.0 / (1.0 + (-z).exp());
    }
    let mut want = vec![0u8; mmq_preq::q8_1_bytes(k, n)];
    mmq_preq::quantize_ref(&y, k as usize, n as usize, &mut want);
    assert!(ours[0] == want[..ours[0].len()], "the blocks moved");
}

/// On CUDA the normed op writes ggml-cuda's blocks for ggml's normed and gated values, and the
/// matmul gives ggml's product, bit for bit, at qwen4exp's DeltaNet shape (48 heads of 128).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gate_norm_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for n in [256, 300, 1000, 4096] {
        let ours = case_norm(&mut gpu, true, 48, 2560, n);
        let ggml = case_norm(&mut gpu, false, 48, 2560, n);
        for (i, what) in ["quantized", "product"].iter().enumerate() {
            let diff = ours[i].iter().zip(&ggml[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "48 x 128, {n} tokens: {diff} {what} bytes moved");
        }
    }
}

/// On CUDA the op writes ggml-cuda's blocks for ggml's gated values, and the matmul gives
/// ggml's product, bit for bit, at qwen4exp's shapes (attention: 24 heads of 256, the gate
/// interleaved; DeltaNet: 6144 values a token as one run) from 256 tokens up; and each path's
/// time.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gate_quantized_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (d, h) in [(256i64, 24i64), (6144, 1)] {
        for n in [256, 300, 1000, 4096] {
            let t0 = std::time::Instant::now();
            let ours = case(&mut gpu, true, d, h, 2560, n);
            let t1 = std::time::Instant::now();
            let ggml = case(&mut gpu, false, d, h, 2560, n);
            println!(
                "{h} x {d}, {n} tokens: ours {:.1} ms, ggml {:.1} ms (with setup)",
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
            for (i, what) in ["quantized", "product"].iter().enumerate() {
                let diff = ours[i].iter().zip(&ggml[i]).filter(|(a, b)| a != b).count();
                assert_eq!(diff, 0, "{h} x {d}, {n} tokens: {diff} {what} bytes moved");
            }
        }
    }
}
