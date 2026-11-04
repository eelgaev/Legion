//! The shared quantization: which inputs it takes, the CPU reference, and (ignored, needs a GPU)
//! every projection's product bit for bit against ggml's `mul_mat` on CUDA.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn input_fits_takes_contiguous_rows_of_512() {
    let ctx = Context::new(8, 8);
    for (k, ok) in [
        (2560, true),
        (512, true),
        (6144, true),
        (384, false),
        (2048 + 64, false),
    ] {
        let x = ctx.new_tensor(types::F32, &[k, 300]);
        assert_eq!(input_fits(x), ok, "{k} values a token");
    }
    // Other types, strided rows, more dimensions: not taken.
    assert!(!input_fits(ctx.new_tensor(types::F16, &[2560, 300])));
    let wide = ctx.new_tensor(types::F32, &[4096, 300]);
    assert!(!input_fits(ctx.view_2d(wide, [2560, 300], 4096 * 4, 0)));
    assert!(!input_fits(ctx.new_tensor(types::F32, &[2560, 30, 10])));
}

#[test]
fn takes_needs_the_gpu_and_the_row_length() {
    // On no GPU (the default until the engine sets one) no weight is taken.
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[2560, 4096]);
    let q = quantize(&ctx, x);
    assert_eq!(q.n(), 4096);
    assert!(!q.takes(ctx.new_tensor(types::Q8_0, &[2560, 6144])));
    assert!(!q.takes(ctx.new_tensor(types::Q8_0, &[6144, 2560])));
}

/// The test input: noise with an all-zero group and tiny values (flushed to zero on the GPU).
fn input(k: i64, n: i64) -> Vec<f32> {
    let mut v = noise((k * n) as usize, 91);
    v[64..96].fill(0.0);
    v[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    v
}

/// `x` (F32 `[k, n]`) times Q8_0 weights of `rows` rows each on `backend`: with `ours`, one
/// [`quantize`] and [`Shared::mul_mat`] for every weight; else ggml's `mul_mat`s. The products'
/// bytes, then (`ours`) the blocks' without MMQ's padding.
fn case(backend: &mut Backend, ours: bool, k: i64, rows: &[i64], n: i64) -> Vec<Vec<u8>> {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = rows
        .iter()
        .map(|&r| ctx.new_tensor(types::Q8_0, &[k, r]).as_input())
        .collect();
    let mut outs: Vec<_> = if ours {
        let q = quantize(&ctx, x);
        let mut y: Vec<_> = ws.iter().map(|&w| q.mul_mat(&ctx, w)).collect();
        y.push(q.xq.expect("the prompt path's node"));
        y
    } else {
        ws.iter().map(|&w| ctx.mul_mat(w, x)).collect()
    };
    outs = outs.into_iter().map(|t| t.as_output()).collect();
    let mut graph = ctx.new_graph(32);
    for &t in &outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&input(k, n));
    for (i, (&w, &r)) in ws.iter().zip(rows).enumerate() {
        w.set_bytes(&q8_0_matrix(k as usize, r as usize, 93 + i as u64));
    }
    backend.compute(&graph).unwrap();
    let data = (k / mmq_preq::BLOCK * n) as usize * mmq_preq::BLOCK_BYTES;
    let mut bytes: Vec<_> = outs.iter().map(|t| t.bytes()).collect();
    if ours {
        bytes.last_mut().unwrap().truncate(data);
    }
    bytes
}

/// On the CPU the op's blocks are the reference quantization of `x`.
#[test]
fn cpu_shared_quantize_is_the_reference() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let (k, n) = (1024i64, 5i64);
    let ours = case(&mut cpu, true, k, &[128], n);
    let mut want = vec![0u8; mmq_preq::q8_1_bytes(k, n)];
    mmq_preq::quantize_ref(&input(k, n), k as usize, n as usize, &mut want);
    let blocks = ours.last().unwrap();
    assert!(blocks[..] == want[..blocks.len()], "the blocks moved");
}

/// Every projection's product from the shared blocks is ggml's `mul_mat`, bit for bit, at
/// qwen4exp's shapes (2560 values a token; a DeltaNet layer's qkv and gate, an attention layer's
/// K, a one-row gate) from 256 tokens up; and each path's time.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_shared_q8_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let rows = [10240i64, 6144, 512, 1];
    for n in [256, 300, 1000, 4096] {
        let t0 = std::time::Instant::now();
        let ours = case(&mut gpu, true, 2560, &rows, n);
        let t1 = std::time::Instant::now();
        let ggml = case(&mut gpu, false, 2560, &rows, n);
        println!(
            "{n} tokens: ours {:.1} ms, ggml {:.1} ms (with setup)",
            (t1 - t0).as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
        for (i, r) in rows.iter().enumerate() {
            let diff = ours[i].iter().zip(&ggml[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "{r} rows, {n} tokens: {diff} product bytes moved");
        }
    }
}
