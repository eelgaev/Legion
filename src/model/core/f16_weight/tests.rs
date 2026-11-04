//! The F16 weight copy: when a matmul takes it ([`crate::model::core::f16_joined`]), the CPU
//! reference, and (ignored, needs a GPU) the joined matmul against ggml's `mul_mat` bit for bit.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::f16_joined;
use crate::model::core::mmq_preq::tests::q8_0_matrix;

#[test]
fn joined_needs_a_registered_f16_input_and_a_q8_0_weight() {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 6144]);
    let x = ctx.new_tensor(types::F32, &[2560, 64]);
    // No F16 copy of x: ggml's own conversion.
    assert!(f16_joined(&ctx, w, x).is_none());
    let x16 = ctx.new_tensor(types::F16, &[2560, 64]);
    ctx.set_shared_cast(x, types::F16, x16);
    let (w16, got) = f16_joined(&ctx, w, x).expect("joined");
    assert_eq!(got.raw(), x16.raw());
    assert_eq!((w16.ty(), w16.ne()), (types::F16, [2560, 6144, 1, 1]));
    // Other weight types, and fewer tokens than ggml-cuda's cuBLAS threshold: not taken.
    for ty in [types::F32, types::F16, types::BF16, types::Q4_K] {
        assert!(f16_joined(&ctx, ctx.new_tensor(ty, &[2560, 512]), x).is_none());
    }
    let few = ctx.new_tensor(types::F32, &[2560, 63]);
    ctx.set_shared_cast(few, types::F16, ctx.new_tensor(types::F16, &[2560, 63]));
    assert!(f16_joined(&ctx, w, few).is_none());
    // Stacked (expert) weights: not taken.
    let stacked = ctx.new_tensor(types::Q8_0, &[2560, 512, 4]);
    assert!(f16_joined(&ctx, stacked, x).is_none());
}

/// On the CPU the op is each block's `d · q` rounded to F16.
#[test]
fn cpu_f16_weight_is_d_times_q() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let (k, m) = (96usize, 5usize);
    let bytes = q8_0_matrix(k, m, 41);
    let ctx = Context::new(8, 8);
    let w = ctx
        .new_tensor(types::Q8_0, &[k as i64, m as i64])
        .as_input();
    let y = dequant(&ctx, w).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    w.set_bytes(&bytes);
    cpu.compute(&graph).unwrap();
    let mut want = Vec::with_capacity(k * m * 2);
    for block in bytes.chunks(34) {
        // SAFETY: plain conversions.
        let d = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([block[0], block[1]])) };
        for &q in &block[2..] {
            let h = unsafe { ffi::ggml_fp32_to_fp16(f32::from(q as i8) * d) };
            want.extend_from_slice(&h.to_le_bytes());
        }
    }
    assert!(y.bytes() == want, "the F16 values moved");
}

/// `x` (F32 `[k, n]`) times Q8_0 weights of `rows` rows each on `backend`: with `ours`, every
/// matmul from the F16 weight copy and one F16 cast of `x` ([`f16_joined`]); else ggml's
/// `mul_mat`s. The products' bytes.
#[cfg(feature = "cuda")]
fn case(backend: &mut Backend, ours: bool, k: i64, rows: &[i64], n: i64) -> Vec<Vec<u8>> {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = rows
        .iter()
        .map(|&r| ctx.new_tensor(types::Q8_0, &[k, r]).as_input())
        .collect();
    if ours {
        ctx.set_shared_cast(x, types::F16, ctx.cast(x, types::F16));
    }
    let outs: Vec<_> = ws
        .iter()
        .map(|&w| {
            let y = match f16_joined(&ctx, w, x).filter(|_| ours) {
                Some((w16, x16)) => ctx.mul_mat(w16, x16),
                None => ctx.mul_mat(w, x),
            };
            y.as_output()
        })
        .collect();
    let mut graph = ctx.new_graph(32);
    for &t in &outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let mut v = crate::model::core::mmq_preq::tests::noise((k * n) as usize, 42);
    v[64..96].fill(0.0);
    v[160..192].iter_mut().for_each(|v| *v *= 1e-38);
    v[256..288].iter_mut().for_each(|v| *v *= 1e5);
    x.set_f32(&v);
    for (i, (&w, &r)) in ws.iter().zip(rows).enumerate() {
        w.set_bytes(&q8_0_matrix(k as usize, r as usize, 43 + i as u64));
    }
    backend.compute(&graph).unwrap();
    outs.iter().map(|t| t.bytes()).collect()
}

/// On a V100 the joined matmul is ggml's `mul_mat`, bit for bit, at qwen4exp's shapes (2560
/// values a token; DeltaNet qkv and gate, an attention layer's K, the shared expert's gate) from
/// 64 tokens up (below, ggml runs MMQ and the join is not taken); and each path's time.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_f16_joined_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if crate::residency::cuda::caps(0).map(|c| (c.cc_major, c.cc_minor)) != Some((7, 0)) {
        println!("not a V100: ggml runs these matmuls as MMQ there, not compared");
        return;
    }
    let rows = [12288i64, 6144, 512, 1024];
    for n in [64, 65, 300, 1000, 4096] {
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
