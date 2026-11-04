//! The small-step shared quantization: which shapes it takes, which node quantizes, the CPU
//! reference against the prompt path's, and (ignored, needs a GPU) every projection bit for bit
//! against ggml's `mul_mat` on CUDA.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};
use crate::model::core::shared_q8;

#[test]
fn input_fits_takes_9_to_255_tokens_of_rows_of_512() {
    let ctx = Context::new(16, 8);
    for (k, n, ok) in [
        (2560, 9, true),
        (2560, 64, true),
        (2560, 255, true),
        (512, 16, true),
        (2560, 8, false),
        (2560, 256, false),
        (384, 16, false),
        (2048 + 64, 16, false),
    ] {
        let x = ctx.new_tensor(types::F32, &[k, n]);
        assert_eq!(input_fits(x), ok, "{k} values, {n} tokens");
    }
    // The widest input's blocks must fit the scratch buffer.
    assert!(!input_fits(ctx.new_tensor(types::F32, &[512 * 64, 255])));
    // Other types, strided rows, more dimensions: not taken.
    assert!(!input_fits(ctx.new_tensor(types::F16, &[2560, 16])));
    let wide = ctx.new_tensor(types::F32, &[4096, 16]);
    assert!(!input_fits(ctx.view_2d(wide, [2560, 16], 4096 * 4, 0)));
    assert!(!input_fits(ctx.new_tensor(types::F32, &[2560, 4, 4])));
}

#[test]
fn weight_fits_takes_plain_q8_0_of_the_row_length() {
    let ctx = Context::new(8, 8);
    let w = |t, ne: &[i64]| ctx.new_tensor(t, ne);
    assert!(weight_fits(w(types::Q8_0, &[2560, 6144]), 2560));
    assert!(!weight_fits(w(types::Q8_0, &[2560, 6144]), 5120));
    assert!(!weight_fits(w(types::Q4_K, &[2560, 6144]), 2560));
    assert!(!weight_fits(w(types::F32, &[2560, 64]), 2560));
    assert!(!weight_fits(w(types::Q8_0, &[2560, 64, 2]), 2560));
}

/// Where the Q8_0 MMQ op does not take the matmul (here: no CUDA build or GPU), the op takes the
/// weights [`weight_fits`] takes for the input's row length.
#[test]
fn takes_weight_is_weight_fits_without_q8_mmq() {
    let ctx = Context::new(8, 8);
    let x = ctx.new_tensor(types::F32, &[2560, 32]);
    let w = |t, ne: &[i64]| ctx.new_tensor(t, ne);
    assert!(takes_weight(w(types::Q8_0, &[2560, 6144]), x));
    assert!(!takes_weight(w(types::Q8_0, &[5120, 6144]), x));
    assert!(!takes_weight(w(types::Q4_K, &[2560, 6144]), x));
}

#[test]
fn takes_input_needs_the_gpu() {
    // On no GPU (the default until the engine sets one) nothing is taken.
    let ctx = Context::new(8, 8);
    assert!(!takes_input(ctx.new_tensor(types::F32, &[2560, 32])));
}

/// The first node of a run reading one input quantizes; the next ones reuse its blocks, also
/// with other ops between them; a node after one of another input quantizes again.
#[test]
fn the_first_node_of_each_input_quantizes() {
    let ctx = Context::new(64, 64);
    let (k, n) = (512i64, 16i64);
    let x1 = ctx.new_tensor(types::F32, &[k, n]);
    let x2 = ctx.new_tensor(types::F32, &[k, n]);
    let w = |r| ctx.new_tensor(types::Q8_0, &[k, r]);
    let a = mul_mat(&ctx, w(128), x1);
    let b = mul_mat(&ctx, w(64), x1);
    let other = ctx.scale(b, 2.0);
    let c = mul_mat(&ctx, w(128), x1);
    let d = mul_mat(&ctx, w(128), x2);
    let e = mul_mat(&ctx, w(128), x1);
    let f = mul_mat(&ctx, w(64), x1);
    let mut graph = ctx.new_graph(64);
    for t in [a, other, c, d, e, f] {
        graph.expand(t);
    }
    // SAFETY: the graph's node array holds `n` valid tensor pointers.
    let nodes =
        unsafe { std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph.raw()), graph.nodes()) };
    let at = |t: Tensor<'_>| nodes.iter().position(|&p| p == t.raw()).unwrap();
    // SAFETY: indices of the graph's nodes.
    let reused = |t| unsafe { reuses(nodes, at(t), &SHARED_Q8_SMALL) };
    assert!(at(a) < at(b) && at(b) < at(other) && at(other) < at(c));
    assert!(!reused(a), "the graph's first");
    assert!(reused(b));
    assert!(reused(c), "a scale between");
    assert!(!reused(d), "another input");
    assert!(!reused(e), "after another input");
    assert!(reused(f));
    // A node of another op reading the same input is not one of these.
    // SAFETY: as above.
    assert!(!unsafe { reuses(nodes, at(other), &SHARED_Q8_SMALL) });
}

/// The test input: noise with an all-zero group.
fn input(k: i64, n: i64) -> Vec<f32> {
    let mut v = noise((k * n) as usize, 77);
    v[96..128].fill(0.0);
    v
}

/// `x` (F32 `[k, n]`) times Q8_0 weights of `rows` rows each on `backend`: `ours` this op for
/// each, `prompt` the prompt path's node of blocks and its matmuls, neither ggml's `mul_mat`s.
fn case(
    backend: &mut Backend,
    ours: bool,
    prompt: bool,
    k: i64,
    rows: &[i64],
    n: i64,
) -> Vec<Vec<u8>> {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = rows
        .iter()
        .map(|&r| ctx.new_tensor(types::Q8_0, &[k, r]).as_input())
        .collect();
    let outs: Vec<_> = if ours {
        ws.iter().map(|&w| mul_mat(&ctx, w, x)).collect()
    } else if prompt {
        let q = shared_q8::quantize(&ctx, x);
        ws.iter()
            .map(|&w| crate::model::core::mmq_preq::mul_mat(&ctx, w, q.xq.unwrap(), x))
            .collect()
    } else {
        ws.iter().map(|&w| ctx.mul_mat(w, x)).collect()
    };
    let outs: Vec<_> = outs.into_iter().map(|t| t.as_output()).collect();
    let mut graph = ctx.new_graph(32);
    for &t in &outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&input(k, n));
    for (i, (&w, &r)) in ws.iter().zip(rows).enumerate() {
        w.set_bytes(&q8_0_matrix(k as usize, r as usize, 51 + i as u64));
    }
    backend.compute(&graph).unwrap();
    outs.iter().map(|t| t.bytes()).collect()
}

/// On the CPU each product is the prompt path's reference, bit for bit.
#[test]
fn cpu_products_are_the_prompt_paths() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(3);
    let rows = [128i64, 64, 2];
    let ours = case(&mut cpu, true, false, 1024, &rows, 12);
    let prompt = case(&mut cpu, false, true, 1024, &rows, 12);
    assert_eq!(ours, prompt);
}

/// Every projection from one quantization is ggml's `mul_mat`, bit for bit, at qwen4exp's
/// shapes (2560 values a token: a DeltaNet layer's qkv and gate, an attention layer's Q and K)
/// from 9 to 255 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_shared_q8_small_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let rows = [12288i64, 6144, 10240, 512];
    for n in [9, 16, 33, 64, 128, 255] {
        let ours = case(&mut gpu, true, false, 2560, &rows, n);
        let ggml = case(&mut gpu, false, false, 2560, &rows, n);
        for (i, r) in rows.iter().enumerate() {
            let diff = ours[i].iter().zip(&ggml[i]).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "{r} rows, {n} tokens: {diff} product bytes moved");
        }
    }
}
