//! The paged attention op over quantized caches (q8_0, q4_0, K and V separately): the CUDA
//! kernels (decode and the sm_80+ prefill) against the f64 oracle over the dequantized cache,
//! and the CPU op the same.

use super::tests::{Shape, noise, oracle};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{KV_BLOCK, KvType};

/// `d`-wide heads, one sequence at `n_past` with `n_tok` tokens, caches of `kt` / `vt`.
#[allow(clippy::too_many_arguments)]
fn quant_case(
    backend: &mut Backend,
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    window: usize,
    n_past: usize,
    n_tok: usize,
    kt: KvType,
    vt: KvType,
    tol: f32,
) {
    let row = d * n_head_kv;
    let len = n_past + n_tok;
    let n_blocks = len.div_ceil(KV_BLOCK) + 1;
    let cells = n_blocks * KV_BLOCK;
    // Blocks reversed: sequence block i in pool block n_blocks - 1 - i.
    let table: Vec<u32> = (0..len.div_ceil(KV_BLOCK))
        .map(|i| (n_blocks - 1 - i) as u32)
        .collect();
    let kd = noise(cells * row, 41);
    let vd = noise(cells * row, 43);
    let ctx = Context::new(64, 64);
    let input = |ne: &[i64], ty| ctx.new_tensor(ty, ne).as_input();
    let q = input(&[d as i64, n_head as i64, n_tok as i64, 1], types::F32);
    let kp = input(&[row as i64, cells as i64], kt.ggml());
    let vp = input(&[row as i64, cells as i64], vt.ggml());
    let tab = input(&[table.len() as i64, 1], types::I32);
    let past = input(&[1], types::I32);
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, 1);
    let scratch = input(&scratch_ne, types::F32);
    let scale = 1.0 / (d as f32).sqrt();
    let out = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let store = |t: crate::ggml::Tensor<'_>, ty: KvType, x: &[f32]| -> Vec<f32> {
        if ty == KvType::F16 {
            t.set_bytes(&super::tests::f16_bytes(x));
            let mut y = x.to_vec();
            round_f16(&mut y);
            y
        } else {
            let raw = crate::ggml::quantize(ty.ggml(), x, row);
            t.set_bytes(&raw);
            crate::ggml::dequantize(ty.ggml(), &raw, x.len())
        }
    };
    let k = store(kp, kt, &kd);
    let v = store(vp, vt, &vd);
    let qv: Vec<f32> = noise(q.elements(), 47).iter().map(|x| 2.0 * x).collect();
    q.set_f32(&qv);
    tab.set_i32(&table.iter().map(|&b| b as i32).collect::<Vec<_>>());
    past.set_i32(&[n_past as i32]);
    backend.compute(&graph).unwrap();
    let sh = Shape {
        d,
        n_head,
        n_head_kv,
        n_tok,
        scale: scale as f64,
        window,
    };
    // The prefill kernel rounds dequantized values to F16 for its MMAs; the oracle takes them
    // as they are, so the tolerance carries one F16 rounding of K and V too.
    let want = oracle(&sh, &qv, &k, &v, &[n_past], std::slice::from_ref(&table));
    let got = out.to_f32();
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < tol,
        "{kt:?}/{vt:?}, d {d}, heads {n_head}/{n_head_kv}, window {window}, n_past {n_past}, \
         {n_tok} tokens: max |diff| {diff}"
    );
}

fn cases(backend: &mut Backend, tol: f32) {
    use KvType::*;
    for (kt, vt) in [(Q8_0, Q8_0), (Q8_0, Q4_0), (Q4_0, Q4_0), (F16, Q8_0)] {
        for (d, n_head, n_head_kv, window) in [(128, 8, 2, 0), (256, 16, 8, 100), (512, 16, 2, 0)] {
            quant_case(backend, d, n_head, n_head_kv, window, 300, 37, kt, vt, tol);
            quant_case(backend, d, n_head, n_head_kv, window, 700, 1, kt, vt, tol);
        }
    }
}

#[test]
fn cpu_op_over_quantized_caches_matches_oracle() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    cases(&mut backend, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_kernels_over_quantized_caches_match_oracle() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    // Production only sends quantized caches to the kernels where `quant_ok()` (the sm_80+
    // prefill kernel), so there is nothing to check elsewhere.
    if !quant_ok() {
        eprintln!("skipped: the quantized-cache paged kernels need sm_80+ on this device");
        return;
    }
    // The prefill kernel's F16 weights against the running max and F16 K/V (see tests.rs).
    cases(&mut backend, 1e-3);
}

#[test]
fn mtp_head_keeps_its_types_but_where_the_target_pages_and_it_cannot() {
    use crate::model::core::KvTypes;
    let q8 = KvTypes {
        k: KvType::Q8_0,
        v: KvType::Q8_0,
    };
    // On the CPU and beside dense attention (which gathers then too): the target's types.
    assert_eq!(mtp_head_types(q8, false, true), q8);
    assert_eq!(mtp_head_types(q8, true, false), q8);
    // F16 stays F16.
    assert_eq!(
        mtp_head_types(KvTypes::default(), true, true),
        KvTypes::default()
    );
    // On a GPU next to sparse attention: what the paged kernels read on it, else F16.
    let want = if reads_types(q8) {
        q8
    } else {
        KvTypes::default()
    };
    assert_eq!(mtp_head_types(q8, true, true), want);
    let q5 = KvTypes {
        k: KvType::Q5_1,
        v: KvType::Q5_1,
    };
    assert_eq!(mtp_head_types(q5, true, true), KvTypes::default());
}
