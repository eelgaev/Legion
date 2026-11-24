//! [`gated_codes`] (`--dense-fp8-prompt`, #1183) against what it replaces: the attention op's
//! output gated and quantized by [`crate::model::fp8_act::gate`], codes and scales bit for bit
//! (on the CPU; on CUDA, ignored tests, the kernels of one GPU against each other).

use super::tests::{cpu, f16_bytes, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::fp8_act;

/// A prompt step of one sequence: `tokens` tokens from position 0, attending to a sparse,
/// causal set of positions each.
struct Shape {
    d: i64,
    n_head: i64,
    n_head_kv: i64,
    tokens: i64,
}

/// Per token: the positions it attends to, ascending (itself always).
fn picks(tokens: usize) -> Vec<Vec<i32>> {
    (0..tokens)
        .map(|t| {
            (0..=t)
                .filter(|&p| p == t || (p * 7 + t) % 3 != 0)
                .map(|p| p as i32)
                .collect()
        })
        .collect()
}

/// The packed activations of the reference (`fp8_act::gate` of the attention output) and of
/// [`gated_codes`], computed on `backend`.
fn run(s: &Shape, backend: &mut Backend) -> (Vec<u8>, Vec<u8>) {
    let (d, nh, nkv, n) = (s.d, s.n_head, s.n_head_kv, s.tokens);
    let k = d * nh;
    let blocks = (n as usize).div_ceil(KV_BLOCK);
    let cells = (blocks * KV_BLOCK) as i64;
    let ctx = Context::new(64, 64);
    let input = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input();
    // The joined projection's rows: [q_h | gate_h] per head, as attention_res views them.
    let qg = input(types::F32, &[2 * k, n]);
    let nb = [(2 * d) as usize * 4, qg.nb()[1]];
    let q = ctx.view_3d(qg, [d, nh, n], nb, 0);
    let gate = ctx.view_3d(qg, [d, nh, n], nb, d as usize * 4);
    let kp = input(types::F16, &[d * nkv, cells]);
    let vp = input(types::F16, &[d * nkv, cells]);
    let table = input(types::I32, &[blocks as i64, 1]);
    let sel = input(types::I32, &[1 + n, n]);
    let att = attention(&ctx, q, kp, vp, table, sel);
    let o = ctx.reshape_2d(att, k, n);
    let want = fp8_act::gate(&ctx, gate, o, false).as_output();
    let got = gated_codes(&ctx, o, gate).expect("a prompt step's attention output");
    // SAFETY: a view of the op's node.
    let node = ctx.wrap(unsafe { (*got.raw()).view_src }).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(want);
    graph.expand(node);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();

    let gates: Vec<f32> = noise((2 * k * n) as usize, 11);
    // q then 4x the gate's spread per head (sigmoids away from 1/2).
    let qgd: Vec<f32> = gates
        .iter()
        .enumerate()
        .map(|(i, &x)| if (i as i64 / d) % 2 == 0 { x } else { 4.0 * x })
        .collect();
    qg.set_f32(&qgd);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 12)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 13)));
    // Blocks out of order.
    let tab: Vec<i32> = (0..blocks as i32).rev().collect();
    table.set_i32(&tab);
    let mut rows = vec![0i32; ((1 + n) * n) as usize];
    for (t, p) in picks(n as usize).iter().enumerate() {
        let row = &mut rows[t * (1 + n as usize)..];
        row[0] = p.len() as i32;
        row[1..=p.len()].copy_from_slice(p);
    }
    sel.set_i32(&rows);
    backend.compute(&graph).unwrap();
    (want.bytes(), got.bytes())
}

/// Where the packed activations of two runs first differ, as (token, column) or the scale's
/// token.
fn first_diff(a: &[u8], b: &[u8], k: usize, n: usize) -> Option<String> {
    let i = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some(if i < k * n {
        format!("code of token {}, column {}", i / k, i % k)
    } else {
        format!("scale of token {}", (i - k * n) / 4)
    })
}

#[test]
fn cpu_gated_codes_match_attention_then_gate() {
    let mut backend = cpu();
    for s in [
        Shape {
            d: 64,
            n_head: 8,
            n_head_kv: 2,
            tokens: 80,
        },
        Shape {
            d: 32,
            n_head: 6,
            n_head_kv: 3,
            tokens: 130,
        },
    ] {
        let (want, got) = run(&s, &mut backend);
        assert_eq!(want.len(), got.len());
        let (k, n) = ((s.d * s.n_head) as usize, s.tokens as usize);
        assert_eq!(first_diff(&want, &got, k, n), None);
    }
}

#[test]
fn gated_codes_take_prompt_attention_only() {
    let ctx = Context::new(32, 32);
    let (d, nh, n) = (32i64, 4i64, 80i64);
    let qg = ctx.new_tensor(types::F32, &[2 * d * nh, n]);
    let nb = [(2 * d) as usize * 4, qg.nb()[1]];
    let gate = ctx.view_3d(qg, [d, nh, n], nb, d as usize * 4);
    // Not an attention output.
    let x = ctx.new_tensor(types::F32, &[d * nh, n]);
    assert!(gated_codes(&ctx, x, gate).is_none());
    // A decode step's (the parts behind the output).
    let q = ctx.view_3d(qg, [d, nh, 8], nb, 0);
    let kv = ctx.new_tensor(types::F16, &[d, 64]);
    let table = ctx.new_tensor(types::I32, &[1, 1]);
    let sel = ctx.new_tensor(types::I32, &[9, 8]);
    let att = attention(&ctx, q, kv, kv, table, sel);
    let gate8 = ctx.view_3d(qg, [d, nh, 8], nb, d as usize * 4);
    let o = ctx.reshape_2d(att, d * nh, 8);
    assert!(gated_codes(&ctx, o, gate8).is_none());
}

/// On a GPU, `gated_codes` (on a GB10 `attn_gw_kernel`'s codes epilogue, elsewhere the
/// attention kernel and then the gate's) against the attention kernel and `fp8_act`'s gate
/// kernel: qwen4exp's shape (head size 256, 24 heads over 2 kv heads), a token count that is
/// not a multiple of anything, and head size 128 with 8 heads per kv head.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gated_codes_match_attention_then_gate() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for s in [
        Shape {
            d: 256,
            n_head: 24,
            n_head_kv: 2,
            tokens: 777,
        },
        Shape {
            d: 128,
            n_head: 16,
            n_head_kv: 2,
            tokens: 300,
        },
    ] {
        let (want, got) = run(&s, &mut backend);
        let (k, n) = ((s.d * s.n_head) as usize, s.tokens as usize);
        assert_eq!(first_diff(&want, &got, k, n), None, "head size {}", s.d);
    }
}
