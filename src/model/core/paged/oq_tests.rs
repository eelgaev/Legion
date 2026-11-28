//! The attention output quantized for a Q4_0 prompt matmul ([`attention_quant`], #1161) against
//! the matmul's own quantization of the F32 output ([`crate::model::q40_mm::quantize`]): the
//! same bytes in every column, on the CPU and (ignored tests) on a GPU, where the prompt kernels
//! write them from their output fragments and every other kernel leaves them to the op.

use super::tests::{f16_bytes, hparams, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvPool;
use crate::model::q40_mm::{get_col, n_padded, quant_bytes, quantize};

/// One group of `n_tok` tokens per sequence (at `n_past` each, over shuffled blocks) in one
/// graph on `backend`, as the plain node and as an [`attention_quant`] node: (the plain node's
/// F32 output, its quantization by [`quantize`], the quant node's bytes).
fn plain_and_quant(
    backend: &mut Backend,
    (d, n_head, n_head_kv): (usize, usize, usize),
    n_tok: usize,
    n_past: &[usize],
    window: usize,
) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let ns = n_past.len();
    let used: Vec<usize> = n_past
        .iter()
        .map(|p| (p + n_tok).div_ceil(KV_BLOCK))
        .collect();
    let width = *used.iter().max().unwrap();
    let n_blocks = width + 1;
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let (kp, vp) = kv.pools(0);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 21)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 22)));
    // Sequence s's block i in pool block (i * 7 + 3 + s) mod n_blocks.
    let mut table = Vec::new();
    for (s, &u) in used.iter().enumerate() {
        table.extend((0..u).map(|i| ((i * 7 + 3 + s) % n_blocks) as i32));
        table.resize((s + 1) * width, 0);
    }
    let scale = 1.0 / (d as f32).sqrt();
    let (k, n) = ((d * n_head) as i64, (n_tok * ns) as i64);

    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let attn = || attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0);
    let plain = attn().as_output();
    let want = quantize(&ctx, ctx.reshape_2d(plain, k, n)).as_output();
    let got = attention_quant(&ctx, ctx.reshape_2d(attn(), k, n))
        .expect("a plain node of the whole op")
        .as_output();
    assert_eq!(got.ty(), types::I8);
    assert_eq!(
        got.ne()[0] as usize,
        quant_bytes(k, n) + (k * n) as usize * 4
    );
    let mut graph = ctx.new_graph(64);
    graph.expand(plain);
    graph.expand(want);
    graph.expand(got);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Larger than unit queries, so the softmax is far from uniform.
    let qv: Vec<f32> = noise(q.elements(), 23).iter().map(|x| 3.0 * x).collect();
    q.set_f32(&qv);
    tab.set_i32(&table);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();
    (plain.to_f32(), want.bytes(), got.bytes())
}

/// The first column of `k` values (of `n`) whose quants or scales differ between `want` and
/// `got`, bit for bit.
fn first_diff(want: &[u8], got: &[u8], k: usize, n: usize) -> Option<usize> {
    let n_pad = n_padded(n as i64) as usize;
    (0..n).find(|&c| {
        let (qa, da) = get_col(want, k, n_pad, c);
        let (qb, db) = get_col(got, k, n_pad, c);
        qa != qb || da.iter().zip(&db).any(|(x, y)| x.to_bits() != y.to_bits())
    })
}

fn check(
    backend: &mut Backend,
    shape: (usize, usize, usize),
    n_tok: usize,
    n_past: &[usize],
    window: usize,
    f32_too: bool,
) {
    let (plain, want, got) = plain_and_quant(backend, shape, n_tok, n_past, window);
    let (k, n) = (shape.0 * shape.1, n_tok * n_past.len());
    let qb = quant_bytes(k as i64, n as i64);
    let label = format!("{shape:?}, {n_tok} tokens at {n_past:?}, window {window}");
    if let Some(c) = first_diff(&want, &got[..qb], k, n) {
        panic!("{label}: column {c} differs");
    }
    if f32_too {
        // The CPU writes the F32 output too, after the quantization's bytes.
        let f: Vec<u32> = got[qb..]
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert!(
            f.iter().zip(&plain).all(|(a, b)| *a == b.to_bits()),
            "{label}: F32 output"
        );
    }
}

/// On the CPU: the bytes of the plain node's output quantized, and that output after them, at
/// one and two sequences, a window, a part tile's edges.
#[test]
fn cpu_attention_quant_is_the_quantized_output() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    check(&mut backend, (64, 8, 2), 70, &[0], 0, true);
    check(&mut backend, (64, 8, 2), 37, &[130, 5], 64, true);
    check(&mut backend, (32, 16, 2), 129, &[3], 0, true);
}

/// Only the reshape of one plain node of the whole op: not a part, not another op.
#[test]
fn attention_quant_takes_one_plain_node() {
    let backend = Backend::cpu().unwrap();
    let hp = hparams(64, 8, 2);
    let kv = KvPool::new(&hp, backend.buffer_type(), 4).unwrap();
    let (kp, vp) = kv.pools(0);
    let ctx = Context::new(64, 64);
    let q = ctx.new_tensor(types::F32, &[64, 8, 40, 1]);
    let tab = ctx.new_tensor(types::I32, &[4, 1]);
    let past = ctx.new_tensor(types::I32, &[1]);
    let scratch = ctx.new_tensor(types::F32, &scratch_shape(64, 8, 2, 40, 1));
    let plain = attention_with(&ctx, q, kp, vp, tab, past, scratch, 0.125, 0, 0);
    assert!(attention_quant(&ctx, ctx.reshape_2d(plain, 512, 40)).is_some());
    let [first, second] = attention_parts(&ctx, q, kp, vp, tab, past, scratch, 0.125, 16);
    assert!(attention_quant(&ctx, ctx.reshape_2d(first, 512, 40)).is_none());
    assert!(attention_quant(&ctx, ctx.reshape_2d(second, 512, 40)).is_none());
    // Not the reshape of the node: a copy of it, or the node itself.
    let other = ctx.cont(ctx.reshape_2d(plain, 512, 40));
    assert!(attention_quant(&ctx, other).is_none());
    assert!(attention_quant(&ctx, plain).is_none());
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// [`check`] on the GPU, whose launch must write the quantization itself (`fused`, on sm_80+)
/// or leave it to the op.
#[cfg(feature = "cuda")]
fn check_gpu(
    backend: &mut Backend,
    shape: (usize, usize, usize),
    n_tok: usize,
    n_past: &[usize],
    window: usize,
    fused: bool,
) {
    let before = oq_launches();
    check(backend, shape, n_tok, n_past, window, false);
    let wrote = oq_launches() - before;
    assert_eq!(
        wrote,
        i64::from(fused && quant_ok()),
        "{shape:?}, {n_tok} tokens at {n_past:?}: launches that wrote the quantization"
    );
}

/// Gemma 4 31B's attention (sliding layers: heads of 256, 2 query heads a kv head, a 1024-token
/// window, paged_prefill_qm on a GB10; global layers: heads of 512, 8 query heads a kv head,
/// paged_prefill_mma) in prompt steps of 512 to 4097 tokens, one and two sequences, from
/// position 0 and later: the quantization the kernels write is the op's of their F32 output, bit
/// for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_attention_quant_is_byte_exact() {
    let mut backend = gpu();
    let before = qm_launches();
    for (shape, window) in [((256, 32, 16), 1024), ((512, 32, 4), 0)] {
        check_gpu(&mut backend, shape, 512, &[0], window, true);
        check_gpu(&mut backend, shape, 1000, &[300, 0], window, true);
        check_gpu(&mut backend, shape, 2048, &[1536], window, true);
        check_gpu(&mut backend, shape, 4097, &[0], window, true);
    }
    println!("paged_prefill_qm launches: {}", qm_launches() - before);
}

/// The same on paged_prefill_mma at heads of 256 (the queries-as-M kernel off for this thread),
/// and with the FP32 FMA kernel for this thread, which leaves the quantization to the op.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_attention_quant_on_other_kernels() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_qm_prefill_here(None);
            set_ffma_prefill_here(false);
        }
    }
    let _reset = Reset;
    let mut backend = gpu();
    set_qm_prefill_here(Some(false));
    let before = qm_launches();
    check_gpu(&mut backend, (256, 32, 16), 1000, &[300, 0], 1024, true);
    check_gpu(&mut backend, (256, 32, 16), 4097, &[0], 1024, true);
    assert_eq!(qm_launches(), before, "paged_prefill_qm ran");
    set_ffma_prefill_here(true);
    let before = ffma_launches();
    check_gpu(&mut backend, (256, 32, 16), 1000, &[300, 0], 1024, false);
    check_gpu(&mut backend, (512, 32, 4), 600, &[0], 0, false);
    assert!(ffma_launches() > before, "the FFMA kernel did not run");
}
