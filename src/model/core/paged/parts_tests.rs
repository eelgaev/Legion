//! A prompt group's attention in two nodes ([`attention_parts`], #1133) against the one node:
//! the same bytes in every row, on the CPU and (ignored tests) on a GPU, where the two parts are
//! two launches of the one launch's kernel cut at a multiple of its tokens per CTA ([`cut`]).

use super::tests::{f16_bytes, hparams, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvPool;

/// The cut (`launch_part`'s rule) is the first multiple of the kernel's tokens per CTA from `n0` on, at most the group.
#[test]
fn cut_rounds_up_to_the_kernels_tile() {
    // paged_prefill_mma at 6 query heads a kv head: 10 tokens a CTA; paged_prefill_qm: 21.
    assert_eq!(cut(8192, 10, 4096), 4100);
    assert_eq!(cut(8192, 21, 4096), 4116);
    assert_eq!(cut(8192, 10, 4100), 4100);
    assert_eq!(cut(8192, 21, 4095), 4095);
    assert_eq!(cut(8192, 21, 4094), 4095);
    assert_eq!(cut(16384, 21, 8192), 8211);
    // Past the group: the first part is the whole group, the second nothing.
    assert_eq!(cut(25, 21, 13), 21);
    assert_eq!(cut(20, 21, 13), 20);
    assert_eq!(cut(100, 10, 100), 100);
    assert_eq!(cut(100, 10, 0), 0);
    for (n, qt) in [(8192, 10), (8192, 21), (4097, 21), (12345, 5), (64, 64)] {
        for n0 in [0, 1, n / 2, n / 2 + 1, n - 1, n] {
            let c = cut(n, qt, n0);
            assert!(c >= n0.min(n) && c <= n, "{n} / {qt} at {n0}: {c}");
            assert!(c == n || c % qt == 0, "{n} / {qt} at {n0}: {c}");
            assert!(c < n0 + qt, "{n} / {qt} at {n0}: {c}");
        }
    }
}

/// The parameters' slots after every other node's ([`build`]'s 8 at most).
#[test]
fn part_parameters_follow_the_ops_others() {
    assert_eq!((P_CUT, P_PART), (8, 9));
    assert!(P_PART < crate::ggml::EXTERNAL_PARAMS);
}

/// One group of `n_tok` tokens of one sequence at `n_past` over shuffled blocks, the one node
/// and the two parts cut at `n0` in one graph on `backend`: (the one node's output, the two
/// parts' output, the first part's rows before the second part's cut as the first node holds
/// them).
fn whole_and_parts(
    backend: &mut Backend,
    (d, n_head, n_head_kv): (usize, usize, usize),
    n_tok: usize,
    n_past: usize,
    n0: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let n_blocks = (n_past + n_tok).div_ceil(KV_BLOCK) + 1;
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let (kp, vp) = kv.pools(0);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 21)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 22)));
    // Blocks out of order: sequence block i in pool block (i * 7 + 3) mod n_blocks (some shared
    // where 7 divides n_blocks, the same for both).
    let used = (n_past + n_tok).div_ceil(KV_BLOCK);
    let table: Vec<i32> = (0..used).map(|i| ((i * 7 + 3) % n_blocks) as i32).collect();
    let scale = 1.0 / (d as f32).sqrt();

    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(types::F32, &[d as i64, n_head as i64, n_tok as i64, 1])
        .as_input();
    let tab = ctx.new_tensor(types::I32, &[used as i64, 1]).as_input();
    let past = ctx.new_tensor(types::I32, &[1]).as_input();
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, 1);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let whole = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, 0, 0).as_output();
    let [first, second] = attention_parts(&ctx, q, kp, vp, tab, past, scratch, scale, n0 as i64);
    let (first, second) = (first.as_output(), second.as_output());
    let mut graph = ctx.new_graph(64);
    graph.expand(whole);
    graph.expand(first);
    graph.expand(second);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Larger than unit queries, so the softmax is far from uniform.
    let qv: Vec<f32> = noise(q.elements(), 23).iter().map(|x| 3.0 * x).collect();
    q.set_f32(&qv);
    tab.set_i32(&table);
    past.set_i32(&[n_past as i32]);
    backend.compute(&graph).unwrap();
    let row = d * n_head;
    let head = first.to_f32()[..n0.min(n_tok) * row].to_vec();
    (whole.to_f32(), second.to_f32(), head)
}

/// Bit for bit, with the first differing value's row and token.
fn assert_same_bits(label: &str, want: &[f32], got: &[f32], row: usize) {
    assert_eq!(want.len(), got.len(), "{label}: lengths");
    if let Some(i) = (0..want.len()).find(|&i| want[i].to_bits() != got[i].to_bits()) {
        panic!(
            "{label}: token {} value {}: whole {} vs parts {}",
            i / row,
            i % row,
            want[i],
            got[i]
        );
    }
}

fn check(
    backend: &mut Backend,
    shape: (usize, usize, usize),
    n_tok: usize,
    n_past: usize,
    n0: usize,
) {
    let (whole, parts, head) = whole_and_parts(backend, shape, n_tok, n_past, n0);
    let row = shape.0 * shape.1;
    let label = format!("{shape:?}, {n_tok} tokens at {n_past}, cut at {n0}");
    assert_same_bits(&label, &whole, &parts, row);
    assert_same_bits(
        &format!("{label}, first part"),
        &whole[..head.len()],
        &head,
        row,
    );
}

/// On the CPU every row is its own sum: the parts give the one node's bytes at any cut,
/// 6 query heads a kv head, from position 0 and later, cuts off the tiles and at the edges.
#[test]
fn cpu_parts_match_the_one_node() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (n_tok, n_past, n0) in [
        (150, 0, 75),
        (150, 0, 73),
        (131, 70, 64),
        (131, 70, 1),
        (40, 200, 0),
        (40, 200, 40),
    ] {
        check(&mut backend, (32, 12, 2), n_tok, n_past, n0);
    }
    check(&mut backend, (64, 8, 2), 97, 33, 48);
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

/// [`check`] on the GPU, which must run both parts as parts on sm_80+ (two launches more).
#[cfg(feature = "cuda")]
fn check_gpu(
    backend: &mut Backend,
    shape: (usize, usize, usize),
    n_tok: usize,
    n_past: usize,
    n0: usize,
) {
    let before = part_launches();
    check(backend, shape, n_tok, n_past, n0);
    if quant_ok() {
        assert_eq!(
            part_launches() - before,
            2,
            "{shape:?}, {n_tok} tokens at {n_past}: the parts did not run as two launches"
        );
    }
}

/// Qwen3.8-27B's attention (heads of 256, 6 query heads a kv head; a rank's 12 over 2 under
/// `--tp 2`, and all 24 over 4) on the default kernels (paged_prefill_qm on a GB10 at these
/// shapes, 21 tokens a CTA; paged_prefill_mma elsewhere, 10): an 8K prompt step cut at 4,096 (a
/// multiple of neither) and a 4K step at 1,000 positions in, against the one launch, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_parts_match_the_one_launch() {
    let mut backend = gpu();
    let before = qm_launches();
    for shape in [(256, 12, 2), (256, 24, 4)] {
        check_gpu(&mut backend, shape, 8192, 0, 4096);
        check_gpu(&mut backend, shape, 4096, 1000, 2048);
        check_gpu(&mut backend, shape, 3001, 517, 1500);
    }
    println!("paged_prefill_qm launches: {}", qm_launches() - before);
}

/// The same with the queries-as-M kernel off for this thread: paged_prefill_mma (10 tokens a
/// CTA) on every sm_80+ GPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_parts_match_the_one_launch_on_paged_prefill_mma() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_qm_prefill_here(None);
        }
    }
    set_qm_prefill_here(Some(false));
    let _reset = Reset;
    let mut backend = gpu();
    let before = qm_launches();
    check_gpu(&mut backend, (256, 12, 2), 8192, 0, 4096);
    check_gpu(&mut backend, (256, 12, 2), 4096, 1000, 2048);
    check_gpu(&mut backend, (128, 12, 2), 2500, 300, 1250);
    assert_eq!(qm_launches(), before, "paged_prefill_qm ran");
}
