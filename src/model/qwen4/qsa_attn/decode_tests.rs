//! The split decode kernel (steps of at most [`DECODE_TOKENS`] tokens) on CUDA, ignored tests:
//! its output bits over many shapes (hashes to compare across builds, plus the oracle and
//! gather-path tolerances), and its time per call in decode and verify steps at long context.

use super::tests::{Case, bits_hash, f16_bytes, hparams, noise, run_typed, run_typed_with};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvType::{self, F16, Q4_0, Q8_0};
use crate::model::core::{KvPool, KvTypes};
use std::sync::atomic::Ordering;

fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// A case of `n_seqs` sequences of `n_tok` tokens at the end of `n_ctx` positions (each further
/// sequence 7 shorter), pools of 4, 512 picked, blocks out of order.
fn case(
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_ctx: usize,
    n_tok: usize,
    n_seqs: usize,
) -> Case {
    let n_blocks = n_ctx / KV_BLOCK + 1;
    Case {
        d,
        n_head,
        n_head_kv,
        kpool: 4,
        n_top: 512,
        n_tok,
        n_past: (0..n_seqs).map(|s| n_ctx - n_tok - 7 * s).collect(),
        tables: (0..n_seqs)
            .map(|s| {
                (0..n_blocks)
                    .map(|i| (s * n_blocks + i * 37 % n_blocks) as u32)
                    .collect()
            })
            .collect(),
    }
}

/// (head size, heads, kv heads, context, tokens, sequences) of [`cuda_decode_bits`].
const SHAPES: [(usize, usize, usize, usize, usize, usize); 18] = [
    (256, 24, 2, 10, 1, 1),
    (256, 24, 2, 37, 2, 1),
    (256, 24, 2, 100, 1, 1),
    (256, 24, 2, 613, 1, 1),
    (256, 24, 2, 2050, 1, 1),
    (256, 24, 2, 8192, 1, 1),
    (256, 24, 2, 8192, 3, 1),
    (256, 24, 2, 5000, 3, 2),
    (256, 24, 2, 40000, 1, 1),
    (256, 24, 2, 9000, 1, 16),
    (256, 24, 2, 8192, 64, 1),
    (128, 16, 2, 4096, 1, 1),
    (128, 16, 2, 777, 3, 2),
    (128, 16, 1, 4096, 2, 1),
    (256, 16, 1, 3000, 1, 1),
    (128, 8, 8, 4096, 4, 1),
    (256, 4, 2, 2500, 1, 1),
    (128, 12, 2, 3333, 2, 1),
];

/// The cache types a shape is checked with: F16, and the quantized pairs at qwen4exp's shape.
fn types_of(d: usize, n_head: usize, n_head_kv: usize, n_ctx: usize) -> Vec<KvTypes> {
    let mut types = vec![KvTypes::default()];
    if (d, n_head, n_head_kv) == (256, 24, 2) && (2050..=9000).contains(&n_ctx) {
        types.extend([(Q8_0, Q8_0), (Q8_0, Q4_0), (Q4_0, Q4_0)].map(|(k, v)| KvTypes { k, v }));
    }
    types
}

/// The decode kernel's output bits over shapes that reach every branch of it: qwen4exp's (head
/// size 256, 24 heads over 2 kv heads) in decode and verify steps from a few keys (fewer than
/// the parts: empty parts) to 40K, partial chunks, peaky softmaxes (a large query scale), the
/// quantized caches, and the other head counts and sizes the kernel takes. Prints one hash per
/// case: a kernel change that keeps every hash keeps the outputs bit for bit.
#[test]
#[ignore = "needs a GPU"]
fn cuda_decode_bits() {
    let mut backend = gpu();
    for &(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs) in &SHAPES {
        let c = case(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs);
        for t in types_of(d, n_head, n_head_kv, n_ctx) {
            for q_scale in [3.0, 24.0] {
                let r = run_typed(&c, &mut backend, q_scale, t);
                println!(
                    "decode bits {:?}/{:?} d {d} {n_head}/{n_head_kv} heads {n_seqs} x {n_tok} at \
                     {n_ctx} q x{q_scale}: {:016x} (vs exact {:.2e}, vs dense {:.2e})",
                    t.k, t.v, r.op_hash, r.op_exact, r.op_gather
                );
                // A peaky softmax leaves the dense path's F16 query and weights further off.
                if q_scale == 3.0 {
                    assert!(r.op_gather < 2e-3, "|op - dense| {}", r.op_gather);
                    assert!(r.op_exact < 3e-5, "|op - exact| {}", r.op_exact);
                }
                assert!(r.op_exact < 1e-4, "|op - exact| {}", r.op_exact);
            }
        }
    }
}

/// Milliseconds per decode-kernel call at qwen4exp's shape and the output's hash, per head-groups
/// variant (see [`DECODE_HG_OVERRIDE`]): `n_seqs`
/// sequences (each its own block order over one pool of `n_ctx` positions) of `n_tok` tokens at
/// the end of the context, every token with 512 pseudo-random pools of 4 before it plus its
/// tail. The graph holds `nodes` calls (as many as a step's sparse layers), so the time per
/// call carries little of the graph launch.
fn decode_ms(
    backend: &mut Backend,
    types: KvTypes,
    n_ctx: usize,
    (n_tok, n_seqs): (usize, usize),
    nodes: usize,
    reps: usize,
    variants: &[i32],
) -> Vec<(f64, u64)> {
    let n_blocks = n_ctx / KV_BLOCK + 1;
    let hp = hparams(256, 24, 2);
    let kv = KvPool::new_ring(&hp, backend.buffer_type(), n_blocks, &[0], None, types).unwrap();
    let (kp, vp) = kv.pools(0);
    let fill = |t: Tensor<'_>, ty: KvType, seed: u64| {
        let x = noise(t.elements(), seed);
        if ty == F16 {
            t.set_bytes(&f16_bytes(&x));
        } else {
            t.set_bytes(&crate::ggml::quantize(ty.ggml(), &x, t.ne()[0] as usize));
        }
    };
    fill(kp, types.k, 1);
    fill(vp, types.v, 2);
    let ctx = Context::new(4 * nodes + 16, 4 * nodes + 16);
    let q = ctx
        .new_tensor(types::F32, &[256, 24, n_tok as i64, n_seqs as i64])
        .as_input();
    let table = ctx
        .new_tensor(types::I32, &[n_blocks as i64, n_seqs as i64])
        .as_input();
    let cap = 512 * 4 + 3;
    let ng = n_tok * n_seqs;
    let sel = ctx
        .new_tensor(types::I32, &[1 + cap as i64, ng as i64])
        .as_input();
    let outs: Vec<Tensor<'_>> = (0..nodes)
        .map(|_| attention(&ctx, q, kp, vp, table, sel).as_output())
        .collect();
    let mut graph = ctx.new_graph(4 * nodes + 16);
    for &o in &outs {
        graph.expand(o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(
        &noise(q.elements(), 3)
            .iter()
            .map(|x| 3.0 * x)
            .collect::<Vec<_>>(),
    );
    let mut tab = Vec::with_capacity(n_blocks * n_seqs);
    for s in 0..n_seqs {
        tab.extend((0..n_blocks).map(|i| ((i * (2 * s + 1) * 37 + s) % n_blocks) as i32));
    }
    table.set_i32(&tab);
    let mut rows = Vec::with_capacity(ng * (1 + cap));
    let r = noise(ng * 512, 4);
    for s in 0..n_seqs {
        for t in 0..n_tok {
            let p = n_ctx - n_tok - 7 * s + t;
            let tt = s * n_tok + t;
            let mut pools: Vec<usize> = (0..512)
                .map(|j| ((r[tt * 512 + j].abs() * (p / 4) as f32) as usize).min(p / 4 - 1))
                .collect();
            pools.sort_unstable();
            pools.dedup();
            let mut row: Vec<i32> = pools
                .iter()
                .flat_map(|&b| (0..4).map(move |k| (b * 4 + k) as i32))
                .collect();
            row.extend(((p + 1) / 4 * 4..=p).map(|x| x as i32));
            let n = row.len();
            rows.push(n as i32);
            rows.extend(row);
            rows.resize(rows.len() + cap - n, -1);
        }
    }
    sel.set_i32(&rows);
    let _reset = HgReset;
    variants
        .iter()
        .map(|&hg| {
            DECODE_HG_OVERRIDE.store(hg, Ordering::Relaxed);
            backend.compute(&graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(&graph).unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / (reps * nodes) as f64;
            let hash = bits_hash(&outs[0].to_f32());
            for o in &outs[1..] {
                assert_eq!(
                    bits_hash(&o.to_f32()),
                    hash,
                    "calls on the same inputs differ"
                );
            }
            (ms, hash)
        })
        .collect()
}

/// Puts the decode kernel's head groups back to the launcher's choice, even when an assert fails.
struct HgReset;

impl Drop for HgReset {
    fn drop(&mut self) {
        DECODE_HG_OVERRIDE.store(0, Ordering::Relaxed);
    }
}

/// Times the decode kernel at qwen4exp's shape, F16 cache (and q8_0 at 131K): decode steps of 1
/// token, MTP verify steps of 3 (two drafts), and 4 and 16 sequences of 1 and of 3 tokens, at
/// 16K, 32K, 64K and 131K. Twelve calls per graph, the reference kernel (`decode_kernel`) and the
/// fast one with the launcher's head groups, then 1, 2 and 4 head groups forced for the smaller
/// steps; every variant's hash is the reference's.
#[test]
#[ignore = "needs a GPU; a timing, not a check"]
fn cuda_decode_speed() {
    let mut backend = gpu();
    let steps = [(1, 1), (3, 1), (1, 4), (3, 4), (1, 16), (3, 16)];
    for n_ctx in [16384, 32768, 65536, 131072] {
        let mut types = vec![KvTypes::default()];
        if n_ctx == 131072 {
            types.push(KvTypes { k: Q8_0, v: Q8_0 });
        }
        for t in types {
            for &(n_tok, n_seqs) in &steps {
                let mut variants = vec![-1, 0];
                if n_tok * n_seqs <= 4 {
                    variants.extend([1, 2, 4]);
                }
                let r = decode_ms(&mut backend, t, n_ctx, (n_tok, n_seqs), 12, 50, &variants);
                let line: Vec<String> = variants
                    .iter()
                    .zip(&r)
                    .map(|(hg, (ms, _))| match hg {
                        -1 => format!("reference {ms:.4}"),
                        0 => format!("fast {ms:.4}"),
                        _ => format!("hg{hg} {ms:.4}"),
                    })
                    .collect();
                println!(
                    "decode speed {:?}/{:?} {n_seqs} x {n_tok} tokens at {n_ctx}: ms per call {}, \
                     hash {:016x}",
                    t.k,
                    t.v,
                    line.join(", "),
                    r[0].1
                );
                for (hg, (_, h)) in variants.iter().zip(&r) {
                    assert_eq!(
                        *h, r[0].1,
                        "head groups {hg}: the output differs from the reference"
                    );
                }
            }
        }
    }
}

/// The fast decode kernel against the reference (`decode_kernel`), bit for bit, with the
/// launcher's head groups and with 1, 2, 3 and 5 forced (5 leaves the last group of 12 heads
/// empty and gives the others 3 of their 4 head slots), over [`cuda_decode_bits`]' shapes.
#[test]
#[ignore = "needs a GPU"]
fn cuda_decode_matches_reference() {
    let mut backend = gpu();
    let _reset = HgReset;
    for &(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs) in &SHAPES {
        let c = case(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs);
        for t in types_of(d, n_head, n_head_kv, n_ctx) {
            for q_scale in [3.0, 24.0] {
                DECODE_HG_OVERRIDE.store(-1, Ordering::Relaxed);
                let mut got = Vec::new();
                let want = run_typed_with(&c, &mut backend, q_scale, t, &mut |rerun| {
                    for hg in [0, 1, 2, 3, 5] {
                        DECODE_HG_OVERRIDE.store(hg, Ordering::Relaxed);
                        got.push((hg, rerun()));
                    }
                })
                .op_hash;
                for (hg, h) in got {
                    assert_eq!(
                        h, want,
                        "{:?}/{:?} d {d} {n_head}/{n_head_kv} heads {n_seqs} x {n_tok} at \
                         {n_ctx} q x{q_scale}, head groups {hg}",
                        t.k, t.v
                    );
                }
            }
        }
    }
}

/// A short run for a profiler (`nsys`, `ncu`): steps of 1 x 1, 1 x 3 and 16 x 3 tokens at 131K,
/// F16, the reference kernel then the fast one, 12 calls per graph, 5 graphs each.
#[test]
#[ignore = "needs a GPU; for a profiler"]
fn cuda_decode_profile() {
    let mut backend = gpu();
    for step in [(1, 1), (1, 3), (16, 3)] {
        let r = decode_ms(
            &mut backend,
            KvTypes::default(),
            131072,
            step,
            12,
            5,
            &[-1, 0],
        );
        println!(
            "decode profile {step:?}: reference {:.4} ms, fast {:.4} ms per call",
            r[0].0, r[1].0
        );
    }
}
