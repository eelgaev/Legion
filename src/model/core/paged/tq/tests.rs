//! The TurboQuant paged attention op (its CPU implementation, the rotated-basis math the kernels
//! do) against plain attention in the original basis over the same caches decoded: `R q · R k =
//! q · k` and `R^T Σ w R v = Σ w v`, so the two agree up to F32 rounding.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{Hparams, KvPool, KvType, KvTypes};

fn hparams(head_dim: i64, n_head: i64, n_head_kv: i64) -> Hparams {
    Hparams {
        n_layer: 1,
        n_embd: head_dim * n_head,
        n_head,
        n_head_kv,
        head_dim,
        n_rot: head_dim as i32,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        full_attention_interval: 1,
        ssm_d_conv: 4,
        ssm_d_state: 16,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Fills `pool` (of type `ty`, TurboQuant cache `tq` if any) with noise; returns every cell's
/// values as the attention reads them, in the original basis (`[d × n_kv]` per cell).
fn fill(
    pool: Tensor<'_>,
    ty: KvType,
    tq: Option<Cache>,
    cells: usize,
    row: usize,
    seed: u64,
) -> Vec<f32> {
    let x = noise(cells * row, seed);
    match tq {
        Some(c) => {
            let bytes = c.layout.bytes();
            assert_eq!(pool.ne()[0] as usize, bytes);
            let total = pool.ne()[1] as usize;
            let mut raw = vec![0u8; total * bytes];
            let mut out = vec![0f32; cells * row];
            for i in 0..cells {
                let cell = &mut raw[i * bytes..][..bytes];
                c.layout.encode(c.seed, &x[i * row..][..row], cell);
                out[i * row..][..row].copy_from_slice(&c.layout.decode(c.seed, cell, false));
            }
            pool.set_bytes(&raw);
            out
        }
        None => {
            assert_eq!(ty, KvType::F16);
            let total = pool.elements();
            let mut all = noise(total, seed);
            all[..x.len()].copy_from_slice(&x);
            let mut h = vec![0u16; total];
            // SAFETY: equal lengths.
            unsafe { ffi::ggml_fp32_to_fp16_row(all.as_ptr(), h.as_mut_ptr(), total as i64) };
            pool.set_bytes(&h.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>());
            h[..cells * row]
                .iter()
                // SAFETY: a pure conversion.
                .map(|&v| unsafe { ffi::ggml_fp16_to_fp32(v) })
                .collect()
        }
    }
}

struct Case {
    types: KvTypes,
    d: i64,
    n_head: i64,
    n_kv: i64,
    n_tok: usize,
    n_past: Vec<usize>,
    /// Per sequence, its table (pool blocks, by position / 64, or by that modulo `ring`).
    tables: Vec<Vec<i32>>,
    window: usize,
    ring: usize,
}

fn check(c: Case) {
    let hp = hparams(c.d, c.n_head, c.n_kv);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let n_blocks = *c.tables.iter().flatten().max().unwrap() as usize + 1;
    let kv = KvPool::new_ring(&hp, backend.buffer_type(), n_blocks, &[0], None, c.types).unwrap();
    let caches = kv.tq(0).expect("a TurboQuant cache");
    let (kp, vp) = kv.pools(0);
    let (d, nh, g) = (
        c.d as usize,
        c.n_head as usize,
        (c.n_head / c.n_kv) as usize,
    );
    let row = d * c.n_kv as usize;
    let cells = n_blocks * KV_BLOCK;
    let kd = fill(kp, c.types.k, caches.k, cells, row, 1);
    let vd = fill(vp, c.types.v, caches.v, cells, row, 2);

    let ns = c.n_past.len();
    let width = c.tables.iter().map(Vec::len).max().unwrap();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(types::F32, &[c.d, c.n_head, c.n_tok as i64, ns as i64])
        .as_input();
    let table = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scale = 1.0 / (d as f32).sqrt();
    let out = ctx
        .cont(attention(
            &ctx, q, kp, vp, table, past, caches, scale, c.window, c.ring,
        ))
        .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let qd = noise(q.elements(), 3);
    q.set_f32(&qd);
    let mut t = Vec::new();
    for (s, tab) in c.tables.iter().enumerate() {
        t.extend(tab);
        t.resize((s + 1) * width, 0);
    }
    table.set_i32(&t);
    past.set_i32(&c.n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();
    let got = out.to_f32();

    // Plain attention in f64 over the decoded caches.
    let mut want = Vec::new();
    for (s, tab) in c.tables.iter().enumerate() {
        for tok in 0..c.n_tok {
            for h in 0..nh {
                let qr = &qd[((s * c.n_tok + tok) * nh + h) * d..][..d];
                let at = |p: usize| {
                    let e = if c.ring > 0 {
                        p / KV_BLOCK % c.ring
                    } else {
                        p / KV_BLOCK
                    };
                    (tab[e] as usize * KV_BLOCK + p % KV_BLOCK) * row + h / g * d
                };
                let visible = c.n_past[s] + tok + 1;
                let first = if c.window > 0 {
                    visible.saturating_sub(c.window)
                } else {
                    0
                };
                let sc: Vec<f64> = (first..visible)
                    .map(|p| {
                        (0..d)
                            .map(|i| qr[i] as f64 * kd[at(p) + i] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = w.iter().sum();
                for i in 0..d {
                    let acc: f64 = (first..visible)
                        .zip(&w)
                        .map(|(p, w)| w / sum * vd[at(p) + i] as f64)
                        .sum();
                    want.push(acc as f32);
                }
            }
        }
    }
    assert_eq!(got.len(), want.len());
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 2e-5,
        "{:?}/{:?}, d {}, {} tokens, window {}, ring {}: max |op - reference| {diff}",
        c.types.k,
        c.types.v,
        c.d,
        c.n_tok,
        c.window,
        c.ring
    );
}

fn kv(k: KvType, v: KvType) -> KvTypes {
    KvTypes { k, v }
}

#[test]
fn decode_matches_original_basis() {
    for (k, v) in [
        (KvType::Tq4, KvType::Tq4),
        (KvType::Tq3, KvType::Tq3),
        (KvType::F16, KvType::Tq3),
        (KvType::Tq4, KvType::F16),
    ] {
        check(Case {
            types: kv(k, v),
            d: 128,
            n_head: 8,
            n_kv: 2,
            n_tok: 1,
            n_past: vec![150, 9],
            tables: vec![vec![3, 0, 5], vec![1]],
            window: 0,
            ring: 0,
        });
    }
}

#[test]
fn prefill_window_and_ring() {
    // Several tokens per sequence, wide heads, a sliding window over a ring of 3 entries.
    check(Case {
        types: kv(KvType::Tq3, KvType::Tq4),
        d: 256,
        n_head: 4,
        n_kv: 1,
        n_tok: 12,
        n_past: vec![230],
        tables: vec![vec![4, 1, 2]],
        window: 100,
        ring: 3,
    });
    check(Case {
        types: kv(KvType::Tq4, KvType::Tq4),
        d: 512,
        n_head: 2,
        n_kv: 1,
        n_tok: 20,
        n_past: vec![0, 40],
        tables: vec![vec![0], vec![2, 1]],
        window: 0,
        ring: 0,
    });
}

#[test]
fn reads_and_codes() {
    assert_eq!(cfg!(feature = "cuda"), reads(kv(KvType::Q8_0, KvType::Tq3)));
    assert!(!reads(kv(KvType::Q8_0, KvType::Q4_0)));
    assert!(!reads(kv(KvType::Bf16, KvType::Tq3)));
    let c = Cache {
        layout: turboquant::Layout {
            d: 128,
            n_kv: 2,
            bits: 4,
        },
        seed: 0,
    };
    assert_eq!(code(Some(c), ffi::ggml_type_GGML_TYPE_I8), Some(4));
    assert_eq!(code(None, types::F16), Some(0));
    assert_eq!(code(None, types::Q8_0), Some(1));
    assert_eq!(code(None, ffi::ggml_type_GGML_TYPE_Q4_0), Some(2));
    assert_eq!(code(None, types::BF16), None);
    // Decode groups take the direct kernel and its splits; longer ones pieces of the staging.
    let p = Plan::new(256, 32, 16, 1, 4, 64);
    assert!(p.direct && p.ints == p.len);
    let p = Plan::new(256, 32, 16, 512, 1, 64);
    assert!(!p.direct && p.cap >= 1 && p.len > p.stage);
    assert_eq!(p.o - p.q, 256 * 32 * 512);
}
