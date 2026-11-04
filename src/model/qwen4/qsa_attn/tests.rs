//! The sparse attention ops against the gather path they replace (the same selection as a
//! mask over the gathered span, then mul_mat / soft_max_ext / mul_mat) and an f64 oracle; on
//! CUDA (ignored tests) also against the CPU implementations of the selection ops, exactly.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{KvPool, KvType, KvTypes};

pub(super) use super::bench::{f16_bytes, hparams, noise};

fn bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn max_diff(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// One test case: a group of sequences (blocks out of order), `n_tok` tokens each after
/// `n_past[s]`, pools of `kpool`, `n_top` pools picked per token.
pub(super) struct Case {
    pub(super) d: usize,
    pub(super) n_head: usize,
    pub(super) n_head_kv: usize,
    pub(super) kpool: usize,
    pub(super) n_top: usize,
    pub(super) n_tok: usize,
    pub(super) n_past: Vec<usize>,
    pub(super) tables: Vec<Vec<u32>>,
}

/// The host side of a case: what the indexer and the step's tables would hold.
struct Tables {
    n_pool: usize,
    /// I32 [kpool, n_pool]
    pool_idxs: Vec<i32>,
    /// I32 [2, tokens]
    vis: Vec<i32>,
    /// F32 [n_pool, tokens]: random scores for visible pools, -inf elsewhere
    score: Vec<f32>,
    /// I32 [n_top, tokens]: the best pools, shuffled
    top: Vec<i32>,
    /// I32 [kpool - 1, tokens]
    tail: Vec<i32>,
    pos: Vec<i32>,
    /// Per token: the positions it attends to, ascending.
    want: Vec<Vec<i32>>,
}

impl Case {
    fn tokens(&self) -> usize {
        self.n_tok * self.n_past.len()
    }

    fn tables(&self) -> Tables {
        let kp = self.kpool;
        let mut pool_idxs = Vec::new();
        let mut first = Vec::new();
        for &np in &self.n_past {
            first.push(pool_idxs.len() / kp);
            for b in 0..(np + self.n_tok) / kp {
                pool_idxs.extend((0..kp).map(|k| (b * kp + k) as i32));
            }
        }
        // Two padding entries, as the step pads its listing: never visible.
        pool_idxs.extend(std::iter::repeat_n(i32::MAX, 2 * kp));
        let n_pool = pool_idxs.len() / kp;
        let n_top = self.n_top.min(n_pool);
        let mut rng = noise(self.tokens() * (n_pool + n_top), 7).into_iter();
        let (mut vis, mut score, mut top, mut tail, mut pos, mut want) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for (s, &np) in self.n_past.iter().enumerate() {
            for t in 0..self.n_tok {
                let p = np + t;
                let (f, e) = (first[s], first[s] + (p + 1) / kp);
                vis.extend([f as i32, e as i32]);
                let row: Vec<f32> = (0..n_pool)
                    .map(|j| {
                        let r = rng.next().unwrap().abs();
                        if (f..e).contains(&j) {
                            r
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                let mut order: Vec<usize> = (0..n_pool).collect();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                let mut picked: Vec<usize> = order[..n_top].to_vec();
                // Any order: rotate by a pseudo-random amount and reverse.
                let r = (rng.next().unwrap().abs() * n_top as f32) as usize % n_top.max(1);
                picked.rotate_left(r);
                picked.reverse();
                let mut sel: Vec<i32> = picked
                    .iter()
                    .filter(|&&j| row[j].is_finite())
                    .flat_map(|&j| pool_idxs[j * kp..][..kp].to_vec())
                    .collect();
                let n_tail = (p + 1) % kp;
                for k in 0..kp - 1 {
                    let x = if k < n_tail { (p - k) as i32 } else { i32::MAX };
                    tail.push(x);
                    if k < n_tail {
                        sel.push(x);
                    }
                }
                sel.sort_unstable();
                want.push(sel);
                top.extend(picked.iter().map(|&j| j as i32));
                score.extend(row);
                pos.push(p as i32);
            }
        }
        Tables {
            n_pool,
            pool_idxs,
            vis,
            score,
            top,
            tail,
            pos,
            want,
        }
    }
}

/// Rows of an I32 selection output: the counted positions of every token.
fn selections(sel: &[i32], row_len: usize) -> Vec<Vec<i32>> {
    sel.chunks(row_len)
        .map(|r| {
            let n = r[0] as usize;
            assert!(r[1 + n..].iter().all(|&x| x == -1), "padding");
            r[1..1 + n].to_vec()
        })
        .collect()
}

/// f64 attention over each token's positions from the (F16) cache. With `round`, the query
/// and the softmax weights are rounded to F16 first, as the gather path's matmuls round them.
fn oracle(c: &Case, q: &[f32], k: &[f32], v: &[f32], want: &[Vec<i32>], round: bool) -> Vec<f32> {
    let (d, g) = (c.d, c.n_head / c.n_head_kv);
    let kv_row = d * c.n_head_kv;
    let mut out = Vec::new();
    for (tok, sel) in want.iter().enumerate() {
        let tab = &c.tables[tok / c.n_tok];
        for h in 0..c.n_head {
            let at = (tok * c.n_head + h) * d;
            let mut qr = q[at..at + d].to_vec();
            if round {
                round_f16(&mut qr);
            }
            let cell = |p: i32| {
                let p = p as usize;
                (tab[p / KV_BLOCK] as usize * KV_BLOCK + p % KV_BLOCK) * kv_row + h / g * d
            };
            let scores: Vec<f64> = sel
                .iter()
                .map(|&p| {
                    let dot: f64 = (0..d).map(|i| qr[i] as f64 * k[cell(p) + i] as f64).sum();
                    dot / (d as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let mut w: Vec<f64> = exp.iter().map(|x| x / sum).collect();
            if round {
                let mut r: Vec<f32> = w.iter().map(|&x| x as f32).collect();
                round_f16(&mut r);
                w = r.into_iter().map(f64::from).collect();
            }
            for i in 0..d {
                let acc: f64 = sel
                    .iter()
                    .zip(&w)
                    .map(|(&p, &w)| w * v[cell(p) + i] as f64)
                    .sum();
                out.push(acc as f32);
            }
        }
    }
    out
}

/// Max differences of a case's results.
pub(super) struct Diffs {
    /// The op against the oracle that rounds as the gather path does.
    pub(super) op_rounded: f32,
    /// The op against the exact oracle.
    pub(super) op_exact: f32,
    /// The op's mean error over the exact oracle's mean magnitude: a truncating accumulator
    /// shows as a bias.
    pub(super) op_bias: f64,
    pub(super) op_gather: f32,
    pub(super) gather_rounded: f32,
    pub(super) gather_exact: f32,
    /// FNV-1a of the op's output bits: equal across builds when two kernels agree bit for bit.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) op_hash: u64,
}

pub(super) use super::bench::bits_hash;

/// Runs a case on `backend`: pool mask, selection and attention ops, plus the gather path over
/// the same selection, against the oracles.
fn run(c: &Case, backend: &mut Backend, q_scale: f32) -> Diffs {
    run_typed(c, backend, q_scale, KvTypes::default())
}

/// [`run`] with K and V caches of `types`: the oracles read the values the cache holds as F16
/// (a quantized cache's dequantized and rounded to F16, as the gather path reads them).
pub(super) fn run_typed(c: &Case, backend: &mut Backend, q_scale: f32, types: KvTypes) -> Diffs {
    run_typed_with(c, backend, q_scale, types, &mut |_| {})
}

/// [`run_typed`], then `rerun` gets a function that computes the graph again (on the same
/// inputs) and returns the attention op's output hash: for comparing kernel variants without
/// running the oracles again.
pub(super) fn run_typed_with(
    c: &Case,
    backend: &mut Backend,
    q_scale: f32,
    types: KvTypes,
    rerun: &mut dyn FnMut(&mut dyn FnMut() -> u64),
) -> Diffs {
    let hp = hparams(c.d as i64, c.n_head as i64, c.n_head_kv as i64);
    let n_blocks = *c.tables.iter().flatten().max().unwrap() as usize + 1;
    let kv = KvPool::new_ring(&hp, backend.buffer_type(), n_blocks, &[0], None, types).unwrap();
    let (kp, vp) = kv.pools(0);
    let row = c.d * c.n_head_kv;
    // The values written to a pool of `ty`, as attention reads them.
    let store = |t: Tensor<'_>, ty: KvType, x: Vec<f32>| -> Vec<f32> {
        let mut y = if ty == KvType::F16 {
            t.set_bytes(&f16_bytes(&x));
            x
        } else {
            let raw = crate::ggml::quantize(ty.ggml(), &x, row);
            t.set_bytes(&raw);
            crate::ggml::dequantize(ty.ggml(), &raw, x.len())
        };
        round_f16(&mut y);
        y
    };
    let kdat = store(kp, types.k, noise(kp.elements(), 1));
    let vdat = store(vp, types.v, noise(vp.elements(), 2));

    let tb = c.tables();
    let ns = c.n_past.len();
    let ng = c.tokens();
    let (d, nt) = (c.d as i64, c.n_tok as i64);
    let n_top = c.n_top.min(tb.n_pool) as i64;
    let end = c.n_past.iter().max().unwrap() + c.n_tok;
    let n_kv = end.div_ceil(KV_BLOCK) * KV_BLOCK;
    let width = c.tables.iter().map(Vec::len).max().unwrap();

    let ctx = Context::new(64, 64);
    let input = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input();
    let q = input(types::F32, &[d, c.n_head as i64, nt, ns as i64]);
    let table = input(types::I32, &[width as i64, ns as i64]);
    let vis = input(types::I32, &[2, ng as i64]);
    let score = input(types::F32, &[tb.n_pool as i64, ng as i64]);
    let top = input(types::I32, &[n_top, ng as i64]);
    let pidx = input(types::I32, &[c.kpool as i64, tb.n_pool as i64]);
    let tail = input(types::I32, &[c.kpool as i64 - 1, ng as i64]);
    let pos = input(types::I32, &[ng as i64]);
    let mask_op = pool_mask(&ctx, vis, tb.n_pool as i64).as_output();
    let sel = select(&ctx, top, score, pidx, tail, pos).as_output();
    let att = attention(&ctx, q, kp, vp, table, sel).as_output();

    let blocks = input(types::I32, &[(n_kv / KV_BLOCK * ns) as i64]);
    let mask = input(types::F32, &[n_kv as i64, nt, 1, ns as i64]);
    let (k, vt) = kv.read(&ctx, 0, blocks, n_kv as i64, ns as i64);
    let kq = ctx.mul_mat(ctx.permute(k, [0, 2, 1, 3]), ctx.permute(q, [0, 2, 1, 3]));
    let kq = ctx.soft_max_ext(kq, mask, 1.0 / (d as f32).sqrt(), 0.0);
    let gathered = ctx
        .cont(ctx.permute(ctx.mul_mat(vt, kq), [0, 2, 1, 3]))
        .as_output();

    let mut graph = ctx.new_graph(64);
    for t in [mask_op, sel, att, gathered] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();

    let qdat: Vec<f32> = noise(q.elements(), 3).iter().map(|x| q_scale * x).collect();
    let mut t = Vec::new();
    let mut b = Vec::new();
    for (s, tab) in c.tables.iter().enumerate() {
        t.extend(tab.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
        let used = (c.n_past[s] + c.n_tok).div_ceil(KV_BLOCK);
        b.extend((0..n_kv / KV_BLOCK).map(|i| tab[i.min(used - 1)] as i32));
    }
    let mut m = vec![f32::NEG_INFINITY; n_kv * ng];
    for (tok, w) in tb.want.iter().enumerate() {
        for &p in w {
            m[tok * n_kv + p as usize] = 0.0;
        }
    }
    // The graph allocator may reuse an input's memory once its last reader has run: every
    // compute sets them all again.
    let set_inputs = || {
        q.set_f32(&qdat);
        table.set_i32(&t);
        blocks.set_i32(&b);
        mask.set_f32(&m);
        vis.set_bytes(&bytes(&tb.vis));
        score.set_f32(&tb.score);
        top.set_bytes(&bytes(&tb.top));
        pidx.set_bytes(&bytes(&tb.pool_idxs));
        tail.set_bytes(&bytes(&tb.tail));
        pos.set_bytes(&bytes(&tb.pos));
    };
    set_inputs();
    backend.compute(&graph).unwrap();

    // The selection ops are exact on every backend.
    let mask_got: Vec<u16> = mask_op
        .bytes()
        .chunks(2)
        .map(|x| u16::from_ne_bytes([x[0], x[1]]))
        .collect();
    for (tok, row) in mask_got.chunks(tb.n_pool).enumerate() {
        let (f, e) = (tb.vis[2 * tok] as usize, tb.vis[2 * tok + 1] as usize);
        for (j, &x) in row.iter().enumerate() {
            assert_eq!(x, if (f..e).contains(&j) { KEEP } else { DROP }, "mask");
        }
    }
    let got = selections(&sel.to_i32(), sel.ne()[0] as usize);
    assert_eq!(got, tb.want, "selection");

    let (a, g) = (att.to_f32(), gathered.to_f32());
    rerun(&mut || {
        set_inputs();
        backend.compute(&graph).unwrap();
        assert_eq!(
            selections(&sel.to_i32(), sel.ne()[0] as usize),
            tb.want,
            "selection"
        );
        bits_hash(&att.to_f32())
    });
    let rounded = oracle(c, &qdat, &kdat, &vdat, &tb.want, true);
    let exact = oracle(c, &qdat, &kdat, &vdat, &tb.want, false);
    let bias: f64 = a
        .iter()
        .zip(&exact)
        .map(|(&x, &y)| x as f64 - y as f64)
        .sum();
    let mag: f64 = exact.iter().map(|&y| (y as f64).abs()).sum();
    Diffs {
        op_rounded: max_diff(&a, &rounded),
        op_exact: max_diff(&a, &exact),
        op_bias: bias / mag,
        op_gather: max_diff(&a, &g),
        gather_rounded: max_diff(&g, &rounded),
        gather_exact: max_diff(&g, &exact),
        op_hash: bits_hash(&a),
    }
}

/// See paged attention's tests: ggml's CPU matmuls on aarch64 accumulate F16 products in F16.
pub(super) const GATHER_TOLERANCE: f32 = 2e-3;

pub(super) fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b
}

#[test]
fn select_row_sorts_pools_and_drops_dead_entries() {
    // Pools 0..4 of 4 (positions 0..16), token at 17: pools 3, 0 and a dead pool 4 picked.
    let members = |j: usize, k: usize| (j * 4 + k) as i32;
    let score = |j: usize| if j < 4 { 1.0 } else { f32::NEG_INFINITY };
    let mut out = vec![0; 1 + 3 * 4 + 3];
    select_row(
        &[3, 4, 0],
        score,
        members,
        4,
        &[17, 16, i32::MAX],
        17,
        &mut out,
    );
    assert_eq!(
        out,
        [10, 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, -1, -1, -1, -1, -1]
    );
}

#[test]
fn cpu_ops_match_gather_and_oracle() {
    let cases = [
        // Fewer visible pools than the budget for most tokens: dense attention.
        Case {
            d: 32,
            n_head: 4,
            n_head_kv: 2,
            kpool: 4,
            n_top: 64,
            n_tok: 9,
            n_past: vec![40],
            tables: vec![vec![1, 0]],
        },
        // Two sequences, a small budget: most keys dropped; GQA 8.
        Case {
            d: 64,
            n_head: 8,
            n_head_kv: 1,
            kpool: 4,
            n_top: 3,
            n_tok: 6,
            n_past: vec![61, 100],
            tables: vec![vec![3, 1], vec![0, 4]],
        },
    ];
    let mut backend = cpu();
    for c in &cases {
        let r = run(c, &mut backend, 1.0);
        // The CPU op rounds as the gather path does.
        assert!(
            r.op_rounded < 1e-5,
            "|op - oracle| {} (unrounded oracle: op {}, gather {}; op bias {})",
            r.op_rounded,
            r.op_exact,
            r.gather_exact,
            r.op_bias
        );
        assert!(
            r.op_gather < GATHER_TOLERANCE,
            "|op - gather| {} (|gather - oracle| {})",
            r.op_gather,
            r.gather_rounded
        );
    }
}

/// The CUDA kernels against the f64 oracle and against ggml's dense CUDA path over the same
/// selection, with the prompt kernel's chain off and on: qwen4exp's shape (head size 256, 24
/// heads over 2 kv heads, pools of 4, 512 pools picked) over 2,048 keys (every visible key
/// selected) and over 8K (sparse), and head size 128 with 8 heads per kv head (the other
/// kernel instances), up to a 2,048-token chunk. V is positive, so a truncating accumulator
/// would show as a bias.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_oracle_and_dense_path() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use std::sync::atomic::Ordering;
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    // (head size, heads, kv heads, context, tokens, sequences). Up to DECODE_TOKENS tokens
    // (decode and verify steps) take the split kernel: one token, a verify step of 4, two
    // sequences of 3, and 64.
    // LEGION_TINY=1 (a compute-sanitizer run): 300 tokens (the prompt kernel, both head sizes)
    // and one token (the split kernel) only.
    let tiny = std::env::var("LEGION_TINY").is_ok_and(|v| v == "1");
    let cases = [
        (256, 24, 2, 2048, 64, 1),
        (256, 24, 2, 8192, 128, 1),
        (256, 24, 2, 300, 300, 1),
        (256, 24, 2, 8192, 1, 1),
        (256, 24, 2, 8192, 4, 1),
        (256, 24, 2, 5000, 3, 2),
        (128, 16, 2, 300, 300, 1),
        (128, 16, 2, 4096, 2048, 1),
        // Other query heads per kv head on the prompt kernels (more than DECODE_TOKENS
        // tokens, so not the split kernel): 16, and 1 at both head sizes.
        (128, 16, 1, 2048, 128, 1),
        (256, 2, 2, 2048, 300, 1),
        (128, 8, 8, 4096, 128, 1),
    ];
    // Back to the environment's choice even when an assert fails.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CHAIN_OVERRIDE.store(0, Ordering::Relaxed);
        }
    }
    let _reset = Reset;
    // On sm_80+ the prompt kernel is the mma.m16n8k16 one, which has no chain: the chain pass
    // tests the Volta prompt kernel only on sm_70/75.
    for chain in [false, true] {
        CHAIN_OVERRIDE.store(if chain { 2 } else { 1 }, Ordering::Relaxed);
        for &(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs) in cases
            .iter()
            .filter(|c| !tiny || c.4 == 300 || (c.3, c.4) == (8192, 1))
        {
            if chain && n_tok * n_seqs <= DECODE_TOKENS as usize {
                continue; // the split kernel has no chain
            }
            let n_blocks = n_ctx / KV_BLOCK + 1;
            let tables: Vec<Vec<u32>> = (0..n_seqs)
                .map(|s| {
                    (0..n_blocks)
                        .map(|i| (s * n_blocks + i * 37 % n_blocks) as u32)
                        .collect()
                })
                .collect();
            let c = Case {
                d,
                n_head,
                n_head_kv,
                kpool: 4,
                n_top: 512,
                n_tok,
                n_past: (0..n_seqs).map(|s| n_ctx - n_tok - 7 * s).collect(),
                tables,
            };
            let r = run(&c, &mut backend, 3.0);
            println!(
                "chain {chain}, d {d}, {n_head}/{n_head_kv} heads, {n_seqs} x {n_tok} tokens at {n_ctx}: \
                 op vs exact {:.2e} (bias {:.1e}), vs rounded {:.2e}, vs dense {:.2e}; \
                 dense vs exact {:.2e}, vs rounded {:.2e}; op bits {:016x}",
                r.op_exact,
                r.op_bias,
                r.op_rounded,
                r.op_gather,
                r.gather_exact,
                r.gather_rounded,
                r.op_hash
            );
            assert!(r.op_exact < 3e-5, "|op - exact| {}", r.op_exact);
            assert!(r.op_bias.abs() < 1e-6, "relative bias {}", r.op_bias);
            assert!(r.op_gather < 2e-3, "|op - dense| {}", r.op_gather);
        }
    }
}

/// Times the CUDA ops at qwen4exp's shape: a 4,096-token chunk at the end of a 32K context,
/// every token with 512 pools of 4 (2,048 + tail keys). Prints milliseconds per op call.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a timing, not a check"]
fn cuda_attention_speed() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (n_ctx, n_tok) = (32768usize, 4096usize);
    let (ms, bits) = attention_ms(&mut backend, KvTypes::default(), n_ctx, n_tok, 5);
    println!("qsa_attn: {n_tok} tokens at {n_ctx}: {ms:.2} ms per call (output bits {bits:016x})");
}

#[cfg(feature = "cuda")]
/// Milliseconds per [`attention`] call on `backend` (the mean of `reps` after a warm-up) at
/// qwen4exp's shape with caches of `types`: `n_tok` tokens at the end of an `n_ctx` context,
/// every token with 512 pools of 4 before it plus its tail. Also returns the output's
/// [`bits_hash`], so a kernel change that keeps it keeps these outputs bit for bit.
pub(super) fn attention_ms(
    backend: &mut Backend,
    types: KvTypes,
    n_ctx: usize,
    n_tok: usize,
    reps: usize,
) -> (f64, u64) {
    let t = super::bench::prompt_ms(
        backend,
        types,
        n_ctx,
        n_tok,
        reps,
        1,
        false,
        &[PromptKernel::Auto],
    );
    (t[0].ms, t[0].bits)
}

/// [`scores`] and ggml's `lightning_indexer(q, keys, w, pool_mask(vis))` on `backend`, for
/// `ng` tokens over `n_pool` pools (`nh` heads of 128 dims: 4 in Flash-Next's indexer, 32 in
/// GLM-5.3-Flash's): (op, ggml; ggml's only for 4 heads, else empty),
/// and the seconds `reps` runs of each took. Token `t` sees pools `t % 7 .. n_pool - 3 * t`.
fn scores_case(
    backend: &mut Backend,
    n_pool: i64,
    ng: i64,
    nh: i64,
    reps: usize,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let d = 128i64;
    let ctx = Context::new(32, 32);
    let q = ctx.new_tensor(types::F32, &[d, nh, ng]).as_input();
    let keys = ctx.new_tensor(types::F32, &[d, 1, n_pool]).as_input();
    let w = ctx.new_tensor(types::F32, &[nh, ng]).as_input();
    let vis = ctx.new_tensor(types::I32, &[2, ng]).as_input();
    let op = scores(&ctx, q, keys, w, vis, None, None).as_output();
    let mask = pool_mask(&ctx, vis, n_pool);
    let reference = ctx.lightning_indexer(q, keys, w, mask).as_output();
    let (mut g_op, mut g_ref, mut all) = (ctx.new_graph(32), ctx.new_graph(32), ctx.new_graph(32));
    g_op.expand(op);
    g_ref.expand(reference);
    all.expand(op);
    all.expand(reference);
    ctx.alloc_graph(&all, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise((d * nh * ng) as usize, 1));
    keys.set_f32(&noise((d * n_pool) as usize, 2));
    w.set_f32(&vec![1.0 / (d as f32).sqrt(); (nh * ng) as usize]);
    let v: Vec<i32> = (0..ng)
        .flat_map(|t| [(t % 7) as i32, (n_pool - 3 * t).max(0) as i32])
        .collect();
    vis.set_bytes(&bytes(&v));
    let mut time = |g: &crate::ggml::Graph<'_>| {
        backend.compute(g).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(g).unwrap();
        }
        t0.elapsed().as_secs_f64() / reps.max(1) as f64
    };
    let t_op = time(&g_op);
    if nh != 4 {
        return (op.to_f32(), Vec::new(), t_op, 0.0);
    }
    let t_ref = time(&g_ref);
    (op.to_f32(), reference.to_f32(), t_op, t_ref)
}

/// [`scores`] with the key gather and bias in the op against the chain it replaces in the model
/// (`add(scores(reshape(get_rows(pooled, cells))), bias)`) on `backend`: (fused, chain). Pool j's
/// key sits in row `cells[j]` of `n_pool + 5` rows (a permutation, rows past the pools unused);
/// token `t` sees pools `t % 7 .. n_pool - 3 * t`.
/// With `f16`, the keys are the pool half of an F16 `[2 × d, rows]` cache, as the model's
/// indexer cache stores them.
fn fused_case(backend: &mut Backend, n_pool: i64, ng: i64, f16: bool) -> (Vec<f32>, Vec<f32>) {
    let (d, nh) = (128i64, 4i64);
    let n_rows = n_pool + 5;
    let ctx = Context::new(32, 32);
    let q = ctx.new_tensor(types::F32, &[d, nh, ng]).as_input();
    let store = if f16 {
        ctx.new_tensor(types::F16, &[2 * d, n_rows])
    } else {
        ctx.new_tensor(types::F32, &[d, n_rows])
    }
    .as_input();
    let pooled = if f16 {
        ctx.view_2d(store, [d, n_rows], store.nb()[1], d as usize * 2)
    } else {
        store
    };
    let cells = ctx.new_tensor(types::I32, &[n_pool]).as_input();
    let bias = ctx.new_tensor(types::F32, &[n_pool]).as_input();
    let w = ctx.new_tensor(types::F32, &[nh, ng]).as_input();
    let vis = ctx.new_tensor(types::I32, &[2, ng]).as_input();
    let fused = scores(&ctx, q, pooled, w, vis, Some(bias), Some(cells)).as_output();
    let keys = ctx.reshape_3d(ctx.get_rows(pooled, cells), d, 1, n_pool);
    let chain = ctx
        .add(scores(&ctx, q, keys, w, vis, None, None), bias)
        .as_output();
    let mut all = ctx.new_graph(32);
    all.expand(fused);
    all.expand(chain);
    ctx.alloc_graph(&all, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise((d * nh * ng) as usize, 1));
    let keys = noise((store.ne()[0] * n_rows) as usize, 2);
    if f16 {
        let mut h = vec![0u16; keys.len()];
        // SAFETY: equal lengths.
        unsafe { ffi::ggml_fp32_to_fp16_row(keys.as_ptr(), h.as_mut_ptr(), h.len() as i64) };
        store.set_bytes(&h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
    } else {
        store.set_f32(&keys);
    }
    // a stride coprime to n_pool + 5 walks a permutation of the rows
    let step = (1..)
        .map(|k| 7 * k + 3)
        .find(|s| gcd(*s, n_rows) == 1)
        .unwrap();
    let c: Vec<i32> = (0..n_pool)
        .map(|j| ((j * step + 2) % n_rows) as i32)
        .collect();
    cells.set_bytes(&bytes(&c));
    bias.set_f32(
        &noise(n_pool as usize, 5)
            .iter()
            .map(|x| 0.1 * x)
            .collect::<Vec<_>>(),
    );
    w.set_f32(
        &noise((nh * ng) as usize, 6)
            .iter()
            .map(|x| 0.1 + 0.05 * x.abs())
            .collect::<Vec<_>>(),
    );
    let v: Vec<i32> = (0..ng)
        .flat_map(|t| [(t % 7) as i32, (n_pool - 3 * t).max(0) as i32])
        .collect();
    vis.set_bytes(&bytes(&v));
    backend.compute(&all).unwrap();
    (fused.to_f32(), chain.to_f32())
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn assert_bits_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "entry {i}: {x} vs {y}");
    }
}

#[test]
fn cpu_scores_fuse_gather_and_bias() {
    for f16 in [false, true] {
        let (fused, chain) = fused_case(&mut cpu(), 200, 20, f16);
        assert_bits_eq(&fused, &chain);
    }
}

/// On CUDA: the gather and bias in the kernel give the chain's bits, at a small shape (partial
/// tiles, a pool count not a multiple of 4) and a chunk's, from F32 keys and F16 cache rows.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scores_fuse_gather_and_bias() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    for (n_pool, ng) in [(301i64, 37i64), (4096, 1024), (4099, 1024)] {
        for f16 in [false, true] {
            let (fused, chain) = fused_case(&mut gpu, n_pool, ng, f16);
            assert_bits_eq(&fused, &chain);
        }
    }
}

/// The same -inf pattern, and finite scores within `tol` of the reference's largest.
fn assert_scores_close(op: &[f32], reference: &[f32], tol: f32) {
    let scale = reference
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |m, v| m.max(v.abs()));
    for (i, (a, b)) in op.iter().zip(reference).enumerate() {
        assert_eq!(a.is_finite(), b.is_finite(), "entry {i}: {a} vs {b}");
        if b.is_finite() {
            assert!((a - b).abs() <= tol * scale, "entry {i}: {a} vs {b}");
        }
    }
}

/// On CUDA: GLM-5.3-Flash's 32 indexer heads (eight launches of 4, each adding to the last)
/// against the CPU's F32 scores.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scores_take_32_heads() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    for (n_pool, ng) in [(301i64, 37i64), (2048, 1024)] {
        let (op, _, _, _) = scores_case(&mut gpu, n_pool, ng, 32, 0);
        let (cpu_op, _, _, _) = scores_case(&mut cpu(), n_pool, ng, 32, 0);
        assert_scores_close(&op, &cpu_op, 2e-6);
    }
}

#[test]
fn cpu_scores_match_lightning_indexer() {
    let (op, reference, _, _) = scores_case(&mut cpu(), 200, 20, 4, 0);
    assert_scores_close(&op, &reference, 1e-5);
}

/// On CUDA: the tensor-core scores against ggml's lightning indexer, at a chunk's shape, and
/// the time of both (Strata-AC922 52a35fe measured 8x at 57K pools).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scores_match_lightning_indexer() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    // LEGION_TINY=1: the first case only (for compute-sanitizer).
    let cases = [(300i64, 40i64), (2048, 1024), (15000, 4096)];
    let n = if std::env::var_os("LEGION_TINY").is_some() {
        1
    } else {
        cases.len()
    };
    for &(n_pool, ng) in &cases[..n] {
        let reps = if ng >= 1024 { 5 } else { 1 };
        let (op, reference, t_op, t_ref) = scores_case(&mut gpu, n_pool, ng, 4, reps);
        let scale = reference
            .iter()
            .filter(|v| v.is_finite())
            .fold(0f32, |m, v| m.max(v.abs()));
        let diff = op
            .iter()
            .zip(&reference)
            .filter(|(_, b)| b.is_finite())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        // Against the CPU's F32 scores (a reference for both GPU paths).
        let (cpu_op, _, _, _) = scores_case(&mut cpu(), n_pool, ng, 4, 0);
        let rel = |x: &[f32]| {
            x.iter()
                .zip(&cpu_op)
                .filter(|(_, b)| b.is_finite())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale
        };
        println!(
            "{n_pool} pools x {ng} tokens: tensor cores {:.3} ms, ggml {:.3} ms; max |diff| / max: \
             op vs ggml {:.2e}, op vs CPU {:.2e}, ggml vs CPU {:.2e}",
            t_op * 1e3,
            t_ref * 1e3,
            diff / scale,
            rel(&op),
            rel(&reference)
        );
        assert_scores_close(&op, &cpu_op, 2e-6);
    }
}

#[test]
fn top_k_row_takes_the_threshold_rule() {
    let mut out = [0i32; 4];
    // Above the threshold first (ascending), then the lowest-index ties.
    top_k_row(&[1.0, 3.0, 2.0, 2.0, 5.0, 2.0], &mut out);
    assert_eq!(out, [1, 4, 2, 3]);
    // ggml's order: -0 below +0, -inf lowest, ties of -inf to the lowest index.
    let mut out = [0i32; 3];
    top_k_row(&[-0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY], &mut out);
    assert_eq!(out, [0, 1, 2]);
    let mut out = [0i32; 2];
    top_k_row(&[-0.0, -1.0, 0.0, -0.0], &mut out);
    assert_eq!(out, [2, 0]);
}

/// Score rows as the indexer makes them: -inf outside each row's visible range, ties when
/// `levels` > 0 (values rounded to that many steps), and zeros.
fn topk_scores(n: usize, rows: usize, levels: u32, seed: u64) -> Vec<f32> {
    let mut v = noise(n * rows, seed);
    for (r, row) in v.chunks_mut(n).enumerate() {
        let (first, end) = (r % 13, n - (3 * r) % (n / 2));
        for (j, x) in row.iter_mut().enumerate() {
            *x = if j < first || j >= end {
                f32::NEG_INFINITY
            } else if levels > 0 {
                (x.abs() * levels as f32).floor()
            } else if j % 17 == 0 {
                0.0
            } else {
                x.abs()
            };
        }
    }
    v
}

/// [`top_k`] (and ggml's `top_k`) of `scores` (`[n, rows]`) on `backend`, with the time of each
/// over `reps` runs.
fn topk_case(
    backend: &mut Backend,
    scores: &[f32],
    n: i64,
    k: i64,
    reps: usize,
) -> (Vec<i32>, Vec<i32>, f64, f64) {
    let rows = scores.len() as i64 / n;
    let ctx = Context::new(8, 8);
    let s = ctx.new_tensor(types::F32, &[n, rows]).as_input();
    let own = top_k(&ctx, s, k).as_output();
    let ggml = ctx.top_k(s, k as i32).as_output();
    let (mut g_own, mut g_ggml, mut all) = (ctx.new_graph(8), ctx.new_graph(8), ctx.new_graph(8));
    g_own.expand(own);
    g_ggml.expand(ggml);
    all.expand(own);
    all.expand(ggml);
    ctx.alloc_graph(&all, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    s.set_f32(scores);
    let mut time = |g: &crate::ggml::Graph<'_>| {
        backend.compute(g).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(g).unwrap();
        }
        t0.elapsed().as_secs_f64() / reps.max(1) as f64
    };
    let (t_own, t_ggml) = (time(&g_own), time(&g_ggml));
    (own.to_i32(), ggml.to_i32(), t_own, t_ggml)
}

/// The rows as sets of the finite-scored picks (what the select keeps).
fn live_sets(top: &[i32], scores: &[f32], n: usize, k: usize) -> Vec<Vec<i32>> {
    top.chunks(k)
        .enumerate()
        .map(|(r, row)| {
            let mut v: Vec<i32> = row
                .iter()
                .copied()
                .filter(|&j| scores[r * n + j as usize].is_finite())
                .collect();
            v.sort_unstable();
            v
        })
        .collect()
}

#[test]
fn cpu_top_k_keeps_ggmls_pools() {
    let (n, k) = (300usize, 40usize);
    let scores = topk_scores(n, 20, 0, 7);
    let (own, ggml, _, _) = topk_case(&mut cpu(), &scores, n as i64, k as i64, 0);
    assert_eq!(
        live_sets(&own, &scores, n, k),
        live_sets(&ggml, &scores, n, k)
    );
}

/// On CUDA: the kernel equals the CPU rule exactly (registers and the re-reading path, with
/// ties), keeps ggml's finite picks where nothing ties at the threshold, and its time against
/// ggml's at a 4K chunk of a 123K context (about 31K pools, 512 picks).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_top_k_matches_cpu_and_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    let cases = [
        (300usize, 40usize, 64usize, 0u32),
        (2048, 64, 512, 0),
        (2048, 64, 512, 50),
        (15000, 96, 512, 0),
        (31000, 4096, 512, 0),
        (31000, 64, 512, 1000),
        (70000, 48, 512, 0),
        (70000, 48, 512, 300),
    ];
    for (i, &(n, rows, k, levels)) in cases.iter().enumerate() {
        let scores = topk_scores(n, rows, levels, 11 + i as u64);
        let reps = if rows >= 1024 { 5 } else { 1 };
        let (own, ggml, t_own, t_ggml) = topk_case(&mut gpu, &scores, n as i64, k as i64, reps);
        let mut want = vec![0i32; k * rows];
        for (r, out) in want.chunks_mut(k).enumerate() {
            top_k_row(&scores[r * n..(r + 1) * n], out);
        }
        assert_eq!(
            own, want,
            "{n} x {rows}, k {k}, levels {levels}: kernel vs CPU rule"
        );
        // ggml's picks: the same but for ties at the threshold, which it takes in thread order.
        let (a, b) = (
            live_sets(&own, &scores, n, k),
            live_sets(&ggml, &scores, n, k),
        );
        for (r, (x, y)) in a.iter().zip(&b).enumerate() {
            let row = &scores[r * n..(r + 1) * n];
            let thr = row[want[r * k + k - 1] as usize];
            let odd: Vec<i32> = x
                .iter()
                .filter(|j| !y.contains(j))
                .chain(y.iter().filter(|j| !x.contains(j)))
                .copied()
                .collect();
            assert!(
                x.len() == y.len() && odd.iter().all(|&j| row[j as usize] == thr),
                "{n} x {rows}, levels {levels}, row {r}: live picks vs ggml differ off the threshold"
            );
        }
        println!(
            "{n} x {rows}, k {k}, levels {levels}: own {:.3} ms, ggml {:.3} ms",
            t_own * 1e3,
            t_ggml * 1e3
        );
    }
}

#[cfg(feature = "cuda")]
/// [`scores_exact`] and ggml's chain (pool mask, lightning indexer, bias add) at one shape on
/// `backend`, with the time of each over `reps` runs.
fn scores_exact_case(
    backend: &mut Backend,
    n_pool: i64,
    ng: i64,
    reps: usize,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let (d, nh) = (128i64, 4i64);
    let ctx = Context::new(32, 32);
    let q = ctx.new_tensor(types::F32, &[d, nh, ng]).as_input();
    let keys = ctx.new_tensor(types::F32, &[d, 1, n_pool]).as_input();
    let w = ctx.new_tensor(types::F32, &[nh, ng]).as_input();
    let vis = ctx.new_tensor(types::I32, &[2, ng]).as_input();
    let bias = ctx.new_tensor(types::F32, &[n_pool]).as_input();
    let op = scores_exact(&ctx, q, keys, w, vis, bias).as_output();
    let mask = pool_mask(&ctx, vis, n_pool);
    let reference = ctx
        .add(ctx.lightning_indexer(q, keys, w, mask), bias)
        .as_output();
    let (mut g_op, mut g_ref, mut all) = (ctx.new_graph(32), ctx.new_graph(32), ctx.new_graph(32));
    g_op.expand(op);
    g_ref.expand(reference);
    all.expand(op);
    all.expand(reference);
    ctx.alloc_graph(&all, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise((d * nh * ng) as usize, 3));
    keys.set_f32(&noise((d * n_pool) as usize, 4));
    w.set_f32(&vec![1.0 / (d as f32).sqrt(); (nh * ng) as usize]);
    let v: Vec<i32> = (0..ng)
        .flat_map(|t| [(t % 7) as i32, (n_pool - 3 * t).max(0) as i32])
        .collect();
    vis.set_bytes(&bytes(&v));
    // As the model's: index in the listing x 1e-30, 0 for some.
    let b: Vec<f32> = (0..n_pool)
        .map(|j| if j % 5 == 0 { 0.0 } else { j as f32 * 1e-30 })
        .collect();
    bias.set_f32(&b);
    let mut time = |g: &crate::ggml::Graph<'_>| {
        backend.compute(g).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(g).unwrap();
        }
        t0.elapsed().as_secs_f64() / reps.max(1) as f64
    };
    let (t_op, t_ref) = (time(&g_op), time(&g_ref));
    (op.to_f32(), reference.to_f32(), t_op, t_ref)
}

#[cfg(feature = "cuda")]
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// On CUDA: the one-op scores equal ggml's chain bit for bit (its tile kernel), and the CPU
/// implementation equals both; the time of each at a 4K chunk.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scores_exact_match_ggml_bitwise() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    for &(n_pool, ng) in &[(300i64, 40i64), (2049, 1000), (31000, 4096)] {
        let reps = if ng >= 1024 { 5 } else { 1 };
        let (op, reference, t_op, t_ref) = scores_exact_case(&mut gpu, n_pool, ng, reps);
        let differ = op
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{n_pool} pools x {ng} tokens: exact op {:.3} ms, ggml chain {:.3} ms, {differ} of {} differ",
            t_op * 1e3,
            t_ref * 1e3,
            op.len()
        );
        assert_eq!(differ, 0, "{n_pool} x {ng}: op vs ggml");
        if n_pool <= 2049 {
            let (cpu_op, _, _, _) = scores_exact_case(&mut cpu(), n_pool, ng, 0);
            assert!(same_bits(&cpu_op, &op), "{n_pool} x {ng}: CPU op vs CUDA");
        }
    }
}

/// A step's selection both ways, on `backend`: the indexer's chain (pool mask, lightning
/// indexer, pool bias, [`top_k`] of `n_top`, [`select`]) and [`select_all`], for sequences of
/// `pools[s]` complete pools (listed one after another, padded to `n_pool`) with `n_tok` tokens
/// each; 4 indexer heads of 128 dims as Flash-Next's. Returns both selections and the seconds
/// `reps` runs of each took.
fn pick_all_case(
    backend: &mut Backend,
    pools: &[usize],
    n_pool: usize,
    n_top: usize,
    n_tok: usize,
    reps: usize,
) -> (Vec<i32>, Vec<i32>, f64, f64) {
    let (d, nh, kpool) = (128i64, 4i64, 4usize);
    let ng = pools.len() * n_tok;
    let ctx = Context::new(64, 64);
    let q = ctx.new_tensor(types::F32, &[d, nh, ng as i64]).as_input();
    let keys = ctx
        .new_tensor(types::F32, &[d, 1, n_pool as i64])
        .as_input();
    let w = ctx.new_tensor(types::F32, &[nh, ng as i64]).as_input();
    let vis = ctx.new_tensor(types::I32, &[2, ng as i64]).as_input();
    let bias = ctx.new_tensor(types::F32, &[n_pool as i64]).as_input();
    let pidx = ctx
        .new_tensor(types::I32, &[kpool as i64, n_pool as i64])
        .as_input();
    let tail = ctx
        .new_tensor(types::I32, &[kpool as i64 - 1, ng as i64])
        .as_input();
    let pos = ctx.new_tensor(types::I32, &[ng as i64]).as_input();
    let mask = pool_mask(&ctx, vis, n_pool as i64);
    let score = ctx.add(ctx.lightning_indexer(q, keys, w, mask), bias);
    let top = top_k(&ctx, score, n_top as i64);
    let chain = select(&ctx, top, score, pidx, tail, pos).as_output();
    let all = select_all(&ctx, vis, n_top as i64, pidx, tail, pos).as_output();
    let (mut g_chain, mut g_all, mut both) =
        (ctx.new_graph(64), ctx.new_graph(64), ctx.new_graph(64));
    g_chain.expand(chain);
    g_all.expand(all);
    both.expand(chain);
    both.expand(all);
    ctx.alloc_graph(&both, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise(d as usize * nh as usize * ng, 1));
    keys.set_f32(&noise(d as usize * n_pool, 2));
    w.set_f32(&vec![1.0 / (d as f32).sqrt(); nh as usize * ng]);
    // The listing: each sequence's pools, its members' positions in the sequence.
    let (mut v, mut b, mut members, mut tails, mut ps) = (vec![], vec![], vec![], vec![], vec![]);
    for &n in pools {
        let first = b.len() as i32;
        for j in 0..n {
            b.push(j as f32 * 1e-30);
            members.extend((0..kpool).map(|k| (j * kpool + k) as i32));
        }
        let end = n * kpool + kpool - 1;
        for t in 0..n_tok {
            let p = (end - (n_tok - 1 - t)) as i32;
            v.extend([first, b.len() as i32]);
            tails.extend((0..kpool as i32 - 1).map(|k| p - k));
            ps.push(p);
        }
    }
    assert!(b.len() < n_pool);
    b.resize(n_pool, 0.0);
    members.resize(n_pool * kpool, i32::MAX);
    vis.set_bytes(&bytes(&v));
    bias.set_f32(&b);
    pidx.set_bytes(&bytes(&members));
    tail.set_bytes(&bytes(&tails));
    pos.set_bytes(&bytes(&ps));
    let mut time = |g: &crate::ggml::Graph<'_>| {
        backend.compute(g).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(g).unwrap();
        }
        t0.elapsed().as_secs_f64() / reps.max(1) as f64
    };
    let (t_chain, t_all) = (time(&g_chain), time(&g_all));
    (chain.to_i32(), all.to_i32(), t_chain, t_all)
}

/// (pools per sequence, listed pools, picks, tokens per sequence): one decode sequence, a
/// verify group that lists more pools than the budget though no token sees more, a full
/// budget, and a prompt chunk.
const PICK_ALL_CASES: [(&[usize], usize, usize, usize); 4] = [
    (&[100], 128, 128, 3),
    (&[300, 260], 576, 512, 3),
    (&[512, 7, 130, 64], 768, 512, 2),
    (&[500], 512, 512, 64),
];

#[test]
fn cpu_select_all_is_the_scored_selection() {
    for &(pools, n_pool, n_top, n_tok) in &PICK_ALL_CASES {
        let (chain, all, _, _) = pick_all_case(&mut cpu(), pools, n_pool, n_top, n_tok, 0);
        assert_eq!(chain, all, "{pools:?} over {n_pool}, {n_top} picks");
        // The first token selects its pools' members and its tail.
        assert_eq!(all[0] as usize, pools[0] * 4 + 3, "{pools:?}");
    }
}

/// On CUDA: the same rows as the CPU and as the scored chain, and the time of each (the chain
/// without the indexer query's projection and norm, which the early exit also drops).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_select_all_is_the_scored_selection() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    for &(pools, n_pool, n_top, n_tok) in &PICK_ALL_CASES {
        let (chain, all, t_chain, t_all) =
            pick_all_case(&mut gpu, pools, n_pool, n_top, n_tok, 200);
        let (cpu_chain, _, _, _) = pick_all_case(&mut cpu(), pools, n_pool, n_top, n_tok, 0);
        assert_eq!(chain, all, "{pools:?} over {n_pool}, {n_top} picks");
        assert_eq!(all, cpu_chain, "{pools:?} over {n_pool}: CUDA vs CPU");
        println!(
            "pick_all {pools:?} x {n_tok} over {n_pool}, {n_top} picks: chain {:.2} us, select_all {:.2} us",
            t_chain * 1e6,
            t_all * 1e6
        );
    }
}
