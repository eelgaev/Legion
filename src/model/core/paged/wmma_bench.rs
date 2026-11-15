//! [`WMMA_SPLIT`] (#1211) against the WMMA kernel it replaces: one graph with the op twice over
//! the same cache, once with the placeholder scratch (the WMMA kernel) and once with the split's
//! scratch, every output value compared bit for bit; and the time of either alone. Used by the
//! GPU tests (`wmma_split_tests.rs`) and by `legion paged-split-bench`, which runs them from a
//! release build on the GPU it is given (the CPU op ignores the scratch, so there the two agree
//! by construction and only the plumbing is checked). [`compare`] and [`time`] turn
//! [`WMMA_SPLIT`] on for the process (off by default).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{Hparams, KvPool};

/// An attention layer's shape: head size, query heads, kv heads.
pub type Heads = (usize, usize, usize);

/// One layer of `heads` with its K/V in a `KvPool` of `n_blocks` blocks of noise.
pub fn pool(backend: &Backend, (d, n_head, n_head_kv): Heads, n_blocks: usize) -> KvPool {
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).expect("KV pool");
    let (kp, vp) = kv.pools(0);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 41)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 42)));
    kv
}

/// A group: `n_tok` tokens per sequence after `n_past[s]` cached positions, through `tables`
/// (each padded to `width` entries, a ring of `ring` entries when > 0), sliding window `window`
/// (0: none), softmax scale `scale`.
pub struct Group<'t> {
    pub heads: Heads,
    pub n_tok: usize,
    pub n_past: &'t [usize],
    pub tables: &'t [Vec<u32>],
    pub width: usize,
    pub window: usize,
    pub ring: usize,
    pub scale: f32,
}

/// The split's scratch for sequences `seqs` at a time and the weights of `chunk` blocks at a
/// time (elements; [`wmma_split_plan`] makes the same choice back from it), or `None` for
/// the size [`wmma_split_floats`] gives the group.
fn scratch_len(g: &Group, room: Option<(usize, usize)>) -> usize {
    let (d, n_head, n_head_kv) = g.heads;
    match room {
        Some((seqs, chunk)) => {
            let (tiles, nr) = wmma_rows(n_head / n_head_kv, g.n_tok);
            seqs * tiles * n_head_kv * wmma_lane_floats(d, nr, g.width, chunk)
        }
        None => wmma_split_floats(d, n_head, n_head_kv, g.n_tok, g.n_past.len(), g.width),
    }
}

/// Whether the group takes the split (not more CTAs than it is for, [`wmma_split_floats`]).
pub fn splits(g: &Group) -> bool {
    scratch_len(g, None) > 0
}

/// Values of the split's output (every row) whose bits differ from the WMMA kernel's, of how
/// many, with the scratch of `room` ([`scratch_len`]). Panics on a non-finite output.
pub fn compare(
    backend: &mut Backend,
    kv: &KvPool,
    g: &Group,
    room: Option<(usize, usize)>,
) -> (usize, usize) {
    set_verify_split_wmma(true);
    let (kp, vp) = kv.pools(0);
    let (d, n_head, _) = g.heads;
    let ns = g.n_past.len();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, g.n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[g.width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let placeholder = ctx.new_tensor(types::F32, &[1, 1, 1, 1]).as_input();
    let len = scratch_len(g, room);
    assert!(len > 0, "the group does not take the split");
    let scratch = ctx
        .new_tensor(types::F32, &[len as i64, 1, 1, 1])
        .as_input();
    let attn = |s| {
        attention_tiered(
            &ctx, q, kp, vp, tab, past, s, g.scale, g.window, g.ring, None,
        )
    };
    let full = attn(placeholder).as_output();
    let split = attn(scratch).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(full);
    graph.expand(split);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .expect("graph allocation");
    fill(q, tab, past, g);
    // Garbage in the scratch: every value the split reads it writes first.
    scratch.set_f32(&vec![f32::NAN; len]);
    let launches = vsplit_launches();
    backend.compute(&graph).expect("compute");
    // A GPU that runs the WMMA kernel must have taken the split, not fallen back to that kernel
    // (no shared memory for it), or both outputs would be the WMMA kernel's.
    let gpu = backend.device().kind() != crate::ggml::DeviceKind::Cpu;
    assert!(
        !gpu || !split_ok() || vsplit_launches() > launches,
        "the split kernel did not run"
    );
    let (a, b) = (full.to_f32(), split.to_f32());
    assert!(
        a.iter().chain(&b).all(|v| v.is_finite()),
        "non-finite output"
    );
    let differ = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    (differ, a.len())
}

/// Milliseconds per run of the op alone, with the WMMA kernel (`split` false) or the split
/// (the scratch [`wmma_split_floats`] gives the group), best of `rounds` rounds of `reps` runs.
pub fn time(
    backend: &mut Backend,
    kv: &KvPool,
    g: &Group,
    split: bool,
    reps: usize,
    rounds: usize,
) -> f64 {
    set_verify_split_wmma(true);
    let (kp, vp) = kv.pools(0);
    let (d, n_head, _) = g.heads;
    let ns = g.n_past.len();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, g.n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[g.width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let len = if split {
        scratch_len(g, None).max(1)
    } else {
        1
    };
    let scratch = ctx
        .new_tensor(types::F32, &[len as i64, 1, 1, 1])
        .as_input();
    let out = attention_tiered(
        &ctx, q, kp, vp, tab, past, scratch, g.scale, g.window, g.ring, None,
    )
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .expect("graph allocation");
    fill(q, tab, past, g);
    backend.compute(&graph).expect("compute");
    let mut best = f64::INFINITY;
    for _ in 0..rounds.max(1) {
        let start = std::time::Instant::now();
        for _ in 0..reps.max(1) {
            backend.compute(&graph).expect("compute");
        }
        let _ = out.to_f32();
        best = best.min(start.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64);
    }
    best
}

/// The query (noise at a spread that gives peaked and flat rows), tables and positions.
pub(super) fn fill(q: Tensor, tab: Tensor, past: Tensor, g: &Group) {
    let d = g.heads.0;
    let qmul = 2.0 * g.scale * (d as f32).sqrt().min(16.0);
    q.set_f32(
        &noise(q.elements(), 31)
            .iter()
            .map(|x| qmul * x)
            .collect::<Vec<_>>(),
    );
    let mut t = Vec::with_capacity(g.width * g.tables.len());
    for (s, tb) in g.tables.iter().enumerate() {
        assert!(tb.len() <= g.width);
        assert!(g.ring > 0 || tb.len() * KV_BLOCK >= g.n_past[s] + g.n_tok);
        t.extend(tb.iter().map(|&x| x as i32));
        t.resize((s + 1) * g.width, 0);
    }
    tab.set_i32(&t);
    past.set_i32(&g.n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
}

/// The edge cases: every GQA ratio the kernels take, heads of 128, 256 and 512, 2 to 16 tokens,
/// shuffled tables (one padded past its blocks), contexts that end before, on and after block
/// boundaries, windows that start inside blocks, a ring, sequences run in turns and weights in
/// chunks. Returns (cases, values compared, values that differ), printing each failing case.
pub fn edges(backend: &mut Backend, shapes: &[Heads], n_toks: &[usize]) -> (usize, usize, usize) {
    let tables = [
        vec![5, 0, 9, 3, 7, 11, 13, 2, 17],
        vec![2, 10, 4],
        vec![8, 1, 6, 4, 12],
    ];
    let long: Vec<u32> = (0..150).map(|i| (i * 7 % 150) as u32 + 20).collect();
    let (mut cases, mut total, mut differ) = (0, 0, 0);
    for &heads in shapes {
        let (d, _, _) = heads;
        let kv = pool(backend, heads, 170);
        let rsd = 1.0 / (d as f32).sqrt();
        for &n_tok in n_toks {
            let mut run = |g: &Group, room, what: &str| {
                let (x, n) = compare(backend, &kv, g, room);
                cases += 1;
                total += n;
                differ += x;
                if x > 0 {
                    println!(
                        "DIFFER {x} of {n}: heads {heads:?}, {n_tok} tokens, n_past {:?}, window {}, \
                         ring {}, {what}",
                        g.n_past, g.window, g.ring
                    );
                }
            };
            for (n_past, window, scale) in [
                ([447, 3, 190], 0, rsd),
                ([512 - n_tok, 64 - n_tok, 255], 0, rsd),
                ([500, 129, 63], 100, rsd),
                ([449, 60, 250], 64, 0.08),
            ] {
                let g = Group {
                    heads,
                    n_tok,
                    n_past: &n_past,
                    tables: &tables,
                    width: 9,
                    window,
                    ring: 0,
                    scale,
                };
                if scratch_len(&g, None) == 0 {
                    continue; // more CTAs than the split takes
                }
                run(&g, None, "whole scratch");
                run(&g, Some((1, 9)), "one sequence per run");
                run(&g, Some((2, 9)), "two sequences per run");
            }
            // A ring of 4 entries under a window of 100 positions, past its first lap.
            let ring_tables = [vec![3, 1, 0, 2], vec![7, 5, 4, 6]];
            let g = Group {
                heads,
                n_tok,
                n_past: &[1000, 77],
                tables: &ring_tables,
                width: 4,
                window: 100,
                ring: 4,
                scale: rsd,
            };
            if scratch_len(&g, None) > 0 {
                run(&g, None, "ring");
            }
            // 150 blocks (9,600 positions): weights in chunks of 64 and 100 blocks.
            let g = Group {
                heads,
                n_tok,
                n_past: &[150 * KV_BLOCK - n_tok - 33],
                tables: std::slice::from_ref(&long),
                width: 150,
                window: 0,
                ring: 0,
                scale: rsd,
            };
            if scratch_len(&g, None) > 0 {
                run(&g, None, "long, whole");
                run(&g, Some((1, 64)), "long, chunks of 64");
                run(&g, Some((1, 100)), "long, chunks of 100");
            }
        }
    }
    (cases, total, differ)
}

/// A one-layer model's hyperparameters, for its KV pool.
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

/// Deterministic values in [-1, 1) (xorshift).
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

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut h = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(v.as_ptr(), h.as_mut_ptr(), v.len() as i64) };
    h.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
