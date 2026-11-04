//! The decode expert matvec's dot-product variants (`expert_mv::Dot`, #288) on a GPU: each must
//! give the per-user kernel's bits, for every expert format Qwen3.8-Flash-Next routes through
//! (gate/up Q4_K and Q5_K 2560 -> 640, down Q5_1 and Q8_0 640 -> 2560, the MTP head's Q8_0) and
//! every step size the op takes; `dot_timings` prints the time per op of each.
//!
//! `cargo test --release --features cuda --test expert_mv_dot_cuda -- --ignored --nocapture --test-threads 1`
//! Host-memory runs need a GPU that can read pinned host memory in place.

#![cfg(feature = "cuda")]

use legion::ggml::{
    Backend, BufferType, ComputeAllocator, Context, DeviceKind, backend_by_name, device_by_kind,
    quantize, types,
};
use legion::model::core::expert_mv::{self, Dot};

/// The modes, the per-user kernel (main's) first.
const MODES: [Dot; 4] = [Dot::PerUser, Dot::Auto, Dot::Once, Dot::Sub16];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// `n_expert` experts of `rows` x `k` random weights, quantized to `ty`.
fn experts(ty: u32, k: usize, rows: usize, n_expert: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let w: Vec<f32> = (0..k * rows * n_expert).map(|_| rng.f32() * 0.05).collect();
    quantize(ty, &w, k)
}

/// Distinct random ids per token; later tokens share half their ids with token 0, so an expert
/// has up to `n_tok` users (MTP verify tokens often share experts).
fn ids(n_used: usize, n_tok: usize, n_expert: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for t in 0..n_tok {
        let mut chosen: Vec<i32> = if t == 0 {
            vec![n_expert as i32 - 1]
        } else {
            Vec::new()
        };
        while chosen.len() < n_used {
            let e = if t > 0 && chosen.len() < n_used / 2 {
                out[chosen.len()]
            } else {
                (rng.next() % n_expert as u64) as i32
            };
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        out.extend(chosen);
    }
    out
}

struct Case {
    name: &'static str,
    ty: u32,
    k: usize,
    rows: usize,
    /// Gate and up through the SwiGLU op (else one matrix through `mul_mat_id`).
    glu: bool,
    /// One activation column per token (gate/up) or per pair (down).
    shared_x: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "gate/up Q4_K",
        ty: types::Q4_K,
        k: 2560,
        rows: 640,
        glu: true,
        shared_x: true,
    },
    Case {
        name: "gate/up Q5_K",
        ty: types::Q5_K,
        k: 2560,
        rows: 640,
        glu: true,
        shared_x: true,
    },
    // The MTP head's experts: Q8_0 rows too long for the SwiGLU op's tile.
    Case {
        name: "gate Q8_0",
        ty: types::Q8_0,
        k: 2560,
        rows: 640,
        glu: false,
        shared_x: true,
    },
    Case {
        name: "down Q5_1",
        ty: types::Q5_1,
        k: 640,
        rows: 2560,
        glu: false,
        shared_x: false,
    },
    Case {
        name: "down Q8_0",
        ty: types::Q8_0,
        k: 640,
        rows: 2560,
        glu: false,
        shared_x: false,
    }, // GLM-5.3-Flash's down experts in a few layers: 210-byte blocks, rows of 1680 bytes.
    Case {
        name: "down Q6_K",
        ty: types::Q6_K,
        k: 2048,
        rows: 4096,
        glu: false,
        shared_x: false,
    },
];

/// Copies of the expert stacks a timed graph holds, so each op reads other memory than the last
/// (as a decode step's layers do).
const LAYERS: usize = 8;

/// Where the experts live: VRAM (cache slots, `--expert-mv-device`), pinned host memory read in
/// place (missed experts).
fn placements() -> [(&'static str, BufferType); 2] {
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .unwrap();
    [
        ("vram", gpu.buffer_type()),
        ("host", gpu.host_buffer_type().unwrap()),
    ]
}

/// One case's graph (`layers` ops over their own copies of the stacks) built once, then run
/// over every id set (each: ids for every layer) under each dot mode of `plan` (indices into
/// [`MODES`]) in turn. Per entry of `plan`: layer 0's output of the last run and the mean time
/// per op.
#[allow(clippy::too_many_arguments)]
fn measure(
    gpu: &mut Backend,
    buft: BufferType,
    c: &Case,
    stacks: &[Vec<u8>],
    n_expert: usize,
    (n_used, n_tok): (usize, usize),
    x: &[f32],
    sets: &[Vec<Vec<i32>>],
    plan: &[usize],
) -> Vec<(Vec<f32>, f64)> {
    let layers = sets[0].len();
    let weights = Context::new(2 * layers + 1, 0);
    let shape = [c.k as i64, c.rows as i64, n_expert as i64];
    let ws: Vec<Vec<_>> = (0..layers)
        .map(|_| {
            stacks
                .iter()
                .map(|_| weights.new_tensor(c.ty, &shape))
                .collect()
        })
        .collect();
    // NaN bytes after the experts: a read past the last row shows.
    let poison = weights.new_tensor(types::I8, &[1 << 16]);
    weights.alloc_tensors(buft).unwrap();
    for l in &ws {
        for (w, q) in l.iter().zip(stacks) {
            w.set_bytes(q);
        }
    }
    poison.set_bytes(&vec![0xff; 1 << 16]);
    let n_x = if c.shared_x { 1 } else { n_used };
    let ctx = Context::new(1 + 8 * layers, 16 * layers + 8);
    let xt = ctx.new_tensor(types::F32, &[c.k as i64, n_x as i64, n_tok as i64]);
    let its: Vec<_> = (0..layers)
        .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
        .collect();
    ctx.alloc_tensors(gpu.buffer_type()).unwrap();
    xt.set_f32(x);
    let outs: Vec<_> = ws
        .iter()
        .zip(&its)
        .map(|(l, &it)| {
            if c.glu {
                expert_mv::glu_tiered(&ctx, l[0], l[1], xt, it, 0)
            } else {
                expert_mv::mul_mat_id(&ctx, l[0], xt, it)
            }
        })
        .collect();
    let mut graph = ctx.new_graph(16 * layers + 8);
    for o in &outs {
        graph.expand(*o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    let mut results = Vec::new();
    for &dot in plan {
        expert_mv::set_dot(MODES[dot]);
        let mut total = 0.0;
        for set in sets {
            for (it, ids) in its.iter().zip(set) {
                it.set_i32(ids);
            }
            let t = std::time::Instant::now();
            gpu.compute(&graph).unwrap();
            total += t.elapsed().as_secs_f64();
        }
        results.push((outs[0].to_f32(), total / (sets.len() * layers) as f64));
    }
    expert_mv::set_dot(Dot::default());
    results
}

fn stacks(c: &Case, n_expert: usize, seed: u64) -> Vec<Vec<u8>> {
    let n = if c.glu { 2 } else { 1 };
    (0..n)
        .map(|m| experts(c.ty, c.k, c.rows, n_expert, seed + m as u64))
        .collect()
}

/// How many values differ in their bits.
fn differ(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Every dot mode gives the per-user kernel's bits, in VRAM and host memory, for 1 to 16 tokens.
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn dot_modes_are_bit_identical() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let (n_expert, n_used) = (32usize, 10usize);
    for c in CASES {
        let st = stacks(c, n_expert, 3);
        for (where_, buft) in placements() {
            for n_tok in [1usize, 2, 3, 4, 5, 8, 16] {
                let n_x = if c.shared_x { 1 } else { n_used };
                let mut rng = Rng(9 + n_tok as u64);
                let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
                let set = vec![vec![ids(n_used, n_tok, n_expert, 5 + n_tok as u64)]];
                let r = measure(
                    &mut gpu,
                    buft,
                    c,
                    &st,
                    n_expert,
                    (n_used, n_tok),
                    &x,
                    &set,
                    &[0, 1, 2, 3],
                );
                let want = &r[0].0;
                assert!(want.iter().all(|v| v.is_finite()), "{} {where_}", c.name);
                for (dot, (got, _)) in r.iter().enumerate().skip(1) {
                    let n = differ(got, want);
                    assert_eq!(
                        n,
                        0,
                        "{} {where_} {n_tok} tokens, {:?}: {n} of {} differ",
                        c.name,
                        MODES[dot],
                        want.len()
                    );
                }
            }
        }
    }
}

/// The time per op of each dot mode, per format, placement and step size (1, 2, 3, 4, 8, 16
/// tokens; 10 experts per token). Modes run interleaved over `ROUNDS` rounds and the best round
/// counts, so clock and link noise hits them alike. `EXPERT_MV_TOKS=1,3` picks the step sizes,
/// `EXPERT_MV_WHERE=vram` one placement, and `EXPERT_MV_SAME=1` gives every token the same
/// experts (each expert then has exactly as many users as the step has tokens).
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn dot_timings() {
    const ROUNDS: usize = 5;
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let (n_expert, n_used) = (256usize, 10usize);
    let toks: Vec<usize> = std::env::var("EXPERT_MV_TOKS")
        .unwrap_or("1,2,3,4,8,16".into())
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let where_only = std::env::var("EXPERT_MV_WHERE").ok();
    let same = std::env::var("EXPERT_MV_SAME").is_ok_and(|v| v == "1");
    println!(
        "{:<14} {:<5} {:>6} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6} {:>6}",
        "case",
        "where",
        "tokens",
        "user µs",
        "auto µs",
        "once µs",
        "sub16 µs",
        "auto",
        "once",
        "sub16"
    );
    for c in CASES {
        let st = stacks(c, n_expert, 7);
        for (where_, buft) in placements() {
            if where_only.as_deref().is_some_and(|w| w != where_) {
                continue;
            }
            for &n_tok in &toks {
                let n_x = if c.shared_x { 1 } else { n_used };
                let mut rng = Rng(11);
                let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
                let sets: Vec<Vec<Vec<i32>>> = (0..10)
                    .map(|i| {
                        (0..LAYERS as u64)
                            .map(|l| {
                                let seed = 100 + 16 * i + l;
                                if same {
                                    ids(n_used, 1, n_expert, seed).repeat(n_tok)
                                } else {
                                    ids(n_used, n_tok, n_expert, seed)
                                }
                            })
                            .collect()
                    })
                    .collect();
                // A warm-up run, then the modes interleaved.
                let plan: Vec<usize> = std::iter::once(0)
                    .chain((0..ROUNDS).flat_map(|_| 0..MODES.len()))
                    .collect();
                let r = measure(
                    &mut gpu,
                    buft,
                    c,
                    &st,
                    n_expert,
                    (n_used, n_tok),
                    &x,
                    &sets,
                    &plan,
                );
                let mut best = [f64::MAX; MODES.len()];
                let mut outs: Vec<Vec<f32>> = vec![Vec::new(); MODES.len()];
                for (&dot, (o, t)) in plan.iter().zip(r).skip(1) {
                    best[dot] = best[dot].min(t);
                    outs[dot] = o;
                }
                for dot in 1..MODES.len() {
                    let n = differ(&outs[dot], &outs[0]);
                    assert_eq!(n, 0, "{} {:?}", c.name, MODES[dot]);
                }
                println!(
                    "{:<14} {:<5} {:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>6.3} {:>6.3} {:>6.3}",
                    c.name,
                    where_,
                    n_tok,
                    best[0] * 1e6,
                    best[1] * 1e6,
                    best[2] * 1e6,
                    best[3] * 1e6,
                    best[1] / best[0],
                    best[2] / best[0],
                    best[3] / best[0]
                );
            }
        }
    }
}
