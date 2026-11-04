//! Legion's decode MoE expert matvec (`src/cuda/expert_mv/`) on a GPU against ggml's
//! `mul_mat_id`, with the experts in VRAM and in host memory read in place (AC922 NVLink2 + ATS,
//! GB10 / Grace unified memory), on Qwen3.8-Flash-Next's expert shapes: gate/up Q4_K 2560 -> 640,
//! down Q5_1 640 -> 2560, 512 experts, 10 per token; 1 token (decode) and 4 tokens (MTP verify).
//!
//! `cargo test --release --features cuda --test expert_mv_cuda -- --ignored --nocapture --test-threads 1`
//! Host-memory runs need a GPU that can read pageable host memory (ATS or unified memory).

#![cfg(feature = "cuda")]

use legion::ggml::{
    Backend, BufferType, ComputeAllocator, Context, DeviceKind, backend_by_name, device_by_kind,
    quantize, types,
};
use legion::model::core::expert_mv;

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

/// `n_expert` experts of `rows` x `k` random weights, quantized to `ty` (random blocks for the
/// formats whose ggml quantizers want an importance matrix).
fn experts(ty: u32, k: usize, rows: usize, n_expert: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let block = match ty {
        types::IQ2_XXS => Some(66),
        types::IQ2_S => Some(82),
        _ => None,
    };
    if let Some(bytes) = block {
        // Any codes are valid; every F16 scale (at the block's start) set to 0.01.
        let mut out = vec![0u8; k / 256 * rows * n_expert * bytes];
        for (i, b) in out.iter_mut().enumerate() {
            *b = if i % bytes < 2 {
                [0x1f, 0x21][i % bytes]
            } else {
                rng.next() as u8
            };
        }
        return out;
    }
    let w: Vec<f32> = (0..k * rows * n_expert).map(|_| rng.f32() * 0.05).collect();
    quantize(ty, &w, k)
}

/// Distinct random ids per token; later tokens share half their ids with token 0 (as MTP verify
/// tokens often do).
fn ids(n_used: usize, n_tok: usize, n_expert: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for t in 0..n_tok {
        // Token 0 always takes the last expert, whose last row ends where the buffer does.
        let mut chosen: Vec<i32> = if t == 0 {
            vec![n_expert as i32 - 1]
        } else {
            Vec::new()
        };
        while chosen.len() < n_used {
            let e = if t > 0 && chosen.len() < n_used / 2 {
                out[chosen.len()] // half of each later token's experts repeat token 0's
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

/// ggml's `mul_mat_id` the way its mmvq computes it: in steps of at most 8 tokens (a bigger
/// step goes to MMQ, which rounds differently), the results concatenated. The reference for
/// the op, which computes every step size like mmvq.
fn ggml_mmvq<'a>(
    ctx: &'a Context,
    w: legion::ggml::Tensor<'a>,
    x: legion::ggml::Tensor<'a>,
    ids: legion::ggml::Tensor<'a>,
) -> legion::ggml::Tensor<'a> {
    let [k, n_x, n_tok, _] = x.ne();
    let n_used = ids.ne()[0];
    let mut out: Option<legion::ggml::Tensor<'a>> = None;
    for t0 in (0..n_tok).step_by(8) {
        let c = (n_tok - t0).min(8);
        let xs = ctx.view_3d(
            x,
            [k, n_x, c],
            [x.nb()[1], x.nb()[2]],
            t0 as usize * x.nb()[2],
        );
        let is = ctx.view_2d(ids, [n_used, c], ids.nb()[1], t0 as usize * ids.nb()[1]);
        let r = ctx.mul_mat_id(w, xs, is);
        out = Some(match out {
            Some(o) => ctx.concat(o, r, 2),
            None => r,
        });
    }
    out.expect("tokens")
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Legion,
    Ggml,
}

/// Copies of the expert stack a timed graph holds, so each op reads other memory than the last
/// (as a decode step's layers do) and launch / sync overhead spreads over many ops.
const LAYERS: usize = 8;

/// Runs `mul_mat_id` over `layers` copies of the expert stack in one graph (layer `l` with
/// `ids[l]`), first once with `ids`, then once per entry of `timed` (each entry: ids for every
/// layer). Returns layer 0's first result and the mean time per op over the timed runs.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &mut Backend,
    weights_buft: BufferType,
    path: Path,
    ty: u32,
    wq: &[u8],
    (k, rows, n_expert): (usize, usize, usize),
    x: &[f32],
    n_x: usize,
    ids: &[i32],
    (n_used, n_tok): (usize, usize),
    layers: usize,
    timed: &[Vec<Vec<i32>>],
) -> (Vec<f32>, f64) {
    let weights = Context::new(layers + 1, 0);
    let ws: Vec<_> = (0..layers)
        .map(|_| weights.new_tensor(ty, &[k as i64, rows as i64, n_expert as i64]))
        .collect();
    // NaN bytes right after the experts (in the same buffer): a kernel that reads past the
    // last row, as ggml's MMQ does by up to a K tile, gets NaN, as it would from a file mapping.
    let poison = weights.new_tensor(types::I8, &[1 << 16]);
    weights.alloc_tensors(weights_buft).unwrap();
    for w in &ws {
        w.set_bytes(wq);
    }
    poison.set_bytes(&vec![0xff; 1 << 16]);
    let ctx = Context::new(1 + 32 * layers, 64 * layers + 8);
    let xt = ctx.new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64]);
    let its: Vec<_> = (0..layers)
        .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
        .collect();
    ctx.alloc_tensors(gpu.buffer_type()).unwrap();
    xt.set_f32(x);
    for it in &its {
        it.set_i32(ids);
    }
    let outs: Vec<_> = ws
        .iter()
        .zip(&its)
        .map(|(&w, &it)| match path {
            Path::Legion => expert_mv::mul_mat_id(&ctx, w, xt, it),
            Path::Ggml => ggml_mmvq(&ctx, w, xt, it),
        })
        .collect();
    let mut graph = ctx.new_graph(64 * layers + 8);
    for o in &outs {
        graph.expand(*o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    gpu.compute(&graph).unwrap();
    let first = outs[0].to_f32();
    // Other experts every run, as in decode: nothing stays in L2 or migrates to the GPU.
    let mut total = 0.0;
    for set in timed {
        for (it, ids) in its.iter().zip(set) {
            it.set_i32(ids);
        }
        let t = std::time::Instant::now();
        gpu.compute(&graph).unwrap();
        total += t.elapsed().as_secs_f64();
    }
    (first, total / (timed.len() * layers).max(1) as f64)
}

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

struct Case {
    name: &'static str,
    ty: u32,
    k: usize,
    rows: usize,
    shared_x: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "gate/up Q4_K",
        ty: types::Q4_K,
        k: 2560,
        rows: 640,
        shared_x: true,
    },
    Case {
        name: "down Q5_1",
        ty: types::Q5_1,
        k: 640,
        rows: 2560,
        shared_x: false,
    },
    // Rows of 680 bytes (not a multiple of 16).
    Case {
        name: "down Q8_0",
        ty: types::Q8_0,
        k: 640,
        rows: 2560,
        shared_x: false,
    },
    Case {
        name: "gate/up Q5_K",
        ty: types::Q5_K,
        k: 2560,
        rows: 640,
        shared_x: true,
    }, // GLM-5.3-Flash's down experts in a few layers: 210-byte blocks, rows of 1680 bytes.
    Case {
        name: "down Q6_K",
        ty: types::Q6_K,
        k: 2048,
        rows: 4096,
        shared_x: false,
    },
    // Project Maya's GLM-5.3-Flash quants (#559): gate/up 4096 -> 2048, down 2048 -> 4096.
    Case {
        name: "gate/up IQ2_XXS",
        ty: types::IQ2_XXS,
        k: 4096,
        rows: 2048,
        shared_x: true,
    },
    Case {
        name: "gate/up Q2_K",
        ty: types::Q2_K,
        k: 4096,
        rows: 2048,
        shared_x: true,
    },
    Case {
        name: "gate/up IQ3_S",
        ty: types::IQ3_S,
        k: 4096,
        rows: 2048,
        shared_x: true,
    },
    Case {
        name: "down IQ2_S",
        ty: types::IQ2_S,
        k: 2048,
        rows: 4096,
        shared_x: false,
    },
    Case {
        name: "down IQ3_XXS",
        ty: types::IQ3_XXS,
        k: 2048,
        rows: 4096,
        shared_x: false,
    },
    Case {
        name: "down Q3_K",
        ty: types::Q3_K,
        k: 2048,
        rows: 4096,
        shared_x: false,
    },
    Case {
        name: "down IQ4_XS",
        ty: types::IQ4_XS,
        k: 2048,
        rows: 4096,
        shared_x: false,
    },
];

/// Where the experts live: VRAM, pinned host memory (as #37's host arena: read in place at the
/// link rate), plain pageable host memory (ATS on the AC922).
fn placements() -> [(&'static str, BufferType); 3] {
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .unwrap();
    [
        ("vram", gpu.buffer_type()),
        ("pinned", gpu.host_buffer_type().unwrap()),
        (
            "pageable",
            device_by_kind(DeviceKind::Cpu).unwrap().buffer_type(),
        ),
    ]
}

#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn matches_ggml_mul_mat_id() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let n_expert = 64;
    for c in CASES {
        let wq = experts(c.ty, c.k, c.rows, n_expert, 3);
        for &(n_used, n_tok) in &[(10, 1), (10, 4), (3, 2), (8, 8), (10, 48)] {
            let n_x = if c.shared_x { 1 } else { n_used };
            let mut rng = Rng(9);
            let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
            let idv = ids(n_used, n_tok, n_expert, 5);
            let shape = (c.k, c.rows, n_expert);
            let buft = gpu.buffer_type();
            let (want, _) = run(
                &mut gpu,
                buft,
                Path::Ggml,
                c.ty,
                &wq,
                shape,
                &x,
                n_x,
                &idv,
                (n_used, n_tok),
                1,
                &[],
            );
            for (where_, buft) in placements() {
                let (got, _) = run(
                    &mut gpu,
                    buft,
                    Path::Legion,
                    c.ty,
                    &wq,
                    shape,
                    &x,
                    n_x,
                    &idv,
                    (n_used, n_tok),
                    1,
                    &[],
                );
                let d = rel_diff(&got, &want);
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "{} {where_}: non-finite output",
                    c.name
                );
                assert!(
                    d < 1e-5,
                    "{} {where_} {n_used}x{n_tok}: relative difference {d:e}",
                    c.name
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn timings_against_ggml() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let n_expert = 256;
    println!(
        "{:<14} {:<8} {:>6} {:>6} {:>10} {:>10} {:>9} {:>9}",
        "case", "where", "tokens", "tile", "legion µs", "ggml µs", "legion GB/s", "ggml GB/s"
    );
    for c in CASES {
        let wq = experts(c.ty, c.k, c.rows, n_expert, 7);
        let expert_bytes = wq.len() / n_expert;
        for (where_, buft) in placements() {
            for n_tok in [1, 4] {
                let n_used = 10;
                let n_x = if c.shared_x { 1 } else { n_used };
                let mut rng = Rng(11);
                let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
                let idv = ids(n_used, n_tok, n_expert, 13);
                let mut distinct = idv.clone();
                distinct.sort();
                distinct.dedup();
                let gb = (distinct.len() * expert_bytes) as f64 / 1e9;
                let shape = (c.k, c.rows, n_expert);
                let timed: Vec<Vec<Vec<i32>>> = (0..10)
                    .map(|i| {
                        (0..LAYERS as u64)
                            .map(|l| ids(n_used, n_tok, n_expert, 100 + 16 * i + l))
                            .collect()
                    })
                    .collect();
                let (b, tb) = run(
                    &mut gpu,
                    buft,
                    Path::Ggml,
                    c.ty,
                    &wq,
                    shape,
                    &x,
                    n_x,
                    &idv,
                    (n_used, n_tok),
                    LAYERS,
                    &timed,
                );
                let tiles = std::env::var("EXPERT_MV_TILES").unwrap_or_default();
                for tile in tiles.split(',').map(|t| t.parse().unwrap_or(0)) {
                    expert_mv::set_tile_bytes(tile);
                    let (a, ta) = run(
                        &mut gpu,
                        buft,
                        Path::Legion,
                        c.ty,
                        &wq,
                        shape,
                        &x,
                        n_x,
                        &idv,
                        (n_used, n_tok),
                        LAYERS,
                        &timed,
                    );
                    println!(
                        "{:<14} {:<8} {:>6} {:>6} {:>10.1} {:>10.1} {:>9.1} {:>9.1}  diff {:.1e}",
                        c.name,
                        where_,
                        n_tok,
                        tile,
                        ta * 1e6,
                        tb * 1e6,
                        gb / ta,
                        gb / tb,
                        rel_diff(&a, &b)
                    );
                }
                expert_mv::set_tile_bytes(0);
            }
        }
    }
}

/// Ids into a tiered tensor of `n_slots` VRAM slots followed by `n_expert` host experts: each
/// token takes `n_used` distinct experts, each a VRAM slot with probability `hit`.
fn tiered_ids(
    n_used: usize,
    n_tok: usize,
    (n_slots, n_expert): (usize, usize),
    hit: f64,
    seed: u64,
) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for t in 0..n_tok {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < n_used {
            let e = if t > 0 && chosen.len() < n_used / 2 {
                out[chosen.len()]
            } else if (rng.next() % 1000) as f64 / 1000.0 < hit {
                (rng.next() % n_slots as u64) as i32
            } else {
                (n_slots as u64 + rng.next() % n_expert as u64) as i32
            };
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        out.extend(chosen);
    }
    out
}

/// Tiered expert tensors (#37: VRAM cache slots, then a pinned host arena's bytes mapped after
/// them) at several hit rates: Legion's op (VRAM experts through tiles or their own kernel,
/// `vram_kernel` 0 / 1) against ggml's `mul_mat_id`, which reads both tiers with mmvq, and for
/// gate/up shapes `glu_tiered` against ggml's mul_mat_id + silu + mul. Checks the results agree
/// and prints the time per op.
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn tiered_timings() {
    use legion::residency::cuda::{caps, host_arena, tiered};
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let (n_slots, n_expert, n_used) = (96usize, 256usize, 10usize);
    println!(
        "{:<14} {:>5} {:>6} µs: ggml, legion (VRAM in tiles), legion (VRAM kernel), legion (VRAM in place, prefetched); gate/up: the same for SwiGLU",
        "case", "hit", "tokens"
    );
    for c in CASES {
        let wq = experts(c.ty, c.k, c.rows, n_expert, 7);
        let per = wq.len() / n_expert;
        // LAYERS tiered tensors, slot s holding expert (s * 7) % n_expert.
        let hosts = Context::new(LAYERS, 0);
        let hs: Vec<_> = (0..LAYERS)
            .map(|_| hosts.new_tensor(c.ty, &[c.k as i64, c.rows as i64, n_expert as i64]))
            .collect();
        hosts.alloc_tensors(arena).unwrap();
        let tiers: Vec<Context> = hs
            .iter()
            .enumerate()
            .map(|(l, h)| {
                h.set_bytes(&wq);
                let tc = tiered(0, *h, n_slots, None, &format!("t{l}")).unwrap();
                let t = tc.tensor(&format!("t{l}")).unwrap();
                for s in 0..n_slots {
                    let e = s * 7 % n_expert;
                    t.set_bytes_at(s * per, &wq[e * per..(e + 1) * per]);
                }
                tc
            })
            .collect();
        let ws: Vec<_> = tiers
            .iter()
            .enumerate()
            .map(|(l, tc)| tc.tensor(&format!("t{l}")).unwrap())
            .collect();
        let hits: Vec<f64> = std::env::var("EXPERT_MV_HITS")
            .unwrap_or("0,0.6,0.9,1".into())
            .split(',')
            .map(|h| h.parse().unwrap())
            .collect();
        for &hit in &hits {
            for n_tok in [1usize, 4] {
                let n_x = if c.shared_x { 1 } else { n_used };
                let mut rng = Rng(11);
                let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
                let sets: Vec<Vec<Vec<i32>>> = (0..11)
                    .map(|i| {
                        (0..LAYERS as u64)
                            .map(|l| {
                                tiered_ids(
                                    n_used,
                                    n_tok,
                                    (n_slots, n_expert),
                                    hit,
                                    100 + 16 * i + l,
                                )
                            })
                            .collect()
                    })
                    .collect();
                let mut times = Vec::new();
                let mut results = Vec::new();
                // (glu, legion, vram_kernel; 9: 0 with set_prefetch): mul_mat_id, then
                // (gate/up) the SwiGLU of the stack with itself, ggml's mul_mat_id + silu + mul
                // against glu_tiered.
                let mut runs = vec![
                    (false, false, 0),
                    (false, true, 0),
                    (false, true, 1),
                    (false, true, 9),
                ];
                if c.shared_x {
                    runs.extend([
                        (true, false, 0),
                        (true, true, 0),
                        (true, true, 1),
                        (true, true, 9),
                    ]);
                }
                for &(glu, legion, vram_kernel) in &runs {
                    expert_mv::set_vram_kernel(vram_kernel % 9);
                    expert_mv::set_prefetch(vram_kernel == 9);
                    let ctx = Context::new(1 + 6 * LAYERS, 8 * LAYERS + 8);
                    let xt = ctx.new_tensor(types::F32, &[c.k as i64, n_x as i64, n_tok as i64]);
                    let its: Vec<_> = (0..LAYERS)
                        .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
                        .collect();
                    ctx.alloc_tensors(gpu.buffer_type()).unwrap();
                    xt.set_f32(&x);
                    let outs: Vec<_> = ws
                        .iter()
                        .zip(&its)
                        .map(|(&w, &it)| match (glu, legion) {
                            (false, true) => expert_mv::mul_mat_id_tiered(&ctx, w, xt, it, n_slots),
                            (false, false) => ctx.mul_mat_id(w, xt, it),
                            (true, true) => expert_mv::glu_tiered(&ctx, w, w, xt, it, n_slots),
                            (true, false) => ctx.mul(
                                ctx.silu(ctx.mul_mat_id(w, xt, it)),
                                ctx.mul_mat_id(w, xt, it),
                            ),
                        })
                        .collect();
                    let mut graph = ctx.new_graph(8 * LAYERS + 8);
                    for o in &outs {
                        graph.expand(*o);
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    let mut total = 0.0;
                    for (i, set) in sets.iter().enumerate() {
                        for (it, ids) in its.iter().zip(set) {
                            it.set_i32(ids);
                        }
                        let t = std::time::Instant::now();
                        gpu.compute(&graph).unwrap();
                        if i > 0 {
                            total += t.elapsed().as_secs_f64();
                        }
                    }
                    times.push(total / ((sets.len() - 1) * LAYERS) as f64);
                    results.push(outs[0].to_f32());
                }
                expert_mv::set_vram_kernel(0);
                expert_mv::set_prefetch(false);
                for (i, r) in results.iter().enumerate() {
                    let want = &results[if i < 4 { 0 } else { 4 }];
                    assert!(r.iter().all(|v| v.is_finite()), "non-finite output");
                    let d = rel_diff(r, want);
                    assert!(d < 1e-5, "{} hit {hit}: relative difference {d:e}", c.name);
                }
                let cols: Vec<String> = times.iter().map(|t| format!("{:>8.1}", t * 1e6)).collect();
                println!(
                    "{:<14} {:>5.2} {:>6} {}",
                    c.name,
                    hit,
                    n_tok,
                    cols.join(" ")
                );
            }
        }
        drop(tiers);
    }
}

/// `glu_tiered` (`silu(gate x) * (up x)` in one op) against ggml's two `mul_mat_id`s, silu and
/// mul, with gate and up distinct stacks in VRAM and in pinned host memory.
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn glu_matches_ggml() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let n_expert = 64;
    for c in CASES.iter().filter(|c| c.shared_x) {
        let gq = experts(c.ty, c.k, c.rows, n_expert, 3);
        let uq = experts(c.ty, c.k, c.rows, n_expert, 4);
        for (where_, buft) in placements().into_iter().take(2) {
            for &(n_used, n_tok) in &[(10, 1), (10, 4), (8, 8), (10, 48)] {
                let mut rng = Rng(9);
                let x: Vec<f32> = (0..c.k * n_tok).map(|_| rng.f32()).collect();
                let idv = ids(n_used, n_tok, n_expert, 5);
                let weights = Context::new(2, 0);
                let shape = [c.k as i64, c.rows as i64, n_expert as i64];
                let (g, u) = (
                    weights.new_tensor(c.ty, &shape),
                    weights.new_tensor(c.ty, &shape),
                );
                weights.alloc_tensors(buft).unwrap();
                g.set_bytes(&gq);
                u.set_bytes(&uq);
                let ctx = Context::new(64, 128);
                let xt = ctx.new_tensor(types::F32, &[c.k as i64, 1, n_tok as i64]);
                let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
                ctx.alloc_tensors(gpu.buffer_type()).unwrap();
                xt.set_f32(&x);
                it.set_i32(&idv);
                let ours = expert_mv::glu_tiered(&ctx, g, u, xt, it, 0);
                let theirs = ctx.mul(
                    ctx.silu(ggml_mmvq(&ctx, g, xt, it)),
                    ggml_mmvq(&ctx, u, xt, it),
                );
                let mut graph = ctx.new_graph(128);
                graph.expand(ours);
                graph.expand(theirs);
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                gpu.compute(&graph).unwrap();
                let (a, b) = (ours.to_f32(), theirs.to_f32());
                assert!(
                    a.iter().all(|v| v.is_finite()),
                    "{} {where_}: non-finite",
                    c.name
                );
                let d = rel_diff(&a, &b);
                assert!(
                    d < 1e-5,
                    "{} {where_} {n_used}x{n_tok}: relative difference {d:e}",
                    c.name
                );
            }
        }
    }
}

/// Ids outside the expert tensor (-1, `n_expert`, `i32::MAX`: a corrupt or padded routing
/// entry) are skipped by both kernels instead of reading before or past the weights (an MMU
/// fault, Xid 31, on the GPU), and every other pair's output is unchanged. Run it under
/// `compute-sanitizer --tool memcheck` as well.
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn ids_out_of_range_are_skipped() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let n_expert = 16;
    let (n_used, n_tok) = (4, 3);
    let good = [3, 0, 15, 7, 3, 9, 1, 2, 15, 4, 5, 6];
    let bad = [
        3,
        -1,
        15,
        7,
        3,
        9,
        n_expert as i32,
        2,
        i32::MAX,
        4,
        5,
        i32::MIN,
    ];
    for c in CASES {
        let wq = experts(c.ty, c.k, c.rows, n_expert, 3);
        let n_x = if c.shared_x { 1 } else { n_used };
        let mut rng = Rng(9);
        let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
        let shape = (c.k, c.rows, n_expert);
        for (where_, buft) in placements() {
            let mut out = |ids: &[i32]| {
                run(
                    &mut gpu,
                    buft,
                    Path::Legion,
                    c.ty,
                    &wq,
                    shape,
                    &x,
                    n_x,
                    ids,
                    (n_used, n_tok),
                    1,
                    &[],
                )
                .0
            };
            let (want, got) = (out(&good), out(&bad));
            for (pair, (&g, &b)) in good.iter().zip(&bad).enumerate() {
                if g == b {
                    let r = pair * c.rows..(pair + 1) * c.rows;
                    assert_eq!(got[r.clone()], want[r], "{} {where_}: pair {pair}", c.name);
                }
            }
        }
    }
}

/// What `--expert-mv-device` relies on (#309): for Qwen3.8-Flash-Next's gate/up experts
/// (Q4_K, 2560 -> 640) in GPU memory, the SwiGLU op of a 2- to 8-token step equals ggml's
/// `silu(mul_mat_id(gate)) * mul_mat_id(up)` bit for bit (a warp per row, ggml's lanes and
/// sums). That is per op only: end to end, greedy text can still differ (#472).
#[test]
#[ignore = "needs a CUDA GPU"]
fn glu_on_device_is_ggml_bit_for_bit() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let (k, rows, n_expert, n_used) = (2560usize, 640usize, 64usize, 10usize);
    let gq = experts(types::Q4_K, k, rows, n_expert, 3);
    let uq = experts(types::Q4_K, k, rows, n_expert, 4);
    for n_tok in 2..=8usize {
        let mut rng = Rng(9);
        let x: Vec<f32> = (0..k * n_tok).map(|_| rng.f32()).collect();
        let idv = ids(n_used, n_tok, n_expert, 5);
        let weights = Context::new(2, 0);
        let shape = [k as i64, rows as i64, n_expert as i64];
        let (g, u) = (
            weights.new_tensor(types::Q4_K, &shape),
            weights.new_tensor(types::Q4_K, &shape),
        );
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        g.set_bytes(&gq);
        u.set_bytes(&uq);
        let ctx = Context::new(64, 128);
        let xt = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
        ctx.alloc_tensors(gpu.buffer_type()).unwrap();
        xt.set_f32(&x);
        it.set_i32(&idv);
        let ours = expert_mv::glu_tiered(&ctx, g, u, xt, it, 0);
        let theirs = ctx.mul(
            ctx.silu(ctx.mul_mat_id(g, xt, it)),
            ctx.mul_mat_id(u, xt, it),
        );
        let mut graph = ctx.new_graph(128);
        graph.expand(ours);
        graph.expand(theirs);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        gpu.compute(&graph).unwrap();
        let (a, b) = (ours.to_f32(), theirs.to_f32());
        let differ = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(differ, 0, "{n_tok} tokens: {differ} of {} differ", a.len());
    }
}

/// `--expert-mv-prefetch` (#416, after Strata-AC922's S28): the tile kernel reads VRAM experts
/// in place with each lane's loads up front, in the tile path's lanes and order, so its outputs
/// equal the default's bit for bit: VRAM-only and mixed steps, 1-8 tokens (in place for 1-2,
/// the tiles after), plain and SwiGLU, every expert type the op takes.
#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn prefetch_is_bitwise_tiles() {
    use legion::residency::cuda::{caps, host_arena, tiered};
    expert_mv::set_enabled(true);
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let (n_slots, n_expert, n_used) = (24usize, 64usize, 10usize);
    for c in CASES {
        let wq = experts(c.ty, c.k, c.rows, n_expert, 7);
        let per = wq.len() / n_expert;
        let hosts = Context::new(1, 0);
        let h = hosts.new_tensor(c.ty, &[c.k as i64, c.rows as i64, n_expert as i64]);
        hosts.alloc_tensors(arena).unwrap();
        h.set_bytes(&wq);
        let tc = tiered(0, h, n_slots, None, "t").unwrap();
        let w = tc.tensor("t").unwrap();
        for s in 0..n_slots {
            let e = s * 7 % n_expert;
            w.set_bytes_at(s * per, &wq[e * per..(e + 1) * per]);
        }
        for hit in [1.0, 0.6] {
            for n_tok in 1..=8usize {
                for glu in [false, true] {
                    if glu && !c.shared_x {
                        continue;
                    }
                    let n_x = if c.shared_x { 1 } else { n_used };
                    let mut rng = Rng(3 + n_tok as u64);
                    let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
                    let id = tiered_ids(n_used, n_tok, (n_slots, n_expert), hit, 40 + n_tok as u64);
                    let mut outs = Vec::new();
                    for pf in [false, true] {
                        expert_mv::set_prefetch(pf);
                        let ctx = Context::new(8, 16);
                        let xt =
                            ctx.new_tensor(types::F32, &[c.k as i64, n_x as i64, n_tok as i64]);
                        let it = ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]);
                        ctx.alloc_tensors(gpu.buffer_type()).unwrap();
                        xt.set_f32(&x);
                        it.set_i32(&id);
                        let o = if glu {
                            expert_mv::glu_tiered(&ctx, w, w, xt, it, n_slots)
                        } else {
                            expert_mv::mul_mat_id_tiered(&ctx, w, xt, it, n_slots)
                        };
                        let mut graph = ctx.new_graph(16);
                        graph.expand(o);
                        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                            .unwrap();
                        gpu.compute(&graph).unwrap();
                        outs.push(o.to_f32());
                    }
                    expert_mv::set_prefetch(false);
                    assert!(outs[0].iter().all(|v| v.is_finite()), "non-finite output");
                    let differ = outs[0]
                        .iter()
                        .zip(&outs[1])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differ, 0,
                        "{} hit {hit} tokens {n_tok} glu {glu}: {differ} values differ",
                        c.name
                    );
                }
            }
        }
        drop(tc);
    }
}
