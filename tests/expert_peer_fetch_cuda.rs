//! A tiered layer's expert matvecs with the running GPU's NVLink peer fetching a share of the
//! missed experts (#252, `legion::model::core::expert_mv::peer`). The shapes and ids of
//! `tests/expert_mv_overlap_cuda.rs` (Qwen3.8-Flash-Next: gate/up Q4_K 2560 -> 640, down Q5_1
//! 640 -> 2560, 10 experts per token, 1 and 3 tokens, several hit rates) on CUDA0, its peer
//! CUDA1: every arrangement's outputs must be the plain ops' bits, and the time per layer is
//! printed for each.
//!
//! `cargo test --release --features cuda --test expert_peer_fetch_cuda -- --ignored --nocapture --test-threads 1`
//! Needs two GPUs on one host NUMA node with peer access that can read pinned host memory in
//! place (an AC922 pair: `CUDA_VISIBLE_DEVICES=0,1`). `PEER_HITS` (default `0,0.6,0.8,0.9`)
//! picks the hit rates. Columns: `plain`: unsplit; `side`: #253's side stream (8 CTAs); `peer`:
//! the peer alone takes every other missed expert (24 CTAs), `peer-c<n>`: on n CTAs, `peer-2/3`:
//! two of every three; `both`: the side stream copies the even slots, the peer the odd ones;
//! `peer-busy`: the switch on but the peer not idle (the plain ops).

#![cfg(feature = "cuda")]

use legion::ggml::{ComputeAllocator, Context, backend_by_name, quantize, types};
use legion::model::core::expert_mv::{self, overlap, peer};

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

fn experts(ty: u32, k: usize, rows: usize, n_expert: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let w: Vec<f32> = (0..k * rows * n_expert).map(|_| rng.f32() * 0.05).collect();
    quantize(ty, &w, k)
}

/// `n_used` distinct ids per token over a tiered tensor of `n_slots` VRAM slots then `n_expert`
/// host experts, each a slot with probability `hit`; later tokens repeat half of token 0's.
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

const LAYERS: usize = 8;

/// One arrangement: label, side stream, peer, peer CTAs (0: default), peer share `(take,
/// every)` (0, 0: default), peer idle.
type Run = (&'static str, bool, bool, usize, (usize, usize), bool);

#[test]
#[ignore = "needs two CUDA GPUs on one node with peer access that can read host memory"]
fn peer_fetch_is_bitwise_and_timed() {
    use legion::residency::cuda::{caps, host_arena, share_host_part, tiered};
    expert_mv::set_enabled(true);
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let node = |d| caps(d).map_or(-1, |c| c.host_numa_id);
    assert_eq!(node(0), node(1), "CUDA0 and CUDA1 on different host nodes");
    let arena = host_arena(node(0).max(0) as u32, &[0, 1]).expect("host arena");
    let (n_slots, n_expert, n_used) = (96usize, 256usize, 10usize);
    let (k, mid) = (2560usize, 640usize);
    let gq = experts(types::Q4_K, k, mid, n_expert, 7);
    let uq = experts(types::Q4_K, k, mid, n_expert, 8);
    let dq = experts(types::Q5_1, mid, k, n_expert, 9);
    // LAYERS tiered gate, up and down stacks; slot s holds expert (s * 7) % n_expert.
    let hosts = Context::new(3 * LAYERS, 0);
    let hs: Vec<_> = (0..LAYERS)
        .flat_map(|_| {
            [
                hosts.new_tensor(types::Q4_K, &[k as i64, mid as i64, n_expert as i64]),
                hosts.new_tensor(types::Q4_K, &[k as i64, mid as i64, n_expert as i64]),
                hosts.new_tensor(types::Q5_1, &[mid as i64, k as i64, n_expert as i64]),
            ]
        })
        .collect();
    hosts.alloc_tensors(arena).unwrap();
    let tiers: Vec<Context> = hs
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let wq = [&gq, &uq, &dq][i % 3];
            let per = wq.len() / n_expert;
            h.set_bytes(wq);
            let tc = tiered(0, *h, n_slots, None, &format!("t{i}")).unwrap();
            let t = tc.tensor(&format!("t{i}")).unwrap();
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
        .map(|(i, tc)| tc.tensor(&format!("t{i}")).unwrap())
        .collect();
    let shared: Vec<usize> = ws.iter().map(|&t| share_host_part(t, 1).unwrap()).collect();
    peer::connect(0, 1, shared).expect("CUDA1 as CUDA0's peer");
    let hits: Vec<f64> = std::env::var("PEER_HITS")
        .unwrap_or("0,0.6,0.8,0.9".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let runs: Vec<Run> = vec![
        ("plain", false, false, 0, (0, 0), false),
        ("side", true, false, 0, (0, 0), false),
        ("peer", false, true, 0, (0, 0), true),
        ("peer-c12", false, true, 12, (0, 0), true),
        ("peer-c48", false, true, 48, (0, 0), true),
        ("peer-2/3", false, true, 0, (2, 3), true),
        ("both", true, true, 0, (0, 0), true),
        ("peer-busy", false, true, 0, (0, 0), false),
    ];
    let head: Vec<&str> = runs.iter().map(|r| r.0).collect();
    println!(
        "{:>5} {:>6} µs per layer (gate/up + down): {}",
        "hit",
        "tokens",
        head.join(" ")
    );
    for &hit in &hits {
        for n_tok in [1usize, 3] {
            let mut rng = Rng(11);
            let x: Vec<f32> = (0..k * n_tok).map(|_| rng.f32()).collect();
            let sets: Vec<Vec<Vec<i32>>> = (0..11)
                .map(|i| {
                    (0..LAYERS as u64)
                        .map(|l| {
                            tiered_ids(n_used, n_tok, (n_slots, n_expert), hit, 100 + 16 * i + l)
                        })
                        .collect()
                })
                .collect();
            let mut times = Vec::new();
            let mut results: Vec<Vec<f32>> = Vec::new();
            for &(_, side, with_peer, c, (take, every), idle) in &runs {
                overlap::set_enabled(side);
                peer::set_enabled(with_peer);
                peer::set_ctas(c);
                peer::set_share(take, every);
                let ctx = Context::new(1 + 8 * LAYERS, 12 * LAYERS + 8);
                let xt = ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]);
                let its: Vec<_> = (0..LAYERS)
                    .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
                    .collect();
                ctx.alloc_tensors(gpu.buffer_type()).unwrap();
                xt.set_f32(&x);
                let outs: Vec<_> = its
                    .iter()
                    .enumerate()
                    .map(|(l, &it)| {
                        let (g, u, d) = (ws[3 * l], ws[3 * l + 1], ws[3 * l + 2]);
                        if side || with_peer {
                            overlap::glu_down_tiered(&ctx, [g, u, d], xt, it, n_slots)
                                .expect("split fits")
                        } else {
                            let act = expert_mv::glu_tiered(&ctx, g, u, xt, it, n_slots);
                            expert_mv::mul_mat_id_tiered(&ctx, d, act, it, n_slots)
                        }
                    })
                    .collect();
                let mut graph = ctx.new_graph(12 * LAYERS + 8);
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
                    {
                        let _idle = peer::Idle::new(idle);
                        gpu.compute(&graph).unwrap();
                    }
                    if i > 0 {
                        total += t.elapsed().as_secs_f64();
                    }
                }
                times.push(total / ((sets.len() - 1) * LAYERS) as f64);
                // Every layer's output of the last set.
                results.push(outs.iter().flat_map(|o| o.to_f32()).collect());
            }
            overlap::set_enabled(false);
            peer::set_enabled(false);
            peer::set_ctas(0);
            peer::set_share(0, 0);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert!(
                results[0].iter().all(|v| v.is_finite()),
                "non-finite output"
            );
            for (r, run) in results.iter().zip(&head).skip(1) {
                assert!(
                    bits(r) == bits(&results[0]),
                    "hit {hit} tokens {n_tok} {run}: not the plain bits"
                );
            }
            let cols: Vec<String> = times.iter().map(|t| format!("{:>9.1}", t * 1e6)).collect();
            println!("{:>5.2} {:>6} {}", hit, n_tok, cols.join(" "));
        }
    }
    drop(tiers);
}
