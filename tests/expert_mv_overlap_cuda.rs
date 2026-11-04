//! A tiered layer's expert matvecs with the missed experts staged into VRAM on a side stream
//! while the cached ones are computed (#253, `legion::model::core::expert_mv::overlap`). On
//! Qwen3.8-Flash-Next's expert shapes (gate/up Q4_K 2560 -> 640, down Q5_1 640 -> 2560), 10
//! experts per token, 1 token (decode) and 3 (MTP k = 2 verify), several hit rates: the split's
//! outputs must be the unsplit ops' bits, and the time per layer is printed for each copy CTA
//! count, priority and number of staged experts, and with the VRAM experts read in place
//! (`--expert-mv-prefetch`, #416).
//!
//! `cargo test --release --features cuda --test expert_mv_overlap_cuda -- --ignored --nocapture --test-threads 1`
//! Needs a GPU that can read pinned host memory in place. `OVERLAP_HITS` (default
//! `0,0.6,0.7,0.9,1`) and `OVERLAP_CTAS` (default `8,16,32`) pick the grid. Columns: `plain`:
//! unsplit; `c<n>`: split, the copy on n CTAs; `c16-lo`: the side stream at the compute stream's
//! priority; `c16-s4` / `c16-all`: 4 / every missed expert staged (default 16); `V-only` /
//! `H-only`: unsplit, the VRAM / host experts alone (the others' ids made -1); `pf`, `c16-pf`,
//! `V-only-pf`: `plain`, `c16`, `V-only` with `expert_mv::set_prefetch`.

#![cfg(feature = "cuda")]

use legion::ggml::{ComputeAllocator, Context, backend_by_name, quantize, types};
use legion::model::core::expert_mv::{self, overlap};

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

fn list<T: std::str::FromStr>(var: &str, default: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    std::env::var(var)
        .unwrap_or(default.into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect()
}

#[test]
#[ignore = "needs a CUDA GPU that can read host memory"]
fn split_is_bitwise_and_timed() {
    use legion::residency::cuda::{caps, host_arena, tiered};
    // As the engine does for models it runs on CUDA.
    expert_mv::set_enabled(true);
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
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
    let hits: Vec<f64> = list("OVERLAP_HITS", "0,0.6,0.7,0.9,1");
    let ctas: Vec<usize> = list("OVERLAP_CTAS", "8,16,32");
    // (label, split, copy CTAs, high priority side stream, staged experts, only one tier's
    // experts: 0 both, 1 VRAM (host ids made -1, which the kernels skip), 2 host; VRAM experts
    // read in place, loads up front)
    let mut runs = vec![("plain".to_string(), false, 0, true, 0, 0, false)];
    for &c in &ctas {
        runs.push((format!("c{c}"), true, c, true, 0, 0, false));
    }
    runs.push(("c16-lo".into(), true, 16, false, 0, 0, false));
    runs.push(("c16-s4".into(), true, 16, true, 4, 0, false));
    runs.push(("c16-all".into(), true, 16, true, 512, 0, false));
    runs.push(("V-only".into(), false, 0, true, 0, 1, false));
    runs.push(("H-only".into(), false, 0, true, 0, 2, false));
    runs.push(("pf".into(), false, 0, true, 0, 0, true));
    runs.push(("c16-pf".into(), true, 16, true, 0, 0, true));
    runs.push(("V-only-pf".into(), false, 0, true, 0, 1, true));
    let head: Vec<String> = runs.iter().map(|r| r.0.clone()).collect();
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
            for &(_, split, c, high, staged, only, pf) in &runs {
                expert_mv::set_prefetch(pf);
                overlap::set_enabled(split);
                overlap::set_ctas(c);
                overlap::set_high_priority(high);
                overlap::set_stage_pairs(staged);
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
                        if split {
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
                        let keep = |&e: &i32| match only {
                            1 => (e as usize) < n_slots,
                            2 => (e as usize) >= n_slots,
                            _ => true,
                        };
                        let ids: Vec<i32> =
                            ids.iter().map(|e| if keep(e) { *e } else { -1 }).collect();
                        it.set_i32(&ids);
                    }
                    let t = std::time::Instant::now();
                    gpu.compute(&graph).unwrap();
                    if i > 0 {
                        total += t.elapsed().as_secs_f64();
                    }
                }
                times.push(total / ((sets.len() - 1) * LAYERS) as f64);
                // Every layer's output of the last set.
                results.push(outs.iter().flat_map(|o| o.to_f32()).collect());
            }
            expert_mv::set_prefetch(false);
            overlap::set_enabled(false);
            overlap::set_ctas(0);
            overlap::set_high_priority(true);
            overlap::set_stage_pairs(0);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert!(
                results[0].iter().all(|v| v.is_finite()),
                "non-finite output"
            );
            for ((r, run), _) in results
                .iter()
                .zip(&head)
                .zip(&runs)
                .skip(1)
                .filter(|(_, r)| r.5 == 0)
            {
                assert!(
                    bits(r) == bits(&results[0]),
                    "hit {hit} tokens {n_tok} ctas {run}: not the unsplit bits"
                );
            }
            let at = |label: &str| head.iter().position(|h| h == label).expect("run");
            assert!(
                bits(&results[at("V-only-pf")]) == bits(&results[at("V-only")]),
                "hit {hit} tokens {n_tok}: VRAM experts read in place are not the tiles' bits"
            );
            let cols: Vec<String> = times.iter().map(|t| format!("{:>7.1}", t * 1e6)).collect();
            println!("{:>5.2} {:>6} {}", hit, n_tok, cols.join(" "));
        }
    }
    drop(tiers);
}
