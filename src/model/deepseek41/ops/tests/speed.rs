//! Kernel times of the attention-side ops (the indexer's scores, the candidate mask, the top-k
//! and the sparse attention) at the real model's sizes, one sequence, random inputs (ignored,
//! needs a GPU):
//!
//! `cargo test --release --features cuda --lib ds4_kernel_speed -- --ignored --nocapture`
//!
//! Per case and op: the wall time of one op (graphs of the op alone, run back to back) and an
//! FNV-1a hash of its output bytes. The inputs are the same in every build, so equal hashes
//! between two builds mean the same output bits.

use std::time::Instant;

use super::*;

/// One sequence's case: `ng` queries ending at position `end - 1`, compression `ratio`.
struct Case {
    name: &'static str,
    end: usize,
    ng: usize,
    ratio: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "decode 4K r1",
        end: 4096,
        ng: 1,
        ratio: 1,
    },
    Case {
        name: "decode 32K r1",
        end: 32768,
        ng: 1,
        ratio: 1,
    },
    Case {
        name: "decode 32K r2",
        end: 32768,
        ng: 1,
        ratio: 2,
    },
    Case {
        name: "decode 128K r1",
        end: 131072,
        ng: 1,
        ratio: 1,
    },
    Case {
        name: "2 queries 32K r1",
        end: 32768,
        ng: 2,
        ratio: 1,
    },
    Case {
        name: "4 queries 32K r1",
        end: 32768,
        ng: 4,
        ratio: 1,
    },
    Case {
        name: "8 queries 32K r1",
        end: 32768,
        ng: 8,
        ratio: 1,
    },
    Case {
        name: "16 queries 32K r1",
        end: 32768,
        ng: 16,
        ratio: 1,
    },
    Case {
        name: "prefill 1K@8K r1",
        end: 8192,
        ng: 1024,
        ratio: 1,
    },
    Case {
        name: "prefill 1K@32K r1",
        end: 32768,
        ng: 1024,
        ratio: 1,
    },
    Case {
        name: "prefill 1K@32K r2",
        end: 32768,
        ng: 1024,
        ratio: 2,
    },
];

const D: usize = 512;
const NH: usize = 64;
const IDX_D: usize = 128;
const IDX_H: usize = 32;
const TOP_K: usize = 512;
const WINDOW: usize = 128;
const BS: usize = 8;
const TB: usize = 2048;
/// The kv source's cache row: window latent | compressed latent | index key.
const ROW: usize = 2 * D + IDX_D;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| f32_to_f16(*x).to_ne_bytes())
        .collect()
}

/// Input data for [`time_op`].
enum In {
    F32(Vec<f32>),
    F16(Vec<u8>),
    I32(Vec<i32>),
}

/// `f` over inputs of these shapes and values on `gpu`: (µs per run, output bytes).
fn time_op(
    gpu: &mut Backend,
    inputs: &[(&[i64], &In)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> (f64, Vec<u8>) {
    let ctx = Context::new(32, 16);
    let ts: Vec<Tensor<'_>> = inputs
        .iter()
        .map(|(ne, d)| {
            let ty = match d {
                In::F32(_) => types::F32,
                In::F16(_) => types::F16,
                In::I32(_) => types::I32,
            };
            ctx.new_tensor(ty, ne).as_input()
        })
        .collect();
    let out = f(&ctx, &ts).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    for &t in &ts {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    for (t, (_, d)) in ts.iter().zip(inputs) {
        match d {
            In::F32(v) => t.set_f32(v),
            In::F16(b) => t.set_bytes(b),
            In::I32(v) => t.set_i32(v),
        }
    }
    gpu.compute(&graph).unwrap();
    let t0 = Instant::now();
    gpu.compute(&graph).unwrap();
    let once = t0.elapsed().as_secs_f64();
    let iters = ((0.5 / once) as usize).clamp(3, 50);
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.compute(&graph).unwrap();
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (us, out.bytes())
}

#[test]
#[ignore = "needs a GPU; prints kernel times"]
fn ds4_kernel_speed() {
    ON_GPU.set(true);
    let mut gpu = backend();
    ON_GPU.set(false);
    let kb = crate::model::core::KV_BLOCK;
    let mut total = 0f64;
    for c in CASES {
        let n_blocks = c.end.div_ceil(kb);
        let cells = n_blocks * kb;
        let tab = table(n_blocks);
        let tab_ne = [n_blocks as i64, 1];
        let pos: Vec<i32> = (c.end - c.ng..c.end).map(|p| p as i32).collect();
        let ng = c.ng as i64;
        let nc = (c.end / c.ratio).div_ceil(256) * 256;
        let cache = In::F16(f16_bytes(&noise(ROW * cells, 40)));
        let iq = In::F32(noise(IDX_D * IDX_H * c.ng, 41));
        let w = In::F32(noise(IDX_H * c.ng, 42));
        let q = In::F32(noise(D * NH * c.ng, 43));
        let sinks = In::F32(noise(NH, 44));
        let tab_in = In::I32(tab);
        let pos_in = In::I32(pos);
        let none = In::I32(vec![0]);
        let (iq_ne, w_ne) = ([IDX_D as i64, IDX_H as i64, ng], [IDX_H as i64, ng]);
        let (score_ne, one) = ([nc as i64, ng], [1i64]);
        let cache_ne = [ROW as i64, cells as i64];
        let pos_ne = [ng];
        let mut report = |op: &str, (us, bytes): (f64, Vec<u8>)| {
            total += us;
            eprintln!(
                "ds4 speed | {:<18} | {:<16} | {:>12.1} us | hash {:016x}",
                c.name,
                op,
                us,
                fnv(&bytes)
            );
            bytes
        };
        let scores = |gpu: &mut Backend, mask: Option<&In>| {
            let mut inputs: Vec<(&[i64], &In)> = vec![
                (&iq_ne, &iq),
                (&w_ne, &w),
                (&cache_ne, &cache),
                (&tab_ne, &tab_in),
                (&pos_ne, &pos_in),
                (&one, &none),
            ];
            if let Some(m) = mask {
                inputs.push((&score_ne, m));
            }
            time_op(gpu, &inputs, |ctx, t| {
                index_scores(
                    ctx,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t.get(6).copied(),
                    ng,
                    c.ratio,
                    2 * D as i64,
                    nc as i64,
                    t[5],
                )
            })
        };
        let sc = report("index_scores", scores(&mut gpu, None));
        let as_f32 = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_ne_bytes(*x))
                .collect()
        };
        let sc_in = In::F32(as_f32(&sc));
        let cand = report(
            "cand_mask",
            time_op(
                &mut gpu,
                &[(&score_ne, &sc_in), (&pos_ne, &pos_in)],
                |ctx, t| cand_mask(ctx, t[0], t[1], c.ratio, BS, TB),
            ),
        );
        let cand_in = In::F32(as_f32(&cand));
        report("index_scores mask", scores(&mut gpu, Some(&cand_in)));
        let k = TOP_K.min(nc);
        let sel = report(
            "topk",
            time_op(&mut gpu, &[(&score_ne, &sc_in)], |ctx, t| {
                topk(ctx, t[0], k)
            }),
        );
        let sel_in = In::I32(
            sel.as_chunks::<4>()
                .0
                .iter()
                .map(|x| i32::from_ne_bytes(*x))
                .collect(),
        );
        let sel_ne = [k as i64, ng];
        let q_ne = [D as i64, NH as i64, ng];
        let sinks_ne = [NH as i64];
        report(
            "attn",
            time_op(
                &mut gpu,
                &[
                    (&q_ne, &q),
                    (&cache_ne, &cache),
                    (&tab_ne, &tab_in),
                    (&pos_ne, &pos_in),
                    (&sinks_ne, &sinks),
                    (&sel_ne, &sel_in),
                ],
                |ctx, t| {
                    let comp = Compressed {
                        kcache: t[1],
                        sel: t[5],
                        ratio: c.ratio,
                        off: D as i64,
                    };
                    attn(
                        ctx,
                        t[0],
                        t[1],
                        t[2],
                        None,
                        t[3],
                        t[4],
                        Some(comp),
                        ng,
                        1.0 / (D as f32).sqrt(),
                        WINDOW,
                        &[],
                    )
                },
            ),
        );
    }
    eprintln!("ds4 speed | total {total:.1} us");
}
