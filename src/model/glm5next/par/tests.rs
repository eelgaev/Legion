//! The parallel decode's split tables and the exchange, against unsplit CPU references: four
//! CPU "GPUs" (threads with a graph each) exchanging through host memory.

use std::sync::Arc;

use super::exchange::{self, Exchange};
use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, types};
use crate::model::glm5next::ops;

/// Deterministic values in [-1, 1).
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

#[test]
fn rows_split_evenly_and_cover_everything() {
    for total in [1i64, 7, 8, 9, 4096, 154_880, 12_289] {
        for n_dev in 1..=4 {
            let mut next = 0;
            for d in 0..n_dev {
                let r = rows(total, n_dev, d);
                assert_eq!(r.start, next, "{total} over {n_dev}: contiguous");
                let len = r.end - r.start;
                assert!(len == total / n_dev as i64 || len == total / n_dev as i64 + 1);
                next = r.end;
            }
            assert_eq!(next, total);
        }
    }
}

#[test]
fn experts_split_by_socket() {
    let b = expert_bounds(288, 4);
    assert_eq!(b, vec![0, 72, 144, 216, 288]);
    // GPUs 0 and 1 (node 0) own the first half, 2 and 3 (node 8) the second: every expert's
    // host copy lives on its owners' socket.
    for e in 0..288u32 {
        let d = expert_owner(&b, e);
        assert!(b[d] <= e && e < b[d + 1]);
        assert_eq!(d / 2, (e / 144) as usize);
    }
    assert_eq!(expert_bounds(10, 3), vec![0, 3, 6, 10]);
    assert_eq!(expert_owner(&[0, 3, 6, 10], 9), 2);
}

#[test]
fn cross_balance_implies_the_pair_balance() {
    let plan = ParPlan::new(4, vec![0; 4], 288);
    assert!(!plan.pair_balance && !plan.cross_balance);
    let cross = plan.clone().balance_pairs(false).balance_cross(true);
    assert!(cross.pair_balance && cross.cross_balance);
    let pair = plan.balance_pairs(true).balance_cross(false);
    assert!(pair.pair_balance && !pair.cross_balance);
}

#[test]
fn expert_shares_size_each_gpus_range() {
    assert_eq!(
        expert_bounds_shared(288, 4, &[76, 76, 72, 64]),
        vec![0, 76, 152, 224, 288]
    );
    // Even shares, no shares, or one per GPU missing: the even split.
    assert_eq!(
        expert_bounds_shared(288, 4, &[1, 1, 1, 1]),
        expert_bounds(288, 4)
    );
    assert_eq!(expert_bounds_shared(288, 4, &[]), expert_bounds(288, 4));
    assert_eq!(expert_bounds_shared(288, 4, &[3, 1]), expert_bounds(288, 4));
    assert_eq!(
        expert_bounds_shared(288, 4, &[0, 0, 0, 0]),
        expert_bounds(288, 4)
    );
    // Every GPU keeps at least one expert.
    assert_eq!(
        expert_bounds_shared(8, 4, &[0, 100, 0, 0]),
        vec![0, 1, 6, 7, 8]
    );
    let plan = ParPlan::new(4, vec![0; 4], 288).share_experts(&[76, 76, 72, 64]);
    assert_eq!(plan.owned(3), 224..288);
    assert_eq!(expert_owner(&plan.bounds, 223), 2);
}

#[test]
fn copy_names_carry_their_device() {
    assert_eq!(
        shard_name("blk.3.attn_q.weight", 2),
        "blk.3.attn_q.weight@s2"
    );
    assert_eq!(copy_device("blk.3.attn_q.weight@s2"), Some(2));
    assert_eq!(copy_device(&mirror_name("output_norm.weight", 1)), Some(1));
    assert_eq!(copy_device("blk.3.attn_q.weight"), None);
    assert_eq!(copy_device("a@x1"), None);
}

#[test]
fn an_outer_shard_is_its_range_of_the_file() {
    let t = TensorInfo {
        name: "w".into(),
        ggml_type: types::Q8_0,
        shape: vec![4096, 10],
        file_index: 1,
        offset: 1000,
        size: 10 * 4352,
    };
    let path = std::path::Path::new("unused");
    let (s, r) = shard_of(&t, Cut::Outer { unit: 1 }, 4, 1, path).unwrap();
    // 10 rows over 4: 3, 3, 2, 2; GPU 1 has rows 3..6, read in place.
    assert!(r.is_none());
    assert_eq!(s.shape, vec![4096, 3]);
    assert_eq!(s.name, "w@s1");
    assert_eq!(
        (s.file_index, s.offset, s.size),
        (1, 1000 + 3 * 4352, 3 * 4352)
    );
    let (s, _) = shard_of(&t, Cut::Outer { unit: 1 }, 4, 3, path).unwrap();
    assert_eq!((s.shape[1], s.offset), (2, 1000 + 8 * 4352));
    // In units of 5 rows: GPUs 0 and 1 one unit each, 2 and 3 none.
    let (s, _) = shard_of(&t, Cut::Outer { unit: 5 }, 4, 1, path).unwrap();
    assert_eq!((s.shape[1], s.offset), (5, 1000 + 5 * 4352));
    assert!(shard_of(&t, Cut::Outer { unit: 4 }, 4, 0, path).is_err());
    // A batched matmul (MLA's k_b, v_b): its matrices, the last dimension.
    let kb = TensorInfo {
        name: "kb".into(),
        ggml_type: types::F16,
        shape: vec![8, 6, 64],
        file_index: 0,
        offset: 64,
        size: 8 * 6 * 64 * 2,
    };
    let (s, _) = shard_of(&kb, Cut::Outer { unit: 1 }, 4, 2, path).unwrap();
    assert_eq!(s.shape, vec![8, 6, 16]);
    assert_eq!((s.offset, s.size), (64 + 32 * 8 * 6 * 2, 16 * 8 * 6 * 2));
    assert!(shard_of(&kb, Cut::Mirror, 4, 0, path).is_err());
}

/// A file holding `parts` (F32 values) back to back, and each part's tensor info as a
/// `[k, len / k]` matrix.
fn f32_file(
    dir: &std::path::Path,
    parts: &[(&str, usize, &[f32])],
) -> (std::path::PathBuf, Vec<TensorInfo>) {
    let path = dir.join("w.bin");
    let mut bytes = vec![0u8; 32];
    let mut infos = Vec::new();
    for (name, k, v) in parts {
        infos.push(TensorInfo {
            name: name.to_string(),
            ggml_type: types::F32,
            shape: vec![*k as u64, (v.len() / k) as u64],
            file_index: 0,
            offset: bytes.len() as u64,
            size: v.len() as u64 * 4,
        });
        bytes.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    }
    std::fs::write(&path, bytes).unwrap();
    (path, infos)
}

/// A shard's values as the load reads them: its range of the file, or its recipe's bytes.
fn read_shard(path: &std::path::Path, info: &TensorInfo, recipe: Option<Recipe>) -> Vec<u8> {
    match recipe {
        None => {
            let all = std::fs::read(path).unwrap();
            all[info.offset as usize..(info.offset + info.size) as usize].to_vec()
        }
        Some(Recipe::Computed(f)) => {
            let b = f().unwrap();
            assert_eq!(b.len() as u64, info.size, "`{}`", info.name);
            b
        }
        Some(Recipe::Gather(_)) => panic!("a shard is a range or computed"),
    }
}

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

#[test]
fn a_column_shard_is_a_slice_of_every_row() {
    let dir = tempfile::tempdir().unwrap();
    // [k = 12, m = 3]: 12 columns in units of 3, one unit per GPU.
    let w: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let (path, infos) = f32_file(dir.path(), &[("o", 12, &w)]);
    for d in 0..4 {
        let (s, r) = shard_of(&infos[0], Cut::Cols { unit: 3 }, 4, d, &path).unwrap();
        assert_eq!(s.shape, vec![3, 3]);
        assert_eq!(s.name, format!("o@s{d}"));
        let got = to_f32(&read_shard(&path, &s, r));
        let want: Vec<f32> = (0..3)
            .flat_map(|row| (0..3).map(move |c| (row * 12 + d * 3 + c) as f32))
            .collect();
        assert_eq!(got, want, "GPU {d}");
    }
    // Whole quant blocks only: Q8_0's 32 values, columns cut at 16 split one.
    let q = TensorInfo {
        name: "q".into(),
        ggml_type: types::Q8_0,
        shape: vec![64, 2],
        file_index: 0,
        offset: 0,
        size: 2 * 68,
    };
    assert!(shard_of(&q, Cut::Cols { unit: 16 }, 4, 1, &path).is_err());
    let (s, _) = shard_of(&q, Cut::Cols { unit: 32 }, 2, 1, &path).unwrap();
    assert_eq!((s.shape.clone(), s.size), (vec![32, 2], 2 * 34));
    // Not a matrix.
    let v = TensorInfo {
        shape: vec![64],
        size: 256,
        ..infos[0].clone()
    };
    assert!(shard_of(&v, Cut::Cols { unit: 1 }, 4, 0, &path).is_err());
}

/// The tiny GLM's layer kinds: KDA, MLA, dense FFN and MoE.
fn tiny_params() -> crate::model::glm5next::Glm5Params {
    crate::model::glm5next::Glm5Params {
        mla: vec![false, true, false, true],
        n_dense: 1,
        n_ff: 8,
        q_lora: 4,
        kv_lora: 4,
        head_mla: 4,
        kda_head: 4,
        kda_lower: -5.0,
        idx_heads: 2,
        idx_dim: 4,
        top_k: 4,
        kpool: 2,
        hc: 4,
        hc_iters: 3,
        hc_eps: 1e-6,
        weights_norm: true,
        weights_scale: 1.0,
        clamp_exp: vec![0.0; 4],
        clamp_shexp: vec![0.0; 4],
        norm_eps: 1e-6,
        nextn: 0,
    }
}

#[test]
fn layer_cuts_split_heads_and_ffn_rows() {
    let p = tiny_params();
    let cut = |il: usize, w: &str| {
        layer_cuts(&p, il)
            .into_iter()
            .find(|(n, _)| *n == format!("blk.{il}.{w}"))
            .map(|(_, c)| c)
    };
    // KDA, dense: heads of `kda_head` rows, the output projection's matching columns.
    assert_eq!(cut(0, "attn_q.weight"), Some(Cut::Outer { unit: 4 }));
    assert_eq!(cut(0, "ssm_a"), Some(Cut::Outer { unit: 1 }));
    assert_eq!(cut(0, "attn_output.weight"), Some(Cut::Cols { unit: 4 }));
    assert_eq!(cut(0, "ssm_f_a.weight"), Some(Cut::Mirror));
    assert_eq!(
        cut(0, "ffn_gate.weight"),
        Some(Cut::Outer { unit: FFN_UNIT })
    );
    assert_eq!(
        cut(0, "ffn_down.weight"),
        Some(Cut::Cols { unit: FFN_UNIT })
    );
    assert_eq!(cut(0, "attn_q_b.weight"), None);
    assert_eq!(cut(0, "ffn_gate_inp.weight"), None);
    // MLA, MoE: the latents' down-projections mirrored, the heads' up-projections split;
    // the router mirrored, the shared expert split.
    assert_eq!(cut(1, "attn_kv_a_mqa.weight"), Some(Cut::Mirror));
    assert_eq!(cut(1, "attn_q_b.weight"), Some(Cut::Outer { unit: 4 }));
    assert_eq!(cut(1, "attn_k_b.weight"), Some(Cut::Outer { unit: 1 }));
    assert_eq!(cut(1, "attn_output.weight"), Some(Cut::Cols { unit: 4 }));
    assert_eq!(cut(1, "ffn_gate_inp.weight"), Some(Cut::Mirror));
    assert_eq!(
        cut(1, "ffn_up_shexp.weight"),
        Some(Cut::Outer { unit: FFN_UNIT })
    );
    assert_eq!(cut(1, "ffn_gate.weight"), None);
    assert_eq!(cut(1, "attn_q.weight"), None);
    // No weight twice.
    for il in 0..4 {
        let mut names: Vec<String> = layer_cuts(&p, il).into_iter().map(|(n, _)| n).collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "layer {il}");
    }
}

#[test]
fn mirrors_are_named_off_the_owner_only() {
    let plan = ParPlan::new(4, vec![0, 1, 2, 3], 8);
    assert_eq!(plan.mirror_sfx(1, 1), "");
    assert_eq!(plan.mirror_sfx(1, 2), "@m2");
    assert_eq!((plan.last(), plan.owned(3)), (3, 6..8));
}

/// Input data of one CPU "GPU"'s graph.
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

/// Runs `f` on the CPU on `n_dev` threads at once (one graph each, device `d` given its own
/// inputs), exchanging through `ex`; returns each thread's output.
fn run_par(
    ex: &Arc<Exchange>,
    inputs: impl Fn(usize) -> Vec<(Vec<i64>, Data)> + Sync,
    f: impl for<'a> Fn(&'a Context, usize, &[Tensor<'a>]) -> Tensor<'a> + Sync,
) -> Vec<Vec<f32>> {
    run_par_step(ex, 1, inputs, f)
}

/// [`run_par`] as step `step` of the exchange (steps after one another: their own values).
fn run_par_step(
    ex: &Arc<Exchange>,
    step: u64,
    inputs: impl Fn(usize) -> Vec<(Vec<i64>, Data)> + Sync,
    f: impl for<'a> Fn(&'a Context, usize, &[Tensor<'a>]) -> Tensor<'a> + Sync,
) -> Vec<Vec<f32>> {
    ex.begin_step(step);
    let out = std::thread::scope(|s| {
        let handles: Vec<_> = (0..ex.n_dev())
            .map(|d| {
                let (inputs, f) = (&inputs, &f);
                s.spawn(move || {
                    let inputs = inputs(d);
                    let mut backend = Backend::cpu().unwrap();
                    let ctx = Context::new(64, 64);
                    let ts: Vec<Tensor<'_>> = inputs
                        .iter()
                        .map(|(ne, v)| {
                            let ty = match v {
                                Data::F32(_) => types::F32,
                                Data::I32(_) => types::I32,
                            };
                            ctx.new_tensor(ty, ne).as_input()
                        })
                        .collect();
                    let out = f(&ctx, d, &ts).as_output();
                    let mut graph = ctx.new_graph(64);
                    graph.expand(out);
                    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                        .unwrap();
                    for (t, (_, v)) in ts.iter().zip(&inputs) {
                        match v {
                            Data::F32(v) => t.set_f32(v),
                            Data::I32(v) => t.set_i32(v),
                        }
                    }
                    backend.compute(&graph).unwrap();
                    out.to_f32()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(!ex.failed(), "an exchange timed out");
    out
}

/// The rows `r` of a `[k, m]` row-major matrix.
fn row_slice(w: &[f32], k: usize, r: std::ops::Range<i64>) -> Vec<f32> {
    w[r.start as usize * k..r.end as usize * k].to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A matmul split by rows over four GPUs and gathered is bitwise the unsplit one, on every
/// GPU; two segments at once (KDA's q / k / v come as three), uneven splits included.
#[test]
fn split_linear_gathered_is_the_unsplit_one() {
    let (k, n) = (64usize, 3i64);
    let totals = [37i64, 64];
    let ws: Vec<Vec<f32>> = totals
        .iter()
        .enumerate()
        .map(|(i, &m)| noise(k * m as usize, 10 + i as u64))
        .collect();
    let x = noise(k * n as usize, 3);
    let ex = Exchange::host(4, 1 << 16, &expert_bounds(8, 4));
    let full = run_par(
        &Exchange::host(1, 1 << 16, &[0, 8]),
        |_| {
            let mut v: Vec<(Vec<i64>, Data)> = totals
                .iter()
                .zip(&ws)
                .map(|(&m, w)| (vec![k as i64, m], Data::F32(w.clone())))
                .collect();
            v.push((vec![k as i64, n], Data::F32(x.clone())));
            v
        },
        |ctx, _, t| {
            let a = ops::linear(ctx, t[0], t[2], 1.0, None);
            let b = ops::linear(ctx, t[1], t[2], 1.0, None);
            ctx.concat(
                ctx.reshape_2d(a, totals[0] * n, 1),
                ctx.reshape_2d(b, totals[1] * n, 1),
                0,
            )
        },
    )
    .remove(0);
    let outs = run_par(
        &ex,
        |d| {
            let mut v: Vec<(Vec<i64>, Data)> = totals
                .iter()
                .zip(&ws)
                .map(|(&m, w)| {
                    let r = rows(m, 4, d);
                    (
                        vec![k as i64, r.end - r.start],
                        Data::F32(row_slice(w, k, r)),
                    )
                })
                .collect();
            v.push((vec![k as i64, n], Data::F32(x.clone())));
            v
        },
        |ctx, d, t| {
            let parts = [
                ops::linear(ctx, t[0], t[2], 1.0, None),
                ops::linear(ctx, t[1], t[2], 1.0, None),
            ];
            exchange::gather(ctx, &ex, d, &parts, &totals, None, None)
        },
    );
    for (d, o) in outs.iter().enumerate() {
        assert_eq!(bits(o), bits(&full), "GPU {d}");
    }
}

/// A gather to one GPU: it has everything (the others send only).
#[test]
fn gather_to_root() {
    let (m, n) = (10i64, 2i64);
    let full = noise((m * n) as usize, 5);
    // [m, n]: token t's rows contiguous.
    let part = |d: usize| -> Vec<f32> {
        let r = rows(m, 4, d);
        (0..n)
            .flat_map(|t| full[(t * m + r.start) as usize..(t * m + r.end) as usize].to_vec())
            .collect()
    };
    let ex = Exchange::host(4, 1 << 12, &expert_bounds(8, 4));
    for step in 0..3 {
        let outs = run_par(
            &ex,
            |d| {
                let len = rows(m, 4, d).count() as i64;
                vec![(vec![len, n], Data::F32(part(d)))]
            },
            |ctx, d, t| {
                let g = exchange::gather(ctx, &ex, d, &[t[0]], &[m], Some(3), None);
                // A second exchange after it: the counts stay in step.
                exchange::gather(ctx, &ex, d, &[t[0]], &[m], None, Some(g))
            },
        );
        for (d, o) in outs.iter().enumerate() {
            assert_eq!(bits(o), bits(&full), "step {step} GPU {d}");
        }
    }
}

#[test]
fn bcast_hands_the_owners_tensor_to_all() {
    let (m, n) = (9i64, 2i64);
    let full = noise((m * n) as usize, 6);
    let ex = Exchange::host(4, 1 << 12, &expert_bounds(8, 4));
    let outs = run_par(
        &ex,
        |d| {
            // Every GPU has some input to order the exchange after; only the owner's is used.
            let v = if d == 1 {
                full.clone()
            } else {
                vec![0.0; full.len()]
            };
            vec![(vec![m, n], Data::F32(v))]
        },
        |ctx, d, t| {
            let x = (d == 1).then_some(t[0]);
            let after = (d != 1).then_some(t[0]);
            exchange::bcast(ctx, &ex, d, 1, x, m, n, after)
        },
    );
    for (d, o) in outs.iter().enumerate() {
        assert_eq!(bits(o), bits(&full), "GPU {d}");
    }
}

/// The routed experts' columns come from their owners and the shared expert's from its GPU;
/// every GPU ends with the one-GPU tensor, whatever garbage the non-owners' columns held.
#[test]
fn select_takes_each_column_from_its_owner() {
    let (e, n_used, n) = (5i64, 3i64, 2i64);
    let n_expert = 12;
    let bounds = expert_bounds(n_expert, 4);
    // Choices: some GPUs own several, GPU 2 none of token 0's.
    let ids: Vec<i32> = vec![0, 11, 4, 3, 3, 9];
    let truth = noise((e * n_used * n) as usize, 7);
    let shared = noise((e * n) as usize, 8);
    let ex = Exchange::host(4, 1 << 12, &bounds);
    let sh = 2;
    let outs = run_par(
        &ex,
        |d| {
            // Non-owned columns: garbage (NaN), as a GPU's expert_mv leaves them.
            let y: Vec<f32> = (0..(n_used * n) as usize)
                .flat_map(|q| {
                    let own = expert_owner(&bounds, ids[q] as u32) == d;
                    (0..e as usize).map(move |i| (q, i, own))
                })
                .map(|(q, i, own)| {
                    if own {
                        truth[q * e as usize + i]
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            let mut v = vec![
                (vec![e, n_used, n], Data::F32(y)),
                (vec![n_used, n], Data::I32(ids.clone())),
            ];
            if d == sh {
                v.push((vec![e, n], Data::F32(shared.clone())));
            }
            v
        },
        |ctx, d, t| {
            let s = (d == sh).then(|| t[2]);
            exchange::select(ctx, &ex, d, t[0], t[1], sh, s, None)
        },
    );
    let mut want = truth.clone();
    want.extend(&shared);
    for (d, o) in outs.iter().enumerate() {
        assert_eq!(bits(o), bits(&want), "GPU {d}");
    }
    // Each GPU waited for the others at least a little (the waits are counted).
    assert_eq!(ex.take_waits().len(), 4);
    // Taken: the matrix starts again from zero.
    let m = ex.take_wait_matrix();
    assert_eq!(m.len(), 4);
    assert!(m.iter().all(|r| r.len() == 4 && r.iter().all(|&w| w == 0)));
}

#[test]
fn ep_layout_gives_every_gpu_its_experts_of_every_layer() {
    // Six decoder layers over four GPUs (owners 0, 0, 1, 2, 3, 3), the MTP head as entry 6.
    let plan = ParPlan::new(4, vec![0, 0, 1, 2, 3, 3], 288);
    let (ep, cuda) = ep_layers(&plan, 7);
    assert_eq!(ep.real.len(), 7 + 6 * 3);
    assert_eq!((ep.real.len(), ep.owned.len()), (cuda.len(), cuda.len()));
    // The model's entries: the owner's experts; the MTP head all, on the last GPU.
    assert_eq!(ep.owned[3], Some(144..216));
    assert_eq!((ep.owned[6].clone(), cuda[6]), (None, 3));
    for il in 0..6 {
        // Every GPU serves layer il once, its own experts.
        let mut on: Vec<(usize, std::ops::Range<u32>)> = (0..ep.real.len())
            .filter(|&e| ep.real[e] == il)
            .map(|e| (cuda[e], ep.owned[e].clone().unwrap()))
            .collect();
        on.sort_by_key(|x| x.0);
        let want: Vec<_> = (0..4).map(|d| (d, plan.owned(d))).collect();
        assert_eq!(on, want, "layer {il}");
        for c in (0..4).filter(|&c| c != plan.owner[il]) {
            let e = copy_entry(7, 4, il, plan.owner[il], c);
            assert_eq!((ep.real[e], cuda[e]), (il, c));
        }
    }
}

#[test]
fn ep_layout_splits_the_mtp_head_over_every_gpu() {
    // `--glm-parallel-mtp`: the head's entry the last GPU's experts, a copy per other GPU after
    // the decoder layers' copies.
    let plain = ParPlan::new(4, vec![0, 0, 1, 2, 3, 3], 288);
    assert!(mtp_copies(&plain, 7).is_empty());
    let plan = plain.split_mtp(true);
    let (ep, cuda) = ep_layers(&plan, 7);
    let copies = mtp_copies(&plan, 7);
    assert_eq!(copies, 7 + 6 * 3..7 + 6 * 3 + 3);
    assert_eq!(ep.real.len(), copies.end);
    assert_eq!((ep.owned.len(), cuda.len()), (copies.end, copies.end));
    assert_eq!((ep.owned[6].clone(), cuda[6]), (Some(plan.owned(3)), 3));
    let mut on: Vec<(usize, std::ops::Range<u32>)> = (0..ep.real.len())
        .filter(|&e| ep.real[e] == 6)
        .map(|e| (cuda[e], ep.owned[e].clone().unwrap()))
        .collect();
    on.sort_by_key(|x| x.0);
    let want: Vec<_> = (0..4).map(|d| (d, plan.owned(d))).collect();
    assert_eq!(on, want);
    for (c, e) in copies.enumerate() {
        assert_eq!((ep.real[e], cuda[e]), (6, c));
    }
    // The decoder layers' copies are where they were.
    let (old, _) = ep_layers(&ParPlan::new(4, vec![0, 0, 1, 2, 3, 3], 288), 7);
    assert_eq!(ep.real[7..old.real.len()], old.real[7..]);
    // Without the head in the cache (no MTP), no copies.
    let (ep, _) = ep_layers(&plan, 6);
    assert_eq!(ep.real.len(), 6 + 6 * 3);
    assert!(mtp_copies(&plan, 6).is_empty());
}

#[test]
fn mtp_cuts_mirror_the_router_and_split_the_shared_expert() {
    let cuts = mtp::mtp_cuts(6);
    let cut = |w: &str| {
        cuts.iter()
            .find(|(n, _)| n == &format!("blk.6.{w}"))
            .map(|x| x.1)
    };
    for w in [
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "nextn.shared_head_norm.weight",
    ] {
        assert_eq!(cut(w), Some(Cut::Mirror), "{w}");
    }
    let outer = Some(Cut::Outer { unit: FFN_UNIT });
    assert_eq!(cut("ffn_gate_shexp.weight"), outer);
    assert_eq!(cut("ffn_up_shexp.weight"), outer);
    assert_eq!(
        cut("ffn_down_shexp.weight"),
        Some(Cut::Cols { unit: FFN_UNIT })
    );
    // The front stays whole on the last GPU.
    assert_eq!(cuts.len(), 7);
    assert!(cut("nextn.eh_proj.weight").is_none());
}

#[test]
fn the_slot_fits_every_exchange() {
    // GLM-5.3-Flash: 4096 wide, 4 streams, 154880 tokens, 4 GPUs, 20 logits rows.
    let b = slot_bytes(4096, 4, 154_880, 4, MAX_TOKENS, 20);
    let n = MAX_TOKENS as i64;
    assert!(b >= exchange::reduce_bytes(4096, n));
    assert!(b >= exchange::gather_bytes(&[154_880], 20, 4, 0));
    assert!(b >= 4096 * 4 * MAX_TOKENS * 4);
    // The streams' broadcast is the largest: 32 MiB.
    assert!(b.is_multiple_of(4096) && b <= 33 << 20);
}

/// Prompt steps (`--glm-prompt-step`, #686): 512 to 8192 tokens, 2048 by default; the host slots
/// grow with them (128 MiB at 2048), the VRAM slots stay a 512-token reduce's.
#[test]
fn prompt_steps_size_the_host_slots() {
    assert_eq!(step_tokens(PROMPT_TOKENS), 2048);
    assert_eq!(step_tokens(100), MAX_TOKENS);
    assert_eq!(step_tokens(1 << 20), MAX_PROMPT_TOKENS);
    let b = slot_bytes(4096, 4, 154_880, 4, step_tokens(PROMPT_TOKENS), 20);
    assert_eq!(b, 128 << 20);
    assert!(b >= exchange::reduce_bytes(4096, 2048));
    assert_eq!(peer_slot_bytes(4096, MAX_TOKENS), 8 << 20);
}

/// On four GPUs every reduce with a chunk for each GPU to own goes as a reduce-scatter and an
/// all-gather (#686): prompt steps' and 7 tokens' or more; a verify step's 3 tokens go as
/// before, and so does everything on two GPUs (no bytes saved).
#[test]
fn reduces_scatter_once_every_gpu_owns_a_chunk() {
    let ex = Exchange::host(4, 16, &expert_bounds(8, 4));
    assert!(ex.scatters(exchange::reduce_bytes(4096, 2048)));
    assert!(ex.scatters(exchange::reduce_bytes(4096, MAX_TOKENS as i64)));
    assert!(ex.scatters(exchange::reduce_bytes(4096, 7)));
    assert!(!ex.scatters(exchange::reduce_bytes(4096, 6)));
    assert!(!ex.scatters(exchange::reduce_bytes(4096, 3)));
    assert!(!Exchange::host(2, 16, &[0, 4, 8]).scatters(64 << 20));
    assert!(!Exchange::host(1, 16, &[0, 8]).scatters(64 << 20));
    let mut ex = Exchange::host_with(4, 16, &expert_bounds(8, 4), [0; 8], 0, 1024);
    assert!(!ex.scatters(3 * 1024) && ex.scatters(4 * 1024));
    ex.set_scatter_min(usize::MAX);
    assert!(!ex.scatters(64 << 20));
}

/// Every GPU ends with the sum of the GPUs' parts added in GPU order: the same bits on every
/// GPU, step after step.
#[test]
fn reduce_adds_the_parts_in_gpu_order_on_every_gpu() {
    let (e, n) = (37i64, 3i64);
    let len = (e * n) as usize;
    let parts: Vec<Vec<f32>> = (0..4).map(|d| noise(len, 20 + d)).collect();
    let want: Vec<f32> = (0..len)
        .map(|i| ((parts[0][i] + parts[1][i]) + parts[2][i]) + parts[3][i])
        .collect();
    // The sum's partial sums the other way round: not the same bits, so the order is checked.
    let other: Vec<f32> = (0..len)
        .map(|i| ((parts[3][i] + parts[2][i]) + parts[1][i]) + parts[0][i])
        .collect();
    assert_ne!(bits(&want), bits(&other));
    let ex = Exchange::host(4, exchange::reduce_bytes(e, n), &expert_bounds(8, 4));
    for step in 0..3 {
        let outs = run_par(
            &ex,
            |d| vec![(vec![e, n], Data::F32(parts[d].clone()))],
            |ctx, d, t| {
                let s = exchange::reduce(ctx, &ex, d, t[0], None);
                // A second one after it (as the layers chain them): the sum of four sums.
                let s2 = exchange::reduce(ctx, &ex, d, t[0], Some(s));
                ctx.add(s, ctx.scale(s2, 0.0))
            },
        );
        for (d, o) in outs.iter().enumerate() {
            assert_eq!(bits(o), bits(&want), "step {step} GPU {d}");
        }
    }
}

#[test]
fn peer_writes_pick_the_gpus_of_a_node() {
    let nodes = [0, 0, 8, 8];
    let all = |_, _| true;
    let m = exchange::peer_masks(exchange::PeerWrites::Pairs, &nodes, all);
    assert_eq!(m[..4], [0b0010, 0b0001, 0b1000, 0b0100]);
    let m = exchange::peer_masks(exchange::PeerWrites::All, &nodes, all);
    assert_eq!(m[..4], [0b1110, 0b1101, 0b1011, 0b0111]);
    let m = exchange::peer_masks(exchange::PeerWrites::Off, &nodes, all);
    assert!(m.iter().all(|&x| x == 0));
    // GPU 0 cannot reach GPU 1's memory: GPU 1 gets nothing from it; 2 and 3 are untouched.
    let m = exchange::peer_masks(exchange::PeerWrites::Pairs, &nodes, |w, r| (w, r) != (0, 1));
    assert_eq!(m[..4], [0b0010, 0, 0b1000, 0b0100]);
    // GPUs of no known node are never a pair; `all` still takes them.
    let none = crate::residency::fallback::NO_NODE;
    let m = exchange::peer_masks(exchange::PeerWrites::Pairs, &[none, none, 8, 8], all);
    assert_eq!(m[..4], [0, 0, 0b1000, 0b0100]);
    let m = exchange::peer_masks(exchange::PeerWrites::All, &[none, none], all);
    assert_eq!(m[..2], [0b10, 0b01]);
    // The pairs come from the GPUs' reported nodes: an unreported one is no node, not the
    // node 0 its inbox may fall back to.
    let gpus = exchange::gpu_nodes(&[None, None, Some(8), Some(8)]);
    assert_eq!(gpus, [none, none, 8, 8]);
    let m = exchange::peer_masks(exchange::PeerWrites::Pairs, &gpus, all);
    assert_eq!(m[..4], [0, 0, 0b1000, 0b0100]);
    let gpus = exchange::gpu_nodes(&[Some(0), Some(0), Some(8), Some(8)]);
    let m = exchange::peer_masks(exchange::PeerWrites::Pairs, &gpus, all);
    assert_eq!(m[..4], [0b0010, 0b0001, 0b1000, 0b0100]);
    // The default is the pairs.
    assert_eq!(exchange::PeerWrites::default(), exchange::PeerWrites::Pairs);
    // A whole step's reduce goes GPU to GPU: 8 MiB a slot at 512 tokens.
    assert_eq!(peer_slot_bytes(4096, MAX_TOKENS), 8 << 20);
}

/// The exchanges in many small chunks, through VRAM inboxes (host memory in the twin) for some
/// writers and host inboxes for the others (too big for a VRAM slot, or no peer), give the same
/// bits as one chunk through host memory: gathers split mid-row, a gather to one GPU, a
/// broadcast, reduces in GPU order (scattered to the chunks' owners and gathered back, too),
/// step after step.
#[test]
fn chunked_peer_exchanges_keep_the_bits() {
    let nodes = [0, 0, 8, 8];
    let bounds = expert_bounds(8, 4);
    let slot = 1 << 16;
    let pairs = exchange::peer_masks(exchange::PeerWrites::Pairs, &nodes, |_, _| true);
    let all = exchange::peer_masks(exchange::PeerWrites::All, &nodes, |_, _| true);
    let none = usize::MAX;
    let setups = [
        // As before #671: one chunk, host memory.
        ([0; exchange::MAX_DEV], 0, exchange::CHUNK_MIN, none),
        // Pairs, every payload fits: the smallest chunks.
        (pairs, slot, 16, none),
        // Everyone, 300-byte slots: the reduce (5920 bytes) and some gathers take host memory.
        (all, 304, 64, none),
        // The reduces scattered (#686): through the pairs' VRAM inboxes, through host memory.
        (pairs, slot, 64, 1),
        (all, 304, 1024, 4096),
    ];
    let n = 3i64;
    let totals = [37i64, 64];
    let gfull: Vec<Vec<f32>> = totals
        .iter()
        .enumerate()
        .map(|(i, &m)| noise(m as usize * n as usize, 30 + i as u64))
        .collect();
    // GPU d's rows of segment s: [len, n], token t's rows contiguous.
    let part = |s: usize, d: usize| -> Vec<f32> {
        let (m, r) = (totals[s], rows(totals[s], 4, d));
        (0..n)
            .flat_map(|t| gfull[s][(t * m + r.start) as usize..(t * m + r.end) as usize].to_vec())
            .collect()
    };
    let gwant: Vec<f32> = gfull.concat();
    let (bm, bn) = (9i64, 20i64);
    let bfull = noise((bm * bn) as usize, 40);
    let (re, rn) = (37i64, 40i64);
    let rparts: Vec<Vec<f32>> = (0..4).map(|d| noise((re * rn) as usize, 50 + d)).collect();
    let rwant: Vec<f32> = (0..rparts[0].len())
        .map(|i| ((rparts[0][i] + rparts[1][i]) + rparts[2][i]) + rparts[3][i])
        .collect();
    for (si, &(mask, peer_slot, chunk_min, scatter_min)) in setups.iter().enumerate() {
        let mut ex = Exchange::host_with(4, slot, &bounds, mask, peer_slot, chunk_min);
        ex.set_scatter_min(scatter_min);
        let rbytes = exchange::reduce_bytes(re, rn);
        assert_eq!(ex.scatters(rbytes), scatter_min != none, "setup {si}");
        let ex = Arc::new(ex);
        for step in 1..=3u64 {
            let inputs = |d: usize| {
                let b = if d == 1 {
                    bfull.clone()
                } else {
                    vec![0.0; bfull.len()]
                };
                vec![
                    (
                        vec![rows(totals[0], 4, d).count() as i64, n],
                        Data::F32(part(0, d)),
                    ),
                    (
                        vec![rows(totals[1], 4, d).count() as i64, n],
                        Data::F32(part(1, d)),
                    ),
                    (vec![bm, bn], Data::F32(b)),
                    (vec![re, rn], Data::F32(rparts[d].clone())),
                ]
            };
            // Gather (two segments) to GPU 2, then to all, a broadcast, two reduces: the
            // results one after another (the root gather on GPU 2 only).
            let outs = run_par_step(&ex, step, inputs, |ctx, d, t| {
                let g1 = exchange::gather(ctx, &ex, d, &t[..2], &totals, Some(2), None);
                let g = exchange::gather(ctx, &ex, d, &t[..2], &totals, None, Some(g1));
                fn flat1<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
                    ctx.reshape_1d(x, x.elements() as i64)
                }
                let flat = |x| flat1(ctx, x);
                let x = (d == 1).then_some(t[2]);
                // After the gather; the others' unused input in the graph too.
                let after = if d == 1 {
                    g
                } else {
                    ctx.concat(flat(g), flat(t[2]), 0)
                };
                let b = exchange::bcast(ctx, &ex, d, 1, x, bm, bn, Some(after));
                let r = exchange::reduce(ctx, &ex, d, t[3], Some(b));
                let r2 = exchange::reduce(ctx, &ex, d, t[3], Some(r));
                let mut out = ctx.concat(flat(g), flat(b), 0);
                out = ctx.concat(out, flat(r), 0);
                out = ctx.concat(out, flat(r2), 0);
                if d == 2 {
                    out = ctx.concat(out, flat(g1), 0);
                }
                out
            });
            let mut want = gwant.clone();
            want.extend(&bfull);
            want.extend(&rwant);
            want.extend(&rwant);
            for (d, o) in outs.iter().enumerate() {
                let mut w = want.clone();
                if d == 2 {
                    w.extend(&gwant);
                }
                assert_eq!(bits(o), bits(&w), "setup {si} step {step} GPU {d}");
            }
            assert!(!ex.failed(), "setup {si} step {step}");
        }
    }
}

/// The GPUs' F32 shards of the weights in `w` ([`Cut`] each), read from a file as the load
/// reads them: per GPU, its shards in order.
fn shards(
    dir: &std::path::Path,
    w: &[(&str, usize, &[f32], Cut)],
    n_dev: usize,
) -> Vec<Vec<(Vec<i64>, Vec<f32>)>> {
    let parts: Vec<(&str, usize, &[f32])> = w.iter().map(|&(n, k, v, _)| (n, k, v)).collect();
    let (path, infos) = f32_file(dir, &parts);
    (0..n_dev)
        .map(|d| {
            infos
                .iter()
                .zip(w)
                .map(|(t, &(_, _, _, cut))| {
                    let (s, r) = shard_of(t, cut, n_dev, d, &path).unwrap();
                    let ne = s.shape.iter().map(|&x| x as i64).collect();
                    (ne, to_f32(&read_shard(&path, &s, r)))
                })
                .collect()
        })
        .collect()
}

/// `|a - b|` within `tol` of the reference `b`'s largest magnitude (sums in another order
/// round their partial sums, which are of that size, not the result's), value by value.
fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let m = b.iter().fold(1.0f32, |m, y| m.max(y.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * m,
            "{what} [{i}]: {x} vs {y} (max {m})"
        );
    }
}

/// Megatron's FFN split (the dense FFN and the shared expert): every GPU its rows of gate and
/// up and its columns of down, read from the file as the load reads them ([`shard_of`]), and
/// the all-reduce of the partial sums is the unsplit FFN (up to the order of the sum), the
/// same bits on every GPU.
#[test]
fn ffn_split_by_rows_and_reduced_is_the_unsplit_one() {
    let (k, m, n) = (64usize, 4 * FFN_UNIT as usize, 3i64);
    let (wg, wu, wd) = (noise(k * m, 30), noise(k * m, 31), noise(m * k, 32));
    let x = noise(k * n as usize, 33);
    fn ffn<'a>(ctx: &'a Context, t: &[Tensor<'a>]) -> Tensor<'a> {
        let g = ops::linear(ctx, t[0], t[3], 1.0, None);
        let u = ops::linear(ctx, t[1], t[3], 1.0, None);
        let h = ctx.mul(ctx.silu(g), u);
        // A copy, as the graph's reduce takes it.
        ctx.cont(ops::linear(ctx, t[2], h, 1.0, None))
    }
    let full = run_par(
        &Exchange::host(1, 1 << 16, &[0, 8]),
        |_| {
            vec![
                (vec![k as i64, m as i64], Data::F32(wg.clone())),
                (vec![k as i64, m as i64], Data::F32(wu.clone())),
                (vec![m as i64, k as i64], Data::F32(wd.clone())),
                (vec![k as i64, n], Data::F32(x.clone())),
            ]
        },
        |ctx, _, t| ffn(ctx, t),
    )
    .remove(0);
    let dir = tempfile::tempdir().unwrap();
    let rows_cut = Cut::Outer { unit: FFN_UNIT };
    let cols_cut = Cut::Cols { unit: FFN_UNIT };
    let sh = shards(
        dir.path(),
        &[
            ("gate", k, &wg, rows_cut),
            ("up", k, &wu, rows_cut),
            ("down", m, &wd, cols_cut),
        ],
        4,
    );
    assert_eq!(sh[1][2].0, vec![FFN_UNIT, k as i64]);
    let ex = Exchange::host(4, exchange::reduce_bytes(k as i64, n), &expert_bounds(8, 4));
    let outs = run_par(
        &ex,
        |d| {
            let mut v: Vec<(Vec<i64>, Data)> = sh[d]
                .iter()
                .map(|(ne, w)| (ne.clone(), Data::F32(w.clone())))
                .collect();
            v.push((vec![k as i64, n], Data::F32(x.clone())));
            v
        },
        |ctx, d, t| exchange::reduce(ctx, &ex, d, ffn(ctx, t), None),
    );
    for (d, o) in outs.iter().enumerate() {
        assert_eq!(bits(o), bits(&outs[0]), "GPU {d}");
    }
    assert_close(&outs[0], &full, 1e-5, "split FFN");
}

/// Attention split by heads: every GPU its heads' rows of the input projection, a per-head
/// nonlinear step (a head norm, as KDA's), its heads' columns of the output projection; the
/// all-reduce is the unsplit layer.
#[test]
fn heads_split_and_reduced_are_the_unsplit_layer() {
    let (k, dh, nh, n) = (32usize, 8i64, 8i64, 3i64);
    let inner = (dh * nh) as usize;
    let (wq, wo) = (noise(k * inner, 40), noise(inner * k, 41));
    let x = noise(k * n as usize, 42);
    fn layer<'a>(ctx: &'a Context, t: &[Tensor<'a>]) -> Tensor<'a> {
        let (dh, n) = (8, t[2].ne()[1]);
        let q = ctx.cont(ops::linear(ctx, t[0], t[2], 1.0, None));
        let heads = q.ne()[0] / dh;
        let q = ops::rms_norm(ctx, ctx.reshape_3d(q, dh, heads, n), None, 1e-6);
        let q = ctx.reshape_2d(ctx.cont(q), dh * heads, n);
        ctx.cont(ops::linear(ctx, t[1], q, 1.0, None))
    }
    let full = run_par(
        &Exchange::host(1, 1 << 16, &[0, 8]),
        |_| {
            vec![
                (vec![k as i64, inner as i64], Data::F32(wq.clone())),
                (vec![inner as i64, k as i64], Data::F32(wo.clone())),
                (vec![k as i64, n], Data::F32(x.clone())),
            ]
        },
        |ctx, _, t| layer(ctx, t),
    )
    .remove(0);
    let dir = tempfile::tempdir().unwrap();
    let sh = shards(
        dir.path(),
        &[
            ("q", k, &wq, Cut::Outer { unit: dh }),
            ("o", inner, &wo, Cut::Cols { unit: dh }),
        ],
        4,
    );
    // Two heads per GPU.
    assert_eq!(sh[3][0].0, vec![k as i64, 2 * dh]);
    let ex = Exchange::host(4, exchange::reduce_bytes(k as i64, n), &expert_bounds(8, 4));
    let outs = run_par(
        &ex,
        |d| {
            let mut v: Vec<(Vec<i64>, Data)> = sh[d]
                .iter()
                .map(|(ne, w)| (ne.clone(), Data::F32(w.clone())))
                .collect();
            v.push((vec![k as i64, n], Data::F32(x.clone())));
            v
        },
        |ctx, d, t| exchange::reduce(ctx, &ex, d, layer(ctx, t), None),
    );
    for (d, o) in outs.iter().enumerate() {
        assert_eq!(bits(o), bits(&outs[0]), "GPU {d}");
    }
    assert_close(&outs[0], &full, 1e-5, "split heads");
}

/// A GPU's routed sum: its own choices' weighted columns only (the others' columns hold
/// garbage the expert matvec left), plus the shared expert's part; over the GPUs the parts
/// add up to the whole MoE output.
#[test]
fn ep_sum_takes_the_owned_choices_only() {
    let (e, n_used, n) = (5i64, 3i64, 2i64);
    let n_expert = 12;
    let bounds = expert_bounds(n_expert, 4);
    let ids: Vec<i32> = vec![0, 11, 4, 3, 3, 9];
    let y = noise((e * n_used * n) as usize, 50);
    let w = noise((n_used * n) as usize, 51);
    let shared = noise((e * n) as usize, 52);
    // One GPU's sum, the reference way.
    let want = |d: usize, add: bool| -> Vec<f32> {
        let owned = bounds[d] as i32..bounds[d + 1] as i32;
        (0..n as usize)
            .flat_map(|t| (0..e as usize).map(move |i| (t, i)))
            .map(|(t, i)| {
                let mut acc = 0.0f32;
                for j in 0..n_used as usize {
                    let q = t * n_used as usize + j;
                    if owned.contains(&ids[q]) {
                        acc += w[q] * y[q * e as usize + i];
                    }
                }
                if add {
                    acc += shared[t * e as usize + i];
                }
                acc
            })
            .collect()
    };
    let mut total = vec![0.0f32; (e * n) as usize];
    for d in 0..4 {
        // Non-owned columns: NaN, which must not reach the sum.
        let yd: Vec<f32> = y
            .chunks(e as usize)
            .zip(&ids)
            .flat_map(|(col, &s)| {
                let own = expert_owner(&bounds, s as u32) == d;
                col.iter().map(move |&v| if own { v } else { f32::NAN })
            })
            .collect();
        let add = d == 1;
        let out = run_par(
            &Exchange::host(1, 1 << 12, &[0, 8]),
            |_| {
                let mut v = vec![
                    (vec![e, n_used, n], Data::F32(yd.clone())),
                    (vec![n_used, n], Data::F32(w.clone())),
                    (vec![n_used, n], Data::I32(ids.clone())),
                ];
                if add {
                    v.push((vec![e, n], Data::F32(shared.clone())));
                }
                v
            },
            |ctx, _, t| {
                let owned = bounds[d]..bounds[d + 1];
                exchange::ep_sum(ctx, t[0], t[1], t[2], owned, t.get(3).copied())
            },
        )
        .remove(0);
        assert_eq!(bits(&out), bits(&want(d, add)), "GPU {d}");
        for (a, b) in total.iter_mut().zip(&out) {
            *a += b;
        }
    }
    // Every choice counted once: the whole weighted sum plus the shared expert.
    let whole: Vec<f32> = (0..n as usize)
        .flat_map(|t| (0..e as usize).map(move |i| (t, i)))
        .map(|(t, i)| {
            let mut acc = shared[t * e as usize + i];
            for j in 0..n_used as usize {
                let q = t * n_used as usize + j;
                acc += w[q] * y[q * e as usize + i];
            }
            acc
        })
        .collect();
    assert_close(&total, &whole, 1e-5, "the GPUs' parts");
}
