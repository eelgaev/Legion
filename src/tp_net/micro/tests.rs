use super::*;

/// Every column of every rank's share is in exactly one micro-batch's piece, in order.
#[test]
fn natural_pieces_split_every_share() {
    for (n, ranks) in [(4096, 2), (4097, 2), (17, 3), (5, 2), (1, 1), (1024, 4)] {
        for r in 0..ranks {
            let (a, m) = share(n, r, ranks);
            let (b0, c0) = piece(n, r, ranks, 0, false);
            let (b1, c1) = piece(n, r, ranks, 1, false);
            assert_eq!(
                (b0, b0 + c0, b1 + c1),
                (a, b1, a + m),
                "n {n} rank {r}/{ranks}"
            );
        }
    }
}

/// The compact layout puts every rank's piece of a micro-batch after the other, in rank order,
/// and spans the micro-batch.
#[test]
fn compact_pieces_follow_rank_order() {
    for (n, ranks) in [(4096, 2), (4099, 2), (17, 3), (1024, 4)] {
        for j in 0..MICRO {
            let mut at = 0;
            for r in 0..ranks {
                let (b, c) = piece(n, r, ranks, j, true);
                assert_eq!(b, at, "n {n} rank {r}/{ranks} micro {j}");
                assert_eq!(c, piece(n, r, ranks, j, false).1);
                at += c;
            }
            assert_eq!(span_cols(n, ranks, j, true), at);
            assert_eq!(span_cols(n, ranks, j, false), n);
        }
        let both: usize = (0..MICRO).map(|j| span_cols(n, ranks, j, true)).sum();
        assert_eq!(both, n);
    }
}

#[test]
fn params_round_trip() {
    let p = Params {
        comm: 3,
        gather: true,
        n: 4096,
        j: 1,
        compact: true,
        reduce: false,
    };
    assert_eq!(p.list(), [3, 1, 4096, 1, 1, 0]);
    let r = Params { reduce: true, ..p };
    assert_eq!(r.list()[5], 1);
}

/// A micro-batch all-reduce (#619) of each half of a tensor gives the halves of the whole
/// tensor's all-reduce bit for bit: the wire type goes by the whole tensor's values, even
/// where a half alone would travel in F32.
#[test]
fn halves_all_reduce_as_whole() {
    use super::super::Wire;
    let (row, cols) = (5, 12);
    let len = row * cols;
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(10));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                c.set_collectives(Wire::Bf16, 0);
                // BF16 for the whole tensor's exchange, F32 for a half's alone.
                c.set_bf16_min(len);
                assert_eq!((c.wire(len), c.wire(len / 2)), (Wire::Bf16, Wire::F32));
                let src: Vec<f32> = (0..len)
                    .map(|i| 1.0 + (i * 7 + r * 31) as f32 / 1024.0 + r as f32 * 1e-3)
                    .collect();
                let mut whole = vec![0f32; len];
                c.all_reduce_host(&src, &mut whole).unwrap();
                let mut halves = vec![0f32; len];
                let cut = row * (cols / 2);
                let (a, b) = halves.split_at_mut(cut);
                c.all_reduce_host_as(&src[..cut], a, len).unwrap();
                c.all_reduce_host_as(&src[cut..], b, len).unwrap();
                (whole, halves)
            })
        })
        .collect();
    for t in threads {
        let (whole, halves) = t.join().unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&whole), bits(&halves));
    }
}

/// An odd step's halves (#619): half 1 has one more column than half 0; the sums are still the
/// whole all-reduce's bits.
#[test]
fn odd_halves_all_reduce_as_whole() {
    use super::super::Wire;
    let (row, cols) = (5, 13);
    let len = row * cols;
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(10));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                c.set_collectives(Wire::Bf16, 0);
                c.set_bf16_min(len);
                let src: Vec<f32> = (0..len)
                    .map(|i| 1.0 + (i * 7 + r * 31) as f32 / 1024.0 + r as f32 * 1e-3)
                    .collect();
                let mut whole = vec![0f32; len];
                c.all_reduce_host(&src, &mut whole).unwrap();
                let mut halves = vec![0f32; len];
                let cut = row * (cols / 2);
                assert_eq!((cut, len - cut), (row * 6, row * 7));
                let (a, b) = halves.split_at_mut(cut);
                c.all_reduce_host_as(&src[..cut], a, len).unwrap();
                c.all_reduce_host_as(&src[cut..], b, len).unwrap();
                (whole, halves)
            })
        })
        .collect();
    for t in threads {
        let (whole, halves) = t.join().unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&whole), bits(&halves));
    }
}

/// `--tp-overlap` splits only steps whose every micro-batch has at least `MIN_COLS` columns
/// (#1176): a 1,536-token step stays whole, 4,096 and up split.
#[test]
fn short_steps_stay_whole() {
    assert_eq!(MIN_COLS, 1024);
    for (n, ranks, split) in [
        (1536, 2, false),
        (4095, 2, false),
        (4096, 2, true),
        (4608, 2, true),
        (8192, 4, true),
        (8191, 4, false),
    ] {
        assert_eq!(worth(n, ranks), split, "n {n} ranks {ranks}");
    }
}

/// A caller's lower floor (#560 r12, GLM's 256): a long prompt's 1,650- and 2,470-token tail
/// steps split over two ranks, a 1,023-token one does not; above `MIN_COLS` it changes nothing.
#[test]
fn lower_floor_splits_tail_steps() {
    for (n, split) in [(1650, true), (2470, true), (1024, true), (1023, false)] {
        assert_eq!(worth_at(n, 2, 256), split, "n {n}");
    }
    assert!(!worth_at(4095, 2, 4096));
    assert_eq!(worth_at(4095, 2, 2048), worth(4095, 2));
}

/// `finish_add` (#1133 r2) of an odd step's two halves through a CPU graph on two ranks: each
/// half's sum with its residual added lands in its columns of one output, the bits of ggml's
/// add of the whole step's all-reduce and the residual, on both ranks.
#[test]
fn halves_finish_add_into_one_output() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (row, n) = (6usize, 9i64);
    let (n0, n1) = (n / 2, n - n / 2);
    let len = row * n as usize;
    let value = |r: usize, k: usize, i: usize| {
        let x = ((i * 7 + k * 13 + r * 31) % 97) as f32;
        if r == 0 { x * 1e7 } else { x * 0.3 + 1e-3 }
    };
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(20));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = super::super::register(c);
                let mut backend = Backend::cpu().unwrap();
                let ctx = Context::new(32, 32);
                let x = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let half = |t, a: i64, c: i64| {
                    ctx.view_2d(t, [row as i64, c], row * 4, a as usize * row * 4)
                };
                let mut graph = ctx.new_graph(32);
                let s0 = all_reduce_start(&ctx, handle.id(), half(x, 0, n0), n as usize);
                let s1 = all_reduce_start(&ctx, handle.id(), half(x, n0, n1), n as usize);
                graph.expand(s0.node);
                graph.expand(s1.node);
                let y0 = finish_add(&ctx, s0, half(res, 0, n0), n, 0, None).as_output();
                graph.expand(y0);
                let y = finish_add(&ctx, s1, half(res, n0, n1), n, n0, Some(y0)).as_output();
                graph.expand(y);
                let plain = ctx
                    .add(super::super::all_reduce(&ctx, handle.id(), x), res)
                    .as_output();
                graph.expand(plain);
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&(0..len).map(|i| value(r, k, i)).collect::<Vec<_>>());
                    res.set_f32(&(0..len).map(|i| value(1, k + 5, i)).collect::<Vec<_>>());
                    backend.compute(&graph).unwrap();
                    out.push((y.to_f32(), plain.to_f32()));
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for k in 0..3 {
        let want: Vec<f32> = (0..len)
            .map(|i| (value(0, k, i) + value(1, k, i)) + value(1, k + 5, i))
            .collect();
        for r in &results {
            assert_eq!(bits(&r[k].0), bits(&want), "halves, round {k}");
            assert_eq!(bits(&r[k].1), bits(&want), "plain, round {k}");
        }
    }
}

/// Three parts of a step (#1133: Qwen3.5+'s FFN parts) finish with their residuals into one
/// output, each later finish into the one before (a view of the first's memory): the plain
/// all-reduce's bits through a CPU graph on two ranks, at most two exchanges in flight, only the
/// last finish an output (ggml's allocator keeps the first's memory for the later ones).
#[test]
fn three_parts_finish_add_into_one_output() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (row, n) = (6usize, 11i64);
    let parts = [(0i64, 5i64), (5, 4), (9, 2)];
    let len = row * n as usize;
    let value = |r: usize, k: usize, i: usize| {
        let x = ((i * 11 + k * 17 + r * 29) % 89) as f32;
        if r == 0 { x * 1e7 } else { x * 0.3 + 1e-3 }
    };
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(20));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = super::super::register(c);
                let mut backend = Backend::cpu().unwrap();
                let ctx = Context::new(48, 48);
                let x = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let part = |t, a: i64, c: i64| {
                    ctx.view_2d(t, [row as i64, c], row * 4, a as usize * row * 4)
                };
                // Each part's partial its own node (a matmul's output in the model).
                let p: Vec<_> = parts
                    .iter()
                    .map(|&(a, c)| ctx.scale(part(x, a, c), 1.0))
                    .collect();
                let mut graph = ctx.new_graph(48);
                let start = |j: usize| all_reduce_start(&ctx, handle.id(), p[j], n as usize);
                let (s0, s1) = (start(0), start(1));
                graph.expand(s0.node);
                graph.expand(s1.node);
                let (a, c) = parts[0];
                let y0 = finish_add(&ctx, s0, part(res, a, c), n, a, None);
                graph.expand(y0);
                let s2 = start(2);
                graph.expand(s2.node);
                let (a, c) = parts[1];
                let y1 = finish_add(&ctx, s1, part(res, a, c), n, a, Some(y0));
                graph.expand(y1);
                let (a, c) = parts[2];
                let y = finish_add(&ctx, s2, part(res, a, c), n, a, Some(y1)).as_output();
                graph.expand(y);
                // SAFETY: nodes of `ctx`, not freed.
                unsafe {
                    assert_eq!((*y1.raw()).view_src, y0.raw());
                    assert_eq!((*y.raw()).view_src, y0.raw(), "the memory's owner");
                    assert_eq!((*y.raw()).view_offs, 0);
                }
                let plain = ctx
                    .add(super::super::all_reduce(&ctx, handle.id(), x), res)
                    .as_output();
                graph.expand(plain);
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&(0..len).map(|i| value(r, k, i)).collect::<Vec<_>>());
                    res.set_f32(&(0..len).map(|i| value(1, k + 5, i)).collect::<Vec<_>>());
                    backend.compute(&graph).unwrap();
                    out.push((y.to_f32(), plain.to_f32()));
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for k in 0..3 {
        let want: Vec<f32> = (0..len)
            .map(|i| (value(0, k, i) + value(1, k, i)) + value(1, k + 5, i))
            .collect();
        for r in &results {
            assert_eq!(bits(&r[k].0), bits(&want), "parts, round {k}");
            assert_eq!(bits(&r[k].1), bits(&want), "plain, round {k}");
        }
    }
}

/// Gemma 4's output projection in parts (#1181): half 0 of an odd step in two parts, each
/// finished without a residual into columns of one half-0 output (the second into the first's
/// memory), half 1 started between them and finished plainly: the plain all-reduce's bits per
/// column through a CPU graph on two ranks, the nodes no residual and no norm.
#[test]
fn parts_finish_cols_into_one_half() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (row, n) = (6usize, 11i64);
    let (n0, n1) = (n / 2, n - n / 2);
    let parts = [(0i64, 2i64), (2, n0 - 2), (n0, n1)];
    let len = row * n as usize;
    let value = |r: usize, k: usize, i: usize| {
        let x = ((i * 13 + k * 19 + r * 31) % 97) as f32;
        if r == 0 { x * 1e7 } else { x * 0.3 + 1e-3 }
    };
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(20));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = super::super::register(c);
                let mut backend = Backend::cpu().unwrap();
                let ctx = Context::new(48, 48);
                let x = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let part = |t, a: i64, c: i64| {
                    ctx.view_2d(t, [row as i64, c], row * 4, a as usize * row * 4)
                };
                // Each part's partial its own node (a matmul's output in the model).
                let p: Vec<_> = parts
                    .iter()
                    .map(|&(a, c)| ctx.scale(part(x, a, c), 1.0))
                    .collect();
                let mut graph = ctx.new_graph(48);
                let start = |j: usize| all_reduce_start(&ctx, handle.id(), p[j], n as usize);
                let (s0, s1) = (start(0), start(1));
                graph.expand(s0.node);
                graph.expand(s1.node);
                let y0 = finish_cols(&ctx, s0, n0, parts[0].0, None);
                graph.expand(y0);
                let s2 = start(2);
                graph.expand(s2.node);
                let y = finish_cols(&ctx, s1, n0, parts[1].0, Some(y0)).as_output();
                graph.expand(y);
                let h1 = finish(&ctx, s2, None).as_output();
                graph.expand(h1);
                // SAFETY: nodes of `ctx`, not freed.
                unsafe {
                    assert_eq!((*y.raw()).view_src, y0.raw(), "the memory's owner");
                    assert_eq!((*y.raw()).view_offs, 0);
                    assert_eq!(add_of(&*y0.raw()), (None, 0));
                    assert_eq!(add_of(&*y.raw()), (None, parts[1].0 as usize));
                    assert!(norm_of(&*y.raw()).is_none());
                }
                assert_eq!(y.ne(), [row as i64, n0, 1, 1]);
                let plain = super::super::all_reduce(&ctx, handle.id(), x).as_output();
                graph.expand(plain);
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&(0..len).map(|i| value(r, k, i)).collect::<Vec<_>>());
                    backend.compute(&graph).unwrap();
                    let mut joined = y.to_f32();
                    joined.extend(h1.to_f32());
                    out.push((joined, plain.to_f32()));
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for k in 0..3 {
        let want: Vec<f32> = (0..len).map(|i| value(0, k, i) + value(1, k, i)).collect();
        for r in &results {
            assert_eq!(bits(&r[k].0), bits(&want), "parts, round {k}");
            assert_eq!(bits(&r[k].1), bits(&want), "plain, round {k}");
        }
    }
}

/// The finishes that compute the next norm too (#1133): an odd step's halves
/// (`finish_add_norm`, with the BF16 rows) and the plain `all_reduce_add_norm` (codes only)
/// give the sums, codes, scales and BF16 rows of the separate ops (the all-reduce plus the
/// residual, then `fp8_act`'s norm ops) through a CPU graph on two ranks.
#[test]
fn finishes_with_the_norm_match_the_separate_ops() {
    use crate::ggml::{Backend, ComputeAllocator};
    use crate::model::fp8_act;
    use crate::tp_net::{Norm, all_reduce, all_reduce_add_norm, normed};
    let (row, n) = (8usize, 9i64);
    let (n0, n1) = (n / 2, n - n / 2);
    let len = row * n as usize;
    let eps = 1e-6f32;
    let value = |r: usize, k: usize, i: usize| {
        let x = ((i * 7 + k * 13 + r * 31) % 97) as f32;
        if r == 0 {
            x * 1e3 - 4e4
        } else {
            x * 0.3 + 1e-3
        }
    };
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(20));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = super::super::register(c);
                let id = handle.id();
                let mut backend = Backend::cpu().unwrap();
                let ctx = Context::new(64, 64);
                let x = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let w = ctx.new_tensor(types::F32, &[row as i64]).as_input();
                let half = |t, a: i64, c: i64| {
                    ctx.view_2d(t, [row as i64, c], row * 4, a as usize * row * 4)
                };
                let norm = |bf16| Norm {
                    w,
                    eps,
                    bf16,
                    a4: false,
                };
                let mut graph = ctx.new_graph(64);
                let s0 = all_reduce_start(&ctx, id, half(x, 0, n0), n as usize);
                let s1 = all_reduce_start(&ctx, id, half(x, n0, n1), n as usize);
                graph.expand(s0.node);
                graph.expand(s1.node);
                let f0 = finish_add_norm(&ctx, s0, half(res, 0, n0), n, 0, None, norm(true));
                graph.expand(f0);
                let f1 = finish_add_norm(&ctx, s1, half(res, n0, n1), n, n0, Some(f0), norm(true));
                graph.expand(f1);
                let hv = normed(&ctx, f1, row as i64, n, true);
                let pv = all_reduce_add_norm(&ctx, id, x, res, norm(false));
                let y = ctx.add(all_reduce(&ctx, id, x), res);
                let (q, b) = fp8_act::rms_norm_bf16(&ctx, y, w, eps);
                let q2 = fp8_act::rms_norm(&ctx, y, w, eps);
                let outs = [hv.y, hv.q, hv.b.unwrap(), pv.y, pv.q, y, q, b, q2];
                for t in outs {
                    graph.expand(t.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                w.set_f32(&(0..row).map(|i| 0.5 + i as f32 * 0.25).collect::<Vec<_>>());
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&(0..len).map(|i| value(r, k, i)).collect::<Vec<_>>());
                    res.set_f32(&(0..len).map(|i| value(1, k + 5, i)).collect::<Vec<_>>());
                    backend.compute(&graph).unwrap();
                    out.push(outs.map(|t| t.bytes()));
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for (k, r0) in results[0].iter().enumerate() {
        let [hy, hq, hb, py, pq, y, q, b, q2] = r0;
        assert_eq!(hy, y, "halves' sums, round {k}");
        assert_eq!(hq, q, "halves' codes, round {k}");
        assert_eq!(hb, b, "halves' BF16 rows, round {k}");
        assert_eq!(py, y, "plain sums, round {k}");
        assert_eq!(pq, q2, "plain codes, round {k}");
        assert_eq!(&results[1][k], r0, "rank 1, round {k}");
    }
}

/// A whole output's finish with the W4A4 rows of its norm (#1133, `Norm::a4`): on two ranks
/// through a CPU graph, the sums of the all-reduce plus the residual, and per token the codes,
/// scales and decode scale of the W4A4 quantization of `mul(rms_norm(y), w)` (ggml's norm, then
/// `nvfp4_a4::quantize_op`), where `normed_layout` puts them; the node's tokens are its own.
#[test]
fn a4_normed_finish_matches_the_separate_ops() {
    use crate::ggml::{Backend, ComputeAllocator};
    use crate::model::nvfp4_a4;
    use crate::tp_net::{Norm, all_reduce};
    let (row, n) = (128usize, 5i64);
    let len = row * n as usize;
    let eps = 1e-6f32;
    let value = |r: usize, k: usize, i: usize| {
        let x = ((i * 7 + k * 13 + r * 31) % 97) as f32 - 48.0;
        if r == 0 {
            x * 0.37 + if i % 61 == 0 { 300.0 } else { 0.0 }
        } else {
            x * 0.011
        }
    };
    let comms = super::super::tests::comms(2, std::time::Duration::from_secs(20));
    let threads: Vec<_> = comms
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = super::super::register(c);
                let id = handle.id();
                let mut backend = Backend::cpu().unwrap();
                let ctx = Context::new(64, 64);
                let x = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let res = ctx.new_tensor(types::F32, &[row as i64, n]).as_input();
                let w = ctx.new_tensor(types::F32, &[row as i64]).as_input();
                let norm = Norm {
                    w,
                    eps,
                    bf16: false,
                    a4: true,
                };
                let mut graph = ctx.new_graph(64);
                let s = all_reduce_start(&ctx, id, ctx.scale(x, 1.0), n as usize);
                graph.expand(s.node);
                let f = finish_add_norm(&ctx, s, res, n, 0, None, norm).as_output();
                graph.expand(f);
                // SAFETY: a node of `ctx`, not freed.
                let (nm, width) = norm_of(unsafe { &*f.raw() }).expect("a normed finish");
                assert!(nm.a4 && !nm.bf16);
                assert_eq!(width, n as usize);
                let y = ctx.add(all_reduce(&ctx, id, x), res).as_output();
                let q = nvfp4_a4::quantize_op(&ctx, ctx.mul(ctx.rms_norm(y, eps), w)).as_output();
                graph.expand(y);
                graph.expand(q);
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                w.set_f32(&(0..row).map(|i| 0.5 + i as f32 * 0.01).collect::<Vec<_>>());
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&(0..len).map(|i| value(r, k, i)).collect::<Vec<_>>());
                    res.set_f32(&(0..len).map(|i| value(1, k + 5, i)).collect::<Vec<_>>());
                    backend.compute(&graph).unwrap();
                    out.push([f.bytes(), y.bytes(), q.bytes()]);
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    let (k, n) = (row as i64, n as usize);
    let ([codes, scales, dec], _) = nvfp4_a4::normed_layout(k, n as i64);
    let (q_ld, s_ld, [q_codes, q_scales, q_dec]) = nvfp4_a4::quant_layout(k, n as i64);
    let (q_ld, s_ld) = (q_ld as usize, s_ld as usize);
    for (round, r0) in results[0].iter().enumerate() {
        let [f, y, q] = r0;
        assert_eq!(&f[..4 * row * n], &y[..], "sums, round {round}");
        for t in 0..n {
            let part = |b: &[u8], at: usize, len: usize| b[at..at + len].to_vec();
            let at = |base: usize, ld: usize| base + t * ld;
            assert_eq!(
                part(f, at(codes, q_ld), q_ld),
                part(q, at(q_codes, q_ld), q_ld),
                "codes, token {t}, round {round}"
            );
            assert_eq!(
                part(f, at(scales, s_ld), s_ld),
                part(q, at(q_scales, s_ld), s_ld),
                "scales, token {t}, round {round}"
            );
            assert_eq!(
                part(f, at(dec, 4), 4),
                part(q, at(q_dec, 4), 4),
                "decode scale, token {t}, round {round}"
            );
        }
        assert_eq!(&results[1][round], r0, "rank 1, round {round}");
    }
}
