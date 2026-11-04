//! The rewrite on CPU graphs: the chains it takes over against ggml's own nodes computing them.

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context, ffi, op_of, types};

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// A MoE layer's tail as the models build it, for `n` tokens of `e` columns and `k` experts:
/// `mul(experts, w)`, a view per expert and their adds in order, then (with `gate`)
/// `add(sum, mul(sh, sigmoid(g)))`, computed on the CPU, rewritten first when `rewrite`:
/// (output, what the rewrite did).
fn tail(e: i64, k: i64, n: i64, gate: bool, rewrite: bool) -> (Vec<f32>, Counts) {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let ctx = Context::new(256, 256);
    let experts = ctx.new_tensor(types::F32, &[e, k, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[1, k, n]).as_input();
    let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
    // The gate logit a row of a wider matmul output, as in the models.
    let joined = ctx.new_tensor(types::F32, &[3, n]).as_input();
    let g = ctx.view_2d(joined, [1, n], joined.nb()[1], 4);
    let weighted = ctx.mul(experts, w);
    let nb = weighted.nb();
    let view = |j: i64| ctx.view_2d(weighted, [e, n], nb[2], j as usize * nb[1]);
    let mut sum = ctx.add(view(0), view(1));
    for j in 2..k {
        sum = ctx.add(sum, view(j));
    }
    let out = if gate {
        ctx.add(sum, ctx.mul(sh, ctx.sigmoid(g)))
    } else {
        sum
    }
    .as_output();
    let mut graph = ctx.new_graph(256);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run.
    let c = if rewrite {
        unsafe { rewrite_graph(graph.raw(), false) }
    } else {
        Counts::default()
    };
    if rewrite {
        // SAFETY: the output node is a valid tensor.
        let op = unsafe { op_of(out.raw()) }.map(|op| op.name);
        assert_eq!(op, Some("moe_sum_gate"), "the output is the op");
    }
    experts.set_f32(&noise((e * k * n) as usize, 11));
    w.set_f32(&noise((k * n) as usize, 12));
    // Without the gate its inputs are not in the graph and get no memory.
    if gate {
        sh.set_f32(&noise((e * n) as usize, 13));
        let l: Vec<f32> = noise((3 * n) as usize, 14)
            .iter()
            .map(|v| v * 8.0)
            .collect();
        joined.set_f32(&l);
    }
    backend.compute(&graph).unwrap();
    (out.to_f32(), c)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn weighted_sum_rounds_as_ggml() {
    for (e, k, n) in [(64, 10, 1), (64, 10, 3), (33, 2, 5)] {
        let (want, _) = tail(e, k, n, false, false);
        let (got, c) = tail(e, k, n, false, true);
        assert_eq!(c.sum, 1, "{e}x{k}x{n}: {c:?}");
        assert_eq!(c.none, k as usize - 2, "{e}x{k}x{n}: the adds but the last");
        assert!(same_bits(&got, &want), "{e}x{k}x{n}");
    }
}

#[test]
fn weighted_sum_and_gate_as_ggml() {
    for (e, k, n) in [(64, 10, 1), (64, 10, 3), (33, 2, 5)] {
        let (want, _) = tail(e, k, n, true, false);
        let (got, c) = tail(e, k, n, true, true);
        // The inputs are graph inputs (never reused), so the whole tail is one op.
        assert_eq!(c.sum_gate, 1, "{e}x{k}x{n}: {c:?}");
        assert_eq!(
            c.none,
            k as usize + 2,
            "{e}x{k}x{n}: the mul, the other adds, the gate's two"
        );
        // The sum and the gate's mul and add round as ggml's; the CPU's sigmoid may differ from
        // the op's in the last bit (on CUDA both are ggml-cuda's expression).
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                "{e}x{k}x{n} [{i}]: {a} vs {b}"
            );
        }
    }
}

#[test]
fn a_sum_read_elsewhere_is_left_alone() {
    let backend = Backend::cpu().unwrap();
    let ctx = Context::new(64, 64);
    let experts = ctx.new_tensor(types::F32, &[8, 3, 2]).as_input();
    let w = ctx.new_tensor(types::F32, &[1, 3, 2]).as_input();
    let weighted = ctx.mul(experts, w);
    let nb = weighted.nb();
    let view = |j: usize| ctx.view_2d(weighted, [8, 2], nb[2], j * nb[1]);
    let partial = ctx.add(view(0), view(1));
    let sum = ctx.add(partial, view(2)).as_output();
    // The first add read outside the chain: the chain cannot go.
    let other = ctx.add(partial, partial).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(sum);
    graph.expand(other);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run.
    let c = unsafe { rewrite_graph(graph.raw(), false) };
    assert_eq!(c, Counts::default());
}

#[test]
fn ids_remap_is_cont_and_get_rows() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    let (k, n, rows) = (10i64, 3i64, 64i64);
    let ctx = Context::new(64, 64);
    // The ids a strided view (the router's top-k of a wider tensor), so the cont copies.
    // Kept (an output) so the allocator cannot hand its memory to the get_rows, which then
    // overlaps the strided ids and is rightly left alone (so on aarch64).
    let wide = ctx
        .new_tensor(types::I32, &[k + 2, n])
        .as_input()
        .as_output();
    let ids = ctx.view_2d(wide, [k, n], wide.nb()[1], 4);
    let table = ctx.new_tensor(types::I32, &[1, rows]).as_input();
    let flat = ctx.reshape_1d(ctx.cont(ids), k * n);
    let out = ctx.get_rows(table, flat).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run.
    let c = unsafe { rewrite_graph(graph.raw(), false) };
    assert_eq!((c.remap, c.none), (1, 1), "{c:?}");
    // SAFETY: the output node is a valid tensor.
    assert_eq!(
        unsafe { op_of(out.raw()) }.map(|op| op.name),
        Some("ids_remap")
    );
    let w: Vec<i32> = (0..(k + 2) * n)
        .map(|i| ((i * 37 + 5) % rows) as i32)
        .collect();
    let t: Vec<i32> = (0..rows).map(|i| (i * 7 + 1000) as i32).collect();
    wide.set_i32(&w);
    table.set_i32(&t);
    backend.compute(&graph).unwrap();
    let want: Vec<i32> = (0..k * n)
        .map(|i| {
            let (i0, i1) = (i % k, i / k);
            t[w[(i1 * (k + 2) + 1 + i0) as usize] as usize]
        })
        .collect();
    assert_eq!(out.to_i32(), want);
}

/// #836: the readers of each node (ascending, once per source slot, a node reading one twice
/// listed twice) as a scan of every node's sources finds them; tensors that are not nodes
/// have none.
#[test]
fn users_are_each_nodes_readers() {
    let ctx = Context::new(64, 64);
    let x = ctx.new_tensor(types::F32, &[8, 4]).as_input();
    let a = ctx.scale(x, 2.0);
    let b = ctx.add(a, a);
    let v = ctx.view_2d(a, [8, 2], a.nb()[1], 0);
    let c = ctx.mul(b, a);
    let d = ctx.add(ctx.cont(v), ctx.view_2d(c, [8, 2], c.nb()[1], 0));
    let mut graph = ctx.new_graph(64);
    graph.expand(d);
    // SAFETY: a live graph; its node array holds valid tensors.
    unsafe {
        let raw = graph.raw();
        let n = ffi::ggml_graph_n_nodes(raw) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(raw), n).to_vec();
        let g = Graph::new(raw, nodes.clone());
        for &t in &nodes {
            let want: Vec<usize> = (0..n)
                .flat_map(|i| {
                    (*nodes[i])
                        .src
                        .iter()
                        .filter(move |&&s| s == t)
                        .map(move |_| i)
                })
                .collect();
            assert_eq!(g.users(t), want.as_slice());
        }
        let ia = nodes.iter().position(|&t| t == a.raw()).unwrap();
        assert!(g.users(a.raw()).len() >= 4, "{:?}", g.users(a.raw()));
        assert!(g.users(a.raw()).iter().all(|&i| i > ia));
        assert!(g.users(x.raw()).is_empty(), "x is a leaf");
    }
}
