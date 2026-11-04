//! The pair exchange on two CPU "GPUs" (threads with a graph each), against the same sums
//! computed here: tasks out, parts back, layer after layer, over two rounds.

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context};

const E: i64 = 8;
const USED: i64 = 2;
const N: i64 = 3;
const LAYERS: usize = 3;
const ROWS: i64 = 5;

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

fn ids(l: usize) -> Vec<i32> {
    (0..USED * N)
        .map(|i| ((i as usize * 3 + l) % ROWS as usize) as i32)
        .collect()
}

/// The owner (GPU 0): per layer, sends its rows with the choices and weights, computes `3 x`,
/// adds the helper's part; the sum is the next layer's rows.
fn owner(link: &PairLink, x0: &[f32]) -> Vec<f32> {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(256, 256);
    let x_in = ctx.new_tensor(types::F32, &[E, N]).as_input();
    let ins: Vec<_> = (0..LAYERS)
        .map(|_| {
            let i = ctx.new_tensor(types::I32, &[USED, N]).as_input();
            let w = ctx.new_tensor(types::F32, &[USED, N]).as_input();
            (i, w)
        })
        .collect();
    let mut x = x_in;
    for &(i, w) in &ins {
        let send = send_task(&ctx, link, 0, 1, x, i, w);
        let own = ctx.scale(x, 3.0);
        x = recv_add(&ctx, link, 0, &[send], own, &[]);
    }
    let out = x.as_output();
    let mut graph = ctx.new_graph(256);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x_in.set_f32(x0);
    for (l, &(i, w)) in ins.iter().enumerate() {
        i.set_i32(&ids(l));
        w.set_f32(&noise((USED * N) as usize, 100 + l as u64));
    }
    backend.compute(&graph).unwrap();
    out.to_f32()
}

/// The helper (GPU 1): per layer, takes the task, computes `table[:, ids[0, t]] * w[0, t] +
/// 2 x` and sends it back.
fn helper(link: &PairLink, table: &[f32]) {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(256, 256);
    let tab = ctx.new_tensor(types::F32, &[E, ROWS]).as_input();
    let mut prev: Option<Tensor<'_>> = None;
    for _ in 0..LAYERS {
        let i = recv_task(&ctx, link, 1, N, prev.as_slice());
        let g = ctx.get_rows(tab, ctx.reshape_1d(i, USED * N));
        let g = ctx.reshape_3d(g, E, USED, N);
        let first = ctx.cont(ctx.view_2d(g, [E, N], g.nb()[2], 0));
        let w = link.task_w(&ctx, 1, N);
        let w0 = ctx.cont(ctx.view_2d(w, [1, N], w.nb()[1], 0));
        // The rows are read after the take (the add's first source holds it).
        let part = ctx.add(ctx.mul(first, w0), ctx.scale(link.task_x(&ctx, 1, N), 2.0));
        prev = Some(send_part(&ctx, link, 1, 0, part));
    }
    let mut graph = ctx.new_graph(256);
    graph.expand(prev.unwrap().as_output());
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    tab.set_f32(table);
    backend.compute(&graph).unwrap();
}

fn expected(x0: &[f32], table: &[f32]) -> Vec<f32> {
    let mut x = x0.to_vec();
    for l in 0..LAYERS {
        let (i, w) = (ids(l), noise((USED * N) as usize, 100 + l as u64));
        x = (0..(E * N) as usize)
            .map(|k| {
                let (r, t) = (k % E as usize, k / E as usize);
                let g = table[i[t * USED as usize] as usize * E as usize + r];
                let part = g * w[t * USED as usize] + x[k] * 2.0;
                x[k] * 3.0 + part
            })
            .collect();
    }
    x
}

#[test]
fn tasks_go_out_and_parts_come_back_summed() {
    let cpu = Backend::cpu().unwrap().buffer_type();
    let link = PairLink::new(&[cpu, cpu], cpu, E, USED, 4).unwrap();
    let x0 = noise((E * N) as usize, 1);
    let table = noise((E * ROWS) as usize, 2);
    let want = expected(&x0, &table);
    // Two rounds: the second's values are past every flag the first left.
    for round in 1..=2 {
        link.begin(round);
        let got = std::thread::scope(|s| {
            let h = s.spawn(|| helper(&link, &table));
            let got = owner(&link, &x0);
            h.join().unwrap();
            got
        });
        assert!(!link.failed(), "a wait timed out");
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "round {round}"
        );
    }
}

#[test]
fn a_round_restarts_the_counts_above_the_last() {
    let cpu = Backend::cpu().unwrap().buffer_type();
    let link = PairLink::new(&[cpu, cpu], cpu, E, USED, 4).unwrap();
    link.begin(3);
    let base = 3u64 << ROUND_SHIFT;
    for d in 0..2 {
        for k in 0..2 {
            assert_eq!(link.posted[d][k].load(Ordering::Relaxed), base);
            assert_eq!(link.taken[d][k].load(Ordering::Relaxed), base);
        }
    }
    // A round's messages stay below the next round's first value.
    assert!(base + (1 << ROUND_SHIFT) - 1 < (4u64 << ROUND_SHIFT) + 1);
    assert!(!link.failed());
}
