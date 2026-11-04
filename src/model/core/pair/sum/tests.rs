//! The pair all-reduce on two CPU "GPUs" (a thread with a graph each): sums layer after layer,
//! in one piece and in several, over repeated runs of the same graphs.

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context};

const E: i64 = 12;
const LAYERS: usize = 4;

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

/// Rank `me`'s graph: per layer, its part `x * (me + 2) + bias` summed with the other's; the
/// sum is the next layer's input. Runs `runs` times; returns each run's output.
fn rank(sum: &PairSum, me: usize, n: i64, x0: &[f32], runs: usize) -> Vec<Vec<f32>> {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(256, 256);
    let x_in = ctx.new_tensor(types::F32, &[E, n]).as_input();
    let bias = ctx.new_tensor(types::F32, &[E, n]).as_input();
    let mut x = x_in;
    for _ in 0..LAYERS {
        let part = ctx.add(ctx.scale(x, me as f32 + 2.0), bias);
        x = all_reduce(&ctx, sum, me, part);
    }
    let out = x.as_output();
    let mut graph = ctx.new_graph(256);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    (0..runs)
        .map(|_| {
            // The allocator may reuse the inputs' memory for the graph's own tensors.
            x_in.set_f32(x0);
            bias.set_f32(&noise((E * n) as usize, 7 + me as u64));
            backend.compute(&graph).unwrap();
            out.to_f32()
        })
        .collect()
}

fn expected(x0: &[f32]) -> Vec<f32> {
    let b: Vec<Vec<f32>> = (0..2).map(|r| noise(x0.len(), 7 + r)).collect();
    let mut x = x0.to_vec();
    for _ in 0..LAYERS {
        x = (0..x.len())
            .map(|i| {
                let p0 = x[i] * 2.0 + b[0][i];
                let p1 = x[i] * 3.0 + b[1][i];
                p0 + p1
            })
            .collect();
    }
    x
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn check(slot_bytes: usize, n: i64) {
    let cpu = Backend::cpu().unwrap().buffer_type();
    let sum = PairSum::new([cpu, cpu], cpu, slot_bytes).unwrap();
    let x0 = noise((E * n) as usize, 1);
    let want = expected(&x0);
    let (a, b) = std::thread::scope(|s| {
        let h = s.spawn(|| rank(&sum, 1, n, &x0, 3));
        let a = rank(&sum, 0, n, &x0, 3);
        (a, h.join().unwrap())
    });
    assert!(!sum.failed(), "a wait timed out");
    for (run, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(bits(a), bits(&want), "slot {slot_bytes}, run {run}: rank 0");
        assert_eq!(bits(b), bits(&want), "slot {slot_bytes}, run {run}: rank 1");
    }
    // One message per piece per sum on each rank.
    let per_sum = ((E * n * 4) as usize).div_ceil(slot_bytes) as u64;
    for r in 0..2 {
        assert_eq!(
            sum.sent[r].load(Ordering::Relaxed),
            per_sum * LAYERS as u64 * 3
        );
    }
}

#[test]
fn both_ranks_get_the_same_sums() {
    check(SLOT_BYTES, 5);
}

#[test]
fn a_sum_bigger_than_a_slot_goes_in_pieces() {
    // 12 x 7 values = 336 bytes in slots of 64: six pieces, the last short.
    check(64, 7);
}

#[test]
fn pieces_cover_the_sum() {
    let cpu = Backend::cpu().unwrap().buffer_type();
    let sum = PairSum::new([cpu, cpu], cpu, 32).unwrap();
    let p: Vec<_> = sum.pieces(100).collect();
    assert_eq!(p, vec![(0, 32), (32, 32), (64, 32), (96, 4)]);
    assert_eq!(sum.pieces(32).count(), 1);
    assert_eq!(sum.slot_bytes(), 32);
}

#[test]
fn threads_meet_at_the_same_sum() {
    let cpu = Backend::cpu().unwrap().buffer_type();
    let sum = PairSum::new([cpu, cpu], cpu, 32).unwrap();
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            for k in 1..=50 {
                sum.meet(&sum.arrived, 1, k).unwrap();
                sum.meet(&sum.queued, 1, k).unwrap();
            }
        });
        for k in 1..=50 {
            sum.meet(&sum.arrived, 0, k).unwrap();
            // Rank 1 reached sum k: never further than k + 1 while this one is at k.
            assert!(sum.queued[1].load(Ordering::Acquire) <= k);
            sum.meet(&sum.queued, 0, k).unwrap();
        }
        h.join().unwrap();
    });
    assert!(!sum.failed());
}
