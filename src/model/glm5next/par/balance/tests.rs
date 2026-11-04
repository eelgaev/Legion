//! Tests for [`super`] (`--glm-pair-balance`).

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context};

/// Four GPUs of 8 experts each: pairs (0, 1) over 0..16 and (2, 3) over 16..32.
const BOUNDS: [u32; 5] = [0, 8, 16, 24, 32];

/// GPU `me`'s assignment of `sel` through the op on the CPU backend.
fn run(sel: &[i32], cached: &[i32], me: usize) -> Vec<i32> {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let s = ctx
        .new_tensor(types::I32, &[4, sel.len() as i64 / 4])
        .as_input();
    let c = ctx
        .new_tensor(types::I32, &[cached.len() as i64])
        .as_input();
    let out = ep_assign(&ctx, s, c, &BOUNDS, me).unwrap().as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    s.set_i32(sel);
    c.set_i32(cached);
    backend.compute(&graph).unwrap();
    out.to_i32()
}

fn cached_of(ids: &[i32]) -> Vec<i32> {
    let mut c = vec![0; 32];
    for &e in ids {
        c[e as usize] = 1;
    }
    c
}

/// Every choice of a pair's experts goes to exactly one GPU of the pair (with that GPU's
/// first expert as its value), and none to the other pair; a cached expert to its owner.
#[test]
fn every_choice_goes_to_one_gpu_of_its_pair() {
    // Two tokens of four choices; experts 1, 2, 3 (GPU 0's) and 9 (GPU 1's) uncached.
    let sel = [1, 2, 3, 20, 9, 2, 4, 17];
    let cached = cached_of(&[4, 17, 20]);
    let outs: Vec<Vec<i32>> = (0..4).map(|me| run(&sel, &cached, me)).collect();
    for (i, &e) in sel.iter().enumerate() {
        let takers: Vec<usize> = (0..4).filter(|&d| outs[d][i] != -1).collect();
        assert_eq!(takers.len(), 1, "choice {i} (expert {e}): {takers:?}");
        let d = takers[0];
        assert_eq!(outs[d][i], BOUNDS[d] as i32);
        let owner = (e / 8) as usize;
        assert_eq!(d & !1, owner & !1, "expert {e} left its pair");
        if cached[e as usize] != 0 {
            assert_eq!(d, owner, "cached expert {e} moved");
        }
    }
    // GPU 0 starts with the cached 4 (cost 1); the uncached in id order go to the less busy:
    // 1 -> GPU 1 (0 < 1), 2 -> GPU 0 (1 < 12), 3 -> GPU 1 (12 < 13), 9 -> GPU 0 (13 < 24).
    let who = |e: i32| (0..4).find(|&d| outs[d][sel.iter().position(|&s| s == e).unwrap()] != -1);
    let got: Vec<Option<usize>> = [1, 2, 3, 4, 9, 17, 20].map(who).to_vec();
    let want = [1, 0, 1, 0, 0, 2, 2].map(Some).to_vec();
    assert_eq!(got, want);
}

/// The op's output is [`assign_host`]'s, the host reference, and a repeated expert (two
/// tokens choosing it) goes to the same GPU both times.
#[test]
fn op_matches_the_host_reference() {
    let mut x = 0x9e37_79b9_u32;
    let mut next = |m: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % m
    };
    for _ in 0..20 {
        let sel: Vec<i32> = (0..12).map(|_| next(32) as i32).collect();
        let cached: Vec<i32> = (0..32).map(|_| (next(3) == 0) as i32).collect();
        for (me, &first) in BOUNDS[..4].iter().enumerate() {
            let p0 = me & !1;
            let p = [
                BOUNDS[p0] as i32,
                BOUNDS[p0 + 1] as i32,
                BOUNDS[p0 + 2] as i32,
                (me - p0) as i32,
                first as i32,
            ];
            let out = run(&sel, &cached, me);
            assert_eq!(out, assign_host(&sel, &cached, &p));
            for i in 0..sel.len() {
                for j in 0..sel.len() {
                    if sel[i] == sel[j] {
                        assert_eq!(out[i], out[j]);
                    }
                }
            }
        }
    }
}

/// GPU `me`'s cross assignment of `sel` through the op on the CPU backend.
fn run_cross(sel: &[i32], cached: &[i32], bounds: &[u32], me: usize) -> Vec<i32> {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let s = ctx
        .new_tensor(types::I32, &[4, sel.len() as i64 / 4])
        .as_input();
    let c = ctx
        .new_tensor(types::I32, &[cached.len() as i64])
        .as_input();
    let out = ep_assign_cross(&ctx, s, c, bounds, me).unwrap().as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    s.set_i32(sel);
    c.set_i32(cached);
    backend.compute(&graph).unwrap();
    out.to_i32()
}

/// `--glm-cross-balance`: every choice goes to exactly one GPU (with its first expert as the
/// value), a cached one to its owner, and GPU 0's four misses spread over all four GPUs.
#[test]
fn cross_spreads_one_gpus_misses_over_both_pairs() {
    // Experts 1, 2, 3, 5 (GPU 0's) uncached; 9, 17, 25 cached on GPUs 1, 2, 3.
    let sel = [1, 2, 3, 9, 5, 17, 25, 2];
    let cached = cached_of(&[9, 17, 25]);
    let outs: Vec<Vec<i32>> = (0..4)
        .map(|me| run_cross(&sel, &cached, &BOUNDS, me))
        .collect();
    let who = |i: usize| {
        let takers: Vec<usize> = (0..4).filter(|&d| outs[d][i] != -1).collect();
        assert_eq!(takers.len(), 1, "choice {i}: {takers:?}");
        assert_eq!(outs[takers[0]][i], BOUNDS[takers[0]] as i32);
        takers[0]
    };
    let got: Vec<usize> = (0..sel.len()).map(who).collect();
    // Costs 0/1/1/1: 1 -> GPU 0, 2 -> GPU 1, 3 -> GPU 2, 5 -> GPU 3 (as in ep_balance's test).
    assert_eq!(got, vec![0, 1, 2, 1, 3, 2, 3, 1]);
}

/// The cross op's output is [`assign_cross_host`]'s for two to four GPUs, and a repeated
/// expert goes to the same GPU both times.
#[test]
fn cross_op_matches_the_host_reference() {
    let mut x = 0x2545_f491_u32;
    let mut next = |m: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % m
    };
    for bounds in [&[0u32, 16, 32][..], &[0, 10, 20, 32], &BOUNDS] {
        for _ in 0..10 {
            let sel: Vec<i32> = (0..12).map(|_| next(32) as i32).collect();
            let cached: Vec<i32> = (0..32).map(|_| (next(3) == 0) as i32).collect();
            let n_dev = bounds.len() - 1;
            let mut taken = vec![0; sel.len()];
            for me in 0..n_dev {
                let mut p = vec![n_dev as i32, me as i32];
                p.extend(bounds.iter().map(|&b| b as i32));
                let out = run_cross(&sel, &cached, bounds, me);
                assert_eq!(out, assign_cross_host(&sel, &cached, &p));
                for i in 0..sel.len() {
                    taken[i] += usize::from(out[i] != -1);
                    for j in 0..sel.len() {
                        if sel[i] == sel[j] {
                            assert_eq!(out[i], out[j]);
                        }
                    }
                }
            }
            assert!(taken.iter().all(|&t| t == 1), "{taken:?}");
        }
    }
}

/// The cross op takes two to [`CROSS_MAX_DEV`] GPUs.
#[test]
fn cross_needs_two_to_seven_gpus() {
    let ctx = Context::new(4, 0);
    let s = ctx.new_tensor(types::I32, &[4, 1]);
    let c = ctx.new_tensor(types::I32, &[24]);
    assert!(ep_assign_cross(&ctx, s, c, &[0, 24], 0).is_none());
    assert!(ep_assign_cross(&ctx, s, c, &[0, 8, 16, 24], 2).is_some());
    let eight: Vec<u32> = (0..=8).map(|d| d * 3).collect();
    assert!(ep_assign_cross(&ctx, s, c, &eight, 0).is_none());
}

/// The last GPU of an odd count has no pair: no assignment, its owned choices as before.
#[test]
fn a_lone_gpu_keeps_its_owned_choices() {
    let ctx = Context::new(4, 0);
    let s = ctx.new_tensor(types::I32, &[4, 1]);
    let c = ctx.new_tensor(types::I32, &[24]);
    assert!(ep_assign(&ctx, s, c, &[0, 8, 16, 24], 2).is_none());
    assert!(ep_assign(&ctx, s, c, &[0, 8, 16, 24], 1).is_some());
}
