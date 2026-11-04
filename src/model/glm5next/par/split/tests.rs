//! Tests for [`super`] (`--glm-pair-split`).

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context};

/// Four GPUs of 8 experts each: pairs (0, 1) over 0..16 and (2, 3) over 16..32.
const BOUNDS: [u32; 5] = [0, 8, 16, 24, 32];

/// GPU `me`'s matvec ids for `sel` (tiered ids `sel + 100`) through the op on the CPU backend.
fn run(sel: &[i32], cached: &[i32], me: usize) -> Vec<i32> {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(8, 8);
    let n = sel.len() as i64 / 4;
    let s = ctx.new_tensor(types::I32, &[4, n]).as_input();
    let c = ctx
        .new_tensor(types::I32, &[cached.len() as i64])
        .as_input();
    let m = ctx.new_tensor(types::I32, &[4, n]).as_input();
    let out = ep_split(&ctx, s, c, m, &BOUNDS, me).unwrap().as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    s.set_i32(sel);
    c.set_i32(cached);
    m.set_i32(&sel.iter().map(|e| e + 100).collect::<Vec<_>>());
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

/// A cached expert goes whole to its owner, an uncached one in halves to both GPUs of its
/// pair, and nothing to the other pair; the ids are the tiered ones.
#[test]
fn uncached_experts_go_in_halves_to_both_gpus_of_their_pair() {
    // Two tokens of four choices; experts 4, 17, 20 cached.
    let sel = [1, 2, 4, 20, 9, 2, 17, 30];
    let cached = cached_of(&[4, 17, 20]);
    let outs: Vec<Vec<i32>> = (0..4).map(|me| run(&sel, &cached, me)).collect();
    for (i, &e) in sel.iter().enumerate() {
        let owner = (e / 8) as usize;
        let mapped = e + 100;
        for (d, out) in outs.iter().enumerate() {
            let want = if d & !1 != owner & !1 {
                -1
            } else if cached[e as usize] == 0 {
                mapped | expert_mv::HALF_BIT
            } else if d == owner {
                mapped
            } else {
                -1
            };
            assert_eq!(out[i], want, "choice {i} (expert {e}) on GPU {d}");
        }
    }
}

/// The op's output is [`split_host`]'s, the host reference.
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
        let mapped: Vec<i32> = sel.iter().map(|e| e + 100).collect();
        for me in 0..4 {
            let p0 = me & !1;
            let p = [
                BOUNDS[p0] as i32,
                BOUNDS[p0 + 1] as i32,
                BOUNDS[p0 + 2] as i32,
                (me - p0) as i32,
                expert_mv::HALF_BIT,
            ];
            assert_eq!(
                run(&sel, &cached, me),
                split_host(&sel, &cached, &mapped, &p)
            );
        }
    }
}

/// The last GPU of an odd count has no pair: no split ids, and no side.
#[test]
fn a_lone_gpu_has_no_half() {
    let ctx = Context::new(4, 0);
    let s = ctx.new_tensor(types::I32, &[4, 1]);
    let c = ctx.new_tensor(types::I32, &[24]);
    let m = ctx.new_tensor(types::I32, &[4, 1]);
    assert!(ep_split(&ctx, s, c, m, &[0, 8, 16, 24], 2).is_none());
    assert!(ep_split(&ctx, s, c, m, &[0, 8, 16, 24], 1).is_some());
    assert_eq!(side(2, 3), None);
    assert_eq!(
        (side(0, 4), side(1, 4), side(3, 4)),
        (Some(0), Some(1), Some(1))
    );
}
