//! The share's split rule, its CPU placement and its host side (the pool against the
//! one-thread reference).

use std::sync::atomic::AtomicU32;

use super::pool::{Exchange, Job, Pool, groups, reference, run_now};
use super::*;
use crate::cpu::QuantFormat;
use crate::ggml::{self, types};

#[test]
fn threshold_and_dither() {
    assert_eq!(threshold(0.0), 0);
    assert_eq!(threshold(1.0), THRESHOLD_ONE);
    assert_eq!(threshold(f32::NAN), 0);
    assert_eq!(threshold(2.0), THRESHOLD_ONE);
    assert_eq!(threshold(0.4), 410);
    assert!((0..1000).all(|l| dither(l) < THRESHOLD_ONE));
    // Spread over the range (the layers' counts then average to the share).
    let mean = (0..48).map(|l| dither(l) as f64).sum::<f64>() / 48.0;
    assert!((300.0..724.0).contains(&mean), "mean dither {mean}");
    // The CUDA kernel takes the value from here (`dither` param); pinned anyway.
    assert_eq!(dither(0), {
        let mut h = 0x85EB_CA77u32;
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        h & 1023
    });
}

/// Ids below 4 are VRAM slots, `4 + e` host expert `e` (of 16).
const SLOTS: i32 = 4;

#[test]
fn select_takes_the_least_used_host_experts() {
    // Token 0: slot 0, host 1, host 2; token 1: slot 3, host 2, host 9; token 2: host 2, host 9,
    // host 5. Users: e1 1, e2 3, e5 1, e9 2.
    let ids = [0, 4 + 1, 4 + 2, 3, 4 + 2, 4 + 9, 4 + 2, 4 + 9, 4 + 5];
    assert!(select(&ids, SLOTS, 16, 0, 0).iter().all(|&h| !h));
    let all = select(&ids, SLOTS, 16, THRESHOLD_ONE, 0);
    let host: Vec<bool> = ids.iter().map(|&id| id >= SLOTS).collect();
    assert_eq!(all, host);
    for layer in 0..64 {
        for thr in [100, 256, 410, 512, 700, 900] {
            let s = select(&ids, SLOTS, 16, thr, layer);
            // floor(share * 4 + dither) experts, all pairs of each.
            let c = ((thr * 4 + dither(layer)) / THRESHOLD_ONE) as usize;
            let taken: Vec<i32> = {
                let mut t: Vec<i32> = ids.iter().zip(&s).filter(|p| *p.1).map(|p| *p.0).collect();
                t.sort_unstable();
                t.dedup();
                t
            };
            assert_eq!(taken.len(), c, "layer {layer} threshold {thr}");
            for (i, &id) in ids.iter().enumerate() {
                assert!(id >= SLOTS || !s[i], "a VRAM slot taken");
                assert_eq!(s[i], taken.contains(&id), "pairs of an expert differ");
            }
            // By users, then expert: e1 (1), e5 (1), e9 (2), e2 (3).
            let order = [4 + 1, 4 + 5, 4 + 9, 4 + 2];
            assert_eq!(taken.len(), c);
            for want in &order[..c] {
                assert!(
                    taken.contains(want),
                    "layer {layer} threshold {thr}: {taken:?}"
                );
            }
        }
    }
    // Ids past the stack (another op's markers) are never the host's.
    assert_eq!(
        select(&[4 + 16, -2, 4 + 3], SLOTS, 16, THRESHOLD_ONE, 0),
        [false, false, true]
    );
}

#[test]
fn select_averages_to_the_share() {
    // One distinct host expert per layer: over many layers about `share` of them.
    let thr = threshold(0.3);
    let n = (0..1000)
        .filter(|&l| select(&[SLOTS + 7], SLOTS, 16, thr, l)[0])
        .count();
    assert!((230..370).contains(&n), "{n} of 1000");
}

#[test]
fn pool_cpus_take_smt_threads_per_core() {
    // 6 cores of 4 threads: CPUs 80..104.
    let cpus: Vec<u32> = (80..104).collect();
    let sib = |c: u32| {
        let b = c / 4 * 4;
        (b..b + 4).collect::<Vec<u32>>()
    };
    assert_eq!(
        pool_cpus(&cpus, 2, 2, sib),
        [80, 81, 84, 85, 88, 89, 92, 93]
    );
    assert_eq!(pool_cpus(&cpus, 1, 0, sib), [80, 84, 88, 92, 96, 100]);
    assert_eq!(pool_cpus(&cpus, 4, 5, sib), [80, 81, 82, 83]);
    // Unknown siblings: every CPU its own core.
    assert_eq!(pool_cpus(&[1, 2, 3], 2, 1, |_| Vec::new()), [1, 2]);
    // A core cut by the allowed set keeps what it has.
    assert_eq!(pool_cpus(&[81, 84, 85, 86], 2, 0, sib), [81, 84, 85]);
}

#[test]
fn groups_by_expert_in_first_pair_order() {
    let ids = [3, -1, 7, 7, 3, -1];
    assert_eq!(
        groups(&ids, 3),
        [(3, vec![(0, 0), (1, 1)]), (7, vec![(0, 2), (1, 0)])]
    );
    assert!(groups(&[-1, -1], 2).is_empty());
}

#[test]
fn layout_fits_the_largest_step() {
    let l = Layout::new(2560);
    assert!(l.ready + 4 <= l.done && l.done + 4 <= l.arrived && l.arrived + 4 <= l.ids);
    assert!(l.ids + ops::MAX_TOKENS * ops::MAX_USED * 4 <= l.x);
    assert!(l.x + ops::MAX_TOKENS * 2560 * 4 <= l.out);
    assert_eq!(l.bytes, l.out + ops::MAX_TOKENS * ops::MAX_USED * 2560 * 4);
    assert!(l.done - l.ready >= 128, "flags share a cache line");
}

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// `n_expert` experts of Qwen3.8-Flash-Next's formats (Q4_K up/gate, Q5_1 down) at a small
/// shape, in one host buffer per kind.
fn experts(n_embd: usize, n_ff: usize, n_expert: usize) -> (Vec<Vec<u8>>, HostExperts) {
    let mut rng = Rng(5);
    let mut kinds = Vec::new();
    for (ty, k, rows) in [
        (types::Q4_K, n_embd, n_ff),
        (types::Q4_K, n_embd, n_ff),
        (types::Q5_1, n_ff, n_embd),
    ] {
        let w: Vec<f32> = (0..k * rows * n_expert).map(|_| rng.f() * 0.1).collect();
        kinds.push(ggml::quantize(ty, &w, k));
    }
    let he = HostExperts {
        base: [
            kinds[0].as_ptr() as usize,
            kinds[1].as_ptr() as usize,
            kinds[2].as_ptr() as usize,
        ],
        expert_bytes: [
            kinds[0].len() / n_expert,
            kinds[1].len() / n_expert,
            kinds[2].len() / n_expert,
        ],
        up_gate: QuantFormat::Q4K,
        down: QuantFormat::Q5_1,
        n_embd,
        n_ff,
        n_expert,
    };
    (kinds, he)
}

/// The pool (rows split over threads, two devices' queues, flags as the GPU would raise them)
/// writes the same rows as the one-thread reference, bit for bit, for several steps in a row;
/// rows of other pairs untouched.
#[test]
fn pool_matches_the_reference() {
    let (n_embd, n_ff, n_expert, n_used, n_tok) = (512, 256, 12, 4, 3);
    let (_kinds, he) = experts(n_embd, n_ff, n_expert);
    let pool = Pool::new(&[], 5);
    let queues = [pool.add_queue(), pool.add_queue()];
    let mut rng = Rng(9);
    // Per device: flags, ids, x, out.
    let flags = [
        [AtomicU32::new(0), AtomicU32::new(0)],
        [AtomicU32::new(0), AtomicU32::new(0)],
    ];
    let mut bufs: Vec<(Vec<i32>, Vec<f32>, Vec<f32>)> = (0..2)
        .map(|_| {
            (
                vec![-1; n_tok * n_used],
                vec![0.0; n_tok * n_embd],
                vec![-7.0; n_tok * n_used * n_embd],
            )
        })
        .collect();
    for seq in 1..=6u32 {
        let d = seq as usize % 2;
        let (ids, x, out) = &mut bufs[d];
        // The "GPU" writes the inputs: some pairs the host's (several tokens per expert).
        for (i, id) in ids.iter_mut().enumerate() {
            let e = (i as u32 * 5 + seq) % (n_expert as u32 + 3);
            *id = if e < n_expert as u32 {
                e as i32 % 5
            } else {
                -1
            };
        }
        x.iter_mut().for_each(|v| *v = rng.f());
        out.iter_mut().for_each(|v| *v = -7.0);
        let ex = Exchange {
            ready: &flags[d][0],
            done: &flags[d][1],
            arrived: std::ptr::null(),
            ids: ids.as_ptr(),
            x: x.as_ptr(),
            out: out.as_mut_ptr(),
            max_tok: n_tok,
            max_used: n_used,
            n_embd,
        };
        let job = Job {
            seq: seq.div_ceil(2),
            layer: seq,
            n_tok,
            n_used,
            experts: he,
            exchange: ex,
        };
        // SAFETY: the buffers and experts outlive the call.
        unsafe { run_now(&pool, queues[d], job) };
        let mut want = vec![-7.0f32; out.len()];
        // SAFETY: the experts live in `_kinds`.
        unsafe { reference(&he, n_used, ids, x, &mut want) };
        assert!(groups(ids, n_used).iter().any(|g| g.1.len() > 1));
        assert!(
            out.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {seq}: the pool differs from the reference"
        );
        assert!(
            ids.iter()
                .enumerate()
                .all(|(i, &e)| e >= 0 || out[i * n_embd..][..n_embd].iter().all(|&v| v == -7.0))
        );
    }
    drop(pool);
}
