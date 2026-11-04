use super::*;

#[test]
fn pairs_are_consecutive_gpus() {
    assert_eq!(pair(0, 4), 0..2);
    assert_eq!(pair(1, 4), 0..2);
    assert_eq!(pair(3, 4), 2..4);
    // The last GPU of an odd count is alone.
    assert_eq!(pair(2, 3), 2..3);
    assert_eq!(pair(0, 1), 0..1);
}

#[test]
fn cached_experts_stay_with_their_owner() {
    // Experts 0..72 GPU 0, 72..144 GPU 1: all cached, all on GPU 0.
    let chosen = [(1, 0, true), (5, 0, true), (9, 0, true), (80, 1, true)];
    assert_eq!(assign_pair(&chosen, 4), vec![0, 0, 0, 1]);
}

#[test]
fn misses_go_to_the_less_busy_gpu_of_the_pair() {
    // GPU 0 has three misses, GPU 1 one hit: the misses alternate from GPU 1 on.
    let chosen = [(1, 0, false), (5, 0, false), (9, 0, false), (80, 1, true)];
    // Costs: GPU 1 starts at 1; miss 1 -> GPU 0 (0 < 1), miss 5 -> GPU 1 (1 < 12),
    // miss 9 -> GPU 0 (12 < 13).
    assert_eq!(assign_pair(&chosen, 4), vec![0, 1, 0, 1]);
}

#[test]
fn ties_keep_the_owner_and_pairs_stay_apart() {
    // One miss on GPU 1 with GPU 0 idle: a tie, the owner keeps it. GPU 2's miss never goes
    // to GPU 0 or 1, however idle.
    let chosen = [(80, 1, false), (150, 2, false), (160, 2, false)];
    assert_eq!(assign_pair(&chosen, 4), vec![1, 2, 3]);
}

#[test]
fn every_choice_goes_to_one_gpu_of_its_owners_pair() {
    let mut rng = 0x2545_f491u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for _ in 0..200 {
        let mut chosen: Vec<(u32, usize, bool)> = (0..24)
            .map(|_| {
                let e = (next() % 288) as u32;
                (e, (e / 72) as usize, next() % 5 != 0)
            })
            .collect();
        chosen.sort_unstable_by_key(|c| c.0);
        chosen.dedup_by_key(|c| c.0);
        let got = assign_pair(&chosen, 4);
        for (c, &d) in chosen.iter().zip(&got) {
            assert!(pair(c.1, 4).contains(&d), "{c:?} -> {d}");
            if c.2 {
                assert_eq!(d, c.1);
            }
        }
    }
}

fn entry(real: usize, gpu: usize, chosen: &[(u32, bool)]) -> EntryPass {
    EntryPass {
        real,
        gpu,
        chosen: chosen.to_vec(),
    }
}

#[test]
fn stats_count_owned_work_and_the_pair_balance() {
    let mut s = BalanceStats::default();
    // Layer 3: GPU 0 two misses, GPU 1 nothing, GPUs 2, 3 one hit each. Layer 4: even.
    s.add_pass(&[
        entry(3, 0, &[(1, false), (2, false)]),
        entry(3, 1, &[]),
        entry(3, 2, &[(150, true)]),
        entry(3, 3, &[(220, true)]),
        entry(4, 0, &[(1, true)]),
        entry(4, 1, &[(80, true)]),
        entry(4, 2, &[(150, true)]),
        entry(4, 3, &[(220, true)]),
    ]);
    assert_eq!(s.steps, 1);
    assert_eq!(s.layers, 2);
    assert_eq!(s.hits, vec![1, 1, 2, 2]);
    assert_eq!(s.misses, vec![2, 0, 0, 0]);
    // Misses 2/0/0/0: max 2, mean 0.5.
    assert_eq!(s.miss_excess, 1.5);
    // Costs 24/0/1/1: 24 - 6.5.
    assert_eq!(s.cost_excess, 17.5);
    // Paired: 12/12/1/1: 12 - 6.5.
    assert_eq!(s.pair_excess, 5.5);
    // Cross: GPU 0's misses go to GPUs 0 and 1 (12 < 19): 12/12/1/1 as well.
    assert_eq!(s.cross_excess, 5.5);
    assert_eq!(s.take().steps, 1);
    assert_eq!(s, BalanceStats::default());
}

#[test]
fn stats_skip_single_layer_passes_and_lone_entries() {
    let mut s = BalanceStats::default();
    // An MTP head's pass: one layer.
    s.add_pass(&[entry(46, 0, &[(1, false)]), entry(46, 3, &[(220, true)])]);
    // A pipeline: each layer on one GPU.
    s.add_pass(&[entry(3, 0, &[(1, false)]), entry(4, 1, &[(80, true)])]);
    assert_eq!(s.steps, 0);
    assert_eq!(s.layers, 0);
}

#[test]
fn cross_misses_go_where_they_finish_first() {
    // GPU 0 has four misses, GPUs 1-3 one hit each (costs 0/1/1/1). A miss costs 12 in the
    // owner's pair and 18 on the other pair:
    // 1 -> GPU 0 (12; GPU 1 13, GPUs 2, 3 19), 2 -> GPU 1 (13 < 24), 3 -> GPU 2 (19 < 24, 25),
    // 4 -> GPU 3 (19; GPU 0 24, GPU 1 25, GPU 2 37).
    let chosen = [
        (1, 0, false),
        (2, 0, false),
        (3, 0, false),
        (4, 0, false),
        (80, 1, true),
        (150, 2, true),
        (220, 3, true),
    ];
    assert_eq!(assign_cross(&chosen, 4), vec![0, 1, 2, 3, 1, 2, 3]);
    // The pair balance keeps them on GPUs 0 and 1: 0, 1, 0, 1 (busiest 25 against 19).
    assert_eq!(assign_pair(&chosen, 4), vec![0, 1, 0, 1, 1, 2, 3]);
}

#[test]
fn cross_ties_keep_the_owner_then_the_pair() {
    // GPU 1's miss with GPU 0 idle: 12 on either, the owner keeps it.
    assert_eq!(assign_cross(&[(80, 1, false)], 4), vec![1]);
    // GPU 2 at 6 (six hits) and a miss of its own: 18 there, 18 on idle GPU 0 across the
    // sockets, 12 on idle GPU 3: GPU 3.
    let mut chosen: Vec<(u32, usize, bool)> = (144..150).map(|e| (e, 2, true)).collect();
    chosen.push((150, 2, false));
    let got = assign_cross(&chosen, 4);
    assert_eq!(got[6], 3);
    // GPUs 2 and 3 both at 6: 18 on the owner, 18 on GPU 3, 18 on GPU 0: the owner.
    let mut chosen: Vec<(u32, usize, bool)> = (144..150).map(|e| (e, 2, true)).collect();
    chosen.extend((216..222).map(|e| (e, 3, true)));
    chosen.push((150, 2, false));
    chosen.sort_unstable_by_key(|c| c.0);
    let got = assign_cross(&chosen, 4);
    let i = chosen.iter().position(|c| c.0 == 150).unwrap();
    assert_eq!(got[i], 2);
}

#[test]
fn cross_keeps_cached_experts_and_a_lone_gpu_count() {
    let mut rng = 0x1234_5679u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for n_dev in [2usize, 3, 4] {
        let per = 288 / n_dev as u64;
        for _ in 0..100 {
            let mut chosen: Vec<(u32, usize, bool)> = (0..24)
                .map(|_| {
                    let e = (next() % (per * n_dev as u64)) as u32;
                    (e, (e as u64 / per) as usize, next() % 4 != 0)
                })
                .collect();
            chosen.sort_unstable_by_key(|c| c.0);
            chosen.dedup_by_key(|c| c.0);
            let got = assign_cross(&chosen, n_dev);
            for (c, &d) in chosen.iter().zip(&got) {
                assert!(d < n_dev);
                if c.2 {
                    assert_eq!(d, c.1);
                }
            }
        }
    }
    // Two GPUs are one pair: the cross balance is the pair balance.
    let chosen = [(1, 0, false), (2, 0, false), (3, 0, false), (150, 1, true)];
    assert_eq!(assign_cross(&chosen, 2), assign_pair(&chosen, 2));
}

#[test]
fn stats_count_the_cross_balance() {
    let mut s = BalanceStats::default();
    s.add_pass(&[
        entry(3, 0, &[(1, false), (2, false), (3, false), (4, false)]),
        entry(3, 1, &[(80, true)]),
        entry(3, 2, &[(150, true)]),
        entry(3, 3, &[(220, true)]),
        entry(4, 0, &[(1, true)]),
        entry(4, 1, &[(80, true)]),
        entry(4, 2, &[(150, true)]),
        entry(4, 3, &[(220, true)]),
    ]);
    // Layer 3 owned: 48/1/1/1, mean 12.75. Pair: 24/25/1/1. Cross: 12/13/19/19.
    assert_eq!(s.cost_excess, 48.0 - 12.75);
    assert_eq!(s.pair_excess, 25.0 - 12.75);
    assert_eq!(s.cross_excess, 19.0 - 12.75);
}
