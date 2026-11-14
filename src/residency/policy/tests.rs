//! Slot bookkeeping, swap planning and slot sizing.

use super::*;

#[test]
fn table_points_at_ready_slots_only() {
    let mut l = LayerSlots::new(8, 3);
    l.fill(&[5, 2, 5, 7]);
    assert_eq!(l.ready(), vec![Some(5), Some(2), Some(7)]);
    assert_eq!(l.table(), vec![3, 4, 1, 6, 7, 0, 9, 2]);
    assert_eq!(l.decode(1), Some((2, true)));
    assert_eq!(l.decode(3 + 6), Some((6, false)));
    assert_eq!(l.decode(3 + 8), None);
}

#[test]
fn serves_only_ready_experts() {
    let mut l = LayerSlots::new(4, 2);
    l.fill(&[1]);
    assert!(l.serves(1));
    assert!(!l.serves(0));
    // A filling slot does not serve its expert yet.
    assert_eq!(l.refill(&[3]), vec![(1, 3)]);
    assert!(!l.serves(3));
    l.complete(1);
    assert!(l.serves(3));
    assert!(!l.serves(9), "past the experts");
}

#[test]
fn record_counts_hits_and_heat() {
    let mut l = LayerSlots::new(4, 1);
    l.fill(&[1]);
    // expert 1 from its slot (0), expert 3 from host (1 + 3), twice.
    l.record(&[0, 4, 4]);
    assert_eq!((l.hits, l.misses), (1, 2));
    assert_eq!(l.heat(), &[0.0, 1.0, 0.0, 2.0]);
    l.decay(0.5);
    assert_eq!(l.heat(), &[0.0, 0.5, 0.0, 1.0]);
}

#[test]
fn record_capped_limits_heat_per_pass_not_counts() {
    let mut l = LayerSlots::new(4, 1);
    l.fill(&[1]);
    // expert 1 from its slot (0) three times, expert 3 from host (1 + 3) twice, expert 2 once.
    let ids = [0, 4, 0, 4, 0, 3];
    l.record_capped(&ids, 1.0);
    assert_eq!((l.hits, l.misses), (3, 3));
    assert_eq!(l.heat(), &[0.0, 1.0, 1.0, 1.0]);
    l.record_capped(&ids, 2.0);
    assert_eq!((l.hits, l.misses), (6, 6));
    assert_eq!(l.heat(), &[0.0, 3.0, 2.0, 3.0]);
    // 0: no cap, the same as `record`.
    let (mut a, mut b) = (LayerSlots::new(4, 1), LayerSlots::new(4, 1));
    a.fill(&[1]);
    b.fill(&[1]);
    a.record_capped(&ids, 0.0);
    b.record(&ids);
    assert_eq!(a.heat(), b.heat());
    assert_eq!((a.hits, a.misses), (b.hits, b.misses));
}

#[test]
fn record_capped_skips_unowned_experts() {
    let mut l = LayerSlots::new(4, 0);
    l.set_owned(0..2);
    l.record_capped(&[1, 1, 3, 3], 1.0);
    assert_eq!(l.heat(), &[0.0, 1.0, 0.0, 0.0]);
    assert_eq!((l.hits, l.misses), (0, 2));
}

#[test]
fn heat_pass_cap_keeps_one_pass_burst_out() {
    // A host expert chosen by all 3 tokens of one pass: uncapped it beats a cold holder by
    // more than the default margin (3 > 1.5) and is copied in; capped at 1 it waits for a
    // second pass.
    let rule = SwapRule::default();
    let mut free = LayerSlots::new(4, 1);
    free.fill(&[0]);
    free.record_capped(&[1 + 2; 3], 0.0);
    assert_eq!(plan_swaps(&mut [Some(&mut free)], rule).len(), 1);
    let mut capped = LayerSlots::new(4, 1);
    capped.fill(&[0]);
    capped.record_capped(&[1 + 2; 3], 1.0);
    assert!(plan_swaps(&mut [Some(&mut capped)], rule).is_empty());
    capped.decay(0.95);
    capped.record_capped(&[1 + 2; 3], 1.0);
    assert_eq!(plan_swaps(&mut [Some(&mut capped)], rule).len(), 1);
}

#[test]
fn swap_takes_slot_out_of_table_until_complete() {
    let mut l = LayerSlots::new(4, 1);
    l.fill(&[0]);
    l.record(&[1 + 2; 10]);
    let rule = SwapRule {
        budget: 4,
        margin: 1.0,
        ratio: 0.1,
    };
    let swaps = plan_swaps(&mut [Some(&mut l)], rule);
    assert_eq!(
        swaps,
        vec![Swap {
            layer: 0,
            slot: 0,
            old: Some(0),
            new: 2
        }]
    );
    // While filling, both experts read from host.
    assert_eq!(l.table(), vec![1, 2, 3, 4]);
    // Nothing more to plan for that slot while it fills.
    assert!(plan_swaps(&mut [Some(&mut l)], rule).is_empty());
    l.complete(0);
    assert_eq!(l.table(), vec![1, 2, 0, 4]);
}

#[test]
fn swaps_need_a_clear_gain_and_respect_the_budget() {
    let mut a = LayerSlots::new(4, 1);
    a.fill(&[0]);
    a.record(&[0; 10]); // expert 0: 10 (resident)
    a.record(&[1 + 1; 11]); // expert 1: 11, not enough over 10 with a margin of 1.5
    let mut b = LayerSlots::new(4, 2);
    b.record(&[2 + 3; 3]); // empty slots fill with anything routed
    b.record(&[2 + 2; 5]);
    let rule = SwapRule {
        budget: 1,
        ..SwapRule::default()
    };
    let swaps = plan_swaps(&mut [Some(&mut a), None, Some(&mut b)], rule);
    assert_eq!(swaps.len(), 1);
    assert_eq!(swaps[0].layer, 2);
    assert_eq!(swaps[0].new, 2, "the hottest first");
    let swaps = plan_swaps(&mut [Some(&mut a), None, Some(&mut b)], rule);
    assert_eq!((swaps[0].layer, swaps[0].new), (2, 3));
    assert!(plan_swaps(&mut [Some(&mut a), None, Some(&mut b)], rule).is_empty());
}

#[test]
fn slots_follow_the_ranking_then_fill_round_robin() {
    // Layer 1 is on another device (no slot size).
    let ranked = [(0, 3), (2, 1), (1, 0), (0, 3), (0, 1), (2, 2)];
    let got = allocate_slots(&ranked, &[10, 0, 20], 4, 70);
    // Ranked: (0,3) 10, (2,1) 20, (0,1) 10, (2,2) 20 = 60; then round robin: layer 0
    // takes expert 0 (10 bytes), layer 2 cannot fit 20 more.
    assert_eq!(got, vec![vec![3, 1, 0], vec![], vec![1, 2]]);
    let none = allocate_slots(&[], &[10, 10], 3, 45);
    assert_eq!(none, vec![vec![0, 1], vec![0, 1]]);
    // Layer 1 is not in the ranking: an even share (40 / 2 = 2 slots) first.
    let got = allocate_slots(&[(0, 2), (0, 1), (0, 0)], &[10, 10], 4, 40);
    assert_eq!(got, vec![vec![2, 1], vec![0, 1]]);
}

#[test]
fn clear_and_refill_keep_the_residents() {
    let mut l = LayerSlots::new(8, 3);
    l.fill(&[5, 2, 7]);
    let held = l.clear();
    assert_eq!(held, vec![5, 2, 7]);
    // Every expert reads from host memory.
    assert_eq!(l.table(), (3..11).collect::<Vec<i32>>());
    let copies = l.refill(&held);
    assert_eq!(copies, vec![(0, 5), (1, 2), (2, 7)]);
    // Filling slots are not in the table until their copies complete.
    assert_eq!(l.table()[5], 3 + 5);
    l.complete(0);
    assert_eq!(l.table()[5], 0);
    assert_eq!(l.table()[2], 3 + 2);
    // A second refill finds no empty slot.
    assert!(l.refill(&[1]).is_empty());
}

/// The planning before it ranked only `budget` candidates: every host expert and slot of a
/// layer sorted in full (stably), then paired.
fn full_sort_plan(layers: &mut [LayerSlots], rule: SwapRule) -> Vec<Swap> {
    let mut all: Vec<(f32, Swap)> = Vec::new();
    for (layer, l) in layers.iter().enumerate() {
        let mut hot: Vec<u32> = (0..l.slot_of.len() as u32)
            .filter(|&e| l.slot_of[e as usize].is_none() && l.heat[e as usize] > 0.0)
            .collect();
        hot.sort_by(|a, b| l.heat[*b as usize].total_cmp(&l.heat[*a as usize]));
        let mut cold: Vec<(f32, u32, Option<u32>)> = l
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, slot)| match *slot {
                Slot::Empty => Some((f32::NEG_INFINITY, s as u32, None)),
                Slot::Ready(e) => Some((l.heat[e as usize], s as u32, Some(e))),
                Slot::Filling(_) => None,
            })
            .collect();
        cold.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (&new, &(h_old, slot, old)) in hot.iter().zip(&cold) {
            let h_new = l.heat[new as usize];
            let gain = h_new - h_old.max(0.0);
            if !(old.is_none() || (gain > rule.margin && h_new > h_old * (1.0 + rule.ratio))) {
                break;
            }
            all.push((
                gain,
                Swap {
                    layer,
                    slot,
                    old,
                    new,
                },
            ));
        }
    }
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    all.truncate(rule.budget);
    for (_, s) in &all {
        layers[s.layer].start(s.slot, s.old, s.new);
    }
    all.into_iter().map(|(_, s)| s).collect()
}

/// Ranking only the budget's worth of candidates plans exactly the swaps a full sort did,
/// ties included (heats of whole routings collide often), over many rounds of random routing
/// with copies completing in between.
#[test]
fn budgeted_ranking_plans_the_full_sorts_swaps() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    for budget in [1, 3, 8, 24] {
        let rule = SwapRule {
            budget,
            ..SwapRule::default()
        };
        let (n_expert, n_slots) = (64, 12);
        let mut a: Vec<LayerSlots> = (0..3)
            .map(|i| {
                let mut l = LayerSlots::new(n_expert, n_slots - 2 * i);
                l.fill(
                    &(0..(n_slots - 4) as u32)
                        .map(|e| e * 3 % 64)
                        .collect::<Vec<_>>(),
                );
                l
            })
            .collect();
        let mut b = a.clone();
        for _ in 0..200 {
            for (la, lb) in a.iter_mut().zip(b.iter_mut()) {
                let ids: Vec<i32> = (0..10)
                    .map(|_| {
                        let e = (rnd(16) * rnd(4)) as usize % n_expert;
                        let t = la.table();
                        t[e]
                    })
                    .collect();
                la.record(&ids);
                lb.record(&ids);
                la.decay(0.9);
                lb.decay(0.9);
            }
            let mut va: Vec<Option<&mut LayerSlots>> = a.iter_mut().map(Some).collect();
            let got = plan_swaps(&mut va, rule);
            let want = full_sort_plan(&mut b, rule);
            assert_eq!(got, want, "budget {budget}");
            // Some copies finish before the next step.
            for s in got.iter().filter(|_| rnd(3) > 0) {
                a[s.layer].complete(s.slot);
                b[s.layer].complete(s.slot);
            }
        }
    }
}

/// The budget binds (more slots and candidates than it takes), with hand-set heats full of
/// ties, infinities and NaNs: the budgeted ranking still plans the full sort's swaps.
#[test]
fn budgeted_ranking_matches_with_ties_and_odd_heats() {
    let vals = [
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.5,
        f32::INFINITY,
        f32::NAN,
        -0.0,
        7.0,
    ];
    for budget in [2, 8, 24] {
        let rule = SwapRule {
            budget,
            ..SwapRule::default()
        };
        for seed in 0..50u64 {
            let mut l = LayerSlots::new(96, 40);
            l.fill(
                &(0..40u32)
                    .map(|e| (e * 7 + seed as u32) % 96)
                    .collect::<Vec<_>>(),
            );
            for (e, h) in l.heat.iter_mut().enumerate() {
                *h = vals[(e * 31 + seed as usize * 17) % vals.len()];
            }
            let mut a = [l.clone()];
            let mut b = [l];
            let mut va: Vec<Option<&mut LayerSlots>> = a.iter_mut().map(Some).collect();
            let got = plan_swaps(&mut va, rule);
            assert_eq!(
                got,
                full_sort_plan(&mut b, rule),
                "budget {budget} seed {seed}"
            );
        }
    }
}

#[test]
fn clear_from_keeps_the_first_slots() {
    let mut l = LayerSlots::new(8, 4);
    l.fill(&[5, 2, 7, 1]);
    let held = l.clear_from(2);
    assert_eq!(held, vec![7, 1]);
    assert_eq!(l.table(), vec![4, 4 + 1, 1, 4 + 3, 4 + 4, 0, 4 + 6, 4 + 7]);
    // The refill fills the emptied slots, in order.
    assert_eq!(l.refill(&held), vec![(2, 7), (3, 1)]);
}

#[test]
fn a_lend_refills_the_hottest_experts_it_did_not_keep() {
    let mut l = LayerSlots::new(8, 4);
    assert_eq!(l.ranked(), None);
    l.fill(&[5, 2, 7, 1]);
    // A long prompt keeps slots 0 and 1 and routes to 3 most, then 2, then 7.
    let held = l.clear_from(2);
    l.add_heat(&[0.0, 0.0, 5.0, 9.0, 0.0, 0.5, 0.0, 4.0]);
    let ranked = l.ranked().expect("routed");
    assert_eq!(ranked[..4], [3, 2, 7, 5]);
    // The kept 5 and 2 stay; the lent slots take 3 and 7, not 7 and 1.
    assert_eq!(l.refill(&ranked), vec![(2, 3), (3, 7)]);
    assert_eq!(held, vec![7, 1]);
}

#[test]
fn auto_keep_leaves_the_ring_and_the_need() {
    const MIB: u64 = 1 << 20;
    // 12 layers of 300 slots of 3 MiB over 512 experts, a ring of 2.
    let layers = vec![(300, 3 * MIB); 12];
    let keep = auto_keep(&layers, 0, 512, 2, 1536 * MIB);
    assert!(keep.iter().all(|&k| k == keep[0] && k > 0 && k < 300));
    let lent = 12 * (300 - keep[0]) as u64 * 3 * MIB;
    let ring = 2 * ((512 - keep[0] + 1) as u64 * 3 * MIB + 6 * MIB);
    assert!(lent - 12 * 6 * MIB >= ring + 1536 * MIB);
    // One more slot each would not leave it.
    let more = 12 * (299 - keep[0]) as u64 * 3 * MIB;
    let ring = 2 * ((511 - keep[0] + 1) as u64 * 3 * MIB + 6 * MIB);
    assert!(more - 12 * 6 * MIB < ring + 1536 * MIB);
    // Nothing fits: nothing kept.
    assert_eq!(auto_keep(&[(10, 3 * MIB)], 0, 512, 2, 0), vec![0]);
    // A layer that lends all its slots unstaged leaves the staged ones more.
    let with = auto_keep(&layers, 200 * 3 * MIB, 512, 2, 1536 * MIB);
    assert!(with[0] > keep[0]);
}

/// What a lend with `keep` makes of `layers` ([`lend_parts`], [`plan_staging`]): the bytes no
/// staging buffer takes and the new VRAM the buffers need.
fn carve(layers: &[LendLayer], keep: &[usize], n_expert: usize, cover: bool) -> (u64, u64) {
    let (parts, bytes) = lend_parts(layers, keep, n_expert, GRANULE as usize);
    let plan = crate::residency::experts::plan_staging(&parts, bytes, 2, cover);
    let mut taken = vec![false; parts.len()];
    let mut new = 0;
    for (i, p) in &plan {
        let have: usize = p.iter().map(|&j| parts[j]).sum();
        new += bytes[i / 2]
            .next_multiple_of(GRANULE as usize)
            .saturating_sub(have) as u64;
        p.iter().for_each(|&j| taken[j] = true);
    }
    let left = (0..parts.len())
        .filter(|&j| !taken[j])
        .map(|j| parts[j] as u64)
        .sum();
    (left, new)
}

// Qwen3.8-Flash-Next UD-Q4_K_XL slots per kind (gate, up, down): Q4_K / Q4_K / Q5_1 or Q8_0
// down; the MTP head's all Q8_0.
const GATE: usize = 921_600;
const DOWN: usize = 1_228_800;
const DOWN8: usize = 1_740_800;
/// The bytes ggml-cuda's quantized matmuls read past a Q5_1 down slot's last row (640 columns
/// padded to 1024: 12 blocks of 24 bytes).
const DOWN_OVER: usize = 288;

/// #814: a layer of 364 down slots kept to 237 cuts its tail at a granule where slot 235 ends.
/// While the tail is lent nothing is mapped there, and the matmuls' reads past that slot's
/// last row fault, so it is not kept; a slot that ends before the cut with room stays kept.
#[test]
fn a_slot_ending_at_the_tail_is_not_kept() {
    let gran = GRANULE as usize;
    let geometry = |slots: usize, keep: usize| {
        let vram = (slots * DOWN).next_multiple_of(gran);
        let first = vram - slots * DOWN;
        (first, (first + keep * DOWN) / gran * gran)
    };
    let (first, split) = geometry(364, 237);
    assert_eq!(first + 236 * DOWN, split);
    assert_eq!(kept_slots(first, split, DOWN, 0, 364), 236);
    assert_eq!(kept_slots(first, split, DOWN, DOWN_OVER, 364), 235);
    // Room past the last kept slot: the same count.
    let (first, split) = geometry(334, 237);
    assert!(first + 236 * DOWN + DOWN_OVER <= split);
    assert_eq!(kept_slots(first, split, DOWN, DOWN_OVER, 334), 236);
    // A whole slot part (no tail) keeps every slot.
    assert_eq!(kept_slots(first, split, DOWN, DOWN_OVER, 10), 10);
    // The staging buffers then hold one expert more of each kind.
    let layer = |over| LendLayer {
        slots: 364,
        bytes: [GATE, GATE, DOWN],
        over: [0, 0, over],
        staged: true,
    };
    let (_, without) = lend_parts(&[layer(0)], &[237], 512, gran);
    let (_, with) = lend_parts(&[layer(DOWN_OVER)], &[237], 512, gran);
    for k in 0..3 {
        assert_eq!(with[k], without[k] + [GATE, GATE, DOWN][k]);
    }
}

/// #767: a 4-GPU AC922's last stage at `--layer-split 14,28,42` (layers 42..48 and the MTP head, 486
/// slots each). By bytes alone, the MTP head's 2.4 GiB of lent slots hold the ring and the
/// long step, so every staged layer kept all of its slots; but the staging buffers then took
/// the head's three allocations whole and needed new VRAM besides, and the prompt's compute
/// buffers spilled into host memory. Carved, the staged layers lend enough slot tails for the
/// buffers, and the head's slots stay free for the prompt.
#[test]
fn auto_keep_carved_leaves_a_short_stage_its_need() {
    const MIB: u64 = 1 << 20;
    let need = 1470 * MIB;
    let staged = |down| LendLayer {
        slots: 486,
        bytes: [GATE, GATE, down],
        over: [0; 3],
        staged: true,
    };
    let mut layers = vec![staged(DOWN); 4];
    layers.extend([staged(DOWN8), staged(DOWN8)]);
    layers.push(LendLayer {
        slots: 486,
        bytes: [DOWN8; 3],
        over: [0; 3],
        staged: false,
    });
    let (keep, by_bytes) = auto_keep_carved(&layers, 512, 2, need, true);
    assert_eq!(by_bytes, vec![486; 6]);
    // By bytes: no VRAM left for the prompt, new VRAM for the buffers.
    let (left, new) = carve(&layers, &by_bytes, 512, true);
    assert!(new > 0 && left < need / 2, "{left} {new}");
    // Carved: a little less kept, the buffers made of the lent slots, the need left.
    assert!(keep.iter().all(|&k| k > 440 && k < 486), "{keep:?}");
    let (left, new) = carve(&layers, &keep, 512, true);
    assert_eq!(new, 0);
    assert!(left + 3 * GRANULE * 6 >= need, "{left}");
    // One more slot each would not leave it.
    let more: Vec<usize> = keep.iter().map(|k| k + 1).collect();
    let (left, new) = carve(&layers, &more, 512, true);
    assert!(left.saturating_sub(new) + 3 * GRANULE * 6 < need);
}

/// The even split's stages (12 layers, no MTP head or one) carve as the byte count expects:
/// the keep does not change (#767: the default layout must keep its sizing).
#[test]
fn auto_keep_carved_keeps_an_even_split() {
    const MIB: u64 = 1 << 20;
    for (slots, mtp, need) in [
        (270, None, 1335 * MIB),
        (300, None, 1339 * MIB),
        (250, Some(250), 1323 * MIB),
        (180, Some(180), 1323 * MIB),
    ] {
        let mut layers: Vec<LendLayer> = (0..12)
            .map(|i| LendLayer {
                slots,
                bytes: [GATE, GATE, if i % 5 == 2 { DOWN8 } else { DOWN }],
                over: [0; 3],
                staged: true,
            })
            .collect();
        layers.extend(mtp.map(|slots| LendLayer {
            slots,
            bytes: [DOWN8; 3],
            over: [0; 3],
            staged: false,
        }));
        for cover in [true, false] {
            let (keep, by_bytes) = auto_keep_carved(&layers, 512, 2, need, cover);
            assert_eq!(keep, by_bytes, "{slots} {mtp:?} {cover}");
            assert!(keep[0] > 0, "{slots}");
        }
    }
}

#[test]
fn auto_keep_stages_only_a_layers_own_share() {
    const MIB: u64 = 1 << 20;
    // GLM's parallel decode on a V100: 42 entries of 34 slots of 6.8 MiB, each owning 64 of
    // the 256 experts (the staging ring holds only those), a ring of 2.
    let b = 68 * MIB / 10;
    let layers = vec![(34, b); 42];
    let all = auto_keep(&layers, 0, 256, 2, 49 * MIB);
    let own = auto_keep(&layers, 0, 64, 2, 49 * MIB);
    assert!(own[0] > all[0], "{own:?} vs {all:?}");
    // The share it lends still holds two buffers of the 64 less its kept slots, and the need.
    let lent = 42 * (34 - own[0]) as u64 * b;
    let ring = 2 * ((64 - own[0] + 1) as u64 * b + 6 * MIB);
    assert!(lent - 42 * 6 * MIB >= ring + 49 * MIB);
}

#[test]
fn owned_slots_take_only_their_experts() {
    // Layer 0 owns experts 4..8, layer 1 any: the ranking's other experts of layer 0 are
    // skipped, and the round robin fills layer 0 from its own range only.
    let ranked = [(0, 1), (0, 5), (1, 2), (0, 6), (1, 0)];
    let owned = [Some(4..8), None];
    let out = allocate_slots_in(&ranked, &[10, 10], &owned, 8, 80);
    assert_eq!(out[0], vec![5, 6, 4, 7]);
    assert_eq!(out[1], vec![2, 0, 1, 3]);
    // Without owners it is the plain allocation.
    assert_eq!(
        allocate_slots_in(&ranked, &[10, 10], &[], 8, 80),
        allocate_slots(&ranked, &[10, 10], 8, 80)
    );
    // An even share for a layer the ranking does not name counts its own experts only.
    let out = allocate_slots_in(&[(1, 3)], &[10, 10], &[Some(6..8), None], 8, 1000);
    assert_eq!(out[0], vec![6, 7]);
}

#[test]
fn split_only_keeps_the_counts_with_the_lowest_experts() {
    // The ranking gives layer 1 three slots and layer 0 one; the contents become each
    // layer's lowest experts (its own range under expert parallelism).
    let ranked = [(1, 5), (1, 2), (0, 7), (1, 6)];
    let alloc = allocate_slots(&ranked, &[10, 10], 8, 40);
    assert_eq!(alloc, vec![vec![7], vec![5, 2, 6]]);
    assert_eq!(lowest_experts(&alloc, &[], 8), vec![vec![0], vec![0, 1, 2]]);
    assert_eq!(
        lowest_experts(&alloc, &[Some(4..8), None], 8),
        vec![vec![4], vec![0, 1, 2]]
    );
    // An empty layer stays empty.
    assert_eq!(
        lowest_experts(&[vec![], vec![3]], &[], 4),
        vec![vec![], vec![0]]
    );
}

#[test]
fn owned_layer_ignores_the_others_routings() {
    let mut l = LayerSlots::new(8, 2);
    l.set_owned(4..8);
    l.fill(&[4]);
    // Expert 1 (host id 2 + 1) and expert 5 (host), expert 4 (slot 0).
    l.record(&[3, 7, 0, 3]);
    assert_eq!((l.hits, l.misses), (1, 1));
    assert_eq!(l.heat()[1], 0.0);
    l.add_heat(&[9.0; 8]);
    assert_eq!(l.heat()[0], 0.0);
    assert!(l.ranked().unwrap().iter().all(|&e| (4..8).contains(&e)));
    assert!(l.owns(7) && !l.owns(3));
}

#[test]
fn kept_window_takes_the_most_cached_then_kept_experts() {
    let set = |n: usize, on: &[usize]| {
        let mut v = vec![false; n];
        on.iter().for_each(|&e| v[e] = true);
        v
    };
    // Cached 2,3,4,7 (kept: 2,7): windows of 3 holding 3 cached: 2..5 only.
    let cached = set(10, &[2, 3, 4, 7]);
    let kept = set(10, &[2, 7]);
    assert_eq!(kept_window(&cached, &kept, 0..10, 3), Some(2));
    // Ties on cached go to the most kept, then the lowest.
    let cached = set(8, &[0, 1, 4, 5]);
    assert_eq!(kept_window(&cached, &set(8, &[5]), 0..8, 2), Some(4));
    assert_eq!(kept_window(&cached, &set(8, &[]), 0..8, 2), Some(0));
    // Inside the owned experts only; none when it does not fit.
    assert_eq!(kept_window(&cached, &set(8, &[]), 3..8, 2), Some(4));
    assert_eq!(kept_window(&cached, &set(8, &[]), 3..8, 6), None);
    assert_eq!(kept_window(&cached, &set(8, &[]), 0..8, 0), None);
}

#[test]
fn miss_runs_counts_the_gaps_between_kept_experts() {
    let kept = [true, false, false, true, false, true, true, false];
    assert_eq!(miss_runs(&kept, 0..8), 3);
    assert_eq!(miss_runs(&kept, 3..8), 2);
    assert_eq!(miss_runs(&[true; 4], 0..4), 0);
    assert_eq!(miss_runs(&[false; 4], 0..4), 1);
}

#[test]
fn repack_fills_the_kept_slots_with_a_window() {
    // 12 experts, 6 slots, the first 4 kept: decode left 1, 5, 8, 10 in them and 4, 6 in
    // the tail, so the prompt would copy 0, 2-4, 6-7, 9, 11: 5 runs.
    let mut l = LayerSlots::new(12, 6);
    l.fill(&[1, 5, 8, 10, 4, 6]);
    let moves = l.repack_kept(4, true);
    // Windows 3..7, 4..8 and 5..9 hold 3 cached experts; 5..9 the most kept ones (5, 8),
    // which stay. 6 moves from tail slot 5 into slot 0, 7 comes from host memory (6 + 7)
    // into slot 3. The prompt then copies 0-4 and 9-11.
    assert_eq!(moves, vec![(0, 5), (3, 13)]);
    assert_eq!(
        l.ready(),
        vec![Some(6), Some(5), Some(8), Some(7), Some(4), None]
    );
    assert_eq!(l.clear_from(4), vec![4], "6 left the tail");
    let table = l.table();
    assert_eq!(&table[5..9], &[1, 0, 3, 2]);
    assert!(
        table
            .iter()
            .enumerate()
            .all(|(e, &id)| (5..9).contains(&e) || id == 6 + e as i32)
    );
    // Already a window: nothing to do.
    assert!(l.repack_kept(4, true).is_empty());
}

#[test]
fn repack_reads_the_tail_from_host_memory_without_its_vram() {
    let mut l = LayerSlots::new(8, 4);
    l.fill(&[0, 3, 5, 1]);
    // Kept 0, 3: misses 1-2 and 4-7. Without the tail's bytes, cached = kept: windows of 2
    // holding the most kept experts tie at one, the lowest (0..2) wins; 1 comes from host
    // memory (4 + 1) into slot 1.
    let moves = l.repack_kept(2, false);
    assert_eq!(moves, vec![(1, 5)]);
    assert_eq!(l.ready()[..2], [Some(0), Some(1)]);
    // Expert 1's old tail slot no longer holds it, so a clear does not hand it back.
    assert_eq!(l.clear_from(2), vec![5]);
}

#[test]
fn repack_keeps_contiguous_or_filling_slots() {
    // Kept slots already a window: misses in one or two runs, no gain.
    let mut l = LayerSlots::new(8, 4);
    l.fill(&[2, 3, 4, 7]);
    assert!(l.repack_kept(3, true).is_empty());
    // A kept slot still filling: left alone.
    let mut l = LayerSlots::new(8, 4);
    l.fill(&[0, 3]);
    l.clear_from(1);
    assert_eq!(l.refill(&[6]), vec![(1, 6)]);
    assert!(l.repack_kept(2, true).is_empty());
    // Owned experts only (expert parallelism): the window stays inside them.
    let mut l = LayerSlots::new(8, 2);
    l.set_owned(4..8);
    l.fill(&[4, 6]);
    // Kept 4, 6 (misses 5, 7): the window 4..6, 5 from host memory (2 + 5).
    assert_eq!(l.repack_kept(2, true), vec![(1, 2 + 5)]);
    assert_eq!(l.ready(), vec![Some(4), Some(5)]);
}

#[test]
fn per_device_budget_overrides_only_the_devices_it_names() {
    let rule = SwapRule::default();
    let per = [48, 0];
    assert_eq!(rule.on_device(&per, 0).budget, 48);
    assert_eq!(rule.on_device(&per, 1).budget, 0);
    // Devices past the list keep the rule's budget; margin and ratio never change.
    assert_eq!(rule.on_device(&per, 2), rule);
    assert_eq!(rule.on_device(&[], 0), rule);
    assert_eq!(rule.on_device(&per, 0).margin, rule.margin);
}

#[test]
fn layers_left_coldest_takes_the_candidates_of_least_heat_until_the_need() {
    let lent = [Some(100), None, Some(100), Some(50), None, Some(60)];
    let heat = [1.0, 0.0, 5.0, 9.0, 0.0, 7.0];
    // The coldest candidate covers it: layer 0, though it is the first.
    assert_eq!(
        layers_left_coldest(&lent, &heat, 100),
        vec![true, false, false, false, false, false]
    );
    // Then the next coldest (layer 2); never a layer that is not a candidate.
    assert_eq!(
        layers_left_coldest(&lent, &heat, 150),
        vec![true, false, true, false, false, false]
    );
    // Equal heat: the later layer first, as `layers_left_for`.
    assert_eq!(
        layers_left_coldest(&lent, &[0.0; 6], 100),
        layers_left_for(&lent, 100)
    );
    assert_eq!(layers_left_coldest(&lent, &heat, 0), vec![false; 6]);
    assert_eq!(
        layers_left_coldest(&lent, &heat, 10_000),
        vec![true, false, true, true, false, true]
    );
}

#[test]
fn layers_left_for_takes_the_last_candidates_until_the_need() {
    let lent = [Some(100), None, Some(100), Some(50), None, Some(60)];
    // From the last one down: 60, then 50 (110 >= 100); the rest are refilled.
    assert_eq!(
        layers_left_for(&lent, 100),
        vec![false, false, false, true, false, true]
    );
    // Exactly reached by the last candidate: no more.
    assert_eq!(
        layers_left_for(&lent, 60),
        vec![false, false, false, false, false, true]
    );
    // Nothing to leave.
    assert_eq!(layers_left_for(&lent, 0), vec![false; 6]);
    // Never enough: every candidate, none of the others.
    assert_eq!(
        layers_left_for(&lent, 10_000),
        vec![true, false, true, true, false, true]
    );
    assert!(layers_left_for(&[], 5).is_empty());
}

/// The bytes of a slot plan.
fn plan_bytes(plan: &[Vec<u32>], slot_bytes: &[u64]) -> u64 {
    plan.iter()
        .zip(slot_bytes)
        .map(|(e, &b)| e.len() as u64 * b)
        .sum()
}

#[test]
fn shrunk_budget_takes_the_drift_or_a_doubling_step() {
    const MIB: u64 = 1 << 20;
    // Others took 300 MiB: that and a granule go, more than one slot.
    assert_eq!(shrunk_budget(1000 * MIB, 300 * MIB, 6 * MIB, 0), 698 * MIB);
    // Nothing measured: one slot of the failed layer, then 2, 4, ... of them.
    assert_eq!(shrunk_budget(1000 * MIB, 0, 6 * MIB, 0), 994 * MIB);
    assert_eq!(shrunk_budget(1000 * MIB, 0, 6 * MIB, 1), 988 * MIB);
    assert_eq!(shrunk_budget(1000 * MIB, 0, 6 * MIB, 3), 952 * MIB);
    // The step outgrows a small drift.
    assert_eq!(shrunk_budget(1000 * MIB, MIB, 6 * MIB, 2), 976 * MIB);
    // At least a granule, even for tiny slots.
    assert_eq!(shrunk_budget(1000 * MIB, 0, 1024, 0), 998 * MIB);
    // Never below 0, never overflowing, however many failures.
    assert_eq!(shrunk_budget(1000 * MIB, 0, 6 * MIB, 20), 0);
    assert_eq!(
        shrunk_budget(1000 * MIB, u64::MAX, u64::MAX, MAX_FIT_TRIES),
        0
    );
    assert_eq!(shrunk_budget(0, 0, 6 * MIB, 0), 0);
}

#[test]
fn a_replan_after_drift_keeps_the_room_beside_the_cache() {
    const MIB: u64 = 1 << 20;
    // 4 layers of 64 experts, 6 MiB a slot; the plan saw 1100 MiB free and left 100 MiB beside
    // a 1000 MiB cache. Another process then took 300 MiB.
    let slot_bytes = [6 * MIB; 4];
    let margin = 100 * MIB;
    let free_now = 800 * MIB;
    let before = allocate_slots(&[], &slot_bytes, 64, 1000 * MIB);
    assert!(plan_bytes(&before, &slot_bytes) > free_now - margin);
    let b = shrunk_budget(1000 * MIB, 300 * MIB, 6 * MIB, 0);
    let after = allocate_slots(&[], &slot_bytes, 64, b);
    let bytes = plan_bytes(&after, &slot_bytes);
    assert!(bytes <= free_now - margin, "{} MiB", bytes / MIB);
    // Within a slot per layer of what fits, every layer keeps a share, and each layer's
    // experts are the first ones of the bigger plan.
    assert!(bytes + 4 * 6 * MIB > free_now - margin);
    for (a, b) in after.iter().zip(&before) {
        assert!(!a.is_empty());
        assert_eq!(&b[..a.len()], &a[..]);
    }
}

#[test]
fn repeated_failures_without_a_drift_shrink_in_a_few_tries() {
    const MIB: u64 = 1 << 20;
    // A hidden limit (a GB10's shared memory: the free figure says it fits): 900 MiB of slots
    // fit, the plan asks 1000 MiB.
    let slot_bytes = [6 * MIB; 4];
    let limit = 900 * MIB;
    let ranked: Vec<(usize, u32)> = (0..64).flat_map(|e| (0..4).map(move |l| (l, e))).collect();
    let mut b = 1000 * MIB;
    let mut plan = allocate_slots(&ranked, &slot_bytes, 64, b);
    let mut failures = 0;
    while plan_bytes(&plan, &slot_bytes) > limit {
        b = shrunk_budget(b, 0, 6 * MIB, failures);
        failures += 1;
        plan = allocate_slots(&ranked, &slot_bytes, 64, b);
    }
    let bytes = plan_bytes(&plan, &slot_bytes);
    assert!(failures <= 6, "{failures} tries");
    // The cut is at most twice the shortfall, plus a step of slack.
    let short = 1000 * MIB - limit;
    assert!(
        1000 * MIB - bytes <= 2 * short + 6 * MIB,
        "{} MiB",
        bytes / MIB
    );
}

#[test]
fn saved_heat_picks_the_contents_and_keeps_the_layer_sizes() {
    let alloc = vec![vec![0, 1, 2], vec![0, 1], vec![], vec![0, 1, 2, 3]];
    // Layer 1's ranking is longer than its slots, layer 3's shorter, layer 0 has none.
    let order = vec![vec![], vec![7, 0, 5], vec![4], vec![9, 1]];
    assert_eq!(
        hottest_experts(&alloc, &order),
        vec![vec![0, 1, 2], vec![7, 0], vec![], vec![9, 1, 0, 2]]
    );
    // A ranking of fewer layers leaves the others as they were.
    assert_eq!(hottest_experts(&alloc, &order[..1]), alloc);
}

#[test]
fn a_limit_empties_the_last_slots_and_keeps_them_empty() {
    let mut l = LayerSlots::new(8, 4);
    l.fill(&[0, 1, 2, 3]);
    // Slots 2 and 3 lose their memory: their experts read from host memory.
    assert_eq!(l.set_limit(2), vec![2, 3]);
    assert_eq!(l.limit(), 2);
    assert_eq!(l.table(), vec![0, 1, 6, 7, 8, 9, 10, 11]);
    // Nothing takes them: neither a refill nor a swap, however hot the newcomer.
    assert!(l.refill(&[5, 6]).is_empty());
    l.record(&[4 + 7, 4 + 7, 4 + 7, 4 + 7, 4 + 7, 4 + 7, 4 + 7, 4 + 7]);
    let swaps = plan_swaps(&mut [Some(&mut l)], SwapRule::default());
    assert!(swaps.iter().all(|s| s.slot < 2), "{swaps:?}");
    // A lend's clear and refill stay inside the limit too.
    l.clear();
    assert_eq!(l.refill(&[7, 6, 5]).len(), 2);
    // The memory is back: the slots fill again.
    assert!(l.set_limit(4).is_empty());
    assert_eq!(l.refill(&[5, 4]), vec![(2, 5), (3, 4)]);
    // Never past the slots there are.
    l.set_limit(99);
    assert_eq!(l.limit(), 4);
}

#[test]
fn even_cuts_take_a_slot_a_layer_from_the_last_one_down() {
    const G: usize = 2 << 20;
    // 3 MiB a slot in three kinds of 1 MiB. The first empty slot keeps its memory: three
    // slots free one granule a kind.
    let b = [1 << 20; 3];
    assert_eq!(cut_spare(0), 0);
    assert_eq!(cut_spare(3), 2);
    assert_eq!(cut_bytes(&b, 1, G), 0);
    assert_eq!(cut_bytes(&b, 2, G), 0);
    assert_eq!(cut_bytes(&b, 3, G), 3 * G as u64);
    assert_eq!(cut_bytes(&b, 5, G), 6 * G as u64);
    let layers = [Some((b, 4)), None, Some((b, 4)), Some((b, 1))];
    assert_eq!(even_cuts(&layers, G, 0), Some(vec![0, 0, 0, 0]));
    // Two slots each free nothing yet; the third round stops at the first layer that does
    // (layer 3 may give one slot only).
    assert_eq!(even_cuts(&layers, G, 1), Some(vec![2, 0, 3, 1]));
    assert_eq!(
        even_cuts(&layers, G, 3 * G as u64 + 1),
        Some(vec![3, 0, 3, 1])
    );
    // All they may give: 4 + 4 slots free 6 granules (a fourth slot frees no whole granule
    // more), layer 3's one slot none.
    assert_eq!(even_cuts(&layers, G, 6 * G as u64), Some(vec![3, 0, 3, 1]));
    assert_eq!(even_cuts(&layers, G, 6 * G as u64 + 1), None);
    // No two layers differ by more than a slot.
    let many = vec![Some((b, 50)); 48];
    let cuts = even_cuts(&many, G, 960 << 20).unwrap();
    let (lo, hi) = (cuts.iter().min().unwrap(), cuts.iter().max().unwrap());
    assert!(hi - lo <= 1, "{cuts:?}");
    let freed: u64 = cuts.iter().map(|&c| cut_bytes(&b, c, G)).sum();
    assert!(freed >= 960 << 20 && freed < (960 << 20) + 3 * G as u64);
}
