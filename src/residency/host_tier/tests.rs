use super::*;

const GIB: u64 = 1 << 30;
/// Qwen3.8-Flash-Next UD-Q4_K_XL: one expert's up, gate and down.
const EXPERT: u64 = 921_600 + 921_600 + 1_228_800;
/// 48 layers of that one size.
const LAYERS: [[u64; 3]; 48] = [[921_600, 921_600, 1_228_800]; 48];

#[test]
fn layers_of_one_size_stage_back_to_back() {
    let (second, bytes) = stage_layout(&LAYERS, 135);
    assert!(second.iter().all(|&s| s == 135));
    assert_eq!(
        bytes.iter().sum::<u64>(),
        EXPERT * (2 * 135 + GAP as u64 + 1)
    );
}

#[test]
fn layers_of_mixed_sizes_share_one_staging_buffer_a_kind() {
    // The real file: 42 layers as above, one with Q5_K up and gate, five with Q8_0 down.
    let mut layers = LAYERS.to_vec();
    layers[0] = [1_126_400, 1_126_400, 1_228_800];
    for l in &mut layers[43..] {
        l[2] = 1_740_800;
    }
    let places = 256;
    let (second, bytes) = stage_layout(&layers, places);
    assert!(mixed(&layers) && !mixed(&LAYERS));
    // No first set (it starts within the gap, or a place later) nor the place behind it
    // reaches any layer's second set, in any kind.
    for k in 0..3 {
        for (l, &s) in layers.iter().zip(&second) {
            assert!(s >= places + 2);
            // Layers of the same size line up (same start in the gap); the others' first
            // sets end before this one's second starts, wherever in the gap each starts.
            let others = layers.iter().filter(|o| o[k] != l[k]);
            for o in others {
                assert!(s as u64 * l[k] >= (GAP + places + 2) as u64 * o[k]);
            }
            assert!((GAP + s + places + 1) as u64 * l[k] <= bytes[k]);
        }
    }
    // A third less than a buffer per type (six of them, two sets and the gap each).
    let per_type: u64 = 921_600 * 2 + 1_126_400 * 2 + 1_228_800 + 1_740_800;
    let before = per_type * (2 * places + GAP + 1) as u64;
    assert!(bytes.iter().sum::<u64>() * 3 < before * 2 + before / 10);
    // One index for a layer's three kinds: the kind that needs the most room decides.
    assert_eq!(second[47], 321);
    assert_eq!(second[1], 372);
}

fn mem(total: u64, available: u64) -> MemInfo {
    MemInfo {
        total: total * GIB,
        available: available * GIB,
    }
}

fn table(in_ram: bool) -> Option<Table> {
    Some(Table {
        bytes: 27 * GIB,
        in_ram,
        direct: false,
    })
}

#[test]
fn everything_fits_changes_nothing() {
    // 71.7 GiB of experts, a 4 GiB row cache and a 12.5 GiB margin in 120 GiB.
    assert_eq!(
        size(mem(125, 120), &LAYERS, 512, table(false), 0),
        Sizing::Whole
    );
    // No table.
    assert_eq!(size(mem(125, 90), &LAYERS, 512, None, 0), Sizing::Whole);
}

#[test]
fn a_table_asked_into_ram_counts_whole() {
    // 71.7 + 27 + 12.5 > 105.
    let Sizing::Bounded(p) = size(mem(125, 105), &LAYERS, 512, table(true), 0) else {
        panic!("bounded");
    };
    assert_eq!(p.table, 27 * GIB);
    assert_eq!(p.margin, 125 * GIB / 10);
}

#[test]
fn a_64_gib_machine_gets_most_experts() {
    let Sizing::Bounded(p) = size(mem(64, 58), &LAYERS, 512, table(false), 0) else {
        panic!("bounded");
    };
    // 58 - 8 (margin) - 4 (row cache) = 46 GiB of 140.6 MiB slots and the staging they leave
    // to do: two sets of 135 places.
    assert_eq!(p.margin, 8 * GIB);
    assert_eq!(p.table, 4 * GIB);
    assert_eq!((p.places, p.sets), (135, 2));
    assert_eq!(p.stage_bytes, EXPERT * (2 * 135 + GAP as u64 + 1));
    assert_eq!(p.slots, 329);
    assert_eq!(p.slot_bytes, 329 * 48 * EXPERT);
    assert!(p.slot_bytes + p.stage_bytes + p.table + p.margin <= 58 * GIB);
    // One slot more would not fit.
    assert!(
        p.slot_bytes + 48 * EXPERT + p.stage_bytes - 2 * EXPERT + p.table + p.margin > 58 * GIB
    );
}

#[test]
fn the_staging_sets_hold_what_the_ram_slots_cannot() {
    // Many RAM slots: a set holds a big step's layer less what its VRAM slots hold.
    assert_eq!(stage_sets(512, 329), (135, 2));
    // Few: half the experts each, a place per expert together.
    assert_eq!(stage_sets(512, 98), (256, 2));
    assert_eq!(stage_sets(512, 32), (256, 2));
    // Nearly all experts in RAM: the two sets still hold a step's worst case.
    let (places, sets) = stage_sets(512, 500);
    assert_eq!((places, sets), (14, 2));
    assert!(places * sets >= 512 - 500 + 16);
}

#[test]
fn a_steps_staged_experts_take_their_set_first() {
    // Layer 3 of two sets of 4: its set is the second one; 1 place holds a read ahead.
    assert_eq!(assign_places(3, 1, 2, 4, 1), Some((vec![5, 6, 7], false)));
    // One more spills into the other set, from its first place.
    assert_eq!(
        assign_places(5, 1, 2, 4, 1),
        Some((vec![5, 6, 7, 0, 1], true))
    );
    assert_eq!(assign_places(7, 1, 2, 4, 1).map(|a| a.1), Some(true));
    // More experts than places.
    assert_eq!(assign_places(8, 1, 2, 4, 1), None);
    assert_eq!(assign_places(0, 0, 2, 4, 4), Some((vec![], false)));
}

#[test]
fn a_read_ahead_takes_what_the_last_big_step_read() {
    let mut c = SlotCache::new(8, 2);
    plan_step(&mut c, 8, &[5]);
    let mut in_vram = vec![false; 8];
    in_vram[2] = true;
    // Expert 5 is in a RAM slot now and 2 in VRAM: neither is read.
    assert_eq!(plan_ahead(&c, &in_vram, &[1, 2, 5, 6, 7], 8), vec![1, 6, 7]);
    // No more than a set's places.
    assert_eq!(plan_ahead(&c, &in_vram, &[1, 2, 5, 6, 7], 2), vec![1, 6]);
    assert!(plan_ahead(&c, &in_vram, &[], 8).is_empty());
}

#[test]
fn a_32_gib_machine_gets_fewer() {
    let Sizing::Bounded(p) = size(mem(32, 27), &LAYERS, 512, table(false), 0) else {
        panic!("bounded");
    };
    // A sixth of the machine is kept free, not 8 GiB: 19 slots more.
    assert_eq!(p.margin, 32 * GIB / 6);
    assert_eq!(p.slots, 117);
    assert_eq!((p.places, p.sets), (256, 2));
    // The KV spill comes off the experts.
    let Sizing::Bounded(q) = size(mem(32, 27), &LAYERS, 512, table(false), 2 * GIB) else {
        panic!("bounded");
    };
    assert_eq!(q.kv_spill, 2 * GIB);
    assert_eq!(q.slots, 103);
}

#[test]
fn a_small_machine_keeps_a_sixth_free_and_never_under_4_gib() {
    assert_eq!(margin(16 * GIB), 4 * GIB);
    assert_eq!(margin(24 * GIB), 4 * GIB);
    assert_eq!(margin(32 * GIB), 32 * GIB / 6);
    // From 48 GiB on as before: 8 GiB, a tenth past 80 GiB.
    assert_eq!(margin(48 * GIB), 8 * GIB);
    assert_eq!(margin(64 * GIB), 8 * GIB);
    assert_eq!(margin(125 * GIB), 125 * GIB / 10);
}

#[test]
fn too_little_memory_is_told() {
    assert!(matches!(
        size(mem(16, 13), &LAYERS, 512, table(false), 0),
        Sizing::TooSmall(_)
    ));
}

#[test]
fn ram_slots_give_up_the_coldest_expert_once_heat_is_known() {
    let mut c = SlotCache::new(8, 3);
    plan_step(&mut c, 8, &[5, 2, 6]);
    plan_step(&mut c, 8, &[2, 6]);
    // Least recently used: 5 goes.
    let mut lru = c.clone();
    assert_eq!(plan_step(&mut lru, 8, &[1]).loads, vec![(1, Some(0))]);
    // By heat 5 is the most routed of the three and stays; 6 is the coldest.
    c.set_heat(&[0.0, 0.0, 2.0, 0.0, 0.0, 9.0, 1.0, 0.0]);
    assert_eq!(plan_step(&mut c, 8, &[1]).loads, vec![(1, Some(2))]);
    assert_eq!(c.slot(6), None);
    // A step's own experts never go, whatever their heat; equal heat: least recently used.
    c.set_heat(&[0.0; 8]);
    let t = plan_step(&mut c, 8, &[5, 3]);
    assert_eq!(t.loads, vec![(3, Some(1))]);
    assert!(t.cached[5]);
}

#[test]
fn a_step_takes_cached_experts_from_their_slots() {
    let mut c = SlotCache::new(8, 3);
    let t = plan_step(&mut c, 8, &[5, 2]);
    assert_eq!(t.loads, vec![(5, Some(0)), (2, Some(1))]);
    assert_eq!((t.index[5], t.index[2]), (0, 1));
    assert!(!t.cached[5] && !t.cached[2]);
    // The next step finds both and reads one more into the free slot.
    let t = plan_step(&mut c, 8, &[2, 7, 5]);
    assert_eq!(t.loads, vec![(7, Some(2))]);
    assert_eq!((t.index[2], t.index[7], t.index[5]), (1, 2, 0));
    assert!(t.cached[2] && t.cached[5] && !t.cached[7]);
    assert_eq!(t.index[0], usize::MAX);
}

#[test]
fn a_step_evicts_the_least_recently_used() {
    let mut c = SlotCache::new(8, 2);
    plan_step(&mut c, 8, &[0]);
    plan_step(&mut c, 8, &[1]);
    plan_step(&mut c, 8, &[0]);
    // Expert 1's slot is the older one.
    let t = plan_step(&mut c, 8, &[3]);
    assert_eq!(t.loads, vec![(3, Some(1))]);
    assert_eq!(c.slot(1), None);
    assert_eq!(c.slot(0), Some(0));
}

#[test]
fn a_big_step_stages_what_the_slots_cannot_hold() {
    let mut c = SlotCache::new(8, 2);
    plan_step(&mut c, 8, &[6]);
    // Five experts for two slots: the cached one stays, one more gets the other slot, three
    // go to their staging places (past the slots and the gap, by expert).
    let t = plan_step(&mut c, 8, &[1, 6, 3, 4, 7]);
    assert_eq!(t.loads, vec![(1, Some(1)), (3, None), (4, None), (7, None)]);
    assert_eq!(t.index[6], 0);
    assert_eq!(t.index[1], 1);
    assert_eq!(t.index[3], 2 + GAP + 3);
    assert_eq!(t.index[7], 2 + GAP + 7);
    // Staged experts are not cached.
    assert_eq!(c.slot(3), None);
    assert_eq!(c.filled(), 2);
}

#[test]
fn a_vram_swap_trades_places_with_a_ram_slot() {
    let mut c = SlotCache::new(8, 2);
    plan_step(&mut c, 8, &[3]);
    plan_step(&mut c, 8, &[4]);
    // VRAM takes expert 4 (RAM slot 1) and drops expert 7: its bytes go to the least
    // recently used RAM slot, expert 3's.
    let f = plan_swap(&mut c, 5, 4, Some(7));
    assert_eq!(
        f,
        Flight {
            layer: 5,
            from: Some(1),
            back: Some((7, 0))
        }
    );
    // Until the copies are done expert 4 still reads from its RAM slot and 7 is in neither.
    assert_eq!((c.slot(4), c.slot(7), c.slot(3)), (Some(1), None, None));
    land_swap(&mut c, &f);
    assert_eq!((c.slot(4), c.slot(7)), (None, Some(0)));
    assert_eq!(c.filled(), 1);
    // The next swap's dropped expert takes the slot the last one emptied: nothing is evicted.
    let f = plan_swap(&mut c, 5, 7, Some(4));
    assert_eq!((f.from, f.back), (Some(0), Some((4, 1))));
    land_swap(&mut c, &f);
    assert_eq!((c.slot(7), c.slot(4)), (None, Some(1)));
}

#[test]
fn a_step_leaves_the_slots_of_a_swap_under_way_alone() {
    let mut c = SlotCache::new(8, 3);
    plan_step(&mut c, 8, &[0]);
    plan_step(&mut c, 8, &[1]);
    plan_step(&mut c, 8, &[2]);
    let f = plan_swap(&mut c, 0, 1, Some(5));
    assert_eq!((f.from, f.back), (Some(1), Some((5, 0))));
    // A step that wants the dropped expert before its bytes are in RAM reads it from the
    // checkpoint into the one slot no swap holds; a second miss is staged.
    let t = plan_step(&mut c, 8, &[5, 6]);
    assert_eq!(t.loads, vec![(5, Some(2)), (6, None)]);
    land_swap(&mut c, &f);
    // The reserved slot is given up: the expert has one already.
    assert_eq!((c.slot(1), c.slot(5)), (None, Some(2)));
    assert_eq!(c.filled(), 1);
}

#[test]
fn a_swap_writes_nothing_back_when_ram_has_the_expert() {
    let mut c = SlotCache::new(8, 2);
    plan_step(&mut c, 8, &[6]);
    // The new expert is in no RAM slot (read from the checkpoint), the old one in one.
    let f = plan_swap(&mut c, 0, 2, Some(6));
    assert_eq!((f.from, f.back), (None, None));
    land_swap(&mut c, &f);
    assert_eq!(c.slot(6), Some(0));
    // A first fill drops nothing.
    let f = plan_swap(&mut c, 0, 6, None);
    assert_eq!((f.from, f.back), (Some(0), None));
    land_swap(&mut c, &f);
    assert_eq!(c.filled(), 0);
}

#[test]
fn a_staged_pass_reads_a_layers_missing_experts_in_batches_of_a_set() {
    use passes::{Held, PassSrc, plan_pass};
    // 10 RAM slots, 2 sets of 2 places. Layer 0: one expert in RAM slot 7, three to read;
    // layer 1: all in RAM; layer 2: one to read.
    let layers = vec![
        vec![Held::Stage, Held::Ram(7), Held::Stage, Held::Stage],
        vec![Held::Ram(1), Held::Ram(2)],
        vec![Held::Stage],
    ];
    let (srcs, batches) = plan_pass(&layers, 10, 2, 2).unwrap();
    let at = |place: usize| 10 + GAP + place;
    let read = |batch: usize, place: usize| PassSrc::Read {
        batch,
        at: at(place),
    };
    assert_eq!(
        srcs[0],
        vec![read(0, 0), PassSrc::Ram(7), read(0, 1), read(1, 2)]
    );
    assert_eq!(srcs[1], vec![PassSrc::Ram(1), PassSrc::Ram(2)]);
    // A batch is one layer's: layer 2 starts a new one, in the first set again.
    assert_eq!(srcs[2], vec![read(2, 0)]);
    let p = To::Place;
    assert_eq!(
        batches,
        vec![
            vec![(0, 0, p(0)), (0, 2, p(1))],
            vec![(0, 3, p(2))],
            vec![(2, 0, p(0))]
        ]
    );
    // Batch `b` and batch `b + sets` share places: the reader waits for the first's release.
    assert_eq!(passes::wait_arg(3) + 1, passes::release_arg(3));
}

#[test]
fn a_staged_pass_reads_into_the_ram_slots_it_was_given() {
    use passes::{Held, PassSrc, plan_pass};
    // 10 RAM slots, 2 sets of 2 places: slots 4 and 5 are free for two of the three experts
    // the checkpoint holds. They ride in the batch and take none of its places.
    let layers = vec![vec![Held::Load(4), Held::Stage, Held::Load(5)]];
    let (srcs, batches) = plan_pass(&layers, 10, 2, 2).unwrap();
    let read = |batch: usize, at: usize| PassSrc::Read { batch, at };
    assert_eq!(srcs[0], vec![read(0, 4), read(0, 10 + GAP), read(1, 5)]);
    assert_eq!(
        batches,
        vec![
            vec![(0, 0, To::Slot(4)), (0, 1, To::Place(0))],
            vec![(0, 2, To::Slot(5))]
        ]
    );
}

#[test]
fn a_staged_pass_without_places_reads_nothing() {
    use passes::{Held, plan_pass};
    assert!(plan_pass(&[vec![Held::Ram(0)]], 4, 0, 0).is_ok());
    assert!(plan_pass(&[vec![Held::Stage]], 4, 0, 0).is_err());
}

#[test]
fn the_debug_memory_is_one_value_or_total_and_available() {
    assert_eq!(debug_mem("58"), Some((58 * GIB, 58 * GIB)));
    assert_eq!(debug_mem("32:27"), Some((32 * GIB, 27 * GIB)));
    assert_eq!(
        debug_mem(" 32 : 27.5 "),
        Some((32 * GIB, 27 * GIB + GIB / 2))
    );
    for bad in ["", "0", "-4", "x", "32:", ":27", "27:32", "32:0", "1:2:3"] {
        assert_eq!(debug_mem(bad), None, "{bad:?}");
    }
    // A 32 GiB machine with 27 available keeps its own margin: fewer bytes for the slots
    // than a 27 GiB machine with all of it available.
    let room = |(total, available): (u64, u64)| available - margin(total);
    assert!(room(debug_mem("32:27").unwrap()) < room(debug_mem("27").unwrap()));
}

#[test]
fn prefill_takes_the_empty_slots_in_order() {
    let mut cache = SlotCache::new(8, 3);
    cache.plan(&[5]);
    // 5 is cached already; 6, 1 and 7 are wanted, two slots are empty.
    assert_eq!(
        plan_prefill(&mut cache, &[6, 5, 1, 7]),
        vec![(6, 1), (1, 2)]
    );
    assert_eq!(cache.filled(), 3);
    assert_eq!(plan_prefill(&mut cache, &[7]), vec![]);
}

#[test]
fn sources_are_named_like_the_models_tensors() {
    assert_eq!(
        skipped_tensors(1),
        vec![
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight"
        ]
    );
}
