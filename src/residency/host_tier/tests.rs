use super::*;

const GIB: u64 = 1 << 30;
/// Qwen3.8-Flash-Next UD-Q4_K_XL: one expert's up, gate and down.
const EXPERT: u64 = 921_600 + 921_600 + 1_228_800;

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
        size(mem(125, 120), 48 * EXPERT, EXPERT, 512, table(false), 0),
        Sizing::Whole
    );
    // No table.
    assert_eq!(
        size(mem(125, 90), 48 * EXPERT, EXPERT, 512, None, 0),
        Sizing::Whole
    );
}

#[test]
fn a_table_asked_into_ram_counts_whole() {
    // 71.7 + 27 + 12.5 > 105.
    let Sizing::Bounded(p) = size(mem(125, 105), 48 * EXPERT, EXPERT, 512, table(true), 0) else {
        panic!("bounded");
    };
    assert_eq!(p.table, 27 * GIB);
    assert_eq!(p.margin, 125 * GIB / 10);
}

#[test]
fn a_64_gib_machine_gets_most_experts() {
    let Sizing::Bounded(p) = size(mem(64, 58), 48 * EXPERT, EXPERT, 512, table(false), 0) else {
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
    let Sizing::Bounded(p) = size(mem(32, 27), 48 * EXPERT, EXPERT, 512, table(false), 0) else {
        panic!("bounded");
    };
    assert_eq!(p.slots, 98);
    assert_eq!((p.places, p.sets), (256, 2));
    // The KV spill comes off the experts.
    let Sizing::Bounded(q) = size(mem(32, 27), 48 * EXPERT, EXPERT, 512, table(false), 2 * GIB)
    else {
        panic!("bounded");
    };
    assert_eq!(q.kv_spill, 2 * GIB);
    assert!(q.slots < p.slots);
}

#[test]
fn too_little_memory_is_told() {
    assert!(matches!(
        size(mem(16, 14), 48 * EXPERT, EXPERT, 512, table(false), 0),
        Sizing::TooSmall(_)
    ));
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
