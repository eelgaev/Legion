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
    // 58 - 8 (margin) - 4 (row cache) - 1.5 (staging) = 44.5 GiB of 140.6 MiB slots.
    assert_eq!(p.margin, 8 * GIB);
    assert_eq!(p.table, 4 * GIB);
    assert_eq!(p.stage_bytes, EXPERT * (512 + GAP as u64 + 1));
    assert_eq!(p.slots, 324);
    assert_eq!(p.slot_bytes, 324 * 48 * EXPERT);
    assert!(p.slot_bytes + p.stage_bytes + p.table + p.margin <= 58 * GIB);
}

#[test]
fn a_32_gib_machine_gets_fewer() {
    let Sizing::Bounded(p) = size(mem(32, 27), 48 * EXPERT, EXPERT, 512, table(false), 0) else {
        panic!("bounded");
    };
    assert_eq!(p.slots, 98);
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
