use super::*;

#[test]
fn counts_hits_and_copies_of_both_predictors() {
    let mut s = PrefetchStats::default();
    // GPU 1 chose 3 (hit), 7 (miss), 9 (miss), 12 (miss). The router named 7, 9 and 20 (20
    // uncached, not chosen); the previous pass chose 3, 9 and 30 (30 uncached).
    let chosen = [(3, true), (7, false), (9, false), (12, false)];
    let uncached = |e: u32| e != 3;
    s.add(
        &EntryPredicted {
            gpu: 1,
            chosen: &chosen,
            next: &[7, 9, 20],
            prev: &[3, 9, 30],
        },
        uncached,
    );
    s.end_pass();
    assert_eq!(s.steps, 1);
    assert_eq!(s.gpus.len(), 2);
    assert_eq!(s.gpus[0], GpuCounts::default());
    let g = &s.gpus[1];
    assert_eq!((g.chosen, g.misses), (4, 3));
    assert_eq!((g.next_hits, g.prev_hits, g.either_hits), (2, 1, 2));
    assert_eq!(g.next_chosen, 2);
    // Copies: the router's 7, 9, 20; the previous pass's 9, 30 (3 is cached).
    assert_eq!((g.next_copies, g.prev_copies), (3, 2));
    let t = s.take();
    assert_eq!(t.steps, 1);
    assert_eq!(s, PrefetchStats::default());
}

#[test]
fn either_counts_a_miss_once() {
    let mut s = PrefetchStats::default();
    let chosen = [(5, false), (6, false)];
    s.add(
        &EntryPredicted {
            gpu: 0,
            chosen: &chosen,
            next: &[5],
            prev: &[5, 6],
        },
        |_| true,
    );
    let g = &s.gpus[0];
    assert_eq!((g.next_hits, g.prev_hits, g.either_hits), (1, 2, 2));
}

#[test]
fn owned_distinct_drops_unwritten_and_foreign_ids() {
    let ids = [9, -1, 3, 9, 80, 3, 1];
    assert_eq!(owned_distinct(&ids, |e| e < 72), vec![1, 3, 9]);
    assert!(owned_distinct(&[-1, -1], |_| true).is_empty());
}

#[test]
fn ring_halves_are_aligned_and_hold_cap_slots() {
    let l = RingLayout::new(4, 7 << 20 | 3, 4096, 288);
    for at in [
        l.guess, l.raw, l.real, l.x, l.logits, l.done, l.slots, l.slot, l.half,
    ] {
        assert_eq!(at % 256, 0, "{l:?}");
    }
    assert_eq!(l.copy_table, 0);
    assert!(l.logits - l.x >= MAX_TOKENS * 4096 * 4);
    assert!(l.done - l.logits >= MAX_TOKENS * 288 * 4);
    assert!(l.slot >= 7 << 20 | 3);
    assert_eq!(l.half, l.slots + 4 * l.slot);
    assert_eq!(l.bytes(), 2 * l.half);
    assert_eq!((l.half_of(5), l.half_of(6)), (l.half, 0));
}

#[test]
fn predicts_small_steps_only() {
    assert!(predicts(1, 8) && predicts(3, 8) && predicts(16, 8));
    assert!(!predicts(0, 8) && !predicts(17, 8) && !predicts(16, 64));
}

#[test]
fn staged_in_pair_ranks_by_first_guess_and_caps_each_side() {
    // Pair experts 8..16, the second GPU's from 12; 9 and 13 cached.
    let mut cached = vec![0; 20];
    cached[9] = 1;
    cached[13] = 1;
    let raw = [9, 10, 3, 14, 10, 11, 15, 8, 13, 12, -1];
    let out = staged_in_pair(&raw, &cached, (8, 12, 16), 2);
    let staged: Vec<usize> = (0..20).filter(|&e| out[e] != 0 && cached[e] == 0).collect();
    // First GPU: 10, 11 (8 past its cap); second: 14, 15 (12 past its cap); 3 outside the pair.
    assert_eq!(staged, [10, 11, 14, 15]);
    assert_eq!(out[9], 1);
    assert_eq!(out[13], 1);
}
