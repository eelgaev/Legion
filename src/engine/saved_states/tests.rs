use super::*;

fn store(s: &mut SavedStates, prefix: &[u32]) -> usize {
    s.store(prefix, Kind::Anchor, None).unwrap().slot
}

fn found(
    s: &mut SavedStates,
    tokens: &[u32],
    matched: usize,
    block: usize,
) -> Option<(usize, usize)> {
    s.find(tokens, matched, block).map(|f| (f.slot, f.len))
}

#[test]
fn finds_the_longest_saved_prefix_within_the_limit() {
    let mut s = SavedStates::new(4);
    let a = store(&mut s, &[1, 2]);
    let b = store(&mut s, &[1, 2, 3, 4]);
    store(&mut s, &[9, 9, 9]);
    assert_eq!(found(&mut s, &[1, 2, 3, 4, 5], 10, 1), Some((b, 4)));
    // The KV cache matched only 3 tokens.
    assert_eq!(found(&mut s, &[1, 2, 3, 4, 5], 3, 1), Some((a, 2)));
    assert_eq!(found(&mut s, &[1, 3], 10, 1), None);
    assert_eq!(found(&mut s, &[1, 2, 3, 4], 1, 1), None);
    // At least one token is left to compute.
    assert_eq!(found(&mut s, &[1, 2, 3, 4], 10, 1), Some((a, 2)));
}

#[test]
fn store_skips_known_prefixes_and_recycles_the_least_recently_used() {
    let mut s = SavedStates::new(2);
    let a = store(&mut s, &[1]);
    let b = store(&mut s, &[2]);
    assert_ne!(a, b);
    assert_eq!(s.store(&[1], Kind::Anchor, None), None, "already saved");
    // [1] was used more recently than [2]: [2]'s slot goes.
    s.find(&[1, 5], 10, 1);
    assert_eq!(store(&mut s, &[3]), b);
    assert!(s.contains(&[1]) && s.contains(&[3]) && !s.contains(&[2]));
    assert_eq!(SavedStates::new(0).store(&[1], Kind::Anchor, None), None);
}

#[test]
fn a_tail_entry_needs_its_full_blocks_cached_and_hands_its_block_over() {
    let mut s = SavedStates::new(2);
    let prefix: Vec<u32> = (0..10).collect();
    let slot = s
        .store(&prefix, Kind::Anchor, Some(BlockId(7)))
        .unwrap()
        .slot;
    let next: Vec<u32> = (0..14).collect();
    // Blocks of 4: tokens 0..8 must be in the KV cache, the tail block holds 8..10.
    assert_eq!(s.find(&next, 4, 4), None);
    let f = s.find(&next, 8, 4).unwrap();
    assert_eq!(
        f,
        Found {
            slot,
            len: 10,
            tail: Some(BlockId(7))
        }
    );
    assert_eq!(s.take(slot), Some(BlockId(7)));
    assert!(!s.contains(&prefix));
    assert_eq!(s.find(&next, 12, 4), None);
}

#[test]
fn replaced_and_dropped_tails_go_back_to_the_caller() {
    let mut s = SavedStates::new(1);
    s.store(&[1, 2, 3], Kind::Anchor, Some(BlockId(5))).unwrap();
    let st = s.store(&[4], Kind::Anchor, None).unwrap();
    assert_eq!(st.dropped, Some(BlockId(5)));
    let mut s = SavedStates::new(3);
    s.store(&[1], Kind::Anchor, Some(BlockId(1))).unwrap();
    s.store(&[2], Kind::Anchor, None).unwrap();
    s.store(&[3], Kind::Anchor, Some(BlockId(3))).unwrap();
    let mut d = s.drop_tails();
    d.sort_by_key(|b| b.0);
    assert_eq!(d, vec![BlockId(1), BlockId(3)]);
    assert!(s.contains(&[2]) && !s.contains(&[1]) && !s.contains(&[3]));
}

#[test]
fn only_tails_from_a_block_on_are_dropped() {
    let mut s = SavedStates::new(4);
    s.store(&[1], Kind::Anchor, Some(BlockId(1))).unwrap();
    s.store(&[2], Kind::Anchor, None).unwrap();
    s.store(&[3], Kind::Anchor, Some(BlockId(7))).unwrap();
    s.store(&[4], Kind::Anchor, Some(BlockId(4))).unwrap();
    let mut d = s.drop_tails_from(BlockId(4));
    d.sort_by_key(|b| b.0);
    assert_eq!(d, vec![BlockId(4), BlockId(7)]);
    assert!(s.contains(&[1]) && s.contains(&[2]));
    assert!(!s.contains(&[3]) && !s.contains(&[4]));
    assert!(s.drop_tails_from(BlockId(4)).is_empty());
}

#[test]
fn spaced_saves_fill_at_most_half_the_slots_until_a_prompt_resumes_from_one() {
    let mut s = SavedStates::new(4);
    store(&mut s, &[9]);
    for p in 1..=6u32 {
        s.store(&[1, p], Kind::Spaced, None).unwrap();
    }
    // One free slot went to the third spaced save; after that they replace each other.
    assert!(s.contains(&[9]), "the chat's state stays");
    assert_eq!(
        s.slots.iter().flatten().filter(|e| e.spaced).count(),
        3,
        "free slots fill first"
    );
    assert!(s.contains(&[1, 6]) && s.contains(&[1, 5]) && !s.contains(&[1, 1]));
    // A hit makes a spaced save an anchor: the next spaced saves leave it.
    let hit = found(&mut s, &[1, 5, 0], 10, 1).unwrap().0;
    s.store(&[1, 7], Kind::Spaced, None).unwrap();
    s.store(&[1, 8], Kind::Spaced, None).unwrap();
    assert!(s.contains(&[1, 5]) && s.slots[hit].as_ref().is_some_and(|e| !e.spaced));
    assert!(s.contains(&[9]));
    // With one spaced save in two slots, the next replaces it; an anchor takes the least
    // recently used slot of any kind.
    let mut s = SavedStates::new(2);
    s.store(&[1], Kind::Spaced, None).unwrap();
    store(&mut s, &[2]);
    assert_eq!(s.store(&[3], Kind::Spaced, None).unwrap().slot, 0);
    assert_eq!(store(&mut s, &[4]), 1);
}

#[test]
fn save_points_are_the_branch_point_and_the_last_full_block() {
    let a = |v: &[usize]| v.iter().map(|&p| (p, Kind::Anchor)).collect::<Vec<_>>();
    // Nothing cached: only the end of the last full block.
    assert_eq!(save_points(530, 0, 0, 64, 0), a(&[512]));
    // KV matched 384 tokens but no state there: save at the branch point too.
    assert_eq!(save_points(530, 384, 0, 64, 0), a(&[384, 512]));
    // Resumed at 384: only what lies beyond.
    assert_eq!(save_points(530, 384, 384, 64, 0), a(&[512]));
    // Prompt shorter than a block: nothing to save.
    assert!(save_points(40, 0, 0, 64, 0).is_empty());
    // Whole prompt in full blocks and cached: the end is the branch point.
    assert_eq!(save_points(128, 128, 64, 64, 0), a(&[128]));
}

#[test]
fn interval_save_points_are_spaced_whole_blocks_before_the_last_full_block() {
    use Kind::{Anchor, Spaced};
    // Every 200 tokens, rounded up to 256 (blocks of 64).
    assert_eq!(
        save_points(1100, 0, 0, 64, 200),
        vec![
            (256, Spaced),
            (512, Spaced),
            (768, Spaced),
            (1024, Spaced),
            (1088, Anchor)
        ]
    );
    // A spaced point on the branch point stays an anchor; none at or before `resumed`.
    assert_eq!(
        save_points(1100, 512, 320, 64, 256),
        vec![(512, Anchor), (768, Spaced), (1024, Spaced), (1088, Anchor)]
    );
    // None at the last full block itself.
    assert_eq!(
        save_points(1024, 0, 0, 64, 512),
        vec![(512, Spaced), (1024, Anchor)]
    );
}

#[test]
fn prefix_len_of_a_slot() {
    let mut s = SavedStates::new(2);
    assert_eq!(s.prefix_len(0), None);
    let slot = store(&mut s, &[1, 2, 3]);
    assert_eq!(s.prefix_len(slot), Some(3));
    s.take(slot);
    assert_eq!(s.prefix_len(slot), None);
    assert_eq!(s.prefix_len(9), None);
}
