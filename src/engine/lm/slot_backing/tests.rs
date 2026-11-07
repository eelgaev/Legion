use super::*;

const C: usize = LAZY_CHUNK_BLOCKS;

#[test]
fn slots_are_backed_in_whole_chunks_up_to_the_slot() {
    let mut b = SlotBacking::new(4, 3 * C, 0, 100 * C, 0);
    let p = b.plan(1, 1, &[], &[]).unwrap().unwrap();
    assert_eq!(
        p,
        Backing {
            trim: vec![],
            demote: vec![],
            to: C,
            lent: 0
        }
    );
    b.backed(1, p.to, 0);
    // Within the chunk: nothing to do.
    assert_eq!(b.plan(1, C, &[], &[]).unwrap(), None);
    assert_eq!(b.plan(1, C + 1, &[], &[]).unwrap().unwrap().to, 2 * C);
    // Never past the slot.
    assert_eq!(b.plan(1, 3 * C, &[], &[]).unwrap().unwrap().to, 3 * C);
    assert_eq!(b.used(), C);
}

#[test]
fn vram_blocks_are_always_backed_and_never_counted() {
    let mut b = SlotBacking::new(2, 4 * C, 2 * C, C, 0);
    assert_eq!(b.plan(0, 2 * C, &[], &[]).unwrap(), None);
    let p = b.plan(0, 2 * C + 1, &[], &[]).unwrap().unwrap();
    assert_eq!(p.to, 3 * C);
    b.backed(0, p.to, 0);
    assert_eq!(b.used(), C);
    b.trimmed(0);
    assert_eq!(b.used(), 0);
}

#[test]
fn idle_slots_give_back_their_room_least_recently_freed_first() {
    let mut b = SlotBacking::new(4, 4 * C, 0, 4 * C, 0);
    for (s, n) in [(0, 2 * C), (1, C), (2, C)] {
        b.backed(s, n, 0);
    }
    assert_eq!(b.used(), 4 * C);
    // Slot 3 needs a chunk; slots 2 and 0 are idle (0 freed first).
    let p = b.plan(3, 1, &[0, 2], &[]).unwrap().unwrap();
    assert_eq!(
        p,
        Backing {
            trim: vec![0],
            demote: vec![],
            to: C,
            lent: 0
        }
    );
    for &s in &p.trim {
        b.trimmed(s);
    }
    b.backed(3, p.to, 0);
    assert_eq!(b.used(), 3 * C);
    // Idle slots with nothing on the host are skipped; short with none left: an error.
    assert!(b.plan(3, 4 * C, &[0], &[]).is_err());
    let p = b.plan(3, 3 * C, &[0, 2], &[]).unwrap().unwrap();
    assert_eq!(p.trim, vec![2]);
}

#[test]
fn shared_vram_goes_to_the_first_blocks_as_slots_grow() {
    let mut b = SlotBacking::new(4, 4 * C, 0, usize::MAX, 3 * C);
    let p = b.plan(0, 2 * C, &[], &[]).unwrap().unwrap();
    assert_eq!((p.to, p.lent), (2 * C, 2 * C));
    b.backed(0, p.to, p.lent);
    assert_eq!(b.used(), 0);
    // Past the shared VRAM: the host.
    let p = b.plan(0, 4 * C, &[], &[]).unwrap().unwrap();
    assert_eq!((p.to, p.lent), (4 * C, 3 * C));
    b.backed(0, p.to, p.lent);
    assert_eq!(b.used(), C);
    // None left for another running slot.
    let p = b.plan(1, C, &[], &[]).unwrap().unwrap();
    assert_eq!((p.to, p.lent, p.demote.len()), (C, 0, 0));
    b.backed(1, p.to, p.lent);
    assert_eq!(b.used(), 2 * C);
    // Given back: its shared VRAM goes to the next slot that grows, its host blocks moved in.
    b.trimmed(0);
    assert_eq!(b.used(), C);
    let p = b.plan(1, C, &[], &[]).unwrap().unwrap();
    assert_eq!((p.to, p.lent), (C, C));
    b.backed(1, p.to, p.lent);
    assert_eq!(b.used(), 0);
    assert_eq!(b.plan(1, C, &[], &[]).unwrap(), None);
}

#[test]
fn warm_slots_give_their_shared_vram_least_recently_finished_first() {
    let mut b = SlotBacking::new(3, 4 * C, 0, usize::MAX, 4 * C);
    b.backed(0, 2 * C, 2 * C);
    b.backed(1, 2 * C, 2 * C);
    // Slot 2 needs three chunks: slot 1 finished first, then slot 0.
    let p = b.plan(2, 3 * C, &[], &[1, 0]).unwrap().unwrap();
    assert_eq!(p.demote, vec![(1, 0), (0, C)]);
    assert_eq!((p.to, p.lent), (3 * C, 3 * C));
    for &(w, keep) in &p.demote {
        let (to, _) = b.held(w);
        b.backed(w, to, keep);
    }
    b.backed(2, p.to, p.lent);
    // The warm slots' blocks moved to the host, the running slot's are all in VRAM.
    assert_eq!(b.used(), 3 * C);
    assert_eq!(b.held(2), (3 * C, 3 * C));
    assert_eq!(b.held(0), (2 * C, C));
}

#[test]
fn shared_vram_counts_against_no_host_budget() {
    let mut b = SlotBacking::new(2, 4 * C, 0, C, 2 * C);
    let p = b.plan(0, 3 * C, &[], &[]).unwrap().unwrap();
    assert_eq!((p.to, p.lent), (3 * C, 2 * C));
    b.backed(0, p.to, p.lent);
    assert_eq!(b.used(), C);
    assert!(b.plan(0, 4 * C, &[], &[]).is_err());
}
