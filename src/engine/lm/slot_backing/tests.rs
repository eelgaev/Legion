use super::*;

const C: usize = LAZY_CHUNK_BLOCKS;

#[test]
fn slots_are_backed_in_whole_chunks_up_to_the_slot() {
    let mut b = SlotBacking::new(4, 3 * C, 0, 100 * C);
    let p = b.plan(1, 1, &[]).unwrap().unwrap();
    assert_eq!(
        p,
        Backing {
            trim: vec![],
            to: C
        }
    );
    b.backed(1, p.to);
    // Within the chunk: nothing to do.
    assert_eq!(b.plan(1, C, &[]).unwrap(), None);
    assert_eq!(b.plan(1, C + 1, &[]).unwrap().unwrap().to, 2 * C);
    // Never past the slot.
    assert_eq!(b.plan(1, 3 * C, &[]).unwrap().unwrap().to, 3 * C);
    assert_eq!(b.used(), C);
}

#[test]
fn vram_blocks_are_always_backed_and_never_counted() {
    let mut b = SlotBacking::new(2, 4 * C, 2 * C, C);
    assert_eq!(b.plan(0, 2 * C, &[]).unwrap(), None);
    let p = b.plan(0, 2 * C + 1, &[]).unwrap().unwrap();
    assert_eq!(p.to, 3 * C);
    b.backed(0, p.to);
    assert_eq!(b.used(), C);
    b.trimmed(0);
    assert_eq!(b.used(), 0);
}

#[test]
fn idle_slots_give_back_their_room_least_recently_freed_first() {
    let mut b = SlotBacking::new(4, 4 * C, 0, 4 * C);
    for (s, n) in [(0, 2 * C), (1, C), (2, C)] {
        b.backed(s, n);
    }
    assert_eq!(b.used(), 4 * C);
    // Slot 3 needs a chunk; slots 2 and 0 are idle (0 freed first).
    let p = b.plan(3, 1, &[0, 2]).unwrap().unwrap();
    assert_eq!(
        p,
        Backing {
            trim: vec![0],
            to: C
        }
    );
    for &s in &p.trim {
        b.trimmed(s);
    }
    b.backed(3, p.to);
    assert_eq!(b.used(), 3 * C);
    // Idle slots with nothing on the host are skipped; short with none left: an error.
    assert!(b.plan(3, 4 * C, &[0]).is_err());
    let p = b.plan(3, 3 * C, &[0, 2]).unwrap().unwrap();
    assert_eq!(p.trim, vec![2]);
}
