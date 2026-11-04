use super::*;

#[test]
fn writes_split_into_pieces_of_a_chunk() {
    let mut p = Pending::default();
    p.push(0x1000, 256, 40);
    p.push(0x9000, 512, 2 * CHUNK + 12);
    assert_eq!(
        p.pieces,
        vec![
            Piece {
                dst: 0x1000,
                src: 256,
                len: 40
            },
            Piece {
                dst: 0x9000,
                src: 512,
                len: CHUNK as u32
            },
            Piece {
                dst: 0x9000 + CHUNK as u64,
                src: (512 + CHUNK) as u32,
                len: CHUNK as u32
            },
            Piece {
                dst: 0x9000 + 2 * CHUNK as u64,
                src: (512 + 2 * CHUNK) as u32,
                len: 12
            },
        ]
    );
    // Every byte once.
    let total: u32 = p.pieces.iter().map(|q| q.len).sum();
    assert_eq!(total as usize, 40 + 2 * CHUNK + 12);
}

#[test]
fn empty_writes_hold_nothing() {
    let mut p = Pending::default();
    p.push(0x1000, 0, 0);
    assert!(p.is_empty());
    assert!(!p.overlaps(0x1000, 4));
}

#[test]
fn overlaps_only_where_ranges_meet() {
    let mut p = Pending::default();
    p.push(1000, 0, 100);
    assert!(p.overlaps(1000, 1));
    assert!(p.overlaps(1099, 1));
    assert!(p.overlaps(900, 101));
    assert!(p.overlaps(950, 500));
    assert!(!p.overlaps(1100, 8), "just after");
    assert!(!p.overlaps(900, 100), "just before");
    assert!(!p.overlaps(1050, 0), "an empty write");
    p.clear();
    assert!(p.is_empty());
    assert!(!p.overlaps(1000, 100));
}

#[test]
fn the_switch_turns_packing_off_and_on() {
    set_input_pack(false);
    assert!(!enabled());
    set_input_pack(true);
    assert!(enabled());
}
