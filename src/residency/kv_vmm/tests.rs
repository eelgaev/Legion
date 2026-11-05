use super::*;

const MIB: usize = 1 << 20;

#[test]
fn a_tensor_is_cut_into_chunks_from_its_start() {
    let p = pieces(4 * MIB, 70 * MIB, 2 * MIB, 32 * MIB);
    assert_eq!(
        p,
        vec![
            (4 * MIB, 32 * MIB),
            (36 * MIB, 32 * MIB),
            (68 * MIB, 6 * MIB)
        ]
    );
    // The pieces cover the tensor exactly, back to back.
    assert_eq!(p.iter().map(|x| x.1).sum::<usize>(), 70 * MIB);
    assert!(p.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
}

#[test]
fn a_small_tensor_is_one_piece() {
    assert_eq!(pieces(0, 2 * MIB, 2 * MIB, 32 * MIB), vec![(0, 2 * MIB)]);
    assert_eq!(
        pieces(6 * MIB, 32 * MIB, 2 * MIB, 32 * MIB),
        vec![(6 * MIB, 32 * MIB)]
    );
    assert!(pieces(0, 0, 2 * MIB, 32 * MIB).is_empty());
}

#[test]
fn chunks_are_whole_granules() {
    // A chunk that is no multiple of the granule is cut down to one; below a granule, one.
    assert_eq!(
        pieces(0, 8 * MIB, 2 * MIB, 5 * MIB),
        vec![(0, 4 * MIB), (4 * MIB, 4 * MIB)]
    );
    assert_eq!(
        pieces(0, 4 * MIB, 2 * MIB, MIB),
        vec![(0, 2 * MIB), (2 * MIB, 2 * MIB)]
    );
}

#[test]
fn a_lend_spans_the_blocks_past_the_kept_ones() {
    // 1,280-byte rows, 64 rows a block, 2,048 blocks: blocks 512.. from byte 512 x 64 x 1280.
    assert_eq!(
        lend_range(1280, 64, 512, 2048),
        (512 * 64 * 1280, 2048 * 64 * 1280)
    );
    // Keeping every block (or more) lends nothing.
    let (from, to) = lend_range(1280, 64, 4096, 2048);
    assert_eq!(from, to);
}

#[test]
fn kept_tokens_are_whole_blocks_within_the_pool() {
    assert_eq!(keep_blocks(32768, 64, 2048), 512);
    assert_eq!(keep_blocks(100, 64, 2048), 2);
    assert_eq!(keep_blocks(1, 64, 2048), 1);
    assert_eq!(keep_blocks(1 << 20, 64, 2048), 2048);
}

#[test]
#[should_panic]
fn an_unaligned_tensor_is_refused() {
    pieces(MIB, 2 * MIB, 2 * MIB, 32 * MIB);
}

#[test]
fn held_blocks_keep_the_pieces_they_touch() {
    // 1 KiB rows, 64 rows a block: 64 KiB blocks, 64 to a 4 MiB piece. Blocks 512.. lent.
    let (row, block) = (1024, 64);
    let from = 512 * 64 * 1024;
    let lent: Vec<(usize, usize)> = (0..4).map(|i| (from + i * 4 * MIB, 4 * MIB)).collect();
    // Nothing held: nothing kept.
    assert!(held_pieces(&lent, row, block, &[]).is_empty());
    // A block in the first piece, two in the third, one past the lent pieces.
    let held = [512, 640, 700, 900];
    assert_eq!(held_pieces(&lent, row, block, &held), vec![0, 2]);
    // The last block of a piece and the first of the next.
    assert_eq!(held_pieces(&lent, row, block, &[575, 576]), vec![0, 1]);
    // Blocks below the lent pieces (kept anyway) touch none.
    assert!(held_pieces(&lent, row, block, &[0, 511]).is_empty());
}

#[test]
fn a_block_across_two_pieces_keeps_both() {
    // 3,000-byte rows: a block (192,000 bytes) does not divide a 1 MiB piece.
    let (row, block) = (3000, 64);
    let lent = vec![(0, MIB), (MIB, MIB), (2 * MIB, MIB / 2)];
    // Block 5 spans bytes 960,000..1,152,000: pieces 0 and 1.
    assert_eq!(held_pieces(&lent, row, block, &[5]), vec![0, 1]);
    // Block 10 starts at 1,920,000 and ends past 2 MiB: pieces 1 and 2.
    assert_eq!(held_pieces(&lent, row, block, &[10]), vec![1, 2]);
    // An empty piece holds nothing.
    assert!(held_pieces(&[(0, 0)], row, block, &[0]).is_empty());
}
