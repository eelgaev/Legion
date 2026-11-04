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
