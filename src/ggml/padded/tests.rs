use super::{PAD_KEEPS_FREE, PASS_PAD, light, padded, pass};

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

#[test]
fn pads_a_quarter_and_64_mib_with_memory_to_spare() {
    assert_eq!(padded(400 * MIB, 16 * GIB), 564 * MIB);
    assert_eq!(padded(0, usize::MAX), 64 * MIB);
}

/// The 11.8K-token step graph of #490: 4,522 MiB with 5,864 MiB free. Padded in full it left
/// 148 MiB; now it leaves what the padding keeps free.
#[test]
fn padding_leaves_the_reserve_free() {
    let (size, free) = (4522 * MIB, 5864 * MIB);
    let got = padded(size, free);
    assert!(got > size && got < size + size / 4 + 64 * MIB);
    assert_eq!(free - got, PAD_KEEPS_FREE);
}

#[test]
fn no_padding_without_room() {
    assert_eq!(padded(4522 * MIB, 4522 * MIB + PAD_KEEPS_FREE), 4522 * MIB);
    // Not even room for the request: it is asked for as it is (and fails as it would have).
    assert_eq!(padded(6 * GIB, 5 * GIB), 6 * GIB);
}

/// #363: the GB10's 1,575 MiB prompt-chunk buffer gets ~40 MiB, room for hundreds of the
/// ~0.04 MiB steps it grew by per new KV length.
#[test]
fn light_pads_a_sixty_fourth_and_16_mib() {
    assert_eq!(
        light(1536 * MIB, 64 * GIB),
        1536 * MIB + 24 * MIB + 16 * MIB
    );
    assert_eq!(light(0, usize::MAX), 16 * MIB);
}

#[test]
fn light_padding_leaves_the_reserve_free() {
    let (size, free) = (4 * GIB, 4 * GIB + PAD_KEEPS_FREE + 10 * MIB);
    assert_eq!(light(size, free), size + 10 * MIB);
    assert_eq!(light(size, size), size);
}

/// #37: a 10,944-token staged pass of 4,209 MiB with 4,913 MiB free (a 32K prompt's three
/// passes on an RTX 4090): no quarter, the pass pad within the 512 MiB it leaves.
#[test]
fn a_pass_pads_a_little_where_the_quarter_does_not() {
    let (size, free) = (4209 * MIB, 4913 * MIB);
    assert_eq!(pass(size, free), size + PASS_PAD);
    // Only as far as 512 MiB stays free; never below the request.
    assert_eq!(pass(size, size + 520 * MIB), size + 8 * MIB);
    assert_eq!(pass(size, size), size);
    // Room for the quarter: as a lent cache's other steps.
    let roomy = 16 * GIB;
    assert_eq!(pass(400 * MIB, roomy), padded(400 * MIB, roomy));
}
