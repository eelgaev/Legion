use super::*;

#[test]
fn vram_part_leaves_room_in_whole_granules() {
    let g = 2 << 20;
    let k = SPILL_KEEPS_FREE;
    // Fits: all of it in VRAM.
    assert_eq!(vram_part(100 << 20, 4 << 30, k, g), 100 << 20);
    // Short: what leaves SPILL_KEEPS_FREE free.
    assert_eq!(vram_part(4 << 30, 1 << 30, k, g), 512 << 20);
    assert_eq!(vram_part(4 << 30, (512 << 20) + (3 << 20), k, g), 2 << 20);
    // Nothing to spare: all on the host.
    assert_eq!(vram_part(4 << 30, 100 << 20, k, g), 0);
}

#[test]
fn vram_part_of_a_decode_state_keeps_its_room() {
    let g = 2 << 20;
    // A 10 GB ring state next to 6.5 GiB free, 2 GiB kept: 4.5 GiB in VRAM, the rest spills.
    let free = (13 << 30) / 2;
    assert_eq!(vram_part(10_000 << 20, free, 2 << 30, g), 9 << 29);
    // It fits with the room kept: all in VRAM, as without the spill.
    assert_eq!(vram_part(4 << 30, free, 2 << 30, g), 4 << 30);
}
