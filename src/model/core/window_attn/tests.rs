use super::slot_stage_copies;

const ALL: usize = usize::MAX;

#[test]
fn off_never_copies() {
    assert!(!slot_stage_copies((0, 0, ALL), 4096, 262_144));
}

#[test]
fn decode_and_verify_steps_read_in_place() {
    // One token, or an MTP verify of k + 1 rows: flash attention reads the span once anyway.
    assert!(!slot_stage_copies((64, 0, ALL), 1, 131_072));
    assert!(!slot_stage_copies((64, 0, ALL), 4, 131_072));
    assert!(!slot_stage_copies((64, 0, ALL), 63, 131_072));
}

#[test]
fn prompt_steps_copy_spans_past_the_vram_part() {
    assert!(slot_stage_copies((64, 0, ALL), 64, 256));
    assert!(slot_stage_copies((64, 8192, ALL), 512, 8448));
    assert!(slot_stage_copies((64, 8192, ALL), 2048, 122_880));
}

#[test]
fn spans_inside_the_vram_part_read_in_place() {
    assert!(!slot_stage_copies((64, 8192, ALL), 512, 8192));
    assert!(!slot_stage_copies((64, 8192, ALL), 512, 4096));
}

#[test]
fn spans_past_the_room_read_in_place() {
    assert!(slot_stage_copies((64, 0, 196_608), 512, 196_608));
    assert!(!slot_stage_copies((64, 0, 196_608), 512, 196_672));
    assert!(!slot_stage_copies((64, 0, 0), 512, 64));
}
