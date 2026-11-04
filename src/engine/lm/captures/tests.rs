use super::chunk_end;
use crate::engine::saved_states::Kind::{Anchor, Spaced};

#[test]
fn without_captures_every_point_ends_the_chunk() {
    let saves = [(1024, Spaced), (2048, Spaced), (2368, Anchor)];
    assert_eq!(chunk_end(&saves, 0, 4096, false), 1024);
    assert_eq!(chunk_end(&saves[1..], 1024, 4096, false), 2048);
    assert_eq!(chunk_end(&saves[2..], 2048, 4096, false), 2368);
    // A point at or past the planned end leaves it.
    assert_eq!(chunk_end(&saves, 0, 1024, false), 1024);
    assert_eq!(chunk_end(&[], 0, 4096, false), 4096);
}

#[test]
fn spaced_points_inside_the_chunk_are_captured_anchors_still_cut() {
    let saves = [(1024, Spaced), (2048, Spaced), (2368, Anchor)];
    assert_eq!(chunk_end(&saves, 0, 4096, true), 2368);
    assert_eq!(chunk_end(&saves[..2], 0, 4096, true), 4096);
}

#[test]
fn points_the_step_cannot_split_at_cut() {
    // 924 tokens into a chunk from 100: not a multiple of 64.
    assert_eq!(chunk_end(&[(1024, Spaced)], 100, 4096, true), 1024);
    // 32 tokens before the planned end: the last piece would be too short. Cut there; the
    // earlier point is then inside the shorter chunk.
    let saves = [(1024, Spaced), (4064, Spaced)];
    assert_eq!(chunk_end(&saves, 0, 4096, true), 4064);
    // Cutting at a point can leave an earlier one too close to the new end: cut again.
    let saves = [(4032, Spaced), (4064, Spaced)];
    assert_eq!(chunk_end(&saves, 0, 4096, true), 4032);
}
