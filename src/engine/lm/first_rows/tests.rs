use super::*;

#[test]
fn reads_ahead_the_first_chunk_only() {
    // A short prompt: all of it.
    assert_eq!(ahead_len(120, 2048, 4096), 120);
    // A long one: its first chunk.
    assert_eq!(ahead_len(8192, 2048, 4096), 2048);
    // The step's token budget caps the chunk.
    assert_eq!(ahead_len(8192, 2048, 1024), 1024);
    // Nothing to read for an empty prompt.
    assert_eq!(ahead_len(0, 2048, 4096), 0);
    // Zero settings are taken as one token per step.
    assert_eq!(ahead_len(10, 0, 0), 1);
}
