use super::*;

#[test]
fn default_prefill_slots_keep_the_old_size() {
    // Four prompt groups plus the decode or verify group: every step at the default slots.
    for n in 0..=GRAPH_SIZE_GROUPS {
        assert_eq!(graph_size(n), GRAPH_SIZE, "{n} groups");
    }
}

#[test]
fn more_groups_get_whole_multiples() {
    assert_eq!(graph_size(GRAPH_SIZE_GROUPS + 1), 2 * GRAPH_SIZE);
    assert_eq!(graph_size(2 * GRAPH_SIZE_GROUPS), 2 * GRAPH_SIZE);
    // Sixteen prompt groups and a verify group (`--prefill-slots 16` at 16 users).
    assert_eq!(graph_size(17), 4 * GRAPH_SIZE);
    for n in 1..=300 {
        // Never fewer tensors per group than the default size gives five groups.
        assert!(
            graph_size(n) * GRAPH_SIZE_GROUPS >= GRAPH_SIZE * n,
            "{n} groups"
        );
        assert_eq!(graph_size(n) % GRAPH_SIZE, 0);
    }
}
