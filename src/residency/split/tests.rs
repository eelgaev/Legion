use super::*;

const MIB: usize = 1 << 20;

#[test]
fn glm_experts_split_at_their_owners_socket() {
    // GLM-5.3-Flash: 288 experts, GPUs 0, 1 (node 0) own 0..144, GPUs 2, 3 (node 8) the rest.
    let nodes: Vec<u32> = (0..288).map(|e| if e < 144 { 0 } else { 8 }).collect();
    for per in [4_718_592usize, 5_767_168] {
        let alloc = (288 * per).next_multiple_of(2 * MIB);
        let p = expert_pieces(&nodes, per, alloc, 2 * MIB);
        assert_eq!(p, vec![(144 * per, 0), (alloc - 144 * per, 8)]);
        // Every GPU reads its own experts from its own node only.
        assert_eq!(foreign_bytes(&p, per, 0..144, 0), 0);
        assert_eq!(foreign_bytes(&p, per, 144..288, 8), 0);
        assert_eq!(foreign_bytes(&p, per, 0..288, 0), 144 * per);
    }
}

#[test]
fn a_cut_off_the_granule_goes_down_to_it() {
    let nodes = [0, 0, 1, 1];
    // 3 MiB experts, 2 MiB granules: the cut at 6 MiB is one; at 3 experts' 9 MiB it would not be.
    let p = expert_pieces(&nodes, 3 * MIB, 12 * MIB, 2 * MIB);
    assert_eq!(p, vec![(6 * MIB, 0), (6 * MIB, 1)]);
    let p = expert_pieces(&[0, 0, 0, 1], 3 * MIB, 12 * MIB, 2 * MIB);
    assert_eq!(p, vec![(8 * MIB, 0), (4 * MIB, 1)]);
    // Expert 2 ends at 9 MiB: its last 1 MiB is on node 1.
    assert_eq!(foreign_bytes(&p, 3 * MIB, 0..3, 0), MIB);
    assert_eq!(foreign_bytes(&p, 3 * MIB, 3..4, 1), 0);
}

#[test]
fn one_node_is_one_piece() {
    assert_eq!(
        expert_pieces(&[5; 10], 3 * MIB, 30 * MIB, 2 * MIB),
        vec![(30 * MIB, 5)]
    );
    // Runs shorter than a granule stay with the run before.
    assert_eq!(
        expert_pieces(&[0, 1, 0], MIB / 2, 2 * MIB, 2 * MIB),
        vec![(2 * MIB, 0)]
    );
    assert_eq!(
        expert_pieces(&[], MIB, 2 * MIB, 2 * MIB),
        vec![(2 * MIB, 0)]
    );
}

#[test]
fn a_fill_is_cut_where_the_pieces_node_changes() {
    let p = [(6 * MIB, 0), (6 * MIB, 8)];
    // A whole tensor: one cut per piece; a range inside one piece stays whole.
    assert_eq!(
        cut_by_node(0..12 * MIB, &p, Some(0)),
        vec![(0..6 * MIB, Some(0)), (6 * MIB..12 * MIB, Some(8))]
    );
    assert_eq!(
        cut_by_node(MIB..2 * MIB, &p, Some(0)),
        vec![(MIB..2 * MIB, Some(0))]
    );
    assert_eq!(
        cut_by_node(5 * MIB..7 * MIB, &p, None),
        vec![(5 * MIB..6 * MIB, Some(0)), (6 * MIB..7 * MIB, Some(8))]
    );
    // The tensor's bytes end before its allocation's padding: the last piece takes the rest.
    assert_eq!(
        cut_by_node(7 * MIB..13 * MIB, &p, None),
        vec![(7 * MIB..13 * MIB, Some(8))]
    );
    // Runs on one node merge; no pieces: the range on the given node.
    let q = [(2 * MIB, 0), (2 * MIB, 0), (2 * MIB, 8)];
    assert_eq!(
        cut_by_node(0..6 * MIB, &q, None),
        vec![(0..4 * MIB, Some(0)), (4 * MIB..6 * MIB, Some(8))]
    );
    assert_eq!(cut_by_node(0..MIB, &[], Some(3)), vec![(0..MIB, Some(3))]);
    assert_eq!(cut_by_node(0..0, &p, Some(3)), vec![]);
}
