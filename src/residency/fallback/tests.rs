use super::*;
use crate::residency::Platform;

#[test]
fn known_node_is_kept_as_before() {
    // An AC922 GPU on node 8 and a GB10 on node 0: the node itself, the no-node probe never asked.
    for n in [0u32, 8] {
        let got = host_node(Some(n), |_| true, || panic!("asked for TYPE_HOST"));
        assert_eq!(got, Some(n));
    }
    // A driver that cannot place memory on the node: none, as before (no fallback to another).
    assert_eq!(
        host_node(Some(8), |_| false, || panic!("asked for TYPE_HOST")),
        None
    );
}

#[test]
fn unknown_node_uses_type_host_then_node_zero() {
    assert_eq!(host_node(None, |_| true, || true), Some(NO_NODE));
    assert_eq!(HostLoc::of(NO_NODE), HostLoc::Host);
    // A driver without TYPE_HOST in cuMemCreate: HOST_NUMA node 0, as CUDA 12's docs ask on a
    // system without NUMA.
    assert_eq!(host_node(None, |n| n == 0, || false), Some(0));
    assert_eq!(host_node(None, |_| false, || false), None);
}

#[test]
fn host_loc_round_trips() {
    for n in [0u32, 1, 8, 255] {
        assert_eq!(HostLoc::of(n), HostLoc::Numa(n));
        assert_eq!(HostLoc::of(n).node(), n);
    }
    assert_eq!(HostLoc::Host.node(), NO_NODE);
}

#[test]
fn no_node_is_never_reported_as_a_node() {
    assert_eq!(known_node(8), Some(8));
    assert_eq!(known_node(0), Some(0));
    assert_eq!(known_node(NO_NODE), None);
    assert_eq!(node_label(8), "8");
    assert_eq!(node_label(NO_NODE), "none");
}

#[test]
fn notes_name_the_location_type() {
    assert!(host_node_note(Some(NO_NODE)).contains("CU_MEM_LOCATION_TYPE_HOST"));
    assert!(host_node_note(Some(0)).contains("node 0"));
    assert!(host_node_note(None).contains("no step-buffer spill"));
}

#[test]
fn pageable_access_reads_in_place() {
    // ATS (AC922), the GB10's shared memory, HMM: whatever registration would allow.
    for (reg, ro) in [(true, true), (true, false), (false, false)] {
        assert_eq!(host_expert_read(true, reg, ro), HostExpertRead::InPlace);
    }
}

#[test]
fn no_pageable_access_registers_or_refuses() {
    assert_eq!(
        host_expert_read(false, true, true),
        HostExpertRead::Register { read_only: true }
    );
    assert_eq!(
        host_expert_read(false, true, false),
        HostExpertRead::Register { read_only: false }
    );
    assert!(matches!(
        host_expert_read(false, false, true),
        HostExpertRead::Refuse(_)
    ));
}

#[test]
fn page_spans_round_and_merge() {
    const P: usize = 4096;
    // Two tensors sharing a page merge; a third pages away stays apart; empty ones go.
    let r = [(P + 10, 100), (P + 200, P), (10 * P, 1), (3 * P, 0)];
    assert_eq!(page_spans(&r, P), vec![(P, 2 * P), (10 * P, P)]);
    // Touching spans merge too, and order does not matter.
    let r = [(2 * P, P), (0, 2 * P)];
    assert_eq!(page_spans(&r, P), vec![(0, 3 * P)]);
    // Contained ranges add nothing.
    let r = [(0, 8 * P), (P, P)];
    assert_eq!(page_spans(&r, P), vec![(0, 8 * P)]);
    assert!(page_spans(&[], P).is_empty());
}

#[test]
fn peer_stage_pairs_checked_in_the_writing_direction() {
    // GPUs 0,1 paired, 2 without a peer: the peer (writer) must reach the device.
    let pairs = [Some(1), Some(0), None];
    assert!(missing_peer_access(&pairs, |_, _| true).is_empty());
    let one_way = |p: usize, d: usize| !(p == 1 && d == 0);
    assert_eq!(missing_peer_access(&pairs, one_way), vec![(1, 0)]);
    assert_eq!(
        missing_peer_access(&pairs, |_, _| false),
        vec![(1, 0), (0, 1)]
    );
}

#[test]
fn numa_auto_prefers_only_on_pcie() {
    assert!(numa_auto_prefers(Platform::DiscretePcie));
    for p in [
        Platform::CoherentNvlink,
        Platform::NvlinkC2c,
        Platform::Integrated,
    ] {
        assert!(!numa_auto_prefers(p));
    }
}
