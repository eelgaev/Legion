use super::*;

fn steady() -> Check {
    Check {
        on: true,
        kept: Some(7),
        epoch: 3,
        seen: Some(Seen {
            serial: 7,
            epoch: 3,
        }),
        weighed: true,
        same: 5,
        last_tail: Some(u64::MAX),
        tail: u64::MAX,
    }
}

#[test]
fn an_unchanged_kept_graph_skips() {
    assert!(skip(steady()));
}

#[test]
fn anything_else_compares() {
    let off = Check {
        on: false,
        ..steady()
    };
    assert!(!skip(off), "turned off");
    let plain = Check {
        kept: None,
        seen: None,
        ..steady()
    };
    assert!(!skip(plain), "a graph that is not kept");
    let other = Check {
        kept: Some(8),
        ..steady()
    };
    assert!(!skip(other), "another kept graph at the same address");
    let epoch = Check {
        epoch: 4,
        ..steady()
    };
    assert!(!skip(epoch), "words_may_change since");
    let first = Check {
        seen: None,
        ..steady()
    };
    assert!(!skip(first), "no full compare of this graph yet");
    let fresh = Check {
        weighed: false,
        ..steady()
    };
    assert!(!skip(fresh), "not in the steady form yet");
    let late = Check {
        tail: 12,
        ..steady()
    };
    assert!(!skip(late), "the late input read elsewhere");
    let none = Check {
        last_tail: None,
        ..steady()
    };
    assert!(!skip(none), "no words kept");
}

#[test]
fn every_so_often_compares_in_full() {
    let due = Check {
        same: VERIFY_EVERY - 1,
        ..steady()
    };
    assert!(!skip(due));
    let after = Check {
        same: VERIFY_EVERY,
        ..steady()
    };
    assert!(skip(after));
    let saturated = Check {
        same: u32::MAX,
        ..steady()
    };
    // The count saturates (no overflow panic); u32::MAX is no multiple of VERIFY_EVERY.
    assert!(skip(saturated));
}

#[test]
fn seen_records_kept_graphs_only() {
    assert_eq!(seen(None, 2), None);
    assert_eq!(
        seen(Some(5), 2),
        Some(Seen {
            serial: 5,
            epoch: 2
        })
    );
}

#[test]
fn words_may_change_starts_a_new_epoch() {
    let before = epoch();
    words_may_change();
    assert!(epoch() > before);
}

#[test]
fn only_an_unchanged_kept_run_holds_its_entry() {
    assert!(hold(true, true, false, false));
    assert!(!hold(false, true, false, false), "not a kept graph");
    assert!(
        !hold(true, false, false, false),
        "compared its words: the shared cache"
    );
    assert!(
        !hold(true, true, true, false),
        "a new set goes into the store"
    );
}

/// #858 `--stage-graph-hold-kept`: every run of a kept graph holds its entry (and its captures,
/// new ones too) with the graph; other graphs still go through the shared cache.
#[test]
fn hold_all_keeps_every_kept_run_with_its_graph() {
    for skipped in [false, true] {
        for made in [false, true] {
            assert!(hold(true, skipped, made, true), "{skipped} {made}");
            assert!(!hold(false, skipped, made, true), "not a kept graph");
        }
    }
}

fn found() -> LateAt {
    LateAt {
        serial: 7,
        epoch: 3,
        tensor: 0x1000,
        nodes: 900,
        at: 41,
    }
}

#[test]
fn a_late_reader_is_found_once_per_kept_graph() {
    assert_eq!(
        LateAt::again(Some(found()), true, Some(7), 3, 0x1000, 900),
        Some(41)
    );
}

#[test]
fn a_late_reader_is_searched_again_otherwise() {
    let again = |last, on, kept, epoch, tensor, nodes| {
        LateAt::again(last, on, kept, epoch, tensor, nodes).is_some()
    };
    assert!(!again(None, true, Some(7), 3, 0x1000, 900), "not found yet");
    assert!(
        !again(Some(found()), false, Some(7), 3, 0x1000, 900),
        "turned off"
    );
    assert!(
        !again(Some(found()), true, None, 3, 0x1000, 900),
        "not a kept graph"
    );
    assert!(
        !again(Some(found()), true, Some(8), 3, 0x1000, 900),
        "another kept graph"
    );
    assert!(
        !again(Some(found()), true, Some(7), 4, 0x1000, 900),
        "a new epoch"
    );
    assert!(
        !again(Some(found()), true, Some(7), 3, 0x2000, 900),
        "another input"
    );
    assert!(
        !again(Some(found()), true, Some(7), 3, 0x1000, 901),
        "other nodes"
    );
}
