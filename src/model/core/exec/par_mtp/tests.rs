use super::*;

#[test]
fn only_split_parallel_passes_with_output_rows_run_on_every_gpu() {
    assert!(runs_par(true, true, 1));
    assert!(runs_par(true, true, 8));
    // A take-in piece without output rows stays on the last GPU.
    assert!(!runs_par(true, true, 0));
    // Without --glm-parallel-mtp, or without parallel steps.
    assert!(!runs_par(true, false, 4));
    assert!(!runs_par(false, true, 4));
}

#[test]
fn the_mtp_entries_are_the_head_and_its_copies() {
    // 6 layers over 4 GPUs: the head is entry 6, its copies 25..28.
    let copies = 25..28;
    assert!(is_mtp_entry(6, 6, &copies));
    for e in copies.clone() {
        assert!(is_mtp_entry(e, 6, &copies));
    }
    // The decoder layers and their copies are not.
    for e in (0..6).chain(7..25) {
        assert!(!is_mtp_entry(e, 6, &copies), "{e}");
    }
    assert!(!is_mtp_entry(28, 6, &copies));
    // No copies: the head alone.
    assert!(!is_mtp_entry(7, 6, &(7..7)));
}

/// A GPU's job queues both graphs in the order given, the second even when the first fails
/// to start, and returns the first error (#723).
#[test]
fn a_gpus_job_queues_both_graphs_in_order() {
    use std::cell::RefCell;
    let order = RefCell::new(Vec::new());
    let r = in_order(
        || {
            order.borrow_mut().push("first");
            Ok(())
        },
        || {
            order.borrow_mut().push("second");
            Ok(())
        },
    );
    assert!(r.is_ok());
    assert_eq!(*order.borrow(), ["first", "second"]);
    order.borrow_mut().clear();
    let r = in_order(
        || {
            order.borrow_mut().push("first");
            Err(GgmlError::Compute(1))
        },
        || {
            order.borrow_mut().push("second");
            Err(GgmlError::Compute(2))
        },
    );
    assert!(matches!(r, Err(GgmlError::Compute(1))));
    assert_eq!(*order.borrow(), ["first", "second"]);
    order.borrow_mut().clear();
    let r = in_order(
        || {
            order.borrow_mut().push("first");
            Ok(())
        },
        || {
            order.borrow_mut().push("second");
            Err(GgmlError::Compute(2))
        },
    );
    assert!(matches!(r, Err(GgmlError::Compute(2))));
    assert_eq!(*order.borrow(), ["first", "second"]);
}

/// The #723 order applies to steps of at most the set number of sequences; 0 turns it off
/// (#725).
#[test]
fn the_overlap_applies_up_to_its_sequence_count() {
    // The default, one sequence.
    assert!(overlap_applies(1, 1));
    assert!(!overlap_applies(1, 2));
    assert!(!overlap_applies(1, 4));
    // Off.
    assert!(!overlap_applies(0, 1));
    // Every step (--glm-mtp-overlap).
    assert!(overlap_applies(usize::MAX, 4));
    // A step without sequences never.
    assert!(!overlap_applies(1, 0));
}
