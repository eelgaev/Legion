use super::*;

/// Ids are made once per name, in order; an unknown id has a placeholder name.
#[test]
fn names_intern_once() {
    let mut n = Names::default();
    assert_eq!(n.id("kernel:a"), 0);
    assert_eq!(n.id("memset"), 1);
    assert_eq!(n.id("kernel:a"), 0);
    assert_eq!(n.name(1), "memset");
    assert_eq!(n.name(7), "?");
}

/// Equal lists: everything is prefix, nothing differs.
#[test]
fn compare_equal_lists() {
    let d = compare(&[1, 2, 3], &[1, 2, 3]);
    assert_eq!(d.prefix, 3);
    assert_eq!(d.suffix, 0);
    assert!(d.counts.is_empty());
    assert!(d.same(3, 3));
}

/// An inserted node shows up as one count change between the shared front and back.
#[test]
fn compare_insertion() {
    let d = compare(&[1, 2, 3, 4], &[1, 2, 9, 3, 4]);
    assert_eq!((d.prefix, d.suffix), (2, 2));
    assert_eq!(d.counts, vec![(9, 1)]);
    assert!(!d.same(4, 5));
}

/// A swapped kernel at one position: one more of the new kind, one fewer of the old.
#[test]
fn compare_swap_largest_changes_first() {
    let d = compare(&[1, 5, 5, 3], &[1, 7, 7, 3]);
    assert_eq!((d.prefix, d.suffix), (1, 1));
    assert_eq!(d.counts, vec![(5, -2), (7, 2)]);
    assert!(!d.same(4, 4));
}

/// The back never overlaps the front, even with repeated kinds.
#[test]
fn compare_suffix_does_not_overlap_prefix() {
    let d = compare(&[1, 1], &[1, 1, 1]);
    assert_eq!((d.prefix, d.suffix), (2, 0));
    assert_eq!(d.counts, vec![(1, 1)]);
    let d = compare(&[], &[4]);
    assert_eq!((d.prefix, d.suffix), (0, 0));
    assert_eq!(d.counts, vec![(4, 1)]);
}

/// The nearest capture is of the same node count if any (the newest), else of the closest
/// count, the smaller on a tie; only of the same device.
#[test]
fn recent_nearest() {
    let mut r = Recent::default();
    assert_eq!(r.nearest(0, 3), None);
    r.push(0, vec![1, 1, 1]);
    r.push(0, vec![2, 2, 2]);
    r.push(0, vec![3; 5]);
    r.push(1, vec![9; 4]);
    assert_eq!(r.nearest(0, 3), Some(&[2u32, 2, 2][..]));
    assert_eq!(
        r.nearest(0, 4),
        Some(&[2u32, 2, 2][..]),
        "tie: the smaller count"
    );
    assert_eq!(r.nearest(0, 6), Some(&[3u32; 5][..]));
    assert_eq!(r.nearest(1, 3), Some(&[9u32; 4][..]));
    assert_eq!(r.nearest(2, 3), None);
}

/// A device keeps at most RECENT captures, dropping its oldest; other devices' stay.
#[test]
fn recent_drops_the_oldest_of_the_device() {
    let mut r = Recent::default();
    r.push(1, vec![42]);
    for i in 0..RECENT as u32 + 1 {
        r.push(0, vec![i; 2]);
    }
    assert_eq!(r.nearest(0, 2), Some(&[RECENT as u32; 2][..]));
    assert_eq!(r.list.iter().filter(|(d, _)| *d == 0).count(), RECENT);
    assert!(!r.list.iter().any(|(d, s)| *d == 0 && s[0] == 0));
    assert_eq!(r.nearest(1, 1), Some(&[42u32][..]));
}

/// CUDA's result and node type codes have names; others are unknown.
#[test]
fn code_names() {
    assert_eq!(result_name(2), "topology_changed");
    assert_eq!(result_name(5), "parameters_changed");
    assert_eq!(result_name(8), "attributes_changed");
    assert_eq!(result_name(99), "unknown");
    assert_eq!(node_type_name(0), "kernel");
    assert_eq!(node_type_name(2), "memset");
    assert_eq!(node_type_name(40), "unknown");
}
