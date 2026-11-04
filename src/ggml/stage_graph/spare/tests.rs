use super::*;

/// Small captures and a cap of 0 keep nothing: the value comes back to be destroyed.
#[test]
fn small_or_off_are_not_kept() {
    let mut s = Spares::default();
    assert_eq!(s.put(0, MIN_NODES - 1, 0, 1, 8), vec![1]);
    assert_eq!(s.put(0, MIN_NODES, 0, 2, 0), vec![2]);
    assert_eq!(s.len(0), 0);
    assert_eq!(s.take(0, MIN_NODES - 1, 0), None);
}

/// A spare is taken only for its own device and node count, the newest first, and once.
#[test]
fn take_matches_device_and_nodes_newest_first() {
    let mut s = Spares::default();
    let n = MIN_NODES + 100;
    assert!(s.put(0, n, 0, 10, 8).is_empty());
    assert!(s.put(0, n, 0, 11, 8).is_empty());
    assert!(s.put(1, n, 0, 20, 8).is_empty());
    assert!(s.put(0, n + 1, 0, 12, 8).is_empty());
    assert_eq!(s.take(0, n + 2, 0), None, "no spare of that node count");
    assert_eq!(s.take(2, n, 0), None, "no spare on that device");
    assert_eq!(s.take(0, n, 0), Some(11));
    assert_eq!(s.take(0, n, 0), Some(10));
    assert_eq!(s.take(0, n, 0), None);
    assert_eq!(s.take(1, n, 0), Some(20));
    assert_eq!(s.take(0, n + 1, 0), Some(12));
}

/// Past the cap a device's oldest spares go (returned to be destroyed); other devices'
/// spares do not count against it.
#[test]
fn cap_drops_the_oldest_of_the_device() {
    let mut s = Spares::default();
    let n = MIN_NODES;
    assert!(s.put(1, n, 0, 100, 2).is_empty());
    assert!(s.put(0, n, 0, 1, 2).is_empty());
    assert!(s.put(0, n + 5, 0, 2, 2).is_empty());
    assert_eq!(s.put(0, n, 0, 3, 2), vec![1]);
    assert_eq!(s.len(0), 2);
    assert_eq!(s.len(1), 1);
    assert_eq!(s.take(0, n, 0), Some(3));
    assert_eq!(s.take(0, n + 5, 0), Some(2));
    assert_eq!(s.take(1, n, 0), Some(100));
}

/// A spare is taken only by a capture whose clustered kernels sit at the same nodes with the
/// same cluster shapes (#1132: a node instantiated without a cluster keeps running without one).
#[test]
fn take_matches_cluster_layout() {
    let mut s = Spares::default();
    let n = MIN_NODES + 7;
    let plain = cluster_key([(0, [0, 0, 0]), (5, [1, 1, 1])]);
    let at5 = cluster_key([(5, [8, 1, 1])]);
    assert!(s.put(0, n, plain, 1, 8).is_empty());
    assert!(s.put(0, n, at5, 2, 8).is_empty());
    assert_eq!(s.take(0, n, cluster_key([(6, [8, 1, 1])])), None);
    assert_eq!(s.take(0, n, cluster_key([(5, [4, 1, 1])])), None);
    assert_eq!(s.take(0, n, plain), Some(1));
    assert_eq!(s.take(0, n, plain), None);
    assert_eq!(s.take(0, n, at5), Some(2));
}

/// No cluster of more than one CTA is key 0; a cluster's node index and shape both count.
#[test]
fn cluster_key_tells_layouts_apart() {
    assert_eq!(cluster_key([]), 0);
    assert_eq!(
        cluster_key([(3, [0, 0, 0]), (9, [1, 1, 1]), (4, [1, 0, 1])]),
        0
    );
    let a = cluster_key([(3, [8, 1, 1])]);
    assert_ne!(a, 0);
    assert_eq!(a, cluster_key([(1, [1, 1, 1]), (3, [8, 1, 1])]));
    assert_ne!(a, cluster_key([(4, [8, 1, 1])]));
    assert_ne!(a, cluster_key([(3, [4, 1, 1])]));
    assert_ne!(a, cluster_key([(3, [8, 1, 1]), (7, [8, 1, 1])]));
}

/// A capture is taken over only when unlaunched for at least the minimum, the longest
/// unlaunched first.
#[test]
fn pick_idle_takes_the_longest_unlaunched_past_the_minimum() {
    assert_eq!(pick_idle([].into_iter(), 10), None);
    let none: Option<usize> = None;
    assert_eq!(pick_idle([5, 9].into_iter(), 10), none, "short");
    assert_eq!(pick_idle([5, 10, 3].into_iter(), 10), Some(1), "at min");
    assert_eq!(pick_idle([12, 40, 40, 11].into_iter(), 10), Some(1), "ties");
    assert_eq!(pick_idle([0, 0].into_iter(), 0), Some(0));
}
