use super::*;

#[test]
fn keeps_several_checkpoints_per_ring() {
    let mut c = RingCheckpoints::new(4);
    let a = c.store(0, 1024).unwrap();
    let b = c.store(0, 2048).unwrap();
    let d = c.store(1, 512).unwrap();
    assert_eq!(c.of(0), vec![(1024, a), (2048, b)]);
    assert_eq!(c.of(1), vec![(512, d)]);
    assert_eq!(c.latest(0), Some(2048));
    // The same point again reuses its slot.
    assert_eq!(c.store(0, 2048), Some(b));
}

#[test]
fn best_resumes_the_most_tokens_where_the_copy_holds_the_window() {
    let mut c = RingCheckpoints::new(8);
    for at in [1000, 2000, 3000] {
        c.store(0, at);
    }
    // The window before m is in a copy saved at `at` when m >= at - 500.
    let holds = |m: usize, at: usize| m + 500 >= at;
    // The prompt shares 2500 tokens: the copy at 2000 resumes 2000; the one at 3000 holds
    // nothing before 2500 (2500 + 500 >= 3000 does).
    assert_eq!(
        c.best(0, 2500, holds).map(|b| (b.0, b.1)),
        Some((2500, 3000))
    );
    assert_eq!(
        c.best(0, 2400, holds).map(|b| (b.0, b.1)),
        Some((2000, 2000))
    );
    // Past every copy: the latest.
    assert_eq!(
        c.best(0, 9000, holds).map(|b| (b.0, b.1)),
        Some((3000, 3000))
    );
    assert_eq!(c.best(0, 400, holds), None);
    assert_eq!(c.best(1, 5000, holds), None);
}

#[test]
fn truncate_and_clear_drop_a_rings_checkpoints() {
    let mut c = RingCheckpoints::new(8);
    for at in [1000, 2000, 3000] {
        c.store(0, at);
    }
    c.store(1, 1000);
    c.truncate(0, 2000);
    assert_eq!(
        c.of(0).iter().map(|x| x.0).collect::<Vec<_>>(),
        vec![1000, 2000]
    );
    c.clear(0);
    assert!(c.of(0).is_empty());
    assert_eq!(c.of(1).len(), 1);
}

#[test]
fn a_full_store_keeps_each_rings_latest() {
    let mut c = RingCheckpoints::new(3);
    let a1 = c.store(0, 1000).unwrap();
    c.store(0, 2000).unwrap();
    let b = c.store(1, 500).unwrap();
    // Ring 2 takes ring 0's older copy, not either ring's latest.
    assert_eq!(c.store(2, 700), Some(a1));
    assert_eq!(c.of(0).len(), 1);
    assert_eq!(c.of(1), vec![(500, b)]);
    // Every slot is some ring's latest: ring 0's next copy replaces its own.
    let s = c.store(0, 3000).unwrap();
    assert_eq!(c.of(0), vec![(3000, s)]);
    assert_eq!(c.of(1).len(), 1);
    assert_eq!(c.of(2).len(), 1);
}

#[test]
fn a_full_store_takes_the_least_recently_used() {
    let mut c = RingCheckpoints::new(3);
    let a = c.store(0, 1000).unwrap();
    let b = c.store(0, 2000).unwrap();
    c.store(0, 3000).unwrap();
    c.touch(a);
    // 2000 is now the least recently used non-latest.
    assert_eq!(c.store(1, 100), Some(b));
}

#[test]
fn no_slots_no_checkpoints() {
    let mut c = RingCheckpoints::new(0);
    assert!(c.is_empty());
    assert_eq!(c.store(0, 10), None);
}

#[test]
fn due_over_the_last_span_every_so_many_tokens() {
    // A 9000-token prompt, span 4096, every 1024.
    assert!(!due(4096, 9000, None, 1024, 4096));
    assert!(due(5120, 9000, None, 1024, 4096));
    assert!(!due(5632, 9000, Some(5120), 1024, 4096));
    assert!(due(6144, 9000, Some(5120), 1024, 4096));
    // The end always saves on its own path.
    assert!(!due(9000, 9000, Some(8192), 1024, 4096));
    assert!(!due(6144, 9000, None, 0, 4096));
}
