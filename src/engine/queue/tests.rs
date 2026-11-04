use super::*;

fn sched(priority: i32, key: Option<&str>) -> Scheduling {
    Scheduling {
        priority,
        share_key: key.map(ShareKey::of),
    }
}

/// Pops every job with nothing running, in the order the queue gives them.
fn drain(q: &mut WaitQueue<&'static str>) -> Vec<&'static str> {
    std::iter::from_fn(|| q.pop(16, []).map(|(j, _)| j)).collect()
}

#[test]
fn fcfs_keeps_arrival_order_and_ignores_priority() {
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    q.push("a", sched(5, None));
    q.push("b", sched(-3, None));
    q.push("c", sched(0, None));
    assert_eq!(drain(&mut q), ["a", "b", "c"]);
}

#[test]
fn fcfs_put_back_returns_to_the_front() {
    // Today's loop: `pop_front`, and `push_front` when the KV pool is short.
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    for j in ["a", "b", "c"] {
        q.push(j, Scheduling::default());
    }
    let (job, ticket) = q.pop(16, []).unwrap();
    assert_eq!(job, "a");
    q.push("d", Scheduling::default());
    q.put_back(job, ticket);
    assert_eq!(drain(&mut q), ["a", "b", "c", "d"]);
}

#[test]
fn priority_serves_lowest_value_first_then_arrival() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("low1", sched(10, None));
    q.push("high1", sched(-1, None));
    q.push("mid", sched(0, None));
    q.push("low2", sched(10, None));
    q.push("high2", sched(-1, None));
    assert_eq!(drain(&mut q), ["high1", "high2", "mid", "low1", "low2"]);
}

#[test]
fn priority_ties_keep_arrival_order() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    for j in ["a", "b", "c", "d"] {
        q.push(j, sched(3, None));
    }
    assert_eq!(drain(&mut q), ["a", "b", "c", "d"]);
}

#[test]
fn priority_put_back_keeps_the_place() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("low", sched(1, None));
    q.push("high", sched(0, None));
    let (job, ticket) = q.pop(16, []).unwrap();
    assert_eq!(job, "high");
    assert_eq!(ticket.order(), (0, 1));
    q.put_back(job, ticket);
    assert_eq!(drain(&mut q), ["high", "low"]);
}

#[test]
fn cancelled_jobs_leave_the_queue_and_the_counts() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("a", sched(0, Some("k1")));
    q.push("b", sched(2, Some("k2")));
    q.push("c", sched(2, Some("k1")));
    q.retain(|j| *j != "b");
    assert_eq!(q.len(), 2);
    assert_eq!(q.waiting_keys(), 1);
    assert_eq!(
        q.classes().iter().collect::<Vec<_>>(),
        [("0".to_string(), 1), ("2".to_string(), 1)]
    );
    q.retain(|_| false);
    assert!(q.is_empty());
    assert_eq!(q.waiting_keys(), 0);
    assert_eq!(q.classes(), WaitingClasses::default());
}

#[test]
fn fair_share_alternates_between_keys() {
    // One key floods, another sends a few later: the slots alternate between them.
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    for j in ["a1", "a2", "a3", "a4"] {
        q.push(j, sched(0, Some("A")));
    }
    for j in ["b1", "b2"] {
        q.push(j, sched(0, Some("B")));
    }
    let mut running = Vec::new();
    let mut order = Vec::new();
    while let Some((job, t)) = q.pop(16, running.clone()) {
        running.push(t.scheduling().share_key);
        order.push(job);
    }
    assert_eq!(order, ["a1", "b1", "a2", "b2", "a3", "a4"]);
}

#[test]
fn round_robin_when_keys_hold_the_same() {
    // One slot: each admission finds nothing running, so the turn decides.
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    for j in ["a1", "a2", "a3"] {
        q.push(j, sched(0, Some("A")));
    }
    for j in ["b1", "b2"] {
        q.push(j, sched(0, Some("B")));
    }
    assert_eq!(drain(&mut q), ["a1", "b1", "a2", "b2", "a3"]);
}

#[test]
fn fair_share_goes_to_the_key_holding_fewest() {
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    q.push("a", sched(0, Some("A")));
    q.push("b", sched(0, Some("B")));
    let running = [Some(ShareKey::of("A")), Some(ShareKey::of("A"))];
    assert_eq!(q.pop(16, running).unwrap().0, "b");
}

#[test]
fn share_cap_holds_a_key_while_others_wait() {
    // Priority first, but key A holds its share (4 of 8, two keys): B's lower-priority request
    // gets the slot.
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("a", sched(0, Some("A")));
    q.push("b", sched(9, Some("B")));
    let a = Some(ShareKey::of("A"));
    assert_eq!(q.pop(8, [a; 4]).unwrap().0, "b");
    // Nobody else waits: A may take every slot.
    assert_eq!(q.pop(8, [a; 7]).unwrap().0, "a");
}

#[test]
fn share_counts_running_keys_that_do_not_wait() {
    // Keys A and B wait, C only runs: three keys, a share of ceil(9 / 3) = 3.
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("a", sched(0, Some("A")));
    q.push("b", sched(5, Some("B")));
    let (a, b, c) = (
        Some(ShareKey::of("A")),
        Some(ShareKey::of("B")),
        Some(ShareKey::of("C")),
    );
    let running = [a, a, a, b, b, c];
    assert_eq!(q.pop(9, running).unwrap().0, "b");
}

#[test]
fn share_cap_never_leaves_a_slot_idle() {
    // A and B both hold their share (16 / 3 keys -> 6) and C holds 3: a free slot still goes
    // to the best waiting request.
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("a", sched(0, Some("A")));
    q.push("b", sched(1, Some("B")));
    let (a, b, c) = (
        Some(ShareKey::of("A")),
        Some(ShareKey::of("B")),
        Some(ShareKey::of("C")),
    );
    let running: Vec<_> = [a; 6].into_iter().chain([b; 6]).chain([c; 3]).collect();
    assert_eq!(q.pop(16, running).unwrap().0, "a");
}

#[test]
fn fair_share_within_the_best_priority() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("a-low", sched(5, Some("A")));
    q.push("a-high", sched(0, Some("A")));
    q.push("b-high", sched(0, Some("B")));
    let a = Some(ShareKey::of("A"));
    // A holds one, B none: B's high-priority request first, then A's, then the low one.
    let mut running = vec![a];
    let mut order = Vec::new();
    while let Some((job, t)) = q.pop(16, running.clone()) {
        running.push(t.scheduling().share_key);
        order.push(job);
    }
    assert_eq!(order, ["b-high", "a-high", "a-low"]);
}

#[test]
fn requests_without_a_key_count_as_one_key() {
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    q.push("n1", sched(0, None));
    q.push("n2", sched(0, None));
    q.push("k1", sched(0, Some("K")));
    assert_eq!(q.waiting_keys(), 2);
    let mut running = Vec::new();
    let mut order = Vec::new();
    while let Some((job, t)) = q.pop(16, running.clone()) {
        running.push(t.scheduling().share_key);
        order.push(job);
    }
    assert_eq!(order, ["n1", "k1", "n2"]);
}

#[test]
fn classes_merge_past_the_limit() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    for p in 0..10 {
        q.push((), sched(p, None));
    }
    q.push((), sched(-2, None));
    let classes: Vec<_> = q.classes().iter().collect();
    assert_eq!(classes.len(), PRIORITY_CLASSES);
    assert_eq!(classes[0], ("-2".to_string(), 1));
    assert_eq!(classes[PRIORITY_CLASSES - 1], (">=6".to_string(), 4));
}

#[test]
fn summary_only_when_priorities_are_used() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    assert_eq!(q.classes().summary(), None);
    q.push((), sched(0, None));
    assert_eq!(q.classes().summary(), None);
    q.push((), sched(-1, None));
    q.push((), sched(0, None));
    assert_eq!(q.classes().summary().as_deref(), Some("p-1:1 p0:2"));
}

#[test]
fn classes_of_two_queues_merge_by_priority() {
    let mut a = WaitQueue::new(SchedulingPolicy::Priority);
    let mut b = WaitQueue::new(SchedulingPolicy::Priority);
    a.push((), sched(0, None));
    a.push((), sched(-1, None));
    b.push((), sched(0, None));
    b.push((), sched(3, None));
    let m = a.classes().merge(&b.classes());
    assert_eq!(m.summary().as_deref(), Some("p-1:1 p0:2 p3:1"));
    assert_eq!(
        WaitingClasses::default().merge(&WaitingClasses::default()),
        WaitingClasses::default()
    );
    // Past the limit the last class takes the rest, as one queue's would.
    let mut c = WaitQueue::new(SchedulingPolicy::Priority);
    for p in 0..=PRIORITY_CLASSES as i32 {
        c.push((), sched(p, None));
    }
    let m = c.classes().merge(&b.classes());
    let classes: Vec<_> = m.iter().collect();
    assert_eq!(classes.len(), PRIORITY_CLASSES);
    assert_eq!(classes[0], ("0".to_string(), 2));
    assert_eq!(classes[3], ("3".to_string(), 2));
    assert_eq!(classes[PRIORITY_CLASSES - 1], (">=7".to_string(), 2));
}

#[test]
fn share_key_display_is_short_and_stable() {
    let k = ShareKey::of("sk-secret-key");
    let shown = k.to_string();
    assert_eq!(shown.len(), 8);
    assert!(!shown.contains("secret"));
    assert_eq!(shown, ShareKey::of("sk-secret-key").to_string());
    assert_ne!(k, ShareKey::of("sk-other-key"));
}

#[test]
fn requeued_jobs_go_to_the_front_under_fcfs() {
    // #627: a preempted sequence is admitted before every job that waited behind it.
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    for j in ["a", "b"] {
        q.push(j, Scheduling::default());
    }
    let (a, ta) = q.pop(16, []).unwrap();
    let (b, tb) = q.pop(16, []).unwrap();
    q.push("c", Scheduling::default());
    q.push("d", Scheduling::default());
    // Preempted in the opposite order: requeued jobs keep their arrival order.
    q.requeue(b, tb);
    q.requeue(a, ta);
    let (job, ticket) = q.pop(16, []).unwrap();
    assert_eq!(job, "a");
    assert!(ticket.requeued());
    // Admission had to wait: back to the front, still requeued.
    q.put_back(job, ticket);
    assert_eq!(drain(&mut q), ["a", "b", "c", "d"]);
}

#[test]
fn requeued_jobs_lead_their_priority_class_only() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    q.push("victim", sched(5, None));
    let (victim, t) = q.pop(16, []).unwrap();
    q.push("urgent", sched(0, None));
    q.push("same", sched(5, None));
    q.push("lazy", sched(9, None));
    q.requeue(victim, t);
    assert_eq!(drain(&mut q), ["urgent", "victim", "same", "lazy"]);
}

#[test]
fn requeued_jobs_lead_under_fair_share_within_the_share() {
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    q.push("a1", sched(0, Some("A")));
    let (a1, t) = q.pop(16, []).unwrap();
    q.push("b1", sched(0, Some("B")));
    q.push("a2", sched(0, Some("A")));
    q.requeue(a1, t);
    // B holds nothing and A one, but the requeued job leads its class.
    let a = Some(ShareKey::of("A"));
    assert_eq!(q.pop(16, [a]).unwrap().0, "a1");
}
