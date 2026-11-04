use super::*;
use crate::engine::queue::Scheduling;

/// Tickets for jobs of these priorities, in arrival order.
fn tickets(priorities: &[i32]) -> Vec<Ticket> {
    let mut q = WaitQueue::new(SchedulingPolicy::Fcfs);
    for &priority in priorities {
        q.push(
            (),
            Scheduling {
                priority,
                share_key: None,
            },
        );
    }
    std::iter::from_fn(|| q.pop(16, []).map(|(_, t)| t)).collect()
}

/// The index `make_room` would preempt first among `(ticket, n_past)`.
fn first_victim(policy: SchedulingPolicy, seqs: &[(Ticket, usize)]) -> usize {
    (0..seqs.len())
        .max_by_key(|&i| victim_rank(policy, &seqs[i].0, seqs[i].1))
        .unwrap()
}

#[test]
fn headroom_covers_a_verify_step_and_the_drafts_after_it() {
    // Without drafts: the token fed, and one to spare.
    assert_eq!(headroom(0), 2);
    // A verify step writes `1 + k` positions and keeps up to all of them; the next drafts
    // then sit `k` further.
    for k in 1..8 {
        assert!(headroom(k) > 2 * k, "k={k}");
    }
}

#[test]
fn cover_is_the_next_step_capped_at_the_whole_length() {
    let b = KV_BLOCK;
    // A prompt of exactly one block still needs room for the token it samples.
    assert_eq!(cover(b, 0, 100), 2);
    assert_eq!(cover(b - 2, 0, 100), 1);
    assert_eq!(cover(b - 1, 0, 100), 2);
    // Never more than `reserve` would take.
    assert_eq!(cover(10 * b, 0, 4), 4);
    // Speculation's drafts need more room.
    assert_eq!(cover(b - 4, 1, 100), 1);
    assert_eq!(cover(b - 3, 1, 100), 2);
}

#[test]
fn fcfs_preempts_the_least_progress_then_the_latest() {
    let t = tickets(&[0, 0, 0]);
    let p = SchedulingPolicy::Fcfs;
    // Fewest cached tokens first, whatever the arrival.
    assert_eq!(first_victim(p, &[(t[0], 50), (t[1], 900), (t[2], 300)]), 0);
    // A tie goes to the latest arrival.
    assert_eq!(first_victim(p, &[(t[0], 64), (t[1], 64), (t[2], 640)]), 1);
}

#[test]
fn fcfs_ignores_priority_when_preempting() {
    let t = tickets(&[5, -5]);
    let p = SchedulingPolicy::Fcfs;
    assert_eq!(first_victim(p, &[(t[0], 900), (t[1], 100)]), 1);
}

#[test]
fn priority_preempts_the_lowest_priority_first() {
    let t = tickets(&[-1, 3, 0]);
    let p = SchedulingPolicy::Priority;
    // The highest value (least urgent) goes first, even with the most progress.
    assert_eq!(first_victim(p, &[(t[0], 10), (t[1], 5000), (t[2], 20)]), 1);
    // Within a class, the least progress.
    let t = tickets(&[2, 2, -1]);
    assert_eq!(first_victim(p, &[(t[0], 700), (t[1], 300), (t[2], 1)]), 1);
}

#[test]
fn on_demand_falls_back_where_sequences_cannot_leave() {
    let r = KvAdmission::Reserve;
    let o = KvAdmission::OnDemand;
    assert_eq!(resolve(r, 0, 0), r);
    assert_eq!(resolve(o, 0, 0), o);
    assert_eq!(resolve(o, 1, 0), o);
    assert_eq!(resolve(o, 2, 0), r);
    assert_eq!(resolve(o, 0, 8), r);
}

#[test]
fn reserve_is_the_default() {
    assert_eq!(KvAdmission::default(), KvAdmission::Reserve);
}
