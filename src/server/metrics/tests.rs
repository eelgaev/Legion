use super::*;
use crate::engine::queue::{Scheduling, SchedulingPolicy, ShareKey, WaitQueue};

#[test]
fn speculation_gauges() {
    let m = Metrics::new();
    let body = m.render(
        1,
        EngineStats {
            spec: crate::spec::SpecCounters {
                drafted: 40,
                accepted: 30,
                steps: 10,
            },
            spec_mean_k: 2.5,
            ..Default::default()
        },
    );
    for line in [
        "legion_engine_spec_draft_tokens 40",
        "legion_engine_spec_accepted_tokens 30",
        "legion_engine_spec_verify_steps 10",
        "legion_engine_spec_acceptance_rate 0.75",
        "legion_engine_spec_mean_accepted_length 4.0",
        "legion_engine_spec_mean_k 2.5",
    ] {
        assert!(body.lines().any(|l| l == line), "{line}\n{body}");
    }
}

#[test]
fn waiting_by_priority_and_keys() {
    let mut q = WaitQueue::new(SchedulingPolicy::Priority);
    for (priority, key) in [(10, "a"), (10, "a"), (-1, "b")] {
        let share_key = Some(ShareKey::of(key));
        q.push(
            (),
            Scheduling {
                priority,
                share_key,
            },
        );
    }
    let m = Metrics::new();
    let stats = |q: &WaitQueue<()>| EngineStats {
        waiting: q.len() as u32,
        waiting_classes: q.classes(),
        waiting_keys: q.waiting_keys(),
        ..Default::default()
    };
    let body = m.render(1, stats(&q));
    for line in [
        "legion_engine_waiting 3",
        "legion_engine_waiting_by_priority{priority=\"-1\"} 1",
        "legion_engine_waiting_by_priority{priority=\"10\"} 2",
        "legion_engine_waiting_keys 2",
    ] {
        assert!(body.lines().any(|l| l == line), "{line}\n{body}");
    }
    // A drained class drops out of the next scrape.
    q.retain(|_| false);
    let body = m.render(1, stats(&q));
    assert!(
        !body
            .lines()
            .any(|l| l.starts_with("legion_engine_waiting_by_priority{")),
        "{body}"
    );
    assert!(
        body.lines().any(|l| l == "legion_engine_waiting_keys 0"),
        "{body}"
    );
}

#[test]
fn replica_gauges_only_with_replicas() {
    use crate::engine::replicas::ReplicaStats;
    let m = Metrics::new();
    let body = m.render(0, EngineStats::default());
    assert!(!body.contains("legion_engine_replica_requests{"), "{body}");
    let replica = |assigned, running, waiting, used| ReplicaStats {
        assigned,
        stats: EngineStats {
            running,
            waiting,
            kv_blocks_used: used,
            kv_blocks_total: 100,
            ..Default::default()
        },
    };
    m.set_replicas(&[replica(5, 4, 1, 60), replica(2, 2, 0, 10)]);
    let body = m.render(7, EngineStats::default());
    for line in [
        "legion_engine_replica_requests{replica=\"0\"} 5",
        "legion_engine_replica_requests{replica=\"1\"} 2",
        "legion_engine_replica_running{replica=\"0\"} 4",
        "legion_engine_replica_waiting{replica=\"0\"} 1",
        "legion_engine_replica_kv_blocks_used{replica=\"1\"} 10",
        "legion_engine_replica_kv_blocks_total{replica=\"1\"} 100",
    ] {
        assert!(body.lines().any(|l| l == line), "{line}\n{body}");
    }
}
