use super::*;
use futures::StreamExt;

fn request(text: &str) -> GenerateRequest {
    GenerateRequest {
        model: "mock".into(),
        prompt: Prompt::Text(text.into()),
        sampling: Default::default(),
        max_tokens: 100,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: Default::default(),
        scheduling: Default::default(),
    }
}

#[tokio::test]
async fn stats_count_running_and_waiting() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(20),
        max_batch: 1,
        ..Default::default()
    });
    assert_eq!(engine.stats(), EngineStats::default());
    let mut first = engine.generate(request("a b c d")).unwrap();
    let _second = engine.generate(request("a b c d")).unwrap();
    // Once the first sequence has started, it runs and the second waits behind it.
    while !matches!(first.next().await, Some(TokenEvent::Started { .. })) {}
    let stats = engine.stats();
    assert_eq!((stats.running, stats.waiting), (1, 1));
    while first.next().await.is_some() {}
}

/// Drains a stream to the `Finished` usage.
async fn usage_of(mut s: TokenStream) -> Usage {
    loop {
        match s.next().await {
            Some(TokenEvent::Finished { usage, .. }) => return usage,
            Some(_) => continue,
            None => panic!("stream ended without finishing"),
        }
    }
}

#[tokio::test]
async fn usage_reports_prefill_and_decode_time() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(10),
        ..Default::default()
    });
    let usage = usage_of(engine.generate(request("a b c d")).unwrap()).await;
    assert!(usage.completion_tokens > 1);
    let prefill = usage.prefill_ms.expect("prefill measured");
    let decode = usage.decode_ms.expect("decode measured");
    // One step of prompt processing, the rest decoding.
    assert!(prefill > 0.0, "prefill {prefill}");
    assert!(decode > 0.0, "decode {decode}");
}

#[tokio::test]
async fn usage_without_generated_tokens() {
    let engine = MockEngine::new(MockConfig::default());
    let mut req = request("a b c d");
    req.max_tokens = 0;
    let usage = usage_of(engine.generate(req).unwrap()).await;
    assert_eq!(usage.completion_tokens, 0);
    assert!(usage.prefill_ms.is_some());
    assert_eq!(usage.decode_ms, Some(0.0));
}

fn with_priority(text: &str, priority: i32, key: Option<&str>) -> GenerateRequest {
    GenerateRequest {
        scheduling: Scheduling {
            priority,
            share_key: key.map(super::super::queue::ShareKey::of),
        },
        ..request(text)
    }
}

/// Each request's queue wait, one batch slot, the requests submitted in this order.
async fn queue_waits(policy: SchedulingPolicy, reqs: Vec<GenerateRequest>) -> Vec<f64> {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(5),
        max_batch: 1,
        scheduling: policy,
        ..Default::default()
    });
    // The loop runs only once this task awaits: every request is queued before the first is
    // admitted.
    let streams: Vec<_> = reqs
        .into_iter()
        .map(|r| engine.generate(r).unwrap())
        .collect();
    let usages = futures::future::join_all(streams.into_iter().map(usage_of)).await;
    usages
        .iter()
        .map(|u| u.queue_ms.expect("queue wait measured"))
        .collect()
}

#[tokio::test]
async fn fcfs_admits_in_arrival_order_whatever_the_priority() {
    let waits = queue_waits(
        SchedulingPolicy::Fcfs,
        vec![
            with_priority("a b c", 0, None),
            with_priority("a b c", 5, None),
            with_priority("a b c", -1, None),
        ],
    )
    .await;
    assert!(waits[0] < waits[1] && waits[1] < waits[2], "{waits:?}");
}

#[tokio::test]
async fn priority_admits_the_lowest_value_first() {
    let waits = queue_waits(
        SchedulingPolicy::Priority,
        vec![
            with_priority("a b c", 0, None),
            with_priority("a b c", 5, None),
            with_priority("a b c", -1, None),
        ],
    )
    .await;
    assert!(waits[2] < waits[0] && waits[0] < waits[1], "{waits:?}");
}

#[tokio::test]
async fn fair_share_alternates_keys() {
    // Key A sends three, then key B two: A, B, A, B, A.
    let waits = queue_waits(
        SchedulingPolicy::Fcfs,
        vec![
            with_priority("a b c", 0, Some("A")),
            with_priority("a b c", 0, Some("A")),
            with_priority("a b c", 0, Some("A")),
            with_priority("a b c", 0, Some("B")),
            with_priority("a b c", 0, Some("B")),
        ],
    )
    .await;
    let mut order: Vec<usize> = (0..waits.len()).collect();
    order.sort_by(|&x, &y| waits[x].total_cmp(&waits[y]));
    assert_eq!(order, [0, 3, 1, 4, 2], "{waits:?}");
}

#[tokio::test]
async fn stats_report_waiting_classes_and_keys() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(20),
        max_batch: 1,
        ..Default::default()
    });
    let mut first = engine.generate(with_priority("a b", 0, None)).unwrap();
    let _b = engine.generate(with_priority("a b", 3, Some("X"))).unwrap();
    let _c = engine.generate(with_priority("a b", 0, Some("Y"))).unwrap();
    while !matches!(first.next().await, Some(TokenEvent::Started { .. })) {}
    let stats = engine.stats();
    assert_eq!(stats.waiting, 2);
    assert_eq!(stats.waiting_keys, 2);
    assert_eq!(
        stats.waiting_classes.summary().as_deref(),
        Some("p0:1 p3:1")
    );
    while first.next().await.is_some() {}
}
