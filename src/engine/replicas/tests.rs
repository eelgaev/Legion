use super::*;
use crate::engine::{Prompt, TokenEvent};
use std::sync::atomic::AtomicBool;

#[test]
fn replica_count_parses_auto_and_counts() {
    assert_eq!("auto".parse(), Ok(ReplicaCount::Auto));
    assert_eq!("1".parse(), Ok(ReplicaCount::Fixed(1)));
    assert_eq!(" 2 ".parse(), Ok(ReplicaCount::Fixed(2)));
    for bad in ["0", "", "two", "-1"] {
        assert!(bad.parse::<ReplicaCount>().is_err(), "{bad:?}");
    }
    assert_eq!(ReplicaCount::Auto.to_string(), "auto");
    assert_eq!(ReplicaCount::Fixed(3).to_string(), "3");
}

#[test]
fn resolve_counts_tp_groups() {
    use ReplicaCount::*;
    // The default never asks for more GPUs.
    assert_eq!(Fixed(1).resolve(1, 0), Ok(1));
    assert_eq!(Fixed(1).resolve(2, 0), Ok(1));
    // `auto`: one copy without --tp, else every whole group.
    assert_eq!(Auto.resolve(1, 4), Ok(1));
    assert_eq!(Auto.resolve(2, 4), Ok(2));
    assert_eq!(Auto.resolve(2, 5), Ok(2));
    assert_eq!(Auto.resolve(2, 3), Ok(1));
    assert_eq!(Auto.resolve(4, 4), Ok(1));
    assert_eq!(Auto.resolve(2, 0), Ok(1));
    // A fixed count needs --tp and its GPUs.
    assert_eq!(Fixed(2).resolve(2, 4), Ok(2));
    assert!(Fixed(2).resolve(1, 4).unwrap_err().contains("--tp 2"));
    assert!(Fixed(2).resolve(2, 3).unwrap_err().contains("needs 4 GPUs"));
}

fn cand(assigned: usize, free_kv: u32, cached: usize) -> Candidate {
    Candidate {
        assigned,
        free_kv,
        cached,
    }
}

#[test]
fn choose_takes_the_least_loaded() {
    assert_eq!(choose(&[cand(2, 10, 0), cand(1, 10, 0)], 100, 4), 1);
    // Ties: the most free KV, then the lowest index.
    assert_eq!(choose(&[cand(1, 10, 0), cand(1, 20, 0)], 100, 4), 1);
    assert_eq!(choose(&[cand(1, 20, 0), cand(1, 20, 0)], 100, 4), 0);
    assert_eq!(choose(&[cand(0, 0, 0)], 100, 4), 0);
}

#[test]
fn choose_follows_a_cached_prefix_while_the_replica_has_room() {
    // Replica 0 holds 900 of 1000 tokens: it goes there though busier, while under max_seqs.
    let c = [cand(3, 10, 900), cand(1, 10, 0)];
    assert_eq!(choose(&c, 1000, 4), 0);
    // Full (max_seqs in flight) and busier than the least loaded: by load.
    let c = [cand(4, 10, 900), cand(1, 10, 0)];
    assert_eq!(choose(&c, 1000, 4), 1);
    // Full but no busier: still the hit.
    let c = [cand(4, 10, 900), cand(4, 10, 0)];
    assert_eq!(choose(&c, 1000, 4), 0);
    // The longer hit wins.
    let c = [cand(0, 10, 600), cand(1, 10, 900)];
    assert_eq!(choose(&c, 1000, 4), 1);
}

#[test]
fn choose_ignores_short_or_partial_hits() {
    // Under MIN_HIT_TOKENS.
    let c = [cand(2, 10, MIN_HIT_TOKENS - 1), cand(0, 10, 0)];
    assert_eq!(choose(&c, MIN_HIT_TOKENS, 4), 1);
    // Less than half of the prompt (a shared system prompt).
    let c = [cand(2, 10, 400), cand(0, 10, 0)];
    assert_eq!(choose(&c, 1000, 4), 1);
    // Exactly half counts.
    let c = [cand(2, 10, 500), cand(0, 10, 0)];
    assert_eq!(choose(&c, 1000, 4), 0);
}

fn toks(range: std::ops::Range<u32>) -> Arc<[u32]> {
    range.collect::<Vec<_>>().into()
}

#[test]
fn prefix_index_finds_the_longest_common_prefix() {
    let mut ix = PrefixIndex::default();
    assert_eq!(ix.longest(&[1, 2, 3]), 0);
    ix.record(toks(0..10));
    ix.record(vec![0, 1, 2, 99].into());
    assert_eq!(ix.longest(&[0, 1, 2, 3, 4, 77]), 5);
    assert_eq!(ix.longest(&[0, 1, 2, 99, 5]), 4);
    assert_eq!(ix.longest(&[7]), 0);
    // A prompt shorter than a remembered one matches whole.
    assert_eq!(ix.longest(&[0, 1]), 2);
}

#[test]
fn prefix_index_forgets_the_oldest() {
    let mut ix = PrefixIndex::default();
    for i in 0..=PREFIX_ENTRIES as u32 {
        ix.record(vec![i, i, i].into());
    }
    assert_eq!(ix.prompts.len(), PREFIX_ENTRIES);
    assert_eq!(ix.longest(&[0, 0, 0]), 0);
    assert_eq!(ix.longest(&[1, 1, 1]), 3);
    assert_eq!(ix.tokens, 3 * PREFIX_ENTRIES);

    // Past PREFIX_TOKENS the oldest go, but the newest stays even alone.
    let mut ix = PrefixIndex::default();
    let big = PREFIX_TOKENS as u32 / 4 * 3;
    ix.record(toks(0..big));
    ix.record(toks(1..big));
    assert_eq!(ix.prompts.len(), 1);
    assert_eq!(ix.tokens, big as usize - 1);
    assert_eq!(ix.longest(&[0]), 0);
    assert_eq!(ix.longest(&[1, 2]), 2);
    let mut ix = PrefixIndex::default();
    ix.record(toks(0..PREFIX_TOKENS as u32 + 1));
    assert_eq!(ix.prompts.len(), 1);
}

#[test]
fn merge_stats_adds_counts_and_weights_the_draft_depth() {
    assert_eq!(merge_stats(&[]), EngineStats::default());
    let mut a = EngineStats {
        running: 3,
        decoding: 2,
        prefilling: 1,
        waiting: 1,
        kv_blocks_used: 10,
        kv_blocks_total: 100,
        generated_tokens: 50,
        prompt_tokens: 7,
        preemptions: 1,
        spec_mean_k: 3.0,
        ..Default::default()
    };
    a.rooflines[0].memory_gb_s = 1.5;
    a.spec.drafted = 30;
    a.spec.accepted = 20;
    a.spec.steps = 10;
    a.model_bytes.decode.steps = 5;
    let mut b = EngineStats {
        running: 1,
        decoding: 1,
        waiting: 2,
        kv_blocks_used: 5,
        kv_blocks_total: 80,
        generated_tokens: 8,
        spec_mean_k: 0.0,
        ..Default::default()
    };
    b.rooflines[0].memory_gb_s = 9.0;
    b.spec.drafted = 3;
    b.model_bytes.decode.steps = 2;
    let m = merge_stats(&[a, b]);
    assert_eq!(
        (m.running, m.decoding, m.prefilling, m.waiting),
        (4, 3, 1, 3)
    );
    assert_eq!((m.kv_blocks_used, m.kv_blocks_total), (15, 180));
    assert_eq!(
        (m.generated_tokens, m.prompt_tokens, m.preemptions),
        (58, 7, 1)
    );
    assert_eq!(
        (m.spec.drafted, m.spec.accepted, m.spec.steps),
        (33, 20, 10)
    );
    assert_eq!(m.model_bytes.decode.steps, 7);
    assert!((m.spec_mean_k - 2.0).abs() < 1e-9, "{}", m.spec_mean_k);
    // The rooflines are the first replica's.
    assert_eq!(m.roofline().memory_gb_s, 1.5);
    // One replica: its stats unchanged.
    assert_eq!(merge_stats(&[a]), a);
}

/// A replica that queues every request until `full`, and counts what it took.
#[derive(Default)]
struct Fake {
    full: AtomicBool,
    taken: AtomicUsize,
    free_kv: u32,
}

impl Fake {
    fn with_kv(free_kv: u32) -> Self {
        Self {
            free_kv,
            ..Default::default()
        }
    }
}

impl Engine for Fake {
    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }

    fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, EngineError> {
        unreachable!("the router submits")
    }

    fn stats(&self) -> EngineStats {
        EngineStats {
            kv_blocks_total: 1000,
            kv_blocks_used: 1000 - self.free_kv,
            running: self.taken.load(Ordering::Relaxed) as u32,
            ..Default::default()
        }
    }
}

impl Replica for Fake {
    type Prepared = Vec<u32>;

    fn prepare(&self, req: GenerateRequest) -> Result<Vec<u32>, EngineError> {
        match req.prompt {
            Prompt::Text(t) => Ok(t.bytes().map(u32::from).collect()),
            Prompt::Chat { .. } => Err(EngineError::InvalidRequest("text only".into())),
        }
    }

    fn prompt(p: &Vec<u32>) -> &[u32] {
        p
    }

    fn submit(&self, p: Vec<u32>) -> Result<TokenStream, Refused<Vec<u32>>> {
        if self.full.load(Ordering::Relaxed) {
            return Err(Refused::Full(Box::new(p)));
        }
        self.taken.fetch_add(1, Ordering::Relaxed);
        let started = futures::stream::iter([TokenEvent::Started {
            prompt_tokens: p.len() as u32,
        }]);
        Ok(Box::pin(started.chain(futures::stream::pending())))
    }
}

fn request(text: &str) -> GenerateRequest {
    GenerateRequest {
        model: "fake".into(),
        prompt: Prompt::Text(text.into()),
        sampling: Default::default(),
        max_tokens: 10,
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

fn taken(e: &ReplicaEngine<Fake>) -> Vec<usize> {
    e.members()
        .iter()
        .map(|r| r.taken.load(Ordering::Relaxed))
        .collect()
}

fn assigned(e: &ReplicaEngine<Fake>) -> Vec<u32> {
    e.replica_stats().iter().map(|s| s.assigned).collect()
}

#[test]
fn router_spreads_requests_and_counts_them_until_dropped() {
    // Replica 1 has more free KV: the first request goes there, then they alternate.
    let e = ReplicaEngine::new(vec![Fake::with_kv(100), Fake::with_kv(200)], 4, false);
    let first = e.generate(request("a")).unwrap();
    assert_eq!(taken(&e), [0, 1]);
    let streams: Vec<_> = (0..3).map(|_| e.generate(request("b")).unwrap()).collect();
    assert_eq!(taken(&e), [2, 2]);
    assert_eq!(assigned(&e), [2, 2]);
    // A finished (dropped) stream frees its replica's place: the next one goes there.
    drop(first);
    assert_eq!(assigned(&e), [2, 1]);
    let _next = e.generate(request("c")).unwrap();
    assert_eq!(taken(&e), [2, 3]);
    drop(streams);
    assert_eq!(assigned(&e), [0, 1]);
    // The merged stats are the replicas' sums.
    assert_eq!(e.stats().running, 5);
    assert_eq!(e.stats().kv_blocks_total, 2000);
}

#[tokio::test]
async fn router_streams_the_replicas_events() {
    let e = ReplicaEngine::new(vec![Fake::default(), Fake::default()], 4, false);
    let mut s = e.generate(request("abc")).unwrap();
    assert!(matches!(
        s.next().await,
        Some(TokenEvent::Started { prompt_tokens: 3 })
    ));
}

#[test]
fn router_tries_the_next_replica_when_one_is_full() {
    let e = ReplicaEngine::new(vec![Fake::default(), Fake::default()], 4, false);
    e.members()[0].full.store(true, Ordering::Relaxed);
    let _a = e.generate(request("a")).unwrap();
    let _b = e.generate(request("b")).unwrap();
    assert_eq!(taken(&e), [0, 2]);
    // The refused tries left no count behind.
    assert_eq!(assigned(&e), [0, 2]);
    e.members()[1].full.store(true, Ordering::Relaxed);
    assert!(matches!(
        e.generate(request("c")),
        Err(EngineError::Overloaded)
    ));
    assert_eq!(assigned(&e), [0, 2]);
}

#[test]
fn router_passes_preparation_errors_through() {
    let e = ReplicaEngine::new(vec![Fake::default(), Fake::default()], 4, false);
    let mut req = request("a");
    req.prompt = Prompt::Chat {
        system: None,
        messages: vec![],
        tools: vec![],
    };
    assert!(matches!(
        e.generate(req),
        Err(EngineError::InvalidRequest(_))
    ));
    assert_eq!(assigned(&e), [0, 0]);
}

#[test]
fn router_sends_a_cached_prefix_back_to_its_replica() {
    let long = "x".repeat(2 * MIN_HIT_TOKENS);
    let e = ReplicaEngine::new(vec![Fake::default(), Fake::default()], 4, true);
    let _first = e.generate(request(&long)).unwrap();
    assert_eq!(taken(&e), [1, 0]);
    // The next turn of that conversation goes to replica 0 though it is busier...
    let _turn = e.generate(request(&format!("{long}yy"))).unwrap();
    assert_eq!(taken(&e), [2, 0]);
    // ... an unrelated prompt by load.
    let _other = e
        .generate(request(&"z".repeat(2 * MIN_HIT_TOKENS)))
        .unwrap();
    assert_eq!(taken(&e), [2, 1]);

    // Without prefix caching the same prompts go by load alone.
    let e = ReplicaEngine::new(vec![Fake::default(), Fake::default()], 4, false);
    let _first = e.generate(request(&long)).unwrap();
    let _turn = e.generate(request(&format!("{long}yy"))).unwrap();
    assert_eq!(taken(&e), [1, 1]);
}
