//! Prometheus metrics for `GET /metrics`.
//!
//! One [`Metrics`] registry per [`AppState`](super::AppState). HTTP request counts are recorded
//! by `middleware::metrics`; token counts and latencies are recorded by [`MeteredEngine`], which
//! wraps the engine's token streams, so every front end (OpenAI, Anthropic, responses) is
//! measured the same way without touching its route. Labels stay low-cardinality: the matched
//! route pattern and the status class, never a model name, request id or path parameter.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use futures::StreamExt;
use prometheus_client::encoding::{EncodeLabelSet, text::encode};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;

use crate::engine::{
    Engine, EngineError, EngineStats, GenerateRequest, ModelInfo, TokenEvent, TokenStream,
};

/// The kind of step model bytes are counted for (#391): `prefill`, `decode` (decode and verify
/// steps) or `draft` (MTP head passes).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct PhaseLabels {
    phase: &'static str,
}

type F64Gauge = Gauge<f64, AtomicU64>;

/// A request priority class of the waiting queue (#625): the value, or `>=N` for the merged
/// tail when more classes wait than are tracked.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct PriorityLabels {
    priority: String,
}

/// A replica of the model (`--tp-replicas`, #679): its index, in GPU order.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ReplicaLabels {
    replica: String,
}

mod bandwidth;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RequestLabels {
    route: String,
    status: &'static str,
}

/// 1 ms ... 65.5 s in powers of two, covering inter-token gaps up to long generations.
fn latency_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.001, 2.0, 17))
}

/// All metrics and the registry that renders them.
pub struct Metrics {
    registry: Registry,
    requests: Family<RequestLabels, Counter>,
    prompt_tokens: Counter,
    completion_tokens: Counter,
    ttft: Histogram,
    inter_token: Histogram,
    e2e: Histogram,
    in_flight: Gauge,
    running: Gauge,
    waiting: Gauge,
    waiting_by_priority: Family<PriorityLabels, Gauge>,
    waiting_keys: Gauge,
    kv_blocks_used: Gauge,
    kv_blocks_total: Gauge,
    prefix_cache_hit_tokens: Gauge,
    kv_preemptions: Gauge,
    spec_draft_tokens: Gauge,
    spec_accepted_tokens: Gauge,
    spec_verify_steps: Gauge,
    spec_acceptance_rate: Gauge<f64, AtomicU64>,
    spec_mean_accepted_length: Gauge<f64, AtomicU64>,
    spec_mean_k: Gauge<f64, AtomicU64>,
    ngram_rows: Gauge,
    ngram_cache_hits: Gauge,
    ngram_cache_misses: Gauge,
    ngram_cache_hit_rate: F64Gauge,
    ngram_hot_rows: Gauge,
    model_bytes: Family<PhaseLabels, F64Gauge>,
    model_seconds: Family<PhaseLabels, F64Gauge>,
    model_gb_s: Family<PhaseLabels, F64Gauge>,
    bandwidth: bandwidth::BandwidthMetrics,
    replica_assigned: Family<ReplicaLabels, Gauge>,
    replica_running: Family<ReplicaLabels, Gauge>,
    replica_waiting: Family<ReplicaLabels, Gauge>,
    replica_kv_blocks_used: Family<ReplicaLabels, Gauge>,
    replica_kv_blocks_total: Family<ReplicaLabels, Gauge>,
}

impl Metrics {
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let requests = Family::<RequestLabels, Counter>::default();
        registry.register(
            "legion_http_requests",
            "HTTP requests by matched route and status class",
            requests.clone(),
        );
        let prompt_tokens = Counter::default();
        registry.register(
            "legion_prompt_tokens",
            "Prompt tokens processed",
            prompt_tokens.clone(),
        );
        let completion_tokens = Counter::default();
        registry.register(
            "legion_completion_tokens",
            "Completion tokens generated",
            completion_tokens.clone(),
        );
        let ttft = latency_histogram();
        registry.register_with_unit(
            "legion_time_to_first_token",
            "Time from submitting a request to its first text or reasoning event",
            prometheus_client::registry::Unit::Seconds,
            ttft.clone(),
        );
        let inter_token = latency_histogram();
        registry.register_with_unit(
            "legion_inter_token_latency",
            "Gap between consecutive text or reasoning events of one request",
            prometheus_client::registry::Unit::Seconds,
            inter_token.clone(),
        );
        let e2e = latency_histogram();
        registry.register_with_unit(
            "legion_request_latency",
            "Time from submitting a request to the end of its token stream",
            prometheus_client::registry::Unit::Seconds,
            e2e.clone(),
        );
        let gauge = |registry: &mut Registry, name: &str, help: &str| {
            let g = Gauge::default();
            registry.register(name, help, g.clone());
            g
        };
        let in_flight = gauge(
            &mut registry,
            "legion_requests_in_flight",
            "HTTP requests being served, not counting this scrape",
        );
        let running = gauge(
            &mut registry,
            "legion_engine_running",
            "Sequences the engine is decoding",
        );
        let waiting = gauge(
            &mut registry,
            "legion_engine_waiting",
            "Sequences queued in the engine",
        );
        let waiting_by_priority = Family::<PriorityLabels, Gauge>::default();
        registry.register(
            "legion_engine_waiting_by_priority",
            "Sequences queued in the engine, by request priority (lower is admitted first under --scheduling-policy priority; the last class may be `>=N`)",
            waiting_by_priority.clone(),
        );
        let waiting_keys = gauge(
            &mut registry,
            "legion_engine_waiting_keys",
            "Distinct --fair-share keys with queued requests (requests without a key count as one)",
        );
        let kv_blocks_used = gauge(
            &mut registry,
            "legion_engine_kv_blocks_used",
            "KV cache blocks in use",
        );
        let kv_blocks_total = gauge(
            &mut registry,
            "legion_engine_kv_blocks_total",
            "KV cache blocks available",
        );
        let prefix_cache_hit_tokens = gauge(
            &mut registry,
            "legion_engine_prefix_cache_hit_tokens",
            "Prompt tokens served from the prefix cache since start",
        );
        let kv_preemptions = gauge(
            &mut registry,
            "legion_engine_kv_preemptions",
            "Sequences preempted for KV cache blocks since start (--kv-admission on-demand)",
        );
        let spec_draft_tokens = gauge(
            &mut registry,
            "legion_engine_spec_draft_tokens",
            "Speculative draft tokens verified since start",
        );
        let spec_accepted_tokens = gauge(
            &mut registry,
            "legion_engine_spec_accepted_tokens",
            "Speculative draft tokens accepted since start",
        );
        let spec_verify_steps = gauge(
            &mut registry,
            "legion_engine_spec_verify_steps",
            "Sequence steps that verified at least one draft, since start",
        );
        let spec_acceptance_rate = Gauge::<f64, AtomicU64>::default();
        registry.register(
            "legion_engine_spec_acceptance_rate",
            "Accepted / verified draft tokens since start (0 before any draft)",
            spec_acceptance_rate.clone(),
        );
        let spec_mean_accepted_length = Gauge::<f64, AtomicU64>::default();
        registry.register(
            "legion_engine_spec_mean_accepted_length",
            "Tokens appended per verify step, the bonus token included (0 before any)",
            spec_mean_accepted_length.clone(),
        );
        let spec_mean_k = Gauge::<f64, AtomicU64>::default();
        registry.register(
            "legion_engine_spec_mean_k",
            "Mean selected draft depth of decoding sequences in the last drafting step",
            spec_mean_k.clone(),
        );
        // The n-gram (or Engram) table's RAM row cache when its rows come from the SSD.
        let ngram_rows = gauge(
            &mut registry,
            "legion_engine_ngram_rows",
            "N-gram table rows asked of the SSD tier's RAM row cache since start (fetches and prefetches; 0 for the RAM tiers)",
        );
        let ngram_cache_hits = gauge(
            &mut registry,
            "legion_engine_ngram_cache_hits",
            "Of legion_engine_ngram_rows, rows the RAM row cache held",
        );
        let ngram_cache_misses = gauge(
            &mut registry,
            "legion_engine_ngram_cache_misses",
            "Of legion_engine_ngram_rows, rows read from the SSD",
        );
        let ngram_cache_hit_rate = F64Gauge::default();
        registry.register(
            "legion_engine_ngram_cache_hit_rate",
            "legion_engine_ngram_cache_hits over legion_engine_ngram_rows since start (0 before any)",
            ngram_cache_hit_rate.clone(),
        );
        let ngram_hot_rows = gauge(
            &mut registry,
            "legion_engine_ngram_hot_rows",
            "N-gram table rows the RAM budget's hot set pins now (--ngram-ram SIZE)",
        );
        // Model bytes (#391): what the steps must read of the weights, the routed experts and
        // the KV cache, not measured traffic.
        let model_bytes = Family::<PhaseLabels, F64Gauge>::default();
        registry.register(
            "legion_engine_model_bytes",
            "Bytes the steps had to read since start (weights, routed experts, KV; required traffic, not measured), by phase",
            model_bytes.clone(),
        );
        let model_seconds = Family::<PhaseLabels, F64Gauge>::default();
        registry.register(
            "legion_engine_model_seconds",
            "Wall time of the steps counted in legion_engine_model_bytes, by phase",
            model_seconds.clone(),
        );
        let model_gb_s = Family::<PhaseLabels, F64Gauge>::default();
        registry.register(
            "legion_engine_model_gb_per_second",
            "legion_engine_model_bytes over legion_engine_model_seconds, in GB/s, by phase",
            model_gb_s.clone(),
        );
        let bandwidth = bandwidth::BandwidthMetrics::new(&mut registry);
        let family = |registry: &mut Registry, name: &str, help: &str| {
            let f = Family::<ReplicaLabels, Gauge>::default();
            registry.register(name, help, f.clone());
            f
        };
        let replica_assigned = family(
            &mut registry,
            "legion_engine_replica_requests",
            "Requests routed to each replica (--tp-replicas) and not finished",
        );
        let replica_running = family(
            &mut registry,
            "legion_engine_replica_running",
            "Sequences each replica (--tp-replicas) is decoding",
        );
        let replica_waiting = family(
            &mut registry,
            "legion_engine_replica_waiting",
            "Sequences queued in each replica (--tp-replicas)",
        );
        let replica_kv_blocks_used = family(
            &mut registry,
            "legion_engine_replica_kv_blocks_used",
            "KV cache blocks in use, per replica (--tp-replicas)",
        );
        let replica_kv_blocks_total = family(
            &mut registry,
            "legion_engine_replica_kv_blocks_total",
            "KV cache blocks available, per replica (--tp-replicas)",
        );
        Self {
            replica_assigned,
            replica_running,
            replica_waiting,
            replica_kv_blocks_used,
            replica_kv_blocks_total,
            model_bytes,
            model_seconds,
            model_gb_s,
            bandwidth,
            spec_mean_k,
            registry,
            requests,
            prompt_tokens,
            completion_tokens,
            ttft,
            inter_token,
            e2e,
            in_flight,
            running,
            waiting,
            waiting_by_priority,
            waiting_keys,
            kv_blocks_used,
            kv_blocks_total,
            prefix_cache_hit_tokens,
            kv_preemptions,
            spec_draft_tokens,
            spec_accepted_tokens,
            spec_verify_steps,
            spec_acceptance_rate,
            spec_mean_accepted_length,
            ngram_rows,
            ngram_cache_hits,
            ngram_cache_misses,
            ngram_cache_hit_rate,
            ngram_hot_rows,
        }
    }

    /// Exports the GPUs' hardware counters (`legion serve --hw-counters`, #391).
    pub fn add_hw_counters(&mut self, counters: Arc<crate::hw::counters::HwCounters>) {
        self.bandwidth.add_hw(&mut self.registry, counters);
    }

    /// Refreshes the per-replica gauges (`--tp-replicas`); a single engine has none.
    pub fn set_replicas(&self, replicas: &[crate::engine::replicas::ReplicaStats]) {
        for (i, r) in replicas.iter().enumerate() {
            let l = ReplicaLabels {
                replica: i.to_string(),
            };
            self.replica_assigned
                .get_or_create(&l)
                .set(r.assigned.into());
            self.replica_running
                .get_or_create(&l)
                .set(r.stats.running.into());
            self.replica_waiting
                .get_or_create(&l)
                .set(r.stats.waiting.into());
            self.replica_kv_blocks_used
                .get_or_create(&l)
                .set(r.stats.kv_blocks_used.into());
            self.replica_kv_blocks_total
                .get_or_create(&l)
                .set(r.stats.kv_blocks_total.into());
        }
    }

    /// Counts one finished HTTP request.
    pub fn record_request(&self, route: &str, status: u16) {
        let status = match status {
            100..=199 => "1xx",
            200..=299 => "2xx",
            300..=399 => "3xx",
            400..=499 => "4xx",
            _ => "5xx",
        };
        self.requests
            .get_or_create(&RequestLabels {
                route: route.to_string(),
                status,
            })
            .inc();
    }

    /// Renders the Prometheus text format, with the point-in-time gauges refreshed first.
    /// `in_flight` includes the scrape itself, which is not reported.
    pub fn render(&self, in_flight: usize, stats: EngineStats) -> String {
        self.in_flight.set(in_flight.saturating_sub(1) as i64);
        self.running.set(stats.running.into());
        self.waiting.set(stats.waiting.into());
        // Only the classes waiting now: a drained class drops out instead of reading 0 forever.
        self.waiting_by_priority.clear();
        for (priority, n) in stats.waiting_classes.iter() {
            self.waiting_by_priority
                .get_or_create(&PriorityLabels { priority })
                .set(n.into());
        }
        self.waiting_keys.set(stats.waiting_keys.into());
        self.kv_blocks_used.set(stats.kv_blocks_used.into());
        self.kv_blocks_total.set(stats.kv_blocks_total.into());
        self.prefix_cache_hit_tokens
            .set(stats.prefix_cache_hit_tokens as i64);
        self.kv_preemptions.set(stats.preemptions as i64);
        self.spec_mean_k.set(stats.spec_mean_k);
        let spec = stats.spec;
        self.spec_draft_tokens.set(spec.drafted as i64);
        self.spec_accepted_tokens.set(spec.accepted as i64);
        self.spec_verify_steps.set(spec.steps as i64);
        self.spec_acceptance_rate
            .set(spec.acceptance_rate().unwrap_or(0.0));
        self.spec_mean_accepted_length
            .set(spec.mean_accepted_length().unwrap_or(0.0));
        self.ngram_rows.set(stats.ngram_rows as i64);
        self.ngram_cache_hits.set(stats.ngram_cache_hits as i64);
        self.ngram_cache_misses
            .set(stats.ngram_rows.saturating_sub(stats.ngram_cache_hits) as i64);
        self.ngram_cache_hit_rate.set(if stats.ngram_rows > 0 {
            stats.ngram_cache_hits as f64 / stats.ngram_rows as f64
        } else {
            0.0
        });
        self.ngram_hot_rows.set(stats.ngram_hot_rows as i64);
        let mb = stats.model_bytes;
        for (phase, p) in [
            ("prefill", mb.prefill),
            ("decode", mb.decode),
            ("draft", mb.draft),
        ] {
            let l = PhaseLabels { phase };
            self.model_bytes.get_or_create(&l).set(p.total() as f64);
            self.model_seconds.get_or_create(&l).set(p.seconds);
            self.model_gb_s.get_or_create(&l).set(p.gb_s());
        }
        self.bandwidth.set(&stats);
        let mut out = String::new();
        encode(&mut out, &self.registry).expect("writing to a String cannot fail");
        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// An [`Engine`] that records token counts and latencies for every stream it hands out.
pub struct MeteredEngine {
    inner: Arc<dyn Engine>,
    metrics: Arc<Metrics>,
    /// The dashboard's request table and latency windows (#1198).
    stats: Option<Arc<super::stats::StatsHub>>,
}

impl MeteredEngine {
    pub fn new(inner: Arc<dyn Engine>, metrics: Arc<Metrics>) -> Self {
        Self {
            inner,
            metrics,
            stats: None,
        }
    }

    /// Also tracks every stream for the dashboard.
    pub fn with_stats(mut self, stats: Arc<super::stats::StatsHub>) -> Self {
        self.stats = Some(stats);
        self
    }
}

impl Engine for MeteredEngine {
    fn models(&self) -> Vec<ModelInfo> {
        self.inner.models()
    }

    fn stats(&self) -> EngineStats {
        self.inner.stats()
    }

    fn replica_stats(&self) -> Vec<crate::engine::replicas::ReplicaStats> {
        self.inner.replica_stats()
    }

    fn count_tokens(&self, req: &GenerateRequest) -> Result<u32, EngineError> {
        self.inner.count_tokens(req)
    }

    fn check_response_format(
        &self,
        format: &crate::engine::ResponseFormat,
    ) -> Result<(), crate::grammar::GrammarError> {
        self.inner.check_response_format(format)
    }

    fn generate(&self, req: GenerateRequest) -> Result<TokenStream, EngineError> {
        let stream = self.inner.generate(req)?;
        let mut timing = StreamTiming::new(self.metrics.clone());
        if let Some(stats) = &self.stats {
            timing.live = Some((stats.clone(), stats.begin()));
        }
        Ok(Box::pin(stream.inspect(move |ev| timing.observe(ev))))
    }
}

/// Per-request timing state. Dropping it (stream finished, errored or abandoned) records the
/// end-to-end latency.
struct StreamTiming {
    metrics: Arc<Metrics>,
    start: Instant,
    last_output: Option<Instant>,
    /// The dashboard's hub and this request's row in it.
    live: Option<(Arc<super::stats::StatsHub>, Arc<super::stats::LiveRequest>)>,
}

impl StreamTiming {
    fn new(metrics: Arc<Metrics>) -> Self {
        Self {
            metrics,
            start: Instant::now(),
            last_output: None,
            live: None,
        }
    }

    fn observe(&mut self, ev: &TokenEvent) {
        match ev {
            TokenEvent::Started { prompt_tokens } => {
                if let Some((_, r)) = &self.live {
                    r.started(*prompt_tokens);
                }
            }
            TokenEvent::Text(_) | TokenEvent::Reasoning(_) => {
                let now = Instant::now();
                let (window, seconds) = match self.last_output.replace(now) {
                    None => {
                        let s = (now - self.start).as_secs_f64();
                        self.metrics.ttft.observe(s);
                        (0, s)
                    }
                    Some(prev) => {
                        let s = (now - prev).as_secs_f64();
                        self.metrics.inter_token.observe(s);
                        (1, s)
                    }
                };
                if let Some((hub, r)) = &self.live {
                    r.output(now);
                    if window == 0 { &hub.ttft } else { &hub.itl }.record(now, seconds);
                }
            }
            TokenEvent::Finished { usage, .. } => {
                self.metrics
                    .prompt_tokens
                    .inc_by(usage.prompt_tokens.into());
                self.metrics
                    .completion_tokens
                    .inc_by(usage.completion_tokens.into());
            }
            _ => {}
        }
    }
}

impl Drop for StreamTiming {
    fn drop(&mut self) {
        self.metrics.e2e.observe(self.start.elapsed().as_secs_f64());
        if let Some((hub, r)) = &self.live {
            hub.end(r);
        }
    }
}

#[cfg(test)]
mod tests;
