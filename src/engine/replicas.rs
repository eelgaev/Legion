//! `legion serve --tp-replicas` (#679): several copies of one model in one server, each on its
//! own group of GPUs (a `--tp` group: on an AC922 an NVLink pair) with its threads and host
//! memory on that group's NUMA node, behind one [`Engine`].
//!
//! Every replica is a whole engine with its own scheduler, KV cache and prefix cache; the router
//! only decides where a new request goes, and the request stays there to the end (its KV is
//! local). It places a request on the replica with the fewest requests in flight (ties: the most
//! free KV blocks, then the lowest index), except that a prompt whose cached prefix one replica
//! most likely holds goes there ([`choose`]). The router knows nothing of how a replica runs
//! (tensor parallel over the meta device, or anything else), only [`Replica`].

use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;

use super::{
    Engine, EngineError, EngineStats, GenerateRequest, ModelInfo, ResponseFormat, TokenStream,
};

/// An engine the router can hand requests to in two halves: [`Self::prepare`] (checks and
/// tokenizes, the same on every replica) and [`Self::submit`] (queues it on this replica).
pub trait Replica: Engine {
    type Prepared: Send;

    fn prepare(&self, req: GenerateRequest) -> Result<Self::Prepared, EngineError>;

    /// The prepared request's prompt tokens.
    fn prompt(p: &Self::Prepared) -> &[u32];

    fn submit(&self, p: Self::Prepared) -> Result<TokenStream, Refused<Self::Prepared>>;
}

/// Why [`Replica::submit`] did not queue a request.
pub enum Refused<P> {
    /// The replica's waiting queue is full: the request comes back for another replica (boxed:
    /// the rare path keeps the `Result` small).
    Full(Box<P>),
    Failed(EngineError),
}

/// `--tp-replicas`: how many copies of the model to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaCount {
    /// As many `--tp` groups as the visible GPUs hold (one copy without `--tp`).
    Auto,
    Fixed(usize),
}

impl FromStr for ReplicaCount {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "auto" => Ok(Self::Auto),
            n => match n.parse::<usize>() {
                Ok(n) if n > 0 => Ok(Self::Fixed(n)),
                _ => Err(format!("expected auto or a count of at least 1, got {s:?}")),
            },
        }
    }
}

impl std::fmt::Display for ReplicaCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Fixed(n) => write!(f, "{n}"),
        }
    }
}

impl ReplicaCount {
    /// The number of replicas for `tp` GPUs each with `visible` GPUs: `Fixed(n)` past 1 needs
    /// `--tp 2` or more and `n × tp` GPUs; `Auto` takes `visible / tp` groups under `--tp`, one
    /// copy otherwise.
    pub fn resolve(self, tp: usize, visible: usize) -> Result<usize, String> {
        let tp = tp.max(1);
        match self {
            Self::Fixed(1) => Ok(1),
            Self::Auto if tp < 2 => Ok(1),
            Self::Auto => Ok((visible / tp).max(1)),
            Self::Fixed(n) if tp < 2 => Err(format!(
                "--tp-replicas {n}: each replica is a --tp group; pass --tp 2 or more"
            )),
            Self::Fixed(n) if n * tp > visible => Err(format!(
                "--tp-replicas {n} with --tp {tp} needs {} GPUs, {visible} visible",
                n * tp
            )),
            Self::Fixed(n) => Ok(n),
        }
    }
}

/// A prompt counts as cached on a replica when that replica most likely holds at least this
/// many of its tokens...
const MIN_HIT_TOKENS: usize = 256;
/// ... and at least half of the prompt: a system prompt every request shares does not pin
/// traffic to one replica (both soon hold it), the earlier turns of a conversation do.
const HIT_SHARE: usize = 2;
/// Prompts remembered per replica, most recent first, and their tokens at most.
const PREFIX_ENTRIES: usize = 32;
const PREFIX_TOKENS: usize = 4 << 20;

/// One replica's view for [`choose`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Candidate {
    /// Requests the router sent there that have not finished.
    pub assigned: usize,
    /// Its free KV cache blocks (from its last stats).
    pub free_kv: u32,
    /// The prompt tokens it most likely has cached ([`PrefixIndex`]).
    pub cached: usize,
}

/// The replica a new request of `prompt_len` tokens goes to. The least loaded one (fewest
/// requests in flight, then most free KV, then the lowest index), unless a replica most likely
/// holds a long enough part of the prompt ([`MIN_HIT_TOKENS`], [`HIT_SHARE`]): then that one
/// (the longest hit), as long as it can still start the request at once (fewer than `max_seqs`
/// in flight) or is no busier than the least loaded.
pub fn choose(c: &[Candidate], prompt_len: usize, max_seqs: usize) -> usize {
    let least = (0..c.len())
        .min_by_key(|&i| (c[i].assigned, std::cmp::Reverse(c[i].free_kv), i))
        .unwrap_or(0);
    let hit = (0..c.len())
        .filter(|&i| c[i].cached >= MIN_HIT_TOKENS && c[i].cached * HIT_SHARE >= prompt_len)
        .max_by_key(|&i| {
            (
                c[i].cached,
                std::cmp::Reverse(c[i].assigned),
                std::cmp::Reverse(i),
            )
        });
    match hit {
        Some(h) if c[h].assigned < max_seqs || c[h].assigned <= c[least].assigned => h,
        _ => least,
    }
}

/// The prompts a replica was given lately, to guess what its prefix cache holds. A guess: it
/// does not see the replica evict, nor the tokens a sequence generated.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    prompts: VecDeque<Arc<[u32]>>,
    tokens: usize,
}

impl PrefixIndex {
    /// Remembers `prompt`, forgetting the oldest past [`PREFIX_ENTRIES`] / [`PREFIX_TOKENS`].
    pub fn record(&mut self, prompt: Arc<[u32]>) {
        self.tokens += prompt.len();
        self.prompts.push_front(prompt);
        while self.prompts.len() > PREFIX_ENTRIES
            || (self.tokens > PREFIX_TOKENS && self.prompts.len() > 1)
        {
            if let Some(p) = self.prompts.pop_back() {
                self.tokens -= p.len();
            }
        }
    }

    /// The longest common prefix of `prompt` with a remembered prompt.
    pub fn longest(&self, prompt: &[u32]) -> usize {
        self.prompts
            .iter()
            .map(|p| p.iter().zip(prompt).take_while(|(a, b)| a == b).count())
            .max()
            .unwrap_or(0)
    }
}

/// Decrements its replica's in-flight count when the request's stream is dropped.
struct InFlight(Arc<AtomicUsize>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One replica's state for `/metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReplicaStats {
    /// Requests the router sent there that have not finished.
    pub assigned: u32,
    pub stats: EngineStats,
}

/// Several [`Replica`]s behind one [`Engine`].
pub struct ReplicaEngine<R: Replica> {
    replicas: Vec<R>,
    /// Each replica's `--max-seqs`.
    max_seqs: usize,
    assigned: Vec<Arc<AtomicUsize>>,
    /// Per replica, with prefix caching on; `None` routes by load alone.
    prefixes: Option<Mutex<Vec<PrefixIndex>>>,
}

impl<R: Replica> ReplicaEngine<R> {
    /// `replicas` must serve the same model with the same settings; `prefix_cache` whether
    /// they cache prefixes (cache hits then route to the replica holding them).
    pub fn new(replicas: Vec<R>, max_seqs: usize, prefix_cache: bool) -> Self {
        assert!(!replicas.is_empty(), "at least one replica");
        let n = replicas.len();
        Self {
            replicas,
            max_seqs: max_seqs.max(1),
            assigned: (0..n).map(|_| Arc::new(AtomicUsize::new(0))).collect(),
            prefixes: prefix_cache
                .then(|| Mutex::new((0..n).map(|_| PrefixIndex::default()).collect())),
        }
    }

    /// The replicas, in GPU order.
    pub fn members(&self) -> &[R] {
        &self.replicas
    }

    /// The replicas in the order a request of `prompt` tries them: [`choose`]'s pick first,
    /// then the rest by load.
    fn order(&self, prompt: &[u32]) -> Vec<usize> {
        let cached: Vec<usize> = match &self.prefixes {
            Some(p) => {
                let p = p.lock().unwrap_or_else(|e| e.into_inner());
                p.iter().map(|ix| ix.longest(prompt)).collect()
            }
            None => vec![0; self.replicas.len()],
        };
        let mut c: Vec<Candidate> = (self.replicas.iter().zip(&self.assigned).zip(cached))
            .map(|((r, a), cached)| {
                let s = r.stats();
                Candidate {
                    assigned: a.load(Ordering::Acquire),
                    free_kv: s.kv_blocks_total.saturating_sub(s.kv_blocks_used),
                    cached,
                }
            })
            .collect();
        let first = choose(&c, prompt.len(), self.max_seqs);
        let mut order = vec![first];
        while order.len() < c.len() {
            for &i in &order {
                c[i].assigned = usize::MAX;
                c[i].cached = 0;
            }
            order.push(choose(&c, 0, self.max_seqs));
        }
        order
    }
}

impl<R: Replica> Engine for ReplicaEngine<R> {
    fn models(&self) -> Vec<ModelInfo> {
        self.replicas[0].models()
    }

    fn check_response_format(
        &self,
        format: &ResponseFormat,
    ) -> Result<(), crate::grammar::GrammarError> {
        self.replicas[0].check_response_format(format)
    }

    fn generate(&self, req: GenerateRequest) -> Result<TokenStream, EngineError> {
        // Every replica loads the same model: any of them tokenizes for all.
        let mut p = self.replicas[0].prepare(req)?;
        let prompt: Option<Arc<[u32]>> = self.prefixes.as_ref().map(|_| R::prompt(&p).into());
        for i in self.order(R::prompt(&p)) {
            self.assigned[i].fetch_add(1, Ordering::AcqRel);
            let in_flight = InFlight(self.assigned[i].clone());
            match self.replicas[i].submit(p) {
                Ok(stream) => {
                    if let (Some(ix), Some(prompt)) = (&self.prefixes, prompt) {
                        ix.lock().unwrap_or_else(|e| e.into_inner())[i].record(prompt);
                    }
                    // The count drops with the stream (finished, failed or abandoned).
                    return Ok(Box::pin(stream.map(move |ev| {
                        let _ = &in_flight;
                        ev
                    })));
                }
                Err(Refused::Full(back)) => p = *back,
                Err(Refused::Failed(e)) => return Err(e),
            }
        }
        Err(EngineError::Overloaded)
    }

    fn count_tokens(&self, req: &GenerateRequest) -> Result<u32, EngineError> {
        self.replicas[0].count_tokens(req)
    }

    fn stats(&self) -> EngineStats {
        let all: Vec<EngineStats> = self.replicas.iter().map(Engine::stats).collect();
        merge_stats(&all)
    }

    fn replica_stats(&self) -> Vec<ReplicaStats> {
        (self.replicas.iter().zip(&self.assigned))
            .map(|(r, a)| ReplicaStats {
                assigned: a.load(Ordering::Acquire) as u32,
                stats: r.stats(),
            })
            .collect()
    }
}

/// The replicas' stats as one engine's: counts and totals add up, the draft depth is the mean
/// over the decoding sequences, the rooflines are the first replica's (the replicas run on
/// alike GPUs).
pub fn merge_stats(all: &[EngineStats]) -> EngineStats {
    let Some(first) = all.first() else {
        return EngineStats::default();
    };
    let mut out = *first;
    for s in &all[1..] {
        out.running += s.running;
        out.prefilling += s.prefilling;
        out.decoding += s.decoding;
        out.waiting += s.waiting;
        out.waiting_classes = out.waiting_classes.merge(&s.waiting_classes);
        out.waiting_keys += s.waiting_keys;
        out.prompt_tokens += s.prompt_tokens;
        out.generated_tokens += s.generated_tokens;
        out.kv_blocks_used += s.kv_blocks_used;
        out.kv_blocks_total += s.kv_blocks_total;
        out.expert_cache_hits += s.expert_cache_hits;
        out.expert_cache_misses += s.expert_cache_misses;
        out.expert_cache_slots += s.expert_cache_slots;
        out.ngram_rows += s.ngram_rows;
        out.ngram_cache_hits += s.ngram_cache_hits;
        out.ngram_hot_rows += s.ngram_hot_rows;
        out.prefix_cache_hit_tokens += s.prefix_cache_hit_tokens;
        out.preemptions += s.preemptions;
        out.spec.drafted += s.spec.drafted;
        out.spec.accepted += s.spec.accepted;
        out.spec.steps += s.spec.steps;
        add_totals(&mut out.model_bytes, &s.model_bytes);
    }
    let decoding: u32 = all.iter().map(|s| s.decoding).sum();
    if decoding > 0 {
        out.spec_mean_k = all
            .iter()
            .map(|s| s.spec_mean_k * f64::from(s.decoding))
            .sum::<f64>()
            / f64::from(decoding);
    }
    out
}

fn add_totals(a: &mut crate::ggml::model_bytes::Totals, b: &crate::ggml::model_bytes::Totals) {
    add_phase(&mut a.prefill, &b.prefill);
    add_phase(&mut a.decode, &b.decode);
    add_phase(&mut a.draft, &b.draft);
    for (x, y) in a.stages.iter_mut().zip(&b.stages) {
        add_phase(&mut x.prefill, &y.prefill);
        add_phase(&mut x.decode, &y.decode);
    }
    a.n_stages = a.n_stages.max(b.n_stages);
}

fn add_phase(a: &mut crate::ggml::model_bytes::Phase, b: &crate::ggml::model_bytes::Phase) {
    a.steps += b.steps;
    a.dense += b.dense;
    a.experts_upper += b.experts_upper;
    a.experts_device += b.experts_device;
    a.experts_host += b.experts_host;
    a.staged += b.staged;
    a.kv += b.kv;
    a.kv_host += b.kv_host;
    a.seconds += b.seconds;
}

#[cfg(test)]
mod tests;
