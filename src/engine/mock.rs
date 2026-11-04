//! A GPU-free engine for tests and API work.
//!
//! It runs the same shape of loop a real engine will: requests queue up (admitted in the order
//! of [`super::queue`]), up to `max_batch` sequences are active at once, and every step
//! advances each active sequence by one token.
//! Output is deterministic: the reply echoes the last user message ("You said: ..."), one
//! whitespace-separated word per token; images in that message are announced first, one
//! `[image <mime> <N> bytes]` per image in the order received. With `thinking`, a fixed
//! reasoning text comes first.
//! A last user message of `/tool <name> <json>`, where `<name>` is one of the request's
//! tools, makes the mock call that tool instead of replying with text.
//!
//! With `logprobs`, each answer token gets a log probability of `-0.1 * (i % 5)` (`i` counts
//! answer tokens from 0) and `n` alternatives: the token itself first, then the fixed strings
//! `" alt1"`, `" alt2"`, ... at `-1.0 - 0.5 * k`.
//!
//! `ignore_eos` keeps the sequence going with `" lorem"` filler tokens after the scripted
//! answer ran out, until `max_tokens`; `min_tokens` pads with the same filler until that many
//! tokens exist.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::queue::{Scheduling, SchedulingPolicy, WaitQueue, WaitingClasses};
use super::stop::{LogprobHold, StopMatcher, StopResult};
use super::{
    Engine, EngineError, EngineStats, FinishReason, GenerateRequest, ImageInput, ModelInfo, Prompt,
    Role, TokenEvent, TokenLogprob, TokenStream, Tool, ToolCall, Usage,
};

pub const MOCK_REASONING: &str = "The user wants a reply. I will echo their message.";

#[derive(Debug, Clone)]
pub struct MockConfig {
    pub model: String,
    /// Simulated time per decode step (one token for every active sequence).
    pub step: Duration,
    /// Sequences decoded together; the rest wait in the queue.
    pub max_batch: usize,
    pub context_length: u32,
    /// Requests allowed to wait for a batch slot. When this many are queued (not yet
    /// admitted), `generate` fails with [`EngineError::Overloaded`]. A waiting request that
    /// is cancelled frees its slot when the loop next sweeps the queue (once per step).
    pub max_waiting: usize,
    /// Whether the served model advertises vision and echoes images in its reply.
    pub vision: bool,
    /// The sampling defaults the model declares (none by default).
    pub sampling: super::sampling_defaults::ModelSampling,
    /// The order waiting requests are admitted in (`--scheduling-policy`, see [`super::queue`]).
    pub scheduling: SchedulingPolicy,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            model: "mock".into(),
            step: Duration::from_millis(5),
            max_batch: 64,
            context_length: 32768,
            max_waiting: 1024,
            vision: true,
            sampling: Default::default(),
            scheduling: SchedulingPolicy::Fcfs,
        }
    }
}

pub struct MockEngine {
    cfg: MockConfig,
    created: u64,
    submit: mpsc::UnboundedSender<Sequence>,
    load: Arc<Load>,
    /// Sequences submitted but not yet admitted to the batch.
    queued: Arc<AtomicUsize>,
}

/// Scheduler occupancy, written by the loop and read by [`Engine::stats`].
#[derive(Default)]
struct Load {
    running: AtomicU32,
    waiting: AtomicU32,
    /// Prompt tokens admitted and tokens generated since start (the status line's rates).
    prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    /// The waiting requests by priority, and the fair-share keys among them.
    queue: std::sync::Mutex<(WaitingClasses, u32)>,
}

/// One place in the waiting queue; frees itself when dropped (admitted, cancelled, or lost).
struct QueueSlot(Arc<AtomicUsize>);

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct Sequence {
    /// Held while the sequence waits; released on admission.
    slot: Option<QueueSlot>,
    reasoning: VecDeque<String>,
    answer: VecDeque<String>,
    /// Emitted after the reasoning, then the sequence finishes with `ToolCalls`.
    tool_call: Option<ToolCall>,
    max_tokens: u32,
    generated: u32,
    prompt_tokens: u32,
    stop: StopMatcher,
    /// `GenerateRequest::ignore_eos`: keep emitting filler once the answer ran out.
    ignore_eos: bool,
    /// `GenerateRequest::min_tokens`: pad with filler until this many tokens exist.
    min_tokens: u32,
    /// `SamplingParams::logprobs`.
    logprobs: Option<u32>,
    /// Answer tokens sampled so far.
    answered: u32,
    /// `GenerateRequest::scheduling`.
    sched: Scheduling,
    /// When `generate` queued the sequence.
    submitted: Instant,
    /// When the scheduler admitted the sequence (start of prompt processing).
    admitted: Instant,
    /// When the first token was generated: prompt processing ended, decode began.
    first: Option<Instant>,
    lp_hold: LogprobHold,
    out: mpsc::UnboundedSender<TokenEvent>,
    cancel: tokio_util::sync::CancellationToken,
}

impl MockEngine {
    /// Starts the scheduler loop; must be called inside a Tokio runtime.
    pub fn new(cfg: MockConfig) -> Self {
        let (submit, rx) = mpsc::unbounded_channel();
        let load = Arc::new(Load::default());
        let waiting = WaitQueue::new(cfg.scheduling);
        tokio::spawn(run_loop(
            rx,
            waiting,
            cfg.step,
            cfg.max_batch.max(1),
            load.clone(),
        ));
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            cfg,
            created,
            submit,
            load,
            queued: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Engine for MockEngine {
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: self.cfg.model.clone(),
            owned_by: "legion".into(),
            created: self.created,
            context_length: Some(self.cfg.context_length),
            vision: self.cfg.vision,
            sampling: self.cfg.sampling.clone(),
        }]
    }

    fn stats(&self) -> EngineStats {
        let running = self.load.running.load(Ordering::Relaxed);
        let (waiting_classes, waiting_keys) =
            self.load.queue.lock().map(|q| *q).unwrap_or_default();
        EngineStats {
            running,
            decoding: running,
            waiting: self.load.waiting.load(Ordering::Relaxed),
            waiting_classes,
            waiting_keys,
            prompt_tokens: self.load.prompt_tokens.load(Ordering::Relaxed),
            generated_tokens: self.load.generated_tokens.load(Ordering::Relaxed),
            ..Default::default()
        }
    }

    fn count_tokens(&self, req: &GenerateRequest) -> Result<u32, EngineError> {
        if req.model != self.cfg.model {
            return Err(EngineError::ModelNotFound(req.model.clone()));
        }
        let (prompt_text, _, _) = flatten(&req.prompt);
        let (image_tokens, _) = prompt_images(&req.prompt);
        Ok(words(&prompt_text).len() as u32 + image_tokens)
    }

    fn check_response_format(
        &self,
        format: &super::ResponseFormat,
    ) -> Result<(), crate::grammar::GrammarError> {
        match format {
            super::ResponseFormat::JsonObject => Ok(()),
            super::ResponseFormat::JsonSchema(schema) => crate::grammar::check_json_schema(schema),
        }
    }

    fn generate(&self, req: GenerateRequest) -> Result<TokenStream, EngineError> {
        let prompt_tokens = self.count_tokens(&req)?;
        let (_, last_user, last_role) = flatten(&req.prompt);
        let (_, last_images) = prompt_images(&req.prompt);
        if prompt_tokens > self.cfg.context_length {
            return Err(EngineError::ContextOverflow {
                prompt: prompt_tokens,
                context: self.cfg.context_length,
            });
        }
        let tool_call = match &req.prompt {
            Prompt::Chat { tools, .. } if last_role == Some(Role::User) => {
                requested_tool(tools, last_user.as_deref())
            }
            _ => None,
        };
        // Each image in the last user message is announced before the echoed text.
        let announced: String = last_images
            .iter()
            .map(|img| format!("[image {} {} bytes] ", img.mime, img.bytes.len()))
            .collect();
        let json_reply = match &req.response_format {
            Some(super::ResponseFormat::JsonObject) => true,
            Some(super::ResponseFormat::JsonSchema(schema)) => {
                schema.get("type").and_then(serde_json::Value::as_str) == Some("object")
                    && schema
                        .get("required")
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(Vec::is_empty)
            }
            None => false,
        };
        let tool_call = if json_reply { None } else { tool_call };
        let reply = if json_reply {
            "{}".to_string()
        } else {
            match (&last_user, &tool_call) {
                (_, Some(_)) => String::new(),
                (Some(text), None) if !text.trim().is_empty() => {
                    format!("{announced}You said: {text}")
                }
                _ => format!("{announced}Hello from the mock engine."),
            }
        };
        let reasoning = if req.thinking == Some(true) && req.response_format.is_none() {
            words(MOCK_REASONING)
        } else {
            VecDeque::new()
        };
        let slot = self.reserve_slot()?;
        let (out, rx) = mpsc::unbounded_channel();
        let seq = Sequence {
            slot: Some(slot),
            reasoning,
            answer: words(&reply),
            tool_call,
            max_tokens: req.max_tokens,
            generated: 0,
            prompt_tokens,
            stop: StopMatcher::new(req.stop),
            ignore_eos: req.ignore_eos,
            min_tokens: req.min_tokens,
            logprobs: req.sampling.logprobs,
            answered: 0,
            sched: req.scheduling,
            submitted: Instant::now(),
            admitted: Instant::now(),
            first: None,
            lp_hold: LogprobHold::default(),
            out,
            cancel: req.cancel,
        };
        self.submit
            .send(seq)
            .map_err(|_| EngineError::Internal("mock scheduler stopped".into()))?;
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}

impl MockEngine {
    /// Takes a place in the waiting queue, or fails with `Overloaded` when it is full.
    fn reserve_slot(&self) -> Result<QueueSlot, EngineError> {
        self.queued
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.cfg.max_waiting).then_some(n + 1)
            })
            .map_err(|_| EngineError::Overloaded)?;
        Ok(QueueSlot(self.queued.clone()))
    }
}

/// Splits text into tokens: words, each after the first carrying its leading space.
fn words(text: &str) -> VecDeque<String> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| {
            if i == 0 {
                w.to_string()
            } else {
                format!(" {w}")
            }
        })
        .collect()
}

/// All prompt text joined (for the token count), the last user message, and the role of the last message.
fn flatten(prompt: &Prompt) -> (String, Option<String>, Option<Role>) {
    match prompt {
        Prompt::Text(t) => (t.clone(), Some(t.clone()), Some(Role::User)),
        Prompt::Chat {
            system,
            messages,
            tools,
        } => {
            let mut all = system.clone().unwrap_or_default();
            for t in tools {
                all.push(' ');
                all.push_str(&t.name);
            }
            for m in messages {
                all.push(' ');
                all.push_str(&m.content);
            }
            let last_user = messages
                .iter()
                .rev()
                .find(|m| m.role == Some(Role::User))
                .map(|m| m.content.clone());
            let last_role = messages.last().and_then(|m| m.role);
            (all, last_user, last_role)
        }
    }
}

/// Image tokens for the prompt count (one per image, wherever it appears), and the images of
/// the last user message (which the mock echoes).
fn prompt_images(prompt: &Prompt) -> (u32, Vec<ImageInput>) {
    match prompt {
        Prompt::Text(_) => (0, Vec::new()),
        Prompt::Chat { messages, .. } => {
            let total = messages.iter().map(|m| m.images.len() as u32).sum();
            let last_user = messages
                .iter()
                .rev()
                .find(|m| m.role == Some(Role::User))
                .map(|m| m.images.clone())
                .unwrap_or_default();
            (total, last_user)
        }
    }
}

/// The tool call a `/tool <name> <json>` user message asks for: `Some` only when `name` is
/// one of the request's tools and the rest of the message parses as JSON. The arguments are
/// kept exactly as given.
fn requested_tool(tools: &[Tool], last_user: Option<&str>) -> Option<ToolCall> {
    let rest = last_user?
        .trim()
        .strip_prefix("/tool")?
        .strip_prefix(char::is_whitespace)?;
    let (name, arguments) = rest.split_once(char::is_whitespace)?;
    let arguments = arguments.trim();
    if name.is_empty()
        || arguments.is_empty()
        || !tools.iter().any(|t| t.name == name)
        || serde_json::from_str::<serde_json::Value>(arguments).is_err()
    {
        return None;
    }
    Some(ToolCall {
        id: "call_0".into(),
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

/// Publishes the scheduler's occupancy for [`Engine::stats`].
fn publish(load: &Load, active: &[Sequence], waiting: &WaitQueue<Sequence>) {
    load.running.store(active.len() as u32, Ordering::Relaxed);
    load.waiting.store(waiting.len() as u32, Ordering::Relaxed);
    if let Ok(mut q) = load.queue.lock() {
        *q = (waiting.classes(), waiting.waiting_keys());
    }
}

async fn run_loop(
    mut rx: mpsc::UnboundedReceiver<Sequence>,
    mut waiting: WaitQueue<Sequence>,
    step: Duration,
    max_batch: usize,
    load: Arc<Load>,
) {
    let mut active: Vec<Sequence> = Vec::new();
    loop {
        if active.is_empty() && waiting.is_empty() {
            match rx.recv().await {
                Some(seq) => {
                    let sched = seq.sched;
                    waiting.push(seq, sched);
                }
                None => return,
            }
        }
        while let Ok(seq) = rx.try_recv() {
            let sched = seq.sched;
            waiting.push(seq, sched);
        }
        while active.len() < max_batch {
            let running = active.iter().map(|s| s.sched.share_key);
            let Some((mut seq, _)) = waiting.pop(max_batch, running) else {
                break;
            };
            seq.slot = None;
            seq.admitted = Instant::now();
            load.prompt_tokens
                .fetch_add(u64::from(seq.prompt_tokens), Ordering::Relaxed);
            let _ = seq.out.send(TokenEvent::Started {
                prompt_tokens: seq.prompt_tokens,
            });
            active.push(seq);
        }
        publish(&load, &active, &waiting);
        if !step.is_zero() {
            tokio::time::sleep(step).await;
        } else {
            tokio::task::yield_now().await;
        }
        let mut generated = 0u64;
        active.retain_mut(|seq| {
            let before = seq.generated;
            let keep = advance(seq);
            generated += u64::from(seq.generated.saturating_sub(before));
            keep
        });
        load.generated_tokens
            .fetch_add(generated, Ordering::Relaxed);
        // Waiting requests whose client left are dropped without ever running.
        waiting.retain(|s| !s.cancel.is_cancelled() && !s.out.is_closed());
        publish(&load, &active, &waiting);
    }
}

/// The mock's log probabilities for answer token number `i`.
fn token_logprob(token: &str, i: u32, n: u32) -> TokenLogprob {
    let logprob = -0.1 * (i % 5) as f32;
    let mut top = Vec::new();
    if n > 0 {
        top.push((token.to_string(), logprob));
        top.extend((1..n).map(|k| (format!(" alt{k}"), -1.0 - 0.5 * (k - 1) as f32)));
    }
    TokenLogprob {
        token: token.to_string(),
        logprob,
        bytes: token.as_bytes().to_vec(),
        top,
    }
}

/// Sends the text the stop matcher released, preceded by the log probabilities of the tokens
/// it carries (see [`LogprobHold`]).
fn send_text(seq: &mut Sequence, text: String, flush: bool) {
    let lps = if seq.logprobs.is_none() {
        Vec::new()
    } else if flush {
        seq.lp_hold.flush(&text)
    } else {
        seq.lp_hold.release(&text)
    };
    if !lps.is_empty() {
        let _ = seq.out.send(TokenEvent::Logprobs(lps));
    }
    if !text.is_empty() {
        let _ = seq.out.send(TokenEvent::Text(text));
    }
}

/// Generates one token for `seq`. Returns false once the sequence is finished.
fn advance(seq: &mut Sequence) -> bool {
    if seq.cancel.is_cancelled() || seq.out.is_closed() {
        return false;
    }
    let finish = |seq: &mut Sequence, reason, stop_sequence| {
        // Prefill runs from admission to the first token; decode from there to here.
        // A sequence that never generated a token reports the whole span as prefill.
        let now = Instant::now();
        let (prefill_ms, decode_ms) = match seq.first {
            Some(first) => (
                Some(first.duration_since(seq.admitted).as_secs_f64() * 1e3),
                Some(now.duration_since(first).as_secs_f64() * 1e3),
            ),
            None => (
                Some(now.duration_since(seq.admitted).as_secs_f64() * 1e3),
                Some(0.0),
            ),
        };
        let _ = seq.out.send(TokenEvent::Finished {
            reason,
            stop_sequence,
            usage: Usage {
                prompt_tokens: seq.prompt_tokens,
                completion_tokens: seq.generated,
                cached_tokens: 0,
                prefill_ms,
                decode_ms,
                queue_ms: Some(seq.admitted.duration_since(seq.submitted).as_secs_f64() * 1e3),
                spec: None,
            },
        });
        false
    };
    if seq.generated >= seq.max_tokens {
        let rest = seq.stop.flush();
        send_text(seq, rest, true);
        return finish(seq, FinishReason::Length, None);
    }
    if let Some(tok) = seq.reasoning.pop_front() {
        seq.first.get_or_insert_with(Instant::now);
        seq.generated += 1;
        let _ = seq.out.send(TokenEvent::Reasoning(tok));
        return true;
    }
    if let Some(call) = seq.tool_call.take() {
        seq.first.get_or_insert_with(Instant::now);
        seq.generated += 1;
        let _ = seq.out.send(TokenEvent::ToolCall(call));
        return finish(seq, FinishReason::ToolCalls, None);
    }
    let tok = match seq.answer.pop_front() {
        Some(tok) => tok,
        // The scripted answer ran out. `ignore_eos` and `min_tokens` keep the sequence
        // going with filler tokens instead of emitting the stop token (up to `max_tokens`,
        // which is checked above).
        None if seq.ignore_eos || seq.generated < seq.min_tokens => " lorem".to_string(),
        None => {
            let rest = seq.stop.flush();
            send_text(seq, rest, true);
            return finish(seq, FinishReason::Stop, None);
        }
    };
    seq.first.get_or_insert_with(Instant::now);
    seq.generated += 1;
    if let Some(n) = seq.logprobs {
        let lp = token_logprob(&tok, seq.answered, n);
        seq.lp_hold.push(&tok, lp);
    }
    seq.answered += 1;
    seq.stop.set_checking(seq.generated >= seq.min_tokens);
    match seq.stop.push(&tok) {
        StopResult::Continue(text) => {
            send_text(seq, text, false);
            true
        }
        StopResult::Stopped { text, stop } => {
            send_text(seq, text, false);
            finish(seq, FinishReason::StopSequence, Some(stop))
        }
    }
}

#[cfg(test)]
mod tests;
