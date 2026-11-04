//! Sequences' text off the engine thread (#47, `--stream-thread-seqs`): detokenizing, the
//! reasoning and tool-call parsers, log probabilities and the sends to the clients' streams.
//!
//! The engine thread picks every token and decides when a sequence ends; what the text of
//! those tokens becomes never feeds back into the next step, except through stop strings,
//! which end a sequence, and a grammar (`response_format`, `strict` tools), which reads the
//! reasoning parser's state at every token. So a sequence without either can have its text handled by a
//! thread of its own ([`Text::Away`]): the engine thread queues each token (and the sequence's
//! end) in an [`Outbox`], sends the step's queue at once, and goes on to the next step while
//! the stream thread turns the tokens into events. Each sequence's events leave in the order
//! they would have on the engine thread, with the same contents: the same code runs
//! ([`TextState`]), only on another thread. At C=128 on a tiny model this was ~240 us of the
//! engine thread's ~280 us of host time per step (the sends wake the clients' tasks).
//!
//! Small batches keep their text on the engine thread ([`Text::Here`]): there a step's text is
//! cheaper than waking another thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;

use tokio::sync::mpsc::UnboundedSender;

use super::super::reasoning::ReasoningParser;
use super::super::sampling;
use super::super::stop::{LogprobHold, StopMatcher, StopResult};
use super::super::tool_parser::{self, ToolParser};
use super::super::{FinishReason, TokenEvent, TokenLogprob, Usage};
use crate::spec::SeqId;
use crate::tokenizer::{DecodeStream, Tokenizer};

/// What a sequence's text state is made of, but the detokenizer (made where it runs).
pub(super) struct TextSpec {
    /// Whether a grammar constrains the sequence (`response_format`, `strict` tools): it reads
    /// the reasoning parser's state at every token ([`Text::inside_reasoning`]), so the text
    /// stays on the engine thread.
    pub(super) grammar: bool,
    pub(super) reasoning: Option<ReasoningParser>,
    pub(super) tools: Option<Box<dyn ToolParser>>,
    pub(super) stop: StopMatcher,
    pub(super) logprobs: Option<u32>,
    pub(super) out: UnboundedSender<TokenEvent>,
}

/// A sequence's text: the generated tokens through the detokenizer, the reasoning and
/// tool-call parsers and the stop strings, with their log probabilities, sent to its stream.
pub(super) struct TextState<'t> {
    tokenizer: &'t Tokenizer,
    detok: DecodeStream<'t>,
    reasoning: Option<ReasoningParser>,
    tools: Option<Box<dyn ToolParser>>,
    stop: StopMatcher,
    logprobs: Option<u32>,
    lp_hold: LogprobHold,
    out: UnboundedSender<TokenEvent>,
}

/// How a sequence ends ([`TextState::end`]).
pub(super) struct End {
    /// Whether the parsers and the stop matcher release what they hold first.
    pub(super) flush: bool,
    pub(super) reason: FinishReason,
    pub(super) stop_sequence: Option<String>,
    pub(super) usage: Usage,
    /// A stop token's end: `Stop` becomes `ToolCalls` when the sequence called tools.
    pub(super) tools: bool,
}

impl<'t> TextState<'t> {
    pub(super) fn new(tokenizer: &'t Tokenizer, spec: TextSpec) -> Self {
        Self {
            tokenizer,
            detok: tokenizer.decode_stream(),
            reasoning: spec.reasoning,
            tools: spec.tools,
            stop: spec.stop,
            logprobs: spec.logprobs,
            lp_hold: LogprobHold::default(),
            out: spec.out,
        }
    }

    /// Streams `token`'s text (`raw`: its row's logits, for its log probability). `checking`:
    /// whether stop strings are checked yet (`min_tokens`). Returns the stop string the text
    /// completed, once the text before it is sent: the sequence ends there.
    pub(super) fn token(
        &mut self,
        token: u32,
        raw: Option<Vec<f32>>,
        checking: bool,
    ) -> Option<String> {
        let piece = self.detok.push(token);
        let mut visible = String::new();
        for ev in self.parse(&piece) {
            match ev {
                TokenEvent::Text(t) => visible.push_str(&t),
                other => {
                    let _ = self.out.send(other);
                }
            }
        }
        if let (Some(n), Some(raw)) = (self.logprobs, raw)
            && !visible.is_empty()
        {
            let lp = self.token_logprob(&raw, token, n);
            self.lp_hold.push(&visible, lp);
        }
        self.stop.set_checking(checking);
        match self.stop.push(&visible) {
            StopResult::Continue(text) => {
                self.send_text(text, false);
                None
            }
            StopResult::Stopped { text, stop } => {
                self.send_text(text, false);
                Some(stop)
            }
        }
    }

    /// Model text through the reasoning and tool-call parsers.
    fn parse(&mut self, piece: &str) -> Vec<TokenEvent> {
        let events = match &mut self.reasoning {
            Some(r) => r.push(piece),
            None => vec![TokenEvent::Text(piece.to_string())],
        };
        self.through_tools(events)
    }

    fn through_tools(&mut self, events: Vec<TokenEvent>) -> Vec<TokenEvent> {
        let Some(tools) = &mut self.tools else {
            return events;
        };
        let mut out = Vec::new();
        for ev in events {
            match ev {
                TokenEvent::Text(t) => out.extend(tools.push(&t)),
                other => out.push(other),
            }
        }
        out
    }

    fn had_calls(&self) -> bool {
        self.tools.as_ref().is_some_and(|t| t.had_calls())
    }

    /// End of generation: release everything the parsers still hold.
    fn flush(&mut self) {
        let mut events = Vec::new();
        let tail = self.detok.finish();
        if !tail.is_empty() {
            events.extend(self.parse(&tail));
        }
        if let Some(r) = &mut self.reasoning {
            let rest = r.finish();
            events.extend(self.through_tools(rest));
        }
        if let Some(t) = &mut self.tools {
            events.extend(t.finish());
        }
        let mut visible = String::new();
        for ev in events {
            match ev {
                TokenEvent::Text(t) => visible.push_str(&t),
                other => {
                    let _ = self.out.send(other);
                }
            }
        }
        // Held text may still complete a stop string.
        let text = match self.stop.push(&visible) {
            StopResult::Continue(text) => text + &self.stop.flush(),
            StopResult::Stopped { text, .. } => text,
        };
        self.send_text(text, true);
    }

    fn token_logprob(&self, raw: &[f32], token: u32, n: u32) -> TokenLogprob {
        let (logprob, top) = sampling::logprobs(raw, token, n as usize);
        let text =
            |id: u32| String::from_utf8_lossy(self.tokenizer.token_to_piece(id)).into_owned();
        TokenLogprob {
            token: text(token),
            logprob,
            bytes: self.tokenizer.token_to_piece(token).to_vec(),
            top: top.into_iter().map(|(id, lp)| (text(id), lp)).collect(),
        }
    }

    /// Sends text the stop matcher released, preceded by its tokens' log probabilities.
    fn send_text(&mut self, text: String, flush: bool) {
        if self.logprobs.is_some() {
            let lps = if flush {
                self.lp_hold.flush(&text)
            } else {
                self.lp_hold.release(&text)
            };
            if !lps.is_empty() {
                let _ = self.out.send(TokenEvent::Logprobs(lps));
            }
        }
        if !text.is_empty() {
            let _ = self.out.send(TokenEvent::Text(text));
        }
    }

    /// The sequence's last events: what the parsers hold (with `end.flush`), then `Finished`.
    pub(super) fn end(&mut self, end: End) {
        if end.flush {
            self.flush();
        }
        let reason = if end.tools {
            tool_parser::finish_reason(self.had_calls(), end.reason)
        } else {
            end.reason
        };
        let _ = self.out.send(TokenEvent::Finished {
            reason,
            stop_sequence: end.stop_sequence,
            usage: end.usage,
        });
    }

    /// Sends `ev` as it is.
    pub(super) fn send(&self, ev: TokenEvent) {
        let _ = self.out.send(ev);
    }
}

/// What the engine thread tells the stream thread about a sequence.
pub(super) enum Msg {
    /// A sequence starts: its text state.
    Open(Box<TextSpec>),
    /// See [`TextState::token`].
    Token {
        token: u32,
        raw: Option<Vec<f32>>,
        checking: bool,
    },
    /// The sequence ended: see [`TextState::end`].
    End(Box<End>),
    /// The sequence failed: an `Error` event ends it.
    Fail(String),
}

/// A step's messages, in order.
type Batch = Vec<(SeqId, Msg)>;

/// Where a sequence's text is handled.
pub(super) enum Text<'t> {
    /// On the engine thread.
    Here(Box<TextState<'t>>),
    /// On the stream thread, through the [`Outbox`].
    Away(SeqId),
}

impl Text<'_> {
    /// [`TextState::token`] where the text is. Away, the stop strings are never checked (a
    /// sequence with stop strings stays here), so it returns `None`.
    pub(super) fn token(
        &mut self,
        outbox: &mut Outbox,
        token: u32,
        raw: Option<Vec<f32>>,
        checking: bool,
    ) -> Option<String> {
        match self {
            Text::Here(text) => text.token(token, raw, checking),
            Text::Away(id) => {
                outbox.push(
                    *id,
                    Msg::Token {
                        token,
                        raw,
                        checking,
                    },
                );
                None
            }
        }
    }

    /// [`TextState::end`] where the text is.
    pub(super) fn end(&mut self, outbox: &mut Outbox, end: End) {
        match self {
            Text::Here(text) => text.end(end),
            Text::Away(id) => outbox.push(*id, Msg::End(Box::new(end))),
        }
    }

    /// Whether the text so far ends inside the reasoning block. Only a sequence with a grammar
    /// asks, and its text is always here ([`Outbox::text`]).
    pub(super) fn inside_reasoning(&self) -> bool {
        match self {
            Text::Here(text) => text.reasoning.as_ref().is_some_and(ReasoningParser::inside),
            Text::Away(_) => {
                debug_assert!(false, "a sequence with a grammar went away");
                false
            }
        }
    }

    /// The sequence failed: an `Error` event ends its stream.
    pub(super) fn fail(&mut self, outbox: &mut Outbox, e: String) {
        match self {
            Text::Here(text) => text.send(TokenEvent::Error(e)),
            Text::Away(id) => outbox.push(*id, Msg::Fail(e)),
        }
    }
}

/// The engine thread's side of the stream thread: the step's messages, sent at once.
pub(super) struct Outbox {
    /// Sequences running at once from which new ones go to the stream thread (0: never).
    min_seqs: usize,
    tokenizer: Arc<Tokenizer>,
    batch: Batch,
    /// Started on the first sequence that goes away.
    thread: Option<Thread>,
    /// Sequences whose text exists, either side ([`Live`]).
    live: Arc<AtomicUsize>,
}

struct Thread {
    tx: std::sync::mpsc::Sender<Batch>,
    /// Batches the thread is done with, kept for their capacity.
    spare: std::sync::mpsc::Receiver<Batch>,
    handle: JoinHandle<()>,
}

/// Counts a sequence as live while it exists ([`Outbox::text`]).
pub(super) struct Live(Arc<AtomicUsize>);

impl Drop for Live {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Outbox {
    pub(super) fn new(tokenizer: Arc<Tokenizer>, min_seqs: usize) -> Self {
        Self {
            min_seqs,
            tokenizer,
            batch: Vec::new(),
            thread: None,
            live: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A new sequence's text, here or away: away once `min_seqs` sequences (this one included)
    /// are live, unless the sequence has stop strings (their match ends it, which the engine
    /// thread must know before the next step) or a grammar (it follows the reasoning parser).
    /// Called on the engine thread.
    pub(super) fn text<'t>(
        &mut self,
        tokenizer: &'t Tokenizer,
        id: SeqId,
        spec: TextSpec,
    ) -> (Text<'t>, Live) {
        let live = self.live.fetch_add(1, Ordering::Relaxed) + 1;
        let guard = Live(self.live.clone());
        let away =
            self.min_seqs > 0 && live >= self.min_seqs && spec.stop.is_empty() && !spec.grammar;
        if !away || !self.start() {
            return (Text::Here(Box::new(TextState::new(tokenizer, spec))), guard);
        }
        self.batch.push((id, Msg::Open(Box::new(spec))));
        (Text::Away(id), guard)
    }

    /// Queues `msg` for sequence `id` (sent with the step's others by [`Self::flush`]).
    pub(super) fn push(&mut self, id: SeqId, msg: Msg) {
        self.batch.push((id, msg));
    }

    /// Sends the queued messages to the stream thread.
    pub(super) fn flush(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let Some(t) = &self.thread else {
            return;
        };
        let next = t.spare.try_recv().unwrap_or_default();
        let batch = std::mem::replace(&mut self.batch, next);
        // The thread outlives every send (it ends when the sender drops).
        let _ = t.tx.send(batch);
    }

    /// Starts the stream thread unless it runs; false when it cannot start.
    fn start(&mut self) -> bool {
        if self.thread.is_some() {
            return true;
        }
        let (tx, rx) = std::sync::mpsc::channel::<Batch>();
        let (back, spare) = std::sync::mpsc::channel::<Batch>();
        let tokenizer = self.tokenizer.clone();
        // Off the engine thread's core (#250), worked out here on the engine thread.
        let cpus = crate::hw::host_core::off_host_core(&[]);
        let spawned = std::thread::Builder::new()
            .name("legion-stream".into())
            .spawn(move || {
                crate::hw::host_core::pin_helper(&cpus);
                run(&tokenizer, rx, back);
            });
        match spawned {
            Ok(handle) => {
                self.thread = Some(Thread { tx, spare, handle });
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "cannot start the stream thread; text stays on the engine thread");
                self.min_seqs = 0;
                false
            }
        }
    }
}

impl Drop for Outbox {
    /// Sends what is queued and waits for the stream thread to send it on.
    fn drop(&mut self) {
        self.flush();
        if let Some(t) = self.thread.take() {
            drop(t.tx);
            let _ = t.handle.join();
        }
    }
}

/// The stream thread: every batch's messages in order, until the engine thread's side drops.
fn run(
    tokenizer: &Tokenizer,
    rx: std::sync::mpsc::Receiver<Batch>,
    back: std::sync::mpsc::Sender<Batch>,
) {
    let mut seqs: std::collections::HashMap<SeqId, TextState<'_>> = Default::default();
    for mut batch in rx {
        for (id, msg) in batch.drain(..) {
            match msg {
                Msg::Open(spec) => {
                    seqs.insert(id, TextState::new(tokenizer, *spec));
                }
                Msg::Token {
                    token,
                    raw,
                    checking,
                } => {
                    if let Some(s) = seqs.get_mut(&id) {
                        let stopped = s.token(token, raw, checking);
                        debug_assert!(stopped.is_none(), "a sequence with stop strings went away");
                    }
                }
                Msg::End(end) => {
                    if let Some(mut s) = seqs.remove(&id) {
                        s.end(*end);
                    }
                }
                Msg::Fail(e) => {
                    if let Some(s) = seqs.remove(&id) {
                        s.send(TokenEvent::Error(e));
                    }
                }
            }
        }
        let _ = back.send(batch);
    }
}

#[cfg(test)]
mod tests;
