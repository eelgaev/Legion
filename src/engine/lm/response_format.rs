//! Grammar-constrained sampling: the per-model lazy compiler and bounded LRU of grammars, each
//! sequence's [`Constraint`] (`response_format`, or strict tool calls), and the mask threads
//! that compute a sequence's next mask while the model computes its next logits.
//!
//! # Off the critical path
//!
//! A step's sampled token fixes the grammar state of the next step. Right after it is accepted,
//! [`Constraint::accept`] hands that state to a [`MaskPool`] thread, which finds its mask (a
//! table lookup once the state was seen, a walk of the vocabulary's trie the first time) while
//! the engine thread prepares and runs the next forward pass. Sampling then only waits for a
//! mask that is almost always ready, and applies it. With `--grammar-threads 0` the mask is
//! found on the engine thread at sampling time instead.
//!
//! A request with neither a response format nor strict tools has no `Constraint`: its steps
//! run exactly as before.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use super::LmEngine;
use crate::engine::{ResponseFormat, Tool};
use crate::grammar::{Grammar, GrammarCompiler, GrammarError, Matcher, Rejected, ToolSpec};
use crate::tokenizer::Tokenizer;

impl LmEngine {
    /// (successful compilations, cache hits), for diagnostics and cache-sharing tests.
    pub fn grammar_cache_stats(&self) -> (usize, usize) {
        let cache = self.grammars.lock().unwrap_or_else(|e| e.into_inner());
        (cache.compilations, cache.hits)
    }

    pub(super) fn compile_format(
        &self,
        format: &ResponseFormat,
    ) -> Result<crate::grammar::Grammar, crate::grammar::GrammarError> {
        self.grammars
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .compile(&self.tokenizer, format)
    }

    /// The call grammar of a request's tools when one of them is `strict` and this vocabulary's
    /// tool-call format is enforced; `None` otherwise. A strict schema the grammar cannot
    /// express leaves the calls unconstrained, as before strict tools were enforced.
    pub(super) fn compile_strict_tools(&self, tools: &[Tool]) -> Option<Grammar> {
        if !tools.iter().any(|t| t.strict) {
            return None;
        }
        let Some((_, end)) = self.chat_format.strict_tools() else {
            tracing::debug!("strict tools are not enforced for this model's tool-call format");
            return None;
        };
        let compiled = self
            .grammars
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .compile_tools(&self.tokenizer, tools, end);
        match compiled {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!(%e, "strict tools left unconstrained");
                None
            }
        }
    }
}

#[derive(Default)]
pub(super) struct GrammarCache {
    compiler: Option<GrammarCompiler>,
    entries: VecDeque<(String, Grammar)>,
    pub hits: usize,
    pub compilations: usize,
}

impl GrammarCache {
    pub fn compile(
        &mut self,
        tokenizer: &Tokenizer,
        format: &ResponseFormat,
    ) -> Result<Grammar, GrammarError> {
        // The sentinel cannot collide with a serialized JSON schema.
        let key = match format {
            ResponseFormat::JsonObject => "json_object".to_string(),
            ResponseFormat::JsonSchema(schema) => {
                serde_json::to_string(schema).expect("JSON value")
            }
        };
        self.get_or_compile(key, tokenizer, |c| match format {
            ResponseFormat::JsonObject => c.compile_json_object(),
            ResponseFormat::JsonSchema(schema) => c.compile_json_schema(schema),
        })
    }

    /// The call grammar of `tools` (strict ones follow their schema), ending at `end`.
    pub fn compile_tools(
        &mut self,
        tokenizer: &Tokenizer,
        tools: &[Tool],
        end: u32,
    ) -> Result<Grammar, GrammarError> {
        let specs: Vec<ToolSpec<'_>> = tools
            .iter()
            .map(|t| ToolSpec {
                name: &t.name,
                strict: t.strict.then_some(&t.parameters),
            })
            .collect();
        // A JSON array after the prefix: no collision with a response format's key.
        let key = format!(
            "tools:{end}:{}",
            serde_json::to_string(
                &(specs.iter())
                    .map(|s| (s.name, s.strict))
                    .collect::<Vec<_>>()
            )
            .expect("JSON value")
        );
        self.get_or_compile(key, tokenizer, |c| c.compile_tool_calls(&specs, end))
    }

    fn get_or_compile(
        &mut self,
        key: String,
        tokenizer: &Tokenizer,
        compile: impl FnOnce(&GrammarCompiler) -> Result<Grammar, GrammarError>,
    ) -> Result<Grammar, GrammarError> {
        if let Some(i) = self.entries.iter().position(|(k, _)| k == &key) {
            let entry = self.entries.remove(i).unwrap();
            let grammar = entry.1.clone();
            self.entries.push_back(entry);
            self.hits += 1;
            return Ok(grammar);
        }
        let compiler = self
            .compiler
            .get_or_insert_with(|| GrammarCompiler::new(tokenizer));
        let grammar = compile(compiler)?;
        self.compilations += 1;
        if self.entries.len() == 64 {
            self.entries.pop_front();
        }
        self.entries.push_back((key, grammar.clone()));
        Ok(grammar)
    }
}

/// A sequence's grammar.
pub(super) struct Constraint {
    kind: Kind,
    /// The current state's mask, being found by a mask thread since the last token.
    pending: Option<Arc<Slot>>,
    pool: Arc<MaskPool>,
}

enum Kind {
    /// `response_format`: every token, from the first; the grammar's end ends the sequence.
    Format(Matcher),
    /// Strict tool calls: free text until `open` is sampled outside reasoning, then the call's
    /// grammar (`call`) up to its end token, then free text again.
    Tools {
        grammar: Grammar,
        open: u32,
        call: Option<Matcher>,
    },
}

impl Constraint {
    /// A response format's constraint; its first mask is found while the prompt is read.
    pub fn format(grammar: &Grammar, pool: &Arc<MaskPool>) -> Self {
        let mut c = Self {
            kind: Kind::Format(grammar.matcher()),
            pending: None,
            pool: pool.clone(),
        };
        c.prefetch();
        c
    }

    /// Strict tool calls (`grammar` from [`GrammarCache::compile_tools`]) opened by `open`.
    pub fn tools(grammar: &Grammar, open: u32, pool: &Arc<MaskPool>) -> Self {
        Self {
            kind: Kind::Tools {
                grammar: grammar.clone(),
                open,
                call: None,
            },
            pending: None,
            pool: pool.clone(),
        }
    }

    /// Whether the next token is constrained (it needs its full logits row and no drafts).
    pub fn active(&self) -> bool {
        match &self.kind {
            Kind::Format(_) => true,
            Kind::Tools { call, .. } => call.is_some(),
        }
    }

    /// The token that would start constraining, while strict tools wait for a call: drafts
    /// stop before it, so a call's tokens are never drafted unconstrained.
    pub fn trigger(&self) -> Option<u32> {
        match &self.kind {
            Kind::Tools {
                open, call: None, ..
            } => Some(*open),
            _ => None,
        }
    }

    fn matcher(&self) -> Option<&Matcher> {
        match &self.kind {
            Kind::Format(m) => Some(m),
            Kind::Tools { call, .. } => call.as_ref(),
        }
    }

    /// Masks the tokens the grammar does not allow next out of `logits` (only while
    /// [`Self::active`]). Returns false when no allowed token has a usable logit (including
    /// min-token masking).
    pub fn mask(&mut self, logits: &mut [f32]) -> bool {
        let found = self.pending.take().and_then(|slot| slot.wait());
        // No mask thread, or it failed: find it here.
        let mask = found.or_else(|| self.matcher().and_then(Matcher::shared_mask));
        apply_mask(mask.as_deref().unwrap_or(&[]), logits)
    }

    /// Advances by the sampled `token`; `reasoning`: the model is inside its reasoning, where a
    /// tool call does not start. `Ok(true)`: a response format is complete and the sequence
    /// ends. The next state's mask is then found by a mask thread.
    pub fn accept(&mut self, token: u32, reasoning: bool) -> Result<bool, Rejected> {
        self.pending = None;
        match &mut self.kind {
            Kind::Format(m) => {
                m.accept_token(token)?;
                if m.is_finished() {
                    return Ok(true);
                }
            }
            Kind::Tools {
                grammar,
                open,
                call,
            } => match call {
                Some(m) => {
                    m.accept_token(token)?;
                    if m.is_finished() {
                        *call = None;
                        return Ok(false);
                    }
                }
                None if token == *open && !reasoning => *call = Some(grammar.matcher()),
                None => return Ok(false),
            },
        }
        self.prefetch();
        Ok(false)
    }

    fn prefetch(&mut self) {
        if let Some(m) = self.matcher() {
            self.pending = self.pool.submit(m.clone());
        }
    }
}

/// Sets every logit the bitset `mask` (bit `id % 32` of word `id / 32`) does not allow to
/// minus infinity, a word at a time; ids past the mask are not allowed. Returns whether an
/// allowed logit is finite.
pub(super) fn apply_mask(mask: &[u32], logits: &mut [f32]) -> bool {
    let mut any = false;
    for (w, chunk) in logits.chunks_mut(32).enumerate() {
        let bits = mask.get(w).copied().unwrap_or(0);
        if bits == 0 {
            chunk.fill(f32::NEG_INFINITY);
            continue;
        }
        for (i, l) in chunk.iter_mut().enumerate() {
            if bits >> i & 1 == 0 {
                *l = f32::NEG_INFINITY;
            } else {
                any |= l.is_finite();
            }
        }
    }
    any
}

/// Where a mask thread leaves one mask.
#[derive(Default)]
struct Slot {
    /// `Some(found)` once the thread is done; `found` is `None` if it failed (or the grammar
    /// was finished).
    mask: Mutex<Option<Option<Arc<[u32]>>>>,
    ready: Condvar,
}

impl Slot {
    fn put(&self, found: Option<Arc<[u32]>>) {
        *self.mask.lock().unwrap_or_else(|e| e.into_inner()) = Some(found);
        self.ready.notify_one();
    }

    fn wait(&self) -> Option<Arc<[u32]>> {
        let mut g = self.mask.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(found) = g.take() {
                return found;
            }
            g = self.ready.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

type MaskJob = (Matcher, Arc<Slot>);

/// The mask threads (`--grammar-threads`), started by the engine thread when a sequence first
/// needs one, off its core (#250), and stopped with the engine.
pub(super) struct MaskPool {
    threads: usize,
    tx: OnceLock<Option<Mutex<mpsc::Sender<MaskJob>>>>,
}

impl MaskPool {
    pub fn new(threads: usize) -> Arc<Self> {
        Arc::new(Self {
            threads,
            tx: OnceLock::new(),
        })
    }

    /// Queues `m`'s mask; `None`: no threads, the caller finds it itself.
    fn submit(&self, m: Matcher) -> Option<Arc<Slot>> {
        if self.threads == 0 {
            return None;
        }
        let tx = self.tx.get_or_init(|| self.start()).as_ref()?;
        let slot = Arc::new(Slot::default());
        let sent = tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send((m, slot.clone()));
        sent.ok().map(|()| slot)
    }

    fn start(&self) -> Option<Mutex<mpsc::Sender<MaskJob>>> {
        let (tx, rx) = mpsc::channel::<MaskJob>();
        let rx = Arc::new(Mutex::new(rx));
        let pin = crate::hw::host_core::off_host_core(&[]);
        let mut started = 0;
        for i in 0..self.threads {
            let (rx, pin) = (rx.clone(), pin.clone());
            let spawned = std::thread::Builder::new()
                .name(format!("legion-mask-{i}"))
                .spawn(move || {
                    crate::hw::host_core::pin_helper(&pin);
                    loop {
                        let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok((m, slot)) = job else {
                            return;
                        };
                        let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            m.shared_mask()
                        }));
                        slot.put(found.ok().flatten());
                    }
                });
            match spawned {
                Ok(_) => started += 1,
                Err(e) => tracing::warn!(%e, "cannot start a grammar mask thread"),
            }
        }
        (started > 0).then(|| Mutex::new(tx))
    }
}

#[cfg(test)]
mod tests;
