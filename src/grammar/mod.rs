//! Grammar-constrained decoding core: JSON Schema -> byte automaton -> per-step token mask.
//!
//! The model engine (`engine/lm/response_format.rs`) masks logits with a matcher's mask before
//! sampling for OpenAI JSON mode and structured outputs, and inside strict tool calls (the
//! Qwen3-Coder call body of `tool_call.rs`). Its mask threads find the next mask
//! ([`Matcher::shared_mask`]) while the model computes the next logits.
//!
//! # Design
//!
//! 1. **Schema to AST** (`schema.rs`). A JSON Schema is lowered to a small tree of byte-level
//!    nodes (literals, strings, numbers, objects, arrays, alternatives). Unsatisfiable branches
//!    (`false` schemas, recursion past the depth limit) are pruned while lowering.
//! 2. **AST to NFA** (`matcher.rs`). The tree is turned into a Thompson NFA over BYTES, built in
//!    continuation-passing style so loops (array items, object keys) share one copy of their body.
//!    Because every schema occurrence is instantiated separately, the automaton is finite: no
//!    stack is needed. The price is that recursion must be unrolled (see "Limits").
//! 3. **Lazy DFA**. A matcher state is the id of a DFA state (a set of NFA states) built on demand
//!    by subset construction; byte transitions are cached in a 256-entry table per state.
//! 4. **Bytes to tokens**. [`GrammarCompiler::new`] copies every token piece
//!    ([`Tokenizer::token_to_piece`]) into a byte trie once. The mask for a DFA state is found by
//!    walking the trie from that state: a subtree is cut as soon as the automaton dies, and every
//!    token ending at a live node is allowed. Tokens that split a UTF-8 character work for free
//!    because the automaton consumes raw bytes and tracks partial characters itself.
//! 5. **Mask cache**. The finished mask is stored per DFA state (an `Arc<[u32]>`). The repeated
//!    states in a generation (inside a string, between digits, after a comma) hit the cache, so a
//!    step is a hash-free table lookup plus a copy of `ceil(vocab / 32)` words. Only the first
//!    visit of a state pays for the trie walk. The cache lives in the [`Grammar`] and is shared by
//!    all its [`Matcher`]s, so concurrent requests with the same schema warm it together.
//!
//! ## Special tokens and EOS
//!
//! Control, user-defined and unknown tokens ([`Tokenizer::is_special`]) and tokens whose piece is
//! empty are never produced by the grammar: their text would otherwise match as literal bytes
//! inside a JSON string while the tokenizer renders them as control markers. The grammar's end
//! tokens (EOS and EOT when the vocabulary has them; a tool call's `</tool_call>` token for
//! [`GrammarCompiler::compile_tool_calls`]) are allowed exactly when the automaton is in an
//! accepting state ([`Matcher::is_complete`]); accepting one finishes the matcher, after which
//! nothing is allowed. A complete document may still be extendable (more digits of a top-level
//! number), so an end token is an addition to the mask, not always the only option. No
//! whitespace is allowed after the top-level value.
//!
//! The mask is a bitset over the vocabulary: bit `id % 32` of word `id / 32`, set = allowed.
//!
//! ## Supported schema subset
//!
//! * `type`: `string`, `number`, `integer`, `boolean`, `null`, `object`, `array`, or a list of
//!   those; a schema with no `type` is inferred from `properties`/`items`, or accepts any JSON.
//! * `enum`, `const` (any JSON value, emitted in compact form), `anyOf`, `oneOf` (treated like
//!   `anyOf`: the grammar does not enforce exclusivity), boolean schemas `true`/`false`.
//! * `$ref` to a location inside the same document (`#`, `#/$defs/x`, `#/definitions/x`, any JSON
//!   pointer). Recursive references are unrolled [`Options::recursion_depth`] times; past that
//!   the recursive branch is unavailable (an optional property or an empty array still works).
//! * strings: `minLength`/`maxLength` (in characters); `format` and other annotations are ignored.
//!   Strings are valid JSON and valid UTF-8; `\uXXXX` escapes never produce lone surrogates, so
//!   the output always parses.
//! * numbers: JSON number syntax, exponent limited to two digits (so the value stays finite).
//! * objects: `properties` are emitted in declaration order, `required` ones always, others
//!   optionally. `additionalProperties` absent or `false` means no other keys (the usual
//!   strict-mode behaviour). Without `properties`, `additionalProperties` as `true` or a schema
//!   gives a free-form map of that value type. Mixing `properties` with a true/schema
//!   `additionalProperties` is rejected.
//! * arrays: `items`, `minItems`, `maxItems`.
//! * JSON whitespace between tokens: up to [`Options::max_whitespace`] of ` `, `\n`, `\t`, `\r`.
//!
//! Anything else that changes validity (`pattern`, `minimum`/`maximum`/`multipleOf`, `allOf`,
//! `not`, `if`, `patternProperties`, `prefixItems`, `uniqueItems`, `minProperties`, ...) returns
//! [`GrammarError::Unsupported`] rather than being silently ignored.
//!
//! ## Limits
//!
//! * Free-form values ("any JSON", `response_format: json_object`) are unrolled to
//!   [`Options::any_depth`] levels of nesting; deeper structures cannot be produced.
//! * The NFA is capped at [`Options::max_nfa_states`]; huge `minItems`/`maxLength` counters or
//!   deep unrolling fail with [`GrammarError::TooLarge`].
//! * A true pushdown matcher (unbounded recursion, per-position mask splitting as in xgrammar)
//!   would lift both limits at the cost of a context-dependent part of the mask.

mod matcher;
mod schema;
mod tool_call;
mod validate;

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::tokenizer::Tokenizer;

pub use matcher::{Matcher, Rejected};
pub use tool_call::ToolSpec;
pub use validate::check_json_schema;

use matcher::{Lazy, Nfa};

/// Why a grammar could not be built.
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    /// The schema is malformed (wrong keyword types, dangling `$ref`, ...).
    #[error("invalid schema: {0}")]
    Invalid(String),
    /// The schema uses a feature this engine does not implement.
    #[error("unsupported schema feature: {0}")]
    Unsupported(String),
    /// The schema expands past [`Options::max_nfa_states`].
    #[error("schema too large: automaton exceeds {0} states")]
    TooLarge(usize),
    /// No document can satisfy the schema.
    #[error("schema is unsatisfiable")]
    Unsatisfiable,
}

/// Compilation knobs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Longest run of whitespace allowed between JSON tokens.
    pub max_whitespace: usize,
    /// Nesting levels of free-form JSON values.
    pub any_depth: usize,
    /// How many times a `$ref` may recur on one path before it is cut off.
    pub recursion_depth: usize,
    /// Upper bound on NFA states.
    pub max_nfa_states: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_whitespace: 4,
            any_depth: 6,
            recursion_depth: 3,
            max_nfa_states: 400_000,
        }
    }
}

/// Token pieces of the vocabulary: a byte trie plus a flat copy of the pieces.
pub(crate) struct TokenTable {
    vocab: usize,
    /// Tokens that end the generation (EOS, EOT).
    pub(crate) stop: Vec<u32>,
    pieces: Vec<u8>,
    /// `offsets[id]..offsets[id + 1]` is the piece of `id` (empty = never allowed).
    offsets: Vec<u32>,
    // Trie in CSR form: node `n` has edges `edge_start[n]..edge_start[n + 1]`.
    edge_start: Vec<u32>,
    edge_byte: Vec<u8>,
    edge_child: Vec<u32>,
    // Tokens ending at node `n`: `toks[tok_start[n]..tok_start[n + 1]]`.
    tok_start: Vec<u32>,
    toks: Vec<u32>,
}

impl TokenTable {
    fn new(tok: &Tokenizer) -> Self {
        let vocab = tok.vocab_size();
        let mut pieces = Vec::new();
        let mut offsets = Vec::with_capacity(vocab + 1);
        for id in 0..vocab as u32 {
            offsets.push(pieces.len() as u32);
            if !tok.is_special(id) {
                pieces.extend_from_slice(tok.token_to_piece(id));
            }
        }
        offsets.push(pieces.len() as u32);

        // Build with per-node child lists, then flatten.
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut ending: Vec<Vec<u32>> = vec![Vec::new()];
        for id in 0..vocab {
            let (a, b) = (offsets[id] as usize, offsets[id + 1] as usize);
            if a == b {
                continue;
            }
            let mut node = 0usize;
            for &byte in &pieces[a..b] {
                let found = children[node].iter().find(|e| e.0 == byte).map(|e| e.1);
                node = match found {
                    Some(c) => c as usize,
                    None => {
                        let c = children.len();
                        children.push(Vec::new());
                        ending.push(Vec::new());
                        children[node].push((byte, c as u32));
                        c
                    }
                };
            }
            ending[node].push(id as u32);
        }
        let mut edge_start = Vec::with_capacity(children.len() + 1);
        let mut edge_byte = Vec::new();
        let mut edge_child = Vec::new();
        let mut tok_start = Vec::with_capacity(children.len() + 1);
        let mut toks = Vec::new();
        for (kids, end) in children.iter_mut().zip(&ending) {
            edge_start.push(edge_byte.len() as u32);
            tok_start.push(toks.len() as u32);
            kids.sort_unstable();
            for &(b, c) in kids.iter() {
                edge_byte.push(b);
                edge_child.push(c);
            }
            toks.extend_from_slice(end);
        }
        edge_start.push(edge_byte.len() as u32);
        tok_start.push(toks.len() as u32);

        let mut stop: Vec<u32> = [tok.eos_id(), tok.eot_id()].into_iter().flatten().collect();
        stop.sort_unstable();
        stop.dedup();
        TokenTable {
            vocab,
            stop,
            pieces,
            offsets,
            edge_start,
            edge_byte,
            edge_child,
            tok_start,
            toks,
        }
    }

    pub(crate) fn vocab(&self) -> usize {
        self.vocab
    }

    pub(crate) fn mask_words(&self) -> usize {
        self.vocab.div_ceil(32)
    }

    pub(crate) fn piece(&self, id: u32) -> &[u8] {
        match self.offsets.get(id as usize..id as usize + 2) {
            Some(w) => &self.pieces[w[0] as usize..w[1] as usize],
            None => &[],
        }
    }

    pub(crate) fn edges(&self, node: u32) -> std::ops::Range<usize> {
        self.edge_start[node as usize] as usize..self.edge_start[node as usize + 1] as usize
    }

    pub(crate) fn tokens_at(&self, node: u32) -> &[u32] {
        &self.toks
            [self.tok_start[node as usize] as usize..self.tok_start[node as usize + 1] as usize]
    }

    pub(crate) fn edge(&self, i: usize) -> (u8, u32) {
        (self.edge_byte[i], self.edge_child[i])
    }
}

/// Builds the token trie once and compiles schemas against it.
pub struct GrammarCompiler {
    table: Arc<TokenTable>,
    options: Options,
}

impl GrammarCompiler {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self::with_options(tokenizer, Options::default())
    }

    pub fn with_options(tokenizer: &Tokenizer, options: Options) -> Self {
        GrammarCompiler {
            table: Arc::new(TokenTable::new(tokenizer)),
            options,
        }
    }

    /// Words in the mask `Matcher::fill_mask` expects.
    pub fn mask_words(&self) -> usize {
        self.table.mask_words()
    }

    pub fn compile_json_schema(&self, schema: &Value) -> Result<Grammar, GrammarError> {
        let node = schema::lower(schema, &self.options)?;
        self.finish(&node)
    }

    /// Any JSON object (`response_format: json_object`).
    pub fn compile_json_object(&self) -> Result<Grammar, GrammarError> {
        let node = schema::any_object(&self.options);
        self.finish(&node)
    }

    /// The body of a strict tool call to one of `tools` in the Qwen3-Coder format (see
    /// `tool_call.rs`): what follows the call's open token. `end` (the `</tool_call>` token)
    /// is the grammar's only end token, allowed once a call is complete.
    pub fn compile_tool_calls(
        &self,
        tools: &[ToolSpec<'_>],
        end: u32,
    ) -> Result<Grammar, GrammarError> {
        let node = tool_call::qwen3_coder(tools, &self.options)?;
        self.finish_with(&node, vec![end])
    }

    fn finish(&self, node: &schema::Node) -> Result<Grammar, GrammarError> {
        self.finish_with(node, self.table.stop.clone())
    }

    fn finish_with(&self, node: &schema::Node, stop: Vec<u32>) -> Result<Grammar, GrammarError> {
        let (nfa, start) = Nfa::build(node, &self.options)?;
        let lazy = Lazy::new(&nfa, start);
        Ok(Grammar {
            inner: Arc::new(GrammarInner {
                table: self.table.clone(),
                stop,
                nfa,
                lazy: Mutex::new(lazy),
            }),
        })
    }
}

pub(crate) struct GrammarInner {
    pub(crate) table: Arc<TokenTable>,
    /// The tokens that end a complete document: EOS / EOT, or a tool call's end token.
    pub(crate) stop: Vec<u32>,
    pub(crate) nfa: Nfa,
    pub(crate) lazy: Mutex<Lazy>,
}

/// A compiled grammar. Cheap to clone; clones share the automaton and the mask cache.
#[derive(Clone)]
pub struct Grammar {
    pub(crate) inner: Arc<GrammarInner>,
}

impl Grammar {
    /// A fresh matcher at the start of the document.
    pub fn matcher(&self) -> Matcher {
        Matcher::new(self.clone())
    }

    /// Words in the mask `Matcher::fill_mask` expects.
    pub fn mask_words(&self) -> usize {
        self.inner.table.mask_words()
    }

    /// Number of NFA states (a size indicator).
    pub fn nfa_states(&self) -> usize {
        self.inner.nfa.len()
    }

    /// Number of DFA states materialised so far.
    pub fn dfa_states(&self) -> usize {
        self.inner
            .lazy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
}
