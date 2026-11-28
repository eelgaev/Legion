//! A request's thinking budget (`thinking_budget_tokens`, Anthropic's `budget_tokens`): once
//! the reply's reasoning holds that many tokens, the engine writes the chat format's own end of
//! reasoning in place of the model's next tokens (`</think>`, Gemma 4's `<channel|>`, Inkling-2's
//! and Muse Glimmer's end of a message) and lets the answer run from there.
//!
//! The budget follows the tokens, not the text (which may be on its stream thread): [`Marks`]
//! are the format's markers as token ids, and [`ThinkBudget`] walks every token a sequence
//! keeps. Only the tokens inside the reasoning count, not its markers. The close goes in at a
//! point where the text is whole: never inside a character split over tokens. A step that
//! verifies drafts must not hold a token after the one that is replaced (the drafts after it
//! were drafted from the model's token), so [`ThinkBudget::cut`] ends the drafts there; the
//! close then takes a step per token. A request without a budget has no [`ThinkBudget`], so
//! nothing here runs for it.
//!
//! Strata (MIT) has the same per-request budget (its `reasoning_budget_tokens`, which works here
//! too); it closes Qwen's `<think>` the same way, at a whole character.

use crate::tokenizer::Tokenizer;

/// A chat format's reasoning markers, as token ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Marks {
    /// The token that opens the reasoning, or the header that decides ([`Header`]).
    open: u32,
    header: Option<Header>,
    /// The token the model ends its reasoning with.
    close: u32,
    /// What the budget writes to end it (it holds `close`; Qwen's ends in `\n\n` after it).
    inject: Vec<u32>,
    /// The reply starts in a header: Muse's generation prompt ends at `<|start|>assistant`.
    starts_in_header: bool,
}

/// After [`Marks::open`], text that says whether reasoning follows.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Header {
    /// Gemma 4's `<|channel>NAME`: reasoning when the name is this one.
    Name(&'static str),
    /// Muse's `<|start|>assistant to=RECIPIENT<|message|>`: reasoning when the header before
    /// the `end` token ends with this recipient.
    Until { end: u32, recipient: &'static str },
}

/// Text in a header past this length is not one (an unframed reply): outside.
const MAX_HEADER: usize = 64;

impl Marks {
    /// Reasoning between the tokens `open` and `close` (`<think>`...`</think>`), ended with the
    /// text `inject`.
    pub(super) fn tags(tok: &Tokenizer, open: &str, close: &str, inject: &str) -> Option<Self> {
        Self::new(tok, open, None, close, inject, false)
    }

    /// Gemma 4: a channel `open` + `name` up to `close`.
    pub(super) fn channel(
        tok: &Tokenizer,
        open: &str,
        name: &'static str,
        close: &str,
    ) -> Option<Self> {
        Self::new(tok, open, Some(Header::Name(name)), close, close, false)
    }

    /// Muse Glimmer: messages from `start` through the header to `message`, the ones to
    /// `recipient` reasoning, each ended by `close`; the reply starts in the first header.
    pub(super) fn messages(
        tok: &Tokenizer,
        start: &str,
        message: &str,
        recipient: &'static str,
        close: &str,
    ) -> Option<Self> {
        let end = tok.token_id(message)?;
        Self::new(
            tok,
            start,
            Some(Header::Until { end, recipient }),
            close,
            close,
            true,
        )
    }

    fn new(
        tok: &Tokenizer,
        open: &str,
        header: Option<Header>,
        close: &str,
        inject: &str,
        starts_in_header: bool,
    ) -> Option<Self> {
        let open = tok.token_id(open)?;
        let close = tok.token_id(close)?;
        let inject = tok.encode_with(inject, false, true);
        inject.contains(&close).then_some(Self {
            open,
            header,
            close,
            inject,
            starts_in_header,
        })
    }

    /// The tokens the budget writes.
    #[cfg(test)]
    pub(super) fn inject(&self) -> &[u32] {
        &self.inject
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    Outside,
    /// In a header, its text so far.
    Header(String),
    Inside,
    /// Writing the close: the next one is `inject[i]`.
    Closing(usize),
}

/// One sequence's budget: where its reply is and how many reasoning tokens it has written.
#[derive(Clone)]
pub(super) struct ThinkBudget<'t> {
    marks: Marks,
    tok: &'t Tokenizer,
    budget: u32,
    used: u32,
    state: State,
    /// Bytes still to come of the last character (one split over tokens).
    pending: u8,
}

impl<'t> ThinkBudget<'t> {
    /// `budget` reasoning tokens for a reply in `marks`' format; `inside`: the prompt opened
    /// the reasoning (`Job::reasoning`).
    pub(super) fn new(marks: &Marks, tok: &'t Tokenizer, budget: u32, inside: bool) -> Self {
        let state = if inside {
            State::Inside
        } else if marks.starts_in_header {
            State::Header(String::new())
        } else {
            State::Outside
        };
        Self {
            marks: marks.clone(),
            tok,
            budget,
            used: 0,
            state,
            pending: 0,
        }
    }

    /// The token the sequence must take next instead of its own: the close is due or under way.
    pub(super) fn forced(&self) -> Option<u32> {
        match self.state {
            State::Inside if self.used >= self.budget && self.pending == 0 => {
                Some(self.marks.inject[0])
            }
            State::Closing(i) => Some(self.marks.inject[i]),
            _ => None,
        }
    }

    /// The reasoning tokens counted so far.
    #[cfg(test)]
    pub(super) fn used(&self) -> u32 {
        self.used
    }

    /// Whether the sequence is in its reasoning (or closing it).
    #[cfg(test)]
    pub(super) fn inside(&self) -> bool {
        matches!(self.state, State::Inside | State::Closing(_))
    }

    /// The sequence kept `token` (the [`Self::forced`] one when there was one).
    pub(super) fn accept(&mut self, token: u32) {
        if self.forced().is_some() {
            let i = match self.state {
                State::Closing(i) => i + 1,
                _ => 1,
            };
            self.state = if i < self.marks.inject.len() {
                State::Closing(i)
            } else {
                State::Outside
            };
            self.pending = 0;
            return;
        }
        let marks = &self.marks;
        self.state = match std::mem::replace(&mut self.state, State::Outside) {
            State::Outside | State::Closing(_) if token == marks.open => match marks.header {
                Some(_) => State::Header(String::new()),
                None => {
                    self.pending = 0;
                    State::Inside
                }
            },
            State::Outside | State::Closing(_) => State::Outside,
            State::Inside if token == marks.close => State::Outside,
            State::Inside => {
                self.used = self.used.saturating_add(1);
                self.pending = pending_after(self.pending, self.tok.token_to_piece(token));
                State::Inside
            }
            State::Header(_) if token == marks.close => State::Outside,
            State::Header(_) if token == marks.open => State::Header(String::new()),
            State::Header(mut text) => match marks.header {
                Some(Header::Until { end, recipient }) => {
                    if token == end {
                        if text.trim_end().ends_with(recipient) {
                            self.pending = 0;
                            State::Inside
                        } else {
                            State::Outside
                        }
                    } else {
                        text.push_str(&String::from_utf8_lossy(self.tok.token_to_piece(token)));
                        if text.len() > MAX_HEADER {
                            State::Outside
                        } else {
                            State::Header(text)
                        }
                    }
                }
                Some(Header::Name(name)) => {
                    text.push_str(&String::from_utf8_lossy(self.tok.token_to_piece(token)));
                    if text.starts_with(name) {
                        self.pending = 0;
                        State::Inside
                    } else if name.starts_with(text.as_str()) {
                        State::Header(text)
                    } else {
                        State::Outside
                    }
                }
                None => State::Outside,
            },
        };
    }

    /// Ends `drafts` where a token of a step that verifies them would be replaced: the step's
    /// last token (the one after the drafts kept) is then the first replaced one.
    pub(super) fn cut(&self, drafts: &mut Vec<u32>) {
        if self.forced().is_some() {
            drafts.clear();
            return;
        }
        // Outside, only an opener among the drafts can start the reasoning.
        if self.state == State::Outside && !drafts.contains(&self.marks.open) {
            return;
        }
        let mut walk = self.clone();
        for (i, &t) in drafts.iter().enumerate() {
            walk.accept(t);
            if walk.forced().is_some() {
                drafts.truncate(i + 1);
                return;
            }
        }
    }
}

/// The bytes still to come of the last character after `piece`, `pending` before it.
fn pending_after(mut pending: u8, piece: &[u8]) -> u8 {
    for &b in piece {
        pending = if pending > 0 && b & 0xC0 == 0x80 {
            pending - 1
        } else if b & 0xE0 == 0xC0 {
            1
        } else if b & 0xF0 == 0xE0 {
            2
        } else if b & 0xF8 == 0xF0 {
            3
        } else {
            0
        };
    }
    pending
}

/// How a chat template takes a request's `reasoning_effort` itself, if it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Switch {
    /// It reads `enable_thinking` (Qwen, GLM, Gemma 4) or `reasoning_effort` (Inkling-2):
    /// the effort goes there as before.
    Native,
    /// Muse Glimmer's `reasoning_strength`, `low` or `high` ([`strength`]).
    Strength,
    /// None: the effort becomes a budget ([`effort_budget`]).
    None,
}

impl Switch {
    /// The switch of the template `src`.
    pub(super) fn of(src: &str) -> Self {
        if src.contains("reasoning_strength") {
            Self::Strength
        } else if src.contains("enable_thinking") || src.contains("reasoning_effort") {
            Self::Native
        } else {
            Self::None
        }
    }
}

/// Muse Glimmer's `reasoning_strength` for an OpenAI effort.
pub(super) fn strength(effort: &str) -> Option<&'static str> {
    match effort {
        "none" | "minimal" | "low" => Some("low"),
        "medium" | "high" | "xhigh" | "max" => Some("high"),
        _ => None,
    }
}

/// The budget an OpenAI effort stands for, for templates without a switch of their own
/// (`None`: no budget).
pub(super) fn effort_budget(effort: &str) -> Option<u32> {
    match effort {
        "none" => Some(0),
        "minimal" => Some(256),
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => Some(16384),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
