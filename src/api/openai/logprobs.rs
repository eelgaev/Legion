//! Log probability wire types: the chat `logprobs.content[]` shape and the legacy completions
//! `{tokens, token_logprobs, top_logprobs, text_offset}` shape.

use serde::Serialize;

use crate::engine::TokenLogprob;

/// Chat `choices[].logprobs`, on the response and on stream chunks.
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatTokenLogprob>,
    /// Always `null`: there are no refusals.
    pub refusal: Option<Vec<ChatTokenLogprob>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<ChatTopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

impl ChatLogprobs {
    pub fn new(tokens: &[TokenLogprob]) -> Self {
        Self {
            content: tokens
                .iter()
                .map(|t| ChatTokenLogprob {
                    token: t.token.clone(),
                    logprob: t.logprob,
                    bytes: t.bytes.clone(),
                    top_logprobs: t
                        .top
                        .iter()
                        .map(|(token, logprob)| ChatTopLogprob {
                            token: token.clone(),
                            logprob: *logprob,
                            bytes: token.as_bytes().to_vec(),
                        })
                        .collect(),
                })
                .collect(),
            refusal: None,
        }
    }
}

/// Legacy completions `choices[].logprobs`.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    /// One `{token: logprob}` object per position.
    pub top_logprobs: Vec<serde_json::Map<String, serde_json::Value>>,
    /// Character offset of each token in the returned `text`.
    pub text_offset: Vec<usize>,
}

impl CompletionLogprobs {
    /// `offset` is the character offset of the first token; it advances by each token's length.
    /// A position without alternatives (`logprobs: 0`) lists the chosen token alone.
    pub fn new(tokens: &[TokenLogprob], offset: &mut usize) -> Self {
        let mut out = Self {
            tokens: Vec::with_capacity(tokens.len()),
            token_logprobs: Vec::with_capacity(tokens.len()),
            top_logprobs: Vec::with_capacity(tokens.len()),
            text_offset: Vec::with_capacity(tokens.len()),
        };
        for t in tokens {
            out.tokens.push(t.token.clone());
            out.token_logprobs.push(t.logprob);
            out.text_offset.push(*offset);
            *offset += t.token.chars().count();
            let mut top = serde_json::Map::new();
            if t.top.is_empty() {
                top.insert(t.token.clone(), t.logprob.into());
            }
            for (token, lp) in &t.top {
                top.insert(token.clone(), (*lp).into());
            }
            out.top_logprobs.push(top);
        }
        out
    }

    /// Appends `more` (as built by [`CompletionLogprobs::new`]).
    pub fn extend(&mut self, more: CompletionLogprobs) {
        self.tokens.extend(more.tokens);
        self.token_logprobs.extend(more.token_logprobs);
        self.top_logprobs.extend(more.top_logprobs);
        self.text_offset.extend(more.text_offset);
    }
}
