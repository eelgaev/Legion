//! Chat output conventions detected once from a model's vocabulary. Architecture ids do not
//! determine the template or tokenizer a checkpoint ships with.

use crate::engine::Tool;
use crate::engine::reasoning::ReasoningParser;
use crate::engine::tool_parser::{Gemma4Parser, Qwen3CoderParser, ToolParser};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub(super) struct ChatFormat {
    reasoning: (&'static str, &'static str),
    prompt_reasoning: PromptReasoning,
    tools: fn(&[Tool]) -> Box<dyn ToolParser>,
    extra_stop_tokens: Vec<u32>,
    /// The `<tool_call>` and `</tool_call>` tokens of a vocabulary whose tool calls are in the
    /// Qwen3-Coder format, which strict tools constrain; `None`: strict is not enforced.
    strict_tools: Option<(u32, u32)>,
}

#[derive(Clone, Copy)]
enum PromptReasoning {
    Disabled,
    /// The template may already have opened the reasoning span.
    FromPrompt,
    /// The model opens its reasoning channel during generation, except when the prompt ends
    /// with an open channel (Gemma 4 after a tool response with thinking on).
    OpenAtEnd,
}

impl ChatFormat {
    pub(super) fn detect(tok: &Tokenizer) -> Self {
        let gemma4 = ["<|channel>", "<channel|>", "<|tool_call>"]
            .iter()
            .all(|t| tok.token_id(t).is_some());
        let tools: fn(&[Tool]) -> Box<dyn ToolParser> = if gemma4 {
            |tools| Box::new(Gemma4Parser::new(tools))
        } else {
            |tools| Box::new(Qwen3CoderParser::new(tools))
        };
        let (reasoning, prompt_reasoning) = if gemma4 {
            // Gemma 4 opens its own thought channel; even a template with thinking off
            // closes an empty channel before generation starts. After a tool response with
            // thinking on, the template opens the channel itself.
            (
                ("<|channel>thought", "<channel|>"),
                PromptReasoning::OpenAtEnd,
            )
        } else {
            (
                ("<think>", "</think>"),
                if tok.token_id("<think>").is_some() {
                    PromptReasoning::FromPrompt
                } else {
                    PromptReasoning::Disabled
                },
            )
        };
        // Discover each end-of-generation token independently, as before, even when a
        // vocabulary lacks the full set of reasoning/tool markers. Gemma 4's tool-response
        // opener hands control to the caller just like its end-of-turn token does.
        let extra_stop_tokens = ["<|im_end|>", "<|endoftext|>", "<turn|>", "<|tool_response>"]
            .into_iter()
            .filter_map(|t| tok.token_id(t))
            .collect();
        let strict_tools = if gemma4 {
            None
        } else {
            tok.token_id("<tool_call>")
                .zip(tok.token_id("</tool_call>"))
        };
        Self {
            reasoning,
            prompt_reasoning,
            tools,
            extra_stop_tokens,
            strict_tools,
        }
    }

    /// The tokens that open and close a tool call, when strict tools are enforced for this
    /// vocabulary.
    pub(super) fn strict_tools(&self) -> Option<(u32, u32)> {
        self.strict_tools
    }

    pub(super) fn prompt_reasoning(&self, text: &str) -> Option<bool> {
        match self.prompt_reasoning {
            PromptReasoning::Disabled => None,
            PromptReasoning::OpenAtEnd => Some(text.trim_end().ends_with(self.reasoning.0)),
            PromptReasoning::FromPrompt => Some(
                match (text.rfind(self.reasoning.0), text.rfind(self.reasoning.1)) {
                    (Some(open), Some(close)) => open > close,
                    (Some(_), None) => true,
                    _ => false,
                },
            ),
        }
    }

    pub(super) fn reasoning_parser(&self, inside: bool) -> ReasoningParser {
        ReasoningParser::new(self.reasoning.0, self.reasoning.1, inside)
    }

    pub(super) fn tool_parser(&self, tools: &[Tool]) -> Box<dyn ToolParser> {
        (self.tools)(tools)
    }

    pub(super) fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        [tok.eos_id(), tok.eot_id()]
            .into_iter()
            .flatten()
            .chain(self.extra_stop_tokens.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests;
