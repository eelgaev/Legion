//! Chat output conventions detected once from a model's vocabulary. Architecture ids do not
//! determine the template or tokenizer a checkpoint ships with.

use super::think_budget::Marks;
use crate::engine::Tool;
use crate::engine::reasoning::ReasoningParser;
use crate::engine::tool_parser::{
    Gemma4Parser, GlmParser, InklingParser, MuseParser, Qwen3CoderParser, ToolParser, inkling, muse,
};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub(super) struct ChatFormat {
    reasoning: (&'static str, &'static str),
    prompt_reasoning: PromptReasoning,
    /// `(tools, hand undeclared names to the client) -> parser` ([`ChatFormat::tool_parser`]).
    tools: fn(&[Tool], bool) -> Box<dyn ToolParser>,
    extra_stop_tokens: Vec<u32>,
    /// Tokens that never end a reply, even when the vocabulary names one its end-of-turn
    /// token (Muse Glimmer's `<|eom|>`, which ends one message of a turn).
    not_stop: Vec<u32>,
    /// The `<tool_call>` and `</tool_call>` tokens of a vocabulary whose tool calls are in the
    /// Qwen3-Coder format, which strict tools constrain; `None`: strict is not enforced.
    strict_tools: Option<(u32, u32)>,
    /// Control tokens the detokenizer prints so the parsers can read them (Inkling-2's and
    /// Muse Glimmer's message markup); empty for the other formats.
    shown: Vec<u32>,
    /// Every reply is framed in markup the tool parser takes out, so it runs without tools
    /// too (Inkling-2, Muse Glimmer).
    framed: bool,
    /// A Qwen model's `<think>` format ([`ChatFormat::for_arch`]): a reply after a think block
    /// the template closed has its content's leading whitespace trimmed
    /// ([`ChatFormat::trims_content`]).
    qwen_think: bool,
    /// The reasoning markers a thinking budget follows and the close it writes
    /// ([`super::think_budget`]); `None`: requests' budgets do nothing.
    think: Option<Marks>,
}

/// The architectures (`general.architecture`) whose replies after a closed think block have
/// their content's leading whitespace trimmed: the Qwen ones.
const QWEN_ARCHS: [&str; 3] = ["qwen35", "qwen35moe", "qwen4exp"];

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
        // Inkling-2: `<|message_model|><|content_thinking|>...<|end_message|>` then
        // `<|message_model|><|content_text|>...<|end_message|>` or a JSON tool call, all of
        // it control tokens.
        let shown: Vec<u32> = inkling::MARKUP
            .iter()
            .map(|t| tok.token_id(t))
            .collect::<Option<_>>()
            .unwrap_or_default();
        if !shown.is_empty() {
            return Self {
                reasoning: (inkling::THINKING, inkling::END),
                // The prompt may hold earlier turns' closed thinking, never an open one.
                prompt_reasoning: PromptReasoning::FromPrompt,
                tools: |tools, _| Box::new(InklingParser::new(tools)),
                extra_stop_tokens: ["<|endoftext|>"]
                    .into_iter()
                    .filter_map(|t| tok.token_id(t))
                    .collect(),
                not_stop: Vec::new(),
                strict_tools: None,
                shown,
                framed: true,
                qwen_think: false,
                think: Marks::tags(tok, inkling::THINKING, inkling::END, inkling::END),
            };
        }
        // Muse Glimmer: messages `<|start|>assistant to=RECIPIENT<|message|>...` ending in
        // `<|eom|>` (more of the turn follows) or `<|eot|>` (the turn's end). The parser takes
        // the reasoning (`to=self`) out too, so there is no reasoning parser.
        let markup: Option<Vec<u32>> = muse::MARKUP.iter().map(|t| tok.token_id(t)).collect();
        if let Some(mut shown) = markup {
            shown.extend(muse::ATEM.iter().filter_map(|t| tok.token_id(t)));
            return Self {
                reasoning: (" to=self<|message|>", muse::EOM),
                prompt_reasoning: PromptReasoning::Disabled,
                tools: |tools, _| Box::new(MuseParser::new(tools)),
                extra_stop_tokens: tok.token_id(muse::EOT).into_iter().collect(),
                not_stop: tok.token_id(muse::EOM).into_iter().collect(),
                strict_tools: None,
                shown,
                framed: true,
                qwen_think: false,
                think: Marks::messages(tok, muse::START, muse::MESSAGE, "to=self", muse::EOM),
            };
        }
        let gemma4 = ["<|channel>", "<channel|>", "<|tool_call>"]
            .iter()
            .all(|t| tok.token_id(t).is_some());
        // GLM (4.5 and later) writes its arguments as `<arg_key>`/`<arg_value>` pairs.
        let glm = ["<tool_call>", "<arg_key>", "<arg_value>"]
            .iter()
            .all(|t| tok.token_id(t).is_some());
        let tools: fn(&[Tool], bool) -> Box<dyn ToolParser> = if gemma4 {
            |tools, call_undeclared| Box::new(Gemma4Parser::new(tools, call_undeclared))
        } else if glm {
            |tools, call_undeclared| Box::new(GlmParser::new(tools, call_undeclared))
        } else {
            |tools, call_undeclared| Box::new(Qwen3CoderParser::new(tools, call_undeclared))
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
        // opener and GLM's `<|observation|>` (the turn of the tool's results, which ends a
        // reply that called tools) hand control to the caller just like an end-of-turn token.
        let extra_stop_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
            "<turn|>",
            "<|tool_response>",
            "<|observation|>",
        ]
        .into_iter()
        .filter_map(|t| tok.token_id(t))
        .collect();
        // Strict tools constrain the Qwen3-Coder format only.
        let strict_tools = if gemma4 || glm {
            None
        } else {
            tok.token_id("<tool_call>")
                .zip(tok.token_id("</tool_call>"))
        };
        // The close of a thinking budget, as the templates write a closed reasoning: Qwen's
        // `<think>\n...\n</think>\n\n`, GLM's `<think>...</think>`.
        let think = if gemma4 {
            Marks::channel(tok, "<|channel>", "thought", "<channel|>")
        } else if glm {
            Marks::tags(tok, "<think>", "</think>", "</think>")
        } else {
            Marks::tags(tok, "<think>", "</think>", "\n</think>\n\n")
        };
        Self {
            reasoning,
            prompt_reasoning,
            tools,
            extra_stop_tokens,
            not_stop: Vec::new(),
            strict_tools,
            shown: Vec::new(),
            framed: false,
            think,
            qwen_think: false,
        }
    }

    /// The markers a thinking budget follows in this format, if it has them.
    pub(super) fn think_marks(&self) -> Option<&Marks> {
        self.think.as_ref()
    }

    /// The format for a model of architecture `arch` (`general.architecture`): a Qwen model's
    /// `<think>` format trims a reply's leading whitespace after a closed think block
    /// ([`Self::trims_content`]).
    pub(super) fn for_arch(mut self, arch: Option<&str>) -> Self {
        self.qwen_think = arch.is_some_and(|a| QWEN_ARCHS.contains(&a))
            && self.reasoning.0 == "<think>"
            && matches!(self.prompt_reasoning, PromptReasoning::FromPrompt)
            && !self.framed;
        self
    }

    /// Whether the reply to `prompt` (the rendered chat) has its content's leading whitespace
    /// trimmed, as llama.cpp's chat parser does: a Qwen model's reply right after a think block
    /// the template closed (thinking off), which starts with the model's ` word` token.
    /// Reasoning is never trimmed, nor another model's reply.
    pub(super) fn trims_content(&self, prompt: &str) -> bool {
        self.qwen_think && prompt.trim_end().ends_with(self.reasoning.1)
    }

    /// Control tokens the detokenizer must print for this format's parsers.
    pub(super) fn shown(&self) -> &[u32] {
        &self.shown
    }

    /// Whether the tool parser runs on every reply, tools or not (it also takes out the
    /// reply's framing).
    pub(super) fn framed(&self) -> bool {
        self.framed
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

    /// The reply's parser for `tools`; `call_undeclared` is `--undeclared-tool-calls call`.
    pub(super) fn tool_parser(&self, tools: &[Tool], call_undeclared: bool) -> Box<dyn ToolParser> {
        (self.tools)(tools, call_undeclared)
    }

    pub(super) fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        [tok.eos_id(), tok.eot_id()]
            .into_iter()
            .flatten()
            .chain(self.extra_stop_tokens.iter().copied())
            .filter(|t| !self.not_stop.contains(t))
            .collect()
    }
}

#[cfg(test)]
mod tests;
