//! Drains an engine stream into a complete reply, for non-streaming requests.

use futures::StreamExt;

use crate::engine::{FinishReason, TokenEvent, TokenLogprob, TokenStream, ToolCall, Usage};

#[derive(Debug, Default)]
pub struct Collected {
    pub text: String,
    pub reasoning: String,
    /// In output order, aligned with `text`; empty unless the request asked for logprobs.
    pub logprobs: Vec<TokenLogprob>,
    pub tool_calls: Vec<ToolCall>,
    pub finish: Option<FinishReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Returns the reply, or the engine's error message if it failed mid-generation.
pub async fn collect(mut stream: TokenStream) -> Result<Collected, String> {
    let mut out = Collected::default();
    while let Some(ev) = stream.next().await {
        match ev {
            TokenEvent::Started { prompt_tokens } => out.usage.prompt_tokens = prompt_tokens,
            TokenEvent::Text(t) => out.text.push_str(&t),
            TokenEvent::Logprobs(l) => out.logprobs.extend(l),
            TokenEvent::Reasoning(t) => out.reasoning.push_str(&t),
            TokenEvent::ToolCall(c) => out.tool_calls.push(c),
            TokenEvent::Finished {
                reason,
                stop_sequence,
                usage,
            } => {
                out.finish = Some(reason);
                out.stop_sequence = stop_sequence;
                out.usage = usage;
                break;
            }
            TokenEvent::Error(e) => return Err(e),
        }
    }
    if out.finish.is_none() {
        return Err("engine stream ended without finishing".into());
    }
    Ok(out)
}
