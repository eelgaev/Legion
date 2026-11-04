//! Anthropic Messages API wire types (`POST /v1/messages`).
//!
//! Requests ignore unknown fields, and unknown content block types parse as
//! [`InputBlock::Unknown`], so newer clients keep working.

mod error;
mod messages;
mod stream;

pub use error::*;
pub use messages::*;
pub use stream::*;

/// Anthropic stop reasons.
pub fn stop_reason(r: crate::engine::FinishReason) -> &'static str {
    use crate::engine::FinishReason::*;
    match r {
        Stop | Cancelled => "end_turn",
        StopSequence => "stop_sequence",
        Length => "max_tokens",
        ToolCalls => "tool_use",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FinishReason;

    #[test]
    fn test_stop_reason() {
        assert_eq!(stop_reason(FinishReason::Stop), "end_turn");
        assert_eq!(stop_reason(FinishReason::Cancelled), "end_turn");
        assert_eq!(stop_reason(FinishReason::StopSequence), "stop_sequence");
        assert_eq!(stop_reason(FinishReason::Length), "max_tokens");
        assert_eq!(stop_reason(FinishReason::ToolCalls), "tool_use");
    }
}
