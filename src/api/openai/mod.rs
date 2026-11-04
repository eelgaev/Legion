//! OpenAI-compatible wire types: chat completions, completions, models, errors.
//!
//! Requests ignore unknown fields so that clients sending newer or vendor-specific
//! parameters still work.

mod chat;
mod completions;
mod error;
mod logprobs;
mod models;
mod responses;

pub use chat::*;
pub use completions::*;
pub use error::*;
pub use logprobs::*;
pub use models::*;
pub use responses::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

impl From<crate::engine::Usage> for Usage {
    fn from(u: crate::engine::Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.prompt_tokens + u.completion_tokens,
            prompt_tokens_details: (u.cached_tokens > 0).then_some(PromptTokensDetails {
                cached_tokens: u.cached_tokens,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// llama.cpp-style `timings` object: server-measured prompt-processing (prefill) and
/// generation (decode) speed. Attached to OpenAI chat/completion responses when the
/// engine reports them (see `crate::engine::Usage`).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Timings {
    pub prompt_n: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_ms: Option<f64>,
    /// Omitted when `prompt_ms` is missing or zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    pub predicted_n: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_ms: Option<f64>,
    /// Omitted when `predicted_ms` is missing or zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
    /// Speculative decoding (omitted when off): draft tokens verified (llama.cpp's name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_n: Option<u64>,
    /// Of those, accepted (llama.cpp's name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_n_accepted: Option<u64>,
    /// `draft_n_accepted / draft_n`; omitted before any draft.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_acceptance_rate: Option<f64>,
    /// Tokens appended per verify step, the bonus token included; omitted before any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_mean_accepted_length: Option<f64>,
    /// Time the request waited in the engine's queue for a batch slot (Legion's own field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
}

impl Timings {
    /// The `timings` object for an engine usage, or `None` when the engine measured
    /// neither prefill nor decode and did not speculate.
    pub fn from_usage(usage: &crate::engine::Usage) -> Option<Self> {
        if usage.prefill_ms.is_none() && usage.decode_ms.is_none() && usage.spec.is_none() {
            return None;
        }
        let spec = usage.spec;
        Some(Self {
            prompt_n: usage.prompt_tokens,
            prompt_ms: usage.prefill_ms,
            prompt_per_second: per_second(usage.prompt_tokens, usage.prefill_ms),
            predicted_n: usage.completion_tokens,
            predicted_ms: usage.decode_ms,
            predicted_per_second: per_second(usage.completion_tokens, usage.decode_ms),
            draft_n: spec.map(|s| s.drafted),
            draft_n_accepted: spec.map(|s| s.accepted),
            draft_acceptance_rate: spec.and_then(|s| s.acceptance_rate()),
            draft_mean_accepted_length: spec.and_then(|s| s.mean_accepted_length()),
            queue_ms: usage.queue_ms,
        })
    }
}

/// Tokens per second from an elapsed wall time in ms; `None` when the time is missing or 0.
fn per_second(tokens: u32, ms: Option<f64>) -> Option<f64> {
    ms.filter(|ms| *ms != 0.0)
        .map(|ms| tokens as f64 / ms * 1000.0)
}

/// OpenAI finish reasons.
pub fn finish_reason(r: crate::engine::FinishReason) -> &'static str {
    use crate::engine::FinishReason::*;
    match r {
        Stop | StopSequence | Cancelled => "stop",
        Length => "length",
        ToolCalls => "tool_calls",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FinishReason;

    #[test]
    fn test_finish_reason() {
        assert_eq!(finish_reason(FinishReason::Stop), "stop");
        assert_eq!(finish_reason(FinishReason::StopSequence), "stop");
        assert_eq!(finish_reason(FinishReason::Cancelled), "stop");
        assert_eq!(finish_reason(FinishReason::Length), "length");
        assert_eq!(finish_reason(FinishReason::ToolCalls), "tool_calls");
    }

    #[test]
    fn timings_omits_per_second_when_ms_zero_or_missing() {
        let usage = crate::engine::Usage {
            prompt_tokens: 4,
            completion_tokens: 6,
            prefill_ms: Some(0.0),
            decode_ms: Some(0.0),
            ..Default::default()
        };
        let t = serde_json::to_value(Timings::from_usage(&usage).unwrap()).unwrap();
        assert_eq!(t["prompt_n"], 4);
        assert_eq!(t["prompt_ms"], 0.0);
        assert!(t.get("prompt_per_second").is_none(), "{t}");
        assert_eq!(t["predicted_n"], 6);
        assert_eq!(t["predicted_ms"], 0.0);
        assert!(t.get("predicted_per_second").is_none(), "{t}");

        // Only one side measured: the other side's ms and per-second keys are absent.
        let usage = crate::engine::Usage {
            prompt_tokens: 4,
            completion_tokens: 6,
            prefill_ms: Some(2.0),
            decode_ms: None,
            ..Default::default()
        };
        let t = serde_json::to_value(Timings::from_usage(&usage).unwrap()).unwrap();
        assert_eq!(t["prompt_per_second"].as_f64().unwrap(), 2000.0);
        assert!(t.get("predicted_ms").is_none(), "{t}");
        assert!(t.get("predicted_per_second").is_none(), "{t}");
    }

    #[test]
    fn timings_carry_speculation_counts() {
        let usage = crate::engine::Usage {
            prompt_tokens: 4,
            completion_tokens: 6,
            spec: Some(crate::spec::SpecCounters {
                drafted: 8,
                accepted: 2,
                steps: 2,
            }),
            ..Default::default()
        };
        let t = serde_json::to_value(Timings::from_usage(&usage).unwrap()).unwrap();
        assert_eq!(t["draft_n"], 8);
        assert_eq!(t["draft_n_accepted"], 2);
        assert_eq!(t["draft_acceptance_rate"], 0.25);
        assert_eq!(t["draft_mean_accepted_length"], 2.0);
        assert!(t.get("prompt_ms").is_none(), "{t}");
        // Speculation on, nothing drafted yet: counts, no ratios.
        let usage = crate::engine::Usage {
            spec: Some(Default::default()),
            ..Default::default()
        };
        let t = serde_json::to_value(Timings::from_usage(&usage).unwrap()).unwrap();
        assert_eq!(t["draft_n"], 0);
        assert!(t.get("draft_acceptance_rate").is_none(), "{t}");
    }

    #[test]
    fn timings_absent_when_engine_measured_nothing() {
        let usage = crate::engine::Usage {
            prompt_tokens: 4,
            completion_tokens: 6,
            ..Default::default()
        };
        assert!(Timings::from_usage(&usage).is_none());
    }
}
