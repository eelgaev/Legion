//! `POST /v1/chat/completions`

use std::convert::Infallible;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures::{StreamExt, stream};
use tokio_util::sync::DropGuard;

use super::json_event;
use crate::api::openai::{
    AssistantMessage, ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatLogprobs, ChunkChoice, Delta, FunctionCall, FunctionCallDelta,
    Timings, ToolCall, ToolCallDelta, Usage, finish_reason,
};
use crate::api::{new_id, unix_now};
use crate::engine::{TokenEvent, TokenLogprob, ToolCall as EngineToolCall};
use crate::server::AppState;
use crate::server::collect::collect;
use crate::server::convert;
use crate::server::error::{ApiError, ErrorKind, OpenAiError};
use crate::server::fair_share::{self, CallerKey};

pub async fn chat_completions(
    State(state): State<AppState>,
    caller: Option<Extension<CallerKey>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(req) = body?;
    let mut gen_req = convert::chat_request(&state, &req)?;
    gen_req.scheduling.share_key = fair_share::share_key(
        state.config.fair_share,
        caller.map(|Extension(c)| c),
        req.user.as_deref(),
    );
    let model = state.resolve_model(&req.model)?.served_name;
    // Dropping the guard (client gone, handler future dropped) cancels the request.
    let guard = gen_req.cancel.clone().drop_guard();
    let tokens = state.engine.generate(gen_req)?;
    let id = new_id("chatcmpl-");
    let created = unix_now();

    if !req.stream {
        let c = collect(tokens).await.map_err(|message| ApiError {
            kind: ErrorKind::Internal,
            message,
            param: None,
        })?;
        drop(guard);
        let timings = Timings::from_usage(&c.usage);
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some(c.text),
                    reasoning_content: (!c.reasoning.is_empty()).then_some(c.reasoning),
                    tool_calls: c.tool_calls.into_iter().map(tool_call).collect(),
                },
                finish_reason: c.finish.map(finish_reason),
                logprobs: req.logprobs.then(|| ChatLogprobs::new(&c.logprobs)),
            }],
            usage: c.usage.into(),
            timings,
        };
        return Ok(Json(resp).into_response());
    }

    let mut st = ChatStream {
        id,
        created,
        model,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
        tool_calls: 0,
        want_logprobs: req.logprobs,
        pending_logprobs: Vec::new(),
        _guard: guard,
    };
    let first = st.chunk(
        Delta {
            role: Some("assistant"),
            content: Some(String::new()),
            ..Default::default()
        },
        None,
    );
    let events = stream::once(async move { first })
        .chain(tokens.flat_map(move |ev| stream::iter(st.on_event(ev))))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);
    Ok(Sse::new(events).into_response())
}

fn tool_call(c: EngineToolCall) -> ToolCall {
    ToolCall {
        id: c.id,
        kind: "function".into(),
        function: FunctionCall {
            name: c.name,
            arguments: c.arguments,
        },
    }
}

struct ChatStream {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    tool_calls: u32,
    want_logprobs: bool,
    /// Logprobs received but not yet sent: they ride on the `Text` chunk that follows them.
    pending_logprobs: Vec<TokenLogprob>,
    _guard: DropGuard,
}

impl ChatStream {
    fn chunk(&self, delta: Delta, finish: Option<&'static str>) -> Event {
        self.chunk_with(delta, finish, None, None)
    }

    /// The pending logprobs as a chunk's `logprobs`, if the request asked for them.
    fn take_logprobs(&mut self) -> Option<ChatLogprobs> {
        let taken = std::mem::take(&mut self.pending_logprobs);
        (self.want_logprobs && !taken.is_empty()).then(|| ChatLogprobs::new(&taken))
    }

    fn chunk_with(
        &self,
        delta: Delta,
        finish: Option<&'static str>,
        logprobs: Option<ChatLogprobs>,
        timings: Option<Timings>,
    ) -> Event {
        json_event(&ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish,
                logprobs,
            }],
            usage: None,
            timings,
        })
    }

    fn on_event(&mut self, ev: TokenEvent) -> Vec<Event> {
        match ev {
            TokenEvent::Started { .. } => vec![],
            TokenEvent::Logprobs(l) => {
                self.pending_logprobs.extend(l);
                vec![]
            }
            TokenEvent::Text(t) => {
                let logprobs = self.take_logprobs();
                vec![self.chunk_with(
                    Delta {
                        content: Some(t),
                        ..Default::default()
                    },
                    None,
                    logprobs,
                    None,
                )]
            }
            TokenEvent::Reasoning(t) => vec![self.chunk(
                Delta {
                    reasoning_content: Some(t),
                    ..Default::default()
                },
                None,
            )],
            TokenEvent::ToolCall(c) => {
                let index = self.tool_calls;
                self.tool_calls += 1;
                vec![self.chunk(
                    Delta {
                        tool_calls: Some(vec![ToolCallDelta {
                            index,
                            id: Some(c.id),
                            kind: Some("function"),
                            function: FunctionCallDelta {
                                name: Some(c.name),
                                arguments: c.arguments,
                            },
                        }]),
                        ..Default::default()
                    },
                    None,
                )]
            }
            TokenEvent::Finished { reason, usage, .. } => {
                let logprobs = self.take_logprobs();
                let timings = Timings::from_usage(&usage);
                // With `include_usage` the timings ride on the usage chunk; without it,
                // on the finish chunk.
                let finish_timings = if self.include_usage { None } else { timings };
                let mut out = vec![self.chunk_with(
                    Delta::default(),
                    Some(finish_reason(reason)),
                    logprobs,
                    finish_timings,
                )];
                if self.include_usage {
                    out.push(json_event(&ChatCompletionChunk {
                        id: self.id.clone(),
                        object: "chat.completion.chunk",
                        created: self.created,
                        model: self.model.clone(),
                        choices: vec![],
                        usage: Some(Usage::from(usage)),
                        timings,
                    }));
                }
                out
            }
            TokenEvent::Error(message) => vec![json_event(
                &ApiError {
                    kind: ErrorKind::Internal,
                    message,
                    param: None,
                }
                .to_openai(),
            )],
        }
    }
}
