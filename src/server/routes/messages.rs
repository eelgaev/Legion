//! `POST /v1/messages`

use std::convert::Infallible;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures::{StreamExt, stream};
use tokio_util::sync::DropGuard;

use crate::api::anthropic::{
    BlockDelta, DeltaUsage, ErrorBody, MessageDeltaBody, MessageResponse, MessagesRequest,
    OutputBlock, StreamEvent, Usage, stop_reason,
};
use crate::api::new_id;
use crate::engine::{TokenEvent, ToolCall};
use crate::server::AppState;
use crate::server::collect::collect;
use crate::server::convert;
use crate::server::error::{AnthropicError, ApiError, ErrorKind};
use crate::server::fair_share::{self, CallerKey};

pub async fn messages(
    State(state): State<AppState>,
    caller: Option<Extension<CallerKey>>,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, AnthropicError> {
    let Json(req) = body?;
    let mut gen_req = convert::messages_request(&state, &req)?;
    gen_req.scheduling.share_key = fair_share::share_key(
        state.config.fair_share,
        caller.map(|Extension(c)| c),
        req.user_id(),
    );
    let model = state.resolve_model(&req.model)?.served_name;
    // Dropping the guard (client gone, handler future dropped) cancels the request.
    let guard = gen_req.cancel.clone().drop_guard();
    let tokens = state.engine.generate(gen_req)?;
    let id = new_id("msg_");

    if !req.stream {
        let c = collect(tokens).await.map_err(|message| ApiError {
            kind: ErrorKind::Internal,
            message,
            param: None,
        })?;
        drop(guard);
        let mut content = Vec::new();
        if !c.reasoning.is_empty() {
            content.push(OutputBlock::Thinking {
                thinking: c.reasoning,
                signature: String::new(),
            });
        }
        if !c.text.is_empty() || (content.is_empty() && c.tool_calls.is_empty()) {
            content.push(OutputBlock::Text { text: c.text });
        }
        content.extend(c.tool_calls.into_iter().map(|t| OutputBlock::ToolUse {
            input: parse_input(&t.arguments),
            id: t.id,
            name: t.name,
        }));
        let resp = MessageResponse {
            id,
            kind: "message".into(),
            role: "assistant".into(),
            content,
            model,
            stop_reason: c.finish.map(|r| stop_reason(r).to_string()),
            stop_sequence: c.stop_sequence,
            usage: Usage {
                input_tokens: c.usage.prompt_tokens,
                output_tokens: c.usage.completion_tokens,
                ..Default::default()
            },
        };
        return Ok(Json(resp).into_response());
    }

    let mut st = MessageStream {
        id,
        model,
        started: false,
        open: None,
        next_index: 0,
        _guard: guard,
    };
    let events = tokens
        .flat_map(move |ev| stream::iter(st.on_event(ev)))
        .map(Ok::<_, Infallible>);
    Ok(Sse::new(events).into_response())
}

fn parse_input(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
}

struct MessageStream {
    id: String,
    model: String,
    started: bool,
    /// The content block currently open, with its index.
    open: Option<(BlockKind, u32)>,
    next_index: u32,
    _guard: DropGuard,
}

fn sse(ev: &StreamEvent) -> Event {
    Event::default()
        .event(ev.name())
        .json_data(ev)
        .expect("API types always serialize")
}

impl MessageStream {
    fn start(&mut self, prompt_tokens: u32, out: &mut Vec<Event>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(sse(&StreamEvent::MessageStart {
            message: MessageResponse {
                id: self.id.clone(),
                kind: "message".into(),
                role: "assistant".into(),
                content: vec![],
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage {
                    input_tokens: prompt_tokens,
                    ..Default::default()
                },
            },
        }));
        out.push(sse(&StreamEvent::Ping));
    }

    fn close(&mut self, out: &mut Vec<Event>) {
        if let Some((_, index)) = self.open.take() {
            out.push(sse(&StreamEvent::ContentBlockStop { index }));
        }
    }

    /// Opens a block of `kind` unless one is already open, and returns its index.
    fn block(&mut self, kind: BlockKind, out: &mut Vec<Event>) -> u32 {
        if let Some((open, index)) = self.open {
            if open == kind {
                return index;
            }
            self.close(out);
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some((kind, index));
        let content_block = match kind {
            BlockKind::Thinking => OutputBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            BlockKind::Text => OutputBlock::Text {
                text: String::new(),
            },
        };
        out.push(sse(&StreamEvent::ContentBlockStart {
            index,
            content_block,
        }));
        index
    }

    fn tool_call(&mut self, c: ToolCall, out: &mut Vec<Event>) {
        self.close(out);
        let index = self.next_index;
        self.next_index += 1;
        out.push(sse(&StreamEvent::ContentBlockStart {
            index,
            content_block: OutputBlock::ToolUse {
                id: c.id,
                name: c.name,
                input: serde_json::json!({}),
            },
        }));
        out.push(sse(&StreamEvent::ContentBlockDelta {
            index,
            delta: BlockDelta::InputJsonDelta {
                partial_json: c.arguments,
            },
        }));
        out.push(sse(&StreamEvent::ContentBlockStop { index }));
    }

    fn on_event(&mut self, ev: TokenEvent) -> Vec<Event> {
        let mut out = Vec::new();
        if let TokenEvent::Started { prompt_tokens } = ev {
            self.start(prompt_tokens, &mut out);
            return out;
        }
        self.start(0, &mut out);
        match ev {
            TokenEvent::Started { .. } | TokenEvent::Logprobs(_) => {}
            TokenEvent::Text(text) => {
                let index = self.block(BlockKind::Text, &mut out);
                out.push(sse(&StreamEvent::ContentBlockDelta {
                    index,
                    delta: BlockDelta::TextDelta { text },
                }));
            }
            TokenEvent::Reasoning(thinking) => {
                let index = self.block(BlockKind::Thinking, &mut out);
                out.push(sse(&StreamEvent::ContentBlockDelta {
                    index,
                    delta: BlockDelta::ThinkingDelta { thinking },
                }));
            }
            TokenEvent::ToolCall(c) => self.tool_call(c, &mut out),
            TokenEvent::Finished {
                reason,
                stop_sequence,
                usage,
            } => {
                self.close(&mut out);
                out.push(sse(&StreamEvent::MessageDelta {
                    delta: MessageDeltaBody {
                        stop_reason: Some(stop_reason(reason).to_string()),
                        stop_sequence,
                    },
                    usage: DeltaUsage {
                        output_tokens: usage.completion_tokens,
                    },
                }));
                out.push(sse(&StreamEvent::MessageStop));
            }
            TokenEvent::Error(message) => {
                self.close(&mut out);
                out.push(sse(&StreamEvent::Error {
                    error: ErrorBody {
                        kind: "api_error".into(),
                        message,
                        param: None,
                    },
                }));
            }
        }
        out
    }
}
