//! `POST /v1/responses` (stateless subset)

use std::convert::Infallible;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio_util::sync::DropGuard;

use crate::api::openai::{
    IncompleteDetails, OutputContent, OutputItem, OutputText, ReasoningOut, ResponsesObject,
    ResponsesRequest, ResponsesUsage,
};
use crate::api::{new_id, unix_now};
use crate::engine::{FinishReason, TokenEvent, ToolCall};
use crate::server::AppState;
use crate::server::collect::collect;
use crate::server::convert;
use crate::server::error::{ApiError, ErrorKind, OpenAiError};
use crate::server::fair_share::{self, CallerKey};

pub async fn responses(
    State(state): State<AppState>,
    caller: Option<Extension<CallerKey>>,
    body: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(req) = body?;
    let mut gen_req = convert::responses_request(&state, &req)?;
    gen_req.scheduling.share_key = fair_share::share_key(
        state.config.fair_share,
        caller.map(|Extension(c)| c),
        req.user.as_deref(),
    );
    let model = state.resolve_model(&req.model)?.served_name;
    // Dropping the guard (client gone, handler future dropped) cancels the request.
    let guard = gen_req.cancel.clone().drop_guard();
    let tokens = state.engine.generate(gen_req)?;
    let base = Base::new(&req, model);

    if !req.stream {
        let c = collect(tokens).await.map_err(|message| ApiError {
            kind: ErrorKind::Internal,
            message,
            param: None,
        })?;
        drop(guard);
        let mut output = Vec::new();
        if !c.reasoning.is_empty() {
            output.push(reasoning_item(new_id("rs_"), c.reasoning));
        }
        if !c.text.is_empty() || c.tool_calls.is_empty() {
            output.push(message_item(new_id("msg_"), c.text));
        }
        output.extend(c.tool_calls.into_iter().map(call_item));
        let finish = c.finish.unwrap_or(FinishReason::Stop);
        let resp = base.object(finish_status(finish), output, Some(c.usage.into()));
        return Ok(Json(resp).into_response());
    }

    let mut st = ResponsesStream {
        base,
        seq: 0,
        open: None,
        output: Vec::new(),
        _guard: guard,
    };
    let first = st.created();
    let events = stream::iter(first)
        .chain(tokens.flat_map(move |ev| stream::iter(st.on_event(ev))))
        .map(Ok::<_, Infallible>);
    Ok(Sse::new(events).into_response())
}

fn finish_status(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Length => "incomplete",
        _ => "completed",
    }
}

fn reasoning_item(id: String, text: String) -> OutputItem {
    OutputItem::Reasoning {
        id,
        summary: vec![],
        content: vec![OutputText::reasoning(text)],
        status: "completed",
    }
}

fn message_item(id: String, text: String) -> OutputItem {
    OutputItem::Message {
        id,
        role: "assistant",
        status: "completed",
        content: vec![OutputContent::text(text)],
    }
}

fn call_item(c: ToolCall) -> OutputItem {
    OutputItem::FunctionCall {
        id: new_id("fc_"),
        call_id: c.id,
        name: c.name,
        arguments: c.arguments,
        status: "completed",
    }
}

/// The request fields echoed back in every response object.
struct Base {
    id: String,
    created_at: u64,
    model: String,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    effort: Option<String>,
    tools: Vec<Value>,
    text_format: Value,
}

impl Base {
    fn new(req: &ResponsesRequest, model: String) -> Self {
        Self {
            id: new_id("resp_"),
            created_at: unix_now(),
            model,
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            effort: req.reasoning.as_ref().and_then(|r| r.effort.clone()),
            tools: req.tools.clone(),
            text_format: req
                .text
                .as_ref()
                .and_then(|t| t.get("format"))
                .cloned()
                .unwrap_or_else(|| json!({"type":"text"})),
        }
    }

    fn object(
        &self,
        status: &'static str,
        output: Vec<OutputItem>,
        usage: Option<ResponsesUsage>,
    ) -> ResponsesObject {
        ResponsesObject {
            id: self.id.clone(),
            object: "response",
            created_at: self.created_at,
            status,
            error: None,
            incomplete_details: (status == "incomplete").then_some(IncompleteDetails {
                reason: "max_output_tokens",
            }),
            instructions: self.instructions.clone(),
            max_output_tokens: self.max_output_tokens,
            model: self.model.clone(),
            output,
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: ReasoningOut {
                effort: self.effort.clone(),
                summary: None,
            },
            store: false,
            temperature: self.temperature,
            text: json!({"format": self.text_format}),
            tool_choice: "auto",
            tools: self.tools.clone(),
            top_p: self.top_p,
            truncation: "disabled",
            usage,
            user: None,
            metadata: json!({}),
        }
    }
}

/// The output item currently being streamed.
enum Open {
    Reasoning {
        id: String,
        index: u32,
        text: String,
    },
    Message {
        id: String,
        index: u32,
        text: String,
    },
}

struct ResponsesStream {
    base: Base,
    seq: u32,
    open: Option<Open>,
    /// Items already finished, in output order.
    output: Vec<OutputItem>,
    _guard: DropGuard,
}

impl ResponsesStream {
    /// Builds one SSE event: the `type` and `sequence_number` fields are added here.
    fn event(&mut self, kind: &str, mut data: Value) -> Event {
        let o = data.as_object_mut().expect("event data is an object");
        o.insert("type".into(), kind.into());
        o.insert("sequence_number".into(), self.seq.into());
        self.seq += 1;
        Event::default()
            .event(kind)
            .json_data(&data)
            .expect("API types always serialize")
    }

    fn created(&mut self) -> Vec<Event> {
        let resp = json!({"response": self.base.object("in_progress", vec![], None)});
        vec![
            self.event("response.created", resp.clone()),
            self.event("response.in_progress", resp),
        ]
    }

    fn next_index(&self) -> u32 {
        self.output.len() as u32
    }

    /// Finishes the open item, if any.
    fn close(&mut self, out: &mut Vec<Event>) {
        match self.open.take() {
            None => {}
            Some(Open::Reasoning { id, index, text }) => {
                out.push(self.event(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                ));
                let item = reasoning_item(id, text);
                out.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.output.push(item);
            }
            Some(Open::Message { id, index, text }) => {
                out.push(self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text, "logprobs": []}),
                ));
                out.push(self.event(
                    "response.content_part.done",
                    json!({
                        "item_id": id, "output_index": index, "content_index": 0,
                        "part": OutputContent::text(text.clone()),
                    }),
                ));
                let item = message_item(id, text);
                out.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.output.push(item);
            }
        }
    }

    fn open_reasoning(&mut self, out: &mut Vec<Event>) {
        let id = new_id("rs_");
        let index = self.next_index();
        let item = json!({"id": id, "type": "reasoning", "summary": [], "content": [], "status": "in_progress"});
        out.push(self.event(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        ));
        self.open = Some(Open::Reasoning {
            id,
            index,
            text: String::new(),
        });
    }

    fn open_message(&mut self, out: &mut Vec<Event>) {
        let id = new_id("msg_");
        let index = self.next_index();
        let item = json!({"id": id, "type": "message", "role": "assistant", "status": "in_progress", "content": []});
        out.push(self.event(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        ));
        out.push(self.event(
            "response.content_part.added",
            json!({
                "item_id": id, "output_index": index, "content_index": 0,
                "part": OutputContent::text(String::new()),
            }),
        ));
        self.open = Some(Open::Message {
            id,
            index,
            text: String::new(),
        });
    }

    fn on_event(&mut self, ev: TokenEvent) -> Vec<Event> {
        let mut out = Vec::new();
        match ev {
            // The Responses API ignores logprobs; the engine is never asked for them here.
            TokenEvent::Started { .. } | TokenEvent::Logprobs(_) => {}
            TokenEvent::Reasoning(t) => {
                if !matches!(self.open, Some(Open::Reasoning { .. })) {
                    self.close(&mut out);
                    self.open_reasoning(&mut out);
                }
                if let Some(Open::Reasoning { id, index, text }) = &mut self.open {
                    text.push_str(&t);
                    let data = json!({"item_id": id, "output_index": index, "content_index": 0, "delta": t});
                    out.push(self.event("response.reasoning_text.delta", data));
                }
            }
            TokenEvent::Text(t) => {
                if !matches!(self.open, Some(Open::Message { .. })) {
                    self.close(&mut out);
                    self.open_message(&mut out);
                }
                if let Some(Open::Message { id, index, text }) = &mut self.open {
                    text.push_str(&t);
                    let data = json!({
                        "item_id": id, "output_index": index, "content_index": 0,
                        "delta": t, "logprobs": [],
                    });
                    out.push(self.event("response.output_text.delta", data));
                }
            }
            TokenEvent::ToolCall(c) => {
                self.close(&mut out);
                let index = self.next_index();
                let item = call_item(c);
                let OutputItem::FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                    ..
                } = &item
                else {
                    unreachable!("call_item builds function calls");
                };
                let added = json!({
                    "id": id, "type": "function_call", "call_id": call_id, "name": name,
                    "arguments": "", "status": "in_progress",
                });
                let (id, arguments) = (id.clone(), arguments.clone());
                out.push(self.event(
                    "response.output_item.added",
                    json!({"output_index": index, "item": added}),
                ));
                out.push(self.event(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": index, "delta": arguments}),
                ));
                out.push(self.event(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": arguments}),
                ));
                out.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.output.push(item);
            }
            TokenEvent::Finished { reason, usage, .. } => {
                self.close(&mut out);
                // A reply with no output at all still has one (empty) message.
                if self.output.is_empty() {
                    self.open_message(&mut out);
                    self.close(&mut out);
                }
                let status = finish_status(reason);
                let resp =
                    self.base
                        .object(status, std::mem::take(&mut self.output), Some(usage.into()));
                let kind = if status == "completed" {
                    "response.completed"
                } else {
                    "response.incomplete"
                };
                out.push(self.event(kind, json!({"response": resp})));
            }
            TokenEvent::Error(message) => {
                let err = json!({"code": "server_error", "message": message});
                let mut resp = self.base.object("failed", vec![], None);
                resp.error = Some(err.clone());
                out.push(self.event("response.failed", json!({"response": resp})));
                out.push(self.event(
                    "error",
                    json!({"code": "server_error", "message": message, "param": null}),
                ));
            }
        }
        out
    }
}
