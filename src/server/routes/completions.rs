//! `POST /v1/completions` (legacy text completions)

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
    CompletionChoice, CompletionLogprobs, CompletionRequest, CompletionResponse, Timings, Usage,
    finish_reason,
};
use crate::api::{new_id, unix_now};
use crate::engine::{TokenEvent, TokenLogprob};
use crate::server::AppState;
use crate::server::collect::collect;
use crate::server::convert;
use crate::server::error::{ApiError, ErrorKind, OpenAiError};
use crate::server::fair_share::{self, CallerKey};

pub async fn completions(
    State(state): State<AppState>,
    caller: Option<Extension<CallerKey>>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(req) = body?;
    let mut gen_req = convert::completion_request(&state, &req)?;
    gen_req.scheduling.share_key = fair_share::share_key(
        state.config.fair_share,
        caller.map(|Extension(c)| c),
        req.user.as_deref(),
    );
    let echo = match (&gen_req.prompt, req.echo) {
        (crate::engine::Prompt::Text(p), true) => p.clone(),
        _ => String::new(),
    };
    let want_logprobs = req.logprobs.is_some();
    // Character offset of the first generated token in the returned `text`.
    let mut text_offset = echo.chars().count();
    let model = state.resolve_model(&req.model)?.served_name;
    let guard = gen_req.cancel.clone().drop_guard();
    let tokens = state.engine.generate(gen_req)?;
    let id = new_id("cmpl-");
    let created = unix_now();

    if !req.stream {
        let c = collect(tokens).await.map_err(|message| ApiError {
            kind: ErrorKind::Internal,
            message,
            param: None,
        })?;
        drop(guard);
        let timings = Timings::from_usage(&c.usage);
        return Ok(Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text: echo + &c.text,
                finish_reason: c.finish.map(finish_reason),
                logprobs: want_logprobs
                    .then(|| CompletionLogprobs::new(&c.logprobs, &mut text_offset)),
            }],
            usage: Some(c.usage.into()),
            timings,
        })
        .into_response());
    }

    let mut st = CompletionStream {
        id,
        created,
        model,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
        want_logprobs,
        text_offset,
        pending_logprobs: Vec::new(),
        _guard: guard,
    };
    let first = (!echo.is_empty()).then(|| st.chunk(echo, None, None));
    let events = stream::iter(first)
        .chain(tokens.flat_map(move |ev| stream::iter(st.on_event(ev))))
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);
    Ok(Sse::new(events).into_response())
}

struct CompletionStream {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    want_logprobs: bool,
    /// Character offset in the streamed text of the next token's first character.
    text_offset: usize,
    /// Logprobs received but not yet sent: they ride on the `Text` chunk that follows them.
    pending_logprobs: Vec<TokenLogprob>,
    _guard: DropGuard,
}

impl CompletionStream {
    fn chunk(&self, text: String, finish: Option<&'static str>, usage: Option<Usage>) -> Event {
        self.chunk_with(text, finish, usage, None, None)
    }

    fn chunk_with(
        &self,
        text: String,
        finish: Option<&'static str>,
        usage: Option<Usage>,
        logprobs: Option<CompletionLogprobs>,
        timings: Option<Timings>,
    ) -> Event {
        json_event(&CompletionResponse {
            id: self.id.clone(),
            object: "text_completion",
            created: self.created,
            model: self.model.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: finish,
                logprobs,
            }],
            usage,
            timings,
        })
    }

    fn on_event(&mut self, ev: TokenEvent) -> Vec<Event> {
        match ev {
            TokenEvent::Logprobs(l) => {
                self.pending_logprobs.extend(l);
                vec![]
            }
            TokenEvent::Text(t) => {
                let taken = std::mem::take(&mut self.pending_logprobs);
                let logprobs = (self.want_logprobs && !taken.is_empty())
                    .then(|| CompletionLogprobs::new(&taken, &mut self.text_offset));
                vec![self.chunk_with(t, None, None, logprobs, None)]
            }
            TokenEvent::Finished { reason, usage, .. } => {
                // The final chunk carries `usage` (with `include_usage`) and the
                // finish reason, so the timings ride on it either way.
                let timings = Timings::from_usage(&usage);
                let usage = self.include_usage.then(|| Usage::from(usage));
                vec![self.chunk_with(
                    String::new(),
                    Some(finish_reason(reason)),
                    usage,
                    None,
                    timings,
                )]
            }
            TokenEvent::Error(message) => vec![json_event(
                &ApiError {
                    kind: ErrorKind::Internal,
                    message,
                    param: None,
                }
                .to_openai(),
            )],
            TokenEvent::Started { .. } | TokenEvent::Reasoning(_) | TokenEvent::ToolCall(_) => {
                vec![]
            }
        }
    }
}
