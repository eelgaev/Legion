//! `--warmup-prompt` (#760): one synthetic prompt run before the server takes requests. A step
//! larger than any before it grows its buffers, each allocated anew at its full size: the
//! step's compute buffer (each kept prompt-step graph has its own; a new shape takes over the
//! oldest one's), ggml-cuda's pool and, under `--tp-rank`, the pinned exchange buffers, which
//! are registered with the NIC again. On two DGX Sparks (Qwen3.8-Flash-Next, steps of up to
//! 4,608 tokens) the first 8K prompt took 4.1 s and the next one 3.3 s, its first step ~0.6 s
//! slower; a later 9K prompt (steps of 4,544) paid it again. A warmup of three full steps
//! (3 x `--max-batch-tokens`) grows them all once; #758's warmup of 8,192 tokens (two steps of
//! 4,096) left the next 8K prompt's 4,160-token step to grow them again.
//!
//! `--warmup-shapes` (#836, on by default for the model engine): short prompts of a few sizes
//! first. On one DGX Spark the first prompt of a new short size paid a one-time ~130 ms burst
//! (lazy CUDA module loads of the kernels only small steps use, ggml-cuda pool growth).

use std::time::Instant;

use anyhow::Context as _;
use futures::StreamExt as _;
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};

/// A request for a prompt of `words` repeated words, greedy, one output token.
fn request(model: &str, words: usize) -> GenerateRequest {
    GenerateRequest {
        model: model.to_string(),
        prompt: Prompt::Text("the ".repeat(words)),
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: 1,
        stop: Vec::new(),
        ignore_eos: true,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: Default::default(),
        scheduling: Default::default(),
    }
}

/// Runs a prompt of about `tokens` tokens through `engine` and waits for it to finish. Returns
/// the prompt's token count.
pub(super) async fn run(engine: &dyn Engine, tokens: usize) -> anyhow::Result<u32> {
    let model = engine
        .models()
        .first()
        .map(|m| m.id.clone())
        .context("the engine serves no model")?;
    // Most tokenizers make one token of each repeated word; scale once if not.
    let mut req = request(&model, tokens);
    if let Ok(n) = engine.count_tokens(&req)
        && n > 0
        && n as usize != tokens
    {
        req = request(&model, (tokens * tokens / n as usize).max(1));
    }
    let started = Instant::now();
    let mut stream = engine.generate(req)?;
    let mut prompt_tokens = 0;
    while let Some(event) = stream.next().await {
        match event {
            TokenEvent::Started { prompt_tokens: n } => prompt_tokens = n,
            TokenEvent::Error(e) => anyhow::bail!("warmup prompt failed: {e}"),
            TokenEvent::Finished { .. } => break,
            _ => {}
        }
    }
    tracing::info!(
        prompt_tokens,
        seconds = started.elapsed().as_secs_f64(),
        "warmup prompt done"
    );
    Ok(prompt_tokens)
}

/// The `--warmup-shapes` sizes to run for a context of `context` tokens: in the given order,
/// without zeros, repeats and sizes above half the context (a warmup must never fail for lack
/// of room).
pub(super) fn shape_plan(sizes: &[usize], context: Option<u32>) -> Vec<usize> {
    let cap = context.map_or(usize::MAX, |c| c as usize / 2);
    let mut out: Vec<usize> = Vec::new();
    for &n in sizes {
        if n > 0 && n <= cap && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// `--warmup-shapes` (#836): one short prompt per size, so the first request of each size does
/// not pay the one-time kernel loads and buffer growth of its first prompt step. A failure only
/// logs a warning: the server starts either way.
pub(super) async fn shapes(engine: &dyn Engine, sizes: &[usize]) {
    let context = engine.models().first().and_then(|m| m.context_length);
    let plan = shape_plan(sizes, context);
    if plan.is_empty() {
        return;
    }
    let started = Instant::now();
    for n in plan.iter().copied() {
        if let Err(e) = run(engine, n).await {
            tracing::warn!(tokens = n, "warmup shape skipped: {e:#}");
        }
    }
    tracing::info!(
        sizes = ?plan,
        seconds = started.elapsed().as_secs_f64(),
        "warmup shapes done"
    );
}

#[cfg(test)]
mod tests;
