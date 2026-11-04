//! `--pipeline-mtp-split` (#765): load, measure the decoding groups' rounds on a short synthetic
//! run, and load again with the layer split the measurement suggests
//! ([`legion::engine::lm::pipe_split`]) when it differs from the one loaded.

use std::time::Instant;

use anyhow::Context as _;
use futures::StreamExt as _;
use legion::engine::lm::{LmConfig, LmEngine, drafts_with_mtp_head};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};

/// Tokens each synthetic sequence decodes: with MTP k=2 about 50 steps each, a few hundred
/// rounds for four groups.
const TOKENS: u32 = 128;

/// Synthetic runs before giving up on a measurement.
const TRIES: usize = 3;

/// Prompts of ordinary text, so routing (and the expert cache) behave as on real requests.
const PROMPTS: &[&str] = &[
    "Explain how a refrigerator moves heat from its inside to the room, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the main differences between TCP and UDP and when each is used.",
    "Summarize the causes and consequences of the French Revolution.",
    "Give a recipe for a simple vegetable soup and explain each step.",
    "What are the trade-offs between renting and buying a home? Discuss both sides.",
    "Explain the idea behind binary search and write it in Python.",
    "Describe a walk through a forest in autumn, with attention to sounds and smells.",
];

/// Why `cfg` does not measure a split (none when it does).
pub(super) fn skip_reason(cfg: &LmConfig) -> Option<&'static str> {
    if cfg.pipeline_groups < 2 {
        Some("needs --pipeline-groups 2 or more")
    } else if !drafts_with_mtp_head(&cfg.spec) {
        Some("needs the MTP head (--spec mtp)")
    } else if !cfg.layer_split.is_empty() {
        Some("--layer-split is given")
    } else if cfg.node.is_some() || cfg.tp > 1 || cfg.tp_net.is_some() {
        Some("not with --node-rank or tensor parallelism")
    } else if cfg.gpus < 2 {
        Some("needs a layer split over several GPUs (--gpus)")
    } else {
        None
    }
}

/// A greedy request decoding `max_tokens` tokens of prompt `i`, whatever it samples.
fn request(model: &str, i: usize, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model: model.to_string(),
        prompt: Prompt::Text(PROMPTS[i % PROMPTS.len()].to_string()),
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens,
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

/// Runs `n` requests of `max_tokens` tokens at once on `engine` and waits for all.
pub(super) async fn run_batch(
    engine: &dyn Engine,
    n: usize,
    max_tokens: u32,
) -> anyhow::Result<()> {
    let model = engine
        .models()
        .first()
        .map(|m| m.id.clone())
        .context("the engine serves no model")?;
    let streams = (0..n)
        .map(|i| engine.generate(request(&model, i, max_tokens)))
        .collect::<Result<Vec<_>, _>>()?;
    let finish = streams.into_iter().map(|mut s| async move {
        while let Some(event) = s.next().await {
            match event {
                TokenEvent::Error(e) => anyhow::bail!("measuring run failed: {e}"),
                TokenEvent::Finished { .. } => break,
                _ => {}
            }
        }
        Ok(())
    });
    for r in futures::future::join_all(finish).await {
        r?;
    }
    Ok(())
}

async fn load_once(cfg: LmConfig) -> anyhow::Result<LmEngine> {
    Ok(tokio::task::spawn_blocking(move || LmEngine::load(cfg)).await??)
}

/// Loads the model for `cfg`; with `--pipeline-mtp-split` (where it applies), measures the
/// groups' rounds and loads again with the split they suggest.
pub(super) async fn load(cfg: LmConfig) -> anyhow::Result<LmEngine> {
    if !cfg.pipeline_mtp_split {
        return load_once(cfg).await;
    }
    if let Some(why) = skip_reason(&cfg) {
        tracing::warn!("--pipeline-mtp-split: {why}; the layer split stays as it is");
        return load_once(cfg).await;
    }
    let engine = load_once(cfg.clone()).await?;
    // Enough sequences that every group forms.
    let seqs = (cfg.pipeline_groups * cfg.pipeline_min_group.max(1)).min(cfg.max_seqs);
    let started = Instant::now();
    let mut hint = None;
    for _ in 0..TRIES {
        run_batch(&engine, seqs, TOKENS).await?;
        hint = engine.mtp_split_hint();
        if hint.is_some() {
            break;
        }
    }
    let Some(h) = hint else {
        tracing::warn!(
            seqs,
            "--pipeline-mtp-split: too few rounds with every stage busy to measure; the layer \
             split stays as it is"
        );
        return Ok(engine);
    };
    tracing::info!(
        rounds = h.rounds,
        layer_ms = h.layer_ms,
        last_stage_extra_ms = h.extra_ms,
        extra_layers = h.extra_layers,
        current = ?h.current,
        split = ?h.cuts,
        seconds = started.elapsed().as_secs_f64(),
        "--pipeline-mtp-split: measured the decoding groups' rounds"
    );
    if h.cuts == h.current {
        return Ok(engine);
    }
    // The worker frees the devices when the engine drops (it joins its thread).
    tokio::task::spawn_blocking(move || drop(engine)).await?;
    let mut cfg = cfg;
    cfg.layer_split = h.cuts.clone();
    tracing::info!(split = ?h.cuts, "--pipeline-mtp-split: loading again with the split");
    load_once(cfg).await
}

#[cfg(test)]
mod tests;
