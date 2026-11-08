//! `legion run`: load a GGUF, generate once, print the answer (llama-cli style).

use std::io::{IsTerminal, Read, Write};
use std::time::Instant;

use anyhow::Context;
use futures::StreamExt;
use legion::engine::lm::{LmConfig, LmEngine};
use legion::engine::{
    Engine, FinishReason, GenerateRequest, Message, Prompt, Role, SamplingParams, TokenEvent, Usage,
};
use tokio_util::sync::CancellationToken;

use super::serve::ModelArgs;

#[derive(clap::Args)]
pub struct RunArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// System prompt (chat mode only).
    #[arg(long)]
    system: Option<String>,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Nucleus sampling threshold.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Sample from the k most likely tokens; 0 turns it off.
    #[arg(long, default_value_t = 20)]
    top_k: u32,
    /// Sampling seed (default: random).
    #[arg(long)]
    seed: Option<u64>,
    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    /// Let the model reason before answering; the reasoning goes to stderr.
    #[arg(long)]
    thinking: bool,
    /// Send PROMPT as is, without the chat template.
    #[arg(long)]
    raw: bool,
    /// Print prompt and output token counts and speeds on stderr when done.
    #[arg(long)]
    stats: bool,
    /// The prompt; read from stdin when omitted.
    prompt: Option<String>,
}

pub async fn run(args: RunArgs) -> anyhow::Result<()> {
    let m = args.model;
    let residency = m.residency();
    let model_path = m.resolve_model()?;
    anyhow::ensure!(m.ctx_size != Some(0), "--ctx-size must be at least 1");
    anyhow::ensure!(m.threads > 0, "--threads must be at least 1");
    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            if std::io::stdin().is_terminal() {
                eprintln!("reading the prompt from stdin (end with Ctrl-D)");
            }
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .context("reading the prompt from stdin")?;
            text.trim_end_matches('\n').to_string()
        }
    };

    let mut cfg = LmConfig::new(model_path, m.device.choice());
    cfg.ctx_size = m.ctx_size.unwrap_or(0);
    cfg.threads = m.threads;
    cfg.host_experts = m.host_experts;
    cfg.hf_fp8 = m.hf_fp8;
    cfg.nvfp4_act = m.nvfp4_act;
    cfg.fp8_act = m.fp8_act;
    cfg.nvfp4_mm = m.nvfp4_mm;
    cfg.fp16 = legion::weights::Fp16Load {
        mode: m.weights_fp16,
        gates: m.weights_fp16_gates,
        hc_inject_f32: m.hc_inject_f32,
    };
    cfg.nvfp4_a4 = m.nvfp4_a4;
    cfg.attention = m.attention;
    cfg.moe_kernel = m.moe_kernel;
    cfg.split_batches = m.split_batches;
    cfg.pipeline_groups = m.pipeline_groups;
    cfg.pipeline_min_group = m.pipeline_min_group;
    cfg.ngram = m.ngram_options()?;
    cfg.residency = residency;
    cfg.mmproj = m.mmproj.clone();
    cfg.gpus = m.gpus;
    cfg.tp = m.tp;
    cfg.tp_per_gpu = m.tp_per_gpu;
    cfg.layer_split = m.layer_split.clone();
    cfg.split_last_extra = m.split_last_extra;
    cfg.expert_mv = m.expert_mv == super::serve::ExpertMv::Legion;
    cfg.paged_prefill_m884 = m.paged_prefill_m884;
    cfg.glm_route_bias = m.glm_route_bias;
    cfg.glm_prune = m.glm_prune()?;
    cfg.switches = m.switches.switches();
    cfg.max_seqs = 1;
    let engine = tokio::task::spawn_blocking(move || LmEngine::load(cfg)).await??;
    let model = engine
        .models()
        .into_iter()
        .next()
        .map(|m| m.id)
        .context("the engine serves no model")?;

    let cancel = CancellationToken::new();
    let req = GenerateRequest {
        model,
        prompt: if args.raw {
            Prompt::Text(prompt)
        } else {
            Prompt::Chat {
                system: args.system,
                messages: vec![Message {
                    role: Some(Role::User),
                    content: prompt,
                    ..Default::default()
                }],
                tools: Vec::new(),
            }
        },
        sampling: SamplingParams {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
            ..Default::default()
        },
        max_tokens: args.max_tokens,
        stop: Vec::new(),
        ignore_eos: false,
        min_tokens: 0,
        thinking: Some(args.thinking),
        response_format: None,
        template_kwargs: Default::default(),
        cancel: cancel.clone(),
        scheduling: Default::default(),
    };

    let started = Instant::now();
    let mut stream = engine.generate(req)?;
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    let mut thinking_open = false;
    let mut first_output: Option<Instant> = None;
    let mut ctrl_c = std::pin::pin!(tokio::signal::ctrl_c());
    let usage: Usage = loop {
        let event = tokio::select! {
            _ = &mut ctrl_c => {
                // Let the engine free the sequence and shut down before exiting; exiting
                // under its running threads crashes in ggml's teardown.
                cancel.cancel();
                while stream.next().await.is_some() {}
                drop(stream);
                drop(engine);
                let _ = writeln!(out);
                let _ = out.flush();
                std::process::exit(130);
            }
            ev = stream.next() => ev,
        };
        match event {
            Some(TokenEvent::Reasoning(t)) => {
                first_output.get_or_insert_with(Instant::now);
                if !thinking_open {
                    thinking_open = true;
                    write!(err, "<think>")?;
                }
                write!(err, "{t}")?;
                err.flush()?;
            }
            Some(TokenEvent::Text(t)) => {
                first_output.get_or_insert_with(Instant::now);
                if thinking_open {
                    thinking_open = false;
                    writeln!(err, "</think>")?;
                }
                write!(out, "{t}")?;
                out.flush()?;
            }
            Some(TokenEvent::Finished { reason, usage, .. }) => {
                if thinking_open {
                    writeln!(err, "</think>")?;
                }
                if reason == FinishReason::Cancelled {
                    anyhow::bail!("generation was cancelled");
                }
                break usage;
            }
            Some(TokenEvent::Error(e)) => {
                let _ = writeln!(out);
                anyhow::bail!("generation failed: {e}");
            }
            Some(_) => {}
            None => anyhow::bail!("the engine ended the stream without finishing"),
        }
    };
    writeln!(out)?;
    if args.stats {
        let now = Instant::now();
        let first = first_output.unwrap_or(now);
        let wall = (
            first.duration_since(started).as_secs_f64() * 1e3,
            now.duration_since(first).as_secs_f64() * 1e3,
        );
        writeln!(err, "{}", stats_line(&usage, wall))?;
    }
    Ok(())
}

/// The `--stats` line. `wall` is (prefill ms, decode ms) measured here, used where the engine
/// reported none.
fn stats_line(usage: &Usage, wall: (f64, f64)) -> String {
    let prefill = usage.prefill_ms.unwrap_or(wall.0);
    let decode = usage.decode_ms.unwrap_or(wall.1);
    format!(
        "prompt {} tok ({:.1} tok/s) · output {} tok ({:.1} tok/s)",
        usage.prompt_tokens,
        rate(usage.prompt_tokens, prefill),
        usage.completion_tokens,
        rate(usage.completion_tokens, decode),
    )
}

fn rate(tokens: u32, ms: f64) -> f64 {
    if ms > 0.0 {
        f64::from(tokens) * 1e3 / ms
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_prefers_engine_timings() {
        let u = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            prefill_ms: Some(500.0),
            decode_ms: Some(1000.0),
            ..Default::default()
        };
        assert_eq!(
            stats_line(&u, (1.0, 1.0)),
            "prompt 10 tok (20.0 tok/s) · output 20 tok (20.0 tok/s)"
        );
    }

    #[test]
    fn stats_falls_back_to_wall_clock() {
        let u = Usage {
            prompt_tokens: 4,
            completion_tokens: 3,
            ..Default::default()
        };
        assert_eq!(
            stats_line(&u, (1000.0, 0.0)),
            "prompt 4 tok (4.0 tok/s) · output 3 tok (0.0 tok/s)"
        );
    }
}
