//! `legion bench --workload bandwidth` (#391): one prefill of about `--prompt-tokens` tokens,
//! then `--concurrency` parallel decodes of `--gen-tokens` tokens after the same prompt
//! (temperature 0, fixed length), so decode reads that context's KV (and a KV host tier's
//! share of it). Each phase runs between two `/metrics` scrapes of a Legion server. Prints the analytic
//! model bytes (device memory, host link, the device roofline's share) next to the GPUs'
//! hardware counters when the server runs with `--hw-counters`.

use std::time::{Instant, SystemTime};

use anyhow::bail;
use futures::StreamExt;
use legion::bench::bandwidth::{self, GpuCounters, PhaseBytes, Scrape};
use legion::bench::{self, BenchConfig};
use serde::Serialize;
use serde_json::json;

use super::save;
use super::workload::{SaveSpec, WorkloadArgs, prefill_prompt, write_save};

/// One measured phase.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseReport {
    /// Wall time from the first request sent to the last answer, seconds.
    pub wall_s: f64,
    /// Requests that failed (0 expected).
    pub failed: usize,
    /// Output tokens over the phase and their rate from the first token on (`wall_s` less
    /// `ttft_s`: a prompt the prefix cache missed is prefilled again before decoding).
    pub output_tokens: u64,
    pub output_tok_s: f64,
    /// Time to the first token of the phase's first request, seconds.
    pub ttft_s: Option<f64>,
    /// The analytic model bytes of the server's steps in the window: prefill steps for the
    /// prefill phase; decode/verify steps and draft passes for the decode phase.
    pub model: Vec<(String, PhaseBytes)>,
    /// The GPUs' hardware counters over the window (empty without `--hw-counters`).
    pub gpus: Vec<GpuCounters>,
}

/// `legion bench --workload bandwidth --json`.
#[derive(Debug, Clone, Serialize)]
pub struct BandwidthReport {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub concurrency: usize,
    /// The first stage's device-memory roofline the percentages use, GB/s (0: unknown).
    pub roofline_memory_gb_s: f64,
    pub prefill: PhaseReport,
    pub decode: PhaseReport,
}

async fn metrics(client: &reqwest::Client, cfg: &BenchConfig) -> anyhow::Result<Scrape> {
    let url = &cfg.url;
    match bandwidth::scrape(client, url, cfg.api_key.as_deref()).await {
        Some(s)
            if s.get("legion_engine_model_bytes", &[("phase", "decode")])
                .is_some() =>
        {
            Ok(s)
        }
        Some(_) => bail!("{url}/metrics has no legion_engine_model_bytes: not a Legion server?"),
        None => bail!("could not read {url}/metrics (with --api-key if the server needs one)"),
    }
}

/// Runs `prompts` with at most `concurrency` in flight, between two scrapes.
async fn phase(
    client: &reqwest::Client,
    cfg: &BenchConfig,
    prompts: Vec<String>,
    phases: &[&str],
) -> anyhow::Result<(PhaseReport, f64)> {
    let n = prompts.len();
    let before = metrics(client, cfg).await?;
    let start = Instant::now();
    let outcomes: Vec<bench::RequestOutcome> = futures::stream::iter(prompts)
        .map(|p| async move { bench::run_request(client, cfg, &p).await })
        .buffer_unordered(n.max(1))
        .collect()
        .await;
    let wall_s = start.elapsed().as_secs_f64();
    let after = metrics(client, cfg).await?;
    let model = bandwidth::model_bytes_between(&before, &after).unwrap_or_default();
    let pick = |name: &str| match name {
        "prefill" => model.prefill,
        "decode" => model.decode,
        _ => model.draft,
    };
    let output_tokens: u64 = outcomes.iter().map(|o| o.output_tokens).sum();
    let ttft_s = outcomes
        .iter()
        .filter_map(|o| o.ttft)
        .map(|t| t.as_secs_f64())
        .reduce(f64::min);
    let decode_s = wall_s - ttft_s.unwrap_or(0.0);
    if let Some(e) = outcomes.iter().find_map(|o| o.error.as_ref()) {
        eprintln!("  request failed: {e}");
    }
    Ok((
        PhaseReport {
            wall_s,
            failed: outcomes.iter().filter(|o| !o.ok).count(),
            output_tokens,
            output_tok_s: if decode_s > 0.0 {
                output_tokens as f64 / decode_s
            } else {
                0.0
            },
            ttft_s,
            model: phases.iter().map(|p| (p.to_string(), pick(p))).collect(),
            gpus: bandwidth::gpu_counters_between(&before, &after),
        },
        model.roofline_memory_gb_s,
    ))
}

pub async fn run(
    args: WorkloadArgs,
    prompt_tokens: usize,
    gen_tokens: usize,
    concurrency: usize,
) -> anyhow::Result<()> {
    if prompt_tokens == 0 || gen_tokens == 0 || concurrency == 0 {
        bail!("--workload bandwidth needs positive --prompt-tokens, --gen-tokens, --concurrency");
    }
    if let Some(dir) = &args.save {
        save::check_free(dir, args.force)?;
    }
    let date = save::rfc3339_utc(SystemTime::now());
    let client = bench::client()?;
    let cfg = |gen_tokens: usize| BenchConfig {
        url: args.url.clone(),
        model: args.model.clone(),
        api: args.api,
        prompt_tokens,
        gen_tokens,
        shared_prefix_tokens: 0,
        seed: 0,
        api_key: args.api_key.clone(),
        min_tokens: false,
        priority: None,
        declared_max_tokens: None,
    };
    let (prefill_cfg, decode_cfg) = (cfg(1), cfg(gen_tokens));
    // Warm-up (discarded): graphs, the expert cache's first routes.
    eprintln!("warm-up");
    let _ = phase(&client, &decode_cfg, vec!["Hello".into()], &[]).await?;
    eprintln!("prefill: about {prompt_tokens} prompt tokens, 1 output token");
    let (prefill, _) = phase(
        &client,
        &prefill_cfg,
        vec![prefill_prompt(prompt_tokens)],
        &["prefill"],
    )
    .await?;
    // The prefill's prompt again: the prefix cache (when on) skips its prefill, and each decode
    // step reads `prompt_tokens` of KV.
    eprintln!("decode: {concurrency} x {gen_tokens} tokens after the same prompt");
    let prompts = vec![prefill_prompt(prompt_tokens); concurrency];
    let (decode, roofline) = phase(&client, &decode_cfg, prompts, &["decode", "draft"]).await?;
    let report = BandwidthReport {
        prompt_tokens,
        gen_tokens,
        concurrency,
        roofline_memory_gb_s: roofline,
        prefill,
        decode,
    };
    let table = format_table(&report);
    let body = serde_json::to_string_pretty(&report)?;
    if args.json {
        println!("{body}");
    } else {
        print!("{table}");
    }
    if let Some(dir) = &args.save {
        let spec = SaveSpec {
            artifact: "bandwidth.json",
            body: &body,
            prompt_tokens: json!(prompt_tokens),
            gen_tokens,
            table: &format!("```\n{table}```\n"),
        };
        write_save(dir, &args, date, spec, &client).await?;
        eprintln!("saved to {}", dir.display());
    }
    Ok(())
}

/// The analytic figures per phase with each GPU's counters under them.
pub fn format_table(r: &BandwidthReport) -> String {
    let mut out = format!(
        "prefill: {:.3} s wall, TTFT {:.3} s; decode: {} x {} tokens, TTFT {:.3} s, then \
         {:.1} tok/s; device roofline {}\n",
        r.prefill.wall_s,
        r.prefill.ttft_s.unwrap_or(0.0),
        r.concurrency,
        r.gen_tokens,
        r.decode.ttft_s.unwrap_or(0.0),
        r.decode.output_tok_s,
        if r.roofline_memory_gb_s > 0.0 {
            format!("{:.0} GB/s", r.roofline_memory_gb_s)
        } else {
            "unknown".into()
        }
    );
    out.push_str(&bandwidth::table_header());
    for p in [&r.prefill, &r.decode] {
        for (i, (name, bytes)) in p.model.iter().enumerate() {
            // The hardware counters cover the whole window: under its last row.
            let gpus: &[GpuCounters] = if i + 1 == p.model.len() { &p.gpus } else { &[] };
            out.push_str(&bandwidth::format_phase(name, bytes, gpus));
        }
    }
    out
}

#[cfg(test)]
mod tests;
