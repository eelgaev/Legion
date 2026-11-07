//! `legion serve --tui` (#1198): the dashboard in the server's terminal, and the header every
//! stats client sees (`GET /v1/stats`: model, quant, layout, context, sequences, version).

use std::io::IsTerminal;
use std::time::{Duration, Instant};

use legion::dashboard::logs;
use legion::dashboard::snapshot::{Header, display_model, quant_from_name};
use legion::dashboard::term::{self, Exit};
use legion::dashboard::ui::{App, Conn};
use legion::engine::lm::report::LoadReport;
use legion::server::AppState;

/// How often the in-process dashboard takes a snapshot.
const EVERY: Duration = Duration::from_millis(250);

/// Whether `--tui` can run: stdout must be a terminal (one warning otherwise).
pub(super) fn usable(asked: bool) -> bool {
    if asked && !std::io::stdout().is_terminal() {
        tracing::warn!("--tui: stdout is not a terminal; serving without the dashboard");
        return false;
    }
    asked
}

/// The header's layout part from a loaded model's report.
pub(super) fn layout(report: &LoadReport) -> Header {
    let gpu_stages: Vec<_> = report
        .stages
        .iter()
        .filter(|s| s.device.starts_with("CUDA"))
        .collect();
    let devices: Vec<String> = if report.tp > 0 {
        report
            .stages
            .iter()
            .flat_map(|s| s.tp_devices.iter().map(|d| d.0.clone()))
            .collect()
    } else {
        report
            .stages
            .iter()
            .map(|s| match &s.description {
                Some(d) => format!("{} {d}", s.device),
                None => s.device.clone(),
            })
            .collect()
    };
    Header {
        gpus: if report.tp > 0 {
            report.tp as u32
        } else {
            gpu_stages.len() as u32
        },
        tp: report.tp as u32,
        stages: report.stages.len() as u32,
        devices,
        ctx_size: report.kv.ctx_size as u32,
        kv_types: (!report.kv.types.is_empty()).then(|| report.kv.types.clone()),
        kv_device_tokens: report.kv.device_tokens as u64,
        kv_host_tokens: report.kv.host_tokens as u64,
        expert_mode: report.expert_mode.clone(),
        ..Header::default()
    }
}

/// The rest of the header.
pub(super) fn finish(
    mut h: Header,
    model: String,
    model_path: Option<&str>,
    max_seqs: u32,
    spec: (&str, u32),
    version: String,
) -> Header {
    let model = display_model(&model, model_path);
    h.quant = model_path
        .and_then(quant_from_name)
        .or_else(|| quant_from_name(&model));
    h.model = model;
    h.max_seqs = max_seqs;
    h.spec = spec.0.to_string();
    h.spec_k = if spec.0 == "none" { 0 } else { spec.1 };
    h.version = version;
    h
}

/// Starts the dashboard on its own thread. `q` gives the terminal back to the plain logs and
/// leaves the server running; Ctrl-C stops the server (as it does without --tui).
pub(super) fn start(state: &AppState) {
    let hub = state.stats.clone();
    let engine = state.engine.clone();
    hub.pin();
    let rt = tokio::runtime::Handle::current();
    let spawned = std::thread::Builder::new()
        .name("dashboard".into())
        .spawn(move || {
            let _rt = rt.enter();
            let mut app = App::new("serve --tui", Conn::Local);
            let mut last: Option<Instant> = None;
            let mut seq = 0u64;
            let result = term::run(&mut app, |app| {
                if last.is_none_or(|t| t.elapsed() >= EVERY) {
                    last = Some(Instant::now());
                    let snap = hub.snapshot(&engine.stats(), hub.attach(), Instant::now());
                    app.push_snapshot(snap);
                }
                let lines = logs::ring().since(seq, 5);
                if let Some(l) = lines.last() {
                    seq = l.seq;
                    app.push_logs(lines, false);
                }
            });
            match result {
                Ok(Exit::Quit) => {
                    hub.unpin();
                    tracing::info!("dashboard closed (q); the server keeps running");
                }
                Ok(Exit::Interrupt) => {
                    // Raw mode made Ctrl-C a key: deliver it as the signal the server waits for.
                    let _ = nix::sys::signal::raise(nix::sys::signal::Signal::SIGINT);
                }
                Err(e) => {
                    hub.unpin();
                    tracing::warn!(error = %e, "--tui: the dashboard stopped");
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "--tui: cannot start the dashboard");
    }
}
