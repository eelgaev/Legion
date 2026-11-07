//! The stats snapshot (`GET /v1/stats`, schema version [`VERSION`]): one JSON object with
//! everything the dashboard shows. Counters are cumulative (since the server started, or since
//! the hardware sampler started for the `gpus` counters), so a client turns two snapshots into
//! rates ([`Rates::between`]) and a missed snapshot loses nothing. Values a backend cannot
//! report are `null`, never a made-up number.

use serde::{Deserialize, Serialize};

/// The schema version; bumped when a field changes meaning or goes away (new fields do not).
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    /// Seconds since the server started.
    pub uptime_s: f64,
    /// Unix milliseconds when the snapshot was taken.
    pub time_ms: u64,
    pub header: Header,
    pub engine: EngineSnap,
    pub requests: Vec<RequestRow>,
    pub latency: Latency,
    /// The hardware sampler: whether it runs, and why not.
    pub hw: HwStatus,
    pub gpus: Vec<GpuSnap>,
    /// The tensor-parallel exchange over the network (two machines), when there is one.
    pub network: Option<LinkSnap>,
}

/// What is served, and how.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub model: String,
    /// Read from the model's file or directory name (`Q4_K_XL`, `NVFP4`, ...), `null` when the
    /// name does not say.
    pub quant: Option<String>,
    pub gpus: u32,
    /// Tensor-parallel GPUs (0: not tensor parallel).
    pub tp: u32,
    /// Pipeline stages (one per GPU of a layer split).
    pub stages: u32,
    /// The devices the stages run on (`CUDA0: Tesla V100-SXM2-16GB`).
    pub devices: Vec<String>,
    pub ctx_size: u32,
    pub max_seqs: u32,
    /// `none`, `mtp`, `dflash`, `ngram`, ...
    pub spec: String,
    /// Max drafts per step (`--spec-k`), 0 without speculative decoding.
    pub spec_k: u32,
    /// The K and V types (`f16/f16`).
    pub kv_types: Option<String>,
    /// VRAM and host-tier KV capacity, in tokens.
    pub kv_device_tokens: u64,
    pub kv_host_tokens: u64,
    /// Where routed experts live (`device`, `host`, `cache`), `null` for dense models.
    pub expert_mode: Option<String>,
    /// `0.1.0 (abc1234, ggml 1a2b3c4, cuda on)`.
    pub version: String,
}

/// The engine's counters ([`crate::engine::EngineStats`] plus the dashboard's own).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EngineSnap {
    pub running: u32,
    pub prefilling: u32,
    pub decoding: u32,
    pub waiting: u32,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub kv_blocks_used: u32,
    pub kv_blocks_total: u32,
    pub prefix_cache_hit_tokens: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub expert_cache_slots: u32,
    pub expert_cache_layers: u32,
    pub spec: SpecSnap,
    /// Model bytes by kind of step (required traffic, not measured).
    pub prefill: PhaseBytes,
    pub decode: PhaseBytes,
    pub draft: PhaseBytes,
    /// Per pipeline stage (prefill + decode), when the layout has several.
    pub stages: Vec<PhaseBytes>,
    /// The engine thread's time per step by phase, cumulative while a dashboard is attached
    /// (empty for engines without it).
    pub step_phases: Vec<PhaseTime>,
    /// Steps counted in `step_phases`.
    pub steps: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecSnap {
    pub drafted: u64,
    pub accepted: u64,
    pub steps: u64,
    /// The policy's mean draft depth in the last drafting step.
    pub mean_k: f64,
    /// Per draft position (0 = the first draft): `[drafted, accepted]`, counted while a
    /// dashboard is attached.
    pub by_position: Vec<[u64; 2]>,
}

/// One kind of step's model bytes (#391), cumulative.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PhaseBytes {
    /// Read from device memory (weights, resident experts, KV).
    pub device: u64,
    /// Over the host link (host experts read in place, staged copies, host-tier KV).
    pub link: u64,
    /// Of `link`, routed experts streamed or staged from the host.
    pub experts_streamed: u64,
    /// Of `link`, the KV host tier.
    pub kv_host: u64,
    pub seconds: f64,
    pub steps: u64,
}

impl PhaseBytes {
    pub fn add(&mut self, o: &PhaseBytes) {
        self.device += o.device;
        self.link += o.link;
        self.experts_streamed += o.experts_streamed;
        self.kv_host += o.kv_host;
        self.seconds += o.seconds;
        self.steps += o.steps;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhaseTime {
    pub name: String,
    pub us: u64,
}

/// One request the server is handling.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RequestRow {
    pub id: u64,
    /// `queued`, `prefill` or `decode`.
    pub state: String,
    /// `chat`, `completions`, `messages`, `responses` (or `other`).
    pub endpoint: String,
    /// `null` until the engine admitted the request.
    pub prompt_tokens: Option<u32>,
    /// Output events (text and reasoning pieces) so far.
    pub generated: u32,
    pub age_ms: u64,
    pub ttft_ms: Option<u64>,
    /// Generated over the time since the first output.
    pub tok_s: Option<f64>,
}

/// TTFT and inter-token latency over the last five minutes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p95_ms: Option<f64>,
    pub ttft_samples: u64,
    pub itl_p50_ms: Option<f64>,
    pub itl_p95_ms: Option<f64>,
    pub itl_samples: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HwStatus {
    pub sampling: bool,
    /// Why there are no hardware counters (`NVML: ...`), when there are none.
    pub reason: Option<String>,
}

/// One GPU as NVML sees it, plus Legion's own traffic attributed to it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuSnap {
    /// NVML index.
    pub index: u32,
    pub name: String,
    pub sm_busy_pct: Option<f64>,
    /// NVML memory utilization: the share of time device memory was read or written, not a
    /// share of its bandwidth.
    pub mem_busy_pct: Option<f64>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub power_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    /// One shared memory for the CPU and the GPU (GB10).
    pub unified: bool,
    pub links: Vec<LinkSnap>,
    pub memory: MemorySnap,
}

/// Device-memory bandwidth: GPM's measured utilization and Legion's own bytes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemorySnap {
    /// Peak bandwidth in GB/s (bus width x memory clock from NVML, or the known spec).
    pub peak_gb_s: Option<f64>,
    /// Where `peak_gb_s` comes from.
    pub peak_source: String,
    /// NVML GPM DRAM bandwidth utilization over the last sample, 0-100; `null` without GPM.
    pub gpm_dram_pct: Option<f64>,
    /// Legion's model bytes read from this device's memory, cumulative, and the step seconds.
    pub legion_bytes: u64,
    pub legion_seconds: f64,
}

/// What a link connects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    #[default]
    Pcie,
    /// NVLink to the host CPU (an AC922's POWER9).
    NvlinkCpu,
    /// NVLink to other GPUs (or an NVSwitch).
    NvlinkPeer,
    /// NVLink-C2C (Grace Hopper / Grace Blackwell, GB10).
    C2c,
    /// The tensor-parallel exchange between machines (TCP or RDMA).
    Network,
}

impl LinkKind {
    pub fn label(self) -> &'static str {
        match self {
            LinkKind::Pcie => "PCIe",
            LinkKind::NvlinkCpu => "NVLink CPU",
            LinkKind::NvlinkPeer => "NVLink peer",
            LinkKind::C2c => "C2C",
            LinkKind::Network => "TP net",
        }
    }
}

/// One link gauge's data: cumulative bytes each way, measured and Legion's own.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkSnap {
    pub kind: LinkKind,
    /// `PCIe 4.0 x16`, `3 links NVLink 2`, ...
    pub detail: String,
    /// Full scale per direction in GB/s, `null` when nothing tells.
    pub full_scale_gb_s: Option<f64>,
    /// Where the full scale comes from (`NVML max gen x width`, `known spec`, ...).
    pub scale_source: String,
    /// Measured bytes sent and received by the GPU (or the host for `network`), cumulative;
    /// `null` when no counter exists.
    pub tx_bytes: Option<f64>,
    pub rx_bytes: Option<f64>,
    /// Legion's own bytes over this link, cumulative (into the GPU / out of it).
    pub legion_rx_bytes: f64,
    pub legion_tx_bytes: f64,
}

/// Rates between two snapshots.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    pub seconds: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// Draft tokens accepted per verify step, and tokens appended per verify step.
    pub accepted_per_step: Option<f64>,
    pub expert_hit_rate: Option<f64>,
    /// Routed experts streamed or staged from the host, GB/s.
    pub expert_stream_gb_s: f64,
    pub prefix_hit_tok_s: f64,
    /// Per GPU, per link (same order as the snapshot): `(tx, rx, legion)` GB/s.
    pub links: Vec<Vec<LinkRate>>,
    pub network: Option<LinkRate>,
    /// Per GPU: measured and Legion's device-memory GB/s.
    pub memory: Vec<MemoryRate>,
    /// Per step phase, microseconds per step over the interval.
    pub step_us: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LinkRate {
    pub tx_gb_s: Option<f64>,
    pub rx_gb_s: Option<f64>,
    pub legion_gb_s: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MemoryRate {
    /// GPM DRAM utilization x peak.
    pub measured_gb_s: Option<f64>,
    /// Legion's device bytes over the step time.
    pub legion_gb_s: f64,
}

fn rate(now: f64, before: f64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        (now - before).max(0.0) / seconds
    }
}

fn link_rate(cur: &LinkSnap, prev: Option<&LinkSnap>, seconds: f64) -> LinkRate {
    let gb = |now: Option<f64>, before: Option<f64>| match (now, before) {
        (Some(n), Some(b)) => Some(rate(n, b, seconds) / 1e9),
        // The first snapshot of a link: no rate yet.
        (Some(_), None) => Some(0.0),
        (None, _) => None,
    };
    let (ptx, prx, plrx, pltx) = prev.map_or((None, None, 0.0, 0.0), |p| {
        (p.tx_bytes, p.rx_bytes, p.legion_rx_bytes, p.legion_tx_bytes)
    });
    let legion = if prev.is_some() {
        rate(
            cur.legion_rx_bytes + cur.legion_tx_bytes,
            plrx + pltx,
            seconds,
        ) / 1e9
    } else {
        0.0
    };
    LinkRate {
        tx_gb_s: gb(cur.tx_bytes, ptx),
        rx_gb_s: gb(cur.rx_bytes, prx),
        legion_gb_s: legion,
    }
}

impl Rates {
    /// The rates from `prev` to `cur` (`prev` older). Counters that went backwards (a restart
    /// or a sampler that started again) read 0.
    pub fn between(prev: &Snapshot, cur: &Snapshot) -> Rates {
        let seconds = (cur.uptime_s - prev.uptime_s).max(0.0);
        let (e, p) = (&cur.engine, &prev.engine);
        let prefill_tok_s = rate(e.prompt_tokens as f64, p.prompt_tokens as f64, seconds);
        let decode_tok_s = rate(
            e.generated_tokens as f64,
            p.generated_tokens as f64,
            seconds,
        );
        let d_steps = e.spec.steps.saturating_sub(p.spec.steps);
        let accepted_per_step = (d_steps > 0)
            .then(|| e.spec.accepted.saturating_sub(p.spec.accepted) as f64 / d_steps as f64);
        let hits = e.expert_cache_hits.saturating_sub(p.expert_cache_hits);
        let misses = e.expert_cache_misses.saturating_sub(p.expert_cache_misses);
        let expert_hit_rate = (hits + misses > 0).then(|| hits as f64 / (hits + misses) as f64);
        let streamed = |s: &EngineSnap| {
            (s.prefill.experts_streamed + s.decode.experts_streamed + s.draft.experts_streamed)
                as f64
        };
        let expert_stream_gb_s = rate(streamed(e), streamed(p), seconds) / 1e9;
        let prefix_hit_tok_s = rate(
            e.prefix_cache_hit_tokens as f64,
            p.prefix_cache_hit_tokens as f64,
            seconds,
        );
        let links = cur
            .gpus
            .iter()
            .map(|g| {
                let pg = prev.gpus.iter().find(|x| x.index == g.index);
                g.links
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        let pl = pg
                            .and_then(|pg| pg.links.get(i))
                            .filter(|pl| pl.kind == l.kind);
                        link_rate(l, pl, seconds)
                    })
                    .collect()
            })
            .collect();
        let network = cur
            .network
            .as_ref()
            .map(|n| link_rate(n, prev.network.as_ref(), seconds));
        let memory = cur
            .gpus
            .iter()
            .map(|g| {
                let pg = prev.gpus.iter().find(|x| x.index == g.index);
                let m = &g.memory;
                let measured_gb_s = match (m.gpm_dram_pct, m.peak_gb_s) {
                    (Some(pct), Some(peak)) => Some(pct / 100.0 * peak),
                    _ => None,
                };
                let legion_gb_s = pg.map_or(0.0, |pg| {
                    let dt = m.legion_seconds - pg.memory.legion_seconds;
                    if dt > 0.0 {
                        m.legion_bytes.saturating_sub(pg.memory.legion_bytes) as f64 / dt / 1e9
                    } else {
                        0.0
                    }
                });
                MemoryRate {
                    measured_gb_s,
                    legion_gb_s,
                }
            })
            .collect();
        let d_phase_steps = e.steps.saturating_sub(p.steps);
        let step_us = e
            .step_phases
            .iter()
            .map(|ph| {
                let before = p
                    .step_phases
                    .iter()
                    .find(|x| x.name == ph.name)
                    .map_or(0, |x| x.us);
                let us = if d_phase_steps > 0 {
                    ph.us.saturating_sub(before) as f64 / d_phase_steps as f64
                } else {
                    0.0
                };
                (ph.name.clone(), us)
            })
            .collect();
        Rates {
            seconds,
            prefill_tok_s,
            decode_tok_s,
            accepted_per_step,
            expert_hit_rate,
            expert_stream_gb_s,
            prefix_hit_tok_s,
            links,
            network,
            memory,
            step_us,
        }
    }
}

/// Quantization names a model's file or directory name carries, longest first.
const QUANTS: &[&str] = &[
    "UD-Q4_K_XL",
    "UD-Q2_K_XL",
    "UD-Q3_K_XL",
    "UD-Q5_K_XL",
    "UD-Q6_K_XL",
    "UD-Q8_K_XL",
    "Q4_K_XL",
    "Q2_K_XL",
    "Q3_K_XL",
    "Q5_K_XL",
    "Q6_K_XL",
    "Q8_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
    "IQ4_NL",
    "MXFP4",
    "NVFP4",
    "FP8",
    "BF16",
    "F16",
    "F32",
    "QAT",
    "AWQ",
    "GPTQ",
];

/// The name the header shows. A model loaded from the Hugging Face cache has its snapshot's
/// commit hash as its id (`.../models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc69...`):
/// then the repository's name; otherwise the id.
pub fn display_model(id: &str, path: Option<&str>) -> String {
    let hash = id.len() == 40 && id.bytes().all(|b| b.is_ascii_hexdigit());
    let repo = path.and_then(|p| {
        p.split('/')
            .rev()
            .find_map(|c| c.strip_prefix("models--"))
            .and_then(|r| r.rsplit("--").next())
            .filter(|r| !r.is_empty())
    });
    match repo {
        Some(r) if hash => r.to_string(),
        _ => id.to_string(),
    }
}

/// The quantization a model's path names (`Q4_K_XL`, `NVFP4`, `QAT Q4_0`), from its file
/// name and its directory's; `None` when neither says.
pub fn quant_from_name(path: &str) -> Option<String> {
    let parts: Vec<&str> = path.rsplit('/').filter(|p| !p.is_empty()).take(2).collect();
    let upper = parts.join(" ").to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let boundary = |i: usize| i >= bytes.len() || !bytes[i].is_ascii_alphanumeric();
    let mut found: Vec<&str> = Vec::new();
    for q in QUANTS {
        let hit = upper
            .match_indices(q)
            .any(|(at, _)| (at == 0 || boundary(at - 1)) && boundary(at + q.len()));
        if hit && !found.iter().any(|f| f.contains(q)) {
            found.push(q);
        }
    }
    // QAT first: `QAT Q4_0`.
    found.sort_by_key(|q| *q != "QAT");
    (!found.is_empty()).then(|| found.join(" "))
}
