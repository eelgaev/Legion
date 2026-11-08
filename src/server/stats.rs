//! The dashboard's data on the server side (#1198): the requests in flight, five-minute latency
//! windows, the hardware sampler started on demand, and [`StatsHub::snapshot`], which turns
//! them and the engine's stats into a [`Snapshot`] (`GET /v1/stats`, `serve --tui`).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::dashboard::hub;
use crate::dashboard::latency::LatencyWindow;
use crate::dashboard::scale;
use crate::dashboard::snapshot::{
    EngineSnap, GpuSnap, Header, HwStatus, Latency, LinkKind, LinkSnap, MemorySnap, PhaseBytes,
    PhaseTime, RequestRow, Snapshot, SpecSnap, VERSION,
};
use crate::engine::EngineStats;
use crate::hw::counters::{GpuLive, HwCounters};

/// How often the on-demand sampler samples.
const SAMPLE_EVERY: Duration = Duration::from_millis(250);
/// The on-demand sampler stops this long after the last stats request.
pub const IDLE_STOP: Duration = Duration::from_secs(60);
/// A failed sampler start is retried no sooner than this.
const RETRY_START: Duration = Duration::from_secs(30);

tokio::task_local! {
    /// The API a request came in through, set by the metrics middleware.
    pub static ENDPOINT: &'static str;
}

/// The endpoint name for a matched route.
pub fn endpoint_of(route: &str) -> &'static str {
    match route {
        "/v1/chat/completions" => "chat",
        "/v1/completions" => "completions",
        "/v1/messages" => "messages",
        "/v1/responses" => "responses",
        _ => "other",
    }
}

/// One request in flight, updated by its stream (atomics only).
pub struct LiveRequest {
    id: u64,
    endpoint: &'static str,
    start: Instant,
    /// `u32::MAX` until the engine admitted it.
    prompt: AtomicU32,
    /// Nanoseconds from `start` to the first output, 0 before.
    first_ns: AtomicU64,
    generated: AtomicU32,
}

impl LiveRequest {
    pub fn started(&self, prompt_tokens: u32) {
        self.prompt.store(prompt_tokens, Ordering::Relaxed);
    }

    /// One output event at `now`.
    pub fn output(&self, now: Instant) {
        if self.first_ns.load(Ordering::Relaxed) == 0 {
            let ns = now.saturating_duration_since(self.start).as_nanos() as u64;
            self.first_ns.store(ns.max(1), Ordering::Relaxed);
        }
        self.generated.fetch_add(1, Ordering::Relaxed);
    }

    fn row(&self, now: Instant) -> RequestRow {
        let prompt = self.prompt.load(Ordering::Relaxed);
        let first = self.first_ns.load(Ordering::Relaxed);
        let generated = self.generated.load(Ordering::Relaxed);
        let age = now.saturating_duration_since(self.start);
        let state = if first > 0 {
            "decode"
        } else if prompt != u32::MAX {
            "prefill"
        } else {
            "queued"
        };
        let tok_s = (first > 0).then(|| {
            let since_first = age.as_secs_f64() - first as f64 / 1e9;
            if since_first > 0.0 {
                generated.saturating_sub(1) as f64 / since_first
            } else {
                0.0
            }
        });
        RequestRow {
            id: self.id,
            state: state.into(),
            endpoint: self.endpoint.into(),
            prompt_tokens: (prompt != u32::MAX).then_some(prompt),
            generated,
            age_ms: age.as_millis() as u64,
            ttft_ms: (first > 0).then_some(first / 1_000_000),
            tok_s,
        }
    }
}

/// The on-demand sampler's state.
#[derive(Default)]
struct Sampler {
    counters: Option<Arc<HwCounters>>,
    error: Option<String>,
    tried: Option<Instant>,
    last_use: Option<Instant>,
}

pub struct StatsHub {
    pub start: Instant,
    pub header: Mutex<Header>,
    next_id: AtomicU64,
    live: Mutex<Vec<Arc<LiveRequest>>>,
    pub ttft: LatencyWindow,
    pub itl: LatencyWindow,
    /// `--hw-counters`' sampler: always used, never stopped.
    fixed: Option<Arc<HwCounters>>,
    sampler: Mutex<Sampler>,
    /// `serve --tui`: the dashboard stays attached (no idle stop).
    pinned: AtomicBool,
    reaper: AtomicBool,
    nic: std::sync::OnceLock<Option<(f64, String)>>,
}

impl StatsHub {
    pub fn new(fixed: Option<Arc<HwCounters>>) -> Self {
        let start = Instant::now();
        Self {
            start,
            header: Mutex::new(Header::default()),
            next_id: AtomicU64::new(1),
            live: Mutex::new(Vec::new()),
            ttft: LatencyWindow::new(start),
            itl: LatencyWindow::new(start),
            fixed,
            sampler: Mutex::new(Sampler::default()),
            pinned: AtomicBool::new(false),
            reaper: AtomicBool::new(false),
            nic: std::sync::OnceLock::new(),
        }
    }

    pub fn set_header(&self, h: Header) {
        if let Ok(mut g) = self.header.lock() {
            *g = h;
        }
    }

    /// Tracks a new request (one lock per request, none per token).
    pub fn begin(&self) -> Arc<LiveRequest> {
        let r = Arc::new(LiveRequest {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            endpoint: ENDPOINT.try_with(|e| *e).unwrap_or("other"),
            start: Instant::now(),
            prompt: AtomicU32::new(u32::MAX),
            first_ns: AtomicU64::new(0),
            generated: AtomicU32::new(0),
        });
        if let Ok(mut l) = self.live.lock() {
            l.push(r.clone());
        }
        r
    }

    pub fn end(&self, r: &Arc<LiveRequest>) {
        if let Ok(mut l) = self.live.lock()
            && let Some(i) = l.iter().position(|x| Arc::ptr_eq(x, r))
        {
            l.swap_remove(i);
        }
    }

    /// Keeps the dashboard attached for good (`serve --tui`).
    pub fn pin(&self) {
        self.pinned.store(true, Ordering::Relaxed);
        hub::set_wanted(true);
    }

    /// The pinned dashboard closed: the idle stop applies again (from now).
    pub fn unpin(&self) {
        self.pinned.store(false, Ordering::Relaxed);
        if let Ok(mut s) = self.sampler.lock() {
            s.last_use = Some(Instant::now());
        }
    }

    /// A stats client is here: the engine's dashboard counters on, and the sampler started
    /// (if it is not running). Returns the sampler, or why there is none.
    pub fn attach(self: &Arc<Self>) -> Result<Arc<HwCounters>, String> {
        hub::set_wanted(true);
        self.spawn_reaper();
        let mut s = self.sampler.lock().unwrap_or_else(|e| e.into_inner());
        s.last_use = Some(Instant::now());
        if let Some(c) = &self.fixed {
            return Ok(c.clone());
        }
        if let Some(c) = &s.counters {
            return Ok(c.clone());
        }
        if s.tried.is_some_and(|t| t.elapsed() < RETRY_START) {
            return Err(s.error.clone().unwrap_or_else(|| "starting".into()));
        }
        s.tried = Some(Instant::now());
        match HwCounters::start(SAMPLE_EVERY) {
            Ok(c) => {
                let c = Arc::new(c);
                tracing::info!(gpus = ?c.gpus(), "dashboard: hardware sampler started (NVML)");
                s.counters = Some(c.clone());
                s.error = None;
                Ok(c)
            }
            Err(e) => {
                s.error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Stops the on-demand sampler and the engine's dashboard counters once no client asked
    /// for [`IDLE_STOP`].
    fn spawn_reaper(self: &Arc<Self>) {
        if self.reaper.swap(true, Ordering::Relaxed) {
            return;
        }
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            self.reaper.store(false, Ordering::Relaxed);
            return;
        };
        let hub = Arc::downgrade(self);
        rt.spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                let Some(h) = hub.upgrade() else { return };
                if h.pinned.load(Ordering::Relaxed) {
                    continue;
                }
                let mut s = h.sampler.lock().unwrap_or_else(|e| e.into_inner());
                if s.last_use.is_some_and(|t| t.elapsed() >= IDLE_STOP) {
                    s.last_use = None;
                    hub::set_wanted(false);
                    if s.counters.take().is_some() {
                        tracing::info!("dashboard: no stats client for 60 s, sampler stopped");
                    }
                }
            }
        });
    }

    /// The snapshot at `now` from the engine's `stats` and the sampler's (`hw`) counters.
    pub fn snapshot(
        &self,
        stats: &EngineStats,
        hw: Result<Arc<HwCounters>, String>,
        now: Instant,
    ) -> Snapshot {
        let header = self.header.lock().map(|h| h.clone()).unwrap_or_default();
        let mut requests: Vec<RequestRow> = self
            .live
            .lock()
            .map(|l| l.iter().map(|r| r.row(now)).collect())
            .unwrap_or_default();
        requests.sort_by_key(|r| r.id);
        let (ttft_p50_ms, ttft_p95_ms, ttft_samples) = self.ttft.percentiles(now);
        let (itl_p50_ms, itl_p95_ms, itl_samples) = self.itl.percentiles(now);
        let engine = engine_snap(stats);
        let (gpus, hw_status) = match hw {
            Ok(c) => (
                gpu_snaps(&c.live(), stats, &header),
                HwStatus {
                    sampling: true,
                    reason: None,
                },
            ),
            Err(reason) => (
                Vec::new(),
                HwStatus {
                    sampling: false,
                    reason: Some(reason),
                },
            ),
        };
        let network = hub::tp_bytes().map(|(tx, rx)| {
            let nic = self.nic.get_or_init(scale::fastest_nic);
            LinkSnap {
                kind: LinkKind::Network,
                detail: nic.as_ref().map_or_else(String::new, |n| n.1.clone()),
                full_scale_gb_s: nic.as_ref().map(|n| n.0),
                scale_source: if nic.is_some() {
                    "NIC link speed (sysfs)".into()
                } else {
                    "no NIC speed in sysfs".into()
                },
                // Legion's own frames are the only counter: measured = Legion's.
                tx_bytes: None,
                rx_bytes: None,
                legion_rx_bytes: rx as f64,
                legion_tx_bytes: tx as f64,
            }
        });
        Snapshot {
            version: VERSION,
            uptime_s: now.saturating_duration_since(self.start).as_secs_f64(),
            time_ms: crate::dashboard::logs::now_ms(),
            header,
            engine,
            requests,
            latency: Latency {
                ttft_p50_ms,
                ttft_p95_ms,
                ttft_samples,
                itl_p50_ms,
                itl_p95_ms,
                itl_samples,
            },
            hw: hw_status,
            gpus,
            network,
        }
    }
}

fn phase(p: &crate::ggml::model_bytes::Phase) -> PhaseBytes {
    PhaseBytes {
        device: p.device(),
        link: p.link(),
        experts_streamed: p.experts_host + p.staged,
        kv_host: p.kv_host,
        seconds: p.seconds,
        steps: p.steps,
    }
}

fn engine_snap(s: &EngineStats) -> EngineSnap {
    let mb = &s.model_bytes;
    let stages = if mb.n_stages > 1 {
        mb.stages[..mb.n_stages.min(mb.stages.len())]
            .iter()
            .map(|st| {
                let mut p = phase(&st.prefill);
                p.add(&phase(&st.decode));
                p
            })
            .collect()
    } else {
        Vec::new()
    };
    let (phases, steps) = hub::step_phases();
    EngineSnap {
        running: s.running,
        prefilling: s.prefilling,
        decoding: s.decoding,
        waiting: s.waiting,
        prompt_tokens: s.prompt_tokens,
        generated_tokens: s.generated_tokens,
        kv_blocks_used: s.kv_blocks_used,
        kv_blocks_total: s.kv_blocks_total,
        prefix_cache_hit_tokens: s.prefix_cache_hit_tokens,
        expert_cache_hits: s.expert_cache_hits,
        expert_cache_misses: s.expert_cache_misses,
        expert_cache_slots: s.expert_cache_slots,
        expert_cache_layers: s.expert_cache_layers,
        spec: SpecSnap {
            drafted: s.spec.drafted,
            accepted: s.spec.accepted,
            steps: s.spec.steps,
            mean_k: s.spec_mean_k,
            by_position: hub::spec_positions(),
        },
        prefill: phase(&mb.prefill),
        decode: phase(&mb.decode),
        draft: phase(&mb.draft),
        stages,
        step_phases: if steps > 0 {
            phases
                .into_iter()
                .map(|(name, us)| PhaseTime {
                    name: name.into(),
                    us,
                })
                .collect()
        } else {
            Vec::new()
        },
        steps,
    }
}

/// Legion's bytes over the host link and from device memory for GPU `i` of `n`: its pipeline
/// stage's where the layout has one stage per GPU, else an even share.
fn legion_share(e: &EngineSnap, i: usize, n: usize) -> PhaseBytes {
    if e.stages.len() == n && n > 1 {
        return e.stages[i];
    }
    let mut all = e.prefill;
    all.add(&e.decode);
    all.add(&e.draft);
    let n = n.max(1) as f64;
    PhaseBytes {
        device: (all.device as f64 / n) as u64,
        link: (all.link as f64 / n) as u64,
        experts_streamed: (all.experts_streamed as f64 / n) as u64,
        kv_host: (all.kv_host as f64 / n) as u64,
        // Every GPU of a tensor-parallel step runs the whole step's time.
        seconds: all.seconds,
        steps: all.steps,
    }
}

/// The GPU panel's and link gauges' data, from the sampler (`live`) and the engine's bytes.
pub fn gpu_snaps(live: &[GpuLive], stats: &EngineStats, header: &Header) -> Vec<GpuSnap> {
    let e = engine_snap(stats);
    // The GPUs the engine runs on: the first `gpus` the process sees (all without a count).
    let used = if header.gpus > 0 {
        (header.gpus as usize).min(live.len())
    } else {
        live.len()
    };
    live[..used]
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let share = legion_share(&e, i, used);
            GpuSnap {
                index: g.index,
                name: g.info.name.clone(),
                sm_busy_pct: g.sm_util_pct,
                // A GB10's NVML memory utilization reads 0 under a decode that moves its whole
                // bandwidth (measured on a DGX Spark), so a unified part reports none.
                mem_busy_pct: (!g.info.unified).then_some(g.mem_util_pct).flatten(),
                vram_used: g.vram_used,
                vram_total: g.vram_total,
                power_w: g.power_w,
                sm_clock_mhz: g.sm_clock_mhz,
                mem_clock_mhz: g.mem_clock_mhz,
                unified: g.info.unified,
                links: links_of(g, &share),
                memory: {
                    let (peak, source) = scale::memory_peak_gb_s(
                        g.info.mem_bus_bits,
                        g.info.max_mem_mhz,
                        &g.info.name,
                    );
                    MemorySnap {
                        peak_gb_s: peak,
                        peak_source: source,
                        gpm_dram_pct: g.gpm_dram_pct,
                        legion_bytes: share.device,
                        legion_seconds: share.seconds,
                    }
                },
            }
        })
        .collect()
}

/// A GPU's link gauges: its host link (C2C, NVLink to the CPU or PCIe) carries Legion's host
/// bytes; NVLink to peers gets its own gauge.
pub fn links_of(g: &GpuLive, share: &PhaseBytes) -> Vec<LinkSnap> {
    let info = &g.info;
    let mut out = Vec::new();
    let host_bytes = share.link as f64;
    if info.has_c2c() {
        let (full, source) = scale::c2c_gb_s(info.c2c_links, info.c2c_max_mb_s, &info.name);
        let detail = match info.c2c_links {
            Some(n) if n > 0 => format!("{n} links"),
            _ => String::new(),
        };
        out.push(LinkSnap {
            kind: LinkKind::C2c,
            detail,
            full_scale_gb_s: full,
            scale_source: source,
            tx_bytes: g.c2c.map(|c| c.0),
            rx_bytes: g.c2c.map(|c| c.1),
            legion_rx_bytes: host_bytes,
            legion_tx_bytes: 0.0,
        });
    } else if info.cpu_links() > 0 {
        let n = info.cpu_links();
        let v = info.nvlink_version.unwrap_or(0);
        out.push(LinkSnap {
            kind: LinkKind::NvlinkCpu,
            detail: format!("{n} links NVLink {v}"),
            full_scale_gb_s: scale::nvlink_gb_s(n, v),
            scale_source: "NVML links x NVLink version speed".into(),
            tx_bytes: Some(g.nvlink_cpu.0),
            rx_bytes: Some(g.nvlink_cpu.1),
            legion_rx_bytes: host_bytes,
            legion_tx_bytes: 0.0,
        });
    } else {
        let (generation, width) = (info.pcie_max_gen, info.pcie_max_width);
        out.push(LinkSnap {
            kind: LinkKind::Pcie,
            detail: match (generation, width) {
                (Some(gn), Some(w)) => scale::pcie_label(gn, w),
                _ => String::new(),
            },
            full_scale_gb_s: generation
                .zip(width)
                .and_then(|(gn, w)| scale::pcie_gb_s(gn, w)),
            scale_source: "NVML max link gen x max width".into(),
            tx_bytes: g.pcie_counted.then_some(g.totals.pcie_tx_bytes),
            rx_bytes: g.pcie_counted.then_some(g.totals.pcie_rx_bytes),
            legion_rx_bytes: host_bytes,
            legion_tx_bytes: 0.0,
        });
    }
    if info.peer_links() > 0 {
        let n = info.peer_links();
        let v = info.nvlink_version.unwrap_or(0);
        out.push(LinkSnap {
            kind: LinkKind::NvlinkPeer,
            detail: format!("{n} links NVLink {v}"),
            full_scale_gb_s: scale::nvlink_gb_s(n, v),
            scale_source: "NVML links x NVLink version speed".into(),
            tx_bytes: Some(g.nvlink_peer.0),
            rx_bytes: Some(g.nvlink_peer.1),
            // Peer copies are not in the model bytes.
            legion_rx_bytes: 0.0,
            legion_tx_bytes: 0.0,
        });
    }
    out
}

#[cfg(test)]
mod tests;
