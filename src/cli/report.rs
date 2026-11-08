//! Pure rendering for the interactive `legion serve` startup report: the hardware and the
//! checkpoint before the load, then where the load put everything ([`LoadReport`]).

use std::fmt::Write as _;

use clap::ValueEnum;
use legion::engine::EngineStats;
use legion::engine::lm::report::{HostMemory, HostUse, LoadReport, ModelFacts, NgramPlacement};
use legion::hw::{HardwareInfo, MemoryKind, NumaNodeInfo};

/// The terminal facts colour and the live status line depend on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TermEnv {
    /// `NO_COLOR` is set (to anything).
    pub no_color: bool,
    /// `TERM=dumb`: no ANSI sequences at all.
    pub dumb: bool,
}

impl TermEnv {
    pub(super) fn from_env() -> Self {
        Self {
            no_color: std::env::var_os("NO_COLOR").is_some(),
            dumb: std::env::var_os("TERM").is_some_and(|t| t == "dumb"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(super) enum ColorChoice {
    /// Colour when the output is a terminal (not with `TERM=dumb` or `NO_COLOR`).
    #[default]
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    pub(super) fn use_color(self, tty: bool, env: TermEnv) -> bool {
        // no-color.org: NO_COLOR changes the default; an explicit flag wins.
        match self {
            Self::Auto => tty && !env.dumb && !env.no_color,
            Self::Always => true,
            Self::Never => false,
        }
    }

    pub(super) fn show_report(self, tty: bool) -> bool {
        tty || self == Self::Always
    }
}

/// `--status-line`: the one-line live server status redrawn in place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(super) enum StatusLineChoice {
    /// On when stdout is a terminal that takes ANSI sequences (not `TERM=dumb`).
    #[default]
    Auto,
    On,
    Off,
}

impl StatusLineChoice {
    /// Whether to draw the live line; never under JSON logs, whose lines it would break.
    pub(super) fn live(self, tty: bool, env: TermEnv, json: bool) -> bool {
        !json
            && match self {
                Self::Auto => tty && !env.dumb,
                Self::On => true,
                Self::Off => false,
            }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Section {
    pub title: String,
    pub lines: Vec<String>,
}

/// A block of the report as data, so plain and ANSI renderings are deterministic and terminal
/// concerns stay out of model loading.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct StartupReport {
    pub heading: String,
    pub model: String,
    pub sections: Vec<Section>,
}

impl StartupReport {
    pub(super) fn new(model: impl Into<String>) -> Self {
        Self {
            heading: "LEGION SERVE · STARTUP".into(),
            model: model.into(),
            ..Self::default()
        }
    }

    /// The block printed once the model has loaded.
    pub(super) fn loaded(model: impl Into<String>) -> Self {
        Self {
            heading: "LEGION SERVE · LOADED".into(),
            ..Self::new(model)
        }
    }

    pub(super) fn section(
        mut self,
        title: impl Into<String>,
        lines: impl IntoIterator<Item = String>,
    ) -> Self {
        let lines: Vec<String> = lines.into_iter().collect();
        if !lines.is_empty() {
            self.sections.push(Section {
                title: title.into(),
                lines,
            });
        }
        self
    }

    pub(super) fn with_hardware(self, hardware: &HardwareInfo, threads: usize) -> Self {
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| cpu_model(&text));
        self.section(
            "Hardware",
            hardware_lines(hardware, threads, cpu.as_deref()),
        )
    }

    /// The checkpoint's own facts (metadata only, read before the weights).
    pub(super) fn with_model(self, facts: &ModelFacts, sampling: String) -> Self {
        let mut quant = facts.quantization.clone();
        quant.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let total: u64 = quant.iter().map(|q| q.1).sum();
        let quant = quant
            .iter()
            .map(|(kind, bytes)| format!("{kind} {}", gib(*bytes)))
            .collect::<Vec<_>>()
            .join(" + ");
        let mut lines = vec![
            format!(
                "architecture {} · {} layers · native context {} tokens",
                facts.architecture,
                facts.layers,
                count(facts.native_context)
            ),
            format!("weights {} on disk: {quant}", gib(total)),
        ];
        lines.push(if facts.experts_per_layer == 0 {
            "dense: no routed experts".into()
        } else {
            format!(
                "{} routed experts per layer, {} per token",
                facts.experts_per_layer, facts.experts_used
            )
        });
        if let Some(bytes) = facts.ngram_bytes {
            lines.push(format!("per-layer n-gram table {}", gib(bytes)));
        }
        lines.push(sampling);
        self.section("Model", lines)
    }

    /// Where the load put everything, from the load's own layout and allocations.
    /// `numa`: the nodes' memory once loaded.
    pub(super) fn with_load(self, load: &LoadReport, numa: &[NumaNodeInfo]) -> Self {
        self.section("Placement", placement_lines(load))
            .section("Memory", memory_lines(load, numa))
            .section("KV cache", kv_lines(load))
            .section("Experts", expert_lines(load))
            .section("N-gram table", table_lines(load.ngram.as_ref()))
            .section("Engram tables", table_lines(load.engram.as_ref()))
    }

    pub(super) fn render(&self, color: bool) -> String {
        let mut out = String::new();
        let (bold, cyan, green, reset) = if color {
            ("\x1b[1m", "\x1b[36m", "\x1b[32m", "\x1b[0m")
        } else {
            ("", "", "", "")
        };
        let _ = writeln!(
            out,
            "{bold}{cyan}{}{reset}  {green}{}{reset}",
            self.heading, self.model
        );
        for section in &self.sections {
            let _ = writeln!(out, "\n{bold}{cyan}{}{reset}", section.title);
            for line in &section.lines {
                let _ = writeln!(out, "  {line}");
            }
        }
        out
    }
}

/// The hardware section's lines; `cpu`: the CPU's model, if known.
pub(super) fn hardware_lines(
    hardware: &HardwareInfo,
    threads: usize,
    cpu: Option<&str>,
) -> Vec<String> {
    let mut lines = Vec::new();

    let kind = hardware.unified_memory.kind;
    match &hardware.gpu {
        Some(report) if report.gpus.is_empty() => lines.push("GPUs: none".into()),
        Some(report) => {
            for gpu in &report.gpus {
                let cc = gpu
                    .compute_capability
                    .map_or_else(|| "unknown".into(), |(a, b)| format!("{a}.{b}"));
                lines.push(format!(
                    "GPU {}: {} · compute {cc} · VRAM {} total / {} free",
                    gpu.index,
                    gpu.name,
                    gib(gpu.memory_total),
                    gib(gpu.memory_free)
                ));
                lines.push(format!(
                    "  PCI {} · NUMA node {} · PCIe gen {} x{} (max gen {} x{})",
                    if gpu.pci_bus_id.is_empty() {
                        "unknown"
                    } else {
                        &gpu.pci_bus_id
                    },
                    opt(gpu.numa_node),
                    opt(gpu.pcie_gen),
                    opt(gpu.pcie_width),
                    opt(gpu.pcie_max_gen),
                    opt(gpu.pcie_max_width)
                ));
                lines.push(format!("  NVLink {}", nvlinks(gpu)));
            }
            lines.push(format!("memory: {}", memory_kind(kind)));
        }
        None => lines.push(format!(
            "GPUs: unknown (NVML: {})",
            hardware.gpu_error.as_deref().unwrap_or("unavailable")
        )),
    }
    for node in &hardware.numa_nodes {
        lines.push(format!(
            "NUMA node {}: RAM {} total / {} free · {} CPUs",
            node.id,
            gib(node.mem_total),
            gib(node.mem_free),
            node.cpus
        ));
    }
    let pool = if threads == 0 {
        "default threads".into()
    } else {
        format!("{threads} threads")
    };
    lines.push(format!(
        "CPU: {} ({}) · {pool}",
        cpu.unwrap_or("unknown model"),
        hardware.arch
    ));
    lines
}

/// The last line of the report: what is serving, and whether the live line follows.
pub(super) fn ready_line(
    model: &str,
    context: Option<u32>,
    stats: EngineStats,
    color: bool,
    status_line: &str,
) -> String {
    let (bold, green, reset) = if color {
        ("\x1b[1m", "\x1b[32m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let kv = if stats.kv_blocks_total > 0 {
        format!(
            " · KV cache {} tokens",
            count(stats.kv_blocks_total as usize * legion::model::core::KV_BLOCK)
        )
    } else {
        String::new()
    };
    format!(
        "\n{bold}{green}READY{reset}  model {model} · context {} tokens{kv} · status line {status_line}\n\n",
        context.map_or_else(|| "unknown".into(), |n| count(n as usize)),
    )
}

pub(super) fn placement_lines(load: &LoadReport) -> Vec<String> {
    let mut lines = Vec::new();
    let experts = load.expert_mode.as_deref().map_or_else(String::new, |m| {
        format!(" · routed experts: {}", expert_mode(m))
    });
    if load.tp > 1 {
        lines.push(format!(
            "tensor parallel over {} GPUs: every layer on each, split by heads and rows{experts}",
            load.tp
        ));
    } else {
        lines.push(format!(
            "{} pipeline stage{}{experts}",
            load.stages.len(),
            if load.stages.len() == 1 { "" } else { "s" }
        ));
    }
    for (i, s) in load.stages.iter().enumerate() {
        lines.push(format!(
            "stage {i} · {}: layers {}..{} ({}) · weights {}",
            device_name(&s.device, s.description.as_deref()),
            s.layers.start,
            s.layers.end,
            s.layers.len(),
            gib(s.weights)
        ));
    }
    lines
}

pub(super) fn memory_lines(load: &LoadReport, numa: &[NumaNodeInfo]) -> Vec<String> {
    let mut lines = Vec::new();
    for (i, s) in load.stages.iter().enumerate() {
        lines.push(format!(
            "stage {i} · {}",
            device_name(&s.device, s.description.as_deref())
        ));
        let mut held = vec![
            format!("weights {}", gib(s.weights)),
            format!("KV {} ({} tokens)", gib(s.kv_bytes), count(s.kv_tokens)),
            format!("states {}", gib(s.states_bytes)),
        ];
        if s.kv_staging_bytes > 0 {
            held.push(format!("KV host-tier staging {}", gib(s.kv_staging_bytes)));
        }
        lines.push(format!("  {}", held.join(" · ")));
        if s.expert_layers > 0 {
            let mut experts = format!(
                "  expert slots {} ({} slots in {} layers)",
                gib(s.expert_slot_bytes),
                count(s.expert_slots),
                s.expert_layers
            );
            if let Some(bytes) = s.expert_staging_bytes {
                let _ = write!(
                    experts,
                    " · long-prompt staging {} (lent by the slots)",
                    gib(bytes)
                );
            }
            lines.push(experts);
        }
        let mut free = Vec::new();
        if let Some(bytes) = s.compute_bytes {
            free.push(format!("step compute buffers {} (measured)", gib(bytes)));
        }
        if let Some(bytes) = s.kv_reserve_bytes {
            free.push(format!("KV fit reserve {}", gib(bytes)));
        }
        if let Some(bytes) = s.expert_reserve_bytes {
            free.push(format!("expert cache reserve {}", gib(bytes)));
        }
        if let (Some(f), Some(t)) = (s.free_bytes, s.total_bytes) {
            free.push(format!("free now {} of {}", gib(f), gib(t)));
        }
        if !free.is_empty() {
            lines.push(format!("  {}", free.join(" · ")));
        }
        for (name, f, t) in &s.tp_devices {
            lines.push(format!("  {name}: free now {} of {}", gib(*f), gib(*t)));
        }
    }
    // Host memory by node, then what is not bound to one.
    let mut nodes: Vec<Option<u32>> = load.host.iter().map(|h| h.node).collect();
    let tables = [("n-gram", &load.ngram), ("Engram", &load.engram)];
    nodes.extend(
        tables
            .iter()
            .filter_map(|(_, t)| t.as_ref().map(|t| t.node)),
    );
    nodes.extend(numa.iter().map(|n| Some(n.id)));
    nodes.sort_by_key(|n| (n.is_none(), *n));
    nodes.dedup();
    for node in nodes {
        let mut held: Vec<String> = load
            .host
            .iter()
            .filter(|h| h.node == node)
            .map(host_use)
            .collect();
        for (what, n) in tables {
            let Some(n) = n.as_ref().filter(|n| n.node == node) else {
                continue;
            };
            let table = if what == "Engram" { "tables" } else { "table" };
            held.push(match n.io.as_str() {
                "ram" if n.locked => format!("{what} {table} {} (locked)", gib(n.bytes)),
                "ram" => format!("{what} {table} {} (not locked)", gib(n.bytes)),
                "direct" if n.hot_rows > 0 => {
                    format!("{what} RAM budget {}", gib(n.ram_cache_bytes))
                }
                "direct" => format!("{what} row cache {}", gib(n.ram_cache_bytes)),
                _ => format!("{what} {table} {} ({})", gib(n.bytes), n.io),
            });
        }
        let info = node.and_then(|id| numa.iter().find(|n| n.id == id));
        if held.is_empty() && info.is_none() {
            continue;
        }
        if held.is_empty() {
            held.push("nothing pinned by the load".into());
        }
        if let Some(info) = info {
            held.push(format!(
                "RAM free now {} of {}",
                gib(info.mem_free),
                gib(info.mem_total)
            ));
        }
        let name = node.map_or_else(
            || "host, no NUMA node bound".into(),
            |n| format!("NUMA node {n}"),
        );
        lines.push(format!("{name}: {}", held.join(" · ")));
    }
    lines
}

fn host_use(h: &HostMemory) -> String {
    let what = match h.kind {
        HostUse::ExpertArena => "pinned expert arena",
        HostUse::MappedExperts => "routed experts mapped from the file",
        HostUse::SharedWeights => "pinned shared weights (embeddings)",
        HostUse::HostWeights => "host weights",
        HostUse::KvHostTier => "KV host tier",
        HostUse::SavedStates => "pinned saved prefix-cache states",
    };
    match h.tokens {
        Some(t) => format!("{what} {} ({} tokens)", gib(h.bytes), count(t)),
        None => format!("{what} {}", gib(h.bytes)),
    }
}

pub(super) fn kv_lines(load: &LoadReport) -> Vec<String> {
    let kv = &load.kv;
    let mut lines = vec![format!(
        "context {} tokens ({}) · types {} · {} per token",
        count(kv.ctx_size),
        if kv.native {
            "model native"
        } else {
            "--ctx-size"
        },
        kv.types,
        bytes(kv.bytes_per_token)
    )];
    let host = if kv.host_tokens > 0 {
        format!(
            " · host tier {} tokens ({}){}",
            count(kv.host_tokens),
            if kv.host_auto {
                "auto"
            } else {
                "--kv-host-tokens"
            },
            if kv.staging_tokens > 0 {
                format!(
                    ", staged through {} device tokens",
                    count(kv.staging_tokens)
                )
            } else {
                String::new()
            }
        )
    } else {
        " · no host tier".into()
    };
    lines.push(format!(
        "{} tokens in device memory{host}",
        count(kv.device_tokens)
    ));
    if let Some((sets, positions)) = kv.rings {
        lines.push(format!(
            "sliding-window layers: {sets} rings of {} positions",
            count(positions)
        ));
    }
    lines
}

pub(super) fn expert_lines(load: &LoadReport) -> Vec<String> {
    let (Some(e), Some(mode)) = (&load.experts, &load.expert_mode) else {
        return Vec::new();
    };
    let mut lines = vec![format!(
        "{} per layer, {} per token · {}",
        e.n_expert,
        e.n_used,
        expert_mode(mode)
    )];
    if e.cached_layers > 0 {
        let resident = if e.slots_min == e.slots_max {
            count(e.slots_min)
        } else {
            format!("{}-{}", e.slots_min, e.slots_max)
        };
        let streamed_min = e.n_expert.saturating_sub(e.slots_max);
        let streamed_max = e.n_expert.saturating_sub(e.slots_min);
        let streamed = if streamed_min == streamed_max {
            count(streamed_min)
        } else {
            format!("{streamed_min}-{streamed_max}")
        };
        lines.push(format!(
            "slot cache in {} layers: {resident} resident, {streamed} read from host memory per layer",
            e.cached_layers
        ));
    }
    if mode != "device" {
        lines.push(format!(
            "decode and verify steps read host experts {}",
            if e.fetch == "copy" {
                "by copying the selected ones to VRAM first"
            } else {
                "in place"
            }
        ));
    }
    lines
}

/// A lookup table's placement: the n-gram table's, or the Engram tables'.
pub(super) fn table_lines(table: Option<&NgramPlacement>) -> Vec<String> {
    let Some(n) = table else {
        return Vec::new();
    };
    let node = n
        .node
        .map_or_else(|| "no NUMA node bound".into(), |n| format!("NUMA node {n}"));
    let tier = match n.io.as_str() {
        "ram" if n.locked => format!("RAM, locked, {node}"),
        "ram" => format!("RAM, not locked, {node}"),
        "direct" if n.hot_rows > 0 => format!(
            "direct SSD reads, {} RAM budget on {node}: hot set up to {} ({} rows, the rows \
             hit most), the rest a row cache",
            gib(n.ram_cache_bytes),
            gib(n.hot_bytes),
            count(n.hot_rows as usize)
        ),
        "direct" => format!(
            "direct SSD reads, {} RAM row cache on {node}",
            gib(n.ram_cache_bytes)
        ),
        "mmap" => "mapped from the file (page cache)".into(),
        other => other.to_string(),
    };
    vec![
        format!("{} · {tier}", gib(n.bytes)),
        format!("why: {}", n.reason),
    ]
}

fn expert_mode(mode: &str) -> &str {
    match mode {
        "device" => "all in VRAM",
        "host" => "host memory, read in place",
        "cache" => "host memory, hottest cached in VRAM",
        other => other,
    }
}

fn device_name(device: &str, description: Option<&str>) -> String {
    match description {
        Some(d) if d != device => format!("{device} ({d})"),
        _ => device.to_string(),
    }
}

fn nvlinks(gpu: &legion::hw::gpu::GpuInfo) -> String {
    let up: Vec<_> = gpu.nvlinks.iter().filter(|l| l.up).collect();
    if up.is_empty() {
        return if gpu.nvlinks.is_empty() {
            "none reported".into()
        } else {
            "no active links".into()
        };
    }
    let mut remotes = std::collections::BTreeMap::<&str, usize>::new();
    for l in &up {
        *remotes
            .entry(l.remote.as_deref().unwrap_or("unknown"))
            .or_default() += 1;
    }
    let to = remotes
        .iter()
        .map(|(r, n)| format!("{n} to {r}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} active links: {to}", up.len())
}

fn memory_kind(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::IntegratedShared => "integrated GPU sharing host memory",
        MemoryKind::CoherentNvlink => "coherent host memory over NVLink (ATS)",
        MemoryKind::Discrete => "discrete VRAM",
    }
}

/// The CPU's model from `/proc/cpuinfo`: x86's `model name`, POWER's `cpu`, and on aarch64
/// `model name` where the kernel gives it, else the implementer and part numbers.
pub(super) fn cpu_model(text: &str) -> Option<String> {
    let field = |key: &str| {
        text.lines().find_map(|line| {
            let (k, v) = line.split_once(':')?;
            (k.trim() == key && !v.trim().is_empty()).then(|| v.trim().to_string())
        })
    };
    field("model name")
        .or_else(|| field("cpu"))
        .or_else(|| {
            let part = field("CPU part")?;
            Some(match field("CPU implementer") {
                Some(imp) => format!("implementer {imp} part {part}"),
                None => format!("part {part}"),
            })
        })
        .or_else(|| field("Model"))
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn bytes(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KiB", n as f64 / (1u64 << 10) as f64)
    } else {
        format!("{n} B")
    }
}

/// `131072` as `131,072`.
pub(super) fn count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn opt(value: Option<u32>) -> String {
    value.map_or_else(|| "unknown".into(), |n| n.to_string())
}

#[cfg(test)]
mod tests;
