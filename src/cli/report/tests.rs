use super::*;
use legion::engine::lm::report::{
    ExpertSummary, HostMemory, HostUse, KvSummary, LoadReport, NgramPlacement, StageMemory,
};
use legion::hw::gpu::{GpuInfo, GpuReport, NvLinkInfo};
use legion::hw::{NumaNodeInfo, UnifiedMemory};

const GIB: u64 = 1 << 30;

fn gpu(index: u32, node: Option<u32>, nvlinks: Vec<NvLinkInfo>) -> GpuInfo {
    GpuInfo {
        index,
        name: "Tesla V100-SXM2-16GB".into(),
        uuid: format!("GPU-{index}"),
        pci_bus_id: format!("0004:0{index}:00.0"),
        compute_capability: Some((7, 0)),
        memory_total: 16 * GIB,
        memory_free: 15 * GIB,
        pcie_gen: Some(3),
        pcie_width: Some(16),
        pcie_max_gen: None,
        pcie_max_width: None,
        numa_node: node,
        nvlinks,
    }
}

fn node(id: u32, total: u64, free: u64) -> NumaNodeInfo {
    NumaNodeInfo {
        id,
        cpus: 64,
        cpu_list: String::new(),
        mem_total: total * GIB,
        mem_free: free * GIB,
    }
}

fn hardware(
    gpu: Result<Vec<GpuInfo>, String>,
    kind: MemoryKind,
    nodes: Vec<NumaNodeInfo>,
) -> HardwareInfo {
    let (gpu, gpu_error) = match gpu {
        Ok(gpus) => (
            Some(GpuReport {
                driver_version: None,
                cuda_driver_version: None,
                gpus,
            }),
            None,
        ),
        Err(e) => (None, Some(e)),
    };
    HardwareInfo {
        arch: "ppc64le".into(),
        gpu,
        gpu_error,
        numa_nodes: nodes,
        unified_memory: UnifiedMemory {
            kind,
            reason: String::new(),
        },
    }
}

fn link(remote: &str) -> NvLinkInfo {
    NvLinkInfo {
        link: 0,
        up: true,
        remote: Some(remote.into()),
    }
}

#[test]
fn no_nvml_says_why_and_still_lists_the_host() {
    let hw = hardware(
        Err("libnvidia-ml.so not found".into()),
        MemoryKind::Discrete,
        vec![node(0, 64, 60)],
    );
    let lines = hardware_lines(&hw, 8, None);
    assert_eq!(
        lines,
        [
            "GPUs: unknown (NVML: libnvidia-ml.so not found)",
            "NUMA node 0: RAM 64.0 GiB total / 60.0 GiB free · 64 CPUs",
            "CPU: unknown model (ppc64le) · 8 threads",
        ]
    );
    let report = StartupReport::new("m").with_hardware(&hw, 8);
    assert_eq!(report.sections[0].title, "Hardware");
}

#[test]
fn zero_gpus() {
    let hw = hardware(Ok(vec![]), MemoryKind::Discrete, vec![]);
    assert_eq!(
        hardware_lines(&hw, 0, Some("AMD Ryzen 9")),
        ["GPUs: none", "CPU: AMD Ryzen 9 (ppc64le) · default threads"]
    );
}

#[test]
fn a_gpu_whose_numa_node_is_unknown() {
    let mut g = gpu(0, None, vec![]);
    g.pci_bus_id.clear();
    let hw = hardware(Ok(vec![g]), MemoryKind::Discrete, vec![]);
    let lines = hardware_lines(&hw, 0, None);
    assert_eq!(
        lines[1],
        "  PCI unknown · NUMA node unknown · PCIe gen 3 x16 (max gen unknown xunknown)"
    );
    assert_eq!(lines[2], "  NVLink none reported");
    assert_eq!(lines[3], "memory: discrete VRAM");
    assert!(
        lines
            .iter()
            .all(|l| !l.contains("None") && !l.contains("Some"))
    );
}

/// An AC922: two sockets, two V100s on each, NVLink to the CPU and to the peer GPU.
#[test]
fn power9_two_node_fixture() {
    let links = || {
        vec![
            link("cpu"),
            link("cpu"),
            link("cpu"),
            link("gpu"),
            link("gpu"),
            link("gpu"),
        ]
    };
    let gpus = vec![
        gpu(0, Some(0), links()),
        gpu(1, Some(0), links()),
        gpu(2, Some(8), links()),
        gpu(3, Some(8), links()),
    ];
    let hw = hardware(
        Ok(gpus),
        MemoryKind::CoherentNvlink,
        vec![node(0, 256, 200), node(8, 256, 210)],
    );
    let lines = hardware_lines(
        &hw,
        16,
        cpu_model("cpu\t\t: POWER9, altivec supported\n").as_deref(),
    );
    assert_eq!(lines.len(), 4 * 3 + 1 + 2 + 1);
    assert_eq!(
        lines[0],
        "GPU 0: Tesla V100-SXM2-16GB · compute 7.0 · VRAM 16.0 GiB total / 15.0 GiB free"
    );
    assert_eq!(lines[2], "  NVLink 6 active links: 3 to cpu, 3 to gpu");
    assert!(lines[7].contains("NUMA node 8"), "{}", lines[7]);
    assert_eq!(lines[12], "memory: coherent host memory over NVLink (ATS)");
    assert_eq!(
        lines[14],
        "NUMA node 8: RAM 256.0 GiB total / 210.0 GiB free · 64 CPUs"
    );
    assert_eq!(
        lines[15],
        "CPU: POWER9, altivec supported (ppc64le) · 16 threads"
    );
}

#[test]
fn cpu_models_of_x86_power9_and_aarch64() {
    let x86 =
        "processor\t: 0\ncpu family\t: 25\nmodel name\t: AMD Ryzen 9 7950X\ncpu MHz\t\t: 4500\n";
    assert_eq!(cpu_model(x86).as_deref(), Some("AMD Ryzen 9 7950X"));
    let power = "processor\t: 0\ncpu\t\t: POWER9 (raw), altivec supported\nclock\t\t: 3800MHz\n";
    assert_eq!(
        cpu_model(power).as_deref(),
        Some("POWER9 (raw), altivec supported")
    );
    let arm = "processor\t: 0\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU part\t: 0xd85\n";
    assert_eq!(
        cpu_model(arm).as_deref(),
        Some("implementer 0x41 part 0xd85")
    );
    assert_eq!(cpu_model("processor\t: 0\n"), None);
}

/// Two V100 stages of a MoE with host experts in arenas on both sockets, a KV host tier and
/// the n-gram table locked on node 0.
fn load_fixture() -> LoadReport {
    let stage = |i: usize, layers: std::ops::Range<usize>, weights: u64| StageMemory {
        device: format!("CUDA{i}"),
        description: Some("Tesla V100-SXM2-16GB".into()),
        layers,
        weights,
        kv_tokens: 32768,
        kv_bytes: 2 * GIB,
        kv_staging_bytes: GIB / 2,
        states_bytes: GIB / 4,
        expert_layers: 20,
        expert_slots: 1280,
        expert_slot_bytes: 6 * GIB,
        expert_staging_bytes: Some(GIB),
        compute_bytes: Some(GIB + GIB / 2),
        kv_reserve_bytes: Some(2 * GIB),
        expert_reserve_bytes: Some(3 * GIB),
        free_bytes: Some(GIB / 2),
        total_bytes: Some(16 * GIB),
        tp_devices: vec![],
    };
    LoadReport {
        expert_mode: Some("cache".into()),
        tp: 0,
        stages: vec![stage(0, 0..20, 3 * GIB), stage(1, 20..40, 4 * GIB)],
        host: vec![
            HostMemory {
                node: Some(0),
                kind: HostUse::ExpertArena,
                bytes: 40 * GIB,
                tokens: None,
            },
            HostMemory {
                node: Some(0),
                kind: HostUse::KvHostTier,
                bytes: 8 * GIB,
                tokens: Some(98304),
            },
            HostMemory {
                node: Some(8),
                kind: HostUse::ExpertArena,
                bytes: 41 * GIB,
                tokens: None,
            },
            HostMemory {
                node: None,
                kind: HostUse::SharedWeights,
                bytes: GIB,
                tokens: None,
            },
        ],
        kv: KvSummary {
            ctx_size: 131072,
            native: true,
            device_tokens: 32768,
            host_tokens: 98304,
            staging_tokens: 16384,
            host_auto: true,
            types: "f16/f16".into(),
            bytes_per_token: 40960,
            rings: None,
        },
        experts: Some(ExpertSummary {
            n_expert: 256,
            n_used: 8,
            cached_layers: 40,
            slots_min: 60,
            slots_max: 68,
            fetch: "in-place".into(),
        }),
        ngram: Some(NgramPlacement {
            bytes: 27 * GIB,
            io: "ram".into(),
            locked: true,
            node: Some(0),
            ram_cache_bytes: 0,
            reason: "fits in node 0's free memory".into(),
        }),
        engram: None,
    }
}

#[test]
fn loaded_report_shows_every_stage_and_node() {
    let numa = [node(0, 256, 120), node(8, 256, 200)];
    let text = StartupReport::loaded("Qwen3.8-Flash-Next")
        .with_load(&load_fixture(), &numa)
        .render(false);
    assert_eq!(
        text,
        "LEGION SERVE · LOADED  Qwen3.8-Flash-Next

Placement
  2 pipeline stages · routed experts: host memory, hottest cached in VRAM
  stage 0 · CUDA0 (Tesla V100-SXM2-16GB): layers 0..20 (20) · weights 3.0 GiB
  stage 1 · CUDA1 (Tesla V100-SXM2-16GB): layers 20..40 (20) · weights 4.0 GiB

Memory
  stage 0 · CUDA0 (Tesla V100-SXM2-16GB)
    weights 3.0 GiB · KV 2.0 GiB (32,768 tokens) · states 0.2 GiB · KV host-tier staging 0.5 GiB
    expert slots 6.0 GiB (1,280 slots in 20 layers) · long-prompt staging 1.0 GiB (lent by the slots)
    step compute buffers 1.5 GiB (measured) · KV fit reserve 2.0 GiB · expert cache reserve 3.0 GiB · free now 0.5 GiB of 16.0 GiB
  stage 1 · CUDA1 (Tesla V100-SXM2-16GB)
    weights 4.0 GiB · KV 2.0 GiB (32,768 tokens) · states 0.2 GiB · KV host-tier staging 0.5 GiB
    expert slots 6.0 GiB (1,280 slots in 20 layers) · long-prompt staging 1.0 GiB (lent by the slots)
    step compute buffers 1.5 GiB (measured) · KV fit reserve 2.0 GiB · expert cache reserve 3.0 GiB · free now 0.5 GiB of 16.0 GiB
  NUMA node 0: pinned expert arena 40.0 GiB · KV host tier 8.0 GiB (98,304 tokens) · n-gram table 27.0 GiB (locked) · RAM free now 120.0 GiB of 256.0 GiB
  NUMA node 8: pinned expert arena 41.0 GiB · RAM free now 200.0 GiB of 256.0 GiB
  host, no NUMA node bound: pinned shared weights (embeddings) 1.0 GiB

KV cache
  context 131,072 tokens (model native) · types f16/f16 · 40.0 KiB per token
  32,768 tokens in device memory · host tier 98,304 tokens (auto), staged through 16,384 device tokens

Experts
  256 per layer, 8 per token · host memory, hottest cached in VRAM
  slot cache in 40 layers: 60-68 resident, 188-196 read from host memory per layer
  decode and verify steps read host experts in place

N-gram table
  27.0 GiB · RAM, locked, NUMA node 0
  why: fits in node 0's free memory
"
    );
    assert!(!text.contains("auto tier") && !text.contains("Some(") && !text.contains("None"));
}

#[test]
fn unknown_and_absent_values_are_said_so_or_left_out() {
    let mut load = load_fixture();
    load.expert_mode = None;
    load.experts = None;
    load.ngram = Some(NgramPlacement {
        bytes: 27 * GIB,
        io: "direct".into(),
        locked: false,
        node: None,
        ram_cache_bytes: 4 * GIB,
        reason: "not enough free memory to lock it".into(),
    });
    for s in &mut load.stages {
        s.free_bytes = None;
        s.compute_bytes = None;
        s.expert_layers = 0;
    }
    let text = StartupReport::loaded("m")
        .with_load(&load, &[])
        .render(false);
    assert!(!text.contains("\nExperts\n"), "{text}");
    assert!(!text.contains("free now 0.5"), "{text}");
    assert!(
        text.contains("27.0 GiB · direct SSD reads, 4.0 GiB RAM row cache on no NUMA node bound"),
        "{text}"
    );
    assert!(text.contains("host, no NUMA node bound: pinned shared weights (embeddings) 1.0 GiB · n-gram row cache 4.0 GiB"), "{text}");
}

#[test]
fn ansi_is_present_only_when_enabled() {
    let report = StartupReport::loaded("m").with_load(&load_fixture(), &[]);
    assert!(report.render(true).contains("\x1b[36m"));
    assert!(!report.render(false).contains("\x1b["));
}

#[test]
fn colour_and_the_live_line_follow_the_terminal() {
    let tty = TermEnv::default();
    let dumb = TermEnv { dumb: true, ..tty };
    let no_color = TermEnv {
        no_color: true,
        ..tty
    };
    assert!(ColorChoice::Auto.use_color(true, tty));
    assert!(!ColorChoice::Auto.use_color(false, tty));
    assert!(!ColorChoice::Auto.use_color(true, dumb));
    assert!(ColorChoice::Always.use_color(false, dumb));
    assert!(!ColorChoice::Auto.use_color(true, no_color));
    assert!(ColorChoice::Always.use_color(true, no_color));
    assert!(StatusLineChoice::Auto.live(true, tty, false));
    assert!(!StatusLineChoice::Auto.live(false, tty, false));
    // The verified bug: TERM=dumb NO_COLOR=1 on a TTY drew the line.
    let both = TermEnv {
        dumb: true,
        no_color: true,
    };
    assert!(!StatusLineChoice::Auto.live(true, both, false));
    assert!(StatusLineChoice::On.live(false, dumb, false));
    assert!(!StatusLineChoice::On.live(true, tty, true));
    assert!(!StatusLineChoice::Off.live(true, tty, false));
}

#[test]
fn counts_have_separators() {
    assert_eq!(count(0), "0");
    assert_eq!(count(999), "999");
    assert_eq!(count(131072), "131,072");
    assert_eq!(count(1048576), "1,048,576");
}

#[test]
fn engram_tables_get_their_own_section_and_host_memory() {
    let mut load = load_fixture();
    load.ngram = None;
    load.engram = Some(NgramPlacement {
        bytes: 103 * GIB,
        io: "direct".into(),
        locked: false,
        node: Some(8),
        ram_cache_bytes: 8 * GIB,
        reason: "unified GPU memory: read rows from the SSD".into(),
    });
    let numa = [node(0, 256, 120), node(8, 256, 200)];
    let text = StartupReport::loaded("DeepSeek-V4.1-Flash")
        .with_load(&load, &numa)
        .render(false);
    assert!(!text.contains("N-gram table"), "{text}");
    assert!(
        text.contains(
            "\nEngram tables\n  103.0 GiB · direct SSD reads, 8.0 GiB RAM row cache on NUMA node 8\n  why: unified GPU memory: read rows from the SSD\n"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "NUMA node 8: pinned expert arena 41.0 GiB · Engram row cache 8.0 GiB · RAM free"
        ),
        "{text}"
    );
    load.engram.as_mut().unwrap().io = "ram".into();
    load.engram.as_mut().unwrap().locked = true;
    let text = StartupReport::loaded("m")
        .with_load(&load, &numa)
        .render(false);
    assert!(text.contains("Engram tables 103.0 GiB (locked)"), "{text}");
}
