//! The stats hub's pieces without a GPU: link gauges from fake NVML samples (a PCIe card, an
//! AC922's V100 with NVLink to the CPU and to peers, a GB10 on C2C), request rows, snapshots.

use std::time::{Duration, Instant};

use super::*;
use crate::dashboard::scale::LinkClass;
use crate::hw::counters::GpuInfo;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6 * b.abs().max(1.0)
}

/// An RTX 4090 as NVML reports it: PCIe 4.0 x16, 384-bit GDDR6X at 10501 MHz.
fn pcie_card(counted: bool) -> GpuLive {
    let mut g = GpuLive {
        index: 0,
        info: GpuInfo {
            name: "NVIDIA GeForce RTX 4090".into(),
            pcie_max_gen: Some(4),
            pcie_max_width: Some(16),
            mem_bus_bits: Some(384),
            max_mem_mhz: Some(10501),
            ..GpuInfo::default()
        },
        pcie_counted: counted,
        ..GpuLive::default()
    };
    g.totals.pcie_tx_bytes = 1.5e9;
    g.totals.pcie_rx_bytes = 3.0e9;
    g
}

/// An AC922 V100: three NVLink 2 links to the POWER9, three to the other GPU.
fn ac922_v100(index: u32) -> GpuLive {
    GpuLive {
        index,
        info: GpuInfo {
            name: "Tesla V100-SXM2-16GB".into(),
            pcie_max_gen: Some(3),
            pcie_max_width: Some(16),
            nvlinks: vec![
                LinkClass::Cpu,
                LinkClass::Cpu,
                LinkClass::Cpu,
                LinkClass::Peer,
                LinkClass::Peer,
                LinkClass::Peer,
            ],
            nvlink_version: Some(2),
            mem_bus_bits: Some(4096),
            max_mem_mhz: Some(877),
            ..GpuInfo::default()
        },
        nvlink_cpu: (4e9, 9e9),
        nvlink_peer: (1e9, 2e9),
        ..GpuLive::default()
    }
}

/// A DGX Spark's GB10: unified memory on C2C; NVML reports no C2C links or memory clock.
fn gb10(c2c: Option<(f64, f64)>) -> GpuLive {
    GpuLive {
        index: 0,
        info: GpuInfo {
            name: "NVIDIA GB10".into(),
            unified: true,
            gpm: true,
            ..GpuInfo::default()
        },
        c2c,
        gpm_dram_pct: Some(40.0),
        ..GpuLive::default()
    }
}

fn share(link: u64, device: u64) -> PhaseBytes {
    PhaseBytes {
        link,
        device,
        seconds: 2.0,
        steps: 10,
        ..PhaseBytes::default()
    }
}

#[test]
fn a_pcie_card_gets_one_pcie_gauge_at_gen_times_width() {
    let links = links_of(&pcie_card(true), &share(7e9 as u64, 0));
    assert_eq!(links.len(), 1);
    let l = &links[0];
    assert_eq!(l.kind, LinkKind::Pcie);
    assert_eq!(l.detail, "PCIe 4.0 x16");
    assert!(close(l.full_scale_gb_s.unwrap(), 1.969 * 16.0));
    assert_eq!(l.tx_bytes, Some(1.5e9));
    assert_eq!(l.rx_bytes, Some(3.0e9));
    assert_eq!(l.legion_rx_bytes, 7e9);
}

#[test]
fn pcie_without_nvml_throughput_has_no_needles_but_keeps_legion_bytes() {
    let links = links_of(&pcie_card(false), &share(5, 0));
    assert_eq!(links[0].tx_bytes, None);
    assert_eq!(links[0].rx_bytes, None);
    assert_eq!(links[0].legion_rx_bytes, 5.0);
    assert!(links[0].full_scale_gb_s.is_some());
}

#[test]
fn an_unknown_pcie_generation_has_no_full_scale() {
    let mut g = pcie_card(true);
    g.info.pcie_max_gen = None;
    let links = links_of(&g, &share(0, 0));
    assert_eq!(links[0].full_scale_gb_s, None);
    assert_eq!(links[0].detail, "");
}

#[test]
fn an_ac922_gpu_splits_nvlink_into_cpu_and_peer() {
    let links = links_of(&ac922_v100(0), &share(11, 0));
    assert_eq!(links.len(), 2);
    assert_eq!(links[0].kind, LinkKind::NvlinkCpu);
    assert_eq!(links[0].detail, "3 links NVLink 2");
    assert_eq!(links[0].full_scale_gb_s, Some(75.0));
    assert_eq!(links[0].tx_bytes, Some(4e9));
    assert_eq!(links[0].rx_bytes, Some(9e9));
    assert_eq!(links[0].legion_rx_bytes, 11.0);
    assert_eq!(links[1].kind, LinkKind::NvlinkPeer);
    assert_eq!(links[1].full_scale_gb_s, Some(75.0));
    assert_eq!(links[1].tx_bytes, Some(1e9));
    // Peer copies are not model bytes.
    assert_eq!(links[1].legion_rx_bytes, 0.0);
}

#[test]
fn a_gb10_gets_a_c2c_gauge_measured_only_with_gpm_bytes() {
    let without = links_of(&gb10(None), &share(3, 0));
    assert_eq!(without.len(), 1);
    assert_eq!(without[0].kind, LinkKind::C2c);
    assert_eq!(without[0].tx_bytes, None);
    assert_eq!(without[0].legion_rx_bytes, 3.0);
    let with = links_of(&gb10(Some((2e9, 6e9))), &share(3, 0));
    assert_eq!(with[0].tx_bytes, Some(2e9));
    assert_eq!(with[0].rx_bytes, Some(6e9));
}

#[test]
fn c2c_full_scale_comes_from_nvml_links_when_reported() {
    let mut g = gb10(None);
    g.info.c2c_links = Some(4);
    g.info.c2c_max_mb_s = Some(100_000);
    let links = links_of(&g, &share(0, 0));
    assert_eq!(links[0].full_scale_gb_s, Some(400.0));
    assert_eq!(links[0].detail, "4 links");
}

#[test]
fn gpu_snaps_take_the_engines_gpus_and_share_its_bytes() {
    let live = [ac922_v100(0), ac922_v100(1), ac922_v100(2)];
    let header = Header {
        gpus: 2,
        ..Header::default()
    };
    let snaps = gpu_snaps(&live, &EngineStats::default(), &header);
    assert_eq!(snaps.len(), 2);
    assert_eq!(snaps[1].index, 1);
    assert!(close(
        snaps[0].memory.peak_gb_s.unwrap(),
        4096.0 / 8.0 * 877.0 * 2.0 / 1e3
    ));
    assert_eq!(snaps[0].links.len(), 2);
    // No GPU count: every GPU NVML sees.
    let all = gpu_snaps(&live, &EngineStats::default(), &Header::default());
    assert_eq!(all.len(), 3);
}

#[test]
fn a_gb10_memory_gauge_uses_the_spec_peak_and_gpm() {
    let snaps = gpu_snaps(&[gb10(None)], &EngineStats::default(), &Header::default());
    let m = &snaps[0].memory;
    assert_eq!(m.peak_gb_s, Some(273.0));
    assert_eq!(m.gpm_dram_pct, Some(40.0));
    assert!(snaps[0].unified);
}

#[test]
fn a_unified_parts_nvml_memory_utilization_is_not_reported() {
    let mut g = gb10(None);
    g.mem_util_pct = Some(0.0);
    let snaps = gpu_snaps(&[g], &EngineStats::default(), &Header::default());
    assert_eq!(snaps[0].mem_busy_pct, None);
    let mut v = ac922_v100(0);
    v.mem_util_pct = Some(44.0);
    let snaps = gpu_snaps(&[v], &EngineStats::default(), &Header::default());
    assert_eq!(snaps[0].mem_busy_pct, Some(44.0));
}

#[test]
fn the_share_is_per_stage_with_one_stage_per_gpu_else_even() {
    let mut e = EngineSnap {
        prefill: share(10, 100),
        decode: share(10, 100),
        ..EngineSnap::default()
    };
    let even = legion_share(&e, 1, 2);
    assert_eq!(even.link, 10);
    assert_eq!(even.device, 100);
    e.stages = vec![share(1, 2), share(3, 4)];
    let staged = legion_share(&e, 1, 2);
    assert_eq!(staged.link, 3);
    assert_eq!(staged.device, 4);
}

#[test]
fn endpoints_by_route() {
    assert_eq!(endpoint_of("/v1/chat/completions"), "chat");
    assert_eq!(endpoint_of("/v1/completions"), "completions");
    assert_eq!(endpoint_of("/v1/messages"), "messages");
    assert_eq!(endpoint_of("/v1/responses"), "responses");
    assert_eq!(endpoint_of("/v1/models"), "other");
}

#[test]
fn a_request_goes_queued_prefill_decode_and_leaves() {
    let hub = StatsHub::new(None);
    let r = hub.begin();
    let now = Instant::now();
    let row = |hub: &StatsHub, at: Instant| {
        hub.snapshot(&EngineStats::default(), Err("test".into()), at)
            .requests
    };
    let rows = row(&hub, now);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].state, "queued");
    assert_eq!(rows[0].endpoint, "other");
    assert_eq!(rows[0].prompt_tokens, None);
    r.started(1234);
    assert_eq!(row(&hub, now)[0].state, "prefill");
    assert_eq!(row(&hub, now)[0].prompt_tokens, Some(1234));
    let first = r.start + Duration::from_millis(250);
    r.output(first);
    for i in 1..=10 {
        r.output(first + Duration::from_millis(10 * i));
    }
    let later = first + Duration::from_secs(1);
    let rows = row(&hub, later);
    assert_eq!(rows[0].state, "decode");
    assert_eq!(rows[0].generated, 11);
    assert_eq!(rows[0].ttft_ms, Some(250));
    // Ten tokens after the first, over the second since it.
    assert!(close(rows[0].tok_s.unwrap(), 10.0));
    hub.end(&r);
    assert!(row(&hub, later).is_empty());
}

#[tokio::test]
async fn requests_take_their_endpoint_from_the_middleware() {
    let hub = StatsHub::new(None);
    let r = ENDPOINT.scope("chat", async { hub.begin() }).await;
    assert_eq!(r.endpoint, "chat");
}

#[test]
fn a_snapshot_without_a_sampler_says_why() {
    let hub = StatsHub::new(None);
    hub.set_header(Header {
        model: "m".into(),
        ..Header::default()
    });
    let s = hub.snapshot(
        &EngineStats {
            prompt_tokens: 5,
            ..EngineStats::default()
        },
        Err("NVML: not found".into()),
        Instant::now(),
    );
    assert_eq!(s.version, VERSION);
    assert_eq!(s.header.model, "m");
    assert_eq!(s.engine.prompt_tokens, 5);
    assert!(!s.hw.sampling);
    assert_eq!(s.hw.reason.as_deref(), Some("NVML: not found"));
    assert!(s.gpus.is_empty());
}

#[test]
fn latencies_land_in_the_snapshot() {
    let hub = StatsHub::new(None);
    let now = Instant::now();
    for _ in 0..10 {
        hub.ttft.record(now, 0.2);
        hub.itl.record(now, 0.02);
    }
    let s = hub.snapshot(&EngineStats::default(), Err(String::new()), now);
    assert_eq!(s.latency.ttft_samples, 10);
    assert_eq!(s.latency.itl_samples, 10);
    let p50 = s.latency.ttft_p50_ms.unwrap();
    assert!((200.0..=240.0).contains(&p50), "{p50}");
}
