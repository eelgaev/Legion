use super::*;

fn metrics(scale: f64) -> String {
    let mut s = String::from("# HELP x\n# TYPE x gauge\nlegion_engine_running 0\n");
    for (phase, bytes, secs) in [
        ("prefill", 0.0, 0.0),
        ("decode", 10e9, 0.1),
        ("draft", 0.0, 0.0),
    ] {
        s.push_str(&format!(
            "legion_engine_model_bytes{{phase=\"{phase}\"}} {}\n\
             legion_engine_model_seconds{{phase=\"{phase}\"}} {}\n",
            bytes * scale,
            secs * scale
        ));
    }
    for (kind, gb) in [
        ("dense", 2.0),
        ("experts_upper", 0.0),
        ("experts_device", 5.0),
        ("experts_host", 1.0),
        ("staged", 0.5),
        ("kv", 1.5),
        ("kv_host", 0.0),
    ] {
        s.push_str(&format!(
            "legion_engine_model_kind_bytes{{phase=\"decode\",kind=\"{kind}\"}} {}\n",
            gb * 1e9 * scale
        ));
    }
    s.push_str("legion_engine_roofline_memory_gb_per_second{stage=\"0\"} 894.0\n");
    for gpu in ["2", "3"] {
        s.push_str(&format!(
            "legion_gpu_sampled_seconds{{gpu=\"{gpu}\"}} {}\n\
             legion_gpu_nvlink_rx_bytes{{gpu=\"{gpu}\"}} {}\n\
             legion_gpu_memory_busy_seconds{{gpu=\"{gpu}\"}} {}\n",
            2.0 * scale,
            4e9 * scale,
            scale
        ));
    }
    s
}

#[test]
fn scrape_finds_samples_whatever_the_label_order() {
    let s = Scrape::parse(&metrics(1.0));
    assert_eq!(s.get("legion_engine_running", &[]), Some(0.0));
    assert_eq!(
        s.get(
            "legion_engine_model_kind_bytes",
            &[("kind", "kv"), ("phase", "decode")]
        ),
        Some(1.5e9)
    );
    assert_eq!(s.get("x", &[]), None);
    assert_eq!(s.values_of("legion_gpu_sampled_seconds", "gpu"), ["2", "3"]);
}

#[test]
fn model_bytes_split_device_memory_and_host_link() {
    let (a, b) = (Scrape::parse(&metrics(1.0)), Scrape::parse(&metrics(3.0)));
    let m = model_bytes_between(&a, &b).unwrap();
    let d = m.decode;
    assert!((d.gb - 20.0).abs() < 1e-9);
    assert!((d.seconds - 0.2).abs() < 1e-12);
    assert!((d.gb_s - 100.0).abs() < 1e-9);
    // Device: dense 2 + experts 5 + KV 1.5, doubled over the window.
    assert!((d.device_gb - 17.0).abs() < 1e-9);
    assert!((d.link_gb - 3.0).abs() < 1e-9);
    assert!((d.experts_gb - 12.0).abs() < 1e-9);
    assert!((d.kv_gb - 3.0).abs() < 1e-9);
    assert!((d.device_gb_s - 85.0).abs() < 1e-9);
    assert!((d.roofline_pct.unwrap() - 85.0 / 894.0 * 100.0).abs() < 1e-9);
    assert_eq!(m.prefill.gb, 0.0);
    assert_eq!(m.prefill.gb_s, 0.0);
    assert_eq!(m.roofline_memory_gb_s, 894.0);
    let none = Scrape::parse("legion_engine_running 0\n");
    assert!(model_bytes_between(&none, &none).is_none());
}

#[test]
fn servers_without_kinds_count_all_bytes_as_device_memory() {
    let old = |b: f64| {
        Scrape::parse(&format!(
            "legion_engine_model_bytes{{phase=\"decode\"}} {b}\n\
             legion_engine_model_seconds{{phase=\"decode\"}} {}\n",
            b / 1e10
        ))
    };
    let m = model_bytes_between(&old(1e9), &old(5e9)).unwrap();
    assert!((m.decode.device_gb - 4.0).abs() < 1e-9);
    assert_eq!(m.decode.roofline_pct, None);
}

#[test]
fn gpu_counters_are_window_deltas() {
    let (a, b) = (Scrape::parse(&metrics(1.0)), Scrape::parse(&metrics(2.0)));
    let g = gpu_counters_between(&a, &b);
    assert_eq!(g.len(), 2);
    assert_eq!(g[0].gpu, "2");
    assert_eq!(g[0].seconds, 2.0);
    assert_eq!(g[0].nvlink_rx_gb, 4.0);
    assert_eq!(g[0].nvlink_rx_gb_s, 2.0);
    assert_eq!(g[0].memory_busy_pct, 50.0);
    assert!(gpu_counters_between(&Scrape::default(), &Scrape::default()).is_empty());
    let row = format_phase("decode", &model_bytes_between(&a, &b).unwrap().decode, &g);
    assert!(row.starts_with("decode"), "{row}");
    assert!(row.contains("GPU 2"), "{row}");
}
