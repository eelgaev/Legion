use super::*;

fn phase(names: &[&str], gpus: usize) -> PhaseReport {
    PhaseReport {
        wall_s: 2.0,
        failed: 0,
        output_tokens: 256,
        output_tok_s: 128.0,
        ttft_s: Some(0.5),
        model: names
            .iter()
            .map(|n| {
                (
                    n.to_string(),
                    PhaseBytes {
                        gb: 100.0,
                        seconds: 1.0,
                        device_gb: 90.0,
                        device_gb_s: 90.0,
                        roofline_pct: Some(10.0),
                        ..Default::default()
                    },
                )
            })
            .collect(),
        gpus: (0..gpus)
            .map(|g| GpuCounters {
                gpu: g.to_string(),
                ..Default::default()
            })
            .collect(),
    }
}

#[test]
fn table_puts_each_window_counters_under_its_last_row() {
    let r = BandwidthReport {
        prompt_tokens: 8192,
        gen_tokens: 256,
        concurrency: 4,
        roofline_memory_gb_s: 894.0,
        prefill: phase(&["prefill"], 1),
        decode: phase(&["decode", "draft"], 1),
    };
    let t = format_table(&r);
    let lines: Vec<&str> = t.lines().collect();
    assert!(lines[0].contains("device roofline 894 GB/s"), "{t}");
    assert!(lines[1].starts_with("phase"), "{t}");
    assert!(lines[2].starts_with("prefill"), "{t}");
    assert!(lines[3].contains("GPU 0"), "{t}");
    assert!(lines[4].starts_with("decode"), "{t}");
    assert!(lines[5].starts_with("draft"), "{t}");
    assert!(lines[6].contains("GPU 0"), "{t}");
    assert!(lines[2].contains("10%"), "{t}");
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["decode"]["model"][1][0], "draft");
}
