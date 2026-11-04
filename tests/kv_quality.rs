//! Quantized KV caches' quality on a real model (#295): teacher-forced over a long text, every
//! scored position's next-token distribution with K/V in each type pair against F16 (KL, top-1).
//! Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_KV_MODEL=<model.gguf> LEGION_TEST_KV_TEXT=<file> [LEGION_TEST_KV_TOKENS=8192]
//!  cargo test --release --features cuda --test kv_quality -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn quantized_caches_against_f16() {
    use legion::ggml::{DeviceKind, device_by_kind};
    use legion::gguf::GgufFile;
    use legion::model::core::{
        Executor, Group, KV_BLOCK, KvType, KvTypes, Logits, Model, Placement, SeqInput,
    };
    use legion::tokenizer::Tokenizer;

    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let gguf = GgufFile::open(env("LEGION_TEST_KV_MODEL")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let n_tokens: usize = std::env::var("LEGION_TEST_KV_TOKENS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(8192);
    let text = std::fs::read_to_string(env("LEGION_TEST_KV_TEXT")).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);
    const CHUNK: usize = 512;
    // Scored: the last 64 positions of every chunk.
    const SCORED: usize = 64;

    // log-softmax rows of the scored positions, with K/V in `types`.
    let run = |types: KvTypes| -> Vec<Vec<f32>> {
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU device")
            .init()
            .unwrap();
        let cpu = legion::ggml::Backend::cpu().unwrap();
        let model = Model::load(&gguf, |i| {
            if Model::placement(&i.name, false) == Placement::Host {
                cpu.buffer_type()
            } else {
                gpu.buffer_type()
            }
        })
        .unwrap();
        let n = model.hp.n_layer;
        let blocks: Vec<u32> = (0..n_tokens.div_ceil(KV_BLOCK) as u32).collect();
        let mut exec =
            Executor::new_staged_ring(model, vec![(gpu, 0..n)], blocks.len(), 1, 0, None, types)
                .unwrap();
        let mut rows = Vec::new();
        for from in (0..n_tokens).step_by(CHUNK) {
            let to = (from + CHUNK).min(n_tokens);
            let out = exec
                .step(&[Group {
                    slot: 0,
                    seqs: vec![SeqInput {
                        tokens: &ids[from..to],
                        n_past: from,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::All,
                        media: None,
                    }],
                    verify: false,
                }])
                .unwrap();
            for r in &out.rows[out.rows.len().saturating_sub(SCORED)..] {
                let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
                rows.push(r.iter().map(|&v| (v as f64 - lse) as f32).collect());
            }
        }
        rows
    };
    let reference = run(KvTypes::default());
    use KvType::*;
    for (k, v) in [
        (Q8_0, Q8_0),
        (Q8_0, Q4_0),
        (Q4_0, Q4_0),
        (Q5_1, Q5_1),
        (Iq4Nl, Iq4Nl),
    ] {
        let rows = run(KvTypes { k, v });
        let (mut kl_sum, mut kl_max, mut top1) = (0f64, 0f64, 0usize);
        for (a, r) in rows.iter().zip(&reference) {
            let kl: f64 = r
                .iter()
                .zip(a)
                .map(|(&lr, &la)| (lr as f64).exp() * (lr as f64 - la as f64))
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let am = |x: &Vec<f32>| (0..x.len()).max_by(|&i, &j| x[i].total_cmp(&x[j])).unwrap();
            top1 += usize::from(am(a) == am(r));
        }
        let n = rows.len();
        println!(
            "K {k:?} / V {v:?}: {n} positions of {n_tokens}: KL mean {:.2e} max {:.2e}, top-1 \
             {top1}/{n} ({:.1}%)",
            kl_sum / n as f64,
            kl_max,
            100.0 * top1 as f64 / n as f64
        );
        assert!(kl_sum / (n as f64) < 0.5, "K {k:?} / V {v:?}: mean KL");
    }
}
