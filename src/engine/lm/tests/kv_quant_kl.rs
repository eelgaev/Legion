//! Quantized KV caches on qwen4exp (#263: q8_0 / q4_0 K and V read by the sparse attention
//! kernels) against the F16 cache on a real model: teacher-forced over a long text, every scored
//! position's next-token distribution (KL against F16, top-1 agreement), with F16 run twice for
//! the noise floor. Chunks of more than 64 tokens take the prompt kernel, of at most 64 the
//! decode kernel. Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_KV_MODEL=<model.gguf> LEGION_TEST_KV_TEXT=<file> [LEGION_TEST_KV_TOKENS=16384]
//!  [LEGION_TEST_KV_CHUNK=2048] [LEGION_TEST_KV_TYPES=q8_0/q8_0,q8_0/q4_0,q4_0/q4_0]
//!  cargo test --release --features cuda --lib kv_quant_kl -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn quantized_caches_against_f16_on_sparse_attention() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use crate::gguf::GgufFile;
    use crate::model::core::{Executor, Group, KV_BLOCK, KvType, KvTypes, Logits, Model, SeqInput};
    use crate::tokenizer::Tokenizer;
    use clap::ValueEnum;

    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let num = |k: &str, d: usize| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    let n_tokens = num("LEGION_TEST_KV_TOKENS", 16384);
    let chunk = num("LEGION_TEST_KV_CHUNK", 2048);
    // Scored: the last positions of every chunk (each row is the whole vocabulary).
    let scored = chunk.min(64);
    let opt = |k: &str| std::env::var(k).ok();
    let pairs: Vec<KvTypes> = opt("LEGION_TEST_KV_TYPES")
        .unwrap_or_else(|| "q8_0/q8_0,q8_0/q4_0,q4_0/q4_0".into())
        .split(',')
        .map(|p| {
            let (k, v) = p.split_once('/').expect("k/v");
            let ty = |s: &str| KvType::from_str(s, true).unwrap();
            KvTypes { k: ty(k), v: ty(v) }
        })
        .collect();

    let model_path = env("LEGION_TEST_KV_MODEL");
    let gguf = GgufFile::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(env("LEGION_TEST_KV_TEXT")).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    // log-softmax rows of the scored positions with K/V in `types`; `passes` runs over the same
    // load (the first is the result, the others the noise floor)
    let run = |types: KvTypes, passes: usize| -> Vec<Vec<Vec<f32>>> {
        let mut cfg = LmConfig::new(&model_path, DeviceChoice::Cuda);
        cfg.ctx_size = n_tokens;
        cfg.kv_cache_tokens = Some(n_tokens);
        cfg.cache_types = types;
        let layout = Layout::plan(&cfg, &gguf).unwrap();
        assert_eq!(layout.stages.len(), 1, "one GPU stage");
        let mut backends = layout.backends().unwrap();
        let place = |info: &crate::gguf::TensorInfo| layout.place(info);
        let opts = crate::load_io::Options::configured();
        let mut model = Model::load_converted(&gguf, place, &opts, cfg.fp16).unwrap();
        open_ngram_table(&mut model, &gguf, &layout, None, &cfg.ngram).unwrap();
        let n = model.hp.n_layer;
        let blocks: Vec<u32> = (0..n_tokens.div_ceil(KV_BLOCK) as u32).collect();
        let mut exec = Executor::new_staged_ring(
            model,
            vec![(backends.remove(0), 0..n)],
            blocks.len(),
            1,
            0,
            None,
            types,
        )
        .unwrap();
        assert!(exec.steps_paged(), "the sparse attention kernels (paged)");
        (0..passes)
            .map(|_| {
                exec.clear_slot(0);
                let mut rows = Vec::new();
                for from in (0..n_tokens).step_by(chunk) {
                    let to = (from + chunk).min(n_tokens);
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
                    for r in &out.rows[out.rows.len().saturating_sub(scored)..] {
                        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        let lse = r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
                        rows.push(r.iter().map(|&v| (v as f64 - lse) as f32).collect());
                    }
                }
                rows
            })
            .collect()
    };
    // (mean KL, top-1 agreement)
    let compare = |label: &str, rows: &[Vec<f32>], reference: &[Vec<f32>]| -> (f64, f64) {
        let (mut kl_sum, mut kl_max, mut top1) = (0f64, 0f64, 0usize);
        for (a, r) in rows.iter().zip(reference) {
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
        let k = rows.len();
        let (mean, agree) = (kl_sum / k as f64, top1 as f64 / k as f64);
        println!(
            "{label}: {k} positions of {n_tokens} (chunks of {chunk}): KL mean {mean:.3e} max \
             {kl_max:.3e}, top-1 {top1}/{k} ({:.2}%)",
            100.0 * agree
        );
        (mean, agree)
    };
    let f16 = run(KvTypes::default(), 2);
    compare("F16 again (noise floor)", &f16[1], &f16[0]);
    for types in pairs {
        let rows = run(types, 1);
        let (kl, _) = compare(
            &format!("K {:?} / V {:?} vs F16", types.k, types.v),
            &rows[0],
            &f16[0],
        );
        assert!(kl < 0.5, "K {:?} / V {:?}: mean KL {kl}", types.k, types.v);
    }
}
