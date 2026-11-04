//! The separate hyper-connection norm rounded as the fused one (`--hc-norm-fma`, opt-in, #409)
//! against the default on a real model (qwen4exp, sm_7x: elsewhere the two are the same
//! kernel): teacher-forced over a long text, every scored position's next-token distribution
//! (KL against the default, top-1 agreement, rows bit for bit). Also ggml's combine with the
//! separate norm (`--hc-post-norm-ggml`) under each: with the FMA form it must give the fused
//! op's rows bit for bit (#376). The switches act while graphs are built and kernels launched,
//! so the model loads once per variant (one at a time). Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_HC_MODEL=<model.gguf> LEGION_TEST_HC_TEXT=<file> [LEGION_TEST_HC_TOKENS=16384]
//!  [LEGION_TEST_HC_CHUNK=2048] cargo test --release --features cuda --lib hc_norm_fma_kl -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn hc_norm_fma_against_default() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use crate::gguf::GgufFile;
    use crate::model::core::{Executor, Group, KV_BLOCK, KvTypes, Logits, Model, SeqInput};
    use crate::model::qwen4;
    use crate::tokenizer::Tokenizer;

    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let num = |k: &str, d: usize| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    let n_tokens = num("LEGION_TEST_HC_TOKENS", 16384);
    let chunk = num("LEGION_TEST_HC_CHUNK", 2048);
    // Scored: the last positions of every chunk (each row is the whole vocabulary).
    const SCORED: usize = 64;
    assert!(chunk >= SCORED, "chunks of at least {SCORED} tokens");

    let model_path = env("LEGION_TEST_HC_MODEL");
    let gguf = GgufFile::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(env("LEGION_TEST_HC_TEXT")).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    // log-softmax rows of the scored positions with the norm's FMA form or not and the fused
    // combine-and-norm or ggml's chain; `passes` runs over the same load (the first is the
    // result, the others the noise floor)
    let run = |fma: bool, fused: bool, passes: usize| -> Vec<Vec<Vec<f32>>> {
        qwen4::set_norm_fma(fma);
        qwen4::set_fused_post_norm(fused);
        let mut cfg = LmConfig::new(&model_path, DeviceChoice::Cuda);
        cfg.ctx_size = n_tokens;
        cfg.kv_cache_tokens = Some(n_tokens);
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
            KvTypes::default(),
        )
        .unwrap();
        let rows = (0..passes)
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
                    for r in &out.rows[out.rows.len().saturating_sub(SCORED)..] {
                        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        let lse = r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
                        rows.push(r.iter().map(|&v| (v as f64 - lse) as f32).collect());
                    }
                }
                rows
            })
            .collect();
        qwen4::set_norm_fma(false);
        qwen4::set_fused_post_norm(true);
        rows
    };
    // Rows bit for bit equal: the number of positions whose whole row is.
    let compare = |label: &str, rows: &[Vec<f32>], reference: &[Vec<f32>]| -> usize {
        let (mut kl_sum, mut kl_max, mut top1, mut same) = (0f64, 0f64, 0usize, 0usize);
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
            same += usize::from(a.iter().zip(r).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        let k = rows.len();
        println!(
            "{label}: {k} positions of {n_tokens} (chunks of {chunk}): KL mean {:.3e} max {:.3e}, \
             top-1 {top1}/{k} ({:.2}%), rows bit-identical {same}/{k}",
            kl_sum / k as f64,
            kl_max,
            100.0 * top1 as f64 / k as f64
        );
        same
    };
    let default = run(false, true, 2);
    compare("default again (noise floor)", &default[1], &default[0]);
    let fma = run(true, true, 1);
    compare("--hc-norm-fma vs default", &fma[0], &default[0]);
    let chain = run(false, false, 1);
    compare("--hc-post-norm-ggml vs default", &chain[0], &default[0]);
    let fma_chain = run(true, false, 1);
    let same = compare(
        "--hc-norm-fma --hc-post-norm-ggml vs --hc-norm-fma",
        &fma_chain[0],
        &fma[0],
    );
    assert_eq!(
        same,
        fma[0].len(),
        "with the FMA form, ggml's combine and the separate norm must give the fused op's rows"
    );
}
