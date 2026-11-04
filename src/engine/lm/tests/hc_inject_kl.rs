//! BF16 hyper-connection inject weights loaded as F32 (`--hc-inject-f32`, opt-in) against
//! the weights as stored, on a real model whose GGUF keeps them in BF16 (the IQ2_XS
//! Qwen3.8-Flash-Next): teacher-forced over a long text, every scored position's next-token
//! distribution (KL against the stored weights' path, top-1 agreement). With F32 weights the
//! prompt steps take the fused mixer norm (`hc_post_norm`); as stored, ggml's chain. The
//! conversion happens at load, so the model loads once per variant (one at a time).
//! Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_HC_MODEL=<model.gguf> LEGION_TEST_HC_TEXT=<file> [LEGION_TEST_HC_TOKENS=16384]
//!  [LEGION_TEST_HC_CHUNK=2048] cargo test --release --features cuda --lib hc_inject_kl -- --ignored --nocapture`

/// The inputs of the measurements on `LEGION_TEST_HC_*` (this one, `ple_ops_kl.rs` and
/// `hc_quant.rs`, which read no environment themselves): the model, the text file, and the token and chunk counts, which
/// default to `tokens` and `chunk`.
#[cfg(feature = "cuda")]
pub(super) struct HcInputs {
    pub model: String,
    pub text: String,
    pub tokens: usize,
    pub chunk: usize,
}

#[cfg(feature = "cuda")]
pub(super) fn hc_inputs(tokens: usize, chunk: usize) -> HcInputs {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let num = |k: &str, d: usize| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    HcInputs {
        model: env("LEGION_TEST_HC_MODEL"),
        text: env("LEGION_TEST_HC_TEXT"),
        tokens: num("LEGION_TEST_HC_TOKENS", tokens),
        chunk: num("LEGION_TEST_HC_CHUNK", chunk),
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn hc_inject_f32_against_stored() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use crate::gguf::GgufFile;
    use crate::model::core::{Executor, Group, KV_BLOCK, KvTypes, Logits, Model, SeqInput};
    use crate::tokenizer::Tokenizer;

    let HcInputs {
        model: model_path,
        text,
        tokens: n_tokens,
        chunk,
    } = hc_inputs(16384, 2048);
    // Scored: the last positions of every chunk (each row is the whole vocabulary).
    const SCORED: usize = 64;
    assert!(chunk >= SCORED, "chunks of at least {SCORED} tokens");

    let gguf = GgufFile::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(text).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    // log-softmax rows of the scored positions, the inject weights as F32 or as stored; `passes`
    // runs over the same load (the first is the result, the others the noise floor)
    let run = |inject_f32: bool, passes: usize| -> Vec<Vec<Vec<f32>>> {
        let mut cfg = LmConfig::new(&model_path, DeviceChoice::Cuda);
        cfg.ctx_size = n_tokens;
        cfg.kv_cache_tokens = Some(n_tokens);
        cfg.fp16.hc_inject_f32 = inject_f32;
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
                    for r in &out.rows[out.rows.len().saturating_sub(SCORED)..] {
                        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        let lse = r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
                        rows.push(r.iter().map(|&v| (v as f64 - lse) as f32).collect());
                    }
                }
                rows
            })
            .collect()
    };
    let compare = |label: &str, rows: &[Vec<f32>], reference: &[Vec<f32>]| {
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
        println!(
            "{label}: {k} positions of {n_tokens} (chunks of {chunk}): KL vs stored weights mean \
             {:.3e} max {:.3e}, top-1 {top1}/{k} ({:.2}%)",
            kl_sum / k as f64,
            kl_max,
            100.0 * top1 as f64 / k as f64
        );
    };
    let stored = run(false, 2);
    compare("stored weights again (noise floor)", &stored[1], &stored[0]);
    let f32_inject = run(true, 1);
    compare("inject weights as F32", &f32_inject[0], &stored[0]);
}
