//! The PLE ops (`--ple-ops`, #439) against ggml's chain (the default) on a real model
//! (qwen4exp): teacher-forced over a long text in prompt chunks, then steps of one to three
//! tokens, the next-token rows compared bit for bit. The switch acts while graphs are built, so
//! the model loads once per variant. Ignored: needs a GPU and the model.
//!
//! With `GGML_CUDA_DISABLE_FUSION=1` every scored row must be the chain's: the ops give the
//! chain's bits. With ggml-cuda's fusions on the rows may differ, and the test only reports it:
//! the two graphs place tensors apart, and ggml-cuda fuses an MoE layer's weighted expert sum
//! (with FMA adds) only where its output does not overlap the router weights (on the V100,
//! Qwen3.8-Flash-Next, 1,716-token chunks: layer 0's sum is fused beside the chain, not beside
//! the ops).
//!
//! `LEGION_TEST_HC_MODEL=<model.gguf> LEGION_TEST_HC_TEXT=<file> [LEGION_TEST_HC_TOKENS=10292]
//!  [LEGION_TEST_HC_CHUNK=1716] [GGML_CUDA_DISABLE_FUSION=1] cargo test --release --features cuda
//!  --lib ple_ops_kl -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn ple_ops_kl_against_chain() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use super::hc_inject_kl::{HcInputs, hc_inputs};
    use crate::gguf::GgufFile;
    use crate::model::core::{Executor, Group, KV_BLOCK, KvTypes, Logits, Model, SeqInput};
    use crate::model::qwen4;
    use crate::tokenizer::Tokenizer;

    let HcInputs {
        model: model_path,
        text,
        tokens: n_prompt,
        chunk,
    } = hc_inputs(10292, 1716);
    // After the prompt: steps of these sizes, every row scored.
    const TAIL: [usize; 8] = [1, 2, 3, 1, 3, 2, 1, 1];
    const SCORED: usize = 16;
    let n_tokens = n_prompt + TAIL.iter().sum::<usize>();

    let gguf = GgufFile::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(text).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);
    let mut steps: Vec<(usize, usize)> = (0..n_prompt)
        .step_by(chunk)
        .map(|f| (f, (f + chunk).min(n_prompt)))
        .collect();
    let mut at = n_prompt;
    for t in TAIL {
        steps.push((at, at + t));
        at += t;
    }

    let run = |ops: bool| -> Vec<(usize, Vec<f32>)> {
        qwen4::set_fused_ple(ops);
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
        exec.clear_slot(0);
        let mut rows = Vec::new();
        for &(from, to) in &steps {
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
            let k = out.rows.len();
            for (i, r) in out.rows.iter().enumerate().skip(k.saturating_sub(SCORED)) {
                rows.push((to - k + i, r.clone()));
            }
        }
        qwen4::set_fused_ple(false);
        rows
    };
    // Rows bit for bit equal, and the first position that is not (with its largest difference).
    let compare = |label: &str, rows: &[(usize, Vec<f32>)], reference: &[(usize, Vec<f32>)]| {
        assert_eq!(rows.len(), reference.len());
        let mut first = None;
        let mut same = 0;
        for ((pos, a), (_, b)) in rows.iter().zip(reference) {
            if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                same += 1;
            } else if first.is_none() {
                let d = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                first = Some((*pos, d));
            }
        }
        println!(
            "{label}: {same}/{} rows bit-identical (prompt {n_prompt} in chunks of {chunk}, then \
             steps {TAIL:?}); first differing position {first:?}",
            rows.len()
        );
        same
    };
    let chain = run(false);
    let again = run(false);
    compare("chain again (noise floor)", &again, &chain);
    let ops = run(true);
    let same = compare("ple ops vs chain", &ops, &chain);
    if std::env::var_os("GGML_CUDA_DISABLE_FUSION").is_some() {
        assert_eq!(same, ops.len(), "the ops must give the chain's rows");
    }
}
