//! qwen4exp's combine-and-norm writing its down matmul's input quantized for ggml's MMQ (#363)
//! against ggml quantizing it (`--hc-quant-ggml`) on a real model, on a GPU the op is chosen for
//! (compute capability 12.x): every block the op writes is ggml's `quantize_mmq_q8_1` output of
//! the same activations, byte for byte (the op's test check), and every scored position's
//! logits row is the same, bit for bit. The switch acts while graphs are built, so the model
//! loads once per variant (one at a time). Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_HC_MODEL=<model.gguf> LEGION_TEST_HC_TEXT=<file> [LEGION_TEST_HC_TOKENS=12288]
//!  [LEGION_TEST_HC_CHUNK=4096] cargo test --release --features cuda --lib hc_quant -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn hc_quant_against_ggml_quantization() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use super::hc_inject_kl::{HcInputs, hc_inputs};
    use crate::gguf::GgufFile;
    use crate::model::core::{
        Executor, Group, KV_BLOCK, KvTypes, Logits, Model, SeqInput, mmq_preq,
    };
    use crate::tokenizer::Tokenizer;

    let HcInputs {
        model: model_path,
        text,
        tokens: n_tokens,
        chunk,
    } = hc_inputs(12288, 4096);
    // Scored: the last positions of every chunk (each row is the whole vocabulary).
    const SCORED: usize = 64;
    assert!(chunk >= SCORED, "chunks of at least {SCORED} tokens");
    let cc_major = crate::residency::cuda::caps(0).map(|c| c.cc_major);
    assert_eq!(
        cc_major,
        Some(12),
        "the op is chosen on compute capability 12.x only"
    );

    let gguf = GgufFile::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(text).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    // The scored positions' logits rows, with the op (and its check) or ggml's quantization.
    let run = |on: bool| -> Vec<Vec<f32>> {
        mmq_preq::set_gpu(cc_major);
        mmq_preq::set_mmq_preq(on);
        mmq_preq::set_check(on);
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
            rows.extend(
                out.rows[out.rows.len().saturating_sub(SCORED)..]
                    .iter()
                    .cloned(),
            );
        }
        mmq_preq::set_check(false);
        mmq_preq::set_mmq_preq(true);
        rows
    };
    let ours = run(true);
    let (launches, bytes, diff) = mmq_preq::checked();
    let ggml = run(false);
    println!(
        "hc_quant: {launches} matmuls checked, {bytes} bytes of quantized activations, \
         {diff} different from ggml's"
    );
    assert!(launches > 0, "the op never ran");
    assert_eq!(diff, 0, "the op's quantization differs from ggml's");
    let same = ours
        .iter()
        .zip(&ggml)
        .filter(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        })
        .count();
    println!(
        "hc_quant: {same}/{} scored rows bit-identical ({n_tokens} tokens, chunks of {chunk})",
        ours.len()
    );
    assert_eq!(same, ours.len(), "logits differ from ggml's quantization");
}
