//! The fused expert op's per-pair outputs in F16 / BF16 (`--moe-pair-out`, opt-in) against
//! F32 on a real model: teacher-forced over a long text, every scored position's next-token
//! distribution (KL against F32, top-1 agreement). The model loads as the engine loads it
//! (layout, converted weights, the n-gram table), once; the format changes between passes.
//! Ignored: needs a GPU and the model.
//!
//! `LEGION_TEST_PAIR_MODEL=<model.gguf> LEGION_TEST_PAIR_TEXT=<file> [LEGION_TEST_PAIR_TOKENS=8192]
//!  [LEGION_TEST_PAIR_CHUNK=2048] cargo test --release --features cuda --lib pair_out_kl -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn pair_outputs_against_f32() {
    use super::super::{DeviceChoice, LmConfig, layout::Layout, open_ngram_table};
    use crate::gguf::GgufFile;
    use crate::model::core::{
        Executor, Group, KV_BLOCK, KvTypes, Logits, Model, MoeKernel, SeqInput,
    };
    use crate::model::moe_ffn::PairOut;
    use crate::tokenizer::Tokenizer;

    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let num = |k: &str, d: usize| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    let n_tokens = num("LEGION_TEST_PAIR_TOKENS", 8192);
    let chunk = num("LEGION_TEST_PAIR_CHUNK", 2048);
    // Scored: the last positions of every chunk (each row is the whole vocabulary).
    const SCORED: usize = 64;
    assert!(
        chunk >= 64 && chunk >= SCORED,
        "chunks of 64+ tokens take the fused op"
    );

    let mut cfg = LmConfig::new(env("LEGION_TEST_PAIR_MODEL"), DeviceChoice::Cuda);
    cfg.ctx_size = n_tokens;
    cfg.kv_cache_tokens = Some(n_tokens);
    let gguf = GgufFile::open(&cfg.model_path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(env("LEGION_TEST_PAIR_TEXT")).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

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

    // log-softmax rows of the scored positions, routed experts through `kernel`, per-pair
    // outputs in `pair`
    let mut run = |kernel: MoeKernel, pair: PairOut| -> Vec<Vec<f32>> {
        exec.set_moe_kernel(kernel);
        exec.set_moe_pair_out(pair);
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
    };
    let reference = run(MoeKernel::Auto, PairOut::F32);
    // F32 again first: the noise floor (0 if the passes are deterministic); last, ggml's
    // mul_mat_id path (MMQ on CUDA) for scale: what the fused op itself moved (#370).
    for (kernel, pair) in [
        (MoeKernel::Auto, PairOut::F32),
        (MoeKernel::Auto, PairOut::F16),
        (MoeKernel::Auto, PairOut::Bf16),
        (MoeKernel::Graph, PairOut::F32),
    ] {
        let rows = run(kernel, pair);
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
        let k = rows.len();
        println!(
            "{kernel:?} pair outputs {pair:?}: {k} positions of {n_tokens} (chunks of {chunk}): KL vs F32 mean \
             {:.3e} max {:.3e}, top-1 {top1}/{k} ({:.2}%)",
            kl_sum / k as f64,
            kl_max,
            100.0 * top1 as f64 / k as f64
        );
    }
}
