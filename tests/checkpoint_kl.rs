//! Checkpoints of one model against a reference checkpoint (#520), teacher-forced on CUDA: the
//! same token ids (the reference's tokenizer) prefilled in chunks of `LEGION_TEST_KL_CHUNK`
//! (default 2048) tokens, whose last 64 positions are scored. Each checkpoint is loaded, run and
//! dropped in turn, so only one is resident at a time. Prints, against the reference, max |logit
//! diff|, top-1 agreement and KL(reference || checkpoint); with two or more checkpoints, also the
//! second against the first. Asserts nothing: quantized checkpoints differ from the reference.
//! A path may end in `@gather` or `@paged` to pick the attention path for that run (default:
//! `auto`); the same checkpoint may appear more than once.
//!
//! `LEGION_TEST_KL_REF=<dir or gguf>[@path] LEGION_TEST_KL_MODELS=<dir or gguf>[@path][,...]
//!  LEGION_TEST_KL_TEXT=<file> [LEGION_TEST_KL_TOKENS=8192]
//!  cargo test --release --features cuda --test checkpoint_kl -- --ignored --nocapture`

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real checkpoints"]
fn checkpoints_against_reference() {
    use legion::ggml::{DeviceKind, device_by_kind};
    use legion::gguf::GgufFile;
    use legion::model::core::{Attention, Executor, Group, KV_BLOCK, Logits, Model, SeqInput};
    use legion::tokenizer::Tokenizer;

    // `<path>[@gather|@paged]`.
    let spec = |s: &str| -> (String, Attention) {
        match s.rsplit_once('@') {
            Some((p, "gather")) => (p.to_string(), Attention::Gather),
            Some((p, "paged")) => (p.to_string(), Attention::Paged),
            _ => (s.to_string(), Attention::Auto),
        }
    };

    const SCORED: usize = 64;
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let num = |k: &str, d: usize| std::env::var(k).map_or(d, |s| s.parse().unwrap());
    let n_tokens = num("LEGION_TEST_KL_TOKENS", 8192);
    let chunk = num("LEGION_TEST_KL_CHUNK", 2048);
    assert!(chunk > SCORED, "chunks of more than {SCORED} tokens");
    let reference = env("LEGION_TEST_KL_REF");
    let models = env("LEGION_TEST_KL_MODELS");
    let models: Vec<&str> = models.split(',').collect();
    let text = std::fs::read_to_string(env("LEGION_TEST_KL_TEXT")).unwrap();
    let mut ids = Tokenizer::from_gguf(&GgufFile::open(spec(&reference).0).unwrap())
        .unwrap()
        .encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    // The scored rows of one checkpoint.
    let run = |label: &str| -> Vec<Vec<f32>> {
        let (path, attention) = spec(label);
        let gguf = GgufFile::open(&path).unwrap();
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU device")
            .init()
            .unwrap();
        let t = std::time::Instant::now();
        let model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
        let blocks: Vec<u32> = (0..n_tokens.div_ceil(KV_BLOCK) as u32).collect();
        let mut exec = Executor::new(model, gpu, blocks.len(), 1, 0).unwrap();
        exec.set_attention(attention);
        let mut rows = Vec::new();
        for from in (0..n_tokens).step_by(chunk) {
            let to = (from + chunk).min(n_tokens);
            let mid = to.saturating_sub(SCORED).max(from);
            for (lo, hi, logits) in [(from, mid, Logits::None), (mid, to, Logits::All)] {
                if lo == hi {
                    continue;
                }
                let out = exec
                    .step(&[Group {
                        slot: 0,
                        seqs: vec![SeqInput {
                            tokens: &ids[lo..hi],
                            n_past: lo,
                            blocks: &blocks,
                            ring: 0,
                            logits,
                            media: None,
                        }],
                        verify: false,
                    }])
                    .unwrap();
                rows.extend(out.rows);
            }
        }
        println!(
            "{label} (paged: {}): {} rows in {:.1} s (load and run)",
            exec.paged(),
            rows.len(),
            t.elapsed().as_secs_f64()
        );
        rows
    };
    let want = run(&reference);
    let got: Vec<Vec<Vec<f32>>> = models.iter().map(|m| run(m)).collect();
    /// A run's logit rows.
    type Rows<'r> = &'r [Vec<f32>];
    let mut pairs: Vec<(String, Rows<'_>, Rows<'_>)> = models
        .iter()
        .zip(&got)
        .map(|(m, rows)| (format!("{m} vs the reference"), &rows[..], &want[..]))
        .collect();
    if got.len() >= 2 {
        let label = format!("{} vs {}", models[1], models[0]);
        pairs.push((label, &got[1], &got[0]));
    }
    for (label, ours, theirs) in pairs {
        let (diff, top1, n, kl_mean, kl_max) = agreement(ours, theirs);
        println!(
            "{label}, {n_tokens} tokens in chunks of {chunk}: max |diff| {diff:.2e}, \
             top-1 {top1}/{n}, KL mean {kl_mean:.3e} max {kl_max:.3e}"
        );
    }
}

#[cfg(feature = "cuda")]
fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

#[cfg(feature = "cuda")]
fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

/// (max |diff|, top-1 agreements, rows, mean KL, max KL) of `ours` against `reference`; rows of
/// different lengths (padded vocabularies) are compared on the shorter.
#[cfg(feature = "cuda")]
fn agreement(ours: &[Vec<f32>], reference: &[Vec<f32>]) -> (f32, usize, usize, f64, f64) {
    assert_eq!(ours.len(), reference.len());
    let (mut diff, mut top1, mut kl_sum, mut kl_max) = (0f32, 0, 0f64, 0f64);
    for (o, r) in ours.iter().zip(reference) {
        let n = o.len().min(r.len());
        let (o, r) = (&o[..n], &r[..n]);
        for (x, y) in o.iter().zip(r) {
            diff = diff.max((x - y).abs());
        }
        top1 += (argmax(o) == argmax(r)) as usize;
        let (lo, lr) = (log_softmax(o), log_softmax(r));
        let kl: f64 = lr.iter().zip(&lo).map(|(r, o)| r.exp() * (r - o)).sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
    }
    (diff, top1, ours.len(), kl_sum / ours.len() as f64, kl_max)
}
