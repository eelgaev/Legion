//! glm5-next (GLM-5.3-Flash) from the shipped ModelOpt NVFP4 safetensors against Hugging Face
//! transformers' own `Glm5NextTextModel` (F32 on the CPU, NVFP4 dequantized exactly, W4A16), on
//! a reduced-depth copy of the checkpoint (`tools/glm5next-ref/truncate.py`; references from
//! `tools/glm5next-ref/reference.py`). Ignored: needs the copy and the references.
//!
//! `$R` holds `ids<i>.txt` (the prompt's token ids) and `ref<i>/` (reference.py's output, every
//! prompt position's logits: `--rows` at least the prompt length) for `i = 0, 1, ...`.
//!
//! ```sh
//! LEGION_TEST_GLM5_HF=$COPY LEGION_TEST_GLM5_HF_REF=$R \
//!   cargo test --release [--features cuda] --test glm5next_hf -- --ignored --nocapture --test-threads 1
//! ```
//!
//! Per prompt and backend: top-1 agreement over the prompt's positions and the largest logit
//! difference. The CPU must agree at every position; CUDA runs with the GPU's kernels (the same
//! W4A16 arithmetic, other summation orders).

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn f32s(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The prompts and their reference logits (`[positions][vocab]`), in order.
fn cases() -> Vec<(Vec<u32>, Vec<f32>)> {
    let dir = env("LEGION_TEST_GLM5_HF_REF");
    let mut out = Vec::new();
    for i in 0.. {
        let Ok(ids) = std::fs::read_to_string(format!("{dir}/ids{i}.txt")) else {
            break;
        };
        let ids: Vec<u32> = ids.split_whitespace().map(|t| t.parse().unwrap()).collect();
        out.push((ids, f32s(&format!("{dir}/ref{i}/logits.f32"))));
    }
    assert!(!out.is_empty(), "no ids0.txt in {dir}");
    out
}

/// Compares every prompt's logits on `backend`; returns (top-1 matches, positions, max |d|).
fn compare(label: &str, backend: &mut Backend, model: &Model) -> (usize, usize, f32) {
    let n_vocab = model.hp.n_vocab as usize;
    let (mut hits, mut n, mut worst) = (0, 0, 0f32);
    for (i, (ids, want)) in cases().iter().enumerate() {
        let got = model.prompt_logits_all(backend, ids).unwrap();
        let rows = want.len() / n_vocab;
        assert!(rows <= ids.len(), "reference has more rows than the prompt");
        let skip = ids.len() - rows;
        let (mut h, mut d) = (0, 0f32);
        for r in 0..rows {
            let ours = &got[(skip + r) * n_vocab..(skip + r + 1) * n_vocab];
            let theirs = &want[r * n_vocab..(r + 1) * n_vocab];
            h += (argmax(ours) == argmax(theirs)) as usize;
            d = ours
                .iter()
                .zip(theirs)
                .fold(d, |m, (a, b)| m.max((a - b).abs()));
        }
        let last = &want[(rows - 1) * n_vocab..];
        let scale = last.iter().fold(0f32, |m, x| m.max(x.abs()));
        println!(
            "{label} prompt {i}: {} tokens, top-1 {h}/{rows}, max |logit d| {d:.4} (logits up to {scale:.2})",
            ids.len()
        );
        hits += h;
        n += rows;
        worst = worst.max(d);
    }
    println!("{label}: top-1 {hits}/{n}, max |logit d| {worst:.4}");
    (hits, n, worst)
}

#[test]
#[ignore = "needs a reduced-depth GLM-5.3-Flash NVFP4 copy and transformers references"]
fn cpu_matches_transformers() {
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5_HF")).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(
        std::env::var("LEGION_TEST_THREADS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(16),
    );
    let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let (hits, n, _) = compare("cpu", &mut cpu, &model);
    assert_eq!(hits, n);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU, a reduced-depth GLM-5.3-Flash NVFP4 copy and transformers references"]
fn cuda_matches_transformers() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5_HF")).unwrap();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    let model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let (hits, n, _) = compare("cuda", &mut gpu, &model);
    // Other summation orders may flip a near tie; most positions must agree.
    assert!(hits * 10 >= n * 9, "top-1 {hits}/{n}");
}

/// Every prompt's logits with the model's routed experts read by the SSD expert tier with
/// `slots` slots per layer (on `buft`), against the same model with its experts loaded.
fn tier_matches_loaded(
    backend: &mut Backend,
    model: &mut Model,
    gguf: &GgufFile,
    buft: legion::ggml::BufferType,
    slots: usize,
) {
    use legion::residency::ssd_experts::{ExpertSsd, SsdArgs, SsdExperts, tier_layers};
    let hp = &model.hp;
    let (n_expert, n_used) = (hp.n_expert as usize, hp.n_expert_used as usize);
    let layers = tier_layers(gguf, hp.n_layer).expect("NVFP4 experts with sources");
    // Each prompt, and its last token alone (a step small enough for the slots).
    let prompts: Vec<Vec<u32>> = cases()
        .into_iter()
        .flat_map(|(ids, _)| [ids.clone(), ids[ids.len() - 1..].to_vec()])
        .collect();
    let want: Vec<Vec<f32>> = prompts
        .iter()
        .map(|ids| model.prompt_logits_all(backend, ids).unwrap())
        .collect();
    // One expert across the three matrices (all the same size).
    let up = gguf
        .tensor(&format!("blk.{}.ffn_up_exps.weight", layers[0]))
        .unwrap();
    let expert = 3 * (up.shape[0] * up.shape[1]) as usize / 64 * 36;
    let bytes = (2 * n_expert + layers.len() * slots) * expert;
    let args = SsdArgs {
        mode: ExpertSsd::On,
        ..SsdArgs::default()
    };
    let tier = SsdExperts::new(
        gguf,
        &layers,
        hp.n_layer,
        n_expert,
        n_used,
        bytes as u64,
        buft,
        512,
        &args,
    )
    .unwrap();
    model.set_ssd_experts(tier);
    // Twice: the first round fills the cache (prompts stage, single tokens miss), the second
    // runs over a filled one (evictions, hits).
    for round in 0..2 {
        for (i, ids) in prompts.iter().enumerate() {
            let got = model.prompt_logits_all(backend, ids).unwrap();
            assert!(
                got == want[i],
                "slots {slots} round {round} prompt {i} ({} tokens): logits differ",
                ids.len()
            );
        }
    }
    let s = model.ssd_experts().unwrap().stats();
    println!(
        "tier slots {slots}: hits {} misses {} hit rate {:.2}",
        s.hits,
        s.misses,
        s.hit_rate()
    );
}

#[test]
#[ignore = "needs a reduced-depth GLM-5.3-Flash NVFP4 copy"]
fn cpu_ssd_tier_matches_loaded_experts() {
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5_HF")).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(
        std::env::var("LEGION_TEST_THREADS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(16),
    );
    let mut model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let buft = cpu.buffer_type();
    let n_used = model.hp.n_expert_used as usize;
    // As few slots as a token routes to (evictions every step), then room for most experts.
    for slots in [n_used, 200] {
        tier_matches_loaded(&mut cpu, &mut model, &gguf, buft, slots);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an integrated GPU (the tier's memory is the host's) and a reduced-depth copy"]
fn cuda_ssd_tier_matches_loaded_experts() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5_HF")).unwrap();
    let dev = device_by_kind(DeviceKind::IntegratedGpu).expect("no integrated GPU");
    let buft = dev.host_buffer_type().expect("pinned host memory");
    let mut gpu = dev.init().unwrap();
    let mut model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let n_used = model.hp.n_expert_used as usize;
    for slots in [n_used, 200] {
        tier_matches_loaded(&mut gpu, &mut model, &gguf, buft, slots);
    }
}
