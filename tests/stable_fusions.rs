//! `--stable-fusions` (#472) on the CPU, through the real graphs of the tiny MoE models
//! (qwen4exp, Gemma 4, Nemotron-H): the dependency nodes it adds only move where ggml's
//! allocator frees tensors, so every logit keeps its bits. One test, as the switch is
//! process-wide.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Model, set_stable_fusions};

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

/// The logits of every position of a short prompt, with the switch off and on.
fn logits_off_on(path: &std::path::Path) -> (Vec<u32>, Vec<u32>) {
    let gguf = GgufFile::open(path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let tokens: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 200).collect();
    let mut run = |on: bool| {
        set_stable_fusions(on);
        let logits = model.prompt_logits_all(&mut backend, &tokens).unwrap();
        set_stable_fusions(false);
        logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    };
    let off = run(false);
    let on = run(true);
    assert!(!off.is_empty());
    (off, on)
}

#[test]
fn pinned_moe_graphs_keep_every_logit() {
    let dir = tempfile::tempdir().unwrap();
    let qwen4 = dir.path().join("qwen4.gguf");
    legion::model::qwen4::tiny::write(&qwen4, None, 1).unwrap();
    let gemma4 = dir.path().join("gemma4.gguf");
    legion::model::gemma4::tiny::write(&gemma4, 1).unwrap();
    let nemotron = dir.path().join("nemotron.gguf");
    legion::model::nemotron_h::tiny::write(&nemotron, 1).unwrap();
    for path in [&qwen4, &gemma4, &nemotron] {
        let (off, on) = logits_off_on(path);
        assert!(off == on, "{}: the switch moved the logits", path.display());
    }
}
