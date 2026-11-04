//! NVFP4 weights in the qwen35 graph on the CPU, with tiny random models: the per-tensor
//! `.scale` tensors from llama.cpp's converter multiply the matmul results exactly as a
//! weight-side scale would.
//!
//! Doubling an NVFP4 block scale (exponent + 1) doubles every weight of the block exactly, and the
//! CPU's NVFP4 x Q8_0 dot product (per-block products summed in F32) then gives exactly twice the
//! result. So a model whose `.scale`s are 2^s must produce the same logits, bit for bit, as one
//! without `.scale` tensors whose block exponents are shifted by s; a model that ignored the
//! scales would differ from it.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::model::qwen35::tiny::{self, TinyConfig, TinyNvfp4};

const PROMPT: &[u32] = &[72, 101, 108, 108, 111, 44, 32, 78, 86, 70, 80, 52, 33];

fn logits(nvfp4: TinyNvfp4) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    let cfg = TinyConfig {
        nvfp4: Some(nvfp4),
        ..Default::default()
    };
    tiny::write(&path, &cfg).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    model.prompt_logits_all(&mut backend, PROMPT).unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn scales_equal_shifted_block_scales() {
    let plain = logits(TinyNvfp4 {
        scale_exp: None,
        block_exp_shift: 0,
    });
    assert!(plain.iter().all(|v| v.is_finite()));
    let scaled = logits(TinyNvfp4 {
        scale_exp: Some(1),
        block_exp_shift: 0,
    });
    let shifted = logits(TinyNvfp4 {
        scale_exp: None,
        block_exp_shift: 1,
    });
    assert_eq!(
        scaled, shifted,
        "scale 2 must equal doubled block scales bit for bit"
    );
    // The scales matter: ignoring them would give `plain`.
    assert!(max_diff(&scaled, &plain) > 1e-3);
}

#[test]
fn unit_scales_change_nothing() {
    let plain = logits(TinyNvfp4 {
        scale_exp: None,
        block_exp_shift: 2,
    });
    let unit = logits(TinyNvfp4 {
        scale_exp: Some(0),
        block_exp_shift: 2,
    });
    assert_eq!(plain, unit);
    // And a scale of 1/4 undoes a block shift of 2.
    let undone = logits(TinyNvfp4 {
        scale_exp: Some(-2),
        block_exp_shift: 2,
    });
    let base = logits(TinyNvfp4 {
        scale_exp: None,
        block_exp_shift: 0,
    });
    assert_eq!(undone, base);
}
