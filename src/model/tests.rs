//! The architecture registry: each GGUF architecture id gets its module's hyperparameters.

use super::core::Hparams;
use super::{gemma4, nemotron_h, qwen4, qwen35};
use crate::gguf::GgufFile;

fn hparams(write: impl FnOnce(&std::path::Path)) -> Hparams {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write(&path);
    Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap()
}

#[test]
fn qwen35_parses_to_its_module() {
    let hp = hparams(|p| qwen35::tiny::write(p, &Default::default()).unwrap());
    assert_eq!(hp.arch.name(), "qwen35");
    assert!(qwen4::params(&hp).is_none());
    assert!(hp.is_recurrent(0) && !hp.is_recurrent(3));
    assert_eq!(hp.arch.residual_streams(), 1);
}

#[test]
fn qwen4exp_parses_to_the_qwen4_module() {
    let hp = hparams(|p| qwen4::tiny::write(p, None, 1).unwrap());
    assert_eq!(hp.arch.name(), "qwen4exp");
    let q = qwen4::params(&hp).expect("qwen4exp hyperparameters");
    assert_eq!(hp.arch.residual_streams(), q.hc.n);
    assert!(hp.arch.sigmoid_gate() && hp.arch.output_ready() && hp.arch.mtp());
    assert!(!hp.arch.external_drafters());
    let c = hp
        .arch
        .token_conv(&hp)
        .expect("the n-gram embedding's conv");
    assert_eq!(c.width, q.hc.n * hp.n_embd);
    // The sparse-attention tables, then the MoE row map (`pad_rows`).
    let rows = crate::model::core::pad_rows::on() as usize;
    assert_eq!(hp.arch.group_inputs_len(), 8 + rows);
}

#[test]
fn gemma4_parses_to_its_module() {
    let hp = hparams(|p| gemma4::tiny::write(p, 1).unwrap());
    assert_eq!(hp.arch.name(), "gemma4");
    let g = gemma4::params(&hp).expect("Gemma 4 hyperparameters");
    assert_eq!(hp.arch.swa_window(), g.window);
    assert!(!hp.has_recurrent() && !hp.has_slot_state());
    assert!((0..hp.n_layer).all(|il| hp.is_attention(il)));
}

#[test]
fn nemotron_h_parses_to_its_module() {
    let hp = hparams(|p| nemotron_h::tiny::write(p, 1).unwrap());
    assert_eq!(hp.arch.name(), "nemotron_h");
    let nm = nemotron_h::params(&hp).expect("Nemotron-H hyperparameters");
    assert_eq!(hp.arch.recurrent(), super::core::Recurrent::Mamba2);
    assert!(hp.arch.slot_ids());
    for (il, b) in nm.blocks.iter().enumerate() {
        assert_eq!(hp.is_recurrent(il), *b == nemotron_h::Block::Mamba);
    }
}

#[test]
fn hparams_compare_their_architectures() {
    let a = hparams(|p| qwen35::tiny::write(p, &Default::default()).unwrap());
    let b = hparams(|p| gemma4::tiny::write(p, 1).unwrap());
    assert_eq!(a, a.clone());
    assert_ne!(a.arch, b.arch);
}

#[test]
fn unknown_architectures_are_named_in_the_error() {
    let err = super::parse_hparams(&empty_gguf(), "llama").unwrap_err();
    assert_eq!(
        err.to_string(),
        "architecture `llama` is not qwen35, qwen35moe, qwen4exp, nemotron_h, nemotron_h_moe, gemma4, glm5-next, deepseek41 or muse-glimmer"
    );
}

fn empty_gguf() -> GgufFile {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    crate::gguf::GgufWriter::new().write(&path).unwrap();
    GgufFile::open(&path).unwrap()
}
