//! The stage trim (#266): each stage's device holds only its own layers' weights, in every
//! mode: the FP16 load mode with and without the gates, the MTP head, host experts and tied
//! embeddings. Converted tensors replace their originals: no second copy anywhere, so the
//! expert cache of a stage sees exactly the converted tensors' own size change (none for BF16,
//! half the F32 gates' bytes with `--weights-fp16-gates`). Strata-AC922 fd96643 had to trim its
//! converted copies separately; here they go through the same per-tensor placement.

use std::ops::Range;
use std::path::Path;

use super::super::layout::Layout;
use crate::ggml::{BufferType, DeviceKind, device_by_kind, types};
use crate::gguf::{GgufFile, GgufWriter, TensorInfo};
use crate::load_io::Options;
use crate::residency::ExpertMode;
use crate::weights::{Fp16Load, Weights, WeightsFp16};

const N_LAYER: usize = 4;
const D: i64 = 8;
// Bytes per tensor as stored (BF16 = 2, F32 = 4) and per layer.
const ATTN_Q: u64 = 8 * 8 * 2;
const ATTN_NORM: u64 = 8 * 4;
const GATE_INP: u64 = 8 * 4 * 4;
const EXPS: u64 = 8 * 8 * 4 * 2;
const LAYER_DENSE: u64 = ATTN_Q + ATTN_NORM + GATE_INP;
const OUTPUT_NORM: u64 = 8 * 4;
const MTP: u64 = 8 * 8 * 2 + 16 * 8 * 2;
const TOKEN_EMBD: u64 = 8 * 16 * 2;
const ROPE: u64 = 4 * 4;

fn bf16(n: usize) -> Vec<u8> {
    // 1.0 in BF16: in F16's range, so the FP16 mode converts it.
    std::iter::repeat_n(0x3F80u16.to_ne_bytes(), n)
        .flatten()
        .collect()
}

fn f32s(n: usize) -> Vec<u8> {
    std::iter::repeat_n(0.5f32.to_ne_bytes(), n)
        .flatten()
        .collect()
}

/// A model with tied embeddings (no `output.weight`): BF16 projections and experts, F32 norms
/// and router gates per layer.
fn write_model(path: &Path) {
    let mut w = GgufWriter::new();
    w.add_tensor("token_embd.weight", types::BF16, &[D, 16], &bf16(128))
        .unwrap();
    w.add_tensor("rope_freqs.weight", types::F32, &[4], &f32s(4))
        .unwrap();
    w.add_tensor("output_norm.weight", types::F32, &[D], &f32s(8))
        .unwrap();
    for il in 0..N_LAYER {
        let t = |n: &str| format!("blk.{il}.{n}.weight");
        w.add_tensor(&t("attn_q"), types::BF16, &[D, D], &bf16(64))
            .unwrap();
        w.add_tensor(&t("attn_norm"), types::F32, &[D], &f32s(8))
            .unwrap();
        w.add_tensor(&t("ffn_gate_inp"), types::F32, &[D, 4], &f32s(32))
            .unwrap();
        w.add_tensor(&t("ffn_up_exps"), types::BF16, &[D, D, 4], &bf16(256))
            .unwrap();
    }
    w.write(path).unwrap();
}

/// An MTP head in a GGUF of its own (`blk.{n_layer}.*`).
fn write_mtp(path: &Path) {
    let mut w = GgufWriter::new();
    let l = N_LAYER;
    w.add_tensor(
        &format!("blk.{l}.attn_q.weight"),
        types::BF16,
        &[D, D],
        &bf16(64),
    )
    .unwrap();
    w.add_tensor(
        &format!("blk.{l}.nextn.eh_proj.weight"),
        types::BF16,
        &[16, D],
        &bf16(128),
    )
    .unwrap();
    w.write(path).unwrap();
}

fn layout(ranges: &[Range<usize>], mode: ExpertMode) -> Layout {
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let host = cpu.buffer_type();
    Layout {
        // A device on every stage (the CPU's), so placement takes the GPU path.
        stages: ranges.iter().map(|r| (Some(cpu), r.clone())).collect(),
        mode,
        arenas: None,
        shared_host: (ranges.len() > 1).then_some(host),
        host,
        tp_devices: Vec::new(),
        head_copy: false,
        node: None,
        glm_par: false,
    }
}

const OFF: Fp16Load = Fp16Load {
    mode: WeightsFp16::Off,
    gates: false,
    hc_inject_f32: false,
};
const LOAD: Fp16Load = Fp16Load {
    mode: WeightsFp16::Load,
    gates: false,
    hc_inject_f32: false,
};
const GATES: Fp16Load = Fp16Load {
    mode: WeightsFp16::Load,
    gates: true,
    hc_inject_f32: false,
};

struct Loaded {
    main: Weights,
    mtp: Weights,
}

fn load(dir: &Path, l: &Layout, fp16: Fp16Load) -> Loaded {
    let place = |info: &TensorInfo| -> BufferType { l.place(info) };
    let main = GgufFile::open(dir.join("m.gguf")).unwrap();
    let main = Weights::load_converted(&main, place, &Options::with_threads(2), fp16).unwrap();
    let mtp = Weights::load(&GgufFile::open(dir.join("mtp.gguf")).unwrap(), place).unwrap();
    Loaded { main, mtp }
}

fn model_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_model(&dir.path().join("m.gguf"));
    write_mtp(&dir.path().join("mtp.gguf"));
    dir
}

/// Every byte allocated is a tensor's own: the converted ones hold no second copy.
fn assert_no_extra_copy(w: &Weights) {
    let tensors: u64 = w.sizes().map(|(_, b)| b).sum();
    let placed: u64 = w.placement().iter().map(|(_, b)| b).sum();
    assert_eq!(tensors, placed);
}

#[test]
fn stages_hold_only_their_own_layers_in_every_fp16_mode() {
    let dir = model_dir();
    for mode in [ExpertMode::Device, ExpertMode::Host] {
        let l = layout(&[0..2, 2..4], mode);
        // Host experts: not in any stage's device memory.
        let layer = LAYER_DENSE + if mode == ExpertMode::Device { EXPS } else { 0 };
        // The head's norm and the MTP head on the last stage; the tied embeddings and the
        // RoPE factors in host memory every stage reads.
        let off = [2 * layer, 2 * layer + OUTPUT_NORM + MTP];
        for (fp16, gates_saved) in [(OFF, 0), (LOAD, 0), (GATES, GATE_INP / 2)] {
            let w = load(dir.path(), &l, fp16);
            assert_no_extra_copy(&w.main);
            assert_no_extra_copy(&w.mtp);
            let got = l.stage_bytes([&w.main, &w.mtp]);
            // Each stage changes by its own layers' converted tensors only.
            let want = off.map(|b| b - 2 * gates_saved);
            assert_eq!(got, want, "{mode:?} {fp16:?}");
            // The converted tensors really are F16 (BF16 kept its size).
            let ty = |n: &str| w.main.get(n).unwrap().ty();
            let bf16_now = if fp16.mode == WeightsFp16::Load {
                types::F16
            } else {
                types::BF16
            };
            assert_eq!(ty("blk.3.attn_q.weight"), bf16_now);
            assert_eq!(ty("token_embd.weight"), bf16_now);
            let gate_now = if fp16.gates { types::F16 } else { types::F32 };
            assert_eq!(ty("blk.0.ffn_gate_inp.weight"), gate_now);
        }
    }
}

#[test]
fn one_stage_holds_everything_device_side_including_tied_embeddings() {
    let dir = model_dir();
    let l = layout(std::slice::from_ref(&(0..N_LAYER)), ExpertMode::Device);
    for (fp16, gates_saved) in [(OFF, 0), (LOAD, 0), (GATES, GATE_INP / 2)] {
        let w = load(dir.path(), &l, fp16);
        let all = N_LAYER as u64 * (LAYER_DENSE + EXPS - gates_saved)
            + OUTPUT_NORM
            + MTP
            + TOKEN_EMBD
            + ROPE;
        assert_eq!(l.stage_bytes([&w.main, &w.mtp]), vec![all], "{fp16:?}");
    }
}

#[test]
fn four_stages_split_the_layers_and_the_head_goes_last() {
    let dir = model_dir();
    let l = layout(&[0..1, 1..2, 2..3, 3..4], ExpertMode::Device);
    for fp16 in [OFF, LOAD] {
        let w = load(dir.path(), &l, fp16);
        let layer = LAYER_DENSE + EXPS;
        assert_eq!(
            l.stage_bytes([&w.main, &w.mtp]),
            vec![layer, layer, layer, layer + OUTPUT_NORM + MTP],
            "{fp16:?}"
        );
    }
}

/// The tied head's device copy ([`super::super::layout::HEAD_COPY`]) over two stages: once, on
/// the last stage, at the embeddings' loaded size and type, in every FP16 load mode.
#[test]
fn a_tied_heads_copy_counts_once_on_the_last_stage() {
    let dir = model_dir();
    let l = layout(&[0..2, 2..4], ExpertMode::Device);
    let main = GgufFile::open(dir.path().join("m.gguf")).unwrap();
    let copy = [super::super::layout::HEAD_COPY];
    let mut opts = Options::with_threads(2);
    opts.aliases = &copy;
    for fp16 in [OFF, LOAD, GATES] {
        let plain = load(dir.path(), &l, fp16);
        let w = Weights::load_converted(&main, |i| l.place(i), &opts, fp16).unwrap();
        assert_no_extra_copy(&w);
        let without = l.stage_bytes([&plain.main, &plain.mtp]);
        let with = l.stage_bytes([&w, &plain.mtp]);
        assert_eq!(with, vec![without[0], without[1] + TOKEN_EMBD], "{fp16:?}");
        let ty = |n: &str| w.get(n).unwrap().ty();
        assert_eq!(ty("output.weight"), ty("token_embd.weight"), "{fp16:?}");
    }
}

/// [`Layout::plan`] makes no copy on the CPU (one memory).
#[test]
fn plan_copies_no_tied_head_on_the_cpu() {
    use crate::engine::lm::{DeviceChoice, LmConfig};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::gemma4::tiny::write(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let cpu = LmConfig {
        layer_split: vec![2],
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    };
    assert!(!Layout::plan(&cpu, &gguf).unwrap().head_copy);
}

/// On GPUs: no copy for one GPU (the embeddings are in its memory), the copy for a layer split
/// over two (an AC922, 2x V100).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs two GPUs"]
fn plan_copies_a_tied_head_to_the_last_of_two_gpus() {
    use crate::engine::lm::{DeviceChoice, LmConfig};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::gemma4::tiny::write(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let one = LmConfig::new(&path, DeviceChoice::Cuda);
    assert!(!Layout::plan(&one, &gguf).unwrap().head_copy, "one GPU");
    let two = LmConfig {
        gpus: 2,
        layer_split: vec![2],
        ..LmConfig::new(&path, DeviceChoice::Cuda)
    };
    assert!(Layout::plan(&two, &gguf).unwrap().head_copy);
}

/// An MTP head's own GGUF goes to the last stage whatever its block numbers: a Gemma 4
/// assistant's blocks 0 and 1 too. By the target's placement they went to the first stage's
/// GPU, where the last stage's kernels read them: an illegal access (Xid 31) on the 31B over
/// two V100s. Its own token embeddings (its tied output head) go there too; its RoPE factors
/// stay in the host memory every GPU reads.
#[test]
fn a_gemma_assistants_blocks_go_to_the_last_stage() {
    let dir = model_dir();
    let path = dir.path().join("assistant.gguf");
    crate::model::gemma4::tiny::write_assistant(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let l = layout(&[0..2, 2..4], ExpertMode::Device);
    let tensors = gguf.tensors();
    assert!(tensors.iter().any(|t| t.name.starts_with("blk.0.")));
    let mut on_device = 0;
    for t in &tensors {
        let stage = l.mtp_device_stage(&t.name);
        match t.name.as_str() {
            "rope_freqs.weight" => assert_eq!(stage, None),
            n => {
                assert_eq!(stage, Some(1), "{n}");
                on_device += t.size;
            }
        }
    }
    let mtp = Weights::load(&gguf, |i| l.place_mtp(i)).unwrap();
    let w = load(dir.path(), &l, OFF);
    let layer = LAYER_DENSE + EXPS;
    assert_eq!(
        l.stage_bytes_with_mtp(&w.main, Some(&mtp)),
        vec![2 * layer, 2 * layer + OUTPUT_NORM + on_device]
    );
    // The target's own placement would put the assistant's blocks on the first stage.
    assert_ne!(l.stage_bytes([&mtp])[0], 0);
}

/// On two GPUs the assistant's blocks are in the last GPU's memory, not the first's (an AC922,
/// 2x V100).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs two GPUs"]
fn a_gemma_assistants_blocks_go_to_the_last_of_two_gpus() {
    use crate::engine::lm::{DeviceChoice, LmConfig};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    let apath = dir.path().join("assistant.gguf");
    crate::model::gemma4::tiny::write(&path, 1).unwrap();
    crate::model::gemma4::tiny::write_assistant(&apath, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let two = LmConfig {
        gpus: 2,
        layer_split: vec![2],
        ..LmConfig::new(&path, DeviceChoice::Cuda)
    };
    let l = Layout::plan(&two, &gguf).unwrap();
    let gpu = |s: usize| l.stages[s].0.unwrap().buffer_type();
    let assistant = GgufFile::open(&apath).unwrap();
    let q = assistant.tensor("blk.0.attn_q.weight").unwrap();
    assert!(l.place_mtp(&q) == gpu(1));
    assert!(l.place_mtp(&q) != gpu(0));
    // Its tied head on the last GPU; the target's embeddings in shared host memory.
    let embd = assistant.tensor("token_embd.weight").unwrap();
    assert!(l.place_mtp(&embd) == gpu(1));
    assert!(l.place(&embd) != gpu(1));
    assert!(
        l.place(&q) == gpu(0),
        "the target's placement: the first GPU"
    );
}
