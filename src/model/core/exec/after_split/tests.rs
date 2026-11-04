use super::*;

/// Two stages on two cache devices, the MTP layer (40) on the last one.
fn two_devices(il: usize) -> Option<usize> {
    Some(usize::from(il >= 20))
}

#[test]
fn splits_every_stages_target_layers() {
    assert_eq!(
        stage_split(&[0..20, 20..41], 40, two_devices),
        Some(vec![0..20, 20..40])
    );
    let four = |il: usize| Some(il / 10);
    assert_eq!(
        stage_split(&[0..10, 10..20, 20..30, 30..41], 40, four),
        Some(vec![0..10, 10..20, 20..30, 30..40])
    );
}

#[test]
fn one_stage_does_not_split() {
    let one = vec![Range { start: 0, end: 41 }];
    assert_eq!(stage_split(&one, 40, two_devices), None);
    assert_eq!(stage_split(&[], 40, two_devices), None);
}

#[test]
fn a_device_in_two_stages_does_not_split() {
    assert_eq!(stage_split(&[0..20, 20..41], 40, |_| Some(0)), None);
    let crossing = |il: usize| Some(usize::from(il >= 22));
    assert_eq!(stage_split(&[0..20, 20..41], 40, crossing), None);
    // Four stages, the middle two on one device.
    let shared = |il: usize| Some([0, 1, 1, 2][il / 10]);
    assert_eq!(
        stage_split(&[0..10, 10..20, 20..30, 30..41], 40, shared),
        None
    );
}

#[test]
fn untiered_layers_split_as_long_as_one_is_tiered() {
    let sparse = |il: usize| il.is_multiple_of(2).then_some(usize::from(il >= 20));
    assert_eq!(
        stage_split(&[0..20, 20..41], 40, sparse),
        Some(vec![0..20, 20..40])
    );
    // A stage without tiered layers has nothing to learn, which is fine.
    let last_only = |il: usize| (il >= 20).then_some(1);
    assert_eq!(
        stage_split(&[0..20, 20..41], 40, last_only),
        Some(vec![0..20, 20..40])
    );
    assert_eq!(stage_split(&[0..20, 20..41], 40, |_| None), None);
}

#[test]
fn stages_must_cover_the_layers_back_to_back() {
    assert_eq!(stage_split(&[0..20, 21..41], 40, two_devices), None);
    assert_eq!(stage_split(&[1..20, 20..41], 40, two_devices), None);
    // A last stage of MTP layers only.
    assert_eq!(stage_split(&[0..40, 40..41], 40, two_devices), None);
}

// The executor with the split: the tiny qwen4exp model over two CPU stages whose expert caches
// are separate devices (two host buffer types), as two GPUs are in production.

use crate::ggml::model_bytes::Kind;
use crate::ggml::{Backend, BufferType, ffi};
use crate::gguf::GgufFile;
use crate::model::core::{Executor, Group, Logits, Model, SeqInput};
use crate::residency::experts::{ExpertResidency, LayerPlace, ResidencyStats};
use crate::residency::policy::SwapRule;
use crate::residency::{ExpertMode, ResidencyConfig};

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b.set_use_ref(true);
    b
}

/// The CPU buffer type again, at another address: the cache sees another device, the CPU
/// backend the same host memory.
fn other_cpu_buft() -> BufferType {
    let cpu = Backend::cpu().unwrap().buffer_type();
    // SAFETY: a plain copy of a static C struct (function pointers, device, context); the copy
    // is leaked, so it outlives every buffer allocated from it.
    let copy: &'static mut ffi::ggml_backend_buffer_type =
        Box::leak(Box::new(unsafe { std::ptr::read(cpu.0.as_ptr()) }));
    BufferType(std::ptr::NonNull::from(copy))
}

/// The tiny model over two stages (layers `0..2` and `2..`; with `mtp`, the MTP head on the last
/// stage's cache device), two of eight experts per layer cached, swaps at any gain.
fn two_stage(dir: &std::path::Path, overlap: bool, second: BufferType, mtp: bool) -> Executor {
    let b = cpu();
    let mut m = Model::load(&GgufFile::open(dir.join("tiny.gguf")).unwrap(), |_| {
        b.buffer_type()
    })
    .unwrap();
    if mtp {
        m.load_mtp(&GgufFile::open(dir.join("mtp.gguf")).unwrap(), |_| {
            b.buffer_type()
        })
        .unwrap();
    }
    let n_layer = m.hp.n_layer;
    let mut e =
        Executor::new_staged(m, vec![(cpu(), 0..2), (cpu(), 2..n_layer)], 16, 4, 0).unwrap();
    if mtp {
        e.enable_mtp().unwrap();
    }
    // The last layer's hidden state for every token: otherwise a prompt step's last layer runs
    // only its last token, and the cache would read the rest of that layer's route log stale.
    e.capture_hidden(vec![n_layer - 1]);
    let hp = e.hp().clone();
    let per: u64 = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"]
        .iter()
        .map(|k| {
            let t = e.model().tensor(&format!("blk.0.{k}.weight")).unwrap();
            (t.nbytes() / t.ne()[2] as usize) as u64
        })
        .sum();
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget: 6,
            margin: 0.0,
            ratio: 0.0,
        },
        decay: 0.9,
        ..Default::default()
    };
    let first = b.buffer_type();
    let model = e.model();
    let r = ExpertResidency::new(
        hp.n_layer + usize::from(mtp),
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        |n| model.tensor(n),
        |il| {
            Some(LayerPlace {
                cuda: None,
                buft: if il < 2 { first } else { second },
            })
        },
        |_| 2 * per * (hp.n_layer as u64 + 1),
        &cfg,
    )
    .unwrap();
    e.set_experts(Some(r));
    e.set_after_overlap(overlap);
    e
}

/// Two sequences' prompts, then decode steps (with `mtp`, an MTP pass after every other one):
/// every output, then the cache's counters and tables.
fn run(e: &mut Executor, mtp: bool) -> (Vec<Vec<f32>>, ResidencyStats, Vec<Vec<i32>>) {
    let blocks: [[u32; 4]; 2] = [[0, 1, 2, 3], [4, 5, 6, 7]];
    let prompts: [Vec<u32>; 2] = [
        "split by stage".bytes().map(u32::from).collect(),
        "no GPU waits, 42".bytes().map(u32::from).collect(),
    ];
    let mut outs = Vec::new();
    let mut n_past = [0usize; 2];
    for (i, p) in prompts.iter().enumerate() {
        let g = Group {
            slot: i,
            seqs: vec![SeqInput {
                tokens: p,
                n_past: 0,
                blocks: &blocks[i],
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
            verify: false,
        };
        outs.push(e.step(&[g]).unwrap().rows.concat());
        n_past[i] = p.len();
    }
    let mut last = [prompts[0][0], prompts[1][0]];
    for t in 0..10 {
        let toks = [[last[0]], [last[1]]];
        let g = Group {
            slot: 0,
            seqs: (0..2)
                .map(|i| SeqInput {
                    tokens: &toks[i],
                    n_past: n_past[i],
                    blocks: &blocks[i],
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                })
                .collect(),
            verify: false,
        };
        let out = e.step(&[g]).unwrap();
        outs.push(out.rows.concat());
        last = [out.argmax[0], out.argmax[1]];
        n_past[0] += 1;
        n_past[1] += 1;
        if mtp && t % 2 == 0 {
            let toks2 = [[last[0]], [last[1]]];
            let mg = Group {
                slot: 0,
                seqs: (0..2)
                    .map(|i| SeqInput {
                        tokens: &toks2[i],
                        n_past: n_past[i] - 1,
                        blocks: &blocks[i],
                        ring: 0,
                        logits: Logits::Last,
                        media: None,
                    })
                    .collect(),
                verify: false,
            };
            let m = e.mtp_step(&[mg], &out.hidden[0]).unwrap();
            outs.push(m.pmax.clone());
        }
    }
    // Whatever is still deferred, as the next pass would run it.
    e.flush_deferred().unwrap();
    let r = e.model().experts().unwrap();
    let n_cache = e.hp().n_layer + usize::from(mtp);
    let tables = (0..n_cache).map(|il| r.table(il).unwrap()).collect();
    (outs, r.stats(), tables)
}

#[test]
fn split_cache_work_is_bit_identical_and_learns_the_same() {
    let dir = tempfile::tempdir().unwrap();
    crate::model::qwen4::tiny::write(
        dir.path().join("tiny.gguf"),
        Some(&dir.path().join("mtp.gguf")),
        11,
    )
    .unwrap();
    let second = other_cpu_buft();
    let mut serial = two_stage(dir.path(), false, second, false);
    let mut split = two_stage(dir.path(), true, second, false);
    assert_eq!(
        split.stage_split(),
        Some(vec![0..2, 2..4]),
        "two cache devices, one per stage"
    );
    let a = run(&mut serial, false);
    let b = run(&mut split, false);
    assert_eq!(a.0.len(), b.0.len());
    for (i, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        assert!(
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            "output {i} differs"
        );
    }
    assert_eq!(a.1, b.1, "the cache learned and swapped the same");
    assert!(a.1.swaps > 0, "the cache swapped experts");
    assert!(a.1.logged_steps > 0);
    assert_eq!(a.2, b.2, "the same tables");
}

#[test]
fn a_decode_step_without_mtp_defers_only_the_last_stages_share() {
    let dir = tempfile::tempdir().unwrap();
    crate::model::qwen4::tiny::write(
        dir.path().join("tiny.gguf"),
        Some(&dir.path().join("mtp.gguf")),
        11,
    )
    .unwrap();
    let second = other_cpu_buft();
    for (overlap, mtp) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut e = two_stage(dir.path(), overlap, second, mtp);
        let blocks = [0u32, 1, 2, 3];
        // Longer than a verify step (`DECODE_MAX_TOKENS`).
        let prompt: Vec<u32> = "a prompt step, twenty tokens"
            .bytes()
            .map(u32::from)
            .collect();
        let step = |e: &mut Executor, tokens: &[u32], n_past: usize| {
            let g = Group {
                slot: 0,
                seqs: vec![SeqInput {
                    tokens,
                    n_past,
                    blocks: &blocks,
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                }],
                verify: false,
            };
            e.step(&[g]).unwrap();
        };
        step(&mut e, &prompt, 0);
        // A prompt step keeps its cache work whole (beside the next MTP pass, or now).
        assert_eq!(e.deferred_tail, None);
        step(&mut e, &prompt[..1], prompt.len());
        let deferred = (e.deferred_tail.clone(), e.deferred_target);
        match (overlap, mtp) {
            (true, false) => assert_eq!(deferred, (Some((1, Kind::Decode, 2..4)), None)),
            // With MTP the step's work goes whole beside the next MTP pass, as before.
            (true, true) => assert_eq!(deferred, (None, Some((1, Kind::Decode)))),
            _ => assert_eq!(deferred, (None, None)),
        }
    }
}
