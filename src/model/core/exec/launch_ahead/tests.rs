use super::*;

#[test]
fn queued_steps_launch_every_next_stage_ahead() {
    // Four stages: stages 1-3 are launched while the one before computes.
    for s in 0..3 {
        assert!(wanted(true, true, s, 4, false));
    }
}

#[test]
fn the_last_stage_has_nothing_to_launch_ahead() {
    assert!(!wanted(true, true, 3, 4, false));
    assert!(!wanted(true, true, 0, 1, false));
}

#[test]
fn the_switch_turns_it_off() {
    assert!(!wanted(false, true, 0, 4, false));
}

#[test]
fn steps_with_synchronous_fills_launch_after_the_wait() {
    // Prompt, pipelined, pair-parallel and split-model steps (`step_fill::queued`).
    assert!(!wanted(true, false, 0, 4, false));
}

#[test]
fn the_peer_fetch_keeps_the_launch_after_the_wait() {
    // The idle peer's help belongs to a stage that runs alone.
    assert!(!wanted(true, true, 0, 2, true));
}

// The executor with it: the tiny qwen4exp model over three CPU stages, with and without the
// launch ahead, gives the same bits.

use crate::ggml::Backend;
use crate::gguf::GgufFile;
use crate::model::core::{Executor, Group, Logits, Model, SeqInput};

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b.set_use_ref(true);
    b
}

/// The tiny model (with `mtp`, its MTP head on the last stage) over three stages.
fn three_stage(dir: &std::path::Path, ahead: bool, mtp: bool) -> Executor {
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
    assert!(n_layer >= 3, "a stage per layer range");
    let stages = vec![
        (cpu(), 0..1),
        (cpu(), 1..n_layer - 1),
        (cpu(), n_layer - 1..n_layer),
    ];
    let mut e = Executor::new_staged(m, stages, 16, 4, 0).unwrap();
    if mtp {
        e.enable_mtp().unwrap();
    }
    e.capture_hidden(vec![n_layer - 1]);
    e.set_launch_ahead(ahead);
    e
}

/// Two sequences' prompts, then decode steps of both (with `mtp`, an MTP pass after each):
/// every output's bits.
fn run(e: &mut Executor, mtp: bool) -> Vec<Vec<u32>> {
    let blocks: [[u32; 4]; 2] = [[0, 1, 2, 3], [4, 5, 6, 7]];
    let prompts: [Vec<u32>; 2] = [
        "launch ahead".bytes().map(u32::from).collect(),
        "the next stage, 7".bytes().map(u32::from).collect(),
    ];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
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
        outs.push(bits(&e.step(&[g]).unwrap().rows.concat()));
        n_past[i] = p.len();
    }
    let mut last = [prompts[0][0], prompts[1][0]];
    for _ in 0..6 {
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
        outs.push(bits(&out.rows.concat()));
        outs.push(out.argmax.clone());
        last = [out.argmax[0], out.argmax[1]];
        n_past[0] += 1;
        n_past[1] += 1;
        if mtp {
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
            outs.push(bits(&m.pmax));
        }
    }
    outs
}

#[test]
fn launching_ahead_gives_the_same_bits() {
    let dir = tempfile::tempdir().unwrap();
    crate::model::qwen4::tiny::write(
        dir.path().join("tiny.gguf"),
        Some(&dir.path().join("mtp.gguf")),
        13,
    )
    .unwrap();
    for mtp in [false, true] {
        let mut after = three_stage(dir.path(), false, mtp);
        let mut ahead = three_stage(dir.path(), true, mtp);
        assert!(!after.launches_ahead(true, 0));
        assert!(ahead.launches_ahead(true, 0) && ahead.launches_ahead(true, 1));
        let a = run(&mut after, mtp);
        let b = run(&mut ahead, mtp);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x, y, "output {i} differs (mtp {mtp})");
        }
    }
}
