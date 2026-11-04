//! `--gemma4-compact-global-kv` (#551): the tiny model with the 31B's partial RoPE, its global
//! layer without `attn_v` ([`tiny::K_EQ_V_LAYER`]) on a compact cache, against the plain cache.

use super::*;
use crate::ggml::Backend;
use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};

#[test]
fn rotated_dims_follow_the_factors() {
    let f = [1.0, 1e30, 0.5, 1e30];
    let r = rotated_dims(8, 8, Some(&f));
    assert_eq!(
        r,
        [true, false, true, false, true, false, true, false],
        "pairs (i, i + 4)"
    );
    // Past n_rot nothing turns; without factors everything up to it does.
    assert_eq!(
        rotated_dims(8, 4, None),
        [true, true, true, true, false, false, false, false]
    );
    assert_eq!(
        dim_runs(&rotated_dims(8, 8, Some(&[1.0, 1.0, 1e30, 1e30]))),
        [(0, 2, true), (2, 2, false), (4, 2, true), (6, 2, false)]
    );
}

#[test]
fn compact_layers_are_the_global_ones_without_attn_v() {
    let dir = tempfile::tempdir().unwrap();
    let (partial, plain) = (dir.path().join("p.gguf"), dir.path().join("f.gguf"));
    tiny::write_partial_rope(&partial, 3).unwrap();
    tiny::write_without_ple(&plain, 3).unwrap();
    let b = Backend::cpu().unwrap();
    let load = |p: &std::path::Path| {
        let gguf = crate::gguf::GgufFile::open(p).unwrap();
        Model::load(&gguf, |_| b.buffer_type()).unwrap()
    };
    let mut m = load(&partial);
    let full = KvPool::block_bytes(&m.hp);
    assert_eq!(m.set_gemma4_compact_global_kv(), [tiny::K_EQ_V_LAYER]);
    let r = 2 * tiny::ROTATED_PAIRS as i64;
    assert_eq!(m.hp.compact_kv(tiny::K_EQ_V_LAYER), r);
    let (d, n_kv) = m.hp.kv_dims(tiny::K_EQ_V_LAYER);
    // K and V of d each become one granule of d + r, per head and cell.
    let saved = (2 * d - (d + r)) * n_kv * 2 * crate::model::core::KV_BLOCK as i64;
    assert_eq!(full - KvPool::block_bytes(&m.hp), saved as usize);
    // Every pair rotates: nothing to leave out.
    assert!(load(&plain).set_gemma4_compact_global_kv().is_empty());
}

/// Greedy tokens and the last rows' logits of two sequences (prompts, decode, both batched),
/// through `attention`, compact or not.
fn run(
    gguf: &crate::gguf::GgufFile,
    compact: bool,
    attention: Attention,
    cells: bool,
) -> (Vec<u32>, Vec<f32>) {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    let mut m = Model::load(gguf, |_| b.buffer_type()).unwrap();
    if compact {
        assert_eq!(m.set_gemma4_compact_global_kv(), [tiny::K_EQ_V_LAYER]);
    }
    m.set_cell_gather(cells);
    let mut exec = Executor::new(m, b, 16, 2, 0).unwrap();
    exec.set_attention(attention);
    let blocks: Vec<Vec<u32>> = vec![(0..8).collect(), (8..16).collect()];
    let prompts: Vec<Vec<u32>> = vec![
        (0..150).map(|i| (i * 7 + 3) % 50).collect(),
        (0..70).map(|i| (i * 5 + 1) % 50).collect(),
    ];
    let mut n_past = [0usize; 2];
    let (mut toks, mut rows) = (Vec::new(), Vec::new());
    let mut step = |exec: &mut Executor, seqs: &[(usize, Vec<u32>)]| -> Vec<u32> {
        let groups: Vec<Group> = seqs
            .iter()
            .map(|(s, t)| Group {
                slot: *s,
                verify: false,
                seqs: vec![SeqInput {
                    tokens: t,
                    n_past: n_past[*s],
                    blocks: &blocks[*s],
                    logits: Logits::Last,
                    ring: *s,
                    media: None,
                }],
            })
            .collect();
        let r = exec.step(&groups).unwrap();
        for (s, t) in seqs {
            n_past[*s] += t.len();
        }
        rows.extend(r.rows.concat());
        toks.extend(&r.argmax);
        r.argmax
    };
    let mut last = [0u32; 2];
    last[0] = step(&mut exec, &[(0, prompts[0].clone())])[0];
    last[1] = step(&mut exec, &[(1, prompts[1].clone())])[0];
    for _ in 0..6 {
        let r = step(&mut exec, &[(0, vec![last[0]]), (1, vec![last[1]])]);
        last = [r[0], r[1]];
    }
    (toks, rows)
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // The values are stored in F16 and the key weights move to the queries: close, not equal.
    assert!(
        diff <= 1e-2 * scale.max(1.0),
        "{what}: max |diff| {diff} of {scale}"
    );
}

#[test]
fn compact_cache_matches_the_plain_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_partial_rope(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    for (attention, cells, what) in [
        (Attention::Gather, false, "block gather"),
        (Attention::Gather, true, "cell gather"),
        (Attention::Paged, false, "paged"),
    ] {
        // The cell gather is ggml's CPU flash attention (F16), outside this tolerance on
        // aarch64 (#906).
        if cells && cfg!(target_arch = "aarch64") {
            continue;
        }
        let (toks, rows) = run(&gguf, false, attention, cells);
        let (t, r) = run(&gguf, true, attention, cells);
        close(&rows, &r, what);
        assert_eq!(toks, t, "{what}");
    }
}

/// Tensor parallelism over two CPU "devices": the compact cache split by the Megatron rule
/// (the tiny global layer's one kv head keeps it whole on each), against one CPU.
#[test]
fn compact_cache_on_a_meta_device() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_partial_rope(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let forward = |mut b: Backend| {
        b.set_threads(2);
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        assert_eq!(m.set_gemma4_compact_global_kv(), [tiny::K_EQ_V_LAYER]);
        m.set_cell_gather(true);
        let mut st = m.new_state(b.buffer_type(), 256).unwrap();
        let mut alloc = None;
        let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 50).collect();
        let mut out = m
            .forward(&mut b, &mut alloc, &mut st, &prompt, false)
            .unwrap();
        for t in 0..10u32 {
            out.extend(
                m.forward(&mut b, &mut alloc, &mut st, &[t % 50], false)
                    .unwrap(),
            );
        }
        out
    };
    let one = forward(Backend::cpu().unwrap());
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    close(&one, &forward(meta.init().unwrap()), "meta");
}

/// Tensor parallelism's runtime: the executor over a meta device of two CPU "devices", the
/// global layer's two kv heads split one per device, F16 and Q8_0 granules, against one CPU.
#[test]
fn compact_cache_split_by_heads() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::core::{KvType, KvTypes};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_partial_rope_heads(&path, 3, 2).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let run = |mut b: Backend, types: KvTypes| -> (Vec<u32>, Vec<f32>) {
        b.set_threads(2);
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        assert_eq!(m.set_gemma4_compact_global_kv(), [tiny::K_EQ_V_LAYER]);
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 16, 2, 0, None, types).unwrap();
        exec.set_attention(Attention::Gather);
        let blocks: Vec<u32> = (0..16).collect();
        let mut toks: Vec<u32> = (0..90).map(|i| (i * 7 + 3) % 50).collect();
        let (mut n_past, mut out, mut rows) = (0, Vec::new(), Vec::new());
        for _ in 0..6 {
            let r = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: &toks,
                        n_past,
                        blocks: &blocks,
                        logits: Logits::Last,
                        ring: 0,
                        media: None,
                    }],
                }])
                .unwrap();
            n_past += toks.len();
            rows.extend(r.rows.concat());
            out.push(r.argmax[0]);
            toks = vec![r.argmax[0]];
        }
        (out, rows)
    };
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    for v in [KvType::F16, KvType::Q8_0] {
        let types = KvTypes { k: KvType::F16, v };
        let (t1, r1) = run(Backend::cpu().unwrap(), types);
        let meta = crate::ggml::meta::meta_device(
            &[cpu, cpu],
            crate::model::core::tp::megatron_rule(&hp, 2),
        )
        .unwrap();
        let (t2, r2) = run(meta.init().unwrap(), types);
        close(&r1, &r2, &format!("{v:?}"));
        assert_eq!(t1, t2, "{v:?}");
    }
}

/// The MTP assistant reads the target's compact cache: the same target tokens and drafts as
/// over the plain cache.
#[test]
fn assistant_reads_the_compact_cache() {
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write_partial_rope(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let run = |compact: bool, attention: Attention| -> Vec<u32> {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let mut m = Model::load(&gguf, |_| b.buffer_type()).unwrap();
        if compact {
            m.set_gemma4_compact_global_kv();
        }
        m.load_mtp(&agguf, |_| b.buffer_type()).unwrap();
        let n_layer = m.hp.n_layer;
        let n_embd = m.hp.n_embd as usize;
        let mut exec = Executor::new(m, b, 16, 2, 0).unwrap();
        exec.set_attention(attention);
        exec.enable_mtp().unwrap();
        exec.capture_hidden(vec![n_layer - 1]);
        let blocks: Vec<u32> = (0..16).collect();
        fn seq<'a>(tokens: &'a [u32], n_past: usize, blocks: &'a [u32]) -> Group<'a> {
            Group {
                slot: 0,
                verify: false,
                seqs: vec![SeqInput {
                    tokens,
                    n_past,
                    blocks,
                    logits: Logits::Last,
                    ring: 0,
                    media: None,
                }],
            }
        }
        let mut out = Vec::new();
        let mut toks: Vec<u32> = (0..90).map(|i| (i * 7 + 3) % 50).collect();
        let mut n_past = 0;
        for _ in 0..4 {
            let r = exec.step(&[seq(&toks, n_past, &blocks)]).unwrap();
            let (next, h) = (
                r.argmax[0],
                r.hidden[0][(toks.len() - 1) * n_embd..].to_vec(),
            );
            n_past += toks.len();
            out.push(next);
            let fed = [next];
            out.push(
                exec.mtp_step(&[seq(&fed, n_past, &blocks)], &h)
                    .unwrap()
                    .argmax[0],
            );
            toks = vec![next];
        }
        out
    };
    for attention in [Attention::Gather, Attention::Paged] {
        assert_eq!(run(false, attention), run(true, attention), "{attention:?}");
    }
}

/// Tensor parallelism with sequence slots (the owner's `--tp 2 --max-seqs 4` serving): each
/// sequence's span read in place through views of the granules split by heads, slots 0 and 1
/// (the V view's offset within the granules, as each device's slice must see it), F16 and
/// Q8_0, against one CPU; and both sequences decoding in one batch, through slots or gathered
/// cells (the outputs joined before they go back in head order). A V alias placed in the
/// meta buffer (13a969b) had no device address: CUDA zeroed its row padding at a wild one
/// (Xid 13); the joined outputs' reorder read the wrong dims on the second device.
#[test]
fn compact_cache_slots_split_by_heads() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::core::{KvType, KvTypes};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_partial_rope_heads(&path, 3, 2).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let run = |mut b: Backend, types: KvTypes, slots: bool| -> (Vec<u32>, Vec<f32>) {
        b.set_threads(2);
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        assert_eq!(m.set_gemma4_compact_global_kv(), [tiny::K_EQ_V_LAYER]);
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 16, 2, 0, None, types).unwrap();
        exec.set_attention(Attention::Gather);
        if slots {
            exec.set_kv_slot_blocks(8);
            exec.resize_kv(16).unwrap();
        }
        let blocks: Vec<Vec<u32>> = vec![(0..8).collect(), (8..16).collect()];
        let prompts: Vec<Vec<u32>> = vec![
            (0..150).map(|i| (i * 7 + 3) % 50).collect(),
            (0..70).map(|i| (i * 5 + 1) % 50).collect(),
        ];
        let mut n_past = [0usize; 2];
        let (mut toks, mut rows) = (Vec::new(), Vec::new());
        let mut step = |exec: &mut Executor, seqs: &[(usize, Vec<u32>)]| -> Vec<u32> {
            let groups: Vec<Group> = seqs
                .iter()
                .map(|(s, t)| Group {
                    slot: *s,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: t,
                        n_past: n_past[*s],
                        blocks: &blocks[*s],
                        logits: Logits::Last,
                        ring: *s,
                        media: None,
                    }],
                })
                .collect();
            let r = exec.step(&groups).unwrap();
            for (s, t) in seqs {
                n_past[*s] += t.len();
            }
            rows.extend(r.rows.concat());
            toks.extend(&r.argmax);
            r.argmax
        };
        let mut last = [0u32; 2];
        last[0] = step(&mut exec, &[(0, prompts[0].clone())])[0];
        last[1] = step(&mut exec, &[(1, prompts[1].clone())])[0];
        for _ in 0..6 {
            let r = step(&mut exec, &[(0, vec![last[0]]), (1, vec![last[1]])]);
            last = [r[0], r[1]];
        }
        (toks, rows)
    };
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta = || {
        let dev = crate::ggml::meta::meta_device(
            &[cpu, cpu],
            crate::model::core::tp::megatron_rule(&hp, 2),
        )
        .unwrap();
        dev.init().unwrap()
    };
    for v in [KvType::F16, KvType::Q8_0] {
        let types = KvTypes { k: KvType::F16, v };
        let (t1, r1) = run(Backend::cpu().unwrap(), types, false);
        let (t2, r2) = run(Backend::cpu().unwrap(), types, true);
        if v == KvType::F16 {
            // Q8_0: the gather dequantizes to F16, flash attention reads Q8_0 (as without
            // the compact cache): further apart than `close`.
            close(&r1, &r2, &format!("{v:?} one CPU, slots"));
            assert_eq!(t1, t2, "{v:?} one CPU, slots");
        }
        let (t3, r3) = run(meta(), types, false);
        close(&r1, &r3, &format!("{v:?} meta, cells"));
        assert_eq!(t1, t3, "{v:?} meta, cells");
        let (t4, r4) = run(meta(), types, true);
        close(&r2, &r4, &format!("{v:?} meta, slots"));
        assert_eq!(t2, t4, "{v:?} meta, slots");
    }
}
