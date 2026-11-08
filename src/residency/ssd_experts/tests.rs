//! The SSD expert tier: slot choice, reads and repacks, slot and staging modes.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::cache::{Plan, SlotCache};
use super::io::{ALIGN, aligned_span};
use super::*;
use crate::ggml::{DeviceKind, device_by_kind};
use crate::gguf::{FileRange, GgufWriter, Nvfp4Source, TensorInfo};
use crate::hf::nvfp4::repack_nvfp4;

#[test]
fn slots_go_to_empty_ones_first_then_least_recently_used() {
    let mut c = SlotCache::new(8, 3);
    assert_eq!(
        c.plan(&[4, 1]),
        Plan {
            hits: 0,
            loads: vec![(4, 0), (1, 1)]
        }
    );
    assert_eq!(c.plan(&[1]).loads, vec![]);
    // Expert 4 (slot 0) is the least recently used; slot 2 is still empty.
    assert_eq!(c.plan(&[7, 6]).loads, vec![(7, 2), (6, 0)]);
    assert_eq!(c.slot(4), None);
    assert_eq!(c.slot(6), Some(0));
    // Expert 1 is the least recently used but the step needs it; 6 and 7 tie, lowest slot goes.
    let p = c.plan(&[1, 3]);
    assert_eq!((p.hits, p.loads), (1, vec![(3, 0)]));
    assert_eq!(c.filled(), 3);
}

#[test]
fn a_step_never_evicts_its_own_experts() {
    let mut c = SlotCache::new(16, 4);
    for step in 0..50usize {
        let want: Vec<usize> = (0..4).map(|i| (step * 3 + i * 5) % 16).collect();
        c.plan(&want);
        for &e in &want {
            assert!(c.slot(e).is_some(), "step {step}: expert {e}");
        }
    }
}

#[test]
fn empty_slots_take_experts_without_evicting() {
    let mut c = SlotCache::new(8, 2);
    c.plan(&[3]);
    assert_eq!(c.take_empty(3), None);
    assert_eq!(c.take_empty(5), Some(1));
    assert_eq!(c.take_empty(6), None);
    assert_eq!((c.slot(3), c.slot(5)), (Some(0), Some(1)));
}

#[test]
fn aligned_spans_cover_the_range() {
    let r = FileRange {
        file: 0,
        offset: 4095,
        len: 2,
    };
    assert_eq!(aligned_span(r), (0, 2 * ALIGN as usize));
    let r = FileRange {
        file: 0,
        offset: 8192,
        len: 4096,
    };
    assert_eq!(aligned_span(r), (8192, 4096));
}

#[test]
fn big_reads_split_into_whole_row_pieces() {
    use super::io::{Job, PIECE, split};
    let (rows, k) = (48, 65536);
    let r = |offset, len| FileRange {
        file: 2,
        offset,
        len,
    };
    let job = Job::Read {
        src: Nvfp4Source {
            weight: r(100, (rows * k / 2) as u64),
            scale: r(7, (rows * k / 16) as u64),
            scale2: 0.5,
        },
        rows,
        k,
        dst: 1 << 20,
    };
    let per = PIECE / (k / 2);
    let pieces = split(job);
    assert_eq!(pieces.len(), rows.div_ceil(per));
    let mut next = (0, 100, 7, 1 << 20);
    for p in &pieces {
        let Job::Read {
            src, rows: n, dst, ..
        } = *p
        else {
            panic!("a read")
        };
        assert_eq!(src.weight, r(next.1, (n * k / 2) as u64));
        assert_eq!(src.scale, r(next.2, (n * k / 16) as u64));
        assert_eq!((src.scale2, dst), (0.5, next.3));
        next = (
            next.0 + n,
            next.1 + src.weight.len,
            next.2 + src.scale.len,
            next.3 + n * k / 64 * 36,
        );
    }
    assert_eq!(next.0, rows);
    let copy = Job::Copy {
        src: 1,
        dst: 2,
        len: 3,
    };
    assert_eq!(split(copy).len(), 1);
}

const ROWS: usize = 4;
const K: usize = 128;
const N_EXPERT: usize = 6;
/// Tier layers (by layer index) of the test model.
const LAYERS: [usize; 2] = [1, 3];

/// Bytes of expert `e` of `kind` in `layer`: (codes, scales).
fn expert_data(layer: usize, kind: usize, e: usize) -> (Vec<u8>, Vec<u8>) {
    let mut x = (layer * 131 + kind * 17 + e * 7 + 1) as u32;
    let mut next = || {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (x >> 16) as u8
    };
    let w = (0..ROWS * K / 2).map(|_| next()).collect();
    let s = (0..ROWS * K / 16).map(|_| next()).collect();
    (w, s)
}

fn scale2(layer: usize, kind: usize, e: usize) -> f32 {
    1.0 + (layer * 100 + kind * 10 + e) as f32 / 1024.0
}

/// A checkpoint file of every expert (codes and scales at odd offsets) and a synthetic model
/// over it.
fn model(dir: &Path) -> GgufFile {
    let path = dir.join("experts.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    let mut at = 0u64;
    let mut put = |f: &mut std::fs::File, bytes: &[u8]| {
        f.write_all(&[0xee; 13]).unwrap();
        f.write_all(bytes).unwrap();
        let r = FileRange {
            file: 0,
            offset: at + 13,
            len: bytes.len() as u64,
        };
        at += 13 + bytes.len() as u64;
        r
    };
    let mut stacks = HashMap::new();
    let mut tensors = Vec::new();
    for &il in &LAYERS {
        for (k, kind) in KINDS.iter().enumerate() {
            let experts = (0..N_EXPERT)
                .map(|e| {
                    let (w, s) = expert_data(il, k, e);
                    Nvfp4Source {
                        weight: put(&mut f, &w),
                        scale: put(&mut f, &s),
                        scale2: scale2(il, k, e),
                    }
                })
                .collect();
            let name = format!("blk.{il}.{kind}.weight");
            stacks.insert(
                name.clone(),
                Arc::new(Nvfp4Stack {
                    rows: ROWS,
                    k: K,
                    experts,
                }),
            );
            tensors.push(TensorInfo {
                name,
                ggml_type: types::NVFP4,
                shape: vec![K as u64, ROWS as u64, N_EXPERT as u64],
                file_index: 0,
                offset: 0,
                size: (N_EXPERT * ROWS * K / 64 * 36) as u64,
            });
            // Layer 3 has per-expert scales, layer 1 none (all 1 in a real checkpoint).
            if il == 3 {
                tensors.push(TensorInfo {
                    name: format!("blk.{il}.{kind}.scale"),
                    ggml_type: types::F32,
                    shape: vec![N_EXPERT as u64],
                    file_index: 0,
                    offset: 0,
                    size: 4 * N_EXPERT as u64,
                });
            }
        }
    }
    f.sync_all().unwrap();
    let mut g = GgufFile::synthetic(
        GgufWriter::new(),
        dir.to_path_buf(),
        vec![path],
        tensors,
        HashMap::new(),
    )
    .unwrap();
    g.set_nvfp4_stacks(stacks);
    g
}

fn scratch() -> tempfile::TempDir {
    // Under the build directory: tmpfs refuses O_DIRECT (that falls back, still exact).
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

fn tier(gguf: &GgufFile, slots: usize, direct: bool) -> SsdExperts {
    let expert = 3 * ROWS * K / 64 * 36;
    let bytes = (2 * N_EXPERT + LAYERS.len() * slots) * expert;
    let buft = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
    let args = SsdArgs {
        mode: ExpertSsd::On,
        readers: 3,
        direct,
        ..SsdArgs::default()
    };
    SsdExperts::new(gguf, &LAYERS, 4, N_EXPERT, 2, bytes as u64, buft, 64, &args).unwrap()
}

/// The expert matrix the tensor named `name` holds at index `at`, and its scale there.
fn held(t: &SsdExperts, name: &str, at: usize) -> (Vec<u8>, f32) {
    let w = t.ctx.tensor(name).unwrap().bytes();
    let mb = ROWS * K / 64 * 36;
    let s = t.ctx.tensor(&format!("{name}.scale")).unwrap().to_f32();
    (w[at * mb..(at + 1) * mb].to_vec(), s[at])
}

fn want(layer: usize, kind: usize, e: usize) -> (Vec<u8>, f32) {
    let (w, s) = expert_data(layer, kind, e);
    (repack_nvfp4(&w, &s, ROWS, K), scale2(layer, kind, e))
}

#[test]
fn the_tier_finds_its_layers_and_skips_their_tensors() {
    let dir = scratch();
    let g = model(dir.path());
    assert_eq!(tier_layers(&g, 4), Some(LAYERS.to_vec()));
    let skip = skipped_tensors(&LAYERS);
    assert!(skip.contains(&"blk.3.ffn_down_exps.scale".to_string()));
    assert_eq!(skip.len(), 12);
}

#[test]
fn small_steps_read_their_experts_into_slots() {
    for direct in [true, false] {
        let dir = scratch();
        let g = model(dir.path());
        let t = tier(&g, 3, direct);
        assert_eq!(t.n_slots(), 3);
        // One token of two experts (pairs <= slots): slot mode.
        let out = t.fetch_host(3, &[5, 2], 1).unwrap();
        assert_eq!(out, vec![0, 1]);
        for (k, kind) in KINDS.iter().enumerate() {
            for (slot, e) in [(0, 5), (1, 2)] {
                assert_eq!(held(&t, &format!("ssd.3.{kind}"), slot), want(3, k, e));
            }
        }
        // Expert 2 stays; 0 takes the empty slot.
        assert_eq!(t.fetch_host(3, &[2, 0], 1).unwrap(), vec![1, 2]);
        // 5 is the least recently used.
        assert_eq!(t.fetch_host(3, &[4, 0], 1).unwrap(), vec![0, 2]);
        assert_eq!(held(&t, "ssd.3.ffn_gate_exps", 0), want(3, 1, 4));
        let s = t.stats();
        assert_eq!((s.hits, s.misses), (2, 4));
        // Layer 1's slots are its own.
        assert_eq!(t.fetch_host(1, &[5, 1], 1).unwrap(), vec![0, 1]);
        assert_eq!(held(&t, "ssd.1.ffn_up_exps", 1), want(1, 0, 1));
    }
}

#[test]
fn big_steps_stage_every_routed_expert_and_read_the_next_layer_ahead() {
    let dir = scratch();
    let g = model(dir.path());
    let t = tier(&g, 3, true);
    // Cache experts 1 and 4 of layer 1 first.
    t.fetch_host(1, &[1, 4], 1).unwrap();
    // Layer 1 is ordinal 0: staging buffer 0. Cached ones copied, the rest read.
    let ids = [0, 1, 4, 5, 1, 0];
    assert_eq!(t.fetch_host(1, &ids, 3).unwrap(), ids.to_vec());
    for (k, kind) in KINDS.iter().enumerate() {
        for e in [0, 1, 4, 5] {
            assert_eq!(held(&t, &format!("ssd.stage0.{kind}"), e), want(1, k, e));
        }
    }
    // The empty slot took expert 0 (the first one read); the cached ones stayed.
    assert_eq!(held(&t, "ssd.1.ffn_down_exps", 2), want(1, 2, 0));
    assert_eq!(t.fetch_host(1, &[0, 4], 1).unwrap(), vec![2, 1]);
    // Again a big step on layer 1, then layer 3 (buffer 1), read ahead meanwhile.
    t.fetch_host(1, &ids, 3).unwrap();
    let ids3 = [3, 2, 5, 0, 1, 2];
    assert_eq!(t.fetch_host(3, &ids3, 3).unwrap(), ids3.to_vec());
    for (k, kind) in KINDS.iter().enumerate() {
        for e in [0, 1, 2, 3, 5] {
            assert_eq!(held(&t, &format!("ssd.stage1.{kind}"), e), want(3, k, e));
        }
    }
    assert!(t.stats().stage_bytes > 0);
}

#[test]
fn a_tier_too_small_for_a_token_is_refused() {
    let dir = scratch();
    let g = model(dir.path());
    let expert = 3 * ROWS * K / 64 * 36;
    let buft = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
    let bytes = (2 * N_EXPERT + LAYERS.len()) * expert;
    let r = SsdExperts::new(
        &g,
        &LAYERS,
        4,
        N_EXPERT,
        2,
        bytes as u64,
        buft,
        64,
        &SsdArgs::default(),
    );
    assert!(r.is_err());
}

fn trace_of(n_expert: usize, steps: &[(bool, Vec<u16>)]) -> replay_trace::Trace {
    replay_trace::Trace {
        header: replay_trace::Header {
            layers: vec![3],
            n_expert,
            n_slots: 0,
        },
        records: steps
            .iter()
            .map(|(big, ids)| replay_trace::Record {
                ord: 0,
                big: *big,
                n_tokens: 1,
                ids: ids.clone(),
            })
            .collect(),
        predictions: Vec::new(),
    }
}

use super::trace as replay_trace;

#[test]
fn traces_round_trip_and_a_cut_record_is_dropped() {
    let dir = scratch();
    let path = dir.path().join("t.bin");
    let header = replay_trace::Header {
        layers: vec![1, 3],
        n_expert: 300,
        n_slots: 110,
    };
    {
        let mut w = replay_trace::Writer::create(&path, &header).unwrap();
        w.record(1, false, 1, &[299, 7]);
        w.record(0, true, 2, &[4, 5, 4, 6]);
    }
    let t = replay_trace::Trace::read(&path).unwrap();
    assert_eq!(t.header, header);
    assert_eq!(
        t.records,
        vec![
            replay_trace::Record {
                ord: 1,
                big: false,
                n_tokens: 1,
                ids: vec![299, 7]
            },
            replay_trace::Record {
                ord: 0,
                big: true,
                n_tokens: 2,
                ids: vec![4, 5, 4, 6]
            },
        ]
    );
    let bytes = std::fs::read(&path).unwrap();
    let cut = replay_trace::Trace::parse(&bytes[..bytes.len() - 1]).unwrap();
    assert_eq!(cut.records.len(), 1);
    assert!(replay_trace::Trace::parse(b"LSXX0000000000000000").is_err());
}

/// A hot set of 3 experts routed every step beside a scan of cold ones, 4 slots: LRU loses the
/// hot ones to the scan, heat and the pinned split keep them, Belady is the ceiling.
#[test]
fn replays_rank_the_policies() {
    use replay::{Policy, Setup, replay};
    let mut steps = vec![(true, vec![0, 1, 2, 3, 4, 5])];
    for i in 0..200u16 {
        let cold = 3 + (i % 20);
        let ids = match i % 3 {
            0 => vec![0, cold],
            1 => vec![1, cold],
            _ => vec![2, cold],
        };
        steps.push((false, ids));
    }
    let t = trace_of(24, &steps);
    let run = |policy| {
        replay(
            &t,
            &Setup {
                policy,
                n_slots: 4,
                seed: None,
            },
        )
    };
    let lru = run(Policy::Lru);
    let heat = run(Policy::Heat {
        decay: 0.95,
        prompts: false,
    });
    let pinned = run(Policy::Pinned {
        frac: 0.5,
        decay: 0.95,
    });
    let best = run(Policy::Belady);
    assert_eq!(lru.tokens, 200);
    assert_eq!(lru.hits + lru.misses, 400);
    assert!(heat.hit_rate() > lru.hit_rate() + 0.2, "{heat:?} {lru:?}");
    assert!(
        pinned.hit_rate() > lru.hit_rate() + 0.2,
        "{pinned:?} {lru:?}"
    );
    assert!(best.hits >= heat.hits.max(pinned.hits));
    // The first big step filled the 4 empty slots with 0..=3: no first-step misses for them.
    let mut c = SlotCache::new(24, 4);
    for e in 0..4 {
        c.take_empty(e);
    }
    assert_eq!(c.plan(&[0, 3]).hits, 2);
}

#[test]
fn a_profile_from_a_trace_seeds_a_replay() {
    use replay::{Policy, Setup, profile, replay, seed_heat};
    let train = trace_of(
        8,
        &[(false, vec![6, 7]), (false, vec![7, 6]), (true, vec![6, 5])],
    );
    let p = profile(&train);
    assert_eq!(p.n_layers, 4);
    assert_eq!(&p.ranked[..2], &[(3, 6), (3, 7)]);
    let seed = seed_heat(&p, &[3], 8, 16.0);
    assert!(seed[0][6] > seed[0][7] && seed[0][7] > seed[0][5] && seed[0][0] == 0.0);
    let eval = trace_of(8, &[(false, vec![6, 7])]);
    let o = replay(
        &eval,
        &Setup {
            policy: Policy::Lru,
            n_slots: 2,
            seed: Some(seed),
        },
    );
    assert_eq!((o.hits, o.misses), (2, 0));
}

/// Heat keeps an expert routed every step through a scan that LRU would evict it for, and
/// counts the big steps' pairs only when asked.
#[test]
fn a_heat_replay_keeps_the_hot_expert() {
    use replay::{Policy, Setup, replay};
    let run = |t: &replay_trace::Trace, policy| {
        replay(
            t,
            &Setup {
                policy,
                n_slots: 3,
                seed: None,
            },
        )
    };
    let heat = |prompts| Policy::Heat {
        decay: 0.9,
        prompts,
    };
    // 0 every fourth step, three cold experts in between: LRU drops 0 each time, heat keeps
    // it once its heat has built up (from the third round).
    let mut steps = Vec::new();
    for r in 0..12u16 {
        let base = 1 + 3 * (r % 2);
        for e in base..base + 3 {
            steps.push((false, vec![e]));
        }
        steps.push((false, vec![0]));
    }
    let t = trace_of(8, &steps);
    let lru = run(&t, Policy::Lru);
    let hot = run(&t, heat(false));
    assert_eq!(lru.hits, 0);
    assert_eq!(hot.hits, 10);
    // A big step routing 7 many times: with prompts its heat outlasts 5's single use.
    let t = trace_of(
        8,
        &[
            (true, vec![7; 20]),
            (false, vec![5]),
            (false, vec![6]),
            (false, vec![4]),
            (false, vec![7]),
        ],
    );
    assert_eq!(run(&t, heat(true)).hits, 1);
    assert_eq!(run(&t, heat(false)).hits, 0);
}

/// The tier's trace hook records every fetch: layer ordinal, mode, tokens and routed ids.
#[test]
fn the_tier_traces_its_fetches() {
    let dir = scratch();
    let g = model(dir.path());
    let path = dir.path().join("trace.bin");
    let header = replay_trace::Header {
        layers: LAYERS.to_vec(),
        n_expert: N_EXPERT,
        n_slots: 3,
    };
    {
        let t = tier(&g, 3, true);
        t.inner.state.lock().unwrap().trace =
            Some(replay_trace::Writer::create(&path, &header).unwrap());
        t.fetch_host(3, &[5, 2], 1).unwrap();
        t.fetch_host(1, &[0, 1, 2, 3], 2).unwrap();
    }
    let tr = replay_trace::Trace::read(&path).unwrap();
    assert_eq!(tr.header, header);
    let got: Vec<(usize, bool, usize, Vec<u16>)> = tr
        .records
        .into_iter()
        .map(|r| (r.ord, r.big, r.n_tokens, r.ids))
        .collect();
    assert_eq!(
        got,
        vec![(1, false, 1, vec![5, 2]), (0, true, 2, vec![0, 1, 2, 3])]
    );
}

use super::trace::{Prediction, Predictor};

/// Version 2 traces carry predictions (each placed before the next fetch); version 1 traces,
/// without any, still read.
#[test]
fn traces_carry_predictions_and_version_1_still_reads() {
    let dir = scratch();
    let path = dir.path().join("t2.bin");
    let header = replay_trace::Header {
        layers: vec![1, 3],
        n_expert: 300,
        n_slots: 110,
    };
    {
        let mut w = replay_trace::Writer::create(&path, &header).unwrap();
        w.record(0, false, 1, &[1, 2]);
        w.predict(1, Predictor::PostAttn, 2, &[7, 8, 9, 7, 3, 4]);
        w.predict(1, Predictor::Mtp, 1, &[5, 6]);
        w.record(1, false, 2, &[7, 3, 8, 4]);
    }
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
    let t = replay_trace::Trace::parse(&bytes).unwrap();
    assert_eq!(t.records.len(), 2);
    assert_eq!(
        t.predictions[0],
        Prediction {
            ord: 1,
            predictor: Predictor::PostAttn,
            n_tokens: 2,
            ids: vec![7, 8, 9, 7, 3, 4],
            before: 1,
        }
    );
    assert_eq!(t.predictions[1].predictor, Predictor::Mtp);
    // Rank by rank: both tokens' best, then their second (7 twice counts once).
    assert_eq!(t.predictions[0].width(), 3);
    assert_eq!(t.predictions[0].top(2), vec![7, 8, 3]);
    assert_eq!(t.predictions[0].top(9), vec![7, 8, 3, 9, 4]);
    // The same fetches as a version 1 file.
    let mut v1 = bytes[..20 + 8].to_vec();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    for (ord, ids) in [(0u16, [1u16, 2]), (1, [7, 3])] {
        v1.extend_from_slice(&[0, 0]);
        v1.extend_from_slice(&ord.to_le_bytes());
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&2u32.to_le_bytes());
        for e in ids {
            v1.extend_from_slice(&e.to_le_bytes());
        }
    }
    let t1 = replay_trace::Trace::parse(&v1).unwrap();
    assert_eq!(t1.records.len(), 2);
    assert!(t1.predictions.is_empty());
    v1[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert!(replay_trace::Trace::parse(&v1).is_err());
}

#[test]
fn a_prefetch_keeps_what_it_names_and_respects_its_budget() {
    let mut c = SlotCache::new(10, 3);
    c.plan(&[0]);
    c.plan(&[1]);
    c.plan(&[2]);
    // 1 named (kept, now recent); 5 and 6 take the least recently used other slots: 0, then 2.
    assert_eq!(c.prefetch(&[1, 5, 6], usize::MAX), vec![(5, 0), (6, 2)]);
    assert_eq!((c.slot(1), c.slot(0), c.slot(2)), (Some(1), None, None));
    // Budget 1: only the best missing one; never more than the slots it does not name.
    assert_eq!(c.prefetch(&[7, 8], 1), vec![(7, 0)]);
    assert_eq!(c.prefetch(&[3, 4, 7, 9], usize::MAX).len(), 2);
    c.drop_slot(0);
    assert_eq!(c.filled(), 2);
    assert_eq!(c.plan(&[9]).loads, vec![(9, 0)]);
}

fn tier_with(gguf: &GgufFile, slots: usize, prefetch: SsdPrefetch, margin: usize) -> SsdExperts {
    let args = SsdArgs {
        prefetch,
        prefetch_margin: margin,
        ..SsdArgs::default()
    };
    tier_args(gguf, slots, args)
}

/// A tier of `slots` slots per layer and 3 readers with `args`' prefetch settings.
fn tier_args(gguf: &GgufFile, slots: usize, args: SsdArgs) -> SsdExperts {
    let expert = 3 * ROWS * K / 64 * 36;
    let bytes = (2 * N_EXPERT + LAYERS.len() * slots) * expert;
    let buft = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
    let args = SsdArgs {
        mode: ExpertSsd::On,
        readers: 3,
        ..args
    };
    SsdExperts::new(gguf, &LAYERS, 4, N_EXPERT, 2, bytes as u64, buft, 64, &args).unwrap()
}

/// `--ssd-prefetch-ranks`: only each token's best ranks are read ahead; the step reads the
/// rest itself, and the output is the same.
#[test]
fn the_ranks_cap_reads_only_the_surest_experts_ahead() {
    let dir = scratch();
    let g = model(dir.path());
    let args = SsdArgs {
        prefetch: SsdPrefetch::Router,
        prefetch_ranks: 1,
        ..SsdArgs::default()
    };
    let t = tier_args(&g, 3, args);
    // The prediction still ranks n_used 2 (the trace's width); 1 of them is read ahead.
    assert_eq!(t.predict_width(3, Predictor::PostAttn, 1), Some(2));
    t.predict_host(3, Predictor::PostAttn, &[4, 1], 1).unwrap();
    let out = t.fetch_host(3, &[1, 4], 1).unwrap();
    assert_slots(&t, 3, &out, &[1, 4]);
    let s = t.stats();
    assert_eq!(
        (s.hits, s.misses, s.prefetched, s.prefetch_used),
        (1, 1, 1, 1)
    );

    // router-both: the cap holds the post-attention lookahead; the post-MoE one reads the rest.
    let args = SsdArgs {
        prefetch: SsdPrefetch::RouterBoth,
        prefetch_ranks: 1,
        ..SsdArgs::default()
    };
    let t = tier_args(&g, 3, args);
    t.predict_host(3, Predictor::PostAttn, &[4, 1], 1).unwrap();
    assert_eq!(t.stats().prefetched, 1);
    t.predict_host(3, Predictor::PostMoe, &[4, 1], 1).unwrap();
    let out = t.fetch_host(3, &[1, 4], 1).unwrap();
    assert_slots(&t, 3, &out, &[1, 4]);
    let s = t.stats();
    assert_eq!(
        (s.hits, s.misses, s.prefetched, s.prefetch_used),
        (2, 0, 2, 2)
    );
}

/// `--ssd-prefetch router`: a post-attention prediction reads its experts into slots; the
/// layer's fetch finds the ones it routes to there (the same bytes as a read on demand) and
/// counts the others wasted. Other predictors and big steps prefetch nothing.
#[test]
fn the_router_prefetch_reads_predicted_experts_before_the_fetch() {
    let dir = scratch();
    let g = model(dir.path());
    let off = tier(&g, 3, true);
    assert_eq!(off.predict_width(3, Predictor::PostAttn, 1), None);
    let t = tier_with(&g, 3, SsdPrefetch::Router, 1);
    // n_used 2 + margin 1; 2 tokens of 2 fill the 3 slots: a big step, no prediction.
    assert_eq!(t.predict_width(3, Predictor::PostAttn, 1), Some(3));
    assert_eq!(t.predict_width(3, Predictor::PostMoe, 1), None);
    assert_eq!(t.predict_width(3, Predictor::PostAttn, 2), None);
    assert_eq!(t.predict_width(2, Predictor::PostAttn, 1), None);
    // Ranked 4, 1, 5: all three read ahead into layer 3's slots.
    t.predict_host(3, Predictor::PostAttn, &[4, 1, 5], 1)
        .unwrap();
    let out = t.fetch_host(3, &[1, 4], 1).unwrap();
    for (k, kind) in KINDS.iter().enumerate() {
        for (&s, e) in out.iter().zip([1, 4]) {
            assert_eq!(
                held(&t, &format!("ssd.3.{kind}"), s as usize),
                want(3, k, e)
            );
        }
    }
    let s = t.stats();
    assert_eq!((s.hits, s.misses), (2, 0));
    assert_eq!(s.prefetched, 3);
    assert_eq!(s.prefetch_used, 2);
    assert_eq!(s.prefetch_wasted + s.prefetch_dropped, 1);
    // The same steps without the prefetch: the same experts in the matmuls' slots.
    let out_off = off.fetch_host(3, &[1, 4], 1).unwrap();
    for kind in KINDS {
        for (i, e) in [1, 4].into_iter().enumerate() {
            assert_eq!(
                held(&off, &format!("ssd.3.{kind}"), out_off[i] as usize),
                held(&t, &format!("ssd.3.{kind}"), out[i] as usize),
                "expert {e}"
            );
        }
    }
    // A post-MoE prediction (trace only) and a prediction for another layer read nothing the
    // fetch of this one waits for.
    t.predict_host(3, Predictor::PostMoe, &[0, 2, 3], 1)
        .unwrap();
    assert_eq!(t.stats().prefetched, 3);
    t.predict_host(1, Predictor::PostAttn, &[0, 2, 3], 1)
        .unwrap();
    t.fetch_host(3, &[0, 2], 1).unwrap();
    let s = t.stats();
    assert_eq!(s.prefetched, 6);
    assert_eq!(s.prefetch_used, 2);
    // Layer 1 holds whatever of 0, 2, 3 completed; a fetch of it is still exact.
    let out = t.fetch_host(1, &[3, 0], 1).unwrap();
    for (k, kind) in KINDS.iter().enumerate() {
        for (&s, e) in out.iter().zip([3, 0]) {
            assert_eq!(
                held(&t, &format!("ssd.1.{kind}"), s as usize),
                want(1, k, e)
            );
        }
    }
}

/// Asserts that the slots `out` maps `experts` of layer `il` to hold those experts.
fn assert_slots(t: &SsdExperts, il: usize, out: &[i32], experts: &[usize]) {
    for (k, kind) in KINDS.iter().enumerate() {
        for (&s, &e) in out.iter().zip(experts) {
            assert_eq!(
                held(t, &format!("ssd.{il}.{kind}"), s as usize),
                want(il, k, e),
                "layer {il} {kind} expert {e}"
            );
        }
    }
}

/// #615: the readers take a step's own reads first, then reads ahead (the routed count before
/// the margin), each class in order; a claimed read ahead moves up to the step's reads, and a
/// cancelled group leaves the queues at once.
#[test]
fn reader_queues_take_demand_first_and_drop_cancelled_groups() {
    use super::io::{Class, Job, Pool};
    let dir = scratch();
    let path = dir.path().join("f.bin");
    std::fs::write(&path, [0u8; 16]).unwrap();
    // No reader threads: every job stays queued.
    let pool = Pool::with_threads(&[path], 0, false).unwrap();
    let buf = [0u8; 4];
    let job = || Job::Copy {
        src: buf.as_ptr() as usize,
        dst: buf.as_ptr() as usize + 2,
        len: 1,
    };
    let extra = pool.submit_as(vec![job(), job()], Class::Extra);
    let ahead = pool.submit_as(vec![job()], Class::Ahead);
    let ahead2 = pool.submit_as(vec![job()], Class::Ahead);
    let demand = pool.submit(vec![job()]);
    let groups = [&demand, &ahead, &ahead2, &extra];
    let order = |p: &Pool| -> Vec<usize> {
        p.queued()
            .iter()
            .map(|g| groups.iter().position(|x| Arc::ptr_eq(x, g)).unwrap())
            .collect()
    };
    assert_eq!(order(&pool), vec![0, 1, 2, 3, 3]);
    pool.promote(&ahead2);
    assert_eq!(order(&pool), vec![0, 2, 1, 3, 3]);
    pool.cancel(&extra);
    assert!(extra.done() && extra.skipped());
    assert!(extra.wait().is_ok());
    assert_eq!(order(&pool), vec![0, 2, 1]);
    assert!(!ahead.done() && !ahead.skipped());
}

/// #615: a slot with a read in flight is never a victim, of a step or of a prefetch.
#[test]
fn pinned_slots_are_never_victims() {
    let mut c = SlotCache::new(10, 3);
    c.plan(&[0]);
    c.plan(&[1]);
    c.plan(&[2]);
    c.pin(0, true);
    // Slot 0 (expert 0) is the least recently used, but pinned: slot 1 goes.
    assert_eq!(c.plan(&[5]).loads, vec![(5, 1)]);
    assert_eq!(c.slot(0), Some(0));
    assert_eq!(c.prefetch(&[6, 7], usize::MAX), vec![(6, 2), (7, 1)]);
    // Dropping the slot unpins and empties it: the next step takes it first.
    c.drop_slot(0);
    assert_eq!(c.plan(&[8]).loads, vec![(8, 0)]);
}

/// #615: `--ssd-prefetch-early` moves only a post-attention lookahead that reads ahead.
#[test]
fn early_lookahead_needs_a_post_attention_prefetch() {
    let dir = scratch();
    let g = model(dir.path());
    let expert = 3 * ROWS * K / 64 * 36;
    let bytes = (2 * N_EXPERT + LAYERS.len() * 3) * expert;
    let buft = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
    let early = |prefetch| {
        let args = SsdArgs {
            mode: ExpertSsd::On,
            readers: 1,
            prefetch,
            prefetch_early: true,
            ..SsdArgs::default()
        };
        SsdExperts::new(&g, &LAYERS, 4, N_EXPERT, 2, bytes as u64, buft, 64, &args)
            .unwrap()
            .prefetch_early()
    };
    assert!(early(SsdPrefetch::Router));
    assert!(early(SsdPrefetch::RouterBoth));
    assert!(!early(SsdPrefetch::RouterPost));
    assert!(!early(SsdPrefetch::Off));
}

/// #615: `router-post` reads ahead by the post-MoE prediction alone; `router-both` starts from
/// the post-attention one and lets the post-MoE one cancel what it no longer names and add
/// what it does. The step's experts are exact either way, a cancelled one it routes to after
/// all included.
#[test]
fn post_moe_predictions_read_ahead_and_refine_post_attention_ones() {
    let dir = scratch();
    let g = model(dir.path());
    let post = tier_with(&g, 3, SsdPrefetch::RouterPost, 0);
    assert_eq!(post.predict_width(3, Predictor::PostAttn, 1), None);
    assert_eq!(post.predict_width(3, Predictor::PostMoe, 1), Some(2));
    post.predict_host(3, Predictor::PostAttn, &[0, 2], 1)
        .unwrap();
    assert_eq!(post.stats().prefetched, 0);
    post.predict_host(3, Predictor::PostMoe, &[4, 1], 1)
        .unwrap();
    let out = post.fetch_host(3, &[1, 4], 1).unwrap();
    assert_slots(&post, 3, &out, &[1, 4]);
    let s = post.stats();
    assert_eq!(
        (s.hits, s.misses, s.prefetched, s.prefetch_used),
        (2, 0, 2, 2)
    );

    let both = tier_with(&g, 3, SsdPrefetch::RouterBoth, 0);
    assert_eq!(both.predict_width(3, Predictor::PostAttn, 1), Some(2));
    assert_eq!(both.predict_width(3, Predictor::PostMoe, 1), Some(2));
    both.predict_host(3, Predictor::PostAttn, &[4, 1], 1)
        .unwrap();
    // The post-MoE ranking keeps 1, no longer names 4 and adds 5.
    both.predict_host(3, Predictor::PostMoe, &[1, 5], 1)
        .unwrap();
    let out = both.fetch_host(3, &[5, 1], 1).unwrap();
    assert_slots(&both, 3, &out, &[5, 1]);
    let s = both.stats();
    assert_eq!(
        (s.hits, s.misses, s.prefetched, s.prefetch_used),
        (2, 0, 3, 2)
    );
    // A big step ends every read ahead: 4's is counted wasted or dropped.
    both.fetch_host(3, &[0, 1, 2, 3, 4, 5], 3).unwrap();
    let s = both.stats();
    assert_eq!(s.prefetch_wasted + s.prefetch_dropped, 1);
    let out = both.fetch_host(3, &[4, 0], 1).unwrap();
    assert_slots(&both, 3, &out, &[4, 0]);

    // The step routes to the expert the post-MoE ranking dropped: it waits for that read and
    // reads the expert again if the cancel stopped it.
    let both = tier_with(&g, 3, SsdPrefetch::RouterBoth, 0);
    both.predict_host(1, Predictor::PostAttn, &[4, 1], 1)
        .unwrap();
    both.predict_host(1, Predictor::PostMoe, &[1, 5], 1)
        .unwrap();
    let out = both.fetch_host(1, &[4, 1], 1).unwrap();
    assert_slots(&both, 1, &out, &[4, 1]);
    let s = both.stats();
    assert_eq!(s.hits + s.misses, 2);
    assert_eq!(s.prefetched, 3);
    // Layer 3 is untouched by layer 1's reads ahead.
    let out = both.fetch_host(3, &[2, 3], 1).unwrap();
    assert_slots(&both, 3, &out, &[2, 3]);
}

/// The offline lookahead: a predictor that names each step's experts has full recall, and its
/// prefetch turns every miss after the first fills into a read ahead.
#[test]
fn lookahead_replays_count_recall_and_reads_ahead() {
    use super::lookahead::{self, Lookahead};
    let mut t = trace_of(16, &[]);
    for i in 0..40u16 {
        let ids = vec![i % 16, (i * 7 + 3) % 16];
        // The prediction ranks the two routed ones first, then two others.
        let pred = vec![ids[1], ids[0], (i + 5) % 16, (i + 9) % 16];
        t.predictions.push(Prediction {
            ord: 0,
            predictor: Predictor::PostAttn,
            n_tokens: 1,
            ids: pred,
            before: t.records.len(),
        });
        t.records.push(replay_trace::Record {
            ord: 0,
            big: false,
            n_tokens: 1,
            ids,
        });
    }
    let r = lookahead::recall(&t, Predictor::PostAttn, 0, 4);
    assert_eq!((r.steps, r.predicted_steps), (40, 40));
    assert_eq!(r.recall(), 1.0);
    assert_eq!(r.miss_recall(), 1.0);
    assert!(lookahead::recall(&t, Predictor::Mtp, 0, 4).routed == 0);
    let la = |margin| Lookahead {
        predictor: Predictor::PostAttn,
        margin,
        n_slots: 4,
        budget: usize::MAX,
        ranks: usize::MAX,
    };
    let lru = replay::replay(
        &t,
        &replay::Setup {
            policy: replay::Policy::Lru,
            n_slots: 4,
            seed: None,
        },
    );
    let o = lookahead::replay(&t, &la(0));
    assert_eq!(o.misses, 0);
    assert_eq!(o.wasted, 0);
    assert_eq!(o.prefetched, lru.misses);
    assert_eq!(o.tokens, 40);
    // A margin of 2 reads the two others too: wasted reads, no misses.
    let o2 = lookahead::replay(&t, &la(2));
    assert_eq!(o2.misses, 0);
    assert!(o2.wasted > 0 && o2.reads() > o.reads());
}

/// Runs `f` on a thread of its own and fails if it does not finish within `secs` seconds (a
/// hung thread is left behind; the test fails instead of hanging the run).
fn within(secs: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(()) => {}
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("did not finish within {secs} s (a reader thread missed the stop?)")
        }
        // The closure panicked: its own message is on stderr.
        Err(e) => panic!("{e}"),
    }
}

/// #643: dropping the readers right after they start never hangs. The stop flag was set
/// outside the queue's lock, so a reader between its stop check and its wait missed the
/// wakeup and the drop's join waited for ever.
#[test]
fn readers_created_and_dropped_at_once_never_hang() {
    use super::io::Pool;
    within(120, || {
        let dir = scratch();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        for i in 0..2000 {
            drop(Pool::with_threads(std::slice::from_ref(&path), 1 + i % 4, false).unwrap());
        }
    });
}

/// #643: the whole tier, created and dropped many times in a row (what
/// `early_lookahead_needs_a_post_attention_prefetch` does four times), never hangs.
#[test]
fn tiers_created_and_dropped_in_a_row_never_hang() {
    within(120, || {
        let dir = scratch();
        let g = model(dir.path());
        let expert = 3 * ROWS * K / 64 * 36;
        let bytes = (2 * N_EXPERT + LAYERS.len() * 2) * expert;
        let buft = device_by_kind(DeviceKind::Cpu).unwrap().buffer_type();
        for i in 0..1000 {
            let args = SsdArgs {
                mode: ExpertSsd::On,
                readers: 1 + i % 3,
                prefetch: SsdPrefetch::Router,
                prefetch_early: i % 2 == 0,
                ..SsdArgs::default()
            };
            let t = SsdExperts::new(&g, &LAYERS, 4, N_EXPERT, 2, bytes as u64, buft, 64, &args);
            drop(t.unwrap());
        }
    });
}

/// #1205: a piece of a chunked step evicts the experts its later pieces need last; with no
/// later pieces it plans the slots [`SlotCache::plan`] does.
#[test]
fn pieces_keep_the_experts_later_pieces_need() {
    let mut c = SlotCache::new(8, 3);
    c.plan(&[0, 1, 2]);
    // All three tie on use: plain LRU would take slot 0 (expert 0, needed next), this takes
    // expert 1 (never needed again).
    let mut next = vec![u32::MAX; 8];
    (next[0], next[2]) = (1, 4);
    let p = c.plan_ahead(&[3], &next);
    assert_eq!((p.hits, p.loads), (0, vec![(3, 1)]));
    // Expert 2 is needed later than 0 and 3.
    next[3] = 2;
    let p = c.plan_ahead(&[5], &next);
    assert_eq!(p.loads, vec![(5, 2)]);
    assert_eq!(
        (c.slot(0), c.slot(3), c.slot(5)),
        (Some(0), Some(1), Some(2))
    );

    let (mut a, mut b) = (SlotCache::new(16, 4), SlotCache::new(16, 4));
    let never = vec![u32::MAX; 16];
    for step in 0..40usize {
        let want: Vec<usize> = (0..3).map(|i| (step * 5 + i * 7) % 16).collect();
        assert_eq!(a.plan(&want), b.plan_ahead(&want, &never), "step {step}");
    }
}

#[test]
fn piece_rows_split_evenly() {
    assert_eq!(piece_rows(14, 2), vec![0..7, 7..14]);
    assert_eq!(piece_rows(27, 3), vec![0..9, 9..18, 18..27]);
    assert_eq!(
        piece_rows(64, 5),
        vec![0..13, 13..26, 26..39, 39..52, 52..64]
    );
    assert_eq!(piece_rows(3, 5), vec![0..1, 1..2, 2..3]);
    assert_eq!(piece_rows(5, 1), vec![0..5]);
}

/// Decode and verify steps of more pairs than slots, up to twice the experts, go in pieces of
/// at most the slots' rows; smaller steps, prompt steps and bigger ones do not.
#[test]
fn decode_steps_past_the_slots_go_in_pieces() {
    let dir = scratch();
    let g = model(dir.path());
    // 5 slots, 2 experts a token: 2 rows a piece; 6 experts: at most 12 pairs (6 rows).
    let t = tier(&g, 5, true);
    assert_eq!(t.pieces(2, true), None);
    assert_eq!(t.pieces(3, true), Some(2));
    assert_eq!(t.pieces(5, true), Some(3));
    assert_eq!(t.pieces(6, true), Some(3));
    assert_eq!(t.pieces(7, true), None);
    assert_eq!(t.pieces(3, false), None);
    for n in 3..=6 {
        let k = t.pieces(n, true).unwrap();
        for r in piece_rows(n, k) {
            assert!(r.len() * 2 <= t.n_slots(), "{n} rows: piece {r:?}");
        }
    }
}

/// A chunked step's pieces read their rows' experts into slots, one after the other, each
/// keeping the experts the rows after it route to; the op's params give the same slots.
#[test]
fn a_chunked_step_reads_each_piece_into_slots() {
    for direct in [true, false] {
        let dir = scratch();
        let g = model(dir.path());
        let t = tier(&g, 3, direct);
        let ids = [0, 1, 2, 3, 0, 4];
        assert!(t.fetch_host_rows(3, &ids, 0..2).is_err());
        assert!(t.fetch_host_rows(3, &ids, 2..4).is_err());
        let out = t.fetch_host_rows(3, &ids, 0..1).unwrap();
        assert_eq!(out, vec![0, 1]);
        assert_slots(&t, 3, &out, &[0, 1]);
        // Expert 3 evicts 1 (never needed again), not 0 (the last row's).
        let out = t.fetch_host_rows(3, &ids, 1..2).unwrap();
        assert_eq!(out, vec![2, 1]);
        assert_slots(&t, 3, &out, &[2, 3]);
        let out = t.fetch_host_rows(3, &ids, 2..3).unwrap();
        assert_eq!(out[0], 0);
        assert_slots(&t, 3, &out, &[0, 4]);
        let s = t.stats();
        assert_eq!((s.hits, s.misses), (1, 5));
        assert_eq!(s.stage_bytes, 0);

        // The fetch op's work on the same step: a piece by its params, then the whole step.
        let u = tier(&g, 3, direct);
        let inner = lookup(u.id);
        let mut out = vec![0; 2];
        for (row, want) in [(0, [0, 1]), (1, [2, 1])] {
            run_fetch(&inner, &[u.id, 1, 0, row, 1], 2, &ids, &mut out).unwrap();
            assert_eq!(out, want, "row {row}");
        }
        assert!(run_fetch(&inner, &[u.id, 1, 0, 2, 2], 2, &ids, &mut out).is_err());
        let mut all = vec![0; 2];
        run_fetch(&inner, &[u.id, 1, 0, 0, 0], 2, &ids[..2], &mut all).unwrap();
        assert_slots(&u, 3, &all, &[0, 1]);
    }
}
