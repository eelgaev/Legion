//! The expert residency path (tiered expert tensors, the route log, swaps) on the CPU backend
//! with its host-copy backing: whatever the cache holds and however it moves experts between
//! steps, every logit must be bit-identical to the plain expert tensors, with batched
//! sequences, chunked prefill, MTP passes and speculative verify steps. The same for the layers
//! split over pipeline stages (each its own CPU backend here, a GPU each in production).

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
use legion::residency::experts::{ExpertResidency, LayerPlace};
use legion::residency::policy::SwapRule;
use legion::residency::profile::Profile;
use legion::residency::{ExpertMode, ResidencyConfig};
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

fn write_models(dir: &std::path::Path) {
    legion::model::qwen4::tiny::write(dir.join("tiny.gguf"), Some(&dir.join("mtp.gguf")), 11)
        .unwrap();
}

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

fn executor(dir: &std::path::Path) -> Executor {
    staged(dir, &[])
}

/// The tiny model with its layers split at `cuts` (one CPU backend per stage).
fn staged(dir: &std::path::Path, cuts: &[usize]) -> Executor {
    let b = cpu();
    let mut m = Model::load(&GgufFile::open(dir.join("tiny.gguf")).unwrap(), |_| {
        b.buffer_type()
    })
    .unwrap();
    m.load_mtp(&GgufFile::open(dir.join("mtp.gguf")).unwrap(), |_| {
        b.buffer_type()
    })
    .unwrap();
    let n_layer = m.hp.n_layer;
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(n_layer);
    let stages = bounds.windows(2).map(|w| (cpu(), w[0]..w[1])).collect();
    drop(b);
    let mut e = Executor::new_staged(m, stages, 16, 4, 0).unwrap();
    e.enable_mtp().unwrap();
    e.capture_hidden(vec![n_layer - 1]);
    e
}

/// Bytes of one expert (up + gate + down) of the tiny model.
fn expert_bytes(e: &Executor) -> u64 {
    ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"]
        .iter()
        .map(|k| {
            let t = e.model().tensor(&format!("blk.0.{k}.weight")).unwrap();
            (t.nbytes() / t.ne()[2] as usize) as u64
        })
        .sum()
}

fn tier(e: &mut Executor, slots_per_layer: u64, cfg: &ResidencyConfig) {
    let hp = e.hp().clone();
    let per = expert_bytes(e);
    let buft = Backend::cpu().unwrap().buffer_type();
    let model = e.model();
    let r = ExpertResidency::new(
        hp.n_layer + 1,
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        |n| model.tensor(n),
        |_| Some(LayerPlace { cuda: None, buft }),
        |_| slots_per_layer * per * (hp.n_layer as u64 + 1),
        cfg,
    )
    .unwrap();
    e.set_experts(Some(r));
}

/// Two sequences prefilled in chunks, then decoded together, then a verify step and MTP
/// passes; every output row of every step.
fn run(e: &mut Executor) -> Vec<Vec<f32>> {
    let w = e.mtp_width();
    let blocks: [[u32; 4]; 2] = [[0, 1, 2, 3], [4, 5, 6, 7]];
    let prompts: [Vec<u32>; 2] = [
        "expert residency keeps the hottest experts in VRAM"
            .bytes()
            .map(u32::from)
            .collect(),
        "the rest is read in place over the link, 0123456789"
            .bytes()
            .map(u32::from)
            .collect(),
    ];
    let mut outs = Vec::new();
    let mut n_past = [0usize; 2];
    // Chunked prefill, one sequence per group.
    for (i, p) in prompts.iter().enumerate() {
        for chunk in p.chunks(20) {
            let g = Group {
                slot: i,
                seqs: vec![SeqInput {
                    tokens: chunk,
                    n_past: n_past[i],
                    blocks: &blocks[i],
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                }],
                verify: false,
            };
            let out = e.step(&[g]).unwrap();
            outs.push(out.rows.concat());
            n_past[i] += chunk.len();
        }
    }
    // Batched decode: both sequences, one token each, greedy chain.
    let mut last = [prompts[0][0], prompts[1][0]];
    for _ in 0..12 {
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
        // An MTP pass over the step's residuals.
        let h = &out.hidden[0];
        assert_eq!(h.len(), 2 * w);
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
        let m = e.mtp_step(&[mg], h).unwrap();
        outs.push(m.pmax.clone());
        outs.push(m.argmax.iter().map(|&x| x as f32).collect());
    }
    outs
}

#[test]
fn tiered_experts_are_bit_identical_while_swapping() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run(&mut executor(dir.path()));

    // Two of eight experts per layer cached; swaps at any gain, many per step.
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
    let mut e = executor(dir.path());
    tier(&mut e, 2, &cfg);
    let tiered = run(&mut e);
    assert_eq!(plain.len(), tiered.len());
    for (i, (a, b)) in plain.iter().zip(&tiered).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len(),
            "output {i} differs"
        );
    }
    let r = e.model().experts().unwrap();
    let s = r.stats();
    assert_eq!(s.layers, 5, "4 layers and the MTP head");
    assert_eq!(s.slots, 10);
    // Prefill chunks (101 prompt tokens) and 12 decode steps × 2 tokens, 2 experts each in 4
    // layers, plus 12 MTP passes × 2 tokens × 2 experts.
    assert_eq!(s.hits + s.misses, (101 + 12 * 2) * 2 * 4 + 12 * 2 * 2);
    assert!(s.swaps > 0, "{s:?}");
    assert!(s.hits > 0 && s.misses > 0, "{s:?}");
    // The table names every expert exactly once.
    for il in 0..5 {
        let t = r.table(il).unwrap();
        let mut seen: Vec<i32> = t.iter().map(|&x| if x >= 2 { x - 2 } else { -1 }).collect();
        seen.retain(|&x| x >= 0);
        assert!(t.iter().filter(|&&x| x < 2).count() <= 2);
        assert_eq!(seen.len() + t.iter().filter(|&&x| x < 2).count(), 8);
    }

    // Gathering the selected experts first gives the same logits.
    let copy = ResidencyConfig {
        fetch: legion::residency::Fetch::Copy,
        ..cfg.clone()
    };
    let mut e = executor(dir.path());
    tier(&mut e, 2, &copy);
    assert_eq!(run(&mut e), plain, "fetch = copy");

    // No cache at all (every expert read from the host part) and a full cache.
    for slots in [0, 8] {
        let mut e = executor(dir.path());
        tier(&mut e, slots, &cfg);
        assert_eq!(run(&mut e), plain, "{slots} slots per layer");
        let s = e.model().experts().unwrap().stats();
        if slots == 8 {
            assert_eq!(s.misses, 0);
        } else {
            assert_eq!(s.hits, 0);
        }
    }
}

/// A layer split gives the same logits, bit for bit, with and without tiered experts.
#[test]
fn stages_are_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run(&mut executor(dir.path()));
    for cuts in [vec![1], vec![2], vec![1, 3]] {
        let mut e = staged(dir.path(), &cuts);
        assert_eq!(e.stage_layers().len(), cuts.len() + 1);
        assert_eq!(run(&mut e), plain, "split at {cuts:?}");
    }
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget: 3,
            margin: 0.0,
            ratio: 0.0,
        },
        ..Default::default()
    };
    let mut e = staged(dir.path(), &[2]);
    tier(&mut e, 3, &cfg);
    assert_eq!(run(&mut e), plain, "split with tiered experts");
    // Prompt chunks pipelined over the stages in micro-chunks (20-token chunks: 8 + 8 + 4).
    for cuts in [vec![1], vec![2], vec![1, 2], vec![1, 3], vec![1, 2, 3]] {
        for micro in [7, 8] {
            let mut e = staged(dir.path(), &cuts);
            e.set_pipeline_micro(micro);
            assert_eq!(
                run(&mut e),
                plain,
                "pipelined by {micro}, split at {cuts:?}"
            );
        }
    }
}

/// A decoding sequence and a short prompt's chunk ride ahead of a long prompt chunk
/// (`set_pipeline_riders`, #707): the riders as one micro-chunk, then the long chunk's
/// micro-chunks. Rows and the captured layer's rows come in the order of groups, bit for bit
/// as one whole step on one stage.
fn run_riders(e: &mut Executor) -> Vec<Vec<f32>> {
    let blocks: [[u32; 4]; 3] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
    let text = "riders go first, then the long prompt's micro-chunks, rows in group order";
    let long: Vec<u32> = text.bytes().map(u32::from).collect();
    let short: Vec<u32> = long[..9].iter().map(|t| t + 1).collect();
    let mut outs = Vec::new();
    // Sequence 0 reads a prompt, then decodes beside the others.
    let first = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &long[..20],
            n_past: 0,
            blocks: &blocks[0],
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let out = e.step(&[first]).unwrap();
    let next = [out.argmax[0]];
    let seq = |slot: usize, tokens, n_past| Group {
        slot,
        seqs: vec![SeqInput {
            tokens,
            n_past,
            blocks: &blocks[slot],
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let groups = [
        seq(0, &next[..], 20),
        seq(1, &short[..], 0),
        seq(2, &long[..], 0),
    ];
    let out = e.step(&groups).unwrap();
    assert_eq!(out.rows.len(), 3);
    outs.extend(out.rows);
    assert_eq!(
        out.hidden[0].len(),
        (1 + short.len() + long.len()) * e.mtp_width()
    );
    outs.extend(out.hidden);
    outs
}

#[test]
fn riders_ahead_of_a_long_chunk_are_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_riders(&mut executor(dir.path()));
    for cuts in [vec![2], vec![1, 3]] {
        for (micro, riders) in [(8, false), (8, true), (7, true), (100, true)] {
            let mut e = staged(dir.path(), &cuts);
            e.set_pipeline_micro(micro);
            e.set_pipeline_riders(riders);
            assert_eq!(
                run_riders(&mut e),
                plain,
                "micro {micro}, riders {riders}, split at {cuts:?}"
            );
        }
    }
}

#[test]
fn profile_picks_first_residents_and_learning_saves_one() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let path = dir.path().join("profile.bin");
    // Layer 0: experts 5 and 3 first; layer 2: expert 7.
    Profile {
        n_layers: 5,
        n_expert: 8,
        ranked: vec![(0, 5), (2, 7), (0, 3)],
    }
    .write(&path)
    .unwrap();
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        profile: Some(path.clone()),
        swaps: SwapRule {
            budget: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut e = executor(dir.path());
    tier(&mut e, 2, &cfg);
    {
        let r = e.model().experts().unwrap();
        // Layer 0's slots: 5, 3 (ranked); layer 2: 7, then the lowest id.
        let t0 = r.table(0).unwrap();
        assert_eq!((t0[5], t0[3]), (0, 1));
        let t2 = r.table(2).unwrap();
        assert_eq!((t2[7], t2[0]), (0, 1));
    }
    run(&mut e);
    drop(e);
    let saved = Profile::read(&path).unwrap();
    assert_eq!((saved.n_layers, saved.n_expert), (5, 8));
    assert!(saved.ranked.len() > 3, "learned pairs are saved");
}

fn engine(dir: &std::path::Path, residency: ResidencyConfig, k: usize) -> LmEngine {
    engine_split(dir, residency, k, &[])
}

fn engine_split(
    dir: &std::path::Path,
    residency: ResidencyConfig,
    k: usize,
    layer_split: &[usize],
) -> LmEngine {
    engine_split_with(dir, residency, k, layer_split, true)
}

fn engine_split_with(
    dir: &std::path::Path,
    residency: ResidencyConfig,
    k: usize,
    layer_split: &[usize],
    split_batches: bool,
) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        max_batch_tokens: 48,
        spec: SpecConfig {
            kind: if k > 0 { SpecKind::Mtp } else { SpecKind::None },
            k,
            draft: Some(dir.join("mtp.gguf")),
            p_min: 0.0,
            ..Default::default()
        },
        use_ref: true,
        residency,
        layer_split: layer_split.to_vec(),
        // Long prompts in 40-token chunks, pipelined over the stages in 16-token micro-chunks.
        prefill_chunk: 40,
        pipeline_micro: 16,
        split_batches,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

async fn answers(e: &LmEngine) -> Vec<String> {
    let prompts = [
        "Many users at once: ".repeat(4),
        "speculative".into(),
        "routing 0123456789".repeat(3),
        "abc".into(),
    ];
    let streams: Vec<_> = prompts
        .iter()
        .map(|p| {
            e.generate(GenerateRequest {
                model: "tiny".into(),
                prompt: Prompt::Text(p.clone()),
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                max_tokens: 32,
                stop: vec![],
                ignore_eos: false,
                min_tokens: 0,
                thinking: None,
                response_format: None,
                template_kwargs: Default::default(),
                cancel: CancellationToken::new(),
                scheduling: Default::default(),
            })
            .unwrap()
        })
        .collect();
    let mut out = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { .. } => break,
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
        out.push(text);
    }
    out
}

/// Two stages running decode and verify batches as one micro-batch per stage (stage 0 on the
/// second half while stage 1 runs the first) answer exactly as whole batches, with continuous
/// batching, MTP verify steps and a swapping expert cache.
#[tokio::test(flavor = "multi_thread")]
async fn split_batches_answer_the_same() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let cached = ResidencyConfig {
        mode: ExpertMode::Cache,
        cache_bytes: Some(5 * 2 * 6528),
        swaps: SwapRule {
            budget: 4,
            margin: 0.0,
            ratio: 0.0,
        },
        ..Default::default()
    };
    for k in [0, 3] {
        for residency in [ResidencyConfig::default(), cached.clone()] {
            let whole = answers(&engine_split_with(
                dir.path(),
                residency.clone(),
                k,
                &[2],
                false,
            ))
            .await;
            let split = answers(&engine_split_with(
                dir.path(),
                residency.clone(),
                k,
                &[2],
                true,
            ))
            .await;
            assert_eq!(whole, split, "spec k = {k}, {:?}", residency.mode);
            assert!(whole.iter().all(|t| !t.is_empty()));
        }
    }
}

/// Continuous batching (4 requests over 3 sequence slots), MTP drafts and verify steps: the
/// engine with a small swapping expert cache answers exactly as without one.
#[tokio::test(flavor = "multi_thread")]
async fn engine_with_expert_cache_answers_the_same() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    for k in [0, 3] {
        let plain = answers(&engine(dir.path(), ResidencyConfig::default(), k)).await;
        let cached = ResidencyConfig {
            mode: ExpertMode::Cache,
            // About two of eight experts per layer.
            cache_bytes: Some(5 * 2 * 6528),
            swaps: SwapRule {
                budget: 4,
                margin: 0.0,
                ratio: 0.0,
            },
            ..Default::default()
        };
        let got = answers(&engine(dir.path(), cached.clone(), k)).await;
        assert_eq!(plain, got, "spec k = {k}");
        assert!(plain.iter().all(|t| !t.is_empty()));
        // Two pipeline stages, with and without the cache.
        let split = answers(&engine_split(
            dir.path(),
            ResidencyConfig::default(),
            k,
            &[2],
        ))
        .await;
        assert_eq!(plain, split, "spec k = {k}, two stages");
        let both = answers(&engine_split(dir.path(), cached, k, &[1])).await;
        assert_eq!(plain, both, "spec k = {k}, two stages and the cache");
    }
}

/// The tiered tensor on CUDA (VRAM slots + a pinned host arena's bytes) against the same
/// experts in VRAM: one Q5_1 expert matmul with K = 640 (as Qwen3.8-Flash-Next's
/// `ffn_down_exps`, not a multiple of 512) for 4 tokens (ggml-cuda's MMVQ) and 32 tokens (MMQ,
/// which reads past each expert's last row: the arena's zeroed tail must keep that finite).
/// `--features cuda`; ignored: needs a GPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiered_mul_mat_id_matches_vram() {
    use legion::ggml::types;
    for (ty, k, rows) in [
        (types::Q5_1, 640i64, 256i64),
        (types::Q4_K, 2560, 640),
        (types::Q8_0, 640, 2560),
    ] {
        tiered_case(ty, k, rows);
    }
}

#[cfg(feature = "cuda")]
fn tiered_case(ty: legion::ggml::ffi::ggml_type, k: i64, rows: i64) {
    use legion::ggml::types;
    use legion::ggml::{ComputeAllocator, Context, DeviceKind, device_by_kind};
    use legion::residency::cuda::{caps, host_arena, tiered};
    let (n_expert, used) = (64i64, 8i64);
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 2000) as f32 / 1000.0 - 1.0
    };
    let w: Vec<f32> = (0..k * rows * n_expert).map(|_| rnd()).collect();
    let bytes = legion::ggml::quantize(ty, &w, k as usize);
    let weights = Context::new(2, 0);
    let in_vram = weights.new_tensor(ty, &[k, rows, n_expert]).named("vram");
    weights.alloc_tensors(dev.buffer_type()).unwrap();
    in_vram.set_bytes(&bytes);
    let host_ctx = Context::new(1, 0);
    let in_host = host_ctx.new_tensor(ty, &[k, rows, n_expert]).named("host");
    host_ctx.alloc_tensors(arena).unwrap();
    in_host.set_bytes(&bytes);
    // Slots 0..4 hold experts 3, 7, 1, 12.
    let cached: Vec<usize> = (0..40)
        .map(|i| (i * 7 + 3) % 64)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let n_slots = cached.len();
    let tiered_ctx = tiered(0, in_host, n_slots, None, "tiered").unwrap();
    let t = tiered_ctx.tensor("tiered").unwrap();
    let per = in_host.nb()[2];
    for (s, &e) in cached.iter().enumerate() {
        t.set_bytes_at(s * per, &bytes[e * per..(e + 1) * per]);
    }
    // The same slots followed by a VRAM staging buffer holding the uncached experts.
    let mem = legion::residency::cuda::DeviceMem::new(0, in_host.nbytes() + per).unwrap();
    let staged_ctx = legion::residency::cuda::staged(0, t, &mem, "staged").unwrap();
    let st = staged_ctx.tensor("staged").unwrap();
    for e in (0..n_expert as usize).filter(|e| !cached.contains(e)) {
        st.set_bytes_at((n_slots + e) * per, &bytes[e * per..(e + 1) * per]);
    }
    let table: Vec<i32> = (0..n_expert as usize)
        .map(|e| match cached.iter().position(|&c| c == e) {
            Some(s) => s as i32,
            None => (n_slots + e) as i32,
        })
        .collect();
    for n in [4i64, 32, 301] {
        let ctx = Context::new(16, 16);
        let x = ctx.new_tensor(types::F32, &[k, 1, n]).as_input();
        let ids = ctx.new_tensor(types::I32, &[used, n]).as_input();
        let ids_t = ctx.new_tensor(types::I32, &[used, n]).as_input();
        let a = ctx.mul_mat_id(in_vram, x, ids).as_output();
        let b = ctx.mul_mat_id(t, x, ids_t).as_output();
        let c = ctx.mul_mat_id(st, x, ids_t).as_output();
        let mut g = ctx.new_graph(16);
        g.expand(a);
        g.expand(b);
        g.expand(c);
        ctx.alloc_graph(&g, ComputeAllocator::new(dev.buffer_type()))
            .unwrap();
        let xs: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        x.set_f32(&xs);
        // Every token routes to 4 distinct experts, cached and not.
        let picks: Vec<i32> = (0..n * used)
            .map(|i| ((i / used * 5 + (i % used) * 3) % n_expert) as i32)
            .collect();
        ids.set_i32(&picks);
        let remapped: Vec<i32> = picks.iter().map(|&e| table[e as usize]).collect();
        ids_t.set_i32(&remapped);
        backend.compute(&g).unwrap();
        let c = c.to_f32();
        let (a, b) = (a.to_f32(), b.to_f32());
        assert!(
            c.iter().all(|v| v.is_finite()),
            "{n} tokens: staged non-finite"
        );
        let dc = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!("{ty} {n} tokens: staged max |diff| {dc}");
        assert!(
            b.iter().all(|v| v.is_finite()),
            "{n} tokens: non-finite output"
        );
        let scale = a.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!("{ty} {n} tokens: max |diff| {diff} of {scale}");
        assert!(
            diff <= 1e-3 * scale,
            "{n} tokens: max |diff| {diff} of {scale}"
        );
    }
}

/// Up, gate and down as the fused op takes them.
#[cfg(feature = "cuda")]
fn refs<'a>(v: &[legion::ggml::Tensor<'a>]) -> legion::model::moe_ffn::ExpertRefs<'a> {
    legion::model::moe_ffn::ExpertRefs {
        up: v[0],
        gate: v[1],
        down: v[2],
        table: None,
    }
}

/// The fused expert op over a layer's staged tensors (VRAM slots followed by a staging buffer
/// of the uncached experts) with remapped ids, against the op over the same experts in VRAM:
/// the same bytes through the same kernels, so bit-identical; also over the tiered tensors read
/// in place from the host arena (what a forced op does without staging).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_fused_moe_over_staged_experts_matches_vram() {
    use legion::ggml::types;
    use legion::ggml::{ComputeAllocator, Context, DeviceKind, device_by_kind};
    use legion::model::moe_ffn::{PairOut, moe_ffn};
    use legion::residency::cuda::{DeviceMem, caps, host_arena, staged, tiered};
    let (n_embd, n_ff, n_expert, used) = (512i64, 256i64, 64i64, 8i64);
    let kinds = [
        (types::Q4_K, n_embd, n_ff),
        (types::Q4_K, n_embd, n_ff),
        (types::Q5_1, n_ff, n_embd),
    ];
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let mut seed = 777u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 2000) as f32 / 1000.0 - 1.0
    };
    let cached: Vec<usize> = (0..24).map(|i| (i * 7 + 3) % 64).collect();
    let cached: Vec<usize> = cached
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let n_slots = cached.len();
    let table: Vec<i32> = (0..n_expert as usize)
        .map(|e| match cached.iter().position(|&c| c == e) {
            Some(s) => s as i32,
            None => (n_slots + e) as i32,
        })
        .collect();
    let vram_ctx = Context::new(3, 0);
    let host_ctx = Context::new(3, 0);
    let mut vram = Vec::new();
    let mut host = Vec::new();
    let mut data = Vec::new();
    for (i, &(ty, k, m)) in kinds.iter().enumerate() {
        let w: Vec<f32> = (0..k * m * n_expert).map(|_| 0.1 * rnd()).collect();
        data.push(legion::ggml::quantize(ty, &w, k as usize));
        vram.push(
            vram_ctx
                .new_tensor(ty, &[k, m, n_expert])
                .named(&format!("v{i}")),
        );
        host.push(
            host_ctx
                .new_tensor(ty, &[k, m, n_expert])
                .named(&format!("h{i}")),
        );
    }
    vram_ctx.alloc_tensors(dev.buffer_type()).unwrap();
    host_ctx.alloc_tensors(arena).unwrap();
    // Per kind: the tiered tensor's context, the staged one's and its buffer.
    let mut parts = Vec::new();
    for i in 0..3 {
        let bytes = &data[i];
        vram[i].set_bytes(bytes);
        host[i].set_bytes(bytes);
        let per = host[i].nb()[2];
        let tc = tiered(0, host[i], n_slots, None, "tiered").unwrap();
        let t = tc.tensor("tiered").unwrap();
        for (s, &e) in cached.iter().enumerate() {
            t.set_bytes_at(s * per, &bytes[e * per..(e + 1) * per]);
        }
        let mem = DeviceMem::new(0, host[i].nbytes() + per).unwrap();
        let sc = staged(0, t, &mem, "staged").unwrap();
        let st = sc.tensor("staged").unwrap();
        for e in (0..n_expert as usize).filter(|e| !cached.contains(e)) {
            st.set_bytes_at((n_slots + e) * per, &bytes[e * per..(e + 1) * per]);
        }
        parts.push((tc, sc, mem));
    }
    let tiered_t: Vec<_> = parts
        .iter()
        .map(|p| p.0.tensor("tiered").unwrap())
        .collect();
    let staged_t: Vec<_> = parts
        .iter()
        .map(|p| p.1.tensor("staged").unwrap())
        .collect();
    for n in [64i64, 300] {
        let ctx = Context::new(64, 64);
        let x = ctx.new_tensor(types::F32, &[n_embd, n]).as_input();
        let ids = ctx.new_tensor(types::I32, &[used, n]).as_input();
        let ids_t = ctx.new_tensor(types::I32, &[used, n]).as_input();
        let w = ctx.new_tensor(types::F32, &[used, n]).as_input();
        let a = ctx
            .cont(moe_ffn(&ctx, x, ids, w, refs(&vram), PairOut::F32))
            .as_output();
        let b = ctx
            .cont(moe_ffn(&ctx, x, ids_t, w, refs(&staged_t), PairOut::F32))
            .as_output();
        let c = ctx
            .cont(moe_ffn(&ctx, x, ids_t, w, refs(&tiered_t), PairOut::F32))
            .as_output();
        let mut g = ctx.new_graph(64);
        for t in [a, b, c] {
            g.expand(t);
        }
        ctx.alloc_graph(&g, ComputeAllocator::new(dev.buffer_type()))
            .unwrap();
        let xs: Vec<f32> = (0..n_embd * n).map(|_| rnd()).collect();
        x.set_f32(&xs);
        let picks: Vec<i32> = (0..n * used)
            .map(|i| ((i / used * 5 + (i % used) * 3) % n_expert) as i32)
            .collect();
        ids.set_i32(&picks);
        let remapped: Vec<i32> = picks.iter().map(|&e| table[e as usize]).collect();
        ids_t.set_i32(&remapped);
        let ws: Vec<f32> = (0..used * n).map(|_| rnd().abs() + 0.1).collect();
        w.set_f32(&ws);
        backend.compute(&g).unwrap();
        let (a, b, c) = (a.to_f32(), b.to_f32(), c.to_f32());
        assert!(a.iter().all(|v| v.is_finite()), "{n} tokens: non-finite");
        let bits = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(bits(&a, &b), "{n} tokens: staged differs from VRAM");
        assert!(
            bits(&a, &c),
            "{n} tokens: tiered in place differs from VRAM"
        );
    }
}

/// On CUDA: lending the cache's VRAM to a long prompt points every lookup at host memory and
/// sets up staging, and giving it back refills the slots in the background with the experts
/// they held, their bytes intact (two layers, three Q4_K / Q5_1 kinds, the second layer's
/// gate and up Q8_0 like an MTP head's, a host arena), also after a lend that stopped the
/// refill. Staging only the first layer sizes the staging buffers for it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_lend_and_restore_the_cache() {
    use legion::ggml::types;
    use legion::ggml::{Context, DeviceKind, Tensor, device_by_kind};
    use legion::residency::cuda::{caps, host_arena};
    use legion::residency::experts::KINDS;
    let (n_embd, n_ff, n_expert, n_layers) = (512i64, 256i64, 32usize, 2usize);
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let ctx = Context::new(3 * n_layers, 0);
    let mut seed = 99u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 2000) as f32 / 1000.0 - 1.0
    };
    let mut made = Vec::new();
    for il in 0..n_layers {
        for kind in KINDS {
            let (ty, k, m) = if kind == "ffn_down_exps" {
                (types::Q5_1, n_ff, n_embd)
            } else if il == 1 {
                (types::Q8_0, n_embd, n_ff)
            } else {
                (types::Q4_K, n_embd, n_ff)
            };
            let t = ctx
                .new_tensor(ty, &[k, m, n_expert as i64])
                .named(&format!("blk.{il}.{kind}.weight"));
            let w: Vec<f32> = (0..k * m * n_expert as i64).map(|_| 0.1 * rnd()).collect();
            made.push((t, legion::ggml::quantize(ty, &w, k as usize)));
        }
    }
    ctx.alloc_tensors(arena).unwrap();
    for (t, bytes) in &made {
        t.set_bytes(bytes);
    }
    let find = |name: &str| -> Option<Tensor<'_>> { ctx.tensor(name) };
    let place = |il: usize| {
        (il < n_layers).then_some(LayerPlace {
            cuda: Some(0),
            buft: dev.buffer_type(),
        })
    };
    // Bytes of one expert of each layer.
    let expert: u64 = made.iter().map(|(t, _)| t.nb()[2] as u64).sum();
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        staging: true,
        ..Default::default()
    };
    // Ten slots per layer.
    let budget = |_: Option<usize>| 10 * expert + 3 * (2u64 << 20) * n_layers as u64;
    let mut r = ExpertResidency::new(n_layers, n_expert, 8, find, place, budget, &cfg).unwrap();
    assert!(r.can_lend());
    // Two buffers per kind, each one expert past the widest staged layer's, 2 MiB granules.
    let footprint = |layer: usize| -> u64 {
        made[layer * 3..][..3]
            .iter()
            .map(|(t, _)| 2 * ((n_expert as u64 + 1) * t.nb()[2] as u64 + (2 << 20)))
            .sum()
    };
    assert_eq!(r.stage_bytes(), vec![(0, footprint(1))]);
    r.stage_below(1);
    assert_eq!(r.stage_bytes(), vec![(0, footprint(0))]);
    assert!(footprint(0) < footprint(1));
    // Slot s of layer il holds the bytes of the expert the table maps there.
    let check_slots = |r: &ExpertResidency| -> usize {
        let mut cached = 0;
        for il in 0..n_layers {
            let table = r.table(il).unwrap();
            let l = r.layer(il).unwrap();
            for (e, &id) in table.iter().enumerate() {
                if (id as usize) >= l.n_slots() {
                    continue;
                }
                cached += 1;
                for k in 0..3 {
                    let w = l.weight(k);
                    let per = w.nb()[2];
                    let all = w.bytes();
                    let got = &all[id as usize * per..(id as usize + 1) * per];
                    let host = &made[il * 3 + k].1[e * per..(e + 1) * per];
                    assert_eq!(got, host, "layer {il} expert {e} kind {k}");
                }
            }
        }
        cached
    };
    let before = check_slots(&r);
    assert_eq!(before, 10 * n_layers);
    let held: Vec<Vec<i32>> = (0..n_layers).map(|il| r.table(il).unwrap()).collect();
    for round in 0..2 {
        r.lend().unwrap();
        if round == 1 {
            // Lending again right after a restore stops its refill (#447); the next restore
            // still brings every held expert back. The lend waits for what the restore's
            // thread frees first (#466: the prompt's compute buffers).
            let freed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let f = freed.clone();
            r.restore_after(move || f.store(true, std::sync::atomic::Ordering::SeqCst))
                .unwrap();
            r.lend().unwrap();
            assert!(freed.load(std::sync::atomic::Ordering::SeqCst));
        }
        assert!(r.is_lent());
        for il in 0..n_layers {
            let n_slots = r.layer(il).unwrap().n_slots();
            assert!(
                r.table(il)
                    .unwrap()
                    .iter()
                    .all(|&id| id as usize >= n_slots),
                "round {round}: every lookup reads host memory while lent"
            );
        }
        let freed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = freed.clone();
        r.restore_after(move || f.store(true, std::sync::atomic::Ordering::SeqCst))
            .unwrap();
        assert!(!r.is_lent());
        // The refill finishes in the background; steps poll it.
        let t0 = std::time::Instant::now();
        while check_slots(&r) < before {
            assert!(
                t0.elapsed().as_secs() < 30,
                "round {round}: the refill never finished"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
            r.after_step(0, |_| true).unwrap();
        }
        assert!(
            freed.load(std::sync::atomic::Ordering::SeqCst),
            "round {round}: refilled before the restore freed what it was given"
        );
        let now: Vec<Vec<i32>> = (0..n_layers).map(|il| r.table(il).unwrap()).collect();
        for (a, b) in held.iter().zip(&now) {
            let cached = |t: &Vec<i32>| {
                let mut v: Vec<usize> = (0..t.len()).filter(|&e| (t[e] as usize) < 10).collect();
                v.sort();
                v
            };
            assert_eq!(
                cached(a),
                cached(b),
                "round {round}: the same experts come back"
            );
        }
    }
}

/// A prompt, then six verify steps of three tokens each with commits of one or two: every
/// output row.
fn verify_run(e: &mut Executor) -> Vec<Vec<f32>> {
    e.enable_spec(4).unwrap();
    let blocks = [0u32, 1, 2, 3];
    let prompt: Vec<u32> = "a layer split fills the next stage early"
        .bytes()
        .map(u32::from)
        .collect();
    let g = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &prompt,
            n_past: 0,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let mut outs = vec![e.step(&[g]).unwrap().rows.concat()];
    let mut n_past = prompt.len();
    let mut last = prompt[0];
    for round in 0..6u32 {
        let toks = [last, 40 + round, 41 + round];
        let g = Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens: &toks,
                n_past,
                blocks: &blocks,
                ring: 0,
                logits: Logits::All,
                media: None,
            }],
            verify: true,
        };
        let out = e.step(&[g]).unwrap();
        outs.push(out.rows.concat());
        // Keep one or two of the three fed positions.
        let keep = 1 + (round as usize % 2);
        e.commit(&[(0, keep)]).unwrap();
        n_past += keep;
        last = out.argmax[keep - 1];
    }
    e.wait_commit();
    outs
}

/// Verify steps (speculative, three rows) and their commits on a layer split of the tiny
/// qwen4exp (PLE, sparse attention): a later stage's inputs are filled while the stage before
/// it runs, and every row must equal the one-stage run bit for bit.
#[test]
fn split_verify_steps_are_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = verify_run(&mut executor(dir.path()));
    for cuts in [vec![1], vec![2], vec![1, 3]] {
        let got = verify_run(&mut staged(dir.path(), &cuts));
        assert_eq!(got.len(), plain.len());
        for (i, (a, b)) in plain.iter().zip(&got).enumerate() {
            assert!(
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "split at {cuts:?}: output {i} differs"
            );
        }
    }
}

/// Snapshots of every tiered layer's host table and uploaded row.
fn tables(r: &ExpertResidency, n: usize) -> Vec<(Vec<i32>, Vec<i32>)> {
    (0..n)
        .map(|il| (r.table(il).unwrap(), r.uploaded_table(il).unwrap()))
        .collect()
}

/// One prefill step of `n` tokens through every layer, routing logged.
fn prefill(e: &mut Executor, n: usize) {
    let blocks = [0u32, 1, 2, 3];
    let tokens: Vec<u32> = (0..n as u32).map(|t| (t * 37 + 11) % 200).collect();
    let g = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &tokens,
            n_past: 0,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    e.step(&[g]).unwrap();
}

/// A scoped after-step (`after_step_only`) plans, copies and uploads for its own layers only:
/// another layer's table, uploaded row and copies stay as they were, so a pass through that
/// layer may run meanwhile.
#[test]
fn scoped_after_step_leaves_other_layers_alone() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let mut e = executor(dir.path());
    // Learn without swapping, then let a scoped call swap at any gain.
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget: 0,
            margin: 0.0,
            ratio: 0.0,
        },
        decay: 0.9,
        ..Default::default()
    };
    tier(&mut e, 2, &cfg);
    let n = 48;
    prefill(&mut e, n);
    let r = e.experts_mut().unwrap();
    let n_layers = 5;
    let before = tables(r, n_layers);
    for (il, (t, up)) in before.iter().enumerate() {
        assert_eq!(t, up, "layer {il}: uploaded before the scoped call");
    }
    r.set_swap_rule(SwapRule {
        budget: 64,
        margin: 0.0,
        ratio: 0.0,
    });
    let a = |il: usize| il < 2;
    r.after_step_only(n, a).unwrap();
    let after = tables(r, n_layers);
    assert!(r.stats().swaps > 0, "the routing should move experts");
    for il in 0..n_layers {
        if a(il) {
            assert_eq!(after[il].0, after[il].1, "layer {il}: its row is uploaded");
        } else {
            assert_eq!(after[il], before[il], "layer {il} was touched");
            assert_eq!(r.copies_pending(il), 0);
        }
    }
    assert!(
        (0..n_layers).any(|il| a(il) && after[il] != before[il]),
        "a selected layer swapped"
    );
    // The other layers' own call, with nothing logged: no learning, nothing for the first ones.
    r.after_step_only(0, |il| !a(il)).unwrap();
    assert_eq!(tables(r, n_layers), after);
}

/// The tiny qwen4exp on GPU 0 with its experts tiered (two slots per layer, no swaps yet)
/// after one logged prefill step of `n` tokens; the number of tiered layers.
#[cfg(feature = "cuda")]
fn cuda_tiered(dir: &std::path::Path, n: usize) -> (Executor, usize) {
    let mut e = cuda_cache(dir, 2, 0);
    let n_layers = e.hp().n_layer;
    prefill(&mut e, n);
    (e, n_layers)
}

/// The tiny qwen4exp on GPU 0 with its experts tiered: `slots` VRAM slots per layer, at most
/// `budget` swaps per step (at any gain; 0: none), lendable to long prompts.
#[cfg(feature = "cuda")]
fn cuda_cache(dir: &std::path::Path, slots: u64, budget: usize) -> Executor {
    use legion::ggml::{DeviceKind, device_by_kind};
    use legion::residency::cuda::{caps, host_arena};
    let path = dir.join("tiny-qwen4exp.gguf");
    legion::model::qwen4::tiny::write_cuda_indexer(&path, None, 5).unwrap();
    let dev = device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let node = caps(0).map_or(0, |c| c.host_numa_id.max(0) as u32);
    let arena = host_arena(node, &[0]).expect("host arena");
    let gpu = dev.buffer_type();
    // The per-layer n-gram table is read on the host.
    let host = Backend::cpu().unwrap().buffer_type();
    let model = Model::load(&GgufFile::open(&path).unwrap(), |t| {
        if t.name.contains("_exps") {
            arena
        } else if t.name.starts_with("per_layer_token_embd") {
            host
        } else {
            gpu
        }
    })
    .unwrap();
    let mut e = Executor::new(model, dev.init().unwrap(), 16, 4, 0).unwrap();
    let hp = e.hp().clone();
    let per = expert_bytes(&e);
    let n_layers = hp.n_layer;
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget,
            margin: 0.0,
            ratio: 0.0,
        },
        decay: 0.9,
        // Lendable to long prompts (only `lend` uses it; the steps here are small).
        staging: true,
        ..Default::default()
    };
    let model = e.model();
    let r = ExpertResidency::new(
        n_layers,
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        |n| model.tensor(n),
        |_| {
            Some(LayerPlace {
                cuda: Some(0),
                buft: gpu,
            })
        },
        // `slots` per layer, and each kind's range rounded up to 2 MiB granules.
        |_| (slots * per + 3 * (2 << 20)) * n_layers as u64,
        &cfg,
    )
    .unwrap();
    e.set_experts(Some(r));
    e
}

/// Greedy output must not follow the expert cache (#327): the default kernels read an expert
/// in a VRAM slot as they read one in host memory, so every row of a run (a prompt, then
/// verify steps and their commits) has the same bits whatever the cache holds: no slots, a
/// few or every expert, with swaps landing between steps or none, and from run to run.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_do_not_follow_the_expert_cache() {
    let dir = tempfile::tempdir().unwrap();
    let host = verify_run(&mut cuda_cache(dir.path(), 0, 0));
    for (slots, budget) in [(2, 0), (2, 64), (2, 64), (5, 64), (8, 0)] {
        let mut e = cuda_cache(dir.path(), slots, budget);
        let got = verify_run(&mut e);
        let s = e.model().experts().unwrap().stats();
        if budget > 0 && slots < 8 {
            assert!(s.swaps > 0, "{slots} slots: no swaps ({s:?})");
        }
        assert_eq!(got.len(), host.len());
        for (i, (a, b)) in host.iter().zip(&got).enumerate() {
            assert!(
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{slots} slots, {budget} swaps per step: output {i} differs"
            );
        }
    }
}

/// The scoped after-step on a GPU: copies started for layers A complete only in A's own call,
/// never in a call for layers B, and B's uploaded rows do not change meanwhile.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scoped_after_step_keeps_copies_to_their_layers() {
    let dir = tempfile::tempdir().unwrap();
    let n = 48;
    let (mut e, n_layers) = cuda_tiered(dir.path(), n);
    let r = e.experts_mut().unwrap();
    let before = tables(r, n_layers);
    r.set_swap_rule(SwapRule {
        budget: 64,
        margin: 0.0,
        ratio: 0.0,
    });
    let a = |il: usize| il < 2;
    r.after_step_only(n, a).unwrap();
    let pending: Vec<usize> = (0..n_layers).map(|il| r.copies_pending(il)).collect();
    assert!(
        pending.iter().sum::<usize>() > 0,
        "the routing should move experts: {:?}",
        r.stats()
    );
    for (il, &p) in pending.iter().enumerate() {
        if !a(il) {
            assert_eq!(p, 0, "layer {il} got copies");
        }
    }
    let swapped = tables(r, n_layers);
    // Let the copies finish; a call for the other layers must leave them queued.
    std::thread::sleep(std::time::Duration::from_millis(200));
    r.after_step_only(0, |il| !a(il)).unwrap();
    for (il, &p) in pending.iter().enumerate() {
        assert_eq!(r.copies_pending(il), p, "layer {il}'s copies moved");
    }
    assert_eq!(tables(r, n_layers), swapped);
    for il in (0..n_layers).filter(|&il| !a(il)) {
        assert_eq!(swapped[il], before[il], "layer {il} was touched");
    }
    // Their own call completes them and uploads their rows.
    r.after_step_only(0, a).unwrap();
    for il in 0..n_layers {
        assert_eq!(r.copies_pending(il), 0, "layer {il}");
        let (t, up) = (r.table(il).unwrap(), r.uploaded_table(il).unwrap());
        assert_eq!(t, up, "layer {il}");
    }
    for il in (0..n_layers).filter(|&il| !a(il)) {
        assert_eq!(r.table(il).unwrap(), before[il].0, "layer {il} was touched");
    }
}

/// A copy batch the whole-cache after-step started for several layers completes per layer:
/// a scoped call marks only its own layers' slots filled and uploads only their rows, the
/// rest stay queued for their own call.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mixed_copy_batch_completes_per_layer() {
    let dir = tempfile::tempdir().unwrap();
    let n = 48;
    let (mut e, n_layers) = cuda_tiered(dir.path(), n);
    let r = e.experts_mut().unwrap();
    r.set_swap_rule(SwapRule {
        budget: 64,
        margin: 0.0,
        ratio: 0.0,
    });
    // One batch of copies for every layer that swaps.
    r.after_step(n, |_| true).unwrap();
    let pending: Vec<usize> = (0..n_layers).map(|il| r.copies_pending(il)).collect();
    let a = |il: usize| il < 2;
    assert!(
        (0..n_layers).any(|il| a(il) && pending[il] > 0)
            && (0..n_layers).any(|il| !a(il) && pending[il] > 0),
        "both sides should swap: {pending:?}"
    );
    let uploaded: Vec<Vec<i32>> = (0..n_layers)
        .map(|il| r.uploaded_table(il).unwrap())
        .collect();
    std::thread::sleep(std::time::Duration::from_millis(200));
    r.after_step_only(0, a).unwrap();
    for (il, &p) in pending.iter().enumerate() {
        if a(il) {
            assert_eq!(r.copies_pending(il), 0, "layer {il}");
            assert_eq!(r.table(il), r.uploaded_table(il), "layer {il}");
        } else {
            assert_eq!(r.copies_pending(il), p, "layer {il}'s copies moved");
            assert_eq!(r.uploaded_table(il).unwrap(), uploaded[il], "layer {il}");
        }
    }
    r.after_step_only(0, |il| !a(il)).unwrap();
    for il in 0..n_layers {
        assert_eq!(r.copies_pending(il), 0, "layer {il}");
        assert_eq!(r.table(il), r.uploaded_table(il), "layer {il}");
    }
}

/// The scoped after-step reads the routing log only over the rows of the layers that ran
/// (here not starting at row 0) and learns exactly what the whole-cache one learns for them.
#[test]
fn scoped_after_step_learns_the_same_routing() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget: 0,
            margin: 0.0,
            ratio: 0.0,
        },
        decay: 0.9,
        ..Default::default()
    };
    let n = 40;
    let ran = |il: usize| (2..4).contains(&il);
    let mut stats = Vec::new();
    for scoped in [false, true] {
        let mut e = executor(dir.path());
        tier(&mut e, 2, &cfg);
        prefill(&mut e, n);
        let r = e.experts_mut().unwrap();
        let before = r.layer_stats();
        if scoped {
            r.after_step_only(n, ran).unwrap();
        } else {
            r.after_step(n, ran).unwrap();
        }
        let after = r.layer_stats();
        for (b, a) in before.iter().zip(&after) {
            if ran(b.0) {
                assert!(a.2 + a.3 > b.2 + b.3, "layer {} learned nothing", b.0);
            } else {
                assert_eq!(a, b, "layer {} learned", b.0);
            }
        }
        stats.push(after);
    }
    assert_eq!(stats[0], stats[1]);
}

/// A step's after-step split in two calls (#362: the layers before the last stage's beside the
/// first MTP pass, the last stage's beside the second), `after_step_more` then
/// `after_step_only`, learns what one call over all the layers learns and counts one logged
/// step, not two.
#[test]
fn split_after_step_learns_and_counts_as_one() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let cfg = ResidencyConfig {
        mode: ExpertMode::Cache,
        swaps: SwapRule {
            budget: 0,
            margin: 0.0,
            ratio: 0.0,
        },
        decay: 0.9,
        ..Default::default()
    };
    let n = 40;
    let mut runs = Vec::new();
    for split in [false, true] {
        let mut e = executor(dir.path());
        let n_layer = e.hp().n_layer;
        tier(&mut e, 2, &cfg);
        prefill(&mut e, n);
        let r = e.experts_mut().unwrap();
        let logged = r.stats().logged_steps;
        // A decode-sized pass (logged), learned over the target layers.
        if split {
            r.after_step_more(3, |il| il < 2).unwrap();
            assert_eq!(r.stats().logged_steps, logged, "counted by the last call");
            r.after_step_only(3, |il| (2..n_layer).contains(&il))
                .unwrap();
        } else {
            r.after_step_only(3, |il| il < n_layer).unwrap();
        }
        let s = r.stats();
        assert_eq!(s.logged_steps, logged + 1, "split {split}");
        runs.push((
            r.layer_stats(),
            (0..=n_layer).map(|il| r.table(il)).collect::<Vec<_>>(),
        ));
    }
    assert_eq!(runs[0], runs[1]);
}

/// The tiny qwen4exp on GPU 0 (all weights in device memory).
#[cfg(feature = "cuda")]
fn cuda_plain(dir: &std::path::Path) -> Executor {
    cuda_plain_with(dir, false)
}

/// [`cuda_plain`], with every matmul's activations kept at 8 bits or wider (`a8`) as
/// `legion serve` keeps them by default (`--nvfp4-act a8`).
#[cfg(feature = "cuda")]
fn cuda_plain_with(dir: &std::path::Path, a8: bool) -> Executor {
    use legion::ggml::{DeviceKind, device_by_kind};
    let path = dir.join("tiny-qwen4exp.gguf");
    legion::model::qwen4::tiny::write_cuda_indexer(&path, None, 5).unwrap();
    let dev = device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let gpu = dev.buffer_type();
    // The per-layer n-gram table is read on the host.
    let host = Backend::cpu().unwrap().buffer_type();
    let mut model = Model::load(&GgufFile::open(&path).unwrap(), |t| {
        if t.name.starts_with("per_layer_token_embd") {
            host
        } else {
            gpu
        }
    })
    .unwrap();
    model.keep_a8(a8);
    Executor::new(model, dev.init().unwrap(), 16, 4, 0).unwrap()
}

/// The DeltaNet prep op (l2 norms, g, beta and the replay stores in one kernel; off with
/// `--delta-prep-ggml`) gives the same rows as ggml's chain, through verify steps and their
/// commits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_delta_prep_verify_steps_are_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    legion::model::core::set_delta_prep(false);
    let plain = verify_run(&mut cuda_plain(dir.path()));
    legion::model::core::set_delta_prep(true);
    let fused = verify_run(&mut cuda_plain(dir.path()));
    legion::model::core::set_delta_prep(false);
    assert_eq!(fused.len(), plain.len());
    for (i, (a, b)) in plain.iter().zip(&fused).enumerate() {
        assert!(
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "output {i} differs"
        );
    }
}

/// Weight pairs read as one matmul by small steps (off with `--no-joined-matmuls`) and the
/// shared-expert gate op (on with `--shexp-gate`) give the same rows as the separate
/// matmuls and ggml's chain, through prompt and verify steps and their commits (#256).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_joined_matmuls_and_shexp_gate_are_bit_identical() {
    use legion::model::core::{joined_matmuls_built, set_joined_matmuls, set_shexp_gate};
    let dir = tempfile::tempdir().unwrap();
    set_joined_matmuls(false);
    set_shexp_gate(false);
    // As `legion serve` builds its graphs (8-bit activation marks on every matmul).
    let plain = verify_run(&mut cuda_plain_with(dir.path(), true));
    set_joined_matmuls(true);
    set_shexp_gate(true);
    let mut e = cuda_plain_with(dir.path(), true);
    let built = joined_matmuls_built();
    let fused = verify_run(&mut e);
    // Every DeltaNet layer's alpha / beta and every MoE layer's router / gate, read by the
    // verify steps' graphs.
    assert!(e.model().joined_pairs() > 0, "nothing joined");
    assert!(
        joined_matmuls_built() > built,
        "no graph read a joined pair"
    );
    assert_eq!(fused.len(), plain.len());
    for (i, (a, b)) in plain.iter().zip(&fused).enumerate() {
        assert!(
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "output {i} differs"
        );
    }
    // Back to the defaults for the tests that follow.
    set_shexp_gate(false);
}

/// After a long prompt lends the slots and gives them back, the cache refills and learns
/// again with scoped after-steps alone (as MTP decoding runs them): each call points its own
/// layers at their refilled slots, and swaps start again. (The refill used to wait for an
/// unscoped after-step, which MTP decoding never runs: hit rate 0 after every long prompt.)
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_scoped_after_steps_refill_and_learn_after_a_restore() {
    let dir = tempfile::tempdir().unwrap();
    let n = 48;
    let (mut e, n_layers) = cuda_tiered(dir.path(), n);
    let r = e.experts_mut().unwrap();
    assert!(r.can_lend(), "the cache should be lendable");
    let cached = |r: &ExpertResidency, il: usize| {
        let n_slots = r.layer(il).unwrap().n_slots();
        r.uploaded_table(il)
            .unwrap()
            .iter()
            .filter(|&&id| (id as usize) < n_slots)
            .count()
    };
    let full: Vec<usize> = (0..n_layers).map(|il| cached(r, il)).collect();
    assert!(full.iter().all(|&c| c > 0), "{full:?}");
    r.lend().unwrap();
    r.restore().unwrap();
    let a = |il: usize| il < 2;
    let t0 = std::time::Instant::now();
    while (0..n_layers)
        .filter(|&il| a(il))
        .any(|il| cached(r, il) == 0)
    {
        assert!(t0.elapsed().as_secs() < 30, "layers A never refilled");
        std::thread::sleep(std::time::Duration::from_millis(20));
        r.after_step_only(0, a).unwrap();
    }
    for il in (0..n_layers).filter(|&il| !a(il)) {
        assert_eq!(cached(r, il), 0, "layer {il} refilled in a call for others");
    }
    r.after_step_only(0, |il| !a(il)).unwrap();
    for (il, &c) in full.iter().enumerate() {
        assert_eq!(cached(r, il), c, "layer {il}");
        assert_eq!(r.table(il), r.uploaded_table(il), "layer {il}");
    }
    // And it learns: swaps start again from scoped calls alone.
    let before = r.stats().swaps;
    r.set_swap_rule(SwapRule {
        budget: 64,
        margin: 0.0,
        ratio: 0.0,
    });
    r.after_step_only(n, |_| true).unwrap();
    assert!(r.stats().swaps > before, "no swaps after the restore");
}

/// The gated norm op (`--gated-norm`) gives the same rows as ggml's chain, through verify
/// steps and their commits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gated_norm_verify_steps_are_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    legion::model::core::set_gated_norm(false);
    let plain = verify_run(&mut cuda_plain(dir.path()));
    legion::model::core::set_gated_norm(true);
    let fused = verify_run(&mut cuda_plain(dir.path()));
    legion::model::core::set_gated_norm(false);
    assert_eq!(fused.len(), plain.len());
    for (i, (a, b)) in plain.iter().zip(&fused).enumerate() {
        assert!(
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "output {i} differs"
        );
    }
}
