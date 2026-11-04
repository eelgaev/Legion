use super::*;
use crate::ggml::Backend;
use crate::gguf::{GgufWriter, Value};

fn tiny_gguf(dir: &tempfile::TempDir, tables: tiny::Tables) -> GgufFile {
    let path = dir.path().join("tiny.gguf");
    tiny::write_with(&path, 5, tables).unwrap();
    GgufFile::open(&path).unwrap()
}

/// A GGUF of `deepseek41` metadata `keys` (and `block_count` 2) plus the tensors `names`.
fn stub_gguf(dir: &tempfile::TempDir, keys: &[(&str, Value)], names: &[&str]) -> GgufFile {
    let path = dir.path().join("stub.gguf");
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String(ARCH.into()))
        .unwrap();
    g.set(&format!("{ARCH}.block_count"), &Value::U32(2))
        .unwrap();
    for (k, v) in keys {
        g.set(&format!("{ARCH}.{k}"), v).unwrap();
    }
    for name in names {
        g.add_tensor(name, types::F32, &[4], &[0; 16]).unwrap();
    }
    g.write(&path).unwrap();
    GgufFile::open(&path).unwrap()
}

fn unsupported(r: Result<Hparams, ModelError>) -> String {
    match r {
        Err(ModelError::Unsupported(s)) => s,
        Err(e) => panic!("expected an unsupported-model error, got {e}"),
        Ok(_) => panic!("expected an unsupported-model error, got a model"),
    }
}

#[test]
fn tiny_parses_with_its_sharing_structure() {
    let dir = tempfile::tempdir().unwrap();
    let hp = parse(&tiny_gguf(&dir, tiny::Tables::Inline)).unwrap();
    assert_eq!(hp.arch.name(), ARCH);
    assert_eq!(hp.n_layer, tiny::N_LAYER);
    assert_eq!(hp.n_vocab, tiny::n_vocab());
    let p = params(&hp).expect("deepseek41 hyperparameters");
    assert_eq!(p.ratios, tiny::RATIOS.map(|r| r as usize).to_vec());
    assert_eq!(p.kv_source, [false, true, false, true, false, false]);
    assert_eq!(p.index_source, [false, true, false, true, false, true]);
    assert_eq!(
        p.candidates,
        Some((
            tiny::CAND_LAYER as usize,
            tiny::CAND_BLOCK as usize,
            tiny::CAND_BLOCKS as usize
        ))
    );
    // Layer 2 reads layer 1's cache and selection, layers 4 and 5 layer 3's cache.
    assert_eq!((p.kv_of(2), p.index_of(2)), (Some(1), Some(1)));
    assert_eq!((p.kv_of(4), p.index_of(4)), (Some(3), Some(3)));
    assert_eq!((p.kv_of(5), p.index_of(5)), (Some(3), Some(5)));
    assert!(!p.masked(3) && p.masked(5));
    // Window-only layers: plain RoPE; the compressing ones YaRN on the compress base.
    assert!(!p.rope.yarn && p.rope_c.yarn);
    assert_eq!(p.rope_c.base, 160_000.0);
    assert_eq!(hp.arch.residual_streams(), tiny::HC);
    assert_eq!(hp.arch.kv_dims(&hp, 0), (tiny::HEAD, 1));
    assert_eq!(hp.arch.kv_dims(&hp, 1), (2 * tiny::HEAD + tiny::IDX_DIM, 1));
    let e = p.engram.as_ref().expect("Engram");
    assert_eq!(e.layers, [1, 3]);
    assert_eq!(e.cols(), 4);
    assert_eq!(
        hp.arch.token_rows(&hp, &(0..tiny::N_LAYER)),
        2 * 4 * tiny::ENGRAM_DIM
    );
    let conv = hp.arch.token_conv(&hp).expect("the hash history");
    assert_eq!(conv.n_hist, 2);
}

#[test]
fn a_missing_engram_tables_file_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let msg = unsupported(parse(&tiny_gguf(&dir, tiny::Tables::Missing)));
    assert!(msg.contains("blk.1.engram_embd.weight"), "{msg}");
    let expected = dir.path().join(tiny::ENGRAM_FILE);
    assert!(msg.contains(&expected.display().to_string()), "{msg}");
}

#[test]
fn engram_tables_in_their_own_gguf_parse() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, tiny::Tables::Separate);
    assert_eq!(g.shard_count(), 2);
    let t = g.tensor("blk.3.engram_embd.weight").unwrap();
    assert_eq!(t.file_index, 1);
    let hp = parse(&g).unwrap();
    assert_eq!(params(&hp).unwrap().engram.as_ref().unwrap().layers, [1, 3]);
}

/// The table rows per layer must be the converter's `engram.num_embeddings` (the sum of the
/// layer's primes), the width `engram_head_dim`.
#[test]
fn engram_table_shapes_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_with(&path, 5, tiny::Tables::Separate).unwrap();
    // Tables of another model: one row short.
    let rows = |li: usize| tiny::PRIMES[li].iter().sum::<i64>();
    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("x".into()))
        .unwrap();
    for (li, l) in tiny::ENGRAM_LAYERS.iter().enumerate() {
        let n = tiny::ENGRAM_DIM * (rows(li) - 1);
        w.add_tensor(
            &format!("blk.{l}.engram_embd.weight"),
            types::F32,
            &[tiny::ENGRAM_DIM, rows(li) - 1],
            &vec![0; n as usize * 4],
        )
        .unwrap();
    }
    w.write(dir.path().join(tiny::ENGRAM_FILE)).unwrap();
    let msg = unsupported(parse(&GgufFile::open(&path).unwrap()));
    assert!(msg.contains("blk.1.engram_embd.weight"), "{msg}");
    assert!(msg.contains(&format!("expected [8, {}]", rows(0))), "{msg}");
}

#[test]
fn the_vision_tower_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let g = stub_gguf(&dir, &[], &["v.patch_embd.weight"]);
    assert!(unsupported(parse(&g)).contains("vision"));
}

#[test]
fn other_routers_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let g = stub_gguf(&dir, &[("expert_gating_func", Value::U32(2))], &[]);
    assert!(unsupported(parse(&g)).contains("sqrt(softplus)"));
}

#[test]
fn hash_routed_layers_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let g = stub_gguf(
        &dir,
        &[
            ("expert_gating_func", Value::U32(4)),
            ("hash_layer_count", Value::U32(3)),
        ],
        &[],
    );
    assert!(unsupported(parse(&g)).contains("hash-routed"));
}

#[test]
fn engram_tables_and_hash_parameters_stay_in_host_memory() {
    for name in [
        "blk.1.engram_embd.weight",
        "blk.1.engram_embd.scale",
        "engram_token_map.weight",
        "engram_primes.weight",
        "engram_offsets.weight",
        "engram_multipliers.weight",
    ] {
        assert_eq!(
            Model::placement(name, false),
            crate::model::core::Placement::Host,
            "{name}"
        );
    }
    for name in [
        "blk.1.engram_q.weight",
        "blk.1.engram_wkv.weight",
        "token_embd.weight",
    ] {
        assert_eq!(
            Model::placement(name, false),
            crate::model::core::Placement::Device,
            "{name}"
        );
    }
}

/// The hash by hand for the tiny model's first Engram layer: multipliers 1, 1000006, 2000011,
/// token map `(t·31 + 7) mod 97`, primes 13, 17, 19, 23 at offsets 0, 13, 30, 49.
#[test]
fn engram_hash_matches_by_hand() {
    let e = EngramParams {
        layers: vec![1, 3],
        ngram: 3,
        heads: 2,
        dim: 8,
        pad: 2,
        tables: Vec::new(),
    };
    let map: Vec<i64> = (0..300).map(|t| (t * 31 + 7) % 97).collect();
    let hash = Hash {
        map: &map,
        primes: vec![13, 17, 19, 23, 29, 31, 37, 41],
        offsets: vec![0, 13, 30, 49, 0, 29, 60, 97],
        mult: vec![1, 1_000_006, 2_000_011, 1_000_010, 9, 2_000_017],
        e: &e,
    };
    let mut out = Vec::new();
    // Token 10 after 5, nothing before: the trigram reads the pad token (2) for the start.
    hash.ids(0, 10, &[NO_TOKEN, 5], &mut out);
    let (t10, t5, pad) = (26i64, 65i64, 69i64);
    let bi = t10 ^ (t5 * 1_000_006);
    let tri = bi ^ (pad * 2_000_011);
    assert_eq!(out, [bi % 13, bi % 17 + 13, tri % 19 + 30, tri % 23 + 49]);
    // A blocked predecessor blocks the older ones too.
    hash.ids(0, 10, &[7, NO_TOKEN], &mut out);
    let bi = t10 ^ (pad * 1_000_006);
    let tri = bi ^ (pad * 2_000_011);
    assert_eq!(out, [bi % 13, bi % 17 + 13, tri % 19 + 30, tri % 23 + 49]);
    // The second layer: its own multipliers and columns.
    hash.ids(1, 10, &[7, 5], &mut out);
    let t7 = (7 * 31 + 7) % 97;
    let bi = (t10 * 1_000_010) ^ (t5 * 9);
    let tri = bi ^ (t7 * 2_000_017);
    assert_eq!(out, [bi % 29, bi % 31 + 29, tri % 37 + 60, tri % 41 + 97]);
}

/// A safetensors checkpoint's FP8 table row: E4M3 values times their 32-wide E8M0 scale.
#[test]
fn fp8_engram_rows_take_their_block_scales() {
    // 1.0, -1.5 and 448 in the first block (scale 2^-3), 0.5 ... in the second (scale 2^2).
    let mut src = vec![0x38u8; 64];
    src[1] = 0xbc;
    src[2] = 0x7e;
    src[32] = 0x30;
    let v = fp8_row_f32(&src, &[124, 129, 0xff]);
    assert_eq!(v.len(), 64);
    assert_eq!(&v[..4], [0.125, -0.1875, 56.0, 0.125]);
    assert_eq!(v[32], 2.0);
    assert_eq!(v[63], 4.0);
}

#[test]
fn engram_hash_wraps_like_int64() {
    let e = EngramParams {
        layers: vec![0],
        ngram: 2,
        heads: 1,
        dim: 1,
        pad: 0,
        tables: Vec::new(),
    };
    let map = [i64::MAX / 3, 5];
    let hash = Hash {
        map: &map,
        primes: vec![1_000_003],
        offsets: vec![0],
        mult: vec![7, 11],
        e: &e,
    };
    let mut out = Vec::new();
    hash.ids(0, 0, &[1], &mut out);
    let rolling = (i64::MAX / 3).wrapping_mul(7) ^ 55;
    assert_eq!(out, [rolling.rem_euclid(1_000_003)]);
    assert!(out[0] >= 0);
}

fn backend() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b
}

fn prompt() -> Vec<u32> {
    (0..44).map(|i| (i * 37 + 11) % 250).collect()
}

/// Runs `prompt` in chunks of the sizes `chunks` (the rest one by one), every position's
/// logits.
fn chunked(m: &Model, prompt: &[u32], chunks: &[usize]) -> Vec<f32> {
    let mut b = backend();
    let mut st = m.new_state(b.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut out = Vec::new();
    let mut at = 0;
    for &c in chunks {
        out.extend(
            m.forward(&mut b, &mut alloc, &mut st, &prompt[at..at + c], true)
                .unwrap(),
        );
        at += c;
    }
    for &t in &prompt[at..] {
        out.extend(m.forward(&mut b, &mut alloc, &mut st, &[t], true).unwrap());
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The whole graph on the CPU: one prompt, the same prompt in uneven chunks and token by token
/// give the same logits up to rounding (the caches, compression, selections and Engram history
/// carried between steps), finite and deterministic.
#[test]
fn tiny_runs_end_to_end_the_same_in_any_chunking() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, tiny::Tables::Inline);
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let p = prompt();
    let once = m.prompt_logits_all(&mut b, &p).unwrap();
    let nv = tiny::n_vocab() as usize;
    assert_eq!(once.len(), p.len() * nv);
    assert!(once.iter().all(|v| v.is_finite()));
    let scale = once.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(scale > 0.1, "logits of a real computation ({scale})");
    // Positions differ: the model sees its context.
    assert!(max_diff(&once[..nv], &once[nv..2 * nv]) > 1e-3);

    let again = m.prompt_logits_all(&mut b, &p).unwrap();
    assert_eq!(once, again, "deterministic on the CPU");

    let tol = 2e-4 * scale;
    let split = chunked(&m, &p, &[13, 7, 16]);
    let d = max_diff(&once, &split);
    assert!(d <= tol, "chunked prompt: {d} > {tol}");
    let steps = chunked(&m, &p, &[1]);
    let d = max_diff(&once, &steps);
    assert!(d <= tol, "token by token: {d} > {tol}");
}

/// The prompt reaches past the window and the selection: the indexers score every visible
/// group, and after the candidate layer the blocks outside the kept ones are masked out.
#[test]
fn tiny_prompt_exercises_selection_and_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, tiny::Tables::Inline);
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let p = prompt();
    let (_, trace) = m.prompt_trace(&mut b, &p).unwrap();
    let scores = |il: usize| {
        trace
            .iter()
            .find(|t| t.name == format!("index_score-{il}"))
            .unwrap_or_else(|| panic!("no index scores of layer {il}"))
    };
    let last = p.len() - 1;
    for il in [1, 3, 5] {
        let t = scores(il);
        let nc = t.ne[0] as usize;
        assert_eq!(t.ne[1] as usize, p.len());
        let row = &t.data[last * nc..(last + 1) * nc];
        let visible = p.len() / tiny::RATIOS[il] as usize;
        assert!(row[visible..].iter().all(|v| *v == f32::NEG_INFINITY));
        let finite = row[..visible].iter().filter(|v| v.is_finite()).count();
        if il == 5 {
            // Masked: the kept candidate blocks only.
            let kept = (tiny::CAND_BLOCK * tiny::CAND_BLOCKS) as usize;
            assert_eq!(finite, kept, "layer {il}");
        } else {
            assert_eq!(finite, visible, "layer {il}");
        }
        assert!(visible > tiny::TOP_K as usize);
    }
}

/// Only a kv source pooling groups (layer 1, ratio 2) keeps V; every other layer's cache is K
/// only, its V an alias of K's memory, and the cache bytes count K alone.
#[test]
fn tiny_caches_keep_v_for_pooling_sources_only() {
    use crate::model::core::KvTypes;
    let dir = tempfile::tempdir().unwrap();
    let hp = Hparams::from_gguf(&tiny_gguf(&dir, tiny::Tables::Inline)).unwrap();
    let with_v: Vec<usize> = (0..hp.n_layer).filter(|&il| hp.kv_has_v(il)).collect();
    assert_eq!(with_v, [1]);
    let kv = KvPool::new(&hp, backend().buffer_type(), 2).unwrap();
    let f16 = KvTypes::default();
    for il in 0..hp.n_layer {
        let (d, _) = hp.kv_dims(il);
        let k_bytes = f16.k.bytes(d as usize, d as usize);
        let want = if il == 1 { 2 * k_bytes } else { k_bytes };
        assert_eq!(hp.kv_cell_bytes(il, f16), want, "layer {il}");
        let (k, v) = kv.pools(il);
        // SAFETY: allocated tensors; only their addresses are read.
        let same = unsafe { (*k.raw()).data == (*v.raw()).data };
        assert_eq!(same, il != 1, "layer {il}");
        assert_eq!(k.nbytes(), v.nbytes(), "layer {il}");
    }
    assert_eq!(kv.bytes_per_block(), KvPool::block_bytes(&hp));
}

/// Another seed's weights give other logits (the weights reach the output).
#[test]
fn tiny_logits_depend_on_the_weights() {
    let dir = tempfile::tempdir().unwrap();
    let mut b = backend();
    let p: Vec<u32> = prompt()[..12].to_vec();
    let a = {
        let m = Model::load(&tiny_gguf(&dir, tiny::Tables::Inline), |_| b.buffer_type()).unwrap();
        m.prompt_logits(&mut b, &p).unwrap()
    };
    let path = dir.path().join("other.gguf");
    tiny::write(&path, 6).unwrap();
    let g = GgufFile::open(&path).unwrap();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let c = m.prompt_logits(&mut b, &p).unwrap();
    assert!(max_diff(&a, &c) > 1e-3);
}

/// The Engram tables in a GGUF of their own (as the real model ships them) give the logits of
/// the same model in one file, bit for bit.
#[test]
fn tiny_engram_tables_from_their_own_gguf_match_one_file() {
    let mut b = backend();
    let p: Vec<u32> = prompt()[..20].to_vec();
    let mut logits = |tables| {
        let dir = tempfile::tempdir().unwrap();
        let g = tiny_gguf(&dir, tables);
        let m = Model::load(&g, |_| b.buffer_type()).unwrap();
        (g.shard_count(), m.prompt_logits_all(&mut b, &p).unwrap())
    };
    let (one, inline) = logits(tiny::Tables::Inline);
    let (two, separate) = logits(tiny::Tables::Separate);
    assert_eq!((one, two), (1, 2));
    assert!(inline.iter().all(|v| v.is_finite()));
    assert_eq!(inline, separate);
}

/// The layers that are not kv sources read their window only, so their caches are rings: the
/// tiny model's layers 0, 2, 4 and 5 are, the sources 1 and 3 are not.
#[test]
fn tiny_window_layers_are_sliding() {
    let dir = tempfile::tempdir().unwrap();
    let hp = Hparams::from_gguf(&tiny_gguf(&dir, tiny::Tables::Inline)).unwrap();
    assert_eq!(hp.arch.swa_window(), tiny::WINDOW as usize);
    let sliding: Vec<usize> = (0..hp.n_layer).filter(|&il| hp.sliding_layer(il)).collect();
    assert_eq!(sliding, [0, 2, 4, 5]);
}

/// The engine gives the window layers rings only where prefix caching stays (the token history
/// is a per-sequence state, which rings leave without a cached prefix): with `--swa-ring`, or
/// prefix caching off; never with `--swa-full`. With any `--attention`: the ops read in place.
#[test]
fn tiny_rings_need_swa_ring_or_no_prefix_cache() {
    use crate::engine::lm::{DeviceChoice, LmConfig, use_rings};
    let dir = tempfile::tempdir().unwrap();
    let hp = Hparams::from_gguf(&tiny_gguf(&dir, tiny::Tables::Inline)).unwrap();
    assert!(hp.has_slot_state());
    let mut cfg = LmConfig::new("unused", DeviceChoice::Cuda);
    assert!(cfg.prefix_cache_states > 0);
    assert!(!use_rings(&hp, &cfg, false));
    cfg.swa_ring = true;
    assert!(use_rings(&hp, &cfg, false));
    cfg.swa_full = true;
    assert!(!use_rings(&hp, &cfg, false));
    cfg.swa_full = false;
    cfg.swa_ring = false;
    cfg.prefix_cache_states = 0;
    assert!(use_rings(&hp, &cfg, false));
}

/// Batch steps through the paged ops with the window layers' caches in rings give the tokens of
/// the same steps over whole caches: two sequences, prompts in chunks and decode steps, one of
/// them far past a ring's length (its window wraps around the ring several times).
#[test]
fn tiny_rings_give_the_whole_cache_tokens() {
    use crate::model::core::state::KvRing;
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, tiny::Tables::Inline);
    const STEP: usize = 16;
    let run = |ring: bool| -> Vec<u32> {
        let b = backend();
        let m = Model::load(&g, |_| b.buffer_type()).unwrap();
        let n_layer = m.hp.n_layer;
        // 2 blocks: 128 positions per ring.
        let ring = ring.then(|| KvRing::for_window(m.hp.arch.swa_window(), STEP, 2));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 16, 3, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Paged);
        let blocks: Vec<Vec<u32>> = vec![(0..8).collect(), (8..16).collect()];
        let mut n_past = [0usize; 2];
        let mut out = Vec::new();
        let mut step = |exec: &mut Executor, seqs: &[(usize, Vec<u32>)]| -> Vec<u32> {
            let groups: Vec<Group> = seqs
                .iter()
                .map(|(s, toks)| Group {
                    slot: *s,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past: n_past[*s],
                        blocks: &blocks[*s],
                        logits: Logits::Greedy,
                        ring: *s,
                        media: None,
                    }],
                })
                .collect();
            let r = exec.step(&groups).unwrap();
            for (s, toks) in seqs {
                n_past[*s] += toks.len();
            }
            r.argmax
        };
        let long: Vec<u32> = (0..300).map(|i| (i * 37 + 11) % 250).collect();
        let short: Vec<u32> = prompt()[..30].to_vec();
        let mut last = [0u32; 2];
        for (s, p) in [long, short].iter().enumerate() {
            for c in p.chunks(STEP) {
                let r = step(&mut exec, &[(s, c.to_vec())]);
                last[s] = r[0];
                out.extend(r);
            }
        }
        for _ in 0..4 {
            let r = step(&mut exec, &[(0, vec![last[0]]), (1, vec![last[1]])]);
            last = [r[0], r[1]];
            out.extend(r);
        }
        out
    };
    let whole = run(false);
    assert_eq!(whole, run(true), "rings change the tokens");
}

/// The whole tiny graph on the GPU (the CUDA ops, the GLM dense kernels and the fused expert
/// op) gives the CPU's intermediates and logits up to rounding, in one prompt and token by
/// token, up to the first query whose indexer selection rounding flips (a near tie).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_matches_the_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, tiny::Tables::Inline);
    let p = prompt();
    let mut b = backend();
    let (cpu, cpu_trace) = Model::load(&g, |_| b.buffer_type())
        .unwrap()
        .prompt_trace(&mut b, &p)
        .unwrap();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    // The Engram tables and hash parameters stay in host memory, as the engine places them.
    let host = Backend::cpu().unwrap().buffer_type();
    let m = Model::load(&g, |info| match Model::placement(&info.name, false) {
        crate::model::core::Placement::Host => host,
        crate::model::core::Placement::Device => gpu.buffer_type(),
    })
    .unwrap();
    let (once, gpu_trace) = m.prompt_trace(&mut gpu, &p).unwrap();
    // An indexer selection is per query, so where rounding flips a near tie (the tiny model's
    // relu scores have several within the tolerance) only that query and the later ones
    // follow another, equally valid, selection. Every selection apart must be a top-k of the
    // CPU's scores up to the tolerance; everything before the first query apart must agree.
    let n = p.len();
    let finite_scale = |v: &[f32]| {
        v.iter()
            .filter(|v| v.is_finite())
            .fold(1e-3f32, |a, v| a.max(v.abs()))
    };
    let mut until = n;
    for (c, g) in cpu_trace.iter().zip(&gpu_trace) {
        assert_eq!(c.name, g.name);
        let Some(layer) = c.name.strip_prefix("sel-") else {
            continue;
        };
        let k = c.ne[0] as usize;
        let score = cpu_trace
            .iter()
            .find(|t| t.name == format!("index_score-{layer}"))
            .expect("the selection's scores");
        let nc = score.ne[0] as usize;
        let tol = 2e-3 * finite_scale(&score.data);
        for q in 0..until {
            let (cs, gs) = (&c.data[q * k..][..k], &g.data[q * k..][..k]);
            if cs == gs {
                continue;
            }
            let row = &score.data[q * nc..][..nc];
            let mut best: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
            best.sort_by(|a, b| b.total_cmp(a));
            let kth = best
                .get(k.min(best.len()).saturating_sub(1))
                .copied()
                .unwrap_or(f32::INFINITY);
            assert!(
                gs.iter()
                    .filter(|&&j| j >= 0.0)
                    .all(|&j| row[j as usize] >= kth - tol)
                    && gs.iter().filter(|&&j| j >= 0.0).count()
                        == cs.iter().filter(|&&j| j >= 0.0).count(),
                "{}: query {q} selects {gs:?} on the GPU, {cs:?} on the CPU: not a near tie of {row:?}",
                c.name
            );
            until = q;
            break;
        }
    }
    // Past the window, with more compressed positions than the selection takes.
    assert!(
        until > tiny::WINDOW as usize,
        "a selection apart already at query {until}"
    );
    // Rows of the queries before `until`: the token dimension is the last one of size n.
    let rows = |ne: &[i64; 4], v: &[f32]| -> usize {
        let td = (0..4)
            .rev()
            .find(|&i| ne[i] == n as i64)
            .expect("a token dim");
        let per: i64 = ne[..td].iter().product();
        assert_eq!(v.len() as i64, per * n as i64);
        per as usize * until
    };
    // Equal infinities (masked scores) agree; the scale is over the finite values. The bounds
    // are rounding: the Q8_0 experts' activations are quantized differently by the GPU's fused
    // expert op and by ggml's CPU dot products (about 0.3% of the scale by the last layers),
    // and the relu index scores sum it over heads.
    let apart: Vec<String> = cpu_trace
        .iter()
        .zip(&gpu_trace)
        .filter_map(|(c, g)| {
            let len = rows(&c.ne, &c.data);
            let scale = finite_scale(&c.data[..len]);
            let (i, d) = c.data[..len]
                .iter()
                .zip(&g.data)
                .map(|(x, y)| if x == y { 0.0 } else { (x - y).abs() })
                .enumerate()
                .fold((0, 0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
            (d > 1e-2 * scale).then(|| {
                format!(
                    "{} {:?}: {d} of {scale} at {i} (cpu {}, gpu {})",
                    c.name, c.ne, c.data[i], g.data[i]
                )
            })
        })
        .collect();
    let len = cpu.len() / n * until;
    let scale = cpu[..len].iter().fold(0f32, |a, v| a.max(v.abs()));
    let tol = 5e-3 * scale;
    let d = max_diff(&once[..len], &cpu[..len]);
    assert!(
        d <= tol && apart.is_empty(),
        "one prompt on the GPU, {until} of {n} queries: {d} > {tol}, apart: {apart:#?}"
    );
    let mut st = m.new_state(gpu.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut steps = Vec::new();
    for &t in &p[..until] {
        steps.extend(
            m.forward(&mut gpu, &mut alloc, &mut st, &[t], true)
                .unwrap(),
        );
    }
    let d = max_diff(&steps, &cpu[..len]);
    assert!(
        d <= tol,
        "token by token on the GPU, {until} queries: {d} > {tol}"
    );
}
