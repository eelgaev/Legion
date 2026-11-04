//! Gemma 4 with tensor parallelism (#283) on two GPUs: a prompt, then decode steps of one
//! sequence through the executor on a meta device of CUDA0 and CUDA1, timed per step. For
//! profiles (nsys) of the TP decode step.
//!
//! `LEGION_TEST_GEMMA4_TP=<gguf> cargo test --release --features cuda --test gemma4_tp_cuda -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use legion::ggml::{Backend, DeviceKind, devices};
use legion::gguf::GgufFile;
use legion::model::core::{
    Attention, Executor, Group, Hparams, Logits, Model, Placement, SeqInput,
};

#[test]
#[ignore = "needs two GPUs and a Gemma 4 GGUF (LEGION_TEST_GEMMA4_TP)"]
fn tp_decode_steps() {
    let path = std::env::var("LEGION_TEST_GEMMA4_TP").expect("LEGION_TEST_GEMMA4_TP");
    let gguf = GgufFile::open(&path).unwrap();
    let hp = Hparams::from_gguf(&gguf).unwrap();
    let gpus: Vec<_> = devices()
        .into_iter()
        .filter(|d| d.kind() == DeviceKind::Gpu)
        .take(2)
        .collect();
    assert_eq!(gpus.len(), 2, "two GPUs");
    let meta =
        legion::ggml::meta::meta_device(&gpus, legion::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let backend = meta.init().unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let mut model = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
        Placement::Host => host,
        Placement::Device => backend.buffer_type(),
    })
    .unwrap();
    model.set_cell_gather(true);
    let blocks: Vec<u32> = (0..32).collect();
    let mut exec = Executor::new(model, backend, 32, 2, 0).unwrap();
    exec.set_attention(Attention::Gather);
    let prompt: Vec<u32> = (0..200).map(|i| 1000 + (i * 37) % 5000).collect();
    let steps: usize = std::env::var("LEGION_TEST_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let mut n_past = 0;
    let mut tok = prompt.clone();
    let mut times = Vec::new();
    for step in 0..=steps {
        let t = std::time::Instant::now();
        let r = exec
            .step(&[Group {
                slot: 0,
                verify: false,
                seqs: vec![SeqInput {
                    tokens: &tok,
                    n_past,
                    blocks: &blocks,
                    logits: Logits::Greedy,
                    ring: 0,
                    media: None,
                }],
            }])
            .unwrap();
        if step > 0 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        n_past += tok.len();
        tok = vec![r.argmax[0]];
    }
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{} decode steps: mean {mean:.2} ms, median {:.2} ms ({:.1} tok/s)",
        times.len(),
        times[times.len() / 2],
        1e3 / times[times.len() / 2]
    );
}

/// The engine's speculative cycle under TP, with the MTP assistant: a 4-token verify step
/// (greedy rows), its commit, then three one-token assistant passes; against verify steps
/// alone. Prints per-part times and the kept-graph re-initializations (host time spent before
/// launches when graphs alternate).
///
/// `LEGION_TEST_GEMMA4_TP_MTP=<assistant gguf>` too.
#[test]
#[ignore = "needs two GPUs, a Gemma 4 GGUF and its assistant (LEGION_TEST_GEMMA4_TP{,_MTP})"]
fn tp_verify_cycle() {
    let path = std::env::var("LEGION_TEST_GEMMA4_TP").expect("LEGION_TEST_GEMMA4_TP");
    let apath = std::env::var("LEGION_TEST_GEMMA4_TP_MTP").expect("LEGION_TEST_GEMMA4_TP_MTP");
    let gguf = GgufFile::open(&path).unwrap();
    let agguf = GgufFile::open(&apath).unwrap();
    let hp = Hparams::from_gguf(&gguf).unwrap();
    let gpus: Vec<_> = devices()
        .into_iter()
        .filter(|d| d.kind() == DeviceKind::Gpu)
        .take(2)
        .collect();
    assert_eq!(gpus.len(), 2, "two GPUs");
    let meta =
        legion::ggml::meta::meta_device(&gpus, legion::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let backend = meta.init().unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let place = |info: &legion::gguf::TensorInfo| match Model::placement(&info.name, false) {
        Placement::Host => host,
        Placement::Device => backend.buffer_type(),
    };
    let mut model = Model::load(&gguf, place).unwrap();
    model.load_mtp(&agguf, place).unwrap();
    model.set_cell_gather(true);
    let n_layer = model.hp.n_layer;
    let n_embd = model.hp.n_embd as usize;
    let blocks: Vec<u32> = (0..32).collect();
    let mut exec = Executor::new(model, backend, 32, 2, 0).unwrap();
    exec.set_attention(Attention::Gather);
    exec.enable_spec(4).unwrap();
    exec.enable_mtp().unwrap();
    exec.capture_hidden(vec![n_layer - 1]);
    let w = exec.mtp_width();
    fn seq<'a>(tokens: &'a [u32], blocks: &'a [u32], n_past: usize, logits: Logits) -> Group<'a> {
        Group {
            slot: 0,
            verify: false,
            seqs: vec![SeqInput {
                tokens,
                n_past,
                blocks,
                logits,
                ring: 0,
                media: None,
            }],
        }
    }
    let prompt: Vec<u32> = (0..200).map(|i| 1000 + (i * 37) % 5000).collect();
    let r = exec
        .step(&[seq(&prompt, &blocks, 0, Logits::Greedy)])
        .unwrap();
    let mut n_past = prompt.len();
    let mut last = r.argmax[0];
    let cycles: usize = std::env::var("LEGION_TEST_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for with_mtp in [false, true] {
        let (mut verify, mut drafts, mut whole) = (Vec::new(), Vec::new(), Vec::new());
        let (r0, t0) = legion::ggml::meta::reinit_stats();
        for c in 0..cycles {
            let t = std::time::Instant::now();
            let toks = [last, 1000, 1001, 1002];
            let mut g = seq(&toks, &blocks, n_past, Logits::FirstGreedy(4));
            g.verify = true;
            let r = exec.step(&[g]).unwrap();
            exec.commit(&[(0, 1)]).unwrap();
            n_past += 1;
            last = r.argmax[0];
            let tv = ms(t);
            let td = std::time::Instant::now();
            if with_mtp {
                let mut h = r.hidden[0][..n_embd].to_vec();
                let mut tok = last;
                for _ in 0..3 {
                    let fed = [tok];
                    let d = exec
                        .mtp_step(&[seq(&fed, &blocks, n_past, Logits::Last)], &h)
                        .unwrap();
                    tok = d.argmax[0];
                    h = d.hidden[..w].to_vec();
                }
            }
            if c > 2 {
                verify.push(tv);
                drafts.push(ms(td));
                whole.push(ms(t));
            }
        }
        let (r1, t1) = legion::ggml::meta::reinit_stats();
        println!(
            "{}: verify {:.2} ms, drafts {:.2} ms, cycle {:.2} ms (medians); {:.1} re-inits per cycle, {:.2} ms each",
            if with_mtp {
                "verify + 3 MTP passes"
            } else {
                "verify only"
            },
            median(verify),
            median(drafts),
            median(whole),
            (r1 - r0) as f64 / cycles as f64,
            (t1 - t0).as_secs_f64() * 1e3 / ((r1 - r0).max(1)) as f64,
        );
    }
}

/// Teacher-forced: the same tokens through tensor parallelism (the cell gather, flash
/// attention, all-reduces) and through the layer split (two stages, paged attention), both
/// with rings as served. Per position the next-token distributions are compared: KL, top-1
/// agreement, and at every top-1 disagreement the layer split's own margin between its top
/// two tokens (a near-tie there means rounding decides, not a bug).
///
/// Asserted for TP against the layer split as served: KL mean at most 1e-2 and top-1
/// agreement at least 97% (measured on Gemma 4 31B over 2048 tokens: 2.45e-3, 98.8%), and no
/// flip where the layer split's top token led by more than 2.0 log-prob (measured 0.92). A
/// broken split or all-reduce lands far outside these. The two control comparisons (the
/// attention kernel alone, the split alone) are printed only.
///
/// `LEGION_TEST_GEMMA4_TP=<gguf> LEGION_TEST_TP_TEXT=<file> [LEGION_TEST_TP_TOKENS=2048]`
#[test]
#[ignore = "needs two GPUs, a Gemma 4 GGUF and a text (LEGION_TEST_GEMMA4_TP, LEGION_TEST_TP_TEXT)"]
fn tp_against_layer_split() {
    use legion::model::core::KV_BLOCK;
    use legion::model::core::KvRing;
    let path = std::env::var("LEGION_TEST_GEMMA4_TP").expect("LEGION_TEST_GEMMA4_TP");
    let gguf = GgufFile::open(&path).unwrap();
    let hp = Hparams::from_gguf(&gguf).unwrap();
    let tok = legion::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(std::env::var("LEGION_TEST_TP_TEXT").unwrap()).unwrap();
    let n_tokens: usize = std::env::var("LEGION_TEST_TP_TOKENS")
        .map(|s| s.parse().expect("LEGION_TEST_TP_TOKENS: a token count"))
        .unwrap_or(2048);
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);
    const CHUNK: usize = 256;
    let window = hp.arch.swa_window();
    let n_layer = hp.n_layer;
    let blocks: Vec<u32> = (0..n_tokens.div_ceil(KV_BLOCK) as u32 + 1).collect();
    let gpus: Vec<_> = devices()
        .into_iter()
        .filter(|d| d.kind() == DeviceKind::Gpu)
        .take(2)
        .collect();
    assert_eq!(gpus.len(), 2, "two GPUs");
    let host = Backend::cpu().unwrap().buffer_type();
    // Every position's log-softmax row.
    let run = |exec: &mut Executor| -> Vec<Vec<f32>> {
        let mut rows = Vec::new();
        for from in (0..n_tokens).step_by(CHUNK) {
            let to = (from + CHUNK).min(n_tokens);
            let out = exec
                .step(&[Group {
                    slot: 0,
                    seqs: vec![SeqInput {
                        tokens: &ids[from..to],
                        n_past: from,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::All,
                        media: None,
                    }],
                    verify: false,
                }])
                .unwrap();
            for r in &out.rows {
                let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
                rows.push(r.iter().map(|&v| (v as f64 - lse) as f32).collect());
            }
        }
        rows
    };
    let ring = || Some(KvRing::for_window(window, CHUNK, 1));
    // The layer split: layers halved over the GPUs, the shared tensors in pinned host
    // memory; `cells`: attention as under TP (cell gather, flash attention), else as served
    // (paged).
    let split = |cells: bool| -> Vec<Vec<f32>> {
        let b0 = gpus[0].init().unwrap();
        let b1 = gpus[1].init().unwrap();
        let shared = gpus[0].host_buffer_type().unwrap_or(host);
        let half = n_layer / 2;
        let (t0, t1) = (b0.buffer_type(), b1.buffer_type());
        let mut model = Model::load(&gguf, |info| {
            let name = info.name.as_str();
            if Model::placement(name, false) == Placement::Host {
                return host;
            }
            if name == "token_embd.weight" || name == "rope_freqs.weight" {
                return shared;
            }
            let il = name
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|s| s.parse::<usize>().ok());
            match il {
                Some(il) if il < half => t0,
                _ => t1,
            }
        })
        .unwrap();
        model.set_cell_gather(cells);
        let mut exec = Executor::new_staged_ring(
            model,
            vec![(b0, 0..half), (b1, half..n_layer)],
            blocks.len(),
            2,
            0,
            ring(),
            Default::default(),
        )
        .unwrap();
        if cells {
            exec.set_attention(Attention::Gather);
        }
        run(&mut exec)
    };
    let split_rows = split(false);
    let cell_rows = split(true);
    let tp_rows = {
        let meta =
            legion::ggml::meta::meta_device(&gpus, legion::model::core::tp::megatron_rule(&hp, 2))
                .unwrap();
        let backend = meta.init().unwrap();
        let mut model = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            Placement::Host => host,
            Placement::Device => backend.buffer_type(),
        })
        .unwrap();
        model.set_cell_gather(true);
        let mut exec = Executor::new_staged_ring(
            model,
            vec![(backend, 0..n_layer)],
            blocks.len(),
            2,
            0,
            ring(),
            Default::default(),
        )
        .unwrap();
        exec.set_attention(Attention::Gather);
        run(&mut exec)
    };
    let top2 = |r: &[f32]| {
        let (mut a, mut b) = (0, 1);
        if r[b] > r[a] {
            std::mem::swap(&mut a, &mut b);
        }
        for i in 2..r.len() {
            if r[i] > r[a] {
                b = a;
                a = i;
            } else if r[i] > r[b] {
                b = i;
            }
        }
        (a, r[a] - r[b])
    };
    let compare = |what: &str, reference: &[Vec<f32>], other: &[Vec<f32>]| {
        let (mut kl_sum, mut kl_max, mut agree) = (0f64, 0f64, 0usize);
        let mut flips = Vec::new();
        for (i, (s, t)) in reference.iter().zip(other).enumerate() {
            let kl: f64 = s
                .iter()
                .zip(t)
                .map(|(&ls, &lt)| (ls as f64).exp() * (ls as f64 - lt as f64))
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let ((a, margin), (b, _)) = (top2(s), top2(t));
            if a == b {
                agree += 1;
            } else {
                flips.push((i, margin, kl));
            }
        }
        let n = reference.len();
        let worst = flips.iter().map(|f| f.1).fold(0.0f32, f32::max);
        println!(
            "{what}: {n} positions, KL mean {:.2e} max {:.2e}, top-1 agree {agree}/{n} \
             ({:.2}%), largest reference margin at a disagreement {worst:.4}",
            kl_sum / n as f64,
            kl_max,
            100.0 * agree as f64 / n as f64
        );
        for (i, margin, kl) in &flips {
            println!("  differs at {i}: reference margin {margin:.4} (log-prob), KL {kl:.2e}");
        }
        (kl_sum / n as f64, agree as f64 / n as f64, worst)
    };
    compare("layer split, paged vs cell gather", &split_rows, &cell_rows);
    compare("cell gather, layer split vs TP", &cell_rows, &tp_rows);
    let (kl, top1, worst) = compare("layer split paged vs TP", &split_rows, &tp_rows);
    assert!(kl <= 1e-2, "TP vs the layer split: KL mean {kl:.2e}");
    assert!(
        top1 >= 0.97,
        "TP vs the layer split: top-1 {:.2}%",
        100.0 * top1
    );
    assert!(
        worst <= 2.0,
        "TP flips a token the layer split led by {worst:.2} log-prob"
    );
}
