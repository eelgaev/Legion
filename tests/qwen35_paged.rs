//! Paged attention (`Attention::Paged`) against the gather path (`Attention::Gather`) on the
//! real model, through the batched executor: prefill chunks next to decode steps, sequences on
//! interleaved KV blocks, contexts of several blocks (so decode splits and the prefill kernel's
//! multi-block walk are exercised).
//!
//! Ignored: needs the real model. CPU: `LEGION_TEST_QWEN35=<model.gguf> cargo test --release
//! --test qwen35_paged -- --ignored --nocapture`. GPU (`--features cuda`):
//! `LEGION_TEST_QWEN35_CUDA=<model.gguf>`, which also runs `cuda_step_times`, a per-step timing
//! table of both paths.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Attention, Executor, Group, KV_BLOCK, Logits, Model, SeqInput};
use legion::tokenizer::Tokenizer;

const TEXT: &str = "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the lunar \
    module Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled the \
    goal President Kennedy had set eight years earlier, and the next mission";

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

/// (max |diff|, top-1 agreements, rows, mean KL, max KL) of `ours` against `reference`.
fn agreement(ours: &[Vec<f32>], reference: &[Vec<f32>]) -> (f32, usize, usize, f64, f64) {
    assert_eq!(ours.len(), reference.len());
    let (mut diff, mut top1, mut kl_sum, mut kl_max) = (0f32, 0, 0f64, 0f64);
    for (o, r) in ours.iter().zip(reference) {
        for (x, y) in o.iter().zip(r) {
            diff = diff.max((x - y).abs());
        }
        top1 += (argmax(o) == argmax(r)) as usize;
        let (lo, lr) = (log_softmax(o), log_softmax(r));
        let kl: f64 = lr.iter().zip(&lo).map(|(r, o)| r.exp() * (r - o)).sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
    }
    (diff, top1, ours.len(), kl_sum / ours.len() as f64, kl_max)
}

/// Prompts: a long one (~300 tokens, five blocks, prefilled in two chunks), and two short.
fn prompts(tok: &Tokenizer) -> Vec<Vec<u32>> {
    let long = [TEXT; 6].join(" ");
    [
        long.as_str(),
        "The capital of France is",
        "Write a haiku about the sea.",
    ]
    .iter()
    .map(|t| tok.encode(t, true))
    .collect()
}

/// The scenario, on `exec`: step 1 prefills prompt 1 whole and the first half of prompt 0;
/// step 2 decodes 1 next to prompt 0's second half; step 3 decodes 0 and 1 next to prompt 2;
/// then `steps` three-sequence decode steps. Tokens fed back are `forced` when given (so two
/// runs see the same inputs), else greedy. Returns every sequence's logits rows and tokens.
#[allow(clippy::type_complexity)]
fn scenario(
    exec: &mut Executor,
    prompts: &[Vec<u32>],
    steps: usize,
    forced: Option<&[Vec<u32>]>,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>) {
    // Sequence i owns blocks i, i + 3, i + 6, ... (interleaved).
    let blocks: Vec<Vec<u32>> = (0..3u32)
        .map(|i| (0..16u32).map(|b| i + 3 * b).collect())
        .collect();
    assert!(prompts.iter().all(|p| p.len() + steps + 1 < 16 * KV_BLOCK));
    let mut rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 3];
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); 3];
    let mut n_past = [0usize; 3];
    let half = prompts[0].len() / 2;

    let chunk = |i: usize, from: usize, to: usize| SeqInput {
        tokens: &prompts[i][from..to],
        n_past: from,
        blocks: &blocks[i],
        ring: 0,
        logits: if to == prompts[i].len() {
            Logits::Last
        } else {
            Logits::None
        },
        media: None,
    };
    let out = exec
        .step(&[
            Group {
                slot: 0,
                seqs: vec![chunk(1, 0, prompts[1].len())],
                verify: false,
            },
            Group {
                slot: 1,
                seqs: vec![chunk(0, 0, half)],
                verify: false,
            },
        ])
        .unwrap();
    rows[1].push(out.rows[0].clone());
    n_past[1] = prompts[1].len();
    n_past[0] = half;

    let next = |rows: &Vec<Vec<Vec<f32>>>, tokens: &mut Vec<Vec<u32>>, i: usize| -> u32 {
        let k = tokens[i].len();
        let t = match forced {
            Some(f) => f[i][k],
            None => argmax(rows[i].last().unwrap()) as u32,
        };
        tokens[i].push(t);
        t
    };
    for s in 0..steps + 2 {
        // Running sequences, in slot order: 1 (slot 0), then 0, then 2.
        let running: Vec<usize> = match s {
            0 => vec![1],
            1 => vec![1, 0],
            _ => vec![1, 0, 2],
        };
        let fed: Vec<u32> = running
            .iter()
            .map(|&i| next(&rows, &mut tokens, i))
            .collect();
        let mut groups = vec![Group {
            slot: 0,
            seqs: running
                .iter()
                .zip(&fed)
                .map(|(&i, t)| SeqInput {
                    tokens: std::slice::from_ref(t),
                    n_past: n_past[i],
                    blocks: &blocks[i],
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                })
                .collect(),
            verify: false,
        }];
        match s {
            0 => groups.push(Group {
                slot: 1,
                seqs: vec![chunk(0, half, prompts[0].len())],
                verify: false,
            }),
            1 => groups.push(Group {
                slot: 2,
                seqs: vec![chunk(2, 0, prompts[2].len())],
                verify: false,
            }),
            _ => {}
        }
        let out = exec.step(&groups).unwrap();
        for (k, &i) in running.iter().enumerate() {
            rows[i].push(out.rows[k].clone());
            n_past[i] += 1;
        }
        match s {
            // Prompt 0 joins the decode slots next to 1. The test moves no states: prompt 0
            // prefilled in slot 1, which is where it decodes.
            0 => {
                rows[0].push(out.rows[1].clone());
                n_past[0] = prompts[0].len();
            }
            1 => {
                rows[2].push(out.rows[2].clone());
                n_past[2] = prompts[2].len();
            }
            _ => {}
        }
    }
    (rows, tokens)
}

const STEPS: usize = 12;

/// Runs the scenario with gather, then paged attention, on executors made by `make`; prints
/// and returns the paged run's agreement with the gather run, per sequence. Also returns both
/// runs and the tokens fed.
#[allow(clippy::type_complexity)]
fn compare(
    make: &mut dyn FnMut() -> Executor,
    prompts: &[Vec<u32>],
    label: &str,
) -> (
    Vec<(f32, usize, usize, f64, f64)>,
    [Vec<Vec<Vec<f32>>>; 2],
    Vec<Vec<u32>>,
) {
    let mut exec = make();
    exec.set_attention(Attention::Gather);
    let (reference, tokens) = scenario(&mut exec, prompts, STEPS, None);
    drop(exec);
    let mut exec = make();
    exec.set_attention(Attention::Paged);
    assert!(exec.paged());
    let (paged, _) = scenario(&mut exec, prompts, STEPS, Some(&tokens));
    let agree = (0..3)
        .map(|i| {
            let a = agreement(&paged[i], &reference[i]);
            println!(
                "{label} paged vs gather, seq {i}: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                a.0, a.1, a.2, a.3, a.4
            );
            a
        })
        .collect();
    (agree, [reference, paged], tokens)
}

/// On the CPU (reference kernels), paged attention's CPU implementation rounds like the
/// gather path's F16 matmuls and differs only in summation order (below 1e-5 per attention
/// output, see the unit tests in src/model/core/paged). Q8_0 activation quantization then amplifies that over the layers as it does any
/// rounding change (see tests/qwen35.rs), so the check is statistical. Measured: KL mean
/// 3e-5..6e-4, one top-1 flip in 42 rows.
#[test]
#[ignore = "needs the real model"]
fn paged_matches_gather_cpu() {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN35")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let prompts = prompts(&tok);
    let mut make = || {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        cpu.set_use_ref(true);
        let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
        Executor::new(model, cpu, 48, 3, 0).unwrap()
    };
    for (i, a) in compare(&mut make, &prompts, "CPU")
        .0
        .into_iter()
        .enumerate()
    {
        assert!(a.1 * 10 >= a.2 * 9, "seq {i}: top-1 {}/{}", a.1, a.2);
        assert!(a.3 < 2e-3, "seq {i}: mean KL {}", a.3);
        assert!(a.4 < 2e-2, "seq {i}: max KL {}", a.4);
    }
}

/// On CUDA: the paged kernels (flash-decoding for decode groups, tensor cores for prefill
/// chunks; F32 accumulation) against the gather path's mul_mat / soft_max. Different kernels and
/// rounding (cuBLAS takes F16 operands), so statistical: KL at the 1e-4 level, as far from the
/// gather path as that is from the CPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_paged_matches_gather() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN35_CUDA")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let prompts = prompts(&tok);
    let mut make = || {
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU device")
            .init()
            .unwrap();
        let model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
        Executor::new(model, gpu, 48, 3, 0).unwrap()
    };
    let (agree, runs, tokens) = compare(&mut make, &prompts, "CUDA");
    // Both CUDA paths against the CPU (gather, default kernels) on the same tokens: the paged
    // path is no further from it than the gather path. Printed before any bound is asserted.
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let mut exec = Executor::new(model, cpu, 48, 3, 0).unwrap();
    let (on_cpu, _) = scenario(&mut exec, &prompts, STEPS, Some(&tokens));
    let mut kl = [0f64; 2];
    for (k, name) in ["gather", "paged"].iter().enumerate() {
        for i in 0..3 {
            let a = agreement(&runs[k][i], &on_cpu[i]);
            kl[k] += a.3 / 3.0;
            println!(
                "CUDA {name} vs CPU, seq {i}: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                a.0, a.1, a.2, a.3, a.4
            );
        }
    }
    for (i, a) in agree.into_iter().enumerate() {
        assert!(a.1 * 10 >= a.2 * 9, "seq {i}: top-1 {}/{}", a.1, a.2);
        assert!(a.3 < 2e-3, "seq {i}: mean KL {}", a.3);
        assert!(a.4 < 2e-2, "seq {i}: max KL {}", a.4);
    }
    assert!(
        kl[1] < 2.0 * kl[0] + 1e-4,
        "paged KL {} vs gather KL {}",
        kl[1],
        kl[0]
    );
}

/// A long prompt on CUDA, teacher-forced (#520): paged attention against the gather path over
/// `LEGION_TEST_PAGED_TOKENS` (default 16384) tokens of the text in `LEGION_TEST_PAGED_TEXT`,
/// prefilled in chunks of `LEGION_TEST_PAGED_CHUNK` (default 2048) tokens whose last 64 positions
/// are scored (a step of their own, so only those rows come back). On one load: gather twice
/// (its run-to-run floor), then paged twice. Prints, against the first gather run, the rows that
/// are bit-identical, max |logit diff|, top-1 agreement and KL; asserts only that each path
/// repeats its own bits. On a BF16 checkpoint the two paths agree closely (KL ~5e-5 over 16K
/// tokens); a quantized one amplifies the last-bit attention differences into larger logit
/// differences (KL ~2e-2 on NVFP4), so there the measure is each path's distance from the BF16
/// reference (`tests/checkpoint_kl.rs`, #530).
///
/// `LEGION_TEST_QWEN35_CUDA=<model.gguf or checkpoint dir> LEGION_TEST_PAGED_TEXT=<file>
///  cargo test --release --features cuda --test qwen35_paged cuda_long_prompt_kl -- --ignored --nocapture`
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_long_prompt_kl() {
    use legion::ggml::{DeviceKind, device_by_kind};
    const SCORED: usize = 64;
    let num = |k: &str, d: usize| std::env::var(k).map_or(d, |s| s.parse().unwrap());
    let n_tokens = num("LEGION_TEST_PAGED_TOKENS", 16384);
    let chunk = num("LEGION_TEST_PAGED_CHUNK", 2048);
    assert!(chunk > SCORED, "chunks of more than {SCORED} tokens");
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN35_CUDA")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = std::fs::read_to_string(env("LEGION_TEST_PAGED_TEXT")).unwrap();
    let mut ids = tok.encode(&text, true);
    assert!(ids.len() >= n_tokens, "the text has {} tokens", ids.len());
    ids.truncate(n_tokens);

    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let blocks: Vec<u32> = (0..n_tokens.div_ceil(KV_BLOCK) as u32).collect();
    let mut exec = Executor::new(model, gpu, blocks.len(), 1, 0).unwrap();
    let mut run = |a: Attention| -> Vec<Vec<f32>> {
        exec.set_attention(a);
        assert_eq!(exec.paged(), a == Attention::Paged, "{a:?}");
        exec.clear_slot(0);
        let mut rows = Vec::new();
        for from in (0..n_tokens).step_by(chunk) {
            let to = (from + chunk).min(n_tokens);
            let mid = to.saturating_sub(SCORED).max(from);
            for (lo, hi, logits) in [(from, mid, Logits::None), (mid, to, Logits::All)] {
                if lo == hi {
                    continue;
                }
                let out = exec
                    .step(&[Group {
                        slot: 0,
                        seqs: vec![SeqInput {
                            tokens: &ids[lo..hi],
                            n_past: lo,
                            blocks: &blocks,
                            ring: 0,
                            logits,
                            media: None,
                        }],
                        verify: false,
                    }])
                    .unwrap();
                rows.extend(out.rows);
            }
        }
        rows
    };
    let gather = run(Attention::Gather);
    let gather2 = run(Attention::Gather);
    let paged = run(Attention::Paged);
    let paged2 = run(Attention::Paged);
    let same = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).filter(|(x, y)| x == y).count();
    for (label, rows) in [
        ("gather again", &gather2),
        ("paged", &paged),
        ("paged again", &paged2),
    ] {
        let a = agreement(rows, &gather);
        println!(
            "{label} vs gather, {n_tokens} tokens in chunks of {chunk}: rows bit-identical {}/{}, \
             max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
            same(rows, &gather),
            rows.len(),
            a.0,
            a.1,
            a.2,
            a.3,
            a.4
        );
    }
    assert_eq!(
        same(&gather2, &gather),
        gather.len(),
        "gather repeats its bits"
    );
    assert_eq!(same(&paged2, &paged), paged.len(), "paged repeats its bits");
}

/// Per-step wall time, gather vs paged, on the same KV contents: decode steps (batch `b`, every
/// sequence `ctx` tokens in); mixed steps (that decode batch plus a 512-token prefill chunk at
/// alternating depths, so the gather path's shape changes every step as it does while a
/// prompt prefills; and once more with alternating chunk lengths, so the paged graph is rebuilt
/// every step too); prefill chunks alone. Prints a table; asserts nothing.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_step_times() {
    use legion::ggml::{DeviceKind, device_by_kind};
    use std::time::Instant;
    const CHUNK: usize = 512;
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN35_CUDA")).unwrap();
    let gpu = device_by_kind(DeviceKind::Gpu).unwrap().init().unwrap();
    let model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as u32;
    // 160k positions: room for b × ctx plus the prefilling sequence.
    let total_blocks = 2600;
    let mut exec = Executor::new(model, gpu, total_blocks, 129 + 1, 0).unwrap();
    exec.set_max_seq_tokens(8192);
    println!(
        "batch  ctx | decode ms: gather  paged | mixed ms: gather  paged  paged, new shape | \
         prefill ms: gather  paged"
    );
    let configs: [(usize, usize); 7] = [
        (1, 4096),
        (8, 4096),
        (32, 1024),
        (32, 4096),
        (64, 1024),
        (128, 512),
        (128, 1024),
    ];
    for (b, ctx) in configs {
        let per_seq = (ctx + 1).div_ceil(KV_BLOCK);
        let pre_blocks = (2 * CHUNK + ctx).div_ceil(KV_BLOCK);
        assert!(b * per_seq + pre_blocks <= total_blocks);
        let blocks: Vec<Vec<u32>> = (0..b)
            .map(|i| (0..per_seq).map(|k| (i * per_seq + k) as u32).collect())
            .collect();
        let pre: Vec<u32> = (0..pre_blocks).map(|k| (b * per_seq + k) as u32).collect();
        let tokens: Vec<u32> = (0..ctx + 2 * CHUNK)
            .map(|i| (i as u32 * 7919 + 13) % n_vocab)
            .collect();
        // Fill every sequence's KV (contents only matter for being finite).
        exec.set_attention(Attention::Paged);
        for (i, bl) in blocks.iter().enumerate() {
            for from in (0..ctx).step_by(CHUNK) {
                let to = (from + CHUNK).min(ctx);
                exec.step(&[Group {
                    slot: i,
                    seqs: vec![SeqInput {
                        tokens: &tokens[from..to],
                        n_past: from,
                        blocks: bl,
                        ring: 0,
                        logits: Logits::None,
                        media: None,
                    }],
                    verify: false,
                }])
                .unwrap();
            }
        }
        let one = [tokens[0]];
        let decode = Group {
            slot: 0,
            seqs: blocks
                .iter()
                .map(|bl| SeqInput {
                    tokens: &one,
                    n_past: ctx,
                    blocks: bl,
                    ring: 0,
                    logits: Logits::Greedy,
                    media: None,
                })
                .collect(),
            verify: false,
        };
        let prefill = |at: usize, n: usize| Group {
            slot: b,
            seqs: vec![SeqInput {
                tokens: &tokens[at..at + n],
                n_past: at,
                blocks: &pre,
                ring: 0,
                logits: Logits::None,
                media: None,
            }],
            verify: false,
        };
        // Chunks at alternating depths; "new shape" also alternates the chunk length, so even
        // the paged graph is rebuilt every step.
        let at = |k: usize| {
            if k.is_multiple_of(2) {
                ctx - CHUNK.min(ctx)
            } else {
                ctx
            }
        };
        let len = |k: usize, vary: bool| if vary && k % 2 == 1 { CHUNK - 1 } else { CHUNK };
        let mut row = format!("{b:5} {ctx:4} |");
        let runs: [(&str, &[Attention], usize); 4] = [
            ("decode", &[Attention::Gather, Attention::Paged], 30),
            ("mixed", &[Attention::Gather, Attention::Paged], 10),
            ("mixed-vary", &[Attention::Paged], 10),
            ("prefill", &[Attention::Gather, Attention::Paged], 10),
        ];
        for (kind, attns, n) in runs {
            row += if kind == "mixed-vary" {
                "  "
            } else {
                "           "
            };
            for &attn in attns {
                exec.set_attention(attn);
                let run = |exec: &mut Executor, k: usize| {
                    let groups = match kind {
                        "decode" => vec![decode.clone()],
                        "mixed" => vec![decode.clone(), prefill(at(k), len(k, false))],
                        "mixed-vary" => vec![decode.clone(), prefill(at(k), len(k, true))],
                        _ => vec![prefill(at(k), CHUNK)],
                    };
                    exec.step(&groups).unwrap();
                };
                for k in 0..3 {
                    run(&mut exec, k);
                }
                let t = Instant::now();
                for k in 0..n {
                    run(&mut exec, k);
                }
                row += &format!(" {:6.2}", t.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
            if kind != "mixed" {
                row += " |";
            }
        }
        println!("{row}");
    }
}
