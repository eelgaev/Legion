//! Speculative decoding core on the CPU with the tiny random model and reference kernels:
//! verify steps plus recurrent-state commits reproduce plain greedy decoding bit for bit, for
//! oracle, adversarial, random and partly right drafters, with several sequences of different
//! draft lengths in one batch and a prefill chunk in the same step.

use legion::engine::sampling::Rng;
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Group, KV_BLOCK, Logits, Model, SeqInput};
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::TokenDrafter;
use legion::spec::accept::{self, Dist, Target};

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

fn tiny_model(dir: &tempfile::TempDir) -> GgufFile {
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    GgufFile::open(&path).unwrap()
}

fn argmax(x: &[f32]) -> u32 {
    legion::engine::sampling::argmax(x) as u32
}

/// Plain greedy decode of `prompt` for `n` tokens through `Model::forward`: the generated
/// tokens and the logits row that picked each (row `j`: after the prompt and `j` tokens).
fn plain(
    model: &Model,
    backend: &mut Backend,
    prompt: &[u32],
    n: usize,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut state = model.new_state(backend.buffer_type(), 1024).unwrap();
    let mut alloc = None;
    let mut rows = vec![
        model
            .forward(backend, &mut alloc, &mut state, prompt, false)
            .unwrap(),
    ];
    let mut tokens = Vec::new();
    for _ in 0..n {
        let t = argmax(rows.last().unwrap());
        tokens.push(t);
        rows.push(
            model
                .forward(backend, &mut alloc, &mut state, &[t], false)
                .unwrap(),
        );
    }
    rows.pop();
    (tokens, rows)
}

/// Proposes the true continuation.
struct Oracle {
    prompt_len: usize,
    truth: Vec<u32>,
}

impl TokenDrafter for Oracle {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        let g = tokens.len() - self.prompt_len;
        out.extend(self.truth.iter().skip(g).take(max));
    }
}

/// Proposes a wrong token at every position.
struct Adversary {
    prompt_len: usize,
    truth: Vec<u32>,
}

impl TokenDrafter for Adversary {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        let g = tokens.len() - self.prompt_len;
        out.extend(self.truth.iter().skip(g).take(max).map(|t| (t + 1) % 259));
    }
}

/// Random tokens, a random number of them; with `right` > 0, each position is the true token
/// with that probability (so acceptance stops at random places).
struct Noisy {
    prompt_len: usize,
    truth: Vec<u32>,
    right: f64,
    rng: Rng,
}

impl TokenDrafter for Noisy {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        let g = tokens.len() - self.prompt_len;
        let n = (self.rng.next_u64() % (max as u64 + 1)) as usize;
        for i in 0..n {
            let truth = self.truth.get(g + i).copied().unwrap_or(0);
            out.push(if self.rng.next_f64() < self.right {
                truth
            } else {
                (self.rng.next_u64() % 259) as u32
            });
        }
    }
}

/// Greedy rows of a verify step.
struct Greedy<'a> {
    rows: &'a [Vec<f32>],
}

impl Target for Greedy<'_> {
    fn dist(&mut self, i: usize) -> Dist {
        Dist::Greedy(argmax(&self.rows[i]))
    }
    fn uniform(&mut self) -> f64 {
        unreachable!("greedy")
    }
    fn push(&mut self, _: u32) {}
}

struct Seq {
    tokens: Vec<u32>,
    prompt_len: usize,
    n_past: usize,
    blocks: Vec<u32>,
    drafter: Box<dyn TokenDrafter>,
    /// Logits row that picked each generated token.
    rows: Vec<Vec<f32>>,
    drafted: usize,
    accepted: usize,
}

const K: usize = 4;
const STEPS: usize = 40;

/// Runs speculative greedy decoding of every sequence for `STEPS` steps in one executor: the
/// last sequence's prompt prefills in step 2, next to the others' verify step.
fn speculate(exec: &mut Executor, seqs: &mut [Seq], max_k: &mut dyn FnMut() -> usize) {
    let late = seqs.len() - 1;
    for s in 0..STEPS {
        let decoding: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].n_past > 0).collect();
        let prefill: Vec<usize> = (0..seqs.len())
            .filter(|&i| seqs[i].n_past == 0 && (i != late || s == 2))
            .collect();
        // Drafts, padded to a common length.
        let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); seqs.len()];
        for &i in &decoding {
            let k = max_k();
            let seq = &mut seqs[i];
            seq.drafter.propose(&seq.tokens, k, &mut drafts[i]);
            drafts[i].truncate(k);
        }
        let n_tok = 1 + decoding.iter().map(|&i| drafts[i].len()).max().unwrap_or(0);
        let fed: Vec<Vec<u32>> = (0..seqs.len())
            .map(|i| {
                let last = *seqs[i].tokens.last().unwrap();
                let mut f = vec![last];
                f.extend(&drafts[i]);
                f.resize(n_tok, last);
                f
            })
            .collect();
        let mut groups = Vec::new();
        if !decoding.is_empty() {
            groups.push(Group {
                slot: 0,
                seqs: decoding
                    .iter()
                    .map(|&i| SeqInput {
                        tokens: &fed[i],
                        n_past: seqs[i].n_past,
                        blocks: &seqs[i].blocks,
                        ring: 0,
                        logits: Logits::First(1 + drafts[i].len()),
                        media: None,
                    })
                    .collect(),
                verify: n_tok > 1,
            });
        }
        for &i in &prefill {
            groups.push(Group {
                slot: i,
                seqs: vec![SeqInput {
                    tokens: &seqs[i].tokens,
                    n_past: 0,
                    blocks: &seqs[i].blocks,
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                }],
                verify: false,
            });
        }
        let out = exec.step(&groups).unwrap();
        drop(groups);
        let mut rows = out.rows.into_iter();
        let mut commits = Vec::new();
        for &i in &decoding {
            let seq = &mut seqs[i];
            let mine: Vec<Vec<f32>> = rows.by_ref().take(1 + drafts[i].len()).collect();
            let o = accept::verify(&mut Greedy { rows: &mine }, &drafts[i], &[], |_| false);
            seq.drafted += drafts[i].len();
            seq.accepted += o.accepted;
            seq.rows.extend(mine.into_iter().take(o.appended.len()));
            eprintln!(
                "DBG step {s} seq {i} n_past {} n_tok {n_tok} k {} acc {} rows_total {}",
                seq.n_past,
                drafts[i].len(),
                o.accepted,
                seq.rows.len()
            );
            seq.n_past += o.appended.len();
            seq.tokens.extend(&o.appended);
            seq.drafter.accepted(&o.appended);
            if n_tok > 1 {
                commits.push((i, o.appended.len()));
            }
        }
        for &i in &prefill {
            let seq = &mut seqs[i];
            let row = rows.next().unwrap();
            seq.n_past = seq.tokens.len();
            seq.tokens.push(argmax(&row));
            seq.rows.push(row);
        }
        exec.commit(&commits).unwrap();
    }
}

#[test]
fn greedy_speculation_is_bit_identical_to_plain_decode() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = tiny_model(&dir);
    greedy_speculation_matches(&gguf, cpu);
}

/// The same on the tiny qwen4exp: hyper-connections, sparse attention (pool keys recomputed
/// after a rejected draft completed a pool), and the per-layer embedding, whose conv history
/// and n-gram token history roll back with the DeltaNet state.
#[test]
fn qwen4exp_speculation_is_bit_identical_to_plain_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-qwen4exp.gguf");
    legion::model::qwen4::tiny::write(&path, None, 3).unwrap();
    greedy_speculation_matches(&GgufFile::open(&path).unwrap(), cpu);
}

/// The same on CUDA, whose copies take other paths than the CPU's (a commit's conv window of
/// one slot: ggml's element-wise copy kernel rather than a 2D memcpy). A prompt or verify
/// step's matmuls sum in another order than a decode step's there, so the logits agree to
/// `CUDA_TOL` rather than bit for bit; a state rolled back wrong derails the tokens.
///
/// `cargo test --release --features cuda --test spec cuda_ -- --ignored`
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA GPU"]
fn cuda_speculation_matches_plain_decode() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = tiny_model(&dir);
    greedy_speculation_matches_to(
        &gguf,
        || legion::ggml::backend_by_name("CUDA0").unwrap(),
        CUDA_TOL,
    );
}

/// Largest |difference| of a CUDA run's logits from plain decoding on CUDA.
#[cfg(feature = "cuda")]
const CUDA_TOL: f32 = 2e-2;

fn greedy_speculation_matches(gguf: &GgufFile, new_backend: fn() -> Backend) {
    greedy_speculation_matches_to(gguf, new_backend, 0.0);
}

/// `tol`: the largest |difference| of a logit from plain decoding; 0: the same bits.
fn greedy_speculation_matches_to(gguf: &GgufFile, new_backend: fn() -> Backend, tol: f32) {
    let mut backend = new_backend();
    let model = Model::load(gguf, |_| backend.buffer_type()).unwrap();
    let prompts: Vec<Vec<u32>> = [
        "The quick brown fox jumps over the lazy dog. ".repeat(3),
        "speculative decoding".to_string(),
        "0123456789".repeat(9), // past one KV block
        "abc".to_string(),
        "Late prompt, prefilled next to a verify step.".to_string(),
    ]
    .iter()
    .map(|t| t.bytes().map(u32::from).collect())
    .collect();
    let n = STEPS * (K + 1) + K + 1;
    let truth: Vec<(Vec<u32>, Vec<Vec<f32>>)> = prompts
        .iter()
        .map(|p| plain(&model, &mut backend, p, n))
        .collect();
    drop(model);

    let n_seq = prompts.len();
    let blocks_per = 8;
    let model = Model::load(gguf, |_| backend.buffer_type()).unwrap();
    let mut exec = Executor::new(model, new_backend(), n_seq * blocks_per, n_seq, 0).unwrap();
    exec.enable_spec(K + 1).unwrap();
    let mut seqs: Vec<Seq> = prompts
        .iter()
        .zip(&truth)
        .enumerate()
        .map(|(i, (p, (t, _)))| {
            let (prompt_len, truth) = (p.len(), t.clone());
            let drafter: Box<dyn TokenDrafter> = match i {
                0 => Box::new(Oracle { prompt_len, truth }),
                1 => Box::new(Adversary { prompt_len, truth }),
                2 => Box::new(Noisy {
                    prompt_len,
                    truth,
                    right: 0.0,
                    rng: Rng::new(7),
                }),
                _ => Box::new(Noisy {
                    prompt_len,
                    truth,
                    right: 0.7,
                    rng: Rng::new(i as u64),
                }),
            };
            Seq {
                tokens: p.clone(),
                prompt_len,
                n_past: 0,
                // Interleaved blocks: sequence i owns i, i + n, i + 2n, ...
                blocks: (0..blocks_per).map(|b| (i + n_seq * b) as u32).collect(),
                drafter,
                rows: Vec::new(),
                drafted: 0,
                accepted: 0,
            }
        })
        .collect();
    assert!(prompts.iter().all(|p| p.len() + n < blocks_per * KV_BLOCK));
    let mut rng = Rng::new(3);
    speculate(&mut exec, &mut seqs, &mut || {
        // Mostly K, sometimes fewer, sometimes none.
        [K, K, K, 2, 1, 0][(rng.next_u64() % 6) as usize]
    });

    for (i, (seq, (tokens, rows))) in seqs.iter().zip(&truth).enumerate() {
        let generated = &seq.tokens[seq.prompt_len..];
        assert!(
            generated.len() >= STEPS - 3,
            "seq {i}: {} tokens",
            generated.len()
        );
        // With a tolerance, a near tie may pick another token: compare up to the row that
        // picked it, which still follows the same tokens.
        let same = generated
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if tol == 0.0 {
            assert_eq!(same, generated.len(), "seq {i}: tokens differ");
        } else {
            assert!(same >= 10, "seq {i}: tokens differ from token {same}");
        }
        let mut worst = 0f32;
        for (j, (a, b)) in seq.rows.iter().zip(rows).enumerate().take(same + 1) {
            let diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff);
            assert!(
                if tol == 0.0 {
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                } else {
                    diff <= tol
                },
                "seq {i}: logits row {j} differs by {diff}"
            );
        }
        println!(
            "seq {i}: {} tokens ({same} as plain decoding, logits within {worst:.2e}), {} drafted, {} accepted",
            generated.len(),
            seq.drafted,
            seq.accepted
        );
    }
    // The drafters did what they were meant to.
    assert!(seqs[0].accepted > 0 && seqs[0].accepted == seqs[0].drafted);
    assert!(seqs[1].drafted > 0 && seqs[1].accepted == 0);
    assert!(seqs[3].accepted > 0 && seqs[3].accepted < seqs[3].drafted);
}

/// Verify steps on the paged-attention path (its CPU reference implementation): speculative
/// decoding with paged attention reproduces plain decoding with paged attention bit for bit.
#[test]
fn verify_steps_work_with_paged_attention() {
    use legion::model::core::Attention;
    let dir = tempfile::tempdir().unwrap();
    let gguf = tiny_model(&dir);
    let prompts: Vec<Vec<u32>> = [
        "paged attention, verified",
        "0123456789".repeat(8).as_str(),
        "xyz",
    ]
    .iter()
    .map(|t| t.bytes().map(u32::from).collect())
    .collect();
    let n_seq = prompts.len() + 1;
    let blocks_per = 8;
    let run = |drafters: Vec<Box<dyn TokenDrafter>>, max_k: &mut dyn FnMut() -> usize| {
        let backend = cpu();
        let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        let mut exec = Executor::new(model, backend, n_seq * blocks_per, n_seq, 0).unwrap();
        exec.set_attention(Attention::Paged);
        assert!(exec.paged());
        exec.enable_spec(K + 1).unwrap();
        // The late sequence (prefilled in step 2) repeats the first prompt.
        let mut all = prompts.clone();
        all.push(prompts[0].clone());
        let mut seqs: Vec<Seq> = all
            .iter()
            .zip(drafters)
            .enumerate()
            .map(|(i, (p, drafter))| Seq {
                tokens: p.clone(),
                prompt_len: p.len(),
                n_past: 0,
                blocks: (0..blocks_per).map(|b| (i + n_seq * b) as u32).collect(),
                drafter,
                rows: Vec::new(),
                drafted: 0,
                accepted: 0,
            })
            .collect();
        speculate(&mut exec, &mut seqs, max_k);
        seqs
    };
    struct Silent;
    impl TokenDrafter for Silent {
        fn propose(&mut self, _: &[u32], _: usize, _: &mut Vec<u32>) {}
    }
    let reference = run(
        (0..n_seq).map(|_| Box::new(Silent) as _).collect(),
        &mut || 0,
    );
    let drafters = reference
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let (prompt_len, truth) = (r.prompt_len, r.tokens[r.prompt_len..].to_vec());
            Box::new(Noisy {
                prompt_len,
                truth,
                right: if i == 0 { 1.0 } else { 0.6 },
                rng: Rng::new(i as u64 + 10),
            }) as Box<dyn TokenDrafter>
        })
        .collect();
    let mut rng = Rng::new(5);
    let spec = run(drafters, &mut || {
        [K, K, 2, 0][(rng.next_u64() % 4) as usize]
    });
    for (i, (a, b)) in reference.iter().zip(&spec).enumerate() {
        let n = a.tokens.len().min(b.tokens.len());
        assert!(n > a.prompt_len + STEPS / 2);
        assert_eq!(a.tokens[..n], b.tokens[..n], "seq {i}: tokens differ");
        for (j, (x, y)) in a.rows.iter().zip(&b.rows).enumerate() {
            assert!(
                x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                "seq {i}: logits row {j} differs"
            );
        }
    }
    assert!(spec.iter().map(|s| s.accepted).sum::<usize>() > 0);
}

/// Hidden-state capture for model drafters: a step returns the chosen layers' outputs for
/// every token, equal to the same layers' outputs traced in a single-sequence pass.
#[test]
fn captured_hidden_states_match_the_trace() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = tiny_model(&dir);
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_embd = model.hp.n_embd as usize;
    let last = model.hp.n_layer - 1;
    let prompt: Vec<u32> = "hidden states for the drafter"
        .bytes()
        .map(u32::from)
        .collect();
    let (_, traced) = model.prompt_trace(&mut backend, &prompt).unwrap();
    let layer = |il: usize| {
        traced
            .iter()
            .find(|t| t.name == format!("l_out-{il}"))
            .unwrap()
            .data
            .clone()
    };

    let mut exec = Executor::new(model, cpu(), 4, 2, 0).unwrap();
    exec.capture_hidden(vec![last, 0]);
    let blocks = [0u32, 1, 2, 3];
    let split = 10;
    let out = exec
        .step(&[Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens: &prompt[..split],
                n_past: 0,
                blocks: &blocks,
                ring: 0,
                logits: Logits::None,
                media: None,
            }],
            verify: false,
        }])
        .unwrap();
    assert_eq!(out.hidden.len(), 2);
    assert_eq!(out.hidden[0].len(), split * n_embd);
    let mut got = out.hidden;
    // The rest as a one-token step and a chunk: still every token's rows, in order.
    for (n_past, chunk) in [
        (split, &prompt[split..split + 1]),
        (split + 1, &prompt[split + 1..]),
    ] {
        let out = exec
            .step(&[Group {
                slot: 0,
                seqs: vec![SeqInput {
                    tokens: chunk,
                    n_past,
                    blocks: &blocks,
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                }],
                verify: false,
            }])
            .unwrap();
        for (g, h) in got.iter_mut().zip(out.hidden) {
            assert_eq!(h.len(), chunk.len() * n_embd);
            g.extend(h);
        }
    }
    assert!(
        got[0]
            .iter()
            .zip(layer(last))
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        got[1]
            .iter()
            .zip(layer(0))
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(got[0].len(), layer(last).len());
}

/// The measurement behind the recurrent-state rollback choice (see `legion::spec`): per verify
/// step at Qwen3.5-4B DeltaNet sizes (24 recurrent layers; 32 value heads and 16 key heads of
/// 128; conv kernel 4) with k = 4 drafts, on the CPU with 8 threads, for a batch of 4
/// sequences, the DeltaNet recurrence alone (the rest of the step is the same either way):
/// plain one-token decode, a verify op that keeps only the final state (replay), a verify op
/// that writes all 5 per-position snapshots, and the rollbacks: replaying the delta rule over
/// the kept tokens, or copying one snapshot. Random data; no model.
///
/// `cargo test --release --test spec spec_rollback_cost -- --ignored --nocapture`
#[test]
#[ignore = "a CPU measurement, not a check"]
fn spec_rollback_cost() {
    use legion::ggml::{ComputeAllocator, Context, types};
    const LAYERS: usize = 24;
    const NV: i64 = 32;
    const NK: i64 = 16;
    const S: i64 = 128;
    const NS: i64 = 4;
    const NT: i64 = 5;
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    let buft = backend.buffer_type();

    // Inputs for every layer: q, k, v, g, beta for NT tokens and the states.
    let data = Context::new(LAYERS * 8, 0);
    let mut rng = Rng::new(1);
    let mut layers = Vec::new();
    for _ in 0..LAYERS {
        let t = |ne: &[i64]| data.new_tensor(types::F32, ne);
        layers.push((
            t(&[S, NK, NT, NS]),
            t(&[S, NK, NT, NS]),
            t(&[S, NV, NT, NS]),
            t(&[1, NV, NT, NS]),
            t(&[1, NV, NT, NS]),
            t(&[S, S, NV, NS]),
        ));
    }
    data.alloc_tensors(buft).unwrap();
    for &(q, k, v, g, b, st) in &layers {
        for (t, scale, off) in [
            (q, 0.09, 0.0),
            (k, 0.09, 0.0),
            (v, 1.0, 0.0),
            (g, 0.1, -0.2),
            (b, 0.5, 0.5),
            (st, 0.01, 0.0),
        ] {
            let n = t.elements();
            let vals: Vec<f32> = (0..n)
                .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * scale + off)
                .collect();
            t.set_f32(&vals);
        }
    }

    // One graph over all layers: `tokens` tokens per sequence, `snaps` snapshots; `write`
    // copies the final state back (plain decode, replay) or snapshot 0 (a snapshot rollback,
    // which copies one state).
    let time = |backend: &mut Backend, tokens: i64, snaps: i64, write: bool, op: bool| {
        let ctx = Context::new(LAYERS * 16 + 16, LAYERS * 16 + 16);
        let mut graph = ctx.new_graph(LAYERS * 16 + 16);
        for &(q, k, v, g, b, st) in &layers {
            fn tokens_view<'a>(
                ctx: &'a Context,
                t: legion::ggml::Tensor<'a>,
                ne: [i64; 3],
            ) -> legion::ggml::Tensor<'a> {
                let nb = t.nb();
                ctx.view_4d(t, [ne[0], ne[1], ne[2], NS], [nb[1], nb[2], nb[3]], 0)
            }
            let view = |t, a, h| tokens_view(&ctx, t, [a, h, tokens]);
            if op {
                let gv = ctx.cont(view(g, 1, NV));
                let bv = ctx.cont(view(b, 1, NV));
                let out = ctx.gated_delta_net(
                    view(q, S, NK),
                    view(k, S, NK),
                    view(v, S, NV),
                    gv,
                    bv,
                    st,
                    snaps,
                );
                let es = 4;
                let fin = ctx.view_4d(
                    out,
                    [S, S, NV, NS],
                    [
                        S as usize * es,
                        (S * S) as usize * es,
                        (S * S * NV) as usize * es,
                    ],
                    (S * NV * tokens * NS) as usize * es,
                );
                graph.expand(if write { ctx.cpy(fin, st) } else { out });
            } else {
                // A snapshot rollback: one state copy per sequence.
                let other = ctx.new_tensor(types::F32, &[S, S, NV, NS]);
                graph.expand(ctx.cpy(st, other));
            }
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        backend.compute(&graph).unwrap();
        let mut ms: Vec<f64> = (0..7)
            .map(|_| {
                let t = std::time::Instant::now();
                backend.compute(&graph).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        ms[3]
    };
    let state_mib = (S * S * NV * 4) as f64 * LAYERS as f64 / (1 << 20) as f64;
    println!(
        "DeltaNet state per sequence: {state_mib:.1} MiB; batch of {NS}, k = {}",
        NT - 1
    );
    let plain = time(&mut backend, 1, 1, true, true);
    println!("plain decode (1 token, state written back): {plain:.2} ms");
    let verify = time(&mut backend, NT, 1, false, true);
    println!("verify, final state only (replay design):   {verify:.2} ms");
    let snap = time(&mut backend, NT, NT, false, true);
    println!(
        "verify, {NT} snapshots:                         {snap:.2} ms, +{:.0} MiB per sequence",
        state_mib * NT as f64
    );
    for keep in [1, 3, NT] {
        let replay = time(&mut backend, keep, 1, true, true);
        println!("rollback by replay of {keep} tokens:              {replay:.2} ms");
    }
    let copy = time(&mut backend, NT, 1, true, false);
    println!("rollback by snapshot copy:                     {copy:.2} ms");
    // The replay pool (legion::model::core::SpecPool) per sequence: conv window and the
    // delta rule's inputs for NT tokens.
    let conv_dim = S * NV + 2 * S * NK;
    let floats = (3 + NT) * conv_dim + NT * (2 * S * NK + S * NV + 2 * NV);
    println!(
        "replay buffers: {:.1} MiB per sequence",
        (floats * 4) as f64 * LAYERS as f64 / (1 << 20) as f64
    );
}

/// The commit graph has room for its nodes with many recurrent layers and a run of several
/// slots, as with 36 DeltaNet layers and four sequences keeping as many tokens (each layer's
/// run copies g and beta to contiguous memory then).
#[test]
fn a_commit_of_many_layers_and_slots_fits_its_graph() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny48.gguf");
    let config = TinyConfig {
        n_layer: 48,
        ..TinyConfig::default()
    };
    tiny::write(&path, &config).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::load(&gguf, |_| cpu().buffer_type()).unwrap();
    let n_seq = 4;
    let mut exec = Executor::new(model, cpu(), 4 * n_seq, n_seq, 0).unwrap();
    exec.enable_spec(K + 1).unwrap();
    let blocks: Vec<Vec<u32>> = (0..n_seq)
        .map(|i| (0..4).map(|b| (i + n_seq * b) as u32).collect())
        .collect();
    let prompt: Vec<u32> = "many layers".bytes().map(u32::from).collect();
    let groups: Vec<Group> = (0..n_seq)
        .map(|i| Group {
            slot: i,
            seqs: vec![SeqInput {
                tokens: &prompt,
                n_past: 0,
                blocks: &blocks[i],
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
            verify: false,
        })
        .collect();
    exec.step(&groups).unwrap();
    let fed = [1u32, 2, 3];
    let verify = Group {
        slot: 0,
        seqs: (0..n_seq)
            .map(|i| SeqInput {
                tokens: &fed,
                n_past: prompt.len(),
                blocks: &blocks[i],
                ring: 0,
                logits: Logits::All,
                media: None,
            })
            .collect(),
        verify: true,
    };
    let rows = exec.step(&[verify]).unwrap().rows;
    assert_eq!(rows.len(), n_seq * fed.len());
    let commits: Vec<(usize, usize)> = (0..n_seq).map(|i| (i, 2)).collect();
    exec.commit(&commits).unwrap();
}

/// A commit only starts its graphs ([`Executor::commit`]); whatever reads the states another
/// way waits for it. After the last commit of a speculative run, copying a slot's state and
/// clearing it (both wait) leave a state that decodes on as the committed one: the copy put
/// back gives the same next logits as the running slot did.
#[test]
fn state_copies_wait_for_the_commit() {
    use legion::model::core::Pool;
    let dir = tempfile::tempdir().unwrap();
    let gguf = tiny_model(&dir);
    let prompt: Vec<u32> = "speculative decoding".bytes().map(u32::from).collect();
    let model = Model::load(&gguf, |_| cpu().buffer_type()).unwrap();
    let mut exec = Executor::new(model, cpu(), 8, 1, 1).unwrap();
    exec.enable_spec(K + 1).unwrap();
    let blocks: Vec<u32> = (0..8).collect();
    let feed = |exec: &mut Executor, tokens: &[u32], n_past: usize, verify: bool| {
        let g = Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past,
                blocks: &blocks,
                ring: 0,
                logits: Logits::All,
                media: None,
            }],
            verify,
        };
        exec.step(&[g]).unwrap().rows
    };
    let rows = feed(&mut exec, &prompt, 0, false);
    let t0 = argmax(rows.last().unwrap());
    // A verify step of three tokens, of which two are kept.
    let n = prompt.len();
    feed(&mut exec, &[t0, 1, 2], n, true);
    exec.commit(&[(0, 2)]).unwrap();
    // The commit is only started; a state copy waits for it.
    assert!(exec.commit_in_flight());
    // Save the committed state, decode one token, then restore and decode it again.
    exec.copy_states(&[(Pool::Seq, 0, Pool::Saved, 0)]).unwrap();
    assert!(!exec.commit_in_flight());
    let a = feed(&mut exec, &[3], n + 2, false);
    exec.clear_slot(0);
    exec.copy_states(&[(Pool::Saved, 0, Pool::Seq, 0)]).unwrap();
    let b = feed(&mut exec, &[3], n + 2, false);
    assert_eq!(a, b);
}
