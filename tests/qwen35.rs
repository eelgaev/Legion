//! Qwen3.5/3.6 forward pass (dense `qwen35` and `qwen35moe`) against llama.cpp, and CUDA against
//! CPU. Measured with Qwen3.6-35B-A3B MXFP4 and Qwen3.5-4B Q8_0.
//!
//! All tests here are ignored: they need the real model and reference files made with
//! `tools/llama-ref/llama_ref.cpp` (build instructions in its header). Setup, with
//! `D=<reference dir>` and `M=<model.gguf>`:
//!
//! ```sh
//! LEGION_TEST_QWEN35=$M LEGION_TEST_QWEN35_REF_DIR=$D \
//!   cargo test --release --test qwen35 write_prompt_ids -- --ignored   # writes $D/<name>.ids
//! for p in $D/*.ids; do n=${p%.ids}
//!   REF_MODE=exact   llama_ref $M $p $n.exact.f32
//!   REF_MODE=default llama_ref $M $p $n.default.f32
//! done
//! mkdir $D/france.dump && REF_MODE=exact llama_ref $M $D/france.ids /dev/null $D/france.dump '-[0-9]+$'
//! LEGION_TEST_QWEN35=$M LEGION_TEST_QWEN35_REF_DIR=$D \
//!   cargo test --release --test qwen35 -- --ignored --nocapture --test-threads 1
//! ```
//!
//! What should match and what may not:
//! - `exact` references (llama.cpp with flash attention and CPU weight repacking off, F16 KV
//!   cache) run the same ggml kernels as our graph, so logits agree to the last bit (measured:
//!   0.0 on both prompts). The 1e-4 bound is only there so a libm or compiler difference doesn't
//!   fail the test; a real graph bug is orders of magnitude above it.
//! - `default` references are llama.cpp as users run it. Its CPU flash attention rounds Q to F16
//!   and accumulates in F16, and repacked MXFP4 experts sum in another order. Those are 1-ulp to
//!   1e-4 differences, but the next quantized matmul quantizes its input to Q8_0, so a tiny change
//!   can flip a rounding step and the drift grows ~10x per few layers (max |logit diff| ~0.5).
//!   These are checked statistically: top-1 agreement over all positions and KL divergence.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::tokenizer::Tokenizer;

/// (name, text). The second is long enough (~50 tokens) for routing flips to show up in the
/// statistics.
const PROMPTS: &[(&str, &str)] = &[
    ("france", "The capital of France is"),
    (
        "apollo",
        "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the lunar module \
         Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled the goal \
         President Kennedy had set eight years earlier, and the next mission",
    ),
];

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn ref_dir() -> std::path::PathBuf {
    env("LEGION_TEST_QWEN35_REF_DIR").into()
}

fn read_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect(),
    )
}

fn load() -> (GgufFile, Backend) {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN35")).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    (gguf, backend)
}

fn prompt_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, true)
}

/// Writes `<ref dir>/<name>.ids` for every prompt, the input for `llama_ref`.
#[test]
#[ignore = "needs the real model"]
fn write_prompt_ids() {
    let (gguf, _) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let dir = ref_dir();
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in PROMPTS {
        let ids = prompt_ids(&tok, text);
        let list: Vec<String> = ids.iter().map(u32::to_string).collect();
        std::fs::write(dir.join(format!("{name}.ids")), list.join(",")).unwrap();
        println!("{name}: {} tokens", ids.len());
    }
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

/// Agreement of all-position logits `ours` with `theirs`, both `[n][n_vocab]`.
struct Agreement {
    max_diff: f32,
    top1: usize,
    positions: usize,
    kl_mean: f64,
    kl_max: f64,
}

fn agreement(ours: &[f32], theirs: &[f32], n_vocab: usize) -> Agreement {
    assert_eq!(ours.len(), theirs.len(), "logit count");
    let mut a = Agreement {
        max_diff: 0.0,
        top1: 0,
        positions: ours.len() / n_vocab,
        kl_mean: 0.0,
        kl_max: 0.0,
    };
    for (o, t) in ours.chunks_exact(n_vocab).zip(theirs.chunks_exact(n_vocab)) {
        for (x, y) in o.iter().zip(t) {
            a.max_diff = a.max_diff.max((x - y).abs());
        }
        a.top1 += (argmax(o) == argmax(t)) as usize;
        let (lo, lt) = (log_softmax(o), log_softmax(t));
        // KL(llama.cpp || ours): how much probability mass we misplace, in nats.
        let kl: f64 = lt.iter().zip(&lo).map(|(t, o)| t.exp() * (t - o)).sum();
        a.kl_mean += kl / a.positions as f64;
        a.kl_max = a.kl_max.max(kl);
    }
    a
}

#[test]
#[ignore = "needs the real model and llama.cpp reference logits"]
fn prompt_logits_match_llama_cpp() {
    let (gguf, mut backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let dir = ref_dir();
    let mut checked = 0;
    for (name, text) in PROMPTS {
        let ids = prompt_ids(&tok, text);
        if let Ok(saved) = std::fs::read_to_string(dir.join(format!("{name}.ids"))) {
            let saved: Vec<u32> = saved
                .trim()
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(
                ids, saved,
                "{name}: references were made from other token ids"
            );
        }
        let t = std::time::Instant::now();
        let logits = model.prompt_logits_all(&mut backend, &ids).unwrap();
        println!(
            "{name}: {} tokens in {:.2} s",
            ids.len(),
            t.elapsed().as_secs_f64()
        );
        assert_eq!(logits.len(), ids.len() * n_vocab);

        if let Some(exact) = read_f32(&dir.join(format!("{name}.exact.f32"))) {
            let a = agreement(&logits, &exact, n_vocab);
            println!(
                "  exact:   max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                a.max_diff, a.top1, a.positions, a.kl_mean, a.kl_max
            );
            // Same kernels on both sides: anything above rounding noise is a bug.
            assert!(
                a.max_diff <= 1e-4,
                "{name}: logits differ by {}",
                a.max_diff
            );
            assert_eq!(a.top1, a.positions, "{name}: top-1 differs");
            checked += 1;
        }
        if let Some(default) = read_f32(&dir.join(format!("{name}.default.f32"))) {
            let a = agreement(&logits, &default, n_vocab);
            println!(
                "  default: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                a.max_diff, a.top1, a.positions, a.kl_mean, a.kl_max
            );
            // Rounding drift only (see the module docs). Measured: max |diff| 0.51 / 1.61, mean KL
            // 1.5e-3 / 2.1e-3, max KL 5.7e-3 / 1.4e-2, top-1 5/5 and 51/51 (france / apollo).
            // The bounds leave headroom over that. A wrong rope section,
            // head mapping or routing rule pushes KL to O(1).
            assert!(a.max_diff < 2.5, "{name}: logits differ by {}", a.max_diff);
            assert!(a.kl_mean < 0.015, "{name}: mean KL {}", a.kl_mean);
            assert!(a.kl_max < 0.1, "{name}: max KL {}", a.kl_max);
            assert!(
                a.top1 * 10 >= a.positions * 9,
                "{name}: top-1 agrees at {}/{} positions",
                a.top1,
                a.positions
            );
            checked += 1;
        }
        if *name == "france" {
            // The last-token path (one output row) must agree with the all-positions one.
            let last = model.prompt_logits(&mut backend, &ids).unwrap();
            let tail = &logits[logits.len() - n_vocab..];
            let d = last
                .iter()
                .zip(tail)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                d <= 1e-3,
                "last-token logits differ from all-position ones by {d}"
            );
        }
    }
    assert!(
        checked > 0,
        "no reference logits found in {}",
        dir.display()
    );
}

/// Every traced intermediate against a full-tensor llama.cpp dump (`llama_ref ... <dump_dir>`,
/// exact mode) for one prompt (LEGION_TEST_QWEN35_TRACE_PROMPT, default `france`), printing the first tensor that drifts. With
/// LEGION_TEST_QWEN35_DUMP=<dir>, our tensors are also written there (`<name>.bin`, raw f32).
#[test]
#[ignore = "needs the real model and a llama.cpp tensor dump"]
fn trace_against_llama_cpp() {
    let (gguf, mut backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let pick = std::env::var("LEGION_TEST_QWEN35_TRACE_PROMPT").unwrap_or("france".into());
    let (name, text) = *PROMPTS
        .iter()
        .find(|(n, _)| *n == pick)
        .expect("no such prompt");
    let ids = prompt_ids(&tok, text);
    let (logits, traced) = model.prompt_trace(&mut backend, &ids).unwrap();

    if let Ok(dir) = std::env::var("LEGION_TEST_QWEN35_DUMP") {
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = String::new();
        for t in &traced {
            let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
            std::fs::write(format!("{dir}/{}.bin", t.name), bytes).unwrap();
            manifest += &format!("{} {:?}\n", t.name, t.ne);
        }
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_ne_bytes()).collect();
        std::fs::write(format!("{dir}/logits.f32"), bytes).unwrap();
        std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
    }

    // llama.cpp names a tensor several times while building it (`Kcur-3` is the matmul, the
    // reshape and the rope output); the last one is the finished value.
    let dump = ref_dir().join(format!("{name}.dump"));
    let manifest = std::fs::read_to_string(dump.join("manifest.txt")).unwrap();
    let mut files = std::collections::HashMap::new();
    for line in manifest.lines() {
        let mut f = line.split(' ');
        let file = f.next().unwrap();
        let ty = f.nth(1).unwrap();
        let base = file.rsplitn(3, '.').nth(2).unwrap();
        files.insert(base.to_string(), (file.to_string(), ty == "i32"));
    }
    let mut first_drift = None;
    let mut worst = 0f64;
    for t in &traced {
        let Some((file, is_i32)) = files.get(&t.name) else {
            continue;
        };
        let theirs: Vec<f32> = read_f32(&dump.join(file))
            .unwrap()
            .into_iter()
            .map(|v| {
                if *is_i32 {
                    v.to_bits() as i32 as f32
                } else {
                    v
                }
            })
            .collect();
        assert_eq!(theirs.len(), t.data.len(), "{}: element count", t.name);
        let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
        let max_diff = t
            .data
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max) as f64;
        let rel = max_diff / scale;
        println!("{:<24} max |diff| {max_diff:.2e}  rel {rel:.2e}", t.name);
        if rel > 1e-5 && first_drift.is_none() {
            first_drift = Some(t.name.clone());
        }
        worst = worst.max(rel);
    }
    println!("worst relative difference {worst:.2e}, first drift: {first_drift:?}");
    assert!(first_drift.is_none(), "drift starts at {first_drift:?}");
}

/// The same prompts on the CUDA backend (all weights in the GPU's buffer type) against the CPU
/// backend. LEGION_TEST_QWEN35_CUDA=<model.gguf>, separate from LEGION_TEST_QWEN35 so the GPU is
/// only used when asked for. Build with `--features cuda`.
///
/// The backends run different kernels (CUDA quantizes activations to Q8_1 for MMQ/MMVQ, uses its
/// own reductions and fused ops), so this is a statistical check like the `default` one above.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let path = env("LEGION_TEST_QWEN35_CUDA");
    let gguf = GgufFile::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();

    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .expect("no GPU device")
        .init()
        .unwrap();
    println!("GPU backend: {}", gpu.name());

    let t = std::time::Instant::now();
    let on_cpu = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    println!("CPU load {:.1} s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let on_gpu = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    println!("GPU load {:.1} s", t.elapsed().as_secs_f64());
    let n_vocab = on_cpu.hp.n_vocab as usize;

    for (name, text) in PROMPTS {
        let ids = prompt_ids(&tok, text);
        let t = std::time::Instant::now();
        let reference = on_cpu.prompt_logits_all(&mut cpu, &ids).unwrap();
        let cpu_s = t.elapsed().as_secs_f64();
        // First GPU run includes graph allocation and kernel warm-up; time the second.
        on_gpu.prompt_logits_all(&mut gpu, &ids).unwrap();
        let t = std::time::Instant::now();
        let logits = on_gpu.prompt_logits_all(&mut gpu, &ids).unwrap();
        let gpu_s = t.elapsed().as_secs_f64();
        let a = agreement(&logits, &reference, n_vocab);
        println!(
            "{name} ({} tokens): CPU {:.1} ms, GPU {:.1} ms; max |diff| {:.2e}, top-1 {}/{}, \
             KL mean {:.2e} max {:.2e}",
            ids.len(),
            cpu_s * 1e3,
            gpu_s * 1e3,
            a.max_diff,
            a.top1,
            a.positions,
            a.kl_mean,
            a.kl_max
        );
        assert!(a.kl_mean < 0.015, "{name}: mean KL {}", a.kl_mean);
        assert!(a.kl_max < 0.1, "{name}: max KL {}", a.kl_max);
        assert!(
            a.top1 * 10 >= a.positions * 9,
            "{name}: top-1 agrees at {}/{} positions",
            a.top1,
            a.positions
        );
    }
}

/// Prefill + one-token decode steps against one pass over the whole text: for a prompt and 10
/// greedy tokens, the logits after prefill and after each decode step equal the
/// all-positions logits of `prompt_logits_all(prompt + generated)` at the same position.
///
/// With the CPU's reference kernels (`set_use_ref`) every op is batch-invariant, the attention
/// span is padded to the same length, and the DeltaNet state is carried exactly, so the two
/// agree to the last bit; the bound only absorbs libm noise. The decode steps run the same
/// kernels with or without `use_ref` (a one-token matmul is always a GEMV), so this also pins
/// what production decode computes.
#[test]
#[ignore = "needs the real model"]
fn decode_matches_full_prompt() {
    decode_against_full_prompt(true);
}

/// The same with the default kernels: batched prefill matmuls use tiled GEMM (another summation
/// order than the one-token GEMV; MoE expert matmuls too), so this is only rounding-close.
/// Measured: 4B max |diff| 0.27, max KL 7.8e-4; 35B MoE max |diff| 7.0 (one routing flip),
/// max KL 1.4e-2; same greedy tokens in both.
#[test]
#[ignore = "needs the real model"]
fn decode_close_to_full_prompt_default_kernels() {
    decode_against_full_prompt(false);
}

fn decode_against_full_prompt(exact: bool) {
    let (gguf, mut backend) = load();
    backend.set_use_ref(exact);
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
    let mut alloc = None;
    for (name, text) in PROMPTS {
        let prompt = prompt_ids(&tok, text);
        state.reset();
        let t = std::time::Instant::now();
        let mut step_logits = vec![
            model
                .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
                .unwrap(),
        ];
        let prefill_s = t.elapsed().as_secs_f64();
        let mut generated = Vec::new();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let next = argmax(step_logits.last().unwrap()) as u32;
            generated.push(next);
            step_logits.push(
                model
                    .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                    .unwrap(),
            );
        }
        let decode_s = t.elapsed().as_secs_f64();
        assert_eq!(state.n_past(), prompt.len() + 10);

        let mut full_ids = prompt.clone();
        full_ids.extend(&generated);
        let full = model.prompt_logits_all(&mut backend, &full_ids).unwrap();
        let rows = prompt.len() - 1..full_ids.len();
        let a = agreement(
            &step_logits.concat(),
            &full[rows.start * n_vocab..rows.end * n_vocab],
            n_vocab,
        );
        println!(
            "{name}: prefill {} tokens {:.1} ms, 10 decode steps {:.1} ms/token; vs full pass: \
             max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}; {:?}",
            prompt.len(),
            prefill_s * 1e3,
            decode_s * 1e2,
            a.max_diff,
            a.top1,
            a.positions,
            a.kl_mean,
            a.kl_max,
            tok.decode(&generated)
        );
        if exact {
            assert!(
                a.max_diff <= 1e-4,
                "{name}: logits differ by {}",
                a.max_diff
            );
        } else {
            // Rounding drift, as in `prompt_logits_match_llama_cpp`'s default references.
            assert!(a.kl_max < 0.1, "{name}: max KL {}", a.kl_max);
        }
        // Greedy decode picked these tokens, so they must be the full pass's top-1 too.
        assert_eq!(a.top1, a.positions, "{name}: top-1 differs");
    }
}

/// Continuous batching changes nothing a sequence computes: with the CPU's reference kernels,
/// three prompts prefilled and decoded together (one step prefilling two prompts as separate
/// groups, one step mixing a two-sequence decode with the third prompt's prefill, then
/// three-sequence decode steps; KV blocks interleaved across the sequences) give the same
/// logits, bit for bit, as each prompt run alone through [`Model::forward`], and so the same
/// greedy tokens.
#[test]
#[ignore = "needs the real model"]
fn batched_decode_matches_alone() {
    use legion::model::core::{Executor, Group, KV_BLOCK, Logits, SeqInput};
    const STEPS: usize = 8;
    let (gguf, mut backend) = load();
    backend.set_use_ref(true);
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let texts = [PROMPTS[0].1, PROMPTS[1].1, "Write a haiku about the sea."];
    let prompts: Vec<Vec<u32>> = texts.iter().map(|t| prompt_ids(&tok, t)).collect();

    // Alone: prefill, then greedy decode; logits after prefill and after every step.
    let mut alone = Vec::new();
    let mut alloc = None;
    for p in &prompts {
        let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
        let mut rows = vec![
            model
                .forward(&mut backend, &mut alloc, &mut state, p, false)
                .unwrap(),
        ];
        for _ in 0..STEPS {
            let next = argmax(rows.last().unwrap()) as u32;
            rows.push(
                model
                    .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                    .unwrap(),
            );
        }
        alone.push(rows);
    }
    drop(alloc);

    // Batched, in one executor. Sequence i owns blocks i, i + 3, i + 6, ...
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    cpu.set_use_ref(true);
    let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let mut exec = Executor::new(model, cpu, 3 * 8, 3, 0).unwrap();
    let blocks: Vec<Vec<u32>> = (0..3u32)
        .map(|i| (0..8u32).map(|b| i + 3 * b).collect())
        .collect();
    assert!(prompts.iter().all(|p| p.len() + STEPS < 8 * KV_BLOCK));
    let mut batched: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 3];
    let mut n_past = [0usize; 3];
    let mut next = [0u32; 3];
    let prefill = |i: usize| SeqInput {
        tokens: &prompts[i],
        n_past: 0,
        blocks: &blocks[i],
        ring: 0,
        logits: Logits::Last,
        media: None,
    };
    let row = |out: &legion::model::core::StepOutput, k: usize| {
        assert_eq!(out.rows[k].len(), n_vocab);
        // The device argmax is the greedy pick.
        assert_eq!(out.argmax[k] as usize, argmax(&out.rows[k]));
        out.rows[k].clone()
    };

    // Step 1: prompts 0 and 1, as two groups (slots 0 and 1).
    let out = exec
        .step(&[
            Group {
                slot: 0,
                seqs: vec![prefill(0)],
                verify: false,
            },
            Group {
                slot: 1,
                seqs: vec![prefill(1)],
                verify: false,
            },
        ])
        .unwrap();
    for i in 0..2 {
        batched[i].push(row(&out, i));
        n_past[i] = prompts[i].len();
    }
    for s in 0..=STEPS {
        for i in 0..3 {
            if let Some(l) = batched[i].last() {
                next[i] = argmax(l) as u32;
            }
        }
        let running = if s == 0 { 2 } else { 3 };
        let decode = Group {
            slot: 0,
            seqs: (0..running)
                .map(|i| SeqInput {
                    tokens: std::slice::from_ref(&next[i]),
                    n_past: n_past[i],
                    blocks: &blocks[i],
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                })
                .collect(),
            verify: false,
        };
        let n_dec = decode.seqs.len();
        // Step 2 also prefills prompt 2 (slot 2) next to the two-sequence decode.
        let groups = if s == 0 {
            vec![
                decode,
                Group {
                    slot: 2,
                    seqs: vec![prefill(2)],
                    verify: false,
                },
            ]
        } else {
            vec![decode]
        };
        let out = exec.step(&groups).unwrap();
        for k in 0..n_dec {
            batched[k].push(row(&out, k));
            n_past[k] += 1;
        }
        if s == 0 {
            batched[2].push(row(&out, 2));
            n_past[2] = prompts[2].len();
        }
    }

    for i in 0..3 {
        let a = alone[i].concat();
        let b = batched[i][..=STEPS].concat();
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let tokens =
            |rows: &[Vec<f32>]| -> Vec<u32> { rows.iter().map(|r| argmax(r) as u32).collect() };
        println!(
            "seq {i}: max |diff| {diff:.2e}, {:?}",
            tok.decode(&tokens(&batched[i]))
        );
        assert_eq!(tokens(&alone[i]), tokens(&batched[i][..=STEPS]), "seq {i}");
        assert_eq!(diff, 0.0, "seq {i}: batched logits differ by {diff}");
    }
}

/// Decode on CUDA against decode on the CPU: a prompt and 10 greedy tokens (picked by the CPU
/// run), step by step. On the GPU a decode step updates the delta-rule state in place (the
/// gated_delta_net output is placed over the state slots), which the CPU path never does, so
/// this pins that path. Statistical, like `cuda_matches_cpu`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_decode_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let path = env("LEGION_TEST_QWEN35_CUDA");
    let gguf = GgufFile::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .expect("no GPU device")
        .init()
        .unwrap();
    let on_cpu = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let on_gpu = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let n_vocab = on_cpu.hp.n_vocab as usize;
    for (name, text) in PROMPTS {
        let prompt = prompt_ids(&tok, text);
        let mut runs = Vec::new();
        let mut generated: Vec<u32> = Vec::new();
        for (model, backend) in [(&on_cpu, &mut cpu), (&on_gpu, &mut gpu)] {
            let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
            let mut alloc = None;
            let mut rows = vec![
                model
                    .forward(backend, &mut alloc, &mut state, &prompt, false)
                    .unwrap(),
            ];
            for i in 0..10 {
                // The CPU run picks the tokens; the GPU run feeds the same ones.
                if generated.len() == i {
                    generated.push(argmax(rows.last().unwrap()) as u32);
                }
                rows.push(
                    model
                        .forward(backend, &mut alloc, &mut state, &[generated[i]], false)
                        .unwrap(),
                );
            }
            runs.push(rows.concat());
        }
        let a = agreement(&runs[1], &runs[0], n_vocab);
        println!(
            "{name}: decode CUDA vs CPU: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
            a.max_diff, a.top1, a.positions, a.kl_mean, a.kl_max
        );
        assert!(a.kl_mean < 0.015, "{name}: mean KL {}", a.kl_mean);
        assert!(a.kl_max < 0.1, "{name}: max KL {}", a.kl_max);
        assert!(
            a.top1 * 10 >= a.positions * 9,
            "{name}: top-1 {}/{}",
            a.top1,
            a.positions
        );
    }
}
