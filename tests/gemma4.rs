//! Gemma 4 forward pass (`gemma4`: E2B / E4B with per-layer embeddings and shared KV, the dense
//! 31B, the 26B-A4B MoE) against llama.cpp, and CUDA against CPU.
//!
//! All tests here are ignored: they need a real model and reference files made with
//! `tools/llama-ref/llama_ref.cpp` (build instructions in its header). Setup, with
//! `D=<reference dir>`, `M=<model.gguf>` and `A=<its MTP assistant gguf>`:
//!
//! ```sh
//! LEGION_TEST_GEMMA4=$M LEGION_TEST_GEMMA4_REF_DIR=$D \
//!   cargo test --release --test gemma4 write_prompt_ids -- --ignored   # writes $D/<name>.ids
//! for p in $D/*.ids; do n=${p%.ids}
//!   REF_MODE=exact   llama_ref $M $p $n.exact.f32
//!   REF_MODE=default llama_ref $M $p $n.default.f32
//! done
//! mkdir $D/france.dump && REF_MODE=exact llama_ref $M $D/france.ids /dev/null $D/france.dump '-[0-9]+$'
//! for n in france apollo; do REF_LAST=1 REF_ASSISTANT=$A llama_ref $M $D/$n.ids $D/$n.mtp; done
//! LEGION_TEST_GEMMA4=$M LEGION_TEST_GEMMA4_MTP=$A LEGION_TEST_GEMMA4_REF_DIR=$D \
//!   cargo test --release --test gemma4 -- --ignored --nocapture --test-threads 1
//! ```
//!
//! The `long` prompt is longer than every Gemma 4 sliding window (512 in E2B, 1024 in the big
//! models), so its later positions only match llama.cpp if the sliding-window mask is right.
//! `exact` references (flash attention and weight repacking off) run the same ggml kernels as
//! Legion's graph, so the logits agree to the last bit; `default` ones (llama.cpp's CPU flash
//! attention, repacked weights) differ by rounding that Q8_0 activation quantization amplifies,
//! and are checked statistically (see `tests/qwen35.rs` for the details of both).

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::tokenizer::Tokenizer;

const APOLLO: &str = "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the \
    lunar module Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled \
    the goal President Kennedy had set eight years earlier, and the next mission";

/// (name, text).
fn prompts() -> Vec<(&'static str, String)> {
    // Past the largest sliding window (1024): numbered paragraphs, so no two are the same.
    let mut long = String::new();
    for i in 1..=13 {
        long += &format!(
            "Section {i}. The rivers of the northern valley flood every spring, and the farmers \
             of village number {i} move their sheep to the hills, count them twice, and write \
             the totals in a ledger kept since the year {}. ",
            1700 + 7 * i
        );
        long += APOLLO;
        long += " was planned. ";
    }
    long += "The ledger of village number 4 says that in the year";
    vec![
        ("france", "The capital of France is".to_string()),
        ("apollo", APOLLO.to_string()),
        ("long", long),
    ]
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn ref_dir() -> std::path::PathBuf {
    env("LEGION_TEST_GEMMA4_REF_DIR").into()
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
    let gguf = GgufFile::open(env("LEGION_TEST_GEMMA4")).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    (gguf, backend)
}

fn prompt_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    // Gemma 4 GGUFs differ in `add_bos_token`; the model always expects <bos> first.
    let mut ids = tok.encode(text, true);
    if let Some(bos) = tok.bos_id()
        && ids.first() != Some(&bos)
    {
        ids.insert(0, bos);
    }
    ids
}

/// Writes `<ref dir>/<name>.ids` for every prompt, the input for `llama_ref`.
#[test]
#[ignore = "needs the real model"]
fn write_prompt_ids() {
    let (gguf, _) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let dir = ref_dir();
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in prompts() {
        let ids = prompt_ids(&tok, &text);
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
    for (name, text) in prompts() {
        let ids = prompt_ids(&tok, &text);
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
            // Rounding drift only. Measured (mean / max KL, france / apollo): E2B Q8_0
            // 1.4e-3 / 6.2e-3 and 6.4e-3 / 7.9e-2; 26B-A4B Q4_0 1.6e-2 / 3.7e-2 and
            // 3.6e-2 / 0.71 (one position where a routing flip moves the top experts). A wrong
            // mask, RoPE or routing rule moves every position, the mean to O(1).
            assert!(a.kl_mean < 0.05, "{name}: mean KL {}", a.kl_mean);
            assert!(a.kl_max < 1.0, "{name}: max KL {}", a.kl_max);
            assert!(
                a.top1 * 10 >= a.positions * 8,
                "{name}: top-1 agrees at {}/{} positions",
                a.top1,
                a.positions
            );
            checked += 1;
        }
        if name == "france" {
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
/// exact mode) for one prompt (LEGION_TEST_GEMMA4_TRACE_PROMPT, default `france`), printing the
/// first tensor that drifts.
#[test]
#[ignore = "needs the real model and a llama.cpp tensor dump"]
fn trace_against_llama_cpp() {
    let (gguf, mut backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let pick = std::env::var("LEGION_TEST_GEMMA4_TRACE_PROMPT").unwrap_or("france".into());
    let (name, text) = prompts()
        .into_iter()
        .find(|(n, _)| *n == pick)
        .expect("no such prompt");
    let ids = prompt_ids(&tok, &text);
    let (_, traced) = model.prompt_trace(&mut backend, &ids).unwrap();

    // llama.cpp names a tensor several times while building it; the last one is the finished
    // value.
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
    let mut compared = 0;
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
        compared += 1;
    }
    assert!(compared > 0, "no traced tensor is in the dump");
    assert!(first_drift.is_none(), "drift starts at {first_drift:?}");
}

/// Prefill + one-token decode steps against one pass over the whole text, with the CPU's
/// reference kernels (batch-invariant): the logits after prefill and after each of 10 greedy
/// steps equal the full pass's at the same positions, bit for bit. Covers the decode path of
/// the sliding-window mask and the shared-KV layers.
#[test]
#[ignore = "needs the real model"]
fn decode_matches_full_prompt() {
    let (gguf, mut backend) = load();
    backend.set_use_ref(true);
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let mut state = model.new_state(backend.buffer_type(), 2048).unwrap();
    let mut alloc = None;
    for (name, text) in prompts() {
        if name == "long" && std::env::var("LEGION_TEST_GEMMA4_SHORT").is_ok() {
            continue;
        }
        let prompt = prompt_ids(&tok, &text);
        state.reset();
        let mut step_logits = vec![
            model
                .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
                .unwrap(),
        ];
        let mut generated = Vec::new();
        for _ in 0..10 {
            let next = argmax(step_logits.last().unwrap()) as u32;
            generated.push(next);
            step_logits.push(
                model
                    .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                    .unwrap(),
            );
        }
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
            "{name}: {} prompt tokens; decode vs full pass: max |diff| {:.2e}, top-1 {}/{}; {:?}",
            prompt.len(),
            a.max_diff,
            a.top1,
            a.positions,
            tok.decode(&generated)
        );
        assert!(
            a.max_diff <= 1e-4,
            "{name}: logits differ by {}",
            a.max_diff
        );
        assert_eq!(a.top1, a.positions, "{name}: top-1 differs");
    }
}

/// Continuous batching changes nothing a sequence computes: with the CPU's reference kernels,
/// three prompts prefilled and decoded together (two prompts as separate groups, then a
/// two-sequence decode next to the third prompt's prefill, then three-sequence decode steps;
/// KV blocks interleaved across the sequences) give the same logits, bit for bit, as each
/// prompt run alone.
#[test]
#[ignore = "needs the real model"]
fn batched_decode_matches_alone() {
    use legion::model::core::{Executor, Group, KV_BLOCK, Logits, SeqInput};
    const STEPS: usize = 6;
    let (gguf, mut backend) = load();
    backend.set_use_ref(true);
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let texts = [
        "The capital of France is",
        APOLLO,
        "Write a haiku about the sea.",
    ];
    let prompts: Vec<Vec<u32>> = texts.iter().map(|t| prompt_ids(&tok, t)).collect();

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
    drop(model);

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
        assert_eq!(out.argmax[k] as usize, argmax(&out.rows[k]));
        out.rows[k].clone()
    };
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

/// The prompts on the CUDA backend against the CPU backend (LEGION_TEST_GEMMA4_CUDA=<gguf>).
/// Different kernels, so statistical.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let path = env("LEGION_TEST_GEMMA4_CUDA");
    let gguf = GgufFile::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let on_cpu = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let on_gpu = Model::load(&gguf, |i| {
        if Model::placement(&i.name, false) == legion::model::core::Placement::Host {
            cpu.buffer_type()
        } else {
            gpu.buffer_type()
        }
    })
    .unwrap();
    let n_vocab = on_cpu.hp.n_vocab as usize;
    for (name, text) in prompts() {
        let ids = prompt_ids(&tok, &text);
        let reference = on_cpu.prompt_logits_all(&mut cpu, &ids).unwrap();
        on_gpu.prompt_logits_all(&mut gpu, &ids).unwrap();
        let t = std::time::Instant::now();
        let logits = on_gpu.prompt_logits_all(&mut gpu, &ids).unwrap();
        let gpu_s = t.elapsed().as_secs_f64();
        let a = agreement(&logits, &reference, n_vocab);
        println!(
            "{name} ({} tokens): GPU {:.1} ms; max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} \
             max {:.2e}",
            ids.len(),
            gpu_s * 1e3,
            a.max_diff,
            a.top1,
            a.positions,
            a.kl_mean,
            a.kl_max
        );
        // llama.cpp's CUDA run on the same GGUF (`REF_GPU=1 llama_ref ... <name>.cuda.f32`),
        // when the reference directory has it: the two CUDA runs agree with each other.
        let cuda_ref = std::env::var("LEGION_TEST_GEMMA4_REF_DIR")
            .ok()
            .and_then(|d| read_f32(&std::path::Path::new(&d).join(format!("{name}.cuda.f32"))));
        if let Some(theirs) = cuda_ref {
            let c = agreement(&logits, &theirs, n_vocab);
            println!(
                "  vs llama.cpp CUDA: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                c.max_diff, c.top1, c.positions, c.kl_mean, c.kl_max
            );
            // The 26B-A4B on `long` (13 near-identical paragraphs: near-tied keys under
            // unscaled attention, so rounding flips which one a query attends to): every pair
            // of CPU, llama.cpp CUDA and Legion CUDA runs is 0.26-0.31 mean KL apart.
            assert!(
                c.kl_mean < 0.5,
                "{name}: mean KL vs llama.cpp CUDA {}",
                c.kl_mean
            );
        }
        // Against the CPU: rounding drift. CUDA quantizes activations to Q8_1 (the CPU to Q8_0)
        // and sums in other orders, and the QAT Q4_0 models amplify that. llama.cpp's own CUDA
        // run is as far from its CPU run (mean KL, llama.cpp vs Legion): 26B-A4B `france` 0.13
        // vs 0.155, `long` 0.316 (top-1 1036 of 1373) vs 0.307 (1034); 31B `france` 0.026 vs
        // 0.066; E2B 1.3e-3 vs 2.1e-3. A wrong mask or rule moves KL to O(1) everywhere.
        assert!(a.kl_mean < 0.5, "{name}: mean KL {}", a.kl_mean);
        // Top-1 over a few near-uniform early positions is noise (26B `france`: 3 of 6 against
        // the CPU and against llama.cpp's CUDA run alike); the KL above bounds those.
        assert!(
            a.positions < 20 || a.top1 * 10 >= a.positions * 6,
            "{name}: top-1 agrees at {}/{} positions",
            a.top1,
            a.positions
        );
    }
}

/// Every traced intermediate on CUDA against the CPU (LEGION_TEST_GEMMA4_CUDA, the prompt
/// LEGION_TEST_GEMMA4_TRACE_PROMPT, default `france`), printed in order, to see where the backends part (diagnostic: never fails).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_trace_vs_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let gguf = GgufFile::open(env("LEGION_TEST_GEMMA4_CUDA")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let pick = std::env::var("LEGION_TEST_GEMMA4_TRACE_PROMPT").unwrap_or("france".into());
    let (_, text) = prompts()
        .into_iter()
        .find(|(n, _)| *n == pick)
        .expect("no such prompt");
    let ids = prompt_ids(&tok, &text);
    let on_cpu = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let (_, a) = on_cpu.prompt_trace(&mut cpu, &ids).unwrap();
    drop(on_cpu);
    let on_gpu = Model::load(&gguf, |i| {
        if Model::placement(&i.name, false) == legion::model::core::Placement::Host {
            cpu.buffer_type()
        } else {
            gpu.buffer_type()
        }
    })
    .unwrap();
    let (_, b) = on_gpu.prompt_trace(&mut gpu, &ids).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.name, y.name);
        let scale = x.data.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let d = x
            .data
            .iter()
            .zip(&y.data)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        println!("{:<24} rel {:.2e}", x.name, d / scale);
    }
}

/// The MTP assistant (LEGION_TEST_GEMMA4_MTP=<gemma4-assistant gguf>) against llama.cpp's
/// draft steps (`REF_ASSISTANT=.. llama_ref <model> <ref dir>/<name>.ids <ref dir>/<name>.mtp`,
/// exact mode): after each prompt, the same tokens fed at the same position (the target's
/// greedy token first, with its residual of the last prompt position; each draft after, with
/// the assistant's own output residual). Same kernels on both sides, so the greedy drafts, their
/// probabilities and the residuals agree to the last bit (measured: 0.0 on E2B, 26B-A4B, 31B).
#[test]
#[ignore = "needs the real model, its assistant and llama.cpp references"]
fn assistant_matches_llama_cpp() {
    use legion::model::core::{Executor, Group, KV_BLOCK, Logits, SeqInput};
    let (gguf, backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let assistant = GgufFile::open(env("LEGION_TEST_GEMMA4_MTP")).unwrap();
    model
        .load_mtp(&assistant, |_| backend.buffer_type())
        .unwrap();
    let (n_layer, n_vocab, n_embd) = (
        model.hp.n_layer,
        model.hp.n_vocab as usize,
        model.hp.n_embd as usize,
    );
    let n_blocks = 2048 / KV_BLOCK;
    let mut exec = Executor::new(model, backend, n_blocks, 1, 0).unwrap();
    exec.enable_mtp().unwrap();
    exec.capture_hidden(vec![n_layer - 1]);
    let blocks: Vec<u32> = (0..n_blocks as u32).collect();
    let dir = ref_dir();
    let mut checked = 0;
    for (name, text) in prompts() {
        let file = |ext: &str| dir.join(format!("{name}.mtp.{ext}"));
        let Some(logits) = read_f32(&file("draft.f32")) else {
            continue;
        };
        let theirs_h = read_f32(&file("draft_h.f32")).unwrap();
        let fed: Vec<u32> = std::fs::read_to_string(file("draft.ids"))
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let ids = prompt_ids(&tok, &text);
        let seq = |tokens, n_past| Group {
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
        let out = exec.step(&[seq(&ids, 0)]).unwrap();
        assert_eq!(out.argmax[0], fed[0], "{name}: the target's next token");
        let mut h = out.hidden[0][(ids.len() - 1) * n_embd..].to_vec();
        for (s, t) in fed.iter().enumerate() {
            let o = exec
                .mtp_step(&[seq(std::slice::from_ref(t), ids.len())], &h)
                .unwrap();
            let row = &logits[s * n_vocab..(s + 1) * n_vocab];
            let want = argmax(row) as u32;
            let p_theirs = log_softmax(row)[want as usize].exp() as f32;
            let th = &theirs_h[s * n_embd..(s + 1) * n_embd];
            let scale = th.iter().fold(0f32, |m, x| m.max(x.abs()));
            let dh = o
                .hidden
                .iter()
                .zip(th)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "{name} step {s}: draft {} (llama.cpp {want}), p {:.4} ({p_theirs:.4}), residual \
                 max |diff| {dh:.2e}",
                o.argmax[0], o.pmax[0]
            );
            assert_eq!(o.argmax[0], want, "{name} step {s}: draft");
            assert!(
                (o.pmax[0] - p_theirs).abs() < 1e-5,
                "{name} step {s}: draft probability {} vs {p_theirs}",
                o.pmax[0]
            );
            // The bound only absorbs libm differences.
            assert!(
                dh <= 1e-4 * scale,
                "{name} step {s}: residual differs by {dh}"
            );
            h = o.hidden;
            checked += 1;
        }
    }
    assert!(checked > 0, "no assistant references in {}", dir.display());
}
