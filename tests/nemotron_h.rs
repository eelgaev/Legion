//! Nemotron-H (`nemotron_h_moe`) forward pass against llama.cpp, and CUDA against CPU.
//! Measured with Nemotron 3.5 Lightning 30B-A3B NVFP4 (llama.cpp's own converter output).
//!
//! All tests here are ignored: they need the real model and reference files made with
//! `tools/llama-ref/llama_ref.cpp` (build instructions in its header). Setup, with
//! `D=<reference dir>` and `M=<model.gguf>`:
//!
//! ```sh
//! LEGION_TEST_NEMOTRON=$M LEGION_TEST_NEMOTRON_REF_DIR=$D \
//!   cargo test --release --test nemotron_h write_prompt_ids -- --ignored   # writes $D/<name>.ids
//! for p in $D/*.ids; do n=${p%.ids}
//!   REF_MODE=exact REF_GEN=32 llama_ref $M $p $n.exact.f32
//!   REF_MODE=default          llama_ref $M $p $n.default.f32
//! done
//! LEGION_TEST_NEMOTRON=$M LEGION_TEST_NEMOTRON_REF_DIR=$D \
//!   cargo test --release --test nemotron_h -- --ignored --nocapture --test-threads 1
//! ```
//!
//! `exact` references (llama.cpp with flash attention and CPU weight repacking off, F16 KV
//! cache) run the same ggml kernels as our graph, so logits must agree to the last bit and the
//! greedy continuation (`REF_GEN`, decoded one token at a time) token for token. `default`
//! references (llama.cpp's own CPU defaults) differ by rounding only and are checked
//! statistically, as in `tests/qwen35.rs`.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::tokenizer::Tokenizer;

/// (name, text).
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
    env("LEGION_TEST_NEMOTRON_REF_DIR").into()
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

fn read_ids(path: &std::path::Path) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect(),
    )
}

fn load() -> (GgufFile, Backend) {
    let gguf = GgufFile::open(env("LEGION_TEST_NEMOTRON")).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    (gguf, backend)
}

fn prompt_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, true)
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
        println!("{name}: {} tokens {ids:?}", ids.len());
    }
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
        if let Some(saved) = read_ids(&dir.join(format!("{name}.ids"))) {
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
            // Same kernels on both sides: the bits must match.
            assert_eq!(a.max_diff, 0.0, "{name}: logits differ by {}", a.max_diff);
            checked += 1;
        }
        if let Some(default) = read_f32(&dir.join(format!("{name}.default.f32"))) {
            let a = agreement(&logits, &default, n_vocab);
            println!(
                "  default: max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
                a.max_diff, a.top1, a.positions, a.kl_mean, a.kl_max
            );
            // Rounding drift only (flash attention, repacked weights).
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
    }
    assert!(
        checked > 0,
        "no reference logits found in {}",
        dir.display()
    );
}

/// Greedy decoding (prefill, then one token per step through the recurrent states and the KV
/// cache) against llama.cpp's greedy continuation in exact mode (`REF_GEN`): the same tokens.
#[test]
#[ignore = "needs the real model and llama.cpp reference continuations"]
fn greedy_decode_matches_llama_cpp() {
    let (gguf, mut backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let dir = ref_dir();
    let mut checked = 0;
    let mut alloc = None;
    for (name, text) in PROMPTS {
        let Some(theirs) = read_ids(&dir.join(format!("{name}.exact.f32.gen"))) else {
            continue;
        };
        let prompt = prompt_ids(&tok, text);
        let mut state = model
            .new_state(backend.buffer_type(), prompt.len() + theirs.len() + 1)
            .unwrap();
        let mut logits = model
            .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
            .unwrap();
        let mut ours = Vec::new();
        for _ in 0..theirs.len() {
            let next = argmax(&logits) as u32;
            ours.push(next);
            logits = model
                .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                .unwrap();
        }
        println!("{name}: {:?}", tok.decode(&ours));
        assert_eq!(ours, theirs, "{name}: greedy continuation differs");
        checked += 1;
    }
    assert!(
        checked > 0,
        "no reference continuations in {}",
        dir.display()
    );
}

/// Every traced intermediate against a full-tensor llama.cpp dump (`llama_ref ... <dump_dir>`,
/// exact mode, in `<ref dir>/<prompt>.dump`) for one prompt (LEGION_TEST_NEMOTRON_TRACE_PROMPT,
/// default `france`), printing the first tensor that drifts.
#[test]
#[ignore = "needs the real model and a llama.cpp tensor dump"]
fn trace_against_llama_cpp() {
    let (gguf, mut backend) = load();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let pick = std::env::var("LEGION_TEST_NEMOTRON_TRACE_PROMPT").unwrap_or("france".into());
    let (name, text) = *PROMPTS
        .iter()
        .find(|(n, _)| *n == pick)
        .expect("no such prompt");
    let ids = prompt_ids(&tok, text);
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
        if theirs.len() != t.data.len() {
            println!(
                "{:<28} element count {} vs {}",
                t.name,
                t.data.len(),
                theirs.len()
            );
            continue;
        }
        compared += 1;
        let max_diff = t
            .data
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("{:<28} max |diff| {max_diff:.2e}", t.name);
        if max_diff > 0.0 && first_drift.is_none() {
            first_drift = Some(t.name.clone());
        }
    }
    println!("{compared} tensors compared, first drift: {first_drift:?}");
    assert!(compared > 0, "no traced tensor found in the dump");
    assert!(first_drift.is_none(), "drift starts at {first_drift:?}");
}

/// Prefill + one-token decode steps against one pass over the whole text, with the CPU's
/// reference kernels (every op batch-invariant): the logits after prefill and after each of 8
/// greedy decode steps equal the all-positions logits of the full text, bit for bit. This pins
/// the Mamba-2 conv and SSM state carried between steps.
#[test]
#[ignore = "needs the real model"]
fn decode_matches_full_prompt() {
    let (gguf, mut backend) = load();
    backend.set_use_ref(true);
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
    let mut alloc = None;
    for (name, text) in PROMPTS {
        let prompt = prompt_ids(&tok, text);
        state.reset();
        let mut step_logits = vec![
            model
                .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
                .unwrap(),
        ];
        let mut generated = Vec::new();
        for _ in 0..8 {
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
            "{name}: vs full pass: max |diff| {:.2e}, top-1 {}/{}; {:?}",
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

/// The same prompts on the CUDA backend against the CPU backend
/// (LEGION_TEST_NEMOTRON_CUDA=<model.gguf>, `--features cuda`). Different kernels (CUDA
/// quantizes activations for NVFP4 MMQ/MMVQ, its own reductions), so a statistical check.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real model"]
fn cuda_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let path = env("LEGION_TEST_NEMOTRON_CUDA");
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
    let on_gpu = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let n_vocab = on_cpu.hp.n_vocab as usize;
    for (name, text) in PROMPTS {
        let ids = prompt_ids(&tok, text);
        let reference = on_cpu.prompt_logits_all(&mut cpu, &ids).unwrap();
        let logits = on_gpu.prompt_logits_all(&mut gpu, &ids).unwrap();
        let a = agreement(&logits, &reference, n_vocab);
        println!(
            "{name} ({} tokens): max |diff| {:.2e}, top-1 {}/{}, KL mean {:.2e} max {:.2e}",
            ids.len(),
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

/// Real prompts for the activation-precision comparison: chat turns as the template renders
/// them, and a longer document.
#[cfg(feature = "cuda")]
fn precision_prompts() -> Vec<(&'static str, String)> {
    let chat = |user: &str| {
        format!(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
    };
    let long = [
        "The Apollo program was the third United States human spaceflight program carried out by",
        "the National Aeronautics and Space Administration, which accomplished landing the first",
        "humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's",
        "administration as a three-person spacecraft to follow the one-person Project Mercury, it",
        "was later dedicated to President John F. Kennedy's national goal of landing a man on the",
        "Moon and returning him safely to the Earth by the end of the 1960s. Kennedy's goal was",
        "accomplished on the Apollo 11 mission when astronauts Neil Armstrong and Buzz Aldrin",
        "landed their Apollo Lunar Module on July 20, 1969, and walked on the lunar surface, while",
        "Michael Collins remained in lunar orbit in the command and service module. Apollo ran",
        "from 1961 to 1972, with the first crewed flight in 1968. It encountered a major setback",
        "in 1967 when an Apollo 1 cabin fire killed the entire crew during a prelaunch test.",
        "After the first successful landing, sufficient flight hardware remained for nine follow-on",
        "landings with a plan for extended lunar geological and astrophysical exploration. Budget",
        "cuts forced the cancellation of three of these. Five of the remaining six missions",
        "achieved successful landings, but the Apollo 13 landing was prevented by an oxygen tank",
        "explosion in transit to the Moon, which destroyed the service module's capability to",
        "provide electrical power, crippling the CSM's propulsion and life support systems. The",
        "crew returned to Earth safely by using the lunar module as a lifeboat for these functions.",
        "Apollo used the Saturn family of rockets as launch vehicles, which were also used for an",
        "Apollo Applications Program, which consisted of Skylab, a space station that supported",
        "three crewed missions in 1973 and 1974, and the Apollo-Soyuz Test Project, a joint United",
        "States-Soviet Union low Earth orbit mission in 1975. Summarize the program in three bullet",
        "points and explain why the 1967 fire changed the spacecraft design.",
    ]
    .join(" ");
    vec![
        (
            "chat-capital",
            chat("What is the capital of France? Answer briefly."),
        ),
        (
            "chat-code",
            chat(
                "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
            ),
        ),
        (
            "chat-math",
            chat(
                "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes? Show the steps.",
            ),
        ),
        ("long-doc", chat(&long)),
    ]
}

/// How much the NVFP4 experts' activation precision moves Nemotron 3.5's logits, from its
/// safetensors checkpoint (LEGION_TEST_NEMOTRON_HF=<dir>, `--features cuda`). Reference: the CPU
/// with every NVFP4 weight widened to its exact BF16 values, so only activations round, to
/// BF16 (the checkpoint's W4A16 semantics). Compared: CUDA as llama.cpp runs it (experts not
/// in `prec_a4` take 4-bit activations in batched matmuls), CUDA with 8-bit activations
/// everywhere (`keep_a8`), and the CPU's Q8_0 activations. KL(reference || run) per position,
/// over every prompt position, and top-1 agreement. LEGION_TEST_NEMOTRON_OUT=<file> also
/// writes the table as JSON.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the real checkpoint"]
fn nvfp4_activation_precision() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let dir = env("LEGION_TEST_NEMOTRON_HF");
    let gguf = GgufFile::open(&dir).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let prompts: Vec<(&str, Vec<u32>)> = precision_prompts()
        .into_iter()
        .map(|(n, t)| (n, tok.encode_with(&t, false, true)))
        .collect();

    // GPU runs first (then freed), the CPU reference after: unified memory holds one at a time.
    let mut runs: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
    {
        let mut gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU")
            .init()
            .unwrap();
        let mut model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
        for (label, a8) in [
            ("cuda-w4a4 (llama.cpp default)", false),
            ("cuda-a8 (keep_a8)", true),
        ] {
            model.keep_a8(a8);
            let rows = prompts
                .iter()
                .map(|(_, ids)| model.prompt_logits_all(&mut gpu, ids).unwrap())
                .collect();
            runs.push((label, rows));
        }
    }
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let q8: Vec<Vec<f32>> = prompts
        .iter()
        .map(|(_, ids)| model.prompt_logits_all(&mut cpu, ids).unwrap())
        .collect();
    runs.push(("cpu-q8_0 activations", q8));
    let n = model
        .widen_nvfp4(cpu.buffer_type(), &gguf.tensors())
        .unwrap();
    println!("reference: {n} NVFP4 tensors widened to BF16");
    let reference: Vec<Vec<f32>> = prompts
        .iter()
        .map(|(_, ids)| model.prompt_logits_all(&mut cpu, ids).unwrap())
        .collect();
    let n_vocab = model.hp.n_vocab as usize;

    let mut json = Vec::new();
    for (label, rows) in &runs {
        let mut all_kl = Vec::new();
        let (mut top1, mut positions) = (0, 0);
        for ((name, _), (ours, theirs)) in prompts.iter().zip(rows.iter().zip(&reference)) {
            let a = agreement(ours, theirs, n_vocab);
            println!(
                "{label:<30} {name:<13} KL mean {:.2e} max {:.2e}, top-1 {}/{}",
                a.kl_mean, a.kl_max, a.top1, a.positions
            );
            all_kl.push((a.kl_mean * a.positions as f64, a.kl_max));
            top1 += a.top1;
            positions += a.positions;
        }
        let kl_mean = all_kl.iter().map(|k| k.0).sum::<f64>() / positions as f64;
        let kl_max = all_kl.iter().map(|k| k.1).fold(0f64, f64::max);
        println!(
            "{label:<30} all           KL mean {kl_mean:.2e} max {kl_max:.2e}, top-1 {top1}/{positions}"
        );
        json.push(serde_json::json!({
            "run": label, "kl_mean": kl_mean, "kl_max": kl_max,
            "top1": top1, "positions": positions,
        }));
        assert!(kl_mean.is_finite());
    }
    if let Ok(out) = std::env::var("LEGION_TEST_NEMOTRON_OUT") {
        std::fs::write(out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}
