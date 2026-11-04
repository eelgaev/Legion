//! glm5-next (GLM-5.3-Flash) against llama.cpp, on a reduced-depth copy of the real model
//! (the first layers plus the MTP block; any `--layers N` cut of the GGUF made with a
//! byte-for-byte truncation). References are llama.cpp eval-callback dumps of the same model
//! on the CPU: `<dir>/<call>/<seq>_<name>.bin` (F32, ggml order) with `<dir>/index.txt`.
//!
//! ```sh
//! LEGION_TEST_GLM5=$M LEGION_TEST_GLM5_REF_DIR=$D LEGION_TEST_GLM5_PROMPT="785 6722 ..." \
//!   LEGION_TEST_GLM5_ARGMAX="50349 91535 34788" \
//!   cargo test --release --test glm5next -- --ignored --nocapture --test-threads 1
//! ```

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

fn ids(k: &str) -> Vec<u32> {
    env(k)
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn backend() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(
        std::env::var("LEGION_TEST_THREADS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(32),
    );
    b
}

/// The reference tensors of decode call `call`: name -> (ne, data). Of repeated names the
/// last one is kept (llama.cpp names some intermediates per layer more than once).
fn reference(dir: &str, call: usize) -> Vec<(String, [i64; 4], Vec<f32>)> {
    let index = std::fs::read_to_string(format!("{dir}/index.txt")).unwrap();
    let mut out: Vec<(String, [i64; 4], Vec<f32>)> = Vec::new();
    for line in index.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f[0].parse::<usize>().unwrap() != call {
            continue;
        }
        let (_, name) = f[1].split_once('_').unwrap();
        let ne = [1, 2, 3, 4].map(|i| f[1 + i].parse::<i64>().unwrap());
        let bytes = std::fs::read(format!("{dir}/{call}/{}.bin", f[1])).unwrap();
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        out.retain(|(n, _, _)| n != name);
        out.push((name.to_string(), ne, data));
    }
    out
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Every traced intermediate of the prompt against llama.cpp's (printed: the largest
/// difference relative to the reference's largest value), and the prompt's argmax.
#[test]
#[ignore]
fn prompt_matches_llama_cpp() {
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5")).unwrap();
    let mut backend = backend();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let prompt = ids("LEGION_TEST_GLM5_PROMPT");
    let want = ids("LEGION_TEST_GLM5_ARGMAX");
    let refs = reference(&env("LEGION_TEST_GLM5_REF_DIR"), 0);
    let (logits, taps) = model.prompt_trace(&mut backend, &prompt).unwrap();
    let mut worst = 0f32;
    for t in &taps {
        let name = t.name.strip_suffix("-0").filter(|n| *n == "result_norm");
        let name = name.unwrap_or(&t.name);
        let Some((_, ne, r)) = refs.iter().find(|(n, _, _)| n == name) else {
            println!("{:28} {:?}: no reference", t.name, t.ne);
            continue;
        };
        if r.len() != t.data.len() {
            println!("{:28} {:?} vs reference {:?}", t.name, t.ne, ne);
            continue;
        }
        let scale = r.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
        let d = r
            .iter()
            .zip(&t.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let rel = if d.is_nan() { f32::INFINITY } else { d / scale };
        println!(
            "{:28} {:?}: max |d| {d:.3e} of {scale:.3e} ({rel:.2e})",
            t.name, t.ne
        );
        worst = worst.max(rel);
    }
    let n_vocab = logits.len() / prompt.len();
    let last = &logits[logits.len() - n_vocab..];
    let got = argmax(last);
    println!(
        "argmax {got} ({:.4}), worst relative difference {worst:.2e}",
        last[got as usize]
    );
    assert_eq!(got, want[0]);
}

/// Prompt then greedy decode: the argmax of every step as llama.cpp's.
#[test]
#[ignore]
fn greedy_matches_llama_cpp() {
    let gguf = GgufFile::open(env("LEGION_TEST_GLM5")).unwrap();
    let mut backend = backend();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let prompt = ids("LEGION_TEST_GLM5_PROMPT");
    let want = ids("LEGION_TEST_GLM5_ARGMAX");
    let mut state = model.new_state(backend.buffer_type(), 64).unwrap();
    let mut alloc = None;
    let mut got = vec![argmax(
        &model
            .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
            .unwrap(),
    )];
    while got.len() < want.len() {
        let next = *got.last().unwrap();
        let l = model
            .forward(&mut backend, &mut alloc, &mut state, &[next], false)
            .unwrap();
        got.push(argmax(&l));
    }
    println!("greedy {got:?}");
    assert_eq!(got, want);
}
