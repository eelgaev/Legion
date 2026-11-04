//! How the kernels of NVFP4 matmuls (`--nvfp4-mm`: their activation precision) move a real
//! checkpoint's logits on CUDA, against a high-precision reference. Ignored: needs a GPU
//! (`--features cuda`) and a ModelOpt NVFP4 (+ FP8) checkpoint directory:
//!
//! ```sh
//! LEGION_TEST_NVFP4_HF=<checkpoint dir> [LEGION_TEST_NVFP4_OUT=<file.json>] \
//!   cargo test --release --features cuda --test nvfp4_act -- --ignored --nocapture
//! ```
//!
//! Reference: the CPU with every NVFP4 weight widened to its exact BF16 values and the FP8
//! weights through the W8A16 op with F32 activations (exact weights, BF16-or-better
//! activations), as in `tests/fp8_act.rs`. Compared on CUDA, each with the FP8 layers at W8A16
//! (so the NVFP4 matmuls' share stands alone) and at W8A8 (the default):
//! - `ggml`: ggml's MMVQ / MMQ, Q8_1 activations everywhere (before Legion's W4A16 kernels);
//! - `auto` (the default): Legion's W4A16 kernels for batches of 9+ tokens (here: the prefill),
//!   ggml's MMVQ for the one-token decode steps;
//! - `legion`: W4A16 everywhere;
//! - `+a4 experts` / `+a4 prefill` / `+a4 all` (`--nvfp4-a4`, GPUs with the FP4 kernels):
//!   4-bit activations on the W4A4 kernels for the routed experts' prefill batches, every
//!   prefill batch (more than 64 tokens: here the long prompt's), or every matmul on Legion's
//!   kernels (with `legion`: the decode steps too).
//!
//! Each prompt's last tokens run as one-token decode steps (teacher forced), the rest as one
//! prefill, so prefill and decode positions are reported apart. KL(reference || run) per
//! position and top-1 agreement.

#[cfg(feature = "cuda")]
mod cuda {
    use legion::ggml::{Backend, DeviceKind, device_by_kind};
    use legion::gguf::GgufFile;
    use legion::model::core::{Fp8Act, Model, Nvfp4A4, Nvfp4Mm};
    use legion::tokenizer::Tokenizer;

    /// Decode steps at the end of each prompt.
    const DECODE: usize = 16;

    fn prompts() -> Vec<(&'static str, String)> {
        let chat = |user: &str| {
            format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        };
        let long = [
            "The Apollo program was the third United States human spaceflight program carried",
            "out by the National Aeronautics and Space Administration, which accomplished landing",
            "the first humans on the Moon from 1969 to 1972. First conceived during Dwight D.",
            "Eisenhower's administration as a three-person spacecraft to follow the one-person",
            "Project Mercury, it was later dedicated to President John F. Kennedy's national goal",
            "of landing a man on the Moon and returning him safely to the Earth by the end of the",
            "1960s. Kennedy's goal was accomplished on the Apollo 11 mission when astronauts Neil",
            "Armstrong and Buzz Aldrin landed their Apollo Lunar Module on July 20, 1969, and",
            "walked on the lunar surface, while Michael Collins remained in lunar orbit in the",
            "command and service module. Apollo ran from 1961 to 1972, with the first crewed",
            "flight in 1968. It encountered a major setback in 1967 when an Apollo 1 cabin fire",
            "killed the entire crew during a prelaunch test. After the first successful landing,",
            "sufficient flight hardware remained for nine follow-on landings with a plan for",
            "extended lunar geological and astrophysical exploration. Budget cuts forced the",
            "cancellation of three of these. Five of the remaining six missions achieved",
            "successful landings, but the Apollo 13 landing was prevented by an oxygen tank",
            "explosion in transit to the Moon. Summarize the program in three bullet points and",
            "explain why the 1967 fire changed the spacecraft design.",
        ]
        .join(" ");
        vec![
            (
                "chat-capital",
                chat("What is the capital of France? Answer briefly."),
            ),
            (
                "chat-code",
                chat("Write a Python function that returns the n-th Fibonacci number iteratively."),
            ),
            (
                "chat-math",
                chat(
                    "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?",
                ),
            ),
            ("long-doc", chat(&long)),
        ]
    }

    fn log_softmax(x: &[f32]) -> Vec<f64> {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
        x.iter().map(|&v| v as f64 - lse).collect()
    }

    fn argmax(x: &[f32]) -> usize {
        (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
    }

    /// (sum of KL, max KL, top-1 matches, positions) of `rows` against `reference`.
    #[derive(Default, Clone, Copy)]
    struct Stats {
        kl_sum: f64,
        kl_max: f64,
        top1: usize,
        n: usize,
    }

    impl Stats {
        fn add(&mut self, ours: &[f32], theirs: &[f32]) {
            let (lo, lt) = (log_softmax(ours), log_softmax(theirs));
            let kl: f64 = lt.iter().zip(&lo).map(|(t, o)| t.exp() * (t - o)).sum();
            self.kl_sum += kl;
            self.kl_max = self.kl_max.max(kl);
            self.top1 += (argmax(ours) == argmax(theirs)) as usize;
            self.n += 1;
        }

        fn json(self) -> serde_json::Value {
            serde_json::json!({
                "kl_mean": self.kl_sum / self.n as f64, "kl_max": self.kl_max,
                "top1": self.top1, "positions": self.n,
            })
        }

        fn show(self) -> String {
            format!(
                "KL mean {:.2e} max {:.2e}, top-1 {}/{}",
                self.kl_sum / self.n as f64,
                self.kl_max,
                self.top1,
                self.n
            )
        }
    }

    #[test]
    #[ignore = "needs a GPU and a ModelOpt NVFP4 checkpoint"]
    fn nvfp4_activation_precision() {
        let dir = std::env::var("LEGION_TEST_NVFP4_HF").expect("LEGION_TEST_NVFP4_HF is not set");
        let gguf = GgufFile::open(&dir).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let prompts: Vec<(&str, Vec<u32>)> = prompts()
            .into_iter()
            .map(|(n, t)| (n, tok.encode_with(&t, false, true)))
            .collect();

        // GPU runs first (then freed), the CPU reference after.
        let a4 = legion::model::nvfp4_a4::device_supported();
        let mut modes = Vec::new();
        for (fp8, w) in [(Fp8Act::A16, "w8a16"), (Fp8Act::A8, "w8a8")] {
            for (name, mm, a4) in [
                ("ggml", Nvfp4Mm::Ggml, Nvfp4A4::Off),
                ("auto", Nvfp4Mm::Auto, Nvfp4A4::Off),
                ("legion", Nvfp4Mm::Legion, Nvfp4A4::Off),
            ]
            .into_iter()
            .chain(
                a4.then_some([
                    ("auto+a4 experts", Nvfp4Mm::Auto, Nvfp4A4::Experts),
                    ("auto+a4 prefill", Nvfp4Mm::Auto, Nvfp4A4::Prefill),
                    ("legion+a4 all", Nvfp4Mm::Legion, Nvfp4A4::All),
                ])
                .into_iter()
                .flatten(),
            ) {
                modes.push((format!("{name}, fp8 {w}"), mm, a4, fp8));
            }
        }
        let mut runs: Vec<(String, Vec<Vec<Vec<f32>>>)> = Vec::new();
        {
            let mut gpu = device_by_kind(DeviceKind::Gpu)
                .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
                .expect("no GPU")
                .init()
                .unwrap();
            let mut model = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
            model.keep_a8(true);
            let n_vocab = model.hp.n_vocab as usize;
            for (label, mm, a4, fp8) in modes {
                model.set_nvfp4_mm(mm);
                model.set_nvfp4_a4(a4);
                model.set_fp8_act(fp8);
                let mut per_prompt = Vec::new();
                for (_, ids) in &prompts {
                    let split = ids.len() - DECODE.min(ids.len() / 2);
                    let mut state = model.new_state(gpu.buffer_type(), ids.len() + 1).unwrap();
                    let mut alloc = None;
                    let first = model
                        .forward(&mut gpu, &mut alloc, &mut state, &ids[..split], true)
                        .unwrap();
                    let mut rows: Vec<Vec<f32>> =
                        first.chunks(n_vocab).map(|r| r.to_vec()).collect();
                    for &t in &ids[split..] {
                        rows.push(
                            model
                                .forward(&mut gpu, &mut alloc, &mut state, &[t], false)
                                .unwrap(),
                        );
                    }
                    per_prompt.push(rows);
                }
                runs.push((label, per_prompt));
            }
        }
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let mut model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
        model.set_fp8_act(Fp8Act::A16);
        let widened = model
            .widen_nvfp4(cpu.buffer_type(), &gguf.tensors())
            .unwrap();
        println!("reference: {widened} NVFP4 tensors widened to BF16, FP8 at F32 activations");
        let n_vocab = model.hp.n_vocab as usize;
        let reference: Vec<Vec<f32>> = prompts
            .iter()
            .map(|(_, ids)| model.prompt_logits_all(&mut cpu, ids).unwrap())
            .collect();

        let mut json = Vec::new();
        for (label, per_prompt) in &runs {
            let (mut prefill, mut decode) = (Stats::default(), Stats::default());
            for (((name, ids), rows), theirs) in prompts.iter().zip(per_prompt).zip(&reference) {
                let split = ids.len() - DECODE.min(ids.len() / 2);
                let mut s = (Stats::default(), Stats::default());
                for (p, row) in rows.iter().enumerate() {
                    let t = &theirs[p * n_vocab..(p + 1) * n_vocab];
                    if p < split {
                        s.0.add(row, t);
                        prefill.add(row, t);
                    } else {
                        s.1.add(row, t);
                        decode.add(row, t);
                    }
                }
                println!(
                    "{label:<28} {name:<13} prefill {} | decode {}",
                    s.0.show(),
                    s.1.show()
                );
            }
            println!(
                "{label:<28} all           prefill {} | decode {}",
                prefill.show(),
                decode.show()
            );
            json.push(serde_json::json!({
                "run": label, "prefill": prefill.json(), "decode": decode.json(),
            }));
            assert!(prefill.kl_sum.is_finite() && decode.kl_sum.is_finite());
        }
        if let Ok(out) = std::env::var("LEGION_TEST_NVFP4_OUT") {
            std::fs::write(out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        }
    }
}
