//! MTP speculative decoding on the tiny random qwen35 (gated DeltaNet + gated attention) with
//! its MTP head (#397), in the model's file (as the HF converter writes Qwen3.5's) and in a GGUF
//! of its own, on the CPU with reference kernels: greedy output with `--spec mtp` is
//! bit-identical to plain decoding, for every draft length, several sequences batched, and the
//! head drafts (the verify path's DeltaNet replay is the one qwen4exp uses).

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::{SpecConfig, SpecCounters, SpecKind};
use tokio_util::sync::CancellationToken;

fn prompts() -> Vec<String> {
    vec![
        "Multi-token prediction drafts the next tokens. ".repeat(3),
        "speculative decoding".into(),
        "0123456789".repeat(5),
        "abc".into(),
    ]
}

fn engine(dir: &std::path::Path, spec: SpecConfig) -> LmEngine {
    engine_of(dir, "tiny.gguf", spec)
}

fn engine_of(dir: &std::path::Path, model: &str, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        max_batch_tokens: 48,
        spec,
        use_ref: true,
        ..LmConfig::new(dir.join(model), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// `--spec mtp` with the head in the model's file (`draft` None) or in a file of its own.
fn mtp(draft: Option<std::path::PathBuf>, k: usize, p_min: f32) -> SpecConfig {
    SpecConfig {
        kind: SpecKind::Mtp,
        k,
        draft,
        p_min,
        ..Default::default()
    }
}

#[derive(Debug, PartialEq)]
struct Reply {
    text: String,
    reason: FinishReason,
    tokens: u32,
    spec: Option<SpecCounters>,
}

async fn run_all(engine: &LmEngine, sampling: SamplingParams) -> Vec<Reply> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| {
            engine
                .generate(GenerateRequest {
                    model: "tiny".into(),
                    prompt: Prompt::Text(p.clone()),
                    sampling: sampling.clone(),
                    max_tokens: 40,
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
    let mut replies = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended without Finished") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { reason, usage, .. } => {
                    replies.push(Reply {
                        text,
                        reason,
                        tokens: usage.completion_tokens,
                        spec: usage.spec,
                    });
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
    }
    replies
}

/// The tiny model without a head (`tiny.gguf`), with it (`tiny-mtp.gguf`), and the head alone
/// (`mtp.gguf`).
fn write_models(dir: &std::path::Path) {
    let cfg = TinyConfig {
        seed: 7,
        ..Default::default()
    };
    tiny::write(dir.join("tiny.gguf"), &cfg).unwrap();
    tiny::write(dir.join("tiny-mtp.gguf"), &TinyConfig { mtp: true, ..cfg }).unwrap();
    tiny::write_mtp_head(dir.join("mtp.gguf"), &cfg).unwrap();
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen35_mtp_greedy_output_is_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    assert!(plain.iter().all(|r| r.tokens > 5), "{plain:?}");
    // The head in the model's file must not change plain decoding either.
    let plain_with_head = run_all(
        &engine_of(dir.path(), "tiny-mtp.gguf", SpecConfig::default()),
        greedy(),
    )
    .await;
    assert_eq!(plain, plain_with_head);
    for k in 1..=4 {
        for (model, draft) in [
            ("tiny-mtp.gguf", None),
            ("tiny.gguf", Some(dir.path().join("mtp.gguf"))),
        ] {
            let e = engine_of(dir.path(), model, mtp(draft.clone(), k, 0.0));
            let replies = run_all(&e, greedy()).await;
            for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
                assert_eq!(
                    (&a.text, a.reason, a.tokens),
                    (&b.text, b.reason, b.tokens),
                    "k {k}, {model}: prompt {i} differs"
                );
            }
            let s = e.stats().spec;
            println!("k {k} {model}: {s:?}");
            assert!(s.drafted > 0, "k {k}, {model}: nothing drafted");
        }
    }
}

/// The head in the model's file and the same head in a file of its own draft the same tokens.
#[tokio::test(flavor = "multi_thread")]
async fn qwen35_mtp_head_files_draft_alike() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    // A fixed depth: the adaptive one follows measured timings.
    let fixed = |draft| SpecConfig {
        fixed: true,
        ..mtp(draft, 3, 0.0)
    };
    let a = engine_of(dir.path(), "tiny-mtp.gguf", fixed(None));
    let b = engine_of(
        dir.path(),
        "tiny.gguf",
        fixed(Some(dir.path().join("mtp.gguf"))),
    );
    let (ra, rb) = (run_all(&a, greedy()).await, run_all(&b, greedy()).await);
    assert_eq!(ra, rb);
    assert_eq!(a.stats().spec, b.stats().spec);
}

fn executor(dir: &std::path::Path) -> legion::model::core::Executor {
    executor_with(dir, false)
}

/// The tiny model with its head on the CPU, the head's matrices as Q8_0 copies with `requant`.
fn executor_with(dir: &std::path::Path, requant: bool) -> legion::model::core::Executor {
    use legion::ggml::Backend;
    use legion::gguf::GgufFile;
    use legion::model::core::{Executor, Model};
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    let mut m = Model::load(&GgufFile::open(dir.join("tiny-mtp.gguf")).unwrap(), |_| {
        b.buffer_type()
    })
    .unwrap();
    if requant {
        let (n, _, _) = m.requant_mtp(legion::ggml::types::Q8_0).unwrap();
        assert!(n > 0, "no matrix requantized");
    }
    let n_layer = m.hp.n_layer;
    let mut e = Executor::new(m, b, 4, 1, 0).unwrap();
    e.enable_mtp().unwrap();
    e.capture_hidden(vec![n_layer - 1]);
    e
}

fn group<'a>(
    tokens: &'a [u32],
    n_past: usize,
    blocks: &'a [u32],
    logits: legion::model::core::Logits,
) -> legion::model::core::Group<'a> {
    legion::model::core::Group {
        slot: 0,
        seqs: vec![legion::model::core::SeqInput {
            tokens,
            n_past,
            blocks,
            ring: 0,
            logits,
            media: None,
        }],
        verify: false,
    }
}

/// The head against a plain f32 reimplementation of Qwen3-Next's MTP layer (vLLM's
/// `qwen3_next_mtp.py`) at position 0, where attention over the one key is its value: the
/// draft and the output hidden come out the same (to rounding).
#[test]
fn qwen35_mtp_head_matches_a_reference() {
    use legion::model::core::Logits;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let mut e = executor(dir.path());
    let m = e.model();
    let hp = &m.hp;
    let (n, eps) = (hp.n_embd as usize, hp.rms_eps);
    let get = |name: &str| m.tensor(name).unwrap_or_else(|| panic!("{name}")).to_f32();
    let blk = |name: &str| get(&format!("blk.{}.{name}", hp.n_layer));
    let rms = |x: &[f32], g: &[f32]| -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter().zip(g).map(|(v, g)| v * s * g).collect()
    };
    // y = W x for a ggml weight of [k, rows] (rows of k).
    let mv = |w: &[f32], x: &[f32]| -> Vec<f32> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());

    let tok = 42u32;
    let h: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
        .collect();
    let emb = get("token_embd.weight");
    let en = rms(&emb[tok as usize * n..][..n], &blk("nextn.enorm.weight"));
    let hn = rms(&h, &blk("nextn.hnorm.weight"));
    let cat: Vec<f32> = en.iter().chain(&hn).copied().collect();
    let cur = mv(&blk("nextn.eh_proj.weight"), &cat);
    let x = rms(&cur, &blk("attn_norm.weight"));
    let (nh, nkv, d) = (
        hp.n_head as usize,
        hp.n_head_kv as usize,
        hp.head_dim as usize,
    );
    let qg = mv(&blk("attn_q.weight"), &x);
    let v = mv(&blk("attn_v.weight"), &x);
    // One key at position 0: each head's output is its KV head's value (RoPE at 0 is the
    // identity and the query/key norms do not matter), times the sigmoid of its gate.
    let mut o = vec![0f32; nh * d];
    for hh in 0..nh {
        let kv = hh / (nh / nkv);
        for j in 0..d {
            o[hh * d + j] = v[kv * d + j] * sig(qg[hh * 2 * d + d + j]);
        }
    }
    let cur = add(&cur, &mv(&blk("attn_output.weight"), &o));
    let x = rms(&cur, &blk("post_attention_norm.weight"));
    let g = mv(&blk("ffn_gate.weight"), &x);
    let u = mv(&blk("ffn_up.weight"), &x);
    let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g * sig(*g) * u).collect();
    let cur = add(&cur, &mv(&blk("ffn_down.weight"), &act));
    let want_h = rms(&cur, &blk("nextn.shared_head_norm.weight"));
    let logits = mv(&get("output.weight"), &want_h);
    let want = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;

    let blocks = [0u32, 1, 2, 3];
    let out = e
        .mtp_step(&[group(&[tok], 0, &blocks, Logits::Last)], &h)
        .unwrap();
    assert_eq!(out.argmax, vec![want]);
    let scale = want_h.iter().fold(0f32, |a, v| a.max(v.abs()));
    let diff = out
        .hidden
        .iter()
        .zip(&want_h)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(
        diff / scale < 1e-4,
        "hidden differs by {diff} (scale {scale})"
    );
}

/// The head against a plain f32 reimplementation of Qwen3-Next's MTP layer (vLLM's
/// `qwen3_next_mtp.py`, Hugging Face's Qwen3-Next attention) over several positions, one pass a
/// position as drafting chains them: the query and key RMS norms, rotary embedding on the text
/// positions (rotate-half pairs over `n_rot` dimensions), causal softmax over the head's own
/// cache with grouped KV heads, and the sigmoid output gate. Same drafts, output hidden within
/// 1e-3 relative (measured: up to 1.4e-4, the F16 cache's rounding; a rotation one position
/// off gives 5e-2).
#[test]
fn qwen35_mtp_head_matches_a_reference_over_positions() {
    use legion::model::core::Logits;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let mut e = executor(dir.path());
    let m = e.model();
    let hp = &m.hp;
    let (n, eps) = (hp.n_embd as usize, hp.rms_eps);
    let get = |name: &str| m.tensor(name).unwrap_or_else(|| panic!("{name}")).to_f32();
    let blk = |name: &str| get(&format!("blk.{}.{name}", hp.n_layer));
    let rms = |x: &[f32], g: &[f32]| -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter().zip(g).map(|(v, g)| v * s * g).collect()
    };
    let mv = |w: &[f32], x: &[f32]| -> Vec<f32> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let (nh, nkv, d, n_rot) = (
        hp.n_head as usize,
        hp.n_head_kv as usize,
        hp.head_dim as usize,
        hp.n_rot as usize,
    );
    let base = hp.rope_freq_base;
    // Per head of `x` (`[heads][d]`): RMS norm with `g`, then the rotation at position `p` of
    // the first `n_rot` dimensions in rotate-half pairs (i, i + n_rot / 2).
    let norm_rope = |x: &[f32], g: &[f32], p: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for head in x.chunks(d) {
            let mut y = rms(head, g);
            for i in 0..n_rot / 2 {
                let theta = p as f32 * base.powf(-2.0 * i as f32 / n_rot as f32);
                let (s, c) = theta.sin_cos();
                let (a, b) = (y[i], y[i + n_rot / 2]);
                y[i] = a * c - b * s;
                y[i + n_rot / 2] = a * s + b * c;
            }
            out.extend(y);
        }
        out
    };

    let emb = get("token_embd.weight");
    let (enorm, hnorm) = (blk("nextn.enorm.weight"), blk("nextn.hnorm.weight"));
    let (eh, attn_norm) = (blk("nextn.eh_proj.weight"), blk("attn_norm.weight"));
    let (wq, wk, wv) = (
        blk("attn_q.weight"),
        blk("attn_k.weight"),
        blk("attn_v.weight"),
    );
    let (qn, kn) = (blk("attn_q_norm.weight"), blk("attn_k_norm.weight"));
    let wo = blk("attn_output.weight");
    let post = blk("post_attention_norm.weight");
    let (wg, wu, wd) = (
        blk("ffn_gate.weight"),
        blk("ffn_up.weight"),
        blk("ffn_down.weight"),
    );
    let (out_norm, output) = (blk("nextn.shared_head_norm.weight"), get("output.weight"));

    let tokens: Vec<u32> = "a multi-token head".bytes().map(u32::from).collect();
    let rows: Vec<Vec<f32>> = (0..tokens.len())
        .map(|t| {
            (0..n)
                .map(|i| ((i * 37 + t * 11) % 23) as f32 / 7.0 - 1.5)
                .collect()
        })
        .collect();
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    let blocks = [0u32, 1, 2, 3];
    for (p, (&tok, h)) in tokens.iter().zip(&rows).enumerate() {
        let en = rms(&emb[tok as usize * n..][..n], &enorm);
        let hn = rms(h, &hnorm);
        let cat: Vec<f32> = en.iter().chain(&hn).copied().collect();
        let cur = mv(&eh, &cat);
        let x = rms(&cur, &attn_norm);
        let qg = mv(&wq, &x);
        let q: Vec<f32> = qg.chunks(2 * d).flat_map(|c| c[..d].to_vec()).collect();
        let gate: Vec<f32> = qg.chunks(2 * d).flat_map(|c| c[d..].to_vec()).collect();
        let q = norm_rope(&q, &qn, p);
        keys.push(norm_rope(&mv(&wk, &x), &kn, p));
        values.push(mv(&wv, &x));
        let mut o = vec![0f32; nh * d];
        for hh in 0..nh {
            let kv = hh / (nh / nkv);
            let qh = &q[hh * d..][..d];
            let scores: Vec<f32> = keys
                .iter()
                .map(|k| {
                    let kh = &k[kv * d..][..d];
                    qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
                })
                .collect();
            let top = scores.iter().fold(f32::MIN, |a, &s| a.max(s));
            let ws: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let sum: f32 = ws.iter().sum();
            for (wt, v) in ws.iter().zip(&values) {
                for j in 0..d {
                    o[hh * d + j] += wt / sum * v[kv * d + j];
                }
            }
            for j in 0..d {
                o[hh * d + j] *= sig(gate[hh * d + j]);
            }
        }
        let cur = add(&cur, &mv(&wo, &o));
        let x = rms(&cur, &post);
        let g = mv(&wg, &x);
        let u = mv(&wu, &x);
        let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g * sig(*g) * u).collect();
        let cur = add(&cur, &mv(&wd, &act));
        let want_h = rms(&cur, &out_norm);
        let logits = mv(&output, &want_h);
        let want = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;

        let out = e
            .mtp_step(&[group(&[tok], p, &blocks, Logits::Last)], h)
            .unwrap();
        assert_eq!(out.argmax, vec![want], "position {p}");
        let scale = want_h.iter().fold(0f32, |a, v| a.max(v.abs()));
        let diff = out
            .hidden
            .iter()
            .zip(&want_h)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(
            diff / scale < 1e-3,
            "position {p}: hidden differs by {diff} (scale {scale})"
        );
    }
}

/// The head's cache across passes: a prompt taken in as two chunks and then the bonus token
/// gives the same draft, hidden and probability, bit for bit, as one pass over all of it.
#[test]
fn qwen35_mtp_passes_chain_through_the_head_cache() {
    use legion::model::core::Logits;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let prompt: Vec<u32> = "multi-token prediction".bytes().map(u32::from).collect();
    let blocks = [0u32, 1, 2, 3];
    let mut e = executor(dir.path());
    let out = e.step(&[group(&prompt, 0, &blocks, Logits::Last)]).unwrap();
    let bonus = out.argmax[0];
    let h = &out.hidden[0];
    let w = e.mtp_width();
    assert_eq!(h.len(), prompt.len() * w);
    let mut fed = prompt.clone();
    fed.push(bonus);
    let mut rows = vec![0f32; w];
    rows.extend_from_slice(h);
    let whole = e
        .mtp_step(&[group(&fed, 0, &blocks, Logits::Last)], &rows)
        .unwrap();
    for cut in [1, 9, fed.len() - 1] {
        let mut e = executor(dir.path());
        e.step(&[group(&prompt, 0, &blocks, Logits::Last)]).unwrap();
        e.mtp_step(
            &[group(&fed[..cut], 0, &blocks, Logits::None)],
            &rows[..cut * w],
        )
        .unwrap();
        let split = e
            .mtp_step(
                &[group(&fed[cut..], cut, &blocks, Logits::Last)],
                &rows[cut * w..],
            )
            .unwrap();
        assert_eq!(whole.argmax, split.argmax, "cut {cut}");
        assert_eq!(whole.pmax, split.pmax, "cut {cut}");
        assert!(
            whole
                .hidden
                .iter()
                .zip(&split.hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "cut {cut}"
        );
    }
}

/// `--spec-draft-vocab` on the qwen35 head: drafts over the first tokens only, greedy output
/// still bit-identical to plain decoding.
#[tokio::test(flavor = "multi_thread")]
async fn qwen35_mtp_draft_vocab_keeps_greedy_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    for vocab in [1, 40] {
        let e = LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 48,
            spec: mtp(None, 3, 0.0),
            draft_vocab: vocab,
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny-mtp.gguf"), DeviceChoice::Cpu)
        })
        .unwrap();
        let replies = run_all(&e, greedy()).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "vocab {vocab}: prompt {i} differs"
            );
        }
        assert!(e.stats().spec.drafted > 0, "vocab {vocab}");
    }
}

/// `--spec-draft-lm-head q4` on the qwen35 head (all rows, or the first 40): drafts with a 4-bit
/// copy of the output head, greedy output still bit-identical to plain decoding.
#[tokio::test(flavor = "multi_thread")]
async fn qwen35_mtp_q4_lm_head_keeps_greedy_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    for vocab in [0, 40] {
        let e = LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 48,
            spec: SpecConfig {
                draft_lm_head: legion::spec::DraftLmHead::Q4,
                ..mtp(None, 3, 0.0)
            },
            draft_vocab: vocab,
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny-mtp.gguf"), DeviceChoice::Cpu)
        })
        .unwrap();
        let replies = run_all(&e, greedy()).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "vocab {vocab}: prompt {i} differs"
            );
        }
        let s = e.stats().spec;
        assert!(s.drafted > 0, "vocab {vocab}: {s:?}");
    }
}

/// The head's matrices requantized to Q8_0 (`--spec-draft-head-type q8_0`): read instead of the
/// F32 ones, and the head's output stays close to the full-precision one.
#[test]
fn qwen35_mtp_head_requantized_stays_close() {
    use legion::model::core::Logits;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let blocks = [0u32, 1, 2, 3];
    let tok = [42u32];
    let w = executor(dir.path()).mtp_width();
    let h: Vec<f32> = (0..w)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
        .collect();
    let run = |requant: bool| {
        let mut e = executor_with(dir.path(), requant);
        e.mtp_step(&[group(&tok, 0, &blocks, Logits::Last)], &h)
            .unwrap()
            .hidden
    };
    let (full, q8) = (run(false), run(true));
    let scale = full.iter().fold(0f32, |a, v| a.max(v.abs()));
    let diff = full
        .iter()
        .zip(&q8)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
    assert!(diff > 0.0, "the copies were not read");
    assert!(
        diff / scale < 0.05,
        "Q8_0 head differs by {diff} (scale {scale})"
    );
}

/// #527: a long prompt goes through the head in pieces without logits (only their K/V written),
/// the same number of tokens each over a longer context. Those passes do not read their
/// attention inputs, and ggml's graph allocator reused one piece's plan for the next (the same
/// nodes, the same sizes) without seeing the attention mask widen with the context: the mask
/// ran past the compute buffer and `ggml_backend_tensor_alloc` aborted. Each piece now has room
/// for it, and the pieces leave the head's cache as one pass over the prompt does.
#[test]
fn qwen35_mtp_take_in_pieces_over_a_growing_context() {
    use legion::ggml::Backend;
    use legion::gguf::GgufFile;
    use legion::model::core::{Executor, Logits, Model};
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let (piece, n) = (64, 2048);
    let blocks: Vec<u32> = (0..(n / 64 + 2) as u32).collect();
    let fed: Vec<u32> = (0..=n).map(|i| (i * 7 % 200) as u32 + 10).collect();
    let new = || {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(4);
        b.set_use_ref(true);
        let m = Model::load(
            &GgufFile::open(dir.path().join("tiny-mtp.gguf")).unwrap(),
            |_| b.buffer_type(),
        )
        .unwrap();
        let mut e = Executor::new(m, b, blocks.len(), 1, 0).unwrap();
        e.enable_mtp().unwrap();
        e
    };
    let mut e = new();
    assert!(!e.mtp_paged(), "the CPU head gathers its keys");
    let w = e.mtp_width();
    let rows: Vec<f32> = (0..fed.len() * w)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 64.0)
        .collect();
    for at in (0..n).step_by(piece) {
        e.mtp_step(
            &[group(&fed[at..at + piece], at, &blocks, Logits::None)],
            &rows[at * w..(at + piece) * w],
        )
        .unwrap();
    }
    let split = e
        .mtp_step(
            &[group(&fed[n..], n, &blocks, Logits::Last)],
            &rows[n * w..],
        )
        .unwrap();
    let mut e = new();
    let whole = e
        .mtp_step(&[group(&fed, 0, &blocks, Logits::Last)], &rows)
        .unwrap();
    assert_eq!(whole.argmax, split.argmax);
    let close = whole
        .hidden
        .iter()
        .zip(&split.hidden)
        .all(|(a, b)| (a - b).abs() <= 1e-4 * (1.0 + a.abs()));
    assert!(close, "the pieces' cache differs from one pass's");
}

/// #527: with `--spec mtp`, a prompt that leaves room for one token in the context runs to the
/// end, and one that fills the context is refused before it is admitted (HTTP 400).
#[tokio::test(flavor = "multi_thread")]
async fn qwen35_mtp_prompt_size_check() {
    use legion::engine::EngineError;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let e = engine_of(dir.path(), "tiny-mtp.gguf", mtp(None, 3, 0.0));
    let req = |text: String, max_tokens| GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Text(text),
        sampling: greedy(),
        max_tokens,
        stop: vec![],
        ignore_eos: true,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    };
    // The longest prefix of a long text that still leaves a token free (ctx_size 512).
    let long: String = (0..2000)
        .map(|i| (b'a' + (i * 7 % 26) as u8) as char)
        .collect();
    let count = |n: usize| e.count_tokens(&req(long[..n].into(), 1)).unwrap() as usize;
    let (mut lo, mut hi) = (1, long.len());
    assert!(count(hi) >= 512, "the text must fill the context");
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if count(mid) < 512 { lo = mid } else { hi = mid }
    }
    assert_eq!(count(lo), 511);
    let mut s = e.generate(req(long[..lo].into(), 8)).unwrap();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Finished { reason, usage, .. } => {
                assert_eq!(usage.prompt_tokens, 511);
                // The token at the context's last position, then the one its logits give.
                assert_eq!((reason, usage.completion_tokens), (FinishReason::Length, 2));
                break;
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
    match e.generate(req(long[..hi].into(), 8)) {
        Err(EngineError::ContextOverflow { prompt, context }) => {
            assert_eq!((prompt as usize, context), (count(hi), 512));
        }
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("a prompt that fills the context was admitted"),
    }
}
