//! The EAGLE-3 drafter on the CPU with tiny random checkpoints: it loads, drafts the right
//! shapes, matches a plain-Rust reference of the head, rolls its cache back, reports the
//! distribution it drew from, and keeps the engine's greedy output bit-identical to plain
//! decoding (the target is the tiny random qwen35). No real model is involved.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::sampling::Rng;
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::safetensors::Checkpoint;
use legion::spec::eagle3::tiny::{TinyEagle3, tiny_offsets, write};
use legion::spec::eagle3::{DraftMode, Eagle3Drafter, Eagle3Options};
use legion::spec::{Draft, DraftRequest, ModelDrafter, SpecConfig, SpecKind, TargetInfo};
use tokio_util::sync::CancellationToken;

const N_EMBD: usize = 64;
const N_VOCAB: usize = 259;

struct Fixture {
    _dir: tempfile::TempDir,
    gguf: PathBuf,
    head: PathBuf,
}

fn fixture(c: &TinyEagle3) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("tiny.gguf");
    tiny::write(&gguf, &TinyConfig::default()).unwrap();
    let head = dir.path().join("head");
    std::fs::create_dir(&head).unwrap();
    write(&head, c).unwrap();
    Fixture {
        _dir: dir,
        gguf,
        head,
    }
}

fn info(f: &Fixture) -> TargetInfo {
    TargetInfo {
        n_layer: 4,
        n_embd: N_EMBD,
        n_vocab: N_VOCAB,
        model: f.gguf.clone(),
        draft: Some(f.head.clone()),
        gpu: false,
        threads: 4,
        ctx_size: 512,
        max_seqs: 4,
        use_ref: false,
    }
}

fn drafter(f: &Fixture, mode: DraftMode) -> Eagle3Drafter {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    Eagle3Drafter::load_on(&info(f), Eagle3Options { mode }, backend).unwrap()
}

/// A sequence's tokens and, per fused layer, hidden rows for every position.
struct Seq {
    tokens: Vec<u32>,
    hidden: [Vec<f32>; 3],
}

fn seq(len: usize, seed: u64) -> Seq {
    let mut rng = Rng::new(seed);
    let mut row = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * 2.0)
            .collect()
    };
    let tokens = (0..len + 1)
        .map(|_| (row(1)[0].abs() * 100.0) as u32 % N_VOCAB as u32)
        .collect();
    Seq {
        tokens,
        hidden: [row(len * N_EMBD), row(len * N_EMBD), row(len * N_EMBD)],
    }
}

impl Seq {
    /// One call of the drafter over positions `start..start + ran`, wanting `max` drafts.
    fn step(&self, d: &mut Eagle3Drafter, id: u64, start: usize, ran: usize, max: usize) -> Draft {
        let mut out = Draft::default();
        let rows = |l: usize| &self.hidden[l][start * N_EMBD..(start + ran) * N_EMBD];
        d.step(&mut [DraftRequest {
            seq: id,
            tokens: &self.tokens,
            start,
            ran,
            hidden: vec![rows(0), rows(1), rows(2)],
            max,
            out: &mut out,
        }])
        .unwrap();
        out
    }
}

#[test]
fn tiny_head_loads_and_drafts_the_right_shapes() {
    let f = fixture(&TinyEagle3::default());
    let s = seq(16, 1);
    let mut d = drafter(&f, DraftMode::Greedy);
    assert_eq!(d.hidden_layers(), vec![0, 1, 3]);
    assert_eq!(d.vocab_map().len(), 128);

    // A prefill chunk only builds the cache.
    let chunk = s.step(&mut d, 5, 0, 12, 0);
    assert_eq!(chunk, Draft::default());
    assert_eq!(d.cached(5), Some(0..12));
    // A decode step drafts k tokens, all inside the draft vocabulary (even byte tokens).
    let draft = s.step(&mut d, 5, 12, 4, 4);
    assert_eq!(d.cached(5), Some(0..16));
    assert_eq!(draft.tokens.len(), 4);
    assert!(draft.probs.is_empty(), "greedy drafts are point masses");
    for &t in &draft.tokens {
        assert!(t < 256 && t % 2 == 0, "{t} is outside the draft vocabulary");
    }
    // Another sequence, one token at a time, drafting 1..=3.
    for max in 1..=3 {
        let again = s.step(&mut d, 6, 15, 1, max);
        assert_eq!(again.tokens.len(), max);
    }
    d.release(5);
    d.release(6);
    assert_eq!(d.cached(5), None);
}

#[test]
fn sampled_drafts_carry_their_distribution() {
    let f = fixture(&TinyEagle3::default());
    let s = seq(10, 2);
    let mut d = drafter(&f, DraftMode::Sample { seed: 11 });
    let draft = s.step(&mut d, 1, 0, 10, 3);
    assert_eq!((draft.tokens.len(), draft.probs.len()), (3, 3));
    for (t, q) in draft.tokens.iter().zip(&draft.probs) {
        assert_eq!(q.len(), N_VOCAB);
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(q[*t as usize] > 0.0, "the draft was drawn from q");
        for (i, &p) in q.iter().enumerate() {
            assert!(
                i % 2 == 0 && i < 256 || p == 0.0,
                "mass outside the draft vocab"
            );
        }
    }
}

/// The first draft is a draw from the reported distribution.
#[test]
fn drafts_are_drawn_from_the_reported_distribution() {
    let f = fixture(&TinyEagle3::default());
    let s = seq(6, 3);
    let mut d = drafter(&f, DraftMode::Sample { seed: 99 });
    const N: usize = 1500;
    let mut counts = vec![0usize; N_VOCAB];
    let mut q = Vec::new();
    for i in 0..N {
        let draft = s.step(&mut d, i as u64, 0, 6, 1);
        counts[draft.tokens[0] as usize] += 1;
        if q.is_empty() {
            q = draft.probs[0].clone();
        } else {
            // Same context, same distribution.
            let max_diff = q
                .iter()
                .zip(&draft.probs[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max_diff < 1e-6, "{max_diff}");
        }
        d.release(i as u64);
    }
    let mut top: Vec<usize> = (0..N_VOCAB).collect();
    top.sort_by(|&a, &b| q[b].total_cmp(&q[a]));
    for &t in &top[..4] {
        let p = f64::from(q[t]);
        let freq = counts[t] as f64 / N as f64;
        let sigma = (p * (1.0 - p) / N as f64).sqrt();
        assert!(
            (freq - p).abs() < 5.0 * sigma + 1e-3,
            "token {t}: drawn {freq:.4}, q {p:.4} (sigma {sigma:.4})"
        );
    }
    // Nothing was drawn outside q's support.
    assert!((0..N_VOCAB).all(|t| counts[t] == 0 || q[t] > 0.0));
}

// ---- a plain-Rust reference of the head ----

struct Weights {
    ck: Checkpoint,
}

impl Weights {
    fn get(&self, name: &str) -> Vec<f32> {
        self.ck
            .data(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let m = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (m + 1e-5).sqrt();
    x.iter().zip(w).map(|(v, w)| v * s * w).collect()
}

/// NeoX RoPE on one head: pairs `(i, i + d / 2)`.
fn rope(x: &mut [f32], pos: usize) {
    let d = x.len();
    for i in 0..d / 2 {
        let a = pos as f32 * 10_000f32.powf(-(2.0 * i as f32) / d as f32);
        let (s, c) = a.sin_cos();
        let (x0, x1) = (x[i], x[i + d / 2]);
        x[i] = x0 * c - x1 * s;
        x[i + d / 2] = x0 * s + x1 * c;
    }
}

/// The reference head: one layer, an unbounded f32 cache.
struct Reference {
    w: Weights,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Reference {
    fn new(dir: &Path) -> Self {
        Self {
            w: Weights {
                ck: Checkpoint::open(dir).unwrap(),
            },
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Cell `pos`; `g` is the fused hidden state. Returns (h2, logits over the draft vocab).
    fn step(&mut self, token: u32, g: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let w = &self.w;
        let p = "midlayer";
        let (nh, nkv, hd) = (4, 2, 16);
        let emb = w.get("embed_tokens.weight")[token as usize * N_EMBD..][..N_EMBD].to_vec();
        let en = rms(&emb, &w.get(&format!("{p}.input_layernorm.weight")));
        let hn = rms(g, &w.get(&format!("{p}.hidden_norm.weight")));
        let x: Vec<f32> = en.iter().chain(&hn).copied().collect();
        let mut q = matvec(&w.get(&format!("{p}.self_attn.q_proj.weight")), &x);
        let mut k = matvec(&w.get(&format!("{p}.self_attn.k_proj.weight")), &x);
        let v = matvec(&w.get(&format!("{p}.self_attn.v_proj.weight")), &x);
        q.chunks_exact_mut(hd).for_each(|h| rope(h, pos));
        k.chunks_exact_mut(hd).for_each(|h| rope(h, pos));
        self.k.truncate(pos);
        self.v.truncate(pos);
        self.k.push(k);
        self.v.push(v);
        let mut o = vec![0f32; nh * hd];
        for head in 0..nh {
            let kv = head / (nh / nkv);
            let qh = &q[head * hd..(head + 1) * hd];
            let mut sc: Vec<f32> = self
                .k
                .iter()
                .map(|k| {
                    qh.iter()
                        .zip(&k[kv * hd..(kv + 1) * hd])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / (hd as f32).sqrt()
                })
                .collect();
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sc.iter_mut().for_each(|s| *s = (*s - m).exp());
            let z: f32 = sc.iter().sum();
            for (s, v) in sc.iter().zip(&self.v) {
                for j in 0..hd {
                    o[head * hd + j] += s / z * v[kv * hd + j];
                }
            }
        }
        let attn = matvec(&w.get(&format!("{p}.self_attn.o_proj.weight")), &o);
        let h1: Vec<f32> = g.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let pn = rms(&h1, &w.get(&format!("{p}.post_attention_layernorm.weight")));
        let gate = matvec(&w.get(&format!("{p}.mlp.gate_proj.weight")), &pn);
        let up = matvec(&w.get(&format!("{p}.mlp.up_proj.weight")), &pn);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = matvec(&w.get(&format!("{p}.mlp.down_proj.weight")), &act);
        let h2: Vec<f32> = h1.iter().zip(&down).map(|(a, b)| a + b).collect();
        let logits = matvec(&w.get("lm_head.weight"), &rms(&h2, &w.get("norm.weight")));
        (h2, logits)
    }
}

fn softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

/// Draft ids to target-vocabulary probabilities, for comparing with `Draft::probs`.
fn to_target(probs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; N_VOCAB];
    for (i, o) in tiny_offsets(probs.len()).iter().enumerate() {
        out[(i as i64 + o) as usize] = probs[i];
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

/// The ggml head, fed a prompt in two chunks and then drafting a chain of three, computes what
/// the plain-Rust head does (F16 cache rounding aside): fc, both norms, concatenation, GQA,
/// NeoX RoPE, the residuals, the MLP, the final norm and the draft-vocabulary head, and the
/// chain's feedback of the pre-norm output.
#[test]
fn head_matches_a_plain_reference() {
    let f = fixture(&TinyEagle3::default());
    let s = seq(14, 4);
    let mut d = drafter(&f, DraftMode::Sample { seed: 5 });
    s.step(&mut d, 1, 0, 6, 0);
    let draft = s.step(&mut d, 1, 6, 8, 3);

    let mut r = Reference::new(&f.head);
    let fc = r.w.get("fc.weight");
    let fused = |p: usize| -> Vec<f32> {
        let x: Vec<f32> = (0..3)
            .flat_map(|l| s.hidden[l][p * N_EMBD..(p + 1) * N_EMBD].to_vec())
            .collect();
        matvec(&fc, &x)
    };
    let mut last = None;
    for p in 0..14 {
        last = Some(r.step(s.tokens[p + 1], &fused(p), p));
    }
    let (mut h, mut logits) = last.unwrap();
    for i in 0..3 {
        let probs = softmax(&logits);
        let diff = max_diff(&to_target(&probs), &draft.probs[i]);
        assert!(diff < 1e-3, "draft {i}: probabilities differ by {diff}");
        // Follow the drafter's own choice down the chain.
        let t = draft.tokens[i];
        if i + 1 < 3 {
            (h, logits) = r.step(t, &h, 14 + i);
        }
    }
}

/// Rolling back: a call that starts below the committed end replaces the cells from there on,
/// and drafts exactly as a drafter that never saw the rejected positions.
#[test]
fn rollback_matches_a_fresh_drafter() {
    let f = fixture(&TinyEagle3::default());
    let a = seq(16, 6);
    // `b` agrees with `a` up to position 12, then differs.
    let mut b = seq(16, 7);
    b.tokens[..13].copy_from_slice(&a.tokens[..13]);
    for l in 0..3 {
        b.hidden[l][..12 * N_EMBD].copy_from_slice(&a.hidden[l][..12 * N_EMBD]);
    }

    let mut rolled = drafter(&f, DraftMode::Sample { seed: 1 });
    a.step(&mut rolled, 1, 0, 10, 0);
    // Verify step: positions 10..16 were fed to the head, then only 10 and 11 were kept.
    let first = a.step(&mut rolled, 1, 10, 6, 3);
    assert_eq!(rolled.cached(1), Some(0..16));
    let after = b.step(&mut rolled, 1, 12, 4, 3);
    assert_eq!(rolled.cached(1), Some(0..16));

    let mut fresh = drafter(&f, DraftMode::Sample { seed: 1 });
    b.step(&mut fresh, 9, 0, 10, 0);
    b.step(&mut fresh, 9, 10, 2, 0);
    let expect = b.step(&mut fresh, 9, 12, 4, 3);
    assert_eq!(after.probs.len(), 3);
    // The chain's later steps agree while the (sampled) drafts do.
    let same = after
        .tokens
        .iter()
        .zip(&expect.tokens)
        .take_while(|(x, y)| x == y)
        .count();
    for i in 0..=same.min(2) {
        let diff = max_diff(&after.probs[i], &expect.probs[i]);
        assert!(diff < 1e-5, "draft {i}: {diff}");
    }
    // The rejected tail did change the answer (the test would be vacuous otherwise).
    assert!(max_diff(&first.probs[0], &after.probs[0]) > 1e-4);

    // Rolling back into the middle of the prompt, and to zero.
    let mut rolled = drafter(&f, DraftMode::Greedy);
    a.step(&mut rolled, 1, 0, 14, 0);
    a.step(&mut rolled, 1, 7, 7, 0);
    assert_eq!(rolled.cached(1), Some(0..14));
    a.step(&mut rolled, 1, 3, 2, 0);
    assert_eq!(rolled.cached(1), Some(0..5));
    a.step(&mut rolled, 1, 0, 1, 0);
    assert_eq!(rolled.cached(1), Some(0..1));
}

/// A `start` past the committed end (positions the prefix cache skipped) restarts the cache.
#[test]
fn a_gap_restarts_the_cache() {
    let f = fixture(&TinyEagle3::default());
    let s = seq(20, 8);
    let mut d = drafter(&f, DraftMode::Greedy);
    // A sequence resumed from the prefix cache starts at 8.
    s.step(&mut d, 3, 8, 4, 0);
    assert_eq!(d.cached(3), Some(8..12));
    s.step(&mut d, 3, 12, 4, 2);
    assert_eq!(d.cached(3), Some(8..16));
    s.step(&mut d, 3, 18, 2, 2);
    assert_eq!(d.cached(3), Some(18..20));
    // The target's last position has no next token yet: nothing is taken in.
    let short = Seq {
        tokens: s.tokens[..10].to_vec(),
        hidden: s.hidden.clone(),
    };
    let draft = short.step(&mut d, 4, 8, 2, 3);
    assert_eq!(draft, Draft::default());
    assert_eq!(d.cached(4), Some(8..9));
}

#[test]
fn every_variant_loads_and_drafts() {
    let base = TinyEagle3::default();
    let variants = [
        TinyEagle3 {
            bf16: true,
            ..base.clone()
        },
        TinyEagle3 {
            midlayer: false,
            ..base.clone()
        },
        TinyEagle3 {
            attention_bias: true,
            qk_norm: true,
            ..base.clone()
        },
        TinyEagle3 {
            norm_before_residual: true,
            ..base.clone()
        },
        TinyEagle3 {
            norm_before_fc: true,
            ..base.clone()
        },
        TinyEagle3 {
            fc_norm: true,
            ..base.clone()
        },
        TinyEagle3 {
            norm_output: true,
            ..base.clone()
        },
        TinyEagle3 {
            llama3_rope: true,
            ..base.clone()
        },
        // The draft vocabulary is the target's: no d2t / t2d.
        TinyEagle3 {
            draft_vocab: N_VOCAB,
            map: false,
            ..base.clone()
        },
        // Layer ids from the default recipe (n_layer 4: layers 1, 1 and 0).
        TinyEagle3 {
            layer_ids: None,
            ..base.clone()
        },
    ];
    let s = seq(9, 9);
    for (i, v) in variants.iter().enumerate() {
        let f = fixture(v);
        let mut d = drafter(&f, DraftMode::Sample { seed: 3 });
        let draft = s.step(&mut d, 1, 0, 9, 3);
        assert_eq!(draft.tokens.len(), 3, "variant {i}");
        for q in &draft.probs {
            assert!(q.iter().all(|p| p.is_finite()), "variant {i}");
            assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-4, "variant {i}");
        }
    }
    // Each flag changes the computation (probabilities differ from the base head's).
    let probs = |v: &TinyEagle3| {
        let f = fixture(v);
        let mut d = drafter(&f, DraftMode::Sample { seed: 3 });
        s.step(&mut d, 1, 0, 9, 1).probs.remove(0)
    };
    let reference = probs(&base);
    for i in [2, 3, 4, 5, 7] {
        // (norm_output only changes the chain's feedback, not the first draft.)
        assert!(
            max_diff(&reference, &probs(&variants[i])) > 1e-5,
            "variant {i} computes the same as the base head"
        );
    }
    // BF16 weights stay close to the F32 head's: same seed, rounded matrices.
    assert!(max_diff(&reference, &probs(&variants[0])) < 5e-3);
}

#[test]
fn the_head_reads_all_three_hidden_layers() {
    let f = fixture(&TinyEagle3::default());
    let a = seq(8, 10);
    let probs = |s: &Seq| {
        let mut d = drafter(&f, DraftMode::Sample { seed: 2 });
        s.step(&mut d, 1, 0, 8, 1).probs.remove(0)
    };
    let base = probs(&a);
    for l in 0..3 {
        let mut b = Seq {
            tokens: a.tokens.clone(),
            hidden: a.hidden.clone(),
        };
        b.hidden[l][3 * N_EMBD] += 1.0;
        assert!(
            max_diff(&base, &probs(&b)) > 1e-6,
            "layer slot {l} is ignored"
        );
    }
}

#[test]
fn shared_embeddings_come_from_the_target_gguf() {
    let f = fixture(&TinyEagle3 {
        embed: false,
        ..Default::default()
    });
    let d = drafter(&f, DraftMode::Greedy);
    let gguf = legion::gguf::GgufFile::open(&f.gguf).unwrap();
    let t = gguf.tensor("token_embd.weight").unwrap();
    let bytes = std::fs::read(&f.gguf).unwrap();
    for token in [0u32, 97, 258] {
        let at = t.offset as usize + token as usize * N_EMBD * 4;
        let expect: Vec<f32> = bytes[at..at + N_EMBD * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect();
        assert_eq!(d.embedding(token).unwrap(), expect);
    }
    assert!(d.embedding(259).is_err());
    // And it drafts.
    let s = seq(6, 11);
    let mut d = d;
    assert_eq!(s.step(&mut d, 1, 0, 6, 2).tokens.len(), 2);
}

#[test]
fn bad_checkpoints_say_what_is_wrong() {
    let load = |i: TargetInfo| {
        Eagle3Drafter::load_on(&i, Eagle3Options::default(), Backend::cpu().unwrap())
            .err()
            .expect("should not load")
    };
    let f = fixture(&TinyEagle3::default());
    let mut i = info(&f);
    i.n_embd = 32;
    assert!(load(i).contains("32"));
    let mut i = info(&f);
    i.n_layer = 3;
    assert!(load(i).contains("layer 3 does not exist"));
    let mut i = info(&f);
    i.draft = None;
    assert!(load(i).contains("--spec-draft"));
    let mut i = info(&f);
    i.draft = Some(f.head.join("nowhere"));
    assert!(!load(i).is_empty());
    let mut i = info(&f);
    i.n_vocab = 100;
    assert!(load(i).contains("outside"));

    // A draft vocabulary below the target's needs the map.
    let f = fixture(&TinyEagle3 {
        map: false,
        ..Default::default()
    });
    assert!(load(info(&f)).contains("d2t"));
    // No config.json.
    let f = fixture(&TinyEagle3::default());
    std::fs::remove_file(f.head.join("config.json")).unwrap();
    assert!(load(info(&f)).contains("config.json"));
    // A config for another shape.
    let f = fixture(&TinyEagle3::default());
    let cfg = std::fs::read_to_string(f.head.join("config.json")).unwrap();
    std::fs::write(
        f.head.join("config.json"),
        cfg.replace("\"intermediate_size\": 96", "\"intermediate_size\": 97"),
    )
    .unwrap();
    assert!(load(info(&f)).contains("shape"));
}

// ---- through the engine ----

const K: usize = 4;
const MAX_TOKENS: u32 = 48;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4),
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "speculative decoding".into(),
        "0123456789".repeat(3),
        "abc".into(),
    ]
}

fn engine(gguf: &Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        // Prompts prefill in several steps next to decoding sequences.
        max_batch_tokens: 40,
        spec,
        use_ref: true,
        ..LmConfig::new(gguf, DeviceChoice::Cpu)
    })
    .unwrap()
}

/// Every prompt at once: (text, tokens generated, spec counters).
async fn run_all(
    engine: &LmEngine,
    sampling: SamplingParams,
) -> Vec<(String, u32, Option<legion::spec::SpecCounters>)> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| {
            engine
                .generate(GenerateRequest {
                    model: "tiny".into(),
                    prompt: Prompt::Text(p.clone()),
                    sampling: sampling.clone(),
                    max_tokens: MAX_TOKENS,
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
                TokenEvent::Finished { usage, .. } => {
                    replies.push((text, usage.completion_tokens, usage.spec));
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
    }
    replies
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

/// `--spec eagle3 --spec-draft <dir>`: speculative greedy output equals plain decoding, bit for
/// bit, whatever the head proposes (it is random here, so it is mostly wrong).
#[tokio::test(flavor = "multi_thread")]
async fn greedy_output_with_eagle3_equals_plain_decoding() {
    let f = fixture(&TinyEagle3::default());
    let plain = run_all(&engine(&f.gguf, SpecConfig::default()), greedy()).await;
    let e = engine(
        &f.gguf,
        SpecConfig {
            kind: SpecKind::Eagle3,
            k: K,
            draft: Some(f.head.clone()),
            ..Default::default()
        },
    );
    let replies = run_all(&e, greedy()).await;
    for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
        assert_eq!((&a.0, a.1), (&b.0, b.1), "prompt {i}");
    }
    let c = e.stats().spec;
    assert!(c.drafted > 0, "the head never drafted: {c:?}");
    assert!(c.accepted <= c.drafted);
    println!("eagle3 greedy: {c:?}");
}

/// A bad `--spec-draft` fails the engine's load with the reason.
#[test]
fn engine_load_reports_a_bad_head() {
    let f = fixture(&TinyEagle3::default());
    std::fs::remove_file(f.head.join("model.safetensors")).unwrap();
    let r = LmEngine::load(LmConfig {
        spec: SpecConfig {
            kind: SpecKind::Eagle3,
            k: K,
            draft: Some(f.head.clone()),
            ..Default::default()
        },
        ..LmConfig::new(&f.gguf, DeviceChoice::Cpu)
    });
    let Err(e) = r else { panic!("should not load") };
    assert!(e.to_string().contains("model.safetensors"), "{e}");
}

/// Sampled decoding with the head drawing from its distribution runs through exact rejection
/// sampling and respects the counters.
#[tokio::test(flavor = "multi_thread")]
async fn sampled_decoding_with_draft_distributions_runs() {
    let f = fixture(&TinyEagle3::default());
    let gguf_head = f.head.clone();
    let cfg = SpecConfig {
        k: K,
        model_drafter: Some(std::sync::Arc::new(move |t: &TargetInfo| {
            let mut t = t.clone();
            t.draft = Some(gguf_head.clone());
            Ok(Box::new(Eagle3Drafter::load(
                &t,
                Eagle3Options {
                    mode: DraftMode::Sample { seed: 4 },
                },
            )?) as Box<dyn ModelDrafter>)
        })),
        ..Default::default()
    };
    let e = engine(&f.gguf, cfg);
    let sampling = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        seed: Some(3),
        ..Default::default()
    };
    let replies = run_all(&e, sampling).await;
    assert!(replies.iter().all(|r| r.1 > 0));
    let c = e.stats().spec;
    assert!(c.drafted > 0 && c.accepted <= c.drafted, "{c:?}");
}

/// The target's greedy continuation of every prompt, computed without the engine.
fn truths(path: &Path) -> Vec<(Vec<u32>, Vec<u32>)> {
    let gguf = legion::gguf::GgufFile::open(path).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    backend.set_use_ref(true);
    let model = legion::model::core::Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    prompts()
        .iter()
        .map(|p| {
            let prompt: Vec<u32> = p.bytes().map(u32::from).collect();
            let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
            let mut alloc = None;
            let mut logits = model
                .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
                .unwrap();
            let mut out = Vec::new();
            for _ in 0..MAX_TOKENS as usize + K {
                let t = legion::engine::sampling::argmax(&logits) as u32;
                out.push(t);
                logits = model
                    .forward(&mut backend, &mut alloc, &mut state, &[t], false)
                    .unwrap();
            }
            (prompt, out)
        })
        .collect()
}

/// The EAGLE-3 head takes in every step (prefill chunks, verify steps with accepted drafts of
/// several positions) while its drafts are replaced by the true continuation, so that every
/// draft is accepted and the head sees long accepted runs.
struct Hybrid {
    head: Eagle3Drafter,
    truths: std::sync::Arc<Vec<(Vec<u32>, Vec<u32>)>>,
}

impl ModelDrafter for Hybrid {
    fn hidden_layers(&self) -> Vec<usize> {
        self.head.hidden_layers()
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        self.head.step(batch)?;
        for r in batch.iter_mut() {
            if r.max == 0 {
                continue;
            }
            assert_eq!(r.out.tokens.len(), r.max);
            let truth = self
                .truths
                .iter()
                .filter(|(p, _)| r.tokens.len() > p.len() && r.tokens.starts_with(p))
                .max_by_key(|(p, _)| p.len())
                .and_then(|(p, t)| {
                    let done = &r.tokens[p.len()..];
                    t.starts_with(done).then(|| &t[done.len()..])
                })
                .unwrap_or(&[]);
            for (d, t) in r.out.tokens.iter_mut().zip(truth) {
                *d = *t;
            }
            if truth.len() < r.max {
                r.out.tokens.truncate(truth.len());
            }
        }
        Ok(())
    }

    fn release(&mut self, seq: u64) {
        self.head.release(seq);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_head_follows_long_accepted_runs() {
    let f = fixture(&TinyEagle3::default());
    let truths = std::sync::Arc::new(truths(&f.gguf));
    let head_dir = f.head.clone();
    let cfg = SpecConfig {
        k: K,
        model_drafter: Some(std::sync::Arc::new(move |t: &TargetInfo| {
            let mut t = t.clone();
            t.draft = Some(head_dir.clone());
            Ok(Box::new(Hybrid {
                head: Eagle3Drafter::load(&t, Eagle3Options::default())?,
                truths: truths.clone(),
            }) as Box<dyn ModelDrafter>)
        })),
        ..Default::default()
    };
    let plain = run_all(&engine(&f.gguf, SpecConfig::default()), greedy()).await;
    let e = engine(&f.gguf, cfg);
    let replies = run_all(&e, greedy()).await;
    for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
        assert_eq!((&a.0, a.1), (&b.0, b.1), "prompt {i}");
    }
    let c = e.stats().spec;
    assert!(c.drafted > 0 && c.accepted == c.drafted, "{c:?}");
}
