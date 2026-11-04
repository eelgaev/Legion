//! The qwen35 MTP head on a real Qwen3.5 / 3.8 checkpoint (#397), against a plain f32
//! reimplementation of Qwen3-Next's MTP layer as Hugging Face's attention and vLLM's
//! `qwen3_next_mtp.py` (Apache-2.0) run it, which reads the checkpoint's own tensors by their
//! Hugging Face names: `mtp.pre_fc_norm_embedding` / `mtp.pre_fc_norm_hidden` / `mtp.norm` and
//! the layer's norms as zero-centred RMS norms (`1 + w`), `mtp.fc` over `[embedding ; hidden]`,
//! `q_proj`'s per-head `[query | gate]` rows, the query / key norms, the rotation of the first
//! `partial_rotary_factor * head_dim` dimensions in rotate-half pairs, causal softmax over the
//! head's own keys with grouped KV heads, the sigmoid gate, the SwiGLU MLP, and `lm_head`
//! (NVFP4 as ModelOpt packs it, or 16-bit). It shares nothing with the converter
//! (`src/hf/qwen35.rs`) that maps the head to `blk.{n_layer}.nextn.*`.
//!
//! The target takes a prompt (Legion's step, its normed last hidden captured), then the head
//! takes the prompt in as the engine does (position `p` pairs token `p` with the target's
//! hidden at `p - 1`, zeros at 0), runs one pass per position over the last few, and one draft
//! pass fed its own output. The reference runs the same inputs. Per output position it prints
//! the drafts, their probabilities, the reference's top-2 margin, and the output hidden's
//! distance; drafts must agree wherever the reference's margin is clear of rounding.
//!
//! Ignored: needs the checkpoint. On the CPU, and in a `cuda` build on the GPU too:
//!
//! ```text
//! LEGION_TEST_QWEN35_MTP_HF=<snapshot dir> cargo test --release [--features cuda] \
//!   --test spec_mtp_qwen35_hf -- --ignored --nocapture
//! ```

use legion::ggml::{Backend, DeviceKind, device_by_kind};
use legion::gguf::GgufFile;
use legion::hf::nvfp4::e4m3_to_f32;
use legion::model::core::{Executor, Group, KV_BLOCK, Logits, Model, MtpOutput, SeqInput};
use legion::safetensors::{Checkpoint, Dtype};
use legion::tokenizer::Tokenizer;
use serde_json::Value;

const TEXT: &str = "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the lunar \
    module Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled the \
    goal President Kennedy had set eight years earlier";

/// Positions the head runs one pass at a time (the prompt's last ones and the bonus token).
const PASSES: usize = 5;

/// A tensor of the checkpoint as f32 (BF16, F16 or F32), and its shape.
fn load(ck: &Checkpoint, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (_, t) = ck.locate(name).unwrap_or_else(|| panic!("no `{name}`"));
    let (dtype, shape) = (t.dtype.clone(), t.shape.clone());
    let b = ck.data(name).unwrap();
    let v = match dtype {
        Dtype::BF16 => bf16s(b),
        Dtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        d => panic!("`{name}`: {d:?}"),
    };
    (v, shape)
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// A zero-centred RMS norm (`Qwen3NextRMSNorm`): `x / rms(x) * (1 + w)`.
fn norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * s * (1.0 + w)).collect()
}

/// `y = W x` for a row-major `W` of rows of `x.len()`, rows split over threads.
fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
    let rows = w.len() / x.len();
    let mut y = vec![0f32; rows];
    let per = rows.div_ceil(16).max(1);
    std::thread::scope(|s| {
        for (i, out) in y.chunks_mut(per).enumerate() {
            let w = &w[i * per * x.len()..];
            s.spawn(move || {
                for (r, o) in out.iter_mut().enumerate() {
                    let row = &w[r * x.len()..][..x.len()];
                    *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
                }
            });
        }
    });
    y
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The output head's logits for each of `xs`: NVFP4 as ModelOpt packs it (`weight` U8 rows of
/// `k / 2` bytes, element `2i` in the low nibble; `weight_scale` E4M3 per 16; the F32
/// `weight_scale_2`), or a 16/32-bit matrix.
fn lm_head(ck: &Checkpoint, name: &str, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (_, t) = ck.locate(name).unwrap();
    if t.dtype != Dtype::U8 {
        let (w, _) = load(ck, name);
        return xs.iter().map(|x| mv(&w, x)).collect();
    }
    let k = xs[0].len();
    let rows = t.shape[0];
    let base = name.strip_suffix(".weight").unwrap();
    let w = ck.data(name).unwrap();
    let sc = ck.data(&format!("{base}.weight_scale")).unwrap();
    let (s2, _) = load(ck, &format!("{base}.weight_scale_2"));
    let s2 = s2[0];
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut out = vec![vec![0f32; rows]; xs.len()];
    let per = rows.div_ceil(16);
    let mut parts: Vec<Vec<Vec<f32>>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..rows.div_ceil(per))
            .map(|i| {
                s.spawn(move || {
                    let (r0, r1) = (i * per, ((i + 1) * per).min(rows));
                    let mut part = vec![vec![0f32; r1 - r0]; xs.len()];
                    let mut row = vec![0f32; k];
                    for r in r0..r1 {
                        for (e, v) in row.iter_mut().enumerate() {
                            let byte = w[r * k / 2 + e / 2];
                            let code = (byte >> (4 * (e % 2))) & 0xf;
                            let mag = E2M1[(code & 7) as usize];
                            let val = if code & 8 != 0 { -mag } else { mag };
                            *v = val * e4m3_to_f32(sc[r * k / 16 + e / 16]) * s2;
                        }
                        for (p, x) in part.iter_mut().zip(xs) {
                            p[r - r0] = row.iter().zip(x).map(|(a, b)| a * b).sum();
                        }
                    }
                    part
                })
            })
            .collect();
        parts = handles.into_iter().map(|h| h.join().unwrap()).collect();
    });
    for (i, part) in parts.into_iter().enumerate() {
        for (o, p) in out.iter_mut().zip(part) {
            o[i * per..i * per + p.len()].copy_from_slice(&p);
        }
    }
    out
}

/// The reference head over every position of `tokens` (each with its hidden row in `rows`):
/// the output hidden of the positions in `outputs`.
fn reference(
    ck: &Checkpoint,
    tc: &Value,
    tokens: &[u32],
    rows: &[Vec<f32>],
    outputs: std::ops::Range<usize>,
) -> Vec<Vec<f32>> {
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64);
    let rope = tc.get("rope_parameters").unwrap_or(tc);
    let eps = num(tc, "rms_norm_eps").unwrap() as f32;
    let nh = num(tc, "num_attention_heads").unwrap() as usize;
    let nkv = num(tc, "num_key_value_heads").unwrap() as usize;
    let d = num(tc, "head_dim").unwrap() as usize;
    let theta = num(rope, "rope_theta")
        .or_else(|| num(tc, "rope_theta"))
        .unwrap() as f32;
    let partial = num(rope, "partial_rotary_factor")
        .or_else(|| num(tc, "partial_rotary_factor"))
        .unwrap_or(1.0);
    let n_rot = (d as f64 * partial) as usize;
    let l = |n: &str| load(ck, &format!("mtp.layers.0.{n}")).0;
    let (fc, en_w, hn_w) = (
        load(ck, "mtp.fc.weight").0,
        load(ck, "mtp.pre_fc_norm_embedding.weight").0,
        load(ck, "mtp.pre_fc_norm_hidden.weight").0,
    );
    let (in_w, post_w, out_w) = (
        l("input_layernorm.weight"),
        l("post_attention_layernorm.weight"),
        load(ck, "mtp.norm.weight").0,
    );
    let (wq, wk, wv, wo) = (
        l("self_attn.q_proj.weight"),
        l("self_attn.k_proj.weight"),
        l("self_attn.v_proj.weight"),
        l("self_attn.o_proj.weight"),
    );
    let (qn, kn) = (l("self_attn.q_norm.weight"), l("self_attn.k_norm.weight"));
    let (wg, wu, wd) = (
        l("mlp.gate_proj.weight"),
        l("mlp.up_proj.weight"),
        l("mlp.down_proj.weight"),
    );
    let emb_name = ck
        .tensors()
        .into_iter()
        .map(|t| t.name)
        .find(|n| n.ends_with("embed_tokens.weight") && !n.starts_with("mtp"))
        .expect("embed_tokens");
    let (emb_t, emb_b) = (
        ck.locate(&emb_name).unwrap().1.clone(),
        ck.data(&emb_name).unwrap(),
    );
    assert_eq!(emb_t.dtype, Dtype::BF16, "embed_tokens as BF16");
    let n = emb_t.shape[1];
    let embed = |t: u32| bf16s(&emb_b[t as usize * n * 2..][..n * 2]);
    // Per head: the norm, then the rotation at `p` of the first `n_rot` dimensions.
    let norm_rope = |x: &[f32], g: &[f32], p: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for head in x.chunks(d) {
            let mut y = norm(head, g, eps);
            for i in 0..n_rot / 2 {
                let a = p as f64 * (theta as f64).powf(-2.0 * i as f64 / n_rot as f64);
                let (s, c) = (a.sin() as f32, a.cos() as f32);
                let (u, v) = (y[i], y[i + n_rot / 2]);
                y[i] = u * c - v * s;
                y[i + n_rot / 2] = u * s + v * c;
            }
            out.extend(y);
        }
        out
    };
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    let mut hidden = Vec::new();
    for (p, (&tok, h)) in tokens.iter().zip(rows).enumerate() {
        let cat: Vec<f32> = norm(&embed(tok), &en_w, eps)
            .into_iter()
            .chain(norm(h, &hn_w, eps))
            .collect();
        let cur = mv(&fc, &cat);
        let x = norm(&cur, &in_w, eps);
        keys.push(norm_rope(&mv(&wk, &x), &kn, p));
        values.push(mv(&wv, &x));
        if !outputs.contains(&p) {
            continue;
        }
        let qg = mv(&wq, &x);
        let q: Vec<f32> = qg.chunks(2 * d).flat_map(|c| c[..d].to_vec()).collect();
        let gate: Vec<f32> = qg.chunks(2 * d).flat_map(|c| c[d..].to_vec()).collect();
        let q = norm_rope(&q, &qn, p);
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
        let x = norm(&cur, &post_w, eps);
        let g = mv(&wg, &x);
        let u = mv(&wu, &x);
        let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g * sig(*g) * u).collect();
        let cur = add(&cur, &mv(&wd, &act));
        hidden.push(norm(&cur, &out_w, eps));
    }
    hidden
}

/// Runs the target over `prompt` and the head as the module describes on `backend`: the head's
/// tokens and hidden rows by position, the first position run on its own, and its outputs.
fn run_head(
    gguf: &GgufFile,
    backend: Backend,
    prompt: &[u32],
) -> (Vec<u32>, Vec<Vec<f32>>, usize, Vec<MtpOutput>) {
    let n = prompt.len();
    let blocks: Vec<u32> = (0..(n + 8).div_ceil(KV_BLOCK).max(2) as u32).collect();
    let model = Model::load(gguf, |_| backend.buffer_type()).unwrap();
    let n_layer = model.hp.n_layer;
    let mut e = Executor::new(model, backend, blocks.len(), 1, 0).unwrap();
    e.enable_mtp().unwrap();
    e.capture_hidden(vec![n_layer - 1]);
    let w = e.mtp_width();

    let out = e.step(&[group(prompt, 0, &blocks, Logits::Last)]).unwrap();
    let bonus = out.argmax[0];
    let target = &out.hidden[0];
    assert_eq!(target.len(), n * w);
    // Position p: token p with the target's hidden at p - 1 (zeros at 0); the bonus token last.
    let mut fed = prompt.to_vec();
    fed.push(bonus);
    let mut rows: Vec<Vec<f32>> = vec![vec![0.0; w]];
    rows.extend(target.chunks(w).map(<[f32]>::to_vec));
    let first = fed.len() - PASSES;
    let take_in: Vec<f32> = rows[..first].concat();
    e.mtp_step(&[group(&fed[..first], 0, &blocks, Logits::None)], &take_in)
        .unwrap();
    let mut got = Vec::new();
    for p in first..fed.len() {
        got.push(
            e.mtp_step(&[group(&fed[p..p + 1], p, &blocks, Logits::Last)], &rows[p])
                .unwrap(),
        );
    }
    // A draft fed the head's own output.
    let last = got.last().unwrap();
    let (draft, draft_h) = (last.argmax[0], last.hidden.clone());
    fed.push(draft);
    rows.push(draft_h);
    let p = fed.len() - 1;
    got.push(
        e.mtp_step(&[group(&fed[p..], p, &blocks, Logits::Last)], &rows[p])
            .unwrap(),
    );
    (fed, rows, first, got)
}

fn group<'a>(tokens: &'a [u32], n_past: usize, blocks: &'a [u32], logits: Logits) -> Group<'a> {
    Group {
        slot: 0,
        seqs: vec![SeqInput {
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

#[test]
#[ignore = "needs a Qwen3.5 / 3.8 checkpoint with its MTP head"]
fn real_checkpoint_mtp_head_matches_a_reference() {
    let dir = std::env::var("LEGION_TEST_QWEN35_MTP_HF")
        .expect("LEGION_TEST_QWEN35_MTP_HF=<checkpoint dir>");
    let ck = Checkpoint::open(&dir).unwrap();
    let config = ck.config().expect("config.json").clone();
    let tc = config.get("text_config").unwrap_or(&config).clone();

    let gguf = GgufFile::open_with(&dir, &Default::default()).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let prompt = tok.encode(TEXT, true);
    // The CPU, and in a `cuda` build the GPU.
    let mut backends = vec![{
        let mut b = Backend::cpu().unwrap();
        b.set_threads(8);
        b
    }];
    if cfg!(feature = "cuda")
        && let Some(d) =
            device_by_kind(DeviceKind::Gpu).or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
    {
        backends.push(d.init().unwrap());
    }
    for backend in backends {
        let name = backend.name();
        println!("{name}: prompt {} tokens", prompt.len());
        let (fed, rows, first, got) = run_head(&gguf, backend, &prompt);
        compare(&ck, &tc, &name, &fed, &rows, first, &got);
    }
}

/// Checks the head's outputs `got` (positions `first..`) against the reference on the same
/// inputs.
fn compare(
    ck: &Checkpoint,
    tc: &Value,
    name: &str,
    fed: &[u32],
    rows: &[Vec<f32>],
    first: usize,
    got: &[MtpOutput],
) {
    let want = reference(ck, tc, fed, rows, first..fed.len());
    let lm = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
        .into_iter()
        .find(|n| ck.locate(n).is_some())
        .unwrap();
    let logits = lm_head(ck, lm, &want);
    let mut agree = 0;
    for (i, ((g, h), l)) in got.iter().zip(&want).zip(&logits).enumerate() {
        let p = first + i;
        let mut order: Vec<usize> = (0..l.len()).collect();
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        let (top, second) = (order[0], order[1]);
        let mx = l[top];
        let z: f64 = l.iter().map(|&v| ((v - mx) as f64).exp()).sum();
        let p_top = 1.0 / z;
        let margin = l[top] - l[second];
        let scale = h.iter().fold(0f32, |a, v| a.max(v.abs()));
        let diff = g
            .hidden
            .iter()
            .zip(h)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let dot: f64 = g
            .hidden
            .iter()
            .zip(h)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let na: f64 = g
            .hidden
            .iter()
            .map(|a| (*a as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = h.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        let next = fed.get(p + 1).copied();
        println!(
            "{name} position {p}: legion {} (p {:.3}) reference {top} (p {p_top:.3}, margin {margin:.2}) \
             next token {next:?}; hidden max |diff| {diff:.3e} (scale {scale:.2}), cosine {cos:.6}",
            g.argmax[0], g.pmax[0]
        );
        agree += usize::from(g.argmax[0] == top as u32);
        if margin > 0.5 {
            assert_eq!(g.argmax[0], top as u32, "position {p}: the drafts differ");
            assert!(
                (g.pmax[0] as f64 - p_top).abs() < 0.05,
                "position {p}: probability {} vs {p_top}",
                g.pmax[0]
            );
        }
        assert!(cos > 0.999, "position {p}: hidden cosine {cos}");
    }
    println!("{name}: drafts agree at {agree} of {} positions", got.len());
}
