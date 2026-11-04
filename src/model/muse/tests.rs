use super::*;
use crate::ggml::Backend;

#[test]
fn every_nth_layer_is_global() {
    let swa = swa_layers(8, 4);
    assert_eq!(
        swa,
        [true, true, true, false, true, true, true, false].to_vec()
    );
    assert!(swa_layers(3, 0).iter().all(|&s| s));
    assert!(swa_layers(3, 1).iter().all(|&s| !s));
}

#[test]
fn parses_the_tiny_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let hp = Hparams::from_gguf(&gguf).unwrap();
    assert_eq!(hp.arch.name(), ARCH);
    let p = params(&hp).unwrap();
    assert_eq!(p.swa, swa_layers(tiny::N_LAYER, tiny::PATTERN as usize));
    assert_eq!(p.window, tiny::WINDOW as usize);
    assert_eq!(p.logit_scale, tiny::LOGIT_SCALE);
    assert_eq!(p.softcap, tiny::SOFTCAP);
    assert_eq!(p.rope_base_swa, tiny::ROPE_BASE);
    assert_eq!(hp.head_dim, tiny::HEAD);
    assert_eq!(hp.n_head_kv, tiny::N_HEAD_KV);
    assert_eq!(hp.n_vocab, tiny::n_vocab());
    assert!(hp.arch.sliding_layer(&hp, 0));
    assert!(!hp.arch.sliding_layer(&hp, 3));
}

/// The tiny model's forward pass in plain Rust (f64), from the weights as written: logits for
/// every position of `tokens`, `[n][n_vocab]`.
fn reference(seed: u64, tokens: &[u32]) -> Vec<f32> {
    use std::collections::HashMap;
    let w: HashMap<String, Vec<f64>> = tiny::tensors(seed)
        .into_iter()
        .map(|(n, _, v)| (n, v.into_iter().map(f64::from).collect()))
        .collect();
    let (e, nh, nkv, d, nff) = (
        tiny::E as usize,
        tiny::N_HEAD as usize,
        tiny::N_HEAD_KV as usize,
        tiny::HEAD as usize,
        tiny::N_FF as usize,
    );
    let nv = tiny::n_vocab() as usize;
    let eps = f64::from(tiny::RMS_EPS);
    let rms = |x: &[f64], eps: f64, wt: Option<&[f64]>| -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| v * s * wt.map_or(1.0, |w| w[i]))
            .collect()
    };
    // `W [n_in, n_out] @ x`.
    let mv = |wt: &[f64], x: &[f64]| -> Vec<f64> {
        let n_in = x.len();
        wt.chunks(n_in)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let lw = |il: usize, n: &str| w[&format!("blk.{il}.{n}")].as_slice();
    let rope = |x: &mut [f64], p: usize| {
        for i in 0..d / 2 {
            let theta = p as f64 * f64::from(tiny::ROPE_BASE).powf(-2.0 * i as f64 / d as f64);
            let (s, c) = theta.sin_cos();
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = a * s + b * c;
        }
    };
    let swa = swa_layers(tiny::N_LAYER, tiny::PATTERN as usize);
    let window = tiny::WINDOW as usize;
    let n = tokens.len();
    let mut xs: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&t| {
            let t = t as usize;
            rms(&w["token_embd.weight"][t * e..(t + 1) * e], eps, None)
        })
        .collect();
    for (il, &swa) in swa.iter().enumerate() {
        // Per position: normed input, queries (per head), keys and values (per kv head).
        let mut qs = Vec::new();
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        let mut gates = Vec::new();
        for (p, x) in xs.iter().enumerate() {
            let h = rms(x, eps, Some(lw(il, "attn_norm.weight")));
            let mut q = mv(lw(il, "attn_q.weight"), &h);
            let mut k = mv(lw(il, "attn_k.weight"), &h);
            let v = mv(lw(il, "attn_v.weight"), &h);
            for (t, nw) in [
                (&mut q, "attn_q_norm.weight"),
                (&mut k, "attn_k_norm.weight"),
            ] {
                for head in t.chunks_mut(d) {
                    let normed = rms(head, eps, Some(lw(il, nw)));
                    head.copy_from_slice(&normed);
                    if swa {
                        rope(head, p);
                    }
                }
            }
            gates.push(mv(lw(il, "attn_gate.weight"), &h));
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let scale = 1.0 / (d as f64).sqrt();
        for p in 0..n {
            let lo = if swa {
                (p + 1).saturating_sub(window)
            } else {
                0
            };
            let mut o = vec![0.0; nh * d];
            for hd in 0..nh {
                let kh = hd / (nh / nkv);
                let q = &qs[p][hd * d..(hd + 1) * d];
                let s: Vec<f64> = (lo..=p)
                    .map(|j| {
                        let k = &ks[j][kh * d..(kh + 1) * d];
                        scale * q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>()
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::MIN, f64::max);
                let ex: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = ex.iter().sum();
                for (j, a) in (lo..=p).zip(&ex) {
                    let v = &vs[j][kh * d..(kh + 1) * d];
                    for (oi, vi) in o[hd * d..(hd + 1) * d].iter_mut().zip(v) {
                        *oi += a / z * vi;
                    }
                }
            }
            for (oi, g) in o.iter_mut().zip(&gates[p]) {
                *oi *= 1.0 / (1.0 + (-g).exp());
            }
            let o = mv(lw(il, "attn_output.weight"), &o);
            let o = rms(&o, 1e-8, Some(lw(il, "post_attention_norm.weight")));
            let inp: Vec<f64> = o.iter().zip(&xs[p]).map(|(a, b)| a + b).collect();
            let h = rms(&inp, eps, Some(lw(il, "ffn_norm.weight")));
            let g = mv(lw(il, "ffn_gate.weight"), &h);
            let u = mv(lw(il, "ffn_up.weight"), &h);
            let a: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            assert_eq!(a.len(), nff);
            let f = mv(lw(il, "ffn_down.weight"), &a);
            let f = rms(&f, 1e-8, Some(lw(il, "post_ffw_norm.weight")));
            xs[p] = f.iter().zip(&inp).map(|(a, b)| a + b).collect();
        }
    }
    let c = f64::from(tiny::SOFTCAP);
    let mut out = Vec::with_capacity(n * nv);
    for x in &xs {
        let h = rms(x, eps, Some(&w["output_norm.weight"]));
        for l in mv(&w["output.weight"], &h) {
            let l = l * f64::from(tiny::LOGIT_SCALE);
            out.push((c * (l / c).tanh()) as f32);
        }
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    (diff, a.iter().fold(0f32, |m, v| m.max(v.abs())))
}

/// The whole prompt in one pass, past the window: every position's logits match the plain
/// Rust forward pass.
#[test]
fn prompt_matches_reference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    let m = Model::load(&gguf, |_| b.buffer_type()).unwrap();
    let prompt: Vec<u32> = (0..24).map(|i| (i * 7 + 3) % 50).collect();
    let got = m.prompt_logits_all(&mut b, &prompt).unwrap();
    let want = reference(3, &prompt);
    let (diff, scale) = max_diff(&want, &got);
    assert!(
        diff <= 2e-3 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// A short prompt, then decode token by token past the window: each step's logits match the
/// reference over the whole sequence.
#[test]
fn decode_matches_reference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 5).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    let m = Model::load(&gguf, |_| b.buffer_type()).unwrap();
    let toks: Vec<u32> = (0..20).map(|i| (i * 11 + 5) % 60).collect();
    let mut st = m.new_state(b.buffer_type(), 64).unwrap();
    let mut alloc = None;
    let mut got = m
        .forward(&mut b, &mut alloc, &mut st, &toks[..4], true)
        .unwrap();
    for &t in &toks[4..] {
        got.extend(m.forward(&mut b, &mut alloc, &mut st, &[t], false).unwrap());
    }
    let want = reference(5, &toks);
    let (diff, scale) = max_diff(&want, &got);
    assert!(
        diff <= 2e-3 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// The executor with rings on the sliding-window layers (window 8) against the whole span:
/// the same greedy tokens and last hidden states, through prompts longer than a ring.
#[test]
fn windowed_ring_gather_matches_whole_span() {
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let run = |windowed: bool| -> (Vec<u32>, Vec<f32>) {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let host = Backend::cpu().unwrap().buffer_type();
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let ring = windowed.then(|| crate::model::core::state::KvRing::for_window(8, 64, 1));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 8, 2, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Gather);
        exec.capture_hidden(vec![n_layer - 1]);
        let blocks: Vec<u32> = (0..8).collect();
        let prompt: Vec<u32> = (0..200).map(|i| ((i * 7 + 3) % 50) as u32).collect();
        let mut out = Vec::new();
        let mut hidden = Vec::new();
        let mut n_past = 0;
        let mut last = 0;
        let step = |exec: &mut Executor, toks: &[u32], n_past: usize| {
            let r = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::Greedy,
                        media: None,
                    }],
                }])
                .unwrap();
            (r.argmax[0], r.hidden[0].clone())
        };
        for chunk in prompt.chunks(64) {
            (last, _) = step(&mut exec, chunk, n_past);
            n_past += chunk.len();
        }
        for _ in 0..16 {
            let (t, h) = step(&mut exec, &[last], n_past);
            out.push(t);
            hidden.extend(h);
            n_past += 1;
            last = t;
        }
        (out, hidden)
    };
    let (whole, hb) = run(false);
    let (windowed, hw) = run(true);
    assert_eq!(whole, windowed);
    let (diff, scale) = max_diff(&hb, &hw);
    assert!(
        diff <= 1e-3 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// Layers captured for a drafter in an early pipeline stage come out of that stage: two CPU
/// stages capture the same hidden states as one.
#[test]
fn captures_span_pipeline_stages() {
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let run = |split: bool| -> (Vec<u32>, Vec<Vec<f32>>) {
        let cpu = || {
            let mut b = Backend::cpu().unwrap();
            b.set_threads(2);
            b
        };
        let b = cpu();
        let m = Model::load(&gguf, |_| b.buffer_type()).unwrap();
        let n_layer = m.hp.n_layer;
        let stages = if split {
            vec![(b, 0..3), (cpu(), 3..n_layer)]
        } else {
            vec![(b, 0..n_layer)]
        };
        let mut exec =
            Executor::new_staged_ring(m, stages, 8, 2, 0, None, Default::default()).unwrap();
        exec.set_attention(Attention::Gather);
        exec.capture_hidden(vec![0, 2, 4]);
        let blocks: Vec<u32> = (0..8).collect();
        let prompt: Vec<u32> = (0..30).map(|i| ((i * 7 + 3) % 50) as u32).collect();
        let r = exec
            .step(&[Group {
                slot: 0,
                verify: false,
                seqs: vec![SeqInput {
                    tokens: &prompt,
                    n_past: 0,
                    blocks: &blocks,
                    ring: 0,
                    logits: Logits::Greedy,
                    media: None,
                }],
            }])
            .unwrap();
        (r.argmax, r.hidden)
    };
    let (one, h1) = run(false);
    let (two, h2) = run(true);
    assert_eq!(one, two);
    assert_eq!(h1.len(), 3);
    assert_eq!(h2.len(), 3);
    for (a, b) in h1.iter().zip(&h2) {
        let (diff, scale) = max_diff(a, b);
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "max |diff| {diff} of {scale}"
        );
    }
}
