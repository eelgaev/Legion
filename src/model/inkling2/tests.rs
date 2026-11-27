use std::collections::HashMap;

use super::*;
use crate::cpu::gpu_exact::{f16_to_f32, f32_to_f16};
use crate::ggml::Backend;
use crate::gguf::Value;
#[cfg(feature = "cuda")]
use crate::model::core::TracedTensor;
use tiny::Layout;

fn tiny_gguf(dir: &tempfile::TempDir, layout: Layout) -> GgufFile {
    let path = dir.path().join(format!("tiny-{layout:?}.gguf"));
    tiny::write(&path, 7, layout).unwrap();
    GgufFile::open(&path).unwrap()
}

fn backend() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b
}

/// Past the sliding window, the relative-bias extents and the log-scale floor.
fn prompt() -> Vec<u32> {
    (0..37).map(|i| (i * 37 + 11) % 250).collect()
}

/// The largest `|a - b|`: a NaN on either side, or an infinity on one only, is infinitely far
/// (`f32::max` alone would drop a NaN).
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            if x.is_nan() || y.is_nan() {
                f32::INFINITY
            } else if x == y {
                0.0
            } else {
                (x - y).abs()
            }
        })
        .fold(0.0, f32::max)
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |a, x| a.max(x.abs()))
}

/// Runs `prompt` in chunks of the sizes `chunks` (the rest one by one), every position's
/// logits.
fn chunked(m: &Model, prompt: &[u32], chunks: &[usize]) -> Vec<f32> {
    let mut b = backend();
    let mut st = m.new_state(b.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut out = Vec::new();
    let mut at = 0;
    for &c in chunks {
        out.extend(
            m.forward(&mut b, &mut alloc, &mut st, &prompt[at..at + c], true)
                .unwrap(),
        );
        at += c;
    }
    for &t in &prompt[at..] {
        out.extend(m.forward(&mut b, &mut alloc, &mut st, &[t], true).unwrap());
    }
    out
}

#[test]
fn both_layouts_parse_to_the_same_hyperparameters() {
    let dir = tempfile::tempdir().unwrap();
    let new = Hparams::from_gguf(&tiny_gguf(&dir, Layout::Mainline)).unwrap();
    let old = Hparams::from_gguf(&tiny_gguf(&dir, Layout::Old)).unwrap();
    assert_eq!(new.arch.name(), ARCH);
    assert_eq!(old.arch.name(), ARCH_OLD);
    let (p, q) = (params(&new).unwrap(), params(&old).unwrap());
    assert!(!p.old_layout && q.old_layout);
    assert_eq!(
        Ink2Params {
            old_layout: false,
            ..q.clone()
        },
        *p
    );
    assert_eq!(new.n_layer, tiny::n_slots());
    assert_eq!(new.n_layer, old.n_layer);
    assert_eq!(p.n_phys, tiny::N_PHYS);
    assert_eq!(p.core, tiny::CORE);
    assert_eq!(p.passes, tiny::PASSES);
    assert_eq!(p.n_vocab, tiny::n_vocab());
    assert_eq!(new.n_vocab, tiny::n_vocab());
    // Slots 8..12 and 12..16 run the core again.
    let phys: Vec<usize> = (0..new.n_layer).map(|il| p.phys(il)).collect();
    assert_eq!(phys, [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 2, 3, 4, 5]);
    assert_eq!(p.slot(0, 3), 3);
    assert_eq!(p.slot(1, 2), 8);
    assert_eq!(p.slot(2, 5), 15);
    for il in 0..new.n_layer {
        assert_eq!(new.arch.sliding_layer(&new, il), tiny::swa(p.phys(il)));
    }
    assert_eq!(new.arch.swa_window(), tiny::WINDOW as usize);
    let c = new.arch.token_conv(&new).expect("the conv history");
    assert_eq!(c.history, tiny::CONV - 1);
    let kvw = tiny::HEAD * tiny::N_HEAD_KV;
    assert_eq!(c.width, 16 * (2 * kvw + 2 * tiny::E));
    assert!(new.has_slot_state() && !new.has_recurrent());
}

#[test]
fn controller_names_follow_the_layout() {
    let dir = tempfile::tempdir().unwrap();
    let new = Hparams::from_gguf(&tiny_gguf(&dir, Layout::Mainline)).unwrap();
    let old = Hparams::from_gguf(&tiny_gguf(&dir, Layout::Old)).unwrap();
    let (p, q) = (params(&new).unwrap(), params(&old).unwrap());
    assert_eq!(p.ctrl_name(6, 0, "down"), "blk.6.loop_q_ctrl_down.weight");
    assert_eq!(p.ctrl_name(7, 3, "gain"), "blk.7.loop_ffn_ctrl_gain.weight");
    assert_eq!(q.ctrl_name(6, 0, "down"), "blk.6.loop_attn_down.weight");
    assert_eq!(q.ctrl_name(7, 3, "forget"), "blk.7.loop_mlp_forget.weight");
    assert_eq!(q.ctrl_name(8, 2, "gain"), "blk.8.loop_v_gain.weight");
}

#[test]
fn bad_loops_and_shapes_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let err = |edit: &dyn Fn(&str) -> Option<Value>| {
        let path = dir.path().join("edited.gguf");
        tiny::write_edited(&path, 7, Layout::Mainline, edit).unwrap();
        Hparams::from_gguf(&GgufFile::open(&path).unwrap())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    };
    let e = err(&|k| (k == "loop.start").then_some(Value::U32(3)));
    assert!(e.contains("invalid loop core"), "{e}");
    let e = err(&|k| (k == "loop.count").then_some(Value::U32(1)));
    assert!(e.contains("invalid loop core"), "{e}");
    let e = err(&|k| (k == "loop.varying").then_some(Value::Bool(true)));
    assert!(e.contains("variable number of loop passes"), "{e}");
    let e = err(&|k| (k == "attention.key_length").then_some(Value::U32(48)));
    assert!(e.contains("head sizes") || e.contains("heads of 48"), "{e}");
    let e = err(&|k| (k == "shortconv.l_cache").then_some(Value::U32(9)));
    assert!(e.contains("taps"), "{e}");
    // A loop count whose slots overflow is an error, not a panic or a wrapped count.
    let e = err(&|k| (k == "loop.count").then_some(Value::U64(u64::MAX)));
    assert!(e.contains("loop.count"), "{e}");
    let e = err(&|k| (k == "expert_used_count").then_some(Value::U32(tiny::N_EXPERT as u32 + 1)));
    assert!(e.contains("expert_used_count"), "{e}");
    let e = err(&|k| (k == "expert_used_count").then_some(Value::U32(0)));
    assert!(e.contains("expert_used_count"), "{e}");
}

#[test]
fn unknown_layouts_name_both_architectures() {
    assert!(crate::model::ARCHITECTURES.contains("inkling2_loop"));
    assert!(crate::model::ARCHITECTURES.contains(", inkling2 "));
}

/// The tiny model's weights by name, for [`reference`].
struct Weights(HashMap<String, (Vec<i64>, Vec<f32>)>);

impl Weights {
    fn new(seed: u64) -> Self {
        Self(
            tiny::tensors(seed)
                .into_iter()
                .map(|(n, s, v)| (n, (s, v)))
                .collect(),
        )
    }

    fn w(&self, name: &str) -> &[f32] {
        &self.0.get(name).unwrap_or_else(|| panic!("{name}")).1
    }

    fn l(&self, ip: usize, name: &str) -> &[f32] {
        self.w(&format!("blk.{ip}.{name}"))
    }

    fn has(&self, ip: usize, name: &str) -> bool {
        self.0.contains_key(&format!("blk.{ip}.{name}"))
    }
}

type Act = Vec<Vec<f32>>;

/// `w` (ggml `[n_in, n_out]`) times `x`.
fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
    let n_in = x.len();
    w.chunks(n_in)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn mm(w: &[f32], x: &Act) -> Act {
    x.iter().map(|x| mv(w, x)).collect()
}

fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let s = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * s * w.map_or(1.0, |w| w[i]))
        .collect()
}

fn rms_all(x: &Act, w: &[f32]) -> Act {
    x.iter().map(|x| rms(x, Some(w), tiny::RMS_EPS)).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The depth mix (llama.cpp's `mix`): per token a softmax over the RMS-normed states of their
/// products with each query `/ sqrt(n_embd)`, the normed states summed by it.
fn mix(states: &[&Act], q: &[f32]) -> Vec<Act> {
    let e = states[0][0].len();
    let nq = q.len() / e;
    let n = states[0].len();
    (0..nq)
        .map(|j| {
            let q = &q[j * e..][..e];
            (0..n)
                .map(|t| {
                    let normed: Vec<Vec<f32>> =
                        states.iter().map(|s| rms(&s[t], None, MIX_EPS)).collect();
                    let scores: Vec<f32> = normed
                        .iter()
                        .map(|s| {
                            s.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() / (e as f32).sqrt()
                        })
                        .collect();
                    let top = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
                    let p: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                    let sum: f32 = p.iter().sum();
                    (0..e)
                        .map(|i| normed.iter().zip(&p).map(|(s, p)| s[i] * p / sum).sum())
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// A loop controller: `cur - tanh(forget) cur + tanh(gain) msg` over `down · [rms cur | rms msg]`.
fn control(w: &Weights, ip: usize, c: usize, cur: &Act, msg: &Act) -> Act {
    let name = |part: &str| format!("loop_{}_ctrl_{part}.weight", CTRL_NAMES[c]);
    cur.iter()
        .zip(msg)
        .map(|(x, m)| {
            let mut g = rms(x, None, MIX_EPS);
            g.extend(rms(m, None, MIX_EPS));
            let g = mv(w.l(ip, &name("down")), &g);
            let gain = mv(w.l(ip, &name("gain")), &g);
            let forget = mv(w.l(ip, &name("forget")), &g);
            (0..x.len())
                .map(|i| x[i] - forget[i].tanh() * x[i] + gain[i].tanh() * m[i])
                .collect()
        })
        .collect()
}

/// The causal short conv with its residual (llama.cpp's `convolve`), from a zero history:
/// `taps` (`[K, c]`: tap `K - 1` on the token itself), plus the input-dependent taps `dynamic`
/// (`[K · heads]` a token, lag-major within a head), scaled by `1 / sqrt(taps + 1)`.
fn conv(x: &Act, taps: &[f32], dynamic: Option<&Act>) -> Act {
    let k = tiny::CONV as usize;
    let c = x[0].len();
    let at = |t: isize, ch: usize| if t < 0 { 0.0 } else { x[t as usize][ch] };
    let scale = 1.0 / ((if dynamic.is_some() { 2 * k } else { k } + 1) as f32).sqrt();
    (0..x.len())
        .map(|t| {
            (0..c)
                .map(|ch| {
                    let mut s = x[t][ch];
                    for j in 0..k {
                        s += taps[ch * k + j] * at(t as isize + j as isize - (k as isize - 1), ch);
                    }
                    if let Some(d) = dynamic {
                        let heads = d[t].len() / k;
                        let head = ch / (c / heads);
                        for lag in 0..k {
                            s += at(t as isize - lag as isize, ch) * d[t][head * k + lag];
                        }
                    }
                    s * scale
                })
                .collect()
        })
        .collect()
}

/// Rows of `d` scaled by `s` (`[d, h]`), clipped to an L2 norm of `thr`, times `tau`.
fn clip(x: &mut [f32], d: usize, s: Option<&[f32]>, thr: f32, tau: f32) {
    for (h, row) in x.chunks_mut(d).enumerate() {
        if let Some(s) = s {
            for (i, v) in row.iter_mut().enumerate() {
                *v *= s[h * d + i];
            }
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
        let f = (thr / norm).clamp(0.0, 1.0) * tau;
        row.iter_mut().for_each(|v| *v *= f);
    }
}

fn f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// Block `ip`'s attention over its inputs (normed), the whole prompt at positions `0..n`.
fn attention(w: &Weights, p: &Ink2Params, ip: usize, inputs: &[Act; 3]) -> Act {
    let (d, nh, nkv) = (
        tiny::HEAD as usize,
        tiny::N_HEAD as usize,
        tiny::N_HEAD_KV as usize,
    );
    let nr = tiny::N_REL as usize;
    let swa = p.swa[ip];
    let n = inputs[0].len();
    let tau = |t: usize| if swa { 1.0 } else { p.log_scale.at(t as i32) };
    let mut q = mm(w.l(ip, "attn_q.weight"), &inputs[0]);
    let kp = mm(w.l(ip, "attn_k.weight"), &inputs[1]);
    let vp = mm(w.l(ip, "attn_v.weight"), &inputs[2]);
    let (kd, vd) = if swa {
        (None, None)
    } else {
        (
            Some(mm(w.l(ip, "attn_k_conv_proj.weight"), &inputs[1])),
            Some(mm(w.l(ip, "attn_v_conv_proj.weight"), &inputs[2])),
        )
    };
    let mut k = conv(&kp, w.l(ip, "attn_k_conv.weight"), kd.as_ref());
    let mut v = conv(&vp, w.l(ip, "attn_v_conv.weight"), vd.as_ref());
    let mut r = mm(w.l(ip, "attn_rel.weight"), &inputs[0]);
    let sinks = mm(w.l(ip, "attn_sink_proj.weight"), &inputs[0]);
    for t in 0..n {
        clip(
            &mut q[t],
            d,
            Some(w.l(ip, "attn_q_clip.weight")),
            p.qk_clip,
            tau(t),
        );
        clip(&mut k[t], d, None, p.qk_clip, 1.0);
        clip(
            &mut r[t],
            nr,
            Some(w.l(ip, "attn_rel_clip.weight")),
            p.rel_clip,
            tau(t),
        );
        let vn: Vec<f32> = v[t]
            .chunks(d)
            .flat_map(|h| rms(h, None, tiny::RMS_EPS))
            .collect();
        v[t] = vn;
        // The cache keeps K and V in F16.
        k[t].iter_mut()
            .chain(v[t].iter_mut())
            .for_each(|x| *x = f16(*x));
    }
    let rel = w.l(ip, "attn_rel_bias.weight");
    let extent = rel.len() / nr;
    let scale = 1.0 / (d as f32).sqrt();
    let gate = mm(w.l(ip, "attn_gate.weight"), &inputs[0]);
    let o: Act = (0..n)
        .map(|t| {
            let mut out = vec![0f32; nh * d];
            let lo = if swa {
                (t + 1).saturating_sub(p.window)
            } else {
                0
            };
            for h in 0..nh {
                let g = h / (nh / nkv);
                let qh = &q[t][h * d..][..d];
                let logits: Vec<f32> = (lo..=t)
                    .map(|j| {
                        let dot: f32 = qh.iter().zip(&k[j][g * d..][..d]).map(|(a, b)| a * b).sum();
                        let dist = t - j;
                        let bias: f32 = if dist < extent {
                            (0..nr).map(|i| r[t][h * nr + i] * rel[dist * nr + i]).sum()
                        } else {
                            0.0
                        };
                        dot * scale + bias
                    })
                    .collect();
                let sink = sinks[t][h];
                let top = logits.iter().fold(sink, |a, &b| a.max(b));
                let e: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
                let sum = e.iter().sum::<f32>() + (sink - top).exp();
                for (jj, j) in (lo..=t).enumerate() {
                    for i in 0..d {
                        out[h * d + i] += e[jj] / sum * v[j][g * d + i];
                    }
                }
            }
            out.iter()
                .zip(&gate[t])
                .map(|(o, g)| o * sigmoid(*g))
                .collect()
        })
        .collect();
    conv(
        &mm(w.l(ip, "attn_output.weight"), &o),
        w.l(ip, "attn_post_conv.weight"),
        None,
    )
}

/// Block `ip`'s routed experts over the normed `h`.
fn experts(w: &Weights, p: &Ink2Params, ip: usize, h: &Act) -> Act {
    let (e, f) = (tiny::E as usize, tiny::N_FF as usize);
    let gu = w.l(ip, "ffn_gate_up_exps.weight");
    let down = w.l(ip, "ffn_down_exps.weight");
    let bias = w.l(ip, "exp_probs_b.bias");
    h.iter()
        .map(|x| {
            let probs: Vec<f32> = mv(w.l(ip, "ffn_gate_inp.weight"), x)
                .iter()
                .map(|&l| sigmoid(l))
                .collect();
            let mut order: Vec<usize> = (0..probs.len()).collect();
            order.sort_by(|&a, &b| (probs[b] + bias[b]).total_cmp(&(probs[a] + bias[a])));
            let chosen = &order[..tiny::N_USED as usize];
            let sum: f32 = chosen.iter().map(|&i| probs[i]).sum();
            let mut y = vec![0f32; e];
            for &ex in chosen {
                let wt = probs[ex] / sum * p.weights_scale;
                let g = mv(&gu[ex * 2 * f * e..][..2 * f * e], x);
                let a: Vec<f32> = (0..f)
                    .map(|i| {
                        let (gv, uv) = (g[i], g[f + i]);
                        let (b, lb) = (p.situ_beta, p.situ_lb);
                        b * (gv / b).tanh() * sigmoid(gv) * lb * (uv / lb).tanh()
                    })
                    .collect();
                let o = mv(&down[ex * e * f..][..e * f], &a);
                y.iter_mut().zip(o).for_each(|(y, o)| *y += wt * o);
            }
            y
        })
        .collect()
}

fn add(a: &Act, b: &Act) -> Act {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(a, b)| a + b).collect())
        .collect()
}

/// One layer of block `ip` (llama.cpp's `build_layer`).
fn layer(
    w: &Weights,
    p: &Ink2Params,
    ip: usize,
    cur: Act,
    blocks: &mut Vec<Act>,
    msgs: Option<&[Act]>,
) -> Act {
    let swa = p.swa[ip];
    let states = |blocks: &[Act], cur: &Act| -> Vec<Act> {
        let mut s = blocks.to_vec();
        s.push(cur.clone());
        s
    };
    let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
    if w.has(ip, "attn_res_query.weight") {
        let s = states(blocks, &cur);
        let refs: Vec<&Act> = s.iter().collect();
        let mixed = mix(&refs, w.l(ip, "attn_res_query.weight"));
        for (j, x) in inputs.iter_mut().enumerate() {
            *x = mixed[if swa { 0 } else { j }].clone();
        }
    }
    if let Some(m) = msgs {
        for (j, x) in inputs.iter_mut().enumerate() {
            *x = control(w, ip, j, x, &m[j]);
        }
    }
    let inputs = if swa && msgs.is_some() {
        inputs.map(|x| rms_all(&x, w.l(ip, "attn_norm.weight")))
    } else if swa {
        let x = rms_all(&inputs[0], w.l(ip, "attn_norm.weight"));
        [x.clone(), x.clone(), x]
    } else {
        let [a, b, c] = inputs;
        [
            rms_all(&a, w.l(ip, "attn_q_inp_norm.weight")),
            rms_all(&b, w.l(ip, "attn_k_inp_norm.weight")),
            rms_all(&c, w.l(ip, "attn_v_inp_norm.weight")),
        ]
    };
    let out = attention(w, p, ip, &inputs);
    let cur = if ip % p.block == 0 {
        blocks.push(cur);
        out
    } else {
        add(&cur, &out)
    };
    let s = states(blocks, &cur);
    let refs: Vec<&Act> = s.iter().collect();
    let mut h = mix(&refs, w.l(ip, "ffn_res_query.weight")).remove(0);
    if let Some(m) = msgs {
        h = control(w, ip, 3, &h, &m[3]);
    }
    let h = rms_all(&h, w.l(ip, "ffn_norm.weight"));
    let y = conv(
        &experts(w, p, ip, &h),
        w.l(ip, "ffn_post_conv.weight"),
        None,
    );
    add(&cur, &y)
}

/// The tiny model's logits of every position of `tokens` in plain Rust, after llama.cpp's
/// `inkling2.cpp` and `inkling2-loop.cpp` graphs.
fn reference(w: &Weights, p: &Ink2Params, tokens: &[u32]) -> Vec<f32> {
    let e = tiny::E as usize;
    let embd = w.w("token_embd.weight");
    let mut cur: Act = tokens
        .iter()
        .map(|&t| {
            let row = &embd[t as usize * e..][..e];
            rms(row, Some(w.w("token_embd_norm.weight")), tiny::RMS_EPS)
        })
        .collect();
    let mut blocks = Vec::new();
    for ip in 0..p.core.start {
        cur = layer(w, p, ip, cur, &mut blocks, None);
    }
    let (prelude, prelude_blocks) = (cur.clone(), blocks.clone());
    for pass in 0..p.passes {
        let mut bank: Vec<Act> = Vec::new();
        if pass > 0 {
            bank = blocks[prelude_blocks.len() + 1..].to_vec();
            bank.push(cur);
            cur = prelude.clone();
            blocks = prelude_blocks.clone();
        }
        for ip in p.core.clone() {
            let msgs = (pass > 0).then(|| {
                let refs: Vec<&Act> = bank.iter().collect();
                mix(&refs, w.l(ip, "loop_msg_query.weight"))
            });
            cur = layer(w, p, ip, cur, &mut blocks, msgs.as_deref());
        }
    }
    for ip in p.core.end..p.n_phys {
        cur = layer(w, p, ip, cur, &mut blocks, None);
    }
    blocks.push(cur);
    let refs: Vec<&Act> = blocks.iter().collect();
    let x = mix(&refs, w.w("output_res_query.weight")).remove(0);
    let x = rms_all(&x, w.w("output_norm.weight"));
    let nv = p.n_vocab as usize;
    x.iter()
        .flat_map(|x| {
            mv(w.w("output.weight"), x)[..nv]
                .iter()
                .map(|l| p.softcap * (l / p.softcap).tanh())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The CPU graph (Legion's ops and their CPU references) gives the plain-Rust reference's
/// logits: the loop, the depth mixes, the controllers, the convs, the relative bias, sinks,
/// window and log scale, the router and the head.
#[test]
fn tiny_matches_the_reference() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let p = prompt();
    let got = m.prompt_logits_all(&mut b, &p).unwrap();
    let want = reference(&Weights::new(7), m.ink(), &p);
    let scale = scale_of(&want);
    assert!(scale > 0.1, "logits of a real computation ({scale})");
    let d = max_diff(&got, &want);
    assert!(
        d <= 2e-3 * scale,
        "the graph against the reference: {d} of {scale}"
    );
}

/// The earlier layout (other controller names, message queries in two global tables, 16-entry
/// arrays) gives the same logits as the mainline one, bit for bit.
#[test]
fn tiny_old_layout_gives_the_same_logits() {
    let dir = tempfile::tempdir().unwrap();
    let p = prompt();
    let run = |layout| {
        let g = tiny_gguf(&dir, layout);
        let mut b = backend();
        let m = Model::load(&g, |_| b.buffer_type()).unwrap();
        m.prompt_logits_all(&mut b, &p).unwrap()
    };
    assert_eq!(run(Layout::Mainline), run(Layout::Old));
}

/// One prompt, the same prompt in uneven chunks and token by token give the same logits up to
/// rounding (the KV caches and conv histories carried between steps), deterministically.
#[test]
fn tiny_runs_the_same_in_any_chunking() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let p = prompt();
    let once = m.prompt_logits_all(&mut b, &p).unwrap();
    let nv = tiny::n_vocab() as usize;
    assert_eq!(once.len(), p.len() * nv);
    assert!(once.iter().all(|v| v.is_finite()));
    assert!(max_diff(&once[..nv], &once[nv..2 * nv]) > 1e-3);
    assert_eq!(once, m.prompt_logits_all(&mut b, &p).unwrap());
    let tol = 2e-4 * scale_of(&once);
    let split = chunked(&m, &p, &[13, 7, 11]);
    let d = max_diff(&once, &split);
    assert!(d <= tol, "chunked prompt: {d} > {tol}");
    let steps = chunked(&m, &p, &[1]);
    let d = max_diff(&once, &steps);
    assert!(d <= tol, "token by token: {d} > {tol}");
}

/// #1221: every op of a decode step (one token, and four of one sequence) is one that a small
/// step's stage graph captures once a loaded Inkling-2 turns the shared ops' switch on, so the
/// step is one capture instead of ~2,000 launches from the host. A new op left out of the
/// lists fails here.
#[test]
fn a_decode_step_is_captured_whole() {
    use crate::model::core::stage_ops::{INKLING, INKLING_SHARED, listed_as};
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let mut st = m.new_state(b.buffer_type(), 256).unwrap();
    let mut alloc = None;
    m.forward(&mut b, &mut alloc, &mut st, &prompt(), false)
        .unwrap();
    for tokens in [&[7u32][..], &[7, 8, 9, 10]] {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        m.step_external_ops(&st, tokens, |op, _| {
            *counts.entry(op.name).or_default() += 1;
        })
        .unwrap();
        let total: usize = counts.values().sum();
        eprintln!("{} tokens: {total} external ops {counts:?}", tokens.len());
        assert!(counts.contains_key("ink_attn") && counts.contains_key("glm_rms_norm"));
        for name in counts.keys() {
            assert!(listed_as(name, true, true), "`{name}` cuts the capture");
            if INKLING.contains(name) || INKLING_SHARED.contains(name) {
                assert!(!listed_as(name, false, true), "`{name}` in prompt steps");
            }
        }
    }
}

/// The engine's batch steps (paged caches, two sequences, the sliding slots in rings or whole)
/// give the tokens of whole caches: the conv histories are per sequence.
#[test]
fn tiny_rings_and_batches_give_the_same_tokens() {
    use crate::model::core::state::KvRing;
    use crate::model::core::{Attention, Executor, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    const STEP: usize = 16;
    let run = |ring: bool, together: bool| -> Vec<u32> {
        let b = backend();
        let m = Model::load(&g, |_| b.buffer_type()).unwrap();
        let n_layer = m.hp.n_layer;
        let ring = ring.then(|| KvRing::for_window(m.hp.arch.swa_window(), STEP, 2));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 16, 3, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Paged);
        let blocks: Vec<Vec<u32>> = vec![(0..8).collect(), (8..16).collect()];
        let mut n_past = [0usize; 2];
        let mut out = Vec::new();
        let mut step = |exec: &mut Executor, seqs: &[(usize, Vec<u32>)]| -> Vec<u32> {
            let groups: Vec<Group> = seqs
                .iter()
                .map(|(s, toks)| Group {
                    slot: *s,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past: n_past[*s],
                        blocks: &blocks[*s],
                        logits: Logits::Greedy,
                        ring: *s,
                        media: None,
                    }],
                })
                .collect();
            let r = exec.step(&groups).unwrap();
            for (s, toks) in seqs {
                n_past[*s] += toks.len();
            }
            r.argmax
        };
        // The long prompt runs past the ring's first reuse, and (after a first chunk of 5) one
        // step crosses into the reused entry while its first query reads the block before.
        let reuse = KvRing::for_window(tiny::WINDOW as usize, STEP, 2).blocks
            * crate::model::core::KV_BLOCK;
        let long: Vec<u32> = (0..reuse + 20)
            .map(|i| (i * 37 + 11) as u32 % 250)
            .collect();
        let short: Vec<u32> = prompt()[..30].to_vec();
        let mut last = [0u32; 2];
        for (s, p) in [long, short].iter().enumerate() {
            for c in std::iter::once(&p[..5]).chain(p[5..].chunks(STEP)) {
                let r = step(&mut exec, &[(s, c.to_vec())]);
                last[s] = r[0];
                out.extend(r);
            }
        }
        for _ in 0..4 {
            if together {
                let r = step(&mut exec, &[(0, vec![last[0]]), (1, vec![last[1]])]);
                last = [r[0], r[1]];
                out.extend(r);
            } else {
                let mut r = Vec::new();
                for s in 0..2 {
                    r.extend(step(&mut exec, &[(s, vec![last[s]])]));
                }
                last = [r[0], r[1]];
                out.extend(r);
            }
        }
        out
    };
    let whole = run(false, false);
    assert_eq!(whole, run(false, true), "batched decode changes the tokens");
    assert_eq!(whole, run(true, true), "rings change the tokens");
}

/// Runs `f` with `route` as this thread's [`super::FORCED_ROUTE`], cleared after (also when
/// `f` panics).
fn with_route<T>(route: &[i32], f: impl FnOnce() -> T) -> T {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            super::FORCED_ROUTE.with(|r| *r.borrow_mut() = None);
        }
    }
    super::FORCED_ROUTE.with(|r| *r.borrow_mut() = Some(route.to_vec()));
    let _clear = Clear;
    f()
}

/// The route a traced prompt took: every slot's `ffn_moe_topk`, slot after slot.
fn route_of(trace: &crate::model::core::Trace) -> Vec<i32> {
    (0..tiny::n_slots())
        .flat_map(|il| {
            let name = format!("ffn_moe_topk-{il}");
            let t = (trace.1.iter().find(|t| t.name == name)).unwrap_or_else(|| panic!("{name}"));
            t.data.iter().map(|&x| x as i32).collect::<Vec<_>>()
        })
        .collect()
}

/// Token `t`'s part of a prompt's route of `n` tokens: per slot its `N_USED` ids.
fn route_at(route: &[i32], n: usize, t: usize) -> Vec<i32> {
    let k = tiny::N_USED as usize;
    route
        .chunks(k * n)
        .flat_map(|slot| slot[t * k..(t + 1) * k].to_vec())
        .collect()
}

/// A forced route ([`super::FORCED_ROUTE`]) of the experts the router picks anyway changes
/// nothing: the same logits and intermediates to the bit, in one prompt and token by token.
#[test]
fn a_route_forced_to_its_own_choices_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let p = prompt();
    let mut b = backend();
    let m = Model::load(&g, |_| b.buffer_type()).unwrap();
    let own = m.prompt_trace(&mut b, &p).unwrap();
    let route = route_of(&own);
    let forced = with_route(&route, || m.prompt_trace(&mut b, &p).unwrap());
    assert_eq!(own.0, forced.0, "logits");
    for t in &own.1 {
        let f = forced.1.iter().find(|f| f.name == t.name).unwrap();
        assert_eq!(t.data, f.data, "{}", t.name);
    }
    let router = forced
        .1
        .iter()
        .filter(|t| t.name.starts_with("ffn_moe_router-"));
    assert_eq!(
        router.count(),
        tiny::n_slots(),
        "the router's own choices traced"
    );
    let mut st = m.new_state(b.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut steps = Vec::new();
    for (i, &t) in p.iter().enumerate() {
        let r = route_at(&route, p.len(), i);
        steps.extend(with_route(&r, || {
            m.forward(&mut b, &mut alloc, &mut st, &[t], true).unwrap()
        }));
    }
    assert_eq!(steps, chunked(&m, &p, &[]), "token by token");
}

/// The whole tiny graph on the GPU (the CUDA kernels of [`ops`] and GLM's dense ones) gives
/// the CPU's logits up to rounding, in one prompt and token by token, on the CPU's route:
/// the tiny model's random router has near ties that the GPU's rounding may break the other
/// way (then every later slot differs), so the routed experts take the CPU's choices
/// ([`super::FORCED_ROUTE`]) and [`cuda_tiny_traces_match_the_cpu`] checks the GPU router's
/// own. The GPU's logits on its own route are printed.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_matches_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let p = prompt();
    let mut b = backend();
    let cm = Model::load(&g, |_| b.buffer_type()).unwrap();
    let cpu = cm.prompt_logits_all(&mut b, &p).unwrap();
    let route = route_of(&cm.prompt_trace(&mut b, &p).unwrap());
    let mut gpu = gpu_backend();
    let m = Model::load(&g, |_| gpu.buffer_type()).unwrap();
    let tol = 5e-3 * scale_of(&cpu);
    let own = m.prompt_logits_all(&mut gpu, &p).unwrap();
    eprintln!(
        "the GPU on its own route: {:.3e} (tolerance {tol:.3e})",
        max_diff(&own, &cpu)
    );
    let once = with_route(&route, || m.prompt_logits_all(&mut gpu, &p).unwrap());
    let d = max_diff(&once, &cpu);
    assert!(d <= tol, "one prompt on the GPU: {d} > {tol}");
    let mut st = m.new_state(gpu.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut steps = Vec::new();
    for (i, &t) in p.iter().enumerate() {
        let r = route_at(&route, p.len(), i);
        steps.extend(with_route(&r, || {
            m.forward(&mut gpu, &mut alloc, &mut st, &[t], true)
                .unwrap()
        }));
    }
    let d = max_diff(&steps, &cpu);
    assert!(d <= tol, "token by token on the GPU: {d} > {tol}");
}

/// The first GPU device.
#[cfg(feature = "cuda")]
fn gpu_backend() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// #1221: a prompt step's conv on the GPU (a thread per token, then the history) gives the
/// bits of the decode steps' token walk, outputs and history, with and without dynamic taps,
/// for steps longer and shorter than the history.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_conv_prompt_steps_are_the_token_walk() {
    use crate::ggml::ComputeAllocator;
    let mut gpu = gpu_backend();
    let vals = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };
    let (k, c, heads, width, off, scale) = (4i64, 96i64, 4i64, 160i64, 32i64, 0.75f32);
    for (n_tok, n_seqs, dynamic) in [(37i64, 2i64, true), (37, 2, false), (2, 3, true)] {
        let ng = n_tok * n_seqs;
        let ctx = Context::new(64 + 16 * n_tok as usize, 64 + 16 * n_tok as usize);
        let x = ctx.new_tensor(types::F32, &[c, ng]).as_input();
        let w = ctx.new_tensor(types::F32, &[k, c]).as_input();
        let d = ctx.new_tensor(types::F32, &[k * heads, ng]).as_input();
        let st = [0, 1].map(|_| {
            ctx.new_tensor(types::F32, &[k - 1, width, n_seqs])
                .as_input()
                .as_output()
        });
        let once = ops::conv(&ctx, x, w, st[0], dynamic.then_some(d), n_tok, off, scale);
        let once = once.as_output();
        let mut graph = ctx.new_graph(64 + 8 * n_tok as usize);
        graph.expand(once);
        // The walk's inputs: token `i` of every sequence (columns `s · n_tok + i`), each its
        // own contiguous `[rows, n_seqs]` tensor, as the ops take them.
        let mut walk = Vec::new();
        let mut walk_in = Vec::new();
        for _ in 0..n_tok {
            let xi = ctx.new_tensor(types::F32, &[c, n_seqs]).as_input();
            let di = dynamic.then(|| ctx.new_tensor(types::F32, &[k * heads, n_seqs]).as_input());
            let o = ops::conv(&ctx, xi, w, st[1], di, 1, off, scale);
            let o = o.as_output();
            graph.expand(o);
            walk.push(o);
            walk_in.push((xi, di));
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        let (xv, dv) = (
            vals((c * ng) as usize, 1),
            vals((k * heads * ng) as usize, 3),
        );
        x.set_f32(&xv);
        w.set_f32(&vals((k * c) as usize, 2));
        if dynamic {
            d.set_f32(&dv);
        }
        let cols = |v: &[f32], rows: usize, i: usize| -> Vec<f32> {
            (0..n_seqs as usize)
                .flat_map(|s| v[(s * n_tok as usize + i) * rows..][..rows].to_vec())
                .collect()
        };
        for (i, (xi, di)) in walk_in.iter().enumerate() {
            xi.set_f32(&cols(&xv, c as usize, i));
            if let Some(di) = di {
                di.set_f32(&cols(&dv, (k * heads) as usize, i));
            }
        }
        let hist = vals(((k - 1) * width * n_seqs) as usize, 4);
        for s in st {
            s.set_f32(&hist);
        }
        gpu.compute(&graph).unwrap();
        let once = once.to_f32();
        let walk: Vec<Vec<f32>> = walk.into_iter().map(|o| o.to_f32()).collect();
        let case = format!("{n_tok} tokens, {n_seqs} sequences, dynamic {dynamic}");
        for s in 0..n_seqs as usize {
            for i in 0..n_tok as usize {
                let t = s * n_tok as usize + i;
                let (a, b) = (
                    &once[t * c as usize..][..c as usize],
                    &walk[i][s * c as usize..][..c as usize],
                );
                assert!(
                    a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()),
                    "{case}: token {i} of sequence {s}"
                );
            }
        }
        assert_eq!(st[0].bytes(), st[1].bytes(), "{case}: the history");
        assert_ne!(st[0].to_f32(), hist, "{case}: the history moved");
    }
}

/// #1221: the GPU expert matmul's decode (a block per pair) and prompt (a block per expert, in
/// tiles) kernels both give the bits of the sums it always had: warp_sum over the 32 lanes of
/// lane l's fused multiply-add chain over the groups of `w[32 gi + l] · x[32 gi + l]`. Q8_0,
/// MXFP8 and F32 stacks, shared and per-pair inputs, out-of-range ids, a K tail past whole
/// passes and an expert taking more pairs than its list holds at once.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_expert_matmul_keeps_its_lane_sums() {
    use crate::ggml::ComputeAllocator;
    let mut gpu = gpu_backend();
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let (k, m, n_exp, n_used) = (320usize, 40usize, 6usize, 3usize);
    // The weights as the kernels read them, and their bytes.
    let q8 = |next: &mut dyn FnMut() -> u64| -> (Vec<f32>, Vec<u8>) {
        let (mut v, mut b) = (Vec::new(), Vec::new());
        for _ in 0..k * m * n_exp / 32 {
            let d = f32_to_f16(((next() >> 40) as f32 / (1u64 << 24) as f32 + 0.5) / 64.0);
            b.extend(d.to_le_bytes());
            for _ in 0..32 {
                let q = (next() >> 56) as u8 as i8;
                b.push(q as u8);
                v.push(f16_to_f32(d) * q as f32);
            }
        }
        (v, b)
    };
    let mx = |next: &mut dyn FnMut() -> u64| -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let sc: Vec<f32> = (0..k * m * n_exp / 32)
            .map(|_| (1 + (next() >> 60)) as f32 / 32.0)
            .collect();
        let mut b = Vec::new();
        let v = (0..k * m * n_exp)
            .map(|i| {
                let mut q = (next() >> 56) as u8;
                if q & 0x7f == 0x7f {
                    q ^= 1;
                }
                b.push(q);
                crate::model::fp8::e4m3_to_f32(q) * sc[i / 32]
            })
            .collect();
        (v, b, sc)
    };
    let f = |next: &mut dyn FnMut() -> u64, n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            .collect()
    };
    // Lane sums in the kernels' order.
    let lane_sum = |w: &[f32], x: &[f32]| -> f32 {
        let mut lanes = [0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            for gi in 0..k / 32 {
                *acc = w[gi * 32 + l].mul_add(x[gi * 32 + l], *acc);
            }
        }
        for o in [16, 8, 4, 2, 1] {
            let prev = lanes;
            for (l, v) in lanes.iter_mut().enumerate() {
                *v = prev[l] + prev[l ^ o];
            }
        }
        lanes[0]
    };
    for ty in [types::Q8_0, types::I8, types::F32] {
        let (wv, wb, sc) = match ty {
            types::Q8_0 => {
                let (v, b) = q8(&mut next);
                (v, b, None)
            }
            types::I8 => {
                let (v, b, s) = mx(&mut next);
                (v, b, Some(s))
            }
            _ => {
                let v = f(&mut next, k * m * n_exp);
                let b = v.iter().flat_map(|v| v.to_le_bytes()).collect();
                (v, b, None)
            }
        };
        // Decode steps, a short prompt, and a prompt routing most pairs to expert 1.
        let cases = [
            (1usize, true),
            (2, false),
            (9, true),
            (400, false),
            (400, true),
        ];
        for (n_tok, shared) in cases {
            let ids: Vec<i32> = (0..n_tok * n_used)
                .map(|i| match next() % 16 {
                    0 => -1,
                    1 => n_exp as i32,
                    2..=4 => (next() % n_exp as u64) as i32,
                    _ if n_tok > 100 => 1,
                    _ => (i % n_exp) as i32,
                })
                .collect();
            let nx = if shared { 1 } else { n_used };
            let xv = f(&mut next, k * nx * n_tok);
            let ctx = Context::new(16, 16);
            let w = ctx
                .new_tensor(ty, &[k as i64, m as i64, n_exp as i64])
                .as_input();
            let scale = sc.as_ref().map(|_| {
                ctx.new_tensor(types::F32, &[(k / 32) as i64, m as i64, n_exp as i64])
                    .as_input()
            });
            let x = ctx
                .new_tensor(types::F32, &[k as i64, nx as i64, n_tok as i64])
                .as_input();
            let id = ctx
                .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                .as_input();
            let y = ops::mm_id(&ctx, w, scale, x, id).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(y);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            w.set_bytes(&wb);
            if let (Some(t), Some(v)) = (scale, &sc) {
                t.set_f32(v);
            }
            x.set_f32(&xv);
            let id_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
            id.set_bytes(&id_bytes);
            gpu.compute(&graph).unwrap();
            let y = y.to_f32();
            let tn = crate::ggml::type_name(ty);
            let case = format!("{tn} stack, {n_tok} tokens, shared input {shared}");
            for (pr, &e) in ids.iter().enumerate() {
                let (t, j) = (pr / n_used, pr % n_used);
                let xr = &xv[(t * nx + if shared { 0 } else { j }) * k..][..k];
                for r in 0..m {
                    let want = if e < 0 || e as usize >= n_exp {
                        0.0
                    } else {
                        lane_sum(&wv[(e as usize * m + r) * k..][..k], xr)
                    };
                    assert_eq!(
                        y[pr * m + r].to_bits(),
                        want.to_bits(),
                        "{case}: pair {pr} (expert {e}), row {r}"
                    );
                }
            }
        }
    }
}

/// `t`'s per-expert entries of each token in the order of the experts' ids `ids` (`[k, n]`):
/// `t` `[k, n]` (the ids, the weights) or `[w, k, n]` (the experts' rows).
#[cfg(feature = "cuda")]
fn by_expert_id(t: &TracedTensor, ids: &TracedTensor) -> Vec<f32> {
    let (k, n) = (ids.ne[0] as usize, ids.ne[1] as usize);
    let w = if t.data.len() == k * n {
        1
    } else {
        t.ne[0] as usize
    };
    assert_eq!(t.data.len(), w * k * n, "{}", t.name);
    let mut out = Vec::with_capacity(t.data.len());
    for tok in 0..n {
        let id = |j: usize| ids.data[tok * k + j];
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| id(a).total_cmp(&id(b)));
        for j in order {
            let at = (tok * k + j) * w;
            out.extend_from_slice(&t.data[at..at + w]);
        }
    }
    out
}

/// The router's selection scores, `sigmoid(logit) + bias`, of token `tok` (`logits` `[n_expert,
/// n]`; no bias: 0).
#[cfg(feature = "cuda")]
fn route_scores(logits: &TracedTensor, bias: Option<&[f32]>, tok: usize) -> Vec<f32> {
    let ne = logits.ne[0] as usize;
    (0..ne)
        .map(|e| {
            let l = logits.data[tok * ne + e];
            1.0 / (1.0 + (-l).exp()) + bias.map_or(0.0, |b| b[e])
        })
        .collect()
}

/// Where two routes `a` and `b` (`[k, n]` ids) of slot `il` first pick different experts: the
/// token, both choices, and each one's selection scores on both sides (from the traces'
/// `ffn_moe_logits-{il}` and `bias`), with the CPU's gap between its last pick and the best
/// expert it left out.
#[cfg(feature = "cuda")]
fn route_report(
    cpu: &[TracedTensor],
    gpu: &[TracedTensor],
    il: usize,
    a: &TracedTensor,
    b: &TracedTensor,
    bias: Option<&[f32]>,
) -> String {
    let k = a.ne[0] as usize;
    let set = |t: &TracedTensor, tok: usize| {
        let mut v: Vec<i32> = t.data[tok * k..(tok + 1) * k]
            .iter()
            .map(|&x| x as i32)
            .collect();
        v.sort();
        v
    };
    let Some(tok) = (0..a.ne[1] as usize).find(|&t| set(a, t) != set(b, t)) else {
        return format!("slot {il}: the same experts in another order");
    };
    let name = format!("ffn_moe_logits-{il}");
    let find = |ts: &[TracedTensor]| ts.iter().find(|t| t.name == name).cloned();
    let (Some(cl), Some(gl)) = (find(cpu), find(gpu)) else {
        return format!("slot {il} token {tok}: no {name}");
    };
    let (cs, gs) = (route_scores(&cl, bias, tok), route_scores(&gl, bias, tok));
    let mut ranked: Vec<usize> = (0..cs.len()).collect();
    ranked.sort_by(|&x, &y| cs[y].total_cmp(&cs[x]));
    let gap = cs[ranked[k - 1]] - ranked.get(k).map_or(f32::MIN, |&e| cs[e]);
    let (sa, sb) = (set(a, tok), set(b, tok));
    let mut experts: Vec<i32> = sa.iter().chain(&sb).copied().collect();
    experts.sort();
    experts.dedup();
    let scores: Vec<String> = experts
        .iter()
        .map(|&e| format!("{e}: cpu {:.6} gpu {:.6}", cs[e as usize], gs[e as usize]))
        .collect();
    format!(
        "slot {il} token {tok}: cpu picks {sa:?}, gpu {sb:?}; scores {}; the CPU's gap after its \
         last pick {gap:.3e}{}",
        scores.join(", "),
        if bias.is_some() { "" } else { " (no bias)" }
    )
}

/// The GPU's traced intermediates against the CPU's, in the CPU's graph order, matched by name:
/// a line each (see them with `--nocapture`). The routed experts' taps (`ffn_moe_topk`,
/// `_weights`, `_gate_up`, `_act`, `_down`) are compared in the order of the experts' ids, so
/// the same experts in another order match; where the routes differ, [`route_report`] says
/// how (`bias`: the router's bias of a slot, when known). Returns the first tap whose worst
/// difference is over `tol` times the CPU tensor's largest value (NaN counts as over), with
/// that element on both, the logits' worst difference, the tap furthest off (its worst
/// difference over its scale, and its name), and the structural mismatches (a CPU tap missing
/// on the GPU or of another shape), apart from the numbers.
#[cfg(feature = "cuda")]
fn compare_traces(
    cpu: &crate::model::core::Trace,
    gpu: &crate::model::core::Trace,
    tol: f32,
    bias: &dyn Fn(usize) -> Option<Vec<f32>>,
) -> (Option<String>, f32, (f32, String), Vec<String>) {
    let ((cl, ct), (gl, gt)) = (cpu, gpu);
    let mut by_name = HashMap::new();
    for g in gt {
        by_name.entry(g.name.as_str()).or_insert(g);
    }
    let cpu_named = |name: &str| ct.iter().find(|t| t.name == name);
    eprintln!("{} CPU taps, {} GPU taps", ct.len(), gt.len());
    let mut first = None;
    let mut worst = (0f32, String::new());
    let mut routes_reported = 0;
    let mut structural = Vec::new();
    for (i, c) in ct.iter().enumerate() {
        let Some(g) = by_name.get(c.name.as_str()) else {
            eprintln!("{i:5} {:<22} {:?} not on the GPU", c.name, c.ne);
            structural.push(format!("{}: not on the GPU", c.name));
            continue;
        };
        if g.ne != c.ne {
            eprintln!(
                "{i:5} {:<22} cpu {:?} gpu {:?}  <- shape",
                c.name, c.ne, g.ne
            );
            first.get_or_insert_with(|| format!("{}: cpu {:?} gpu {:?}", c.name, c.ne, g.ne));
            structural.push(format!("{}: cpu {:?} gpu {:?}", c.name, c.ne, g.ne));
            continue;
        }
        let (base, il) = c.name.rsplit_once('-').unwrap_or((c.name.as_str(), ""));
        let moe = [
            "ffn_moe_topk",
            "ffn_moe_weights",
            "ffn_moe_gate_up",
            "ffn_moe_act",
            "ffn_moe_down",
        ];
        let ids = format!("ffn_moe_topk-{il}");
        let (cd, gd) = match (
            moe.contains(&base),
            cpu_named(&ids),
            by_name.get(ids.as_str()),
        ) {
            (true, Some(ci), Some(gi)) => (by_expert_id(c, ci), by_expert_id(g, gi)),
            _ => (c.data.clone(), g.data.clone()),
        };
        let scale = scale_of(&cd).max(1e-6);
        let (at, d) = cd
            .iter()
            .zip(&gd)
            .map(|(x, y)| (x - y).abs())
            .map(|d| if d.is_nan() { f32::INFINITY } else { d })
            .enumerate()
            .fold((0, 0f32), |b, (j, d)| if d > b.1 { (j, d) } else { b });
        let bad = d > tol * scale;
        if d / scale > worst.0 {
            worst = (d / scale, c.name.clone());
        }
        let mark = if bad { "  <-" } else { "" };
        eprintln!(
            "{i:5} {:<22} {:?} diff {d:.3e} scale {scale:.3e}{mark}",
            c.name, c.ne
        );
        let route = (bad && base == "ffn_moe_topk").then(|| {
            let il: usize = il.parse().unwrap_or(0);
            route_report(ct, gt, il, c, g, bias(il).as_deref())
        });
        if let Some(r) = &route
            && routes_reported < 4
        {
            eprintln!("      route: {r}");
            routes_reported += 1;
        }
        if bad && first.is_none() {
            let [n0, n1, n2, _] = c.ne.map(|n| n.max(1) as usize);
            let idx = [
                at % n0,
                at / n0 % n1,
                at / (n0 * n1) % n2,
                at / (n0 * n1 * n2),
            ];
            let order = if cd == c.data { "" } else { " (by expert id)" };
            first = Some(format!(
                "{} {:?} at {idx:?}{order}: cpu {} gpu {} (diff {d}, scale {scale}){}",
                c.name,
                c.ne,
                cd[at],
                gd[at],
                route.map_or(String::new(), |r| format!("; {r}"))
            ));
        }
    }
    let d = max_diff(cl, gl);
    eprintln!("logits: diff {d:.3e} scale {:.3e}", scale_of(cl));
    if let Some(f) = &first {
        eprintln!("first mismatch: {f}");
    }
    eprintln!("furthest off: {} at {:.3e} of its scale", worst.1, worst.0);
    if !structural.is_empty() {
        eprintln!("{} structural mismatches", structural.len());
    }
    (first, d, worst, structural)
}

/// The tiny model's router bias of slot `il`.
#[cfg(feature = "cuda")]
fn tiny_bias(il: usize) -> Option<Vec<f32>> {
    let ip = if il < tiny::N_PHYS {
        il
    } else {
        tiny::CORE.start + (il - tiny::N_PHYS) % tiny::CORE.len()
    };
    let name = format!("blk.{ip}.exp_probs_b.bias");
    tiny::tensors(7)
        .into_iter()
        .find(|t| t.0 == name)
        .map(|t| t.2)
}

/// [`cuda_tiny_matches_the_cpu`] node by node: every traced intermediate of one prompt on the
/// GPU against the CPU on the CPU's route (see [`compare_traces`]), the first over 5e-3 of its
/// scale printed. The F16 rounding drifts past that over the slots, so it fails on the logits
/// (5e-3) and on a node over 5e-2, which a wrong kernel reaches. The GPU router's own choices (`ffn_moe_router`) must be the CPU's up to near
/// ties: an expert the CPU left out only with a selection score within 5e-3 of the logits'
/// scale of its last pick's. First, for the record, the whole comparison on the GPU's own
/// route (where the routes part, [`route_report`] shows the scores). It also prints how far
/// the GPU's untraced logits are from its traced ones: the traces keep their tensors'
/// memory, so a gap there points at memory reused too early.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_traces_match_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let p = prompt();
    let mut b = backend();
    let cpu = Model::load(&g, |_| b.buffer_type())
        .unwrap()
        .prompt_trace(&mut b, &p)
        .unwrap();
    let route = route_of(&cpu);
    let mut gpu = gpu_backend();
    let m = Model::load(&g, |_| gpu.buffer_type()).unwrap();
    eprintln!("== the GPU on its own route");
    let own = m.prompt_trace(&mut gpu, &p).unwrap();
    compare_traces(&cpu, &own, 5e-3, &tiny_bias);
    eprintln!("== the GPU on the CPU's route");
    let plain = with_route(&route, || m.prompt_logits_all(&mut gpu, &p).unwrap());
    let traced = with_route(&route, || m.prompt_trace(&mut gpu, &p).unwrap());
    eprintln!(
        "GPU logits untraced vs traced: {:.3e}; untraced vs the CPU: {:.3e}",
        max_diff(&plain, &traced.0),
        max_diff(&plain, &cpu.0)
    );
    let (_, d, worst, structural) = compare_traces(&cpu, &traced, 5e-3, &tiny_bias);
    assert!(
        structural.is_empty(),
        "taps missing or reshaped: {structural:?}"
    );
    let tol = 5e-3 * scale_of(&cpu.0);
    assert!(d <= tol, "logits: {d} > {tol}");
    // The GPU router's own choices against the CPU's.
    let k = tiny::N_USED as usize;
    let mut ties = 0;
    for il in 0..tiny::n_slots() {
        let named = |ts: &[TracedTensor], name: String| {
            ts.iter()
                .find(|t| t.name == name)
                .cloned()
                .unwrap_or_else(|| panic!("{name}"))
        };
        let ids = named(&cpu.1[..], format!("ffn_moe_topk-{il}"));
        let own = named(&traced.1[..], format!("ffn_moe_router-{il}"));
        let logits = named(&cpu.1[..], format!("ffn_moe_logits-{il}"));
        let eps = 5e-3 * scale_of(&logits.data);
        let bias = tiny_bias(il).unwrap();
        for tok in 0..p.len() {
            let picks = |t: &TracedTensor| -> Vec<usize> {
                t.data[tok * k..(tok + 1) * k]
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            };
            let (cp, gp) = (picks(&ids), picks(&own));
            // k distinct experts in range (any order) before the near-tie allowance.
            let raw = &own.data[tok * k..(tok + 1) * k];
            let mut distinct = gp.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() == k
                    && raw
                        .iter()
                        .all(|&x| x >= 0.0 && x < tiny::N_EXPERT as f32 && x.fract() == 0.0),
                "slot {il} token {tok}: the GPU router picks {raw:?}, not {k} distinct experts \
                 of {}",
                tiny::N_EXPERT
            );
            let s = route_scores(&logits, Some(bias.as_slice()), tok);
            let last = cp.iter().map(|&e| s[e]).fold(f32::MAX, f32::min);
            for &e in gp.iter().filter(|e| !cp.contains(e)) {
                assert!(
                    s[e] >= last - eps,
                    "slot {il} token {tok}: the GPU router picks {gp:?}, the CPU {cp:?}; \
                     expert {e} scores {} on the CPU, its last pick {last} (not a near tie, \
                     {eps})",
                    s[e]
                );
                ties += 1;
            }
        }
    }
    eprintln!("the GPU router: the CPU's choices but {ties} near ties");
    // The F16 rounding drifts through the slots; a wrong kernel is off by far more.
    assert!(
        worst.0 <= 5e-2,
        "{} is off by {} of its scale",
        worst.1,
        worst.0
    );
}

/// The real model (`LEGION_TEST_INKLING_GGUF`) on the GPU against the CPU over a short prompt:
/// the same greedy token at every position. It prints every traced intermediate's difference
/// and the first over 5e-2 of its scale (see [`compare_traces`]; the GPU products round their
/// inputs where the CPU's need not, so that one is a lead, not a failure). The CPU's copy of
/// the weights is dropped before the GPU's loads.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU and the Inkling-2-Mini GGUF"]
fn cuda_real_gguf_matches_the_cpu() {
    let path = std::env::var_os("LEGION_TEST_INKLING_GGUF").expect("set LEGION_TEST_INKLING_GGUF");
    let g = GgufFile::open(&path).unwrap();
    let p: Vec<u32> = (0..8).map(|i| 1000 + i * 7919).collect();
    let mut b = Backend::cpu().unwrap();
    b.set_threads(std::thread::available_parallelism().map_or(4, |n| n.get()));
    let cm = Model::load(&g, |_| b.buffer_type()).unwrap();
    let cpu = cm.prompt_trace(&mut b, &p).unwrap();
    drop(cm);
    let mut gpu = gpu_backend();
    let m = Model::load(&g, |_| gpu.buffer_type()).unwrap();
    let plain = m.prompt_logits_all(&mut gpu, &p).unwrap();
    let traced = m.prompt_trace(&mut gpu, &p).unwrap();
    let nv = cpu.0.len() / p.len();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
            .0
    };
    let tokens = |all: &[f32]| -> Vec<usize> { all.chunks(nv).map(argmax).collect() };
    let (ct, gt) = (tokens(&cpu.0), tokens(&plain));
    eprintln!(
        "greedy tokens: cpu {ct:?} gpu {gt:?}; logits diff {:.3e} scale {:.3e}; \
         GPU untraced vs traced {:.3e}",
        max_diff(&cpu.0, &plain),
        scale_of(&cpu.0),
        max_diff(&plain, &traced.0)
    );
    compare_traces(&cpu, &traced, 5e-2, &|_| None);
    assert_eq!(ct, gt, "greedy tokens");
}

/// The real GGUFs (`LEGION_TEST_INKLING_GGUF`, llama.cpp's `inkling2_loop`;
/// `LEGION_TEST_INKLING_OLD_GGUF`, the earlier `inkling2`): Inkling-2-Mini's shape, the same
/// hyperparameters from both, every tensor the graph reads present. Both must be set.
#[test]
#[ignore = "needs the Inkling-2-Mini GGUFs"]
fn real_ggufs_parse_to_inkling2_mini() {
    let mut seen = Vec::new();
    for var in ["LEGION_TEST_INKLING_GGUF", "LEGION_TEST_INKLING_OLD_GGUF"] {
        let path = std::env::var_os(var).unwrap_or_else(|| panic!("set {var}"));
        let g = GgufFile::open(&path).unwrap();
        let hp = Hparams::from_gguf(&g).unwrap();
        let p = params(&hp).unwrap().clone();
        assert_eq!(
            (hp.n_layer, p.n_phys, p.core.clone(), p.passes),
            (48, 24, 6..18, 3)
        );
        assert_eq!(
            (hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim),
            (2304, 24, 8, 128)
        );
        assert_eq!((hp.n_expert, hp.n_expert_used, p.n_vocab), (128, 8, 200058));
        let globals: Vec<usize> = (0..24).filter(|&i| !p.swa[i]).collect();
        assert_eq!(globals, [5, 11, 17, 23]);
        let names = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_gate.weight",
            "attn_rel.weight",
            "attn_rel_bias.weight",
            "attn_sink_proj.weight",
            "ffn_gate_up_exps.weight",
            "ffn_down_exps.weight",
        ];
        for ip in 0..24 {
            for n in names {
                assert!(
                    g.tensor(&format!("blk.{ip}.{n}")).is_some(),
                    "{var}: blk.{ip}.{n}"
                );
            }
            // `ink_layer` requires the attention mix's query after block 0.
            if ip > 0 {
                let n = format!("blk.{ip}.attn_res_query.weight");
                assert!(g.tensor(&n).is_some(), "{var}: {n}");
            }
            if p.core.contains(&ip) {
                for c in 0..4 {
                    for part in ["down", "gain", "forget"] {
                        let n = p.ctrl_name(ip, c, part);
                        assert!(g.tensor(&n).is_some(), "{var}: {n}");
                    }
                }
            }
        }
        seen.push(Ink2Params {
            old_layout: false,
            ..p
        });
    }
    assert_eq!(seen[0], seen[1], "the two layouts' hyperparameters");
}

/// The NVFP4 checkpoint (`LEGION_TEST_INKLING_HF=<dir>`, opened by [`crate::hf`]) parses to the
/// mainline GGUF's hyperparameters (`LEGION_TEST_INKLING_GGUF`, when given).
#[test]
#[ignore = "needs the Inkling-2-Mini NVFP4 checkpoint"]
fn real_checkpoint_parses_like_the_gguf() {
    let dir = std::env::var_os("LEGION_TEST_INKLING_HF").expect("set LEGION_TEST_INKLING_HF");
    let g = GgufFile::open(&dir).unwrap();
    let hp = Hparams::from_gguf(&g).unwrap();
    let p = params(&hp).unwrap();
    assert_eq!(hp.n_layer, 48);
    for (ip, ty) in [
        (0, types::NVFP4),
        (19, types::NVFP4),
        (20, types::I8),
        (23, types::I8),
    ] {
        let t = g
            .tensor(&format!("blk.{ip}.ffn_gate_up_exps.weight"))
            .unwrap();
        assert_eq!(t.ggml_type, ty, "blk.{ip} experts");
    }
    if let Some(path) = std::env::var_os("LEGION_TEST_INKLING_GGUF") {
        let gg = Hparams::from_gguf(&GgufFile::open(path).unwrap()).unwrap();
        assert_eq!(params(&gg).unwrap(), p);
        assert_eq!(
            (gg.n_embd, gg.n_head, gg.n_head_kv, gg.head_dim, gg.n_vocab),
            (hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_vocab)
        );
    }
}

/// One small step of a controller both ways on `b`: the five nodes (`ink_ctrl_in`, the down,
/// gain and forget products through `glm_linear` as `glm_mm` makes them, `ink_ctrl_out`) and
/// the fused two (`ink_ctrl_down`, `ink_ctrl_gate`); the down projections and the outputs must
/// match bit for bit (#1221 r5). Rows of `e` values, controllers of rank `rank`, up to 8 tokens.
fn fused_controller_keeps_the_bits(b: &mut Backend) {
    use crate::ggml::ComputeAllocator;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut vals = move |n: usize, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * amp
            })
            .collect()
    };
    for (e, rank) in [(2304i64, 64i64), (256, 32), (128, 512)] {
        for n in [1i64, 2, 3, 4, 5, 8] {
            let ctx = Context::new(64, 64);
            let new = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input();
            let (cur, msg) = (new(&[e, n]), new(&[e, n]));
            let dw = new(&[2 * e, rank]);
            let (gw, fw) = (new(&[rank, e]), new(&[rank, e]));
            assert!(ops::ctrl_takes(dw, gw, fw, n), "e {e} rank {rank} n {n}");
            let g = ops::ctrl_in(&ctx, cur, msg, MIX_EPS);
            let d5 = glm_ops::linear_h(&ctx, dw, g, None, 1.0, None).as_output();
            let gain = glm_ops::linear_h(&ctx, gw, d5, None, 1.0, None);
            let forget = glm_ops::linear_h(&ctx, fw, d5, None, 1.0, None);
            let o5 = ops::ctrl_out(&ctx, cur, msg, gain, forget).as_output();
            let d2 = ops::ctrl_down(&ctx, cur, msg, dw, MIX_EPS).as_output();
            let o2 = ops::ctrl_gate(&ctx, cur, msg, d2, gw, fw).as_output();
            let mut graph = ctx.new_graph(64);
            for t in [d5, o5, d2, o2] {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(b.buffer_type()))
                .unwrap();
            cur.set_f32(&vals((e * n) as usize, 4.0));
            msg.set_f32(&vals((e * n) as usize, 1.0));
            dw.set_f32(&vals((2 * e * rank) as usize, 0.05));
            gw.set_f32(&vals((rank * e) as usize, 0.5));
            fw.set_f32(&vals((rank * e) as usize, 0.5));
            b.compute(&graph).unwrap();
            let case = format!("e {e}, rank {rank}, {n} tokens");
            let same = |x: &[f32], y: &[f32]| {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            };
            assert!(
                same(&d5.to_f32(), &d2.to_f32()),
                "{case}: the down projection"
            );
            let out = o2.to_f32();
            assert!(same(&o5.to_f32(), &out), "{case}: the output");
            assert!(out.iter().any(|&v| v != 0.0), "{case}: an output");
        }
    }
}

/// [`fused_controller_keeps_the_bits`] on the CPU references.
#[test]
fn fused_controller_keeps_the_cpu_bits() {
    fused_controller_keeps_the_bits(&mut backend());
}

/// [`fused_controller_keeps_the_bits`] on the GPU: the fused kernels against `glm_linear`'s F32
/// kernels (the split and shared-input ones, by token count).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_fused_controller_keeps_the_bits() {
    fused_controller_keeps_the_bits(&mut gpu_backend());
}
