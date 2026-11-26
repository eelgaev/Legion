use std::collections::HashMap;

use super::*;
use crate::cpu::gpu_exact::{f16_to_f32, f32_to_f16};
use crate::ggml::Backend;
use crate::gguf::Value;
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

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
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
        let long: Vec<u32> = (0..90).map(|i| (i * 37 + 11) % 250).collect();
        let short: Vec<u32> = prompt()[..30].to_vec();
        let mut last = [0u32; 2];
        for (s, p) in [long, short].iter().enumerate() {
            for c in p.chunks(STEP) {
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

/// The whole tiny graph on the GPU (the CUDA kernels of [`ops`] and GLM's dense ones) gives
/// the CPU's logits up to rounding, in one prompt and token by token.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_matches_the_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_gguf(&dir, Layout::Mainline);
    let p = prompt();
    let mut b = backend();
    let cpu = Model::load(&g, |_| b.buffer_type())
        .unwrap()
        .prompt_logits_all(&mut b, &p)
        .unwrap();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let m = Model::load(&g, |_| gpu.buffer_type()).unwrap();
    let once = m.prompt_logits_all(&mut gpu, &p).unwrap();
    let tol = 5e-3 * scale_of(&cpu);
    let d = max_diff(&once, &cpu);
    assert!(d <= tol, "one prompt on the GPU: {d} > {tol}");
    let mut st = m.new_state(gpu.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let mut steps = Vec::new();
    for &t in &p {
        steps.extend(
            m.forward(&mut gpu, &mut alloc, &mut st, &[t], true)
                .unwrap(),
        );
    }
    let d = max_diff(&steps, &cpu);
    assert!(d <= tol, "token by token on the GPU: {d} > {tol}");
}

/// The real GGUFs (`LEGION_TEST_INKLING_GGUF`, llama.cpp's `inkling2_loop`;
/// `LEGION_TEST_INKLING_OLD_GGUF`, the earlier `inkling2`): Inkling-2-Mini's shape, the same
/// hyperparameters from both, every tensor the graph reads present.
#[test]
#[ignore = "needs the Inkling-2-Mini GGUFs"]
fn real_ggufs_parse_to_inkling2_mini() {
    let mut seen = Vec::new();
    for var in ["LEGION_TEST_INKLING_GGUF", "LEGION_TEST_INKLING_OLD_GGUF"] {
        let Some(path) = std::env::var_os(var) else {
            continue;
        };
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
    if let [a, b] = &seen[..] {
        assert_eq!(a, b);
    }
}

/// The NVFP4 checkpoint (`LEGION_TEST_INKLING_HF=<dir>`, opened by [`crate::hf`]) parses to the
/// mainline GGUF's hyperparameters (`LEGION_TEST_INKLING_GGUF`, when given).
#[test]
#[ignore = "needs the Inkling-2-Mini NVFP4 checkpoint"]
fn real_checkpoint_parses_like_the_gguf() {
    let Some(dir) = std::env::var_os("LEGION_TEST_INKLING_HF") else {
        return;
    };
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
