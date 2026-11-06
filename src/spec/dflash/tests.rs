//! The drafter against a plain-Rust implementation of the same network, over the tiny
//! checkpoint; the GGUF and Hugging Face loaders; rollback, the window ring and error cases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::tiny::{self, HfDtype, TinyDraft};
use super::*;
use crate::engine::sampling::Rng;
use crate::gguf::GgufFile;
use crate::model::qwen35::tiny as target_tiny;

pub(super) const N_VOCAB: usize = 259;

pub(super) struct Fixture {
    pub(super) _dir: tempfile::TempDir,
    pub(super) root: PathBuf,
    pub(super) target: PathBuf,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let target = root.join("target.gguf");
        target_tiny::write(&target, &target_tiny::TinyConfig::default()).unwrap();
        Self {
            _dir: dir,
            root,
            target,
        }
    }

    pub(super) fn info(&self, draft: &Path) -> TargetInfo {
        TargetInfo {
            n_layer: 4,
            n_embd: 64,
            n_vocab: N_VOCAB,
            model: self.target.clone(),
            draft: Some(draft.to_path_buf()),
            gpu: false,
            threads: 2,
            ctx_size: 4096,
            max_seqs: 2,
            use_ref: false,
        }
    }

    pub(super) fn gguf(&self, name: &str, c: &TinyDraft) -> PathBuf {
        let p = self.root.join(name);
        tiny::write_gguf(&p, c).unwrap();
        p
    }

    pub(super) fn load(&self, draft: &Path, opts: DflashOptions) -> DflashDrafter {
        DflashDrafter::load(&self.info(draft), opts).unwrap()
    }
}

/// The distribution (softmax at temperature 1) is what the tests compare.
pub(super) fn sampling(window: usize) -> DflashOptions {
    DflashOptions {
        window,
        temperature: 1.0,
        seed: 5,
        ..DflashOptions::default()
    }
}

/// The hidden state of target layer `layer` at position `pos`: a pure function, so a test can
/// give the same rows again after a rollback.
pub(super) fn hidden(pos: usize, layer: usize, salt: u64) -> Vec<f32> {
    let mut r = Rng::new(1000 * pos as u64 + layer as u64 + 77 * salt);
    (0..64)
        .map(|_| (r.next_f64() as f32 * 2.0 - 1.0) * 2.0)
        .collect()
}

pub(super) fn tokens_to(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 7 + 3) % 250).collect()
}

/// One `step` over a sequence: takes in positions `start..start + ran` of `tokens` (hidden states
/// salted by `salt`) and drafts up to `max`.
pub(super) fn step(
    d: &mut dyn ModelDrafter,
    seq: u64,
    tokens: &[u32],
    start: usize,
    ran: usize,
    max: usize,
    salt: u64,
) -> Draft {
    let layers = d.hidden_layers();
    let rows: Vec<Vec<f32>> = layers
        .iter()
        .enumerate()
        .map(|(li, _)| {
            (start..start + ran)
                .flat_map(|p| hidden(p, li, salt))
                .collect()
        })
        .collect();
    let mut out = Draft::default();
    d.step(&mut [DraftRequest {
        seq,
        tokens,
        start,
        ran,
        hidden: rows.iter().map(|v| v.as_slice()).collect(),
        max,
        out: &mut out,
    }])
    .unwrap();
    out
}

pub(super) fn read_f32(path: &Path, name: &str) -> Vec<f32> {
    let g = GgufFile::open(path).unwrap();
    let t = g.tensor(name).unwrap();
    let bytes = std::fs::read(path).unwrap();
    bytes[t.offset as usize..(t.offset + t.size) as usize]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

/// The network in plain Rust: no caches, no graphs; the context rows are given each time.
pub(super) struct Reference {
    c: TinyDraft,
    pub(super) w: HashMap<String, Vec<f32>>,
    pub(super) embd: Vec<f32>,
    pub(super) head: Vec<f32>,
    /// Target ids of the head's rows, when the vocabulary is reduced.
    pub(super) d2t: Option<Vec<u32>>,
    pub(super) round: fn(f32) -> f32,
}

pub(super) fn matvec(w: &[f32], inn: usize, x: &[f32]) -> Vec<f32> {
    w.chunks_exact(inn)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

pub(super) fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(w).map(|(v, w)| v * s * w).collect()
}

/// NeoX RoPE of one head.
fn rope(x: &mut [f32], pos: usize, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f32 * base.powf(-2.0 * i as f32 / n_rot as f32);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

impl Reference {
    pub(super) fn new(fx: &Fixture, c: &TinyDraft, round: fn(f32) -> f32) -> Self {
        let w = tiny::effective_tensors(c)
            .into_iter()
            .map(|(n, _, v)| (n, v.into_iter().map(round).collect()))
            .collect();
        let embd = read_f32(&fx.target, "token_embd.weight");
        let head = match &c.draft_vocab {
            Some(_) => tiny::effective_tensors(c)
                .into_iter()
                .find(|t| t.0 == "output.weight")
                .unwrap()
                .2
                .into_iter()
                .map(round)
                .collect(),
            None => read_f32(&fx.target, "output.weight"),
        };
        Self {
            c: c.clone(),
            w,
            embd,
            head,
            d2t: c.draft_vocab.clone(),
            round,
        }
    }

    fn t(&self, il: usize, n: &str) -> &[f32] {
        &self.w[&format!("blk.{il}.{n}")]
    }

    fn fused(&self, feats: &[Vec<f32>]) -> Vec<f32> {
        let x: Vec<f32> = feats.iter().flatten().copied().collect();
        let g = matvec(&self.w["fc.weight"], x.len(), &x);
        rms(&g, &self.w["enc.output_norm.weight"])
    }

    /// K and V (per kv head) of the row `x` at `pos`, rounded like the F16 cache.
    fn kv(&self, il: usize, x: &[f32], pos: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let c = &self.c;
        let e = c.n_embd;
        let k = matvec(self.t(il, "attn_k.weight"), e, x);
        let v = if c.shared_kv {
            rms(&k, &vec![1.0; k.len()])
                .chunks(c.head_dim)
                .flat_map(|h| h.to_vec())
                .collect()
        } else {
            matvec(self.t(il, "attn_v.weight"), e, x)
        };
        let v = if c.shared_kv {
            // Per-head RMS without weights.
            k.chunks(c.head_dim)
                .flat_map(|h| rms(h, &vec![1.0; c.head_dim]))
                .collect()
        } else {
            v
        };
        let mut ks = Vec::new();
        for h in k.chunks(c.head_dim) {
            let mut h = rms(h, self.t(il, "attn_k_norm.weight"));
            rope(&mut h, pos, c.head_dim, c.rope_base);
            ks.push(h.into_iter().map(f16).collect());
        }
        let vs = v
            .chunks(c.head_dim)
            .map(|h| h.iter().map(|&x| f16(x)).collect())
            .collect();
        (ks, vs)
    }

    /// Logits of block rows `first..n`: `ctx` is `(position, [layer features])` of the visible
    /// context, `tokens` the block (last token, then masks), anchored at `anchor`.
    pub(super) fn logits(
        &self,
        ctx: &[(usize, Vec<Vec<f32>>)],
        tokens: &[u32],
        anchor: usize,
        first: usize,
    ) -> Vec<Vec<f32>> {
        self.hidden_rows(ctx, tokens, anchor, first)
            .iter()
            .map(|x| matvec(&self.head, self.c.n_embd, x))
            .collect()
    }

    /// The final-normed hidden states of block rows `first..n`.
    pub(super) fn hidden_rows(
        &self,
        ctx: &[(usize, Vec<Vec<f32>>)],
        tokens: &[u32],
        anchor: usize,
        first: usize,
    ) -> Vec<Vec<f32>> {
        let c = &self.c;
        let (e, d, nh, nkv) = (c.n_embd, c.head_dim, c.n_head, c.n_head_kv);
        let n = tokens.len();
        let g: Vec<Vec<f32>> = ctx.iter().map(|(_, f)| self.fused(f)).collect();
        let mut h: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                self.embd[t as usize * e..(t as usize + 1) * e]
                    .iter()
                    .map(|x| x * c.embedding_scale)
                    .collect()
            })
            .collect();
        for il in 0..c.n_layer {
            let ctx_kv: Vec<_> = ctx
                .iter()
                .zip(&g)
                .map(|((p, _), g)| (*p, self.kv(il, g, *p)))
                .collect();
            let x: Vec<Vec<f32>> = h
                .iter()
                .map(|r| rms(r, self.t(il, "attn_norm.weight")))
                .collect();
            let (x, attn_dynamic) = self.conv_in(il, "attn", x);
            let blk_kv: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(j, x)| self.kv(il, x, anchor + j))
                .collect();
            let mut attn_out = Vec::new();
            for (j, xj) in x.iter().enumerate() {
                let q = matvec(self.t(il, "attn_q.weight"), e, xj);
                let mut o = vec![0f32; nh * d];
                for hd in 0..nh {
                    let mut qh = rms(&q[hd * d..(hd + 1) * d], self.t(il, "attn_q_norm.weight"));
                    rope(&mut qh, anchor + j, c.head_dim, c.rope_base);
                    let kvh = hd / (nh / nkv);
                    let mut keys: Vec<(&[f32], &[f32])> = Vec::new();
                    for (p, (ks, vs)) in &ctx_kv {
                        let ok = c.sliding_window.is_none_or(|w| anchor + j < p + w);
                        if ok {
                            keys.push((&ks[kvh], &vs[kvh]));
                        }
                    }
                    for (ks, vs) in &blk_kv {
                        keys.push((&ks[kvh], &vs[kvh]));
                    }
                    let sink = self
                        .w
                        .get(&format!("blk.{il}.attn_sinks.weight"))
                        .map(|s| s[hd]);
                    let scores: Vec<f32> = keys
                        .iter()
                        .map(|(k, _)| {
                            k.iter().zip(&qh).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
                        })
                        .collect();
                    let m = scores
                        .iter()
                        .cloned()
                        .chain(sink)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    // A sink is a key with no value: it only adds to the denominator.
                    let z: f32 = ex.iter().sum::<f32>() + sink.map_or(0.0, |s| (s - m).exp());
                    for (w, (_, v)) in ex.iter().zip(&keys) {
                        for i in 0..d {
                            o[hd * d + i] += w / z * v[i];
                        }
                    }
                }
                attn_out.push(matvec(self.t(il, "attn_output.weight"), nh * d, &o));
            }
            let attn_out = self.conv_out(il, "attn", attn_out, attn_dynamic);
            let h1: Vec<Vec<f32>> = attn_out
                .iter()
                .zip(&h)
                .map(|(a, h)| a.iter().zip(h).map(|(a, b)| a + b).collect())
                .collect();
            let f: Vec<Vec<f32>> = h1
                .iter()
                .map(|r| rms(r, self.t(il, "ffn_norm.weight")))
                .collect();
            let (f, ffn_dynamic) = self.conv_in(il, "ffn", f);
            let down: Vec<Vec<f32>> = f
                .iter()
                .map(|f| {
                    let gate = matvec(self.t(il, "ffn_gate.weight"), e, f);
                    let up = matvec(self.t(il, "ffn_up.weight"), e, f);
                    let act: Vec<f32> = gate
                        .iter()
                        .zip(&up)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect();
                    matvec(self.t(il, "ffn_down.weight"), c.n_ff, &act)
                })
                .collect();
            let down = self.conv_out(il, "ffn", down, ffn_dynamic);
            h = down
                .iter()
                .zip(&h1)
                .map(|(a, h)| a.iter().zip(h).map(|(a, b)| a + b).collect())
                .collect();
        }
        let _ = self.round;
        (first..n)
            .map(|j| rms(&h[j], &self.w["output_norm.weight"]))
            .collect()
    }

    /// DFlash 2: the block rows through the input side of layer `il`'s convolution around
    /// `part`, and the dynamic coefficients `[position][2 · kernel · groups]` of both sides.
    /// Without DFlash 2: the rows as they are.
    fn conv_in(
        &self,
        il: usize,
        part: &str,
        rows: Vec<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let Some(v2) = &self.c.v2 else {
            return (rows, Vec::new());
        };
        let e = self.c.n_embd;
        let dynamic: Vec<Vec<f32>> = rows
            .iter()
            .map(|x| matvec(self.t(il, &format!("{part}_conv_proj.weight")), e, x))
            .collect();
        let base = self.t(il, &format!("{part}_conv_base"));
        (conv_ref(&rows, &dynamic, base, 0, v2), dynamic)
    }

    /// DFlash 2: the output side of the convolution [`Self::conv_in`] began.
    fn conv_out(
        &self,
        il: usize,
        part: &str,
        rows: Vec<Vec<f32>>,
        dynamic: Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        match &self.c.v2 {
            Some(v2) => conv_ref(
                &rows,
                &dynamic,
                self.t(il, &format!("{part}_conv_base")),
                1,
                v2,
            ),
            None => rows,
        }
    }

    /// Softmax of the logits over the target vocabulary.
    pub(super) fn probs(&self, logits: &[f32]) -> Vec<f32> {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = ex.iter().sum();
        let mut p = vec![0f32; N_VOCAB];
        for (i, e) in ex.iter().enumerate() {
            let t = self.d2t.as_ref().map_or(i, |m| m[i] as usize);
            p[t] = e / z;
        }
        p
    }
}

/// z-lab's `_grouped_dynamic_convolve`, one side: `out[t][c] = sum over taps of
/// (base[side][tap][c] + dynamic[t][side][tap][c / group]) * rows[t - tap][c]`, rows before the
/// block being zero. `base` is `[n_embd, kernel, 2]` and `dynamic[t]` `[groups, kernel, 2]`
/// (ggml order).
pub(super) fn conv_ref(
    rows: &[Vec<f32>],
    dynamic: &[Vec<f32>],
    base: &[f32],
    side: usize,
    v2: &Dflash2,
) -> Vec<Vec<f32>> {
    let (k, group) = (v2.conv_kernel, v2.conv_group);
    (0..rows.len())
        .map(|t| {
            let e = rows[t].len();
            let groups = e / group;
            (0..e)
                .map(|ch| {
                    (0..k.min(t + 1))
                        .map(|tap| {
                            let b = base[(side * k + tap) * e + ch];
                            let dy = dynamic[t][(side * k + tap) * groups + ch / group];
                            (b + dy) * rows[t - tap][ch]
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn f16(x: f32) -> f32 {
    half_round(x)
}

/// Rounds to the nearest F16 value and back.
fn half_round(x: f32) -> f32 {
    let bytes = crate::ggml::quantize(crate::ggml::types::F16, &[x], 1);
    crate::ggml::dequantize(crate::ggml::types::F16, &bytes, 1)[0]
}

pub(super) fn ctx_rows(
    positions: impl Iterator<Item = usize>,
    salt: u64,
) -> Vec<(usize, Vec<Vec<f32>>)> {
    positions
        .map(|p| (p, vec![hidden(p, 0, salt), hidden(p, 1, salt)]))
        .collect()
}

pub(super) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn check_against_reference(
    fx: &Fixture,
    c: &TinyDraft,
    path: &Path,
    window: usize,
    n_ctx: usize,
    visible: std::ops::Range<usize>,
    tol: f32,
) -> Draft {
    let mut d = fx.load(path, sampling(window));
    let r = Reference::new(fx, c, |x| x);
    let tokens = tokens_to(n_ctx + 1);
    // Prefill in two chunks, then draft: the anchor is the last token.
    let split = n_ctx / 3;
    let _ = step(&mut d, 1, &tokens[..split + 1], 0, split, 0, 0);
    let out = step(
        &mut d,
        1,
        &tokens,
        split,
        n_ctx - split,
        c.block_size - 1,
        0,
    );
    assert_eq!(out.tokens.len(), c.block_size - 1);
    assert_eq!(out.probs.len(), c.block_size - 1);
    let mut block = vec![c.mask_token; c.block_size];
    block[0] = tokens[n_ctx];
    let want = r.logits(&ctx_rows(visible, 0), &block, n_ctx, 1);
    for (i, row) in want.iter().enumerate() {
        let p = r.probs(row);
        let diff = max_diff(&p, &out.probs[i]);
        assert!(diff < tol, "position {i}: probabilities differ by {diff}");
        assert!((out.probs[i].iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(out.probs[i][out.tokens[i] as usize] > 0.0);
    }
    out
}

#[test]
fn matches_the_reference_network() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    check_against_reference(&fx, &c, &path, 64, 21, 0..21, 2e-3);
}

#[test]
fn shared_kv_embedding_scale_and_sliding_window() {
    let fx = Fixture::new();
    let c = TinyDraft {
        shared_kv: true,
        embedding_scale: 0.5,
        sliding_window: Some(6),
        ..TinyDraft::default()
    };
    let path = fx.gguf("d.gguf", &c);
    // The window of 6 hides the older context from every block position.
    let out = check_against_reference(&fx, &c, &path, 64, 21, 0..21, 2e-3);
    // And it matters: the same network without the window gives other distributions.
    let open = TinyDraft {
        sliding_window: None,
        ..c.clone()
    };
    let r = Reference::new(&fx, &open, |x| x);
    let mut block = vec![c.mask_token; c.block_size];
    block[0] = tokens_to(22)[21];
    let wide = r.logits(&ctx_rows(0..21, 0), &block, 21, 1);
    let diff: f32 = (0..wide.len())
        .map(|i| max_diff(&r.probs(&wide[i]), &out.probs[i]))
        .fold(0.0, f32::max);
    assert!(diff > 3e-3, "diff {diff}");
}

#[test]
fn the_window_ring_keeps_the_newest_positions() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    // A ring of 8 positions, 20 of context: only 12..20 are visible.
    check_against_reference(&fx, &c, &path, 8, 20, 12..20, 2e-3);
}

#[test]
fn ring_keep_from_is_the_window_before_the_last_token() {
    assert_eq!(ring_keep_from(0, 8), 0);
    assert_eq!(ring_keep_from(9, 8), 0);
    assert_eq!(ring_keep_from(10, 8), 1);
    assert_eq!(ring_keep_from(2515, 520), 1994);
}

/// The old take-in loop's first piece taken in, as `ingest_from` must give it (#1133).
fn ingest_from_by_loop(n_tokens: usize, start: usize, ran: usize, window: usize) -> usize {
    let keep = ring_keep_from(n_tokens, window);
    let mut at = ran.saturating_sub(window);
    while at < ran {
        let n = INGEST_CHUNK.min(ran - at);
        if start + at + n <= keep {
            at += n;
            continue;
        }
        return at;
    }
    ran
}

#[test]
fn ingest_from_is_the_first_piece_taken_in() {
    // A 32K prompt in two 16K steps with the 4,096-row window: the first step's rows are all
    // overwritten, the second's are read from its last window on.
    assert_eq!(ingest_from(32_746, 0, 16_384, 4096), 16_384);
    assert_eq!(ingest_from(32_747, 16_384, 16_362, 4096), 16_362 - 4096);
    // A prompt in one step.
    assert_eq!(ingest_from(8_228, 0, 8_227, 4096), 8_227 - 4096);
    // Shorter than the window: every row.
    assert_eq!(ingest_from(300, 0, 299, 4096), 0);
    for window in [8, 520, 4096] {
        for n_tokens in [0, 1, 100, 2515, 9000, 40_000] {
            for start in [0, 7, 512, 2000, 16_384] {
                for ran in [0, 1, 100, 511, 512, 513, 2048, 4097, 16_384] {
                    let got = ingest_from(n_tokens, start, ran, window);
                    assert_eq!(
                        got,
                        ingest_from_by_loop(n_tokens, start, ran, window),
                        "{n_tokens} {start} {ran} {window}"
                    );
                    // A smaller token count (the one known before the step) never starts later.
                    assert!(ingest_from(n_tokens.saturating_sub(1), start, ran, window) <= got);
                }
            }
        }
    }
}

#[test]
fn a_long_prompt_skips_rows_its_later_chunks_overwrite() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    // A ring wider than an ingest piece (512 rows), a prompt of three chunks: the first chunk
    // and the first piece of the second lie below the ring's final window (from 1995) once
    // the whole prompt is known, and are left out.
    let window = 520;
    let tokens = tokens_to(2516);
    let chunks = [(0, 700, 0), (700, 1300, 0), (2000, 515, c.block_size - 1)];
    let mut whole = fx.load(&path, sampling(window));
    let mut cut = fx.load(&path, sampling(window));
    let (mut a, mut b) = (Draft::default(), Draft::default());
    for (start, ran, max) in chunks {
        // As the engine calls it (the whole prompt as `tokens`), and as if each chunk were the
        // prompt's end (no later rows known, every row of the window taken in).
        a = step(&mut cut, 1, &tokens, start, ran, max, 0);
        let known = &tokens[..start + ran + 1];
        b = step(&mut whole, 1, known, start, ran, max, 0);
    }
    assert_eq!(a.tokens.len(), c.block_size - 1);
    assert_eq!(a, b);
}

#[test]
fn hugging_face_checkpoints_load_like_gguf() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let gguf = fx.gguf("d.gguf", &c);
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::F32).unwrap();
    let tokens = tokens_to(13);
    let mut a = fx.load(&gguf, sampling(64));
    let mut b = fx.load(&hf, sampling(64));
    let (da, db) = (
        step(&mut a, 1, &tokens, 0, 12, 7, 0),
        step(&mut b, 1, &tokens, 0, 12, 7, 0),
    );
    assert_eq!(da, db);
    // BF16 weights: the same network on rounded weights.
    let bf = fx.root.join("bf16");
    tiny::write_hf(&bf, &c, HfDtype::Bf16).unwrap();
    let mut d = fx.load(&bf, sampling(64));
    let out = step(&mut d, 1, &tokens, 0, 12, 7, 0);
    let r = Reference::new(&fx, &c, tiny::bf16_round);
    let mut block = vec![c.mask_token; 8];
    block[0] = tokens[12];
    let want = r.logits(&ctx_rows(0..12, 0), &block, 12, 1);
    for (i, row) in want.iter().enumerate() {
        assert!(
            max_diff(&r.probs(row), &out.probs[i]) < 5e-3,
            "position {i}"
        );
    }
}

#[test]
fn rollback_equals_never_having_seen_the_rejected_rows() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(15);
    let mut a = fx.load(&path, sampling(64));
    // Rows 0..12 (prefill), a verify step's rows 12..15 of which the target keeps one (12);
    // then the next step starts at 13 with other hidden states.
    step(&mut a, 1, &tokens[..13], 0, 12, 0, 0);
    let first = step(&mut a, 1, &tokens[..15], 12, 3, 4, 0);
    assert_eq!(first.tokens.len(), 4);
    let after = step(&mut a, 1, &tokens[..14], 12, 2, 5, 1);
    // A drafter that only ever saw the accepted rows.
    let mut b = fx.load(&path, sampling(64));
    step(&mut b, 1, &tokens[..13], 0, 12, 0, 0);
    let fresh = step(&mut b, 1, &tokens[..14], 12, 2, 5, 1);
    assert_eq!(after.probs, fresh.probs);
}

#[test]
fn a_block_is_as_long_as_asked_and_at_most_the_trained_size() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let mut d = fx.load(&path, DflashOptions::default());
    assert_eq!(d.hidden_layers(), vec![1, 3]);
    assert_eq!(d.max_drafts(), 7);
    let tokens = tokens_to(9);
    for (max, want) in [(0, 0), (1, 1), (4, 4), (7, 7), (20, 7)] {
        let out = step(
            &mut d,
            1,
            &tokens,
            if max == 0 { 0 } else { 8 },
            if max == 0 { 8 } else { 0 },
            max,
            0,
        );
        assert_eq!(out.tokens.len(), want, "max {max}");
        // Greedy drafting: point masses.
        assert!(out.probs.is_empty());
        assert!(out.tokens.iter().all(|&t| (t as usize) < N_VOCAB));
    }
}

#[test]
fn greedy_drafts_are_the_most_likely_tokens() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(17);
    let mut g = fx.load(&path, DflashOptions::default());
    let mut s = fx.load(&path, sampling(4096));
    let greedy = step(&mut g, 1, &tokens, 0, 16, 7, 0);
    let dist = step(&mut s, 1, &tokens, 0, 16, 7, 0);
    for (i, q) in dist.probs.iter().enumerate() {
        let best = q
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(greedy.tokens[i] as usize, best, "position {i}");
    }
}

#[test]
fn sampled_drafts_follow_their_distributions() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(17);
    let mut counts = vec![vec![0u32; N_VOCAB]; 3];
    let mut probs = Vec::new();
    // Sample the same block many times (the context is fixed, the draws differ).
    let mut d = fx.load(&path, sampling(64));
    step(&mut d, 1, &tokens, 0, 16, 0, 0);
    let n = 400;
    for _ in 0..n {
        let out = step(&mut d, 1, &tokens, 16, 0, 3, 0);
        for (i, &t) in out.tokens.iter().enumerate() {
            counts[i][t as usize] += 1;
        }
        probs = out.probs;
    }
    for (i, q) in probs.iter().enumerate() {
        // The most likely token's frequency is within 5 standard deviations of q.
        let top = q
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let (p, f) = (q[top] as f64, counts[i][top] as f64 / n as f64);
        let sd = (p * (1.0 - p) / n as f64).sqrt();
        assert!((f - p).abs() < 5.0 * sd + 1e-3, "position {i}: {f} vs {p}");
    }
}

#[test]
fn a_reduced_draft_vocabulary_maps_to_target_tokens() {
    let fx = Fixture::new();
    // 40 draft tokens standing for target ids 3, 9, 15, ...
    let ids: Vec<u32> = (0..40).map(|i| 3 + 6 * i).collect();
    let c = TinyDraft {
        draft_vocab: Some(ids.clone()),
        ..TinyDraft::default()
    };
    let gguf = fx.gguf("d.gguf", &c);
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::F32).unwrap();
    let tokens = tokens_to(14);
    for path in [&gguf, &hf] {
        let mut d = fx.load(path, sampling(64));
        let out = step(&mut d, 1, &tokens, 0, 13, 7, 0);
        let r = Reference::new(&fx, &c, |x| x);
        let mut block = vec![c.mask_token; 8];
        block[0] = tokens[13];
        let want = r.logits(&ctx_rows(0..13, 0), &block, 13, 1);
        for (i, row) in want.iter().enumerate() {
            assert!(
                max_diff(&r.probs(row), &out.probs[i]) < 2e-3,
                "position {i}"
            );
            assert!(ids.contains(&out.tokens[i]));
            // Mass only on the draft vocabulary.
            let outside: f32 = out.probs[i]
                .iter()
                .enumerate()
                .filter(|(t, _)| !ids.contains(&(*t as u32)))
                .map(|(_, p)| p)
                .sum();
            assert_eq!(outside, 0.0);
        }
    }
}

#[test]
fn sequences_do_not_share_state() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let (ta, tb) = (tokens_to(11), tokens_to(14));
    let mut d = fx.load(&path, sampling(64));
    // Interleaved, one of them released and its cache reused.
    step(&mut d, 1, &ta, 0, 10, 0, 0);
    let b1 = step(&mut d, 2, &tb, 0, 13, 4, 3);
    let a1 = step(&mut d, 1, &ta, 10, 0, 4, 0);
    d.release(1);
    let a2 = step(&mut d, 3, &ta, 0, 10, 0, 0);
    assert!(a2.tokens.is_empty());
    let a2 = step(&mut d, 3, &ta, 10, 0, 4, 0);
    assert_eq!(a1.probs, a2.probs);
    let mut alone = fx.load(&path, sampling(64));
    let b_alone = step(&mut alone, 9, &tb, 0, 13, 4, 3);
    assert_eq!(b1.probs, b_alone.probs);
}

#[test]
fn bad_checkpoints_are_refused_with_a_reason() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let err = |info: TargetInfo| DflashDrafter::load(&info, DflashOptions::default()).err();

    // No --spec-draft.
    let mut info = fx.info(&path);
    info.draft = None;
    assert!(err(info).unwrap().contains("--spec-draft"));

    // The target is smaller than the layers the drafter reads.
    let mut info = fx.info(&path);
    info.n_layer = 3;
    assert!(err(info).unwrap().contains("layer 3"));

    // A different hidden size.
    let mut info = fx.info(&path);
    info.n_embd = 32;
    assert!(err(info).is_some());

    // A different vocabulary and no d2t.
    let mut info = fx.info(&path);
    info.n_vocab = 300;
    assert!(err(info).unwrap().contains("vocabulary"));

    // A drafter with a DSpark head.
    let markov = tiny::write_gguf_with(
        fx.root.join("m.gguf"),
        &c,
        &[("markov_w1.weight".into(), vec![4, 259], vec![0.0; 4 * 259])],
    );
    markov.unwrap();
    let e = err(fx.info(&fx.root.join("m.gguf"))).unwrap();
    assert!(e.contains("dspark"), "{e}");

    // Gemma-style layer norms.
    tiny::write_gguf_with(
        fx.root.join("g.gguf"),
        &c,
        &[(
            "blk.0.post_attention_norm.weight".into(),
            vec![64],
            vec![1.0; 64],
        )],
    )
    .unwrap();
    let e = err(fx.info(&fx.root.join("g.gguf"))).unwrap();
    assert!(e.contains("post_attention_norm"), "{e}");

    // A DFlash 2 config with only some of its settings.
    let mut w = crate::gguf::GgufWriter::new();
    w.set(
        "general.architecture",
        &crate::gguf::Value::String("dflash".into()),
    )
    .unwrap();
    w.set("dflash.conv_kernel_size", &crate::gguf::Value::U32(4))
        .unwrap();
    w.add_tensor("fc.weight", crate::ggml::types::F32, &[1, 1], &[0; 4])
        .unwrap();
    w.write(fx.root.join("v2.gguf")).unwrap();
    let e = err(fx.info(&fx.root.join("v2.gguf"))).unwrap();
    assert!(
        e.contains("DFlash 2 drafter without `conv_group_size`"),
        "{e}"
    );

    // Another architecture.
    let e = err(fx.info(&fx.target)).unwrap();
    assert!(e.contains("not a DFlash drafter"), "{e}");

    // A directory without a checkpoint.
    let empty = fx.root.join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(err(fx.info(&empty)).is_some());
}

#[test]
fn hugging_face_names_and_configs() {
    use super::weights::hf_name;
    assert_eq!(
        hf_name("hidden_norm.weight").unwrap(),
        "enc.output_norm.weight"
    );
    assert_eq!(hf_name("model.fc.weight").unwrap(), "fc.weight");
    assert_eq!(
        hf_name("model.layers.3.self_attn.o_proj.weight").unwrap(),
        "blk.3.attn_output.weight"
    );
    assert_eq!(
        hf_name("layers.0.post_attention_layernorm.weight").unwrap(),
        "blk.0.ffn_norm.weight"
    );
    assert!(hf_name("layers.0.something.weight").is_none());

    let cfg = |v: serde_json::Value| Config::from_hf(&v, Flavor::Dflash);
    let base = serde_json::json!({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 96,
        "dflash_config": {"mask_token_id": 5, "target_layer_ids": [1, 3]},
    });
    let c = cfg(base.clone()).unwrap();
    assert_eq!((c.head_dim, c.block_size, c.n_rot), (16, 16, 16));
    assert_eq!(c.target_layers, vec![1, 3]);
    let mut v = base.clone();
    v["rope_parameters"] = serde_json::json!({"partial_rotary_factor": 0.5, "rope_theta": 1e6});
    let c = cfg(v).unwrap();
    assert_eq!((c.n_rot, c.rope_base), (8, 1e6));
    let mut v = base.clone();
    v["dflash_config"]["conv_kernel_size"] = 4.into();
    assert!(matches!(cfg(v), Err(ConfigError::Unsupported(_))));
    let mut v = base.clone();
    v["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    assert!(matches!(cfg(v), Err(ConfigError::Unsupported(_))));
    let mut v = base.clone();
    v["num_key_value_heads"] = 3.into();
    assert!(matches!(cfg(v), Err(ConfigError::Invalid(_))));
    let mut v = base;
    v["dflash_config"]["target_layer_ids"] = serde_json::json!([]);
    assert!(matches!(cfg(v), Err(ConfigError::Invalid(_))));
}
