use super::*;
use crate::ggml::Backend;
use crate::vision::tiny;

fn encoder_with(dir: &std::path::Path, mm_bias: bool) -> Encoder {
    let path = dir.join("mmproj.gguf");
    tiny::write_glm5v(&path, 24, 13, mm_bias).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    Encoder::load(&path, cpu).unwrap()
}

fn encoder(dir: &std::path::Path) -> Encoder {
    encoder_with(dir, false)
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    tiny::bmp(w, h, |x, y| {
        [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
    })
}

/// A tensor of the tower as F32 values: the tiny tower's are F32, a checkpoint's matrices F16
/// (BF16 also read).
fn values(enc: &Encoder, name: &str) -> Vec<f32> {
    widen(enc.weight(name).unwrap())
}

fn widen(t: crate::ggml::Tensor<'_>) -> Vec<f32> {
    use crate::cpu::gpu_exact::f16_to_f32;
    let b = t.bytes();
    match t.ty() {
        crate::ggml::types::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        crate::ggml::types::F16 => b
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        crate::ggml::types::BF16 => b
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        ty => panic!("`{}`: type {ty:?}", t.name()),
    }
}

fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v *= s * w;
    }
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let s = 1.0 / (var + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * s * w[i] + b[i];
    }
}

/// `y = W x (+ b)` for a ggml matrix `[k, n]` (row `o` is `w[o * k..]`), summed in F64; a big
/// one (a checkpoint's) on all cores.
fn matvec(w: &[f32], b: Option<&[f32]>, x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    let row = move |o: usize| {
        let y: f64 = w[o * k..][..k]
            .iter()
            .zip(x)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        (y + b.map_or(0.0, |b| f64::from(b[o]))) as f32
    };
    if n * k < 1 << 20 {
        return (0..n).map(row).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = n.div_ceil(threads);
    let mut y = vec![0f32; n];
    std::thread::scope(|s| {
        for (ci, chunk) in y.chunks_mut(per).enumerate() {
            s.spawn(move || {
                for (j, v) in chunk.iter_mut().enumerate() {
                    *v = row(ci * per + j);
                }
            });
        }
    });
    y
}

/// erf to about 1e-7 (Abramowitz and Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    let (s, x) = (x.signum(), x.abs() as f64);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    s * (1.0 - poly * (-x * x).exp()) as f32
}

fn swiglu_clamp(g: f32, u: f32, limit: f32) -> f32 {
    let g = g.min(limit);
    g / (1.0 + (-g).exp()) * u.clamp(-limit, limit)
}

/// `ROPE_VISION` of one head `x` (`d` values) at patch (row `r`, column `c`): pairs `j` and
/// `j + d / 2`; the first quarter's angles from the row, the second's from the column, each
/// at `base^(-4 i / d)`.
fn rope(x: &mut [f32], r: i32, c: i32) {
    let d = x.len();
    let (half, quarter) = (d / 2, d / 4);
    for j in 0..half {
        let (pos, i) = if j < quarter {
            (r, j)
        } else {
            (c, j - quarter)
        };
        let theta = pos as f32 * ROPE_THETA.powf(-4.0 * i as f32 / d as f32);
        let (s, cs) = theta.sin_cos();
        let (a, b) = (x[j], x[j + half]);
        x[j] = a * cs - b * s;
        x[j + half] = a * s + b * cs;
    }
}

/// The SwiGLU MLP of a block or the projector: `prefix{gate,up,down}`, biases where present.
struct Mlp {
    w: [Vec<f32>; 3],
    b: [Option<Vec<f32>>; 3],
}

impl Mlp {
    fn load(enc: &Encoder, prefix: &str) -> Self {
        let get = |s: &str| enc.weight(&format!("{prefix}{s}")).map(widen);
        let [w, b] =
            [".weight", ".bias"].map(|p| ["gate", "up", "down"].map(|m| get(&format!("{m}{p}"))));
        Self {
            w: w.map(Result::unwrap),
            b: b.map(Result::ok),
        }
    }

    fn apply(&self, x: &[f32], limit: f32) -> Vec<f32> {
        let ff = self.w[0].len() / x.len();
        let n = self.w[2].len() / ff;
        let g = matvec(&self.w[0], self.b[0].as_deref(), x, ff);
        let u = matvec(&self.w[1], self.b[1].as_deref(), x, ff);
        let h: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| swiglu_clamp(g, u, limit))
            .collect();
        matvec(&self.w[2], self.b[2].as_deref(), &h, n)
    }
}

/// The tower written out loop by loop from the module docs, patches row by row: the oracle of
/// the graph's layout (the two kernels, which quarter rotates by what, the 2×2 convolution's
/// order, the output's order).
fn reference(enc: &Encoder, p: &Preprocessed) -> Vec<f32> {
    reference_stages(enc, p).pop().unwrap().1
}

/// [`reference`] and the rows after each stage, named as [`encode_tapped`]'s; patches row by
/// row (not in merge order).
pub(crate) fn reference_stages(enc: &Encoder, p: &Preprocessed) -> Vec<(String, Vec<f32>)> {
    let mut stages = Vec::new();
    let c = enc.config().clone();
    let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
    let (e, ps) = (c.n_embd, c.patch_size);
    let (heads, d) = (c.n_head, c.d_head());
    let n = gx * gy;
    let (w_px, h_px) = (p.width as usize, p.height as usize);
    let (k0, k1) = (
        values(enc, "v.patch_embd.weight"),
        values(enc, "v.patch_embd.weight.1"),
    );
    let pb = values(enc, "v.patch_embd.bias");
    let mut x = vec![0f32; n * e];
    for i in 0..n {
        let (px, py) = (i % gx, i / gx);
        for o in 0..e {
            let mut acc = 0.0;
            for ch in 0..3 {
                for ky in 0..ps {
                    for kx in 0..ps {
                        let v = p.pixels[ch * w_px * h_px + (py * ps + ky) * w_px + px * ps + kx];
                        let at = ((o * 3 + ch) * ps + ky) * ps + kx;
                        acc += (k0[at] + k1[at]) * v;
                    }
                }
            }
            x[i * e + o] = acc + pb[o];
        }
    }
    stages.push(("patches".to_string(), x.clone()));
    for il in 0..c.n_layer {
        let wv = |s: &str| values(enc, &format!("v.blk.{il}.{s}"));
        let (ln1, ln2) = (wv("ln1.weight"), wv("ln2.weight"));
        let (qn, kn) = (wv("attn_q_norm.weight"), wv("attn_k_norm.weight"));
        let (wqkv, bqkv) = (wv("attn_qkv.weight"), wv("attn_qkv.bias"));
        let (wo, bo) = (wv("attn_out.weight"), wv("attn_out.bias"));
        let ffn = Mlp::load(enc, &format!("v.blk.{il}.ffn_"));
        let mut qkv = vec![0f32; n * 3 * e];
        for i in 0..n {
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, &ln1, c.eps);
            let mut t = matvec(&wqkv, Some(&bqkv), &h, 3 * e);
            let (r, col) = ((i / gx) as i32, (i % gx) as i32);
            for hd in 0..heads {
                for (part, nw) in [(0, &qn), (1, &kn)] {
                    let s = &mut t[part * e + hd * d..][..d];
                    rms(s, nw, c.eps);
                    rope(s, r, col);
                }
            }
            qkv[i * 3 * e..][..3 * e].copy_from_slice(&t);
        }
        let scale = 1.0 / (d as f32).sqrt();
        let mut attn = vec![0f32; n * e];
        for i in 0..n {
            for hd in 0..heads {
                let qi = &qkv[i * 3 * e + hd * d..][..d];
                let s: Vec<f32> = (0..n)
                    .map(|j| {
                        let kj = &qkv[j * 3 * e + e + hd * d..][..d];
                        qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = s.iter().cloned().fold(f32::MIN, f32::max);
                let ex: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
                let z: f32 = ex.iter().sum();
                for (j, pj) in ex.iter().enumerate() {
                    for t in 0..d {
                        attn[i * e + hd * d + t] += pj / z * qkv[j * 3 * e + 2 * e + hd * d + t];
                    }
                }
            }
        }
        for i in 0..n {
            let o = matvec(&wo, Some(&bo), &attn[i * e..][..e], e);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
        }
        stages.push((format!("blk.{il}.attn"), x.clone()));
        for i in 0..n {
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, &ln2, c.eps);
            let o = ffn.apply(&h, c.swiglu_limit);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
        }
        stages.push((format!("blk.{il}"), x.clone()));
    }
    let post = values(enc, "v.post_ln.weight");
    for i in 0..n {
        rms(&mut x[i * e..][..e], &post, c.eps);
    }
    stages.push(("post_ln".to_string(), x.clone()));
    // The merger: each 2×2 block's four patches through the convolution ([kw, kh, e, o]: kw
    // is the patch's column in the block, kh its row), then the projector.
    let (conv, cb) = (
        values(enc, "mm.patch_merger.weight"),
        values(enc, "mm.patch_merger.bias"),
    );
    let mw = cb.len();
    let fc = values(enc, "mm.model.fc.weight");
    let (nw, nb) = (
        values(enc, "mm.post_norm.weight"),
        values(enc, "mm.post_norm.bias"),
    );
    let proj = Mlp::load(enc, "mm.");
    let (ox, oy) = (gx / 2, gy / 2);
    let [mut down, mut fcs, mut lns, mut out] = [(); 4].map(|_| Vec::new());
    for t in 0..ox * oy {
        let (tx, ty) = (t % ox, t / ox);
        let mut m = cb.clone();
        for (o, mv) in m.iter_mut().enumerate() {
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (ty * 2 + dy) * gx + tx * 2 + dx;
                    for ch in 0..e {
                        *mv += conv[((o * e + ch) * 2 + dy) * 2 + dx] * x[i * e + ch];
                    }
                }
            }
        }
        let mut h = matvec(&fc, None, &m, mw);
        down.extend(m);
        fcs.extend_from_slice(&h);
        layer_norm(&mut h, &nw, &nb, MERGER_EPS);
        for v in h.iter_mut() {
            *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
        }
        lns.extend_from_slice(&h);
        out.extend(proj.apply(&h, c.swiglu_limit));
    }
    for (name, v) in [
        ("downsample", down),
        ("fc", fcs),
        ("ln_gelu", lns),
        ("out", out),
    ] {
        stages.push((name.to_string(), v));
    }
    stages
}

fn assert_close(got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= rel * scale, "{i}: {a} vs {b}");
    }
}

/// The mmproj's keys and tensors: GLM-5.3-Flash's tower, merged 2 × 2, clamped at its
/// `swiglu_clamp`, the language model's width from `mm.down`; every tensor [`names`] lists is
/// there.
#[test]
fn reads_the_mmproj() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder(dir.path());
    let hp = enc.hparams();
    assert_eq!(hp.projector, crate::vision::Projector::Glm5v);
    assert_eq!((hp.patch_size, hp.spatial_merge_size), (6, 2));
    assert_eq!((hp.image_min_pixels, hp.image_max_pixels), (576, 9216));
    assert_eq!(hp.max_image_tokens(), 64);
    let c = enc.config();
    assert_eq!((c.n_embd, c.n_head, c.n_layer, c.n_out), (16, 2, 2, 24));
    assert_eq!(c.swiglu_limit, tiny::GLM_SWIGLU_CLAMP);
    assert_eq!(c.eps, 1e-5);
    for name in names(c.n_layer) {
        assert!(enc.weight(&name).is_ok(), "{name}");
    }
}

/// llama.cpp requires `clip.vision.swiglu_clamp` for `glm5v`; so does this tower.
#[test]
fn refuses_an_mmproj_without_the_clamp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    tiny::write_glm5v(&path, 24, 13, false).unwrap();
    let g = crate::gguf::GgufFile::open(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut w = crate::gguf::GgufWriter::new();
    for (k, v) in g.metadata() {
        if k != "clip.vision.swiglu_clamp" && !k.starts_with("GGUF.") {
            w.set(&k, &v).unwrap();
        }
    }
    for t in g.tensors() {
        let shape: Vec<i64> = t.shape.iter().map(|&v| v as i64).collect();
        let data = &bytes[t.offset as usize..][..t.size as usize];
        w.add_tensor(&t.name, t.ggml_type, &shape, data).unwrap();
    }
    let stripped = dir.path().join("stripped.gguf");
    w.write(&stripped).unwrap();
    let err = match Encoder::load(&stripped, Backend::cpu().unwrap()) {
        Ok(_) => panic!("an mmproj without swiglu_clamp loaded"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("swiglu_clamp"), "{err}");
}

/// A 60 × 36 image is a 10 × 6 patch grid, 5 × 3 rows of the language model's width, the same
/// bits every time; other pixels give other rows.
#[test]
fn encodes_an_image_to_one_row_per_merged_block() {
    let dir = tempfile::tempdir().unwrap();
    let mut enc = encoder(dir.path());
    let a = enc.encode_bytes(&gradient(60, 36)).unwrap();
    assert_eq!((a.nx, a.ny, a.n_embd), (5, 3, 24));
    assert!(a.data.iter().all(|v| v.is_finite()));
    assert_eq!(a, enc.encode_bytes(&gradient(60, 36)).unwrap());
    let other = enc
        .encode_bytes(&tiny::bmp(60, 36, |x, _| [x as u8 * 4, 0, 255]))
        .unwrap();
    assert_ne!(a.data, other.data);
}

/// The graph computes what the loops of [`reference`] compute, on a grid that is not square
/// (rows and columns told apart), with and without projector MLP biases. The random weights
/// are large enough that the clamps cut values.
#[test]
fn matches_the_reference_loops() {
    for mm_bias in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut enc = encoder_with(dir.path(), mm_bias);
        let p = crate::vision::preprocess(enc.hparams(), &gradient(78, 45)).unwrap();
        assert_eq!((p.grid_w, p.grid_h), (14, 8));
        let got = enc.encode(&p).unwrap();
        assert_close(&got.data, &reference(&enc, &p), 2e-3);
    }
}

/// Small images on a checkpoint's tower (`LEGION_TEST_VISION_HF=<dir>`, its real weights; at
/// most 80 patches, as the loops are slow): the CPU tower's rows after every stage against
/// [`reference_stages`]', printed stage by stage, so a gap shows where it starts. The CPU tower
/// rounds the matrices' inputs to F16 and the loops do not, so they are close, not equal.
#[test]
#[ignore = "needs a checkpoint, slow"]
fn cpu_tower_matches_the_reference_on_a_checkpoint() {
    let Some(hf) = std::env::var_os("LEGION_TEST_VISION_HF") else {
        return;
    };
    let g = crate::hf::open_vision(std::path::Path::new(&hf))
        .unwrap()
        .expect("a vision tower");
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut enc = Encoder::from_gguf(&g, cpu).unwrap();
    let mut failed = Vec::new();
    for (label, img) in checkpoint_images() {
        let p = crate::vision::preprocess(enc.hparams(), &img).unwrap();
        let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
        let (_, taps) = enc.encode_stages(&p).unwrap();
        let t = std::time::Instant::now();
        let want = reference_stages(&enc, &p);
        println!(
            "{label}: {} x {} px, {gx} x {gy} patches, loops {:.1} s",
            p.width,
            p.height,
            t.elapsed().as_secs_f64()
        );
        let n = gx * gy;
        for ((name, got), (wname, want)) in taps.iter().zip(&want) {
            assert_eq!(name, wname);
            let (got, rows) = if per_patch(name) {
                (merge_to_rows(got, gx, gy, got.len() / n), n)
            } else {
                (got.clone(), n / 4)
            };
            let d = Diff::new(&got, want, rows);
            println!("  {name:>12}: {d}");
            let close = d.bad == 0 && d.min_cos > 0.99;
            if name == "out" && !close {
                failed.push(format!("{label}: {d}"));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the CPU tower is not the loops': {failed:?}"
    );
}

/// The small images of [`cpu_tower_matches_the_reference_on_a_checkpoint`] (and the GPU's
/// stage test): smooth colours on a 140 × 112 canvas (20 tokens), and sharp stripes grown to
/// the 16-token floor.
pub(crate) fn checkpoint_images() -> Vec<(&'static str, Vec<u8>)> {
    let smooth = tiny::bmp(140, 112, |x, y| {
        [
            (x * 255 / 139) as u8,
            (y * 255 / 111) as u8,
            (128 + (x as i32 - y as i32) / 2) as u8,
        ]
    });
    vec![
        ("smooth 140x112", smooth),
        ("stripes 56x56", gradient(56, 56)),
    ]
}

/// GLM-5.3-Flash's canvases (factor 28, 16 to 8,000 tokens): aligned up; a small image grows
/// to the fewest tokens, a big one shrinks to the most (llama.cpp's
/// `mtmd_image_preprocessor_glm5v`).
#[test]
fn canvas_sizes() {
    use crate::vision::preprocess::{glm5_canvas, glm5_content};
    let (f, min, max) = (28, 16 * 784, 8000 * 784);
    let canvas = |w, h| glm5_canvas(w, h, f, min, max).unwrap();
    assert_eq!(canvas(1024, 768), (1036, 784));
    assert_eq!(canvas(4000, 100), (4004, 112));
    assert_eq!(canvas(10, 10), (112, 112));
    assert_eq!(canvas(8000, 8000), (2492, 2492));
    assert_eq!(canvas(28, 28), (112, 112));
    for (w, h) in [(1, 1), (3, 5000), (7000, 9), (1920, 1080), (8000, 6000)] {
        let (cw, ch) = canvas(w, h);
        assert_eq!((cw % f, ch % f), (0, 0), "{w}x{h}");
        assert!((min..=max).contains(&(cw * ch)), "{w}x{h}: {cw}x{ch}");
        let (rw, rh) = glm5_content(w, h, (cw, ch), min);
        assert!(rw <= cw && rh <= ch && rw >= 1 && rh >= 1, "{w}x{h}");
    }
    // The content is never scaled up unless the image is under the budget's floor.
    assert_eq!(glm5_content(1024, 768, (1036, 784), min), (1024, 768));
    assert_eq!(glm5_content(8000, 8000, (2492, 2492), min), (2492, 2492));
    assert_eq!(glm5_content(6, 5, (24, 16), 256), (19, 16));
    assert!(glm5_canvas(0, 5, f, min, max).is_err());
}

/// A fixed image on the tiny tower (factor 12, 576 to 9,216 pixels): a 32 × 20 white image keeps
/// its size at the top left of a 36 × 24 canvas, black (normalized) on the right and below; a
/// 9 × 5 one, under the floor, grows to 36 × 20 on the same canvas.
#[test]
fn places_the_image_top_left_on_a_black_canvas() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder(dir.path());
    let hp = enc.hparams();
    for ((w, h), (rw, rh)) in [((32, 20), (32, 20)), ((9, 5), (36, 20))] {
        let white = tiny::bmp(w, h, |_, _| [255, 255, 255]);
        let p = crate::vision::preprocess(hp, &white).unwrap();
        assert_eq!((p.width, p.height), (36, 24), "{w}x{h}");
        assert_eq!((p.grid_w, p.grid_h), (6, 4));
        let n = (p.width * p.height) as usize;
        for ch in 0..3 {
            for y in 0..p.height {
                for x in 0..p.width {
                    let v = p.pixels[ch * n + (y * p.width + x) as usize];
                    // (1 - 0.5) / 0.25 inside, (0 - 0.5) / 0.25 outside.
                    let want = if x < rw && y < rh { 2.0 } else { -2.0 };
                    assert!((v - want).abs() < 1e-5, "{w}x{h} ({x}, {y}): {v}");
                }
            }
        }
    }
}

/// The HF checkpoint's names as the tiny mmproj's (`model.visual.*`, row-major shapes, the
/// Conv3D patch kernel's two frames interleaved), F32.
fn checkpoint_tensors(enc: &Encoder, mm_bias: bool) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let c = enc.config().clone();
    let get = |n: &str| values(enc, n);
    let shape = |n: &str| -> Vec<usize> {
        let t = enc.weight(n).unwrap();
        let mut s: Vec<usize> = t.ne().iter().map(|&v| v as usize).collect();
        while s.len() > 1 && s.last() == Some(&1) {
            s.pop();
        }
        s.reverse();
        s
    };
    let mut out = Vec::new();
    let mut put = |hf: &str, gg: &str| out.push((format!("model.visual.{hf}"), shape(gg), get(gg)));
    put("patch_embed.proj.bias", "v.patch_embd.bias");
    for il in 0..c.n_layer {
        for (hf, gg) in [
            ("norm1.weight", "ln1.weight"),
            ("norm2.weight", "ln2.weight"),
            ("attn.q_norm.weight", "attn_q_norm.weight"),
            ("attn.k_norm.weight", "attn_k_norm.weight"),
            ("attn.qkv.weight", "attn_qkv.weight"),
            ("attn.qkv.bias", "attn_qkv.bias"),
            ("attn.proj.weight", "attn_out.weight"),
            ("attn.proj.bias", "attn_out.bias"),
            ("mlp.gate_proj.weight", "ffn_gate.weight"),
            ("mlp.gate_proj.bias", "ffn_gate.bias"),
            ("mlp.up_proj.weight", "ffn_up.weight"),
            ("mlp.up_proj.bias", "ffn_up.bias"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
            ("mlp.down_proj.bias", "ffn_down.bias"),
        ] {
            put(&format!("blocks.{il}.{hf}"), &format!("v.blk.{il}.{gg}"));
        }
    }
    put("post_layernorm.weight", "v.post_ln.weight");
    put("downsample.weight", "mm.patch_merger.weight");
    put("downsample.bias", "mm.patch_merger.bias");
    put("merger.proj.weight", "mm.model.fc.weight");
    put("merger.post_projection_norm.weight", "mm.post_norm.weight");
    put("merger.post_projection_norm.bias", "mm.post_norm.bias");
    for (hf, gg) in [
        ("gate_proj", "mm.gate"),
        ("up_proj", "mm.up"),
        ("down_proj", "mm.down"),
    ] {
        put(&format!("merger.{hf}.weight"), &format!("{gg}.weight"));
        if mm_bias {
            put(&format!("merger.{hf}.bias"), &format!("{gg}.bias"));
        }
    }
    // Conv3D [n, c, 2, p, p]: frame 0 then frame 1 of each (output, channel).
    let (k0, k1) = (get("v.patch_embd.weight"), get("v.patch_embd.weight.1"));
    let p = c.patch_size;
    let plane = p * p;
    let both = k0
        .chunks_exact(plane)
        .zip(k1.chunks_exact(plane))
        .flat_map(|(a, b)| a.iter().chain(b).copied())
        .collect();
    out.push((
        "model.visual.patch_embed.proj.weight".into(),
        vec![c.n_embd, 3, 2, p, p],
        both,
    ));
    out
}

/// GGUF name mapping (#1219): a GLM-5.3-Flash checkpoint's `model.visual.*` tower, written
/// from the tiny mmproj under the checkpoint's names, opens as the same mmproj: the same keys
/// (pixel limits from the processor's token limits, the clamp from `swiglu_limit`), every
/// tensor the graph reads with the same values, and the same rows for an image.
#[test]
fn checkpoint_tower_maps_to_the_mmproj_names() {
    for mm_bias in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut want = encoder_with(dir.path(), mm_bias);
        let ckpt = dir.path().join("ckpt");
        std::fs::create_dir(&ckpt).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, v) in checkpoint_tensors(&want, mm_bias) {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            header.insert(
                name,
                serde_json::json!({"dtype": "F32", "shape": shape,
                    "data_offsets": [data.len(), data.len() + bytes.len()]}),
            );
            data.extend(bytes);
        }
        let mut h = serde_json::to_vec(&header).unwrap();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(data);
        std::fs::write(ckpt.join("model.safetensors"), file).unwrap();
        let config = serde_json::json!({
            "model_type": "glm5_next",
            "vision_config": {
                "depth": 2, "hidden_size": 16, "num_heads": 2, "intermediate_size": 32,
                "out_hidden_size": tiny::GLM_MERGE_WIDTH, "patch_size": 6,
                "spatial_merge_size": 2, "temporal_patch_size": 2, "rms_norm_eps": 1e-5,
                "swiglu_limit": tiny::GLM_SWIGLU_CLAMP, "image_size": 48,
            }
        });
        std::fs::write(ckpt.join("config.json"), config.to_string()).unwrap();
        let processor = serde_json::json!({"image_processor": {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.25, 0.25, 0.25],
            "min_image_tokens": 4, "max_image_tokens": 64,
        }});
        std::fs::write(ckpt.join("processor_config.json"), processor.to_string()).unwrap();

        let g = crate::hf::open_vision(&ckpt)
            .unwrap()
            .expect("a vision tower");
        assert_eq!(
            g.get("clip.projector_type")
                .and_then(|v| v.as_str().map(String::from)),
            Some("glm5v".to_string())
        );
        let mut got = Encoder::from_gguf(&g, Backend::cpu().unwrap()).unwrap();
        assert_eq!(
            format!("{:?}", got.hparams()),
            format!("{:?}", want.hparams())
        );
        assert_eq!(got.config(), want.config());
        let mut all = names(got.config().n_layer);
        if mm_bias {
            all.extend(["mm.gate.bias", "mm.up.bias", "mm.down.bias"].map(String::from));
        }
        for name in all {
            assert_eq!(values(&got, &name), values(&want, &name), "{name}");
        }
        let img = gradient(60, 36);
        assert_eq!(
            got.encode_bytes(&img).unwrap(),
            want.encode_bytes(&img).unwrap()
        );
    }
}
