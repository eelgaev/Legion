use super::*;
use crate::ggml::Backend;
use crate::vision::tiny;

fn encoder(dir: &std::path::Path) -> Encoder {
    let path = dir.join("mmproj.gguf");
    tiny::write_gemma4v(&path, 24, 11).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    Encoder::load(&path, cpu).unwrap()
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    tiny::bmp(w, h, |x, y| {
        [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
    })
}

/// A tensor of the tiny tower as F32 values.
fn values(enc: &Encoder, name: &str) -> Vec<f32> {
    enc.weight(name).unwrap().to_f32()
}

/// An F32 or F16 tensor of a tower as F32 values.
fn widened(enc: &Encoder, name: &str) -> Vec<f32> {
    let t = enc.weight(name).unwrap();
    if t.ty() == crate::ggml::types::F16 {
        t.bytes()
            .chunks_exact(2)
            .map(|h| crate::cpu::gpu_exact::f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
            .collect()
    } else {
        t.to_f32()
    }
}

fn rms(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v *= s * w.map_or(1.0, |w| w[i]);
    }
}

/// `y = W x` for a ggml matrix `[k, n]` (row `o` is `w[o * k..]`).
fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n)
        .map(|o| w[o * k..][..k].iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// NEOX rotation of `x[..half]` (pairs `j`, `j + half / 2`) at position `pos`.
fn rope(x: &mut [f32], half: usize, pos: i32) {
    let q = half / 2;
    for j in 0..q {
        let theta = pos as f32 * ROPE_THETA.powf(-2.0 * j as f32 / half as f32);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[j], x[j + q]);
        x[j] = a * c - b * s;
        x[j + q] = a * s + b * c;
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// The tower written out loop by loop from the module docs: the oracle of the graph's layout
/// (patch order, the two tables, which half rotates by what, the pool's order).
fn reference(enc: &Encoder, p: &Preprocessed) -> Vec<f32> {
    let c = enc.config().clone();
    let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
    let (e, ps, k) = (c.n_embd, c.patch_size, c.merge);
    let (heads, d) = (c.n_head, c.d_head());
    let n = gx * gy;
    let (w_px, h_px) = (p.width as usize, p.height as usize);
    let patch = values(enc, "v.patch_embd.weight");
    let tbl = values(enc, "v.position_embd.weight");
    let mut x = vec![0f32; n * e];
    for i in 0..n {
        let (px, py) = (i % gx, i / gx);
        for o in 0..e {
            let mut acc = 0.0;
            for ch in 0..3 {
                for ky in 0..ps {
                    for kx in 0..ps {
                        let v = p.pixels[ch * w_px * h_px + (py * ps + ky) * w_px + px * ps + kx];
                        acc += patch[((o * 3 + ch) * ps + ky) * ps + kx] * (2.0 * v - 1.0);
                    }
                }
            }
            x[i * e + o] = acc + tbl[px * e + o] + tbl[(c.pos_side + py) * e + o];
        }
    }
    let ff = values(enc, "v.blk.0.ffn_gate.weight").len() / e;
    for il in 0..c.n_layer {
        let wv = |s: &str| values(enc, &format!("v.blk.{il}.{s}.weight"));
        let (ln1, ln2, apn, fpn) = (
            wv("ln1"),
            wv("ln2"),
            wv("attn_post_norm"),
            wv("ffn_post_norm"),
        );
        let (qn, kn) = (wv("attn_q_norm"), wv("attn_k_norm"));
        let (wq, wk, wvv, wo) = (wv("attn_q"), wv("attn_k"), wv("attn_v"), wv("attn_out"));
        let (wg, wu, wd) = (wv("ffn_gate"), wv("ffn_up"), wv("ffn_down"));
        let mut q = vec![0f32; n * e];
        let mut kk = vec![0f32; n * e];
        let mut v = vec![0f32; n * e];
        for i in 0..n {
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, Some(&ln1), c.eps);
            q[i * e..][..e].copy_from_slice(&matvec(&wq, &h, e));
            kk[i * e..][..e].copy_from_slice(&matvec(&wk, &h, e));
            v[i * e..][..e].copy_from_slice(&matvec(&wvv, &h, e));
            let (px, py) = ((i % gx) as i32, (i / gx) as i32);
            for hd in 0..heads {
                let at = i * e + hd * d;
                for (t, nw) in [(&mut q, &qn), (&mut kk, &kn)] {
                    let s = &mut t[at..at + d];
                    rms(s, Some(nw), c.eps);
                    rope(&mut s[..d / 2], d / 2, px);
                    rope(&mut s[d / 2..], d / 2, py);
                }
                rms(&mut v[at..at + d], None, c.eps);
            }
        }
        let mut attn = vec![0f32; n * e];
        for i in 0..n {
            for hd in 0..heads {
                let qi = &q[i * e + hd * d..][..d];
                let s: Vec<f32> = (0..n)
                    .map(|j| {
                        qi.iter()
                            .zip(&kk[j * e + hd * d..][..d])
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let m = s.iter().cloned().fold(f32::MIN, f32::max);
                let ex: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
                let z: f32 = ex.iter().sum();
                for (j, p) in ex.iter().enumerate() {
                    for t in 0..d {
                        attn[i * e + hd * d + t] += p / z * v[j * e + hd * d + t];
                    }
                }
            }
        }
        for i in 0..n {
            let mut o = matvec(&wo, &attn[i * e..][..e], e);
            rms(&mut o, Some(&apn), c.eps);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, Some(&ln2), c.eps);
            let g = matvec(&wg, &h, ff);
            let u = matvec(&wu, &h, ff);
            let gu: Vec<f32> = g.iter().zip(&u).map(|(a, b)| gelu(*a) * b).collect();
            let mut o = matvec(&wd, &gu, e);
            rms(&mut o, Some(&fpn), c.eps);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
        }
    }
    let (bias, scale) = (values(enc, "v.std_bias"), values(enc, "v.std_scale"));
    let proj = values(enc, "mm.input_projection.weight");
    let (ox, oy) = (gx / k, gy / k);
    let mut out = Vec::with_capacity(ox * oy * c.n_out);
    for t in 0..ox * oy {
        let (tx, ty) = (t % ox, t / ox);
        let mut m = vec![0f32; e];
        for dy in 0..k {
            for dx in 0..k {
                let i = (ty * k + dy) * gx + tx * k + dx;
                for (o, mv) in m.iter_mut().enumerate() {
                    *mv += x[i * e + o] / (k * k) as f32;
                }
            }
        }
        for (o, mv) in m.iter_mut().enumerate() {
            *mv = (*mv * (e as f32).sqrt() - bias[o]) * scale[o];
        }
        rms(&mut m, None, c.eps);
        out.extend(matvec(&proj, &m, c.n_out));
    }
    out
}

/// Gemma 4's sizes: a 16 × 16 image grows to the fewest pixels (70 tokens of 12 × 12 pixels),
/// straight to 108 × 108 with no padding: a 27 × 27 patch grid, 9 × 9 pooled rows of the
/// language model's width, the same bits every time; other pixels give other rows.
#[test]
fn encodes_an_image_to_one_row_per_pooled_block() {
    let dir = tempfile::tempdir().unwrap();
    let mut enc = encoder(dir.path());
    assert_eq!(enc.hparams().projector, crate::vision::Projector::Gemma4v);
    assert_eq!(enc.hparams().spatial_merge_size, 3);
    assert_eq!(enc.config().n_out, 24);
    let small = enc.encode_bytes(&gradient(16, 16)).unwrap();
    assert_eq!((small.nx, small.ny, small.n_embd), (9, 9, 24));
    assert!(small.data.iter().all(|v| v.is_finite()));
    assert_eq!(small, enc.encode_bytes(&gradient(16, 16)).unwrap());
    let other = enc
        .encode_bytes(&tiny::bmp(16, 16, |x, _| [x as u8 * 16, 0, 255]))
        .unwrap();
    assert_ne!(small.data, other.data);
    let wide = enc.encode_bytes(&gradient(240, 60)).unwrap();
    assert!(wide.nx > wide.ny, "{} × {}", wide.nx, wide.ny);
}

/// The graph computes what the loops of [`reference`] compute, on a grid that is not square
/// (columns and rows told apart).
#[test]
fn matches_the_reference_loops() {
    let dir = tempfile::tempdir().unwrap();
    let mut enc = encoder(dir.path());
    let p = crate::vision::preprocess(enc.hparams(), &gradient(150, 90)).unwrap();
    assert_ne!(p.grid_w, p.grid_h);
    let got = enc.encode(&p).unwrap();
    let want = reference(&enc, &p);
    assert_eq!(got.data.len(), want.len());
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    for (i, (a, b)) in got.data.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 2e-3 * scale, "{i}: {a} vs {b}");
    }
}

/// Raster order: columns, then rows.
#[test]
fn patch_positions_are_raster() {
    let (xs, ys) = patch_positions(3, 2);
    assert_eq!(xs, [0, 1, 2, 0, 1, 2]);
    assert_eq!(ys, [0, 0, 0, 1, 1, 1]);
}

/// Gemma 4 resizes to the aligned size itself: no black border.
#[test]
fn resizes_without_padding() {
    let dir = tempfile::tempdir().unwrap();
    let enc = encoder(dir.path());
    let white = tiny::bmp(200, 50, |_, _| [255, 255, 255]);
    let p = crate::vision::preprocess(enc.hparams(), &white).unwrap();
    assert_eq!((p.width % 12, p.height % 12), (0, 0));
    assert!(p.pixels.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

/// Every size Gemma 4's resize returns is within its pixel budget, so its longest side is at
/// most the GPU tower's position rows (`image_max_pixels / 48 / 16`, a 48-pixel-wide image that
/// tall) and both towers take the same grids; thin images llama.cpp leaves over the budget
/// shrink to it, the rest keep llama.cpp's sizes.
#[test]
fn thin_images_stay_within_the_pixel_budget() {
    use crate::vision::preprocess::smart_resize;
    let (f, min, max) = (48, 70 * 48 * 48, 1120 * 48 * 48);
    let gpu_rows = max / f / 16;
    assert_eq!(gpu_rows, 3360);
    // llama.cpp: 48 × 56784 (3,549 rows); here the long side shrinks to the budget.
    assert_eq!(smart_resize(48, 60000, f, min, max).unwrap(), (48, 53760));
    assert_eq!(smart_resize(60000, 48, f, min, max).unwrap(), (53760, 48));
    // Within the budget: llama.cpp's sizes.
    assert_eq!(smart_resize(1024, 768, f, min, max).unwrap(), (1008, 768));
    assert_eq!(smart_resize(48, 50000, f, min, max).unwrap(), (48, 50016));
    assert_eq!(smart_resize(4000, 3000, f, min, max).unwrap(), (1824, 1344));
    for w in [1, 16, 47, 48, 60, 95] {
        for h in [
            1, 100, 5_000, 53_760, 55_000, 60_000, 65_000, 70_000, 1_000_000,
        ] {
            for (w, h) in [(w, h), (h, w)] {
                // Too thin even for llama.cpp (a side over 65,536 px): refused, as before.
                let Ok((rw, rh)) = smart_resize(w, h, f, min, max) else {
                    continue;
                };
                assert!(rw * rh <= max, "{w}x{h}: {rw}x{rh}");
                assert!((rw / 16).max(rh / 16) <= gpu_rows, "{w}x{h}: {rw}x{rh}");
                assert_eq!((rw % f, rh % f), (0, 0), "{w}x{h}");
                // Tokens: at most the tower's largest image (the steps and rings hold it).
                assert!((rw / f) * (rh / f) <= 1120, "{w}x{h}: {rw}x{rh}");
            }
        }
    }
}

/// The most tokens a Gemma 4 image takes, from the tower's pixel limits: the steps and the
/// sliding-window rings are sized to hold it whole (#1217).
#[test]
fn the_largest_image_is_the_pixel_budget_in_tokens() {
    let mut hp = crate::vision::VisionHparams {
        projector: crate::vision::Projector::Gemma4v,
        patch_size: 16,
        spatial_merge_size: 3,
        image_min_pixels: 70 * 48 * 48,
        image_max_pixels: 1120 * 48 * 48,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    };
    assert_eq!(hp.max_image_tokens(), 1120);
    // An mmproj's own limit, not a whole number of tokens: rounded down.
    hp.image_max_pixels = 600 * 48 * 48 + 47;
    assert_eq!(hp.max_image_tokens(), 600);
    // Qwen3-VL's 2×2 merge of 16-pixel patches, 4,096 tokens.
    hp.spatial_merge_size = 2;
    hp.image_max_pixels = 4096 * 32 * 32;
    assert_eq!(hp.max_image_tokens(), 4096);
}

/// A Gemma 4 checkpoint's tower (`LEGION_TEST_VISION_HF=<dir>`, opened by
/// [`crate::hf::open_vision`]) against the converter's mmproj of the same model
/// (`LEGION_TEST_GEMMA_MMPROJ=<mmproj-F16.gguf>`): the same keys and every tensor the graph reads
/// with the same values (F16 matrices from BF16 round the same way), the projector's shape.
#[test]
#[ignore = "needs a Gemma 4 checkpoint and its mmproj"]
fn checkpoint_tower_matches_the_mmproj() {
    let (Some(hf), Some(mmproj)) = (
        std::env::var_os("LEGION_TEST_VISION_HF"),
        std::env::var_os("LEGION_TEST_GEMMA_MMPROJ"),
    ) else {
        return;
    };
    let g = crate::hf::open_vision(std::path::Path::new(&hf))
        .unwrap()
        .expect("a vision tower");
    let load = |g: &crate::gguf::GgufFile| {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        Encoder::from_gguf(g, cpu).unwrap()
    };
    let ours = load(&g);
    let theirs = load(&crate::gguf::GgufFile::open(mmproj).unwrap());
    assert_eq!(
        format!("{:?}", ours.hparams()),
        format!("{:?}", theirs.hparams())
    );
    assert_eq!(ours.config(), theirs.config());
    for name in names(ours.config().n_layer) {
        let (a, b) = (widened(&ours, &name), widened(&theirs, &name));
        assert_eq!(a.len(), b.len(), "{name}");
        // The QAT GGUF's projector is the QAT checkpoint's own (cosine 0.9996 to the base
        // model's, which the NVFP4 checkpoints quantize); its tower is the base model's.
        if name == "mm.input_projection.weight" {
            continue;
        }
        // Where the mmproj keeps a tensor F16 that is F32 here (a kernel), only values below
        // F16's normal range may differ, by under one F16 subnormal step.
        let diff = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x != y && !(y.abs() < 6.2e-5 && (*x - *y).abs() < 6e-8))
            .count();
        assert_eq!(diff, 0, "{name}: {diff} of {} values differ", a.len());
    }
}
