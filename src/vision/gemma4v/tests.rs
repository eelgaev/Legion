use super::*;
use crate::ggml::Backend;
use crate::vision::tiny;

fn encoder(dir: &std::path::Path) -> Encoder {
    encoder_as(dir, tiny::GEMMA_31B)
}

/// The tiny tower of `kind` (the 31B's, or the clipped E4B-sized one, #1220).
fn encoder_as(dir: &std::path::Path, kind: tiny::GemmaKind) -> Encoder {
    let path = dir.join(format!("mmproj-{}.gguf", kind.clipped as u8));
    tiny::write_gemma4v_as(&path, 24, 11, kind).unwrap();
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

/// A clipped linear (HF's `Gemma4ClippableLinear`): `x` clamped to `c.inp`, multiplied, the
/// product clamped to `c.out`; without `c` a plain product.
fn lin(w: &[f32], x: &[f32], n: usize, c: Option<Clip>) -> Vec<f32> {
    let Some(c) = c else {
        return matvec(w, x, n);
    };
    let x: Vec<f32> = x.iter().map(|v| v.min(c.inp.1).max(c.inp.0)).collect();
    matvec(w, &x, n)
        .into_iter()
        .map(|v| v.min(c.out.1).max(c.out.0))
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
        let cl = |m: &str| clip(&enc.weights, il, m).unwrap();
        let mut q = vec![0f32; n * e];
        let mut kk = vec![0f32; n * e];
        let mut v = vec![0f32; n * e];
        for i in 0..n {
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, Some(&ln1), c.eps);
            q[i * e..][..e].copy_from_slice(&lin(&wq, &h, e, cl("attn_q")));
            kk[i * e..][..e].copy_from_slice(&lin(&wk, &h, e, cl("attn_k")));
            v[i * e..][..e].copy_from_slice(&lin(&wvv, &h, e, cl("attn_v")));
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
            let mut o = lin(&wo, &attn[i * e..][..e], e, cl("attn_out"));
            rms(&mut o, Some(&apn), c.eps);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
            let mut h = x[i * e..][..e].to_vec();
            rms(&mut h, Some(&ln2), c.eps);
            let g = lin(&wg, &h, ff, cl("ffn_gate"));
            let u = lin(&wu, &h, ff, cl("ffn_up"));
            let gu: Vec<f32> = g.iter().zip(&u).map(|(a, b)| gelu(*a) * b).collect();
            let mut o = lin(&wd, &gu, e, cl("ffn_down"));
            rms(&mut o, Some(&fpn), c.eps);
            for (t, val) in o.iter().enumerate() {
                x[i * e + t] += val;
            }
        }
    }
    // Without the standardization: `(x - 0) * 1`.
    let (bias, scale) = if standardizes(&enc.weights) {
        (values(enc, "v.std_bias"), values(enc, "v.std_scale"))
    } else {
        (vec![0.0; e], vec![1.0; e])
    };
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
    use crate::vision::preprocess::{resize_for, smart_resize};
    let hp = gemma_limits();
    let (f, min, max) = (48, 70 * 48 * 48, 1120 * 48 * 48);
    let gpu_rows = max / f / 16;
    assert_eq!(gpu_rows, 3360);
    // llama.cpp: 48 × 56784 (3,549 rows); here the long side shrinks to the budget.
    assert_eq!(smart_resize(48, 60000, f, min, max).unwrap(), (48, 56784));
    assert_eq!(resize_for(&hp, 48, 60000).unwrap(), (48, 53760));
    assert_eq!(resize_for(&hp, 60000, 48).unwrap(), (53760, 48));
    // Within the budget: llama.cpp's sizes.
    assert_eq!(resize_for(&hp, 1024, 768).unwrap(), (1008, 768));
    assert_eq!(resize_for(&hp, 48, 50000).unwrap(), (48, 50016));
    assert_eq!(resize_for(&hp, 4000, 3000).unwrap(), (1824, 1344));
    for w in [1, 16, 47, 48, 60, 95] {
        for h in [
            1, 100, 5_000, 53_760, 55_000, 60_000, 65_000, 70_000, 1_000_000,
        ] {
            for (w, h) in [(w, h), (h, w)] {
                // Too thin even for llama.cpp (a side over 65,536 px): refused, as before.
                let Ok((rw, rh)) = resize_for(&hp, w, h) else {
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

/// Gemma 4's tower limits: 16-pixel patches merged 3 × 3, 70 to 1,120 tokens.
fn gemma_limits() -> crate::vision::VisionHparams {
    crate::vision::VisionHparams {
        projector: crate::vision::Projector::Gemma4v,
        patch_size: 16,
        spatial_merge_size: 3,
        image_min_pixels: 70 * 48 * 48,
        image_max_pixels: 1120 * 48 * 48,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    }
}

/// Only Gemma 4's resize keeps thin images within the pixel budget (#1217): Qwen-VL's sizes
/// stay llama.cpp's (`calc_size_preserved_ratio`), as before the budget clamp, also a thin
/// image the one-factor floor leaves over a checkpoint's small budget. 16-pixel patches merged
/// 2 × 2 and a 32,768-pixel budget: 32 × 2000 px stays 32 × 1408 (45,056 pixels), where the
/// clamp would give 32 × 1024.
#[test]
fn only_gemma_images_shrink_to_the_pixel_budget() {
    use crate::vision::preprocess::{resize_for, smart_resize};
    let mut qwen = crate::vision::VisionHparams {
        projector: crate::vision::Projector::Qwen3vl,
        patch_size: 16,
        spatial_merge_size: 2,
        image_min_pixels: 8192,
        image_max_pixels: 32768,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    };
    for (w, h, size) in [
        (32, 2000, (32, 1408)),
        (2000, 32, (1408, 32)),
        (1, 3000, (32, 9888)),
        (300, 300, (160, 160)),
        (64, 128, (64, 128)),
    ] {
        assert_eq!(resize_for(&qwen, w, h).unwrap(), size, "{w}x{h}");
        assert_eq!(
            smart_resize(w, h, 32, 8192, 32768).unwrap(),
            size,
            "{w}x{h}"
        );
    }
    // The same limits through Gemma 4's path: within the budget.
    let mut gemma = qwen.clone();
    gemma.projector = crate::vision::Projector::Gemma4v;
    assert_eq!(resize_for(&gemma, 32, 2000).unwrap(), (32, 1024));
    assert_eq!(resize_for(&gemma, 2000, 32).unwrap(), (1024, 32));
    assert_eq!(resize_for(&gemma, 1, 3000).unwrap(), (32, 1024));
    assert_eq!(resize_for(&gemma, 300, 300).unwrap(), (160, 160));
    // Qwen3-VL's own budget: the reference's sizes (`preprocess::tests::smart_resize_table`).
    qwen.image_max_pixels = 4194304;
    assert_eq!(resize_for(&qwen, 1, 4096).unwrap(), (32, 4096));
    assert_eq!(resize_for(&qwen, 8192, 2048).unwrap(), (4096, 1024));
}

/// The most tokens a Gemma 4 image takes, from the tower's pixel limits: the steps and the
/// sliding-window rings are sized to hold it whole (#1217).
#[test]
fn the_largest_image_is_the_pixel_budget_in_tokens() {
    let mut hp = gemma_limits();
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
    // The optional tensors too (#1220): the standardization's and the clips' scalars, where
    // either side has them (both must).
    let n_layer = ours.config().n_layer;
    let mut optional = vec!["v.std_bias".to_string(), "v.std_scale".to_string()];
    for il in 0..n_layer {
        for m in CLIPPED {
            optional.extend(CLIP_SCALARS.map(|s| format!("v.blk.{il}.{m}{s}")));
        }
    }
    optional.retain(|n| ours.weights.get(n).is_some() || theirs.weights.get(n).is_some());
    for name in names(n_layer).into_iter().chain(optional) {
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

/// The clipped tower (#1220, the E2B/E4B-sized towers' kind, no standardization) computes what
/// the loops compute with HF's `Gemma4ClippableLinear` (clamp the input, multiply, clamp the
/// product) on a grid that is not square; the clamps bite: the same weights without the
/// scalars give other rows.
#[test]
fn clipped_tower_matches_the_reference_loops() {
    let dir = tempfile::tempdir().unwrap();
    let mut enc = encoder_as(dir.path(), tiny::GEMMA_E4B);
    let p = crate::vision::preprocess(enc.hparams(), &gradient(150, 90)).unwrap();
    assert_ne!(p.grid_w, p.grid_h);
    let got = enc.encode(&p).unwrap();
    let want = reference(&enc, &p);
    assert_eq!(got.data.len(), want.len());
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    for (i, (a, b)) in got.data.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 2e-3 * scale, "{i}: {a} vs {b}");
    }
    let plain = tiny::GemmaKind {
        standardize: false,
        clipped: false,
    };
    let mut unclipped = encoder_as(dir.path(), plain);
    let other = unclipped.encode(&p).unwrap();
    let diff = other
        .data
        .iter()
        .zip(&got.data)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff > 0.05 * scale,
        "the clamps changed little: {diff} of {scale}"
    );
}

/// A clip's scalars as mtmd reads them: all four, or the missing ones at `±f32::MAX` (block 1's
/// `attn_out` has only `input_max` and `output_min`); a matrix without any is a plain product;
/// the 31B's tower has none and standardizes, the clipped one does not.
#[test]
fn clips_read_with_mtmd_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let e4b = encoder_as(dir.path(), tiny::GEMMA_E4B);
    assert!(!standardizes(&e4b.weights));
    assert_eq!(
        clip(&e4b.weights, 0, "attn_q").unwrap(),
        Some(Clip {
            inp: (-1.0, 0.875),
            out: (-0.5, 0.5),
        })
    );
    assert_eq!(
        clip(&e4b.weights, 1, "attn_out").unwrap(),
        Some(Clip {
            inp: (-f32::MAX, 0.625),
            out: (-0.25, f32::MAX),
        })
    );
    for il in 0..2 {
        for m in CLIPPED {
            let want = tiny::gemma_clip(il as u32, m);
            let c = clip(&e4b.weights, il, m).unwrap().unwrap();
            let got = [c.inp.0, c.inp.1, c.out.0, c.out.1];
            let defaults = [-f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
            for ((g, w), d) in got.iter().zip(want).zip(defaults) {
                assert_eq!(*g, w.unwrap_or(d), "{il} {m}");
            }
        }
    }
    let big = encoder(dir.path());
    assert!(standardizes(&big.weights));
    for m in CLIPPED {
        assert_eq!(clip(&big.weights, 0, m).unwrap(), None, "{m}");
    }
}

/// Towers mtmd's in-place input clamp would make order-dependent are refused (`k` clamping the
/// input `q` and `v` share differently), as are clips the graph would not apply (on the
/// projection) and half a standardization; an audio tower's clips are not the vision tower's.
#[test]
fn refuses_towers_it_cannot_clamp_like_mtmd() {
    let dir = tempfile::tempdir().unwrap();
    let load = |name: &str,
                std: bool,
                clip: &dyn Fn(u32, &str) -> [Option<f32>; 4],
                extra: &[(&str, f32)]| {
        let path = dir.path().join(name);
        tiny::write_gemma4v_with(&path, 24, 3, std, clip, extra).unwrap();
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(1);
        Encoder::load(&path, cpu)
    };
    let k_differs = |il: u32, m: &str| {
        let mut c = tiny::gemma_clip(il, m);
        if il == 1 && m == "attn_k" {
            c[0] = Some(-2.0);
        }
        c
    };
    let gate_differs = |il: u32, m: &str| {
        let mut c = tiny::gemma_clip(il, m);
        if m == "ffn_up" {
            c[1] = None;
        }
        c
    };
    assert!(matches!(
        load("k.gguf", false, &k_differs, &[]),
        Err(VisionError::Unsupported(_))
    ));
    assert!(matches!(
        load("up.gguf", false, &gate_differs, &[]),
        Err(VisionError::Unsupported(_))
    ));
    assert!(matches!(
        load(
            "proj.gguf",
            false,
            &tiny::gemma_clip,
            &[("mm.input_projection.input_min", -1.0)]
        ),
        Err(VisionError::Unsupported(_))
    ));
    assert!(matches!(
        load(
            "half.gguf",
            false,
            &tiny::gemma_clip,
            &[("v.std_bias", 0.5)]
        ),
        Err(VisionError::Metadata(_))
    ));
    // An unsloth mmproj's audio tower carries clips of its own: not read here.
    load(
        "audio.gguf",
        false,
        &tiny::gemma_clip,
        &[
            ("a.blk.0.attn_q.input_min", -3.0),
            ("mm.a.input_projection.output_max", 3.0),
        ],
    )
    .unwrap();
}

/// Nemotron 3.5 Bolt's tower's preprocessing (its processor's mean 0 and std 1, 16-pixel
/// patches merged 3 × 3, mtmd's 70 to 1,120 tokens) on a fixed image: a 640 × 480 image is
/// 624 × 480 pixels, 39 × 30 patches, 13 × 10 = 130 tokens; each pixel is `byte / 255`,
/// planar (red plane, green, blue); a flat colour stays flat through the resize.
#[test]
fn preprocesses_a_fixed_image_for_the_e4b_tower() {
    let mut hp = gemma_limits();
    hp.image_mean = [0.0; 3];
    hp.image_std = [1.0; 3];
    let flat = tiny::bmp(640, 480, |_, _| [51, 102, 204]);
    let p = crate::vision::preprocess(&hp, &flat).unwrap();
    assert_eq!((p.width, p.height), (624, 480));
    assert_eq!((p.grid_w, p.grid_h), (39, 30));
    let k = hp.spatial_merge_size;
    assert_eq!((p.grid_w / k) * (p.grid_h / k), 130);
    let plane = (p.width * p.height) as usize;
    assert_eq!(p.pixels.len(), 3 * plane);
    for (c, byte) in [51u8, 102, 204].into_iter().enumerate() {
        let want = byte as f32 / 255.0;
        assert!(
            p.pixels[c * plane..][..plane]
                .iter()
                .all(|&v| (v - want).abs() < 1e-6),
            "channel {c}"
        );
    }
    // A small image grows to the fewest tokens (70 → 9 × 9 = 81 at a square), a big one
    // shrinks within 1,120.
    let small = crate::vision::preprocess(&hp, &gradient(16, 16)).unwrap();
    assert_eq!((small.grid_w / k, small.grid_h / k), (9, 9));
    let big = crate::vision::preprocess(&hp, &gradient(4000, 3000)).unwrap();
    assert_eq!((big.width, big.height), (1824, 1344));
    assert!((big.grid_w / k) * (big.grid_h / k) <= 1120);
}

/// A tiny Gemma 4 checkpoint (BF16, as Nemotron 3.5 Bolt E4B's snapshot ships): 2 blocks of
/// width 16, 2 heads, 32-wide MLPs, 4-pixel patches, a 64-row position table, a 24-wide text
/// model; every value exact in BF16 and F16. With `clipped`, `use_clipped_linears` and each
/// clippable linear's four 0-D bounds, but block 1's `down_proj` without `output_max`.
fn write_gemma_checkpoint(dir: &std::path::Path, clipped: bool) {
    let (e, heads, ff, p, rows, text) = (16usize, 2usize, 32usize, 4usize, 64usize, 24usize);
    let config = serde_json::json!({
        "model_type": "gemma4",
        "vision_config": {
            "hidden_size": e, "num_hidden_layers": 2, "num_attention_heads": heads,
            "intermediate_size": ff, "patch_size": p, "pooling_kernel_size": 3,
            "rms_norm_eps": 1e-6, "use_clipped_linears": clipped,
        },
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let pre = serde_json::json!({"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]});
    std::fs::write(dir.join("preprocessor_config.json"), pre.to_string()).unwrap();
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut k = 0usize;
    let mut add = |name: String, shape: Vec<usize>| {
        let n = shape.iter().product::<usize>();
        let v = (0..n)
            .map(|i| ((i + k) % 7) as f32 * 0.125 - 0.375)
            .collect();
        k += 1;
        tensors.push((name, shape, v));
    };
    let t = |s: &str| format!("model.vision_tower.{s}");
    add(t("patch_embedder.input_proj.weight"), vec![e, p * p * 3]);
    add(
        t("patch_embedder.position_embedding_table"),
        vec![2, rows, e],
    );
    add(
        "model.embed_vision.embedding_projection.weight".into(),
        vec![text, e],
    );
    for il in 0..2 {
        let l = |s: &str| t(&format!("encoder.layers.{il}.{s}"));
        for n in [
            "input_layernorm",
            "pre_feedforward_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
        ] {
            add(l(&format!("{n}.weight")), vec![e]);
        }
        for n in ["q_norm", "k_norm"] {
            add(l(&format!("self_attn.{n}.weight")), vec![e / heads]);
        }
        for (n, shape) in [
            ("self_attn.q_proj", vec![e, e]),
            ("self_attn.k_proj", vec![e, e]),
            ("self_attn.v_proj", vec![e, e]),
            ("self_attn.o_proj", vec![e, e]),
            ("mlp.gate_proj", vec![ff, e]),
            ("mlp.up_proj", vec![ff, e]),
            ("mlp.down_proj", vec![e, ff]),
        ] {
            add(l(&format!("{n}.linear.weight")), shape);
        }
    }
    if clipped {
        for il in 0..2 {
            for (proj, m) in [
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_out"),
                ("mlp.gate_proj", "ffn_gate"),
                ("mlp.up_proj", "ffn_up"),
                ("mlp.down_proj", "ffn_down"),
            ] {
                let [a, b, c, d] = tiny::gemma_clip(il as u32, m);
                let bounds = [
                    ("input_min", a.unwrap_or(-8.0)),
                    ("input_max", b.unwrap_or(8.0)),
                    ("output_min", c.unwrap_or(-8.0)),
                    ("output_max", d.unwrap_or(8.0)),
                ];
                for (s, v) in bounds {
                    if il == 1 && m == "ffn_down" && s == "output_max" {
                        continue;
                    }
                    let name = t(&format!("encoder.layers.{il}.{proj}.{s}"));
                    tensors.push((name, vec![], vec![v]));
                }
            }
        }
    }
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, v) in &tensors {
        let bytes: Vec<u8> = v
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16", "shape": shape,
                "data_offsets": [data.len(), data.len() + bytes.len()],
            }),
        );
        data.extend(bytes);
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

/// A Gemma 4 checkpoint's tower under the mmproj's names (#1217, #1220): the converter's
/// renames (`.linear` dropped), BF16 matrices as F16, norms and the position table F32, the
/// patch linear as the conv kernel; a clipped tower's 0-D BF16 bounds as F32 `[1]` scalars
/// `v.blk.{i}.{m}.{input,output}_{min,max}`, a missing one at mtmd's `±f32::MAX`; no
/// standardization when the checkpoint has none. The bounds of a checkpoint that does not
/// use them stay out (HF's `Gemma4ClippableLinear` skips them too). Both towers run.
#[test]
fn checkpoint_names_map_to_the_mmproj() {
    use crate::ggml::types;
    for clipped in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        write_gemma_checkpoint(dir.path(), clipped);
        if !clipped {
            // The bounds on a checkpoint that does not use them.
            let keep = tempfile::tempdir().unwrap();
            write_gemma_checkpoint(keep.path(), true);
            std::fs::copy(
                keep.path().join("model.safetensors"),
                dir.path().join("model.safetensors"),
            )
            .unwrap();
        }
        let g = crate::hf::open_vision(dir.path())
            .unwrap()
            .expect("a vision tower");
        let ty = |n: &str| g.tensor(n).unwrap_or_else(|| panic!("no `{n}`"));
        assert_eq!(ty("v.patch_embd.weight").shape, [4, 4, 3, 16]);
        assert_eq!(ty("v.patch_embd.weight").ggml_type, types::F32);
        assert_eq!(ty("v.position_embd.weight").shape, [16, 64, 2]);
        assert_eq!(ty("v.position_embd.weight").ggml_type, types::F32);
        assert_eq!(ty("mm.input_projection.weight").shape, [16, 24]);
        assert_eq!(ty("mm.input_projection.weight").ggml_type, types::F16);
        assert_eq!(ty("v.blk.1.ffn_down.weight").shape, [32, 16]);
        assert_eq!(ty("v.blk.1.ffn_down.weight").ggml_type, types::F16);
        assert_eq!(ty("v.blk.0.attn_q_norm.weight").shape, [8]);
        assert_eq!(ty("v.blk.0.ln1.weight").ggml_type, types::F32);
        assert!(g.tensor("v.std_bias").is_none() && g.tensor("v.std_scale").is_none());
        let scalars = g
            .tensors()
            .iter()
            .filter(|t| CLIP_SCALARS.iter().any(|s| t.name.ends_with(s)))
            .count();
        assert_eq!(scalars, if clipped { 2 * 7 * 4 - 1 } else { 0 });
        if clipped {
            assert_eq!(ty("v.blk.0.attn_k.input_max").shape, [1]);
            assert_eq!(ty("v.blk.0.attn_k.input_max").ggml_type, types::F32);
        }
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        let mut enc = Encoder::from_gguf(&g, cpu).unwrap();
        assert!(!standardizes(&enc.weights));
        // The patch linear `[n, p · p · 3]` (channels last) as `[n, 3, p, p]`.
        let shipped: Vec<f32> = (0..16 * 48)
            .map(|i| (i % 7) as f32 * 0.125 - 0.375)
            .collect();
        let kernel = widened(&enc, "v.patch_embd.weight");
        for (o, row) in shipped.chunks(48).enumerate() {
            for c in 0..3 {
                for y in 0..4 {
                    for x in 0..4 {
                        let at = ((o * 3 + c) * 4 + y) * 4 + x;
                        assert_eq!(kernel[at], row[(y * 4 + x) * 3 + c], "{o} {c} {y} {x}");
                    }
                }
            }
        }
        for il in 0..2 {
            for m in CLIPPED {
                let got = clip(&enc.weights, il, m).unwrap();
                if !clipped {
                    assert_eq!(got, None, "{il} {m}");
                    continue;
                }
                let [a, b, c, d] = tiny::gemma_clip(il as u32, m);
                let last = if il == 1 && m == "ffn_down" {
                    f32::MAX
                } else {
                    d.unwrap_or(8.0)
                };
                let want = Clip {
                    inp: (a.unwrap_or(-8.0), b.unwrap_or(8.0)),
                    out: (c.unwrap_or(-8.0), last),
                };
                assert_eq!(got, Some(want), "{il} {m}");
            }
        }
        let p = crate::vision::preprocess(enc.hparams(), &gradient(60, 36)).unwrap();
        let rows = enc.encode(&p).unwrap();
        assert_eq!(rows.n_embd, 24);
        assert!(rows.data.iter().all(|v| v.is_finite()));
    }
}
