use super::*;
use crate::ggml::Backend;
use crate::vision::tiny;

/// The tiny tower on the GPU against the CPU tower (the oracle) on image grids that resample
/// the position table and one that does not, with more than one chunk of queries' worth of
/// patches: the rows agree to F16 rounding, and the GPU gives the same bits every time.
#[test]
#[ignore = "needs a GPU"]
fn gpu_tower_matches_the_cpu_tower() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    tiny::write(&path, 64, 7, false).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    let mut enc = Encoder::load(&path, cpu).unwrap();
    let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
        return;
    };
    for (w, h) in [(16, 16), (32, 32), (40, 24), (200, 120)] {
        let img = tiny::bmp(w, h, |x, y| {
            [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
        });
        let p = super::super::preprocess(enc.hparams(), &img).unwrap();
        let want = enc.encode(&p).unwrap();
        let got = gpu.encode(&p).unwrap();
        assert_eq!(
            (got.nx, got.ny, got.n_embd),
            (want.nx, want.ny, want.n_embd)
        );
        let scale = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .data
            .iter()
            .zip(&want.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 0.02 * scale,
            "{w} x {h}: max abs error {err} at a scale of {scale}"
        );
        assert_eq!(
            gpu.encode(&p).unwrap(),
            got,
            "{w} x {h}: the same bits again"
        );
        // The grid's cap moves no bits: one CTA (it loops over every tile), half an SM's worth,
        // uncapped, the default.
        for share in ["1/1000", "1/2", "0", "1"] {
            gpu.set_share(crate::vision::share::parse(share).unwrap()[0]);
            assert_eq!(gpu.encode(&p).unwrap(), got, "{w} x {h}: share {share}");
        }
    }
}

/// The tiny Gemma 4 tower (#1217) on the GPU against the CPU tower on square, wide and tall-ish
/// grids up to the pixel budget: the rows agree to F16 rounding, the same bits every time.
#[test]
#[ignore = "needs a GPU"]
fn gpu_gemma_tower_matches_the_cpu_tower() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    tiny::write_gemma4v(&path, 64, 5).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    let mut enc = Encoder::load(&path, cpu).unwrap();
    let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
        return;
    };
    assert!(matches!(gpu, GpuEncoder::Gemma(_)));
    for (w, h) in [(16, 16), (150, 90), (240, 60), (400, 400)] {
        let img = tiny::bmp(w, h, |x, y| {
            [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
        });
        let p = super::super::preprocess(enc.hparams(), &img).unwrap();
        let want = enc.encode(&p).unwrap();
        let got = gpu.encode(&p).unwrap();
        assert_eq!(
            (got.nx, got.ny, got.n_embd),
            (want.nx, want.ny, want.n_embd)
        );
        let scale = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .data
            .iter()
            .zip(&want.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 0.02 * scale,
            "{w} x {h}: max abs error {err} at a scale of {scale}"
        );
        assert_eq!(
            gpu.encode(&p).unwrap(),
            got,
            "{w} x {h}: the same bits again"
        );
    }
}

/// The tiny Gemma 4 tower's rows do not depend on its share of the GPU: uncapped grids (one
/// block per work item) against one GEMM CTA and two blocks per small kernel (each loops over
/// every item), half an SM's worth and the default.
#[test]
#[ignore = "needs a GPU"]
fn gpu_gemma_tower_rows_do_not_depend_on_the_share() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    tiny::write_gemma4v(&path, 64, 5).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    let enc = Encoder::load(&path, cpu).unwrap();
    let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
        return;
    };
    assert!(matches!(gpu, GpuEncoder::Gemma(_)));
    for (w, h) in [(16, 16), (240, 60), (400, 400)] {
        let img = tiny::bmp(w, h, |x, y| {
            [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
        });
        let p = super::super::preprocess(enc.hparams(), &img).unwrap();
        gpu.set_share(crate::vision::share::Share::FULL);
        let want = gpu.encode(&p).unwrap();
        for share in ["1/1000", "1/2", "1"] {
            gpu.set_share(crate::vision::share::parse(share).unwrap()[0]);
            assert_eq!(gpu.encode(&p).unwrap(), want, "{w} x {h}: share {share}");
        }
    }
}

/// The tiny GLM-5.3-Flash tower (#1219; 6-pixel patches, 108 values a patch padded to 112 as the
/// real 588 are to 592) on the GPU against the CPU tower, with and without projector MLP biases,
/// on grids from the floor of the pixel budget to its ceiling: the rows agree within 2% of the
/// largest CPU value (F16 rounding of the matrices and activations), the same bits every time
/// and at every share of the GPU.
#[test]
#[ignore = "needs a GPU"]
fn gpu_glm_tower_matches_the_cpu_tower() {
    for mm_bias in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmproj.gguf");
        tiny::write_glm5v(&path, 64, 11, mm_bias).unwrap();
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        let mut enc = Encoder::load(&path, cpu).unwrap();
        let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
            return;
        };
        assert!(matches!(gpu, GpuEncoder::Glm(_)));
        for (w, h) in [(24, 24), (78, 45), (150, 90), (96, 96)] {
            let img = tiny::bmp(w, h, |x, y| {
                [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
            });
            let p = super::super::preprocess(enc.hparams(), &img).unwrap();
            let want = enc.encode(&p).unwrap();
            gpu.set_share(crate::vision::share::Share::FULL);
            let got = gpu.encode(&p).unwrap();
            assert_eq!(
                (got.nx, got.ny, got.n_embd),
                (want.nx, want.ny, want.n_embd)
            );
            let scale = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = got
                .data
                .iter()
                .zip(&want.data)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                err <= 0.02 * scale,
                "{w} x {h} (biases {mm_bias}): max abs error {err} at a scale of {scale}"
            );
            assert_eq!(
                gpu.encode(&p).unwrap(),
                got,
                "{w} x {h}: the same bits again"
            );
            for share in ["1/1000", "1/2", "1"] {
                gpu.set_share(crate::vision::share::parse(share).unwrap()[0]);
                assert_eq!(gpu.encode(&p).unwrap(), got, "{w} x {h}: share {share}");
            }
        }
    }
}

/// The tiny GLM-5.3-Flash tower refuses a grid over its pixel budget (9,216 pixels, 16 × 16
/// patches of 6) with an error, as its scratch is reserved for the budget, and takes one at it.
#[test]
#[ignore = "needs a GPU"]
fn gpu_glm_tower_refuses_a_grid_over_the_pixel_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    tiny::write_glm5v(&path, 64, 11, false).unwrap();
    let enc = Encoder::load(&path, Backend::cpu().unwrap()).unwrap();
    assert_eq!(enc.hparams().image_max_pixels, 96 * 96);
    let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
        return;
    };
    let grid = |gx: u32, gy: u32| {
        let (w, h) = (gx * tiny::GLM_PATCH, gy * tiny::GLM_PATCH);
        crate::vision::Preprocessed {
            width: w,
            height: h,
            grid_w: gx,
            grid_h: gy,
            pixels: vec![0.0; 3 * (w * h) as usize],
        }
    };
    assert!(matches!(
        gpu.encode(&grid(18, 16)),
        Err(VisionError::Dimensions)
    ));
    assert!(matches!(
        gpu.encode(&grid(16, 18)),
        Err(VisionError::Dimensions)
    ));
    assert!(gpu.encode(&grid(16, 16)).is_ok());
}

/// A checkpoint's tower (`LEGION_TEST_VISION_HF=<dir>`) on the images of
/// `LEGION_TEST_VISION_IMAGES=<dir>`: the GPU rows against the CPU tower's, with each image's
/// grid, both encode times, the largest absolute error, the largest error relative to the
/// largest CPU value, the smallest and the mean cosine of a GPU row with its CPU row; every
/// image, then the ones that miss: a row under 0.99, or for GLM-5.3-Flash's tower what
/// [`crate::vision::glm5v::rows_agree`] asks (its outlier rows amplify rounding, see there).
#[test]
#[ignore = "needs a GPU"]
fn gpu_tower_on_a_checkpoint() {
    let (Some(hf), Some(images)) = (
        std::env::var_os("LEGION_TEST_VISION_HF"),
        std::env::var_os("LEGION_TEST_VISION_IMAGES"),
    ) else {
        return;
    };
    let g = crate::hf::open_vision(std::path::Path::new(&hf))
        .unwrap()
        .expect("a vision tower");
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut enc = Encoder::from_gguf(&g, cpu).unwrap();
    let mut gpu = GpuEncoder::new(&enc, 0).unwrap().expect("a GPU tower");
    let mut files: Vec<_> = std::fs::read_dir(images)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let mut failed = Vec::new();
    for f in files {
        let p = super::super::preprocess(enc.hparams(), &std::fs::read(&f).unwrap()).unwrap();
        let t = std::time::Instant::now();
        let want = enc.encode(&p).unwrap();
        let t_cpu = t.elapsed().as_secs_f64();
        gpu.encode(&p).unwrap();
        let t = std::time::Instant::now();
        let got = gpu.encode(&p).unwrap();
        let t_gpu = t.elapsed().as_secs_f64();
        assert_eq!(got.data.len(), want.data.len());
        assert!(
            got.data.iter().chain(&want.data).all(|v| v.is_finite()),
            "{f:?}: rows not finite"
        );
        let scale = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .data
            .iter()
            .zip(&want.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let rows = got.data.len() / got.n_embd;
        let d = crate::vision::glm5v::Diff::new(&got.data, &want.data, rows);
        let cos = d.min_cos;
        println!(
            "{}: {} x {} px, {} x {} rows, CPU {t_cpu:.3} s, GPU {t_gpu:.4} s, max abs {err:.4e}, \
             max rel {:.4e} (of {scale:.3}), min row cosine {cos:.6}, mean row cosine {:.6}, \
             bit-identical {}, GPU rows {:016x}",
            f.file_name().unwrap().to_string_lossy(),
            p.width,
            p.height,
            got.nx,
            got.ny,
            err / scale,
            d.mean_cos,
            got == want,
            got.data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
            })
        );
        let agree = if enc.hparams().projector == super::super::Projector::Glm5v {
            crate::vision::glm5v::rows_agree(&d)
        } else {
            cos > 0.99
        };
        if !agree {
            failed.push(format!("{f:?}: {d}"));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// GLM-5.3-Flash's tower from a checkpoint (`LEGION_TEST_VISION_HF=<dir>`) on the GPU against
/// the CPU tower stage by stage (the patches, each block's residual after its attention and
/// after its MLP, the final norm, the 2×2 convolution, the projector's linear, its LayerNorm
/// and GELU, the rows out), printed, with the first stage under a row cosine of 0.999 marked:
/// where the two part. The images: the small ones of the CPU tower's reference test, then
/// those of `LEGION_TEST_VISION_IMAGES=<dir>` if set. On the small ones both towers' rows are
/// also held against the plain loops (`reference_stages`), so a gap shows which side is off.
/// It fails when the rows out do not agree as [`crate::vision::glm5v::rows_agree`] asks; a
/// stage under 0.999 alone does not (a few outlier rows amplify rounding, see there).
#[test]
#[ignore = "needs a GPU and a checkpoint"]
fn gpu_glm_stages_on_a_checkpoint() {
    use crate::vision::glm5v::{Diff, merge_to_rows, per_patch, rows_agree, tests as cpu_tests};
    let Some(hf) = std::env::var_os("LEGION_TEST_VISION_HF") else {
        return;
    };
    let g = crate::hf::open_vision(std::path::Path::new(&hf))
        .unwrap()
        .expect("a vision tower");
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut enc = Encoder::from_gguf(&g, cpu).unwrap();
    let mut gpu = GpuEncoder::new(&enc, 0).unwrap().expect("a GPU tower");
    let mut images: Vec<(String, Vec<u8>, bool)> = cpu_tests::checkpoint_images()
        .into_iter()
        .map(|(l, b)| (l.to_string(), b, true))
        .collect();
    if let Some(dir) = std::env::var_os("LEGION_TEST_VISION_IMAGES") {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        for f in files {
            let label = f.file_name().unwrap().to_string_lossy().into_owned();
            images.push((label, std::fs::read(&f).unwrap(), false));
        }
    }
    let mut failed = Vec::new();
    for (label, img, small) in images {
        let p = super::super::preprocess(enc.hparams(), &img).unwrap();
        let (gx, gy) = (p.grid_w as usize, p.grid_h as usize);
        let n = gx * gy;
        let (_, want) = enc.encode_stages(&p).unwrap();
        let (_, got) = gpu.encode_stages(&p).unwrap();
        println!(
            "{label}: {} x {} px, {gx} x {gy} patches",
            p.width, p.height
        );
        assert_eq!(got.len(), want.len());
        let mut first = None;
        for ((name, a), (wname, b)) in got.iter().zip(&want) {
            assert_eq!(name, wname);
            let d = Diff::new(a, b, if per_patch(name) { n } else { n / 4 });
            let close = d.bad == 0 && d.min_cos > 0.999;
            let mark = if first.is_none() && !close {
                first = Some(name.clone());
                "  <- first under 0.999"
            } else {
                ""
            };
            println!("  {name:>12}: {d}{mark}");
            if name == "out" && !rows_agree(&d) {
                failed.push(format!("{label}: {d} (first under 0.999: {first:?})"));
            }
        }
        if small {
            // Both towers against the loops, after each block's MLP and the projector's stages.
            let loops = cpu_tests::reference_stages(&enc, &p);
            println!("  against the loops (GPU; CPU):");
            for (((name, a), (_, b)), (_, l)) in got.iter().zip(&want).zip(&loops) {
                if name.ends_with(".attn") {
                    continue;
                }
                let (rows, a, b) = if per_patch(name) {
                    let w = a.len() / n;
                    (n, merge_to_rows(a, gx, gy, w), merge_to_rows(b, gx, gy, w))
                } else {
                    (n / 4, a.clone(), b.clone())
                };
                println!(
                    "  {name:>12}: {}; {}",
                    Diff::new(&a, l, rows),
                    Diff::new(&b, l, rows)
                );
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the GPU tower parts from the CPU's: {failed:?}"
    );
}

/// The tiny clipped Gemma 4 tower (#1220: Nemotron 3.5 Bolt's kind, clipped linears, no
/// standardization) on the GPU against the CPU tower: the rows agree to F16 rounding (largest
/// error within 2% of the largest CPU value, as the unclipped tower's), the same bits every
/// time and whatever the share, and the GPU's clamps bite (the same weights without the
/// scalars give other rows).
#[test]
#[ignore = "needs a GPU"]
fn gpu_clipped_gemma_tower_matches_the_cpu_tower() {
    let dir = tempfile::tempdir().unwrap();
    let load = |name: &str, kind| {
        let path = dir.path().join(name);
        tiny::write_gemma4v_as(&path, 64, 5, kind).unwrap();
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(2);
        Encoder::load(&path, cpu).unwrap()
    };
    let mut enc = load("clipped.gguf", tiny::GEMMA_E4B);
    let plain = load(
        "plain.gguf",
        tiny::GemmaKind {
            standardize: false,
            clipped: false,
        },
    );
    let Some(mut gpu) = GpuEncoder::new(&enc, 0).unwrap() else {
        return;
    };
    let mut gpu_plain = GpuEncoder::new(&plain, 0).unwrap().expect("a GPU tower");
    for (w, h) in [(16, 16), (150, 90), (240, 60), (400, 400)] {
        let img = tiny::bmp(w, h, |x, y| {
            [(x * 13) as u8, (y * 29) as u8, ((x + y) * 7) as u8]
        });
        let p = super::super::preprocess(enc.hparams(), &img).unwrap();
        let want = enc.encode(&p).unwrap();
        let got = gpu.encode(&p).unwrap();
        assert_eq!(
            (got.nx, got.ny, got.n_embd),
            (want.nx, want.ny, want.n_embd)
        );
        let scale = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .data
            .iter()
            .zip(&want.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 0.02 * scale,
            "{w} x {h}: max abs error {err} at a scale of {scale}"
        );
        assert_eq!(gpu.encode(&p).unwrap(), got, "{w} x {h}: the same bits");
        for share in ["1/1000", "1/2", "1"] {
            gpu.set_share(crate::vision::share::parse(share).unwrap()[0]);
            assert_eq!(gpu.encode(&p).unwrap(), got, "{w} x {h}: share {share}");
        }
        gpu.set_share(crate::vision::share::Share::FULL);
        let other = gpu_plain.encode(&p).unwrap();
        let diff = other
            .data
            .iter()
            .zip(&got.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff > 0.05 * scale, "{w} x {h}: the clamps changed little");
    }
}

/// A Gemma 4 tower with clipped linears (#1220; `LEGION_TEST_VISION_HF=<checkpoint dir or
/// mmproj GGUF>`, e.g. Nemotron 3.5 Bolt E4B's BF16 snapshot) on the images of
/// `LEGION_TEST_VISION_IMAGES=<dir>`: GPU rows against the CPU tower's, per image the mean and
/// smallest row cosine and the largest absolute error. Passes when every image's mean cosine is
/// at least 0.995 and its smallest at least 0.5 (as #240's `rows_agree`: a few rows of a
/// clipped tower may sit on a clamp's edge).
#[test]
#[ignore = "needs a GPU and a Gemma 4 checkpoint"]
fn gpu_clipped_tower_rows_agree_on_a_checkpoint() {
    let (Some(hf), Some(images)) = (
        std::env::var_os("LEGION_TEST_VISION_HF"),
        std::env::var_os("LEGION_TEST_VISION_IMAGES"),
    ) else {
        return;
    };
    let path = std::path::Path::new(&hf);
    let g = if path.is_dir() {
        crate::hf::open_vision(path)
            .unwrap()
            .expect("a vision tower")
    } else {
        crate::gguf::GgufFile::open(path).unwrap()
    };
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(8);
    let mut enc = Encoder::from_gguf(&g, cpu).unwrap();
    let clipped = (0..enc.config().n_layer).any(|il| {
        crate::vision::gemma4v::CLIPPED.iter().any(|m| {
            crate::vision::gemma4v::clip(&enc.weights, il, m)
                .unwrap()
                .is_some()
        })
    });
    println!(
        "tower: {} layers, width {}, clipped linears {clipped}",
        enc.config().n_layer,
        enc.config().n_embd
    );
    let mut gpu = GpuEncoder::new(&enc, 0).unwrap().expect("a GPU tower");
    let mut files: Vec<_> = std::fs::read_dir(images)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let mut failed = Vec::new();
    for f in files {
        let p = super::super::preprocess(enc.hparams(), &std::fs::read(&f).unwrap()).unwrap();
        let want = enc.encode(&p).unwrap();
        let got = gpu.encode(&p).unwrap();
        assert_eq!(got.data.len(), want.data.len());
        let err = got
            .data
            .iter()
            .zip(&want.data)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let cos: Vec<f64> = got
            .data
            .chunks(got.n_embd)
            .zip(want.data.chunks(want.n_embd))
            .map(|(a, b)| {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum();
                let na: f64 = a.iter().map(|x| (x * x) as f64).sum();
                let nb: f64 = b.iter().map(|x| (x * x) as f64).sum();
                dot / (na * nb).sqrt()
            })
            .collect();
        let mean = cos.iter().sum::<f64>() / cos.len() as f64;
        let min = cos.iter().cloned().fold(1f64, f64::min);
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        println!(
            "{name}: {} x {} px, {} x {} rows, mean row cosine {mean:.6}, min row cosine \
             {min:.6}, max abs {err:.4e}",
            p.width, p.height, got.nx, got.ny
        );
        if !(mean >= 0.995 && min >= 0.5) {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "rows disagree on {failed:?}");
}
