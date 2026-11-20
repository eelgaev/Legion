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

/// A checkpoint's tower (`LEGION_TEST_VISION_HF=<dir>`) on the images of
/// `LEGION_TEST_VISION_IMAGES=<dir>`: the GPU rows against the CPU tower's, with each image's
/// grid, both encode times, the largest absolute error, the largest error relative to the
/// largest CPU value, and the smallest cosine of a GPU row with its CPU row.
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
        let cos = got
            .data
            .chunks(got.n_embd)
            .zip(want.data.chunks(want.n_embd))
            .map(|(a, b)| {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum();
                let na: f64 = a.iter().map(|x| (x * x) as f64).sum();
                let nb: f64 = b.iter().map(|x| (x * x) as f64).sum();
                dot / (na * nb).sqrt()
            })
            .fold(1f64, f64::min);
        println!(
            "{}: {} x {} px, {} x {} rows, CPU {t_cpu:.3} s, GPU {t_gpu:.4} s, max abs {err:.4e}, \
             max rel {:.4e} (of {scale:.3}), min row cosine {cos:.6}, bit-identical {}",
            f.file_name().unwrap().to_string_lossy(),
            p.width,
            p.height,
            got.nx,
            got.ny,
            err / scale,
            got == want
        );
        assert!(cos > 0.99, "{f:?}: min row cosine {cos}");
    }
}
