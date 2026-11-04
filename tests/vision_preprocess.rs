mod common;

use legion::gguf::{GgufFile, GgufWriter, Value};
use legion::vision::{VisionHparams, preprocess};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    width: u32,
    height: u32,
    grid_w: u32,
    grid_h: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    image_min_pixels: u32,
    image_max_pixels: u32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    stride: usize,
    sum: f64,
    samples: Vec<f32>,
}

fn metadata() -> GgufWriter {
    let mut writer = GgufWriter::new();
    for (key, value) in [
        ("general.architecture", Value::String("clip".into())),
        (
            "clip.projector_type",
            Value::String("qwen3vl_merger".into()),
        ),
        ("clip.vision.patch_size", Value::U32(16)),
        ("clip.vision.image_size", Value::U32(768)),
        (
            "clip.vision.image_mean",
            Value::Array(vec![Value::F32(0.5); 3]),
        ),
        (
            "clip.vision.image_std",
            Value::Array(vec![Value::F32(0.5); 3]),
        ),
    ] {
        writer.set(key, &value).unwrap();
    }
    writer
}

fn parse(writer: &GgufWriter) -> Result<VisionHparams, legion::vision::VisionError> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mmproj.gguf");
    writer.write(&path).unwrap();
    VisionHparams::from_gguf(&GgufFile::open(path).unwrap())
}

#[test]
fn test_image_matches_llama_cpp() {
    let reference: Reference =
        serde_json::from_str(include_str!("data/vision/test-1.json")).unwrap();
    let hp = parse(&metadata()).unwrap();
    assert_eq!(hp.patch_size, reference.patch_size);
    assert_eq!(hp.spatial_merge_size, reference.spatial_merge_size);
    assert_eq!(hp.image_min_pixels, reference.image_min_pixels);
    assert_eq!(hp.image_max_pixels, reference.image_max_pixels);
    assert_eq!(hp.image_mean, reference.image_mean);
    assert_eq!(hp.image_std, reference.image_std);
    let image = preprocess(&hp, include_bytes!("data/vision/test-1.jpeg")).unwrap();
    assert_eq!(
        (image.width, image.height),
        (reference.width, reference.height)
    );
    assert_eq!(
        (image.grid_w, image.grid_h),
        (reference.grid_w, reference.grid_h)
    );
    assert_eq!(
        image.pixels.len(),
        3 * image.width as usize * image.height as usize
    );
    assert_eq!(
        image.pixels.len().div_ceil(reference.stride),
        reference.samples.len()
    );
    let mut mismatches = Vec::new();
    let mut max_difference = 0.0f32;
    for (index, (actual, expected)) in image
        .pixels
        .iter()
        .step_by(reference.stride)
        .zip(&reference.samples)
        .enumerate()
    {
        let difference = (actual - expected).abs();
        max_difference = max_difference.max(difference);
        if difference > 1e-6 {
            mismatches.push((index * reference.stride, *actual, *expected));
        }
    }
    let sum: f64 = image.pixels.iter().map(|&v| v as f64).sum();
    assert!(
        mismatches.is_empty(),
        "{} / {} samples differ; max difference {max_difference}; first {:?}; sum {sum} vs {}",
        mismatches.len(),
        reference.samples.len(),
        mismatches.first(),
        reference.sum
    );
    assert!((sum - reference.sum).abs() <= reference.sum.abs() * 1e-3);
}

#[test]
fn metadata_overrides_and_validation() {
    let mut writer = metadata();
    writer
        .set("clip.vision.spatial_merge_size", &Value::U32(3))
        .unwrap();
    writer
        .set("clip.vision.image_min_pixels", &Value::U32(9216))
        .unwrap();
    writer
        .set("clip.vision.image_max_pixels", &Value::U32(18432))
        .unwrap();
    let hp = parse(&writer).unwrap();
    assert_eq!(hp.spatial_merge_size, 3);
    assert_eq!((hp.image_min_pixels, hp.image_max_pixels), (9216, 18432));
    for (key, value) in [
        ("general.architecture", Value::String("qwen35".into())),
        (
            "clip.projector_type",
            Value::String("qwen2vl_merger".into()),
        ),
        ("clip.vision.patch_size", Value::U32(0)),
        ("clip.vision.patch_size", Value::U32(u32::MAX)),
        ("clip.vision.spatial_merge_size", Value::U32(0)),
        ("clip.vision.image_min_pixels", Value::U32(0)),
        ("clip.vision.image_max_pixels", Value::U32(1)),
        (
            "clip.vision.image_max_pixels",
            Value::String("invalid".into()),
        ),
        (
            "clip.vision.image_std",
            Value::Array(vec![Value::F32(0.0); 3]),
        ),
        (
            "clip.vision.image_std",
            Value::Array(vec![Value::F32(f32::NAN); 3]),
        ),
        (
            "clip.vision.image_mean",
            Value::Array(vec![Value::F32(0.5); 2]),
        ),
    ] {
        let mut writer = metadata();
        writer.set(key, &value).unwrap();
        assert!(parse(&writer).is_err(), "{key}: {value}");
    }
    assert!(parse(&GgufWriter::new()).is_err());
}

#[test]
fn bad_images_return_errors() {
    let hp = parse(&metadata()).unwrap();
    for bytes in [
        &[][..],
        b"not an image",
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xff",
        b"GIF89a",
        b"BM",
        b"RIFF\x00\x00\x00\x00WEBP",
    ] {
        assert!(preprocess(&hp, bytes).is_err());
    }
    let jpeg = include_bytes!("data/vision/test-1.jpeg");
    assert!(preprocess(&hp, &jpeg[..100]).is_err());
}

#[test]
fn stb_formats_discard_alpha_and_use_first_gif_frame() {
    let hp = parse(&metadata()).unwrap();
    for bytes in [
        include_bytes!("data/vision/transparent.png").as_slice(),
        include_bytes!("data/vision/color.bmp").as_slice(),
        include_bytes!("data/vision/animated.gif").as_slice(),
    ] {
        let image = preprocess(&hp, bytes).unwrap();
        assert_eq!((image.width, image.height), (128, 64));
        for (plane, byte) in image
            .pixels
            .as_chunks::<{ 128 * 64 }>()
            .0
            .iter()
            .zip([12, 34, 56])
        {
            let expected = (byte as f32 / 255.0 - 0.5) / 0.5;
            assert!(plane.iter().all(|&v| v == expected));
        }
    }
}

#[test]
fn resize_centers_black_padding_before_normalization() {
    let mut hp = parse(&metadata()).unwrap();
    hp.image_min_pixels = 1;
    let image = preprocess(&hp, include_bytes!("data/vision/color.bmp")).unwrap();
    assert_eq!((image.width, image.height), (32, 32));
    for (plane, byte) in image
        .pixels
        .as_chunks::<{ 32 * 32 }>()
        .0
        .iter()
        .zip([12, 34, 56])
    {
        for (y, row) in plane.as_chunks::<32>().0.iter().enumerate() {
            let expected = if (8..24).contains(&y) {
                (byte as f32 / 255.0 - 0.5) / 0.5
            } else {
                -1.0
            };
            assert!(row.iter().all(|&v| v == expected), "row {y}");
        }
    }
}

#[test]
fn webp_uses_image_decoder() {
    use image::ImageEncoder;

    let mut bytes = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(&mut bytes)
        .write_image(&[12, 34, 56], 1, 1, image::ExtendedColorType::Rgb8)
        .unwrap();
    let hp = parse(&metadata()).unwrap();
    let image = preprocess(&hp, &bytes).unwrap();
    assert_eq!((image.width, image.height), (96, 96));
    for (plane, byte) in image
        .pixels
        .as_chunks::<{ 96 * 96 }>()
        .0
        .iter()
        .zip([12, 34, 56])
    {
        let expected = (byte as f32 / 255.0 - 0.5) / 0.5;
        assert!(plane.iter().all(|&v| v == expected));
    }
}
